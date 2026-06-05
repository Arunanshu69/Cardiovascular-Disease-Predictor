import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
import warnings
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors

# Suppress joblib resource tracker warnings
warnings.filterwarnings('ignore', message='resource_tracker:')

# Import modules
from src.preprocessing import DataCleaner, OutlierRemover, Normalizer, FeatureEngineer
from src.diffusion import TabDDPMTrainer, TabDDPMGenerator
from src.feature_selection import CorrelationAnalyzer, FeatureImportanceAnalyzer
from src.optimization import GAPSOHybridOptimizer
from src.models import XGBoostModel
from src.evaluation import MetricsCalculator, Visualizer
from src.explainability import SHAPAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Heart Disease Prediction Pipeline')
    parser.add_argument('--data_path', type=str, default='data/pkiohd.csv', help='Path to dataset')
    parser.add_argument('--target_column', type=str, default='CARDIO_DISEASE', help='Target column name')
    parser.add_argument('--use_diffusion', default=True, action='store_true', help='Use diffusion for synthetic data')
    parser.add_argument('--use_optimization', action='store_true', help='Use GA-PSO optimization')
    parser.add_argument('--external_validation', type=str, default=None, help='Path to external validation dataset')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--delimiter', type=str, default=';', help='CSV delimiter')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Heart Disease Prediction Pipeline")
    print("="*60)
    
    # Load data
    print("\n[1/8] Loading data...")
    
    # Try to detect delimiter if not specified or if default causes issues
    try:
        df = pd.read_csv(args.data_path, delimiter=args.delimiter)
        # Check if we got a single column (delimiter issue)
        if df.shape[1] == 1 and ',' in str(df.columns[0]):
            print("Detected potential delimiter issue, trying comma delimiter...")
            df = pd.read_csv(args.data_path, delimiter=',')
    except Exception as e:
        print(f"Error with delimiter '{args.delimiter}': {e}")
        print("Trying comma delimiter...")
        df = pd.read_csv(args.data_path, delimiter=',')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Validate target column
    if args.target_column not in df.columns:
        print(f"Error: Target column '{args.target_column}' not found in dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Data preprocessing
    print("\n[2/8] Data preprocessing...")
    cleaner = DataCleaner()
    df = cleaner.fit_transform(df, args.target_column)
    
    # Skip outlier removal for datasets with severe class imbalance
    # to preserve minority class samples
    if 'framingham' in args.data_path.lower():
        print("Skipping outlier removal for Framingham dataset to preserve minority class.")
        df_clean = df
        df_outliers = pd.DataFrame()
    else:
        outlier_remover = OutlierRemover(method='both')
        df_clean, df_outliers = outlier_remover.fit_transform(df)
    
    # Feature engineering
    print("\n[2.5/8] Feature engineering...")
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.transform(df_clean, 
                                              add_bmi=True,
                                              add_pulse_pressure=True,
                                              add_age_groups=True,
                                              add_bp_category=True,
                                              add_risk_score=True)
    
    normalizer = Normalizer(method='standard')
    df_normalized = normalizer.fit_transform(df_engineered, args.target_column)
    
    print(f"Cleaned dataset shape: {df_normalized.shape}")
    
    # Check if both classes are present after preprocessing
    if args.target_column in df_normalized.columns:
        class_counts = df_normalized[args.target_column].value_counts()
        print(f"Class distribution after preprocessing: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            print("ERROR: Only one class present after preprocessing. Cannot train binary classifier.")
            print("This may be due to aggressive outlier removal. Consider:")
            print("  1. Disabling outlier removal")
            print("  2. Using less aggressive outlier thresholds")
            print("  3. Checking if the target column has correct values")
            return
    
    # Diffusion-based synthetic data generation
    if args.use_diffusion:
        print("\n[3/8] Training TabDDPM for synthetic data generation...")
        
        # Separate minority class for training diffusion model
        df_minority = df_normalized[df_normalized[args.target_column] == 1]
        df_majority = df_normalized[df_normalized[args.target_column] == 0]
        
        # Check if minority class has enough samples
        if len(df_minority) == 0:
            print("Warning: No samples in minority class. Skipping diffusion-based data generation.")
        elif len(df_minority) < 10:
            print(f"Warning: Minority class has only {len(df_minority)} samples. Skipping diffusion-based data generation.")
        else:
            # Train diffusion model on minority class features only
            trainer = TabDDPMTrainer(epochs=50)
            trainer.fit(df_minority, args.target_column)
            
            generator = TabDDPMGenerator(trainer)
            
            # Generate synthetic samples to balance the dataset
            samples_needed = len(df_majority) - len(df_minority)
            if samples_needed > 0:
                feature_names = [col for col in df_normalized.columns if col != args.target_column]
                synthetic_samples = generator.sample(samples_needed, feature_names)
                synthetic_samples[args.target_column] = 1
                
                # Combine with original data
                df_balanced = pd.concat([df_normalized, synthetic_samples], ignore_index=True)
                
                # Save comparison plots
                generator.compare_distributions(df_minority[feature_names], synthetic_samples[feature_names], 
                                               save_path=os.path.join(args.output_dir, 'distribution_comparison.png'))
                generator.compare_correlations(df_normalized, df_balanced, 
                                              save_path=os.path.join(args.output_dir, 'correlation_comparison.png'))
                
                df_normalized = df_balanced
                print(f"Balanced dataset: {len(df_normalized)} samples")
            else:
                print("Dataset is already balanced.")
    
    # Feature selection
    print("\n[4/8] Feature selection...")
    corr_analyzer = CorrelationAnalyzer()
    corr_analyzer.compute_correlation(df_normalized)
    
    feature_importance = FeatureImportanceAnalyzer()
    feature_importance.fit(df_normalized.drop(columns=[args.target_column]), 
                          df_normalized[args.target_column])
    
    selected_features = feature_importance.select_top_features(n_features=15)
    X = df_normalized[selected_features]
    y = df_normalized[args.target_column]
    
    print(f"Selected {len(selected_features)} features")
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Disable optimization if severe class imbalance or single class
    if len(class_counts) < 2:
        print("Warning: Only one class present in dataset. Disabling optimization.")
        args.use_optimization = False
    elif class_counts.min() / class_counts.sum() < 0.01:  # Less than 1% minority class
        print("Warning: Severe class imbalance detected. Disabling optimization.")
        args.use_optimization = False
    
    # GA-PSO optimization
    if args.use_optimization:
        print("\n[5/8] GA-PSO hyperparameter optimization...")
        optimizer = GAPSOHybridOptimizer(ga_generations=30, pso_iterations=50)
        best_params = optimizer.optimize(X, y, cv=5, verbose=True)
        
        # Save optimization history
        history = optimizer.get_optimization_history()
        history['ga_history'].to_csv(os.path.join(args.output_dir, 'ga_history.csv'), index=False)
        history['pso_history'].to_csv(os.path.join(args.output_dir, 'pso_history.csv'), index=False)
    else:
        best_params = None
    
    # Train XGBoost model
    print("\n[6/8] Training XGBoost model...")
    model = XGBoostModel()
    training_results = model.train(X, y, params=best_params, cv=5)
    
    # Save model
    model.save_model(os.path.join(args.output_dir, 'xgboost_model.pkl'))
    
    # Evaluation
    print("\n[7/8] Model evaluation...")
    metrics_calc = MetricsCalculator()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = metrics_calc.calculate_all_metrics(y_test, y_pred, y_proba)
    print(metrics_calc.generate_metrics_report(metrics, "XGBoost"))
    
    # Visualizations
    visualizer = Visualizer()
    visualizer.plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    visualizer.plot_roc_curve(y_test, y_proba, save_path=os.path.join(args.output_dir, 'roc_curve.png'))
    visualizer.plot_precision_recall_curve(y_test, y_proba, save_path=os.path.join(args.output_dir, 'pr_curve.png'))
    
    # SHAP analysis
    print("\n[8/8] SHAP explainability analysis...")
    shap_analyzer = SHAPAnalyzer(model.model)
    shap_analyzer.fit(X_test)
    shap_analyzer.plot_summary(X_test, save_path=os.path.join(args.output_dir, 'shap_summary.png'))
    shap_analyzer.plot_feature_importance(X_test, save_path=os.path.join(args.output_dir, 'shap_importance.png'))
    
    # External validation
    if args.external_validation:
        print("\n[External] External validation on Framingham dataset...")
        df_external = pd.read_csv(args.external_validation)
        
        # Preprocess external data
        df_external = cleaner.transform(df_external, args.target_column)
        df_external = normalizer.transform(df_external, args.target_column)
        
        # Ensure same features
        X_external = df_external[selected_features]
        y_external = df_external[args.target_column]
        
        external_results = model.evaluate_on_external_data(X_external, y_external)
        print(f"External AUC: {external_results['auc']:.4f}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
