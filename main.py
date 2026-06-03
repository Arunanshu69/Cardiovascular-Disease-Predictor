import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import modules
from src.preprocessing import DataCleaner, OutlierRemover, Normalizer
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
    parser.add_argument('--use_diffusion', action='store_true', help='Use diffusion for synthetic data')
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
    df = pd.read_csv(args.data_path, delimiter=args.delimiter)
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
    
    outlier_remover = OutlierRemover(method='both')
    df_clean, df_outliers = outlier_remover.fit_transform(df)
    
    normalizer = Normalizer(method='standard')
    df_normalized = normalizer.fit_transform(df_clean, args.target_column)
    
    print(f"Cleaned dataset shape: {df_normalized.shape}")
    
    # Diffusion-based synthetic data generation
    if args.use_diffusion:
        print("\n[3/8] Training TabDDPM for synthetic data generation...")
        trainer = TabDDPMTrainer(epochs=50)
        trainer.fit(df_normalized, args.target_column)
        
        generator = TabDDPMGenerator(trainer)
        df_balanced = generator.generate_balanced_dataset(df_normalized, args.target_column)
        
        # Save comparison plots
        generator.compare_distributions(df_normalized, df_balanced[df_balanced[args.target_column] == 1], 
                                       save_path=os.path.join(args.output_dir, 'distribution_comparison.png'))
        generator.compare_correlations(df_normalized, df_balanced, 
                                      save_path=os.path.join(args.output_dir, 'correlation_comparison.png'))
        
        df_normalized = df_balanced
    
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
    
    # GA-PSO optimization
    if args.use_optimization:
        print("\n[5/8] GA-PSO hyperparameter optimization...")
        optimizer = GAPSOHybridOptimizer(ga_generations=20, pso_iterations=30)
        best_params = optimizer.optimize(X, y, cv=3, verbose=True)
        
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
