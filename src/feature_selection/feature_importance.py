import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap


class FeatureImportanceAnalyzer:
    """
    Feature importance analysis using XGBoost.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureImportanceAnalyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.shap_values = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **xgb_params) -> 'FeatureImportanceAnalyzer':
        """
        Fit XGBoost model for feature importance analysis.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **xgb_params: Additional XGBoost parameters
            
        Returns:
            self: Fitted FeatureImportanceAnalyzer instance
        """
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'logloss'
        }
        
        # Update with provided parameters
        default_params.update(xgb_params)
        
        # Train model
        self.model = XGBClassifier(**default_params)
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance DataFrame.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.feature_importance
    
    def select_top_features(self, n_features: int) -> List[str]:
        """
        Select top N features by importance.
        
        Args:
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        top_features = self.feature_importance.head(n_features)['feature'].tolist()
        
        print(f"Selected top {n_features} features by importance: {top_features}")
        
        return top_features
    
    def plot_feature_importance(self, 
                               n_features: int = 20,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            n_features: Number of top features to plot
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        top_features = self.feature_importance.head(n_features)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Top {n_features} Feature Importance (XGBoost)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for feature importance.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            SHAP values array
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        
        return self.shap_values
    
    def plot_shap_summary(self, 
                          X: pd.DataFrame,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            X: Feature DataFrame
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, X, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_shap_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with SHAP-based feature importance
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return shap_df
    
    def recursive_feature_elimination(self, 
                                     X: pd.DataFrame, 
                                     y: pd.Series,
                                     n_features_to_select: int = 10,
                                     step: int = 1) -> List[str]:
        """
        Perform recursive feature elimination using XGBoost.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features_to_select: Number of features to select
            step: Number of features to remove at each iteration
            
        Returns:
            List of selected feature names
        """
        current_features = X.columns.tolist()
        
        while len(current_features) > n_features_to_select:
            # Train model on current features
            X_current = X[current_features]
            self.fit(X_current, y)
            
            # Get feature importance
            importance = self.get_feature_importance()
            
            # Remove least important features
            features_to_remove = importance.tail(step)['feature'].tolist()
            current_features = [f for f in current_features if f not in features_to_remove]
            
            print(f"Removed features: {features_to_remove}")
            print(f"Remaining features: {len(current_features)}")
        
        # Final fit on selected features
        self.fit(X[current_features], y)
        
        print(f"Final selected features: {current_features}")
        
        return current_features
    
    def evaluate_feature_subset(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                feature_subset: List[str],
                                cv: int = 5) -> float:
        """
        Evaluate a feature subset using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_subset: List of features to evaluate
            cv: Number of cross-validation folds
            
        Returns:
            Mean cross-validation score
        """
        X_subset = X[feature_subset]
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**default_params)
        
        scores = cross_val_score(model, X_subset, y, cv=cv, scoring='roc_auc')
        
        return scores.mean()
