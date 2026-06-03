import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Optional, Dict, Tuple
import joblib


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    """
    
    def __init__(self, model):
        """
        Initialize SHAP Analyzer.
        
        Args:
            model: Trained XGBoost model
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame):
        """
        Fit SHAP explainer on the data.
        
        Args:
            X: Feature DataFrame
        """
        self.feature_names = X.columns.tolist()
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X)
        
        print("SHAP explainer fitted successfully.")
    
    def get_shap_values(self) -> np.ndarray:
        """
        Get SHAP values.
        
        Returns:
            SHAP values array
        """
        if self.shap_values is None:
            raise ValueError("SHAP analyzer has not been fitted. Call fit() first.")
        
        return self.shap_values
    
    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            X: Feature DataFrame
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.shap_values is None:
            self.fit(X)
        
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
    
    def plot_force_plot(self, X: pd.DataFrame, instance_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot SHAP force plot for a single instance.
        
        Args:
            X: Feature DataFrame
            instance_idx: Index of the instance to explain
            save_path: Optional path to save the plot
        """
        if self.shap_values is None:
            self.fit(X)
        
        # Get instance data
        instance_data = X.iloc[instance_idx]
        instance_shap = self.shap_values[instance_idx]
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            instance_shap,
            instance_data,
            feature_names=self.feature_names,
            show=False,
            matplotlib=True
        )
        
        plt.title(f'SHAP Force Plot - Instance {instance_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP force plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, X: pd.DataFrame, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot SHAP feature importance (mean absolute SHAP values).
        
        Args:
            X: Feature DataFrame
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.shap_values is None:
            self.fit(X)
        
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, X, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_dependence(self, X: pd.DataFrame, feature: str, 
                       interaction_feature: Optional[str] = None,
                       save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot SHAP dependence plot for a feature.
        
        Args:
            X: Feature DataFrame
            feature: Feature name to plot
            interaction_feature: Optional interaction feature
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.shap_values is None:
            self.fit(X)
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not in feature names.")
        
        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature, 
            self.shap_values, 
            X, 
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP dependence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP analyzer has not been fitted. Call fit() first.")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """
        Get explanation for a single instance.
        
        Args:
            X: Feature DataFrame
            instance_idx: Index of the instance to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.fit(X)
        
        instance_data = X.iloc[instance_idx]
        instance_shap = self.shap_values[instance_idx]
        base_value = self.explainer.expected_value
        
        # Sort features by SHAP value magnitude
        feature_impact = []
        for i, feature in enumerate(self.feature_names):
            feature_impact.append({
                'feature': feature,
                'value': instance_data.iloc[i],
                'shap_value': instance_shap[i],
                'impact': abs(instance_shap[i])
            })
        
        feature_impact.sort(key=lambda x: x['impact'], reverse=True)
        
        explanation = {
            'instance_index': instance_idx,
            'base_value': base_value,
            'prediction': base_value + instance_shap.sum(),
            'feature_contributions': feature_impact
        }
        
        return explanation
    
    def plot_waterfall(self, X: pd.DataFrame, instance_idx: int = 0, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            X: Feature DataFrame
            instance_idx: Index of the instance to explain
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.shap_values is None:
            self.fit(X)
        
        instance_data = X.iloc[instance_idx]
        instance_shap = self.shap_values[instance_idx]
        
        plt.figure(figsize=figsize)
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap,
                base_values=self.explainer.expected_value,
                data=instance_data.values,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_explainer(self, filepath: str):
        """
        Save the SHAP explainer.
        
        Args:
            filepath: Path to save the explainer
        """
        if self.explainer is None:
            raise ValueError("SHAP analyzer has not been fitted. Call fit() first.")
        
        explainer_data = {
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'feature_names': self.feature_names
        }
        
        joblib.dump(explainer_data, filepath)
        print(f"SHAP explainer saved to {filepath}")
    
    def load_explainer(self, filepath: str) -> 'SHAPAnalyzer':
        """
        Load a SHAP explainer.
        
        Args:
            filepath: Path to load the explainer from
            
        Returns:
            self: SHAPAnalyzer instance with loaded explainer
        """
        explainer_data = joblib.load(filepath)
        
        self.explainer = explainer_data['explainer']
        self.shap_values = explainer_data['shap_values']
        self.feature_names = explainer_data['feature_names']
        
        print(f"SHAP explainer loaded from {filepath}")
        return self
