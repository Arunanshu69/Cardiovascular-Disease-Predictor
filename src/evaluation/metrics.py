import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    matthews_corrcoef
)


class MetricsCalculator:
    """
    Metrics calculator for model evaluation.
    """
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        pass
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: np.ndarray) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        if tn + fp == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def calculate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_proba)
    
    def calculate_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_proba)
    
    def compare_models(self, metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(metrics_dict).T
        
        # Add rank for each metric
        for metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def calculate_improvement(self, baseline_metrics: Dict, 
                             new_metrics: Dict) -> Dict:
        """
        Calculate improvement between baseline and new metrics.
        
        Args:
            baseline_metrics: Baseline metrics dictionary
            new_metrics: New metrics dictionary
            
        Returns:
            Dictionary with improvements
        """
        improvements = {}
        
        for metric in baseline_metrics:
            if metric in new_metrics:
                improvement = new_metrics[metric] - baseline_metrics[metric]
                improvements[metric] = {
                    'absolute': improvement,
                    'percentage': (improvement / baseline_metrics[metric]) * 100 if baseline_metrics[metric] != 0 else 0
                }
        
        return improvements
    
    def generate_metrics_report(self, metrics: Dict, model_name: str = "Model") -> str:
        """
        Generate a formatted metrics report.
        
        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        report = f"\n{'='*60}\n"
        report += f"{model_name} Performance Report\n"
        report += f"{'='*60}\n"
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                report += f"{metric:20s}: {value:.4f}\n"
            else:
                report += f"{metric:20s}: {value}\n"
        
        report += f"{'='*60}\n"
        
        return report
