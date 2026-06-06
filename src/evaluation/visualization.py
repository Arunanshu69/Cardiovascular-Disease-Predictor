import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class Visualizer:
    """
    Visualization module for model evaluation.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize Visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[list] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model",
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = np.trapezoid(tpr, fpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (8, 6)):
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = np.trapezoid(precision, recall)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                top_n: int = 20,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Plot metrics comparison across models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        df = pd.DataFrame(metrics_dict).T
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Bar chart of metrics
        df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_ylabel('Score')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Radar chart
        metrics = df.columns.tolist()
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = plt.subplot(122, projection='polar')
        
        for model in df.index:
            values = df.loc[model].values.tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model)
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_title('Model Performance Radar Chart')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optimization_history(self, history_df: pd.DataFrame,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 5)):
        """
        Plot optimization history.
        
        Args:
            history_df: DataFrame with optimization history
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot best score over iterations
        axes[0].plot(history_df['best_score'], label='Best Score', linewidth=2)
        axes[0].plot(history_df['avg_score'], label='Average Score', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Iteration/Generation')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Optimization Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot improvement
        if len(history_df) > 1:
            improvement = history_df['best_score'].diff().fillna(0)
            axes[1].bar(range(len(improvement)), improvement)
            axes[1].set_xlabel('Iteration/Generation')
            axes[1].set_ylabel('Score Improvement')
            axes[1].set_title('Score Improvement per Iteration')
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
