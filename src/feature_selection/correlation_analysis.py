import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from scipy.stats import pearsonr, spearmanr


class CorrelationAnalyzer:
    """
    Correlation analysis module for feature selection.
    """
    
    def __init__(self, method: str = 'pearson', threshold: float = 0.8):
        """
        Initialize CorrelationAnalyzer.
        
        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            threshold: Threshold for identifying highly correlated features
        """
        self.method = method
        self.threshold = threshold
        self.correlation_matrix = None
        self.highly_correlated_features = None
    
    def compute_correlation(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute correlation matrix.
        
        Args:
            df: Input DataFrame
            columns: List of columns to analyze (if None, analyze all numerical columns)
            
        Returns:
            Correlation matrix
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_subset = df[columns]
        
        if self.method == 'pearson':
            self.correlation_matrix = df_subset.corr(method='pearson')
        elif self.method == 'spearman':
            self.correlation_matrix = df_subset.corr(method='spearman')
        elif self.method == 'kendall':
            self.correlation_matrix = df_subset.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")
        
        return self.correlation_matrix
    
    def get_highly_correlated_features(self, df: pd.DataFrame, 
                                       target_column: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated feature pairs.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from correlation analysis)
            
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        if self.correlation_matrix is None:
            self.compute_correlation(df)
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        highly_correlated = []
        for col1 in upper_triangle.columns:
            for col2 in upper_triangle.index:
                corr_value = upper_triangle.loc[col2, col1]
                if not pd.isna(corr_value) and abs(corr_value) >= self.threshold:
                    # Skip if either feature is the target
                    if target_column and (col1 == target_column or col2 == target_column):
                        continue
                    highly_correlated.append((col1, col2, corr_value))
        
        # Sort by absolute correlation
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.highly_correlated_features = highly_correlated
        
        return highly_correlated
    
    def get_correlation_with_target(self, df: pd.DataFrame, 
                                   target_column: str) -> pd.Series:
        """
        Get correlation of each feature with the target.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Series with correlation values
        """
        if self.correlation_matrix is None:
            self.compute_correlation(df)
        
        if target_column not in self.correlation_matrix.columns:
            raise ValueError(f"Target column '{target_column}' not in correlation matrix")
        
        target_corr = self.correlation_matrix[target_column].drop(target_column)
        target_corr = target_corr.sort_values(key=abs, ascending=False)
        
        return target_corr
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                         target_column: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove one feature from each highly correlated pair.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to keep)
            
        Returns:
            Tuple of (DataFrame with removed features, list of removed feature names)
        """
        highly_correlated = self.get_highly_correlated_features(df, target_column)
        
        features_to_remove = set()
        
        for feat1, feat2, corr in highly_correlated:
            # Keep the feature with higher correlation to target if target is specified
            if target_column:
                if target_column in self.correlation_matrix.columns:
                    corr1 = abs(self.correlation_matrix.loc[feat1, target_column])
                    corr2 = abs(self.correlation_matrix.loc[feat2, target_column])
                    
                    if corr1 < corr2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
                else:
                    # If no target, remove the second feature
                    features_to_remove.add(feat2)
            else:
                # If no target, remove the second feature
                features_to_remove.add(feat2)
        
        df_filtered = df.drop(columns=list(features_to_remove))
        
        print(f"Removed {len(features_to_remove)} highly correlated features: {list(features_to_remove)}")
        
        return df_filtered, list(features_to_remove)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        if self.correlation_matrix is None:
            self.compute_correlation(df)
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'{self.method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_target_correlation(self, df: pd.DataFrame, 
                               target_column: str,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)):
        """
        Plot correlation of features with target.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        target_corr = self.get_correlation_with_target(df, target_column)
        
        plt.figure(figsize=figsize)
        colors = ['red' if x < 0 else 'green' for x in target_corr.values]
        target_corr.plot(kind='barh', color=colors)
        plt.xlabel(f'{self.method.capitalize()} Correlation')
        plt.ylabel('Features')
        plt.title(f'Feature Correlation with {target_column}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Target correlation plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def select_features_by_correlation(self, df: pd.DataFrame, 
                                       target_column: str,
                                       n_features: int = 10) -> List[str]:
        """
        Select top features based on correlation with target.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        target_corr = self.get_correlation_with_target(df, target_column)
        
        selected_features = target_corr.abs().nlargest(n_features).index.tolist()
        
        print(f"Selected top {n_features} features by correlation: {selected_features}")
        
        return selected_features
