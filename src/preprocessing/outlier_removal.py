import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierRemover:
    """
    Outlier removal module using Z-Score, IQR, Isolation Forest, and Local Outlier Factor methods.
    """
    
    def __init__(self, method: Literal['zscore', 'iqr', 'both', 'isolation_forest', 'lof', 'ensemble'] = 'both', 
                 z_threshold: float = 3.0, 
                 iqr_multiplier: float = 1.5,
                 contamination: float = 0.05,
                 n_neighbors: int = 20):
        """
        Initialize OutlierRemover.
        
        Args:
            method: Method to use for outlier detection ('zscore', 'iqr', 'both', 'isolation_forest', 'lof', 'ensemble')
            z_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
            contamination: Expected proportion of outliers for Isolation Forest and LOF
            n_neighbors: Number of neighbors for Local Outlier Factor
        """
        self.method = method
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.outlier_mask = None
        self.removed_indices = None
    
    def detect_zscore_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.Series:
        """
        Detect outliers using Z-Score method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (if None, check all numerical columns)
            
        Returns:
            Boolean Series indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                col_outliers = z_scores > self.z_threshold
                # Convert to Series with proper index
                col_outliers = pd.Series(col_outliers, index=df[col].dropna().index)
                outlier_mask = outlier_mask | col_outliers.reindex(df.index, fill_value=False)
        
        return outlier_mask
    
    def detect_iqr_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (if None, check all numerical columns)
            
        Returns:
            Boolean Series indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        return outlier_mask
    
    def detect_isolation_forest_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.Series:
        """
        Detect outliers using Isolation Forest method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (if None, check all numerical columns)
            
        Returns:
            Boolean Series indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_subset = df[columns].copy()
        
        # Handle missing values
        df_subset = df_subset.fillna(df_subset.median())
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(df_subset)
        
        # Isolation Forest returns -1 for outliers, 1 for inliers
        outlier_mask = pd.Series(outlier_labels == -1, index=df.index)
        
        return outlier_mask
    
    def detect_lof_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.Series:
        """
        Detect outliers using Local Outlier Factor method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (if None, check all numerical columns)
            
        Returns:
            Boolean Series indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_subset = df[columns].copy()
        
        # Handle missing values
        df_subset = df_subset.fillna(df_subset.median())
        
        # Fit Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
        outlier_labels = lof.fit_predict(df_subset)
        
        # LOF returns -1 for outliers, 1 for inliers
        outlier_mask = pd.Series(outlier_labels == -1, index=df.index)
        
        return outlier_mask
    
    def fit(self, df: pd.DataFrame, columns: Optional[list] = None) -> 'OutlierRemover':
        """
        Fit the outlier remover on the dataset.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            
        Returns:
            self: Fitted OutlierRemover instance
        """
        if self.method == 'zscore':
            self.outlier_mask = self.detect_zscore_outliers(df, columns)
        elif self.method == 'iqr':
            self.outlier_mask = self.detect_iqr_outliers(df, columns)
        elif self.method == 'both':
            zscore_mask = self.detect_zscore_outliers(df, columns)
            iqr_mask = self.detect_iqr_outliers(df, columns)
            self.outlier_mask = zscore_mask | iqr_mask
        elif self.method == 'isolation_forest':
            self.outlier_mask = self.detect_isolation_forest_outliers(df, columns)
        elif self.method == 'lof':
            self.outlier_mask = self.detect_lof_outliers(df, columns)
        elif self.method == 'ensemble':
            # Ensemble: combine all methods
            zscore_mask = self.detect_zscore_outliers(df, columns)
            iqr_mask = self.detect_iqr_outliers(df, columns)
            iso_mask = self.detect_isolation_forest_outliers(df, columns)
            lof_mask = self.detect_lof_outliers(df, columns)
            # Outlier if detected by at least 2 methods
            self.outlier_mask = (zscore_mask.astype(int) + iqr_mask.astype(int) + 
                                iso_mask.astype(int) + lof_mask.astype(int)) >= 2
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.removed_indices = df[self.outlier_mask].index
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove outliers from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, removed outliers DataFrame)
        """
        if self.outlier_mask is None:
            raise ValueError("OutlierRemover has not been fitted. Call fit() first.")
        
        df_cleaned = df[~self.outlier_mask].copy()
        df_outliers = df[self.outlier_mask].copy()
        
        return df_cleaned, df_outliers
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            
        Returns:
            Tuple of (cleaned DataFrame, removed outliers DataFrame)
        """
        return self.fit(df, columns).transform(df)
    
    def get_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of outliers by column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outlier summary
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary_data = []
        
        for col in numerical_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Z-score outliers
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                z_outliers = (z_scores > self.z_threshold).sum()
                
                # IQR outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                summary_data.append({
                    'Column': col,
                    'Z-Score Outliers': z_outliers,
                    'IQR Outliers': iqr_outliers,
                    'Z-Score %': (z_outliers / len(df)) * 100,
                    'IQR %': (iqr_outliers / len(df)) * 100
                })
        
        return pd.DataFrame(summary_data)
    
    def cap_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Cap outliers instead of removing them (winsorization).
        
        Args:
            df: Input DataFrame
            columns: List of columns to cap
            
        Returns:
            DataFrame with capped outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_capped = df.copy()
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_capped
