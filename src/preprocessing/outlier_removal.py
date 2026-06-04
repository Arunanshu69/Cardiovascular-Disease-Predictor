import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal
from scipy import stats


class OutlierRemover:
    """
    Outlier removal module using Z-Score and IQR methods.
    """
    
    def __init__(self, method: Literal['zscore', 'iqr', 'both'] = 'both', 
                 z_threshold: float = 3.0, 
                 iqr_multiplier: float = 1.5):
        """
        Initialize OutlierRemover.
        
        Args:
            method: Method to use for outlier detection ('zscore', 'iqr', or 'both')
            z_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
        """
        self.method = method
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
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
