import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataCleaner:
    """
    Data cleaning module for handling missing values and basic data preprocessing.
    """
    
    def __init__(self):
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.numerical_columns = None
        self.categorical_columns = None
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'DataCleaner':
        """
        Fit the data cleaner on the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from imputation)
            
        Returns:
            self: Fitted DataCleaner instance
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        # Identify numerical and categorical columns
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store imputation values
        self.numerical_imputer = {}
        self.categorical_imputer = {}
        
        for col in self.numerical_columns:
            self.numerical_imputer[col] = df[col].median()
        
        for col in self.categorical_columns:
            self.categorical_imputer[col] = df[col].mode()[0] if not df[col].mode().empty else df[col].value_counts().idxmax()
        
        return self
    
    def transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Apply data cleaning transformations to the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from imputation)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            target = None
        
        # Impute numerical columns with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(self.numerical_imputer[col])
        
        # Impute categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(self.categorical_imputer[col])
        
        # Reattach target column if it was separated
        if target is not None:
            df[target_column] = target
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from imputation)
            
        Returns:
            Cleaned DataFrame
        """
        return self.fit(df, target_column).transform(df, target_column)
    
    def get_missing_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value information
        """
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_percent
        })
        missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        return missing_info
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        removed_rows = initial_rows - len(df_cleaned)
        
        print(f"Removed {removed_rows} duplicate rows ({(removed_rows/initial_rows)*100:.2f}%)")
        
        return df_cleaned
    
    def handle_inconsistent_categories(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Handle inconsistent categorical values (e.g., case sensitivity, extra spaces).
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with consistent categorical values
        """
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                # Strip whitespace and convert to lowercase
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        return df
