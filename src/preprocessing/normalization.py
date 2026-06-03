import pandas as pd
import numpy as np
from typing import Optional, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib


class Normalizer:
    """
    Feature scaling and normalization module.
    """
    
    def __init__(self, method: Literal['standard', 'minmax', 'robust'] = 'standard'):
        """
        Initialize Normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', or 'robust')
        """
        self.method = method
        self.scaler = None
        self.numerical_columns = None
        self.scalers_dict = {}
    
    def fit(self, df: pd.DataFrame, columns: Optional[list] = None, 
            target_column: Optional[str] = None) -> 'Normalizer':
        """
        Fit the normalizer on the dataset.
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize (if None, normalize all numerical columns)
            target_column: Name of target column (to exclude from normalization)
            
        Returns:
            self: Fitted Normalizer instance
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        if columns is None:
            self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numerical_columns = [col for col in columns if col in df.columns]
        
        # Initialize scaler based on method
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Fit scaler on numerical columns
        if self.numerical_columns and self.scaler is not None:
            self.scaler.fit(df[self.numerical_columns])
        
        return self
    
    def transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Apply normalization to the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from normalization)
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            target = None
        
        if self.numerical_columns and self.scaler is not None:
            df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        
        # Reattach target column if it was separated
        if target is not None:
            df[target_column] = target
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[list] = None,
                     target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            target_column: Name of target column (to exclude from normalization)
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(df, columns, target_column).transform(df, target_column)
    
    def inverse_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Inverse transform the normalized data back to original scale.
        
        Args:
            df: Normalized DataFrame
            target_column: Name of target column (to exclude from inverse transformation)
            
        Returns:
            DataFrame in original scale
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            target = None
        
        if self.numerical_columns and self.scaler is not None:
            df[self.numerical_columns] = self.scaler.inverse_transform(df[self.numerical_columns])
        
        # Reattach target column if it was separated
        if target is not None:
            df[target_column] = target
        
        return df
    
    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler to disk.
        
        Args:
            filepath: Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        scaler_data = {
            'scaler': self.scaler,
            'method': self.method,
            'numerical_columns': self.numerical_columns
        }
        joblib.dump(scaler_data, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> 'Normalizer':
        """
        Load a fitted scaler from disk.
        
        Args:
            filepath: Path to load the scaler from
            
        Returns:
            self: Normalizer instance with loaded scaler
        """
        scaler_data = joblib.load(filepath)
        self.scaler = scaler_data['scaler']
        self.method = scaler_data['method']
        self.numerical_columns = scaler_data['numerical_columns']
        
        print(f"Scaler loaded from {filepath}")
        return self
    
    def get_scaling_params(self) -> dict:
        """
        Get scaling parameters for each numerical column.
        
        Returns:
            Dictionary with scaling parameters
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        params = {}
        
        if self.method == 'standard':
            params['mean'] = dict(zip(self.numerical_columns, self.scaler.mean_))
            params['std'] = dict(zip(self.numerical_columns, self.scaler.scale_))
        elif self.method == 'minmax':
            params['min'] = dict(zip(self.numerical_columns, self.scaler.data_min_))
            params['max'] = dict(zip(self.numerical_columns, self.scaler.data_max_))
        elif self.method == 'robust':
            params['center'] = dict(zip(self.numerical_columns, self.scaler.center_))
            params['scale'] = dict(zip(self.numerical_columns, self.scaler.scale_))
        
        return params
