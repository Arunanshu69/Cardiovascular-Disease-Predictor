import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Feature engineering module for creating derived features.
    """
    
    def __init__(self):
        self.engineered_features = None
        # Column name mapping for different datasets
        self.column_mapping = {
            # pkiohd dataset
            'HEIGHT': ['HEIGHT', 'height'],
            'WEIGHT': ['WEIGHT', 'weight'],
            'AP_HIGH': ['AP_HIGH', 'sysBP', 'systolic'],
            'AP_LOW': ['AP_LOW', 'diaBP', 'diastolic'],
            'AGE': ['AGE', 'age'],
            'CHOLESTEROL': ['CHOLESTEROL', 'totChol', 'cholesterol'],
            'GLUCOSE': ['GLUCOSE', 'glucose'],
            'GENDER': ['GENDER', 'male', 'sex'],
            'SMOKE': ['SMOKE', 'currentSmoker', 'smoking'],
            'ALCOHOL': ['ALCOHOL', 'alcohol'],
            'PHYSICAL_ACTIVITY': ['PHYSICAL_ACTIVITY', 'physical_activity']
        }
    
    def find_column(self, df: pd.DataFrame, column_name: str) -> str:
        """
        Find the actual column name in the DataFrame based on mapping.
        
        Args:
            df: Input DataFrame
            column_name: Base column name to look for
            
        Returns:
            Actual column name if found, None otherwise
        """
        if column_name in df.columns:
            return column_name
        
        # Check mapped alternatives
        if column_name in self.column_mapping:
            for alt_name in self.column_mapping[column_name]:
                if alt_name in df.columns:
                    return alt_name
        
        return None
    
    def add_bmi(self, df: pd.DataFrame, height_col: str = 'HEIGHT', 
                weight_col: str = 'WEIGHT') -> pd.DataFrame:
        """
        Add BMI (Body Mass Index) feature from height and weight.
        
        Args:
            df: Input DataFrame
            height_col: Name of height column (in cm)
            weight_col: Name of weight column (in kg)
            
        Returns:
            DataFrame with BMI column added
        """
        df = df.copy()
        
        # Find actual column names using mapping
        actual_height = self.find_column(df, height_col)
        actual_weight = self.find_column(df, weight_col)
        
        if actual_height and actual_weight:
            # BMI = weight (kg) / height (m)^2
            height_m = df[actual_height] / 100
            df['BMI'] = df[actual_weight] / (height_m ** 2)
            print(f"Added BMI feature from {actual_height} and {actual_weight}")
        else:
            print(f"Warning: {height_col} or {weight_col} not found in DataFrame")
        
        return df
    
    def add_pulse_pressure(self, df: pd.DataFrame, systolic_col: str = 'AP_HIGH',
                         diastolic_col: str = 'AP_LOW') -> pd.DataFrame:
        """
        Add pulse pressure feature (systolic - diastolic blood pressure).
        
        Args:
            df: Input DataFrame
            systolic_col: Name of systolic blood pressure column
            diastolic_col: Name of diastolic blood pressure column
            
        Returns:
            DataFrame with pulse pressure column added
        """
        df = df.copy()
        
        # Find actual column names using mapping
        actual_systolic = self.find_column(df, systolic_col)
        actual_diastolic = self.find_column(df, diastolic_col)
        
        if actual_systolic and actual_diastolic:
            df['PULSE_PRESSURE'] = df[actual_systolic] - df[actual_diastolic]
            print(f"Added PULSE_PRESSURE feature from {actual_systolic} and {actual_diastolic}")
        else:
            print(f"Warning: {systolic_col} or {diastolic_col} not found in DataFrame")
        
        return df
    
    def add_age_groups(self, df: pd.DataFrame, age_col: str = 'AGE') -> pd.DataFrame:
        """
        Add age group categories (numeric encoding only).
        
        Args:
            df: Input DataFrame
            age_col: Name of age column
            
        Returns:
            DataFrame with age group column added
        """
        df = df.copy()
        
        # Find actual column name using mapping
        actual_age = self.find_column(df, age_col)
        
        if actual_age:
            # Define age groups with numeric encoding directly
            def categorize_age(age):
                if age < 30:
                    return 0  # Young Adult
                elif age < 40:
                    return 1  # Adult
                elif age < 50:
                    return 2  # Middle Age
                elif age < 60:
                    return 3  # Senior
                elif age < 70:
                    return 4  # Elderly
                else:
                    return 5  # Senior+
            
            df['AGE_GROUP'] = df[actual_age].apply(categorize_age)
            print(f"Added AGE_GROUP feature from {actual_age}")
        else:
            print(f"Warning: {age_col} not found in DataFrame")
        
        return df
    
    def add_blood_pressure_category(self, df: pd.DataFrame, systolic_col: str = 'AP_HIGH',
                                   diastolic_col: str = 'AP_LOW') -> pd.DataFrame:
        """
        Add blood pressure category based on systolic and diastolic values (numeric encoding only).
        
        Args:
            df: Input DataFrame
            systolic_col: Name of systolic blood pressure column
            diastolic_col: Name of diastolic blood pressure column
            
        Returns:
            DataFrame with BP category column added
        """
        df = df.copy()
        
        # Find actual column names using mapping
        actual_systolic = self.find_column(df, systolic_col)
        actual_diastolic = self.find_column(df, diastolic_col)
        
        if actual_systolic and actual_diastolic:
            def categorize_bp(row):
                systolic = row[actual_systolic]
                diastolic = row[actual_diastolic]
                
                if systolic < 120 and diastolic < 80:
                    return 0  # Normal
                elif systolic < 140 or diastolic < 90:
                    return 1  # Elevated
                elif systolic < 160 or diastolic < 100:
                    return 2  # High Stage 1
                else:
                    return 3  # High Stage 2
            
            df['BP_CATEGORY'] = df.apply(categorize_bp, axis=1)
            print(f"Added BP_CATEGORY feature")
        else:
            print(f"Warning: {systolic_col} or {diastolic_col} not found in DataFrame")
        
        return df
    
    def add_risk_score(self, df: pd.DataFrame, cholesterol_col: str = 'CHOLESTEROL',
                      glucose_col: str = 'GLUCOSE', age_col: str = 'AGE') -> pd.DataFrame:
        """
        Add simple metabolic risk score based on cholesterol, glucose, and age.
        
        Args:
            df: Input DataFrame
            cholesterol_col: Name of cholesterol column
            glucose_col: Name of glucose column
            age_col: Name of age column
            
        Returns:
            DataFrame with risk score column added
        """
        df = df.copy()
        
        # Find actual column names using mapping
        actual_cholesterol = self.find_column(df, cholesterol_col)
        actual_glucose = self.find_column(df, glucose_col)
        actual_age = self.find_column(df, age_col)
        
        required_cols = [actual_cholesterol, actual_glucose, actual_age]
        if all(col for col in required_cols):
            # Simple risk score calculation
            # Normalize each component and sum
            def calculate_risk_score(row):
                cholesterol_score = (row[actual_cholesterol] - 1) / 2  # Assuming 1-3 scale
                glucose_score = (row[actual_glucose] - 1) / 2  # Assuming 1-3 scale
                age_score = row[actual_age] / 100  # Normalize age
                return cholesterol_score + glucose_score + age_score
            
            df['RISK_SCORE'] = df.apply(calculate_risk_score, axis=1)
            print(f"Added RISK_SCORE feature")
        else:
            print(f"Warning: Required columns not found for risk score")
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                 add_bmi: bool = True,
                 add_pulse_pressure: bool = True,
                 add_age_groups: bool = True,
                 add_bp_category: bool = True,
                 add_risk_score: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame
            add_bmi: Whether to add BMI feature
            add_pulse_pressure: Whether to add pulse pressure feature
            add_age_groups: Whether to add age group features
            add_bp_category: Whether to add blood pressure category features
            add_risk_score: Whether to add risk score feature
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        if add_bmi:
            df = self.add_bmi(df)
        
        if add_pulse_pressure:
            df = self.add_pulse_pressure(df)
        
        if add_age_groups:
            df = self.add_age_groups(df)
        
        if add_bp_category:
            df = self.add_blood_pressure_category(df)
        
        if add_risk_score:
            df = self.add_risk_score(df)
        
        # Track engineered features (all new features are numeric)
        self.engineered_features = [col for col in df.columns if col not in 
                                   ['AGE', 'GENDER', 'HEIGHT', 'WEIGHT', 'AP_HIGH', 'AP_LOW', 
                                    'CHOLESTEROL', 'GLUCOSE', 'SMOKE', 'ALCOHOL', 'PHYSICAL_ACTIVITY',
                                    'CARDIO_DISEASE', 'TenYearCHD']]
        
        print(f"Engineered features: {self.engineered_features}")
        
        return df
