import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib


class XGBoostModel:
    """
    XGBoost model wrapper for heart disease prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize XGBoost Model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def train(self, 
              X: pd.DataFrame, 
              y: pd.Series,
              params: Optional[Dict] = None,
              test_size: float = 0.2,
              cv: int = 10) -> Dict:
        """
        Train XGBoost model with optional cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            params: XGBoost hyperparameters
            test_size: Test set size for train-test split
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = X.columns.tolist()
        
        # Default parameters with regularization to prevent overfitting
        if params is None:
            params = {
                'max_depth': 4,  # Restricted to prevent overly complex trees
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'min_child_weight': 1,
                'reg_alpha': 0.5,  # Increased L1 regularization
                'reg_lambda': 2.0,  # Increased L2 regularization
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'n_jobs': -1
            }
        
        # Add class weight for imbalanced datasets
        params['scale_pos_weight'] = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Initialize and train model with early stopping
        self.model = XGBClassifier(**params)
        
        # Use early stopping if n_estimators is large enough
        if params.get('n_estimators', 100) > 50:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        self.is_fitted = True
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'params': params,
            'feature_names': self.feature_names
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'XGBoostModel':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self: XGBoostModel instance with loaded model
        """
        try:
            model_data = joblib.load(filepath)
            
            # Validate that the loaded data has the expected keys
            required_keys = ['model', 'feature_names', 'random_state']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Loaded model data is missing required key: {key}")
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.random_state = model_data['random_state']
            self.is_fitted = True
            
            print(f"Model loaded from {filepath}")
            return self
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading model from {filepath}: {e}")
    
    def get_model_params(self) -> Dict:
        """
        Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        return self.model.get_params()
    
    def evaluate_on_external_data(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series) -> Dict:
        """
        Evaluate model on external dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Ensure features match
        if set(X.columns) != set(self.feature_names):
            missing_features = set(self.feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_names)
            
            if missing_features:
                raise ValueError(f"Missing features in external data: {missing_features}")
            if extra_features:
                print(f"Warning: Extra features in external data will be ignored: {extra_features}")
            
            X = X[self.feature_names]
        
        # Predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"External evaluation completed!")
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return results
