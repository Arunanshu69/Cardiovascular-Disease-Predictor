import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for XGBoost hyperparameter refinement.
    """
    
    def __init__(self,
                 n_particles: int = 30,
                 iterations: int = 50,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 random_state: int = 42):
        """
        Initialize Particle Swarm Optimizer.
        
        Args:
            n_particles: Number of particles
            iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            random_state: Random state for reproducibility
        """
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        
        # Hyperparameter search space
        self.param_bounds = {
            'max_depth': (3, 6),  # Reduced upper bound for simpler models
            'learning_rate': (0.01, 0.2),
            'n_estimators': (100, 500),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0),  # Restricted
            'gamma': (0.0, 2.0),  # Restricted
            'min_child_weight': (1, 5),  # Restricted
            'reg_alpha': (0.1, 2.0),  # Increased upper bound for L1 regularization
            'reg_lambda': (1.0, 5.0)  # Increased upper bound for L2 regularization
        }
        
        self.best_params = None
        self.best_score = None
        self.history = []
        
        # Particle swarm components
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = None
    
    def _initialize_particles(self, initial_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize particle positions and velocities.
        
        Args:
            initial_params: Optional initial parameters to center particles around
            
        Returns:
            Tuple of (positions, velocities)
        """
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)
        
        # Initialize positions
        positions = np.zeros((self.n_particles, n_params))
        
        for i, param in enumerate(param_names):
            min_val, max_val = self.param_bounds[param]
            
            if initial_params and param in initial_params:
                # Center around initial parameters with some noise
                center = initial_params[param]
                range_val = max_val - min_val
                positions[:, i] = np.random.uniform(
                    max(min_val, center - 0.3 * range_val),
                    min(max_val, center + 0.3 * range_val),
                    self.n_particles
                )
            else:
                positions[:, i] = np.random.uniform(min_val, max_val, self.n_particles)
        
        # Initialize velocities
        velocities = np.zeros((self.n_particles, n_params))
        for i, param in enumerate(param_names):
            min_val, max_val = self.param_bounds[param]
            range_val = max_val - min_val
            velocities[:, i] = np.random.uniform(-0.1 * range_val, 0.1 * range_val, self.n_particles)
        
        return positions, velocities
    
    def _evaluate_particle(self, position: np.ndarray, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
        """
        Evaluate fitness of a particle position.
        
        Args:
            position: Particle position array
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            ROC-AUC score
        """
        param_names = list(self.param_bounds.keys())
        params = {}
        
        for i, param in enumerate(param_names):
            if param in ['max_depth', 'n_estimators', 'min_child_weight']:
                params[param] = int(np.clip(position[i], self.param_bounds[param][0], self.param_bounds[param][1]))
            else:
                params[param] = np.clip(position[i], self.param_bounds[param][0], self.param_bounds[param][1])
        
        try:
            # Use StratifiedKFold for balanced class distribution
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Calculate scale_pos_weight for class imbalance
                scale_pos_weight = (len(y_train_fold) - y_train_fold.sum()) / max(y_train_fold.sum(), 1)
                
                model = XGBClassifier(
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    gamma=params['gamma'],
                    min_child_weight=params['min_child_weight'],
                    reg_alpha=params.get('reg_alpha', 0.1),
                    reg_lambda=params.get('reg_lambda', 1.0),
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=1
                )
                
                model.fit(X_train_fold, y_train_fold)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                
                from sklearn.metrics import roc_auc_score
                fold_score = roc_auc_score(y_val_fold, y_pred_proba)
                scores.append(fold_score)
            
            mean_score = np.mean(scores)
            
            # Handle NaN or invalid scores
            if np.isnan(mean_score) or np.isinf(mean_score):
                return 0.0
            
            return mean_score
        except Exception as e:
            return 0.0
    
    def _update_velocity(self, particle_idx: int) -> np.ndarray:
        """
        Update particle velocity.
        
        Args:
            particle_idx: Index of particle
            
        Returns:
            Updated velocity
        """
        r1 = np.random.random(len(self.param_bounds))
        r2 = np.random.random(len(self.param_bounds))
        
        cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.particles[particle_idx])
        
        # Handle case where global_best_position is None
        if self.global_best_position is not None:
            social = self.c2 * r2 * (self.global_best_position - self.particles[particle_idx])
        else:
            social = np.zeros_like(self.particles[particle_idx])
        
        new_velocity = self.w * self.velocities[particle_idx] + cognitive + social
        
        return new_velocity
    
    def _update_position(self, particle_idx: int) -> np.ndarray:
        """
        Update particle position.
        
        Args:
            particle_idx: Index of particle
            
        Returns:
            Updated position
        """
        new_position = self.particles[particle_idx] + self.velocities[particle_idx]
        
        # Clip to bounds
        for i, param in enumerate(self.param_bounds.keys()):
            min_val, max_val = self.param_bounds[param]
            new_position[i] = np.clip(new_position[i], min_val, max_val)
        
        return new_position
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, 
                initial_params: Optional[Dict] = None, verbose: bool = True) -> Dict:
        """
        Optimize XGBoost hyperparameters using particle swarm optimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            initial_params: Optional initial parameters from GA
            verbose: Whether to print progress
            
        Returns:
            Best parameter dictionary
        """
        np.random.seed(self.random_state)
        
        # Initialize particles
        self.particles, self.velocities = self._initialize_particles(initial_params)
        
        # Initialize personal best
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.zeros(self.n_particles)
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        if verbose:
            print(f"Starting Particle Swarm Optimization...")
            print(f"Particles: {self.n_particles}, Iterations: {self.iterations}")
            if initial_params:
                print(f"Initial parameters from GA: {initial_params}")
        
        for iteration in range(self.iterations):
            # Evaluate all particles
            for i in range(self.n_particles):
                score = self._evaluate_particle(self.particles[i], X, y, cv)
                
                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                self.velocities[i] = self._update_velocity(i)
                self.particles[i] = self._update_position(i)
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'best_score': self.global_best_score,
                'avg_score': np.mean(self.personal_best_scores)
            })
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}")
                print(f"Best Score: {self.global_best_score:.4f}")
                print(f"Average Score: {np.mean(self.personal_best_scores):.4f}")
        
        # Convert best position to parameter dictionary
        param_names = list(self.param_bounds.keys())
        self.best_params = {}
        for i, param in enumerate(param_names):
            if param in ['max_depth', 'n_estimators', 'min_child_weight']:
                self.best_params[param] = int(self.global_best_position[i])
            else:
                self.best_params[param] = float(self.global_best_position[i])
        
        self.best_score = self.global_best_score
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best Score: {self.best_score:.4f}")
            print(f"Best Parameters: {self.best_params}")
        
        return self.best_params
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with optimization history
        """
        return pd.DataFrame(self.history)
