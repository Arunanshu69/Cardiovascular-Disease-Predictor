import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import random


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for XGBoost hyperparameter optimization.
    """
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 30,
                 tournament_size: int = 3,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 random_state: int = 42):
        """
        Initialize Genetic Algorithm Optimizer.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            tournament_size: Size of tournament for selection
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            random_state: Random state for reproducibility
        """
        # Validate tournament_size
        if tournament_size >= population_size:
            tournament_size = min(population_size - 1, 3)
            print(f"Warning: tournament_size >= population_size. Setting tournament_size to {tournament_size}")
        
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
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
    
    def _initialize_population(self) -> list:
        """
        Initialize random population.
        
        Returns:
            List of parameter dictionaries
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                if param in ['max_depth', 'n_estimators', 'min_child_weight']:
                    individual[param] = random.randint(min_val, max_val)
                else:
                    individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: Dict, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
        """
        Evaluate fitness of an individual using cross-validation.
        
        Args:
            individual: Parameter dictionary
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            ROC-AUC score
        """
        try:
            model = XGBClassifier(
                max_depth=int(individual['max_depth']),
                learning_rate=individual['learning_rate'],
                n_estimators=int(individual['n_estimators']),
                subsample=individual['subsample'],
                colsample_bytree=individual['colsample_bytree'],
                gamma=individual['gamma'],
                min_child_weight=int(individual['min_child_weight']),
                reg_alpha=individual.get('reg_alpha', 0.1),
                reg_lambda=individual.get('reg_lambda', 1.0),
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()
            
            # Handle NaN or invalid scores
            if np.isnan(mean_score) or np.isinf(mean_score):
                return 0.0
            
            return mean_score
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return 0.0
    
    def _tournament_selection(self, population: list, fitness_scores: list) -> Dict:
        """
        Select individual using tournament selection.
        
        Args:
            population: List of individuals
            fitness_scores: List of fitness scores
            
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Perform crossover between two parents with improved strategy.
        Uses blend crossover for continuous parameters and uniform crossover for discrete.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        child1, child2 = {}, {}
        
        for param in self.param_bounds.keys():
            if param in ['max_depth', 'n_estimators', 'min_child_weight']:
                # Uniform crossover for discrete parameters
                if random.random() < 0.5:
                    child1[param] = parent1[param]
                    child2[param] = parent2[param]
                else:
                    child1[param] = parent2[param]
                    child2[param] = parent1[param]
            else:
                # Blend crossover (arithmetic) for continuous parameters
                alpha = random.uniform(0, 1)
                child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict, generation: int = 0) -> Dict:
        """
        Perform mutation on an individual with adaptive mutation rate.
        Mutation rate decreases as generations progress to encourage convergence.
        
        Args:
            individual: Individual to mutate
            generation: Current generation number for adaptive mutation
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # Adaptive mutation rate: decreases as generations increase
        adaptive_mutation_prob = self.mutation_prob * (1 - generation / self.generations)
        adaptive_mutation_prob = max(adaptive_mutation_prob, 0.01)  # Minimum mutation rate
        
        for param, (min_val, max_val) in self.param_bounds.items():
            if random.random() < adaptive_mutation_prob:
                if param in ['max_depth', 'n_estimators', 'min_child_weight']:
                    mutated[param] = random.randint(min_val, max_val)
                else:
                    # Gaussian mutation for continuous parameters
                    sigma = (max_val - min_val) * 0.1  # 10% of range
                    mutated[param] = individual[param] + random.gauss(0, sigma)
                    # Clip to bounds
                    mutated[param] = max(min_val, min(max_val, mutated[param]))
        
        return mutated
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, verbose: bool = True) -> Dict:
        """
        Optimize XGBoost hyperparameters using genetic algorithm.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            verbose: Whether to print progress
            
        Returns:
            Best parameter dictionary
        """
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Initialize population
        population = self._initialize_population()
        
        if verbose:
            print(f"Starting Genetic Algorithm optimization...")
            print(f"Population size: {self.population_size}, Generations: {self.generations}")
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_fitness(individual, X, y, cv)
                fitness_scores.append(score)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            if self.best_score is None or fitness_scores[best_idx] > self.best_score:
                self.best_score = fitness_scores[best_idx]
                self.best_params = population[best_idx].copy()
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_score': self.best_score,
                'avg_score': np.mean(fitness_scores),
                'best_params': self.best_params.copy()
            })
            
            if verbose and (generation + 1) % 5 == 0:
                print(f"Generation {generation + 1}/{self.generations}")
                print(f"Best Score: {self.best_score:.4f}")
                print(f"Average Score: {np.mean(fitness_scores):.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(self.best_params.copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation with adaptive rate based on generation
                child1 = self._mutate(child1, generation)
                child2 = self._mutate(child2, generation)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
        
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
