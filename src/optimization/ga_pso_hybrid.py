import pandas as pd
from typing import Dict, Optional
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .particle_swarm import ParticleSwarmOptimizer


class GAPSOHybridOptimizer:
    """
    Hybrid GA-PSO optimizer for XGBoost hyperparameter optimization.
    
    Uses Genetic Algorithm for global exploration and Particle Swarm Optimization
    for local refinement around the best GA solutions.
    """
    
    def __init__(self,
                 # GA parameters
                 ga_population_size: int = 50,
                 ga_generations: int = 30,
                 ga_tournament_size: int = 3,
                 ga_crossover_prob: float = 0.8,
                 ga_mutation_prob: float = 0.2,
                 # PSO parameters
                 pso_n_particles: int = 30,
                 pso_iterations: int = 50,
                 pso_w: float = 0.7,
                 pso_c1: float = 1.5,
                 pso_c2: float = 1.5,
                 # Common parameters
                 random_state: int = 42):
        """
        Initialize GA-PSO Hybrid Optimizer.
        
        Args:
            ga_population_size: GA population size
            ga_generations: GA number of generations
            ga_tournament_size: GA tournament size
            ga_crossover_prob: GA crossover probability
            ga_mutation_prob: GA mutation probability
            pso_n_particles: PSO number of particles
            pso_iterations: PSO number of iterations
            pso_w: PSO inertia weight
            pso_c1: PSO cognitive coefficient
            pso_c2: PSO social coefficient
            random_state: Random state for reproducibility
        """
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            population_size=ga_population_size,
            generations=ga_generations,
            tournament_size=ga_tournament_size,
            crossover_prob=ga_crossover_prob,
            mutation_prob=ga_mutation_prob,
            random_state=random_state
        )
        
        self.pso_optimizer = ParticleSwarmOptimizer(
            n_particles=pso_n_particles,
            iterations=pso_iterations,
            w=pso_w,
            c1=pso_c1,
            c2=pso_c2,
            random_state=random_state
        )
        
        self.best_params = None
        self.best_score = None
        self.ga_params = None
        self.pso_params = None
    
    def optimize(self, 
                X: pd.DataFrame, 
                y: pd.Series, 
                cv: int = 5, 
                verbose: bool = True) -> Dict:
        """
        Optimize XGBoost hyperparameters using hybrid GA-PSO approach.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            verbose: Whether to print progress
            
        Returns:
            Best parameter dictionary
        """
        if verbose:
            print("=" * 60)
            print("GA-PSO Hybrid Optimization")
            print("=" * 60)
            print("\nPhase 1: Genetic Algorithm (Global Exploration)")
            print("-" * 60)
        
        # Phase 1: GA for global exploration
        self.ga_params = self.ga_optimizer.optimize(X, y, cv, verbose)
        
        if verbose:
            print("\nPhase 2: Particle Swarm Optimization (Local Refinement)")
            print("-" * 60)
        
        # Phase 2: PSO for local refinement around GA best
        self.pso_params = self.pso_optimizer.optimize(X, y, cv, self.ga_params, verbose)
        
        # Compare results and select best
        if self.ga_optimizer.best_score >= self.pso_optimizer.best_score:
            self.best_params = self.ga_params
            self.best_score = self.ga_optimizer.best_score
            if verbose:
                print("\nGA parameters selected as final parameters.")
        else:
            self.best_params = self.pso_params
            self.best_score = self.pso_optimizer.best_score
            if verbose:
                print("\nPSO parameters selected as final parameters.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("Hybrid Optimization Completed")
            print("=" * 60)
            print(f"Final Best Score: {self.best_score:.4f}")
            print(f"Final Best Parameters: {self.best_params}")
            print(f"GA Score: {self.ga_optimizer.best_score:.4f}")
            print(f"PSO Score: {self.pso_optimizer.best_score:.4f}")
        
        return self.best_params
    
    def get_optimization_history(self) -> Dict[str, pd.DataFrame]:
        """
        Get optimization history for both GA and PSO.
        
        Returns:
            Dictionary with GA and PSO history DataFrames
        """
        return {
            'ga_history': self.ga_optimizer.get_optimization_history(),
            'pso_history': self.pso_optimizer.get_optimization_history()
        }
    
    def get_comparison_report(self) -> Dict:
        """
        Get comparison report between GA and PSO results.
        
        Returns:
            Dictionary with comparison metrics
        """
        return {
            'ga_params': self.ga_params,
            'ga_score': self.ga_optimizer.best_score,
            'pso_params': self.pso_params,
            'pso_score': self.pso_optimizer.best_score,
            'final_params': self.best_params,
            'final_score': self.best_score,
            'improvement': self.pso_optimizer.best_score - self.ga_optimizer.best_score
        }
