import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class TabDDPMGenerator:
    """
    TabDDPM sample generator for synthetic data generation.
    """
    
    def __init__(self, trainer):
        """
        Initialize TabDDPM Generator.
        
        Args:
            trainer: Trained TabDDPMTrainer instance
        """
        self.trainer = trainer
        self.device = trainer.device
        self.model = trainer.model
        self.num_timesteps = trainer.num_timesteps
        self.feature_means = trainer.feature_means
        self.feature_stds = trainer.feature_stds
    
    def _p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion process (p) - single step.
        
        Args:
            x: Noisy data
            t: Timestep
            
        Returns:
            Less noisy data
        """
        schedule = self.trainer._get_noise_schedule()
        
        # Predict noise
        predicted_noise = self.model(x, t)
        
        # Get schedule values for timestep t
        alpha_t = schedule['alphas'][t].unsqueeze(-1)
        alpha_cumprod_t = schedule['alphas_cumprod'][t].unsqueeze(-1)
        beta_t = schedule['betas'][t].unsqueeze(-1)
        
        # Compute mean
        mean = (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) / torch.sqrt(alpha_t)
        
        # Add noise if not final step
        if t[0] > 0:
            noise = torch.randn_like(x)
            variance = beta_t
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean
        
        return x
    
    @torch.no_grad()
    def sample(self, num_samples: int, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            feature_names: List of feature names for the DataFrame
            
        Returns:
            DataFrame with synthetic samples
        """
        self.model.eval()
        
        # Start from random noise
        shape = (num_samples, len(self.feature_means))
        x = torch.randn(shape, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            x = self._p_sample(x, t_tensor)
        
        # Denormalize
        x_numpy = x.cpu().numpy()
        x_denormalized = x_numpy * self.feature_stds + self.feature_means
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_means))]
        
        synthetic_df = pd.DataFrame(x_denormalized, columns=feature_names)
        
        return synthetic_df
    
    def generate_balanced_dataset(self, 
                                   df: pd.DataFrame, 
                                   target_column: str,
                                   minority_class: int = 1) -> pd.DataFrame:
        """
        Generate synthetic samples to balance the dataset.
        
        Args:
            df: Original imbalanced DataFrame
            target_column: Name of target column
            minority_class: Value of the minority class
            
        Returns:
            Balanced DataFrame
        """
        # Separate classes
        df_majority = df[df[target_column] != minority_class]
        df_minority = df[df[target_column] == minority_class]
        
        # Calculate number of samples needed
        samples_needed = len(df_majority) - len(df_minority)
        
        if samples_needed <= 0:
            print("Dataset is already balanced or minority class is majority.")
            return df
        
        print(f"Generating {samples_needed} synthetic samples for minority class...")
        
        # Get feature names (excluding target)
        feature_names = [col for col in df.columns if col != target_column]
        
        # Train on minority class
        minority_features = df_minority[feature_names].values
        self.trainer.fit(df_minority, target_column)
        
        # Generate synthetic samples
        synthetic_samples = self.sample(samples_needed, feature_names)
        
        # Add target column
        synthetic_samples[target_column] = minority_class
        
        # Combine with original data
        balanced_df = pd.concat([df, synthetic_samples], ignore_index=True)
        
        print(f"Balanced dataset: {len(balanced_df)} samples")
        print(f"Class distribution:\n{balanced_df[target_column].value_counts()}")
        
        return balanced_df
    
    def compare_distributions(self, 
                             real_data: pd.DataFrame, 
                             synthetic_data: pd.DataFrame,
                             save_path: Optional[str] = None):
        """
        Compare distributions between real and synthetic data.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            save_path: Optional path to save the plots
        """
        # Get common columns
        common_cols = [col for col in real_data.columns if col in synthetic_data.columns]
        
        # Create subplots
        n_cols = min(4, len(common_cols))
        n_rows = (len(common_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(common_cols):
            ax = axes[idx]
            
            # Plot real data distribution
            ax.hist(real_data[col].dropna(), bins=30, alpha=0.5, label='Real', density=True)
            
            # Plot synthetic data distribution
            ax.hist(synthetic_data[col].dropna(), bins=30, alpha=0.5, label='Synthetic', density=True)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution Comparison: {col}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(common_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_correlations(self,
                           real_data: pd.DataFrame,
                           synthetic_data: pd.DataFrame,
                           save_path: Optional[str] = None):
        """
        Compare correlation matrices between real and synthetic data.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            save_path: Optional path to save the plots
        """
        # Get common numerical columns
        common_cols = [col for col in real_data.select_dtypes(include=[np.number]).columns 
                      if col in synthetic_data.columns]
        
        real_corr = real_data[common_cols].corr()
        synthetic_corr = synthetic_data[common_cols].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real data correlation
        sns.heatmap(real_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Real Data Correlation Matrix')
        
        # Synthetic data correlation
        sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1])
        axes[1].set_title('Synthetic Data Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Calculate correlation difference
        corr_diff = np.abs(real_corr - synthetic_corr).mean().mean()
        print(f"Mean absolute correlation difference: {corr_diff:.4f}")
