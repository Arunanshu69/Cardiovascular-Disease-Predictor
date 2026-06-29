import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
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
        
        # Get schedule values for timestep t (ensure they're on the same device)
        alpha_t = schedule['alphas'][t].unsqueeze(-1).to(self.device)
        alpha_cumprod_t = schedule['alphas_cumprod'][t].unsqueeze(-1).to(self.device)
        beta_t = schedule['betas'][t].unsqueeze(-1).to(self.device)
        
        # Compute mean with epsilon to prevent division by zero
        mean = (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t + 1e-8)) * predicted_noise) / torch.sqrt(alpha_t + 1e-8)
        
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
        
        # Get minority class features for sampling reference
        minority_features = df_minority[feature_names].values
        
        # Generate synthetic samples using the already-trained model
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
        from scipy.stats import ks_2samp, wasserstein_distance
        
        # Set better style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Calculate statistical metrics
        metrics = {}
        for col in real_data.columns:
            if pd.api.types.is_numeric_dtype(real_data[col]) and col in synthetic_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
                
                # Wasserstein distance
                w_dist = wasserstein_distance(real_data[col].dropna(), synthetic_data[col].dropna())
                
                metrics[col] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'wasserstein_distance': w_dist
                }
        
        # Calculate mean absolute correlation difference
        real_corr = real_data.corr()
        synth_corr = synthetic_data.corr()
        if real_corr.shape == synth_corr.shape:
            mean_corr_diff = np.abs(real_corr - synth_corr).mean().mean()
            metrics['mean_correlation_difference'] = mean_corr_diff
            print(f"Mean absolute correlation difference: {mean_corr_diff:.4f}")
        
        # Print summary statistics
        print("Distribution similarity metrics:")
        for col, col_metrics in metrics.items():
            if col != 'mean_correlation_difference':
                print(f"  {col}: KS stat={col_metrics['ks_statistic']:.4f}, p-value={col_metrics['ks_pvalue']:.4f}, Wasserstein={col_metrics['wasserstein_distance']:.4f}")
        
        # Get common columns
        common_cols = [col for col in real_data.columns if col in synthetic_data.columns]
        
        # Create subplots with better sizing
        n_cols = min(3, len(common_cols))
        n_rows = (len(common_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(common_cols):
            ax = axes[idx]
            
            real_vals = real_data[col].dropna()
            synth_vals = synthetic_data[col].dropna()
            
            # Calculate optimal bins using Freedman-Diaconis rule
            def freedman_diaconis_bins(data):
                q25, q75 = np.percentile(data, [25, 75])
                iqr = q75 - q25
                n = len(data)
                bin_width = 2 * iqr / (n ** (1/3))
                if bin_width == 0:
                    bin_width = 1
                bins = int((data.max() - data.min()) / bin_width)
                return max(10, min(50, bins))
            
            bins_real = freedman_diaconis_bins(real_vals)
            bins_synth = freedman_diaconis_bins(synth_vals)
            bins = max(bins_real, bins_synth)
            
            # Plot histograms with better styling
            ax.hist(real_vals, bins=bins, alpha=0.6, label='Real Data', 
                   density=True, color='#2E86AB', edgecolor='black', linewidth=0.5)
            ax.hist(synth_vals, bins=bins, alpha=0.6, label='Synthetic Data', 
                   density=True, color='#F24236', edgecolor='black', linewidth=0.5)
            
            # Add KDE curves for smoother comparison
            from scipy.stats import gaussian_kde
            try:
                if len(real_vals) > 5 and real_vals.std() > 0:
                    kde_real = gaussian_kde(real_vals)
                    x_range = np.linspace(min(real_vals.min(), synth_vals.min()), 
                                         max(real_vals.max(), synth_vals.max()), 200)
                    ax.plot(x_range, kde_real(x_range), color='#2E86AB', linewidth=2, linestyle='-')
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip KDE if data has zero variance
            
            try:
                if len(synth_vals) > 5 and synth_vals.std() > 0:
                    kde_synth = gaussian_kde(synth_vals)
                    x_range = np.linspace(min(real_vals.min(), synth_vals.min()), 
                                         max(real_vals.max(), synth_vals.max()), 200)
                    ax.plot(x_range, kde_synth(x_range), color='#F24236', linewidth=2, linestyle='--')
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip KDE if data has zero variance
            
            # Add statistical metrics to the plot
            if col in metrics:
                ks_stat = metrics[col]['ks_statistic']
                w_dist = metrics[col]['wasserstein_distance']
                ax.text(0.98, 0.95, f'KS: {ks_stat:.3f}\nWD: {w_dist:.3f}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(col, fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Distribution Comparison: {col}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(common_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
