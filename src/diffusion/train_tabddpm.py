import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
import os
import joblib


class TabularDataset(Dataset):
    """
    Custom Dataset for tabular data.
    """
    
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class TabDDPMModel(nn.Module):
    """
    Simplified TabDDPM model for tabular data generation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super(TabDDPMModel, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim))  # +1 for timestep
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input data
            t: Timestep
            
        Returns:
            Predicted noise
        """
        # Concatenate data with timestep (add timestep as a single feature)
        t = t.unsqueeze(-1)  # Shape: (batch_size, 1)
        x_t = torch.cat([x, t], dim=-1)  # Shape: (batch_size, input_dim + 1)
        
        return self.network(x_t)


class TabDDPMTrainer:
    """
    TabDDPM trainer for synthetic data generation.
    """
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 100):
        """
        Initialize TabDDPM Trainer.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            hidden_dim: Hidden dimension for neural network
            num_layers: Number of hidden layers
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Diffusion schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.feature_means = None
        self.feature_stds = None
    
    def _get_noise_schedule(self) -> Dict[str, torch.Tensor]:
        """
        Get noise schedule for diffusion process.
        
        Returns:
            Dictionary with noise schedule parameters
        """
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(self.alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - self.alphas_cumprod)
        }
    
    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process (q).
        
        Args:
            x_start: Starting data
            t: Timestep
            noise: Optional noise tensor
            
        Returns:
            Noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        schedule = self._get_noise_schedule()
        
        sqrt_alphas_cumprod_t = schedule['sqrt_alphas_cumprod'][t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = schedule['sqrt_one_minus_alphas_cumprod'][t].unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _p_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for training.
        
        Args:
            model: Diffusion model
            x_start: Starting data
            t: Timestep
            
        Returns:
            Loss
        """
        noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start, t, noise)
        
        predicted_noise = model(x_noisy, t)
        
        loss = nn.MSELoss()(noise, predicted_noise)
        return loss
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'TabDDPMTrainer':
        """
        Train the TabDDPM model.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (to exclude from training)
            
        Returns:
            self: Trained TabDDPMTrainer instance
        """
        df = df.copy()
        
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        # Convert to numpy
        data = df.values.astype(np.float32)
        
        # Normalize data
        self.feature_means = np.mean(data, axis=0)
        self.feature_stds = np.std(data, axis=0) + 1e-8
        data_normalized = (data - self.feature_means) / self.feature_stds
        
        # Create dataset and dataloader
        dataset = TabularDataset(data_normalized)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = data.shape[1]
        self.model = TabDDPMModel(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        print(f"Training TabDDPM on {self.device}...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (batch.shape[0],), device=self.device).long()
                
                # Calculate loss
                loss = self._p_losses(self.model, batch, t)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        return self
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'num_timesteps': self.num_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'TabDDPMTrainer':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self: TabDDPMTrainer instance with loaded model
        """
        model_data = torch.load(filepath, map_location=self.device)
        
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.num_timesteps = model_data['num_timesteps']
        self.beta_start = model_data['beta_start']
        self.beta_end = model_data['beta_end']
        self.hidden_dim = model_data['hidden_dim']
        self.num_layers = model_data['num_layers']
        
        # Rebuild noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Initialize model and load state
        input_dim = len(self.feature_means)
        self.model = TabDDPMModel(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return self
