"""
Flow Dynamics Module
===================

Flow Matching and biological dynamics modeling for cancer analysis including
FlowMatchingModule, CouplingLayer, and biological dynamics modeling.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import from config module
try:
    from config import *
except ImportError:
    # Fallback definitions if config module not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching models"""
    sigma_min: float = 1e-4
    sigma_max: float = 20.0
    time_steps: int = 1000
    ode_method: str = 'euler'
    adaptive_step: bool = True
    tolerance: float = 1e-5
    
@dataclass
class BiologicalParameters:
    """Biological parameters for tumor dynamics"""
    growth_rate: float = 0.1  # per day
    carrying_capacity: float = 1.0  # normalized
    invasion_rate: float = 0.05  # per day
    apoptosis_rate: float = 0.02  # per day
    angiogenesis_rate: float = 0.03  # per day
    hypoxia_threshold: float = 0.3  # oxygen level
    immune_response: float = 0.1  # immune killing rate
    drug_efficacy: float = 0.0  # treatment effect
    mutation_rate: float = 1e-6  # per cell division
    stemness_factor: float = 0.05  # cancer stem cell fraction

@dataclass
class DynamicsResult:
    """Results from dynamics simulation"""
    time_points: np.ndarray
    tumor_mass: np.ndarray
    invasion_radius: np.ndarray
    oxygen_levels: np.ndarray
    drug_concentration: np.ndarray
    immune_cells: np.ndarray
    apoptotic_cells: np.ndarray
    proliferating_cells: np.ndarray
    biological_parameters: BiologicalParameters
    simulation_metadata: Dict

class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, split_dim: Optional[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.split_dim = split_dim or input_dim // 2
        self.transform_dim = input_dim - self.split_dim
        
        # Translation and scaling networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.transform_dim),
            nn.Tanh()  # Keep scaling bounded
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.transform_dim)
        )
        
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward or reverse coupling transformation"""
        
        x1, x2 = x.split([self.split_dim, self.transform_dim], dim=1)
        
        if not reverse:
            # Forward transformation
            scale = self.scale_net(x1)
            translate = self.translate_net(x1)
            
            # Apply affine transformation
            y2 = x2 * torch.exp(scale) + translate
            y = torch.cat([x1, y2], dim=1)
            
            # Log determinant of Jacobian
            log_det = scale.sum(dim=1)
            
        else:
            # Reverse transformation
            scale = self.scale_net(x1)
            translate = self.translate_net(x1)
            
            # Reverse affine transformation
            y2 = (x2 - translate) * torch.exp(-scale)
            y = torch.cat([x1, y2], dim=1)
            
            # Log determinant of Jacobian (negative for reverse)
            log_det = -scale.sum(dim=1)
        
        return y, log_det

class FlowMatchingModule(nn.Module):
    """Flow matching module for continuous normalizing flows"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 time_embedding_dim: int = 128,
                 config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.config = config or FlowMatchingConfig()
        
        # Time embedding network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU()
        )
        
        # Vector field network
        self.vector_field = self._build_vector_field_network()
        
        # Coupling layers for additional flow complexity
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
    def _build_vector_field_network(self) -> nn.Module:
        """Build the vector field network"""
        
        layers = []
        input_size = self.input_dim + self.time_embedding_dim
        
        # Input layer
        layers.extend([
            nn.Linear(input_size, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim)
        ])
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(0.1)
            ])
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute vector field at time t"""
        
        # Embed time
        t_embedded = self.time_embedding(t.unsqueeze(-1))
        
        # Broadcast time embedding to match batch size
        if t_embedded.dim() == 2 and x.dim() == 2:
            if t_embedded.size(0) == 1 and x.size(0) > 1:
                t_embedded = t_embedded.expand(x.size(0), -1)
        
        # Concatenate input and time embedding
        xt = torch.cat([x, t_embedded], dim=-1)
        
        # Compute vector field
        vector_field = self.vector_field(xt)
        
        return vector_field
    
    def sample_path(self, x0: torch.Tensor, x1: torch.Tensor, 
                   num_steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a flow matching path from x0 to x1"""
        
        batch_size = x0.size(0)
        device = x0.device
        
        # Time points
        t_span = torch.linspace(0, 1, num_steps, device=device)
        
        # Initialize path
        path = torch.zeros(num_steps, batch_size, self.input_dim, device=device)
        path[0] = x0
        
        # Integrate ODE
        dt = 1.0 / (num_steps - 1)
        
        for i in range(1, num_steps):
            t = t_span[i-1]
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Compute vector field
            v_t = self.forward(path[i-1], t_tensor)
            
            # Euler step
            path[i] = path[i-1] + dt * v_t
        
        return path, t_span
    
    def flow_matching_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss"""
        
        batch_size = x0.size(0)
        device = x0.device
        
        # Sample random times
        t = torch.rand(batch_size, device=device)
        
        # Interpolate between x0 and x1
        sigma_t = self.config.sigma_min + t * (self.config.sigma_max - self.config.sigma_min)
        noise = torch.randn_like(x0)
        
        # Flow matching interpolation
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1 + sigma_t.unsqueeze(-1) * noise
        
        # Target vector field (conditional flow matching)
        target_v = x1 - x0 + sigma_t.unsqueeze(-1) * noise
        
        # Predicted vector field
        pred_v = self.forward(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

class LogisticTumorDynamics(nn.Module):
    """Logistic tumor growth dynamics with biological constraints"""
    
    def __init__(self, spatial_dim: int = 2, num_species: int = 4):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.num_species = num_species  # tumor, immune, oxygen, drug
        
        # Learnable biological parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        self.carrying_capacity = nn.Parameter(torch.tensor(1.0))
        self.invasion_rate = nn.Parameter(torch.tensor(0.05))
        self.immune_kill_rate = nn.Parameter(torch.tensor(0.1))
        self.oxygen_consumption = nn.Parameter(torch.tensor(0.02))
        self.drug_efficacy = nn.Parameter(torch.tensor(0.0))
        
        # Spatial interaction network
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(num_species, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_species, kernel_size=1)
        )
        
    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute derivatives for logistic tumor dynamics"""
        
        # Extract species concentrations
        tumor = state[:, 0:1]  # Tumor cell density
        immune = state[:, 1:2]  # Immune cell density
        oxygen = state[:, 2:3]  # Oxygen concentration
        drug = state[:, 3:4]   # Drug concentration
        
        # Logistic growth with carrying capacity
        growth_term = self.growth_rate * tumor * (1 - tumor / self.carrying_capacity)
        
        # Immune-mediated killing
        immune_kill = self.immune_kill_rate * immune * tumor / (tumor + 0.1)
        
        # Drug-induced death
        drug_kill = self.drug_efficacy * drug * tumor
        
        # Oxygen consumption
        oxygen_consumption = self.oxygen_consumption * tumor
        
        # Spatial interactions
        spatial_effects = self.spatial_interaction(state)
        
        # Tumor dynamics
        dtumor_dt = growth_term - immune_kill - drug_kill + spatial_effects[:, 0:1]
        
        # Immune dynamics (simplified)
        dimmune_dt = 0.05 * tumor - 0.1 * immune + spatial_effects[:, 1:2]
        
        # Oxygen dynamics
        doxygen_dt = 0.1 - oxygen_consumption - 0.05 * oxygen + spatial_effects[:, 2:3]
        
        # Drug dynamics (decay)
        ddrug_dt = -0.1 * drug + spatial_effects[:, 3:4]
        
        # Combine derivatives
        derivatives = torch.cat([dtumor_dt, dimmune_dt, doxygen_dt, ddrug_dt], dim=1)
        
        return derivatives

class FlowMatchingLogisticModel(nn.Module):
    """Combined flow matching and logistic dynamics model"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 flow_layers: int = 3,
                 spatial_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        
        # Flow matching component
        self.flow_matching = FlowMatchingModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=flow_layers
        )
        
        # Logistic dynamics component
        self.logistic_dynamics = LogisticTumorDynamics(
            spatial_dim=spatial_dim,
            num_species=4
        )
        
        # Feature encoder for biological parameters
        self.parameter_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)  # 10 biological parameters
        )
        
        # Spatial feature processor
        self.spatial_processor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, input_dim)
        )
        
    def encode_biological_parameters(self, features: torch.Tensor) -> BiologicalParameters:
        """Encode features into biological parameters"""
        
        params = self.parameter_encoder(features)
        
        # Apply constraints to ensure biological plausibility
        growth_rate = torch.sigmoid(params[:, 0]) * 0.3  # 0-0.3 per day
        carrying_capacity = torch.sigmoid(params[:, 1]) * 2.0  # 0-2.0
        invasion_rate = torch.sigmoid(params[:, 2]) * 0.1  # 0-0.1 per day
        apoptosis_rate = torch.sigmoid(params[:, 3]) * 0.05  # 0-0.05 per day
        angiogenesis_rate = torch.sigmoid(params[:, 4]) * 0.05  # 0-0.05 per day
        hypoxia_threshold = torch.sigmoid(params[:, 5]) * 0.5  # 0-0.5
        immune_response = torch.sigmoid(params[:, 6]) * 0.2  # 0-0.2 per day
        drug_efficacy = torch.sigmoid(params[:, 7]) * 1.0  # 0-1.0
        mutation_rate = torch.sigmoid(params[:, 8]) * 1e-5  # 0-1e-5
        stemness_factor = torch.sigmoid(params[:, 9]) * 0.1  # 0-0.1
        
        # Return mean values for single parameter set
        return BiologicalParameters(
            growth_rate=float(growth_rate.mean().item()),
            carrying_capacity=float(carrying_capacity.mean().item()),
            invasion_rate=float(invasion_rate.mean().item()),
            apoptosis_rate=float(apoptosis_rate.mean().item()),
            angiogenesis_rate=float(angiogenesis_rate.mean().item()),
            hypoxia_threshold=float(hypoxia_threshold.mean().item()),
            immune_response=float(immune_response.mean().item()),
            drug_efficacy=float(drug_efficacy.mean().item()),
            mutation_rate=float(mutation_rate.mean().item()),
            stemness_factor=float(stemness_factor.mean().item())
        )
    
    def simulate_tumor_dynamics(self, 
                               initial_state: torch.Tensor,
                               biological_params: BiologicalParameters,
                               time_span: Tuple[float, float] = (0, 100),
                               num_points: int = 100) -> DynamicsResult:
        """Simulate tumor dynamics using logistic growth model"""
        
        def dynamics_ode(t, y):
            """ODE system for tumor dynamics"""
            
            # Extract state variables
            tumor_mass = y[0]
            invasion_radius = y[1] 
            oxygen_level = y[2]
            drug_conc = y[3]
            immune_cells = y[4]
            
            # Logistic growth with carrying capacity
            growth_rate = biological_params.growth_rate
            carrying_capacity = biological_params.carrying_capacity
            
            # Tumor growth
            dtumor_dt = growth_rate * tumor_mass * (1 - tumor_mass / carrying_capacity)
            
            # Invasion dynamics
            dinvasion_dt = biological_params.invasion_rate * tumor_mass * (1 - invasion_radius)
            
            # Oxygen consumption and delivery
            doxygen_dt = (biological_params.angiogenesis_rate * 
                         (biological_params.hypoxia_threshold - oxygen_level) - 
                         0.01 * tumor_mass * oxygen_level)
            
            # Drug clearance
            ddrug_dt = -0.1 * drug_conc  # First-order elimination
            
            # Immune response
            dimmune_dt = biological_params.immune_response * tumor_mass - 0.1 * immune_cells
            
            # Apply treatment effects
            if drug_conc > 0.1:
                dtumor_dt -= biological_params.drug_efficacy * drug_conc * tumor_mass
            
            # Apply immune killing
            dtumor_dt -= biological_params.immune_response * immune_cells * tumor_mass / (tumor_mass + 0.1)
            
            # Ensure non-negative values
            dtumor_dt = max(dtumor_dt, -tumor_mass)
            
            return [dtumor_dt, dinvasion_dt, doxygen_dt, ddrug_dt, dimmune_dt]
        
        # Initial conditions
        y0 = initial_state.cpu().numpy().flatten()[:5]  # Take first 5 components
        
        # Solve ODE
        t_eval = np.linspace(time_span[0], time_span[1], num_points)
        solution = solve_ivp(dynamics_ode, time_span, y0, t_eval=t_eval, method='RK45')
        
        # Extract results
        tumor_mass = solution.y[0]
        invasion_radius = solution.y[1]
        oxygen_levels = solution.y[2]
        drug_concentration = solution.y[3]
        immune_cells = solution.y[4]
        
        # Calculate derived quantities
        apoptotic_cells = biological_params.apoptosis_rate * tumor_mass
        proliferating_cells = biological_params.growth_rate * tumor_mass * (1 - tumor_mass / biological_params.carrying_capacity)
        
        return DynamicsResult(
            time_points=solution.t,
            tumor_mass=tumor_mass,
            invasion_radius=invasion_radius,
            oxygen_levels=oxygen_levels,
            drug_concentration=drug_concentration,
            immune_cells=immune_cells,
            apoptotic_cells=apoptotic_cells,
            proliferating_cells=proliferating_cells,
            biological_parameters=biological_params,
            simulation_metadata={
                'solver': 'RK45',
                'time_span': time_span,
                'num_points': num_points,
                'success': solution.success
            }
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                spatial_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass combining flow matching and logistic dynamics"""
        
        # Flow matching component
        flow_vector = self.flow_matching(x, t)
        
        # Encode biological parameters
        bio_params_encoded = self.parameter_encoder(x)
        
        # Process spatial data if available
        if spatial_data is not None:
            spatial_features = self.spatial_processor(spatial_data)
            x = x + spatial_features  # Residual connection
        
        # Prepare state for logistic dynamics
        if spatial_data is not None:
            # Use spatial data for dynamics
            spatial_batch = spatial_data.unsqueeze(1) if spatial_data.dim() == 3 else spatial_data
            if spatial_batch.size(1) < 4:
                # Pad to 4 channels if needed
                padding = torch.zeros(spatial_batch.size(0), 4 - spatial_batch.size(1), 
                                    spatial_batch.size(2), spatial_batch.size(3), 
                                    device=spatial_data.device)
                spatial_batch = torch.cat([spatial_batch, padding], dim=1)
            
            dynamics_vector = self.logistic_dynamics(spatial_batch, t)
        else:
            # Create dummy spatial data
            batch_size = x.size(0)
            dummy_spatial = torch.zeros(batch_size, 4, 8, 8, device=x.device)
            dynamics_vector = self.logistic_dynamics(dummy_spatial, t)
            dynamics_vector = torch.mean(dynamics_vector.view(batch_size, -1), dim=1, keepdim=True)
            dynamics_vector = dynamics_vector.expand(-1, x.size(1))
        
        # Combine flow and dynamics
        combined_vector = flow_vector + 0.1 * dynamics_vector.view(flow_vector.shape)
        
        return {
            'flow_vector': flow_vector,
            'dynamics_vector': dynamics_vector,
            'combined_vector': combined_vector,
            'biological_parameters': bio_params_encoded
        }

class FlowMatchingTrainer:
    """Trainer for flow matching models"""
    
    def __init__(self, model: FlowMatchingLogisticModel, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        # Training history
        self.training_history = {
            'losses': [],
            'flow_losses': [],
            'dynamics_losses': [],
            'epochs': []
        }
    
    def train_step(self, x0: torch.Tensor, x1: torch.Tensor,
                  spatial_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = x0.size(0)
        
        # Sample random time points
        t = torch.rand(batch_size, device=self.device)
        
        # Flow matching loss
        flow_loss = self.model.flow_matching.flow_matching_loss(x0, x1)
        
        # Forward pass for dynamics
        output = self.model(x0, t, spatial_data)
        
        # Dynamics consistency loss
        if spatial_data is not None:
            # Encourage dynamics to be physically reasonable
            dynamics_loss = F.mse_loss(
                output['dynamics_vector'],
                torch.zeros_like(output['dynamics_vector'])
            ) * 0.1
        else:
            dynamics_loss = torch.tensor(0.0, device=self.device)
        
        # Biological parameter regularization
        bio_params = output['biological_parameters']
        param_reg = torch.mean(bio_params ** 2) * 1e-4
        
        # Total loss
        total_loss = flow_loss + dynamics_loss + param_reg
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'flow_loss': flow_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'param_reg': param_reg.item()
        }
    
    def train(self, train_loader, num_epochs: int = 100, 
              val_loader=None, save_path: Optional[str] = None):
        """Train the model"""
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)):
                    x0, x1 = batch[0].to(self.device), batch[1].to(self.device)
                    spatial_data = batch[2].to(self.device) if len(batch) > 2 else None
                else:
                    # Single tensor case - create target as noisy version
                    x0 = torch.randn_like(batch).to(self.device)
                    x1 = batch.to(self.device)
                    spatial_data = None
                
                losses = self.train_step(x0, x1, spatial_data)
                epoch_losses.append(losses['total_loss'])
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {losses['total_loss']:.6f}")
            
            # Average epoch loss
            avg_loss = np.mean(epoch_losses)
            self.training_history['losses'].append(avg_loss)
            self.training_history['epochs'].append(epoch)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
            else:
                logger.info(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}")
            
            # Update scheduler
            self.scheduler.step()
    
    def validate(self, val_loader) -> float:
        """Validate the model"""
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x0, x1 = batch[0].to(self.device), batch[1].to(self.device)
                    spatial_data = batch[2].to(self.device) if len(batch) > 2 else None
                else:
                    x0 = torch.randn_like(batch).to(self.device)
                    x1 = batch.to(self.device)
                    spatial_data = None
                
                flow_loss = self.model.flow_matching.flow_matching_loss(x0, x1)
                val_losses.append(flow_loss.item())
        
        return np.mean(val_losses)

def create_sample_flow_data(batch_size: int = 32, input_dim: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample data for flow matching training"""
    
    # Source distribution (noise)
    x0 = torch.randn(batch_size, input_dim)
    
    # Target distribution (structured data)
    # Create multimodal target with some structure
    targets = []
    for i in range(batch_size):
        # Create different modes
        mode = i % 3
        if mode == 0:
            # Mode 1: Gaussian cluster
            target = torch.randn(input_dim) * 0.5 + 1.0
        elif mode == 1:
            # Mode 2: Different Gaussian cluster
            target = torch.randn(input_dim) * 0.3 - 1.0
        else:
            # Mode 3: Structured pattern
            target = torch.sin(torch.linspace(0, 2*np.pi, input_dim)) + torch.randn(input_dim) * 0.2
        
        targets.append(target)
    
    x1 = torch.stack(targets)
    
    return x0, x1

def run_flow_dynamics_demo():
    """Run a demonstration of flow dynamics analysis"""
    
    print("=" * 60)
    print("FLOW DYNAMICS ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    input_dim = 64
    batch_size = 16
    hidden_dim = 128
    
    # Create model
    model = FlowMatchingLogisticModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        flow_layers=3,
        spatial_dim=2
    )
    
    print("MODEL ARCHITECTURE:")
    print(f"Input Dimension: {input_dim}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Flow Matching Layers: 3")
    print(f"Logistic Dynamics: Enabled")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create sample data
    x0, x1 = create_sample_flow_data(batch_size, input_dim)
    
    # Sample spatial data
    spatial_data = torch.randn(batch_size, 1, 16, 16)  # Sample spatial patterns
    
    print("SAMPLE DATA:")
    print(f"Source (x0) shape: {x0.shape}")
    print(f"Target (x1) shape: {x1.shape}")
    print(f"Spatial data shape: {spatial_data.shape}")
    print()
    
    # Forward pass
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        output = model(x0, t, spatial_data)
        bio_params = model.encode_biological_parameters(x0)
    
    print("FORWARD PASS RESULTS:")
    print(f"Flow vector shape: {output['flow_vector'].shape}")
    print(f"Dynamics vector shape: {output['dynamics_vector'].shape}")
    print(f"Combined vector shape: {output['combined_vector'].shape}")
    print(f"Biological parameters shape: {output['biological_parameters'].shape}")
    print()
    
    print("BIOLOGICAL PARAMETERS:")
    print(f"Growth Rate: {bio_params.growth_rate:.4f} per day")
    print(f"Carrying Capacity: {bio_params.carrying_capacity:.4f}")
    print(f"Invasion Rate: {bio_params.invasion_rate:.4f} per day")
    print(f"Apoptosis Rate: {bio_params.apoptosis_rate:.4f} per day")
    print(f"Angiogenesis Rate: {bio_params.angiogenesis_rate:.4f} per day")
    print(f"Hypoxia Threshold: {bio_params.hypoxia_threshold:.4f}")
    print(f"Immune Response: {bio_params.immune_response:.4f} per day")
    print(f"Drug Efficacy: {bio_params.drug_efficacy:.4f}")
    print(f"Mutation Rate: {bio_params.mutation_rate:.2e}")
    print(f"Stemness Factor: {bio_params.stemness_factor:.4f}")
    print()
    
    # Simulate tumor dynamics
    initial_state = torch.tensor([0.1, 0.05, 0.8, 0.0, 0.1])  # Initial conditions
    
    dynamics_result = model.simulate_tumor_dynamics(
        initial_state=initial_state,
        biological_params=bio_params,
        time_span=(0, 100),
        num_points=100
    )
    
    print("TUMOR DYNAMICS SIMULATION:")
    print(f"Simulation Time: {dynamics_result.time_points[-1]:.1f} days")
    print(f"Final Tumor Mass: {dynamics_result.tumor_mass[-1]:.4f}")
    print(f"Final Invasion Radius: {dynamics_result.invasion_radius[-1]:.4f}")
    print(f"Final Oxygen Level: {dynamics_result.oxygen_levels[-1]:.4f}")
    print(f"Final Immune Cells: {dynamics_result.immune_cells[-1]:.4f}")
    print(f"Simulation Success: {dynamics_result.simulation_metadata['success']}")
    print()
    
    # Flow matching path
    with torch.no_grad():
        sample_x0 = x0[:4]  # Take first 4 samples
        sample_x1 = x1[:4]
        
        path, t_span = model.flow_matching.sample_path(
            sample_x0, sample_x1, num_steps=50
        )
    
    print("FLOW MATCHING PATH:")
    print(f"Path shape: {path.shape}")
    print(f"Time span: [{t_span[0]:.2f}, {t_span[-1]:.2f}]")
    print(f"Initial distance: {torch.norm(sample_x0 - sample_x1, dim=1).mean():.4f}")
    print(f"Final distance: {torch.norm(path[-1] - sample_x1, dim=1).mean():.4f}")
    print()
    
    # Training demonstration
    print("TRAINING DEMONSTRATION:")
    
    # Create simple dataset
    train_data = [(x0[i:i+4], x1[i:i+4], spatial_data[i:i+4]) for i in range(0, batch_size, 4)]
    
    # Initialize trainer
    trainer = FlowMatchingTrainer(model)
    
    # Single training step
    losses = trainer.train_step(x0[:8], x1[:8], spatial_data[:8])
    
    print(f"Training step completed:")
    print(f"  Total Loss: {losses['total_loss']:.6f}")
    print(f"  Flow Loss: {losses['flow_loss']:.6f}")
    print(f"  Dynamics Loss: {losses['dynamics_loss']:.6f}")
    print(f"  Parameter Regularization: {losses['param_reg']:.6f}")
    print()
    
    print("FLOW DYNAMICS DEMO COMPLETED SUCCESSFULLY!")
    
    return {
        'model': model,
        'biological_parameters': bio_params,
        'dynamics_result': dynamics_result,
        'flow_path': path,
        'training_losses': losses
    }

if __name__ == "__main__":
    # Run demonstration
    result = run_flow_dynamics_demo()
