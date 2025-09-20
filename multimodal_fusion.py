"""
Multimodal Fusion Module
=======================

Integration engine for multimodal cancer analysis including CrossModalAttention,
IntegratedCancerAnalysisModel, and feature fusion logic.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict
import math

# Import from config module
try:
    from config import *
except ImportError:
    # Fallback definitions if config module not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CANCER_TYPES_COUNT = 47

logger = logging.getLogger(__name__)

@dataclass
class MultimodalConfig:
    """Configuration for multimodal fusion models"""
    embed_dim: int = 256
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    num_transformer_layers: int = 6
    fusion_strategy: str = 'attention'  # 'attention', 'concatenation', 'gated'
    
@dataclass
class FusionWeights:
    """Learned fusion weights for different modalities"""
    wsi_weight: float
    genomics_weight: float
    clinical_weight: float
    methylation_weight: float
    confidence_score: float

@dataclass
class MultimodalPrediction:
    """Results from multimodal prediction"""
    patient_id: str
    cancer_type_prediction: Dict[str, float]
    stage_prediction: Dict[str, float]
    survival_prediction: Dict[str, float]
    treatment_response: Dict[str, float]
    biomarker_predictions: Dict[str, float]
    risk_scores: Dict[str, float]
    fusion_weights: FusionWeights
    confidence_scores: Dict[str, float]
    modality_contributions: Dict[str, Dict[str, float]]
    integrated_features: torch.Tensor
    attention_maps: Dict[str, torch.Tensor]
    prediction_timestamp: datetime

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for multimodal fusion"""
    
    def __init__(self, embed_dim: int, num_modalities: int, num_heads: int = 8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value networks for each modality
        self.query_nets = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])
        self.key_nets = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])
        self.value_nets = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])
        
        # Output projection for each modality
        self.output_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_modalities)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Scale factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, modal_features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-head cross-modal attention
        
        Args:
            modal_features: List of tensors [batch_size, embed_dim] for each modality
            
        Returns:
            attended_features: List of attended feature tensors
            attention_maps: Dictionary of attention weight tensors
        """
        
        batch_size = modal_features[0].size(0)
        
        # Generate queries, keys, and values for all modalities
        queries = []
        keys = []
        values = []
        
        for i, (feat, q_net, k_net, v_net) in enumerate(zip(modal_features, self.query_nets, self.key_nets, self.value_nets)):
            q = q_net(feat).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
            k = k_net(feat).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
            v = v_net(feat).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
            
            queries.append(q)
            keys.append(k)
            values.append(v)
        
        # Compute cross-modal attention for each modality
        attended_features = []
        attention_maps = {}
        
        for i in range(self.num_modalities):
            # Compute attention scores with all other modalities
            attention_scores = []
            attended_values = []
            
            for j in range(self.num_modalities):
                # Compute attention: Q_i * K_j^T
                scores = torch.matmul(queries[i], keys[j].transpose(-2, -1)) / self.scale  # [B, H, 1]
                attention_scores.append(scores)
                
                # Apply softmax to get attention weights
                attn_weights = F.softmax(scores, dim=-1)
                
                # Apply attention to values: Attention * V_j
                attended_value = torch.matmul(attn_weights, values[j].unsqueeze(-2)).squeeze(-2)  # [B, H, D]
                attended_values.append(attended_value)
            
            # Combine attended values from all modalities
            combined_attended = torch.stack(attended_values, dim=0).mean(dim=0)  # [B, H, D]
            combined_attended = combined_attended.view(batch_size, self.embed_dim)  # [B, E]
            
            # Apply output projection and residual connection
            output = self.output_projections[i](combined_attended)
            output = self.dropout(output)
            output = self.layer_norms[i](output + modal_features[i])  # Residual connection
            
            attended_features.append(output)
            
            # Store attention maps for interpretability
            attention_maps[f'modality_{i}'] = torch.stack(attention_scores, dim=1).mean(dim=2)  # [B, M, H]
        
        return attended_features, attention_maps

class ModalitySpecificEncoder(nn.Module):
    """Modality-specific feature encoder"""
    
    def __init__(self, input_dim: int, embed_dim: int, modality_type: str):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.modality_type = modality_type
        
        if modality_type == 'wsi':
            # WSI-specific processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim * 2),
                nn.Dropout(0.2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(0.1)
            )
            
        elif modality_type == 'genomics':
            # Genomics-specific processing with attention to mutation patterns
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.15),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            )
            
        elif modality_type == 'clinical':
            # Clinical data processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU()
            )
            
        elif modality_type == 'methylation':
            # Methylation-specific processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.Tanh(),  # Methylation values are bounded
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh()
            )
        else:
            # Generic encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1)
            )
        
        # Modality-specific output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through modality-specific encoder"""
        
        encoded = self.encoder(x)
        output = self.output_projection(encoded)
        
        return output

class GatedFusion(nn.Module):
    """Gated fusion mechanism for multimodal integration"""
    
    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
        # Feature transformation networks
        self.transform_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, modal_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated fusion
        
        Args:
            modal_features: List of feature tensors for each modality
            
        Returns:
            fused_features: Gated fusion of all modalities
            gate_weights: Learned gate weights for each modality
        """
        
        # Concatenate all modality features
        concatenated_features = torch.cat(modal_features, dim=1)
        
        # Compute gate weights
        gate_weights = self.gate_network(concatenated_features)
        
        # Transform each modality
        transformed_features = []
        for i, (feat, transform_net) in enumerate(zip(modal_features, self.transform_networks)):
            transformed = transform_net(feat)
            transformed_features.append(transformed)
        
        # Apply gating
        fused_features = torch.zeros_like(transformed_features[0])
        for i, transformed in enumerate(transformed_features):
            fused_features += gate_weights[:, i:i+1] * transformed
        
        return fused_features, gate_weights

class AdaptiveFusion(nn.Module):
    """Adaptive fusion mechanism that learns optimal fusion strategy"""
    
    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),  # 3 fusion strategies
            nn.Softmax(dim=1)
        )
        
        # Different fusion strategies
        self.cross_attention = CrossModalAttention(embed_dim, num_modalities)
        self.gated_fusion = GatedFusion(embed_dim, num_modalities)
        
        # Simple concatenation + projection
        self.concat_projection = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, modal_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with adaptive fusion strategy selection
        
        Args:
            modal_features: List of feature tensors for each modality
            
        Returns:
            fused_features: Adaptively fused features
            fusion_info: Information about fusion strategy and weights
        """
        
        # Concatenate features for strategy selection
        concatenated = torch.cat(modal_features, dim=1)
        strategy_weights = self.strategy_selector(concatenated)
        
        # Apply each fusion strategy
        # 1. Cross-modal attention
        attended_features, attention_maps = self.cross_attention(modal_features)
        attention_fused = torch.stack(attended_features).mean(dim=0)
        
        # 2. Gated fusion
        gated_fused, gate_weights = self.gated_fusion(modal_features)
        
        # 3. Concatenation fusion
        concat_fused = self.concat_projection(concatenated)
        
        # Combine strategies based on learned weights
        fused_features = (
            strategy_weights[:, 0:1] * attention_fused +
            strategy_weights[:, 1:2] * gated_fused +
            strategy_weights[:, 2:3] * concat_fused
        )
        
        fusion_info = {
            'strategy_weights': strategy_weights,
            'attention_maps': attention_maps,
            'gate_weights': gate_weights,
            'selected_strategy': torch.argmax(strategy_weights, dim=1)
        }
        
        return fused_features, fusion_info

class IntegratedCancerAnalysisModel(nn.Module):
    """Integrated multimodal cancer analysis model"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        
        # Modality-specific encoders
        self.wsi_encoder = ModalitySpecificEncoder(
            input_dim=512, embed_dim=config.embed_dim, modality_type='wsi'
        )
        self.genomics_encoder = ModalitySpecificEncoder(
            input_dim=2048, embed_dim=config.embed_dim, modality_type='genomics'
        )
        self.clinical_encoder = ModalitySpecificEncoder(
            input_dim=156, embed_dim=config.embed_dim, modality_type='clinical'
        )
        self.methylation_encoder = ModalitySpecificEncoder(
            input_dim=1000, embed_dim=config.embed_dim, modality_type='methylation'
        )
        
        # Fusion mechanism
        if config.fusion_strategy == 'attention':
            self.fusion_module = CrossModalAttention(
                embed_dim=config.embed_dim,
                num_modalities=4,
                num_heads=config.num_attention_heads
            )
        elif config.fusion_strategy == 'gated':
            self.fusion_module = GatedFusion(
                embed_dim=config.embed_dim,
                num_modalities=4
            )
        elif config.fusion_strategy == 'adaptive':
            self.fusion_module = AdaptiveFusion(
                embed_dim=config.embed_dim,
                num_modalities=4
            )
        else:
            # Simple concatenation
            self.fusion_module = nn.Sequential(
                nn.Linear(config.embed_dim * 4, config.embed_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.embed_dim, config.embed_dim)
            )
        
        # Transformer layers for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Prediction heads
        self.cancer_type_head = self._build_prediction_head(CANCER_TYPES_COUNT)
        self.stage_prediction_head = self._build_prediction_head(12)  # TNM stages
        self.survival_prediction_head = self._build_prediction_head(1, activation='sigmoid')
        self.treatment_response_head = self._build_prediction_head(4)  # Response categories
        self.biomarker_head = self._build_prediction_head(50)  # Multiple biomarkers
        self.risk_score_head = self._build_prediction_head(1, activation='sigmoid')
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _build_prediction_head(self, output_dim: int, activation: Optional[str] = None) -> nn.Module:
        """Build a prediction head with optional activation"""
        
        layers = [
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, output_dim)
        ]
        
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, 
                wsi_data: torch.Tensor,
                genomics_data: torch.Tensor,
                clinical_data: torch.Tensor,
                methylation_data: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the integrated model
        
        Args:
            wsi_data: WSI features [batch_size, wsi_dim]
            genomics_data: Genomics features [batch_size, genomics_dim]
            clinical_data: Clinical features [batch_size, clinical_dim]
            methylation_data: Methylation features [batch_size, methylation_dim]
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing all predictions and optional attention maps
        """
        
        # Encode each modality
        wsi_features = self.wsi_encoder(wsi_data)
        genomics_features = self.genomics_encoder(genomics_data)
        clinical_features = self.clinical_encoder(clinical_data)
        methylation_features = self.methylation_encoder(methylation_data)
        
        modal_features = [wsi_features, genomics_features, clinical_features, methylation_features]
        
        # Apply fusion
        if self.config.fusion_strategy == 'attention':
            fused_features_list, attention_maps = self.fusion_module(modal_features)
            fused_features = torch.stack(fused_features_list).mean(dim=0)
            fusion_info = {'attention_maps': attention_maps}
            
        elif self.config.fusion_strategy in ['gated', 'adaptive']:
            fused_features, fusion_info = self.fusion_module(modal_features)
            
        else:
            # Simple concatenation
            concatenated = torch.cat(modal_features, dim=1)
            fused_features = self.fusion_module(concatenated)
            fusion_info = {}
        
        # Apply transformer for sequence modeling
        # Add sequence dimension for transformer
        fused_features_seq = fused_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        transformer_output = self.transformer(fused_features_seq)
        final_features = transformer_output.squeeze(1)  # [batch_size, embed_dim]
        
        # Generate predictions
        predictions = {
            'cancer_type': self.cancer_type_head(final_features),
            'stage': self.stage_prediction_head(final_features),
            'survival': self.survival_prediction_head(final_features),
            'treatment_response': self.treatment_response_head(final_features),
            'biomarkers': self.biomarker_head(final_features),
            'risk_score': self.risk_score_head(final_features),
            'uncertainty': self.uncertainty_head(final_features),
            'integrated_features': final_features
        }
        
        # Add fusion information if requested
        if return_attention and fusion_info:
            predictions.update(fusion_info)
        
        return predictions
    
    def compute_modality_contributions(self, 
                                     wsi_data: torch.Tensor,
                                     genomics_data: torch.Tensor,
                                     clinical_data: torch.Tensor,
                                     methylation_data: torch.Tensor) -> Dict[str, float]:
        """Compute the contribution of each modality to the final prediction"""
        
        # Get baseline prediction with all modalities
        full_prediction = self.forward(wsi_data, genomics_data, clinical_data, methylation_data)
        baseline_score = full_prediction['risk_score']
        
        # Compute contribution by ablation
        contributions = {}
        
        # WSI contribution
        zero_wsi = torch.zeros_like(wsi_data)
        pred_no_wsi = self.forward(zero_wsi, genomics_data, clinical_data, methylation_data)
        contributions['wsi'] = float((baseline_score - pred_no_wsi['risk_score']).abs().mean())
        
        # Genomics contribution
        zero_genomics = torch.zeros_like(genomics_data)
        pred_no_genomics = self.forward(wsi_data, zero_genomics, clinical_data, methylation_data)
        contributions['genomics'] = float((baseline_score - pred_no_genomics['risk_score']).abs().mean())
        
        # Clinical contribution
        zero_clinical = torch.zeros_like(clinical_data)
        pred_no_clinical = self.forward(wsi_data, genomics_data, zero_clinical, methylation_data)
        contributions['clinical'] = float((baseline_score - pred_no_clinical['risk_score']).abs().mean())
        
        # Methylation contribution
        zero_methylation = torch.zeros_like(methylation_data)
        pred_no_methylation = self.forward(wsi_data, genomics_data, clinical_data, zero_methylation)
        contributions['methylation'] = float((baseline_score - pred_no_methylation['risk_score']).abs().mean())
        
        # Normalize contributions
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            contributions = {k: v / total_contrib for k, v in contributions.items()}
        
        return contributions

def create_sample_multimodal_data(batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Create sample multimodal data for testing"""
    
    return {
        'wsi_data': torch.randn(batch_size, 512),  # WSI features
        'genomics_data': torch.randn(batch_size, 2048),  # Genomics features
        'clinical_data': torch.randn(batch_size, 156),  # Clinical features
        'methylation_data': torch.randn(batch_size, 1000)  # Methylation features
    }

def run_multimodal_fusion_demo():
    """Run a demonstration of multimodal fusion analysis"""
    
    print("=" * 60)
    print("MULTIMODAL FUSION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = MultimodalConfig(
        embed_dim=256,
        hidden_dim=512,
        num_attention_heads=8,
        dropout_rate=0.1,
        num_transformer_layers=3,
        fusion_strategy='adaptive'
    )
    
    # Create model
    model = IntegratedCancerAnalysisModel(config)
    
    print("MODEL ARCHITECTURE:")
    print(f"Embedding Dimension: {config.embed_dim}")
    print(f"Hidden Dimension: {config.hidden_dim}")
    print(f"Attention Heads: {config.num_attention_heads}")
    print(f"Transformer Layers: {config.num_transformer_layers}")
    print(f"Fusion Strategy: {config.fusion_strategy}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create sample data
    batch_size = 4
    sample_data = create_sample_multimodal_data(batch_size)
    
    print("SAMPLE DATA:")
    for modality, data in sample_data.items():
        print(f"{modality}: {data.shape}")
    print()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(
            wsi_data=sample_data['wsi_data'],
            genomics_data=sample_data['genomics_data'],
            clinical_data=sample_data['clinical_data'],
            methylation_data=sample_data['methylation_data'],
            return_attention=True
        )
    
    print("PREDICTION RESULTS:")
    print(f"Cancer Type Predictions: {predictions['cancer_type'].shape}")
    print(f"Stage Predictions: {predictions['stage'].shape}")
    print(f"Survival Predictions: {predictions['survival'].shape}")
    print(f"Treatment Response: {predictions['treatment_response'].shape}")
    print(f"Biomarker Predictions: {predictions['biomarkers'].shape}")
    print(f"Risk Scores: {predictions['risk_score'].shape}")
    print(f"Uncertainty Scores: {predictions['uncertainty'].shape}")
    print()
    
    # Sample predictions for first patient
    print("SAMPLE PREDICTIONS (First Patient):")
    print(f"Risk Score: {predictions['risk_score'][0].item():.4f}")
    print(f"Survival Probability: {predictions['survival'][0].item():.4f}")
    print(f"Uncertainty: {predictions['uncertainty'][0].item():.4f}")
    print()
    
    # Compute modality contributions
    contributions = model.compute_modality_contributions(
        sample_data['wsi_data'],
        sample_data['genomics_data'],
        sample_data['clinical_data'],
        sample_data['methylation_data']
    )
    
    print("MODALITY CONTRIBUTIONS:")
    for modality, contribution in contributions.items():
        print(f"{modality.upper()}: {contribution:.3f} ({contribution*100:.1f}%)")
    print()
    
    # Fusion strategy analysis
    if 'strategy_weights' in predictions:
        strategy_weights = predictions['strategy_weights']
        print("FUSION STRATEGY WEIGHTS:")
        strategies = ['Attention', 'Gated', 'Concatenation']
        for i, strategy in enumerate(strategies):
            weight = strategy_weights[0, i].item()
            print(f"{strategy}: {weight:.3f} ({weight*100:.1f}%)")
        print()
    
    # Attention analysis
    if 'attention_maps' in predictions:
        print("ATTENTION MAPS:")
        for key, attention_map in predictions['attention_maps'].items():
            print(f"{key}: {attention_map.shape}")
        print()
    
    # Feature analysis
    integrated_features = predictions['integrated_features']
    print("INTEGRATED FEATURES:")
    print(f"Shape: {integrated_features.shape}")
    print(f"Mean: {integrated_features.mean().item():.4f}")
    print(f"Std: {integrated_features.std().item():.4f}")
    print(f"Min: {integrated_features.min().item():.4f}")
    print(f"Max: {integrated_features.max().item():.4f}")
    print()
    
    # Test different fusion strategies
    print("FUSION STRATEGY COMPARISON:")
    strategies = ['attention', 'gated', 'adaptive', 'concatenation']
    
    for strategy in strategies:
        config_test = MultimodalConfig(
            embed_dim=256,
            hidden_dim=512,
            fusion_strategy=strategy
        )
        model_test = IntegratedCancerAnalysisModel(config_test)
        model_test.eval()
        
        with torch.no_grad():
            pred_test = model_test(
                sample_data['wsi_data'],
                sample_data['genomics_data'],
                sample_data['clinical_data'],
                sample_data['methylation_data']
            )
        
        risk_score = pred_test['risk_score'][0].item()
        uncertainty = pred_test['uncertainty'][0].item()
        
        print(f"{strategy.upper()}: Risk={risk_score:.4f}, Uncertainty={uncertainty:.4f}")
    
    print()
    print("MULTIMODAL FUSION DEMO COMPLETED SUCCESSFULLY!")
    
    return {
        'model': model,
        'predictions': predictions,
        'contributions': contributions,
        'sample_data': sample_data
    }

if __name__ == "__main__":
    # Run demonstration
    result = run_multimodal_fusion_demo()
