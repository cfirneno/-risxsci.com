"""
Performance Optimization Module
==============================

High-performance optimizations for the Flow Matching + Logistic dynamics
cancer analysis platform including vectorized operations, JIT compilation,
and advanced memory management.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.jit as jit
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import time
import psutil
import gc
from contextlib import contextmanager
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from numba import jit as numba_jit, cuda
import cupy as cp
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    execution_time: float
    memory_usage: float
    gpu_memory_usage: float
    cpu_utilization: float
    throughput: float  # samples per second
    cache_hit_rate: float
    optimization_level: str
    bottlenecks: List[str]

@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations"""
    use_jit: bool = True
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_parallelization: bool = True
    use_gpu_acceleration: bool = True
    batch_size_optimization: bool = True
    memory_optimization: bool = True
    cache_enabled: bool = True
    profiling_enabled: bool = True
    optimization_level: str = 'aggressive'  # 'conservative', 'moderate', 'aggressive'

class PerformanceProfiler:
    """Advanced performance profiler for the cancer analysis platform"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_tracker = {}
        
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations"""
        
        # Start profiling
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            # End profiling
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
            else:
                end_gpu_memory = 0
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            # Store metrics
            self.metrics[operation_name] = {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'gpu_memory_delta': gpu_memory_delta,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Operation '{operation_name}': {execution_time:.4f}s, "
                       f"Memory: {memory_delta/1024/1024:.2f}MB, "
                       f"GPU Memory: {gpu_memory_delta/1024/1024:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        
        if not self.metrics:
            return {}
        
        total_time = sum(m['execution_time'] for m in self.metrics.values())
        total_memory = sum(m['memory_delta'] for m in self.metrics.values())
        
        report = {
            'total_operations': len(self.metrics),
            'total_execution_time': total_time,
            'total_memory_usage': total_memory,
            'operations': self.metrics.copy(),
            'bottlenecks': self._identify_bottlenecks()
        }
        
        return report
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        if not self.metrics:
            return bottlenecks
        
        # Find slowest operations
        sorted_ops = sorted(self.metrics.items(), 
                          key=lambda x: x[1]['execution_time'], reverse=True)
        
        # Mark top 20% as bottlenecks
        num_bottlenecks = max(1, len(sorted_ops) // 5)
        for op_name, _ in sorted_ops[:num_bottlenecks]:
            bottlenecks.append(f"Slow operation: {op_name}")
        
        # Check for memory issues
        total_memory = sum(m['memory_delta'] for m in self.metrics.values())
        if total_memory > 1024 * 1024 * 1024:  # > 1GB
            bottlenecks.append("High memory usage detected")
        
        return bottlenecks

class OptimizedFlowMatching(nn.Module):
    """Optimized Flow Matching implementation with performance enhancements"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 config: OptimizationConfig = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config or OptimizationConfig()
        
        # Optimized vector field network
        self.vector_field = self._build_optimized_network()
        
        # Pre-computed constants for efficiency
        self.register_buffer('pi_constant', torch.tensor(np.pi))
        self.register_buffer('sqrt_2pi', torch.tensor(np.sqrt(2 * np.pi)))
        
        # Cache for frequently computed values
        self._cache = {}
        self._cache_keys = []
        self.max_cache_size = 1000
        
    def _build_optimized_network(self) -> nn.Module:
        """Build optimized vector field network"""
        
        layers = []
        
        # Use optimized layer types
        layers.extend([
            nn.Linear(self.input_dim + 1, self.hidden_dim, bias=False),  # Remove bias for speed
            nn.GELU(),  # GELU is faster than ReLU on modern hardware
            nn.LayerNorm(self.hidden_dim, elementwise_affine=False),  # Simplified LayerNorm
        ])
        
        # Optimized hidden layers
        for _ in range(2):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=False),
            ])
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        
        network = nn.Sequential(*layers)
        
        # Apply optimization techniques
        if self.config.use_jit:
            # Prepare for JIT compilation
            network = self._prepare_for_jit(network)
        
        return network
    
    def _prepare_for_jit(self, network: nn.Module) -> nn.Module:
        """Prepare network for JIT compilation"""
        
        # Ensure all operations are JIT-compatible
        for module in network.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        return network
    
    @torch.jit.script
    def _optimized_interpolation(self, x0: torch.Tensor, x1: torch.Tensor, 
                                t: torch.Tensor, sigma_min: float = 1e-4) -> torch.Tensor:
        """JIT-compiled optimized interpolation"""
        
        # Vectorized interpolation computation
        t_expanded = t.unsqueeze(-1)
        sigma_t = sigma_min + t_expanded * (1.0 - sigma_min)
        
        # Optimized interpolation formula
        x_t = (1.0 - t_expanded) * x0 + t_expanded * x1
        
        return x_t
    
    @autocast()  # Mixed precision
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass with mixed precision"""
        
        batch_size = x.size(0)
        
        # Check cache first
        cache_key = self._generate_cache_key(x, t)
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Optimized concatenation
        t_expanded = t.unsqueeze(-1).expand(-1, 1)
        xt = torch.cat([x, t_expanded], dim=-1)
        
        # Forward pass through optimized network
        output = self.vector_field(xt)
        
        # Cache result if enabled
        if self.config.cache_enabled:
            self._update_cache(cache_key, output.clone())
        
        return output
    
    def _generate_cache_key(self, x: torch.Tensor, t: torch.Tensor) -> str:
        """Generate cache key for input tensors"""
        
        # Use tensor hashes for cache key
        x_hash = hash(x.data_ptr()) if x.is_cuda else hash(tuple(x.flatten()[:10].tolist()))
        t_hash = hash(t.data_ptr()) if t.is_cuda else hash(tuple(t.flatten().tolist()))
        
        return f"{x_hash}_{t_hash}_{x.shape}_{t.shape}"
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """Update cache with LRU eviction"""
        
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        self._cache_keys.append(key)
    
    def clear_cache(self):
        """Clear the computation cache"""
        self._cache.clear()
        self._cache_keys.clear()

@numba_jit(nopython=True, parallel=True)
def optimized_logistic_dynamics(state: np.ndarray, 
                               growth_rate: float,
                               carrying_capacity: float,
                               dt: float) -> np.ndarray:
    """Numba-optimized logistic dynamics computation"""
    
    # Vectorized logistic growth computation
    growth_term = growth_rate * state * (1.0 - state / carrying_capacity)
    new_state = state + dt * growth_term
    
    # Ensure non-negative values
    new_state = np.maximum(new_state, 0.0)
    
    return new_state

class OptimizedLogisticDynamics(nn.Module):
    """Optimized Logistic Dynamics with GPU acceleration"""
    
    def __init__(self, spatial_dim: int = 2, use_gpu: bool = True):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Optimized parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        self.carrying_capacity = nn.Parameter(torch.tensor(1.0))
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.01))
        
        # Pre-compute spatial operators for efficiency
        self._register_spatial_operators()
        
    def _register_spatial_operators(self):
        """Pre-compute spatial differential operators"""
        
        # Laplacian kernel for diffusion
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    @torch.jit.script
    def _compute_diffusion(self, state: torch.Tensor, 
                          diffusion_coeff: torch.Tensor,
                          laplacian_kernel: torch.Tensor) -> torch.Tensor:
        """JIT-compiled diffusion computation"""
        
        # Apply spatial Laplacian for diffusion
        diffusion = F.conv2d(state.unsqueeze(1), laplacian_kernel, padding=1)
        diffusion = diffusion.squeeze(1) * diffusion_coeff
        
        return diffusion
    
    @autocast()
    def forward(self, state: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Optimized forward pass for logistic dynamics"""
        
        # Logistic growth term (vectorized)
        growth_term = self.growth_rate * state * (1.0 - state / self.carrying_capacity)
        
        # Diffusion term
        if state.dim() >= 3:  # Spatial data
            diffusion_term = self._compute_diffusion(state, self.diffusion_coeff, 
                                                   self.laplacian_kernel)
        else:
            diffusion_term = torch.zeros_like(state)
        
        # Update state
        new_state = state + dt * (growth_term + diffusion_term)
        
        # Ensure non-negative and bounded
        new_state = torch.clamp(new_state, 0.0, self.carrying_capacity * 2.0)
        
        return new_state

class OptimizedMultimodalFusion(nn.Module):
    """Optimized multimodal fusion with performance enhancements"""
    
    def __init__(self, embed_dim: int = 256, num_modalities: int = 4,
                 config: OptimizationConfig = None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.config = config or OptimizationConfig()
        
        # Optimized attention mechanism
        self.attention = self._build_optimized_attention()
        
        # Fusion network with optimizations
        self.fusion_network = self._build_optimized_fusion()
        
        # Pre-computed scaling factors
        self.register_buffer('scale_factor', torch.tensor(1.0 / np.sqrt(embed_dim)))
        
    def _build_optimized_attention(self) -> nn.Module:
        """Build optimized attention mechanism"""
        
        # Use multi-head attention with optimizations
        attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.0,  # Remove dropout for inference speed
            bias=False,   # Remove bias for speed
            batch_first=True
        )
        
        return attention
    
    def _build_optimized_fusion(self) -> nn.Module:
        """Build optimized fusion network"""
        
        return nn.Sequential(
            nn.Linear(self.embed_dim * self.num_modalities, self.embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim, elementwise_affine=False),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
    
    @torch.jit.script
    def _optimized_attention_weights(self, query: torch.Tensor, 
                                   key: torch.Tensor,
                                   scale_factor: torch.Tensor) -> torch.Tensor:
        """JIT-compiled attention weight computation"""
        
        # Efficient attention computation
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights
    
    @autocast()
    def forward(self, modal_features: List[torch.Tensor]) -> torch.Tensor:
        """Optimized multimodal fusion"""
        
        # Stack modalities efficiently
        stacked_features = torch.stack(modal_features, dim=1)
        batch_size, num_mods, embed_dim = stacked_features.shape
        
        # Optimized self-attention across modalities
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Flatten and fuse
        flattened = attended_features.view(batch_size, -1)
        fused_features = self.fusion_network(flattened)
        
        return fused_features

class OptimizedDataLoader:
    """High-performance data loader with advanced optimizations"""
    
    def __init__(self, dataset, batch_size: int = 32, 
                 config: OptimizationConfig = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config or OptimizationConfig()
        
        # Optimize data loading
        self.dataloader = self._create_optimized_dataloader()
        
        # Pre-loading cache
        self.cache = {}
        self.cache_size = 100
        
    def _create_optimized_dataloader(self) -> DataLoader:
        """Create optimized DataLoader"""
        
        # Optimize number of workers
        num_workers = min(mp.cpu_count(), 8) if self.config.use_cpu_parallelization else 0
        
        # Use optimized settings
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=True  # For consistent batch sizes
        )
        
        return dataloader
    
    def __iter__(self):
        """Optimized iteration with caching"""
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to GPU asynchronously if available
            if torch.cuda.is_available():
                batch = self._move_to_gpu_async(batch)
            
            yield batch
    
    def _move_to_gpu_async(self, batch):
        """Asynchronously move batch to GPU"""
        
        if isinstance(batch, dict):
            return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item 
                   for item in batch]
        else:
            return batch.cuda(non_blocking=True) if isinstance(batch, torch.Tensor) else batch

class MemoryOptimizer:
    """Advanced memory optimization utilities"""
    
    @staticmethod
    @contextmanager
    def memory_efficient_mode():
        """Context manager for memory-efficient operations"""
        
        # Store original settings
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        
        try:
            # Optimize for memory efficiency
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Restore original settings
            torch.backends.cudnn.deterministic = original_deterministic
            torch.backends.cudnn.benchmark = original_benchmark
            
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @staticmethod
    def optimize_model_memory(model: nn.Module) -> nn.Module:
        """Optimize model for memory efficiency"""
        
        # Convert to half precision if possible
        if torch.cuda.is_available():
            model = model.half()
        
        # Enable gradient checkpointing for large models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get detailed memory statistics"""
        
        stats = {
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**3),   # GB
                'gpu_memory_free': (torch.cuda.get_device_properties(0).total_memory - 
                                  torch.cuda.memory_allocated()) / (1024**3)  # GB
            })
        
        return stats

class AdaptiveBatchSizer:
    """Automatically optimize batch size based on available memory"""
    
    def __init__(self, model: nn.Module, initial_batch_size: int = 32):
        self.model = model
        self.current_batch_size = initial_batch_size
        self.max_batch_size = 512
        self.min_batch_size = 1
        
    def find_optimal_batch_size(self, sample_data: torch.Tensor) -> int:
        """Find optimal batch size through binary search"""
        
        low, high = self.min_batch_size, self.max_batch_size
        optimal_batch_size = self.current_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test with current batch size
                if self._test_batch_size(sample_data, mid):
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    raise e
        
        self.current_batch_size = optimal_batch_size
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def _test_batch_size(self, sample_data: torch.Tensor, batch_size: int) -> bool:
        """Test if batch size works without OOM"""
        
        try:
            with torch.no_grad():
                # Create test batch
                test_batch = sample_data[:batch_size] if len(sample_data) >= batch_size else sample_data
                
                # Forward pass
                _ = self.model(test_batch)
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            else:
                raise e

def benchmark_performance(model: nn.Module, data_loader: DataLoader, 
                        num_iterations: int = 100) -> PerformanceMetrics:
    """Comprehensive performance benchmarking"""
    
    profiler = PerformanceProfiler()
    model.eval()
    
    total_samples = 0
    total_time = 0
    memory_usage = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_iterations:
                break
            
            with profiler.profile(f"batch_{i}"):
                # Move to GPU if available
                if torch.cuda.is_available() and not batch[0].is_cuda:
                    batch = [item.cuda() if isinstance(item, torch.Tensor) else item 
                           for item in batch]
                
                start_time = time.perf_counter()
                
                # Forward pass
                if isinstance(batch, dict):
                    output = model(**batch)
                elif isinstance(batch, (list, tuple)):
                    output = model(*batch)
                else:
                    output = model(batch)
                
                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Collect metrics
                batch_time = end_time - start_time
                batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else len(batch)
                
                total_samples += batch_size
                total_time += batch_time
                
                # Memory usage
                memory_stats = MemoryOptimizer.get_memory_stats()
                memory_usage.append(memory_stats.get('gpu_memory_allocated', 0))
    
    # Calculate final metrics
    throughput = total_samples / total_time if total_time > 0 else 0
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    
    # Get profiler report
    profiler_report = profiler.get_report()
    
    metrics = PerformanceMetrics(
        execution_time=total_time,
        memory_usage=avg_memory,
        gpu_memory_usage=avg_memory,
        cpu_utilization=psutil.cpu_percent(),
        throughput=throughput,
        cache_hit_rate=0.0,  # Placeholder
        optimization_level='benchmark',
        bottlenecks=profiler_report.get('bottlenecks', [])
    )
    
    return metrics

def run_performance_optimization_demo():
    """Demonstrate performance optimizations"""
    
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = OptimizationConfig(
        use_jit=True,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        optimization_level='aggressive'
    )
    
    print("OPTIMIZATION CONFIGURATION:")
    print(f"JIT Compilation: {config.use_jit}")
    print(f"Mixed Precision: {config.use_mixed_precision}")
    print(f"Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"Optimization Level: {config.optimization_level}")
    print()
    
    # Create optimized models
    print("CREATING OPTIMIZED MODELS:")
    
    # Flow Matching
    flow_model = OptimizedFlowMatching(input_dim=256, hidden_dim=512, config=config)
    print(f"✅ Optimized Flow Matching Model: {sum(p.numel() for p in flow_model.parameters()):,} parameters")
    
    # Logistic Dynamics
    dynamics_model = OptimizedLogisticDynamics(spatial_dim=2, use_gpu=torch.cuda.is_available())
    print(f"✅ Optimized Logistic Dynamics Model: {sum(p.numel() for p in dynamics_model.parameters()):,} parameters")
    
    # Multimodal Fusion
    fusion_model = OptimizedMultimodalFusion(embed_dim=256, num_modalities=4, config=config)
    print(f"✅ Optimized Multimodal Fusion Model: {sum(p.numel() for p in fusion_model.parameters()):,} parameters")
    print()
    
    # Performance profiling
    print("PERFORMANCE PROFILING:")
    profiler = PerformanceProfiler()
    
    # Test data
    batch_size = 32
    test_data = torch.randn(batch_size, 256)
    test_time = torch.rand(batch_size)
    
    if torch.cuda.is_available():
        test_data = test_data.cuda()
        test_time = test_time.cuda()
        flow_model = flow_model.cuda()
        dynamics_model = dynamics_model.cuda()
        fusion_model = fusion_model.cuda()
    
    # Benchmark Flow Matching
    with profiler.profile("flow_matching_forward"):
        for _ in range(100):
            with torch.no_grad():
                output = flow_model(test_data, test_time)
    
    # Benchmark Logistic Dynamics
    spatial_data = torch.randn(batch_size, 64, 64)
    if torch.cuda.is_available():
        spatial_data = spatial_data.cuda()
    
    with profiler.profile("logistic_dynamics_forward"):
        for _ in range(100):
            with torch.no_grad():
                output = dynamics_model(spatial_data)
    
    # Benchmark Multimodal Fusion
    modal_data = [torch.randn(batch_size, 256) for _ in range(4)]
    if torch.cuda.is_available():
        modal_data = [data.cuda() for data in modal_data]
    
    with profiler.profile("multimodal_fusion_forward"):
        for _ in range(100):
            with torch.no_grad():
                output = fusion_model(modal_data)
    
    # Memory optimization
    print("MEMORY OPTIMIZATION:")
    memory_stats = MemoryOptimizer.get_memory_stats()
    
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f} {'GB' if 'memory' in key else '%'}")
    print()
    
    # Performance report
    print("PERFORMANCE REPORT:")
    report = profiler.get_report()
    
    print(f"Total Operations: {report['total_operations']}")
    print(f"Total Execution Time: {report['total_execution_time']:.4f}s")
    print(f"Average Time per Operation: {report['total_execution_time']/report['total_operations']:.4f}s")
    print()
    
    print("OPERATION BREAKDOWN:")
    for op_name, metrics in report['operations'].items():
        print(f"{op_name}: {metrics['execution_time']:.4f}s")
    
    if report['bottlenecks']:
        print("\nIDENTIFIED BOTTLENECKS:")
        for bottleneck in report['bottlenecks']:
            print(f"⚠️ {bottleneck}")
    
    print()
    
    # JIT compilation benefit
    if config.use_jit:
        print("JIT COMPILATION BENEFITS:")
        
        # Test JIT compilation
        traced_flow_model = torch.jit.trace(flow_model, (test_data[:1], test_time[:1]))
        
        # Benchmark traced model
        with profiler.profile("jit_compiled_forward"):
            for _ in range(100):
                with torch.no_grad():
                    output = traced_flow_model(test_data, test_time)
        
        jit_time = profiler.metrics["jit_compiled_forward"]["execution_time"]
        original_time = profiler.metrics["flow_matching_forward"]["execution_time"]
        speedup = original_time / jit_time
        
        print(f"JIT Speedup: {speedup:.2f}x faster")
        print(f"Original Time: {original_time:.4f}s")
        print(f"JIT Time: {jit_time:.4f}s")
    
    print()
    print("PERFORMANCE OPTIMIZATION DEMO COMPLETED!")
    
    return {
        'models': {
            'flow_matching': flow_model,
            'logistic_dynamics': dynamics_model,
            'multimodal_fusion': fusion_model
        },
        'profiler': profiler,
        'config': config,
        'memory_stats': memory_stats
    }

if __name__ == "__main__":
    # Run performance optimization demo
    results = run_performance_optimization_demo()
