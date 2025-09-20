"""
Distributed Computing Module
===========================

Enterprise-scale distributed computing for the Flow Matching + Logistic dynamics
cancer analysis platform including multi-node training, federated learning,
cloud auto-scaling, and fault-tolerant processing.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.pipeline.sync import Pipe
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
import os
import json
import pickle
import hashlib
import asyncio
import aiohttp
import zmq
import redis
import kubernetes
from kubernetes import client, config
import docker
import boto3
import ray
from ray import tune, serve
import horovod.torch as hvd
from mpi4py import MPI
import socket
import threading
from contextlib import contextmanager
import psutil
import yaml
from cryptography.fernet import Fernet
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    use_mixed_precision: bool = True
    gradient_compression: bool = True
    elastic_training: bool = True
    fault_tolerance: bool = True
    checkpointing: bool = True
    federated_learning: bool = False
    privacy_budget: float = 1.0
    differential_privacy: bool = False

@dataclass
class NodeInfo:
    """Information about distributed nodes"""
    node_id: str
    rank: int
    hostname: str
    ip_address: str
    gpu_count: int
    memory_gb: float
    cpu_cores: int
    status: str  # 'active', 'inactive', 'failed', 'joining'
    last_heartbeat: datetime
    workload: float  # 0.0 to 1.0

@dataclass
class FederatedClient:
    """Federated learning client information"""
    client_id: str
    institution: str
    data_size: int
    model_version: str
    privacy_level: str
    last_update: datetime
    performance_metrics: Dict[str, float]
    encryption_key: str

class DistributedTrainingManager:
    """Manager for distributed training across multiple nodes"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.nodes = {}
        self.training_state = {}
        self.checkpoints = {}
        
        # Initialize distributed training
        self._init_distributed()
        
        # Setup fault tolerance
        if config.fault_tolerance:
            self._setup_fault_tolerance()
        
        # Setup checkpointing
        if config.checkpointing:
            self._setup_checkpointing()
    
    def _init_distributed(self):
        """Initialize distributed training environment"""
        
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=timedelta(minutes=30)
            )
        
        # Initialize Horovod if available
        try:
            hvd.init()
            logger.info(f"Horovod initialized - Rank: {hvd.rank()}, Size: {hvd.size()}")
        except:
            logger.info("Horovod not available, using PyTorch distributed")
        
        logger.info(f"Distributed training initialized - Rank: {self.config.rank}, "
                   f"World Size: {self.config.world_size}")
    
    def _setup_fault_tolerance(self):
        """Setup fault tolerance mechanisms"""
        
        # Node health monitoring
        self.health_monitor = NodeHealthMonitor()
        
        # Backup nodes registry
        self.backup_nodes = []
        
        # Failure detection interval
        self.failure_detection_interval = 30  # seconds
        
        logger.info("Fault tolerance mechanisms initialized")
    
    def _setup_checkpointing(self):
        """Setup distributed checkpointing"""
        
        # Checkpoint directory
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint frequency
        self.checkpoint_frequency = 100  # every N iterations
        
        # Checkpoint versioning
        self.checkpoint_version = 0
        
        logger.info(f"Checkpointing initialized - Directory: {self.checkpoint_dir}")
    
    def train_distributed_model(self, model: nn.Module, 
                               train_loader, 
                               optimizer,
                               num_epochs: int = 10) -> Dict[str, Any]:
        """Train model using distributed computing"""
        
        # Wrap model for distributed training
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Choose distributed strategy
        if self.config.world_size > 1:
            model = self._wrap_distributed_model(model)
        
        # Distributed data sampler
        if hasattr(train_loader.dataset, '__len__'):
            sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
            
            # Recreate dataloader with distributed sampler
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=sampler,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory
            )
        
        # Training loop with fault tolerance
        training_metrics = {
            'epoch_times': [],
            'losses': [],
            'node_failures': 0,
            'checkpoints_saved': 0
        }
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            try:
                # Set epoch for distributed sampler
                if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                epoch_loss = self._train_epoch(model, train_loader, optimizer, epoch)
                
                # Synchronize metrics across nodes
                epoch_loss = self._synchronize_metrics(epoch_loss)
                
                # Checkpointing
                if self.config.checkpointing and epoch % (self.checkpoint_frequency // 10) == 0:
                    self._save_checkpoint(model, optimizer, epoch, epoch_loss)
                    training_metrics['checkpoints_saved'] += 1
                
                epoch_time = time.time() - epoch_start
                training_metrics['epoch_times'].append(epoch_time)
                training_metrics['losses'].append(epoch_loss)
                
                if self.config.rank == 0:  # Only log from master node
                    logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Time={epoch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Training error on rank {self.config.rank}: {e}")
                
                if self.config.fault_tolerance:
                    # Attempt recovery
                    recovery_success = self._handle_training_failure(model, optimizer, epoch)
                    training_metrics['node_failures'] += 1
                    
                    if not recovery_success:
                        logger.error("Failed to recover from training error")
                        break
                else:
                    raise e
        
        return training_metrics
    
    def _wrap_distributed_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training"""
        
        # Choose distributed strategy based on model size
        model_size = sum(p.numel() for p in model.parameters())
        
        if model_size > 1e9:  # > 1B parameters
            # Use Fully Sharded Data Parallel for very large models
            model = FSDP(
                model,
                cpu_offload=torch.distributed.fsdp.CPUOffload(offload_params=True)
            )
            logger.info("Using Fully Sharded Data Parallel (FSDP)")
            
        elif hasattr(model, 'layers') and len(model.layers) > 10:
            # Use Pipeline Parallel for deep models
            balance = [1] * len(model.layers)  # Equal distribution
            model = Pipe(model, balance=balance, devices=list(range(torch.cuda.device_count())))
            logger.info("Using Pipeline Parallel")
            
        else:
            # Use standard Distributed Data Parallel
            model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                gradient_as_bucket_view=True,
                broadcast_buffers=False
            )
            logger.info("Using Distributed Data Parallel (DDP)")
        
        return model
    
    def _train_epoch(self, model: nn.Module, train_loader, optimizer, epoch: int) -> float:
        """Train single epoch with distributed coordination"""
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device
                if torch.cuda.is_available():
                    if isinstance(batch, dict):
                        batch = {k: v.cuda() if torch.cuda.is_available() and hasattr(v, 'cuda') else v 
                               for k, v in batch.items()}
                    elif isinstance(batch, (list, tuple)):
                        batch = [item.cuda() if torch.cuda.is_available() and hasattr(item, 'cuda') else item 
                               for item in batch]
                
                # Forward pass
                optimizer.zero_grad()
                
                if isinstance(batch, dict):
                    output = model(**batch)
                    loss = output.get('loss', output.get('logits', output))
                elif isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        output = model(batch[0])
                        loss = nn.functional.mse_loss(output, batch[1])
                    else:
                        output = model(batch[0])
                        loss = output.mean()  # Dummy loss for demo
                else:
                    output = model(batch)
                    loss = output.mean()  # Dummy loss for demo
                
                if isinstance(loss, dict):
                    loss = loss['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient synchronization (handled automatically by DDP)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Periodic sync check
                if batch_idx % 100 == 0:
                    self._check_node_health()
                
            except Exception as e:
                logger.error(f"Batch training error: {e}")
                continue
        
        return epoch_loss / max(num_batches, 1)
    
    def _synchronize_metrics(self, metric: float) -> float:
        """Synchronize metrics across all nodes"""
        
        if self.config.world_size <= 1:
            return metric
        
        # Convert to tensor for synchronization
        metric_tensor = torch.tensor(metric, dtype=torch.float32)
        
        if torch.cuda.is_available():
            metric_tensor = metric_tensor.cuda()
        
        # All-reduce to average across nodes
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        metric_tensor /= self.config.world_size
        
        return metric_tensor.item()
    
    def _check_node_health(self):
        """Check health of distributed nodes"""
        
        try:
            # Simple heartbeat mechanism
            heartbeat = {
                'rank': self.config.rank,
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'gpu_memory': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            }
            
            # In a real implementation, this would send heartbeat to a central coordinator
            logger.debug(f"Node {self.config.rank} heartbeat: {heartbeat}")
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
    
    def _save_checkpoint(self, model: nn.Module, optimizer, epoch: int, loss: float):
        """Save distributed training checkpoint"""
        
        if self.config.rank != 0:  # Only master saves checkpoints
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last 5 checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove all but the 5 most recent
        for old_checkpoint in checkpoints[5:]:
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def _handle_training_failure(self, model: nn.Module, optimizer, epoch: int) -> bool:
        """Handle training failures with recovery"""
        
        try:
            logger.info(f"Attempting recovery from training failure at epoch {epoch}")
            
            # Try to load latest checkpoint
            latest_checkpoint = self._find_latest_checkpoint()
            
            if latest_checkpoint:
                checkpoint = torch.load(latest_checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Recovered from checkpoint: {latest_checkpoint}")
                return True
            
            # If no checkpoint, try to reinitialize
            logger.warning("No checkpoint found, continuing with current state")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint"""
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda x: x.stat().st_mtime)

class FederatedLearningManager:
    """Manager for federated learning across institutions"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.clients = {}
        self.global_model = None
        self.aggregation_weights = {}
        
        # Privacy mechanisms
        if config.differential_privacy:
            self.privacy_engine = DifferentialPrivacyEngine(config.privacy_budget)
        
        # Secure communication
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        logger.info("Federated Learning Manager initialized")
    
    def register_client(self, client: FederatedClient):
        """Register a new federated learning client"""
        
        self.clients[client.client_id] = client
        
        # Set aggregation weight based on data size
        total_data = sum(c.data_size for c in self.clients.values())
        self.aggregation_weights[client.client_id] = client.data_size / total_data
        
        logger.info(f"Registered federated client: {client.client_id}")
    
    def federated_training_round(self, global_model: nn.Module, 
                                num_local_epochs: int = 5) -> Dict[str, Any]:
        """Execute one round of federated training"""
        
        round_metrics = {
            'participating_clients': len(self.clients),
            'model_updates': [],
            'aggregation_time': 0,
            'communication_time': 0
        }
        
        # Send global model to clients
        comm_start = time.time()
        client_updates = self._distribute_model_to_clients(global_model, num_local_epochs)
        round_metrics['communication_time'] = time.time() - comm_start
        
        # Aggregate client updates
        agg_start = time.time()
        aggregated_model = self._aggregate_client_updates(client_updates)
        round_metrics['aggregation_time'] = time.time() - agg_start
        
        # Update global model
        global_model.load_state_dict(aggregated_model)
        
        round_metrics['model_updates'] = len(client_updates)
        
        return round_metrics
    
    def _distribute_model_to_clients(self, model: nn.Module, 
                                   num_epochs: int) -> Dict[str, Dict]:
        """Distribute global model to clients for local training"""
        
        client_updates = {}
        
        for client_id, client in self.clients.items():
            try:
                # Serialize model
                model_state = model.state_dict()
                serialized_model = pickle.dumps(model_state)
                
                # Encrypt model for secure transmission
                encrypted_model = self.cipher.encrypt(serialized_model)
                
                # Simulate client training (in reality, this would be sent over network)
                update = self._simulate_client_training(client, encrypted_model, num_epochs)
                client_updates[client_id] = update
                
            except Exception as e:
                logger.error(f"Failed to train on client {client_id}: {e}")
                continue
        
        return client_updates
    
    def _simulate_client_training(self, client: FederatedClient, 
                                encrypted_model: bytes, 
                                num_epochs: int) -> Dict:
        """Simulate local training on client (in reality, this happens remotely)"""
        
        try:
            # Decrypt model
            serialized_model = self.cipher.decrypt(encrypted_model)
            model_state = pickle.loads(serialized_model)
            
            # Simulate training process
            time.sleep(0.1)  # Simulate training time
            
            # Add noise for differential privacy
            if self.config.differential_privacy:
                model_state = self.privacy_engine.add_noise_to_gradients(model_state)
            
            # Simulate gradient update
            for key in model_state:
                if model_state[key].dtype.is_floating_point:
                    noise = torch.randn_like(model_state[key]) * 0.01
                    model_state[key] += noise
            
            # Return encrypted update
            serialized_update = pickle.dumps(model_state)
            encrypted_update = self.cipher.encrypt(serialized_update)
            
            return {
                'client_id': client.client_id,
                'encrypted_update': encrypted_update,
                'data_size': client.data_size,
                'training_loss': np.random.uniform(0.1, 0.5),  # Simulated loss
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Client training simulation failed: {e}")
            return {}
    
    def _aggregate_client_updates(self, client_updates: Dict[str, Dict]) -> Dict:
        """Aggregate client model updates using federated averaging"""
        
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}
        
        # Decrypt all updates
        decrypted_updates = {}
        total_weight = 0
        
        for client_id, update in client_updates.items():
            try:
                encrypted_update = update['encrypted_update']
                serialized_update = self.cipher.decrypt(encrypted_update)
                model_state = pickle.loads(serialized_update)
                
                weight = self.aggregation_weights.get(client_id, 1.0)
                decrypted_updates[client_id] = {
                    'model_state': model_state,
                    'weight': weight
                }
                total_weight += weight
                
            except Exception as e:
                logger.error(f"Failed to decrypt update from {client_id}: {e}")
                continue
        
        if not decrypted_updates:
            logger.error("No valid updates to aggregate")
            return {}
        
        # Federated averaging
        first_update = list(decrypted_updates.values())[0]['model_state']
        aggregated_state = {}
        
        for key in first_update:
            weighted_sum = torch.zeros_like(first_update[key])
            
            for client_id, update_data in decrypted_updates.items():
                model_state = update_data['model_state']
                weight = update_data['weight']
                weighted_sum += model_state[key] * weight
            
            aggregated_state[key] = weighted_sum / total_weight
        
        logger.info(f"Aggregated {len(decrypted_updates)} client updates")
        return aggregated_state

class DifferentialPrivacyEngine:
    """Differential privacy engine for federated learning"""
    
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.noise_scale = self._calculate_noise_scale()
    
    def _calculate_noise_scale(self) -> float:
        """Calculate noise scale based on privacy budget"""
        # Simplified calculation - in practice, use formal DP analysis
        return 1.0 / self.privacy_budget
    
    def add_noise_to_gradients(self, model_state: Dict) -> Dict:
        """Add differential privacy noise to model gradients"""
        
        noisy_state = {}
        
        for key, tensor in model_state.items():
            if tensor.dtype.is_floating_point:
                # Add Gaussian noise calibrated to sensitivity and privacy budget
                noise = torch.normal(0, self.noise_scale, tensor.shape)
                noisy_state[key] = tensor + noise
            else:
                noisy_state[key] = tensor
        
        return noisy_state

class CloudAutoScaler:
    """Automatic cloud resource scaling for distributed training"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.min_nodes = 1
        self.max_nodes = 100
        self.target_utilization = 0.8
        
        # Cloud provider clients
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
        
        # Kubernetes client
        try:
            config.load_incluster_config()  # If running in cluster
            self.k8s_client = client.AppsV1Api()
        except:
            try:
                config.load_kube_config()  # If running locally
                self.k8s_client = client.AppsV1Api()
            except:
                self.k8s_client = None
                logger.warning("Kubernetes client not available")
        
        # Resource monitoring
        self.monitoring_interval = 60  # seconds
        self.scaling_decisions = []
        
        logger.info("Cloud Auto Scaler initialized")
    
    def start_auto_scaling(self):
        """Start automatic scaling monitoring"""
        
        def scaling_loop():
            while True:
                try:
                    self._monitor_and_scale()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Auto-scaling error: {e}")
                    time.sleep(self.monitoring_interval)
        
        # Start scaling in background thread
        scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        scaling_thread.start()
        
        logger.info("Auto-scaling started")
    
    def _monitor_and_scale(self):
        """Monitor resource utilization and make scaling decisions"""
        
        # Get current metrics
        metrics = self._get_cluster_metrics()
        
        if not metrics:
            return
        
        current_nodes = metrics['node_count']
        avg_utilization = metrics['avg_utilization']
        queue_length = metrics.get('queue_length', 0)
        
        # Scaling decision logic
        target_nodes = current_nodes
        
        if avg_utilization > self.target_utilization or queue_length > 10:
            # Scale up
            target_nodes = min(current_nodes + 1, self.max_nodes)
            action = 'scale_up'
        elif avg_utilization < self.target_utilization * 0.5 and queue_length == 0:
            # Scale down
            target_nodes = max(current_nodes - 1, self.min_nodes)
            action = 'scale_down'
        else:
            action = 'no_change'
        
        # Execute scaling if needed
        if target_nodes != current_nodes:
            success = self._execute_scaling(target_nodes, action)
            
            self.scaling_decisions.append({
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'from_nodes': current_nodes,
                'to_nodes': target_nodes,
                'utilization': avg_utilization,
                'success': success
            })
            
            logger.info(f"Scaling decision: {action} from {current_nodes} to {target_nodes} nodes")
    
    def _get_cluster_metrics(self) -> Dict[str, Any]:
        """Get current cluster resource metrics"""
        
        try:
            if self.k8s_client:
                return self._get_k8s_metrics()
            else:
                return self._get_local_metrics()
        except Exception as e:
            logger.error(f"Failed to get cluster metrics: {e}")
            return {}
    
    def _get_k8s_metrics(self) -> Dict[str, Any]:
        """Get metrics from Kubernetes cluster"""
        
        try:
            # Get node count
            nodes = client.CoreV1Api().list_node()
            node_count = len(nodes.items)
            
            # Get pod metrics (simplified)
            pods = client.CoreV1Api().list_pod_for_all_namespaces()
            running_pods = sum(1 for pod in pods.items if pod.status.phase == 'Running')
            
            # Estimate utilization (simplified)
            avg_utilization = min(running_pods / max(node_count * 10, 1), 1.0)
            
            return {
                'node_count': node_count,
                'running_pods': running_pods,
                'avg_utilization': avg_utilization,
                'queue_length': 0  # Would need custom queue metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kubernetes metrics: {e}")
            return {}
    
    def _get_local_metrics(self) -> Dict[str, Any]:
        """Get local system metrics"""
        
        return {
            'node_count': 1,
            'avg_utilization': psutil.cpu_percent() / 100.0,
            'memory_utilization': psutil.virtual_memory().percent / 100.0,
            'queue_length': 0
        }
    
    def _execute_scaling(self, target_nodes: int, action: str) -> bool:
        """Execute scaling action"""
        
        try:
            if self.k8s_client and action == 'scale_up':
                return self._scale_up_k8s_deployment(target_nodes)
            elif self.k8s_client and action == 'scale_down':
                return self._scale_down_k8s_deployment(target_nodes)
            else:
                logger.info(f"Simulated scaling to {target_nodes} nodes")
                return True
                
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    def _scale_up_k8s_deployment(self, target_nodes: int) -> bool:
        """Scale up Kubernetes deployment"""
        
        try:
            # Scale worker deployment
            deployment_name = "cancer-analysis-workers"
            namespace = "default"
            
            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_nodes
            
            # Apply update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled up deployment to {target_nodes} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scale up failed: {e}")
            return False
    
    def _scale_down_k8s_deployment(self, target_nodes: int) -> bool:
        """Scale down Kubernetes deployment"""
        
        try:
            # Similar to scale up but with graceful shutdown
            deployment_name = "cancer-analysis-workers"
            namespace = "default"
            
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            
            deployment.spec.replicas = target_nodes
            
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Scaled down deployment to {target_nodes} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scale down failed: {e}")
            return False

class NodeHealthMonitor:
    """Monitor health of distributed nodes"""
    
    def __init__(self):
        self.nodes = {}
        self.health_checks = {}
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'gpu_memory': 90.0,
            'disk_usage': 80.0
        }
    
    def register_node(self, node_info: NodeInfo):
        """Register a node for health monitoring"""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"Registered node for monitoring: {node_info.node_id}")
    
    def check_node_health(self, node_id: str) -> Dict[str, Any]:
        """Check health of a specific node"""
        
        if node_id not in self.nodes:
            return {'status': 'unknown', 'error': 'Node not registered'}
        
        try:
            # In a real implementation, this would check remote node health
            health_status = {
                'node_id': node_id,
                'status': 'healthy',
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat(),
                'alerts': []
            }
            
            # Check GPU health if available
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
                health_status['gpu_memory_usage'] = gpu_memory_used
                
                if gpu_memory_used > self.alert_thresholds['gpu_memory']:
                    health_status['alerts'].append(f"High GPU memory usage: {gpu_memory_used:.1f}%")
            
            # Check alert thresholds
            for metric, threshold in self.alert_thresholds.items():
                if metric in health_status and health_status[metric] > threshold:
                    health_status['alerts'].append(f"High {metric}: {health_status[metric]:.1f}%")
                    health_status['status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            return {
                'node_id': node_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def run_distributed_computing_demo():
    """Demonstrate distributed computing capabilities"""
    
    print("=" * 60)
    print("DISTRIBUTED COMPUTING DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = DistributedConfig(
        backend='gloo',  # Use gloo for CPU demo
        world_size=1,
        rank=0,
        use_mixed_precision=True,
        fault_tolerance=True,
        checkpointing=True,
        federated_learning=True,
        differential_privacy=True,
        privacy_budget=1.0
    )
    
    print("DISTRIBUTED CONFIGURATION:")
    print(f"Backend: {config.backend}")
    print(f"World Size: {config.world_size}")
    print(f"Fault Tolerance: {config.fault_tolerance}")
    print(f"Federated Learning: {config.federated_learning}")
    print(f"Differential Privacy: {config.differential_privacy}")
    print()
    
    # Initialize distributed training manager
    print("INITIALIZING DISTRIBUTED TRAINING:")
    training_manager = DistributedTrainingManager(config)
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy training data
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 256),
        torch.randn(1000, 1)
    )
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Dataset: {len(dataset)} samples")
    print()
    
    # Distributed training demo
    print("DISTRIBUTED TRAINING DEMO:")
    
    training_metrics = training_manager.train_distributed_model(
        model, train_loader, optimizer, num_epochs=3
    )
    
    print("Training Results:")
    print(f"  Epochs Completed: {len(training_metrics['epoch_times'])}")
    print(f"  Average Epoch Time: {np.mean(training_metrics['epoch_times']):.2f}s")
    print(f"  Final Loss: {training_metrics['losses'][-1] if training_metrics['losses'] else 'N/A':.4f}")
    print(f"  Checkpoints Saved: {training_metrics['checkpoints_saved']}")
    print(f"  Node Failures: {training_metrics['node_failures']}")
    print()
    
    # Federated learning demo
    if config.federated_learning:
        print("FEDERATED LEARNING DEMO:")
        
        fed_manager = FederatedLearningManager(config)
        
        # Register mock clients
        clients = [
            FederatedClient(
                client_id=f"hospital_{i}",
                institution=f"Medical Center {i}",
                data_size=np.random.randint(1000, 5000),
                model_version="1.0",
                privacy_level="high",
                last_update=datetime.now(),
                performance_metrics={'accuracy': np.random.uniform(0.7, 0.9)},
                encryption_key="mock_key"
            )
            for i in range(5)
        ]
        
        for client in clients:
            fed_manager.register_client(client)
        
        print(f"✅ Registered {len(clients)} federated clients")
        
        # Run federated training round
        fed_metrics = fed_manager.federated_training_round(model, num_local_epochs=3)
        
        print("Federated Training Results:")
        print(f"  Participating Clients: {fed_metrics['participating_clients']}")
        print(f"  Model Updates: {fed_metrics['model_updates']}")
        print(f"  Aggregation Time: {fed_metrics['aggregation_time']:.2f}s")
        print(f"  Communication Time: {fed_metrics['communication_time']:.2f}s")
        print()
    
    # Cloud auto-scaling demo
    print("CLOUD AUTO-SCALING DEMO:")
    
    auto_scaler = CloudAutoScaler(config)
    
    # Get current cluster metrics
    metrics = auto_scaler._get_cluster_metrics()
    
    print("Cluster Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Simulate scaling decision
    auto_scaler._monitor_and_scale()
    
    if auto_scaler.scaling_decisions:
        latest_decision = auto_scaler.scaling_decisions[-1]
        print(f"\nLatest Scaling Decision:")
        print(f"  Action: {latest_decision['action']}")
        print(f"  Nodes: {latest_decision['from_nodes']} → {latest_decision['to_nodes']}")
        print(f"  Utilization: {latest_decision['utilization']:.2f}")
        print(f"  Success: {latest_decision['success']}")
    print()
    
    # Node health monitoring demo
    print("NODE HEALTH MONITORING DEMO:")
    
    health_monitor = NodeHealthMonitor()
    
    # Register current node
    node_info = NodeInfo(
        node_id="demo_node_1",
        rank=0,
        hostname=socket.gethostname(),
        ip_address=socket.gethostbyname(socket.gethostname()),
        gpu_count=torch.cuda.device_count(),
        memory_gb=psutil.virtual_memory().total / (1024**3),
        cpu_cores=psutil.cpu_count(),
        status="active",
        last_heartbeat=datetime.now(),
        workload=0.5
    )
    
    health_monitor.register_node(node_info)
    
    # Check node health
    health_status = health_monitor.check_node_health("demo_node_1")
    
    print("Node Health Status:")
    print(f"  Status: {health_status['status']}")
    print(f"  CPU Usage: {health_status.get('cpu_usage', 0):.1f}%")
    print(f"  Memory Usage: {health_status.get('memory_usage', 0):.1f}%")
    print(f"  Disk Usage: {health_status.get('disk_usage', 0):.1f}%")
    
    if health_status.get('alerts'):
        print("  Alerts:")
        for alert in health_status['alerts']:
            print(f"    ⚠️ {alert}")
    
    print()
    print("DISTRIBUTED COMPUTING DEMO COMPLETED!")
    
    return {
        'config': config,
        'training_metrics': training_metrics,
        'federated_metrics': fed_metrics if config.federated_learning else None,
        'cluster_metrics': metrics,
        'health_status': health_status
    }

if __name__ == "__main__":
    # Run distributed computing demo
    results = run_distributed_computing_demo()
