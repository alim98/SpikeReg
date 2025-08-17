"""
Training utilities for SpikeReg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Optional, Tuple, List
import yaml
import json

from .models import SpikeRegUNet, PretrainedUNet, convert_pretrained_to_spiking
from .registration import IterativeRegistration
from .losses import SpikeRegLoss
from .utils.warping import SpatialTransformer
from .utils.metrics import dice_score, jacobian_determinant_stats


class SpikeRegTrainer:
    """
    Trainer class for SpikeReg models
    
    Handles pretraining, conversion, and fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = "cuda",
        multi_gpu_config: Dict = None
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Multi-GPU configuration
        self.multi_gpu_config = multi_gpu_config or {}
        self.use_multi_gpu = self.multi_gpu_config.get('use_multi_gpu', False)
        self.gpu_ids = self.multi_gpu_config.get('gpu_ids', None)
        self.distributed = self.multi_gpu_config.get('distributed', False)
        
        # Setup multi-GPU environment
        self._setup_multi_gpu()
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # Initialize models
        self.pretrained_model = None
        self.spiking_model = None
        self.current_model = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize components
        self._init_models()
        self._init_optimizer()
        self._init_loss()
        
        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer()
        # Transformer for discrete segmentation labels using nearest neighbor
        self.seg_spatial_transformer = SpatialTransformer(mode='nearest', padding_mode='border')

        # file for writing logs
        self.log_file = os.path.join(self.log_dir, 'training_log.txt')
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        if not self.use_multi_gpu or not torch.cuda.is_available():
            print(f"Training on single device: {self.device}")
            with open(self.log_file, 'a') as f:
                f.write(f"Training on single device: {self.device}\n")
            return
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"Warning: Only {num_gpus} GPU available, disabling multi-GPU training")
            self.use_multi_gpu = False
            with open(self.log_file, 'a') as f:
                f.write(f"Warning: Only {num_gpus} GPU available, disabling multi-GPU training\n")
            return
        
        # Setup GPU IDs
        if self.gpu_ids is None:
            self.gpu_ids = list(range(num_gpus))
        else:
            self.gpu_ids = [int(id) for id in self.gpu_ids]
        
        # Validate GPU IDs
        for gpu_id in self.gpu_ids:
            if gpu_id >= num_gpus:
                with open(self.log_file, 'a') as f:
                    f.write(f"Error: GPU ID {gpu_id} not available. Only {num_gpus} GPUs detected.\n")
                raise ValueError(f"GPU ID {gpu_id} not available. Only {num_gpus} GPUs detected.")
        
        print(f"Setting up multi-GPU training on GPUs: {self.gpu_ids}")
        with open(self.log_file, 'a') as f:
            f.write(f"Setting up multi-GPU training on GPUs: {self.gpu_ids}\n")
        
        # Set primary GPU
        if 'cuda' in str(self.device):
            torch.cuda.set_device(self.gpu_ids[0])
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
        
        if self.distributed:
            print("Using DistributedDataParallel")
            with open(self.log_file, 'a') as f:
                f.write("Using DistributedDataParallel\n")
            # Note: Full DDP setup would require additional initialization
            # For now, we'll use DataParallel as it's simpler
            print("Warning: DistributedDataParallel not fully implemented, falling back to DataParallel")
        else:
            print("Using DataParallel")
            with open(self.log_file, 'a') as f:
                f.write("Using DataParallel\n")
    
    def _wrap_model_for_multi_gpu(self, model):
        """Wrap model for multi-GPU training"""
        if not self.use_multi_gpu or len(self.gpu_ids) < 2:
            return model
        
        if self.distributed:
            # TODO: Implement DistributedDataParallel
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=self.gpu_ids)
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
        
        print(f"Model wrapped for multi-GPU training on devices: {self.gpu_ids}")
        with open(self.log_file, 'a') as f:
            f.write(f"Model wrapped for multi-GPU training on devices: {self.gpu_ids}\n")
        return model
    
    def _init_models(self):
        """Initialize models based on training phase"""
        if self.config['training']['pretrain']:
            # Initialize pretrained U-Net
            self.pretrained_model = PretrainedUNet(self.config['model']).to(self.device)
            with open(self.log_file, 'a') as f:
                f.write(f"Initialized pretrained model: {self.pretrained_model.__class__.__name__}\n")
            # Wrap for multi-GPU if enabled
            self.pretrained_model = self._wrap_model_for_multi_gpu(self.pretrained_model)
            self.current_model = self.pretrained_model
        else:
            # Initialize spiking model directly
            self.spiking_model = SpikeRegUNet(self.config['model']).to(self.device)
            # Wrap for multi-GPU if enabled
            self.spiking_model = self._wrap_model_for_multi_gpu(self.spiking_model)
            self.current_model = self.spiking_model
            
            # Create iterative registration wrapper
            self.registration = IterativeRegistration(
                self.spiking_model,
                num_iterations=self.config['training'].get('num_iterations', 10),
                early_stop_threshold=self.config['training'].get('early_stop_threshold', 0.001)
            )
            with open(self.log_file, 'a') as f:
                f.write(f"Initialized spiking model: {self.spiking_model.__class__.__name__}\n")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.current_model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.current_model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
        with open(self.log_file, 'a') as f:
            f.write(f"Initialized optimizer: {optimizer_config['type']} with lr={optimizer_config['lr']}\n")
        
        # Learning rate scheduler, casting config values to correct types
        scheduler_config = self.config['training'].get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            T_max = int(scheduler_config.get('T_max', 100))
            eta_min = float(scheduler_config.get('eta_min', 1e-6))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        elif scheduler_config.get('type') == 'step':
            step_size = int(scheduler_config.get('step_size', 30))
            gamma = float(scheduler_config.get('gamma', 0.1))
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            self.scheduler = None
        with open(self.log_file, 'a') as f:
            if self.scheduler:
                f.write(f"Initialized scheduler: {scheduler_config.get('type', 'none')}\n")
            else:
                f.write("No scheduler initialized\n")
    
    def _init_loss(self):
        """Initialize loss function"""
        loss_config = self.config['training']['loss']
        
        if self.current_model == self.pretrained_model:
            # Simple loss for pretraining
            self.criterion = nn.MSELoss()
            with open(self.log_file, 'a') as f:
                f.write("Using MSE loss for pretrained model\n")
        else:
            # Full SpikeReg loss
            self.criterion = SpikeRegLoss(
                similarity_type=loss_config.get('similarity_type', 'ncc'),
                similarity_weight=loss_config.get('similarity_weight', 1.0),
                regularization_type=loss_config.get('regularization_type', 'bending'),
                regularization_weight=loss_config.get('regularization_weight', 0.01),
                spike_weight=loss_config.get('spike_weight', 0.001),
                spike_balance_weight=loss_config.get('spike_balance_weight', 0.01),
                target_spike_rate=loss_config.get('target_spike_rate', 0.1)
            )
        with open(self.log_file, 'a') as f:
            f.write(f"Initialized loss function: {self.criterion.__class__.__name__}\n")
            f.write(f"Loss config: {json.dumps(loss_config, indent=2)}\n")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.current_model.train()
        epoch_losses = []
        epoch_metrics = {}
        with open(self.log_file, 'a') as f:
            f.write(f"Starting training epoch {self.epoch}\n")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            fixed = batch['fixed'].to(self.device)
            moving = batch['moving'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.current_model == self.pretrained_model:
                # Pretrained model - direct prediction
                displacement = self.current_model(fixed, moving)
                warped = self.spatial_transformer(moving, displacement)
                
                # Simple MSE loss
                loss = self.criterion(warped, fixed)
                loss_dict = {'total': loss.item()}
                spike_counts = {}
                
            else:
                # Spiking model - iterative registration
                output = self.registration(fixed, moving, return_all_iterations=False)
                displacement = output['displacement']
                warped = output['warped']
                if 'spike_count_history' in output:
                    spike_counts = output['spike_count_history'][-1] if output['spike_count_history'] else {}
                else:
                    spike_counts = output.get('spike_counts', {})
                
                # Full loss computation
                loss, loss_dict = self.criterion(
                    fixed, moving, displacement, warped, spike_counts
                )
                # Ensure loss_dict values are plain Python floats for safe logging / NumPy ops
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        loss_dict[k] = v.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.current_model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics (total is now float)
            epoch_losses.append(float(loss_dict['total']))
            
            # Log to tensorboard
            if self.global_step % self.config['training'].get('log_interval', 10) == 0:
                self._log_training_step(loss_dict, spike_counts)
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss_dict['total'])
            
            self.global_step += 1

        # Compute epoch statistics
        epoch_metrics['loss'] = np.mean(epoch_losses)
        
        with open(self.log_file, 'a') as f:
            f.write(f"Completed training epoch {self.epoch}\n")
            f.write(f"Epoch {self.epoch} losses: {json.dumps(epoch_losses)}\n")
            f.write(f"Epoch {self.epoch} metrics: {json.dumps(epoch_metrics)}\n")

        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.current_model.eval()
        val_losses = []
        val_dice_scores = []
        val_jacobian_stats = []
        with open(self.log_file, 'a') as f:
            f.write(f"Starting validation epoch {self.epoch}\n")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                fixed = batch['fixed'].to(self.device)
                moving = batch['moving'].to(self.device)
                
                # Forward pass
                if self.current_model == self.pretrained_model:
                    displacement = self.current_model(fixed, moving)
                    warped = self.spatial_transformer(moving, displacement)
                    loss = self.criterion(warped, fixed)
                    val_losses.append(loss.item())
                else:
                    output = self.registration(fixed, moving)
                    displacement = output['displacement']
                    warped = output['warped']
                    
                    # Compute loss
                    loss, _ = self.criterion(
                        fixed, moving, displacement, warped, 
                        output.get('spike_counts', {})
                    )
                    val_losses.append(loss.item())
                
                # Compute metrics
                if 'segmentation_fixed' in batch and 'segmentation_moving' in batch:
                    seg_fixed = batch['segmentation_fixed'].to(self.device)
                    seg_moving = batch['segmentation_moving'].to(self.device)
                    # Warp segmentation labels with nearest neighbor interpolation
                    seg_warped = self.seg_spatial_transformer(seg_moving.float(), displacement).long()
                    
                    # Dice score
                    dice = dice_score(seg_warped, seg_fixed)
                    # average over batch and classes to get a single score
                    val_dice_scores.append(dice.mean().item())
                
                # Jacobian determinant statistics
                jac_stats = jacobian_determinant_stats(displacement)
                val_jacobian_stats.append(jac_stats)
        
        # Compute validation metrics
        metrics = {
            'val_loss': np.mean(val_losses),
            'val_dice': np.mean(val_dice_scores) if val_dice_scores else 0.0,
            'val_jac_negative': np.mean([s['negative_fraction'] for s in val_jacobian_stats])
        }

        with open(self.log_file, 'a') as f:
            f.write(f"Completed validation epoch {self.epoch}\n")
            f.write(f"Validation losses: {json.dumps(val_losses)}\n")
            f.write(f"Validation metrics: {json.dumps(metrics)}\n")
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """Main training loop"""
        print(f"Training on {self.device}")

        with open(self.log_file, 'a') as f:
            f.write(f"Starting training for {num_epochs} epochs\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if epoch % self.config['training'].get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(f'model_epoch_{epoch}.pth')
            
            # Print epoch summary
            print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Dice: {val_metrics['val_dice']:.4f}")
            
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, "
                        f"Val Dice: {val_metrics['val_dice']:.4f}\n")
        
    def convert_to_spiking(self):
        """Convert pretrained model to spiking"""
        if self.pretrained_model is None:
            raise ValueError("No pretrained model to convert")
        
        print("Converting pretrained model to spiking...")
        with open(self.log_file, 'a') as f:
            f.write("Converting pretrained model to spiking...\n")
        
        # Get the underlying model if wrapped with DataParallel
        base_model = self.pretrained_model
        if hasattr(self.pretrained_model, 'module'):
            base_model = self.pretrained_model.module
        
        # Convert model
        self.spiking_model = convert_pretrained_to_spiking(
            base_model,
            self.config['model'],
            threshold_percentile=self.config['conversion'].get('threshold_percentile', 99.0)
        ).to(self.device)
        
        # Wrap for multi-GPU if enabled
        self.spiking_model = self._wrap_model_for_multi_gpu(self.spiking_model)
        
        # Update current model
        self.current_model = self.spiking_model
        
        # Reinitialize optimizer and loss for spiking model
        self._init_optimizer()
        self._init_loss()
        
        # Create registration wrapper (using base model for registration)
        base_spiking_model = self.spiking_model
        if hasattr(self.spiking_model, 'module'):
            base_spiking_model = self.spiking_model.module
        
        self.registration = IterativeRegistration(
            base_spiking_model,
            num_iterations=self.config['training'].get('num_iterations', 10),
            early_stop_threshold=self.config['training'].get('early_stop_threshold', 0.001)
        )

        with open(self.log_file, 'a') as f:
            f.write(f"Converted to spiking model: {self.spiking_model.__class__.__name__}\n")
        with open(self.log_file, 'a') as f:
            f.write(f"Spiking model config: {json.dumps(self.config['model'], indent=2)}\n")
        
        print("Conversion complete!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.current_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        with open(self.log_file, 'a') as f:
            f.write(f"Saved checkpoint: {path}\n")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load model weights
        self.current_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
        with open(self.log_file, 'a') as f:
            f.write(f"Loaded checkpoint from epoch {self.epoch}\n")
    
    def _log_training_step(self, loss_dict: Dict[str, float], spike_counts: Dict[str, float]):
        """Log training step to tensorboard"""
        # Log losses
        for name, value in loss_dict.items():
            self.writer.add_scalar(f'train/loss_{name}', value, self.global_step)
        
        # Log spike counts
        for layer, count in spike_counts.items():
            self.writer.add_scalar(f'train/spikes_{layer}', count, self.global_step)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)


            ### !!!
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch metrics to tensorboard"""
        # Training metrics
        for name, value in train_metrics.items():
            self.writer.add_scalar(f'epoch/train_{name}', value, self.epoch)
        
        # Validation metrics
        for name, value in val_metrics.items():
            self.writer.add_scalar(f'epoch/{name}', value, self.epoch)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 