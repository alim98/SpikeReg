"""
Training utilities for SpikeReg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from aim import Repo, Run
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Optional, Tuple, List
import argparse
import yaml
import json
from contextlib import redirect_stdout

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
        multi_gpu_config: Dict = None,
        aim_repo: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # file for writing logs
        self.log_file = os.path.join(self.log_dir, 'training_log.txt')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Multi-GPU configuration
        self.multi_gpu_config = multi_gpu_config or {}
        self.use_multi_gpu = self.multi_gpu_config.get('use_multi_gpu', False)
        self.gpu_ids = self.multi_gpu_config.get('gpu_ids', None)
        self.distributed = self.multi_gpu_config.get('distributed', False)
        
        # Setup multi-GPU / distributed environment
        self._setup_multi_gpu()
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize Aim only on main process
        # Consider environments spawned via torch.distributed.run
        local_rank = os.getenv("LOCAL_RANK")
        rank = os.getenv("RANK")
        is_main_process = True
        if local_rank is not None and local_rank != '0':
            is_main_process = False
        if rank is not None and rank != '0':
            is_main_process = False
        
        self.is_main_process = is_main_process

        self.aim_run = None
        if is_main_process:
            # Priority: explicit arg > env var AIM_REPO > parent of checkpoint_dir
            self.aim_repo_path = aim_repo or os.getenv("AIM_REPO") or os.path.dirname(self.checkpoint_dir)
            self.aim_repo = Repo(self.aim_repo_path)
            self.aim_run = Run(repo=self.aim_repo)
            self.aim_run["config"] = self.config
            self.aim_run["checkpoint_dir"] = self.checkpoint_dir
            self.aim_run["log_dir"] = self.log_dir
            # Set run name to job id if provided/available
            effective_run_name = run_name or os.getenv("SLURM_JOB_ID")
            if effective_run_name:
                try:
                    # Aim SDK may support setting a user-defined name
                    self.aim_run.name = str(effective_run_name)
                except Exception:
                    # Fallback to a tracked field
                    self.aim_run["run_name"] = str(effective_run_name)
            # Always store job id field if present
            if os.getenv("SLURM_JOB_ID"):
                self.aim_run["job_id"] = os.getenv("SLURM_JOB_ID")
        else:
            with open(self.log_file, 'a') as f:
                f.write("Aim logging disabled on non-main process (rank/LOCAL_RANK != 0)\n")
        
        # Store run hash for continuation
        self.aim_run_hash = None
        
        # Initialize models
        self.pretrained_model = None
        self.spiking_model = None
        self.current_model = None
        
        # Training state (epoch-based only)
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_phase = 'pretrain'  # Track current phase: 'pretrain' or 'finetune'
        
        # Initialize components
        self._init_models()
        self._init_optimizer()
        self._init_loss()
        
        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer()
        # Transformer for discrete segmentation labels using nearest neighbor
        self.seg_spatial_transformer = SpatialTransformer(mode='nearest', padding_mode='border')
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if not torch.cuda.is_available():
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
        
        # Distributed: single GPU per process based on LOCAL_RANK
        if self.distributed:
            local_rank_env = os.getenv("LOCAL_RANK", "0")
            local_rank = int(local_rank_env)
            self.gpu_ids = [local_rank]
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            print(f"Using distributed single-GPU per process on cuda:{local_rank}")
            with open(self.log_file, 'a') as f:
                f.write(f"Using distributed single-GPU per process on cuda:{local_rank}\n")
            return

        # Non-distributed multi-GPU via DataParallel
        if not self.use_multi_gpu:
            print(f"Training on single device (no DataParallel): {self.device}")
            with open(self.log_file, 'a') as f:
                f.write(f"Training on single device (no DataParallel): {self.device}\n")
            return

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
        
        print("Using DataParallel")
        with open(self.log_file, 'a') as f:
            f.write("Using DataParallel\n")
    
    def _wrap_model_for_multi_gpu(self, model):
        """Wrap model for multi-GPU training"""
        # In distributed mode we rely on torch.distributed.run for multi-process parallelism
        if self.distributed:
            return model

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
    
    def _init_spiking_model_for_loading(self):
        """Initialize spiking model structure for loading a converted checkpoint"""
        if self.spiking_model is None:
            from SpikeReg.models import SpikeRegUNet
            self.spiking_model = SpikeRegUNet(self.config['model']).to(self.device)
            self.spiking_model = self._wrap_model_for_multi_gpu(self.spiking_model)
            self.current_model = self.spiking_model
            
            from SpikeReg.registration import IterativeRegistration
            base_spiking_model = self.spiking_model
            if hasattr(self.spiking_model, 'module'):
                base_spiking_model = self.spiking_model.module
            
            self.registration = IterativeRegistration(
                base_spiking_model,
                num_iterations=self.config['training'].get('num_iterations', 10),
                early_stop_threshold=self.config['training'].get('early_stop_threshold', 0.001)
            )
            
            self.training_phase = 'finetune'
            self._init_optimizer()
            self._init_loss()
            
            with open(self.log_file, 'a') as f:
                f.write(f"Initialized spiking model for checkpoint loading: {self.spiking_model.__class__.__name__}\n")
    
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
        # Aggregate per-component training loss averages
        train_loss_component_sums: Dict[str, float] = {}
        train_loss_component_count: int = 0
        with open(self.log_file, 'a') as f:
            f.write(f"Starting training epoch {self.epoch}\n")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        with open(os.path.join(self.log_dir, "stdout.txt"), "a") as f:
            with redirect_stdout(f):
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
                        # Log per-component even in pretrain phase
                        loss_value = float(loss.item())
                        loss_dict = {
                            'mse': loss_value,
                            'total': loss_value,
                        }
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
                        # Log only critical warnings (NaN detection) - removed heavy per-batch logging
                        if 'hasNaN' in output and output['hasNaN']:
                            with open(self.log_file, 'a') as f:
                                f.write(f"Warning: Batch {batch_idx} has NaN in displacement or warped\n")
                                    
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
                    # Aggregate per-component losses
                    for comp_name, comp_value in loss_dict.items():
                        try:
                            value_float = float(comp_value)
                        except Exception:
                            continue
                        train_loss_component_sums[comp_name] = train_loss_component_sums.get(comp_name, 0.0) + value_float
                    train_loss_component_count += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix(loss=loss_dict['total'])

        # Compute epoch statistics
        epoch_metrics['loss'] = np.mean(epoch_losses)
        # Add per-component mean losses for the epoch
        if train_loss_component_count > 0:
            for comp_name, comp_sum in train_loss_component_sums.items():
                epoch_metrics[f'loss_{comp_name}'] = comp_sum / train_loss_component_count
        
        with open(self.log_file, 'a') as f:
            f.write(f"Completed training epoch {self.epoch}\n")
            # Log only summary metrics, not all individual losses (reduces log size)
            f.write(f"Epoch {self.epoch} metrics: {json.dumps(epoch_metrics)}\n")

        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.current_model.eval()
        val_losses = []
        val_dice_scores = []
        val_jacobian_stats = []
        # Aggregate per-component validation loss averages
        val_loss_component_sums: Dict[str, float] = {}
        val_loss_component_count: int = 0
        with open(self.log_file, 'a') as f:
            f.write(f"Starting validation epoch {self.epoch}\n")
        
        with torch.no_grad():
            with open(os.path.join(self.log_dir, "stdout.txt"), "a") as f:
                with redirect_stdout(f):
                    for batch in tqdm(val_loader, desc="Validation"):
                        # Get data
                        fixed = batch['fixed'].to(self.device)
                        moving = batch['moving'].to(self.device)
                        
                        # Forward pass
                        if self.current_model == self.pretrained_model:
                            displacement = self.current_model(fixed, moving)
                            warped = self.spatial_transformer(moving, displacement)
                            loss = self.criterion(warped, fixed)
                            loss_value = float(loss.item())
                            val_losses.append(loss_value)
                            # record val components for pretrain
                            val_loss_dict = {'mse': loss_value, 'total': loss_value}
                            try:
                                for comp_name, comp_value in val_loss_dict.items():
                                    value_float = float(comp_value)
                                    val_loss_component_sums[comp_name] = val_loss_component_sums.get(comp_name, 0.0) + value_float
                                val_loss_component_count += 1
                            except Exception:
                                pass
                        else:
                            output = self.registration(fixed, moving)
                            displacement = output['displacement']
                            warped = output['warped']
                            
                            # Compute loss
                            loss, val_loss_dict = self.criterion(
                                fixed, moving, displacement, warped, 
                                output.get('spike_counts', {})
                            )
                            val_losses.append(loss.item())
                            # aggregate per-component validation losses if provided
                            try:
                                for comp_name, comp_value in val_loss_dict.items():
                                    value_float = float(comp_value.item() if torch.is_tensor(comp_value) else comp_value)
                                    val_loss_component_sums[comp_name] = val_loss_component_sums.get(comp_name, 0.0) + value_float
                                val_loss_component_count += 1
                            except Exception:
                                pass
                            with open(self.log_file, 'a') as f:
                                f.write(f"Validation output similarity score: {output['similarity_scores'].tolist()}\n")
                                f.write(f"Spike counts number: {output['spike_count_history_number']}\n")
                                if 'hasNaN' in output and output['hasNaN']:
                                    f.write(f"Warning: Has NaN in displacement or warped, displacement: {output['displacement']}\n")
                        
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
        # Add per-component mean validation losses for the epoch
        if val_loss_component_count > 0:
            for comp_name, comp_sum in val_loss_component_sums.items():
                metrics[f'val_loss_{comp_name}'] = comp_sum / val_loss_component_count

        with open(self.log_file, 'a') as f:
            f.write(f"Completed validation epoch {self.epoch}\n")
            # Log only summary metrics, not all individual losses (reduces log size)
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

        # Continue from the NEXT epoch if resuming (current epoch is already completed)
        start_epoch = self.epoch + 1
        if self.epoch >= 0:
            print(f"📊 Resuming training from epoch {start_epoch} (checkpoint was saved after completing epoch {self.epoch})")
        
        with open(self.log_file, 'a') as f:
            f.write(f"Resuming training from epoch {start_epoch} (completed epoch {self.epoch}), continuing to {num_epochs}\n")
        
        for epoch in range(start_epoch, num_epochs):
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
            
            prefix = "pretrain" if self.training_phase == "pretrain" else "finetune"
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(f'{prefix}_best_model.pth')
            
            # Regular checkpoint
            if epoch % self.config['training'].get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(f'{prefix}_epoch_{epoch}.pth')
                # Cleanup old checkpoints to save disk space
                # self._cleanup_old_checkpoints()
            
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
        
        # Switch to finetune phase
        self.training_phase = 'finetune'
        
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
        """Save model checkpoint (only on main process to avoid race conditions)"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'training_phase': self.training_phase,
            'model_state_dict': self.current_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'aim_run_hash': self.aim_run.hash if hasattr(self.aim_run, 'hash') else None,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self.checkpoint_dir, filename)
        
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved checkpoint: {path}")
        with open(self.log_file, 'a') as f:
            f.write(f"Saved checkpoint: {path}\n")
    
    # def _cleanup_old_checkpoints(self):
    #     """Remove old epoch checkpoints, keeping only the most recent N"""
    #     keep_last = self.config['training'].get('save_keep_last', 5)
    #     if keep_last <= 0:
    #         return  # Keep all checkpoints
        
    #     # Find all epoch checkpoints
    #     import glob
    #     import re
    #     checkpoint_pattern = os.path.join(self.checkpoint_dir, 'model_epoch_*.pth')
    #     checkpoints = glob.glob(checkpoint_pattern)
        
    #     # Extract epoch numbers and sort
    #     epoch_ckpts = []
    #     for ckpt in checkpoints:
    #         match = re.search(r'model_epoch_(\d+)\.pth', ckpt)
    #         if match:
    #             epoch_num = int(match.group(1))
    #             epoch_ckpts.append((epoch_num, ckpt))
        
    #     # Sort by epoch number (newest first)
    #     epoch_ckpts.sort(reverse=True)
        
    #     # Remove old checkpoints beyond keep_last
    #     for epoch_num, ckpt_path in epoch_ckpts[keep_last:]:
    #         try:
    #             os.remove(ckpt_path)
    #             with open(self.log_file, 'a') as f:
    #                 f.write(f"Removed old checkpoint: {ckpt_path}\n")
    #         except Exception as e:
    #             with open(self.log_file, 'a') as f:
    #                 f.write(f"Failed to remove checkpoint {ckpt_path}: {e}\n")
    def load_checkpoint(self, filename: str):
        """Load model checkpoint with staggered delays to avoid concurrent file access race conditions"""
        import sys, numpy as np
        import time
        
        if not self.is_main_process:
            rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
            rank_num = int(rank) if rank.isdigit() else 0
            time.sleep(0.5 + 0.1 * rank_num)
        
        # Handle both relative and absolute paths
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self.checkpoint_dir, filename)
        
        # Resolve symlinks and normalize path
        path = os.path.realpath(path)
        
        # Check if file exists before attempting to load
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}. Please check the checkpoint path.")
        
        print(f"Loading checkpoint from: {path}")
        
        # 0) legacy NumPy aliases MUST be defined before torch.load
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        
        # keep weights_only=False for old pickles (trusted source)
        # Add retry logic for file access issues or partially written archives
        max_retries = 5
        for attempt in range(max_retries):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                break
            except (EOFError, IOError, OSError, RuntimeError) as e:
                msg = str(e)
                non_retriable = isinstance(e, RuntimeError) and "PytorchStreamReader failed reading zip archive" not in msg and "failed finding central directory" not in msg
                if non_retriable or attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
                print(f"Retry {attempt + 1}/{max_retries} loading checkpoint due to: {e}")

        # 1) choose the state dict
        state = checkpoint.get("model_state_dict", checkpoint)

        # 2) Handle DataParallel/DDP prefix
        # Check if current model is wrapped in DataParallel
        is_wrapped = isinstance(self.current_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
        
        # Check if checkpoint has module prefix
        has_module_prefix = any(k.startswith("module.") for k in state.keys())
        
        if has_module_prefix and not is_wrapped:
            # Checkpoint has prefix but model doesn't - strip it
            state = {k[len("module."):]: v for k, v in state.items()}
        elif not has_module_prefix and is_wrapped:
            # Checkpoint doesn't have prefix but model does - add it
            state = {"module." + k: v for k, v in state.items()}

        # 2.5) Check for channel mismatches in decoder blocks BEFORE loading
        model_dict = self.current_model.state_dict()
        keys_to_remove = []
        for key in list(state.keys()):
            if 'decoder_blocks' in key and 'upconv' in key and 'weight' in key:
                if key in model_dict:
                    checkpoint_shape = state[key].shape
                    model_shape = model_dict[key].shape
                    if checkpoint_shape != model_shape:
                        print(f"[load_checkpoint] Channel mismatch in {key}: checkpoint {checkpoint_shape} vs model {model_shape}. Skipping this weight (layer will use default initialization).")
                        keys_to_remove.append(key)
                        # Also remove related bias and bn weights
                        bias_key = key.replace('.weight', '.bias')
                        bn_weight_key = key.replace('.conv.weight', '.bn.weight')
                        bn_bias_key = key.replace('.conv.weight', '.bn.bias')
                        bn_mean_key = key.replace('.conv.weight', '.bn.running_mean')
                        bn_var_key = key.replace('.conv.weight', '.bn.running_var')
                        for related_key in [bias_key, bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]:
                            if related_key in state:
                                keys_to_remove.append(related_key)
        
        for key in keys_to_remove:
            if key in state:
                del state[key]

        # 3) load weights ONCE (lenient)
        missing, unexpected = self.current_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[load_checkpoint] missing:", missing)
            print("[load_checkpoint] unexpected:", unexpected)

        # 4) restore training state (guarded)
        self.epoch         = int(checkpoint.get('epoch', 0))
        self.best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
        self.training_phase = checkpoint.get('training_phase', 'pretrain')  # Default to pretrain if not saved

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print("[load_checkpoint] optimizer state load skipped:", e)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print("[load_checkpoint] scheduler state load skipped:", e)

        print(f"Loaded checkpoint from epoch {self.epoch}")
        with open(self.log_file, 'a') as f:
            f.write(f"Loaded checkpoint from epoch {self.epoch}\n")
        
        # Store parent run hash as a field for traceability (avoid unsupported reopen-by-hash)
        if self.aim_run is not None and 'aim_run_hash' in checkpoint and checkpoint['aim_run_hash']:
            try:
                self.aim_run["parent_run_hash"] = str(checkpoint['aim_run_hash'])
            except Exception:
                pass

    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch metrics to Aim with phase separation"""
        if self.aim_run is None:
            return
        
        # Log with phase prefix for clear separation in Aim
        phase = self.training_phase
        
        # Training metrics with phase
        for name, value in train_metrics.items():
            # name could be 'loss' or 'loss_total' or 'loss_similarity' etc.
            self.aim_run.track(value, name=f'{phase}/train_{name}', step=self.epoch, context={'phase': phase})

        # Validation metrics with phase
        for name, value in val_metrics.items():
            self.aim_run.track(value, name=f'{phase}/{name}', step=self.epoch, context={'phase': phase})
        
        # Log learning rate with phase
        current_lr = self.optimizer.param_groups[0]['lr']
        self.aim_run.track(current_lr, name=f'{phase}/learning_rate', step=self.epoch, context={'phase': phase})
        
        # Also log to generic epoch metrics for overall tracking
        for name, value in train_metrics.items():
            self.aim_run.track(value, name=f'epoch/train_{name}', step=self.epoch)
        for name, value in val_metrics.items():
            self.aim_run.track(value, name=f'epoch/{name}', step=self.epoch)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 


def _create_loaders_from_config(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create OASIS dataloaders based on config paths and settings."""
    from examples.oasis_dataset import create_oasis_loaders

    data_cfg = config.get('data', {})
    train_dir = data_cfg.get('train_dir')
    val_dir = data_cfg.get('val_dir', train_dir)
    batch_size = int(config.get('training', {}).get('batch_size', 2))
    patch_size = int(data_cfg.get('patch_size', config.get('model', {}).get('patch_size', 32)))
    patch_stride = int(data_cfg.get('patch_stride', 16))
    num_workers = int(config.get('training', {}).get('num_workers', 4))

    # The dataset factory expects data_root to be the parent of L2R_2021_Task3_train
    # If train_dir is /u/almik/SpikeReg2/data/L2R_2021_Task3_train, then data_root should be /u/almik/SpikeReg2/data
    if train_dir.endswith('/L2R_2021_Task3_train'):
        data_root = train_dir[:-len('/L2R_2021_Task3_train')]
    else:
        # Fallback: assume train_dir is the data root
        data_root = train_dir

    # Get additional training config for data loading optimization
    train_cfg = config.get('training', {})
    pin_memory = train_cfg.get('pin_memory', True)
    prefetch_factor = train_cfg.get('prefetch_factor', 2)
    
    train_loader, val_loader = create_oasis_loaders(
        data_root,
        batch_size=batch_size,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=int(config.get('data', {}).get('patches_per_pair', 20)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, val_loader


def main_cli():
    """Minimal CLI to run training using the YAML config without overriding it."""
    parser = argparse.ArgumentParser(description='SpikeReg Training')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint output directory')
    parser.add_argument('--log_dir', required=True, help='Log output directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--name', help='Run name (e.g., SLURM job id)')
    parser.add_argument('--start_from_checkpoint', help='Path to checkpoint to resume from')
    parser.add_argument('--aim_repo', type=str, default=None, help='Path to Aim repository (overrides AIM_REPO env)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build data loaders strictly from config
    train_loader, val_loader = _create_loaders_from_config(cfg)

    # Multi-GPU config: rely on config flags if present; default to DataParallel via trainer
    multi_gpu_cfg = cfg.get('multi_gpu', {}) if isinstance(cfg.get('multi_gpu', {}), dict) else {}

    trainer = SpikeRegTrainer(
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        multi_gpu_config=multi_gpu_cfg,
        aim_repo=args.aim_repo,
        run_name=args.name,
    )

    # Resume from checkpoint if provided
    if args.start_from_checkpoint:
        print(f"Resuming from checkpoint: {args.start_from_checkpoint}")
        trainer.load_checkpoint(args.start_from_checkpoint)
        print(f"Resumed from epoch {trainer.epoch}")

    # Training phases per config
    pretrain_epochs = int(cfg.get('training', {}).get('pretrain_epochs', 0))
    finetune_epochs = int(cfg.get('training', {}).get('finetune_epochs', 0))
    do_pretrain = bool(cfg.get('training', {}).get('pretrain', True))

    # Check if pretrain is already complete
    pretrained_model_path = os.path.join(trainer.checkpoint_dir, 'pretrained_model.pth')
    converted_model_path = os.path.join(trainer.checkpoint_dir, 'converted_model.pth')
    final_model_path = os.path.join(trainer.checkpoint_dir, 'final_model.pth')
    
    pretrain_complete = os.path.exists(pretrained_model_path)
    converted_complete = os.path.exists(converted_model_path)
    training_complete = os.path.exists(final_model_path)
    
    if training_complete:
        print("Training already complete! Found final_model.pth")
        return
    
    # If epochs=0, run indefinitely until timeout
    if pretrain_epochs == 0 and finetune_epochs == 0:
        print("Running training indefinitely until timeout...")
        trainer.train(train_loader, val_loader, num_epochs=999999)  # Large number to run until timeout
    else:
        # Check if we need to run pretrain
        if do_pretrain and trainer.pretrained_model is not None and pretrain_epochs > 0:
            if pretrain_complete:
                print(f"Pretrain already complete! Found {pretrained_model_path}")
                print("Skipping pretrain phase, loading pretrained model...")
                trainer.load_checkpoint('pretrained_model.pth')
                
                if not converted_complete:
                    print("Converting pretrained model to spiking...")
                    trainer.convert_to_spiking()
                    trainer.save_checkpoint('converted_model.pth')
                else:
                    print(f"Conversion already complete! Found {converted_model_path}")
                    print("Checking if converted checkpoint architecture matches current model...")
                    trainer._init_spiking_model_for_loading()
                    
                    checkpoint = torch.load(converted_model_path, map_location='cpu', weights_only=False)
                    ckpt_state = checkpoint.get("model_state_dict", checkpoint)
                    model_state = trainer.spiking_model.state_dict()
                    
                    architecture_mismatch = False
                    for ckpt_key in list(ckpt_state.keys()):
                        if 'decoder_blocks' in ckpt_key and 'upconv' in ckpt_key and 'weight' in ckpt_key:
                            model_key = ckpt_key
                            if ckpt_key.startswith('module.') and not any(k.startswith('module.') for k in model_state.keys()):
                                model_key = ckpt_key[len('module.'):]
                            elif not ckpt_key.startswith('module.') and any(k.startswith('module.') for k in model_state.keys()):
                                model_key = 'module.' + ckpt_key
                            
                            if model_key in model_state:
                                if ckpt_state[ckpt_key].shape != model_state[model_key].shape:
                                    print(f"Architecture mismatch detected in {ckpt_key}: checkpoint {ckpt_state[ckpt_key].shape} vs model {model_state[model_key].shape}")
                                    architecture_mismatch = True
                                    break
                    
                    if architecture_mismatch:
                        print("Converted checkpoint has old architecture. Re-converting from pretrained model...")
                        trainer.convert_to_spiking()
                        trainer.save_checkpoint('converted_model.pth')
                    else:
                        trainer.load_checkpoint('converted_model.pth')
            else:
                # Run pretrain
                trainer.train(train_loader, val_loader, num_epochs=pretrain_epochs)
                trainer.save_checkpoint('pretrained_model.pth')
                trainer.convert_to_spiking()
                trainer.save_checkpoint('converted_model.pth')

        # Fine-tune spiking model
        if finetune_epochs > 0:
            # Check if spiking model is initialized (either from conversion or direct init)
            if trainer.spiking_model is None:
                if converted_complete:
                    print(f"Loading converted model for finetune: {converted_model_path}")
                    print("Initializing spiking model for checkpoint loading...")
                    trainer._init_spiking_model_for_loading()
                    
                    checkpoint = torch.load(converted_model_path, map_location='cpu', weights_only=False)
                    ckpt_state = checkpoint.get("model_state_dict", checkpoint)
                    model_state = trainer.spiking_model.state_dict()
                    
                    architecture_mismatch = False
                    for ckpt_key in list(ckpt_state.keys()):
                        if 'decoder_blocks' in ckpt_key and 'upconv' in ckpt_key and 'weight' in ckpt_key:
                            model_key = ckpt_key
                            if ckpt_key.startswith('module.') and not any(k.startswith('module.') for k in model_state.keys()):
                                model_key = ckpt_key[len('module.'):]
                            elif not ckpt_key.startswith('module.') and any(k.startswith('module.') for k in model_state.keys()):
                                model_key = 'module.' + ckpt_key
                            
                            if model_key in model_state:
                                if ckpt_state[ckpt_key].shape != model_state[model_key].shape:
                                    print(f"Architecture mismatch detected in {ckpt_key}: checkpoint {ckpt_state[ckpt_key].shape} vs model {model_state[model_key].shape}")
                                    architecture_mismatch = True
                                    break
                    
                    if architecture_mismatch:
                        print("Converted checkpoint has old architecture. Re-converting from pretrained model...")
                        if pretrain_complete:
                            trainer.load_checkpoint('pretrained_model.pth')
                            trainer.convert_to_spiking()
                            trainer.save_checkpoint('converted_model.pth')
                        else:
                            print("Warning: Cannot re-convert - pretrained model not found. Loading checkpoint with mismatched layers...")
                            trainer.load_checkpoint('converted_model.pth')
                    else:
                        trainer.load_checkpoint('converted_model.pth')
                elif pretrain_complete:
                    print(f"Loading pretrained model and converting: {pretrained_model_path}")
                    trainer.load_checkpoint('pretrained_model.pth')
                    trainer.convert_to_spiking()
                    trainer.save_checkpoint('converted_model.pth')
                else:
                    print("Warning: No pretrained/converted model found for finetune. Skipping finetune.")
                    return
            
            # Reset epoch counter for finetune phase to start from 0
            # This ensures finetune trains epochs 0 to (finetune_epochs-1)
            trainer.epoch = -1
            trainer.train(train_loader, val_loader, num_epochs=finetune_epochs)
            trainer.save_checkpoint('final_model.pth')


if __name__ == '__main__':
    main_cli()