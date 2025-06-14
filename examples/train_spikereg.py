#!/usr/bin/env python3
"""
Example training script for SpikeReg

This script demonstrates the complete training pipeline:
1. Pretraining with standard U-Net
2. Converting to spiking neural network
3. Fine-tuning with spike-aware training
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikereg import SpikeRegTrainer
from spikereg.training import load_config, save_config
from utils.preprocessing import normalize_volume, PatchNormalizer
from utils.patch_utils import extract_patches, PatchAugmentor


class MedicalVolumeDataset(Dataset):
    """
    Example dataset for medical volume pairs
    
    This is a placeholder - replace with your actual data loading logic
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 32,
        patch_stride: int = 16,
        augment: bool = False,
        max_patches_per_volume: int = 50
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.augment = augment
        self.max_patches_per_volume = max_patches_per_volume
        
        # Find all volume pairs
        self.volume_pairs = self._find_volume_pairs()
        
        # Initialize augmentor
        if augment:
            self.augmentor = PatchAugmentor()
        
        # Extract all patches
        self.patches = []
        self._extract_all_patches()
    
    def _find_volume_pairs(self):
        """Find fixed and moving volume pairs"""
        pairs = []
        
        # Look for files with pattern: fixed_*.nii.gz, moving_*.nii.gz
        fixed_files = sorted(self.data_dir.glob("fixed_*.nii.gz"))
        
        for fixed_file in fixed_files:
            # Find corresponding moving file
            volume_id = fixed_file.stem.replace("fixed_", "")
            moving_file = self.data_dir / f"moving_{volume_id}.nii.gz"
            
            if moving_file.exists():
                pairs.append({
                    'fixed': fixed_file,
                    'moving': moving_file,
                    'id': volume_id
                })
        
        return pairs
    
    def _extract_all_patches(self):
        """Extract patches from all volume pairs"""
        print(f"Extracting patches from {len(self.volume_pairs)} volume pairs...")
        
        for pair in self.volume_pairs:
            # Load volumes (placeholder - use nibabel or SimpleITK in practice)
            fixed_vol = self._load_volume(pair['fixed'])
            moving_vol = self._load_volume(pair['moving'])
            
            # Convert to tensors
            fixed_tensor = torch.from_numpy(fixed_vol).float().unsqueeze(0).unsqueeze(0)
            moving_tensor = torch.from_numpy(moving_vol).float().unsqueeze(0).unsqueeze(0)
            
            # Normalize
            fixed_tensor = normalize_volume(fixed_tensor)
            moving_tensor = normalize_volume(moving_tensor)
            
            # Extract patches
            fixed_patches, coords = extract_patches(
                fixed_tensor, self.patch_size, self.patch_stride
            )
            moving_patches, _ = extract_patches(
                moving_tensor, self.patch_size, self.patch_stride
            )
            
            # Randomly sample patches if too many
            if len(fixed_patches) > self.max_patches_per_volume:
                indices = np.random.choice(
                    len(fixed_patches), 
                    self.max_patches_per_volume, 
                    replace=False
                )
                fixed_patches = [fixed_patches[i] for i in indices]
                moving_patches = [moving_patches[i] for i in indices]
            
            # Store patches
            for fixed_patch, moving_patch in zip(fixed_patches, moving_patches):
                self.patches.append({
                    'fixed': fixed_patch.squeeze(0),  # Remove batch dim
                    'moving': moving_patch.squeeze(0),
                    'volume_id': pair['id']
                })
        
        print(f"Extracted {len(self.patches)} patches total")
    
    def _load_volume(self, path):
        """Load medical volume - placeholder implementation"""
        # In practice, use nibabel or SimpleITK:
        # import nibabel as nib
        # return nib.load(path).get_fdata()
        
        # For demo, return random volume
        return np.random.randn(128, 128, 128).astype(np.float32)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_data = self.patches[idx]
        
        fixed = patch_data['fixed']
        moving = patch_data['moving']
        
        # Apply augmentation if enabled
        if self.augment and self.augmentor:
            # Augment both with same transformation
            combined = torch.cat([fixed, moving], dim=0).unsqueeze(0)
            augmented = self.augmentor.augment(combined)
            fixed = augmented[0, 0:1]
            moving = augmented[0, 1:2]
        
        return {
            'fixed': fixed,
            'moving': moving,
            'volume_id': patch_data['volume_id']
        }


def create_data_loaders(config):
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = MedicalVolumeDataset(
        config['data']['train_dir'],
        patch_size=config['data']['patch_size'],
        patch_stride=config['data']['patch_stride'],
        augment=True
    )
    
    val_dataset = MedicalVolumeDataset(
        config['data']['val_dir'],
        patch_size=config['data']['patch_size'],
        patch_stride=config['data']['patch_stride'],
        augment=False
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train SpikeReg model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Save configuration to checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_config(config, os.path.join(args.checkpoint_dir, 'config.yaml'))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = SpikeRegTrainer(
        config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training pipeline
    if config['training']['pretrain'] and trainer.pretrained_model is not None:
        # Phase 1: Pretrain standard U-Net
        print("\n=== Phase 1: Pretraining Standard U-Net ===")
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['training']['pretrain_epochs']
        )
        
        # Save pretrained model
        trainer.save_checkpoint('pretrained_model.pth')
        
        # Phase 2: Convert to SNN
        print("\n=== Phase 2: Converting to Spiking Neural Network ===")
        trainer.convert_to_spiking()
        
        # Save converted model
        trainer.save_checkpoint('converted_model.pth')
    
    # Phase 3: Fine-tune SNN
    print("\n=== Phase 3: Fine-tuning Spiking Neural Network ===")
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['finetune_epochs']
    )
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    print("\nTraining complete!")


if __name__ == '__main__':
    main() 