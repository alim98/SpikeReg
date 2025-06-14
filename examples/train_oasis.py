#!/usr/bin/env python3
"""
Training script for SpikeReg on OASIS dataset

This script trains SpikeReg on the Learn2Reg OASIS brain MRI dataset
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikereg import SpikeRegTrainer
from spikereg.training import load_config, save_config
from examples.oasis_dataset import create_oasis_loaders


def update_config_for_oasis(config, args):
    """Update configuration for OASIS dataset specifics"""
    # OASIS volumes are 160x224x192
    # Using patch size 64 for good coverage
    config['data']['patch_size'] = args.patch_size
    config['data']['patch_stride'] = args.patch_stride
    config['model']['patch_size'] = args.patch_size
    
    # Adjust batch size based on available memory
    config['training']['batch_size'] = args.batch_size
    
    # OASIS-specific paths
    config['data']['train_dir'] = args.data_root
    config['data']['val_dir'] = args.data_root
    config['data']['test_dir'] = args.data_root
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SpikeReg on OASIS dataset')
    parser.add_argument('--data-root', type=str, 
                        default='/Users/ali/Documents/codes/SpikeReg/SpikeReg/data/OASIS',
                        help='Path to OASIS dataset root')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/oasis',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/oasis',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Dataset specific arguments
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Size of patches to extract')
    parser.add_argument('--patch-stride', type=int, default=32,
                        help='Stride for patch extraction')
    parser.add_argument('--patches-per-pair', type=int, default=20,
                        help='Number of patches per volume pair')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training options
    parser.add_argument('--pretrain-epochs', type=int, default=30,
                        help='Number of pretraining epochs')
    parser.add_argument('--finetune-epochs', type=int, default=50,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pretraining phase')
    
    args = parser.parse_args()
    
    # Load and update configuration
    config = load_config(args.config)
    config = update_config_for_oasis(config, args)
    
    # Update training epochs
    config['training']['pretrain_epochs'] = args.pretrain_epochs
    config['training']['finetune_epochs'] = args.finetune_epochs
    config['training']['pretrain'] = not args.skip_pretrain
    
    # Save configuration
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_config(config, os.path.join(args.checkpoint_dir, 'config.yaml'))
    
    # Create data loaders
    print(f"Loading OASIS dataset from {args.data_root}")
    train_loader, val_loader = create_oasis_loaders(
        args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patches_per_pair=args.patches_per_pair,
        num_workers=args.num_workers
    )
    
    print(f"Training set: {len(train_loader.dataset)} patches")
    print(f"Validation set: {len(val_loader.dataset)} patches")
    
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
        print("\n" + "="*60)
        print("Phase 1: Pretraining Standard U-Net")
        print("="*60)
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['training']['pretrain_epochs']
        )
        
        # Save pretrained model
        trainer.save_checkpoint('pretrained_model.pth')
        print("Saved pretrained model")
        
        # Phase 2: Convert to SNN
        print("\n" + "="*60)
        print("Phase 2: Converting to Spiking Neural Network")
        print("="*60)
        trainer.convert_to_spiking()
        
        # Save converted model
        trainer.save_checkpoint('converted_model.pth')
        print("Saved converted model")
    
    # Phase 3: Fine-tune SNN
    print("\n" + "="*60)
    print("Phase 3: Fine-tuning Spiking Neural Network")
    print("="*60)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['finetune_epochs']
    )
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    
    # Print final validation metrics
    print("\nFinal validation metrics:")
    val_metrics = trainer.validate(val_loader)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main() 