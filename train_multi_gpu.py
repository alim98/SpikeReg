#!/usr/bin/env python3
"""
Multi-GPU Training Launcher for SpikeReg on OASIS Dataset

This script provides convenient presets for training with multiple GPUs.
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Launch multi-GPU SpikeReg training')
    
    # Preset configurations
    parser.add_argument('--preset', type=str, default='dual_gpu',
                        choices=['single_gpu', 'dual_gpu', 'all_gpu'],
                        help='Training preset configuration')
    
    # Override options
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--patch-size', type=int, default=None,
                        help='Override patch size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of workers')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root path')
    
    # Training options
    parser.add_argument('--pretrain-epochs', type=int, default=5,
                        help='Number of pretraining epochs')
    parser.add_argument('--finetune-epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/multi_gpu',
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Define presets
    presets = {
        'single_gpu': {
            'multi_gpu': False,
            'gpu_ids': '0',
            'batch_size': 4,
            'patch_size': 32,
            'num_workers': 4,
            'description': 'Single GPU (RTX 4090) - Conservative settings'
        },
        'dual_gpu': {
            'multi_gpu': True,
            'gpu_ids': '0,1',
            'batch_size': 8,  # 4 per GPU
            'patch_size': 32,
            'num_workers': 8,  # 4 per GPU
            'description': 'Dual GPU (2x RTX 4090) - Balanced performance'
        },
        'all_gpu': {
            'multi_gpu': True,
            'gpu_ids': None,  # Use all available
            'batch_size': 16,  # Scale with number of GPUs
            'patch_size': 32,
            'num_workers': 12,
            'description': 'All available GPUs - Maximum performance'
        }
    }
    
    # Get preset configuration
    preset_config = presets[args.preset]
    print(f"Using preset: {args.preset}")
    print(f"Description: {preset_config['description']}")
    
    # Build command
    cmd = [
        sys.executable, 'examples/train_oasis.py',
        '--pretrain-epochs', str(args.pretrain_epochs),
        '--finetune-epochs', str(args.finetune_epochs),
        '--checkpoint-dir', args.checkpoint_dir,
        '--log-dir', 'logs/multi_gpu',
    ]
    
    # Add multi-GPU options
    if preset_config['multi_gpu']:
        cmd.append('--multi-gpu')
        if preset_config['gpu_ids']:
            cmd.extend(['--gpu-ids', preset_config['gpu_ids']])
    
    # Add configuration options (with overrides)
    batch_size = args.batch_size or preset_config['batch_size']
    patch_size = args.patch_size or preset_config['patch_size']
    num_workers = args.num_workers or preset_config['num_workers']
    
    cmd.extend([
        '--batch-size', str(batch_size),
        '--patch-size', str(patch_size),
        '--num-workers', str(num_workers),
        '--patch-stride', '16',  # 50% overlap for more training data
    ])
    
    # Data root
    if args.data_root:
        cmd.extend(['--data-root', args.data_root])
    
    # Print configuration
    print("\n" + "="*60)
    print("MULTI-GPU TRAINING CONFIGURATION")
    print("="*60)
    print(f"Preset: {args.preset}")
    print(f"Multi-GPU: {preset_config['multi_gpu']}")
    if preset_config['multi_gpu'] and preset_config['gpu_ids']:
        print(f"GPU IDs: {preset_config['gpu_ids']}")
    print(f"Batch Size: {batch_size}")
    print(f"Patch Size: {patch_size}")
    print(f"Workers: {num_workers}")
    print(f"Pretrain Epochs: {args.pretrain_epochs}")
    print(f"Finetune Epochs: {args.finetune_epochs}")
    print("="*60)
    
    # Show expected performance
    if args.preset == 'dual_gpu':
        print("\n📊 EXPECTED PERFORMANCE (2x RTX 4090):")
        print("   • Training Speed: ~3-4x faster than single GPU")
        print("   • Memory Usage: ~24GB per GPU")
        print("   • Batch Size: 8 (4 per GPU)")
        print("   • Estimated Time: ~2-3 hours for full training")
    
    print(f"\n🚀 Starting training with command:")
    print(" ".join(cmd))
    print()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('logs/multi_gpu', exist_ok=True)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main() 