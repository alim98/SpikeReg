#!/usr/bin/env python3
"""
Test script to verify OASIS dataset loading
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.oasis_dataset import OASISDataset, create_oasis_loaders
import matplotlib.pyplot as plt
import torch

def test_dataset_loading():
    """Test basic dataset loading"""
    print("Testing OASIS dataset loading...")
    
    data_root = '/Users/ali/Documents/codes/SpikeReg/SpikeReg/data/OASIS'
    
    # Create a small dataset for testing
    dataset = OASISDataset(
        data_root,
        split='train',
        patch_size=64,
        patch_stride=32,
        patches_per_pair=5,
        fixed_pairs=True
    )
    
    print(f"Dataset created successfully!")
    print(f"Number of patches: {len(dataset)}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Fixed shape: {sample['fixed'].shape}")
    print(f"Moving shape: {sample['moving'].shape}")
    
    if 'segmentation_fixed' in sample:
        print(f"Fixed segmentation shape: {sample['segmentation_fixed'].shape}")
        print(f"Moving segmentation shape: {sample['segmentation_moving'].shape}")
    
    return dataset, sample


def test_data_loaders():
    """Test data loader creation"""
    print("\n\nTesting data loader creation...")
    
    data_root = '/Users/ali/Documents/codes/SpikeReg/SpikeReg/data/OASIS'
    
    train_loader, val_loader = create_oasis_loaders(
        data_root,
        batch_size=2,
        patch_size=64,
        patch_stride=32,
        patches_per_pair=5,
        num_workers=0  # Set to 0 for testing
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch fixed shape: {batch['fixed'].shape}")
    print(f"Batch moving shape: {batch['moving'].shape}")
    
    return train_loader, batch


def visualize_sample(sample):
    """Visualize a sample from the dataset"""
    print("\n\nVisualizing sample...")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Get middle slices
    fixed = sample['fixed'][0]  # Remove channel dimension
    moving = sample['moving'][0]
    
    # Axial, Coronal, Sagittal views
    d, h, w = fixed.shape
    
    # Fixed image
    axes[0, 0].imshow(fixed[d//2, :, :], cmap='gray')
    axes[0, 0].set_title('Fixed - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fixed[:, h//2, :], cmap='gray')
    axes[0, 1].set_title('Fixed - Coronal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fixed[:, :, w//2], cmap='gray')
    axes[0, 2].set_title('Fixed - Sagittal')
    axes[0, 2].axis('off')
    
    # Moving image
    axes[1, 0].imshow(moving[d//2, :, :], cmap='gray')
    axes[1, 0].set_title('Moving - Axial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(moving[:, h//2, :], cmap='gray')
    axes[1, 1].set_title('Moving - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(moving[:, :, w//2], cmap='gray')
    axes[1, 2].set_title('Moving - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('oasis_test_visualization.png', dpi=150)
    print("Saved visualization to oasis_test_visualization.png")
    
    # If segmentations are available, visualize them too
    if 'segmentation_fixed' in sample:
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
        
        fixed_seg = sample['segmentation_fixed'][0]
        moving_seg = sample['segmentation_moving'][0]
        
        axes2[0].imshow(fixed_seg[d//2, :, :], cmap='tab20')
        axes2[0].set_title('Fixed Segmentation - Axial')
        axes2[0].axis('off')
        
        axes2[1].imshow(moving_seg[d//2, :, :], cmap='tab20')
        axes2[1].set_title('Moving Segmentation - Axial')
        axes2[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('oasis_segmentation_visualization.png', dpi=150)
        print("Saved segmentation visualization to oasis_segmentation_visualization.png")


def main():
    try:
        # Test dataset loading
        dataset, sample = test_dataset_loading()
        
        # Test data loaders
        train_loader, batch = test_data_loaders()
        
        # Visualize
        visualize_sample(sample)
        
        print("\n\nAll tests passed! OASIS dataset is ready for training.")
        
        # Print some statistics
        print("\nDataset Statistics:")
        print(f"Total training patches: {len(train_loader.dataset)}")
        print(f"Patches per epoch: {len(train_loader) * train_loader.batch_size}")
        print(f"Memory per batch (approx): {batch['fixed'].element_size() * batch['fixed'].nelement() * 2 / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 