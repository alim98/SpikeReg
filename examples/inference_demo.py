#!/usr/bin/env python3
"""
Example inference script for SpikeReg

Demonstrates how to use a trained SpikeReg model for registration
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikereg import SpikeRegInference
from utils.metrics import compute_registration_metrics
from utils.warping import SpatialTransformer


def load_volume(path):
    """
    Load medical volume from file
    
    Placeholder - replace with actual loading using nibabel or SimpleITK
    """
    # In practice:
    # import nibabel as nib
    # return nib.load(path).get_fdata()
    
    # For demo, return synthetic volume
    print(f"Loading volume from {path} (using synthetic data for demo)")
    volume = create_synthetic_volume()
    return volume


def create_synthetic_volume():
    """Create synthetic volume for demonstration"""
    # Create 3D volume with some structures
    size = 128
    volume = np.zeros((size, size, size))
    
    # Add sphere
    center = size // 2
    radius = size // 4
    z, y, x = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    volume[mask] = 1.0
    
    # Add some texture
    noise = np.random.randn(size, size, size) * 0.1
    volume = volume + noise
    
    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    return volume.astype(np.float32)


def create_synthetic_deformation():
    """Create synthetic deformation for moving volume"""
    size = 128
    
    # Create smooth deformation field
    sigma = 20
    displacement = np.random.randn(3, size, size, size) * 5
    
    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    for i in range(3):
        displacement[i] = gaussian_filter(displacement[i], sigma)
    
    return displacement


def visualize_registration(fixed, moving, warped, displacement, save_path=None):
    """Visualize registration results"""
    # Take center slices
    slice_idx = fixed.shape[1] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: images
    axes[0, 0].imshow(fixed[0, slice_idx], cmap='gray')
    axes[0, 0].set_title('Fixed Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving[0, slice_idx], cmap='gray')
    axes[0, 1].set_title('Moving Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(warped[0, slice_idx], cmap='gray')
    axes[0, 2].set_title('Warped Image')
    axes[0, 2].axis('off')
    
    # Second row: difference images and deformation
    diff_before = np.abs(fixed[0, slice_idx] - moving[0, slice_idx])
    axes[1, 0].imshow(diff_before, cmap='hot')
    axes[1, 0].set_title('Difference Before')
    axes[1, 0].axis('off')
    
    diff_after = np.abs(fixed[0, slice_idx] - warped[0, slice_idx])
    axes[1, 1].imshow(diff_after, cmap='hot')
    axes[1, 1].set_title('Difference After')
    axes[1, 1].axis('off')
    
    # Deformation magnitude
    disp_mag = np.sqrt(np.sum(displacement**2, axis=0))
    axes[1, 2].imshow(disp_mag[slice_idx], cmap='jet')
    axes[1, 2].set_title('Deformation Magnitude')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_spike_activity(spike_counts, save_path=None):
    """Visualize spike activity across layers"""
    layers = list(spike_counts.keys())
    counts = list(spike_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(layers, counts)
    plt.xlabel('Layer')
    plt.ylabel('Spike Rate')
    plt.title('Spike Activity Across Network Layers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spike activity plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='SpikeReg inference demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--fixed', type=str, default='demo_fixed.nii.gz',
                        help='Path to fixed volume')
    parser.add_argument('--moving', type=str, default='demo_moving.nii.gz',
                        help='Path to moving volume')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model}")
    inference = SpikeRegInference(
        args.model,
        device=args.device,
        patch_size=32,
        patch_stride=16,
        batch_size=8
    )
    
    # Load volumes
    print("Loading volumes...")
    fixed_volume = load_volume(args.fixed)
    moving_volume = load_volume(args.moving)
    
    # For demo: create synthetic deformation
    if not Path(args.moving).exists():
        print("Creating synthetic moving volume with deformation...")
        displacement_gt = create_synthetic_deformation()
        transformer = SpatialTransformer()
        
        # Apply deformation to create moving volume
        fixed_tensor = torch.from_numpy(fixed_volume).unsqueeze(0).unsqueeze(0)
        disp_tensor = torch.from_numpy(displacement_gt).unsqueeze(0)
        moving_tensor = transformer(fixed_tensor, disp_tensor)
        moving_volume = moving_tensor.squeeze().numpy()
    
    # Perform registration
    print("\nPerforming registration...")
    start_time = time.time()
    
    # Register with detailed output
    registration_output = inference.register(
        fixed_volume,
        moving_volume,
        return_all_patches=True,
        progress_bar=True
    )
    
    end_time = time.time()
    print(f"Registration completed in {end_time - start_time:.2f} seconds")
    
    # Extract results
    displacement_field = registration_output['displacement_field']
    patch_iterations = registration_output['patch_iterations']
    
    # Apply deformation
    print("Applying deformation...")
    warped_volume = inference.apply_deformation(moving_volume, displacement_field)
    
    # Compute metrics
    print("\nComputing metrics...")
    fixed_tensor = torch.from_numpy(fixed_volume).unsqueeze(0).unsqueeze(0)
    moving_tensor = torch.from_numpy(moving_volume).unsqueeze(0).unsqueeze(0)
    warped_tensor = torch.from_numpy(warped_volume).unsqueeze(0).unsqueeze(0)
    disp_tensor = torch.from_numpy(displacement_field).unsqueeze(0)
    
    metrics = compute_registration_metrics(
        fixed_tensor,
        moving_tensor,
        warped_tensor,
        disp_tensor
    )
    
    # Print metrics
    print("\nRegistration Metrics:")
    print(f"  NCC: {metrics['ncc']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Jacobian (mean): {metrics['jacobian_mean']:.4f}")
    print(f"  Jacobian (std): {metrics['jacobian_std']:.4f}")
    print(f"  Negative Jacobian: {metrics['jacobian_negative_fraction']*100:.2f}%")
    
    # Print iteration statistics
    avg_iterations = np.mean(patch_iterations)
    print(f"\nAverage iterations per patch: {avg_iterations:.2f}")
    
    # Compute Jacobian determinant
    print("\nComputing Jacobian determinant...")
    jacobian_det = inference.compute_jacobian_determinant(displacement_field)
    
    # Save results
    print("\nSaving results...")
    
    # Save displacement field
    np.save(os.path.join(args.output_dir, 'displacement_field.npy'), displacement_field)
    
    # Save warped volume
    np.save(os.path.join(args.output_dir, 'warped_volume.npy'), warped_volume)
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Prepare data for visualization
        fixed_vis = fixed_volume[np.newaxis, ...]
        moving_vis = moving_volume[np.newaxis, ...]
        warped_vis = warped_volume[np.newaxis, ...]
        
        # Main registration visualization
        visualize_registration(
            fixed_vis,
            moving_vis,
            warped_vis,
            displacement_field,
            save_path=os.path.join(args.output_dir, 'registration_results.png')
        )
        
        # Jacobian determinant visualization
        plt.figure(figsize=(10, 8))
        slice_idx = jacobian_det.shape[0] // 2
        plt.imshow(jacobian_det[slice_idx], cmap='RdBu_r', vmin=0, vmax=2)
        plt.colorbar(label='Jacobian Determinant')
        plt.title('Jacobian Determinant (center slice)')
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, 'jacobian_determinant.png'), 
                    dpi=150, bbox_inches='tight')
        print(f"Saved Jacobian visualization")
        
        # If we have spike information from a single patch (for demo)
        if hasattr(inference.model, 'spike_history') and inference.model.spike_history:
            spike_counts = inference.model.spike_history[-1]
            visualize_spike_activity(
                spike_counts,
                save_path=os.path.join(args.output_dir, 'spike_activity.png')
            )
    
    print(f"\nAll results saved to {args.output_dir}")
    print("Demo complete!")


if __name__ == '__main__':
    main() 