#!/usr/bin/env python3
"""
Inference script for SpikeReg model
Run inference on OASIS data to visualize registration results
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml

# Import from SpikeReg package
from SpikeReg.models import SpikeRegUNet
from SpikeReg.registration import IterativeRegistration
from SpikeReg.utils.warping import SpatialTransformer
from SpikeReg.utils.metrics import compute_registration_metrics
from SpikeReg.utils.preprocessing import normalize_volume


def load_model(checkpoint_path, config_path=None, device='cuda'):
    """Load trained SpikeReg model from checkpoint"""
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_config = config.get('model', {})
    elif 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        print("Warning: No config found, using default configuration")
        model_config = {
            'patch_size': 32,
            'in_channels': 2,
            'base_channels': 16,
            'encoder_channels': [32, 64, 128],
            'decoder_channels': [64, 32, 16],
            'encoder_time_windows': [10, 8, 6, 4],
            'decoder_time_windows': [4, 6, 8, 10],
            'encoder_tau_u': [0.9, 0.8, 0.8, 0.7],
            'decoder_tau_u': [0.7, 0.8, 0.8, 0.9],
            'skip_merge': ['concatenate', 'average', 'concatenate', 'none'],
            'displacement_scale': 1.0,
            'input_time_window': 10
        }
    
    # Create model
    model = SpikeRegUNet(model_config).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel models (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, model_config


def load_oasis_volume(data_dir, volume_idx):
    """Load an OASIS volume from the validation set"""
    data_path = Path(data_dir)
    
    # Try to load from validation set
    val_dir = data_path / "L2R_2021_Task3_val"
    if val_dir.exists():
        # Look for images in validation set
        img_files = sorted(val_dir.glob("img*.nii.gz"))
        if img_files and volume_idx < len(img_files):
            img_path = img_files[volume_idx]
            print(f"Loading volume from: {img_path}")
            volume = nib.load(str(img_path)).get_fdata().astype(np.float32)
            return volume
    
    # Try training set if validation doesn't work
    train_dir = data_path / "L2R_2021_Task3_train"
    if train_dir.exists():
        subj_dirs = sorted(train_dir.glob("OASIS_OAS1_*_MR1"))
        if subj_dirs and volume_idx < len(subj_dirs):
            subj_dir = subj_dirs[volume_idx]
            img_path = subj_dir / "aligned_norm.nii.gz"
            if not img_path.exists():
                # Try any nii.gz file
                nii_files = list(subj_dir.glob("*.nii.gz"))
                if nii_files:
                    img_path = nii_files[0]
            
            if img_path and img_path.exists():
                print(f"Loading volume from: {img_path}")
                volume = nib.load(str(img_path)).get_fdata().astype(np.float32)
                return volume
    
    raise FileNotFoundError(f"Could not find volume at index {volume_idx} in {data_dir}")


def visualize_registration_results(fixed, moving, warped, displacement, 
                                   metrics_dict, output_dir, slice_idx=None):
    """Create comprehensive visualization of registration results"""
    
    # Convert to numpy if needed
    if torch.is_tensor(fixed):
        fixed = fixed.squeeze().cpu().numpy()
    if torch.is_tensor(moving):
        moving = moving.squeeze().cpu().numpy()
    if torch.is_tensor(warped):
        warped = warped.squeeze().cpu().numpy()
    if torch.is_tensor(displacement):
        displacement = displacement.squeeze().cpu().numpy()
    
    # Select middle slice if not specified
    if slice_idx is None:
        slice_idx = fixed.shape[0] // 2
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Axial view
    ax1 = plt.subplot(3, 5, 1)
    ax1.imshow(fixed[slice_idx], cmap='gray')
    ax1.set_title('Fixed (Axial)', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 5, 2)
    ax2.imshow(moving[slice_idx], cmap='gray')
    ax2.set_title('Moving (Axial)', fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 5, 3)
    ax3.imshow(warped[slice_idx], cmap='gray')
    ax3.set_title('Warped (Axial)', fontsize=12)
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 5, 4)
    diff_before = np.abs(fixed[slice_idx] - moving[slice_idx])
    im4 = ax4.imshow(diff_before, cmap='hot', vmin=0, vmax=diff_before.max())
    ax4.set_title('Diff Before', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(3, 5, 5)
    diff_after = np.abs(fixed[slice_idx] - warped[slice_idx])
    im5 = ax5.imshow(diff_after, cmap='hot', vmin=0, vmax=diff_before.max())
    ax5.set_title('Diff After', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Row 2: Coronal view
    coronal_idx = fixed.shape[1] // 2
    
    ax6 = plt.subplot(3, 5, 6)
    ax6.imshow(fixed[:, coronal_idx, :], cmap='gray')
    ax6.set_title('Fixed (Coronal)', fontsize=12)
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 5, 7)
    ax7.imshow(moving[:, coronal_idx, :], cmap='gray')
    ax7.set_title('Moving (Coronal)', fontsize=12)
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 5, 8)
    ax8.imshow(warped[:, coronal_idx, :], cmap='gray')
    ax8.set_title('Warped (Coronal)', fontsize=12)
    ax8.axis('off')
    
    ax9 = plt.subplot(3, 5, 9)
    diff_before_cor = np.abs(fixed[:, coronal_idx, :] - moving[:, coronal_idx, :])
    im9 = ax9.imshow(diff_before_cor, cmap='hot', vmin=0, vmax=diff_before_cor.max())
    ax9.set_title('Diff Before', fontsize=12)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    
    ax10 = plt.subplot(3, 5, 10)
    diff_after_cor = np.abs(fixed[:, coronal_idx, :] - warped[:, coronal_idx, :])
    im10 = ax10.imshow(diff_after_cor, cmap='hot', vmin=0, vmax=diff_before_cor.max())
    ax10.set_title('Diff After', fontsize=12)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)
    
    # Row 3: Displacement field and metrics
    ax11 = plt.subplot(3, 5, 11)
    disp_mag = np.sqrt(np.sum(displacement**2, axis=0))
    im11 = ax11.imshow(disp_mag[slice_idx], cmap='jet')
    ax11.set_title('Displacement Magnitude', fontsize=12)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)
    
    ax12 = plt.subplot(3, 5, 12)
    im12 = ax12.imshow(displacement[0, slice_idx], cmap='RdBu_r')
    ax12.set_title('Displacement X', fontsize=12)
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046)
    
    ax13 = plt.subplot(3, 5, 13)
    im13 = ax13.imshow(displacement[1, slice_idx], cmap='RdBu_r')
    ax13.set_title('Displacement Y', fontsize=12)
    ax13.axis('off')
    plt.colorbar(im13, ax=ax13, fraction=0.046)
    
    ax14 = plt.subplot(3, 5, 14)
    im14 = ax14.imshow(displacement[2, slice_idx], cmap='RdBu_r')
    ax14.set_title('Displacement Z', fontsize=12)
    ax14.axis('off')
    plt.colorbar(im14, ax=ax14, fraction=0.046)
    
    # Metrics text box
    ax15 = plt.subplot(3, 5, 15)
    ax15.axis('off')
    metrics_text = "Registration Metrics:\n\n"
    metrics_text += f"NCC: {metrics_dict.get('ncc', 0):.4f}\n"
    metrics_text += f"MSE: {metrics_dict.get('mse', 0):.4f}\n"
    metrics_text += f"SSIM: {metrics_dict.get('ssim', 0):.4f}\n\n"
    metrics_text += f"Jacobian (mean): {metrics_dict.get('jacobian_mean', 0):.4f}\n"
    metrics_text += f"Jacobian (std): {metrics_dict.get('jacobian_std', 0):.4f}\n"
    metrics_text += f"Negative Jacobian: {metrics_dict.get('jacobian_negative_fraction', 0)*100:.2f}%\n\n"
    metrics_text += f"Iterations: {metrics_dict.get('iterations', 0)}\n"
    metrics_text += f"Converged: {metrics_dict.get('converged', False)}\n"
    metrics_text += f"Time: {metrics_dict.get('time', 0):.2f}s"
    
    ax15.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SpikeReg Registration Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'registration_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def run_patch_inference(model, fixed, moving, config, device='cuda'):
    """Run inference on a patch"""
    
    # Create iterative registration
    registration = IterativeRegistration(
        model,
        num_iterations=config.get('num_iterations', 10),
        early_stop_threshold=config.get('early_stop_threshold', 0.001)
    )
    
    # Run registration
    with torch.no_grad():
        output = registration(fixed, moving, return_all_iterations=True)
    
    return output


def main():
    print("="*70)
    print("SpikeReg Inference Script")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Paths
    project_root = Path(__file__).parent
    
    # Try multiple checkpoint locations
    checkpoint_paths = [
        project_root / "checkpoints/final_oasis/best_model.pth",
        project_root / "checkpoints/final_oasis/converted_model.pth",
        project_root / "checkpoints/spikereg/runs/22434639/checkpoints/converted_model.pth",
        project_root / "checkpoints/spikereg/runs/22434639/checkpoints/final_model.pth",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("Error: No checkpoint found!")
        print("Looked in:")
        for path in checkpoint_paths:
            print(f"  - {path}")
        return
    
    config_path = project_root / "checkpoints/final_oasis/config.yaml"
    data_dir = Path("/u/almik/SpikeReg2/data")
    output_dir = project_root / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    print("\n" + "="*70)
    print("Loading Model")
    print("="*70)
    model, model_config = load_model(checkpoint_path, config_path, device)
    
    # Load test volumes
    print("\n" + "="*70)
    print("Loading Test Volumes")
    print("="*70)
    
    try:
        # Load two different volumes for registration
        fixed_volume = load_oasis_volume(data_dir, volume_idx=0)
        moving_volume = load_oasis_volume(data_dir, volume_idx=1)
        
        print(f"Fixed volume shape: {fixed_volume.shape}")
        print(f"Moving volume shape: {moving_volume.shape}")
    except Exception as e:
        print(f"Error loading volumes: {e}")
        print("Using synthetic data instead...")
        
        # Create synthetic data for demonstration
        size = 128
        fixed_volume = np.zeros((size, size, size), dtype=np.float32)
        moving_volume = np.zeros((size, size, size), dtype=np.float32)
        
        # Add some structures
        center = size // 2
        radius = size // 4
        z, y, x = np.ogrid[:size, :size, :size]
        mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
        fixed_volume[mask] = 1.0
        
        # Add slight offset for moving
        mask_moving = (x - center - 5)**2 + (y - center - 3)**2 + (z - center)**2 <= radius**2
        moving_volume[mask_moving] = 1.0
        
        # Add noise
        fixed_volume += np.random.randn(size, size, size) * 0.1
        moving_volume += np.random.randn(size, size, size) * 0.1
    
    # Normalize volumes
    fixed_tensor = torch.from_numpy(fixed_volume).float().unsqueeze(0).unsqueeze(0)
    moving_tensor = torch.from_numpy(moving_volume).float().unsqueeze(0).unsqueeze(0)
    
    fixed_tensor = normalize_volume(fixed_tensor)
    moving_tensor = normalize_volume(moving_tensor)
    
    # Extract center patch for testing
    patch_size = model_config.get('patch_size', 32)
    D, H, W = fixed_tensor.shape[2:]
    
    # Extract center patch
    d_start = (D - patch_size) // 2
    h_start = (H - patch_size) // 2
    w_start = (W - patch_size) // 2
    
    fixed_patch = fixed_tensor[:, :, 
                                d_start:d_start+patch_size,
                                h_start:h_start+patch_size,
                                w_start:w_start+patch_size].to(device)
    moving_patch = moving_tensor[:, :,
                                  d_start:d_start+patch_size,
                                  h_start:h_start+patch_size,
                                  w_start:w_start+patch_size].to(device)
    
    print(f"\nPatch size: {patch_size}")
    print(f"Fixed patch shape: {fixed_patch.shape}")
    print(f"Moving patch shape: {moving_patch.shape}")
    
    # Run inference
    print("\n" + "="*70)
    print("Running Registration")
    print("="*70)
    
    start_time = time.time()
    output = run_patch_inference(model, fixed_patch, moving_patch, model_config, device)
    inference_time = time.time() - start_time
    
    print(f"\nRegistration completed in {inference_time:.2f} seconds")
    print(f"Iterations: {output['iterations']}")
    print(f"Converged: {output['converged']}")
    
    # Extract results
    displacement = output['displacement']
    warped = output['warped']
    
    # Compute metrics
    print("\n" + "="*70)
    print("Computing Metrics")
    print("="*70)
    
    metrics = compute_registration_metrics(
        fixed_patch,
        moving_patch,
        warped,
        displacement
    )
    
    print("\nRegistration Metrics:")
    print(f"  NCC: {metrics['ncc']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Jacobian (mean): {metrics['jacobian_mean']:.4f}")
    print(f"  Jacobian (std): {metrics['jacobian_std']:.4f}")
    print(f"  Negative Jacobian: {metrics['jacobian_negative_fraction']*100:.2f}%")
    
    # Add iteration info to metrics
    metrics['iterations'] = output['iterations']
    metrics['converged'] = output['converged']
    metrics['time'] = inference_time
    
    # Check for spike count history
    if 'spike_count_history' in output and output['spike_count_history']:
        print("\nSpike Activity:")
        final_spikes = output['spike_count_history'][-1]
        for layer, count in final_spikes.items():
            print(f"  {layer}: {count:.4f}")
    
    # Visualize results
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    visualize_registration_results(
        fixed_patch,
        moving_patch,
        warped,
        displacement,
        metrics,
        output_dir
    )
    
    # Save numerical results
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    # Save displacement field
    np.save(output_dir / 'displacement_field.npy', displacement.cpu().numpy())
    print(f"Saved displacement field to: {output_dir / 'displacement_field.npy'}")
    
    # Save warped volume
    np.save(output_dir / 'warped_patch.npy', warped.cpu().numpy())
    print(f"Saved warped patch to: {output_dir / 'warped_patch.npy'}")
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                       for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - registration_results.png  (visualization)")
    print(f"  - displacement_field.npy     (deformation field)")
    print(f"  - warped_patch.npy          (registered image)")
    print(f"  - metrics.json              (quantitative metrics)")


if __name__ == '__main__':
    main()

