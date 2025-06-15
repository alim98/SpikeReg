"""
Patch extraction and stitching utilities for SpikeReg
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union


def extract_patches(
    volume: torch.Tensor,
    patch_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]],
    padding: str = 'constant',
    padding_value: float = 0.0
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
    """
    Extract overlapping patches from 3D volume
    
    Args:
        volume: Input volume [B, C, D, H, W]
        patch_size: Size of patches (int or tuple)
        stride: Stride between patches (int or tuple)
        padding: Padding mode for boundaries
        padding_value: Value for constant padding
        
    Returns:
        patches: List of patch tensors
        coordinates: List of patch top-left coordinates
    """
    B, C, D, H, W = volume.shape
    
    # Convert to tuple if needed
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Calculate padding needed
    pad_d = (pd - D % sd) % sd
    pad_h = (ph - H % sh) % sh
    pad_w = (pw - W % sw) % sw
    
    # Pad volume if necessary
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        padding_amounts = (0, pad_w, 0, pad_h, 0, pad_d)  # (left, right, top, bottom, front, back)
        volume = F.pad(volume, padding_amounts, mode=padding, value=padding_value)
        D_padded, H_padded, W_padded = D + pad_d, H + pad_h, W + pad_w
    else:
        D_padded, H_padded, W_padded = D, H, W
    
    # Extract patches
    patches = []
    coordinates = []
    
    for d in range(0, D_padded - pd + 1, sd):
        for h in range(0, H_padded - ph + 1, sh):
            for w in range(0, W_padded - pw + 1, sw):
                # Extract patch
                patch = volume[:, :, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
                coordinates.append((d, h, w))
    
    return patches, coordinates


def stitch_patches(
    patches: List[torch.Tensor],
    coordinates: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
    patch_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]],
    blend_mode: str = 'average',
    window_type: str = 'cosine'
) -> torch.Tensor:
    """
    Stitch patches back into full volume with blending
    
    Args:
        patches: List of patch tensors
        coordinates: List of patch top-left coordinates
        output_shape: Target output shape (D, H, W)
        patch_size: Size of patches
        stride: Stride between patches
        blend_mode: How to blend overlapping regions
        window_type: Type of blending window
        
    Returns:
        stitched: Stitched volume [B, C, D, H, W]
    """
    if not patches:
        raise ValueError("No patches to stitch")
    
    # Get dimensions from first patch
    B, C = patches[0].shape[:2]
    D, H, W = output_shape
    
    # Convert to tuple if needed
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    
    pd, ph, pw = patch_size
    
    # Initialize output and weight volumes
    output = torch.zeros(B, C, D, H, W, device=patches[0].device)
    weights = torch.zeros(B, 1, D, H, W, device=patches[0].device)
    
    # Create blending window
    window = create_blending_window(patch_size, window_type, device=patches[0].device)
    
    # Place patches
    for patch, (d, h, w) in zip(patches, coordinates):
        # Calculate valid region (handle boundary)
        d_end = min(d + pd, D)
        h_end = min(h + ph, H)
        w_end = min(w + pw, W)
        
        # Get valid patch region
        valid_d = d_end - d
        valid_h = h_end - h
        valid_w = w_end - w
        
        if blend_mode == 'average':
            # Add weighted patch
            output[:, :, d:d_end, h:h_end, w:w_end] += \
                patch[:, :, :valid_d, :valid_h, :valid_w] * window[:, :, :valid_d, :valid_h, :valid_w]
            weights[:, :, d:d_end, h:h_end, w:w_end] += \
                window[:, :, :valid_d, :valid_h, :valid_w]
        
        elif blend_mode == 'max':
            # Take maximum value
            output[:, :, d:d_end, h:h_end, w:w_end] = torch.maximum(
                output[:, :, d:d_end, h:h_end, w:w_end],
                patch[:, :, :valid_d, :valid_h, :valid_w]
            )
            weights[:, :, d:d_end, h:h_end, w:w_end] = torch.ones_like(
                weights[:, :, d:d_end, h:h_end, w:w_end]
            )
    
    # Normalize by weights
    if blend_mode == 'average':
        output = output / (weights + 1e-8)
    
    return output


def create_blending_window(
    patch_size: Tuple[int, int, int],
    window_type: str = 'cosine',
    device: torch.device = None
) -> torch.Tensor:
    """
    Create 3D blending window for smooth patch transitions
    
    Args:
        patch_size: Size of patch (D, H, W)
        window_type: Type of window function
        device: Device to create tensor on
        
    Returns:
        window: 3D blending window [1, 1, D, H, W]
    """
    pd, ph, pw = patch_size
    
    if window_type == 'cosine':
        # Create 1D cosine windows
        d_window = torch.hann_window(pd, periodic=False, device=device)
        h_window = torch.hann_window(ph, periodic=False, device=device)
        w_window = torch.hann_window(pw, periodic=False, device=device)
        
        # Create 3D window
        window = d_window.view(-1, 1, 1) * h_window.view(1, -1, 1) * w_window.view(1, 1, -1)
    
    elif window_type == 'linear':
        # Linear taper at boundaries
        margin = min(pd, ph, pw) // 4
        window = torch.ones(pd, ph, pw, device=device)
        
        # Apply linear taper
        for i in range(margin):
            weight = (i + 1) / margin
            window[i, :, :] *= weight
            window[-i-1, :, :] *= weight
            window[:, i, :] *= weight
            window[:, -i-1, :] *= weight
            window[:, :, i] *= weight
            window[:, :, -i-1] *= weight
    
    elif window_type == 'gaussian':
        # Gaussian window
        sigma = min(pd, ph, pw) / 6.0
        
        d_coords = torch.arange(pd, device=device) - pd / 2
        h_coords = torch.arange(ph, device=device) - ph / 2
        w_coords = torch.arange(pw, device=device) - pw / 2
        
        d_gauss = torch.exp(-(d_coords ** 2) / (2 * sigma ** 2))
        h_gauss = torch.exp(-(h_coords ** 2) / (2 * sigma ** 2))
        w_gauss = torch.exp(-(w_coords ** 2) / (2 * sigma ** 2))
        
        window = d_gauss.view(-1, 1, 1) * h_gauss.view(1, -1, 1) * w_gauss.view(1, 1, -1)
    
    else:
        # Uniform window (no blending)
        window = torch.ones(pd, ph, pw, device=device)
    
    # Add batch and channel dimensions
    window = window.unsqueeze(0).unsqueeze(0)
    
    return window


def extract_patches_multiresolution(
    volume: torch.Tensor,
    patch_sizes: List[Union[int, Tuple[int, int, int]]],
    strides: List[Union[int, Tuple[int, int, int]]],
    scales: List[float] = None
) -> List[Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]]:
    """
    Extract patches at multiple resolutions
    
    Args:
        volume: Input volume [B, C, D, H, W]
        patch_sizes: Patch sizes for each resolution
        strides: Strides for each resolution
        scales: Scaling factors for each resolution
        
    Returns:
        all_patches: List of (patches, coordinates) for each resolution
    """
    if scales is None:
        scales = [1.0] * len(patch_sizes)
    
    all_patches = []
    
    for patch_size, stride, scale in zip(patch_sizes, strides, scales):
        # Resize volume if needed
        if scale != 1.0:
            scaled_volume = F.interpolate(
                volume, 
                scale_factor=scale, 
                mode='trilinear', 
                align_corners=True
            )
        else:
            scaled_volume = volume
        
        # Extract patches at this resolution
        patches, coords = extract_patches(scaled_volume, patch_size, stride)
        all_patches.append((patches, coords))
    
    return all_patches


class PatchAugmentor:
    """
    Augment patches for training with 3D transformations
    """
    
    def __init__(
        self,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: float = 0.1,
        flip_prob: float = 0.5,
        noise_std: float = 0.01,
        intensity_shift: float = 0.1,
        intensity_scale: Tuple[float, float] = (0.9, 1.1)
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
    
    def augment(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to patch"""
        B, C, D, H, W = patch.shape
        device = patch.device
        
        # Random affine transformation
        if self.rotation_range > 0 or self.scale_range != (1.0, 1.0) or self.translation_range > 0:
            # Generate random parameters
            angle = torch.rand(3, device=device) * 2 * self.rotation_range - self.rotation_range
            angle = angle * np.pi / 180  # Convert to radians
            
            scale = torch.rand(1, device=device) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            
            translation = torch.rand(3, device=device) * 2 * self.translation_range - self.translation_range
            translation = translation * torch.tensor([D, H, W], device=device)
            
            # Create affine matrix
            affine_matrix = create_affine_matrix_3d(angle, scale, translation)
            
            # Apply transformation
            grid = F.affine_grid(affine_matrix.unsqueeze(0), patch.shape, align_corners=True)
            patch = F.grid_sample(patch, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Random flips
        if self.flip_prob > 0:
            for dim in [2, 3, 4]:  # D, H, W dimensions
                if torch.rand(1).item() < self.flip_prob:
                    patch = torch.flip(patch, dims=[dim])
        
        # Intensity augmentation
        if self.intensity_shift > 0:
            shift = torch.rand(1, device=device) * 2 * self.intensity_shift - self.intensity_shift
            patch = patch + shift
        
        if self.intensity_scale != (1.0, 1.0):
            scale = torch.rand(1, device=device) * (self.intensity_scale[1] - self.intensity_scale[0]) + self.intensity_scale[0]
            patch = patch * scale
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(patch) * self.noise_std
            patch = patch + noise
        
        return patch


def create_affine_matrix_3d(
    angles: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor
) -> torch.Tensor:
    """
    Create 3D affine transformation matrix
    
    Args:
        angles: Rotation angles around x, y, z axes (radians)
        scale: Scaling factor
        translation: Translation vector
        
    Returns:
        affine: 3x4 affine matrix
    """
    device = angles.device
    
    # Rotation matrices
    rx = angles[0]
    ry = angles[1]
    rz = angles[2]
    
    # Rotation around x
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rx), -torch.sin(rx)],
        [0, torch.sin(rx), torch.cos(rx)]
    ], device=device)
    
    # Rotation around y
    Ry = torch.tensor([
        [torch.cos(ry), 0, torch.sin(ry)],
        [0, 1, 0],
        [-torch.sin(ry), 0, torch.cos(ry)]
    ], device=device)
    
    # Rotation around z
    Rz = torch.tensor([
        [torch.cos(rz), -torch.sin(rz), 0],
        [torch.sin(rz), torch.cos(rz), 0],
        [0, 0, 1]
    ], device=device)
    
    # Combined rotation
    R = torch.mm(torch.mm(Rz, Ry), Rx)
    
    # Apply scaling
    R = R * scale
    
    # Create affine matrix
    affine = torch.cat([R, translation.unsqueeze(1)], dim=1)
    
    return affine


def compute_patch_overlap(
    patch_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]]
) -> float:
    """
    Compute overlap percentage between adjacent patches
    
    Args:
        patch_size: Size of patches
        stride: Stride between patches
        
    Returns:
        overlap: Overlap percentage (0-100)
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    
    overlap_d = max(0, patch_size[0] - stride[0]) / patch_size[0]
    overlap_h = max(0, patch_size[1] - stride[1]) / patch_size[1]
    overlap_w = max(0, patch_size[2] - stride[2]) / patch_size[2]
    
    # Average overlap across dimensions
    overlap = (overlap_d + overlap_h + overlap_w) / 3 * 100
    
    return overlap 