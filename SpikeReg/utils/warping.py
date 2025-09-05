"""
Warping and spatial transformation utilities for SpikeReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SpatialTransformer(nn.Module):
    """
    Spatial transformer for warping images with displacement fields
    
    Implements differentiable warping for 3D medical images
    """
    
    def __init__(self, mode: str = 'bilinear', padding_mode: str = 'border'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
    
    def forward(
        self, 
        src: torch.Tensor, 
        displacement: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp source image with displacement field
        
        Args:
            src: Source image [B, C, D, H, W]
            displacement: Displacement field [B, 3, D, H, W]
            
        Returns:
            warped: Warped image [B, C, D, H, W]
        """
        B, C, D, H, W = src.shape
        device = src.device
        
        # Create sampling grid
        grid = self.create_grid(B, D, H, W, device)
        
        # Add displacement to grid
        # Displacement is in voxel units, need to normalize to [-1, 1]
        
# after (reorder to x,y,z):
        dx = displacement[:, 2] / (W - 1) * 2  # x
        dy = displacement[:, 1] / (H - 1) * 2  # y
        dz = displacement[:, 0] / (D - 1) * 2  # z
        disp_norm = torch.stack([dx, dy, dz], dim=1)
        new_grid = grid + disp_norm.permute(0, 2, 3, 4, 1)

        
        # Warp image
        warped = F.grid_sample(
            src, new_grid, 
            mode=self.mode, 
            padding_mode=self.padding_mode,
            align_corners=True
        )
        
        return warped
    
    @staticmethod
    def create_grid(
        batch_size: int, 
        depth: int, 
        height: int, 
        width: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Create normalized 3D sampling grid
        
        Returns grid with values in [-1, 1]
        """
        # Create 1D tensors for each dimension
        d_range = torch.linspace(-1, 1, depth, device=device)
        h_range = torch.linspace(-1, 1, height, device=device)
        w_range = torch.linspace(-1, 1, width, device=device)
        
        # Create meshgrid
        grid_d, grid_h, grid_w = torch.meshgrid(d_range, h_range, w_range, indexing='ij')
        
        # Stack and expand for batch
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # Note: x, y, z order for grid_sample
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        return grid


class DiffeomorphicTransformer(nn.Module):
    """
    Diffeomorphic spatial transformer using scaling and squaring
    
    Ensures smooth, invertible transformations
    """
    
    def __init__(
        self, 
        scaling_steps: int = 7,
        mode: str = 'bilinear',
        padding_mode: str = 'border'
    ):
        super().__init__()
        self.scaling_steps = scaling_steps
        self.transformer = SpatialTransformer(mode, padding_mode)
    
    def forward(
        self, 
        src: torch.Tensor, 
        velocity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply diffeomorphic transformation using velocity field
        
        Args:
            src: Source image [B, C, D, H, W]
            velocity: Velocity field [B, 3, D, H, W]
            
        Returns:
            warped: Warped image
            displacement: Final displacement field
        """
        # Scale velocity field
        v = velocity / (2 ** self.scaling_steps)
        
        # Initialize displacement as velocity
        displacement = v.clone()
        
        # Scaling and squaring
        for _ in range(self.scaling_steps):
            # Compose displacement with itself
            displacement = displacement + self.transformer(displacement, displacement)
        
        # Apply final transformation
        warped = self.transformer(src, displacement)
        
        return warped, displacement
    
    def inverse_transform(
        self,
        displacement: torch.Tensor,
        max_iterations: int = 20,
        tolerance: float = 1e-5
    ) -> torch.Tensor:
        """
        Compute inverse displacement field using fixed-point iteration
        
        Args:
            displacement: Forward displacement field [B, 3, D, H, W]
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            inv_displacement: Inverse displacement field
        """
        B, _, D, H, W = displacement.shape
        device = displacement.device
        
        # Initialize inverse as negative displacement
        inv_displacement = -displacement.clone()
        
        # Create identity grid
        # grid = SpatialTransformer.create_grid(B, D, H, W, device)
                # replace identity built via create_grid(...) with voxel-space identity
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        identity = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # [1,3,D,H,W]

        # identity = grid.permute(0, 4, 1, 2, 3).contiguous()
        
        # Fixed-point iteration
        for i in range(max_iterations):
            # Compute composed displacement
            warped_inv = self.transformer(inv_displacement, displacement)
            
            # Update: inv_disp = -disp ∘ (id + inv_disp)
            update = -self.transformer(displacement, inv_displacement + identity) - identity
            
            # Check convergence
            change = (update - inv_displacement).abs().mean()
            if change < tolerance:
                break
            
            inv_displacement = update
        
        return inv_displacement


def compose_displacements(
    disp1: torch.Tensor,
    disp2: torch.Tensor,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Compose two displacement fields: disp = disp2 ∘ disp1
    
    Args:
        disp1: First displacement field [B, 3, D, H, W]
        disp2: Second displacement field [B, 3, D, H, W]
        mode: Interpolation mode
        
    Returns:
        composed: Composed displacement field
    """
    transformer = SpatialTransformer(mode=mode)
    
    # Warp disp2 by disp1
    warped_disp2 = transformer(disp2, disp1)
    
    # Add displacements
    composed = disp1 + warped_disp2
    
    return composed


def integrate_velocity_field(
    velocity: torch.Tensor,
    steps: int = 7,
    method: str = 'scaling_squaring'
) -> torch.Tensor:
    """
    Integrate velocity field to get displacement field
    
    Args:
        velocity: Velocity field [B, 3, D, H, W]
        steps: Number of integration steps
        method: Integration method
        
    Returns:
        displacement: Integrated displacement field
    """
    if method == 'scaling_squaring':
        # Use scaling and squaring
        transformer = DiffeomorphicTransformer(scaling_steps=steps)
        _, displacement = transformer(velocity, velocity)  # Dummy src
        return displacement
    
    elif method == 'euler':
        # Simple Euler integration
        dt = 1.0 / steps
        displacement = torch.zeros_like(velocity)
        
        for _ in range(steps):
            displacement = displacement + dt * velocity
        
        return displacement
    
    elif method == 'rk4':
        # Runge-Kutta 4th order
        dt = 1.0 / steps
        displacement = torch.zeros_like(velocity)
        transformer = SpatialTransformer()
        
        for _ in range(steps):
            k1 = velocity
            k2 = transformer(velocity, displacement + 0.5 * dt * k1)
            k3 = transformer(velocity, displacement + 0.5 * dt * k2)
            k4 = transformer(velocity, displacement + dt * k3)
            
            displacement = displacement + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return displacement
    
    else:
        raise ValueError(f"Unknown integration method: {method}")


def resize_displacement_field(
    displacement: torch.Tensor,
    size: Tuple[int, int, int],
    mode: str = 'trilinear'
) -> torch.Tensor:
    """
    Resize displacement field to new spatial dimensions
    
    Args:
        displacement: Displacement field [B, 3, D, H, W]
        size: Target size (D', H', W')
        mode: Interpolation mode
        
    Returns:
        resized: Resized displacement field
    """
    B, C, D, H, W = displacement.shape
    D_new, H_new, W_new = size
    
    # Scale displacement values proportionally
    scale_d = D_new / D
    scale_h = H_new / H
    scale_w = W_new / W
    
    # Resize each component
    displacement_scaled = displacement.clone()
    displacement_scaled[:, 0] *= scale_d
    displacement_scaled[:, 1] *= scale_h
    displacement_scaled[:, 2] *= scale_w
    
    # Interpolate to new size
    resized = F.interpolate(
        displacement_scaled, 
        size=size, 
        mode=mode, 
        align_corners=True
    )
    
    return resized


def apply_affine_to_displacement(
    displacement: torch.Tensor,
    affine_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply affine transformation to displacement field
    
    Args:
        displacement: Displacement field [B, 3, D, H, W]
        affine_matrix: Affine matrix [B, 3, 4] or [B, 4, 4]
        
    Returns:
        transformed: Transformed displacement field
    """
    B, _, D, H, W = displacement.shape
    device = displacement.device
    
    # Create coordinate grid
    grid = SpatialTransformer.create_grid(B, D, H, W, device)
    coords = grid.view(B, -1, 3)  # [B, D*H*W, 3]
    
    # Add homogeneous coordinate
    ones = torch.ones(B, coords.shape[1], 1, device=device)
    coords_homo = torch.cat([coords, ones], dim=-1)  # [B, D*H*W, 4]
    
    # Apply affine transformation
    if affine_matrix.shape[-2:] == (3, 4):
        # 3x4 matrix
        transformed_coords = torch.bmm(coords_homo, affine_matrix.transpose(1, 2))
    else:
        # 4x4 matrix
        transformed_coords = torch.bmm(coords_homo, affine_matrix.transpose(1, 2))
        transformed_coords = transformed_coords[:, :, :3]
    
    # Compute displacement
    affine_disp = (transformed_coords - coords).view(B, D, H, W, 3)
    affine_disp = affine_disp.permute(0, 4, 1, 2, 3)
    
    # Compose with existing displacement
    transformer = SpatialTransformer()
    composed = affine_disp + transformer(displacement, affine_disp)
    
    return composed


def smooth_displacement_field(
    displacement: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: Optional[int] = None
) -> torch.Tensor:
    """
    Smooth displacement field with Gaussian filter
    
    Args:
        displacement: Displacement field [B, 3, D, H, W]
        sigma: Standard deviation of Gaussian
        kernel_size: Size of smoothing kernel (auto if None)
        
    Returns:
        smoothed: Smoothed displacement field
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()
    
    # Create 3D kernel
    g_3d = g_1d.view(1, 1, -1) * g_1d.view(1, -1, 1) * g_1d.view(-1, 1, 1)
    g_3d = g_3d.unsqueeze(0).unsqueeze(0)
    
    # Apply smoothing to each component
    device = displacement.device
    g_3d = g_3d.to(device)
    padding = kernel_size // 2
    
    smoothed = []
    for i in range(3):
        component = displacement[:, i:i+1]
        smoothed_component = F.conv3d(component, g_3d, padding=padding)
        smoothed.append(smoothed_component)
    
    return torch.cat(smoothed, dim=1) 