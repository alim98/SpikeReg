"""
Evaluation metrics for SpikeReg
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


def normalized_cross_correlation(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    window_size: int = 9,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute normalized cross-correlation between images
    
    Args:
        fixed: Fixed image [B, 1, D, H, W]
        warped: Warped moving image [B, 1, D, H, W]
        window_size: Size of local window
        eps: Small epsilon for stability
        
    Returns:
        ncc: NCC values [B]
    """
    B = fixed.shape[0]
    
    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, window_size, device=fixed.device)
    kernel = kernel / kernel.sum()
    
    # Compute local means
    pad = window_size // 2
    fixed_mean = F.conv3d(fixed, kernel, padding=pad)
    warped_mean = F.conv3d(warped, kernel, padding=pad)
    
    # Compute local variances and covariance
    fixed_sq = F.conv3d(fixed ** 2, kernel, padding=pad)
    warped_sq = F.conv3d(warped ** 2, kernel, padding=pad)
    fixed_warped = F.conv3d(fixed * warped, kernel, padding=pad)
    
    fixed_var = fixed_sq - fixed_mean ** 2
    warped_var = warped_sq - warped_mean ** 2
    covar = fixed_warped - fixed_mean * warped_mean
    
    # TODO: check if the NaN problem is resolved
    safe_sqrt = torch.sqrt(torch.clamp(fixed_var, min=0.0) * torch.clamp(warped_var, min=0.0))

    # Compute NCC
    ncc = covar / (safe_sqrt + eps)
    
    # Average over spatial dimensions
    ncc = ncc.view(B, -1).mean(dim=1)
    
    return ncc
def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    smooth: float = 1e-5
) -> torch.Tensor:
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 5 and target.size(1) == 1:
        target = target[:, 0]

    if pred.dim() == 4:
        if num_classes is None:
            num_classes = int(torch.max(pred.max(), target.max()).item()) + 1
        pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()
    elif pred.dim() == 5:
        if pred.size(1) != target.size(1):
            raise ValueError(f"pred and target channel mismatch: {pred.size(1)} vs {target.size(1)}")
        pred = pred.float()
        target = target.float()
    else:
        raise ValueError(f"Unsupported pred dim: {pred.dim()}")

    intersection = (pred * target).sum(dim=(2, 3, 4))
    pred_sum = pred.sum(dim=(2, 3, 4))
    target_sum = target.sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    dice = torch.clamp(dice, 0.0, 1.0)
    return dice


def jacobian_determinant(displacement: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian determinant of displacement field
    
    Args:
        displacement: Displacement field [B, 3, D, H, W]
        
    Returns:
        det: Jacobian determinant [B, D-2, H-2, W-2]
    """
    B, _, D, H, W = displacement.shape
    
    # Compute gradients using central differences
    # Note: This reduces spatial dimensions by 2
    grad_x = (displacement[:, :, 2:, 1:-1, 1:-1] - displacement[:, :, :-2, 1:-1, 1:-1]) / 2
    grad_y = (displacement[:, :, 1:-1, 2:, 1:-1] - displacement[:, :, 1:-1, :-2, 1:-1]) / 2
    grad_z = (displacement[:, :, 1:-1, 1:-1, 2:] - displacement[:, :, 1:-1, 1:-1, :-2]) / 2
    
    # Add identity
    grad_x[:, 0] += 1
    grad_y[:, 1] += 1
    grad_z[:, 2] += 1
    
    # Compute determinant
    # J = [[dx/dx, dx/dy, dx/dz],
    #      [dy/dx, dy/dy, dy/dz],
    #      [dz/dx, dz/dy, dz/dz]]
    
    det = (grad_x[:, 0] * grad_y[:, 1] * grad_z[:, 2] +
           grad_x[:, 1] * grad_y[:, 2] * grad_z[:, 0] +
           grad_x[:, 2] * grad_y[:, 0] * grad_z[:, 1] -
           grad_x[:, 2] * grad_y[:, 1] * grad_z[:, 0] -
           grad_x[:, 1] * grad_y[:, 0] * grad_z[:, 2] -
           grad_x[:, 0] * grad_y[:, 2] * grad_z[:, 1])
    
    return det


def jacobian_determinant_stats(displacement: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics of Jacobian determinant
    
    Args:
        displacement: Displacement field [B, 3, D, H, W]
        
    Returns:
        stats: Dictionary with determinant statistics
    """
    det = jacobian_determinant(displacement)
    
    stats = {
        'mean': det.mean().item(),
        'std': det.std().item(),
        'min': det.min().item(),
        'max': det.max().item(),
        'negative_fraction': (det < 0).float().mean().item(),
        'folding_fraction': (det <= 0).float().mean().item()
    }
    
    return stats

def target_registration_error(
    landmarks_fixed: torch.Tensor,
    landmarks_moving: torch.Tensor,
    displacement: torch.Tensor,
    spacing: Optional[Tuple[float, float, float]] = None
) -> torch.Tensor:
    device = displacement.device
    landmarks_fixed = landmarks_fixed.to(device).float()
    landmarks_moving = landmarks_moving.to(device).float()

    if displacement.dim() != 5 or displacement.size(1) != 3:
        raise ValueError(f"displacement must be [B,3,D,H,W], got {displacement.shape}")
    if displacement.size(0) != 1:
        displacement = displacement[:1]

    _, _, D, H, W = displacement.shape

    lm = landmarks_moving.clone()
    lm[:, 0] = (lm[:, 0] / (W - 1)) * 2 - 1
    lm[:, 1] = (lm[:, 1] / (H - 1)) * 2 - 1
    lm[:, 2] = (lm[:, 2] / (D - 1)) * 2 - 1

    landmarks_grid = lm.view(-1, 1, 1, 1, 3)

    disp_at_landmarks = F.grid_sample(
        displacement.expand(landmarks_grid.size(0), -1, -1, -1, -1),
        landmarks_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    disp_values = disp_at_landmarks.squeeze(-1).squeeze(-1).squeeze(-1)

    warped_landmarks = landmarks_moving + disp_values

    delta = warped_landmarks - landmarks_fixed

    if spacing is not None:
        spacing_tensor = torch.tensor(spacing, device=device).view(1, 3)
        delta = delta * spacing_tensor

    error = torch.linalg.norm(delta, dim=1)
    return error


def surface_distance(
    seg1: torch.Tensor,
    seg2: torch.Tensor,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Dict[str, float]:
    """
    Compute surface distance metrics between segmentations
    
    Args:
        seg1: First segmentation [D, H, W]
        seg2: Second segmentation [D, H, W]
        spacing: Voxel spacing
        
    Returns:
        metrics: Dictionary with surface distance metrics
    """
    # Extract surfaces using morphological operations
    kernel = torch.ones(1, 1, 3, 3, 3, device=seg1.device)
    
    # Erode to find boundaries
    seg1_float = seg1.float().unsqueeze(0).unsqueeze(0)
    seg2_float = seg2.float().unsqueeze(0).unsqueeze(0)
    
    seg1_eroded = F.conv3d(seg1_float, kernel, padding=1) < kernel.sum()
    seg2_eroded = F.conv3d(seg2_float, kernel, padding=1) < kernel.sum()
    seg1_bool = seg1_float > 0.5
    seg2_bool = seg2_float > 0.5
    surface1 = seg1_bool & ~seg1_eroded
    surface2 = seg2_bool & ~seg2_eroded

    # surface1 = seg1_float & ~seg1_eroded
    # surface2 = seg2_float & ~seg2_eroded
    
    # Get surface point coordinates
    coords1 = torch.nonzero(surface1.squeeze())
    coords2 = torch.nonzero(surface2.squeeze())
    
    if len(coords1) == 0 or len(coords2) == 0:
        return {
            'mean_surface_distance': 0.0,
            'hausdorff_distance': 0.0,
            'hausdorff_95': 0.0
        }
    
    # Apply spacing if provided
    if spacing is not None:
        spacing_tensor = torch.tensor(spacing, device=seg1.device)
        coords1 = coords1.float() * spacing_tensor
        coords2 = coords2.float() * spacing_tensor
    
    # Compute pairwise distances
    # This can be memory intensive for large surfaces
    if len(coords1) * len(coords2) > 1e8:
        # Subsample for large surfaces
        stride1 = max(1, len(coords1) // 10000)
        stride2 = max(1, len(coords2) // 10000)
        coords1 = coords1[::stride1]
        coords2 = coords2[::stride2]
    
    # Compute distances from surface1 to surface2
    dists_1to2 = torch.cdist(coords1.float(), coords2.float())
    min_dists_1to2 = dists_1to2.min(dim=1)[0]
    
    # Compute distances from surface2 to surface1
    dists_2to1 = dists_1to2.t()
    min_dists_2to1 = dists_2to1.min(dim=1)[0]
    
    # Combine distances
    all_dists = torch.cat([min_dists_1to2, min_dists_2to1])
    
    # Compute metrics
    metrics = {
        'mean_surface_distance': all_dists.mean().item(),
        'hausdorff_distance': all_dists.max().item(),
        'hausdorff_95': torch.quantile(all_dists, 0.95).item()
    }
    
    return metrics


def structural_similarity_index(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03,
    data_range: float = 1.0
) -> torch.Tensor:
    """
    Compute SSIM between two images
    
    Args:
        img1: First image [B, 1, D, H, W]
        img2: Second image [B, 1, D, H, W]
        window_size: Size of sliding window
        K1, K2: SSIM constants
        data_range: Range of data values
        
    Returns:
        ssim: SSIM values [B]
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=img1.device) - window_size // 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()
    
    window = g_1d.view(1, 1, -1) * g_1d.view(1, -1, 1) * g_1d.view(-1, 1, 1)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)  # right before the conv3d calls

    # Compute statistics
    mu1 = F.conv3d(img1, window, padding=window_size // 2)
    mu2 = F.conv3d(img2, window, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv3d(img1 ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv3d(img2 ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Average over spatial dimensions
    ssim = ssim_map.view(img1.shape[0], -1).mean(dim=1)
    
    return ssim


def compute_registration_metrics(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    warped: torch.Tensor,
    displacement: torch.Tensor,
    fixed_seg: Optional[torch.Tensor] = None,
    moving_seg: Optional[torch.Tensor] = None,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive registration metrics
    
    Args:
        fixed: Fixed image
        moving: Moving image
        warped: Warped moving image
        displacement: Displacement field
        fixed_seg: Fixed segmentation (optional)
        moving_seg: Moving segmentation (optional)
        spacing: Voxel spacing
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Image similarity metrics
    metrics['ncc'] = normalized_cross_correlation(fixed, warped).mean().item()
    metrics['mse'] = F.mse_loss(fixed, warped).item()
    metrics['ssim'] = structural_similarity_index(fixed, warped).mean().item()
    
    # Deformation regularity
    jac_stats = jacobian_determinant_stats(displacement)
    metrics.update({f'jacobian_{k}': v for k, v in jac_stats.items()})
    
    # Segmentation metrics if available
    if fixed_seg is not None and moving_seg is not None:
        # Warp segmentation
        from .warping import SpatialTransformer
        transformer = SpatialTransformer(mode='nearest')
        warped_seg = transformer(moving_seg.float(), displacement)
        
        # Dice score
        dice = dice_score(warped_seg, fixed_seg)
        metrics['dice'] = dice.mean().item()
        
        # Surface distance
        if warped_seg.shape[0] == 1:  # Single volume
            surf_metrics = surface_distance(
                warped_seg.squeeze().long(),
                fixed_seg.squeeze().long(),
                spacing
            )
            metrics.update({f'surface_{k}': v for k, v in surf_metrics.items()})
    
    return metrics 