"""
Iterative registration policy and inference for SpikeReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from .models import SpikeRegUNet
from .utils.warping import SpatialTransformer
from .utils.patch_utils import extract_patches, stitch_patches
from .utils.metrics import normalized_cross_correlation


class IterativeRegistration(nn.Module):
    """
    Iterative residual registration policy
    
    Progressively refines deformation field through multiple iterations
    """
    
    def __init__(
        self,
        model: SpikeRegUNet,
        num_iterations: int = 10,
        early_stop_threshold: float = 0.001,
        convergence_window: int = 3
    ):
        super().__init__()
        
        self.model = model
        self.num_iterations = num_iterations
        self.early_stop_threshold = early_stop_threshold
        self.convergence_window = convergence_window
        
        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer()
        
        # History tracking
        self.reset_history()
    
    def reset_history(self):
        """Reset iteration history"""
        self.displacement_history = []
        self.similarity_history = []
        self.spike_count_history = []
        self.spike_count_history_number = []
    
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        initial_displacement: Optional[torch.Tensor] = None,
        return_all_iterations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Iterative registration forward pass
        
        Args:
            fixed: Fixed image [B, 1, D, H, W]
            moving: Moving image [B, 1, D, H, W]
            initial_displacement: Initial displacement field [B, 3, D, H, W]
            return_all_iterations: Whether to return results from all iterations
            
        Returns:
            output: Dictionary containing:
                - displacement: Final displacement field
                - warped: Final warped moving image
                - iterations: Number of iterations performed
                - similarity_scores: Similarity at each iteration
                - converged: Whether early stopping occurred
        """
        B, _, D, H, W = fixed.shape
        device = fixed.device
        
        # Initialize displacement field
        if initial_displacement is None:
            displacement = torch.zeros(B, 3, D, H, W, device=device)
        else:
            displacement = initial_displacement.clone()
        
        # Reset history
        self.reset_history()
        
        # Iterative refinement
        converged = False
        num_iters = 0
        
        for iteration in range(self.num_iterations):
            # Warp moving image with current displacement
            warped = self.spatial_transformer(moving, displacement)
            
            # Compute similarity
            similarity = normalized_cross_correlation(fixed, warped)
            self.similarity_history.append(similarity.mean().item())
            
            # Check convergence
            if iteration >= self.convergence_window:
                recent_similarities = self.similarity_history[-self.convergence_window:]
                improvement = max(recent_similarities) - min(recent_similarities)
                if improvement < self.early_stop_threshold:
                    converged = True
                    num_iters = iteration + 1
                    break
            
            # Predict residual displacement
            self.model.reset_all_neurons()
            model_output = self.model(fixed, warped)
            residual_displacement = model_output['displacement']
            
            # Update displacement field
            displacement = displacement + residual_displacement
            
            # Store history
            self.displacement_history.append(displacement.clone())
            self.spike_count_history.append(model_output['spike_counts'])
            self.spike_count_history_number.append(model_output['spike_counts_number'])
            
            num_iters = iteration + 1
            # check nan values in displacement
            if torch.isnan(displacement).any():
                print(f"Warning! NaN detected in displacement at iteration {iteration}.")
        
        # Final warping
        warped_final = self.spatial_transformer(moving, displacement)
        
        # Prepare output
        output = {
            'displacement': displacement,
            'warped': warped_final,
            'iterations': num_iters,
            'similarity_scores': torch.tensor(self.similarity_history),
            'converged': converged
        }
        # check if NaN values are present in displacement and warped
        hasNaN = False
        if torch.isnan(displacement).any() or torch.isnan(warped_final).any():
            print("Warning! Displacement or warped image contains NaN values.")
            hasNaN = True
        
        
        if return_all_iterations or hasNaN:
            output['displacement_history'] = self.displacement_history
            output['spike_count_history'] = self.spike_count_history
            output['similarity_history'] = self.similarity_history
        output['spike_count_history_number'] = self.spike_count_history_number
        output['hasNaN'] = hasNaN
        
        return output


class SpikeRegInference:
    """
    High-level inference interface for SpikeReg
    
    Handles patch extraction, model inference, and field stitching
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        device: str = 'cuda',
        patch_size: int = 32,
        patch_stride: int = 16,
        batch_size: int = 8
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.batch_size = batch_size
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        if config is None:
            config = checkpoint.get('config', {})
        
        self.model = SpikeRegUNet(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create iterative registration module
        self.registration = IterativeRegistration(
            self.model,
            num_iterations=config.get('num_iterations', 10),
            early_stop_threshold=config.get('early_stop_threshold', 0.001)
        )
        
        self.config = config
    
    def preprocess_volume(
        self,
        volume: np.ndarray,
        percentile_clip: Tuple[float, float] = (2, 98)
    ) -> torch.Tensor:
        """
        Preprocess medical volume for registration
        
        Args:
            volume: Input volume array
            percentile_clip: Percentiles for intensity clipping
            
        Returns:
            preprocessed: Normalized tensor
        """
        # Clip intensities
        p_low, p_high = np.percentile(volume, percentile_clip)
        volume = np.clip(volume, p_low, p_high)
        
        # Normalize to [0, 1]
        volume = (volume - p_low) / (p_high - p_low + 1e-8)
        
        # Convert to tensor
        tensor = torch.from_numpy(volume).float()
        
        # Add batch and channel dimensions if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def register(
        self,
        fixed: Union[np.ndarray, torch.Tensor],
        moving: Union[np.ndarray, torch.Tensor],
        return_all_patches: bool = False,
        progress_bar: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Register moving image to fixed image
        
        Args:
            fixed: Fixed/reference volume
            moving: Moving volume to be registered
            return_all_patches: Whether to return patch-wise results
            progress_bar: Whether to show progress bar
            
        Returns:
            displacement_field: Full resolution displacement field
            or dictionary with additional outputs if return_all_patches=True
        """
        # Preprocess inputs
        if isinstance(fixed, np.ndarray):
            fixed = self.preprocess_volume(fixed)
        if isinstance(moving, np.ndarray):
            moving = self.preprocess_volume(moving)
        
        # Extract patches
        fixed_patches, patch_coords = extract_patches(
            fixed, self.patch_size, self.patch_stride
        )
        moving_patches, _ = extract_patches(
            moving, self.patch_size, self.patch_stride
        )
        
        num_patches = len(fixed_patches)
        all_displacements = []
        all_similarities = []
        all_iterations = []
        
        # Process patches in batches
        with torch.no_grad():
            iterator = range(0, num_patches, self.batch_size)
            if progress_bar:
                iterator = tqdm(iterator, desc="Processing patches")
            
            for i in iterator:
                batch_end = min(i + self.batch_size, num_patches)
                
                # Get batch
                fixed_batch = torch.stack(fixed_patches[i:batch_end])
                moving_batch = torch.stack(moving_patches[i:batch_end])
                
                # Run registration
                output = self.registration(fixed_batch, moving_batch)
                
                # Store results
                all_displacements.extend(output['displacement'].cpu())
                all_similarities.extend(output['similarity_scores'].cpu())
                all_iterations.extend([output['iterations']] * (batch_end - i))
        
        # Stitch patches into full displacement field
        full_shape = fixed.shape[2:]  # [D, H, W]
        displacement_field = stitch_patches(
            all_displacements,
            patch_coords,
            full_shape,
            self.patch_size,
            self.patch_stride,
            blend_mode='cosine'
        )
        
        # Apply optional smoothing
        if self.config.get('smooth_displacement', True):
            displacement_field = self._smooth_displacement_field(displacement_field)
        
        # Convert to numpy
        displacement_np = displacement_field.cpu().numpy()
        
        if return_all_patches:
            return {
                'displacement_field': displacement_np,
                'patch_displacements': [d.cpu().numpy() for d in all_displacements],
                'patch_similarities': all_similarities,
                'patch_iterations': all_iterations,
                'patch_coordinates': patch_coords
            }
        else:
            return displacement_np
    
    def apply_deformation(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        displacement_field: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Apply displacement field to warp a volume
        
        Args:
            volume: Volume to warp
            displacement_field: Displacement field
            
        Returns:
            warped: Warped volume
        """
        # Convert to tensors
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
            if volume.dim() == 3:
                volume = volume.unsqueeze(0).unsqueeze(0)
        
        if isinstance(displacement_field, np.ndarray):
            displacement_field = torch.from_numpy(displacement_field).float()
            if displacement_field.dim() == 4:
                displacement_field = displacement_field.unsqueeze(0)
        
        # Move to device
        volume = volume.to(self.device)
        displacement_field = displacement_field.to(self.device)
        
        # Apply warping
        spatial_transformer = SpatialTransformer()
        warped = spatial_transformer(volume, displacement_field)
        
        # Convert back to numpy
        warped_np = warped.squeeze().cpu().numpy()
        
        return warped_np
    
    def _smooth_displacement_field(
        self,
        displacement: torch.Tensor,
        kernel_size: int = 3,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Smooth displacement field with Gaussian kernel
        
        Args:
            displacement: Displacement field tensor
            kernel_size: Size of smoothing kernel
            sigma: Standard deviation of Gaussian
            
        Returns:
            smoothed: Smoothed displacement field
        """
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g_1d = g_1d / g_1d.sum()
        
        # Create 3D kernel
        g_3d = g_1d.view(1, 1, -1) * g_1d.view(1, -1, 1) * g_1d.view(-1, 1, 1)
        g_3d = g_3d.unsqueeze(0).unsqueeze(0)
        
        # Apply smoothing to each component
        smoothed = []
        for i in range(3):
            component = displacement[:, i:i+1]
            smoothed_component = F.conv3d(
                component,
                g_3d.to(displacement.device),
                padding=kernel_size // 2
            )
            smoothed.append(smoothed_component)
        
        return torch.cat(smoothed, dim=1)
    
    def compute_jacobian_determinant(
        self,
        displacement_field: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute Jacobian determinant of displacement field
        
        Useful for checking if transformation is diffeomorphic (det > 0)
        
        Args:
            displacement_field: Displacement field [3, D, H, W]
            
        Returns:
            jacobian_det: Jacobian determinant map
        """
        if isinstance(displacement_field, np.ndarray):
            displacement_field = torch.from_numpy(displacement_field).float()
        
        if displacement_field.dim() == 4:
            displacement_field = displacement_field.unsqueeze(0)
        
        # Move to device
        displacement_field = displacement_field.to(self.device)
        
        # Compute gradients
        B, C, D, H, W = displacement_field.shape
        
        # Add identity to get deformation field
        identity = torch.stack(torch.meshgrid(
            torch.arange(D, device=self.device, dtype=torch.float32),
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        ), dim=0).unsqueeze(0)
        
        deformation = displacement_field + identity
        
        # Compute spatial gradients
        grad_x = torch.gradient(deformation, dim=2)[0]
        grad_y = torch.gradient(deformation, dim=3)[0]
        grad_z = torch.gradient(deformation, dim=4)[0]
        
        # Compute Jacobian determinant
        # J = [[dx/dx, dx/dy, dx/dz],
        #      [dy/dx, dy/dy, dy/dz],
        #      [dz/dx, dz/dy, dz/dz]]
        
        det = (grad_x[:, 0] * grad_y[:, 1] * grad_z[:, 2] +
               grad_x[:, 1] * grad_y[:, 2] * grad_z[:, 0] +
               grad_x[:, 2] * grad_y[:, 0] * grad_z[:, 1] -
               grad_x[:, 2] * grad_y[:, 1] * grad_z[:, 0] -
               grad_x[:, 1] * grad_y[:, 0] * grad_z[:, 2] -
               grad_x[:, 0] * grad_y[:, 2] * grad_z[:, 1])
        
        return det.squeeze().cpu().numpy() 