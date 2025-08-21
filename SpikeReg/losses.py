"""
Loss functions for SpikeReg training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class NormalizedCrossCorrelation(nn.Module):
    """
    Normalized Cross-Correlation loss for image similarity
    
    NCC = sum((F - mean(F)) * (M - mean(M))) / (std(F) * std(M) * N)
    """
    
    def __init__(self, window_size: int = 9, eps: float = 1e-8):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        
        # Create averaging kernel
        self.register_buffer('kernel', self._create_window_3d(window_size))
    
    def _create_window_3d(self, window_size: int) -> torch.Tensor:
        """Create 3D averaging window"""
        kernel = torch.ones(1, 1, window_size, window_size, window_size)
        kernel = kernel / kernel.sum()
        return kernel
    
    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute NCC loss
        
        Args:
            fixed: Fixed image [B, 1, D, H, W]
            warped: Warped moving image [B, 1, D, H, W]
            
        Returns:
            ncc_loss: 1 - NCC (to minimize)
        """
        B, C, D, H, W = fixed.shape
        
        # Compute local means
        pad = self.window_size // 2
        kernel = self.kernel
        if kernel.device != fixed.device:
            kernel = kernel.to(fixed.device)
        fixed_mean = F.conv3d(fixed, kernel, padding=pad)
        warped_mean = F.conv3d(warped, kernel, padding=pad)
        
        # Compute local variances and covariance
        fixed_sq = F.conv3d(fixed ** 2, kernel, padding=pad)
        warped_sq = F.conv3d(warped ** 2, kernel, padding=pad)
        fixed_warped = F.conv3d(fixed * warped, kernel, padding=pad)
        
        fixed_var = fixed_sq - fixed_mean ** 2
        warped_var = warped_sq - warped_mean ** 2
        covar = fixed_warped - fixed_mean * warped_mean
        
        # Compute NCC
        ncc = covar / (torch.sqrt(fixed_var * warped_var) + self.eps)
        
        # Return negative NCC for minimization
        return -ncc.mean()


class MutualInformation(nn.Module):
    """
    Mutual Information loss for multi-modal registration
    
    MI = H(F) + H(M) - H(F, M)
    where H is entropy
    """
    
    def __init__(self, num_bins: int = 32, sigma: float = 1.0):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
    
    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute MI loss using differentiable histograms
        
        Args:
            fixed: Fixed image [B, 1, D, H, W]
            warped: Warped moving image [B, 1, D, H, W]
            
        Returns:
            mi_loss: -MI (to minimize)
        """
        B = fixed.shape[0]
        
        # Flatten spatial dimensions
        fixed_flat = fixed.view(B, -1)
        warped_flat = warped.view(B, -1)
        
        # Normalize to [0, 1]
        fixed_norm = (fixed_flat - fixed_flat.min(dim=1, keepdim=True)[0]) / \
                     (fixed_flat.max(dim=1, keepdim=True)[0] - fixed_flat.min(dim=1, keepdim=True)[0] + 1e-8)
        warped_norm = (warped_flat - warped_flat.min(dim=1, keepdim=True)[0]) / \
                      (warped_flat.max(dim=1, keepdim=True)[0] - warped_flat.min(dim=1, keepdim=True)[0] + 1e-8)
        
        # Compute soft histograms using Gaussian kernels
        bins = torch.linspace(0, 1, self.num_bins, device=fixed.device)
        
        # Compute marginal histograms
        hist_fixed = self._soft_histogram(fixed_norm, bins)
        hist_warped = self._soft_histogram(warped_norm, bins)
        
        # Compute joint histogram
        hist_joint = self._soft_joint_histogram(fixed_norm, warped_norm, bins)
        
        # Compute entropies
        h_fixed = self._entropy(hist_fixed)
        h_warped = self._entropy(hist_warped)
        h_joint = self._entropy(hist_joint)
        
        # Compute MI
        mi = h_fixed + h_warped - h_joint
        
        return -mi.mean()
    
    def _soft_histogram(self, values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Compute soft histogram using Gaussian kernels"""
        B, N = values.shape
        num_bins = bins.shape[0]
        
        # Expand dimensions for broadcasting
        values = values.unsqueeze(2)  # [B, N, 1]
        bins = bins.unsqueeze(0).unsqueeze(0)  # [1, 1, num_bins]
        
        # Compute distances to bins
        distances = (values - bins) ** 2
        
        # Apply Gaussian kernel
        weights = torch.exp(-distances / (2 * self.sigma ** 2))
        
        # Normalize and sum
        hist = weights.sum(dim=1)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        
        return hist
    
    def _soft_joint_histogram(
        self, 
        values1: torch.Tensor, 
        values2: torch.Tensor, 
        bins: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft joint histogram"""
        B, N = values1.shape
        num_bins = bins.shape[0]
        
        # Expand dimensions
        values1 = values1.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1]
        values2 = values2.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1]
        bins1 = bins.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1, 1, num_bins, 1]
        bins2 = bins.unsqueeze(0).unsqueeze(0).unsqueeze(1)  # [1, 1, 1, num_bins]
        
        # Compute 2D Gaussian kernel
        distances1 = (values1 - bins1) ** 2
        distances2 = (values2 - bins2) ** 2
        weights = torch.exp(-(distances1 + distances2) / (2 * self.sigma ** 2))
        
        # Sum over pixels
        hist = weights.sum(dim=1)
        hist = hist / (hist.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        return hist
    
    def _entropy(self, hist: torch.Tensor) -> torch.Tensor:
        """Compute entropy from histogram"""
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        
        if hist.dim() == 2:  # Marginal histogram
            entropy = -(hist * torch.log(hist)).sum(dim=1)
        else:  # Joint histogram
            entropy = -(hist * torch.log(hist)).sum(dim=(1, 2))
        
        return entropy


class BendingEnergy(nn.Module):
    """
    Bending energy regularization for smooth deformations
    
    Penalizes second-order derivatives of displacement field
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute bending energy
        
        Args:
            displacement: Displacement field [B, 3, D, H, W]
            
        Returns:
            bending_energy: Scalar loss value
        """
        # Compute second-order derivatives
        dx = torch.gradient(displacement, dim=2)[0]
        dy = torch.gradient(displacement, dim=3)[0]
        dz = torch.gradient(displacement, dim=4)[0]
        
        dxx = torch.gradient(dx, dim=2)[0]
        dyy = torch.gradient(dy, dim=3)[0]
        dzz = torch.gradient(dz, dim=4)[0]
        dxy = torch.gradient(dx, dim=3)[0]
        dxz = torch.gradient(dx, dim=4)[0]
        dyz = torch.gradient(dy, dim=4)[0]
        
        # Compute bending energy
        bending = (dxx ** 2 + dyy ** 2 + dzz ** 2 + 
                   2 * dxy ** 2 + 2 * dxz ** 2 + 2 * dyz ** 2)
        
        return self.weight * bending.mean()


class DiffusionRegularizer(nn.Module):
    """
    Diffusion regularization for smooth deformations
    
    Penalizes first-order derivatives (gradient magnitude)
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion regularization
        
        Args:
            displacement: Displacement field [B, 3, D, H, W]
            
        Returns:
            diffusion_loss: Scalar loss value
        """
        # Compute gradients
        dx = torch.gradient(displacement, dim=2)[0]
        dy = torch.gradient(displacement, dim=3)[0]
        dz = torch.gradient(displacement, dim=4)[0]
        
        # Compute gradient magnitude
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-8)
        
        return self.weight * grad_mag.mean()


class SpikingRegularizer(nn.Module):
    """
    Regularization specific to spiking neural networks
    
    Penalizes excessive spike activity to promote energy efficiency
    """
    
    def __init__(
        self, 
        spike_weight: float = 0.001,
        balance_weight: float = 0.01,
        target_rate: float = 0.1
    ):
        super().__init__()
        self.spike_weight = spike_weight
        self.balance_weight = balance_weight
        self.target_rate = target_rate
    
    def forward(self, spike_counts: Dict[str, float]) -> torch.Tensor:
        """
        Compute spiking regularization
        
        Args:
            spike_counts: Dictionary of spike rates per layer
            
        Returns:
            spike_loss: Scalar loss value
        """
        total_spikes = 0
        spike_imbalance = 0
        
        print("\n[SpikingRegularizer] Debug Info:")
        print("Spike counts input:", spike_counts)
        
        for layer_name, spike_rate in spike_counts.items():
            total_spikes += spike_rate
            
            # Penalize deviation from target rate
            spike_imbalance += (spike_rate - self.target_rate) ** 2
        
        print("  Total spikes:", total_spikes)
        print("  Spike imbalance:", spike_imbalance)
        # L1 penalty on total spikes
        spike_loss = self.spike_weight * total_spikes
        print("  Spike loss (L1):", spike_loss)

        # L2 penalty on rate imbalance
        balance_loss = self.balance_weight * spike_imbalance
        print("  Balance loss (L2):", balance_loss)
        
        return spike_loss + balance_loss


class SpikeRegLoss(nn.Module):
    """
    Combined loss function for SpikeReg training
    
    Combines similarity, regularization, and spike penalties
    """
    
    def __init__(
        self,
        similarity_type: str = "ncc",
        similarity_weight: float = 1.0,
        regularization_type: str = "bending",
        regularization_weight: float = 0.01,
        spike_weight: float = 0.001,
        spike_balance_weight: float = 0.01,
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        
        # Similarity loss
        if similarity_type == "ncc":
            self.similarity_loss = NormalizedCrossCorrelation()
        elif similarity_type == "mse":
            self.similarity_loss = nn.MSELoss()
        elif similarity_type == "mi":
            self.similarity_loss = MutualInformation()
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        # Regularization loss
        if regularization_type == "bending":
            self.regularization_loss = BendingEnergy(regularization_weight)
        elif regularization_type == "diffusion":
            self.regularization_loss = DiffusionRegularizer(regularization_weight)
        else:
            self.regularization_loss = None
        
        # Spike regularization
        self.spike_regularizer = SpikingRegularizer(
            spike_weight, spike_balance_weight, target_spike_rate
        )
        
        self.similarity_weight = similarity_weight
    
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        displacement: torch.Tensor,
        warped: torch.Tensor,
        spike_counts: Dict[str, float]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss
        
        Args:
            fixed: Fixed image [B, 1, D, H, W]
            moving: Moving image [B, 1, D, H, W]
            displacement: Displacement field [B, 3, D, H, W]
            warped: Warped moving image [B, 1, D, H, W]
            spike_counts: Spike statistics per layer
            
        Returns:
            total_loss: Combined scalar loss
            loss_components: Dictionary of individual loss components
        """
        # Similarity loss
        sim_loss = self.similarity_loss(fixed, warped)
        
        # Regularization loss
        if self.regularization_loss is not None:
            reg_loss = self.regularization_loss(displacement)
        else:
            reg_loss = torch.tensor(0.0, device=fixed.device)
        
        # Spike regularization
        spike_loss = self.spike_regularizer(spike_counts)
        
        # Combine losses
        total_loss = (self.similarity_weight * sim_loss + 
                     reg_loss + 
                     spike_loss)
        # check if total_loss is NaN
        if torch.isnan(total_loss).any():
            print("Warning! Total loss contains NaN values.")
            
        # Store components for logging
        loss_components = {
            'similarity': sim_loss,
            'regularization': reg_loss,
            'spike': spike_loss,
            'total': total_loss
        }
        
        return total_loss, loss_components 