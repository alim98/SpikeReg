"""
Preprocessing utilities for SpikeReg
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def normalize_volume(
    volume: torch.Tensor,
    percentile_range: Tuple[float, float] = (2, 98),
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize volume intensities to [0, 1] range
    
    Args:
        volume: Input volume tensor [B, C, D, H, W] or [C, D, H, W]
        percentile_range: Percentiles for clipping outliers
        eps: Small epsilon for numerical stability
        
    Returns:
        normalized: Normalized volume
    """
    # Handle both 4D and 5D tensors
    if volume.dim() == 4:
        # Single volume [C, D, H, W]
        C, D, H, W = volume.shape
        normalized = torch.zeros_like(volume)
        
        for c in range(C):
            # Get volume for this channel
            vol = volume[c]
            
            # Compute percentiles
            p_low = torch.quantile(vol.flatten(), percentile_range[0] / 100.0)
            p_high = torch.quantile(vol.flatten(), percentile_range[1] / 100.0)
            
            # Clip and normalize
            vol_clipped = torch.clamp(vol, p_low, p_high)
            normalized[c] = (vol_clipped - p_low) / (p_high - p_low + eps)
            
    elif volume.dim() == 5:
        # Batched volumes [B, C, D, H, W]
        B, C, D, H, W = volume.shape
        normalized = torch.zeros_like(volume)
        
        for b in range(B):
            for c in range(C):
                # Get volume for this batch and channel
                vol = volume[b, c]
                
                # Compute percentiles
                p_low = torch.quantile(vol.flatten(), percentile_range[0] / 100.0)
                p_high = torch.quantile(vol.flatten(), percentile_range[1] / 100.0)
                
                # Clip and normalize
                vol_clipped = torch.clamp(vol, p_low, p_high)
                normalized[b, c] = (vol_clipped - p_low) / (p_high - p_low + eps)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {volume.dim()}D tensor with shape {volume.shape}")
    
    return normalized


def poisson_rate_coding(
    intensities: torch.Tensor,
    time_window: int,
    dt: float = 1.0,
    max_rate: float = 1.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Convert continuous intensities to spike trains using Poisson rate coding
    
    Args:
        intensities: Input intensities [B, C, D, H, W] in range [0, 1]
        time_window: Number of time steps
        dt: Time step duration in milliseconds
        max_rate: Maximum firing rate (spikes per millisecond)
        seed: Random seed for reproducibility
        
    Returns:
        spikes: Binary spike tensor [B, T, C, D, H, W]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    B, C, D, H, W = intensities.shape
    device = intensities.device
    
    # Scale intensities to firing rates
    rates = intensities * max_rate
    
    # Generate spikes for each time step
    spikes = []
    for t in range(time_window):
        # Poisson sampling: spike if random < rate * dt
        random_vals = torch.rand(B, C, D, H, W, device=device)
        spike_t = (random_vals < rates * dt).float()
        spikes.append(spike_t)
    
    # Stack along time dimension
    spike_tensor = torch.stack(spikes, dim=1)  # [B, T, C, D, H, W]
    
    return spike_tensor


def temporal_rate_coding(
    intensities: torch.Tensor,
    time_window: int,
    coding_scheme: str = "time_to_first_spike",
    tau: float = 10.0
) -> torch.Tensor:
    """
    Convert intensities to spike trains using temporal coding
    
    Args:
        intensities: Input intensities [B, C, D, H, W] in range [0, 1]
        time_window: Number of time steps
        coding_scheme: Type of temporal coding
        tau: Time constant for coding
        
    Returns:
        spikes: Binary spike tensor [B, T, C, D, H, W]
    """
    B, C, D, H, W = intensities.shape
    device = intensities.device
    
    if coding_scheme == "time_to_first_spike":
        # Higher intensity = earlier spike
        # Time to spike = tau * (1 - intensity)
        spike_times = (tau * (1 - intensities)).long()
        spike_times = torch.clamp(spike_times, 0, time_window - 1)
        
        # Create spike tensor
        spikes = torch.zeros(B, time_window, C, D, H, W, device=device)
        
        # Set spikes at computed times
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            t = spike_times[b, c, d, h, w]
                            if t < time_window:
                                spikes[b, t, c, d, h, w] = 1.0
    
    elif coding_scheme == "phase":
        # Phase coding: spike timing encodes intensity as phase
        phases = intensities * 2 * np.pi
        spikes = []
        
        for t in range(time_window):
            # Oscillatory reference signal
            ref_phase = 2 * np.pi * t / time_window
            
            # Spike when phase matches reference
            spike_t = (torch.cos(phases - ref_phase) > 0.9).float()
            spikes.append(spike_t)
        
        spikes = torch.stack(spikes, dim=1)
    
    else:
        raise ValueError(f"Unknown coding scheme: {coding_scheme}")
    
    return spikes


def burst_coding(
    intensities: torch.Tensor,
    time_window: int,
    burst_duration: int = 3,
    max_bursts: int = 5
) -> torch.Tensor:
    """
    Convert intensities to spike trains using burst coding
    
    Higher intensities produce more burst events
    
    Args:
        intensities: Input intensities [B, C, D, H, W] in range [0, 1]
        time_window: Number of time steps
        burst_duration: Duration of each burst
        max_bursts: Maximum number of bursts
        
    Returns:
        spikes: Binary spike tensor [B, T, C, D, H, W]
    """
    B, C, D, H, W = intensities.shape
    device = intensities.device
    
    # Number of bursts proportional to intensity
    num_bursts = (intensities * max_bursts).long()
    
    # Generate spike tensor
    spikes = torch.zeros(B, time_window, C, D, H, W, device=device)
    
    # Random burst start times
    for b in range(B):
        for c in range(C):
            burst_starts = torch.randint(
                0, time_window - burst_duration,
                (D, H, W, max_bursts),
                device=device
            )
            
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        n_burst = num_bursts[b, c, d, h, w]
                        for burst_idx in range(n_burst):
                            start_t = burst_starts[d, h, w, burst_idx]
                            end_t = min(start_t + burst_duration, time_window)
                            spikes[b, start_t:end_t, c, d, h, w] = 1.0
    
    return spikes


def adaptive_rate_coding(
    intensities: torch.Tensor,
    time_window: int,
    adaptation_tau: float = 5.0,
    refractory_period: int = 2
) -> torch.Tensor:
    """
    Rate coding with adaptation and refractory period
    
    Firing rate adapts over time to maintain efficiency
    
    Args:
        intensities: Input intensities [B, C, D, H, W] in range [0, 1]
        time_window: Number of time steps
        adaptation_tau: Time constant for rate adaptation
        refractory_period: Minimum time between spikes
        
    Returns:
        spikes: Binary spike tensor [B, T, C, D, H, W]
    """
    B, C, D, H, W = intensities.shape
    device = intensities.device
    
    # Initialize
    spikes = torch.zeros(B, time_window, C, D, H, W, device=device)
    last_spike = torch.full((B, C, D, H, W), -refractory_period, device=device)
    adaptation = torch.zeros_like(intensities)
    
    for t in range(time_window):
        # Check refractory period
        can_spike = (t - last_spike) >= refractory_period
        
        # Adapted rate
        rate = intensities * torch.exp(-adaptation / adaptation_tau)
        
        # Generate spikes
        random_vals = torch.rand_like(intensities)
        spike_t = (random_vals < rate) & can_spike
        
        # Update state
        spikes[:, t] = spike_t.float()
        last_spike[spike_t] = t
        adaptation[spike_t] += 1.0
        
        # Decay adaptation
        adaptation *= torch.exp(-1.0 / adaptation_tau)
    
    return spikes


class PatchNormalizer:
    """
    Normalize patches with consistent statistics
    
    Useful for ensuring stable spike generation across patches
    """
    
    def __init__(
        self,
        method: str = "percentile",
        percentile_range: Tuple[float, float] = (2, 98),
        global_stats: bool = False
    ):
        self.method = method
        self.percentile_range = percentile_range
        self.global_stats = global_stats
        self.stats = {}
    
    def fit(self, patches: torch.Tensor):
        """Compute normalization statistics from patches"""
        if self.method == "percentile":
            if self.global_stats:
                # Global percentiles across all patches
                flat = patches.flatten()
                self.stats['p_low'] = torch.quantile(flat, self.percentile_range[0] / 100.0)
                self.stats['p_high'] = torch.quantile(flat, self.percentile_range[1] / 100.0)
            else:
                # Per-patch percentiles
                B = patches.shape[0]
                p_lows = []
                p_highs = []
                
                for b in range(B):
                    flat = patches[b].flatten()
                    p_lows.append(torch.quantile(flat, self.percentile_range[0] / 100.0))
                    p_highs.append(torch.quantile(flat, self.percentile_range[1] / 100.0))
                
                self.stats['p_lows'] = torch.stack(p_lows)
                self.stats['p_highs'] = torch.stack(p_highs)
        
        elif self.method == "zscore":
            if self.global_stats:
                self.stats['mean'] = patches.mean()
                self.stats['std'] = patches.std()
            else:
                self.stats['means'] = patches.mean(dim=(1, 2, 3, 4))
                self.stats['stds'] = patches.std(dim=(1, 2, 3, 4))
    
    def transform(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply normalization to patches"""
        if self.method == "percentile":
            if self.global_stats:
                normalized = torch.clamp(patches, self.stats['p_low'], self.stats['p_high'])
                normalized = (normalized - self.stats['p_low']) / \
                           (self.stats['p_high'] - self.stats['p_low'] + 1e-8)
            else:
                B = patches.shape[0]
                normalized = torch.zeros_like(patches)
                
                for b in range(B):
                    p_low = self.stats['p_lows'][b]
                    p_high = self.stats['p_highs'][b]
                    normalized[b] = torch.clamp(patches[b], p_low, p_high)
                    normalized[b] = (normalized[b] - p_low) / (p_high - p_low + 1e-8)
        
        elif self.method == "zscore":
            if self.global_stats:
                normalized = (patches - self.stats['mean']) / (self.stats['std'] + 1e-8)
            else:
                B = patches.shape[0]
                normalized = torch.zeros_like(patches)
                
                for b in range(B):
                    mean = self.stats['means'][b].view(1, 1, 1, 1)
                    std = self.stats['stds'][b].view(1, 1, 1, 1)
                    normalized[b] = (patches[b] - mean) / (std + 1e-8)
            
            # Clip to reasonable range and shift to [0, 1]
            normalized = torch.clamp(normalized, -3, 3)
            normalized = (normalized + 3) / 6
        
        return normalized
    
    def fit_transform(self, patches: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step"""
        self.fit(patches)
        return self.transform(patches) 