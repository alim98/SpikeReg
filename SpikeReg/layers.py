"""
Spiking neural network layers for SpikeReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
from .neurons import LIFNeuron, AdaptiveLIFNeuron, LateralInhibition


class SpikingConv3d(nn.Module):
    """
    3D Spiking Convolutional Layer
    
    Combines spatial convolution with LIF neurons
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        tau_u: float = 0.9,
        v_th: float = 1.0,
        lateral_inhibition: bool = False,
        k_fraction: float = 0.1,
        adaptive: bool = False,
        **neuron_kwargs
    ):
        super().__init__()
        
        # Convolutional layer
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # Batch normalization (for ANN-to-SNN conversion)
        self.bn = nn.BatchNorm3d(out_channels)
        
        # Spiking neurons
        if adaptive:
            self.neurons = AdaptiveLIFNeuron(tau_u=tau_u, v_th=v_th, **neuron_kwargs)
        else:
            self.neurons = LIFNeuron(tau_u=tau_u, v_th=v_th, **neuron_kwargs)
        
        # Lateral inhibition
        self.lateral_inhibition = lateral_inhibition
        if lateral_inhibition:
            self.inhibition = LateralInhibition(k_fraction=k_fraction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spiking conv layer
        
        Args:
            x: Input spike tensor [B, C, D, H, W]
            
        Returns:
            spikes: Output spike tensor [B, C', D', H', W']
        """
        # Spatial convolution
        x = self.conv(x)
        x = self.bn(x)
        
        # Apply lateral inhibition to membrane potentials if enabled
        if self.lateral_inhibition and hasattr(self.neurons, 'membrane') and self.neurons.membrane is not None:
            self.neurons.membrane = self.inhibition(self.neurons.membrane)
        
        # Generate spikes through LIF neurons
        spikes = self.neurons(x)
        
        return spikes
    
    def reset_neurons(self):
        """Reset neuron states"""
        self.neurons.reset_state(1, 1, 1, 1, 1)  # Will be resized on first forward


class SpikingTransposeConv3d(nn.Module):
    """
    3D Spiking Transposed Convolutional Layer for upsampling
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 2,
        stride: Union[int, Tuple[int, int, int]] = 2,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        tau_u: float = 0.9,
        v_th: float = 1.0,
        **neuron_kwargs
    ):
        super().__init__()
        
        # Transposed convolutional layer
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm3d(out_channels)
        
        # Spiking neurons
        self.neurons = LIFNeuron(tau_u=tau_u, v_th=v_th, **neuron_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking transpose conv layer"""
        x = self.conv(x)
        x = self.bn(x)
        spikes = self.neurons(x)
        return spikes
    
    def reset_neurons(self):
        """Reset neuron states"""
        self.neurons.reset_state(1, 1, 1, 1, 1)


class SpikingEncoderBlock(nn.Module):
    """
    Encoder block for Spiking U-Net
    
    Contains:
    - Spiking Conv3d with stride for downsampling
    - Optional residual connection
    - Lateral inhibition for sparsity
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        tau_u: float = 0.9,
        time_window: int = 10,
        lateral_inhibition: bool = True,
        residual: bool = False
    ):
        super().__init__()
        
        self.time_window = time_window
        self.residual = residual
        
        # Main path
        self.conv = SpikingConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size//2,
            tau_u=tau_u, lateral_inhibition=lateral_inhibition
        )
        
        # Residual path (if enabled)
        if residual and stride == 1:
            self.shortcut = nn.Identity()
        elif residual:
            self.shortcut = SpikingConv3d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, tau_u=tau_u
            )
        else:
            self.shortcut = None
    
    def forward(self, x: torch.Tensor, time_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block
        
        Args:
            x: Input spike tensor or continuous tensor
            time_steps: Number of time steps to process
            
        Returns:
            spikes: Accumulated spikes over time window
            skip: Skip connection output (spike rates)
        """
        if time_steps is None:
            time_steps = self.time_window
        
        # Reset neurons
        self.conv.reset_neurons()
        if self.shortcut is not None:
            self.shortcut.reset_neurons()
        
        # Process over time
        spike_outputs = []
        for t in range(time_steps):
            # Main path
            spikes = self.conv(x)
            
            # Residual connection
            if self.shortcut is not None:
                spikes = spikes + self.shortcut(x)
            
            spike_outputs.append(spikes)
        
        # Stack spikes over time
        spike_tensor = torch.stack(spike_outputs, dim=1)  # [B, T, C, D, H, W]
        
        # Compute spike rates for skip connection
        spike_rates = spike_tensor.mean(dim=1)  # [B, C, D, H, W]
        
        return spike_tensor, spike_rates


class SpikingDecoderBlock(nn.Module):
    """
    Decoder block for Spiking U-Net
    
    Contains:
    - Spiking TransposeConv3d for upsampling
    - Skip connection merging
    - Optional attention mechanism
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        tau_u: float = 0.9,
        time_window: int = 10,
        skip_merge: str = "concatenate",  # "concatenate", "average", "attention"
        attention: bool = False
    ):
        super().__init__()
        
        self.time_window = time_window
        self.skip_merge = skip_merge
        self.attention = attention
        
        # Determine input channels based on skip merge strategy
        if skip_merge == "concatenate":
            total_in_channels = in_channels + skip_channels
        else:
            total_in_channels = in_channels
        
        # Upsampling path
        self.upconv = SpikingTransposeConv3d(
            total_in_channels, out_channels, kernel_size,
            stride=stride, tau_u=tau_u
        )
        
        # Skip connection processing
        if skip_merge == "attention" and attention:
            self.attention_gate = AttentionGate(in_channels, skip_channels, out_channels)
        
        # Feature refinement
        self.refine = SpikingConv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, tau_u=tau_u
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder block
        
        Args:
            x: Input spike tensor from previous layer
            skip: Skip connection from encoder
            time_steps: Number of time steps to process
            
        Returns:
            spikes: Output spike tensor
        """
        if time_steps is None:
            time_steps = self.time_window
        
        # Reset neurons
        self.upconv.reset_neurons()
        self.refine.reset_neurons()
        
        # Merge with skip connection
        if self.skip_merge == "concatenate":
            # Convert spike tensor to rates for concatenation
            x_rates = x.mean(dim=1) if x.dim() > 5 else x
            merged = torch.cat([x_rates, skip], dim=1)
        elif self.skip_merge == "average":
            x_rates = x.mean(dim=1) if x.dim() > 5 else x
            merged = (x_rates + skip) / 2
        elif self.skip_merge == "attention" and self.attention:
            x_rates = x.mean(dim=1) if x.dim() > 5 else x
            skip = self.attention_gate(x_rates, skip)
            merged = torch.cat([x_rates, skip], dim=1)
        else:
            merged = x.mean(dim=1) if x.dim() > 5 else x
        
        # Process over time
        spike_outputs = []
        for t in range(time_steps):
            # Upsample
            spikes = self.upconv(merged)
            
            # Refine features
            spikes = self.refine(spikes)
            
            spike_outputs.append(spikes)
        
        # Stack spikes over time
        spike_tensor = torch.stack(spike_outputs, dim=1)  # [B, T, C, D, H, W]
        
        return spike_tensor


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections
    
    Helps focus on relevant features from encoder
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to skip connection
        
        Args:
            g: Gating signal from decoder path
            x: Skip connection from encoder
            
        Returns:
            Attended skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class OutputProjection(nn.Module):
    """
    Output projection layer for displacement field
    
    Converts spike rates to continuous displacement values
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,  # 3D displacement field
        time_window: int = 5,
        scale_factor: float = 1.0
    ):
        super().__init__()
        
        self.time_window = time_window
        self.scale_factor = scale_factor
        
        # Final convolution
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=True
        )
        
        # Smooth output
        self.smooth = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False, groups=out_channels
        )
        
        # Initialize smoothing kernel
        with torch.no_grad():
            kernel = torch.ones(out_channels, 1, 3, 3, 3) / 27.0
            self.smooth.weight.data = kernel
    
    def forward(self, spike_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert spikes to displacement field
        
        Args:
            spike_tensor: Input spikes [B, T, C, D, H, W]
            
        Returns:
            displacement: Continuous displacement field [B, 3, D, H, W]
        """
        # Compute spike rates
        spike_rates = spike_tensor.mean(dim=1)  # [B, C, D, H, W]
        
        # Project to displacement channels
        displacement = self.conv(spike_rates)
        
        # Smooth the output
        displacement = self.smooth(displacement)
        
        # Scale output
        displacement = displacement * self.scale_factor
        
        return displacement 