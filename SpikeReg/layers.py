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
    
    def reset_neurons(self, batch_size=1, *spatial_dims):
        """Reset neuron states
        
        Note: If spatial_dims are not provided, uses placeholder shape (1,1,1,1,1).
        The neuron will resize on first forward call if membrane is None.
        For shape-aware reset, pass actual dimensions: reset_neurons(batch_size, C, D, H, W)
        """
        if spatial_dims:
            device = next(self.conv.parameters()).device if list(self.conv.parameters()) else torch.device('cpu')
            self.neurons.reset_state(batch_size, *spatial_dims, device=device)
        else:
            device = next(self.conv.parameters()).device if list(self.conv.parameters()) else torch.device('cpu')
            self.neurons.reset_state(1, 1, 1, 1, 1, device=device)


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
    
    def reset_neurons(self, batch_size=1, *spatial_dims):
        """Reset neuron states
        
        Note: If spatial_dims are not provided, uses placeholder shape (1,1,1,1,1).
        The neuron will resize on first forward call if membrane is None.
        For shape-aware reset, pass actual dimensions: reset_neurons(batch_size, C, D, H, W)
        """
        if spatial_dims:
            device = next(self.conv.parameters()).device if list(self.conv.parameters()) else torch.device('cpu')
            self.neurons.reset_state(batch_size, *spatial_dims, device=device)
        else:
            device = next(self.conv.parameters()).device if list(self.conv.parameters()) else torch.device('cpu')
            self.neurons.reset_state(1, 1, 1, 1, 1, device=device)

class SpikingEncoderBlock(nn.Module):
    """
    Encoder block for Spiking U-Net
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

        self.conv = SpikingConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2,
            tau_u=tau_u, lateral_inhibition=lateral_inhibition
        )

        if residual and stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        elif residual:
            self.shortcut = SpikingConv3d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, tau_u=tau_u
            )
        else:
            self.shortcut = None

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if time_steps is None:
            time_steps = self.time_window

        B, C, D, H, W = x.shape

        conv = self.conv.conv
        kD, kH, kW = conv.kernel_size
        sD, sH, sW = conv.stride
        pD, pH, pW = conv.padding
        dD, dH, dW = conv.dilation

        out_D = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
        out_H = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        out_W = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out_C = conv.out_channels

        self.conv.reset_neurons(B, out_C, out_D, out_H, out_W)

        if hasattr(self.shortcut, "reset_neurons"):
            sc_conv = self.shortcut.conv
            kD2, kH2, kW2 = sc_conv.kernel_size
            sD2, sH2, sW2 = sc_conv.stride
            pD2, pH2, pW2 = sc_conv.padding
            dD2, dH2, dW2 = sc_conv.dilation

            out_D2 = (D + 2 * pD2 - dD2 * (kD2 - 1) - 1) // sD2 + 1
            out_H2 = (H + 2 * pH2 - dH2 * (kH2 - 1) - 1) // sH2 + 1
            out_W2 = (W + 2 * pW2 - dW2 * (kW2 - 1) - 1) // sW2 + 1
            out_C2 = sc_conv.out_channels

            self.shortcut.reset_neurons(B, out_C2, out_D2, out_H2, out_W2)

        spike_outputs = []
        for t in range(time_steps):
            spikes = self.conv(x)
            if self.shortcut is not None:
                spikes = spikes + self.shortcut(x)
            spike_outputs.append(spikes)

        spike_tensor = torch.stack(spike_outputs, dim=1)
        spike_rates = spike_tensor.mean(dim=1)

        return spike_tensor, spike_rates
class SpikingDecoderBlock(nn.Module):
    """
    Decoder block for Spiking U-Net
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
        skip_merge: str = "concatenate",
        attention: bool = False
    ):
        super().__init__()

        self.time_window = time_window
        self.skip_merge = skip_merge
        self.attention = attention
        self.skip_adapter = None
        self._timestep_warned = False

        self.upconv = SpikingTransposeConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, tau_u=tau_u
        )

        if skip_merge == "attention" and attention:
            self.attention_gate = AttentionGate(out_channels, skip_channels, out_channels)

        if skip_merge in ["concatenate", "attention"]:
            refine_in_channels = out_channels + skip_channels
        else:
            refine_in_channels = out_channels

        if skip_merge == "average" and skip_channels != out_channels:
            self.skip_adapter = nn.Sequential(
                nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.refine = SpikingConv3d(
            refine_in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, tau_u=tau_u
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_steps: Optional[int] = None
    ) -> torch.Tensor:

        if time_steps is None:
            time_steps = self.time_window

        spike_outputs = []

        seq_len = x.shape[1] if x.dim() > 5 else 1
        first_idx = 0
        first_current = x[:, first_idx] if x.dim() > 5 else x

        B, C, D, H, W = first_current.shape

        upc = self.upconv.conv
        kD, kH, kW = upc.kernel_size
        sD, sH, sW = upc.stride
        pD, pH, pW = upc.padding
        dD, dH, dW = upc.dilation
        opD, opH, opW = upc.output_padding

        out_D = (D - 1) * sD - 2 * pD + dD * (kD - 1) + opD + 1
        out_H = (H - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
        out_W = (W - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
        out_C = upc.out_channels

        self.upconv.reset_neurons(B, out_C, out_D, out_H, out_W)
        first_up = self.upconv(first_current)
        target_spatial_size = first_up.shape[2:]
        self.upconv.reset_neurons(B, out_C, out_D, out_H, out_W)

        refine_out_C = self.refine.conv.out_channels
        td, th, tw = target_spatial_size
        self.refine.reset_neurons(B, refine_out_C, td, th, tw)

        if skip.shape[2:] != target_spatial_size:
            skip = F.interpolate(skip, size=target_spatial_size, mode="trilinear", align_corners=False)

        for t in range(time_steps):
            idx = t if t < seq_len else seq_len - 1
            if t >= seq_len and not self._timestep_warned:
                print(f"[SpikingDecoderBlock] Requested timestep {t} exceeds input sequence length {seq_len}. Reusing last frame.")
                self._timestep_warned = True

            current = x[:, idx] if x.dim() > 5 else x

            up = self.upconv(current)

            if up.shape[2:] != skip.shape[2:]:
                up = F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=False)

            if self.skip_merge == "concatenate":
                merged = torch.cat([up, skip], dim=1)
            elif self.skip_merge == "average":
                skip_aligned = self.skip_adapter(skip) if self.skip_adapter is not None else skip
                merged = (up + skip_aligned) / 2
            elif self.skip_merge == "attention" and self.attention:
                gated = self.attention_gate(up, skip)
                merged = torch.cat([up, gated], dim=1)
            else:
                merged = up

            if merged.shape[1] != self.refine.conv.in_channels:
                raise RuntimeError(
                    f"Channel mismatch in SpikingDecoderBlock: merged features have {merged.shape[1]} channels, "
                    f"but refine layer expects {self.refine.conv.in_channels} channels."
                )

            spikes = self.refine(merged)
            spike_outputs.append(spikes)

        spike_tensor = torch.stack(spike_outputs, dim=1)

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
        
        # Initialize smoothing kernel as fixed averaging kernel
        with torch.no_grad():
            kernel = torch.ones(out_channels, 1, 3, 3, 3) / 27.0
            self.smooth.weight.data = kernel
            self.smooth.weight.requires_grad_(False)
    
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