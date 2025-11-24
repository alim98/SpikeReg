"""
Spiking U-Net model for medical image registration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .layers import (
    SpikingEncoderBlock, SpikingDecoderBlock, 
    OutputProjection, SpikingConv3d
)
from .neurons import LIFNeuron


class SpikeRegUNet(nn.Module):
    """
    Spiking U-Net for deformable image registration
    
    Architecture:
    - 4-level encoder with progressive downsampling
    - Symmetric decoder with skip connections
    - Iterative residual displacement prediction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.patch_size = config.get('patch_size', 32)
        self.in_channels = config.get('in_channels', 2)  # Fixed + Moving
        self.base_channels = config.get('base_channels', 16)
        self.encoder_channels = config.get('encoder_channels', [16, 32, 64, 128])
        self.decoder_channels = config.get('decoder_channels', [64, 32, 16, 16])
        
        # Time windows for each level (decreasing)
        self.encoder_time_windows = config.get('encoder_time_windows', [10, 8, 6, 4])
        self.decoder_time_windows = config.get('decoder_time_windows', [4, 6, 8, 10])
        
        # Neuron parameters for each level
        self.encoder_tau_u = config.get('encoder_tau_u', [0.9, 0.8, 0.8, 0.7])
        self.decoder_tau_u = config.get('decoder_tau_u', [0.7, 0.8, 0.8, 0.9])
        
        # Skip connection merge strategies
        self.skip_merge = config.get('skip_merge', ['concatenate', 'average', 'concatenate', 'none'])
        
        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        in_ch = self.in_channels
        for i, out_ch in enumerate(self.encoder_channels):
            block = SpikingEncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                tau_u=self.encoder_tau_u[i],
                time_window=self.encoder_time_windows[i],
                lateral_inhibition=(i > 0),  # No inhibition in first layer
                residual=(i > 1)  # Residual connections in deeper layers
            )
            self.encoder_blocks.append(block)
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = SpikingConv3d(
            self.encoder_channels[-1],
            self.encoder_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            tau_u=0.7,
            lateral_inhibition=True
        )
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        prev_out_ch = self.encoder_channels[-1]
        for i in range(len(self.decoder_channels)):
            in_ch = prev_out_ch
            
            if i + 2 <= len(self.encoder_channels):
                skip_ch = self.encoder_channels[-(i+2)]
            else:
                skip_ch = self.encoder_channels[0]
            
            out_ch = self.decoder_channels[i]
            
            block = SpikingDecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                skip_channels=skip_ch,
                kernel_size=2,
                stride=2,
                tau_u=self.decoder_tau_u[i],
                time_window=self.decoder_time_windows[i],
                skip_merge=self.skip_merge[i],
                attention=(i < 2)
            )
            self.decoder_blocks.append(block)
            
            prev_out_ch = out_ch
        
        # Output projection
        self.output_projection = OutputProjection(
            in_channels=self.decoder_channels[-1],
            out_channels=3,  # 3D displacement field
            time_window=5,
            scale_factor=config.get('displacement_scale', 1.0)
        )
        
        # Input spike encoding parameters
        self.register_buffer('spike_encoding_window', torch.tensor(config.get('input_time_window', 10)))
        self._spike_rate_checked = False
    
    def encode_to_spikes(self, x: torch.Tensor, time_window: int = 10) -> torch.Tensor:
        """
        Encode continuous input to spike trains using rate coding
        
        Args:
            x: Continuous input tensor [B, C, D, H, W]
            time_window: Number of time steps for encoding
            
        Returns:
            spikes: Binary spike tensor [B, T, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        device = x.device
        
        # Ensure values are in [0, 1]
        x = torch.clamp(x, 0, 1)
        
        # Sanity check: verify input spike rate (only on first batch)
        if not self._spike_rate_checked:
            mean_rate = x.mean().item()
            print(f"Input spike rate (mean): {mean_rate:.4f}")
            if mean_rate < 0.01:
                print(f"Warning: Input spike rate is very low ({mean_rate:.4f}). Input may be under-stimulated, which could cause dead gradients.")
            self._spike_rate_checked = True
        
        # Generate Poisson spike trains
        spikes = []
        for t in range(time_window):
            # Random sampling for each time step
            random_vals = torch.rand_like(x)
            spike = (random_vals < x).float()
            spikes.append(spike)
        
        # Stack along time dimension
        spike_tensor = torch.stack(spikes, dim=1)  # [B, T, C, D, H, W]
        
        return spike_tensor
    
    def forward(
        self, 
        fixed: torch.Tensor, 
        moving: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SpikeReg U-Net
        
        Args:
            fixed: Fixed image patch [B, 1, D, H, W]
            moving: Moving image patch [B, 1, D, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            output: Dictionary containing:
                - displacement: Predicted displacement field [B, 3, D, H, W]
                - spike_counts: Spike count statistics per layer
                - features: Intermediate features (if requested)
        """
        # Concatenate fixed and moving images
        x = torch.cat([fixed, moving], dim=1)  # [B, 2, D, H, W]
        
        # Debug removed - was causing log spam
        # x_min = x.min().item()
        # x_max = x.max().item()
        # print(f"[SpikeRegUNet DEBUG] Input range: min={x_min}, max={x_max}")

        # Encode to spikes
        spike_input = self.encode_to_spikes(x, self.spike_encoding_window.item())

        # check for NaN
        if torch.isnan(spike_input).any():
            print("Warning: NaN detected in spike input tensor!")

        # For first encoder, average spikes over time as input
        x_rates = spike_input.mean(dim=1)
        
        # Encoder path
        encoder_features = []
        spike_counts = {}
        spike_counts_number = {}
        
        for i, encoder in enumerate(self.encoder_blocks):
            spike_tensor, skip_features = encoder(x_rates, self.encoder_time_windows[i])
            encoder_features.append(skip_features)
            x_rates = spike_tensor.mean(dim=1)  # Convert to rates for next layer
            
            # Check for NaN in spike tensor
            if torch.isnan(spike_tensor).any():
                print(f"Warning: NaN detected in encoder_{i} spike tensor!")

            # Record spike statistics
            spike_counts[f'encoder_{i}'] = spike_tensor.mean()
            spike_counts_number[f'encoder_{i}'] = spike_tensor.sum().cpu().detach().numpy()
        # Bottleneck
        self.bottleneck.reset_neurons()
        bottleneck_spikes = []
        for t in range(self.encoder_time_windows[-1]):
            spikes = self.bottleneck(x_rates)
            bottleneck_spikes.append(spikes)
            # Check for NaN in bottleneck spikes
            if torch.isnan(spikes).any():
                print("Warning: NaN detected in bottleneck spikes!")
        
        x = torch.stack(bottleneck_spikes, dim=1)
        spike_counts['bottleneck'] = x.mean()
        spike_counts_number['bottleneck'] = x.sum().cpu().detach().numpy()
        
        # Decoder path
        decoder_features = []
        for i, decoder in enumerate(self.decoder_blocks):
            skip_idx = -(i+2)
            if skip_idx >= -len(encoder_features):
                skip = encoder_features[skip_idx]
            else:
                skip = encoder_features[0]  # Use first encoder features for last decoder
            
            x = decoder(x, skip, self.decoder_time_windows[i])
            # Check for NaN in decoder output
            if torch.isnan(x).any():
                print(f"Warning: NaN detected in decoder_{i} output!")
            decoder_features.append(x.mean(dim=1))
            
            # Record spike statistics
            spike_counts[f'decoder_{i}'] = x.mean()
            spike_counts_number[f'decoder_{i}'] = x.sum().cpu().detach().numpy()
        
        # Output projection
        displacement = self.output_projection(x)
        # Check for NaN in displacement
        if torch.isnan(displacement).any():
            print("Warning: NaN detected in displacement output!")
        
        # Prepare output
        output = {
            'displacement': displacement,
            'spike_counts': spike_counts,
            'spike_counts_number': spike_counts_number
        }
        
        if return_features:
            output['encoder_features'] = encoder_features
            output['decoder_features'] = decoder_features
        
        return output
    
    def reset_all_neurons(self):
        """Reset all neuron states in the network"""
        for block in self.encoder_blocks:
            block.conv.reset_neurons()
            if hasattr(block, 'shortcut') and block.shortcut is not None:
                block.shortcut.reset_neurons()
        
        self.bottleneck.reset_neurons()
        
        for block in self.decoder_blocks:
            block.upconv.reset_neurons()
            block.refine.reset_neurons()


class PretrainedUNet(nn.Module):
    """
    Standard U-Net for pretraining before SNN conversion
    
    Used to learn good spatial features with standard backprop
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.in_channels = config.get('in_channels', 2)
        self.encoder_channels = config.get('encoder_channels', [16, 32, 64, 128])
        self.decoder_channels = config.get('decoder_channels', [64, 32, 16, 16])
        self.skip_merge = config.get('skip_merge', ['concatenate', 'average', 'concatenate', 'none'])
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch in self.encoder_channels:
            encoder = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.encoders.append(encoder)
            in_ch = out_ch
        
        # Decoder - match SpikeRegUNet skip_merge logic exactly
        self.decoders = nn.ModuleList()
        self.decoder_refines = nn.ModuleList()
        self.decoder_skip_adapters = nn.ModuleList()
        N = len(self.encoder_channels)
        prev_ch = self.encoder_channels[-1]
        for i, out_ch in enumerate(self.decoder_channels):
            upconv = nn.Sequential(
                nn.ConvTranspose3d(prev_ch, out_ch, 2, stride=2),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.decoders.append(upconv)
            
            # Get skip channels for this decoder level
            if i + 2 <= len(self.encoder_channels):
                skip_ch = self.encoder_channels[-(i+2)]
            else:
                skip_ch = self.encoder_channels[0]
            
            # Determine refine input channels based on skip_merge strategy
            merge_strategy = self.skip_merge[i] if i < len(self.skip_merge) else 'none'
            if merge_strategy in ["concatenate", "attention"]:
                refine_in_ch = out_ch + skip_ch
            else:
                refine_in_ch = out_ch
            
            # Create skip adapter for average merge if channels don't match
            skip_adapter = None
            if merge_strategy == "average" and skip_ch != out_ch:
                skip_adapter = nn.Sequential(
                    nn.Conv3d(skip_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_ch)
                )
            self.decoder_skip_adapters.append(skip_adapter)
            
            refine = nn.Sequential(
                nn.Conv3d(refine_in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.decoder_refines.append(refine)
            prev_ch = out_ch
        
        # Output
        self.output = nn.Conv3d(self.decoder_channels[-1], 3, 1)
    
    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """Forward pass through pretrained U-Net"""
        x = torch.cat([fixed, moving], dim=1)
        
        # Encoder
        skip_features = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_features.append(x)
        
        # ------------------------------------------------------------------
        # Debug printing: run only on the first forward call to help diagnose
        # channel mismatches between encoder outputs and decoder expectations.
        # ------------------------------------------------------------------
        # Debug code removed - was causing massive log spam with DataParallel
        
        # Decoder - match SpikeRegUNet skip_merge logic exactly
        for i, (decoder, refine) in enumerate(zip(self.decoders, self.decoder_refines)):
            x = decoder(x)
            
            # Get skip feature for this decoder level
            skip_idx = -(i+2)
            if skip_idx >= -len(skip_features):
                skip = skip_features[skip_idx]
            else:
                skip = skip_features[0]
            
            # Apply skip_merge strategy
            merge_strategy = self.skip_merge[i] if i < len(self.skip_merge) else 'none'
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="trilinear", align_corners=False)
            
            if merge_strategy == "concatenate":
                x = torch.cat([x, skip], dim=1)
            elif merge_strategy == "average":
                skip_adapter = self.decoder_skip_adapters[i]
                if skip_adapter is not None:
                    skip = skip_adapter(skip)
                x = (x + skip) / 2.0
            elif merge_strategy == "attention":
                x = torch.cat([x, skip], dim=1)
            else:
                pass
            
            x = refine(x)
        
        # Output
        displacement = self.output(x)
        #check for NaN in output
        if torch.isnan(displacement).any():
            print("Warning: NaN detected in pretrainedUNet output displacement tensor!")
        return displacement


def convert_pretrained_to_spiking(
    pretrained_model: PretrainedUNet,
    config: Dict,
    threshold_percentile: float = 99.0
) -> SpikeRegUNet:
    """
    Convert a pretrained U-Net to SpikeReg U-Net
    
    Args:
        pretrained_model: Trained standard U-Net
        config: SpikeReg configuration
        threshold_percentile: Percentile for threshold normalization
        
    Returns:
        spiking_model: Converted SpikeReg model
    """
    spiking_model = SpikeRegUNet(config)
    
    # Transfer encoder weights
    for i, (pretrained_enc, spiking_enc) in enumerate(
        zip(pretrained_model.encoders, spiking_model.encoder_blocks)
    ):
        # Extract conv and bn from pretrained sequential
        conv_weight = pretrained_enc[0].weight.data
        conv_bias = pretrained_enc[0].bias.data if pretrained_enc[0].bias is not None else None
        bn_weight = pretrained_enc[1].weight.data
        bn_bias = pretrained_enc[1].bias.data
        bn_mean = pretrained_enc[1].running_mean
        bn_var = pretrained_enc[1].running_var
        
        # Transfer to spiking layer
        spiking_enc.conv.conv.weight.data = conv_weight.clone()
        if conv_bias is not None:
            spiking_enc.conv.conv.bias.data = conv_bias.clone()
        
        spiking_enc.conv.bn.weight.data = bn_weight.clone()
        spiking_enc.conv.bn.bias.data = bn_bias.clone()
        spiking_enc.conv.bn.running_mean = bn_mean.clone()
        spiking_enc.conv.bn.running_var = bn_var.clone()
    
    # Transfer decoder weights
    for i, (pretrained_dec, pretrained_refine, spiking_dec) in enumerate(
        zip(pretrained_model.decoders, pretrained_model.decoder_refines, spiking_model.decoder_blocks)
    ):
        # Transfer upconv weights (ConvTranspose3d)
        upconv_weight = pretrained_dec[0].weight.data
        upconv_bias = pretrained_dec[0].bias.data if pretrained_dec[0].bias is not None else None
        upconv_bn_weight = pretrained_dec[1].weight.data
        upconv_bn_bias = pretrained_dec[1].bias.data
        upconv_bn_mean = pretrained_dec[1].running_mean
        upconv_bn_var = pretrained_dec[1].running_var
        
        # With unified architectures, channels should match - warn if they don't
        in_ch_required = spiking_dec.upconv.conv.in_channels
        if upconv_weight.shape[0] != in_ch_required:
            raise RuntimeError(
                f"[convert_pretrained_to_spiking] Architecture mismatch in decoder {i} upconv: "
                f"pretrained has {upconv_weight.shape[0]} input channels, "
                f"spiking model expects {in_ch_required}. "
                f"Architectures must match exactly - check encoder_channels and decoder_channels config."
            )
        
        # Transfer upconv to spiking layer
        spiking_dec.upconv.conv.weight.data = upconv_weight.clone()
        if upconv_bias is not None:
            spiking_dec.upconv.conv.bias.data = upconv_bias.clone()
        
        spiking_dec.upconv.bn.weight.data = upconv_bn_weight.clone()
        spiking_dec.upconv.bn.bias.data = upconv_bn_bias.clone()
        spiking_dec.upconv.bn.running_mean = upconv_bn_mean.clone()
        spiking_dec.upconv.bn.running_var = upconv_bn_var.clone()
        
        # Transfer skip adapter weights if they exist (for average merge with channel mismatch)
        pretrained_skip_adapter = pretrained_model.decoder_skip_adapters[i]
        if pretrained_skip_adapter is not None and spiking_dec.skip_adapter is not None:
            adapter_weight = pretrained_skip_adapter[0].weight.data
            adapter_bias = pretrained_skip_adapter[0].bias if pretrained_skip_adapter[0].bias is not None else None
            adapter_bn_weight = pretrained_skip_adapter[1].weight.data
            adapter_bn_bias = pretrained_skip_adapter[1].bias.data
            adapter_bn_mean = pretrained_skip_adapter[1].running_mean
            adapter_bn_var = pretrained_skip_adapter[1].running_var
            
            spiking_dec.skip_adapter[0].weight.data = adapter_weight.clone()
            if adapter_bias is not None:
                spiking_dec.skip_adapter[0].bias.data = adapter_bias.clone()
            
            spiking_dec.skip_adapter[1].weight.data = adapter_bn_weight.clone()
            spiking_dec.skip_adapter[1].bias.data = adapter_bn_bias.clone()
            spiking_dec.skip_adapter[1].running_mean = adapter_bn_mean.clone()
            spiking_dec.skip_adapter[1].running_var = adapter_bn_var.clone()
        
        # Transfer refine weights (Conv3d)
        refine_weight = pretrained_refine[0].weight.data
        refine_bias = pretrained_refine[0].bias.data if pretrained_refine[0].bias is not None else None
        refine_bn_weight = pretrained_refine[1].weight.data
        refine_bn_bias = pretrained_refine[1].bias.data
        refine_bn_mean = pretrained_refine[1].running_mean
        refine_bn_var = pretrained_refine[1].running_var
        
        # With unified architectures, refine input channels should match - warn if they don't
        refine_in_ch_required = spiking_dec.refine.conv.in_channels
        if refine_weight.shape[1] != refine_in_ch_required:
            raise RuntimeError(
                f"[convert_pretrained_to_spiking] Architecture mismatch in decoder {i} refine: "
                f"pretrained has {refine_weight.shape[1]} input channels, "
                f"spiking model expects {refine_in_ch_required}. "
                f"Architectures must match exactly - check skip_merge config."
            )
        
        # Transfer refine to spiking layer
        spiking_dec.refine.conv.weight.data = refine_weight.clone()
        if refine_bias is not None:
            spiking_dec.refine.conv.bias.data = refine_bias.clone()
        
        spiking_dec.refine.bn.weight.data = refine_bn_weight.clone()
        spiking_dec.refine.bn.bias.data = refine_bn_bias.clone()
        spiking_dec.refine.bn.running_mean = refine_bn_mean.clone()
        spiking_dec.refine.bn.running_var = refine_bn_var.clone()
    
    # Transfer output weights
    spiking_model.output_projection.conv.weight.data = pretrained_model.output.weight.data.clone()
    if pretrained_model.output.bias is not None:
        spiking_model.output_projection.conv.bias.data = pretrained_model.output.bias.data.clone()
    
    # Normalize thresholds based on activation statistics
    # (This would require running calibration data through the network)
    
    return spiking_model 