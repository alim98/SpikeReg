"""
Spiking neuron models with surrogate gradient support for SpikeReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable


class SurrogateFunction:
    """Base class for surrogate gradient functions"""
    
    @staticmethod
    def forward(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        """Heaviside step function for forward pass"""
        return (x >= 0).float()
    
    @staticmethod
    def backward(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        """Surrogate gradient for backward pass"""
        raise NotImplementedError


class FastSigmoid(SurrogateFunction):
    """Fast sigmoid surrogate gradient"""
    
    @staticmethod
    def backward(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        return alpha * torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))


class SuperSpike(SurrogateFunction):
    """SuperSpike surrogate gradient from Zenke & Ganguli (2018)"""
    
    @staticmethod
    def backward(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        return alpha / (1 + alpha * torch.abs(x)) ** 2


class SpikeFunctionSurrogate(torch.autograd.Function):
    """Custom autograd function for spike generation with surrogate gradients"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, surrogate: SurrogateFunction, alpha: float):
        ctx.save_for_backward(input)
        ctx.surrogate = surrogate
        ctx.alpha = alpha
        return surrogate.forward(input, alpha)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * ctx.surrogate.backward(input, ctx.alpha)
        return grad_input, None, None


def spike_function(x: torch.Tensor, surrogate: SurrogateFunction = FastSigmoid(), alpha: float = 10.0):
    """Apply spike function with surrogate gradient"""
    return SpikeFunctionSurrogate.apply(x, surrogate, alpha)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with surrogate gradient support
    
    Membrane dynamics:
    u[t+1] = tau_u * u[t] + sum(w_ij * s_j[t]) - v_th * s[t]
    s[t] = spike_function(u[t] - v_th)
    
    Args:
        tau_u: Membrane leak factor (0 < tau_u < 1)
        v_th: Firing threshold
        v_reset: Reset potential after spike
        surrogate: Surrogate gradient function
        alpha: Surrogate gradient sharpness
        refractory: Refractory period in timesteps
        learnable_params: Whether tau_u and v_th are learnable
    """
    
    def __init__(
        self,
        tau_u: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        surrogate: SurrogateFunction = FastSigmoid(),
        alpha: float = 10.0,
        refractory: int = 0,
        learnable_params: bool = False
    ):
        super().__init__()
        
        if learnable_params:
            self.tau_u = nn.Parameter(torch.tensor(tau_u))
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('tau_u', torch.tensor(tau_u))
            self.register_buffer('v_th', torch.tensor(v_th))
        
        self.register_buffer('v_reset', torch.tensor(v_reset))
        self.surrogate = surrogate
        self.alpha = alpha
        self.refractory = refractory
        
        # State variables
        self.membrane = None
        self.refractory_counter = None
        self.spike_history = []
    
    def reset_state(self, batch_size: int, *spatial_dims: int, device: torch.device = None):
        """Reset neuron state for new sequence"""
        shape = (batch_size, *spatial_dims)
        # Use provided device, or fall back to this neuron's buffer device (v_th)
        device = device if device is not None else self.v_th.device
        
        self.membrane = torch.zeros(shape, device=device)
        self.refractory_counter = torch.zeros(shape, device=device)
        self.spike_history = []
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LIF neuron
        
        Args:
            input_current: Input current at current timestep
            
        Returns:
            spikes: Binary spike tensor
        """
        if self.membrane is None:
            self.reset_state(input_current.shape[0], *input_current.shape[1:], device=input_current.device)
        
        # Apply refractory mask
        refractory_mask = (self.refractory_counter <= 0).float()
        
        # Update membrane potential
        self.membrane = self.tau_u * self.membrane + input_current * refractory_mask
        
        # Generate spikes
        spike = spike_function(self.membrane - self.v_th, self.surrogate, self.alpha)
        spike = spike * refractory_mask
        
        # Reset membrane and update refractory counter
        self.membrane = self.membrane * (1 - spike) + self.v_reset * spike
        self.refractory_counter = torch.maximum(
            self.refractory_counter - 1,
            spike * self.refractory
        )
        
        # Store spike history
        self.spike_history.append(spike)
        
        return spike
    
    def get_spike_rate(self, window: Optional[int] = None) -> torch.Tensor:
        """Compute spike rate over time window"""
        if not self.spike_history:
            return torch.zeros_like(self.membrane)
        
        if window is None:
            spikes = torch.stack(self.spike_history)
        else:
            spikes = torch.stack(self.spike_history[-window:])
        
        return spikes.mean(dim=0)
    
    def get_total_spikes(self) -> torch.Tensor:
        """Get total spike count"""
        if not self.spike_history:
            return torch.zeros_like(self.membrane)
        
        return torch.stack(self.spike_history).sum(dim=0)


class AdaptiveLIFNeuron(LIFNeuron):
    """
    Adaptive LIF neuron with threshold adaptation
    
    Implements homeostatic plasticity through adaptive threshold
    """
    
    def __init__(
        self,
        tau_u: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        tau_adapt: float = 0.99,
        beta_adapt: float = 0.1,
        **kwargs
    ):
        super().__init__(tau_u, v_th, v_reset, **kwargs)
        
        self.register_buffer('tau_adapt', torch.tensor(tau_adapt))
        self.register_buffer('beta_adapt', torch.tensor(beta_adapt))
        self.v_th_adapt = None
    
    def reset_state(self, batch_size: int, *spatial_dims: int, device: torch.device = None):
        """Reset neuron state including adaptive threshold"""
        super().reset_state(batch_size, *spatial_dims, device)
        shape = (batch_size, *spatial_dims)
        device = device or next(self.parameters()).device
        self.v_th_adapt = torch.ones(shape, device=device) * self.v_th
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive threshold"""
        if self.membrane is None:
            self.reset_state(input_current.shape[0], *input_current.shape[1:], device=input_current.device)
        
        # Apply refractory mask
        refractory_mask = (self.refractory_counter <= 0).float()
        
        # Update membrane potential
        self.membrane = self.tau_u * self.membrane + input_current * refractory_mask
        
        # Generate spikes with adaptive threshold
        spike = spike_function(self.membrane - self.v_th_adapt, self.surrogate, self.alpha)
        spike = spike * refractory_mask
        
        # Reset membrane and update refractory counter
        self.membrane = self.membrane * (1 - spike) + self.v_reset * spike
        self.refractory_counter = torch.maximum(
            self.refractory_counter - 1,
            spike * self.refractory
        )
        
        # Update adaptive threshold
        self.v_th_adapt = self.tau_adapt * self.v_th_adapt + self.beta_adapt * spike
        
        # Store spike history
        self.spike_history.append(spike)
        
        return spike


class LateralInhibition(nn.Module):
    """
    Lateral inhibition mechanism for sparse activity
    
    Implements k-winner-take-all competition
    """
    
    def __init__(self, k_fraction: float = 0.1, dim: int = 1):
        super().__init__()
        self.k_fraction = k_fraction
        self.dim = dim
    
    def forward(self, membrane: torch.Tensor) -> torch.Tensor:
        """Apply lateral inhibition to membrane potentials"""
        batch_size = membrane.shape[0]
        flat_membrane = membrane.view(batch_size, -1)
        
        # Compute k for k-winner-take-all
        k = max(1, int(self.k_fraction * flat_membrane.shape[1]))
        
        # Find top-k values
        topk_vals, topk_idx = torch.topk(flat_membrane, k, dim=1)
        
        # Create inhibition mask
        mask = torch.zeros_like(flat_membrane)
        mask.scatter_(1, topk_idx, 1)
        
        # Reshape to original dimensions
        mask = mask.view_as(membrane)
        
        # Apply inhibition
        return membrane * mask 