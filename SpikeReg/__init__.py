"""
SpikeReg: Energy-Efficient Deformable Medical Image Registration using Spiking Neural Networks
"""

from .models import SpikeRegUNet
from .registration import SpikeRegInference
# Make training imports optional (they require 'aim' which may not be installed)
try:
    from .training import SpikeRegTrainer
except ImportError:
    SpikeRegTrainer = None
from .neurons import LIFNeuron

__version__ = "0.1.0"
__all__ = ["SpikeRegUNet", "SpikeRegInference", "SpikeRegTrainer", "LIFNeuron"]

# -----------------------------------------------------------------------------
# Provide lowercase import alias so external scripts can `import spikereg`
# without depending on the exact capitalization of the package directory.
# -----------------------------------------------------------------------------

import sys as _sys

# Register this module under the lowercase name if not already present.
_sys.modules.setdefault('spikereg', _sys.modules[__name__]) 