#aim ```aim up --repo /u/almik/SpikeReg5/SpikeReg/checkpoints/spikereg --port 6006```
# SpikeReg: Energy-Efficient Deformable Medical Image Registration using Spiking Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

SpikeReg is a research prototype for deformable medical image registration with Spiking Neural Networks (SNNs). The project is set up to train an ANN registration baseline, convert it to a spiking model, fine-tune with surrogate gradients, and evaluate accuracy against operation-count based energy proxies.

## 🚀 Key Features

- **Energy Analysis**: Reports MAC/AC counts and analytical energy proxies from measured spike rates
- **Full-Volume L2R Evaluation**: Includes label-wise Dice, HD95, NCC, and Jacobian folding metrics
- **Neuromorphic-Oriented**: Designed around sparse spiking activity and ANN-to-SNN conversion
- **Progressive Refinement**: Iterative residual policy for coarse-to-fine registration
- **Patch-based Processing**: Memory-efficient tiling for large medical volumes

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (for training)
- See `requirements.txt` for full dependencies

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spikereg.git
cd spikereg

# Install in development mode
pip install -e .

# Or install with neuromorphic support
pip install -e ".[neuromorphic]"
```

## 🎯 Quick Start

### Training a SpikeReg Model

```python
from spikereg import SpikeRegTrainer
from spikereg.training import load_config

# Load configuration
config = load_config("configs/spikereg_l2r_config.yaml")

# Create trainer
trainer = SpikeRegTrainer(config)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Inference

```python
from spikereg import SpikeRegInference
import nibabel as nib

# Load trained model
inference = SpikeRegInference("checkpoints/best_model.pth")

# Load medical volumes
fixed = nib.load("path/to/fixed.nii.gz").get_fdata()
moving = nib.load("path/to/moving.nii.gz").get_fdata()

# Perform registration
deformation_field = inference.register(fixed, moving)
warped = inference.apply_deformation(moving, deformation_field)
```

## 🏗️ Architecture Overview

SpikeReg implements a Spiking U-Net architecture with:
- **Encoder**: 4-level hierarchical feature extraction with decreasing time windows
- **Decoder**: Symmetric expansion with skip connections
- **LIF Neurons**: Leaky Integrate-and-Fire dynamics with refractory periods
- **Iterative Policy**: Progressive refinement through residual deformations

## 📊 Benchmarks

No publication-ready benchmark numbers are claimed yet. Use `evaluate_oasis.py` to evaluate a checkpoint on the L2R validation pairs:

```bash
python evaluate_oasis.py \
  --checkpoint checkpoints/spikereg/final_model.pth \
  --config configs/spikereg_l2r_config.yaml \
  --dataset-format pkl \
  --data-root /path/to/OASIS_L2R_2021_task03/Test \
  --output-json results/l2r_eval.json
```

Report at minimum mean +/- std label-wise Dice, HD95, NCC, and the fraction of voxels with non-positive Jacobian determinant. Energy numbers should be reported as analytical proxies unless measured on actual neuromorphic hardware.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=spikereg tests/
```

## 📖 Documentation

For detailed documentation, see:
- [Architecture Details](docs/architecture.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)
- [Neuromorphic Deployment](docs/deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📝 Citation

If you use SpikeReg in your research, please cite:

```bibtex
@article{spikereg2024,
  title={SpikeReg: Spiking Neural Networks for Energy-Efficient Medical Image Registration},
  author={Your Name et al.},
  journal={Medical Image Analysis},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Intel Labs for neuromorphic hardware support
- Medical imaging datasets from [source]
- Inspired by biological vision systems 
