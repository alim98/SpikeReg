# Training SpikeReg on OASIS Dataset

This guide explains how to train SpikeReg on your OASIS brain MRI dataset.

## Dataset Structure

Your OASIS dataset is located at: `/Users/ali/Documents/codes/SpikeReg/SpikeReg/data/OASIS`

The dataset contains:
- 414 training brain MRI volumes
- Brain segmentation labels
- Brain masks
- Volume dimensions: 160 x 224 x 192

## Quick Start

### 1. Test Dataset Loading

First, verify that the dataset loads correctly:

```bash
cd /Users/ali/Documents/codes/SpikeReg
python examples/test_oasis_loading.py
```

This will:
- Load sample patches from the dataset
- Create visualization images
- Verify data loaders work correctly

### 2. Start Training

To train SpikeReg on OASIS with default settings:

```bash
python examples/train_oasis.py --config configs/oasis_config.yaml --device cpu
```

For GPU training (if available):

```bash
python examples/train_oasis.py --config configs/oasis_config.yaml --device cuda
```

### 3. Training Options

Common training options:

```bash
# Adjust batch size based on available memory
python examples/train_oasis.py --batch-size 2

# Change patch size (default is 64)
python examples/train_oasis.py --patch-size 48 --patch-stride 24

# Adjust number of training epochs
python examples/train_oasis.py --pretrain-epochs 20 --finetune-epochs 30

# Skip pretraining (if you have a pretrained model)
python examples/train_oasis.py --skip-pretrain --resume checkpoints/oasis/pretrained_model.pth

# Resume from checkpoint
python examples/train_oasis.py --resume checkpoints/oasis/checkpoint_epoch_10.pth
```

## Training Phases

SpikeReg uses a three-phase training approach:

1. **Phase 1: Pretraining (30 epochs)**
   - Trains a standard U-Net for registration
   - Faster convergence without spike constraints

2. **Phase 2: ANN-to-SNN Conversion**
   - Converts the pretrained U-Net to spiking version
   - Calibrates spike thresholds

3. **Phase 3: Fine-tuning (50 epochs)**
   - Fine-tunes the spiking network
   - Optimizes for both registration accuracy and energy efficiency

## Configuration

The OASIS-specific configuration (`configs/oasis_config.yaml`) includes:

- **Patch size**: 64³ (good coverage for brain structures)
- **Similarity metric**: NCC (Normalized Cross-Correlation)
- **Regularization**: Bending energy (smooth deformations)
- **Augmentation**: Light augmentation suitable for brain MRI
- **No flipping**: Brain asymmetry preserved

## Monitoring Training

During training, you can monitor progress with TensorBoard:

```bash
tensorboard --logdir logs/oasis
```

This shows:
- Loss curves
- Registration accuracy metrics
- Spike activity statistics
- Sample registration results

## Output

After training, you'll find:
- `checkpoints/oasis/pretrained_model.pth` - Pretrained U-Net
- `checkpoints/oasis/converted_model.pth` - Converted SNN
- `checkpoints/oasis/final_model.pth` - Fine-tuned SNN
- `logs/oasis/` - TensorBoard logs

## Memory Requirements

Approximate memory usage:
- Batch size 4, patch size 64: ~4-6 GB GPU memory
- Batch size 2, patch size 64: ~2-3 GB GPU memory
- CPU training: ~8-16 GB RAM recommended

## Tips for Best Results

1. **Start with smaller patches** if you encounter memory issues
2. **Use fixed pairs** for reproducible results
3. **Monitor NCC scores** - should improve to >0.9 for good registration
4. **Check Jacobian determinants** - should stay positive (no folding)
5. **Adjust regularization** if deformations are too smooth/irregular

## Inference

After training, use the model for registration:

```python
from spikereg import SpikeRegInference

# Load model
model = SpikeRegInference('checkpoints/oasis/final_model.pth', device='cuda')

# Register two brain volumes
displacement = model.register(fixed_volume, moving_volume)
```

## Troubleshooting

If you encounter issues:

1. **Out of memory**: Reduce batch size or patch size
2. **Slow training**: Reduce patches_per_pair or use fewer workers
3. **Poor convergence**: Adjust learning rate or regularization weights
4. **Loading errors**: Check nibabel is installed (`pip install nibabel`)

## Next Steps

After training:
1. Evaluate on test set
2. Export for neuromorphic deployment
3. Compare with standard registration methods
4. Fine-tune hyperparameters for your specific use case 