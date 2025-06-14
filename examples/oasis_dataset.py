"""
OASIS dataset loader for SpikeReg

Handles the Learn2Reg OASIS dataset for unpaired brain MRI registration
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import normalize_volume, PatchNormalizer
from utils.patch_utils import extract_patches, PatchAugmentor


class OASISDataset(Dataset):
    """
    OASIS dataset for unpaired brain MRI registration
    
    Creates pairs of brain MRI scans for registration training
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 64,
        patch_stride: int = 32,
        patches_per_pair: int = 20,
        augment: bool = False,
        use_labels: bool = True,
        pairs_per_epoch: Optional[int] = None,
        fixed_pairs: bool = False,
        seed: int = 42
    ):
        """
        Args:
            data_root: Path to OASIS dataset root
            split: 'train' or 'val' or 'test'
            patch_size: Size of patches to extract
            patch_stride: Stride for patch extraction
            patches_per_pair: Number of patches to extract per volume pair
            augment: Whether to apply data augmentation
            use_labels: Whether to load segmentation labels
            pairs_per_epoch: Number of random pairs to generate per epoch
            fixed_pairs: If True, use fixed pairing (each volume with next)
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        self.fixed_pairs = fixed_pairs
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load dataset info
        with open(self.data_root / 'OASIS_dataset.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        # Get volume list based on split
        if split == 'train':
            self.volumes = self.dataset_info['training'][:350]  # First 350 for training
        elif split == 'val':
            self.volumes = self.dataset_info['training'][350:]  # Last 64 for validation
        elif split == 'test':
            # Test set is separate but not included in this dataset
            # Using some training volumes for demo
            self.volumes = self.dataset_info['training'][:39]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Initialize augmentor if needed
        if augment:
            self.augmentor = PatchAugmentor(
                rotation_range=10.0,
                scale_range=(0.9, 1.1),
                translation_range=0.05,
                flip_prob=0.5,
                noise_std=0.01
            )
        
        # Generate pairs
        self.pairs = self._generate_pairs(pairs_per_epoch)
        
        # Pre-extract patches if using fixed pairs
        if self.fixed_pairs:
            self._preextract_patches()
    
    def _generate_pairs(self, num_pairs: Optional[int] = None) -> List[Tuple[int, int]]:
        """Generate volume pairs for registration"""
        pairs = []
        n_volumes = len(self.volumes)
        
        if self.fixed_pairs:
            # Fixed pairing: each volume with next one
            for i in range(n_volumes - 1):
                pairs.append((i, i + 1))
            # Add last with first to complete circle
            pairs.append((n_volumes - 1, 0))
        else:
            # Random pairing
            if num_pairs is None:
                num_pairs = n_volumes * 2  # Default: 2 pairs per volume
            
            for _ in range(num_pairs):
                idx1 = random.randint(0, n_volumes - 1)
                idx2 = random.randint(0, n_volumes - 1)
                while idx2 == idx1:  # Ensure different volumes
                    idx2 = random.randint(0, n_volumes - 1)
                pairs.append((idx1, idx2))
        
        return pairs
    
    def _load_volume(self, volume_info: Dict) -> Dict[str, np.ndarray]:
        """Load a volume with its image, label, and mask"""
        # Adjust paths relative to data root
        image_path = self.data_root / volume_info['image'].replace('./', '')
        
        # Load image
        img_nii = nib.load(str(image_path))
        image = img_nii.get_fdata().astype(np.float32)
        
        # Load label if available and requested
        label = None
        if self.use_labels and 'label' in volume_info:
            label_path = self.data_root / volume_info['label'].replace('./', '')
            if label_path.exists():
                label_nii = nib.load(str(label_path))
                label = label_nii.get_fdata().astype(np.int32)
        
        # Load mask if available
        mask = None
        if 'mask' in volume_info:
            mask_path = self.data_root / volume_info['mask'].replace('./', '')
            if mask_path.exists():
                mask_nii = nib.load(str(mask_path))
                mask = mask_nii.get_fdata().astype(np.float32)
        
        return {
            'image': image,
            'label': label,
            'mask': mask,
            'affine': img_nii.affine
        }
    
    def _preextract_patches(self):
        """Pre-extract patches for all pairs (for fixed pairing)"""
        print(f"Pre-extracting patches for {len(self.pairs)} pairs...")
        self.all_patches = []
        
        for pair_idx, (idx1, idx2) in enumerate(self.pairs):
            # Load volumes
            vol1_data = self._load_volume(self.volumes[idx1])
            vol2_data = self._load_volume(self.volumes[idx2])
            
            # Convert to tensors and normalize
            fixed = torch.from_numpy(vol1_data['image']).float().unsqueeze(0).unsqueeze(0)
            moving = torch.from_numpy(vol2_data['image']).float().unsqueeze(0).unsqueeze(0)
            
            fixed = normalize_volume(fixed)
            moving = normalize_volume(moving)
            
            # Extract patches
            fixed_patches, coords = extract_patches(fixed, self.patch_size, self.patch_stride)
            moving_patches, _ = extract_patches(moving, self.patch_size, self.patch_stride)
            
            # Apply masks if available
            if vol1_data['mask'] is not None and vol2_data['mask'] is not None:
                mask1 = torch.from_numpy(vol1_data['mask']).float().unsqueeze(0).unsqueeze(0)
                mask2 = torch.from_numpy(vol2_data['mask']).float().unsqueeze(0).unsqueeze(0)
                
                mask1_patches, _ = extract_patches(mask1, self.patch_size, self.patch_stride)
                mask2_patches, _ = extract_patches(mask2, self.patch_size, self.patch_stride)
                
                # Filter patches with sufficient mask coverage
                valid_patches = []
                for i, (fp, mp, m1p, m2p) in enumerate(
                    zip(fixed_patches, moving_patches, mask1_patches, mask2_patches)
                ):
                    mask_coverage = (m1p.mean() + m2p.mean()) / 2
                    if mask_coverage > 0.5:  # At least 50% valid voxels
                        valid_patches.append(i)
                
                fixed_patches = [fixed_patches[i] for i in valid_patches]
                moving_patches = [moving_patches[i] for i in valid_patches]
                coords = [coords[i] for i in valid_patches]
            
            # Randomly sample patches
            n_patches = min(len(fixed_patches), self.patches_per_pair)
            if n_patches < len(fixed_patches):
                indices = np.random.choice(len(fixed_patches), n_patches, replace=False)
                fixed_patches = [fixed_patches[i] for i in indices]
                moving_patches = [moving_patches[i] for i in indices]
            
            # Add labels if available
            fixed_label_patches = None
            moving_label_patches = None
            if self.use_labels and vol1_data['label'] is not None and vol2_data['label'] is not None:
                fixed_label = torch.from_numpy(vol1_data['label']).long().unsqueeze(0).unsqueeze(0)
                moving_label = torch.from_numpy(vol2_data['label']).long().unsqueeze(0).unsqueeze(0)
                
                fixed_label_patches, _ = extract_patches(fixed_label, self.patch_size, self.patch_stride)
                moving_label_patches, _ = extract_patches(moving_label, self.patch_size, self.patch_stride)
                
                if n_patches < len(fixed_label_patches):
                    fixed_label_patches = [fixed_label_patches[i] for i in indices]
                    moving_label_patches = [moving_label_patches[i] for i in indices]
            
            # Store patches
            for i in range(len(fixed_patches)):
                patch_data = {
                    'fixed': fixed_patches[i].squeeze(0),
                    'moving': moving_patches[i].squeeze(0),
                    'pair_idx': pair_idx
                }
                
                if fixed_label_patches is not None:
                    patch_data['fixed_label'] = fixed_label_patches[i].squeeze(0)
                    patch_data['moving_label'] = moving_label_patches[i].squeeze(0)
                
                self.all_patches.append(patch_data)
        
        print(f"Extracted {len(self.all_patches)} patches total")
    
    def __len__(self):
        if self.fixed_pairs:
            return len(self.all_patches)
        else:
            return len(self.pairs) * self.patches_per_pair
    
    def __getitem__(self, idx):
        if self.fixed_pairs:
            # Return pre-extracted patch
            patch_data = self.all_patches[idx]
            
            fixed = patch_data['fixed']
            moving = patch_data['moving']
            
            # Apply augmentation if enabled
            if self.augment and self.augmentor:
                # Stack for joint augmentation
                combined = torch.cat([fixed, moving], dim=0).unsqueeze(0)
                augmented = self.augmentor.augment(combined)
                fixed = augmented[0, 0:1]
                moving = augmented[0, 1:2]
            
            output = {
                'fixed': fixed,
                'moving': moving,
                'pair_idx': patch_data['pair_idx']
            }
            
            if 'fixed_label' in patch_data:
                output['segmentation_fixed'] = patch_data['fixed_label']
                output['segmentation_moving'] = patch_data['moving_label']
            
            return output
        
        else:
            # Dynamic patch extraction
            pair_idx = idx // self.patches_per_pair
            patch_idx = idx % self.patches_per_pair
            
            idx1, idx2 = self.pairs[pair_idx]
            
            # Load volumes
            vol1_data = self._load_volume(self.volumes[idx1])
            vol2_data = self._load_volume(self.volumes[idx2])
            
            # Convert to tensors and normalize
            fixed = torch.from_numpy(vol1_data['image']).float().unsqueeze(0).unsqueeze(0)
            moving = torch.from_numpy(vol2_data['image']).float().unsqueeze(0).unsqueeze(0)
            
            fixed = normalize_volume(fixed)
            moving = normalize_volume(moving)
            
            # Extract random patch
            D, H, W = fixed.shape[2:]
            pd, ph, pw = self.patch_size, self.patch_size, self.patch_size
            
            # Random position
            d_start = random.randint(0, max(0, D - pd))
            h_start = random.randint(0, max(0, H - ph))
            w_start = random.randint(0, max(0, W - pw))
            
            # Extract patches
            fixed_patch = fixed[:, :, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            moving_patch = moving[:, :, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            
            # Apply augmentation if enabled
            if self.augment and self.augmentor:
                combined = torch.cat([fixed_patch, moving_patch], dim=1)
                augmented = self.augmentor.augment(combined)
                fixed_patch = augmented[:, 0:1]
                moving_patch = augmented[:, 1:2]
            
            output = {
                'fixed': fixed_patch.squeeze(0),
                'moving': moving_patch.squeeze(0),
                'pair_idx': pair_idx
            }
            
            # Add labels if available
            if self.use_labels and vol1_data['label'] is not None and vol2_data['label'] is not None:
                fixed_label = torch.from_numpy(vol1_data['label']).long()
                moving_label = torch.from_numpy(vol2_data['label']).long()
                
                fixed_label_patch = fixed_label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                moving_label_patch = moving_label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                
                output['segmentation_fixed'] = fixed_label_patch.unsqueeze(0)
                output['segmentation_moving'] = moving_label_patch.unsqueeze(0)
            
            return output


def create_oasis_loaders(
    data_root: str,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4
):
    """
    Create data loaders for OASIS dataset
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = OASISDataset(
        data_root,
        split='train',
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=True,
        use_labels=True,
        fixed_pairs=True  # Use fixed pairs for reproducibility
    )
    
    val_dataset = OASISDataset(
        data_root,
        split='val',
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=False,
        use_labels=True,
        fixed_pairs=True
    )
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset
    import matplotlib.pyplot as plt
    
    dataset = OASISDataset(
        '/Users/ali/Documents/codes/SpikeReg/SpikeReg/data/OASIS',
        split='train',
        patch_size=64,
        patches_per_pair=5,
        fixed_pairs=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Fixed shape: {sample['fixed'].shape}")
    print(f"Moving shape: {sample['moving'].shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    slice_idx = sample['fixed'].shape[1] // 2
    axes[0].imshow(sample['fixed'][0, slice_idx], cmap='gray')
    axes[0].set_title('Fixed')
    axes[0].axis('off')
    
    axes[1].imshow(sample['moving'][0, slice_idx], cmap='gray')
    axes[1].set_title('Moving')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('oasis_sample.png')
    print("Saved visualization to oasis_sample.png") 