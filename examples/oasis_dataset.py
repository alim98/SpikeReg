import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from SpikeReg.utils.patch_utils import extract_patches, PatchAugmentor

try:
    import nibabel as nib
except ImportError:  # PKL-only L2R runs do not need nibabel.
    nib = None


# ---------------------------------------------------------------------------
# Volume helpers  (ported from hvit/src/data/datasets.py)
# ---------------------------------------------------------------------------

def resize_volume(volume: torch.Tensor, spatial_size, mode: str) -> torch.Tensor:
    if tuple(int(d) for d in volume.shape[1:]) == tuple(int(d) for d in spatial_size):
        return volume
    volume = volume.unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(volume.float(), size=spatial_size, mode=mode)
    else:
        resized = F.interpolate(volume.float(), size=spatial_size, mode=mode, align_corners=False)
    return resized.squeeze(0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """1st–99th percentile clip → [0, 1], robust to MRI outliers."""
    image = image.astype(np.float32)
    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return np.zeros_like(image, dtype=np.float32)
    valid = image[finite_mask]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    image[~finite_mask] = 0.0
    return image.astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI loading. Install nibabel or use dataset_format: pkl.")
    volume = np.asarray(nib.load(str(path)).dataobj)
    if volume.ndim == 4:
        volume = volume[..., 0]
    return volume


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OASISSubject:
    subject_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class OASISL2RMetadata:
    train_subject_ids: List[str]
    val_subject_ids: List[str]
    test_subject_ids: List[str]
    val_pairs: List[Tuple[str, str]]
    test_pairs: List[Tuple[str, str]]
    native_shape: Tuple[int, int, int]
    eval_label_ids: List[int]


# ---------------------------------------------------------------------------
# Metadata / subject discovery  (ported from hvit/src/data/datasets.py)
# ---------------------------------------------------------------------------

def _parse_oasis_l2r_subject_id(relative_path) -> str:
    if isinstance(relative_path, dict):
        relative_path = relative_path.get("image", "")
    return Path(relative_path).name.replace("_0000.nii.gz", "")


def _parse_oasis_l2r_pair_list(entries: List[Dict]) -> List[Tuple[str, str]]:
    return [
        (
            _parse_oasis_l2r_subject_id(item["fixed"]),
            _parse_oasis_l2r_subject_id(item["moving"]),
        )
        for item in entries
    ]


def load_oasis_l2r_metadata(data_path: Path) -> OASISL2RMetadata:
    metadata_path = data_path / "OASIS_dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"L2R OASIS metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text())
    native_shape = tuple(int(d) for d in payload["tensorImageShape"]["0"])
    eval_label_ids = list(range(1, 36))
    train_subject_ids = sorted(
        _parse_oasis_l2r_subject_id(e["image"]) for e in payload.get("training", [])
    )
    val_pairs = _parse_oasis_l2r_pair_list(payload.get("registration_val", []))
    val_subject_ids = sorted({sid for pair in val_pairs for sid in pair})
    test_pairs = _parse_oasis_l2r_pair_list(payload.get("registration_test", []))
    test_subject_ids = sorted(_parse_oasis_l2r_subject_id(p) for p in payload.get("test", []))
    # Exclude val subjects from the training set
    train_subject_ids = [s for s in train_subject_ids if s not in set(val_subject_ids)]
    return OASISL2RMetadata(
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=test_subject_ids,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        native_shape=native_shape,
        eval_label_ids=eval_label_ids,
    )


def discover_oasis_l2r_subjects(data_path: Path, subset: str = "train") -> Dict[str, OASISSubject]:
    if subset == "train":
        image_dir = data_path / "imagesTr"
        label_dir = data_path / "labelsTr"
    elif subset == "test":
        image_dir = data_path / "imagesTs"
        label_dir = data_path / "labelsTs"
    else:
        raise ValueError(f"Unsupported L2R subset: {subset}")
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"L2R OASIS layout requires {image_dir.name}/ and {label_dir.name}/ under {data_path}"
        )
    subjects: Dict[str, OASISSubject] = {}
    for image_path in sorted(image_dir.glob("*.nii.gz")):
        stem = image_path.name
        label_path = label_dir / stem
        if not label_path.exists():
            continue
        subject_id = stem.replace("_0000.nii.gz", "")
        subjects[subject_id] = OASISSubject(
            subject_id=subject_id,
            image_path=image_path,
            label_path=label_path,
        )
    if not subjects:
        raise FileNotFoundError(f"No L2R OASIS image/label pairs found under {data_path}")
    return subjects


# ---------------------------------------------------------------------------
# Full-volume L2R dataset
# ---------------------------------------------------------------------------

class OASISL2RDataset(Dataset):
    """
    Full-volume OASIS L2R Task-3 dataset.

    Expects the official L2R layout::

        <data_path>/
            imagesTr/OASIS_0001_0000.nii.gz  ...
            labelsTr/OASIS_0001_0000.nii.gz  ...
            OASIS_dataset.json

    Train split: random pairs, length = num_steps.
    Val   split: official L2R val pairs from OASIS_dataset.json, is_pair mode.

    Each item is a dict::

        {
            "fixed":               float32 [1, D, H, W],
            "moving":              float32 [1, D, H, W],
            "segmentation_fixed":  int64   [1, D, H, W],  (if use_labels)
            "segmentation_moving": int64   [1, D, H, W],  (if use_labels)
        }
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        input_dim: Tuple[int, int, int] = (160, 192, 224),
        is_pair: bool = False,
        num_steps: int = 1000,
        max_subjects: int = 0,
        max_pairs: int = 0,
        use_labels: bool = True,
    ):
        self.input_dim = tuple(int(d) for d in input_dim)
        self.is_pair = is_pair
        self.num_steps = num_steps
        self.split = split
        self.use_labels = use_labels

        self.metadata = load_oasis_l2r_metadata(Path(data_path))
        self.native_shape = self.metadata.native_shape
        self.eval_label_ids = list(self.metadata.eval_label_ids)
        self.resampled = self.input_dim != tuple(self.native_shape)

        train_subjects = discover_oasis_l2r_subjects(Path(data_path), subset="train")

        if split == "train":
            split_ids = list(self.metadata.train_subject_ids)
            if max_subjects > 0:
                split_ids = split_ids[:max_subjects]
            self.subjects = [train_subjects[sid] for sid in split_ids if sid in train_subjects]
            if len(self.subjects) < 2:
                raise ValueError(f"Need at least 2 train subjects; got {len(self.subjects)}")
            self.pairs = None  # random sampling

        elif split == "val":
            self.subjects = [
                train_subjects[sid]
                for sid in self.metadata.val_subject_ids
                if sid in train_subjects
            ]
            self.pairs = [
                (train_subjects[f], train_subjects[m])
                for f, m in self.metadata.val_pairs
                if f in train_subjects and m in train_subjects
            ]
            if max_pairs > 0:
                self.pairs = self.pairs[:max_pairs]
            if not self.pairs:
                raise ValueError("No val pairs could be resolved from OASIS_dataset.json.")

        else:
            raise ValueError(f"Unsupported split: {split!r}. Use 'train' or 'val'.")

        self.num_labels = (
            max(int(load_volume(s.label_path).max()) for s in self.subjects) + 1
        )

    def _load_subject(self, subject: OASISSubject) -> Tuple[torch.Tensor, torch.Tensor]:
        image = normalize_image(load_volume(subject.image_path))
        label = load_volume(subject.label_path).astype(np.int64)
        image_t = torch.from_numpy(image).unsqueeze(0)   # [1, D, H, W]
        label_t = torch.from_numpy(label).unsqueeze(0)   # [1, D, H, W]
        if self.resampled:
            image_t = resize_volume(image_t, self.input_dim, mode="trilinear")
            label_t = resize_volume(label_t.float(), self.input_dim, mode="nearest").long()
        return image_t, label_t.long()

    def __getitem__(self, index: int) -> Dict:
        if self.split == "val":
            fixed_subj, moving_subj = self.pairs[index]
        else:
            rng = random.Random(index)
            fixed_subj, moving_subj = rng.sample(self.subjects, 2)

        fixed_img, fixed_lbl = self._load_subject(fixed_subj)
        moving_img, moving_lbl = self._load_subject(moving_subj)

        out: Dict = {"fixed": fixed_img, "moving": moving_img}
        if self.use_labels:
            out["segmentation_fixed"] = fixed_lbl
            out["segmentation_moving"] = moving_lbl
        return out

    def __len__(self) -> int:
        if self.split == "val":
            return len(self.pairs)
        return self.num_steps


class L2RPKLDataset(Dataset):
    """
    Full-volume L2R PKL dataset used by the HViT/OASIS_L2R_2021_task03 pipeline.

    Train directory:
        All/p_XXXX.pkl -> (image, segmentation)

    Validation directory:
        Test/p_XXXX_YYYY.pkl -> (fixed, moving, fixed_seg, moving_seg)
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        input_dim: Tuple[int, int, int] = (160, 192, 224),
        num_steps: int = 1000,
        max_pairs: int = 0,
        use_labels: bool = True,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.input_dim = tuple(int(d) for d in input_dim)
        self.num_steps = int(num_steps)
        self.use_labels = use_labels
        self.eval_label_ids = list(range(1, 36))

        if split == "train":
            self.subject_files = sorted(self.data_path.glob("*.pkl"))
            if len(self.subject_files) < 2:
                raise FileNotFoundError(f"Need at least 2 L2R subject PKLs under {self.data_path}")
            self.pair_files = None
        elif split in {"val", "test"}:
            self.pair_files = sorted(self.data_path.glob("*.pkl"))
            if max_pairs > 0:
                self.pair_files = self.pair_files[:max_pairs]
            if not self.pair_files:
                raise FileNotFoundError(f"No L2R pair PKLs found under {self.data_path}")
            self.subject_files = []
        else:
            raise ValueError(f"Unsupported split: {split!r}")

    def _resize_image(self, image: np.ndarray) -> torch.Tensor:
        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        if tuple(image_t.shape[1:]) != self.input_dim:
            image_t = resize_volume(image_t, self.input_dim, mode="trilinear")
        image_t = torch.clamp(image_t, 0.0, 1.0)
        return image_t

    def _resize_label(self, label: np.ndarray) -> torch.Tensor:
        label_t = torch.from_numpy(label.astype(np.int64)).unsqueeze(0)
        if tuple(label_t.shape[1:]) != self.input_dim:
            label_t = resize_volume(label_t.float(), self.input_dim, mode="nearest").long()
        return label_t.long()

    def _load_subject(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = pickle.load(open(path, "rb"))
        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            raise ValueError(f"Expected subject PKL tuple(image, seg), got {type(payload)} from {path}")
        image, label = payload[0], payload[1]
        return self._resize_image(image), self._resize_label(label)

    def _load_pair(self, path: Path) -> Dict[str, torch.Tensor]:
        payload = pickle.load(open(path, "rb"))
        if not isinstance(payload, (tuple, list)) or len(payload) not in {2, 4}:
            raise ValueError(f"Expected pair PKL tuple(fixed, moving[, fixed_seg, moving_seg]), got {type(payload)} from {path}")

        fixed = self._resize_image(payload[0])
        moving = self._resize_image(payload[1])
        out: Dict[str, torch.Tensor] = {"fixed": fixed, "moving": moving}

        if self.use_labels and len(payload) == 4:
            out["segmentation_fixed"] = self._resize_label(payload[2])
            out["segmentation_moving"] = self._resize_label(payload[3])
        return out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.split == "train":
            # DistributedSampler passes rank-specific indices; using the index
            # here keeps random pair generation rank-sharded and reproducible.
            rng = random.Random(index)
            fixed_path, moving_path = rng.sample(self.subject_files, 2)
            fixed_img, fixed_lbl = self._load_subject(fixed_path)
            moving_img, moving_lbl = self._load_subject(moving_path)
            out: Dict[str, torch.Tensor] = {"fixed": fixed_img, "moving": moving_img}
            if self.use_labels:
                out["segmentation_fixed"] = fixed_lbl
                out["segmentation_moving"] = moving_lbl
            return out

        return self._load_pair(self.pair_files[index])

    def __len__(self) -> int:
        if self.split == "train":
            return self.num_steps
        return len(self.pair_files)


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def create_oasis_full_volume_loaders(
    data_path: str,
    batch_size: int = 1,
    target_shape: Tuple[int, int, int] = (160, 192, 224),
    num_steps: int = 1000,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Return (train_loader, val_loader) backed by the official L2R OASIS layout.

    data_path should be the L2R root that contains imagesTr/, labelsTr/,
    and OASIS_dataset.json (i.e. what the config calls train_dir).
    """
    train_ds = OASISL2RDataset(
        data_path=data_path,
        split="train",
        input_dim=target_shape,
        is_pair=False,
        num_steps=num_steps,
        use_labels=True,
    )
    val_ds = OASISL2RDataset(
        data_path=data_path,
        split="val",
        input_dim=target_shape,
        is_pair=True,
        use_labels=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def create_l2r_pkl_loaders(
    train_path: str,
    val_path: str,
    batch_size: int = 1,
    target_shape: Tuple[int, int, int] = (160, 192, 224),
    num_steps: int = 1000,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = L2RPKLDataset(
        data_path=train_path,
        split="train",
        input_dim=target_shape,
        num_steps=num_steps,
        use_labels=True,
    )
    val_ds = L2RPKLDataset(
        data_path=val_path,
        split="val",
        input_dim=target_shape,
        use_labels=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Legacy patch-based loader  (kept for non-full-volume mode)
# ---------------------------------------------------------------------------

def _read_pairs_csv(csv_path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not csv_path or not csv_path.exists():
        return pairs
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                pairs.append((int(str(row["fixed"]).strip()), int(str(row["moving"]).strip())))
            except (KeyError, ValueError, TypeError):
                continue
    return pairs


class L2RTask3Dataset(Dataset):
    """Patch-based OASIS L2R Task-3 dataset (legacy, for non-full-volume mode)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: int = 64,
        patch_stride: int = 32,
        patches_per_pair: int = 20,
        augment: bool = False,
        use_labels: bool = True,
        pairs_csv: Optional[str] = None,
        fixed_pairs: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        self.pairs_csv = Path(pairs_csv) if pairs_csv else None

        random.seed(seed)
        np.random.seed(seed)

        if self.augment:
            self.augmentor = PatchAugmentor()
        else:
            self.augmentor = None

        self.volumes = self._discover_volumes()

        if self.split == "val" and self.pairs_csv is not None:
            self.pairs = self._pairs_from_csv()
            if len(self.pairs) == 0:
                self.pairs = self._generate_pairs(len(self.volumes) * 2)
        else:
            self.pairs = self._generate_pairs(len(self.volumes) * 2)

    def _discover_volumes(self) -> List[dict]:
        volumes = []
        if self.split == "train":
            for subj_dir in sorted(self.data_dir.glob("OASIS_OAS1_*_MR1")):
                try:
                    sid = int(subj_dir.name.split("_")[2])
                except (IndexError, ValueError):
                    continue
                img = subj_dir / "aligned_norm.nii.gz"
                if not img.exists():
                    nii_files = list(subj_dir.glob("*.nii.gz"))
                    img = nii_files[0] if nii_files else None
                label = subj_dir / "aligned_seg35.nii.gz"
                if self.use_labels and not label.exists():
                    seg_files = list(subj_dir.glob("seg*.nii.gz"))
                    label = seg_files[0] if seg_files else None
                volumes.append({"id": sid, "image": img, "label": label})
        elif self.split in {"val", "test"}:
            for img_path in sorted(self.data_dir.glob("img*.nii.gz")):
                try:
                    sid_str = img_path.stem.replace("img", "")
                    if sid_str.endswith(".nii"):
                        sid_str = sid_str[:-4]
                    sid = int(sid_str)
                except ValueError:
                    continue
                label_path = self.data_dir / f"seg{sid:04d}.nii.gz"
                label = label_path if label_path.exists() else None
                volumes.append({"id": sid, "image": img_path, "label": label})
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return volumes

    def _pairs_from_csv(self) -> List[Tuple[int, int]]:
        csv_pairs = _read_pairs_csv(self.pairs_csv)
        id_to_idx = {v["id"]: i for i, v in enumerate(self.volumes)}
        return [(id_to_idx[f], id_to_idx[m]) for f, m in csv_pairs if f in id_to_idx and m in id_to_idx]

    def _generate_pairs(self, num_pairs: int) -> List[Tuple[int, int]]:
        n = len(self.volumes)
        pairs = []
        for _ in range(num_pairs):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)
            pairs.append((i, j))
        return pairs

    def _load_volume(self, info: dict) -> dict:
        img = nib.load(str(info["image"])).get_fdata().astype(np.float32)
        label = None
        if self.use_labels and info.get("label") and Path(info["label"]).exists():
            label = nib.load(str(info["label"])).get_fdata().astype(np.int32)
        return {"image": img, "label": label}

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, idx: int) -> dict:
        pair_idx = idx // self.patches_per_pair
        idx1, idx2 = self.pairs[pair_idx]
        vol1 = self._load_volume(self.volumes[idx1])
        vol2 = self._load_volume(self.volumes[idx2])

        fixed = normalize_image(vol1["image"])
        moving = normalize_image(vol2["image"])
        fixed = torch.from_numpy(fixed).float().unsqueeze(0)
        moving = torch.from_numpy(moving).float().unsqueeze(0)

        fixed_patches, _ = extract_patches(fixed, self.patch_size, self.patch_stride)
        moving_patches, _ = extract_patches(moving, self.patch_size, self.patch_stride)

        if not fixed_patches or not moving_patches:
            raise RuntimeError("Failed to extract patches")

        max_valid_idx = min(len(fixed_patches), len(moving_patches)) - 1

        def _is_valid(p: torch.Tensor) -> bool:
            return (p > 0.10).float().mean().item() > 0.50

        patch_idx = None
        for _ in range(50):
            cand = random.randint(0, max_valid_idx)
            if _is_valid(fixed_patches[cand]) and _is_valid(moving_patches[cand]):
                patch_idx = cand
                break
        if patch_idx is None:
            patch_idx = 0

        out = {
            "fixed": fixed_patches[patch_idx],
            "moving": moving_patches[patch_idx],
            "pair_idx": pair_idx,
        }

        if self.augment and self.augmentor is not None:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            out["fixed"] = self.augmentor.augment(out["fixed"])
            torch.manual_seed(seed)
            out["moving"] = self.augmentor.augment(out["moving"])

        if self.use_labels and vol1["label"] is not None and vol2["label"] is not None:
            fixed_lbl = torch.from_numpy(vol1["label"]).long().unsqueeze(0)
            moving_lbl = torch.from_numpy(vol2["label"]).long().unsqueeze(0)
            fixed_lbl_patches, _ = extract_patches(fixed_lbl, self.patch_size, self.patch_stride)
            moving_lbl_patches, _ = extract_patches(moving_lbl, self.patch_size, self.patch_stride)
            if (fixed_lbl_patches and moving_lbl_patches
                    and patch_idx < len(fixed_lbl_patches)
                    and patch_idx < len(moving_lbl_patches)):
                out["segmentation_fixed"] = fixed_lbl_patches[patch_idx]
                out["segmentation_moving"] = moving_lbl_patches[patch_idx]

        return out


OASISDataset = L2RTask3Dataset


def create_task3_loaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:

    pairs_csv = Path(val_dir) / "pairs_val.csv"
    pairs_csv_str = str(pairs_csv) if pairs_csv.exists() else None

    train_ds = L2RTask3Dataset(train_dir, split="train", patch_size=patch_size,
                               patch_stride=patch_stride, patches_per_pair=patches_per_pair,
                               augment=True, use_labels=True)
    val_ds = L2RTask3Dataset(val_dir, split="val", patch_size=patch_size,
                             patch_stride=patch_stride, patches_per_pair=patches_per_pair,
                             augment=False, use_labels=True, pairs_csv=pairs_csv_str)

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)

    test_loader = None
    if test_dir is not None:
        test_ds = L2RTask3Dataset(test_dir, split="test", patch_size=patch_size,
                                  patch_stride=patch_stride, patches_per_pair=patches_per_pair,
                                  augment=False, use_labels=False)
        test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def create_oasis_loaders(
    data_root: str,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = Path(data_root)
    train_dir = root / "L2R_2021_Task3_train"
    val_dir = root / "L2R_2021_Task3_val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train directory {train_dir} not found.")
    if not val_dir.exists():
        raise FileNotFoundError(f"Expected val directory {val_dir} not found.")
    train_loader, val_loader, _ = create_task3_loaders(
        str(train_dir), str(val_dir), test_dir=None,
        batch_size=batch_size, patch_size=patch_size, patch_stride=patch_stride,
        patches_per_pair=patches_per_pair, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader
