#!/usr/bin/env python3
"""Visualise a full-volume pair from the Neurite-OASIS dataset.

Usage (from project root):
    python SpikeReg/test/visualise_oasis_volume.py \
           --fixed-id 438 --moving-id 439 \
           --show-warped --model checkpoints/oasis/final_model.pth
If you omit the IDs a random pair will be drawn from *pairs_val.csv* inside
``data/L2R_2021_Task3_val`` (or from train list if the CSV is missing).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_volume(data_root: Path, sid: int) -> np.ndarray:
    """Load aligned_norm.nii.gz for a given subject id (e.g. 438)."""
    subj_dir = data_root / "L2R_2021_Task3_train" / f"OASIS_OAS1_{sid:04d}_MR1"
    if not subj_dir.exists():
        subj_dir = data_root / "L2R_2021_Task3_val" / f"OASIS_OAS1_{sid:04d}_MR1"
    img_path = subj_dir / "aligned_norm.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Volume file {img_path} not found")
    return nib.load(str(img_path)).get_fdata().astype(np.float32)


def _pick_random_pair(data_root: Path) -> tuple[int, int]:
    """Return two subject ids that form a validation pair (or random)."""
    pairs_csv = data_root / "L2R_2021_Task3_val" / "pairs_val.csv"
    if pairs_csv.exists():
        import csv

        with open(pairs_csv) as f:
            pairs = [(int(r["fixed"]), int(r["moving"])) for r in csv.DictReader(f)]
        return random.choice(pairs)
    # fallback random
    subs = sorted((data_root / "L2R_2021_Task3_train").glob("OASIS_OAS1_*_MR1"))
    ids = [int(p.name.split("_")[2]) for p in subs]
    return random.sample(ids, 2)


def _plot_volume_triple(fixed: np.ndarray, moving: np.ndarray, warped: np.ndarray | None = None):
    """Show one axial/coronal/sagittal slice for each volume."""
    def center_slices(vol):
        d, h, w = vol.shape
        return vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]

    f_ax, f_co, f_sa = center_slices(fixed)
    m_ax, m_co, m_sa = center_slices(moving)
    if warped is not None:
        w_ax, w_co, w_sa = center_slices(warped)

    ncols = 3 if warped is None else 4
    fig, axes = plt.subplots(3, ncols, figsize=(3 * ncols, 9))

    # Titles row 0
    titles = ["Fixed", "Moving"] + (["Warped"] if warped is not None else []) + ["Diff"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t)

    # Row: axial
    imgs_ax = [f_ax, m_ax] + ([w_ax] if warped is not None else [])
    imgs_ax.append(np.abs(f_ax - (w_ax if warped is not None else m_ax)))
    # Row: coronal
    imgs_co = [f_co, m_co] + ([w_co] if warped is not None else [])
    imgs_co.append(np.abs(f_co - (w_co if warped is not None else m_co)))
    # Row: sagittal
    imgs_sa = [f_sa, m_sa] + ([w_sa] if warped is not None else [])
    imgs_sa.append(np.abs(f_sa - (w_sa if warped is not None else m_sa)))

    for row, imgs in enumerate([imgs_ax, imgs_co, imgs_sa]):
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
def save_volume_triple(fixed: np.ndarray, moving: np.ndarray, warped: np.ndarray | None = None, save_path: str | Path = "/teamspace/studios/this_studio/SpikeReg/test/volume_triple.png"):
    """Save one axial/coronal/sagittal slice for each volume to a file."""
    def center_slices(vol):
        d, h, w = vol.shape
        return vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]

    f_ax, f_co, f_sa = center_slices(fixed)
    m_ax, m_co, m_sa = center_slices(moving)
    if warped is not None:
        w_ax, w_co, w_sa = center_slices(warped)

    ncols = 3 if warped is None else 4
    fig, axes = plt.subplots(3, ncols, figsize=(3 * ncols, 9))

    # Titles row 0
    titles = ["Fixed", "Moving"] + (["Warped"] if warped is not None else []) + ["Diff"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t)

    # Row: axial
    imgs_ax = [f_ax, m_ax] + ([w_ax] if warped is not None else [])
    imgs_ax.append(np.abs(f_ax - (w_ax if warped is not None else m_ax)))
    # Row: coronal
    imgs_co = [f_co, m_co] + ([w_co] if warped is not None else [])
    imgs_co.append(np.abs(f_co - (w_co if warped is not None else m_co)))
    # Row: sagittal
    imgs_sa = [f_sa, m_sa] + ([w_sa] if warped is not None else [])
    imgs_sa.append(np.abs(f_sa - (w_sa if warped is not None else m_sa)))

    for row, imgs in enumerate([imgs_ax, imgs_co, imgs_sa]):
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="Path to dataset root")
    parser.add_argument("--fixed-id", type=int, help="Subject ID for fixed image (e.g. 438)")
    parser.add_argument("--moving-id", type=int, help="Subject ID for moving image")
    parser.add_argument("--show-warped", action="store_true", help="Warp moving with trained model")
    parser.add_argument("--model", default="checkpoints/oasis/final_model.pth")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.fixed_id is None or args.moving_id is None:
        args.fixed_id, args.moving_id = _pick_random_pair(data_root)
        print(f"Random pair chosen: fixed={args.fixed_id}, moving={args.moving_id}")

    fixed_vol = _load_volume(data_root, args.fixed_id)
    moving_vol = _load_volume(data_root, args.moving_id)

    warped_vol = None
    if args.show_warped:
        from SpikeReg import SpikeRegInference

        ckpt = Path(args.model)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt} not found")

        infer = SpikeRegInference(
            str(ckpt), device=args.device, patch_size=64, patch_stride=32, batch_size=8
        )
        out = infer.register(fixed_vol, moving_vol)
        warped_vol = infer.apply_deformation(moving_vol, out["displacement_field"])

    save_volume_triple(fixed_vol, moving_vol, warped_vol)


if __name__ == "__main__":
    main() 