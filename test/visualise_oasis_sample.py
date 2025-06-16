#!/usr/bin/env python3
"""Quick visualisation of one training sample from the OASIS loader.

Run from the project root:
    python SpikeReg/test/visualise_oasis_sample.py

It displays the centre axial slice of the fixed and moving patches, and
optionally the warped patch after registration with a trained model.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import torch
import os, sys

# Ensure project root is on Python path so 'examples' can be imported when the
# script is executed from within SpikeReg/test.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.oasis_dataset import create_oasis_loaders


# -----------------------------------------------------------------------------
# Helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_single_sample(
    data_root: str,
    patch_size: int,
    patch_stride: int,
    min_mean: float = 0.05,
    max_attempts: int = 20,
):
    """Return a sample whose fixed patch has mean intensity > *min_mean*.

    This avoids plotting background-only patches that look like noise.
    """

    train_loader, _ = create_oasis_loaders(
        data_root=data_root,
        batch_size=1,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=1,
        num_workers=0,
    )

    for attempt, sample in enumerate(train_loader, 1):
        if "segmentation_fixed" in sample and sample["segmentation_fixed"].max() > 0:
            return sample
        if attempt >= max_attempts:
            break

    # If no non-empty patch found, return the last sampled one
    print(
        f"Warning: couldn't find a patch with mean>{min_mean} after {max_attempts} tries; using last sample."
    )
    return sample


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise an OASIS sample")
    parser.add_argument("--data-root", default="data", help="Path to OASIS root")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--show-warped", action="store_true", help="Show warped patch using a trained model")
    parser.add_argument("--model", type=str, default="checkpoints/oasis/final_model.pth", help="Checkpoint for inference")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--min-mean", type=float, default=0.05, help="Minimum mean intensity to accept a patch")

    args = parser.parse_args()

    sample = _get_single_sample(
        args.data_root, args.patch_size, args.patch_stride, min_mean=args.min_mean
    )

    fixed  = sample["fixed"][0, 0].numpy()   # [D,H,W]
    moving = sample["moving"][0, 0].numpy()

    warped = None
    if args.show_warped:
        from SpikeReg import SpikeRegInference

        if not Path(args.model).exists():
            raise FileNotFoundError(f"Checkpoint {args.model} not found. Train a model first or disable --show-warped.")

        inference = SpikeRegInference(
            args.model,
            device=args.device,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            batch_size=8,
        )
        # SpikeRegInference expects full volumes; wrap our patch with batch=1
        fixed_vol  = fixed[None, ...]
        moving_vol = moving[None, ...]
        out = inference.register(fixed_vol, moving_vol)
        warped = inference.apply_deformation(moving_vol, out["displacement_field"])[0]

    # Plot axial slice in the middle
    z = fixed.shape[0] // 2
    ncols = 3 if warped is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(fixed[z], cmap="gray")
    axes[0].set_title("Fixed")
    axes[0].axis("off")

    axes[1].imshow(moving[z], cmap="gray")
    axes[1].set_title("Moving")
    axes[1].axis("off")

    if warped is not None:
        axes[2].imshow(warped[z], cmap="gray")
        axes[2].set_title("Warped")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    # Save the figure if output path is provided
    # if args.output:
    output_path = "/teamspace/studios/this_studio/SpikeReg/test/oasis_sample.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main() 