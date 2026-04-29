import torch

from SpikeReg.utils.patch_utils import extract_patches, stitch_patches


def test_stitch_patches_accepts_cosine_blend_mode():
    volume = torch.ones(1, 3, 8, 8, 8)
    patches, coords = extract_patches(volume, patch_size=4, stride=2)

    stitched = stitch_patches(
        patches,
        coords,
        output_shape=(8, 8, 8),
        patch_size=4,
        stride=2,
        blend_mode="cosine",
    )

    torch.testing.assert_close(stitched, volume)
