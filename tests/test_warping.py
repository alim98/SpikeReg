import torch

from SpikeReg.utils.warping import SpatialTransformer


def _ramp_volume(depth=5, height=6, width=7):
    z, y, x = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return (100 * z + 10 * y + x).view(1, 1, depth, height, width)


def test_spatial_transformer_zero_displacement_is_identity():
    src = _ramp_volume()
    disp = torch.zeros(1, 3, *src.shape[2:])
    warped = SpatialTransformer()(src, disp)
    torch.testing.assert_close(warped, src)


def test_spatial_transformer_positive_z_samples_next_slice():
    src = _ramp_volume()
    disp = torch.zeros(1, 3, *src.shape[2:])
    disp[:, 0] = 1.0
    warped = SpatialTransformer()(src, disp)
    torch.testing.assert_close(warped[:, :, :-1], src[:, :, 1:])


def test_spatial_transformer_positive_y_samples_next_row():
    src = _ramp_volume()
    disp = torch.zeros(1, 3, *src.shape[2:])
    disp[:, 1] = 1.0
    warped = SpatialTransformer()(src, disp)
    torch.testing.assert_close(warped[:, :, :, :-1], src[:, :, :, 1:])


def test_spatial_transformer_positive_x_samples_next_column():
    src = _ramp_volume()
    disp = torch.zeros(1, 3, *src.shape[2:])
    disp[:, 2] = 1.0
    warped = SpatialTransformer()(src, disp)
    torch.testing.assert_close(warped[:, :, :, :, :-1], src[:, :, :, :, 1:])
