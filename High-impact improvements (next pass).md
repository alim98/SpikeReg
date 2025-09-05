Decoder skip selection (last stage): for the last decoder, you fall back to encoder_features[0] if -(i+2) goes out of range, so the last two decoder blocks may use the same skip. Prefer “no skip” at the shallowest stage or make channels consistent for a 4-level symmetric UNet. (See selection logic in SpikeReg/models.py forward.)

Docs ↔ config alignment: Your default YAML has 3 encoder levels ([32,64,128]) but the model defaults mention 4 levels and time windows of length 4. Make sure README + configs/default_config.yaml and SpikeReg/models.py tell the same story.

Requirements cleanup: requirements.txt lists nibabel twice (one pinned, one unpinned). Keep only the pinned line.

Replace always-on prints with logging: Lots of debug prints in model forward (min/max, NaN checks) and training. Consider a verbose flag or logging so training isn’t I/O-bound. (See SpikeReg/models.py forward & trainer logs.)

Vector-field composition correctness: Your scaling-and-squaring step composes d = d + warp(d, d). It’s a common trick, but add a comment (or test) verifying that channels are interpreted in voxel space and that composition matches your chosen convention.

Sanity tests to run now (quick, low-friction)

Axis check: Create a delta displacement translating +1 voxel in +x and confirm a single-voxel impulse shifts 1 voxel along x after warping. This catches issue (1) immediately.

Inverse check: Sample a small random field u, compute v = inverse_transform(u), then warp an image with u followed by v — should be ≈ identity (PSNR high).

CUDA SSIM: Move tensors to CUDA and call SSIM to ensure no device errors.

Surface distance: Call with tiny binary cubes; ensure booleans don’t crash and metrics are finite.

Minor polish

Export load_config from a small spikereg.config module, or re-export from spikereg/__init__.py, so README imports stay clean.

Remove the stray ### !!! marker in trainer logs.

Consider pinning PyTorch/torchvision pairs in requirements to avoid ABI mismatches.