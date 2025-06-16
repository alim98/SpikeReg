Ran tool

Ran tool

Ran tool

Ran tool

Read file: SpikeReg/README.md

Read file: SpikeReg/SpikeReg/layers.py

Read file: SpikeReg/SpikeReg/neurons.py

Read file: SpikeReg/SpikeReg/models.py

Read file: SpikeReg/SpikeReg/training.py

Read file: SpikeReg/SpikeReg/registration.py

Read file: SpikeReg/SpikeReg/losses.py

Read file: SpikeReg/SpikeReg/__init__.py

What SpikeReg is about (ignoring everything that lives purely in “data” or in a *Dataset / DataLoader* file)

1. High-level goal  
   • Provide an energy-efficient, **spiking** alternative to common CNN-based deformable registration (e.g. VoxelMorph).  
   • Accept two 3-D medical volumes (fixed & moving) and output a dense 𝛿-field that warps the moving image onto the fixed one.  
   • Do so through a UNet-like encoder/decoder that has been re-implemented with Leaky-Integrate-and-Fire (LIF) neurons instead of ReLU’s, plus an iterative residual-refinement policy that gives coarse→fine alignment.

2. Core building blocks (code you will touch most often)

   a. `SpikeReg/SpikeReg/neurons.py`  
      – Implements surrogate-gradient LIF and Adaptive-LIF neurons, including optional lateral inhibition and spike-rate statistics that training uses for regularisation.

   b. `SpikeReg/SpikeReg/layers.py`  
      – Wraps standard Conv3d and ConvTranspose3d in the spiking logic above.  
      – Supplies reusable blocks: `SpikingEncoderBlock`, `SpikingDecoderBlock`, an attention gate, and a last-layer `OutputProjection` that maps spike trains to a 3-channel displacement field.

   c. `SpikeReg/SpikeReg/models.py`  
      – Assembles the blocks into the **SpikeRegUNet**.  
      – Contains an “ANN twin” (`PretrainedUNet`) that can be trained first, then converted layer-by-layer to spiking (`convert_pretrained_to_spiking`) as a form of weight initialisation.  
      – Also gives helper `encode_to_spikes` for Poisson rate coding of the input pair.

   d. `SpikeReg/SpikeReg/registration.py`  
      – `IterativeRegistration` wraps the UNet in a loop that:  
        1. warps the moving image with the current field,  
        2. feeds (fixed, warped) back into the UNet to predict a **residual** field,  
        3. adds the residual to the running field, until similarity saturates or a fixed iteration budget is reached.  
      – `SpikeRegInference` is the user-facing API: loads a checkpoint, cuts a full CT/MR volume into overlapping patches, runs the iterative loop in batches, stitches the result, optionally smooths / post-processes, and exposes helper utilities (apply deformation, Jacobian determinant, etc.).

   e. `SpikeReg/SpikeReg/losses.py`  
      – Groups three ingredients into one `SpikeRegLoss`:  
        • Image similarity (NCC or MI)  
        • Field regularisation (bending energy or diffusion)  
        • Spiking constraints (keep average spike-rate near a target & encourage layer-balance)  

   f. `SpikeReg/SpikeReg/training.py`  
      – `SpikeRegTrainer` glues everything together: pretraining (optional), conversion, fine-tuning, validation, TensorBoard logging, saving & loading checkpoints, LR schedulers, gradient clipping, etc.  
      – Interacts with the similarity / spike statistics coming out of `SpikeRegLoss` and with the `IterativeRegistration` wrapper for forward passes.

   g. Utils  
      – `utils.warping.py` = differentiable Spatial Transformer (trilinear sampler)  
      – `utils.patch_utils.py` = extract / stitch 3-D patches, plus a small augmentation class for random flips, intensity noise, etc.  
      – `utils.metrics.py` = Dice, NCC, Jacobian-determinant stats, Hausdorff distance…  
      – `utils.preprocessing.py` = normalisation / clipping helpers.

3. Typical workflow encoded in the repo

   1. (Optional) **Train** `PretrainedUNet` with standard back-prop.  
   2. Call `convert_pretrained_to_spiking` → produces a SpikeRegUNet whose weights & thresholds are initialised from the ANN.  
   3. Fine-tune the spiking network with `SpikeRegTrainer`, which automatically includes spike-regularisation.  
   4. Export best checkpoint.  
   5. Use `SpikeRegInference.register()` for deployment or evaluation.

4. Where you would extend / modify things

   • Experiment with new surrogate functions or neuron dynamics → edit `neurons.py`.  
   • Change architecture depth, channel counts, attention, etc. → update config YAML & block definitions in `models.py`.  
   • Explore different loss trade-offs or add topology-preserving terms → `losses.py`.  
   • Try alternative stopping criteria or coarse-to-fine schedules → `registration.py`.  
   • Integrate a neuromorphic backend (Loihi driver, lava-dlf) → wrap calls inside `models.py` & `layers.py`.

Everything above runs independently of the dataset definitions; the `Task3Dataset` and the `create_task3_loaders` helper only provide the *input batches* consumed by the trainer.