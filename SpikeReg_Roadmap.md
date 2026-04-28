# SpikeReg: From Broken Prototype to High-Impact Publication

## A Brutal Honest Assessment + Complete Roadmap

---

## PART 1: WHAT YOU ACTUALLY HAVE (AND WHAT'S WRONG)

### 1.1 The Core Idea

You're building **the first spiking neural network (SNN) for 3D deformable medical image registration**. The pitch: replace the standard CNN-based UNet (VoxelMorph, TransMorph, etc.) with a spiking UNet that uses Leaky Integrate-and-Fire (LIF) neurons, achieving comparable accuracy at dramatically lower energy consumption — making it deployable on neuromorphic hardware like Intel Loihi 2.

**This is actually a great idea.** As of late 2025, there is **no published work** applying SNNs to deformable image registration. SNNs for medical imaging are limited to classification and segmentation (one paper deployed a U-Net on Loihi for 2D segmentation). Registration is completely untouched territory. If you execute properly, this is a genuinely novel contribution.

### 1.2 What's Fundamentally Broken Right Now

Here's why your current results are bad — issue by issue:

#### PROBLEM 1: The Pretrain Loss is MSE on Warped vs Fixed — This is Wrong

In `training.py` line 436, during pretraining:
```python
loss = self.criterion(warped, fixed)  # MSE
```
MSE is a terrible similarity metric for inter-subject brain registration. Brains from different subjects have different intensity profiles. **NCC should be used from the very beginning**, even in pretraining. MSE will push the ANN into learning identity-like mappings or degenerate solutions because it penalizes any intensity difference, not just misalignment.

#### PROBLEM 2: Poisson Rate Coding Destroys Spatial Information

In `models.py` `encode_to_spikes()`, you do Poisson rate coding: for each voxel, you generate random spikes proportional to the intensity. Then in the forward pass (line 189), you **immediately average these spikes back to rates** before feeding to the encoder:
```python
x_rates = spike_input.mean(dim=1)
```
This means you:
1. Take a continuous image
2. Convert it to noisy binary spikes
3. Average those spikes back to a noisy version of the original image

This is **pure noise injection** that destroys gradient flow. It adds stochasticity for no benefit during GPU-based training. The Poisson encoding only makes sense on actual neuromorphic hardware at inference time.

#### PROBLEM 3: Tiny Patch Size (32³) Loses Global Context

Your config uses `patch_size: 32` and you're resampling volumes to 128³. A 32³ patch is only **1/64th** of the volume. Brain registration requires long-range spatial correspondence — you need to know where the hippocampus is relative to the cortex. State-of-the-art methods (VoxelMorph, TransMorph) operate on **full resolution volumes** (160×192×224) or at minimum 128³ full-brain. Your patch-based approach means:
- Each patch only sees a tiny local region
- The stitching creates discontinuities at boundaries
- You lose all global anatomical context

#### PROBLEM 4: The Iterative Registration Loop is Over-Engineered and Unstable

`IterativeRegistration` runs 10 iterations, each time resetting all neurons, running full forward pass, predicting a residual displacement, and composing. This is computationally expensive and the composition of noisy residuals accumulates errors. The NaN warnings scattered everywhere in the code confirm this instability. For a first proof-of-concept, a single-pass prediction (like VoxelMorph) is more stable and sufficient.

#### PROBLEM 5: No Proper Threshold Calibration After ANN→SNN Conversion

`convert_pretrained_to_spiking()` copies weights and BN parameters but the comment on line 533 says:
```python
# Normalize thresholds based on activation statistics
# (This would require running calibration data through the network)
```
**This is never implemented.** Without proper threshold calibration, LIF neurons either fire constantly (all-ones, losing information) or never fire (dead neurons, zero gradients). This is the #1 reason ANN-to-SNN conversion fails in practice.

#### PROBLEM 6: Dataset Mismatch and Evaluation Gap

You're using the old L2R 2021 Task 3 OASIS data (neurite-oasis, ~400 subjects, 160³ skull-stripped T1w MRIs). This is fine as a starting dataset, but:
- Learn2Reg has moved to LUMIR (3,384 subjects, OpenBHB + OASIS) for the 2024/2025 challenges
- Your current eval only computes NCC, MSE, SSIM, and Jacobian stats on patches — not the standard metrics (Dice on anatomical ROIs, TRE on landmarks, HD95, NDV)
- You have no proper test set evaluation, no comparison to baselines

#### PROBLEM 7: The Energy Claims are Fabricated

Your README claims "~10⁻² × lower energy consumption" and "4.5 mJ on Loihi 2" — but there is zero actual measurement or even simulation backing this. These are placeholder numbers. You cannot claim energy results without either: (a) actual neuromorphic hardware deployment, or (b) a rigorous analytical energy model using operation counts and published hardware specs.

---

## PART 2: THE PUBLICATION TARGET AND POSITIONING

### 2.1 Where This Could Be Published (If Done Right)

- **Top tier**: MICCAI 2026/2027 (Medical Image Computing), Medical Image Analysis (MedIA)
- **Strong tier**: IEEE TMI, NeuroImage, IPMI
- **SNN-focused**: Frontiers in Neuroscience, Neuromorphic Computing and Engineering

### 2.2 The Novel Claims You Can Actually Make

1. **First SNN for deformable medical image registration** — genuinely novel
2. **ANN-to-SNN conversion pipeline for dense prediction tasks in 3D** — most SNN literature is classification; dense 3D regression is hard and underexplored
3. **Systematic energy analysis** showing theoretical energy reduction on neuromorphic hardware
4. **Accuracy-efficiency Pareto analysis** — even if Dice is slightly lower than TransMorph, being within 2-3% at 10-100× lower energy is publishable

### 2.3 What Reviewers Will Demand

1. Dice scores on standard anatomical ROIs (35 Mindboggle labels for OASIS)
2. Comparison against VoxelMorph, VoxelMorph-diff, TransMorph, LKU-Net (at minimum)
3. Jacobian analysis (% negative det, smoothness)
4. Ablation studies (ANN vs SNN, effect of time steps, effect of spike regularization)
5. Rigorous energy estimation (not fabricated numbers)

---

## PART 3: THE COMPLETE STEP-BY-STEP ROADMAP

### PHASE 0: Fix the Foundations (Week 1-2)

#### Step 0.1: Switch to Full-Volume Processing at 128³

**Why**: Patch-based registration at 32³ is fundamentally limited. All competitive methods work on full volumes.

**What to change**:
- Set `patch_size` = `resample_size` = `[128, 128, 128]` (or work at `[160, 192, 160]` if GPU memory allows)
- Remove the patch extraction/stitching from training entirely
- Keep patch-based inference as optional for deployment on memory-constrained hardware
- Reduce `encoder_channels` to `[16, 32, 64, 128]` to fit on GPU (your current `[32, 64, 128, 256]` at full resolution will OOM)

#### Step 0.2: Fix the Pretrain Loss

**Change in `_init_loss()`**: Replace MSE with NCC for pretraining:
```python
if self.training_phase == 'pretrain':
    from .losses import NormalizedCrossCorrelation, DiffusionRegularizer
    self.similarity = NormalizedCrossCorrelation(window_size=9)
    self.regularizer = DiffusionRegularizer(weight=1.0)
```

#### Step 0.3: Fix the Spatial Transformer Convention

Your `warping.py` reorders displacement channels (line 49-52). This is fragile and error-prone. Verify with a known displacement (e.g., rigid translation) that the warping produces the correct result. Many failed registration projects die here.

**Test**: Create a synthetic pair where moving = fixed shifted by 5 voxels in z. The predicted displacement should be approximately [5, 0, 0] or [0, 0, 5] depending on your convention. If it isn't, fix the spatial transformer.

### PHASE 1: Get the ANN Baseline Working Properly (Week 2-4)

#### Step 1.1: Train a Proper ANN U-Net Baseline

Before touching SNNs, you need a working ANN that achieves competitive Dice. Your `PretrainedUNet` is a reasonable architecture. Train it with:

- **Loss**: NCC (window=9) + λ × diffusion regularization (λ = 1.0)
- **Volume**: Full 128³ (not patches)
- **Optimizer**: Adam, lr=1e-4, 300 epochs minimum
- **Augmentation**: Random affine (rotation ±10°, scale 0.95-1.05), intensity augmentation
- **Target Dice**: ~0.72-0.76 on Mindboggle-35 labels (this is the VoxelMorph range for OASIS)

#### Step 1.2: Implement Proper Evaluation

Create a dedicated evaluation script that computes:

1. **Dice score** per anatomical ROI (using OASIS segmentation labels, Mindboggle-35 protocol)
2. **Mean Dice** across all ROIs (the primary metric)
3. **% of voxels with |Jac det| ≤ 0** (smoothness/invertibility)
4. **95% Hausdorff distance** (surface accuracy)
5. **NCC on held-out test pairs**

Use the standard OASIS evaluation pairs (the L2R Task3 validation pairs from `pairs_val.csv`).

#### Step 1.3: Compare Against Published Baselines

Train or use published checkpoints for:
- **VoxelMorph** (the standard CNN baseline) — Dice ~0.74 on OASIS
- **VoxelMorph-diff** (diffeomorphic variant) — Dice ~0.73
- **TransMorph** (transformer baseline) — Dice ~0.75-0.76
- **SyN (ANTs)** — classical optimization baseline — Dice ~0.75

If your ANN can't match VoxelMorph (~0.74 Dice), the SNN version won't either. Fix the ANN first.

### PHASE 2: Proper ANN-to-SNN Conversion (Week 4-6)

#### Step 2.1: Implement Threshold Calibration

This is the missing critical piece. After training the ANN, run a calibration pass:

```python
def calibrate_thresholds(pretrained_model, calibration_loader, percentile=99.5):
    """Run calibration data through ANN and set SNN thresholds based on activation statistics"""
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(output.detach().cpu())
        return fn
    
    # Register hooks on every conv+bn+relu block
    for name, module in pretrained_model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration data
    with torch.no_grad():
        for batch in calibration_loader:
            pretrained_model(batch['fixed'].cuda(), batch['moving'].cuda())
    
    # Compute thresholds per layer
    thresholds = {}
    for name, activations in activation_stats.items():
        all_acts = torch.cat([a.flatten() for a in activations])
        thresholds[name] = torch.quantile(all_acts[all_acts > 0], percentile/100).item()
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return thresholds
```

Then set each LIF neuron's `v_th` to the calibrated threshold for that layer.

#### Step 2.2: Remove Poisson Encoding During Training

During GPU training, do NOT use Poisson rate coding. Instead:
- Pass continuous values directly into the spiking network
- The LIF neurons will naturally integrate and fire based on the continuous input current
- This preserves gradient flow and removes unnecessary stochasticity

Only use Poisson encoding at inference time on neuromorphic hardware (or for energy analysis).

#### Step 2.3: Tune Time Steps Carefully

Start with **T=4 time steps** (not 10). More time steps = more computation, diminishing accuracy returns. The optimal T for most ANN-to-SNN conversion tasks is 4-8. Use T as a hyperparameter and report the accuracy-latency tradeoff.

### PHASE 3: SNN Fine-Tuning (Week 6-8)

#### Step 3.1: Surrogate Gradient Training

After conversion + calibration, fine-tune the SNN with:
- **Loss**: NCC + diffusion regularization + spike rate regularization
- **Optimizer**: Adam, lr=5e-5 (lower than ANN)
- **Freeze BN** (already implemented — good)
- **Gradient clipping**: 1.0 (already implemented)
- **Spike rate target**: 0.05-0.15 (tune this — too low = dead neurons, too high = no energy saving)

#### Step 3.2: Single-Pass Prediction First

Remove the iterative registration loop for now. Train with a single forward pass predicting the displacement field directly. Once this works, you can add cascaded refinement (2-3 iterations max, not 10).

#### Step 3.3: Monitor Spike Rates Per Layer

Log the mean spike rate per layer every epoch. Healthy range is 0.05-0.20. If any layer drops below 0.01, it's dead — lower its threshold. If any layer exceeds 0.5, it's wasteful — raise its threshold or increase regularization.

### PHASE 4: Rigorous Evaluation (Week 8-10)

#### Step 4.1: Full Benchmark Suite

Run evaluation on the OASIS test set with:
- Your ANN baseline
- Your SNN (T=4, T=6, T=8)
- VoxelMorph (retrained or published)
- TransMorph (published checkpoint)
- ANTs SyN (classical baseline)

Report a table like:

| Method | Dice ↑ | HD95 ↓ | % |Jac|≤0 ↓ | #Params | OPs (G) | Est. Energy |
|--------|--------|--------|------------|---------|---------|-------------|
| ANTs SyN | 0.749 | - | 0.0% | - | - | - |
| VoxelMorph | 0.740 | 4.8 | 0.3% | 300K | 45G | ~450 mJ |
| TransMorph | 0.758 | 4.5 | 0.4% | 46M | 150G | ~1500 mJ |
| SpikeReg-ANN | 0.73x | - | - | - | - | - |
| SpikeReg-SNN (T=4)| 0.71x | - | - | - | xG ACs | ~y mJ |
| SpikeReg-SNN (T=8)| 0.72x | - | - | - | xG ACs | ~y mJ |

**Your target**: SNN Dice within 2-3% of VoxelMorph. This is realistic and publishable.

#### Step 4.2: Proper Energy Estimation

Do NOT fabricate numbers. Instead, compute energy analytically:

```
E_SNN = Σ_layers (spike_rate × #neurons × #synapses × E_AC)
E_ANN = Σ_layers (#neurons × #synapses × E_MAC)
```

Where:
- `E_AC` (accumulate, spike-triggered) ≈ 0.9 pJ at 45nm (from published hardware specs)
- `E_MAC` (multiply-accumulate, standard) ≈ 4.6 pJ at 45nm

The energy ratio is: `E_SNN/E_ANN = avg_spike_rate × (E_AC / E_MAC)` ≈ `spike_rate × 0.2`

If your average spike rate is 0.1, you get **50× energy reduction**. This is a legitimate, conservative claim.

Cite: Horowitz, "Computing's Energy Problem," ISSCC 2014 for the energy-per-operation numbers.

#### Step 4.3: Ablation Studies

Run and report:
1. **ANN vs SNN**: Same architecture, with and without spiking neurons
2. **Effect of T**: T=1,2,4,8,16 time steps → Dice and energy tradeoff curve
3. **Effect of spike regularization**: λ_spike = 0, 1e-4, 1e-3, 1e-2
4. **Threshold calibration**: With vs without calibration
5. **Conversion vs direct training**: ANN→SNN conversion vs training SNN from scratch

### PHASE 5: Consider Dataset Upgrade (Week 8-10, parallel)

#### Option A: Stick with OASIS (Simpler, Sufficient for a First Paper)

- Use neurite-oasis preprocessing (FreeSurfer-processed, 35 ROI labels available)
- Standard 414 subjects, 80/20 split
- This is what VoxelMorph originally used — direct comparison is easy
- Sufficient for MICCAI or TMI

#### Option B: Upgrade to LUMIR/OpenBHB (Stronger, More Current)

- 3,384 subjects, 160×224×192 at 1mm isotropic
- This is the current L2R benchmark standard
- Makes the paper more competitive for Learn2Reg comparison
- But requires more compute and preprocessing work

**Recommendation**: Start with OASIS (Option A). If results are good, add LUMIR experiments in revision or follow-up.

### PHASE 6: Paper Writing (Week 10-12)

#### Suggested Paper Structure

**Title**: "SpikeReg: Energy-Efficient Deformable Medical Image Registration via Spiking Neural Networks"

**Abstract**: First SNN for deformable registration, ANN-to-SNN conversion with calibration, accuracy within X% of VoxelMorph at Y× lower estimated energy.

1. **Introduction**: Registration is critical → deep learning methods are accurate but energy-hungry → SNNs promise energy efficiency → no prior work on SNNs for registration → we bridge this gap
2. **Related Work**: (a) Deep registration (VoxelMorph, TransMorph, etc.), (b) SNNs for medical imaging (classification, segmentation only), (c) ANN-to-SNN conversion
3. **Method**: (a) Architecture (Spiking U-Net), (b) ANN pretraining, (c) Conversion with threshold calibration, (d) SNN fine-tuning with surrogate gradients, (e) Energy estimation model
4. **Experiments**: (a) Dataset, (b) Baselines, (c) Main results table, (d) Ablation studies, (e) Energy analysis, (f) Qualitative results (deformation fields, warped images)
5. **Discussion**: Limitations (no actual hardware deployment, inference speed on GPU is slower), future work (Loihi deployment, multi-modal registration)
6. **Conclusion**

---

## PART 4: PRIORITY-ORDERED TODO LIST

### Must-Do (No paper without these)

- [ ] Fix the spatial transformer — verify with synthetic test
- [ ] Switch to full-volume (128³) training, remove patch-based training loop
- [ ] Replace MSE pretrain loss with NCC + diffusion reg
- [ ] Train ANN baseline to competitive Dice (~0.74 on OASIS)
- [ ] Implement threshold calibration for ANN→SNN conversion
- [ ] Remove Poisson encoding from training (use direct current injection)
- [ ] Reduce time steps to T=4-8
- [ ] Remove iterative loop (single-pass first)
- [ ] Implement standard eval: Dice (35 ROIs), % negative Jacobian
- [ ] Run baselines: VoxelMorph, TransMorph, ANTs
- [ ] Compute analytical energy estimates (not fake numbers)
- [ ] Run ablation studies

### Should-Do (Strengthens the paper significantly)

- [ ] Add diffeomorphic variant (integrate velocity field via scaling & squaring — your `DiffeomorphicTransformer` already exists but isn't used)
- [ ] Add cascaded registration (2-3 iterations, not 10)
- [ ] Report TRE if landmark annotations are available
- [ ] Report HD95
- [ ] Visualize deformation fields (grid warping visualization)
- [ ] Add LUMIR benchmark results

### Nice-to-Have (Impressive but not required)

- [ ] Actual Loihi 2 deployment (requires Intel NRC membership)
- [ ] Multi-modal registration (CT-MR)
- [ ] Comparison with knowledge distillation / pruning as alternative efficiency methods
- [ ] Real latency measurements on neuromorphic simulator (Lava framework)

---

## PART 5: KEY METRICS TO HIT

| Metric | Target | Why |
|--------|--------|-----|
| ANN Dice (OASIS, 35 ROIs) | ≥ 0.740 | Must match VoxelMorph |
| SNN Dice (T=4) | ≥ 0.710 | Within 3% of ANN = publishable |
| SNN Dice (T=8) | ≥ 0.720 | Closer to ANN, higher energy |
| % negative Jacobian | < 1% | Deformation must be smooth |
| Avg spike rate | 0.05 - 0.15 | Sweet spot for energy + accuracy |
| Estimated energy ratio | 5-50× lower | Main selling point |

---

## PART 6: WHAT MAKES THIS PUBLISHABLE

The novelty is **clear and defensible**:

1. **No prior work** on SNNs for deformable registration (verified via literature search — all SNN medical imaging work is classification or segmentation)
2. **Practical pipeline**: ANN pretraining → calibrated conversion → surrogate-gradient fine-tuning for 3D dense prediction
3. **Energy-accuracy tradeoff**: A Pareto curve showing you can trade 2-3% Dice for 10-50× energy reduction
4. **Reproducibility**: Clean codebase, standard dataset, standard metrics

**What NOT to claim**: Don't claim SOTA accuracy. Don't claim actual hardware measurements you haven't done. Don't claim real-time performance without measuring it. Be honest about the accuracy gap and position the contribution as opening a new efficiency frontier for registration.

---

## SUMMARY: THE HONEST BOTTOM LINE

Your project has a **genuinely novel and timely idea** (SNN for registration) trapped inside a **broken implementation** (wrong loss, wrong encoding, wrong patch size, no calibration, fake energy numbers). The code structure is reasonable — it's not a rewrite-from-scratch situation. But every design choice needs to be reconsidered based on what actually works in both the registration and SNN literatures.

If you follow this roadmap, you have a realistic shot at a MICCAI or TMI paper within 3-4 months. The key insight: your contribution is NOT "beating TransMorph on Dice" — it's "achieving reasonable registration accuracy at dramatically lower energy cost using spiking neural networks, for the first time."
