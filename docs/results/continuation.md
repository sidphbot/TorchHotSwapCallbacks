# Continuation Tuning Results

Validated experiment results from late-stage continuation tuning on MNIST and CIFAR-10.

All experiments use the standard hotcb continuation routine: train a baseline with checkpoints, branch from the best-validation anchor, apply a single mutation recipe, train for 500 extra steps, compare.

---

## MNIST — Small CNN (14K params)

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1.5rem 0;">
<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0;">Baseline Configuration</h4>

| Parameter | Value |
|-----------|-------|
| **Model** | Conv(1→16)→Pool→Conv(16→32)→Pool→FC(1568→10) |
| **Parameters** | ~14,000 |
| **Optimizer** | Adam |
| **Learning rate** | 1e-3 |
| **Weight decay** | 1e-4 |
| **Batch size** | 128 |
| **Training steps** | 1500 (~3 epochs) |
| **Data** | MNIST (60K train / 10K val) |

</div>
<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0;">Baseline Results</h4>

| Metric | Value |
|--------|-------|
| **val_accuracy** | 98.66% |
| **val_loss** | 0.0381 |
| **train_loss** | 0.0125 |
| **grad_norm_ema** | 0.67 |

Converged naturally — no autopilot, no recipe.
</div>
</div>

### Continuation Leaderboard

Anchor: `best_val_anchor` (step 1250, just before best validation). 500 extra training steps per branch.

| Rank | Recipe | Resume Mode | Best val_accuracy | Delta | Delta % | Status |
|:----:|--------|-------------|:-----------------:|:-----:|:-------:|:------:|
| 1 | **lr_quarter** | full_resume | **98.88%** | +0.0022 | **+0.22%** | WON |
| 2 | **lr_half** | full_resume | 98.85% | +0.0019 | +0.19% | WON |
| 3 | **swa_tail** | full_resume | 98.85% | +0.0019 | +0.19% | WON |
| 4 | **wd_reduce** | full_resume | 98.74% | +0.0008 | +0.08% | WON |
| 5 | **grad_clip_tight** | full_resume | 98.73% | +0.0007 | +0.07% | WON |
| 6 | **ema_tail** | full_resume | 98.70% | +0.0004 | +0.04% | WON |

**Winner: lr_quarter** — quartering the LR (1e-3 → 2.5e-4) gave the most improvement, stabilizing late-training Adam oscillations.

??? info "Resume mode comparison (lr_half + swa_tail)"

    Running lr_half and swa_tail with both `full_resume` and `weights_only`:

    | Branch | Recipe | Resume | val_accuracy | Delta |
    |--------|--------|--------|:------------:|:-----:|
    | 1 | lr_half | **weights_only** | **98.85%** | +0.19% |
    | 2 | swa_tail | **weights_only** | 98.81% | +0.15% |
    | 3 | swa_tail | full_resume | 98.79% | +0.13% |
    | 4 | lr_half | full_resume | 98.77% | +0.11% |

    **Insight:** `weights_only` outperforms `full_resume` on MNIST. Fresh Adam state (zero momentum buffers) allows the optimizer to take a new direction from the converged basin — the accumulated momentum was pointing in a suboptimal direction.

<div style="border: 2px dashed #555; border-radius: 8px; padding: 2rem; margin: 1.5rem 0; text-align: center; color: #888;">
<strong>MNIST Training Curves</strong><br>
<em>Placeholder: Attach val_accuracy over steps — baseline (solid) vs branches (dashed)</em><br>
<small>Generate with: <code>python scripts/run_continuation_experiments.py --task mnist</code></small>
</div>

### Replicate

```bash
# 1. Train baseline (~3 min CPU, ~1 min GPU)
hotcb continue baseline --task mnist --out runs/mnist_baseline --ckpt-interval 250

# 2. Run continuation (~1 min per branch)
hotcb continue run --task mnist --run runs/mnist_baseline \
  --metric val_accuracy --mode max --extra-steps 500

# 3. View report
hotcb continue report --dir runs/mnist_baseline/continuation
```

---

## CIFAR-10 — Small CNN (62K params)

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1.5rem 0;">
<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0;">Baseline Configuration</h4>

| Parameter | Value |
|-----------|-------|
| **Model** | Conv(3→32,BN)→Pool→Conv(32→64,BN)→Pool→FC(4096→128)→FC(128→10) |
| **Parameters** | ~62,000 |
| **Optimizer** | SGD (momentum=0.9) |
| **Learning rate** | 0.01 |
| **Weight decay** | 1e-4 |
| **Batch size** | 128 |
| **Augmentation** | RandomCrop(32, pad=4), RandomHorizontalFlip |
| **Training steps** | 2000 (~5 epochs) |
| **Data** | CIFAR-10 (50K train / 10K val) |

</div>
<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0;">Baseline Results</h4>

| Metric | Value |
|--------|-------|
| **val_accuracy** | 68.86% |
| **val_loss** | 0.9116 |
| **train_loss** | 0.8956 |
| **grad_norm_ema** | 2.63 |

5 epochs on CIFAR-10 with this small CNN — room for continuation improvement.
</div>
</div>

### Continuation Leaderboard

Anchor: `best_val_anchor` (step 2000 — end of training was also best). 500 extra training steps per branch.

| Rank | Recipe | Resume Mode | Best val_accuracy | Delta | Delta % | Status |
|:----:|--------|-------------|:-----------------:|:-----:|:-------:|:------:|
| 1 | **lr_half** | full_resume | **73.26%** | +0.0440 | **+6.39%** | WON |
| 2 | **grad_clip_tight** | full_resume | 73.18% | +0.0432 | +6.27% | WON |
| 3 | **lr_quarter** | full_resume | 73.07% | +0.0421 | +6.11% | WON |
| 4 | **swa_tail** | full_resume | 72.77% | +0.0391 | +5.68% | WON |
| 5 | **ema_tail** | full_resume | 72.00% | +0.0314 | +4.56% | WON |
| 6 | **wd_reduce** | full_resume | 71.45% | +0.0259 | +3.76% | WON |

**Winner: lr_half** — halving the SGD learning rate (0.01 → 0.005) gave the largest improvement. The original LR was too high for late-stage convergence, causing oscillation around the minimum.

### Why CIFAR-10 improved more than MNIST

The CIFAR-10 baseline (5 epochs with a small CNN) was still in a regime where:

1. **LR was too high for the current loss landscape** — SGD at lr=0.01 was oscillating past the minimum. Halving or quartering the LR let it settle.
2. **Training hadn't fully converged** — 5 epochs is moderate for CIFAR-10. Extra steps with better hyperparameters extract real gains, not just noise.
3. **Grad clipping helped significantly** — SGD without clipping had occasional gradient spikes. Clipping to 0.5 eliminated these without suppressing useful gradients.
4. **SWA is well-matched to SGD** — the weight averaging algorithm is specifically designed for SGD-style optimizers (Izmailov et al., 2018).

MNIST, being a simpler task, was nearly fully converged in 3 epochs — improvements there are genuine but small.

<div style="border: 2px dashed #555; border-radius: 8px; padding: 2rem; margin: 1.5rem 0; text-align: center; color: #888;">
<strong>CIFAR-10 Training Curves</strong><br>
<em>Placeholder: Attach val_accuracy over steps — baseline (solid) vs top 3 branches (dashed)</em><br>
<small>Generate with: <code>python scripts/run_continuation_experiments.py --task cifar10</code></small>
</div>

### Replicate

```bash
# 1. Train baseline (~5-8 min CPU, ~2-3 min GPU)
hotcb continue baseline --task cifar10 --out runs/cifar10_baseline --ckpt-interval 250

# 2. Run continuation (~2 min per branch)
hotcb continue run --task cifar10 --run runs/cifar10_baseline \
  --metric val_accuracy --mode max --extra-steps 500

# 3. View report
hotcb continue report --dir runs/cifar10_baseline/continuation
```

---

## Recipe Effectiveness Matrix

Cross-task view of which recipes work best:

| Recipe | MNIST Delta | CIFAR-10 Delta | Mechanism | Best For |
|--------|:-----------:|:--------------:|-----------|----------|
| **lr_half** | +0.19% | **+6.39%** | Reduce oscillation amplitude | Over-high LR, SGD |
| **lr_quarter** | **+0.22%** | +6.11% | Aggressive stabilization | Noisy tail, Adam |
| **swa_tail** | +0.19% | +5.68% | Weight averaging along trajectory | SGD, long tail phase |
| **ema_tail** | +0.04% | +4.56% | Exponential shadow weights | Short continuation |
| **grad_clip_tight** | +0.07% | +6.27% | Suppress gradient spikes | SGD, unstable gradients |
| **wd_reduce** | +0.08% | +3.76% | Relax regularization | Over-regularized models |

### Key Findings

1. **LR reduction is the most reliable recipe** — works on both Adam and SGD, both tasks
2. **SWA is particularly effective for SGD** — matches published results (Izmailov 2018)
3. **Grad clipping delivers outsized gains when SGD is noisy** — nearly tied with lr_half on CIFAR-10
4. **EMA and wd_reduce are conservative** — small but reliable gains, good safety profile
5. **weights_only resume beats full_resume on MNIST** — fresh optimizer state allows new optimization paths

---

## Budget Analysis

| | MNIST | CIFAR-10 |
|--|:-----:|:--------:|
| **Baseline training time** | ~60s (GPU) | ~120s (GPU) |
| **Continuation per branch** | ~12s | ~18s |
| **Total continuation (6 branches)** | ~76s | ~106s |
| **Best improvement** | +0.22% | +6.39% |
| **Extra compute cost** | ~1.3x baseline | ~0.9x baseline |

**Cost-effectiveness:** On CIFAR-10, spending <1x the original training budget yielded a +6.4% improvement. On MNIST, the marginal gains were smaller but came at very low cost.

---

## How to Add Custom Recipes

```python
from hotcb.routines.continuation.models import MutationRecipe, MutationOp

custom_recipe = MutationRecipe(
    name="aux_off",
    description="Disable auxiliary loss in tail phase",
    ops=[
        MutationOp(target="loss.aux_weight", value=0.0),
    ],
)

# Add to config
config.recipes.append(custom_recipe)
```

---

<div style="border: 1px solid #666; border-radius: 8px; padding: 1rem 1.5rem; margin: 2rem 0; background: rgba(255, 200, 0, 0.05);">
<strong>Reproducibility note:</strong> Results vary slightly between runs due to random initialization, data shuffling, and GPU non-determinism. Set <code>seed=42</code> in conditions for maximum reproducibility. The relative ordering of recipes is consistent across runs.
</div>
