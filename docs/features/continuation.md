# Continuation Tuning

<div style="border-left: 4px solid #4ecdc4; padding: 1rem 1.5rem; margin: 1.5rem 0; background: rgba(78, 205, 196, 0.05); border-radius: 0 8px 8px 0;">
<strong>You already paid the cost to reach a good basin. Now squeeze extra performance from it without restarting from scratch.</strong>
</div>

Continuation tuning takes a converged run, branches from one or more earlier checkpoints, applies small safe mutations through hotcb, and keeps only branches that improve the validation objective.

This is **not** full HPO. It is a bounded, safe, local continuation search around already successful runs.

---

## When to Use

- Improve a converged run by 0.2–6% on val metric
- Polish multi-loss training where late auxiliary weights are suboptimal
- Recover from "good but slightly over-regularized" endings
- Test whether scheduler / optimizer inertia blocked a better endpoint
- Systematically compare late-stage recipe variants from the same anchor

## How It Works

```
Base Run (converged) → Select Anchors → Generate Branches → Train → Compare → Promote Winner
                           │                    │
                     (checkpoints)     (recipe × resume mode)
```

### 1. Train a baseline with checkpoints

```bash
hotcb continue baseline --task cifar10 --out runs/cifar10_baseline --ckpt-interval 250
```

This trains the default CIFAR-10 CNN (62K params, SGD lr=0.01) for 2000 steps, saving checkpoints every 250 steps.

### 2. Run continuation branches

```bash
hotcb continue run \
  --task cifar10 \
  --run runs/cifar10_baseline \
  --metric val_accuracy \
  --mode max \
  --extra-steps 500
```

This automatically:

1. **Selects anchors** — checkpoints near best validation and near end of training
2. **Plans branches** — crosses 6 default recipes × resume modes
3. **Runs each branch** — resumes from checkpoint, applies mutation recipe via hotcb commands
4. **Compares results** — ranks by primary metric delta, applies guardrails
5. **Reports winner** — with full leaderboard and recipe rankings

---

## The 6 Default Recipes

Each recipe is a small, safe, late-stage mutation:

| Recipe | What it does | Why it works |
|--------|-------------|-------------|
| **lr_half** | Halve learning rate | Reduces late-training oscillations, allows tighter convergence |
| **lr_quarter** | Quarter learning rate | More aggressive stabilization — useful for noisy SGD |
| **swa_tail** | Enable Stochastic Weight Averaging | Averages weights along trajectory — smoother loss landscape (Izmailov 2018) |
| **ema_tail** | Enable Exponential Moving Average | Shadow weights with decay=0.999 — less aggressive than SWA |
| **grad_clip_tight** | Tighten gradient clipping to 0.5–1.0 | Suppresses late gradient spikes without losing signal |
| **wd_reduce** | Halve weight decay | Relaxes regularization pressure — lets model fit tighter in tail |

??? info "Recipe details — mutation ops"

    Each recipe translates to hotcb commands written to `hotcb.commands.jsonl`:

    ```json
    {"module": "opt", "op": "set_params", "params": {"lr": 0.005}, "source": "continuation_recipe"}
    ```

    Relative mutations (e.g., `value: 0.5, relative: True`) resolve against the base run's final metrics.
    SWA/EMA enable commands go to `custom` module actuators.

---

## Resume Modes

How state is restored from a checkpoint determines what the continuation can explore:

| Mode | Loads | Use when |
|------|-------|----------|
| **full_resume** | Model + optimizer + scheduler | True continuation — preserves momentum, step count |
| **reset_scheduler** | Model + optimizer | Trajectory is good but LR schedule became too conservative |
| **weights_only** | Model weights only | Escape optimizer inertia — fresh momentum for new direction |

??? tip "When to use weights_only"

    On MNIST, `weights_only` outperformed `full_resume` for both `lr_half` and `swa_tail`.
    Fresh Adam momentum allowed the model to take a new optimization path from the
    converged basin, finding a slightly better minimum.

    This is particularly useful when the original optimizer accumulated momentum
    that points away from the best direction at the current loss landscape position.

---

## Anchor Selection

Anchors are checkpoints from which branches start. The system supports:

**Automatic selection** (default):

- **best_val_anchor** — checkpoint near best validation metric (typically the best checkpoint or one step before)
- **end_minus_anchor** — checkpoint near end of training (~80% mark)

**Manual selection:**

```bash
# Specify exact checkpoint paths
hotcb continue run --task mnist --run runs/mnist_baseline --anchors runs/mnist_baseline/checkpoints/step_001000.pt
```

??? info "Plateau detection"

    `AnchorSelector.detect_plateau_checkpoint()` scans the metric history for the first window
    where improvement drops below 1% relative — the anchor is placed just before that point.
    Useful for identifying where the training stopped making progress.

---

## Guardrails

Branches are rejected if they violate safety constraints:

- **NaN/Inf detection** — any NaN or Inf in metrics marks branch as `UNSTABLE`
- **Secondary metric regression** — configurable max regression on protected metrics
- **Acceptance threshold** — `min_improvement_delta` must be met to count as `WON`

```python
from hotcb.routines.continuation import GuardrailSpec

guardrails = GuardrailSpec(
    forbid_nan=True,
    secondary_metrics={
        "val_loss": {"mode": "min", "max_regression": 0.01},
    },
)
```

---

## Branch Statuses

Each branch ends with a definitive status:

| Status | Meaning |
|--------|---------|
| `won` | Improved primary metric beyond threshold, no guardrail violations |
| `lost` | Did not improve enough |
| `unstable` | NaN/Inf detected during training |
| `inconclusive` | Delta within noise threshold (<1%) |
| `rejected_guardrail` | Improved primary but violated a secondary metric constraint |
| `budget_exhausted` | Ran out of extra steps without clear outcome |
| `resume_failed` | Checkpoint couldn't be loaded or training crashed |

---

## Programmatic API

```python
from hotcb.routines.continuation import (
    ContinuationConfig, ObjectiveSpec, BudgetSpec,
    ContinuationPlanner, ContinuationLauncher, ContinuationReport,
    AnchorSelector, ResumeMode,
)

# Select anchors from a converged run
selector = AnchorSelector("runs/cifar10_baseline")
anchors = selector.select_auto(max_anchors=2, primary_metric="val_accuracy", mode="max")

# Configure
config = ContinuationConfig(
    base_run_dir="runs/cifar10_baseline",
    task="cifar10",
    objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
    budget=BudgetSpec(max_extra_steps=500, branches_per_anchor=6),
    anchors=anchors,
    resume_modes=[ResumeMode.FULL_RESUME],
)

# Plan and run
planner = ContinuationPlanner(config)
branches = planner.plan_with_defaults()  # uses 6 default recipes

launcher = ContinuationLauncher(config)
result = launcher.run_all(branches)

# Report
report = ContinuationReport(result)
print(report.full_report())
report.save("runs/cifar10_continuation")
```

---

## CLI Reference

```bash
# Train baseline with checkpoints
hotcb continue baseline --task mnist --out runs/mnist_baseline --ckpt-interval 250

# Run continuation with default recipes
hotcb continue run --task mnist --run runs/mnist_baseline --metric val_accuracy --mode max

# Run with specific recipes and resume modes
hotcb continue run --task cifar10 --run runs/cifar10_baseline \
  --recipes lr_half swa_tail --resume-modes full_resume weights_only \
  --extra-steps 500 --branches-per-anchor 4

# Run with autopilot-guided continuation
hotcb continue run --task mnist --run runs/mnist_baseline \
  --autopilot auto --packs stability_basics finish_strong

# View a previous report
hotcb continue report --dir runs/mnist_continuation
```

---

## Output Artifacts

Every continuation routine produces:

```
runs/mnist_continuation/
  continuation_report.txt      # Human-readable report
  continuation_summary.json    # Machine-readable summary
  leaderboard.json             # Per-branch comparison data
  branches/
    best_val_anchor__lr_half__full_resume_0/
      branch_config.json       # Branch parameters
      hotcb.commands.jsonl     # Applied mutations
      hotcb.metrics.jsonl      # Training metrics
      hotcb.applied.jsonl      # Mutation ledger
    best_val_anchor__swa_tail__full_resume_2/
      ...
```

---

## Evaluation Conditions

20 pre-defined continuation conditions for systematic evaluation:

??? info "MNIST Conditions (10)"

    | Condition | Recipe | Resume | Hypothesis |
    |-----------|--------|--------|------------|
    | `mnist_cont_lr_half` | lr_half | full_resume | Reduced oscillation, +0.1-0.5% |
    | `mnist_cont_lr_quarter` | lr_quarter | full_resume | Strong stabilization |
    | `mnist_cont_lr_half_weights_only` | lr_half | weights_only | Fresh momentum escapes basin |
    | `mnist_cont_swa_tail` | swa_tail | full_resume | Weight averaging smooths |
    | `mnist_cont_ema_tail` | ema_tail | full_resume | Shadow weights improve generalization |
    | `mnist_cont_swa_lr_half` | swa+lr_half | full_resume | Combined smoothing |
    | `mnist_cont_wd_reduce` | wd_reduce | full_resume | Less late regularization |
    | `mnist_cont_grad_clip_tight` | grad_clip | full_resume | Suppress late spikes |
    | `mnist_cont_kitchen_sink` | lr+swa+clip | full_resume | Maximal treatment |
    | `mnist_cont_autopilot` | autopilot | full_resume | Policy-driven mutations |

??? info "CIFAR-10 Conditions (10)"

    | Condition | Recipe | Resume | Hypothesis |
    |-----------|--------|--------|------------|
    | `cifar10_cont_lr_half` | lr_half | full_resume | SGD with halved LR finds tighter min |
    | `cifar10_cont_lr_quarter` | lr_quarter | full_resume | Very stable tail convergence |
    | `cifar10_cont_lr_half_weights_only` | lr_half | weights_only | Reset SGD momentum |
    | `cifar10_cont_swa_tail` | swa_tail | full_resume | SWA well-studied for SGD (Izmailov 2018) |
    | `cifar10_cont_ema_tail` | ema_tail | full_resume | EMA smoother for short continuation |
    | `cifar10_cont_swa_lr_half` | swa+lr_half | full_resume | Canonical SWA recipe |
    | `cifar10_cont_wd_reduce` | wd_reduce | full_resume | Relax regularization in tail |
    | `cifar10_cont_grad_clip_tight` | grad_clip | full_resume | Stabilize SGD late-phase |
    | `cifar10_cont_kitchen_sink` | lr+swa+clip | full_resume | Most aggressive treatment |
    | `cifar10_cont_autopilot` | autopilot | full_resume | Policy-driven mutations |

---

## When It Works Best

**Best fit:**

- Multi-loss training (late auxiliary weights often suboptimal)
- Long converged runs (expensive early-phase basin already paid for)
- Noisy late plateau (SWA/EMA particularly effective)
- Evidence that endgame recipe is suboptimal
- Domains where tiny gains matter (competition, production deployment)

**Less useful:**

- Clearly undertrained runs (just train longer)
- Highly chaotic training with poor checkpoint quality
- Recipes with no mutable late-stage knobs

---

<div style="border: 1px solid #666; border-radius: 8px; padding: 1rem 1.5rem; margin: 2rem 0; background: rgba(255, 200, 0, 0.05);">
<strong>Warning:</strong> This routine optimizes against the validation set. Keep a clean untouched test set for final model selection and reporting. Repeated continuation search on the same validation set can overfit selection to validation behavior.
</div>
