# Policy Pack Improvement Results

hotcb ships 5 policy packs with 12 rules total. Each rule has a validated scenario test demonstrating correct firing and recovery behavior.

This page documents the evaluation conditions, what each policy pack detects and corrects, and how to replicate the results.

---

## Policy Pack Overview

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #162032 100%);">
<h3 style="margin-top: 0;">stability_basics</h3>
<p style="color: #aaa; font-size: 0.9rem;">Core stability interventions — NaN guard, loss spike recovery, gradient spike clipping, emergency LR floor.</p>
<p><strong>4 rules</strong> | Critical/High confidence</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #162032 100%);">
<h3 style="margin-top: 0;">multi_loss_assist</h3>
<p style="color: #aaa; font-size: 0.9rem;">Multi-task loss balancing — dominance detection, warmup scheduling, conflict resolution.</p>
<p><strong>2 rules</strong> | Medium confidence</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #162032 100%);">
<h3 style="margin-top: 0;">plateau_recovery</h3>
<p style="color: #aaa; font-size: 0.9rem;">Stagnation detection — LR boost on extended plateau.</p>
<p><strong>1 rule</strong> | Medium confidence</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #162032 100%);">
<h3 style="margin-top: 0;">distillation_assist</h3>
<p style="color: #aaa; font-size: 0.9rem;">Knowledge distillation support — teacher-student divergence detection, temperature adjustments.</p>
<p><strong>2 rules</strong> | Medium confidence</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #162032 100%);">
<h3 style="margin-top: 0;">finish_strong</h3>
<p style="color: #aaa; font-size: 0.9rem;">Late-training optimization — LR reduction near end, best checkpoint saving, final tuning.</p>
<p><strong>3 rules</strong> | Medium confidence</p>
</div>

</div>

---

## Autopilot Recovery Results

Evaluation conditions demonstrate autopilot recovery from deliberately bad hyperparameters:

### MNIST Recovery (14K CNN, Adam)

| Condition | Problem | Autopilot | Policy Packs | Outcome |
|-----------|---------|:---------:|:------------:|---------|
| `mnist_baseline` | None | off | — | 98%+ val_accuracy (reference) |
| `mnist_high_lr_auto` | LR 30x too high (0.03) | auto | stability + plateau | grad_spike_clip fires, LR halved |
| `mnist_high_lr_no_auto` | LR 30x too high (0.03) | off | — | Grad spikes persist, worse convergence |
| `mnist_divergent_lr_auto` | LR 50x too high (0.05) | auto | stability | lr_emergency_floor fires, LR crushed |
| `mnist_high_wd_auto` | WD 1000x too high (0.1) | auto | stability + plateau | Plateau rules detect stagnation |

### CIFAR-10 Recovery (62K CNN, SGD)

| Condition | Problem | Autopilot | Policy Packs | Outcome |
|-----------|---------|:---------:|:------------:|---------|
| `cifar10_baseline` | None | off | — | 65%+ val_accuracy (reference) |
| `cifar10_high_lr_auto` | LR 15x too high (0.15) | auto | stability + plateau | grad_spike_clip fires, stabilizes |
| `cifar10_high_lr_no_auto` | LR 15x too high (0.15) | off | — | Worse convergence (control) |
| `cifar10_divergent_lr_auto` | LR 20x too high (0.2) | auto | stability | Multi-rule recovery |
| `cifar10_high_wd_auto` | WD 100x too high (0.01) | auto | stability + plateau | Plateau rules compensate |

??? tip "Replicate these conditions"

    ```bash
    # Run a single condition
    python -c "
    from hotcb.eval import EvalHarness, MNIST_CONDITIONS
    h = EvalHarness('runs/eval_mnist')
    r = h.run(MNIST_CONDITIONS[1])  # mnist_high_lr_auto
    print(f'{r.condition_name}: {r.final_metrics}')
    print(f'Autopilot actions: {len(r.autopilot_actions)}')
    "
    ```

---

## Rule Catalog

??? info "stability_basics — 4 rules (click to expand)"

    | Rule ID | Condition | Action | Confidence | Fires When |
    |---------|-----------|--------|:----------:|------------|
    | `nan_guard` | `nan_detected > 0` | Halve LR, enable safe mode | critical | Any NaN in metrics |
    | `loss_spike_recovery` | Loss > 3x EMA | Halve LR | high | Sudden loss explosion |
    | `grad_spike_clip` | grad_norm > 10 | Halve LR | high | Gradient norm spike |
    | `lr_emergency_floor` | Loss > 100 | Set LR to 1e-5 | critical | Near-divergence |

??? info "multi_loss_assist — 2 rules"

    | Rule ID | Condition | Action | Confidence |
    |---------|-----------|--------|:----------:|
    | `loss_dominance_rebalance` | One loss >5x others | Reduce dominant weight | medium |
    | `multi_loss_warmup` | Early steps + high conflict | Reduce aux weights | medium |

??? info "plateau_recovery — 1 rule"

    | Rule ID | Condition | Action | Confidence |
    |---------|-----------|--------|:----------:|
    | `plateau_lr_boost` | Loss plateau >50 steps, <1% change | Increase LR 2x | medium |

??? info "distillation_assist — 2 rules"

    | Rule ID | Condition | Action | Confidence |
    |---------|-----------|--------|:----------:|
    | `distill_divergence` | Student diverges from teacher | Reduce distill weight | medium |
    | `distill_temperature` | KL too high | Lower temperature | medium |

??? info "finish_strong — 3 rules"

    | Rule ID | Condition | Action | Confidence |
    |---------|-----------|--------|:----------:|
    | `finish_lr_reduction` | Near end + high LR | Reduce LR by 50% | medium |
    | `finish_checkpoint` | Val metric improves | Save best checkpoint | medium |
    | `finish_grad_clip` | Late + noisy gradients | Tighten grad clip | medium |

---

## Scenario Validation

Each rule has a dedicated scenario test — a short training run designed to trigger that specific rule:

```bash
# List all scenarios
hotcb scenario list

# Run a specific scenario
hotcb scenario run stability_nan

# Run all scenarios
hotcb scenario run_all
```

| Scenario | Pack | Expected Rule | Validates |
|----------|------|:------------:|-----------|
| `stability_nan` | stability_basics | nan_guard | NaN injection → LR halved |
| `stability_loss_spike` | stability_basics | loss_spike_recovery | Loss explosion → LR cut |
| `stability_grad_spike` | stability_basics | grad_spike_clip | Gradient spike → LR halved |
| `stability_emergency` | stability_basics | lr_emergency_floor | Extreme loss → LR floor |
| `multi_loss_dominance` | multi_loss_assist | loss_dominance_rebalance | Weight imbalance → rebalance |
| `multi_loss_warmup` | multi_loss_assist | multi_loss_warmup | Early conflict → warmup |
| `plateau_recovery` | plateau_recovery | plateau_lr_boost | Stagnation → LR boost |
| `distill_divergence` | distillation_assist | distill_divergence | Divergence → weight reduce |
| `distill_temperature` | distillation_assist | distill_temperature | High KL → temp adjust |
| `finish_lr_reduction` | finish_strong | finish_lr_reduction | Late high LR → reduce |
| `finish_checkpoint` | finish_strong | finish_checkpoint | Improved val → checkpoint |
| `finish_grad_clip` | finish_strong | finish_grad_clip | Late noise → clip |

---

## Combined: Autopilot + Continuation

The most powerful workflow combines autopilot recovery with continuation tuning:

1. **Train with autopilot** — catches instability, plateaus, and divergence during training
2. **Continue from checkpoints** — squeezes extra performance from the converged result

```bash
# Step 1: Train with autopilot
hotcb continue baseline --task cifar10 --out runs/cifar10_auto

# Step 2: Continue with recipes
hotcb continue run --task cifar10 --run runs/cifar10_auto \
  --metric val_accuracy --mode max \
  --autopilot auto --packs stability_basics finish_strong
```

<div style="border: 2px dashed #555; border-radius: 8px; padding: 2rem; margin: 1.5rem 0; text-align: center; color: #888;">
<strong>Autopilot + Continuation Pipeline Diagram</strong><br>
<em>Placeholder: Flowchart showing Train → Autopilot → Checkpoint → Branch → Compare → Promote</em>
</div>

---

## Evaluation Harness

Run systematic evaluations programmatically:

```python
from hotcb.eval import EvalHarness, EvalReport, MNIST_CONDITIONS, CIFAR10_CONDITIONS

harness = EvalHarness("runs/eval_suite")

# Run all MNIST conditions
results = harness.run_all(MNIST_CONDITIONS)

# Generate comparison report
report = EvalReport(results)
print(report.full_report())
print(report.autopilot_summary())
print(report.research_summary())
```

??? info "Available condition sets"

    | Set | Count | Tasks |
    |-----|:-----:|-------|
    | `GOLDEN_CONDITIONS` | 16 | Multi-task synthetic (golden demo) |
    | `FINETUNE_CONDITIONS` | 4 | Transfer learning synthetic |
    | `SIMPLE_CONDITIONS` | 3 | Single-task synthetic |
    | `MNIST_CONDITIONS` | 5 | Real MNIST CNN |
    | `CIFAR10_CONDITIONS` | 5 | Real CIFAR-10 CNN |
    | `MNIST_CONTINUATION_CONDITIONS` | 10 | MNIST late-stage continuation |
    | `CIFAR10_CONTINUATION_CONDITIONS` | 10 | CIFAR-10 late-stage continuation |
    | `ALL_CONDITIONS` | 10 | All real conditions (default) |
    | `ALL_WITH_SYNTHETIC` | 38 | Everything |
