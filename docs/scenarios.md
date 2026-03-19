# Policy Pack Scenarios

Scenarios are short, reproducible training runs that demonstrate each policy pack rule in action. Each scenario is a self-contained project that follows the same integration pattern as real external projects.

## Running Scenarios

### CLI

```bash
hotcb scenario list                          # list all 12 scenarios
hotcb scenario run stability_nan             # headless run + verify
hotcb scenario run stability_nan --dashboard # with live dashboard
hotcb scenario run --all                     # run all headless
hotcb scenario run --pack stability_basics   # run all for a pack
```

### Dashboard mode

```bash
hotcb demo --scenario stability_nan          # live dashboard with cyan annotations
```

### Programmatic

```python
from hotcb.scenarios import get
from hotcb.scenarios.runner import ScenarioRunner

config = get("stability_nan")
runner = ScenarioRunner(step_delay=0)
result = runner.run(config)
print(result.passed, result.rules_fired)
```

## Scenario Catalog

### Stability Pack (`stability_basics`)

| Scenario | Trigger | Rule | Steps |
|----------|---------|------|-------|
| `stability_nan` | `nan_detected > 0` at step 30 | `nan_guard` | 100 |
| `stability_spike` | `grad_norm > 10` at steps 20-25 | `grad_spike_clip` | 100 |
| `stability_divergence` | Loss increases >0.5 over 5 steps | `loss_spike_recovery` | 100 |

### Multi-Loss Pack (`multi_loss_assist`)

| Scenario | Trigger | Rule | Steps |
|----------|---------|------|-------|
| `multi_loss_dominance` | `aux_loss > 3 * train_loss` at step ~50 | `loss_ratio_target` | 150 |
| `multi_loss_instability` | `aux_loss > 10` at step ~40 | `aux_instability_rollback` | 100 |
| `multi_loss_warmup` | `step < 100` | `aux_warmup_ramp` | 120 |

### Distillation Pack (`distillation_assist`)

| Scenario | Trigger | Rule | Steps |
|----------|---------|------|-------|
| `distill_warmup` | `step < 200` | `summary_first_warmup` | 250 |
| `distill_divergence` | `distill_loss > 20` at step ~60 | `temperature_guard` | 150 |

### Plateau Pack (`plateau_recovery`)

| Scenario | Trigger | Rule | Steps |
|----------|---------|------|-------|
| `plateau_stagnation` | Loss range < 0.002 for 25 steps | `stagnation_detect` | 150 |
| `plateau_restart` | Val loss range < 0.003 for 30 steps | `cosine_restart` | 200 |

### Finish Strong Pack (`finish_strong`)

| Scenario | Trigger | Rule | Steps |
|----------|---------|------|-------|
| `finish_lr_reduction` | `step > 900` (starts at step 850) | `mutation_lockdown` | 150 |
| `finish_checkpoint` | `val_loss < 0.01` at step ~60 | `best_checkpoint` | 100 |

## Dashboard Annotations

When viewing scenarios in dashboard mode, autopilot-fired rules appear as **cyan** vertical annotations on the chart, labeled with the rule name (e.g., "AP: nan_guard"). Manual mutations appear in the standard orange.

## Writing Your Own Scenarios

Each scenario is a directory under `scenarios/` with three files:

### `scenario.yaml`
```yaml
name: my_scenario
pack: stability_basics
description: "Short description"
max_steps: 100
key_metric: train_loss
framework: bare
expected_rules:
  - "stability_basics.nan_guard"
seed: 42
```

### `train.py`
```python
def train_fn(run_dir, max_steps=100, step_delay=0.05, stop_event=None):
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, mutable_state

    # ... set up kernel, metrics, actuators ...
    for step in range(1, max_steps + 1):
        # ... compute metrics ...
        env = {
            "framework": "synthetic", "phase": "train", "step": step,
            "optimizer": opt,
            "metrics": {"train_loss": loss, "lr": lr, "step": step},
            "log": lambda s: None,
        }
        kernel.apply(env, events=["train_step_end"])
```

### `README.md`
Describes what the scenario demonstrates and which rule it triggers.

## Build Verification

```bash
python tests/build_test_scenarios.py           # run all, exit 0/1
python tests/build_test_scenarios.py --pack stability_basics  # filter by pack
```
