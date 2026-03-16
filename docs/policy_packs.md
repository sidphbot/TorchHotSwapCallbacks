# Policy Pack Reference

Policy packs are reusable YAML bundles of autopilot rules. Each pack targets a specific training scenario. Packs are loaded and unloaded at runtime via the API or CLI.

---

## Quick Start

```bash
# Start dashboard with autopilot in suggest mode
hotcb serve --dir runs/exp1 --autopilot suggest

# Load a policy pack via CLI
hotcb --dir runs/exp1 set autopilot.pack=stability_basics

# Or via the REST API
curl -X POST http://localhost:8421/api/autopilot/pack/load \
  -H "Content-Type: application/json" \
  -d '{"pack": "stability_basics"}'
```

---

## Pack Catalog

### stability_basics

Core stability heuristics for any training run. This is the recommended starting pack.

| Rule | Condition | Action | Confidence | Priority |
|------|-----------|--------|------------|----------|
| `nan_guard` | `train_loss` is NaN or Inf | LR x 0.5 | critical | critical |
| `grad_spike_clip` | `grad_norm > 10.0` | LR x 0.5 | high | high |
| `lr_emergency_floor` | `train_loss > 100.0` | LR x 0.1 | critical | critical |
| `loss_spike_recovery` | Loss diverges by >0.5 in 5 steps | LR x 0.3 | high | high |

Notable behavior:
- `lr_emergency_floor` suppresses `grad_spike_clip` when it fires (the emergency floor is more aggressive, so the gentler clip is redundant)
- `nan_guard` has a 50-step cooldown to avoid repeated firing on persistent NaN

### multi_loss_assist

Helpers for multi-task training with auxiliary losses. Assumes metrics named `aux_loss`, `train_loss`, and a `conflict_score` metric.

| Rule | Condition | Action | Confidence | Priority |
|------|-----------|--------|------------|----------|
| `aux_conflict_reduce` | `conflict_score > 0.7` | Set aux_weight to 0.5 | medium | medium |
| `loss_ratio_target` | `aux_loss > 3 * train_loss` | Set aux_weight to 0.3 | medium | medium |
| `aux_instability_rollback` | `aux_loss > 10.0` | Set aux_weight to 0.1 | high | high |
| `aux_warmup_ramp` | `step < 100` | Set aux_weight to 0.1 | low | low |

Notable behavior:
- `aux_conflict_reduce` includes `rollback_if: no_improvement_after: 200` -- if reducing aux weight does not improve metrics within 200 steps, the action is considered for rollback
- Rules target loss module parameters, so your training must expose `aux_weight` as a mutable state key

### distillation_assist

Policy pack for knowledge distillation training. Manages the balance between distillation and task-specific losses.

| Rule | Condition | Action | Confidence | Priority |
|------|-----------|--------|------------|----------|
| `summary_first_warmup` | `step < 200` | Set distill_weight to 0.9 | medium | medium |
| `spatial_delayed_ramp` | `step > 500` | Set spatial_weight to 0.5 | low | low |
| `feature_health_check` | `feature_loss > 5.0` | LR x 0.5 | high | high |
| `temperature_guard` | `distill_loss > 20.0` | Set distill_weight to 0.3 | high | high |

Notable behavior:
- `feature_health_check` includes `rollback_if: no_improvement_after: 300`
- Warmup and ramp rules are step-based (fire once due to cooldown), not repeated

### plateau_recovery

Strategies for recovering from training plateaus. Uses the `plateau` condition type for automatic stagnation detection.

| Rule | Condition | Action | Confidence | Priority |
|------|-----------|--------|------------|----------|
| `stagnation_detect` | `train_loss` plateaus for 25 steps (epsilon 0.002) | LR x 0.5 | high | high |
| `cosine_restart` | `val_loss` plateaus for 30 steps (epsilon 0.003) | LR x 2.0 | medium | medium |
| `aux_emphasis_shift` | `train_loss` plateaus for 20 steps (epsilon 0.001) | Set aux_weight to 0.8 | low | low |
| `conservative_finish` | `val_loss` plateaus for 40 steps (epsilon 0.001) | LR x 0.1 | high | high |

Notable behavior:
- `cosine_restart` has `bounds: {max: 0.01}` -- the warm restart will not push LR above 0.01 regardless of the 2x multiplier
- `cosine_restart` includes `rollback_if: no_improvement_after: 100`
- `conservative_finish` suppresses `cosine_restart` when it fires (late-stage conservative reduction overrides warm restarts)

### finish_strong

Late-stage training policies for optimal convergence. Step-based rules that activate in the final phase of training.

| Rule | Condition | Action | Confidence | Priority |
|------|-----------|--------|------------|----------|
| `enable_swa_late` | `step > 800` | Enable SWA callback | medium | medium |
| `enable_ema_late` | `step > 700` | Enable EMA callback | medium | medium |
| `mutation_lockdown` | `step > 900` | LR x 0.1 | high | high |
| `best_checkpoint` | `val_loss < 0.01` | Enable checkpoint callback | low | low |

Notable behavior:
- SWA and EMA rules assume you have registered callbacks with IDs `swa` and `ema`
- `best_checkpoint` is metric-based (not step-based) -- fires when val_loss drops below threshold
- All rules have high cooldowns (200-500 steps) to fire at most once

---

## YAML DSL Reference

Each policy pack is a YAML file with this structure:

```yaml
name: my_pack                      # Display name
description: What this pack does   # Human-readable description
version: "1.0"                     # Semantic version
requires: []                       # Other packs this depends on (informational)

rules:
  - id: my_rule_id                 # Unique within the pack (prefixed with pack name at load)
    condition: plateau             # Condition type (see below)
    metric: train_loss             # Metric name to evaluate
    params:                        # Condition-specific parameters
      window: 25
      epsilon: 0.002
    action:                        # Command to execute when condition fires
      module: opt
      op: set_params
      params:
        lr_mult: 0.5               # Use lr_mult for relative changes
    confidence: high               # "low", "medium", "high", "critical"
    priority: high                 # "low", "medium", "high", "critical"
    cooldown: 50                   # Min steps between firings of this rule
    description: "Human-readable"  # Shown in dashboard
    bounds:                        # Optional: absolute limits on action params
      min: 0.0001
      max: 0.01
    rollback_if:                   # Optional: auto-rollback conditions
      no_improvement_after: 200
    suppress_rules:                # Optional: rule IDs to suppress when this fires
      - "pack_name.other_rule"
```

### Condition Types

#### `plateau`

Detects metric stagnation over a sliding window.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `window` | int | 5 | Number of recent values to examine |
| `epsilon` | float | 0.001 | Max range (max - min) to qualify as plateau |

Fires when: `max(recent) - min(recent) <= epsilon`

#### `divergence`

Detects sharp metric increase.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `window` | int | 10 | Number of steps to look back |
| `threshold` | float | 2.0 | Minimum increase to trigger (falls back to `AutopilotConfig.divergence_threshold`) |

Fires when: `metric[-1] - metric[-window] > threshold`

#### `overfitting`

Checks train/val loss ratio.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ratio_threshold` | float | 0.5 | Train/val ratio below this triggers (falls back to `AutopilotConfig.ratio_threshold`) |
| `train_metric` | str | `"train_loss"` | Train metric name (also tries `"train/loss"`) |
| `val_metric` | str | `"val_loss"` | Val metric name (also tries `"val/loss"`) |

Fires when: `train_loss / val_loss < ratio_threshold`

#### `custom`

Evaluates a Python expression against current metric values.

| Param | Type | Description |
|-------|------|-------------|
| `expression` | str | Python expression using metric names as variables |

The expression runs in a restricted namespace containing:
- All current metric values as variables (with `/`, `.`, `-` replaced by `_`)
- `abs`, `min`, `max`, `math` builtins
- No other builtins (safe eval)

Examples:
```yaml
expression: "train_loss > 100.0"
expression: "grad_norm > 10.0"
expression: "math.isnan(train_loss) or math.isinf(train_loss)"
expression: "aux_loss > 3 * train_loss"
expression: "step > 800"
```

If a metric referenced in the expression is not yet available, the condition silently returns false (no error).

### Action Parameters

Actions are written to `hotcb.commands.jsonl` and processed by HotKernel.

**Multiplier params**: Use `lr_mult` and `wd_mult` for relative changes. The engine resolves these to absolute values using the latest metric history at apply time.

```yaml
# Relative: halve current LR
action:
  module: opt
  op: set_params
  params:
    lr_mult: 0.5

# Absolute: set LR to specific value
action:
  module: opt
  op: set_params
  params:
    lr: 0.0001

# Loss weight
action:
  module: loss
  op: set_params
  params:
    aux_weight: 0.3

# Callback control
action:
  module: cb
  op: enable
  params:
    id: swa
```

### Priority and Conflict Resolution

When multiple rules fire in the same evaluation cycle:

1. Rules are sorted by priority: critical > high > medium > low
2. Higher-priority rules build a suppress set from their `suppress_rules` lists
3. For rules targeting the same actuator (same `module:op:param_keys`), only the highest-priority rule fires
4. Suppressed rules are skipped entirely

### Bounds and Rollback

**bounds**: Clamp action parameters to absolute limits before applying. This prevents rules with multipliers from pushing values outside safe ranges.

```yaml
bounds:
  min: 0.0001    # LR won't go below this
  max: 0.01      # LR won't go above this
```

**rollback_if**: Declare conditions under which the action should be considered for rollback.

```yaml
rollback_if:
  no_improvement_after: 200    # Steps to wait for improvement
```

---

## Custom Rule Authoring

### Writing a Custom Pack

Create a YAML file following the DSL format above. Place it in `src/hotcb/server/guidelines/` to make it discoverable via `list_packs()`, or load it from any path via `load_guidelines()`.

```yaml
name: my_custom_pack
description: Custom rules for my training setup
version: "1.0"
requires:
  - stability_basics    # Informational: recommend loading this first

rules:
  - id: my_custom_rule
    condition: custom
    metric: my_metric
    params:
      expression: "my_metric > threshold_value"
    action:
      module: opt
      op: set_params
      params:
        lr_mult: 0.5
    confidence: medium
    priority: medium
    cooldown: 30
    description: "My custom intervention"
```

### Tips for Custom Rules

1. **Start with `suggest` mode** to observe what your rules propose before enabling auto-apply
2. **Set appropriate cooldowns** -- too low causes oscillation, too high misses problems
3. **Use `suppress_rules`** to prevent conflicting rules from firing together
4. **Add `bounds`** to multiplier-based rules to prevent extreme values
5. **Add `rollback_if`** for medium-confidence rules so bad interventions are reversible
6. **Use `custom` conditions** for complex logic; `plateau` and `divergence` for standard patterns
7. **Test with `hotcb demo`** before deploying to real training

### Loading Custom Packs at Runtime

```python
# Via AutopilotEngine
engine.load_pack("my_custom_pack")     # from guidelines/ directory
engine.load_guidelines("/path/to.yaml") # from any path

# Via REST API
POST /api/autopilot/pack/load {"pack": "my_custom_pack"}

# List available packs
GET /api/autopilot/packs
```

### Sharing Packs

Policy packs are plain YAML files. Share them by:
1. Adding to `src/hotcb/server/guidelines/` and submitting a PR
2. Distributing as standalone `.yaml` files loaded via `load_guidelines(path)`
3. Publishing as part of a pip-installable package that installs files to the guidelines directory
