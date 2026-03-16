# EXECUTE.md — Full Autopilot Spec Implementation Plan

> **Instructions for Claude Code**: Read this file top-to-bottom. Execute each phase
> sequentially. Within each phase, use subagents for parallelizable work.
> The user is asleep — do not ask questions. Make decisions, document them in
> `DECISIONS.md`, and keep moving. If blocked, skip and note in the log.
>
> **TDD Protocol**: For every new feature, write tests FIRST (success + failure cases),
> run them to confirm they fail, THEN implement until tests pass. Never ship code
> without passing tests.
>
> **Commit Protocol**: Commit after each completed sub-phase with a descriptive message.
> Do not batch large changes into single commits.
>
> **Decision Protocol**: Before implementing anything non-trivial, write a brief
> decision record in `.claude/plans/DECISIONS.md` with: what, why, alternatives
> considered, chosen approach. This is your reasoning log.
>
> **Error Protocol**: If tests fail after implementation, fix them. If you realize
> a design decision was wrong mid-implementation, update DECISIONS.md, adjust, continue.
> Do not revert large amounts of work — iterate forward.
>
> **Subagent Protocol**: Launch subagents for independent work. Use `isolation: "worktree"`
> for parallel code changes. Merge worktree branches when done. Use foreground agents
> for research/planning, background agents for independent implementation.

---

## Pre-flight

Before starting any phase:

1. `git status` — ensure clean working tree on `main`
2. `python3 -m pytest src/hotcb/tests/ --no-cov -q` — confirm baseline (expect 856 pass)
3. Read `CLAUDE.md` and `.claude/plans/STREAMS.md` for full context
4. Create `.claude/plans/DECISIONS.md` with header:
   ```markdown
   # Decision Log
   Records of architectural and implementation decisions made during autonomous execution.
   ```
5. Create branch: `git checkout -b autopilot-spec-v1`

---

## Phase 1: Observability v2 (Layer 1)

**Stream:** `observability-v2`
**Goal:** Enrich training state with health signals that policies and AI need.
**Files to modify:** `server/autopilot.py`, `server/ai_prompts.py`, `server/app.py`
**Files to create:** `src/hotcb/health.py`, `src/hotcb/tests/test_health.py`

### 1.1 — Plan & Document

Write a decision record covering:
- Where does `TrainingHealthState` live? (new `health.py` module — keeps autopilot.py focused on rules)
- What signals are computable from existing metrics JSONL? (grad_norm, per-loss values, total loss trend — all already logged by MetricsCollector if user logs them)
- What signals require new metric keys? (conflict_score is derived, not logged directly)
- How does the health state flow? (metrics → TrendCompressor → HealthState → autopilot rules + AI prompts)

### 1.2 — TrainingHealthState dataclass

**Tests first** (`test_health.py`):
```
test_health_state_from_empty_metrics → returns default "unknown" labels
test_health_state_from_stable_training → labels: stable-improving
test_health_state_from_diverging_training → labels: numerically-unsafe
test_health_state_from_plateau → labels: stable-plateaued
test_health_state_from_oscillating_loss → labels: oscillatory
test_health_state_nan_detection → nan_count incremented, numerically-unsafe
test_health_state_loss_spike_detection → spike flag set when loss jumps >3x EMA
test_health_state_serialization → to_dict() / from_dict() roundtrip
```

**Implementation** (`health.py`):
- `TrainingHealthState` dataclass with fields:
  - `labels: List[str]` — derived state labels
  - `numeric_stability: dict` — nan_count, inf_count, spike_count, last_spike_step
  - `optimization_health: dict` — total_loss_trend, grad_norm_trend, plateau_score, oscillation_score
  - `multi_loss: dict` — per_loss_trends, dominance_ratios, conflict_score (if multi-loss detected)
  - `step: int`, `timestamp: float`
- `compute_health_state(metric_history: dict, window: int = 100) -> TrainingHealthState`
  - Takes `S.metricsData`-style dict of `{metric_name: [{step, value}]}`
  - Computes all signals from the last `window` points
  - Classifies into labels

### 1.3 — Gradient health signals

**Tests first** (add to `test_health.py`):
```
test_grad_norm_trend_rising → optimization_health.grad_norm_trend = "rising"
test_grad_norm_trend_stable → "stable"
test_grad_norm_absent → gracefully skipped, no crash
test_clipping_rate_detection → if grad_norm frequently hits clip threshold
```

**Implementation:**
- In `compute_health_state()`, look for `grad_norm` or `grad_norm_total` in metric history
- Compute trend via linear regression (reuse `TrendCompressor` logic)
- Detect clipping rate: fraction of steps where grad_norm == clip_value (within 1%)

### 1.4 — Multi-loss conflict detection

**Tests first** (add to `test_health.py`):
```
test_conflict_score_no_multi_loss → conflict_score = 0, no crash
test_conflict_score_aligned_losses → score near 0
test_conflict_score_conflicting_losses → score > 0.5
test_dominance_ratio_balanced → ratios near 1.0
test_dominance_ratio_one_loss_dominates → ratio > 3.0 for dominant loss
test_conflict_detection_requires_minimum_history → returns 0 if < 10 points
```

**Implementation:**
- Detect multi-loss: look for metrics matching `*_loss` pattern (>1 found = multi-loss)
- Dominance ratio: `max(loss_i_grad_norm) / min(loss_j_grad_norm)` — approximate from loss magnitudes if grad norms not available: `abs(slope_i) / abs(slope_j)`
- Conflict score: correlation of loss trends — if losses move in opposite directions, score increases
  - Use detrended cross-correlation of loss series over window
  - Score 0 = aligned, 1 = fully opposed

### 1.5 — Derived state labels

**Tests first** (add to `test_health.py`):
```
test_label_stable_improving → loss decreasing steadily, no spikes
test_label_stable_plateaued → loss flat for >50 steps
test_label_aux_conflicted → multi-loss with conflict_score > 0.3
test_label_numerically_unsafe → NaN detected or loss spike
test_label_oscillatory → loss variance high relative to mean
test_label_converged_likely → loss flat + validation stable + low grad norm
test_label_collapse_risk → feature variance dropping (if available) or loss going to 0
test_multiple_labels_simultaneously → can be both oscillatory and aux-conflicted
```

**Implementation:**
- Label classifier function: `_classify_labels(health: TrainingHealthState) -> List[str]`
- Each label has a condition function, all are evaluated, multiple can be true
- Labels are ordered by severity: `numerically-unsafe` > `collapse-risk` > `aux-conflicted` > `oscillatory` > `stable-plateaued` > `stable-improving` > `converged-likely`

### 1.6 — Wire into autopilot and AI prompts

**Tests first** (add to existing `test_autopilot.py` and `test_ai_engine.py`):
```
test_autopilot_evaluate_includes_health_state → health state attached to alert context
test_ai_prompt_includes_health_labels → build_context() output contains health label section
test_ai_prompt_includes_conflict_score → if multi-loss, conflict score in prompt
test_health_endpoint_returns_state → GET /api/state/health returns JSON
```

**Implementation:**
- `AutopilotEngine.evaluate_async()` → call `compute_health_state()` on metric history, attach to evaluation context
- `ai_prompts.build_context()` → add "Training Health" section with labels and key signals
- `app.py` → add `GET /api/state/health` endpoint returning `health_state.to_dict()`

### 1.7 — Dashboard health panel

**Tests:** None needed (frontend JS, tested manually).

**Implementation:**
- `init.js` → poll `/api/state/health` every 10s
- `panels.js` or `charts.js` → update health card with label badges (colored by severity)
- CSS → health label badge styles (red for unsafe, yellow for warning, green for stable)

### 1.8 — Commit & verify

```bash
git add src/hotcb/health.py src/hotcb/tests/test_health.py ...modified files...
git commit -m "feat(observability): TrainingHealthState with grad health, conflict scores, derived labels"
pytest  # must pass
```

Update `STREAMS.md`: mark `observability-v2` tasks as done.

---

## Phase 2: Execution Safety (Layer 3)

**Stream:** `execution-safety`
**Goal:** Rollback, action tracking, bounds enforcement, mutation budget.
**Files to modify:** `kernel.py`, `actuators/state.py`, `server/api.py`, `server/autopilot.py`
**Files to create:** `src/hotcb/tests/test_execution_safety.py`

> **Note:** This phase runs independently of Phase 1. If using subagents, can be
> parallelized with Phase 1 via worktree isolation.

### 2.1 — Plan & Document

Decision record:
- Rollback granularity: per-actuator snapshot stack vs whole-state snapshot? → Whole-state snapshot stack (simpler, MutableState already has `snapshot_all/restore_all`)
- Snapshot retention: keep last N snapshots? → Yes, configurable `max_snapshots` (default 10)
- Action effect tracking: where to store? → In mutation ledger entries, add `observed_effect` field populated async after cooldown
- Mutation budget: per-step or rolling window? → Rolling window (last N steps)

### 2.2 — Snapshot stack and explicit rollback

**Tests first** (`test_execution_safety.py`):
```
test_snapshot_created_on_apply → after MutableState.apply(), snapshot stack grows
test_rollback_restores_previous_state → after rollback, actuator values match snapshot
test_rollback_multiple_times → can rollback through stack
test_rollback_empty_stack → returns error, no crash
test_snapshot_stack_capped → after max_snapshots, oldest dropped
test_rollback_records_in_ledger → rollback op appears in hotcb.applied.jsonl
test_rollback_via_kernel_op → kernel processes "rollback" op correctly
```

**Implementation:**
- Add `_snapshot_stack: List[dict]` to `MutableState`
- `MutableState.apply()` → push snapshot before applying
- `MutableState.rollback(n=1)` → pop and restore
- `HotKernel._handle_default_op()` → handle `op="rollback"`, call `mutable_state.rollback()`
- Add ledger entry for rollback ops

### 2.3 — Action effect tracking

**Tests first** (add to `test_execution_safety.py`):
```
test_effect_tracker_records_baseline → on mutation, captures metric values at mutation step
test_effect_tracker_computes_delta → after N steps, computes metric change
test_effect_tracker_classifies_outcome → improved/neutral/degraded based on key metric
test_effect_tracker_timeout → if no new metrics in N steps, marks as "timeout"
test_effect_tracker_multiple_concurrent → tracks multiple mutations independently
test_effect_in_ledger → observed_effect field populated in applied.jsonl
```

**Implementation:**
- `EffectTracker` class in `health.py` (or new `tracking.py`):
  - `on_mutation(step, param_key, old_value, new_value, metric_snapshot: dict)` — records baseline
  - `on_metrics(step, metrics: dict)` — checks pending effects, computes deltas after cooldown_steps
  - `get_pending() -> List[PendingEffect]`
  - `get_completed() -> List[CompletedEffect]`
- Wire into kernel: after successful `mutable_state.apply()`, call `effect_tracker.on_mutation()`
- Wire into kernel step: on each `kernel.apply()`, call `effect_tracker.on_metrics()`
- CompletedEffect includes: `{param_key, step, delta: {metric: change}, outcome: "improved"|"neutral"|"degraded"}`

### 2.4 — Rollback triggers

**Tests first** (add to `test_execution_safety.py`):
```
test_auto_rollback_on_degradation → key metric worsens by >threshold after mutation → auto rollback
test_auto_rollback_respects_cooldown → doesn't trigger during cooldown window
test_auto_rollback_disabled_by_config → can disable auto-rollback
test_auto_rollback_logs_reason → ledger entry includes "auto_rollback" reason
test_no_auto_rollback_on_improvement → key metric improves → no rollback
```

**Implementation:**
- `EffectTracker.check_rollback_triggers()` — called on each `on_metrics()`
- If key metric degrades by > `rollback_threshold` (default 10%) within `rollback_window` steps after mutation → returns rollback request
- Kernel processes rollback request via normal rollback path
- Configurable via `AutopilotConfig` or `DashboardConfig`

### 2.5 — Rule bounds enforcement

**Tests first** (add to `test_execution_safety.py`):
```
test_rule_action_validated_against_actuator_bounds → rule proposes lr=5.0 but max=1.0 → clamped
test_rule_action_for_unknown_actuator → rejected with error
test_rule_action_respects_type → rule proposes string for FLOAT actuator → rejected
test_rule_bounds_same_as_ai_bounds → same validation path for rules and AI actions
```

**Implementation:**
- In `AutopilotEngine._apply_action()`, before writing command to JSONL:
  - Look up actuator in MutableState (needs reference or describe_all cache)
  - Call `actuator.validate(proposed_value)`
  - Clamp if out of bounds, reject if type mismatch
- Share validation logic with `ai_prompts.parse_ai_response()` action validation

### 2.6 — Mutation budget

**Tests first** (add to `test_execution_safety.py`):
```
test_mutation_budget_allows_within_limit → 3 mutations in 100 steps, budget=5 → allowed
test_mutation_budget_blocks_over_limit → 6th mutation in 100 steps, budget=5 → blocked
test_mutation_budget_rolling_window → old mutations fall out of window
test_mutation_budget_configurable → budget=0 means unlimited
test_mutation_budget_applies_to_rules_and_ai → both channels count against same budget
```

**Implementation:**
- Track mutation timestamps in `AutopilotEngine` or kernel
- Before applying any mutation (rule or AI), check: count of mutations in last `budget_window` steps < `mutation_budget`
- Configurable: `mutation_budget` (default 10), `budget_window_steps` (default 200)

### 2.7 — Rollback API endpoint and dashboard button

**Tests first** (add to `test_server_app.py` or `test_execution_safety.py`):
```
test_rollback_endpoint_returns_success → POST /api/rollback returns restored values
test_rollback_endpoint_empty_stack → returns 400 with error message
test_rollback_endpoint_with_snapshot_id → specific snapshot restore
```

**Implementation:**
- `api.py` → `POST /api/rollback` endpoint
- Dashboard: add rollback button to timeline items (JS in `panels.js`)

### 2.8 — Commit & verify

```bash
git add ...
git commit -m "feat(execution): rollback stack, effect tracking, mutation budget, bounds enforcement"
pytest  # must pass
```

Update `STREAMS.md`: mark `execution-safety` tasks as done.

---

## Phase 3: Actuator Expansion

**Stream:** `actuator-expand`
**Goal:** New actuator families — model, data/curriculum, safety.
**Files to create:** `src/hotcb/actuators/data.py`, `src/hotcb/actuators/model.py`, `src/hotcb/tests/test_actuator_expand.py`
**Files to modify:** `actuators/__init__.py`, `adapters/lightning.py`, `adapters/hf.py`, `capabilities.py`

> **Note:** Fully independent — can be parallelized with Phase 1 and Phase 2 via worktree.

### 3.1 — Plan & Document

Decision record:
- Model actuators: use `named_modules()` not `named_children()` for deeper groups
- Data actuators: `setattr` closure pattern, same as loss_actuators
- Safety actuators: live in `actuators/__init__.py` alongside other constructors
- All new actuators follow exact same pattern: convenience constructor returns `List[HotcbActuator]`
- `HotDataKernel` is just sugar over `data_actuators()` — not a new module type

### 3.2 — Model actuators: freeze/unfreeze

**Tests first** (`test_actuator_expand.py`):
```
test_model_actuators_creates_bool_per_group → 2 groups → 2 BOOL actuators
test_freeze_actuator_apply_disables_grad → after apply(False), module.requires_grad == False for all params
test_unfreeze_actuator_apply_enables_grad → after apply(True), requires_grad == True
test_freeze_actuator_pattern_matching → group "encoder.*" matches encoder.layer1, encoder.layer2
test_freeze_unknown_group → returns error on apply
test_model_actuators_empty_model → returns empty list, no crash
test_freeze_actuator_describe_space → type=bool, group="model"
test_model_actuators_with_real_torch_module → integration test with nn.Module (skip if no torch)
```

**Implementation** (`actuators/model.py`):
```python
def model_actuators(
    model,  # nn.Module
    groups: Dict[str, List[str]],  # {"trunk": ["encoder.*"], "head": ["decoder.*"]}
) -> List[HotcbActuator]:
```
- For each group, create BOOL actuator with `param_key=f"freeze_{group_name}"`
- `apply_fn` closure: iterate `model.named_modules()`, match against patterns, call `requires_grad_(value)`
- Initial value: `True` (all unfrozen by default)

### 3.3 — Gradient clipping actuator

**Tests first** (add to `test_actuator_expand.py`):
```
test_grad_clip_actuator_created → FLOAT type, group="optimizer"
test_grad_clip_apply_changes_value → apply(0.5) updates stored clip value
test_grad_clip_bounds → min=0.01, max=100.0
test_grad_clip_zero_disables → apply(0) or apply(None) disables clipping
test_grad_clip_describe_space → correct schema for dashboard
```

**Implementation:**
- `grad_clip_actuator(initial_value, bounds)` in `actuators/__init__.py`
- The `apply_fn` stores the value in a mutable container; the adapter's training loop reads it
- For Lightning: `apply_fn` mutates `trainer.gradient_clip_val` directly

### 3.4 — SWA/EMA actuators

**Tests first** (add to `test_actuator_expand.py`):
```
test_swa_actuator_created → BOOL type, group="model"
test_ema_actuator_created → BOOL type, group="model"
test_swa_apply_true_wraps_model → if torch available, wraps with AveragedModel
test_swa_apply_false_unwraps → restores original model
test_swa_without_torch → graceful skip/error
test_ema_apply_stores_flag → sets flag that training loop can check
```

**Implementation:**
- `swa_actuator(model)` and `ema_actuator(model)` in `actuators/__init__.py`
- SWA: `apply_fn` wraps/unwraps model with `torch.optim.swa_utils.AveragedModel`
- EMA: `apply_fn` sets a flag on a container dict; actual EMA logic lives in training loop or adapter hook
- Both are BOOL actuators

### 3.5 — Data actuators

**Tests first** (add to `test_actuator_expand.py`):
```
test_data_actuators_from_attrs → creates actuators for each declared attr
test_data_actuator_apply_setattr → apply(0.5) calls setattr(dataset, "aug_strength", 0.5)
test_data_actuator_type_inference → float attr → FLOAT type, int → INT, str with choices → CHOICE
test_data_actuator_bounds_from_spec → min/max from spec dict
test_data_actuator_missing_attr → attr not on dataset → error on apply, not on create
test_data_actuator_group_is_data → all actuators have group="data"
test_data_actuators_empty_attrs → returns empty list
test_data_actuator_describe_space → correct schema
```

**Implementation** (`actuators/data.py`):
```python
def data_actuators(
    dataset_or_loader,
    attrs: Dict[str, dict],  # {"aug_strength": {"type": "float", "min": 0, "max": 1}}
) -> List[HotcbActuator]:
```
- For each attr, create actuator with `apply_fn` = `setattr(obj, attr_name, value)` closure
- Type inference from spec dict: `"float"` → FLOAT, `"int"` → INT, `"bool"` → BOOL, `"choice"` → CHOICE
- If spec has `"choices"`, use CHOICE type
- If spec has `"log"` or `"log_scale"`, use LOG_FLOAT

### 3.6 — HotDataKernel convenience class

**Tests first** (add to `test_actuator_expand.py`):
```
test_hot_data_kernel_creates_actuators → wraps dataset, returns actuator list
test_hot_data_kernel_type_inference → infers type from current attr value on dataset
test_hot_data_kernel_with_bounds → respects declared bounds
test_hot_data_kernel_no_attrs → auto-detects known attr names if present
```

**Implementation** (`actuators/data.py`):
```python
class HotDataKernel:
    """Convenience wrapper — discovers mutable attributes on a dataset."""
    KNOWN_ATTRS = ["augmentation_strength", "aug_strength", "curriculum_stage",
                   "difficulty", "sample_weights", "mix_ratio", "hard_case_weight"]

    def __init__(self, dataset, mutable_attrs=None):
        # If mutable_attrs not given, scan for KNOWN_ATTRS that exist on dataset
        # Build specs, call data_actuators()
        self.actuators = data_actuators(dataset, attrs=specs)
```

### 3.7 — Safety actuators

**Tests first** (add to `test_actuator_expand.py`):
```
test_safe_mode_actuator → BOOL, group="safety"
test_mutation_lock_actuator → BOOL, group="safety"
test_mutation_lock_blocks_apply → when locked, MutableState.apply() returns error
test_safe_mode_halves_deltas → not enforced in actuator itself, just a flag
```

**Implementation:**
- `safety_actuators()` in `actuators/__init__.py` — returns `[safe_mode, mutation_lock]`
- `safe_mode`: BOOL actuator, sets flag in container dict
- `mutation_lock`: BOOL actuator, when True, kernel checks before applying any mutation
- Wire `mutation_lock` check into `MutableState.apply()` or `HotKernel._handle_default_op()`

### 3.8 — Lightning adapter auto-discovery

**Tests first** (add to `test_integration_lightning.py` or new file):
```
test_lightning_discovers_freezeable_groups → capabilities includes freezeable_groups
test_lightning_discovers_grad_clip → capabilities includes grad_clip_available
test_lightning_discovers_dataset_attrs → capabilities includes data_actuator_keys
test_lightning_wires_freeze_actuators → freeze actuators in mutable_state
test_lightning_wires_data_actuators → data actuators in mutable_state (if dataset has known attrs)
```

**Implementation:**
- `_detect_capabilities()` → add `freezeable_groups` from `pl_module.named_children()` top-level names
- `_detect_capabilities()` → add `data_actuator_keys` by scanning dataset for known mutable attrs
- `_wire_mutable_state()` → create freeze actuators for discovered groups
- `_wire_mutable_state()` → create data actuators for discovered dataset attrs

### 3.9 — Capabilities extension

**Tests first** (add to existing capabilities tests):
```
test_capabilities_includes_new_fields → freezeable_groups, data_actuator_keys, grad_clip_available, swa_available
test_capabilities_json_roundtrip → save/load preserves new fields
test_capabilities_backwards_compatible → old JSON without new fields loads fine
```

**Implementation:**
- Add fields to `TrainingCapabilities` dataclass
- Update `save()` and `from_dict()` / `from_json()` for new fields
- Update `controls_from_capabilities()` in `server/config.py` to generate controls for freeze/data actuators

### 3.10 — Export in `__init__.py`

- Add `data_actuators`, `model_actuators`, `grad_clip_actuator`, `swa_actuator`, `ema_actuator`, `safety_actuators` to `actuators/__init__.py`
- Add `HotDataKernel` to `actuators/data.py` exports

### 3.11 — Commit & verify

```bash
git add ...
git commit -m "feat(actuators): model freeze, grad clip, SWA/EMA, data/curriculum, safety actuators"
pytest  # must pass
```

Update `STREAMS.md`: mark `actuator-expand` tasks as done.

---

## Phase 4: Policy Packs (Layer 2)

**Stream:** `policy-packs`
**Depends on:** Phase 1 (observability) + Phase 3 (actuators)
**Goal:** Ship 5 default rule packs + YAML DSL enrichment.
**Files to create:** `src/hotcb/server/guidelines/*.yaml`, `src/hotcb/tests/test_policy_packs.py`
**Files to modify:** `server/autopilot.py`, `server/app.py`, `server/static/js/panels.js`

### 4.1 — Plan & Document

Decision record:
- Pack storage: YAML files in `server/guidelines/` directory, shipped with package
- Pack loading: `AutopilotEngine.load_pack(name)` reads YAML, adds rules
- Multiple packs can be loaded simultaneously (rules merge, IDs prefixed with pack name)
- Rule DSL enrichment: add `bounds`, `rollback_if`, `priority`, `suppress_rules` fields to `AutopilotRule`
- Pack metadata: each YAML has `name`, `description`, `version`, `requires` (actuator families needed)
- Conflict resolution: if two rules fire for same actuator in same cooldown window, higher priority wins

### 4.2 — Rule DSL enrichment

**Tests first** (`test_policy_packs.py`):
```
test_rule_with_bounds → rule with bounds field validates proposed value
test_rule_with_rollback_if → rollback condition evaluated after cooldown
test_rule_with_priority → higher priority rule wins over lower
test_rule_with_suppress → when rule A fires, rule B is suppressed
test_rule_priority_default → rules without priority get "medium"
test_rule_serialization → enriched rule round-trips through YAML
test_rule_conflict_detection → two rules for same param in same window → only higher priority applies
```

**Implementation:**
- Extend `AutopilotRule` dataclass:
  ```python
  bounds: Optional[Dict[str, float]] = None      # {"min": 0.001, "max": 0.1}
  rollback_if: Optional[Dict[str, Any]] = None   # {"no_improvement_after": 1000}
  priority: str = "medium"                        # "low", "medium", "high", "critical"
  suppress_rules: Optional[List[str]] = None      # rule IDs to suppress when this fires
  ```
- Add priority ordering: `_PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}`
- In `_evaluate_rules()`, sort fired rules by priority, apply conflict resolution

### 4.3 — Pack YAML format and loader

**Tests first** (add to `test_policy_packs.py`):
```
test_load_pack_from_yaml → loads pack, rules added to engine
test_load_pack_prefixes_ids → pack "stability" → rule IDs "stability.nan_guard" etc
test_load_pack_metadata → pack name, description, version accessible
test_load_pack_requires_check → pack with requires=["freeze"] warns if freeze actuators absent
test_load_multiple_packs → rules from both packs active
test_unload_pack → removes all rules from a pack
test_list_available_packs → returns names of YAML files in guidelines/
test_load_nonexistent_pack → returns error
test_pack_yaml_syntax_error → returns error, engine unaffected
```

**Implementation:**
- `AutopilotEngine.load_pack(pack_name: str)` → reads `guidelines/{pack_name}.yaml`
- `AutopilotEngine.unload_pack(pack_name: str)` → removes rules with matching prefix
- `AutopilotEngine.list_packs()` → scans guidelines directory
- Pack YAML format:
  ```yaml
  name: stability_basics
  description: Core stability heuristics for any training run
  version: "1.0"
  requires: []  # no special actuators needed
  rules:
    - id: nan_guard
      condition: custom
      metric_name: train_loss
      params:
        expr: "math.isnan(value) or math.isinf(value)"
        window: 1
      action: reduce_lr_factor
      action_params:
        factor: 0.5
      confidence: critical
      priority: critical
      cooldown: 50
      description: "Emergency LR cut on NaN/Inf loss"
  ```

### 4.4 — Pack 1: Stability Basics

**Tests first** (add to `test_policy_packs.py`):
```
test_stability_pack_loads → all rules parse without error
test_stability_nan_guard_fires → NaN in loss → reduce_lr action
test_stability_grad_spike_fires → grad_norm > 10x mean → enable clipping
test_stability_lr_emergency_fires → loss diverging > 5x → lr halved
test_stability_pack_rule_count → exactly N rules in pack
```

**YAML** (`server/guidelines/stability_basics.yaml`):
- `nan_guard`: NaN/Inf loss → reduce LR by 50%, cooldown 50 steps, critical priority
- `grad_spike_clip`: grad_norm > 10x rolling mean → enable grad clipping at 2x mean, cooldown 200
- `lr_emergency`: loss > 5x EMA → reduce LR by 70%, cooldown 100, critical priority
- `loss_spike_recovery`: single-step spike > 3x previous → reduce LR by 20%, cooldown 50

### 4.5 — Pack 2: Multi-Loss Assist

**Tests first** (add to `test_policy_packs.py`):
```
test_multiloss_pack_loads → all rules parse
test_multiloss_conflict_mitigation → high conflict_score → reduce aux weight
test_multiloss_ratio_targeting → one loss dominates → rebalance
test_multiloss_aux_rollback → aux loss causes instability → reduce to floor
```

**YAML** (`server/guidelines/multi_loss_assist.yaml`):
- `aux_conflict_reduce`: conflict_score > 0.3 for 3 windows → reduce highest-gradient aux loss weight by 15%
- `loss_ratio_targeting`: dominance ratio > 3.0 → reduce dominant loss weight by 10%
- `aux_instability_rollback`: loss spike after aux weight increase → rollback aux weight
- `aux_warmup_ramp`: early training (step < 10% of max) → gradually increase aux weights from 0.1 to target

### 4.6 — Pack 3: Distillation Assist

**Tests first**:
```
test_distill_pack_loads → all rules parse
test_distill_spatial_delay → spatial loss weight 0 until step N, then ramp
test_distill_summary_first → summary loss prioritized early
```

**YAML** (`server/guidelines/distillation_assist.yaml`):
- `summary_first_warmup`: if step < warmup_steps → keep spatial/detail loss weights low
- `spatial_delayed_ramp`: ramp spatial loss from 0 to target over N steps after warmup
- `feature_health_check`: if feature variance dropping → reduce spatial weight
- `topk_scheduling`: if spatial loss plateaus → reduce top-k ratio gradually

### 4.7 — Pack 4: Plateau Recovery

**Tests first**:
```
test_plateau_pack_loads → all rules parse
test_plateau_detection_fires → flat loss for >100 steps → action
test_plateau_lr_restart → stagnation → cosine restart or reduce LR
```

**YAML** (`server/guidelines/plateau_recovery.yaml`):
- `stagnation_detect`: loss range < 1% over 100 steps → reduce LR by 30%
- `cosine_restart`: if plateau persists after LR reduce → propose cosine warmup restart
- `aux_emphasis_shift`: if primary plateaued but aux improving → increase primary weight
- `conservative_finish`: if near end (>80% steps) → lock mutations, reduce LR gently

### 4.8 — Pack 5: Finish Strong

**Tests first**:
```
test_finish_pack_loads → all rules parse
test_finish_swa_trigger → late training → enable SWA
test_finish_mutation_lock → final 10% steps → enable mutation_locked
```

**YAML** (`server/guidelines/finish_strong.yaml`):
- `enable_swa_late`: step > 75% of max_steps → enable SWA (requires SWA actuator)
- `enable_ema_late`: step > 80% → enable EMA
- `mutation_lockdown`: step > 90% → enable mutation_locked, only safety rules can fire
- `best_checkpoint`: validation loss reaches new minimum → log checkpoint recommendation

### 4.9 — Pack API and dashboard

**Tests first** (add to `test_server_app.py`):
```
test_api_list_packs → GET /api/autopilot/packs returns pack list with metadata
test_api_load_pack → POST /api/autopilot/packs/load with pack name → rules added
test_api_unload_pack → POST /api/autopilot/packs/unload → rules removed
test_api_active_packs → GET /api/autopilot/packs/active returns loaded packs
```

**Implementation:**
- API endpoints in autopilot router:
  - `GET /api/autopilot/packs` → list available packs with metadata
  - `POST /api/autopilot/packs/load` → load pack by name
  - `POST /api/autopilot/packs/unload` → unload pack
  - `GET /api/autopilot/packs/active` → list currently loaded packs
- Dashboard: add pack selector UI in autopilot panel (toggles per pack)

### 4.10 — Commit & verify

```bash
git add ...
git commit -m "feat(policy): 5 policy packs, YAML DSL enrichment, pack API and dashboard selector"
pytest  # must pass
```

Update `STREAMS.md`: mark `policy-packs` tasks as done.

---

## Phase 5: AI Feedback Loop

**Stream:** `ai-feedback-loop`
**Depends on:** Phase 2 (execution-safety for effect tracking)
**Goal:** AI learns from action outcomes, cross-run learning, adaptive cadence.
**Files to modify:** `server/ai_engine.py`, `server/ai_prompts.py`, `server/autopilot.py`
**Files to create:** `src/hotcb/tests/test_ai_feedback.py`

### 5.1 — Plan & Document

Decision record:
- Action outcomes: use `EffectTracker.get_completed()` from Phase 2
- Include last 5 completed effects in AI prompt as "Recent Action Outcomes" table
- Cross-run: `AIState.run_history` already exists — synthesize into system prompt
- Adaptive cadence: if last 3 actions improved key metric → shorten interval; if neutral → lengthen
- Failure recovery: single retry with 5s backoff; on second failure → fallback to rules-only for N steps

### 5.2 — Action outcome reporting in AI prompt

**Tests first** (`test_ai_feedback.py`):
```
test_ai_prompt_includes_action_outcomes → build_context with completed effects → outcomes in prompt
test_ai_prompt_no_outcomes → no completed effects → section omitted
test_outcome_formatting → improved/neutral/degraded labeled correctly
test_outcome_includes_metric_deltas → shows delta values per metric
```

**Implementation:**
- In `build_context()`, if `effect_tracker` provided and has completed effects:
  - Add "Recent Action Outcomes" section
  - Table: step | param | old→new | key_metric_delta | outcome
- Pass `effect_tracker` through from `AutopilotEngine.evaluate_async()`

### 5.3 — Cross-run context synthesis

**Tests first** (add to `test_ai_feedback.py`):
```
test_cross_run_context_in_prompt → run_history with 2 prior runs → context in system prompt
test_cross_run_empty → no prior runs → section omitted
test_cross_run_weight_recent → most recent run's learnings appear first
test_cross_run_max_entries → caps at 3 most recent runs
```

**Implementation:**
- In `build_context()` system prompt section, add "Prior Run History":
  - For each run in `ai_state.run_history[-3:]`:
    - Run number, steps completed, final key metric value
    - Learnings (from `carried_context`)
    - Verdict (if present)

### 5.4 — Adaptive cadence

**Tests first** (add to `test_ai_feedback.py`):
```
test_cadence_shortens_on_success → 3 consecutive improved → interval halved (min 50 steps)
test_cadence_lengthens_on_neutral → 3 consecutive neutral → interval doubled (max 500 steps)
test_cadence_resets_on_degraded → degraded outcome → reset to default interval
test_cadence_respects_bounds → never below min_interval or above max_interval
```

**Implementation:**
- Track outcome streak in `AIState` (new field: `outcome_streak: List[str]`)
- In `should_invoke()`, compute adaptive interval from streak:
  - 3+ improved → `base_interval * 0.5`
  - 3+ neutral → `base_interval * 2.0`
  - Any degraded → reset to `base_interval`

### 5.5 — Failure recovery

**Tests first** (add to `test_ai_feedback.py`):
```
test_llm_retry_on_timeout → first call times out, second succeeds → decision returned
test_llm_retry_on_error → first call errors, second succeeds
test_llm_fallback_after_double_failure → two failures → falls back to rules-only for N steps
test_fallback_recovers → after N steps, AI re-enabled
test_retry_backoff → second call delayed by 5s
```

**Implementation:**
- In `_call_llm()`, wrap with retry logic:
  ```python
  for attempt in range(2):
      try:
          return await self._raw_llm_call(...)
      except Exception:
          if attempt == 0:
              await asyncio.sleep(5)
              continue
          self._fallback_until_step = current_step + fallback_steps
          return None
  ```
- In `should_invoke()`, check `_fallback_until_step`

### 5.6 — Mode auto-progression

**Tests first** (add to `test_ai_feedback.py`):
```
test_auto_progress_suggest_to_auto → 5 consecutive accepted suggestions → mode becomes ai_auto
test_auto_progress_respects_config → disabled by default, opt-in via config
test_auto_progress_resets_on_reject → rejected suggestion resets counter
test_auto_progress_does_not_downgrade → already in ai_auto → stays there
```

**Implementation:**
- Track `accepted_suggestion_streak` in `AutopilotEngine`
- If mode is `ai_suggest` and streak >= `auto_progress_threshold` (default 5):
  - Upgrade to `ai_auto`
  - Log mode change in ledger
- Config: `auto_progress_enabled: bool = False`, `auto_progress_threshold: int = 5`

### 5.7 — Commit & verify

```bash
git add ...
git commit -m "feat(ai): action outcome feedback, cross-run learning, adaptive cadence, retry/fallback"
pytest  # must pass
```

Update `STREAMS.md`: mark `ai-feedback-loop` tasks as done.

---

## Phase 6: Documentation

**Stream:** `docs-autopilot-spec`

### 6.1 — docs/autopilot.md

Write comprehensive autopilot architecture doc:
- 3-layer architecture diagram (text-based)
- Mode descriptions (0-4)
- State flow: metrics → health → policy → execution → ledger
- Configuration reference

### 6.2 — docs/policy_packs.md

Write policy pack reference:
- Pack catalog with descriptions
- YAML DSL reference (all fields, types, examples)
- Custom rule authoring guide
- How to create and share custom packs

### 6.3 — docs/guarantee_envelope.md

Write guarantee envelope definition:
- What "convergence assist" means
- Envelope conditions (numerically stable, metrics accessible, actuators exposed, constraints satisfiable)
- What's guaranteed inside envelope (stability, bounded intervention, rollback, audit)
- What's NOT guaranteed (mathematical convergence proof)

### 6.4 — Update README

Add autopilot section to README:
- One-line positioning
- Quick start with policy packs
- Link to detailed docs

### 6.5 — Commit

```bash
git commit -m "docs: autopilot architecture, policy packs, guarantee envelope"
```

---

## Phase 7: Integration Tests & Cleanup

### 7.1 — Integration tests (from `feature-test-coverage` stream)

**Tests** (`test_integration.py`):
```
test_demo_10_steps_produces_metrics → run demo 10 steps, check hotcb.metrics.jsonl
test_demo_with_policy_pack → load stability pack, run demo, verify rules evaluated
test_health_state_computed_during_demo → health state populated after demo steps
test_rollback_during_demo → apply mutation, rollback, verify restored
test_full_autopilot_loop → suggest mode, rule fires, action proposed
```

### 7.2 — Error handling cleanup (from `fix-error-handling` stream)

- Audit all `except Exception: pass` and `except:` sites
- Replace with specific exception types + `log.warning()`
- Prioritize: kernel.py, autopilot.py, ai_engine.py

### 7.3 — Adapter import guards (from `fix-adapter-imports` stream)

- Wrap `import pytorch_lightning` and `import transformers` with try/except
- Raise friendly `ImportError` with install instructions

### 7.4 — Final test run & commit

```bash
pytest  # full suite must pass
git add -A
git commit -m "test: integration tests, error handling cleanup, adapter import guards"
```

---

## Phase 8: Final Assembly

### 8.1 — Update STREAMS.md

Mark all completed streams as `done`.

### 8.2 — Update CLAUDE.md

Add sections for:
- Policy packs (what ships, how to load)
- Health state system
- New actuator families
- Guarantee envelope concept

### 8.3 — Update memory files

Update `MEMORY.md` and related memory files with new module paths, test counts, architecture notes.

### 8.4 — Final commit on feature branch

```bash
git add -A
git commit -m "chore: update docs, streams, memory for autopilot spec v1"
```

### 8.5 — Summary

Write a summary of everything implemented to `.claude/plans/EXECUTE_RESULTS.md`:
- Phase completion status
- Total new tests added
- Total new files created
- Key architectural decisions made
- Any items deferred or blocked

---

## Execution Notes for Autonomous Mode

### Parallelism Strategy

**Round 1** (3 parallel worktree agents):
- Agent A: Phase 1 (observability-v2)
- Agent B: Phase 2 (execution-safety)
- Agent C: Phase 3 (actuator-expand)

**Round 2** (after Round 1 merges):
- Agent D: Phase 4 (policy-packs) — needs Phase 1 + 3
- Agent E: Phase 5 (ai-feedback-loop) — needs Phase 2

**Round 3** (sequential):
- Phase 6 (docs)
- Phase 7 (integration tests + cleanup)
- Phase 8 (final assembly)

### Merge Strategy

After each worktree agent completes:
1. Switch to main branch
2. Merge worktree branch: `git merge <worktree-branch> --no-ff`
3. Run full test suite
4. If conflicts, resolve them (prefer the feature branch changes)
5. If tests fail, fix before proceeding

### Decision-Making Guidelines

When facing ambiguity:
- **Simpler is better** — fewer abstractions, less indirection
- **Follow existing patterns** — look at how `optimizer_actuators()` works, do the same
- **Closure pattern** — capture live objects, mutate via closures. Never add middleware layers.
- **Fail open** — if a feature can't detect something, skip gracefully (don't crash)
- **Test boundaries** — test the public API, not internal helpers
- **YAML over code** — policy packs are data, not logic. Keep rules declarative.

### Known Risks

1. **Phase 1 health.py may need TrendCompressor refactoring** — if so, extract shared math into a `math_utils.py` rather than duplicating
2. **Phase 3 model actuators need torch** — gate behind try/except, tests should `skipIf` no torch
3. **Phase 4 pack YAML conditions may need new condition types** — add them to `_VALID_CONDITIONS` in autopilot.py
4. **Phase 5 async retry needs careful timeout handling** — use `asyncio.wait_for` with explicit timeout

### Quality Gates

Before each commit:
- [ ] `pytest` passes (all tests)
- [ ] No new `except:` or `except Exception: pass` introduced
- [ ] New code has tests (success + failure cases)
- [ ] Decision documented in DECISIONS.md
- [ ] STREAMS.md updated
