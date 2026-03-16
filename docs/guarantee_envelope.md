# Guarantee Envelope

This document defines what hotcb's autopilot system guarantees and does not guarantee about training outcomes.

---

## What "Convergence Assist" Means

hotcb's autopilot helps training converge faster and more reliably by detecting problems early and applying corrective interventions. It does **not** guarantee mathematical convergence. The system is a best-effort optimization assistant operating on empirical signals -- it monitors metric trends, detects anomalies, and adjusts hyperparameters within bounded ranges.

The autopilot is designed to catch common training failures (NaN/Inf, gradient spikes, loss plateaus, multi-loss conflicts) and apply well-understood corrective actions (LR reduction, loss weight adjustment, rollback). It cannot reason about model architecture, data quality, or theoretical convergence properties.

---

## Envelope Conditions

All of the following must be true for autopilot guarantees to hold:

### 1. Numerically Stable Baseline

The training setup must be capable of producing finite loss values under some hyperparameter configuration. If the model architecture or data inherently produces NaN/Inf for all parameter settings, the autopilot cannot recover training.

*What this means in practice:* The model should produce at least a few steps of finite loss before issues arise. The autopilot can recover from occasional NaN/Inf spikes, but not from a setup that never produces valid gradients.

### 2. Metrics Accessible

The user must log the key metrics that the autopilot monitors. At minimum:
- A primary loss metric (e.g., `train_loss`)
- Any metrics referenced by loaded policy pack rules

For AI modes, logging `grad_norm`, validation metrics, and per-task loss terms provides richer context for decision-making.

*How to satisfy:* Use `MetricsCollector` (passed to `HotKernel`) to log metrics each step. The adapters (Lightning, HF) do this automatically.

### 3. Actuators Exposed

The parameters the autopilot needs to adjust must be registered as actuators and accessible via the `MutableState` system.

- **Optimizer**: Pass `env["optimizer"]` or use an adapter (Lightning/HF auto-discover)
- **Loss weights**: Register via `loss_actuators(weights_dict)` or pass `env["mutable_state"]`
- **Callbacks**: Register callbacks via `hotcb cb load`

*What this means:* If the autopilot proposes an LR change but no optimizer actuator is registered, the action writes to `hotcb.commands.jsonl` but has no effect on training.

### 4. Constraints Satisfiable

The bounds configured on actuators and in policy pack rules must not conflict with training requirements. For example:
- LR bounds must include values where the model can learn
- Loss weight bounds must allow meaningful gradient flow from all terms
- Mutation budget must allow enough interventions to recover from problems

*What this means:* If you set `lr` bounds to `[1e-7, 1e-6]` but the model needs `1e-3` to learn, the autopilot is constrained to a region where training cannot progress.

---

## What's Guaranteed Inside the Envelope

When all envelope conditions are met, the autopilot provides these guarantees:

### Stability Interventions

- **NaN/Inf guard**: Loss values that are NaN or Inf trigger immediate LR reduction (via `stability_basics.nan_guard`)
- **Spike recovery**: Sharp loss increases (>threshold in N steps) trigger graduated LR reduction
- **Gradient spike clip**: Extreme gradient norms trigger LR reduction to prevent divergence

These interventions are bounded and graduated -- they reduce LR by a factor (e.g., 0.5x), not to an arbitrary value.

### Bounded Intervention

- **Mutation budget**: At most N mutations per M steps (default: 10 per 200 steps). Exceeding the budget causes actions to be rejected, not queued.
- **Bounds enforcement**: All parameter changes are clamped to actuator min/max bounds before application. Type mismatches (e.g., setting a float parameter to a string) are rejected.
- **Priority conflict resolution**: When multiple rules target the same parameter, only the highest-priority rule's action is applied.
- **Cooldown enforcement**: Each rule has a minimum number of steps between firings. No rule can fire repeatedly without cooldown.

### Rollback Capability

- **Snapshot stack**: `MutableState` pushes a snapshot before every mutation. The stack holds up to 10 snapshots (configurable).
- **Manual rollback**: `MutableState.rollback(n)` restores the nth-previous snapshot, re-applying values via actuator `apply_fn` closures.
- **Rollback-if conditions**: Policy pack rules can declare `rollback_if: {no_improvement_after: N}` to flag actions for rollback when no improvement is observed.

### Audit Trail

- **Full ledger**: Every applied mutation is recorded in `hotcb.applied.jsonl` with step number, timestamp, module, operation, parameters, and source (manual/autopilot/ai_autopilot).
- **AI decision history**: Every LLM invocation records reasoning, proposed actions, cost, and outcome in the `LLMAutopilotEngine` history (accessible via `/api/autopilot/ai/history`).
- **Effect tracking**: `EffectTracker` records baseline and post-intervention metrics for each mutation, classifying outcomes as improved/degraded/neutral/timeout.

---

## What's NOT Guaranteed

### Mathematical Convergence Proof

The autopilot operates on empirical trend analysis, not theoretical convergence guarantees. A model with fundamental architecture problems, insufficient capacity, or adversarial data distributions may not converge regardless of hyperparameter tuning.

### Optimal Hyperparameters

The autopilot makes graduated, corrective adjustments -- it is not a hyperparameter search algorithm. For systematic hyperparameter optimization, use the `tune` module (`hotcb[tune]`) which provides Bayesian HPO via Optuna. The autopilot and tune module serve different purposes:
- **Autopilot**: Reactive stabilization and plateau recovery during a single run
- **Tune**: Proactive search across hyperparameter space over multiple trials

### Faster Training in All Cases

Autopilot interventions add a small per-step overhead (rule evaluation, health state computation). In cases where training is already well-configured and stable, the autopilot may not provide any benefit beyond monitoring. Setting mode to `off` or `suggest` eliminates auto-apply overhead while retaining observability.

### Correct Behavior with Adversarial Metric Inputs

The autopilot trusts the metrics it receives. If a training script reports fabricated or adversarial metric values (e.g., always reporting `loss=0.0` or oscillating between extreme values), the autopilot will react to those signals as if they were real. There is no mechanism to detect metric tampering.

### Cross-Run Optimal Strategy

AI multi-run memory (`hotcb.ai.state.json`) carries learnings across runs, but it is limited to the LLM's context window and the structured format of carried context. The system does not build a persistent knowledge base or learn optimal strategies across many runs. Multi-run awareness is bounded to the `max_runs` setting (default 3).

### Safe Behavior Beyond Actuator Bounds

If actuator bounds are misconfigured (e.g., allowing LR values that cause numeric overflow), the autopilot will apply values within those bounds even if they are unsafe. Bounds are the user's responsibility and define the autopilot's operating envelope.
