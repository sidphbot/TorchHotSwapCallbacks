# Autopilot Architecture

hotcb's autopilot is a layered system for autonomous training optimization. It ranges from simple rule-based interventions to full LLM-driven decision-making, with safety guards at every level.

---

## 3-Layer Architecture

```
                    +------------------------------------------+
                    |         Layer 3: EXECUTION               |
                    |                                          |
                    |  MutableState (snapshot rollback)         |
                    |  EffectTracker (action outcomes)          |
                    |  Mutation budget  |  Bounds enforcement   |
                    +------------------+-----------------------+
                              ^
                              | validated actions
                    +------------------------------------------+
                    |         Layer 2: POLICY                   |
                    |                                          |
                    |  Rule engine (AutopilotEngine)            |
                    |  Priority / suppress / conflict resolution|
                    |  Policy packs (YAML DSL)                  |
                    |  LLM decision engine (ai_suggest/ai_auto) |
                    +------------------+-----------------------+
                              ^
                              | health signals + alerts
                    +------------------------------------------+
                    |         Layer 1: OBSERVABILITY            |
                    |                                          |
                    |  TrainingHealthState                      |
                    |  Numeric stability (NaN/Inf/spike)        |
                    |  Gradient health (norm trend, clip rate)  |
                    |  Multi-loss conflict (dominance, corr)    |
                    |  Derived labels (oscillatory, plateaued)  |
                    |  TrendCompressor (slope/volatility/dir)   |
                    +------------------------------------------+
                              ^
                              | raw metrics
                    +------------------------------------------+
                    |  hotcb.metrics.jsonl (step, metrics dict) |
                    +------------------------------------------+
```

### Layer 1: Observability

The observability layer computes `TrainingHealthState` from raw metric history. It runs every evaluation cycle and produces structured diagnostics plus derived labels.

**Health signals:**

| Signal | Source | What it detects |
|--------|--------|-----------------|
| Numeric stability | All metrics | NaN count, Inf count, loss spikes (value > 3x EMA), `nan_detected` flag |
| Gradient health | `grad_norm` metric | Norm trend (rising/flat/falling), gradient clipping rate |
| Multi-loss conflict | `*_loss` metrics | Per-loss trend slopes, dominance ratios, detrended cross-correlation conflict score |
| Loss trend | Primary loss | Total loss trend direction, plateau score, oscillation score |

Health-derived fields (`conflict_score`, `loss_cv`, `grad_trend`) are automatically injected into the custom expression evaluator namespace, making them available to policy pack rule expressions without requiring explicit metric logging.

**Derived labels** (attached to health state):

| Label | Condition |
|-------|-----------|
| `numerically-unsafe` | Any NaN, Inf, or loss spike detected |
| `collapse-risk` | Loss near zero for 10+ consecutive steps |
| `aux-conflicted` | Multi-loss conflict score > 0.3 |
| `oscillatory` | Detrended loss CV > 0.1 |
| `stable-plateaued` | Loss flat (CV < 0.01, near-zero slope) for 50+ steps |
| `stable-improving` | Loss decreasing steadily with low residual noise |
| `converged-likely` | Train + val loss flat, low gradient norm |

**Trend compression** (`TrendCompressor`):

Raw metric streams are compressed into `TrendSummary` objects for token-efficient LLM context. Each summary contains slope, volatility (none/low/medium/high), direction classification (steep_down through spike), and notable events (new min, trend reversal, recent spike).

### Layer 2: Policy

The policy layer decides *what to do* based on health signals and alerts.

**Rule engine** (`AutopilotEngine`):
- Evaluates all enabled `AutopilotRule` instances against current metrics
- Checks cooldown periods per rule
- Sorts fired rules by priority (critical > high > medium > low)
- Builds a suppress set from higher-priority rules
- Resolves conflicts when multiple rules target the same actuator (higher priority wins)

**Policy packs**: Reusable YAML rule bundles. See [Policy Pack Reference](policy_packs.md).

**LLM decision engine** (`LLMAutopilotEngine`): In AI modes, rules act as the sensor/alert layer. The LLM receives compressed trend context, alert summaries, health state, and action history, then proposes actions from a constrained vocabulary (13 action types with typed, bounded parameters).

### Layer 3: Execution

The execution layer validates and applies actions with safety guarantees.

**MutableState** with snapshot rollback:
- Every `apply()` automatically pushes a snapshot before mutation
- `rollback(n)` pops n snapshots and restores actuator values
- Snapshot stack is capped (default 10) to bound memory

**EffectTracker** for action outcomes:
- Records baseline metrics at mutation time
- After a configurable cooldown, compares metrics to classify outcome as improved/degraded/neutral/timeout
- Completed effects are fed back to the LLM for outcome-aware decisions

**Mutation budget**: Configurable limit on mutations within a sliding window (default: 10 mutations per 200 steps). Actions that exceed the budget are rejected.

**Bounds enforcement**: Actions are validated against actuator descriptions loaded from `hotcb.actuators.json`. Numeric values are clamped to min/max bounds; type mismatches are rejected.

---

## Autopilot Modes

### Mode 0: `off`

No autopilot. All training control is manual via CLI, dashboard, or API. The engine does not evaluate rules or process metrics.

### Mode 1: `suggest`

Rule-based suggestion mode. The engine evaluates all enabled rules each step. When conditions fire, actions are recorded with status `"proposed"` and displayed in the dashboard. The human operator reviews and accepts or rejects each proposal.

Actions are never auto-applied regardless of confidence level.

### Mode 2: `auto`

Rule-based auto-apply mode. Critical, high, and medium confidence actions are applied automatically. Low confidence actions are proposed for human review. All actions pass through validation (bounds check, type check) and mutation budget enforcement before application.

### Mode 3: `ai_suggest`

LLM-driven suggestion mode. Rules run as the sensor layer, generating alerts. The `LLMAutopilotEngine` decides whether to invoke the LLM based on:
- **On-alert**: any rule fired
- **Periodic cadence**: adaptive interval based on outcome streak
- **AI-requested**: LLM specifies when to check back (`next_check` in response)

The LLM proposes actions from the constrained `ACTION_SCHEMA` vocabulary. All proposals are shown in the dashboard for human review. Accepted suggestions increment a streak counter; after a configurable threshold (default 5), the mode can auto-progress to `ai_auto`.

### Mode 4: `ai_auto`

LLM-driven auto-apply mode. Same invocation logic as `ai_suggest`, but proposed actions are applied immediately (with the same validation, bounds, and budget guards). Safety features:

- **Budget cap**: Configurable USD limit on LLM API costs; falls back to rule-based when exhausted
- **Action bounds**: All parameter changes bounded by `ACTION_SCHEMA` min/max
- **Minimum cooldown**: At least 10 steps between LLM invocations (configurable)
- **Fallback mode**: After two consecutive LLM call failures, enters fallback (skip invocations for 100 steps)
- **Noop bias**: LLM is prompted to prefer doing nothing when training is healthy
- **Auto-disable**: Engine disables itself on budget exhaustion

---

## State Flow

```
metrics (each step)
    |
    v
[Update metric history] --> [Compute TrainingHealthState]
    |                               |
    v                               v
[Evaluate rules]            [Health labels + signals]
    |                               |
    v                               v
[Fired rules sorted by priority]   [Feed to LLM context (AI modes)]
    |                               |
    v                               v
[Conflict resolution + suppress]   [LLM decision]
    |                               |
    +----------- actions -----------+
                    |
                    v
            [Validate action]
            - Type check vs actuator schema
            - Clamp to bounds
            - Check mutation budget
                    |
                    v
            [Apply via commands JSONL]
                    |
                    v
            [EffectTracker records baseline]
                    |
                    v (after cooldown)
            [Evaluate outcome: improved/degraded/neutral]
                    |
                    v
            [Write to hotcb.applied.jsonl (ledger)]
            - Includes rule_id and source="autopilot" for autopilot-applied actions
```

---

## Configuration Reference

### AutopilotConfig (in DashboardConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `divergence_threshold` | float | 2.0 | Default metric increase threshold for divergence detection |
| `ratio_threshold` | float | 0.5 | Default train/val ratio threshold for overfitting detection |
| `ai_min_interval` | int | 10 | Minimum steps between LLM invocations |
| `ai_max_wait` | int | 200 | Maximum steps before forcing a periodic LLM check |
| `ai_default_cadence` | int | 50 | Default periodic LLM invocation interval |

Env vars: `HOTCB_DIVERGENCE_THRESHOLD`, `HOTCB_RATIO_THRESHOLD`, `HOTCB_AI_MIN_INTERVAL`, `HOTCB_AI_MAX_WAIT`, `HOTCB_AI_DEFAULT_CADENCE`.

### AIConfig (LLM engine)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | str | `"openai"` | LLM provider (any OpenAI-compatible API) |
| `model` | str | `"gpt-4o-mini"` | Model identifier |
| `api_key` | str | `""` | API key (falls back to `HOTCB_AI_KEY` env var) |
| `base_url` | str | `"https://api.openai.com/v1"` | API base URL |
| `temperature` | float | 0.3 | Sampling temperature |
| `max_tokens` | int | 1024 | Max response tokens |
| `cadence` | int | 50 | Default periodic cadence (steps) |
| `budget_cap` | float | 5.0 | USD spending limit |
| `max_runs` | int | 3 | Max training reruns before AI stops proposing |

### AIState (persisted in `hotcb.ai.state.json`)

| Field | Description |
|-------|-------------|
| `key_metric` | Primary optimization target (e.g., `"val_loss"`) |
| `key_metric_mode` | Direction: `"auto"`, `"min"`, or `"max"` |
| `watch_metrics` | Additional metrics under close monitoring |
| `run_number` | Current run number (increments across restarts) |
| `run_history` | Previous run summaries with learnings |
| `carried_context` | Context string carried to next run |
| `next_check_step` | AI-requested next invocation step |
| `cadence_override` | AI-requested cadence change |
| `outcome_streak` | Recent action outcomes for adaptive cadence |

### AutopilotEngine constructor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mutation_budget` | 10 | Max mutations per window |
| `budget_window_steps` | 200 | Sliding window size for budget |

---

## Research Integration

When the research module is active, the autopilot automatically creates "discovered" hypothesis nodes in the research graph whenever a rule fires:

- **Auto-discovery**: `ResearchEngine.on_rule_fired(action, step)` creates a `HypothesisNode(status="discovered", source="auto_rule:{rule_id}")`. Deduplicates by `(condition, intervention.params)`.
- **Evidence collection**: `ResearchEngine.on_effect_completed(effect, step)` creates `EvidenceNode` from EffectTracker outcomes, updates hypothesis confidence as `supports / (supports + contradicts)`.
- **NN predictions**: When `nn_mode=True`, the autopilot can query `OutcomePredictor.predict(intervention, context)` before applying an action, adding NN confidence as a signal.

The Research tab in the dashboard shows all autopilot-created hypotheses alongside manually created ones in the interactive tree visualization.

## Rule Calibration (Planned)

The 38 numeric knobs across all 5 policy packs (thresholds, multipliers, windows, cooldowns) can be auto-tuned via Bayesian optimization over scenario outcomes:

```bash
hotcb autopilot calibrate --pack stability_basics --trials 100
hotcb autopilot calibrate --compare           # original vs calibrated thresholds
```

This reuses the tune module's Optuna infrastructure (TPE sampler, segment scoring, EMA-based recipe evolution). See the `rule-calibration` stream in STREAMS.md.

---

## Related Documentation

- [Policy Pack Reference](policy_packs.md) -- rule catalog, YAML DSL, custom authoring
- [Scenario Catalog](scenarios.md) -- 12 demoable scenario tests for all 5 policy packs
- [Guarantee Envelope](guarantee_envelope.md) -- what autopilot guarantees and does not guarantee
