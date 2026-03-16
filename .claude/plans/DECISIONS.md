# Decision Log

Records of architectural and implementation decisions made during autonomous execution.

---

## Phase 1: Observability v2

### D1.1 — TrainingHealthState location
**What:** Where does `TrainingHealthState` live?
**Decision:** New `src/hotcb/health.py` module.
**Why:** Keeps autopilot.py focused on rule evaluation. Health computation is domain logic, not server logic.
**Alternatives:** Could live in `server/autopilot.py` (rejected — too coupled) or `metrics/` (rejected — health is derived state, not raw metrics).

### D1.2 — Signal sources
**What:** What signals are computable from existing metrics JSONL?
**Decision:** grad_norm, per-loss values, total loss trend are all already available if user logs them. conflict_score is derived from loss trends, not logged directly.
**How:** `compute_health_state()` takes `{metric_name: [{step, value}]}` dict and computes all signals from the last `window` points.

### D1.3 — Health state flow
**What:** How does health state flow through the system?
**Decision:** metrics → TrendCompressor-style math → HealthState → autopilot rules + AI prompts.
**How:** Reuse linear regression logic from TrendCompressor. Extract shared math if needed into health.py directly (no separate math_utils.py unless clearly warranted).

---

## Phase 2: Execution Safety

### D2.1 — Rollback granularity
**What:** Per-actuator snapshot stack vs whole-state snapshot?
**Decision:** Whole-state snapshot stack via existing `snapshot_all()`/`restore_all()`.
**Why:** Simpler, already implemented in MutableState. Per-actuator would add complexity without clear benefit.

### D2.2 — Snapshot retention
**What:** Keep last N snapshots?
**Decision:** Yes, configurable `max_snapshots` (default 10).
**Why:** Unbounded stack is a memory risk for long training runs.

### D2.3 — Action effect tracking location
**What:** Where to store effect tracking?
**Decision:** New `EffectTracker` class, either in `health.py` or `tracking.py`.
**Why:** Effect tracking is about observing mutation outcomes — related to health but distinct enough for its own class.

### D2.4 — Mutation budget scope
**What:** Per-step or rolling window?
**Decision:** Rolling window (last N steps). Default: 10 mutations per 200 steps.
**Why:** Per-step is too restrictive. Rolling window smooths burst mutations while preventing runaway intervention.

---

## Phase 3: Actuator Expansion

### D3.1 — Model actuator traversal
**What:** Use `named_modules()` vs `named_children()`?
**Decision:** `named_modules()` for deeper group matching with regex patterns.
**Why:** `named_children()` only gets top-level; real models have nested encoder.layer1.attn etc.

### D3.2 — Data actuator pattern
**What:** How do data actuators work?
**Decision:** `setattr` closure pattern, identical to loss_actuators approach.
**Why:** Consistency with existing actuator system. No new patterns needed.

### D3.3 — Safety actuators scope
**What:** Where do safety actuators live?
**Decision:** In `actuators/__init__.py` alongside other convenience constructors.
**Why:** They're small (just two BOOL actuators), don't warrant a separate file.

### D3.4 — All actuators follow same pattern
**What:** Convenience constructor returns `List[HotcbActuator]`.
**Decision:** Yes, every new actuator family follows exact same pattern as `optimizer_actuators()`.
**Why:** Consistency. Users learn one pattern, framework code stays uniform.

### D3.5 — HotDataKernel
**What:** Is HotDataKernel a new module type?
**Decision:** No. It's sugar over `data_actuators()` with auto-discovery of known attribute names.
**Why:** Avoids adding a new kernel module type. The actuator system is the right abstraction.

---

## Phase 4: Policy Packs

### D4.1 — Pack storage format
**What:** Where do policy packs live?
**Decision:** YAML files in `server/guidelines/` directory, shipped with the package.
**Why:** Data, not code. YAML is human-readable, editable, shareable. Same directory as existing default.yaml.

### D4.2 — Pack loading semantics
**What:** How are packs loaded and identified?
**Decision:** `load_pack(name)` reads `guidelines/{name}.yaml`, prefixes all rule IDs with `{name}.` to avoid collisions. Multiple packs can be loaded simultaneously.
**Why:** Prefixing prevents ID collisions. Multiple packs enable composability (e.g., stability + multi-loss).

### D4.3 — Rule conflict resolution
**What:** How to handle two rules firing for the same actuator?
**Decision:** Priority ordering (low < medium < high < critical). Higher priority wins within the same cooldown window. `suppress_rules` field for explicit suppression.
**Why:** Simple, predictable. Users can understand which rule will win.

### D4.4 — Pack YAML format
**What:** What metadata does a pack carry?
**Decision:** `name`, `description`, `version`, `requires` (actuator families needed). `requires` is advisory (warning, not blocking).
**Why:** Metadata enables UI display. `requires` helps users understand what actuators a pack needs.

---

## Phase 5: AI Feedback Loop

### D5.1 — Action outcomes in prompt
**What:** How does AI learn from past actions?
**Decision:** Include last 5 completed effects from EffectTracker as "Recent Action Outcomes" table in AI prompt user message.
**Why:** Direct feedback loop. AI sees what worked and what didn't, can adjust strategy.

### D5.2 — Cross-run synthesis
**What:** How to carry context across runs?
**Decision:** AIState.run_history already exists. Synthesize last 3 runs into system prompt with run number, steps, final key metric, learnings, verdict.
**Why:** Multi-run learning is already structured in AIState; just needs prompt integration.

### D5.3 — Adaptive cadence mechanism
**What:** How to adjust AI check frequency?
**Decision:** Track outcome streak. 3+ improved → halve interval (min 50). 3+ neutral → double (max 500). Any degraded → reset to base.
**Why:** Simple heuristic. When AI is helping, check more often. When neutral, save API costs.

### D5.4 — Failure recovery strategy
**What:** What happens when LLM call fails?
**Decision:** Single retry with 5s backoff. On second failure, fallback to rules-only for N steps (default 100), then re-enable.
**Why:** Graceful degradation. Training shouldn't stall because an API is down.

### D5.5 — Auto-progression
**What:** Should suggest mode auto-upgrade to auto?
**Decision:** Opt-in via config (`auto_progress_enabled=False`). After 5 consecutive accepted suggestions, upgrade to ai_auto.
**Why:** Builds trust gradually. Disabled by default so users opt in explicitly.
