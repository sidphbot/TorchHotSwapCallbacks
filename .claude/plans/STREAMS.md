# Streams

> **Protocol:** claim a stream (`status → active`, note your branch), work,
> update checkboxes + log, release when done (`→ done`) or paused (`→ planned`).
> New stream: add a section + row to table. Use `/stream` to browse/attach.
> Use `/stream branch <name>` to import an existing git branch as a stream.

| ID | Type | Pri | Status | Branch | Summary |
|----|------|-----|--------|--------|---------|
| v2-stabilization | chore | p0 | active | main | Post-2.0 stabilization: maintenance fixes, demo restructure, dashboard UX, docs, coordination |
| observability-v2 | feature | p0 | done | autopilot-spec-v1 | Layer 1: health signals, conflict scores, derived state labels, grad norms |
| policy-packs | feature | p0 | done | autopilot-spec-v1 | Layer 2: default guidelines, policy packs (stability/multi-loss/distill/plateau/finish), YAML DSL |
| execution-safety | feature | p1 | done | autopilot-spec-v1 | Layer 3: explicit rollback, action feedback loop, bounds enforcement on rules, transactional apply |
| actuator-expand | feature | p1 | done | autopilot-spec-v1 | New actuator families: freeze/unfreeze, grad clipping, SWA/EMA, curriculum basics |
| ai-feedback-loop | feature | p1 | done | autopilot-spec-v1 | AI learns from action effects, cross-run learning, adaptive cadence |
| fix-error-handling | fix | p2 | done | autopilot-spec-v1 | Replace bare `except: pass` with specific types + logging |
| fix-api-consistency | refactor | p3 | planned | — | Standardize REST responses to `{status, data?, error?}` |
| fix-adapter-imports | fix | p3 | done | autopilot-spec-v1 | Gate lightning/hf imports with friendly ImportError |
| feature-test-coverage | test | p1 | done | autopilot-spec-v1 | Integration tests for demos, launcher, dashboard E2E |
| docs-examples-refresh | docs | p2 | planned | — | Verify examples, add notebooks |
| docs-autopilot-spec | docs | p1 | done | autopilot-spec-v1 | Autopilot product spec, policy pack docs, guarantee envelope definition |
| chore-release-prep | chore | p1 | planned | — | sdist/wheel validation, changelog, PyPI publish |

Dependencies:
- `policy-packs` depends on `observability-v2` (packs need health signals to trigger on)
- `ai-feedback-loop` depends on `execution-safety` (feedback needs action tracking)
- `chore-release-prep` blocks on `v2-stabilization` + `feature-test-coverage`
- `actuator-expand` is independent, can run in parallel with observability/policy work

---

## v2-stabilization
**Goal:** Full post-2.0 stabilization pass — audit, fix, restructure, document.
**Branch:** `main` (merged from `claude_skill` + `big_maintainance_round`)
**Scope:**
- MAINTENANCE.md P0-P2 audit and fixes (28 items fixed)
- Demo restructuring to HotKernel integration path
- Dashboard UX (controls hydration, stale data, run dir backup)
- Docs cleanup (INTEGRATION.md, concepts.md, custom_training_configs.md)
- Claude Code skill (`.claude/skills/hotcb-autopilot/`)
- Multi-agent coordination system (`.claude/plans/`, `/stream` command)

**Done:**
- [x] MAINTENANCE.md audit — all P0 user-reported (1.1-1.6)
- [x] Frontend P1 fixes (API error handling, WS backoff, listener leaks, Three.js, tooltip colors, forecast polling, intervals)
- [x] Backend P1 fixes (malformed JSON, JSONL locking, FreezeState validation, duplicate imports, deps)
- [x] Packaging P2 (MANIFEST.in, py.typed)
- [x] Accessibility P2 (focus styles, ARIA labels)
- [x] Demo restructuring — 3 demos rewritten to HotKernel + MetricsCollector + actuators
- [x] Docs — fixed mc.log() refs, updated framework examples, deleted legacy examples/
- [x] NaN/inf, [object Object], chart waiting, staged knob highlights, sys import
- [x] Claude Code skill (SKILL.md with 5-phase autopilot protocol)
- [x] Dashboard slider sync from WS metrics, `_slidersInitialized`
- [x] Launcher run dir backup (`_backup_run_dir_if_needed`)
- [x] Launcher JSONL truncation on start (was skip-if-exists)
- [x] `weight_decay` added to demo metrics
- [x] Multi-agent coordination (STREAMS.md + /stream command)
- [x] Unified actuator model — single `HotcbActuator` + `MutableState` replaces old opt/loss modules
- [x] Dashboard: pin buttons always visible in metrics dropdown
- [x] Dashboard: line connector fix (sort by step before render)
- [x] Dashboard: WS initial burst skip for metrics (REST loads full history)
- [x] Dashboard: controls from `hotcb.capabilities.json` fallback (5-tier chain)
- [x] Dashboard: controls polling until real controls appear
- [x] Dashboard: metrics dropdown search bar
- [x] Dashboard: show all metrics by default (was filtered to "common" >20)
- [x] Dashboard: control slider layout — label on own line for long param names
- [x] Dashboard: slider delta overlay (red gradient + tick mark for applied baseline)
- [x] Dashboard: default pinned metric cards (train_loss, val_loss, key_metric)
- [x] Merged to main

**Remaining:**
- [ ] Manual verify: sliders sync on each demo config
- [ ] Manual verify: no stale timeline on restart
- [ ] Manual verify: backup dir created on re-run

**Log:**
- 2026-03-12: MAINTENANCE.md audit, P0-P2 fixes, demo restructuring, docs cleanup
- 2026-03-13: Dashboard UX fixes (slider sync, stale data, backup). Plans system created. 754 tests pass.
- 2026-03-15: Unified actuator model merged. Dashboard fixes for serve mode (capabilities fallback, line connectors, WS burst, pin buttons). Merged to main. 856 tests pass.
- 2026-03-16: Metrics dropdown search bar, show all metrics, control slider layout, delta overlay, default pins.

---

## observability-v2
**Goal:** Layer 1 — Enrich the training state object with health signals the policy engine and AI need.
**Scope:** Extend `autopilot.py` + `ai_prompts.py` + `metrics/` to emit structured health state.

**Tasks:**
- [x] **Gradient health signals**: Extract grad_norm, grad_norm_per_group, clipping_rate from metrics JSONL; add to trend compression
- [x] **Multi-loss conflict detection**: Compute per-loss gradient norms, dominance ratios, pairwise conflict score from loss breakdown metrics; emit `conflict_score` derived metric
- [x] **Derived state labels**: Add state classifier that emits labels from trend data: `stable-improving`, `stable-plateaued`, `aux-conflicted`, `numerically-unsafe`, `oscillatory`, `converged-likely`, `collapse-risk`
- [x] **Health state object**: Create `TrainingHealthState` dataclass aggregating all signals; exposed at `/api/state/health`
- [x] **Numeric stability signals**: NaN/Inf counters, loss spike detection, AMP scaler stats (when available in metrics)
- [x] **Plateau/oscillation refinement**: Make existing `_eval_plateau` and `_eval_divergence` richer — multiple windows, trend persistence
- [x] **Dashboard health panel**: Wire health state labels into dashboard health card (currently only EMA score)

**Non-goals (v2 scope):**
- Representation health (feature stats, activation analysis) — deferred, needs FeatureCapture integration
- Batch-level variance — needs per-batch hooks not yet in MetricsCollector

**Files:** `server/autopilot.py`, `server/ai_prompts.py`, `metrics/collector.py`, `server/app.py`

---

## policy-packs
**Goal:** Layer 2 — Ship default rule packs and a YAML DSL for user-authored policies. This is the biggest product differentiator.
**Depends on:** `observability-v2` (packs trigger on health signals)

**Tasks:**
- [x] **Default guidelines YAML**: Create `server/guidelines/` directory with built-in rule packs as YAML files
- [x] **Pack 1 — Stability Basics**: NaN/Inf guard, gradient spike clipping, LR emergency reduction, mutation rollback
- [x] **Pack 2 — Multi-Loss Assist**: Aux warmup ramp, loss ratio targeting, conflict mitigation (reduce conflicting weight), aux rollback on instability
- [x] **Pack 3 — Distillation Assist**: Summary-first warmup, spatial-loss delayed ramp, top-k scheduling, feature variance health checks
- [x] **Pack 4 — Plateau Recovery**: Stagnation detection, LR restart/decay strategy, safe schedule switch, conservative finish mode
- [x] **Pack 5 — Finish Strong**: SWA enable, EMA enable, best-window checkpointing trigger, mutation lock-down in late training
- [x] **Rule DSL enrichment**: Add `bounds`, `rollback_if`, `priority` fields to `AutopilotRule`; add rule composition (if A fired → suppress B)
- [x] **Pack loading API**: `POST /api/autopilot/packs/load` with pack name; `GET /api/autopilot/packs` lists available packs
- [x] **Dashboard pack selector**: UI to browse/enable/disable packs
- [x] **Rule bounds enforcement**: Validate rule action params against actuator bounds (same validation AI actions get)
- [x] **Rule conflict detection**: Detect when two rules propose conflicting actions in same cooldown window

**Files:** `server/guidelines/*.yaml`, `server/autopilot.py`, `server/app.py`, `server/static/js/panels.js`

---

## execution-safety
**Goal:** Layer 3 — Strengthen the mutation execution path with rollback, feedback, and transactional semantics.
**Depends on:** None (can start in parallel)

**Tasks:**
- [x] **Explicit rollback command**: Add `rollback` op to kernel — restores last snapshot from MutableState
- [x] **Rollback API endpoint**: `POST /api/rollback` with optional `snapshot_id`
- [x] **Action effect tracking**: After mutation, record metric deltas over next N steps; store in ledger as `observed_effect`
- [x] **Rollback triggers**: Auto-rollback if key metric degrades by > X% within N steps of mutation
- [x] **Transactional multi-param apply**: Group related param changes into a single snapshot/apply/verify cycle
- [x] **Rule bounds enforcement**: Rules go through same `validate()` path as AI actions before execution
- [x] **Mutation budget**: Configurable max mutations per N steps; enforce across rules + AI
- [x] **Dashboard rollback button**: Per-timeline-item rollback action in mutation timeline

**Files:** `kernel.py`, `actuators/state.py`, `server/api.py`, `server/autopilot.py`, `server/static/js/panels.js`

---

## actuator-expand
**Goal:** Expand actuator families beyond optimizer/loss to cover the spec's full actuator surface.
**Independent — can run in parallel.**

### A. Model actuators
- [x] **Freeze actuators**: `freeze_module(name)` / `unfreeze_module(name)` — BOOL actuator per freezeable group, `apply_fn` calls `module.requires_grad_(bool)` on named children
- [x] **Convenience constructor**: `model_actuators(model, groups={"trunk": ["encoder.*"], "head": ["decoder.*"]})` — user defines named groups by module name patterns, returns `List[HotcbActuator]`
- [x] **Lightning auto-discover**: Adapter introspects `pl_module.named_children()`, generates freeze actuators for top-level groups (e.g. `freeze_encoder`, `freeze_decoder`)
- [x] **Gradient clipping actuator**: `grad_clip_value` FLOAT actuator; Lightning adapter reads existing `trainer.gradient_clip_val` as initial, `apply_fn` mutates `trainer.gradient_clip_val`
- [x] **SWA/EMA actuators**: `enable_swa` / `enable_ema` BOOL actuators; `apply_fn` wraps model with `torch.optim.swa_utils.AveragedModel` on enable, unwraps on disable

### B. Data / Curriculum actuators — hook philosophy
Same pattern as everything else: work with available hooks, never own the dataloader.

**Design:**
- `data_actuators(dataset_or_loader, attrs={...})` — convenience constructor, maps named attributes to actuators
- Each actuator's `apply_fn` is a `setattr` closure on the captured object
- User declares what's mutable — hotcb doesn't inspect internals

**Adapter auto-discovery (Lightning):**
- Lightning adapter checks `trainer.train_dataloader` → `dataset` for known mutable attrs
- Common patterns detected: `augmentation_strength`, `curriculum_stage`, `difficulty`, `sample_weights`, `mix_ratio`
- Detected attrs become FLOAT/INT/CHOICE actuators with sensible defaults

**Bare PyTorch — `HotDataKernel`:**
- Lightweight wrapper: `HotDataKernel(dataset, mutable_attrs={"aug_strength": {"type": "float", "min": 0, "max": 1}})`
- Does NOT wrap the dataloader — just registers actuators for declared attrs
- Returns `List[HotcbActuator]` that user passes to `mutable_state()` alongside other actuators
- Alternatively: user's dataset just exposes attrs, user calls `data_actuators(my_dataset, attrs=...)` directly — `HotDataKernel` is sugar

**Concrete actuator families:**
- [x] **Augmentation severity**: FLOAT `aug_strength` — `setattr(dataset, attr, value)`; works with any dataset that has a mutable strength param
- [x] **Curriculum stage**: INT or CHOICE `curriculum_stage` — controls which data subset / difficulty tier is active
- [x] **Sample weights / mix ratio**: FLOAT `mix_ratio` — for blended datasets (e.g. real vs synthetic ratio)
- [x] **Hard case reweighting**: FLOAT `hard_case_weight` — downweight difficult samples temporarily during instability
- [x] **Augmentation policy**: CHOICE `aug_policy` — switch between augmentation presets

**Implementation:**
- [x] `data_actuators()` in `actuators/__init__.py` — maps attr names to actuators via setattr closures
- [x] `HotDataKernel` class in new `actuators/data.py` — convenience wrapper with type inference
- [x] Lightning adapter: auto-discover mutable dataset attrs in `_detect_capabilities` + `_wire_mutable_state`
- [x] HF adapter: same pattern via `trainer.train_dataset`
- [x] Capabilities reporting: extend `hotcb.capabilities.json` with `data_actuator_keys` field
- [x] Dashboard: data actuators appear in controls panel under "data" group

**Philosophy:**
- If the dataset has the attr → we can control it
- If it doesn't → user adds it to their dataset class, then registers with `data_actuators()`
- No magic introspection of transforms pipelines — user declares the interface
- Same as loss_actuators: captures the dict/object reference, closure mutates it

### C. Safety actuators
- [x] **Safe mode actuator**: BOOL `safe_mode` — when enabled, restricts mutation aggressiveness (halves all proposed deltas)
- [x] **Mutation lock**: BOOL `mutation_locked` — when enabled, no new mutations accepted (useful for late-training stability)

### D. Capabilities reporting
- [x] Extend `TrainingCapabilities` with `freezeable_groups`, `data_actuator_keys`, `grad_clip_available`, `swa_available`
- [x] Extend `hotcb.capabilities.json` schema
- [x] Dashboard config: generate controls from new capability fields

**Files:** `actuators/__init__.py`, `actuators/data.py` (new), `actuators/actuator.py`, `actuators/state.py`, `adapters/lightning.py`, `adapters/hf.py`, `capabilities.py`, `kernel.py`

---

## ai-feedback-loop
**Goal:** Close the loop — AI learns whether its actions helped, adapts cadence, uses cross-run history.
**Depends on:** `execution-safety` (needs action effect tracking)

**Tasks:**
- [x] **Action outcome reporting**: After cooldown window, compute metric delta and classify outcome (improved/neutral/degraded); include in next AI prompt as "recent action outcomes"
- [x] **Cross-run context synthesis**: On run start, summarize previous run's learnings into AI system prompt; weight recent runs higher
- [x] **Adaptive cadence**: Track AI decision quality; increase check frequency when actions are helping, decrease when neutral
- [x] **Confidence calibration**: Track AI's stated confidence vs actual outcomes; report calibration score
- [x] **Failure recovery**: Retry LLM call once on timeout/error; fallback to rule-based evaluation if AI unavailable
- [x] **Mode auto-progression**: Start in suggest, auto-promote to auto after N consecutive successful suggestions accepted

**Files:** `server/ai_engine.py`, `server/ai_prompts.py`, `server/autopilot.py`

---

## fix-error-handling
**Goal:** Replace silent `except Exception: pass` with specific types + `log.warning()`.
**Files:** `cli.py`, `kernel.py`, `recipe.py` (audit for others)
- [x] Audit all bare-except sites
- [x] Replace with specific exceptions + logging
- [x] Verify tests pass

---

## fix-api-consistency
**Goal:** Unify REST responses to `{status, data?, error?}`. Breaking change — needs frontend + SKILL.md updates.
**Files:** `api.py`, `utils.js`, `controls.js`, `init.js`, `panels.js`, `SKILL.md`, `INTEGRATION.md`
- [x] Catalog current response shapes
- [x] Design envelope schema
- [x] Update backend + frontend + docs

---

## fix-adapter-imports
**Goal:** Friendly error when `pytorch_lightning` / `transformers` not installed.
**Files:** `adapters/lightning.py`, `adapters/hf.py`
- [x] Wrap imports in try/except with install instructions
- [x] Add test for the friendly error message

---

## feature-test-coverage
**Goal:** Integration tests for demo→launcher→dashboard→stop cycle. Currently 856 unit tests, zero integration.
**Files:** `src/hotcb/tests/`
- [x] Test demo functions: run 10 steps, check metrics JSONL fields
- [x] Test launcher lifecycle: start → status → stop → reset
- [x] Test run dir backup: existing data → start → verify backup
- [x] Test `/api/state/controls` returns correct values
- [x] Test WS initial data burst

---

## docs-examples-refresh
**Goal:** Verify `docs/examples/` match v2.0, add Jupyter notebooks.
**Files:** `docs/examples/*.py`
- [ ] Verify 3 existing examples run
- [ ] Create notebook using `launch()` API
- [ ] Create notebook for autopilot comparison

---

## docs-autopilot-spec
**Goal:** Formalize the autopilot product spec — guarantee envelope, supported problem classes, policy pack docs.
- [x] Write `docs/autopilot.md` — 3-layer architecture, modes, guarantee envelope
- [x] Write `docs/policy_packs.md` — pack catalog, YAML DSL reference, custom rule authoring guide
- [x] Write `docs/guarantee_envelope.md` — what "convergence assist" means, supported surfaces, constraints
- [x] Update README with autopilot positioning

---

## chore-release-prep
**Goal:** PyPI 2.0.0 publish. Depends on `v2-stabilization` + `feature-test-coverage`.
- [ ] Merge fix branches to main
- [ ] Build sdist + wheel, verify static files included
- [ ] Test install in fresh venv, run `hotcb demo`
- [ ] Write CHANGELOG.md
- [ ] Tag + publish
