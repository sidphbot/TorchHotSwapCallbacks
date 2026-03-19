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
| eval-real-pytorch | feature | p0 | done | autopilot-spec-v1 | Real PyTorch eval (MNIST/CIFAR-10/ImageNet/COCO), continuation tuning, dashboard UX, research tab |
| dashboard-e2e-coverage | test | p0 | planned | — | Comprehensive Playwright E2E tests for every dashboard interaction (~150 tests) |
| demo-isolation | refactor | p0 | planned | — | Extract demos into standalone external-style projects, zero internal imports, proper INTEGRATION.md contract |
| checkpoint-scenarios | feature | p0 | planned | — | Verifiable real-training scenarios with pretrained checkpoints — deterministic resume, data curriculum, policy pack validation |
| rule-calibration | feature | p1 | planned | — | Auto-tune autopilot rule thresholds via Bayesian optimization over scenario outcomes, reusing tune module's Optuna infrastructure |
| compare-eval-perf | refactor | p1 | planned | — | Compare/eval tab run discovery & load optimization — structural changes to save/load/discovery |
| serve-vs-launch | design | p1 | planned | — | Serve (hooks) vs Launch (encapsulate): clarify architecture, decide if both modes needed |

Dependencies:
- `policy-packs` depends on `observability-v2` (packs need health signals to trigger on)
- `ai-feedback-loop` depends on `execution-safety` (feedback needs action tracking)
- `chore-release-prep` blocks on `v2-stabilization` + `feature-test-coverage`
- `actuator-expand` is independent, can run in parallel with observability/policy work
- `dashboard-e2e-coverage` is independent, can run any time — blocks `chore-release-prep`
- `demo-isolation` blocks `docs-examples-refresh` (examples should use the new isolated demo pattern)
- `checkpoint-scenarios` depends on `demo-isolation` (MNIST/CIFAR-10 must follow INTEGRATION.md before checkpointing)
- `rule-calibration` depends on `checkpoint-scenarios` (needs real scenario outcomes as objective function)

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

## eval-real-pytorch
**Goal:** Replace synthetic demo eval conditions with real PyTorch training (MNIST/CIFAR-10). Fix research tab rendering. Dashboard UX for eval workflow.
**Branch:** `worktree-agent-a398a776`
**Depends on:** None (standalone)

### A. Real PyTorch Training Tasks
- [x] **MNIST small CNN** (`src/hotcb/eval/tasks.py`): `_MnistCNN` ~14K params, Conv→Pool→Conv→Pool→FC, Adam lr=1e-3. 3 epochs (~1500 steps), ~1-2min GPU.
- [x] **CIFAR-10 small CNN** (`src/hotcb/eval/tasks.py`): `_CifarCNN` ~62K params, Conv→BN→Pool→Conv→BN→Pool→FC→FC, SGD lr=0.01 mom=0.9. 5 epochs (~2000 steps), ~2-3min GPU.
- [x] **HotKernel integration**: Both tasks wire up MetricsCollector + optimizer_actuators + MutableState (same path as demos)
- [x] **Real metrics**: train_loss, val_loss, val_accuracy, grad_norm, grad_norm_ema, lr, weight_decay
- [x] **Final validation**: kernel.apply() with val_loss/val_accuracy at end of run (not just kernel.close())

### B. Eval Conditions
- [x] **5 MNIST conditions**: baseline, high_lr_auto (lr=0.03), high_lr_no_auto, high_wd_auto (wd=0.1), divergent_lr_auto (lr=0.05)
- [x] **5 CIFAR-10 conditions**: baseline, high_lr_auto (lr=0.15), high_lr_no_auto, divergent_lr_auto (lr=0.2), high_wd_auto (wd=0.01)
- [x] **Empirically validated thresholds**: MNIST lr=0.03→grad~24 triggers grad_spike_clip, lr=0.05→grad~93 triggers emergency
- [x] **Removed low_lr conditions**: Plateau detection doesn't work with per-batch noise (range>>epsilon=0.002)
- [x] **No-autopilot control conditions**: A/B comparison (same bad params, autopilot=off)
- [x] **Custom YAML conditions**: `--conditions-file` loads user-defined conditions from YAML
- [x] **Default suite is real-only**: `ALL_CONDITIONS = REAL_CONDITIONS`
- [x] **Synthetic conditions accessible**: `--demo golden/finetune/simple` or `--include-synthetic` flag
- [x] **Harness updated**: `_get_train_fn()` supports "mnist" and "cifar10" demo types
- [x] **CLI updated**: `--demo` choices include mnist/cifar10, `--max-steps` defaults to None (task-specific)

### C. Research Tab Fix
- [x] **Cytoscape zero-dimension fix**: Container now has `width:100%;position:relative;min-height:400px`
- [x] **Deferred init**: `initResearchTab()` checks `getBoundingClientRect()` > 10px before creating Cytoscape; retries if container not visible
- [x] **Resize on tab switch**: `cy.resize()` + `cy.fit()` after layout settles (500ms timeout)
- [x] **HTML nesting fix**: Compare tab-content missing `</div>` — research was nested inside compare, always invisible
- [x] **Verified via Playwright**: Research tab renders Cytoscape graph, stats populate, eval banner shows

### D. Dashboard UX (eval workflow)
- [x] **Focus button**: Each run in comparison sidebar has Focus button → loads metrics into Metrics tab
- [x] **Focus rewrite**: `_focusRun` now properly updates `S.metricNames`, `S.latestMetrics`, `S.appliedData`, calls `updateMetricToggles()`, `computeHealth()`
- [x] **Race condition guard**: `_autoFocusRun` waits for `window._initialLoadDone` before triggering (prevents clearing data mid-load)
- [x] **Run labels**: `hotcb.run.json` per condition → comparison view shows condition name
- [x] **Evidence outcome**: Real training uses val_accuracy threshold (>0.5 improved, >0.2 neutral, else degraded)
- [x] **Tab layout**: All non-metrics tabs show condensed header only (compare-active-mode for all tabs, not just compare)
- [x] **Metrics blank fix**: Removed parent dir JSONL bootstrap, pre-create first condition's run dir instead
- [x] **Eval status endpoint**: Checks both cli_run_dir (parent) and resolved run_dir
- [x] **Circular import fixes**: xgboost in projections.py and umap/sklearn in manifolds.py — lazy imports
- [x] **Panel loading indicators**: All panels show contextual loading spinners — compare sidebar, compare chart, research graph, manifold, recipe editor, autopilot rules, focus mode
- [x] **Research graph tree redesign**: Cytoscape.js graph rebuilt as tree/mindmap — root "Experiment" → run nodes → hypotheses → evidence. Run nodes clickable to focus metrics. Hover tooltips with confidence/status/NN score. Auto-refresh 8s.
- [x] **Run discovery integration**: Research graph merges `/api/research/graph` + `/api/runs/discover` into unified tree via `buildTreeElements()`
- [ ] **Live progressive flow**: Dashboard starts immediately with untested hypotheses, updates as conditions complete
- [ ] **Graph live update**: Auto-refresh (8s) already in research.js; needs eval harness to write graph incrementally

### E. Convert remaining synthetic tests to real
- [ ] **Map golden conditions to MNIST/CIFAR-10**: high_lr, high_wd, weight_imbalance → equivalent real conditions
- [ ] **Map finetune conditions**: Transfer learning scenario with CIFAR-10 pretrained features
- [ ] **Map SWA/adversarial conditions**: SWA toggle during real training
- [ ] **Remove synthetic conditions from default suite**: Already done — synthetic accessible via `--include-synthetic`

### F. Playwright E2E Tests
- [x] **Test infrastructure**: `tests/conftest.py` with session-scoped server fixture, synthetic eval data generator
- [x] **30 tests passing**: Initial load, tab switching, compare runs, research graph, focus mode, autopilot, config, recipe
- [x] **Dependency**: `pip install -e ".[e2e]"` → playwright + pytest-playwright

**Log:**
- 2026-03-16: Created MNIST/CIFAR-10 tasks, 10 real conditions, updated harness/CLI, fixed research tab init, all 1006 tests pass
- 2026-03-16: Fixed HTML nesting (compare missing </div>), _focusRun rewrite (S.metricNames/appliedData/toggles), race condition guard. Tuned eval conditions (empirically validated LR thresholds). Added custom YAML conditions. Tab layout fix (all non-metrics tabs condensed). Metrics blank fix (parent bootstrap removed). Circular import fixes. 30 Playwright E2E tests, 1157 backend tests pass.
- 2026-03-17: Panel loading indicators (overlay + inline loaders for all data-fetching panels). Research graph tree redesign (root→runs→hypotheses→evidence, click-to-focus, hover tooltips, 8s auto-refresh). Documentation update (README, CLAUDE.md, INTEGRATION.md, all docs/*, SKILL.md, STREAMS.md).
- 2026-03-18: MobileNetV2 paper-faithful training (RMSProp, ExponentialLR, AMP, torch.compile). SSDLite COCO detection. RMSProp alpha/momentum actuators. Continuation tuning module (18 recipes). 40+ eval conditions. ImageNet/COCO datasets. CLIP-COCO VLM task. Dashboard layout fix (premature </div>). Research tab seamless refresh (differential Cytoscape updates, viewport preservation). Layout regression E2E tests.
- 2026-03-19: Documentation updates (CLAUDE.md eval/continuation sections, INTEGRATION.md checkpoint/continuation patterns, README.md eval+continuation sections). Claude Code commit hook (read_doc_gen_commit).

---

## dashboard-e2e-coverage
**Goal:** Comprehensive Playwright E2E test coverage for every dashboard interaction. The UI is fragile and the main blocker to product quality — track coverage separately as a first-class concern.
**Branch:** —
**Depends on:** None (independent)
**Current state:** 30 tests in `tests/test_dashboard_e2e.py` covering Tier 1 (initial load, tabs, compare, research, focus). Target: ~150 tests.

### Tier 1 — Critical Path (30/30 done)
- [x] Initial load: chart renders, toggles populate, step counter, WS status, no JS errors
- [x] Tab switching: all 7 tabs visible, compare-active-mode toggle on/off
- [x] Compare tab: runs listed, Focus button, focus switches to metrics, run selection toggle
- [x] Research tab: Cytoscape canvas, stats populated, eval banner, layout selector, stream filter
- [x] Focus mode: metrics loaded into S.metricsData/metricNames, toggles updated
- [x] Autopilot rules tab: container visible, add button exists
- [x] Config wizard tab: form inputs visible
- [x] Recipe tab: toolbar buttons visible

### Tier 2 — Metric Toggles & Chart Interactions (~20 tests)
- [ ] Metric dropdown: open/close on button click, close on outside click
- [ ] Metric dropdown: search filter narrows list, clears on empty
- [ ] Metric dropdown: "All On" / "All Off" buttons toggle all metrics
- [ ] Metric dropdown: click metric dot toggles visibility, chart updates
- [ ] Metric dropdown: pin icon creates metric card, double-click dot pins
- [ ] Chart: normalization button toggles Y-axis scale
- [ ] Chart: step range buttons (All, Last 200, Last 500) filter data
- [ ] Chart: custom range inputs (min/max + Go) set range
- [ ] Chart: mutation annotations visible (vertical dashed lines)
- [ ] Chart: waiting overlay shown when no data, dismissed on first metrics

### Tier 3 — Knob Panel & Apply (~15 tests)
- [ ] Knobs: slider renders per actuator with label + value display
- [ ] Knobs: slider change updates value text and marks row as staged
- [ ] Knobs: staged row gets red delta overlay on slider track
- [ ] Knobs: Apply button sends correct API calls (opt/loss grouped)
- [ ] Knobs: Apply debounce prevents duplicate submissions
- [ ] Knobs: bool toggle (checkbox) marks staged
- [ ] Knobs: choice select marks staged
- [ ] Knobs: Schedule modal opens, fields populate, submit works

### Tier 4 — Autopilot Mode & Status (~15 tests)
- [ ] Mode select: changing mode calls POST /api/autopilot/mode
- [ ] Status polling: action list populates with recent actions
- [ ] Suggest mode: proposed actions show "Apply" button
- [ ] Accept action: click Apply on proposed → badge turns green
- [ ] AI modes: AI config section visible with key metric selector
- [ ] AI key metric: select change calls POST /api/autopilot/ai/key_metric
- [ ] AI status: cost info, call count, reasoning panel update

### Tier 5 — Training Launcher (~10 tests)
- [ ] Config select populated from GET /api/train/configs
- [ ] Start button: calls POST /api/train/start with form values
- [ ] Stop button: calls POST /api/train/stop, shows "Stopping..."
- [ ] Reset button: confirm dialog, clears all state
- [ ] Status polling: updates button enabled/disabled states
- [ ] Run summary: auto-popup on training stop with metric table

### Tier 6 — Mutation Timeline & Impact (~10 tests)
- [ ] Timeline item added on WS applied event
- [ ] Click timeline item: highlights annotation on chart
- [ ] Click timeline item: opens impact analysis panel with metric deltas
- [ ] Second click: deselects item, closes impact panel
- [ ] Impact panel: close button dismisses
- [ ] Color coding: cyan=autopilot, orange=manual, yellow=recipe

### Tier 7 — Recipe Editor (~15 tests)
- [ ] Add button opens modal, form submit creates entry
- [ ] Entry list renders with drag handles
- [ ] Inline edit: pencil button opens edit form in place
- [ ] Inline edit: save commits changes, cancel reverts
- [ ] Delete button: removes entry after confirm
- [ ] Drag-and-drop reorder calls POST /api/recipe/move
- [ ] Import modal: paste JSONL or path
- [ ] Export button: triggers download
- [ ] Validate button: shows valid/error alert
- [ ] Diff button: shows +/- colored diff view
- [ ] Replay preview: shows timeline items

### Tier 8 — Autopilot Rules Editor (~10 tests)
- [ ] Add rule modal: form submit creates rule
- [ ] Rule toggle: enable/disable changes icon
- [ ] Inline edit: pencil opens form, save commits
- [ ] Delete: confirm dialog, removes rule
- [ ] Reload defaults: POST /api/autopilot/guidelines

### Tier 9 — Pinned Metric Cards (~8 tests)
- [ ] Pin from dropdown creates card with mini chart
- [ ] Card close button removes card
- [ ] Key metric star: confirm dialog, POST /api/autopilot/ai/key_metric
- [ ] Focus/expand button: toggles metric-focus-mode on body
- [ ] Mini chart shows data matching main chart metric

### Tier 10 — Health, Chat, Callbacks, Alerts (~12 tests)
- [ ] Health panel: score displays, color coded, badges from API
- [ ] Health toggle: collapse/expand details
- [ ] Chat: send message, bot reply rendered
- [ ] Callbacks: load form, submit, refresh list
- [ ] Callbacks: enable/disable/unload buttons
- [ ] Alerts: add rule modal, alert list renders

### Tier 11 — Theme, Freeze, Tour, Persistence, Manifold (~15 tests)
- [ ] Theme select: 4 themes change CSS, persisted in localStorage
- [ ] Freeze select: confirm dialog, API call, unlock button
- [ ] Tour: start button, step through, skip
- [ ] localStorage: pinned metrics restored on reload
- [ ] localStorage: active tab restored on reload
- [ ] Manifold: method selector changes 3D view
- [ ] Compare chart: normalize toggle, zoom mode, overlay info

**Log:**
- 2026-03-16: Created. 30 Tier 1 tests passing. Plan covers ~150 total tests across 11 tiers.

---

## demo-isolation
**Goal:** Extract all demos (simple, golden, finetune, MNIST, CIFAR-10) into standalone external-style projects. Zero `from hotcb.*` internal imports in demo training code. Each demo follows the INTEGRATION.md contract exactly as an external project would — proving the integration story works.
**Branch:** —
**Depends on:** None (independent)

### Problem
Current demos live inside `src/hotcb/` and use internal imports:
- `src/hotcb/demo.py` — imports `HotKernel`, `MetricsCollector`, `optimizer_actuators` directly
- `src/hotcb/golden_demo.py` — same + `loss_actuators`, `swa_actuator`
- `src/hotcb/finetune_demo.py` — same
- `src/hotcb/eval/tasks.py` — MNIST/CIFAR-10 with direct `hotcb.*` imports
- Eval harness (`harness.py`) imports training functions by `from hotcb.golden_demo import _golden_training`

This creates hidden coupling: demos "know" about internal module paths, use private `_OptProxy`, bypass the public API. If a user reads the demo code as a reference, they learn internal patterns instead of the documented integration path.

### Target Architecture
```
examples/                          # top-level, outside src/hotcb/
  simple/
    train.py                       # Option B from INTEGRATION.md
    recipe.jsonl                   # optional scheduled changes
    README.md
  golden/
    train.py                       # multi-task, loss weights, feature capture
    recipe.jsonl
    README.md
  finetune/
    train.py                       # transfer learning, LR schedule
    recipe.jsonl
    README.md
  mnist/
    train.py                       # real PyTorch CNN, Adam, 3 epochs
    README.md
  cifar10/
    train.py                       # real PyTorch CNN, SGD, 5 epochs
    README.md
```

### Principles
1. **Each `train.py` is self-contained** — only `import hotcb` as an installed package, same as any external user
2. **Follows INTEGRATION.md Option B** — `HotKernel` + `MetricsCollector` + actuators
3. **`train_fn(run_dir, max_steps, step_delay, stop_event)` contract** — launchable via `hotcb launch`
4. **Synthetic demos keep `_OptProxy`** — lightweight, no torch needed for dashboard showcase. Real demos (mnist/cifar10) use real torch.
5. **No internal knowledge** — demo code should be copy-pasteable into any external project
6. **README.md per demo** — explains what it demonstrates, how to run standalone
7. **Torch dependency is fine** — product is torch-oriented. Don't constrain design to avoid torch imports.

### Tasks

#### A. Create `examples/` directory structure
- [ ] Create `examples/{simple,golden,finetune,mnist,cifar10}/` directories
- [ ] Each gets a `README.md` with run instructions

#### B. Rewrite synthetic demos as external projects
- [ ] **simple/train.py**: Keep `_OptProxy` pattern — lightweight dashboard showcase. Move to `examples/simple/`, strip internal imports.
- [ ] **golden/train.py**: Same — keep synthetic, move out, strip internals. Multi-task with recipe.
- [ ] **finetune/train.py**: Same — keep synthetic, move out, strip internals. LR schedule via recipe.

#### C. Move real training tasks to examples
- [ ] **mnist/train.py**: Extract from `src/hotcb/eval/tasks.py:mnist_training()` into standalone script. Imports only `hotcb.kernel`, `hotcb.metrics`, `hotcb.actuators`.
- [ ] **cifar10/train.py**: Same extraction from `cifar10_training()`.

#### D. Update eval harness to use external demos
- [ ] `harness.py` loads training functions by path (not `from hotcb.golden_demo import`)
- [ ] `_get_train_fn()` resolves `"examples/mnist/train.py:train"` or `"module:fn"` strings
- [ ] Remove `from hotcb.demo import`, `from hotcb.golden_demo import`, `from hotcb.finetune_demo import`

#### E. Update CLI
- [ ] `hotcb demo` resolves to `examples/simple/train.py` (installed as package data or discovered)
- [ ] `hotcb demo --golden` resolves to `examples/golden/train.py`
- [ ] `hotcb demo --scenario` resolves scenario train scripts the same way
- [ ] Discovery: check `examples/` relative to package, or `--train-fn module:fn` for arbitrary

#### F. Delete internal demo modules
- [ ] Remove `src/hotcb/demo.py`
- [ ] Remove `src/hotcb/golden_demo.py`
- [ ] Remove `src/hotcb/finetune_demo.py`
- [ ] Remove `src/hotcb/eval/tasks.py` (training functions moved to examples)
- [ ] Keep `src/hotcb/eval/conditions.py`, `harness.py`, `report.py` — they're eval infrastructure, not demos

#### G. Verify the isolation
- [ ] Each demo runs standalone: `cd examples/mnist && python train.py`
- [ ] Each demo runs via launch: `hotcb launch --train-fn examples/mnist/train:train`
- [ ] Each demo runs via CLI: `hotcb demo --demo mnist`
- [ ] Eval harness works with external demo paths
- [ ] No `from hotcb.demo` or `from hotcb.golden_demo` anywhere in `src/hotcb/`

#### H. Update docs
- [ ] `docs/examples/` references point to `examples/` directory
- [ ] INTEGRATION.md mentions `examples/` as reference implementations
- [ ] README.md quick-start uses external demo pattern

**Log:**
- 2026-03-16: Created. Current demos use 6 internal import paths. Target: zero internal imports in demo training code.

---

## checkpoint-scenarios
**Goal:** Ship pretrained MNIST/CIFAR-10 checkpoints (model + optimizer state) as test fixtures. Resume training, verify loss matches within tolerance, then apply data/param/policy interventions and verify expected behavior. These are **gold-standard regression tests** for the full autopilot + actuator stack on real training dynamics.
**Branch:** —
**Depends on:** `demo-isolation` (MNIST/CIFAR-10 demos must follow INTEGRATION.md Option B before we checkpoint them)

### Design
- Checkpoints are `.pt` files: `{model_state_dict, optimizer_state_dict, step, last_metrics, config}`
- Small: ~2MB MNIST, ~15MB CIFAR-10 ResNet-20
- Generated once by `scripts/create_checkpoints.py`, stored in `scenarios/checkpoints/` (gitignored)
- Downloaded on first use via `scripts/download_checkpoints.py` (from GitHub releases or local generation)
- Each scenario = resume from checkpoint → verify baseline → apply intervention → verify outcome

### Scenario Inventory

| ID | Dataset | Intervention | Verification | Policy Pack |
|----|---------|-------------|--------------|-------------|
| `determinism_mnist` | MNIST | None (identical config) | `|loss_actual - loss_expected| < 1e-4` for 10 steps | — |
| `determinism_cifar` | CIFAR-10 | None (identical config) | `|loss_actual - loss_expected| < 1e-3` for 10 steps | — |
| `lr_recovery_cifar` | CIFAR-10 | Spike LR 10x via commands.jsonl | `grad_spike_clip` fires, loss recovers within 50 steps | stability_basics |
| `nan_guard_mnist` | MNIST | Inject artificial NaN via extreme LR | `nan_guard` fires, LR halves, loss recovers | stability_basics |
| `data_curriculum_mnist` | MNIST | Switch easy→hard digits via `data_actuators` | Loss spikes temporarily, recovers within 30 steps | — |
| `freeze_unfreeze_cifar` | CIFAR-10 | Freeze conv layers via `model_actuators` | Grad norm drops, only FC params update | — |
| `loss_reweight_mnist` | MNIST | Shift class weights via `loss_actuators` | Per-class accuracy changes accordingly | — |
| `policy_stability_cifar` | CIFAR-10 | Load `stability_basics` + trigger all 4 rules in sequence | Each rule fires at expected condition, no over-correction | stability_basics |
| `plateau_recovery_cifar` | CIFAR-10 | Freeze model briefly → unfreeze → verify plateau detection | `stagnation_detect` fires, LR restart helps | plateau_recovery |
| `multi_loss_conflict` | MNIST (2-head) | Run with conflicting aux loss | `aux_conflict_reduce` fires, conflict_score drops | multi_loss_assist |

### Tasks

#### A. Checkpoint Creation Infrastructure
- [ ] **`scripts/create_checkpoints.py`**: Train MNIST (300 steps) + CIFAR-10 (500 steps), save checkpoint with optimizer state + last metrics
- [ ] **Deterministic seeding**: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic=True`
- [ ] **Checkpoint format**: `{"model_state_dict", "optimizer_state_dict", "step", "last_metrics", "config", "torch_version"}`
- [ ] **`scripts/download_checkpoints.py`**: Download from GitHub releases or regenerate locally
- [ ] **Version pinning**: Record torch version in checkpoint, warn if mismatch on resume

#### B. Scenario Runner (extends existing `src/hotcb/scenarios/runner.py`)
- [ ] **`CheckpointScenario` config**: Extends `ScenarioConfig` with `checkpoint_path`, `baseline_tolerance`, `intervention_step`, `verification_fn`
- [ ] **Resume helper**: `_resume_from_checkpoint(checkpoint_path, model, optimizer)` — loads state, verifies step count
- [ ] **Baseline verification phase**: Run N steps with identical config, assert metrics within tolerance of checkpoint's `last_metrics`
- [ ] **Intervention phase**: At `intervention_step`, apply the scenario's mutation via `hotcb.commands.jsonl`
- [ ] **Outcome verification phase**: Assert expected metric behavior (e.g., loss recovers, rule fires, grad norm drops)

#### C. Implement 10 Scenarios
- [ ] **determinism_mnist**: Resume + 10 steps, `|Δloss| < 1e-4`
- [ ] **determinism_cifar**: Resume + 10 steps, `|Δloss| < 1e-3`
- [ ] **lr_recovery_cifar**: Write `{"module":"opt","op":"set_params","params":{"lr":0.1}}` → verify `grad_spike_clip` fires → loss recovers
- [ ] **nan_guard_mnist**: Extreme LR → NaN → `nan_guard` fires → recovery
- [ ] **data_curriculum_mnist**: Switch dataset difficulty via `data_actuators` → loss spike → adapt
- [ ] **freeze_unfreeze_cifar**: `model_actuators` freeze conv → grad_norm drops → unfreeze → recovers
- [ ] **loss_reweight_mnist**: `loss_actuators` shift class weights → per-class accuracy shifts
- [ ] **policy_stability_cifar**: Sequential rule triggers, verify all 4 stability_basics rules
- [ ] **plateau_recovery_cifar**: Artificial plateau → plateau_recovery pack kicks in
- [ ] **multi_loss_conflict**: 2-head MNIST, conflicting aux → multi_loss_assist fires

#### D. CLI Integration
- [ ] **`hotcb scenario list`**: Shows checkpoint scenarios alongside existing ones
- [ ] **`hotcb scenario run <id>`**: Runs checkpoint scenario, prints pass/fail with metric details
- [ ] **`hotcb scenario run --all-checkpoint`**: Runs all 10, summary table
- [ ] **`hotcb scenario verify-checkpoints`**: Checks checkpoints exist, re-generates if missing

#### E. CI / Test Integration
- [ ] **`src/hotcb/tests/test_checkpoint_scenarios.py`**: pytest parametrized over all 10 scenarios
- [ ] **Markers**: `@pytest.mark.checkpoint` — skip if checkpoints not downloaded
- [ ] **GPU/CPU flexibility**: Scenarios work on CPU (slower but functional), GPU preferred
- [ ] **Tolerance tuning**: Document expected variances per scenario (CPU vs GPU, torch version)

#### F. Docs
- [ ] **`docs/scenarios.md`**: Updated with checkpoint scenario catalog, how to create custom checkpoints
- [ ] **README**: Mention `scripts/download_checkpoints.py` for first-time setup

**Log:**
- 2026-03-16: Created. 10 scenarios designed covering determinism, all actuator families, 3 policy packs.

---

## rule-calibration
**Goal:** Auto-tune the 38 numeric knobs across 6 autopilot policy packs using Bayesian optimization (Optuna) over scenario outcomes. Reuses tune module infrastructure heavily.
**Branch:** —
**Depends on:** `checkpoint-scenarios` (needs real scenario outcomes as objective function)

### What We're Tuning
38 numeric knobs across 6 policy packs (see full inventory below). These fall into 4 categories:

| Category | Examples | Count | Search Strategy |
|----------|----------|-------|-----------------|
| **Metric thresholds** | `grad_norm > 10.0`, `train_loss > 100.0` | 12 | Log-uniform (scale-invariant) |
| **Action multipliers** | `lr_mult: 0.5`, `aux_weight: 0.3` | 10 | Uniform [0.05, 2.0] |
| **Window sizes** | `window: 5`, `window: 25` | 8 | Integer [3, 50] |
| **Cooldown periods** | `cooldown: 20`, `cooldown: 200` | 8 | Integer [5, 500] |

### Reuse from `modules/tune/`

| Tune Component | Reuse For | Adaptation Needed |
|----------------|-----------|-------------------|
| `search.py` — Optuna TPE | Threshold proposal engine | Wrap rule params as `suggest_float`/`suggest_int` calls |
| `evaluator.py` — segment scoring | Scenario outcome scoring | Replace metric pre/post with scenario pass/fail + recovery speed |
| `schemas.py` — `MutationSpec` | `RuleParamSpec(bounds, prior_center)` | New dataclass, same pattern |
| `recipe.py` — `evolve_recipe()` | `evolve_thresholds()` — EMA shift toward winners | Same EMA logic, different target (YAML params vs recipe params) |
| `storage.py` — JSONL + YAML I/O | Calibration artifacts | Same writers, new file names |
| `constraints.py` — safety checks | Prevent degenerate thresholds | Bounds clipping (e.g., cooldown can't be 0) |

**Not reused:** `controller.py` (event-driven online tuning — calibration is offline batch), `state.py` (runtime state machine — calibration runs complete scenarios)

### Architecture

```
src/hotcb/calibration/
    __init__.py              # exports RuleCalibrator
    spec.py                  # RuleParamSpec, CalibrationConfig, search space definition
    objective.py             # Scenario-based objective function
    calibrator.py            # RuleCalibrator — wraps Optuna study, runs trials
    evolve.py                # evolve_thresholds() — EMA across calibration rounds
    report.py                # Calibration results, comparison tables
```

### Objective Function
Each Optuna trial:
1. Generate candidate YAML pack with proposed thresholds
2. Run N checkpoint scenarios with that pack loaded
3. Score each scenario: `recovery_speed * (1 - overshoot_penalty) * firing_precision`
   - **recovery_speed**: Steps to recover ÷ max allowed steps (lower = better)
   - **overshoot_penalty**: Did the action over-correct? (e.g., LR too low, loss stalls)
   - **firing_precision**: Did the rule fire at the right step? (±window tolerance)
4. Aggregate: weighted mean across scenarios (stability scenarios weighted higher)

### Search Space Definition (`spec.py`)
```python
@dataclass
class RuleParamSpec:
    """Analogous to tune's MutationSpec — defines search bounds for one rule knob."""
    rule_id: str           # e.g., "stability_basics.grad_spike_clip"
    param_path: str        # e.g., "params.expression" or "cooldown" or "action.params.lr_mult"
    param_type: str        # "threshold", "multiplier", "window", "cooldown"
    bounds: tuple          # (lo, hi) — search range
    prior_center: float    # current value from shipped YAML
    distribution: str      # "log_uniform", "uniform", "int_uniform"
```

Auto-generated from YAML packs: parse each rule, extract numeric fields, assign bounds by category.

### Calibration Modes

**Mode 1 — Offline Batch (`hotcb autopilot calibrate`)**
- Runs full Optuna study: 50-100 trials × N scenarios per trial
- Uses `optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=10))`
- SQLite persistence: `hotcb.calibration.study.sqlite`
- Output: `calibrated_{pack_name}.yaml` with tuned thresholds + confidence intervals
- Runtime: 30-60min with 10 scenarios on GPU, parallelizable

**Mode 2 — Online Adaptation (future, via research module)**
- After each `CompletedEffect` from an autopilot rule firing, score the outcome
- Feed into `OutcomePredictor` (research learner) as counterfactual: "would threshold X+δ have been better?"
- When prediction confidence > 0.8, propose threshold update (suggest mode) or apply (auto mode)
- This is the `PatternDiscoverer` Tier 3 extension

### Tasks

#### A. Search Space Infrastructure
- [ ] **`spec.py`**: `RuleParamSpec` dataclass, `extract_search_space(pack_yaml) → List[RuleParamSpec]`
- [ ] **Auto-extraction**: Parse YAML, classify numeric fields into threshold/multiplier/window/cooldown
- [ ] **Bounds assignment**: Sensible defaults per category (thresholds: 0.1x-10x current, multipliers: [0.01, 2.0], windows: [3, 100], cooldowns: [5, 500])
- [ ] **Override file**: Optional `calibration_config.yaml` for custom bounds per knob

#### B. Objective Function
- [ ] **`objective.py`**: `score_scenario(scenario_result, expected) → float`
- [ ] **Recovery speed**: Measure steps from intervention to metric recovery (within 10% of pre-intervention baseline)
- [ ] **Overshoot penalty**: If post-action metric overshoots past pre-intervention by >20%, penalize
- [ ] **Firing precision**: Was the rule triggered within ±5 steps of the expected trigger point?
- [ ] **Aggregate**: `calibration_objective(trial, scenarios, pack_yaml) → float` — weighted sum

#### C. Calibrator (reuses tune's Optuna pattern)
- [ ] **`calibrator.py`**: `RuleCalibrator(pack_name, scenarios, config)`
- [ ] **Optuna study**: `create_study(storage="sqlite:///hotcb.calibration.study.sqlite", sampler=TPESampler(n_startup_trials=10))`
- [ ] **Trial → YAML**: Each trial generates temp YAML with proposed thresholds, loads into AutopilotEngine
- [ ] **Parallelism**: `study.optimize(objective, n_trials=100, n_jobs=1)` (scenarios aren't thread-safe yet)
- [ ] **Best params → YAML**: `export_calibrated_pack(study, pack_name) → calibrated_{pack}.yaml`
- [ ] **Confidence intervals**: Bootstrap from top-10% trials, report ±σ per knob

#### D. Threshold Evolution
- [ ] **`evolve.py`**: `evolve_thresholds(base_yaml, calibration_results, alpha=0.3) → evolved_yaml`
- [ ] **Same EMA as tune's `evolve_recipe()`**: `new_center = α * winner + (1-α) * prior_center`
- [ ] **Multi-round**: Run calibration → evolve → re-calibrate with tighter bounds around new center
- [ ] **Convergence**: Stop when threshold shift < ε for all knobs (2-3 rounds typically sufficient)

#### E. CLI
- [ ] **`hotcb autopilot calibrate`**: Run offline calibration for a pack
  - `--pack stability_basics` (which pack to calibrate)
  - `--scenarios all` or `--scenarios lr_recovery_cifar,nan_guard_mnist` (subset)
  - `--trials 100` (Optuna trial count)
  - `--output calibrated_stability_basics.yaml`
- [ ] **`hotcb autopilot calibrate --status`**: Show current study progress
- [ ] **`hotcb autopilot calibrate --compare`**: Side-by-side table of original vs calibrated thresholds
- [ ] **`hotcb autopilot calibrate --evolve`**: Run evolution pass on existing results

#### F. Report & Visualization
- [ ] **`report.py`**: `CalibrationReport` — original vs calibrated per knob, scenario scores, trial history
- [ ] **Table output**: CLI-friendly ASCII table + optional CSV/JSON export
- [ ] **Dashboard integration**: `/api/autopilot/calibration/status` endpoint (shows calibration progress if running)

#### G. Tests
- [ ] **`test_rule_calibration.py`**: Unit tests for spec extraction, objective scoring, evolution
- [ ] **Mock scenarios**: Fast synthetic scenario stubs for testing calibrator without real training
- [ ] **Integration**: Calibrate `stability_basics` on 2 fast scenarios (10 trials), verify output YAML is valid

#### H. Knob Inventory (all 38)

**stability_basics (8 knobs)**:
- nan_guard: lr_mult=0.5, cooldown=50
- grad_spike_clip: threshold=10.0, lr_mult=0.5, cooldown=20
- lr_emergency_floor: threshold=100.0, lr_mult=0.1, cooldown=100
- loss_spike_recovery: window=5, threshold=0.5, lr_mult=0.3, cooldown=30

**multi_loss_assist (7 knobs)**:
- aux_conflict_reduce: conflict_score threshold=0.7, aux_weight=0.5, rollback_horizon=200, cooldown=50
- loss_ratio_target: ratio=3.0, aux_weight=0.3, cooldown=100
- aux_instability_rollback: threshold=10.0, aux_weight=0.1, cooldown=50
- aux_warmup_ramp: epoch=100, aux_weight=0.1, cooldown=20

**distillation_assist (7 knobs)**:
- summary_first_warmup: epoch=200, distill_weight=0.9, cooldown=50
- spatial_delayed_ramp: epoch=500, spatial_weight=0.5, cooldown=100
- feature_health_check: threshold=5.0, lr_mult=0.5, rollback_horizon=300, cooldown=50
- temperature_guard: threshold=20.0, distill_weight=0.3, cooldown=100

**plateau_recovery (10 knobs)**:
- stagnation_detect: window=25, epsilon=0.002, cooldown=50, lr_mult=0.5
- cosine_restart: window=30, epsilon=0.003, cooldown=80, lr_mult=2.0, bounds.max=0.01
- aux_emphasis_shift: window=20, epsilon=0.001, cooldown=60, aux_weight=0.8
- conservative_finish: window=40, epsilon=0.001, cooldown=200, lr_mult=0.1

**finish_strong (4 knobs)**:
- enable_swa_late: epoch=800, cooldown=500
- enable_ema_late: epoch=700, cooldown=500
- mutation_lockdown: epoch=900, cooldown=500, lr_mult=0.1
- best_checkpoint: threshold=0.01, cooldown=200

**Note:** Step-based thresholds (epoch=100, epoch=800) are **relative to total training steps**. Calibrator normalizes to fractions (0.0-1.0) during search, converts back to absolute steps using `max_steps`.

**Log:**
- 2026-03-16: Created. 38 knobs inventoried. Tune module reuse analysis done: search.py (Optuna TPE), evaluator.py (scoring), recipe.py (EMA evolution), schemas.py (MutationSpec pattern), storage.py (I/O).

---

## chore-release-prep
**Goal:** PyPI 2.0.0 publish. Depends on `v2-stabilization` + `feature-test-coverage`.
- [ ] Merge fix branches to main
- [ ] Build sdist + wheel, verify static files included
- [ ] Test install in fresh venv, run `hotcb demo`
- [ ] Write CHANGELOG.md
- [ ] Tag + publish

---

## compare-eval-perf

**Goal:** The Compare tab and eval run discovery are slow (especially with many runs). Diagnose structural bottlenecks and optimize.

**Problem statement:**
- Compare tab calls `/api/runs/discover` which scans directories and reads metrics files
- With many eval runs, this takes several seconds — feels broken
- Research tab also calls `/api/runs/discover` on every refresh, compounding the problem
- No caching, no incremental loading, no pagination

**Discussion topics:**
- Should we cache discovered runs in memory (with TTL or filesystem watcher)?
- Should `/api/runs/discover` be async/streaming instead of blocking?
- Should metrics files use a summary index (e.g., `hotcb.summary.json` written periodically) to avoid parsing full JSONL?
- Can we precompute run metadata (step count, metric names, last update) at write time?
- Should Compare tab lazy-load metrics only for selected runs (not all)?
- Research graph's `fetchRuns()` could reuse a shared cache with Compare tab

**Scope:**
- [ ] Profile `/api/runs/discover` with 10+ runs — identify bottleneck (fs scan vs JSONL parse)
- [ ] Design caching strategy (in-memory dict with mtime checks, or filesystem watcher)
- [ ] Implement `hotcb.summary.json` written by MetricsCollector every N steps
- [ ] Compare tab: lazy-load metrics on selection, not on tab open
- [ ] Research tab: share cached run list, avoid duplicate API calls

---

## serve-vs-launch

**Goal:** Clarify the architectural difference between `hotcb serve` (hooks/external) and `hotcb launch` (encapsulated), and whether both modes are needed.

**Current situation:**
- **`hotcb serve --dir runs/exp1`**: Dashboard server that tails JSONL files from a run directory. Training runs independently — user's own script writes metrics, reads commands. The "hooks" approach: user keeps full control, hotcb is a side-channel observer/controller.
- **`hotcb launch`**: Starts training + dashboard + autopilot in one call. Returns `LaunchHandle`. The "encapsulated" approach: hotcb owns the training lifecycle.

**Key questions:**
1. Are these genuinely different use cases, or is `launch` just convenience sugar over `serve`?
2. Should seasoned users who have their own pipeline/infra always use `serve`?
3. Is `launch` mainly for getting started quickly / demos / notebooks?
4. Do we need both integration paths in the docs, or should we pick one as primary?
5. If both exist, should `launch` internally just call `serve` + start training thread?

**Arguments for both modes:**
- `serve` is zero-intrusion — works with any training script that writes JSONL, any infra (SLURM, K8s, etc.)
- `launch` is batteries-included — great for notebooks, demos, quick experiments
- `launch` can auto-configure actuators, autopilot, metrics collection
- `serve` users may have custom metrics pipelines that don't use MetricsCollector

**Arguments for consolidation:**
- Two modes = two integration docs = confusion for new users
- `launch` is just `serve` + threading + `train_fn()` — not a fundamentally different architecture
- Both read the same JSONL files, both use the same dashboard

**Design options:**
- A) Keep both, document `serve` as "production/advanced" and `launch` as "getting started/notebook"
- B) Keep both, make `launch` explicitly a thin wrapper over `serve` (currently it mostly is)
- C) Merge into one mode with optional `--train-fn` flag
- D) Add a third mode: `hotcb attach <pid>` for connecting to an already-running process

**Decision:**
- [ ] Map user stories for each mode
- [ ] Audit `launch.py` to see what it does beyond serve + thread
- [ ] Decide primary vs secondary mode
- [ ] Update docs/getting-started to reflect decision
