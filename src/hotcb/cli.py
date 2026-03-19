from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .util import append_jsonl, ensure_dir


def _cmd_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.commands.jsonl")


def _cfg_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.yaml")


def _applied_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.applied.jsonl")


def _recipe_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.recipe.jsonl")


def _freeze_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.freeze.json")


def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"Expected key=value, got: {s}")
        k, v = s.split("=", 1)
        v = v.strip()

        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue

        if v.startswith("{") or v.startswith("["):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass

        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass

        out[k] = v
    return out


def cmd_init(args: argparse.Namespace) -> None:
    run_dir = args.dir
    ensure_dir(run_dir)
    cfg = _cfg_path(run_dir)
    if not os.path.exists(cfg):
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("version: 1\n")
        print(f"Wrote {cfg}")
    else:
        print(f"Exists: {cfg}")

    for p in [_cmd_path(run_dir), _applied_path(run_dir), _recipe_path(run_dir), _freeze_path(run_dir)]:
        if not os.path.exists(p):
            ensure_dir(os.path.dirname(p))
            with open(p, "w", encoding="utf-8") as f:
                pass
            print(f"Created {p}")
        else:
            print(f"Exists: {p}")


def _append_command(run_dir: str, obj: Dict[str, Any]) -> None:
    append_jsonl(_cmd_path(run_dir), obj)


def cmd_cb(args: argparse.Namespace) -> None:
    op = args.cb_command
    obj: Dict[str, Any] = {"module": "cb", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    if op == "load":
        if not args.file and not args.path:
            raise SystemExit("Provide --file for python_file or --path for module import")
        obj["target"] = {"kind": "python_file" if args.file else "module", "path": args.file or args.path, "symbol": args.symbol}
        obj["init"] = _parse_kv(args.init or [])
        if args.enabled is not None:
            obj["enabled"] = args.enabled
    _append_command(args.dir, obj)
    print(f"queued cb {op} for {args.id}")


def cmd_opt(args: argparse.Namespace) -> None:
    op = args.opt_command
    obj: Dict[str, Any] = {"module": "opt", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued opt {op} for {args.id}")


def cmd_loss(args: argparse.Namespace) -> None:
    op = args.loss_command
    obj: Dict[str, Any] = {"module": "loss", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued loss {op} for {args.id}")


def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = {
        "mode": args.mode,
        "recipe_path": args.recipe,
        "adjust_path": args.adjust,
        "policy": args.policy,
        "step_offset": args.step_offset,
    }
    ensure_dir(args.dir)
    with open(_freeze_path(args.dir), "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg))
    print(f"Wrote freeze state -> {_freeze_path(args.dir)}")


_OPT_KEYS = {"lr", "weight_decay", "clip_norm", "scheduler_scale", "scheduler_drop", "group", "groups"}


def _infer_module(keys: set) -> str:
    """Auto-route kv keys to opt or loss module (spec §15.4)."""
    if keys & _OPT_KEYS:
        return "opt"
    for k in keys:
        if k.endswith("_w") or k.startswith("terms.") or k.startswith("ramps."):
            return "loss"
    raise SystemExit(
        f"Cannot auto-route keys {keys} to opt or loss. Use explicit subcommand."
    )


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = args.dir
    # Freeze state
    fp = _freeze_path(run_dir)
    freeze = {}
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                freeze = json.load(f)
        except Exception:
            pass
    print(f"Freeze mode: {freeze.get('mode', 'off')}")
    if freeze.get("recipe_path"):
        print(f"  recipe: {freeze['recipe_path']}")
    if freeze.get("adjust_path"):
        print(f"  adjust: {freeze['adjust_path']}")
    if freeze.get("policy"):
        print(f"  policy: {freeze['policy']}")

    # Latest applied entries per module
    applied = _applied_path(run_dir)
    if not os.path.exists(applied):
        print("No applied ledger found.")
        return
    latest: Dict[str, dict] = {}
    try:
        with open(applied, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                mod = rec.get("module", "")
                key = f"{mod}:{rec.get('id', '')}"
                latest[key] = rec
    except Exception:
        pass
    if latest:
        print(f"\nLast applied entries ({len(latest)} handles):")
        for key, rec in latest.items():
            decision = rec.get("decision", "?")
            step = rec.get("step", "?")
            op = rec.get("op", "?")
            print(f"  {key}: {op} @ step {step} [{decision}]")


def cmd_sugar_enable(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb enable <id>` defaults to cb."""
    obj: Dict[str, Any] = {"module": "cb", "op": "enable", "id": args.id}
    _append_command(args.dir, obj)
    print(f"queued cb enable for {args.id}")


def cmd_sugar_disable(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb disable <id>` defaults to cb."""
    obj: Dict[str, Any] = {"module": "cb", "op": "disable", "id": args.id}
    _append_command(args.dir, obj)
    print(f"queued cb disable for {args.id}")


def cmd_sugar_set(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb set k=v` auto-routes to opt or loss."""
    params = _parse_kv(args.kv or [])
    module = _infer_module(set(params.keys()))
    obj: Dict[str, Any] = {"module": module, "op": "set_params", "id": args.id, "params": params}
    _append_command(args.dir, obj)
    print(f"queued {module} set_params for {args.id}")


# ═══════════════════════════════════════════════════════════════════════
# Continuation tuning commands
# ═══════════════════════════════════════════════════════════════════════

def cmd_continue_baseline(args: argparse.Namespace) -> None:
    """Train a baseline model with periodic checkpoints."""
    import json as _json
    task = args.task
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Bootstrap JSONL files
    for fname in ["hotcb.commands.jsonl", "hotcb.applied.jsonl", "hotcb.metrics.jsonl"]:
        open(os.path.join(out_dir, fname), "w").close()
    with open(os.path.join(out_dir, "hotcb.freeze.json"), "w") as f:
        _json.dump({"mode": "off"}, f)
    # No-recipe sentinel
    open(os.path.join(out_dir, "hotcb.eval.no_recipe"), "w").close()

    print(f"Training {task} baseline with checkpoints → {out_dir}")

    if task == "mnist":
        from hotcb.eval.tasks import mnist_training_with_checkpoints
        default_steps = 1500
        fn = mnist_training_with_checkpoints
    elif task == "cifar10":
        from hotcb.eval.tasks import cifar10_training_with_checkpoints
        default_steps = 2000
        fn = cifar10_training_with_checkpoints
    elif task == "coco_mobilenet":
        from hotcb.eval.tasks_coco import coco_mobilenet_training_with_checkpoints
        default_steps = 5000
        fn = coco_mobilenet_training_with_checkpoints
    elif task == "imagenet_mobilenet":
        from hotcb.eval.tasks_imagenet import imagenet_mobilenet_training_with_checkpoints
        default_steps = 10000
        fn = imagenet_mobilenet_training_with_checkpoints
    elif task == "clip_coco":
        from hotcb.eval.tasks_vlm import clip_coco_training_with_checkpoints
        default_steps = 5000
        fn = clip_coco_training_with_checkpoints
    else:
        raise SystemExit(f"Unknown task: {task}")

    max_steps = args.steps or default_steps
    fn(out_dir, max_steps=max_steps, checkpoint_interval=args.ckpt_interval)
    print(f"Baseline complete. Checkpoints: {out_dir}/checkpoints/")


def cmd_continue_run(args: argparse.Namespace) -> None:
    """Run continuation tuning from a converged base run."""
    from hotcb.routines.continuation import (
        ContinuationConfig, ObjectiveSpec, GuardrailSpec, BudgetSpec,
        ContinuationPlanner, ContinuationLauncher, ContinuationReport,
        AnchorSelector, ResumeMode,
    )
    from hotcb.routines.continuation.planner import (
        DEFAULT_RECIPES, AGGRESSIVE_RECIPES, COMBO_RECIPES,
        MULTI_STAGE_RECIPES, ALL_RECIPES,
    )

    base_run_dir = args.run
    output_dir = args.out or os.path.join(base_run_dir, "continuation")

    # Read base metrics
    best_path = os.path.join(base_run_dir, "best_metrics.json")
    base_best = {}
    if os.path.exists(best_path):
        with open(best_path) as f:
            base_best = json.load(f)

    # Read final metrics from metrics JSONL
    metrics_path = os.path.join(base_run_dir, "hotcb.metrics.jsonl")
    base_final = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            lines = f.readlines()
        if lines:
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        base_final = rec.get("metrics", rec)
                        break
                    except json.JSONDecodeError:
                        continue

    # Select anchors
    selector = AnchorSelector(base_run_dir)
    anchors = selector.select_auto(
        max_anchors=args.max_anchors,
        primary_metric=args.metric,
        mode=args.mode,
    )

    if not anchors:
        raise SystemExit(f"No checkpoints found in {base_run_dir}/checkpoints/")

    print(f"Found {len(anchors)} anchor(s):")
    for a in anchors:
        print(f"  {a.anchor_id}: step={a.step}, {a.reason.value}")

    # Filter recipes if specified
    recipe_groups = {
        "default": DEFAULT_RECIPES,
        "aggressive": AGGRESSIVE_RECIPES,
        "combo": COMBO_RECIPES,
        "multi_stage": MULTI_STAGE_RECIPES,
        "all": ALL_RECIPES,
    }

    recipes = list(DEFAULT_RECIPES)
    if args.recipes:
        expanded = []
        for name in args.recipes:
            if name in recipe_groups:
                expanded.extend(recipe_groups[name])
            else:
                match = [r for r in ALL_RECIPES if r.name == name]
                expanded.extend(match)
        if expanded:
            recipes = expanded

    resume_modes = [ResumeMode(m) for m in args.resume_modes]

    config = ContinuationConfig(
        base_run_dir=base_run_dir,
        base_final_metrics=base_final,
        base_best_metrics=base_best,
        task=args.task,
        objective=ObjectiveSpec(
            primary_metric=args.metric,
            mode=args.mode,
        ),
        guardrails=GuardrailSpec(),
        budget=BudgetSpec(
            max_anchors=args.max_anchors,
            branches_per_anchor=args.branches_per_anchor,
            max_extra_steps=args.extra_steps,
        ),
        anchors=anchors,
        recipes=recipes,
        resume_modes=resume_modes,
        autopilot=args.autopilot,
        policy_packs=args.packs or [],
        output_dir=output_dir,
    )

    print(f"\nPlanning {config.total_branches} branches...")
    planner = ContinuationPlanner(config)
    branches = planner.plan()
    print(f"Planned {len(branches)} branches")

    print(f"\nRunning continuation routine...")
    launcher = ContinuationLauncher(config, output_dir)
    result = launcher.run_all(branches)

    report = ContinuationReport(result)
    print("\n" + report.full_report())
    report_path = report.save(output_dir)
    print(f"\nReport saved: {report_path}")


def cmd_continue_report(args: argparse.Namespace) -> None:
    """Show a previously generated continuation report."""
    report_path = os.path.join(args.cont_dir, "continuation_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            print(f.read())
    else:
        print(f"No report found at {report_path}")


def cmd_tune(args: argparse.Namespace) -> None:
    op = args.tune_command
    obj: Dict[str, Any] = {"module": "tune", "op": op}
    if op == "enable":
        mode = getattr(args, "mode", "active")
        obj["params"] = {"mode": mode}
    elif op == "set":
        obj["op"] = "set"
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued tune {op}")


def cmd_tune_status(args: argparse.Namespace) -> None:
    run_dir = args.dir
    recipe_path = os.path.join(run_dir, "hotcb.tune.recipe.yaml")
    summary_path = os.path.join(run_dir, "hotcb.tune.summary.json")

    if os.path.exists(recipe_path):
        print(f"Tune recipe: {recipe_path}")
    else:
        print("No tune recipe found.")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print(f"Mode: {summary.get('mode', '?')}")
            print(f"Mutations: {summary.get('total_mutations', 0)} total, {summary.get('applied_mutations', 0)} applied")
            print(f"Accept rate: {summary.get('accept_rate', 0):.1%}")
            segs = summary.get("segments_by_decision", {})
            for d, c in segs.items():
                print(f"  {d}: {c}")
        except Exception:
            print("Failed to read tune summary.")
    else:
        print("No tune summary found.")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run benchmarks."""
    from .bench.tasks import BUILTIN_TASKS
    from .bench.runner import BenchmarkRunner
    from .bench.report import BenchmarkReport

    task_factory = BUILTIN_TASKS.get(args.task)
    if task_factory is None:
        raise SystemExit(f"Unknown task: {args.task}. Available: {list(BUILTIN_TASKS)}")

    max_steps = args.max_steps
    task = task_factory(max_steps=max_steps) if max_steps else task_factory()

    runner = BenchmarkRunner(output_dir=args.output_dir)
    conditions = [c.strip() for c in args.conditions.split(",")]

    for cond in conditions:
        if cond == "baseline":
            result = runner.run_baseline(task)
        elif cond == "auto_tune":
            result = runner.run_with_hotcb(task)
        elif cond == "recipe_replay":
            # Need a recipe from a prior auto_tune run
            prev = [r for r in runner.results if r.recipe_path]
            if not prev:
                print("Skipping recipe_replay: no recipe available")
                continue
            result = runner.run_recipe_replay(task, prev[-1].recipe_path)
        else:
            print(f"Unknown condition: {cond}")
            continue
        print(f"  {cond}: loss={result.final_metrics.get('loss', '?'):.6f} "
              f"steps={result.total_steps} time={result.total_time_sec:.3f}s")

    report = BenchmarkReport(runner.results)
    print()
    print(report.summary_table())
    report.to_json(os.path.join(args.output_dir, "benchmark.json"))
    report.to_csv(os.path.join(args.output_dir, "benchmark.csv"))
    print(f"\nResults saved to {args.output_dir}/")


def cmd_bench_eval(args: argparse.Namespace) -> None:
    """Run autopilot evaluation against a published benchmark."""
    from .bench.eval_autopilot import AutopilotEval

    ev = AutopilotEval(output_dir=args.output_dir)

    phases = [p.strip() for p in args.phases.split(",")]

    for phase in phases:
        if phase == "baseline":
            print(f"Running published baseline for {args.task} ...")
            result = ev.run_published_baseline(args.task)
            acc = result.final_metrics.get("val_accuracy", "?")
            print(f"  Baseline done: val_accuracy={acc}  time={result.total_time_sec:.1f}s")
        elif phase == "autopilot":
            print(f"Running autopilot challenge for {args.task} ...")
            result = ev.run_autopilot_challenge(
                args.task,
                guidelines_path=args.guidelines,
            )
            acc = result.final_metrics.get("val_accuracy", "?")
            print(f"  Autopilot done: val_accuracy={acc}  time={result.total_time_sec:.1f}s")
        else:
            print(f"Unknown phase: {phase}")

    print()
    print(ev.report())


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the dashboard server."""
    from .server.app import run_server, create_app

    multi_dirs = None
    if args.dirs:
        multi_dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]

    autopilot_mode = getattr(args, "autopilot", None)
    key_metric = getattr(args, "key_metric", None)

    if autopilot_mode and autopilot_mode != "off":
        # Need to create app manually to configure autopilot before start
        import uvicorn

        app = create_app(
            args.dir,
            poll_interval=args.poll_interval,
            multi_dirs=multi_dirs,
        )

        # Configure autopilot
        ap_engine = app.state.autopilot_engine
        ai_engine = getattr(app.state, "ai_engine", None)

        if key_metric and ai_engine:
            ai_engine.state.key_metric = key_metric
            ai_engine.save_state()

        try:
            ap_engine.set_mode(autopilot_mode)
            print(f"Autopilot mode: {autopilot_mode}")
            if key_metric:
                print(f"Key metric: {key_metric}")
        except ValueError as e:
            print(f"Warning: {e}")

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        run_server(
            run_dir=args.dir,
            host=args.host,
            port=args.port,
            poll_interval=args.poll_interval,
            multi_dirs=multi_dirs,
        )


def cmd_scenario_list(args: argparse.Namespace) -> None:
    """List available scenarios."""
    from .scenarios import list_scenarios

    scenarios = list_scenarios()
    if not scenarios:
        print("No scenarios found.")
        return
    print(f"{'Name':<30} {'Pack':<25} {'Steps':<8} {'Framework':<10} Description")
    print("-" * 110)
    for s in scenarios:
        print(f"{s.name:<30} {s.pack:<25} {s.max_steps:<8} {s.framework:<10} {s.description[:40]}")


def cmd_scenario_run(args: argparse.Namespace) -> None:
    """Run one or more scenarios."""
    from .scenarios import get, list_scenarios
    from .scenarios.runner import ScenarioRunner, run_all

    if args.all or args.pack:
        results = run_all(
            pack=args.pack,
            step_delay=args.step_delay,
            verbose=True,
        )
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"\n{passed}/{total} scenarios passed")
        if passed < total:
            for r in results:
                if not r.passed:
                    print(f"  FAIL: {r.name} — missing rules: {r.missing_rules}")
                    if r.error:
                        print(f"        error: {r.error}")
            raise SystemExit(1)
        return

    if args.name is None:
        raise SystemExit("Provide a scenario name, --all, or --pack <name>")

    config = get(args.name)
    runner = ScenarioRunner(step_delay=args.step_delay, verbose=True)
    result = runner.run(
        config,
        dashboard=args.dashboard,
        host=args.host,
        port=args.port,
    )

    status = "PASS" if result.passed else "FAIL"
    print(f"\nScenario {result.name}: {status}")
    print(f"  Rules fired: {result.rules_fired}")
    if result.missing_rules:
        print(f"  Missing rules: {result.missing_rules}")
    if result.error:
        print(f"  Error: {result.error}")
    if not result.passed:
        raise SystemExit(1)


def cmd_demo(args: argparse.Namespace) -> None:
    """Launch a demo: synthetic training + live dashboard."""
    # --scenario flag: delegate to scenario runner with dashboard
    scenario_name = getattr(args, "scenario", None)
    if scenario_name:
        from .scenarios import get
        from .scenarios.runner import ScenarioRunner

        config = get(scenario_name)
        runner = ScenarioRunner(step_delay=args.step_delay, verbose=True)
        result = runner.run(
            config,
            dashboard=True,
            host=args.host,
            port=args.port,
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"\nScenario {result.name}: {status}")
        print(f"  Rules fired: {result.rules_fired}")
        return

    autopilot = getattr(args, "autopilot", "off")
    key_metric = getattr(args, "key_metric", None)

    if autopilot != "off":
        # Use launch() API to get autopilot wired before training starts
        from .launch import launch

        config = "multitask" if args.golden else "simple"
        handle = launch(
            config=config,
            run_dir=args.demo_dir if args.demo_dir else None,
            autopilot=autopilot,
            key_metric=key_metric or "val_loss",
            max_steps=args.max_steps,
            max_time=getattr(args, "max_time", None),
            step_delay=args.step_delay,
            host=args.host,
            port=args.port,
            serve=True,
            block=True,
        )
        return

    if args.golden:
        from .golden_demo import run_golden_demo

        run_golden_demo(
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            run_dir=args.demo_dir if args.demo_dir else None,
        )
    else:
        from .demo import run_demo

        run_demo(
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            run_dir=args.demo_dir if args.demo_dir else None,
        )


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch training + dashboard + autopilot in one command."""
    from .launch import launch

    import sys

    train_fn = getattr(args, "train_fn", None)
    config = getattr(args, "config", "multitask")

    handle = launch(
        train_fn=train_fn,
        config=config,
        config_file=getattr(args, "config_file", None),
        run_dir=args.dir if args.dir != "." else None,
        autopilot=args.autopilot,
        key_metric=args.key_metric or "val_loss",
        ai_model=getattr(args, "ai_model", "gpt-4o-mini"),
        ai_budget=getattr(args, "ai_budget", 5.0),
        ai_cadence=getattr(args, "ai_cadence", 50),
        max_steps=args.max_steps,
        max_time=getattr(args, "max_time", None),
        step_delay=args.step_delay,
        host=args.host,
        port=args.port,
        seed=getattr(args, "seed", None),
        serve=True,
        block=True,
    )


def cmd_eval_list(args: argparse.Namespace) -> None:
    """List available evaluation conditions."""
    from .eval.conditions import ALL_CONDITIONS, ALL_WITH_SYNTHETIC, load_conditions_yaml
    conditions = list(ALL_WITH_SYNTHETIC if getattr(args, 'include_synthetic', False) else ALL_CONDITIONS)
    if getattr(args, 'conditions_file', None):
        user = load_conditions_yaml(args.conditions_file)
        conditions.extend(user)
    print(f"{'Name':<40} {'Demo':<10} {'Autopilot':<8} {'NN':<4} Description")
    print("-" * 120)
    for c in conditions:
        nn = "yes" if c.nn_mode else ""
        print(f"{c.name:<40} {c.demo:<10} {c.autopilot:<8} {nn:<4} {c.description[:55]}")
    print(f"\n{len(conditions)} conditions available")
    if not getattr(args, 'include_synthetic', False):
        print("  (use --include-synthetic to see golden/finetune/simple demo conditions)")


def cmd_eval_run(args: argparse.Namespace) -> None:
    """Run evaluation conditions with live dashboard.

    Flow:
    1. Pre-create all hypotheses as "proposed" in research graph
    2. Start dashboard server immediately (background thread)
    3. Run conditions one-by-one, updating graph live after each
    4. Frontend auto-refreshes graph + auto-focuses latest completed run
    """
    import copy
    import json as _json
    import sys
    import threading
    import time

    from .eval.conditions import ALL_CONDITIONS, ALL_WITH_SYNTHETIC, EvalCondition, load_conditions_yaml
    from .eval.harness import EvalHarness
    from .eval.report import EvalReport
    from .research.engine import ResearchEngine

    # Resolve output dir
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # NOTE: Do NOT create hotcb.metrics.jsonl in the parent output_dir.
    # The dashboard's _resolve_active_run_dir() scans for subdirs with metrics.
    # If the parent has metrics.jsonl, the tailer watches that (empty) file
    # instead of the actual sub-run metrics — causing blank dashboard.

    # Load user-defined conditions from YAML if provided
    user_conditions: list = []
    if getattr(args, 'conditions_file', None):
        user_conditions = load_conditions_yaml(args.conditions_file)
        w = sys.stderr.write
        w(f"  Loaded {len(user_conditions)} custom condition(s) from {args.conditions_file}\n")

    # Pool: include synthetic demos if --include-synthetic or --demo targets one
    pool = ALL_WITH_SYNTHETIC if getattr(args, 'include_synthetic', False) else ALL_CONDITIONS
    if args.demo in ("golden", "finetune", "simple"):
        pool = ALL_WITH_SYNTHETIC
    # Merge user conditions into pool
    full_pool = list(pool) + user_conditions

    # Select conditions
    if args.conditions:
        names = [n.strip() for n in args.conditions.split(",")]
        conditions = []
        for name in names:
            found = [c for c in full_pool if c.name == name]
            if not found:
                print(f"Unknown condition: {name}")
                print(f"Use 'hotcb eval list --include-synthetic' to see all conditions")
                if user_conditions:
                    print(f"Custom conditions: {', '.join(c.name for c in user_conditions)}")
                raise SystemExit(1)
            conditions.extend(found)
    elif getattr(args, 'conditions_file', None) and not args.demo:
        # If only --conditions-file given, run those conditions only
        conditions = user_conditions
    elif args.demo:
        conditions = [c for c in full_pool if c.demo == args.demo]
    else:
        conditions = list(full_pool)

    max_steps = args.max_steps
    step_delay = args.step_delay

    w = sys.stderr.write
    w(f"\n  hotcb Evaluation Harness\n")
    w(f"  Output: {output_dir}\n")
    steps_str = str(max_steps) if max_steps else "task-default"
    w(f"  Conditions: {len(conditions)} | Steps: {steps_str} | Delay: {step_delay}s\n")

    # ── Phase 1: Pre-create all hypotheses as "proposed" ──
    combined_engine = ResearchEngine(output_dir)
    hyp_map = {}  # condition_name -> hyp node_id
    for cond in conditions:
        stream_id = f"eval_{cond.demo}"
        try:
            combined_engine.graph.create_stream(stream_id, f"Evaluation: {cond.demo}")
        except Exception:
            pass
        hyp = combined_engine.hypothesize(
            condition=cond.hypothesis_condition or cond.description,
            intervention=cond.initial_overrides.get("opt", cond.initial_overrides.get("loss", {})),
            expected_outcome=cond.hypothesis_expected or "see description",
            stream_id=stream_id,
        )
        hyp_map[cond.name] = hyp.node_id
    combined_engine.graph.save_snapshot()

    # Write initial eval status
    _write_eval_status(output_dir, conditions, completed=[], running=None, done=False)

    # ── Phase 2: Start dashboard immediately ──
    host = args.host
    port = args.port
    no_serve = getattr(args, 'no_serve', False)
    server_thread = None

    if not no_serve:
        # Pre-create first condition's run dir so the server can resolve it
        first_run_dir = os.path.join(output_dir, conditions[0].name)
        os.makedirs(first_run_dir, exist_ok=True)
        for fname in ["hotcb.metrics.jsonl", "hotcb.commands.jsonl", "hotcb.applied.jsonl"]:
            p = os.path.join(first_run_dir, fname)
            if not os.path.exists(p):
                open(p, "w").close()
        with open(os.path.join(first_run_dir, "hotcb.freeze.json"), "w") as f:
            _json.dump({"mode": "off"}, f)

        w(f"  Dashboard: http://localhost:{port}  (Research tab shows {len(conditions)} untested hypotheses)\n\n")
        server_thread = threading.Thread(
            target=_eval_serve, args=(output_dir, host, port), daemon=True,
        )
        server_thread.start()
        time.sleep(0.5)  # let server bind
    else:
        w(f"\n")

    # ── Phase 3: Run conditions one-by-one, update graph live ──
    harness = EvalHarness(output_dir=output_dir)
    t0 = time.monotonic()
    completed_names = []

    for i, cond in enumerate(conditions, 1):
        c = copy.deepcopy(cond)
        c.step_delay = step_delay
        if max_steps is not None:
            c.max_steps = max_steps

        # Transition hypothesis to "testing"
        hyp_id = hyp_map.get(c.name)
        if hyp_id:
            combined_engine.graph.transition_hypothesis(hyp_id, "testing")
            combined_engine.graph.save_snapshot()

        # Update eval status: this condition is now running
        _write_eval_status(output_dir, conditions, completed_names, running=c.name, done=False)

        w(f"  [{i}/{len(conditions)}] {c.name}...")
        try:
            result = harness.run(c)
            loss = result.final_metrics.get("train_loss", "?")
            acts = len(result.autopilot_actions)
            w(f" loss={loss} actions={acts}\n")
            _write_run_meta(result.run_dir, c)

            # Add evidence to research graph
            if hyp_id:
                _add_eval_evidence(combined_engine, hyp_id, result, c)
                combined_engine.graph.save_snapshot()

            completed_names.append(c.name)
        except Exception as e:
            w(f" FAILED: {e}\n")
            # Mark hypothesis as inconclusive on failure
            if hyp_id:
                combined_engine.graph.transition_hypothesis(hyp_id, "inconclusive")
                combined_engine.graph.save_snapshot()
            completed_names.append(c.name)

        # Update eval status: this condition is done, focus it
        _write_eval_status(output_dir, conditions, completed_names, running=None,
                           done=(i == len(conditions)), focus_run=c.name)

    elapsed = time.monotonic() - t0
    w(f"\n  Done in {elapsed:.1f}s\n\n")

    # Save eval results JSON
    harness.save_results()

    # Print report
    report = EvalReport(harness.results)
    print(report.full_report())

    print(f"\nResults saved to: {output_dir}")

    if server_thread:
        print(f"  Dashboard running at http://localhost:{port}")
        print(f"  Press Ctrl+C to stop.\n")
        try:
            server_thread.join()
        except KeyboardInterrupt:
            w("\n  Shutting down.\n")
    else:
        print(f"\nTo view in dashboard:")
        print(f"  hotcb eval serve --output-dir {output_dir}")


def cmd_eval_report(args: argparse.Namespace) -> None:
    """Print report from saved eval results."""
    import json as _json
    from .eval.harness import EvalResult
    from .eval.report import EvalReport

    results_path = os.path.join(args.output_dir, "eval_results.json")
    if not os.path.exists(results_path):
        print(f"No eval results at {results_path}")
        print("Run 'hotcb eval run' first.")
        raise SystemExit(1)

    with open(results_path) as f:
        data = _json.load(f)

    # Reconstruct EvalResult objects (without full metric history)
    results = []
    for d in data:
        results.append(EvalResult(
            condition_name=d["condition_name"],
            demo=d["demo"],
            description=d["description"],
            final_metrics=d.get("final_metrics", {}),
            metric_history=[],
            applied=d.get("applied", []),
            autopilot_actions=d.get("autopilot_actions", []),
            total_steps=d.get("total_steps", 0),
            intervention_count=d.get("intervention_count", 0),
            elapsed_seconds=d.get("elapsed_seconds", 0),
            run_dir=d.get("run_dir", ""),
            seed=d.get("seed"),
            research_stats=d.get("research_stats", {}),
        ))

    report = EvalReport(results)
    print(report.full_report())


def cmd_eval_serve(args: argparse.Namespace) -> None:
    """Serve dashboard for eval results."""
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        print(f"Output dir not found: {output_dir}")
        raise SystemExit(1)
    _eval_serve(output_dir, args.host, args.port)


def _eval_serve(output_dir: str, host: str, port: int) -> None:
    """Launch dashboard server on eval output dir."""
    from .server.app import run_server
    run_server(run_dir=output_dir, host=host, port=port, poll_interval=1.0)


def _write_eval_status(output_dir, conditions, completed, running=None, done=False, focus_run=None):
    """Write eval progress file — polled by frontend for live updates."""
    import json as _json
    status = {
        "total": len(conditions),
        "completed": len(completed),
        "completed_names": list(completed),
        "running": running,
        "done": done,
        "focus_run": focus_run,
        "conditions": [c.name for c in conditions],
    }
    path = os.path.join(output_dir, "hotcb.eval.status.json")
    with open(path, "w") as f:
        _json.dump(status, f)


def _add_eval_evidence(engine, hyp_id, result, cond):
    """Add evidence from a completed eval result to the research graph."""
    final_loss = result.final_metrics.get("train_loss", 999)
    val_loss = result.final_metrics.get("val_loss", 999)
    val_acc = result.final_metrics.get("val_accuracy", 0)
    # Determine outcome
    if cond.demo in ("mnist", "cifar10"):
        outcome = "improved" if val_acc > 0.5 else ("neutral" if val_acc > 0.2 else "degraded")
    else:
        outcome = "improved" if val_loss < 2.0 else "degraded"
    engine.graph.add_evidence(
        hypothesis_id=hyp_id,
        outcome=outcome,
        delta={
            "train_loss": final_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "interventions": result.intervention_count,
            "autopilot_actions": len(result.autopilot_actions),
        },
        context={
            "condition": cond.name,
            "steps": result.total_steps,
            "autopilot": cond.autopilot,
            "nn_mode": cond.nn_mode,
        },
        source="eval_harness",
    )
    # Conclude hypothesis based on outcome
    if outcome == "improved":
        engine.graph.transition_hypothesis(hyp_id, "confirmed")
    elif outcome == "degraded":
        engine.graph.transition_hypothesis(hyp_id, "refuted")
    else:
        engine.graph.transition_hypothesis(hyp_id, "inconclusive")


def _write_run_meta(run_dir: str, cond) -> None:
    """Write hotcb.run.json so comparison view shows condition name."""
    import json as _json
    meta = {
        "config_name": cond.name,
        "config_id": cond.name,
        "demo": cond.demo,
        "description": cond.description,
        "autopilot": cond.autopilot,
        "nn_mode": cond.nn_mode,
    }
    with open(os.path.join(run_dir, "hotcb.run.json"), "w") as f:
        _json.dump(meta, f, indent=2)


def cmd_research_stream_list(args: argparse.Namespace) -> None:
    from .research.graph import ResearchGraph
    g = ResearchGraph(args.dir)
    if not g.load_snapshot():
        g.replay_events()
    streams = g.list_streams()
    if not streams:
        print("No research streams.")
        return
    print(f"{'ID':<14} {'Name':<30} {'Status':<12} Hyps  Obs")
    print("-" * 80)
    for s in streams:
        print(f"{s.stream_id:<14} {s.name:<30} {s.status:<12} {len(s.hypothesis_ids):<5} {len(s.observation_ids)}")


def cmd_research_stream_new(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    s = engine.graph.create_stream(args.name, args.description or "")
    engine.graph.save_snapshot()
    print(f"Created stream {s.stream_id}: {s.name}")


def cmd_research_stream_conclude(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    ok = engine.graph.conclude_stream(args.stream_id, args.conclusion)
    if ok:
        engine.graph.save_snapshot()
        print(f"Concluded stream {args.stream_id}")
    else:
        print(f"Failed to conclude stream {args.stream_id}")
        raise SystemExit(1)


def cmd_research_observe(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    node = engine.observe(
        text=args.text, step=args.step,
        tags=tags, stream_id=args.stream or None,
    )
    engine.graph.save_snapshot()
    print(f"Observation {node.node_id}: {node.text[:60]}")


def cmd_research_hyp_add(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    intervention = json.loads(args.intervention) if args.intervention else {}
    node = engine.hypothesize(
        condition=args.condition,
        intervention=intervention,
        expected_outcome=args.expected,
        stream_id=args.stream or None,
    )
    engine.graph.save_snapshot()
    print(f"Hypothesis {node.node_id}: {node.condition}")


def cmd_research_hyp_list(args: argparse.Namespace) -> None:
    from .research.graph import ResearchGraph
    g = ResearchGraph(args.dir)
    if not g.load_snapshot():
        g.replay_events()
    status = getattr(args, "status", None)
    hyps = g.hypotheses_by_status(status) if status else g.all_hypotheses()
    if not hyps:
        print("No hypotheses.")
        return
    print(f"{'ID':<14} {'Status':<14} {'Conf':<6} {'Ev':<4} Condition")
    print("-" * 80)
    for h in hyps:
        print(f"{h.node_id:<14} {h.status:<14} {h.confidence:.2f}  {h.evidence_count:<4} {h.condition[:40]}")


def cmd_research_hyp_show(args: argparse.Namespace) -> None:
    from .research.graph import ResearchGraph
    from .research.types import HypothesisNode
    g = ResearchGraph(args.dir)
    if not g.load_snapshot():
        g.replay_events()
    node = g.get_node(args.hyp_id)
    if not isinstance(node, HypothesisNode):
        print(f"Hypothesis not found: {args.hyp_id}")
        raise SystemExit(1)
    print(f"ID:        {node.node_id}")
    print(f"Status:    {node.status}")
    print(f"Condition: {node.condition}")
    print(f"Expected:  {node.expected_outcome}")
    print(f"Confidence: {node.confidence:.2%}")
    print(f"NN Conf:   {node.nn_confidence:.2%}")
    print(f"Evidence:  {node.evidence_count}")
    print(f"Tests:     {node.test_count}")
    print(f"Source:    {node.source}")
    print(f"Intervention: {json.dumps(node.intervention, indent=2)}")


def cmd_research_hyp_test(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    try:
        result = engine.test_hypothesis(args.hyp_id)
        engine.graph.save_snapshot()
        print(f"Testing hypothesis {args.hyp_id}")
        print(f"Command written: {json.dumps(result['command'])}")
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


def cmd_research_hyp_conclude(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    ok = engine.conclude_hypothesis(args.hyp_id, args.status)
    if ok:
        engine.graph.save_snapshot()
        print(f"Hypothesis {args.hyp_id} → {args.status}")
    else:
        print(f"Invalid transition for {args.hyp_id}")
        raise SystemExit(1)


def cmd_research_model_status(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir, nn_mode=True)
    learner = engine._get_learner()
    if learner is None:
        print("NN model not available (torch not installed or no pretrained weights)")
        return
    status = learner.status
    for k, v in status.items():
        print(f"  {k}: {v}")


def cmd_research_export(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    from .research.export import export_graph
    engine = ResearchEngine(args.dir)
    path = export_graph(engine.graph, args.out, stream_id=args.stream or None)
    print(f"Exported to {path}")


def cmd_research_import(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    from .research.export import import_graph
    engine = ResearchEngine(args.dir)
    count = import_graph(engine.graph, args.path)
    engine.graph.save_snapshot()
    print(f"Imported {count} items")


def cmd_research_merge(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    from .research.export import merge_from_run
    engine = ResearchEngine(args.dir)
    count = merge_from_run(engine.graph, args.run_dir)
    engine.graph.save_snapshot()
    print(f"Merged {count} new items")


def cmd_research_export_recipe(args: argparse.Namespace) -> None:
    from .research.engine import ResearchEngine
    engine = ResearchEngine(args.dir)
    path = engine.export_confirmed_recipe(out_path=args.out or None)
    print(f"Recipe exported to {path}")


def cmd_research_load_conditions(args: argparse.Namespace) -> None:
    """Load continuation conditions as research hypotheses."""
    run_dir = args.dir
    from hotcb.research.engine import ResearchEngine
    engine = ResearchEngine(run_dir)
    hyps = engine.load_continuation_conditions(args.task, args.base_run)
    print(f"Loaded {len(hyps)} continuation conditions as hypotheses")
    for h in hyps:
        print(f"  {h.node_id}: {h.condition[:60]}")


def cmd_research_launch_tests(args: argparse.Namespace) -> None:
    """Launch continuation tests for selected hypotheses."""
    run_dir = args.dir
    from hotcb.research.engine import ResearchEngine
    engine = ResearchEngine(run_dir)
    hyp_ids = [h.strip() for h in args.hyps.split(",")]
    results = engine.launch_condition_tests(hyp_ids, args.base_run)
    for r in results:
        status = r.get("outcome", r.get("error", "unknown"))
        delta = r.get("primary_delta_pct", "")
        print(f"  {r['hypothesis_id']}: {status}" + (f" ({delta:+.2f}%)" if isinstance(delta, (int, float)) else ""))


def cmd_tune_export_recipe(args: argparse.Namespace) -> None:
    run_dir = args.dir
    out = args.out or os.path.join(run_dir, "hotcb.tune.recipe.yaml")
    summary_path = os.path.join(run_dir, "hotcb.tune.summary.json")
    if not os.path.exists(summary_path):
        print(f"No tune summary at {summary_path}")
        raise SystemExit(1)
    # Just copy the summary as a starting point
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ensure_dir(os.path.dirname(out) or ".")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Exported tune summary -> {out}")
    except Exception as e:
        print(f"Failed: {e}")
        raise SystemExit(1)


def cmd_recipe_validate(args: argparse.Namespace) -> None:
    """Validate a recipe file for schema correctness."""
    path = args.recipe or _recipe_path(args.dir)
    if not os.path.exists(path):
        print(f"Recipe file not found: {path}")
        raise SystemExit(1)
    errors: List[str] = []
    entries = 0
    required_fields = {"at", "module", "op"}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  line {i}: invalid JSON: {e}")
                continue
            entries += 1
            missing = required_fields - set(rec.keys())
            if missing:
                errors.append(f"  line {i}: missing fields: {missing}")
            at = rec.get("at", {})
            if not isinstance(at, dict) or "step" not in at:
                errors.append(f"  line {i}: 'at' must contain 'step'")
            if rec.get("module") not in {"cb", "opt", "loss", "tune"}:
                errors.append(f"  line {i}: module must be cb/opt/loss, got '{rec.get('module')}'")
    if errors:
        print(f"Recipe {path}: {len(errors)} errors in {entries} entries:")
        for e in errors:
            print(e)
        raise SystemExit(1)
    print(f"Recipe {path}: {entries} entries, valid.")


def cmd_recipe_patch_template(args: argparse.Namespace) -> None:
    """Generate a YAML patch template from a recipe file."""
    recipe_path = args.recipe
    output_path = args.output
    if not os.path.exists(recipe_path):
        print(f"Recipe file not found: {recipe_path}")
        raise SystemExit(1)

    seen: list = []
    seen_keys: set = set()
    try:
        with open(recipe_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("module"), rec.get("op"), rec.get("id"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    seen.append({"module": rec.get("module", ""), "op": rec.get("op", ""), "id": rec.get("id", "")})
    except FileNotFoundError:
        print(f"Recipe file not found: {recipe_path}")
        raise SystemExit(1)

    lines: List[str] = [
        f"# Generated from {recipe_path}",
        "patches:",
    ]
    for entry in seen:
        lines.append("  - match:")
        lines.append(f"      module: {entry['module']}")
        lines.append(f"      op: {entry['op']}")
        lines.append(f"      id: {entry['id']}")
        lines.append("    # replace_params: {}")
        lines.append("    # shift_step: 0")
        lines.append("    # drop: false")

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote patch template with {len(seen)} entries -> {output_path}")


def cmd_recipe_export(args: argparse.Namespace) -> None:
    applied = _applied_path(args.dir)
    out_path = args.out or _recipe_path(args.dir)
    entries: List[dict] = []
    try:
        with open(applied, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("decision") != "applied":
                    continue
                if rec.get("module") not in {"cb", "opt", "loss"}:
                    continue
                payload = rec.get("payload") or {}
                entry = {
                    "at": {"step": rec.get("step", 0), "event": rec.get("event", "train_step_end")},
                    "module": rec.get("module"),
                    "op": rec.get("op"),
                    "id": rec.get("id"),
                }
                # merge payload keys that map to op
                for k in ("params", "target", "init", "enabled"):
                    if k in payload:
                        entry[k] = payload[k]
                entries.append(entry)
    except FileNotFoundError:
        print(f"No applied ledger at {applied}")
        return

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in entries:
            f.write(json.dumps(rec) + "\n")
    print(f"Exported recipe with {len(entries)} entries -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hotcb", description="hotcb live training control plane")
    p.add_argument("--dir", default=".", help="Run directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Bootstrap run directory")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("status", help="Show run status")
    ps.set_defaults(func=cmd_status)

    # Syntactic sugar commands (spec §15.4)
    pen = sub.add_parser("enable", help="Enable a callback (sugar for cb enable)")
    pen.add_argument("id")
    pen.set_defaults(func=cmd_sugar_enable)

    pdis = sub.add_parser("disable", help="Disable a callback (sugar for cb disable)")
    pdis.add_argument("id")
    pdis.set_defaults(func=cmd_sugar_disable)

    pset_sugar = sub.add_parser("set", help="Set params (auto-routes to opt or loss)")
    pset_sugar.add_argument("--id", default="main")
    pset_sugar.add_argument("kv", nargs="*")
    pset_sugar.set_defaults(func=cmd_sugar_set)

    pf = sub.add_parser("freeze", help="Write freeze state file")
    pf.add_argument("--mode", choices=["off", "prod", "replay", "replay_adjusted"], required=True)
    pf.add_argument("--recipe", help="Recipe path for replay modes")
    pf.add_argument("--adjust", help="Adjustment overlay path")
    pf.add_argument("--policy", default="best_effort", choices=["best_effort", "strict"])
    pf.add_argument("--step-offset", type=int, default=0)
    pf.set_defaults(func=cmd_freeze)

    pr = sub.add_parser("recipe", help="Recipe utilities")
    sr = pr.add_subparsers(dest="recipe_cmd", required=True)
    pre = sr.add_parser("export", help="Export recipe from applied ledger")
    pre.add_argument("--out", help="Output path (default: runs/<dir>/hotcb.recipe.jsonl)")
    pre.set_defaults(func=cmd_recipe_export)

    prv = sr.add_parser("validate", help="Validate a recipe file")
    prv.add_argument("--recipe", help="Recipe path to validate")
    prv.set_defaults(func=cmd_recipe_validate)

    p_pt = sr.add_parser("patch-template", help="Generate adjust.yaml template from recipe")
    p_pt.add_argument("--recipe", default="hotcb.recipe.jsonl")
    p_pt.add_argument("--output", default="hotcb.adjust.yaml")
    p_pt.set_defaults(func=cmd_recipe_patch_template)

    pcb = sub.add_parser("cb", help="Callback module commands")
    pcb_sub = pcb.add_subparsers(dest="cb_command", required=True)
    for name in ["enable", "disable", "unload"]:
        ps = pcb_sub.add_parser(name)
        ps.add_argument("id")
        ps.set_defaults(func=cmd_cb)
    pset = pcb_sub.add_parser("set_params")
    pset.add_argument("id")
    pset.add_argument("kv", nargs="*")
    pset.set_defaults(func=cmd_cb)
    pload = pcb_sub.add_parser("load")
    pload.add_argument("id")
    pload.add_argument("--file", help="Python file path", dest="file")
    pload.add_argument("--path", help="Module path")
    pload.add_argument("--symbol", required=True)
    pload.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    pload.add_argument("--init", nargs="*", default=[])
    pload.set_defaults(func=cmd_cb)

    popt = sub.add_parser("opt", help="Optimizer control")
    opt_sub = popt.add_subparsers(dest="opt_command", required=True)
    for name in ["enable", "disable"]:
        po = opt_sub.add_parser(name)
        po.add_argument("--id", default="main")
        po.set_defaults(func=cmd_opt)
    pset_opt = opt_sub.add_parser("set_params")
    pset_opt.add_argument("--id", default="main")
    pset_opt.add_argument("kv", nargs="*")
    pset_opt.set_defaults(func=cmd_opt)

    ploss = sub.add_parser("loss", help="Loss control")
    loss_sub = ploss.add_subparsers(dest="loss_command", required=True)
    for name in ["enable", "disable"]:
        pl = loss_sub.add_parser(name)
        pl.add_argument("--id", default="main")
        pl.set_defaults(func=cmd_loss)
    pset_loss = loss_sub.add_parser("set_params")
    pset_loss.add_argument("--id", default="main")
    pset_loss.add_argument("kv", nargs="*")
    pset_loss.set_defaults(func=cmd_loss)

    pbench = sub.add_parser("bench", help="Run benchmarks")
    bench_sub = pbench.add_subparsers(dest="bench_cmd")

    # `hotcb bench run` — original benchmark runner
    pbench_run = bench_sub.add_parser("run", help="Run benchmark conditions")
    pbench_run.add_argument("--task", default="synthetic_quadratic",
                            help="Task name (synthetic_quadratic, synthetic_classification, cifar10_resnet20)")
    pbench_run.add_argument("--output-dir", default="./bench_output", help="Output directory")
    pbench_run.add_argument("--conditions", default="baseline,auto_tune",
                            help="Comma-separated conditions to run")
    pbench_run.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    pbench_run.set_defaults(func=cmd_bench)

    # `hotcb bench eval` — autopilot evaluation
    pbench_eval = bench_sub.add_parser("eval", help="Run autopilot evaluation against published benchmark")
    pbench_eval.add_argument("--task", default="cifar10_resnet20",
                             help="Task name (cifar10_resnet20)")
    pbench_eval.add_argument("--output-dir", default="./eval_output", help="Output directory")
    pbench_eval.add_argument("--phases", default="baseline,autopilot",
                             help="Comma-separated phases: baseline, autopilot")
    pbench_eval.add_argument("--guidelines", default=None,
                             help="Path to YAML guidelines file for autopilot rules")
    pbench_eval.set_defaults(func=cmd_bench_eval)

    # Also allow bare `hotcb bench` to fall through to `run` for backwards compat
    pbench.add_argument("--task", default="synthetic_quadratic",
                        help="Task name (synthetic_quadratic, synthetic_classification, cifar10_resnet20)")
    pbench.add_argument("--output-dir", default="./bench_output", help="Output directory")
    pbench.add_argument("--conditions", default="baseline,auto_tune",
                        help="Comma-separated conditions to run")
    pbench.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    pbench.set_defaults(func=cmd_bench)

    # --- Scenario subcommand ---
    pscenario = sub.add_parser("scenario", help="Run policy pack scenario tests")
    scenario_sub = pscenario.add_subparsers(dest="scenario_cmd", required=True)

    psc_list = scenario_sub.add_parser("list", help="List available scenarios")
    psc_list.set_defaults(func=cmd_scenario_list)

    psc_run = scenario_sub.add_parser("run", help="Run a scenario (or all)")
    psc_run.add_argument("name", nargs="?", default=None, help="Scenario name (omit for --all)")
    psc_run.add_argument("--all", action="store_true", help="Run all scenarios")
    psc_run.add_argument("--pack", default=None, help="Run all scenarios for a given pack")
    psc_run.add_argument("--dashboard", action="store_true", help="Run with live dashboard")
    psc_run.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    psc_run.add_argument("--port", type=int, default=8421, help="Dashboard bind port")
    psc_run.add_argument("--step-delay", type=float, default=0.0, help="Seconds between steps (0 for headless)")
    psc_run.set_defaults(func=cmd_scenario_run)

    # --- Eval subcommand ---
    peval = sub.add_parser("eval", help="Run evaluation harness (controlled experiments)")
    eval_sub = peval.add_subparsers(dest="eval_cmd", required=True)

    peval_list = eval_sub.add_parser("list", help="List available conditions")
    peval_list.add_argument("--include-synthetic", action="store_true",
                            help="Also list golden/finetune/simple synthetic demo conditions")
    peval_list.add_argument("--conditions-file", default=None,
                            help="YAML file with custom conditions to include in listing")
    peval_list.set_defaults(func=cmd_eval_list)

    peval_run = eval_sub.add_parser("run", help="Run evaluation conditions")
    peval_run.add_argument("--output-dir", default="./eval_output", help="Output directory for results")
    peval_run.add_argument("--conditions", default=None,
                           help="Comma-separated condition names (default: all real)")
    peval_run.add_argument("--demo", default=None, choices=["golden", "finetune", "simple", "mnist", "cifar10"],
                           help="Run all conditions for a specific demo")
    peval_run.add_argument("--conditions-file", default=None,
                           help="YAML file with custom conditions (see docs/eval.md)")
    peval_run.add_argument("--include-synthetic", action="store_true",
                           help="Include golden/finetune/simple synthetic demo conditions")
    peval_run.add_argument("--max-steps", type=int, default=None, help="Steps per condition (default: task-specific)")
    peval_run.add_argument("--step-delay", type=float, default=0.0, help="Delay between steps (0 for headless)")
    peval_run.add_argument("--no-serve", action="store_true", help="Skip live dashboard (headless mode)")
    peval_run.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    peval_run.add_argument("--port", type=int, default=8421, help="Dashboard bind port")
    peval_run.set_defaults(func=cmd_eval_run)

    peval_report = eval_sub.add_parser("report", help="Print report from saved results")
    peval_report.add_argument("--output-dir", default="./eval_output", help="Output directory with results")
    peval_report.set_defaults(func=cmd_eval_report)

    peval_serve = eval_sub.add_parser("serve", help="Serve dashboard for eval results")
    peval_serve.add_argument("--output-dir", default="./eval_output", help="Output directory with results")
    peval_serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    peval_serve.add_argument("--port", type=int, default=8421, help="Bind port")
    peval_serve.set_defaults(func=cmd_eval_serve)

    pdemo = sub.add_parser("demo", help="Launch synthetic training with live dashboard")
    pdemo.add_argument("--golden", action="store_true",
                       help="Run the golden demo (multi-task with recipe-driven loss shifts and feature capture)")
    pdemo.add_argument("--host", default="0.0.0.0", help="Bind host")
    pdemo.add_argument("--port", type=int, default=8421, help="Bind port")
    pdemo.add_argument("--max-steps", type=int, default=500, help="Number of training steps")
    pdemo.add_argument("--max-time", type=float, default=None, help="Wall-clock time limit in seconds (stops training when reached)")
    pdemo.add_argument("--step-delay", type=float, default=0.15, help="Seconds between steps")
    pdemo.add_argument("--demo-dir", default=None, help="Run directory (default: temp dir)")
    pdemo.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                        default="off", help="Start with autopilot mode enabled")
    pdemo.add_argument("--key-metric", default=None, help="Primary optimization metric for AI autopilot")
    pdemo.add_argument("--scenario", default=None, help="Run a named scenario with dashboard (alias for scenario run --dashboard)")
    pdemo.set_defaults(func=cmd_demo)

    pserve = sub.add_parser("serve", help="Start the live dashboard server")
    pserve.add_argument("--host", default="0.0.0.0", help="Bind host")
    pserve.add_argument("--port", type=int, default=8421, help="Bind port")
    pserve.add_argument("--poll-interval", type=float, default=0.5, help="JSONL poll interval (seconds)")
    pserve.add_argument("--dirs", help="Comma-separated additional run dirs for multi-run comparison")
    pserve.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                         default="off", help="Start with autopilot mode enabled (server only, no training)")
    pserve.add_argument("--key-metric", default=None, help="Primary optimization metric for AI autopilot")
    pserve.set_defaults(func=cmd_serve)

    plaunch = sub.add_parser("launch", help="Start training + dashboard + autopilot in one command")
    plaunch.add_argument("--config", default="multitask", help="Built-in config: simple, multitask, finetune")
    plaunch.add_argument("--config-file", default='hotcb.launch.json', help="Path to hotcb.launch.json (values used as defaults, CLI flags override)")
    plaunch.add_argument("--train-fn", default=None, help="Custom training function (module.path:fn_name)")
    plaunch.add_argument("--host", default="0.0.0.0", help="Bind host")
    plaunch.add_argument("--port", type=int, default=8421, help="Bind port")
    plaunch.add_argument("--max-steps", type=int, default=None, help="Number of training steps (default: from config file, or 1000)")
    plaunch.add_argument("--max-time", type=float, default=None, help="Wall-clock time limit in seconds (stops training when reached)")
    plaunch.add_argument("--step-delay", type=float, default=None, help="Seconds between steps (default: from config file, or 0.1)")
    plaunch.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                          default="off", help="Autopilot mode")
    plaunch.add_argument("--key-metric", default="val_loss", help="Primary optimization metric")
    plaunch.add_argument("--ai-model", default="gpt-4o-mini", help="LLM model for AI autopilot")
    plaunch.add_argument("--ai-budget", type=float, default=5.0, help="Max USD for AI calls")
    plaunch.add_argument("--ai-cadence", type=int, default=50, help="Steps between AI check-ins")
    plaunch.add_argument("--seed", type=int, default=None, help="Random seed")
    plaunch.set_defaults(func=cmd_launch)

    # --- Research subcommand ---
    presearch = sub.add_parser("research", help="Research & learning module")
    research_sub = presearch.add_subparsers(dest="research_cmd", required=True)

    # research stream
    prs = research_sub.add_parser("stream", help="Manage research streams")
    rs_sub = prs.add_subparsers(dest="stream_cmd", required=True)

    prs_list = rs_sub.add_parser("list", help="List streams")
    prs_list.set_defaults(func=cmd_research_stream_list)

    prs_new = rs_sub.add_parser("new", help="Create a stream")
    prs_new.add_argument("name", help="Stream name")
    prs_new.add_argument("--description", default="", help="Stream description")
    prs_new.set_defaults(func=cmd_research_stream_new)

    prs_conclude = rs_sub.add_parser("conclude", help="Conclude a stream")
    prs_conclude.add_argument("stream_id", help="Stream ID")
    prs_conclude.add_argument("--conclusion", required=True, help="Conclusion text")
    prs_conclude.set_defaults(func=cmd_research_stream_conclude)

    # research observe
    pobs = research_sub.add_parser("observe", help="Record an observation")
    pobs.add_argument("text", help="Observation text")
    pobs.add_argument("--stream", default=None, help="Stream ID")
    pobs.add_argument("--tags", default="", help="Comma-separated tags")
    pobs.add_argument("--step", type=int, default=0, help="Training step")
    pobs.set_defaults(func=cmd_research_observe)

    # research hyp
    phyp = research_sub.add_parser("hyp", help="Hypothesis operations")
    hyp_sub = phyp.add_subparsers(dest="hyp_cmd", required=True)

    phyp_add = hyp_sub.add_parser("add", help="Add a hypothesis")
    phyp_add.add_argument("--condition", required=True, help="Condition expression")
    phyp_add.add_argument("--intervention", required=True, help="Intervention JSON")
    phyp_add.add_argument("--expected", required=True, help="Expected outcome")
    phyp_add.add_argument("--stream", default=None, help="Stream ID")
    phyp_add.set_defaults(func=cmd_research_hyp_add)

    phyp_list = hyp_sub.add_parser("list", help="List hypotheses")
    phyp_list.add_argument("--status", default=None, help="Filter by status")
    phyp_list.set_defaults(func=cmd_research_hyp_list)

    phyp_show = hyp_sub.add_parser("show", help="Show hypothesis details")
    phyp_show.add_argument("hyp_id", help="Hypothesis ID")
    phyp_show.set_defaults(func=cmd_research_hyp_show)

    phyp_test = hyp_sub.add_parser("test", help="Test a hypothesis")
    phyp_test.add_argument("hyp_id", help="Hypothesis ID")
    phyp_test.set_defaults(func=cmd_research_hyp_test)

    phyp_conclude = hyp_sub.add_parser("conclude", help="Conclude a hypothesis")
    phyp_conclude.add_argument("hyp_id", help="Hypothesis ID")
    phyp_conclude.add_argument("--status", required=True,
                               choices=["confirmed", "refuted", "inconclusive", "archived"])
    phyp_conclude.set_defaults(func=cmd_research_hyp_conclude)

    # research model
    pmodel = research_sub.add_parser("model", help="NN model operations")
    model_sub = pmodel.add_subparsers(dest="model_cmd", required=True)

    pmodel_status = model_sub.add_parser("status", help="Show model status")
    pmodel_status.set_defaults(func=cmd_research_model_status)

    # research export/import/merge
    prexp = research_sub.add_parser("export", help="Export research graph")
    prexp.add_argument("--out", required=True, help="Output JSON path")
    prexp.add_argument("--stream", default=None, help="Export single stream")
    prexp.set_defaults(func=cmd_research_export)

    primp = research_sub.add_parser("import", help="Import research graph")
    primp.add_argument("path", help="Import JSON path")
    primp.set_defaults(func=cmd_research_import)

    prmerge = research_sub.add_parser("merge", help="Merge from another run")
    prmerge.add_argument("run_dir", help="Source run directory")
    prmerge.set_defaults(func=cmd_research_merge)

    prrecipe = research_sub.add_parser("export-recipe", help="Export confirmed hypotheses as recipe")
    prrecipe.add_argument("--out", default=None, help="Output JSONL path")
    prrecipe.set_defaults(func=cmd_research_export_recipe)

    # research load-conditions
    prlc = research_sub.add_parser("load-conditions", help="Load continuation conditions as hypotheses")
    prlc.add_argument("--task", required=True, choices=["mnist", "cifar10"], help="Task name")
    prlc.add_argument("--base-run", default="", help="Base run directory")
    prlc.set_defaults(func=cmd_research_load_conditions)

    # research launch-tests
    prlt = research_sub.add_parser("launch-tests", help="Launch continuation tests for hypotheses")
    prlt.add_argument("--hyps", required=True, help="Comma-separated hypothesis IDs")
    prlt.add_argument("--base-run", required=True, help="Base run directory with checkpoints")
    prlt.set_defaults(func=cmd_research_launch_tests)

    # ── continue (continuation tuning) ──
    pcont = sub.add_parser("continue", help="Continuation tuning from converged checkpoints")
    cont_sub = pcont.add_subparsers(dest="continue_command", required=True)

    pcont_run = cont_sub.add_parser("run", help="Run continuation routine")
    pcont_run.add_argument("--run", required=True, help="Base run directory with checkpoints/")
    pcont_run.add_argument("--task", default="mnist", choices=["mnist", "cifar10", "coco_mobilenet", "imagenet_mobilenet", "imagenet_mobilenetv2_paper", "coco_detection", "clip_coco"])
    pcont_run.add_argument("--metric", default="val_accuracy", help="Primary metric")
    pcont_run.add_argument("--mode", default="max", choices=["min", "max"])
    pcont_run.add_argument("--recipes", nargs="*", help="Recipe names or groups: default, aggressive, combo, multi_stage, all")
    pcont_run.add_argument("--resume-modes", nargs="*", default=["full_resume"],
                           choices=["full_resume", "reset_scheduler", "weights_only"])
    pcont_run.add_argument("--max-anchors", type=int, default=2)
    pcont_run.add_argument("--branches-per-anchor", type=int, default=6)
    pcont_run.add_argument("--extra-steps", type=int, default=500)
    pcont_run.add_argument("--autopilot", default="off", choices=["off", "auto", "suggest"])
    pcont_run.add_argument("--packs", nargs="*", default=[])
    pcont_run.add_argument("--out", default=None, help="Output directory")
    pcont_run.set_defaults(func=cmd_continue_run)

    pcont_report = cont_sub.add_parser("report", help="Show continuation report")
    pcont_report.add_argument("--dir", dest="cont_dir", required=True, help="Continuation output dir")
    pcont_report.set_defaults(func=cmd_continue_report)

    pcont_baseline = cont_sub.add_parser("baseline", help="Train a baseline with checkpoints")
    pcont_baseline.add_argument("--task", default="mnist", choices=["mnist", "cifar10", "coco_mobilenet", "imagenet_mobilenet", "imagenet_mobilenetv2_paper", "coco_detection", "clip_coco"])
    pcont_baseline.add_argument("--steps", type=int, default=None, help="Max steps (default: task default)")
    pcont_baseline.add_argument("--out", required=True, help="Output run directory")
    pcont_baseline.add_argument("--ckpt-interval", type=int, default=250, help="Checkpoint save interval")
    pcont_baseline.set_defaults(func=cmd_continue_baseline)

    ptune = sub.add_parser("tune", help="Tune module control")
    tune_sub = ptune.add_subparsers(dest="tune_command", required=True)
    pt_enable = tune_sub.add_parser("enable", help="Enable tuning")
    pt_enable.add_argument("--mode", default="active", choices=["active", "observe", "suggest"])
    pt_enable.set_defaults(func=cmd_tune)
    pt_disable = tune_sub.add_parser("disable", help="Disable tuning")
    pt_disable.set_defaults(func=cmd_tune)
    pt_status = tune_sub.add_parser("status", help="Show tune status")
    pt_status.set_defaults(func=cmd_tune_status)
    pt_set = tune_sub.add_parser("set", help="Set tune recipe params")
    pt_set.add_argument("kv", nargs="*")
    pt_set.set_defaults(func=cmd_tune)
    pt_export = tune_sub.add_parser("export-recipe", help="Export tune recipe")
    pt_export.add_argument("--out", help="Output path")
    pt_export.set_defaults(func=cmd_tune_export_recipe)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
