"""Evaluation harness — runs conditions headlessly and collects results."""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .conditions import EvalCondition


@dataclass
class EvalResult:
    """Result of a single evaluation run."""

    condition_name: str
    demo: str
    description: str

    # Final metric values
    final_metrics: Dict[str, float] = field(default_factory=dict)
    # Full metric history (step -> metrics dict)
    metric_history: List[Dict[str, Any]] = field(default_factory=list)
    # Applied mutations (from applied.jsonl)
    applied: List[Dict[str, Any]] = field(default_factory=list)
    # Autopilot actions fired
    autopilot_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Summary stats
    total_steps: int = 0
    intervention_count: int = 0
    elapsed_seconds: float = 0.0
    run_dir: str = ""
    seed: Optional[int] = None

    # Research graph stats (if research engine was active)
    research_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class EvalHarness:
    """Run evaluation conditions headlessly — no dashboard server needed."""

    def __init__(self, output_dir: Optional[str] = None):
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="hotcb_eval_")
        os.makedirs(self._output_dir, exist_ok=True)
        self._results: List[EvalResult] = []

    @property
    def results(self) -> List[EvalResult]:
        return list(self._results)

    def run(self, condition: EvalCondition) -> EvalResult:
        """Run a single evaluation condition. Returns EvalResult."""
        run_dir = os.path.join(self._output_dir, condition.name)
        os.makedirs(run_dir, exist_ok=True)

        # Bootstrap files
        for fname in [
            "hotcb.commands.jsonl",
            "hotcb.applied.jsonl",
            "hotcb.metrics.jsonl",
        ]:
            open(os.path.join(run_dir, fname), "w").close()
        with open(os.path.join(run_dir, "hotcb.freeze.json"), "w") as f:
            json.dump({"mode": "off"}, f)

        # Sentinel to suppress demo recipe injection
        if not condition.use_recipe:
            open(os.path.join(run_dir, "hotcb.eval.no_recipe"), "w").close()

        # Set seed
        if condition.seed is not None:
            random.seed(condition.seed)

        # Get training function + default steps
        train_fn, default_steps = _get_train_fn(condition.demo)
        max_steps = condition.max_steps or default_steps

        # Write initial param overrides to commands.jsonl (applied at step 1)
        _write_initial_overrides(run_dir, condition.initial_overrides)

        # Write scheduled commands file (picked up by shim during training)
        if condition.scheduled_commands:
            _write_scheduled_commands(run_dir, condition.scheduled_commands)

        # Write recipe entries if condition says to (demo-specific)
        if condition.use_recipe:
            _write_recipe(run_dir, condition.demo)

        # Set up autopilot if enabled
        autopilot_engine = None
        autopilot_actions = []
        if condition.autopilot != "off":
            autopilot_engine = _create_autopilot(condition, run_dir)

        # Set up research engine
        research_engine = _create_research_engine(run_dir, condition)

        # Run training (with scheduled command injection if needed)
        stop_event = threading.Event()
        t0 = time.monotonic()

        sidecars = []
        if condition.scheduled_commands:
            injector = _ScheduledInjector(run_dir, condition.scheduled_commands)
            injector.start()
            sidecars.append(injector)

        # Inline autopilot: sidecar thread reads metrics, evaluates rules,
        # writes corrective commands to commands.jsonl during training
        if autopilot_engine is not None:
            ap_sidecar = _AutopilotSidecar(
                run_dir, autopilot_engine, research_engine,
            )
            ap_sidecar.start()
            sidecars.append(ap_sidecar)

        train_fn(
            run_dir,
            max_steps=max_steps,
            step_delay=condition.step_delay,
            _stop_event=stop_event,
        )

        for sc in sidecars:
            sc.stop()

        elapsed = time.monotonic() - t0

        # Collect autopilot actions from sidecar
        if autopilot_engine is not None and hasattr(ap_sidecar, 'actions'):
            autopilot_actions = ap_sidecar.actions

        # Collect results
        metrics_history = _read_jsonl(os.path.join(run_dir, "hotcb.metrics.jsonl"))
        applied = _read_jsonl(os.path.join(run_dir, "hotcb.applied.jsonl"))

        final_metrics = {}
        if metrics_history:
            last = metrics_history[-1]
            final_metrics = last.get("metrics", last)

        # Research stats
        research_stats = {}
        if research_engine:
            research_stats = research_engine.graph.stats()

        result = EvalResult(
            condition_name=condition.name,
            demo=condition.demo,
            description=condition.description,
            final_metrics=final_metrics,
            metric_history=metrics_history,
            applied=applied,
            autopilot_actions=autopilot_actions,
            total_steps=max_steps,
            intervention_count=len(applied) + len(autopilot_actions),
            elapsed_seconds=round(elapsed, 2),
            run_dir=run_dir,
            seed=condition.seed,
            research_stats=research_stats,
        )

        self._results.append(result)
        return result

    def run_all(self, conditions: List[EvalCondition]) -> List[EvalResult]:
        """Run all conditions sequentially."""
        results = []
        for cond in conditions:
            results.append(self.run(cond))
        return results

    def save_results(self, path: Optional[str] = None):
        """Save all results to a JSON file."""
        path = path or os.path.join(self._output_dir, "eval_results.json")
        data = [r.to_dict() for r in self._results]
        # Strip verbose history for the summary file
        for d in data:
            d["metric_history_length"] = len(d.pop("metric_history", []))
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path


# ═══════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════

def _get_train_fn(demo: str):
    """Return (training_function, default_max_steps) for a demo."""
    if demo == "golden":
        from hotcb.golden_demo import _golden_training
        return _golden_training, 800
    elif demo == "finetune":
        from hotcb.finetune_demo import _finetune_training
        return _finetune_training, 600
    elif demo == "simple":
        from hotcb.demo import _demo_training
        return _demo_training, 500
    elif demo == "mnist":
        from hotcb.eval.tasks import mnist_training
        return mnist_training, 1500
    elif demo == "cifar10":
        from hotcb.eval.tasks import cifar10_training
        return cifar10_training, 2000
    elif demo == "imagenet_mobilenet":
        from hotcb.eval.tasks_imagenet import imagenet_mobilenet_training
        return imagenet_mobilenet_training, 10000
    elif demo == "imagenet_mobilenetv2_paper":
        from hotcb.eval.tasks_imagenet import imagenet_mobilenetv2_paper_training
        return imagenet_mobilenetv2_paper_training, 13345  # ~1 epoch
    elif demo == "coco_detection":
        from hotcb.eval.tasks_coco_detection import coco_ssdlite_mobilenetv2_training
        return coco_ssdlite_mobilenetv2_training, 5000
    else:
        raise ValueError(f"Unknown demo: {demo}")


class _ScheduledInjector:
    """Background thread that injects commands at scheduled steps by watching metrics."""

    def __init__(self, run_dir: str, scheduled: Dict[int, Dict[str, Any]]):
        self._run_dir = run_dir
        self._scheduled = dict(scheduled)  # step -> command
        self._fired: set = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        metrics_path = os.path.join(self._run_dir, "hotcb.metrics.jsonl")
        commands_path = os.path.join(self._run_dir, "hotcb.commands.jsonl")
        last_pos = 0

        while not self._stop.is_set():
            if not os.path.exists(metrics_path):
                time.sleep(0.01)
                continue

            # Read new metrics to find current step
            try:
                with open(metrics_path) as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
            except OSError:
                time.sleep(0.01)
                continue

            current_step = 0
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    s = rec.get("step", 0)
                    if s > current_step:
                        current_step = s
                except json.JSONDecodeError:
                    continue

            # Inject scheduled commands
            for step, cmd in list(self._scheduled.items()):
                if step <= current_step and step not in self._fired:
                    self._fired.add(step)
                    try:
                        with open(commands_path, "a") as f:
                            f.write(json.dumps(cmd) + "\n")
                    except OSError:
                        pass

            # All fired?
            if len(self._fired) >= len(self._scheduled):
                break

            time.sleep(0.005)


class _AutopilotSidecar:
    """Background thread that runs autopilot inline during training.

    Watches metrics.jsonl, evaluates autopilot rules, writes corrective
    commands to commands.jsonl so the kernel applies them next step.
    """

    def __init__(self, run_dir: str, autopilot_engine, research_engine=None):
        self._run_dir = run_dir
        self._autopilot = autopilot_engine
        self._research = research_engine
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.actions: List[Dict[str, Any]] = []

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        metrics_path = os.path.join(self._run_dir, "hotcb.metrics.jsonl")
        commands_path = os.path.join(self._run_dir, "hotcb.commands.jsonl")
        last_pos = 0

        while not self._stop_event.is_set():
            if not os.path.exists(metrics_path):
                time.sleep(0.005)
                continue

            try:
                with open(metrics_path) as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
            except OSError:
                time.sleep(0.005)
                continue

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = rec.get("step")
                metrics = rec.get("metrics", rec)
                if step is None:
                    continue

                try:
                    # evaluate() in "auto" mode writes corrections to commands.jsonl
                    actions = self._autopilot.evaluate(int(step), metrics)
                    if actions:
                        from dataclasses import asdict
                        for a in actions:
                            ad = asdict(a)
                            ad["eval_step"] = step
                            self.actions.append(ad)

                            # Feed into research engine
                            if self._research:
                                try:
                                    self._research.on_rule_fired(a, int(step))
                                except Exception:
                                    pass

                    # Feed metrics into research engine
                    if self._research:
                        try:
                            self._research.on_metrics(int(step), metrics)
                        except Exception:
                            pass
                except Exception:
                    pass

            time.sleep(0.005)


def _write_scheduled_commands(run_dir: str, scheduled: Dict[int, Dict[str, Any]]):
    """Write a reference file of scheduled commands (for debugging/logging)."""
    path = os.path.join(run_dir, "hotcb.eval.scheduled.json")
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in scheduled.items()}, f, indent=2)


def _write_initial_overrides(run_dir: str, overrides: Dict[str, Any]):
    """Write param overrides as commands at step 0."""
    if not overrides:
        return
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(commands_path, "a") as f:
        for module, params in overrides.items():
            cmd = {
                "module": module,
                "op": "set_params",
                "params": params,
                "source": "eval_override",
            }
            f.write(json.dumps(cmd) + "\n")


def _write_recipe(run_dir: str, demo: str):
    """Write the demo's built-in recipe schedule to commands.jsonl.

    Note: The golden/finetune demos write their own recipe internally
    (at scheduled steps). When use_recipe=True, we do nothing extra —
    the demo function itself handles it. When use_recipe=False, we need
    to prevent the demo from writing recipe commands.

    Since we can't easily prevent the demo from writing its recipe,
    we use a different approach: we always let the demo write its recipe,
    and for use_recipe=False conditions, we truncate the commands file
    after initial overrides so the recipe entries get overwritten.

    Actually: the demos write recipe commands at scheduled steps during
    training (not upfront). So use_recipe flag doesn't need special handling
    for the existing demo code — the demo always writes its recipe.

    For a true baseline (no recipe), we need demo training functions that
    can be told to skip recipe injection. We'll handle this by checking
    for a sentinel file.
    """
    # For use_recipe=True, nothing to do — demo writes its own
    pass


def _create_autopilot(condition: EvalCondition, run_dir: str):
    """Create an AutopilotEngine with specified policy packs."""
    try:
        from hotcb.server.autopilot import AutopilotEngine
        engine = AutopilotEngine(run_dir=run_dir, mode=condition.autopilot)
        for pack_name in condition.policy_packs:
            try:
                engine.load_pack(pack_name)
            except Exception:
                pass
        return engine
    except Exception:
        return None


def _create_research_engine(run_dir: str, condition: EvalCondition):
    """Create a ResearchEngine for the eval run."""
    try:
        from hotcb.research.engine import ResearchEngine
        return ResearchEngine(run_dir, nn_mode=condition.nn_mode)
    except Exception:
        return None


def _run_autopilot_posthoc(
    run_dir: str,
    autopilot_engine,
    research_engine,
) -> List[Dict[str, Any]]:
    """Replay metrics through autopilot engine post-hoc and collect actions.

    This is an approximation — real autopilot runs inline. Post-hoc replay
    evaluates rules on the same metric history, but can't apply the corrective
    actions (since training already happened). This tells us what the autopilot
    *would have done*, which is useful for verifying rule firing patterns.
    """
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []

    actions_collected = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            metrics = rec.get("metrics", rec)
            if step is None:
                continue

            try:
                actions = autopilot_engine.evaluate(int(step), metrics)
                if actions:
                    from dataclasses import asdict
                    for a in actions:
                        ad = asdict(a)
                        ad["eval_step"] = step
                        actions_collected.append(ad)
                        # Feed into research engine
                        if research_engine:
                            try:
                                research_engine.on_rule_fired(a, int(step))
                            except Exception:
                                pass

                # Feed metrics into research engine
                if research_engine:
                    try:
                        research_engine.on_metrics(int(step), metrics)
                    except Exception:
                        pass
            except Exception:
                pass

    return actions_collected


def _read_jsonl(path: str) -> list:
    """Read a JSONL file into a list of dicts."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
