"""
hotcb.scenarios.runner — Runs scenario training with concurrent autopilot.

The ScenarioRunner starts a training thread and an autopilot tailer thread.
Training writes metrics via HotKernel.  The tailer polls the JSONL, feeds
``AutopilotEngine.evaluate()``, and writes commands that the kernel picks
up on the next step.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import ScenarioConfig, ScenarioResult
from . import scenarios_root

log = logging.getLogger("hotcb.scenarios.runner")


class ScenarioRunner:
    """Execute scenario training with concurrent rule-based autopilot."""

    def __init__(self, *, step_delay: float = 0.0, verbose: bool = False):
        self.step_delay = step_delay
        self.verbose = verbose

    def run(
        self,
        config: ScenarioConfig,
        *,
        run_dir: Optional[str] = None,
        dashboard: bool = False,
        host: str = "0.0.0.0",
        port: int = 8421,
    ) -> ScenarioResult:
        """Run a single scenario end-to-end.

        Parameters
        ----------
        config : ScenarioConfig
            Scenario metadata (pack, expected rules, etc.).
        run_dir : str | None
            Where to store run artifacts.  Uses a temp dir if None.
        dashboard : bool
            If True, use the launch() API to serve the dashboard.
        host / port : str / int
            Dashboard bind address (only used when *dashboard* is True).
        """
        if dashboard:
            return self._run_with_dashboard(config, host=host, port=port)
        return self._run_headless(config, run_dir=run_dir)

    # ------------------------------------------------------------------
    # Headless execution (training thread + autopilot tailer thread)
    # ------------------------------------------------------------------

    def _run_headless(
        self, config: ScenarioConfig, *, run_dir: Optional[str] = None,
    ) -> ScenarioResult:
        own_dir = run_dir is None
        if own_dir:
            run_dir = tempfile.mkdtemp(prefix=f"hotcb_scenario_{config.name}_")

        try:
            return self._execute(config, run_dir)
        finally:
            if own_dir:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _execute(self, config: ScenarioConfig, run_dir: str) -> ScenarioResult:
        from hotcb.server.autopilot import AutopilotEngine

        # Seed the run directory
        self._seed_run_dir(run_dir, config)

        # Load the training function from the scenario
        train_fn = self._load_train_fn(config)

        # Set up autopilot engine
        engine = AutopilotEngine(run_dir=run_dir, mode="auto")
        engine.load_pack(config.pack)

        # Shared state
        stop_event = threading.Event()
        actions_fired: List[str] = []
        applied_cmds: List[Dict] = []
        train_error: List[str] = []

        # Autopilot tailer: polls metrics JSONL and evaluates rules
        tailer_pos = [0]  # mutable to share with drain

        def _process_lines(metrics_path, commands_path):
            """Read new JSONL lines and evaluate autopilot rules."""
            if not os.path.exists(metrics_path):
                return
            with open(metrics_path, "r", encoding="utf-8") as f:
                f.seek(tailer_pos[0])
                lines = f.readlines()
                tailer_pos[0] = f.tell()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step", 0)
                metrics = record.get("metrics", {})
                actions = engine.evaluate(step, metrics)
                for action in actions:
                    actions_fired.append(action.rule_id)
                    if action.status == "applied":
                        cmd = {
                            "module": action.proposed_action.get("module", "opt"),
                            "op": action.proposed_action.get("op", "set_params"),
                            "params": action.proposed_action.get("params", {}),
                            "source": "autopilot",
                            "rule_id": action.rule_id,
                            "ts": time.time(),
                        }
                        applied_cmds.append(cmd)
                        with open(commands_path, "a", encoding="utf-8") as cf:
                            cf.write(json.dumps(cmd) + "\n")

        def autopilot_tailer():
            metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
            commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
            while not stop_event.is_set():
                try:
                    _process_lines(metrics_path, commands_path)
                except Exception as exc:
                    if not stop_event.is_set():
                        log.debug("Tailer error: %s", exc)
                time.sleep(0.01)

        # Training thread
        def train_wrapper():
            try:
                train_fn(
                    run_dir=run_dir,
                    max_steps=config.max_steps,
                    step_delay=self.step_delay,
                    stop_event=stop_event,
                )
            except Exception as exc:
                train_error.append(str(exc))
            finally:
                stop_event.set()

        tailer_thread = threading.Thread(
            target=autopilot_tailer, daemon=True, name="scenario-tailer",
        )
        train_thread = threading.Thread(
            target=train_wrapper, daemon=True, name="scenario-train",
        )

        tailer_thread.start()
        train_thread.start()

        # Wait for training to finish
        train_thread.join(timeout=config.max_steps * max(self.step_delay, 0.01) + 30)
        stop_event.set()
        tailer_thread.join(timeout=2)

        # Final drain: process any remaining JSONL lines the tailer missed
        metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
        commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        try:
            _process_lines(metrics_path, commands_path)
        except Exception:
            pass

        # Determine pass/fail
        unique_fired = list(dict.fromkeys(actions_fired))  # preserve order, dedup
        expected_met = all(r in unique_fired for r in config.expected_rules)
        error = train_error[0] if train_error else None

        return ScenarioResult(
            name=config.name,
            passed=expected_met and error is None,
            rules_fired=unique_fired,
            expected_rules=config.expected_rules,
            steps_run=config.max_steps,
            error=error,
            applied_commands=applied_cmds,
        )

    # ------------------------------------------------------------------
    # Dashboard execution (uses launch() API)
    # ------------------------------------------------------------------

    def _run_with_dashboard(
        self,
        config: ScenarioConfig,
        *,
        host: str = "0.0.0.0",
        port: int = 8421,
    ) -> ScenarioResult:
        from hotcb.launch import launch

        train_fn = self._load_train_fn(config)

        # Wrap to match launch() contract
        def _wrapper(run_dir, max_steps, step_delay, stop_event):
            self._seed_run_dir(run_dir, config)
            train_fn(
                run_dir=run_dir,
                max_steps=max_steps,
                step_delay=step_delay,
                stop_event=stop_event,
            )

        handle = launch(
            train_fn=_wrapper,
            autopilot="auto",
            key_metric=config.key_metric,
            max_steps=config.max_steps,
            step_delay=self.step_delay if self.step_delay > 0 else 0.05,
            host=host,
            port=port,
            serve=True,
            block=True,
        )

        # Check applied log for rule firings
        applied = handle.applied(last_n=500)
        rules_fired = list(dict.fromkeys(
            cmd.get("rule_id", "")
            for cmd in applied
            if cmd.get("source") == "autopilot" and cmd.get("rule_id")
        ))
        expected_met = all(r in rules_fired for r in config.expected_rules)

        return ScenarioResult(
            name=config.name,
            passed=expected_met,
            rules_fired=rules_fired,
            expected_rules=config.expected_rules,
            steps_run=config.max_steps,
            applied_commands=applied,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_run_dir(run_dir: str, config: ScenarioConfig) -> None:
        """Bootstrap the run directory with empty JSONL files."""
        os.makedirs(run_dir, exist_ok=True)
        for fname in [
            "hotcb.commands.jsonl",
            "hotcb.applied.jsonl",
            "hotcb.metrics.jsonl",
            "hotcb.recipe.jsonl",
        ]:
            path = os.path.join(run_dir, fname)
            if not os.path.exists(path):
                open(path, "w").close()
        freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
        if not os.path.exists(freeze_path):
            with open(freeze_path, "w") as f:
                json.dump({"mode": "off"}, f)

        # Copy artifact data if specified
        if config.artifact_dir:
            artifact_path = scenarios_root() / config.artifact_dir
            if artifact_path.is_dir():
                for item in artifact_path.iterdir():
                    dest = os.path.join(run_dir, item.name)
                    if item.is_file():
                        shutil.copy2(str(item), dest)

    @staticmethod
    def _load_train_fn(config: ScenarioConfig):
        """Import the train_fn from a scenario's train.py module."""
        root = scenarios_root()
        scenario_dir = root / config.name
        train_py = scenario_dir / "train.py"
        if not train_py.exists():
            raise FileNotFoundError(
                f"No train.py found for scenario {config.name!r} at {train_py}"
            )
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"scenario_{config.name}", str(train_py),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "train_fn", None)
        if fn is None:
            raise AttributeError(
                f"Scenario {config.name!r} train.py must define a train_fn() function"
            )
        return fn


def run_all(
    *,
    pack: Optional[str] = None,
    step_delay: float = 0.0,
    verbose: bool = False,
) -> List[ScenarioResult]:
    """Run all (or pack-filtered) scenarios headless. Returns list of results."""
    from . import discover
    discover()
    from . import SCENARIO_REGISTRY

    runner = ScenarioRunner(step_delay=step_delay, verbose=verbose)
    results = []
    for name, config in sorted(SCENARIO_REGISTRY.items()):
        if pack and config.pack != pack:
            continue
        if verbose:
            print(f"  Running {name}...", end=" ", flush=True)
        result = runner.run(config)
        results.append(result)
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} (rules: {result.rules_fired})")
    return results
