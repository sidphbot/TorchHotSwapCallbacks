"""Continuation launcher — resume training from checkpoint with mutations."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    BranchResult,
    BranchSpec,
    BranchStatus,
    ContinuationConfig,
    MutationOp,
    ResumeMode,
    RoutineResult,
)


class ContinuationLauncher:
    """Execute continuation branches by resuming from checkpoints."""

    def __init__(self, config: ContinuationConfig, output_dir: Optional[str] = None):
        self._config = config
        self._output_dir = output_dir or config.output_dir or os.path.join(
            config.base_run_dir, "continuation"
        )
        os.makedirs(self._output_dir, exist_ok=True)

    def run_branch(self, branch: BranchSpec) -> BranchResult:
        """Execute a single continuation branch."""
        # Find anchor
        anchor = None
        for a in self._config.anchors:
            if a.anchor_id == branch.anchor_id:
                anchor = a
                break
        if anchor is None:
            return BranchResult(
                branch_id=branch.branch_id,
                anchor_id=branch.anchor_id,
                recipe_name=branch.recipe.name,
                resume_mode=branch.resume_mode.value,
                status=BranchStatus.RESUME_FAILED,
            )

        # Create branch run directory
        branch_dir = os.path.join(self._output_dir, "branches", branch.branch_id)
        os.makedirs(branch_dir, exist_ok=True)

        # Bootstrap JSONL files
        for fname in [
            "hotcb.commands.jsonl",
            "hotcb.applied.jsonl",
            "hotcb.metrics.jsonl",
        ]:
            open(os.path.join(branch_dir, fname), "w").close()
        with open(os.path.join(branch_dir, "hotcb.freeze.json"), "w") as f:
            json.dump({"mode": "off"}, f)

        # No-recipe sentinel (continuation doesn't use base demo recipes)
        open(os.path.join(branch_dir, "hotcb.eval.no_recipe"), "w").close()

        # Write mutation recipe as initial commands
        self._write_mutation_commands(branch_dir, branch, anchor)

        # Save branch config for traceability
        self._save_branch_config(branch_dir, branch, anchor)

        # Get training function
        train_fn = _get_continuation_train_fn(self._config.task)
        if train_fn is None:
            return BranchResult(
                branch_id=branch.branch_id,
                anchor_id=branch.anchor_id,
                recipe_name=branch.recipe.name,
                resume_mode=branch.resume_mode.value,
                status=BranchStatus.RESUME_FAILED,
            )

        # Set up autopilot if enabled
        autopilot_engine = None
        autopilot_actions: List[Dict] = []
        if self._config.autopilot != "off":
            autopilot_engine = _create_autopilot(self._config, branch_dir)

        # Run training
        stop_event = threading.Event()
        t0 = time.monotonic()

        sidecars = []
        if autopilot_engine is not None:
            from hotcb.eval.harness import _AutopilotSidecar
            ap_sidecar = _AutopilotSidecar(branch_dir, autopilot_engine)
            ap_sidecar.start()
            sidecars.append(ap_sidecar)

        try:
            train_fn(
                branch_dir,
                max_steps=branch.extra_steps,
                step_delay=0.0,
                _stop_event=stop_event,
                checkpoint_path=anchor.checkpoint_path,
                resume_mode=branch.resume_mode.value,
            )
        except Exception as exc:
            for sc in sidecars:
                sc.stop()
            return BranchResult(
                branch_id=branch.branch_id,
                anchor_id=branch.anchor_id,
                recipe_name=branch.recipe.name,
                resume_mode=branch.resume_mode.value,
                status=BranchStatus.RESUME_FAILED,
                run_dir=branch_dir,
                guardrail_violations=[str(exc)],
            )

        for sc in sidecars:
            sc.stop()

        elapsed = time.monotonic() - t0

        # Collect autopilot actions
        if autopilot_engine is not None and hasattr(sidecars[0], 'actions'):
            autopilot_actions = sidecars[0].actions

        # Collect results
        metrics_history = _read_jsonl(os.path.join(branch_dir, "hotcb.metrics.jsonl"))
        applied = _read_jsonl(os.path.join(branch_dir, "hotcb.applied.jsonl"))

        final_metrics = {}
        best_metrics = {}
        if metrics_history:
            final_metrics = metrics_history[-1].get("metrics", metrics_history[-1])
            best_metrics = self._find_best_metrics(
                metrics_history,
                self._config.objective.primary_metric,
                self._config.objective.mode,
            )

        # Compute delta vs base
        base_val = self._config.base_best_metrics.get(
            self._config.objective.primary_metric,
            self._config.base_final_metrics.get(self._config.objective.primary_metric),
        )
        branch_val = best_metrics.get(self._config.objective.primary_metric)

        primary_delta = 0.0
        primary_delta_pct = 0.0
        if base_val is not None and branch_val is not None:
            primary_delta = branch_val - base_val
            if base_val != 0:
                primary_delta_pct = primary_delta / abs(base_val) * 100

        # Determine status
        status = self._determine_status(
            final_metrics, best_metrics, metrics_history,
            primary_delta, primary_delta_pct,
        )

        return BranchResult(
            branch_id=branch.branch_id,
            anchor_id=branch.anchor_id,
            recipe_name=branch.recipe.name,
            resume_mode=branch.resume_mode.value,
            status=status,
            final_metrics=final_metrics,
            best_metrics=best_metrics,
            metric_history=metrics_history,
            primary_delta=round(primary_delta, 6),
            primary_delta_pct=round(primary_delta_pct, 2),
            total_steps=len(metrics_history),
            elapsed_seconds=round(elapsed, 2),
            run_dir=branch_dir,
            applied_mutations=applied,
            autopilot_actions=autopilot_actions,
        )

    def run_all(self, branches: List[BranchSpec]) -> RoutineResult:
        """Execute all branches sequentially and produce a routine result."""
        t0 = time.monotonic()
        results: List[BranchResult] = []

        for branch in branches:
            result = self.run_branch(branch)
            results.append(result)

        elapsed = time.monotonic() - t0

        # Find winner
        winner = self._select_winner(results)
        recommendation = "keep_base_run"
        if winner is not None:
            recommendation = f"promote_{winner.branch_id}"

        return RoutineResult(
            config=self._config,
            branches=results,
            winner=winner,
            recommendation=recommendation,
            elapsed_seconds=round(elapsed, 2),
        )

    def _write_mutation_commands(
        self, branch_dir: str, branch: BranchSpec, anchor,
    ) -> None:
        """Write mutation recipe ops as hotcb commands."""
        commands_path = os.path.join(branch_dir, "hotcb.commands.jsonl")

        # Read base metrics to resolve relative values
        base_metrics = {**self._config.base_final_metrics, **anchor.metrics_at_anchor}

        with open(commands_path, "a") as f:
            for op in branch.recipe.ops:
                cmd = _mutation_op_to_command(op, base_metrics)
                if cmd:
                    delay = getattr(op, "delay_steps", 0) or 0
                    if delay > 0:
                        cmd["min_step"] = anchor.step + delay
                    f.write(json.dumps(cmd) + "\n")

    def _save_branch_config(self, branch_dir: str, branch: BranchSpec, anchor) -> None:
        """Save branch configuration for traceability."""
        config = {
            "branch_id": branch.branch_id,
            "anchor_id": anchor.anchor_id,
            "anchor_step": anchor.step,
            "checkpoint_path": anchor.checkpoint_path,
            "recipe_name": branch.recipe.name,
            "resume_mode": branch.resume_mode.value,
            "extra_steps": branch.extra_steps,
            "ops": [{"target": o.target, "value": o.value, "relative": o.relative}
                    for o in branch.recipe.ops],
        }
        with open(os.path.join(branch_dir, "branch_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _find_best_metrics(
        self,
        history: List[Dict],
        metric: str,
        mode: str,
    ) -> Dict[str, float]:
        """Find the metrics snapshot where primary metric was best."""
        best_val = None
        best_metrics = {}
        for entry in history:
            m = entry.get("metrics", entry)
            val = m.get(metric)
            if val is None:
                continue
            if best_val is None:
                best_val = val
                best_metrics = dict(m)
            elif (mode == "min" and val < best_val) or (mode == "max" and val > best_val):
                best_val = val
                best_metrics = dict(m)
        return best_metrics

    def _determine_status(
        self,
        final_metrics: Dict,
        best_metrics: Dict,
        history: List[Dict],
        primary_delta: float,
        primary_delta_pct: float,
    ) -> BranchStatus:
        """Determine branch outcome status."""
        obj = self._config.objective
        guardrails = self._config.guardrails

        # Check for NaN/Inf
        if guardrails.forbid_nan:
            for entry in history:
                m = entry.get("metrics", entry)
                for v in m.values():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return BranchStatus.UNSTABLE

        # Check guardrail secondary metrics
        for metric_name, spec in guardrails.secondary_metrics.items():
            base_val = self._config.base_best_metrics.get(
                metric_name, self._config.base_final_metrics.get(metric_name)
            )
            branch_val = best_metrics.get(metric_name)
            if base_val is not None and branch_val is not None:
                max_reg = spec.get("max_regression", float("inf"))
                sec_mode = spec.get("mode", "min")
                if sec_mode == "min":
                    regression = branch_val - base_val
                else:
                    regression = base_val - branch_val
                if regression > max_reg:
                    return BranchStatus.REJECTED_GUARDRAIL

        # Check primary metric improvement
        min_delta = obj.min_improvement_delta
        if obj.mode == "max":
            if primary_delta >= min_delta:
                return BranchStatus.WON
        else:
            if -primary_delta >= min_delta:  # for min mode, improvement is negative delta
                return BranchStatus.WON

        if abs(primary_delta_pct) < 1.0:
            return BranchStatus.INCONCLUSIVE

        return BranchStatus.LOST

    def _select_winner(self, results: List[BranchResult]) -> Optional[BranchResult]:
        """Select the best winning branch."""
        winners = [r for r in results if r.status == BranchStatus.WON]
        if not winners:
            return None

        mode = self._config.objective.mode
        if mode == "max":
            return max(winners, key=lambda r: r.primary_delta)
        else:
            return min(winners, key=lambda r: r.primary_delta)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _mutation_op_to_command(op: MutationOp, base_metrics: Dict[str, float]) -> Optional[Dict]:
    """Convert a MutationOp to a hotcb command dict."""
    target = op.target

    # Map target names to hotcb command format
    if target in ("lr", "weight_decay", "betas"):
        module = "opt"
        param_key = target
    elif target in ("swa_enabled", "ema_enabled", "grad_clip"):
        module = "custom"
        param_key = target
    elif target.startswith("loss.") or target.startswith("losses."):
        module = "loss"
        param_key = target.split(".", 1)[1]
    else:
        module = "custom"
        param_key = target

    value = op.value
    if op.relative and isinstance(value, (int, float)):
        # Resolve relative value against base
        base_val = base_metrics.get(param_key, base_metrics.get(target))
        if base_val is not None:
            value = base_val * value
        else:
            return None  # can't resolve relative without base

    return {
        "module": module,
        "op": "set_params",
        "params": {param_key: value},
        "source": "continuation_recipe",
    }


def _get_continuation_train_fn(task: str):
    """Get the checkpoint-aware training function for a task."""
    if task == "mnist":
        from hotcb.eval.tasks import mnist_continuation_training
        return mnist_continuation_training
    elif task == "cifar10":
        from hotcb.eval.tasks import cifar10_continuation_training
        return cifar10_continuation_training
    elif task == "coco_mobilenet":
        try:
            from hotcb.eval.tasks_coco import coco_mobilenet_continuation_training
            return coco_mobilenet_continuation_training
        except ImportError:
            return None
    elif task == "imagenet_mobilenet":
        try:
            from hotcb.eval.tasks_imagenet import imagenet_mobilenet_continuation_training
            return imagenet_mobilenet_continuation_training
        except ImportError:
            return None
    elif task == "imagenet_mobilenetv2_paper":
        try:
            from hotcb.eval.tasks_imagenet import imagenet_mobilenetv2_paper_continuation_training
            return imagenet_mobilenetv2_paper_continuation_training
        except ImportError:
            return None
    elif task == "coco_detection":
        try:
            from hotcb.eval.tasks_coco_detection import coco_ssdlite_mobilenetv2_continuation_training
            return coco_ssdlite_mobilenetv2_continuation_training
        except ImportError:
            return None
    elif task == "clip_coco":
        try:
            from hotcb.eval.tasks_vlm import clip_coco_continuation_training
            return clip_coco_continuation_training
        except ImportError:
            return None
    return None


def _create_autopilot(config: ContinuationConfig, run_dir: str):
    """Create an AutopilotEngine for a branch."""
    try:
        from hotcb.server.autopilot import AutopilotEngine
        engine = AutopilotEngine(run_dir=run_dir, mode=config.autopilot)
        for pack in config.policy_packs:
            try:
                engine.load_pack(pack)
            except Exception:
                pass
        return engine
    except Exception:
        return None


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
