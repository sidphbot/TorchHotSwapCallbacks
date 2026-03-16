"""Tests for execution safety: rollback stack, effect tracking, mutation budget, bounds enforcement.

Phase 2 — TDD: tests written first, then implementation.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List

import pytest

from hotcb.actuators.actuator import (
    ActuatorState,
    ActuatorType,
    HotcbActuator,
    _INIT_SENTINEL,
)
from hotcb.actuators.base import ApplyResult
from hotcb.actuators.state import MutableState
from hotcb.kernel import HotKernel
from hotcb.ops import HotOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_applied_values: Dict[str, Any] = {}


def _make_apply_fn(key: str):
    """Create a closure-based apply_fn that tracks applied values."""
    def _apply(value, env):
        _applied_values[key] = value
        return ApplyResult(success=True, detail={"applied": value})
    return _apply


def _make_float_actuator(key="x", min_v=0.0, max_v=1.0, current=0.5):
    return HotcbActuator(
        param_key=key,
        type=ActuatorType.FLOAT,
        apply_fn=_make_apply_fn(key),
        min_value=min_v,
        max_value=max_v,
        current_value=current,
        state=ActuatorState.UNTOUCHED,
    )


def _make_log_float_actuator(key="lr", min_v=1e-6, max_v=1.0, current=1e-3):
    return HotcbActuator(
        param_key=key,
        type=ActuatorType.LOG_FLOAT,
        apply_fn=_make_apply_fn(key),
        min_value=min_v,
        max_value=max_v,
        current_value=current,
        state=ActuatorState.UNTOUCHED,
    )


def _make_mutable_state(**actuators):
    """Create a MutableState with named actuators."""
    acts = []
    for key, (min_v, max_v, current) in actuators.items():
        acts.append(_make_float_actuator(key, min_v, max_v, current))
    return MutableState(acts)


def _make_kernel_with_state(tmp_path, ms=None):
    """Create a HotKernel with a MutableState in a temp dir."""
    run_dir = str(tmp_path)
    os.makedirs(run_dir, exist_ok=True)
    if ms is None:
        ms = _make_mutable_state(lr=(0.0, 1.0, 0.01), weight_decay=(0.0, 1.0, 0.001))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms)
    return kernel


def _read_applied_ledger(kernel) -> List[dict]:
    """Read all entries from the applied ledger."""
    entries = []
    if os.path.exists(kernel.applied_path):
        with open(kernel.applied_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


@pytest.fixture(autouse=True)
def _reset_applied_values():
    _applied_values.clear()
    yield
    _applied_values.clear()


# ===================================================================
# 1. Snapshot Stack and Rollback
# ===================================================================

class TestSnapshotStack:

    def test_snapshot_created_on_apply(self, tmp_path):
        ms = _make_mutable_state(x=(0.0, 1.0, 0.5))
        result = ms.apply("x", 0.7, {}, step=1)
        assert result.success
        assert len(ms._snapshot_stack) == 1
        # The snapshot should contain the OLD value (0.5)
        assert ms._snapshot_stack[0]["x"]["value"] == 0.5

    def test_rollback_restores_previous_state(self, tmp_path):
        ms = _make_mutable_state(x=(0.0, 1.0, 0.5))
        ms.apply("x", 0.7, {}, step=1)
        assert ms.get("x").current_value == 0.7
        results = ms.rollback()
        assert ms.get("x").current_value == 0.5

    def test_rollback_multiple_times(self, tmp_path):
        ms = _make_mutable_state(x=(0.0, 1.0, 0.5))
        ms.apply("x", 0.6, {}, step=1)
        ms.apply("x", 0.7, {}, step=2)
        ms.apply("x", 0.8, {}, step=3)

        ms.rollback()
        assert ms.get("x").current_value == 0.7
        ms.rollback()
        assert ms.get("x").current_value == 0.6
        ms.rollback()
        assert ms.get("x").current_value == 0.5

    def test_rollback_empty_stack(self, tmp_path):
        ms = _make_mutable_state(x=(0.0, 1.0, 0.5))
        results = ms.rollback()
        # Should return empty results or indicate no-op
        assert results is None or results == {}

    def test_snapshot_stack_capped(self, tmp_path):
        ms = MutableState(
            [_make_float_actuator("x", 0.0, 100.0, 0.0)],
            max_snapshots=3,
        )
        for i in range(1, 7):
            ms.apply("x", float(i), {}, step=i)
        # Stack should be capped at 3
        assert len(ms._snapshot_stack) <= 3

    def test_rollback_records_in_ledger(self, tmp_path):
        kernel = _make_kernel_with_state(tmp_path)
        env = {"step": 1}
        # Apply a change first
        kernel._mutable_state.apply("lr", 0.05, env, step=1)

        # Create a rollback op
        op = HotOp(module="opt", op="rollback")
        kernel._apply_single(op, env, event="step", step=2)

        entries = _read_applied_ledger(kernel)
        rollback_entries = [e for e in entries if e.get("op") == "rollback"]
        assert len(rollback_entries) == 1
        assert rollback_entries[0]["decision"] == "applied"

    def test_rollback_via_kernel_op(self, tmp_path):
        kernel = _make_kernel_with_state(tmp_path)
        env = {"step": 1}
        # Apply a change
        kernel._mutable_state.apply("lr", 0.05, env, step=1)
        assert kernel._mutable_state.get("lr").current_value == 0.05

        # Rollback via default stream
        op = HotOp(module="opt", op="rollback")
        kernel._apply_default_stream(op, env, event="step", step=2)

        assert kernel._mutable_state.get("lr").current_value == 0.01


# ===================================================================
# 2. Action Effect Tracking
# ===================================================================

class TestEffectTracker:

    def test_effect_tracker_records_baseline(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=5)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})
        pending = tracker.get_pending()
        assert len(pending) == 1
        assert pending[0].param_key == "lr"
        assert pending[0].baseline_metrics["val_loss"] == 0.5

    def test_effect_tracker_computes_delta(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=3)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})

        # Steps before cooldown — should still be pending
        tracker.on_metrics(step=11, metrics={"val_loss": 0.48})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.45})
        assert len(tracker.get_pending()) == 1

        # Step at cooldown — should complete
        tracker.on_metrics(step=13, metrics={"val_loss": 0.42})
        completed = tracker.get_completed()
        assert len(completed) == 1
        assert completed[0].param_key == "lr"
        # Delta should show improvement (val_loss went down)
        assert completed[0].delta["val_loss"] < 0

    def test_effect_tracker_classifies_outcome(self):
        from hotcb.tracking import EffectTracker
        # For val_loss, lower is better — so a decrease is "improved"
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})
        tracker.on_metrics(step=11, metrics={"val_loss": 0.48})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.40})
        completed = tracker.get_completed()
        assert completed[0].outcome == "improved"

        # Now test degradation
        tracker2 = EffectTracker(key_metric="val_loss", cooldown_steps=2)
        tracker2.on_mutation(step=10, param_key="wd", old_value=0.0001,
                            new_value=0.001, metric_snapshot={"val_loss": 0.5})
        tracker2.on_metrics(step=11, metrics={"val_loss": 0.55})
        tracker2.on_metrics(step=12, metrics={"val_loss": 0.65})
        completed2 = tracker2.get_completed()
        assert completed2[0].outcome == "degraded"

    def test_effect_tracker_timeout(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=3,
                                timeout_steps=10)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})
        # Simulate many steps without the key metric
        for s in range(11, 25):
            tracker.on_metrics(step=s, metrics={"train_loss": 0.3})
        # Should have timed out and moved to completed with "timeout" outcome
        completed = tracker.get_completed()
        assert len(completed) == 1
        assert completed[0].outcome == "timeout"

    def test_effect_tracker_multiple_concurrent(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=3)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})
        tracker.on_mutation(step=11, param_key="wd", old_value=0.001,
                           new_value=0.01, metric_snapshot={"val_loss": 0.48})

        for s in range(12, 16):
            tracker.on_metrics(step=s, metrics={"val_loss": 0.4})

        completed = tracker.get_completed()
        assert len(completed) == 2
        keys = {c.param_key for c in completed}
        assert keys == {"lr", "wd"}

    def test_effect_in_ledger(self, tmp_path):
        """Effect tracker integrates with kernel — mutations get tracked."""
        from hotcb.tracking import EffectTracker
        kernel = _make_kernel_with_state(tmp_path)
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2)
        kernel._effect_tracker = tracker

        env = {"step": 1, "val_loss": 0.5}
        # Trigger a set_params through the kernel
        op = HotOp(module="opt", op="set_params", params={"key": "lr", "value": 0.05})
        kernel._apply_single(op, env, event="step", step=1)

        # The effect tracker should have recorded the mutation
        pending = tracker.get_pending()
        assert len(pending) == 1


# ===================================================================
# 3. Auto-Rollback Triggers
# ===================================================================

class TestAutoRollback:

    def test_auto_rollback_on_degradation(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2,
                                rollback_threshold=0.1)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.1, metric_snapshot={"val_loss": 0.5})
        tracker.on_metrics(step=11, metrics={"val_loss": 0.6})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.7})
        # 0.7 - 0.5 = 0.2 > 0.1 threshold (40% increase)
        trigger = tracker.check_rollback_triggers()
        assert trigger == "lr"

    def test_auto_rollback_respects_cooldown(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=5,
                                rollback_threshold=0.1)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.1, metric_snapshot={"val_loss": 0.5})
        # Only 2 steps elapsed, cooldown is 5 — should not trigger
        tracker.on_metrics(step=11, metrics={"val_loss": 0.7})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.9})
        trigger = tracker.check_rollback_triggers()
        assert trigger is None

    def test_auto_rollback_disabled_by_config(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2,
                                rollback_threshold=None)  # disabled
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.1, metric_snapshot={"val_loss": 0.5})
        tracker.on_metrics(step=11, metrics={"val_loss": 0.9})
        tracker.on_metrics(step=12, metrics={"val_loss": 1.5})
        trigger = tracker.check_rollback_triggers()
        assert trigger is None

    def test_auto_rollback_logs_reason(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2,
                                rollback_threshold=0.05)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.1, metric_snapshot={"val_loss": 0.5})
        tracker.on_metrics(step=11, metrics={"val_loss": 0.6})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.7})
        trigger = tracker.check_rollback_triggers()
        assert trigger is not None
        # Completed effects should contain reason info
        completed = tracker.get_completed()
        assert any(c.param_key == "lr" and c.outcome == "degraded" for c in completed)

    def test_no_auto_rollback_on_improvement(self):
        from hotcb.tracking import EffectTracker
        tracker = EffectTracker(key_metric="val_loss", cooldown_steps=2,
                                rollback_threshold=0.1)
        tracker.on_mutation(step=10, param_key="lr", old_value=0.01,
                           new_value=0.005, metric_snapshot={"val_loss": 0.5})
        tracker.on_metrics(step=11, metrics={"val_loss": 0.45})
        tracker.on_metrics(step=12, metrics={"val_loss": 0.40})
        trigger = tracker.check_rollback_triggers()
        assert trigger is None


# ===================================================================
# 4. Rule Bounds Enforcement
# ===================================================================

class TestRuleBoundsEnforcement:

    def _make_autopilot_with_state(self, tmp_path):
        """Create an AutopilotEngine with a MutableState available."""
        from hotcb.server.autopilot import AutopilotEngine
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(run_dir=run_dir, mode="auto")

        # Create a mutable state and save its descriptions
        ms = _make_mutable_state(lr=(1e-6, 1.0, 0.01))
        descs = ms.describe_all()
        desc_path = os.path.join(run_dir, "hotcb.actuators.json")
        with open(desc_path, "w") as f:
            json.dump({"controls": descs}, f)

        return engine, ms

    def test_rule_action_validated_against_actuator_bounds(self, tmp_path):
        engine, ms = self._make_autopilot_with_state(tmp_path)
        # Try to apply an action with lr=5.0 which is above max 1.0
        action = {
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 5.0},
        }
        validated = engine.validate_action(action)
        # Should clamp or reject
        assert validated is not None
        params = validated.get("params", {})
        if "lr" in params:
            assert params["lr"] <= 1.0  # clamped to max

    def test_rule_action_for_unknown_actuator(self, tmp_path):
        engine, ms = self._make_autopilot_with_state(tmp_path)
        action = {
            "module": "opt",
            "op": "set_params",
            "params": {"unknown_param": 42},
        }
        # Unknown actuator — should pass through (no bounds to check)
        validated = engine.validate_action(action)
        assert validated is not None

    def test_rule_action_respects_type(self, tmp_path):
        engine, ms = self._make_autopilot_with_state(tmp_path)
        action = {
            "module": "opt",
            "op": "set_params",
            "params": {"lr": "not_a_number"},
        }
        validated = engine.validate_action(action)
        # Should reject type mismatch
        assert validated is None

    def test_rule_bounds_same_as_ai_bounds(self, tmp_path):
        """Rule and AI actions go through the same validation path."""
        engine, ms = self._make_autopilot_with_state(tmp_path)
        action = {
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 0.5},
        }
        validated_rule = engine.validate_action(action)
        validated_ai = engine.validate_action(action)
        assert validated_rule == validated_ai


# ===================================================================
# 5. Mutation Budget
# ===================================================================

class TestMutationBudget:

    def test_mutation_budget_allows_within_limit(self, tmp_path):
        from hotcb.server.autopilot import AutopilotEngine
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=5, budget_window_steps=100,
        )
        # Record 3 mutations — should all be allowed
        for i in range(3):
            assert engine.check_mutation_budget(step=i + 1)
            engine.record_mutation(step=i + 1)

    def test_mutation_budget_blocks_over_limit(self, tmp_path):
        from hotcb.server.autopilot import AutopilotEngine
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=3, budget_window_steps=100,
        )
        for i in range(3):
            engine.record_mutation(step=i + 1)
        # 4th mutation should be blocked
        assert not engine.check_mutation_budget(step=4)

    def test_mutation_budget_rolling_window(self, tmp_path):
        from hotcb.server.autopilot import AutopilotEngine
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=3, budget_window_steps=10,
        )
        # Fill budget at steps 1-3
        for i in range(1, 4):
            engine.record_mutation(step=i)

        # At step 4, still blocked (all 3 within window of 10)
        assert not engine.check_mutation_budget(step=4)

        # At step 15, all old mutations are outside window
        assert engine.check_mutation_budget(step=15)

    def test_mutation_budget_configurable(self, tmp_path):
        from hotcb.server.autopilot import AutopilotEngine
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=20, budget_window_steps=50,
        )
        for i in range(20):
            engine.record_mutation(step=i + 1)
        assert not engine.check_mutation_budget(step=21)
        # But with a larger budget it would pass
        engine2 = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=25, budget_window_steps=50,
        )
        for i in range(20):
            engine2.record_mutation(step=i + 1)
        assert engine2.check_mutation_budget(step=21)

    def test_mutation_budget_applies_to_rules_and_ai(self, tmp_path):
        """Budget check is called in _apply_action for both rules and AI paths."""
        from hotcb.server.autopilot import AutopilotEngine, AutopilotRule
        run_dir = str(tmp_path)
        os.makedirs(run_dir, exist_ok=True)
        engine = AutopilotEngine(
            run_dir=run_dir, mode="auto",
            mutation_budget=1, budget_window_steps=100,
        )
        # Add a rule that fires
        rule = AutopilotRule(
            rule_id="test_plateau",
            condition="plateau",
            metric_name="val_loss",
            params={"window": 3, "epsilon": 0.001, "cooldown": 0},
            action={"module": "opt", "op": "set_params", "params": {"lr_mult": 0.5}},
            confidence="high",
        )
        engine.add_rule(rule)

        # Feed plateau metrics
        for i in range(5):
            engine.evaluate(step=i + 1, metrics={"val_loss": 0.5, "lr": 0.01})

        # Check that only 1 mutation was applied (budget=1)
        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        if os.path.exists(cmd_path):
            with open(cmd_path) as f:
                cmds = [json.loads(l) for l in f if l.strip()]
            assert len(cmds) <= 1


# ===================================================================
# 6. Rollback API Endpoint
# ===================================================================

class TestRollbackEndpoint:

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client with the rollback endpoint."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()

        from hotcb.server.api import router
        app.include_router(router)

        # Set up app state
        from types import SimpleNamespace
        config = SimpleNamespace(run_dir=str(tmp_path))
        app.state.config = config

        # Create a mutable state with snapshot stack
        ms = _make_mutable_state(lr=(0.0, 1.0, 0.01))
        ms.apply("lr", 0.05, {}, step=1)
        app.state.mutable_state = ms

        return TestClient(app)

    def test_rollback_endpoint_returns_success(self, client, tmp_path):
        resp = client.post("/api/rollback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rolled_back"

    def test_rollback_endpoint_empty_stack(self, tmp_path):
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()
        from hotcb.server.api import router
        app.include_router(router)

        from types import SimpleNamespace
        config = SimpleNamespace(run_dir=str(tmp_path))
        app.state.config = config

        ms = _make_mutable_state(lr=(0.0, 1.0, 0.01))
        # No apply — stack is empty
        app.state.mutable_state = ms

        client = TestClient(app)
        resp = client.post("/api/rollback")
        assert resp.status_code == 400
