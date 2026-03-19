"""Integration tests for hotcb — full-stack exercises."""
from __future__ import annotations

import json
import math
import os

import pytest


# ---------------------------------------------------------------------------
# 1. Demo produces metrics
# ---------------------------------------------------------------------------


def test_demo_10_steps_produces_metrics(tmp_path):
    """Run the demo training loop for 10 steps and verify metrics file."""
    run_dir = str(tmp_path)

    # Bootstrap required files
    for fname in ("hotcb.commands.jsonl", "hotcb.applied.jsonl",
                  "hotcb.metrics.jsonl", "hotcb.recipe.jsonl"):
        open(os.path.join(run_dir, fname), "w").close()
    with open(os.path.join(run_dir, "hotcb.freeze.json"), "w") as f:
        json.dump({"mode": "off"}, f)

    from hotcb.demo import _demo_training
    import threading

    stop = threading.Event()
    _demo_training(run_dir, max_steps=10, step_delay=0.0, _stop_event=stop)

    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    assert os.path.exists(metrics_path), "hotcb.metrics.jsonl should be created"

    lines = [l for l in open(metrics_path).readlines() if l.strip()]
    assert len(lines) >= 1, "Should have at least 1 metric line"

    # Check that train_loss appears in at least one record
    found_train_loss = False
    for line in lines:
        rec = json.loads(line)
        metrics = rec.get("metrics", rec)
        if "train_loss" in metrics or rec.get("name") == "train_loss":
            found_train_loss = True
            break
    assert found_train_loss, "train_loss should appear in metrics"


# ---------------------------------------------------------------------------
# 2. Demo with policy pack
# ---------------------------------------------------------------------------


def test_demo_with_policy_pack(tmp_path):
    """Load stability_basics pack, feed NaN metrics, verify rule fires."""
    run_dir = str(tmp_path)
    os.makedirs(run_dir, exist_ok=True)

    from hotcb.server.autopilot import AutopilotEngine

    engine = AutopilotEngine(run_dir=run_dir, mode="suggest")
    count = engine.load_pack("stability_basics")
    assert count > 0, "stability_basics should have rules"

    # Feed nan_detected flag — should trigger nan_guard rule
    actions = engine.evaluate(step=100, metrics={
        "train_loss": 1.0,
        "val_loss": 1.0,
        "grad_norm": 1.0,
        "nan_detected": 1,
    })

    # The nan_guard rule uses custom condition with nan_detected > 0
    nan_actions = [a for a in actions if "nan" in a.rule_id.lower() or "nan" in a.condition_met.lower()]
    assert len(nan_actions) > 0, "NaN guard rule should fire on nan_detected flag"
    assert nan_actions[0].status == "proposed"  # suggest mode


# ---------------------------------------------------------------------------
# 3. Health state computed
# ---------------------------------------------------------------------------


def test_health_state_computed_during_demo():
    """Verify compute_health_state produces labels for decreasing loss."""
    from hotcb.health import compute_health_state

    # Build synthetic metric history: steadily decreasing loss
    metric_history = {
        "train_loss": [
            {"step": i, "value": 2.0 - i * 0.05 + (0.001 * (i % 3))}
            for i in range(30)
        ],
    }

    state = compute_health_state(metric_history, window=100)
    assert state is not None
    assert isinstance(state.labels, list)
    # Steadily decreasing loss should be labeled stable-improving
    assert "stable-improving" in state.labels, (
        f"Expected 'stable-improving' in labels, got: {state.labels}"
    )


# ---------------------------------------------------------------------------
# 4. Rollback during demo
# ---------------------------------------------------------------------------


def test_rollback_during_demo(tmp_path):
    """Create MutableState, apply change, rollback, verify restored."""
    from hotcb.actuators import optimizer_actuators, mutable_state

    run_dir = str(tmp_path)

    # Use _OptProxy-style dict
    class _Opt:
        def __init__(self):
            self.param_groups = [{"lr": 1e-3, "weight_decay": 1e-4}]

    opt = _Opt()
    ms = mutable_state(optimizer_actuators(opt))
    env = {"step": 1}

    ms.initialize(env)

    # Check initial lr
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    # Apply a change
    result = ms.apply("lr", 5e-4, env, step=1)
    assert result.success
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)

    # Snapshot stack should exist
    assert len(ms._snapshot_stack) > 0

    # Rollback
    restored = ms.rollback(n=1, env=env)
    assert restored is not None
    assert "lr" in restored
    assert restored["lr"].success

    # Value should be restored
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# 5. Full autopilot loop
# ---------------------------------------------------------------------------


def test_full_autopilot_loop(tmp_path):
    """Autopilot in suggest mode, feed plateau metrics, verify proposal."""
    run_dir = str(tmp_path)
    os.makedirs(run_dir, exist_ok=True)

    from hotcb.server.autopilot import AutopilotEngine, AutopilotRule

    engine = AutopilotEngine.with_default_guidelines(
        run_dir=run_dir, mode="suggest"
    )

    # Also add a plateau rule targeting train_loss
    engine.add_rule(AutopilotRule(
        rule_id="test_plateau",
        condition="plateau",
        metric_name="train_loss",
        params={"window": 10, "epsilon": 0.001, "cooldown": 1},
        action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
        confidence="high",
        description="Reduce LR on plateau",
    ))

    # Feed flat loss for 25 steps
    actions_all = []
    for step in range(1, 26):
        actions = engine.evaluate(step=step, metrics={
            "train_loss": 1.5,
            "val_loss": 1.6,
            "grad_norm": 0.5,
        })
        actions_all.extend(actions)

    # After enough flat steps, the plateau rule should have fired
    plateau_actions = [a for a in actions_all if a.rule_id == "test_plateau"]
    assert len(plateau_actions) > 0, "Plateau rule should fire after flat metrics"
    assert plateau_actions[0].status == "proposed"
