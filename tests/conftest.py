"""Playwright E2E test fixtures for the hotcb dashboard.

Starts a real hotcb server against pre-populated eval data
and provides a Playwright page fixture for browser testing.
"""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

# Default test port — chosen to avoid conflicts with dev servers
_TEST_PORT = 18429
_TEST_URL = f"http://localhost:{_TEST_PORT}"


def _write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_test_eval_dir(base: str) -> str:
    """Create a minimal eval output dir with 2 conditions and metrics."""
    os.makedirs(base, exist_ok=True)

    conditions = ["cond_baseline", "cond_high_lr"]

    for cond in conditions:
        cond_dir = os.path.join(base, cond)
        os.makedirs(cond_dir, exist_ok=True)

        # Metrics: 50 steps of train_loss + grad_norm
        metrics = []
        for step in range(50):
            loss = 2.0 - step * 0.03 if cond == "cond_baseline" else 3.0 - step * 0.02
            grad = 5.0 + step * 0.1 if cond == "cond_baseline" else 15.0 - step * 0.1
            metrics.append({
                "step": step,
                "metrics": {
                    "train_loss": round(loss, 4),
                    "grad_norm": round(grad, 4),
                    "lr": 0.001 if cond == "cond_baseline" else 0.05,
                },
            })
        _write_jsonl(os.path.join(cond_dir, "hotcb.metrics.jsonl"), metrics)

        # Applied: a couple of mutations for the high_lr condition
        applied = []
        if cond == "cond_high_lr":
            applied.append({
                "step": 10,
                "module": "opt",
                "op": "set_params",
                "params": {"lr": 0.01},
                "source": "autopilot",
                "rule_id": "grad_spike_clip",
            })
        _write_jsonl(os.path.join(cond_dir, "hotcb.applied.jsonl"), applied)
        _write_jsonl(os.path.join(cond_dir, "hotcb.commands.jsonl"), [])

        # Freeze state
        with open(os.path.join(cond_dir, "hotcb.freeze.json"), "w") as f:
            json.dump({"mode": "off"}, f)

    # Eval status (completed)
    eval_status = {
        "total": 2,
        "completed": 2,
        "completed_names": conditions,
        "running": None,
        "done": True,
        "focus_run": "cond_high_lr",
        "conditions": conditions,
    }
    with open(os.path.join(base, "hotcb.eval.status.json"), "w") as f:
        json.dump(eval_status, f)

    # Research graph snapshot
    research = {
        "nodes": {
            "hyp1": {
                "node_id": "hyp1",
                "node_type": "hypothesis",
                "condition": "baseline config",
                "intervention": {},
                "expected_outcome": "converges normally",
                "status": "confirmed",
                "confidence": 1.0,
                "nn_confidence": 0.0,
                "evidence_count": 1,
                "test_count": 0,
                "step_created": 0,
                "stream_id": "eval_test",
                "source": "manual",
                "mutation_ref": None,
                "diluted": False,
                "created_at": 0,
            },
            "hyp2": {
                "node_id": "hyp2",
                "node_type": "hypothesis",
                "condition": "high lr triggers grad_spike_clip",
                "intervention": {"lr": 0.05},
                "expected_outcome": "autopilot recovers",
                "status": "confirmed",
                "confidence": 1.0,
                "nn_confidence": 0.0,
                "evidence_count": 1,
                "test_count": 0,
                "step_created": 0,
                "stream_id": "eval_test",
                "source": "manual",
                "mutation_ref": None,
                "diluted": False,
                "created_at": 0,
            },
            "ev1": {
                "node_id": "ev1",
                "node_type": "evidence",
                "hypothesis_id": "hyp1",
                "effect_id": None,
                "outcome": "improved",
                "delta": {"train_loss": -1.5},
                "context": {},
                "source": "eval",
                "created_at": 0,
            },
            "ev2": {
                "node_id": "ev2",
                "node_type": "evidence",
                "hypothesis_id": "hyp2",
                "effect_id": None,
                "outcome": "improved",
                "delta": {"train_loss": -2.0},
                "context": {},
                "source": "eval",
                "created_at": 0,
            },
        },
        "edges": {
            "e1": {
                "edge_id": "e1",
                "source": "ev1",
                "target": "hyp1",
                "edge_type": "supports",
            },
            "e2": {
                "edge_id": "e2",
                "source": "ev2",
                "target": "hyp2",
                "edge_type": "supports",
            },
        },
        "streams": {
            "eval_test": {
                "stream_id": "eval_test",
                "name": "eval_test",
                "description": "Test eval stream",
                "hypothesis_ids": ["hyp1", "hyp2"],
                "observation_ids": [],
                "status": "active",
                "conclusion": None,
            },
        },
    }
    with open(os.path.join(base, "hotcb.research.json"), "w") as f:
        json.dump(research, f)
    # Empty event log
    with open(os.path.join(base, "hotcb.research.jsonl"), "w") as f:
        pass

    return base


@pytest.fixture(scope="session")
def eval_dir(tmp_path_factory):
    """Create a temp eval dir with test data (session-scoped for speed)."""
    base = str(tmp_path_factory.mktemp("eval_e2e"))
    return _make_test_eval_dir(base)


@pytest.fixture(scope="session")
def server_url(eval_dir):
    """Start hotcb eval serve and yield the base URL. Killed after session."""
    proc = subprocess.Popen(
        [
            "python3", "-m", "hotcb", "eval", "serve",
            "--output-dir", eval_dir,
            "--port", str(_TEST_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready (poll /api/status)
    import urllib.request
    for attempt in range(40):
        try:
            req = urllib.request.urlopen(f"{_TEST_URL}/api/status", timeout=1)
            if req.status == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    else:
        proc.kill()
        out = proc.stdout.read().decode() if proc.stdout else ""
        raise RuntimeError(f"Server failed to start within 10s.\n{out}")

    yield _TEST_URL

    # Teardown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser_context_args():
    """Playwright browser context args — disable animations for faster tests."""
    return {
        "viewport": {"width": 1400, "height": 900},
        "ignore_https_errors": True,
    }
