"""Tests for the research-continuation bridge."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.research.engine import ResearchEngine
from hotcb.research.types import HypothesisNode


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory."""
    d = str(tmp_path / "test_run")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def engine(run_dir):
    """Create a fresh ResearchEngine."""
    return ResearchEngine(run_dir)


# ─── load_continuation_conditions ────────────────────────────────────

def test_load_mnist_conditions(engine):
    hyps = engine.load_continuation_conditions("mnist")
    assert len(hyps) > 0
    for h in hyps:
        assert isinstance(h, HypothesisNode)
        assert h.status == "proposed"

def test_load_cifar10_conditions(engine):
    hyps = engine.load_continuation_conditions("cifar10")
    assert len(hyps) > 0
    for h in hyps:
        assert h.status == "proposed"

def test_load_unknown_task_returns_empty(engine):
    hyps = engine.load_continuation_conditions("nonexistent")
    assert hyps == []

def test_load_conditions_dedup(engine):
    """Loading twice should not create duplicates."""
    hyps1 = engine.load_continuation_conditions("mnist")
    hyps2 = engine.load_continuation_conditions("mnist")
    assert len(hyps2) == 0  # all already exist

def test_load_conditions_have_intervention(engine):
    hyps = engine.load_continuation_conditions("mnist")
    for h in hyps:
        assert "params" in h.intervention
        assert h.intervention.get("module") == "continuation"
        assert h.intervention.get("op") == "branch"

def test_load_conditions_have_source(engine):
    hyps = engine.load_continuation_conditions("mnist")
    for h in hyps:
        assert h.source.startswith("continuation:")

def test_load_conditions_count_mnist(engine):
    """Should load all MNIST continuation conditions (original + new)."""
    from hotcb.eval.conditions import (
        MNIST_CONTINUATION_CONDITIONS,
        MNIST_NEW_CONTINUATION_CONDITIONS,
    )
    expected = len(MNIST_CONTINUATION_CONDITIONS) + len(MNIST_NEW_CONTINUATION_CONDITIONS)
    hyps = engine.load_continuation_conditions("mnist")
    assert len(hyps) == expected

def test_load_conditions_count_cifar10(engine):
    from hotcb.eval.conditions import (
        CIFAR10_CONTINUATION_CONDITIONS,
        CIFAR10_NEW_CONTINUATION_CONDITIONS,
    )
    expected = len(CIFAR10_CONTINUATION_CONDITIONS) + len(CIFAR10_NEW_CONTINUATION_CONDITIONS)
    hyps = engine.load_continuation_conditions("cifar10")
    assert len(hyps) == expected

def test_load_conditions_with_base_run_dir(engine):
    """base_run_dir is accepted but doesn't affect hypothesis creation."""
    hyps = engine.load_continuation_conditions("mnist", base_run_dir="/tmp/fake")
    assert len(hyps) > 0


# ─── launch_condition_tests (unit-level, no actual training) ─────────

def test_launch_tests_not_found(engine):
    results = engine.launch_condition_tests(["nonexistent_id"], "/tmp/fake")
    assert len(results) == 1
    assert results[0]["error"] == "not found"

def test_launch_tests_wrong_status(engine):
    """Cannot launch test on a non-proposed hypothesis."""
    hyp = engine.hypothesize(
        condition="test",
        intervention={"module": "opt", "params": {"lr": 0.001}},
        expected_outcome="test",
    )
    # Transition proposed → testing → confirmed (state machine requires testing first)
    engine.graph.transition_hypothesis(hyp.node_id, "testing")
    engine.graph.transition_hypothesis(hyp.node_id, "confirmed")
    assert hyp.status == "confirmed"
    results = engine.launch_condition_tests([hyp.node_id], "/tmp/fake")
    assert len(results) == 1
    assert "cannot test" in results[0]["error"]

def test_launch_tests_no_checkpoints(engine, tmp_path):
    """Launch with a valid hypothesis but no checkpoints should fail gracefully."""
    base_dir = str(tmp_path / "base_run")
    os.makedirs(base_dir, exist_ok=True)
    # No checkpoints/ directory

    hyps = engine.load_continuation_conditions("mnist")
    if hyps:
        results = engine.launch_condition_tests([hyps[0].node_id], base_dir)
        assert len(results) == 1
        # Should get error about no checkpoints or similar
        r = results[0]
        assert "error" in r or r.get("outcome") is not None


# ─── Graph state after load ─────────────────────────────────────────

def test_graph_stats_after_load(engine):
    engine.load_continuation_conditions("mnist")
    stats = engine.graph.stats()
    assert stats["hypotheses"] > 0

def test_graph_cytoscape_after_load(engine):
    engine.load_continuation_conditions("mnist")
    cy = engine.graph.to_cytoscape()
    elements = cy.get("elements", [])
    assert len(elements) > 0
