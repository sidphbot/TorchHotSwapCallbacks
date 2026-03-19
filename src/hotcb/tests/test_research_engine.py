"""Tests for ResearchEngine — orchestrator, auto-discovery, evidence, NN predictions."""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Any

import pytest

from hotcb.research.engine import ResearchEngine
from hotcb.research.types import HypothesisNode


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir):
    return ResearchEngine(tmp_dir)


# ---------------------------------------------------------------------------
# Mock types for testing
# ---------------------------------------------------------------------------

@dataclass
class MockAction:
    rule_id: str = "test_rule"
    proposed_action: dict = None
    condition_met: str = "divergence detected"
    status: str = "applied"

    def __post_init__(self):
        if self.proposed_action is None:
            self.proposed_action = {"module": "opt", "op": "set_params", "params": {"lr_mult": 0.5}}


@dataclass
class MockCompletedEffect:
    param_key: str = "lr"
    step: int = 100
    old_value: float = 0.01
    new_value: float = 0.005
    baseline_metrics: Dict[str, float] = None
    final_metrics: Dict[str, float] = None
    delta: Dict[str, float] = None
    outcome: str = "improved"

    def __post_init__(self):
        if self.baseline_metrics is None:
            self.baseline_metrics = {"train_loss": 1.0}
        if self.final_metrics is None:
            self.final_metrics = {"train_loss": 0.8}
        if self.delta is None:
            self.delta = {"train_loss": -0.2}


class MockEffectTracker:
    def __init__(self):
        self._completed: List[MockCompletedEffect] = []

    def add_completed(self, effect):
        self._completed.append(effect)

    def get_completed(self):
        return self._completed


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestEngineBasic:
    def test_observe(self, engine):
        obs = engine.observe("test observation", step=5, tags=["a"])
        assert obs.text == "test observation"
        assert engine.graph.node_count == 1

    def test_hypothesize(self, engine):
        hyp = engine.hypothesize(
            condition="loss > 5",
            intervention={"module": "opt", "params": {"lr": 0.001}},
            expected_outcome="loss decreases",
        )
        assert hyp.status == "proposed"

    def test_refine(self, engine):
        obs = engine.observe("spikes in grad norm")
        hyp = engine.refine(
            obs.node_id,
            condition="grad_norm > 10",
            intervention={"module": "opt", "params": {"lr_mult": 0.5}},
            expected_outcome="grad norm stabilizes",
        )
        assert hyp.stream_id == obs.stream_id
        # Check refines_to edge
        edges = engine.graph.edges_for(obs.node_id)
        assert any(e.edge_type == "refines_to" for e in edges)

    def test_refine_nonexistent_obs(self, engine):
        with pytest.raises(ValueError):
            engine.refine("nonexistent", "c", {}, "e")

    def test_test_hypothesis(self, engine, tmp_dir):
        hyp = engine.hypothesize("cond", {"module": "opt", "op": "set_params", "params": {"lr": 0.001}}, "exp")
        result = engine.test_hypothesis(hyp.node_id)
        assert result["status"] == "testing"
        assert hyp.status == "testing"
        # Command written
        cmd_path = os.path.join(tmp_dir, "hotcb.commands.jsonl")
        assert os.path.exists(cmd_path)

    def test_test_hypothesis_wrong_status(self, engine):
        hyp = engine.hypothesize("cond", {}, "exp")
        engine.graph.transition_hypothesis(hyp.node_id, "testing")
        with pytest.raises(ValueError):
            engine.test_hypothesis(hyp.node_id)

    def test_conclude_hypothesis(self, engine):
        hyp = engine.hypothesize("cond", {}, "exp")
        engine.graph.transition_hypothesis(hyp.node_id, "testing")
        assert engine.conclude_hypothesis(hyp.node_id, "confirmed")
        assert hyp.status == "confirmed"


# ---------------------------------------------------------------------------
# Auto-discovery from autopilot
# ---------------------------------------------------------------------------

class TestAutoDiscovery:
    def test_on_rule_fired_creates_discovered_hyp(self, engine):
        action = MockAction()
        hyp = engine.on_rule_fired(action, step=100)
        assert hyp.status == "discovered"
        assert hyp.source == "auto_rule:test_rule"
        assert engine.graph.node_count == 1

    def test_on_rule_fired_deduplicates(self, engine):
        action = MockAction()
        engine.on_rule_fired(action, step=100)
        engine.on_rule_fired(action, step=200)  # same condition + intervention
        assert engine.graph.node_count == 1

    def test_on_rule_fired_different_actions(self, engine):
        a1 = MockAction(rule_id="r1", proposed_action={"module": "opt", "params": {"lr": 0.1}})
        a2 = MockAction(rule_id="r2", proposed_action={"module": "opt", "params": {"lr": 0.5}})
        engine.on_rule_fired(a1, step=100)
        engine.on_rule_fired(a2, step=200)
        assert engine.graph.node_count == 2


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

class TestEvidenceCollection:
    def test_on_effect_completed_creates_evidence(self, engine):
        hyp = engine.hypothesize(
            "cond",
            {"module": "opt", "params": {"lr": 0.005}},
            "exp",
        )
        effect = MockCompletedEffect(param_key="lr")
        engine.on_effect_completed(effect, step=150)
        evidence = engine.graph.evidence_for(hyp.node_id)
        assert len(evidence) == 1
        assert evidence[0].outcome == "improved"

    def test_confidence_updated(self, engine):
        hyp = engine.hypothesize("cond", {"module": "opt", "params": {"lr": 0.005}}, "exp")
        # Two improved effects
        e1 = MockCompletedEffect(param_key="lr", step=100, outcome="improved")
        e2 = MockCompletedEffect(param_key="lr", step=200, outcome="improved")
        engine.on_effect_completed(e1, step=100)
        engine.on_effect_completed(e2, step=200)
        assert hyp.confidence == 1.0

    def test_on_metrics_polls_tracker(self, engine):
        tracker = MockEffectTracker()
        engine.set_effect_tracker(tracker)
        hyp = engine.hypothesize("cond", {"module": "opt", "params": {"lr": 0.005}}, "exp")
        tracker.add_completed(MockCompletedEffect(param_key="lr"))
        engine.on_metrics(step=100, metrics={"train_loss": 0.5})
        evidence = engine.graph.evidence_for(hyp.node_id)
        assert len(evidence) == 1


# ---------------------------------------------------------------------------
# Export confirmed recipe
# ---------------------------------------------------------------------------

class TestExportRecipe:
    def test_export_confirmed_recipe(self, engine, tmp_dir):
        hyp = engine.hypothesize(
            "cond",
            {"module": "opt", "op": "set_params", "params": {"lr": 0.001}},
            "exp",
        )
        engine.graph.transition_hypothesis(hyp.node_id, "testing")
        engine.graph.transition_hypothesis(hyp.node_id, "confirmed")

        path = engine.export_confirmed_recipe()
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["module"] == "opt"


# ---------------------------------------------------------------------------
# NN mode (basic, without torch)
# ---------------------------------------------------------------------------

class TestNNMode:
    def test_predict_without_model(self, engine):
        # Without nn_mode, returns 0.5
        score = engine.predict_outcome({"params": {"lr": 0.001}}, {})
        assert score == 0.5

    def test_suggest_without_model(self, engine):
        suggestions = engine.suggest_interventions({})
        assert suggestions == []

    def test_discover_without_model(self, engine):
        patterns = engine.discover_patterns()
        assert patterns == []


# ---------------------------------------------------------------------------
# Graph persistence via engine
# ---------------------------------------------------------------------------

class TestEnginePersistence:
    def test_engine_loads_existing_state(self, tmp_dir):
        e1 = ResearchEngine(tmp_dir)
        e1.observe("test obs", step=5)
        e1.graph.save_snapshot()

        e2 = ResearchEngine(tmp_dir)
        assert e2.graph.node_count == 1


# ---------------------------------------------------------------------------
# Shadow tracking via engine
# ---------------------------------------------------------------------------

class TestShadowEngine:
    def test_on_mutation_creates_shadow(self, engine):
        """Untracked mutation creates a shadow hypothesis."""
        engine.on_mutation_applied(50, {
            "param_key": "lr",
            "op": "set_params",
            "module": "opt",
            "params": {"lr": 0.001},
        })
        shadows = engine.graph.shadow_hypotheses()
        assert len(shadows) == 1
        assert shadows[0].status == "shadow"
        assert shadows[0].mutation_ref == "lr:set_params@50"

    def test_on_mutation_skips_tracked(self, engine):
        """Mutation with a matching hypothesis should not create a shadow."""
        engine.hypothesize("lr too high", {"module": "opt", "op": "set_params", "params": {"lr": 0.001}}, "loss drops")
        engine.on_mutation_applied(50, {
            "param_key": "lr",
            "op": "set_params",
            "module": "opt",
            "params": {"lr": 0.001},
        })
        shadows = engine.graph.shadow_hypotheses()
        assert len(shadows) == 0

    def test_on_mutation_deduplicates(self, engine):
        """Same mutation ref should not create duplicate shadows."""
        record = {"param_key": "lr", "op": "set_params", "module": "opt", "params": {}}
        engine.on_mutation_applied(50, record)
        engine.on_mutation_applied(50, record)
        shadows = engine.graph.shadow_hypotheses()
        assert len(shadows) == 1

    def test_contamination_status(self, engine):
        engine.on_mutation_applied(50, {
            "param_key": "lr", "op": "set", "module": "opt", "params": {},
        })
        status = engine.contamination_status
        assert status["has_contamination"] is True
        assert status["shadow_count"] == 1
        assert 50 in status["shadow_steps"]

    def test_acknowledge_shadow_via_engine(self, engine):
        engine.on_mutation_applied(50, {
            "param_key": "lr", "op": "set", "module": "opt", "params": {},
        })
        shadow_id = engine.graph.shadow_hypotheses()[0].node_id
        ok = engine.acknowledge_shadow(shadow_id)
        assert ok
        assert not engine.contamination_status["has_contamination"]

    def test_promote_shadow_via_engine(self, engine):
        engine.on_mutation_applied(50, {
            "param_key": "lr", "op": "set", "module": "opt", "params": {},
        })
        shadow_id = engine.graph.shadow_hypotheses()[0].node_id
        ok = engine.promote_shadow(shadow_id, "lr was too high", "loss should drop")
        assert ok
        hyp = engine.graph.get_node(shadow_id)
        assert hyp.status == "proposed"
        assert hyp.condition == "lr was too high"

    def test_shadow_dilutes_subsequent_evidence(self, engine):
        """Evidence created while shadows exist should be diluted."""
        engine.on_mutation_applied(50, {
            "param_key": "wd", "op": "set", "module": "opt", "params": {},
        })
        # Create a hypothesis and add evidence — should be diluted
        hyp = engine.hypothesize("cond", {"module": "opt", "params": {"lr": 0.001}}, "exp")
        assert hyp.diluted is True

        tracker = MockEffectTracker()
        engine.set_effect_tracker(tracker)
        tracker.add_completed(MockCompletedEffect(param_key="lr"))
        engine.on_metrics(step=100, metrics={"loss": 0.5})

        evidence = engine.graph.evidence_for(hyp.node_id)
        assert len(evidence) == 1
        assert evidence[0].diluted is True
