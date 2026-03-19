"""Tests for research graph, types, persistence, and export/merge."""

import json
import os
import tempfile

import pytest

from hotcb.research.types import (
    ObservationNode, HypothesisNode, EvidenceNode,
    ResearchEdge, ResearchStream, HYP_TRANSITIONS,
)
from hotcb.research.graph import ResearchGraph
from hotcb.research.export import export_graph, import_graph, merge_graph


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def graph(tmp_dir):
    return ResearchGraph(tmp_dir)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_observation_roundtrip(self):
        o = ObservationNode(text="test", step=10, tags=["a", "b"])
        d = o.to_dict()
        o2 = ObservationNode.from_dict(d)
        assert o2.text == "test"
        assert o2.tags == ["a", "b"]

    def test_hypothesis_roundtrip(self):
        h = HypothesisNode(
            condition="loss > 5",
            intervention={"module": "opt", "params": {"lr": 0.001}},
            expected_outcome="loss decreases",
        )
        d = h.to_dict()
        h2 = HypothesisNode.from_dict(d)
        assert h2.condition == "loss > 5"
        assert h2.intervention["module"] == "opt"

    def test_evidence_roundtrip(self):
        e = EvidenceNode(hypothesis_id="abc", outcome="improved", delta={"loss": -0.5})
        d = e.to_dict()
        e2 = EvidenceNode.from_dict(d)
        assert e2.outcome == "improved"
        assert e2.delta["loss"] == -0.5

    def test_edge_roundtrip(self):
        e = ResearchEdge(source_id="a", target_id="b", edge_type="supports")
        d = e.to_dict()
        e2 = ResearchEdge.from_dict(d)
        assert e2.source_id == "a"

    def test_stream_roundtrip(self):
        s = ResearchStream(name="test stream", description="desc")
        d = s.to_dict()
        s2 = ResearchStream.from_dict(d)
        assert s2.name == "test stream"

    def test_hyp_transitions_complete(self):
        for status, targets in HYP_TRANSITIONS.items():
            assert isinstance(targets, set)
        assert "testing" in HYP_TRANSITIONS["proposed"]
        assert "confirmed" in HYP_TRANSITIONS["testing"]


# ---------------------------------------------------------------------------
# Graph CRUD
# ---------------------------------------------------------------------------

class TestGraphCRUD:
    def test_add_observation(self, graph):
        obs = graph.add_observation("test obs", step=5, tags=["x"])
        assert obs.node_type == "observation"
        assert obs.text == "test obs"
        assert graph.node_count == 1

    def test_add_hypothesis(self, graph):
        hyp = graph.add_hypothesis(
            condition="grad > 10",
            intervention={"module": "opt", "params": {"lr_mult": 0.5}},
            expected_outcome="loss recovers",
        )
        assert hyp.status == "proposed"
        assert graph.node_count == 1

    def test_add_evidence_increments_count(self, graph):
        hyp = graph.add_hypothesis("cond", {}, "expected")
        ev = graph.add_evidence(hyp.node_id, "improved", delta={"loss": -0.1})
        assert hyp.evidence_count == 1
        assert ev.hypothesis_id == hyp.node_id
        # Auto-edge created
        assert graph.edge_count == 1

    def test_transition_hypothesis(self, graph):
        hyp = graph.add_hypothesis("cond", {}, "expected")
        assert graph.transition_hypothesis(hyp.node_id, "testing")
        assert hyp.status == "testing"
        assert graph.transition_hypothesis(hyp.node_id, "confirmed")
        assert hyp.status == "confirmed"

    def test_invalid_transition(self, graph):
        hyp = graph.add_hypothesis("cond", {}, "expected")
        assert not graph.transition_hypothesis(hyp.node_id, "confirmed")  # proposed -> confirmed not allowed

    def test_create_stream(self, graph):
        s = graph.create_stream("my stream", "desc")
        assert s.status == "active"
        assert len(graph.list_streams()) == 1

    def test_conclude_stream(self, graph):
        s = graph.create_stream("my stream")
        assert graph.conclude_stream(s.stream_id, "done")
        assert s.status == "concluded"
        assert s.conclusion == "done"

    def test_stream_membership(self, graph):
        s = graph.create_stream("s1")
        obs = graph.add_observation("obs", stream_id=s.stream_id)
        hyp = graph.add_hypothesis("cond", {}, "exp", stream_id=s.stream_id)
        assert obs.node_id in s.observation_ids
        assert hyp.node_id in s.hypothesis_ids

    def test_hypotheses_by_status(self, graph):
        graph.add_hypothesis("a", {}, "x", status="proposed")
        graph.add_hypothesis("b", {}, "y", status="discovered")
        assert len(graph.hypotheses_by_status("proposed")) == 1
        assert len(graph.hypotheses_by_status("discovered")) == 1

    def test_evidence_for(self, graph):
        hyp = graph.add_hypothesis("cond", {}, "exp")
        graph.add_evidence(hyp.node_id, "improved")
        graph.add_evidence(hyp.node_id, "degraded")
        assert len(graph.evidence_for(hyp.node_id)) == 2

    def test_stats(self, graph):
        graph.add_observation("obs")
        graph.add_hypothesis("cond", {}, "exp")
        stats = graph.stats()
        assert stats["observations"] == 1
        assert stats["hypotheses"] == 1

    def test_invalid_edge_type(self, graph):
        with pytest.raises(ValueError):
            graph.add_edge("a", "b", "invalid_type")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_event_log_written(self, graph, tmp_dir):
        graph.add_observation("test")
        path = os.path.join(tmp_dir, "hotcb.research.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "add_observation"

    def test_snapshot_roundtrip(self, graph, tmp_dir):
        graph.add_observation("obs1")
        hyp = graph.add_hypothesis("cond", {"a": 1}, "exp")
        graph.add_evidence(hyp.node_id, "improved")
        graph.create_stream("s1")
        graph.save_snapshot()

        g2 = ResearchGraph(tmp_dir)
        assert g2.load_snapshot()
        assert g2.node_count == graph.node_count
        assert g2.edge_count == graph.edge_count
        assert len(g2.list_streams()) == 1

    def test_replay_events(self, tmp_dir):
        g1 = ResearchGraph(tmp_dir)
        g1.add_observation("obs1", step=1)
        hyp = g1.add_hypothesis("cond", {}, "exp", step=2)
        g1.add_evidence(hyp.node_id, "improved", step=3)
        g1.create_stream("s1")

        g2 = ResearchGraph(tmp_dir)
        count = g2.replay_events()
        assert count > 0
        assert g2.node_count == g1.node_count

    def test_replay_transitions(self, tmp_dir):
        g1 = ResearchGraph(tmp_dir)
        hyp = g1.add_hypothesis("cond", {}, "exp")
        g1.transition_hypothesis(hyp.node_id, "testing")

        g2 = ResearchGraph(tmp_dir)
        g2.replay_events()
        replayed = g2.get_node(hyp.node_id)
        assert replayed.status == "testing"


# ---------------------------------------------------------------------------
# Cytoscape serialization
# ---------------------------------------------------------------------------

class TestCytoscape:
    def test_to_cytoscape_basic(self, graph):
        graph.add_observation("obs")
        hyp = graph.add_hypothesis("cond", {}, "exp")
        graph.add_evidence(hyp.node_id, "improved")
        result = graph.to_cytoscape()
        assert "elements" in result
        nodes = [e for e in result["elements"] if e["group"] == "nodes"]
        edges = [e for e in result["elements"] if e["group"] == "edges"]
        assert len(nodes) == 3
        assert len(edges) == 1

    def test_to_cytoscape_stream_filter(self, graph):
        s = graph.create_stream("s1")
        graph.add_observation("in stream", stream_id=s.stream_id)
        graph.add_observation("out of stream")
        result = graph.to_cytoscape(stream_id=s.stream_id)
        nodes = [e for e in result["elements"] if e["group"] == "nodes"]
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# Export / Import / Merge
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_import_roundtrip(self, tmp_dir):
        g = ResearchGraph(tmp_dir)
        g.add_observation("obs1")
        hyp = g.add_hypothesis("cond", {"a": 1}, "exp")
        g.add_evidence(hyp.node_id, "improved")
        g.create_stream("s1")

        export_path = os.path.join(tmp_dir, "export.json")
        export_graph(g, export_path)

        g2 = ResearchGraph(os.path.join(tmp_dir, "run2"))
        count = import_graph(g2, export_path)
        assert count > 0
        assert g2.node_count == g.node_count

    def test_export_stream_filter(self, tmp_dir):
        g = ResearchGraph(tmp_dir)
        s = g.create_stream("s1")
        g.add_observation("in", stream_id=s.stream_id)
        g.add_observation("out")

        export_path = os.path.join(tmp_dir, "export.json")
        export_graph(g, export_path, stream_id=s.stream_id)

        with open(export_path) as f:
            data = json.load(f)
        assert len(data["nodes"]) == 1

    def test_merge_deduplicates(self, tmp_dir):
        g1 = ResearchGraph(os.path.join(tmp_dir, "run1"))
        obs = g1.add_observation("shared obs")
        g1.save_snapshot()

        export_path = os.path.join(tmp_dir, "export.json")
        export_graph(g1, export_path)

        g2 = ResearchGraph(os.path.join(tmp_dir, "run2"))
        import_graph(g2, export_path)
        initial_count = g2.node_count

        # Merge again — should skip existing
        count = merge_graph(g2, export_path)
        assert g2.node_count == initial_count


# ---------------------------------------------------------------------------
# Shadow / Contamination tracking
# ---------------------------------------------------------------------------

class TestShadowTracking:
    def test_shadow_status_in_transitions(self):
        assert "shadow" in HYP_TRANSITIONS
        assert "proposed" in HYP_TRANSITIONS["shadow"]
        assert "archived" in HYP_TRANSITIONS["shadow"]

    def test_add_shadow_hypothesis(self, graph):
        sh = graph.add_shadow_hypothesis(
            mutation_ref="lr:set_params@50",
            intervention={"module": "opt", "op": "set_params", "params": {"lr": 0.001}},
            step=50,
        )
        assert sh.status == "shadow"
        assert sh.mutation_ref == "lr:set_params@50"
        assert graph.has_unacknowledged_shadows
        assert len(graph.shadow_hypotheses()) == 1

    def test_shadow_contaminates_subsequent_nodes(self, graph):
        """Hypotheses/evidence created after a shadow should be diluted."""
        graph.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        # Now create a regular hypothesis — should be diluted
        hyp = graph.add_hypothesis("cond", {}, "exp", step=20)
        assert hyp.diluted is True

        # Evidence should also be diluted
        ev = graph.add_evidence(hyp.node_id, "improved", step=25)
        assert ev.diluted is True

    def test_shadow_itself_not_diluted(self, graph):
        sh = graph.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        # Shadow nodes are not self-diluted
        assert sh.diluted is False

    def test_acknowledge_clears_contamination(self, graph):
        sh = graph.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        assert graph.has_unacknowledged_shadows
        ok = graph.acknowledge_shadow(sh.node_id)
        assert ok
        assert not graph.has_unacknowledged_shadows

        # New nodes after acknowledge should not be diluted
        hyp = graph.add_hypothesis("cond", {}, "exp", step=30)
        assert hyp.diluted is False

    def test_promote_shadow(self, graph):
        sh = graph.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        ok = graph.promote_shadow(sh.node_id, condition="lr was high", expected_outcome="loss drops")
        assert ok
        assert sh.status == "proposed"
        assert sh.condition == "lr was high"
        assert sh.diluted is False
        assert not graph.has_unacknowledged_shadows

    def test_promote_invalid_node(self, graph):
        hyp = graph.add_hypothesis("cond", {}, "exp")
        assert not graph.promote_shadow(hyp.node_id)

    def test_shadow_persists_via_events(self, tmp_dir):
        g1 = ResearchGraph(tmp_dir)
        sh = g1.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        g2 = ResearchGraph(tmp_dir)
        g2.replay_events()
        assert g2.has_unacknowledged_shadows
        assert len(g2.shadow_hypotheses()) == 1

    def test_acknowledge_persists_via_events(self, tmp_dir):
        g1 = ResearchGraph(tmp_dir)
        sh = g1.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        g1.acknowledge_shadow(sh.node_id)

        g2 = ResearchGraph(tmp_dir)
        g2.replay_events()
        assert not g2.has_unacknowledged_shadows

    def test_shadow_snapshot_roundtrip(self, tmp_dir):
        g1 = ResearchGraph(tmp_dir)
        g1.add_shadow_hypothesis(
            mutation_ref="lr:set@10", intervention={}, step=10,
        )
        g1.save_snapshot()

        g2 = ResearchGraph(tmp_dir)
        g2.load_snapshot()
        assert g2.has_unacknowledged_shadows
        assert len(g2.shadow_hypotheses()) == 1

    def test_stats_include_shadow_info(self, graph):
        graph.add_shadow_hypothesis(mutation_ref="x", intervention={}, step=1)
        graph.add_hypothesis("cond", {}, "exp", step=5)  # diluted
        stats = graph.stats()
        assert stats["shadow_count"] == 1
        assert stats["diluted_count"] == 1
        assert stats["unacknowledged_shadows"] == 1

    def test_diluted_nodes_query(self, graph):
        graph.add_shadow_hypothesis(mutation_ref="x", intervention={}, step=1)
        graph.add_hypothesis("cond", {}, "exp", step=5)
        diluted = graph.diluted_nodes()
        assert len(diluted) == 1

    def test_multiple_shadows_independent(self, graph):
        sh1 = graph.add_shadow_hypothesis(mutation_ref="a", intervention={}, step=10)
        sh2 = graph.add_shadow_hypothesis(mutation_ref="b", intervention={}, step=20)
        assert len(graph._shadow_steps) == 2

        graph.acknowledge_shadow(sh1.node_id)
        # Still contaminated because sh2 is unacknowledged
        assert graph.has_unacknowledged_shadows
        hyp = graph.add_hypothesis("cond", {}, "exp", step=30)
        assert hyp.diluted is True

        graph.acknowledge_shadow(sh2.node_id)
        assert not graph.has_unacknowledged_shadows
        hyp2 = graph.add_hypothesis("cond2", {}, "exp2", step=40)
        assert hyp2.diluted is False
