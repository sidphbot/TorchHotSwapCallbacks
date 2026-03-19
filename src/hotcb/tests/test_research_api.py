"""Tests for the research REST API endpoints."""

import json
import os
import tempfile

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        # Create minimal files so app doesn't crash
        open(os.path.join(d, "hotcb.metrics.jsonl"), "w").close()
        open(os.path.join(d, "hotcb.applied.jsonl"), "w").close()
        yield d


@pytest.fixture
def client(tmp_dir):
    from hotcb.server.app import create_app
    app = create_app(tmp_dir)
    with TestClient(app) as c:
        yield c


class TestResearchAPI:
    def test_get_graph_empty(self, client):
        r = client.get("/api/research/graph")
        assert r.status_code == 200
        assert r.json()["elements"] == []

    def test_get_stats(self, client):
        r = client.get("/api/research/stats")
        assert r.status_code == 200
        assert r.json()["hypotheses"] == 0

    def test_observe(self, client):
        r = client.post("/api/research/observe", json={
            "text": "loss is spiking",
            "step": 50,
            "tags": ["stability"],
        })
        assert r.status_code == 200
        assert r.json()["text"] == "loss is spiking"

    def test_hypothesize(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "grad_norm > 10",
            "intervention": {"module": "opt", "params": {"lr_mult": 0.5}},
            "expected_outcome": "loss recovers",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "proposed"

    def test_list_hypotheses(self, client):
        client.post("/api/research/hypothesize", json={
            "condition": "cond1", "intervention": {}, "expected_outcome": "exp",
        })
        r = client.get("/api/research/hypotheses")
        assert r.status_code == 200
        assert len(r.json()["hypotheses"]) == 1

    def test_list_hypotheses_by_status(self, client):
        client.post("/api/research/hypothesize", json={
            "condition": "cond1", "intervention": {}, "expected_outcome": "exp",
        })
        r = client.get("/api/research/hypotheses?status=proposed")
        assert len(r.json()["hypotheses"]) == 1
        r = client.get("/api/research/hypotheses?status=confirmed")
        assert len(r.json()["hypotheses"]) == 0

    def test_get_hypothesis(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        hyp_id = r.json()["node_id"]
        r = client.get(f"/api/research/hypotheses/{hyp_id}")
        assert r.status_code == 200
        assert r.json()["condition"] == "cond"

    def test_get_hypothesis_not_found(self, client):
        r = client.get("/api/research/hypotheses/nonexistent")
        assert r.status_code == 404

    def test_test_hypothesis(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "cond",
            "intervention": {"module": "opt", "op": "set_params", "params": {"lr": 0.001}},
            "expected_outcome": "exp",
        })
        hyp_id = r.json()["node_id"]
        r = client.post(f"/api/research/hypotheses/{hyp_id}/test")
        assert r.status_code == 200
        assert r.json()["status"] == "testing"

    def test_conclude_hypothesis(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        hyp_id = r.json()["node_id"]
        # Transition to testing first
        client.post(f"/api/research/hypotheses/{hyp_id}/test")
        r = client.post(f"/api/research/hypotheses/{hyp_id}/conclude", json={
            "status": "confirmed", "reason": "evidence supports it",
        })
        assert r.status_code == 200

    def test_conclude_invalid_transition(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        hyp_id = r.json()["node_id"]
        r = client.post(f"/api/research/hypotheses/{hyp_id}/conclude", json={
            "status": "confirmed",
        })
        assert r.status_code == 400

    def test_streams_crud(self, client):
        r = client.post("/api/research/streams", json={"name": "stability", "description": "investigating"})
        assert r.status_code == 200
        sid = r.json()["stream_id"]

        r = client.get("/api/research/streams")
        assert len(r.json()["streams"]) == 1

        r = client.post(f"/api/research/streams/{sid}/conclude", json={"conclusion": "lr reduction works"})
        assert r.status_code == 200

    def test_refine(self, client):
        r = client.post("/api/research/observe", json={"text": "obs"})
        obs_id = r.json()["node_id"]
        r = client.post(f"/api/research/refine/{obs_id}", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        assert r.status_code == 200

    def test_evidence(self, client):
        r = client.post("/api/research/hypothesize", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        hyp_id = r.json()["node_id"]
        r = client.get(f"/api/research/evidence/{hyp_id}")
        assert r.status_code == 200
        assert r.json()["evidence"] == []

    def test_nn_status(self, client):
        r = client.get("/api/research/nn/status")
        assert r.status_code == 200

    def test_nn_predict(self, client):
        r = client.post("/api/research/nn/predict", json={
            "intervention": {"params": {"lr": 0.001}},
            "context": {},
        })
        assert r.status_code == 200
        assert "predicted_p_improved" in r.json()

    def test_graph_with_data(self, client):
        client.post("/api/research/observe", json={"text": "obs"})
        client.post("/api/research/hypothesize", json={
            "condition": "cond", "intervention": {}, "expected_outcome": "exp",
        })
        r = client.get("/api/research/graph")
        elements = r.json()["elements"]
        nodes = [e for e in elements if e["group"] == "nodes"]
        assert len(nodes) == 2

    def test_export_recipe(self, client):
        r = client.post("/api/research/export-recipe", json={})
        assert r.status_code == 200
        assert "path" in r.json()


class TestShadowAPI:
    def test_list_shadows_empty(self, client):
        r = client.get("/api/research/shadows")
        assert r.status_code == 200
        assert r.json()["shadows"] == []

    def test_contamination_empty(self, client):
        r = client.get("/api/research/contamination")
        assert r.status_code == 200
        assert r.json()["has_contamination"] is False

    def test_diluted_empty(self, client):
        r = client.get("/api/research/diluted")
        assert r.status_code == 200
        assert r.json()["diluted"] == []

    def test_acknowledge_invalid(self, client):
        r = client.post("/api/research/shadows/fake123/acknowledge")
        assert r.status_code == 400

    def test_promote_invalid(self, client):
        r = client.post("/api/research/shadows/fake123/promote", json={
            "condition": "", "expected_outcome": "",
        })
        assert r.status_code == 400
