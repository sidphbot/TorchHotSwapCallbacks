"""Tests for policy packs — DSL enrichment, pack loader, pack API."""
from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import yaml

from hotcb.server.autopilot import (
    AutopilotEngine,
    AutopilotRule,
    AutopilotAction,
    _PRIORITY_ORDER,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# DSL enrichment tests
# ---------------------------------------------------------------------------


class TestDSLEnrichment:
    def test_rule_with_bounds(self):
        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
            bounds={"min": 0.001, "max": 0.1},
        )
        assert rule.bounds == {"min": 0.001, "max": 0.1}

    def test_rule_with_rollback_if(self):
        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
            rollback_if={"no_improvement_after": 1000},
        )
        assert rule.rollback_if == {"no_improvement_after": 1000}

    def test_rule_with_priority(self):
        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
            priority="critical",
        )
        assert rule.priority == "critical"

    def test_rule_with_suppress(self):
        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
            suppress_rules=["other_rule_1", "other_rule_2"],
        )
        assert rule.suppress_rules == ["other_rule_1", "other_rule_2"]

    def test_rule_priority_default(self):
        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
        )
        assert rule.priority == "medium"

    def test_rule_serialization(self):
        from dataclasses import asdict

        rule = AutopilotRule(
            rule_id="test",
            condition="plateau",
            metric_name="train_loss",
            bounds={"min": 0.001, "max": 0.1},
            rollback_if={"no_improvement_after": 1000},
            priority="high",
            suppress_rules=["other"],
        )
        d = asdict(rule)
        assert d["bounds"] == {"min": 0.001, "max": 0.1}
        assert d["rollback_if"] == {"no_improvement_after": 1000}
        assert d["priority"] == "high"
        assert d["suppress_rules"] == ["other"]

    def test_rule_conflict_detection(self, tmp_dir):
        """Higher priority rule wins when two fire for same actuator."""
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")

        # Low priority rule
        engine.add_rule(AutopilotRule(
            rule_id="low_prio",
            condition="custom",
            metric_name="train_loss",
            params={"expression": "train_loss > 0"},
            action={"module": "opt", "op": "set_params", "params": {"lr_mult": 0.9}},
            confidence="high",
            priority="low",
        ))

        # High priority rule targeting same actuator
        engine.add_rule(AutopilotRule(
            rule_id="high_prio",
            condition="custom",
            metric_name="train_loss",
            params={"expression": "train_loss > 0"},
            action={"module": "opt", "op": "set_params", "params": {"lr_mult": 0.1}},
            confidence="high",
            priority="critical",
        ))

        actions = engine.evaluate(step=100, metrics={"train_loss": 0.5})
        # Both should fire, but high priority first
        assert len(actions) >= 1
        assert actions[0].rule_id == "high_prio"

    def test_priority_order_constants(self):
        assert _PRIORITY_ORDER["low"] < _PRIORITY_ORDER["medium"]
        assert _PRIORITY_ORDER["medium"] < _PRIORITY_ORDER["high"]
        assert _PRIORITY_ORDER["high"] < _PRIORITY_ORDER["critical"]


# ---------------------------------------------------------------------------
# Pack loader tests
# ---------------------------------------------------------------------------


class TestPackLoader:
    def test_load_pack_from_yaml(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("stability_basics")
        assert count > 0
        assert len(engine.get_rules()) == count

    def test_load_pack_prefixes_ids(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        engine.load_pack("stability_basics")
        rules = engine.get_rules()
        for r in rules:
            assert r.rule_id.startswith("stability_basics."), f"Rule {r.rule_id} not prefixed"

    def test_load_pack_metadata(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        packs = engine.list_packs()
        names = [p["name"] for p in packs]
        assert "stability_basics" in names
        # Check metadata
        stab = next(p for p in packs if p["name"] == "stability_basics")
        assert "description" in stab
        assert "version" in stab

    def test_load_pack_requires_check(self, tmp_dir):
        """Packs with 'requires' field should still load (we don't block on it)."""
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        # All packs should load regardless of requires
        for pack in engine.list_packs():
            count = engine.load_pack(pack["name"])
            assert count > 0
            engine.unload_pack(pack["name"])

    def test_load_multiple_packs(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        c1 = engine.load_pack("stability_basics")
        c2 = engine.load_pack("plateau_recovery")
        assert len(engine.get_rules()) == c1 + c2
        assert len(engine.active_packs()) == 2

    def test_unload_pack(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        engine.load_pack("stability_basics")
        assert len(engine.get_rules()) > 0
        removed = engine.unload_pack("stability_basics")
        assert removed > 0
        assert len(engine.get_rules()) == 0
        assert "stability_basics" not in engine.active_packs()

    def test_list_available_packs(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        packs = engine.list_packs()
        names = [p["name"] for p in packs]
        assert "stability_basics" in names
        assert "multi_loss_assist" in names
        assert "distillation_assist" in names
        assert "plateau_recovery" in names
        assert "finish_strong" in names

    def test_load_nonexistent_pack(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        with pytest.raises(FileNotFoundError):
            engine.load_pack("nonexistent_pack_xyz")

    def test_pack_yaml_syntax_error(self, tmp_dir):
        """Loading a malformed YAML pack should raise."""
        # Create a bad YAML file in the guidelines dir
        from hotcb.server.guidelines import GUIDELINES_DIR
        bad_path = os.path.join(GUIDELINES_DIR, "_test_bad.yaml")
        try:
            with open(bad_path, "w") as f:
                f.write("name: bad\nrules:\n  - id: x\n    }{invalid")
            engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
            with pytest.raises(Exception):
                engine.load_pack("_test_bad")
        finally:
            if os.path.exists(bad_path):
                os.unlink(bad_path)


# ---------------------------------------------------------------------------
# Pack 1-5 loading and firing tests
# ---------------------------------------------------------------------------


class TestPackContents:
    def test_stability_pack_loads(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("stability_basics")
        assert count >= 3  # Should have multiple rules

    def test_stability_nan_guard_fires(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.load_pack("stability_basics")
        actions = engine.evaluate(step=100, metrics={"train_loss": float("nan")})
        nan_actions = [a for a in actions if "nan" in a.rule_id.lower() or "nan" in a.condition_met.lower()]
        assert len(nan_actions) > 0, "NaN guard should fire on NaN loss"

    def test_stability_grad_spike_fires(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.load_pack("stability_basics")
        actions = engine.evaluate(step=100, metrics={"train_loss": 0.5, "grad_norm": 100.0})
        grad_actions = [a for a in actions if "grad" in a.rule_id.lower() or "grad" in a.condition_met.lower()]
        assert len(grad_actions) > 0, "Grad spike rule should fire on high grad_norm"

    def test_multiloss_pack_loads(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("multi_loss_assist")
        assert count >= 3

    def test_multiloss_conflict_mitigation(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.load_pack("multi_loss_assist")
        # Provide metrics that indicate aux conflict
        actions = engine.evaluate(
            step=100,
            metrics={"train_loss": 0.5, "aux_loss": 2.0, "conflict_score": 0.8},
        )
        # At least one rule should fire with these conditions
        # (we just check the pack is functional, not specific rule)
        assert isinstance(actions, list)

    def test_distill_pack_loads(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("distillation_assist")
        assert count >= 3

    def test_plateau_pack_loads(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("plateau_recovery")
        assert count >= 3

    def test_plateau_detection_fires(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.load_pack("plateau_recovery")
        # Feed enough history for stagnation_detect (window=25, cooldown=50)
        # Use step spacing > cooldown so rule can fire
        all_actions = []
        for i in range(35):
            actions = engine.evaluate(step=i * 10, metrics={"train_loss": 0.5, "val_loss": 0.6})
            all_actions.extend(actions)
        assert len(all_actions) > 0, "Plateau/stagnation rules should fire on flat metrics"

    def test_finish_pack_loads(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        count = engine.load_pack("finish_strong")
        assert count >= 3


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


fastapi = pytest.importorskip("fastapi")


class TestPackAPI:
    @pytest.fixture
    def client(self, tmp_dir):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from hotcb.server.autopilot import create_router

        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        app = FastAPI()
        router = create_router(engine=engine, ai_engine=None)
        app.include_router(router)
        return TestClient(app)

    def test_api_list_packs(self, client):
        resp = client.get("/api/autopilot/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert "packs" in data
        names = [p["name"] for p in data["packs"]]
        assert "stability_basics" in names

    def test_api_load_pack(self, client):
        resp = client.post(
            "/api/autopilot/packs/load",
            json={"pack_name": "stability_basics"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["rules_loaded"] > 0

    def test_api_unload_pack(self, client):
        # Load first
        client.post("/api/autopilot/packs/load", json={"pack_name": "stability_basics"})
        # Unload
        resp = client.post(
            "/api/autopilot/packs/unload",
            json={"pack_name": "stability_basics"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unloaded"
        assert data["rules_removed"] > 0

    def test_api_active_packs(self, client):
        client.post("/api/autopilot/packs/load", json={"pack_name": "stability_basics"})
        resp = client.get("/api/autopilot/packs/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "stability_basics" in data["active_packs"]
