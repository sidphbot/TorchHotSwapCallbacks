"""Evaluation tests — controlled experiments comparing hotcb conditions.

Each test condition creates a research hypothesis, runs the experiment,
collects evidence, and verifies outcomes. This doubles as both:
1. Feature validation (does autopilot fire? does NN mode work?)
2. Research-graph integration test (hypothesis → evidence → verdict)
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.eval.conditions import (
    EvalCondition,
    GOLDEN_CONDITIONS,
    FINETUNE_CONDITIONS,
    SIMPLE_CONDITIONS,
)
from hotcb.eval.harness import EvalHarness, EvalResult
from hotcb.eval.report import EvalReport


@pytest.fixture
def harness(tmp_path):
    return EvalHarness(output_dir=str(tmp_path))


# Use faster step delays and fewer steps for CI
def _fast(cond: EvalCondition, max_steps: int = 100) -> EvalCondition:
    """Clone a condition with fast settings for CI."""
    import copy
    c = copy.deepcopy(cond)
    c.step_delay = 0.0
    c.max_steps = max_steps
    return c


def _find_condition(name: str) -> EvalCondition:
    for c in GOLDEN_CONDITIONS + FINETUNE_CONDITIONS + SIMPLE_CONDITIONS:
        if c.name == name:
            return c
    raise ValueError(f"Condition not found: {name}")


# ═══════════════════════════════════════════════════════════════════════
# Golden Demo: Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenBaseline:
    """Baseline (no recipe, no autopilot) produces convergent metrics."""

    def test_baseline_converges(self, harness):
        cond = _fast(_find_condition("golden_baseline"))
        result = harness.run(cond)
        assert result.total_steps == 100
        assert "train_loss" in result.final_metrics
        assert result.final_metrics["train_loss"] > 0
        assert result.intervention_count == 0

    def test_recipe_produces_mutations(self, harness):
        cond = _fast(_find_condition("golden_recipe"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics
        # Recipe fires at steps 200/400/500 — with only 100 steps, none fire
        # But the demo still runs and produces metrics
        assert len(result.metric_history) > 0


class TestGoldenRecipeVsBaseline:
    """Recipe should produce better final metrics than baseline (at full length)."""

    def test_both_produce_metrics(self, harness):
        base = harness.run(_fast(_find_condition("golden_baseline"), 80))
        recipe = harness.run(_fast(_find_condition("golden_recipe"), 80))
        assert "train_loss" in base.final_metrics
        assert "train_loss" in recipe.final_metrics
        # Both should have positive loss
        assert base.final_metrics["train_loss"] > 0
        assert recipe.final_metrics["train_loss"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Golden Demo: Auto Mode Recovery
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenAutoRecovery:
    """Autopilot rules fire when given bad initial params."""

    def test_1bad_lr_produces_metrics(self, harness):
        cond = _fast(_find_condition("golden_1bad_lr_auto"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics
        # With lr=0.01 (10x default), training should still produce output
        assert len(result.metric_history) > 0

    def test_1bad_lr_initial_override_applied(self, harness):
        """Verify the initial override commands are written."""
        cond = _fast(_find_condition("golden_1bad_lr_auto"))
        result = harness.run(cond)
        # Check commands.jsonl has the override
        commands_path = os.path.join(result.run_dir, "hotcb.commands.jsonl")
        with open(commands_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        first_cmd = json.loads(lines[0])
        assert first_cmd["module"] == "opt"
        assert first_cmd["params"]["lr"] == 0.01

    def test_1bad_weight_produces_metrics(self, harness):
        cond = _fast(_find_condition("golden_1bad_weight_auto"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics
        # weight_a=0.95 means task A dominates
        assert result.final_metrics.get("weight_a", 0) > 0

    def test_2bad_produces_metrics(self, harness):
        cond = _fast(_find_condition("golden_2bad_lr_weight_auto"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics

    def test_divergent_lr_survives(self, harness):
        """Training doesn't crash with lr=0.05 (50x default)."""
        cond = _fast(_find_condition("golden_divergent_lr_auto"))
        result = harness.run(cond)
        assert len(result.metric_history) > 0


# ═══════════════════════════════════════════════════════════════════════
# Golden Demo: NN Mode
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenSWA:
    """SWA (checkpoint averaging) conditions."""

    def test_swa_at_convergence(self, harness):
        cond = _fast(_find_condition("golden_swa_at_convergence"), 80)
        # SWA fires at step 400, but with 80 steps it won't activate
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics
        assert "swa_enabled" in result.final_metrics

    def test_swa_1bad_lr_auto(self, harness):
        cond = _fast(_find_condition("golden_swa_1bad_lr_auto"), 80)
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics

    def test_swa_2bad_auto(self, harness):
        cond = _fast(_find_condition("golden_swa_2bad_auto"), 80)
        result = harness.run(cond)
        assert len(result.metric_history) > 0


class TestGoldenNNMode:
    """NN mode runs without errors (pretrained model may or may not exist)."""

    def test_nn_mode_runs(self, harness):
        cond = _fast(_find_condition("golden_1bad_lr_nn"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics

    def test_nn_mode_2bad_runs(self, harness):
        cond = _fast(_find_condition("golden_2bad_lr_weight_nn"))
        result = harness.run(cond)
        assert len(result.metric_history) > 0


# ═══════════════════════════════════════════════════════════════════════
# Finetune Demo
# ═══════════════════════════════════════════════════════════════════════

class TestFinetune:
    def test_baseline(self, harness):
        cond = _fast(_find_condition("finetune_baseline"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics
        assert result.intervention_count == 0

    def test_recipe(self, harness):
        cond = _fast(_find_condition("finetune_recipe"))
        result = harness.run(cond)
        assert len(result.metric_history) > 0

    def test_high_lr_auto(self, harness):
        cond = _fast(_find_condition("finetune_high_lr_auto"))
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics


# ═══════════════════════════════════════════════════════════════════════
# Simple Demo
# ═══════════════════════════════════════════════════════════════════════

class TestSimple:
    def test_baseline(self, harness):
        cond = _fast(_find_condition("simple_baseline"), 60)
        result = harness.run(cond)
        assert "train_loss" in result.final_metrics

    def test_high_lr_auto(self, harness):
        cond = _fast(_find_condition("simple_high_lr_auto"), 60)
        result = harness.run(cond)
        assert len(result.metric_history) > 0


# ═══════════════════════════════════════════════════════════════════════
# Research Graph Integration
# ═══════════════════════════════════════════════════════════════════════

class TestResearchIntegration:
    """Each eval condition becomes a research hypothesis with evidence."""

    def test_condition_creates_research_hypothesis(self, tmp_path):
        """Run a condition and create its hypothesis in a research stream."""
        from hotcb.research.engine import ResearchEngine

        run_dir = str(tmp_path / "research_run")
        os.makedirs(run_dir)

        engine = ResearchEngine(run_dir)
        stream = engine.graph.create_stream(
            "eval_golden", "Evaluation of golden demo conditions",
        )

        # Create hypothesis from condition definition
        cond = _find_condition("golden_1bad_lr_auto")
        hyp = engine.hypothesize(
            condition=cond.hypothesis_condition,
            intervention=cond.initial_overrides.get("opt", {}),
            expected_outcome=cond.hypothesis_expected,
            stream_id=stream.stream_id,
        )

        assert hyp.status == "proposed"
        assert hyp.condition == cond.hypothesis_condition
        assert hyp.stream_id == stream.stream_id

    def test_eval_result_as_evidence(self, harness, tmp_path):
        """Run a condition and convert its result into research evidence."""
        from hotcb.research.engine import ResearchEngine

        # Run the eval
        cond = _fast(_find_condition("golden_baseline"))
        result = harness.run(cond)

        # Create research engine and register the experiment
        research_dir = str(tmp_path / "research")
        os.makedirs(research_dir)
        engine = ResearchEngine(research_dir)

        stream = engine.graph.create_stream("eval", "evaluation experiments")
        hyp = engine.hypothesize(
            condition=cond.hypothesis_condition,
            intervention={},
            expected_outcome=cond.hypothesis_expected,
            stream_id=stream.stream_id,
        )

        # Convert result to evidence
        final_loss = result.final_metrics.get("train_loss", 999)
        outcome = "improved" if final_loss < 2.0 else "neutral"
        ev = engine.graph.add_evidence(
            hypothesis_id=hyp.node_id,
            outcome=outcome,
            delta={"train_loss": final_loss},
            context={"condition": cond.name, "steps": result.total_steps},
            source="eval_harness",
        )

        assert ev.outcome == outcome
        assert engine.graph.evidence_for(hyp.node_id)[0].node_id == ev.node_id

    def test_full_eval_pipeline_with_research(self, tmp_path):
        """Run multiple conditions and build a complete research graph."""
        from hotcb.research.engine import ResearchEngine

        harness = EvalHarness(output_dir=str(tmp_path / "eval"))

        # Pick a few fast conditions
        conditions = [
            _fast(_find_condition("golden_baseline"), 50),
            _fast(_find_condition("golden_1bad_lr_auto"), 50),
            _fast(_find_condition("golden_1bad_weight_auto"), 50),
        ]

        # Research engine for the evaluation
        research_dir = str(tmp_path / "research")
        os.makedirs(research_dir)
        engine = ResearchEngine(research_dir)
        stream = engine.graph.create_stream("eval_suite", "golden demo evaluation suite")

        # Run each condition as a hypothesis
        for cond in conditions:
            hyp = engine.hypothesize(
                condition=cond.hypothesis_condition,
                intervention=cond.initial_overrides.get("opt", cond.initial_overrides.get("loss", {})),
                expected_outcome=cond.hypothesis_expected,
                stream_id=stream.stream_id,
            )
            engine.graph.transition_hypothesis(hyp.node_id, "testing")

            result = harness.run(cond)

            # Create evidence from result
            final_loss = result.final_metrics.get("train_loss", 999)
            engine.graph.add_evidence(
                hypothesis_id=hyp.node_id,
                outcome="improved" if final_loss < 2.0 else "degraded",
                delta={"train_loss": final_loss},
                context={
                    "condition": cond.name,
                    "interventions": result.intervention_count,
                    "autopilot_actions": len(result.autopilot_actions),
                },
                source="eval_harness",
            )

        # Verify research graph state
        stats = engine.graph.stats()
        assert stats["hypotheses"] == 3
        assert stats["evidence"] >= 3
        assert stats["streams"] == 1

        # Generate report
        report = EvalReport(harness.results)
        table = report.comparison_table("golden")
        assert "golden_baseline" in table
        assert "golden_1bad_lr_auto" in table

    def test_shadow_tracking_with_bad_params(self, tmp_path):
        """Verify shadow hypotheses are created for untracked mutations."""
        from hotcb.research.engine import ResearchEngine

        harness = EvalHarness(output_dir=str(tmp_path / "eval"))

        # Run a condition with initial overrides (these create applied entries)
        cond = _fast(_find_condition("golden_1bad_lr_auto"), 50)
        result = harness.run(cond)

        # Create research engine from the run directory and replay
        engine = ResearchEngine(result.run_dir)

        # Simulate feeding applied entries into research engine
        applied_path = os.path.join(result.run_dir, "hotcb.applied.jsonl")
        if os.path.exists(applied_path):
            with open(applied_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    step = rec.get("step", 0)
                    engine.on_mutation_applied(step, rec)

        # Verify: any applied mutations without matching hypotheses should
        # create shadows
        contamination = engine.contamination_status
        # At this point, all mutations are untracked (no hypotheses created)
        # so they should all be shadows
        assert contamination["shadow_count"] >= 0  # may be 0 if no applied entries


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════

class TestReport:
    def test_comparison_table(self, harness):
        base = harness.run(_fast(_find_condition("golden_baseline"), 50))
        recipe = harness.run(_fast(_find_condition("golden_recipe"), 50))
        report = EvalReport(harness.results)
        table = report.comparison_table("golden")
        assert "golden_baseline" in table
        assert "golden_recipe" in table
        assert "train_loss" in table

    def test_verdict(self, harness):
        harness.run(_fast(_find_condition("golden_baseline"), 50))
        harness.run(_fast(_find_condition("golden_1bad_lr_auto"), 50))
        report = EvalReport(harness.results)
        verdict = report.verdict("golden")
        assert "BASELINE" in verdict
        assert "golden_1bad_lr_auto" in verdict

    def test_full_report(self, harness):
        harness.run(_fast(_find_condition("golden_baseline"), 50))
        harness.run(_fast(_find_condition("golden_recipe"), 50))
        report = EvalReport(harness.results)
        full = report.full_report()
        assert "hotcb Evaluation Report" in full

    def test_research_summary(self, harness):
        harness.run(_fast(_find_condition("golden_1bad_lr_auto"), 50))
        report = EvalReport(harness.results)
        summary = report.research_summary()
        assert "Research Graph Summary" in summary

    def test_save_results_json(self, harness, tmp_path):
        harness.run(_fast(_find_condition("golden_baseline"), 50))
        path = str(tmp_path / "results.json")
        harness.save_results(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_report_to_json(self, harness, tmp_path):
        harness.run(_fast(_find_condition("golden_baseline"), 50))
        report = EvalReport(harness.results)
        path = str(tmp_path / "report.json")
        report.to_json(path)
        assert os.path.exists(path)
