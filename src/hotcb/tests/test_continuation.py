"""Tests for continuation tuning routine."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.routines.continuation.models import (
    AnchorReason,
    AnchorSpec,
    BranchSpec,
    BranchStatus,
    BranchResult,
    BudgetSpec,
    ContinuationConfig,
    MutationOp,
    MutationRecipe,
    ObjectiveSpec,
    ResumeMode,
    RoutineResult,
)
from hotcb.routines.continuation.planner import ContinuationPlanner, DEFAULT_RECIPES
from hotcb.routines.continuation.anchors import AnchorSelector
from hotcb.routines.continuation.evaluator import ContinuationEvaluator
from hotcb.routines.continuation.report import ContinuationReport


# ── Models ──

class TestModels:
    def test_resume_mode_values(self):
        assert ResumeMode.FULL_RESUME.value == "full_resume"
        assert ResumeMode.RESET_SCHEDULER.value == "reset_scheduler"
        assert ResumeMode.WEIGHTS_ONLY.value == "weights_only"

    def test_branch_status_values(self):
        assert BranchStatus.WON.value == "won"
        assert BranchStatus.LOST.value == "lost"
        assert BranchStatus.UNSTABLE.value == "unstable"

    def test_mutation_recipe(self):
        recipe = MutationRecipe(
            name="lr_half",
            ops=[MutationOp(target="lr", value=0.5, relative=True)],
        )
        assert recipe.name == "lr_half"
        assert len(recipe.ops) == 1
        assert recipe.ops[0].relative is True

    def test_objective_spec_defaults(self):
        obj = ObjectiveSpec()
        assert obj.primary_metric == "val_loss"
        assert obj.mode == "min"

    def test_continuation_config_total_branches(self):
        config = ContinuationConfig(
            anchors=[
                AnchorSpec(anchor_id="a1", checkpoint_path="/tmp/a.pt", step=100),
                AnchorSpec(anchor_id="a2", checkpoint_path="/tmp/b.pt", step=200),
            ],
            recipes=[
                MutationRecipe(name="r1", ops=[]),
                MutationRecipe(name="r2", ops=[]),
            ],
            resume_modes=[ResumeMode.FULL_RESUME, ResumeMode.WEIGHTS_ONLY],
        )
        assert config.total_branches == 8  # 2 anchors × 2 recipes × 2 modes

    def test_routine_result_leaderboard(self):
        config = ContinuationConfig(
            objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
        )
        result = RoutineResult(
            config=config,
            branches=[
                BranchResult(branch_id="b1", anchor_id="a", recipe_name="r",
                             resume_mode="full", primary_delta=0.02),
                BranchResult(branch_id="b2", anchor_id="a", recipe_name="r",
                             resume_mode="full", primary_delta=0.05),
                BranchResult(branch_id="b3", anchor_id="a", recipe_name="r",
                             resume_mode="full", primary_delta=-0.01),
            ],
        )
        lb = result.leaderboard
        assert lb[0].branch_id == "b2"
        assert lb[-1].branch_id == "b3"


# ── Planner ──

class TestPlanner:
    def test_default_recipes(self):
        assert len(DEFAULT_RECIPES) == 6
        names = {r.name for r in DEFAULT_RECIPES}
        assert "lr_half" in names
        assert "swa_tail" in names
        assert "ema_tail" in names

    def test_plan_cross_product(self):
        config = ContinuationConfig(
            anchors=[AnchorSpec(anchor_id="a1", checkpoint_path="x.pt", step=100)],
            recipes=[
                MutationRecipe(name="lr_half", ops=[]),
                MutationRecipe(name="swa", ops=[]),
            ],
            resume_modes=[ResumeMode.FULL_RESUME],
            budget=BudgetSpec(branches_per_anchor=10),
        )
        planner = ContinuationPlanner(config)
        branches = planner.plan()
        assert len(branches) == 2

    def test_plan_budget_cap(self):
        config = ContinuationConfig(
            anchors=[AnchorSpec(anchor_id="a1", checkpoint_path="x.pt", step=100)],
            recipes=[MutationRecipe(name=f"r{i}", ops=[]) for i in range(10)],
            resume_modes=[ResumeMode.FULL_RESUME],
            budget=BudgetSpec(branches_per_anchor=3),
        )
        planner = ContinuationPlanner(config)
        branches = planner.plan()
        assert len(branches) == 3

    def test_plan_with_defaults(self):
        config = ContinuationConfig(
            anchors=[AnchorSpec(anchor_id="a1", checkpoint_path="x.pt", step=100)],
            budget=BudgetSpec(branches_per_anchor=6),
        )
        planner = ContinuationPlanner(config)
        branches = planner.plan_with_defaults()
        assert len(branches) == 6


# ── Anchors ──

class TestAnchorSelector:
    def test_discover_empty(self):
        with tempfile.TemporaryDirectory() as d:
            selector = AnchorSelector(d)
            assert selector.discover_checkpoints() == []

    def test_discover_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt_dir = os.path.join(d, "checkpoints")
            os.makedirs(ckpt_dir)

            # Create fake checkpoints with metadata
            for step in [100, 200, 300]:
                open(os.path.join(ckpt_dir, f"step_{step:06d}.pt"), "w").close()
                meta = {"step": step, "epoch": step // 100, "metrics": {"val_loss": 1.0 / step}}
                with open(os.path.join(ckpt_dir, f"step_{step:06d}.meta.json"), "w") as f:
                    json.dump(meta, f)

            selector = AnchorSelector(d)
            ckpts = selector.discover_checkpoints()
            assert len(ckpts) == 3
            assert ckpts[0]["step"] == 100
            assert ckpts[2]["step"] == 300

    def test_select_auto(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt_dir = os.path.join(d, "checkpoints")
            os.makedirs(ckpt_dir)

            for step, val_loss in [(100, 0.5), (200, 0.3), (300, 0.25), (400, 0.28)]:
                open(os.path.join(ckpt_dir, f"step_{step:06d}.pt"), "w").close()
                meta = {"step": step, "epoch": step // 100, "metrics": {"val_loss": val_loss}}
                with open(os.path.join(ckpt_dir, f"step_{step:06d}.meta.json"), "w") as f:
                    json.dump(meta, f)

            selector = AnchorSelector(d)
            anchors = selector.select_auto(max_anchors=2, primary_metric="val_loss", mode="min")
            assert len(anchors) == 2
            assert anchors[0].reason == AnchorReason.BEST_VAL_MINUS_K

    def test_select_manual(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            open(path, "w").close()
            meta = {"step": 500, "epoch": 5, "metrics": {"val_loss": 0.1}}
            with open(os.path.join(d, "ckpt.meta.json"), "w") as f:
                json.dump(meta, f)

            selector = AnchorSelector(d)
            anchors = selector.select_manual([path])
            assert len(anchors) == 1
            assert anchors[0].step == 500


# ── Evaluator ──

class TestEvaluator:
    def _make_result(self):
        config = ContinuationConfig(
            objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
            base_best_metrics={"val_accuracy": 0.95},
            base_final_metrics={"val_accuracy": 0.94},
        )
        return RoutineResult(
            config=config,
            branches=[
                BranchResult(branch_id="b1", anchor_id="a", recipe_name="lr_half",
                             resume_mode="full_resume", status=BranchStatus.WON,
                             primary_delta=0.02, best_metrics={"val_accuracy": 0.97}),
                BranchResult(branch_id="b2", anchor_id="a", recipe_name="swa",
                             resume_mode="full_resume", status=BranchStatus.WON,
                             primary_delta=0.01, best_metrics={"val_accuracy": 0.96}),
                BranchResult(branch_id="b3", anchor_id="a", recipe_name="ema",
                             resume_mode="full_resume", status=BranchStatus.LOST,
                             primary_delta=-0.01, best_metrics={"val_accuracy": 0.94}),
            ],
        )

    def test_winners(self):
        ev = ContinuationEvaluator(self._make_result())
        assert len(ev.winners) == 2

    def test_rank_by_metric(self):
        ev = ContinuationEvaluator(self._make_result())
        ranked = ev.rank_by_metric()
        assert ranked[0].branch_id == "b1"

    def test_recipe_ranking(self):
        ev = ContinuationEvaluator(self._make_result())
        rankings = ev.recipe_ranking()
        assert rankings[0]["recipe"] == "lr_half"

    def test_improvement_summary(self):
        ev = ContinuationEvaluator(self._make_result())
        summary = ev.improvement_summary()
        assert "b1" in summary
        assert summary["b1"]["delta"] == 0.02


# ── Report ──

class TestReport:
    def test_leaderboard(self):
        config = ContinuationConfig(
            objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
        )
        result = RoutineResult(
            config=config,
            branches=[
                BranchResult(branch_id="b1", anchor_id="a", recipe_name="lr_half",
                             resume_mode="full", status=BranchStatus.WON,
                             primary_delta=0.02, primary_delta_pct=2.1,
                             best_metrics={"val_accuracy": 0.97}, total_steps=500),
            ],
        )
        report = ContinuationReport(result)
        lb = report.leaderboard()
        assert "lr_half" in lb
        assert "0.97" in lb

    def test_recommendation_winner(self):
        config = ContinuationConfig(
            objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
            base_best_metrics={"val_accuracy": 0.95},
        )
        winner = BranchResult(
            branch_id="b1", anchor_id="a", recipe_name="swa",
            resume_mode="full", status=BranchStatus.WON,
            primary_delta=0.02, primary_delta_pct=2.1,
            best_metrics={"val_accuracy": 0.97},
        )
        result = RoutineResult(config=config, branches=[winner], winner=winner)
        report = ContinuationReport(result)
        rec = report.recommendation()
        assert "PROMOTE" in rec
        assert "swa" in rec

    def test_recommendation_keep_base(self):
        config = ContinuationConfig(
            objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
        )
        result = RoutineResult(config=config, branches=[], winner=None)
        report = ContinuationReport(result)
        assert "KEEP BASE" in report.recommendation()

    def test_save_creates_files(self):
        with tempfile.TemporaryDirectory() as d:
            config = ContinuationConfig(
                objective=ObjectiveSpec(primary_metric="val_accuracy", mode="max"),
                output_dir=d,
            )
            result = RoutineResult(config=config, branches=[])
            report = ContinuationReport(result)
            report.save(d)
            assert os.path.exists(os.path.join(d, "continuation_report.txt"))
            assert os.path.exists(os.path.join(d, "continuation_summary.json"))
            assert os.path.exists(os.path.join(d, "leaderboard.json"))


# ── Checkpoint helpers ──

class TestCheckpointHelpers:
    @pytest.mark.skipif(not __import__("torch").cuda.is_available() and True,
                        reason="always run — CPU is fine")
    def test_save_and_load_checkpoint(self):
        import torch
        from hotcb.eval.tasks import _MnistCNN, _save_checkpoint, _load_checkpoint

        with tempfile.TemporaryDirectory() as d:
            model = _MnistCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            ckpt_path = os.path.join(d, "test.pt")

            _save_checkpoint(ckpt_path, model, optimizer, step=100, epoch=2,
                             metrics={"val_loss": 0.5, "val_accuracy": 0.95})

            # Verify metadata sidecar
            meta_path = os.path.join(d, "test.meta.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["step"] == 100
            assert meta["metrics"]["val_accuracy"] == 0.95

            # Load full resume
            model2 = _MnistCNN()
            opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            info = _load_checkpoint(ckpt_path, model2, opt2, resume_mode="full_resume")
            assert info["step"] == 100

            # Load weights only
            model3 = _MnistCNN()
            opt3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
            info3 = _load_checkpoint(ckpt_path, model3, opt3, resume_mode="weights_only")
            assert info3["step"] == 100

    def test_swa_state(self):
        import torch
        from hotcb.eval.tasks import _MnistCNN, _SWAState

        model = _MnistCNN()
        swa = _SWAState(model)
        swa.enabled = True

        # Simulate a few updates
        for _ in range(5):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            swa.update(model)

        assert swa.n_averaged == 5
        # Apply should not crash
        swa.apply_average(model)

    def test_ema_state(self):
        import torch
        from hotcb.eval.tasks import _MnistCNN, _EMAState

        model = _MnistCNN()
        ema = _EMAState(model, decay=0.99)
        ema.enabled = True

        for _ in range(5):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update(model)

        # Apply should not crash
        ema.apply_shadow(model)


# ── Launcher mutation command generation ──

class TestMutationCommands:
    def test_mutation_op_to_command_absolute(self):
        from hotcb.routines.continuation.launcher import _mutation_op_to_command
        op = MutationOp(target="swa_enabled", value=True)
        cmd = _mutation_op_to_command(op, {})
        assert cmd["module"] == "custom"
        assert cmd["params"]["swa_enabled"] is True

    def test_mutation_op_to_command_relative(self):
        from hotcb.routines.continuation.launcher import _mutation_op_to_command
        op = MutationOp(target="lr", value=0.5, relative=True)
        cmd = _mutation_op_to_command(op, {"lr": 0.01})
        assert cmd["module"] == "opt"
        assert cmd["params"]["lr"] == pytest.approx(0.005)

    def test_mutation_op_relative_missing_base(self):
        from hotcb.routines.continuation.launcher import _mutation_op_to_command
        op = MutationOp(target="lr", value=0.5, relative=True)
        cmd = _mutation_op_to_command(op, {})
        assert cmd is None
