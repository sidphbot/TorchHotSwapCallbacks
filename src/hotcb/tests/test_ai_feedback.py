"""Tests for AI feedback loop: action outcomes, cross-run learning,
adaptive cadence, retry/fallback, mode auto-progression."""

import asyncio
import os
import tempfile
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hotcb.server.ai_engine import AIConfig, AIState, LLMAutopilotEngine
from hotcb.server.ai_prompts import build_context
from hotcb.server.autopilot import AutopilotEngine
from hotcb.tracking import CompletedEffect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_state_dict(**overrides):
    base = AIState().to_dict()
    base.update(overrides)
    return base


def _make_effects(outcomes=None):
    """Build a list of CompletedEffect with given outcomes."""
    if outcomes is None:
        outcomes = ["improved"]
    effects = []
    for i, outcome in enumerate(outcomes):
        effects.append(CompletedEffect(
            param_key=f"lr" if i == 0 else f"param_{i}",
            step=100 + i * 50,
            old_value=1e-3 / (i + 1),
            new_value=5e-4 / (i + 1),
            baseline_metrics={"val_loss": 0.5 + i * 0.1},
            final_metrics={"val_loss": 0.45 + i * 0.1},
            delta={"val_loss": -0.05},
            outcome=outcome,
        ))
    return effects


def _make_engine(tmp_path=None):
    """Create an LLMAutopilotEngine with a temp dir."""
    d = tmp_path or tempfile.mkdtemp()
    cfg = AIConfig(api_key="test-key-1234")
    engine = LLMAutopilotEngine(run_dir=d, config=cfg)
    engine._last_invoked_step = 0
    return engine


# ===========================================================================
# 1. Action Outcome Reporting in AI Prompt
# ===========================================================================


class TestActionOutcomes:
    def test_ai_prompt_includes_action_outcomes(self):
        effects = _make_effects(["improved", "degraded"])
        msgs = build_context(
            step=200,
            metric_history={"val_loss": [0.5, 0.4, 0.35]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(),
            completed_effects=effects,
        )
        user_msg = msgs[1]["content"]
        assert "Recent Action Outcomes" in user_msg
        assert "improved" in user_msg
        assert "degraded" in user_msg

    def test_ai_prompt_no_outcomes(self):
        msgs = build_context(
            step=200,
            metric_history={"val_loss": [0.5, 0.4]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(),
            completed_effects=[],
        )
        user_msg = msgs[1]["content"]
        assert "Recent Action Outcomes" not in user_msg

    def test_outcome_formatting(self):
        effects = _make_effects(["improved"])
        msgs = build_context(
            step=200,
            metric_history={"val_loss": [0.5, 0.4]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(),
            completed_effects=effects,
        )
        user_msg = msgs[1]["content"]
        # Should have table headers
        assert "Param" in user_msg
        assert "Outcome" in user_msg
        # Should have the param key
        assert "lr" in user_msg

    def test_outcome_includes_metric_deltas(self):
        effects = [CompletedEffect(
            param_key="lr",
            step=150,
            old_value=1e-3,
            new_value=5e-4,
            baseline_metrics={"val_loss": 0.5},
            final_metrics={"val_loss": 0.45},
            delta={"val_loss": -0.05},
            outcome="improved",
        )]
        msgs = build_context(
            step=200,
            metric_history={"val_loss": [0.5, 0.45]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(),
            completed_effects=effects,
        )
        user_msg = msgs[1]["content"]
        assert "-0.05" in user_msg


# ===========================================================================
# 2. Cross-Run Context in Prompt
# ===========================================================================


class TestCrossRunContext:
    def test_cross_run_context_in_prompt(self):
        run_history = [
            {
                "run_id": "run_001",
                "final_key_metric": 0.342,
                "ai_verdict": "partial_success",
                "carried_learnings": ["lr too high initially"],
            }
        ]
        msgs = build_context(
            step=50,
            metric_history={"val_loss": [0.5]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(run_history=run_history, run_number=2),
        )
        system_msg = msgs[0]["content"]
        assert "run_001" in system_msg
        assert "partial_success" in system_msg
        assert "lr too high initially" in system_msg

    def test_cross_run_empty(self):
        msgs = build_context(
            step=50,
            metric_history={"val_loss": [0.5]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(run_history=[]),
        )
        system_msg = msgs[0]["content"]
        assert "Previous Run History" not in system_msg

    def test_cross_run_weight_recent(self):
        """Most recent runs should appear in context."""
        run_history = [
            {"run_id": f"run_{i:03d}", "final_key_metric": 0.5 - i * 0.1,
             "ai_verdict": "continue", "carried_learnings": [f"learning_{i}"]}
            for i in range(5)
        ]
        msgs = build_context(
            step=50,
            metric_history={"val_loss": [0.5]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(run_history=run_history),
        )
        system_msg = msgs[0]["content"]
        # Last 3 should be present (runs 2, 3, 4)
        assert "run_004" in system_msg
        assert "run_003" in system_msg
        assert "run_002" in system_msg

    def test_cross_run_max_entries(self):
        """Only last 3 runs shown."""
        run_history = [
            {"run_id": f"run_{i:03d}", "final_key_metric": 0.5,
             "ai_verdict": "ok", "carried_learnings": []}
            for i in range(10)
        ]
        msgs = build_context(
            step=50,
            metric_history={"val_loss": [0.5]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state=_ai_state_dict(run_history=run_history),
        )
        system_msg = msgs[0]["content"]
        # run_000 through run_006 should NOT appear
        assert "run_000" not in system_msg
        assert "run_006" not in system_msg
        # run_007 through run_009 SHOULD appear (last 3)
        assert "run_007" in system_msg
        assert "run_008" in system_msg
        assert "run_009" in system_msg


# ===========================================================================
# 3. Adaptive Cadence
# ===========================================================================


class TestAdaptiveCadence:
    def test_cadence_shortens_on_success(self):
        engine = _make_engine()
        engine.config.cadence = 100
        # 3 consecutive improved
        for _ in range(3):
            engine.update_outcome_streak("improved")
        assert engine.state.outcome_streak == ["improved"] * 3
        # Now should_invoke should use shorter cadence
        # Check adaptive interval
        interval = engine._adaptive_interval()
        assert interval == 50  # 100 * 0.5

    def test_cadence_lengthens_on_neutral(self):
        engine = _make_engine()
        engine.config.cadence = 100
        for _ in range(3):
            engine.update_outcome_streak("neutral")
        interval = engine._adaptive_interval()
        assert interval == 200  # 100 * 2.0

    def test_cadence_resets_on_degraded(self):
        engine = _make_engine()
        engine.config.cadence = 100
        for _ in range(3):
            engine.update_outcome_streak("improved")
        engine.update_outcome_streak("degraded")
        interval = engine._adaptive_interval()
        assert interval == 100  # reset to base

    def test_cadence_respects_bounds(self):
        engine = _make_engine()
        # Very small cadence — floor should be 50
        engine.config.cadence = 60
        for _ in range(3):
            engine.update_outcome_streak("improved")
        interval = engine._adaptive_interval()
        assert interval >= 50  # min bound

        # Very large cadence — ceiling should be 500
        engine.config.cadence = 400
        for _ in range(3):
            engine.update_outcome_streak("neutral")
        interval = engine._adaptive_interval()
        assert interval <= 500  # max bound


# ===========================================================================
# 4. Failure Recovery
# ===========================================================================


class TestFailureRecovery:
    @pytest.mark.asyncio
    async def test_llm_retry_on_timeout(self):
        engine = _make_engine()
        call_count = 0

        async def flaky_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout")
            return '{"reasoning":"ok","actions":[],"next_check":{"mode":"periodic"}}', 0.01

        engine._raw_llm_call = flaky_call
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await engine._call_llm([{"role": "user", "content": "test"}])
        assert result[0] is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_llm_retry_on_error(self):
        engine = _make_engine()
        call_count = 0

        async def flaky_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            return '{"reasoning":"ok","actions":[],"next_check":{"mode":"periodic"}}', 0.01

        engine._raw_llm_call = flaky_call
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await engine._call_llm([{"role": "user", "content": "test"}])
        assert result is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_llm_fallback_after_double_failure(self):
        engine = _make_engine()
        engine._last_invoked_step = 100

        async def always_fail(messages):
            raise ConnectionError("refused")

        engine._raw_llm_call = always_fail
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await engine._call_llm(
                [{"role": "user", "content": "test"}],
                current_step=100,
            )
        assert result == (None, 0.0)
        assert engine._fallback_until_step == 200  # 100 + 100

    @pytest.mark.asyncio
    async def test_fallback_recovers(self):
        engine = _make_engine()
        engine._fallback_until_step = 150

        # At step 100, should_invoke returns False (in fallback)
        assert engine.should_invoke(100, []) is False

        # At step 200, should_invoke can return True (past fallback)
        engine._fallback_until_step = 150
        engine._last_invoked_step = 0
        # Periodic cadence should kick in
        engine.config.cadence = 50
        assert engine.should_invoke(200, []) is True


# ===========================================================================
# 5. Mode Auto-Progression
# ===========================================================================


class TestModeAutoProgression:
    def test_auto_progress_suggest_to_auto(self):
        with tempfile.TemporaryDirectory() as d:
            ap = AutopilotEngine(run_dir=d, mode="ai_suggest")
            ai_engine = MagicMock()
            ap.set_ai_engine(ai_engine)
            ap._auto_progress_enabled = True
            ap._auto_progress_threshold = 3

            # Create 3 proposed actions and accept them
            from hotcb.server.autopilot import AutopilotAction
            for i in range(3):
                action = AutopilotAction(
                    action_id=f"act_{i}",
                    rule_id="ai:set_lr",
                    step=i * 10,
                    wall_time=0.0,
                    condition_met="test",
                    proposed_action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
                    confidence="medium",
                    status="proposed",
                )
                ap._history.append(action)
                ap.accept_action(f"act_{i}")

            assert ap.mode == "ai_auto"

    def test_auto_progress_respects_config(self):
        with tempfile.TemporaryDirectory() as d:
            ap = AutopilotEngine(run_dir=d, mode="ai_suggest")
            ai_engine = MagicMock()
            ap.set_ai_engine(ai_engine)
            # Disabled by default
            ap._auto_progress_enabled = False
            ap._auto_progress_threshold = 3

            from hotcb.server.autopilot import AutopilotAction
            for i in range(5):
                action = AutopilotAction(
                    action_id=f"act_{i}",
                    rule_id="ai:set_lr",
                    step=i * 10,
                    wall_time=0.0,
                    condition_met="test",
                    proposed_action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
                    confidence="medium",
                    status="proposed",
                )
                ap._history.append(action)
                ap.accept_action(f"act_{i}")

            # Should still be ai_suggest because auto_progress_enabled=False
            assert ap.mode == "ai_suggest"

    def test_auto_progress_resets_on_reject(self):
        with tempfile.TemporaryDirectory() as d:
            ap = AutopilotEngine(run_dir=d, mode="ai_suggest")
            ai_engine = MagicMock()
            ap.set_ai_engine(ai_engine)
            ap._auto_progress_enabled = True
            ap._auto_progress_threshold = 3

            from hotcb.server.autopilot import AutopilotAction
            # Accept 2
            for i in range(2):
                action = AutopilotAction(
                    action_id=f"act_{i}",
                    rule_id="ai:set_lr",
                    step=i * 10,
                    wall_time=0.0,
                    condition_met="test",
                    proposed_action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
                    confidence="medium",
                    status="proposed",
                )
                ap._history.append(action)
                ap.accept_action(f"act_{i}")

            assert ap._accepted_suggestion_streak == 2

            # Reject one
            action = AutopilotAction(
                action_id="act_reject",
                rule_id="ai:set_lr",
                step=30,
                wall_time=0.0,
                condition_met="test",
                proposed_action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
                confidence="medium",
                status="proposed",
            )
            ap._history.append(action)
            ap.reject_action("act_reject")

            assert ap._accepted_suggestion_streak == 0
            assert ap.mode == "ai_suggest"

    def test_auto_progress_does_not_downgrade(self):
        with tempfile.TemporaryDirectory() as d:
            ap = AutopilotEngine(run_dir=d, mode="ai_auto")
            ai_engine = MagicMock()
            ap.set_ai_engine(ai_engine)
            ap._auto_progress_enabled = True
            ap._auto_progress_threshold = 3

            # Reject shouldn't downgrade from ai_auto
            from hotcb.server.autopilot import AutopilotAction
            action = AutopilotAction(
                action_id="act_0",
                rule_id="ai:set_lr",
                step=0,
                wall_time=0.0,
                condition_met="test",
                proposed_action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
                confidence="medium",
                status="proposed",
            )
            ap._history.append(action)
            ap.reject_action("act_0")

            assert ap.mode == "ai_auto"
