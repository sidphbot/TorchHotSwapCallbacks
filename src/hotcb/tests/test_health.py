"""Tests for hotcb.health — TrainingHealthState and compute_health_state."""
import math
import time

import pytest

from hotcb.health import TrainingHealthState, compute_health_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(metric_name, values, start_step=0):
    """Build a metric_history dict with one metric."""
    return {metric_name: [{"step": start_step + i, "value": v} for i, v in enumerate(values)]}


def _merge(*histories):
    """Merge multiple metric_history dicts."""
    merged = {}
    for h in histories:
        merged.update(h)
    return merged


# ---------------------------------------------------------------------------
# Basic construction & empty input
# ---------------------------------------------------------------------------


class TestHealthStateBasic:
    def test_health_state_from_empty_metrics(self):
        state = compute_health_state({})
        assert isinstance(state, TrainingHealthState)
        assert state.labels == []
        assert state.numeric_stability["nan_count"] == 0
        assert state.numeric_stability["inf_count"] == 0
        assert state.numeric_stability["spike_count"] == 0
        assert state.step == 0

    def test_health_state_from_stable_training(self):
        # Steadily decreasing loss
        values = [1.0 - 0.005 * i for i in range(200)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "stable-improving" in state.labels
        assert "numerically-unsafe" not in state.labels

    def test_health_state_from_diverging_training(self):
        # Loss going up sharply
        values = [0.5 + 0.1 * i for i in range(100)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        # Should NOT have stable-improving
        assert "stable-improving" not in state.labels

    def test_health_state_from_plateau(self):
        # Loss flat for 100 steps
        values = [0.5] * 120
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "stable-plateaued" in state.labels

    def test_health_state_from_oscillating_loss(self):
        # Loss swinging wildly
        values = [1.0 + 0.5 * ((-1) ** i) for i in range(100)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "oscillatory" in state.labels


# ---------------------------------------------------------------------------
# Numeric stability
# ---------------------------------------------------------------------------


class TestNumericStability:
    def test_health_state_nan_detection(self):
        values = [0.5] * 10 + [float("nan")] + [0.5] * 10
        h = _make_history("train_loss", values)
        state = compute_health_state(h)
        assert state.numeric_stability["nan_count"] >= 1
        assert "numerically-unsafe" in state.labels

    def test_health_state_loss_spike_detection(self):
        # Normal values, then a big spike (>3x EMA)
        values = [0.5] * 50 + [5.0] + [0.5] * 10
        h = _make_history("train_loss", values)
        state = compute_health_state(h)
        assert state.numeric_stability["spike_count"] >= 1
        assert state.numeric_stability["last_spike_step"] is not None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_health_state_serialization(self):
        values = [1.0 - 0.005 * i for i in range(100)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h)
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "labels" in d
        assert "numeric_stability" in d
        assert "optimization_health" in d
        assert "multi_loss" in d
        assert "step" in d
        assert "timestamp" in d

        # Round-trip
        state2 = TrainingHealthState.from_dict(d)
        assert state2.labels == state.labels
        assert state2.step == state.step
        assert state2.numeric_stability == state.numeric_stability


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------


class TestGradNorm:
    def test_grad_norm_trend_rising(self):
        # Grad norm steadily increasing
        grad_values = [0.1 + 0.01 * i for i in range(100)]
        h = _make_history("grad_norm", grad_values)
        state = compute_health_state(h, window=100)
        assert state.optimization_health["grad_norm_trend"] in ("rising", "spike")

    def test_grad_norm_trend_stable(self):
        # Grad norm flat
        grad_values = [1.0] * 100
        h = _make_history("grad_norm", grad_values)
        state = compute_health_state(h, window=100)
        assert state.optimization_health["grad_norm_trend"] == "flat"

    def test_grad_norm_absent(self):
        # No grad_norm metric
        h = _make_history("train_loss", [0.5] * 50)
        state = compute_health_state(h)
        assert state.optimization_health["grad_norm_trend"] == "unknown"

    def test_clipping_rate_detection(self):
        # All grad norms near clip threshold (within 1%)
        clip_val = 1.0
        grad_values = [clip_val * 0.995] * 80 + [0.5] * 20
        h = _make_history("grad_norm", grad_values)
        state = compute_health_state(h, window=100, grad_clip_threshold=clip_val)
        assert state.optimization_health["clipping_rate"] > 0.5


# ---------------------------------------------------------------------------
# Multi-loss conflict
# ---------------------------------------------------------------------------


class TestMultiLoss:
    def test_conflict_score_no_multi_loss(self):
        h = _make_history("train_loss", [0.5] * 50)
        state = compute_health_state(h)
        assert state.multi_loss["conflict_score"] == 0.0
        assert state.multi_loss["per_loss_trends"] == {}

    def test_conflict_score_aligned_losses(self):
        # Two losses both decreasing
        h = _merge(
            _make_history("cls_loss", [1.0 - 0.005 * i for i in range(100)]),
            _make_history("recon_loss", [2.0 - 0.01 * i for i in range(100)]),
        )
        state = compute_health_state(h, window=100)
        assert state.multi_loss["conflict_score"] < 0.3

    def test_conflict_score_conflicting_losses(self):
        # One loss going down, the other going up
        h = _merge(
            _make_history("cls_loss", [1.0 - 0.01 * i for i in range(100)]),
            _make_history("recon_loss", [0.5 + 0.01 * i for i in range(100)]),
        )
        state = compute_health_state(h, window=100)
        assert state.multi_loss["conflict_score"] > 0.3

    def test_dominance_ratio_balanced(self):
        # Two losses with similar slopes
        h = _merge(
            _make_history("cls_loss", [1.0 - 0.005 * i for i in range(100)]),
            _make_history("recon_loss", [1.0 - 0.005 * i for i in range(100)]),
        )
        state = compute_health_state(h, window=100)
        ratios = state.multi_loss["dominance_ratios"]
        assert len(ratios) == 2
        # Both should be close to 0.5
        for v in ratios.values():
            assert 0.3 < v < 0.7

    def test_dominance_ratio_one_loss_dominates(self):
        # One loss changing much faster
        h = _merge(
            _make_history("cls_loss", [1.0 - 0.1 * i for i in range(100)]),
            _make_history("recon_loss", [1.0 - 0.001 * i for i in range(100)]),
        )
        state = compute_health_state(h, window=100)
        ratios = state.multi_loss["dominance_ratios"]
        assert ratios["cls_loss"] > 0.8

    def test_conflict_detection_requires_minimum_history(self):
        # Too few points — should not crash and should give 0 conflict
        h = _merge(
            _make_history("cls_loss", [1.0, 0.9]),
            _make_history("recon_loss", [0.5, 0.6]),
        )
        state = compute_health_state(h, window=100)
        assert state.multi_loss["conflict_score"] == 0.0


# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_label_stable_improving(self):
        values = [1.0 - 0.005 * i for i in range(200)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "stable-improving" in state.labels

    def test_label_stable_plateaued(self):
        values = [0.5] * 120
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "stable-plateaued" in state.labels

    def test_label_aux_conflicted(self):
        h = _merge(
            _make_history("cls_loss", [1.0 - 0.01 * i for i in range(100)]),
            _make_history("recon_loss", [0.5 + 0.01 * i for i in range(100)]),
        )
        state = compute_health_state(h, window=100)
        assert "aux-conflicted" in state.labels

    def test_label_numerically_unsafe(self):
        values = [0.5] * 10 + [float("nan")] + [0.5] * 10
        h = _make_history("train_loss", values)
        state = compute_health_state(h)
        assert "numerically-unsafe" in state.labels

    def test_label_oscillatory(self):
        values = [1.0 + 0.5 * ((-1) ** i) for i in range(100)]
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "oscillatory" in state.labels

    def test_label_converged_likely(self):
        # Flat loss + flat val_loss + low grad_norm
        h = _merge(
            _make_history("train_loss", [0.01] * 120),
            _make_history("val_loss", [0.012] * 120),
            _make_history("grad_norm", [0.001] * 120),
        )
        state = compute_health_state(h, window=100)
        assert "converged-likely" in state.labels

    def test_label_collapse_risk(self):
        # Loss going to near-zero — potential collapse
        values = [0.5 - 0.005 * i for i in range(100)]
        # Make last values very close to 0
        values[-10:] = [1e-8] * 10
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "collapse-risk" in state.labels

    def test_multiple_labels_simultaneously(self):
        # NaN + oscillating
        values = [1.0 + 0.5 * ((-1) ** i) for i in range(100)]
        values[50] = float("nan")
        h = _make_history("train_loss", values)
        state = compute_health_state(h, window=100)
        assert "numerically-unsafe" in state.labels
        assert "oscillatory" in state.labels
