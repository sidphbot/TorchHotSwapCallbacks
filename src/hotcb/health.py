"""
hotcb.health — Training health assessment.

Provides ``TrainingHealthState`` and ``compute_health_state()`` to derive
high-level health labels and numeric diagnostics from raw metric history.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class TrainingHealthState:
    """Snapshot of training health derived from metric history."""

    labels: List[str] = field(default_factory=list)
    numeric_stability: dict = field(default_factory=dict)
    optimization_health: dict = field(default_factory=dict)
    multi_loss: dict = field(default_factory=dict)
    step: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "numeric_stability": dict(self.numeric_stability),
            "optimization_health": dict(self.optimization_health),
            "multi_loss": dict(self.multi_loss),
            "step": self.step,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingHealthState":
        return cls(
            labels=list(d.get("labels", [])),
            numeric_stability=dict(d.get("numeric_stability", {})),
            optimization_health=dict(d.get("optimization_health", {})),
            multi_loss=dict(d.get("multi_loss", {})),
            step=d.get("step", 0),
            timestamp=d.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _linear_slope(values: List[float]) -> float:
    """Compute linear regression slope over values indexed 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _classify_trend(slope: float, y_mean: float) -> str:
    """Classify a slope as a trend label relative to mean magnitude."""
    if abs(y_mean) < 1e-10:
        norm = slope
    else:
        norm = slope / abs(y_mean)
    if norm < -0.01:
        return "steep_down"
    elif norm < -0.002:
        return "steady_down"
    elif norm < -0.0002:
        return "slow_down"
    elif norm < 0.0002:
        return "flat"
    elif norm < 0.002:
        return "slow_up"
    elif norm < 0.01:
        return "rising"
    else:
        return "spike"


def _extract_values(records: List[dict]) -> List[float]:
    """Extract numeric values from [{step, value}] records, skipping NaN/Inf."""
    out: List[float] = []
    for r in records:
        v = r.get("value") if isinstance(r, dict) else r
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def _all_values(records: List[dict]) -> List[Optional[float]]:
    """Extract all values including NaN/Inf (as-is)."""
    out: List[Optional[float]] = []
    for r in records:
        v = r.get("value") if isinstance(r, dict) else r
        if isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Numeric stability analysis
# ---------------------------------------------------------------------------


def _compute_numeric_stability(
    metric_history: Dict[str, List[dict]],
) -> dict:
    """Detect NaN, Inf, and loss spikes across all metrics."""
    nan_count = 0
    inf_count = 0
    spike_count = 0
    last_spike_step: Optional[int] = None

    for name, records in metric_history.items():
        for rec in records:
            v = rec.get("value") if isinstance(rec, dict) else rec
            if isinstance(v, float):
                if math.isnan(v):
                    nan_count += 1
                elif math.isinf(v):
                    inf_count += 1

        # Spike detection: value > 3x EMA
        if "loss" in name.lower() or name in ("train_loss", "val_loss"):
            clean = _extract_values(records)
            if len(clean) >= 5:
                ema = clean[0]
                alpha = 0.1
                for idx, v in enumerate(clean[1:], start=1):
                    if v > 3 * max(ema, 1e-10):
                        spike_count += 1
                        # Find the step from the record
                        # idx in clean corresponds to skipped NaN so use original
                        # Just record latest spike step from original records
                        last_spike_step = records[min(idx, len(records) - 1)].get("step", idx)
                    ema = alpha * v + (1 - alpha) * ema

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "spike_count": spike_count,
        "last_spike_step": last_spike_step,
    }


# ---------------------------------------------------------------------------
# Gradient health
# ---------------------------------------------------------------------------


def _compute_optimization_health(
    metric_history: Dict[str, List[dict]],
    window: int,
    grad_clip_threshold: Optional[float] = None,
) -> dict:
    """Compute gradient and optimization health metrics."""
    # Find grad_norm metric
    grad_key = None
    for k in ("grad_norm", "grad_norm_total"):
        if k in metric_history:
            grad_key = k
            break

    if grad_key is None:
        return {
            "grad_norm_trend": "unknown",
            "clipping_rate": 0.0,
            "total_loss_trend": "unknown",
            "plateau_score": 0.0,
            "oscillation_score": 0.0,
        }

    grad_records = metric_history[grad_key]
    grad_vals = _extract_values(grad_records[-window:])

    # Trend
    if len(grad_vals) >= 5:
        slope = _linear_slope(grad_vals)
        y_mean = sum(grad_vals) / len(grad_vals)
        grad_trend = _classify_trend(slope, y_mean)
        # Simplify to coarse labels
        if grad_trend in ("slow_up", "rising", "spike"):
            grad_trend = "rising"
        elif grad_trend in ("slow_down", "steady_down", "steep_down"):
            grad_trend = "falling"
        else:
            grad_trend = "flat"
    else:
        grad_trend = "unknown"

    # Clipping rate
    clipping_rate = 0.0
    if grad_clip_threshold is not None and grad_clip_threshold > 0 and grad_vals:
        threshold_low = grad_clip_threshold * 0.99
        clipped = sum(1 for v in grad_vals if v >= threshold_low)
        clipping_rate = clipped / len(grad_vals)

    # Loss trend and health
    loss_key = None
    for k in ("train_loss", "loss"):
        if k in metric_history:
            loss_key = k
            break
    if loss_key is None:
        # Pick any *_loss key
        for k in metric_history:
            if "loss" in k.lower():
                loss_key = k
                break

    total_loss_trend = "unknown"
    plateau_score = 0.0
    oscillation_score = 0.0

    if loss_key:
        loss_vals = _extract_values(metric_history[loss_key][-window:])
        if len(loss_vals) >= 5:
            slope = _linear_slope(loss_vals)
            y_mean = sum(loss_vals) / len(loss_vals)
            total_loss_trend = _classify_trend(slope, y_mean)

            # Plateau score: how flat is the loss over the window
            std = math.sqrt(sum((v - y_mean) ** 2 for v in loss_vals) / len(loss_vals))
            cv = std / abs(y_mean) if abs(y_mean) > 1e-10 else 0.0
            # plateau_score: 1.0 = perfectly flat, 0.0 = high variance
            plateau_score = max(0.0, 1.0 - cv * 10)

            # Oscillation score: variance relative to mean
            oscillation_score = min(1.0, cv * 5)

    return {
        "grad_norm_trend": grad_trend,
        "clipping_rate": round(clipping_rate, 4),
        "total_loss_trend": total_loss_trend,
        "plateau_score": round(plateau_score, 4),
        "oscillation_score": round(oscillation_score, 4),
    }


# ---------------------------------------------------------------------------
# Multi-loss conflict
# ---------------------------------------------------------------------------


def _compute_multi_loss(
    metric_history: Dict[str, List[dict]],
    window: int,
) -> dict:
    """Detect multi-loss conflict and dominance."""
    # Find all *_loss metrics (excluding aggregate train_loss/val_loss)
    loss_keys = [
        k for k in metric_history
        if k.endswith("_loss")
        and k not in ("train_loss", "val_loss", "total_loss")
    ]

    if len(loss_keys) < 2:
        return {
            "per_loss_trends": {},
            "dominance_ratios": {},
            "conflict_score": 0.0,
        }

    # Compute slopes for each loss
    slopes: Dict[str, float] = {}
    per_loss_trends: Dict[str, str] = {}
    min_len = float("inf")

    for k in loss_keys:
        vals = _extract_values(metric_history[k][-window:])
        if len(vals) < 5:
            # Not enough history
            return {
                "per_loss_trends": {},
                "dominance_ratios": {},
                "conflict_score": 0.0,
            }
        min_len = min(min_len, len(vals))
        slope = _linear_slope(vals)
        y_mean = sum(vals) / len(vals)
        slopes[k] = slope
        per_loss_trends[k] = _classify_trend(slope, y_mean)

    # Dominance ratios: proportion of total |slope|
    abs_slopes = {k: abs(s) for k, s in slopes.items()}
    total_abs = sum(abs_slopes.values())
    if total_abs < 1e-12:
        dominance_ratios = {k: 1.0 / len(loss_keys) for k in loss_keys}
    else:
        dominance_ratios = {k: v / total_abs for k, v in abs_slopes.items()}

    # Conflict score via detrended cross-correlation
    # Simple approach: for each pair, check if slopes have opposite signs
    conflict_score = 0.0
    pairs = 0
    for i, k1 in enumerate(loss_keys):
        for k2 in loss_keys[i + 1:]:
            s1, s2 = slopes[k1], slopes[k2]
            # If slopes have opposite signs, there's conflict
            if abs(s1) > 1e-10 and abs(s2) > 1e-10:
                # Cosine similarity of slopes (scalar version): sign agreement
                cos_sim = (s1 * s2) / (abs(s1) * abs(s2))
                # cos_sim = 1 means aligned, -1 means opposed
                # Convert to conflict: 0 = aligned, 1 = opposed
                pair_conflict = (1 - cos_sim) / 2.0

                # Also compute detrended cross-correlation for more accuracy
                vals1 = _extract_values(metric_history[k1][-window:])
                vals2 = _extract_values(metric_history[k2][-window:])
                n = min(len(vals1), len(vals2))
                if n >= 10:
                    # Detrend
                    dt1 = _detrend(vals1[-n:])
                    dt2 = _detrend(vals2[-n:])
                    # If detrended signals have near-zero variance, fall back
                    # to slope-based conflict (the signals are near-linear)
                    dt1_var = sum(v ** 2 for v in dt1) / n
                    dt2_var = sum(v ** 2 for v in dt2) / n
                    if dt1_var > 1e-12 and dt2_var > 1e-12:
                        corr = _cross_correlation(dt1, dt2)
                        pair_conflict = (1 - corr) / 2.0
                    # else: keep slope-based pair_conflict from above

                conflict_score += pair_conflict
                pairs += 1

    if pairs > 0:
        conflict_score /= pairs

    return {
        "per_loss_trends": per_loss_trends,
        "dominance_ratios": {k: round(v, 4) for k, v in dominance_ratios.items()},
        "conflict_score": round(conflict_score, 4),
    }


def _detrend(values: List[float]) -> List[float]:
    """Remove linear trend from values."""
    n = len(values)
    if n < 2:
        return values
    slope = _linear_slope(values)
    y_mean = sum(values) / n
    x_mean = (n - 1) / 2.0
    return [v - (slope * (i - x_mean) + y_mean) for i, v in enumerate(values)]


def _cross_correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation between two equal-length sequences."""
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
    da = math.sqrt(sum((ai - ma) ** 2 for ai in a))
    db = math.sqrt(sum((bi - mb) ** 2 for bi in b))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return num / (da * db)


# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------


def _classify_labels(
    metric_history: Dict[str, List[dict]],
    numeric_stability: dict,
    optimization_health: dict,
    multi_loss: dict,
    window: int,
) -> List[str]:
    """Evaluate conditions and return derived state labels."""
    labels: List[str] = []

    # numerically-unsafe: NaN/Inf detected or loss spike
    if (
        numeric_stability.get("nan_count", 0) > 0
        or numeric_stability.get("inf_count", 0) > 0
        or numeric_stability.get("spike_count", 0) > 0
    ):
        labels.append("numerically-unsafe")

    # Find a primary loss metric
    loss_key = None
    for k in ("train_loss", "loss"):
        if k in metric_history:
            loss_key = k
            break
    if loss_key is None:
        for k in metric_history:
            if "loss" in k.lower():
                loss_key = k
                break

    # collapse-risk: loss going to near-zero or feature variance dropping
    if loss_key:
        loss_vals = _extract_values(metric_history[loss_key])
        if len(loss_vals) >= 10:
            recent = loss_vals[-10:]
            if all(abs(v) < 1e-6 for v in recent):
                labels.append("collapse-risk")

    # aux-conflicted: multi-loss with conflict_score > 0.3
    if multi_loss.get("conflict_score", 0) > 0.3:
        labels.append("aux-conflicted")

    # oscillatory: loss variance high relative to mean (after detrending)
    if loss_key:
        loss_vals = _extract_values(metric_history[loss_key][-window:])
        if len(loss_vals) >= 10:
            y_mean = sum(loss_vals) / len(loss_vals)
            # Detrend to avoid flagging steady trends as oscillatory
            detrended = _detrend(loss_vals)
            dt_std = math.sqrt(sum(v ** 2 for v in detrended) / len(detrended))
            dt_cv = dt_std / abs(y_mean) if abs(y_mean) > 1e-10 else 0.0
            if dt_cv > 0.1:
                labels.append("oscillatory")

    # stable-plateaued: loss flat for >50 steps
    if loss_key:
        loss_vals = _extract_values(metric_history[loss_key])
        if len(loss_vals) >= 50:
            recent_50 = loss_vals[-50:]
            y_mean = sum(recent_50) / len(recent_50)
            std = math.sqrt(sum((v - y_mean) ** 2 for v in recent_50) / len(recent_50))
            cv = std / abs(y_mean) if abs(y_mean) > 1e-10 else 0.0
            slope = _linear_slope(recent_50)
            norm_slope = slope / abs(y_mean) if abs(y_mean) > 1e-10 else slope
            if cv < 0.01 and abs(norm_slope) < 0.0002:
                labels.append("stable-plateaued")

    # stable-improving: loss decreasing steadily (low residual noise after detrending)
    if loss_key:
        loss_vals = _extract_values(metric_history[loss_key][-window:])
        if len(loss_vals) >= 20:
            slope = _linear_slope(loss_vals)
            y_mean = sum(loss_vals) / len(loss_vals)
            norm_slope = slope / abs(y_mean) if abs(y_mean) > 1e-10 else slope
            # Use detrended CV to check stability (removes the trend itself)
            detrended = _detrend(loss_vals)
            dt_std = math.sqrt(sum(v ** 2 for v in detrended) / len(detrended))
            dt_cv = dt_std / abs(y_mean) if abs(y_mean) > 1e-10 else 0.0
            if norm_slope < -0.0002 and dt_cv < 0.1:
                labels.append("stable-improving")

    # converged-likely: loss flat + validation stable + low grad norm
    if loss_key and "val_loss" in metric_history:
        loss_vals = _extract_values(metric_history[loss_key][-window:])
        val_vals = _extract_values(metric_history["val_loss"][-window:])
        grad_key = None
        for k in ("grad_norm", "grad_norm_total"):
            if k in metric_history:
                grad_key = k
                break
        if len(loss_vals) >= 50 and len(val_vals) >= 50:
            # Check flatness
            loss_mean = sum(loss_vals[-50:]) / 50
            loss_std = math.sqrt(sum((v - loss_mean) ** 2 for v in loss_vals[-50:]) / 50)
            loss_cv = loss_std / abs(loss_mean) if abs(loss_mean) > 1e-10 else 0.0

            val_mean = sum(val_vals[-50:]) / 50
            val_std = math.sqrt(sum((v - val_mean) ** 2 for v in val_vals[-50:]) / 50)
            val_cv = val_std / abs(val_mean) if abs(val_mean) > 1e-10 else 0.0

            grad_low = True
            if grad_key:
                grad_vals = _extract_values(metric_history[grad_key][-window:])
                if grad_vals:
                    grad_mean = sum(grad_vals) / len(grad_vals)
                    # "low" grad norm — relative to typical values
                    grad_low = grad_mean < 0.01

            if loss_cv < 0.01 and val_cv < 0.02 and grad_low:
                labels.append("converged-likely")

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_health_state(
    metric_history: Dict[str, List[dict]],
    window: int = 100,
    grad_clip_threshold: Optional[float] = None,
) -> TrainingHealthState:
    """
    Compute training health state from metric history.

    Parameters
    ----------
    metric_history : dict
        ``{metric_name: [{step, value}, ...]}`` — same format as dashboard metricsData.
    window : int
        Number of recent steps to analyze.
    grad_clip_threshold : float | None
        If provided, used to compute gradient clipping rate.

    Returns
    -------
    TrainingHealthState
    """
    if not metric_history:
        return TrainingHealthState(
            labels=[],
            numeric_stability={"nan_count": 0, "inf_count": 0, "spike_count": 0, "last_spike_step": None},
            optimization_health={"grad_norm_trend": "unknown", "clipping_rate": 0.0,
                                 "total_loss_trend": "unknown", "plateau_score": 0.0,
                                 "oscillation_score": 0.0},
            multi_loss={"per_loss_trends": {}, "dominance_ratios": {}, "conflict_score": 0.0},
            step=0,
            timestamp=time.time(),
        )

    # Determine latest step
    max_step = 0
    for records in metric_history.values():
        if records:
            last = records[-1]
            s = last.get("step", 0) if isinstance(last, dict) else 0
            max_step = max(max_step, s)

    numeric_stability = _compute_numeric_stability(metric_history)
    optimization_health = _compute_optimization_health(
        metric_history, window, grad_clip_threshold
    )
    multi_loss = _compute_multi_loss(metric_history, window)

    labels = _classify_labels(
        metric_history, numeric_stability, optimization_health, multi_loss, window
    )

    return TrainingHealthState(
        labels=labels,
        numeric_stability=numeric_stability,
        optimization_health=optimization_health,
        multi_loss=multi_loss,
        step=max_step,
        timestamp=time.time(),
    )
