"""Feature extraction for the NN learner."""

from __future__ import annotations

from typing import Any, Dict, List

# 13-dimensional InterventionFeature vector
FEATURE_NAMES = [
    # Context (6)
    "loss_value",
    "loss_trend_slope",
    "loss_volatility",
    "grad_norm",
    "step_fraction",
    "conflict_score",
    # Intervention (3)
    "param_type",       # encoded int: 0=float, 1=int, 2=bool, 3=choice, 4=log_float
    "relative_change",
    "absolute_change",
    # Health (4)
    "is_plateau",
    "is_diverging",
    "is_oscillating",
    "health_conflict_score",
]

FEATURE_DIM = len(FEATURE_NAMES)

_PARAM_TYPE_MAP = {
    "FLOAT": 0, "float": 0,
    "INT": 1, "int": 1,
    "BOOL": 2, "bool": 2,
    "CHOICE": 3, "choice": 3,
    "LOG_FLOAT": 4, "log_float": 4,
}


def extract_features(effect, context: Dict[str, Any] = None) -> List[float]:
    """Extract a 13-dim feature vector from a CompletedEffect + optional context."""
    ctx = context or {}
    baseline = getattr(effect, "baseline_metrics", {}) or {}
    final = getattr(effect, "final_metrics", {}) or {}

    old_val = _to_float(getattr(effect, "old_value", 0))
    new_val = _to_float(getattr(effect, "new_value", 0))
    abs_change = new_val - old_val
    rel_change = abs_change / max(abs(old_val), 1e-8)

    loss_val = baseline.get("train_loss", baseline.get("loss", 0.0))
    loss_final = final.get("train_loss", final.get("loss", 0.0))

    return [
        float(loss_val),
        float(ctx.get("loss_trend_slope", 0.0)),
        float(ctx.get("loss_volatility", 0.0)),
        float(baseline.get("grad_norm", ctx.get("grad_norm", 0.0))),
        float(ctx.get("step_fraction", 0.0)),
        float(ctx.get("conflict_score", 0.0)),
        float(_PARAM_TYPE_MAP.get(ctx.get("param_type", "float"), 0)),
        float(rel_change),
        float(abs_change),
        float(ctx.get("is_plateau", 0)),
        float(ctx.get("is_diverging", 0)),
        float(ctx.get("is_oscillating", 0)),
        float(ctx.get("health_conflict_score", ctx.get("conflict_score", 0.0))),
    ]


def features_from_intervention(intervention: dict, context: dict) -> List[float]:
    """Extract features from a proposed intervention + context (no effect yet)."""
    params = intervention.get("params", {})
    # Take the first param value as representative
    val = 0.0
    for v in params.values():
        val = _to_float(v)
        break

    return [
        float(context.get("loss_value", 0.0)),
        float(context.get("loss_trend_slope", 0.0)),
        float(context.get("loss_volatility", 0.0)),
        float(context.get("grad_norm", 0.0)),
        float(context.get("step_fraction", 0.0)),
        float(context.get("conflict_score", 0.0)),
        float(_PARAM_TYPE_MAP.get(context.get("param_type", "float"), 0)),
        float(val),  # relative_change (approximate)
        float(val),  # absolute_change (approximate)
        float(context.get("is_plateau", 0)),
        float(context.get("is_diverging", 0)),
        float(context.get("is_oscillating", 0)),
        float(context.get("health_conflict_score", context.get("conflict_score", 0.0))),
    ]


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
