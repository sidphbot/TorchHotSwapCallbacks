"""Effect tracking for mutations — monitors metric impact after parameter changes.

Records baseline metrics at mutation time, then checks for improvement/degradation
after a configurable cooldown period.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("hotcb.tracking")


@dataclass
class PendingEffect:
    """A mutation whose metric impact is still being observed."""
    param_key: str
    step: int
    old_value: Any
    new_value: Any
    baseline_metrics: Dict[str, float]


@dataclass
class CompletedEffect:
    """A mutation whose metric impact has been measured."""
    param_key: str
    step: int
    old_value: Any
    new_value: Any
    baseline_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    delta: Dict[str, float]
    outcome: str  # "improved", "degraded", "neutral", "timeout"


class EffectTracker:
    """Tracks the metric impact of parameter mutations.

    Parameters
    ----------
    key_metric : str
        The primary metric to evaluate improvement/degradation.
    cooldown_steps : int
        Number of steps to wait after mutation before evaluating.
    timeout_steps : int
        Max steps to wait for the key metric to appear. Default 100.
    rollback_threshold : float or None
        Absolute increase in key_metric that triggers a rollback suggestion.
        Set to None to disable auto-rollback.
    improvement_threshold : float
        Minimum absolute decrease to classify as "improved". Default 0.01.
    """

    def __init__(
        self,
        key_metric: str = "val_loss",
        cooldown_steps: int = 50,
        timeout_steps: int = 100,
        rollback_threshold: Optional[float] = 0.1,
        improvement_threshold: float = 0.01,
    ) -> None:
        self.key_metric = key_metric
        self.cooldown_steps = cooldown_steps
        self.timeout_steps = timeout_steps
        self.rollback_threshold = rollback_threshold
        self.improvement_threshold = improvement_threshold

        self._pending: List[PendingEffect] = []
        self._completed: List[CompletedEffect] = []
        self._current_step: int = 0

    def on_mutation(
        self,
        step: int,
        param_key: str,
        old_value: Any,
        new_value: Any,
        metric_snapshot: Dict[str, float],
    ) -> None:
        """Record baseline metrics at mutation time."""
        effect = PendingEffect(
            param_key=param_key,
            step=step,
            old_value=old_value,
            new_value=new_value,
            baseline_metrics=dict(metric_snapshot),
        )
        self._pending.append(effect)
        log.debug("Effect tracked: %s changed at step %d", param_key, step)

    def on_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Check pending effects and complete those past cooldown."""
        self._current_step = step
        still_pending: List[PendingEffect] = []

        for effect in self._pending:
            elapsed = step - effect.step

            # Check timeout
            if elapsed >= self.timeout_steps:
                self._completed.append(CompletedEffect(
                    param_key=effect.param_key,
                    step=effect.step,
                    old_value=effect.old_value,
                    new_value=effect.new_value,
                    baseline_metrics=effect.baseline_metrics,
                    final_metrics=dict(metrics),
                    delta={},
                    outcome="timeout",
                ))
                log.debug("Effect timed out: %s after %d steps", effect.param_key, elapsed)
                continue

            # Check if cooldown has elapsed and key metric is present
            if elapsed >= self.cooldown_steps and self.key_metric in metrics:
                baseline_val = effect.baseline_metrics.get(self.key_metric)
                current_val = metrics.get(self.key_metric)

                if baseline_val is not None and current_val is not None:
                    delta_val = current_val - baseline_val
                    delta = {self.key_metric: delta_val}

                    # Classify outcome (for loss-like metrics, lower is better)
                    if delta_val < -self.improvement_threshold:
                        outcome = "improved"
                    elif delta_val > self.improvement_threshold:
                        outcome = "degraded"
                    else:
                        outcome = "neutral"

                    self._completed.append(CompletedEffect(
                        param_key=effect.param_key,
                        step=effect.step,
                        old_value=effect.old_value,
                        new_value=effect.new_value,
                        baseline_metrics=effect.baseline_metrics,
                        final_metrics=dict(metrics),
                        delta=delta,
                        outcome=outcome,
                    ))
                    log.debug(
                        "Effect completed: %s outcome=%s delta=%s",
                        effect.param_key, outcome, delta_val,
                    )
                    continue

            still_pending.append(effect)

        self._pending = still_pending

    def get_pending(self) -> List[PendingEffect]:
        """Return list of pending (not yet evaluated) effects."""
        return list(self._pending)

    def get_completed(self) -> List[CompletedEffect]:
        """Return list of completed (evaluated) effects."""
        return list(self._completed)

    def check_rollback_triggers(self) -> Optional[str]:
        """Check if any completed effect warrants a rollback.

        Returns the param_key to rollback, or None.
        """
        if self.rollback_threshold is None:
            return None

        for effect in self._completed:
            if effect.outcome != "degraded":
                continue
            key_delta = effect.delta.get(self.key_metric, 0.0)
            if key_delta > self.rollback_threshold:
                log.info(
                    "Rollback trigger: %s degraded %s by %.4f (threshold %.4f)",
                    effect.param_key, self.key_metric, key_delta, self.rollback_threshold,
                )
                return effect.param_key

        return None
