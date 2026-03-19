"""Continuation evaluator — compare branches and select winners."""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import BranchResult, BranchStatus, RoutineResult


class ContinuationEvaluator:
    """Compare continuation branch results against base run."""

    def __init__(self, result: RoutineResult):
        self._result = result

    @property
    def winners(self) -> List[BranchResult]:
        return [b for b in self._result.branches if b.status == BranchStatus.WON]

    @property
    def failed(self) -> List[BranchResult]:
        return [b for b in self._result.branches
                if b.status in (BranchStatus.UNSTABLE, BranchStatus.RESUME_FAILED,
                                BranchStatus.REJECTED_GUARDRAIL)]

    def rank_by_metric(self, metric: Optional[str] = None) -> List[BranchResult]:
        """Rank all completed branches by a metric (default: primary)."""
        metric = metric or self._result.config.objective.primary_metric
        mode = self._result.config.objective.mode

        completed = [b for b in self._result.branches
                     if b.status not in (BranchStatus.PENDING, BranchStatus.RUNNING,
                                         BranchStatus.RESUME_FAILED)]

        def key(b: BranchResult):
            val = b.best_metrics.get(metric, b.final_metrics.get(metric))
            if val is None:
                return float("-inf") if mode == "max" else float("inf")
            return val

        return sorted(completed, key=key, reverse=(mode == "max"))

    def improvement_summary(self) -> Dict[str, Dict]:
        """Per-branch summary of improvement vs base."""
        summaries = {}
        obj = self._result.config.objective
        base_val = self._result.config.base_best_metrics.get(
            obj.primary_metric,
            self._result.config.base_final_metrics.get(obj.primary_metric),
        )

        for b in self._result.branches:
            branch_val = b.best_metrics.get(obj.primary_metric)
            summaries[b.branch_id] = {
                "recipe": b.recipe_name,
                "resume_mode": b.resume_mode,
                "anchor": b.anchor_id,
                "status": b.status.value,
                "base_value": base_val,
                "branch_value": branch_val,
                "delta": b.primary_delta,
                "delta_pct": b.primary_delta_pct,
                "steps": b.total_steps,
                "elapsed_s": b.elapsed_seconds,
            }
        return summaries

    def recipe_ranking(self) -> List[Dict]:
        """Rank recipes by average improvement across anchors/modes."""
        recipe_deltas: Dict[str, List[float]] = {}
        for b in self._result.branches:
            if b.status in (BranchStatus.PENDING, BranchStatus.RUNNING,
                            BranchStatus.RESUME_FAILED):
                continue
            recipe_deltas.setdefault(b.recipe_name, []).append(b.primary_delta)

        mode = self._result.config.objective.mode
        rankings = []
        for name, deltas in recipe_deltas.items():
            avg = sum(deltas) / len(deltas)
            best = max(deltas) if mode == "max" else min(deltas)
            rankings.append({
                "recipe": name,
                "avg_delta": round(avg, 6),
                "best_delta": round(best, 6),
                "n_branches": len(deltas),
                "n_won": sum(1 for d in deltas if (d > 0 if mode == "max" else d < 0)),
            })

        return sorted(rankings, key=lambda r: r["avg_delta"],
                       reverse=(mode == "max"))
