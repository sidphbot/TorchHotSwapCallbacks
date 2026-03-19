"""Continuation report — human-readable output of routine results."""

from __future__ import annotations

import json
import os
from typing import List, Optional

from .models import BranchResult, BranchStatus, RoutineResult
from .evaluator import ContinuationEvaluator


_WARNING = (
    "This routine optimizes against the validation set. Keep a clean untouched "
    "test set for final model selection and reporting. Repeated continuation "
    "search on the same validation set can overfit selection to validation behavior."
)


class ContinuationReport:
    """Generate reports from continuation routine results."""

    def __init__(self, result: RoutineResult):
        self._result = result
        self._eval = ContinuationEvaluator(result)

    def leaderboard(self) -> str:
        """ASCII leaderboard of all branches."""
        obj = self._result.config.objective
        metric = obj.primary_metric

        header = [
            "Rank", "Branch", "Recipe", "Resume", "Anchor",
            f"Best {metric}", "Delta", "Delta%", "Status", "Steps",
        ]
        rows = []
        ranked = self._eval.rank_by_metric()

        for i, b in enumerate(ranked, 1):
            val = b.best_metrics.get(metric, b.final_metrics.get(metric))
            rows.append([
                str(i),
                b.branch_id[:30],
                b.recipe_name,
                b.resume_mode,
                b.anchor_id[:20],
                f"{val:.6f}" if val is not None else "N/A",
                f"{b.primary_delta:+.6f}",
                f"{b.primary_delta_pct:+.2f}%",
                b.status.value,
                str(b.total_steps),
            ])

        return _format_table(header, rows)

    def recipe_ranking(self) -> str:
        """Show which recipes worked best on average."""
        rankings = self._eval.recipe_ranking()
        header = ["Recipe", "Avg Delta", "Best Delta", "Branches", "Won"]
        rows = []
        for r in rankings:
            rows.append([
                r["recipe"],
                f"{r['avg_delta']:+.6f}",
                f"{r['best_delta']:+.6f}",
                str(r["n_branches"]),
                str(r["n_won"]),
            ])
        return _format_table(header, rows)

    def recommendation(self) -> str:
        """One-line recommendation."""
        winner = self._result.winner
        if winner is None:
            return "Recommendation: KEEP BASE RUN — no branch improved on the primary metric."

        obj = self._result.config.objective
        metric = obj.primary_metric
        base_val = self._result.config.base_best_metrics.get(
            metric, self._result.config.base_final_metrics.get(metric))
        branch_val = winner.best_metrics.get(metric)

        return (
            f"Recommendation: PROMOTE branch `{winner.branch_id}` "
            f"(anchor={winner.anchor_id}, recipe={winner.recipe_name}, "
            f"resume={winner.resume_mode}). "
            f"{metric}: {base_val} -> {branch_val} "
            f"({winner.primary_delta_pct:+.2f}%)."
        )

    def full_report(self) -> str:
        """Complete continuation report."""
        sections = [
            "=" * 70,
            "  hotcb Continuation Tuning Report",
            "=" * 70,
            "",
            f"Base run: {self._result.config.base_run_dir}",
            f"Task: {self._result.config.task}",
            f"Objective: {self._result.config.objective.mode} {self._result.config.objective.primary_metric}",
            f"Total branches: {len(self._result.branches)}",
            f"Total time: {self._result.elapsed_seconds:.1f}s",
            "",
            "--- Leaderboard ---",
            self.leaderboard(),
            "",
            "--- Recipe Rankings ---",
            self.recipe_ranking(),
            "",
            "--- Recommendation ---",
            self.recommendation(),
            "",
            f"WARNING: {_WARNING}",
        ]
        return "\n".join(sections)

    def save(self, path: Optional[str] = None) -> str:
        """Save full report + JSON data."""
        out_dir = path or self._result.config.output_dir or "."

        # Text report
        report_path = os.path.join(out_dir, "continuation_report.txt")
        with open(report_path, "w") as f:
            f.write(self.full_report())

        # JSON summary
        summary = {
            "task": self._result.config.task,
            "base_run_dir": self._result.config.base_run_dir,
            "objective": {
                "primary_metric": self._result.config.objective.primary_metric,
                "mode": self._result.config.objective.mode,
            },
            "total_branches": len(self._result.branches),
            "elapsed_seconds": self._result.elapsed_seconds,
            "winner": self._result.winner.branch_id if self._result.winner else None,
            "recommendation": self._result.recommendation,
            "branches": [],
        }
        for b in self._result.branches:
            summary["branches"].append({
                "branch_id": b.branch_id,
                "recipe": b.recipe_name,
                "resume_mode": b.resume_mode,
                "anchor": b.anchor_id,
                "status": b.status.value,
                "primary_delta": b.primary_delta,
                "primary_delta_pct": b.primary_delta_pct,
                "best_metrics": b.best_metrics,
                "final_metrics": b.final_metrics,
                "steps": b.total_steps,
                "elapsed_s": b.elapsed_seconds,
            })

        json_path = os.path.join(out_dir, "continuation_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Leaderboard JSON
        lb_path = os.path.join(out_dir, "leaderboard.json")
        with open(lb_path, "w") as f:
            json.dump(self._eval.improvement_summary(), f, indent=2, default=str)

        return report_path


def _format_table(header: List[str], rows: List[List[str]]) -> str:
    """Simple ASCII table formatter."""
    if not rows:
        return "(no data)"
    all_rows = [header] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]

    def fmt_row(row):
        return "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    lines = [fmt_row(header)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)
