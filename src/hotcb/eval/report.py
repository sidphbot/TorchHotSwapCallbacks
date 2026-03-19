"""Evaluation report — compare results across conditions."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .harness import EvalResult


class EvalReport:
    """Compare evaluation results and generate reports."""

    def __init__(self, results: List[EvalResult]):
        self._results = results
        # Group by demo
        self._by_demo: Dict[str, List[EvalResult]] = {}
        for r in results:
            self._by_demo.setdefault(r.demo, []).append(r)

    def baseline(self, demo: str) -> Optional[EvalResult]:
        """Find the baseline result for a demo type."""
        for r in self._by_demo.get(demo, []):
            if "baseline" in r.condition_name:
                return r
        return None

    def comparison_table(self, demo: str, metrics: Optional[List[str]] = None) -> str:
        """Generate a comparison table for a demo's conditions.

        Returns a formatted ASCII table.
        """
        results = self._by_demo.get(demo, [])
        if not results:
            return f"No results for demo '{demo}'"

        base = self.baseline(demo)

        # Auto-detect key metrics if not specified
        if metrics is None:
            metrics = _default_metrics(demo)

        # Build table
        header = ["Condition"] + metrics + ["Interventions", "Steps"]
        rows = []

        for r in results:
            row = [r.condition_name]
            for m in metrics:
                val = r.final_metrics.get(m)
                if val is not None:
                    cell = f"{val:.4f}"
                    # Delta vs baseline
                    if base and base is not r:
                        base_val = base.final_metrics.get(m)
                        if base_val is not None and base_val != 0:
                            delta_pct = (val - base_val) / abs(base_val) * 100
                            arrow = "↑" if delta_pct > 0 else "↓"
                            cell += f" ({arrow}{abs(delta_pct):.1f}%)"
                    row.append(cell)
                else:
                    row.append("—")
            row.append(str(r.intervention_count))
            row.append(str(r.total_steps))
            rows.append(row)

        return _format_table(header, rows)

    def research_summary(self) -> str:
        """Summarize research graph stats across all conditions."""
        lines = ["Research Graph Summary", "=" * 40]
        for r in self._results:
            stats = r.research_stats
            if not stats:
                continue
            hyp_statuses = stats.get("hypothesis_statuses", {})
            lines.append(
                f"  {r.condition_name}: "
                f"{stats.get('hypotheses', 0)} hyp "
                f"({hyp_statuses.get('confirmed', 0)}✓ "
                f"{hyp_statuses.get('refuted', 0)}✗ "
                f"{hyp_statuses.get('discovered', 0)}? "
                f"{hyp_statuses.get('shadow', 0)}⊘) "
                f"{stats.get('evidence', 0)} ev "
                f"{stats.get('observations', 0)} obs"
            )
            if stats.get("shadow_count", 0):
                lines.append(
                    f"    ⚠ {stats['shadow_count']} shadows, "
                    f"{stats.get('diluted_count', 0)} diluted"
                )
        return "\n".join(lines)

    def autopilot_summary(self) -> str:
        """Summarize autopilot actions across conditions."""
        lines = ["Autopilot Actions", "=" * 40]
        for r in self._results:
            if not r.autopilot_actions:
                continue
            # Count by rule_id
            rule_counts: Dict[str, int] = {}
            for a in r.autopilot_actions:
                rid = a.get("rule_id", "unknown")
                rule_counts[rid] = rule_counts.get(rid, 0) + 1
            rules_str = ", ".join(f"{k}×{v}" for k, v in rule_counts.items())
            lines.append(f"  {r.condition_name}: {len(r.autopilot_actions)} actions — {rules_str}")
        if len(lines) == 2:
            lines.append("  (no autopilot actions fired)")
        return "\n".join(lines)

    def verdict(self, demo: str) -> str:
        """Generate a human-readable verdict for a demo's evaluation."""
        base = self.baseline(demo)
        if base is None:
            return "No baseline available for comparison"

        lines = [f"Verdict for '{demo}' evaluation", "-" * 40]
        key_metric = _key_metric(demo)
        base_val = base.final_metrics.get(key_metric)
        if base_val is None:
            return f"Key metric '{key_metric}' not found in baseline"

        for r in self._by_demo.get(demo, []):
            if r is base:
                lines.append(f"  BASELINE ({r.condition_name}): {key_metric}={base_val:.4f}")
                continue

            val = r.final_metrics.get(key_metric)
            if val is None:
                lines.append(f"  {r.condition_name}: {key_metric}=N/A")
                continue

            # For loss metrics, lower is better
            improved = val < base_val if "loss" in key_metric else val > base_val
            delta_pct = (val - base_val) / abs(base_val) * 100 if base_val != 0 else 0

            if abs(delta_pct) < 5:
                status = "MATCH"
            elif improved:
                status = "BETTER"
            else:
                status = "WORSE"

            lines.append(
                f"  {status:6s} {r.condition_name}: "
                f"{key_metric}={val:.4f} ({delta_pct:+.1f}%) "
                f"[{r.intervention_count} interventions]"
            )

        return "\n".join(lines)

    def full_report(self) -> str:
        """Generate the complete evaluation report."""
        sections = []
        sections.append("=" * 60)
        sections.append("  hotcb Evaluation Report")
        sections.append("=" * 60)
        sections.append("")

        for demo in sorted(self._by_demo):
            sections.append(self.comparison_table(demo))
            sections.append("")
            sections.append(self.verdict(demo))
            sections.append("")

        sections.append(self.autopilot_summary())
        sections.append("")
        sections.append(self.research_summary())

        return "\n".join(sections)

    def to_json(self, path: str):
        """Export full results as JSON."""
        data = {
            "results": [r.to_dict() for r in self._results],
        }
        # Strip verbose metric history
        for d in data["results"]:
            d["metric_history_length"] = len(d.pop("metric_history", []))
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _default_metrics(demo: str) -> List[str]:
    if demo == "golden":
        return ["train_loss", "val_loss", "task_a_loss", "task_b_loss", "accuracy", "recon_score"]
    elif demo == "finetune":
        return ["train_loss", "val_loss", "accuracy", "grad_norm"]
    elif demo == "simple":
        return ["train_loss", "val_loss", "accuracy"]
    return ["train_loss", "val_loss"]


def _key_metric(demo: str) -> str:
    return "val_loss"


def _format_table(header: List[str], rows: List[List[str]]) -> str:
    """Simple ASCII table formatter."""
    all_rows = [header] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]

    def fmt_row(row):
        return "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    lines = [fmt_row(header)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)
