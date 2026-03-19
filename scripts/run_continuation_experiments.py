#!/usr/bin/env python3
"""Run continuation experiments from converged baselines.

Usage:
    python scripts/run_continuation_experiments.py --task mnist --baseline runs/mnist_baseline
    python scripts/run_continuation_experiments.py --task cifar10 --baseline runs/cifar10_baseline
    python scripts/run_continuation_experiments.py --task all  # runs both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hotcb.routines.continuation import (
    AnchorSelector,
    BudgetSpec,
    ContinuationConfig,
    ContinuationLauncher,
    ContinuationPlanner,
    ContinuationReport,
    GuardrailSpec,
    ObjectiveSpec,
    ResumeMode,
)
from hotcb.routines.continuation.planner import DEFAULT_RECIPES


def run_continuation(task: str, baseline_dir: str, output_dir: str) -> None:
    """Run full continuation experiment for a task."""
    print(f"\n{'='*70}")
    print(f"  Continuation Tuning: {task.upper()}")
    print(f"  Baseline: {baseline_dir}")
    print(f"{'='*70}\n")

    # Read base metrics
    best_path = os.path.join(baseline_dir, "best_metrics.json")
    base_best = {}
    if os.path.exists(best_path):
        with open(best_path) as f:
            base_best = json.load(f)
    print(f"Base best metrics: {json.dumps(base_best, indent=2)}")

    # Read final metrics
    metrics_path = os.path.join(baseline_dir, "hotcb.metrics.jsonl")
    base_final = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    base_final = rec.get("metrics", rec)
                    break
                except json.JSONDecodeError:
                    continue
    print(f"Base final metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in base_final.items()}, indent=2)}")

    # Select anchors
    selector = AnchorSelector(baseline_dir)
    anchors = selector.select_auto(max_anchors=2, primary_metric="val_accuracy", mode="max")
    if not anchors:
        print(f"ERROR: No checkpoints found in {baseline_dir}/checkpoints/")
        return

    print(f"\nAnchors ({len(anchors)}):")
    for a in anchors:
        print(f"  {a.anchor_id}: step={a.step}, reason={a.reason.value}")
        print(f"    metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in a.metrics_at_anchor.items()})}")

    # Configure: all 6 default recipes × 2 resume modes × 2 anchors
    config = ContinuationConfig(
        base_run_dir=baseline_dir,
        base_final_metrics=base_final,
        base_best_metrics=base_best,
        task=task,
        objective=ObjectiveSpec(
            primary_metric="val_accuracy",
            mode="max",
            min_improvement_delta=0.001,  # 0.1% minimum improvement
        ),
        guardrails=GuardrailSpec(
            forbid_nan=True,
            secondary_metrics={
                "val_loss": {"mode": "min", "max_regression": 0.05},
            },
        ),
        budget=BudgetSpec(
            max_anchors=2,
            branches_per_anchor=8,  # 6 recipes × 1 mode + 2 extra mode variants
            max_extra_steps=500,
            eval_interval=100,
            early_stop_patience=4,
        ),
        anchors=anchors,
        recipes=list(DEFAULT_RECIPES),
        resume_modes=[ResumeMode.FULL_RESUME],
        output_dir=output_dir,
    )

    print(f"\nPlanning branches...")
    planner = ContinuationPlanner(config)
    branches = planner.plan()
    print(f"Planned {len(branches)} branches:")
    for b in branches:
        print(f"  {b.branch_id[:50]}...")

    print(f"\nRunning continuation routine...\n")
    t0 = time.time()
    launcher = ContinuationLauncher(config, output_dir)
    result = launcher.run_all(branches)
    elapsed = time.time() - t0

    # Report
    report = ContinuationReport(result)
    full_report = report.full_report()
    print(full_report)

    report_path = report.save(output_dir)
    print(f"\nReport saved: {report_path}")
    print(f"Total time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run continuation experiments")
    parser.add_argument("--task", default="all", choices=["mnist", "cifar10", "all"])
    parser.add_argument("--baseline-mnist", default="runs/mnist_baseline")
    parser.add_argument("--baseline-cifar10", default="runs/cifar10_baseline")
    args = parser.parse_args()

    if args.task in ("mnist", "all"):
        if os.path.isdir(args.baseline_mnist):
            run_continuation("mnist", args.baseline_mnist, "runs/mnist_continuation")
        else:
            print(f"MNIST baseline not found at {args.baseline_mnist}")

    if args.task in ("cifar10", "all"):
        if os.path.isdir(args.baseline_cifar10):
            run_continuation("cifar10", args.baseline_cifar10, "runs/cifar10_continuation")
        else:
            print(f"CIFAR-10 baseline not found at {args.baseline_cifar10}")


if __name__ == "__main__":
    main()
