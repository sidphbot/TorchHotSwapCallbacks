#!/usr/bin/env python3
"""Pre-train the research outcome predictor model.

Usage:
    python scripts/pretrain_research_model.py [--epochs 100] [--include-scenarios]

Generates: src/hotcb/research/learner/pretrained/outcome_predictor.pt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Pre-train research outcome predictor")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--include-scenarios", action="store_true")
    parser.add_argument("--eval", action="store_true",
                        help="Run eval harness conditions for real intervention data")
    parser.add_argument("--eval-steps", type=int, default=300,
                        help="Steps per eval run (higher = better data, slower)")
    parser.add_argument("--n-random", type=int, default=20,
                        help="Number of random conditions for eval data")
    args = parser.parse_args()

    from hotcb.research.learner.pretrain import pretrain

    result = pretrain(
        epochs=args.epochs,
        include_scenarios=args.include_scenarios,
        include_eval=args.eval,
        eval_steps=args.eval_steps,
        n_random=args.n_random,
    )
    print(f"Pre-training complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
