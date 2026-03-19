"""Pre-training pipeline for the outcome predictor.

Data sources (in priority order):
1. **Eval harness runs** — real intervention→outcome data from golden/finetune demos
   with randomized initial conditions. Highest quality because it captures realistic
   dynamics (LR instability, divergence spikes, SWA smoothing).
2. **Synthetic heuristic samples** — random features + rule-based labels (baseline).
3. **Augmentation** — noise-perturbed copies for data diversity.

Usage::

    python scripts/pretrain_research_model.py          # synthetic only (~240 samples)
    python scripts/pretrain_research_model.py --eval    # run eval suite + synthetic (~800+ samples)
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

from .features import FEATURE_DIM
from .data import TrainingDataset
from .model import OutcomePredictor, _PRETRAINED_PATH


# ═══════════════════════════════════════════════════════════════════════
# Source 1: Eval harness runs
# ═══════════════════════════════════════════════════════════════════════

def collect_eval_samples(
    max_steps: int = 300,
    step_delay: float = 0.001,
    n_random_conditions: int = 20,
    seed: int = 42,
) -> List[Tuple[List[float], float]]:
    """Run eval harness conditions and extract intervention→outcome training data.

    For each eval run, we pair consecutive metric windows around mutations:
    - Before mutation: 10-step metric window → context features
    - After mutation: 20-step metric window → outcome label (improved/degraded)

    Returns list of (13-dim feature vector, binary label) tuples.
    """
    import copy
    import tempfile

    from hotcb.eval.conditions import EvalCondition, GOLDEN_CONDITIONS, FINETUNE_CONDITIONS
    from hotcb.eval.harness import EvalHarness

    rng = random.Random(seed)
    output_dir = tempfile.mkdtemp(prefix="hotcb_pretrain_eval_")
    harness = EvalHarness(output_dir=output_dir)

    # Use predefined conditions + random ones for diversity
    conditions = []

    # Predefined — run the core recovery conditions
    for name in [
        "golden_baseline", "golden_recipe",
        "golden_1bad_lr_auto", "golden_1bad_wd_auto", "golden_1bad_weight_auto",
        "golden_2bad_lr_weight_auto", "golden_2bad_wd_weight_auto",
        "golden_divergent_lr_auto", "golden_zero_wd_auto",
        "golden_reversed_weights_auto",
        "finetune_baseline", "finetune_recipe",
        "finetune_high_lr_auto",
    ]:
        for c in GOLDEN_CONDITIONS + FINETUNE_CONDITIONS:
            if c.name == name:
                cond = copy.deepcopy(c)
                cond.step_delay = step_delay
                cond.max_steps = max_steps
                conditions.append(cond)
                break

    # Randomized conditions — diverse failure modes for richer data
    for i in range(n_random_conditions):
        lr_mult = rng.choice([0.02, 0.05, 0.1, 0.5, 2.0, 5.0, 10.0, 50.0])
        wd_mult = rng.choice([0.0, 0.1, 1.0, 10.0, 100.0])
        wa = round(rng.uniform(0.05, 0.95), 2)
        wb = round(1.0 - wa, 2)

        conditions.append(EvalCondition(
            name=f"pretrain_random_{i:03d}",
            description=f"Random: lr×{lr_mult}, wd×{wd_mult}, wa={wa}",
            demo="golden",
            initial_overrides={
                "opt": {"lr": 1e-3 * lr_mult, "weight_decay": 1e-4 * wd_mult},
                "loss": {"weight_a": wa, "weight_b": wb},
            },
            use_recipe=rng.choice([True, False]),
            autopilot=rng.choice(["off", "auto"]),
            policy_packs=["stability_basics", "multi_loss_assist", "plateau_recovery"] if rng.random() > 0.3 else [],
            max_steps=max_steps,
            step_delay=step_delay,
            seed=seed + i,
            hypothesis_condition=f"random init lr×{lr_mult}, wd×{wd_mult}, wa={wa}",
            hypothesis_expected="varies",
        ))

    # Run all conditions
    samples: List[Tuple[List[float], float]] = []
    for cond in conditions:
        try:
            result = harness.run(cond)
            new_samples = _extract_samples_from_run(result)
            samples.extend(new_samples)
        except Exception:
            pass  # skip broken conditions

    # Cleanup
    try:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass

    return samples


def _extract_samples_from_run(result) -> List[Tuple[List[float], float]]:
    """Extract training samples from an eval run's metric history + applied mutations.

    Strategy: for each applied mutation, look at a window of metrics before and after.
    The "before" window gives us context features, the "after" window gives us the label.
    """
    history = result.metric_history
    applied = result.applied
    if not history or len(history) < 20:
        return []

    samples = []

    # Index metrics by step for fast lookup
    metrics_by_step: Dict[int, dict] = {}
    for rec in history:
        step = rec.get("step")
        metrics = rec.get("metrics", rec)
        if step is not None:
            metrics_by_step[step] = metrics

    steps = sorted(metrics_by_step.keys())
    if len(steps) < 20:
        return []

    # For each applied mutation, extract features
    for mut in applied:
        mut_step = mut.get("step", 0)
        if mut_step == 0:
            continue

        # Find window indices
        before_steps = [s for s in steps if mut_step - 15 <= s < mut_step]
        after_steps = [s for s in steps if mut_step < s <= mut_step + 25]

        if len(before_steps) < 3 or len(after_steps) < 3:
            continue

        # Context features from "before" window
        before_metrics = [metrics_by_step[s] for s in before_steps]
        after_metrics = [metrics_by_step[s] for s in after_steps]

        features = _compute_features(before_metrics, mut)
        label = _compute_label(before_metrics, after_metrics)

        if features is not None:
            samples.append((features, label))

    # Also extract "natural" samples (no mutation) at regular intervals
    # These train the model to distinguish "needs intervention" vs "fine"
    for i in range(10, len(steps) - 25, 30):
        window_before = [metrics_by_step[steps[j]] for j in range(max(0, i - 10), i)]
        window_after = [metrics_by_step[steps[j]] for j in range(i, min(len(steps), i + 20))]
        if len(window_before) < 3 or len(window_after) < 3:
            continue

        # "No intervention" sample — features with zero change, label from natural trajectory
        features = _compute_features(window_before, {"params": {}})
        label = _compute_label(window_before, window_after)
        if features is not None:
            samples.append((features, label))

    return samples


def _compute_features(before_metrics: List[dict], mutation: dict) -> List[float]:
    """Compute 13-dim feature vector from metric window + mutation."""
    try:
        losses = [m.get("train_loss", m.get("loss", 0)) for m in before_metrics]
        losses = [l for l in losses if l is not None and not math.isnan(l)]
        if not losses:
            return None

        loss_val = losses[-1]
        slope = (losses[-1] - losses[0]) / max(len(losses), 1) if len(losses) > 1 else 0.0
        volatility = _std(losses) / max(abs(_mean(losses)), 1e-8) if losses else 0.0

        grad_norms = [m.get("grad_norm", 0) for m in before_metrics]
        grad_norm = grad_norms[-1] if grad_norms else 0.0

        # Step fraction (approximate)
        step_frac = 0.5  # default, refined if step info available

        # Conflict score (approximate from loss variance)
        conflict = min(1.0, volatility)

        # Intervention features
        params = mutation.get("params", {})
        # Compute relative change from first param
        rel_change = 0.0
        abs_change = 0.0
        param_type = 0  # FLOAT
        for k, v in params.items():
            try:
                val = float(v)
                abs_change = val
                # Estimate relative change vs typical value
                if "lr" in k:
                    rel_change = (val - 1e-3) / 1e-3
                elif "weight" in k or "lambda" in k:
                    rel_change = val - 0.5  # centered around 0.5
                else:
                    rel_change = val
                break
            except (TypeError, ValueError):
                if isinstance(v, bool):
                    param_type = 2
                    abs_change = float(v)

        # Health signals
        is_plateau = float(abs(slope) < 0.005 and len(losses) > 5)
        is_diverging = float(slope > 0.1 and grad_norm > 10)
        is_oscillating = float(volatility > 0.3)

        return [
            float(loss_val),
            float(slope),
            float(volatility),
            float(grad_norm),
            float(step_frac),
            float(conflict),
            float(param_type),
            float(rel_change),
            float(abs_change),
            float(is_plateau),
            float(is_diverging),
            float(is_oscillating),
            float(conflict),
        ]
    except Exception:
        return None


def _compute_label(before_metrics: List[dict], after_metrics: List[dict]) -> float:
    """Compute binary label: 1.0 if improved, 0.0 if degraded/neutral."""
    try:
        before_losses = [m.get("train_loss", m.get("loss", 0)) for m in before_metrics]
        after_losses = [m.get("train_loss", m.get("loss", 0)) for m in after_metrics]
        before_losses = [l for l in before_losses if l is not None and not math.isnan(l)]
        after_losses = [l for l in after_losses if l is not None and not math.isnan(l)]

        if not before_losses or not after_losses:
            return 0.0

        before_avg = _mean(before_losses[-5:])
        after_avg = _mean(after_losses[-5:])

        # Improved if loss decreased by at least 2%
        if before_avg > 0 and (before_avg - after_avg) / before_avg > 0.02:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def _mean(vals: list) -> float:
    return sum(vals) / max(len(vals), 1)


def _std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ═══════════════════════════════════════════════════════════════════════
# Source 2: Synthetic heuristic samples
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_samples(count: int = 200) -> List[Tuple[List[float], float]]:
    """Generate synthetic training samples from policy pack rule patterns."""
    samples = []
    rng = random.Random(42)

    for _ in range(count):
        # Random context
        loss_val = rng.uniform(0.1, 5.0)
        slope = rng.uniform(-0.5, 0.5)
        volatility = rng.uniform(0.0, 1.0)
        grad_norm = rng.uniform(0.1, 50.0)
        step_frac = rng.uniform(0.0, 1.0)
        conflict = rng.uniform(0.0, 1.0)

        # Random intervention
        param_type = rng.randint(0, 4)
        rel_change = rng.uniform(-0.9, 2.0)
        abs_change = rng.uniform(-1.0, 1.0)

        # Health signals
        is_plateau = float(abs(slope) < 0.01 and step_frac > 0.3)
        is_diverging = float(slope > 0.2 and grad_norm > 20)
        is_oscillating = float(volatility > 0.5)
        health_conflict = conflict

        features = [
            loss_val, slope, volatility, grad_norm, step_frac, conflict,
            param_type, rel_change, abs_change,
            is_plateau, is_diverging, is_oscillating, health_conflict,
        ]

        # Heuristic label: improvement likely when...
        score = 0.0
        # Reducing LR when diverging → good
        if is_diverging and rel_change < 0:
            score += 0.7
        # Small changes when stable → good
        if abs(rel_change) < 0.3 and not is_diverging:
            score += 0.3
        # Large changes when oscillating → bad
        if is_oscillating and abs(rel_change) > 0.5:
            score -= 0.4
        # Plateau + increase → mildly good
        if is_plateau and rel_change > 0:
            score += 0.2

        label = 1.0 if score > 0.2 else 0.0
        samples.append((features, label))

    return samples


# ═══════════════════════════════════════════════════════════════════════
# Source 3: Augmentation
# ═══════════════════════════════════════════════════════════════════════

def augment_samples(
    samples: List[Tuple[List[float], float]],
    factor: float = 1.2,
    noise_std: float = 0.05,
) -> List[Tuple[List[float], float]]:
    """Add noise-perturbed copies of samples."""
    rng = random.Random(123)
    augmented = []
    target = int(len(samples) * factor) - len(samples)
    for i in range(target):
        base_features, label = samples[i % len(samples)]
        noisy = [
            v + rng.gauss(0, noise_std) if isinstance(v, float) else v
            for v in base_features
        ]
        augmented.append((noisy, label))
    return augmented


# ═══════════════════════════════════════════════════════════════════════
# Pre-training pipeline
# ═══════════════════════════════════════════════════════════════════════

def pretrain(
    output_path: str = _PRETRAINED_PATH,
    epochs: int = 100,
    include_scenarios: bool = False,
    include_eval: bool = False,
    eval_steps: int = 300,
    n_random: int = 20,
) -> dict:
    """Run the full pre-training pipeline.

    Args:
        output_path: Where to save the model weights.
        epochs: Training epochs.
        include_scenarios: Collect data from scenario runs (placeholder).
        include_eval: Run eval harness conditions for real intervention data.
        eval_steps: Steps per eval run (higher = better data, slower).
        n_random: Number of random conditions for eval data.
    """
    # 1. Synthetic samples (always included as baseline)
    synthetic = generate_synthetic_samples(200)

    # 2. Eval harness samples (highest quality)
    eval_samples: List[Tuple[List[float], float]] = []
    if include_eval:
        eval_samples = collect_eval_samples(
            max_steps=eval_steps, n_random_conditions=n_random,
        )

    # 3. Scenario samples (optional)
    scenario_samples: List[Tuple[List[float], float]] = []
    if include_scenarios:
        scenario_samples = _collect_scenario_samples()

    all_samples = synthetic + eval_samples + scenario_samples

    # 4. Augment
    augmented = augment_samples(all_samples)
    all_samples.extend(augmented)

    # 5. Build dataset
    dataset = _InMemoryDataset(all_samples)

    # 6. Train
    model = OutcomePredictor()
    model.train_from_dataset(dataset, epochs=epochs)

    # 7. Save
    model.save(output_path)

    # 8. Compute accuracy on training data
    accuracy = _eval_accuracy(model, all_samples)

    return {
        "total_samples": len(all_samples),
        "synthetic": len(synthetic),
        "eval": len(eval_samples),
        "scenario": len(scenario_samples),
        "augmented": len(augmented),
        "train_accuracy": round(accuracy, 4),
        "output_path": output_path,
    }


def _eval_accuracy(model: OutcomePredictor, samples: List[Tuple[List[float], float]]) -> float:
    """Compute binary accuracy on samples."""
    if not samples or model._model is None:
        return 0.0
    import torch
    X = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    y = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    with torch.no_grad():
        preds = model._model(X).squeeze()
        predicted = (preds > 0.5).float()
        correct = (predicted == y).sum().item()
    return correct / len(samples)


def _collect_scenario_samples() -> List[Tuple[List[float], float]]:
    """Collect samples from scenario runs. Returns empty if scenarios unavailable."""
    try:
        from hotcb.scenarios import list_scenarios
        # Placeholder: would need to run scenarios and extract effects
        return []
    except ImportError:
        return []


class _InMemoryDataset:
    """Minimal dataset interface for pre-training (avoids JSONL I/O)."""

    def __init__(self, samples: List[Tuple[List[float], float]]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def features(self):
        return [s[0] for s in self._samples]

    def labels(self):
        return [s[1] for s in self._samples]

    def as_tensors(self):
        import torch
        X = torch.tensor(self.features(), dtype=torch.float32)
        y = torch.tensor(self.labels(), dtype=torch.float32).unsqueeze(1)
        return X, y


if __name__ == "__main__":
    result = pretrain()
    print(f"Pre-training complete: {result}")
