"""Evaluation conditions — parameterized experiment definitions."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalCondition:
    """A single experiment condition."""

    name: str
    description: str
    # Which demo to run: "golden", "finetune", "simple"
    demo: str = "golden"
    # Override initial params (written to commands.jsonl at step 0)
    initial_overrides: Dict[str, Any] = field(default_factory=dict)
    # Whether to inject the demo's recipe
    use_recipe: bool = True
    # Autopilot mode: "off", "auto"
    autopilot: str = "off"
    # Policy packs to load (if autopilot != "off")
    policy_packs: List[str] = field(default_factory=list)
    # NN mode for research engine
    nn_mode: bool = False
    # Additional scheduled commands {step: command_dict}
    scheduled_commands: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # Max steps override
    max_steps: Optional[int] = None
    # Step delay override (faster for eval)
    step_delay: float = 0.01
    # Seed for reproducibility
    seed: Optional[int] = None
    # Research hypothesis text for this condition
    hypothesis_condition: str = ""
    hypothesis_expected: str = ""


def _random_lr(default: float, seed: int) -> float:
    """Randomize LR: 5x-50x off from default (both directions)."""
    rng = random.Random(seed)
    factor = rng.choice([0.02, 0.05, 0.1, 5.0, 10.0, 50.0])
    return round(default * factor, 8)


def _random_wd(default: float, seed: int) -> float:
    """Randomize weight_decay: 10x-100x off from default (both directions)."""
    rng = random.Random(seed)
    factor = rng.choice([0.0, 0.01, 10.0, 100.0, 1000.0])
    return round(default * factor, 8)


def _random_weight(default: float, seed: int) -> float:
    """Randomize loss weight: highly skewed (domination or suppression)."""
    rng = random.Random(seed)
    return round(rng.choice([0.01, 0.05, 0.95, 0.99, 3.0, 5.0]), 4)


# ═══════════════════════════════════════════════════════════════════════
# Golden demo conditions (multi-task classification + reconstruction)
# Defaults: lr=1e-3, wd=1e-4, weight_a=0.7, weight_b=0.3
# ═══════════════════════════════════════════════════════════════════════

GOLDEN_CONDITIONS: List[EvalCondition] = [
    # ── Baselines ──
    EvalCondition(
        name="golden_baseline",
        description="No recipe, no autopilot — default config, natural convergence",
        demo="golden",
        use_recipe=False,
        autopilot="off",
        seed=42,
        hypothesis_condition="default config (lr=1e-3, wd=1e-4, w_a=0.7, w_b=0.3), no interventions",
        hypothesis_expected="natural convergence to target losses by step 800",
    ),
    EvalCondition(
        name="golden_recipe",
        description="Default config + scheduled recipe (steps 200/400/500)",
        demo="golden",
        use_recipe=True,
        autopilot="off",
        seed=42,
        hypothesis_condition="default config + recipe (weight shift at 200, lr drop at 400, rebalance at 500)",
        hypothesis_expected="faster/better convergence than baseline due to scheduled tuning",
    ),

    # ── 1 bad param + auto ──
    EvalCondition(
        name="golden_1bad_lr_auto",
        description="LR randomized (10x default), auto mode recovery",
        demo="golden",
        initial_overrides={"opt": {"lr": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        hypothesis_condition="lr=0.01 (10x default), autopilot auto mode with stability+plateau packs",
        hypothesis_expected="autopilot detects instability, reduces lr, matches or approaches baseline final loss",
    ),
    EvalCondition(
        name="golden_1bad_wd_auto",
        description="Weight decay 100x too high, auto mode recovery",
        demo="golden",
        initial_overrides={"opt": {"weight_decay": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        hypothesis_condition="wd=0.01 (100x default), autopilot auto mode",
        hypothesis_expected="autopilot detects poor convergence, adjusts wd, approaches baseline",
    ),
    EvalCondition(
        name="golden_1bad_weight_auto",
        description="Task A weight = 0.95 (domination), auto mode recovery",
        demo="golden",
        initial_overrides={"loss": {"weight_a": 0.95, "weight_b": 0.05}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "multi_loss_assist"],
        seed=42,
        hypothesis_condition="weight_a=0.95, weight_b=0.05 (task domination), autopilot auto mode",
        hypothesis_expected="autopilot detects imbalance, rebalances weights, task B improves",
    ),

    # ── 2 bad params + auto ──
    EvalCondition(
        name="golden_2bad_lr_weight_auto",
        description="LR 10x too high + task domination, auto mode recovery",
        demo="golden",
        initial_overrides={
            "opt": {"lr": 0.01},
            "loss": {"weight_a": 0.95, "weight_b": 0.05},
        },
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "multi_loss_assist", "plateau_recovery"],
        seed=42,
        hypothesis_condition="lr=0.01 + weight_a=0.95 (2 bad params), autopilot auto mode",
        hypothesis_expected="autopilot stabilizes lr first, then rebalances weights, approaches baseline",
    ),
    EvalCondition(
        name="golden_2bad_wd_weight_auto",
        description="Weight decay 100x + task domination, auto mode recovery",
        demo="golden",
        initial_overrides={
            "opt": {"weight_decay": 0.01},
            "loss": {"weight_a": 0.05, "weight_b": 0.95},
        },
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "multi_loss_assist", "plateau_recovery"],
        seed=42,
        hypothesis_condition="wd=0.01 + weight_b=0.95 (2 bad params), autopilot auto mode",
        hypothesis_expected="autopilot adjusts wd and rebalances weights, approaches baseline",
    ),

    # ── NN-assisted auto mode ──
    EvalCondition(
        name="golden_1bad_lr_nn",
        description="LR 10x + auto + NN predictor (same as 1bad_lr_auto but NN-gated)",
        demo="golden",
        initial_overrides={"opt": {"lr": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        nn_mode=True,
        seed=42,
        hypothesis_condition="lr=0.01, autopilot auto + NN predictor",
        hypothesis_expected="NN-gated decisions improve intervention quality over pure rule-based",
    ),
    EvalCondition(
        name="golden_2bad_lr_weight_nn",
        description="LR 10x + task domination + auto + NN (same as 2bad but NN-gated)",
        demo="golden",
        initial_overrides={
            "opt": {"lr": 0.01},
            "loss": {"weight_a": 0.95, "weight_b": 0.05},
        },
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "multi_loss_assist", "plateau_recovery"],
        nn_mode=True,
        seed=42,
        hypothesis_condition="lr=0.01 + weight_a=0.95, autopilot auto + NN predictor",
        hypothesis_expected="NN-gated decisions match or beat pure rule-based under compound failures",
    ),

    # ── SWA (checkpoint averaging) ──
    EvalCondition(
        name="golden_swa_at_convergence",
        description="Enable SWA after step 400 (convergence phase)",
        demo="golden",
        initial_overrides={},
        use_recipe=True,
        autopilot="off",
        seed=42,
        scheduled_commands={
            400: {"module": "custom", "op": "set_params", "params": {"swa_enabled": True}, "source": "eval_swa"},
        },
        hypothesis_condition="default config + recipe + SWA enabled at convergence (step 400)",
        hypothesis_expected="SWA smooths late-training oscillations, improves final val_loss vs recipe-only",
    ),
    EvalCondition(
        name="golden_swa_1bad_lr_auto",
        description="LR 10x + auto + SWA enabled at convergence",
        demo="golden",
        initial_overrides={"opt": {"lr": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        scheduled_commands={
            200: {"module": "custom", "op": "set_params", "params": {"swa_enabled": True}, "source": "eval_swa"},
        },
        hypothesis_condition="lr=0.01 + autopilot auto + SWA at convergence",
        hypothesis_expected="autopilot stabilizes, SWA further smooths — better than auto alone",
    ),
    EvalCondition(
        name="golden_swa_2bad_auto",
        description="LR 10x + task domination + auto + SWA",
        demo="golden",
        initial_overrides={
            "opt": {"lr": 0.01},
            "loss": {"weight_a": 0.95, "weight_b": 0.05},
        },
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "multi_loss_assist", "plateau_recovery"],
        seed=42,
        scheduled_commands={
            200: {"module": "custom", "op": "set_params", "params": {"swa_enabled": True}, "source": "eval_swa"},
        },
        hypothesis_condition="lr=0.01 + weight_a=0.95 + autopilot + SWA at convergence",
        hypothesis_expected="combined autopilot + SWA handles compound failures better than autopilot alone",
    ),

    # ── Adversarial / stress ──
    EvalCondition(
        name="golden_divergent_lr_auto",
        description="Extremely high LR (50x), near-divergence recovery",
        demo="golden",
        initial_overrides={"opt": {"lr": 0.05}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        seed=42,
        hypothesis_condition="lr=0.05 (50x default), near-divergence start",
        hypothesis_expected="autopilot fires emergency lr cut, prevents divergence, converges",
    ),
    EvalCondition(
        name="golden_zero_wd_auto",
        description="Zero weight decay — overfitting stress test",
        demo="golden",
        initial_overrides={"opt": {"weight_decay": 0.0}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        hypothesis_condition="wd=0.0 (zero regularization)",
        hypothesis_expected="autopilot detects overfitting gap, adjusts wd upward",
    ),
    EvalCondition(
        name="golden_reversed_weights_auto",
        description="Weights inverted (A=0.3, B=0.7) — wrong priority",
        demo="golden",
        initial_overrides={"loss": {"weight_a": 0.3, "weight_b": 0.7}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["multi_loss_assist"],
        seed=42,
        hypothesis_condition="weight_a=0.3, weight_b=0.7 (inverted from default)",
        hypothesis_expected="multi_loss_assist detects imbalance, adjusts toward equilibrium",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Finetune demo conditions (transfer learning)
# Defaults: lr=5e-4, wd=5e-4, main=1.0
# ═══════════════════════════════════════════════════════════════════════

FINETUNE_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="finetune_baseline",
        description="No recipe, default config — stuck at phase 1 LR",
        demo="finetune",
        use_recipe=False,
        autopilot="off",
        seed=42,
        hypothesis_condition="lr=5e-4, no LR schedule recipe",
        hypothesis_expected="converges initially but overshoots without LR reduction",
    ),
    EvalCondition(
        name="finetune_recipe",
        description="Default config + LR schedule recipe (steps 200/400)",
        demo="finetune",
        use_recipe=True,
        autopilot="off",
        seed=42,
        hypothesis_condition="default config + recipe (lr drop at 200, 400)",
        hypothesis_expected="smooth convergence through all 3 phases",
    ),
    EvalCondition(
        name="finetune_high_lr_auto",
        description="LR 10x too high for finetuning, auto mode",
        demo="finetune",
        initial_overrides={"opt": {"lr": 5e-3}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        hypothesis_condition="lr=5e-3 (10x default) for finetuning task",
        hypothesis_expected="autopilot detects instability, reduces lr, achieves stable convergence",
    ),
    EvalCondition(
        name="finetune_no_wd_auto",
        description="Zero weight decay + auto mode — overfitting without regularization",
        demo="finetune",
        initial_overrides={"opt": {"weight_decay": 0.0}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        seed=42,
        hypothesis_condition="wd=0.0 in finetuning (no regularization)",
        hypothesis_expected="autopilot detects val/train gap, increases wd",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Simple demo conditions (optimizer-only control)
# Defaults: lr≈0.003 (via param_groups), wd≈0.0001
# ═══════════════════════════════════════════════════════════════════════

SIMPLE_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="simple_baseline",
        description="Simple demo baseline — default config",
        demo="simple",
        use_recipe=False,
        autopilot="off",
        max_steps=300,
        seed=42,
        hypothesis_condition="default config, no autopilot",
        hypothesis_expected="natural convergence to low loss",
    ),
    EvalCondition(
        name="simple_high_lr_auto",
        description="LR 20x too high, auto mode recovery",
        demo="simple",
        initial_overrides={"opt": {"lr": 0.06}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        max_steps=300,
        seed=42,
        hypothesis_condition="lr=0.06 (20x default), autopilot auto",
        hypothesis_expected="autopilot stabilizes lr, achieves convergence",
    ),
    EvalCondition(
        name="simple_low_lr_auto",
        description="LR 100x too low, auto mode (plateau detection)",
        demo="simple",
        initial_overrides={"opt": {"lr": 3e-5}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["plateau_recovery"],
        max_steps=300,
        seed=42,
        hypothesis_condition="lr=3e-5 (100x too low), autopilot with plateau pack",
        hypothesis_expected="autopilot detects slow convergence, increases lr",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# MNIST conditions (real PyTorch — small CNN, ~14K params)
# Defaults: lr=1e-3 (Adam), wd=1e-4
# ═══════════════════════════════════════════════════════════════════════

MNIST_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="mnist_baseline",
        description="Real MNIST CNN — default config, no interventions",
        demo="mnist",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="default config (Adam lr=1e-3, wd=1e-4), no interventions",
        hypothesis_expected="converges to >98% val_accuracy within 1500 steps (3 epochs)",
    ),
    EvalCondition(
        name="mnist_high_lr_auto",
        description="Real MNIST — LR 30x too high (grad_norm>10), autopilot recovery",
        demo="mnist",
        initial_overrides={"opt": {"lr": 0.03}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.03 (30x default), grad_norm peaks ~24, triggers grad_spike_clip",
        hypothesis_expected="grad_spike_clip fires (grad>10), lr halved to ~0.015, recovers to >90% val_accuracy",
    ),
    EvalCondition(
        name="mnist_high_lr_no_auto",
        description="Real MNIST — LR 30x too high, NO autopilot (control for high_lr_auto)",
        demo="mnist",
        initial_overrides={"opt": {"lr": 0.03}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.03 (30x default), no autopilot — direct comparison with high_lr_auto",
        hypothesis_expected="grad spikes damage model, no recovery, worse final val_loss than high_lr_auto",
    ),
    EvalCondition(
        name="mnist_high_wd_auto",
        description="Real MNIST — weight decay 1000x too high, regularization damage",
        demo="mnist",
        initial_overrides={"opt": {"weight_decay": 0.1}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=0.1 (1000x default), severe over-regularization",
        hypothesis_expected="high wd suppresses learning, plateau rules detect stagnation",
    ),
    EvalCondition(
        name="mnist_divergent_lr_auto",
        description="Real MNIST — LR 50x too high (loss>50), multi-rule recovery",
        demo="mnist",
        initial_overrides={"opt": {"lr": 0.05}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.05 (50x default), loss peaks ~59, grad peaks ~93",
        hypothesis_expected="lr_emergency_floor fires (loss>100), grad_spike_clip fires, lr crushed to ~0.005, stabilizes",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# CIFAR-10 conditions (real PyTorch — small CNN, ~62K params)
# Defaults: lr=0.01 (SGD+momentum), wd=1e-4
# ═══════════════════════════════════════════════════════════════════════

CIFAR10_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="cifar10_baseline",
        description="Real CIFAR-10 CNN — default config, no interventions",
        demo="cifar10",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="default config (SGD lr=0.01, mom=0.9, wd=1e-4), no interventions",
        hypothesis_expected="converges to >65% val_accuracy within 2000 steps (5 epochs)",
    ),
    EvalCondition(
        name="cifar10_high_lr_auto",
        description="Real CIFAR-10 — LR 15x too high (grad_norm>10), autopilot recovery",
        demo="cifar10",
        initial_overrides={"opt": {"lr": 0.15}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.15 (15x default), grad_norm peaks ~14, triggers grad_spike_clip",
        hypothesis_expected="grad_spike_clip fires, lr halved to ~0.075, training stabilizes and converges",
    ),
    EvalCondition(
        name="cifar10_high_lr_no_auto",
        description="Real CIFAR-10 — LR 15x too high, NO autopilot (control for high_lr_auto)",
        demo="cifar10",
        initial_overrides={"opt": {"lr": 0.15}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.15 (15x default), no autopilot — direct comparison with high_lr_auto",
        hypothesis_expected="grad spikes persist uncorrected, worse convergence than autopilot version",
    ),
    EvalCondition(
        name="cifar10_divergent_lr_auto",
        description="Real CIFAR-10 — LR 20x too high (grad~40, loss~20), multi-rule recovery",
        demo="cifar10",
        initial_overrides={"opt": {"lr": 0.2}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.2 (20x default), grad peaks ~40, loss peaks ~20",
        hypothesis_expected="grad_spike_clip + loss_spike_recovery fire, lr reduced until stable convergence",
    ),
    EvalCondition(
        name="cifar10_high_wd_auto",
        description="Real CIFAR-10 — weight decay 100x too high, over-regularization",
        demo="cifar10",
        initial_overrides={"opt": {"weight_decay": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=0.01 (100x default), over-regularization slows learning",
        hypothesis_expected="plateau rules detect stagnation, lr adjusted to compensate",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Continuation conditions — late-stage tuning from converged checkpoints
# These require a base run with checkpoints to already exist.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ContinuationCondition:
    """A continuation experiment condition — branches from a converged checkpoint."""

    name: str
    description: str
    task: str  # "mnist" or "cifar10"
    # Recipe mutations applied at branch start
    recipe_name: str = ""
    recipe_ops: List[Dict[str, Any]] = field(default_factory=list)
    # Resume mode
    resume_mode: str = "full_resume"
    # Extra training steps for the branch
    extra_steps: int = 500
    # Anchor selection: "auto", "best_val", "end_minus", or step number
    anchor_policy: str = "auto"
    # Autopilot during continuation
    autopilot: str = "off"
    policy_packs: List[str] = field(default_factory=list)
    # Validation
    primary_metric: str = "val_accuracy"
    metric_mode: str = "max"
    min_improvement_delta: float = 0.001
    # Research
    hypothesis_condition: str = ""
    hypothesis_expected: str = ""


# ── MNIST continuation conditions ──
# Base: Adam lr=1e-3, wd=1e-4, converged to ~98%+ val_accuracy in 1500 steps

MNIST_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    # --- LR reduction recipes ---
    ContinuationCondition(
        name="mnist_cont_lr_half",
        description="Halve LR from near-best checkpoint — stabilize late oscillations",
        task="mnist",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="LR halved (5e-4) from best-val anchor, full resume with optimizer state",
        hypothesis_expected="reduced oscillation, +0.1-0.5% val_accuracy improvement",
    ),
    ContinuationCondition(
        name="mnist_cont_lr_quarter",
        description="Quarter LR — aggressive tail stabilization",
        task="mnist",
        recipe_name="lr_quarter",
        recipe_ops=[{"target": "lr", "value": 0.25, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="LR quartered (2.5e-4) from best-val anchor, full resume",
        hypothesis_expected="strong stabilization, may plateau faster but more stable final metric",
    ),
    ContinuationCondition(
        name="mnist_cont_lr_half_weights_only",
        description="Halve LR + fresh optimizer — escape momentum inertia",
        task="mnist",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="weights_only",
        extra_steps=500,
        hypothesis_condition="LR halved, weights-only resume (fresh Adam state)",
        hypothesis_expected="fresh momentum allows new gradient direction, potentially better than full_resume",
    ),

    # --- SWA/EMA recipes ---
    ContinuationCondition(
        name="mnist_cont_swa_tail",
        description="Enable SWA from converged checkpoint — weight averaging",
        task="mnist",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="SWA enabled from best-val anchor, full resume",
        hypothesis_expected="SWA smooths weight space, +0.1-0.3% val_accuracy from averaging",
    ),
    ContinuationCondition(
        name="mnist_cont_ema_tail",
        description="Enable EMA from converged checkpoint — exponential shadow weights",
        task="mnist",
        recipe_name="ema_tail",
        recipe_ops=[{"target": "ema_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="EMA (decay=0.999) enabled from best-val anchor, full resume",
        hypothesis_expected="EMA shadow weights improve generalization, smooth final checkpoint",
    ),
    ContinuationCondition(
        name="mnist_cont_swa_lr_half",
        description="SWA + halved LR — combined late-stage smoothing",
        task="mnist",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="SWA + LR halved from best-val anchor",
        hypothesis_expected="combined effect: lower LR reduces oscillation, SWA averages — best of both",
    ),

    # --- Weight decay / grad clip recipes ---
    ContinuationCondition(
        name="mnist_cont_wd_reduce",
        description="Halve weight decay — reduce late-stage regularization pressure",
        task="mnist",
        recipe_name="wd_reduce",
        recipe_ops=[{"target": "weight_decay", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="WD halved (5e-5) from best-val anchor, full resume",
        hypothesis_expected="less regularization allows tighter fit in late phase — if not overfitting",
    ),
    ContinuationCondition(
        name="mnist_cont_grad_clip_tight",
        description="Tighten grad clip to 0.5 — suppress late gradient spikes",
        task="mnist",
        recipe_name="grad_clip_tight",
        recipe_ops=[{"target": "grad_clip", "value": 0.5}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="Grad clip tightened to 0.5 from best-val anchor",
        hypothesis_expected="suppresses any late instability, marginal improvement on stable runs",
    ),

    # --- Combined recipes ---
    ContinuationCondition(
        name="mnist_cont_kitchen_sink",
        description="LR half + SWA + tighter grad clip — maximal late-stage treatment",
        task="mnist",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 1.0},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="LR half + SWA + grad_clip=1.0, full resume from best-val anchor",
        hypothesis_expected="combined treatment covers multiple late-stage failure modes",
    ),

    # --- Autopilot-guided continuation ---
    ContinuationCondition(
        name="mnist_cont_autopilot",
        description="Auto mode continuation — let autopilot decide mutations",
        task="mnist",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=500,
        hypothesis_condition="No manual recipe, autopilot auto mode with finish_strong pack",
        hypothesis_expected="autopilot detects late-stage opportunities and applies appropriate mutations",
    ),
]


# ── CIFAR-10 continuation conditions ──
# Base: SGD lr=0.01, mom=0.9, wd=1e-4, converged to ~65%+ val_accuracy in 2000 steps

CIFAR10_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    # --- LR reduction recipes ---
    ContinuationCondition(
        name="cifar10_cont_lr_half",
        description="Halve LR from near-best checkpoint",
        task="cifar10",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR halved (0.005) from best-val anchor, full resume with SGD momentum",
        hypothesis_expected="SGD with halved LR finds tighter minimum, +0.5-2% val_accuracy",
    ),
    ContinuationCondition(
        name="cifar10_cont_lr_quarter",
        description="Quarter LR — aggressive tail for SGD",
        task="cifar10",
        recipe_name="lr_quarter",
        recipe_ops=[{"target": "lr", "value": 0.25, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR quartered (0.0025) from best-val anchor",
        hypothesis_expected="very stable tail convergence, trades speed for precision",
    ),
    ContinuationCondition(
        name="cifar10_cont_lr_half_weights_only",
        description="Halve LR + fresh SGD — reset momentum for new direction",
        task="cifar10",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="weights_only",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR halved, weights-only resume (fresh SGD momentum)",
        hypothesis_expected="SGD momentum reset allows escape from current trajectory",
    ),

    # --- SWA/EMA recipes ---
    ContinuationCondition(
        name="cifar10_cont_swa_tail",
        description="Enable SWA — known effective for SGD on CIFAR-10",
        task="cifar10",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="SWA enabled from best-val anchor, SGD continues",
        hypothesis_expected="SWA is well-studied for SGD: expect +0.5-1.5% val_accuracy (Izmailov 2018)",
    ),
    ContinuationCondition(
        name="cifar10_cont_ema_tail",
        description="Enable EMA — exponential shadow for SGD",
        task="cifar10",
        recipe_name="ema_tail",
        recipe_ops=[{"target": "ema_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="EMA (decay=0.999) enabled from best-val anchor",
        hypothesis_expected="EMA smoother than SWA for short continuation — expect comparable gains",
    ),
    ContinuationCondition(
        name="cifar10_cont_swa_lr_half",
        description="SWA + halved LR — canonical SWA recipe for SGD",
        task="cifar10",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="SWA + LR halved — matches SWA paper protocol",
        hypothesis_expected="canonical SWA: lower LR + averaging = best single-model improvement",
    ),

    # --- Weight decay / grad clip ---
    ContinuationCondition(
        name="cifar10_cont_wd_reduce",
        description="Halve weight decay — relax regularization in tail",
        task="cifar10",
        recipe_name="wd_reduce",
        recipe_ops=[{"target": "weight_decay", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="WD halved (5e-5) from best-val anchor",
        hypothesis_expected="slight under-regularization late may help if model is capacity-limited",
    ),
    ContinuationCondition(
        name="cifar10_cont_grad_clip_tight",
        description="Tighten grad clip to 1.0 — stabilize SGD late-phase",
        task="cifar10",
        recipe_name="grad_clip_tight",
        recipe_ops=[{"target": "grad_clip", "value": 1.0}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="Grad clip tightened to 1.0 from best-val anchor",
        hypothesis_expected="SGD can spike late — clipping stabilizes without losing convergence",
    ),

    # --- Combined recipes ---
    ContinuationCondition(
        name="cifar10_cont_kitchen_sink",
        description="LR half + SWA + grad clip 2.0 — maximal treatment",
        task="cifar10",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 2.0},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR half + SWA + grad_clip=2.0, full resume",
        hypothesis_expected="combined treatment is the most aggressive — likely best or overfit",
    ),

    # --- Autopilot-guided ---
    ContinuationCondition(
        name="cifar10_cont_autopilot",
        description="Auto mode continuation — autopilot decides",
        task="cifar10",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="No manual recipe, autopilot with finish_strong pack",
        hypothesis_expected="autopilot detects late-stage patterns and intervenes appropriately",
    ),
]


# ── MNIST new recipe continuation conditions ──

MNIST_NEW_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    # --- Aggressive recipes ---
    ContinuationCondition(
        name="mnist_cont_lr_tenth",
        description="Emergency LR reduction to 10% — hard stop on momentum",
        task="mnist",
        recipe_name="lr_tenth",
        recipe_ops=[{"target": "lr", "value": 0.1, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="LR reduced to 10% (1e-4) from best-val anchor, full resume",
        hypothesis_expected="aggressive deceleration — may plateau early but very stable",
    ),
    ContinuationCondition(
        name="mnist_cont_momentum_reset",
        description="LR 30% + fresh optimizer — escape accumulated momentum",
        task="mnist",
        recipe_name="momentum_reset",
        recipe_ops=[{"target": "lr", "value": 0.3, "relative": True}],
        resume_mode="weights_only",
        extra_steps=500,
        hypothesis_condition="LR 30% (3e-4), weights-only resume (fresh Adam)",
        hypothesis_expected="fresh momentum + low LR explores new direction, may find better basin",
    ),
    ContinuationCondition(
        name="mnist_cont_kitchen_sink_aggressive",
        description="Maximum intervention: lr×0.25 + SWA + grad_clip=0.5 + wd×0.25",
        task="mnist",
        recipe_name="kitchen_sink_aggressive",
        recipe_ops=[
            {"target": "lr", "value": 0.25, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 0.5},
            {"target": "weight_decay", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="lr×0.25 + SWA + grad_clip=0.5 + wd×0.25, full resume",
        hypothesis_expected="maximum intervention covers all failure modes — best or overfit",
    ),

    # --- Combo recipes ---
    ContinuationCondition(
        name="mnist_cont_swa_grad_clip",
        description="SWA + gradient clipping — smoothing with stability",
        task="mnist",
        recipe_name="swa_grad_clip",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 1.0},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="SWA + grad_clip=1.0 from best-val anchor",
        hypothesis_expected="combined smoothing + clipping for robust late phase",
    ),
    ContinuationCondition(
        name="mnist_cont_ema_lr_quarter",
        description="EMA + quartered LR — conservative smoothed tail",
        task="mnist",
        recipe_name="ema_lr_quarter",
        recipe_ops=[
            {"target": "ema_enabled", "value": True},
            {"target": "lr", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="EMA + LR quartered from best-val anchor",
        hypothesis_expected="EMA shadow weights + very low LR for precision convergence",
    ),
    ContinuationCondition(
        name="mnist_cont_swa_wd_reduce",
        description="SWA + halved weight decay — averaging with relaxed regularization",
        task="mnist",
        recipe_name="swa_wd_reduce",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "weight_decay", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="SWA + WD halved from best-val anchor",
        hypothesis_expected="SWA averaging + reduced regularization for late tightening",
    ),

    # --- Multi-stage recipes ---
    ContinuationCondition(
        name="mnist_cont_cosine_anneal_tail",
        description="Staged LR decay: 0.7→0.4→0.15 mimicking cosine tail",
        task="mnist",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 150},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 300},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="staged LR decay: 0.7×→0.4×→0.15× at steps 0/150/300",
        hypothesis_expected="smooth cosine-like decay is the gold standard for late convergence",
    ),
    ContinuationCondition(
        name="mnist_cont_staged_swa_lr",
        description="SWA first, then LR cut, then grad clip — phased treatment",
        task="mnist",
        recipe_name="staged_swa_lr",
        recipe_ops=[
            {"target": "swa_enabled", "value": True, "delay_steps": 0},
            {"target": "lr", "value": 0.5, "relative": True, "delay_steps": 100},
            {"target": "grad_clip", "value": 1.0, "delay_steps": 250},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="SWA at step 0, LR×0.5 at step 100, grad_clip=1.0 at step 250",
        hypothesis_expected="phased treatment allows each intervention to take effect before next",
    ),
    ContinuationCondition(
        name="mnist_cont_progressive_wd",
        description="Gradually relax weight decay: 0.75→0.5→0.25 of base",
        task="mnist",
        recipe_name="progressive_wd_reduction",
        recipe_ops=[
            {"target": "weight_decay", "value": 0.75, "relative": True, "delay_steps": 0},
            {"target": "weight_decay", "value": 0.5, "relative": True, "delay_steps": 150},
            {"target": "weight_decay", "value": 0.25, "relative": True, "delay_steps": 300},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        hypothesis_condition="progressive WD reduction: 0.75×→0.5×→0.25× at steps 0/150/300",
        hypothesis_expected="gradual regularization relaxation for late-stage fine convergence",
    ),
]


# ── CIFAR-10 new recipe continuation conditions ──

CIFAR10_NEW_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    # --- Aggressive recipes ---
    ContinuationCondition(
        name="cifar10_cont_lr_tenth",
        description="Emergency LR to 10% — hard SGD deceleration",
        task="cifar10",
        recipe_name="lr_tenth",
        recipe_ops=[{"target": "lr", "value": 0.1, "relative": True}],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR reduced to 10% (0.001) from best-val anchor, SGD",
        hypothesis_expected="aggressive SGD deceleration — stable but may be too slow",
    ),
    ContinuationCondition(
        name="cifar10_cont_momentum_reset",
        description="LR 30% + fresh SGD — reset momentum direction",
        task="cifar10",
        recipe_name="momentum_reset",
        recipe_ops=[{"target": "lr", "value": 0.3, "relative": True}],
        resume_mode="weights_only",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="LR 30% (0.003), weights-only resume (fresh SGD momentum)",
        hypothesis_expected="fresh momentum + low LR for new descent direction",
    ),
    ContinuationCondition(
        name="cifar10_cont_kitchen_sink_aggressive",
        description="Maximum: lr×0.25 + SWA + grad_clip=0.5 + wd×0.25",
        task="cifar10",
        recipe_name="kitchen_sink_aggressive",
        recipe_ops=[
            {"target": "lr", "value": 0.25, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 0.5},
            {"target": "weight_decay", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="lr×0.25 + SWA + grad_clip=0.5 + wd×0.25, SGD full resume",
        hypothesis_expected="maximum intervention on SGD — expected best or worst",
    ),

    # --- Combo recipes ---
    ContinuationCondition(
        name="cifar10_cont_swa_grad_clip",
        description="SWA + gradient clipping for SGD stability",
        task="cifar10",
        recipe_name="swa_grad_clip",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 1.0},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="SWA + grad_clip=1.0 from best-val anchor, SGD",
        hypothesis_expected="SWA is proven for SGD + clipping prevents late spikes",
    ),
    ContinuationCondition(
        name="cifar10_cont_ema_lr_quarter",
        description="EMA + quartered LR — conservative smoothed SGD tail",
        task="cifar10",
        recipe_name="ema_lr_quarter",
        recipe_ops=[
            {"target": "ema_enabled", "value": True},
            {"target": "lr", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="EMA + LR quartered from best-val, SGD",
        hypothesis_expected="EMA + very low LR for precision SGD convergence",
    ),
    ContinuationCondition(
        name="cifar10_cont_swa_wd_reduce",
        description="SWA + halved weight decay",
        task="cifar10",
        recipe_name="swa_wd_reduce",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "weight_decay", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="SWA + WD halved from best-val, SGD",
        hypothesis_expected="SWA averaging + relaxed regularization for SGD late phase",
    ),

    # --- Multi-stage recipes ---
    ContinuationCondition(
        name="cifar10_cont_cosine_anneal_tail",
        description="Staged LR decay: 0.7→0.4→0.15 mimicking cosine tail",
        task="cifar10",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 150},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 300},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="staged SGD LR decay: 0.7×→0.4×→0.15× at 0/150/300",
        hypothesis_expected="cosine-style decay is canonical for SGD CIFAR convergence",
    ),
    ContinuationCondition(
        name="cifar10_cont_staged_swa_lr",
        description="SWA first, then LR cut, then grad clip — phased",
        task="cifar10",
        recipe_name="staged_swa_lr",
        recipe_ops=[
            {"target": "swa_enabled", "value": True, "delay_steps": 0},
            {"target": "lr", "value": 0.5, "relative": True, "delay_steps": 100},
            {"target": "grad_clip", "value": 1.0, "delay_steps": 250},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="SWA at 0, LR×0.5 at 100, grad_clip at 250, SGD",
        hypothesis_expected="phased treatment for SGD — each stage builds on previous",
    ),
    ContinuationCondition(
        name="cifar10_cont_progressive_wd",
        description="Gradually relax WD: 0.75→0.5→0.25 of base",
        task="cifar10",
        recipe_name="progressive_wd_reduction",
        recipe_ops=[
            {"target": "weight_decay", "value": 0.75, "relative": True, "delay_steps": 0},
            {"target": "weight_decay", "value": 0.5, "relative": True, "delay_steps": 150},
            {"target": "weight_decay", "value": 0.25, "relative": True, "delay_steps": 300},
        ],
        resume_mode="full_resume",
        extra_steps=500,
        primary_metric="val_accuracy",
        hypothesis_condition="progressive WD reduction: 0.75×→0.5×→0.25× at 0/150/300",
        hypothesis_expected="gradual regularization relaxation for SGD convergence",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# COCO MobileNetV2 conditions
# Defaults: Adam lr=1e-4, wd=1e-5, MobileNetV2 pretrained
# ═══════════════════════════════════════════════════════════════════════

COCO_MOBILENET_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="coco_mobilenet_baseline",
        description="COCO MobileNetV2 — pretrained, default config",
        demo="coco_mobilenet",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="pretrained MobileNetV2, Adam lr=1e-4, wd=1e-5",
        hypothesis_expected="converges on 80-class COCO classification",
    ),
    EvalCondition(
        name="coco_mobilenet_high_lr_auto",
        description="COCO MobileNetV2 — LR 10x, autopilot recovery",
        demo="coco_mobilenet",
        initial_overrides={"opt": {"lr": 1e-3}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=1e-3 (10x), autopilot auto",
        hypothesis_expected="autopilot stabilizes, approaches baseline accuracy",
    ),
    EvalCondition(
        name="coco_mobilenet_divergent",
        description="COCO MobileNetV2 — LR 50x, near-divergence",
        demo="coco_mobilenet",
        initial_overrides={"opt": {"lr": 5e-3}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=5e-3 (50x), near-divergence",
        hypothesis_expected="autopilot fires emergency cuts, prevents divergence",
    ),
    EvalCondition(
        name="coco_mobilenet_high_wd",
        description="COCO MobileNetV2 — WD 100x, over-regularization",
        demo="coco_mobilenet",
        initial_overrides={"opt": {"weight_decay": 1e-3}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=1e-3 (100x), over-regularization",
        hypothesis_expected="plateau detection fires, adjusts parameters",
    ),
    EvalCondition(
        name="coco_mobilenet_no_auto",
        description="COCO MobileNetV2 — LR 10x, NO autopilot (control)",
        demo="coco_mobilenet",
        initial_overrides={"opt": {"lr": 1e-3}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=1e-3 (10x), no autopilot",
        hypothesis_expected="unassisted — worse than autopilot version",
    ),
]

COCO_MOBILENET_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    ContinuationCondition(
        name="coco_cont_lr_half",
        description="COCO MobileNetV2 — halve LR from converged checkpoint",
        task="coco_mobilenet",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="LR halved from best-val anchor",
        hypothesis_expected="stabilizes late-phase oscillation",
    ),
    ContinuationCondition(
        name="coco_cont_swa_tail",
        description="COCO MobileNetV2 — SWA from converged checkpoint",
        task="coco_mobilenet",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="SWA enabled from best-val anchor",
        hypothesis_expected="weight averaging improves generalization",
    ),
    ContinuationCondition(
        name="coco_cont_swa_lr_half",
        description="COCO MobileNetV2 — SWA + halved LR",
        task="coco_mobilenet",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="SWA + LR halved from best-val anchor",
        hypothesis_expected="combined SWA + lower LR for maximum smoothing",
    ),
    ContinuationCondition(
        name="coco_cont_kitchen_sink",
        description="COCO MobileNetV2 — maximal treatment",
        task="coco_mobilenet",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 1.0},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="LR half + SWA + grad_clip=1.0",
        hypothesis_expected="combined treatment for best result",
    ),
    ContinuationCondition(
        name="coco_cont_cosine_anneal",
        description="COCO MobileNetV2 — staged LR decay",
        task="coco_mobilenet",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 300},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 600},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="staged LR decay: 0.7×→0.4×→0.15×",
        hypothesis_expected="cosine-like decay for smooth convergence",
    ),
    ContinuationCondition(
        name="coco_cont_lr_tenth",
        description="COCO MobileNetV2 — emergency LR to 10%",
        task="coco_mobilenet",
        recipe_name="lr_tenth",
        recipe_ops=[{"target": "lr", "value": 0.1, "relative": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="LR reduced to 10% from best-val anchor",
        hypothesis_expected="hard deceleration — very stable but slow",
    ),
    ContinuationCondition(
        name="coco_cont_ema_lr_quarter",
        description="COCO MobileNetV2 — EMA + quartered LR",
        task="coco_mobilenet",
        recipe_name="ema_lr_quarter",
        recipe_ops=[
            {"target": "ema_enabled", "value": True},
            {"target": "lr", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="EMA + LR quartered from best-val anchor",
        hypothesis_expected="EMA shadow + low LR for precision",
    ),
    ContinuationCondition(
        name="coco_cont_momentum_reset",
        description="COCO MobileNetV2 — LR 30% + fresh optimizer",
        task="coco_mobilenet",
        recipe_name="momentum_reset",
        recipe_ops=[{"target": "lr", "value": 0.3, "relative": True}],
        resume_mode="weights_only",
        extra_steps=1000,
        hypothesis_condition="LR 30%, weights-only resume (fresh Adam)",
        hypothesis_expected="fresh momentum explores new direction",
    ),
    ContinuationCondition(
        name="coco_cont_staged_swa_lr",
        description="COCO MobileNetV2 — phased SWA+LR+clip",
        task="coco_mobilenet",
        recipe_name="staged_swa_lr",
        recipe_ops=[
            {"target": "swa_enabled", "value": True, "delay_steps": 0},
            {"target": "lr", "value": 0.5, "relative": True, "delay_steps": 200},
            {"target": "grad_clip", "value": 1.0, "delay_steps": 500},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        hypothesis_condition="SWA→LR cut→clip at 0/200/500",
        hypothesis_expected="phased treatment with each stage building",
    ),
    ContinuationCondition(
        name="coco_cont_autopilot",
        description="COCO MobileNetV2 — autopilot-guided continuation",
        task="coco_mobilenet",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=1000,
        hypothesis_condition="autopilot auto with finish_strong",
        hypothesis_expected="autopilot finds optimal late-stage interventions",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# ImageNet MobileNetV2 conditions
# Baseline: pretrained 71.878% top-1 (Sandler 2018), SGD lr=0.001
# ═══════════════════════════════════════════════════════════════════════

IMAGENET_MOBILENET_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="imagenet_mobilenet_baseline",
        description="ImageNet MobileNetV2 — pretrained baseline, fine-tune",
        demo="imagenet_mobilenet",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="pretrained MobileNetV2 (71.878%), SGD lr=0.001",
        hypothesis_expected="maintains or improves beyond 71.878% top-1",
    ),
    EvalCondition(
        name="imagenet_mobilenet_high_lr_auto",
        description="ImageNet MobileNetV2 — LR 10x, autopilot",
        demo="imagenet_mobilenet",
        initial_overrides={"opt": {"lr": 0.01}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.01 (10x), autopilot auto",
        hypothesis_expected="autopilot stabilizes, maintains pretrained accuracy",
    ),
    EvalCondition(
        name="imagenet_mobilenet_divergent",
        description="ImageNet MobileNetV2 — LR 50x, divergence risk",
        demo="imagenet_mobilenet",
        initial_overrides={"opt": {"lr": 0.05}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.05 (50x), near-divergence",
        hypothesis_expected="autopilot prevents catastrophic forgetting",
    ),
    EvalCondition(
        name="imagenet_mobilenet_high_wd",
        description="ImageNet MobileNetV2 — WD 100x",
        demo="imagenet_mobilenet",
        initial_overrides={"opt": {"weight_decay": 4e-3}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=4e-3 (100x), over-regularization",
        hypothesis_expected="plateau rules detect and correct",
    ),
    EvalCondition(
        name="imagenet_mobilenet_no_auto",
        description="ImageNet MobileNetV2 — LR 10x, no autopilot (control)",
        demo="imagenet_mobilenet",
        initial_overrides={"opt": {"lr": 0.01}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.01 (10x), no autopilot",
        hypothesis_expected="unassisted degradation from pretrained",
    ),
]

IMAGENET_MOBILENET_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    ContinuationCondition(
        name="imagenet_cont_lr_half",
        description="ImageNet MobileNetV2 — halve LR",
        task="imagenet_mobilenet",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="LR halved from best-val, SGD",
        hypothesis_expected="tighter convergence beyond 71.878%",
    ),
    ContinuationCondition(
        name="imagenet_cont_swa_tail",
        description="ImageNet MobileNetV2 — SWA",
        task="imagenet_mobilenet",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="SWA from best-val anchor, SGD",
        hypothesis_expected="SWA proven for SGD — expect improvement over pretrained",
    ),
    ContinuationCondition(
        name="imagenet_cont_swa_lr_half",
        description="ImageNet MobileNetV2 — SWA + halved LR (canonical)",
        task="imagenet_mobilenet",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="SWA + LR halved — canonical SWA recipe",
        hypothesis_expected="canonical SWA approach should beat pretrained baseline",
    ),
    ContinuationCondition(
        name="imagenet_cont_kitchen_sink",
        description="ImageNet MobileNetV2 — maximal treatment",
        task="imagenet_mobilenet",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 2.0},
        ],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="LR half + SWA + grad_clip=2.0",
        hypothesis_expected="combined for maximum improvement",
    ),
    ContinuationCondition(
        name="imagenet_cont_cosine_anneal",
        description="ImageNet MobileNetV2 — staged LR decay",
        task="imagenet_mobilenet",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 500},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 1000},
        ],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="staged LR decay: 0.7×→0.4×→0.15× for SGD",
        hypothesis_expected="cosine-like SGD decay for SOTA improvement",
    ),
    ContinuationCondition(
        name="imagenet_cont_lr_tenth",
        description="ImageNet MobileNetV2 — emergency LR 10%",
        task="imagenet_mobilenet",
        recipe_name="lr_tenth",
        recipe_ops=[{"target": "lr", "value": 0.1, "relative": True}],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="LR to 10% from best-val, SGD",
        hypothesis_expected="aggressive deceleration for precision",
    ),
    ContinuationCondition(
        name="imagenet_cont_ema_lr_quarter",
        description="ImageNet MobileNetV2 — EMA + quartered LR",
        task="imagenet_mobilenet",
        recipe_name="ema_lr_quarter",
        recipe_ops=[
            {"target": "ema_enabled", "value": True},
            {"target": "lr", "value": 0.25, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="EMA + LR quartered, SGD",
        hypothesis_expected="EMA + very low SGD for precision",
    ),
    ContinuationCondition(
        name="imagenet_cont_momentum_reset",
        description="ImageNet MobileNetV2 — fresh SGD momentum",
        task="imagenet_mobilenet",
        recipe_name="momentum_reset",
        recipe_ops=[{"target": "lr", "value": 0.3, "relative": True}],
        resume_mode="weights_only",
        extra_steps=2000,
        hypothesis_condition="LR 30%, weights-only (fresh SGD momentum)",
        hypothesis_expected="fresh SGD direction from pretrained weights",
    ),
    ContinuationCondition(
        name="imagenet_cont_staged_swa_lr",
        description="ImageNet MobileNetV2 — phased SWA+LR+clip",
        task="imagenet_mobilenet",
        recipe_name="staged_swa_lr",
        recipe_ops=[
            {"target": "swa_enabled", "value": True, "delay_steps": 0},
            {"target": "lr", "value": 0.5, "relative": True, "delay_steps": 500},
            {"target": "grad_clip", "value": 2.0, "delay_steps": 1000},
        ],
        resume_mode="full_resume",
        extra_steps=2000,
        hypothesis_condition="SWA→LR cut→clip at 0/500/1000",
        hypothesis_expected="phased approach for SOTA push",
    ),
    ContinuationCondition(
        name="imagenet_cont_autopilot",
        description="ImageNet MobileNetV2 — autopilot continuation",
        task="imagenet_mobilenet",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=2000,
        hypothesis_condition="autopilot auto with finish_strong",
        hypothesis_expected="autopilot optimizes beyond pretrained baseline",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# CLIP ViT-B/16 on COCO Captions (VLM)
# Defaults: AdamW lr=1e-6, wd=0.1, CLIP ViT-B/16 pretrained
# ═══════════════════════════════════════════════════════════════════════

CLIP_COCO_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="clip_coco_baseline",
        description="CLIP ViT-B/16 COCO fine-tuning — default config",
        demo="clip_coco",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="CLIP ViT-B/16 pretrained, AdamW lr=1e-6, wd=0.1",
        hypothesis_expected="contrastive fine-tuning improves COCO retrieval",
    ),
    EvalCondition(
        name="clip_coco_high_lr_auto",
        description="CLIP — LR 10x, autopilot recovery",
        demo="clip_coco",
        initial_overrides={"opt": {"lr": 1e-5}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=1e-5 (10x), autopilot auto",
        hypothesis_expected="autopilot prevents VLM catastrophic forgetting",
    ),
    EvalCondition(
        name="clip_coco_divergent",
        description="CLIP — LR 100x, catastrophic forgetting risk",
        demo="clip_coco",
        initial_overrides={"opt": {"lr": 1e-4}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=1e-4 (100x), high forgetting risk",
        hypothesis_expected="autopilot fires emergency cuts, preserves pretrained knowledge",
    ),
    EvalCondition(
        name="clip_coco_high_wd",
        description="CLIP — WD 10x (wd=1.0)",
        demo="clip_coco",
        initial_overrides={"opt": {"weight_decay": 1.0}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=1.0 (10x), extreme regularization",
        hypothesis_expected="over-regularization detected and corrected",
    ),
    EvalCondition(
        name="clip_coco_no_auto",
        description="CLIP — LR 10x, no autopilot (control)",
        demo="clip_coco",
        initial_overrides={"opt": {"lr": 1e-5}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=1e-5 (10x), no autopilot",
        hypothesis_expected="unassisted — worse retrieval than autopilot",
    ),
]

CLIP_COCO_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    ContinuationCondition(
        name="clip_cont_lr_half",
        description="CLIP — halve LR from converged checkpoint",
        task="clip_coco",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="LR halved from best R@1 anchor",
        hypothesis_expected="stabilizes contrastive learning",
    ),
    ContinuationCondition(
        name="clip_cont_swa_tail",
        description="CLIP — SWA from converged checkpoint",
        task="clip_coco",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="SWA from best R@1 anchor",
        hypothesis_expected="SWA smooths VLM features",
    ),
    ContinuationCondition(
        name="clip_cont_swa_lr_half",
        description="CLIP — SWA + halved LR",
        task="clip_coco",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="SWA + LR halved for VLM",
        hypothesis_expected="combined smoothing for best retrieval",
    ),
    ContinuationCondition(
        name="clip_cont_ema_tail",
        description="CLIP — EMA from converged checkpoint",
        task="clip_coco",
        recipe_name="ema_tail",
        recipe_ops=[{"target": "ema_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="EMA from best R@1 anchor",
        hypothesis_expected="EMA shadow weights for VLM smoothing",
    ),
    ContinuationCondition(
        name="clip_cont_kitchen_sink",
        description="CLIP — maximal treatment",
        task="clip_coco",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 0.5},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="LR half + SWA + grad_clip=0.5",
        hypothesis_expected="combined for maximum VLM improvement",
    ),
    ContinuationCondition(
        name="clip_cont_cosine_anneal",
        description="CLIP — staged LR decay",
        task="clip_coco",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 300},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 600},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="staged LR decay for VLM",
        hypothesis_expected="cosine-like VLM convergence",
    ),
    ContinuationCondition(
        name="clip_cont_lr_tenth",
        description="CLIP — emergency LR 10%",
        task="clip_coco",
        recipe_name="lr_tenth",
        recipe_ops=[{"target": "lr", "value": 0.1, "relative": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="LR to 10% from best R@1 anchor",
        hypothesis_expected="hard deceleration for VLM precision",
    ),
    ContinuationCondition(
        name="clip_cont_wd_reduce",
        description="CLIP — halve weight decay",
        task="clip_coco",
        recipe_name="wd_reduce",
        recipe_ops=[{"target": "weight_decay", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="WD halved (0.05) from best R@1",
        hypothesis_expected="relaxed regularization for VLM late phase",
    ),
    ContinuationCondition(
        name="clip_cont_progressive_wd",
        description="CLIP — progressive WD reduction",
        task="clip_coco",
        recipe_name="progressive_wd_reduction",
        recipe_ops=[
            {"target": "weight_decay", "value": 0.75, "relative": True, "delay_steps": 0},
            {"target": "weight_decay", "value": 0.5, "relative": True, "delay_steps": 300},
            {"target": "weight_decay", "value": 0.25, "relative": True, "delay_steps": 600},
        ],
        resume_mode="full_resume",
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="progressive WD: 0.75×→0.5×→0.25×",
        hypothesis_expected="gradual regularization relaxation for VLM",
    ),
    ContinuationCondition(
        name="clip_cont_autopilot",
        description="CLIP — autopilot-guided continuation",
        task="clip_coco",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=1000,
        primary_metric="val_retrieval_r1",
        hypothesis_condition="autopilot auto with finish_strong for VLM",
        hypothesis_expected="autopilot finds optimal VLM interventions",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# ImageNet MobileNetV2 — Paper-faithful from-scratch (Sandler et al. 2018)
# RMSProp, lr=0.045, alpha=0.9, momentum=0.9, wd=4e-5, eps=1.0
# ExponentialLR gamma=0.98, batch=96, 300 epochs, target 72.0% top-1
# ═══════════════════════════════════════════════════════════════════════

IMAGENET_PAPER_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="imagenet_paper_baseline",
        description="ImageNet MobileNetV2 — exact paper recipe (RMSProp), no autopilot",
        demo="imagenet_mobilenetv2_paper",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="paper RMSProp (lr=0.045, alpha=0.9, eps=1.0), from scratch, 300 epochs",
        hypothesis_expected="reaches ~72.0% top-1 (paper Table 4)",
    ),
    EvalCondition(
        name="imagenet_paper_high_lr_auto",
        description="ImageNet paper recipe — LR 3× (0.135), autopilot recovery",
        demo="imagenet_mobilenetv2_paper",
        initial_overrides={"opt": {"lr": 0.135}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.135 (3× paper), RMSProp, autopilot auto",
        hypothesis_expected="autopilot stabilizes and recovers toward 72%",
    ),
    EvalCondition(
        name="imagenet_paper_divergent",
        description="ImageNet paper recipe — LR 10× (0.45), emergency stabilization",
        demo="imagenet_mobilenetv2_paper",
        initial_overrides={"opt": {"lr": 0.45}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.45 (10× paper), near-divergence with RMSProp",
        hypothesis_expected="autopilot prevents NaN/divergence",
    ),
    EvalCondition(
        name="imagenet_paper_high_wd",
        description="ImageNet paper recipe — WD 100× (0.004), over-regularization",
        demo="imagenet_mobilenetv2_paper",
        initial_overrides={"opt": {"weight_decay": 0.004}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="wd=0.004 (100× paper), over-regularization",
        hypothesis_expected="plateau rules detect and correct regularization",
    ),
    EvalCondition(
        name="imagenet_paper_no_auto",
        description="ImageNet paper recipe — LR 3×, no autopilot (control)",
        demo="imagenet_mobilenetv2_paper",
        initial_overrides={"opt": {"lr": 0.135}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.135 (3×), no autopilot — control for comparison",
        hypothesis_expected="degraded accuracy without autopilot intervention",
    ),
]


IMAGENET_PAPER_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    ContinuationCondition(
        name="imagenet_paper_cont_lr_half",
        description="ImageNet paper — halve LR from converged",
        task="imagenet_mobilenetv2_paper",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=13345,  # ~1 epoch
        hypothesis_condition="LR halved from best-val, RMSProp",
        hypothesis_expected="tighter convergence beyond 72.0%",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_swa_tail",
        description="ImageNet paper — SWA from converged",
        task="imagenet_mobilenetv2_paper",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=13345,
        hypothesis_condition="SWA from best-val anchor, RMSProp",
        hypothesis_expected="SWA pushes beyond 72.0%",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_swa_lr_half",
        description="ImageNet paper — SWA + halved LR (canonical)",
        task="imagenet_mobilenetv2_paper",
        recipe_name="swa_lr_half",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.5, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=13345,
        hypothesis_condition="SWA + LR halved — canonical SWA recipe",
        hypothesis_expected="canonical SWA approach beats 72.0%",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_cosine_anneal",
        description="ImageNet paper — staged LR decay",
        task="imagenet_mobilenetv2_paper",
        recipe_name="cosine_anneal_tail",
        recipe_ops=[
            {"target": "lr", "value": 0.7, "relative": True, "delay_steps": 0},
            {"target": "lr", "value": 0.4, "relative": True, "delay_steps": 4000},
            {"target": "lr", "value": 0.15, "relative": True, "delay_steps": 8000},
        ],
        resume_mode="full_resume",
        extra_steps=13345,
        hypothesis_condition="staged LR decay: 0.7×→0.4×→0.15× for RMSProp",
        hypothesis_expected="cosine-like decay for precision",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_kitchen_sink",
        description="ImageNet paper — maximal treatment",
        task="imagenet_mobilenetv2_paper",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 2.0},
        ],
        resume_mode="full_resume",
        extra_steps=13345,
        hypothesis_condition="LR half + SWA + grad_clip=2.0",
        hypothesis_expected="combined for maximum improvement beyond 72.0%",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_long_swa",
        description="ImageNet paper — 10 epochs SWA tail",
        task="imagenet_mobilenetv2_paper",
        recipe_name="long_swa",
        recipe_ops=[
            {"target": "swa_enabled", "value": True},
            {"target": "lr", "value": 0.3, "relative": True},
        ],
        resume_mode="full_resume",
        extra_steps=133450,  # ~10 epochs
        hypothesis_condition="10 epochs SWA + LR 30%, long tail optimization",
        hypothesis_expected="extended SWA for maximum precision",
    ),
    ContinuationCondition(
        name="imagenet_paper_cont_autopilot",
        description="ImageNet paper — autopilot continuation",
        task="imagenet_mobilenetv2_paper",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=13345,
        hypothesis_condition="autopilot auto with finish_strong",
        hypothesis_expected="autopilot optimizes beyond 72.0%",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# COCO SSDLite + MobileNetV2 Detection (Paper Table 6)
# SGD lr=0.01, momentum=0.9, wd=4e-5, CosineAnnealing, 200 epochs
# Target: 22.1 mAP (paper Table 6)
# ═══════════════════════════════════════════════════════════════════════

COCO_DETECTION_CONDITIONS: List[EvalCondition] = [
    EvalCondition(
        name="coco_detection_baseline",
        description="COCO SSDLite + MobileNetV2 — paper recipe, target 22.1 mAP",
        demo="coco_detection",
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="SSDLite MobileNetV2, SGD lr=0.01, 320×320, 200 epochs",
        hypothesis_expected="reaches ~22.1 mAP (paper Table 6)",
    ),
    EvalCondition(
        name="coco_detection_high_lr_auto",
        description="COCO detection — LR 3×, autopilot recovery",
        demo="coco_detection",
        initial_overrides={"opt": {"lr": 0.03}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.03 (3×), autopilot auto",
        hypothesis_expected="autopilot stabilizes detection training",
    ),
    EvalCondition(
        name="coco_detection_divergent",
        description="COCO detection — LR 10×, emergency stabilization",
        demo="coco_detection",
        initial_overrides={"opt": {"lr": 0.1}},
        use_recipe=False,
        autopilot="auto",
        policy_packs=["stability_basics"],
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.1 (10×), near-divergence for detection",
        hypothesis_expected="autopilot prevents NaN/divergence",
    ),
    EvalCondition(
        name="coco_detection_no_auto",
        description="COCO detection — LR 3×, no autopilot (control)",
        demo="coco_detection",
        initial_overrides={"opt": {"lr": 0.03}},
        use_recipe=False,
        autopilot="off",
        step_delay=0.0,
        seed=42,
        hypothesis_condition="lr=0.03 (3×), no autopilot — control",
        hypothesis_expected="unassisted degradation baseline",
    ),
]


COCO_DETECTION_CONTINUATION_CONDITIONS: List[ContinuationCondition] = [
    ContinuationCondition(
        name="coco_det_cont_lr_half",
        description="COCO detection — halve LR",
        task="coco_detection",
        recipe_name="lr_half",
        recipe_ops=[{"target": "lr", "value": 0.5, "relative": True}],
        resume_mode="full_resume",
        extra_steps=5000,
        primary_metric="val_mAP",
        hypothesis_condition="LR halved from best-mAP",
        hypothesis_expected="tighter convergence beyond 22.1 mAP",
    ),
    ContinuationCondition(
        name="coco_det_cont_swa_tail",
        description="COCO detection — SWA",
        task="coco_detection",
        recipe_name="swa_tail",
        recipe_ops=[{"target": "swa_enabled", "value": True}],
        resume_mode="full_resume",
        extra_steps=5000,
        primary_metric="val_mAP",
        hypothesis_condition="SWA from best-mAP anchor",
        hypothesis_expected="SWA smoothing improves detection mAP",
    ),
    ContinuationCondition(
        name="coco_det_cont_kitchen_sink",
        description="COCO detection — maximal treatment",
        task="coco_detection",
        recipe_name="kitchen_sink",
        recipe_ops=[
            {"target": "lr", "value": 0.5, "relative": True},
            {"target": "swa_enabled", "value": True},
            {"target": "grad_clip", "value": 5.0},
        ],
        resume_mode="full_resume",
        extra_steps=5000,
        primary_metric="val_mAP",
        hypothesis_condition="LR half + SWA + grad_clip=5.0",
        hypothesis_expected="combined for mAP improvement",
    ),
    ContinuationCondition(
        name="coco_det_cont_autopilot",
        description="COCO detection — autopilot continuation",
        task="coco_detection",
        recipe_name="autopilot_guided",
        recipe_ops=[],
        resume_mode="full_resume",
        autopilot="auto",
        policy_packs=["stability_basics", "plateau_recovery", "finish_strong"],
        extra_steps=5000,
        primary_metric="val_mAP",
        hypothesis_condition="autopilot auto with finish_strong",
        hypothesis_expected="autopilot optimizes beyond 22.1 mAP",
    ),
]


ALL_CONTINUATION_CONDITIONS = (
    MNIST_CONTINUATION_CONDITIONS + MNIST_NEW_CONTINUATION_CONDITIONS +
    CIFAR10_CONTINUATION_CONDITIONS + CIFAR10_NEW_CONTINUATION_CONDITIONS +
    COCO_MOBILENET_CONTINUATION_CONDITIONS +
    IMAGENET_MOBILENET_CONTINUATION_CONDITIONS +
    IMAGENET_PAPER_CONTINUATION_CONDITIONS +
    COCO_DETECTION_CONTINUATION_CONDITIONS +
    CLIP_COCO_CONTINUATION_CONDITIONS
)


# ═══════════════════════════════════════════════════════════════════════
# User-defined conditions from YAML
# ═══════════════════════════════════════════════════════════════════════

def load_conditions_yaml(path: str) -> List[EvalCondition]:
    """Load custom evaluation conditions from a YAML file.

    Format::

        conditions:
          - name: my_experiment
            description: "Test high LR on MNIST"
            demo: mnist              # mnist | cifar10 | golden | finetune | simple
            autopilot: auto          # off | auto | suggest
            policy_packs:
              - stability_basics
            initial_overrides:
              opt:
                lr: 0.1
            max_steps: 500
            seed: 42
            hypothesis_condition: "lr=0.1 on MNIST"
            hypothesis_expected: "autopilot recovers to >95%"
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)

    raw = data if isinstance(data, list) else data.get("conditions", [])
    results: List[EvalCondition] = []
    for entry in raw:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        results.append(EvalCondition(
            name=entry["name"],
            description=entry.get("description", entry["name"]),
            demo=entry.get("demo", "mnist"),
            initial_overrides=entry.get("initial_overrides", {}),
            use_recipe=entry.get("use_recipe", False),
            autopilot=entry.get("autopilot", "off"),
            policy_packs=entry.get("policy_packs", []),
            nn_mode=entry.get("nn_mode", False),
            scheduled_commands={
                int(k): v for k, v in entry.get("scheduled_commands", {}).items()
            },
            max_steps=entry.get("max_steps"),
            step_delay=entry.get("step_delay", 0.0),
            seed=entry.get("seed"),
            hypothesis_condition=entry.get("hypothesis_condition", ""),
            hypothesis_expected=entry.get("hypothesis_expected", ""),
        ))
    return results


# Synthetic demo conditions (kept for backward compat / quick testing)
SYNTHETIC_CONDITIONS = GOLDEN_CONDITIONS + FINETUNE_CONDITIONS + SIMPLE_CONDITIONS

# Real PyTorch conditions — the default eval suite
REAL_CONDITIONS = (
    MNIST_CONDITIONS + CIFAR10_CONDITIONS + COCO_MOBILENET_CONDITIONS +
    IMAGENET_MOBILENET_CONDITIONS + IMAGENET_PAPER_CONDITIONS +
    COCO_DETECTION_CONDITIONS + CLIP_COCO_CONDITIONS
)

# Default: real conditions only. Use --demo golden/finetune/simple for synthetic.
ALL_CONDITIONS = REAL_CONDITIONS
ALL_WITH_REAL = REAL_CONDITIONS  # alias for backward compat
ALL_WITH_SYNTHETIC = REAL_CONDITIONS + SYNTHETIC_CONDITIONS
