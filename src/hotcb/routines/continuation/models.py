"""Data models for the continuation tuning routine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ResumeMode(Enum):
    """How to restore from a checkpoint."""

    FULL_RESUME = "full_resume"          # model + optimizer + scheduler + counters
    RESET_SCHEDULER = "reset_scheduler"  # model + optimizer, fresh scheduler
    WEIGHTS_ONLY = "weights_only"        # model weights only, fresh optimizer/scheduler


class AnchorReason(Enum):
    """Why this checkpoint was chosen as an anchor."""

    BEFORE_PLATEAU = "before_plateau"
    BEFORE_LAST_LR_DROP = "before_last_lr_drop"
    BEST_VAL_MINUS_K = "best_val_minus_k_steps"
    END_MINUS_K = "end_minus_k_steps"
    MANUAL = "manual"


class BranchStatus(Enum):
    """Final status of a continuation branch."""

    PENDING = "pending"
    RUNNING = "running"
    WON = "won"
    LOST = "lost"
    UNSTABLE = "unstable"
    INCONCLUSIVE = "inconclusive"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REJECTED_GUARDRAIL = "rejected_guardrail"
    RESUME_FAILED = "resume_failed"


@dataclass
class MutationOp:
    """A single mutation to apply via hotcb."""

    target: str       # e.g. "optimizer.lr", "loss.aux_weight", "swa_enabled"
    value: Any        # absolute value or multiplier
    relative: bool = False  # if True, value is a multiplier (e.g. 0.5 = halve)
    delay_steps: int = 0    # steps after anchor before this op fires (0 = immediate)


@dataclass
class MutationRecipe:
    """A named set of mutations to apply as a branch recipe."""

    name: str
    description: str = ""
    ops: List[MutationOp] = field(default_factory=list)
    safety_class: str = "safe"  # safe, moderate, experimental


@dataclass
class AnchorSpec:
    """A checkpoint from which branches start."""

    anchor_id: str
    checkpoint_path: str
    step: int
    epoch: int = 0
    reason: AnchorReason = AnchorReason.MANUAL
    metrics_at_anchor: Dict[str, float] = field(default_factory=dict)


@dataclass
class BranchSpec:
    """A planned continuation branch (before execution)."""

    branch_id: str
    anchor_id: str
    recipe: MutationRecipe
    resume_mode: ResumeMode
    extra_steps: int = 0  # how many additional training steps


@dataclass
class BranchResult:
    """Result of executing a continuation branch."""

    branch_id: str
    anchor_id: str
    recipe_name: str
    resume_mode: str
    status: BranchStatus = BranchStatus.PENDING

    # Metrics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison vs base
    primary_delta: float = 0.0       # delta on primary metric vs base
    primary_delta_pct: float = 0.0   # percentage delta

    # Execution info
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    run_dir: str = ""
    applied_mutations: List[Dict[str, Any]] = field(default_factory=list)
    autopilot_actions: List[Dict[str, Any]] = field(default_factory=list)
    guardrail_violations: List[str] = field(default_factory=list)

    # Research
    research_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveSpec:
    """What to optimize."""

    primary_metric: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    min_improvement_delta: float = 0.0  # minimum delta to accept a branch


@dataclass
class GuardrailSpec:
    """Safety guardrails for branch acceptance."""

    forbid_nan: bool = True
    max_relative_step_time_increase: float = 0.25
    secondary_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g. {"val_loss": {"mode": "min", "max_regression": 0.01}}


@dataclass
class BudgetSpec:
    """Budget constraints for the continuation routine."""

    max_anchors: int = 2
    branches_per_anchor: int = 4
    max_extra_steps: int = 500    # extra steps per branch
    eval_interval: int = 50       # evaluate every N steps
    early_stop_patience: int = 4  # patience in eval intervals
    max_recursive_depth: int = 1  # only one level of refinement in V1


@dataclass
class ContinuationConfig:
    """Top-level configuration for a continuation routine."""

    # Base run info
    base_run_dir: str = ""
    base_final_metrics: Dict[str, float] = field(default_factory=dict)
    base_best_metrics: Dict[str, float] = field(default_factory=dict)

    # Training function reference
    task: str = "mnist"  # "mnist" or "cifar10"

    # Objective
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    guardrails: GuardrailSpec = field(default_factory=GuardrailSpec)

    # Budget
    budget: BudgetSpec = field(default_factory=BudgetSpec)

    # Anchors
    anchors: List[AnchorSpec] = field(default_factory=list)

    # Recipes to try
    recipes: List[MutationRecipe] = field(default_factory=list)

    # Resume modes to try (combined with recipes → branches)
    resume_modes: List[ResumeMode] = field(default_factory=lambda: [ResumeMode.FULL_RESUME])

    # Autopilot during continuation branches
    autopilot: str = "off"
    policy_packs: List[str] = field(default_factory=list)

    # Output
    output_dir: str = ""

    @property
    def total_branches(self) -> int:
        return len(self.anchors) * len(self.recipes) * len(self.resume_modes)


@dataclass
class RoutineResult:
    """Final result of the continuation routine."""

    config: ContinuationConfig = field(default_factory=ContinuationConfig)
    branches: List[BranchResult] = field(default_factory=list)
    winner: Optional[BranchResult] = None
    recommendation: str = "keep_base_run"
    elapsed_seconds: float = 0.0

    @property
    def leaderboard(self) -> List[BranchResult]:
        """Branches sorted by primary metric delta (best first)."""
        mode = self.config.objective.mode
        return sorted(
            self.branches,
            key=lambda b: b.primary_delta if mode == "max" else -b.primary_delta,
            reverse=True,
        )
