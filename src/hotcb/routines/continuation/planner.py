"""Continuation planner — generate branch specs from anchors x recipes x resume modes."""

from __future__ import annotations

from typing import List

from .models import (
    AnchorSpec,
    BranchSpec,
    BudgetSpec,
    ContinuationConfig,
    MutationRecipe,
    MutationOp,
    ResumeMode,
)


# ═══════════════════════════════════════════════════════════════════════
# Default recipes — the 6 safe late-stage mutations
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_RECIPES: List[MutationRecipe] = [
    MutationRecipe(
        name="lr_half",
        description="Halve learning rate from anchor onward",
        ops=[MutationOp(target="lr", value=0.5, relative=True)],
    ),
    MutationRecipe(
        name="lr_quarter",
        description="Quarter learning rate — more aggressive tail stabilization",
        ops=[MutationOp(target="lr", value=0.25, relative=True)],
    ),
    MutationRecipe(
        name="swa_tail",
        description="Enable Stochastic Weight Averaging in tail phase",
        ops=[MutationOp(target="swa_enabled", value=True)],
    ),
    MutationRecipe(
        name="ema_tail",
        description="Enable Exponential Moving Average in tail phase",
        ops=[MutationOp(target="ema_enabled", value=True)],
    ),
    MutationRecipe(
        name="grad_clip_tight",
        description="Tighten gradient clipping to 0.5",
        ops=[MutationOp(target="grad_clip", value=0.5)],
    ),
    MutationRecipe(
        name="wd_reduce",
        description="Reduce weight decay by half for late-stage fine convergence",
        ops=[MutationOp(target="weight_decay", value=0.5, relative=True)],
    ),
]


AGGRESSIVE_RECIPES: List[MutationRecipe] = [
    MutationRecipe(
        name="lr_tenth",
        description="Emergency LR reduction to 10% — hard deceleration",
        ops=[MutationOp(target="lr", value=0.1, relative=True)],
        safety_class="experimental",
    ),
    MutationRecipe(
        name="momentum_reset",
        description="LR to 30% with weights-only resume to escape bad momentum",
        ops=[MutationOp(target="lr", value=0.3, relative=True)],
        safety_class="experimental",
    ),
    MutationRecipe(
        name="kitchen_sink_aggressive",
        description="Maximum intervention: lr×0.25 + SWA + grad_clip=0.5 + wd×0.25",
        ops=[
            MutationOp(target="lr", value=0.25, relative=True),
            MutationOp(target="swa_enabled", value=True),
            MutationOp(target="grad_clip", value=0.5),
            MutationOp(target="weight_decay", value=0.25, relative=True),
        ],
        safety_class="experimental",
    ),
]


COMBO_RECIPES: List[MutationRecipe] = [
    MutationRecipe(
        name="swa_grad_clip",
        description="SWA + gradient clipping — smoothing with stability",
        ops=[
            MutationOp(target="swa_enabled", value=True),
            MutationOp(target="grad_clip", value=1.0),
        ],
        safety_class="moderate",
    ),
    MutationRecipe(
        name="ema_lr_quarter",
        description="EMA + quartered LR — conservative smoothed tail",
        ops=[
            MutationOp(target="ema_enabled", value=True),
            MutationOp(target="lr", value=0.25, relative=True),
        ],
        safety_class="moderate",
    ),
    MutationRecipe(
        name="swa_wd_reduce",
        description="SWA + halved weight decay — averaging with relaxed regularization",
        ops=[
            MutationOp(target="swa_enabled", value=True),
            MutationOp(target="weight_decay", value=0.5, relative=True),
        ],
        safety_class="moderate",
    ),
]


MULTI_STAGE_RECIPES: List[MutationRecipe] = [
    MutationRecipe(
        name="cosine_anneal_tail",
        description="Staged LR decay: 0.7→0.4→0.15 mimicking cosine tail",
        ops=[
            MutationOp(target="lr", value=0.7, relative=True, delay_steps=0),
            MutationOp(target="lr", value=0.4, relative=True, delay_steps=150),
            MutationOp(target="lr", value=0.15, relative=True, delay_steps=300),
        ],
        safety_class="moderate",
    ),
    MutationRecipe(
        name="staged_swa_lr",
        description="SWA first, then LR cut, then grad clip — phased treatment",
        ops=[
            MutationOp(target="swa_enabled", value=True, delay_steps=0),
            MutationOp(target="lr", value=0.5, relative=True, delay_steps=100),
            MutationOp(target="grad_clip", value=1.0, delay_steps=250),
        ],
        safety_class="moderate",
    ),
    MutationRecipe(
        name="progressive_wd_reduction",
        description="Gradually relax weight decay: 0.75→0.5→0.25 of base",
        ops=[
            MutationOp(target="weight_decay", value=0.75, relative=True, delay_steps=0),
            MutationOp(target="weight_decay", value=0.5, relative=True, delay_steps=150),
            MutationOp(target="weight_decay", value=0.25, relative=True, delay_steps=300),
        ],
        safety_class="moderate",
    ),
]


ALL_RECIPES = DEFAULT_RECIPES + COMBO_RECIPES + AGGRESSIVE_RECIPES + MULTI_STAGE_RECIPES


class ContinuationPlanner:
    """Generate branch specs from a continuation config."""

    def __init__(self, config: ContinuationConfig):
        self._config = config

    def plan(self) -> List[BranchSpec]:
        """Generate all branch specs: anchors x recipes x resume_modes.

        Respects budget.branches_per_anchor — if the cross-product exceeds
        the budget, recipes are prioritized over resume modes.
        """
        branches: List[BranchSpec] = []
        budget = self._config.budget
        recipes = self._config.recipes or DEFAULT_RECIPES
        resume_modes = self._config.resume_modes or [ResumeMode.FULL_RESUME]

        for anchor in self._config.anchors:
            anchor_branches: List[BranchSpec] = []
            branch_idx = 0

            for recipe in recipes:
                for mode in resume_modes:
                    branch_id = f"{anchor.anchor_id}__{recipe.name}__{mode.value}_{branch_idx}"
                    anchor_branches.append(BranchSpec(
                        branch_id=branch_id,
                        anchor_id=anchor.anchor_id,
                        recipe=recipe,
                        resume_mode=mode,
                        extra_steps=budget.max_extra_steps,
                    ))
                    branch_idx += 1

            # Trim to budget
            branches.extend(anchor_branches[:budget.branches_per_anchor])

        return branches

    def plan_with_defaults(self) -> List[BranchSpec]:
        """Plan using default recipes if none specified."""
        if not self._config.recipes:
            self._config.recipes = list(DEFAULT_RECIPES)
        return self.plan()
