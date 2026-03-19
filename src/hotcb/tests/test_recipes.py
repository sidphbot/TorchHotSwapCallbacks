"""Tests for continuation recipes — all lists parse, delay_steps works, recipe counts."""
from __future__ import annotations

import pytest

from hotcb.routines.continuation.models import MutationOp, MutationRecipe
from hotcb.routines.continuation.planner import (
    DEFAULT_RECIPES,
    AGGRESSIVE_RECIPES,
    COMBO_RECIPES,
    MULTI_STAGE_RECIPES,
    ALL_RECIPES,
    ContinuationPlanner,
)
from hotcb.routines.continuation.models import ContinuationConfig, AnchorSpec, BudgetSpec


# ─── Recipe list validation ──────────────────────────────────────────

def test_default_recipes_count():
    assert len(DEFAULT_RECIPES) == 6

def test_aggressive_recipes_count():
    assert len(AGGRESSIVE_RECIPES) == 3

def test_combo_recipes_count():
    assert len(COMBO_RECIPES) == 3

def test_multi_stage_recipes_count():
    assert len(MULTI_STAGE_RECIPES) == 3

def test_all_recipes_count():
    assert len(ALL_RECIPES) == 15

def test_all_recipes_contains_all_lists():
    names = {r.name for r in ALL_RECIPES}
    for r in DEFAULT_RECIPES + COMBO_RECIPES + AGGRESSIVE_RECIPES + MULTI_STAGE_RECIPES:
        assert r.name in names, f"Missing recipe: {r.name}"


# ─── Recipe structure validation ─────────────────────────────────────

def test_all_recipes_have_names():
    for r in ALL_RECIPES:
        assert r.name, f"Recipe missing name: {r}"
        assert r.ops, f"Recipe {r.name} has no ops"

def test_all_recipes_have_valid_safety_class():
    valid = {"safe", "moderate", "experimental"}
    for r in ALL_RECIPES:
        assert r.safety_class in valid, f"Invalid safety_class for {r.name}: {r.safety_class}"

def test_default_recipes_are_safe():
    for r in DEFAULT_RECIPES:
        assert r.safety_class == "safe", f"{r.name} should be safe"

def test_aggressive_recipes_are_experimental():
    for r in AGGRESSIVE_RECIPES:
        assert r.safety_class == "experimental", f"{r.name} should be experimental"

def test_combo_recipes_are_moderate():
    for r in COMBO_RECIPES:
        assert r.safety_class == "moderate", f"{r.name} should be moderate"

def test_multi_stage_recipes_are_moderate():
    for r in MULTI_STAGE_RECIPES:
        assert r.safety_class == "moderate", f"{r.name} should be moderate"


# ─── Unique names ────────────────────────────────────────────────────

def test_recipe_names_unique():
    names = [r.name for r in ALL_RECIPES]
    assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# ─── delay_steps ─────────────────────────────────────────────────────

def test_delay_steps_default_zero():
    op = MutationOp(target="lr", value=0.5, relative=True)
    assert op.delay_steps == 0

def test_delay_steps_explicit():
    op = MutationOp(target="lr", value=0.5, relative=True, delay_steps=150)
    assert op.delay_steps == 150

def test_multi_stage_recipes_have_delay_steps():
    for recipe in MULTI_STAGE_RECIPES:
        delays = [op.delay_steps for op in recipe.ops]
        assert any(d > 0 for d in delays), f"Recipe {recipe.name} has no delayed ops"
        # First op should be at step 0
        assert delays[0] == 0, f"Recipe {recipe.name} first op should be delay_steps=0"

def test_cosine_anneal_stages():
    recipe = [r for r in MULTI_STAGE_RECIPES if r.name == "cosine_anneal_tail"][0]
    delays = [op.delay_steps for op in recipe.ops]
    assert delays == [0, 150, 300]
    # All ops target lr
    for op in recipe.ops:
        assert op.target == "lr"
        assert op.relative is True

def test_staged_swa_lr_stages():
    recipe = [r for r in MULTI_STAGE_RECIPES if r.name == "staged_swa_lr"][0]
    delays = [op.delay_steps for op in recipe.ops]
    assert delays == [0, 100, 250]
    # First op is swa
    assert recipe.ops[0].target == "swa_enabled"

def test_progressive_wd_reduction_stages():
    recipe = [r for r in MULTI_STAGE_RECIPES if r.name == "progressive_wd_reduction"][0]
    delays = [op.delay_steps for op in recipe.ops]
    assert delays == [0, 150, 300]
    for op in recipe.ops:
        assert op.target == "weight_decay"


# ─── MutationOp backward compatibility ──────────────────────────────

def test_mutation_op_without_delay_steps():
    """Ensure old-style MutationOp without delay_steps still works."""
    op = MutationOp(target="lr", value=0.5)
    assert op.delay_steps == 0
    assert op.relative is False

def test_mutation_op_serialization():
    """Ensure MutationOp with delay_steps can be converted to dict."""
    op = MutationOp(target="lr", value=0.5, relative=True, delay_steps=100)
    from dataclasses import asdict
    d = asdict(op)
    assert d["delay_steps"] == 100
    assert d["target"] == "lr"
    assert d["relative"] is True


# ─── Planner with new recipes ───────────────────────────────────────

def test_planner_with_all_recipes():
    config = ContinuationConfig(
        anchors=[AnchorSpec(anchor_id="a0", checkpoint_path="/tmp/ckpt.pt", step=1000)],
        recipes=list(ALL_RECIPES),
        budget=BudgetSpec(branches_per_anchor=20),
    )
    planner = ContinuationPlanner(config)
    branches = planner.plan()
    assert len(branches) == 15  # 15 recipes × 1 anchor × 1 resume mode

def test_planner_trims_to_budget():
    config = ContinuationConfig(
        anchors=[AnchorSpec(anchor_id="a0", checkpoint_path="/tmp/ckpt.pt", step=1000)],
        recipes=list(ALL_RECIPES),
        budget=BudgetSpec(branches_per_anchor=5),
    )
    planner = ContinuationPlanner(config)
    branches = planner.plan()
    assert len(branches) == 5
