"""Tests for paper-faithful MobileNetV2 training + RMSProp actuators + COCO detection."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════
# RMSProp actuator tests
# ═══════════════════════════════════════════════════════════════════════


class _FakeRMSPropOptimizer:
    """Mimics RMSProp param_groups for actuator tests."""

    def __init__(self):
        self.param_groups = [{
            "lr": 0.045,
            "alpha": 0.9,
            "momentum": 0.9,
            "weight_decay": 0.00004,
            "eps": 1.0,
        }]


class _FakeSGDOptimizer:
    """Mimics SGD param_groups (no alpha)."""

    def __init__(self):
        self.param_groups = [{
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 4e-5,
        }]


class _FakeAdamOptimizer:
    """Mimics Adam param_groups (has betas, no alpha/momentum)."""

    def __init__(self):
        self.param_groups = [{
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.999),
        }]


def test_rmsprop_actuators_include_alpha_and_momentum():
    """optimizer_actuators() creates alpha + momentum for RMSProp."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeRMSPropOptimizer()
    acts = optimizer_actuators(opt)
    keys = [a.param_key for a in acts]
    assert "alpha" in keys, f"Missing alpha actuator, got {keys}"
    assert "momentum" in keys, f"Missing momentum actuator, got {keys}"
    assert "lr" in keys
    assert "weight_decay" in keys


def test_rmsprop_alpha_actuator_bounds():
    """Alpha actuator has correct bounds and initial value."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeRMSPropOptimizer()
    acts = optimizer_actuators(opt)
    alpha_act = [a for a in acts if a.param_key == "alpha"][0]
    assert alpha_act.current_value == 0.9
    assert alpha_act.min_value == 0.8
    assert alpha_act.max_value == 0.999
    assert alpha_act.group == "optimizer"


def test_rmsprop_alpha_apply_mutates_optimizer():
    """Applying alpha actuator mutates all param_groups."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeRMSPropOptimizer()
    acts = optimizer_actuators(opt)
    alpha_act = [a for a in acts if a.param_key == "alpha"][0]
    result = alpha_act.apply_fn(0.95, {})
    assert result.success
    assert opt.param_groups[0]["alpha"] == 0.95


def test_rmsprop_momentum_apply_mutates_optimizer():
    """Applying momentum actuator mutates all param_groups."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeRMSPropOptimizer()
    acts = optimizer_actuators(opt)
    mom_act = [a for a in acts if a.param_key == "momentum"][0]
    result = mom_act.apply_fn(0.8, {})
    assert result.success
    assert opt.param_groups[0]["momentum"] == 0.8


def test_sgd_actuators_include_momentum_but_not_alpha():
    """SGD has momentum but not alpha."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeSGDOptimizer()
    acts = optimizer_actuators(opt)
    keys = [a.param_key for a in acts]
    assert "momentum" in keys
    assert "alpha" not in keys


def test_adam_actuators_have_betas_not_alpha_momentum():
    """Adam has betas but not alpha/momentum."""
    from hotcb.actuators import optimizer_actuators
    opt = _FakeAdamOptimizer()
    acts = optimizer_actuators(opt)
    keys = [a.param_key for a in acts]
    assert "betas" in keys
    assert "alpha" not in keys
    assert "momentum" not in keys


def test_zero_momentum_not_exposed():
    """SGD with momentum=0 should not expose momentum actuator."""
    from hotcb.actuators import optimizer_actuators

    class _ZeroMom:
        param_groups = [{"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0}]

    acts = optimizer_actuators(_ZeroMom())
    keys = [a.param_key for a in acts]
    assert "momentum" not in keys


# ═══════════════════════════════════════════════════════════════════════
# Paper hyperparameter tests
# ═══════════════════════════════════════════════════════════════════════


def test_paper_hyperparameter_constants():
    """Verify paper constants match Sandler et al. 2018 §6.1."""
    from hotcb.eval.tasks_imagenet import (
        PAPER_LR, PAPER_ALPHA, PAPER_MOMENTUM, PAPER_WD,
        PAPER_EPS, PAPER_GAMMA, PAPER_BATCH_SIZE, PAPER_EPOCHS,
    )
    assert PAPER_LR == 0.045
    assert PAPER_ALPHA == 0.9
    assert PAPER_MOMENTUM == 0.9
    assert PAPER_WD == 0.00004
    assert PAPER_EPS == 1.0
    assert PAPER_GAMMA == 0.98
    assert PAPER_BATCH_SIZE == 96
    assert PAPER_EPOCHS == 300


def test_build_model_from_scratch():
    """_build_imagenet_model(pretrained=False) returns random init."""
    from hotcb.eval.tasks_imagenet import _build_imagenet_model
    model = _build_imagenet_model(pretrained=False)
    # Check it's a MobileNetV2 with correct output classes
    assert hasattr(model, "classifier")
    assert model.classifier[1].out_features == 1000


def test_build_model_pretrained():
    """_build_imagenet_model(pretrained=True) loads V1 weights."""
    from hotcb.eval.tasks_imagenet import _build_imagenet_model
    model = _build_imagenet_model(pretrained=True)
    assert hasattr(model, "features")
    assert model.classifier[1].out_features == 1000


# ═══════════════════════════════════════════════════════════════════════
# Condition definition tests
# ═══════════════════════════════════════════════════════════════════════


def test_imagenet_paper_conditions_exist():
    """IMAGENET_PAPER_CONDITIONS has correct conditions."""
    from hotcb.eval.conditions import IMAGENET_PAPER_CONDITIONS
    names = [c.name for c in IMAGENET_PAPER_CONDITIONS]
    assert "imagenet_paper_baseline" in names
    assert "imagenet_paper_high_lr_auto" in names
    assert "imagenet_paper_divergent" in names
    assert "imagenet_paper_high_wd" in names
    assert "imagenet_paper_no_auto" in names


def test_imagenet_paper_baseline_config():
    """Baseline condition has correct settings."""
    from hotcb.eval.conditions import IMAGENET_PAPER_CONDITIONS
    baseline = [c for c in IMAGENET_PAPER_CONDITIONS if c.name == "imagenet_paper_baseline"][0]
    assert baseline.demo == "imagenet_mobilenetv2_paper"
    assert baseline.autopilot == "off"
    assert baseline.use_recipe is False


def test_imagenet_paper_high_lr_uses_3x():
    """High LR condition overrides to 0.135 (3× paper 0.045)."""
    from hotcb.eval.conditions import IMAGENET_PAPER_CONDITIONS
    cond = [c for c in IMAGENET_PAPER_CONDITIONS if c.name == "imagenet_paper_high_lr_auto"][0]
    assert cond.initial_overrides["opt"]["lr"] == 0.135


def test_imagenet_paper_continuation_conditions_exist():
    """IMAGENET_PAPER_CONTINUATION_CONDITIONS has expected conditions."""
    from hotcb.eval.conditions import IMAGENET_PAPER_CONTINUATION_CONDITIONS
    names = [c.name for c in IMAGENET_PAPER_CONTINUATION_CONDITIONS]
    assert "imagenet_paper_cont_lr_half" in names
    assert "imagenet_paper_cont_swa_tail" in names
    assert "imagenet_paper_cont_autopilot" in names


def test_imagenet_paper_continuation_task_name():
    """Continuation conditions reference correct task."""
    from hotcb.eval.conditions import IMAGENET_PAPER_CONTINUATION_CONDITIONS
    for c in IMAGENET_PAPER_CONTINUATION_CONDITIONS:
        assert c.task == "imagenet_mobilenetv2_paper"


def test_coco_detection_conditions_exist():
    """COCO_DETECTION_CONDITIONS has expected conditions."""
    from hotcb.eval.conditions import COCO_DETECTION_CONDITIONS
    names = [c.name for c in COCO_DETECTION_CONDITIONS]
    assert "coco_detection_baseline" in names
    assert "coco_detection_high_lr_auto" in names
    assert "coco_detection_divergent" in names
    assert "coco_detection_no_auto" in names


def test_coco_detection_continuation_conditions_exist():
    """COCO_DETECTION_CONTINUATION_CONDITIONS has expected conditions."""
    from hotcb.eval.conditions import COCO_DETECTION_CONTINUATION_CONDITIONS
    names = [c.name for c in COCO_DETECTION_CONTINUATION_CONDITIONS]
    assert "coco_det_cont_lr_half" in names
    assert "coco_det_cont_swa_tail" in names
    assert "coco_det_cont_autopilot" in names


def test_all_conditions_includes_new():
    """ALL_CONDITIONS includes new condition lists."""
    from hotcb.eval.conditions import (
        ALL_CONDITIONS, IMAGENET_PAPER_CONDITIONS, COCO_DETECTION_CONDITIONS,
    )
    all_names = {c.name for c in ALL_CONDITIONS}
    for c in IMAGENET_PAPER_CONDITIONS:
        assert c.name in all_names, f"Missing {c.name} in ALL_CONDITIONS"
    for c in COCO_DETECTION_CONDITIONS:
        assert c.name in all_names, f"Missing {c.name} in ALL_CONDITIONS"


def test_all_continuation_includes_new():
    """ALL_CONTINUATION_CONDITIONS includes new lists."""
    from hotcb.eval.conditions import (
        ALL_CONTINUATION_CONDITIONS,
        IMAGENET_PAPER_CONTINUATION_CONDITIONS,
        COCO_DETECTION_CONTINUATION_CONDITIONS,
    )
    all_names = {c.name for c in ALL_CONTINUATION_CONDITIONS}
    for c in IMAGENET_PAPER_CONTINUATION_CONDITIONS:
        assert c.name in all_names
    for c in COCO_DETECTION_CONTINUATION_CONDITIONS:
        assert c.name in all_names


# ═══════════════════════════════════════════════════════════════════════
# Harness registration tests
# ═══════════════════════════════════════════════════════════════════════


def test_harness_recognizes_paper_demo():
    """_get_train_fn recognizes 'imagenet_mobilenetv2_paper'."""
    from hotcb.eval.harness import _get_train_fn
    fn, steps = _get_train_fn("imagenet_mobilenetv2_paper")
    assert fn is not None
    assert steps == 13345


def test_harness_recognizes_coco_detection_demo():
    """_get_train_fn recognizes 'coco_detection'."""
    from hotcb.eval.harness import _get_train_fn
    fn, steps = _get_train_fn("coco_detection")
    assert fn is not None
    assert steps == 5000


# ═══════════════════════════════════════════════════════════════════════
# Launcher registration tests
# ═══════════════════════════════════════════════════════════════════════


def test_launcher_recognizes_paper_task():
    """_get_continuation_train_fn recognizes 'imagenet_mobilenetv2_paper'."""
    from hotcb.routines.continuation.launcher import _get_continuation_train_fn
    fn = _get_continuation_train_fn("imagenet_mobilenetv2_paper")
    assert fn is not None


def test_launcher_recognizes_coco_detection_task():
    """_get_continuation_train_fn recognizes 'coco_detection'."""
    from hotcb.routines.continuation.launcher import _get_continuation_train_fn
    fn = _get_continuation_train_fn("coco_detection")
    assert fn is not None


# ═══════════════════════════════════════════════════════════════════════
# COCO Detection dataset tests
# ═══════════════════════════════════════════════════════════════════════


def test_coco_detection_dataset_class_exists():
    """COCODetection class is importable."""
    from hotcb.eval.datasets import COCODetection
    assert COCODetection is not None


# ═══════════════════════════════════════════════════════════════════════
# SSDLite model structure tests
# ═══════════════════════════════════════════════════════════════════════


def test_ssdlite_feature_extractor_exists():
    """SSDLiteFeatureExtractorMobileNetV2 is importable."""
    from hotcb.eval.tasks_coco_detection import SSDLiteFeatureExtractorMobileNetV2
    assert SSDLiteFeatureExtractorMobileNetV2 is not None


def test_ssdlite_detection_head_exists():
    """SSDLiteDetectionHead is importable."""
    from hotcb.eval.tasks_coco_detection import SSDLiteDetectionHead
    assert SSDLiteDetectionHead is not None


def test_detection_hyperparameter_constants():
    """Verify detection constants."""
    from hotcb.eval.tasks_coco_detection import (
        DETECTION_LR, DETECTION_MOMENTUM, DETECTION_WD,
        DETECTION_BATCH_SIZE, DETECTION_EPOCHS, DETECTION_INPUT_SIZE,
    )
    assert DETECTION_LR == 0.01
    assert DETECTION_MOMENTUM == 0.9
    assert DETECTION_WD == 4e-5
    assert DETECTION_BATCH_SIZE == 24
    assert DETECTION_EPOCHS == 200
    assert DETECTION_INPUT_SIZE == (320, 320)


def test_box_iou_single():
    """_box_iou_single computes correct IoU."""
    import torch
    from hotcb.eval.tasks_coco_detection import _box_iou_single

    box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
    box2 = torch.tensor([0.0, 0.0, 10.0, 10.0])
    assert abs(_box_iou_single(box1, box2) - 1.0) < 1e-6

    box3 = torch.tensor([5.0, 5.0, 15.0, 15.0])
    # intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
    iou = _box_iou_single(box1, box3)
    assert abs(iou - 25.0 / 175.0) < 1e-6

    box4 = torch.tensor([20.0, 20.0, 30.0, 30.0])
    assert _box_iou_single(box1, box4) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Export tests
# ═══════════════════════════════════════════════════════════════════════


def test_exports_from_eval_init():
    """New conditions are exported from hotcb.eval."""
    from hotcb.eval import (
        IMAGENET_PAPER_CONDITIONS,
        IMAGENET_PAPER_CONTINUATION_CONDITIONS,
        COCO_DETECTION_CONDITIONS,
        COCO_DETECTION_CONTINUATION_CONDITIONS,
    )
    assert len(IMAGENET_PAPER_CONDITIONS) == 5
    assert len(COCO_DETECTION_CONDITIONS) == 4
    assert len(IMAGENET_PAPER_CONTINUATION_CONDITIONS) >= 5
    assert len(COCO_DETECTION_CONTINUATION_CONDITIONS) >= 3
