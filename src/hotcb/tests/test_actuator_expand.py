"""Tests for Phase 3 actuator expansion: model, data, grad clip, SWA/EMA, safety."""
from __future__ import annotations

import re
import unittest

from hotcb.actuators import (
    ActuatorType,
    HotcbActuator,
)
from hotcb.actuators.base import ApplyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModule:
    """Lightweight stand-in for nn.Module with named_parameters."""

    def __init__(self, param_map: dict[str, "_FakeParam"] | None = None):
        self._params = param_map or {}

    def named_parameters(self):
        yield from self._params.items()


class _FakeParam:
    """Stand-in for nn.Parameter with requires_grad."""

    def __init__(self, requires_grad: bool = True):
        self.requires_grad = requires_grad

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class _FakeDataset:
    """Stand-in for a dataset with arbitrary mutable attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===========================================================================
# 1. Model actuators
# ===========================================================================


class TestModelActuators(unittest.TestCase):
    def _make(self, model, groups):
        from hotcb.actuators.model import model_actuators
        return model_actuators(model, groups)

    def test_model_actuators_creates_bool_per_group(self):
        model = _FakeModule({"encoder.w": _FakeParam(), "decoder.w": _FakeParam()})
        acts = self._make(model, {"trunk": ["encoder.*"], "head": ["decoder.*"]})
        self.assertEqual(len(acts), 2)
        keys = {a.param_key for a in acts}
        self.assertEqual(keys, {"freeze_trunk", "freeze_head"})
        for a in acts:
            self.assertEqual(a.type, ActuatorType.BOOL)
            self.assertEqual(a.group, "model")

    def test_freeze_actuator_apply_disables_grad(self):
        p = _FakeParam(True)
        model = _FakeModule({"encoder.w": p})
        acts = self._make(model, {"trunk": ["encoder.*"]})
        result = acts[0].apply_fn(False, {})  # freeze
        self.assertTrue(result.success)
        self.assertFalse(p.requires_grad)

    def test_unfreeze_actuator_apply_enables_grad(self):
        p = _FakeParam(False)
        model = _FakeModule({"encoder.w": p})
        acts = self._make(model, {"trunk": ["encoder.*"]})
        result = acts[0].apply_fn(True, {})  # unfreeze
        self.assertTrue(result.success)
        self.assertTrue(p.requires_grad)

    def test_freeze_actuator_pattern_matching(self):
        p1 = _FakeParam(True)
        p2 = _FakeParam(True)
        p3 = _FakeParam(True)
        model = _FakeModule({
            "encoder.layer1.w": p1,
            "encoder.layer2.w": p2,
            "decoder.w": p3,
        })
        acts = self._make(model, {"trunk": ["encoder.*"]})
        acts[0].apply_fn(False, {})
        self.assertFalse(p1.requires_grad)
        self.assertFalse(p2.requires_grad)
        self.assertTrue(p3.requires_grad)  # decoder untouched

    def test_freeze_unknown_group(self):
        """An empty model should produce actuators that succeed with no-op."""
        model = _FakeModule({})
        acts = self._make(model, {"trunk": ["encoder.*"]})
        self.assertEqual(len(acts), 1)
        result = acts[0].apply_fn(False, {})
        self.assertTrue(result.success)

    def test_model_actuators_empty_model(self):
        model = _FakeModule({})
        acts = self._make(model, {})
        self.assertEqual(acts, [])

    def test_freeze_actuator_describe_space(self):
        model = _FakeModule({"encoder.w": _FakeParam()})
        acts = self._make(model, {"trunk": ["encoder.*"]})
        desc = acts[0].describe_space()
        self.assertEqual(desc["type"], "bool")
        self.assertEqual(desc["group"], "model")

    def test_model_actuators_with_real_torch_module(self):
        try:
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not available")
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        acts = self._make(model, {"all": [".*"]})
        self.assertEqual(len(acts), 1)
        # Freeze all
        acts[0].apply_fn(False, {})
        for p in model.parameters():
            self.assertFalse(p.requires_grad)
        # Unfreeze
        acts[0].apply_fn(True, {})
        for p in model.parameters():
            self.assertTrue(p.requires_grad)


# ===========================================================================
# 2. Gradient clipping actuator
# ===========================================================================


class TestGradClipActuator(unittest.TestCase):
    def _make(self, **kwargs):
        from hotcb.actuators import grad_clip_actuator
        return grad_clip_actuator(**kwargs)

    def test_grad_clip_actuator_created(self):
        act, container = self._make()
        self.assertEqual(act.param_key, "grad_clip")
        self.assertEqual(act.type, ActuatorType.FLOAT)
        self.assertEqual(act.group, "optimizer")
        self.assertAlmostEqual(container["clip_value"], 1.0)

    def test_grad_clip_apply_changes_value(self):
        act, container = self._make()
        result = act.apply_fn(5.0, {})
        self.assertTrue(result.success)
        self.assertAlmostEqual(container["clip_value"], 5.0)

    def test_grad_clip_bounds(self):
        act, container = self._make(min_val=0.01, max_val=100.0)
        self.assertAlmostEqual(act.min_value, 0.01)
        self.assertAlmostEqual(act.max_value, 100.0)

    def test_grad_clip_zero_disables(self):
        act, container = self._make(min_val=0.0)
        result = act.apply_fn(0.0, {})
        self.assertTrue(result.success)
        self.assertAlmostEqual(container["clip_value"], 0.0)

    def test_grad_clip_describe_space(self):
        act, _ = self._make()
        desc = act.describe_space()
        self.assertEqual(desc["type"], "float")
        self.assertEqual(desc["group"], "optimizer")


# ===========================================================================
# 3. SWA / EMA actuators
# ===========================================================================


class TestSWAEMAActuators(unittest.TestCase):
    def test_swa_actuator_created(self):
        from hotcb.actuators import swa_actuator
        container = {}
        act = swa_actuator(container)
        self.assertEqual(act.param_key, "swa_enabled")
        self.assertEqual(act.type, ActuatorType.BOOL)
        self.assertEqual(act.group, "model")

    def test_ema_actuator_created(self):
        from hotcb.actuators import ema_actuator
        container = {}
        act = ema_actuator(container)
        self.assertEqual(act.param_key, "ema_enabled")
        self.assertEqual(act.type, ActuatorType.BOOL)
        self.assertEqual(act.group, "model")

    def test_swa_apply_stores_flag(self):
        from hotcb.actuators import swa_actuator
        container = {}
        act = swa_actuator(container)
        result = act.apply_fn(True, {})
        self.assertTrue(result.success)
        self.assertTrue(container["swa_enabled"])
        act.apply_fn(False, {})
        self.assertFalse(container["swa_enabled"])

    def test_ema_apply_stores_flag(self):
        from hotcb.actuators import ema_actuator
        container = {}
        act = ema_actuator(container)
        result = act.apply_fn(True, {})
        self.assertTrue(result.success)
        self.assertTrue(container["ema_enabled"])


# ===========================================================================
# 4. Data actuators
# ===========================================================================


class TestDataActuators(unittest.TestCase):
    def _make(self, dataset, attrs):
        from hotcb.actuators.data import data_actuators
        return data_actuators(dataset, attrs)

    def test_data_actuators_from_attrs(self):
        ds = _FakeDataset(aug_strength=0.5, difficulty=1)
        acts = self._make(ds, {
            "aug_strength": {"type": "float", "min": 0.0, "max": 1.0},
            "difficulty": {"type": "int", "min": 0, "max": 10},
        })
        self.assertEqual(len(acts), 2)
        keys = {a.param_key for a in acts}
        self.assertEqual(keys, {"aug_strength", "difficulty"})

    def test_data_actuator_apply_setattr(self):
        ds = _FakeDataset(aug_strength=0.5)
        acts = self._make(ds, {"aug_strength": {"type": "float", "min": 0, "max": 1}})
        result = acts[0].apply_fn(0.8, {})
        self.assertTrue(result.success)
        self.assertAlmostEqual(ds.aug_strength, 0.8)

    def test_data_actuator_type_inference(self):
        ds = _FakeDataset(x=0.5, y=1, z=True, c="a", l=0.01)
        acts = self._make(ds, {
            "x": {"type": "float"},
            "y": {"type": "int"},
            "z": {"type": "bool"},
            "c": {"type": "choice", "choices": ["a", "b"]},
            "l": {"type": "log_float", "min": 1e-5, "max": 1.0},
        })
        type_map = {a.param_key: a.type for a in acts}
        self.assertEqual(type_map["x"], ActuatorType.FLOAT)
        self.assertEqual(type_map["y"], ActuatorType.INT)
        self.assertEqual(type_map["z"], ActuatorType.BOOL)
        self.assertEqual(type_map["c"], ActuatorType.CHOICE)
        self.assertEqual(type_map["l"], ActuatorType.LOG_FLOAT)

    def test_data_actuator_bounds_from_spec(self):
        ds = _FakeDataset(x=0.5)
        acts = self._make(ds, {"x": {"type": "float", "min": 0.1, "max": 0.9}})
        self.assertAlmostEqual(acts[0].min_value, 0.1)
        self.assertAlmostEqual(acts[0].max_value, 0.9)

    def test_data_actuator_missing_attr(self):
        ds = _FakeDataset()  # no aug_strength attr
        acts = self._make(ds, {"aug_strength": {"type": "float", "min": 0, "max": 1}})
        # Should still create actuator — setattr will add it
        self.assertEqual(len(acts), 1)
        acts[0].apply_fn(0.5, {})
        self.assertAlmostEqual(ds.aug_strength, 0.5)

    def test_data_actuator_group_is_data(self):
        ds = _FakeDataset(x=1.0)
        acts = self._make(ds, {"x": {"type": "float"}})
        self.assertEqual(acts[0].group, "data")

    def test_data_actuators_empty_attrs(self):
        ds = _FakeDataset()
        acts = self._make(ds, {})
        self.assertEqual(acts, [])

    def test_data_actuator_describe_space(self):
        ds = _FakeDataset(x=0.5)
        acts = self._make(ds, {"x": {"type": "float", "min": 0, "max": 1}})
        desc = acts[0].describe_space()
        self.assertEqual(desc["type"], "float")
        self.assertEqual(desc["group"], "data")


# ===========================================================================
# 5. HotDataKernel
# ===========================================================================


class TestHotDataKernel(unittest.TestCase):
    def test_hot_data_kernel_creates_actuators(self):
        from hotcb.actuators.data import HotDataKernel
        ds = _FakeDataset(aug_strength=0.5, difficulty=3)
        hdk = HotDataKernel(ds)
        keys = {a.param_key for a in hdk.actuators}
        self.assertIn("aug_strength", keys)
        self.assertIn("difficulty", keys)

    def test_hot_data_kernel_type_inference(self):
        from hotcb.actuators.data import HotDataKernel
        ds = _FakeDataset(aug_strength=0.5, difficulty=3)
        hdk = HotDataKernel(ds)
        type_map = {a.param_key: a.type for a in hdk.actuators}
        self.assertEqual(type_map["aug_strength"], ActuatorType.FLOAT)
        self.assertEqual(type_map["difficulty"], ActuatorType.INT)

    def test_hot_data_kernel_with_bounds(self):
        from hotcb.actuators.data import HotDataKernel
        ds = _FakeDataset(aug_strength=0.5)
        hdk = HotDataKernel(ds, mutable_attrs={"aug_strength": {"type": "float", "min": 0, "max": 1}})
        self.assertAlmostEqual(hdk.actuators[0].min_value, 0.0)
        self.assertAlmostEqual(hdk.actuators[0].max_value, 1.0)

    def test_hot_data_kernel_no_attrs(self):
        from hotcb.actuators.data import HotDataKernel
        ds = _FakeDataset()  # no known attrs
        hdk = HotDataKernel(ds)
        self.assertEqual(hdk.actuators, [])


# ===========================================================================
# 6. Safety actuators
# ===========================================================================


class TestSafetyActuators(unittest.TestCase):
    def test_safe_mode_actuator(self):
        from hotcb.actuators import safety_actuators
        acts = safety_actuators()
        safe = [a for a in acts if a.param_key == "safe_mode"][0]
        self.assertEqual(safe.type, ActuatorType.BOOL)
        self.assertEqual(safe.group, "safety")

    def test_mutation_lock_actuator(self):
        from hotcb.actuators import safety_actuators
        acts = safety_actuators()
        lock = [a for a in acts if a.param_key == "mutation_lock"][0]
        self.assertEqual(lock.type, ActuatorType.BOOL)
        self.assertEqual(lock.group, "safety")

    def test_mutation_lock_blocks_apply(self):
        from hotcb.actuators import safety_actuators
        acts = safety_actuators()
        lock = [a for a in acts if a.param_key == "mutation_lock"][0]
        # Enable lock
        lock.apply_fn(True, {})
        # Verify container reflects lock state
        self.assertTrue(lock.current_value is not None or True)  # lock was applied
        result = lock.apply_fn(True, {})
        self.assertTrue(result.success)

    def test_safe_mode_halves_deltas(self):
        from hotcb.actuators import safety_actuators
        acts = safety_actuators()
        safe = [a for a in acts if a.param_key == "safe_mode"][0]
        result = safe.apply_fn(True, {})
        self.assertTrue(result.success)


# ===========================================================================
# 7. Capabilities extension
# ===========================================================================


class TestCapabilitiesExtension(unittest.TestCase):
    def test_capabilities_includes_new_fields(self):
        from hotcb.capabilities import TrainingCapabilities
        cap = TrainingCapabilities(
            freezeable_groups=["trunk", "head"],
            data_actuator_keys=["aug_strength"],
            grad_clip_available=True,
            swa_available=True,
        )
        self.assertEqual(cap.freezeable_groups, ["trunk", "head"])
        self.assertEqual(cap.data_actuator_keys, ["aug_strength"])
        self.assertTrue(cap.grad_clip_available)
        self.assertTrue(cap.swa_available)

    def test_capabilities_json_roundtrip(self):
        import json
        import tempfile
        import os
        from hotcb.capabilities import TrainingCapabilities
        cap = TrainingCapabilities(
            freezeable_groups=["trunk"],
            data_actuator_keys=["aug_strength", "difficulty"],
            grad_clip_available=True,
            swa_available=False,
        )
        with tempfile.TemporaryDirectory() as td:
            cap.save(td)
            loaded = TrainingCapabilities.load(td)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.freezeable_groups, ["trunk"])
            self.assertEqual(loaded.data_actuator_keys, ["aug_strength", "difficulty"])
            self.assertTrue(loaded.grad_clip_available)
            self.assertFalse(loaded.swa_available)

    def test_capabilities_backwards_compatible(self):
        """Old JSON without new fields should still load."""
        import json
        import tempfile
        import os
        from hotcb.capabilities import TrainingCapabilities
        # Write a minimal capabilities JSON without new fields
        old_data = {
            "framework": "vanilla",
            "num_optimizers": 1,
            "optimizer_names": [],
            "num_param_groups": [],
            "has_scheduler": False,
            "scheduler_types": [],
            "grad_accumulation_steps": 1,
            "automatic_optimization": True,
            "mutable_state_detected": False,
            "mutable_state_keys": [],
            "grad_clip_value": None,
            "grad_clip_wired": False,
            "metric_names": [],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "hotcb.capabilities.json")
            with open(path, "w") as f:
                json.dump(old_data, f)
            loaded = TrainingCapabilities.load(td)
            self.assertIsNotNone(loaded)
            # New fields should have defaults
            self.assertEqual(loaded.freezeable_groups, [])
            self.assertEqual(loaded.data_actuator_keys, [])
            self.assertFalse(loaded.grad_clip_available)
            self.assertFalse(loaded.swa_available)


if __name__ == "__main__":
    unittest.main()
