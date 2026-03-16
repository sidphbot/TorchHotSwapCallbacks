"""Data actuators — expose dataset/loader attributes as live-tunable knobs.

Usage::

    acts = data_actuators(dataset, {"aug_strength": {"type": "float", "min": 0, "max": 1}})

    # Or auto-discover known attributes:
    hdk = HotDataKernel(dataset)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .actuator import ActuatorType, HotcbActuator
from .base import ApplyResult


_TYPE_MAP = {
    "float": ActuatorType.FLOAT,
    "int": ActuatorType.INT,
    "bool": ActuatorType.BOOL,
    "choice": ActuatorType.CHOICE,
    "log_float": ActuatorType.LOG_FLOAT,
    "log": ActuatorType.LOG_FLOAT,
    "log_scale": ActuatorType.LOG_FLOAT,
}


def data_actuators(
    dataset_or_loader: Any,
    attrs: Dict[str, dict],
) -> List[HotcbActuator]:
    """Create actuators that mutate attributes on a dataset or dataloader.

    Parameters
    ----------
    dataset_or_loader:
        Object whose attributes will be set via ``setattr``.
    attrs:
        Mapping of attribute name to spec dict.  Each spec may contain:
        - ``type``: one of "float", "int", "bool", "choice", "log_float", "log", "log_scale"
        - ``min``, ``max``: numeric bounds (for float/int/log_float)
        - ``choices``: list of valid values (for choice type)

    Returns
    -------
    List of ``HotcbActuator`` instances.
    """
    actuators: List[HotcbActuator] = []
    obj_ref = dataset_or_loader

    for attr_name, spec in attrs.items():
        atype = _TYPE_MAP.get(spec.get("type", "float"), ActuatorType.FLOAT)
        _attr = attr_name
        _obj = obj_ref

        def _apply_setattr(
            value: Any,
            env: dict,
            _o=_obj,
            _a=_attr,
        ) -> ApplyResult:
            try:
                setattr(_o, _a, value)
                return ApplyResult(success=True, detail={"attr": _a, "value": value})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        current = getattr(obj_ref, attr_name, None)

        actuators.append(HotcbActuator(
            param_key=attr_name,
            type=atype,
            apply_fn=_apply_setattr,
            label=attr_name,
            group="data",
            min_value=spec.get("min"),
            max_value=spec.get("max"),
            choices=spec.get("choices"),
            current_value=current,
        ))

    return actuators


class HotDataKernel:
    """Auto-discovering data actuator manager.

    Scans a dataset for known mutable attributes and creates actuators
    automatically.  Accepts optional explicit specs via ``mutable_attrs``.
    """

    KNOWN_ATTRS = [
        "augmentation_strength", "aug_strength", "curriculum_stage",
        "difficulty", "sample_weights", "mix_ratio", "hard_case_weight",
    ]

    def __init__(
        self,
        dataset: Any,
        mutable_attrs: Optional[Dict[str, dict]] = None,
    ):
        if mutable_attrs is not None:
            specs = mutable_attrs
        else:
            specs = {}
            for attr in self.KNOWN_ATTRS:
                if hasattr(dataset, attr):
                    val = getattr(dataset, attr)
                    specs[attr] = self._infer_spec(val)

        self.actuators: List[HotcbActuator] = data_actuators(dataset, specs)

    @staticmethod
    def _infer_spec(value: Any) -> dict:
        """Infer a spec dict from the current value's type."""
        if isinstance(value, bool):
            return {"type": "bool"}
        elif isinstance(value, int):
            return {"type": "int"}
        elif isinstance(value, float):
            return {"type": "float"}
        elif isinstance(value, str):
            return {"type": "choice", "choices": [value]}
        return {"type": "float"}
