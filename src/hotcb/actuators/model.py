"""Model actuators — freeze/unfreeze parameter groups by regex pattern.

Usage::

    acts = model_actuators(model, {"trunk": ["encoder.*"], "head": ["decoder.*"]})
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .actuator import ActuatorType, HotcbActuator
from .base import ApplyResult


def model_actuators(
    model: Any,
    groups: Dict[str, List[str]],
) -> List[HotcbActuator]:
    """Create BOOL actuators for freezing/unfreezing named parameter groups.

    Parameters
    ----------
    model:
        Object with ``named_parameters()`` method (e.g. ``nn.Module``).
    groups:
        Mapping of group name to list of regex patterns that match parameter
        names.  Each group becomes one ``freeze_{name}`` BOOL actuator.

    Returns
    -------
    List of ``HotcbActuator`` instances.
    """
    actuators: List[HotcbActuator] = []

    for group_name, patterns in groups.items():
        compiled = [re.compile(p) for p in patterns]
        _model_ref = model

        def _apply_freeze(
            value: Any,
            env: dict,
            _m=_model_ref,
            _pats=compiled,
        ) -> ApplyResult:
            try:
                unfrozen = bool(value)
                for pname, param in _m.named_parameters():
                    if any(pat.search(pname) for pat in _pats):
                        param.requires_grad_(unfrozen)
                return ApplyResult(
                    success=True,
                    detail={"group": group_name, "requires_grad": unfrozen},
                )
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key=f"freeze_{group_name}",
            type=ActuatorType.BOOL,
            apply_fn=_apply_freeze,
            label=f"Freeze {group_name}",
            group="model",
            current_value=True,  # True = unfrozen (requires_grad=True)
        ))

    return actuators
