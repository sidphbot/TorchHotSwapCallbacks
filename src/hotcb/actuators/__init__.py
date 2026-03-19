from .base import BaseActuator, ValidationResult, ApplyResult  # noqa: F401

# --- Unified actuator types (Phase 2+3) ---
from .actuator import (  # noqa: F401
    ActuatorType,
    ActuatorState,
    HotcbActuator,
    Mutation,
    _INIT_SENTINEL,
)
from .state import MutableState  # noqa: F401
from .model import model_actuators  # noqa: F401
from .data import data_actuators, HotDataKernel  # noqa: F401

from typing import Any, Callable, Dict, List, Optional, Tuple


def optimizer_actuators(
    optimizer: Any,
    lr_bounds: Tuple[float, float] = (1e-7, 1.0),
    wd_bounds: Tuple[float, float] = (0.0, 1.0),
) -> List[HotcbActuator]:
    """Create actuators for lr, weight_decay, betas from a torch optimizer (or _OptProxy).

    The ``apply_fn`` closures capture the optimizer reference so mutations
    directly mutate the live object.
    """
    actuators: List[HotcbActuator] = []
    pg = optimizer.param_groups[0] if optimizer.param_groups else {}

    # --- lr ---
    if "lr" in pg:
        _opt_ref = optimizer

        def _apply_lr(value: Any, env: dict, _opt=_opt_ref) -> ApplyResult:
            try:
                new_lr = float(value)
                for g in _opt.param_groups:
                    g["lr"] = new_lr
                # Scheduler coordination
                scheduler = env.get("scheduler")
                if scheduler is not None and hasattr(scheduler, "base_lrs"):
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] = new_lr
                return ApplyResult(success=True, detail={"lr": new_lr})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key="lr",
            type=ActuatorType.LOG_FLOAT,
            apply_fn=_apply_lr,
            metrics_dict_name="lr",
            label="Learning Rate",
            group="optimizer",
            min_value=lr_bounds[0],
            max_value=lr_bounds[1],
            current_value=pg["lr"],
        ))

    # --- weight_decay ---
    if "weight_decay" in pg:
        _opt_ref = optimizer

        def _apply_wd(value: Any, env: dict, _opt=_opt_ref) -> ApplyResult:
            try:
                new_wd = float(value)
                for g in _opt.param_groups:
                    g["weight_decay"] = new_wd
                return ApplyResult(success=True, detail={"weight_decay": new_wd})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key="weight_decay",
            type=ActuatorType.LOG_FLOAT,
            apply_fn=_apply_wd,
            metrics_dict_name="weight_decay",
            label="Weight Decay",
            group="optimizer",
            min_value=wd_bounds[0],
            max_value=wd_bounds[1],
            current_value=pg["weight_decay"],
        ))

    # --- alpha (RMSProp decay) ---
    if "alpha" in pg:
        _opt_ref = optimizer

        def _apply_alpha(value: Any, env: dict, _opt=_opt_ref) -> ApplyResult:
            try:
                new_alpha = float(value)
                for g in _opt.param_groups:
                    g["alpha"] = new_alpha
                return ApplyResult(success=True, detail={"alpha": new_alpha})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key="alpha",
            type=ActuatorType.FLOAT,
            apply_fn=_apply_alpha,
            metrics_dict_name="alpha",
            label="RMSProp Decay",
            group="optimizer",
            min_value=0.8,
            max_value=0.999,
            current_value=pg["alpha"],
        ))

    # --- momentum ---
    if "momentum" in pg and pg["momentum"] > 0:
        _opt_ref = optimizer

        def _apply_momentum(value: Any, env: dict, _opt=_opt_ref) -> ApplyResult:
            try:
                new_mom = float(value)
                for g in _opt.param_groups:
                    g["momentum"] = new_mom
                return ApplyResult(success=True, detail={"momentum": new_mom})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key="momentum",
            type=ActuatorType.FLOAT,
            apply_fn=_apply_momentum,
            metrics_dict_name="momentum",
            label="Momentum",
            group="optimizer",
            min_value=0.0,
            max_value=0.99,
            current_value=pg["momentum"],
        ))

    # --- betas ---
    if "betas" in pg:
        _opt_ref = optimizer

        def _apply_betas(value: Any, env: dict, _opt=_opt_ref) -> ApplyResult:
            try:
                new_betas = tuple(float(b) for b in value)
                for g in _opt.param_groups:
                    g["betas"] = new_betas
                return ApplyResult(success=True, detail={"betas": new_betas})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key="betas",
            type=ActuatorType.TUPLE,
            apply_fn=_apply_betas,
            metrics_dict_name="betas",
            label="Betas",
            group="optimizer",
            current_value=tuple(pg["betas"]),
        ))

    return actuators


def loss_actuators(
    loss_weights: Dict[str, float],
    global_bounds: Tuple[float, float] = (0.0, 100.0),
    key_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[HotcbActuator]:
    """Create FLOAT actuators from a dict of {name: value}.

    The ``apply_fn`` mutates the ORIGINAL dict that was passed in so the
    training loop sees the change immediately.
    """
    key_bounds = key_bounds or {}
    actuators: List[HotcbActuator] = []

    for name, init_value in loss_weights.items():
        bounds = key_bounds.get(name, global_bounds)
        _dict_ref = loss_weights
        _name = name

        def _apply_loss(value: Any, env: dict, _d=_dict_ref, _k=_name) -> ApplyResult:
            try:
                _d[_k] = float(value)
                return ApplyResult(success=True, detail={"key": _k, "value": float(value)})
            except Exception as exc:
                return ApplyResult(success=False, error=str(exc))

        actuators.append(HotcbActuator(
            param_key=name,
            type=ActuatorType.FLOAT,
            apply_fn=_apply_loss,
            metrics_dict_name=name,
            label=name,
            group="loss",
            min_value=bounds[0],
            max_value=bounds[1],
            current_value=init_value,
        ))

    return actuators


def mutable_state(actuators: List[HotcbActuator]) -> MutableState:
    """User-facing constructor for MutableState."""
    return MutableState(actuators)


def grad_clip_actuator(
    initial_value: float = 1.0,
    min_val: float = 0.01,
    max_val: float = 100.0,
) -> Tuple[HotcbActuator, dict]:
    """Create a FLOAT actuator for gradient clipping.

    Returns the actuator and a mutable container dict with key ``clip_value``.
    The training loop should read ``container["clip_value"]`` each step.
    """
    container = {"clip_value": initial_value}

    def _apply_clip(value: Any, env: dict, _c=container) -> ApplyResult:
        try:
            _c["clip_value"] = float(value)
            return ApplyResult(success=True, detail={"clip_value": float(value)})
        except Exception as exc:
            return ApplyResult(success=False, error=str(exc))

    act = HotcbActuator(
        param_key="grad_clip",
        type=ActuatorType.FLOAT,
        apply_fn=_apply_clip,
        label="Gradient Clip",
        group="optimizer",
        min_value=min_val,
        max_value=max_val,
        current_value=initial_value,
    )
    return act, container


def swa_actuator(model_container: dict) -> HotcbActuator:
    """Create a BOOL actuator for SWA (Stochastic Weight Averaging).

    Sets ``model_container["swa_enabled"]``.  Actual model wrapping is the
    adapter's responsibility.
    """
    _c = model_container

    def _apply_swa(value: Any, env: dict, _cont=_c) -> ApplyResult:
        try:
            _cont["swa_enabled"] = bool(value)
            return ApplyResult(success=True, detail={"swa_enabled": bool(value)})
        except Exception as exc:
            return ApplyResult(success=False, error=str(exc))

    return HotcbActuator(
        param_key="swa_enabled",
        type=ActuatorType.BOOL,
        apply_fn=_apply_swa,
        label="SWA",
        group="model",
        current_value=False,
    )


def ema_actuator(model_container: dict) -> HotcbActuator:
    """Create a BOOL actuator for EMA (Exponential Moving Average).

    Sets ``model_container["ema_enabled"]``.  Actual model wrapping is the
    adapter's responsibility.
    """
    _c = model_container

    def _apply_ema(value: Any, env: dict, _cont=_c) -> ApplyResult:
        try:
            _cont["ema_enabled"] = bool(value)
            return ApplyResult(success=True, detail={"ema_enabled": bool(value)})
        except Exception as exc:
            return ApplyResult(success=False, error=str(exc))

    return HotcbActuator(
        param_key="ema_enabled",
        type=ActuatorType.BOOL,
        apply_fn=_apply_ema,
        label="EMA",
        group="model",
        current_value=False,
    )


def safety_actuators() -> List[HotcbActuator]:
    """Create safety actuators: safe_mode and mutation_lock.

    - ``safe_mode``: BOOL flag, when True indicates the system should halve
      mutation deltas (conservative mode).
    - ``mutation_lock``: BOOL flag, when True blocks all mutations.

    Both store their state in a shared container dict accessible via
    ``actuator.detail`` in the ApplyResult.
    """
    container = {"safe_mode": False, "mutation_lock": False}

    def _apply_safe(value: Any, env: dict, _c=container) -> ApplyResult:
        try:
            _c["safe_mode"] = bool(value)
            return ApplyResult(success=True, detail={"safe_mode": bool(value)})
        except Exception as exc:
            return ApplyResult(success=False, error=str(exc))

    def _apply_lock(value: Any, env: dict, _c=container) -> ApplyResult:
        try:
            _c["mutation_lock"] = bool(value)
            return ApplyResult(success=True, detail={"mutation_lock": bool(value)})
        except Exception as exc:
            return ApplyResult(success=False, error=str(exc))

    return [
        HotcbActuator(
            param_key="safe_mode",
            type=ActuatorType.BOOL,
            apply_fn=_apply_safe,
            label="Safe Mode",
            group="safety",
            current_value=False,
        ),
        HotcbActuator(
            param_key="mutation_lock",
            type=ActuatorType.BOOL,
            apply_fn=_apply_lock,
            label="Mutation Lock",
            group="safety",
            current_value=False,
        ),
    ]
