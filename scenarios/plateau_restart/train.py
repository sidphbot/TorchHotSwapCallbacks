"""Scenario: plateau_restart — val_loss flattens triggering cosine warm restart."""
import os
import random
import threading
import time
from typing import Optional


class _OptProxy:
    def __init__(self, **kwargs):
        self.param_groups = [kwargs]


def train_fn(
    run_dir: str,
    max_steps: int = 200,
    step_delay: float = 0.05,
    stop_event: Optional[threading.Event] = None,
) -> None:
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, mutable_state

    random.seed(42)
    opt = _OptProxy(lr=1e-3, weight_decay=1e-4)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(optimizer_actuators(opt))
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    train_loss = 2.0
    val_loss = 2.2
    val_plateau_value = None
    for step in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break
        lr = opt.param_groups[0]["lr"]

        if step <= 70:
            # Normal decay phase
            decay = 0.018 + random.gauss(0, 0.002)
            train_loss = train_loss * (1 - decay)
            val_loss = val_loss * (1 - decay * 0.9) + random.gauss(0, 0.005)
        elif step <= 120:
            # Plateau phase: val_loss stagnates (range < 0.003 over 30+ steps)
            if val_plateau_value is None:
                val_plateau_value = val_loss
            train_loss = train_loss * (1 - 0.002) + random.gauss(0, 0.002)
            val_loss = val_plateau_value + random.uniform(-0.001, 0.001)
        else:
            # Post-restart: cosine restart should have bumped LR, loss resumes decay
            train_loss = train_loss * (1 - 0.01) + random.gauss(0, 0.003)
            val_loss = val_loss * (1 - 0.008) + random.gauss(0, 0.004)

        grad_norm = 1.2 + random.uniform(-0.3, 0.8)

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "step": step,
            },
            "log": lambda s: None,
        }
        kernel.apply(env, events=["train_step_end"])
        if stop_event and stop_event.is_set():
            break
        if step_delay > 0:
            time.sleep(step_delay)

    kernel.close({"framework": "synthetic", "phase": "train", "step": max_steps,
                   "optimizer": opt, "metrics": {"train_loss": round(train_loss, 6),
                   "val_loss": round(val_loss, 6)}, "log": lambda s: None})
