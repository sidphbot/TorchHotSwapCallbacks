"""Scenario: stability_nan — NaN loss triggers emergency LR cut."""
import math
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
    max_steps: int = 100,
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

    loss = 2.0
    for step in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break
        lr = opt.param_groups[0]["lr"]

        # Normal decay for first 30 steps, then simulate NaN detection
        grad_norm = 1.0 + random.gauss(0, 0.2)
        nan_detected = 0
        if step <= 30:
            loss = loss * (1 - 0.02) + random.gauss(0, 0.01)
        elif step == 31:
            # NaN detected — loss goes to NaN territory
            loss = float("nan")
            nan_detected = 1
        else:
            # After NaN event, recover with reduced LR
            if math.isnan(loss):
                loss = 2.0  # reset to safe value
            loss = loss * (1 - 0.01) + random.gauss(0, 0.01)

        # Report nan_detected=1 at step 30 to trigger the rule one step early
        if step == 30:
            nan_detected = 1

        display_loss = round(loss, 6) if not math.isnan(loss) else float("nan")

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": display_loss,
                "grad_norm": round(abs(grad_norm), 4),
                "lr": lr,
                "step": step,
                "nan_detected": nan_detected,
            },
            "log": lambda s: None,
        }
        kernel.apply(env, events=["train_step_end"])
        if stop_event and stop_event.is_set():
            break
        if step_delay > 0:
            time.sleep(step_delay)

    final_loss = round(loss, 6) if not math.isnan(loss) else float("nan")
    kernel.close({"framework": "synthetic", "phase": "train", "step": max_steps,
                   "optimizer": opt, "metrics": {"train_loss": final_loss}, "log": lambda s: None})
