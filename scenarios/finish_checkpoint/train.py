"""Scenario: finish_checkpoint — val_loss drops below 0.01, triggering best checkpoint callback."""
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

    train_loss = 0.08
    val_loss = 0.05
    for step in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break
        lr = opt.param_groups[0]["lr"]

        # Near-converged: exponential decay crossing 0.01 around step ~60
        train_loss = train_loss * (1 - 0.018) + random.gauss(0, 0.001)
        val_loss = val_loss * (1 - 0.027) + random.gauss(0, 0.0005)

        # Clamp to stay positive
        train_loss = max(train_loss, 0.002)
        val_loss = max(val_loss, 0.003)

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
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
