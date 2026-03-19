"""Scenario: finish_lr_reduction — Late-stage training triggers mutation lockdown at step 900."""
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
    max_steps: int = 150,
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

    step_offset = 850
    train_loss = 0.15
    val_loss = 0.18
    for i in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break
        step = step_offset + i  # actual step: 851..1000
        lr = opt.param_groups[0]["lr"]

        # Slow late-stage decay with small noise
        train_loss = train_loss * (1 - 0.003) + random.gauss(0, 0.001)
        val_loss = val_loss * (1 - 0.002) + random.gauss(0, 0.0015)

        # Clamp to stay positive
        train_loss = max(train_loss, 0.01)
        val_loss = max(val_loss, 0.02)

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

    final_step = step_offset + max_steps
    kernel.close({"framework": "synthetic", "phase": "train", "step": final_step,
                   "optimizer": opt, "metrics": {"train_loss": round(train_loss, 6),
                   "val_loss": round(val_loss, 6)}, "log": lambda s: None})
