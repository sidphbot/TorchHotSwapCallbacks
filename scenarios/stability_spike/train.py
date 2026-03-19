"""Scenario: stability_spike — Gradient norm spikes trigger clipping."""
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

        # Normal gradient norms with a spike window at steps 20-25
        if 20 <= step <= 25:
            # Spike: grad_norm between 15 and 20 (well above threshold of 10)
            grad_norm = 15.0 + random.uniform(0, 5.0)
            # Loss destabilizes slightly during spike
            loss = loss + random.uniform(0.02, 0.08)
        else:
            # Normal: grad_norm between 1 and 3
            grad_norm = 1.5 + random.uniform(-0.5, 1.5)
            loss = loss * (1 - 0.02) + random.gauss(0, 0.01)

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(loss, 6),
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
                   "optimizer": opt, "metrics": {"train_loss": round(loss, 6)}, "log": lambda s: None})
