"""Scenario: distill_divergence — Distillation loss spike triggers temperature guard."""
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
    from hotcb.actuators import optimizer_actuators, loss_actuators, mutable_state

    random.seed(42)
    opt = _OptProxy(lr=1e-3, weight_decay=1e-4)
    loss_weights = {"distill_weight": 0.5, "spatial_weight": 0.3}
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    all_actuators = optimizer_actuators(opt) + loss_actuators(loss_weights)
    ms = mutable_state(all_actuators)
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    loss = 2.0
    distill_loss_val = 5.0
    for step in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break
        lr = opt.param_groups[0]["lr"]

        if step < 55:
            # Normal training phase: gradual decay
            decay = 0.015 + random.gauss(0, 0.003)
            loss = max(0.5, loss * (1 - decay) + random.gauss(0, 0.01))
            distill_loss_val = max(1.0, distill_loss_val * (1 - decay * 0.8) + random.gauss(0, 0.05))
        elif step < 75:
            # Divergence phase: distill_loss spikes above 20.0
            distill_loss_val = 22.0 + random.uniform(0, 5.0)
            loss = loss + random.uniform(0.05, 0.15)
        else:
            # Recovery phase: distill_loss drops back (after rule should have fired)
            distill_loss_val = max(2.0, distill_loss_val * 0.92 + random.gauss(0, 0.1))
            loss = max(0.4, loss * (1 - 0.01) + random.gauss(0, 0.01))

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(loss, 6),
                "distill_loss": round(distill_loss_val, 6),
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
