"""multi_loss_dominance scenario.

train_loss decays normally from 2.0.  aux_loss starts similarly but around
step ~50 it grows to 3x train_loss, triggering the loss_ratio_target rule
which should decrease aux_weight.
"""
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
    max_steps: int = 150,
    step_delay: float = 0.05,
    stop_event: Optional[threading.Event] = None,
) -> None:
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, loss_actuators, mutable_state

    random.seed(42)
    opt = _OptProxy(lr=1e-3, weight_decay=1e-4)
    loss_weights = {"aux_weight": 1.0}

    all_actuators = optimizer_actuators(opt) + loss_actuators(loss_weights)
    ms = mutable_state(all_actuators)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(
        run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms,
    )

    train_loss = 2.0
    aux_loss_raw = 1.8

    for step in range(1, max_steps + 1):
        if stop_event and stop_event.is_set():
            break

        lr = opt.param_groups[0]["lr"]
        aw = loss_weights["aux_weight"]

        # train_loss decays smoothly
        decay = 0.012 * min(lr * 300, 1.0)
        train_loss = max(0.1, train_loss - decay + random.gauss(0, 0.015))

        # aux_loss: starts similar, then dominates after step ~50
        if step < 50:
            aux_loss_raw = max(0.1, aux_loss_raw - 0.01 + random.gauss(0, 0.02))
        else:
            ratio_target = 3.5 + 0.5 * math.sin(step * 0.1)
            aux_loss_raw = train_loss * ratio_target + random.gauss(0, 0.05)

        aux_loss = aux_loss_raw * aw

        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(train_loss, 6),
                "aux_loss": round(aux_loss, 6),
                "lr": lr,
                "aux_weight": aw,
                "step": step,
            },
            "log": lambda s: None,
        }
        kernel.apply(env, events=["train_step_end"])
        if stop_event and stop_event.is_set():
            break
        if step_delay > 0:
            time.sleep(step_delay)

    kernel.close({
        "framework": "synthetic", "phase": "train", "step": max_steps,
        "optimizer": opt, "metrics": {"train_loss": round(train_loss, 6)},
        "log": lambda s: None,
    })
