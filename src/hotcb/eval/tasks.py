"""Real PyTorch training functions for evaluation.

Small MNIST and CIFAR-10 CNN tasks that follow the standard train_fn contract:
  train_fn(run_dir, max_steps, step_delay, _stop_event)

These are *real* training runs — actual models, data, gradients, and backprop.
They integrate with HotKernel + MetricsCollector + actuators (same path as demos).

Designed to complete quickly:
  - MNIST:   ~1-2 min on GPU, ~3-4 min on CPU  (3 epochs, ~1400 steps)
  - CIFAR-10: ~2-3 min on GPU, ~5-8 min on CPU  (5 epochs, ~2000 steps)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MNIST Small CNN (~14K params)
# ---------------------------------------------------------------------------

class _MnistCNN(nn.Module):
    """Tiny CNN for MNIST: Conv→Pool→Conv→Pool→FC.  ~14K parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 1×28×28 → 16×28×28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 16×14×14 → 32×14×14
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)          # 16×14×14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          # 32×7×7
        x = x.view(x.size(0), -1)
        return self.fc(x)


def mnist_training(
    run_dir: str,
    *,
    max_steps: int = 1500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Real MNIST training with HotKernel integration.

    3 epochs over 60K images, batch_size=128 → ~1406 steps.
    Default max_steps=1500 gives a small buffer.
    """
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, mutable_state

    device = _pick_device()

    # --- Data ---
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform,
    )
    val_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
    )

    # --- Model + optimizer ---
    model = _MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- hotcb wiring ---
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    # Check for no-recipe sentinel
    _skip_recipe = os.path.exists(os.path.join(run_dir, "hotcb.eval.no_recipe"))
    ms = mutable_state(optimizer_actuators(optimizer))
    kernel = HotKernel(
        run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms,
    )

    # --- Training loop ---
    step = 0
    grad_norm_ema = 1.0
    for epoch in range(100):  # enough epochs — max_steps will cap it
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps:
                break
            if _stop_event and _stop_event.is_set():
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema = 0.95 * grad_norm_ema + 0.05 * grad_norm

            optimizer.step()

            # Training accuracy for this batch
            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()

            step += 1

            # Collect metrics via kernel
            env = {
                "framework": "torch",
                "phase": "train",
                "step": step,
                "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss.item(), 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                },
                "log": lambda s: None,
            }

            # Validate periodically (every ~468 steps = 1 epoch worth)
            if step % 468 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)

            kernel.apply(env, events=["train_step_end"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps:
            break
        if _stop_event and _stop_event.is_set():
            break

    # Final validation — write through kernel.apply so it hits MetricsCollector
    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss.item(), 6) if step > 0 else 0,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])
    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()


# ---------------------------------------------------------------------------
# CIFAR-10 Small CNN (~62K params)
# ---------------------------------------------------------------------------

class _CifarCNN(nn.Module):
    """Small CNN for CIFAR-10.  ~62K parameters.

    Conv(3→32)→BN→ReLU→Pool → Conv(32→64)→BN→ReLU→Pool → FC(64*8*8→128)→FC(128→10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # 32×16×16
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # 64×8×8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def cifar10_training(
    run_dir: str,
    *,
    max_steps: int = 2000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Real CIFAR-10 training with HotKernel integration.

    5 epochs over 50K images, batch_size=128 → ~1953 steps.
    Default max_steps=2000 gives a small buffer.
    """
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, mutable_state

    device = _pick_device()

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)

    # --- Data ---
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    val_transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
    )

    # --- Model + optimizer ---
    model = _CifarCNN().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # --- hotcb wiring ---
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    _skip_recipe = os.path.exists(os.path.join(run_dir, "hotcb.eval.no_recipe"))
    ms = mutable_state(optimizer_actuators(optimizer))
    kernel = HotKernel(
        run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms,
    )

    # --- Training loop ---
    step = 0
    grad_norm_ema = 1.0
    loss_val = 0.0
    for epoch in range(100):  # max_steps will cap it
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps:
                break
            if _stop_event and _stop_event.is_set():
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema = 0.95 * grad_norm_ema + 0.05 * grad_norm

            optimizer.step()

            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()
            loss_val = loss.item()

            step += 1

            env = {
                "framework": "torch",
                "phase": "train",
                "step": step,
                "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss_val, 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                },
                "log": lambda s: None,
            }

            # Validate every ~391 steps (1 epoch for CIFAR-10)
            if step % 391 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)

            kernel.apply(env, events=["train_step_end"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps:
            break
        if _stop_event and _stop_event.is_set():
            break

    # Final validation — write through kernel.apply so it hits MetricsCollector
    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss_val, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])
    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pick_device():
    """Pick GPU if available and has enough free memory, else CPU."""
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            if free > 256 * 1024 * 1024:  # need at least 256MB free
                return torch.device("cuda")
        except Exception:
            pass
        # Try allocating a small tensor to test
        try:
            t = torch.zeros(1, device="cuda")
            del t
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def _cleanup_gpu():
    """Release GPU memory between runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    """Run validation, return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Checkpoint save/load helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    metrics: Dict[str, float],
    scheduler=None,
) -> None:
    """Save a training checkpoint + metadata sidecar."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)

    # Write metadata sidecar for anchor selector
    meta_path = path.rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({"step": step, "epoch": epoch, "metrics": metrics}, f, indent=2)


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    resume_mode: str = "full_resume",
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a checkpoint with the specified resume mode.

    Returns the checkpoint dict (step, epoch, metrics).
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if resume_mode == "full_resume" and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    elif resume_mode == "reset_scheduler" and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Scheduler will be freshly created by caller
    # "weights_only" — don't load optimizer state at all

    return {
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
    }


# ---------------------------------------------------------------------------
# SWA/EMA weight averaging helpers
# ---------------------------------------------------------------------------

class _SWAState:
    """Stochastic Weight Averaging state tracker."""

    def __init__(self, model: nn.Module, swa_start_step: int = 0):
        self.n_averaged = 0
        self.avg_params = {name: p.data.clone() for name, p in model.named_parameters()}
        self.enabled = False
        self.swa_start_step = swa_start_step

    def update(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        self.n_averaged += 1
        for name, p in model.named_parameters():
            if name in self.avg_params:
                self.avg_params[name] += (p.data - self.avg_params[name]) / self.n_averaged

    def apply_average(self, model: nn.Module) -> None:
        """Replace model params with SWA averaged params."""
        if self.n_averaged < 2:
            return
        for name, p in model.named_parameters():
            if name in self.avg_params:
                p.data.copy_(self.avg_params[name])


class _EMAState:
    """Exponential Moving Average state tracker."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = {name: p.data.clone() for name, p in model.named_parameters()}
        self.enabled = False

    def update(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        for name, p in model.named_parameters():
            if name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model params with EMA shadow params."""
        for name, p in model.named_parameters():
            if name in self.shadow_params:
                p.data.copy_(self.shadow_params[name])


# ---------------------------------------------------------------------------
# MNIST with checkpoints + SWA/EMA support
# ---------------------------------------------------------------------------

def mnist_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 1500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval: int = 250,
) -> None:
    """MNIST training that saves periodic checkpoints for continuation."""
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, swa_actuator, ema_actuator,
        grad_clip_actuator, mutable_state,
    )

    device = _pick_device()

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = _MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # SWA/EMA state
    model_container = {"swa_enabled": False, "ema_enabled": False}
    swa = _SWAState(model)
    ema = _EMAState(model)

    # Grad clip container
    clip_act, clip_container = grad_clip_actuator(initial_value=5.0)

    # Actuators: optimizer + SWA + EMA + grad_clip
    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(acts)
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    # Checkpoint dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    grad_norm_ema_val = 1.0
    best_val_acc = 0.0
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps or (_stop_event and _stop_event.is_set()):
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema_val = 0.95 * grad_norm_ema_val + 0.05 * grad_norm

            # Gradient clipping
            clip_val = clip_container["clip_value"]
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

            # SWA/EMA update
            if model_container["swa_enabled"]:
                swa.enabled = True
                swa.update(model)
            if model_container["ema_enabled"]:
                ema.enabled = True
                ema.update(model)

            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()
            step += 1

            env = {
                "framework": "torch", "phase": "train", "step": step, "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss.item(), 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema_val, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                    "swa_enabled": model_container["swa_enabled"],
                    "ema_enabled": model_container["ema_enabled"],
                    "grad_clip": clip_container["clip_value"],
                },
                "log": lambda s: None,
            }

            if step % 468 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_metrics = dict(env["metrics"])

            kernel.apply(env, events=["train_step_end"])

            # Save checkpoint
            if step % checkpoint_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                _save_checkpoint(ckpt_path, model, optimizer, step, epoch, env["metrics"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps or (_stop_event and _stop_event.is_set()):
            break

    # Apply SWA/EMA before final eval if enabled
    if swa.enabled and swa.n_averaged >= 2:
        swa.apply_average(model)
    elif ema.enabled:
        ema.apply_shadow(model)

    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss.item(), 6) if step > 0 else 0,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema_val, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])

    # Save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}_final.pt")
    _save_checkpoint(ckpt_path, model, optimizer, step, epoch, final_env["metrics"])

    # Save best metrics
    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_val_metrics, f, indent=2)

    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()


def mnist_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: str = "",
    resume_mode: str = "full_resume",
) -> None:
    """Resume MNIST training from a checkpoint with hotcb mutation support."""
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, swa_actuator, ema_actuator,
        grad_clip_actuator, mutable_state,
    )

    device = _pick_device()

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = _MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    ckpt_info = _load_checkpoint(checkpoint_path, model, optimizer, resume_mode, device)
    start_step = ckpt_info["step"]

    # SWA/EMA/grad_clip
    model_container = {"swa_enabled": False, "ema_enabled": False}
    swa = _SWAState(model)
    ema = _EMAState(model)
    clip_act, clip_container = grad_clip_actuator(initial_value=5.0)

    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(acts)
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    step = 0
    grad_norm_ema_val = 1.0

    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps or (_stop_event and _stop_event.is_set()):
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema_val = 0.95 * grad_norm_ema_val + 0.05 * grad_norm

            clip_val = clip_container["clip_value"]
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

            if model_container["swa_enabled"]:
                swa.enabled = True
                swa.update(model)
            if model_container["ema_enabled"]:
                ema.enabled = True
                ema.update(model)

            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()
            step += 1

            env = {
                "framework": "torch", "phase": "train",
                "step": start_step + step, "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss.item(), 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema_val, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                    "swa_enabled": model_container["swa_enabled"],
                    "ema_enabled": model_container["ema_enabled"],
                    "grad_clip": clip_container["clip_value"],
                },
                "log": lambda s: None,
            }

            # Validate frequently during continuation (every 100 steps)
            if step % 100 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)

            kernel.apply(env, events=["train_step_end"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps or (_stop_event and _stop_event.is_set()):
            break

    # Apply weight averaging before final eval
    if swa.enabled and swa.n_averaged >= 2:
        swa.apply_average(model)
    elif ema.enabled:
        ema.apply_shadow(model)

    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": start_step + step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss.item(), 6) if step > 0 else 0,
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema_val, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])
    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()


# ---------------------------------------------------------------------------
# CIFAR-10 with checkpoints + SWA/EMA support
# ---------------------------------------------------------------------------

def cifar10_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 2000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval: int = 250,
) -> None:
    """CIFAR-10 training that saves periodic checkpoints for continuation."""
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, swa_actuator, ema_actuator,
        grad_clip_actuator, mutable_state,
    )

    device = _pick_device()
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(_MEAN, _STD),
    ])
    val_transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = _CifarCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model_container = {"swa_enabled": False, "ema_enabled": False}
    swa = _SWAState(model)
    ema = _EMAState(model)
    clip_act, clip_container = grad_clip_actuator(initial_value=5.0)

    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(acts)
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    grad_norm_ema_val = 1.0
    loss_val = 0.0
    best_val_acc = 0.0
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps or (_stop_event and _stop_event.is_set()):
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema_val = 0.95 * grad_norm_ema_val + 0.05 * grad_norm

            clip_val = clip_container["clip_value"]
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

            if model_container["swa_enabled"]:
                swa.enabled = True
                swa.update(model)
            if model_container["ema_enabled"]:
                ema.enabled = True
                ema.update(model)

            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()
            loss_val = loss.item()
            step += 1

            env = {
                "framework": "torch", "phase": "train", "step": step, "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss_val, 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema_val, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                    "swa_enabled": model_container["swa_enabled"],
                    "ema_enabled": model_container["ema_enabled"],
                    "grad_clip": clip_container["clip_value"],
                },
                "log": lambda s: None,
            }

            if step % 391 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_metrics = dict(env["metrics"])

            kernel.apply(env, events=["train_step_end"])

            if step % checkpoint_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                _save_checkpoint(ckpt_path, model, optimizer, step, epoch, env["metrics"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps or (_stop_event and _stop_event.is_set()):
            break

    if swa.enabled and swa.n_averaged >= 2:
        swa.apply_average(model)
    elif ema.enabled:
        ema.apply_shadow(model)

    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss_val, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema_val, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])

    ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}_final.pt")
    _save_checkpoint(ckpt_path, model, optimizer, step, epoch, final_env["metrics"])

    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_val_metrics, f, indent=2)

    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()


def cifar10_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: str = "",
    resume_mode: str = "full_resume",
) -> None:
    """Resume CIFAR-10 training from a checkpoint with hotcb mutation support."""
    import torchvision
    import torchvision.transforms as T
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, swa_actuator, ema_actuator,
        grad_clip_actuator, mutable_state,
    )

    device = _pick_device()
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(_MEAN, _STD),
    ])
    val_transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = _CifarCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ckpt_info = _load_checkpoint(checkpoint_path, model, optimizer, resume_mode, device)
    start_step = ckpt_info["step"]

    model_container = {"swa_enabled": False, "ema_enabled": False}
    swa = _SWAState(model)
    ema = _EMAState(model)
    clip_act, clip_container = grad_clip_actuator(initial_value=5.0)

    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(acts)
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    step = 0
    grad_norm_ema_val = 1.0
    loss_val = 0.0

    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            if step >= max_steps or (_stop_event and _stop_event.is_set()):
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm_ema_val = 0.95 * grad_norm_ema_val + 0.05 * grad_norm

            clip_val = clip_container["clip_value"]
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

            if model_container["swa_enabled"]:
                swa.enabled = True
                swa.update(model)
            if model_container["ema_enabled"]:
                ema.enabled = True
                ema.update(model)

            preds = logits.argmax(dim=1)
            train_acc = (preds == batch_y).float().mean().item()
            loss_val = loss.item()
            step += 1

            env = {
                "framework": "torch", "phase": "train",
                "step": start_step + step, "epoch": epoch,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(loss_val, 6),
                    "grad_norm": round(grad_norm, 4),
                    "grad_norm_ema": round(grad_norm_ema_val, 4),
                    "train_accuracy": round(train_acc, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
                    "swa_enabled": model_container["swa_enabled"],
                    "ema_enabled": model_container["ema_enabled"],
                    "grad_clip": clip_container["clip_value"],
                },
                "log": lambda s: None,
            }

            if step % 100 == 0 or step == 1:
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                env["metrics"]["val_loss"] = round(val_loss, 6)
                env["metrics"]["val_accuracy"] = round(val_acc, 4)

            kernel.apply(env, events=["train_step_end"])

            if step_delay > 0:
                time.sleep(step_delay)

        if step >= max_steps or (_stop_event and _stop_event.is_set()):
            break

    if swa.enabled and swa.n_averaged >= 2:
        swa.apply_average(model)
    elif ema.enabled:
        ema.apply_shadow(model)

    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
    final_env = {
        "framework": "torch", "phase": "train", "step": start_step + step + 1,
        "optimizer": optimizer,
        "metrics": {
            "train_loss": round(loss_val, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "grad_norm": round(grad_norm_ema_val, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        },
    }
    kernel.apply(final_env, events=["train_step_end"])
    kernel.close(final_env)
    del model, optimizer
    _cleanup_gpu()
