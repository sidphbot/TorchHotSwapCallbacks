"""ImageNet MobileNetV2 — paper-faithful from-scratch training (Sandler et al. 2018).

Paper recipe (§6.1):
  Optimizer: RMSProp, alpha=0.9, momentum=0.9, eps=1.0 (TF reference)
  LR: 0.045, ExponentialLR gamma=0.98 per epoch
  Weight decay: 0.00004
  Batch size: 96
  Epochs: 300 (torchvision V1 recipe)
  Target: 72.0% top-1 (Paper Table 4)

Requirements: torchvision, PIL
Data: /media/burplord/kraken_data/off_domain_data/imagenet/

VRAM: ~3GB (batch_size=96, MobileNetV2 ~3.4M params, AMP)
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


# Paper hyperparameters (Sandler et al. 2018, §6.1)
PAPER_LR = 0.045
PAPER_ALPHA = 0.9       # RMSProp decay
PAPER_MOMENTUM = 0.9
PAPER_WD = 0.00004
PAPER_EPS = 1.0         # TF reference implementation
PAPER_GAMMA = 0.98      # ExponentialLR per epoch
PAPER_BATCH_SIZE = 96
PAPER_EPOCHS = 300       # torchvision V1 recipe (paper doesn't specify)


def _get_imagenet_transforms():
    """Standard ImageNet transforms (paper + torchvision standard)."""
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def _build_imagenet_model(pretrained: bool = False):
    """MobileNetV2 — from scratch by default (paper recipe)."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    return mobilenet_v2(weights=weights)


def _setup_actuators(optimizer):
    """Common actuator setup for ImageNet paper tasks."""
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    clip_act, clip_container = grad_clip_actuator(initial_value=5.0)
    model_container = {"swa_enabled": False, "ema_enabled": False}
    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)
    acts.extend(safety_actuators())
    ms = mutable_state(acts)
    return ms, clip_container, model_container


def imagenet_mobilenetv2_paper_training(
    run_dir: str,
    *,
    max_steps: int = 0,
    num_epochs: int = PAPER_EPOCHS,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
    use_amp: bool = True,
    val_samples: int = 0,
    val_interval_epochs: int = 1,
    data_root: Optional[str] = None,
) -> None:
    """Train MobileNetV2 from scratch on ImageNet — paper-faithful RMSProp recipe.

    Epoch-based loop matching paper §6.1. LR decays per epoch (ExponentialLR).
    Uses AMP by default for speed.

    Args:
        max_steps: If >0, overrides num_epochs (step-based training).
        num_epochs: Number of training epochs (default: 300).
        val_samples: Max val samples. 0 = full val set (~50k).
        val_interval_epochs: How often to run full validation.
        data_root: Override ImageNet data directory (default: HDD path).
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from .datasets import ImageNetFlat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    torch.backends.cudnn.benchmark = True

    # Data
    train_tf, val_tf = _get_imagenet_transforms()
    train_ds = ImageNetFlat(split="train", transform=train_tf, root=data_root)
    val_ds = ImageNetFlat(split="val", transform=val_tf, max_samples=val_samples, root=data_root)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=PAPER_BATCH_SIZE, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=PAPER_BATCH_SIZE, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # Model — from scratch
    model = _build_imagenet_model(pretrained=False)
    model.to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    # Paper-exact optimizer: RMSProp
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=PAPER_LR,
        alpha=PAPER_ALPHA,
        momentum=PAPER_MOMENTUM,
        weight_decay=PAPER_WD,
        eps=PAPER_EPS,
    )

    # LR schedule: ×0.98 per epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=PAPER_GAMMA)

    # AMP
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if resume_mode == "full_resume":
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)

    # Actuators + kernel
    ms, clip_container, model_container = _setup_actuators(optimizer)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    # Training
    model.train()
    for epoch in range(start_epoch, num_epochs):
        if stop.is_set():
            break

        for batch_idx, (images, labels) in enumerate(train_loader):
            if stop.is_set():
                break
            if max_steps > 0 and global_step >= max_steps:
                break

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                top5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

            env = {
                "framework": "torch",
                "phase": "train",
                "step": global_step,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "metrics": {
                    "train_loss": round(loss.item(), 6),
                    "train_accuracy": round(acc, 4),
                    "train_top5_accuracy": round(top5, 4),
                    "grad_norm": round(grad_norm, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                "log": lambda s: None,
            }

            kernel.apply(env, events=["train_step_end"])
            global_step += 1

            if step_delay > 0:
                time.sleep(step_delay)

        if max_steps > 0 and global_step >= max_steps:
            break

        # Epoch-end: LR decay
        scheduler.step()

        # Validation
        if (epoch + 1) % val_interval_epochs == 0 or epoch == num_epochs - 1:
            val_metrics = _validate_imagenet(model, val_loader, device, use_amp=use_amp)
            env = {
                "framework": "torch",
                "phase": "val",
                "step": global_step,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "metrics": {
                    "val_loss": round(val_metrics["val_loss"], 6),
                    "val_accuracy": round(val_metrics["val_accuracy"], 4),
                    "val_top5_accuracy": round(val_metrics["val_top5_accuracy"], 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                "log": lambda s: None,
            }
            kernel.apply(env, events=["val_epoch_end"])
            model.train()

    # Final
    val_metrics = _validate_imagenet(model, val_loader, device, use_amp=use_amp)
    env = {
        "framework": "torch",
        "phase": "val",
        "step": global_step,
        "optimizer": optimizer,
        "metrics": {
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_accuracy": round(val_metrics["val_accuracy"], 4),
            "val_top5_accuracy": round(val_metrics["val_top5_accuracy"], 4),
            "epoch": epoch if num_epochs > 0 else 0,
        },
        "log": lambda s: None,
    }
    kernel.apply(env, events=["train_step_end"])
    kernel.close(env)


def imagenet_mobilenetv2_paper_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 0,
    num_epochs: int = PAPER_EPOCHS,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval_epochs: int = 10,
    use_amp: bool = True,
    val_samples: int = 0,
    data_root: Optional[str] = None,
) -> None:
    """Train from scratch with periodic checkpoint saves.

    Saves: model, optimizer, scheduler, scaler, epoch, global_step, best_metrics.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from .datasets import ImageNetFlat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    torch.backends.cudnn.benchmark = True

    train_tf, val_tf = _get_imagenet_transforms()
    train_ds = ImageNetFlat(split="train", transform=train_tf, root=data_root)
    val_ds = ImageNetFlat(split="val", transform=val_tf, max_samples=val_samples, root=data_root)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=PAPER_BATCH_SIZE, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=PAPER_BATCH_SIZE, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = _build_imagenet_model(pretrained=False)
    model.to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=PAPER_LR, alpha=PAPER_ALPHA, momentum=PAPER_MOMENTUM,
        weight_decay=PAPER_WD, eps=PAPER_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=PAPER_GAMMA)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    ms, clip_container, model_container = _setup_actuators(optimizer)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        if stop.is_set():
            break

        for batch_idx, (images, labels) in enumerate(train_loader):
            if stop.is_set():
                break
            if max_steps > 0 and global_step >= max_steps:
                break

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                top5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

            env = {
                "framework": "torch",
                "phase": "train",
                "step": global_step,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "metrics": {
                    "train_loss": round(loss.item(), 6),
                    "train_accuracy": round(acc, 4),
                    "train_top5_accuracy": round(top5, 4),
                    "grad_norm": round(grad_norm, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                "log": lambda s: None,
            }
            kernel.apply(env, events=["train_step_end"])
            global_step += 1

            if step_delay > 0:
                time.sleep(step_delay)

        if max_steps > 0 and global_step >= max_steps:
            break

        # Epoch-end: LR decay
        scheduler.step()

        # Validation
        val_metrics = _validate_imagenet(model, val_loader, device, use_amp=use_amp)
        val_env = {
            "framework": "torch",
            "phase": "val",
            "step": global_step,
            "optimizer": optimizer,
            "metrics": {
                "val_loss": round(val_metrics["val_loss"], 6),
                "val_accuracy": round(val_metrics["val_accuracy"], 4),
                "val_top5_accuracy": round(val_metrics["val_top5_accuracy"], 4),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            },
            "log": lambda s: None,
        }
        kernel.apply(val_env, events=["val_epoch_end"])

        if val_metrics.get("val_accuracy", 0) > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            best_metrics = {**val_env["metrics"], **val_metrics}

        # Checkpoint
        if (epoch + 1) % checkpoint_interval_epochs == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": val_env["metrics"],
                "best_val_accuracy": best_val_acc,
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, ckpt_path)

            # Save best model separately
            if val_metrics.get("val_accuracy", 0) >= best_val_acc:
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save(save_dict, best_path)

        model.train()

    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)


def imagenet_mobilenetv2_paper_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 13345,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Continuation variant — resume from checkpoint with paper recipe."""
    imagenet_mobilenetv2_paper_training(
        run_dir,
        max_steps=max_steps,
        num_epochs=9999,  # step-limited via max_steps
        step_delay=step_delay,
        _stop_event=_stop_event,
        checkpoint_path=checkpoint_path,
        resume_mode=resume_mode,
    )


def imagenet_mobilenetv2_pretrained_validation(
    run_dir: str,
    *,
    max_steps: int = 1,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Validate torchvision V1 pretrained weights — sanity check (71.878% top-1).

    Runs full val set, writes metrics, exits. No training.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from .datasets import ImageNetFlat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_tf = _get_imagenet_transforms()
    val_ds = ImageNetFlat(split="val", transform=val_tf)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=PAPER_BATCH_SIZE, shuffle=False, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = _build_imagenet_model(pretrained=True)
    model.to(device)

    val_metrics = _validate_imagenet(model, val_loader, device, use_amp=True)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, metrics_collector=mc)
    env = {
        "framework": "torch",
        "phase": "val",
        "step": 0,
        "metrics": {
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_accuracy": round(val_metrics["val_accuracy"], 4),
            "val_top5_accuracy": round(val_metrics["val_top5_accuracy"], 4),
        },
        "log": lambda s: None,
    }
    kernel.apply(env, events=["val_epoch_end"])
    kernel.close(env)

    with open(os.path.join(run_dir, "validation_results.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Legacy functions (deprecated — kept for backward compatibility)
# Use imagenet_mobilenetv2_paper_* functions instead.
# ═══════════════════════════════════════════════════════════════════════

def imagenet_mobilenet_training(
    run_dir: str,
    *,
    max_steps: int = 10000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """DEPRECATED: Fine-tune pretrained MobileNetV2 (SGD lr=0.001).

    Use imagenet_mobilenetv2_paper_training() for paper-faithful from-scratch training.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from .datasets import ImageNetFlat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    train_tf, val_tf = _get_imagenet_transforms()
    train_ds = ImageNetFlat(split="train", transform=train_tf)
    val_ds = ImageNetFlat(split="val", transform=val_tf, max_samples=10000)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = _build_imagenet_model(pretrained=True)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-5,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if resume_mode == "full_resume" and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ms, clip_container, model_container = _setup_actuators(optimizer)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    model.train()
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps and not stop.is_set():
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_container["clip_value"]).item()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        top5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "train_loss": round(loss.item(), 6),
                "train_accuracy": round(acc, 4),
                "train_top5_accuracy": round(top5, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        if step % 2000 == 0 and step > 0:
            val_metrics = _validate_imagenet(model, val_loader, device)
            env["metrics"]["val_loss"] = round(val_metrics["val_loss"], 6)
            env["metrics"]["val_accuracy"] = round(val_metrics["val_accuracy"], 4)
            env["metrics"]["val_top5_accuracy"] = round(val_metrics["val_top5_accuracy"], 4)

        kernel.apply(env, events=["train_step_end"])
        step += 1
        if step_delay > 0:
            time.sleep(step_delay)

    val_metrics = _validate_imagenet(model, val_loader, device)
    env = {
        "framework": "torch",
        "phase": "train",
        "step": step,
        "optimizer": optimizer,
        "metrics": {
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_accuracy": round(val_metrics["val_accuracy"], 4),
            "val_top5_accuracy": round(val_metrics["val_top5_accuracy"], 4),
        },
        "log": lambda s: None,
    }
    kernel.apply(env, events=["train_step_end"])
    kernel.close(env)


def imagenet_mobilenet_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 10000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval: int = 2000,
) -> None:
    """DEPRECATED: Pretrained fine-tune with checkpoints.

    Use imagenet_mobilenetv2_paper_training_with_checkpoints() instead.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from .datasets import ImageNetFlat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    train_tf, val_tf = _get_imagenet_transforms()
    train_ds = ImageNetFlat(split="train", transform=train_tf)
    val_ds = ImageNetFlat(split="val", transform=val_tf, max_samples=10000)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = _build_imagenet_model(pretrained=True)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-5,
    )

    ms, clip_container, model_container = _setup_actuators(optimizer)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}

    model.train()
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps and not stop.is_set():
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_container["clip_value"]).item()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        top5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "train_loss": round(loss.item(), 6),
                "train_accuracy": round(acc, 4),
                "train_top5_accuracy": round(top5, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        if step % 2000 == 0 and step > 0:
            val_metrics = _validate_imagenet(model, val_loader, device)
            env["metrics"]["val_loss"] = round(val_metrics["val_loss"], 6)
            env["metrics"]["val_accuracy"] = round(val_metrics["val_accuracy"], 4)
            env["metrics"]["val_top5_accuracy"] = round(val_metrics["val_top5_accuracy"], 4)
            if val_metrics.get("val_accuracy", 0) > best_val_acc:
                best_val_acc = val_metrics["val_accuracy"]
                best_metrics = {**env["metrics"], **val_metrics}

        if step % checkpoint_interval == 0 and step > 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": env["metrics"],
            }, ckpt_path)

        kernel.apply(env, events=["train_step_end"])
        step += 1
        if step_delay > 0:
            time.sleep(step_delay)

    val_metrics = _validate_imagenet(model, val_loader, device)
    if val_metrics.get("val_accuracy", 0) > best_val_acc:
        best_metrics = val_metrics

    ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": val_metrics,
    }, ckpt_path)

    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)


def imagenet_mobilenet_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """DEPRECATED: Continuation for pretrained fine-tune.

    Use imagenet_mobilenetv2_paper_continuation_training() instead.
    """
    imagenet_mobilenet_training(
        run_dir,
        max_steps=max_steps,
        step_delay=step_delay,
        _stop_event=_stop_event,
        checkpoint_path=checkpoint_path,
        resume_mode=resume_mode,
    )


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

def _validate_imagenet(model, val_loader, device, use_amp: bool = False) -> Dict[str, float]:
    """Validation with top-1 and top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct_1 += (logits.argmax(dim=1) == labels).sum().item()
            correct_5 += (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    model.train()
    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct_1 / max(total, 1),
        "val_top5_accuracy": correct_5 / max(total, 1),
    }
