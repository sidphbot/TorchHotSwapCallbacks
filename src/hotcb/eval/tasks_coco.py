"""COCO MobileNetV2 classification — real-world evaluation task.

Custom task: 80-class classification on COCO primary categories.
NOT from MobileNetV2 paper — the paper uses COCO for SSDLite object detection only.
See tasks_coco_detection.py for paper-faithful SSDLite + MobileNetV2 detection.

MobileNetV2 (pretrained) + Linear(1280→80) on COCO primary categories.
Uses the same HotKernel + MetricsCollector + actuator integration as MNIST/CIFAR-10 tasks.

Requirements: torchvision, PIL
Data: /media/burplord/kraken_data/off_domain_data/coco/

Designed for moderate GPU usage:
  - ~2GB VRAM (batch_size=64, MobileNetV2 ~3.4M params)
  - ~10-15 min on GPU for 5000 steps
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


def _get_coco_transforms():
    """ImageNet-normalized transforms for MobileNetV2."""
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
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


def _build_coco_model(num_classes: int = 80, pretrained: bool = True):
    """MobileNetV2 with replaced classifier head."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def coco_mobilenet_training(
    run_dir: str,
    *,
    max_steps: int = 5000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Train MobileNetV2 on COCO classification with HotKernel integration."""
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCOClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    # Data
    train_tf, val_tf = _get_coco_transforms()
    train_ds = COCOClassification(split="train", transform=train_tf)
    val_ds = COCOClassification(split="val", transform=val_tf, max_samples=5000)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # Model
    model = _build_coco_model(num_classes=80)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Resume from checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if resume_mode == "full_resume" and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Actuators
    clip_act, clip_container = grad_clip_actuator(initial_value=1.0)
    model_container = {"swa_enabled": False, "ema_enabled": False}
    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)
    acts.extend(safety_actuators())
    ms = mutable_state(acts)

    # Kernel + metrics
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    # Training loop
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

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_container["clip_value"]).item()

        optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "train_loss": round(loss.item(), 6),
                "train_accuracy": round(acc, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        # Validation every 500 steps
        if step % 500 == 0 and step > 0:
            val_metrics = _validate_coco(model, val_loader, device)
            env["metrics"]["val_loss"] = round(val_metrics["val_loss"], 6)
            env["metrics"]["val_accuracy"] = round(val_metrics["val_accuracy"], 4)

        kernel.apply(env, events=["train_step_end"])

        step += 1
        if step_delay > 0:
            time.sleep(step_delay)

    # Final validation
    val_metrics = _validate_coco(model, val_loader, device)
    env = {
        "framework": "torch",
        "phase": "train",
        "step": step,
        "optimizer": optimizer,
        "metrics": {
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_accuracy": round(val_metrics["val_accuracy"], 4),
        },
        "log": lambda s: None,
    }
    kernel.apply(env, events=["train_step_end"])
    kernel.close(env)


def coco_mobilenet_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 5000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval: int = 1000,
) -> None:
    """Train MobileNetV2 on COCO with periodic checkpoint saving."""
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCOClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    # Data
    train_tf, val_tf = _get_coco_transforms()
    train_ds = COCOClassification(split="train", transform=train_tf)
    val_ds = COCOClassification(split="val", transform=val_tf, max_samples=5000)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # Model + optimizer
    model = _build_coco_model(num_classes=80)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Actuators
    clip_act, clip_container = grad_clip_actuator(initial_value=1.0)
    model_container = {"swa_enabled": False, "ema_enabled": False}
    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)
    acts.extend(safety_actuators())
    ms = mutable_state(acts)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    # Checkpoint dir
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

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "train_loss": round(loss.item(), 6),
                "train_accuracy": round(acc, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        # Validation + checkpoint
        if step % 500 == 0 and step > 0:
            val_metrics = _validate_coco(model, val_loader, device)
            env["metrics"]["val_loss"] = round(val_metrics["val_loss"], 6)
            env["metrics"]["val_accuracy"] = round(val_metrics["val_accuracy"], 4)
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

    # Final save
    val_metrics = _validate_coco(model, val_loader, device)
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


def coco_mobilenet_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Continuation variant — resumes from checkpoint with actuator support."""
    coco_mobilenet_training(
        run_dir,
        max_steps=max_steps,
        step_delay=step_delay,
        _stop_event=_stop_event,
        checkpoint_path=checkpoint_path,
        resume_mode=resume_mode,
    )


def _validate_coco(model, val_loader, device) -> Dict[str, float]:
    """Run validation pass, return metrics dict."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct / max(total, 1),
    }
