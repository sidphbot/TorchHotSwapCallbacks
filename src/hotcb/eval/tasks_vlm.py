"""CLIP ViT-B/16 fine-tuning on COCO captions — VLM evaluation task.

Contrastive learning: match images to captions using open_clip.
Metrics: contrastive_loss, retrieval_r1, retrieval_r5.

Requirements: open-clip-torch>=2.20, PIL
Data: /media/burplord/kraken_data/off_domain_data/coco/

VRAM: ~4GB (batch_size=32, CLIP ViT-B/16 ~150M params)
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


def clip_coco_finetuning(
    run_dir: str,
    *,
    max_steps: int = 5000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Fine-tune CLIP ViT-B/16 on COCO captions with HotKernel."""
    import open_clip

    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCOCaptions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    # Model + tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.to(device)

    # Resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    # Data
    def tokenize_caption(text):
        return tokenizer([text])[0]

    train_ds = COCOCaptions(
        split="train", transform=preprocess, tokenizer=tokenize_caption,
    )
    val_ds = COCOCaptions(
        split="val", transform=preprocess, tokenizer=tokenize_caption,
        max_samples=5000,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    # Optimizer — very low LR for CLIP fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.1,
    )
    if checkpoint_path and resume_mode == "full_resume":
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if "optimizer_state_dict" in ckpt:
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

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps and not stop.is_set():
        try:
            images, texts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, texts = next(train_iter)

        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()

        # CLIP forward
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Contrastive loss (symmetric)
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(len(images), device=device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_container["clip_value"]).item()
        optimizer.step()

        # Retrieval accuracy (batch-level)
        with torch.no_grad():
            r1 = (logits_per_image.argmax(dim=1) == labels).float().mean().item()
            r5 = (logits_per_image.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "contrastive_loss": round(loss.item(), 6),
                "retrieval_r1": round(r1, 4),
                "retrieval_r5": round(r5, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        if step % 500 == 0 and step > 0:
            val_metrics = _validate_clip(model, val_loader, device)
            env["metrics"]["val_contrastive_loss"] = round(val_metrics["val_contrastive_loss"], 6)
            env["metrics"]["val_retrieval_r1"] = round(val_metrics["val_retrieval_r1"], 4)
            env["metrics"]["val_retrieval_r5"] = round(val_metrics["val_retrieval_r5"], 4)

        kernel.apply(env, events=["train_step_end"])
        step += 1
        if step_delay > 0:
            time.sleep(step_delay)

    val_metrics = _validate_clip(model, val_loader, device)
    env = {
        "framework": "torch",
        "phase": "train",
        "step": step,
        "optimizer": optimizer,
        "metrics": {
            "val_contrastive_loss": round(val_metrics["val_contrastive_loss"], 6),
            "val_retrieval_r1": round(val_metrics["val_retrieval_r1"], 4),
            "val_retrieval_r5": round(val_metrics["val_retrieval_r5"], 4),
        },
        "log": lambda s: None,
    }
    kernel.apply(env, events=["train_step_end"])
    kernel.close(env)


def clip_coco_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 5000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval: int = 1000,
) -> None:
    """CLIP fine-tuning with periodic checkpoints."""
    import open_clip

    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCOCaptions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.to(device)

    def tokenize_caption(text):
        return tokenizer([text])[0]

    train_ds = COCOCaptions(
        split="train", transform=preprocess, tokenizer=tokenize_caption,
    )
    val_ds = COCOCaptions(
        split="val", transform=preprocess, tokenizer=tokenize_caption,
        max_samples=5000,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, weight_decay=0.1,
    )

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

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_r1 = 0.0
    best_metrics: Dict[str, float] = {}

    model.train()
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps and not stop.is_set():
        try:
            images, texts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, texts = next(train_iter)

        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(len(images), device=device)
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_container["clip_value"]).item()
        optimizer.step()

        with torch.no_grad():
            r1 = (logits_per_image.argmax(dim=1) == labels).float().mean().item()
            r5 = (logits_per_image.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        env = {
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "metrics": {
                "contrastive_loss": round(loss.item(), 6),
                "retrieval_r1": round(r1, 4),
                "retrieval_r5": round(r5, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            },
            "log": lambda s: None,
        }

        if step % 500 == 0 and step > 0:
            val_metrics = _validate_clip(model, val_loader, device)
            env["metrics"]["val_contrastive_loss"] = round(val_metrics["val_contrastive_loss"], 6)
            env["metrics"]["val_retrieval_r1"] = round(val_metrics["val_retrieval_r1"], 4)
            env["metrics"]["val_retrieval_r5"] = round(val_metrics["val_retrieval_r5"], 4)
            if val_metrics.get("val_retrieval_r1", 0) > best_r1:
                best_r1 = val_metrics["val_retrieval_r1"]
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

    val_metrics = _validate_clip(model, val_loader, device)
    if val_metrics.get("val_retrieval_r1", 0) > best_r1:
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


def clip_coco_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Continuation variant for CLIP fine-tuning."""
    clip_coco_finetuning(
        run_dir,
        max_steps=max_steps,
        step_delay=step_delay,
        _stop_event=_stop_event,
        checkpoint_path=checkpoint_path,
        resume_mode=resume_mode,
    )


def _validate_clip(model, val_loader, device) -> Dict[str, float]:
    """Validate CLIP: contrastive loss + retrieval metrics."""
    model.eval()
    total_loss = 0.0
    total_r1 = 0.0
    total_r5 = 0.0
    total = 0
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            texts = texts.to(device)
            bs = images.size(0)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T

            labels = torch.arange(bs, device=device)
            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            total_loss += loss.item() * bs

            total_r1 += (logits_per_image.argmax(dim=1) == labels).sum().item()
            total_r5 += (logits_per_image.topk(min(5, bs), dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += bs
    model.train()
    return {
        "val_contrastive_loss": total_loss / max(total, 1),
        "val_retrieval_r1": total_r1 / max(total, 1),
        "val_retrieval_r5": total_r5 / max(total, 1),
    }
