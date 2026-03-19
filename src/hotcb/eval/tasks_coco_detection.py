"""COCO SSDLite + MobileNetV2 object detection (Sandler et al. 2018, Table 6).

Paper recipe: MobileNetV2 backbone + SSDLite heads, 320×320 input, 22.1 mAP.
Torchvision has SSDLite for MobileNetV3 but NOT MobileNetV2 — we build a custom
feature extractor because MobileNetV2 InvertedResidual uses `.conv` while the
MobileNetV3 extractor expects `.block`.

Requirements: torchvision, pycocotools
Data: /media/burplord/kraken_data/off_domain_data/coco/

VRAM: ~4-6GB (batch_size=24, 320×320, SSD + backbone)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Training hyperparameters (SSD standard + paper)
DETECTION_LR = 0.01
DETECTION_MOMENTUM = 0.9
DETECTION_WD = 4e-5
DETECTION_BATCH_SIZE = 24
DETECTION_EPOCHS = 200
DETECTION_INPUT_SIZE = (320, 320)


def _build_mobilenetv2_ssdlite(num_classes: int = 91, pretrained_backbone: bool = True):
    """Build SSDLite with MobileNetV2 backbone.

    MobileNetV2 features are split at layer 14 (C4, stride 16).
    Extra InvertedResidual blocks provide multi-scale feature maps.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from torchvision.models.detection.ssd import SSD
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

    weights = MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
    backbone_model = mobilenet_v2(weights=weights)

    feature_extractor = SSDLiteFeatureExtractorMobileNetV2(backbone_model)

    # 6 feature maps, aspect ratios [2, 3] each
    anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)],
        min_ratio=0.2,
        max_ratio=0.95,
    )

    # Get output channels by running a dummy forward
    out_channels = _get_out_channels(feature_extractor, DETECTION_INPUT_SIZE)
    num_anchors = anchor_generator.num_anchors_per_location()

    head = SSDLiteDetectionHead(out_channels, num_anchors, num_classes)

    model = SSD(
        feature_extractor,
        anchor_generator,
        DETECTION_INPUT_SIZE,
        num_classes,
        head=head,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model


class SSDLiteFeatureExtractorMobileNetV2(nn.Module):
    """Extract multi-scale features from MobileNetV2 for SSD.

    Splits MobileNetV2 at layer 14 (C4 position, output stride 16):
    - features[0:14] → C4 feature map (expanded channels)
    - features[14:] → C5 feature map (1280 after conv)
    - 4 extra InvertedResidual-style blocks for smaller scales

    Returns OrderedDict with 6 feature maps.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        features = backbone.features

        # Split backbone
        self.features_c4 = nn.Sequential(*features[:14])  # → stride 16
        self.features_c5 = nn.Sequential(*features[14:])  # → stride 32 (1280 ch)

        # Extra layers for multi-scale detection (stride 64, 128, 256, 512)
        self.extra = nn.ModuleList([
            _InvertedResidualBlock(1280, 512, stride=2),
            _InvertedResidualBlock(512, 256, stride=2),
            _InvertedResidualBlock(256, 256, stride=2),
            _InvertedResidualBlock(256, 128, stride=2),
        ])

        # Output channel counts for each scale
        self._out_channels = [
            features[13].conv[0][0].out_channels if hasattr(features[13], 'conv') else 96,
            1280,  # MobileNetV2 final conv
            512, 256, 256, 128,
        ]

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features = OrderedDict()

        c4 = self.features_c4(x)
        features["0"] = c4

        c5 = self.features_c5(c4)
        features["1"] = c5

        prev = c5
        for i, block in enumerate(self.extra):
            prev = block(prev)
            features[str(i + 2)] = prev

        return features


class _InvertedResidualBlock(nn.Module):
    """Lightweight depthwise-separable block for extra SSD layers."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        mid = in_channels
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SSDLiteDetectionHead(nn.Module):
    """SSDLite prediction head using depthwise separable convolutions."""

    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
    ):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[torch.Tensor]):
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }


class SSDLiteClassificationHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.cls_heads = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            self.cls_heads.append(_dw_sep_conv(channels, anchors * num_classes))
        self.num_classes = num_classes

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        all_cls = []
        for feat, head in zip(x, self.cls_heads):
            out = head(feat)
            out = out.permute(0, 2, 3, 1).contiguous()
            out = out.reshape(out.size(0), -1, self.num_classes)
            all_cls.append(out)
        return torch.cat(all_cls, dim=1)


class SSDLiteRegressionHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        super().__init__()
        self.reg_heads = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            self.reg_heads.append(_dw_sep_conv(channels, anchors * 4))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        all_reg = []
        for feat, head in zip(x, self.reg_heads):
            out = head(feat)
            out = out.permute(0, 2, 3, 1).contiguous()
            out = out.reshape(out.size(0), -1, 4)
            all_reg.append(out)
        return torch.cat(all_reg, dim=1)


def _dw_sep_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Depthwise separable conv for SSDLite heads."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1),
    )


def _get_out_channels(feature_extractor: nn.Module, size: Tuple[int, int]) -> List[int]:
    """Run dummy forward to get output channels per feature map."""
    feature_extractor.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, size[0], size[1])
        features = feature_extractor(dummy)
    feature_extractor.train()
    return [f.size(1) for f in features.values()]


def _get_detection_transforms():
    """Detection transforms: resize to 320×320 + normalize."""
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize(DETECTION_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(DETECTION_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def coco_ssdlite_mobilenetv2_training(
    run_dir: str,
    *,
    max_steps: int = 0,
    num_epochs: int = DETECTION_EPOCHS,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
    use_amp: bool = True,
) -> None:
    """Train SSDLite + MobileNetV2 on COCO detection — paper Table 6 recipe.

    Target: 22.1 mAP (paper), 320×320 input.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCODetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()
    torch.backends.cudnn.benchmark = True

    # Data
    train_ds = COCODetection(split="train")
    val_ds = COCODetection(split="val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=DETECTION_BATCH_SIZE, shuffle=True,
        num_workers=12, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=_detection_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=DETECTION_BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=_detection_collate,
    )

    # Model
    model = _build_mobilenetv2_ssdlite(num_classes=91, pretrained_backbone=True)
    model.to(device)

    # SGD (standard for detection)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=DETECTION_LR,
        momentum=DETECTION_MOMENTUM, weight_decay=DETECTION_WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # Resume
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
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)

    # Actuators
    clip_act, clip_container = grad_clip_actuator(initial_value=10.0)
    model_container = {"swa_enabled": False, "ema_enabled": False}
    acts = optimizer_actuators(optimizer)
    acts.append(swa_actuator(model_container))
    acts.append(ema_actuator(model_container))
    acts.append(clip_act)
    acts.extend(safety_actuators())
    ms = mutable_state(acts)

    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, mutable_state=ms, metrics_collector=mc)

    model.train()
    for epoch in range(start_epoch, num_epochs):
        if stop.is_set():
            break

        for batch_idx, (images, targets) in enumerate(train_loader):
            if stop.is_set():
                break
            if max_steps > 0 and global_step >= max_steps:
                break

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss_dict.values())
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                losses.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                optimizer.step()

            cls_loss = loss_dict.get("classification", torch.tensor(0.0)).item()
            reg_loss = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()

            env = {
                "framework": "torch",
                "phase": "train",
                "step": global_step,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(losses.item(), 6),
                    "train_cls_loss": round(cls_loss, 6),
                    "train_reg_loss": round(reg_loss, 6),
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

        scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = _evaluate_coco_detection(model, val_loader, device, use_amp=use_amp)
            env = {
                "framework": "torch",
                "phase": "val",
                "step": global_step,
                "optimizer": optimizer,
                "metrics": {
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                "log": lambda s: None,
            }
            kernel.apply(env, events=["val_epoch_end"])
            model.train()

    kernel.close({"step": global_step, "log": lambda s: None})


def coco_ssdlite_mobilenetv2_training_with_checkpoints(
    run_dir: str,
    *,
    max_steps: int = 0,
    num_epochs: int = DETECTION_EPOCHS,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_interval_epochs: int = 10,
    use_amp: bool = True,
) -> None:
    """SSDLite + MobileNetV2 training with periodic checkpoint saves."""
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import (
        optimizer_actuators, safety_actuators,
        grad_clip_actuator, swa_actuator, ema_actuator,
        mutable_state,
    )
    from .datasets import COCODetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop = _stop_event or threading.Event()
    torch.backends.cudnn.benchmark = True

    train_ds = COCODetection(split="train")
    val_ds = COCODetection(split="val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=DETECTION_BATCH_SIZE, shuffle=True,
        num_workers=12, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=_detection_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=DETECTION_BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=_detection_collate,
    )

    model = _build_mobilenetv2_ssdlite(num_classes=91, pretrained_backbone=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=DETECTION_LR,
        momentum=DETECTION_MOMENTUM, weight_decay=DETECTION_WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    clip_act, clip_container = grad_clip_actuator(initial_value=10.0)
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
    best_mAP = 0.0
    best_metrics: Dict[str, float] = {}

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        if stop.is_set():
            break

        for batch_idx, (images, targets) in enumerate(train_loader):
            if stop.is_set():
                break
            if max_steps > 0 and global_step >= max_steps:
                break

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss_dict.values())
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                losses.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_container["clip_value"]
                ).item()
                optimizer.step()

            env = {
                "framework": "torch",
                "phase": "train",
                "step": global_step,
                "optimizer": optimizer,
                "metrics": {
                    "train_loss": round(losses.item(), 6),
                    "train_cls_loss": round(loss_dict.get("classification", torch.tensor(0.0)).item(), 6),
                    "train_reg_loss": round(loss_dict.get("bbox_regression", torch.tensor(0.0)).item(), 6),
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

        scheduler.step()

        # Validation + checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = _evaluate_coco_detection(model, val_loader, device, use_amp=use_amp)
            val_env = {
                "framework": "torch",
                "phase": "val",
                "step": global_step,
                "optimizer": optimizer,
                "metrics": {**val_metrics, "lr": optimizer.param_groups[0]["lr"], "epoch": epoch},
                "log": lambda s: None,
            }
            kernel.apply(val_env, events=["val_epoch_end"])

            if val_metrics.get("val_mAP", 0) > best_mAP:
                best_mAP = val_metrics["val_mAP"]
                best_metrics = {**val_env["metrics"], **val_metrics}

        if (epoch + 1) % checkpoint_interval_epochs == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_mAP": best_mAP,
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, ckpt_path)

            if val_metrics.get("val_mAP", 0) >= best_mAP:
                torch.save(save_dict, os.path.join(ckpt_dir, "best.pt"))

        model.train()

    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)


def coco_ssdlite_mobilenetv2_continuation_training(
    run_dir: str,
    *,
    max_steps: int = 5000,
    step_delay: float = 0.0,
    _stop_event: Optional[threading.Event] = None,
    checkpoint_path: Optional[str] = None,
    resume_mode: str = "full_resume",
) -> None:
    """Continuation variant — resume from checkpoint."""
    coco_ssdlite_mobilenetv2_training(
        run_dir,
        max_steps=max_steps,
        num_epochs=9999,
        step_delay=step_delay,
        _stop_event=_stop_event,
        checkpoint_path=checkpoint_path,
        resume_mode=resume_mode,
    )


def _detection_collate(batch):
    """Custom collate for detection — images as list, targets as list of dicts."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def _evaluate_coco_detection(
    model, val_loader, device, use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate detection model, return mAP metrics.

    Uses simple per-class AP calculation. For full COCO mAP evaluation,
    use pycocotools (requires `hotcb[detection]` extra).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Simple evaluation: run forward in eval mode to get detections,
    # then compute average precision per class
    all_predictions = []
    all_targets_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device, non_blocking=True) for img in images]

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    detections = model(images)
            else:
                detections = model(images)

            # detections is a list of dicts with 'boxes', 'scores', 'labels'
            all_predictions.extend(detections)
            all_targets_list.extend(targets)
            num_batches += 1

    # Compute simple mAP@0.5 using IoU matching
    mAP_50 = _compute_simple_mAP(all_predictions, all_targets_list, iou_threshold=0.5)
    mAP_75 = _compute_simple_mAP(all_predictions, all_targets_list, iou_threshold=0.75)

    # Average over IoU thresholds 0.5:0.05:0.95 for COCO-style mAP
    mAP_thresholds = [0.5 + 0.05 * i for i in range(10)]
    mAPs = [_compute_simple_mAP(all_predictions, all_targets_list, iou_threshold=t) for t in mAP_thresholds]
    mAP = sum(mAPs) / len(mAPs)

    model.train()
    return {
        "val_mAP": round(mAP, 4),
        "val_mAP_50": round(mAP_50, 4),
        "val_mAP_75": round(mAP_75, 4),
    }


def _compute_simple_mAP(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Simplified mAP calculation using per-class AP with IoU matching."""
    from collections import defaultdict

    # Gather all predictions and ground truths per class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(list)

    for img_idx, (pred, gt) in enumerate(zip(predictions, targets)):
        if "scores" in pred:
            for score, label, box in zip(
                pred["scores"].cpu(), pred["labels"].cpu(), pred["boxes"].cpu()
            ):
                pred_by_class[label.item()].append((score.item(), box, img_idx))

        if "labels" in gt:
            for label, box in zip(gt["labels"].cpu(), gt["boxes"].cpu()):
                gt_by_class[label.item()].append((box, img_idx))

    if not gt_by_class:
        return 0.0

    # Per-class AP
    aps = []
    for cls_id in gt_by_class:
        preds = pred_by_class.get(cls_id, [])
        gts = gt_by_class[cls_id]

        if not preds:
            aps.append(0.0)
            continue

        # Sort by score descending
        preds.sort(key=lambda x: -x[0])
        n_gt = len(gts)
        matched = set()

        tp = []
        fp = []
        for score, pred_box, pred_img in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_img) in enumerate(gts):
                if gt_img != pred_img:
                    continue
                if gt_idx in matched:
                    continue
                iou = _box_iou_single(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        # Cumulative precision/recall
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        for t, f in zip(tp, fp):
            tp_cum += t
            fp_cum += f
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)

        # 11-point interpolation AP
        ap = 0.0
        for r_thresh in [i / 10.0 for i in range(11)]:
            p_at_r = 0.0
            for p, r in zip(precisions, recalls):
                if r >= r_thresh:
                    p_at_r = max(p_at_r, p)
            ap += p_at_r / 11.0
        aps.append(ap)

    return sum(aps) / max(len(aps), 1)


def _box_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)
