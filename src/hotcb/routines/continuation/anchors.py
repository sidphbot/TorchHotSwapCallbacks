"""Anchor selection — choose checkpoints to branch from."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .models import AnchorReason, AnchorSpec


class AnchorSelector:
    """Select anchor checkpoints from a completed base run."""

    def __init__(self, run_dir: str, checkpoint_dir: Optional[str] = None):
        self._run_dir = run_dir
        self._ckpt_dir = checkpoint_dir or os.path.join(run_dir, "checkpoints")

    def discover_checkpoints(self) -> List[Dict]:
        """Find all saved checkpoints and their metadata."""
        if not os.path.isdir(self._ckpt_dir):
            return []

        checkpoints = []
        for fname in sorted(os.listdir(self._ckpt_dir)):
            if not fname.endswith(".pt") and not fname.endswith(".pth"):
                continue
            meta_path = os.path.join(self._ckpt_dir, fname.rsplit(".", 1)[0] + ".meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            checkpoints.append({
                "path": os.path.join(self._ckpt_dir, fname),
                "filename": fname,
                "step": meta.get("step", 0),
                "epoch": meta.get("epoch", 0),
                "metrics": meta.get("metrics", {}),
            })
        return sorted(checkpoints, key=lambda c: c["step"])

    def select_auto(
        self,
        max_anchors: int = 2,
        primary_metric: str = "val_loss",
        mode: str = "min",
    ) -> List[AnchorSpec]:
        """Automatic anchor selection heuristic.

        Strategy:
        - Anchor A: checkpoint near best validation metric (best_val - small window)
        - Anchor B: checkpoint near end of training (end - medium window)
        """
        checkpoints = self.discover_checkpoints()
        if not checkpoints:
            return []

        anchors: List[AnchorSpec] = []

        # Find best checkpoint by primary metric
        best_ckpt = None
        best_val = None
        for ckpt in checkpoints:
            val = ckpt["metrics"].get(primary_metric)
            if val is None:
                continue
            if best_val is None:
                best_ckpt = ckpt
                best_val = val
            elif (mode == "min" and val < best_val) or (mode == "max" and val > best_val):
                best_ckpt = ckpt
                best_val = val

        # Anchor A: best_val checkpoint (or one step before if available)
        if best_ckpt is not None:
            # Try to find a checkpoint slightly before the best
            best_idx = checkpoints.index(best_ckpt)
            anchor_idx = max(0, best_idx - 1) if best_idx > 0 else best_idx
            ckpt = checkpoints[anchor_idx]
            anchors.append(AnchorSpec(
                anchor_id=f"best_val_anchor",
                checkpoint_path=ckpt["path"],
                step=ckpt["step"],
                epoch=ckpt["epoch"],
                reason=AnchorReason.BEST_VAL_MINUS_K,
                metrics_at_anchor=ckpt["metrics"],
            ))

        # Anchor B: near end of training
        if len(checkpoints) >= 2 and len(anchors) < max_anchors:
            # Pick checkpoint at ~80% of training
            end_idx = max(0, len(checkpoints) - 2)
            # Avoid duplicating anchor A
            if not anchors or checkpoints[end_idx]["path"] != anchors[0].checkpoint_path:
                ckpt = checkpoints[end_idx]
                anchors.append(AnchorSpec(
                    anchor_id=f"end_minus_anchor",
                    checkpoint_path=ckpt["path"],
                    step=ckpt["step"],
                    epoch=ckpt["epoch"],
                    reason=AnchorReason.END_MINUS_K,
                    metrics_at_anchor=ckpt["metrics"],
                ))

        return anchors[:max_anchors]

    def select_manual(self, checkpoint_paths: List[str]) -> List[AnchorSpec]:
        """Create anchors from manually specified checkpoint paths."""
        anchors = []
        for i, path in enumerate(checkpoint_paths):
            meta_path = path.rsplit(".", 1)[0] + ".meta.json"
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            anchors.append(AnchorSpec(
                anchor_id=f"manual_{i}",
                checkpoint_path=path,
                step=meta.get("step", 0),
                epoch=meta.get("epoch", 0),
                reason=AnchorReason.MANUAL,
                metrics_at_anchor=meta.get("metrics", {}),
            ))
        return anchors

    def detect_plateau_checkpoint(
        self,
        primary_metric: str = "val_loss",
        mode: str = "min",
        window: int = 5,
    ) -> Optional[AnchorSpec]:
        """Find the checkpoint just before the metric plateaued."""
        checkpoints = self.discover_checkpoints()
        metric_vals = []
        for ckpt in checkpoints:
            val = ckpt["metrics"].get(primary_metric)
            if val is not None:
                metric_vals.append((ckpt, val))

        if len(metric_vals) < window + 1:
            return None

        # Look for first window where improvement < threshold
        for i in range(len(metric_vals) - window):
            segment = [v for _, v in metric_vals[i:i + window]]
            if mode == "min":
                improvement = segment[0] - segment[-1]
            else:
                improvement = segment[-1] - segment[0]

            # Plateau: less than 1% relative improvement over window
            baseline = abs(segment[0]) if segment[0] != 0 else 1.0
            if improvement / baseline < 0.01:
                # Anchor is the checkpoint just before plateau
                anchor_idx = max(0, i - 1)
                ckpt = metric_vals[anchor_idx][0]
                return AnchorSpec(
                    anchor_id="before_plateau",
                    checkpoint_path=ckpt["path"],
                    step=ckpt["step"],
                    epoch=ckpt["epoch"],
                    reason=AnchorReason.BEFORE_PLATEAU,
                    metrics_at_anchor=ckpt["metrics"],
                )

        return None
