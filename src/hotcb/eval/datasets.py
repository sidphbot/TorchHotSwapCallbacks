"""Dataset classes for real-world evaluation tasks."""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class COCOClassification(Dataset):
    """80-class classification on COCO — primary category per image.

    Reads instances_train2017.json / instances_val2017.json, builds
    image_id → primary_category index. Uses torchvision transforms.

    Data expected at: /media/burplord/kraken_data/off_domain_data/coco/
    Structure:
        coco/
          train2017/          # images
          val2017/            # images
          annotations/
            instances_train2017.json
            instances_val2017.json
    """

    COCO_ROOT = "/media/burplord/kraken_data/off_domain_data/coco"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        root: Optional[str] = None,
        max_samples: int = 0,
    ):
        self.root = root or self.COCO_ROOT
        self.split = split
        self.transform = transform

        img_dir = os.path.join(self.root, f"{split}2017")
        ann_file = os.path.join(
            self.root, "annotations", f"instances_{split}2017.json"
        )

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"COCO annotations not found: {ann_file}\n"
                f"Expected COCO data at {self.root}"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        # Build category id → contiguous index (0-79)
        cat_ids = sorted([c["id"] for c in coco["categories"]])
        self._cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        # Build image_id → (filename, primary_category)
        # Use the first annotation per image as primary category
        img_info = {img["id"]: img["file_name"] for img in coco["images"]}
        img_to_cat: Dict[int, int] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_cat:
                img_to_cat[img_id] = ann["category_id"]

        self.samples: List[Tuple[str, int]] = []
        for img_id, cat_id in img_to_cat.items():
            if img_id in img_info:
                path = os.path.join(img_dir, img_info[img_id])
                label = self._cat_to_idx[cat_id]
                self.samples.append((path, label))

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNetFlat(Dataset):
    """ImageNet with flat directory layout + JSON label mapping.

    Data expected at: /media/burplord/kraken_data/off_domain_data/imagenet/
    Structure:
        imagenet/
          train/              # flat directory of images OR class subdirs
          val/                # flat directory of images OR class subdirs
          train_labels.json   # {filename_stem: class_id} if flat layout
          val_labels.json     # {filename_stem: class_id} if flat layout

    Supports both flat layout (with JSON label mapping) and standard
    ImageFolder layout (class subdirectories).
    """

    IMAGENET_ROOT = "/media/burplord/kraken_data/off_domain_data/imagenet"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        root: Optional[str] = None,
        max_samples: int = 0,
    ):
        self.root = root or self.IMAGENET_ROOT
        self.split = split
        self.transform = transform
        self.num_classes = 1000

        img_dir = os.path.join(self.root, split)
        # Check both root and annotations/ subdir for label files
        label_file = os.path.join(self.root, f"{split}_labels.json")
        if not os.path.exists(label_file):
            label_file = os.path.join(
                self.root, "annotations", f"{split}_labels.json"
            )

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"ImageNet {split} directory not found: {img_dir}\n"
                f"Expected ImageNet data at {self.root}"
            )

        self.samples: List[Tuple[str, int]] = []

        if os.path.exists(label_file):
            # Flat layout with JSON labels
            with open(label_file) as f:
                labels = json.load(f)
            # Detect extension from first file to avoid per-file existence checks
            _ext = ".JPEG"
            for ext in (".jpg", ".jpeg", ".JPEG", ".png"):
                if os.path.exists(os.path.join(img_dir, next(iter(labels)) + ext)):
                    _ext = ext
                    break
            for stem, class_id in labels.items():
                self.samples.append(
                    (os.path.join(img_dir, stem + _ext), int(class_id))
                )
        else:
            # Standard ImageFolder layout
            classes = sorted(
                d for d in os.listdir(img_dir)
                if os.path.isdir(os.path.join(img_dir, d))
            )
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls_name in classes:
                cls_dir = os.path.join(img_dir, cls_name)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append(
                            (os.path.join(cls_dir, fname), class_to_idx[cls_name])
                        )

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class COCODetection(Dataset):
    """COCO object detection dataset — [x1, y1, x2, y2] format for SSD.

    Reads instances_train2017.json / instances_val2017.json.
    Converts COCO [x, y, w, h] → [x1, y1, x2, y2] bboxes.
    Filters iscrowd annotations.

    Data expected at: /media/burplord/kraken_data/off_domain_data/coco/
    """

    COCO_ROOT = "/media/burplord/kraken_data/off_domain_data/coco"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        root: Optional[str] = None,
        max_samples: int = 0,
    ):
        self.root = root or self.COCO_ROOT
        self.split = split
        self.transform = transform

        img_dir = os.path.join(self.root, f"{split}2017")
        ann_file = os.path.join(
            self.root, "annotations", f"instances_{split}2017.json"
        )

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"COCO annotations not found: {ann_file}\n"
                f"Expected COCO data at {self.root}"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        # Build image info lookup
        self._img_info = {img["id"]: img for img in coco["images"]}

        # Group annotations by image_id, filter iscrowd
        from collections import defaultdict
        anns_by_img: dict = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            anns_by_img[ann["image_id"]].append(ann)

        # Only keep images that have at least one annotation
        self.samples: List[Tuple[str, List[dict]]] = []
        for img_id, anns in anns_by_img.items():
            if img_id in self._img_info:
                path = os.path.join(img_dir, self._img_info[img_id]["file_name"])
                self.samples.append((path, anns))

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, anns = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # Convert COCO [x, y, w, h] → [x1, y1, x2, y2]
        boxes = []
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])

        if not boxes:
            # Fallback: return dummy target
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            img = self.transform(img)
            # Scale boxes to match transformed image size
            if isinstance(img, torch.Tensor):
                new_h, new_w = img.shape[-2:]
                scale_x = new_w / w
                scale_y = new_h / h
                target["boxes"][:, [0, 2]] *= scale_x
                target["boxes"][:, [1, 3]] *= scale_y

        return img, target


class COCOCaptions(Dataset):
    """Image-caption pairs from COCO for contrastive learning.

    Reads captions_train2017.json / captions_val2017.json.
    Returns (image, caption_text) pairs.

    Data expected at: /media/burplord/kraken_data/off_domain_data/coco/
    Structure:
        coco/
          train2017/
          val2017/
          annotations/
            captions_train2017.json
            captions_val2017.json
    """

    COCO_ROOT = "/media/burplord/kraken_data/off_domain_data/coco"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        root: Optional[str] = None,
        max_samples: int = 0,
    ):
        self.root = root or self.COCO_ROOT
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        img_dir = os.path.join(self.root, f"{split}2017")
        ann_file = os.path.join(
            self.root, "annotations", f"captions_{split}2017.json"
        )

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"COCO captions not found: {ann_file}\n"
                f"Expected COCO data at {self.root}"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        img_info = {img["id"]: img["file_name"] for img in coco["images"]}

        self.samples: List[Tuple[str, str]] = []
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in img_info:
                path = os.path.join(img_dir, img_info[img_id])
                caption = ann["caption"]
                self.samples.append((path, caption))

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, caption = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.tokenizer:
            caption = self.tokenizer(caption)
        return img, caption
