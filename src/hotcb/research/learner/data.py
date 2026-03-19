"""JSONL-backed training dataset for the NN learner."""

from __future__ import annotations

import json
import os
from typing import List, Tuple


class TrainingDataset:
    """Append-only JSONL dataset of (feature_vector, label) pairs."""

    def __init__(self, path: str):
        self._path = path
        self._samples: List[Tuple[List[float], float]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._samples.append((row["features"], row["label"]))

    def add(self, features: List[float], label: float):
        self._samples.append((features, label))
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps({"features": features, "label": label}) + "\n")

    def __len__(self) -> int:
        return len(self._samples)

    def features(self) -> List[List[float]]:
        return [s[0] for s in self._samples]

    def labels(self) -> List[float]:
        return [s[1] for s in self._samples]

    def as_tensors(self):
        """Return (X, y) as torch tensors. Requires torch."""
        import torch
        X = torch.tensor(self.features(), dtype=torch.float32)
        y = torch.tensor(self.labels(), dtype=torch.float32).unsqueeze(1)
        return X, y
