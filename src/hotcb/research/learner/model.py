"""NN models for the research learner module."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .features import FEATURE_DIM, features_from_intervention

_PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
_PRETRAINED_PATH = os.path.join(_PRETRAINED_DIR, "outcome_predictor.pt")


class OutcomePredictor:
    """Tier 1: MLP 13→32→16→1 (sigmoid). Predicts P(improved)."""

    def __init__(self):
        self._model = None
        self._trained = False

    def _build_model(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def load_pretrained(self) -> bool:
        """Load shipped pre-trained weights. Returns True if successful."""
        if not os.path.exists(_PRETRAINED_PATH):
            return False
        try:
            import torch
            self._model = self._build_model()
            self._model.load_state_dict(torch.load(_PRETRAINED_PATH, map_location="cpu", weights_only=True))
            self._model.eval()
            self._trained = True
            return True
        except Exception:
            self._model = None
            return False

    def train_from_dataset(self, dataset, epochs: int = 100, lr: float = 1e-3):
        """Train the model from a TrainingDataset."""
        import torch
        import torch.nn as nn

        if len(dataset) < 5:
            return

        self._model = self._build_model()
        X, y = dataset.as_tensors()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self._model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self._model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._trained = True

    def finetune(self, dataset, epochs: int = 5, lr: float = 5e-4):
        """Short fine-tuning pass for active learning."""
        if self._model is None:
            self.train_from_dataset(dataset, epochs=epochs, lr=lr)
            return
        import torch
        import torch.nn as nn

        if len(dataset) < 2:
            return

        X, y = dataset.as_tensors()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self._model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self._model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        self._model.eval()

    def predict(self, intervention: dict, context: dict) -> float:
        """Predict P(improved) for a proposed intervention."""
        if self._model is None:
            return 0.5
        import torch
        features = features_from_intervention(intervention, context)
        x = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            return float(self._model(x).item())

    def suggest(self, context: dict, top_k: int = 3) -> List[dict]:
        """Tier 2: grid-search intervention magnitudes to maximize P(improved)."""
        if self._model is None:
            return []

        candidates = []
        # Search over lr multipliers
        for mult in [0.1, 0.2, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
            intervention = {
                "module": "opt",
                "op": "set_params",
                "params": {"lr_mult": mult},
            }
            score = self.predict(intervention, context)
            candidates.append({"intervention": intervention, "predicted_p_improved": score})

        # Search over loss weight changes
        for delta in [-0.3, -0.1, 0.1, 0.3]:
            intervention = {
                "module": "loss",
                "op": "set_params",
                "params": {"weight_delta": delta},
            }
            score = self.predict(intervention, context)
            candidates.append({"intervention": intervention, "predicted_p_improved": score})

        candidates.sort(key=lambda c: c["predicted_p_improved"], reverse=True)
        return candidates[:top_k]

    def discover(self) -> List[dict]:
        """Tier 3: cluster feature vectors for pattern discovery. Placeholder."""
        # Full implementation requires sklearn KMeans, deferred to Phase 7
        return []

    def save(self, path: Optional[str] = None):
        if self._model is None:
            return
        import torch
        path = path or _PRETRAINED_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._model.state_dict(), path)

    @property
    def status(self) -> dict:
        return {
            "loaded": self._model is not None,
            "trained": self._trained,
            "feature_dim": FEATURE_DIM,
            "architecture": "MLP 13→32→16→1",
        }
