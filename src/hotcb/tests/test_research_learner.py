"""Tests for the NN learner module (features, data, model)."""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict

import pytest

from hotcb.research.learner.features import (
    extract_features, features_from_intervention, FEATURE_DIM,
)
from hotcb.research.learner.data import TrainingDataset

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class MockEffect:
    param_key: str = "lr"
    step: int = 100
    old_value: float = 0.01
    new_value: float = 0.005
    baseline_metrics: Dict[str, float] = None
    final_metrics: Dict[str, float] = None
    delta: Dict[str, float] = None
    outcome: str = "improved"

    def __post_init__(self):
        if self.baseline_metrics is None:
            self.baseline_metrics = {"train_loss": 1.0, "grad_norm": 5.0}
        if self.final_metrics is None:
            self.final_metrics = {"train_loss": 0.8}
        if self.delta is None:
            self.delta = {"train_loss": -0.2}


class TestFeatures:
    def test_extract_features_dimension(self):
        effect = MockEffect()
        features = extract_features(effect)
        assert len(features) == FEATURE_DIM
        assert all(isinstance(f, float) for f in features)

    def test_extract_features_with_context(self):
        effect = MockEffect()
        ctx = {"loss_trend_slope": -0.1, "is_plateau": 1}
        features = extract_features(effect, ctx)
        assert features[1] == -0.1  # loss_trend_slope
        assert features[9] == 1.0   # is_plateau

    def test_features_from_intervention(self):
        intervention = {"params": {"lr_mult": 0.5}}
        context = {"loss_value": 2.0, "grad_norm": 10.0}
        features = features_from_intervention(intervention, context)
        assert len(features) == FEATURE_DIM
        assert features[0] == 2.0  # loss_value

    def test_extract_non_numeric_value(self):
        effect = MockEffect(old_value="high", new_value="low")
        features = extract_features(effect)
        assert len(features) == FEATURE_DIM


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class TestTrainingDataset:
    def test_add_and_read(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        ds.add([1.0] * FEATURE_DIM, 1.0)
        ds.add([0.5] * FEATURE_DIM, 0.0)
        assert len(ds) == 2
        assert ds.labels() == [1.0, 0.0]

    def test_persistence(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.jsonl")
        ds1 = TrainingDataset(path)
        ds1.add([1.0] * FEATURE_DIM, 1.0)

        ds2 = TrainingDataset(path)
        assert len(ds2) == 1
        assert ds2.features()[0] == [1.0] * FEATURE_DIM

    def test_empty_dataset(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Model (requires torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestOutcomePredictor:
    def test_build_and_predict(self):
        from hotcb.research.learner.model import OutcomePredictor
        model = OutcomePredictor()
        # Build with random weights (no pretrained)
        model._model = model._build_model()
        model._model.eval()
        model._trained = True
        score = model.predict({"params": {"lr": 0.001}}, {})
        assert 0.0 <= score <= 1.0

    def test_train_from_dataset(self, tmp_dir):
        from hotcb.research.learner.model import OutcomePredictor
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        for i in range(20):
            ds.add([float(i) / 20.0] * FEATURE_DIM, 1.0 if i > 10 else 0.0)

        model = OutcomePredictor()
        model.train_from_dataset(ds, epochs=10)
        assert model._trained
        score = model.predict({"params": {"lr": 0.001}}, {})
        assert 0.0 <= score <= 1.0

    def test_finetune(self, tmp_dir):
        from hotcb.research.learner.model import OutcomePredictor
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        for i in range(10):
            ds.add([float(i) / 10.0] * FEATURE_DIM, 1.0 if i > 5 else 0.0)

        model = OutcomePredictor()
        model.train_from_dataset(ds, epochs=5)
        model.finetune(ds, epochs=2)
        assert model._trained

    def test_suggest(self, tmp_dir):
        from hotcb.research.learner.model import OutcomePredictor
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        for i in range(20):
            ds.add([float(i) / 20.0] * FEATURE_DIM, 1.0 if i > 10 else 0.0)

        model = OutcomePredictor()
        model.train_from_dataset(ds, epochs=10)
        suggestions = model.suggest({}, top_k=3)
        assert len(suggestions) == 3
        assert all("predicted_p_improved" in s for s in suggestions)

    def test_save_and_load(self, tmp_dir):
        from hotcb.research.learner.model import OutcomePredictor
        path = os.path.join(tmp_dir, "data.jsonl")
        ds = TrainingDataset(path)
        for i in range(10):
            ds.add([float(i) / 10.0] * FEATURE_DIM, 1.0 if i > 5 else 0.0)

        model = OutcomePredictor()
        model.train_from_dataset(ds, epochs=5)

        save_path = os.path.join(tmp_dir, "model.pt")
        model.save(save_path)
        assert os.path.exists(save_path)

        model2 = OutcomePredictor()
        model2._model = model2._build_model()
        import torch
        model2._model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
        model2._model.eval()
        model2._trained = True
        score = model2.predict({"params": {"lr": 0.001}}, {})
        assert 0.0 <= score <= 1.0

    def test_status(self):
        from hotcb.research.learner.model import OutcomePredictor
        model = OutcomePredictor()
        s = model.status
        assert s["loaded"] is False
        assert s["feature_dim"] == FEATURE_DIM


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPretrain:
    def test_generate_synthetic(self):
        from hotcb.research.learner.pretrain import generate_synthetic_samples
        samples = generate_synthetic_samples(50)
        assert len(samples) == 50
        assert len(samples[0][0]) == FEATURE_DIM

    def test_augment(self):
        from hotcb.research.learner.pretrain import generate_synthetic_samples, augment_samples
        samples = generate_synthetic_samples(20)
        augmented = augment_samples(samples, factor=1.5)
        assert len(augmented) > 0

    def test_pretrain_pipeline(self, tmp_dir):
        from hotcb.research.learner.pretrain import pretrain
        out_path = os.path.join(tmp_dir, "model.pt")
        result = pretrain(output_path=out_path, epochs=5)
        assert os.path.exists(out_path)
        assert result["total_samples"] > 200
