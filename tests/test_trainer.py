import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from activation_exp.configs import ExperimentConfig
from activation_exp.trainer import train_one_experiment


def _synthetic_loader(n: int, in_channels: int, size: int, batch_size: int):
    x = torch.randn(n, in_channels, size, size)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_trainer_saves_metrics_and_dist():
    config = ExperimentConfig(
        name="test_mnist_relu", dataset="mnist", activation="relu",
        in_channels=1, num_classes=10, epochs=1,
        batch_size=8, lr=1e-3, input_size=(28, 28),
    )
    train_loader = _synthetic_loader(16, 1, 28, 8)
    val_loader = _synthetic_loader(16, 1, 28, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "test_mnist_relu"
        train_one_experiment(config, train_loader, val_loader, results_dir, torch.device("cpu"))

        assert (results_dir / "metrics.json").exists()
        assert (results_dir / "activation_dist.npz").exists()

        metrics = json.loads((results_dir / "metrics.json").read_text())
        assert metrics["epoch"] == [1]
        assert 0.0 <= metrics["train_acc"][0] <= 1.0
        assert 0.0 <= metrics["val_acc"][0] <= 1.0
        assert 0.0 <= metrics["dead_ratio"][0] <= 1.0

        dist = np.load(results_dir / "activation_dist.npz")
        assert "epoch_1" in dist.files
        assert len(dist["epoch_1"]) > 0


def test_trainer_all_activations_run():
    for act in ["relu", "leaky_relu", "swish", "gelu"]:
        config = ExperimentConfig(
            name=f"test_{act}", dataset="mnist", activation=act,
            in_channels=1, num_classes=10, epochs=1,
            batch_size=8, lr=1e-3, input_size=(28, 28),
        )
        train_loader = _synthetic_loader(8, 1, 28, 8)
        val_loader = _synthetic_loader(8, 1, 28, 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            train_one_experiment(config, train_loader, val_loader, Path(tmpdir), torch.device("cpu"))
