import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .configs import ExperimentConfig
from .metrics import compute_dead_ratio, sample_activation_dist
from .model import ACTIVATIONS, CNN, DeepCNN


def train_one_experiment(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    results_dir: Path,
    device: torch.device,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    model_cls = DeepCNN if config.model_type == "deep_cnn" else CNN
    model = model_cls(config.in_channels, config.num_classes, config.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    activation_cls = ACTIVATIONS[config.activation]

    metrics: dict = {
        "config": {
            "dataset": config.dataset,
            "activation": config.activation,
            "epochs": config.epochs,
            "model_type": config.model_type,
        },
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "dead_ratio": [],
    }
    dist_snapshots: dict[str, np.ndarray] = {}

    for epoch in range(1, config.epochs + 1):
        # --- train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += x.size(0)

        # --- validate ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)

        dead_ratio = compute_dead_ratio(model, val_loader, device, activation_cls)

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(t_loss / t_total)
        metrics["train_acc"].append(t_correct / t_total)
        metrics["val_loss"].append(v_loss / v_total)
        metrics["val_acc"].append(v_correct / v_total)
        metrics["dead_ratio"].append(dead_ratio)

        if epoch % 5 == 0 or epoch == 1:
            dist = sample_activation_dist(model, val_loader, device, activation_cls)
            dist_snapshots[f"epoch_{epoch}"] = dist

        print(
            f"[{config.name}] {epoch}/{config.epochs}"
            f"  train_acc={metrics['train_acc'][-1]:.3f}"
            f"  val_acc={metrics['val_acc'][-1]:.3f}"
            f"  dead={dead_ratio:.3f}"
        )

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(results_dir / "activation_dist.npz", **dist_snapshots)
