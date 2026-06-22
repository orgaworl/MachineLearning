import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from activation_exp.metrics import compute_dead_ratio, sample_activation_dist


class _AllNegReLU(nn.Module):
    """ReLU fed always-negative input → 100% dead."""
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class _AllPosReLU(nn.Module):
    """ReLU fed always-positive input → 0% dead."""
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


def _make_loader(x, batch_size=8):
    y = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_dead_ratio_all_dead():
    model = _AllNegReLU()
    loader = _make_loader(torch.full((16, 10), -1.0))
    ratio = compute_dead_ratio(model, loader, torch.device("cpu"), nn.ReLU)
    assert ratio == 1.0


def test_dead_ratio_none_dead():
    model = _AllPosReLU()
    loader = _make_loader(torch.ones(16, 10))
    ratio = compute_dead_ratio(model, loader, torch.device("cpu"), nn.ReLU)
    assert ratio == 0.0


def test_dead_ratio_gelu_returns_float_in_range():
    class _GELUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x)

    model = _GELUModel()
    loader = _make_loader(torch.randn(16, 10))
    ratio = compute_dead_ratio(model, loader, torch.device("cpu"), nn.GELU)
    assert isinstance(ratio, float)
    assert 0.0 <= ratio <= 1.0


def test_dead_ratio_no_hooks_returns_zero():
    model = nn.Linear(10, 5)
    loader = _make_loader(torch.randn(16, 10))
    ratio = compute_dead_ratio(model, loader, torch.device("cpu"), nn.ReLU)
    assert ratio == 0.0


def test_sample_activation_dist_returns_nonempty_array():
    from activation_exp.model import CNN
    model = CNN(in_channels=1, num_classes=10, activation="relu")
    x = torch.randn(8, 1, 28, 28)
    loader = _make_loader(x)
    dist = sample_activation_dist(model, loader, torch.device("cpu"), nn.ReLU)
    assert isinstance(dist, np.ndarray)
    assert len(dist) > 0
