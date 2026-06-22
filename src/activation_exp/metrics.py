from typing import Type

import numpy as np
import torch
import torch.nn as nn


def compute_dead_ratio(
    model: nn.Module,
    dataloader,
    device: torch.device,
    activation_cls: Type[nn.Module],
) -> float:
    """Fraction of activation-layer neurons that never fire (output > 0) across the full dataloader."""
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, activation_cls):
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not hooks:
        return 0.0

    ever_positive: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            model(x.to(device))
            for name, act in activations.items():
                fired = (act > 0).any(dim=0)
                if name not in ever_positive:
                    ever_positive[name] = fired
                else:
                    ever_positive[name] |= fired

    for h in hooks:
        h.remove()

    total = sum(t.numel() for t in ever_positive.values())
    dead = sum((~t).sum().item() for t in ever_positive.values())
    return dead / total if total > 0 else 0.0


def sample_activation_dist(
    model: nn.Module,
    dataloader,
    device: torch.device,
    activation_cls: Type[nn.Module],
    target_layer_idx: int = 1,
    n_batches: int = 3,
) -> np.ndarray:
    """Collect flattened activation values from the target_layer_idx-th activation layer (0-indexed)."""
    captured = [None]
    hook_handle = [None]
    count = [0]

    def hook(module, input, output):
        captured[0] = output.detach().cpu().numpy()

    for module in model.modules():
        if isinstance(module, activation_cls):
            if count[0] == target_layer_idx:
                hook_handle[0] = module.register_forward_hook(hook)
                break
            count[0] += 1

    samples = []
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(x.to(device))
            if captured[0] is not None:
                samples.append(captured[0].flatten())

    if hook_handle[0] is not None:
        hook_handle[0].remove()

    return np.concatenate(samples) if samples else np.array([])
