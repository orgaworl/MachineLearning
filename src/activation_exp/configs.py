from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    name: str
    dataset: str        # "mnist" | "cifar10" | "cifar10_lite"
    activation: str     # "relu" | "leaky_relu" | "swish" | "gelu"
    in_channels: int
    num_classes: int
    epochs: int
    batch_size: int
    lr: float
    input_size: Tuple[int, int]
    model_type: str = field(default="cnn")  # "cnn" | "deep_cnn"


def _make_experiments() -> List[ExperimentConfig]:
    configs = []
    for act in ["relu", "leaky_relu", "swish", "gelu"]:
        configs.append(ExperimentConfig(
            name=f"mnist_{act}", dataset="mnist", activation=act,
            in_channels=1, num_classes=10, epochs=20,
            batch_size=128, lr=1e-3, input_size=(28, 28),
        ))
        configs.append(ExperimentConfig(
            name=f"cifar10_{act}", dataset="cifar10", activation=act,
            in_channels=3, num_classes=10, epochs=30,
            batch_size=128, lr=1e-3, input_size=(32, 32),
        ))
        configs.append(ExperimentConfig(
            name=f"cifar10_lite_{act}", dataset="cifar10_lite", activation=act,
            in_channels=3, num_classes=10, epochs=15,
            batch_size=256, lr=1e-3, input_size=(32, 32),
        ))
        # 深层网络（无BN），用于放大激活函数差异
        configs.append(ExperimentConfig(
            name=f"cifar10_deep_{act}", dataset="cifar10", activation=act,
            in_channels=3, num_classes=10, epochs=30,
            batch_size=128, lr=1e-3, input_size=(32, 32),
            model_type="deep_cnn",
        ))
    return configs


EXPERIMENTS: List[ExperimentConfig] = _make_experiments()
