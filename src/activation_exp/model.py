import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
}


class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, activation: str):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), act(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), act(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), act(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), act(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512), act(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class DeepCNN(nn.Module):
    """8层卷积，无BatchNorm，使激活函数差异（死亡神经元、梯度流）更显著。"""
    def __init__(self, in_channels: int, num_classes: int, activation: str):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), act(),
            nn.Conv2d(64, 64, 3, padding=1), act(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), act(),
            nn.Conv2d(128, 128, 3, padding=1), act(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), act(),
            nn.Conv2d(256, 256, 3, padding=1), act(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), act(),
            nn.Conv2d(512, 512, 3, padding=1), act(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512), act(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

