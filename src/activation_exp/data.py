from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .configs import ExperimentConfig


def get_dataloaders(config: ExperimentConfig):
    if config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("data/", train=True, download=True, transform=transform)
        val_ds = datasets.MNIST("data/", train=False, download=True, transform=transform)
    else:  # cifar10 or cifar10_lite — same underlying dataset
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10("data/", train=True, download=True, transform=train_transform)
        val_ds = datasets.CIFAR10("data/", train=False, download=True, transform=val_transform)

    # pin_memory=False: MPS does not support pinned memory
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=False,
    )
    return train_loader, val_loader
