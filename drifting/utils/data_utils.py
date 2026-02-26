"""
Dataset utilities for Drifting training.
Handles dataset existence checks and transform construction to avoid
unnecessary downloads during repeated runs.
"""

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms


def _mnist_exists(root: Path) -> bool:
    processed = root / "MNIST" / "processed"
    return (processed / "training.pt").exists() and (processed / "test.pt").exists()


def _cifar10_exists(root: Path) -> bool:
    cifar_root = root / "cifar-10-batches-py"
    return (cifar_root / "data_batch_1").exists() and (cifar_root / "test_batch").exists()


def get_dataset(name: str, root: str = "data") -> Tuple[object, object]:
    """Get dataset and transforms while reusing already-downloaded data."""
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    if name.lower() == "mnist":
        mnist_root = root_path / "mnist"
        download = not _mnist_exists(mnist_root)
        status = "downloading" if download else "found existing"
        print(f"MNIST dataset {status} at {mnist_root}")

        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataset = datasets.MNIST(
            str(mnist_root), train=True, download=download, transform=transform
        )
        test_dataset = datasets.MNIST(
            str(mnist_root), train=False, download=download, transform=transform
        )
    elif name.lower() in ["cifar10", "cifar"]:
        download = not _cifar10_exists(root_path)
        status = "downloading" if download else "found existing"
        print(f"CIFAR-10 dataset {status} at {root_path}")

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        train_dataset = datasets.CIFAR10(
            str(root_path), train=True, download=download, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            str(root_path), train=False, download=download, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset
