"""
Dataset utilities for Drifting training.
Handles dataset existence checks, optimized ADM cropping, and transform construction.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision import datasets, transforms


def _mnist_exists(root: Path) -> bool:
    mnist_root = root / "mnist" / "MNIST" 
    return (mnist_root / "t10k-images-idx3-ubyte").exists() and (mnist_root / "t10k-labels-idx1-ubyte").exists()


def _cifar10_exists(root: Path) -> bool:
    cifar_root = root / "cifar-10-batches-py"
    return (cifar_root / "data_batch_1").exists() and (cifar_root / "test_batch").exists()


def _imagenet_exists(root: Path) -> bool:
    """Checks if the standard ImageNet train/val structure exists."""
    return (root / "train").exists() and (root / "val").exists()


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Optimized center cropping implementation from OpenAI's ADM.
    Prevents standard torchvision resize aliasing artifacts which generative models
    otherwise memorize.
    """
    # 1. Box downsampling for massive images
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 2. Bicubic resize so the shortest edge matches the target size
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 3. Exact center crop to make it a perfect square
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    cropped_arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    return Image.fromarray(cropped_arr)


def get_dataset(name: str, root: str = "data", resize: int = 32) -> Tuple[object, object]:
    """Get dataset and transforms while reusing already-downloaded data."""
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    if name.lower() == "mnist":
        mnist_root = root_path / "mnist"
        download = not _mnist_exists(mnist_root)
        print(f"MNIST dataset {'downloading' if download else 'found existing'} at {mnist_root}")

        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = datasets.MNIST(str(mnist_root), train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(str(mnist_root), train=False, download=download, transform=transform)

    elif name.lower() in ["cifar10", "cifar"]:
        download = not _cifar10_exists(root_path)
        print(f"CIFAR-10 dataset {'downloading' if download else 'found existing'} at {root_path}")

        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = datasets.CIFAR10(str(root_path), train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(str(root_path), train=False, download=download, transform=test_transform)

    elif name.lower() == "imagenet":
        # Resolve path
        if _imagenet_exists(root_path):
            imagenet_root = root_path
        elif _imagenet_exists(root_path / "imagenet"):
            imagenet_root = root_path / "imagenet"
        else:
            raise RuntimeError(
                f"ImageNet dataset not found at {root_path}. "
                "Ensure the directory contains 'train' and 'val' subfolders."
            )
        print(f"ImageNet dataset found at {imagenet_root}")

        # Train Pipeline: ADM Crop -> Flip -> Tensor -> Normalize
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Test/FID Pipeline: ADM Crop -> Tensor -> Normalize (No flipping!)
        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.ImageFolder(str(imagenet_root / "train"), transform=train_transform)
        test_dataset = datasets.ImageFolder(str(imagenet_root / "val"), transform=test_transform)

    elif name.lower() == "imagenet-tiny":
        # Resolve path
        if _imagenet_exists(root_path):
            imagenet_root = root_path
        elif _imagenet_exists(root_path / "imagenet-tiny"):
            imagenet_root = root_path / "imagenet-tiny"
        else:
            raise RuntimeError(
                f"ImageNet dataset not found at {root_path}. "
                "Ensure the directory contains 'train' and 'val' subfolders."
            )
        print(f"ImageNet dataset found at {imagenet_root}")

        # Train Pipeline: ADM Crop -> Flip -> Tensor -> Normalize
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Test/FID Pipeline: ADM Crop -> Tensor -> Normalize (No flipping!)
        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.ImageFolder(str(imagenet_root / "train"), transform=train_transform)
        test_dataset = datasets.ImageFolder(str(imagenet_root / "val"), transform=test_transform)


    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset