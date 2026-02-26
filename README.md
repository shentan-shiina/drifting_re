
# Drifting Models

Simple repo for training the drifting model on MNIST and CIFAR-10.

## What is here

- Training loop with drifting loss in [drifting/scripts/train.py](drifting/scripts/train.py).
- Core drifting loss and utilities in [drifting/models](drifting/models) and [drifting/utils](drifting/utils).
- Pre-downloaded MNIST and CIFAR-10 examples under [drifting/data](drifting/data) (the scripts will also download if missing).

## Setup

1. Create an environment with PyTorch, torchvision, and einops installed.
2. Install the repo in editable mode:

``` python
pip install -e .
```

## Quick start

Train on MNIST (32x32, pixel-space features):

``` python
python drifting/scripts/train.py --dataset mnist 
```

Train on CIFAR-10 (32x32, pretrained feature encoder):

``` python
python drifting/scripts/train.py --dataset cifar10
```

## Key flags

We use [Hydra](https://hydra.cc/) for configuration. Key flags include:

- `dataset`: mnist or cifar10
- `data_root`: where datasets are stored/downloaded
- `epochs`, `batch_size`, `lr`: basic training knobs
- `use_feature_encoder`: toggle pretrained encoder; off by default for MNIST

## Outputs

Checkpoints and image grids are saved under [drifting/outputs](drifting/outputs) by default.
