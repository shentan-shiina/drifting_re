"""
Drifting Models Implementation

A PyTorch reproduction of "Generative Modeling via Drifting" (Deng et al., 2026)
for MNIST and CIFAR-10 datasets.
"""

from .drift_dit import DriftDiT, DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
from .drifting import (
    compute_V,
    compute_V_multi_temperature,
    normalize_features,
    normalize_drift,
    DriftingLoss,
    ClassConditionalDriftingLoss,
    drift_step_2d,
)
from .feature_encoder import (
    MultiScaleFeatureEncoder,
    MAEEncoder,
    create_feature_encoder,
    pretrain_mae,
)

__version__ = "0.1.0"
__all__ = [
    # Model
    "DriftDiT",
    "DriftDiT_Tiny",
    "DriftDiT_Small",
    "DriftDiT_models",
    # Drifting
    "compute_V",
    "compute_V_multi_temperature",
    "normalize_features",
    "normalize_drift",
    "DriftingLoss",
    "ClassConditionalDriftingLoss",
    "drift_step_2d",
    # Feature encoder
    "MultiScaleFeatureEncoder",
    "MAEEncoder",
    "create_feature_encoder",
    "pretrain_mae",
]
