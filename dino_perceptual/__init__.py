"""DINO-based perceptual losses and feature extractors.

This package provides:
- DINOv3Perceptual: LPIPS-like perceptual loss using DINOv3 features
- DINOv3Model: Feature extractor for FDD (Frechet DINO Distance)

Example:
    from dino_perceptual import DINOv3Perceptual, DINOv3Model

    # Perceptual loss for training
    loss_fn = DINOv3Perceptual(model_size='B', target_size=512)
    loss = loss_fn(pred_images, target_images).mean()

    # Feature extraction for FDD
    extractor = DINOv3Model()
    features, _ = extractor(images)
"""

from dino_perceptual.perceptual import DINOv3Perceptual
from dino_perceptual.dinov3 import DINOv3Model

__version__ = "0.1.0"
__all__ = ["DINOv3Perceptual", "DINOv3Model"]
