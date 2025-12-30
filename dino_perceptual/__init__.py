"""DINO-based perceptual losses and feature extractors.

This package provides:
- DINOPerceptual: LPIPS-like perceptual loss using DINO features (v2 or v3)
- DINOModel: Feature extractor for FDD (Frechet DINO Distance)

Example:
    from dino_perceptual import DINOPerceptual, DINOModel

    # Perceptual loss for training (uses DINOv3 by default)
    loss_fn = DINOPerceptual(model_size='B', target_size=512)
    loss = loss_fn(pred_images, target_images).mean()

    # Use DINOv2 instead
    loss_fn_v2 = DINOPerceptual(model_size='B', version='v2')

    # Feature extraction for FDD
    extractor = DINOModel()
    features, _ = extractor(images)
"""

from dino_perceptual.perceptual import DINOPerceptual, DINOModel

__version__ = "0.1.2"
__all__ = ["DINOPerceptual", "DINOModel"]
