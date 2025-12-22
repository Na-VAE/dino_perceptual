# dino-perceptual

DINO-based perceptual losses and feature extraction for image generation.

## Installation

```bash
pip install dino-perceptual
```

Or install from source:
```bash
pip install -e .
```

## Usage

### Perceptual Loss for Training

```python
from dino_perceptual import DINOv3Perceptual

# Create loss function (uses DINOv3 ViT-B/16 by default)
loss_fn = DINOv3Perceptual(model_size='B', target_size=512)
loss_fn = loss_fn.to('cuda').eval()

# Compute loss (images should be in [-1, 1] range)
loss = loss_fn(pred_images, target_images).mean()
```

Model sizes: `'S'` (ViT-S/16), `'B'` (ViT-B/16), `'L'` (ViT-L/16), `'H'` (ViT-H/14)

### Feature Extraction for FDD (Frechet DINO Distance)

```python
from dino_perceptual import DINOv3Model

# Create feature extractor
extractor = DINOv3Model()
extractor = extractor.to('cuda').eval()

# Extract features (images should be in [-1, 1] range)
features, _ = extractor(images)  # (B, feature_dim)
```

## API Reference

### DINOv3Perceptual

LPIPS-like perceptual loss using frozen DINOv3 features.

```python
DINOv3Perceptual(
    model_name=None,        # HuggingFace model name (overrides model_size)
    model_size='B',         # Model size: 'S', 'B', 'L', 'H'
    target_size=512,        # Max image size (larger images are downscaled)
    layers='all',           # Which layers to use ('all' or list of indices)
    normalize=True,         # L2-normalize features per token
    resize_to_square=True,  # Resize vs center crop
)
```

### DINOv3Model

Feature extractor for FDD calculation.

```python
DINOv3Model(
    model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    resize_to_square=False,
    target_size=512,
)
```

## License

MIT
