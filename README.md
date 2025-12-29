# DINO Perceptual Loss

A drop-in replacement for LPIPS using DINOv3 features. Achieves better perceptual quality metrics (rFID, FDD) than VGG-based LPIPS while being simpler and more robust.

**Why DINO over VGG?** DINOv3 is trained with self-supervised learning on 1.7B images using modern Vision Transformer architectures. This produces richer, more semantically meaningful features compared to VGG-16's classification-focused features from 2014. The result: **2x better perceptual metrics** with the same training setup.

## Installation

```bash
pip install dino-perceptual
```

Or from source:

```bash
git clone https://github.com/Na-VAE/dino-perceptual.git
cd dino-perceptual
pip install -e .
```

## Quick Start

```python
import torch
from dino_perceptual import DINOPerceptual

# Initialize loss function (uses DINOv3 by default)
loss_fn = DINOPerceptual(model_size="B").cuda().bfloat16().eval()
loss_fn = torch.compile(loss_fn, fullgraph=True)

# Compute perceptual loss between two images
# Images should be tensors in [-1, 1] range with shape (B, 3, H, W)
loss = loss_fn(predicted, target).mean()

# Use DINOv2 instead (legacy)
loss_fn_v2 = DINOPerceptual(model_size="B", version="v2").cuda().eval()
```

## Usage in Autoencoder Training

```python
import torch
import torch.nn as nn
from dino_perceptual import DINOPerceptual

# Initialize models
autoencoder = MyAutoencoder().cuda().bfloat16()
perceptual_loss = DINOPerceptual(model_size="B").cuda().bfloat16().eval()
perceptual_loss = torch.compile(perceptual_loss, fullgraph=True)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

# Training loop
for images in dataloader:
    images = images.cuda().bfloat16()

    # Forward pass
    reconstructed = autoencoder(images)

    # Compute losses
    l1_loss = nn.functional.l1_loss(reconstructed, images)
    dino_loss = perceptual_loss(reconstructed, images).mean()

    # Combined loss (DINO weight ~250-1000 works well)
    total_loss = l1_loss + 250.0 * dino_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Why DINO over LPIPS?

Results from training a 4.5B parameter ViTok autoencoder (Td4-T/16×64, 16× spatial compression, 64 latent channels) on ImageNet:

| Metric | Pixel-only | + LPIPS (λ=0.1) | + DINO (λ=1000) |
|--------|------------|-----------------|-----------------|
| rFID ↓ | 2.13 | 0.72 | **0.30** |
| rFDD ↓ | 6.96 | 2.93 | **1.12** |
| PSNR ↑ | **34.31** | 34.19 | 33.64 |
| SSIM ↑ | **0.925** | 0.923 | 0.914 |

DINO perceptual loss achieves **7× better rFID** and **6× better rFDD** compared to pixel-only training, and **2× better** than LPIPS, with only ~0.7 dB PSNR trade-off (invisible to human perception).

## API Reference

### DINOPerceptual

```python
DINOPerceptual(
    model_size: str = "B",      # "S", "B", "L", "H", or "G"
    version: str = "v3",        # "v2" or "v3" (default: v3)
    target_size: int = 512,     # Resize images to this size
    layers: str = "all",        # Which transformer layers to use
)
```

**Arguments:**
- `model_size`: DINO model variant. "B" (base) is recommended for most use cases.
- `version`: DINO version. "v3" (default) uses the latest models trained on 1.7B images. "v2" for legacy compatibility.
- `target_size`: Images are resized to this size before computing features.
- `layers`: Which transformer layers to extract features from. "all" uses all layers.

**Input format:**
- Tensors in `[-1, 1]` range with shape `(B, 3, H, W)`

**Returns:**
- Per-sample loss tensor of shape `(B,)`

### DINOModel

For feature extraction (e.g., computing FDD):

```python
from dino_perceptual import DINOModel

extractor = DINOModel(model_size="B").cuda().bfloat16().eval()
features, cls_token = extractor(images)  # features: (B, feature_dim)
```

## License

MIT License

## Citation

If you find this code helpful, please cite:

```bibtex
@software{dino_perceptual,
  title={DINO Perceptual Loss},
  author={Hansen-Estruch, Philippe and Chen, Jiahui and Ramanujan, Vivek and Zohar, Orr and Ping, Yan and Sinha, Animesh and Georgopoulos, Markos and Schoenfeld, Edgar and Hou, Ji and Juefei-Xu, Felix and Vishwanath, Sriram and Thabet, Ali},
  year={2025},
  url={https://github.com/Na-VAE/dino-perceptual}
}
```

## Acknowledgments

This work builds on [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research and findings from the ViTok project on scaling visual tokenizers.
