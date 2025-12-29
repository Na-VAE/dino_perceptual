# DINO Perceptual Loss

A drop-in replacement for LPIPS using DINOv2 features. Achieves better perceptual quality metrics (rFID, FDD) than VGG-based LPIPS while being simpler and more robust.

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
from dino_perceptual import DINOPerceptual

# Initialize loss function
loss_fn = DINOPerceptual(model_size="B").cuda().eval()

# Compute perceptual loss between two images
# Images should be tensors in [-1, 1] range with shape (B, 3, H, W)
loss = loss_fn(predicted, target).mean()
```

## Usage in Autoencoder Training

```python
import torch
import torch.nn as nn
from dino_perceptual import DINOPerceptual

# Initialize models
autoencoder = MyAutoencoder().cuda()
perceptual_loss = DINOPerceptual(model_size="B").cuda().eval()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

# Training loop
for images in dataloader:
    images = images.cuda()

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

| Metric | Pixel-only | + LPIPS | + DINO |
|--------|------------|---------|--------|
| rFID | 2.13 | 0.72 | **0.30** |
| rFDD | 6.96 | 2.93 | **1.12** |
| PSNR | **34.31** | 34.19 | 33.64 |

DINO perceptual loss achieves **7x better rFID** and **6x better rFDD** compared to pixel-only training, and **2x better** than LPIPS, with only ~0.7 dB PSNR trade-off.

## API Reference

### DINOPerceptual

```python
DINOPerceptual(
    model_size: str = "B",      # "S", "B", "L", or "G"
    target_size: int = 512,     # Resize images to this size
    layers: str = "all",        # Which transformer layers to use
)
```

**Arguments:**
- `model_size`: DINOv2 model variant. "B" (base) is recommended for most use cases.
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

extractor = DINOModel(model_size="B").cuda().eval()
features, cls_token = extractor(images)  # features: (B, num_patches, dim)
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
