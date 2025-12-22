# DINO Perceptual Examples

## Compare DINO vs LPIPS

Visualize the differences between DINO perceptual loss and LPIPS:

```bash
# Install dependencies
pip install lpips matplotlib scipy tqdm

# Run with sample images
python compare_dino_lpips.py

# Run with your own images
python compare_dino_lpips.py --images /path/to/images --n 100

# Customize output
python compare_dino_lpips.py --output my_analysis.png --device cuda
```

### What it shows:

1. **Scatter Plot**: Correlation between DINO and LPIPS scores across many image/distortion pairs

2. **Disagreement Gallery**: The most interesting part - images where the losses fundamentally disagree:
   - Left: DINO catches damage that LPIPS misses (semantic/structural)
   - Right: LPIPS catches damage that DINO misses (texture/color)

3. **Sensitivity Curves**: How each loss responds to increasing distortion strength

### Key Insight

| Loss | Trained on | Focuses on |
|------|------------|------------|
| **DINO** | Self-supervised vision | Semantic structure, objects, layout |
| **LPIPS** | ImageNet classification | Textures, edges, local patterns |

For VAE training, DINO perceptual loss helps learn meaningful structure, while LPIPS focuses on fine-grained texture matching.
