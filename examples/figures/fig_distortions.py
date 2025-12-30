"""
Generate distortion analysis figure showing DINO loss behavior.

Shows that DINO loss increases monotonically with:
- Gaussian blur (increasing sigma)
- Gaussian noise (increasing sigma)
- JPEG compression (decreasing quality)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Modal GPU runs (DINOv2-B)
blur_data = [
    (0, 0.000000),
    (1, 0.000216),
    (2, 0.000357),
    (4, 0.000611),
    (6, 0.000828),
    (8, 0.000981),
]

noise_data = [
    (0, 0.000000),
    (10, 0.000335),
    (25, 0.000597),
    (50, 0.000928),
    (75, 0.001251),
    (100, 0.001464),
]

jpeg_data = [
    (100, 0.000183),
    (80, 0.000202),
    (50, 0.000262),
    (25, 0.000308),
    (10, 0.000416),
    (5, 0.000490),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Blur
ax = axes[0]
x, y = zip(*blur_data)
ax.plot(x, np.array(y) * 1000, marker='o', markersize=8, linewidth=2.5, color='#3b82f6')
ax.fill_between(x, 0, np.array(y) * 1000, alpha=0.15, color='#3b82f6')
ax.set_xlabel('Blur σ (pixels)')
ax.set_ylabel('DINO Loss (×10⁻³)')
ax.set_title('Gaussian Blur', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(0, 1.1)

# Noise
ax = axes[1]
x, y = zip(*noise_data)
ax.plot(x, np.array(y) * 1000, marker='s', markersize=8, linewidth=2.5, color='#f59e0b')
ax.fill_between(x, 0, np.array(y) * 1000, alpha=0.15, color='#f59e0b')
ax.set_xlabel('Noise σ (pixel intensity)')
ax.set_ylabel('DINO Loss (×10⁻³)')
ax.set_title('Gaussian Noise', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 105)
ax.set_ylim(0, 1.6)

# JPEG
ax = axes[2]
x, y = zip(*jpeg_data)
ax.plot(x, np.array(y) * 1000, marker='^', markersize=8, linewidth=2.5, color='#10b981')
ax.fill_between(x, 0, np.array(y) * 1000, alpha=0.15, color='#10b981')
ax.set_xlabel('JPEG Quality')
ax.set_ylabel('DINO Loss (×10⁻³)')
ax.set_title('JPEG Compression', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Lower quality = more degradation
ax.set_xlim(105, 0)
ax.set_ylim(0, 0.55)

plt.tight_layout()
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_distortions.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_distortions.svg",
            bbox_inches="tight", facecolor="white")
print("Saved: docs/fig_distortions.png and docs/fig_distortions.svg")
