"""
Pareto scatter plots showing the perception-distortion trade-off.

Compares loss configurations and baseline VAEs.
"""

import matplotlib.pyplot as plt
import numpy as np

# Loss ablation data + comparison models
# Format: (name, rFID, rFDD, PSNR, SSIM, group)
# Groups: "pixel" (baseline), "dino" (ours), "other" (external VAEs)
data = [
    # Loss ablation: Charb + X
    ("Pixel only", 5.13, 10.96, 34.81, 0.929, "pixel"),
    ("+ SSIM", 4.87, 10.42, 34.72, 0.931, "pixel"),
    ("+ DINO (α=250)", 0.51, 1.45, 33.93, 0.919, "dino"),
    ("+ DINO (α=1000)", 0.30, 1.12, 33.64, 0.914, "dino"),
    # Final model: Charb + SSIM + DINO
    ("+ SSIM + DINO (α=250)", 0.64, 1.70, 34.05, 0.921, "dino"),
    # Other VAEs for reference
    ("Qwen VAE", 1.32, 7.36, 30.27, 0.860, "other"),
    ("FLUX.1", 0.15, 2.29, 31.10, 0.887, "other"),
]

names = [d[0] for d in data]
rFID = np.array([d[1] for d in data])
rFDD = np.array([d[2] if d[2] is not None else np.nan for d in data])
PSNR = np.array([d[3] for d in data])
SSIM = np.array([d[4] for d in data])
groups = [d[5] for d in data]


def is_pareto_optimal(costs):
    """Find Pareto-optimal points (minimizing all objectives)."""
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_optimal[i] = not np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
    return is_optimal


# Color scheme: pixel baselines gray, DINO green, others blue/purple
colors = {
    "Pixel only": "#9ca3af",
    "+ SSIM": "#6b7280",
    "+ DINO (α=250)": "#34d399",
    "+ DINO (α=1000)": "#059669",
    "+ SSIM + DINO (α=250)": "#dc2626",
    "Qwen VAE": "#8b5cf6",
    "FLUX.1": "#3b82f6",
}

# Markers by group
markers = {
    "pixel": "o",
    "dino": "D",  # Diamond for DINO methods
    "other": "s",  # Square for other VAEs
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_configs = [
    (axes[0, 0], rFID, SSIM, "rFID ↓ (lower is better)", "SSIM ↑", "rFID vs SSIM"),
    (axes[0, 1], rFID, PSNR, "rFID ↓ (lower is better)", "PSNR ↑ (dB)", "rFID vs PSNR"),
    (axes[1, 0], rFDD, SSIM, "rFDD ↓ (lower is better)", "SSIM ↑", "rFDD vs SSIM"),
    (axes[1, 1], rFDD, PSNR, "rFDD ↓ (lower is better)", "PSNR ↑ (dB)", "rFDD vs PSNR"),
]

# Note: Layout is now:
# [rFID vs SSIM]  [rFID vs PSNR]
# [rFDD vs SSIM]  [rFDD vs PSNR]
# Left column = SSIM, Right column = PSNR

for ax, x_data, y_data, x_label, y_label, title in plot_configs:
    # Filter out NaN values
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    names_valid = [n for n, v in zip(names, valid_mask) if v]
    groups_valid = [g for g, v in zip(groups, valid_mask) if v]

    # Pareto frontier (minimize x, maximize y)
    costs = np.column_stack([x_valid, -y_valid])
    pareto_mask = is_pareto_optimal(costs)

    # Draw Pareto frontier line
    pareto_x = x_valid[pareto_mask]
    pareto_y = y_valid[pareto_mask]
    sort_idx = np.argsort(pareto_x)
    ax.plot(pareto_x[sort_idx], pareto_y[sort_idx],
            color="#10b981", alpha=0.3, linewidth=8, zorder=1,
            solid_capstyle='round')

    # Plot points
    for i, (name, group) in enumerate(zip(names_valid, groups_valid)):
        is_pareto = pareto_mask[i]
        marker = markers[group]
        size = 220 if is_pareto else 160
        edge = "#000" if is_pareto else "white"
        edge_w = 2.5 if is_pareto else 1.5

        ax.scatter(x_valid[i], y_valid[i],
                   c=colors[name],
                   s=size,
                   marker=marker,
                   edgecolors=edge,
                   linewidths=edge_w,
                   zorder=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.invert_xaxis()

# Build legend with clear groupings
from matplotlib.lines import Line2D

legend_elements = []

# Section: Marker shapes
legend_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#888888', markersize=12,
                              markeredgecolor='#888888', markeredgewidth=0,
                              label='● Pixel losses'))
legend_elements.append(Line2D([0], [0], marker='D', color='w',
                              markerfacecolor='#888888', markersize=11,
                              markeredgecolor='#888888', markeredgewidth=0,
                              label='◆ + DINO loss'))
legend_elements.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='#888888', markersize=11,
                              markeredgecolor='#888888', markeredgewidth=0,
                              label='■ Other VAEs'))

# Separator
legend_elements.append(Line2D([0], [0], marker='', color='w', label=''))

# Section: Pixel baselines
legend_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#9ca3af', markersize=12,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='Pixel only'))
legend_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#6b7280', markersize=12,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='+ SSIM'))

# Section: DINO methods
legend_elements.append(Line2D([0], [0], marker='D', color='w',
                              markerfacecolor='#34d399', markersize=11,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='+ DINO (α=250)'))
legend_elements.append(Line2D([0], [0], marker='D', color='w',
                              markerfacecolor='#059669', markersize=11,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='+ DINO (α=1000)'))
legend_elements.append(Line2D([0], [0], marker='D', color='w',
                              markerfacecolor='#dc2626', markersize=11,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='+ SSIM + DINO (α=250)'))

# Section: Other VAEs
legend_elements.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='#8b5cf6', markersize=11,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='Qwen VAE'))
legend_elements.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='#3b82f6', markersize=11,
                              markeredgecolor='white', markeredgewidth=1.5,
                              label='FLUX.1'))

# Separator
legend_elements.append(Line2D([0], [0], marker='', color='w', label=''))

# Pareto indicator
legend_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='#cccccc', markersize=12,
                              markeredgecolor='black', markeredgewidth=2.5,
                              label='Pareto optimal'))

fig.legend(handles=legend_elements, loc="lower center", ncol=7,
           frameon=True, fontsize=13, bbox_to_anchor=(0.5, -0.02),
           columnspacing=1.2, handletextpad=0.3)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_pareto.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_pareto.svg",
            bbox_inches="tight", facecolor="white")
print("Saved: docs/fig_pareto.png and docs/fig_pareto.svg")
