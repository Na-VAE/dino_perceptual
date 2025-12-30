"""
Generate FDD interpretation figure.

Shows FDD scores for:
- Same distribution (should be ~0)
- Good reconstruction (low noise)
- Poor reconstruction (high noise)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Modal GPU runs
fdd_data = {
    'Same\ndistribution': -0.00,  # ~0
    'Good recon\n(σ=0.1 noise)': 5.57,
    'Poor recon\n(σ=0.5 noise)': 16.03,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(7, 4))

labels = list(fdd_data.keys())
values = list(fdd_data.values())
colors = ['#10b981', '#f59e0b', '#ef4444']  # green, yellow, red

bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='white', linewidth=2)

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', ha='left', fontweight='bold', fontsize=11)

# Add reference zones
ax.axvline(x=1, color='#10b981', linestyle='--', alpha=0.7, linewidth=1.5, label='Excellent (<1)')
ax.axvline(x=5, color='#f59e0b', linestyle='--', alpha=0.7, linewidth=1.5, label='Good (1-5)')

ax.set_xlabel('Frechet DINO Distance (FDD)')
ax.set_title('FDD Score Interpretation', fontweight='bold')
ax.set_xlim(0, 20)
ax.grid(True, axis='x', alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Invert y-axis so "Same distribution" is at top
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_fdd.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/Users/philippe/dino_perceptual/docs/fig_fdd.svg",
            bbox_inches="tight", facecolor="white")
print("Saved: docs/fig_fdd.png and docs/fig_fdd.svg")
