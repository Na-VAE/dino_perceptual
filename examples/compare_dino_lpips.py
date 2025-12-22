#!/usr/bin/env python3
"""Compare DINO perceptual loss vs LPIPS across a distribution of images.

This script visualizes the key differences between DINO and LPIPS losses:
- Scatter plot showing correlation (and disagreements)
- Gallery of images where the losses disagree most
- Sensitivity curves for different distortion types

Usage:
    python compare_dino_lpips.py                          # Use sample images
    python compare_dino_lpips.py --images /path/to/imgs   # Your images
    python compare_dino_lpips.py --n 100 --output out.png # Customize
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from dino_perceptual import DINOv3Perceptual

# Try to import LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Note: Install lpips for comparison (pip install lpips)")


def load_images(path, n=100, size=256):
    """Load images from a folder."""
    path = Path(path)
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    files = [f for f in path.iterdir() if f.suffix.lower() in extensions][:n]

    images = []
    for f in tqdm(files, desc="Loading images"):
        img = Image.open(f).convert('RGB').resize((size, size), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1
        images.append(tensor)

    return torch.stack(images)


def download_sample_images(n=100):
    """Download sample images from CIFAR or use synthetic."""
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        dataset = CIFAR10(root='/tmp/cifar', train=False, download=True, transform=transform)
        images = torch.stack([dataset[i][0] for i in range(min(n, len(dataset)))])
        return images
    except Exception:
        # Fallback: synthetic images
        print("Using synthetic test images")
        return torch.randn(n, 3, 256, 256).clamp(-1, 1)


# === Distortion Functions ===

def blur(img, sigma):
    """Gaussian blur."""
    if sigma == 0:
        return img
    k = int(sigma * 4) | 1  # Ensure odd
    x = torch.arange(k).float() - k // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 1, -1) * kernel.view(1, 1, -1, 1)
    kernel = kernel.expand(3, 1, -1, -1).to(img.device)
    return F.conv2d(img.unsqueeze(0), kernel, padding=k//2, groups=3).squeeze(0).clamp(-1, 1)


def noise(img, sigma):
    """Gaussian noise."""
    return (img + torch.randn_like(img) * sigma).clamp(-1, 1)


def color_shift(img, strength):
    """Shift colors."""
    shifted = img.clone()
    shifted[0] = (shifted[0] + strength).clamp(-1, 1)
    shifted[2] = (shifted[2] - strength).clamp(-1, 1)
    return shifted


def jpeg_artifact(img, quality):
    """Simulate JPEG artifacts via block averaging."""
    block = max(2, int(32 / max(1, quality)))
    B, C, H, W = 1, *img.shape
    result = img.clone().unsqueeze(0)
    for i in range(0, H - block + 1, block):
        for j in range(0, W - block + 1, block):
            result[:, :, i:i+block, j:j+block] = result[:, :, i:i+block, j:j+block].mean(dim=(2,3), keepdim=True)
    return result.squeeze(0)


def spatial_shift(img, pixels):
    """Shift image spatially."""
    return torch.roll(img, shifts=int(pixels), dims=2)


def patch_shuffle(img, n_swaps):
    """Shuffle patches to break semantic structure."""
    result = img.clone()
    H, W = img.shape[1], img.shape[2]
    patch = 32
    for _ in range(int(n_swaps)):
        i1, j1 = np.random.randint(0, H-patch), np.random.randint(0, W-patch)
        i2, j2 = np.random.randint(0, H-patch), np.random.randint(0, W-patch)
        tmp = result[:, i1:i1+patch, j1:j1+patch].clone()
        result[:, i1:i1+patch, j1:j1+patch] = result[:, i2:i2+patch, j2:j2+patch]
        result[:, i2:i2+patch, j2:j2+patch] = tmp
    return result


DISTORTIONS = {
    'blur': (blur, np.linspace(0, 8, 9)),
    'noise': (noise, np.linspace(0, 0.5, 9)),
    'color': (color_shift, np.linspace(0, 0.8, 9)),
    'jpeg': (jpeg_artifact, np.linspace(10, 1, 9)),
    'shift': (spatial_shift, np.linspace(0, 32, 9)),
    'shuffle': (patch_shuffle, np.linspace(0, 20, 9)),
}


def compute_all_scores(images, dino_fn, lpips_fn, device):
    """Compute DINO and LPIPS scores for all images with all distortions."""
    results = []

    for idx in tqdm(range(len(images)), desc="Computing scores"):
        img = images[idx].to(device)

        for dist_name, (dist_fn, strengths) in DISTORTIONS.items():
            for strength in strengths[1:]:  # Skip 0
                distorted = dist_fn(img, strength)

                with torch.no_grad():
                    dino_score = dino_fn(distorted.unsqueeze(0), img.unsqueeze(0)).item()
                    lpips_score = lpips_fn(distorted.unsqueeze(0), img.unsqueeze(0)).item() if lpips_fn else 0

                results.append({
                    'img_idx': idx,
                    'distortion': dist_name,
                    'strength': strength,
                    'dino': dino_score,
                    'lpips': lpips_score,
                    'original': img.cpu(),
                    'distorted': distorted.cpu(),
                })

    return results


def plot_scatter(results, ax):
    """Plot DINO vs LPIPS scatter."""
    colors = {'blur': 'C0', 'noise': 'C1', 'color': 'C2', 'jpeg': 'C3', 'shift': 'C4', 'shuffle': 'C5'}

    for dist_name in DISTORTIONS.keys():
        subset = [r for r in results if r['distortion'] == dist_name]
        dino = [r['dino'] for r in subset]
        lpips = [r['lpips'] for r in subset]
        ax.scatter(lpips, dino, alpha=0.5, label=dist_name, c=colors[dist_name], s=20)

    # Correlation
    all_dino = [r['dino'] for r in results]
    all_lpips = [r['lpips'] for r in results]
    r, p = stats.pearsonr(all_lpips, all_dino)

    ax.set_xlabel('LPIPS Score', fontsize=12)
    ax.set_ylabel('DINO Score', fontsize=12)
    ax.set_title(f'DINO vs LPIPS (r={r:.2f})', fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_disagreement_gallery(results, axes_row):
    """Show images where losses disagree most."""
    # Normalize scores
    dino_scores = np.array([r['dino'] for r in results])
    lpips_scores = np.array([r['lpips'] for r in results])

    dino_norm = (dino_scores - dino_scores.min()) / (dino_scores.max() - dino_scores.min() + 1e-8)
    lpips_norm = (lpips_scores - lpips_scores.min()) / (lpips_scores.max() - lpips_scores.min() + 1e-8)

    # Find disagreements
    diff = dino_norm - lpips_norm  # Positive = DINO higher, Negative = LPIPS higher

    dino_high_idx = np.argsort(diff)[-3:][::-1]  # DINO >> LPIPS
    lpips_high_idx = np.argsort(diff)[:3]         # LPIPS >> DINO

    def to_numpy(t):
        return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

    # Plot DINO catches, LPIPS misses
    for i, idx in enumerate(dino_high_idx):
        r = results[idx]
        combined = np.concatenate([to_numpy(r['original']), to_numpy(r['distorted'])], axis=1)
        axes_row[i].imshow(combined)
        axes_row[i].set_title(f"DINO={r['dino']:.3f}\nLPIPS={r['lpips']:.3f}\n({r['distortion']})", fontsize=8)
        axes_row[i].axis('off')

    # Plot LPIPS catches, DINO misses
    for i, idx in enumerate(lpips_high_idx):
        r = results[idx]
        combined = np.concatenate([to_numpy(r['original']), to_numpy(r['distorted'])], axis=1)
        axes_row[i + 3].imshow(combined)
        axes_row[i + 3].set_title(f"DINO={r['dino']:.3f}\nLPIPS={r['lpips']:.3f}\n({r['distortion']})", fontsize=8)
        axes_row[i + 3].axis('off')


def plot_sensitivity_curves(results, axes):
    """Plot how each loss responds to increasing distortion."""
    for ax_idx, (dist_name, (_, strengths)) in enumerate(DISTORTIONS.items()):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]

        subset = [r for r in results if r['distortion'] == dist_name]

        # Group by strength
        strength_to_dino = {}
        strength_to_lpips = {}
        for r in subset:
            s = r['strength']
            strength_to_dino.setdefault(s, []).append(r['dino'])
            strength_to_lpips.setdefault(s, []).append(r['lpips'])

        x = sorted(strength_to_dino.keys())
        dino_means = [np.mean(strength_to_dino[s]) for s in x]
        lpips_means = [np.mean(strength_to_lpips[s]) for s in x]

        ax.plot(x, dino_means, 'o-', label='DINO', color='C0')
        ax.plot(x, lpips_means, 's-', label='LPIPS', color='C1')
        ax.set_title(dist_name, fontsize=10)
        ax.set_xlabel('Strength', fontsize=8)
        ax.set_ylabel('Loss', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description='Compare DINO vs LPIPS')
    parser.add_argument('--images', type=str, default=None, help='Image folder')
    parser.add_argument('--n', type=int, default=50, help='Number of images')
    parser.add_argument('--output', type=str, default='dino_vs_lpips.png', help='Output file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load images
    if args.images:
        images = load_images(args.images, n=args.n)
    else:
        images = download_sample_images(n=args.n)
    print(f"Loaded {len(images)} images")

    # Load models
    print("Loading DINO...")
    dino_fn = DINOv3Perceptual(model_size='B').to(device).eval()

    lpips_fn = None
    if HAS_LPIPS:
        print("Loading LPIPS...")
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    # Compute scores
    results = compute_all_scores(images, dino_fn, lpips_fn, device)
    print(f"Computed {len(results)} distortion pairs")

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # Row 1: Scatter plot
    ax_scatter = fig.add_subplot(3, 1, 1)
    plot_scatter(results, ax_scatter)

    # Row 2: Disagreement gallery
    axes_gallery = [fig.add_subplot(3, 6, 7 + i) for i in range(6)]
    fig.text(0.25, 0.62, 'DINO catches, LPIPS misses', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.75, 0.62, 'LPIPS catches, DINO misses', ha='center', fontsize=10, fontweight='bold')
    plot_disagreement_gallery(results, axes_gallery)

    # Row 3: Sensitivity curves
    axes_curves = [fig.add_subplot(3, 6, 13 + i) for i in range(6)]
    plot_sensitivity_curves(results, axes_curves)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
DINO perceptual loss uses self-supervised vision features that capture
semantic/structural information (objects, layout, meaning).

LPIPS uses supervised classification features that focus on
texture/perceptual details (edges, patterns, colors).

Look at the disagreement gallery:
- Left side: Distortions DINO catches but LPIPS misses (semantic damage)
- Right side: Distortions LPIPS catches but DINO misses (texture damage)
""")


if __name__ == '__main__':
    main()
