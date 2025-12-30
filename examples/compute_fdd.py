"""Example: Computing Frechet DINO Distance (FDD).

FDD works like FID but uses DINO CLS tokens instead of Inception features.
This makes it more semantically meaningful for evaluating image quality.

Usage:
    python compute_fdd.py

Requires: scipy, torch, dino_perceptual
"""

import numpy as np
import torch
from scipy import linalg


def compute_fdd(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute Frechet DINO Distance between two feature sets.

    Args:
        real_features: (N, D) array of features from real images
        fake_features: (M, D) array of features from generated/reconstructed images

    Returns:
        FDD score (lower is better)
            - < 1: Excellent reconstruction quality
            - 1-5: Good quality
            - > 5: Poor quality
    """
    # Compute statistics
    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    # Frechet distance
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fdd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fdd)


def main():
    from dino_perceptual import DINOModel

    print("Computing Frechet DINO Distance (FDD)")
    print("=" * 50)

    # Initialize feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    extractor = DINOModel(model_size="B").to(device).eval()

    # Create example image batches
    # In practice, you would load real images from your dataset
    print("\nGenerating example images...")

    # "Real" images: random natural-looking patterns
    torch.manual_seed(42)
    real_images = torch.randn(32, 3, 224, 224).to(device).clamp(-1, 1)

    # "Reconstructed" images: slightly perturbed versions
    reconstructed_images = real_images + torch.randn_like(real_images) * 0.1
    reconstructed_images = reconstructed_images.clamp(-1, 1)

    # "Poor" reconstructions: heavily perturbed
    poor_reconstructions = real_images + torch.randn_like(real_images) * 0.5
    poor_reconstructions = poor_reconstructions.clamp(-1, 1)

    # Extract features
    print("Extracting DINO features...")
    with torch.no_grad():
        real_feats, _ = extractor(real_images)
        recon_feats, _ = extractor(reconstructed_images)
        poor_feats, _ = extractor(poor_reconstructions)

    # Compute FDD
    print("\nComputing FDD scores:")
    print("-" * 50)

    fdd_same = compute_fdd(real_feats.cpu().numpy(), real_feats.cpu().numpy())
    print(f"FDD (same images):           {fdd_same:.4f}  (should be ~0)")

    fdd_good = compute_fdd(real_feats.cpu().numpy(), recon_feats.cpu().numpy())
    print(f"FDD (good reconstruction):   {fdd_good:.4f}")

    fdd_poor = compute_fdd(real_feats.cpu().numpy(), poor_feats.cpu().numpy())
    print(f"FDD (poor reconstruction):   {fdd_poor:.4f}")

    # Interpretation
    print("\n" + "=" * 50)
    print("Interpretation guide:")
    print("  FDD < 1:  Excellent quality")
    print("  FDD 1-5:  Good quality")
    print("  FDD > 5:  Poor quality")


if __name__ == "__main__":
    main()
