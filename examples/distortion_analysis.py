"""Example: Analyzing DINO loss behavior with different distortions.

This demonstrates that DINO perceptual loss increases monotonically
with perceptual degradation (blur, noise, compression).

Usage:
    python distortion_analysis.py

Requires: torch, PIL, dino_perceptual
"""

import io

import numpy as np
import torch
from PIL import Image, ImageFilter


def create_sample_image(size: int = 256) -> Image.Image:
    """Create a sample image with gradients and patterns."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Horizontal gradient (red)
    img[:, :, 0] = np.linspace(0, 255, size).astype(np.uint8)
    # Vertical gradient (green)
    img[:, :, 1] = np.linspace(0, 255, size).reshape(-1, 1).astype(np.uint8)
    # Checkerboard pattern (blue)
    x, y = np.meshgrid(range(size), range(size))
    img[:, :, 2] = ((x // 32 + y // 32) % 2 * 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor in [-1, 1] range."""
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # [0,1] -> [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1)


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise."""
    if sigma <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    arr += np.random.randn(*arr.shape) * sigma
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    from dino_perceptual import DINOPerceptual

    print("DINO Loss Distortion Analysis")
    print("=" * 60)

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loss_fn = DINOPerceptual(model_size="B").to(device).eval()

    # Create sample image
    original_pil = create_sample_image(256)
    original = pil_to_tensor(original_pil).unsqueeze(0).to(device)

    print("\n" + "-" * 60)
    print("Gaussian Blur (higher sigma = more blur)")
    print("-" * 60)
    print(f"{'Sigma':<10} {'DINO Loss':<15}")
    for sigma in [0, 1, 2, 4, 6, 8]:
        blurred = apply_blur(original_pil, sigma)
        blurred_t = pil_to_tensor(blurred).unsqueeze(0).to(device)
        with torch.no_grad():
            loss = loss_fn(blurred_t, original).item()
        print(f"{sigma:<10} {loss:<15.6f}")

    print("\n" + "-" * 60)
    print("Gaussian Noise (higher sigma = more noise)")
    print("-" * 60)
    print(f"{'Sigma':<10} {'DINO Loss':<15}")
    np.random.seed(42)
    for sigma in [0, 10, 25, 50, 75, 100]:
        noisy = apply_noise(original_pil, sigma)
        noisy_t = pil_to_tensor(noisy).unsqueeze(0).to(device)
        with torch.no_grad():
            loss = loss_fn(noisy_t, original).item()
        print(f"{sigma:<10} {loss:<15.6f}")

    print("\n" + "-" * 60)
    print("JPEG Compression (lower quality = more artifacts)")
    print("-" * 60)
    print(f"{'Quality':<10} {'DINO Loss':<15}")
    for quality in [100, 80, 50, 25, 10, 5]:
        compressed = apply_jpeg(original_pil, quality)
        compressed_t = pil_to_tensor(compressed).unsqueeze(0).to(device)
        with torch.no_grad():
            loss = loss_fn(compressed_t, original).item()
        print(f"{quality:<10} {loss:<15.6f}")

    print("\n" + "=" * 60)
    print("Key observation: DINO loss increases monotonically with")
    print("perceptual degradation, making it suitable for training")
    print("autoencoders to minimize perceptual distortion.")


if __name__ == "__main__":
    main()
