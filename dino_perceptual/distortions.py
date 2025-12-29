"""Shared distortion functions for perceptual loss research.

VAE-realistic and ImageNet-C style corruptions for comparing DINO vs LPIPS.
"""

import numpy as np
from PIL import Image, ImageFilter
import io


# =============================================================================
# Basic Distortions (existing patterns)
# =============================================================================

def gaussian_blur(img, sigma):
    """Gaussian blur with configurable sigma."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(img, sigma):
    """Additive Gaussian noise."""
    if sigma <= 0:
        return img
    arr = np.array(img).astype(float)
    arr += np.random.randn(*arr.shape) * sigma
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def color_shift(img, strength):
    """Shift red channel up, blue channel down."""
    if strength <= 0:
        return img
    arr = np.array(img).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + strength), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - strength), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# =============================================================================
# VAE-Realistic Distortions
# =============================================================================

def add_checkerboard(img, block_size=8, strength=0.3):
    """Simulate VAE decoder checkerboard artifacts.

    These occur due to transposed convolution stride misalignment,
    creating periodic grid-like intensity modulation.

    Args:
        img: PIL Image
        block_size: Size of checkerboard blocks (typically 8 or 16)
        strength: Intensity of the artifact (0 to 1)
    """
    if strength <= 0:
        return img
    arr = np.array(img).astype(float)
    H, W = arr.shape[:2]

    # Create alternating pattern at block boundaries
    y_grid = np.arange(H) // block_size
    x_grid = np.arange(W) // block_size
    checker = ((y_grid[:, None] + x_grid[None, :]) % 2).astype(float)

    # Apply modulation
    modulation = 1.0 + strength * (checker - 0.5)
    arr = arr * modulation[:, :, None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_compress(img, quality):
    """JPEG compression artifacts (blocking and ringing).

    Args:
        img: PIL Image
        quality: JPEG quality (1-100), lower = more artifacts
    """
    quality = max(1, min(100, int(quality)))
    if quality >= 100:
        return img

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def motion_blur(img, kernel_size=15, angle=0):
    """Directional motion blur.

    Args:
        img: PIL Image
        kernel_size: Size of the blur kernel
        angle: Direction of motion in degrees
    """
    if kernel_size <= 1:
        return img

    # Create motion blur kernel (line at angle)
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.cos(np.radians(angle)))
        y = int(center + offset * np.sin(np.radians(angle)))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1

    # Normalize
    kernel = kernel / (kernel.sum() + 1e-8)

    # Apply via convolution
    arr = np.array(img).astype(float)
    from scipy.ndimage import convolve
    result = np.stack([convolve(arr[:, :, c], kernel, mode='reflect') for c in range(3)], axis=2)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def color_bleed(img, strength=0.2):
    """Simulate color channel misalignment/bleeding.

    Shifts color channels relative to each other, common in
    poor decoders or compression artifacts.

    Args:
        img: PIL Image
        strength: Amount of shift (0 to 1, maps to pixel shift)
    """
    if strength <= 0:
        return img

    arr = np.array(img).astype(float)
    H, W = arr.shape[:2]
    shift = max(1, int(strength * 10))

    result = np.zeros_like(arr)

    # Shift R channel right
    result[:, shift:, 0] = arr[:, :-shift, 0]
    result[:, :shift, 0] = arr[:, :shift, 0]

    # G channel unchanged
    result[:, :, 1] = arr[:, :, 1]

    # Shift B channel left
    result[:, :-shift, 2] = arr[:, shift:, 2]
    result[:, -shift:, 2] = arr[:, -shift:, 2]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def quantize_colors(img, levels=16):
    """Reduce color depth for banding artifacts.

    Args:
        img: PIL Image
        levels: Number of color levels per channel (2-256)
    """
    levels = max(2, min(256, int(levels)))
    if levels >= 256:
        return img

    arr = np.array(img).astype(float)
    factor = 256.0 / levels
    arr = np.floor(arr / factor) * factor

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def downsample_upsample(img, factor):
    """Downsample then upsample (resolution loss).

    Args:
        img: PIL Image
        factor: Downsample factor (1 = no change, higher = more loss)
    """
    if factor <= 1:
        return img

    factor = int(factor)
    orig_size = img.size
    small_size = (max(1, img.width // factor), max(1, img.height // factor))

    small = img.resize(small_size, Image.BILINEAR)
    return small.resize(orig_size, Image.BILINEAR)


# =============================================================================
# ImageNet-C Style Severity Levels
# =============================================================================

CORRUPTION_SEVERITIES = {
    'gaussian_blur': {
        'fn': gaussian_blur,
        'params': [0, 1, 2, 3, 4, 6],  # sigma values for severity 0-5
    },
    'gaussian_noise': {
        'fn': gaussian_noise,
        'params': [0, 10, 20, 30, 40, 50],  # sigma values
    },
    'jpeg': {
        'fn': jpeg_compress,
        'params': [100, 80, 60, 40, 20, 10],  # quality (inverted)
    },
    'motion_blur': {
        'fn': lambda img, k: motion_blur(img, kernel_size=k, angle=45),
        'params': [0, 5, 10, 15, 20, 25],  # kernel sizes
    },
    'checkerboard': {
        'fn': lambda img, s: add_checkerboard(img, block_size=8, strength=s),
        'params': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # strength
    },
    'color_shift': {
        'fn': color_shift,
        'params': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # strength
    },
    'color_bleed': {
        'fn': color_bleed,
        'params': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # strength
    },
    'quantize': {
        'fn': quantize_colors,
        'params': [256, 64, 32, 16, 8, 4],  # levels (inverted)
    },
}


def apply_corruption(img, corruption_name, severity):
    """Apply a corruption at specified severity level.

    Args:
        img: PIL Image
        corruption_name: Key from CORRUPTION_SEVERITIES
        severity: 0-5 (0 = no corruption)

    Returns:
        Corrupted PIL Image
    """
    if corruption_name not in CORRUPTION_SEVERITIES:
        raise ValueError(f"Unknown corruption: {corruption_name}")

    config = CORRUPTION_SEVERITIES[corruption_name]
    param = config['params'][min(severity, len(config['params']) - 1)]

    return config['fn'](img, param)
