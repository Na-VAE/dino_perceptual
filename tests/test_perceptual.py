"""Tests for DINO perceptual loss."""

import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter


class TestDINOPerceptualImport:
    """Test that the package imports correctly."""

    def test_import_dino_perceptual(self):
        from dino_perceptual import DINOPerceptual
        assert DINOPerceptual is not None

    def test_import_dino_model(self):
        from dino_perceptual import DINOModel
        assert DINOModel is not None

    def test_version(self):
        import dino_perceptual
        assert hasattr(dino_perceptual, "__version__")
        assert dino_perceptual.__version__ == "0.1.2"


class TestDINOPerceptualInit:
    """Test DINOPerceptual initialization."""

    def test_init_v2(self):
        from dino_perceptual import DINOPerceptual
        # Test v2 init (v3 requires gated HuggingFace access)
        loss_fn = DINOPerceptual(model_size="B", version="v2")
        assert loss_fn is not None
        assert loss_fn.version == "v2"
        assert "dinov2" in loss_fn.model_name

    def test_init_custom_target_size(self):
        from dino_perceptual import DINOPerceptual
        loss_fn = DINOPerceptual(model_size="B", version="v2", target_size=256)
        assert loss_fn.target_size == 256


class TestDINOModelInit:
    """Test DINOModel initialization."""

    def test_init_v2(self):
        from dino_perceptual import DINOModel
        # Test v2 init (v3 requires gated HuggingFace access)
        model = DINOModel(model_size="B", version="v2")
        assert model is not None
        assert model.version == "v2"
        assert "dinov2" in model.model_name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDINOPerceptualForward:
    """Test forward pass (requires GPU and model weights)."""

    def test_forward_same_image(self):
        from dino_perceptual import DINOPerceptual

        loss_fn = DINOPerceptual(model_size="B", version="v2").cuda().eval()

        # Create random images in [-1, 1] range
        x = torch.randn(2, 3, 256, 256).cuda()
        x = x.clamp(-1, 1)

        # Same image should have zero loss
        loss = loss_fn(x, x)
        assert loss.shape == (2,)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)

    def test_forward_different_images(self):
        from dino_perceptual import DINOPerceptual

        loss_fn = DINOPerceptual(model_size="B", version="v2").cuda().eval()

        x1 = torch.randn(2, 3, 256, 256).cuda().clamp(-1, 1)
        x2 = torch.randn(2, 3, 256, 256).cuda().clamp(-1, 1)

        loss = loss_fn(x1, x2)
        assert loss.shape == (2,)
        assert (loss > 0).all()

    def test_forward_gradient(self):
        from dino_perceptual import DINOPerceptual

        loss_fn = DINOPerceptual(model_size="B", version="v2").cuda().eval()

        # Create tensor on CUDA first, then enable gradients
        x1 = torch.randn(1, 3, 256, 256, device="cuda", requires_grad=True)
        x2 = torch.randn(1, 3, 256, 256, device="cuda")

        loss = loss_fn(x1, x2).mean()
        loss.backward()

        assert x1.grad is not None
        assert x1.grad.shape == x1.shape


# Helper functions for distortion tests
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor in [-1, 1] range."""
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # [0,1] -> [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor in [-1, 1] range to PIL image."""
    arr = ((t.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur to PIL image."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise to PIL image."""
    if sigma <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    arr += np.random.randn(*arr.shape) * sigma
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression to PIL image."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDINODistortionBehavior:
    """Test that DINO loss increases with image degradation."""

    @pytest.fixture
    def loss_fn(self):
        from dino_perceptual import DINOPerceptual
        return DINOPerceptual(model_size="B", version="v2").cuda().eval()

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with structure (not just noise)."""
        # Create image with gradients and patterns
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Horizontal gradient
        img[:, :, 0] = np.linspace(0, 255, 256).astype(np.uint8)
        # Vertical gradient
        img[:, :, 1] = np.linspace(0, 255, 256).reshape(-1, 1).astype(np.uint8)
        # Checkerboard pattern
        x, y = np.meshgrid(range(256), range(256))
        img[:, :, 2] = ((x // 32 + y // 32) % 2 * 255).astype(np.uint8)
        return Image.fromarray(img)

    def test_blur_increases_loss(self, loss_fn, sample_image):
        """Gaussian blur at increasing sigma should increase loss."""
        original = pil_to_tensor(sample_image).unsqueeze(0).cuda()

        losses = []
        for sigma in [0, 2, 4]:
            blurred = apply_gaussian_blur(sample_image, sigma)
            blurred_t = pil_to_tensor(blurred).unsqueeze(0).cuda()
            loss = loss_fn(blurred_t, original).item()
            losses.append(loss)

        # Loss should increase monotonically with blur
        assert losses[0] < losses[1] < losses[2], f"Blur losses not monotonic: {losses}"

    def test_noise_increases_loss(self, loss_fn, sample_image):
        """Gaussian noise at increasing sigma should increase loss."""
        original = pil_to_tensor(sample_image).unsqueeze(0).cuda()

        losses = []
        for sigma in [0, 25, 50]:
            np.random.seed(42)  # Reproducibility
            noisy = apply_gaussian_noise(sample_image, sigma)
            noisy_t = pil_to_tensor(noisy).unsqueeze(0).cuda()
            loss = loss_fn(noisy_t, original).item()
            losses.append(loss)

        # Loss should increase with noise
        assert losses[0] < losses[1] < losses[2], f"Noise losses not monotonic: {losses}"

    def test_jpeg_increases_loss(self, loss_fn, sample_image):
        """Lower JPEG quality should increase loss."""
        original = pil_to_tensor(sample_image).unsqueeze(0).cuda()

        losses = []
        for quality in [100, 50, 10]:
            compressed = apply_jpeg_compression(sample_image, quality)
            compressed_t = pil_to_tensor(compressed).unsqueeze(0).cuda()
            loss = loss_fn(compressed_t, original).item()
            losses.append(loss)

        # Loss should increase as quality decreases
        assert losses[0] < losses[1] < losses[2], f"JPEG losses not monotonic: {losses}"


def compute_fdd(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute Frechet DINO Distance between two feature sets.

    Args:
        real_features: (N, D) array of features from real images
        fake_features: (M, D) array of features from generated images

    Returns:
        FDD score (lower is better, 0 means identical distributions)
    """
    from scipy import linalg

    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fdd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fdd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFDDComputation:
    """Test Frechet DINO Distance computation."""

    @pytest.fixture
    def extractor(self):
        from dino_perceptual import DINOModel
        return DINOModel(model_size="B", version="v2").cuda().eval()

    def test_fdd_same_distribution(self, extractor):
        """FDD between identical image sets should be ~0."""
        # Create batch of random images
        images = torch.randn(16, 3, 224, 224).cuda().clamp(-1, 1)

        with torch.no_grad():
            features, _ = extractor(images)
            features = features.cpu().numpy()

        # FDD of distribution with itself should be ~0
        fdd = compute_fdd(features, features)
        assert fdd < 1e-5, f"FDD of same distribution should be ~0, got {fdd}"

    def test_fdd_different_distributions(self, extractor):
        """FDD between different image sets should be > 0."""
        # Random images
        images1 = torch.randn(16, 3, 224, 224).cuda().clamp(-1, 1)
        # Constant gray images
        images2 = torch.zeros(16, 3, 224, 224).cuda()

        with torch.no_grad():
            features1, _ = extractor(images1)
            features2, _ = extractor(images2)

        fdd = compute_fdd(features1.cpu().numpy(), features2.cpu().numpy())
        assert fdd > 0, f"FDD between different distributions should be > 0, got {fdd}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLossScaling:
    """Test loss magnitude and scaling behavior."""

    @pytest.fixture
    def loss_fn(self):
        from dino_perceptual import DINOPerceptual
        return DINOPerceptual(model_size="B", version="v2").cuda().eval()

    def test_loss_magnitude_range(self, loss_fn):
        """Verify loss is in reasonable range for natural images."""
        # Two different random images
        x1 = torch.randn(4, 3, 256, 256).cuda().clamp(-1, 1)
        x2 = torch.randn(4, 3, 256, 256).cuda().clamp(-1, 1)

        loss = loss_fn(x1, x2)

        # Loss should be in reasonable range (typically 1e-4 to 1e-1)
        assert (loss > 1e-5).all(), f"Loss too small: {loss}"
        assert (loss < 1.0).all(), f"Loss too large: {loss}"

    def test_loss_scales_with_difference(self, loss_fn):
        """Larger pixel differences should produce larger losses."""
        base = torch.randn(1, 3, 256, 256).cuda().clamp(-1, 1)

        # Small perturbation
        small_diff = base + torch.randn_like(base) * 0.1
        small_diff = small_diff.clamp(-1, 1)

        # Large perturbation
        large_diff = base + torch.randn_like(base) * 0.5
        large_diff = large_diff.clamp(-1, 1)

        loss_small = loss_fn(small_diff, base).item()
        loss_large = loss_fn(large_diff, base).item()

        assert loss_small < loss_large, (
            f"Small diff should have smaller loss: {loss_small} vs {loss_large}"
        )
