"""Run tests and examples on Modal with GPU.

Usage:
    modal run examples/run_tests_modal.py
"""

import modal

app = modal.App("dino-perceptual-tests")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "numpy", "pillow", "scipy", "pytest")
    .add_local_python_source("dino_perceptual")
    .add_local_python_source("tests")
)


@app.function(gpu="A10G", image=image, timeout=600)
def run_tests():
    """Run pytest on GPU."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "-v", "--tb=short", "-x"],
        capture_output=True,
        text=True,
        cwd="/root"
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode


@app.function(gpu="A10G", image=image, timeout=300)
def run_distortion_analysis():
    """Run distortion analysis example."""
    import io
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    from dino_perceptual import DINOPerceptual

    def create_sample_image(size: int = 256) -> Image.Image:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, size).astype(np.uint8)
        img[:, :, 1] = np.linspace(0, 255, size).reshape(-1, 1).astype(np.uint8)
        x, y = np.meshgrid(range(size), range(size))
        img[:, :, 2] = ((x // 32 + y // 32) % 2 * 255).astype(np.uint8)
        return Image.fromarray(img)

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2 - 1
        return torch.from_numpy(arr).permute(2, 0, 1)

    def apply_blur(img, sigma):
        if sigma <= 0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def apply_noise(img, sigma):
        if sigma <= 0:
            return img
        arr = np.array(img).astype(np.float32)
        arr += np.random.randn(*arr.shape) * sigma
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def apply_jpeg(img, quality):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    print("DINO Loss Distortion Analysis")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")

    loss_fn = DINOPerceptual(model_size="B", version="v2").cuda().eval()
    original_pil = create_sample_image(256)
    original = pil_to_tensor(original_pil).unsqueeze(0).cuda()

    results = {"blur": [], "noise": [], "jpeg": []}

    print("\n" + "-" * 60)
    print("Gaussian Blur")
    print("-" * 60)
    for sigma in [0, 1, 2, 4, 6, 8]:
        blurred = apply_blur(original_pil, sigma)
        blurred_t = pil_to_tensor(blurred).unsqueeze(0).cuda()
        with torch.no_grad():
            loss = loss_fn(blurred_t, original).item()
        results["blur"].append((sigma, loss))
        print(f"sigma={sigma:<3}  loss={loss:.6f}")

    print("\n" + "-" * 60)
    print("Gaussian Noise")
    print("-" * 60)
    np.random.seed(42)
    for sigma in [0, 10, 25, 50, 75, 100]:
        noisy = apply_noise(original_pil, sigma)
        noisy_t = pil_to_tensor(noisy).unsqueeze(0).cuda()
        with torch.no_grad():
            loss = loss_fn(noisy_t, original).item()
        results["noise"].append((sigma, loss))
        print(f"sigma={sigma:<3}  loss={loss:.6f}")

    print("\n" + "-" * 60)
    print("JPEG Compression")
    print("-" * 60)
    for quality in [100, 80, 50, 25, 10, 5]:
        compressed = apply_jpeg(original_pil, quality)
        compressed_t = pil_to_tensor(compressed).unsqueeze(0).cuda()
        with torch.no_grad():
            loss = loss_fn(compressed_t, original).item()
        results["jpeg"].append((quality, loss))
        print(f"quality={quality:<3}  loss={loss:.6f}")

    return results


@app.function(gpu="A10G", image=image, timeout=300)
def run_fdd_example():
    """Run FDD computation example."""
    import numpy as np
    import torch
    from scipy import linalg
    from dino_perceptual import DINOModel

    def compute_fdd(real_features, fake_features):
        mu1 = real_features.mean(axis=0)
        mu2 = fake_features.mean(axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(fake_features, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    print("Frechet DINO Distance (FDD) Example")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")

    extractor = DINOModel(model_size="B", version="v2").cuda().eval()

    torch.manual_seed(42)
    real_images = torch.randn(32, 3, 224, 224).cuda().clamp(-1, 1)
    reconstructed = real_images + torch.randn_like(real_images) * 0.1
    reconstructed = reconstructed.clamp(-1, 1)
    poor_recon = real_images + torch.randn_like(real_images) * 0.5
    poor_recon = poor_recon.clamp(-1, 1)

    print("\nExtracting features...")
    with torch.no_grad():
        real_feats, _ = extractor(real_images)
        recon_feats, _ = extractor(reconstructed)
        poor_feats, _ = extractor(poor_recon)

    print(f"Feature shape: {real_feats.shape}")

    fdd_same = compute_fdd(real_feats.cpu().numpy(), real_feats.cpu().numpy())
    fdd_good = compute_fdd(real_feats.cpu().numpy(), recon_feats.cpu().numpy())
    fdd_poor = compute_fdd(real_feats.cpu().numpy(), poor_feats.cpu().numpy())

    print("\n" + "-" * 60)
    print("FDD Results:")
    print("-" * 60)
    print(f"Same distribution:      {fdd_same:.4f}  (should be ~0)")
    print(f"Good reconstruction:    {fdd_good:.4f}")
    print(f"Poor reconstruction:    {fdd_poor:.4f}")
    print("\nInterpretation: <1 excellent, 1-5 good, >5 poor")

    return {"fdd_same": fdd_same, "fdd_good": fdd_good, "fdd_poor": fdd_poor}


@app.local_entrypoint()
def main():
    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)
    test_result = run_tests.remote()
    print(f"\nTest exit code: {test_result}")

    print("\n" + "=" * 70)
    print("Running Distortion Analysis")
    print("=" * 70)
    distortion_results = run_distortion_analysis.remote()

    print("\n" + "=" * 70)
    print("Running FDD Example")
    print("=" * 70)
    fdd_results = run_fdd_example.remote()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Tests: {'PASSED' if test_result == 0 else 'FAILED'}")
    print(f"FDD results: {fdd_results}")
