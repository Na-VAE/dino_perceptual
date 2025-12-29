"""Figure: Gradient Overlay Visualization.

Shows gradient magnitude overlaid on the original image to visualize
where each loss function focuses its attention.

Usage:
    modal run examples/figures/fig_gradient_overlay.py

Output:
    fig_gradient_overlay.png
"""

import modal

volume = modal.Volume.from_name("dino-perceptual-data", create_if_missing=True)

app = modal.App("dino-fig-gradient-overlay")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "numpy", "pillow", "matplotlib", "lpips")
    .add_local_python_source("dino_perceptual")
)


@app.function(gpu="A10G", image=image, volumes={"/data": volume}, timeout=600)
def generate_figure():
    import torch
    import numpy as np
    from PIL import Image, ImageFilter
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from torchvision.datasets import STL10
    import lpips
    import io
    from dino_perceptual import DINOPerceptual

    device = torch.device("cuda")

    # Load models
    dino = DINOPerceptual(model_size="B", version="v2").to(device).eval()
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # Load image
    stl10 = STL10(root="/data/stl10", split="test", download=False)
    for img, label in stl10:
        if label == 3:  # cat
            original_pil = img.resize((512, 512), Image.LANCZOS)
            break

    # Create distorted version (strong blur for visibility)
    distorted_pil = original_pil.filter(ImageFilter.GaussianBlur(radius=8))

    def pil_to_tensor(img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1

    original_t = pil_to_tensor(original_pil).unsqueeze(0).to(device)
    distorted_t = pil_to_tensor(distorted_pil).unsqueeze(0).to(device)

    def get_gradient(loss_fn, x, ref, is_l1=False):
        x = x.clone().requires_grad_(True)
        if is_l1:
            loss = torch.nn.functional.l1_loss(x, ref)
        else:
            loss = loss_fn(x, ref).mean()
        loss.backward()
        grad = x.grad.detach()
        # Normalize gradient to unit norm (all on same scale)
        grad_norm = torch.linalg.vector_norm(grad)
        return grad / (grad_norm + 1e-8)

    # Compute gradients (all normalized to same scale)
    l1_grad = get_gradient(None, distorted_t, original_t, is_l1=True)
    dino_grad = get_gradient(dino, distorted_t, original_t)
    lpips_grad = get_gradient(lpips_fn, distorted_t, original_t)

    def grad_to_heatmap(grad):
        mag = torch.sqrt((grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
        return mag

    # Get magnitudes and shared max for consistent scale
    l1_mag = grad_to_heatmap(l1_grad)
    dino_mag = grad_to_heatmap(dino_grad)
    lpips_mag = grad_to_heatmap(lpips_grad)
    shared_max = max(l1_mag.max(), dino_mag.max(), lpips_mag.max())

    def overlay_heatmap(img_arr, heatmap, alpha=0.6, cmap_name="hot"):
        """Overlay heatmap on image."""
        cmap = plt.get_cmap(cmap_name)
        heatmap_colored = cmap(heatmap)[:, :, :3]
        blended = (1 - alpha) * (img_arr / 255.0) + alpha * heatmap_colored
        return np.clip(blended, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    original_arr = np.array(original_pil)
    distorted_arr = np.array(distorted_pil)

    # Top row: Original, Distorted, Difference
    axes[0, 0].imshow(original_arr)
    axes[0, 0].set_title("Original", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(distorted_arr)
    axes[0, 1].set_title("Distorted (Gaussian Blur)", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    diff = np.abs(original_arr.astype(float) - distorted_arr.astype(float))
    diff = (diff / diff.max() * 255).astype(np.uint8)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title("Pixel Difference", fontsize=14, fontweight="bold")
    axes[0, 2].axis("off")

    # Bottom row: Gradient overlays (all normalized to same scale)
    l1_overlay = overlay_heatmap(distorted_arr, l1_mag / shared_max, alpha=0.5)
    axes[1, 0].imshow(l1_overlay)
    axes[1, 0].set_title("L1 Gradient Focus", fontsize=14, fontweight="bold", color="#e74c3c")
    axes[1, 0].axis("off")

    dino_overlay = overlay_heatmap(distorted_arr, dino_mag / shared_max, alpha=0.5)
    axes[1, 1].imshow(dino_overlay)
    axes[1, 1].set_title("DINO Gradient Focus", fontsize=14, fontweight="bold", color="#27ae60")
    axes[1, 1].axis("off")

    lpips_overlay = overlay_heatmap(distorted_arr, lpips_mag / shared_max, alpha=0.5)
    axes[1, 2].imshow(lpips_overlay)
    axes[1, 2].set_title("LPIPS Gradient Focus", fontsize=14, fontweight="bold", color="#3498db")
    axes[1, 2].axis("off")

    fig.suptitle("Where Does Each Loss Function Focus?", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close()

    return buf.getvalue()


@app.local_entrypoint()
def main():
    img_bytes = generate_figure.remote()
    with open("fig_gradient_overlay.png", "wb") as f:
        f.write(img_bytes)
    print("Saved: fig_gradient_overlay.png")
