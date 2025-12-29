"""Figure: Gradient Visualization.

Shows where each loss function focuses by visualizing gradients
on a distorted image. No optimization - just a single snapshot.

Usage:
    modal run examples/figures/fig_gradients.py

Output:
    fig_gradients.png
"""

import modal

volume = modal.Volume.from_name("dino-perceptual-data", create_if_missing=True)

app = modal.App("dino-fig-gradients")

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
    from torchvision.datasets import STL10
    import lpips
    import io
    from dino_perceptual import DINOPerceptual

    device = torch.device("cuda")

    # Load models (use v2 for broader compatibility)
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
        # Normalize gradient to unit norm (entire gradient normalized to scale 1)
        grad_norm = torch.linalg.vector_norm(grad)
        return grad / (grad_norm + 1e-8)

    # Compute gradients (all normalized to same scale)
    l1_grad = get_gradient(None, distorted_t, original_t, is_l1=True)
    dino_grad = get_gradient(dino, distorted_t, original_t)
    lpips_grad = get_gradient(lpips_fn, distorted_t, original_t)

    def grad_to_heatmap(grad, shared_max=None):
        mag = torch.sqrt((grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
        if shared_max is not None:
            return mag / (shared_max + 1e-8)
        return mag / (mag.max() + 1e-8)

    # Get shared max for consistent colormap across all gradients
    l1_mag = torch.sqrt((l1_grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
    dino_mag = torch.sqrt((dino_grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
    lpips_mag = torch.sqrt((lpips_grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
    shared_max = max(l1_mag.max(), dino_mag.max(), lpips_mag.max())

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Top row: images
    axes[0, 0].imshow(original_pil)
    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(distorted_pil)
    axes[0, 1].set_title("Distorted (blur)", fontsize=12)
    axes[0, 1].axis("off")

    # Difference image
    diff = np.abs(np.array(original_pil).astype(float) - np.array(distorted_pil).astype(float))
    diff = (diff / diff.max() * 255).astype(np.uint8)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title("Pixel Difference", fontsize=12)
    axes[0, 2].axis("off")

    # Bottom row: gradients (all using same scale for fair comparison)
    axes[1, 0].imshow(l1_mag / shared_max, cmap="inferno", vmin=0, vmax=1)
    axes[1, 0].set_title("L1 Gradient", fontsize=12, color="#3498db")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(dino_mag / shared_max, cmap="inferno", vmin=0, vmax=1)
    axes[1, 1].set_title("DINO Gradient", fontsize=12, color="#2ecc71")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(lpips_mag / shared_max, cmap="inferno", vmin=0, vmax=1)
    axes[1, 2].set_title("LPIPS Gradient", fontsize=12, color="#e74c3c")
    axes[1, 2].axis("off")

    fig.suptitle("Where Does Each Loss Focus?", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close()

    return buf.getvalue()


@app.local_entrypoint()
def main():
    img_bytes = generate_figure.remote()
    with open("fig_gradients.png", "wb") as f:
        f.write(img_bytes)
    print("Saved: fig_gradients.png")
