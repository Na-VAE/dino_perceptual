"""Figure: Gradient Visualization Across Different Distortions.

Shows how each loss function focuses on different image regions
across various distortion types (blur, noise, jpeg, color shift).

Usage:
    modal run examples/figures/fig_distortions.py

Output:
    fig_distortions.png
"""

import modal

volume = modal.Volume.from_name("dino-perceptual-data", create_if_missing=True)

app = modal.App("dino-fig-distortions")

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

    # Load models
    dino = DINOPerceptual(model_size="B", version="v2").to(device).eval()  # v2 for now since v3 may not be available
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # Load image
    stl10 = STL10(root="/data/stl10", split="test", download=False)
    for img, label in stl10:
        if label == 3:  # cat
            original_pil = img.resize((512, 512), Image.LANCZOS)
            break

    def pil_to_tensor(img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1

    original_t = pil_to_tensor(original_pil).unsqueeze(0).to(device)

    def apply_jpeg(img, quality=10):
        """Apply JPEG compression."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def apply_color_shift(img, shift=30):
        """Apply color channel shift."""
        arr = np.array(img).astype(float)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shift, 0, 255)  # Red shift
        arr[:, :, 2] = np.clip(arr[:, :, 2] - shift, 0, 255)  # Blue shift
        return Image.fromarray(arr.astype(np.uint8))

    # Create different distortions
    distortions = {
        "Blur": original_pil.filter(ImageFilter.GaussianBlur(radius=4)),
        "Noise": Image.fromarray(
            np.clip(
                np.array(original_pil).astype(float) + np.random.normal(0, 25, original_pil.size[::-1] + (3,)),
                0, 255
            ).astype(np.uint8)
        ),
        "JPEG": apply_jpeg(original_pil, quality=10),
        "Color": apply_color_shift(original_pil, shift=30),
    }

    def get_gradient(loss_fn, x, ref, is_l1=False):
        x = x.clone().requires_grad_(True)
        if is_l1:
            loss = torch.nn.functional.l1_loss(x, ref)
        else:
            loss = loss_fn(x, ref).mean()
        loss.backward()
        return x.grad.detach()

    def grad_to_heatmap(grad):
        mag = torch.sqrt((grad ** 2).sum(dim=1)).squeeze().cpu().numpy()
        return mag / (mag.max() + 1e-8)

    # Create figure: 4 distortions x 4 columns (distorted, L1, DINO, LPIPS)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    for row, (dist_name, dist_pil) in enumerate(distortions.items()):
        dist_t = pil_to_tensor(dist_pil).unsqueeze(0).to(device)

        # Compute gradients
        l1_grad = get_gradient(None, dist_t, original_t, is_l1=True)
        dino_grad = get_gradient(dino, dist_t, original_t)
        lpips_grad = get_gradient(lpips_fn, dist_t, original_t)

        # Column 0: Distorted image
        axes[row, 0].imshow(dist_pil)
        axes[row, 0].set_title(f"{dist_name}" if row == 0 else "", fontsize=11)
        axes[row, 0].set_ylabel(dist_name, fontsize=11, rotation=0, ha="right", va="center")
        axes[row, 0].axis("off")

        # Column 1: L1 gradient
        axes[row, 1].imshow(grad_to_heatmap(l1_grad), cmap="inferno")
        axes[row, 1].set_title("L1 Gradient" if row == 0 else "", fontsize=11, color="#3498db")
        axes[row, 1].axis("off")

        # Column 2: DINO gradient
        axes[row, 2].imshow(grad_to_heatmap(dino_grad), cmap="inferno")
        axes[row, 2].set_title("DINO Gradient" if row == 0 else "", fontsize=11, color="#2ecc71")
        axes[row, 2].axis("off")

        # Column 3: LPIPS gradient
        axes[row, 3].imshow(grad_to_heatmap(lpips_grad), cmap="inferno")
        axes[row, 3].set_title("LPIPS Gradient" if row == 0 else "", fontsize=11, color="#e74c3c")
        axes[row, 3].axis("off")

    fig.suptitle("Gradient Saliency Across Distortion Types", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close()

    return buf.getvalue()


@app.local_entrypoint()
def main():
    img_bytes = generate_figure.remote()
    with open("fig_distortions.png", "wb") as f:
        f.write(img_bytes)
    print("Saved: fig_distortions.png")
