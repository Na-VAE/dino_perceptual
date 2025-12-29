"""DINO Batch Inference on Modal.

Prerequisites:
    modal run examples/setup_data.py  # Run once to download datasets

Usage:
    modal run examples/modal_inference.py
"""

import modal

volume = modal.Volume.from_name("dino-perceptual-data", create_if_missing=True)

app = modal.App("dino-perceptual")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "numpy", "pillow")
    .add_local_python_source("dino_perceptual")
)


@app.function(gpu="A10G", image=image, volumes={"/data": volume}, timeout=300)
def demo():
    """Quick demo of DINOPerceptual and DINOModel."""
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.datasets import STL10
    from dino_perceptual import DINOPerceptual, DINOModel

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Load from persistent volume
    print("\nLoading STL-10 from volume...")
    stl10 = STL10(root="/data/stl10", split="test", download=False)

    def pil_to_tensor(img):
        arr = np.array(img.resize((224, 224), Image.LANCZOS))
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1

    images = []
    for img, label in stl10:
        if len(images) < 8:
            images.append(pil_to_tensor(img))
        else:
            break
    batch = torch.stack(images).to(device)

    # DINOPerceptual
    print("\n--- DINOPerceptual ---")
    perceptual = DINOPerceptual(model_size="B").to(device).eval()

    with torch.no_grad():
        loss_same = perceptual(batch[:1], batch[:1]).item()
        loss_diff = perceptual(batch[:1], batch[1:2]).item()

    print(f"Same image loss: {loss_same:.6f}")
    print(f"Diff image loss: {loss_diff:.6f}")

    # DINOModel
    print("\n--- DINOModel ---")
    extractor = DINOModel(model_size="B").to(device).eval()

    with torch.no_grad():
        features, _ = extractor(batch)

    print(f"Batch size: {len(images)}")
    print(f"Feature shape: {features.shape}")

    # Similarity matrix
    features_norm = features / features.norm(dim=1, keepdim=True)
    sim = (features_norm @ features_norm.T).cpu().numpy()
    print("\nCosine similarity matrix:")
    print(np.array2string(sim, precision=2, suppress_small=True))

    return {"loss_same": loss_same, "loss_diff": loss_diff, "feature_dim": features.shape[1]}


@app.local_entrypoint()
def main():
    result = demo.remote()
    print(f"\nResult: {result}")
