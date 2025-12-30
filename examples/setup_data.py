"""Setup: Download datasets to Modal volume.

Run this ONCE before running other examples.

Usage:
    modal run examples/setup_data.py

This downloads STL-10 (~2.6GB) to a persistent volume so subsequent
runs don't need to re-download.
"""

import modal

volume = modal.Volume.from_name("dino-perceptual-data", create_if_missing=True)

app = modal.App("dino-setup")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "numpy", "pillow",
)


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def download_datasets():
    """Download datasets to persistent volume."""
    from torchvision.datasets import STL10, CIFAR10
    import os

    print("=" * 60)
    print("DINO Perceptual - Dataset Setup")
    print("=" * 60)

    # Check what's already downloaded
    stl10_path = "/data/stl10"
    cifar10_path = "/data/cifar10"

    # STL-10 (96x96 images, 10 classes)
    if os.path.exists(f"{stl10_path}/stl10_binary"):
        print("\n[STL-10] Already downloaded, skipping...")
    else:
        print("\n[STL-10] Downloading (~2.6GB)...")
        STL10(root=stl10_path, split="test", download=True)
        print("[STL-10] Done!")

    # CIFAR-10 (32x32 images, 10 classes) - smaller, useful for quick tests
    if os.path.exists(f"{cifar10_path}/cifar-10-batches-py"):
        print("\n[CIFAR-10] Already downloaded, skipping...")
    else:
        print("\n[CIFAR-10] Downloading (~170MB)...")
        CIFAR10(root=cifar10_path, train=False, download=True)
        print("[CIFAR-10] Done!")

    # Commit changes to volume
    volume.commit()

    # Summary
    print("\n" + "=" * 60)
    print("Setup complete! Datasets saved to 'dino-perceptual-data' volume.")
    print("=" * 60)

    # List contents
    print("\nVolume contents:")
    for root, dirs, files in os.walk("/data"):
        level = root.replace("/data", "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # Don't go too deep
            for f in files[:5]:
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files) - 5} more files")


@app.local_entrypoint()
def main():
    print("Setting up datasets on Modal volume...")
    print("This only needs to be run once.\n")
    download_datasets.remote()
    print("\nDone! You can now run the examples.")
