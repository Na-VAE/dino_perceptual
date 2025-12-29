"""Example: Using DINOPerceptual as a training loss.

Shows how to use DINO perceptual loss in a typical training loop.
Uses DINOv3 by default with bfloat16 and torch.compile for best performance.
"""

import torch
import torch.nn as nn
from dino_perceptual import DINOPerceptual

# Initialize the perceptual loss (uses DINOv3 by default)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perceptual_loss = DINOPerceptual(model_size="B").to(device).bfloat16().eval()
perceptual_loss = torch.compile(perceptual_loss, fullgraph=True)

# Example: Simple autoencoder training
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Training loop example
model = SimpleAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Dummy batch (replace with your dataloader)
batch = torch.randn(4, 3, 256, 256).to(device)

for step in range(100):
    optimizer.zero_grad()

    # Forward pass
    reconstructed = model(batch)

    # Compute losses
    l1_loss = nn.functional.l1_loss(reconstructed, batch)
    dino_loss = perceptual_loss(reconstructed, batch).mean()

    # Combined loss (weight perceptual term appropriately)
    total_loss = l1_loss + 0.1 * dino_loss

    # Backward pass
    total_loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: L1={l1_loss.item():.4f}, DINO={dino_loss.item():.4f}")

print("Done!")
