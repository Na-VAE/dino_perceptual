"""DINOv3 model for feature extraction.

Provides feature extraction for FDD (Frechet DINO Distance) calculation,
a modern alternative to Inception features for FID.

Usage:
    from dino_perceptual import DINOv3Model

    extractor = DINOv3Model()
    features, _ = extractor(images)  # images in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv3Model(nn.Module):
    """DINOv3 feature extractor for FDD calculation.

    Uses DINOv3 models from Hugging Face Transformers to extract CLS token
    features suitable for Frechet distance calculation.

    Args:
        model_name: HuggingFace model name. Default is ViT-B/16.
        resize_to_square: If True, resize to target_size. If False, center crop.
        target_size: Target size for preprocessing.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        resize_to_square: bool = False,
        target_size: int = 512,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

        self.resize_to_square = resize_to_square
        self.target_size = target_size

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size

    def forward(self, x: torch.Tensor):
        """Extract CLS token features from images.

        Args:
            x: Tensor of shape (B, C, H, W) in range [-1, 1].

        Returns:
            features: Tensor of shape (B, feature_dim).
            None: Placeholder for compatibility (no probs needed).
        """
        B, C, H, W = x.shape

        if self.resize_to_square:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                            mode='bicubic', align_corners=False)
        else:
            if H > self.target_size or W > self.target_size:
                crop_h = min(self.target_size, H)
                crop_w = min(self.target_size, W)
                h_start = (H - crop_h) // 2
                w_start = (W - crop_w) // 2
                x = x[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]

        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2

        # ImageNet normalization
        x = (x - self.mean) / self.std

        with torch.inference_mode():
            outputs = self.model(x)

        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features, None
