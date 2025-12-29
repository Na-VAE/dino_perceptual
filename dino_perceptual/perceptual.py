"""DINO-based perceptual loss and feature extraction.

Provides:
- DINOPerceptual: LPIPS-like perceptual loss using DINO features (v2 or v3)
- DINOModel: Feature extractor for FDD (Frechet DINO Distance)

Usage:
    from dino_perceptual import DINOPerceptual, DINOModel

    # Perceptual loss (uses DINOv3 by default)
    loss_fn = DINOPerceptual(model_size="B", target_size=512)
    loss = loss_fn(pred_images, ref_images).mean()

    # Feature extraction
    extractor = DINOModel()
    features, _ = extractor(images)  # images in [-1, 1]
"""

from typing import List, Sequence, Union, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# DINOv3 models (default) - trained on LVD-1689M with modern ViT architecture
DINOV3_MODELS = {
    'S': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'B': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'L': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'H': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
    'G': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
}

# DINOv2 models (legacy)
DINOV2_MODELS = {
    'S': 'facebook/dinov2-small',
    'B': 'facebook/dinov2-base',
    'L': 'facebook/dinov2-large',
    'G': 'facebook/dinov2-giant',
}


def _resolve_model_name(model_size: str, version: str = "v3") -> str:
    """Map a size key to a DINO HF model name.

    Args:
        model_size: Size key ('S', 'B', 'L', 'H', 'G')
        version: DINO version ('v2' or 'v3'). Default 'v3'.
    """
    key = str(model_size).strip().upper()
    if version == "v2":
        return DINOV2_MODELS.get(key, DINOV2_MODELS['B'])
    else:
        return DINOV3_MODELS.get(key, DINOV3_MODELS['B'])


class _DINOBase(nn.Module):
    """Shared base for DINO models with common preprocessing."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        version: str = "v3",
        target_size: int = 512,
        resize_to_square: bool = True,
    ):
        super().__init__()
        resolved_name = model_name or _resolve_model_name(model_size, version)
        self.model = AutoModel.from_pretrained(resolved_name)
        self.model_name = resolved_name
        self.version = version

        self.target_size = int(target_size)
        self.resize_to_square = bool(resize_to_square)

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Freeze
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images: [-1,1] -> normalized, optionally resized."""
        x = (x + 1.0) / 2.0

        B, C, H, W = x.shape
        if self.resize_to_square:
            long_side = max(H, W)
            if long_side > self.target_size:
                scale = float(self.target_size) / float(long_side)
                new_h = max(1, int(round(H * scale)))
                new_w = max(1, int(round(W * scale)))
                x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        else:
            if H > self.target_size or W > self.target_size:
                crop_h = min(self.target_size, H)
                crop_w = min(self.target_size, W)
                h_start = (H - crop_h) // 2
                w_start = (W - crop_w) // 2
                x = x[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]

        x = (x - self.mean) / self.std
        return x


class DINOModel(_DINOBase):
    """DINO feature extractor for FDD calculation.

    Extracts CLS token features suitable for Frechet distance calculation.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H', 'G'). Default 'B'.
        version: DINO version ('v2' or 'v3'). Default 'v3'.
        target_size: Target size for preprocessing.
        resize_to_square: If True, resize to target_size. If False, center crop.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        version: str = "v3",
        target_size: int = 512,
        resize_to_square: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            model_size=model_size,
            version=version,
            target_size=target_size,
            resize_to_square=resize_to_square,
        )

    def forward(self, x: torch.Tensor):
        """Extract CLS token features from images.

        Args:
            x: Tensor of shape (B, C, H, W) in range [-1, 1].

        Returns:
            features: Tensor of shape (B, feature_dim).
            None: Placeholder for compatibility.
        """
        x = self._prep(x)

        with torch.inference_mode():
            outputs = self.model(x)

        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features, None


class DINOPerceptual(_DINOBase):
    """DINO-based perceptual loss function.

    Computes an LPIPS-like distance using frozen DINO ViT features:
    for selected transformer layers, take token-wise features (exclude CLS),
    L2-normalize per token, compute squared differences between real and fake
    feature maps, and average only at the very end to a per-image scalar.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H', 'G'). Default 'B'.
        version: DINO version ('v2' or 'v3'). Default 'v3'.
        target_size: Maximum image size. Larger images are downscaled.
        layers: Which layers to use. 'all' or list of 1-based indices.
        normalize: Whether to L2-normalize features per token.
        resize_to_square: If True, resize preserving aspect ratio. If False, center crop.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        version: str = "v3",
        target_size: int = 512,
        layers: Union[str, Sequence[int]] = "all",
        normalize: bool = True,
        resize_to_square: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            model_size=model_size,
            version=version,
            target_size=target_size,
            resize_to_square=resize_to_square,
        )
        self.layers = layers
        self.normalize_feats = bool(normalize)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)
        return x / denom

    def _select_layers(self, hidden_states: List[torch.Tensor]) -> List[int]:
        """Select which hidden state layers to use."""
        n_total = len(hidden_states)
        if isinstance(self.layers, str) and self.layers == 'all':
            return list(range(1, n_total))
        out = []
        for l in self.layers:
            if not isinstance(l, int) or l < 1 or l >= n_total:
                continue
            out.append(l)
        return out if out else list(range(1, n_total))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target images.

        Args:
            pred: Predicted images (B, C, H, W) in [-1, 1] range.
            target: Target images (B, C, H, W) in [-1, 1] range.

        Returns:
            Per-image loss tensor of shape (B,).
        """
        xp = self._prep(pred)
        xt = self._prep(target).detach()

        out_p = self.model(xp, output_hidden_states=True)
        out_t = self.model(xt, output_hidden_states=True)
        hs_p = out_p.hidden_states
        hs_t = out_t.hidden_states

        idxs = self._select_layers(hs_p)
        losses = []
        for i in idxs:
            fp = hs_p[i]
            ft = hs_t[i]
            if self.normalize_feats:
                fp = self._l2_normalize(fp)
                ft = self._l2_normalize(ft)
            l = (fp - ft).pow(2).mean(dim=(1, 2))
            losses.append(l)

        if len(losses) == 0:
            return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec
