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

import numbers
from typing import List, Sequence, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


# DINOv3 models (default) - trained on LVD-1689M
DINOV3_MODELS = {
    'S': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'B': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'L': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'H': 'facebook/dinov3-vith14-pretrain-lvd1689m',
}

# DINOv2 models (legacy)
DINOV2_MODELS = {
    'S': 'facebook/dinov2-small',
    'B': 'facebook/dinov2-base',
    'L': 'facebook/dinov2-large',
    'G': 'facebook/dinov2-giant',
}


def _resolve_model_name(model_size: str, version: str = "v3") -> str:
    """Map a size key to a DINO HF model name."""
    key = str(model_size).strip().upper()
    if version == "v2":
        return DINOV2_MODELS.get(key, DINOV2_MODELS['B'])
    return DINOV3_MODELS.get(key, DINOV3_MODELS['B'])


def _prep(x: torch.Tensor, target_size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Preprocess images: [-1,1] -> ImageNet normalized, optionally resized."""
    x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
    H, W = x.shape[2], x.shape[3]
    long_side = max(H, W)
    if long_side > target_size:
        scale = target_size / long_side
        new_h, new_w = max(1, round(H * scale)), max(1, round(W * scale))
        x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
    return (x - mean) / std


class _DINOBase(nn.Module):
    """Shared base for DINO models."""

    def __init__(
        self,
        model_name: Optional[str],
        model_size: str,
        version: str,
        target_size: int,
        preprocess: Union[str, bool],
    ):
        super().__init__()
        resolved_name = model_name or _resolve_model_name(model_size, version)
        self.model = AutoModel.from_pretrained(resolved_name)
        self.model_name = resolved_name
        self.version = version
        self.target_size = target_size
        self.preprocess = preprocess

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # HuggingFace processor for "auto" mode
        self._hf_processor = None
        if preprocess == "auto":
            self._hf_processor = AutoImageProcessor.from_pretrained(resolved_name)

        # Freeze
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_dim = self.model.config.hidden_size

    def _apply_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing based on mode."""
        if self.preprocess == "auto" and self._hf_processor is not None:
            x = self._hf_processor(x, return_tensors="pt", do_rescale=False)["pixel_values"]
            return x.to(self.mean.device)
        elif self.preprocess:
            return _prep(x, self.target_size, self.mean, self.std)
        return x


class DINOModel(_DINOBase):
    """DINO feature extractor for FDD calculation.

    Extracts CLS token features suitable for Frechet distance calculation.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H'). Default 'B'.
        version: DINO version ('v2' or 'v3'). Default 'v3'.
        target_size: Target size for preprocessing. Images larger than this are downscaled.
        preprocess: Preprocessing mode:
            - "auto": Use HuggingFace AutoImageProcessor
            - True: Use internal preprocessing (expects [-1, 1] input, default)
            - False: Skip preprocessing (expects already normalized input)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        version: str = "v3",
        target_size: int = 512,
        preprocess: Union[str, bool] = True,
    ):
        super().__init__(model_name, model_size, version, target_size, preprocess)

    def forward(self, x: torch.Tensor):
        """Extract CLS token features from images.

        Args:
            x: Tensor of shape (B, C, H, W). Expected range depends on preprocess mode.

        Returns:
            features: Tensor of shape (B, feature_dim).
            None: Placeholder for compatibility.
        """
        x = self._apply_preprocess(x)
        with torch.inference_mode():
            outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :], None


class DINOPerceptual(_DINOBase):
    """DINO-based perceptual loss function.

    Computes LPIPS-like distance using frozen DINO ViT features:
    for selected transformer layers, take all token features (CLS + patches),
    L2-normalize per token, compute squared differences, and average to
    a per-image scalar.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H'). Default 'B'.
        version: DINO version ('v2' or 'v3'). Default 'v3'.
        target_size: Max image size. Larger images are downscaled preserving aspect ratio.
        layers: Which layers to use. 'all' or list of 1-based indices.
        normalize: Whether to L2-normalize features per token.
        preprocess: Preprocessing mode:
            - "auto": Use HuggingFace AutoImageProcessor
            - True: Use internal preprocessing (expects [-1, 1] input, default)
            - False: Skip preprocessing (expects already normalized input)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        version: str = "v3",
        target_size: int = 512,
        layers: Union[str, Sequence[int]] = "all",
        normalize: bool = True,
        preprocess: Union[str, bool] = True,
    ):
        super().__init__(model_name, model_size, version, target_size, preprocess)
        self.layers = layers
        self.normalize_feats = normalize

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
            if not isinstance(l, numbers.Integral) or l < 1 or l >= n_total:
                continue
            out.append(int(l))
        return out if out else list(range(1, n_total))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target images.

        Args:
            pred: Predicted images (B, C, H, W). Expected range depends on preprocess mode.
            target: Target images (B, C, H, W). Same range as pred.

        Returns:
            Per-image loss tensor of shape (B,).
        """
        xp = self._apply_preprocess(pred)
        xt = self._apply_preprocess(target).detach()

        out_p = self.model(xp, output_hidden_states=True)
        out_t = self.model(xt, output_hidden_states=True)
        hs_p = out_p.hidden_states
        hs_t = out_t.hidden_states

        idxs = self._select_layers(hs_p)
        losses = []
        for i in idxs:
            fp, ft = hs_p[i], hs_t[i]
            if self.normalize_feats:
                fp, ft = self._l2_normalize(fp), self._l2_normalize(ft)
            losses.append((fp - ft).pow(2).mean(dim=(1, 2)))

        if not losses:
            return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        return torch.stack(losses, dim=0).mean(dim=0)
