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


class DINOModel(nn.Module):
    """DINO feature extractor for FDD calculation.

    Extracts CLS token features suitable for Frechet distance calculation.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H'). Default 'B'.
        version: DINO version ('v2' or 'v3'). Default 'v3'.
        target_size: Target size for preprocessing. Images larger than this are downscaled.
        preprocess: Preprocessing mode:
            - "auto": Use HuggingFace AutoImageProcessor (recommended for new code)
            - True: Use internal preprocessing (expects [-1, 1] input)
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
        super().__init__()
        resolved_name = model_name or _resolve_model_name(model_size, version)
        self.model = AutoModel.from_pretrained(resolved_name)
        self.model_name = resolved_name
        self.version = version
        self.target_size = target_size
        self.preprocess_mode = preprocess

        # ImageNet normalization for internal preprocessing
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

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Internal preprocessing: [-1,1] -> ImageNet normalized, optionally resized."""
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        B, C, H, W = x.shape
        long_side = max(H, W)
        if long_side > self.target_size:
            scale = self.target_size / long_side
            new_h, new_w = max(1, round(H * scale)), max(1, round(W * scale))
            x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x: torch.Tensor):
        """Extract CLS token features from images.

        Args:
            x: Tensor of shape (B, C, H, W). Expected range depends on preprocess mode:
               - preprocess=True: [-1, 1]
               - preprocess="auto": [0, 255] uint8 or [0, 1] float
               - preprocess=False: already ImageNet normalized

        Returns:
            features: Tensor of shape (B, feature_dim).
            None: Placeholder for compatibility.
        """
        if self.preprocess_mode == "auto" and self._hf_processor is not None:
            x = self._hf_processor(x, return_tensors="pt", do_rescale=False)["pixel_values"]
            x = x.to(self.mean.device)
        elif self.preprocess_mode:
            x = self._prep(x)

        with torch.inference_mode():
            outputs = self.model(x)

        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features, None


class DINOPerceptual(nn.Module):
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
        super().__init__()
        resolved_name = model_name or _resolve_model_name(model_size, version)
        self.model = AutoModel.from_pretrained(resolved_name)
        self.model_name = resolved_name
        self.version = version
        self.target_size = target_size
        self.layers = layers
        self.normalize_feats = normalize
        self.preprocess_mode = preprocess

        # ImageNet normalization for internal preprocessing
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

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Internal preprocessing: [-1,1] -> ImageNet normalized, optionally resized."""
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        B, C, H, W = x.shape
        long_side = max(H, W)
        if long_side > self.target_size:
            scale = self.target_size / long_side
            new_h, new_w = max(1, round(H * scale)), max(1, round(W * scale))
            x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        x = (x - self.mean) / self.std
        return x

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
        if self.preprocess_mode == "auto" and self._hf_processor is not None:
            xp = self._hf_processor(pred, return_tensors="pt", do_rescale=False)["pixel_values"]
            xt = self._hf_processor(target, return_tensors="pt", do_rescale=False)["pixel_values"]
            xp = xp.to(self.mean.device)
            xt = xt.to(self.mean.device).detach()
        elif self.preprocess_mode:
            xp = self._prep(pred)
            xt = self._prep(target).detach()
        else:
            xp = pred
            xt = target.detach()

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
        return torch.stack(losses, dim=0).mean(dim=0)
