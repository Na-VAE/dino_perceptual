"""DINOv3-based perceptual loss (LPIPS-like).

Computes an LPIPS-like distance using frozen DINOv3 ViT features:
for selected transformer layers, take token-wise features (exclude CLS),
L2-normalize per token, compute squared differences between real and fake
feature maps, and average only at the very end to a per-image scalar.

This mimics LPIPS structure by avoiding early pooling (no CLS/mean pooling),
preserving spatial/token structure until the final MSE reduction.

Usage:
    loss_fn = DINOv3Perceptual(
        model_size="B",  # or "S", "L", "H"
        target_size=512,
        layers="all",
    )
    loss_vec = loss_fn(pred_images, ref_images)  # shape [B]
    loss = loss_vec.mean()
"""

from typing import List, Sequence, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def _resolve_dinov3_model_name(model_size: str) -> str:
    """Map a size key to a default DINOv3 HF model name.

    Supported keys (case-insensitive):
      - 'S' -> ViT-S/16
      - 'B' -> ViT-B/16
      - 'L' -> ViT-L/16
      - 'H' -> ViT-H/14
    """
    key = str(model_size).strip().upper()
    mapping = {
        'S': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'B': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'L': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'H': 'facebook/dinov3-vith14-pretrain-lvd1689m',
    }
    return mapping.get(key, mapping['B'])


class DINOv3Perceptual(nn.Module):
    """DINOv3-based perceptual loss function.

    Args:
        model_name: HuggingFace model name. If None, uses model_size to select.
        model_size: Model size key ('S', 'B', 'L', 'H'). Default 'B'.
        target_size: Maximum image size. Larger images are downscaled.
        layers: Which layers to use. 'all' or list of 1-based indices.
        normalize: Whether to L2-normalize features per token.
        resize_to_square: If True, resize preserving aspect ratio. If False, center crop.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_size: str = "B",
        target_size: int = 512,
        layers: Union[str, Sequence[int]] = "all",
        normalize: bool = True,
        resize_to_square: bool = True,
    ):
        super().__init__()
        resolved_name = model_name or _resolve_dinov3_model_name(model_size)
        self.model = AutoModel.from_pretrained(resolved_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Preprocessing
        self.target_size = int(target_size)
        self.resize_to_square = bool(resize_to_square)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.layers = layers
        self.normalize_feats = bool(normalize)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images: normalize and optionally resize."""
        # Expect x in [-1, 1]; map to [0, 1]
        x = (x + 1.0) / 2.0
        # Downscale-only resize: if larger than target, shrink while preserving aspect ratio.
        if self.resize_to_square:
            B, C, H, W = x.shape
            long_side = max(H, W)
            if long_side > self.target_size:
                scale = float(self.target_size) / float(long_side)
                new_h = max(1, int(round(H * scale)))
                new_w = max(1, int(round(W * scale)))
                x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        else:
            B, C, H, W = x.shape
            if H > self.target_size or W > self.target_size:
                crop_h = min(self.target_size, H)
                crop_w = min(self.target_size, W)
                h_start = (H - crop_h) // 2
                w_start = (W - crop_w) // 2
                x = x[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]
        # ImageNet norm
        x = (x - self.mean) / self.std
        return x

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)
        return x / denom

    def _select_layers(self, hidden_states: List[torch.Tensor]) -> List[int]:
        """Select which hidden state layers to use."""
        n_total = len(hidden_states)
        # We exclude the embedding output (index 0) by default for 'all'
        if isinstance(self.layers, str) and self.layers == 'all':
            return list(range(1, n_total))
        # Interpret provided as 1-based layer indices
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
            # LPIPS-like: squared difference, pool at end
            l = (fp - ft).pow(2).mean(dim=(1, 2))
            losses.append(l)

        if len(losses) == 0:
            return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec
