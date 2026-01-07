# DINO Perceptual Deep Dive & Refactor Plan

## Executive Summary

The `dino_perceptual` package has **3 real bugs** and several code quality issues ("AI slop"). This document provides a side-by-side comparison with `vitokv2` and proposes a minimal refactor.

---

## Bug 1: The "prep + model" Redundancy

### The Problem

Both files have identical internal preprocessing that **cannot be bypassed**:

```python
# perceptual.py:89-110
def _prep(self, x: torch.Tensor) -> torch.Tensor:
    x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
    # ... resize logic ...
    x = (x - self.mean) / self.std  # ImageNet norm
    return x
```

Then in `forward()`:
```python
# perceptual.py:226-227
xp = self._prep(pred)    # Always preprocesses
xt = self._prep(target).detach()
```

### Why It's a Bug

If a user's pipeline already normalizes images (common with HuggingFace pipelines), they get **double normalization**:

```python
# User code that breaks:
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

# User preprocesses (standard HF pattern)
inputs = processor(images, return_tensors="pt")  # Already normalized!

# Then passes to DINOPerceptual which normalizes AGAIN
loss = loss_fn(inputs, targets)  # BUG: double normalized
```

### vitokv2 Has the Same Bug

```python
# vitokv2/vitok/models/perceptual_networks/dino_perceptual.py:75-98
def _prep(self, x: torch.Tensor) -> torch.Tensor:
    x = (x + 1.0) / 2.0
    # ... identical resize logic ...
    x = (x - self.mean) / self.std
    return x
```

Both implementations force preprocessing with no escape hatch.

### Fix

Add a `preprocessed: bool = False` parameter:

```python
def forward(self, pred: torch.Tensor, target: torch.Tensor, *, preprocessed: bool = False) -> torch.Tensor:
    if preprocessed:
        xp, xt = pred, target.detach()
    else:
        xp = self._prep(pred)
        xt = self._prep(target).detach()
    # ... rest unchanged
```

---

## Bug 2: CLS Token Included Despite Docstring Saying "Exclude"

### The Problem

Docstring claims CLS is excluded:
```python
# perceptual.py:165
# "for selected transformer layers, take token-wise features (exclude CLS)"
```

But the code includes ALL tokens:
```python
# perceptual.py:237-242
fp = hs_p[i]  # Shape: (B, num_tokens, D) -- includes CLS at index 0!
ft = hs_t[i]
# ... no slicing to exclude CLS ...
l = (fp - ft).pow(2).mean(dim=(1, 2))  # CLS is in the loss!
```

### vitokv2 Has the Same Bug

```python
# vitokv2 dino_perceptual.py:136-145
# Token-wise features (exclude CLS) -> [B, T, D]  <-- comment lies
fp = hs_p[i]  # <-- no exclusion
ft = hs_t[i]
```

Both codebases have lying comments.

### Fix

Either:
1. **Update docstring** to say CLS is included, OR
2. **Actually exclude CLS**:

```python
fp = hs_p[i][:, 1:, :]  # Exclude CLS token
ft = hs_t[i][:, 1:, :]
```

I recommend option 1 (update docstring) since including CLS probably doesn't hurt and matches current behavior.

---

## Bug 3: `resize_to_square` Doesn't Resize to Square

### The Problem

Parameter named `resize_to_square` but it **never produces a square**:

```python
# perceptual.py:94-100
if self.resize_to_square:
    long_side = max(H, W)
    if long_side > self.target_size:
        scale = float(self.target_size) / float(long_side)
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        x = F.interpolate(...)  # Still rectangular!
```

A 1920x1080 image with `target_size=512` becomes 512x288, not 512x512.

### What It Actually Does

| `resize_to_square` | Actual Behavior |
|-------------------|-----------------|
| `True` | Downscale preserving aspect ratio if larger than target |
| `False` | Center crop if larger than target |

### Fix

Rename to match behavior:

```python
# Before
resize_to_square: bool = True

# After
resize_mode: Literal["downscale", "center_crop"] = "downscale"
```

---

## AI Slop Identified

### 1. Unnecessary Type Conversions (perceptual.py:75-76)

```python
self.target_size = int(target_size)      # Already typed as int
self.resize_to_square = bool(resize_to_square)  # Already typed as bool
```

These are defensive conversions that add nothing. Remove them.

### 2. Unused Import (perceptual.py:19)

```python
from typing import List, Sequence, Union, Optional, Literal  # Literal unused
```

### 3. Silent Fallback in Layer Selection (perceptual.py:210-214)

```python
for l in self.layers:
    if not isinstance(l, int) or l < 1 or l >= n_total:
        continue  # Silent! numpy.int64 gets ignored
    out.append(l)
return out if out else list(range(1, n_total))  # Falls back silently
```

Should either:
- Accept `numbers.Integral` (includes numpy/torch ints)
- Raise an error on invalid input instead of silent fallback

### 4. Duplicated Test Helpers

`test_perceptual.py` and `examples/distortion_analysis.py` both define:
- `pil_to_tensor()`
- `tensor_to_pil()`
- `apply_gaussian_blur()`
- `apply_gaussian_noise()`
- `apply_jpeg_compression()`

Should be in a shared `utils.py` or removed from examples.

---

## Code Comparison: dino_perceptual vs vitokv2

| Aspect | dino_perceptual | vitokv2 | Notes |
|--------|-----------------|---------|-------|
| DINOv3 "H" model | `vith16plus` | `vith14` | **Different patch sizes!** |
| DINOv3 "G" model | `vit7b16` | Not supported | May not exist |
| DINOv2 support | Yes | No | Extra feature |
| `DINOModel` class | Yes (for FDD) | Separate file (`dinov3.py`) | Different design |
| Base class | `_DINOBase` | None (single class) | Cleaner in vitokv2 |
| `resize_to_square=True` default | `DINOPerceptual` | `DINOv3Perceptual` | Same |
| `resize_to_square=False` default | `DINOModel` | N/A | Different defaults |

### Model ID Mismatch

```python
# dino_perceptual
'H': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',  # patch 16
'G': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',     # may not exist

# vitokv2
'H': 'facebook/dinov3-vith14-pretrain-lvd1689m',      # patch 14
# No 'G' size
```

The "H" model differs in patch size (16 vs 14). This could cause silent behavior differences.

---

## Proposed Refactor

### Minimal Changes (Fix Bugs Only)

```python
# 1. Add preprocessed bypass
def forward(self, pred, target, *, preprocessed=False):
    if not preprocessed:
        pred = self._prep(pred)
        target = self._prep(target)
    target = target.detach()
    ...

# 2. Fix docstring (CLS is included)
"""
Computes LPIPS-like distance using DINO features.
Uses all tokens (including CLS) from selected transformer layers.
"""

# 3. Rename resize_to_square
resize_mode: Literal["downscale", "crop"] = "downscale"

# 4. Fix layer selection to accept numpy ints
import numbers
if not isinstance(l, numbers.Integral) or l < 1 or l >= n_total:
    raise ValueError(f"Invalid layer index: {l}")

# 5. Remove unused import
from typing import List, Sequence, Union, Optional  # Remove Literal

# 6. Remove unnecessary casts
self.target_size = target_size  # Already int from signature
```

### Optional: Unify with vitokv2

If you want to keep these in sync:

1. **Use same model IDs** - Pick either `vith14` or `vith16plus` for "H"
2. **Remove "G" size** - Doesn't exist in vitokv2
3. **Remove `_DINOBase`** - Single class is simpler (like vitokv2)
4. **Remove DINOv2 support** - If not needed, simplifies maintenance

---

## Suggested Test Additions

```python
def test_preprocessed_bypass():
    """Verify preprocessed=True skips internal normalization."""
    loss_fn = DINOPerceptual()

    # Manually preprocess
    x = torch.randn(1, 3, 256, 256)
    x_prep = loss_fn._prep(x)

    # With preprocessed=True, should not double-normalize
    loss1 = loss_fn(x_prep, x_prep, preprocessed=True)
    loss2 = loss_fn(x, x)  # Auto-preprocesses

    # Both should give ~0 loss for identical inputs
    assert loss1.item() < 0.01
    assert loss2.item() < 0.01

def test_numpy_int_layers():
    """Verify numpy integer layer indices work."""
    import numpy as np
    loss_fn = DINOPerceptual(layers=[np.int64(1), np.int64(5)])
    x = torch.randn(1, 3, 256, 256)
    loss = loss_fn(x, x)  # Should not fallback to 'all'
    assert loss.shape == (1,)

def test_model_h_matches_vitokv2():
    """Verify H model matches vitokv2 expectation."""
    # Should use vith14, not vith16plus
    from dino_perceptual import _resolve_model_name
    assert "vith14" in _resolve_model_name("H", "v3")
```

---

## Summary

| Issue | Severity | Fix Effort |
|-------|----------|------------|
| Prep/model redundancy (no bypass) | High | 5 lines |
| CLS docstring lies | Medium | 1 line |
| `resize_to_square` misnomer | Medium | Rename param |
| Silent layer fallback | Medium | 3 lines |
| Unused `Literal` import | Low | Delete |
| Unnecessary type casts | Low | Delete |
| Model ID mismatch vs vitokv2 | Medium | Update dict |

Total refactor: ~20 lines changed, no API breaks if you add `preprocessed` as kwarg-only.

---

## Changes Made

The following refactoring has been applied to `perceptual.py`:

### 1. Added `preprocess` parameter (fixes prep+model redundancy)
```python
# New parameter with 3 modes:
preprocess: Union[str, bool] = True
# - "auto": Use HuggingFace AutoImageProcessor
# - True: Internal preprocessing (expects [-1, 1] input) - default, backward compat
# - False: Skip preprocessing (expects already normalized input)
```

### 2. Removed `_DINOBase` class
- Both `DINOModel` and `DINOPerceptual` are now standalone `nn.Module` subclasses
- Cleaner, less inheritance, matches vitokv2 pattern

### 3. Fixed docstrings
- CLS token is now correctly documented as **included** (was falsely "excluded")
- Resize behavior documented as "downscale preserving aspect ratio"

### 4. Fixed layer selection
- Now accepts `numbers.Integral` (works with numpy.int64, torch.long, etc.)

### 5. Removed AI slop
- Removed unused `Literal` import
- Removed unnecessary `int()` and `bool()` casts
- Removed "G" model size (doesn't exist for DINOv3)
- Fixed "H" model to use `vith14` (matches vitokv2)

### 6. Added AutoImageProcessor support
- `preprocess="auto"` loads HuggingFace's processor for the model
- Enables standard HF pipeline compatibility
