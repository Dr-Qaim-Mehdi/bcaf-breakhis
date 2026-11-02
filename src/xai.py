"""
Explanation generation and — the part the BreakHis literature is missing —
quantitative evaluation of whether those explanations are faithful.

Published BreakHis work shows Grad-CAM heatmaps as qualitative figures. A
heatmap that looks plausible to a reader is not evidence that the model used
that region. This module measures it:

  Deletion AUC   (lower better)  progressively erase the highest-attributed
                                 pixels; a faithful map makes confidence fall fast
  Insertion AUC  (higher better) reveal them onto a blurred canvas; a faithful
                                 map restores confidence fast
  AOPC           (higher better) mean confidence drop over the deletion curve
  Average Drop   (lower better)  confidence lost when only the map is kept
  Coherency      (higher better) agreement between the map and the map-of-the-map
  Complexity     (lower better)  L1 mass of the map — diffuse maps explain little
  ADCC           (higher better) harmonic mean of the three above

References: Petsiuk et al. 2018 (deletion/insertion); Samek et al. 2017 (AOPC);
Poppi et al. 2021 (ADCC).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Attribution methods
# ============================================================================
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) with an explicit target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, out):
        self.acts = out[0] if isinstance(out, tuple) else out

    def _save_grad(self, _m, _gi, gout):
        self.grads = gout[0]

    @staticmethod
    def _as_bchw(t: torch.Tensor) -> torch.Tensor:
        """Coerce (B,H,W,C) or (B,N,C) feature maps to (B,C,H,W)."""
        if t.dim() == 4:
            b, d1, d2, d3 = t.shape
            if d1 == d2 and d3 != d1:            # channels-last
                return t.permute(0, 3, 1, 2).contiguous()
            return t
        if t.dim() == 3:                          # token sequence
            b, n, c = t.shape
            s = int(round(n ** 0.5))
            if s * s != n:                        # drop a class token if present
                s = int(round((n - 1) ** 0.5))
                t = t[:, 1:, :]
            return t.transpose(1, 2).reshape(b, c, s, s)
        raise ValueError(f"Cannot interpret feature shape {tuple(t.shape)}")

    def __call__(self, x: torch.Tensor, class_idx=None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=x.device)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        acts = self._as_bchw(self.acts)
        grads = self._as_bchw(self.grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        return _normalise(cam.squeeze(1).detach())

    def close(self):
        self._h1.remove()
        self._h2.remove()


class GradCAMpp(GradCAM):
    """Grad-CAM++ (Chattopadhay et al., 2018) — better for multiple lesion foci."""

    def __call__(self, x: torch.Tensor, class_idx=None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=x.device)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        acts = self._as_bchw(self.acts)
        grads = self._as_bchw(self.grads)
        g2, g3 = grads ** 2, grads ** 3
        denom = 2 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        return _normalise(cam.squeeze(1).detach())


def fusion_attention_map(model, x: torch.Tensor) -> np.ndarray:
    """Architecture-native explanation for BCAFNet.

    Reads the bidirectional cross-attention weights directly, so the map is what
    the fusion actually computed rather than a gradient approximation of it.
    """
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _ = model(x, store_attn=True)
    a_v2c = model.v2c.last_attention   # (B, Nc, Nv)
    a_c2v = model.c2v.last_attention   # (B, Nv, Nc)
    if a_v2c is None or a_c2v is None:
        raise RuntimeError("No attention captured — call with store_attn=True.")

    # How much each conv token is attended to, from both directions.
    imp = a_v2c.mean(dim=2) + a_c2v.mean(dim=1)     # (B, Nc)
    b, n = imp.shape
    s = int(round(n ** 0.5))
    cam = imp.reshape(b, 1, s, s)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                        align_corners=False)
    return _normalise(cam.squeeze(1).detach())


def _normalise(cam: torch.Tensor) -> np.ndarray:
    """Per-sample min-max to [0,1]."""
    b = cam.shape[0]
    flat = cam.reshape(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-8)
    return flat.reshape(cam.shape).cpu().numpy()


# ============================================================================
# Faithfulness metrics
# ============================================================================
@torch.no_grad()
def deletion_insertion_auc(
    model, x: torch.Tensor, saliency: np.ndarray, class_idx: torch.Tensor,
    n_steps: int = 50, blur_sigma: float = 11.0,
) -> dict:
    """Petsiuk et al. (2018) deletion and insertion curves.

    Deletion removes top-attributed pixels to a blurred baseline; insertion
    reveals them onto that baseline. Reports normalised AUCs and their
    difference, which is the single most informative faithfulness scalar.
    """
    model.eval()
    device = x.device
    b, _, h, w = x.shape
    npix = h * w
    baseline = _gaussian_blur(x, blur_sigma)

    order = torch.tensor(
        saliency.reshape(b, -1).argsort(axis=1)[:, ::-1].copy(), device=device
    )
    step = max(1, npix // n_steps)

    del_scores, ins_scores = [], []
    x_del, x_ins = x.clone(), baseline.clone()
    for k in range(0, npix + step, step):
        p_del = F.softmax(model(x_del), dim=1).gather(1, class_idx.view(-1, 1))
        p_ins = F.softmax(model(x_ins), dim=1).gather(1, class_idx.view(-1, 1))
        del_scores.append(p_del.squeeze(1).cpu().numpy())
        ins_scores.append(p_ins.squeeze(1).cpu().numpy())
        if k >= npix:
            break
        idx = order[:, k:k + step]
        for i in range(b):
            rows, cols = idx[i] // w, idx[i] % w
            x_del[i, :, rows, cols] = baseline[i, :, rows, cols]
            x_ins[i, :, rows, cols] = x[i, :, rows, cols]

    dele = np.stack(del_scores, axis=1)   # (B, T)
    ins = np.stack(ins_scores, axis=1)
    return {
        "deletion_auc": _auc(dele),
        "insertion_auc": _auc(ins),
        "faithfulness_gap": _auc(ins) - _auc(dele),
        "deletion_curve": dele.mean(axis=0),
        "insertion_curve": ins.mean(axis=0),
    }


@torch.no_grad()
def aopc(model, x: torch.Tensor, saliency: np.ndarray,
         class_idx: torch.Tensor, n_steps: int = 20) -> np.ndarray:
    """Area Over the Perturbation Curve (Samek et al., 2017).

    Mean confidence drop as top-attributed regions are progressively removed.
    Higher = the map identified genuinely decision-relevant evidence.
    """
    model.eval()
    b, _, h, w = x.shape
    npix = h * w
    baseline = _gaussian_blur(x, 11.0)
    order = torch.tensor(
        saliency.reshape(b, -1).argsort(axis=1)[:, ::-1].copy(), device=x.device
    )
    p0 = F.softmax(model(x), dim=1).gather(1, class_idx.view(-1, 1)).squeeze(1)
    step = max(1, npix // n_steps)

    drops = []
    xp = x.clone()
    for k in range(0, npix, step):
        idx = order[:, k:k + step]
        for i in range(b):
            rows, cols = idx[i] // w, idx[i] % w
            xp[i, :, rows, cols] = baseline[i, :, rows, cols]
        pk = F.softmax(model(xp), dim=1).gather(1, class_idx.view(-1, 1)).squeeze(1)
        drops.append((p0 - pk).cpu().numpy())
    return np.stack(drops, axis=1).mean(axis=1)


@torch.no_grad()
def adcc(model, x: torch.Tensor, saliency: np.ndarray,
         class_idx: torch.Tensor, cam_fn=None) -> dict:
    """Average Drop / Coherency / Complexity and their harmonic mean (ADCC).

    Poppi et al. (2021). ADCC resists the failure modes of each component:
    a map covering the whole image scores a perfect Average Drop but a terrible
    Complexity, so it cannot win on ADCC.

    cam_fn is REQUIRED for a usable result. Coherency compares the attribution
    of the image against the attribution of the masked image, so the callable
    must regenerate the same attribution method. Without it, coherency and ADCC
    are NaN and the metric is meaningless.
    """
    model.eval()
    b = x.shape[0]
    sal = torch.tensor(saliency, device=x.device, dtype=x.dtype).unsqueeze(1)

    p_full = F.softmax(model(x), dim=1).gather(1, class_idx.view(-1, 1)).squeeze(1)
    p_mask = F.softmax(model(x * sal), dim=1).gather(1, class_idx.view(-1, 1)).squeeze(1)

    avg_drop = torch.clamp(p_full - p_mask, min=0) / (p_full + 1e-8)
    avg_drop = avg_drop.cpu().numpy()

    complexity = sal.reshape(b, -1).abs().mean(dim=1).cpu().numpy()

    if cam_fn is not None:
        # Gradient-based attribution methods need autograd, and this function
        # runs under no_grad. Re-enable it locally, on a detached copy, so the
        # coherency term can actually be computed.
        with torch.enable_grad():
            masked = (x * sal).detach().requires_grad_(True)
            sal2 = cam_fn(masked, class_idx)
        a = saliency.reshape(b, -1)
        c = np.asarray(sal2).reshape(b, -1)
        coherency = np.array([_pearson(a[i], c[i]) for i in range(b)])
        coherency = (coherency + 1.0) / 2.0     # map [-1,1] -> [0,1]
    else:
        import warnings
        warnings.warn(
            "adcc() called without cam_fn: coherency, and therefore ADCC "
            "itself, will be NaN. Pass a callable (x, class_idx) -> saliency "
            "that regenerates the SAME attribution method, or the ADCC column "
            "will be empty in your results table.",
            RuntimeWarning, stacklevel=2,
        )
        coherency = np.full(b, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.stack([coherency, 1 - complexity, 1 - avg_drop])
        adcc_val = 3.0 / np.sum(1.0 / np.clip(terms, 1e-8, None), axis=0)

    return {
        "average_drop": avg_drop,
        "coherency": coherency,
        "complexity": complexity,
        "adcc": adcc_val,
    }


# ============================================================================
# helpers
# ============================================================================
def _auc(curve: np.ndarray) -> np.ndarray:
    """Normalised area under a per-sample curve of shape (B, T)."""
    return np.trapezoid(curve, axis=1) / (curve.shape[1] - 1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blurred baseline — a more honest reference than zeros, which are
    themselves out of distribution and inflate apparent faithfulness."""
    k = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    out = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"),
                   g.expand(c, 1, 1, k), groups=c)
    out = F.conv2d(F.pad(out, (0, 0, k // 2, k // 2), mode="reflect"),
                   g.transpose(2, 3).expand(c, 1, k, 1), groups=c)
    return out


def get_target_layer(model, model_key: str):
    """Best-practice Grad-CAM target layer per architecture."""
    if model_key == "bcaf":
        return model.cnn.stages[-1]
    bb = getattr(model, "backbone", model)
    for attr in ("layer4", "features", "stages", "blocks"):
        if hasattr(bb, attr):
            mod = getattr(bb, attr)
            try:
                return mod[-1]
            except (TypeError, IndexError):
                return mod
    raise ValueError(f"Could not resolve a target layer for '{model_key}'.")
