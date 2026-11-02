"""
Architectures: a cross-attention CNN-Transformer hybrid plus matched baselines.

Design rationale for the hybrid (BCAF — Bidirectional Cross-Attention Fusion):
histopathology decisions depend jointly on local cellular morphology (nuclear
pleomorphism, mitotic figures) and on global tissue architecture (glandular
organisation, stromal pattern). A CNN encodes the former with a strong locality
prior; a windowed transformer encodes the latter. Prior hybrids concatenate the
two streams or apply one-directional attention. BCAF lets each stream query the
other, so convolutional features are re-weighted by architectural context and
transformer tokens are re-weighted by cytological evidence, before a gated sum
decides how much each stream contributes per-sample.

Every model exposes the same interface so baselines and hybrid are trained,
calibrated and explained by identical code paths — a precondition for the
statistical comparisons this paper makes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False


# ----------------------------------------------------------------------------
# Backbone construction
# ----------------------------------------------------------------------------
def _is_transformer(name: str) -> bool:
    return any(k in name for k in ("swin", "vit", "deit", "beit", "cait"))


def _make_backbone(name: str, pretrained: bool = True, img_size: int = 224):
    """Return (module, feature_dim, kind) where kind is 'cnn' or 'transformer'.

    Backbones are created with num_classes=0 and global_pool='' so they emit
    spatial feature maps / token sequences rather than pooled logits.

    Windowed transformers (Swin) and patch transformers (ViT) have a fixed
    input resolution baked into their positional embeddings and window
    partitioning. We forward img_size to timm so a non-default resolution is
    handled at construction time, and fail with an actionable message rather
    than an opaque assertion deep inside the forward pass.
    """
    if not _HAS_TIMM:
        raise ImportError("timm is required. pip install timm")

    kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="")
    if _is_transformer(name) and img_size != 224:
        kwargs["img_size"] = img_size
    try:
        m = timm.create_model(name, **kwargs)
    except (TypeError, RuntimeError, AssertionError) as e:
        raise ValueError(
            f"Backbone '{name}' cannot be built at img_size={img_size}. "
            f"Windowed transformers need a resolution divisible by "
            f"patch_size x window_size (e.g. 224 for swin_tiny_patch4_window7). "
            f"Use img_size=224, or choose a CNN backbone. Original error: {e}"
        ) from e

    dim = m.num_features
    return m, dim, ("transformer" if _is_transformer(name) else "cnn")


def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
    """Normalise any backbone output to (B, N, C) token form.

    Handles the three shapes timm backbones emit:
      (B, C, H, W) -> CNN / ConvNeXt feature map
      (B, H, W, C) -> Swin in channels-last layout
      (B, N, C)    -> ViT token sequence
    """
    if feat.dim() == 4:
        b, d1, d2, d3 = feat.shape
        # channels-last if the last dim is the largest and spatial dims match
        if d1 == d2 and d3 != d1:
            return feat.reshape(b, d1 * d2, d3)          # (B,H,W,C)
        return feat.flatten(2).transpose(1, 2)            # (B,C,H,W)
    if feat.dim() == 3:
        return feat
    raise ValueError(f"Unsupported feature shape {tuple(feat.shape)}")


# ----------------------------------------------------------------------------
# Cross-attention fusion
# ----------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """One direction of cross-attention: queries from x, keys/values from y."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)
        self._last_attn = None

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                store_attn: bool = False) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(y)
        out, w = self.attn(q, kv, kv, need_weights=store_attn,
                           average_attn_weights=True)
        if store_attn:
            self._last_attn = w.detach()
        x = x + self.drop(out)
        x = x + self.drop(self.mlp(self.norm_mlp(x)))
        return x

    @property
    def last_attention(self):
        """(B, Nq, Nkv) attention map from the most recent forward pass.

        Used by the XAI module to produce fusion-level explanations that are
        native to the architecture rather than retro-fitted.
        """
        return self._last_attn


class BCAFNet(nn.Module):
    """Bidirectional Cross-Attention Fusion network.

    Args:
        cnn_name: timm CNN backbone (local/cytological stream).
        vit_name: timm transformer backbone (global/architectural stream).
        num_classes: 2 for benign/malignant, 8 for subtype.
        fusion_dim: shared projection width for the two streams.
    """

    def __init__(
        self,
        cnn_name: str = "convnext_tiny",
        vit_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 2,
        fusion_dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.img_size = img_size
        self.cnn, cnn_dim, _ = _make_backbone(cnn_name, pretrained, img_size)
        self.vit, vit_dim, _ = _make_backbone(vit_name, pretrained, img_size)

        self.proj_cnn = nn.Linear(cnn_dim, fusion_dim)
        self.proj_vit = nn.Linear(vit_dim, fusion_dim)

        self.c2v = CrossAttentionBlock(fusion_dim, num_heads, dropout=dropout)
        self.v2c = CrossAttentionBlock(fusion_dim, num_heads, dropout=dropout)

        # Per-sample gate: learns how much each stream should contribute.
        # Reported in the paper as evidence of what the fusion actually does.
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2), nn.GELU(),
            nn.Linear(fusion_dim // 2, 2),
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes)
        )
        self._last_gate = None

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"BCAFNet was built for {self.img_size}x{self.img_size} inputs "
                f"but received {tuple(x.shape[-2:])}. Rebuild the model with "
                f"img_size={x.shape[-1]}, or set TrainConfig.img_size="
                f"{self.img_size}."
            )
        tc = self.proj_cnn(_to_tokens(self.cnn(x)))   # (B, Nc, D)
        tv = self.proj_vit(_to_tokens(self.vit(x)))   # (B, Nv, D)

        fc = self.v2c(tc, tv, store_attn)   # conv tokens attend to global context
        fv = self.c2v(tv, tc, store_attn)   # global tokens attend to local evidence

        pc, pv = fc.mean(1), fv.mean(1)
        w = F.softmax(self.gate(torch.cat([pc, pv], dim=-1)), dim=-1)
        self._last_gate = w.detach()
        fused = w[:, 0:1] * pc + w[:, 1:2] * pv
        return self.head(self.norm(fused))

    @property
    def last_gate(self):
        """(B, 2) softmax weights over [cnn_stream, transformer_stream]."""
        return self._last_gate


class SingleBackboneNet(nn.Module):
    """Baseline wrapper giving any timm backbone the same interface as BCAFNet."""

    def __init__(self, name: str, num_classes: int = 2,
                 dropout: float = 0.1, pretrained: bool = True,
                 img_size: int | None = None):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required. pip install timm")
        kw = dict(pretrained=pretrained, num_classes=0)
        if img_size is not None and _is_transformer(name) and img_size != 224:
            kw["img_size"] = img_size
        self.backbone = timm.create_model(name, **kw)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.backbone.num_features, num_classes)
        )
        self.name = name

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        return self.head(self.backbone(x))


class ConcatFusionNet(nn.Module):
    """Ablation: same two streams, naive concatenation instead of cross-attention.

    This is the control that isolates whether cross-attention contributes
    anything beyond simply having two backbones — the ablation most prior
    hybrid papers omit, and the one reviewers ask for.
    """

    def __init__(self, cnn_name: str = "convnext_tiny",
                 vit_name: str = "swin_tiny_patch4_window7_224",
                 num_classes: int = 2, dropout: float = 0.1,
                 pretrained: bool = True, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.cnn, cd, _ = _make_backbone(cnn_name, pretrained, img_size)
        self.vit, vd, _ = _make_backbone(vit_name, pretrained, img_size)
        self.head = nn.Sequential(
            nn.LayerNorm(cd + vd), nn.Dropout(dropout),
            nn.Linear(cd + vd, num_classes),
        )

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        c = _to_tokens(self.cnn(x)).mean(1)
        v = _to_tokens(self.vit(x)).mean(1)
        return self.head(torch.cat([c, v], dim=-1))


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bcaf":            lambda nc, pt, sz: BCAFNet(num_classes=nc, pretrained=pt, img_size=sz),
    "concat_ablation": lambda nc, pt, sz: ConcatFusionNet(num_classes=nc, pretrained=pt, img_size=sz),
    "resnet50":        lambda nc, pt, sz: SingleBackboneNet("resnet50", nc, pretrained=pt),
    "densenet121":     lambda nc, pt, sz: SingleBackboneNet("densenet121", nc, pretrained=pt),
    "efficientnet_b0": lambda nc, pt, sz: SingleBackboneNet("efficientnet_b0", nc, pretrained=pt),
    "convnext_tiny":   lambda nc, pt, sz: SingleBackboneNet("convnext_tiny", nc, pretrained=pt),
    "swin_tiny":       lambda nc, pt, sz: SingleBackboneNet("swin_tiny_patch4_window7_224", nc, pretrained=pt, img_size=sz),
    "vit_small":       lambda nc, pt, sz: SingleBackboneNet("vit_small_patch16_224", nc, pretrained=pt, img_size=sz),
}

#: Models whose input resolution is fixed by construction.
FIXED_RESOLUTION_MODELS = ("bcaf", "concat_ablation", "swin_tiny", "vit_small")


def build_model(key: str, num_classes: int = 2, pretrained: bool = True,
                img_size: int = 224) -> nn.Module:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](num_classes, pretrained, img_size)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}
