"""
Publication figure generation.

Schematic figures (study design, architecture, splitting protocol) are drawn
here from code so they are reproducible, vector-based and editable, rather than
redrawn by hand in a slide tool for every revision. IEEE journals want vector
PDF at column width; every figure is emitted as both PDF and 600-dpi PNG.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# IEEE single column is 3.5 in; double column 7.16 in.
COL, DCOL = 3.5, 7.16

PALETTE = {
    "cnn":    "#2E6E8E",
    "vit":    "#B5651D",
    "fusion": "#4A7C59",
    "data":   "#5B5B7B",
    "eval":   "#8C4356",
    "light":  "#F2F0EB",
    "ink":    "#22252A",
    "muted":  "#8A8F98",
}


def _setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "pdf.fonttype": 42,      # embed as TrueType — required by IEEE
        "ps.fonttype": 42,
    })


def _save(fig, path_stem: str):
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path_stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    print(f"  wrote {path_stem}.pdf / .png")


def _box(ax, x, y, w, h, label, color, fontsize=7, text_color="white", alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8, edgecolor=color, facecolor=color, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=3, linespacing=1.35)


def _arrow(ax, p0, p1, color=None, style="-|>", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=8, linewidth=lw,
        color=color or PALETTE["ink"], zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------------
# Figure 1 — study design
# ---------------------------------------------------------------------------
def fig_study_design(out="figures/fig1_study_design"):
    _setup()
    fig, ax = plt.subplots(figsize=(DCOL, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")

    _box(ax, 0.05, 3.4, 1.7, 0.85,
         "BreakHis\n7,909 images\n82 patients\n4 magnifications",
         PALETTE["data"], 6.5)
    _box(ax, 0.05, 2.2, 1.7, 0.85,
         "Integrity check\nvs published\ncounts", PALETTE["data"], 6.5, alpha=0.78)

    _box(ax, 2.15, 3.4, 2.0, 0.85,
         "Patient-wise\nStratifiedGroupKFold\n(k = 5)", PALETTE["eval"], 6.5)
    _box(ax, 2.15, 2.2, 2.0, 0.85,
         "Image-wise split\n(leaky control)", PALETTE["muted"], 6.5)

    _box(ax, 4.55, 3.4, 1.9, 0.85,
         "BCAF hybrid\n+ ablation\n+ 4 baselines", PALETTE["fusion"], 6.5)
    _box(ax, 4.55, 2.2, 1.9, 0.85,
         "Identical training\nrecipe for all", PALETTE["fusion"], 6.5, alpha=0.72)

    for i, (lbl, yy) in enumerate([
        ("Discrimination\nacc / F1 / AUC / MCC", 3.55),
        ("Calibration\nECE / ACE / Brier", 2.60),
        ("Faithfulness\ndel-ins / AOPC / ADCC", 1.65),
    ]):
        _box(ax, 6.85, yy, 1.75, 0.72, lbl, PALETTE["cnn"], 6.3,
             alpha=1 - 0.12 * i)

    _box(ax, 8.95, 2.60, 1.0, 0.72,
         "Statistical\ntesting\n+ TOST", PALETTE["vit"], 6.3)

    _arrow(ax, (1.75, 3.82), (2.15, 3.82))
    _arrow(ax, (1.75, 2.62), (2.15, 2.62))
    _arrow(ax, (0.9, 3.4), (0.9, 3.05))
    _arrow(ax, (4.15, 3.82), (4.55, 3.82))
    _arrow(ax, (4.15, 2.62), (4.55, 2.62))
    for yy in (3.91, 2.96, 2.01):
        _arrow(ax, (6.45, 3.6), (6.85, yy + 0.36), rad=0.12)
    for yy in (3.91, 2.96, 2.01):
        _arrow(ax, (8.6, yy), (8.95, 2.96), rad=-0.1)

    ax.text(0.05, 1.55, "Leakage guard: every fold assignment is re-verified before each fit;\n"
                        "a patient appearing in two folds raises rather than degrading silently.",
            fontsize=6.2, color=PALETTE["ink"], style="italic", va="top")
    ax.text(0.05, 0.75, "Novel relative to prior BreakHis work: calibration reporting, "
                        "quantified explanation faithfulness with a random-saliency control,\n"
                        "equivalence testing, and a like-for-like measurement of the "
                        "leakage-induced inflation.",
            fontsize=6.2, color=PALETTE["eval"], va="top")
    _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 3 — BCAF architecture
# ---------------------------------------------------------------------------
def fig_architecture(out="figures/fig3_architecture"):
    _setup()
    fig, ax = plt.subplots(figsize=(DCOL, 2.9))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.2); ax.axis("off")

    _box(ax, 0.05, 1.75, 0.95, 0.75, "H&E\npatch\n224²", PALETTE["data"], 6.4)

    _box(ax, 1.30, 2.75, 1.78, 0.72, "ConvNeXt-T\nlocal (cytology)", PALETTE["cnn"], 6.4)
    _box(ax, 1.30, 0.85, 1.78, 0.72, "Swin-T\nglobal (architecture)", PALETTE["vit"], 6.4)

    _box(ax, 3.25, 2.75, 0.85, 0.72, "linear\nproj", PALETTE["cnn"], 6.2, alpha=0.75)
    _box(ax, 3.25, 0.85, 0.85, 0.72, "linear\nproj", PALETTE["vit"], 6.2, alpha=0.75)

    _box(ax, 4.45, 2.75, 1.75, 0.72,
         "cross-attn\nQ: conv  KV: swin", PALETTE["fusion"], 6.2)
    _box(ax, 4.45, 0.85, 1.75, 0.72,
         "cross-attn\nQ: swin  KV: conv", PALETTE["fusion"], 6.2)

    _box(ax, 6.6, 1.75, 1.15, 0.85, "gated\nsum\nw ∈ ℝ²", PALETTE["fusion"], 6.4, alpha=0.9)
    _box(ax, 8.15, 1.75, 1.0, 0.85, "LN +\nlinear", PALETTE["ink"], 6.4)
    _box(ax, 9.5, 1.9, 0.8, 0.55, "logits", PALETTE["eval"], 6.4)

    _arrow(ax, (1.0, 2.3), (1.30, 3.11), rad=0.15)
    _arrow(ax, (1.0, 1.95), (1.30, 1.21), rad=-0.15)
    _arrow(ax, (3.08, 3.11), (3.25, 3.11))
    _arrow(ax, (3.08, 1.21), (3.25, 1.21))
    _arrow(ax, (4.1, 3.11), (4.45, 3.11))
    _arrow(ax, (4.1, 1.21), (4.45, 1.21))
    # the bidirectional exchange
    _arrow(ax, (4.35, 2.85), (4.35, 1.5), color=PALETTE["muted"], rad=-0.35, style="<|-|>")
    _arrow(ax, (6.2, 3.11), (6.6, 2.4), rad=0.12)
    _arrow(ax, (6.2, 1.21), (6.6, 1.95), rad=-0.12)
    _arrow(ax, (7.75, 2.17), (8.15, 2.17))
    _arrow(ax, (9.15, 2.17), (9.5, 2.17))

    ax.text(3.6, 2.15, "bidirectional\nexchange", fontsize=5.8,
            color=PALETTE["muted"], ha="center", style="italic")
    ax.text(0.05, 0.35,
            "The gate weights w are recorded per sample, so the paper can report "
            "how much each stream actually contributed —\nevidence about what the "
            "fusion does, not merely that it exists. Cross-attention maps double as "
            "architecture-native explanations.",
            fontsize=6.1, color=PALETTE["ink"], va="top")
    _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 4 — why patient-wise splitting matters
# ---------------------------------------------------------------------------
def fig_splitting(out="figures/fig4_splitting"):
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
    rng = np.random.default_rng(3)
    n_pat, per = 12, 6

    for ax, mode in zip(axes, ["image-wise (leaky)", "patient-wise (correct)"]):
        ax.set_xlim(-0.5, per + 0.4); ax.set_ylim(-2.4, n_pat + 0.2)
        ax.axis("off")
        for p in range(n_pat):
            if mode.startswith("image"):
                assign = rng.integers(0, 2, per)          # split within a patient
            else:
                assign = np.full(per, 0 if p < 8 else 1)  # whole patient moves
            for i, a in enumerate(assign):
                ax.add_patch(Rectangle(
                    (i, n_pat - p - 1), 0.86, 0.72,
                    facecolor=PALETTE["cnn"] if a == 0 else PALETTE["eval"],
                    edgecolor="white", linewidth=0.5))
            ax.text(-0.28, n_pat - p - 0.64, f"P{p+1}", fontsize=5.2,
                    ha="right", va="center", color=PALETTE["ink"])
        ax.set_title(mode, fontsize=8, pad=6)
        if mode.startswith("image"):
            ax.text(per / 2, -1.35,
                    "same patient in train AND test\n→ inflated accuracy",
                    fontsize=6, ha="center", va="center", color=PALETTE["eval"])
        else:
            ax.text(per / 2, -1.35,
                    "each patient in exactly one split\n→ generalisation to unseen patients",
                    fontsize=6, ha="center", va="center", color=PALETTE["fusion"])

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], marker="s", color="w", label="train",
               markerfacecolor=PALETTE["cnn"], markersize=6),
        Line2D([0], [0], marker="s", color="w", label="test",
               markerfacecolor=PALETTE["eval"], markersize=6)],
        loc="lower center", ncol=2, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.09, "Rows are patients; cells are images from that patient.",
             ha="center", fontsize=6, style="italic", color=PALETTE["muted"])
    _save(fig, out)


def build_all(out_dir="figures"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    print("Generating schematic figures:")
    fig_study_design(f"{out_dir}/fig1_study_design")
    fig_architecture(f"{out_dir}/fig3_architecture")
    fig_splitting(f"{out_dir}/fig4_splitting")


if __name__ == "__main__":
    import sys
    build_all(sys.argv[1] if len(sys.argv) > 1 else "figures")
