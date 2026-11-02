#!/usr/bin/env python3
"""Publication figures for the full sweep. Vector PDF + 600 dpi PNG."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import os, re, math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES = _os.path.join(_ROOT, "results", "sweep")
OUT = _os.path.join(_ROOT, "results", "derived")
FIG = _os.path.join(_ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
COL, DCOL = 3.5, 7.16

MODELS = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny", "resnet50"]
MAGS = [40, 100, 200, 400]
P = {"bcaf": "#B5651D", "concat_ablation": "#C99A2E", "convnext_tiny": "#2E6E8E",
     "swin_tiny": "#4A7C59", "resnet50": "#8A8F98",
     "ink": "#22252A", "muted": "#8A8F98", "accent": "#8C4356",
     "ok": "#4A7C59", "light": "#F2F0EB"}
NAME = {"bcaf": "BCAF (58.5M)", "concat_ablation": "Concat fusion (55.4M)",
        "convnext_tiny": "ConvNeXt-Tiny (27.8M)", "swin_tiny": "Swin-Tiny (27.5M)",
        "resnet50": "ResNet-50 (23.5M)"}
SHORT = {"bcaf": "BCAF", "concat_ablation": "Concat", "convnext_tiny": "ConvNeXt-T",
         "swin_tiny": "Swin-T", "resnet50": "ResNet-50"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.7, "savefig.dpi": 600, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    "xtick.direction": "out", "ytick.direction": "out",
})


def save(fig, stem):
    fig.savefig(f"{FIG}/{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIG}/{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    print("wrote", stem)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)


def _ord(v):
    n = int(round(v))
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def subtype_from_path(p):
    m = re.search(r"/(benign|malignant)/SOB/([a-z_]+)/", p)
    return m.group(2) if m else "unknown"


# ---------------------------------------------------------------- load
fm = {(m, g): pd.read_csv(f"{RES}/fold_metrics_{m}_m{g}.csv") for m in MODELS for g in MAGS}
oof40 = {}
for m in MODELS:
    d = pd.read_csv(f"{RES}/oof_{m}_m40.csv")
    d["conf"] = d[["prob_0", "prob_1"]].max(axis=1)
    d["subtype"] = d["path"].map(subtype_from_path)
    oof40[m] = d
leak = pd.read_csv(f"{RES}/leakage_by_magnification.csv")
calib = pd.read_csv(f"{RES}/calibration_per_fold.csv")
xai = pd.read_csv(f"{RES}/xai_faithfulness_allfolds.csv")
power = pd.read_csv(f"{OUT}/power_all.csv")
rc = pd.read_csv(f"{OUT}/risk_coverage.csv")
sel = pd.read_csv(f"{OUT}/selective_prediction.csv")


# ============================================================ 1. family x magnification
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.6))
ax = axes[0]
xs = np.arange(len(MAGS))
for i, m in enumerate(MODELS):
    mu = [fm[(m, g)]["img_balanced_accuracy"].mean() * 100 for g in MAGS]
    sd = [fm[(m, g)]["img_balanced_accuracy"].std(ddof=1) * 100 for g in MAGS]
    off = (i - 2) * 0.09
    ax.errorbar(xs + off, mu, yerr=sd, color=P[m], lw=1.3, marker="o", ms=3.2,
                capsize=1.8, elinewidth=0.7, label=NAME[m])
ax.set_xticks(xs)
ax.set_xticklabels([f"{g}$\\times$" for g in MAGS])
ax.set_xlabel("magnification")
ax.set_ylabel("balanced accuracy (%)")
ax.set_ylim(72, 100)
ax.legend(fontsize=6.0, loc="lower left", ncol=1)
ax.set_title("(a)  Five architectures, four magnifications", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
rng = np.random.default_rng(3)
for i, m in enumerate(MODELS):
    for j, g in enumerate(MAGS):
        v = fm[(m, g)]["img_balanced_accuracy"].to_numpy() * 100
        jit = (rng.random(len(v)) - 0.5) * 0.30
        ax.scatter(np.full(len(v), i) + (j - 1.5) * 0.17 + jit, v, s=6,
                   color=P[m], alpha=0.55, linewidths=0)
    allv = np.concatenate([fm[(m, g)]["img_balanced_accuracy"].to_numpy() for g in MAGS]) * 100
    ax.plot([i - 0.42, i + 0.42], [allv.mean()] * 2, color=P["ink"], lw=1.2)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=6.8)
ax.set_ylabel("fold balanced accuracy (%)")
ax.set_ylim(60, 102)
ax.set_title("(b)  Every fold, every magnification", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_family")


# ============================================================ 2. leakage by magnification
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
ax = axes[0]
w = 0.36
pw = leak[leak.protocol.str.startswith("patient")].set_index("magnification")
iw = leak[leak.protocol.str.startswith("image")].set_index("magnification")
xs = np.arange(len(MAGS))
ax.bar(xs - w / 2, [pw.loc[g, "accuracy"] * 100 for g in MAGS], width=w,
       yerr=[pw.loc[g, "accuracy_std"] * 100 for g in MAGS], capsize=2,
       error_kw=dict(lw=0.7), color=P["convnext_tiny"], label="patient-wise")
ax.bar(xs + w / 2, [iw.loc[g, "accuracy"] * 100 for g in MAGS], width=w,
       yerr=[iw.loc[g, "accuracy_std"] * 100 for g in MAGS], capsize=2,
       error_kw=dict(lw=0.7), color=P["accent"], label="image-wise (leaky)")
for i, g in enumerate(MAGS):
    ax.annotate(f"+{iw.loc[g,'inflation_pp']:.2f} pp", xy=(i, 101.4), ha="center",
                fontsize=6.5, color=P["accent"])
ax.set_xticks(xs)
ax.set_xticklabels([f"{g}$\\times$" for g in MAGS])
ax.set_xlabel("magnification")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(70, 116)
ax.set_yticks([70, 80, 90, 100])
ax.legend(fontsize=6.5, loc="upper left", ncol=2)
ax.set_title("(a)  Leakage inflates at every magnification", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
ax.bar(xs - w / 2, [pw.loc[g, "accuracy_std"] * 100 for g in MAGS], width=w,
       color=P["convnext_tiny"], label="patient-wise")
ax.bar(xs + w / 2, [iw.loc[g, "accuracy_std"] * 100 for g in MAGS], width=w,
       color=P["accent"], label="image-wise (leaky)")
for i, g in enumerate(MAGS):
    ax.annotate(f"{iw.loc[g,'sd_ratio']:.1f}$\\times$\nnarrower",
                xy=(i - 0.18, pw.loc[g, "accuracy_std"] * 100 + 0.25), ha="center",
                va="bottom", fontsize=6.5, color=P["ink"])
ax.set_xticks(xs)
ax.set_xticklabels([f"{g}$\\times$" for g in MAGS])
ax.set_xlabel("magnification")
ax.set_ylabel("fold s.d. of accuracy (pp)")
ax.set_ylim(0, 9.4)
ax.legend(fontsize=6.5, loc="upper right", ncol=1)
ax.set_title("(b)  and suppresses the reported dispersion", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_leakage")


# ============================================================ 3. ablation verdict
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
ax = axes[0]
pa = power[(power.model_a == "bcaf") & (power.model_b == "concat_ablation")].set_index("magnification")
d = np.array([pa.loc[g, "mean_diff"] * 100 for g in MAGS])
mde = np.array([pa.loc[g, "mde_nb_pp"] for g in MAGS])
ax.fill_between(xs, -mde, mde, color=P["muted"], alpha=0.20, lw=0,
                label="undetectable at 80% power")
ax.axhline(0, color=P["ink"], lw=0.8)
ax.plot(xs, d, "o-", color=P["bcaf"], lw=1.4, ms=4.5, label="BCAF $-$ concat fusion")
for i, g in enumerate(MAGS):
    ax.annotate(f"{d[i]:+.2f}", xy=(i, d[i]), xytext=(0, 6 if d[i] >= 0 else -12),
                textcoords="offset points", ha="center", fontsize=6.5, color=P["bcaf"])
ax.set_xticks(xs)
ax.set_xticklabels([f"{g}$\\times$" for g in MAGS])
ax.set_xlabel("magnification")
ax.set_ylabel("difference in balanced accuracy (pp)")
ax.set_ylim(-11, 11)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_title("(a)  Cross-attention against its own ablation", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
mods = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny", "resnet50"]
mu = [np.mean([fm[(m, g)]["img_balanced_accuracy"].mean() for g in MAGS]) * 100 for m in mods]
par = [58.45, 55.35, 27.82, 27.52, 23.51]
LOFF = {"bcaf": (0, 9, "center"), "concat_ablation": (-4, -13, "right"),
        "convnext_tiny": (5, 6, "left"), "swin_tiny": (-5, -12, "right"),
        "resnet50": (0, 9, "center")}
for m, x_, y_ in zip(mods, par, mu):
    ax.scatter([x_], [y_], s=34, color=P[m], zorder=4)
    dx, dy, ha_ = LOFF[m]
    ax.annotate(SHORT[m], xy=(x_, y_), xytext=(dx, dy), textcoords="offset points",
                ha=ha_, fontsize=6.5, color=P[m])
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("balanced accuracy, mean over\nfour magnifications (%)")
ax.set_xlim(18, 66)
ax.set_ylim(83.5, 92.0)
ax.set_title("(b)  Capacity buys nothing above ResNet-50", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.8)
save(fig, "fig_ablation")


# ============================================================ 4. power
mod4 = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny"]
mask_modern = power.model_a.isin(mod4) & power.model_b.isin(mod4)
sd_med = float(power.loc[mask_modern, "sd_diff"].median())
alpha, target_power = 0.05, 0.80

fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
ax = axes[0]
ks = np.arange(3, 201)
tc = stats.t.ppf(1 - alpha / 2, ks - 1)
tp = stats.t.ppf(target_power, ks - 1)
rho5 = float(power["nb_rho"].iloc[0])
ax.plot(ks, (tc + tp) * sd_med / np.sqrt(ks) * 100, color=P["ink"], lw=1.4,
        label="naive paired $t$")
ax.plot(ks, (tc + tp) * math.sqrt(rho5) * sd_med * 100 * np.ones_like(ks) * 0 +
        (tc + tp) * np.sqrt(1.0 / ks + (1.0 / ks) / (1 - 1.0 / ks)) * sd_med * 100,
        color=P["accent"], lw=1.4, ls="--", label="Nadeau–Bengio corrected")
obs = np.abs(power.loc[mask_modern, "mean_diff"].to_numpy()) * 100
ax.scatter(np.full(len(obs), 5) + (np.random.default_rng(1).random(len(obs)) - .5) * 2.5,
           obs, s=7, color=P["ok"], zorder=5, alpha=0.8)
ax.annotate("observed differences among\nthe four modern architectures",
            xy=(6, 1.2), xytext=(38, 3.6), fontsize=6.3, color=P["ok"],
            arrowprops=dict(arrowstyle="-", lw=0.6, color=P["ok"]))
leak_pp = leak[leak.protocol.str.startswith("image")]["inflation_pp"].to_numpy()
ax.axhspan(leak_pp.min(), leak_pp.max(), color=P["bcaf"], alpha=0.16, lw=0)
ax.text(198, leak_pp.max() + 0.4, "leakage effect, 6.8–10.9 pp", ha="right",
        fontsize=6.3, color=P["bcaf"])
ax.set_xlabel("number of cross-validation folds $k$")
ax.set_ylabel("minimum detectable difference (pp)")
ax.set_xlim(0, 205)
ax.set_ylim(0, 16)
ax.legend(fontsize=6.3, loc="upper right")
ax.set_title("(a)  Resolving power of the design", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
deltas = np.linspace(0, 0.16, 400)
for k, c, ls in [(5, P["accent"], "-"), (10, P["ink"], "--"), (20, P["muted"], ":")]:
    tcr = stats.t.ppf(1 - alpha / 2, k - 1)
    ncp = deltas / (sd_med / math.sqrt(k))
    pwv = 1 - stats.nct.cdf(tcr, k - 1, ncp) + stats.nct.cdf(-tcr, k - 1, ncp)
    ax.plot(deltas * 100, pwv, color=c, lw=1.3, ls=ls, label=f"$k={k}$")
ax.axhline(0.80, color=P["muted"], lw=0.8, ls="--")
ax.text(15.8, 0.82, "80% power", ha="right", fontsize=6.5, color=P["muted"])
med_obs = float(np.median(obs))
tcr = stats.t.ppf(1 - alpha / 2, 4)
ncp = (med_obs / 100) / (sd_med / math.sqrt(5))
pw_med = 1 - stats.nct.cdf(tcr, 4, ncp) + stats.nct.cdf(-tcr, 4, ncp)
ax.plot([med_obs], [pw_med], "o", ms=4.5, color=P["ok"], zorder=5)
ax.annotate(f"median architectural\ndifference: {med_obs:.2f} pp",
            xy=(med_obs, pw_med), xytext=(3.0, 0.30), fontsize=6.3, color=P["ok"],
            arrowprops=dict(arrowstyle="-", lw=0.6, color=P["ok"]))
ax.set_xlabel("true difference in balanced accuracy (pp)")
ax.set_ylabel("statistical power")
ax.set_xlim(0, 16)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=6.5, loc="lower right")
ax.set_title("(b)  Power against true effect size", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_power")


# ============================================================ 5. selective prediction
rcx = rc[rc.magnification == 40].set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
ax = axes[0]
for m in MODELS:
    cov, risk = np.load(f"{OUT}/rc_{m}_m40.npy")
    ax.plot(cov * 100, risk * 100, color=P[m], lw=1.3,
            label=f"{SHORT[m]}   AURC = {rcx.loc[m,'aurc']:.4f}")
ax.set_xlabel("coverage (% of images retained)")
ax.set_ylabel("selective risk (% error)")
ax.set_xlim(5, 100)
ax.set_ylim(0, 12)
ax.legend(fontsize=6.2, loc="upper left")
ax.set_title("(a)  Risk–coverage at 40$\\times$", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
s40 = sel[sel.magnification == 40]
for m in MODELS:
    s = s40[s40.model == m].sort_values("referral_rate")
    ax.plot(s["referral_rate"] * 100, s["errors_captured"] * 100, "o-",
            color=P[m], lw=1.3, ms=3.0, label=SHORT[m])
ax.plot([0, 100], [0, 100], color=P["muted"], lw=0.8, ls=":")
ax.text(72, 68, "random referral", fontsize=6.5, color=P["muted"], rotation=33)
ax.set_xlabel("referral rate (% sent for review)")
ax.set_ylabel("errors captured (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.legend(fontsize=6.2, loc="lower right")
ax.set_title("(b)  Error capture at a given review burden", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_selective")


# ============================================================ 6. patient structure
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
ax = axes[0]
for m in MODELS:
    gg = oof40[m].groupby("patient_id")["correct"].agg(err=lambda s: (1 - s).sum())
    e = np.sort(gg["err"].to_numpy())[::-1]
    cum = np.cumsum(e) / e.sum() * 100
    x = np.arange(1, len(e) + 1) / len(e) * 100
    ax.plot(np.concatenate([[0], x]), np.concatenate([[0], cum]), color=P[m], lw=1.3,
            label=SHORT[m])
ax.plot([0, 100], [0, 100], color=P["muted"], lw=0.8, ls=":")
ax.text(58, 54, "errors spread evenly", fontsize=6.5, color=P["muted"], rotation=33)
ax.axvline(5 / 81 * 100, color=P["accent"], lw=0.8, ls="--")
ax.text(5 / 81 * 100 + 2, 18, "5 patients", fontsize=6.5, color=P["accent"])
ax.set_xlabel("grouping units, ranked by error count (%)")
ax.set_ylabel("cumulative share of all errors (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.legend(fontsize=6.2, loc="lower right")
ax.set_title("(a)  Errors concentrate in few patients", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
for m in ["bcaf", "convnext_tiny"]:
    sp = np.load(f"{OUT}/sim_bacc_{m}_m40.npy") * 100
    ax.hist(sp, bins=60, color=P[m], alpha=0.45, label=SHORT[m], density=True)
ymax = ax.get_ylim()[1]
for m, fy in [("convnext_tiny", 0.50), ("bcaf", 0.78)]:
    sp = np.load(f"{OUT}/sim_bacc_{m}_m40.npy")
    ba = fm[(m, 40)]["img_balanced_accuracy"]
    obsv = float((ba.max() - ba.min()) * 100)
    pct = float((sp <= obsv / 100).mean() * 100)
    ax.axvline(obsv, color=P[m], lw=1.3, ls="--")
    ax.text(obsv + 0.7, ymax * fy, f"observed\n{obsv:.1f} pp\n({_ord(pct)} pctile)",
            fontsize=6.5, color=P[m], ha="left", va="top")
ax.set_xlabel("spread of fold balanced accuracy, max $-$ min (pp)")
ax.set_ylabel("density")
ax.legend(fontsize=6.5, loc="upper right")
ax.set_title("(b)  Fold spread under reshuffled patients", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_patient_structure")


# ============================================================ 7. calibration heterogeneity
sub = pd.read_csv(f"{OUT}/per_subtype.csv")
sub40 = sub[sub.magnification == 40]
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.7))
ax = axes[0]
c40 = calib[calib.magnification == 40]
w = 0.16
folds = sorted(c40["fold"].unique())
for i, m in enumerate(MODELS):
    t = c40[c40.model == m].sort_values("fold")
    ax.bar(np.array(folds) + (i - 2) * w, t["T"], width=w, color=P[m],
           edgecolor="none", label=SHORT[m])
ax.axhline(1.0, color=P["ink"], lw=0.8, ls="--")
ax.text(4.45, 1.03, "$T=1$ (no change)", fontsize=6.5, ha="right", color=P["ink"])
ax.set_xlabel("evaluation fold")
ax.set_ylabel("fitted temperature $T$")
ax.set_xticks(folds)
ax.set_ylim(0, 1.75)
ax.legend(fontsize=6.0, ncol=2, loc="upper left")
ax.set_title("(a)  Temperature varies across folds", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
order = sub40[sub40.model == "convnext_tiny"].sort_values("accuracy")["subtype"].tolist()
y = np.arange(len(order))
wb = 0.16
for i, m in enumerate(MODELS):
    s = sub40[sub40.model == m].set_index("subtype").loc[order]
    ax.barh(y + (i - 2) * wb, s["accuracy"] * 100, height=wb, color=P[m],
            edgecolor="none", label=SHORT[m])
ax.set_yticks(y)
ax.set_yticklabels([o.replace("_", " ").replace("tumor", "tumour") for o in order],
                   fontsize=6.3)
ax.set_xlabel("accuracy (%)")
ax.set_xlim(0, 105)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.axvline(90, color=P["muted"], lw=0.8, ls=":")
ax.set_title("(b)  Accuracy by tumour subtype (40$\\times$)", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.4)
save(fig, "fig_heterogeneity")


# ============================================================ 8. reliability diagrams
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.7))
ax = axes[0]
nb = 15
edges = np.linspace(0.5, 1.0, nb + 1)
for m in MODELS:
    d = oof40[m]
    conf = d["conf"].to_numpy()
    corr = d["correct"].to_numpy().astype(float)
    xsb, ysb = [], []
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        sel_ = (conf >= lo) & (conf < hi) if i < nb - 1 else (conf >= lo) & (conf <= hi)
        if sel_.sum() < 25:
            continue
        xsb.append(conf[sel_].mean())
        ysb.append(corr[sel_].mean())
    ax.plot(np.array(xsb) * 100, np.array(ysb) * 100, "o-", color=P[m], lw=1.2,
            ms=3.0, label=SHORT[m])
ax.plot([50, 100], [50, 100], color=P["ink"], lw=0.8, ls="--")
ax.text(88, 84, "perfect calibration", fontsize=6.3, color=P["ink"], rotation=38)
ax.set_xlabel("mean confidence in bin (%)")
ax.set_ylabel("empirical accuracy in bin (%)")
ax.set_xlim(50, 101)
ax.set_ylim(40, 101)
ax.legend(fontsize=6.2, loc="upper left")
ax.set_title("(a)  Reliability, out-of-fold at 40$\\times$", fontsize=8, loc="left", pad=4)
style(ax)

ax = axes[1]
cs = pd.read_csv(f"{RES}/calibration_per_fold.csv")
xs4 = np.arange(len(MAGS))
for i, m in enumerate(MODELS):
    mu = [cs[(cs.model == m) & (cs.magnification == g)]["ece_before"].mean() for g in MAGS]
    sd = [cs[(cs.model == m) & (cs.magnification == g)]["ece_before"].std(ddof=1) for g in MAGS]
    ax.errorbar(xs4 + (i - 2) * 0.09, mu, yerr=sd, color=P[m], lw=1.2, marker="o",
                ms=3.0, capsize=1.8, elinewidth=0.7, label=SHORT[m])
ax.set_xticks(xs4)
ax.set_xticklabels([f"{g}$\\times$" for g in MAGS])
ax.set_xlabel("magnification")
ax.set_ylabel("per-fold expected calibration error")
ax.set_ylim(0, 0.30)
ax.legend(fontsize=6.2, loc="upper left", ncol=2)
ax.set_title("(b)  Miscalibration by magnification", fontsize=8, loc="left", pad=4)
style(ax)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_reliability")


# ============================================================ 9. XAI faithfulness
METH = ["gradcam", "gradcam_pp", "fusion_attention", "random_control"]
MLAB = {"gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++",
        "fusion_attention": "Fusion attention", "random_control": "Random control"}
MCOL = {"gradcam": P["convnext_tiny"], "gradcam_pp": P["ok"],
        "fusion_attention": P["bcaf"], "random_control": P["muted"]}
fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.6))
for ax, col, lab, ttl in [
        (axes[0], "gap_mean", "insertion $-$ deletion AUC", "(a)  Faithfulness gap"),
        (axes[1], "adcc_mean", "ADCC", "(b)  ADCC"),
        (axes[2], "aopc_mean", "AOPC", "(c)  AOPC: control highest on average")]:
    for i, meth in enumerate(METH):
        v = xai[xai.method == meth][col].to_numpy()
        ax.bar([i], [v.mean()], width=0.66, color=MCOL[meth],
               hatch="///" if meth == "random_control" else None,
               edgecolor="white" if meth == "random_control" else "none", lw=0.0)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=1)], color=P["ink"], lw=0.8,
                    capsize=2, fmt="none")
        ax.scatter(np.full(len(v), i) + (np.random.default_rng(7 + i).random(len(v)) - .5) * 0.28,
                   v, s=5.5, color=P["ink"], alpha=0.65, zorder=4, linewidths=0)
    ax.set_xticks(range(len(METH)))
    ax.set_xticklabels(["Grad-\nCAM", "Grad-\nCAM++", "Fusion\nattn.",
                        "Random\ncontrol"], fontsize=6.0)
    ax.set_ylabel(lab, fontsize=7)
    ax.set_title(ttl, fontsize=7.2, loc="left", pad=4)
    style(ax)
axes[0].axhline(0, color=P["ink"], lw=0.7)
fig.tight_layout(w_pad=1.2)
save(fig, "fig_xai")
print("pw_med at median sd =", round(pw_med*100,1))
print("figures done")
