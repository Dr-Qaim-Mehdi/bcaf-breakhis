#!/usr/bin/env python3
"""Every quantity quoted in the manuscript, computed from the sweep. Prints a report."""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import json, math
import numpy as np
import pandas as pd
from scipy import stats

RES = _os.path.join(_ROOT, "results", "sweep")
OUT = _os.path.join(_ROOT, "results", "derived")
MODELS = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny", "resnet50"]
MOD4 = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny"]
MAGS = [40, 100, 200, 400]
pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 400)

fm = {(m, g): pd.read_csv(f"{RES}/fold_metrics_{m}_m{g}.csv") for m in MODELS for g in MAGS}
power = pd.read_csv(f"{OUT}/power_all.csv")
rc = pd.read_csv(f"{OUT}/risk_coverage.csv")
sel = pd.read_csv(f"{OUT}/selective_prediction.csv")
leak = pd.read_csv(f"{RES}/leakage_by_magnification.csv")
cal = pd.read_csv(f"{RES}/calibration_per_fold.csv")
xai = pd.read_csv(f"{RES}/xai_faithfulness_allfolds.csv")
stat = pd.read_csv(f"{RES}/stats_family_by_magnification.csv")
perf = pd.read_csv(f"{RES}/performance_by_magnification.csv")

R = {}
print("=" * 78)
print("1. FAMILY PERFORMANCE  (balanced accuracy, mean +- fold sd)")
tab = pd.DataFrame({m: {g: f"{fm[(m,g)]['img_balanced_accuracy'].mean():.4f}"
                        f"+-{fm[(m,g)]['img_balanced_accuracy'].std(ddof=1):.4f}"
                        for g in MAGS} for m in MODELS})
print(tab.to_string())

# best model per magnification
print("\nbest per magnification (balanced accuracy):")
for g in MAGS:
    vals = {m: fm[(m, g)]["img_balanced_accuracy"].mean() for m in MODELS}
    b = max(vals, key=vals.get)
    print(f"  {g}x: {b} {vals[b]:.4f};  worst modern gap to resnet50 = "
          f"{(min(vals[m] for m in MOD4) - vals['resnet50'])*100:.2f} pp; "
          f"best-modern gap = {(vals[b]-vals['resnet50'])*100:.2f} pp")

# spread among the four modern models
print("\nspread among the four modern architectures (pp):")
for g in MAGS:
    vals = [fm[(m, g)]["img_balanced_accuracy"].mean() for m in MOD4]
    print(f"  {g}x: {(max(vals)-min(vals))*100:.2f}")
allmod = [fm[(m, g)]["img_balanced_accuracy"].mean() for m in MOD4 for g in MAGS]
R["modern_spread_by_mag"] = [float((max([fm[(m,g)]['img_balanced_accuracy'].mean() for m in MOD4])
                                    - min([fm[(m,g)]['img_balanced_accuracy'].mean() for m in MOD4])) * 100)
                             for g in MAGS]

# fold spread within model x magnification
sp = {(m, g): (fm[(m, g)]["img_balanced_accuracy"].max()
               - fm[(m, g)]["img_balanced_accuracy"].min()) * 100 for m in MODELS for g in MAGS}
print(f"\nwithin-cell fold spread (pp): min {min(sp.values()):.1f}  "
      f"median {np.median(list(sp.values())):.1f}  max {max(sp.values()):.1f}")
R["fold_spread_pp"] = dict(min=min(sp.values()), median=float(np.median(list(sp.values()))),
                           max=max(sp.values()))
print("  40x per model:", {m: round(sp[(m, 40)], 1) for m in MODELS})

# rank instability: how often does the winner change by fold?
print("\nrank instability at each magnification (winner per fold, among 5 models):")
for g in MAGS:
    win = []
    for f in range(5):
        vals = {m: fm[(m, g)].set_index("fold").loc[f, "img_balanced_accuracy"] for m in MODELS}
        win.append(max(vals, key=vals.get))
    print(f"  {g}x: {win}  -> {len(set(win))} distinct winners")

print("\n" + "=" * 78)
print("2. ABLATION: BCAF vs concat fusion")
pa = power[(power.model_a == "bcaf") & (power.model_b == "concat_ablation")].set_index("magnification")
for g in MAGS:
    r = pa.loc[g]
    print(f"  {g}x: diff {r['mean_diff']*100:+.2f} pp   sd_d {r['sd_diff']:.4f}   "
          f"MDE naive {r['mde_naive_pp']:.2f}  corrected {r['mde_nb_pp']:.2f}  "
          f"power {r['observed_power']*100:.1f}%")
st = stat[(stat.model_a == "bcaf") & (stat.model_b == "concat_ablation")]
print(st[["magnification", "difference", "cohens_d", "p_ttest_corrected", "p_wilcoxon"]].to_string(index=False))
R["ablation"] = {int(g): dict(diff_pp=float(pa.loc[g, "mean_diff"] * 100),
                              mde_nb_pp=float(pa.loc[g, "mde_nb_pp"]),
                              p=float(st[st.magnification == g]["p_ttest_corrected"].iloc[0]))
                 for g in MAGS}
print("  mean |diff| over four magnifications: "
      f"{np.mean([abs(pa.loc[g,'mean_diff']) for g in MAGS])*100:.2f} pp; "
      f"sign flips: {sum(1 for g in MAGS if pa.loc[g,'mean_diff'] < 0)} of 4 favour concat")

print("\n" + "=" * 78)
print("3. RESOLVING POWER")
mask_modern = power.model_a.isin(MOD4) & power.model_b.isin(MOD4)
mm = power[mask_modern]
mr = power[~mask_modern]
print(f"  modern-vs-modern pairs: n={len(mm)}")
print(f"    |observed diff| pp: min {abs(mm.mean_diff).min()*100:.2f}  "
      f"median {abs(mm.mean_diff).median()*100:.2f}  max {abs(mm.mean_diff).max()*100:.2f}")
print(f"    MDE naive pp:      min {mm.mde_naive_pp.min():.2f}  median {mm.mde_naive_pp.median():.2f}  max {mm.mde_naive_pp.max():.2f}")
print(f"    MDE corrected pp:  min {mm.mde_nb_pp.min():.2f}  median {mm.mde_nb_pp.median():.2f}  max {mm.mde_nb_pp.max():.2f}")
print(f"    pairs where |diff| >= corrected MDE: {int((abs(mm.mean_diff)*100 >= mm.mde_nb_pp).sum())} of {len(mm)}")
print(f"    achieved power: median {mm.observed_power.median()*100:.1f}%  max {mm.observed_power.max()*100:.1f}%")
print(f"  vs ResNet-50 pairs: n={len(mr)}")
print(f"    |observed diff| pp: min {abs(mr.mean_diff).min()*100:.2f}  median {abs(mr.mean_diff).median()*100:.2f}  max {abs(mr.mean_diff).max()*100:.2f}")
print(f"    MDE corrected pp: median {mr.mde_nb_pp.median():.2f}")
print(f"    pairs where |diff| >= corrected MDE: {int((abs(mr.mean_diff)*100 >= mr.mde_nb_pp).sum())} of {len(mr)}")
print(f"    pairs where |diff| >= naive MDE: {int((abs(mr.mean_diff)*100 >= mr.mde_naive_pp).sum())} of {len(mr)}")
print(f"  folds needed for 1 pp (modern pairs): naive median {mm.folds_1pp_naive.median():.0f} "
      f"(range {mm.folds_1pp_naive.min():.0f}-{mm.folds_1pp_naive.max():.0f}); "
      f"corrected median {mm.folds_1pp_nb.median():.0f} "
      f"(range {mm.folds_1pp_nb.min():.0f}-{mm.folds_1pp_nb.max():.0f})")
print(f"  Nadeau-Bengio rho = {power.nb_rho.iloc[0]:.3f}")
print(f"  significant after Holm, any pair: {int(stat['significant'].sum())} of {len(stat)}")
print(f"  significant before correction (p_ttest_corrected < .05): "
      f"{int((stat.p_ttest_corrected < 0.05).sum())} of {len(stat)}")
print(stat[stat.p_ttest_corrected < 0.05][["magnification", "model_a", "model_b", "difference",
                                           "p_ttest_corrected", "p_ttest_holm"]].to_string(index=False))
R["mde_modern"] = dict(min=float(mm.mde_nb_pp.min()), median=float(mm.mde_nb_pp.median()),
                       max=float(mm.mde_nb_pp.max()))

eq = pd.read_csv(f"{RES}/stats_equivalence_power.csv")
print(f"\n  TOST at 1 pp margin: equivalent in {int(eq.equivalent.sum())} of {len(eq)} pairs")

print("\n" + "=" * 78)
print("4. LEAKAGE")
print(leak.to_string(index=False))
iw = leak[leak.protocol.str.startswith("image")].set_index("magnification")
pw = leak[leak.protocol.str.startswith("patient")].set_index("magnification")
print(f"\n  inflation pp: {[round(iw.loc[g,'inflation_pp'],2) for g in MAGS]}  "
      f"mean {iw['inflation_pp'].mean():.2f}")
print(f"  balanced-accuracy inflation pp: "
      f"{[round((iw.loc[g,'balanced_accuracy']-pw.loc[g,'balanced_accuracy'])*100,2) for g in MAGS]}")
print(f"  sd ratio: {[round(iw.loc[g,'sd_ratio'],2) for g in MAGS]}")
print(f"  groups shared: {[int(iw.loc[g,'patients_shared_across_folds']) for g in MAGS]}")
print(f"  leaky accuracy range: {iw['accuracy'].min()*100:.2f}-{iw['accuracy'].max()*100:.2f}%")
print(f"  leaky AUC range: {iw['auc'].min():.4f}-{iw['auc'].max():.4f}")
R["leak"] = {int(g): dict(inflation=float(iw.loc[g, "inflation_pp"]),
                          sd_ratio=float(iw.loc[g, "sd_ratio"]),
                          leaky_acc=float(iw.loc[g, "accuracy"]),
                          pw_acc=float(pw.loc[g, "accuracy"])) for g in MAGS}

print("\n" + "=" * 78)
print("5. CALIBRATION (fold-honest temperature scaling)")
cg = cal.groupby(["magnification", "model"]).agg(
    T_mean=("T", "mean"), T_min=("T", "min"), T_max=("T", "max"),
    ece_before=("ece_before", "mean"), ece_after=("ece_after", "mean"),
    brier_before=("brier_before", "mean"), brier_after=("brier_after", "mean")).round(4)
print(cg.to_string())
imp = (cal["ece_after"] < cal["ece_before"])
print(f"\n  fold-level cells where scaling improved ECE: {int(imp.sum())} of {len(cal)}")
agg = cg.reset_index()
print(f"  model x magnification cells where mean ECE improved: "
      f"{int((agg.ece_after < agg.ece_before).sum())} of {len(agg)}")
print(f"  temperature > 1 (overconfident) in {int((cal['T']>1).sum())} of {len(cal)} folds")
print(f"  T range overall: {cal['T'].min():.3f}-{cal['T'].max():.3f}")
for g in MAGS:
    c = cal[cal.magnification == g]
    print(f"    {g}x: T {c['T'].min():.3f}-{c['T'].max():.3f}; "
          f"mean per-fold ECE {c['ece_before'].mean():.4f}")
R["calibration_improved_cells"] = int((agg.ece_after < agg.ece_before).sum())

print("\n" + "=" * 78)
print("6. SELECTIVE PREDICTION")
print(rc.round(4).to_string(index=False))
for g in MAGS:
    r = rc[rc.magnification == g].set_index("model")
    lo, hi = r["aurc"].idxmin(), r["aurc"].idxmax()
    accs = r["accuracy"]
    print(f"  {g}x: AURC {r.loc[lo,'aurc']:.4f} ({lo}) to {r.loc[hi,'aurc']:.4f} ({hi}) "
          f"= {r.loc[hi,'aurc']/r.loc[lo,'aurc']:.2f}x; accuracy spread "
          f"{(accs.max()-accs.min())*100:.2f} pp")
s40 = sel[(sel.magnification == 40) & (sel.tau == 0.90)].set_index("model")
print("\n  at tau=0.90, 40x:")
print(s40[["referral_rate", "selective_accuracy", "errors_captured", "residual_error_rate"]].round(4).to_string())

print("\n" + "=" * 78)
print("7. XAI FAITHFULNESS (BCAF, 5 folds, 200 images each)")
cols = ["deletion_auc_mean", "insertion_auc_mean", "gap_mean", "aopc_mean",
        "adcc_mean", "complexity_mean", "avg_drop_mean"]
xg = xai.groupby("method")[cols].agg(["mean", "std"]).round(4)
print(xg.to_string())
ctrl = xai[xai.method == "random_control"]
for meth in ["gradcam", "gradcam_pp", "fusion_attention"]:
    v = xai[xai.method == meth]
    wins_gap = int((v.sort_values("fold")["gap_mean"].to_numpy()
                    > ctrl.sort_values("fold")["gap_mean"].to_numpy()).sum())
    wins_adcc = int((v.sort_values("fold")["adcc_mean"].to_numpy()
                     > ctrl.sort_values("fold")["adcc_mean"].to_numpy()).sum())
    aopc_beat = int((v.sort_values("fold")["aopc_mean"].to_numpy()
                     > ctrl.sort_values("fold")["aopc_mean"].to_numpy()).sum())
    print(f"  {meth}: beats control on gap in {wins_gap}/5 folds, ADCC in {wins_adcc}/5, "
          f"AOPC in {aopc_beat}/5; gap ratio to control = "
          f"{v['gap_mean'].mean()/ctrl['gap_mean'].mean():.1f}x")
print(f"  control mean AOPC {ctrl['aopc_mean'].mean():.4f} vs best real "
      f"{max(xai[xai.method==m]['aopc_mean'].mean() for m in ['gradcam','gradcam_pp','fusion_attention']):.4f}")

with open(f"{OUT}/numbers.json", "w") as fh:
    json.dump(R, fh, indent=2, default=float)
print("\nwrote numbers.json")
