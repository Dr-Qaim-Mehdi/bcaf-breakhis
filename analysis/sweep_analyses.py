#!/usr/bin/env python3
"""
Secondary analyses over the full sweep: 5 models x 4 magnifications.
Computed entirely from the released out-of-fold predictions in results/sweep/.
No retraining. Outputs to results/derived/.
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import os, re, json, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

RES = _os.path.join(_ROOT, "results", "sweep")
OUT = _os.path.join(_ROOT, "results", "derived")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260908)

MODELS = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny", "resnet50"]
MAGS = [40, 100, 200, 400]
PRETTY = {"bcaf": "BCAF", "concat_ablation": "ConcatFusion",
          "convnext_tiny": "ConvNeXt-Tiny", "swin_tiny": "Swin-Tiny",
          "resnet50": "ResNet-50"}


def subtype_from_path(p):
    m = re.search(r"/(benign|malignant)/SOB/([a-z_]+)/", p)
    return m.group(2) if m else "unknown"


# ---------------------------------------------------------------- load
oof, fm = {}, {}
for m in MODELS:
    for g in MAGS:
        df = pd.read_csv(f"{RES}/oof_{m}_m{g}.csv")
        df["subtype"] = df["path"].map(subtype_from_path)
        df["conf"] = df[["prob_0", "prob_1"]].max(axis=1)
        df["p1"] = df["prob_1"]
        oof[(m, g)] = df.reset_index(drop=True)
        fm[(m, g)] = pd.read_csv(f"{RES}/fold_metrics_{m}_m{g}.csv")

N_BY_MAG = {g: len(oof[(MODELS[0], g)]) for g in MAGS}
print("images per magnification:", N_BY_MAG)
report = {"n_by_mag": N_BY_MAG}

alpha, power_target = 0.05, 0.80


# ================================================== 1. power / MDE, every pair
def power_row(a, b, n_total):
    d = a - b
    k = len(d)
    sd = d.std(ddof=1)
    mean_d = d.mean()
    tcrit = stats.t.ppf(1 - alpha / 2, k - 1)
    tpow = stats.t.ppf(power_target, k - 1)
    mde_naive = (tcrit + tpow) * sd / math.sqrt(k)
    n_test = n_total / k
    n_train = n_total - n_test
    rho = (1.0 / k) + (n_test / n_train)
    se_nb = math.sqrt(rho) * sd
    t_nb = mean_d / se_nb if se_nb > 0 else np.nan
    p_nb = 2 * (1 - stats.t.cdf(abs(t_nb), k - 1)) if se_nb > 0 else np.nan
    mde_nb = (tcrit + tpow) * se_nb
    ncp = abs(mean_d) / (sd / math.sqrt(k)) if sd > 0 else np.inf
    pw = 1 - stats.nct.cdf(tcrit, k - 1, ncp) + stats.nct.cdf(-tcrit, k - 1, ncp)

    def folds_needed(target, corrected):
        for kk in range(3, 5001):
            tc = stats.t.ppf(1 - alpha / 2, kk - 1)
            tp = stats.t.ppf(power_target, kk - 1)
            if corrected:
                nt = n_total / kk
                r = (1.0 / kk) + (nt / (n_total - nt))
                val = (tc + tp) * math.sqrt(r) * sd
            else:
                val = (tc + tp) * sd / math.sqrt(kk)
            if val <= target:
                return kk
        return None

    return dict(k_folds=k, mean_diff=mean_d, sd_diff=sd,
                mde_naive_pp=mde_naive * 100, mde_nb_pp=mde_nb * 100,
                nb_rho=rho, nb_se=se_nb, nb_t=t_nb, nb_p=p_nb,
                observed_power=pw,
                folds_1pp_naive=folds_needed(0.01, False),
                folds_1pp_nb=folds_needed(0.01, True))


rows = []
for g in MAGS:
    for a_, b_ in itertools.combinations(MODELS, 2):
        A = fm[(a_, g)]["img_balanced_accuracy"].to_numpy()
        B = fm[(b_, g)]["img_balanced_accuracy"].to_numpy()
        r = power_row(A, B, N_BY_MAG[g])
        r.update(magnification=g, model_a=a_, model_b=b_,
                 mean_a=A.mean(), mean_b=B.mean())
        rows.append(r)
power_all = pd.DataFrame(rows)
power_all.to_csv(f"{OUT}/power_all.csv", index=False)
print("power done")


# ================================================== 2. risk--coverage / selective
def risk_coverage(conf, correct):
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())


rc_rows, sel_rows = [], []
for g in MAGS:
    for m in MODELS:
        df = oof[(m, g)]
        correct = df["correct"].to_numpy().astype(float)
        conf = df["conf"].to_numpy()
        cov, risk, aurc = risk_coverage(conf, correct)
        n = len(correct)
        n_ok = int(correct.sum())
        perfect = np.concatenate([np.ones(n_ok), np.zeros(n - n_ok)])
        _, _, aurc_opt = risk_coverage(np.arange(n)[::-1].astype(float), perfect)
        rc_rows.append(dict(magnification=g, model=m, accuracy=correct.mean(),
                            aurc=aurc, aurc_optimal=aurc_opt,
                            excess_aurc=aurc - aurc_opt))
        if g == 40:
            np.save(f"{OUT}/rc_{m}_m{g}.npy", np.vstack([cov, risk]))
        total_err = (1 - correct).sum()
        for tau in [0.70, 0.80, 0.90, 0.95, 0.99]:
            ref = conf < tau
            kept = ~ref
            sel_rows.append(dict(
                magnification=g, model=m, tau=tau,
                referral_rate=ref.mean(), coverage=kept.mean(),
                selective_accuracy=correct[kept].mean() if kept.sum() else np.nan,
                errors_captured=(1 - correct)[ref].sum() / total_err if total_err else np.nan,
                residual_error_rate=(1 - correct)[kept].mean() if kept.sum() else np.nan))
pd.DataFrame(rc_rows).to_csv(f"{OUT}/risk_coverage.csv", index=False)
pd.DataFrame(sel_rows).to_csv(f"{OUT}/selective_prediction.csv", index=False)
print("selective done")

# ================================================== 3. patient-level bootstrap
B = 10000
boot_rows, diff_rows = [], []
for g in MAGS:
    pats = np.sort(oof[(MODELS[0], g)]["patient_id"].unique())
    idx_by_pat = {m: {p: oof[(m, g)].index[oof[(m, g)]["patient_id"] == p].to_numpy()
                      for p in pats} for m in MODELS}
    arrs = {m: (oof[(m, g)]["y_true"].to_numpy(),
                oof[(m, g)]["y_pred"].to_numpy(),
                oof[(m, g)]["p1"].to_numpy()) for m in MODELS}
    draws = [RNG.choice(pats, size=len(pats), replace=True) for _ in range(B)]
    store = {m: {"acc": [], "bacc": [], "auc": []} for m in MODELS}
    for sample in draws:
        for m in MODELS:
            idx = np.concatenate([idx_by_pat[m][p] for p in sample])
            y, yp, p1 = arrs[m]
            y, yp, p1 = y[idx], yp[idx], p1[idx]
            acc = (y == yp).mean()
            if len(np.unique(y)) == 2:
                bacc = balanced_accuracy_score(y, yp)
                auc = roc_auc_score(y, p1)
            else:
                bacc = np.nan
                auc = np.nan
            store[m]["acc"].append(acc)
            store[m]["bacc"].append(bacc)
            store[m]["auc"].append(auc)
    for m in MODELS:
        d = oof[(m, g)]
        pt = dict(acc=(d["y_true"] == d["y_pred"]).mean(),
                  bacc=balanced_accuracy_score(d["y_true"], d["y_pred"]),
                  auc=roc_auc_score(d["y_true"], d["p1"]))
        row = dict(magnification=g, model=m)
        for key in ["acc", "bacc", "auc"]:
            v = np.asarray(store[m][key], dtype=float)
            v = v[~np.isnan(v)]
            row[f"{key}_point"] = pt[key]
            row[f"{key}_lo"] = np.percentile(v, 2.5)
            row[f"{key}_hi"] = np.percentile(v, 97.5)
        boot_rows.append(row)
    for a_, b_ in itertools.combinations(MODELS, 2):
        dd = np.asarray(store[a_]["bacc"], dtype=float) - np.asarray(store[b_]["bacc"], dtype=float)
        dd = dd[~np.isnan(dd)]
        da, db = oof[(a_, g)], oof[(b_, g)]
        point = (balanced_accuracy_score(da["y_true"], da["y_pred"])
                 - balanced_accuracy_score(db["y_true"], db["y_pred"]))
        diff_rows.append(dict(magnification=g, model_a=a_, model_b=b_, diff_point=point,
                              lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5),
                              p_two_sided=2 * min((dd <= 0).mean(), (dd >= 0).mean()),
                              frac_favouring_a=float((dd > 0).mean())))
    print("bootstrap done for", g)
pd.DataFrame(boot_rows).to_csv(f"{OUT}/patient_bootstrap_ci.csv", index=False)
pd.DataFrame(diff_rows).to_csv(f"{OUT}/patient_bootstrap_diff.csv", index=False)

# ================================================== 4. per-subtype
sub_rows = []
for g in MAGS:
    for m in MODELS:
        for st, gg in oof[(m, g)].groupby("subtype"):
            sub_rows.append(dict(magnification=g, model=m, subtype=st, n=len(gg),
                                 n_patients=gg["patient_id"].nunique(),
                                 accuracy=gg["correct"].mean(),
                                 mean_conf=gg["conf"].mean()))
pd.DataFrame(sub_rows).to_csv(f"{OUT}/per_subtype.csv", index=False)

# ================================================== 5. error concentration
pat_rows = []
for g in MAGS:
    for m in MODELS:
        df = oof[(m, g)]
        gg = df.groupby("patient_id").agg(n=("correct", "size"),
                                          err=("correct", lambda s: int((1 - s).sum())))
        gg["err_rate"] = gg["err"] / gg["n"]
        gg = gg.sort_values("err", ascending=False)
        tot = gg["err"].sum()
        pat_rows.append(dict(magnification=g, model=m, n_patients=len(gg),
                             n_patients_with_error=int((gg["err"] > 0).sum()),
                             total_errors=int(tot),
                             top5_share=gg["err"].head(5).sum() / tot if tot else np.nan,
                             top10_share=gg["err"].head(10).sum() / tot if tot else np.nan,
                             n_perfect_patients=int((gg["err"] == 0).sum())))
        if g == 40:
            gg.to_csv(f"{OUT}/patient_errors_{m}_m{g}.csv")
pd.DataFrame(pat_rows).to_csv(f"{OUT}/patient_error_concentration.csv", index=False)

# ================================================== 6. fold composition correlations
comp_rows = []
for g in MAGS:
    comp = pd.read_csv(f"{RES}/fold_composition_m{g}.csv")
    for m in MODELS:
        f = fm[(m, g)][["fold", "img_balanced_accuracy"]].merge(comp, on="fold")
        for col in ["pct_malignant", "n_patients", "n_images"]:
            r, p = stats.spearmanr(f[col], f["img_balanced_accuracy"])
            comp_rows.append(dict(magnification=g, model=m, variable=col,
                                  spearman_rho=r, p_value=p, n=len(f)))
pd.DataFrame(comp_rows).to_csv(f"{OUT}/fold_composition_corr.csv", index=False)

# ================================================== 7. patient reallocation simulation
sim_rows = []
NSIM = 20000
for g in MAGS:
    pats = np.sort(oof[(MODELS[0], g)]["patient_id"].unique())
    for m in MODELS:
        df = oof[(m, g)]
        by_pat = {p: df.index[df["patient_id"] == p].to_numpy() for p in pats}
        y = df["y_true"].to_numpy()
        yp = df["y_pred"].to_numpy()
        spreads_b, spreads_a = [], []
        for _ in range(NSIM):
            perm = RNG.permutation(pats)
            groups = np.array_split(perm, 5)
            bs, as_ = [], []
            ok = True
            for grp in groups:
                idx = np.concatenate([by_pat[p] for p in grp])
                yy, pp = y[idx], yp[idx]
                as_.append((yy == pp).mean())
                if len(np.unique(yy)) < 2:
                    ok = False
                    break
                bs.append(balanced_accuracy_score(yy, pp))
            if not ok:
                continue
            spreads_b.append(max(bs) - min(bs))
            spreads_a.append(max(as_) - min(as_))
        sb = np.asarray(spreads_b)
        sa = np.asarray(spreads_a)
        obs_b = fm[(m, g)]["img_balanced_accuracy"].max() - fm[(m, g)]["img_balanced_accuracy"].min()
        obs_a = fm[(m, g)]["img_accuracy"].max() - fm[(m, g)]["img_accuracy"].min()
        sim_rows.append(dict(magnification=g, model=m, n_sim=len(sb),
                             observed_spread_bacc_pp=obs_b * 100,
                             sim_median_bacc_pp=np.median(sb) * 100,
                             sim_p95_bacc_pp=np.percentile(sb, 95) * 100,
                             pctile_of_observed_bacc=float((sb <= obs_b).mean() * 100),
                             observed_spread_acc_pp=obs_a * 100,
                             sim_median_acc_pp=np.median(sa) * 100,
                             sim_p95_acc_pp=np.percentile(sa, 95) * 100,
                             pctile_of_observed_acc=float((sa <= obs_a).mean() * 100)))
        if g == 40:
            np.save(f"{OUT}/sim_bacc_{m}_m{g}.npy", sb)
    print("simulation done for", g)
pd.DataFrame(sim_rows).to_csv(f"{OUT}/fold_spread_simulation.csv", index=False)

with open(f"{OUT}/summary.json", "w") as fh:
    json.dump(report, fh, indent=2, default=float)
print("ALL DONE")
