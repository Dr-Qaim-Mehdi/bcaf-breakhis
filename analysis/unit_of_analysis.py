#!/usr/bin/env python3
"""Unit-of-analysis comparison for the forty pairwise architecture comparisons (paper, Table 4, upper block).

For every magnification and every pair of the five architectures, the paired out-of-fold predictions are compared
in three ways:

  1. naive image-level McNemar test (exact), which treats the images as independent: reported only for contrast;
  2. cluster-aware sign-flip permutation test: each grouping unit's discordance contribution
     (images A got right and B got wrong, minus the reverse) is sign-flipped as one block, 20,000 permutations;
  3. exact McNemar test on patient-level predictions (probabilities averaged over each unit's images).

Usage:  python analysis/unit_of_analysis.py [results_dir] [out_dir]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "sweep")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "derived")
os.makedirs(OUT, exist_ok=True)

MODELS = ["bcaf", "concat_ablation", "convnext_tiny", "swin_tiny", "resnet50"]
MAGS = [40, 100, 200, 400]
N_PERM = 20_000
RNG = np.random.default_rng(20260928)


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def sign_flip_p(d, n_perm=N_PERM):
    """Two-sided sign-flip permutation p-value for sum(d), flipping each cluster's contribution as a block."""
    d = np.asarray(d, dtype=float)
    t_obs = abs(d.sum())
    if t_obs == 0:
        return 1.0
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, d.size))
    t_null = np.abs(signs @ d)
    return (1 + np.sum(t_null >= t_obs - 1e-12)) / (n_perm + 1)


rows = []
for g in MAGS:
    oof = {m: pd.read_csv(os.path.join(RES, f"oof_{m}_m{g}.csv")).set_index("path").sort_index() for m in MODELS}
    for a, b in itertools.combinations(MODELS, 2):
        A, B = oof[a], oof[b]
        assert A.index.equals(B.index), f"prediction files for {a} and {b} at {g}x are not aligned"
        ca, cb = A["correct"].astype(bool), B["correct"].astype(bool)
        # 1. naive image-level
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_naive = exact_mcnemar(n10, n01)
        # 2. cluster-aware sign flip: one discordance contribution per grouping unit
        d = (ca.astype(int) - cb.astype(int)).groupby(A["patient_id"]).sum()
        p_cluster = sign_flip_p(d.to_numpy())
        # 3. patient-level exact McNemar on unit-averaged probabilities
        pa = A.groupby("patient_id").agg(p=("prob_1", "mean"), y=("y_true", "first"))
        pb = B.groupby("patient_id").agg(p=("prob_1", "mean"))
        ra = (pa["p"] >= 0.5).astype(int) == pa["y"]
        rb = (pb["p"].reindex(pa.index) >= 0.5).astype(int) == pa["y"]
        p_patient = exact_mcnemar(int((ra & ~rb).sum()), int((~ra & rb).sum()))
        rows.append(dict(magnification=g, model_a=a, model_b=b, n_images=len(A), n_units=len(d),
                         discordant_a_only=n10, discordant_b_only=n01, p_mcnemar_naive=p_naive,
                         p_cluster_signflip=p_cluster, p_mcnemar_patient=p_patient))

res = pd.DataFrame(rows)
res["involves_resnet50"] = (res["model_a"] == "resnet50") | (res["model_b"] == "resnet50")
res.to_csv(os.path.join(OUT, "unit_of_analysis.csv"), index=False)

print("comparisons declared significant (p < 0.05)      modern pairs   pairs with ResNet-50   all")
for col, name in (("p_mcnemar_naive", "naive image-level McNemar"),
                  ("p_cluster_signflip", "cluster-aware permutation"),
                  ("p_mcnemar_patient", "patient-level exact McNemar")):
    sig = res[col] < 0.05
    print(f"  {name:<44}{int(sig[~res.involves_resnet50].sum()):>8}{int(sig[res.involves_resnet50].sum()):>16}"
          f"{int(sig.sum()):>12}")
ratio = res["p_cluster_signflip"] / res["p_mcnemar_naive"]
print(f"cluster-aware p larger than naive p in {int((ratio > 1).sum())} of {len(res)} comparisons; "
      f"median factor {ratio.median():.1f}")
