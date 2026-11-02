"""
Statistical testing for model comparison.

The single most common reviewer objection to BreakHis papers is that a claimed
improvement of a few tenths of a percent is reported without any test of whether
it exceeds run-to-run variance. This module makes every comparison in the paper
defensible: paired tests, multiple-comparison correction, effect sizes, and
confidence intervals — never a bare point estimate.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Paired tests on the same test set
# ----------------------------------------------------------------------------
def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray,
                exact: bool = True) -> dict:
    """McNemar's test on paired per-sample correctness vectors.

    The right test when two models are evaluated on the identical test set:
    it conditions on the discordant pairs and ignores samples both got right or
    both got wrong. Uses the exact binomial form when n is small.
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Correctness vectors must be the same length.")

    n01 = int(np.sum(~correct_a & correct_b))   # only B right
    n10 = int(np.sum(correct_a & ~correct_b))   # only A right
    n_disc = n01 + n10

    if n_disc == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0,
                "test": "degenerate (no discordant pairs)",
                "odds_ratio": np.nan}

    if exact or n_disc < 25:
        p = float(stats.binomtest(n10, n_disc, 0.5).pvalue)
        statistic, test = float(min(n01, n10)), "exact binomial"
    else:
        statistic = (abs(n10 - n01) - 1) ** 2 / n_disc   # continuity-corrected
        p = float(stats.chi2.sf(statistic, df=1))
        test = "chi-square (continuity corrected)"

    return {"n01": n01, "n10": n10, "statistic": statistic, "p_value": p,
            "test": test,
            "odds_ratio": float(n10 / n01) if n01 > 0 else np.inf}


def corrected_paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray,
                          n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    A plain paired t-test across cross-validation folds is anti-conservative,
    because the folds share training data and so are not independent. The
    correction inflates the variance by (1/k + n_test/n_train). Using the
    uncorrected test is a common and serious error in this literature.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = a - b
    k = len(d)
    if k < 2:
        raise ValueError("Need at least 2 folds.")
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0:
        return {"t_statistic": 0.0, "p_value": 1.0, "mean_difference": float(mean_d),
                "df": k - 1, "correction_factor": np.nan}

    correction = (1.0 / k) + (n_test / n_train)
    t = mean_d / np.sqrt(var_d * correction)
    p = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"t_statistic": float(t), "p_value": p,
            "mean_difference": float(mean_d), "df": k - 1,
            "correction_factor": float(correction)}


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Exact Wilcoxon signed-rank — the non-parametric fold-level comparison.

    Note the floor: with k folds the smallest attainable two-sided p is
    2^-(k-1). At k=5 that is 0.0625, so no result can reach p<0.05. The paper
    must state this rather than reporting 'not significant' as if it were
    evidence of equivalence.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    k = len(a)
    min_p = 2.0 ** (-(k - 1))
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "n_folds": k,
                "min_attainable_p": min_p, "at_floor": False}
    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", mode="exact")
    except (TypeError, ValueError):
        res = stats.wilcoxon(a, b, alternative="two-sided")
    p = float(res.pvalue)
    return {"statistic": float(res.statistic), "p_value": p, "n_folds": k,
            "min_attainable_p": min_p,
            "at_floor": bool(abs(p - min_p) < 1e-9)}


# ----------------------------------------------------------------------------
# Effect size and intervals
# ----------------------------------------------------------------------------
def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d. Report alongside every p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 10000,
                alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    boots = np.array([statistic(rng.choice(v, size=len(v), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(statistic(v)), "ci_low": float(lo),
            "ci_high": float(hi), "alpha": alpha}


def bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn,
                       groups: np.ndarray | None = None,
                       n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 42) -> dict:
    """Bootstrap CI for any metric.

    If `groups` (patient ids) is supplied, resamples whole PATIENTS rather than
    images. This is the correct unit of resampling: images from one patient are
    not independent, and image-level bootstrapping produces intervals that are
    far too narrow.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    boots = []
    if groups is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(metric_fn(y_true[idx], y_pred[idx]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        lookup = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([lookup[g] for g in picked])
            boots.append(metric_fn(y_true[idx], y_pred[idx]))
    boots = np.asarray(boots, float)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(metric_fn(y_true, y_pred)),
            "ci_low": float(lo), "ci_high": float(hi),
            "resample_unit": "patient" if groups is not None else "image"}


# ----------------------------------------------------------------------------
# Multiple comparisons
# ----------------------------------------------------------------------------
def holm_bonferroni(p_values, alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction.

    Uniformly more powerful than plain Bonferroni at the same family-wise error
    rate. Comparing k models pairwise means k(k-1)/2 tests; without correction
    the chance of at least one spurious 'significant' result is large.
    """
    p = np.asarray(list(p_values), float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adjusted[i] = min(running, 1.0)
    return pd.DataFrame({
        "p_raw": p,
        "p_adjusted": adjusted,
        "reject_null": adjusted < alpha,
    })


def pairwise_model_comparison(fold_scores: dict[str, np.ndarray],
                             n_train: int, n_test: int,
                             alpha: float = 0.05) -> pd.DataFrame:
    """Full pairwise comparison table across models, Holm-corrected.

    fold_scores: {model_name: array of per-fold scores}
    """
    names = list(fold_scores)
    rows = []
    for a, b in itertools.combinations(names, 2):
        sa, sb = np.asarray(fold_scores[a]), np.asarray(fold_scores[b])
        tt = corrected_paired_ttest(sa, sb, n_train, n_test)
        wx = wilcoxon_test(sa, sb)
        rows.append({
            "model_a": a, "model_b": b,
            "mean_a": sa.mean(), "mean_b": sb.mean(),
            "difference": sa.mean() - sb.mean(),
            "cohens_d": cohens_d_paired(sa, sb),
            "p_ttest_corrected": tt["p_value"],
            "p_wilcoxon": wx["p_value"],
            "wilcoxon_at_floor": wx["at_floor"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        holm = holm_bonferroni(df["p_ttest_corrected"], alpha)
        df["p_ttest_holm"] = holm["p_adjusted"].values
        df["significant"] = holm["reject_null"].values
    return df


def equivalence_test(a: np.ndarray, b: np.ndarray, margin: float,
                    alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for statistical equivalence.

    Essential for this paper's central claim. 'p > 0.05' means only that a
    difference was not detected; TOST lets you make the positive claim that any
    difference is smaller than a pre-specified margin. Without it, asserting
    that architectures perform equivalently is an unsupported inference from a
    null result.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    k = len(d)
    se = d.std(ddof=1) / np.sqrt(k)
    if se == 0:
        return {"equivalent": True, "p_lower": 0.0, "p_upper": 0.0,
                "mean_difference": float(d.mean()), "margin": margin}
    t_lo = (d.mean() + margin) / se
    t_hi = (d.mean() - margin) / se
    p_lo = float(stats.t.sf(t_lo, df=k - 1))     # H0: diff <= -margin
    p_hi = float(stats.t.cdf(t_hi, df=k - 1))    # H0: diff >= +margin
    return {"equivalent": bool(max(p_lo, p_hi) < alpha),
            "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": float(max(p_lo, p_hi)),
            "mean_difference": float(d.mean()), "margin": margin}
