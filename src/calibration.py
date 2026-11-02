"""
Calibration analysis — how well predicted probabilities match observed frequency.

Accuracy alone cannot tell you whether a model is deployable. A model that is
92% accurate but assigns 0.99 confidence to its errors is more dangerous in a
diagnostic workflow than one that is 90% accurate and correctly signals its own
uncertainty. Calibration is essentially unreported in the BreakHis literature;
this module supplies it.
"""
from __future__ import annotations

import numpy as np

# Torch is needed only for TemperatureScaler. The metrics below are pure NumPy,
# so calibration analysis runs on a machine with no deep-learning stack — which
# is where results are usually analysed and written up.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = nn = F = None
    _HAS_TORCH = False

_Base = nn.Module if _HAS_TORCH else object


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 15) -> float:
    """Equal-width-bin ECE (Guo et al., 2017).

    probs: (N, C) predicted probabilities. labels: (N,) integer ground truth.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def adaptive_calibration_error(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 15) -> float:
    """Equal-mass-bin ECE. More stable than equal-width when confidence is
    concentrated near 1.0, which is the usual regime for these models."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    ace, n = 0.0, len(labels)
    for chunk_c, chunk_a in zip(np.array_split(conf, n_bins),
                                np.array_split(correct, n_bins)):
        if len(chunk_c) == 0:
            continue
        ace += (len(chunk_c) / n) * abs(chunk_a.mean() - chunk_c.mean())
    return float(ace)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray,
                             n_bins: int = 15) -> float:
    """Worst-case bin gap — the number that matters for safety arguments."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            gaps.append(abs(correct[m].mean() - conf[m].mean()))
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score — a strictly proper scoring rule.

    Unlike ECE it penalises both miscalibration and poor discrimination, so it
    cannot be gamed by a model that is uninformative but well calibrated.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray,
                           eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return (bin_centres, observed_accuracy, mean_confidence, bin_counts)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, acc, avg_conf, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        centres.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        acc.append(float(correct[m].mean()) if m.sum() else np.nan)
        avg_conf.append(float(conf[m].mean()) if m.sum() else np.nan)
    return (np.array(centres), np.array(acc), np.array(avg_conf), np.array(counts))


class TemperatureScaler(_Base):
    """Single-parameter post-hoc calibration (Guo et al., 2017).

    CRITICAL: must be fitted on a validation split that is patient-disjoint from
    both the training data and the test data. Fitting on test leaks.
    """

    def __init__(self):
        if not _HAS_TORCH:
            raise ImportError(
                "TemperatureScaler requires PyTorch (pip install torch). "
                "The calibration metrics in this module work without it."
            )
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 200) -> "TemperatureScaler":
        logits = logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)

        t = self.temperature
        if not (0.2 < t < 10.0):
            import warnings
            warnings.warn(
                f"Temperature fitted to {t:.4f}, which is outside the plausible "
                "range (0.2, 10). This usually means the calibration split is "
                "perfectly separated or too small for a stable fit. Report the "
                "raw ECE alongside the scaled one rather than relying on this.",
                RuntimeWarning,
            )
        return self


def calibration_report(probs: np.ndarray, labels: np.ndarray,
                      n_bins: int = 15) -> dict:
    """All calibration metrics in one dict, ready for the results table."""
    return {
        "ECE": expected_calibration_error(probs, labels, n_bins),
        "ACE": adaptive_calibration_error(probs, labels, n_bins),
        "MCE": maximum_calibration_error(probs, labels, n_bins),
        "Brier": brier_score(probs, labels),
        "NLL": negative_log_likelihood(probs, labels),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "accuracy": float((probs.argmax(axis=1) == labels).mean()),
    }
