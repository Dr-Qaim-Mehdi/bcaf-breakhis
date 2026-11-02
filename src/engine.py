"""
Training and evaluation engine.

All models pass through identical code here — same optimiser family, schedule,
augmentation, early stopping and evaluation — so that any measured difference
between architectures is attributable to the architecture rather than to
training-recipe differences. This is a precondition for the paper's claims.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    model_key: str = "bcaf"
    num_classes: int = 2
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    backbone_lr_mult: float = 0.1     # discriminative fine-tuning
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    warmup_epochs: int = 3
    patience: int = 8
    amp: bool = True
    num_workers: int = 2
    seed: int = 42
    class_balanced_loss: bool = True
    task: str = "binary"              # or "subtype"
    protocol: str = "patient_wise"    # or "image_wise"
    magnification: str = "all"        # "all" | "40" | "100" | "200" | "400"
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_transforms(img_size: int, train: bool):
    """Stain-aware augmentation.

    Colour jitter on hue/saturation is the important one for H&E: it simulates
    the staining and scanner variation that is the dominant domain shift between
    pathology labs, and is what a model must be invariant to if it is to
    generalise beyond the acquisition site.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BreakHisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, label_col: str = "binary_label"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row[self.label_col]), i


def make_loaders(train_df, val_df, cfg: TrainConfig, label_col: str):
    tr = BreakHisDataset(train_df, build_transforms(cfg.img_size, True), label_col)
    va = BreakHisDataset(val_df, build_transforms(cfg.img_size, False), label_col)
    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                   num_workers=cfg.num_workers, pin_memory=True, drop_last=True),
        DataLoader(va, batch_size=cfg.batch_size * 2, shuffle=False,
                   num_workers=cfg.num_workers, pin_memory=True),
    )


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, probs, num_classes: int) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        if num_classes == 2:
            m["auc"] = roc_auc_score(y_true, probs[:, 1])
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            m["sensitivity"] = tp / (tp + fn) if (tp + fn) else 0.0
            m["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
            # Clinically the decisive number: a missed malignancy is the costly error.
            m["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0
        else:
            m["auc"] = roc_auc_score(y_true, probs, multi_class="ovr",
                                     average="macro")
    except (ValueError, IndexError):
        m["auc"] = float("nan")
    return m


def aggregate_to_patient(df_idx: pd.DataFrame, probs: np.ndarray,
                        y_true: np.ndarray) -> dict:
    """Patient-level decision by mean probability over that patient's images.

    This is the clinically meaningful unit — a diagnosis is rendered per patient,
    not per field of view — and is almost never reported on BreakHis.
    """
    tmp = pd.DataFrame({"patient_id": df_idx["patient_id"].values, "y": y_true})
    for c in range(probs.shape[1]):
        tmp[f"p{c}"] = probs[:, c]
    g = tmp.groupby("patient_id")
    pcols = [f"p{c}" for c in range(probs.shape[1])]
    pat_probs = g[pcols].mean().values
    pat_true = g["y"].first().values
    pat_pred = pat_probs.argmax(axis=1)
    out = compute_metrics(pat_true, pat_pred, pat_probs, probs.shape[1])
    out["n_patients"] = int(len(pat_true))
    return out


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def _param_groups(model, cfg: TrainConfig):
    head_kw = ("head", "gate", "proj_", "c2v", "v2c", "norm")
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if any(k in n for k in head_kw) else backbone).append(p)
    return [
        {"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": head, "lr": cfg.lr},
    ]


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    """Returns (logits, probs, y_true, order_index)."""
    model.eval()
    L, Y, I = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        L.append(model(x).float().cpu())
        Y.append(y)
        I.append(idx)
    logits = torch.cat(L)
    y = torch.cat(Y).numpy()
    idx = torch.cat(I).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    return logits.numpy(), probs, y, idx




def train_fold(model, train_df, val_df, cfg: TrainConfig, device,
              label_col: str = "binary_label", verbose: bool = True):
    set_seed(cfg.seed)
    train_loader, val_loader = make_loaders(train_df, val_df, cfg, label_col)

    if cfg.class_balanced_loss:
        counts = np.bincount(train_df[label_col].values, minlength=cfg.num_classes)
        w = torch.tensor(counts.sum() / (cfg.num_classes * np.maximum(counts, 1)),
                         dtype=torch.float32, device=device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    model = model.to(device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    steps = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=cfg.epochs * steps,
        pct_start=min(0.4, cfg.warmup_epochs / max(cfg.epochs, 1)),
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"score": -np.inf, "epoch": -1, "state": None}
    history, since_improve = [], 0

    for epoch in range(cfg.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        _, probs, y_true, _ = evaluate(model, val_loader, device, cfg.num_classes)
        met = compute_metrics(y_true, probs.argmax(1), probs, cfg.num_classes)
        met.update(epoch=epoch, train_loss=run_loss / max(seen, 1),
                   seconds=round(time.time() - t0, 1))
        history.append(met)

        # Balanced accuracy as the selection criterion — BreakHis is ~2:1
        # malignant, so plain accuracy rewards majority-class bias.
        score = met["balanced_accuracy"]
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            since_improve = 0
        else:
            since_improve += 1

        if verbose:
            print(f"  ep {epoch:02d} loss {met['train_loss']:.4f} "
                  f"acc {met['accuracy']:.4f} bal {score:.4f} "
                  f"auc {met['auc']:.4f} ({met['seconds']}s)"
                  + ("  <- best" if since_improve == 0 else ""))
        if since_improve >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(best epoch {best['epoch']}, bal {best['score']:.4f})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, pd.DataFrame(history), best


def run_cross_validation(df, cfg: TrainConfig, build_fn, device,
                        label_col: str = "binary_label", verbose: bool = True,
                        ckpt_dir: str | None = None):
    """Full k-fold run. Returns per-fold metrics and stacked out-of-fold outputs.

    The out-of-fold predictions are what every downstream analysis uses —
    statistical tests, calibration, XAI faithfulness — so each sample is scored
    exactly once, by a model that never saw that patient.

    If ckpt_dir is given, each fold's best weights are written there. The
    explanation analysis then loads them instead of retraining, which on a
    six-model grid saves several GPU-hours and — more importantly — guarantees
    the explanations are generated from the exact weights that produced the
    reported metrics, rather than from a second run that merely used the same
    recipe.
    """
    import os
    from data import assert_disjoint

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    folds = sorted(df["fold"].unique())
    rows, oof, ckpts = [], [], []
    for f in folds:
        tr = df[df.fold != f]
        va = df[df.fold == f]
        if cfg.protocol == "patient_wise":
            assert_disjoint(tr, va)
        if verbose:
            print(f"\n--- fold {f}: train {len(tr)} imgs / "
                  f"{tr.patient_id.nunique()} pts | "
                  f"val {len(va)} imgs / {va.patient_id.nunique()} pts ---")

        model = build_fn()
        model, hist, best = train_fold(model, tr, va, cfg, device, label_col, verbose)

        if ckpt_dir:
            cp = os.path.join(ckpt_dir, f"{cfg.model_key}_fold{f}.pt")
            torch.save({"state_dict": model.state_dict(),
                        "model_key": cfg.model_key,
                        "num_classes": cfg.num_classes,
                        "img_size": cfg.img_size,
                        "fold": int(f),
                        "best_epoch": best["epoch"],
                        "best_score": best["score"]}, cp)
            ckpts.append(cp)
            if verbose:
                print(f"  checkpoint -> {cp}")

        _, val_loader = make_loaders(tr, va, cfg, label_col)
        logits, probs, y_true, order = evaluate(model, val_loader, device,
                                                cfg.num_classes)
        va_ord = va.reset_index(drop=True).iloc[order]

        img_m = compute_metrics(y_true, probs.argmax(1), probs, cfg.num_classes)
        pat_m = aggregate_to_patient(va_ord, probs, y_true)
        rows.append({"fold": f, "best_epoch": best["epoch"],
                     **{f"img_{k}": v for k, v in img_m.items()},
                     **{f"pat_{k}": v for k, v in pat_m.items()}})

        oof.append(pd.DataFrame({
            "fold": f,
            "path": va_ord["path"].values,
            "patient_id": va_ord["patient_id"].values,
            "magnification": va_ord["magnification"].values,
            "y_true": y_true,
            "y_pred": probs.argmax(1),
            "correct": (probs.argmax(1) == y_true).astype(int),
            **{f"prob_{c}": probs[:, c] for c in range(probs.shape[1])},
            **{f"logit_{c}": logits[:, c] for c in range(logits.shape[1])},
        }))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows), pd.concat(oof, ignore_index=True)
    if ckpt_dir:
        return (*result, ckpts)
    return result


def load_checkpoint(path: str, build_fn, device):
    """Restore a fold's trained weights saved by run_cross_validation.

    Used by the explanation analysis so that saliency is computed from the very
    weights that produced the reported metrics.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_fn()
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), blob


def summarise_cv(fold_df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std across folds, formatted for a results table."""
    num = fold_df.select_dtypes(include=[np.number]).drop(columns=["fold"],
                                                          errors="ignore")
    return pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1)}).round(4)
