"""
BreakHis data layer with strict patient-wise splitting.

The central correctness requirement of this project: no patient may ever appear
in both a training and an evaluation fold. Every split produced here is checked
by an assertion that fails loudly rather than silently degrading into leakage.

Filename grammar (BreaKHis_v1):
    SOB_B_A-14-22549AB-40-001.png
    |   | |  |          |   `-- sequence number
    |   | |  |          `------ magnification (40|100|200|400)
    |   | |  `----------------- slide / patient identifier (14-22549AB)
    |   | `-------------------- subtype code (A,F,PT,TA | DC,LC,MC,PC)
    |   `---------------------- class (B=benign, M=malignant)
    `-------------------------- biopsy procedure (SOB)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------
# Label vocabulary
# ----------------------------------------------------------------------------
BINARY_CLASSES = ("benign", "malignant")

SUBTYPE_NAMES = {
    "A": "adenosis",
    "F": "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}
SUBTYPE_ORDER = ("A", "F", "PT", "TA", "DC", "LC", "MC", "PC")
SUBTYPE_TO_IDX = {code: i for i, code in enumerate(SUBTYPE_ORDER)}

MAGNIFICATIONS = (40, 100, 200, 400)

#: Extensions treated as candidate images when locating the dataset.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# SOB_B_A-14-22549AB-40-001.png
_FNAME_RE = re.compile(
    r"^SOB[-_](?P<cls>[BM])[-_](?P<subtype>[A-Z]{1,2})[-_]"
    r"(?P<patient>\d{2}-\d+[A-Z]*)[-_]"
    r"(?P<mag>40|100|200|400)[-_]"
    r"(?P<seq>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$",
    re.IGNORECASE,
)


#: Slide/patient identifiers look like "14-22549AB" wherever they appear.
_PATIENT_RE = re.compile(r"(\d{2}-\d+[A-Z]*)", re.IGNORECASE)


def infer_from_path(path: Path, root: Path) -> dict | None:
    """Recover labels from folder structure when filenames have been renamed.

    Some BreakHis mirrors flatten or rebuild the directory tree and rename the
    files, which defeats the filename grammar. Where the folders still encode
    class, subtype, magnification and slide id, we can recover the record.

    Returns None if the patient identifier cannot be recovered. That case must
    NOT be silently tolerated: without patient ids there is no patient-wise
    split, and the central claim of this study cannot be evaluated.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    lowered = [x.lower() for x in parts]

    if any("malignant" in x for x in lowered):
        binary = 1
    elif any("benign" in x for x in lowered):
        binary = 0
    else:
        binary = None

    code = None
    for x in parts:
        xl = x.lower()
        for c, name in SUBTYPE_NAMES.items():
            if xl == name or xl == c.lower():
                code = c
                break
        if code:
            break

    mag = None
    for x in parts:
        m = re.fullmatch(r"(40|100|200|400)\s*x?", x, re.IGNORECASE)
        if m:
            mag = int(m.group(1))
            break

    patient = None
    for x in reversed(parts):
        m = _PATIENT_RE.search(x)
        if m:
            patient = m.group(1).upper()
            break

    if patient is None:
        return None
    if binary is None and code is not None:
        binary = 0 if code in ("A", "F", "PT", "TA") else 1
    if binary is None:
        return None

    return {
        "patient_id": patient,
        "binary_label": binary,
        "subtype_code": code or ("F" if binary == 0 else "DC"),
        "subtype_label": SUBTYPE_TO_IDX[code] if code else (1 if binary == 0 else 4),
        "magnification": mag or 0,
        "_inferred": True,
    }


@dataclass(frozen=True)
class ImageRecord:
    path: str
    patient_id: str
    binary_label: int      # 0 benign, 1 malignant
    subtype_code: str
    subtype_label: int
    magnification: int


def parse_filename(name: str) -> dict | None:
    """Parse one BreakHis filename. Returns None if it does not match."""
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    subtype = m.group("subtype").upper()
    if subtype not in SUBTYPE_TO_IDX:
        return None
    cls = m.group("cls").upper()
    return {
        "patient_id": m.group("patient").upper(),
        "binary_label": 0 if cls == "B" else 1,
        "subtype_code": subtype,
        "subtype_label": SUBTYPE_TO_IDX[subtype],
        "magnification": int(m.group("mag")),
    }


#: Directories worth probing when no explicit root is given.
DEFAULT_SEARCH_PATHS = (
    "/kaggle/input",      # Kaggle mounts each dataset under its own slug here
    "/content",           # Google Colab
    "./data", "../data", "./BreaKHis_v1", "../BreaKHis_v1", ".",
)


def inventory(path, max_files: int = 8) -> dict:
    """Describe what is actually in a directory: counts, extensions, samples.

    This turns "not found" into a diagnosis. Reporting only what is missing
    forces a round trip; reporting what is present usually answers the question.
    """
    from collections import Counter

    path = Path(path)
    if not path.exists():
        return {"exists": False}
    try:
        files = [f for f in path.rglob("*") if f.is_file()]
        subdirs = sorted(d.name for d in path.iterdir() if d.is_dir())[:20]
    except (PermissionError, OSError) as e:
        return {"exists": True, "error": str(e)}
    return {
        "exists": True,
        "n_files": len(files),
        "extensions": Counter(f.suffix.lower() for f in files).most_common(8),
        "n_images": sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS),
        "samples": [str(f.relative_to(path)) for f in files[:max_files]],
        "subdirs": subdirs,
    }


def describe_search_locations(candidates=None) -> str:
    """Readable report of every probe location. Safe to call at any time."""
    out = []
    for c in list(candidates or DEFAULT_SEARCH_PATHS):
        inv = inventory(c)
        if not inv["exists"]:
            out.append(f"  {c}\n      does not exist")
        elif "error" in inv:
            out.append(f"  {c}\n      unreadable: {inv['error']}")
        else:
            out.append(
                f"  {c}\n"
                f"      subdirs   : {inv['subdirs'] or '(none)'}\n"
                f"      files     : {inv['n_files']}  (images: {inv['n_images']})\n"
                f"      extensions: {inv['extensions'] or '(none)'}\n"
                f"      samples   : {inv['samples'][:4] or '(none)'}"
            )
    return "\n".join(out)


def find_dataset_root(candidates=None, verbose: bool = True) -> Path:
    """Locate a BreakHis tree without hardcoding a mount path.

    Kaggle mounts every attached dataset under ``/kaggle/input/<slug>``, and
    BreakHis mirrors circulate under different slugs and sometimes with
    restructured folders, so a hardcoded path breaks for anyone who attached a
    different copy.

    Three passes, most specific first:
      1. a file matching the canonical BreakHis filename grammar
      2. any file whose name begins with ``SOB``
      3. any directory holding a substantial number of images

    A pass-3 match is returned with a warning, because :func:`build_index` can
    often still label such a tree from its folder structure.

    On failure the error carries a full inventory of everything searched, so the
    message diagnoses the problem rather than merely reporting it.
    """
    candidates = list(candidates or DEFAULT_SEARCH_PATHS)

    def probes_for(base: Path):
        try:
            kids = [d for d in sorted(base.iterdir()) if d.is_dir()]
        except (PermissionError, OSError):
            kids = []
        return kids + [base]          # children first: prefer the specific folder

    for pattern, label in ((None, "canonical BreakHis filename"),
                           ("SOB*", "SOB-prefixed filename")):
        for c in candidates:
            base = Path(c)
            if not base.exists():
                continue
            for probe in probes_for(base):
                try:
                    if pattern is None:
                        hit = next((f for f in probe.rglob("*")
                                    if f.is_file() and parse_filename(f.name)), None)
                    else:
                        hit = next((f for f in probe.rglob(pattern) if f.is_file()), None)
                except (PermissionError, OSError):
                    continue
                if hit is not None:
                    if verbose:
                        print(f"Found BreakHis under: {probe}")
                        print(f"  matched by : {label}")
                        print(f"  example    : {hit}")
                    return probe

    for c in candidates:
        base = Path(c)
        if not base.exists():
            continue
        for probe in probes_for(base):
            inv = inventory(probe)
            if inv.get("n_images", 0) >= 50:
                import warnings
                warnings.warn(
                    f"No BreakHis-style filenames found, but {probe} holds "
                    f"{inv['n_images']} images. Using it and inferring labels "
                    f"from the folder structure. VERIFY the counts in Table I "
                    f"before trusting them. Samples: {inv['samples'][:3]}",
                    RuntimeWarning,
                )
                if verbose:
                    print(f"Falling back to image-bearing directory: {probe}")
                return probe

    raise FileNotFoundError(
        "Could not locate a BreakHis image tree.\n\n"
        "WHAT IS ACTUALLY PRESENT:\n" + describe_search_locations(candidates) + "\n\n"
        "HOW TO READ THIS:\n"
        "  * /kaggle/input has no subdirs  -> no dataset is attached. Use the\n"
        "    right-hand panel: Add Data -> search 'breakhis' -> add it.\n"
        "  * a subdir is listed but its samples do not start with 'SOB'\n"
        "    -> restructured mirror. Call build_index() on that path directly;\n"
        "    it will try to infer labels from the folder names.\n"
        "  * otherwise pass the directory explicitly: build_index('/path/to/data')\n"
    )


def acquire_via_kagglehub(slug: str = "ambarish/breakhis", verbose: bool = True):
    """Download BreakHis through kagglehub when no copy is already mounted.

    Inside a Kaggle notebook the runtime is already authenticated, so this needs
    no token and no interaction with the Add Data panel -- which is the step that
    most often goes wrong. Requires Internet to be enabled in notebook settings.

    Returns the download path, or None if the attempt fails for any reason. The
    caller decides what to do about it; this never raises.
    """
    try:
        import kagglehub
    except ImportError:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "kagglehub"],
                           check=True)
            import kagglehub
        except Exception as e:
            if verbose:
                print(f"kagglehub unavailable ({e}); cannot auto-download.")
            return None

    try:
        if verbose:
            print(f"No local copy found. Downloading '{slug}' via kagglehub "
                  f"(a few minutes, ~4 GB)...")
        path = kagglehub.dataset_download(slug)
        if verbose:
            print(f"Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        if verbose:
            print(f"kagglehub download failed: {type(e).__name__}: {e}")
            print("Most likely cause: Internet is OFF in notebook settings "
                  "(Settings -> Internet -> On).")
        return None


def locate_or_acquire(candidates=None, slug: str = "ambarish/breakhis",
                      verbose: bool = True) -> Path:
    """Find a mounted BreakHis copy, or download one if none is present.

    This is the entry point the notebook uses, so a missing dataset resolves
    itself instead of stopping the run.
    """
    try:
        return find_dataset_root(candidates, verbose=verbose)
    except FileNotFoundError as first_error:
        if verbose:
            print("No mounted copy found; attempting download.\n")
        got = acquire_via_kagglehub(slug, verbose=verbose)
        if got is None:
            raise first_error
        try:
            return find_dataset_root([str(got)], verbose=verbose)
        except FileNotFoundError:
            return Path(got)


def build_index(root: str | Path | None = None) -> pd.DataFrame:
    """Walk the BreakHis tree and build a dataframe of every parsable image.

    Pass root=None to auto-discover the dataset (see find_dataset_root).

    Works regardless of intermediate directory layout — it keys entirely off the
    filename grammar, so it tolerates the several folder arrangements the
    dataset circulates in (official tarball, Kaggle mirrors, flattened copies).
    """
    if root is None:
        root = find_dataset_root()
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"BreakHis root does not exist: {root}\n"
            "Call build_index() with no argument to auto-discover it, or see "
            "find_dataset_root() for the locations it probes."
        )

    images = [f for f in root.rglob("*")
              if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

    rows, skipped = [], 0
    for f in images:
        meta = parse_filename(f.name)
        if meta is None:
            skipped += 1
            continue
        meta["_inferred"] = False
        meta["path"] = str(f)
        rows.append(meta)

    n_by_filename = len(rows)

    # Fallback: a restructured mirror whose folders still encode the labels.
    if n_by_filename == 0 and images:
        import warnings
        inferred, unrecoverable = [], 0
        for f in images:
            meta = infer_from_path(f, root)
            if meta is None:
                unrecoverable += 1
                continue
            meta["path"] = str(f)
            inferred.append(meta)

        if not inferred:
            raise RuntimeError(
                f"Found {len(images)} images under {root} but could not recover "
                f"labels from either filenames or folder names.\n"
                f"Sample paths:\n  " +
                "\n  ".join(str(f.relative_to(root)) for f in images[:5]) +
                "\n\nThis mirror is not usable for a patient-wise study: without "
                "slide identifiers there is no way to keep a patient out of both "
                "training and test. Use a mirror that preserves the original "
                "BreaKHis_v1 filenames, e.g. the Kaggle dataset 'ambarish/breakhis'."
            )

        frac = len(inferred) / len(images)
        warnings.warn(
            f"Filenames did not match the BreaKHis grammar. Labels were inferred "
            f"from folder structure for {len(inferred)}/{len(images)} images "
            f"({frac:.1%}); {unrecoverable} had no recoverable patient id and were "
            f"dropped. VERIFY the counts in the dataset table before proceeding, "
            f"and state in the paper that a restructured mirror was used.",
            RuntimeWarning,
        )
        rows, skipped = inferred, unrecoverable

    if not rows:
        raise RuntimeError(
            f"No BreakHis images parsed under {root}.\n"
            f"Images found: {len(images)}\n"
            "Check that the path points at an extracted BreaKHis tree. "
            "Call data.describe_search_locations() to see what is present."
        )

    df = pd.DataFrame(rows)
    # Deduplicate: some mirrors ship the same image under two folder paths.
    df["_key"] = (
        df.subtype_code + "|" + df.patient_id + "|"
        + df.magnification.astype(str) + "|"
        + df.path.map(lambda s: Path(s).name)
    )
    before = len(df)
    df = df.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    df.attrs["n_duplicates_removed"] = before - len(df)
    df.attrs["n_unparsable_files"] = skipped
    df.attrs["labels_inferred_from_path"] = bool(df["_inferred"].any()) if "_inferred" in df else False
    if "_inferred" in df:
        df = df.drop(columns="_inferred")
    return df


def verify_against_published(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the parsed index against the counts published by Spanhol et al.

    Returns a comparison table. Deviations are surfaced rather than raised: a
    Kaggle mirror may legitimately be a subset, but the paper must then say so.
    """
    published = {
        ("benign", "A"): 444, ("benign", "F"): 1014,
        ("benign", "PT"): 453, ("benign", "TA"): 569,
        ("malignant", "DC"): 3451, ("malignant", "LC"): 626,
        ("malignant", "MC"): 792, ("malignant", "PC"): 560,
    }
    out = []
    for (cls, code), expected in published.items():
        want_bin = 0 if cls == "benign" else 1
        got = int(((df.binary_label == want_bin) & (df.subtype_code == code)).sum())
        out.append({
            "class": cls,
            "subtype": SUBTYPE_NAMES[code],
            "published": expected,
            "found": got,
            "delta": got - expected,
            "match": got == expected,
        })
    tbl = pd.DataFrame(out)
    tbl.loc[len(tbl)] = {
        "class": "TOTAL", "subtype": "",
        "published": 7909, "found": int(len(df)),
        "delta": int(len(df)) - 7909, "match": int(len(df)) == 7909,
    }
    return tbl


def make_patient_wise_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "binary_label",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a fold index to every image, grouping by patient.

    StratifiedGroupKFold keeps whole patients together while balancing the class
    ratio across folds as far as the grouping permits. Adds a 'fold' column.
    """
    df = df.copy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(df, y=df[label_col], groups=df["patient_id"])
    ):
        df.loc[df.index[val_idx], "fold"] = fold

    if (df["fold"] < 0).any():
        raise RuntimeError("Fold assignment left unassigned rows.")
    assert_no_patient_leakage(df)
    return df


def assert_no_patient_leakage(df: pd.DataFrame, fold_col: str = "fold") -> None:
    """Hard guarantee that each patient lives in exactly one fold."""
    per_patient = df.groupby("patient_id")[fold_col].nunique()
    offenders = per_patient[per_patient > 1]
    if len(offenders) > 0:
        raise AssertionError(
            "PATIENT LEAKAGE DETECTED — these patients span multiple folds:\n"
            f"{offenders.to_string()}\n"
            "This invalidates the evaluation. Fix the splitter before training."
        )


def assert_disjoint(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Guarantee a train/eval pair shares no patient. Call before every fit."""
    overlap = set(train_df.patient_id) & set(eval_df.patient_id)
    if overlap:
        raise AssertionError(
            f"PATIENT LEAKAGE: {len(overlap)} patient(s) in both train and eval: "
            f"{sorted(overlap)[:10]}"
        )


def make_image_wise_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "binary_label",
    seed: int = 42,
) -> pd.DataFrame:
    """Deliberately leaky image-wise split — the protocol most prior work uses.

    Included ONLY so the paper can quantify the leakage effect by running the
    identical model under both protocols. Never use this to report headline
    performance.
    """
    from sklearn.model_selection import StratifiedKFold

    df = df.copy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(df, y=df[label_col])):
        df.loc[df.index[val_idx], "fold"] = fold
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold composition table for the paper's protocol section."""
    g = df.groupby("fold")
    return pd.DataFrame({
        "n_images": g.size(),
        "n_patients": g["patient_id"].nunique(),
        "n_benign": g["binary_label"].apply(lambda s: int((s == 0).sum())),
        "n_malignant": g["binary_label"].apply(lambda s: int((s == 1).sum())),
        "pct_malignant": g["binary_label"].mean().mul(100).round(2),
    }).reset_index()
