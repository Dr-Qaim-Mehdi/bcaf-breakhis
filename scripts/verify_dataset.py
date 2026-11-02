#!/usr/bin/env python3
"""Verify a local BreakHis copy against the published counts and emit the
dataset-characterisation table + figures the paper needs.

    python verify_dataset.py /path/to/data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data  # noqa: E402


def main(root: str, out_dir: str = "results") -> int:
    df = data.build_index(root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Indexed {len(df)} images | {df.patient_id.nunique()} patients\n")

    check = data.verify_against_published(df)
    print("--- Against Spanhol et al. published counts ---")
    print(check.to_string(index=False))
    check.to_csv(f"{out_dir}/dataset_verification.csv", index=False)

    print("\n--- Images per magnification ---")
    mag = (df.groupby(["magnification", "binary_label"]).size()
             .unstack(fill_value=0)
             .rename(columns={0: "benign", 1: "malignant"}))
    mag["total"] = mag.sum(axis=1)
    print(mag.to_string())
    mag.to_csv(f"{out_dir}/magnification_distribution.csv")

    print("\n--- Images per patient ---")
    per = df.groupby("patient_id").size()
    print(f"min {per.min()}  median {per.median():.0f}  "
          f"max {per.max()}  mean {per.mean():.1f}")
    print("This spread is why image-level bootstrapping understates uncertainty: "
          "a few patients dominate the image count.")

    folds = data.make_patient_wise_folds(df, n_splits=5)
    print("\n--- Patient-wise 5-fold composition ---")
    print(data.split_summary(folds).to_string(index=False))
    folds.to_csv(f"{out_dir}/index_with_folds.csv", index=False)

    data.assert_no_patient_leakage(folds)
    print("\nLEAKAGE CHECK PASSED — no patient appears in more than one fold.")
    print(f"\nWrote index + tables to {out_dir}/")
    return 0 if bool(check.iloc[-1]["match"]) else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"))
