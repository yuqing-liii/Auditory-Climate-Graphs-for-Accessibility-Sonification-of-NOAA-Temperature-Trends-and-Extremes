"""
Label heatwave events (binary) using a reproducible rule:
- For each station, compute the monthly 95th percentile of tmax_c
- Mark "hot_day" when tmax_c >= monthly threshold
- Mark event=1 for runs of >= min_run consecutive hot days

Inputs/outputs:
- By default, reads/writes under <repo>/data/processed
- Override base directory by setting ACC_DATA_DIR
  e.g., export ACC_DATA_DIR="/path/to/workdir"
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
import pandas as pd


def get_base_dir() -> Path:
    """
    Base working directory for generated files.
    Override with ACC_DATA_DIR. Defaults to <repo>/data.
    """
    repo_root = Path(__file__).resolve().parent
    base = os.environ.get("ACC_DATA_DIR")
    return (Path(base).expanduser() if base else (repo_root / "data"))


def label_heatwaves(df: pd.DataFrame, p: float = 0.95, min_run: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    # monthly thresholds
    thresh = df.groupby("month")["tmax_c"].quantile(p).to_dict()
    df["thr"] = df["month"].map(thresh)
    df["hot_day"] = (df["tmax_c"] >= df["thr"]).fillna(False)

    # consecutive run labeling
    df["event"] = 0
    run_len = 0
    start_idx = None
    for i, is_hot in enumerate(df["hot_day"].values):
        if is_hot:
            if run_len == 0:
                start_idx = i
            run_len += 1
        else:
            if run_len >= min_run and start_idx is not None:
                df.loc[start_idx:i - 1, "event"] = 1
            run_len = 0
            start_idx = None

    # handle run at end
    if run_len >= min_run and start_idx is not None:
        df.loc[start_idx:len(df) - 1, "event"] = 1

    return df.drop(columns=["thr"])


def run_one(in_path: Path, out_path: Path, p: float, min_run: int) -> None:
    df = pd.read_csv(in_path)
    out = label_heatwaves(df, p=p, min_run=min_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path.as_posix()}")
    print(f"Event rate: {out['event'].mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label heatwave events in station CSV files.")
    parser.add_argument("--p", type=float, default=0.95, help="Monthly percentile threshold (default: 0.95).")
    parser.add_argument("--min-run", type=int, default=3, help="Minimum consecutive hot-day run length (default: 3).")
    parser.add_argument("--in-path", type=str, default=None, help="Optional single input CSV path.")
    parser.add_argument("--out-path", type=str, default=None, help="Optional single output CSV path.")
    args = parser.parse_args()

    processed_dir = get_base_dir() / "processed"

    # Single-file mode (explicit paths)
    if args.in_path and args.out_path:
        run_one(Path(args.in_path).expanduser(), Path(args.out_path).expanduser(), args.p, args.min_run)
        return

    # Default mode: label NYC + PHX inputs produced by fetch_ghcnd_tmax_tmin.py
    inputs_outputs = [
        (
            processed_dir / "NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv",
            processed_dir / "NYC_labeled.csv",
        ),
        (
            processed_dir / "Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv",
            processed_dir / "PHX_labeled.csv",
        ),
    ]

    for in_path, out_path in inputs_outputs:
        run_one(in_path, out_path, args.p, args.min_run)


if __name__ == "__main__":
    main()
