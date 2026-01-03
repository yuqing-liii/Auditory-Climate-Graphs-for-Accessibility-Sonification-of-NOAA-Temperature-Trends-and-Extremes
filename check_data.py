from pathlib import Path
import os
import pandas as pd

def get_base_dir() -> Path:
    """
    Base working directory for generated files.
    Override with ACC_DATA_DIR. Defaults to <repo>/data.
    """
    repo_root = Path(__file__).resolve().parent
    base = os.environ.get("ACC_DATA_DIR")
    return (Path(base).expanduser() if base else (repo_root / "data"))

processed = get_base_dir() / "processed"

files = [
    ("NYC", processed / "NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv"),
    ("PHX", processed / "Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv"),
]

for name, path in files:
    df = pd.read_csv(path)
    print("\n===", name, "===")
    print("rows:", len(df))
    print("date min/max:", df["date"].min(), df["date"].max())
    print("missing tmax_c:", df["tmax_c"].isna().mean())
    print("missing tmin_c:", df["tmin_c"].isna().mean())
    print("tmax_c range:", df["tmax_c"].min(), df["tmax_c"].max())
    print("tmin_c range:", df["tmin_c"].min(), df["tmin_c"].max())
