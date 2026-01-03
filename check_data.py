import pandas as pd

files = [
    ("NYC", "~/Desktop/data/processed/NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv"),
    ("PHX", "~/Desktop/data/processed/Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv"),
]

for name, path in files:
    df = pd.read_csv(path.replace("~", "/Users/lunali"))
    print("\n===", name, "===")
    print("rows:", len(df))
    print("date min/max:", df["date"].min(), df["date"].max())
    print("missing tmax_c:", df["tmax_c"].isna().mean())
    print("missing tmin_c:", df["tmin_c"].isna().mean())
    print("tmax_c range:", df["tmax_c"].min(), df["tmax_c"].max())
    print("tmin_c range:", df["tmin_c"].min(), df["tmin_c"].max())
