import pandas as pd

def label_heatwaves(df: pd.DataFrame, p=0.95, min_run=3) -> pd.DataFrame:
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
            if run_len >= min_run:
                df.loc[start_idx:i-1, "event"] = 1
            run_len = 0
            start_idx = None
    # handle run at end
    if run_len >= min_run and start_idx is not None:
        df.loc[start_idx:len(df)-1, "event"] = 1

    return df.drop(columns=["thr"])

def main(in_path, out_path):
    df = pd.read_csv(in_path)
    out = label_heatwaves(df, p=0.95, min_run=3)
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Event rate:", out["event"].mean())

if __name__ == "__main__":
    main(
        "/Users/lunali/Desktop/data/processed/NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv",
        "/Users/lunali/Desktop/data/processed/NYC_labeled.csv",
    )
    main(
        "/Users/lunali/Desktop/data/processed/Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv",
        "/Users/lunali/Desktop/data/processed/PHX_labeled.csv",
    )
