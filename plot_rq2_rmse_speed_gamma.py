import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---- paths ----
BASE = Path.home() / "Desktop" / "data" / "results"
FIG_DIR = BASE / "figures_rq2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPEED_CSV = BASE / "RQ2_sweep_speed.csv"
GAMMA_CSV = BASE / "RQ2_sweep_gamma.csv"

# ---- plotting helpers ----
def mean_std_by_param(df: pd.DataFrame, param_col: str, metric_col: str = "rmse"):
    # Expect columns: station, system, year, param_col, metric_col
    g = df.groupby(["station", "system", param_col])[metric_col].agg(["mean", "std"]).reset_index()
    return g

def plot_rmse_curve(g: pd.DataFrame, param_col: str, title: str, xlabel: str, out_prefix: str):
    # Style mapping: system color, station linestyle
    # Keep it simple & paper-friendly
    system_order = ["S1", "S2"] if set(g["system"]) >= {"S1","S2"} else sorted(g["system"].unique())
    station_order = ["NYC", "PHX"] if set(g["station"]) >= {"NYC","PHX"} else sorted(g["station"].unique())
    linestyles = {"NYC": "-", "PHX": "--"}

    plt.figure(figsize=(10, 6), dpi=200)

    for station in station_order:
        for system in system_order:
            sub = g[(g["station"] == station) & (g["system"] == system)].sort_values(param_col)
            if sub.empty:
                continue
            x = sub[param_col].to_numpy()
            y = sub["mean"].to_numpy()
            s = sub["std"].fillna(0).to_numpy()

            label = f"{station}-{system}"
            plt.plot(x, y, marker="o", linestyle=linestyles.get(station, "-"), label=label)
            plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("RMSE (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()

    png_path = FIG_DIR / f"{out_prefix}_RMSE.png"
    pdf_path = FIG_DIR / f"{out_prefix}_RMSE.pdf"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()

    print("Saved:", png_path)
    print("Saved:", pdf_path)

def main():
    if not SPEED_CSV.exists():
        raise SystemExit(f"Missing: {SPEED_CSV}")
    if not GAMMA_CSV.exists():
        raise SystemExit(f"Missing: {GAMMA_CSV}")

    speed = pd.read_csv(SPEED_CSV)
    gamma = pd.read_csv(GAMMA_CSV)

    # Only plot S1/S2 for RQ2 (as you did in other RQ2 figures)
    if "system" in speed.columns:
        speed = speed[speed["system"].isin(["S1", "S2"])].copy()
    if "system" in gamma.columns:
        gamma = gamma[gamma["system"].isin(["S1", "S2"])].copy()

    g_speed = mean_std_by_param(speed, "seconds_per_day", "rmse")
    plot_rmse_curve(
        g_speed,
        "seconds_per_day",
        title="RQ2 (Speed sweep): Fidelity vs playback speed (mean±std across years)",
        xlabel="seconds_per_day (sec/day)  [larger = slower playback]",
        out_prefix="RQ2_speed"
    )

    g_gamma = mean_std_by_param(gamma, "gamma", "rmse")
    plot_rmse_curve(
        g_gamma,
        "gamma",
        title="RQ2 (Compression sweep): Fidelity vs gamma (mean±std across years)",
        xlabel="gamma (smaller = stronger compression)",
        out_prefix="RQ2_gamma"
    )

if __name__ == "__main__":
    main()
