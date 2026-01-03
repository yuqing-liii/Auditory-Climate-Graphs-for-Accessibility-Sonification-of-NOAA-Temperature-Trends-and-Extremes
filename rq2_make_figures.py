import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE = Path.home() / "Desktop" / "data" / "results"
SPEED_CSV = BASE / "RQ2_sweep_speed.csv"
GAMMA_CSV = BASE / "RQ2_sweep_gamma.csv"
SMOOTH_CSV = BASE / "RQ2_sweep_smooth.csv"

OUTDIR = BASE / "figures_rq2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Style (nice + colorblind-friendly)
# -------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors: Okabe-Ito palette (colorblind-safe)
COL_SYSTEM = {
    "S1": "#0072B2",  # blue
    "S2": "#D55E00",  # vermillion
    "S0": "#6E6E6E",  # gray (optional)
}
LINE_STATION = {"NYC": "-", "PHX": "--"}  # station as line style for readability

# -------------------------
# Helpers
# -------------------------
def load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure numeric columns are numeric
    for c in ["rmse","pearson_r","spearman_r","auc","max_f1","seconds_per_day","gamma","smooth_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def agg_mean_std(df: pd.DataFrame, xcol: str, metric: str, systems=("S1","S2"), stations=("NYC","PHX")) -> pd.DataFrame:
    """
    Aggregate across years: mean ± std for each (station, system, xcol).
    """
    sub = df[df["system"].isin(systems) & df["station"].isin(stations)].copy()
    g = sub.groupby(["station","system",xcol])[metric].agg(["mean","std","count"]).reset_index()
    return g.sort_values([ "station","system",xcol ]).reset_index(drop=True)

def plot_lines_with_error(g: pd.DataFrame, xcol: str, metric: str, title: str, xlabel: str, ylabel: str, out_name: str):
    """
    One figure: both stations and both systems.
    - color encodes system (S1,S2)
    - line style encodes station (NYC solid, PHX dashed)
    - error bands show ±1 std across years
    """
    fig = plt.figure(figsize=(7.4, 4.6))
    ax = plt.gca()

    for station in ["NYC","PHX"]:
        for system in ["S1","S2"]:
            s = g[(g["station"]==station) & (g["system"]==system)].sort_values(xcol)
            x = s[xcol].to_numpy()
            y = s["mean"].to_numpy()
            e = s["std"].fillna(0).to_numpy()

            label = f"{station}-{system}"
            ax.plot(
                x, y,
                linestyle=LINE_STATION[station],
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                color=COL_SYSTEM[system],
                label=label
            )
            ax.fill_between(x, y-e, y+e, color=COL_SYSTEM[system], alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Make legend clean
    ax.legend(ncol=2, frameon=True, framealpha=0.9)

    # Save both PNG and PDF
    out_png = OUTDIR / f"{out_name}.png"
    out_pdf = OUTDIR / f"{out_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
    print("Saved:", out_pdf)

def main():
    df_speed = load(SPEED_CSV)
    df_gamma = load(GAMMA_CSV)
    df_smooth = load(SMOOTH_CSV)

    # ---------
    # 1) smooth sweep: RMSE + AUC + max_f1
    # ---------
    g_rmse = agg_mean_std(df_smooth, "smooth_days", "rmse")
    plot_lines_with_error(
        g_rmse, "smooth_days", "rmse",
        title="RQ2 (Smooth sweep): Fidelity vs smoothing window (mean±std across years)",
        xlabel="smooth_days",
        ylabel="RMSE (lower is better)",
        out_name="RQ2_smooth_RMSE"
    )

    g_auc = agg_mean_std(df_smooth, "smooth_days", "auc")
    plot_lines_with_error(
        g_auc, "smooth_days", "auc",
        title="RQ2 (Smooth sweep): Event detectability vs smoothing window (mean±std across years)",
        xlabel="smooth_days",
        ylabel="AUC (higher is better)",
        out_name="RQ2_smooth_AUC"
    )

    g_f1 = agg_mean_std(df_smooth, "smooth_days", "max_f1")
    plot_lines_with_error(
        g_f1, "smooth_days", "max_f1",
        title="RQ2 (Smooth sweep): max-F1 vs smoothing window (mean±std across years)",
        xlabel="smooth_days",
        ylabel="max-F1 (higher is better)",
        out_name="RQ2_smooth_maxF1"
    )

    # ---------
    # 2) speed sweep: AUC + max_f1
    # ---------
    g_auc_sp = agg_mean_std(df_speed, "seconds_per_day", "auc")
    plot_lines_with_error(
        g_auc_sp, "seconds_per_day", "auc",
        title="RQ2 (Speed sweep): Event detectability vs playback speed (mean±std across years)",
        xlabel="seconds_per_day (sec/day)  [larger = slower playback]",
        ylabel="AUC (higher is better)",
        out_name="RQ2_speed_AUC"
    )

    g_f1_sp = agg_mean_std(df_speed, "seconds_per_day", "max_f1")
    plot_lines_with_error(
        g_f1_sp, "seconds_per_day", "max_f1",
        title="RQ2 (Speed sweep): max-F1 vs playback speed (mean±std across years)",
        xlabel="seconds_per_day (sec/day)  [larger = slower playback]",
        ylabel="max-F1 (higher is better)",
        out_name="RQ2_speed_maxF1"
    )

    # ---------
    # 3) gamma sweep: AUC + max_f1
    # ---------
    g_auc_g = agg_mean_std(df_gamma, "gamma", "auc")
    plot_lines_with_error(
        g_auc_g, "gamma", "auc",
        title="RQ2 (Compression sweep): Event detectability vs gamma (mean±std across years)",
        xlabel="gamma (smaller = stronger compression)",
        ylabel="AUC (higher is better)",
        out_name="RQ2_gamma_AUC"
    )

    g_f1_g = agg_mean_std(df_gamma, "gamma", "max_f1")
    plot_lines_with_error(
        g_f1_g, "gamma", "max_f1",
        title="RQ2 (Compression sweep): max-F1 vs gamma (mean±std across years)",
        xlabel="gamma (smaller = stronger compression)",
        ylabel="max-F1 (higher is better)",
        out_name="RQ2_gamma_maxF1"
    )

    print("\nAll figures saved to:", OUTDIR)

if __name__ == "__main__":
    main()
