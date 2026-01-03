import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# =========================
# Config (defaults; overridable via CLI)
# =========================
SR = 22050

# playback speed: seconds per day
# 0.03 -> 1 year ~ 365 * 0.03 = ~11s
DUR_PER_DAY = 0.03

# pitch mapping range (Hz)
FMIN, FMAX = 220.0, 880.0

# smoothing window for S1/S2
SMOOTH_DAYS = 7

# event marker settings for S2
MARKER_FREQ = 1200.0
MARKER_DUR = 0.01      # seconds (<= DUR_PER_DAY)
MARKER_GAIN = 0.8      # added on top of base tone

DEFAULT_YEARS = [1990, 2016, 2023]


def get_base_dir() -> Path:
    """
    Base working directory for generated files.
    Override with ACC_DATA_DIR. Defaults to <repo>/data.
    """
    repo_root = Path(__file__).resolve().parent
    base = os.environ.get("ACC_DATA_DIR")
    return (Path(base).expanduser() if base else (repo_root / "data"))


# =========================
# Helpers
# =========================
def normalize(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin < 1e-9:
        return np.zeros_like(x), xmin, xmax
    return (x - xmin) / (xmax - xmin), xmin, xmax


def pitch_from_temp(temp_c):
    """
    Normalize temps within the window, then map to [FMIN, FMAX].
    This makes each year use the full pitch range (good for listening demos).
    """
    n, _, _ = normalize(temp_c)
    return FMIN + n * (FMAX - FMIN)


def synth_tone(freq, dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def sonify(sub: pd.DataFrame, system: str,
           dur_per_day: float, smooth_days: int,
           marker_freq: float, marker_dur: float, marker_gain: float):
    """
    S0: raw tmax_c -> pitch
    S1: moving average smoothing
    S2: S1 + marker pulse on event days (event==1)
    """
    temp = sub["tmax_c"].to_numpy()

    if system in ("S1", "S2"):
        temp = (
            pd.Series(temp)
            .rolling(smooth_days, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    freqs = pitch_from_temp(temp)

    chunks = []
    for i, f in enumerate(freqs):
        y = synth_tone(f, dur_per_day)

        if system == "S2" and int(sub["event"].iloc[i]) == 1:
            pulse = synth_tone(marker_freq, min(marker_dur, dur_per_day))
            y[: len(pulse)] += marker_gain * pulse

        chunks.append(y)

    y = np.concatenate(chunks)

    # normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y


def run_station(tag: str, in_path: Path, outdir: Path,
                years: list[int],
                dur_per_day: float, smooth_days: int,
                marker_freq: float, marker_dur: float, marker_gain: float):
    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    for year in years:
        sub = df[(df["date"] >= f"{year}-01-01") & (df["date"] <= f"{year}-12-31")].reset_index(drop=True)

        # sanity: show event count
        n_pos = int(sub["event"].sum())
        print(f"\n{tag} {year}: days={len(sub)} event_days={n_pos}")

        for sys in ["S0", "S1", "S2"]:
            y = sonify(
                sub, sys,
                dur_per_day=dur_per_day,
                smooth_days=smooth_days,
                marker_freq=marker_freq,
                marker_dur=marker_dur,
                marker_gain=marker_gain,
            )

            out = outdir / f"{tag}_{sys}_{year}_tmax_d{dur_per_day}_smooth{smooth_days}.wav"
            sf.write(out, y, SR)

            sec = len(y) / SR
            print(f"  wrote {out.name}  ({sec:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export aligned audio samples (WAV) for NYC/PHX and selected years.")
    parser.add_argument("--years", type=str, default="1990,2016,2023",
                        help="Comma-separated years to export (default: 1990,2016,2023).")
    parser.add_argument("--dur-per-day", type=float, default=DUR_PER_DAY, help="Seconds per day (default: 0.03).")
    parser.add_argument("--smooth-days", type=int, default=SMOOTH_DAYS, help="Smoothing window (default: 7).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Optional output directory. Defaults to <base>/audio_samples where base=<repo>/data or ACC_DATA_DIR.")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    base = get_base_dir()
    processed = base / "processed"

    nyc_path = processed / "NYC_labeled.csv"
    phx_path = processed / "PHX_labeled.csv"

    outdir = Path(args.outdir).expanduser() if args.outdir else (base / "audio_samples")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Base dir:", base.as_posix())
    print("Input:", {"NYC": nyc_path.as_posix(), "PHX": phx_path.as_posix()})
    print("Output dir:", outdir.as_posix())
    print("Config:", {"SR": SR, "dur_per_day": args.dur_per_day, "FMIN": FMIN, "FMAX": FMAX, "smooth_days": args.smooth_days})
    run_station("NYC", nyc_path, outdir, years, args.dur_per_day, args.smooth_days, MARKER_FREQ, MARKER_DUR, MARKER_GAIN)
    run_station("PHX", phx_path, outdir, years, args.dur_per_day, args.smooth_days, MARKER_FREQ, MARKER_DUR, MARKER_GAIN)
    print("\nDone.")


if __name__ == "__main__":
    main()
