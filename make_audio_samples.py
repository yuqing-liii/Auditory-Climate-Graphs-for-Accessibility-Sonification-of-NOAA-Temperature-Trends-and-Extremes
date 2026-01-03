import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# =========================
# Config (you can tweak)
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

YEARS = [1990, 2016, 2023]

NYC_PATH = "/Users/lunali/Desktop/data/processed/NYC_labeled.csv"
PHX_PATH = "/Users/lunali/Desktop/data/processed/PHX_labeled.csv"

OUTDIR = Path("/Users/lunali/Desktop/data/audio_samples")
OUTDIR.mkdir(parents=True, exist_ok=True)


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


def sonify(sub: pd.DataFrame, system: str):
    """
    S0: raw tmax_c -> pitch
    S1: 7-day moving average smoothing
    S2: S1 + marker pulse on event days (event==1)
    """
    temp = sub["tmax_c"].to_numpy()

    if system in ("S1", "S2"):
        temp = (
            pd.Series(temp)
            .rolling(SMOOTH_DAYS, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    freqs = pitch_from_temp(temp)

    chunks = []
    for i, f in enumerate(freqs):
        y = synth_tone(f, DUR_PER_DAY)

        if system == "S2" and int(sub["event"].iloc[i]) == 1:
            pulse = synth_tone(MARKER_FREQ, min(MARKER_DUR, DUR_PER_DAY))
            y[: len(pulse)] += MARKER_GAIN * pulse

        chunks.append(y)

    y = np.concatenate(chunks)

    # normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y


def run_station(tag: str, in_path: str):
    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    for year in YEARS:
        sub = df[(df["date"] >= f"{year}-01-01") & (df["date"] <= f"{year}-12-31")].reset_index(drop=True)

        # sanity: show event count
        n_pos = int(sub["event"].sum())
        print(f"\n{tag} {year}: days={len(sub)} event_days={n_pos}")

        for sys in ["S0", "S1", "S2"]:
            y = sonify(sub, sys)

            out = OUTDIR / f"{tag}_{sys}_{year}_tmax_d{DUR_PER_DAY}_smooth{SMOOTH_DAYS}.wav"
            sf.write(out, y, SR)

            sec = len(y) / SR
            print(f"  wrote {out.name}  ({sec:.2f}s)")


def main():
    print("Output dir:", OUTDIR)
    print("Config:", {"SR": SR, "DUR_PER_DAY": DUR_PER_DAY, "FMIN": FMIN, "FMAX": FMAX, "SMOOTH_DAYS": SMOOTH_DAYS})
    run_station("NYC", NYC_PATH)
    run_station("PHX", PHX_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
