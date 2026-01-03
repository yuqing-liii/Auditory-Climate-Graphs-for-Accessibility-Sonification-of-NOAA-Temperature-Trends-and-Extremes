# sonify_make_demos.py
# This script is a direct .py version of the exact snippet you executed in the terminal.
# It generates WAV demos for NYC + PHX, for years [1976, 1998, 2022], systems S0/S1/S2.

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

# ==== Fixed parameters (same as your terminal snippet) ====
SR = 22050
DUR_PER_DAY = 0.03          # 30ms per day -> ~11s per year
FMIN, FMAX = 220.0, 880.0   # pitch range (Hz)
SMOOTH_DAYS = 7             # smoothing window for S1/S2
OUTDIR = Path("/Users/lunali/Desktop/data/audio_samples")
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize(x):
    """Min-max normalize an array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin < 1e-9:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def pitch_from_temp(temp_c):
    """Map temperature values to pitch frequencies."""
    n = normalize(temp_c)
    return FMIN + n * (FMAX - FMIN)


def synth_tone(freq, dur):
    """Synthesize a sine tone at a given frequency and duration."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def sonify(sub, system="S0"):
    """
    S0: pitch only
    S1: pitch with smoothing (7-day rolling mean)
    S2: S1 + event marker pulse (1200 Hz) on event days
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

        # S2: event marker pulse
        if system == "S2" and int(sub["event"].iloc[i]) == 1:
            pulse = synth_tone(1200.0, min(0.01, DUR_PER_DAY))
            y[: len(pulse)] += 0.8 * pulse

        chunks.append(y)

    y = np.concatenate(chunks)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95  # normalize to avoid clipping
    return y


def run_station(tag, in_path, years):
    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    for y in years:
        sub = df[
            (df["date"] >= f"{y}-01-01") & (df["date"] <= f"{y}-12-31")
        ].reset_index(drop=True)

        for sys in ["S0", "S1", "S2"]:
            audio = sonify(sub, sys)
            out = OUTDIR / f"{tag}_{sys}_{y}_tmax.wav"
            sf.write(out, audio, SR)
            print("Wrote", out.name, "sec=", round(len(audio) / SR, 2))


def main():
    years = [1976, 1998, 2022]

    run_station(
        "NYC",
        "/Users/lunali/Desktop/data/processed/NYC_labeled.csv",
        years,
    )
    run_station(
        "PHX",
        "/Users/lunali/Desktop/data/processed/PHX_labeled.csv",
        years,
    )


if __name__ == "__main__":
    main()
