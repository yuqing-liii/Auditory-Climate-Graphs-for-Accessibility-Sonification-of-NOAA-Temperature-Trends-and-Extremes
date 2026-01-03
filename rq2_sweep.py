import os
import math
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

# =========================
# User config
# =========================
DATA_DIR = Path.home() / "Desktop" / "data"
PROCESSED = DATA_DIR / "processed"
RESULTS = DATA_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Your labeled data (already prepared)
STATIONS = {
    "NYC": PROCESSED / "NYC_labeled.csv",
    "PHX": PROCESSED / "PHX_labeled.csv",
}

# Pick years with enough events (avoid all-0)
YEARS = [1990, 2016, 2023]

SYSTEMS = ["S0", "S1", "S2"]  # baseline / narrative / event-enhanced

# Audio synthesis
SR = 22050
F_MIN = 220.0
F_MAX = 880.0

# RQ2 baseline (others stay fixed while sweeping one factor)
BASE_SECONDS_PER_DAY = 0.06
BASE_GAMMA = 0.70
BASE_SMOOTH_DAYS = 31

# Sweep grids (one-factor-at-a-time)
SWEEP_SECONDS = [0.03, 0.05, 0.06, 0.08, 0.12]
SWEEP_GAMMA = [1.00, 0.85, 0.70, 0.55, 0.40]
SWEEP_SMOOTH = [1, 7, 15, 31, 61]


# =========================
# Sonification core
# =========================
def minmax_map(x, xmin, xmax, a, b):
    if xmax <= xmin:
        return (a + b) / 2.0
    return a + (x - xmin) * (b - a) / (xmax - xmin)

def synth_day_sine(freq_hz: float, amp: float, seconds: float, sr: int) -> np.ndarray:
    n = max(16, int(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr
    y = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    # short fade to avoid clicks
    fade = max(4, int(0.01 * n))
    w = np.ones(n, dtype=np.float32)
    w[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    w[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return y * w

def add_event_pulse(y: np.ndarray, sr: int, strength: float = 0.25) -> np.ndarray:
    """Gentle pulse overlay (S2)"""
    y = y.copy()
    n = len(y)
    pulse_len = max(8, int(0.01 * sr))  # 10 ms
    pulse_len = min(pulse_len, n)
    pulse = np.zeros(pulse_len, dtype=np.float32)
    # a short "tick": high-freq damped sinus
    t = np.arange(pulse_len, dtype=np.float32) / sr
    pulse = (np.sin(2 * np.pi * 1800.0 * t) * np.exp(-t * 120.0)).astype(np.float32)
    y[:pulse_len] += strength * pulse
    return y

def apply_gamma_compression(amp: np.ndarray, gamma: float) -> np.ndarray:
    """amp in [0,1] -> compressed"""
    amp = np.clip(amp, 0.0, 1.0)
    return np.power(amp, gamma, dtype=np.float32)

def sonify_daily_series(temp: np.ndarray,
                        event: np.ndarray,
                        system: str,
                        seconds_per_day: float,
                        smooth_days: int,
                        gamma: float,
                        sr: int = SR) -> list:
    """
    Return list of per-day audio segments (np arrays).
    - S0: raw pitch mapping
    - S1: smoothed pitch mapping
    - S2: smoothed pitch + gentle event pulse
    """
    # smoothing for S1/S2 only
    if system in ("S1", "S2") and smooth_days > 1:
        temp_used = pd.Series(temp).rolling(smooth_days, min_periods=1, center=True).mean().to_numpy()
    else:
        temp_used = temp

    # map to pitch range
    xmin, xmax = float(np.nanmin(temp_used)), float(np.nanmax(temp_used))
    freqs = np.array([minmax_map(x, xmin, xmax, F_MIN, F_MAX) for x in temp_used], dtype=np.float32)

    # amplitude: normalize temp to [0,1], then gamma compress (dynamic range control)
    if xmax > xmin:
        amp0 = (temp_used - xmin) / (xmax - xmin)
    else:
        amp0 = np.zeros_like(temp_used, dtype=np.float32)
    amp = apply_gamma_compression(amp0.astype(np.float32), gamma)
    # keep within a comfortable loudness
    amp = 0.15 + 0.75 * amp  # [0.15, 0.90]

    segs = []
    for f, a, ev in zip(freqs, amp, event):
        y = synth_day_sine(float(f), float(a), seconds_per_day, sr)
        if system == "S2" and int(ev) == 1:
            y = add_event_pulse(y, sr, strength=0.20)  # gentle
        segs.append(y)
    return segs, xmin, xmax


# =========================
# Metrics
# =========================
def peak_freq(seg: np.ndarray, sr: int) -> float:
    """Simple pitch proxy: FFT peak frequency."""
    x = seg.astype(np.float32)
    x = x - x.mean()
    n = len(x)
    if n < 16:
        return float("nan")
    win = np.hanning(n).astype(np.float32)
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    k = int(np.argmax(mag))
    return float(k * sr / n)

def temp_from_pitch(freq_hat: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    # inverse map from [F_MIN,F_MAX] to [xmin,xmax]
    freq_hat = np.clip(freq_hat, F_MIN, F_MAX)
    if F_MAX <= F_MIN:
        return np.full_like(freq_hat, (xmin + xmax) / 2.0, dtype=np.float32)
    return xmin + (freq_hat - F_MIN) * (xmax - xmin) / (F_MAX - F_MIN)

def features(seg: np.ndarray, sr: int) -> np.ndarray:
    """Small feature set for event detection."""
    x = seg.astype(np.float32)
    # energy
    energy = float(np.mean(x**2))
    # zero-crossing rate
    zcr = float(((x[:-1] * x[1:]) < 0).mean()) if len(x) > 2 else 0.0
    # spectral centroid & rolloff
    X = np.fft.rfft(x * np.hanning(len(x)).astype(np.float32))
    mag = np.abs(X) + 1e-12
    freqs = np.linspace(0, sr / 2, len(mag), dtype=np.float32)
    centroid = float((freqs * mag).sum() / mag.sum())
    cumsum = np.cumsum(mag)
    roll_k = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff = float(freqs[min(roll_k, len(freqs)-1)])
    return np.array([energy, zcr, centroid, rolloff], dtype=np.float32)

def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, best_thr)."""
    thrs = np.linspace(0.05, 0.95, 19)
    best = (-1.0, 0.5)
    for t in thrs:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best[0]:
            best = (float(f1), float(t))
    return best

def eval_one_setting(df_year: pd.DataFrame,
                     tag: str,
                     year: int,
                     system: str,
                     seconds_per_day: float,
                     gamma: float,
                     smooth_days: int) -> dict:
    sub = df_year.copy()
    temp = sub["tmax_c"].to_numpy(dtype=np.float32)
    y_event = sub["event"].astype(int).to_numpy()

    segs, xmin, xmax = sonify_daily_series(
        temp=temp,
        event=y_event,
        system=system,
        seconds_per_day=seconds_per_day,
        smooth_days=smooth_days,
        gamma=gamma,
        sr=SR,
    )

    # fidelity: pitch -> temp_hat
    freq_hat = np.array([peak_freq(s, SR) for s in segs], dtype=np.float32)
    temp_hat = temp_from_pitch(freq_hat, xmin, xmax)

    # target temp (apply SAME smoothing when system uses smoothing)
    temp_true = temp
    if system in ("S1", "S2") and smooth_days > 1:
        temp_true = pd.Series(temp_true).rolling(smooth_days, min_periods=1, center=True).mean().to_numpy(dtype=np.float32)

    rmse = float(np.sqrt(np.mean((temp_hat - temp_true) ** 2)))
    pr = float(pearsonr(temp_hat, temp_true)[0])
    sr = float(spearmanr(temp_hat, temp_true)[0])

    # detectability: audio features -> event
    n_pos = int(y_event.sum())
    n_neg = int((1 - y_event).sum())
    note = ""

    if n_pos == 0 or n_neg == 0:
        # cannot train classifier
        auc = float("nan")
        maxf1 = float("nan")
        best_thr = float("nan")
        note = "single_class_year"
    else:
        X = np.vstack([features(s, SR) for s in segs])
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_event, test_size=0.30, random_state=42, stratify=y_event
        )
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]

        auc = float(roc_auc_score(yte, prob))
        maxf1, best_thr = best_f1_threshold(yte, prob)

    return {
        "station": tag,
        "year": year,
        "system": system,
        "seconds_per_day": seconds_per_day,
        "gamma": gamma,
        "smooth_days": smooth_days,
        "rmse": rmse,
        "pearson_r": pr,
        "spearman_r": sr,
        "auc": auc,
        "max_f1": maxf1,
        "best_thr": best_thr,
        "event_rate": float(y_event.mean()),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "note": note,
    }


# =========================
# RQ2 sweeps (one-factor-at-a-time)
# =========================
def run_sweep(df: pd.DataFrame, tag: str, years, systems, mode: str) -> pd.DataFrame:
    rows = []
    for year in years:
        df_year = df[df["date"].dt.year == year].copy()
        if df_year.empty:
            continue

        if mode == "speed":
            grid = [(s, BASE_GAMMA, BASE_SMOOTH_DAYS) for s in SWEEP_SECONDS]
        elif mode == "gamma":
            grid = [(BASE_SECONDS_PER_DAY, g, BASE_SMOOTH_DAYS) for g in SWEEP_GAMMA]
        elif mode == "smooth":
            grid = [(BASE_SECONDS_PER_DAY, BASE_GAMMA, w) for w in SWEEP_SMOOTH]
        else:
            raise ValueError("unknown mode")

        for system in systems:
            for seconds_per_day, gamma, smooth_days in grid:
                out = eval_one_setting(
                    df_year=df_year,
                    tag=tag,
                    year=year,
                    system=system,
                    seconds_per_day=float(seconds_per_day),
                    gamma=float(gamma),
                    smooth_days=int(smooth_days),
                )
                rows.append(out)
                print("done", mode, tag, year, system, "sec", seconds_per_day, "gamma", gamma, "smooth", smooth_days)

    return pd.DataFrame(rows)


def main():
    all_outputs = {}

    for tag, path in STATIONS.items():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        # sanity: must have columns
        need = {"date", "tmax_c", "event"}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"{tag} missing columns: {miss} in {path}")

        print(f"\n=== Station {tag} === {path}")
        out_speed = run_sweep(df, tag, YEARS, SYSTEMS, mode="speed")
        out_gamma = run_sweep(df, tag, YEARS, SYSTEMS, mode="gamma")
        out_smooth = run_sweep(df, tag, YEARS, SYSTEMS, mode="smooth")

        all_outputs[(tag, "speed")] = out_speed
        all_outputs[(tag, "gamma")] = out_gamma
        all_outputs[(tag, "smooth")] = out_smooth

    # merge & save
    for mode in ["speed", "gamma", "smooth"]:
        merged = []
        for tag in STATIONS.keys():
            merged.append(all_outputs[(tag, mode)])
        res = pd.concat(merged, ignore_index=True)

        out_path = RESULTS / f"RQ2_sweep_{mode}.csv"
        res.to_csv(out_path, index=False)
        print("\nSaved:", out_path, "| rows:", len(res))

    print("\nAll done. Check:", RESULTS)


if __name__ == "__main__":
    main()
