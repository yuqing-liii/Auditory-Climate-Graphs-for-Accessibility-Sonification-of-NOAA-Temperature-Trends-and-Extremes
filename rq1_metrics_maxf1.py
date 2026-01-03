import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

# =========================
# Reproducible paths
# =========================
def get_base_dir() -> Path:
    """
    Base working directory for generated files.
    Override with ACC_DATA_DIR. Defaults to <repo>/data.
    """
    repo_root = Path(__file__).resolve().parent
    base = os.environ.get('ACC_DATA_DIR')
    return (Path(base).expanduser() if base else (repo_root / 'data'))


# =========================
# Config
# =========================
SR = 22050
DUR_PER_DAY = 0.03
FMIN, FMAX = 220.0, 880.0
SMOOTH_DAYS = 7

YEARS = [1990, 2016, 2023]

BASE_DIR = get_base_dir()
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NYC_PATH = PROCESSED_DIR / "NYC_labeled.csv"
PHX_PATH = PROCESSED_DIR / "PHX_labeled.csv"

OUT_CSV = RESULTS_DIR / "RQ1_metrics_by_year.csv"


# =========================
# Helpers: mapping + synthesis
# =========================
def normalize(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin < 1e-9:
        return np.zeros_like(x), xmin, xmax
    return (x - xmin) / (xmax - xmin), xmin, xmax


def pitch_from_temp(temp_c, xmin, xmax):
    if xmax - xmin < 1e-9:
        n = np.zeros_like(temp_c, dtype=float)
    else:
        n = (temp_c - xmin) / (xmax - xmin)
    return FMIN + n * (FMAX - FMIN)


def temp_from_pitch(freq, xmin, xmax):
    n = (freq - FMIN) / (FMAX - FMIN)
    return xmin + n * (xmax - xmin)


def synth_tone(freq, dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def sonify(sub: pd.DataFrame, system: str):
    """
    S0: tone pitch encodes tmax_c (no smoothing, no marker)
    S1: smoothing (7-day MA)
    S2: smoothing + event marker pulse at 1200Hz
    """
    temp = sub["tmax_c"].to_numpy()

    if system in ("S1", "S2"):
        temp = (
            pd.Series(temp)
            .rolling(SMOOTH_DAYS, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    _, xmin, xmax = normalize(temp)
    freqs = pitch_from_temp(temp, xmin, xmax)

    chunks = []
    for i, f in enumerate(freqs):
        y = synth_tone(f, DUR_PER_DAY)

        if system == "S2" and int(sub["event"].iloc[i]) == 1:
            pulse = synth_tone(1200.0, min(0.01, DUR_PER_DAY))
            y[: len(pulse)] += 0.8 * pulse

        chunks.append(y)

    y = np.concatenate(chunks)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return y, xmin, xmax


def segment_audio(y):
    seglen = int(SR * DUR_PER_DAY)
    n = len(y) // seglen
    y = y[: n * seglen]
    return y.reshape(n, seglen)


def peak_freq(seg):
    """Estimate dominant frequency by FFT peak."""
    w = np.hanning(len(seg))
    sp = np.fft.rfft(seg * w)
    mag = np.abs(sp)
    k = np.argmax(mag[1:]) + 1
    return k * SR / len(seg)


def features(seg):
    """
    Simple features for event detectability.
    Includes high-frequency ratio to capture S2 marker.
    """
    w = np.hanning(len(seg))
    sp = np.fft.rfft(seg * w)
    mag = np.abs(sp) + 1e-12
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)

    rms = np.sqrt(np.mean(seg**2))
    centroid = np.sum(freqs * mag) / np.sum(mag)
    hf = np.sum(mag[freqs > 1000]) / np.sum(mag)
    return np.array([rms, centroid, hf])


def max_f1_over_thresholds(y_true, prob, n_steps=99):
    """
    Return (best_f1, best_threshold, f1_at_0.5)
    """
    f1_at_05 = float(f1_score(y_true, (prob >= 0.5).astype(int), zero_division=0))

    best_f1 = -1.0
    best_t = 0.5
    ths = np.linspace(0.01, 0.99, n_steps)
    for t in ths:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return float(best_f1), float(best_t), f1_at_05


# =========================
# Evaluation
# =========================
def eval_window(df: pd.DataFrame, station_tag: str, year: int, system: str) -> dict:
    sub = df[(df["date"] >= f"{year}-01-01") & (df["date"] <= f"{year}-12-31")].reset_index(
        drop=True
    )

    # --- generate audio and segment by day
    y, xmin, xmax = sonify(sub, system)
    segs = segment_audio(y)

    # --- fidelity: pitch -> temp
    freq_hat = np.array([peak_freq(s) for s in segs])
    temp_hat = temp_from_pitch(freq_hat, xmin, xmax)

    temp_true = sub["tmax_c"].to_numpy()
    if system in ("S1", "S2"):
        temp_true = (
            pd.Series(temp_true)
            .rolling(SMOOTH_DAYS, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    rmse = float(np.sqrt(np.mean((temp_hat - temp_true) ** 2)))
    pr = float(pearsonr(temp_hat, temp_true)[0])
    sr = float(spearmanr(temp_hat, temp_true)[0])

    # --- detectability: audio features -> event label
    X = np.vstack([features(s) for s in segs])
    ylab = sub["event"].astype(int).to_numpy()
    pos = int(ylab.sum())
    neg = int(len(ylab) - pos)

    auc = float("nan")
    f1_at_05 = float("nan")
    f1_max = float("nan")
    best_t = float("nan")
    note = ""

    if pos == 0 or neg == 0:
        note = "single_class_in_year"
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, ylab, test_size=0.3, random_state=42, stratify=ylab
        )

        # Extra safety: test split can still become single-class in rare cases
        if len(set(yte.tolist())) < 2:
            note = "single_class_in_test_split"
        else:
            clf = LogisticRegression(max_iter=300)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]

            auc = float(roc_auc_score(yte, prob))
            f1_max, best_t, f1_at_05 = max_f1_over_thresholds(yte, prob, n_steps=99)

    return {
        "station": station_tag,
        "year": year,
        "system": system,
        "rmse": rmse,
        "pearson_r": pr,
        "spearman_r": sr,
        "auc": auc,
        "f1_at_0.5": f1_at_05,
        "f1_max": f1_max,
        "best_threshold": best_t,
        "event_rate": float(ylab.mean()),
        "n_pos": pos,
        "n_neg": neg,
        "note": note,
    }


def run_station(station_tag: str, path: Path, years):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for year in years:
        for sys in ["S0", "S1", "S2"]:
            out.append(eval_window(df, station_tag, year, sys))
            print("done", station_tag, year, sys)
    return out


def main():
    rows = []
    rows += run_station("NYC", NYC_PATH, YEARS)
    rows += run_station("PHX", PHX_PATH, YEARS)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)
    print(res)


if __name__ == "__main__":
    main()