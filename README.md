# Auditory Climate Chart — Reproducible Sonification Pipeline

## Project overview
This repository implements a reproducible *auditory climate chart* pipeline that sonifies NOAA GHCN-Daily station TMAX/TMIN data and reproduces the paper’s Methods (sonification systems S0/S1/S2) and Results artifacts (RQ1 table, RQ2 sweep CSVs + figures, and aligned audio excerpts).

TMIN is downloaded for completeness / future extensions, but current sonification + labeling + evaluation use TMAX only.

## How to run
The steps below reproduce all artifacts from scratch: download data → label events → run RQ1/RQ2 → render figures → export audio samples.

### 0) Clone the repo
```bash
git clone https://github.com/yuqing-liii/Auditory-Climate-Graphs-for-Accessibility-Sonification-of-NOAA-Temperature-Trends-and-Extremes.git
cd Auditory-Climate-Graphs-for-Accessibility-Sonification-of-NOAA-Temperature-Trends-and-Extremes
```

### 1) Create environment + install dependencies
You can use either `pip` or `conda`.

#### Option A: pip
```bash
python --version
# should be 3.13.x

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

#### Option B: conda
```bash
conda create -n auditory-climate python=3.13 -y
conda activate auditory-climate
pip install -r requirements.txt
```

### 2) Get a NOAA token and set it as an environment variable
This project downloads station-level daily climate data via the NOAA NCEI / GHCN-Daily (CDO) API. The fetch script reads the token from an environment variable and sends it as the CDO API token header.

Request a token (NOAA NCEI):
```text
https://www.ncei.noaa.gov/cdo-web/token
```

Set it as an environment variable named `NOAA_TOKEN` (do not commit tokens to GitHub):

macOS / Linux:
```bash
export NOAA_TOKEN="YOUR_TOKEN_HERE"
echo $NOAA_TOKEN
```

Windows (PowerShell):
```powershell
# current session (takes effect immediately)
$env:NOAA_TOKEN="YOUR_TOKEN_HERE"
echo $env:NOAA_TOKEN
```

Optional: choose where generated files are written by setting `ACC_DATA_DIR`.
If set, scripts will write to `<ACC_DATA_DIR>/processed`, `<ACC_DATA_DIR>/results`, and `<ACC_DATA_DIR>/audio_samples`.
If not set, scripts default to writing under `<repo>/data/`.
```bash
export ACC_DATA_DIR="/path/to/workdir"
```

### 3) Download station data (CSV)
Run the fetch script to download station data from NOAA and save it locally.
```bash
python fetch_ghcnd_tmax_tmin.py
```

Expected outputs (created locally, not committed):
- `data/processed/NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv`
- `data/processed/Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv`

> Note (units): the fetch script converts GHCN-Daily TMAX/TMIN values from tenths of °C to °C by default.  
> If you need to override the scaling, set `GHCND_VALUE_SCALE` (e.g., `export GHCND_VALUE_SCALE="1.0"`).

### 4) Generate event labels (heatwave days)
This step creates a reproducible binary label `event ∈ {0,1}` using the paper’s rule (monthly 95th percentile threshold per station, with a minimum consecutive run length).
```bash
python label_heatwaves.py
```

Expected outputs:
- `data/processed/NYC_labeled.csv`
- `data/processed/PHX_labeled.csv`

### 5) (Optional) Sanity-check the prepared inputs
```bash
python check_data.py
```

### 6) Reproduce RQ1 (system-comparison table)
This script computes the RQ1 metrics table used in the paper (including max-F1 and related metrics).
```bash
python rq1_metrics_maxf1.py
```

Note: the CSV uses column name `f1_max` for max-F1.

Expected outputs:
- `data/results/RQ1_metrics_by_year.csv`

### 7) Reproduce RQ2 (parameter sweeps → CSVs)
This runs one-factor-at-a-time parameter sweeps (smoothing / playback speed / gamma) and writes sweep results as CSV files.
```bash
python rq2_sweep.py
```

Note: sweep CSVs use column name `max_f1` for max-F1.

Expected outputs:
- `data/results/RQ2_sweep_smooth.csv`
- `data/results/RQ2_sweep_speed.csv`
- `data/results/RQ2_sweep_gamma.csv`

### 8) Render RQ2 figures from sweep CSVs
Generate paper figures (PNG/PDF) from the sweep CSV outputs.
```bash
python rq2_make_figures.py
python plot_rq2_rmse_speed_gamma.py
```

Expected outputs:
- `data/results/figures_rq2/` (PNG + PDF)
  - e.g., `RQ2_smooth_RMSE.png`, `RQ2_smooth_AUC.pdf`, `RQ2_speed_maxF1.png`, `RQ2_gamma_RMSE.pdf`, etc.

### 9) Export audio samples (WAV) aligned with evaluated slices
This generates qualitative audio excerpts aligned with the evaluated station–year slices (1990/2016/2023).
```bash
python make_audio_samples.py
```

Audio samples are qualitative demos aligned by station-year slice; parameter settings follow `make_audio_samples.py` defaults (`dur_per_day=0.03`, `smooth_days=7`). RQ2 uses a different synthesis setting for sweeps.

Expected outputs:
- `data/audio_samples/` (WAV)
  - e.g., `NYC_S1_1990_tmax_d0.03_smooth7.wav`, `PHX_S2_2023_tmax_d0.03_smooth7.wav`, etc.

---

## Environment
Tested environment (used to produce the paper artifacts):
- OS: macOS (should also work on Linux/Windows with minor path/env differences)
- Python: 3.13.x
- Hardware: CPU-only (no GPU required)
- Dependency management: pip/conda

---

## Install dependencies
Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

If you prefer conda, create an environment first (see “How to run”).

---

## Data

### Data source
Daily station data is obtained from NOAA NCEI via the GHCN-Daily / CDO API. You need a personal API token.

Token page:
```text
https://www.ncei.noaa.gov/cdo-web/token
```

API documentation:
```text
https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
```

### Where data and outputs are stored (not tracked by Git)
To keep the repository lightweight and avoid committing large datasets, **downloaded/derived CSV files and generated artifacts are not tracked**.

By default, scripts write to:
- `data/processed/` — downloaded station CSVs and labeled inputs
- `data/results/` — RQ1/RQ2 CSV outputs and figures (`data/results/figures_rq2/`)
- `data/audio_samples/` — exported WAV samples

You can override the base directory by setting `ACC_DATA_DIR` (see Step 2). Scripts create output directories automatically if missing.

---

## Reproduce paper artifacts
This repo reproduces three artifact types referenced in the paper.

### RQ1 table (CSV)
Command:
```bash
python rq1_metrics_maxf1.py
```

Output:
- `data/results/RQ1_metrics_by_year.csv`

### RQ2 sweep CSV + figures
Commands:
```bash
python rq2_sweep.py
python rq2_make_figures.py
python plot_rq2_rmse_speed_gamma.py
```

Outputs:
- `data/results/RQ2_sweep_smooth.csv`
- `data/results/RQ2_sweep_speed.csv`
- `data/results/RQ2_sweep_gamma.csv`
- `data/results/figures_rq2/` (PNG + PDF)

### Audio samples (WAV)
Command:
```bash
python make_audio_samples.py
```

Outputs:
- `data/audio_samples/` (WAV)

---


## Repo structure

### Tracked by Git

```text
.
├── README.md
├── requirements.txt
├── fetch_ghcnd_tmax_tmin.py        # download NOAA station data (requires NOAA_TOKEN)
├── label_heatwaves.py              # label heatwave events (writes NYC_labeled.csv / PHX_labeled.csv)
├── check_data.py                   # optional sanity checks for downloaded station CSVs
├── rq1_metrics_maxf1.py            # RQ1 metrics table (writes RQ1_metrics_by_year.csv)
├── rq2_sweep.py                    # RQ2 parameter sweeps (writes RQ2_sweep_{speed,gamma,smooth}.csv)
├── rq2_make_figures.py             # renders RQ2 figures from sweep CSVs (writes data/results/figures_rq2/)
├── plot_rq2_rmse_speed_gamma.py    # additional RMSE plots for speed/gamma (also writes figures_rq2/)
└── make_audio_samples.py           # exports aligned WAV samples (writes data/audio_samples/)
```

### Generated locally (not tracked)

```text
data/
├── processed/                  # station CSVs + labeled inputs
│   ├── NYC_CentralPark_USW00094728_1970-01-01_2024-12-31_TMAX_TMIN.csv
│   ├── Phoenix_Airport_USW00023183_1970-01-01_2024-12-31_TMAX_TMIN.csv
│   ├── NYC_labeled.csv
│   └── PHX_labeled.csv
├── results/                    # numeric artifacts (CSV)
│   ├── RQ1_metrics_by_year.csv
│   ├── RQ2_sweep_speed.csv
│   ├── RQ2_sweep_gamma.csv
│   ├── RQ2_sweep_smooth.csv
│   └── figures_rq2/            # figures exported as PNG + PDF
│       ├── RQ2_smooth_RMSE.{png,pdf}
│       ├── RQ2_smooth_AUC.{png,pdf}
│       ├── RQ2_smooth_maxF1.{png,pdf}
│       ├── RQ2_speed_AUC.{png,pdf}
│       ├── RQ2_speed_maxF1.{png,pdf}
│       ├── RQ2_gamma_AUC.{png,pdf}
│       ├── RQ2_gamma_maxF1.{png,pdf}
│       ├── RQ2_speed_RMSE.{png,pdf}
│       └── RQ2_gamma_RMSE.{png,pdf}
└── audio_samples/              # qualitative artifacts (WAV)
    ├── NYC_S0_1990_tmax_d0.03_smooth7.wav
    ├── NYC_S1_1990_tmax_d0.03_smooth7.wav
    ├── ...
    └── PHX_S2_2023_tmax_d0.03_smooth7.wav
```

---

## Notes / Troubleshooting
- If you see authentication errors when downloading NOAA data, confirm NOAA_TOKEN is set in the same terminal session:
  ```bash
  echo $NOAA_TOKEN
  ```

- Optional: change where generated files are written by setting ACC_DATA_DIR.
  If set, scripts will write to <ACC_DATA_DIR>/processed, <ACC_DATA_DIR>/results, and <ACC_DATA_DIR>/audio_samples instead of <repo>/data/...:
  ```bash
  export ACC_DATA_DIR="/path/to/workdir"
  ```

- If you want a clean rebuild, delete the generated folders (scripts will recreate them):
  ```bash
  rm -rf data/processed data/results data/audio_samples
  ```
  (If you set ACC_DATA_DIR, delete the corresponding folders under that directory instead.)

- Reproducibility: where randomness is used (e.g., train/test splits), scripts use fixed seeds / random_state settings to keep outputs stable.
