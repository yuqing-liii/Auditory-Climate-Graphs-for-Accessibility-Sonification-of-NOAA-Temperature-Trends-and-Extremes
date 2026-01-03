import os
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

# GHCN-Daily TMAX/TMIN values are commonly stored in tenths of °C.
# Default converts 0.1°C -> °C. Override if needed (e.g., export GHCND_VALUE_SCALE="1.0").
VALUE_SCALE = float(os.environ.get("GHCND_VALUE_SCALE", "0.1"))

def get_base_dir() -> Path:
    """
    Base working directory for generated files.
    Override with ACC_DATA_DIR. Defaults to <repo>/data.
    """
    repo_root = Path(__file__).resolve().parent
    base = os.environ.get("ACC_DATA_DIR")
    return (Path(base).expanduser() if base else (repo_root / "data"))



def fetch_year_with_retry(datasetid: str, stationid: str, datatypeid: str,
                          start: str, end: str, token: str,
                          max_retries: int = 8, base_wait: float = 1.5) -> pd.DataFrame:
    """
    Fetch one year of daily data with robust retry logic.
    Retries on 503/429 and common transient network errors.
    """
    headers = {"token": token}
    params = {
        "datasetid": datasetid,
        "stationid": stationid,
        "datatypeid": datatypeid,
        "startdate": start,
        "enddate": end,
        "units": "metric",
        "limit": 1000,  # one year of daily data < 1000
    }

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(BASE, headers=headers, params=params, timeout=60)

            # Success
            if r.status_code == 200:
                js = r.json()
                return pd.DataFrame(js.get("results", []))

            # Retryable server overload / rate limit
            if r.status_code in (429, 503):
                wait = base_wait * (2 ** attempt)
                print(f"  {datatypeid} {start[:4]}: HTTP {r.status_code} (retry {attempt+1}/{max_retries}), waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            # Other errors: print snippet and raise
            print(f"  {datatypeid} {start[:4]}: Request failed HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = base_wait * (2 ** attempt)
            print(f"  {datatypeid} {start[:4]}: network error ({e.__class__.__name__}) retry {attempt+1}/{max_retries}, waiting {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed after retries: {stationid} {datatypeid} {start}..{end}")


def fetch_all_years(datasetid: str, stationid: str, datatypeid: str,
                    start_year: int, end_year: int, token: str) -> pd.DataFrame:
    parts = []
    for y in range(start_year, end_year + 1):
        start = f"{y}-01-01"
        end = f"{y}-12-31"

        df_y = fetch_year_with_retry(datasetid, stationid, datatypeid, start, end, token)

        if not df_y.empty:
            parts.append(df_y)

        # gentle pacing between years
        time.sleep(0.35)

        if y % 5 == 0:
            print(f"  {datatypeid}: fetched up to {y}")

    if parts:
        return pd.concat(parts, ignore_index=True)

    return pd.DataFrame(columns=["date", "datatype", "station", "attributes", "value"])


def tidy_station(stationid: str, token: str, start_year: int = 1970, end_year: int = 2024) -> pd.DataFrame:
    datasetid = "GHCND"
    df_max = fetch_all_years(datasetid, stationid, "TMAX", start_year, end_year, token)
    df_min = fetch_all_years(datasetid, stationid, "TMIN", start_year, end_year, token)

    df = pd.concat([df_max, df_min], ignore_index=True)
    df = df[["date", "datatype", "value"]].copy()
    df["date"] = df["date"].str.slice(0, 10)

    wide = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"TMAX": "tmax_raw", "TMIN": "tmin_raw"})

    # Convert to Celsius using VALUE_SCALE (default 0.1 for tenths-of-°C inputs)
    wide["tmax_c"] = wide["tmax_raw"] * VALUE_SCALE
    wide["tmin_c"] = wide["tmin_raw"] * VALUE_SCALE

    wide = wide.sort_values("date").reset_index(drop=True)
    return wide


def main():
    token = os.environ.get("NOAA_TOKEN")
    if not token:
        raise SystemExit("NOAA_TOKEN not set. Set it first, e.g. export NOAA_TOKEN='YOUR_TOKEN' (macOS/Linux) or $env:NOAA_TOKEN='YOUR_TOKEN' (PowerShell).")

    start = "1970-01-01"
    end = "2024-12-31"

    print(f"Value scale: {VALUE_SCALE}  (set GHCND_VALUE_SCALE to override)")

    stations = {
        "NYC_CentralPark_USW00094728": "GHCND:USW00094728",
        "Phoenix_Airport_USW00023183": "GHCND:USW00023183",
    }

    out_dir = get_base_dir() / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, stationid in stations.items():
        print(f"\nFetching {name} ({stationid}) {start}..{end}")
        wide = tidy_station(stationid, token, start_year=1970, end_year=2024)

        out_path = out_dir / f"{name}_{start}_{end}_TMAX_TMIN.csv"
        wide.to_csv(out_path, index=False)

        print(f"Saved -> {out_path.as_posix()}")
        print(f"Rows: {len(wide)} | Columns: {list(wide.columns)}")


if __name__ == "__main__":
    main()
