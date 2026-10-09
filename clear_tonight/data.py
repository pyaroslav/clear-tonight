"""Open-Meteo data: what the weather model said yesterday about tonight, and what tonight turned out to be.

- Forecast: previous-runs API, `*_previous_day1` = the forecast issued ~24 h ahead (GFS),
  plus `cloud_cover_previous_day2` (issued ~48 h ahead) so we can see how much the
  forecast changed between runs.
- Truth: ERA5 reanalysis cloud cover from the archive API. ERA5 is a model-assisted
  reconstruction, not a person looking up; it's the best consistent "what happened"
  available everywhere. It lags real time by about 5 days.

Weather data by Open-Meteo.com, CC BY 4.0.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data"
FIRST_DAY = date(2024, 2, 1)  # day-ahead GFS archive starts here

CITIES = {
    "seattle":     {"name": "Seattle, WA",     "lat": 47.61, "lon": -122.33},
    "minneapolis": {"name": "Minneapolis, MN", "lat": 44.98, "lon": -93.27},
    "chicago":     {"name": "Chicago, IL",     "lat": 41.88, "lon": -87.63},
    "denver":      {"name": "Denver, CO",      "lat": 39.74, "lon": -104.99},
    "austin":      {"name": "Austin, TX",      "lat": 30.27, "lon": -97.74},
    "london":      {"name": "London, UK",      "lat": 51.51, "lon": -0.13},
}

# Cloud layers (low/mid/high) aren't kept in the day-ahead archive, so total cover only.
FC_VARS = ["cloud_cover", "relative_humidity_2m", "temperature_2m", "dew_point_2m", "wind_speed_10m",
           "precipitation"]

NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4]  # local time; hours after midnight belong to the previous evening
CLEAR = 25  # % cloud cover at or below which an hour counts as "clear enough to bother"


def _get(url: str, params: dict, attempts: int = 5) -> dict:
    """GET with backoff on rate limits and on flaky networks (CI runners see TLS timeouts)."""
    for attempt in range(attempts):
        try:
            r = httpx.get(url, params=params, timeout=120)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _cached(name: str, fetch) -> dict:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    d = fetch()
    p.write_text(json.dumps(d))
    return d


def history(city: str, end: date, lat: float | None = None, lon: float | None = None) -> pd.DataFrame:
    """Hourly forecast-vs-truth table for a named city (or any lat/lon), FIRST_DAY..end (local time)."""
    c = CITIES.get(city) or {"lat": lat, "lon": lon}
    common = {"latitude": c["lat"], "longitude": c["lon"], "timezone": "auto",
              "start_date": FIRST_DAY.isoformat(), "end_date": end.isoformat()}
    fc = _cached(f"{city}-fc-{end}", lambda: _get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {**common, "models": "gfs_seamless",
         "hourly": ",".join([f"{v}_previous_day1" for v in FC_VARS] + ["cloud_cover_previous_day2"])}))
    obs = _cached(f"{city}-obs-{end}", lambda: _get(
        "https://archive-api.open-meteo.com/v1/archive", {**common, "hourly": "cloud_cover"}))

    df = pd.DataFrame(fc["hourly"]).rename(columns=lambda k: k.replace("_previous_day1", "").replace("_previous_day2", "_d2"))
    df["truth"] = pd.DataFrame(obs["hourly"])["cloud_cover"].values if len(obs["hourly"]["time"]) == len(df) else \
        pd.DataFrame(obs["hourly"]).set_index("time").reindex(df["time"])["cloud_cover"].values
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per night hour. Everything here is known the evening before."""
    d = df.copy()
    d["hour"] = d["time"].dt.hour
    d = d[d["hour"].isin(NIGHT_HOURS)].copy()
    # the "night of" date: hours after midnight belong to the evening before
    d["night"] = (d["time"] - pd.to_timedelta((d["hour"] < 12).astype(int), unit="D")).dt.date
    d["hour_of_night"] = (d["hour"] - 20) % 24
    doy = d["time"].dt.dayofyear
    d["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["dewpoint_spread"] = d["temperature_2m"] - d["dew_point_2m"]
    d["run_change"] = d["cloud_cover"] - d["cloud_cover_d2"]          # did the forecast flip between runs?
    night_mean = d.groupby("night")["cloud_cover"].transform("mean")
    d["night_mean_fc"] = night_mean                                     # the forecast's view of the whole night
    d["neighbour_spread"] = d.groupby("night")["cloud_cover"].transform("std").fillna(0)
    d["clear"] = (d["truth"] <= CLEAR).astype("float")
    d.loc[d["truth"].isna(), "clear"] = np.nan
    return d


FEATURES = ["cloud_cover", "relative_humidity_2m", "dewpoint_spread", "wind_speed_10m", "precipitation",
            "cloud_cover_d2", "run_change", "night_mean_fc", "neighbour_spread",
            "hour_of_night", "season_sin", "season_cos"]


def build(city: str, end: date | None = None, lat: float | None = None, lon: float | None = None) -> pd.DataFrame:
    end = end or (date.today() - timedelta(days=7))  # ERA5 lags ~5 days
    return features(history(city, end, lat, lon)).dropna(subset=FEATURES)
