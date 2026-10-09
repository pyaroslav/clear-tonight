"""Tonight's corrected forecast for a place: P(clear) per night hour, the best window, and the Moon.

Uses the same kind of input the model was trained on: the forecast issued ~24 h ahead
for tonight (previous-runs API), so training and live use match.
"""

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ephem
import numpy as np
import pandas as pd

from . import data
from .data import _get


def tonight_rows(lat: float, lon: float, night: date) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "timezone": "auto", "models": "gfs_seamless",
              "start_date": night.isoformat(), "end_date": (night + timedelta(days=1)).isoformat(),
              "hourly": ",".join([f"{v}_previous_day1" for v in data.FC_VARS] + ["cloud_cover_previous_day2"])}
    fc = _get("https://previous-runs-api.open-meteo.com/v1/forecast", params)
    df = pd.DataFrame(fc["hourly"]).rename(columns=lambda k: k.replace("_previous_day1", "").replace("_previous_day2", "_d2"))
    df["time"] = pd.to_datetime(df["time"])
    df["truth"] = np.nan
    f = data.features(df)
    f = f[f["night"] == night]
    return f.dropna(subset=data.FEATURES), fc.get("utc_offset_seconds", 0)


def moon(lat: float, lon: float, night: date, utc_offset: int) -> dict:
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    local_9pm = datetime(night.year, night.month, night.day, 21) - timedelta(seconds=utc_offset)
    obs.date = local_9pm
    m = ephem.Moon(obs)

    def local(t):
        return (ephem.Date(t).datetime() + timedelta(seconds=utc_offset)).strftime("%H:%M")

    out = {"illumination": round(m.phase), "altitude_9pm": round(math.degrees(m.alt))}
    try:
        out["rise"] = local(obs.next_rising(ephem.Moon()))
        out["set"] = local(obs.next_setting(ephem.Moon()))
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        pass
    return out


def best_window(hours: list[dict], threshold: float = 0.5) -> dict | None:
    """Longest run of consecutive hours with P(clear) >= threshold (at least 2 hours)."""
    best, cur = None, []
    for h in hours + [None]:
        if h and h["p_clear"] >= threshold:
            cur.append(h)
            continue
        if len(cur) >= 2 and (best is None or len(cur) > len(best) or
                              (len(cur) == len(best) and np.mean([c["p_clear"] for c in cur]) > np.mean([b["p_clear"] for b in best]))):
            best = cur
        cur = []
    if not best:
        return None
    end = (datetime.strptime(best[-1]["time"], "%H:%M") + timedelta(hours=1)).strftime("%H:%M")
    return {"start": best[0]["time"], "end": end, "hours": len(best),
            "p_mean": round(float(np.mean([b["p_clear"] for b in best])), 2)}


def predict_place(lat: float, lon: float, history: pd.DataFrame, night: date | None = None, model=None) -> dict:
    from .backtest import tabpfn_v2
    night = night or date.today()
    rows, offset = tonight_rows(lat, lon, night)
    if model is None:
        model = tabpfn_v2().fit(history[data.FEATURES], history["clear"].astype(int))
    p = model.predict_proba(rows[data.FEATURES])[:, 1] if len(rows) else []
    hours = [{"time": t.strftime("%H:%M"), "p_clear": round(float(pi), 2),
              "app_cloud_cover": int(cc)} for t, pi, cc in zip(rows["time"], p, rows["cloud_cover"])]
    app_says_clear = [h for h in hours if h["app_cloud_cover"] <= data.CLEAR]
    return {
        "night": night.isoformat(),
        "hours": hours,
        "window": best_window(hours),
        "app_clear_hours": len(app_says_clear),
        "moon": moon(lat, lon, night, offset),
        "trained_on_nights": int(history["night"].nunique()),
    }


def main():
    """Nightly job: one JSON for the static site."""
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "cities": {}}
    for key, c in data.CITIES.items():
        hist = data.build(key)
        out["cities"][key] = {"name": c["name"], "lat": c["lat"], "lon": c["lon"],
                              **predict_place(c["lat"], c["lon"], hist)}
        print(key, out["cities"][key]["window"], flush=True)
    site = Path(__file__).resolve().parents[1] / "site"
    (site / "tonight.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
