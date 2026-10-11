"""Clear Tonight, run locally for any spot.

    python -m clear_tonight --lat 44.98 --lon -93.27            # tonight, corrected forecast + Gemma's plan
    python -m clear_tonight --lat 44.98 --lon -93.27 --photo sky.jpg   # log what the sky actually looked like

Everything runs on this machine: TabPFN v2 for the correction, Gemma 4 through Ollama
for the words and for reading sky photos. The only network calls are to Open-Meteo.
"""

import argparse
import base64
import csv
import json
import os
from datetime import date, datetime
from pathlib import Path

import httpx
import numpy as np

from . import data
from .predict import predict_place

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
MODEL = os.environ.get("CT_MODEL", "gemma4:e4b")
SKY_LOG = data.CACHE / "sky_log.csv"


def gemma(prompt: str, image: bytes | None = None, schema: dict | None = None) -> str:
    msg = {"role": "user", "content": prompt}
    if image:
        msg["images"] = [base64.b64encode(image).decode()]
    body = {"model": MODEL, "messages": [msg], "stream": False, "think": False,
            "options": {"temperature": 0.2}}
    if schema:
        body["format"] = schema
    r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def facts(result: dict) -> list[str]:
    """Plain statements computed in code. Gemma only puts them into words; it never reads the numbers itself."""
    hours = result["hours"]
    if not hours:
        return ["There is no forecast for tonight yet."]
    out = []
    w = result["window"]
    if w:
        out.append(f"Verdict: go. Best window {w['start']} to {w['end']}, about {round(w['p_mean'] * 100)}% chance it is really clear.")
    else:
        out.append("Verdict: probably not worth going out tonight.")
    top = sorted(hours, key=lambda h: -h["p_clear"])[:2]
    out.append("Best hours: " + " and ".join(f"{h['time']} ({round(h['p_clear'] * 100)}%)" for h in sorted(top, key=lambda h: h["time"])) + ".")
    app = sum(h["app_cloud_cover"] <= data.CLEAR for h in hours) / len(hours)
    ours = sum(h["p_clear"] for h in hours) / len(hours)
    if app - ours > 0.25:
        out.append("The weather app is MORE optimistic than the corrected forecast: it shows clear sky, but at this spot "
                   "that kind of forecast often ends up cloudier. Expect some cloud.")
    elif ours - app > 0.25:
        out.append("The corrected forecast is MORE optimistic than the weather app: the app shows cloud, but at this spot "
                   "that kind of forecast often clears.")
    else:
        out.append("The weather app and the corrected forecast broadly agree.")
    m = result["moon"]
    lit = m["illumination"]
    out.append(f"The Moon is {lit}% lit" + (" (a dark sky, good for faint stars)." if lit < 15 else
                                            " (bright; faint stars will be washed out)." if lit > 85 else "."))
    return out


def plan(place: str, result: dict) -> str:
    return gemma(
        f"Someone near {place} wants to know whether to go outside tonight to look at the stars. "
        "Rewrite these facts as 2-3 short, friendly sentences and add one practical tip (warm layers, let your eyes "
        "adjust for 20 minutes, get away from streetlights). Do not change or add any times, numbers or claims. "
        "No markdown, no lists.\n\n" + "\n".join(facts(result)))


def taken_at(path: Path) -> str:
    """When the photo was taken (EXIF), not when it was rated; falls back to now."""
    try:
        from PIL import Image
        stamp = Image.open(path).getexif().get(306)  # DateTime
        return datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S").isoformat(timespec="minutes")
    except Exception:
        return datetime.now().isoformat(timespec="minutes")


DISAGREE = 50  # percentage points between a photo's rating and what else we know before it needs a second look


def check_rating(cover: int, same_minute: list[int], model_cover: float | None) -> str:
    """A photo rating is only ground truth if it agrees with the other evidence.

    Night mode can turn a clear twilight sky flat grey, and Gemma then reads it as overcast with high
    confidence (field/2026-10-10). So a rating far from both the other photos taken around the same time
    and the weather model's cloud cover for that hour is held back for review instead of being trusted.
    """
    reasons = []
    if same_minute and abs(cover - float(np.median(same_minute))) >= DISAGREE:
        reasons.append(f"other photos ~{round(float(np.median(same_minute)))}%")
    if model_cover is not None and abs(cover - model_cover) >= DISAGREE:
        reasons.append(f"weather model {round(model_cover)}%")
    if not reasons:
        return "ok"
    # with no other photos, disagreeing with the model alone is enough; with photos, both must disagree
    if same_minute and model_cover is not None and len(reasons) < 2:
        return "ok"
    return "needs review: " + ", ".join(reasons)


def model_cloud_cover(lat: float, lon: float, when: datetime) -> float | None:
    """The weather model's own cloud cover for the hour the photo was taken (None if unavailable)."""
    try:
        d = data._get("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon, "hourly": "cloud_cover", "timezone": "auto",
            "start_date": when.date().isoformat(), "end_date": when.date().isoformat()}, attempts=2)
        return float(d["hourly"]["cloud_cover"][when.hour])
    except Exception:
        return None


def rate_photo(path: Path, lat: float, lon: float) -> dict:
    schema = {"type": "object", "properties": {
        "cloud_cover_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "stars_visible": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["cloud_cover_percent", "stars_visible", "confidence"]}
    out = json.loads(gemma(
        "This is a photo of the night or evening sky. Estimate what percent of the visible sky is covered by cloud "
        "(0 = completely clear, 100 = overcast), and whether any stars are visible. Ignore trees, buildings and lights.",
        image=path.read_bytes(), schema=schema))
    when = taken_at(path)
    t = datetime.fromisoformat(when)
    nearby = []
    if SKY_LOG.exists():
        with SKY_LOG.open() as f:
            for r in csv.DictReader(f):
                if abs((datetime.fromisoformat(r["time"]) - t).total_seconds()) <= 15 * 60 and r["photo"] != path.name:
                    nearby.append(int(r["cloud_cover_percent"]))
    model = model_cloud_cover(lat, lon, t)
    out["check"] = check_rating(out["cloud_cover_percent"], nearby, model)
    row = {"time": when, "lat": lat, "lon": lon, "photo": path.name, **out}
    SKY_LOG.parent.mkdir(exist_ok=True)
    new = not SKY_LOG.exists()
    with SKY_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return out


def main():
    ap = argparse.ArgumentParser(description="Will the sky really be clear tonight?")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", default="here")
    ap.add_argument("--photo", type=Path, help="a sky photo to log as ground truth (rated by Gemma 4)")
    ap.add_argument("--no-gemma", action="store_true")
    args = ap.parse_args()

    if args.photo:
        r = rate_photo(args.photo, args.lat, args.lon)
        print(f"Gemma 4 sees ~{r['cloud_cover_percent']}% cloud, stars visible: {r['stars_visible']} "
              f"({r['confidence']} confidence). Check: {r['check']}. Logged to {SKY_LOG}.")
        return

    key = f"pt{args.lat:+.2f}{args.lon:+.2f}"
    print(f"Learning how wrong the forecast usually is at {args.lat}, {args.lon} (first run downloads ~32 months)…")
    hist = data.build(key, lat=args.lat, lon=args.lon)
    res = predict_place(args.lat, args.lon, hist)
    print(f"\nNight of {res['night']}, trained on {res['trained_on_nights']} nights")
    for h in res["hours"]:
        bar = "█" * round(h["p_clear"] * 20)
        print(f"  {h['time']}  {bar:<20} {round(h['p_clear'] * 100):3}% clear   (app: {h['app_cloud_cover']:3}% cloud)")
    w = res["window"]
    print(f"\n  {'GO ' + w['start'] + '–' + w['end'] if w else 'Probably not tonight.'}   Moon {res['moon']['illumination']}% lit")
    if not args.no_gemma:
        try:
            print("\n" + plan(args.name, res))
        except httpx.HTTPError as e:
            print(f"\n(Gemma unavailable: {e})")


if __name__ == "__main__":
    main()
