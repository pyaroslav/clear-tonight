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
    row = {"time": datetime.now().isoformat(timespec="minutes"), "lat": lat, "lon": lon,
           "photo": path.name, **out}
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
              f"({r['confidence']} confidence). Logged to {SKY_LOG}.")
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
