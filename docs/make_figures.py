"""Figures for the README and the DEV post: the learning curve and the cover.

    python docs/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BG, FG, MUTED = "#0b1430", "#e8ecf8", "#8f9bc2"
NAMES = {"tabpfn_v2": ("TabPFN v2", "#5fe3a1", 3), "logistic": ("logistic regression", "#ff8a6b", 1.8),
         "gboost": ("gradient boosting", "#b39cff", 1.8), "raw_calibrated": ("recalibrated app number", "#f6e7b4", 1.8)}


def learning_curve():
    df = pd.DataFrame(json.loads((ROOT / "results" / "learning_curve.json").read_text()))
    raw = df[df.model == "raw_forecast"].brier.mean()
    curve = df[df.model != "raw_forecast"].groupby(["model", "k"]).brier.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhline(raw, color=MUTED, ls="--", lw=1.5)
    ax.text(curve.index[-1], raw + 0.004, "weather app as-is", color=MUTED, ha="right", va="bottom", fontsize=11)
    for m, (label, color, lw) in NAMES.items():
        ax.plot(curve.index, curve[m], color=color, lw=lw, marker="o", ms=4, label=label)
    ax.set_xscale("log")
    ax.set_xticks(curve.index, [str(k) for k in curve.index])
    ax.minorticks_off()
    ax.set_xlabel("nights of history the model is trained on", color=MUTED, fontsize=11)
    ax.set_ylabel("Brier score (lower is better)", color=MUTED, fontsize=11)
    ax.set_title("With little history, TabPFN v2 is the only 12-feature model that beats the app",
                 color=FG, fontsize=14, loc="left", pad=12)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#26325a")
    ax.grid(color="#1c2750", lw=0.8)
    ax.legend(frameon=False, labelcolor=FG, fontsize=10.5, loc="upper right")
    fig.text(0.01, 0.01, "6 cities × 6 test months, each trained only on the K nights before it. "
             "Data: Open-Meteo (GFS day-ahead vs ERA5).", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "learning-curve.png", facecolor=BG)


def cover():
    W, H = 1000, 420
    sky = Image.open(ROOT / "field" / "2026-10-10" / "sky-1-1951.jpg").convert("RGB")
    sky = sky.resize((W, int(sky.height * W / sky.width))).crop((0, 60, W, 60 + H))
    sky = ImageEnhance.Brightness(sky).enhance(1.6)
    img = Image.new("RGB", (W, H), BG)
    img = Image.blend(img, sky, 0.75)
    d = ImageDraw.Draw(img)
    font = lambda size, bold=True: ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", size)
    d.text((56, 62), "Clear", font=font(76), fill=FG)
    d.text((56 + d.textlength("Clear ", font=font(76)), 62), "Tonight?", font=font(76), fill="#f6e7b4")
    d.text((58, 168), "Your weather app's cloud forecast, checked by an open model", font=font(26, False), fill=FG)
    # an hour strip like the live page: app cloud vs real clear chance
    hours = ["20", "21", "22", "23", "00", "01", "02"]
    p = [0.84, 0.79, 0.73, 0.69, 0.77, 0.78, 0.79]
    for i, (h, v) in enumerate(zip(hours, p)):
        x = 58 + i * 82
        d.text((x + 22, 236), h, font=font(18, False), fill=MUTED)
        d.rounded_rectangle((x, 262, x + 72, 302), 8, fill="#5fe3a1" if v >= 0.75 else "#f6e7b4")
        d.text((x + 36, 282), f"{round(v * 100)}%", font=font(20), fill=BG, anchor="mm")
    d.text((58, 340), "TabPFN v2  ·  Gemma 4  ·  6 cities  ·  32 months of forecasts  ·  Render",
           font=font(21, False), fill=MUTED)
    img.save(DOCS / "cover.png")


if __name__ == "__main__":
    learning_curve()
    cover()
