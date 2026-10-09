"""Walk-forward backtest: for each of the last N months, train only on what came before it.

    python -m clear_tonight.backtest --cities seattle london --months 12

Hourly metrics (Brier score, accuracy) and the decision that matters to a person:
"is there a 2-hour clear window tonight worth going out for?" (precision / recall).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data

RESULTS = Path(__file__).resolve().parents[1] / "results"


def tabpfn_v2():
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if torch.cuda.is_available():
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    # CPU (e.g. the nightly GitHub runner): one ensemble member instead of the default 4, ~4x faster.
    # The backtest numbers in results/ were produced on a GPU with the default ensemble.
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=1)


MODELS = {
    "tabpfn_v2": tabpfn_v2,
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05),
}


def go_window(night: pd.DataFrame, col: str, threshold: float = 0.5) -> bool:
    """True if two consecutive night hours are both (predicted) clear."""
    v = night.sort_values("hour_of_night")[col].values >= threshold
    return bool(np.any(v[:-1] & v[1:]))


def score(test: pd.DataFrame, p: np.ndarray) -> dict:
    y = test["clear"].values
    t = test.assign(p=p)
    nights = list(t.groupby("night"))
    truth_go = np.array([go_window(n, "clear") for _, n in nights])
    pred_go = np.array([go_window(n, "p") for _, n in nights])
    tp = int((truth_go & pred_go).sum())
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "nights": len(nights), "go_nights": int(truth_go.sum()),
        "go_precision": tp / max(1, int(pred_go.sum())),
        "go_recall": tp / max(1, int(truth_go.sum())),
        "false_go": int((pred_go & ~truth_go).sum()),
        "missed_go": int((truth_go & ~pred_go).sum()),
    }


def run(city: str, months: int) -> list[dict]:
    df = data.build(city).sort_values("time")
    df["month"] = pd.to_datetime(df["night"]).dt.to_period("M")
    folds = sorted(df["month"].unique())[-months:]
    rows = []
    for m in folds:
        train, test = df[df["month"] < m], df[df["month"] == m]
        X, y, Xt = train[data.FEATURES], train["clear"].astype(int), test[data.FEATURES]
        preds = {
            # what the weather app tells you: forecast cover <= 25% means "clear"
            "raw_forecast": (test["cloud_cover"] <= data.CLEAR).astype(float).values,
            # the forecast's cover turned into a probability, then thresholded
            "raw_calibrated": LogisticRegression().fit(train[["cloud_cover"]], y).predict_proba(test[["cloud_cover"]])[:, 1],
            # how often this hour of this calendar month was clear before
            "climatology": np.full(len(test), train[train["time"].dt.month == m.month]["clear"].mean()),
        }
        timing = {}
        for name, make in MODELS.items():
            t0 = time.perf_counter()
            model = make().fit(X, y)
            preds[name] = model.predict_proba(Xt)[:, 1]
            timing[name] = time.perf_counter() - t0
        for name, p in preds.items():
            rows.append({"city": city, "month": str(m), "model": name, "train_rows": len(train),
                         "seconds": timing.get(name, 0.0), **score(test, p)})
        best = min((r for r in rows if r["month"] == str(m)), key=lambda r: r["brier"])
        print(f"{city:12} {m}  train={len(train):5}  best={best['model']:14} brier={best['brier']:.3f}", flush=True)
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(
        brier=("brier", "mean"), accuracy=("accuracy", "mean"),
        go_nights=("go_nights", "sum"), false_go=("false_go", "sum"), missed_go=("missed_go", "sum"),
        seconds=("seconds", "mean"))
    tp = agg["go_nights"] - agg["missed_go"]
    agg["go_precision"] = tp / (tp + agg["false_go"])
    agg["go_recall"] = tp / agg["go_nights"]
    return agg.sort_values("brier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(data.CITIES))
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()
    rows = []
    for c in args.cities:
        rows += run(c, args.months)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "backtest.json").write_text(json.dumps(rows, indent=1))
    print(summarise(rows).round(3).to_string())
    for c in args.cities:
        print(f"\n{c}\n" + summarise([r for r in rows if r["city"] == c]).round(3).to_string())


if __name__ == "__main__":
    main()
