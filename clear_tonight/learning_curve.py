"""How much history does a correction need? Train on only the last K nights before each test month.

    python -m clear_tonight.learning_curve --months 6

The practical question: someone points Clear Tonight at a new spot (their backyard, a
dark-sky park). How many nights of history before the correction is worth having?
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import data
from .backtest import MODELS, RESULTS, score

KS = [7, 14, 30, 60, 120, 365]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(data.CITIES))
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()
    rows = []
    for city in args.cities:
        df = data.build(city).sort_values("time")
        df["month"] = pd.to_datetime(df["night"]).dt.to_period("M")
        for m in sorted(df["month"].unique())[-args.months:]:
            before = df[df["month"] < m]
            test = df[df["month"] == m]
            nights = sorted(before["night"].unique())
            rows.append({"city": city, "month": str(m), "k": 0, "model": "raw_forecast",
                         **score(test, (test["cloud_cover"] <= data.CLEAR).astype(float).values)})
            for k in KS:
                train = before[before["night"].isin(nights[-k:])]
                y = train["clear"].astype(int)
                if y.nunique() < 2:
                    continue
                for name, make in {**MODELS, "raw_calibrated": lambda: LogisticRegression()}.items():
                    cols = ["cloud_cover"] if name == "raw_calibrated" else data.FEATURES
                    p = make().fit(train[cols], y).predict_proba(test[cols])[:, 1]
                    rows.append({"city": city, "month": str(m), "k": k, "model": name, **score(test, p)})
            print(city, m, flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "learning_curve.json").write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    raw = df[df.model == "raw_forecast"]["brier"].mean()
    table = df[df.model != "raw_forecast"].pivot_table(index="k", columns="model", values="brier", aggfunc="mean")
    print(f"\nmean Brier, raw forecast = {raw:.3f}\n" + table.round(3).to_string())
    wins = df[df.model != "raw_forecast"].loc[lambda d: d.groupby(["city", "month", "k"])["brier"].idxmin()]
    print("\nwho wins each (city, month, K):\n" + wins.groupby("k")["model"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
