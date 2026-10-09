"""Condense results/*.json into site/results.json (what the page and the post quote)."""

import json
from pathlib import Path

import pandas as pd

from .backtest import RESULTS, summarise

SITE = Path(__file__).resolve().parents[1] / "site"


def main():
    bt = json.loads((RESULTS / "backtest.json").read_text())
    lc = pd.DataFrame(json.loads((RESULTS / "learning_curve.json").read_text()))
    overall = summarise(bt)
    per_city = {c: summarise([r for r in bt if r["city"] == c]).loc[["tabpfn_v2", "raw_forecast"]]
                for c in sorted({r["city"] for r in bt})}
    curve = lc[lc.model != "raw_forecast"].pivot_table(index="k", columns="model", values="brier", aggfunc="mean")
    out = {
        "months": len({r["month"] for r in bt}),
        "overall": overall.round(3).reset_index().to_dict(orient="records"),
        "per_city": {c: {m: {"brier": round(float(t.loc[m, "brier"]), 3),
                             "go_precision": round(float(t.loc[m, "go_precision"]), 3),
                             "go_recall": round(float(t.loc[m, "go_recall"]), 3),
                             "false_go": int(t.loc[m, "false_go"]), "missed_go": int(t.loc[m, "missed_go"])}
                         for m in t.index} for c, t in per_city.items()},
        "learning_curve": {"raw_forecast": round(float(lc[lc.model == "raw_forecast"]["brier"].mean()), 3),
                           "k": [int(k) for k in curve.index],
                           **{m: [round(float(v), 3) for v in curve[m]] for m in curve.columns}},
    }
    (SITE / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["learning_curve"]))


if __name__ == "__main__":
    main()
