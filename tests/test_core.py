"""Fast, offline tests for the parts a wrong answer would hide in."""

import numpy as np
import pandas as pd

from clear_tonight import data
from clear_tonight.__main__ import facts
from clear_tonight.backtest import go_window, score
from clear_tonight.predict import best_window


def _hours(ps, app=50):
    return [{"time": f"{h:02d}:00", "p_clear": p, "app_cloud_cover": app} for h, p in zip([20, 21, 22, 23, 0, 1, 2, 3, 4], ps)]


def test_best_window_picks_longest_clear_run():
    w = best_window(_hours([.2, .6, .7, .1, .8, .9, .9, .3, .2]))
    assert (w["start"], w["end"], w["hours"]) == ("00:00", "03:00", 3)


def test_single_clear_hour_is_not_a_window():
    assert best_window(_hours([.2, .9, .2, .1, .1, .1, .1, .1, .1])) is None


def test_after_midnight_hours_belong_to_previous_evening():
    t = pd.date_range("2026-10-09 18:00", periods=12, freq="h")
    df = pd.DataFrame({"time": t, **{v: 10.0 for v in data.FC_VARS}, "cloud_cover_d2": 20.0, "truth": 5.0})
    f = data.features(df)
    assert set(f["night"]) == {pd.Timestamp("2026-10-09").date()}
    assert list(f["hour_of_night"]) == list(range(9))
    assert f["clear"].eq(1).all() and f["run_change"].eq(-10).all()


def test_go_window_needs_two_consecutive_hours():
    n = pd.DataFrame({"hour_of_night": range(4), "p": [.9, .1, .9, .1]})
    assert not go_window(n, "p")
    n["p"] = [.1, .9, .9, .1]
    assert go_window(n, "p")


def test_score_counts_wasted_trips():
    test = pd.DataFrame({"night": [1, 1, 2, 2], "hour_of_night": [0, 1, 0, 1], "clear": [0.0, 0.0, 1.0, 1.0]})
    s = score(test, np.array([.9, .9, .9, .9]))
    assert s["false_go"] == 1 and s["missed_go"] == 0 and s["go_precision"] == 0.5


def test_facts_flag_an_overoptimistic_weather_app():
    res = {"hours": _hours([.5] * 9, app=3), "window": None, "moon": {"illumination": 0}}
    text = " ".join(facts(res))
    assert "MORE optimistic" in text and "weather app is" in text and "dark sky" in text
