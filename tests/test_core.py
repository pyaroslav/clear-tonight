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


def test_photo_rating_that_disagrees_with_everything_is_held_back():
    from clear_tonight.__main__ import check_rating
    # field/2026-10-10: three photos ~5-10%, model 0%, night mode turned one grey -> Gemma said 95%
    assert check_rating(95, [5, 10, 10], 0.0).startswith("needs review")
    assert check_rating(10, [5, 95], 0.0) == "ok"
    # one source disagreeing is not enough when both are available
    assert check_rating(90, [85, 95], 0.0) == "ok"
    # alone with the model, a big disagreement is enough
    assert check_rating(95, [], 0.0).startswith("needs review")
    assert check_rating(20, [], None) == "ok"
