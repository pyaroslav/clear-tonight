# Clear Tonight?

[![tests](https://github.com/pyaroslav/clear-tonight/actions/workflows/tests.yml/badge.svg)](https://github.com/pyaroslav/clear-tonight/actions/workflows/tests.yml)

**Will the sky really be clear tonight?** Your weather app's cloud forecast, corrected by an open model that learned from the forecast's own track record at your spot.

Cloud cover decides whether a night walk, a meteor shower or a look at the stars is worth getting off the couch for, and it's the forecast variable weather apps get wrong most often. Clear Tonight takes the forecast that was issued about 24 hours ahead, compares 32 months of those forecasts against what actually happened at the same place, and lets **TabPFN v2** learn the pattern of its mistakes. Then it answers one question for tonight: *go or don't go, and when?*

- **Live page (6 cities, refreshed daily):** see `site/`. Deployed as a static site.
- **Your own spot, fully local:** `python -m clear_tonight --lat .. --lon ..` trains on your coordinates in about 20 s on a GPU (minutes on a CPU), and a local **Gemma 4** (via Ollama) turns the result into a two-line plan. Point it at a sky photo and Gemma 4 rates the cloud cover, so you build your own ground truth.

## Results (walk-forward backtest, last 12 months, training only on earlier data)

| City | Brier, weather app | Brier, Clear Tonight | Error cut | Wasted trips (app → us) | Missed clear nights (app → us) |
|---|---|---|---|---|---|
| Austin | 0.214 | 0.137 | 36% | 27 → 20 | 36 → 50 |
| Chicago | 0.191 | 0.130 | 32% | 33 → 19 | 37 → 53 |
| Denver | 0.279 | 0.164 | 41% | 57 → 35 | 37 → 48 |
| London | 0.194 | 0.140 | 28% | 16 → 10 | 47 → 65 |
| Minneapolis | 0.233 | 0.148 | 36% | 29 → 21 | 51 → 60 |
| Seattle | 0.185 | 0.117 | 37% | 26 → 12 | 31 → 50 |

Brier score: lower is better. A *wasted trip* is a night the forecast said "go" (two consecutive clear hours) and the sky was cloudy. Clear Tonight is more cautious than the app: far fewer wasted trips, more missed clear nights. A plain logistic regression on the same 15 features scores about the same as TabPFN v2 when there is a lot of history.

### How much history does it need?

Training on only the last K nights before each of 6 test months, averaged over 6 cities (raw weather app: **0.237**):

| K nights | TabPFN v2 | Logistic regression | Gradient boosting | Recalibrated app number (1 feature) |
|---|---|---|---|---|
| 7 | 0.214 | 0.357 | 0.259 | 0.199 |
| 14 | 0.204 | 0.316 | 0.277 | 0.183 |
| 30 | 0.186 | 0.260 | 0.248 | 0.179 |
| 60 | 0.176 | 0.199 | 0.253 | 0.179 |
| 120 | 0.171 | 0.161 | 0.237 | 0.176 |
| 365 | 0.153 | 0.149 | 0.168 | 0.174 |

TabPFN v2 is the only 15-feature model that beats the raw forecast from the first week. The others need two (logistic regression) to four (gradient boosting) months before they even match it. With two months TabPFN is the best of all, and with a year a plain logistic regression catches up. That's why it's the right tool for "point it at your backyard".

## Run it

```bash
git clone https://github.com/pyaroslav/clear-tonight && cd clear-tonight
pip install -e .                       # TabPFN v2, pandas, scikit-learn, ephem
ollama pull gemma4:e4b                 # optional, for the plain-language plan and sky photos
python -m clear_tonight --lat 44.98 --lon -93.27 --name Minneapolis
python -m clear_tonight --lat 44.98 --lon -93.27 --photo sky.jpg     # log what the sky looked like
python -m clear_tonight.backtest       # reproduce the tables (GPU recommended)
python -m clear_tonight.learning_curve
```

## Honest limits

- **Truth is ERA5 reanalysis**, a model-assisted reconstruction of cloud cover, not a person looking up. That's why the app logs your own sky photos.
- **One forecast model (GFS)**, 6 cities and 32 months. "Clear" means cloud cover at or below 25%.
- The day-ahead archive only keeps total cloud cover (no low/mid/high layers), which limits what any correction can learn.
- The live page's nightly CPU run uses a single TabPFN ensemble member (the backtest used the default 4 on a GPU).

## Credits and licenses

- Built with PriorLabs-TabPFN (**TabPFN v2**, Prior Labs License 1.1, Apache-2.0 plus attribution). Newer TabPFN versions are non-commercial; this project deliberately uses v2.
- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0): previous-runs API (GFS forecasts issued 1 and 2 days ahead) and the ERA5 archive.
- Moon rise, set and phase: PyEphem. Language and sky photos: Gemma 4 (open weights) via Ollama.
- Code: MIT.
