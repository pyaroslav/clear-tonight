# Field test, Minneapolis, night of 2026-10-10

Forecast made 2026-10-10 08:23 CDT (13:23 UTC), before going outside. Committed before the walk so it can't be edited after.

```
Learning how wrong the forecast usually is at 44.98, -93.27 (first run downloads ~32 months)…

Night of 2026-10-10, trained on 977 nights
  20:00  ███████████████████   96% clear   (app:   0% cloud)
  21:00  ███████████████████   96% clear   (app:   0% cloud)
  22:00  ███████████████████   97% clear   (app:   0% cloud)
  23:00  ███████████████████   97% clear   (app:   0% cloud)
  00:00  ████████████████████  98% clear   (app:   0% cloud)
  01:00  ███████████████████   97% clear   (app:   0% cloud)
  02:00  ███████████████████   97% clear   (app:   0% cloud)
  03:00  ███████████████████   97% clear   (app:   0% cloud)
  04:00  ███████████████████   97% clear   (app:   0% cloud)

  GO 20:00–05:00   Moon 0% lit
```

Gemma 4's plan (`gemma4:e4b`, local):

> Go outside tonight for stargazing; the forecast shows about a 97% chance of really clear skies between 20:00 and 05:00. Since the Moon is 0% lit, it's a great night for faint stars. Remember to wear warm layers when you head out.

Astronomical darkness begins about 20:14 CDT. Moon 0% lit (new Moon). Forecast air temperature 75–71 °F, wind ~12 mph.

## What actually happened

I went out into the backyard at about 7:50 PM, a little before full darkness (about 8:14 PM). The sky was clear, with no cloud anywhere, and the first stars were already out. It was warm enough to just sit and relax instead of walking around, the crickets were going, and the neighbourhood was lovely.

| Photo (taken) | What it shows | Gemma 4 (`gemma4:e4b`) says |
|---|---|---|
| [sky 1](2026-10-10/sky-1-1951.jpg) (19:51) | clear sky, stars | ~5% cloud, stars visible (high confidence) |
| [sky 2](2026-10-10/sky-2-1951.jpg) (19:51) | clear sky, stars, grey twilight glow | ~10% cloud, stars visible (medium) |
| [sky 3](2026-10-10/sky-3-1951.jpg) (19:51) | clear sky, stars, brighter grey glow, a tree at the edge | **~95% cloud, no stars (high)**, wrong |
| [backyard](2026-10-10/backyard-1952.jpg) (19:52) | the yard, a clear blue sky above the trees | ~10% cloud, no stars (medium) |

**Forecast vs. sky:** hit. The forecast said 96–98% chance of clear sky; it was clear. Open-Meteo's own 15-minute analysis also shows 0% cloud from 17:45 to 20:00. One clear night under a confident forecast proves little on its own; the backtest is the real evidence. This is the part a backtest can't show: it was right, and it was worth going out.

**Gemma got one wrong, with high confidence.** Sky 3 is the same clear sky as sky 1 and 2, but the phone's night mode lifted the twilight and city glow to a flat grey, and Gemma read flat grey as overcast. It said the same thing on three re-runs. So the photo log can't be trusted blindly as ground truth: a photo rating that disagrees with the forecast *and* with the other photos from the same minute needs a second look before it's used to train anything.
