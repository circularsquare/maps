# Cost per city

One line per city task, added when its retrospective is in (docs/adding-a-city.md 12). Weighted
tokens from `checks/token_usage.py` (cache reads 0.1, writes 1.25, output 5, input 1); wall time
from first to last call. Calibration: 45% to 46% of Anita's weekly limit over about 9.5M weighted
tokens (2026-10-14, coarse: the meter shows whole percents).

| City | Task | Model | Wall | Weighted tokens | Biggest cost | Notes |
|---|---|---|---|---|---|---|
| London | T-041 water | Opus | 24 min | 1.55M | the England water read (13 min) | one-off tooling |
| London | T-288 stage a | Opus | 92 min | 5.25M | an hour of OSM reads; 29% re-caching after idle waits | first tier 0 city abroad |
| London | T-360 stage b | Opus | 74 min | 3.53M | six wakes after idle waits, 36% | England's official inputs; found the zones_of bug |
