# Cost per city

One line per city task, added when its retrospective is in (docs/adding-a-city.md 12). Weighted
tokens from `checks/token_usage.py` (cache reads 0.1, writes 1.25, output 5, input 1); wall time
from first to last call. Calibration: 45% to 46% of Anita's weekly limit over about 9.5M weighted
tokens (2026-10-14, coarse: the meter shows whole percents). Second point: 51% on the morning of
2026-10-15, after the overnight run (T-363, three cities, park-and-ride T-371/T-389/T-373/T-374,
T-338, T-391, T-327, T-326 stage 1, T-380, T-394, and T-392 still running): 43.7M weighted in
all, orchestrator 2.3M of it. If the run started near 46%, about 9M a percent, as before.

| City | Task | Model | Wall | Weighted tokens | Biggest cost | Notes |
|---|---|---|---|---|---|---|
| London | T-041 water | Opus | 24 min | 1.55M | the England water read (13 min) | one-off tooling |
| London | T-288 stage a | Opus | 92 min | 5.25M | an hour of OSM reads; 29% re-caching after idle waits | first tier 0 city abroad |
| London | T-360 stage b | Opus | 74 min | 3.53M | six wakes after idle waits, 36% | England's official inputs; found the zones_of bug |
| Paris | T-289 | Opus | 32 min | about 2.8M | one wake after a 10-minute foreground browser check (261k) | tier 0 plus INSEE jobs and curve; first city after T-363 |
| Shanghai | T-291 | Opus | 31 min | about 3.5M | built twice (district jobs found after the first build); one 276k wake | tier 0 plus census population and economic census jobs by district |
| Greater Tokyo | T-420 | Opus | 125 min | about 5.9M | first city in Japan (the module); 58-minute build (parks 35, medical 15, T-425); 1.1M waiting in four-minute chunks | census meshes, economic census jobs, municipality flows |
| Greater Tokyo | T-426 rebuild | Opus | about 85 min | orchestrator to measure | the curve script's 25-minute grid; routes 14 | Ibaraki added, dictionary romaji; build 10 minutes after T-425 |
| Berlin | T-290 | Opus | 25 min | about 2.95M | one build; jobs and flows found first, per the playbook | tier 0 plus Zensus 2022, the Pendlerrechnung's jobs and flows |
| Mumbai | T-292 | Opus | 32 min | about 3.2M | built twice (OSM talukas off the 2011 lines) | tier 0, 2011 census grown to 2020, curve fitted on B-28 bands |
