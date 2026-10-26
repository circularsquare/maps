# Routed drives between places, cheaply (T-279)

Research 2026-10-26, no build. Question (Anita, 2026-10-26): routing every near pair of market
places on a whole-world road graph and rebuilding the market packs is too expensive; can it be made
substantially cheaper with little loss of accuracy?

**Answer: yes.** A least-cost path over a coarse global raster (1/60 degree, about 1.85 km cells)
whose speeds come from three global layers we have or can fetch in seconds (GRIP4 road density,
GEDTM30 slope, the T-1285 world water mask with Wikidata's fixed links) predicts OSRM's drive times
better than every formula tried, about as well as routing our own OSM graph, and costs about an
hour on 4 cores of this machine. Nothing in any pack changes.

## The truth sample

OSRM's public demo server (car profile, free-flow times), its table service: one request per
region, up to 24 market places spread over 450 km around a big place, two rounds of anchors
(the first 700 km apart, the second between them), one request every 6 s with a plain User-Agent
naming nothing personal, 99 requests in all. After dropping pairs OSRM snapped more than 3 km off,
pairs whose road is shorter than the straight line (broken snaps), pairs between Great Britain and
the continent and Hokkaido and Honshu (the market adds those crossings itself) and Korea to China
(one landmass in the market, through North Korea): **13,470 pairs, 20-700 km, 991 places, 43
countries, 89 regions.** Road over straight line: median 1.31 (10th-90th percentile 1.17-1.60);
OSRM's mean speed median 82 km/h. On this wider sample today's rule is 3% faster than OSRM at the
median (T-279's 43 big-city pairs had it 10% slower), so the rule's level is right; its spread is
the problem.

Scores are **held out by region**: every model is fitted 89 times, each time without one region,
and scored on that region. "Odds" is the factor by which the model's car moves rail's odds against
the car compared with the routed car, in the market's own terms (perceived car minutes = drive +
road km x $0.15 x 3 min/$ / 2, scale 0.10 x sqrt(60 / car)), as in notes/T-279.md. A median odds
error of x2.5 means half the pairs have rail's odds within a factor 2.5 of what routed drives would
give. The market is that sensitive: 30 perceived minutes at four hours by car is a factor 3-4.

Scripts in `notes/T-279/` (`osrm_tables.py`, `features.py`, `fit.py`, `coarse_test.py`,
`report.py`, `stats.py`; they take a scratch folder holding `osrm/tables.json` and the GRIP
download), data in `artifacts/trainworld/t279/` (`tables.json` OSRM's answers, `pairs.json` every
pair with its features, `fit_results.json`, `coarse_GB.json`).

## The options, measured

Held out by region, 13,470 pairs; and on T-279's 44 named pairs (models fitted on the whole table
sample):

| Model | Minutes off, median / 90th pct | % off, median / 90th | Odds off, median / 90th | T-279 pairs: odds off, median / 90th |
|---|---|---|---|---|
| Today's rule: 15 + 1.25 x straight at 85 km/h | 28 / 110 | 10.0 / 28.8 | x3.9 / x83 | x8.9 / x55 |
| Rule refitted (one fixed term, one speed) | 30 / 104 | 10.8 / 28.2 | x4.4 / x59 | x11.9 / x86 |
| Speed by country | 26 / 86 | 9.0 / 25.0 | x3.6 / x34 | x7.4 / x57 |
| Formula on the straight line: water detour | 30 / 99 | 10.5 / 27.6 | x4.4 / x52 | x18 / x66 |
| ... plus mean slope | 28 / 90 | 9.7 / 25.6 | x3.8 / x39 | x13 / x62 |
| ... plus motorway share (best formula) | 25 / 84 | 8.6 / 23.4 | x3.3 / x29 | x6.5 / x50 |
| **Raster router (v8, below)** | **19 / 80** | **6.8 / 22.2** | **x2.5 / x21** | **x2.8 / x15** |

Other router variants: without the slope term the 90th percentile was a little better and the
named mountain pairs worse; a steeper slope penalty was worse; water at 15 km/h against 6 made
no difference. By distance (rule against router, % off median / 90th): 20-100 km 12.6/38.2 against
10.4/33.3; 100-250 km 11.8/31.4 against 8.5/24.1; 250-450 km 9.7/28.2 against 6.4/21.3; 450-700 km
8.5/23.5 against 5.2/17.8. Pairs more than 20% off: 21% with the rule, 12% with the router.

**Our own OSM graph, full against coarse** (England's cached extract, `europe_united-kingdom_england.ways2.npz`;
speeds by road class, OSRM's defaults rounded, no oneways or turn costs; 55 pairs of 11 English
places; after one level factor each):

| Graph | Points | Build | % off, median / 90th |
|---|---|---|---|
| Every drive class | 12.7M | 280 s | 8.9 / 17.7 |
| Motorway, trunk, primary and links | 1.5M | 28 s | 7.6 / 17.5 |
| ... plus secondary | 2.2M | 37 s | 7.8 / 16.3 |

So a coarse graph loses nothing against a full one, but neither beats OSRM's own profile (speed
limits, turns) by being a graph: the raster router on 335 British pairs, levelled the same way,
was 5.1 / 13.9.

What the router changes on T-279's kinds of pair (median of each kind; "rule / OSRM" is the
rule's minutes over OSRM's):

| Kind | Pairs | Rule / OSRM | Router / OSRM | Odds off, rule | Odds off, router |
|---|---|---|---|---|---|
| Flat, motorway between | 14 | 1.25 | 1.08 | x13 | x2.8 |
| Mountains | 13 | 1.00 | 1.00 | x3.3 | x2.6 |
| Water to go round | 9 | 0.87 | 0.97 | x8.1 | x2.9 |
| Sparse roads | 3 | 1.18 | 1.02 | x7.7 | x4.4 |
| Long, over 500 km | 5 | 1.16 | 0.99 | x40 | x1.6 |

## What it would change in rail shares on the measured corridors

Rail's odds against the car if the market used routed drives, against today's rule; first OSRM's
drive, then the router's (minutes: OSRM / rule / router):

| Corridor | Minutes | Odds with OSRM | Odds with router |
|---|---|---|---|
| Berlin - Hamburg | 189 / 242 / 225 | x0.08 | x0.49 |
| Warsaw - Poznan | 190 / 263 / 265 | x0.03 | x0.96 |
| Dallas - Houston | 273 / 343 / 292 | x0.05 | x0.12 |
| Paris - Lille | 149 / 191 / 165 | x0.09 | x0.22 |
| Shanghai - Nanjing | 202 / 252 / 226 | x0.08 | x0.29 |
| Munich - Berlin | 349 / 455 / 400 | x0.03 | x0.15 |
| Paris - Marseille | 480 / 583 / 517 | x0.04 | x0.15 |
| Tokyo - Nagoya | 246 / 248 / 241 | x0.99 | x0.91 |
| Zurich - Milan | 217 / 197 / 174 | x3.5 | x0.28 |
| Athens - Thessaloniki | 344 / 282 / 326 | x38 | x15 |
| Taipei - Hualien | 222 / 113 / 259 | x7,000 | x34,000 |
| Sofia - Thessaloniki | 229 / 218 / 262 | x1.9 | x7.7 |
| Copenhagen - Aarhus | 213 / 177 / 246 | x15 | x98 |
| Copenhagen - Gothenburg | 257 / 223 / 250 | x9.6 | x7.2 |
| Hiroshima - Matsuyama | 159 / 78 / 114 | x2,200 | x40 |
| Osaka - Fukuoka | 489 / 426 / 378 | x8.6 | x0.19 |
| New York - Boston | 280 / 280 / 248 | x0.70 | x0.25 |
| Amsterdam - Brussels | 168 / 170 / 149 | x0.77 | x0.26 |

The router moves nearly every flat corridor the right way (rail loses trips to a faster car) but
by about half as much as OSRM, and gets the mountain and inlet pairs mostly right (Taipei - Hualien,
Hiroshima - Matsuyama, Athens - Thessaloniki, the Danish pairs). Its misses: dense corridors where
traffic, not road, sets the time (New York - Boston, Amsterdam - Brussels, Hong Kong - Guangzhou
come out 10-13% fast), Osaka - Fukuoka (23% fast, probably cutting across part of the Inland Sea
at the water speed), Zurich - Milan (20% fast, probably because the Gotthard road counts as a
highway in GRIP, which takes no slope factor), and Warsaw - Poznan (as slow as the rule). The rule's own odds error on these
pairs is three times the router's. Remember that OSRM is free flow: real driving is 10-20% slower
in traffic, so on flat pairs the router's smaller move is nearer life than OSRM's numbers suggest.

Rail against the plane does not move (the plane's access drives keep the rule).

## Costs

| Option | Agent time | Compute | Fits this machine? | What a rebuild touches | Accuracy (held out) |
|---|---|---|---|---|---|
| Full OSRM-style routing on a world graph | 2-3 sessions | planet file 80 GB; OSRM's car processing of the planet needs well over 100 GB RAM (my estimate, not run); 6-12 h | No: 64 GB RAM, 25 GB free disk. AWS, a large memory instance | the place-pair table below (not the packs, if stored apart) | the truth itself, but free flow: 10-20% fast against traffic |
| Coarse OSM graph (motorway, trunk, primary) | 2 sessions | Geofabrik extracts for every market landmass, about 70 GB streamed through the 25 GB free, one at a time, GDAL reads of hours; graph about 10% of the full one; routing under an hour | Yes, slowly (a day of downloads and reads); AWS would be quicker | same table | measured on England like a full graph: 7.6% / 17.5% with class speeds; better only with speed limits and turns, i.e. rebuilding OSRM's profile |
| Stitching the packs' road graphs | more than the coarse graph | as the coarse graph | as above | as above | packs keep only zone-to-zone distance factors (within 13 km and between supercells), and the OSM extracts behind them are deleted after building (8 of about 270 cached), so this means downloading everything again; ruled out |
| Formula on the straight line (water detour, slope, motorway share) | 1 session | features only, minutes | Yes | same table, or computed from three small grids | 8.6% / 23.4%; odds x3.3 / x29 |
| Speed by country | under a session | none | Yes | the rule's constant per country | 9.0% / 25.0%; odds x3.6 / x34; and per-country tuning, against "general methods" |
| **Raster router** | **1-2 sessions** | **slope grid from the local GEDTM30 file (about an hour on 4 cores) plus one Dijkstra per place over a window of 1,500 km (2-3 s each, 1,748 places, about 25 min on 4 cores); a few GB of RAM** | **Yes** | **one new place-pair file read by the market; no pack, no join** | **6.8% / 22.2%; odds x2.5 / x21** |
| Only pairs that matter | | 108k place pairs on one landmass lie within 600 km, 234k within 1,000, 370k within 1,500 | | | cuts nothing worth cutting: the router does all of them in under an hour |

## Recommendation: the raster router, as a place-pair table

**What it is, exactly** (the build should reproduce `features.py` variant `v8` and `fit.py`
`Raster`):

1. **Lattice**: cells of 1/60 degree from 180 W and 90 S, as `world.water`.
2. **Water**: the T-1285 world mask (`data/terrain/world.water`, sea and large lakes). Every
   Wikidata fixed link (`data/raw/world_field/wikidata_links.json`, the bridges, tunnels and
   causeways T-375 joins landmasses with) of length 0.3-80 km turns the water cells within half its
   length plus 1 km of its coordinate into land. The two places' own cells are land.
3. **Roads**: GRIP4 density rasters (Meijer et al. 2018, 5 arc-minutes, metres of road per km² of
   land; `GRIP4_density_tp1` highways, `tp2` primary, `total`; 2 MB each from
   `dataportaal.pbl.nl/data/GRIP4/`, CC BY 4.0 there, CC0 on Zenodo). A cell takes its 5' cell's
   values: `hw`, `pri`, `tot`.
4. **Slope**: the mean slope in degrees of the GEDTM30 240 m pixels in the cell
   (`data/raw/gedtm30/gedtm_rf_m_240m_v1.2.tif`, as `zone_terrain.py` computes it; none = 0).
5. **Speed**, km/h: 20; 55 where `tot >= 100`; 85 where `hw + pri >= 25`; times
   `1 / (1 + max(slope - 3, 0) / 6)`; then 110, with no slope factor, where `hw >= 25`. Water cells
   6 km/h (ferries, bridges the mask misses, and a stiff price on a sea crossing the market treats
   as one landmass).
6. **Search**: eight neighbours; a step costs its length in km (111.32 x cos(latitude) / 60 east,
   110.57 / 60 north) times the mean of the two cells' 60 / speed. Dijkstra from each place's
   cell over a window reaching 1,500 km; `m` the least minutes to the other place's cell, `L` that
   path's length in km.
7. **The pair's drive**: `T = 24.0 + 1.252 m` minutes, road `K = 1.107 L` km (fitted on the
   13,470 pairs; at the median 1.5% slower than today's rule, so the car's level stays where Anita
   set it on 2026-10-24; multiply `T` by 0.985 for the rule's exact level).
8. **Use in the market**: per place pair on one landmass within 1,500 km, two ratios to today's
   rule, `rt = T / (15 + 1.25 x gc / 85 x 60)` and `rk = K / (1.25 x gc)`. Every drive the market
   prices between those two places (the corridor split, the solve's zone pairs, commutes between
   zones, the trip explorer) is today's `drive_min(d)` times `rt` and its road km `1.25 d` times
   `rk`, `d` being that zone pair's own distance as now. Within one place, and beyond 1,500 km,
   both ratios are 1. Stored as bytes `round(64 ln r)` like the packs' routes: about 1.5 MB for
   370k pairs. The airports' access drives keep the rule.

**What a build touches**: a new pipeline script (about 250 lines: the grids, the windows, the
table), its file beside `market-rows.json`; `intercity.rs` at its drive call sites (`car_min`'s
callers, the zone-pair and cross-zone commute cars, `market_trip.rs`), `marketRule.ts` and its
`t1148` check, docs/regional.md 5.1. The WASM gets the table at start, so the build agent needs its
own worktree. No pack is rebuilt and no join re-gated; a new place reruns the script (minutes).
Download: GRIP4's three files, 6 MB.

**What it does not fix**: traffic in dense corridors (the router knows roads, not congestion;
New York - Boston and Amsterdam - Brussels 10-13% fast), mountain motorways with low limits
(Zurich - Milan), and narrow seas the market joins as land: the Gulf of Finland (Helsinki to
Tallinn is one landmass through Russia, so today's market drives straight across it in 100
minutes; the router goes round by St Petersburg, 650 minutes) and Korea to China through North
Korea. Those two want a ferry or no car, not a better drive (T-280's ground).

**Room to improve at build time** (not done here, to keep the rule simple): a congestion slowdown
by GRIP total density near the ends, and a higher water cost. Each needs the truth sample, which is
saved.
