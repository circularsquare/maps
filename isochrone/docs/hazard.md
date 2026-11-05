# Land hazard

**Approved by Anita 2026-11-05** (section 8). The band is built (`data/hazard/`, T-066); the page
does not read it yet (T-067). The L10 sample, its check trips and how the numbers were reached are
in `notes/T-042.md`. Sources and the checked numbers: research/friction.md 6.

The search minimises time + K x hazard on every layer (K = 1000 h, SPEC 3), and the hover shows
survival = exp(-sum of hazard x time) along the chosen path. Water hazard is built
(`docs/canoe.md` 3). This adds a hazard rate on land, read by walking, carrying and getting in and
out, so a walk across the Sahara or Greenland is no longer as safe as a walk in Normandy.

## 1. The traveller

A competent walker travelling light: clothing for the local climate, food for the way, 9.5 litres
(10 US quarts) of water filled at every drinkable source, no polar gear, no stove fuel, no rescue.
Risks that depend on who travels and on other people (disease, violence, getting lost) are not
counted; nor is a background risk, so a walk through farmland reads survival 100% and the hover
shows survival only where something here applies (water travel has its own small rates).

## 2. Factors

Rates are per **walking hour**, and stand for the whole day the walker spends there (8 hours
walked, the rest camped), since the walker is out all day and night. Carrying the canoe pays the
same rate over its doubled time.

    rate = min(1, thirst + cold + ice + altitude)        per walking hour, per L1 pixel

| Factor | Rule | Basis | How rough |
|---|---|---|---|
| **Thirst** | c (e^(k d) - 1), with d = km to the nearest drinkable water, k = 0.678 / S, c = 1.53e-4 / S, S = days survived walking with 9.5 L in that heat (FM 21-76 table, by mean daily max temperature: 14 days at 16 C and below, 13.5 at 21, 11.5 at 27, 8 at 32, 5.5 at 38, 3.5 at 43, 3 at 49) | The rule is fitted so a straight dry gap between two waters, walked at 25 km a day, has 1% risk when it takes S/2 days and 50% when it takes S days, rising steeply beyond (section 2.1) | The FM 21-76 survival days are measured (Adolph's 1940s desert studies); the 1%-at-half and the 25 km a day are choices. Order of magnitude, as the rest. |
| **Cold** | 0 while the night low (mean daily min of the month) is -10 C or above; then 1e-5 x 10^((-10 - Tmin) / 10): 1e-4 at -20, 1e-3 at -30, capped at 1e-2 from -40 | A guess; no usable source found | Rough. With the season rule below it fires almost only on ice and the highest ground. |
| **Ice** | 1e-3 on an ice pixel (LC100 snow and ice; where LC100 has no class, north of 80 N and on some Arctic islands, inside an ice outline), and ice is never drinkable water | Greenland ice-sheet teams with polar gear were evacuated ~3% of crossings 2010-12 (~1e-4 an hour); without polar gear about ten times that | Rough; thirst does most of the work far into an ice sheet. |
| **Altitude** | 0 below 3,000 m; 10^(-6 + (z - 3000) / 1000): 1e-6 at 3,000 m, 1e-5 at 4,000, 1e-4 at 5,000, 1e-3 at 6,000, capped at 1e-2 from 7,000 | Nepal trekkers (~1e-6 an hour, 3,000-5,500 m, lodges and groups); Himalayan climbers above base camp (1e-5 to 4e-4); summiteers above 8,000 m (3e-4 to 1e-2) | Order of magnitude; z is the pixel's mean height (GMTED, 1 km). |

Not factors: heat with water at hand (~1e-5 an hour at 50 C, Hajj 2024; small beside thirst),
swamps and dense forest (MAP's speeds already slow them; their danger is mostly disease), hunger
(weeks, and only where thirst or cold already rule).

### 2.1 Thirst

Thirst depends on how long since the last water, which a per-cell rate cannot know. The rate
instead grows with the distance to the nearest drinkable water, so a dry gap's hazard grows
steeply with its width: on a straight gap of width G the walker passes cells at every distance
from 0 to G/2 and back, and the rule is fitted to that path. Survival of one dry gap walked at
3.1 km/h:

| Mean daily max | S | 1% risk at a gap of | 50% at | Rate 20 km from water | 50 km | 100 km |
|---|---|---|---|---|---|---|
| 16 C and below | 14 d | ~190 km | ~350 km | 2e-5 | 1e-4 | 1e-3 |
| 27 C | 11.5 d | ~140 km | ~280 km | 3e-5 | 3e-4 | 5e-3 |
| 38 C | 5.5 d | ~70 km | ~135 km | 3e-4 | 1e-2 | 1 |
| 49 C | 3 d | ~40 km | ~75 km | 5e-3 | 1 | 1 |

A slower walker pays more over the same gap (rate x hours), which is right: it is longer
without water.

**Drinkable water** is any of:
- a river reach of RiverATLAS with mean discharge of at least 0.1 m³/s (a stream of 100 l/s),
  except in a **hot desert**;
- in a hot desert, only a perennial river: mean discharge at least 1 m³/s and at least 0.1 m³/s
  in its driest month (RiverATLAS `dis_m3_pmn`), such as the Nile, Niger or Euphrates;
- a HydroLAKES lake outside hot deserts (salt lakes are not told apart elsewhere: a known
  error, in a few endorheic basins);
- **today only**: an OSM well, spring or settlement (`man_made=water_well`, `natural=spring`,
  `place=city|town|village|hamlet`: a place people live has water). These are infrastructure,
  like roads: on with roads, off in 10,000 BC (**open**, section 5).
- never sea water, and never water on an ice pixel.

A **hot desert** is where annual rain is under 250 mm (CHELSA `bio12`) and the warmer of the
January and July day highs is at least 25 C. There WaterGAP's discharge gives big dry catchments a
mean flow over 0.1 m³/s (wadis), and HydroLAKES' lakes are mostly salt or seasonal. Cold deserts
(the high Arctic, 100-290 mm) keep their streams, which run each summer.

d is measured on a 2 km grid (1/60°) of these, great-circle, to 2,000 km. A cell counts as water
when a river vertex, a lake pixel or a point falls in it.

### 2.2 Season

Each pixel takes **the kinder of January and July**: thirst and cold are worked out for both
months, and the month with the lower sum is used (the Sahara in January, Siberia in July). It is
the season a traveller would choose, as the canoe's sea is local summer (`docs/canoe.md` 4). It is
per place, so one long trip may use January in one place and July in another. Ice and altitude
do not change with the season. A season switch (a fixed month for the whole map) is a later
extra: the build makes both months anyway.

## 3. Sources

| Input | Source | Licence |
|---|---|---|
| Night low and day high, January and July; annual rain | CHELSA V2.1 1981-2010 monthly `tasmin`, `tasmax` and `bio12`, 30" (data/raw/chelsa/, 990 MB) | CC0 1.0; citation asked |
| Rivers | RiverATLAS v1.0 reach lines with `dis_m3_pyr`, and `dis_m3_pmn` for Q >= 1 (data/raw/rivers_work/all_lines.npz and reaches.npz, kept from the canoe build) | CC BY 4.0 |
| Lakes | HydroLAKES (rivers_work/lakes_l1.npy) | CC BY 4.0 |
| Wells, springs, settlements | OSM from QLever's planet endpoint (data/raw/osm_water/, 4.7 M points) | ODbL |
| Ice | LC100 2019 class 70 (data/surface/lc_class.tif); where LC100 has no class, RGI 6.0 regions 03, 05, 07, 09 (OGGM's open mirror) and Natural Earth 10 m glaciated areas (data/raw/ice/) | as LC100; NE public domain; RGI CC BY 4.0 |
| Height | GMTED2010 mean (data/surface/elev.tif) | public domain |

The info panel credits CHELSA (Karger et al. 2017, 2021) and OSM when the band is on. For 10,000
BC the same rule reads CHELSA-TraCE21k at 12 ka and ICE-6G ice, and drops the OSM points (T-012).

## 4. Storage and reading

**Coding** (uint8, like the canoe's `shore` and `kind`): 0 = no hazard; c = 1..240 is a rate of
10^((c - 240) / 40) per walking hour, so 240 = 1, 200 = 0.1, 160 = 0.01, 120 = 1e-3, 80 = 1e-4,
40 = 1e-5, 1 = 1.06e-6. Steps of 5.9% (within 3%), six decades. Rates under 1e-6 are 0: over the
whole 128-day limit they add under 0.1%. 241-255 are unused.

**Files**, beside the speed and canoe data, the same tile grid and names:

| File | Level | Size (gzip) |
|---|---|---|
| `data/hazard/l10.bin.gz` | L10 world, 4320 x 1740, row-major from the north-west | 1.1 MB |
| `data/hazard/l3/<r>_<c>.bin.gz` | L3, 5° tiles of 200 x 200 | 9.3 MB, 1,128 tiles |
| `data/hazard/l1/<r>_<c>.bin.gz` | L1, 5° tiles of 600 x 600 | 29.3 MB, 1,140 tiles |
| `data/hazard/index.json` | grid, coding, the tiles per level (`levels["1"].tiles`, ids r x 72 + c), as `data/tiles/index.json` | |

39.7 MB in all, against 135 MB of speed tiles. Each file is gzip of one uint8 band, as the speed
tiles. L0.1 windows read the L1 pixel under each cell. A separate band, not a byte inside the
speed files: an era (T-012) or a season switch swaps the hazard files and leaves the speeds
alone. A tile with no hazard anywhere is not written (all 0): the reader treats a missing tile
as 0.

**Coarse levels**: an L3 or L10 cell's rate is the time-weighted mean of its L1 land pixels'
rates, weights the pixels' time per metre (the same weights as the cell's speed, so a coarse
straight crossing pays the same hazard as the fine one). A coarse cell with an avoidable bad
corner therefore reads a little worse than the best line through it, as its speed does; the finer
windows correct it.

**What the worker multiplies.** When a level loads, one table per cell: hq = rate x hours per km
(`inv` of its speed), the hazard per km walked. Then:
- a land move: dh = length x f x the mean of hq over the cells whose `inv` the move averages
  (two, or four for a knight move), f = 1 walking, 2 carrying;
- getting in or out: dh = 0.25 h x the land cell's rate, plus the half step on land as a move;
- weight += dt + K x dh, hazard += dh, exactly as water edges.

A walking start then needs the time and hazard arrays: `solveWalk` gains them (or walking runs
the general loop; whichever is faster, measured in stage 2). The stop rule is unchanged: a path
stops when its survival falls under 1%.

## 5. What it does to routes

K stays 1000 h: a cell's weight is its time x (1 + 1000 x rate), so 1e-4 an hour costs 10%
more, 1e-3 twice, 1e-2 eleven times. Over most land the rate is under 1e-5 and routes do not
move: from Cluj, K = 1000 changes the time to the cells reached by 0.1% on average and only 0.2%
of them by more than 5%. Where it bites (the L10 sample, with OSM water, notes/T-042.md):

- **Deserts**: routes keep to rivers and wells. Tamanrasset to Agadez bends east by wells (+1
  day, survival 87% to 99%). With natural water only, Cairo to Khartoum's fastest line cuts the
  Nile's bend across the Nubian desert (2% survival) and K = 1000 follows the river (+11 days,
  96%). In July (the season rule
  turned off), the fastest Tamanrasset to Agadez line would have 2% survival and K = 1000 takes a
  route 7 days longer at 95%.
- **Greenland**: straight over the ice sheet is 66 days and 1% survival; K = 1000 walks round the
  south coast instead (88 days, 84%).
- **High passes**: Leh to Manali shifts off the most glaciated cells (+2 days, 91% to 96%);
  Kathmandu to Lhasa is unchanged (99%).
- **Siberia, steppe, the outback**: unchanged or within a day, 99-100%, since the season rule
  takes Siberia in July and the steppe in January.

No absurd detours were found. The one long one: Tamanrasset to In Salah in July is out of reach
without K (every line under 1%), and with it a 127-day loop by wells at 37%, at the time limit.

**Unreached land.** The stop rule (survival under 1%) leaves the worst deserts and the ice
sheet unpainted on a walking start, as the open ocean on a canoe start: Greenland's ice, the
Libyan desert, the empty quarters of Mali, Mauritania and Arabia. Without the OSM water points
the whole Sahara and Arabia away from the Nile and the Euphrates are out of reach, and a start in
Riyadh reaches 21 cells.

## 6. Build

Plain `python` from the maps root, one core:

    python isochrone/tools/hazard/fetch.py           # CHELSA, OSM water points, ice outlines -> data/raw/  (~6 min)
    python isochrone/tools/hazard/build.py ice       # ice outlines north of 60 N to the L1 grid             (~15 s)
    python isochrone/tools/hazard/build.py dist      # distance to drinkable water, 2 km grid                 (~3 min, 5.7 GB)
    python isochrone/tools/hazard/build.py tiles     # rates per L1 pixel -> data/hazard/                     (~7 min, 5.2 GB)
    python isochrone/tools/hazard/build.py plot      # maps/artifacts/isochrone/t066_world.png

`tiles` works one row of 5° tiles (600 L1 rows) at a time: the rate per L1 pixel, coded into the
L1 tiles, and the time-weighted L3 tiles and L10 rows from the same strip. Land is the speed
tiles' land (MAP's, small water as land), and a coarse cell keeps its rate only where the speed
level has land (a quarter of its pixels). `data/hazard/index.json` has the grid, the tiles that
exist per level and the coding, as `data/tiles/index.json` does. The distance grid covers the
same land (small water included), so every land pixel has a distance. Work files:
`data/raw/hazard_work/` (`ice_l1.npy`, `dist_g2.npy`, ~0.5 GB), kept for a rebuild.

## 7. Known approximations

- Where LC100 has no class (north of 80 N, Svalbard, some Arctic islands), ice is the union of
  RGI 6.0's glaciers and Natural Earth's 10 m glaciated areas. RGI leaves out the Greenland ice
  sheet, so its northern edge comes from NE, generalised to a few km; GIMP's mask would be
  sharper but is behind an Earthdata login (NSIDC). NE also draws some caps a little wider than
  RGI does.
- A coarse cell's rate is a time-weighted mean, so a valley route past glaciers pays some of the
  glaciers' rate at L10 and L3 (Leh to Manali); L1 windows correct it.
- Salt lakes outside hot deserts count as water; seasonal rivers outside hot deserts count all
  year.
- Distance to the nearest water stands in for time since the last water, which puts a gap's
  danger in its middle rather than at its far end; the total over a straight gap is right, a
  path that wanders inside a gap pays more than the real walker would.

## 8. Anita's decisions (2026-11-05, DECISIONS.md)

1. **Today's wells, springs and villages are water** on today's map (with roads, as
   infrastructure); 10,000 BC drops them. With them the Sahara is crossable along wells
   (Tamanrasset to Agadez 99%) and only its emptiest parts go unpainted.
2. **Season**: the kinder of January and July at each place. One month for the whole map would
   be a later switch, a second set of hazard files.
3. **Unreached land**: survival under 1% stops a path on land as on water, so Greenland's ice and
   the empty deserts stay unpainted on walking starts.
4. **The traveller** as in section 1; disease (malaria for an outsider in West Africa) would
   change the tropics most and depends on who travels.

Named deserts as areas (Anita: fine "if it's easier and has similar effect") are not used: the
distance to water already covers them and every dry gap outside them. A desert name could later
help the hover say why survival drops.
