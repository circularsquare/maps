# Canoeing worldwide: data and model (T-003)

Research for the canoe layer (T-005 builds the design from this). Written 2026-10-14.
Everything marked **guess** is a number nobody has measured in the form we need; it is a
starting value to tune by looking at the map, not a finding.

Contents: 1 rivers, 2 lakes, 3 sea, 4 speeds and costs, 5 hazard, 6 10,000 BC,
7 recommended stage 1 model, 8 sources table.

---

## 1. Rivers

### Candidates

| Dataset | What it gives | Resolution | Licence | Size |
|---|---|---|---|---|
| **HydroRIVERS v1.0** | Every river with catchment >= 10 km2 or mean flow >= 0.1 m3/s as reach lines; long-term mean discharge `DIS_AV_CMS` (WaterGAP, 1971-2000), `UPLAND_SKM`, `LENGTH_KM`, `NEXT_DOWN`, stream order | traced from the 15 arc-second HydroSHEDS network; 8.5 M reaches, 35.9 M km | HydroSHEDS licence: free for scientific, educational and commercial use with attribution (custom WWF terms, not CC) | 544 MB global shapefile; 9-116 MB per region (Asia 91 MB). Asia already local: `maps/data/HydroRIVERS_v10_as_shp/` |
| **HydroSHEDS v1 core, 30 arc-second** | D8 flow direction (DIR), flow accumulation (ACC), DEM | 1/120 degree, **the same grid as the MAP surface** | HydroSHEDS licence | global DIR 83 MB, ACC 206 MB, DEM 306 MB; per region 2-48 MB |
| **SWORD v16 / v17c** (SWOT River Database) | Rivers >= 30 m wide: ~10 km reaches and 200 m nodes with **width, slope (m/km), water surface elevation, number of channels, obstruction type (0 none, 1 dam, 2 lock, 3 low-permeable dam, 4 waterfall from HydroFALLS), lakeflag, daily climatological river ice flag**, up to 4 upstream and 4 downstream neighbours (so braids and some distributaries exist) | vector | CC BY 4.0 | v16: 1.6 GB netCDF zip (all continents); v17c: 3.9 GB netCDF zip |
| **GRWL** (Global River Widths from Landsat) | Centrelines and widths at mean discharge, rivers >= 30 m | 30 m | CC BY 4.0 | vector 3.1 GB, mask 394 MB, summary stats 43 MB |
| **GloRiC** | Classes for every HydroRIVERS reach (hydrologic, physio-climatic, geomorphic; 127 reach types) | same reaches as HydroRIVERS | HydroSHEDS licence | 575 MB global shapefile |
| **RiverATLAS** | HydroRIVERS plus 281 attributes, incl. stream gradient and monthly min/max discharge | same reaches | CC BY 4.0 | 2.55 GB |
| **Andreadis et al. 2013** width/depth | Bankfull width and depth per HydroSHEDS river from hydraulic geometry | per reach | see Zenodo record | small |
| **MERIT Hydro** | 3 arc-second flow direction, width, upstream area; best Arctic coverage | 90 m | CC BY-NC 4.0 or ODbL (dual) | tens of GB global |
| **OSM** | `waterway=waterfall` 89,576 objects, `waterway=rapids` 127,046 (taginfo, 2026-10-13); dams, weirs, canoe portage tags | point/line | ODbL | per-tag Overpass pulls are small; the planet is not |
| **GROD** | ~30k human obstructions (dams, locks, weirs) on the GRWL network | point | CC BY 4.0 | small; already folded into SWORD |
| **HydroFALLS** | ~4,000 checked waterfalls | point | not yet released standalone (licence "coming 2024" on the product page); its points are inside SWORD as `obstr_type = 4` | n/a |

### What each one is good for

- **HydroRIVERS is the backbone.** It is the only open global set that reaches small rivers,
  carries discharge, and is light enough to process. Known gotchas (checked on the Asia
  extract in August 2026, see `maps/riverschematic`):
  - Discharge is **naturalised**: no dams, no irrigation. The Yellow River and Amu Darya look
    full. For 10,000 BC that is what we want (climate aside); for "today" it overstates
    rivers that are drained in practice.
  - **No distributaries.** `NEXT_DOWN` is a single id, so every delta is one channel. On a 1 km
    grid this matters little (a delta is mostly water and wetland cells), but the Hooghly or
    the Bassac do not exist as separate routes.
  - **Do not prune by discharge for connectivity**, because discharge falls along drying rivers
    and pruning cuts off their tails. For a canoe, though, a drying river really does become
    unpaddleable, so discharge is the right *navigability* test. Keep the tree built on
    `UPLAND_SKM`/`NEXT_DOWN`, and mark each reach paddleable or not by discharge; a gap is then
    a portage, not a missing piece of network.
  - North of 60 N the v1 network comes from HYDRO1k (1 km), coarser and less accurate.
- **Rivers are thinner than cells.** A river is a line through a 1 km cell, not a cell. So it
  should be a set of **directed edges between cells**, separate from the land layer, not a
  cell type (see section 7).
- **Flow speed.** No global set gives measured velocity. Two ways to estimate it:
  1. **Hydraulic geometry** from discharge alone. The global coefficients used by Andreadis et
     al. 2013 (from Moody and Troutman 2002) are width W = 7.2 Q^0.50 and depth
     D = 0.27 Q^0.39 (metres, m3/s; quoted from memory of the paper, check before use).
     Velocity is then Q / (W D) = 0.51 Q^0.11 m/s. Velocity barely changes along a network,
     which matches the long-standing observation (Leopold and Maddock 1953) that downstream
     velocity rises only slowly with discharge.
  2. **Manning** with slope: v = (1/n) D^(2/3) S^(1/2), D from hydraulic geometry, slope from
     SWORD (big rivers) or from the HydroSHEDS DEM along the flow path. This catches fast,
     steep reaches that hydraulic geometry averages away, but DEM slope at 1 km is noisy.
  Stage 1: use (1) for speed, and use slope only to mark rapids (below).

  Asia extract, by mean discharge, with the speeds hydraulic geometry gives (my run of
  `rivers_stats.py` over 1.43 M reaches, 5.58 M km):

  | Mean flow (m3/s) | Share of length | Width | Depth | Velocity |
  |---|---|---|---|---|
  | < 0.1 | 32% | 2 m | 0.10 m | 0.4 m/s |
  | 0.1-0.3 | 19% | 3 m | 0.14 m | 0.4 m/s |
  | 0.3-1 | 23% | 5 m | 0.21 m | 0.5 m/s |
  | 1-3 | 11% | 9 m | 0.33 m | 0.55 m/s |
  | 3-10 | 6% | 17 m | 0.5 m | 0.6 m/s |
  | 10-100 | 6% | 30-50 m | 0.8-1.3 m | 0.7-0.8 m/s |
  | 100-1000 | 2% | 130 m | 2.5 m | 1.0 m/s |
  | > 1000 | 0.7% | 400 m | 6 m | 1.2 m/s |

  Mean flow >= 1 m3/s keeps 1.47 M of Asia's 5.58 M km (26%); >= 2 keeps 1.01 M km.
- **Navigability.** A loaded canoe wants roughly 0.2-0.3 m of water and a few metres of width.
  By the table that is mean flow of about 1 m3/s. **Recommended threshold: DIS_AV_CMS >= 1**
  (**guess**, tune by eye). Mean flow overstates low-season depth; RiverATLAS has monthly
  minimum discharge if seasonality is wanted later (2.55 GB, so a later step).
- **Rapids and falls.**
  - Waterfalls and dams: SWORD `obstr_type` (rivers >= 30 m) plus OSM `waterway=waterfall`
    and `waterway=dam/weir` for smaller rivers. OSM coverage is uneven (Canada and Scandinavia
    well mapped, much of Africa and Siberia thin).
  - Rapids from slope: SWORD `slope` for big rivers; for the rest, the elevation drop along
    the 30 arc-second flow path divided by reach length. Whitewater rule of thumb (**guess**,
    not from a dataset): under 1 m/km is flat; 1-5 m/km has riffles and rapids that can be run
    or lined; above ~5 m/km on small rivers, or ~2 m/km on rivers above ~1000 m3/s, a laden
    canoe portages.
  - OSM `waterway=rapids` is a useful check where it exists, not a global source.

### Putting rivers on the grid

- **L1 (1 km).** HydroSHEDS 30 arc-second is 1/120 degree from whole-degree edges, exactly the
  MAP grid. Two ways in:
  - Rasterise each HydroRIVERS reach (lines run downstream) into the sequence of 1 km cells it
    passes through. Each consecutive pair is a downstream edge and its reverse an upstream
    edge. Spread the reach's real `LENGTH_KM` over its steps, so meanders cost their true
    length rather than the cell-centre distance.
  - Or follow the 30 arc-second DIR raster from each cell, with discharge taken from the
    HydroRIVERS reach that passes through. Simpler topology, but the DIR path at 1 km is
    already a smoothed river.
  The first is recommended: it keeps true lengths and the reach attributes.
- **L10 (10 km).** Same cell walk on the coarse grid; the edge time is the sum of the fine
  times along the river between the two coarse cells. Small rivers that stay within one
  10 km cell vanish from L10, which is fine: they only matter for short trips, which the view
  refine solves at L1.
- **Size.** HydroRIVERS has 35.9 M km of river; at >= 1 m3/s roughly a quarter (Asia's share),
  ~9 M km, so ~8-9 M downstream edges at L1 and as many upstream. At ~10 bytes an edge that is
  ~90 MB raw for the world, tiled like the speed tiles so a view loads only its own. L10 is
  ~1 M edges, ~10 MB.

## 2. Lakes and reservoirs

- **HydroLAKES v1.0**: 1.4 M lakes >= 10 ha, with area, shoreline length, mean depth, volume,
  residence time. CC BY 4.0. Polygons 820 MB shapefile, pour points 79 MB. **Already local**:
  `maps/data/HydroLAKES_polys_v10_shp/`. It includes reservoirs (so a "today" reservoir is a
  lake, not river) and the Caspian.
- Treat a lake as still water: canoe nodes on every lake cell, edges both ways at paddling
  speed, no current. HydroRIVERS reaches that cross a lake should be dropped inside it (the
  lake cells already connect inflow to outflow).
- Wind waves are what make a big lake dangerous. The fetch-limited wave formula (JONSWAP, as in
  the US Army Coastal Engineering Manual) gives Hs = 0.0016 (U^2/g) sqrt(g F / U^2), capped at
  the fully developed sea 0.21 U^2/g. With a typical wind of U = 7 m/s and fetch F = sqrt(lake
  area): a 1 km2 lake gives 0.1 m (harmless); Lake Superior (82,000 km2) hits the cap at about
  1 m. Use that Hs in the same wave hazard table as the sea (section 5), and the same distance
  from shore rule. U = 7 m/s is a **guess**; ERA5 wind over lakes could replace it later.
- 10,000 BC: HydroLAKES reservoirs should be removed (it has a reservoir flag via its link to
  GRanD; check the `Lake_type` field) and the river restored. Glacial lakes are section 6.

## 3. Coast and open sea

### Data

| Dataset | Gives | Resolution | Licence | Access and size |
|---|---|---|---|---|
| **NOAA Global Drifter Program climatology** (Laurindo et al. 2017, v3.10/3.11) | Near-surface current U, V (15 m drogue, includes wind-driven flow), plus SST; annual and monthly | 0.25 degree, 72.9 S-84.9 N | CC BY 4.0 (ERDDAP licence attribute) | **Open, no login.** AOML ERDDAP `drifter_annualmeans`, `drifter_monthlymeans`; annual U,V ~7 MB as float. Tested: Gulf Stream at 36 N 73 W reads 1.2 m/s |
| **ERA5 via ARCO-ERA5** (Google public bucket) | Hourly significant wave height (combined, wind sea, swell), sea ice cover, SST, 1940-present | 0.25 degree grid (waves native 0.5 degree) | CC BY 4.0 (Copernicus licence since 2 July 2025) | **Open, anonymous HTTPS/GCS**, zarr, one hour per chunk (~2-4 MB). A 30-year climatology from a sample of e.g. one hour every 3 days in the chosen months streams a few GB but stores ~100 MB. Checked: the store lists `significant_height_of_combined_wind_waves_and_swell`, `sea_ice_cover`, `sea_surface_temperature` |
| ERA5 monthly means (CDS) | Same, already averaged | 0.25-0.5 degree | CC BY 4.0 | needs a free ECMWF account (gated; do not register) |
| NOAA WAVEWATCH III hindcast; CSIRO CAWCR wave hindcast | Hourly wave fields | 0.5 / 0.4 degree | US public domain / CC BY 4.0 | open but GRIB2/thredds, monthly files; ARCO-ERA5 is easier |
| **NSIDC Sea Ice Index (G02135)** | Monthly median ice edge 1981-2010, shapefiles | 25 km | open (NOAA) | **Open, no login** (`noaadata.apps.nsidc.org/NOAA/G02135/`), a few MB |
| OSCAR, GlobCurrent / CMEMS | Satellite-derived surface currents | 0.25 degree | open but behind free accounts (NASA Earthdata, Copernicus Marine) | not needed; the drifter climatology covers it |

Not available openly: tidal currents (FES2014, TPXO need registration). Tides matter in narrow
straits (several m/s), which no climatology captures; stage 1 ignores them.

### Model

- **Sea cells are a canoe layer** on every water cell of the grid (land mask from the MAP
  surface's water value). Edges both ways to all 16 neighbours, as for walking.
- **Ground speed with a current** (the standard boat-routing triangle): paddling speed p,
  current c, edge direction d. The paddler steers to cancel the cross-current, so the speed
  along the edge is s = c.d + sqrt(p^2 - (c x d)^2). If the cross-current exceeds p, or s <= 0,
  the edge cannot be paddled. The same formula covers following and head currents, so the
  graph is directed. With p = 5 km/h (1.4 m/s), the Gulf Stream core (1.2 m/s) nearly stops a
  canoe heading against it and doubles its speed with it. Current is read at the edge midpoint.
- **Near-shore currents.** The 0.25 degree climatology is coarse at the coast and has gaps;
  fill gaps with zero and scale current down near shore (x0.5 within 5 km, a **guess**:
  coastal paddlers use eddies and slack water).
- **Distance from shore** (a distance transform on the land mask at L1) drives most of the
  sea hazard (section 5). It is a stand-in for "time since you last could land", which the
  search cannot track per path; it is a lower bound, so it errs safe.
- **Sea ice**: a cell is closed to the canoe when climatological ice cover is above 15% in the
  chosen season. Rivers: SWORD carries a daily ice flag for large rivers; stage 1 can close
  rivers whose mean air temperature for the season is below 0 C, or ignore river ice and
  assume the travel season.
- **Season.** Waves, ice and currents all change a lot through the year (North Atlantic mean
  Hs roughly doubles from summer to winter). Stage 1 recommendation: **local summer** (June-
  August north of the equator, December-February south), the season a canoeist would choose.
  A month switch can come later; it costs nothing but more prepared fields.

## 4. Speeds and costs

### Evidence

- **Voyageurs (fur trade, 18th-19th c.)**, large bark canoes with 6-12 paddlers: about 40-50
  strokes a minute; lake cruising about 5 mph (8 km/h); work days of 14-18 hours with a short
  rest ("pipe") every hour. Portages at about 3 mph (5 km/h) on cleared trails, each man
  carrying two 90 lb (41 kg) packs, several trips per portage.
  (McGill "North West Company" pages; Wolf Ridge notes; ecvoyageurs.)
- **Modern trippers** plan on 2-3 mph (3-5 km/h) averages on flat water; a double portage
  including loading averages about 0.75 mph (1.2 km/h) (paddling.com forum; hobbyist figures).
- **Dugout replica, Taiwan to Yonaguni (2019, published 2025)**: a 7.5 m cedar dugout made with
  replica Paleolithic tools, crew of five, about 225 km of open sea across the Kuroshio in a
  little over 45 hours, including some hours adrift: about 5 km/h overall, paddled through the
  night. Earlier reed and bamboo rafts failed (too slow for the current).
- **Oldest known boats**: dugouts, e.g. the Pesse canoe (Netherlands, about 8000 BC). Bark and
  skin boats probably existed earlier but do not survive.

### Recommended values (one representative canoe)

A light canoe (bark or skin, 2-4 m to 6 m, about 30-50 kg) with a small crew. Alternatives
noted after.

| Item | Value | Basis |
|---|---|---|
| Paddling speed on still water, p | **5 km/h** | trippers 3-5, dugout crossing ~5, voyageur crews 8 |
| Paddling hours per day | **8** (same day as walking) | keeps the map's clock one convention; voyageurs did 14+ as professionals |
| Downstream | p + v (mid-channel, full velocity) | |
| Upstream | max(p - 0.6 v, 1.5 km/h) | 0.6: hugging the bank where current is slower (**guess**); 1.5 km/h floor: lining or poling along the bank, which is how people went up fast rivers (**guess**). So upstream is slow, never impossible, except at falls |
| Unrunnable reach (steep or fall) | portage: carry overland | |
| Portage speed | MAP walking time x **2** | single carry with canoe and load; x3 for a solo double carry. Voyageurs on made trails lost little speed; trackless ground with a boat is much worse (**guess**) |
| Fall or dam | fixed **1 h** plus the carry around it | |
| Put in or take out | **0.25 h** each | unloading and loading (**guess**) |
| Sea | p, plus current by the triangle above | |

**Alternatives.** A heavy dugout (several hundred kg) is faster and steadier at sea but
cannot realistically be carried more than a few hundred metres (rollers, many people), so
portage would end the trip. A big crewed canoe (voyageur style) paddles at ~8 km/h for 14 h
a day. A solo kayaker: ~5.5 km/h but tiring sooner.

**Open-water days.** Crossings were paddled straight through (Taiwan-Yonaguni: 45 h). Making
offshore edges cheaper to reflect this creates an artefact: the search would leave the coast
just to go faster. Recommended: keep the cost at paddling hours everywhere, and count offshore
exposure as 3 real hours per paddling hour for the hazard (you are out there all day and
night). The map then shows crossings as slower than they were, but never invents a shortcut.

## 5. Hazard

### Evidence (thin)

- **Modern whitewater**: 0.87 deaths per 100,000 user-days on 30 US rivers, 1994-1998, all
  boaters (American Whitewater). At ~4 h a day that is ~2e-6 per hour, with helmets, life
  jackets and rescue.
- **Modern non-motorised boating**: 1.4 deaths per 10 million participation days (US Coast
  Guard 2006), ~4e-7 per hour.
- **Greenland kayak hunters (pre-modern)**: kayak hunting had high mortality; drowning was a
  cause but cold, wet and storm exposure killed more (Etudes/Inuit/Studies 2002). No usable
  rate. A rough reading: if a hunter's yearly drowning risk was ~0.5% over ~1,500 hours on the
  water, that is ~3e-6 per hour for skilled people near shore who chose their weather.
- **Cold water**: expected survival in the water is 30-90 minutes at 0-4 C and 1-6 hours at
  10-15 C (Ohio Division of Watercraft chart, from wartime data); cold shock in the first
  minutes is worse than these suggest. So a capsize far from shore in cold water is near
  certain death; in warm water you may right the boat.
- **What crossings people actually made by paddle**: Melos obsidian at Franchthi Cave
  (Aegean, from about 11,000 BC or later, dates contested) needed open-water legs of tens of
  km; reaching Sahul (Australia) needed legs up to ~90 km, much earlier; Taiwan-Yonaguni
  (225 km) was shown possible. Nothing suggests paddled crossings of many hundreds of km;
  the Pacific was settled by sail.

### Recommended rates (per real hour on the water; all **guesses** fitted to the evidence)

Prehistoric people without life jackets: the modern rates times ~5-10.

| Situation | Hazard per hour |
|---|---|
| Flat river or small lake, near the bank | 1e-5 |
| Runnable rapids (1-5 m/km) | 1e-4 |
| Portage, lining | as walking (T-006) |
| Sea or big lake within 3 km of shore | 2e-5 |
| Waves, offshore, by season-mean Hs: <0.5 m / 0.5-1 / 1-1.5 / 1.5-2 / 2-3 / >3 m | 0 / 2e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-2, times f(d) |
| f(d): share of the wave hazard by distance from shore d | d / 20 km, capped at 1 (beyond a day's return you cannot run for shore) |
| Cold water multiplier on the wave term (SST) | x1 above 15 C, x2 at 5-15 C, x3 below 5 C |
| Supplies (water, food), by distance from land | 0 under 250 km; rising linearly to 0.05 at 600 km and beyond (at 80 km a day, 250 km is ~3 days out) |
| Exposure offshore (d > 20 km) | 3 real hours per paddling hour |

**Checks** (worked by hand, summer):
- 20 km strait, Hs ~0.7 m: ~6 h, survival ~99.9%.
- Taiwan to Yonaguni, Hs ~1.2 m, warm: ~45 h of which ~35 paddling hours over 20 km out,
  counted as ~105 real hours at ~1e-3 per hour: survival ~90%.
- A 500 km crossing in Hs ~1.5 m: several days beyond 20 km and some time past 250 km: about
  20-50%.
- North Atlantic (Norway or Ireland to Newfoundland, 3,000+ km, Hs ~2 m in summer, cold):
  hundreds of hours past 600 km from land: survival effectively 0. Iceland by way of Shetland
  and the Faroes (legs of ~300 and ~450 km): poor but not zero, which fits Irish skin boats
  reaching it around AD 800 (with sail).

So "you can't canoe across the North Atlantic" comes mostly from the supplies term and
partly from waves and cold, while short crossings stay cheap. Every number above is to be
tuned on the map against these checks.

### A problem with fastest-path-only for the canoe

SPEC section 3 says the search minimises time and survival is reported along that path. On
water this will often pick a crossing a real paddler would refuse: straight across the Bay of
Biscay instead of round the coast, or open sea instead of a slightly longer coastal route.
The survival shown would then reflect an avoidable risk, and much of the coastal map would
read as deadly. The simplest fix stays a plain Dijkstra: minimise time + K x (cumulative
hazard), where K is how many hours you would spend to avoid one unit of hazard (K = 1000 h
means paying 10 extra hours to cut death risk by 1%). This is Anita's call (it reopens a
2026-10-14 decision); see the report's questions.

## 6. 10,000 BC (about 12,000 years ago)

- **Sea level** was about 55-65 m below today (Lambeck et al. 2014, global ice-equivalent;
  local levels differ by tens of metres near the ice sheets because of crustal rebound).
  10,000 BC sits at the end of the Younger Dryas cold spell (12.9-11.7 ka), so the North
  Atlantic was colder and stormier, with more winter sea ice, than today.
- **Land that was there**: the Bering land bridge (first flooded about 11,000 years ago,
  Jakobsson et al. 2017), most of the North Sea (Doggerland), the Sunda shelf joining Sumatra,
  Java and Borneo to Asia, Sahul (Australia with New Guinea), the Persian Gulf floor, the Yellow
  Sea. These held rivers that no modern dataset has (e.g. the Channel River, the Molengraaff
  rivers of Sunda).
- **Ice**: the Laurentide ice sheet still covered most of Canada east of the Rockies and the
  Scandinavian ice sheet most of Scandinavia. No canoeing on the ice; walking on it is T-004's
  question.
- **Glacial lakes**: Lake Agassiz (Manitoba and Ontario, varying hugely in size around this
  time), the Baltic Ice Lake (drained about 11,700 years ago), the Black Sea as a freshwater lake
  well below today's level. Lake Bonneville had mostly gone. The Sahara was entering its wet
  phase with more lakes and rivers.
- **Sources** (paleo coastlines and ice are T-004's job; listed here so T-005 knows they
  exist):
  - ICE-6G_C (VM5a) paleotopography, 10 arc-minute netCDF per time step incl. 12 ka; Peltier
    group site and the PMIP4 mirror. ICE-7G_NA for North America. Paleo DEM = modern DEM +
    (ICE-6G topography at 12 ka - at 0 ka); land where above zero and not ice.
  - 1 km rebound and paleotopography of glaciated North America (Scientific Data 2023).
  - Dalton et al. 2020: North American ice margins in 36 time slices, shapefiles on theghub.org
    (may need a free account; not checked).
  - DATED-1 (Hughes et al. 2016) for the Eurasian ice sheets.
- **Rivers for 10,000 BC.** Keep the HydroSHEDS network where the land is the same (natural
  discharge suits this era; the climate does not, so flows are today's). Remove modern
  reservoirs and dams. On exposed shelves, route rivers on the paleo DEM (D8 on a bathymetry
  model plus the sea level shift); with a known discharge only from the drainage area that
  feeds them. Under ice, none. Glacial lakes as lake cells from the ice-margin data. All of
  this is a later stage.
- **Sea climate for 10,000 BC**: stage 1 reuses today's currents, waves and ice, which is
  wrong mostly in the North Atlantic. TraCE-21ka (a transient climate simulation through the
  deglaciation) has sea ice and SST at 12 ka if this ever matters.

## 7. Recommended stage 1 model

**Era: today and 10,000 BC share the same water model; 10,000 BC only swaps the land mask
(from T-004) and drops dams and reservoirs.**

### Datasets

| Use | Dataset | Size to fetch |
|---|---|---|
| Rivers, discharge, topology | HydroRIVERS v1.0 (per region, or 544 MB global) | per region 9-116 MB |
| Rapids slope | HydroSHEDS 30 arc-second DEM (per region) along the river cells | 11-48 MB per region |
| Falls and dams | SWORD v16 `obstr_type` (1.6 GB zip: Anita downloads, or read one continent's file out of the zip by HTTP range) plus OSM `waterway=waterfall|dam|weir` by Overpass | Overpass pulls a few MB each |
| Lakes | HydroLAKES (local) | 0 |
| Currents, SST | GDP drifter climatology, monthly, via ERDDAP | ~50 MB |
| Waves, sea ice | ARCO-ERA5, sampled to a local-summer climatology of mean Hs and ice cover | streams a few GB, keeps ~100 MB |
| Sea ice check | NSIDC G02135 median extents | a few MB |

### Graph

- **Nodes**: the walking layer (as now), and a canoe layer: every sea and lake cell, plus every
  cell a paddleable river (mean flow >= 1 m3/s) passes through.
- **River edges**: directed, between consecutive cells of each rasterised reach. Downstream time
  = true length / (p + v); upstream = length / max(p - 0.6 v, 1.5 km/h), with v = 0.51 Q^0.11.
  Reaches steeper than the rapids limit, and falls and dams, become portage edges both ways.
- **Sea and lake edges**: 16 neighbours, time from the current triangle; lakes have no current.
  Sea cells with summer ice > 15% are closed.
- **Transfers**: walking to canoe and back at any cell where both layers meet (river cells, and
  land cells next to sea or lake), 0.25 h each way.
- **Portage**: open question in SPEC. Proposed default (leave the canoe never, carry it at
  walking time x 2) works as a third layer: "walking with a canoe", reachable from the canoe
  layer only, with no transfer back to plain walking. Then "you only have a canoe" is: start in
  the canoe layer; every arrival is at a canoe or carrying it.
- **Hazard**: per cell and layer from section 5, stored as a small class code per edge.
- **Clock**: paddling hours with the same 8-hour day as walking.

### Leave for later

Tides; wind other than through waves; seasonal low water on rivers (RiverATLAS monthly
minimum); distributaries and braids (SWORD); river ice by date; a month switch; paleo rivers
on the exposed shelves and glacial lakes; Younger Dryas sea ice; the heavy-dugout and
big-crew canoe variants; making crossings faster than coastal paddling (continuous paddling)
without creating the leave-the-coast shortcut.

### Performance note for T-005

L10 has ~7.5 M cells, about 5 M of them sea. Adding them all to the coarse search roughly
triples its work (now ~0.3 s). Options: 8 neighbours at sea instead of 16, or a coarser sea
grid away from coasts. Cells over ~600 km from land are almost never on a surviving path, but
a time-only search still crosses them, so they cannot simply be dropped.

---

## 8. Sources

Rivers and lakes
- HydroRIVERS: https://www.hydrosheds.org/products/hydrorivers
- HydroSHEDS core downloads (30 and 15 arc-second sizes): https://www.hydrosheds.org/hydrosheds-core-downloads
- HydroSHEDS about (HYDRO1k north of 60 N): https://www.hydrosheds.org/about
- HydroSHEDS v2 (Americas only so far, 1 arc-second): https://www.hydrosheds.org/hydrosheds-v2
- HydroLAKES (CC BY 4.0): https://www.hydrosheds.org/products/hydrolakes
- GloRiC: https://www.hydrosheds.org/products/gloric
- HydroFALLS: https://www.hydrosheds.org/products/hydrofalls
- SWORD v16 (CC BY 4.0, 1.6 GB netCDF): https://zenodo.org/records/10013982 ; latest v17c: https://zenodo.org/records/10013982/latest
- SWORD Product Description v17b (attributes incl. obstr_type, slope, ice_flag): https://hydroweb.next.theia-land.fr/docs/collections/resources/swot/Product_Description_Document/SWOT_PRD/SWORD_ProductDescription_v17b.pdf
- Altenau et al. 2021, SWORD paper: https://doi.org/10.1029/2021WR030054
- GRWL (CC BY 4.0): https://zenodo.org/records/1297434
- Andreadis, Schumann and Pavelsky 2013, global width and depth: https://doi.org/10.1002/wrcr.20440 ; data https://zenodo.org/records/61758
- OSM tag counts: https://taginfo.openstreetmap.org/tags/waterway=waterfall , https://taginfo.openstreetmap.org/tags/waterway=rapids
- HydroRIVERS gotchas: Anita's notes from `maps/riverschematic` (August 2026)

Sea
- GDP drifter climatology: https://www.aoml.noaa.gov/phod/gdp/mean_velocity.php ; ERDDAP https://erddap.aoml.noaa.gov/gdp/erddap/info/drifter_monthlymeans/index.html
- Laurindo, Mariano and Lumpkin 2017: https://www.aoml.noaa.gov/phod/docs/laurindoetal2017.pdf
- ARCO-ERA5 metadata (checked): https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zmetadata
- Copernicus licence moved to CC BY 4.0 on 2025-07-02: https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464
- NOAA WAVEWATCH III hindcasts: https://ncei.noaa.gov/archive/accession/NCEP-WAVEWATCH
- NSIDC Sea Ice Index G02135 (open folder): https://noaadata.apps.nsidc.org/NOAA/G02135/

Speeds
- Voyageur pace and portages: https://digital.library.mcgill.ca/nwc/history/12.htm , https://digital.library.mcgill.ca/nwc/history/08.htm , https://wolf-ridge.org/wp-content/uploads/2025/12/Voyageur-Canoe.pdf , https://ecvoyageurs.com/t2k/history.html
- Trip-planning averages: https://forums.paddling.com/t/if-you-were-planning-a-trip/21898
- Taiwan-Yonaguni dugout voyage: https://earthsky.org/human-world/stone-age-ocean-voyage-re-enacted/ , https://www.bairdmaritime.com/work-boat-world/research-environment-training/feature-scientists-replicate-ancient-seafaring-with-rudimentary-canoe (papers: Kaifu et al., Science Advances, 25 June 2025)
- Pesse canoe: https://en.wikipedia.org/wiki/Pesse_canoe

Hazard
- American Whitewater fatality rates: https://www.americanwhitewater.org/resources/articles/138/
- Greenland kayak hunting mortality: https://www.erudit.org/fr/revues/etudinuit/2002-v26-n2-etudinuit703/007645ar/
- Cold water survival chart and its limits: https://coldwatersafety.org/survival-estimates ; model study https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11234009/
- Franchthi and Melos obsidian: https://en.wikipedia.org/wiki/Franchthi_Cave

10,000 BC
- Jakobsson et al. 2017, Bering land bridge: https://cp.copernicus.org/articles/13/991/2017/
- ICE-6G_C: https://www.atmosp.physics.utoronto.ca/~peltier/data.php ; PMIP4 mirror https://pmip4.lsce.ipsl.fr/doku.php/data:ice_ice6g_c
- North America 1 km paleotopography: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10593785/
- Dalton et al. 2020 ice margins: https://theghub.org/resources/4878
- Lambeck et al. 2014, sea level since the LGM: https://doi.org/10.1073/pnas.1411762111
