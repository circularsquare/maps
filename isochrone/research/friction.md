# Our own walking surface (T-004)

Research for building the walking speed surface from primary layers, so that eras (today,
10,000 BC) and infrastructure (roads) can be toggled, and so mountains are sharper than in the
Malaria Atlas Project (MAP) surface. Written 2026-10-14.

Samples and scratch scripts used for the checks below are in the session scratchpad
(`t004/`), not in the project: one Copernicus GLO-90 tile (N46 E023, Apuseni mountains west of
Cluj, 5 MB) and the ICE-6G_C 1-degree files for 12 ka and 0 ka (1.5 MB each).

---

## 1. How MAP built theirs

Sources: Weiss et al. 2018, *Nature* 553:333 ("A global map of travel time to cities"),
methods read from the accepted manuscript on Oxford's repository
(https://ora.ox.ac.uk/objects/uuid:e4584ac4-b89c-488a-9f88-57b2fffc1aa8; the Nature page is
paywalled), and Weiss et al. 2020, *Nature Medicine* 26:1835 ("Global maps of travel time to
healthcare facilities", https://www.nature.com/articles/s41591-020-1059-1), whose methods say
the friction surface approach is "identical" to 2018 with newer OSM roads.

**Grid.** 30 arc-seconds (~0.93 km at the equator), 85 N to 60 S. Values are minutes per metre;
km/h = 0.06 / value. Built entirely in Google Earth Engine. The build script itself is not
published as far as I could find; the published "code" is the travel-time step (R, gdistance),
at https://malariaatlas.org/research-project/accessibility_to_healthcare/.

**Combination rule.** Every layer is turned into a speed; each pixel takes the **fastest**
layer present (a road crossing a river is taken as a bridge). National borders were the one
exception in the motorised surface: a fixed 1 hour crossing penalty on every border except
inside Schengen and UK-Ireland.

**Off-road walking speed by land cover** (MODIS MCD12Q1, IGBP classes, 500 m). Medians from an
online survey of 407 people asked how long it takes to walk 1 km through photos of each class:

| MODIS class | km/h |
|---|---|
| Evergreen needleleaf forest | 3.24 |
| Evergreen broadleaf forest | 1.62 |
| Deciduous needleleaf forest | 3.24 |
| Deciduous broadleaf forest | 4.00 |
| Mixed forest | 3.24 |
| Closed shrublands | 3.00 |
| Open shrublands | 4.20 |
| Woody savannas | 4.86 |
| Savannas | 4.86 |
| Grasslands | 4.86 |
| Permanent wetlands | 2.00 |
| Croplands | 2.50 |
| Cropland / natural vegetation mosaic | 3.24 |
| Snow and ice | 1.62 |
| Barren or sparsely vegetated | 3.00 |
| Urban and built-up (not surveyed) | 5.00 |
| Open water in land cover but not in the water layer (not surveyed) | 1.00 |

**Elevation factor** (from the standard atmosphere and the fall of VO2max with altitude;
GMTED2010 elevation), multiplied onto the land cover speed:

    elev_factor = 1.016 * exp(-0.0001072 * elevation_m)
    # 0 m: 1.02, 1000 m: 0.91, 3000 m: 0.74, 5000 m: 0.59

**Slope factor** (Tobler's hiking function, slope angle in degrees from GMTED2010 at 1 km):

    tobler_kmh  = 6 * exp(-3.5 * abs(tan(0.01745 * slope_deg) + 0.05))
    slope_factor = min(tobler_kmh, 5.0) / 5.0

Because slope *angle* is always positive here, this is always Tobler's uphill branch: walking
downhill is never faster, and a slope costs the same both ways. The paper names this as a
limitation, along with the 1 km slope missing finer relief.

    speed = landcover_kmh * elev_factor * slope_factor

**Roads** (motorised surface): OSM rasterised with the fastest road type per pixel, speeds from
country median OSM speed limits per road type (Supplementary Table 1 of the 2020 paper, an xlsx:
https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-020-1059-1/MediaObjects/41591_2020_1059_MOESM2_ESM.xlsx),
a global mean per type where a country has no data. Google's distance-to-road layer added
roads where it was under 500 m, as the generic "road" class. Railways 24.3 km/h.

**Water** (motorised): rivers from CIA World Data Bank II at 10 km/h, lakes from JRC Global
Surface Water (a 1 km pixel counts as lake only if fully covered by water present 80%+ of the
time) at 10 km/h, ocean 19 km/h.

**The walking-only variant** (2020): every road at 5 km/h, every water pixel at 1 km/h ("the
rate for traversing a wetland or swimming"). Everything else as above.

**Checked in our copy** (`data/raw/unzipped/2020_walking_only_friction_surface.geotiff`):

- No border penalty in the walking surface. A column across the straight Egypt-Sudan border at
  22 N (30 E and 27 E) stays at 0.0206 min/m (2.9 km/h, barren desert x elevation factor) right
  through the border. Across the US-Mexico line at 109.5 W the only change is a road at
  0.0120 (5 km/h).
- 0.0600 (1 km/h) is shared by real water and by "open water" land cover pixels, so the current
  grid's rule "0.06 = water, impassable" also blocks some marsh pixels MAP meant to be slow
  land. Minor; our own build will separate the two.

**What this means for us.** The recipe is small and fully known: a land cover table, two
multipliers and a max over layers. Everything era- or toggle-dependent (roads, land cover, water,
coastline) is a separate input, so we can rebuild it with our own layers and keep the toggles as
separate layers rather than one baked surface.

---

## 2. Elevation

### Global DEMs

| DEM | Cell | Licence | Size | Notes |
|---|---|---|---|---|
| Copernicus GLO-90 | 3" (~90 m) | Copernicus DEM licence: free, attribution, commercial use allowed | ~26,000 1x1 degree COG tiles; a hilly tile is 5.3 MB, so roughly 100-130 GB in all | No registration, AWS open data: `https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N46_00_E023_00_DEM/...tif`. Surface model (tree and building tops), 2011-2015 radar. Ocean has no tiles. https://registry.opendata.aws/copernicus-dem/ |
| Copernicus GLO-30 | 1" (~30 m) | same; a few tiles (Armenia, Azerbaijan) not public at 30 m | same tile at 30 m is 43 MB, so roughly 0.8-1 TB | Same bucket pattern, `copernicus-dem-30m`. Only for regional, on-demand refinement. |
| FABDEM | 1" | CC BY-NC-SA 4.0 (non-commercial, share-alike) | ~similar to GLO-30 | GLO-30 with trees and buildings removed. Better ground heights in forest, but the licence would bind the whole map to NC-SA. Not recommended. |
| MERIT DEM | 3" | CC BY-NC 4.0 or ODbL 1.0, user's choice | ~ tens of GB | Download password via a form: gated, so skip unless needed. |
| SRTM v3 | 1"/3" | public domain | | 60 N to 56 S only; Copernicus is better and covers the poles. |
| GMTED2010 | 7.5", 15", 30" | public domain (USGS) | 30" global ~ 1-2 GB | What MAP used. Fine for elevation, too coarse for slope. |
| GEBCO 2024 | 15" | free, attribution (GEBCO terms) | ~7.5 GB netCDF (also GeoTIFF tiles) | Land + sea floor. Needed for exposed shelves in 10,000 BC. https://www.gebco.net/data_and_products/gridded_bathymetry_data/ |
| ETOPO 2022 | 15", 30", 60" | public domain (NOAA) | 60" ~0.5 GB, 30" a few GB | Alternative to GEBCO; 30" is enough for a -60 m shoreline at 1 km. |

Recommendation: **GLO-90 for land**, **GEBCO 2024 (or ETOPO 2022 30") for sea floor** only
where it is needed for the 10,000 BC coast.

### How much 1 km understates steepness (measured)

Apuseni mountains tile (N46 E023, moderate hills to 1,800 m), slope from GLO-90 at 90 m
versus from the same DEM averaged to 30" (MAP's grid):

| | 90 m | 1 km |
|---|---|---|
| median slope | 8.9 deg | 3.3 deg |
| 90th percentile | 22.4 deg | 8.4 deg |
| 99th percentile | 33.0 deg | 14.4 deg |
| steepest cell | 58 deg | 26 deg |
| share of cells over 20 deg | 14% | 0.05% |

MAP's slope penalty averaged over the tile: time multiplier 2.33 when computed per 90 m cell
and averaged, 1.30 from the 1 km slope. Per 1 km cell the ratio has median 1.5, 90th
percentile 2.7, 99th percentile 4.7. The truth sits between the two: averaging every 90 m
cell overcharges, because a walker picks the gentle line through a cell (valley floors,
contours, switchbacks); the 1 km slope undercharges. A least-cost path on the 90 m grid gives
the right number (section 7).

### Anita's thin ridge (100 m wide, 1 km tall)

- At **1 km**, averaging the DEM spreads the wall over the cell: only a tenth of the cell is
  wall, so the cell's mean height rises by about 100 m. That is a ~5 deg bump, which in MAP's
  formula slows walking by about a quarter. **The barrier essentially disappears.** This
  happens in the source DEM, before any averaging of ours, which is why `docs/resolution.md`
  says no averaging can recover it.
- At **90 m** the wall is one cell wide and the step to its neighbours is 1,000 m over 90 m
  (85 deg). Tobler gives under 0.01 km/h, so it is effectively impassable except at gaps, which
  is right.
- At **30 m** passes and gorges a few tens of metres wide also survive. Few real barriers need
  this; 90 m is the sensible global floor, and 30 m an on-demand refinement for chosen ranges.

So: 90 m or finer is needed for the elevation input. Then the aggregation to 1 km and 10 km
must keep crossings by path, not by averaging (section 7), or the 90 m detail is thrown away
again.

### Walking speed against slope

All take grade g = rise / run (negative downhill).

- **Tobler (1993)**: `6 * exp(-3.5 * |g + 0.05|)` km/h. Fastest (6 km/h) at g = -0.05,
  5.0 on the flat. Tobler suggested x0.6 off-path. From one data set; overestimates steep
  descents somewhat. Directed values: grade 0.1 up 3.55, down 5.04; grade 0.3 up 1.76,
  down 2.50; grade 0.5 up 0.88, down 1.24 km/h.
- **Naismith (1892)**: 5 km/h plus 1 hour per 600 m of ascent; descent free. **Langmuir**
  corrections: subtract 10 min per 300 m of gentle descent (5-12 deg), add 10 min per 300 m of
  steep descent (over 12 deg). Simple, hiker-tested, already directed.
- **Irmischer & Clarke (2018)**, from GPS-tracked military cadets on and off path:
  `v = 0.11 + exp(-(100*g + 2)^2 / 1800)` m/s on path; off path the exponential term is
  scaled by about 0.67 (as implemented in common libraries; check against the paper).
- **Campbell et al. (2019)**: a Lorentz-shaped curve fit to GPS tracks of ~2,000 hikers; it
  does not drop as fast as Tobler on steep slopes. On-path only.
- A 2023 data-driven comparison (https://arxiv.org/abs/2303.16065) is a good reference for
  choosing among them.

Recommendation: **directed Tobler normalised to the flat**, multiplied onto the land cover speed,
so flat ground keeps the land cover speed and slope only changes it:

    factor(g) = exp(-3.5 * (|g + 0.05| - 0.05))
    # flat 1.0; gentle descent up to 1.19; grade 0.3 up 0.35, down 0.50

This uses the land cover table for off-path conditions (it already is off-path) and adds the
downhill/uphill difference MAP leaves out. The graph must then be **directed for walking** too,
which changes `SPEC.md` section 3 ("Walking stays symmetric"); see "For Anita". A cap is needed
on steep descents: walking down 45 deg is not faster than up, so treat |g| over ~1 (45 deg) as
impassable on foot in both directions.

**Hill shading on the page (T-045, display only).** The Terrain Tiles on AWS
(https://registry.opendata.aws/terrain-tiles/, Mapzen's joerd build, terrarium PNGs at
`s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`, z0-15, sea floor included,
CORS `*`). Open data; attribution required per source (SRTM, GMTED2010, 3DEP: USGS; ETOPO1: NOAA;
EU-DEM: Copernicus; ArcticDEM; and Canada, Mexico, Norway, Austria, UK, Australia, New Zealand),
listed at https://github.com/tilezen/joerd/blob/master/docs/attribution.md. The info panel names
them and links that list; the map attribution links it too. Not used by the model.

---

## 3. Land cover today

| Source | Cell | Classes | Licence | Size | Notes |
|---|---|---|---|---|---|
| **Copernicus Global Land Cover LC100 v3, 2019** | 100 m | 23 (FAO LCCS: closed/open forest by leaf type, shrubs, herbaceous, cultivated, urban, bare, snow/ice, water, herbaceous wetland, moss and lichen) | CC BY 4.0 | one global COG of 2.4 GB at https://s3-west.nrp-nautilus.io/public-land-cover/cgls-lc100-2019-cog.tif (mirror via source.coop); original on Zenodo https://zenodo.org/records/3939050 | **Recommended.** Separates forest leaf types like MODIS, so MAP's table maps over almost one to one. |
| ESA WorldCover 2021 v200 | 10 m | 11 | CC BY 4.0 | 3x3 degree tiles, 77 MB for one Romanian tile, ~100+ GB global; AWS `esa-worldcover` | Sharper, but no forest types (one "tree cover" class), so rainforest and pine forest would get one speed. Later, as a sharpener. |
| MODIS MCD12Q1 | 500 m | IGBP 17 | public domain | ~ tens of GB per year | What MAP used. Download needs a free NASA Earthdata login (registration), so skip. |
| ESA CCI Land Cover | 300 m | 37 | free with attribution | | Copernicus Climate Data Store needs registration. Skip. |

Draft speed table for LC100, MAP's survey values carried over where a class matches; **guess**
marks where I chose:

| LC100 | km/h |
|---|---|
| 111/113 closed needleleaf | 3.24 |
| 112 closed evergreen broadleaf | 1.62 |
| 114 closed deciduous broadleaf | 4.00 |
| 115/116 closed mixed or unknown | 3.24 |
| 121-126 open forest | 4.00 (guess: between woody savanna 4.86 and closed forest) |
| 122 open evergreen broadleaf | 3.0 (guess) |
| 20 shrubs | 3.6 (mean of closed 3.0 and open 4.2 shrublands) |
| 30 herbaceous | 4.86 |
| 40 cultivated | 2.50 (MAP's croplands; low for a walker on field edges, but it is their survey) |
| 50 urban | 5.0 |
| 60 bare / sparse | 3.0 (MAP); **guess** 2.0 for sand seas if we add an erg mask |
| 70 snow and ice | 1.62 |
| 80 permanent water | not walkable |
| 90 herbaceous wetland | 2.0 |
| 100 moss and lichen (tundra) | 4.2 (guess: as open shrubland) |

Water today: LC100 classes 80 and 200, or JRC Global Surface Water occurrence
(https://global-surface-water.appspot.com/, free with attribution, licence not rechecked; 30 m)
as MAP did. LC100 is enough
for a first version.

---

## 4. 10,000 BC

10,000 BC is about 12.0 ka BP (thousand calendar years before 1950). That is late in the
**Younger Dryas** cold spell (12.9-11.7 ka): ice sheets still large, Europe largely cold steppe
and tundra rather than forest, Baltic and Lake Agassiz still ice-dammed lakes.

### Sea level and coastline

Global mean sea level was roughly 50-65 m below today (Lambeck et al. 2014, PNAS,
https://doi.org/10.1073/pnas.1411762111; the Younger Dryas was a slow-rise interval between the
two meltwater pulses). It is not one number worldwide: land under and near the ice was pressed
down and the sea floor tilted. **ICE-6G_C (VM5a)** gives this per place as `Topo_Diff`, the
topography change from present, so no single sea level is needed:

    paleo_elevation = modern_DEM_with_bathymetry + Topo_Diff(12 ka)   # land where > 0

Checked in the 12 ka file: over deep ocean the difference is +45 to +66 m (median +48);
Doggerland +61 m, Beringia +53 m, Sunda shelf +41 m (all dry land then), Hudson Bay +1,429 m
(ice-covered), Baltic -109 m (pressed down, the Baltic Ice Lake). Land area 159 vs 145 million
km2 today, ice 27 vs 16 million km2, about 15 million km2 of exposed shelf.

- ICE-6G_C: 1x1 degree netCDF per 0.5 kyr, 26 ka to now, ~1.5 MB each, no registration:
  https://www.atmosp.physics.utoronto.ca/~peltier/data.php (file
  `datasets/Ice6G_C_VM5a/I6_C.VM5a_1deg.12.nc.gz`). No licence stated; the page asks users to
  cite Argus et al. 2014 and Peltier et al. 2015. ICE-7G_NA is the newer North American revision,
  same format. The 10-minute versions sit behind a password on the PMIP4 mirror.
- `Topo_Diff` is smooth, so interpolating the 1 degree field bilinearly onto 1 km is fine; the
  shoreline detail then comes from GEBCO / ETOPO.
- A ready-made check for North America: Godbout, Brouard & Roy 2023, *Sci. Data* 10:735,
  1 km paleo-DEMs from ICE-5G/6G/7G (GEBCO 2021 minus rebound), PANGAEA
  https://doi.org/10.1594/PANGAEA.947536 (CC BY 4.0).

### Ice sheets

- ICE-6G_C `sftgif` (ice fraction) at 1 degree: crude margins, fine for a first version.
- **North America**: Dalton et al. 2020, *Quat. Sci. Rev.* 234:106223, ice margin isochrones
  every few hundred years 18-1 ka, shapefiles (13 MB zip) on GHub:
  https://theghub.org/resources/4878/supportingdocs. Licence not stated on the page; check.
- **Eurasia**: DATED-1, Hughes et al. 2016, *Boreas* 45:1, time slices every 1 kyr from 25 to
  10 ka with most-credible / maximum / minimum margins, CC BY 4.0, via PANGAEA and the Bjerknes
  Climate Data Centre (https://bora.uib.no/handle/1956/11200).
- Greenland and Antarctica: today's ice from LC100 snow/ice plus ICE-6G, which is close enough.
- Walking on an ice sheet: MAP's snow and ice 1.62 km/h, before slope; hazard is the real
  limit (section 6).

### Proglacial lakes

No ready global layer. Two ways:

1. **Flood the paleo-DEM**: closed basins in `paleo_elevation` next to the ice margin that would
   hold water. Lake Agassiz and the Baltic Ice Lake appear on their own once the ice-dammed
   depression is filled to its spill point. Doable with a priority-flood fill on the 1 km paleo
   grid, but sensitive to the 1 degree rebound field.
2. **Draw the two big ones** from published outlines (Lake Agassiz at 12 ka; Baltic Ice Lake
   before its 11.7 ka drainage). Small, and better than nothing for a first version.

Recommendation: start with 2, try 1 later.

### Vegetation

- **Beyer, Krapp & Manica 2020**, *Sci. Data* 7:236 (https://doi.org/10.1038/s41597-020-0552-1),
  0.5 degree, every 1-2 kyr for the last 120 kyr, BIOME4 biomes plus climate, Figshare
  (Scientific Data's usual CC BY 4.0; check the Figshare page). **Era-correct** (a 12 ka
  slice) but coarse. Also in the R package `pastclim`.
- **Hengl et al. 2018 potential natural vegetation**, *PeerJ* 6:e5457: 20 BIOME 6000 biomes at
  1 km, ODbL 1.0, Harvard Dataverse https://doi.org/10.7910/DVN/QQHCIK (250 m version on Zenodo
  https://doi.org/10.5281/zenodo.3526619). Sharp, but it is vegetation for **today's** climate
  without people: it would put closed forest over Younger Dryas Europe, which is wrong for
  10,000 BC. Right for a "today without clearing" toggle.
- **HYDE 3.3** (CC BY 4.0, 5 arc-minutes, 10,000 BCE to 2023; 3.4 now out,
  https://public.yoda.uu.nl/geo/UU01/AEZZIT.html): cropland and pasture by era. At 10,000 BCE it
  is essentially zero, so it adds nothing for that era; it is the layer for in-between eras
  (1 AD, 1500) later.

Recommendation: **Beyer 2020 biome at 12 ka** for the 10,000 BC land cover, each biome given a
speed from the MAP table (tundra as open shrubland 4.2, steppe as grassland 4.86, boreal forest
3.24, tropical rainforest 1.62 and so on; a full mapping is a build-time table). Hengl PNV for
"today, no clearing".

---

## 5. Roads and paths as a toggle

- **GRIP4** (Meijer et al. 2018, *ERL* 13:064006; https://www.globio.info/download-grip-dataset,
  Zenodo https://zenodo.org/records/6420961): 21 million km of roads in 5 classes (highway to
  local), one global file geodatabase (>2 GB) or regional shapefiles, ~8.4 GB on Zenodo in all.
  Licence CC0 on GLOBIO/Zenodo, CC BY 4.0 on the Dutch portal; credit it either way. 2018
  vintage, harmonised, no footpaths.
- **OpenStreetMap** (ODbL; credit and share-alike on the derived database): far more complete,
  and it has `highway=track/path/footway`, which matter for walking in mountains (mule tracks
  over passes). Global extraction is heavy (planet ~80 GB pbf, or Geofabrik continent extracts
  with osmium); feasible but a stage of its own.

How MAP used roads for walking: every road pixel at 5 km/h, no difference by class; the road
wins over land cover when faster. Roads matter to a walker mostly in forest, cropland, wetland
and mountains, where off-road speeds are 1.6-3 km/h.

Recommendation: **roads are their own layer**, not baked into the land surface: per cell a
"best road class present" byte. The walking search takes max(land speed, road speed) where
roads are on, so the toggle costs nothing extra. Start with **GRIP4** rasterised to 1 km (and
the 90 m pass graph only where a road exists); add OSM tracks and paths later if mountains look
wrong. Bikes and cars later use the same layer with their own speeds per class.

---

## 6. Walking hazard

The survival model in `SPEC.md` is a constant hazard per hour, summed along the path. The order
of magnitude is the bar. Everything below is **rough**, with how each number was reached.

| Condition | Hazard per walking hour | Basis |
|---|---|---|
| Background, temperate, water and food around | ~1e-6 | Adult hunter-gatherer mortality ~1-2% a year (Gurven & Kaplan 2007, from memory, not rechecked), spread over waking hours. |
| Hot desert, carrying water, known route | 3e-5 to 1e-4 | Tucson Sector 2015: 21 deaths per 10,000 Border Patrol apprehensions (~0.2% per crossing) over a crossing of roughly 2-4 days of walking (Arizona press reports of Pima County / Humane Borders counts). Apprehensions undercount crossings, so this leans high. |
| Hot desert, no water | rises to 1e-2 to 1e-1 after ~1 day | Classic US Army desert table (FM 21-76, from Adolph's 1940s studies): at ~49 C shade maximum, about 2 days resting and 1 day walking without water; ~10 days below ~21 C resting. Quoted from memory, not verified here. |
| Severe cold, no shelter (polar winter) | 1e-4 to 1e-3 | Guess. |
| Ice sheet or glacier | 1e-4 to 1e-3 | Guess: crevasses, cold, no food or fuel. |
| High altitude above base camp (~5,000-7,000 m) | 1e-5 to 1e-4 | Himalayan mountaineering: crude mortality 1-40 per 1,000 people above base camp over expeditions of a few weeks (Weinbruch & Nordby 2013 review, https://himalayandatabase.org/Research/Weinbruch-HiAltMed2013.html). |
| Above 8,000 m | 3e-4 to 1e-2 | Same review: 4-134 deaths per 1,000 summiteers on a descent of 12-24 hours. |
| Dense tropical forest or swamp | ~1e-5 | Guess (injury, getting lost, disease). |

**A design problem worth flagging:** thirst and hunger are not constant hazards. Death in a dry
desert depends on how long since the last water, so a per-cell hazard that ignores history will
let a path cross a 300 km waterless stretch at the same risk as six 50 km stretches with wells.
Two ways within the current model: (a) make the desert hazard a function of **distance to
fresh water** (rivers, lakes, oases) times heat, which captures most of it; (b) later, carry a
"hours since water" state in the search. (a) is cheap and recommended.

Climate and water layers:

- **CHELSA V2.1** (1981-2010 monthly climatology, 30"; EnviDat, opendata.swiss "open" terms:
  commercial use allowed, citation asked; https://www.envidat.ch/metadata/chelsa-climatologies).
  Max temperature of the warmest month and min of the coldest month, ~100 MB a variable-month.
- **CHELSA-TraCE21k** (Karger et al. 2023, *Clim. Past* 19:439): the same 30" grid, monthly
  temperature and precipitation every 100 years back to 21 ka, so **12 ka climate at 1 km**.
  EnviDat https://doi.org/10.16904/envidat.211; licence listed as CC BY (2.0 on one mirror);
  check before use.
- **WorldClim 2.1** (30"): the download page states no licence; prefer CHELSA so today and
  10,000 BC come from the same method.
- **Global Aridity Index v3** (Zomer et al. 2022, 30", CC BY 4.0, Figshare): one layer for
  "how dry", if we want it.
- **HydroRIVERS / HydroLAKES** (HydroSHEDS, free with attribution): for distance to water.

Season is a real choice: crossing the Sahara or Siberia in July and January are different
trips. Annual means are the simplest start.

---

## 7. Recommended plan

One desktop, Python with rasterio / numpy / scipy, at most 6 cores. The idea: build a few
**1 km layers** once (land cover speed, elevation, roads, hazard, per era), plus **directed
crossing costs** from the 90 m DEM, and make every coarser level from those. Toggles pick layers
at search time.

### Keeping ridges and passes when coarsening

Averaging time per metre keeps the straight-through cost but loses passes (`docs/resolution.md`).
The cheap fix that is close to HPA* portals: for each coarse cell and each of the four
directions, compute the **least-cost crossing** through the fine cells with a sweep:

- Start along one edge of the coarse cell at cost 0. Step one fine column at a time towards the
  opposite edge; each step may move up or down one fine row, costing the fine time for that
  move (directed Tobler on the signed grade x land cover speed).
- The cheapest arrival on the far edge is that cell's crossing time in that direction; divided
  by the cell width it becomes a directed time per metre.
- For all cells at once this is about 10 (1 km from 90 m) or 120 (10 km from 90 m) vectorised
  numpy steps over the whole tile, so it costs about as much as reading the tile.

It finds the valley, gap or pass through a cell instead of averaging over the ridge, and it is
directed. What it misses: paths that double back inside a cell, and which **row** you arrive at
(an entry low on one side and exit high on another are not tied together). Full portals (several
entry points per side and the costs between them) fix that later; this sweep gets most of the
benefit first. Four directed values per cell instead of one speed: the 10 km grid goes from
7.5 MB to about 30 MB before gzip; still workable.

### Stages

| Stage | What | Inputs | Download | Time |
|---|---|---|---|---|
| **A. Today, MAP-style, our build** | Reproduce MAP from layers at 1 km: LC100 speeds x elevation factor x slope factor; water from LC100. Compare against MAP's surface (should be close off-road). Checks the pipeline. | LC100 (2.4 GB), GMTED2010 30" or GLO-90 averaged, ~1-2 GB | ~4 GB | a day of work, under an hour of compute |
| **B. Roads toggle** | GRIP4 rasterised to 1 km road-class byte; search uses max(land, road). | GRIP4 regional shapefiles | ~2-8 GB | a day |
| **C. Sharp mountains** | Stream GLO-90 tiles one at a time, compute the directed sweep crossings at 1 km (and 10 km) from 90 m elevation x 1 km land cover, keep only the results, delete each tile. | GLO-90, ~26,000 tiles | 100-130 GB streamed, never all on disk | 2-6 h, mostly download (at 10-20 MB/s); hand to Anita to run |
| **D. 10,000 BC** | Paleo-elevation = GLO-90 / GEBCO + ICE-6G `Topo_Diff` at 12 ka; ice from ICE-6G, refined by Dalton 2020 and DATED-1; Lake Agassiz and Baltic Ice Lake outlines; Beyer 2020 biomes at 12 ka to speeds; no roads. Shelves need stage C over the sea floor too (GEBCO at 15", much coarser than 90 m; fine for flat shelves). | ICE-6G (small), GEBCO 2024 or ETOPO 30" (~2-8 GB), Dalton (13 MB), DATED-1 (small), Beyer 2020 (~1 GB) | ~10 GB | 2-3 days of work |
| **E. Hazard** | Per era: heat (CHELSA / TraCE21k warmest month), cold, altitude, ice, distance to water -> hazard per hour per cell, from the table in section 6. | CHELSA (~1 GB for a few variables), TraCE21k 12 ka slice (~1 GB), HydroRIVERS/LAKES (~1 GB) | ~3 GB | 1-2 days |
| **F. Later** | OSM tracks and paths; GLO-30 for chosen ranges; full portals; WorldCover 10 m as a sharpener; in-between eras with HYDE. | | | |

Disk at the end: under 20 GB of kept inputs plus the derived layers (global 1 km grid of 4
directed uint16 crossing times is ~6 GB raw, ~1-2 GB compressed, per era).

Order: A then B gives a toggleable "today" quickly and validates against MAP. C is the big
quality step for mountains. D gives the 10,000 BC map. E can run alongside D.

---

## Main uncertainties

- MAP's land cover speeds come from a photo survey, not measurements; croplands at 2.5 km/h
  and rainforest at 1.62 km/h are the survey's word. Fine as a start, worth revisiting.
- The sweep crossing ignores turning inside a cell; good for passes, imperfect in a maze of
  small valleys.
- ICE-6G is 1 degree; coastlines near former ice sheets (Scandinavia, Hudson Bay, Scotland)
  will be off by tens of km in places until refined with a regional model or the Godbout 2023
  1 km grids.
- Vegetation at 12 ka from a 0.5 degree climate model is coarse and model-dependent.
- Every hazard number is a rough order of magnitude; several are guesses as marked.
- Licences to confirm before publishing: Dalton 2020 shapefiles, CHELSA-TraCE21k, Beyer 2020
  Figshare, ICE-6G (no licence stated, citation requested).

## For Anita

- **Directed walking.** Downhill faster than uphill means walking edges become directed too,
  which changes `SPEC.md` section 3 ("Walking stays symmetric"). Recommended; the canoe needs a
  directed graph anyway.
- **Vegetation for 10,000 BC**: era-correct but coarse (Beyer 2020, 0.5 degree), or sharp but
  today's climate (Hengl PNV)? Recommended: Beyer for 10,000 BC, Hengl for "today without
  clearing".
- **Season** for hazard: annual average, or a season switch (summer / winter)? Recommended:
  annual average first.
- **Stage C download** (100-130 GB streamed over a few hours): run by her, or in the background
  by an agent? Disk use stays small either way.
