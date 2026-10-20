# Adding a city

Two ways. **A world plan row** goes through `pipeline/world_run.py` with no agent per city ("Running
the pipeline", then sections 1 and 3 for the rules it applies). **By hand** is for a country that needs
tailoring: a new country module, a typed-in table, a survey, a fitted curve (the steps in sections 1
to 12, the substitution table and worked examples under "Outside the US"). Fix this file wherever the
next run finds it wrong; timings and older worked-example detail are in
`notes/adding-a-city-history.md`.

Rules that apply: CLAUDE.md "Working here" (pragmatic metro extent, basic checks, no extent study or
calibration for plan rows) and "General sources and methods only". A US city uses US national sources
(LODES, Census blocks, ACS, TIGER, NCES, IPEDS, FAA). A city in another country uses tier 0 from global
sources, with a country module replacing inputs one at a time ("Outside the US").

Logs, check outputs and images go in `../artifacts/trainworld/` (from the maps root
`artifacts/trainworld/`), never in `notes/`.

## Running the pipeline

A world plan row (section 1) needs no agent: `pipeline/world_run.py` (T-540) does sections 3 to 7
and 11 for it, with the gate (section 8) deciding ship or park. From the maps root:

    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py plan --limit 20    what a run would do
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --limit 20     the next 20 rows
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py status             rows by status

`--rows a,b` or `--countries GB,PT` instead of `--limit`; `--jobs N` the cores in all (default 2,
up to 12 on an overnight run Anita approves; one city at a time per 3 cores); `--port` its own Vite
for the browser check (default 6360). Set `TW_SIM_DIR` and `TW_CARGO_TARGET_DIR` when the tree's
`sim/` may be mid-edit (section 4). Run it in the background and wait once for the process.

- **Zone rows** (whole-country cover, T-723, docs/zones.md 2): `world_plan.py zones --countries ES`
  (run through `world_run.py step world_plan zones ...`, which puts the pipeline on the path) writes
  a country's zone rows into the plan; then `run --countries ES --limit 100`. A zone row builds as
  a metro row does, with `kind` "zone" and the zone curve.
- **Rows** go in plan order, a region at a time: rows sharing a Geofabrik extract, chained. The
  region's extracts are downloaded (four tries) unless their caches or `.osm.pbf` are on disk, and
  read once into the caches (`osm_features`, inland water, roads, admin relations).
- **Per row**: `build_city`, `pack_gate stations` (the gate's station file, section 8), `places`,
  `routes --keep-pbf`, `merge_cells` (which runs `read_pack`'s checks on its output before it
  replaces the pack, and fails the step otherwise), `nocar`, `everyday_rates`, each its own process
  and log; a failed step runs once more after a minute (a timed-out one does not), then the row
  parks. A step whose log has not grown for 20 minutes (`QUIET_S`; `TW_QUIET_S` to change it) dumps
  every thread's stack into its log, again after each further 20 silent minutes, and the runner
  prints a line saying so. Status `built`.
- **Gate**: a row within 50 km of a pack with a Northeast `market` id parks (T-580: the Northeast
  is solved on its own sites). The rest are registered with their market packs (`register.py
  --market`; the browser check needs them), a snapshot of every registered pack taken first, and
  gated with `--browser --since` on the runner's own Vite, started after the registration (it
  reads the registry once). A pack that ships stays registered; one that parks is taken out again
  with its market pack, its market place left unused (place ids are positions), and its failing
  checks are its reason. The browser check serves the main tree's app/, so whoever runs a batch
  says who may rebuild app/ (its WASM) while it runs.
- **Joint gravity**, once for the batch after its last region (T-723; a run stopped before it
  leaves the gated rows registered and `built`, for the next run to gate again): a pack with no
  country in its header (tor, lon, par, sao) takes the registry's
  `country`. Each group a gated pack joined is solved once (members back to res-9,
  `intercity.py --group auto <new> --no-market` on every core of `--jobs`, with four times the step
  limit for a group of over 100 packs, merged again) and gated again without the browser
  or rates; a FAIL restores every member from the backup taken before. When only new packs FAIL,
  they park and the group is solved once more without them (a FAIL then parks every new pack of
  that solve); when an older member FAILs, every new pack parks (T-773). Then `register.py
  <shipped and changed> --market --trips` (or `market_zones.py --rows`
  when only parks happened), and a check that every file `vite.config.ts` copies into a
  production build exists for each shipped pack (it copies the registry's packs, so nothing more
  needs listing). Status `shipped`.
- **Clean up**: the region's `.osm.pbf` files and the OSM caches the runner made go once the
  region is done (`--keep-downloads` keeps them); caches already on disk before, and the small
  `.admin.fgb`, stay.
- **Status** is each row's `status` in `pipeline/world_plan.json` (`planned`, `built`, `shipped`,
  `parked`) with a `run` record: batch, seconds a step, gate warnings, a parked row's reason. A
  batch's report (shipped, parked with reasons, warnings, times) and every log go to
  `../artifacts/trainworld/world_run/<batch>/`.
- **Stopping**: ctrl-c, or the file `trainworld\data\work\world_run\STOP` (the run's first line and
  `status` print its path; with `--root`, that tree's); the city under way finishes its
  steps (they run in their own process group, so ctrl-c does not reach them). Gating and joining
  wait for the next run, which takes up `built` rows without rebuilding them and first takes any
  unshipped row out of the registry. The run ends by printing `status`.
- **Run faults** stop the run and park nothing for them (T-764, T-773): a `TW_SIM_DIR` copy whose
  `data/packs` or `app/src/game/registry.json` beside it is not this tree's (a missing `data` or
  `app` junction is made again to this tree's and noted in the report's problems, T-779; checked
  at the start, before and after each gate, and before each joint solve), and a gate or re-gate
  that leaves a row with no result file (the gate itself died; a pack's own error is a FAIL in its
  result). The rows stay `built`, the region's downloads stay on disk and a joint group is
  restored; fix the fault and run again. The
  runner exits with code 2 and the report's `fault` says what happened.
- **A scratch tree** (`--root data/work/<task>/trainworld`, with `data/raw` a junction to this
  tree's, `sim/` a copy, a registry, the world plan, `borders.json`, `intercity.json`, the plan
  extents and data/work's shared inputs: `world_field.*`, `outside.json`, `tiger_counties_simple.json`)
  takes every step's paths into data/packs, data/work, app/ and the plan instead; the browser check
  is off there. Remove the junction (`rmdir`) before deleting such a tree.

## 1. Choose the city and its footprint

- **A new country first: fill its row in `docs/data-register.md` and pick the procedure
  (archetype) before building.** The archetype names the worked example below to follow.
- **Which cities, and where their edges go: the world plan** (T-536, notes/T-535.md).
  `python pipeline/world_plan.py` writes `pipeline/world_plan.json`, one row per metro in build
  order with its units, origin, name, id, extracts, time zone and archetype, and the extents to
  `data/work/worldplan/extents.geojson`; `check` adds the eight-pack test and the next-50 picture.
  People are GHS-POP 2025 at 1 km, a pixel counted where its centre falls. The rule:
  - **Metros.** Outside Europe and the US, GHSL urban centres (UCDB R2024A) over 1M people; a
    centre's area is the centre, every functional urban area (GHS-FUA R2019A) it holds half the
    people of, and the FUA around its population-weighted centre; centres sharing a FUA are one
    metro. In Europe (geoBoundaries' continent: Russia in, Turkey out), FUAs over 1M instead.
    A smaller urban centre outside every metro's area whose edge lies within 3 km of a metro's
    own centres (in Europe, the centres inside its FUA) joins the nearest such metro's area, since
    GHS-FUA gives a town grown up to a metro a FUA of its own (Ikorodu beside Lagos; T-556).
  - **Order.** The UK, then the rest of Europe, then everywhere else, each largest first. A
    rerun keeps each earlier row's order and id: a row inherits them from the previous plan's row
    it shares the most people with; rows with none follow the last.
  - **Units.** NUTS-3 2024 in Europe, ITL3 in the UK; elsewhere geoBoundaries gbOpen at the finest
    of ADM2-4 where the median person's unit holds 20,000 people or more. A first-level unit that
    level misses a tenth of the people of takes the finest level covering it (Moscow). No
    geoBoundaries at all: the territory whole (Hong Kong, Macau).
  - **Extent.** A unit is in when half its people live in the metro's area, the larger share
    winning. A unit with half its people in an existing pack is taken, and an extent is cut along
    every existing pack and every earlier row, so nothing overlaps. A metro left with no unit takes
    the one at its area's densest free square km; one under 1M people after this is dropped, with
    its reason, in the file's `dropped` list.
  - **Over 40M people**, split along the coarsest admin level whose units fit, each piece grown
    by adjacency, the first from the unit group nearest the metro's densest square km, the rest
    from the most populous group left; a piece under 1M joins a touching piece it fits in, else is
    dropped.
  - **Name**: the centre's (or FUA's); a split piece or the rest of an existing pack's metro takes
    its largest second-level unit's name; a name not in Latin script falls to the next that is.
    **Id**: the first unused of the name's first three letters, then its first two with each later
    letter, then its first with any two later letters.
  - **Origin**: the densest 1 km of the extent. **Time zone** from the origin (timezonefinder),
    `utc_offset` its standard offset rounded down.
  - **Extracts** from Geofabrik's `index-v1.json`: from the continents down, the extract or set of
    children covering the outline with the least total area; none inside another.
  - **Archetype**: the country's in the register (England's A only for English units), else D.

  The rest of this section is for packs made by hand.
- **Where.** Anywhere between 85 S and 85 N: new games are in the Mercator world frame, whose
  track lengths are ground lengths everywhere (T-306, docs/world.md), and the renderer anchors to
  the view (T-307). A zone's market pack keeps its sites in its own pack frame (the Northeast's in
  the New York one) and the market measures distances on the ground (T-128, T-377).
- **Footprint.** Whole county-level units: US 2020 counties (`cb_2020_us_county_500k` GEOIDs;
  Connecticut's old counties), Canadian census divisions, the country's equivalent elsewhere. The
  metro area (OMB, BLS's list, StatCan's CMA) trimmed or widened by judgement to whole units; no
  study. Whole units are also docs/zones.md's unit for non-city zones. **No unit may belong to
  another pack**: compare with every `CITIES[*]["counties"]` in `build_city.py`.
- **A national border** inside the extent needs a factor (SPEC 4, "Borders outside the US") and,
  in the US, a water mask that knows the other side is land (Canada's adapter adds the US states;
  the US one treats everything outside the TIGER states as sea, so Buffalo or Detroit would need
  Canadian land added).
- **Id.** Three lowercase letters, unused.
- **Draw the outline before any build** (T-288: Medway's hole cost a second build): a quick
  picture of the chosen units, checked for holes, islands and gaps against the neighbours.
- **Reference figures with sources**, gathered before the build, never from memory: population
  (census), jobs in the core and in all, the commute mode share, one real line's daily riders.
  Look for population and jobs by district (for the `population` and `jobs` hooks) before the
  first build too, and print GHSL's total in each unit against its census figure: Shanghai found
  its jobs table late and Mumbai its mismatched talukas late, and each built twice (T-291, T-292).
- Take the task line in TASKS.md (`taking, <date>`), start `notes/T-###.md`.

## 2. Back up

`data/work/pre-T###/`: `data/packs/intercity.json` and `app/src/game/registry.json` (the only
existing files a city addition overwrites). Existing city packs are never rebuilt by a city
addition; take their sha256 now so you can show they did not change. If the batch edits shared
pipeline code, rebuild one existing pack into a scratch folder (patch `build_city.PACKS` and
`WORK`) and show its arrays are bit-identical.

## 3. Pipeline config

**A planned city** (a world plan row, T-537) has no hand-written entry: `build_city.py` adds every
`planned` row of `pipeline/world_plan.json` to `CITIES` through `pipeline/plan_config.py`
(`plan_config.py <id>` prints one), so `build_city.py <id>` builds it at tier 0 from global
sources. Each field comes from the row or a rule:

- **Outline**, `pipeline/osm_units.py`: each plan unit takes the OSM `boundary=administrative`
  relation (any admin level, from the row's extracts, cached in `data/raw/osm/<extract>.admin.fgb`)
  whose overlap with it is at least half their union, both taken on land less inland water, or the
  several relations of level 8 or coarser that lie mostly inside it when together they overlap it
  better (a NUTS-3 unit is often two or three districts). Whatever of a unit no relation covers keeps
  the unit's own polygon when that is at least a tenth of it, and the build prints each such unit
  (T-610). Units are clipped to the OSM coastline, holes filled, the outline cut along every
  existing pack, and pieces under 0.05 km2 or with nobody in them (GHSL 100 m) dropped. Units with
  their OSM names go to `data/work/<id>.units.geojson`, the outline to `.outline.geojson`;
  `osm_units.py <id>` writes them and prints each unit's match.
- **From the row**: origin, region name, `utc_offset`, `iso2`, the extracts (also the water
  source's).
- **Water cell** 25 m, or 50 m where the mask would pass 150M pixels (as `japan_zones.py`).
- **Jobs blend**: the population share is ILO's modelled vulnerable employment share (World Bank
  `SL.EMP.VULN.ZS`, cached per country), held to 0.2-0.5; the other two shares move with it from
  tier 0's `JOB_BLEND` (population 0.2) to Mumbai's and Lagos's (0.2, 0.3, 0.5).
- **Curve** borrowed (T-314); **work from home** 0; airports from Wikidata, O&D share 1; hospitals
  and the land-use cap tier 0's defaults; **Latin place names** when most of the units' OSM names
  are not in Latin script.
- **Country keys** (T-555): when the row names a `module`, that module's `plan_hooks(row)` returns
  the keys it supplies for any city in the country, put over the defaults above (a country outline
  also drops the Latin names rule, which reads osm_units' names). A key a module has for one city
  only (a typed-in table, a survey, a fitted curve, an airport's judgement) stays out. A module
  without `plan_hooks` stops the config: add one, `{}` if nothing carries. The keys join on the
  config's first lookup in `CITIES` (`build_city.Cities`), since the modules import build_city.
  After the build, `nocar.py` and `everyday_rates.py` take a planned pack's country from its row
  (England's LSOA car shares for an English row).

  | Module | Keys | For |
  |---|---|---|
  | england | outline (whole ITL3 units, ONS BGC), district totals, population by LSOA, BRES jobs and sectors, GIAS schools, work from home 0.27 | English rows |
  | france | outline (départements from their communes), population by commune, INSEE jobs by commune of work, work from home 0.12 | NUTS-3 units to départements by name |
  | germany | outline (Kreise, VG250), Zensus 100 m population scaled to Gemeinden, Pendlerrechnung jobs, work from home 0.10 | NUTS-3 units to Kreise by VG250's `NUTS` |
  | brazil | population by 2022 census tract | the states of the plan's municipalities |
  | canada | everyday destinations by census tract of work | any province |
  | china, india, korea, nigeria | none | |

  A country outline is the plan's own units in the census's polygons, so it needs no cut along
  packs: the plan gives every unit to one metro.

To build an existing pack's metro as a test, `world_plan.py row <pack> <out.json>` plans it as if
the pack did not exist, a one-row plan file whose row names the pack in `test_of` (the outline is
not cut along it); build it into a scratch folder by patching `build_city.WORK` and `PACKS`.

**A hand-made city**: in `pipeline/build_city.py`, add to `CITIES`:

```python
"wor": {
    "origin": (-71.8023, 42.2626),  # a central point; becomes the pack's frame origin
    "lodes_years": (2023, 2022),
    "water": {"cell_m": 25.0, "min_width_px": 2},
    "boundary_notes": "notes/T-###.md",
    "counties": {"25027": "Worcester MA", "09015": "Windham CT"},
},
```

If a county is in a state not yet in `STATES` (FIPS to LODES abbreviation), add it there, and in
`group_quarters.PL_STATES`. A pack with its own fitted gravity curve sets `"decay_pow"` and
`"decay_km"`; its own border factor `"border_factor"`. Outside the US also `"country"` (section
"Outside the US").

## 4. Build each pack

From the maps root, one city at a time, in the background, with `CARGO_BUILD_JOBS=3` and
`OMP_NUM_THREADS=3` (CPU cap):

    venv\Scripts\python.exe -u trainworld\pipeline\build_city.py wor > artifacts\trainworld\T-###-wor-build.log 2>&1

`build_city.py` puts `~/.cargo/bin` on PATH itself; `pack_gravity` runs from `data/work/target`.
**The sim crate must compile**: another agent mid-edit in `sim/` stops the build at the gravity step;
then point `TW_SIM_DIR` at a copy of the committed crate (`git archive HEAD trainworld/sim`, with
junctions to `data` and `app` beside it for `pack_gate.py`'s rates test and the joint solve:
`mklink /J <copy>\data trainworld\data` and the same for `app`), which `build_city.py`,
`routes.py`, `market_zones.py`, `zones.py` and `pack_gate.py` honour. Canada's adapter has
`canada.py <id> --finish` to resume from the cached cells.

It writes `data/packs/<id>.json`, `.bin`, `.boundary.geojson` (a US pack's: TIGER/Line 2020
counties with their water out to the state and county lines at sea, simplified topologically by
`pipeline/tiger_counties.py`, T-352; `build_city.py --boundary-only [ids]` rewrites only these files
and each pack header's boundary source line) and intermediates in `data/work/<id>.*`, and runs every
non-work purpose (schools, universities, everyday sectors, airports, medical visits, parks), so
`enrich_all.py` is not needed for a new pack. Tier 0 and Canada read every OSM point and area input
(job places, schools and universities, terminals, medical, hospital buildings, parks, tier 0's place
list) from the region's Geofabrik extract, read once by `pipeline/osm_features.py` into
`data/raw/osm/<extract>.points.fgb` and `.areas.fgb` (T-363). Each script still keeps its own small
per-city file (`osm_jobs_<id>.csv.gz`, `osm_medical_<id>.json` and the like; tier 0's
`data/work/<id>.tier0_signals.npz` holds the job places too); delete those to reread a city from the
cache. The US path queries Overpass for parks, medical and airport terminals, which can be slow or
down.

Then the place list for search and station names (T-177): `venv\Scripts\python.exe
trainworld\pipeline\places.py wor` → `data/packs/wor.places.json` (it prints the count by kind).
A tier 0 city takes it from the extract's cached features; any other city makes one Overpass
request, cached in `data/raw/osm_places_wor.json` (used whenever it exists).

Then routed walking and driving (T-275): `venv\Scripts\python.exe trainworld\pipeline\routes.py
wor` adds the `walk` and `drive` arrays from the Geofabrik extract of each region (`EXTRACTS` in
`routes.py`; `--extracts` names others; a US pack also takes every state its water grid touches).
Run it last, after anything that changes the pack's zones (a gravity re-solve leaves the routes
stale, and the game then ignores them); `routes.py --check` lists packs with missing or stale routes.
It needs the sim crate to compile (`pack_walk`), or `TW_SIM_DIR`. Its log gives the near zone pairs'
median drive factor (1.3-1.5 in the US packs) and the snapping distance (a median cell over about
300 m from a road means the region's OSM roads are thin).

Last, merge the sparse cells (T-044, docs/demand-sources.md 4.1): `venv\Scripts\python.exe
trainworld\pipeline\merge_cells.py wor`, seconds. It keeps the res-9 pack in
`data/work/res9/wor.json/.bin` and writes the merged one to `data/packs`: sparse res-8 parents
become one cell, in busier parents the sparse children that touch merge among themselves (T-466), and
a sparse cell left alone joins a touching merged cell of its zone across its parent's edge (T-510).
Every other step reads and writes the res-9 pack: to rerun one on a merged pack,
`merge_cells.py wor --undo` first, then the step, then `merge_cells.py wor` again (it refuses a
merged pack some step has changed since). `merge_cells.py --all` remerges every registered pack.
Then `nocar.py wor` (homes without a car, T-511; every country has a figure by T-562's rules,
docs/data-register.md "Car ownership", and `nocar.py --plan` lists each planned country's) and
`everyday_rates.py wor` (the country's everyday rates, T-509; a worldwide median without a survey),
seconds each; both write the merged pack and its res-9 copy, so rerun them after any step that
rebuilds the pack. The market's zone file (`market_zones.py`, section 7) is made after the merge.

A state new to the game also fetches its 2020 redistricting summary file (`<st>2020.pl.zip`, T-234);
`pipeline/group_quarters.py` keeps a small block table from it (`data/raw/pl2020_p5_<fips>.csv.gz`).
A US build log should show the population check exact or within 0.1%, the jobs check within about 5%
(do not calibrate), NCES schools matched at 95% or more (the build fails under it), and a hospital or
university lump among the top cells by jobs as normal LODES (T-043); the rest is in
`notes/adding-a-city-history.md`.

## 5. Check each pack

    venv\Scripts\python.exe trainworld\pipeline\read_pack.py wor > artifacts\trainworld\T-###-wor-check.log
    venv\Scripts\python.exe trainworld\pipeline\look_water.py wor artifacts\trainworld\T-###-wor-water.png
    venv\Scripts\python.exe trainworld\pipeline\look_pack.py wor artifacts\trainworld\T-###-wor-pack.png

`read_pack.py` must print OK on the cells, commute, gravity (worst row and column sums around
1e-4 or better, zones split by state where it applies, the border factor) and water lines. Water
and land test points need no hand list (T-424): the gate (section 8) tests OSM water polygons'
polylabels and OSM stations against the mask; `WATER_TESTS` in `read_pack.py` keeps the older
cities' hand points. Look at the water image: rivers and lakes where they belong, nothing odd at
the edges. Record cells, population, jobs, commutes, zones and `.bin` raw and gzipped size.
(`look_pack.py`'s titles say "2020 Census" and "LODES WAC" for every pack.)

## 6. Registry

`app/src/game/registry.json` is the only place the app learns about a city (T-125). **Add the line
only once the pack, its boundary and its water mask exist**: the demand pool's reach test fetches
every registered boundary, and the Node checks open every registered pack. Then run
`python pipeline/zone_boxes.py`, which gives every entry its `box` (boundary and water mask extent);
the clock worker fetches a zone's mask only where the view or track comes near its box (T-302).
For a batch outside the Northeast, `python pipeline/register.py <ids> --market --trips` does this
section and section 7's market and local-trips steps in one call, with no need to read the
registry (T-441).

- Append to `cities`: `id`, `name`, `origin` (**exactly** the `build_city.py` origin, since it is
  the pack's frame), `center` (usually the origin), `zoom` 11.2, `utcOffset`, `market`.
- `utcOffset` (T-309): the city's standard time offset from UTC in whole hours, no daylight
  saving; round a half hour down (India's +5:30 is +5). Its periods, weekends and fares run on it.
- `market` is the Northeast's joint commute gravity's place for the zone (`intercity.py` and
  `joint_gravity` take the cities with one), **-1 for every other zone**. Every zone's place in the
  world market, `place`, and its `country` are written by `pipeline/market_zones.py` (step 7),
  which appends the zone to `market.places` and writes its market row into
  `app/src/game/market-rows.json` (T-776: not in the registry, which the app bundles). **Place ids are positions in that list:
  append only, never reorder or renumber.**

## 7. Regional market and zones

Every registered zone is in the world market (docs/regional.md 5.1, T-377). Its pack header
should name its country (`"country"`, ISO 3166 alpha-2; absent means US), which the joint commute
gravity reads for border factors (`pipeline/borders.json`: add the country's admin-1 default
only if its census shows a pack line other than the default 0.4 (`pack_line_default`, T-584), and its
national pairs); the market takes the country from the world field. Then, in order:

    venv\Scripts\python.exe trainworld\pipeline\zones.py             (non-city zones: registers built ones)
    venv\Scripts\python.exe trainworld\pipeline\market_zones.py <id> (its market pack and every row, seconds)
    venv\Scripts\python.exe trainworld\pipeline\zones.py --trips     (each zone's local trips, for the overview)
    venv\Scripts\python.exe trainworld\pipeline\routes.py <id>       (last: T-275's routes, for new packs)

`market_zones.py` also gives the zone's pack its airports and direct flights (T-378, from
OurAirports and the flights project's FR24 file, `maps/flights/routes_fr24.json`; a direct flight
counts from either end, so other zones need no rerun; `--air` redoes only every pack's airports).

A zone in the Northeast's joint commute gravity (`market` set) runs `intercity.py` instead of
`market_zones.py`: it rewrites every Northeast pack's gravity factors (the joint solve), so
existing packs change here too: record the commuters between packs it prints and each corridor's
trips before and after. Any other pack whose outline comes within 50 km of another registered
pack's is in a joint gravity group with it (T-539; `intercity.py --groups` lists them, `--plan`
with the world plan's metros too). Once it is registered: `merge_cells.py <every member> --undo`,
`intercity.py --group auto <id> --no-market` (it rewrites every member's factors; Japan's 45 packs
about 5 minutes with `OMP_NUM_THREADS=8`), `merge_cells.py` on them again, then `register.py
<every member> --market --trips`. A pack with no neighbour that close needs none of this.

A non-city zone is a `CITIES` entry in `build_city.py` with `"kind": "zone"` and a `"name"`, one
county-sized unit (docs/zones.md 2).

## 8. Tests that list cities

- `sim/src/demand/everyday.rs`, test `every_pack_everyday`, and `checks/t066-worker-check.mjs`
  read the registry: nothing to add.
- The browser check needs no hand row: the gate (below) makes one from OSM stations and the mask.
  It requires the test line to carry work and social, and each of shopping, school and medical
  that the pack has sites for within 1.5 km of a station; a purpose with no sites there is
  listed in the gate's `browser` detail, not failed.

**The gate** (T-538), last, once the pack is registered and its routes are in, with your own Vite
running:

    set BASE=http://localhost:<port>
    venv\Scripts\python.exe trainworld\checks\pack_gate.py wor --browser [--log wor=<build log>] [--since <snapshot>]

It prints PASS, WARN, FAIL or SKIP a check and `wor SHIP` or `wor PARK` with the failures; a parked
pack is not shipped until its fault is fixed. `pack_gate.py snapshot <file>` before the build makes
`--since` check that no other pack changed beyond its own gravity group's factors. The thresholds
and what each check catches: notes/T-538.md. Where Anita accepts a failure for one row, its plan row
carries `"gate_relax": {"<check>": "<why>"}` and that check's FAIL counts as a WARN (dus, T-725).

The gate uses no network. Its OSM stations (railway=station|halt nodes, for `stations_on_land` and
the test line) come from `data/work/<id>.stations.json`, which `pack_gate.py stations <id> --extracts
<a,b>` writes from the extracts' `.stations.fgb` caches (`osm_features`, made in the points pass);
the runner does this as a build step, since the caches go after the region and a joint re-gate
comes later (T-770). Without that file the gate makes it from the caches if they are on disk, else
reads the Overpass answer cached for packs gated before T-770 (`data/raw/stations/`), else the
check SKIPs.

## 9. Checks

Native and Node (`npm run wasm --prefix trainworld/app` only if Rust changed):

    cargo test --release --features demand --manifest-path trainworld/sim/Cargo.toml every_pack_everyday -- --ignored --nocapture
    npm run check --prefix trainworld/app
    node trainworld/checks/t066-worker-check.mjs
    node trainworld/checks/t149-worker-check.mjs

`every_pack_everyday` prints each city's per-resident outings: school about 0.11-0.15, university
0.02-0.08, errands and social at New York's rates by construction; the gate fails a pack outside
school 0.06-0.35 or university 0.001-0.08. A pack with no university sites (OSM maps none in the
outline) opens with that purpose empty and only WARNs.

Browser (own Vite with HMR and watching off, `checks/dev-nowatch.mjs`; headless Chrome through
Playwright with its own profile directory; never touch app/dist): the gate's `--browser` (section 8)
runs `checks/city-browser.mjs` on a row it makes, about half a minute a pack; run it in the
background with one wait, since a 10-minute foreground run let the cache expire (T-289). A new game,
then for each new city:

- jump to its centre: it becomes the active city, its pack opens (`twDemand.client.start.loads`),
  its outline is the focus outline (`tw-city-focus`); Homes bubbles show; screenshot to
  `artifacts/trainworld/T-###-<id>.png`;
- ground-level track across water is accepted as a bridge (or refused with `BridgeNoRoom` where
  the crossing is too close to the segment's ends) and the same route in a tunnel accepted;
- a test line through the centre, constructed: at round 3 it carries work, shopping, social,
  school and medical riders, and purposes add up to the line;
- in the market: a bubble selection's longer trips match and reach other places;
- no page errors.

Startup (optional for one city): a production build into a scratch folder (`node
trainworld/app/node_modules/vite/bin/vite.js build trainworld/app --outDir <scratch>/dist
--emptyOutDir`), served by `node trainworld/checks/static-serve.mjs <scratch>/dist 6101`, timed by
`checks/startup-timing.mjs` before and after the registry change. A new game opens only the viewed
city, so a new city should add only its boundary and water-mask fetches.

## 10. Docs

- `SPEC.md` section 3 (the playable cities), section 5 if the market changed.
- `CLAUDE.md` layout: the registered cities and the market id list.
- `docs/demand-worldwide.md`: every rule the new country's data changed or could not follow.
- `notes/T-###.md`: footprint and why, sources, the pack table, checks, what was deleted.
- `DECISIONS.md` for anything decided; `ARCHIVE.md` line, task off TASKS.md.

## 11. Clean up

Once the packs are verified: delete big raw downloads no longer needed (Anita's preference; note
what in the task notes). Per-county TIGER block zips are needed again only by `regions.py` or a
rebuild. A state's `<st>2020.pl.zip` can go once `pl2020_p5_<fips>.csv.gz` exists. Geofabrik
extracts go once `routes.py` and any OSM reads have cached their results, and the extract's OSM
caches (`<extract>.points.fgb`, `.areas.fgb`, `.water.fgb`, `.ways2.npz`) go too once the city is
verified (Anita, 2026-10-15: disk is the limit at hundreds of cities; a full rebuild downloads the
extract again). Keep the small derived tables the build reads. Delete `data/work/pre-T###/` when
the batch is accepted.

## 12. Retrospective

Hundreds of cities will go through this list, so each one ends with a short review in its task
notes (`notes/T-###.md`, "Retrospective", under 30 lines): the parts of the work in order, with
wall time and tokens for each (`python trainworld/checks/token_usage.py --timeline <your
transcript>`; the orchestrator gives the path), what was not worth its cost, and proposed changes
to this list, the tools or the briefs. The orchestrator judges each proposal, adopts or declines
it with a reason, and passes edge cases to Anita.

## Outside the US

A country needs adapters for every input before its first city. They live in one module per
country (`pipeline/canada.py`), chosen by `"country"` in the city's config: `build_city.py`'s
`main`, `build_boundary` and `build_water` hand such a city to it, and `enrich_all.py` and
`enrich_airports.py` have small branches. The pack format, the gravity solver, spreading
(`spread.py`, given the country's polygons) and the water rasterizer (`water.build_mask`, given a
`source`) are shared, so the runtime needs nothing new.

What Canada's adapter uses (sources and URLs in notes/T-277.md), and tier 0, the adapter from
global sources only (`pipeline/tier0.py`, T-287; sources and its measured distance from Toronto's
real pack in notes/T-287.md):

| Input | US | Canada | Tier 0 (any country) |
|---|---|---|---|
| Population and grain | 2020 blocks, TIGER polygons | 2021 dissemination areas (StatCan 98-10-0015-01; polygons from Esri Canada's copy) | GHSL GHS-POP E2020 100 m (R2023A, CC BY), each pixel over four 50 m points off water, into cells; scaled to district totals where the config gives them |
| Institutions | PL 94-171 P5 off homes | none (no small-area figure) | none |
| Jobs at work | LODES WAC by block | census usual place of work by census tract of work (98-10-0498-01), at-home workers put back by income band, no fixed workplace added; to DAs by OSM job places and population | population x jobs a resident (World Bank: ILO employment-to-population 15+ x share 15+), spread 30% by GHSL non-residential built-up volume (GHS-BUILT-V NRES) x height^2, 50% OSM job places, 20% population; or a country's workplace counts by small area spread by that blend (`jobs` hook) |
| Home end | LODES RAC x ACS work from home | commuters by tract of residence (98-10-0480-01) plus at-home workers commuting again | population x commuters a resident |
| Work from home | ACS 2019-2023 by tract and industry | Labour Force Survey May 2024 (24.7%, Toronto CMA), at both ends | one national share from the config (default 0) |
| Everyday destinations | LODES retail, service, food, leisure sectors | the same sectors by census tract of work (98-10-0492-01) | each 2 km zone's jobs (errands also 25% residents) blended with its OSM shops and services, or restaurants, bars, venues and hotels, as far as the pack's OSM is complete; or a country's jobs by sector by small area that pass the plausibility check (`everyday` hook: Japan's mesh, England's BRES); set after the gravity (`tier0.everyday_pack`, T-453) |
| Gravity curve | fitted on LODES OD (T-198) | fitted on census flows between municipalities (98-10-0459-01; `notes/t277_fit.py`) | power 1.4 with the length fitted to a published median or mean commute (`tier0.fit_curve`), or with none, `"curve": "borrowed"` (`tier0.borrowed_curve`, T-314); a country's flow table, fitted (London); both parameters fitted to commuters by distance band (`tier0.fit_curve_bands`, Mumbai; a travel survey's home-to-job distances, Sao Paulo) |
| Regions, border | states, 0.23 | provinces, factor 1 | one region, factor 1 |
| Drawn boundary | TIGER/Line counties, water included (T-352) | cartographic census subdivisions, clipped at the shore (hugs Lake Ontario) | the config's district polygons (a GeoJSON, or a callable that writes one: London's whole ITL3 units, `england.itl3_boundary`) |
| Water | TIGER AREAWATER and states | StatCan lakes and rivers; land = cartographic municipalities plus US states | OSM: `osm_water.source([extracts])` as the config's `water_source` (below) |
| Schools | NCES | Ontario school information file (provincial; other provinces need their own) | OSM `amenity=school` sites; pupils = population x share aged 0-14 (World Bank) x 0.9, by the square root of site area; or a national register with pupils (`schools` hook, England's GIAS) |
| Universities | IPEDS | Council of Ontario Universities headcounts and Ontario's college campus file, at points in `pipeline/ca_campuses.csv` | OSM `amenity=university|college`; Wikidata P2196 students by institution, split by site area; other sites by area at the median density; the total scaled to population x the national tertiary students a resident (World Bank SE.TER.ENRL, T-288) |
| Airports | FAA enplanements, DB1B connecting share | StatCan 23-10-0253-01 enplaned; connecting share by judgement (Pearson 0.75) | OurAirports + Wikidata patronage / 2; connecting share by judgement; pack share from GHSL population |
| Medical, parks, places | OSM (Overpass) | the same scripts, OSM from the extract | the same, from the extract |

**Land use in the OSM signal and the `osm_signal` line (T-460).** Every tier 0 build counts OSM
industrial, commercial and retail land as job places (one per 5,000 m²), each area's weight capped
at 1 km² (`land_cap_km2`) and spread over a 0.003 degree grid inside it (`land_spread_deg`); a
config sets either to 0 to turn it off, or to another value. The rows cached in
`data/raw/tier0/osm_jobs_<id>.csv.gz` keep the settings they were made with: delete the file to
remake them. Where few shops are mapped, land use dominates the signal (Lagos 71%), and uncapped
one refinery took 6.5% of all jobs. Every build prints one line for the gate (`checks/pack_gate.py`):

    osm_signal: errand=1101 leisure=527 job=2464 land=10232 land_share=0.7143 top_place=w732895785 top_place_jobs_share=0.00419 top_cell=6.4348,3.3586 top_cell_jobs_share=0.00288 top_place_name=-

Read it by splitting on spaces, then each field on its first `=`: values hold no spaces. `errand`,
`leisure`, `job`, `land`: the summed OSM weights in the pack's cells (land in 5,000 m² units);
`land_share`: land over all four. `top_place`: the largest place by OSM weight, `w<id>` an OSM way,
`r<id>` a relation (an area's spread rows count together), or `<lat>,<lon>` for a single row when the
cached rows have no OSM id (caches made before T-460); `top_place_jobs_share`: the jobs it brings
through the OSM part of the blend over all jobs; `top_cell`: `<lat>,<lon>` of the cell with the most
jobs and `top_cell_jobs_share` its share of all jobs; `top_place_name`: percent-encoded, `-` when the
place has none. Printed once, after the jobs are set, before the cells are written. In a pack whose
jobs are the proxy alone, a place's weight is capped before the cells are summed (T-583, so
`top_place_jobs_share` is at most 0.0075) and the line's shares are of the weight before capping.

**Water outside the US (T-041).** `pipeline/osm_water.py` gives any city a water mask from
OpenStreetMap: the sea from osmdata.openstreetmap.de's coastline polygons
(`data/raw/osmdata/water-polygons-split-4326.zip`, downloaded once for the world and kept) and
inland water areas (`natural=water`, `landuse=reservoir`, `waterway=riverbank|dock|canal`; not
intermittent, covered or dry basins) from the region's Geofabrik extract, read once per extract
into `data/raw/osm/<extract>.water.fgb`. In a tier 0 config: `"water_source":
osm_water.source(["europe/united-kingdom/england"])`; the extracts must cover the whole box, and
must not overlap: check Geofabrik's region page first, since a region can contain another
(Brandenburg includes Berlin, T-290), and two overlapping extracts count every feature twice.
Neighbouring regions' polygons also overlap in a strip along their border (Japan's Kansai and
Chubu, T-431): a config's `extract` may be a list, and `osm_features.read` then keeps an object found
in two once; for water the overlap does no harm. To read an extract ahead, or to see a city's mask
before its pack exists:

    venv\Scripts\python.exe trainworld\pipeline\osm_water.py read europe/france/ile-de-france
    venv\Scripts\python.exe trainworld\pipeline\osm_water.py mask par --extracts europe/france/ile-de-france --bbox w,s,e,n --origin lon,lat --png artifacts\trainworld\T-###-par-water.png

`mask` writes a water-only pack, `data/work/osm_water/<id>.json/.bin`, draws it with `look_water.py`
and runs `read_pack.py`'s water test places for that id (add them to `WATER_TESTS`, close-ups to
`look_water.VIEWS`). The Geofabrik `.osm.pbf` can go once its `.water.fgb`, `.points.fgb`,
`.areas.fgb` and `.ways2.npz` exist.

A tier 0 city is a `CITIES` entry with `"tier0": True` (the keys are listed in `tier0.py`'s
docstring); `build_city.py <id>` hands it to `tier0.main`. A country's census then replaces one
input at a time (the upgrades table in research/world-sampler.md), in a module per country
(`pipeline/england.py`), through tier0's `population`, `jobs` and `schools` hooks. London (`lon`,
T-288 and T-360, notes/T-288.md) is the worked example: the outline as whole ITL3 units (ONS),
Census 2021 population by LSOA, BRES jobs by LSOA, GIAS schools, the curve fitted on the 2011
WU01EW flows (`notes/t360_curve.py`), work from home one national share. The chain: read the
extract's features and water, `build_city.py lon` (reads every OSM input from the feature cache),
`routes.py lon --extracts europe/united-kingdom/england`, `places.py`.

What London taught, for the next tier 0 city:

- **Look at the outline before anything else**: whole units can leave a hole (Medway between
  Gravesham and Swale cut Sheppey off as an island until it was added).
- **Some GHSL tiles are cut short** at the ocean edge (GHS-BUILT-V NRES R3_C18 starts 1,000 pixels
  east of its nominal corner); `tier0._read_tile_window` places each tile by its GeoTIFF tie point.
- **District totals matter in dense old centres**: GHSL had 12-19% too many people in central
  London boroughs.
- **The jobs proxy undercounts an old mid-rise core** (Westminster and the City at a third of
  BRES) and no blend fixes it without breaking Toronto; use workplace counts wherever a country
  has them (demand-worldwide.md).
- **Measure each upgrade against tier 0 before switching** (`notes/t360_*.py`); a 2 km zone
  difference is what the game feels.

**Worked examples by country**, one per archetype, for the chain of commands (what each took and its
timings: `notes/adding-a-city-history.md`; the notes of each task have the sources):

- **Paris** (`par`, T-289, `pipeline/france.py`): one INSEE file (the 2022 flows, every flow) gives
  both jobs by commune of work and the curve fit (`notes/t289_curve.py`, which also makes the signals
  cache, so the build that follows reuses it). Communes and legal populations from geo.api.gouv.fr,
  one request a département. Take water test places from the water cache's own polygons, never
  from memory.
- **Shanghai** (`sha`, T-291, `pipeline/china.py`): districts are the extract's OSM admin_level 6
  relations clipped to land; relations GDAL cannot assemble (Jing'an, Baoshan) are rebuilt from
  admin_level 8 units or as the province's remainder. Population and jobs scaled to district tables
  from the statistics bureau's census bulletins (the economic census's bulletin no. 1 has employment
  by district): look for both before the first build. Places show `name:en` or pinyin
  (`"latin_names"`).
- **Mumbai** (`mum`, T-292, `pipeline/india.py`): units are OSM district and taluka relations by
  name; the census tables (Primary Census Abstract per district, A-2, B-28) are on
  censusindia.gov.in/nada (certificate checks off for that host only). Compare GHSL with each
  unit's census total in the first build's log; where OSM's lines between units do not follow the
  census's, scale those units as one group. A table of commuters by distance band fits both curve
  parameters (`"curve": {"bands": ..., "road_factor": ...}`, `tier0.fit_curve_bands`).
- **Sao Paulo** (`sao`, T-458, `pipeline/brazil.py`): IBGE's 2022 tracts with their people are one
  GeoPackage a state; a household travel survey's microdata (the Metro's OD 2017) gives jobs by
  zone, work from home and the curve in one file: look for a city's survey first. Merge the curve's
  bands under 4 km (2 km zones cannot split them). The World Bank API can time out: fetch its three
  JSON files into `data/raw/tier0/` with curl.
- **Lagos** (`lag`, T-459, `pipeline/nigeria.py`), from global sources alone: `nigeria.py probe <id>`
  lists the admin relations in the box and `nigeria.py <id>` prints GHSL by unit before any build.
  Where OSM maps few shops and offices, read the build's `osm_signal` line (above): land use
  dominates the signal. Run Python with `PYTHONIOENCODING=utf-8`, or OSM names with tone marks stop
  a print.
- **Berlin** (`ber`, T-290, `pipeline/germany.py`): Brandenburg's extract includes Berlin, so never
  add Geofabrik's Berlin extract beside it. BKG's VG250-EW gives Kreise, Gemeinden and their
  population in one file; the Pendleratlas's CSVs (`/data/csv/<year>/<year>_<table>_L<Land>.csv`,
  found in its JS bundle) give jobs by Gemeinde of work and the top flows for the curve. Water test
  points must sit on a channel's middle (polylabel on the cached polygon); a canal under 50 m wide
  is below the mask's 2-pixel width.
- **Tokyo and Japan** (`tyo`, T-420, `pipeline/japan.py`): a Japanese pack is a config row:
  `prefectures`, `neighbours` (the land beyond the outline, for meshes it cuts) and the region's
  telework share. `python pipeline/japan.py fetch <id>` downloads the census meshes, N03, table 4 and
  P29 first, and `japan.py outline <id>` draws the outline. Run `notes/t420_curve.py <id>` for the
  curve before the build (it makes the signals cache). `routes.py tyo --extracts asia/japan/kanto`
  (Japan is not in `routes.EXTRACTS`). Place names need fugashi and unidic-lite in the venv
  (`japan.kanji_romaji`). A rebuilt pack's caches (`data/work/<id>.*`, the `osm_*_<id>` files under
  `data/raw/`, `<id>_prefectures.geojson`) must go first, or the old extent's are reused. A Hokkaido
  pack names subprefectures (`japan.SUBPREFECTURES`) or one municipality (`"01-01203"`); each
  regional metro fits its own curve before the build. Japanese downloads go through curl
  (`japan._get`, T-441): KSJ's server, and Geofabrik's on a bad night, stall a Python request for
  good. The rest of Japan (T-420 stage 3) is 38 non-city zones built as one batch by
  `pipeline/japan_zones.py`: `plan` (zones, neighbours, extracts, origin, wfh into
  `japan_zones.json`), `build <id>` (outline, a curve fitted over a fixed grid, the pack, places,
  routes, merge, `read_pack`, water picture) and `run <ids> --jobs 4` (a process a zone, a log and
  status line each). Build the few zones least like each other first (an island zone, a big one, a
  merged one): a batch finds a tier 0 bug 9 times over. Check table 4's municipality codes against
  N03's before building (wards redrawn since 2020 starve a municipality of jobs).
- **Seoul** (`seo`, T-457, `pipeline/korea.py`): population is the resident registration count by
  si-gun-gu from jumin.mois.go.kr (`korea.py fetch`, one POST a province, no key); districts are OSM
  relations by name inside their province, the admin areas cached in
  `data/raw/tier0/korea/<id>_admin.fgb`. Check the district names against OSM first: units renamed or
  split since the table's year need a mapping (`REGIONS[...]["osm_names"]`).

What was hard, for the next country (Toronto): a statistics office's geography server may sit
behind a bot check (StatCan's www12), so look for mirrors early (ArcGIS Online copies) and list
the official URLs for Anita; a census taken in a lockdown needs its at-home workers put back;
education data may be provincial, not national, so a second province needs its own files; when
Overpass is down, reading the Geofabrik extract with GDAL is the fallback for every OSM input; and a
fitted curve can differ much from the US one (Toronto's d^-1.2 exp(-d/20 km)), which is why each
pack ships its own.
