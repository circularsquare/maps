# The world run: runner, plan and planned rows

What `pipeline/world_run.py` does for a world plan row with no agent per city: its flags, locks, gate,
join, faults and scratch tree ("Running the pipeline"); the rule that makes the plan ("The world plan");
and how a plan row becomes a build config, with each country module's keys ("A planned city"). Moved
from `docs/adding-a-city.md` when that was split (T-1987). The procedure for a country is
`docs/adding-a-country.md`; a hand-made pack or a country module is `docs/adding-a-city-by-hand.md`.

Logs, check outputs and images go in `../artifacts/trainworld/` (from the maps root
`artifacts/trainworld/`), never in `notes/`.

## Running the pipeline

A world plan row ("The world plan", below) needs no agent: `pipeline/world_run.py` (T-540) does sections 3
to 7 and 11 of `docs/adding-a-city-by-hand.md` for it, with the gate (its section 8) deciding ship or park. From the maps root:

    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py plan --limit 20    what a run would do
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --limit 20     the next 20 rows
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py status             rows by status, locks held
    venv\Scripts\python.exe -I trainworld\pipeline\world_run.py join               the joint solve for gated rows

`--rows a,b` or `--countries GB,PT` instead of `--limit`; `--jobs N` the cores in all (default 2,
up to 12 on an overnight run Anita approves). A run builds `--workers` rows at once (default N),
each step of a build on one thread, since a build step keeps about 0.9 of a core busy whatever
threads it is given (T-966). The workers are the run's slots: a gate chunk takes 2 of them and an
extract's read 1 while builds go on, builds waiting meanwhile (T-1978, T-1980); the join's steps,
which run with no build, take all N. `--port` its own Vite
for the browser check (default 6360); a run stops at its start, and before each gate, when
something already answers on that port (T-1079, T-1986). Set `TW_SIM_DIR` and `TW_CARGO_TARGET_DIR` when the tree's
`sim/` may be mid-edit (`docs/adding-a-city-by-hand.md` 4); the run prints, and reports as a problem, every file where that
copy is not HEAD's `sim/`, saying which match the tree (a change not yet committed). Run it in the
background and wait once for the process.

### Several countries and zone rows

- **Several countries at once** (T-786): a run's builds use about 0.9 cores each (so about 0.9 x
  `--jobs` while it builds, and 1 to 1.5 cores during its region's gate) and its joint solve all
  of them, so start one run per country or region with `--no-join` and its own `--port`, `--jobs` summed
  within the cores allowed; each ends with its gated rows registered, status `gated`. When all
  have ended, one `join` (every gated row, or `--countries`) solves their groups on every core of
  its `--jobs`. A run claims its rows (a row another live run claimed is left out) and holds its
  region's extracts (a run needing the same extract waits). Every read-modify-write of the plan,
  the registry, the market rows and `made.json` takes a lock (`pipeline/shared_lock.py`, files in
  `data/work/world_run/locks/`, released when the process dies), and so do register.py,
  market_zones.py, zones.py, zone_boxes.py and world_plan.py. Gates of several runs overlap, but a
  joint solve overlaps no gate (a gate's snapshot would see its changes): the solve holds the
  `packs` lock and waits for gates under way (each holds `gate-<batch>`), and a gate does not
  start while `packs` is held. A run waiting for a lock prints so after 5 seconds, and `status`
  lists who holds what. A gate check comparing other packs against its snapshot skips packs no
  longer registered (another run's park or rebuild).
- **Three parks in a row on one reason stop the run** (T-775): the reason with the row's id taken
  out (and, for a failed step, its numbers), so a code fault does not park a whole country; rows
  failing a gate check on their own values do not count as one reason.

- **Zone rows** (whole-country cover, T-723, docs/zones.md 2): `world_plan.py zones --countries ES`
  (run through `world_run.py step world_plan zones ...`, which puts the pipeline on the path) writes
  a country's zone rows into the plan; then `run --countries ES --limit 100`. A zone row builds as
  a metro row does, with `kind` "zone" and the zone curve. The free units are grouped by the merge
  rule (T-1140, docs/zones.md 2): a zone under the floor (0.5M, or 100,000 over 100,000 km²; 1M in
  China and Taiwan) joins a neighbour that is its biggest travel partner or gets half as much
  travel as the biggest, else one keeping the joined zone compact (75% of its convex hull), partners
  first, never into a ring, and two zones each holding a GHSL urban centre of 100,000 never by
  shape alone; nothing passing, it stays under the floor. Travel comes from the
  country's commuting table, or from GHSL gravity where there is none, so a new country needs no
  input of its own. A country's own rule (`world_plan.ZONE_RULES`, T-1191) may take its units by
  their parent (`group_level`: Korea's eup, myeon and dong by si, gun and gu) or leave land far
  from people dead space (`reach`: Canada, under 100,000 people within 150 km of a unit's centre,
  with `join_km` 150 so cities across it stay apart; `plan_units` zones and builds its rows on the plan
state's units, not geoBoundaries read afresh, T-1239); give a new country one when its plan units
  are not county-sized or its land is mostly empty. A
  pack built by hand in the country (Seoul, Toronto) stays as it is: the units with half their
  people in it are its, no zone may wrap it, and the join solves it with the new rows as one of the
  country's members (the registry's `country`). Add `--out <scratch.json>` for a dry run, and read
  its merge log in the plan's `zones.<ISO>.merges` before running the rows.
- **A replan's order** (T-2302): the old rows are retired (unregistered, their files moved out of data/packs) before
  the new rows build, since a build cuts every boundary file in data/packs out of its outline (`osm_units.cut_packs`);
  a new row built beside its old zones' packs comes out a sliver and parks. Building elsewhere first (AWS, T-2301) and
  landing after the retire avoids the gap.
- **Zones removed** (T-1053): a replan, merge or re-split that takes zone ids out of the registry
  (a parked row of an earlier commit included) ends, once `zone_outlines.py` has rewritten the
  outlines, with `venv\Scripts\python.exe -I trainworld\pipeline\zone_successors.py` (seconds): each
  id in HEAD's registry and not the working one's gets its successor in
  `app/src/game/zone-successors.json`, the zone now under its old centre, so saves naming it load
  ([architecture.md](architecture.md) 11). `--set wnd=nlc` picks one by hand, `--set wnd=` none;
  `--dry` prints only. Smoke's `t1053-zone-ids` check fails until it has run. A zone taken out by
  hand leaves commutes to its place in its group's other market packs, which `market_pack` refuses
  ("invalid commute between zones"): `market_zones.py <the group's other ids>` rewrites them without
  it (T-1113). The old units' joint and everyday files stay in data/work, read by nothing; once the
  join is done, `pipeline\pair_files.py unused` lists them and `--move` puts them in
  `data/work/retired/` (T-1976).

### Inputs, rows and the gate

- **Country inputs** (T-1979): at the run's start `pipeline/netget.py <ISO2>...` caches the national
  figures every planned row's build reads (World Bank employment and age shares, tertiary
  enrolment, vulnerable employment; nocar's GDP and income groups), seconds when cached; a failure
  only goes in the report's problems. Every network request a build makes (those, GHSL tiles,
  `build_city.fetch`, ArcGIS layers, Wikidata) goes through `netget.request`: a reset connection, a
  timeout, a 429 or a 5xx waits 5, 20, 60, 120 and 240 s and asks again, so a server's bad minute
  costs seconds, not the row.
- **Rows** go in plan order, a region at a time: rows sharing a Geofabrik extract, chained. The
  region's extracts are downloaded (four tries) unless their caches or `.osm.pbf` are on disk, and
  read once into the caches (`osm_features`, inland water, roads, admin relations). Regions overlap
  one deep (T-1980): each runs in its own thread holding its extracts' locks; region k+1 downloads
  and reads once region k's builds have started (never two reads ahead: each GDAL read commits about
  6 GB, T-1080), and builds once region k's builds are done, while k's last gate chunk runs.
- **Per row**: a `planned` row (new, or its units changed) first loses the per-city caches named
  after its id, which an earlier row of that id or its old outline left (T-1743, T-891):
  `data/raw/tier0/osm_jobs_<id>`, `osm_education_<id>`, `osm_*_<id>.json.gz` (older ones `.json`) in
  `data/raw/t142b` and `t196`, `osm_places_<id>.json`, `data/work/<id>`'s outline, units and tier 0 signals, and
  `data/terrain/<id>.terrain` (`zone_terrain.py` makes a grid only where none is, T-1986). Then
  `build_city`, `pack_gate stations` (the gate's station file, `docs/adding-a-city-by-hand.md` 8), `places`,
  with `--coarse T` `coarse_zones` (4 km blocks in thin country, T-852; off by default),
  `routes --keep-pbf`, `merge_cells` (which runs `read_pack`'s checks on its output before it
  replaces the pack, and fails the step otherwise), `nocar`, `everyday_rates`, each its own process
  and log; a failed step runs once more after a minute (a timed-out one does not), then the row
  parks. A step whose log has not grown for 20 minutes (`QUIET_S`; `TW_QUIET_S` to change it; a region's gate gets at least 30 s a row, and its
  browser check prints each row's result as it lands, T-1308) dumps
  every thread's stack into its log, again after each further 20 silent minutes, and the runner
  prints a line saying so. Status `built`.
- **Gate**: the rates test the gate runs (`every_pack_everyday`) is built once at
  the run's start from the sim the gate reads (`TW_SIM_DIR`, else the tree's sim/) and copied to
  `data/work/world_run/everyday-<batch>.exe`, so a change to sim/ mid-run recompiles nothing
  (T-931; if it does not build, each gate builds it as before). The rest are registered with their market packs (`register.py
  --market`; the browser check needs them; only the rows within reach of the new packs are
  worked out again, T-929, and only the joint and everyday files with pairs from the new packs
  read, T-1976), a snapshot of every registered pack taken first, and
  gated with `--browser --since` on the runner's own Vite, started after the registration (it
  reads the registry once): every row's figures first, then the rows split in plan order over
  three browser sessions at once (T-928, T-965; `city-browser.mjs` with `ROWS`, 8 zones a page
  load, a 180 s solve wait, a zone whose wait timed out or whose page crashed tried once more in a
  fresh page). A zone over 20M people gets a session of its own (`pack_gate.BIG_ZONE_PEOPLE`): the
  removal of Jakarta's test line for the next zone crashed its page out of memory, and Playwright's
  waits on a crashed page never end (T-2040). A session whose log has not grown for 10 minutes
  (`BROWSER_STALL_S`; each wait in `city-browser.mjs` prints first and is at most 300 s) is stopped:
  its stuck zone is tried once alone (failed if it sticks alone too) and its later zones go on in a
  fresh session. **In chunks** (T-1978): while the region builds, its built rows are gated whenever
  `GATE_CHUNK` (10, `TW_GATE_CHUNK`; 0 gates once after the builds) wait, each chunk with its own
  snapshot, register and Vite (logs `region<k>.<n>.*`, the first `region<k>.*`), one chunk at a
  time; the rest once the builds end. A pack that ships stays registered; one that parks is taken out again
  with its market pack, its market place left unused (place ids are positions), and its failing
  checks are its reason. Whenever the runner takes packs out of the registry, every registered
  market pack still naming their places is rewritten, however far away (`market_zones.py <ids>`,
  T-1743, T-1986; reading all 2,000 market packs takes about 30 s), or the next `market_pack`
  refuses them. The browser check serves the main tree's app/, so whoever runs a batch
  says who may rebuild app/ (its WASM) while it runs.

### Joint gravity, freeze and clean up

- **Joint gravity**, once for the batch after its last region (T-723; a run stopped before it
  leaves the gated rows registered and `built`, for the next run to gate again; with `--no-join`,
  `join` does it): a pack with no country in its header (tor, lon, par, sao) takes the registry's
  `country`. Each solve unit a gated pack joined (a group, or one country's part of a group
  spanning countries with its neighbours held fixed, docs/zones.md 4) is solved once (members back
  to res-9, `intercity.py --solve <members> --fixed <neighbours> --name <unit>` on every core of
  `--jobs`, with four times the step limit for a unit of over 100 packs, merged again) and gated
  again without the browser
  or rates; a FAIL restores every member from the backup taken before. When only new packs FAIL,
  they park and the group is solved once more without them (a FAIL then parks every new pack of
  that solve); when an older member FAILs, every new pack parks (T-773). A unit's fixed list is
  cut to the packs registered when that unit runs, so a unit that parked earlier takes no later
  unit with it, and the market step leaves out commutes naming packs that left (T-997). Then `register.py
  <shipped and changed> --market --trips` (or `market_zones.py --rows`
  when only parks happened), and a check that every file `vite.config.ts` copies into a
  production build exists for each shipped pack (it copies the registry's packs, so nothing more
  needs listing). Last, `zone_outlines.py` rewrites `data/packs/outlines.json`, the coarse outline of
  every registered zone with the sea off its coast (T-800, T-1145; over the run's `--jobs`, with its sea work kept in `data/work/zone_outlines_cache`, T-1977: about 4.5 minutes at 4 jobs from an empty cache, under a minute when a few zones changed; a failure only prints a warning and leaves the old file, so
  the new zones have no outline until their pack loads), then `realms.py` writes each zone's realm
  and `data/packs/realms.geojson` (T-975, about 1.5 minutes; a failure leaves both as they were, and
  a zone without a realm always shows), then `zone_aliases.py` rewrites
  `app/src/game/zone-aliases.json`, the search aliases of every zone (T-836, seconds; the same
  warning-only failure, so the new zones are found by their own name only until it is rerun).
  Status `shipped`; a shipped row's build hand-offs (`data/work/<id>.cells.*`, `walk_zone.u8*`) are deleted
  then, since nothing reads them after (T-2554).
- **Freeze** (T-928): a join with no market failure and no other run live ends with
  `pipeline/freeze.py`, which hard-links every file a production build copies into
  `data/frozen/<stamp>/` (seconds, no space; the newest three kept). A publish during a later
  batch builds on it (`freeze.py build`, then `tools/deploy.py --dist`), so it never waits for one.
- **Clean up**: the region's `.osm.pbf` files and the OSM caches the runner made go once the
  region is done (`--keep-downloads` keeps them); caches already on disk before, and the small
  `.admin.fgb`, stay.

### Status, stopping and faults

- **Status** is each row's `status` in `pipeline/world_plan.json` (`planned`, `built`, `gated`,
  `shipped`, `parked`) with a `run` record: batch, seconds a step, gate warnings, a parked row's
  reason, the claiming run while one runs it (`claim`). A
  batch's report (shipped, parked with reasons, warnings, times: a region's download, read,
  builds, gate and `gate_after_builds`, the gate's wall after its builds) and every log go to
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

## The world plan

- **Which cities, and where their edges go: the world plan** (T-536, notes/T-535.md).
  `python pipeline/world_plan.py` writes `pipeline/world_plan.json`, one row per metro in build
  order with its units, origin, name, id, extracts, time zone and archetype, and the extents to
  `data/work/worldplan/extents.geojson`; `check` adds the eight-pack test and the next-50 picture.
  People are GHS-POP 2025 at 1 km, a pixel counted where its centre falls. The rule:
  - **Metros.** Outside Europe and the US, GHSL urban centres (UCDB R2024A) over 1M people; a
    centre's area is the centre, every functional urban area (GHS-FUA R2019A) it holds half the
    people of, and the FUA around its population-weighted centre; centres sharing a FUA are one
    metro. In Europe (geoBoundaries' continent: Russia in, Turkey out), FUAs over 1M instead; in
    the US too, planned alone from US counties only (`world_plan.py metros --countries US`,
    docs/zones.md 2).
    A smaller urban centre outside every metro's area whose edge lies within 3 km of a metro's
    own centres (in Europe, the centres inside its FUA) joins the nearest such metro's area, since
    GHS-FUA gives a town grown up to a metro a FUA of its own (Ikorodu beside Lagos; T-556).
  - **Order.** The UK, then the rest of Europe, then everywhere else, each largest first. A
    rerun keeps each earlier row's order and id: a row inherits them from the previous plan's row
    it shares the most people with; rows with none follow the last. A row inheriting a `held` row's id
    stays held (T-774).
  - **Units.** NUTS-3 2024 in Europe, ITL3 in the UK, Census 2020 counties in the US and Puerto
    Rico; elsewhere geoBoundaries gbOpen at the finest
    of ADM2-4 where the median person's unit holds 20,000 people or more. A first-level unit that
    level misses a tenth of the people of takes the finest level covering it (Moscow). No
    geoBoundaries at all: the territory whole (Hong Kong, Macau). Zones take units on the de facto lines: a unit
    held by another side goes to it (`DE_FACTO_TAKE`, Crimea to Russia) or is left out (`LEFT_OUT`, occupied
    Ukraine; docs/zones.md 2).
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
    children covering the outline with the least total area (US states against us-midwest; one
    smaller extract anywhere in the index, as Puerto Rico's, counts too); none inside another.
  - **Archetype**: the country's in the register (England's A only for English units), else D.


## A planned city

**A planned city** (a world plan row, T-537) has no hand-written entry: `build_city.py` adds every
`planned` row of `pipeline/world_plan.json` to `CITIES` through `pipeline/plan_config.py`
(`plan_config.py <id>` prints one), so `build_city.py <id>` builds it at tier 0 from global
sources. Each field comes from the row or a rule:

- **Outline**, `pipeline/osm_units.py`: each plan unit takes the OSM `boundary=administrative`
  relation (any admin level, from the row's extracts, cached in `data/raw/osm/<extract>.admin.fgb`)
  whose overlap with it is at least half their union, both taken on land less inland water, or the
  several relations of level 8 or coarser that lie mostly inside it when together they overlap it
  better (a NUTS-3 unit is often two or three districts). For a territory whole (Hong Kong, Macau:
  Natural Earth's 1:10m outline, which misses reclaimed land and islets) a quarter inside is enough
  (T-1144). Whatever of a unit no relation covers keeps
  the unit's own polygon when that is at least a tenth of it, and the build prints each such unit
  (T-610); a NUTS-3 unit's polygon may not ship (non-commercial terms), so its gap takes the
  geoBoundaries units lying mostly inside it instead, at the finest level with an open licence
  (T-925, data-register.md). Units are clipped to the OSM coastline, holes filled, the outline cut along every
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
  | usa | outline (TIGER/Line counties drawn, the cartographic ones as footprint), people by 2020 block less institutions, LODES WAC jobs and sectors (newest year back to 2016), ACS work from home over the row's counties, the US curve and state factor 0.23 with each cell's state as its region; per-state caches in `data/work/usa/` (`usa.py state <fips>` ahead of a run, `usa.py check <id>` for figures by county) | US rows (unit set USA, county GEOIDs); Puerto Rico rows (PRI) take the outline, people and work from home only |
  | brazil | population by 2022 census tract | the states of the plan's municipalities |
  | canada | everyday destinations by census tract of work | any province |
  | australia | population by 2021 census SA1 (ABS ArcGIS layer, no account) | every Australian row, zones too (T-1857) |
  | china, india, korea, nigeria | none | |

  A country outline is the plan's own units in the census's polygons, so it needs no cut along
  packs: the plan gives every unit to one metro.

To build an existing pack's metro as a test, `world_plan.py row <pack> <out.json>` plans it as if
the pack did not exist, a one-row plan file whose row names the pack in `test_of` (the outline is
not cut along it); build it into a scratch folder by patching `build_city.WORK` and `PACKS`.
