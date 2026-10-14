# Adding a city

Written 2026-10-12 (T-179) from how Boston and Washington (T-134), the five-city batch (T-137), the
non-work purposes (T-142), the border factor (T-162) and the market's derived block (T-174) were
done. **Followed end to end once, for Toronto (T-277, 2026-10-13)**, the first city outside the US;
that run rewrote it. Fix this file wherever the next run finds it wrong.

Rules that apply: CLAUDE.md "City additions in delegated batches" (pragmatic metro extent, basic
checks, no extent study or calibration) and "No city-specific data". A US city uses US national
sources (LODES, Census blocks, ACS, TIGER, NCES, IPEDS, FAA). A city in another country needs that
country's adapters first: section "Outside the US" at the end says what Canada's are and what the
next country will need.

Logs, check outputs and images go in `../artifacts/trainworld/` (from the maps root
`artifacts/trainworld/`), never in `notes/`.

## 1. Choose the city and its footprint

- **Where.** Anywhere between 85 S and 85 N: new games are in the Mercator world frame, whose
  track lengths are ground lengths everywhere (T-306, docs/world.md), and the renderer anchors to
  the view (T-307). Games saved in the Northeast conic (TWT5) or the legacy frame are converted to
  Mercator when they load (T-308). The market keeps its sites in an equirectangular New York frame
  but measures zone distances on the ground (T-128); a city outside it is `market: -1`.
- **Footprint.** Whole county-level units: US 2020 counties (`cb_2020_us_county_500k` GEOIDs;
  Connecticut's old counties), Canadian census divisions, the country's equivalent elsewhere. The
  metro area (OMB, BLS's list, StatCan's CMA) trimmed or widened by judgement to whole units; no
  study. Whole units are also docs/zones.md's unit for non-city zones. **No unit may belong to
  another pack**: compare with every `CITIES[*]["counties"]` in `build_city.py`. T-179's US
  candidates, all disjoint from the packs:
  - Worcester: 25027 Worcester MA, 09015 Windham CT
  - Allentown: 42025 Carbon, 42077 Lehigh, 42095 Northampton, 34041 Warren NJ
  - Albany: 36001 Albany, 36083 Rensselaer, 36091 Saratoga, 36093 Schenectady, 36095 Schoharie
  - Pittsburgh: 42003, 42005, 42007, 42019, 42051, 42073, 42125, 42129
  - Richmond: 51007, 51036, 51041, 51053, 51075, 51085, 51087, 51101, 51127, 51145, 51149,
    51183, 51570, 51670, 51730, 51760 (the 2023 metro without Caroline and King and Queen)
- **A national border** inside the extent needs a factor (SPEC 4, "Borders outside the US") and,
  in the US, a water mask that knows the other side is land (Canada's adapter adds the US states;
  the US one treats everything outside the TIGER states as sea, so Buffalo or Detroit would need
  Canadian land added).
- **Id.** Three lowercase letters, unused (`wor`, `abe`, `alb`, `pit`, `ric` were T-179's; `tor`).
- Take the task line in TASKS.md (`taking, <date>`), start `notes/T-###.md`.

## 2. Back up

`data/work/pre-T###/`: `data/packs/intercity.json` and `app/src/game/registry.json` (the only
existing files a city addition overwrites). Existing city packs are never rebuilt by a city
addition; take their sha256 now so you can show they did not change. If the batch edits shared
pipeline code, rebuild one existing pack into a scratch folder (patch `build_city.PACKS` and
`WORK`; T-277's `us_identity.py` did New Haven in 85 s) and show its arrays are bit-identical.

## 3. Pipeline config

In `pipeline/build_city.py`, add to `CITIES`:

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
`"decay_km"` (the game uses the pack's shipped curve since T-277); its own border factor
`"border_factor"`. Outside the US also `"country"` (section "Outside the US").

## 4. Build each pack

From the maps root, one city at a time, in the background, with `CARGO_BUILD_JOBS=3` and
`OMP_NUM_THREADS=3` (CPU cap):

    venv\Scripts\python.exe -u trainworld\pipeline\build_city.py wor > artifacts\trainworld\T-###-wor-build.log 2>&1

(`build_city.py` puts `~/.cargo/bin` on PATH itself; `pack_gravity` runs from
`data/work/target`. **The sim crate must compile**: another agent mid-edit in `sim/` stops the
build at the gravity step. Canada's adapter has `canada.py <id> --finish` to resume from the
cached cells; a US build reruns whole, about a minute.) It writes `data/packs/<id>.json`, `.bin`,
`.boundary.geojson` (a US pack's: TIGER/Line 2020 counties with their water out to the state and
county lines at sea, simplified topologically by `pipeline/tiger_counties.py`, so the outline and
shade run straight offshore and neighbouring packs share exact borders, T-352; the build itself
filters by the shoreline-clipped cartographic counties; `build_city.py --boundary-only [ids]`
rewrites only these files and each pack header's boundary source line) and intermediates in
`data/work/<id>.*`, and runs every non-work purpose
(schools, universities, everyday sectors, airports, medical visits, parks), so `enrich_all.py` is
not needed for a new pack. Parks, medical and airport terminals query Overpass in the US path,
which can be slow or down (T-196: an hour for nine packs on a bad evening; on 2026-10-13 every
server answered 504 for the Toronto area, so Canada's adapter reads the same OSM tags from the
Geofabrik extract instead, `canada.osm_caches`; the US path could do the same).

Then the place list for search and station names (T-177): `venv\Scripts\python.exe
trainworld\pipeline\places.py wor` → `data/packs/wor.places.json` (one Overpass request, cached
in `data/raw/osm_places_wor.json`; it prints the count by kind; works for any country).

Then routed walking and driving (T-275): `venv\Scripts\python.exe trainworld\pipeline\routes.py
wor` adds the `walk` and `drive` arrays from the Geofabrik extract of each region (`EXTRACTS` in
`routes.py`; Ontario is there, as region id 35 or ON; a US pack also takes every state its water
grid touches). Run it last, after anything that changes the pack's zones (a gravity re-solve
leaves the routes stale, and the game then ignores them); `routes.py --check` lists packs with
missing or stale routes. 1-3 minutes for a small or mid-sized pack, 8 for New York, plus 1-2
minutes to download and read a new extract; it needs the sim crate to compile (`pack_walk`), or
`TW_SIM_DIR` pointing at a copy of `sim/` that does. Its log gives the near zone pairs' median drive
factor (1.3-1.5 in the US packs) and the snapping distance (a median cell over about 300 m from a road means the
region's OSM roads are thin).

Time: about 1 minute for Worcester (1M people) with everything cached; T-134's Washington took
107 s; Toronto 10 minutes for the cells plus 22 for the rest (the 1 GB OSM extract read twice).
US downloads are per county (TIGER PL block zips, AREAWATER zips) plus one NCES query per city;
the state block DBFs, LODES WAC/RAC and the nationwide ACS `.dat` files (470 MB) are usually
cached already. A state new to the game also fetches its 2020 redistricting summary file
(`<st>2020.pl.zip`, 1-73 MB, T-234); `pipeline/group_quarters.py` keeps a small block table from it
(`data/raw/pl2020_p5_<fips>.csv.gz`).

What a US log should show (Worcester's figures):

- `check pop: blocks ... vs Census estimates base 2020 ... (+0.000%)`: exact or within 0.1%.
- `blocks <st>: ... N of them in institutions (T-234) -> ...`: about 0.5-1.5% of the people;
  `job holders on blocks with institutions capped ...`: about 0.1-0.5% of LODES RAC.
- `LODES blocks with no 2020 block centroid: 0 blocks, 0 jobs lost`.
- `check jobs: LODES 2023 ... vs BLS QCEW 2023 ... (+0.9%)`: within about 5%; do not calibrate.
- `work from home: ... blocks without a tract 0`.
- `job adjustments (T-014): 0 blocks` (no city-specific moves for a new city).
- `NCES schools: ... matched ... (100.00%) ... 0 omitted` (the build fails under 95%).
- IPEDS block: `location_match_share` 1.0, a plausible institution count.
- `gravity: ... decay d^-0.8 exp(-d/30 km) x 0.23 across regions` and, for a multi-state pack,
  `trips between regions` of a few percent.
- Top cells by jobs: a hospital or university lump is normal LODES; leave it (T-043).

## 5. Check each pack

    venv\Scripts\python.exe trainworld\pipeline\read_pack.py wor > artifacts\trainworld\T-###-wor-check.log
    venv\Scripts\python.exe trainworld\pipeline\look_water.py wor artifacts\trainworld\T-###-wor-water.png
    venv\Scripts\python.exe trainworld\pipeline\look_pack.py wor artifacts\trainworld\T-###-wor-pack.png

`read_pack.py` must print OK on the cells, commute, gravity (worst row and column sums around
1e-4 or better, zones split by state where it applies, the border factor) and water lines. Add
water and land test points to `WATER_TESTS` in `read_pack.py` and close-ups to `VIEWS` in
`look_water.py`, and look at the water image: rivers and lakes where they belong, nothing odd at
the edges. Record cells, population, jobs, commutes, zones and `.bin` raw and gzipped size.
(`look_pack.py`'s titles say "2020 Census" and "LODES WAC" for every pack.)

## 6. Registry

`app/src/game/registry.json` is the only place the app learns about a city (T-125). **Add the line
only once the pack, its boundary and its water mask exist**: every agent's clock worker loads every
registered city's water mask at startup, and the Node checks open every registered pack.

- Append to `cities`: `id`, `name`, `origin` (**exactly** the `build_city.py` origin, since it is
  the pack's frame), `center` (usually the origin), `zoom` 11.2, `utcOffset`, `market`.
- `utcOffset` (T-309): the city's standard time offset from UTC in whole hours, no daylight
  saving; round a half hour down (India's +5:30 is +5). Its periods, weekends and fares run on it.
- `market` is the city's index in `market.places`, or **-1 for a city outside the regional
  market** (Toronto, until T-276 places it; `intercity.py`, `outside.py`, the city bubbles and the
  worker's longer-trip selection skip it). To join the market, append to `market.places`:
  `{"name": "Worcester", "regionName": "Worcester region"}`. **Market ids are positions in this
  list: append only, never reorder or renumber.**

Vite then serves and copies the packs (`vite.config.ts`), the clock worker loads every registered
city's water mask at startup, `cityReach.ts` and the map read the boundary file, and saves list the
registry's cities in their header (an old save's list stays a subset, so it loads).

## 7. Regional market and zones

Every registered zone is in the market (docs/zones.md). A city's registry `market` is its place
id (append a place, step 6). Its pack header should name its country (`"country"`, ISO 3166
alpha-2; absent means US), which the joint commute gravity reads for border factors
(`pipeline/borders.json`: add the country's admin-1 default and its national pairs). Then, in order:

    venv\Scripts\python.exe trainworld\pipeline\zones.py           (non-city zones: registers built ones)
    venv\Scripts\python.exe trainworld\pipeline\outside.py         (the outside world, seconds)
    venv\Scripts\python.exe trainworld\pipeline\intercity.py       (sites, joint gravity ~12 min, derived block)
    venv\Scripts\python.exe trainworld\pipeline\zones.py --trips   (each zone's local trips, for the overview)
    cd trainworld & ..env\Scripts\python.exe pipelineoutes.py <id>   (last: T-275's routes, for new packs)

`intercity.py` rewrites every zone pack's gravity factors (the joint solve), so existing packs
change here too: record the commuters between packs it prints and each corridor's trips before
and after. A non-city zone is a `CITIES` entry in `build_city.py` with `"kind": "zone"` and a
`"name"`, one county-sized unit (docs/zones.md 2).

## 8. Tests that list cities

- `sim/src/demand/everyday.rs`, test `every_pack_everyday`: its city list is hardcoded; add the id.
- `checks/t066-worker-check.mjs` reads the registry (since T-277 it asserts
  `registry.cities.length`, not 9).
- Write a browser check for the new city or batch; `checks/t277-browser.mjs` (Toronto) is a template.

## 9. Checks

Native and Node (`npm run wasm --prefix trainworld/app` only if Rust changed):

    cargo test --release --features demand --manifest-path trainworld/sim/Cargo.toml every_pack_everyday -- --ignored --nocapture
    npm run check --prefix trainworld/app
    node trainworld/checks/t066-worker-check.mjs
    node trainworld/checks/t149-worker-check.mjs
    node trainworld/checks/t174-market-check.mjs      (a city in the market)

`every_pack_everyday` prints each city's per-resident outings: school about 0.11-0.15, university
0.02-0.08, errands and social at New York's rates by construction (Toronto: school 0.128,
university 0.033).

Browser (own Vite with HMR and watching off, `checks/dev-nowatch.mjs`; headless Chrome through Playwright with its own
profile directory; never touch app/dist). A new game, then for each new city:

- jump to its centre: it becomes the active city, its pack opens (`twDemand.client.start.loads`),
  its boundary layer `tw-city-boundary-<id>` exists; Homes bubbles show; screenshot to
  `artifacts/trainworld/T-###-<id>.png`;
- ground-level track across water is refused with `GroundOverWater` and the same route in a
  tunnel or on a viaduct accepted;
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
extracts go once `routes.py` and any OSM reads have cached their results. Delete
`data/work/pre-T###/` when the batch is accepted.

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
| Jobs at work | LODES WAC by block | census usual place of work by census tract of work (98-10-0498-01), at-home workers put back by income band, no fixed workplace added; to DAs by OSM job places and population | population x jobs a resident (World Bank: ILO employment-to-population 15+ x share 15+), spread 50% by GHSL non-residential built-up volume (GHS-BUILT-V NRES), 30% OSM job places, 20% population |
| Home end | LODES RAC x ACS work from home | commuters by tract of residence (98-10-0480-01) plus at-home workers commuting again | population x commuters a resident |
| Work from home | ACS 2019-2023 by tract and industry | Labour Force Survey May 2024 (24.7%, Toronto CMA), at both ends | one national share from the config (default 0) |
| Everyday destinations | LODES retail, service, food, leisure sectors | OSM shops and services; restaurants, bars, venues, hotels (Geofabrik extract) | the same OSM places, per cell |
| Gravity curve | fitted on LODES OD (T-198) | fitted on census flows between municipalities (98-10-0459-01; `notes/t277_fit.py`) | borrowed from a fitted city, or power 1 with the length fitted to a published median or mean commute (`tier0.fit_curve`) |
| Regions, border | states, 0.23 | provinces, factor 1 | one region, factor 1 |
| Drawn boundary | TIGER/Line counties, water included (T-352) | cartographic census subdivisions, clipped at the shore (hugs Lake Ontario) | the config's district polygons |
| Water | TIGER AREAWATER and states | StatCan lakes and rivers; land = cartographic municipalities plus US states | the config's `water_source` (OSM water is T-041) |
| Schools | NCES | Ontario school information file (provincial; other provinces need their own) | OSM `amenity=school` sites; pupils = population x share aged 0-14 (World Bank) x 0.9, by the square root of site area |
| Universities | IPEDS | Council of Ontario Universities headcounts and Ontario's college campus file, at points in `pipeline/ca_campuses.csv` | OSM `amenity=university|college`; Wikidata P2196 students by institution, split by site area; other sites by area at the median density |
| Airports | FAA enplanements, DB1B connecting share | StatCan 23-10-0253-01 enplaned; connecting share by judgement (Pearson 0.75) | OurAirports + Wikidata patronage / 2; connecting share by judgement; pack share from GHSL population |
| Medical, parks, places | OSM (Overpass) | the same scripts, OSM from the extract | the same, from the extract |

A tier 0 city is a `CITIES` entry with `"tier0": True` (the keys are listed in `tier0.py`'s
docstring); `build_city.py <id>` hands it to `tier0.main`. A country's census then replaces one
input at a time (the upgrades table in research/world-sampler.md).

What was hard, for the next country:

- **Blocked downloads.** StatCan's geography server (www12) sits behind a bot check, so the
  boundary and attribute files came from ArcGIS Online copies (Esri Canada's, and one user's copy
  of the census tracts). Its tables server (www150) is open. Look for such mirrors early, and list
  the official URLs for Anita.
- **No jobs by small area in the standard tables.** The census publishes workers by tract of
  work, but the latest census was taken in a lockdown: put the at-home workers back.
- **Provincial, not national, education data.** A second province needs its own school and
  college files.
- **Overpass** was down for the whole area; reading the Geofabrik extract with GDAL was faster
  and is the fallback for every OSM input.
- **The fitted curve.** Toronto's fits much steeper than the US curve (d^-1.2 exp(-d/20 km));
  the game now takes each pack's shipped curve rather than the US default.
