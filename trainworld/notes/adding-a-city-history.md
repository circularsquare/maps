# adding-a-city.md, material cut in the T-588 tidy (2026-10-17)

Moved out of docs/adding-a-city.md: history, timings and lists that no longer decide a build. The doc keeps the rules and commands.

## Intro as written 2026-10-12 to 2026-10-13 (T-179, T-277)

Written 2026-10-12 (T-179) from how Boston and Washington (T-134), the five-city batch (T-137), the
non-work purposes (T-142), the border factor (T-162) and the market's derived block (T-174) were
done. **Followed end to end once, for Toronto (T-277, 2026-10-13)**, the first city outside the US;
that run rewrote it. Fix this file wherever the next run finds it wrong.

## T-179's US candidate counties (all disjoint from the packs then)

T-179's US
  candidates, all disjoint from the packs:
  - Worcester: 25027 Worcester MA, 09015 Windham CT
  - Allentown: 42025 Carbon, 42077 Lehigh, 42095 Northampton, 34041 Warren NJ
  - Albany: 36001 Albany, 36083 Rensselaer, 36091 Saratoga, 36093 Schenectady, 36095 Schoharie
  - Pittsburgh: 42003, 42005, 42007, 42019, 42051, 42073, 42125, 42129
  - Richmond: 51007, 51036, 51041, 51053, 51075, 51085, 51087, 51101, 51127, 51145, 51149,
    51183, 51570, 51670, 51730, 51760 (the 2023 metro without Caroline and King and Queen)

## Timings and what a US build log should show (section 4)

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

## Worked examples as written, with timings (Outside the US)

Paris (`par`, T-289, `pipeline/france.py`) took about an hour of wall time: Ile-de-France's
extract read in 72 s (features) and a few minutes (water), the build 66 s, `routes.py` 3 minutes.
One INSEE file (the 2022 flows, every flow, 12 MB) gave both the jobs by commune of work and the
curve fit (`notes/t289_curve.py`, which also makes the signals cache, so the build that follows
reuses it). Communes and their legal populations come from geo.api.gouv.fr in one request a
département. Take water test places from the water cache's own polygons, not from memory (three
of five guessed points missed).

Shanghai (`sha`, T-291, `pipeline/china.py`) took about 30 minutes: the extract read in seconds,
the build 7 minutes, `routes.py` 1. Districts come from the extract's OSM admin_level 6
relations clipped to land by the coastline; relations that GDAL cannot assemble (Jing'an,
Baoshan) are rebuilt from admin_level 8 units or as the province's remainder. Population and
jobs are scaled to district tables from the city statistics bureau's census bulletins (the
economic census's bulletin no. 1 has employment by district): look for both before the first
build. Places show `name:en` or pinyin where OSM has it (`"latin_names"`).

Mumbai (`mum`, T-292, `pipeline/india.py`) took about 40 minutes: the western zone extract read in
38 s (features) and 64 s (water), the build 3 minutes, `routes.py` 96 s. Units are OSM district
and taluka relations by name; the census tables (Primary Census Abstract per district, A-2, B-28)
are on censusindia.gov.in/nada (certificate checks off for that host only). Compare GHSL with each
unit's census total on the first build's log: where OSM's lines between units do not follow the
census's, scale those units as one group. A census table of commuters by distance band fits both
curve parameters (`"curve": {"bands": ..., "road_factor": ...}`, `tier0.fit_curve_bands`, about
100 s in the build).

Sao Paulo (`sao`, T-458, `pipeline/brazil.py`) took about an hour: Brazil's Sudeste extract (860
MB) downloaded and read in 6 minutes (features) and 5 (water), the build 2 minutes, `routes.py` 5.
IBGE's 2022 tracts with their people are one GeoPackage a state; a household travel survey's
microdata (the Metro's OD 2017) gives jobs by zone, work from home and the curve in one file:
look for a city's survey first. Merge the curve's bands under 4 km (2 km zones cannot split them).
The World Bank API can time out: fetch its three JSON files into `data/raw/tier0/` with curl.

Lagos (`lag`, T-459, `pipeline/nigeria.py`) took about 36 minutes from global sources alone:
Nigeria's extract (711 MB) downloaded in 4 minutes and read in 7 (features) and 30 s (water), the
build 80 s, `routes.py` 3 minutes. `nigeria.py probe <id>` lists the admin relations in the box and
`nigeria.py <id>` prints GHSL by unit before any build. Where OSM maps few shops and offices, read
the build's `osm_signal` line (above): land use dominates the signal. Run Python with
`PYTHONIOENCODING=utf-8`, or OSM names with tone marks stop a print.

Berlin (`ber`, T-290, `pipeline/germany.py`) took about 50 minutes: Brandenburg's extract (which
includes Berlin: never add Geofabrik's Berlin extract beside it, the features would count twice)
read in 97 s (features) and 107 s (water), the curve script with the signals 4 minutes, the build
3.5 minutes, `routes.py` 3. BKG's VG250-EW gives Kreise, Gemeinden and their population in one
file; the Pendleratlas's CSVs (`/data/csv/<year>/<year>_<table>_L<Land>.csv`, found in its JS
bundle) give jobs by Gemeinde of work and the top flows for the curve. Water test points must sit
on a channel's middle (polylabel on the cached polygon); a canal under 50 m wide is below the
mask's 2-pixel width.

Tokyo (`tyo`, T-420, `pipeline/japan.py`) took about two hours, an hour of it the build (before
T-425, parks 35 minutes and medical 15; now they take 2 minutes together; routes 13). A Japanese pack is a config
row: `prefectures`, `neighbours` (the land beyond the outline, for meshes it cuts) and the region's
telework share; `python pipeline/japan.py fetch <id>` downloads the census meshes, N03, table 4
and P29 first, and `japan.py outline <id>` draws the outline. Run `notes/t420_curve.py <id>` for
the curve before the build (it makes the signals cache; 25 minutes on Greater Tokyo's 4,900 zones).
Rebuilt with Ibaraki (T-426): the extract's features and water 6 minutes, the build 10, `routes.py
tyo --extracts asia/japan/kanto` 14 (Japan is not in `routes.EXTRACTS`). Place names need fugashi
and unidic-lite in the venv (`japan.kanji_romaji`). A rebuilt pack's caches (`data/work/<id>.*`,
the `osm_*_<id>` files under `data/raw/`, `<id>_prefectures.geojson`) must go first, or the old
extent's are reused. Fukuoka, Sapporo, Sendai and Hiroshima (T-432) were a batch of config rows:
a Hokkaido pack names subprefectures (`japan.SUBPREFECTURES`) or one municipality (`"01-01203"`);
each regional metro fits its own curve (4-12 minutes, before the build); builds 2-5 minutes,
`routes.py` 1-3. Japanese downloads go through curl (`japan._get`, T-441): KSJ's server, and
Geofabrik's on a bad night, stall a Python request for good. The rest of Japan (T-420 stage 3) is
38 non-city zones built as one batch by `pipeline/japan_zones.py`: `plan` (zones, neighbours,
extracts, origin, wfh into `japan_zones.json`), `build <id>` (outline, a curve fitted over a fixed
grid in a few minutes, the pack, places, routes, merge, `read_pack`, water picture) and `run
<ids> --jobs 4` (a process a zone, a log and status line each); 2-76 minutes a zone. Build the few
zones least like each other first (an island zone, a big one, a merged one): a batch finds a
tier 0 bug 9 times over. Check table 4's municipality codes against N03's before building (wards
redrawn since 2020 starve a municipality of jobs).

Seoul (`seo`, T-457, `pipeline/korea.py`) took about an hour: South Korea's extract read in 3
minutes (features) and 1 (water), the build 7, `routes.py` 5. Population is the resident
registration count by si-gun-gu from jumin.mois.go.kr (`korea.py fetch`, one POST a province, no
key); districts are OSM relations by name inside their province, the admin areas cached in
`data/raw/tier0/korea/<id>_admin.fgb` (the extract's multipolygon read takes 5 minutes). Check the
district names against OSM first: units renamed or split since the table's year need a mapping
(`REGIONS[...]["osm_names"]`). A test line costs about 200 a station, so a new game's 6,000 builds
about 29.

## What was hard in Toronto

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
