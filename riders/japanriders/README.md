# japanriders — Japan rail flow map

A national map of Japan's passenger rail network where **line thickness is
proportional to passenger throughput per segment** (輸送密度, *yusō mitsudo* —
passengers·day⁻¹ on each route section).

No origin–destination data is involved: each railway segment carries a single
published throughput figure. The map is a static-thickness picture of where the
trains are actually full.

## Pipeline

Two independent layers feed `index.html`: **lines** (passenger throughput per
segment) and **station bubbles** (per-station ridership, all operators).

```
lines ─────────────────────────────────────────────────────────────
data/tsukajinin_20260117.geojson   raw download from gtfs-gis.jp
        │  build.py                pick latest FY, drop sentinels, slim props
        ▼
data/segments.geojson              720-segment national base
        │  build_census.py         stitch in census per-segment detail
        ▼
data/segments.geojson              4,011 segments (225 features tapered,
                                   Tokyo + Nagoya + Osaka)

stations ──────────────────────────────────────────────────────────
data/S12-22/…/S12-22_NumberOfPassengers.geojson   国土数値情報 S12
        │  build_stations.py       all operators · latest 乗降客数 · 1/complex
        ▼
data/stations.geojson              8,368 station points (all operators)

names ─────────────────────────────────────────────────────────────
data/osm_jp_stations.json          OpenStreetMap name:en (Overpass extract)
        │  build_names.py          curated operator/line tables + OSM join
        ▼
data/names_en.json                 EN names — operators · lines · stations · remarks

                                   ▼  index.html
        MapLibre — line width ∝ throughput, bubble area ∝ station 乗降客数
```

Run **`build_census.py`** for the full line map (it calls `build.py`
internally, so it is a strict superset). Run `build.py` alone for the national
base only. Run **`build_stations.py`** for the station bubbles. Run
**`build_names.py`** last — it reads both outputs above. All run in a few
seconds — just run them directly.

### Station bubbles

A point layer drawn on top of the lines: one white circle per station complex,
area proportional to daily 乗降客数 (boardings + alightings). A toggle and a
size slider are in the legend; hover for the figure and the operators served.

- Source: **国土数値情報 S12 駅別乗降客数データ** (MLIT), FY2011–FY2021 — a
  uniform, geocoded, all-operator national file. `build_stations.py` takes each
  record's most recent fiscal year with a positive count, then groups every
  operator's records by S12 group code (駅グループコード — same name within
  300 m, assigned *across* operators) and sums: one bubble per physical station
  complex, carrying the total daily 乗降客数. The bubble sits on the busiest
  record.
- **Interchanges:** because the sum spans operators, 新宿's five operators
  (JR East, Toei, Keio, Odakyu, Tokyo Metro) collapse to one ~3.1M bubble — the
  conventional "world's busiest station" figure, which double-counts transfer
  passengers.
- **Year:** the newest S12 edition is FY2021, so most bubbles are FY2021 —
  earlier than the FY2024/FY2023 line data, and a COVID-depressed year. A
  station with no FY2021 figure falls back to its latest prior year (shown on
  hover). Relative sizes still read true; absolute magnitudes run a little low
  against the lines.
- **Coverage:** 8,368 station complexes, all operators (JR, metro, private,
  third-sector, tram, monorail). S12's JR East feed is partial and some private
  operators withhold unmanned-station counts; every station with visible
  ridership is in.

### v2 census stitch

In the national base, lines are coarse — every non-JR line is one uniform
feature, and many JR lines are one feature per route (gtfs-gis.jp publishes a
single 輸送密度 per line section). `build_census.py` replaces those, across all
three 大都市交通センサス regions (首都圏 / 中京圏 / 近畿圏 — i.e. Tokyo, Nagoya,
and Osaka–Kyoto–Kobe), with real station-to-station segments:

- Source: the 12th 大都市交通センサス `駅間通過人員` tables (station-to-station
  passing volume, 2015), one per region — see `data/census/`.
- Station coordinates come from the national N02 station file; census lines are
  matched to gtfs features by line name within each region's bounding box, with
  gtfs's bundled features (`京王線・高尾線・相模原線`) indexed per ・-component.
  Each census segment is cut from the real geometry by projecting its station
  coordinates onto the nearest strand — so multi-section JR routes and
  same-named collisions (JR vs 近鉄 奈良線) sort themselves out geometrically.
- **Reshape:** each gtfs feature's census segments are scaled so their length-
  weighted average equals that feature's gtfs-gis.jp 輸送密度 — the taper
  *shape* is the census's, the *magnitude* stays on the base map's fiscal year.
- **Census-direct:** gtfs-gis.jp has no 輸送密度 for ~22 lines (most 名鉄
  branches, some 南海 branches) — they are absent from the base map. Their
  geometry *is* in the raw file, so `build_census.py` adds them using the census
  values directly (2015 magnitude, no reshape anchor).
- **Un-surveyed remainder:** the census covers only each metropolitan area, so
  a line running past the census boundary (飯田線, 近鉄大阪線, 高山線 …) would
  lose its outer portion when its original feature is replaced. `build_census.py`
  keeps that remainder as uniform base-density segments — so a stitched line
  stays geometrically whole, tapered where the census reached and flat beyond.

225 gtfs features are stitched (3,497 segments, 330 of them un-surveyed
remainder). Left uniform / unstitched: JR operational services with no 戸籍
line of their own (`京浜東北・根岸線`, `湘南新宿ライン`, `上野東京ライン`,
`埼京線`, the split 常磐/総武 locals — their traffic is already inside the
戸籍 lines' totals); and lines whose census-to-FY ratio is implausible
(Shinkansen, rural lines the census barely sampled). A few short branches
outside every census region that gtfs-gis.jp never measured stay absent
(西鉄 甘木線・太宰府線, 名鉄 築港線, 南海 和歌山港線, the flood-suspended part
of 日田彦山線) — ~23 station bubbles sit on those with no line under them.

### English names

The `EN` / `日本語` toggle swaps the interface strings *and* every operator,
line, station name and segment remark in the tooltips and the info panel.
`build_names.py` writes `data/names_en.json` (keyed `op||name`, so `index.html`
looks names up with no runtime geo-matching):

- **Operators, lines and remarks:** curated JP→EN tables in `build_names.py`.
  Plain-ASCII Hepburn — `Marunouchi Line`, `Tokaido Main Line`, `Tokyo Metro`.
  `_expand_bundles()` also registers each ・-bundled line's components, so the
  table survives gtfs's bundled↔per-component naming swings. Remarks (the
  free-text 〜を含む / opening-date notes) are translated whole.
- **Stations:** joined to OpenStreetMap `name:en`. `data/osm_jp_stations.json`
  is a one-off Overpass extract of every railway station / halt / tram_stop node
  in Japan; each map station is matched by name (with ヶ/ケ, fullwidth, bracket
  and small-kana normalisation) and disambiguated by coordinate. ~99 % match
  automatically; the rest — abolished, renamed, BRT-converted, way-mapped or
  name:en-less tram stops — are a hand-romanised fallback table in the script.

If a future data rebuild introduces a line, operator or remark the tables don't
cover, `build_names.py` prints it as `MISSING` — add it to the table and re-run.

Serve locally (the page `fetch()`es the GeoJSON, so `file://` won't work):

```
python -m http.server 8000     # then open http://localhost:8000/
```

## Data

**Source:** [gtfs-gis.jp 鉄道輸送密度データ](https://gtfs-gis.jp/railway_tsukajinin/)
— a consolidated national GeoJSON maintained by Akira Nishizawa (U-Tokyo CSIS),
updated roughly quarterly. File used: `tsukajinin_20260117` (2026-01-17).

- True per-segment data (FY2024): JR Hokkaido, JR East, JR West, JR Shikoku,
  JR Kyushu — from each company's IR disclosures.
- One value per line section (FY2023): JR Central, all private rail, Tokyo
  Metro, Toei, third-sector, trams — from MLIT 鉄道統計年報.

Sentinels dropped by `build.py`: `-9999` (not surveyed), `-9000` (service
suspended — e.g. flood-damaged sections of 肥薩線 and 日田彦山線).

## License / attribution

Line data: **CC-BY 4.0 / ODbL** — credit Akira Nishizawa (地域・交通データ研究所).
Geometry: 国土数値情報 (鉄道データ N02), MLIT.
Station ridership: 国土数値情報 (駅別乗降客数データ S12), MLIT.
English station names: **© OpenStreetMap contributors** (ODbL), via Overpass.
