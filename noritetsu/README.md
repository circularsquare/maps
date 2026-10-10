# noritetsu

A map of every passenger rail line, and a record of the ones you have ridden. To live at
anita.garden/noritetsu. `spec.md` is the design and §13 is the todo, `HANDOFF.md` is how to
pick the project up and who owns which files when two people work on it at once, and this
file is how to run it and what state it is in.

**Japan, mainland China, South Korea, Taiwan, Hong Kong, Singapore, Switzerland, France,
Belgium, the Netherlands, Austria, Czechia, Poland, Hungary, Portugal, Slovenia, Slovakia,
Romania, Bulgaria, Finland, Lithuania, Latvia, Estonia, Croatia, Greece, Luxembourg and
Russia.**
`HANDOFF.md` "Start here" has the state and the open threads. In 16 of the RINF countries a
national timetable feed decides which register sections trains run over (`gtfs_served.py`,
`gtfs_sources.md`). China comes from
OSM's named track plus 12306's station list (`cn_register.py`). The pipeline is
region-by-region, and the app loads each country as the map moves over it. Hong Kong and
Singapore follow Taiwan's pattern (`hk_register.py`, `sg_register.py`); the EU countries come
from `rinf.py`, one reader for every country in the EU's infrastructure register, with a small
settings file per country in `rinf_countries/`; France
from SNCF Réseau's own register (`fr_register.py`). Each has a `<cc>_sources.md` with its
sources and run commands. `multi_sources.md` is the survey of sources that cover many
countries at once. Switzerland is built from the federal network register
(`schienennetz.py`); Korea from OSM's named track plus Korail's and KRIC's published station
lists (`kr_register.py`, `kr_sources.py`). Both check out against published lengths.

## Run

```powershell
python serve.py                 # http://localhost:8767
```

`serve.py` rather than `python -m http.server` because a .pmtiles archive is read over HTTP
range requests, and SimpleHTTPRequestHandler ignores `Range` and answers 200 with the whole
file. pmtiles.js reads that as a broken server and the map comes up with a basemap and no
rail on it, which looks exactly like a bad build.

## Deploy

Live at https://anita.garden/noritetsu/ (first published 2026-10-09, marked "In progress").

```powershell
python tools/deploy.py --check       # preflight and sizes only
python tools/deploy.py --dry-run     # preflight, then what rclone would send and delete
python tools/deploy.py               # upload data to R2, copy the page into the website repo, verify
python tools/deploy.py --verify      # check the published copy on its own
python tools/deploy.py --skip-data   # page files only (index.html, poster.js, share.js)
```

Then commit and push the website repo by hand (`git add noritetsu/` there); the script never
does. What goes where: `regions.json` and all of `dist/data/` go to Cloudflare R2 under
`r2:anitamaps/noritetsu/` (public at
`https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/noritetsu/`); `index.html`, `poster.js`
and `share.js` go to `website/noritetsu/`. The page finds its data by asking for
`regions.json` beside itself (index.html, "WHERE THE DATA IS"), so **never copy regions.json
or data/ into the website repo**: the published page would then look for its data on GitHub
Pages, which also breaks the range requests a .pmtiles needs. The json goes up gzipped
(`Content-Encoding: gzip`, staged in `data/deploy_gz/`), because R2 compresses nothing; the
.pmtiles and logos go up raw. Both uploads are `rclone sync` limited to noritetsu/data, so a
geometry file the build stopped writing is deleted from R2 too.

The preflight refuses to upload while a build looks active (dist/data or a build log written
in the last 10 minutes, a `tools/slot.py` slot held, a build script running) and when a
country in regions.json lacks any of lines / stations / foot / ways / aliases / types json or
its .pmtiles, when an old `data/<cc>/credits.json` is still there (`logos/credits.json` is the
logos' attribution and ships), when search.json, closed.json or operators.json is missing, or
when the expression lint fails. A full first upload was ~0.9 GB in ~22,000 files.

rclone's `r2:` remote is configured already; only the token goes stale. A 403 on `rclone lsd
r2:` means nothing (listing buckets is an admin call); test with `rclone lsf r2:anitamaps
--max-depth 1`. `SignatureDoesNotMatch` is a bad secret, `directory not found` a wrong bucket
name, any other 403 an expired token: Cloudflare dashboard > R2 > API tokens, Object Read &
Write, into `%APPDATA%\rclone\rclone.conf` under `[r2]`.

## Build

```powershell
python extract.py --region jp --pbf data/raw/japan-260928.osm.pbf   # 3.5 min
python inspect_region.py --region jp                                # is this extract any good
python build_model.py --region jp --register n02:data/raw/N02-24_GML.zip   # 90 s
python build_tiles.py --region jp                                   # 2 min -> dist/data/jp.pmtiles
python check_model.py --region jp                                   # built vs published lengths
python -m unittest discover -s tests

python extract.py --region ch --pbf data/raw/switzerland-260929.osm.pbf   # 1 min
python build_model.py --region ch --register schienennetz:data/raw/schienennetz_2056_de.gdb.zip
python build_tiles.py --region ch                                         # 30 s, 3.0 MB
python check_model.py --region ch

python extract.py --region kr --pbf data/raw/south-korea-260929.osm.pbf   # 25 s
python build_model.py --region kr --register kr_register:data/raw/kr      # 30 s
python build_tiles.py --region kr                                         # 15 s, 1.8 MB
python check_model.py --region kr

python build_model.py --region tw --register tw_register:data/raw/tw      # Taiwan
python build_tiles.py --region tw                                         # 0.6 MB
python check_model.py --region tw
```

Korea's register wants the files listed in `kr_sources.md` in `data/raw/kr/` (all open, no
login; `probe_kric.py --fetch` gets the KRIC ones, and `kr_sources.md` says how to fetch the
data.go.kr ones).

**Tiles come after the model**: `build_tiles.py` reads each piece of track's line colour
from `build_model.py`'s output. Run before it, the tiles build with kind colours only.

Switzerland's register wants `data/raw/ch_servicepoints.csv` beside the network file; both
URLs are in the docstring of `schienennetz.py`.

The `.osm.pbf` comes from Geofabrik (`https://download.geofabrik.de/asia/japan-latest.osm.pbf`,
2.5 GB) and is gitignored. Extraction needs `pip install --user osmium`. All three extracts
were deleted after extracting on 2026-09-30; `data/proc/<region>/` keeps what came out of
them. Geofabrik's `-latest` URLs for Switzerland and Korea were redirect-looping that day; the
dated `<country>-YYMMDD.osm.pbf` files worked.

## What each step does

- **extract.py** — three passes over the .pbf, writing raw material to `data/proc/<region>/`:
  railway ways, rail route relations, stopping places, and coordinates for exactly the nodes
  those reference. Three passes rather than one because pyosmium's node-location index costs
  ~12 bytes for *every* node in the file — 2.4 GB for Japan, ~100 GB for a planet. Nothing here interprets the data, so re-deciding what a line is
  never means re-reading the .pbf.
- **build_tiles.py** — the faint all-lines background, as `dist/data/<region>.pmtiles`.
  Japan: 10.5 MB, 11,104 tiles, z0–13; Switzerland 3.0 MB. Each piece of track carries
  the colour of the line on it (`c`), read from `build_model.py`'s output.
- **build_model.py** — lines, stations and sections, as `dist/data/<region>/`. With `--n02`
  it builds the national register from `n02.py` and merges OSM onto it. Japan (2026-10-02):
  1,094 lines of which **593 are register lines** (27,122 km), 9,073 stations, plus per-line
  geometry in `geom/` fetched on demand, `foot.json` for crediting (track ownership) and
  `ways.json` for resolving a click on track.
- **probe_n02.py** — what is inside 国土数値情報 N02, the Japanese government's own railway
  inventory. Not built on yet; see `spec.md` §12c for why it is going to be.
- **inspect_region.py / check_model.py / probe_line.py** — the checks. `check_model.py
  --coverage` reports how much of the network only a named train reaches, which is the
  measure of how far OSM alone falls short. Run `check_model.py`
  after any change to the model; it compares built line lengths against published operating
  lengths and everything should stay within a couple of percent.

## State

Built and verified:

- Extraction, tiling and the line model, run end to end on Japan. The pipeline is
  region-agnostic; only `check_model.KNOWN` and one naming rule are Japan-specific.
- The map: pan and zoom, LOD z0–z13, two palettes by mode, click a line or station.
- Line lengths validated against published operating lengths — worst deviation 2% across
  eleven lines.
- Crediting by track ownership: every piece of track belongs to exactly one line, and riding
  any service credits the lines whose track it runs over (`ownership.py`, `foot.json`;
  `python check_model.py --region jp --shared 山手線`: 20.1 of the register 山手線's 20.5 km).
- **The tracker.** Click a station or a line on the map, or search either, then tap where you
  got on and off, on the diagram or on the map. Recorded with an optional date, and the
  destination becomes the origin of the next leg. Ridden track is drawn over the map in each
  line's own colour, and percentages are kept per line, per operator and overall. Rides live
  in localStorage and export to / import from a JSON file. A journey can also be traced by
  clicking its stations in order; a whole operator can be marked ridden at once; a ride can
  go the other way round a loop; and every change can be undone (Ctrl+Z) and a ride's date
  and note edited afterwards.

Not built yet — see `spec.md` §13 for the full list:

- Station-to-station routing (input mode 1), which needs a graph across lines rather than the
  per-line one the strip diagram already uses.
- The poster generator, and line detail panels.
- **More regions** (Russia, Germany, Italy, Spain, Sweden, the USA...). 73 GB free as of
  2026-09-30, and the ~90 GB a world build needs is download rather than peak disk.
  `HANDOFF.md` says what a new region's reader has to produce.
- **Track across borders** is drawn where the EU register has a border point (`borders.py`);
  crossings without one (Swiss-German, the Channel Tunnel, borders outside the EU) are added
  by hand when the country on the other side is built.

## Things that cost a rebuild to learn

- **A line's track is a graph, not a chain, and its sections come from an absorbing search.**
  Merging double track end to end doubles a line's length; ordering stations from a terminus
  breaks on loops. `n02.py` explains both, with the numbers they produced.
- **Station bubbles must come from the model, not the tiles.** The tiles carry every station
  node OSM has, which at a complex like Kita-Senju is one per operator stacked on top of each
  other, and no way to tell which to click.
- **Sections have to be cut at the finest granularity available.** Deriving them per variant
  from each variant's own calling points double-counts shared track: the Tokaido Main Line
  came out at 5,719 km against a published 590. `build_model.place_stations` explains it.
- **Shared track has to be found geometrically, not by rail identity.** Fingerprinting the OSM
  nodes a section runs over finds nothing between the Yamanote and the Tohoku Main Line — the
  loop has its own pair of tracks thirty metres away. Track ownership assigns each way to a
  line by geometry, then credits by the ways a service runs on (`ownership.py`).
- **Merge ways into chains before simplifying, or the world view is blank.** An OSM way is a
  few hundred metres, which is sub-pixel at z0, so every one of them gets dropped.
- **The basemap draws its own railways.** Six layers of them, in near-black, from the same
  OSM data. They are hidden on load; left alone they read as a second ghost network.
- **`['*', ['interpolate', ['zoom'], ...], factor]` is invalid MapLibre** and silently drops
  the whole layer. `node tools/lint_map_expressions.js dist/index.html` catches it; run it on
  every edit to index.html.
- **Anchor added layers at the trailing label block**, not at the first symbol layer — in this
  style that is a water label sitting before the roads, which buries the rail network.
- **The console here is cp1252 and the data is not.** Scripts that print Japanese names
  reconfigure stdout to UTF-8 first, or they die in a `print()` after all the work succeeded.

## Sources

- [OpenStreetMap](https://www.openstreetmap.org/copyright), ODbL — all rail geometry, stations,
  lines and colours, via [Geofabrik](https://download.geofabrik.de/) extracts. Attribution is
  on the map; a published derived database would have to be offered under ODbL too.
- [OpenFreeMap](https://openfreemap.org/) / OpenMapTiles — basemap.
- MapLibre GL JS 4.7.1 and pmtiles 3.2.0, pinned to match the other maps in this repo.
- Published operating lengths (営業キロ) in `check_model.py` are the operators' own figures,
  which is also what the Japanese line-completion hobby counts.
- [国土数値情報 N02](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-v3_1.html) (MLIT),
  Public Data Licence 1.0 — Japan's line register.
- [Schienennetz](https://opendata.swiss/de/dataset/schienennetz) (BAV, via geo.admin.ch) and
  the [service-point list](https://opentransportdata.swiss/en/cookbook/masterdata-cookbook/servicepoints/)
  (opentransportdata.swiss) — Switzerland's line register and passenger stops. Open
  government data; free use with the source named.
- Korail's 각 선구별 거리표 and ㈜SR's distance matrix (data.go.kr 15137040, 15040194),
  KRIC's 전국 도시광역철도 역사정보 (data.kric.go.kr 1294), RAFIS 역 정보 (data.go.kr
  15132601), and published lengths from Korail's 2023 철도통계연보 — Korea's station lists,
  section km and English names. Korean public data; each dataset carries its own 공공누리
  licence type, not yet checked one by one, so check before publishing. `kr_sources.md` has
  every file.
