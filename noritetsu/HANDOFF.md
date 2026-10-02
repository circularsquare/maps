# Handoff

The file a managing session or a country agent reads first (rewritten short on 2026-10-02).
`HISTORY.md` has how each country was built, the declined proposals with their numbers, the
traps that cost a rebuild and the session logs. `spec.md` is the design (§13 the todo),
`README.md` how to run things.

## Start here

**30 countries are built**, all in `dist/regions.json` and `dist/data/`. Register lines and km
as shipped on 2026-10-02 (`src != "osm"`, closed sections included):

| cc | register from | lines | km | notes in |
|---|---|---|---|---|
| jp | N02 (`n02.py`) | 593 | 27,122 | spec, HISTORY |
| ch | Schienennetz (`schienennetz.py`) | 402 | 5,567 | HISTORY |
| fr | SNCF Réseau RFN (`fr_register.py`) | 278 | 24,227 | `fr_sources.md` |
| kr | OSM named track + Korail/KRIC lists (`kr_register.py`) | 83 | 4,897 | `kr_sources.md` |
| tw | the same recipe (`tw_register.py`) | 36 | 1,792 | `tw_sources.md` |
| cn | OSM named track + 12306 (`cn_register.py`) | 417 | 121,894 | `cn_sources.md` |
| hk, sg | MTR / LTA lists (`hk_register.py`, `sg_register.py`) | 13, 10 | 293, 269 | `hk_`, `sg_sources.md` |
| be, nl, at | ERA RINF (`rinf.py`) | 148, 96, 125 | 3,199, 2,780, 4,342 | `<cc>_sources.md` |
| cz, pl, hu, pt | RINF | 244, 358, 133, 24 | 9,157, 16,281, 6,736, 2,145 | `<cc>_sources.md` |
| si, sk, bg, ro, fi | RINF | 22, 70, 32, 86, 31 | 1,157, 3,389, 3,595, 9,015, 4,029 | `<cc>_sources.md` |
| lt, lv, ee | RINF | 16, 10, 13 | 1,215, 923, 721 | `baltics_sources.md` |
| hr, gr, lu | RINF | 39, 12, 16 | 2,339, 1,726, 237 | `<cc>_sources.md` |
| ru | the tariff guide, through rinf.py (`ru_register.py`) | 845 | 75,485 | `ru_sources.md` |
| de, it, es | RINF (DB InfraGO; RFI + 8 regional managers; Adif, Adif AV, FGC) | 1,097, 302, 166 | 31,362, 16,604, 14,366 | `<cc>_sources.md` |

**State.** Every country has `foot.json` (track ownership) and no `credits.json`. Track over
a border is drawn where ERA RINF has a border point (`borders.py`, `border_points.json`). A
national timetable feed decides which register sections trains run over in 16 countries (cz,
hu, pt, pl, be, si, sk, ro, bg, fi, lt, lv, ee, hr, lu, gr; off for at and nl;
`gtfs_served.py`, `gtfs_sources.md`). Germany, Italy and Spain were added on 2026-10-02 and
every country rebuilt in one batch the same day (`tools/rebuild.py`, 3 at once: 30 countries
in 23 min; register lines unchanged everywhere; the doubled border ids now one id on both
sides; 305 line ids shared between countries).

### How the work is run

- **One managing session holds the shared files**: `build_model.py`, `ownership.py`,
  `build_tiles.py`, `extract.py`, `rinf.py`, `borders.py`, `line_colours.py`,
  `not_running.py`, `gtfs_served.py`'s hook, `tools/` and `dist/index.html`. It lands every
  shared change and runs `tools/build_regions.py`.
- **Country agents, up to eight at once, own one country each**: `<cc>_register.py` or
  `rinf_countries/<cc>.py`, `<cc>_sources.md`, `colours/<cc>.csv`, `data/raw/<cc>*` (and
  `data/raw/rinf/<cc>`, `data/raw/gtfs/<cc>`), `data/proc/<cc>`, `dist/data/<cc>*`, and their
  own `REGISTER["<cc>"]` in `check_model.py`. For a shared file they send the managing session
  an exact diff. Only one session rebuilds a given country, so outputs are never clobbered.
- An app session, if one runs beside them, owns `dist/index.html` and asks for a new output
  field rather than editing the Python build. Never two sessions in `build_model.py`.
- Builds run freely in noritetsu unless one would take over ~2 hours (Anita). The permission
  classifier blocks subagents' all-country rebuilds and new downloads: the managing session
  runs those, with Anita's say-so.

### Anita's standing decisions

- **Completion counts everything with scheduled passenger service** ("if it is scheduled at
  all, more often than about once a week"). **Named trains (option B)** have no percentage of
  their own and totals leave them out; their track counts through the lines it lies on.
- **One owner per piece of track**: "no double counting, every piece of track belongs to
  exactly one line; for crediting only, trips can still be entered on any service or path"
  (`ownership.py`).
- **Border track** counts in the country it lies in; a ride over the border credits both. A
  section built whole over a border is cut and the far part handed to the neighbour once it is
  built. A crossing RINF has no point for is added to `borders.EXTRA` by hand when the country
  on the other side is built. A border point shows one neutral name ("Belgium – France
  border"), not either country's RINF name.
- **Seasonal lines are drawn as running** (Poland's 96 Muszyna - Leluchów, Croatia's
  Metković - Ploče). **Not running is greyed** and left out of completion (Porvoo is fine
  greyed).
- **Russia**: the tariff section is the line unit. The 2022-annexed railways (Donetsk,
  Luhansk, Melitopol - Kherson; 53 lines, 1,291 km) are left out, given to no country, until a
  source says which trains run there; then set `ANNEX_RUNNING = True` in
  `rinf_countries/ru.py` and add `data/raw/ru/annex.geojson` to `EXTRA_AREAS` in
  `tools/build_regions.py`. Crimea is built with Russia.
- **Croatia's single "B" fast trains stay named trains.** Czechia's JHMD (228/229 out of
  Jindřichův Hradec): unknown whether it runs, leave as is.
- **Data**: inspecting data only reads it (no caches or side files beside it). Ask her before
  new downloads or anything else that writes to data beyond an agreed build.

### Open threads, in the order I would take them

0. **Next session, first** (agreed order at the end of 2026-10-02):
   - **Named trains merged on a shared ref alone.** `merge_osm_twins` merges two named trains
     on the same ref even when their names differ, so in be both European Sleeper routes and
     "Eurostar: Paris - Amsterdam" are merged into the Eurostar (all ref "ES"): a European
     Sleeper ride can't be entered. Tried "named trains never merge on a ref alone" (tools/ab.py,
     not landed, reverted): it splits the European Sleeper and Paris - Amsterdam in be and EC
     Genève - Milano in it, as wanted, but also splits direction twins of one train (NJ 40235 in
     de/at/it, an AVE Madrid - Malaga in es, はやて and 小倉 => 下関 in jp). Next try: a shared ref
     merges named trains only when their names also start with the same word (the rule
     tools/build_regions.py's `line_aliases` already uses); ab.py --all, then rebuild the
     changed countries and build_regions.py.
   - **The GTFS check for de, it, es** (thread 1).
   - **Per-country rules out of build_model.py** (`looks_like_service` branches and the like into
     each country's own file, so country agents stop waiting on the managing session): the last
     of the four process changes Anita agreed on 2026-10-02.

1. **Germany, Italy and Spain are built** (2026-10-02; `de_`, `it_`, `es_sources.md`). Next:
   - **The GTFS check for de, it, es**, which matters most here: Italy drops 672 km of real
     passenger track for want of OSM routes (all of AV Treviglio - Brescia), Spain the Pajares
     base tunnel (984, 49 km) and Chinchilla - Hellín, and Germany keeps freight lines between
     two passenger stations (1280 Buchholz - Allermöhe, 5230 Werntalbahn) that a feed would
     grey. Feeds: gtfs.de fv_free + rv_free (join by name); Trenitalia (deryclem's NeTEx
     conversion) + Trenord + Italo; Renfe AV/LD + Cercanías (stop ids are RINF's uopid less
     "ES").
   - **Hand-added border points** (`borders.EXTRA`; both sides built now): every Swiss-German
     crossing (Basel Bad Bf, Schaffhausen, Koblenz AG, Kaiserstuhl...), Euskotren's E2 Irun -
     Hendaia, and the other route ends Germany's build logs as "no border point" (Wasserbillig,
     Kehl, Enschede, Hergenrath, Zgorzelec...: check which have a RINF point on the far side).
   - DB Infrastrukturdaten's licence (names come from it): multi_sources.md says CC BY 4.0,
     GovData showed none on 2026-10-02. Confirm.
   - Italy's "Nodo di ..." city lines (RFI's unit; Nodo di Roma 194 km): split into the lines
     inside each city as it.wikipedia lists them (Anita: yes, 2026-10-02; not started).
     Spain's catalogue names ("100 Hendaya – Madrid-Chamartín-Clara Campoamor") and its two
     operators (Adif, Adif AV) stay as built (Anita: OK).
   - The Canary Islands (Tenerife tram) need Geofabrik's africa/canary-islands extract.
   - Then Sweden (needs a name map).
2. **Look at the border crossings in the app** (built, not yet seen on the map): K80 Kortrijk -
   Lille, the Eurostar Amsterdam - Paris, HSL 1. Open: be's European Sleeper -> Eurostar merge
   gives the Eurostar a one-sided Antwerpen - Essen border section; fr's extract still builds
   ~30 short foreign sections in its buffer (IC-04, IC-26, L-29; 108 km in all); Longwy
   (202 000) stays 95% (its last 1.2 km has no trains and no node); ch's Feldkirch - Buchs
   border section is greyed as not running although ÖBB trains run there.
   Done 2026-10-02 (cleanup agent, `fr_sources.md`, schienennetz.py docstring): fr_register's
   dead ends end at the RINF border point under its `eEU` id (`snap_borders`, 30 ends; Basel,
   Portbou, Le Locle, Lyon - Genève now 100%); Swiss ends within 30 m of a point take the
   `eEU` id. **Crossings RINF files twice** (a point per country at one spot: bg EU00208 / ro
   EU00209, cz-pl EU00072/73, at-ch EU00118 / CH15472, ch EU00157 / CH15452, sk EU00162/163)
   are one id: `borders.canon` (within 15 m; eEU first, then lowest), applied to a register's
   output by `build_model.canon_border_ids` and to tracing by `borders.load(canonical_only=True)`;
   gtfs_served counts any table id as a border. Checked with tools/ab.py on at ch cz pl bg ro sk.
3. **After ownership** (cleanup agent, 2026-10-02): Seoul 1/3/4, Athens M3, the Austrian
   Railjets and T12 now credit their register lines; PKM's own 14 km is not in RINF (the OSM
   line owns it). Open:
   - **T13** on the Grande Ceinture: register line 990 000 is rail, T13's track light_rail, so
     990 000 owns 13.5 km no ride can credit. Needs a design choice in ownership.py (let a rail
     register line own light_rail ways where no other register line is there, or split it).
   - **France's missing lines**: done 2026-10-02 with Anita's OK. SNCF's line files lack them;
     its per-track file `data/raw/fr/voies-de-ligne.geojson` (39 MB, ODbL) has them, and
     fr_register reads it when present: +226 310 LGV Interconnexion Est 57.3 km, 262 000 Douai -
     Blanc-Misseron 30.1, 657 000 Lamothe - Arcachon 15.8, 768 300 Pasilly - Aisy 15.1, 958 000
     Bondy - Aulnay (T4) 7.8; LGV Est phase 2 on the finer track shape; 258 000's 0.6 km stub
     now lies on 262 000's rails and drops.
   - **Long connecting curves (Rac)** that TGVs use, ~100 km, count nowhere (excluded by design
     in fr_register); 48 exploited Racs of 3 km or more total 250 km.
   - **Geneva CEVA**: absent from the BAV file (2021-07-06 edition, still the only one).
     Lötschberg base tunnel fixed (330/331, 39.7 km; schienennetz now counts an end's
     neighbours, not segments). be 161A on 161's rails is correct: 1613 is 161's second pair.
   - fr: 894 000 CEVA-Annemasse 85% (a gap mid-line); 457 000 Segré - Nantes-État is a 2.4 km
     tram-kind stub (kept); fr's tram-trains on rail track (519 000, 782 000) now own their
     track. (fr's unnamed "Ouigo" and ru's 001/002 Красная стрела are named trains since
     2026-10-02.)
4. **Timetable follow-ups**: spot-check pl 281 Nakło - Chojnice (may have restarted 2026), ro
   600 and 700. Works closures (pl Opole - Nysa to 24 Oct, Jelenia Góra to 30 Oct, Cieszyn to
   13 Dec) stay greyed until the feed is refetched and pl rebuilt. Greece's feed ends
   2026-12-01: refetch then. pl 295 Węgliniec - border may be a summer works diversion.
5. **Russia**: line colours; English names from Wikidata by station code (needs a rinf hook).
6. **Known general issues** (detail in `HISTORY.md`, "second EU round"): `rinf.line_hash`
   gives an unnumbered id the same line id as a public number with the same digits (Romania's
   Blaj - Praid; the fix needs line aliases for saved rides); `n02.walk_order` lists only the
   first piece of a two-piece line in `display` (Bulgaria's 3 and 4); a line with only an infra
   relation is no line and its track is dropped (Bulgaria's Septemvri - Dobrinishte, 125 km,
   Greece's Pelion, Zagreb's funicular).
7. Fixed 2026-10-02, live at the next rebuild: names differing only in a dash ("Charles de
   Gaulle-Étoile" / "Charles de Gaulle — Étoile") are one station (`fold_dashes`; ab.py on 24
   countries: only pl and lu move, one tram stop each, plus fr's 6).
8. **USA**: register lines are FRA NARN subdivisions, Amtrak routes services over them (Anita,
   tentatively); ~10 GB extract; the app may need lines.json split by region. Then India,
   Australia, Canada, the UK and the Nordics (`HISTORY.md` shortlist).
9. **App**: on a phone the open panel covers the map, so map picking and journey tracing only
   work through search (a bottom sheet would fix it); a headless `?region=kr` once opened on
   Japan (FOCUS 'jp'), probably a race between `regions.json` and the START check. The rest is
   spec §13, item 3 onward.

## The process for shared changes

- **Trial first.** `python tools/ab.py <cc...>` builds the working tree into a temp folder and
  compares with `dist/data` (lines, km, sections, closed, station ids, whether foot.json and
  ways.json are byte-identical); it writes nothing in the project. Run it with the change in
  the working tree, before rebuilding, only on the countries the change can touch: a change
  gated on one country (`if region == "hr"`, a COUNTRY key nobody else sets) needs no run on
  the others. A generic change that moves another country's output gets scoped or declined;
  record a declined one with its numbers in `HISTORY.md`.
- **Batch** shared changes into one rebuild.
- **After any shared-file change**, the real rebuild: `python tools/compare_lines.py save
  <cc...>`, `python tools/rebuild.py <cc...>` (model and tiles, 3 countries at once, `-j 1` for
  one, `--model-only`; logs in `data/logs/rebuild_<cc>_<step>.txt`), `python
  tools/compare_lines.py diff <cc...>`, then `python tools/build_regions.py`.
- **`tools/build_regions.py` after every rebuild, not only a first build**: since 2026-10-02 it
  also publishes `line_aliases` in regions.json, every country's twin merges (aliases.json
  `lines`) folded across countries, so a line one country merged and another kept apart still
  joins over the border (the Eurostar: fr kept "Eurostar: Paris - Amsterdam" m5189990 apart,
  be and nl merged it into m5189989, so Paris Nord was cut off at the border). It only folds
  into an OSM line, a piece of 5 km or more, with the same name (untrained) or same first
  word and ref, and prints what it folded and refused. The app folds every line id through it
  on load; two lines of one country folded together become one piece. Below z10, a click's
  "Also on this track" now comes from the footprints (foot.json), not station chords, and a
  long-chord line is measured on its drawn geometry.
- **After a country's first build**, rebuild its built neighbours too, so `split_at_borders`
  hands that country its side of each crossing.
- `check_model.py --region <cc>` after any model change: register lines stay near 1.00 of their
  published lengths, OSM-derived objects within 2%. `python -m unittest discover -s tests`.
- Several extracts at once: `OSMIUM_POOL_THREADS=2` each (extract.py defaults to 4).
  `rebuild.py` sets `OMP_NUM_THREADS=2`, so three countries stay within the ~6 cores builds may
  take; ru peaks near 3 GB, cn and fr near 2 GB.

**Running one country**: `python extract.py --region <cc> --pbf <file>`, `python
build_model.py --region <cc> --register <arg>` (every country's arg is in `tools/rebuild.py`'s
`REGISTER`; any other code is RINF, `rinf:data/raw/rinf/<cc>`), then `build_tiles.py`, which
reads line colours from the model, then `check_model.py`. After every extract, clip where the
reader clips: `hk_register.py --clip`, `sg_register.py --clip` (sg's extract takes `--bbox
103.6,1.2,104.05,1.452` from Malaysia's file; hk's comes from openstreetmap.fr),
`cn_register.py --clip`, `ru_register.py --clip`; Russia then `ru_register.py --convert`.
Each `<cc>_sources.md` has its commands, `HISTORY.md` the 2026-10-01 list with timings (cn,
ru, fr take 6-11 min). `.osm.pbf` extracts are deleted once extracted and checked;
`data/proc/<region>/` holds all a rebuild short of re-extracting needs (use Geofabrik's dated
`<country>-YYMMDD.osm.pbf` if `-latest` redirect-loops).

## Adding a country

The pipeline is region-agnostic; only the register reader and the check tables are not.

1. **OSM half.** The Geofabrik extract (a new download: Anita's say-so), `extract.py`, then
   `python inspect_region.py --region <cc>` before trusting any of it (are `usage` tags kept,
   how many route relations carry a colour, how much of the network a passenger route covers).
   Delete the `.pbf` afterwards; keep one at a time. `python probe_kr_ways.py --region <cc>`
   gives the share of main-line km whose `name` is its line (Korea 98%): if high, Korea's
   recipe works without a geometry register.
2. **The register** (spec §12c: FRA NARN for North America, ERA RINF for the EU, GTFS via the
   Mobility Database as the broad fallback, Wikidata as the glue). Either a module beside
   `n02.py` with `build(path, log) -> (lines, stations, geoms)`, run as `--register
   <module>:<path>` (`kr_register.py` is the template for named-track countries), or for RINF
   **one file `rinf_countries/<cc>.py`** defining `COUNTRY` (rinf.py's docstring lists the keys
   and hooks, each a no-op unless set; be.py, at.py, nl.py are worked examples), then `python
   rinf.py --fetch <cc>`, an extract and `--register rinf:data/raw/rinf/<cc>`.
3. **Check it.** Published line lengths in `REGISTER` in `check_model.py`, passing. A build
   with no outside number checked against it is not finished. A register with its own
   chainage (`km_official`) checks every line.
4. **Put it on the map**: `tools/build_regions.py` after the first build (managing session),
   then rebuild the built neighbours. The app needs no change; a country missing from
   `regions.json` only loads when someone has rides there. Outlines come from religiondots'
   `country_shapes.geojson`, else Natural Earth 1:10m (Luxembourg).
5. **Colours.** `colours/<cc>.csv` (`line,operator,colour,source,url,note`), applied by
   `line_colours.py` over OSM's and Wikidata's. Official sources first (route-map PDFs: read
   the vector fills with PyMuPDF, not pixels); a widely used map where the operator publishes
   none (Korea); mark our own choices `picked`. `python line_colours.py --fetch <cc>` after
   adding it to `COUNTRY` there gives the Wikidata fill; an operator's corporate colour goes in
   `GENERIC`. So far jp, kr, tw, hk, sg have a CSV and jp, kr, ch a Wikidata fill.
6. **Shared hooks**, via the managing session: a `looks_like_service` branch (a single
   long-distance or international train is a named train; an interval product a rider uses as
   a line stays a line), `norm_line_name` prefixes, a `rinf.py` hook. After a first build, read
   the `drop_islands` line of the `build_tiles` log (ways left out, by kind): a real line
   missing from the model shows up there as vanished track.

### What `build()` returns

```python
lines = [{"id": str,       # stable across builds; hash the name and operator, never an index
  "src": "n02",            # any non-"osm" value marks a register line, which is what counts
  "service": False,        # True only for a named train rather than a line
  "name": str, "name_en": str, "ref": str, "colour": str,
  "operator": str, "operator_en": str, "network": str,
  "kind": str,             # rail | subway | light_rail | tram | monorail | funicular | narrow_gauge
  "km": float,             # the sum of its sections
  "variants": int, "straight_sections": int,
  "display": [station_id, ...],                   # reading order for the strip diagram
  "sections": [[station_id, station_id, km], ...]}]   # build_model adds a 4th element
stations = {id: {"id", "name", "name_en", "lon", "lat", "lines": set()}}  # + "junction": True
geoms = {line_id: {"<from>|<to>": [[lon, lat], ...]}}                    # one per section
```

Optional on a line: `"highspeed": bool` only if the register knows (it then only matches OSM
ways with the same `highspeed=yes`, keeping the Shinkansen and the conventional line apart), or
per section `"highspeed_sections": {"a|b": bool}`; `"guided": True` for a guideway (never
shares track with trams); `"km_official"` and `"chain": {"a|b": km}`, the register's own
chainage, which `check_model` compares every line against. `"junction": True` marks a section
end that is not a stop: such a section is kept only if OSM passenger routes run over at least
half of it (`drop_unridden_sections`) or, in a GTFS country, trains do; that keeps base tunnels
and drops freight curves. A section between two stops is never questioned. Then
`merge_sources` matches OSM onto the register, `register_way_lines` ties register lines to OSM
ways (and corrects their kind), `ownership.py` writes `foot.json`, and `not_running.py` marks
sections with no drawn track `closed` (its `SOURCES`; RINF countries use `suspended` and the
GTFS check). `HISTORY.md` "The reader contract's notes" has more.

## Working on the app

`dist/index.html` is one file: constants, map setup, data loading, rides, crediting, then the
panel. Things that will bite:

- **Paint expressions must stay plain literal `const`s in SCREAMING_CASE.** `node
  tools/lint_map_expressions.js dist/index.html` evaluates each `addLayer` with those hoisted;
  run it on every edit. It catches an invalid expression that would silently drop a whole
  layer, and a top-level name declared twice (a leftover second `function goRegion` won
  silently and the page reloaded forever). `['*', ['interpolate', ['zoom'], ...], factor]` is
  invalid: `['zoom']` may only be the direct input of a top-level `interpolate` or `step`.
- Added layers anchor at the first symbol layer **after the last non-symbol layer** (the first
  symbol layer here is a water label before the roads). The basemap's own near-black railways
  are hidden on load. Station bubbles come from the model, never from the tiles.
- **No CSS transitions anywhere**: state changes are instant.
- **The native name is the real name.** Someone riding trains in a country reads the names on
  the trains, and those are not in English. `lineName()` and `stName()` show an English name
  where one exists and fall back to the native one, and that fallback is correct: never
  substitute an id, a transliteration or a placeholder. 61% of register lines have no English
  name and the app has to read well anyway.
- **Ridden and muted states DARKEN line colours inside the expression**, never lower opacity:
  double and quadruple track are stacked ways whose opacity adds up to nearly full strength.
- **A new colour goes in both themes** (the two `:root` blocks, and `THEMES` for map paint and
  the strip's SVG). Layers are added with the dark theme's literals so the lint checks them.
- **To see a change**: `python serve.py`, then `node tools/screenshot.js
  http://localhost:8767/ out.png 15000 probe.js` with Chrome headless on its own
  `--user-data-dir` (exact invocation in `tools/screenshot.js`'s header). **Use a private debug
  port** (`CDP_PORT=9341`): 9222 is shared by every session, and a second Chrome on it silently
  lands on someone else's browser. A probe file can drive the map and assert against it, which
  is how every UI change here has been checked; wait on the map's `idle` with a timeout, or a
  probe can hang. Stop only that Chrome.

How it is put together (countries loading as the map pans, per-country sources and layers,
track clicks, the strip diagram's layout): `HISTORY.md`, "The app as of 2026-10-02".

## Data files, what is gitignored, publishing

`dist/data/<region>/` holds `lines.json`, `stations.json`, `foot.json` (track ownership: what
riding each section credits, [owner section, from, to, a, b]), `ways.json` (lines per drawn
way, owner first), `aliases.json` and `geom/<line>.json`; `dist/data/<region>.pmtiles` is the
tiles. All generated and excluded by the repo's `data/` rule, as are `data/raw/`, `data/proc/`
and `data/logs/`. Tracked: source, `dist/index.html`, `dist/regions.json`, `border_points.json`.

**`aliases.json` matters more than it looks.** Station ids move when a build merges OSM onto a
register, and saved trips name stations by id. The file ships the moves and the app migrates
saved rides on load; `carry_aliases` maps every id the previous build shipped (old alias, same
name within 500 m, nearest within 200 m) and logs what had nothing in reach. A change that
moves ids without updating it silently voids people's history. Its `lines` map does the same
for line ids dropped as a register twin.

**Publishing** (to live at anita.garden/noritetsu) must upload foot.json and remove any old
credits.json; every country needs a foot.json before index.html goes up (without one the app
counts that country's OSM lines whole). The .pmtiles need a server that answers HTTP range
requests (why `serve.py`, not `python -m http.server`).

## Pointers

- `spec.md`: the design; §12c the registers per country, §13 the todo. `README.md`: how to run.
- `HISTORY.md`: how things got this way, declined proposals, the traps, the session logs.
- `<cc>_sources.md` (the Baltics share `baltics_sources.md`): sources, commands, what is off.
  `multi_sources.md`: sources covering many countries. `gtfs_sources.md`: the timetable feeds,
  and "Rollout (2026-10-01)" with every closed line per country.
- `border_proposal/PROPOSAL.md`: the cross-border design, with an "As built" section.
  `ownership_prototype/WRITEUP.md`: track ownership's prototype and comparison.
- Docstrings: `rinf.py` (COUNTRY keys and hooks), `gtfs_served.py` (the check and its fixes),
  `ownership.py` (who owns a way), `n02.py` and `kr_register.py` (traps that cost a rebuild),
  `tools/rebuild.py`, `tools/ab.py`.
