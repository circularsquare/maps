# Handoff

## Start here (managing session handoff, 2026-10-01, second session of the day: maps-ee)

**27 countries are built** and listed in `dist/regions.json`: jp, cn, kr, tw, hk, sg, ch, fr,
be, nl, at, cz, pl, hu, pt, si, sk, ro, bg, fi, lt, lv, ee, and since this session hr, gr, lu.
The sections below say how each was made; every one has a `<cc>_sources.md` (the Baltics share
`baltics_sources.md`) with its sources, run commands and what is still off.

**What maps-ee did (2026-10-01)**, five agents, the managing session holding the shared files:
- **Croatia** (`rinf_countries/hr.py`, 38 register lines, 2,317 km): RINF has no line numbers
  for hr but every section URI carries HŽI's ("SectionOfLine_M202__..."), read through `fix`.
  `SUSPENDED` from HŽI's 2025 statistics (passenger trains per section, table 3.7).
- **Greece** (`rinf_countries/gr.py`, 12 register lines, 1,718 km): OSE's standard gauge only;
  no public line numbers, so `id_name` returns (Greek, English). `fix` adds the new Othrys line
  RINF lacks. Needed: Greek fold in `rinf.norm()` (αυ/ευ as av/ev), `id_name` may return a
  tuple, `highspeed_sections: False` (one main line over old and new alignment).
- **Luxembourg** (`rinf_countries/lu.py`, 16 register lines, 237 km): needed `netref_wkt`
  (points' coordinates hang off their netReference). Not in religiondots' country shapes, so
  `tools/build_regions.py` now falls back to Natural Earth 1:10m for a missing outline.
- **GTFS timetable check live in 14 countries**: cz, hu, pt, pl, be, si, sk, ro, bg, fi, lt,
  lv, ee, hr. Switched off for at and nl (register too fragmented / nothing to gain); feeds
  for lu and gr found but not rolled out (reading them was blocked by the permission
  classifier; Anita's call). `gtfs_sources.md` "Rollout (2026-10-01)" lists every closed
  line per country. Seven fixes to gtfs_served.py, in its docstring.
- **Finland**: Porvoo museum line back (fi.py had 131/132 swapped); drawn greyed, no trains
  in VR's feed.
- **build_model.py, generic, measured on every country before landing** (old/new builds into
  scratch, final outputs compared): (1) a station invented from a bare stop node joins
  `by_name`, so its same-named twin on the other track resolves onto it. Only true twins
  merged: Shanghai Metro lines were up to 27% long from doubled stops and now match published
  lengths, ch line 151 counted Pougny-Chancy - Russin twice, Lisbon/Porto metro interchanges
  became one station each; station ids moved are carried by `carry_aliases`. (2) In
  `merge_osm_twins`, a name with a two-way mark (↔, <=>, ⇄) keeps it where stripping would
  give different lines one name: Poland's ~80 lines called "R", five "Elron", nine "liO TER
  Occitanie", three Narita Express. (3) `looks_like_service` has an `hr` branch.
- **Cross-border track: designed and prototyped, NOT landed.** `border_proposal/PROPOSAL.md`
  with `borders.py`, `border_points.json` and `build_model.diff`. Each country builds its side
  to the RINF border point both neighbours name alike (EU00084 Mouscron-Frontière), so the
  app's existing join connects them. On be/nl/fr: K80 Kortrijk - Lille gets a path, every
  register line identical, uncreditable border track 273 -> 19.5 km. Waits on Anita's four
  decisions in its section 7. The diff predates (1) and (2) above; rebase it.
- Every country rebuilt at the end on the shared code above.

**How the work was run**: one managing session holding the shared files (`build_model.py`,
`build_tiles.py`, `extract.py`, `rinf.py`, `line_colours.py`, `not_running.py`,
`gtfs_served.py`'s hook, `tools/build_regions.py`), and up to eight background agents at once,
each owning one country's files and sending exact diffs for shared changes. The managing
session measured every shared change on all built countries before landing it (a generic
change that moves another country's output gets scoped or declined; the declined ones and
why are recorded below), and ran `tools/build_regions.py` when a country was final. Keep
doing it that way. Anita runs builds freely in noritetsu unless one would take over ~2 hours.

**Open threads, in the order I would take them:**
1. **Cross-border track**: land `border_proposal/` once Anita has answered its section 7
   (rebase the diff first; roll out with compare_lines save / rebuild / diff on every country,
   cz and hu first since the prototype never ran the GTFS hook).
2. **Waiting on Anita** (asked 2026-10-01): seasonal lines (Poland's 96 Muszyna - Leluchów
   reads closed because the feed window misses summer; Croatia's Metković - Ploče is drawn);
   whether lu and gr get the GTFS check; Croatia's single "B" fast trains as named trains.
   GTFS spot-checks: pl 281 Nakło - Chojnice (may have restarted 2026), ro 600 and 700.
   Works closures (pl Opole - Nysa to 24 Oct, Jelenia Góra to 30 Oct, Cieszyn to 13 Dec) stay
   greyed until the feed is refetched and the country rebuilt.
3. **Russia**: `ru_sources.md`. An open official register exists (the tariff guide, sovetgt.org
   tr4 XLS, every section with ordered stations and km); recipe is a converter into rinf.py's
   input. Anita decided Crimea and the 2022-annexed Donetsk/Luhansk/Melitopol-Kherson railways
   (2,722 km the tariff guide files under Russian administration) go with Russia, de facto;
   neither Geofabrik's Russia extract nor religiondots' outline includes the annexed ones, so
   they need their own extract and outline handling. **Still open**: whether the line unit
   is the tariff section (1,117 lines, ~79 km each) or grouped corridors.
4. **More EU**: Germany (big; RINF doubles every section across 2026/2027 versions), Italy,
   Spain, Sweden, which need name maps. Croatia, Greece and Luxembourg are done.
5. **USA**: register lines are FRA NARN subdivisions, Amtrak routes services over them (Anita,
   tentatively). The app may need a country's lines.json split by region at that size.
6. The known general issues listed under "Europe from ERA RINF" below (line_hash clash,
   display order of split lines, lines with only an infra relation: also Greece's Pelion and
   Zagreb's funicular). The Plovdiv yard end is fixed by the GTFS check. Greece found one
   more: Athens Metro M3 runs on OSE track to the Airport, but subway and rail never credit
   each other, so it credits nothing of the register line there.

**Anita's decisions, 2026-10-01, recorded, not yet acted on:**
- **Completion: everything with scheduled passenger service counts**, "if it is scheduled at
  all (more often than about once a week)". This reverses the earlier rule that named trains
  and long-distance services (`service: true`) are lookup-and-input only and count towards
  nothing. Before changing the app, confirm with her what it means for a named train: count
  its own km as a line of its own, or only make sure every track it runs over counts.
- **Russia: the 2022-annexed railways (Donetsk, Luhansk, Melitopol–Kherson) go with Russia**,
  de facto, as Crimea does. Still open: the line unit (tariff sections or corridors).
- **Finland: keep the Porvoo museum line**: done by maps-ee (drawn greyed: no trains in VR's
  feed).
- **Czechia, JHMD (228/229 out of Jindřichův Hradec)**: unknown whether it runs; leave as is.
- **Cross-border track is not drawn**: a route that crosses a border (TER K80 Kortrijk –
  Lille) shows Mouscron and Tourcoing but no line between them. She wants it drawn. Designed
  and prototyped by maps-ee: `border_proposal/` (open thread 1).

**After any shared-file change**: `python tools/compare_lines.py save <cc...>`, then
`python tools/rebuild.py <cc...>` (model and tiles per country, logs in data/logs/), then
`python tools/compare_lines.py diff <cc...>`, then `python tools/build_regions.py`.

Written 2026-09-30. `spec.md` is the design and §13 is the todo; `README.md` is how to run
things. This file is for picking the project up cold, and for **two people working on it at
once without colliding**.

## Where it is

Japan is built end to end and the tracker works. 1,101 lines of which 593 are register lines
(28,156 km), 9,072 stations, a 10.5 MB tile archive, and rides recorded from a strip diagram or
from the map with percentages per line, operator and overall.

**Switzerland is built too** (2026-09-30): 779 lines of which 402 are register lines
(5,567 km), 2,816 stations, a 3.0 MB tile archive. The app loads it when the map moves
over Switzerland (`?region=ch` opens there), as it does every built country listed in
`dist/regions.json`.

**South Korea is built too** (2026-09-30): 130 lines of which 83 are register lines
(4,897 km), 1,215 stations, a 1.8 MB tile archive, in the app at `?region=kr`.

**And Taiwan** (2026-09-30): 55 lines of which 36 are register lines (1,792 km), 519 stations,
a 0.6 MB tile archive, at `?region=tw`. `tw_register.py` follows Korea's pattern (OSM
named track as geometry, importing kr_register's graph helpers); stations from TRA's own
open data (cut into the 16 legal lines by `LEGAL`), OSM route stops for THSR and the
metros, zh.wikipedia for Alishan. Every TRA line within 1% of its published length; sources
and faults in `tw_sources.md`. Two shared changes came with it: `build_tiles.rank_of` draws
usage=tourism narrow gauge (Alishan, and Switzerland's mountain railways), and
`build_model` merges named trains that differ only in train number (THSR maps every train
as its own relation) and strips Taiwan metro prefixes in `norm_line_name`. See "Korea"
below for how, and for what is still off.

**Hong Kong and Singapore** (2026-09-30, maps-d5 with one agent each), on Taiwan's pattern:
`hk_register.py` (13 register lines, 293 km: the ten MTR lines, Light Rail as one line, the
trams and the Peak Tram; station lists from MTR's DATA.GOV.HK files) and `sg_register.py`
(10 register lines, 269 km; station lists from LTA DataMall's station-code file, topped up from
LTA's 2026 system map). Every line within 9% of its published length, most within 2%; the
misses are explained in `hk_sources.md` and `sg_sources.md` (station-to-station against
end-of-track figures, the Peak Tram measured along its slope, Light Rail's one-way street
pairs). Both clip the extract to the country (`--bbox` on `extract.py`, then the reader's own
`--clip` to a polygon), since Geofabrik has no Hong Kong file (openstreetmap.fr does) and ships
Singapore inside Malaysia's.

**Europe from ERA RINF** (2026-09-30, maps-d5 with one agent): `rinf.py` is ONE reader for every
country in the EU's infrastructure register (national line numbers, sections with km, typed
operational points), with a small per-country table. Geometry is traced over OSM track between
consecutive points and rejected when it strays from RINF's section length; names come from
OSM's `route=railway`/`route=tracks` relations (`infra.pkl`), then Wikidata P1671, then first -
last station. Only the infrastructure manager's network is in RINF, so metros, trams and most
private railways stay OSM lines. **Belgium** 145 register lines, 3,196 km; **Austria** 125,
4,342 km (RINF also carries the Steiermärkische Landesbahnen, Montafonerbahn, Raaberbahn and
Neusiedler Seebahn; not GKB, Wiener Lokalbahnen, Stern & Hafferl, Zillertalbahn);
**Netherlands** 96, 2,780 km (ProRail's ids are two station codes, "Asd-Rtd", read back into
"Amsterdam - Rotterdam"). All three median 0.996-1.000 against RINF's own section lengths.
`be_sources.md`, `at_sources.md`, `nl_sources.md`. **A new RINF country is a new file,
`rinf_countries/<cc>.py`** defining `COUNTRY` (rinf.py's docstring says which keys; be.py,
at.py and nl.py are the worked examples), then `python rinf.py --fetch <cc>`, an extract, a
build and a `REGISTER` table. The per-country settings are one file each so several agents can
add RINF countries at once; `rinf.py` itself is shared code, changed only by whoever holds
the shared files. When several extracts run at once, give each
`OSMIUM_POOL_THREADS=2` (extract.py defaults to 4). Which source serves
which country next, measured: `multi_sources.md`.

Four more RINF countries the same day, one agent each, in parallel: **Czechia** 236 register
lines, 9,101 km (lines by timetable number, "010 Kolín – Česká Třebová", read from OSM's
route=tracks relations, since no rule maps SŽ's own numbers onto them); **Poland** 352, 16,171 km
(PLK's line numbers; 29 of 30 checked lines within 2%); **Hungary** 133, 6,721 km (MÁV and
GYSEV; MÁV's RINF section lengths leave out station track, hence `tol_abs`); **Portugal** 24,
2,138 km. Each has `rinf_countries/<cc>.py` and `<cc>_sources.md`.

`rinf.py` grew per-country hooks for them, every one a no-op unless the country sets it, and
each checked by rebuilding be/nl (identical): `skip_line(id)` (ids that are never lines: private
sidings), `no_ref(id)` (ids that never take a line number: Czech siding leads, which otherwise
turned siding junctions into branch points), `osm_rel(tags)` (read a relation's number and name
from all its tags: Czech OSM puts two numberings on `ref`), `im_of(section)` (operator where
RINF's manager code does not tell them apart: GYSEV under MÁV's), `tol_abs`, and
`cut_at_junctions` (end sections where other lines meet: Portugal only, because as a default
it cost Belgium real track, L.12 Antwerp-Essen 32.5 -> 26.0 km). Two generic changes:
trolleybus items never name a line (Wikidata numbers Debrecen's trolleybuses like MÁV lines),
and an end-to-end retrace past a failed piece must lie at least half on the line's own
relation (`OWN_DIRECT`: Poland's disused 223 was drawn as 135 km over two other lines).

**Second EU round** (2026-10-01): Slovenia (22 register lines, 1,157 km), Bulgaria (30, 3,576),
Slovakia (70, 3,376; 19 lines Slovak sources list as without passenger service are greyed as
not running through the new `suspended` hook), with Romania, Finland and the Baltics. New
`rinf.py` hooks, each a no-op unless a country sets it: `name_m`, `stop_names`, `fix(secs,
points)` (a country's in-place corrections to RINF: Slovakia's stations typed 110, lengths in
metres, ids spanning several lines), `suspended`, and `osm_stops` (True: OSM stations an OSM
train route stops at become stops on the sections they lie on, for registers listing only
junction stations, Latvia and Estonia; "all": every OSM rail station, Bulgaria, whose RINF has
no halts). Generic: Cyrillic names are transliterated in `norm()` (they used to normalise to
nothing), and Lithuania's "POINT (+22.69 ...)" coordinates parse. `extract.py` keeps
railway=construction track a passenger route still runs over (Slovenia's line 50 at Preserje,
rebuilt under traffic), and takes `--station-areas` for countries that map stations only as
areas (Bulgaria).

Known general issues the EU agents found, not yet fixed (each worked round per country):
- `rinf.line_hash(cc, ref or "|".join(lids))` gives an UNNUMBERED id the same line id as a
  public number with the same digits, so one silently replaces the other (Romania: CFR SA's
  unnumbered 307 took timetable line 307's id and Blaj - Praid vanished). Fixing it changes
  the ids of unnumbered lines, so it needs line aliases written for saved rides.
- `n02.walk_order` (the `display` order) walks only the first connected piece, so a line in
  two pieces lists only one piece's stations there (Bulgaria's 3 and 4). The app's strip
  diagram lays out the track graph itself, so this shows only where `display` is read.
- A line in no register and with no OSM route relation, only a route=railway one, is not a
  line at all and build_tiles drops its track (Bulgaria's Septemvri - Dobrinishte narrow
  gauge, 125 km). Building lines from infra relations where no route exists would fix it.
- A register line ending at a freight yard with no OSM station and no OSM route over the
  last stretch loses it (Bulgaria's line 8, Trakia - Plovdiv Razpredelitelna, 5.5 km, which
  every Plovdiv - Burgas train uses). The GTFS check should rescue this kind.

**The app** (2026-10-01): a line id shared between countries (an OSM route crossing a border,
built in each country under one id) is merged into one line across countries, so the Eurostar
lights in nl, be and fr; `regions.json` carries `shared_lines` so the app knows which countries
to load. The camera, mode and open panel persist in localStorage (`noritetsu.camera`,
`noritetsu.mode`, `noritetsu.ui`); a first load shows the world.

**Timetable feeds** (`gtfs_sources.md`, 2026-10-01): open GTFS exists for nearly every built
and planned country except China and Russia, mostly via the Transitous mirror. Tested on
Czechia (stops join by code) and Hungary: it rescues the track the OSM-route rule drops and
confirms lines with no trains. `gtfs_served.py` is being built on cz and hu first.

**Mainland China** (`cn_register.py`, 2026-09-30): Korea's recipe at scale. OSM's way names
are China Railway's own line names (97.9% of main-line track km named), so named track is the
register, with 京沪线 and 京沪高铁 as separate lines and the national code (0002, 3002) as ref
from OSM's route=railway relations. Which stations are passenger stops comes from 12306's
public station list (3,350 of 3,404 found in OSM), placed on a line within 300 m of its track.
**417 register lines, 121,894 km** (142 mostly high-speed, 48,368 km); 26 of 30 checked lines
within 3%. `cn_register.py --clip` cuts Hong Kong and everything abroad out after each
extract; Macau stays in cn. Freight: a line with 12306 stations on it is never dropped as
freight whatever OSM's traffic_mode says (胶济线 is 65% freight-tagged and a busy passenger
line), except 浩吉线 and 广珠线, left out by name. cn.pmtiles is 36.6 MB, 3.5x Japan's;
lines.json 1.2 MB, the same as Japan's. `cn_sources.md` has the rest.

**France from SNCF Réseau's own register** (`fr_register.py`, 2026-09-30): the RFN lines by
their 6-digit code with PK chainage, stations with a PK per line, shaped like Switzerland's.
Whole country built: **278 register lines, 24,227 km**, chainage median 0.996 over 264 lines;
Paris-Marseille, Paris-Brest and Paris-Lille all 1.00 of their published lengths. Where
several Wikidata items share an RFN code, the longest names the line (750 000 is Moret -
Lyon-Perrache, not the historic 58 km Saint-Étienne - Lyon piece that shares its code).

The `.osm.pbf` extracts are deleted once extracted and checked; `data/proc/<region>/` holds
what was pulled out of them, which is all a rebuild short of re-extracting needs. Fetch the
extract again from Geofabrik to re-extract (the plain `-latest` URL for Switzerland was
redirect-looping on 2026-09-30; the dated `switzerland-YYMMDD.osm.pbf` worked).

```powershell
python serve.py                                                  # http://localhost:8767
python extract.py --region jp --pbf data/raw/japan-260928.osm.pbf
python build_model.py --region jp --register n02:data/raw/N02-24_GML.zip    # 90 s
python build_tiles.py --region jp        # AFTER build_model: it reads line colours from it
python check_model.py --region jp

python extract.py --region ch --pbf data/raw/switzerland-260929.osm.pbf      # 1 min
python build_model.py --region ch --register schienennetz:data/raw/schienennetz_2056_de.gdb.zip   # 50 s
python build_tiles.py --region ch                                            # 30 s
python check_model.py --region ch

python extract.py --region kr --pbf data/raw/south-korea-260929.osm.pbf     # 25 s
python build_model.py --region kr --register kr_register:data/raw/kr         # 30 s
python build_tiles.py --region kr                                            # 15 s
python check_model.py --region kr

curl -L -o data/raw/hk-260929.osm.pbf https://download.openstreetmap.fr/extracts/asia/china/hong_kong.osm.pbf
python extract.py --region hk --pbf data/raw/hk-260929.osm.pbf              # 5 s
python hk_register.py --clip                                                 # after every extract
python build_model.py --region hk --register hk_register:data/raw/hk         # 3 s

python extract.py --region sg --pbf data/raw/sg/malaysia-singapore-brunei-260929.osm.pbf --bbox 103.6,1.2,104.05,1.452
python sg_register.py --clip                                                 # after every extract
python build_model.py --region sg --register sg_register:data/raw/sg         # 2 s

python build_model.py --region be --register rinf:data/raw/rinf/be           # 45 s
python build_model.py --region at --register rinf:data/raw/rinf/at           # 70 s
python build_model.py --region nl --register rinf:data/raw/rinf/nl           # 40 s
python build_model.py --region cz --register rinf:data/raw/rinf/cz           # 75 s
python build_model.py --region pl --register rinf:data/raw/rinf/pl           # ~3 min
python build_model.py --region hu --register rinf:data/raw/rinf/hu           # 35 s
python build_model.py --region pt --register rinf:data/raw/rinf/pt
python cn_register.py --clip                                                 # after every cn extract
python build_model.py --region cn --register cn_register:data/raw/cn         # 6 min, tiles 5 min
python extract.py --region fr --pbf data/raw/fr/france-260929.osm.pbf        # 6.5 min, 5.1 GB
python build_model.py --region fr --register fr_register:data/raw/fr         # 6 min
python build_tiles.py --region fr                                            # 4 min

python -m unittest discover -s tests
node tools/lint_map_expressions.js dist/index.html
```

The Swiss register needs two files in `data/raw/`, both open data, URLs in the docstring of
`schienennetz.py`: the BAV network (`schienennetz_2056_de.gdb.zip`, 3.4 MB) and the national
service-point list (`ch_servicepoints.csv`, 25 MB), which is what says which network nodes are
passenger stops.

## Korea (built 2026-09-30)

**OSM's named track is the geometry register.** 98% of Korea's main and branch track km in
OSM carries the name of its legal line (경부선, 분당선, 2호선; `python probe_kr_ways.py`),
which is the same shape as N02: track labelled with the register line. So `kr_register.py`
is `n02.py` over OSM ways, with no geometry file of its own and no change to the reader
contract. The legal line is the register unit; 1호선, 경의·중앙선, 수인·분당선 and the KTX
services stay OSM operating patterns and named trains over it, as 京浜東北線 does in Japan.

**Which stations are on which line comes from published lists**, read by `kr_sources.py`
(all open, no login; where each came from is in `kr_sources.md`):

- Korail's 각 선구별 거리표 (data.go.kr 15137040): every Korail legal line, intercity and
  commuter, stations in order and km per section. Plus SR's own 수서 matrix.
- KRIC 1294, the national urban-rail station table: every metro and light-rail operator,
  in order, with km, coordinates and English names. Its Korail rows are scrambled; unused.
- English names: RAFIS (data.go.kr 15132601) for Korail, KRIC 1294 for everyone else.

A listed station is found among OSM's stations by name, nearest the line's own track; an
OSM station whose stop node lies on the track joins too. Proximity alone decides only for
a station NO list names, with no stop node on any named track, tagged served by the line's
mode (`train=yes` for heavy rail), and whose nearest named track is within 80 m: that is how
the 2024-25 openings got on (동해선 영덕-삼척, 서해선 홍성-서화성, 중부내륙선 to 문경,
목포보성선, 군위 on 중앙선's new alignment). The build log lists every one; review it after
a new extract. 천안아산 is 100 m from 아산 and listed, so it never reaches that rule.

**High-speed is per section** (`highspeed_sections`, read by `build_model.section_highspeed`),
from the `highspeed=yes` ways each section lies on. A line-wide flag either stopped KTX-이음
rides crediting 중앙선, 경강선 and 서해선 (partly new 250 km/h alignments), or let KTX on
경부고속선 credit 경부선 beside it into 대전 and 대구.

**Colours**: `colours/kr.csv`, sampled from the widely circulated Korea network map Anita
supplied (Korail publishes no intercity line colours), plus seven `picked` for lines that map
leaves out. Applied by `line_colours.py`, which then fills anything left from Wikidata.

**Checked**: `check_model.py --region kr` has 66 published lengths (2023 철도통계연보
영업거리, KRIC's line table, and the lists' own sums; `REGISTER["kr"]` names each). The
register's own km covers every section of 65 lines, median 0.997. Every metro and nearly
every Korail line lands within 3% of its published length.

**Three things that cost a rebuild, all in `kr_register.py`'s docstrings:**

- **A station must cut every track through it.** OSM maps a Korail station as one node
  beside the track, and a straight high-speed through track has no vertex near the
  platform, so the search slid past 김천(구미) and paired 대전 with 동대구: 경부고속선 at
  1.73. Line graphs are densified to a vertex every 40 m, and a station claims every vertex
  within 150 m (further for a node mapped in its concourse).
- **Start and end a section anywhere near its stations, charged from the centre.** Starting
  at one anchor sent a train the wrong way out of 군북 on a platform road and back (33 km for
  10); charging from the nearest anchor rather than the centre lost a platform length per
  metro section (서울 1호선 at 0.77). The drawn geometry then has to be pinned to the two
  centres as well, or a ridden run shows a gap at every station, like a dashed line.
- **Name variants of one station**: 구의 and 구의(광진구청), 서울 and 서울역, 경성대·부경대
  and 경성대ㆍ부경대, and 동해선's Busan-Ulsan track tagged 동해본선. See `name_key`,
  `base_key`, `NAME_ALIAS`, `STATION_ALIAS`.

**Still off, and why** (the `<--` rows of `check_model.py --region kr`):

- **A named track that stops short of the line's first station loses that section.**
  호남고속선 is named from the junction south of 오송, so 오송-공주 is missing (0.76);
  영동선 (0.86: 영주 and 강릉 ends), 대구선 (0.81), 광주선 (0.63) and 수서평택고속선's 지제
  end are the same. The fix is to trace a listed station that is on no track of its line's
  name over the wider network to that track, as build_model's `TrackGraph` does for
  relation gaps.
- **Newer than the published figure**: 동해선 to 삼척 (1.55), 중부내륙선 to 문경 (1.47),
  인천 1호선 to 검단 (1.09), and the 2024 중앙선 realignment. 서해선 reads 2.70 because OSM
  names the 2024 홍성-서화성 intercity line 서해선 too and the figure is the 대곡-원시
  commuter line alone. GTX-A's figure has only its northern half on its own track name; the
  수서-동탄 half is tagged 수서평택고속선 in OSM.
- 157 listed stations match no OSM station. Nearly all are closed to passengers (small
  호남선, 영동선 and old 중앙선 halts) or freight; the build log lists them.
- Stations have English names from RAFIS and KRIC; lines have none yet beyond what OSM's
  matched route relations hand over (18 did).

Smaller data items still open, all in spec §13: a per-line list of the 20 remaining
straight-line fallbacks for the app; Swiss sub-kilometre km-lines cluttering line lists; the
shorter-route-only rule when a km-line has two routes between stops.

## Between the two tracks

Open, for the app (2026-09-30): a headless load of `?region=kr` once opened on Japan
(FOCUS 'jp') where earlier loads of the same URL opened on Korea, and `goRegion('kr')` from a
probe works. Looks like a race between `regions.json` arriving and the START check.

`looksStraight` in `index.html` guesses a straight-line fallback from geometry (over 2 km,
within 1% of crow-flies). Since 2026-09-30 it only guesses on lines whose `straight_sections`
is over 0 (12 in Japan, no register line), because elsewhere it was taking real straight
track out of strip diagrams.

`build_tiles.py` leaves out any connected piece of track that no line runs over
(`drop_islands`): tourist and amusement-park rides, harbour freight lines.
Its log line says how many ways by kind; the list is worth a look after a new country's first
build, since a real line missing from the model would show up there as vanished track.

`not_running.py` (called from `build_model.main`) marks register sections with no drawn
track beside them as `closed`; the app greys them and leaves them out of completion. It runs
for the register sources in its `SOURCES`.

**Themes.** `index.html` has a dark and a light theme. Colours in the page are the CSS
variables in the two `:root` blocks; map paint and the strip's SVG colours come from
`THEMES`. Layers are added with the dark theme's literals (so the expression lint checks
them) and `applyTheme` sets the rest; switching theme swaps the basemap with `setStyle` and
`carryOver` moves this page's sources and layers across. A new colour goes in both.
Each country has a `track-edge-<cc>` layer under `track-<cc>`: the edge for lines whose
brightness (`LUM_*`, computed in the expression from the line colour) is too close to the
theme's background, with the track drawn at 0.65 of its width over it.

Settled 2026-09-30, for the record:

- **The app reads**: `aliases.json`'s `lines` (rides on an OSM line dropped as its register
  twin move to the register line), any `src` but `osm` as a register, `j` stations as
  diagram rows that are not bubbles, search results or stops, and `dist/regions.json` for
  the countries (`python tools/build_regions.py` after a new one's first build; the app then
  loads it when the map pans over it, with no app change). A track click opens the register
  line when it is the only one there. Verified headless: a ride saved on the old OSM Tozai
  line id migrated with none lost, and a San'in click opens 山陰線 with スーパーおき under it.
- **Heavy rail now credits across the two sources.** `build_credits` compared raw kinds, and
  OSM train routes are `train` where register lines are `rail`, so no OSM train route had
  ever credited a register line: riding the Yamanote as operated counted nothing towards
  山手線. It now compares `kind_family`. That let every Shinkansen service credit the
  conventional line beside it, about 1,600 km of false credit over 16 services, so
  high-speed and conventional sections no longer credit each other
  (`section_highspeed`: OSM sections by the `highspeed=yes` ways they lie on, register
  sections by their line's `highspeed` flag). What is left, 108 km on the Ou Line, is the
  Tsubasa and Komachi really running on it. IC 2 Zürich to Lugano now completes the
  Gotthard and Ceneri base tunnels.
- **Tiles carry `c`**, the colour of the line on each piece of track: the register line's,
  else the commonest among OSM lines, lines before named trains. **So `build_tiles.py` now
  runs after `build_model.py`.** The unused `station` tile layer is gone.
- **Fewer duplicate lines.** A name match with the register now stands when the operators
  are spelled differently but most stations are shared (OSM's "Tokyo Metro" against the
  register's 東京地下鉄 kept the Hanzomon Line listed twice). OSM lines that duplicate each
  other merge too (`merge_osm_twins`): the same through service mapped twice, and direction
  pairs never put under a route_master. 40 in Japan, 5 in Switzerland. OSM line names lose
  their direction part, "(Shibuya -> Chuo-Rinkan)", unless that would give two different
  lines one English name, in which case the one that lost its direction shows its native
  name. Station names are compared through `station_key` (NFKC, the three small-ke
  spellings folded: 市ケ谷 = 市ヶ谷) and, for CJK names close together, one ending or
  starting with the other (新線新宿 = 新宿); without both, the Toei Shinjuku Line stayed
  listed twice. Every dropped id is in `aliases.json` `lines`, so the app's migration
  covers it.
- **Straight-line sections are traced along track** (`TrackGraph` in `build_model`): 471 in
  Japan down to 20, 14 of them named trains; 10 left in Switzerland. JR中央線快速 Tokyo to
  Shinjuku is 10.1 km of track, where it was a 6.1 km chord.

`check_model.py` is the one that matters: it compares built line lengths against published
figures. Register lines should stay near 1.00 (JR Central's 東海道線 is 1.06 because the
figure leaves out its 美濃赤坂 branch and 垂井 bypass). OSM-derived objects should stay within
2%. JR trunk lines used to run 5-20% long; that was `n02.py` letting sections slip past
stations on multi-track lines, fixed 2026-09-30 (station footprints and junction U-turns, in
its docstrings).

## Two tracks, and who owns what

**Do not both edit `build_model.py`.** It is the only file both tracks have reason to touch.

| Track | Owns | Must not touch |
|---|---|---|
| **Countries and data** | `n02.py` and new per-region readers (`kr_*`, `tw_*`...), `extract.py`, `build_tiles.py`, `check_model.py`, `inspect_region.py`, `probe_*.py`, `line_colours.py`, `colours/`, `data/` | `dist/index.html` |
| **Styling and app** | `dist/index.html`, `tools/` | the Python build |

`build_model.py` belongs to the countries track. If the app track needs a new field in the
output, ask for it rather than adding it.

## Adding a country

The pipeline is region-agnostic; only the register reader and the check tables are not.

1. **OSM half.** Get the Geofabrik extract, then
   `python extract.py --region <cc> --pbf <file>` and
   `python inspect_region.py --region <cc>` before trusting any of it. That report is how you
   find out whether `usage` tags are kept in that country, how many route relations carry a
   colour, and how much of the network a passenger route relation actually covers. Delete the
   `.pbf` afterwards; keep one at a time.
2. **The register.** §12c lists where each country's authoritative line inventory comes from:
   FRA NARN for North America, ERA RINF for the EU, GTFS via the Mobility Database as the
   broad fallback, Wikidata as the glue. Write a module beside `n02.py` exposing:

   ```python
   def build(path, log) -> (lines, stations, geoms)
   ```

   and run it with `--register <module>:<path>`. Usually nothing else in the build needs
   editing; the two small exceptions so far are listed under "Next countries" below.
3. **Check it.** Add the country's published line lengths to `REGISTER` in `check_model.py`
   and make them pass. A build with no outside number checked against it is not finished.
4. **Put it on the map.** `python tools/build_regions.py` once, after the first build. The
   app loads each country as the map moves over it, and `dist/regions.json` is how it knows
   where each one is before loading any of it; a country missing from it only loads when
   someone has rides there. The outline comes from religiondots' `country_shapes.geojson`.
5. **Colours.** `colours/<cc>.csv` (`line,operator,colour,source,url,note`) for the
   operators' own line colours, applied by `line_colours.py` over OSM's and Wikidata's.
   Official sources first (route-map PDFs: read the vector fills with PyMuPDF, not pixels);
   a widely used map is acceptable where the operator publishes none (Korea); mark anything
   chosen by us `picked`. `python line_colours.py --fetch <cc>` after adding the country to
   `COUNTRY` there gives the Wikidata fill; put an operator's corporate colour in `GENERIC`
   so it is skipped, or every line of that operator comes out the same.

### Next countries (written 2026-09-30, for the next session)

**The recipe that has now worked twice (Korea, Taiwan) and needs no geometry register:**
OSM track named for its legal line gives the geometry, a published station list per line
says which stations are on it. `kr_register.py` is the template; `tw_register.py` imports its
graph helpers (`line_graph`, `Near`, `neighbours`, `between`) rather than copying them. First
thing to measure in a new country: `python probe_kr_ways.py --region <cc>` (works for any
region), the share of main-line km whose `name` is its line. Korea 98%, Taiwan high; where
it is low, the country needs a geometry register (N02, Schienennetz) or OSM route relations.

**Small shared-code hooks each country has needed**, all in `build_model.py`:
`looks_like_service` (what makes an OSM route a named train: a train brand in Korea, a train
number in Taiwan), `norm_line_name` prefixes (system names like 台北捷運 that OSM puts before
the register's line name). Nothing else in the shared build has needed a per-country branch.

Added for Hong Kong and Singapore (2026-09-30), none of them per-country branches, each
measured on every built country before it landed:

- `norm_line_name` keeps the Han half of a bilingual name ("港鐵東鐵綫 MTR East Rail Line"),
  strips `港鐵` and `MRT `, turns "LRT X Line" into "X LRT", and folds hyphens and dashes to
  spaces. On jp it newly matches six JR Shikoku relations to their register lines
  (土讃線 Dosan), and nothing else anywhere.
- `build_stations` counts a `public_transport=station` only when a tag says rail (Hong Kong
  maps bus termini that way), and a non-tram stop never lands on a tram stop by proximity.
  15 stop nodes in jp and 16 in ch resolve differently, all onto the right station.
- **`carry_aliases`**: aliases.json used to hold only one build's own merges, so a station id
  that stopped existing between builds silently dropped any saved ride naming it. Every id the
  previous build shipped is now mapped on (old alias, then same name within 500 m, then nearest
  within 200 m), and the log says how many were carried and how many had nothing in reach.
  **The first rebuild of jp and ch with the changes above will carry a few** (the Tenjin
  bus-terminal record, "Rüti, Bahnhof", Beatenberg); check that line of the log.
- Declined, with numbers, so nobody re-proposes them blind: skipping the proximity pass for
  named stop nodes (moves 229 in ch, 99 in jp, 35 in kr, nearly all correct spelling-variant
  matches like Genève-Cornavin onto Genève), and a mode-tag clash test (mixed: loses
  市ヶ谷 onto 市ケ谷). Hong Kong's 機場快綫 stays listed twice as a result.

- `build_credits`: **tram and light rail credit each other, but only on the same rails**
  (within `SAME_RAILS_M`, 8 m) and never for a register line marked `"guided": True` (n02
  sets it on N02's guideway codes). France's T11 is route=tram on light_rail track, Hiroden 2
  runs on to the light_rail 宮島線, the Forchbahn runs on Zürich tram track. At 45 m, or
  without the flag, the Astram and Nippori-Toneri guideways credited tramways they cross or
  run beneath. Measured: +174 pairs in jp (with the flag), +42 in ch, none lost.
- `looks_like_service` has an `fr` branch: TGV, Ouigo, Intercités, Eurostar and the like are
  named trains; TER, Transilien and RER are lines. Likewise `pl` (PKP Intercity), `hu` (IC,
  EC, EN...), `pt` (Alfa Pendular, Intercidades, Celta), `cn` (train numbers, G1, K27/28), and
  one shared `EU_TRAIN` rule for at/be/nl/ch/cz (EC, EN, ICE, Nightjet, Eurostar, European
  Sleeper). **The line drawn everywhere in Europe**: a single long-distance or international
  train is a named train; an interval product a rider uses as a line (Swiss IC 1, ÖBB's
  Railjet, Dutch IC, Hungarian IR) stays a line.
- `extract.py` writes `infra.pkl`: OSM's `route=railway`/`route=tracks` relations, named
  track with the national line number in `ref`, kept apart from `rels.pkl` so no passenger-
  route reader sees them. `not_running.SOURCES` has `rfn` (France).

`tools/build_regions.py` takes names from Natural Earth 1:10m now (1:110m has no Singapore or
Hong Kong), names a country after its most populous feature (France's code is on Clipperton
too), and leaves parts more than 15° from the largest out of the opening view (French Guiana).

**Running several country agents at once** (how Korea's research and Taiwan were done,
alongside maps-9f on Japan): each agent owns `<cc>_register.py`, `<cc>_sources.md`,
`colours/<cc>.csv`, `data/raw/<cc>*`, `data/proc/<cc>`, `dist/data/<cc>*` and its own
`REGISTER["<cc>"]` entry (a RINF country owns `rinf_countries/<cc>.py` instead of a
register module), and asks the managing session for any change to a shared file
(`build_model.py`, `build_tiles.py`, `line_colours.py`, `extract.py`, `not_running.py`,
`rinf.py`) and for the `build_regions.py` run. Six ran at once on 2026-09-30 (China, four RINF
countries, and France by the managing session). Only one session rebuilds a given country, so outputs are never
clobbered. Shared-file changes get measured on every built country before they land: a
two-line `rank_of` change added 93 km of Swiss mountain railways and nothing in Korea.

**Shortlist, easiest first** (rewritten 2026-09-30 after `multi_sources.md`; Hong Kong,
Singapore, Belgium, Austria and France's reader are done):
- **More RINF countries with `rinf.py` as it is** (Czechia, Poland, Hungary and Portugal are
  done): Slovakia, Slovenia, Croatia (no line ids in RINF: needs OSM relations for numbers),
  Romania, Bulgaria, Greece, Lithuania, Latvia, Estonia, Finland, Luxembourg. Each is a
  `rinf_countries/<cc>.py`, an extract and a check table.
- **Known weakness of the RINF builds**: a section ending at a junction is kept only where an
  OSM passenger route runs over it, and in countries with patchy route relations that drops
  real passenger track (Czechia ~83 km, e.g. 238 into Havlíčkův Brod) while lines closed to
  passengers but still mapped railway=rail stay drawn (Hungary: parts of 27, 37, 62).
  A timetable feed (GTFS) per country would answer "does a train run here" better than OSM
  routes do; not started.
- **Germany**: RINF (filter the doubled 2026/2027 sections on validity) plus OSM
  `route=tracks` (1,923 with VzG numbers) plus Wikidata station km. Big extract: hand the
  extract to Anita.
- **Italy, Spain, Sweden**: RINF works but each needs a name map for its line ids.
- **India**: Wikidata's station adjacency is near complete (7,351 stations on 316 chains),
  over OSM track.
- **USA**: register lines are FRA NARN subdivisions, Amtrak routes services over them (Anita,
  2026-09-30, tentatively). ~10 GB extract; lines.json may need splitting by region for the app.
- **Russia**: no source researched yet; Crimea is Anita's call before any build.
- **Australia, USA, Canada**: their own subdivision-style registers (Geoscience Australia,
  FRA NARN, NRWN); one reader might serve NARN and NRWN.
- **UK, Norway, Denmark, Ireland**: later. Network Rail's track model now needs an account;
  RINF has no usable line ids for NO, and one id per section for DK and IE.

### What `build()` has to return

`lines` — a list of dicts:

```python
{"id": str,            # stable across builds; hash the name and operator, do not use an index
 "src": "n02",         # any non-"osm" value marks it a register line, which is what counts
 "service": False,     # True only for a named train rather than a line
 "name": str, "name_en": str, "ref": str, "colour": str,
 "operator": str, "operator_en": str, "network": str,
 "kind": str,          # rail | subway | light_rail | tram | monorail | funicular | narrow_gauge
 "km": float,          # the sum of its sections
 "variants": int, "straight_sections": int,
 "display": [station_id, ...],                  # reading order for the strip diagram
 "sections": [[station_id, station_id, km], ...]}   # a 4th element is added by build_model
```

Optional line fields:

- `"highspeed": bool` — only if the register knows. Where given, a register line only matches
  OSM ways with the same `highspeed=yes`, which keeps the Shinkansen and the conventional line
  beside it apart. Omit it and nothing is filtered.
- `"km_official": float` and `"chain": {"a|b": km}` — the register's own chainage, per
  section. `check_model` then compares every line against it, not just the handful in
  `REGISTER`.

`stations` — a dict keyed by station id:

```python
{"id": str, "name": str, "name_en": str, "lon": float, "lat": float, "lines": set()}
```

plus `"junction": True` on a section end that is not a stop. A section with a junction at
either end is kept only if OpenStreetMap passenger routes run over at least half of it
(`build_model.drop_unridden_sections`); that is how base tunnels stay and freight curves go.
A section between two stops is never questioned.

`geoms` — `{line_id: {"<from>|<to>": [[lon, lat], ...]}}`, one entry per section, keys matching
`sections`.

`build_model.merge_sources` does the rest: it matches OSM stations onto register stations by
name and distance (then a second pass for compatible spellings — "Bellevue" onto "Zürich,
Bellevue", 本町3丁目 onto 本町三丁目), hands OSM colours and English names to register lines,
drops an OSM line that is its register line twice over, keeps the rest as operating patterns
and named trains, and then computes the corridor credits. `register_way_lines` then ties
register lines to the OSM ways on the map, and corrects a register line's kind from the
track it lies on (N02 has no subway code, and legally every Osaka Metro line is a tramway).

### What Switzerland taught about registers

`schienennetz.py` is the second reader and differs from `n02.py` in ways the next country
will probably share:

- **A register line need not start at a station.** Swiss km-lines start at junctions as often
  as not; the Gotthard base tunnel has no stop at all. Every end of a line is therefore a
  section end, flagged `junction`, and OSM decides which of those sections passengers ride.
- **Which nodes are passenger stops came from a second file**, the service-point list. A
  platform group ("Chur [Gleis 10-14]", "Visp MGB-bvz") is a node of its own and has to be
  resolved onto its station by name.
- **Names differ in spelling, not just in script.** "Zürich, Bellevue" against "Bellevue".
- **The register's own chainage can jump inside a segment**: Zug Nord to Zug spans 10.1 km of
  chainage for under a kilometre of track. The nine Swiss lines `check_model` lists as off
  their chainage are that, not the build.

### Two traps that cost a rebuild each

Both are in `n02.py`, with the numbers they produced:

- **A line's track is a graph, not a chain.** Merging double track end to end gives a run that
  goes out on one rail and back on the other, so slicing between two stations traverses both.
  The San'in Line came out at 1,330 km against a published 674.
- **Sections come from an absorbing search, not from an ordering.** Ordering stations by
  distance from a terminus cannot work on a loop, because there is no terminus. The Oedo Line
  came out at 178 km against 40.7. A Dijkstra from each station that stops the moment it
  reaches another station gives the inter-station sections directly.

## Working on the app

`dist/index.html` is one file: constants, map setup, data loading, rides, crediting, then the
panel. Things that will bite:

- **Paint expressions must stay plain literal `const`s in SCREAMING_CASE.**
  `tools/lint_map_expressions.js` evaluates each `addLayer` object with those hoisted and
  reports anything it cannot evaluate as unchecked. Run it on every edit; it catches an
  invalid expression that would otherwise silently drop a whole layer.
- `['*', ['interpolate', ['zoom'], ...], factor]` is invalid. `['zoom']` may only be the
  direct input of a top-level `interpolate` or `step`.
- Added layers anchor at the first symbol layer **after the last non-symbol layer**, not at
  the first symbol layer, which in this basemap is a water label sitting before the roads.
- The basemap draws its own railways in near-black; they are hidden on load.
- Station bubbles come from the model, never from the tiles.
- No CSS transitions anywhere: state changes are instant.
- **The native name is the real name.** Someone riding trains in a country reads the names on
  the trains, and those are not in English. `lineName()` and `stName()` show an English name
  where one exists and fall back to the native one, and that fallback is correct behaviour,
  not a gap to paper over: never substitute an id, a transliteration or a placeholder for a
  name the data has in its own script. 61% of register lines have no English name and the app
  has to read well anyway.
- **Countries load as the map pans over them** (`considerCamera` on `moveend`):
  `regions.json` says where each is; a country's tiles come in when it is on screen or near
  it, its line data from z4, and countries holding rides load at start. Each country is its
  own `rail-<cc>` source with `track-<cc>` and `track-hit-<cc>` layers, all loaded countries
  share one model, and section ids are prefixed `<cc>:` as they are read (every country
  numbers its sections from 0). Rides carry `region`. The headline follows the country under
  the middle of the map (`FOCUS`).
- **The base map is in line colours** (`c` in the tiles, kind colour where missing). Ridden
  mode and the muted state DARKEN those colours inside the expression; they do not lower
  opacity, because double and quadruple track are separate ways stacked on top of each
  other and their opacity adds up to nearly full strength.
- **The strip diagram is a layout of the line's track graph** (`lineLayout`, drawn over the
  rows by `drawStrip` as one SVG from the rows' measured positions). The rules, each from
  Anita's feedback, are in the comments there and summarised in spec §13's "Added" note:
  stations of one route stay together, a loop at an end is walked in one column, and
  shortcuts and express track are not drawn.
- **The lint also fails on a top-level name declared twice.** A leftover second
  `function goRegion` that reloaded the page won silently and the page reloaded forever.
- To see a change: `python serve.py`, then drive a real browser:
  `node tools/screenshot.js http://localhost:8767/ out.png 15000 probe.js` with Chrome started
  headless with its own `--user-data-dir`. The header comment in `tools/screenshot.js` has
  the exact invocation. **Use a private debug port** (`CDP_PORT=9341`): 9222 is shared by
  every session, and a second Chrome on it silently lands on someone else's browser. A probe
  file can both drive the map and assert against it, which is how every UI change here has
  been checked. Wait on the map's `idle` with a timeout, or a probe can hang.

### Open for the app

- On a phone the open panel covers the map, so picking on the map and tracing a journey
  only work through search there. A bottom sheet would fix it.
- The rest is in spec §13, item 3 onward.

## Data files, and what is gitignored

`dist/data/<region>/` holds `lines.json`, `stations.json`, `credits.json`, `ways.json`,
`aliases.json` and `geom/<line>.json`; `dist/data/<region>.pmtiles` is the tile archive. All of
it is generated and excluded by the repo's `data/` rule, as are `data/raw/` and `data/proc/`.
Only source is tracked.

**`aliases.json` matters more than it looks.** Station ids move when a build merges OSM onto a
register, and a rider's saved trips name stations by id. The file ships the moves and the app
migrates saved rides on load. A change that moves ids without updating it silently voids
people's history. Its `lines` map does the same for line ids dropped as a register twin.
