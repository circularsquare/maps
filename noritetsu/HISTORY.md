# History

How noritetsu got the way it is, in order. Moved out of `HANDOFF.md` on 2026-10-02 so that file
can stay short; nothing was dropped. `HANDOFF.md` is the current state and the rules; this file
is the record behind them: how each country was built, the proposals that were declined and
their numbers (so nobody re-proposes them blind), the traps that cost a rebuild, and the
designs later replaced. Where a later fact replaced an earlier one, the earlier one is kept
here with a note saying what replaced it. Each country's own story in more detail is in its
`<cc>_sources.md`.

## 2026-09-30: Japan and Switzerland

Japan is built end to end and the tracker works. 1,101 lines of which 593 are register lines
(28,156 km), 9,072 stations, a 10.5 MB tile archive, and rides recorded from a strip diagram or
from the map with percentages per line, operator and overall. (As shipped on 2026-10-02:
1,094 lines, 593 register lines summing to 27,122 km, 9,073 stations.)

**Switzerland is built too** (2026-09-30): 779 lines of which 402 are register lines
(5,567 km), 2,816 stations, a 3.0 MB tile archive. The app loads it when the map moves
over Switzerland (`?region=ch` opens there), as it does every built country listed in
`dist/regions.json`.

The Swiss register needs two files in `data/raw/`, both open data, URLs in the docstring of
`schienennetz.py`: the BAV network (`schienennetz_2056_de.gdb.zip`, 3.4 MB) and the national
service-point list (`ch_servicepoints.csv`, 25 MB), which is what says which network nodes are
passenger stops.

The `.osm.pbf` extracts are deleted once extracted and checked; `data/proc/<region>/` holds
what was pulled out of them, which is all a rebuild short of re-extracting needs. Fetch the
extract again from Geofabrik to re-extract (the plain `-latest` URL for Switzerland was
redirect-looping on 2026-09-30; the dated `switzerland-YYMMDD.osm.pbf` worked).

`HANDOFF.md` was first written on 2026-09-30, for picking the project up cold and for **two
people working on it at once without colliding**.

### Settled 2026-09-30, for the record

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
  Gotthard and Ceneri base tunnels. (Superseded 2026-10-01: `build_credits` and
  `section_highspeed` are gone; track ownership in `ownership.py` reads the same
  `highspeed` / `highspeed_sections` flags.)
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

`check_model.py` compares built line lengths against published figures. JR trunk lines used to
run 5-20% long; that was `n02.py` letting sections slip past stations on multi-track lines,
fixed 2026-09-30 (station footprints and junction U-turns, in its docstrings). JR Central's
東海道線 is 1.06 because the figure leaves out its 美濃赤坂 branch and 垂井 bypass.

`looksStraight` in `index.html` guesses a straight-line fallback from geometry (over 2 km,
within 1% of crow-flies). Since 2026-09-30 it only guesses on lines whose `straight_sections`
is over 0 (12 in Japan, no register line), because elsewhere it was taking real straight
track out of strip diagrams.

`build_tiles.py` leaves out any connected piece of track that no line runs over
(`drop_islands`): tourist and amusement-park rides, harbour freight lines.

### Two traps that cost a rebuild each

Both are in `n02.py`, with the numbers they produced:

- **A line's track is a graph, not a chain.** Merging double track end to end gives a run that
  goes out on one rail and back on the other, so slicing between two stations traverses both.
  The San'in Line came out at 1,330 km against a published 674.
- **Sections come from an absorbing search, not from an ordering.** Ordering stations by
  distance from a terminus cannot work on a loop, because there is no terminus. The Oedo Line
  came out at 178 km against 40.7. A Dijkstra from each station that stops the moment it
  reaches another station gives the inter-station sections directly.

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

### Two tracks, and who owned what (2026-09-30)

The first arrangement, for two sessions at once. **Do not both edit `build_model.py`.** It was
the only file both tracks had reason to touch.

| Track | Owns | Must not touch |
|---|---|---|
| **Countries and data** | `n02.py` and new per-region readers (`kr_*`, `tw_*`...), `extract.py`, `build_tiles.py`, `check_model.py`, `inspect_region.py`, `probe_*.py`, `line_colours.py`, `colours/`, `data/` | `dist/index.html` |
| **Styling and app** | `dist/index.html`, `tools/` | the Python build |

`build_model.py` belonged to the countries track. If the app track needed a new field in the
output, it asked for it rather than adding it. Later replaced by one managing session holding
the shared files and country agents owning one country each (below, and `HANDOFF.md`).

### Open for the app, 2026-09-30

A headless load of `?region=kr` once opened on Japan (FOCUS 'jp') where earlier loads of the
same URL opened on Korea, and `goRegion('kr')` from a probe works. Looks like a race between
`regions.json` arriving and the START check. (Still listed as open in `HANDOFF.md`.)

## 2026-09-30: Korea

130 lines of which 83 are register lines (4,897 km), 1,215 stations, a 1.8 MB tile archive, in
the app at `?region=kr`.

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

**High-speed is per section** (`highspeed_sections`, then read by
`build_model.section_highspeed`, since 2026-10-01 by `ownership.py`), from the `highspeed=yes`
ways each section lies on. A line-wide flag either stopped KTX-이음 rides crediting 중앙선,
경강선 and 서해선 (partly new 250 km/h alignments), or let KTX on 경부고속선 credit 경부선
beside it into 대전 and 대구.

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

**Still off, and why** (the `<--` rows of `check_model.py --region kr`; also in spec §13):

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

Smaller data items still open then, all in spec §13: a per-line list of the 20 remaining
straight-line fallbacks for the app; Swiss sub-kilometre km-lines cluttering line lists; the
shorter-route-only rule when a km-line has two routes between stops.

## 2026-09-30: Taiwan

55 lines of which 36 are register lines (1,792 km), 519 stations, a 0.6 MB tile archive, at
`?region=tw`. `tw_register.py` follows Korea's pattern (OSM named track as geometry, importing
kr_register's graph helpers); stations from TRA's own open data (cut into the 16 legal lines
by `LEGAL`), OSM route stops for THSR and the metros, zh.wikipedia for Alishan. Every TRA line
within 1% of its published length; sources and faults in `tw_sources.md`. Two shared changes
came with it: `build_tiles.rank_of` draws usage=tourism narrow gauge (Alishan, and
Switzerland's mountain railways; the two-line change added 93 km of Swiss mountain railways
and nothing in Korea), and `build_model` merges named trains that differ only in train number
(THSR maps every train as its own relation) and strips Taiwan metro prefixes in
`norm_line_name`.

## 2026-09-30: next countries, as written for the next session

**The recipe that had worked twice (Korea, Taiwan) and needs no geometry register:** OSM
track named for its legal line gives the geometry, a published station list per line says
which stations are on it. `kr_register.py` is the template; `tw_register.py` imports its graph
helpers (`line_graph`, `Near`, `neighbours`, `between`) rather than copying them. First thing
to measure in a new country: `python probe_kr_ways.py --region <cc>` (works for any region),
the share of main-line km whose `name` is its line. Korea 98%, Taiwan high; where it is low,
the country needs a geometry register (N02, Schienennetz) or OSM route relations.

**Small shared-code hooks each country had needed**, all in `build_model.py`:
`looks_like_service` (what makes an OSM route a named train: a train brand in Korea, a train
number in Taiwan), `norm_line_name` prefixes (system names like 台北捷運 that OSM puts before
the register's line name). Nothing else in the shared build had needed a per-country branch.

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
  The first rebuild of jp and ch with these changes was expected to carry a few (the Tenjin
  bus-terminal record, "Rüti, Bahnhof", Beatenberg).
- **Declined, with numbers, so nobody re-proposes them blind**: skipping the proximity pass
  for named stop nodes (moves 229 in ch, 99 in jp, 35 in kr, nearly all correct
  spelling-variant matches like Genève-Cornavin onto Genève), and a mode-tag clash test
  (mixed: loses 市ヶ谷 onto 市ケ谷). Hong Kong's 機場快綫 stays listed twice as a result.

Also from that round:

- `build_credits`: **tram and light rail credit each other, but only on the same rails**
  (within `SAME_RAILS_M`, 8 m) and never for a register line marked `"guided": True` (n02
  sets it on N02's guideway codes). France's T11 is route=tram on light_rail track, Hiroden 2
  runs on to the light_rail 宮島線, the Forchbahn runs on Zürich tram track. At 45 m, or
  without the flag, the Astram and Nippori-Toneri guideways credited tramways they cross or
  run beneath. Measured: +174 pairs in jp (with the flag), +42 in ch, none lost. (The rule
  lives on in `ownership.py`, which kept `SAME_RAILS_M` and the `guided` flag when
  `build_credits` went.)
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
- `tools/build_regions.py` takes names from Natural Earth 1:10m now (1:110m has no Singapore
  or Hong Kong), names a country after its most populous feature (France's code is on
  Clipperton too), and leaves parts more than 15° from the largest out of the opening view
  (French Guiana).

**Running several country agents at once** (how Korea's research and Taiwan were done,
alongside maps-9f on Japan): each agent owns `<cc>_register.py`, `<cc>_sources.md`,
`colours/<cc>.csv`, `data/raw/<cc>*`, `data/proc/<cc>`, `dist/data/<cc>*` and its own
`REGISTER["<cc>"]` entry (a RINF country owns `rinf_countries/<cc>.py` instead of a register
module), and asks the managing session for any change to a shared file and for the
`build_regions.py` run. Six ran at once on 2026-09-30 (China, four RINF countries, and France
by the managing session). Only one session rebuilds a given country, so outputs are never
clobbered. Still how it is done; `HANDOFF.md` has the current form.

**Shortlist, easiest first** (rewritten 2026-09-30 after `multi_sources.md`; Hong Kong,
Singapore, Belgium, Austria and France's reader were done). Superseded item by item since; the
current list is `HANDOFF.md`'s open threads.
- **More RINF countries with `rinf.py` as it is** (Czechia, Poland, Hungary and Portugal done):
  Slovakia, Slovenia, Croatia (no line ids in RINF: needs OSM relations for numbers),
  Romania, Bulgaria, Greece, Lithuania, Latvia, Estonia, Finland, Luxembourg. Each is a
  `rinf_countries/<cc>.py`, an extract and a check table. (All built by 2026-10-01; Croatia's
  numbers came from RINF's section URIs instead, below.)
- **Known weakness of the RINF builds**: a section ending at a junction is kept only where an
  OSM passenger route runs over it, and in countries with patchy route relations that drops
  real passenger track (Czechia ~83 km, e.g. 238 into Havlíčkův Brod) while lines closed to
  passengers but still mapped railway=rail stay drawn (Hungary: parts of 27, 37, 62).
  A timetable feed (GTFS) per country would answer "does a train run here" better than OSM
  routes do; not started then. (Settled 2026-10-01 by `gtfs_served.py`.)
- **Germany**: RINF (filter the doubled 2026/2027 sections on validity) plus OSM
  `route=tracks` (1,923 with VzG numbers) plus Wikidata station km. Big extract: hand the
  extract to Anita.
- **Italy, Spain, Sweden**: RINF works but each needs a name map for its line ids.
- **India**: Wikidata's station adjacency is near complete (7,351 stations on 316 chains),
  over OSM track.
- **USA**: register lines are FRA NARN subdivisions, Amtrak routes services over them (Anita,
  2026-09-30, tentatively). ~10 GB extract; lines.json may need splitting by region for the app.
- **Russia**: no source researched yet; Crimea is Anita's call before any build. (Built
  2026-10-01, below.)
- **Australia, USA, Canada**: their own subdivision-style registers (Geoscience Australia,
  FRA NARN, NRWN); one reader might serve NARN and NRWN.
- **UK, Norway, Denmark, Ireland**: later. Network Rail's track model now needs an account;
  RINF has no usable line ids for NO, and one id per section for DK and IE.

## 2026-09-30: Hong Kong and Singapore

maps-d5 with one agent each, on Taiwan's pattern: `hk_register.py` (13 register lines, 293 km:
the ten MTR lines, Light Rail as one line, the trams and the Peak Tram; station lists from MTR's
DATA.GOV.HK files) and `sg_register.py` (10 register lines, 269 km; station lists from LTA
DataMall's station-code file, topped up from LTA's 2026 system map). Every line within 9% of
its published length, most within 2%; the misses are explained in `hk_sources.md` and
`sg_sources.md` (station-to-station against end-of-track figures, the Peak Tram measured along
its slope, Light Rail's one-way street pairs). Both clip the extract to the country (`--bbox`
on `extract.py`, then the reader's own `--clip` to a polygon), since Geofabrik has no Hong Kong
file (openstreetmap.fr does) and ships Singapore inside Malaysia's.

## 2026-09-30: Europe from ERA RINF, first rounds

maps-d5 with one agent: `rinf.py` is ONE reader for every country in the EU's infrastructure
register (national line numbers, sections with km, typed operational points), with a small
per-country table. Geometry is traced over OSM track between consecutive points and rejected
when it strays from RINF's section length; names come from OSM's `route=railway`/`route=tracks`
relations (`infra.pkl`), then Wikidata P1671, then first - last station. Only the
infrastructure manager's network is in RINF, so metros, trams and most private railways stay
OSM lines. **Belgium** 145 register lines, 3,196 km; **Austria** 125, 4,342 km (RINF also
carries the Steiermärkische Landesbahnen, Montafonerbahn, Raaberbahn and Neusiedler Seebahn;
not GKB, Wiener Lokalbahnen, Stern & Hafferl, Zillertalbahn); **Netherlands** 96, 2,780 km
(ProRail's ids are two station codes, "Asd-Rtd", read back into "Amsterdam - Rotterdam"). All
three median 0.996-1.000 against RINF's own section lengths. `be_sources.md`, `at_sources.md`,
`nl_sources.md`. The per-country settings are one file each (`rinf_countries/<cc>.py`) so
several agents can add RINF countries at once. Which source serves which country next,
measured: `multi_sources.md`.

Four more RINF countries the same day, one agent each, in parallel: **Czechia** 236 register
lines, 9,101 km (lines by timetable number, "010 Kolín – Česká Třebová", read from OSM's
route=tracks relations, since no rule maps SŽ's own numbers onto them); **Poland** 352,
16,171 km (PLK's line numbers; 29 of 30 checked lines within 2%); **Hungary** 133, 6,721 km
(MÁV and GYSEV; MÁV's RINF section lengths leave out station track, hence `tol_abs`);
**Portugal** 24, 2,138 km. Each has `rinf_countries/<cc>.py` and `<cc>_sources.md`.

`rinf.py` grew per-country hooks for them, every one a no-op unless the country sets it, and
each checked by rebuilding be/nl (identical): `skip_line(id)` (ids that are never lines:
private sidings), `no_ref(id)` (ids that never take a line number: Czech siding leads, which
otherwise turned siding junctions into branch points), `osm_rel(tags)` (read a relation's
number and name from all its tags: Czech OSM puts two numberings on `ref`), `im_of(section)`
(operator where RINF's manager code does not tell them apart: GYSEV under MÁV's), `tol_abs`,
and `cut_at_junctions` (end sections where other lines meet: Portugal only, because as a
default it cost Belgium real track, L.12 Antwerp-Essen 32.5 -> 26.0 km). Two generic changes:
trolleybus items never name a line (Wikidata numbers Debrecen's trolleybuses like MÁV lines),
and an end-to-end retrace past a failed piece must lie at least half on the line's own
relation (`OWN_DIRECT`: Poland's disused 223 was drawn as 135 km over two other lines).

## 2026-09-30: Mainland China

`cn_register.py`: Korea's recipe at scale. OSM's way names are China Railway's own line names
(97.9% of main-line track km named), so named track is the register, with 京沪线 and 京沪高铁
as separate lines and the national code (0002, 3002) as ref from OSM's route=railway
relations. Which stations are passenger stops comes from 12306's public station list (3,350 of
3,404 found in OSM), placed on a line within 300 m of its track. **417 register lines,
121,894 km** (142 mostly high-speed, 48,368 km); 26 of 30 checked lines within 3%.
`cn_register.py --clip` cuts Hong Kong and everything abroad out after each extract; Macau
stays in cn. Freight: a line with 12306 stations on it is never dropped as freight whatever
OSM's traffic_mode says (胶济线 is 65% freight-tagged and a busy passenger line), except 浩吉线
and 广珠线, left out by name. cn.pmtiles is 36.6 MB, 3.5x Japan's; lines.json 1.2 MB, the same
as Japan's. `cn_sources.md` has the rest.

## 2026-09-30: France

`fr_register.py`, from SNCF Réseau's own register: the RFN lines by their 6-digit code with PK
chainage, stations with a PK per line, shaped like Switzerland's. Whole country built: **278
register lines, 24,227 km**, chainage median 0.996 over 264 lines; Paris-Marseille, Paris-Brest
and Paris-Lille all 1.00 of their published lengths. Where several Wikidata items share an RFN
code, the longest names the line (750 000 is Moret - Lyon-Perrache, not the historic 58 km
Saint-Étienne - Lyon piece that shares its code). Built by the managing session.

## 2026-10-01: second EU round

Slovenia (22 register lines, 1,157 km), Bulgaria (30, 3,576), Slovakia (70, 3,376; 19 lines
Slovak sources list as without passenger service are greyed as not running through the new
`suspended` hook), with Romania, Finland and the Baltics (`baltics_sources.md`). New `rinf.py`
hooks, each a no-op unless a country sets it: `name_m`, `stop_names`, `fix(secs, points)` (a
country's in-place corrections to RINF: Slovakia's stations typed 110, lengths in metres, ids
spanning several lines), `suspended`, and `osm_stops` (True: OSM stations an OSM train route
stops at become stops on the sections they lie on, for registers listing only junction
stations, Latvia and Estonia; "all": every OSM rail station, Bulgaria, whose RINF has no
halts). Generic: Cyrillic names are transliterated in `norm()` (they used to normalise to
nothing), and Lithuania's "POINT (+22.69 ...)" coordinates parse. `extract.py` keeps
railway=construction track a passenger route still runs over (Slovenia's line 50 at Preserje,
rebuilt under traffic), and takes `--station-areas` for countries that map stations only as
areas (Bulgaria).

Known general issues the EU agents found then (each worked round per country; still open
unless `HANDOFF.md` says otherwise):
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
  every Plovdiv - Burgas train uses). The GTFS check should rescue this kind. (It did: the
  GTFS check keeps it, `gtfs_sources.md` "Bulgaria".)

**The app** (2026-10-01): a line id shared between countries (an OSM route crossing a border,
built in each country under one id) is merged into one line across countries, so the Eurostar
lights in nl, be and fr; `regions.json` carries `shared_lines` so the app knows which countries
to load. The camera, mode and open panel persist in localStorage (`noritetsu.camera`,
`noritetsu.mode`, `noritetsu.ui`); a first load shows the world.

**Timetable feeds** (`gtfs_sources.md`, 2026-10-01, early in the day): open GTFS exists for
nearly every built and planned country except China and Russia, mostly via the Transitous
mirror. Tested on Czechia (stops join by code) and Hungary: it rescues the track the OSM-route
rule drops and confirms lines with no trains. `gtfs_served.py` was then built on cz and hu
first, and rolled out the same day (below).

## 2026-10-01: Anita's decisions, recorded during the day

- **Completion: everything with scheduled passenger service counts**, "if it is scheduled at
  all (more often than about once a week)". Named trains: option B (no completion percentage
  of their own, and totals leave them out; their track counts through the lines it lies on).
- **Russia: the 2022-annexed railways (Donetsk, Luhansk, Melitopol–Kherson) go with Russia**,
  de facto, as Crimea does. **Reversed later the same day**: railways we cannot show trains on
  are given to no one, so they are left out (below, Russia).
- **Finland: keep the Porvoo museum line**: done by maps-ee (drawn greyed: no trains in VR's
  feed).
- **Czechia, JHMD (228/229 out of Jindřichův Hradec)**: unknown whether it runs; leave as is.
- **Cross-border track is not drawn**: a route that crosses a border (TER K80 Kortrijk –
  Lille) shows Mouscron and Tourcoing but no line between them. She wants it drawn. Designed
  and prototyped by maps-ee in `border_proposal/`; built the same day (below).

Later the same day, to maps-ee:
- **Named trains (option B)**: no completion percentage of their own, and totals leave them
  out; their track counts through the lines it lies on. Everything else with scheduled
  passenger service counts.
- **Border track**: counts in the country it lies in (a ride over the border credits both);
  sections already built whole over a border are cut and the far part handed to the
  neighbour once it is built (R2); crossings RINF has no point for are added by hand when the
  country on the other side is built; a border point shows one neutral name ("Belgium -
  France border"), not either country's RINF name.
- **Seasonal lines are drawn as running** (Poland's 96 Muszyna - Leluchów, summer weekends;
  Croatia's Metković - Ploče).
- **Not running is greyed** (Porvoo is fine greyed).
- **Russia: the tariff section is the line unit** (1,117 lines then, ~79 km each; 898 as
  built, of which 53 annexed are left out, so 845 ship).
- **Croatia's single "B" fast trains stay named trains.**
- **lu and gr GTFS**: OK to download the feeds. Inspecting data must only read it (no caches
  or side files beside it); ask her before anything that writes to data beyond an agreed
  build.

These answered the "waiting on Anita" list maps-ee had sent that day: seasonal lines (Poland's
96 Muszyna - Leluchów read closed because the feed window misses summer; Croatia's Metković -
Ploče was drawn), whether lu and gr get the GTFS check, and Croatia's single "B" fast trains as
named trains.

## 2026-10-01: what maps-ee did

The managing session's second session of the day, five agents, the managing session holding
the shared files.

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
- **GTFS timetable check live in 16 countries**: cz, hu, pt, pl, be, si, sk, ro, bg, fi, lt,
  lv, ee, hr, and (after Anita's OK) lu and gr. Switched off for at and nl (register too
  fragmented / nothing to gain). Greece: 509 km greyed on 5 lines (25 Serres - Alexandroupoli
  bus-replaced, 12 and 13 still closed since Storm Daniel, 10, Thessaloniki - Idomeni); its
  feed is kept unslimmed so replacement buses show, and it ends 2026-12-01 (refetch then).
  **Seasonal lines**: `PAST` snapshots (a past copy of a feed, from Mobility Database's public
  daily files) make a section with 16+ trip-days in the snapshot "seasonal", drawn running,
  unless replacement buses now call there. Poland's 2026-07-15 snapshot reopens 96 Muszyna -
  Leluchów and 131 Kraski (summer InterCity) and keeps 295 Węgliniec - border. Downloaded
  with Anita's OK (`data/raw/gtfs/pl/past/`) and pl rebuilt: closed 1,008 -> 910 km, nothing
  else reopened. 295 may be a summer works diversion rather than a seasonal service.
  `gtfs_sources.md` "Rollout (2026-10-01)" lists every closed line per country. Seven fixes to
  gtfs_served.py, in its docstring.
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
- **Cross-border track: landed and rolled out** (every country rebuilt on it, 2026-10-01;
  register lines unchanged everywhere, every station alias resolves to a shipped station, no
  second pass needed; line ids shared between countries 107 -> 151). A route over a border is
  built in each country up to a shared border point, and the app joins the two pieces by that
  station's id. `border_points.json` (tracked; `python borders.py --fetch` regenerates it)
  holds ERA RINF's 229 border points (op-type 90): one id per crossing, the same as a RINF
  register's own junction there (`"e" + uopid`), and one neutral name ("Belgium – France
  border", Natural Earth names, alphabetical). In build_model, `border_tails` adds a section
  from a route's last station in this country to the first border point on its track, when
  the route lists stops beyond that the extract lacks (R1). `split_at_borders` cuts a section
  one extract built whole over a border; it drops the far part once that country is built
  (dist/regions.json) and aliases the far station to the neighbour's id (R2). Border track
  counts in the country it lies in, and a ride over the border credits both registers (Anita,
  2026-10-01). Crossings with no RINF point are not drawn (ch-de and Basel, trams, the Channel
  Tunnel, non-EU borders): add one to `borders.EXTRA` when the country on the other side is
  built (Anita). Each build logs route ends that met no border point. Checked in scratch on
  be, nl, lu, cz, hu, fr against an unmodified build: register lines identical, all 92 border
  points named alike in every country, every dropped station aliased; register km made
  creditable be +136, cz +161, fr +98, nl +32, hu +28, lu +17. `border_proposal/PROPOSAL.md`
  is the design and has an "As built" section. Left open (also in `HANDOFF.md`):
  fr_register's own "Frontière" junctions don't reach the RINF points everywhere (Longwy 83%
  creditable, Thionville - Apach 80%, Morteau 94%, Modane 99%): snap them; be's existing
  European Sleeper -> Eurostar merge gives the Eurostar a one-sided Antwerpen - Essen border
  section; fr's extract still builds ~30 short foreign sections in its buffer (IC-04, IC-26,
  L-29; 108 km across all countries).
- Every country rebuilt at the end on the shared code above.
- **Russia** (`ru_register.py --convert` writes the tariff guide into rinf's input files, then
  `build_model.py --register rinf:data/raw/rinf/ru` with `rinf_countries/ru.py`): 898
  register lines, 76,776 km, one per tariff section; 3,513 lines with OSM, 388 of them named
  trains (`looks_like_service` ru branch: 3 digits + letter). Crimea 12 lines, 543 km. **The
  2022-annexed railways (53 lines, 1,291 km) are LEFT OUT, assigned to no country** (Anita:
  railways we cannot show trains on are not given to anyone; `ANNEX_RUNNING = False` in ru.py
  makes them `skip_line`; OSM has no passenger routes there). So 845 register lines,
  75,485 km ship. When a source says which trains run, set it True (they come back with
  Russia, de facto) and add `data/raw/ru/annex.geojson` to `EXTRA_AREAS` in
  tools/build_regions.py. annex.geojson is the four oblasts cut to where en.wikipedia's war
  maps mark Russian control (the tariff guide lists the whole pre-war Donetsk railway,
  Kramatorsk and Sloviansk included); the ru extract still holds a bbox clip of Geofabrik's
  Ukraine (`ru_register.py --clip`), harmless while those lines are skipped. New rinf hook
  `direct_near_m`. Sizes: build_model 7 min (ownership 74 s of it), lines.json 5.5 MB,
  foot.json 3.5 MB (credits.json was 23.5 MB). Only 16 outside lengths exist (12 within 2%);
  against the tariff km, median 0.995 over 887 lines. `ru_sources.md` "Built" has the rest.
  In regions.json since the end of the session; its outline (197 parts, 9,397 points) doubles
  regions.json to 322 KB.
  Before the build, the open thread read: an open official register exists (the tariff
  guide, sovetgt.org tr4 XLS, every section with ordered stations and km); recipe is a
  converter into rinf.py's input. The tariff guide files 2,722 km of the 2022-annexed
  Donetsk/Luhansk/Melitopol-Kherson railways under Russian administration; neither
  Geofabrik's Russia extract nor religiondots' outline includes the annexed ones, so they
  would need their own extract and outline handling. Line unit: the tariff section (1,117
  lines, ~79 km each; Anita, 2026-10-01).
- **Completion counts everything with scheduled service** (Anita), named trains no
  percentage of their own (option B). First done as an "own track" estimate in the app; then
  replaced, the same day, by the next item.
- **One owner per piece of track** (Anita, 2026-10-01: "no double counting, every piece of
  track belongs to exactly one line; for crediting only, trips can still be entered on any
  service or path"; `ownership.py`, called at the end of `build_model.main`). Every drawn way
  has one owning line: a register line by geometry (name, then an agreeing high-speed flag,
  then nearest; register lines on the same rails go to the lowest ref), station throats to
  the nearest register line, a merged twin's register line, else one OSM line by the fixed
  rule (lowest ref by natural sort, empty last, then name, then id). Named trains own nothing.
  A single-track register line beside another of the same operator gives it its ways (CBT
  Ovest to CBT Est, the Gotthard "Gleis links" lines to 600). Each section's footprint
  (`foot.json`: owner section, from, to, a, b) is what riding it credits, found by node ids,
  not a buffer; the app works out what each line owns from those. Totals count owned track
  once; a line's percentage is the share of its footprint ridden; an OSM line is listed when
  40% of it is off the register. Every build logs (`own:` lines) the fixed-rule places,
  same-rails losses, companions, likely register gaps and named-train-only track.
  credits.json, `build_credits`, `section_highspeed` and the 45 m buffer are gone. Rolled out
  on every country (2026-10-01, `rebuild.py --model-only`: register lines unchanged in all 27,
  every country has foot.json, no credits.json left). **Clicking track** (Anita asked, same
  day) opens the way's owner, with the other lines on that way listed under it ("Also on this
  track"). ways.json lists the owner first; its `unowned` key marks ownerless ways with
  several lines, and a list starting with a named train is ownerless too. The picker only
  appears for ownerless track. Below z10 the nearest listed line opens, register lines first.
  build_tiles reads each way's indices sorted, so the owner's position never changes a
  colour. Prototype and comparison: `ownership_prototype/WRITEUP.md`. Checked on trial builds
  of ch, be, fr, jp, cz, ru: register lines identical, the prototype's test rides match, 24
  old saved rides resolve with none lost. foot.json against credits.json: ru 3.5 vs 23.5 MB,
  fr 0.53 vs 1.19; build_model ru 22 -> 7 min. How often the fixed rule fires (places / km of
  way): be 95 / 243 (trams, metros), ch 16 / 77 (mostly register gaps), fr 139 / 728, jp 14 /
  6, cz 353 / 559 (trams), ru 1,288 / 3,804. Same-rails register losses: jp 105 km (real
  shared track: 成田空港線 -> 北総線 32 km), ru 382, be 36 (161A on 161 at Genval: a tracing
  error).

**How the work was run that day**: one managing session holding the shared files
(`build_model.py`, `build_tiles.py`, `extract.py`, `rinf.py`, `line_colours.py`,
`not_running.py`, `gtfs_served.py`'s hook, `tools/build_regions.py`), and up to eight
background agents at once, each owning one country's files and sending exact diffs for shared
changes. The managing session measured every shared change on all built countries before
landing it (a generic change that moves another country's output gets scoped or declined),
and ran `tools/build_regions.py` when a country was final. Since 2026-10-02 the measuring is
`tools/ab.py`, run only on the countries a change can touch (`HANDOFF.md`).

## 2026-10-02 (evening): what maps-12 did

Took the first three of HANDOFF's "next session, first" items. Each `build_model` change was
trialled with `tools/ab.py --all` (twice: the first run showed the English-name merge missed
Shanghai, whose two records are both made from bare stop nodes in `build_stations`, and that
first-word alone split jp's "小倉 => 下関" from its other direction), then cn ru jp be bg tw
were rebuilt (13 min, register lines unchanged in all six) and `build_regions.py` run.

- Named trains on a shared ref: now also the same first word, or the same two ends either
  way round. Every ref-only decision is logged ("twins on ref").
- Stations with the same English name within 50 m are one: Shanghai Line 14 whole again,
  12 other records merged, each logged ("stations: ...").
- App: the line view lists the trips over the line or the picked stretch (`tripsOver`).
  Checked headless on jp with a two-leg test trip: remove a leg, undo, show trip, delete.
- `tools/ab.py` compares `closed` as a set: ch's order changes from build to build and showed
  as two changed lines.
- jp's foot.json let a Yamanote ride credit Keikyu (0.48 km at 品川) and the Tokaido
  Shinkansen (0.76 km at 新橋). Fixed in ownership.py for Japan only (a way's tagged operator
  picks between two operators' register lines); tried everywhere first, and declined for cn,
  es, ch with the reasons in run()'s comment. jp rebuilt.
- Anita's answers: same-brand named trains stay merged (no preference); timetables are lower
  priority than new countries ("mostly just to verify details"), so de landed and it, es were
  parked in data/raw/gtfs_pending/; the USA next; a load-by-zoom system eventually (HANDOFF 9).
- GTFS: Germany live (gtfs.de); `mid_splits` for junctions RINF never cut a section at
  (Werdau - Neumark), in a separate pass after three tries (gtfs_sources.md); Renfe's cp1252
  and padded headers read. de and pl rebuilt.
- USA built: NARN's US passenger segments (`us_register.py --fetch`, 18,232 segments, 43,570
  km, public domain) plus its other segments under OSM passenger routes (`--fetch-holes`),
  OSM stations from the Geofabrik extract (deleted after; data/proc/us kept). A country agent
  wrote the reader; the managing session landed its build_model diff (us `looks_like_service`
  branch, Brightline changed to a line; `ROUTE_SHARE_BY_LENGTH`) and ran build_regions. 448
  register lines, 39,807 km.
- Later the same day: US second tracks folded into their lines (`companion_of`, her "both
  directions one track unless far apart"); US metro stations merge within 150 m (Manhattan's
  23rd Streets); a reload opens with nothing selected; the app offers the stop past a
  junction end through the operating patterns (`pastEnds`). Tried and dropped, numbers in
  the code: cutting US lines where another line ends on them, and three station-placement
  rules for Long Island City.
- Canada, Sweden and Australia built by three country agents in parallel while the managing
  session ran the downloads, extracts and shared diffs; Finland rebuilt for Haparanda.
  34 countries.

The detail of that day's finished queue items, moved out of HANDOFF's thread 0:

- **Named trains no longer merge on a shared ref alone.** `merge_osm_twins` also wants the
  same first word (`first_word`) or the same two ends either way round (`arrow_ends`, for
  jp's "小倉 => 下関" / "下関 => 小倉"), and logs every ref-only decision ("twins on ref ...
  merged / kept apart"). be: both European Sleepers are their own named train again, the
  Eurostar 205 -> 177 km; jp: はやて and はやぶさ (same OSM ref) are two trains. Still merged,
  as the same word allows: NJ 40235's directions (de at it), AVE Madrid - Málaga (es), it's
  EC Basel - Milano with EC Genève - Milano, nl's Eurostar London and Paris routes, be's two
  European Sleepers. Left merged: Anita had no preference, and credit is the same either way.
- **Stale renames.** `merge_duplicate_stations` folds records with the same English name
  within `EN_DUP_RADIUS_M` (50 m), and `build_stations` joins a record made from a bare stop
  node to an earlier one with the same English name within 50 m, naming it after the native
  name most of its stop nodes carry. Shanghai Line 14 runs 封浜 - 桂桥路 through 浦东南路 (the
  OSM node still says 东昌路); 12 other records merged, all the same station.
- **App, which trips rode this?** The line view lists "Trips on this line", or "Trips over
  A → B" once two stations are picked (`rideCredit` / `tripsOver`).
- **False credits in jp's foot.json.** 23 JR East ways had gone to Keikyu and the Shinkansen,
  whose N02 lines are drawn nearer; `ownership.py` now prefers the register line of the
  operator a way's tag names (`OPERATOR_TAG_REGIONS`, Japan only: tried everywhere, it
  misassigned in cn, es and ch). jp: 360 of 98k ways changed owner, register to register.
- **The GTFS check**: Germany live (gtfs.de); Italy's and Spain's feeds parked in
  `data/raw/gtfs_pending/`; `mid_splits` for junctions RINF never cut a section at.

## 2026-10-03: ten agents at once (managing session maps-0d)

Anita: "feel free to start lots of agents". One managing session ran ten agents in parallel,
each owning a list of files written into its brief (`agent_rules.md` in the session scratch,
summarised in HANDOFF "How the work is run"). Mid-session she paused big downloads for a while
("please dont do any big downloads rn"), so the four new-country agents started on research
and readers from small sources and built once the extracts arrived ("downloads fine now").

New for running many agents at once:
- `tools/slot.py`: every build, extract, trial and tiling run takes one of six shared slots
  (OS file locks in data/logs/slots/, released however the process ends), and numpy/osmium
  threads are capped to the slots held. Ten agents stayed inside Anita's 6-core cap with no
  counting.
- `tools/ab.py` takes `NORITETSU_AB_DIR`, so two agents trialling one country do not build
  over each other.
- `rules/<cc>.py` (refactor agent): build_model.py names no country any more. Every
  per-country rule (named trains, the Finland all-routes rule, Australia's platform names, the
  US metro duplicate radius, US/Canada share-by-length) moved to its country's file, shared
  regexes to `rules/shared.py`; `country_rules()`'s docstring lists the hooks. Checked two
  ways: the old and new functions on every relation and stop of all 34 countries' data/proc
  (0 differences; a deliberately broken copy failed it), and ab.py before/after plus
  side-by-side builds (0 differences). `tests/test_rules.py`. `norm_line_name`'s prefixes
  stay shared: ownership.py and line_colours.py call it without a region.

What each agent did (detail in each `<cc>_sources.md`):
- **Denmark** (RINF; a six-digit RINF id is Banedanmark's line number in its first three
  digits, named from da.wikipedia): 46 register lines, 2,408 km; Rejseplanen's timetable
  live. RINF's Danish lengths leave out station areas (København - Korsør sums to 78 km
  against 111), so da.wikipedia lengths are the check. Five no-op rinf.py hooks (`km_floor`,
  `way_line`, `osm_stop_route`, `osm_stop_extra`) and a fix in `split_at_osm_stops` (it
  dropped the light-rail share of a cut piece, so ownership gave Høje Taastrup-banen to
  Vestbanen).
- **Norway** from Bane NOR's own track register (Banenettverk, Geonorge, NLOD), not RINF
  (RINF's Norwegian sections are in four unconnected pieces, Støren missing); stops and "is
  it run over" from Entur's journey planner. 25 register lines, 3,689 km, all within
  0.92-1.01 of published lengths or explained. All four Swedish crossings end on RINF points.
- **India**: Wikidata's IR line items and station chains, section km from the unofficial IR
  GTFS (Neo2308), written in rinf.py's input format and traced by rinf.py. 720 register lines,
  67,135 km (IR publishes 69,393 route km with freight). 421 junction-to-junction lines named
  after their ends fill what Wikidata does not cover (`FILL` in in_register.py). Korea's
  named-track recipe fails here: 30% of track is named for its line.
- **UK** (`gb`): OSM's named track (97.2% of passenger track-km named), with repairs; 464
  register lines, 15,291 km. ELRs, OSM route=railway relations and Wikipedia lines measured
  and set aside (gb_sources.md). Heritage railways left out of the register.
- **Italy**: timetable check live (Trenitalia, Trenord, EAV); 209 km kept that OSM routes
  dropped, 809 km closed and each checked. "Nodo di ..." city lines split into the
  it.wikipedia lines (`NODE_SPLIT`); five dropped ids carried by `LINE_ALIAS`, which
  build_model now ships. AV Treviglio - Brescia still dropped (needs a high-speed rule).
- **Spain**: timetable check live; Pajares base tunnel and Chinchilla - Hellín kept, 252 km
  closed and checked. rinf.py's `cut_at_junctions` now also takes a set of points.
- **Germany and borders**: every Swiss-German crossing, Konstanz, Selb, Kehl tram, Tønder
  and the Öresund bridge share one border id on both sides (`borders.EXTRA`, new `MOVE` and
  `SAME`); DB's track on Swiss soil now counts only in Switzerland; freight bypasses 1280 and
  1750/1751 greyed; DB InfraGO's licence confirmed CC BY 4.0.
- **Russia**: English station names from Wikidata by ESR code (P2815), kept only where the
  label reads as a romanisation of the Russian (register stops with one 21% -> 41%), line
  English names from their ends (5% -> 46% of lines); one picked colour per regional railway.
- **App**: line data loads from z8, not z4; country totals precomputed into regions.json
  (`owned_totals`, a Python copy of the app's ownTrack + regionTotals); not-running track
  from one `dist/data/closed.json`. Europe at z4: 33 MB of JSON and 158 MB heap -> none and
  23 MB; first track 5.3 s -> 1.3 s.
- **US station placement** (Long Island City): stopped. LIC and Hunterspoint Avenue are in
  no OSM route's stop list, so they have no stop nodes and the HANDOFF idea cannot reach
  them; two more rules tried, numbers in us_register's place_stations comment.

## 2026-10-03: decisions settled by the agents' leans

Anita: "i trust your leans", and from now on local specifics outside the US and Japan are
decided by the session, not put to her (HANDOFF standing decisions). As built:
- au: Puffing Billy and Kuranda (daily) count, 2-3-day heritage lines left out; Queensland's
  Caboolture / Ipswich lines folded into the North Coast Line and Main Line. ca: the Rocky
  Mountaineer a named train whose ex-BC Rail track counts nowhere. se: Tågab's weekly Sunday
  train kept running, Dal Västra Värmland's summer trains greyed.
- gb: OSM track names are the line unit (ELRs set aside); heritage railways' own OSM routes
  count as lines, as in the US; Chiltern's track Harrow - Amersham left to the Metropolitan line.
- in: 421 junction-to-junction lines fill what Wikidata lacks (`FILL`); Jammu & Kashmir and
  Arunachal drawn as India, as trains run.
- no: long-distance F-lines and Oslo - Stockholm are lines, only Nattåg 93 a named train;
  Flåmsbana a line; freight-only and closed track left off; the old Nykirke - Barkåker line
  built until the new alignments open.
- dk: Gedserbanen greyed; Storebæltsforbindelsen its own line; SJ's Tog 80 a line.
- es: Riquelme-Sucina not a stop; 120 west of Salamanca and 822 Ourense - A Gudiña closed
  through `FEED_COMPLETE` in gtfs_served (landed the same day).
- it: Villa Opicina - Bivio d'Aurisina kept; the Nodo split as built.
- de: DB's track on Swiss soil counts only in Switzerland; 1280 greyed despite ICE
  diversions; the Rafzerfeld left to ch.
- ru: one picked colour per regional railway; Crimea's Ukrainian-based English spellings kept.
- app: unvisited countries show "0.00% · 0.0 of N km".

Asked for by her the same evening: the names-only search index (dist/data/search.json, built
by build_regions) and a US station override list (us_register). The search agent found that
Russia's picked colours had been written as bare hex (`2F7FD8`), which draws nothing:
ru_register now writes `#`, and line_colours.apply adds a missing `#` to any table's colour.

## 2026-10-03 (evening): thirteen more regions, eight agents

Anita: "anything else to do? maybe move onto more countries?", and no factual questions to
her at all, the US included ("ill let you know if i see something thats wrong"). The
managing session downloaded and extracted all thirteen Geofabrik files in one script (four
downloads at a time, each extract through the slot tool as it landed; ~4.5 GB, 25 min) while
eight agents researched; each built its country once `data/proc/<cc>` was ready.

- **ie**: RINF's per-section ids grouped into 17 IÉ lines by a section table; the extract
  clipped to the Republic (NI is gb's); NTA's rail feed live; the Dublin - Belfast line joins
  gb at Ireland's own RINF border point moved onto the boundary (`eIEOP42`). 14 published
  lengths all within 3%.
- **tr**: OSM's named track (98.4% named); TCDD's ticketing CDN list decides stops; YHT
  services are lines, the "Ekspresi" trains named. 32 checked lines, 28 within 5%.
- **ua**: the six UZ sheets of Russia's tariff guide, through rinf.py like Russia; outline OSM
  Ukraine less Crimea less the annexed area; no national feed, so poizdato.net's 1,086 train
  pages were crawled (robots.txt allows) into a GTFS for gtfs_served; frontline track OSM had
  retagged disused put back so the timetable greys it by the usual rules. 3,359 km greyed.
- **Balkans** (rs, ba, me, mk, al, xk) by one agent and one reader: Serbia from IŽS's network
  statement chainage, the others from hand lists; five community feeds decide what runs.
  Kosovo is its own region (Trainkos, de facto); the Belgrade - Bar line's stretch through
  Štrpci stays Serbia's.
- **my**: lines laid by shortest path through KTM's and Prasarana's GTFS stops (the KTM track
  is mostly named just "KTM"); ETS a named train; Komuter lines OSM lines over the register.
- **th**: OSM named track plus route=railway relations; SRT's numbered trains are named
  trains, the Red Lines, ARL, BTS, MRT lines. 3,919 km = SRT's 4,044 less 125.6 freight-only.
- **mx**: passenger track only (99.4% named); Tren Maya one line; the three Interoceánico
  lines greyed since the December 2025 derailment.
- **id**: id.wikipedia's 38 line articles (station tables with KAI's km posts) through
  rinf.py; KAI's intercity trains named, commuter and airport trains lines. Median 1.000
  against the km posts.

Shared changes landed by the managing session: `extra_route_stops` (us/ca),
`TWIN_ON_STATIONS` (mx), `served_sections` (one hook for th and my, popped before shipping),
`norm_line_name`'s "Laluan" rule (my), `NO_GROW_OVER_ROUTES` (rs), `FEED_COMPLETE` (es),
borders.EXTRA/MOVE for Dundalk, Padang Besar, Woodlands, Nong Khai, Aranyaprathet,
Vrbnica, Metković, Röszke, Jimbolia; build_regions' `OUTLINE` (Ukraine's own). Then one
batch rebuild of the 19 countries the changes or new neighbours touched.

## 2026-10-03 (night): twenty-one more regions, ten agents

Anita went away ("feel free to keep going"). Same set-up as the evening round: one download
script for all 21 extracts, ten agents, the managing session landing shared diffs and
rebuilding at the end.

- **vn** (OSM named track cut as DRVN's list; North-South 1.00), **nz** (KiwiRail's own
  register and km posts; 15 passenger lines, median 0.997), **za** (a hand list traced by
  rinf.py; PRASA's recovery reports decide what runs), **br** (passenger track only; CPTM by
  named track, SuperVia and Vale's railways by their routes' ways), **ar, cl** (the ways of
  hand-listed passenger routes; Argentina's chainage checked against the ministry's km posts),
  **ir** (OSM route=railway relations; RAI's timetable crawled from iranrail.net).
- **ma, dz, tn, eg** by one reader from the operators' timetables (Annaba - Tunis restarted
  2026-09-15: Algeria built to the border).
- **by, md, kz, uz, kg, tj, tm, ge, am, az** from the same tariff guide Book 1 that Russia and
  Ukraine use, one sheet per railway; timetables crawled where robots.txt allows (poezdato,
  merstren.md, KTZ's ticket site) or written by hand (Azerbaijan, Turkmenistan, Tajikistan).
- Who the map depicts, asked and not acted on (HANDOFF thread 0): Abkhazia and Transnistria
  built as no region's; Donetsk/Luhansk still no region's though a source now shows trains.
- Shared changes landed: `SKIP_ROUTES` (br), rinf.py's `osm_stops_skip` (nafrica) and
  `plain_name` (nz), build_regions' OUTLINE for ge and md, border points for Đồng Đăng,
  Sadakhlo, Gardabani, Souk Ahras, five Belarus - Russia, one Moldova - Ukraine and twenty
  Central Asian crossings. A separate agent then gave Russia and Ukraine their sides of the
  new crossings.

## 2026-10-05: the three handed-off threads finished (maps-33)

The 10-04/05 managing session stopped three agents mid-work (HANDOFF thread 0 then;
`handoff_notes/` has their notes and tools). maps-33 finished all three, in two rounds of
rebuilds.

**Round 1: ownership and the Channel Tunnel.** The previous session's all-country trial
(`aball3`) was still running from 09:42 and finished all 73 countries; gb was re-trialled alone
with the narrowed rules/gb.py stop rule (5 lines differ: HS1 and four named trains). No
register line moved anywhere else. The new-owner rows were read for ru, de, pl, at, us, jp
and gb: the 7509/7510 suburban pair, Kazan's ring trams 5/5a and Magnitogorsk's 9/10 now
share an owner; the Crimean bridge's second track went to the Kerch - Anapa diesel, the
Sheremetyevo Aeroexpress's track to the Savyolovsky register line; Poland's tourist tram 0
takes both tracks where it ran one way (the lowest-ref rule, as before, now over both). No
pair of genuinely different lines. One unrelated fault found in the trial and fixed:
`us_register.osm_line_name` took `Counter.most_common` over votes added in set order, so a
tie between two OSM relations over one track (Melbourne's standard gauge is both the
Adelaide and the Sydney corridor) came out differently per run; ties now go by name. Rebuilt:
gb and fr with tiles, 44 more model-only (13 min and 37 min, 0 failed), then build_regions.
compare_lines: no register line changed but HS1 (+0.29 km) and the new Tunnel sous la Manche.
PATH as Anita rode it, on the shipped data: JSQ-33rd 94.9%, 33rd-Hoboken 96.8%.

**Round 2: RINF lines in pieces.** The dev copy landed in rinf.py with the env switches
removed. Two faults found on the way and fixed before landing:
- Germany's 4721 Untertürkheim - Nürnberger Str vanished in the dev trial. A guard was
  added (a fill at most 30% on track another RINF line was traced over, `HOLE_OTHER`), but
  the fill was not the cause: 4713's fill Nürnberger Straße - Bad Cannstatt is 0% on other
  lines and is the S-Bahn's real track. With 4713's section a hole, the timetable check had
  routed S-Bahn trains over 4721 instead; with the hole filled, nothing is left on 4721.
  Its loss is a false credit removed. The guard stays; it refuses one fill (Wien Penzing -
  Hütteldorf, 48%).
- A fill's stops came from every OSM station a train route stops at, and OSM has a second
  "Wien Hauptbahnhof" node 230 m from the register's: the fill from the car-train terminal
  made it a station of its own and 30 of 33 lines moved to it. A fill now leaves out an OSM
  station with a register stop of its name within 1 km (`FILL_SAME_NAME_M`).
Trial with fills on in all 50 RINF-read countries: only at, be, cz, pl, bg and de fill
anything, so `fill_holes: True` is set in those six; folding moves ru, ua, by, kz, uz. The
1,297 junction ids that go (ru 1,142, ua 115, kz 26, by 8, uz 6) are all junctions. Station
ids that become register stops through a fill (Pusarnitz, Wolfurt, Traismauer) are aliased.
`bridge_pieces` is written and still untrialled.

**Round 3 (after Anita's look at the map the same evening).** She found small gaps left in
Austria, the Eurostar missing a stretch, a French line far apart in pieces, UK "branches that
lead to no station", and a stop past a junction whose track was not drawn on the map; and
asked for a list of every line in pieces, to go through one by one. RINF was queried for the
Tauernbahn's holes (her OK): nothing under any id or country. Landed, each trialled on every
country it could touch, then rebuilt (gb, tr, de, fr, kg, kz, at with tiles; ru, ch model):
- `tools/pieces_report.py` and `handoff_notes/lines_in_pieces.md` (the worklist);
- `build_model.transit_tail`: a route with no stop in a country crosses it border to border
  (Eurostar Amsterdam - London across France; railjets across Germany; Russian trains across
  Kazakhstan; Kazakh trains across Kyrgyzstan);
- `rinf.fill_holes` asks again a pair that passed but lost to a shorter fill (Kamptalbahn);
- `build_model.fold_spurs`: out-and-back spurs at a line's middle junction cut out, or the
  detour dropped beside the line's own direct section. Three tries: guarded first by another
  line ending at the junction (gb's junctions have none, so 1 of 56 folded), then by track
  leaving the junction off the spur (Chodov-úvrať's switchback still folded), finally by the
  angle the line's two legs leave the fork at (a switchback's legs leave in a V);
- dist/index.html: a continuation's track was sliced from the wrong half of a section whose
  points run against its key (Hell Gate Line -> Penn Station).

**Round 4: far pieces split** (Anita: "ok we can split lines", then stop and wrap up).
`build_model.split_far_pieces`, after fold_spurs in main(): a register line's piece over 25 km
from its biggest piece, with two stops or more, is a line of its own (id from the line's id
and the piece's lowest stop; name_en "<name> (<first> – <last>)"; km_official shared by km;
reg_ways, sec_ways and aliases.json `pieces` moved). The register module's KEEP_WHOLE holds
(cn's 青荣城际线). Trialled on the 20 countries with such lines: 54 lines made 75 more in
15 (au 17, fr 8, kz 6, de 6, pl 5, jp 4, in 4, ru 2, ua 2, tm 2, bg, pt, az, nl, al 1); no
station id moved. Rebuilt with tiles. The rest of the worklist is parked at her request.

## The reader contract's notes, as of 2026-10-01

`HANDOFF.md` keeps the contract itself (what `build()` returns) in short form. The longer notes
that went with it:

- `"highspeed": bool`, only if the register knows. Where given, a register line only matches
  OSM ways with the same `highspeed=yes`, which keeps the Shinkansen and the conventional line
  beside it apart. Omit it and nothing is filtered.
- `"km_official": float` and `"chain": {"a|b": km}`: the register's own chainage, per
  section. `check_model` then compares every line against it, not just the handful in
  `REGISTER`.
- A section with a junction at either end is kept only if OpenStreetMap passenger routes run
  over at least half of it (`build_model.drop_unridden_sections`); that is how base tunnels
  stay and freight curves go. A section between two stops is never questioned.
- `build_model.merge_sources` does the rest: it matches OSM stations onto register stations by
  name and distance (then a second pass for compatible spellings: "Bellevue" onto "Zürich,
  Bellevue", 本町3丁目 onto 本町三丁目), hands OSM colours and English names to register
  lines, drops an OSM line that is its register line twice over, keeps the rest as operating
  patterns and named trains, and (until 2026-10-01) then computed the corridor credits.
  `register_way_lines` then ties register lines to the OSM ways on the map, and corrects a
  register line's kind from the track it lies on (N02 has no subway code, and legally every
  Osaka Metro line is a tramway).
- Usually nothing else in the build needs editing for a new reader; the small exceptions are
  the shared hooks (`looks_like_service`, `norm_line_name`) listed under "next countries"
  above.

## The app as of 2026-10-02

Moved out of `HANDOFF.md`'s "Working on the app", which keeps the rules that bite. This is how
`dist/index.html` is put together; still current when moved.

- **Countries load as the map pans over them** (`considerCamera` on `moveend`):
  `regions.json` says where each is; a country's tiles come in when it is on screen or near
  it, its line data from z4, and countries holding rides load at start. Each country is its
  own `rail-<cc>` source with `track-<cc>` and `track-hit-<cc>` layers, all loaded countries
  share one model, and section ids are prefixed `<cc>:` as they are read (every country
  numbers its sections from 0). Rides carry `region`. The headline follows the country under
  the middle of the map (`FOCUS`). A line id shared between countries is one line
  (`shared_lines` in regions.json). Camera, mode and open panel persist in localStorage
  (`noritetsu.camera`, `noritetsu.mode`, `noritetsu.ui`).
- **The base map is in line colours** (`c` in the tiles, kind colour where missing). Ridden
  mode and the muted state DARKEN those colours inside the expression; they do not lower
  opacity, because double and quadruple track are separate ways stacked on top of each
  other and their opacity adds up to nearly full strength.
- **Themes.** `index.html` has a dark and a light theme. Colours in the page are the CSS
  variables in the two `:root` blocks; map paint and the strip's SVG colours come from
  `THEMES`. Layers are added with the dark theme's literals (so the expression lint checks
  them) and `applyTheme` sets the rest; switching theme swaps the basemap with `setStyle` and
  `carryOver` moves this page's sources and layers across. A new colour goes in both.
  Each country has a `track-edge-<cc>` layer under `track-<cc>`: the edge for lines whose
  brightness (`LUM_*`, computed in the expression from the line colour) is too close to the
  theme's background, with the track drawn at 0.65 of its width over it.
- **Clicking track** opens the way's owner, with the other lines on that way listed under it
  ("Also on this track"). The picker only appears for ownerless track (ways.json's
  `unowned`). Below z10 the nearest listed line opens, register lines first.
- `looksStraight` only guesses straight-line fallbacks on lines whose `straight_sections` is
  over 0.
- **The strip diagram is a layout of the line's track graph** (`lineLayout`, drawn over the
  rows by `drawStrip` as one SVG from the rows' measured positions). The rules, each from
  Anita's feedback, are in the comments there and summarised in spec §13's "Added" note:
  stations of one route stay together, a loop at an end is walked in one column, and
  shortcuts and express track are not drawn.

## Run commands as listed on 2026-10-01, with timings

`tools/rebuild.py` now runs model and tiles for any list of countries; these are the single
steps it wraps, with how long each took then.

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

Russia (from `ru_register.py`'s docstring and the build above): `ru_register.py --clip` after
every extract, `ru_register.py --convert`, then `build_model.py --region ru --register
rinf:data/raw/rinf/ru`, 7 min.

## Superseded designs

Kept so the old names in logs and commits make sense.

- **Corridor credits and the 45 m buffer** (2026-09-30 to 2026-10-01): `build_credits` in
  `build_model` wrote `credits.json`, crediting a section to every line whose track lay within
  45 m (8 m, `SAME_RAILS_M`, for tram and light rail), with `section_highspeed` keeping
  high-speed and conventional apart. Replaced by track ownership (`ownership.py`, `foot.json`).
  `merge_sources` no longer computes corridor credits.
- **The "own track" estimate** (2026-10-01): the first form of option B, worked out in the app;
  replaced the same day by one owner per piece of track.
- **Russia's annexed railways with Russia** (morning of 2026-10-01): replaced the same day by
  leaving them out until a source shows trains.
- **Two tracks** (2026-09-30): countries-and-data against styling-and-app, above; replaced by
  the managing session and country agents.
- **"After any shared-file change"** (2026-10-01): `compare_lines.py save`, `rebuild.py`,
  `compare_lines.py diff`, `build_regions.py`, and rebuilding a new country's built neighbours.
  Still the steps for a real rebuild; since 2026-10-02 a trial with `tools/ab.py` comes first.
