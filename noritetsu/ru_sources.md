# Russia (ru): sources, measurements, recipe (surveyed 2026-10-01)

Stage-1 research for adding Russia. No reader written yet. Every number below comes from a
`probe_ru_*.py` script in this folder; their outputs are in `data/raw/ru/probe_*.txt`. Nothing
used needed a login. User-Agent `noritetsu-rail-map/1.0` throughout.

## The short answer

- **There is an official, open line register: Tariff Guide No. 4 (Тарифное руководство № 4),
  Book 1.** The CIS Council for Rail Transport publishes it as XLS, refreshed daily. For every
  railway it lists every tariff section (участок) between two nodes, with every station, passing
  loop, post and passenger halt on it **in order**, each with its six-digit ESR code and its
  tariff km from both ends. It is the Korail 거리표 / RINF of Russia: 1,117 sections and 88,315
  km for Russia plus Crimea (79,651 km in main sections; RZD publishes about 85,600 km of
  route). Book 2 says which points have passenger operations.
- **OSM's named track and infrastructure relations are not a register here.** Way names are
  corridor names ("Транссибирская магистраль", 16,925 km of track), not lines, and only 218
  `route=railway` relations exist, mostly whole corridors. Korea's and China's recipe does not
  apply.
- **OSM finds the register's points by their ESR code.** 11,693 OSM nodes carry `esr:user`;
  10,471 of the register's 12,759 points (82%) are found that way, 7% more by name.
- **OSM's passenger routes are unusually complete**: 3,913 `route=train` relations (3,072
  suburban, 780 long-distance), running over 91% of main-line track km. That is what decides
  which sections and stops are really ridden, which the tariff guide cannot.
- **A dry run of the tracing step already works**: over every gap between consecutive found
  points, OSM track length is 0.991 of the tariff km (82,678 against 83,411 km).
- **Recipe**: convert TR-4 + OSM `esr:user` coordinates into RINF's input shape and let
  `rinf.py` do the tracing, stop matching and checks, with a `rinf_countries/ru.py`. Details
  at the end.

## Run

```powershell
# sources (no login; dated files, see "Sources")
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/tr4_kniga1_2026-09-30.xls https://sovetgt.org/tr4/2026/09/30/Kniga_1_2026-09-30.xls
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/tr4_kniga2_2026-09-29.xls https://sovetgt.org/tr4/2026/09/29/Kniga_2_2026-09-29.xls
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/tr4_kniga3_2026-09-29.xls https://sovetgt.org/tr4/2026/09/29/Kniga_3_2026-09-29.xls
curl -L -A "noritetsu-rail-map/1.0" -o data/raw/russia-260930.osm.pbf https://download.geofabrik.de/russia-260930.osm.pbf   # 4.17 GB, ~10 min

$env:OSMIUM_POOL_THREADS=2; python extract.py --region ru --pbf data/raw/russia-260930.osm.pbf   # 7 min
python probe_ru_esr.py --pbf data/raw/russia-260930.osm.pbf     # 2 min; esr:user nodes -> data/raw/ru/osm_esr.json
# (the .pbf was then deleted; data/proc/ru and osm_esr.json are what the build needs)

python probe_ru_tr4.py      > data/raw/ru/probe_tr4.txt        # parses the XLS -> data/raw/ru/tr4_sections.json
python probe_ru_wikidata.py > data/raw/ru/probe_wikidata.txt   # Wikidata, cached in data/raw/ru/wd_*.json
python probe_ru_osm.py      > data/raw/ru/probe_osm.txt
python probe_ru_stops.py    > data/raw/ru/probe_stops.txt
python probe_ru_trace.py    > data/raw/ru/probe_trace.txt      # 5 min; dry run of the tracing step
python probe_ru_outline.py  > data/raw/ru/probe_outline.txt
python inspect_region.py --region ru;  python probe_kr_ways.py --region ru
```

`extract.py` does not keep `esr:user` on stops, which is why `probe_ru_esr.py` reads it from the
.pbf separately. A reader needs it: either rerun `probe_ru_esr.py` after every new extract, or
(shared-file change, for whoever holds extract.py) add `esr:user` to `STOP_TAGS`.

## Crimea, the 2022 annexations, and what the extract covers

- **Geofabrik's Russia extract includes Crimea.** Its page says so ("we have included Crimea in
  both the Russia and the Ukraine downloads"), and measured: 1,864 km of railway ways, 823 km of
  main and branch track, in Crimea; 72 `route=train` relations touch it. Crimea is also listed as
  its own sub-region (`russia/crimean-fed-district`, 38.7 MB), not needed.
- **religiondots' outline already puts Crimea in `ru`** (Simferopol, Sevastopol, Kerch and
  Dzhankoi all fall in `ru`, none in `ua`), so `tools/build_regions.py` needs no change for it,
  and the same outline can be the reader's clip polygon. (OSM's own Russia boundary relation was
  not checked; it may not include Crimea, so clip to the religiondots outline, not to OSM's.)
- **The extract runs past the border, as China's did**: besides 235,351 km of railway ways in
  `ru`, it carries 1,066 km in Kazakhstan, 542 in China, 204 in Ukraine, 148 Finland, 139
  Belarus, 106 Estonia and smaller amounts in Poland, Lithuania, Mongolia and Latvia. The reader
  needs a `--clip` step like `cn_register.py --clip`.
- **The register goes further than Crimea.** TR-4 files three more railways under the Russian
  administration: Донецкая (Донец, 69 sections, 1,542 km), Луганская (ЛУГАН, 28, 988 km) and
  Мелитопольская-Херсонская (МЕЛИТ, 4, 192 km), the regions annexed in 2022. Geofabrik's Russia
  extract does not cover them (only 204 km of border slivers fall in `ua`), and religiondots puts
  Donetsk, Luhansk and Melitopol in `ua`. The probes leave them out; the Crimea decision does not
  settle them either way. **Question for Anita.**
- Kaliningrad is in the extract and in TR-4 (Калининградская, 17 sections, 731 km); a separate
  part of the outline, left out of the opening view by build_regions' 15-degree rule. Trains
  between it and the mainland cross Lithuania and Belarus; the clip cuts them at the border.

## Sources

| what | where | access | used for |
|---|---|---|---|
| **Tariff Guide No. 4, Books 1-3** (Тарифное руководство № 4) | `https://sovetgt.org/tr4/<yyyy>/<mm>/<dd>/Kniga_<n>_<date>.xls`, index at `https://www.sovetgt.org/index.php?link=65`; directory listing at `https://sovetgt.org/tr4/2026/09/` | open, no login. A day's folder holds only the books that changed (others are a `.txt` saying "no changes since..."), so take each book from its last full folder. The link on the index page pointed at a 2026-10-01 file that did not exist yet (404). The guide's own preface: "Сведения ... являются общедоступными" (publicly available); an official intergovernmental document | the register: sections, points in order, ESR codes, tariff km (Book 1); passenger operations per point (Book 2) |
| tr4.info | `https://tr4.info/railway/<road>`, `/section/<road>/<n>` | open; robots.txt allows; says it is an unofficial presentation of the same data | reading TR-4 by eye; not needed beside the XLS |
| OSM, Geofabrik russia-260930 | `https://download.geofabrik.de/russia-260930.osm.pbf` (the `-latest` page took over 2 min to answer; the dated file downloaded in ~10 min) | ODbL | track, stations, `esr:user` codes, route relations, metros |
| osm.sbin.ru ESR list | `https://osm.sbin.ru/esr/` -> `esr.csv` (26,644 rows: code, name, railway, region, type, Express code), `osm2esr.csv` (19,957 OSM objects with coordinates), `express.csv`, `crimea.csv` (Crimea's old Ukrainian -> new Russian codes) | open, GPL code; updates "irregular, the source stopped updating" (2021 snapshot) | cross-check only: TR-4 is newer and has the order; OSM's own `esr:user` is current |
| Wikidata | `probe_ru_wikidata.py`, CC0 | open | English names (10,524 Russian station items carry the ESR code, P2815), line names ("A — B" items for 508 sections) |
| railwayz.info "Фотолинии" | `https://railwayz.info/photolines/rw/<road>` | robots allows with Crawl-delay 5; non-commercial use with a link, commercial only with permission | per-railway station lists that mark closed halts "(закр.)"; a possible check, not a source to build from |
| Roszheldor open data | `https://rlw.gov.ru/opendata/7708525167-railwaystations`, `...-tarifstations` | **redirects to a Gosuslugi (ESIA) login**: blocked | would be the same station list and TR-4 copy |
| Mintrans open data | `https://mintrans.gov.ru/opendata/7705851331-stoppointsmzd` | timed out from here (probably geo-blocked) | not tried further |
| Yandex Rasp API | `https://api.rasp.yandex-net.ru/v3.0/stations_list/` | **needs an API key** from a Yandex account | every station in Yandex's timetables with its ESR code and transport type: the best "is this a passenger stop today" list. Anita's call whether to get a key |
| tutu.ru | suburban timetables by direction | robots allows the direction pages; a commercial site; not used | |
| GTFS | Mobility Database lists only St Petersburg Metro (inactive) and Moscow Transport (inactive); Transitous has no Russian feed | | nothing for national rail |

## 1. Line registers, candidate by candidate

**TR-4 Book 1** (`probe_ru_tr4.py`). Per railway sheet, sections headed
`1) участок 17-002 "КУСКОВО - ОРЕХОВО-ЗУЕВО" (Основной тарифный участок)`, then rows of
`code, name, km to the first node, km to the second node` (a third column on some). Russian
administration sheets carry "(Р)".

| railway | sections | main | km main | km all | points |
|---|---|---|---|---|---|
| Октябрьская | 135 | 107 | 9,019 | 9,955 | 1,478 |
| Московская | 186 | 133 | 6,554 | 8,573 | 1,581 |
| Горьковская | 83 | 69 | 5,370 | 5,552 | 806 |
| Северная | 61 | 55 | 5,635 | 5,951 | 679 |
| Северо-Кавказская | 102 | 85 | 5,349 | 6,491 | 965 |
| Юго-Восточная | 71 | 58 | 3,755 | 4,404 | 824 |
| Приволжская | 63 | 52 | 3,763 | 4,191 | 545 |
| Куйбышевская | 67 | 58 | 4,390 | 4,869 | 951 |
| Свердловская | 72 | 61 | 6,056 | 7,256 | 993 |
| Южно-Уральская | 54 | 53 | 4,353 | 4,540 | 849 |
| Западно-Сибирская | 64 | 48 | 5,111 | 5,817 | 798 |
| Красноярская | 21 | 19 | 3,147 | 3,193 | 469 |
| Восточно-Сибирская | 16 | 16 | 3,862 | 3,862 | 612 |
| Забайкальская | 17 | 17 | 3,353 | 3,353 | 376 |
| Дальневосточная | 68 | 67 | 7,765 | 7,771 | 615 |
| Калининградская | 17 | 15 | 717 | 731 | 114 |
| Якутии | 4 | 3 | 808 | 1,162 | 26 |
| Крымская | 15 | 15 | 639 | 639 | 135 |
| ИФР-1 (Ленинск–Михайло-Семеновская bridge spur) | 1 | 1 | 5 | 5 | 4 |
| **Russia + Crimea** | **1,117** | **932** | **79,651** | **88,315** | **12,759 distinct** |
| (2022 annexations: Донец, ЛУГАН, МЕЛИТ) | 101 | 94 | 2,662 | 2,722 | 598 (summed) |

- Section types: main tariff section 1,026 (incl. 2022 regions), connecting line 102,
  low-activity (Малодеятельный) 64, Moscow node lines 10, St Petersburg node lines 10, "for
  passenger traffic only" 3, under construction 2, local traffic 1.
- **What it gives**: (1) a line inventory as "A — B" sections between nodes, no names beyond
  that; (2) stations in order, every halt included (Обухово — Чудово lists all 23 points,
  ОП Ижорский Завод and ОП Поповка too); (3) km per point, **whole kilometres only**.
- **Traps**: points repeat under extra codes with "(эксп.)", "(перев.)", "(стык)" suffixes
  (export, transshipment, border-junction codes of one station); some node sections list a
  node's internal stations at 0 km (17-150 "Лихоборы — Серебряный Бор" is the Moscow Central
  Circle's stations, all 0; 01-010 Автово — Цветочная likewise), so the MCC must come from OSM;
  128 inner points sit on two sections ("via" variants, "(через ст. ...)"); names wrap onto a
  second row; 898 of 1,404 section ends are a node of no other section (a branch leaves from a
  point inside another section, not from its end).
- **Book 2 part 1** (sheet РП): 6,345 Russian separation points with their operations; П, Б, О
  are passenger ones (5,463 have one). **Part 2** (sheet ОП): 7,037 Russian stopping points and
  platforms, 7,003 with О/Б/П, 34 marked Х (none).
- **Book 3**: tariff distances between transit points; not needed.

**ESR list, osm.sbin.ru.** Code, name, railway, region, type (станция / остановочный пункт /
разъезд...), Express code. 12,605 of TR-4's 12,759 points are in it, 9,885 with an Express
code. No order, no km. Its OSM link table (`osm2esr.csv`) is a 2021 snapshot; OSM's own
`esr:user` tags are current and cover more (10,471 against 10,241).

**Wikidata** (`probe_ru_wikidata.py`): 1,104 line items with P17 Russia (53 with an OSM
relation, 228 with a length, 58 with P1671); 11,895 stations, **10,524 with an ESR code
(P2815)**, which joins them to TR-4 directly; Crimea's stations carry both P17 Russia and P17
Ukraine (230 and 229 statements in a box around it). Adjacency is thin: 2,161 stations, 151
chained lines, mostly Karelia and Moscow Metro. **508 of the 1,117 sections (54,320 of 88,315
km) have a Wikidata line item named for their two ends** ("Петрозаводск — Беломорск"): ru.wikipedia
divides lines the same way the tariff guide does.

**OSM route=railway relations**: 218 in Russia (215 railway, 3 tracks), only 27 with a ref, and
the big ones are corridors: Байкало-Амурская Магистраль (4,439 ways), the Trans-Siberian in two
direction relations, "Киевское направление", "Continental Landbridge". 54% of main and branch
track km is in one. Not a register.

**railwayz.info**: per-railway schematic and station lists by section, with closed halts
marked; a hobby compilation under a non-commercial licence, behind a crawl delay. Useful to
check a doubtful section by hand.

**RZD's own open data**: none found beyond the Roszheldor and Mintrans portals above (login
and timeout).

## 2. OSM in Russia (`probe_ru_osm.py`, `inspect_region.py`, `probe_kr_ways.py`)

Extract: 272,243 track ways (249,857 rail, 14,764 tram, 5,688 subway, 1,675 narrow gauge),
131,839 stops, 4,925 route and 1,762 route_master relations, 284 infrastructure relations.

Heavy rail in `ru`, main and branch (rank < 2), track km (double track counted twice):

| | km | in a route=railway relation | named way | usage=main | freight-tagged |
|---|---|---|---|---|---|
| all main + branch | 133,191 | 54.0% | 40.5% | 90.0% | 1.0% |
| passenger-used (under a route=train) | 114,699 | 57.6% | 42.1% | 96.1% | 0.0% |
| Crimea, all | 823 | 48.2% | 10.3% | 87.7% | 0 |
| Crimea, passenger-used | 704 | 47.2% | 11.9% | 97.6% | 0 |

- **Passenger coverage** (inspect_region): 91.4% of main-line (rank 0) track km has a route
  relation on it, 33.6% of branch track. The `usage` tag is kept: 72,444 ways main, 5,834
  branch, 22,067 industrial.
- **Way names**: 1,317 distinct, all corridor-level: Транссибирская магистраль 16,925 km of
  track, БАМ (several spellings) ~6,200, Южсиб 1,666, "Казанский ход Транссиба", "Смоленское
  напр. МЖД". Moscow's radial "направления" are named on the way in places. Not legal lines.
- **route=train relations**: 3,913 mostly in `ru` (3,946 in the extract), 1,359 route_masters.
  Service tag: regional 2,753, long_distance 890, commuter 146, high_speed 34, none 46.
  Names are systematic ("Пригородный электропоезд: Дубна => Савёловский вокзал", "Скорый поезд
  124Ы: Красноярск → Абакан", "Скоростной поезд 740У «Аврора»"). Classified: **3,072 suburban
  (электрички, incl. МЦД, РЭКС, Ласточка suburban), 780 long-distance named trains, 61
  unclear.** Operators: АО «ФПК» 758 (long-distance), ЦППК 669, СЗППК 284 and about 25 more
  suburban companies. Caveat: intercity «Ласточка» trains (Minsk — Moscow, Хелюля — St
  Petersburg) carry service=long_distance but my name rule counted them suburban, so the split
  is approximate; `looks_like_service` needs a `ru` rule (a train number in the name, "Скорый
  поезд 124Ы", makes a named train).
- **Metros**: route=subway relations in all seven metro cities: Moscow 34, St Petersburg 12,
  Novosibirsk 4, Nizhny Novgorod 4, Samara 2, Kazan 2, Yekaterinburg 2; Volgograd's metrotram
  (light_rail) 4; two funiculars; 912 tram relations in 61 networks. 6,037 km of urban track,
  5,502 under a route relation.

## 3. Which stations are passenger stops (`probe_ru_stops.py`)

- **TR-4's flags are permissions, not service**: 11,886 of the 12,759 register points (93%)
  carry П, Б or О. Only 47 main sections (1,161 km) have no passenger point at all, so the
  flags cannot find freight-only lines.
- **OSM's routes say where trains call.** 82,985 stop/platform members of route=train
  relations (63,282 suburban). Of the 11,886 flagged points: 8,213 have a route stop within
  400 m, 1,827 have an OSM node but no route stop near it (closed halts, trains passing, or a
  route mapped without stops), 1,846 have no `esr:user` node in OSM.
- By section: 603 sections (31,309 km) have every flagged point served, 433 (55,369 km) some,
  47 (1,161 km) none. The "none" list is what freight-only looks like: Орск — Рудный Клад
  (192 km), Морозовская — Цимлянская, Кротовка — Серные Воды II, Бутурлиновка — Павловск,
  Раненбург — Лев Толстой, Ивдель II — Полуночное...
- **No open per-station passenger list exists.** Roszheldor's station list is behind the
  Gosuslugi login. **Yandex Rasp's `stations_list`** (one call, every station in Yandex's
  timetables with ESR code and train/suburban type) would settle it, but needs a key from a
  Yandex account.
- So: a register point is a stop when TR-4 flags it AND OSM shows it served (a route stop
  within ~400 m, or an OSM station/halt of the same name on the traced track, as `rinf.py`
  already does with NAME_M and `osm_stops`).

## 4. Practical problems

- **The 180th meridian: no change needed.** religiondots' `ru` outline is already split at
  ±180 (no ring edge jumps more than 180°; six Chukotka polygons sit at -180..-169.6). What
  `build_regions.py` would write: bbox `[-180, 41.19, 180, 81.86]` (the whole width, so the
  app's bbox reject never rejects ru by longitude, but `regionsInView` tests each part's own box
  and `inRegion` ray-casts per part, so the result is still right), and view
  `[27.33, 41.19, 180, 77.74]` (the mainland part only; Kaliningrad and Chukotka-east are left
  out by the 15-degree rule). `countryAt` wraps longitude into -180..180. No track lies near the
  meridian (Chukotka has none), so tiles and lines never cross it. Two small things for the
  app/tools owner: the opening view is 153° wide (Russia whole), and the `ru` outline keeps
  9,397 points at TOLERANCE 0.02, roughly +160 KB on a regions.json of 123 KB today; a coarser
  tolerance for very large countries would keep it small.
- **Freight-only lines.** TR-4 lists them alongside passenger lines with О flags on their
  halts. `build_model` never questions a section between two stops, so the reader must decide:
  draw a TR-4 section only where OSM route=train relations cover it (91% of main-line km is
  covered, so the risk is missing relations on minor branches), else mark it as `rinf.py`'s
  `suspended` does. 64 sections are typed Малодеятельный (low activity, 6,109 km).
- **Tariff km are whole kilometres.** Halts 1-3 km apart read ±0.5 km each; the reader's
  length check has to work on merged stop-to-stop sections with an absolute tolerance (~1-1.5
  km, `tol_abs`), as Hungary's does.
- **ESR codes not in OSM**: 1,846 flagged points (and 1,480 of 14,280 point-on-section
  occurrences after the name fallback) have no OSM node to place them.
- **Crimea's ESR codes changed in 2014** (47xxxx -> 85xxxx/86xxxx); OSM uses the new ones (109
  of 135 found), so no mapping is needed.
- **Moscow Central Circle and the MCD diameters**: the MCC is a 0 km node list in TR-4 and
  must stay an OSM line (as metros do); MCD-1..4 are suburban through-services over the radial
  sections and are OSM operating patterns.
- **Size**: Russia has 3,913 train relations against China's 134. After twin merging the OSM
  lines may still number a few thousand, so lines.json could be several MB (Japan's 1.2 MB); a
  per-region split or dropping suburban patterns that merely repeat one register section may be
  needed. The tile archive should be of China's order (cn.pmtiles 36.6 MB).
- **Kaliningrad**: fine, its own TR-4 railway, its own outline part.

## 5. Dry run of the tracing step (`probe_ru_trace.py`)

For every Russia + Crimea section: points found by `esr:user` (11,826 of 14,280 occurrences,
82.8%) or by name between found neighbours (974, 6.8%; 1,480 not found); then the shortest path
over OSM heavy-rail track between consecutive found points, each point claiming every track
vertex within 200 m and the nearest vertex of every way within 600 m, charged from the point.

- Gaps: 11,487 with a path, 96 with none within the limit, 116 with a point over 600 m from
  track.
- Gaps of 5 km or more (6,978): path/tariff median 0.987, 50.8% within 5%, 74.9% within 10%,
  2.2% over 1.25. Integer tariff km explain much of the 5-10% spread.
- **All bridged gaps together: 82,678 km of track against 83,411 tariff km (0.991).**
- Sections where every gap passes (within 15% + 3 km) end to end: 829 of 1,117, **57,788 of
  88,315 tariff km (65%)**. Most of the rest are long Siberian and northern sections with one
  to three bad gaps or a point not found (Лена-Восточная — Хани 1,128 km: 83 of 93 points
  found, no bad gap; Тайшет — Иркутск 670 km: 147 of 153, 2 bad gaps). `rinf.py`'s retrace
  past a failed piece and its own-relation test exist for exactly this.
- The first version claimed one vertex per point and sent 10-25 gaps per Trans-Siberian
  section round a crossover and back (path 1.076 of tariff overall); claiming every track
  vertex in reach fixed it. Korea's lesson holds here too.

## Recommended recipe

**TR-4 sections as the register, traced over OSM by `rinf.py`.** RINF and TR-4 have the same
shape (sections of line between typed points, with lengths), and `rinf.py` already does the
hard parts: tracing between consecutive points with a length check, retracing past failures,
stops matched to OSM stations by name, junction ends left to OSM routes, `osm_stops`,
`suspended`, `tol_abs`, `id_name`. So the Russia work is:

1. **`ru_register.py --convert`** (new, Russia-owned): read the TR-4 XLS (`probe_ru_tr4.py`'s
   parser) and `osm_esr.json`, and write `data/raw/rinf/ru/sections.json` and `points.json` in
   rinf's row format: one line per tariff section (id "01-011"), one section row per
   consecutive point pair with its tariff km difference, points typed station/stop where Book 2
   flags passenger operations and the point is served in OSM, else junction/switch;
   coordinates from `esr:user`, else an OSM station of the same name between placed
   neighbours. Drop "(эксп.)"/"(перев.)"/"(стык)" duplicate codes, 0 km node lists, and the
   2022-annexation sheets (pending Anita). Wikidata via ESR (P2815) for English station names,
   and the "A — B" line items for line names and English names.
2. **`rinf_countries/ru.py`**: `id_name` giving "Обухово — Чудово-Московское" from the section
   header (title case), `tol_abs` ~1.5, `osm_stops: True`, `suspended` for sections no OSM
   train route covers, the RZD regional railways as `im_of`. Whether `rinf.py` needs a hook for
   an input that comes from a file rather than `--fetch` is for whoever holds rinf.py; the
   converter writing rinf's own files should make it unnecessary.
3. Extract (done, `data/proc/ru` kept), `--clip` to religiondots' `ru` outline (a small
   reader step like `cn_register.py --clip`, so it lives in `ru_register.py`), build, tiles.
4. Shared-code asks: `looks_like_service` `ru` branch (a train number in the name, "Скорый
   поезд 124Ы", "«Сапсан»", is a named train; "Пригородный электропоезд: A => B" is a line);
   `esr:user` in extract.py's `STOP_TAGS`; possibly a lines.json size answer.
5. Check: TR-4's own km is the chain for every line (`km_official`/`chain`), so check_model
   compares all 1,117; Wikidata P2043 lengths (228 lines) as outside figures.

The register unit is then the **tariff section**: 1,117 lines averaging 79 km (Japan's 593
register lines average 47 km). Moscow — St Petersburg is seven of them; the Trans-Siberian about
forty. ru.wikipedia divides lines the same way, which is the argument for it; long corridors
(Транссиб, БАМ) stay visible as OSM named trains over them. Grouping sections into corridors
is possible later from OSM's corridor relations, but would be an invention for most of the
network.

**How good a first build would be.** The naive dry run already lays 65% of tariff km end to
end with every gap checked, and the track it found sums to 0.99 of the tariff total. With
rinf.py's retracing and tolerances I would expect 85-90% of the 88,000 km built as register
lines in the first pass, nearly all the remainder being long eastern sections with a point or
two missing in OSM, and a few hundred km of freight-only branches to settle by OSM route
coverage. Stations: about 10,500 register points placed by ESR code. Metros, trams, the MCC
and all suburban and long-distance services come from OSM as elsewhere. Expect build_model
times of China's order (6 min) and a log worth reading for the `suspended` list.

## Open questions

- The 2022-annexed regions (TR-4's Донецкая, Луганская, Мелитопольская railways, 2,722 km):
  in or out? Not in Geofabrik's Russia extract nor in religiondots' `ru` outline; left out by
  every probe.
- A Yandex Rasp API key (Yandex account) would give the one authoritative passenger-station
  list; worth it, or rely on OSM routes?
- Register unit: tariff section ("A — B", 1,117 lines), or corridors?
