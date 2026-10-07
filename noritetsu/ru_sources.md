# Russia (ru): sources, measurements, recipe (surveyed and built 2026-10-01)

Built 2026-10-01 by the recipe at the end: `ru_register.py` converts the tariff guide into
rinf.py's input and `rinf_countries/ru.py` holds the settings. "Built" below is what came out;
the sections after it are the stage-1 research, kept as it was written (its numbers come from
the `probe_ru_*.py` scripts, outputs in `data/raw/ru/probe_*.txt`, and predate the build's
clip and outline). Nothing used needed a login. User-Agent `noritetsu-rail-map/1.0` throughout.

## Built (2026-10-01)

**898 register lines, 76,776 km**, one per tariff section, named by its two ends as
the tariff guide spells them ("Обухово — Чудово-Московское"), operator the regional railway
("Октябрьская железная дорога"). Of them 53 lines (1,291 km) on the 2022-annexed railways are
drawn greyed (below). With OSM's lines and trains: 3,513 lines, 15,832 stations; 388 named
trains (FPK's numbered trains, Сапсан, the long-distance «Ласточка»s) are flagged as such by
build_model's `ru` branch. ru.pmtiles 22.9 MB. (The annexed railways were later left out, then
built again on 2026-10-04 with what runs: "The annexed railways: what runs".)

What the 1,180 tariff sections the converter kept became, by railway (tariff km of the
sections against register km built; the gap is unridden sections dropped, rejected traces,
and pairs left on another section or outside Russia):

| railway | sections | tariff km | lines built | km built |
|---|---|---|---|---|
| Октябрьская | 134 | 9,946 | 113 | 8,597 |
| Московская | 182 | 8,520 | 153 | 7,312 |
| Свердловская | 72 | 7,249 | 53 | 6,413 |
| Дальневосточная | 67 | 6,920 | 49 | 6,078 |
| Северо-Кавказская | 102 | 6,491 | 78 | 5,237 |
| Северная | 59 | 5,936 | 45 | 5,581 |
| Западно-Сибирская | 63 | 5,803 | 53 | 5,055 |
| Горьковская | 82 | 5,550 | 63 | 5,034 |
| Куйбышевская | 63 | 4,827 | 46 | 4,233 |
| Южно-Уральская | 53 | 4,423 | 35 | 3,306 |
| Юго-Восточная | 71 | 4,404 | 50 | 3,515 |
| Приволжская | 61 | 4,181 | 45 | 3,541 |
| Восточно-Сибирская | 16 | 3,862 | 11 | 3,734 |
| Забайкальская | 17 | 3,353 | 11 | 3,075 |
| Красноярская | 21 | 3,193 | 13 | 2,846 |
| Донецкая (2022, greyed) | 55 | 1,366 | 32 | 590 |
| Луганская (2022, greyed) | 23 | 933 | 17 | 530 |
| Якутии | 3 | 808 | 2 | 786 |
| Калининградская | 16 | 728 | 13 | 600 |
| Крымская | 15 | 639 | 12 | 543 |
| Мелитопольская-Херсонская (2022, greyed) | 4 | 192 | 4 | 170 |
| ИФР-1 (the Нижнеленинское bridge spur) | 1 | 5 | 0 | 0 |
| **all** | **1,180** | **89,329** | **898** | **76,776** |

(The Donetsk sheet's 1,366 km already leaves out its node lists; most of the rest of the gap
there is Ukrainian-held: Kramatorsk, Sloviansk, Lyman, Kostiantynivka.)

**Checks.** Against the tariff guide's own km (every line, `km_official`): median 0.995 over
887 lines of 2 km or more; 153 off by more than 5%, nearly all short lines where whole
kilometres matter (3.5 km of track for 3 tariff km) or a stop placed off its line (Инская —
Среднесибирская 1.06: Мичуринец sits 5 km from where the tariff km put it, and both traces
agree). Outside figures (`REGISTER["ru"]`, 17 lines; Wikidata P2043 and ru.wikipedia
infoboxes, few as they are; two more, Пинозеро — Ковдор and Первушино — Заволжск, are built
as no line because OSM has no passenger route on them): 12 within 2%, worst deviation 0.35,
and the four off are explained in their notes: Соблаго — Торжок 0.65 (Торжок — Кувшиново
dropped, OSM's trains run only Осташков — Кувшиново), Петрозаводск — Суоярви I 0.93 (its first
9 km are the same pair of points on 01-027 and counted there), Софрино — Красноармейск 1.11
against Wikidata's 15 km (tariff 16, built 16.6), Вырица — Поселок 0.90 (its last point has no
OSM node).

**Timing and size.** `ru_register.py --convert` 5 s; `build_model` 21 min (rinf.build 4.5 min,
2.8 GB; build_credits 15 min over 64,700 sections), `build_tiles` 2.5 min.
`dist/data/ru/lines.json` is 5.5 MB (1.1 MB gzipped; Japan's and China's are 1.2 MB): OSM
suburban lines 2.7 MB, register lines 1.2 MB, trams 0.8 MB, named trains 0.7 MB.
**`credits.json` is the real size problem: 23.5 MB (4.2 MB gzipped; Japan's 0.9 MB)**, of which
13.6 MB are the credits of named trains (a 9,000 km Россия over hundreds of sections credits
every line beside it) and 12.2 MB of OSM lines. A per-region split of lines.json alone would
not help much; trimming named trains' credits, or loading credits per line on demand, would.
Measured, not changed.

### How it was built

```powershell
# sources (no login)
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/tr4_kniga1_2026-09-30.xls https://sovetgt.org/tr4/2026/09/30/Kniga_1_2026-09-30.xls
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/tr4_kniga2_2026-09-29.xls https://sovetgt.org/tr4/2026/09/29/Kniga_2_2026-09-29.xls
curl -L -A "noritetsu-rail-map/1.0" -o data/raw/ru/russia-260930.osm.pbf https://download.geofabrik.de/russia-260930.osm.pbf          # 4.2 GB
curl -L -A "noritetsu-rail-map/1.0" -o data/raw/ru/ukraine-260930.osm.pbf https://download.geofabrik.de/europe/ukraine-260930.osm.pbf # 0.9 GB
curl -A "noritetsu-rail-map/1.0" -o data/raw/ru/ru_boundary.geojson "https://polygons.openstreetmap.fr/get_geojson.py?id=60189&params=0"

$env:OSMIUM_POOL_THREADS=2
python extract.py --region ru/full --pbf data/raw/ru/russia-260930.osm.pbf                              # 7 min
python extract.py --region ru/ua --pbf data/raw/ru/ukraine-260930.osm.pbf --bbox 32.0,45.9,40.3,50.2    # 2 min
python ru_register.py --esr data/raw/ru/russia-260930.osm.pbf          # esr:user codes -> data/raw/ru/osm_esr.json
python ru_register.py --esr data/raw/ru/ukraine-260930.osm.pbf --ua    # + name:ru of stations -> osm_names_ua.json
# (both .pbf deleted after; data/proc/ru/full and data/proc/ru/ua keep what came out)
python ru_register.py --annex        # data/raw/ru/annex.geojson (fetches the two Wikipedia modules once)
python ru_register.py --annex-trains # data/raw/ru/annex_trains.json from the ua build's poizdato pages (2026-10-04)
python ru_register.py --clip         # ru/full + ru/ua -> data/proc/ru, clipped to the outline
python ru_register.py --wikidata     # data/raw/ru/wdx_lines.json, wdx_stations.json
python ru_register.py --convert      # data/raw/rinf/ru/{sections,points,names}.json
python build_model.py --region ru --register rinf:data/raw/rinf/ru    # 21 min
python build_tiles.py --region ru                                     # 2.5 min
python check_model.py --region ru
python probe_ru_wplengths.py > data/raw/ru/probe_wplengths.txt        # the ru.wikipedia figures in REGISTER
```

### What the converter decides (ru_register.py's docstring has the detail)

- **Points** are placed at the OSM node carrying their ESR code (10,466 of 13,096), else at an
  OSM station of the same name near their placed neighbours (1,183), else left unplaced
  (1,447; rinf.py retraces end to end past them).
- **A point is a stop** when Book 2 gives it a passenger operation AND an OSM train route stops
  within 400 m (9,132 points). 3,325 points Book 2 allows passengers at have no OSM route
  stopping there and are junctions.
- **Long stop-to-stop stretches answer to OSM's routes**: a stretch of 10 km or more past such
  an unserved point, or of 25 km or more with no passenger point, gets its end stops cloned as
  junctions on that line only (`<code>@<section>`), so build_model keeps it only where OSM
  passenger routes run over it. 999 stretches, 27,053 tariff km; without this, freight
  bypasses between two served stations (Безенчук — Кинель, the southern bypass) and closed
  branches (Сенная — Аткарск) were stop-to-stop sections nothing questions. build_model then
  kept 1,103 junction-ended sections (26,878 km) and dropped 241 (6,112 km). The clone is
  joined to its stop by a 0 km piece (so the line stays one piece) and given an unplaced stub
  (so it has three neighbours and rinf.py ends a section there); both vanish in the build.
  Until 2026-10-05 nothing rejoined the kept stretch to its stop once build_model had dropped
  the 0 km link, so 304 lines came out in pieces (Лена-Восточная — Хани in 14). rinf.py's
  `split_pieces` hook now folds each such clone back into its stop after the drop: 36 lines
  in pieces, 1,142 clone junction ids gone (HISTORY.md, "2026-10-05").
- Pairs listed on two sections stay on one (main sections first, then the lower id), and a
  pair another section lists with points between is left to that one (61-004 Сенная —
  Трофимовский I is not Сенная — Аткарск). Extra codes ("(эксп.)", "(перев.)", "(стык)") at the
  same km as a neighbour are dropped; the 7 all-0-km node lists (the Moscow Central Circle's
  stations) are left to OSM.
- **The outline** is OSM's own boundary of Russia (relation 60189, which holds Crimea and the
  territorial sea) plus annex.geojson. religiondots' and Natural Earth's outlines put
  Bagrationovsk station, 2 km inside Kaliningrad oblast, in Poland, and religiondots' coast cut
  the beach line at Sochi. Kazakhstan's stretches of Trans-Siberian sections (Петропавловск,
  Кулунда) and every border stub are cut; rinf.py builds the Russian pieces.
- `rinf.py` gained one hook for Russia, `direct_near_m` (1500 in ru.py): an end-to-end retrace
  must pass within 1,500 m of every placed point of the section it replaces. Сенная — Аткарск
  had traced 209 km round by Saratov. 32 such traces refused.

### Crimea and the 2022-annexed railways

- **Crimea** is in Geofabrik's Russia extract and in OSM's Russia boundary: 12 lines, 543 km,
  built like everywhere else (OSM has Crimea's suburban routes).
- **Donetsk, Luhansk, Melitopol-Kherson.** The tariff guide lists the whole pre-war Donetsk
  railway, Kramatorsk, Sloviansk and Lyman included, which Ukraine holds and Ukrzaliznytsia
  serves. Done: (1) Geofabrik's Ukraine extract, cut by `extract.py --bbox` to the four
  oblasts' box, is merged into data/proc/ru by `--clip`; (2) the clip polygon there is
  `data/raw/ru/annex.geojson`, the four oblasts (OCHA COD-AB admin 1) cut to the Voronoi cells
  of the places en.wikipedia's war maps (Module:Russo-Ukrainian war overview map and detailed
  map, CC BY-SA, fetched 2026-10-01 and kept in data/raw/ru/wp_*.lua) mark as Russian-held;
  contested places count as not held. 83% of the four oblasts; Pokrovsk, Siversk, Huliaipole
  in, Kostiantynivka, Kramatorsk, Orikhiv, Kherson city out. (3) Russia's new ESR codes there
  (89xxxx, 84xxxx, 82xxxx) are not in OSM, which carries Ukrzaliznytsia's, so points are
  placed by their Russian name against OSM's `name:ru` (428 of 591). (4) **OSM has no
  passenger route relation in the occupied area** (only Ukrzaliznytsia's on the Ukrainian
  side and Crimea's Армянск trains), so the OSM test would drop every section. First drawn
  greyed (53 lines, 1,291 km), then left out altogether (Anita, 2026-10-01, `ANNEX_RUNNING =
  False`); since 2026-10-04 built again, running where poizdato's trains run and greyed
  elsewhere: "The annexed railways: what runs" below. (5) OSM's station names there are mostly
  Ukrainian; `--clip` puts their `name:ru` in `name` (1,323 stops), keeping the Ukrainian in
  `name:uk`.
- **The outline the app needs**: religiondots' `ru` (which has Crimea) plus
  `data/raw/ru/annex.geojson`. When Ukraine is built, its outline and extract need that area
  taken out.

### Sources and licences

| source | licence | used for |
|---|---|---|
| Тарифное руководство № 4, Books 1-2 (sovetgt.org) | official intergovernmental document, its preface calls the data publicly available | the register, stops |
| OSM via Geofabrik (Russia, Ukraine 260930), OSM boundary relation 60189 via polygons.openstreetmap.fr | ODbL | track, stations, ESR codes, routes, the outline |
| Wikidata | CC0 | English line names (47, only labels that read as English: "Kaliningrad–Sovetsk railway line"; the rest are other languages' transliterations), lengths for check_model |
| ru.wikipedia line articles | CC BY-SA | lengths for check_model |
| en.wikipedia war maps (two Lua modules) | CC BY-SA | which annexed places Russia holds |
| OCHA COD-AB Ukraine admin 1 (maps/data/asia1m/ukraine) | CC BY-IGO | the four oblasts |

### What is still off

- **No independent per-section lengths to speak of**: 17 outside figures. The tariff km are
  the main check, and they are the register's own.
- **3,325 passenger-permitted points no OSM route stops at are not stops**, and the lines over
  them answer to OSM's route coverage. Where OSM's routes are incomplete, real service is
  dropped: Торжок — Кувшиново is dropped because OSM has trains only Осташков — Кувшиново
  (here OSM is probably right). A Yandex Rasp key would settle stops properly.
- **1,447 points are unplaced** (no ESR node in OSM, no name match). Pieces through them are
  retraced end to end; 351 merged sections were rejected outright (biggest: Новый Уренгой —
  Ямбург under construction, Кулунда — Локоть through Kazakhstan, Орск — Рудный Клад).
- **Some parallel tariff sections still overlap** (about 400 km counted twice): Адлер — Роза
  Хутор and Сириус — Роза Хутор share 40 km from the Adler junction; short station-throat
  sections in Samara, Kirov, Novosibirsk. Each is a real tariff section.
- **Station English names** (2026-10-03, "English names and line colours" above): 3,419 of the
  8,297 register stops shipped have one (1,756 before, from OSM alone), and 590 of 1,674
  junction ends; the rest have no English label on Wikidata that passes the check, and show in
  Russian.
- **The annexed railways run only where poizdato's suburban trains run** (2026-10-04, "The
  annexed railways: what runs"); a new long-distance service (Donetsk - Rostov, Melitopol -
  Crimea) would need a source of its own. The control polygon is as current as the Wikipedia
  modules fetched 2026-10-01; delete data/raw/ru/wp_*.lua and rerun `--annex`, `--clip`,
  `--annex-trains`, `--convert` to refresh, and Ukraine's `ua_register.py --outline --clip`
  after it (both builds cut at the same polygon).
- Ten 0 km sections remain on register lines: the link between a stop and its clone where
  OSM's routes kept it (`eRU<code>@<section>`, a junction at the stop's own place). Harmless,
  but they show as junction rows in those lines. rinf.py's log counts the clone stubs among
  "sections left out for a rejected trace" (1,994, of which 1,643 are stubs).
- Line colours (2026-10-03): every register line in its railway's picked colour
  (colours/ru.csv); 151 OSM lines keep OSM's. `line_colours.py` needs no `ru` entry (its
  Wikidata fill matches line names, and no tariff section is a Wikidata item with a colour).

## The annexed railways: what runs (2026-10-04)

Anita, 2026-10-04: "donetsk luhansk to russia makes sense if theres trains run, yeah, lets do
it", under her rule that only track trains run over is drawn as running. So
`ANNEX_RUNNING = True` (rinf_countries/ru.py): the Донецкая, Луганская and
Мелитопольская-Херсонская sheets are built with Russia, inside annex.geojson (the area Russia
holds, unchanged since 2026-10-01). A section trains run over is drawn as running; the rest is
built and greyed as not running (rinf.py's `suspended`, as Slovakia's suspended lines), so a
rider who went before can still record it.

**The source.** poizdato.net lists the Russian-run suburban trains of the occupied area beside
Ukrzaliznytsia's (the ua build crawled every page, data/raw/ua/poizdato/). `ru_register.py
--annex-trains` keeps the trains with at least two calls, and at least half their calls, at
OSM stations inside annex.geojson, and at least 9 running days in the pages' October-November
calendar: **31 trains**, all of the Donetsk and Luhansk railways' suburban service, copied into
`data/raw/ru/annex_trains.json` (so a refresh of the ua crawl does not move this build). None
calls in Zaporizhzhia or Kherson oblast. Spot-checked against Yandex Rasp the same day, which
shows the same trains with today's operational changes: Donetsk - Ilovaisk 6001-6010 and
Donetsk - Uspenskaya 6119/6120 daily, Ilovaisk - Torez 6602/6604 daily, Debaltseve - Rodakove
6421 daily, Luhansk - Starobilsk 6490/6491 Wednesday, Thursday, Saturday and Sunday; Yandex has
nothing Luhansk - Rodakove, Luhansk - Debaltseve or Donetsk - Volnovakha, nor does poizdato.

**No long-distance train** runs into the annexed railways: crimea.ria.ru (30 May 2025) quotes
the DPR transport minister that only freight and suburban trains run and long-distance
passenger trains wait on "security measures"; the DPR head has since said trains to other
Russian regions are expected by 2028 (donetsk.kp.ru). FPK's and Grand Service Express's Crimea
trains ("Таврия") run over the Kerch bridge, not through Melitopol.

**Melitopol - Kherson (82-001..004) is greyed whole.** Test trains Melitopol - Dzhankoi ran
from 1 July 2022 (iz.ru, ria.ru, June 2022) and stopped; ria-m.tv (29 May 2025): "Пассажирское
сообщение, несмотря на многочисленные обещания, так и не запущено", only freight runs through
Melitopol; tutu.ru on 2026-10-04: 0 trains Melitopol - Dzhankoi; Yandex Rasp: no train at
Melitopol; poizdato: none.

**How the calls become track** (`annex_evidence`). Each call is matched to a Book 1 point of
the three sheets by name (the same consonants, `skel`: Ukrainian "Старобільськ" and Russian
"Старобельск" are both "стрблск"; "1118 Км" to the point named for that kilometre) or by place
(within 400 m of the OSM station of that Ukrainian name); of several candidates the one nearest
along the track to the train's previous call. The pairs on the shortest path between two
matched calls are run over (at most 2x the crow-fly distance + 5 km and 40 tariff km). Points
trains call at are the stops on that track; off it, Book 2's passenger points stay stops so the
greyed lines keep their stations. A tariff section trains run over in part becomes two lines:
the part they run over under its id, the rest under `<id>~` with the same name, greyed. (The
front-line cuts already make several lines of one section, rinf.py's `#1`, `#2`.)

Three hand entries in ru_register.py, each with its reason there: `ANNEX_PLACE_ALIAS` (Book 1
names OSM knows by another: Торез is Чистякове since 2016, Ясиноватая is OSM's "Ясинувата
Захід", Луганск-Северный has only the Ukrainian name, Book 1 misspells Амвросиевка),
`ANNEX_CALL_ALIAS` (the trains' "Ясинувата-Західна" is Ясиноватая) and `ANNEX_ON_PAIR`
(Щебенка, where the Нижнекрынка branch leaves, is on no pair of 89-013). And one link,
`ANNEX_LINKS`: Квашино (стык), the end of 89-018, to Успенская, Russia's station across the
pre-2014 border, ~10 km of track that Book 1 lists on no section (89-018 ends at the
inter-railway junction, 51-007 at "Успенская (эксп.)" placed at the station); the Donetsk -
Uspenskaya trains run over it, so it is on 89-018.

Placement improved on the way: the unique-name pass looked all over Ukraine's ANNEX_BBOX and
put "Донец" (84-014, by Вергунка) at Donets station near Balakliia and the numbered halts
"Ост. пункт 12 км", "143 км", "805 км" near Kremenchuk, cutting 84-014, 84-001, 84-008 and
84-012 in two. Placements far from both neighbours are now undone (8), and points with only a
Ukrainian name in OSM are placed by their consonants (77) or the alias (4): 501 of the 591
annexed points placed, 90 not.

**Line by line, what runs** (tariff km of the pairs; trains by number, all daily unless said):

| section | running | greyed | trains |
|---|---|---|---|
| 89-007 Ясиноватая — Донецк | 14 | | every Donetsk train: 6001-6010, 6119-6122, 6303-6306, 6307/6308, 6503-6506, 6703/6704, 6707-6720 |
| 89-016 Криничная — Ясиноватая | 13 | | Donetsk - Ilovaisk, - Uspenskaya, - Yenakiieve, - Makiivka; 6012 |
| 89-017 Криничная — Иловайск | 27 | | 6001-6010, 6119-6122, 6012 |
| 89-018 Иловайск — Квашино (стык), + the link to Успенская | 41 + 10 | | 6119-6122 Donetsk - Uspenskaya |
| 89-003 Ларино — Иловайск, 89-006 Ларино — Ясиноватая | 27, 33 | | 6931-6934 Ilovaisk - Yasynuvata via Mospyne and Donetsk-2 |
| 89-008 Донецк — Рутченково | 10 | | 6741, 6743 Donetsk - Dolia; 6307/6308, 6703/6704 from Dolia |
| 89-001 Рутченково — Волноваха | 10 (to Доля) | 38 (Доля - Волноваха) | the Dolia trains |
| 89-013 Углегорск — Криничная | 23 (Криничная - Енакиево) | 14 (Енакиево - Углегорск) | 6301, 6303-6306 Yenakiieve |
| 89-014 Щебенка — Разъезд 5 км | 12 (to Нижнекрынка) | 5 | 6301 runs up the branch to Nyzhnia Krynka and back |
| 89-020 Иловайск — Чернухино | 49 (to Торез) | 37 (Торез - Чернухино) | 6602, 6604 Ilovaisk - Torez |
| 89-029 Дебальцево — Депрерадовка, 84-001 Родаково — Депрерадовка | 8, 48 | | 6421 Debaltseve - Rodakove |
| 84-014 Луганск — Ольховая | 19 (to Кондрашевская) | 18 (to the border junction) | 6412/6413 Luhansk - Kindrashivska-Nova; 6490/6491 |
| 84-013 Кондрашевская-Новая — Кондрашевская | 4 | | 6412/6413, 6490/6491 |
| 84-012 Кондрашевская-Новая — Граковка | 90 (to Старобельск) | 103 | 6490/6491 Luhansk - Starobilsk, 4 days a week |
| 84-011 Семейкино-Новое — Кондрашевская-Новая | 1 (Локомотивный - Кондрашевская-Новая) | 46 | 6491 calls at Локомотивний |
| the other 69 sections, Melitopol - Kherson's four included | | all | none |

In tariff km: 439 running (the link included), 1,941 greyed. **As built: 81 register lines,
2,140 km on the three railways; 17 lines (437 km) running, 64 lines (1,703 km) greyed** (by
railway: Donetsk 12 running, 275 km / 42 greyed, 858 km; Luhansk 5, 162 km / 18, 653 km;
Melitopol - Kherson 0 / 4, 192 km). Every running line's trace is within 1.1 km of its tariff
km (Луганск — Ольховая's piece 17.9 of 19), but Иловайск — Квашино with the link, 49.5 against
51.2 (the link's crow-fly x 1.15 is a guess). Greyed lines lose what OSM has no track for
(unplaced points, track lifted).

**The frontline track.** OSM's mappers have retagged the frontline railways `railway=disused`
since 2022, and extract.py keeps no disused track, so Lysychansk - Svatove, Popasna, Bakhmut,
Avdiivka had no rails and were not drawn even greyed. `--clip` now puts back the main-line
disused ways mostly inside annex.geojson from the ua build's `data/raw/ua/osm_disused.pkl`
(`ua_register.py --disused`, 1,051 ways), tagged `noritetsu:osm_railway=disused` as Ukraine's
are: greyed km 1,110 -> 1,703 in the trial. One stale route relation came back over them
(Ukrzaliznytsia's unnamed r3478672, 3.6 km): `rules/ru.py` SKIP_ROUTES.

**The line of control** (checked on the trial build against dist/data/ua): no Russian register
track lies more than ~400 m outside annex.geojson + Russia, no Ukrainian register track more
than ~400 m inside annex.geojson, and no Russian register track lies within 30 m of Ukrainian
register track: the two builds cut at the same polygon (annex.geojson unchanged since
2026-10-01; ua clipped 2026-10-03). Nothing is drawn across the front: a tariff pair with a point
on each side is in neither build, and no train runs over it (all such lines on both sides are
greyed: Kostiantynivka, Pokrovsk, Lyman, Kupiansk - Svatove, Kherson - Snihurivka, Polohy).
Where both sides' tracks stop short of the polygon edge, the gap is no-man's-land track OSM has
no rails for, or a pair straddling the line. The one crossing trains use is the old state
border at Квашино - Успенская (ANNEX_LINKS), Russia on both sides.

## Abkhazia: the Psou crossing (2026-10-04)

Abkhazia is now its own region (xa, caucasus_register.py); borders.py EXTRA has its point
`eXARUPSOU` at the Psou bridge. `BORDER` gives 51-032 (Туапсе-Сортировочная — Веселое) a piece
from Веселое to it, "at": Book 1 ends 51-032 at "Веселое (эксп.)", 2 km past Веселое. Built: the
line 113.3 -> 114.9 km (2.0 tariff km, 1.6 km as the crow flies), and build_model's own tails take
three OSM trains to the point too (479А/480С and 304М/304С to Sukhum, the «Диоскурия» electric
train), +1.6 km each, so the Adler - Sukhum crossing joins xa's line there.

## Borders with Belarus and Kazakhstan (2026-10-03, night)

Belarus and Kazakhstan were built the same evening and end their lines at points in
`borders.EXTRA` ("eBYRU...", "eXKZRU..."). Russia's lines now run on to the same points
(`ru_register.BORDER`, a piece from the last Russian point to the border point, junction-ended,
so it stays only where OSM's passenger routes run over it):

- **Belarus**: each Russian section ends at an export code, the tariff handover at the border
  ("Красное (эксп.)" 3 km past Красное). Those codes are not in OSM and were placed by name
  at their station, so the last pair was rejected (Красное, Злынка, Клястица: 0 m of track for
  3-6 km) or traced as a 10 km loop (Рудня) or ran to the station (Невель II - "Завережье
  (эксп.)"). The piece now replaces that pair, with its tariff km.
- **Kazakhstan**: where the pair crosses, its tariff km are shared by crow-fly distance, as
  casia_register shares them on the far side; Озинки, Исилькуль and Локоть by the section's own
  km or crow-fly x 1.2. The crossing pair itself stays as it was, so no section splits into new
  pieces and no line id moves.
- **Iletsk.** Kazakhstan's railway runs Iletsk I and the lines from it to the border on
  Russian soil, and Book 1 lists them only on its own sheet, so 150 km of Russian track was in
  no build. `FOREIGN` reads those two sections (68-001 to Uyutny, 68-003 to Kos-Aral) and keeps
  their pairs placed inside Russia; operator Қазақстан темір жолы, teal. RZD's 80-030 Orenburg -
  Kanisai ended at "Канисай (рзд) (эксп.)", 8 km past Kanisai, which is Iletsk I (`ALIAS`).
- **--clip** keeps OSM's track up to each of these points: a way mostly abroad used to be cut
  whole (Ezerishche, Ozinki, Petukhovo, Isilkul, Kulunda and five more had no track to the
  border), and one kept whole ran on into Kazakhstan (Черлак's two 15 km ways, 9 km abroad,
  so the piece's track belonged to no line). Each way over a crossing now ends at a new node on
  the border point (ids from 10^15 in coords.npz).

What came out (tools/ab.py against the build before; register lines and km):

| crossing | Russian line | piece | joins Belarus/Kazakhstan's |
|---|---|---|---|
| Osinovka | 17-066 Смоленск — Красное | Красное - border 3.2 km | yes |
| Zaolsha | 17-067#2 Смоленск — Рудня | Рудня - border 10.1 km (was a 10.5 km loop to "Рудня (эксп.)") | yes |
| Ezerishche | 01-072 Невель II — Завережье | Невель-2 - border 19.4 km (tariff 21; was 9.3 km to the station) | yes |
| Alesha | 01-073 Невель I — Клястица | Клястица - border 6.6 km | yes |
| Zakopytye | 17-052 Унеча — Злынка | Злынка - border 5.5 km | yes |
| XKZRU01 | 61-017#1 (new, 14.2 km: Верхний Баскунчак - border) | | yes |
| XKZRU02, 04 | 61-017#2 | Эльтон - border 20.1 and 20.9 km | yes |
| XKZRU03 | 61-017#3 | Кайсацкая - border 19.6 km | yes (kz has an OSM line there) |
| XKZRU08 | 68-003 Илецк I — Кандыагаш (new, 90.5 km) | Кос-Арал - border 3.3 km | yes |
| XKZRU15 | 80-011 Утяк — Петропавловск | Петухово - Горбуново - border 22.6 km | yes |
| XKZRU16 | 83-002 | Исилькуль - border 18.7 km | yes |
| XKZRU17 | 83-018 Иртышское — Осолодино | Черлак - border 6.0 km | yes (kz has no line there) |
| XKZRU18 | 83-018 | Теренгуль - border 10.8 km, and Теренгуль - Осолодино (51 km, new) | yes |
| XKZRU20 | 83-014 Барнаул — Локоть | Веселый Яр - border 15.0 km | yes |
| XKZRU05 Kigash, 06 Ozinki, 07 Uyutny | 61-074, 61-072, 68-001 | none: no OSM passenger route over them | no |

80-030 Orenburg - Iletsk I is now 76.3 km (68.7 before, to Kanisai) and joins 68-003 at
Iletsk-1, whose id moves from OSM's n10679322595 to the register's e10679322595 (aliased).
Ids gone, all junctions: three of the export codes (Рудня's aliased to Рудня, Завережье's and
Канисай's to nothing), Полынный and Ингеловский (61-017#2's ends before, merged into its
pieces now), Локоть (эксп. на Рубцовск) (aliased to OSM's Lokot). No line id moved; 17-067's
two pieces keep theirs (the border point's op is "border/<id>" so it does not sort first).
Register lines 845 -> 847, km +287 (68-003 90, Теренгуль - Осолодино 51, Петухово - Горбуново
21, the Iletsk link 8, the rest border pieces). 35 OSM lines (FPK trains, the Smolensk - Vitebsk and
Velikiye Luki - Alesha diesels) get build_model's own tails to the new points.

## English names and line colours (2026-10-03)

**Station English names: Wikidata by ESR code.** The tariff guide's six-digit codes are
Wikidata's P2815 ("ESR station code"; P2814 is a Danish company register). `ru_register.py
--wikidata-stations` fetches every item with a P2815 in ten queries, one per first digit (2-7 s
each; the single query over all of them is near the service's time limit), with its `en`,
`en-gb` and `mul` labels and en.wikipedia title, plus the 157 Russian items that have only an
Express code (P722), joined to ESR through osm.sbin.ru's esr.csv (5 of them are stops here).
Of the 9,128 points that are stops, 8,336 (91%) have a Wikidata item by code, but only 3,246
have an English label. `--convert` keeps a label only where it reads as a romanisation of the
point's Russian name (`en_score` >= 0.88: transliterated, adjective endings and words such as
Пассажирский / Passenger left out on both sides, numbers equal), after cleaning ("Kirov
railway station" -> "Kirov"). It turns away about 400 labels of 3,642 on all points: Tatar
and Bashkir names filed as English ("Tügäräk Qır" for Круглое Поле, "Qaratun"), Dutch and
German transliterations ("Tsjebangda", "Noviy Oergal", "Rakitnaja"), Finnish names, items for
another or a renamed point ("208 km" on Платформа 210 км, "Nauchny park" on Олимпийская
Деревня), and labels that drop part of the name ("Paveletsky" for
Москва-Пассажирская-Павелецкая). Nothing is transliterated by us: a point with no label
that passes has no English name, and the app shows the Russian one. Crimea's labels are
mostly Ukrainian-based romanisations ("Pryberezhne" for Прибрежная, "Vladyslavivka"): kept,
as that is how English sources spell them. The checked names go into points.json as
`name_en`; rinf.py's `station_en` hook (no-op elsewhere) gives them to register stations OSM
gave no `name:en`, checking again against the name the station is shown under (OSM's, where
the stop matched an OSM station). Result, measured on the build (tools/ab.py, then the
rebuild): **register stops with an English name 1,756 -> 3,419 of 8,297 (21% -> 41%)**,
junction ends 79 -> 590 of 1,674, every station in ru 2,717 -> 4,902 of 15,588. Eight
junctions that took an OSM `name:en` from a merged OSM station before now show Wikidata's
spelling instead ("Jessoila" -> "Essoila", "Tuapse-Sortirovochnaya" -> "Tuapse-Sorting").
No station or line id moved; foot.json and ways.json byte-identical.

**Line English names** are built like the Russian ones, from the same two end points' English
names: "Obukhovo — Chudovo-Moskovskoye", with the via note where there is one ("Voskresensk —
Ilyinsky Pogost (via Lopatino, Berendino)"); none unless both ends and every via station have
one. The 47 Wikidata line labels used before all restate the two ends in an editor's spelling
("Railway line Kovrov - Nizniy Novgorod", "Beloostrov - Vuborg line"), so they now only fill
in where an end has no English name, and only where no other line has the same label.
**Register lines with an English name 44 -> 392 of 845 (46%; 54% of register km)**; 409 of
the 1,180 tariff sections from their end stations, 14 from a Wikidata label. The English
names inherit Wikidata's inconsistent spellings ("Yegoryevsk II — Yegorievsk I",
"Vologda I" beside "Krasnodar-1").

**Line colours.** The tariff sections have no colours, nor does any widely used map colour
Russia's railways (Wikimedia's "Russia Rail Map" and "RZD branches area 2018", the maps on
en/ru.wikipedia, draw every railway alike; OSM's route=railway relations carry no colour).
OSM's own lines already have theirs (the metros, МЦД-1..4, Аэроэкспресс, Nizhny Novgorod's
city trains: 151 lines). So every register line takes its regional railway's colour, picked
(`ROAD_COLOUR` in ru_register.py: nine hues, no two neighbouring railways alike), written by
`ru_register.py --colours` into colours/ru.csv, one row per built line, marked `picked`.
Anita's to confirm.

    python ru_register.py --wikidata-stations   # data/raw/ru/wdx_stations.json, ~1 min
    python ru_register.py --convert             # points.json name_en, names.json name_en
    python ru_register.py --colours             # colours/ru.csv from dist/data/ru/lines.json
    python tools/compare_lines.py save ru
    python tools/slot.py -- python tools/rebuild.py -j 1 ru
    python tools/compare_lines.py diff ru
    python check_model.py --region ru

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

## Run (stage-1 probes, as first run)

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

## Open questions (stage 1), and their answers

- The 2022-annexed regions: in, with Russia, de facto (Anita, 2026-10-01). How: "Built" above.
- A Yandex Rasp API key: not asked for; the build relies on OSM's routes. Still the best way to
  settle which points are passenger stops.
- Register unit: the tariff section (Anita, 2026-10-01).
