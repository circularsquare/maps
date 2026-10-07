# Belarus (by): sources, measurements, recipe (built 2026-10-03)

## Built (2026-10-03)

**75 register lines, 4,931 km** (one per Belarusian Railway tariff section, named by its two
ends in Belarusian: "Мінск-Сартыравальны — Орша-Цэнтральная"; operator Беларуская чыгунка),
of which **98 km on 4 lines are greyed as not running** (Верайцы — Градзянка, Крычаў-1 —
Шасцёраўка, Андрэевічы — Бераставіца, the Бялынкавічы end of Крычаў-1 — Бялынкавічы). With
OSM's lines: 160 lines, 1,097 stations; 38 OSM train lines (3,220 km: regional economy-class
trains, Minsk's city lines, Russia's Smolensk and Pskov-region suburban trains to the border
stations), 27 named trains, the Minsk Metro's 3 lines, 17 tram lines, the children's railway.
by.pmtiles 1.3 MB; build_model 41 s, build_tiles 9 s.

**Where the tariff km went.** Book 1's "Бел" sheet: 87 sections, 5,330 tariff km. Left out: 74
km listed on two sections, 47 km to export points at the borders (no node; the line ends at
its last station and a border piece carries it on), 5 km listed finer elsewhere. Kept 5,204;
rinf.py traced 5,066 km; build_model dropped 135 km of junction-ended track no train runs
over (freight branches: Калій III — Калій IV, Чашнікі — Навалукомль, the airport branch
Смалявічы — Нацыянальны аэрапорт "Мінск", Ксты — Наваполацк, Лупалава — Задняпроўская...).

**Checks.**
- Against the network: Belarusian Railway's operating length is about 5,470 km (en.wikipedia,
  "Belarusian Railway"). Book 1 has 5,330 tariff km; 4,931 km built, the gap being freight
  branches and the border stubs.
- Against the register's own chainage (`km_official`, every line): median 0.994 over 73 lines
  of 2 km or more; 9 off by more than 5%, all short station-throat sections in Brest, Gomel
  and Polotsk (whole tariff kilometres over 2-3 km), Касцюкоўка — Бярозкі 0.90, and Крычаў-1 —
  Шасцёраўка 0.74 (greyed; its last 11 tariff km run to the export point).
- Outside figures (`REGISTER["by"]`, Wikidata P2043 of pl.wikipedia's line articles):
  Гродна — Масты 57.9 of 58.1 (1.00), Ліда — Беняконі 42.4 of 42.6 (0.99), Ліда — Баранавічы
  Цэнтральныя 104.8 of 105.5 (0.99), Варапаева — Друя 88.5 of 88.9 (1.00), Гродна — Брузгі
  20.5 of 21.8 (0.94); two whose figure runs on to the Polish border, built to the last
  station: Брэст-Цэнтральны — Высока-Літоўск 0.85, Брэст-Палескі — Хаціслаў 0.87.
- OSM lines against published lengths (proposed `KNOWN["by"]`, check_model's KNOWN is
  shared): Minsk Metro Маскоўская лінія 19.1 (Wikidata 19.2), Аўтазаводская лінія 18.1
  (18.1); Зеленалужская лінія 7.7 (Wikidata's 17.2 is the planned line, not what runs).
- The timetable: 800 train pages, 766 trains matched (13,889 calls, 12,809 placed);
  gtfs_served found a register path for 1,125 distinct consecutive-call pairs and stepped
  over 172 calls (stations in Russia, Lithuania's transit stops, halts the register has inside
  a section); 4,722 of 5,066 register km served.

## The short answer

- **The register is the CIS tariff guide**, Tariff Guide No. 4 Book 1, the file Russia's and
  Ukraine's builds read (`data/raw/ru/tr4_kniga1_2026-09-30.xls`), sheet "Бел" (road 13): 87
  tariff sections, 988 points, every station, halt and post in order with its ESR code and
  integer tariff km. One section is one register line (`bymd_register.py --cc by`, then
  `rinf.py` through `rinf_countries/by.py`). OSM Belarus carries the ESR code on 956 nodes,
  which place 903 of 986 points.
- **Book 1 never names Minsk-Passazhirsky**, the main station: Minsk's sections end at
  Minsk-Sortirovochny. It is inserted (`bymd_register.INSERT`, OSM's ESR node 140210, Book 2
  gives it passenger operations) on 13-090 to Orsha between Institut Kultury and
  Minsk-Vostochny, and as the end of 13-087 from Krizhovka and Minsk-Severny, whose tariff
  route (5 km for 2.7 crow-fly) runs through it.
- **The timetable is poezdato.net's.** Belarusian Railway's own site (pass.rw.by) answers 403
  ("access restricted") from here; rw.by times out. No open feed exists (none in the Mobility
  Database or Transitous). The Wayback Machine has about 40 of pass.rw.by's 2026 train pages.
  poezdato.net, the Russian-language sister of the poizdato.net Ukraine's build crawled, lists
  Belarusian trains with their calls and a running calendar for October-November 2026;
  robots.txt allows the pages. `--crawl` reads station pages, starting from 25 hubs and
  following linked stations whose names are Book 1's (83), then the stations at the ends of
  tariff sections that the trains call at (26 more), and every train page they list (800),
  1.2 s apart; the site answered in about 4 s a page, so the crawl took about 80 minutes.
  `--timetable` turns the pages into `data/raw/gtfs/by/by_poezdato.gtfs.zip`.

## Lines and named trains (`rules/by.py`)

Belarusian Railway sells everything as a "line" of some class (regional and interregional,
economy and business, city lines). The project's rule decides instead, as for Ukraine and
Russia next door:
- **Lines**: the four-digit regional economy-class trains (6xxx), Minsk's city lines (CL,
  7xxx: Мінск — Рудзенск, Чырвоны Сцяг, Беларусь), Russia's ЦППК Smolensk - Orsha/Vitebsk and
  СЗППК Pskov-region trains, unnumbered "A - B" routes; the Minsk Metro, trams, the children's
  railway.
- **Named trains**: every train numbered 1-999: long-distance and international (001Б
  «Беларусь», the Moscow «Ласточка»s 717-722, the Kaliningrad and St Petersburg trains),
  interregional business and economy class (7xx, 6xx: Мінск — Гродна 629Б/731Б), and regional
  business class (8xx, "RLb": Мінск — Орша 861Б-868Б, mapped one relation per train). A
  route_master with no number of its own is a named train when every train under it is
  (`SERVICE_IF_ALL_ROUTES_ARE`). The interregional business class were the closest call
  (several pairs a day Minsk - regional capitals): named trains, as Ukraine's Інтерсіті+ and
  Russia's numbered «Ласточка»s.
- **One line per service, not per train**: OSM maps the Grodno and Lida regional trains one
  relation (and one route_master) per train, "Цягнік №6252: Гродна => Ліда", so each train
  built as its own line. `--clip` groups the four-digit per-train relations between the same
  two places under one route_master of its own ("Гродна — Ліда", id 9e15 + a hash of the two
  names, stable across builds): 24 relations, 5 lines.
- **Stale OSM relations left out** (`STALE_RELS`): Krichev - Shesterovka (no train calls at
  Shesterovka), Praha - Moskva and Rīga - Minsk (no such trains).

## Stops and names

- **Names are Belarusian**, OSM's `name` of the point's ESR node (893), else an OSM station
  matched by name (Russian name:ru, 73), else Wikidata's be label (2), else Book 1's Russian
  spelling (17, posts and passing loops). English names (60) from OSM's name:en or Wikidata's
  label, kept only where they read as a romanisation of the Russian or Belarusian name
  (strict: 0.88), so mixed spellings ("Jlobin", "Maladziečna train") stay out.
- **A point is a stop** when Book 2 gives it a passenger operation and a train calls there
  (871 by the timetable) or it is a halt on a stretch with halts. A halt-free stretch of 8 km
  or more between two called stations that no train runs over in one hop is cloned as
  junctions so the timetable decides (7 stretches, 89 km).
- Line colours: one picked colour per Belarusian Railway branch (отделение), told by the first
  three digits of the section's first ESR code (`BY_BRANCH`; colours/by.csv, `picked`):
  Brest blue, Baranovichi red, Minsk teal, Gomel orange, Mogilev purple, Vitebsk green.

## Borders

`bymd_register.BORDER`: a piece from the line's last station to the border point.
- **Lithuania**: Гудогай - Kena, ERA RINF's EU00250 (9.4 km). The Moscow/St Petersburg/Adler/
  Chelyabinsk - Kaliningrad trains transit here; Lithuania's build already reaches EU00250
  (Kyviškės — Kena). Benyakoni (EU00251): trains run Lida - Benyakoni only, no piece.
- **Russia**: RINF has no points. Five crossings where OSM's track crosses OSM's boundary of
  Belarus, proposed for borders.EXTRA, each with trains over it: Osinovka - Krasnoye (Minsk -
  Moscow, Orsha - Smolensk), Zaolsha - Rudnya (Vitebsk - Smolensk), Ezerishche - Nevel (St
  Petersburg - Vitebsk/Gomel/Brest), Alesha - Klyastitsa (Pskov-region trains to Alesha),
  Zakopytye - Zlynka (Minsk - Adler and Anapa). Russia's register ends at its last stations
  (Красное, Рудня, Завережье, Клястица, Злынка, sections 17-066, 17-067/17-203, 01-072,
  01-073, 17-052) and needs pieces of its own to these points. No trains at Krichev -
  Shesterovka or Belynkovichi - Unecha: no points.
- **Poland, Latvia, Ukraine**: no passenger trains (tutu.ru: no Brest - Terespol trains; no
  Belarusian train to Poland, Latvia or Ukraine in poezdato.net). No pieces.

## How it was built

    # sources (no login; User-Agent "noritetsu-build/1.0 (hobby rail map)")
    #   polygons.openstreetmap.fr/get_geojson.py?id=59065&params=0 -> data/raw/by/by_boundary.geojson
    #   Wikidata: stations with P2815 in Belarus -> data/raw/by/wd_stations.json
    #   poezdato.net station and train pages -> data/raw/by/poezdato/
    python extract.py --region by --pbf data/raw/belarus-latest.osm.pbf     # the managing session
    python bymd_register.py --cc by --esr data/raw/belarus-latest.osm.pbf   # 30 s
    python bymd_register.py --cc by --wikidata --outline --clip             # 1 min
    python bymd_register.py --cc by --crawl                                 # ~80 min, 1.2 s apart
    python bymd_register.py --cc by --timetable --convert --colours        # 1 min
    python build_model.py --region by --register rinf:data/raw/rinf/by     # 41 s
    python build_tiles.py --region by                                       # 9 s
    python check_model.py --region by

To refresh the timetable, delete data/raw/by/poezdato/ (or the pages to refresh) and rerun
`--crawl --timetable --convert`, then rebuild; each page's calendar covers the current and
next month.

## Sources and licences

| source | licence | used for |
|---|---|---|
| Tariff Guide No. 4, Books 1-2 (sovetgt.org), sheet Бел | official intergovernmental document, its preface calls the data publicly available | the register, stops |
| OSM via Geofabrik (belarus-latest, 2026-10-03), boundary 59065 | ODbL | track, stations, ESR codes, routes, the metro and trams |
| poezdato.net train and station pages | an aggregator; robots.txt allows the pages; only facts taken | the timetable feed |
| Wikidata | CC0 | station places and labels by ESR code; line lengths (pl.wikipedia's items) |

## What is still off

- **The timetable is a third party's copy** (poezdato does not say where its data comes
  from); Belarusian Railway's own site is closed to us. Spot checks agree with OSM's routes.
  Its calendar marks some weekend trains on only a few days (618Б Druya - Minsk: 3-4 October);
  Druya - Voropaevo is served by the timetable all the same.
- Палонка and Сверкава (OSM stations trains call at) are no register stations: Book 1 has no
  point for them; their calls are stepped over.
- 7 junction-ended sections (81 km) stay drawn on OSM's routes alone, on no train's shortest
  path (Лучоса — Прыдзвінская 18.6 km, Ваўкавыск — Рось 18.0, Добруш — Закапыцце 11.6...).
- 17 points carry Book 1's Russian spelling (no OSM or Wikidata name).
