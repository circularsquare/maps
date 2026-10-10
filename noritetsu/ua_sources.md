# Ukraine (ua): sources, measurements, recipe (built 2026-10-03)

## Built (2026-10-03)

**318 register lines, 15,733 km** (one per tariff section, named by its two ends in Ukrainian:
"Чаплине — Покровськ"; operator the regional railway), of which **3,359 km on 100 lines are
greyed as not running** (no train in the timetable) and 12,374 km run. With OSM's lines: 511
lines, 4,313 stations (2,737 with an English name); 48 OSM train lines (3,629 km), 25 named
trains, 7 metro lines, 9 light-rail lines, 103 tram lines, the Kyiv funicular. ua.pmtiles
4.4 MB; dist/data/ua 9.3 MB. build_model 100 s, build_tiles 32 s.

By railway (register km built, closed included): Львівська 3,930, Південно-Західна 3,926,
Одеська 3,222, Південна 2,572, Придніпровська 1,686, Донецька 399 (most of the Donetsk railway
is in the annexed area or greyed).

**Where the tariff km went.** Book 1's six Ukrainian sheets: 21,425 tariff km. Left out as
outside Ukraine as drawn: 3,278 km in the annexed area, 588 km in Crimea, 224 km between
unplaced points outside; 223 km listed on two sections, 35 km listed finer elsewhere. Kept
17,077 km. Of those, rinf.py traced 16,649 km (268 pieces rejected, mostly lines through
Belarus or Moldova and dismantled narrow gauge: Калинівка — Андрусове, Чернігів — Семиходи,
Кучурган — Рені), and build_model dropped 917 km of junction-ended track no train or OSM route
runs over (freight bypasses and spurs), leaving 15,733 km.

**Checks.**
- Against the published network: Ukrzaliznytsia's operating length is 19,790 km without
  Crimea and the 2014-occupied Donbas (uk.wikipedia, "Укрзалізниця"); "over 700 km" more have
  been occupied since 2022 (the same article). The tariff guide's own sheets, cut to Ukraine as
  drawn, keep 17,077 km; built 15,733. The gap to 19,000-odd is mostly track with no passenger
  trains at all (dropped), and the occupied area as annex.geojson draws it (which counts
  contested places as not held, so it takes more than UZ's 700 km).
- Against the register's own chainage (every line, `km_official`): median 0.995 over 315
  lines of 2 km or more; 66 off by more than 5%, nearly all short (whole tariff kilometres
  over 2-7 km) or with a stop placed off its line.
- Outside figures (`REGISTER["ua"]`, Wikidata P2043): Батьово — Королево 68.9 of 68.0
  (1.01), Стрий — Івано-Франківськ 107.5 of 108.0 (0.99), Антонівка — Зарічне (narrow gauge)
  105.4 of 106.6 (0.99). Wikidata's other 71 Ukrainian lengths are corridors, not sections.
- OSM lines against published lengths (proposed `KNOWN["ua"]`, check_model is shared): Kyiv
  M1 22.8 (Wikidata 22.64), M2 20.8 (20.95), M3 23.8 (23.86); Kharkiv
  Холодногірсько-Заводська 17.2 (17.3), Олексіївська 11.0 (10.98), Салтівська 10.2 (10.2,
  en.wikipedia); Dnipro 7.0 (7.8 en.wikipedia, which counts to the depot).
- The timetable: 1,086 train pages, 1,064 trains matched (20,755 calls, 19,788 placed);
  gtfs_served found a register path for 2,389 distinct consecutive-call pairs and stepped over
  851 calls (stations abroad, occupied-area trains, halts the register has inside a section).
  10,383 of 16,649 register km served.

## The short answer

- **The register is the CIS tariff guide**, Tariff Guide No. 4 Book 1, the file Russia's build
  already reads (`data/raw/ru/tr4_kniga1_2026-09-30.xls`). It has six sheets for
  Ukrzaliznytsia's regional railways (Ю-Зап (У), Льв (У), Од (У), Южн (У), Придн (У),
  Дон (У)): **499 tariff sections, 4,216 points, 21,425 tariff km**, every station, halt and
  post in order with its ESR code and integer km. One tariff section is one register line,
  as in Russia (`ua_register.py`, then `rinf.py` through `rinf_countries/ua.py`). The sheets
  are kept current enough for renames since 2016 (Покровськ, not Красноармейськ), but not all
  (Красный Луч in the occupied east).
- **Other registers looked at and dropped.** OSM's named track covers 48% of main and branch
  km (`probe_kr_ways.py`), names mostly "A - B" corridor pieces: not a register. OSM's
  `route=railway` relations: 153 in the extract, corridors and some tariff sections. Wikidata:
  74 Ukrainian line items with a length, nearly all corridors ("Kyiv-Poltava line") or narrow
  gauge; used only as a check.
- **No timetable feed exists.** Ukrzaliznytsia publishes none; its sites (uz.gov.ua,
  swrailway.gov.ua, which serves every railway's suburban timetable) time out from here
  (geo-blocked). The Mobility Database lists only city buses and trams for Ukraine. Transitous
  republishes a generated Ukrzaliznytsia feed (jbb.ghsq.de) that holds only 123 trips, the
  international trains and some Intercity. **poizdato.net** lists every train, suburban and
  long-distance (846 + 240 pages in its sitemap, which a sample of 86 station pages showed to
  be complete), with its calls and a running calendar for October-November 2026; robots.txt
  allows it. `ua_register.py --crawl-trains` fetches the pages (1.2 s apart), `--timetable`
  turns them into a GTFS feed gtfs_served reads like any other.

## Territory

- **Ukraine as drawn** is OSM's boundary of Ukraine (relation 60199, polygons.openstreetmap.fr,
  `data/raw/ua/ua_boundary.geojson`) less OSM's boundary of Russia (relation 60189, the file
  Russia's build uses, which holds Crimea) less `data/raw/ru/annex.geojson` (the 2022-annexed
  oblasts where Russia holds them, from Russia's build). `data/raw/ua/outline.geojson` is that
  outline, for `tools/build_regions.py` (religiondots' `ua` still holds the annexed area).
- The extract is clipped to it, reaching 0.03° into Poland, Slovakia, Hungary, Romania and
  Moldova so crossings keep their track (`ua_register.py --clip`, ru_register's rules: a way
  goes if half its nodes are outside). Tariff pairs with a point outside are left out.
- **No border joins with Russia or Belarus** (no passenger trains). Moldova (built
  2026-10-03) joins at Могилів-Подільський - Otaci ("Borders" below); Odesa - Izmail through
  Basarabeasca and Kuchurhan - Reni through Moldova are cut at the border, the Ukrainian
  pieces standing alone.

## Lines and named trains (`rules/ua.py`)

- **Register lines**: the tariff sections (as Russia). Their track carries every train.
- **Lines from OSM**: suburban trains (приміський поїзд, електричка; Ukrzaliznytsia numbers
  them 6000-7999), Kyiv's city electric train (Міська електричка, routes A and Б), unnamed
  "A - B" routes, MÁV's Fehérgyarmat - Záhony; the Kyiv, Kharkiv and Dnipro metros; Kyiv's and
  Kryvyi Rih's швидкісний трамвай (light rail); trams; the Kyiv funicular; children's railways.
- **Named trains**: every train numbered 1-999 (Інтерсіті+ 7xx, night trains, the regional
  8xx such as Kharkiv - Izium 811/812, local 6xx such as 687 Арциз - Березине, the
  international ones: Оберіг, Київ-Експрес, TLK Wisłok to Rava-Ruska, ZSSK's Zakarpatia),
  and anything OSM tags long_distance, night, international or intercity. One pair a day on a
  long route is a train, not a line a rider uses; its track counts through the tariff
  sections. The regional 8xx were the closest call (one or two pairs a day, regional
  distances): named trains, as Russia's numbered trains are.
- Two OSM `route=train` relations are no passenger train and are flagged named trains so no
  total counts them: "дільниця 40-062 «Миколаїв — Миколаїв-Вантажний»" (a tariff section
  mapped as a route) and the Novokostiantynivka uranium mine branch.

## Wartime service: what runs and what is greyed

Decided by the timetable (`data/raw/gtfs/ua/ua_poizdato.gtfs.zip`, read by gtfs_served as
every national feed is), by the same rules as everywhere else: a stop-to-stop section no
train runs over is greyed as not running; a junction-ended one no train or OSM route runs over
is dropped.

- **OSM marks the frontline railways `railway=disused`** (Kramatorsk, Sloviansk, Lyman,
  Kupiansk, Pokrovsk, the Nikopol bank of the Dnipro, Kherson - Snihurivka; 3,082 ways), and
  extract.py keeps no disused track, so those tariff sections had no rails to trace and
  vanished. `ua_register.py --disused` reads the main-line ones from the .pbf (not spurs,
  sidings, yards, crossovers or tram gauge) and `--clip` puts 1,936 inside Ukraine back as
  track (tagged `noritetsu:osm_railway=disused`). They are then traced and greyed where no
  train runs. OSM is not always right the other way: the Kherson line was disused in OSM while
  trains run Mykolaiv - Kherson (40-049, served).
- **Greyed, 3,359 km on 100 lines.** The largest: Апостолове — Нижньодніпровськ-Вузол
  (161 km) and Запоріжжя-Ліве — Апостолове (119, the Nikopol - Marhanets bank), Огірцеве — and
  Тропа — Куп'янськ-Сортувальний (104, 68), Лозова — Берестин (100), Слов'янськ — 4 км
  (99, Lozova - Barvinkove - Sloviansk), Берестин — Самар-Дніпровський (98), Христинівка —
  Андрусове (94), Покровськ — Павлоград I (92), Покровськ — Дубове, Чаплине — Покровськ,
  Пологи — Запоріжжя 2, Kramatorsk - Sloviansk - Lyman - Sviatohirsk, Kostiantynivka -
  Kramatorsk, the Sumy and Chernihiv border branches (Хутір-Михайлівський, Глухів,
  Новгород-Сіверський), Kherson - Snihurivka, Odesa - Artsyz via Serpneve (the old route
  through Moldova), and long-closed branches the tariff guide still lists (Ларга — Сокиряни,
  Завалля — Вижниця, Вигнанка — Скала-Подільська, Біла-Чортківська — Бучач). Checked against
  the train pages: Pavlohrad is served via Synelnykove, Berestyn from Poltava and Kharkiv,
  Samar-Dniprovskyi only from Dnipro, so the greyed pieces between are right.
- **Left as they are** (gtfs_served's own categories): 841 km "unknown" (OSM routes run there
  but no train in the feed, nothing calls at the stops), 1,008 km "osm" (junction-ended, OSM
  routes over it), 154 km ambiguous, 24 km to a border point.
- **Rescued**: the Chop - Záhony, Chop - Čierna nad Tisou and Yahodyn - Dorohusk border
  pieces, which no OSM route covered.

**The occupied area, for Russia's `ANNEX_RUNNING`**: poizdato lists daily suburban trains on
the Russian-run railways in occupied Donetsk and Luhansk (Донецьк — Іловайськ, Ясинувата —
Єнакієве, Донецьк — Успенська, Доля — Ясинувата, Дебальцеве — Родакове; numbers 6xxx, 61
running days in October-November 2026). That is a source saying which trains run there; the
pages are in data/raw/ua/poizdato/rozklad-elektrychky/ (6002--donetsk--ilovaisk and so on).
Done 2026-10-04 (Anita: "donetsk luhansk to russia makes sense if theres trains run"): Russia's
build takes the area (ANNEX_RUNNING = True), running where these 31 trains run and greyed
elsewhere; ru_register.py --annex-trains copies their calls into
data/raw/ru/annex_trains.json, so refreshing or deleting the crawl here does not move Russia's
build (rerun --annex-trains after a recrawl to pick up changes). ru_sources.md "The annexed
railways: what runs". Ukraine's outline and clip are unchanged: both builds cut at
data/raw/ru/annex.geojson.

## Stops and names

- **Placement** of 4,172 points: OSM ESR node 2,413, Wikidata item by ESR code 1,007, OSM
  station of the same name near the neighbours 482, unplaced 270 (rinf.py retraces past them).
- **A point is a stop** when Book 2 gives it a passenger operation and a train calls there
  (2,042 by the timetable, more by OSM routes), or it is a halt ("ОП" in Book 1, or an OSM
  `railway=halt` within 400 m) on a stretch that has halts: a passenger line, running or not.
  A stretch of 8 km or more between two called stations with no halt, and no train calling at
  both ends one after the other, is freight track: its ends are cloned as junctions on that
  line (as Russia's), so it answers to the timetable and is dropped where no train runs (115
  stretches, 1,455 km). `rinf_countries/ua.py`'s `stop_name` makes every such stop a station
  even where OSM has no station node (Ірпінь has only unnamed stop positions).
- **Names are Ukrainian**: OSM's name of the point's ESR node (2,375), else Wikidata's uk
  label (1,033), else an OSM station matched by name (487), else Book 1's Russian spelling
  (277, mostly posts and passing loops). English names (2,682) from OSM's name:en or
  Wikidata's label, kept only where they read as a romanisation of the Ukrainian name
  (`en_ok` >= 0.75).
- Line colours: one picked colour per regional railway (colours/ua.csv, `picked`, as
  Russia's): Південно-Західна blue, Львівська red, Одеська teal, Південна orange,
  Придніпровська purple, Донецька green.

## Borders

`ua_register.BORDER`: a piece from the last Ukrainian point to the neighbour's RINF border
point, length crow-fly x 1.2, for Przemyśl (EU00173, from Мостиська ІІ), Werchrata (EU00174)
and Hrebenne (EU00175, both from Рава-Руська), Dorohusk (EU00178, Ягодин), Záhony (EU00193)
and Čierna nad Tisou (EU00162; EU00163 is the same spot), both from Чоп, Halmeu (EU00240,
Дяково), Câmpulung la Tisa (EU00241, Тересва), Valea Vișeului (EU00242, Ділове), Vicșani
(EU00243, Багринівка). Built and joined: Przemyśl, Dorohusk, Záhony, Čierna (register pieces)
and Hrebenne (the TLK Wisłok's OSM route). The four Romanian crossings and Werchrata have no
passenger train (Romania's feed closes its side too) and drop as junction-ended track. Left
out: Uzhhorod - Maťovce (EU00161) and Esen - Eperjeske (EU00192), broad-gauge freight; Russia
and Belarus (no trains).

**Moldova** (2026-10-03, once Moldova was built): Могилів-Подільський - Otaci, on 32-073
Жмеринка — Могилів-Подільський, to `eMDUAVALCINET`, the point on the Dniester bridge the
by-md agent put in `borders.EXTRA` (RINF has none): 1.2 km (1.05 km as the crow flies), the
Kyiv - Chișinău trains. `convert()` now takes the border points from `borders.load()` (RINF's
table plus EXTRA, with MOVE applied) rather than border_points.json alone. Odesa - Izmail
through Basarabeasca and Kuchurhan - Reni are still cut at the border on this side: no
`borders.EXTRA` point there yet.

## How it was built

    # sources (no login; User-Agent "noritetsu-build/1.0 (hobby rail map)")
    #   polygons.openstreetmap.fr/get_geojson.py?id=60199&params=0 -> data/raw/ua/ua_boundary.geojson
    #   jbb.ghsq.de/gtfs/ua-ukrzaliznytsya.gtfs.zip              -> data/raw/ua/uz_jbb.gtfs.zip
    #   Wikidata: stations with P2815 and P17 Ukraine -> data/raw/ua/wd_stations.json;
    #             Ukrainian railway lines with P2043  -> data/raw/ua/wd_lines.json
    #   poizdato.net sitemap -> data/raw/ua/poizdato_sitemap.json (846 suburban, 240 long-distance)
    python extract.py --region ua --pbf data/raw/ukraine-latest.osm.pbf     # the managing session
    python ua_register.py --esr data/raw/ukraine-latest.osm.pbf             # 1 min
    python ua_register.py --disused data/raw/ukraine-latest.osm.pbf         # 3 min
    python ua_register.py --outline --clip                                  # 1 min
    python ua_register.py --crawl-trains                                    # ~70 min, 1.2 s apart
    python ua_register.py --timetable --convert --colours                   # 1 min
    python build_model.py --region ua --register rinf:data/raw/rinf/ua     # 100 s
    python build_tiles.py --region ua                                       # 32 s
    python check_model.py --region ua

Book 1 and Book 2 are the files in data/raw/ru (Russia's build fetched them from sovetgt.org).
To refresh the timetable, delete data/raw/ua/poizdato/ (or the pages to refresh) and rerun
`--crawl-trains --timetable --convert`, then rebuild; each page's calendar covers the current
and next month.

## Sources and licences

| source | licence | used for |
|---|---|---|
| Tariff Guide No. 4, Books 1-2 (sovetgt.org), Ukrainian sheets | official intergovernmental document, its preface calls the data publicly available | the register, stops |
| OSM via Geofabrik (ukraine-latest, 2026-10-03), boundaries 60199 and 60189 | ODbL | track (with the disused frontline track), stations, ESR codes, routes, metros and trams, the outline |
| poizdato.net train pages | an aggregator; robots.txt allows the pages; only facts taken (which trains call where, on which days) | the timetable feed |
| Transitous' Ukrzaliznytsia feed (jbb.ghsq.de) | as Transitous publishes it | international trains |
| Wikidata | CC0 | station places, Ukrainian and English names by ESR code, line lengths |
| data/raw/ru/annex.geojson (Russia's build: OCHA COD-AB and en.wikipedia war maps) | CC BY-IGO, CC BY-SA | the area left out (Russia's build since 2026-10-04) |

Tried and not usable: uz.gov.ua and swrailway.gov.ua (the official suburban timetable for every
railway) time out from here; dp.uz.gov.ua has an expired certificate and only frames
swrailway; the Wayback Machine has only scattered swrailway station pages; egtre.info 403.

## Lines deleted as track to no station (checked 2026-10-08)

`prune_dead_track` deleted three ua ids on 2026-10-07; none runs: Миронівка — Богуслав (17.7
km; uk.wikipedia's station article: closed in 2020 with the whole branch, and poizdato has no
Bohuslav page), Богодухів — Гути and Верхньодніпровськ — Дніпровська (no train calls there on
poizdato). Left deleted.

Арциз — Ізмаїл (1 stop of 8), Вербка — Камінь-Каширський and Одеса-Пересип — Колосівка (an end
left a junction) are the clone fault in rinf.split_pieces (handoff_notes/missing_stops.md):
a stop whose every pair is a stretch left to the timetable gets no station record, so its
clone is never folded back. Fixed by the rinf.py diff there (trialled: Арциз — Ізмаїл 8 of 8
stops); ua needs a rebuild once it lands.

## What is still off

- **The timetable is a third party's copy.** poizdato does not say where its data comes from;
  spot checks (Kyiv's city electric train, Kharkiv's suburban trains, no trains to Kramatorsk
  or Kupiansk) agree with what is known. 45 trains show no running day in October-November
  (seasonal or cancelled) and count for nothing.
- 841 km "unknown" stay drawn as running: OSM routes run there but the feed has no train
  (often stale pre-war OSM routes; 712Д Kramatorsk - Kyiv is still mapped).
- OSM's Kyiv metro route masters are named in English ("M1 line"), so the lines show that
  name; their route relations have the Ukrainian names.
- 277 points carry Book 1's Russian spelling (no OSM or Wikidata name).
- Book 1's Ukrainian sheets are not fully current (Красный Луч, renamed Хрустальний in 2016,
  in the occupied part), but every name shown comes from OSM or Wikidata where one exists.
