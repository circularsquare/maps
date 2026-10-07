# Moldova (md): sources, measurements, recipe (built 2026-10-03)

## Built (2026-10-03; Transnistria added 2026-10-04)

**15 register lines, 1,025 km** (one per CFM tariff section, named by its two ends: Romanian in
Moldova, "Ungheni — Chișinău"; Russian in Transnistria, as OSM and the station signs name them,
with OSM's Romanian name as the English one, "Бендеры-1 — Новосавицкая" / "Bender-1 —
Novosavitskaya"; operator Calea Ferată din Moldova), of which **688 km are greyed as not
running** (no train in the timetable), all 85 km in Transnistria among them, and 337 km run:
Ungheni — Chișinău, Bălți — Ungheni, Ocnița — Bălți, Vălcineț — Ocnița and Chișinău — Revaca.
With OSM's lines: 18 lines, 186 stations; the 3 OSM lines are the international named trains
(Prietenia Chișinău — București both ways, UZ's 351Щ to Kyiv). build_model 33 s.

Before 2026-10-04 (Transnistria left out): 14 lines, 933 km, 596 greyed. The Transnistria
build added Бендеры-1 — Новосавицкая (42.3 km, new) and lengthened Chișinău — Бендеры-1 (was
"Chișinău — Cartierul de Nord", 49.7 -> 56.4 km), Bălți-Slobozia — Колбасная (was "— 120 Km",
119.8 -> 154.8) and Бендеры-1 — Căinari (was "Suvorovo — Căinari", 42.7 -> 51.1); line ids
unchanged, 10 stations added, none gone.

**Where the tariff km went.** Book 1's "Млд" sheet: 21 sections, 1,216 tariff km. Left out: 25
km to points abroad or past the border with no node (Яссы-Сокола, export codes), 21 km listed
twice. Kept 1,170; rinf.py traced 1,093 km; build_model dropped junction-ended track no train
runs over (Ocnița's export line to Sokyriany, Etulia's and Cimișlia's export stubs, Lipcani -
Medveja); 1,025 km shipped.

**Checks.**
- Against the network: CFM's network is 1,232 km (2009, en.wikipedia "Calea Ferată din
  Moldova", Transnistria included). Book 1's sheet has 1,216 tariff km; 1,025 km built after
  the export stubs and the dropped freight stubs.
- Against the register's own chainage (`km_official`): median 0.995 over 14 lines; three off by
  more than 5%, all greyed: Chișinău — Căinari 1.71 (Revaca - Gangura traces 22 km where
  Book 1 has 8: the points between have no OSM node and the trace goes round), Suvorovo —
  Căinari 1.07, B.P. 61 Km — Medveja 0.93.
- Outside figure (`REGISTER["md"]`): Ungheni — Chișinău 107.3 against CFM's timetable km (train
  826Г, Ungheni km 0, Chișinău km 107, merstren.md) 1.00. No Wikidata or Wikipedia article
  gives a length for any CFM section.
- The timetable: 16 trains, 143 of 169 calls matched (the rest in Romania and Ukraine beyond
  the stations Romania's and Ukraine's builds have); 19 pairs of consecutive calls found a
  register path; 331 of 993 register km served.

## The short answer

- **The register is the CIS tariff guide**, Tariff Guide No. 4 Book 1, the file Russia's and
  Ukraine's builds read (`data/raw/ru/tr4_kniga1_2026-09-30.xls`), sheet "Млд" (CFM, road 39):
  21 tariff sections, 224 points, 1,216 tariff km, every station, halt and post in order with
  its ESR code and integer tariff km. One section is one register line, as in Russia and
  Ukraine (`bymd_register.py --cc md`, then `rinf.py` through `rinf_countries/md.py`).
- **The timetable is merstren.md's** data file (`orar.js`, "verificat 29.09.2026"; robots.txt
  allows everything): every train in Moldova, CFM's, CFR's and UZ's, with calls, times and
  running days. `bymd_register.py --cc md --timetable` turns it into a GTFS feed
  (`data/raw/gtfs/md/md_merstren.gtfs.zip`) that gtfs_served reads as it reads every
  national feed. Stop coordinates abroad come from CFM's own GTFS (Transitous' rehost,
  `hoermalmeister.github.io/gtfs-rehost/cfm/cfm.zip`, kept as `data/raw/md/cfm_rehost.gtfs.zip`)
  and from Romania's and Ukraine's built stations. CFM's GTFS itself is not used as the feed: it
  has 8 trips, the Kyiv trains with no running day, and none of the trains merstren lists as
  suspended or renumbered (6826/6831 are now 826Г/831Г).
- **What runs in Moldova (merstren, 14.09.2026)**: CFM 826Г/831Г Ungheni - Chișinău (one pair
  daily); CFM/CFR 105Ь/106Ь + IR 401/402 Chișinău - București; UZ 99/100 Kyiv - Bălți -
  Ungheni - București; UZ 351/352 Kyiv - Ocnița - Bălți - Chișinău (Revaca for the airport,
  Kyiv - Chișinău direction only); CFR R 1061/1062 Ungheni - Iași and the Friday-Sunday
  821Ь/824Ь Chișinău - Iași-Socola. Suspended (merstren's list): Chișinău - Odesa (24.02.2022),
  Chișinău - Bender-3 (13.01.2025), Rogojeni - Bălți (13.01.2025), Bălți - Ocnița
  (19.01.2024), Basarabeasca - Zloți, Chișinău - Ungheni's other regional trains, Bălți -
  Ungheni regional trains. merstren lists an Odesa - Chișinău announcement of December 2025 as
  archive only: no such train is in its data.

## Territory: Transnistria (drawn as Moldova's, greyed)

Transnistria's railway (Pridnestrovian Railway, ПЖД) has been run separately from CFM since
2004. No source shows a passenger train there today: CFM's Bender-3 - Chișinău trains were
suspended on 13 January 2025 and Chișinău - Odesa (through Bender and Tiraspol) on 24 February
2022 (merstren.md's suspended list). From 2026-10-03 to 2026-10-04 its track was given to no
country. **Anita, 2026-10-04: "we can gray transnistria if no trains"**: it is Moldova's again,
greyed.
- The outline is OSM's boundary of Moldova (relation 58974) whole (`CC["md"]["minus"]` is
  empty; data/raw/md/outline.geojson, for tools/build_regions.py), and `--clip` keeps
  Transnistria's track. No OSM route relation runs there once the stale 804Ц Bender - Chișinău
  is out (STALE_RELS).
- Book 1's pairs there are in the register: 39-010 Bender I - Tiraspol - Novosavitskaya whole,
  and the Transnistrian ends of 39-004 (Rîbnița - Colbasna), 39-008 (to Bender I) and 39-022
  (from Bender I). 39-011 Livada - Livada (эксп.) (5 km) traces to nothing and is dropped.
- **Greyed by the timetable**, as every other CFM line with no train: merstren's feed has no
  train in Transnistria, so gtfs_served marks those sections not running and the app draws them
  dashed grey and leaves them out of completion. 85 km of track there, all greyed.
- **Operator.** Section rows with both ends in Transnistria carry the manager code "39P"
  (rinf_countries/md.py IM: "Calea Ferată din Moldova; Приднестровская железная дорога"): CFM's
  sheet lists the track and the Pridnestrovian Railway runs it. Only 39-010 takes it as its
  operator. Naming the Pridnestrovian Railway alone made gtfs_served leave 39-010 as
  "unknown" (an operator missing from the feed), drawn as running; with CFM named too it is
  CFM's line to the check, and greyed.
- **Names**: OSM's `name` there is Russian ("Бендеры-1", "Тирасполь", "Колбасная"), as the
  station signs are; that is the name shown. The English name is OSM's name:en, else its
  name:ro ("Bender-1", "Tiraspol", "Cobasna"), the Latin spelling the rest of Moldova uses. A
  line from Moldova into Transnistria takes its Latin end as its own English ("Chișinău —
  Bender-1").

## Lines and named trains (`rules/md.py`)

- **Register lines**: the tariff sections. CFM's only domestic trains (826Г/831Г) have no OSM
  relation; their track counts through the sections.
- **OSM's six route=train relations**: the three international ones are named trains. Three are
  stale and are left out of the clipped extract (`bymd_register.STALE_RELS`): 6931 Bălți
  Slobozia - Ocnița (suspended 2024), 804Ц Bender - Chișinău (suspended 2025, and into
  Transnistria), and Ukraine's "Чернівці - Ларга" over CFM's Lipcani track (Ukraine's trains to
  Larha come from Kamianets-Podilskyi, poizdato.net; none from Chernivtsi through Moldova). As
  OSM routes they kept the timetable from greying the track ("unknown": a route runs there).

## Stops and names

- **Placement** of 218 points: OSM's ESR node 72, Wikidata by ESR code 10, an OSM station whose
  name:ru matches Book 1's 107, an OSM station whose Romanian name matches Book 1's Russian
  rendering of it 11 ("Кишинэу" / "Chișinău", "Бэлць-Слобозия" / "Bălți-Slobozia": `md_latin`
  folds both to one rough Latin form; numbers must agree, so "ОП Кишинэу III" finds
  "Chișinău-3", not Chișinău), unplaced 18 (Russian-only names on closed lines: "ОП Дачи",
  "Разъезд 34 км"; nothing shipped carries a Cyrillic name).
- **Names are Romanian**, OSM's `name`, with editors' bracketed notes left out ("Pelinia
  (localitate-stație de cale ferată)" -> "Pelinia"). No English names: the names are Latin.
- **Stops** as Ukraine's: Book 2's passenger operation and a timetable call or an OSM train
  route stop within 400 m, or a halt on a stretch that has halts. With every train in the feed,
  no halt-free stretch is cloned (`CC["md"]["clone"] = False`): the timetable greys what no
  train runs over.
- **Calls added to merstren's Kyiv trains** so the timetable check finds their path (a
  non-stop run longer than 1.6x crow-fly is stepped over): Vălcineț and Ocnița between
  Mohyliv-Podilskyi and Bălți Oraș, Ungheni and Pîrlița between Bălți Oraș and Chișinău
  (`MD_EXTRA_CALLS`; CFM's own feed lists Ocnița and Pîrlița for 351, and the only track runs
  that way). Revaca is added after Chișinău on 351 (`MD_TAIL`; merstren: "Revaca 10:20,
  shuttle to the airport").
- Line colours: one picked CFM blue (`colours/md.csv`, `picked`).

## Borders

- **Ungheni - Iași** (Romania): ERA RINF's EU00244. A 1.3 km piece from Ungheni station.
  Romania's 600 Făurei - Ungheni reaches the same point (0.25 km, drawn greyed as of its last
  build: Romania needs a rebuild now that Moldova is built).
- **Vălcineț/Otaci - Mohyliv-Podilskyi** (Ukraine): RINF has no point. Proposed borders.EXTRA
  `("eMDUAVALCINET", 27.779499, 48.448785, ["md", "ua"])`, where OSM's track (ways
  41922077/1324001372, the Dniester bridge) crosses OSM's boundary; this build's 0.5 km piece
  from Otaci ends there. Ukraine's side needs a piece too (ua_register.BORDER
  `("MDUAVALCINET", "331904")`, Могилев-Подольский, with BORDER's lookup reading
  borders.load() rather than border_points.json alone). Kyiv trains run here daily.
- No passenger trains: Cantemir/Prut II - Fălciu (EU00245), Giurgiulești - Galați (EU00246),
  Ocnița - Sokyriany, Lipcani (Mămăliga - Larga, UZ's line through Moldova), Basarabeasca and
  Etulia towards Ukraine. No pieces.
- **Transnistria's two crossings into Ukraine**, Novosavitskaya - Kuchurhan and Colbasna -
  Slobidka: no passenger train, so no border point. Book 1's export codes there
  ("Новосавицкая (эксп.)" 392306, "Колбасна (эксп.)" 394803, the handovers 10 and 9 tariff km
  past the stations) are placed where OSM's track crosses OSM's boundary (`EXPORT_AT`:
  29.974995, 46.753474, ways 1037371396/7; 29.252470, 47.809810, way 855238117; from
  `--crossings`), named "Moldova – Ukraine border", so 39-010 and 39-004 run greyed to the
  border (9.7 and 8.9 km). Ukraine's greyed lines end at Kuchurhan (2.1 km east of the crossing:
  Роздільна-Сортувальна — Кучурган) and Slobidka (Подільськ — Слобідка); the track between is
  drawn faint as track no line runs on. Two greyed lines meeting near a border with no shared id
  is what any closed crossing looks like; a shared point would only matter if trains ran.

## How it was built

    # sources (no login; User-Agent "noritetsu-build/1.0 (hobby rail map)")
    #   polygons.openstreetmap.fr/get_geojson.py?id=58974&params=0 -> data/raw/md/md_boundary.geojson
    #   polygons.openstreetmap.fr/get_geojson.py?id=65335&params=0 -> data/raw/md/transnistria_boundary.geojson
    #   www.merstren.md/orar.js -> data/raw/md/merstren_orar.js (bymd_register --merstren)
    #   hoermalmeister.github.io/gtfs-rehost/cfm/cfm.zip -> data/raw/md/cfm_rehost.gtfs.zip
    python extract.py --region md --pbf data/raw/moldova-latest.osm.pbf     # the managing session
    python bymd_register.py --cc md --esr data/raw/moldova-latest.osm.pbf   # 10 s
    python bymd_register.py --cc md --wikidata --outline --clip     # --outline: Transnistria included
    python bymd_register.py --cc md --timetable --convert --colours        # 5 s
    python build_model.py --region md --register rinf:data/raw/rinf/md     # 36 s
    python build_tiles.py --region md                                       # 2 s
    python check_model.py --region md

## Sources and licences

| source | licence | used for |
|---|---|---|
| Tariff Guide No. 4, Books 1-2 (sovetgt.org), sheet Млд | official intergovernmental document, its preface calls the data publicly available | the register, stops |
| OSM via Geofabrik (moldova-latest, 2026-10-03), boundaries 58974 and 65335 | ODbL | track, stations, routes, the outline; 65335 says which rows are the Pridnestrovian Railway's |
| merstren.md's orar.js | a timetable site; robots.txt allows everything; only facts taken | the timetable feed |
| CFM's GTFS, Transitous' rehost | as published | stop coordinates abroad |
| Wikidata | CC0 | station places by ESR code |

## What is still off

- merstren is a third party's compilation ("orar orientativ"); it agrees with CFM's own GTFS
  where both have a train. Its km column is CFM's timetable km, the only outside length.
- 18 points have no OSM node and only Book 1's Russian name (all on greyed lines).
- Transnistria has no train in any source; if CFM's Bender trains or Chișinău - Odesa come
  back, merstren.md will list them and a re-crawl (`--merstren --timetable`) ungreys the track.
- The Kyiv trains' added calls (Ocnița, Ungheni) are places they pass; if they do not stop
  there, nothing changes on the map (the calls only guide the path).
