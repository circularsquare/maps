# Mongolia (mn): sources

Built 2026-10-08 (asia agent): see "Build (2026-10-08)" at the end. The survey below was the research for it.

## Survey (2026-10-08)

### What runs

Ulaanbaatar Railway (UBTZ, Ulaanbaatar Tömör Zam, the Mongolian-Russian joint company) runs
all passenger trains, broad gauge (1,520 mm), diesel. From en.wikipedia "Ulaanbaatar Railway"
and "Rail transport in Mongolia" (read 2026-10-08) and the trains OSM maps:

- **Trans-Mongolian main line**, Sükhbaatar (Naushki border) - Darkhan - Ulaanbaatar - Choir -
  Sainshand - Zamyn-Üüd (Erenhot border), 1,110 km in Mongolia. Daily domestic trains UB -
  Sükhbaatar, UB - Darkhan, UB - Zamyn-Üüd, UB - Choir, UB - Sainshand; international UB -
  Irkutsk (305/306, in OSM), Moscow - UB, and UB - Beijing K23/K24 (weekly, resumed 3 June
  2025), UB - Erenhot/Hohhot. Running.
- **Salkhit - Erdenet** (Darkhan - Erdenet branch, about 164 km): daily UB - Erdenet train
  (rail.cc, "Ulaanbaatar to Erdenet", once daily, 13 h 35 min). Running.
- **Darkhan - Sharyn Gol** (about 63 km): trains 604/605 Darkhan - Shariin Gol are in OSM
  (relations 15323034/15323033); frequency not confirmed. Build as running unless the UBTZ
  timetable says otherwise.
- **Ereentsav - Choibalsan** (238 km, the eastern line from Borzya, Russia): en.wikipedia says
  passenger trains "terminate at Chuluunkhoroot (Ereentsav)", so a Choibalsan - Ereentsav
  passenger train exists; frequency not found. The build agent should confirm from the
  timetable; if it is weekly or better, running, else greyed.
- **Bagakhangai - Baganuur** (96 km), **Khar-Airag - Bor-Öndör** (60 km), **Nalaikh**,
  **Sainshand - Zuunbayan - Khangi** (2023, to the Chinese border), **Tavan Tolgoi -
  Zuunbayan** (226.9 km, 2023), **Tavan Tolgoi - Gashuunsukhait** (Chinese gauge): coal,
  fluorspar and oil lines; no passenger train found. Leave out (freight only), unless the
  timetable shows a train to Baganuur (there used to be one).
- **Ulaanbaatar railbus** (city commuter railbus over the main line, from 2016; OSM relation
  19785754 tagged route=tram): status in 2026 not found. Check the timetable; it would be an
  OSM line over the main line, not a register line.

### Sources

| source | gives | licence | where |
|---|---|---|---|
| OSM via Geofabrik `asia/mongolia-latest.osm.pbf` (59 MB) | 2,192 `railway=rail` ways (only 232 named, 364 `usage=main`, 1,792 with `service`, many in yards), 105 stations and halts; `route=railway` 2707161 "Транс-Монголын төмөр зам", 14543392 Bagakhangai - Baganuur, 14599933 Borzya - Bayantümen, three unnamed; trains 305/306 UB - Irkutsk, 604/605 Darkhan - Sharyn Gol | ODbL | `data/raw/mn/survey/osm_route_relations.json` |
| UBTZ's timetable, https://eticket.ubtz.mn/schedule (and https://www.ubtz.mn) | every passenger train with its stops | (not read) | **refused connections from here** (ECONNREFUSED on both hosts, 2026-10-08): Anita's browser; save the schedule page(s) to `data/raw/mn/` |
| en.wikipedia "Ulaanbaatar Railway", "Rail transport in Mongolia"; ru.wikipedia "Железнодорожный транспорт в Монголии" | lines, lengths (main line 1,108-1,110 km; Choibalsan line 237-238 km; network 1,815 km broad gauge) | CC BY-SA | ru article saved: `data/raw/mn/survey/ruwiki_rail_transport_mongolia.wikitext` |

Not usable:
- **The CIS tariff guide (Book 1)**, which kz/uz/by/ge read: it has no UBTZ sheet (sheet
  names checked in `data/raw/ru/tr4_kniga1_2026-09-30.xls`).
- **osm.sbin.ru's ESR tables**: former USSR only, no Mongolia.
- **OSM named track**: 232 of 2,192 ways named; Korea's recipe will not work.
- No GTFS (Mobility Database 2026-10-08: nothing for MN).

### Recipe

A hand line list through rinf.py, as `nafrica_register.py` and `za_register.py` do: each line
is its ordered list of passenger stations (from the UBTZ timetable's stops, matched to OSM's 105
stations by name; OSM's names are Mongolian Cyrillic), traced over OSM's track by rinf.py.

- Register lines: Trans-Mongolian main line (perhaps cut at Ulaanbaatar into Sükhbaatar - UB
  and UB - Zamyn-Üüd, as UBTZ's trains run; the agent decides), Salkhit - Erdenet, Darkhan -
  Sharyn Gol, Ereentsav - Choibalsan (running or greyed per the timetable).
- Border points: Sükhbaatar - Naushki (ru is built; check whether ru's register runs to the
  border there: Book 1's Заб sheet has Наушки), Zamyn-Üüd - Erenhot (cn is built; cn_sources.md
  names Erenhot as waiting for Mongolia), Ereentsav - Solovyevsk (ru).
- Km checks: main line 1,110 (en.wikipedia), Erdenet branch 164, Sharyn Gol 63, Choibalsan
  238 (ru.wikipedia 237).
- Named trains: every UBTZ train is a numbered once-a-day train; all are named trains (option
  B), as in Central Asia.

Expected: 4-5 lines, about 1,500 km. Extract `--station-areas`. Build time: seconds.

### Open

- The UBTZ timetable (manual download above) decides the Choibalsan line, Sharyn Gol and
  Baganuur, and whether the UB railbus runs.
- Romanised station names: Mongolian Cyrillic in OSM; `name:en` coverage not checked.

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region mn --pbf data/raw/mongolia-latest.osm.pbf --station-areas
    python asia_register.py --clip mn             # also drops NOT_SERVICE routes and DROP_STOPS
    python asia_register.py --convert mn
    python build_model.py --region mn --register asia_register:data/raw/rinf/mn
    python build_tiles.py --region mn; python check_model.py --region mn

Reader: `asia_register.py` (lk_register.py's engine), list `MN` in `asia_lines.py`; settings
`rinf_countries/mn.py`, rules `rules/mn.py` (every route=train a named train), colours
`colours/mn.csv` (picked). Built without UBTZ's timetable (its site refuses scripts), from
OSM's routes and the published sources above.

| line | km built | status | check |
|---|---|---|---|
| Trans-Mongolian (Sükhbaatar - Ulaanbaatar), Naushki border - UB | 393.8 | running | with the south half, border to border 1,104.5 against en.WP 1,110 (0.995, path check) |
| Trans-Mongolian (Ulaanbaatar - Zamyn-Üüd), UB - Erenhot border | 710.7 | running | as above |
| Salkhit - Erdenet | 163.0 | running (daily UB - Erdenet) | 164: 0.99 |
| Darkhan - Sharyn Gol | 61.7 | running (604/605 in OSM) | 63: 0.98 |
| Ereentsav - Choibalsan | 235.1 | greyed | 237.5: 0.99 |

Register 1,564 km: 1,329 running, 235 greyed. No OSM lines; the UB - Irkutsk 305/306 trains and
604/605 are named trains.

Decisions:
- **The main line is two lines, cut at Ulaanbaatar**: UBTZ's domestic trains run from UB north
  or south, never through, and one 1,104 km strip would be hard to read.
- **Sharyn Gol running**: OSM maps trains 604/605 Darkhan - Sharyn Gol; nothing says they
  stopped. **Erdenet running**: rail.cc sells a daily UB - Erdenet train.
- **Choibalsan greyed**: en.wikipedia says passenger trains "terminate at Chuluunkhoroot
  (Ereentsav)", but no timetable or frequency was found (a search on 2026-10-08 turned up
  nothing). The managing session's rule for Mongolia is to grey what cannot be confirmed. Its
  border section (Ereentsav - border) is not drawn: build_model drops a junction-ended section
  of a greyed line, which no route runs over.
- **Left out**: Bagakhangai - Baganuur, Khar-Airag - Bor-Öndör, Sainshand - Zuunbayan - Khangi,
  Tavan Tolgoi - Zuunbayan and - Gashuunsukhait (freight only), and the **Ulaanbaatar railbus**
  (route 19785754 in NOT_SERVICE; no evidence it runs in 2026). Its halts on the main line in
  the city are dropped from the stops (`DROP_STOPS`), so the main line does not stop at them.
- **Borders**: `xNaushkiSukhbaatar` (106.096798, 50.334187; OSM way 27366656 over boundary way
  497684885) and `xZamynUudErenhot` (111.946562, 43.690326; the middle of three tracks over
  boundary way 205080589), both drawn and running; `xEreentsavSolovyevsk` (115.742761,
  49.885818) listed but not drawn. Proposed for borders.EXTRA with Russia's and China's sides
  (handoff_notes/asia_build.md); until then the points show their ids as names.
- Station names: OSM's Cyrillic, with `name:en` where OSM has it (about two thirds); passing
  loops ("зөрлөг") that OSM maps as stations are stops, as everywhere with `osm_stops: "all"`.
