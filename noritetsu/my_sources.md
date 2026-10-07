# Malaysia register sources (surveyed 2026-10-03)

What `my_register.py` reads, where each piece came from, the calls made, and what is still off.
Downloads are in `data/raw/my/` (gitignored); this file is the tracked record of them. Nothing
needed a login, key or account.

## Run

```powershell
# Geofabrik's malaysia-singapore-brunei extract (the same file Singapore is cut from)
python tools/slot.py -- python extract.py --region my --pbf data/raw/malaysia-singapore-brunei-latest.osm.pbf --station-areas   # 40 s
python my_register.py --clip           # drops Singapore (built as sg) and the Thai stubs; 2 s
python tools/slot.py -- python build_model.py --region my --register my_register:data/raw/my   # 35 s
python tools/slot.py -- python build_tiles.py --region my                                     # 4 s
python check_model.py --region my
```

`--station-areas` matters: Malaysia maps many stations only as areas, and without it most KTM
stations had no named OSM record (Ipoh, Taiping, Kampar, Seremban...); it adds 211.

The `.pbf` can go once Malaysia and Singapore are both built from it; `data/proc/my/` holds what
a rebuild needs (clipped, with the station areas).

## The short answer

- **Geometry: OSM's track, by shortest path.** Korea's named-track recipe does not work for KTM:
  1,642 of its 2,850 km of main track is named just "KTM" (`python probe_kr_ways.py --region
  my`), and OSM's KTM `route=railway` relations cover only 210 km of the West Coast Line. The
  metro track is named per line ("Laluan Kajang", "Laluan Kelana Jaya", "LRT 3", "ERL").
- **KTM's lines are the infrastructure lines** (West Coast, East Coast, and the Butterworth,
  Batu Caves, Port Klang, Skypark and Pasir Gudang branches), each the shortest path over KTM
  running track through a list of waypoint stations (`LINES` in my_register.py). **Their
  stations are KTM's own timetable stops** (GTFS, below) that lie on that path.
- **Klang Valley rapid transit: Prasarana's GTFS** gives each line's stations in order; each
  section is the shortest path over that line's named track.
- **KLIA Transit and the Penang Hill Railway**: stations from OSM's own route relation (no
  open feed has them), put in the order the track passes them.
- **Sabah**: OSM's named stations on the Tanjung Aru - Tenom track.
- **Line lengths**: en.wikipedia, mostly; KTM's chainage from en.wikipedia's route diagram.

## Sources

### KTM's GTFS (data.gov.my)

`https://api.data.gov.my/gtfs-static/ktmb` -> `data/raw/my/gtfs/my_ktmb.gtfs.zip` (46 kB,
2026-10-03; calendar 2026-08-18 to 2026-10-18, refreshed daily). Terms: data.gov.my's terms of
use (linked from developer.data.gov.my; the licence text itself was not read: the portal's
catalogue datasets are CC BY 4.0, not confirmed for the API). 9 routes: Komuter's Seremban
and Port Klang Lines, the northern Padang Besar and Ipoh Lines, Shuttle Selatan (Paloh - JB
Sentral and Kempas Baru - Pasir Gudang), ETS, Intercity Shuttle Timur (SH), Ekspres Rakyat
Timuran (ERT) and Shuttle Tebrau (ST). 156 stops, 154 of them in Malaysia that trains call at.
What is wrong with it:

- **Komuter routes are route_type 0 (tram)**, ETS and Intercity 2.
- **Some coordinates are kilometres off** on the East Coast Line: Kuala Gris 11.5 km, Sri
  Mahligai 26 km, Kemubu over 30 km, Chegar Perah sits on Sungai Temau. A stop is therefore
  matched to OSM by name within 2.5 km, else to the one place of that name within 80 km
  (`FAR_NAME_M`).
- Upper case and abbreviated names ("PEL KLANG SEL", "BDR TASEK SELATAN", "KG BERKAM"):
  `WORD` and `NAME_ALIAS` in my_register.py. Spellings that differ from OSM's: Krambit
  (OSM Kerambit), Padang Tungku (Padang Tengku), Kg. Sirian / Sungai Sirian (Kampung Sungai
  Serian / Sungai Serian), Krai (Kuala Krai), Sg Mengkuang Baru (Kampung Baru Sungai
  Mengkuang).
- It has Hat Yai (ETS KL Sentral - Hat Yai, once a day each way) and Woodlands CIQ: left out
  as abroad. One ERT pattern "Merapoh - Woodlands CIQ" (7 stops) looks like a data slip.

### Prasarana's GTFS (data.gov.my)

`https://api.data.gov.my/gtfs-static/prasarana?category=rapid-rail-kl` ->
`my_prasarana_rail.gtfs.zip` (81 kB). Routes AG, KJ, PH (Sri Petaling), KGL, PYL, MR, SA and
BRT (Sunway BRT, a bus: not used). Frequency-based, three trips a route. Stop ids are the
station codes (KJ15, PY01, KG18A), which OSM writes into its station names ("KJ15 KL
Sentral"), so stations match by code first. Route colours are the line colours in
`colours/my.csv`. route_type 1 for every rail route.

### OSM (Geofabrik, via extract.py), ODbL

Track, stations, route relations. Clipped by `my_register.py --clip`: a way stays if any of its
nodes is outside Singapore and Thailand, so the causeway's border-crossing way stays whole. The
border near the crossings is OSM's own admin_level=2 boundary,
`data/raw/my/borders_osm.geojson` (Overpass, 2026-10-03, boxes round the causeway, Padang Besar
and Rantau Panjang); elsewhere Singapore is `sg_register.SG_POLY`. The clip took 1,762 ways
(Singapore's metro, 3 KTM ways in Woodlands, 1 Thai way at Rantau Panjang) and 60 of 113
relations.

`data/raw/my/skypark_disused.json`: the Skypark branch's 20 ways, which OSM tags
railway=disused (extract.py keeps no disused track), from Overpass, 2026-10-03.

### Line lengths (check_model.REGISTER["my"], KNOWN["my"])

- en.wikipedia's route diagram `Template:KTM West Coast Line` (action=raw, 2026-10-03): chainage
  from Butterworth (Bukit Mertajam 10.2, Ipoh 181.0, Tanjung Malim 300.5, KL Sentral 388.0,
  Gemas 562.2, JB Sentral 756.8, Woodlands 759.0) and northwards from Bukit Mertajam (Padang
  Besar 157.8). It predates the 2014 Ipoh - Padang Besar and 2025 Gemas - JB double-track
  realignments, which straightened the line.
- en.wikipedia infoboxes (2026-10-03): Kelana Jaya 46.4, Ampang and Sri Petaling network 45.1
  (Ampang - Sultan Ismail 12.4; Sri Petaling - Putra Heights extension 17.7), Kajang 47,
  Putrajaya 57.7, Shah Alam 37.8, KL Monorail 8.6, KLIA Transit 57 (to KLIA T1), Skypark Line
  26 (KL Sentral - Terminal Skypark), Penang Hill Railway 1,996 m, Sabah State Railway 134,
  Batu Caves - Pulau Sebang 135, Tanjung Malim - Port Klang 126, Butterworth - Padang Besar
  169.8, Butterworth - Ipoh 162 (the chainage says 181.0).
- East Coast Line 526 km (Gemas - Tumpat), en.wikipedia and Wikivoyage ("Jungle Railway");
  527.75 is also quoted.

## The build (2026-10-03)

17 register lines, 1,960 km as shipped (1,961.7 with the two border sections, see below), 328
stations; 31 lines in all with OSM's.

| line | built km | published | ratio | stations |
|---|---|---|---|---|
| Laluan Pantai Barat (West Coast) | 891.8 (893.6 with the border sections) | 904.4 + 1.7 | 0.98 (0.99) | 75 |
| Laluan Pantai Timur (East Coast) | 528.4 | 526 | 1.00 | 55 |
| Laluan Cawangan Butterworth | 10.0 | 10.2 | 0.98 | 3 |
| Laluan Cawangan Batu Caves | 9.5 | (in the Seremban Line's 135) | | 6 |
| Laluan Cawangan Pelabuhan Klang | 42.2 | (in the Port Klang Line's 126) | | 20 |
| Laluan Cawangan Skypark (suspended, greyed) | 9.0 | 26 - 15.1 = 10.9 | 0.83 | 2 |
| Laluan Cawangan Pasir Gudang | 28.7 | none found | | 2 |
| Laluan Keretapi Barat Sabah | 133.9 | 134 | 1.00 | 15 |
| Laluan Kelana Jaya | 45.6 | 46.4 | 0.98 | 37 |
| Laluan Ampang | 14.7 | Ampang - Sultan Ismail 12.4 (built 11.8) | 0.95 | 18 |
| Laluan Sri Petaling | 36.8 | 45.1 - 7.0 = 38.1 | 0.97 | 29 |
| Laluan Kajang | 46.6 | 47 | 0.99 | 29 |
| Laluan Putrajaya | 56.2 | 57.7 | 0.97 | 36 |
| Laluan Shah Alam | 37.5 | 37.8 | 0.99 | 20 |
| Laluan Monorel KL | 8.6 | 8.6 | 1.00 | 11 |
| KLIA Transit | 58.6 | 57 to KLIA T1 (built 56.1) | 0.98 | 6 |
| Keretapi Bukit Bendera (Penang Hill) | 1.87 | 1.996 along a 700 m climb, ~1.87 flat | 0.94 | 7 |

Komuter's lines as OSM builds them: Seremban Line 134.5 of 135 (27/27 stations), Port Klang
Line 127.1 of 126 (34/35), Butterworth - Padang Besar 169.4 of 169.8, Butterworth - Ipoh 174.0
of the chainage's 181.0. Sections of the West Coast Line against the chainage: Padang Besar -
Bukit Mertajam 159.3 of 157.8, Bukit Mertajam - Gemas 540.0 of 552.0, Gemas - JB Sentral 192.6
of 194.6 (the new double track).

Every KTM timetable stop in Malaysia is on a line (154); the Ampang and Sri Petaling network
(AG + SP less their shared Sentul Timur - Chan Sow Lin) is 43.9 km against 45.1.

## Decisions (made here, not put to Anita)

- **What counts as a line.** KTM's infrastructure lines are the register; KTM's services run
  over them. Komuter's lines (Seremban, Port Klang, Butterworth - Padang Besar, Butterworth -
  Ipoh) stay OSM lines on top, as Korea's operating patterns do. **ETS is a named train**
  (`rules/my.py`): a train brand over the West Coast Line, as KTX is in Korea; its track counts
  through the West Coast Line. The KLIA Ekspres (non-stop every 15-20 minutes) stays a line;
  KLIA Transit, which stops everywhere on the same track, is the register line that owns it.
  KTM's Intercity trains (Ekspres Rakyat Timuran, Shuttle Timur, Shuttle Tebrau) and Shuttle
  Selatan are in OSM only as route=railway relations, which build_model does not read: their
  track counts through the register lines.
- **Branches start at the station trains come from**: Port Klang at KL Sentral, Batu Caves at
  Putra, Butterworth at Bukit Mertajam, Skypark at Subang Jaya, Pasir Gudang at Kempas Baru,
  so the first section runs a short way over the main line to the junction. ownership.py gives
  that stretch to one of them (0.82 km at Bangsar went to the Port Klang branch).
- **Interchanges are one station** where two lines' stations share a name within 500 m
  (KL Sentral for KTM, the LRT, KLIA Transit and the monorail; Masjid Jamek; Bandar Tasik
  Selatan; Titiwangsa...). KLIA Transit's "Putrajaya & Cyberjaya" is Putrajaya Sentral.
- **The Skypark Link** (KL Sentral - Terminal Skypark) has been suspended since 2023-02-15
  (KTMB; The Star, 2023-01-20; still suspended in 2026). Its branch is a register line marked
  `suspended`, so it is drawn greyed and left out of completion. OSM's own Skypark Link route
  relations remain and build as KL Sentral - Subang Jaya over the Port Klang branch, which
  would read as running: `rules/my.py` flags that OSM line as a named train, the one hook
  there is for keeping a line out of the totals.
- **The Shah Alam Line** (LRT3) opened 2026-06-29 with 20 of its 25 stations; built.
- **Sabah**: Tanjung Aru - Beaufort and Beaufort - Tenom run daily (the department's site,
  railway.sabah.gov.my, lists daily timetables; its timetable pages did not load here, and the
  "twice a day Beaufort - Tenom" figure is from a journey planner). The North Borneo heritage
  train runs over the same track. Stations are OSM's named stations along the track; its unnamed halts are left out.
- **Left out**: the East Coast Rail Link (passenger service from mid-December 2026 or
  January 2027; add when open), the Melaka Monorail (closed since 2020), the Sunway BRT (a
  bus), the KLIA Aerotrain (an airport people mover; OSM keeps it as an OSM line, like
  Changi's Skytrain), the Pasir Mas - Rantau Panjang branch to Sungai Kolok (railway=disused
  at the bridge, no trains), Kerteh - Kuantan Port (Petronas freight), the Tanjung Pelepas and
  West Port freight lines, the Kek Lok Si and Skyglide inclined lifts, the Bukit Malawati tram
  (1.2 km of tourist track at Kuala Selangor) and the Jejak Warisan heritage track at Kluang.

## Borders

| point | where | lines |
|---|---|---|
| `xPadangBesar` (100.322477, 6.665252) | KTM's track (way 1237531937) meets SRT's Hat Yai - Padang Besar (way 1419678317) on OSM's boundary (way 206957351), 0.48 km north of Padang Besar (Malaysia) station | West Coast Line: Padang Besar - border 0.475 km. SRT's trains from Hat Yai terminate at Padang Besar (Malaysia); KTM's ETS runs to Hat Yai |
| `xWoodlands` (103.769336, 1.452652) | KTM's track (way 925109455) crosses the boundary (way 1455785333) on the Johor Causeway | West Coast Line: JB Sentral - border 1.266 km (Shuttle Tebrau, 31 trips a day) |

Both are proposed for `borders.EXTRA`; until then `MY_BORDER` in my_register.py gives the same
ids and points. Both sections end at a junction and no OSM route relation runs over either, so
build_model drops them unless the `served_junction` hook lands (below). Singapore's side
(Woodlands Train Checkpoint - the border, 1.1 km) is not built in sg (sg_sources.md, "Left
out"); with this point in the table sg could add it as a section ending there.

## The timetable check (gtfs_served): not enabled

Tried with both feeds in `data/raw/gtfs/my/` (route types put right in a copy: KTM's Komuter
0 -> 2, Prasarana's 1 -> 401) and taken out again. It marked the whole Sabah line (133.9 km)
and KLIA Transit's Bandar Tasik Selatan - Putrajaya (22.5 km) not running: neither is in an
open feed, and their end stations are interchanges other feeds' trains call at, so the
"unknown operator" test does not catch them. It also matched none of the East Coast halts the
feed misplaces or misspells (the reader's own aliases are not gtfs_served's). Malaysia's
register sections are all station to station except the two border sections, so the feed adds
nothing the reader does not already know. If a feed is wanted later, the FEEDS entry would be
(data.gov.my, daily):

```python
    "my": [("my_ktmb.gtfs.zip", "https://api.data.gov.my/gtfs-static/ktmb", False),
           ("my_prasarana_rail.gtfs.zip",
            "https://api.data.gov.my/gtfs-static/prasarana?category=rapid-rail-kl", False)],
```

and it would need route types 0 and 1 read as rail for these feeds, plus NAME_ALIAS entries.

## Shared changes it needs (in the report to the managing session)

1. `build_model.norm_line_name`: "Laluan X" reads as "X Line", "Laluan Monorel X" as "X
   Monorail". Without it OSM's route masters "Kelana Jaya Line", "Ampang Line", "Sri Petaling
   Line", "Shah Alam Line" and "KL Monorail" stay listed beside their register lines (31 lines
   instead of 26). Only names starting "Laluan" change: no other country's lines.json has one.
2. `build_model.drop_unridden_sections`: keep a junction-ended section the register lists in
   the line's `served_junction`, and drop the key. Only my_register sets it.
3. `borders.EXTRA`: the two points above.
4. `tools/rebuild.py`: `"my": "my_register:data/raw/my"`, MINUTES `"my": 1`.

Trialled together (scratch runner monkeypatching 1 and 2 into build_model): 26 lines; the two
border sections kept (West Coast Line 893.6 km); the five OSM duplicates gone; nothing else
moves.

## What is still off

- **Penang Hill**: OSM's "Penang Hill Funicular Service" (route=train, 0.93 km of a partial
  relation) stays listed beside the register line; the names do not match.
- **KLIA Transit's station names** are OSM's: "ERL Salak Tinggi", "KLIA T1".
- **Station names are Malay**, as OSM and the signs have them; few carry an English name, which
  is right for Malaysia.
- **The Pasir Gudang branch** has no published length; its one section (Kempas Baru - Pasir
  Gudang, 28.7 km) matches OSM's Shuttle Selatan relation (29.6 km, with the station throats).
- **Skypark**: 9.0 km against a derived 10.9 (WP's 26 km is rounded).
