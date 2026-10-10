# Bangladesh register sources

## Built (2026-10-08)

30 register lines, 3,115 km: 26 running (3,078 km), 4 greyed (37 km); plus Dhaka Metro's MRT
Line 6 as an OSM line (18.9 km). `bd_register.py` + `bd_lines.py` through lk_register.py's
engine (nafrica_register's recipe: a hand line list traced by rinf.py; `no_chain`, the lengths
are our traces), settings `rinf_countries/bd.py`, rules `rules/bd.py`, colours
`colours/bd.csv` (our picks: BR publishes none).

    python extract.py --region bd --pbf data/raw/bangladesh-latest.osm.pbf --station-areas
    python bd_register.py --clip bd        # drops India's track; neighbours' outlines shrunk ~1.1 km
    python bd_register.py --fill bd        # adds Jashore, which has no OSM station node
    python bd_register.py --convert bd     # build_model's hook converts too; the folder must exist
    python build_model.py --region bd --register bd_register:data/raw/rinf/bd
    python build_tiles.py --region bd
    python check_model.py --region bd

- **The list**: each BR line of the "List of railway lines in Bangladesh" with its stations in
  order from the route diagram templates, every point at OSM's station by coordinate (most OSM
  station names are Bengali only; `osm_stops: "all"` adds every other OSM station on the
  traced track). A line partly closed is cut, the closed part its own greyed line.
- **Clipping**: the Akhaura - Shaistaganj line runs within a kilometre of Tripura, and
  nafrica's clip (a way goes if half its nodes are inside another country's simplified
  outline) cut it in two; the engine shrinks the neighbours' outlines by 0.01 degrees first.
- **Line names** are en.wikipedia's (English, en dash); station names OSM's (Bengali, with
  name:en where OSM has one).
- **Checks**: BR's East Zone fare list (distance per station pair) and the Information Book
  2024's new-section lengths, as paths over the built lines (`bd_lines.PATH_CHECKS`, in the
  build log): Dhaka - Joydebpur 0.98, - Narsingdi 0.99, - Mymensingh 1.00, - Dewanganj Bazar
  1.00, - Islampur Bazar 1.00, Akhaura - Brahmanbaria 1.01, - Laksam 0.98, - Chattogram 0.98,
  - Kulaura 0.99, - Sylhet 0.99, Dohazari - Cox's Bazar 0.98, Phultala - Mongla 1.01,
  Majhgram - Dhalarchar 0.98, Kashiani - Gobra 0.99, Dhaka - Bhanga 0.95 (the Information
  Book's 81 km is the project's figure for "Dhaka - Bhanga"; ours is Kamalapur - Bhanga
  Junction). Fare-list pairs across the Bhairab bridge are not used: the list adds notional km
  for it (Bhairab - Ashuganj 26 km for 3 km of track). `check_model.REGISTER["bd"]` has the
  five lines whose whole length one of these measures (worst 0.98).
- Not checked against anything outside: the West Zone lines' lengths (the West fare lists are
  per-train Excel files not yet read). BR's 3,356 route km (2023-24) plus Bhanga - Jashore
  (~88 km, December 2024) is about 3,440; built 3,115 leaves out the closed branches and the
  freight-only and border track (below).

### What runs (decided 2026-10-08)

- **Running**: every intercity and commuter corridor, including Dhaka - Narayanganj (16 trains
  a day since March 2025, Financial Express / TBS), the Padma Bridge Rail Link to Jashore
  (December 2024), Chittagong - Cox's Bazar (December 2023), Khulna - Mongla (June 2024),
  Joydebpur - Tangail - Jamtoil over the Jamuna, Jamalpur - Tarakandi - Ibrahimabad (Dhaka -
  Tarakandi trains via the bridge, Financial Express), Ishwardi - Pabna - Dhalarchar, the
  Nazirhat locals and the Chittagong University shuttle, Mohanganj and Jaria Janjail, Kurigram
  - Ramna Bazar, Rohanpur and Chapainawabganj, Gopalganj - Gobra (Tungipara Express).
- **Greyed** (track in OSM, no passenger train found): Feni - Belonia (closed 1997, Information
  Book), Dewanganj Bazar - Bahadurabad Ghat (closed), Kashiani - Bhatiapara Ghat, Kanchan -
  Birol (the Radhikapur branch: freight). Each is its own line named "<line> (<a> - <b>)".
- **Not drawn: no rail left in OSM**: Sylhet - Chhatak Bazar (no train since 2021, washed out
  in the 2022 flood; a Tk 230 crore rebuild due mid-2026, no reopening reported by October
  2026; OSM keeps only bridges), Kulaura - Shahbajpur (closed 2003, being rebuilt), Trimohini
  - Balashi Ghat. Add each when it reopens and OSM maps it.
- **Not drawn: the crossings into India.** The Maitree (Darshana - Gede), Bandhan (Benapole -
  Petrapole) and Mitali (Chilahati - Haldibari) Expresses have been suspended since July 2024;
  Rohanpur - Singhabad and Birol - Radhikapur carry freight; Akhaura - Agartala was built but
  has never carried passengers; Burimari - Changrabandha has no train. India's build already
  stops at Gede, Petrapole and Haldibari (in_sources.md), so a greyed stub on this side only
  would end at nothing; the lines stop at Darshana, Benapole, Chilahati, Rohanpur, Akhaura and
  Burimari. When the international trains return: add the border points to borders.EXTRA
  (managing session) and extend these lines and India's to them.
- Every OSM train route is a named train (`rules/bd.py`): "Chittagong-Dohazari" is the
  register's own line mapped as a route, and India's Agartala Rajdhani reaches into the
  extract's border band. MRT Line 6 stays a line.

## Survey (2026-10-08)

Research only. Samples in `data/raw/bd/survey/` (about 6.8 MB, most of it two PDFs).

### The short answer

- **No open line register with geometry.** Bangladesh Railway (BR) publishes totals (the
  Information Book), timetables as image PDFs, and fare lists that carry BR's distance per
  station pair. en.wikipedia has a route diagram template for 38 lines with every station in
  order but no km.
- **Recommended recipe: a hand line list traced by rinf.py** (za / nafrica pattern): each
  line's stations in order from its Wikipedia template, traced over OSM track, km our own
  (`no_chain`), checked against BR's fare-list distances (Dhaka - Joydebpur 34 km, Dhaka -
  Narsingdi 57, Dhaka - Bhairab Bazar 86 ...) and the Information Book totals. Which lines
  run: BR's e-ticket API (open, below) for intercity trains, the timetable PDFs for the
  rest, and the Information Book's list of closed branch lines.
- **Expected size**: about 30-35 register lines, 3,000-3,500 km. BR had 3,356 route km at
  the end of 2023-24 (East Zone 1,207 MG + 207.7 DG; West Zone 378.7 MG + 969.1 BG + 521.1
  DG), before the Padma Bridge Rail Link's Bhanga - Jashore section (170 km) opened on 24
  December 2024.
- **Extract**: Geofabrik `asia/bangladesh-latest.osm.pbf`, 339 MB.
- Dhaka Metro **MRT Line 6** (Uttara North - Motijheel, about 20 km) stays an OSM line.

### Sources

| source | what it gives | licence | sample |
|---|---|---|---|
| en.wikipedia route diagram templates ("Templates for railway lines of Bangladesh", 38 pages) | stations in order for every line: Chilahati - Parbatipur - Santahar - Darshana 83 rows, Akhaura - Kulaura - Chhatak 61, Narayanganj - Bahadurabad Ghat 48, Santahar - Kaunia 47, Akhaura - Laksam - Chittagong 45, Dhaka - Jessore 42, Mymensingh - Gouripur - Bhairab 41, Burimari - Lalmonirhat - Parbatipur 41, Darshana - Khulna 40, Iswardi - Sirajganj 38, Chittagong - Cox's Bazar 36, Tongi - Bhairab - Akhaura 32 ... No km column in any of them. Some are plans (Bhanga - Kuakata, Tongi - Manikganj - Paturia, Dhaka - Chittagong high-speed) | CC BY-SA | `wp_rdt_templates.json`, `rdt_akhaura_ctg.wikitext` |
| en.wikipedia "List of railway lines in Bangladesh" | the line inventory by zone (11 East, about 15 West), cross-border points, lines under construction / planned | CC BY-SA | `wp_list_lines.json` |
| Wikidata (via qlever.dev; WDQS rate-limited) | 42 railway-line items with P17 Bangladesh (4 with P2043, 2 with P402); 614 station items, 556 with coordinates; station adjacency on 14 lines, 220 station-line pairs (Narayanganj - Bahadurabad Ghat 44, Akhaura - Kulaura - Chhatak 38, Chilahati - ... - Darshana 30, Akhaura - Laksam - Chittagong 24, Tongi - Bhairab - Akhaura 21, Old Malda - Abdulpur 19, MRT Line 6 17, Iswardi - Sirajganj 16) | CC0 | `../pk/survey/wikidata_counts_pk_bd_lk_np.json` |
| BR e-ticket API, `POST https://railspaapi.shohoz.com/v1.0/web/train-routes` with `{"model": "<train number>", "departure_date_time": "YYYY-MM-DD"}` | no key, no login: a train's stops in order with arrival / departure times, halt minutes and running days (Upakul Express 711: Noakhali, Maijdi Court, Choumuhani ... Biman Bandar, Dhaka). Names are station names in English, no codes, no km, no coordinates. Covers trains sold online (intercity; mail and commuter trains mostly not) | terms not checked | `shohoz_train_routes_711.json` |
| BR timetable no. 54 (from 10 March 2025), East Zone, `railway.gov.bd/pages/static-pages/691997bf933eb65569ddec51` → PDF; West Zone and international pages beside it | train lists (number, name, off day, origin, departure, destination, arrival) and mail / commuter numbers by off day; image PDF (no text layer), Bengali | none stated | `timetable54_east.pdf` (4.8 MB) |
| BR fare lists (`railway.gov.bd/pages/files/69199762933eb65569ddc6b9` East: Bengali and English PDFs; `.../6919975d933eb65569ddc526` West: one Excel or PDF per train group, Silk City / Padma / Dhumketu, Sundarban / Chitra, Rupsha / Benapole...) | per station pair: from, to, **distance in km**, fare per class. The East PDF has a text layer but in a broken Bengali font (digits readable as Bengali numerals) | none stated | `fare_list_east_en.pdf`, `.txt` |
| BR Information Book 2024 (`railway.gov.bd/pages/files/6919975c933eb65569ddc4da`; editions 2021-2024) | route km by zone and gauge, 542 stations, history of openings, **"List of closed branch lines"** with closure dates (Kurigram - Old Kurigram, Modhukhali - Kumarkhali, Dewanganj Bazar - Bahadurabad Ghat, Tarakandi - Jagannathganj Ghat, Narsingdi - Madanganj, Faridpur - Pukuria, Feni - Belonia, Shaistaganj - Habiganj, Shaistaganj - Balla, Kulaura - Shahbazpur ...); Sylhet - Chhatak damaged in the 2022 flood | none stated | `information_book_2024.pdf`, `.txt` |
| data.gov.bd "Intercity Train Routes Bangladesh" | a 2016 PDF of intercity routes; superseded | | |

Not found: any GTFS (Mobility Database and Transitous have nothing for Bangladesh). The LGED
railway layer on UNOSAT's ArcGIS server (`bgd_trs_railways_lged_gdb_IH`) answered "Service
not found"; HDX's "Bangladesh railways" (LGED) would be track geometry only, which OSM
already has.

### OSM (Overpass, 2026-10-08)

The public Overpass servers were overloaded all afternoon, so only counts came back (by way,
not km): 1,737 non-service `railway=rail` ways, 714 of them named (41%); 68 urban-rail ways
(MRT Line 6); 435 station / halt nodes. The route-relation query timed out four times and
returned nothing, so how many `route=train` relations exist is not known; run
`inspect_region.py` and `probe_kr_ways.py` on the extract. At 41% named by way count, Korea's
recipe is unlikely to work.

### What changed recently (for which lines run)

- Padma Bridge Rail Link: Dhaka - Bhanga opened October 2023; Bhanga - Narail - Jashore with
  Jahanabad Express (Dhaka - Khulna) and Rupashi Bangla Express (Dhaka - Benapole) from 24
  December 2024 (bonikbarta, Dhaka Tribune).
- Khulna - Mongla: passenger trains from 1 June 2024 (Benapole - Mongla commuter).
- Chittagong - Cox's Bazar: opened December 2023.
- Akhaura - Agartala: built, no passenger trains.
- International trains (Maitree, Bandhan, Mitali) suspended since mid-2024 (India's survey
  found the same); India's build ends at Gede, Petrapole and Haldibari.

### Recommended recipe

1. Extract with `--station-areas`; `inspect_region.py --region bd`; `probe_kr_ways.py --region
   bd` in case OSM names its track better than expected.
2. `bd_register.py` with `LINES` from the Wikipedia templates (stations in order), split where
   a line is part running, part closed (Information Book list). Trace with rinf.py.
3. Running status: crawl `train-routes` for train numbers 701-830 (intercity) and the other
   numbers in the timetable PDFs; a line with no train calling is greyed.
4. Gauges: metre, broad and dual gauge share stations and some track. rinf.py traces by
   track; where BG and MG run side by side on separate tracks the trace may need the gauge
   (OSM tags `gauge=1000`, `1676`, `1000;1676`).
5. Check: fare-list km between major stations (`check_model.REGISTER["bd"]`), zone totals.

### Open questions (for the country agent to decide)

- Line unit: Wikipedia's lines overlap a little (Chilahati - ... - Darshana vs Darshana -
  Khulna vs Sealdah - Parbatipur historical templates). Use the BR-zone list of the "List of
  railway lines in Bangladesh" article, skip the historical ones (Sealdah - Goalundo, Sealdah
  - Parbatipur, Domohani - Burimari).
- Sylhet - Chhatak (closed by the 2022 flood; rehabilitation contracted): greyed.
- Dhaka Circular Railway and MRT Lines 1, 5: plans or under construction, not built.
- The fare lists' text layer uses a broken Bengali font; the English version of the East
  list has the same layout. Station names may need to be read from the Bengali PDF by
  rendering, or matched by distance.

### Downloads Anita must do by hand

None.
