# Sri Lanka register sources

## Built (2026-10-08)

12 register lines, 1,414 km: 10 running (1,328 km), 2 greyed (86 km). `lk_register.py`
(nafrica_register's recipe: a line list traced by rinf.py), settings `rinf_countries/lk.py`,
rules `rules/lk.py`, colours `colours/lk.csv` (our picks: SLR publishes none).

    python extract.py --region lk --pbf data/raw/sri-lanka-latest.osm.pbf --station-areas
    python lk_register.py --clip lk
    python lk_register.py --convert lk      # build_model's hook converts too; the folder must exist
    python build_model.py --region lk --register lk_register:data/raw/rinf/lk
    python build_tiles.py --region lk
    python check_model.py --region lk

- **The register**: en.wikipedia's by-line tables (`data/raw/lk/survey/wp_stations_by_line.json`),
  every row that has a km from Colombo Fort, in order. A section's length is the difference of
  its ends' km (`chain`, `km_official`). Where the table's km for a station contradicts OSM's
  track and its neighbours' km agree (a misplaced row), the station is listed without km or
  left to `osm_stops`: Secretariat, Kompannavidiya, Peralanda, Kandana, Kapuwatta, Baseline
  Road, Cotta Road, Narahenpita (the Kelani Valley table's first km are off: Fort -
  Narahenpita is 5.1 km crow-fly, the table says 5.04 along the track), Koshinna, Ohiya,
  Hiriyala, Ariviyal Nagar, Laksha Uyana, Koggala, Nelumpokuna, Bambaranda, Wewrukannala; 10 of
  274 sections are traced. Against the chain: median 1.006, none off by 5%.
- **Station names** are OSM's where they differ (Beliatte, Wadurawa, Rozelle, Thambuttegama...);
  `Anuradhapura` is pinned by coordinate (OSM has a second "Anuradhapura" node at New Town).
- **Mannar Line ends at Talaimannar**: OSM's track stops there; the pier station 2.4 km on is
  left out.
- **Every OSM route relation is a named train** (`rules/lk.py`): the twelve are the register's
  own lines mapped as routes, the Mihintale branch (Poson pilgrim specials only, not a line)
  and the Holcim freight line.
- **No border**: Sri Lanka has no rail link abroad.
- Checks (`check_model.REGISTER["lk"]`): Wikidata lengths for Northern 1.00, Batticaloa 1.00,
  Coastal 0.99, Mannar 1.00, Trincomalee 1.00; the table's end-to-end km for the others
  (Puttalam 1.03, Kelani Valley 1.04, the Main Line pieces 1.00-1.01, Matale 1.01).

### What runs (decided 2026-10-08)

Cyclone Ditwah (late November 2025) closed most of the network; by October 2026:

- **Running**: Colombo Fort - Gampola (Rambukkana - Peradeniya - Gampola reopened 2 October
  2026, newsfirst.lk 6 September and newswire.lk; the line beyond Rambukkana had 103 damage
  sites), Nanu Oya - Badulla (four trains a day from 20 June 2026, hirunews.lk 472645,
  newsfirst.lk 18 June 2026), Northern Line to Kankesanturai and Batticaloa Line (resumed 24
  December 2025, adaderana 116335), Mannar Line (from 1 January 2026, hirunews.lk 436616),
  Trincomalee Line (Colombo - Trincomalee trains, the night mail daily from 20 January 2026,
  adaderana 117240), Puttalam, Kelani Valley and Coastal lines (Colombo's suburban lines).
- **Kandy - Matale running**: a school train Kandy - Wattegama was added in January 2026
  (newsfirst.lk 6 January 2026) and nothing says the Matale line's locals stopped; drawn
  running.
- **Greyed: Main Line (Gampola - Nanu Oya)**. The ministry said in September 2026 that
  "the service between Rambukkana and Nanu Oya has not yet resumed", and the 2 October
  reopening went to Gampola. A school train Nawalapitiya - Hatton was announced in January
  2026; with no later word of it and the ministry's statement, the whole stretch is greyed.
- **Greyed: Matale Line (Peradeniya - Kandy)**, its own 5.9 km line: the Kalu Palama bridge
  between Peradeniya and Kandy was "still underway" in September 2026 (newsfirst.lk), the
  deputy minister promised trains to Kandy "by year-end" (newswire.lk, 26 February 2026), and
  the October reopening named Peradeniya and Gampola, not Kandy.
- When a closed stretch reopens: set `suspended=False` in `LINES`, and when the Main Line runs
  through again fold its three pieces back into one `main` entry (aliases for the old ids).

## Survey (2026-10-08)

Research only. Samples in `data/raw/lk/survey/` (en.wikipedia wikitext, 0.3 MB).

### The short answer

- **en.wikipedia's "List of railway stations in Sri Lanka by line" is a line register with
  chainage**: one wikitable per line (Main, Matale, Puttalam, Kelani Valley, Northern, Mannar,
  Trincomalee, Batticaloa, Coastal), each station with its SLR code, district, elevation and
  **km from Colombo Fort** (Maradana 2.08, Kelaniya 7.72 ... Hali-Ela 285.92). That is
  Indonesia's recipe exactly (`id_register.py`: Wikipedia line tables with km posts through
  rinf.py).
- **Recommended recipe**: `lk_register.py` on id_register's pattern: read the by-line tables
  (stations in order, km as `chain` / `km_official`), trace with rinf.py over OSM track.
  OSM names much of the track (below), so Korea's named-track recipe is a fallback worth a
  `probe_kr_ways.py --region lk` after the extract.
- **Expected size**: 10-11 register lines, about 1,450 km (SLR's network is about 1,450 route
  km; Wikidata lengths: Main 292, Northern 339, Batticaloa 212, Coastal 157.9 to Matara +
  Matara - Beliatta 26.75 (2019), Puttalam 133, Mannar 106, Trincomalee 70, Matale 33.8; Kelani
  Valley has none on Wikidata, its table reaches Puwakpitiya at 55.24, Avissawella just beyond).
- **Extract**: Geofabrik `asia/sri-lanka-latest.osm.pbf`, 137 MB.
- **The big caveat is what runs**: Cyclone Ditwah (late November 2025) broke the network in
  hundreds of places and it is being reopened piece by piece (below). No current timetable is
  online.

### Sources

| source | what it gives | licence | sample |
|---|---|---|---|
| en.wikipedia "List of railway stations in Sri Lanka by line" | per line: station, SLR code, district, elevation, km from Colombo Fort. A quick parse found Main 77 rows (to Hali-Ela 285.92; Badulla's row formatted differently), Matale 7, Puttalam 17 (Ragama 16.42 - Puttalam 133.24), Kelani Valley 13 (to Puwakpitiya 55.24), Northern 16 (to Jaffna 392.9), Trincomalee 4, Batticaloa 13, Coastal 36 (to Bambarenda 176.17), Mannar 1: the tables list main stations, not every halt, so halts come from OSM (`osm_stops`) | CC BY-SA | `wp_stations_by_line.json` |
| en.wikipedia line articles + route diagram templates (10) | stations in order without km (Main Line's template lists every halt Badulla - Colombo, with tunnels) | CC BY-SA | `wp_line_articles.json`, `wp_rdt_templates.json` |
| Wikidata (via qlever.dev) | 11 railway-line items, 9 with P2043 length (above); only 84 station items (82 with coordinates); no station adjacency | CC0 | `../pk/survey/wikidata_counts_pk_bd_lk_np.json` |
| OSM, Geofabrik taginfo for sri-lanka (2026-10-08) | 1,308 `railway=rail` ways, 717 named (55%; 446 of all are `service=*`, so most running-line ways are named), 688 `usage=main`; 261 station nodes + 97 station areas, 32 halts; 13 `route=train`, 3 `route=railway`, 1 `route=tracks` relations; 6 abandoned / 7 disused ways | ODbL | |
| DCS Lanka Datta, "The Present Station Code, Distance and Fare table" (nada.statistics.gov.lk/index.php/catalog/49/download/920) | a 1975 Ceylon Government Railway handwritten table in miles; scanned, faint: not usable | | (deleted) |
| Survey Department railway lines (HDX, via the Princeton / BTAA geoportal records) | track geometry; not opened | check | |

Not found: any GTFS for trains (the Mobility Database's only Sri Lankan feed is Lanka Metro
Transit's Colombo bus, mdb-3490, CC BY 4.0). SLR's own schedule page
(`railway.gov.lk/web/index.php?option=com_content&view=article&id=1017&Itemid=210&lang=en`)
says "temporarily unavailable while we complete the upgrade" (updated 18 June 2026); the old
`eservices.railway.gov.lk/schedule/` search answers 404 (and its TLS chain does not verify).
Ticketing moved to Pravesha (`pravesha.lk`) and `seatreservation.railway.gov.lk`.

### Urban rail

None. The Colombo LRT was cancelled in 2020; Colombo's suburban trains run on the Main,
Coastal, Puttalam and Kelani Valley lines and are part of SLR's network.

### What runs after Cyclone Ditwah (found 2026-10-08)

- Main Line: Colombo Fort - Rambukkana ran throughout; Colombo Fort - Peradeniya / Gampola
  resumes on 2 October 2026 (newswire.lk, 28 September 2026: 103 damage sites Rambukkana -
  Gampola). Gampola - Nanu Oya still closed; Nanu Oya - Ella - Badulla reopened in June 2026
  (Badulla - Haputale had reopened 18 days after the storm).
- Northern Line: restored with an Indian grant, regular services resumed (newsonair.gov.in,
  5 June 2026).
- Puttalam Line: services resumed (adaderana.lk news 118493).
- Trincomalee Line: freight resumed December 2025; passenger status not found.
- Batticaloa, Matale, Mannar, Kelani Valley, Coastal: not checked in this survey.

### Open questions (for the country agent to decide)

- Each line's status today, especially Gampola - Nanu Oya (greyed until it reopens?), Kandy -
  Matale, Batticaloa, Trincomalee and Mannar - Talaimannar. geckoroutes.com keeps a "Sri Lanka
  Railway Update (2026 Service Status)" page (`geckoroutes.com/?p=1524832`) but it answers 403
  to scripts: Anita could open it in a browser if the press does not settle a line.
- The Mihintale branch and the Uda Pussellawa railway (closed 1948) are on Wikidata; Mihintale
  carries pilgrimage specials only (Poson): not a running line.
- Coastal Line to Beliatta (2019): check its table carries Matara - Beliatta.
- Kankesanturai: Northern Line runs to Kankesanturai (reopened 2015); the table's last row is
  Jaffna 392.9.

### Downloads Anita must do by hand

None needed (optional: the geckoroutes status page above).
