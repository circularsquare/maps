# United Arab Emirates sources

## Survey (2026-10-08)

Research only; nothing built. Region code `ae`.

### What runs

**Etihad Rail passenger** (standard gauge, 200 km/h; the national network's freight lines
have run since 2016 / 2023):

- **Abu Dhabi (Mohamed Bin Zayed City) - Fujairah (Madinat Al Hilal)**, from 30 June 2026,
  three trains a day each way, 1 h 45 min; from 30 September 2026 calling at **Al Dhaid**
  (Sharjah).
- **Abu Dhabi - Dubai (Al Yalayis, Jumeirah Golf Estates)**, from 30 September 2026, five
  trains a day each way (Gulf News, "Etihad Rail's next expansion is less than a month
  away", 2026; Travel Extra, 2026).
- Four stations open: Mohamed Bin Zayed City, Al Yalayis, Al Dhaid, Fujairah. Next:
  Madinat Zayed and Liwa 30 November 2026; Al Mirfa, Al Dhannah, Al Sila 30 December 2026
  (the Al Dhafra stations, west of Abu Dhabi on the Ghuweifat line); Sharjah University City
  30 March 2027. **Build the line Abu Dhabi - Ghuweifat when those open**; for now the
  western stations are not served.

**Dubai (RTA)**:

| system | lines | km | status |
|---|---|---|---|
| Dubai Metro | Red (OSM: Red1 Centrepoint - Expo City Dubai, Red2 Centrepoint - UAE Exchange; they split at Jabal Ali), Green (Etisalat - Creek) | Red ~67 (with the Route 2020 branch), Green ~22.5 | running. Blue Line under construction (opening 2029; OSM `construction`, "Blue Line Metro") |
| Dubai Tram (Al Sufouh) | one line, Dubai Marina area | 10.6 | running |
| Palm Jumeirah Monorail | Gateway - Atlantis | 5.4 | running |
| DXB airport APMs (Terminal 1, Terminal 3) | airside people movers | | leave out, as Changi's Skytrain |
| Dubai Trolley (Downtown) | | | OSM keeps two relations; the service stopped years ago (it was a short tourist tram round Burj Khalifa). Drop in `--clip` unless a timetable is found |

Abu Dhabi, Sharjah and the northern emirates have no urban rail (Abu Dhabi's "ART" is a
rubber-tyred bus).

### Sources

1. **OSM** (Overpass, 2026-10-08, bbox, `data/raw/ae/survey/osm_routes.json`; the bbox also
   catches Qatar's relations):
   - Etihad Rail track: 363 main ways named "الاتحاد للقطارات" (Etihad Rail), plus
     "Al Gharbia Main Line (GMB)", "Shah Main Line (SHA)", the Sharjah branch, and freight
     spurs ("Industrial City of Abu Dhabi extension", "Jebel Ali Extension", "Al Ghayl
     extension"). Freight spurs must not become passenger line (they are `usage=branch` or
     `industrial`).
   - **Passenger route relations exist already**: 21261180/21261181 "Etihad Rail: Fujairah
     <-> Abu Dhabi" and 21493966 "Dubai -> Abu Dhabi" (the Arabic name on that one still
     says Fujairah: a slip). Check their stop members against the four open stations.
   - Dubai Metro relations: Red1 and Red2 both ways, Green both ways, route_master "Red
     Line"; operator "Dubai Roads & Transport Authority". Metro track partly named ("الخط
     الأحمر", "Red Line Metro", "Green Line Metro").
2. **Dubai RTA GTFS**: the RTA's feed from Dubai Pulse, republished daily by a volunteer at
   `https://gitlab.com/Lach-anonym/dubai-gtfs/-/jobs/artifacts/main/raw/gtfs.zip?job=download-republish-gtfs`
   (16.0 MB; Transitous's `ae.json`; the Mobility Database's mdb-904 is an old archived
   copy). Licence: Dubai Data's open data licence
   (`https://www.dubaipulse.gov.ae/docs/DDE%20_%20DRAFT_Open_Data%20Licence_LONG_Form_English%203.pdf`,
   attribution). Has the metro and tram; not fetched (16 MB, and OSM's relations already give
   the stop order for three lines). Fetch it if station names or order need a check.
3. **Etihad Rail's timetable**: etihadrail.ae's booking pages (not fetched). No GTFS.
4. Published km: en.wikipedia "Dubai Metro", "Dubai Tram", "Palm Jumeirah Monorail",
   "Etihad Rail" (the network ~900 km Ghuweifat - Fujairah when complete).

### Geofabrik

`asia/gcc-states-latest.osm.pbf`, 242 MB (shared with qa and sa; clip to the UAE outline).

### Recipe

The shared Gulf reader proposed in `qa_sources.md` (hand lists traced over OSM track by
rinf.py, stops from OSM route relations): register lines **Etihad Rail Abu Dhabi - Fujairah
(via the junction for Dubai), the Dubai branch to Al Yalayis**, Dubai Metro Red (with the
Expo branch) and Green, Dubai Tram, Palm Monorail.

- Etihad Rail's lines as infrastructure: Abu Dhabi (MBZ City) - Fujairah one line with Al
  Dhaid, and the stretch to Dubai (Al Yalayis) a second short line from the junction where
  the trains leave it (as Malaysia's branches start at the station trains come from).
  Mark the sections as `served_sections` (only two stations per long stretch; they end at
  a junction).
- Expected: 6 register lines, about 450-500 km of Etihad Rail passenger route (MBZ City -
  Fujairah is roughly 330 km over the line; trace to know) plus about 105 km urban, about
  4 + 64 + 11 + 4 stations.

### Open

- Where exactly the Dubai trains leave the Abu Dhabi - Fujairah route: trace on the
  extract.
- Whether the western Al Dhafra stations open on time: re-survey in January 2027.

## Build (2026-10-08)

`mideast_register.py` (sa_sources.md "Build" has the recipe): extract bbox
`51.50,22.60,56.45,26.10` from gcc-states, then `python mideast_register.py --clip ae` and
`build_model.py --region ae --register mideast_register:data/raw/rinf/ae`.

**Register line, running: 1 line, 288.4 km**: Etihad Rail: Abu Dhabi – Fujairah, Mohamed
Bin Zayed City - Al Dhaid - Fujairah (277.9 km), with the Dubai stretch as its branch (10.5
km, junction to Al Yalayis). The Dubai trains leave the Fujairah route at 55.23932,24.95923,
136.2 km from MBZ City (`mideast_register.py --fork`), where there is no station, so it is
one line with a branch rather than two. No published length for the passenger route; the
journey times agree (Al Dhaid - Fujairah 25 min for 59 km, Al Dhaid - Abu Dhabi 84 min for
229 km; Gulf News, Khaleej Times 2026). When the Al Dhafra stations open (Nov-Dec 2026), add
the Ghuweifat line west of Abu Dhabi as a second line.

**OSM lines**: Dubai Metro Red (62.3 km, 34 stations; en.WP 67.1 with the tails: 0.93) and
Green (22.0, 0.98), Dubai Tram (8.5 km station to station), Palm Jumeirah Monorail (5.1), the
DXB airport people mover (2.4; kept, as other countries' airport movers are).

**Decisions**
- OSM's Etihad routes (21261180/1, 21493966) are left out (`rules/ae.py` SKIP_ROUTES): the
  register line again, listing one or two of its stops.
- The Dubai Trolley's two routes are dropped by `--clip` (not running).
- Colour: OSM's Etihad route_master's #A6192E (`colours/ae.csv`).
