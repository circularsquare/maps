# Malawi (mw): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent. The least certain "running" country
in the set.

### What runs (freshest evidence)

| service | status | evidence |
|---|---|---|
| CEAR Limbe - Blantyre - Nkaya - Balaka - Nkaya - Nayuchi (Mozambique border) and back | running, weekly, I judge: the passenger train resumed 1 Aug 2022 with new coaches (business, premier, standard) on exactly this pattern (Limbe to Balaka, on to Nayuchi, back to Balaka, back to Limbe); no suspension reported since. The government's consignment contract with CEAR covers "once a week" passenger trains | Nation Online (mwnation.com "New rail coaches maiden trip on Wednesday"); Africa-Press "Train ride between Balaka and Limbe excites Malawians"; JICA rail sub-sector report |
| Limbe - Makhanga (Shire valley, south) | **not built**: suspended by CEAR (Nyasa Times "Limbe-Makhanga passenger train suspended -CEAR"); line damaged by Cyclone Freddy in 2023 | |
| Balaka - Salima - Lilongwe - Mchinji | **not built**: freight only | |

Nothing newer than 2023 was found either way. If Anita wants only well-evidenced countries,
Malawi is the one to drop.

### Line list

From CEAR's 2015 timetable as transcribed by fahrplancenter.com (CEARMalawiTimetable.html), with
CEAR's km:

1. **Limbe - Nkaya - Balaka** (Limbe 0, Blantyre 8, ... Nkaya 96, Balaka 112). Stops: OSM stations on the line, or the 2015 timetable's list.
2. **Nkaya - Nayuchi** (99 km to Nayuchi, the Mozambique border station on the Nacala line).

Expected: 2 register lines, ~210 km.

### Sources

- km and stops: fahrplancenter.com's CEAR timetable (May 2015) with every station's km.
- Coordinates: OSM, 40 railway=station/halt, 35 named.
- GTFS: none. Wikidata: no station items.
- Licence: OSM ODbL; the timetable is facts.

### OSM quality (Overpass, 2026-10-08)

Track and relation queries timed out (public Overpass overloaded); read from the extract.
Geofabrik `africa/malawi-latest.osm.pbf`, 147 MB.

### Recipe

Hand list traced by rinf.py in the shared eafrica reader, with CEAR's km as `chain` (fahrplancenter
gives every station's km). No border crossing for passengers at Nayuchi/Entre Lagos.

### Open questions

- Is the weekly train still running in 2026? Nothing found since 2023.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"). **Result**: 2 register
lines, 212 km, running (the managing session kept the weekly train): Limbe - Balaka 112.2,
Nkaya - Nayuchi 99.9. check_model 112.2 of 112 and 99.9 of 99 (CEAR's 2015 km).

Decisions: every OSM station is a stop. The app's outline puts Nayuchi station in Mozambique;
BORDERS["XMWMZ1"] with OSM's boundary (way 1413800808, from the OSM API) makes the clip side
by OSM's boundary within 3 km of it. No line ends there and it needs no borders.EXTRA entry.
OSM's CEAR routes: rules/mw.py SKIP_ROUTES; Balaka - Bilila: NOT_SERVICE.
