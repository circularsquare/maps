# Madagascar (mg): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs (freshest evidence)

Two separate metre-gauge systems: Madarail's northern network (TCE Antananarivo - Toamasina,
MLA Moramanga - Lac Alaotra, TA Antananarivo - Antsirabe) and the FCE in the south-east.

| service | status | evidence |
|---|---|---|
| Train urbain, Antananarivo Soarano - Ambohimanambola (16 km, on the TCE line) | running since 16 Dec 2025, two pairs a day (05:00 and 17:30) | Railway Gazette 22 Jan 2026 (403 to our fetch; railwaygazette.com/urban/2026/01/22/... for Anita's browser); en.wikipedia "Rail transport in Madagascar"; AllAfrica 24 Dec 2025 on crowding |
| Madarail Moramanga - Toamasina (via Andasibe, Brickaville, Ambila-Lemaitso) | running, weekly (Thu from Moramanga, Sat back), from 1 June 2023; "victime de son succès" (L'Express de Madagascar, 7 Oct 2023) | newsmada.com "la ligne Toamasina - Moramanga reprend"; nomadicbackpacker.com |
| FCE Fianarantsoa - Manakara (163 km) | running, 2-3 a week, I judge; the line is often shut for months after cyclones or derailments (reopened 15 June after 18 months; reopened 23 Dec after the 20 Sept derailment near Tanakidy, years not stated on newsmada) | newsmada.com; en.wikipedia ("regular" passenger train) |
| Madarail Moramanga - Ambatondrazaka (MLA), OSM route 8503490 | **greyed**: no timetable found | |
| Antananarivo - Moramanga (rest of the TCE), Antananarivo - Antsirabe (TA, reopened for freight Dec 2023) | **not built / greyed**: no regular passenger train found | |

### Line list

1. **Antananarivo (Soarano) - Ambohimanambola**: 16 km, the urban train's stops (OSM or the operator's list; the AllAfrica article names the stations).
2. **Moramanga - Toamasina**: ~240 km, stops Moramanga, Andasibe, ... Brickaville, Ambila-Lemaitso, Toamasina (OSM route 8503488 has the stops).
3. **Fianarantsoa - Manakara**: 163 km, 17-18 stations (OSM route 8503520 and infra relation 3431314).
4. Greyed: Ambohimanambola - Moramanga (joins 1 and 2), Moramanga - Ambatondrazaka.

Expected: 3 running register lines, ~420 km; ~280 greyed.

### Sources

- Stops: OSM routes 8503488, 8503490, 8503520; no operator timetables online.
- Coordinates: OSM (station count timed out; read from the extract).
- GTFS: none.
- Wikidata: 4 station items.
- Licence: OSM ODbL.

### OSM quality (Overpass, 2026-10-08)

Relations: infra "Fianarantsoa-Côte Est" 3431314, "Anasaibe Fer", two unnamed; train routes
Moramanga - Toamasina, Moramanga - Ambatondrazaka (Madarail), Fianarantsoa - Manakara.
Track: 922 km, **870 named (94%)**, by line: "Tananarive Côte Est" 379, "Fianarantsoa-Côte Est"
157, "Tananarive Antsirabe" 154, "Moramanga Lac Alaotra" 143. 82 stations/halts, 66 named.
Geofabrik `africa/madagascar-latest.osm.pbf`, 371 MB (big for ~900 km of rail).

### Recipe

OSM's track is named by line almost everywhere, so either recipe works: named track cut at the
running/greyed points (vn's recipe), or the shared eafrica hand list traced by rinf.py with
`own` set from those names. I would use the shared reader for one code path across the region.
Stops from OSM's routes. Nothing crosses a border.

### Open questions

- FCE and Moramanga - Toamasina: both rest on 2023-era reports plus Wikipedia; no 2026 timetable found.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"; `--fill mg` adds
Toamasina). **Result**: 5 register lines: running 3, 425 km (TCE Moramanga - Toamasina 247.1,
FCE Fianarantsoa - Manakara 163.2, the urban train Soarano - Ambohimanambola 15.1); greyed 2,
249 km (MLA Moramanga - Ambatondrazaka 142.4, TCE Ambohimanambola - Moramanga 106.5).
check_model: FCE 163.2 of 163 (en.WP).

Decisions: every OSM station is a stop. OSM has no Toamasina station: placed on the main line
by the port (49.4160, -18.1633), approximate. Antananarivo - Antsirabe not built (freight
only). OSM's Moramanga - Toamasina and Fianarantsoa - Manakara routes: rules/mg.py
SKIP_ROUTES; Moramanga - Ambatondrazaka: NOT_SERVICE.
