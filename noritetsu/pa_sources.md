# Panama register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe). One register line, the Canal railway, from OSM's
route r2020587; the metro lines are OSM lines.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Ferrocarril de Panamá (Panamá (Corozal) - Colón (Atlantic Passenger Station)), register | 66.6 | 2 | none for the passenger run (see below) | - |
| Línea 1 (OSM) | 17.2 | 15 | 18.1 (en.WP) | 0.95 (tails) |
| Línea 2 with the airport branch (OSM) | 22.0 | 18 + branch | 20.4 + 2.1 | 0.98 |

3 lines, 106 km; all running.

Calls:
- **The Canal railway counts**: one round trip Monday - Friday is scheduled service. Its two
  terminals are OSM's two unnamed stop nodes on the route, named here (`PA_STATIONS`).
- **Not checked against a number**: the 76.6 km (47.6 mi) everyone quotes is Balboa port -
  Cristóbal port, not the passenger run from Corozal to the Atlantic Passenger Station; the
  built 66.6 km is one unbroken piece of the track OSM names "Vía Ferroviaria Panamá - Colón"
  (66.2 km), which is the check that matters.
- The clip renames OSM's route to "Ferrocarril de Panamá" so it drops as the register line's
  twin.
- Línea 3's monorail depot (7.1 km of railway=monorail service=yard) is drawn as yard track;
  the line itself is railway=construction and not built.

Commands:

    python extract.py --region pa --pbf data/raw/panama-latest.osm.pbf
    python latam_register.py --clip pa
    python build_model.py --region pa --register latam_register:data/raw/pa
    python build_tiles.py --region pa
    python check_model.py --region pa

## Survey (2026-10-08)

### What runs

| service | status | in the build |
|---|---|---|
| **Metro de Panamá Línea 1**, Albrook - Villa Zaita, 18.1 km, 15 stations (2014) | running | one line |
| **Metro de Panamá Línea 2**, San Miguelito - Nuevo Tocumen, 20.4 km, 18 stations (2019), plus the airport branch Corredor Sur - Aeropuerto, 2.1 km, 3 stations (16 Mar 2023) | running | one line with its branch (as the operator runs it) |
| **Línea 3** (monorail, Albrook - Ciudad del Futuro, 24.5 km, 11 stations) | under construction (75% in May 2026); first dynamic tests 13 Apr 2026; opening given as October 2028 (La Prensa, 2026) | not built |
| **Panama Canal Railway**, Corozal (Panama City) - Colón (Atlantic Passenger Station), 76 km | one round trip Monday - Friday (07:15 from Panama City, back 17:15), sold mostly to tourists and cruise passengers (travel sources, 2025); panarail.com shows no timetable | one line, counted (weekday scheduled service) |

### Sources

- Metro de Panamá (metrodepanama.gob.pa) for the metro lines; published lengths from
  en.wikipedia "Panama Metro" (18.1, 20.4 + 2.1 km) for the check.
- Panama Canal Railway Company (panarail.com/en/passenger/main.html): passenger service since
  2001; current times only through tour sellers (iberia.com blog, Aug 2025; goway.com).
- No Panamanian feed in the Mobility Database (2026-10-08).
- **OSM**: Overpass summary in `data/raw/pa/survey/osm_summary.txt` (below).
- Geofabrik `central-america/panama-latest.osm.pbf`, 34.6 MB.

### OSM (Overpass, 2026-10-08; `data/raw/pa/survey/osm_summary.txt`)

- Metro routes complete and clean: L1 both directions (#DF2937), L2 both directions
  (#48A23E, operator Metro de Panamá), the airport branch both directions (no ref, no colour);
  no route_masters. Track named "Línea 1" (34.7 km of ways, both tracks), "Línea 2" (42.0),
  "Linea 2 (Ramal Aeropuerto)" (3.0); "Línea 3 (en construcción)" 50.3 km railway=construction,
  which a build leaves out by itself.
- Canal railway: track "Vía Ferroviaria Panamá - Colón", 66.3 km usage=main; one route
  relation r2020587 "Panama Railway" (route=train, PTv2, no from/to/operator). Its stops need
  a look in the extract.

### Recipe

Metros as OSM route relations (US/UK/BR metro rule, `register_way_lines` etc. untouched); the
Canal railway as one hand-listed line (Panamá (Corozal) - Colón, no intermediate passenger
stops), traced over OSM's track by rinf.py, or ar_register's route-ways recipe if OSM has a
passenger route relation for it. Small enough for a shared small-countries reader with Ecuador,
the Dominican Republic and Puerto Rico.

Expected: 3 lines (2 metro + Canal railway), ~107 km (18.1 + 22.5 + ~66 of named track;
the 76 km often quoted is Panama Railroad's own figure), ~37 stations. The Canal railway's
named track would also do it the Korea way (one name, one line).

### Open questions

- The Canal railway's passenger train after 2025: panarail.com gives no timetable; travel
  sellers still sell it. Built as running unless a suspension turns up.
- Línea 2's airport branch: whether OSM has it as a separate route (then OSM lines split it).
