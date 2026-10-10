# Colombia register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe). Two register lines; Medellín's metro lines are
OSM lines.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Tren Turístico de la Sabana (Usaquén - Zipaquirá), register | 37.8 | 4 | - | - |
| Tranvía de Ayacucho (T-A), register | 4.0 | 8 | 4.2 (en.WP) | 0.96 |
| Metro Línea A (OSM) | 24.9 | 22 | 25.8 | 0.96 |
| Metro Línea B (OSM) | 5.5 | 7 | 5.5 | 1.00 |

4 lines, 72 km, all running.

Calls:
- **The Sabana train starts at Usaquén**, not Bogotá's Estación de la Sabana: OSM has no
  passenger route, its "Ferrocarril de La Sabana" route=railway relation did not come through
  the extract, and the track OSM does have stops near Calle 63 without joining anything that
  reaches La Sabana (the ways there are a "Línea 1" heading west). So the line is an extent
  through Usaquén, La Caro, Cajicá and Zipaquirá, and the ~15 km Sabana - Usaquén is left
  out until OSM maps it. Weekend and holiday trains: counted (weekly).
- **The Ayacucho tram is a register line**: OSM's two routes list one stop each (the others
  are stop_area relations of platform ways, which extract.py does not read), so build_model's
  OSM half drops them as variants with under two stops. The register line takes the routes'
  ways and its stations from the stop_areas' platform centroids (read once from the extract;
  `CO_TRAM`). Miraflores has no stop_area in OSM and is left out.
- **Dropped by the clip**: the Bogotá Metro route (under construction), the Barrancabermeja -
  Puerto Berrío ferrobús (not running), and an unnamed route=train over 293 ways of the
  Central freight corridor with no stops.
- Duitama - Sogamoso: still no timetable found; not built.
- OSM has Poblado twice on Línea A (a stop and "Estación del Metro Poblado"), so A shows 22
  stations for 21; OSM's names ("Estación del Metro Itagüí") are left as they are.

Commands:

    python extract.py --region co --pbf data/raw/colombia-latest.osm.pbf
    python latam_register.py --clip co
    python build_model.py --region co --register latam_register:data/raw/co
    python build_tiles.py --region co
    python check_model.py --region co

## Survey (2026-10-08)

### What runs

| service | status | in the build |
|---|---|---|
| **Metro de Medellín** Línea A, Niquía - La Estrella, 25.8 km, 21 stations | running | line |
| Línea B, San Antonio - San Javier, 5.5 km, 7 stations | running | line |
| **Tranvía de Ayacucho** (T-A), San Antonio - Oriente, 4.2 km, 3 stations + 6 stops; Translohr rubber-tyred guided tram | running since 2016 | line, `kind` tram (it is guided on one rail; OSM tags it as tram) |
| **Tren Turístico de la Sabana** (Turistren), Bogotá La Sabana - Usaquén - La Caro - Cajicá - Zipaquirá, ~53 km | Saturdays, Sundays and holidays, one train each way (08:20 out, back ~17:15) | register line Bogotá - Zipaquirá (weekly: counts) |
| Duitama - Sogamoso (Boyacá), Acerías Paz del Río, on the Bogotá - Belencito line | started 18 Jul 2025 under a Ministry authorisation (ANI); 156 seats, ~2 h; no timetable or frequency published anywhere found, and the "tren turístico de Boyacá" runs at Easter (Valora Analitik) | not counted until a timetable shows it weekly (open question) |
| Ferrobús Barrancabermeja - Puerto Berrío (Coopsercol), 130 km | not running: the La Dorada - Chiriguaná corridor is freight only; a March 2026 memorandum starts studies to bring passengers back (Vanguardia, 13 Mar 2026). OSM still has its route r13708000 | not built (drop the route) |
| Bogotá Metro Línea 1 | under construction, opening 2028 at the soonest | not built |
| Regiotram de Occidente (Bogotá - Facatativá) | under construction | not built |
| Metro de la 80 (Medellín) | under construction | not built |
| Cerrejón railway (La Guajira), Fenoco (Cesar - Santa Marta) | coal freight | not built |
| Metrocable (Medellín), TransMiCable (Bogotá), Metroplús | not rail | not built |

### Sources

- Metro de Medellín (metrodemedellin.gov.co); en.wikipedia "Medellín Metro" for lengths
  (A 25.8, B 5.5, T-A 4.2 km).
- Tren de la Sabana: Turistren (turistren.com.co), bogota.gov.co ("Trece razones para subirse al
  Tren Turístico de la Sabana"), Cambio Colombia and Xataka (2025-26: schedules, stops).
- Mobility Database has Bogotá's SITP bus feed (mdb-3358, 2026-04-29) and other bus feeds,
  nothing for Medellín's metro or the Sabana train.
- **OSM**: Overpass summary in `data/raw/co/survey/osm_summary.txt` (below).
- Geofabrik `south-america/colombia-latest.osm.pbf`, 315 MB.

### OSM (Overpass, 2026-10-08; `data/raw/co/survey/osm_summary.txt`)

- Medellín: PTv2 routes both ways for Línea A, Línea B (operator ETMVA) and the T-A tram
  (#009933); no colour on A and B (Wikidata or a picked colour, A blue / B orange as the
  operator's map). No route_masters.
- **The Sabana train has no passenger route relation**: only `route=railway` "Ferrocarril de La
  Sabana" (r8303110) and "La Caro–Zipaquirá" (r8303118), and 28.6 km of track named
  "Ferrocarril de La Sabana" (most of Bogotá's track is unnamed).
- **Traps**: "Metro de Bogotá" route r19566688 (route=subway, ref 1) exists for a line under
  construction (track "Línea 1" 43.6 km railway=construction): drop it. The ferrobús route
  r13708000 (not running): drop it.
- 1,662 km of rail usage=main, mostly freight (Fenoco, Cerrejón is `industrial`).

### Recipe

Medellín's three lines as OSM route relations (US/UK/BR metro rule). The Sabana train as one
hand-listed line (Bogotá La Sabana, Usaquén, La Caro, Cajicá, Zipaquirá) traced over OSM track
by rinf.py (nafrica's hand-list recipe), since OSM has no passenger route for it; or ar_register's
`extent` (track between two named stations) over the two route=railway relations. A `--clip`
drops the Bogotá Metro and ferrobús routes.

Expected: 4 lines, ~89 km, ~45 stations.

### Open questions

- Whether some Sabana trains run on past Zipaquirá to Nemocón (historically on some dates); a
  few dates a year would not count.
- Medellín's T-A tram: OSM tags its track railway=tram (8.3 km of tram ways); a Translohr runs
  on one guide rail, which build_model draws like any tram.
- Duitama - Sogamoso: find a timetable (Acerías Paz del Río); add it as a register line if it
  runs at least weekly.
