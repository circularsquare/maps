# Cuba register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe): four register lines from the four national trains'
OSM routes, the Línea Central first so each branch is the stretch its train adds.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Línea Central: La Habana – Santiago de Cuba | 835.2 | 15 | 835 (es.WP) | 1.00 |
| Línea a Bayamo – Manzanillo (Martí - Manzanillo) | 184.5 | 9 | - | - |
| Ramal a Guantánamo (San Luis - Guantánamo) | 78.3 | 5 | - | - |
| Ramal a Holguín (Cacocum - Holguín) | 17.7 | 2 | - | - |

4 register lines, 1,116 km, 29 stations, all running; the four trains are named trains over
them (rules/cu.py), so nothing else counts.

Calls:
- **Only the four national trains run**, as the survey found. `--clip cu` keeps only their
  four route=train relations (`train_routes_only`) and drops the other 108 OSM train routes
  and the route=tram "Tren Urbano de Las Tunas" (no source shows it running). Their track is
  drawn as base track, uncounted.
- **Stations are the trains' stops in OSM** (15 on the Línea Central: La Habana, Jaruco,
  Matanzas Central, Jovellanos Central, Colón, Santa Clara, Guayos, Ciego de Ávila, Florida,
  Camagüey, Las Tunas, Mir, Cacocum, San Luis - Combinado, Santiago de Cuba). The many other
  station records along the line are where no national train stops in 2026.
- **The Guantánamo branch** leaves the trunk east of San Luis with no station at the
  junction: a shared extent from San Luis (as ar's Lobos) gives it a first section that
  starts at a station.
- The Bayamo - Manzanillo train leaves the trunk at Martí (OSM's route), not Camagüey -
  Bayamo's old regional line.

Commands:

    python extract.py --region cu --pbf data/raw/cuba-latest.osm.pbf       # ~20 s
    python latam_register.py --clip cu      # after every extract
    python build_model.py --region cu --register latam_register:data/raw/cu   # ~30 s
    python build_tiles.py --region cu
    python check_model.py --region cu

## Survey (2026-10-08)

### What runs

Unión de Ferrocarriles de Cuba (UFC). Since the fuel crisis the national trains run on fixed
dates, each route about every eight days (from 16 March 2026; Directorio Cubano publishes the
dates monthly). April 2026, for example:

| route | departures from Havana | back |
|---|---|---|
| La Habana - Santiago de Cuba | 5, 13, 21, 29 Apr | 3, 11, 19, 27 Apr |
| La Habana - Guantánamo | 4, 8, 16, 24 Apr | 2, 6, 14, 22, 30 Apr |
| La Habana - Holguín | 2, 10, 18, 26 Apr | 4, 12, 20, 28 Apr |
| La Habana - Bayamo - Manzanillo | 3, 7, 15, 23 Apr | 5, 13, 21, 29 Apr |

All four run the Línea Central (La Habana - Matanzas - Santa Clara - Ciego de Ávila - Camagüey -
Las Tunas, 835 km to Santiago), so the trunk has a train most days and each branch one every
eight days. Summer specials ran in 2026 (24 Jun), and the trains still run: a Santiago -
Havana train derailed near Las Tunas on 11 Jul 2026 with ~900 aboard.

Counting: every eight days is "about weekly", so all four routes count, and their whole track
is register line: Línea Central La Habana - Santiago, Cacocum - Holguín, the Guantánamo branch
(from San Luis? via the Línea Central's eastern end), and Bayamo - Manzanillo (via the
Cauto / Santa Rita line). Lines with no national train in 2026 (La Habana - Pinar del Río,
- Cienfuegos, - Sancti Spíritus, which ran in 2019 every 2-3 days) are left out until a dated
timetable shows them.

Not running: the **Hershey electric train** (Casablanca - Matanzas), no service since 1 May 2017;
a revival was announced in May 2026 without dates (cubaheadlines.com). Provincial local trains
(trenes locales / ferroómnibus) exist in several provinces, but no timetable is published
anywhere reachable and the fuel crisis has cut many; not built (open question).

### Sources

- Directorio Cubano (directoriocubano.info): "Trenes nacionales en Cuba: publican las salidas
  confirmadas para abril de 2026", "fechas de salida ... segunda mitad de marzo", "Cambian las
  salidas de trenes nacionales en Cuba por la falta de combustible"; the tag page
  `/tag/transporte-ferroviario-en-cuba/` for the latest (July 2026 derailment, June summer
  specials). The dates come from UFC's Viajeros agency posts.
- es.wikipedia "Ferrocarriles de Cuba": Línea Central 835 km; the 2019 route list.
- No feed in the Mobility Database.
- **OSM**: Overpass summary in `data/raw/cu/survey/osm_summary.txt` (below).
- Geofabrik `central-america/cuba-latest.osm.pbf`, 59 MB.

### OSM (Overpass, 2026-10-08; `data/raw/cu/survey/osm_summary.txt`)

- **113 route=train relations**, almost all PTv2, with UFC train numbers as `ref`: the
  national trains (Tren Habana-Santiago 1/2 r6520980, Habana-Guantánamo 3/4 r6520977,
  Habana-Holguín 5/6 r6520978, Habana-Bayamo-Manzanillo 7/8 r6520976, Habana - Pinar 71/72,
  Habana - Cienfuegos 19/20, Habana - Sancti Spíritus 103/104, Santa Clara - Santiago 101/102),
  regional trains (Camagüey - Bayamo 81/82, Santiago - Manzanillo 638/639, Holguín - Santiago
  610/611, Cienfuegos - Santa Clara 331/332 ...), Havana's commuter trains (refs 1-9 from La
  Coubre and Tulipán), and ~40 provincial ferrobús and sugar-estate cochemotor routes
  (Amancio, Manatí, Guantánamo, Banes...). Plus `route=railway` relations for some legal lines
  (Ferrocarril Central r12648771, Línea Sur, Línea Oeste, Ramal Guines...).
- **It is a pre-crisis inventory, not today's timetable**: the Hershey routes (8, 8A, 8B) are
  still there though nothing has run since 2017. Which of the 113 run now cannot be told from
  OSM.
- Track: 1,298 km rail usage=main, 3,729 km usage=branch (mostly sugar lines), largely unnamed.

### Recipe

ar_register's recipe (the ways of hand-listed OSM passenger routes): four register lines from
the four national routes' relations (r6520980, r6520977, r6520978, r6520976), each way to the
first line that lists a route over it: Línea Central (La Habana - Santiago) first, then the
Holguín, Guantánamo and Bayamo - Manzanillo branches as the stretches those routes add. Stations
from the routes' stops. The trains themselves are named trains over the lines. Every other OSM
train route dropped by `--clip` (a keep list, the inverse of nafrica's `NOT_SERVICE`), since
none can be shown to run.

Expected: 4 register lines, roughly 1,150 km (835 Línea Central + ~25 Holguín branch + ~90
Guantánamo branch + ~200 to Bayamo - Manzanillo off the trunk).

### Open questions

- The branches' exact routings fall out of OSM's route relations (no need to settle by hand);
  check the four routes are unbroken in the extract.
- Pinar del Río, Cienfuegos, Sancti Spíritus: any 2026 dated trains (then add them).
- Local trains: OSM has ~80 local and commuter routes, but no 2026 timetable source shows which
  run; would need Viajeros' or the provincial UFC units' posts (Facebook). Left out for now;
  this is the biggest gap (several hundred km of possibly-running local lines).
