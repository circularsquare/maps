# Venezuela register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe): one register line, Caracas - Cúa, from OSM's two
IFE routes (their El Sombrero part is railway=construction, which the recipe never takes, so
the line ends at Cúa by itself). The urban lines are OSM lines.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Sistema Ferroviario Ezequiel Zamora: Caracas – Cúa, register | 41.0 | 4 | 41.4 (es.WP) | 0.99 |
| Caracas Línea 1 (OSM) | 20.2 | 22 | 20.4 | 0.99 |
| Caracas Línea 2 (OSM) | 19.3 | 11 | - | - |
| Caracas Línea 3 (OSM) | 9.9 | 9 | - | - |
| Caracas Línea 4, with Línea 5's Bello Monte, as operated from Zoológico (OSM) | 24.3 | 14+ | - | - |
| Metro Los Teques (OSM) | 11.0 | 5 | 10.7 | 1.02 |
| Metro de Valencia Línea 1 (OSM) | 6.2 | 9 | - | - |
| Metro de Maracaibo Línea 1 (OSM) | 5.9 | 6 | 6.5 | 0.91 |
| Cabletrén Bolivariano (OSM) | 0.9 | 3 | - | - |

9 lines, 139 km, all running.

Calls:
- **Maracaibo's metro is running**: mapa-metro and other listings give its hours (Monday -
  Friday 6:00 - 21:00, weekends 8:00 - 18:00) and nothing found says it stopped.
- **Línea 4 runs over Línea 2's track from Zoológico** (OSM's routes, as operated): the fixed
  rule gives the 33.5 km of shared way to one of them (OSM lines, so the register rules do not
  apply).
- OSM's two IFE routes are renamed by the clip to the register line's name, so they group
  into one OSM line, the twin.
- The Cabletrén builds 0.9 km for ~2.2 km published: OSM's routes are short; left.
- Not built: Puerto Cabello - La Encrucijada (route=railway, track mostly under
  construction), the Centro Occidental's freight, Pijiguaos.

Commands:

    python extract.py --region ve --pbf data/raw/venezuela-latest.osm.pbf
    python latam_register.py --clip ve
    python build_model.py --region ve --register latam_register:data/raw/ve
    python build_tiles.py --region ve
    python check_model.py --region ve

## Survey (2026-10-08)

### What runs

| service | status | in the build |
|---|---|---|
| **Metro de Caracas** Línea 1 (Propatria - Palo Verde, 20.4 km), Línea 2 (El Silencio - Las Adjuntas / Zoológico, 17.8 km), Línea 3 (Plaza Venezuela - La Rinconada, 10.4 km), Línea 4 (Capuchinos - Plaza Venezuela - Zona Rental, 5.5 km), Línea 5 (Zona Rental - Bello Monte, one station, 2015) | running (kept running through the March 2026 transport strike); reliability poor, but scheduled | five lines (OSM may carry 4 and 5 as one, as they are operated) |
| **Metro Los Teques**, Las Adjuntas - Independencia (Línea 1 extended to Independencia, December 2013; ~10.7 km) | running | one line, as OSM's routes run it |
| **Metro de Valencia**, Monumental - Francisco de Miranda (Línea 1 plus the first stretch of Línea 2, 2015; ~7.7 km, 9 stations), operated as one line | running | one line, as OSM's routes run it |
| **Metro de Maracaibo** Línea 1 (Altos de la Vanega - Libertador, 6 stations) | intermittent; status in 2026 unconfirmed | OSM line if running (open question) |
| **Sistema Ferroviario Central Ezequiel Zamora** (IFE), Caracas (Libertador Simón Bolívar, La Rinconada) - Charallave Norte - Charallave Sur - Cúa, 41.4 km, 4 stations | running daily (the only conventional passenger railway) | register line |
| Cabletren Bolivariano (Petare - 5 de Julio, cable-hauled people mover, ~2.2 km) | running as far as known (Metro de Caracas) | OSM line (its routes exist) |
| Centro Occidental Simón Bolívar (Puerto Cabello - Barquisimeto - Acarigua) | freight reactivated; passenger trains reported only as trials | not built |
| Metrocable, Trolmérida, BusCaracas | not rail | not built |

### Sources

- urbanrail.net Caracas page (lengths; last updated long ago), es/en.wikipedia "Metro de
  Caracas", "Metro de Los Teques", "Metro de Valencia", "Metro de Maracaibo", "Sistema
  Ferroviario Ezequiel Zamora" for station lists and published lengths.
- No Venezuelan feed in the Mobility Database (2026-10-08).
- **OSM**: Overpass summary in `data/raw/ve/survey/osm_summary.txt` (below).
- Geofabrik `south-america/venezuela-latest.osm.pbf`, 121 MB.

### OSM (Overpass, 2026-10-08; `data/raw/ve/survey/osm_summary.txt`)

- **Caracas Metro**: PTv2 routes both ways for L1 (#ff7400), L2 (#00dc3c; El Silencio - Las
  Adjuntas and - Zoológico), L3 (#0887ff), L4 (#fff000; Línea 5's Bello Monte station is in the
  L4 routes "Dirección Bello Monte", as it is operated). Track named per line ("Línea 1 del
  Metro de Caracas" 41.6 km of ways ...).
- **Los Teques**: one line, routes Las Adjuntas - Independencia both ways (ref T1); its Línea 2
  is not mapped as a route.
- **Valencia**: routes "Línea 1: Monumental - Francisco de Miranda" both ways (light_rail,
  #cc0000), track "Línea 1 del Metro de Valencia" 13.7 km.
- **Maracaibo**: routes "Línea 1: Altos de la Vanega - Estación Libertador" both ways
  (light_rail, green).
- **Cabletrén Bolivariano**: routes both ways (light_rail, ref CAT, Petare - 5 de Julio).
- **Caracas - Cúa**: r4651199 / r4651227 "Ferrocarril, Línea 1. Caracas => Cúa => El
  Sombrero" (operator IFE). The El Sombrero part is unbuilt (the Tinaco - Anaco and Cúa - El
  Sombrero track is railway=construction, 515 km of it): the route must be cut at Cúa. Track
  "Línea 1, Ferrocarril Ezequiel Zamora" 41.1 km of ways.
- **Puerto Cabello - La Encrucijada**: r5373774 / r5373775 "Tren, Línea 2" (IFE, PTv2 tags
  but `route=railway`), over track mostly railway=construction ("Línea Puerto Cabello – La
  Encrucijada" 114.9 km). Not a running service; not built.
- Freight: "Línea Puerto Cabello – Sabana de Mendoza" 169 km (Centro Occidental), Pijiguaos
  (CVG Ferrominera).

### Recipe

Urban lines as OSM route relations (US/UK/BR metro rule). Caracas - Cúa as ar_register's
route-ways recipe over r4651199 / r4651227, cut at Cúa (an `extent` as ar_register's Cañuelas
- Lobos), stations Libertador Simón Bolívar (La Rinconada), Charallave Norte, Charallave Sur,
Cúa. The route_master-less metro routes build as OSM lines as they stand.

Expected: 1 register line (Caracas - Cúa, ~41 km) plus ~8 OSM lines (Caracas L1-L4, Los
Teques, Valencia, Maracaibo, Cabletrén), ~150 km in all.

### Open questions

- Metro de Maracaibo: running or not in 2026 (no news found; check OSM and recent press at
  build time; build it greyed if stopped).
- Puerto Cabello - Barquisimeto passenger trials (El Ciudadano reports, 403 to a script): watch.
