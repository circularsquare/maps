# Chile register sources (built 2026-10-03)

What `cl_register.py` reads (with ar_register.py's code), what runs and what does not, the
line calls and why, and how the build checks out. Nothing here needed a login, a key or an
account; `data/raw/cl/` holds nothing yet (the reader needs only the extract).

## The short answer

- **Lines: the legal lines' passenger stretches.** OSM names Chile's track for its legal
  line (88% of main-line km: "Línea Central Sur", "Ramal San Rosendo - Talcahuano", "Ramal
  Talca-Constitución"), but almost all of it is freight or closed: Línea Central Sur runs
  1,245 km from Santiago to Puerto Montt and passenger trains use three stretches of it. So,
  as in Argentina, a register line is the track of the OSM passenger routes listed for it
  (`LINES`), named for the legal line and the stretch, or for the one service that runs the
  stretch whole.
- **7 register lines, 767 km**, all running. 25 lines in all with OSM's (Santiago's Metro,
  EFE's services, the Funicular de Santiago), 252 stations.
- **Checks**: five lines against es.wikipedia, 0.99 - 1.00.

## What runs (checked 2026-10-03) and what this build does with it

| service | status | in the build |
|---|---|---|
| Tren Nos (Alameda - Nos), Tren Rancagua, Tren San Fernando, Tren Curicó - Linares, Tren Chillán (TerraSur, five each way a day) | running | track: register line "Línea Central Sur (Alameda – Chillán)"; the services stay OSM lines over it |
| Expreso Chillán (four stops) | running, a few a day | named train (rules/cl.py) |
| Tren Victoria - Temuco, Tren Temuco - Pitrufquén | running (Pitrufquén six a day since March 2025) | register line "Línea Central Sur (Victoria – Pitrufquén)"; the two services stay OSM lines |
| Tren Llanquihue - Puerto Montt | running (220,000 riders in 2025; Saturdays added for summer 2026) | register line "Línea Central Sur (Llanquihue – Puerto Montt)" |
| Tren Santiago - Temuco (night train) | long weekends only in 2026 | not counted (under "more often than about once a week"); OSM has no route, and Chillán - Victoria is no register line |
| Corto Laja (Tren Laja - Talcahuano) and Biotren L1 (Mercado - Hualqui) | running | register line "Tren Laja – Talcahuano" (Ramal San Rosendo - Talcahuano, Laja - Talcahuano); Biotren L1 stays an OSM line over it |
| Biotren L2 (Concepción - Coronel) | running | register line "Biotren Línea 2" (Ramal Concepción - Curanilahue, Concepción - Coronel) |
| Buscarril Talca - Constitución | twice a day each way (2026 timetable) | register line "Tren Talca – Constitución" |
| Merval, Limache - Puerto | running | register line "Tren Limache – Puerto" |
| Metro de Santiago lines 1, 2, 3, 4, 4A, 5, 6 | running | OSM lines |
| Metro Line 7 | under construction (2027 at the soonest) | not built: `--clip` drops its routes, which run over track OSM names "Línea 7 (en construcción)" |
| Tren Alameda - Melipilla, Santiago - Batuco | under construction / projected | not built (`--clip` drops their routes) |
| Tren Arica - Poconchile | a few Saturdays a year (29 Aug, 3 Oct, 5 Dec 2026) | named train; its track is no register line |
| Tacna - Arica (Peru's FCTA) | no passenger train in 2026 (track being rebuilt) | named train as a stopgap (rules/cl.py): build_model cannot grey an OSM line |
| El Valdiviano, Tren del Recuerdo, Góndola Carril (excursions) | excursions | named trains |
| Funicular de Santiago | running | OSM line |
| Valparaíso's ascensores | several running | not built: OSM has no route relation for any (26 funicular ways left out by build_tiles); open below |

**Borders.** No passenger train crosses to Argentina, Bolivia or Peru. The Geofabrik extract
holds the Tren del Fin del Mundo (Ushuaia, Argentina), Bolivia's Viacha - Charaña buscarril,
the Río Turbio coal line and stubs of the Arica - La Paz and Uyuni lines; `--clip` drops them
(the area more than ~1.5 km beyond Chile's outline in religiondots' shapes).

## Sources

- **OSM** (Geofabrik `chile-latest`, extracted into `data/proc/cl` 2026-10-03; ODbL), then
  `python cl_register.py --clip`.
- **Published lengths**: es.wikipedia infoboxes of the services (Tren Estación Central -
  Chillán 397.6 km, Tren Victoria - Temuco 65.5, Tren Pitrufquén - Temuco 29.6, Tren Laja -
  Talcahuano 87.3, Tren Talca - Constitución 88, Tren Limache - Puerto 43); Santiago's Metro
  from es.wikipedia's Metro de Santiago article.
- **Timetable feeds** looked at and not used: the DTPM feed (Mobility Database mdb-3357,
  `dtpm.cl`, 4 July 2026) has Santiago's Metro and EFE's Santiago-area trains but nothing
  south of the Metropolitan Region; no national EFE feed was found. Every Chilean register
  section lies between stations a route stops at, so there is nothing for gtfs_served to
  decide.

## The line calls

- **Register lines are the legal lines' stretches that passenger trains run**, as Korea's are
  its legal lines: Línea Central Sur in three pieces, each a line of its own (one line in
  three far-apart pieces would show only the first in the strip diagram, HANDOFF's
  n02.walk_order note), named "Línea Central Sur (Alameda – Chillán)" and so on. EFE's
  services over them stay OSM lines, as Korea's 1호선 does over 경부선.
- **Where one service runs a stretch whole, the register line takes the service's name**
  (Tren Laja – Talcahuano, Biotren Línea 2, Tren Talca – Constitución, Tren Limache – Puerto),
  with the legal line in `name_en`, so OSM's line for it is the register line's twin and is
  dropped. Kept apart they were the same line twice, and the buscarril's OSM copy measured
  131 km for 88 (its two directions list different request stops).
- **Alameda - Chillán is one line** (399.3 km, 28 stations): five services nest along it
  (Nos, Rancagua, San Fernando, Curicó - Linares, Chillán), and the legal line runs on.
- **The Expreso Chillán is a named train**, the Chillán service a line (five a day each way,
  stopping at all twelve of its stations).
- **The Santiago - Temuco night train is not counted**: in 2026 it runs on long weekends only.

## Checks

`python check_model.py --region cl` (2026-10-03):

| line | built | published | ratio |
|---|---|---|---|
| Línea Central Sur (Alameda – Chillán) | 399.3 | 397.6 | 1.00 |
| Línea Central Sur (Victoria – Pitrufquén) | 94.2 | 95.1 (65.5 + 29.6) | 0.99 |
| Tren Laja – Talcahuano | 86.9 | 87.3 | 1.00 |
| Tren Talca – Constitución | 88.3 | 88 | 1.00 |
| Tren Limache – Puerto | 42.9 | 43 | 1.00 |
| Metro 1, 2, 4, 6 (OSM lines) | 19.0, 24.4, 23.1, 14.4 | 20, 25.9, 24.7, 15 | 0.93 - 0.96 (published lengths take in tail track) |

Not checked: Llanquihue - Puerto Montt (27.4 km, Llanquihue - La Paloma) and Biotren Línea 2
(28.3; es.wikipedia's 23 km for Concepción - Coronel does not fit its own 66.6 km for the two
lines, and OSM's L1 measures 39.7, so 28.3 is the likelier).

## Commands

    python extract.py --region cl --pbf data/raw/chile-latest.osm.pbf         # ~45 s
    python cl_register.py --clip            # after every extract
    python cl_register.py --report
    python build_model.py --region cl --register cl_register:data/raw/cl      # ~30 s
    python build_tiles.py --region cl                                         # ~5 s, 0.5 MB
    python check_model.py --region cl

## Open

- Valparaíso's ascensores: OSM has their track named but no route relations; they would
  need an `extent`-style track line each, with their two stations.
- Watch for: Metro Line 7 and Alameda - Melipilla opening (`--clip` drops "(en
  construcción)" routes; once OSM renames them they build by themselves), the Santiago -
  Temuco train going weekly, Tacna - Arica's return (then take it off `NAMED`).
