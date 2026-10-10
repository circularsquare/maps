# Ecuador register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe): one register line, the Nariz del Diablo, from
OSM's routes r2016146 / r2016147; Quito's metro and Cuenca's tram are OSM lines.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Tren Nariz del Diablo (Alausí - Sibambe), register | 9.2 | 2 | 12.5 | 0.74 |
| Metro Línea 1, Quito (OSM) | 21.4 | 15 | 22.6 | 0.95 |
| Tranvía Cuatro Ríos, Cuenca (OSM) | 10.0 | 22 | 10.7 | 0.93 |

3 lines, 41 km, all running.

Calls:
- **The Nariz del Diablo is short of its 12.5 km by the switchback**: trains run into each
  reversing tail and back out, and a section is a shortest path between the two stations,
  which cuts across the zigzag. OSM's route ways are 11.2 km in all. Left as is; check_model
  flags it with that note.
- **The clip drops all 14 closed Tren Ecuador routes by id** (Tren de los Volcanes, Tren a
  las Nubes / Transandino, Tren de las Maravillas, Tren del Hielo II, Baños del Inca, Tren de
  la Libertad, Tren de la Dulzura, Urbina - Sibambe, an unnamed one): nothing has run since
  2020. Their track is drawn as track.
- The clip names OSM's Nariz del Diablo route_master (it had no name) so it drops as the
  register line's twin.
- Cuenca's tram stops carry OSM's long names ("Tranvía - Nº 1 - Río Tarqui"); left as OSM
  has them (tram stops are not renamed by any rules hook).

Commands:

    python extract.py --region ec --pbf data/raw/ecuador-latest.osm.pbf
    python latam_register.py --clip ec
    python build_model.py --region ec --register latam_register:data/raw/ec
    python build_tiles.py --region ec
    python check_model.py --region ec

## Survey (2026-10-08)

### What runs

| service | status | in the build |
|---|---|---|
| **Metro de Quito**, Línea 1, Quitumbe - El Labrador, 22.6 km, 15 stations | running since 1 Dec 2023 (commercial); extension El Labrador - La Ofelia only in studies (Primicias, 2026) | one line |
| **Tranvía de Cuenca**, Río Tarqui - Parque Industrial, ~10.7 km, 27 stops (20 stations by some counts) | running since May 2020 | one line |
| **Tren Nariz del Diablo**, Alausí - Sibambe, 12.5 km (the switchbacks) | running since July 2025 (reopened officially 20 Aug 2025): Wednesday - Sunday and holidays, departures 08:00, 11:00, 14:00 from Alausí, round trips | one line, counts (daily-ish, scheduled) |
| Tren Ecuador's other routes (Quito - Machachi / Boliche, Durán - Yaguachi - Bucay, Riobamba - Urbina, Ibarra - Salinas "Tren de la Libertad") | closed since Ferrocarriles del Ecuador was wound up in 2020. The government calls Riobamba - Urbina, Ibarra - Salinas and Quito - Latacunga the next candidates; nothing scheduled | not built |
| Ibarra's "Tren Expreso Polar" | Christmas special in Ibarra (Fri and Sat evenings, 11 Dec 2025 - 6 Jan 2026) | not counted |

The Alausí - Sibambe trips are out-and-back excursions from Alausí, but they are scheduled
several times a day five days a week and sold as the line; counted, as Argentina counts the Tren
Solar de la Quebrada.

### Sources

- Quito: Metro de Quito (metrodequito.gob.ec); published length 22.6 km.
- Cuenca: Tranvía de Cuenca (tranvia.cuenca.gob.ec); published length ~10.7 km (es.wikipedia
  "Tranvía de Cuenca"; check at build time).
- Alausí - Sibambe: Primicias (primicias.ec, "Tren Nariz del Diablo: turismo, feriado, precio,
  horario"), Teleamazonas (reopening, Aug 2025), Vistazo (tests, 5 Jul 2025). 12.5 km published.
- No Ecuadorian feed in the Mobility Database (2026-10-08).
- **OSM**: Overpass summary in `data/raw/ec/survey/osm_summary.txt` (below).
- Geofabrik `south-america/ecuador-latest.osm.pbf`, 119 MB.

### OSM (Overpass, 2026-10-08; `data/raw/ec/survey/osm_summary.txt`)

- Quito Metro: PTv2 routes both ways (ref 1, #E31D1B), track "Línea 1 del Metro de Quito"
  43.4 km of ways. Cuenca: tram routes "Cuatro Ríos Ida / Vuelta" (ref "Linea 1", operator
  Alcaldía de Cuenca), track "Tranvía 4 Ríos" 19.8 km of ways.
- **Nariz del Diablo has routes both ways**: r2016146 "Alausi - Sibambe" and r2016147
  (ref "Nariz del Diablo", operator still "Ferrocarriles del Ecuador Empresa Publica", no PTv2).
- **Trap: the closed Tren Ecuador routes are all still in OSM** as running route=train:
  Tren de los Volcanes (Quito - Ambato both ways), Tren a las Nubes / Ferrocarril Transandino
  (Durán - Quito), Tren de las Maravillas, Tren del Hielo II (Ambato - Urbina), Baños del Inca
  (Coyoctor - El Tambo), Tren de la Libertad (Ibarra / Otavalo - Salinas), Tren de la Dulzura
  (Naranjito - Durán), "Ruta del tren Urbina - Sibambe" (r16002308), and an unnamed r16002309.
  They must be dropped by id (a `NOT_SERVICE` list as in nafrica_register's `--clip`), or they
  build ~500 km of lines that have not run since 2020. Track: 348 km named
  "Ferrocarriles del Ecuador E.P." as usage=main, plus usage=tourism stretches ("Urbina -
  Sibambe" 31.6, "Linea Ibarra - Salinas" 13.2).

### Recipe

Three short lines, two of them urban. Metro and tram as OSM route relations (the US/UK/BR
metro rule); the Nariz del Diablo as ar_register's recipe, the ways of OSM's routes r2016146 /
r2016147 (Alausí, Sibambe). The essential step is the clip: every other route=train relation
in Ecuador dropped by id. A shared small-countries reader on ar_register's code (with Panama,
the Dominican Republic, Puerto Rico, Uruguay, Costa Rica) would carry it.

Expected: 3 lines, ~46 km (22.6 + 10.7 + 12.5), ~45 stations.

### Open questions

- Whether OSM's Quito Metro routes list all 15 stations (they are route=subway, PTv2).
- Watch for: Riobamba - Urbina, Ibarra - Salinas reopening (2026 plans, no dates).
