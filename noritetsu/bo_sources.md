# Bolivia register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe): four register lines from the weekly trains' OSM
routes; the trains themselves are named trains (rules/bo.py); Cochabamba's Mi Tren lines are
OSM lines.

| line | built km | stations | published | ratio |
|---|---|---|---|---|
| Ferrocarril Santa Cruz – Puerto Quijarro | 638.7 | 6 | 651 (FO sector Este) | 0.98 |
| Ferrocarril Oruro – Villazón | 599.3 | 5 | - | - |
| Ferrocarril Santa Cruz – Yacuiba | 536.1 | 5 | 539 to Pocitos | 0.99 |
| Ferrocarril Viacha – Charaña | 207.1 | 5 | ~210 (survey) | 0.99 |
| Línea Verde, Cochabamba (OSM) | 27.7 | 25 | 27 (Los Tiempos) | 1.02 |
| Línea Roja (OSM) | 8.4 | 11 | - | - |
| Línea Amarilla (OSM) | 6.0 | 6 | - | - |

4 register lines, 1,981 km, all running; plus Mi Tren's three lines, 42 km.

Calls:
- **Oruro - Villazón** takes the ways of the Expreso del Sur's route (which has ways only to
  Uyuni) and of the old "Uyuni-Villazón" route r3397889 (kept for its ways; a named train like
  the others). Stations as FCA's 2026 ferrobús timetable: Oruro, Uyuni, Atocha, Tupiza,
  Villazón.
- **Santa Cruz - Yacuiba starts at Santa Cruz**: its first km are the Quijarro line's track,
  taken as a shared extent (as ar's Lobos) so the line's first section starts at a station.
- **Santa Cruz - Puerto Quijarro stations**: Santa Cruz, Cotoca, San José de Chiquitos,
  Roboré, El Carmen Rivero Torrez, Puerto Quijarro. Pailón, Aguas Calientes and Puerto Suárez
  are calls too but have no OSM record: left out.
- **Línea Amarilla counts**: OSM has it both ways, hourly, and nothing says it stopped.
  Línea Verde runs to Suticollo (Los Tiempos, April 2024: every 30 minutes the whole way).
- **Dropped by the clip**: El Alto - Guaqui both ways, Potosí - Sucre, the Vila Vila - Aiquile
  buscarril, Oruro's Tren Urbano Qamaqi both ways, the Oruro - Machacamarca tourist train
  (none in FCA's 2026 timetables).
- **Borders**: no passenger train crosses any (Villazón - La Quiaca, Charaña - Visviri,
  Quijarro - Corumbá, Yacuiba - Pocitos), so no border points are proposed.

Commands:

    python extract.py --region bo --pbf data/raw/bolivia-latest.osm.pbf
    python latam_register.py --clip bo
    python build_model.py --region bo --register latam_register:data/raw/bo
    python build_tiles.py --region bo
    python check_model.py --region bo

## Survey (2026-10-08)

### What runs

Bolivia has two unconnected metre-gauge networks plus Cochabamba's light rail. Everything below
runs at least weekly; La Paz's Mi Teleférico is a cable car and is not rail.

**Western network, Ferroviaria Andina (FCA)** (ferroviaria-andina.com.bo, timetables saved
2026-10-08):

| service | days | stations | in the build |
|---|---|---|---|
| Expreso del Sur - Ferrobús, Oruro - Uyuni - Atocha - Tupiza - Villazón | out Monday night (Oruro 21:30, Uyuni 04:23 Tue, Atocha 08:08, Tupiza 11:43, Villazón 16:30); back Thursday (Villazón 14:30, Tupiza 17:20, Atocha 21:40, Uyuni 00:03 Fri, Oruro 07:55) | Oruro, Uyuni, Atocha, Tupiza, Villazón (FCA lists only these; the old Expreso del Sur also called at Poopó, Challapata, Río Mulato) | register line Oruro - Villazón, weekly: counts |
| Buscarril Atocha - Tupiza | Monday, Thursday, both ways | Atocha, Tupiza | over the same line |
| Buscarril Viacha - Charaña | out Monday and Thursday 08:30 - 14:10, back Tuesday and Friday 09:00 - 14:40 | Viacha ... Charaña (Chilean border; the Arica line) | register line Viacha - Charaña |

The Wara Wara del Sur (the slower Oruro - Villazón train) is gone; the ferrobús replaced both.
The El Alto - Tiwanaku - Guaqui tourist train (FCA, 2019 news) and the Sucre - Potosí buscarril
appear only in old news; FCA's 2026 site lists neither: not counted. Uyuni - Avaroa (to Chile)
is freight only. Cochabamba's "Bus Carril" (FCA, 2024: "debidamente resguardado") is not
running.

**Eastern network, Ferroviaria Oriental (FO)**:

| service | status | in the build |
|---|---|---|
| Santa Cruz - Puerto Quijarro (Expreso Oriental, ferrobús) | once a week each way at least: Expreso Oriental out Friday 13:00, back Sunday (IRJ, on its restoration); stations Cotoca, Pailón, San José de Chiquitos, Roboré, Aguas Calientes, Puerto Suárez, Quijarro | register line Santa Cruz - Puerto Quijarro, 651 km |
| Santa Cruz - Yacuiba | one train a week, arriving Yacuiba Wednesday, back the same day 15:00, ~24 h (prensamercosur.org, 30 Sep 2026: Yacuiba promoting it against bus fares) | register line Santa Cruz - Yacuiba (539 km sector Sur less the Pocitos stub) |
| Santa Cruz - Montero (sector Norte, 62 km) | freight | not built |

FO's own website (fo.com.bo) now shows only freight and logistics; passenger timetables are not
on it. Both lines' current days need a check at build time (open questions).

**Cochabamba, Tren Metropolitano (Mi Tren)**, metre-gauge light rail (Stadler trains) on the
old FCA track:
- Línea Verde: San Antonio (central station) - Colcapirhua - Quillacollo (- Vinto - Sipe Sipe
  planned), every half hour, 27 trips a day (2023).
- Línea Roja: San Antonio - Av. Petrolera km 7.5 (Santa Vera Cruz station, extended April 2024).
- Línea Amarilla: platform works Nov 2024; OSM now has its routes both ways (Cochabamba -
  Maica Chica, hourly), so it looks open: counted if a 2025-26 source confirms it.
Running as of the last news found (Los Tiempos, Nov 2024: Verde back to normal after works);
nothing found for 2025-2026 saying it stopped.

### Sources

- **FCA timetables**: `https://ferroviaria-andina.com.bo/Ferrobus`, `/Buscarril` (saved as
  `data/raw/bo/survey/fca_ferrobus.html`, `fca_buscarril.html`; the timetable is in the page's
  JSON). The old host www.fca.com.bo redirects there and has an expired certificate.
- **FO**: no timetable online; IRJ ("Bolivia's Eastern Railway restores express passenger
  service", railjournal.com) for the Quijarro train and its stops; prensamercosur.org
  (30 Sep 2026; 403 to a script, read through the search snippet) for Yacuiba.
- **Cochabamba**: Trufi Association's GTFS built from OSM (Mobility Database mdb-3507,
  `https://raw.githubusercontent.com/trufi-association/trufi-gtfs-builder/refs/heads/main/examples/Bolivia-Cochabamba/out/cochabamba.gtfs.zip`,
  ODbL) carries the Tren Metropolitano, but it is OSM restated, so no second source.
- **OSM**: Overpass summary in `data/raw/bo/survey/osm_summary.txt` (below).
- Geofabrik `south-america/bolivia-latest.osm.pbf`, 166 MB.

### OSM (Overpass, 2026-10-08; `data/raw/bo/survey/osm_summary.txt`)

- **Every running long-distance service has a PTv2 route**: Expreso del Sur Oruro - Villazón
  r2082775 (one direction; plus "Uyuni-Villazón" r3397889, old), Expreso Oriental Santa Cruz -
  Puerto Quijarro r8262149 (one direction, `interval=168:00`, i.e. weekly), Ferrobús T-1 / T-2
  Yacuiba - Santa Cruz r8262212 / r2084750 (weekly), Buscarril Viacha → Charaña r3397885 (one
  direction).
- **Stale routes to drop**: El Alto - Puerto Guaqui both ways (the tourist train), Potosí -
  Sucre, Buscarril 253 Vila Vila - Aiquile, Oruro's "Tren Urbano Qamaqi" both ways, "Tren
  Turístico a Oruro - Machacamarca". None appears in FCA's 2026 timetables.
- **Cochabamba**: PTv2 routes both ways for Línea Roja (Cochabamba - Kiñiloma, every 15 min),
  Línea Verde (Cochabamba - Suticollo, every 30 min) and **Línea Amarilla** (Cochabamba -
  Maica Chica, hourly; the reverse route added recently, r19604339), operator "Mi Tren". So
  the Amarilla appears to be open; light_rail track 41.8 km.
- Legal lines are `route=railway` relations (Ferrocarril Oruro-Viacha, Río Mulatos-Potosí-Sucre,
  Santa Cruz-Puerto Quijarro, Santa Cruz-Yacuiba, Uyuni-Avaroa, Arica - La Paz...); 3,029 km of
  rail usage=main, mostly unnamed.

### Recipe

ar_register's recipe (the ways of hand-listed OSM passenger routes): four register lines from
the routes above, each named for the legal line and stretch ("Ferrocarril Oruro - Villazón",
"Viacha - Charaña", "Santa Cruz - Puerto Quijarro", "Santa Cruz - Yacuiba"), stations from the
routes' stops checked against the operators' timetables; the stale routes dropped by `--clip`.
nafrica's hand list through rinf.py is the fallback where a route is broken. Cochabamba's three
lines as OSM route relations (the metro rule in the US/UK/BR builds).

Expected: 4 register lines, about 2,000 km (Oruro - Villazón ~620, Viacha - Charaña ~210,
Santa Cruz - Quijarro 651, Santa Cruz - Yacuiba ~535), plus Cochabamba's 3 lines (~45 km).

Borders: Villazón faces La Quiaca (Argentina; no train on the Argentine side), Charaña faces
Visviri (Chile; no passenger train), Quijarro faces Corumbá (Brazil; no passenger train),
Yacuiba faces Pocitos (Argentina; freight). No border join is needed; `--clip` to Bolivia's
outline.

### Open questions

- FO's current passenger days for Quijarro (the IRJ article is older than 2026) and whether the
  Quijarro ferrobús still runs; checked only through news.
- Whether the ferrobús Oruro - Villazón calls at Poopó, Challapata and Río Mulato on request
  (FCA's page lists five stations only).
- Cochabamba's Línea Amarilla: OSM says open, no news found either way.
