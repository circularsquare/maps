# Mexico register sources (built 2026-10-03)

What `mx_register.py` reads, what runs and what does not, the line calls and why, and how the
build checks out. Downloads are in `data/raw/mx/` (gitignored); this file is the record. Nothing
here needed a login, a key or an account.

## The short answer

- **Lines: OpenStreetMap's named passenger track** (kr_register's recipe, as the UK). OSM names
  99.4% of Mexico's main-line rail track for its line (`python probe_kr_ways.py --region mx`):
  the ARTF's line letters on the freight network (Línea A, B, Q, T, Z...) and the new lines'
  own names ("Tren Maya", "El Insurgente", "Ferrocarril Suburbano Ramal 1", "Ferrocarril Felipe
  Ángeles"); metro track is named for its line too ("Línea 1" ... "Línea B", "Línea 1
  Metrorrey", "Línea 1 del Tren Eléctrico Urbano").
- **Only passenger track**, the way the US build keeps NARN's passenger-coded track. Almost all
  of Mexico's ~20,000 km of railway is freight (Ferromex, CPKC, Ferrosur) and is not a line.
- **Stations**: OSM's stop positions on the track, the stops OSM's route relations list, the
  railway=station nodes beside intercity track, and Wikidata's station items where OSM has no
  rail record (Chihuahua, Creel and Divisadero on the Chepe; the AIFA branch's six new stops;
  Línea K).
- **27 register lines, 3,376 km** (2,569 km running; three suspended lines, 807 km, greyed),
  34 lines in all, 379 stations.

## What runs (checked 2026-10-03) and what this build does with it

| service | status | in the build |
|---|---|---|
| Tren Maya, 1,554 km, 34 stations | running (all seven tramos since Dec 2024) | register line "Tren Maya" |
| El Insurgente (Tren Interurbano México - Toluca), Zinacantepec - Observatorio | running; Santa Fe - Observatorio opened 2 Feb 2026 | register line "El Insurgente" |
| Tren Suburbano, Buenavista - Cuautitlán | running | register line "Tren Suburbano" |
| Tren Felipe Ángeles, Lechería - AIFA (trains from Buenavista) | opened 26 Apr 2026 | branch of "Tren Suburbano" (below) |
| El Chepe Regional (Chihuahua - Los Mochis) and Express (Los Mochis - Creel, Mon/Thu/Sat out, Tue/Fri/Sun back) | running | register line "Chihuahua al Pacífico"; the Express a named train |
| Tren Interoceánico Línea Z (Coatzacoalcos - Salina Cruz), FA (Coatzacoalcos - Palenque), K (Ixtepec - Tonalá) | passenger trains suspended since the derailment of 28 Dec 2025; the government says early 2027 at the soonest; another derailment on Z in July 2026 | three register lines, `suspended` (greyed, out of completion) |
| Mexico City Metro (12 lines), Tren Ligero (Tasqueña - Xochimilco) | running | register lines |
| Metrorrey lines 1-3 | running | register lines |
| Metrorrey lines 4 and 6 (monorail) | not open: Line 6 missed the World Cup (Metrorrey, 1 June 2026); state now says end of 2027 | not built (OSM: "Línea 6 (en construcción)") |
| Mi Tren (Guadalajara) lines 1-4 | running; Line 4 (Las Juntas - Tlajomulco) opened 15 Dec 2025 | register lines |
| Aerotrén (Mexico City airport people mover) | running | OSM line |
| José Cuervo Express, Tequila Express (Guadalajara - Tequila / Amatitán) | Saturdays only, day-tour excursion | not counted (Anita: more often than about weekly); named trains if OSM had them as lines (it has a route with no stops, so nothing builds) |
| Tren Ligero de Campeche | OSM has its stops but no track under them | not built |
| AIFA - Pachuca | under construction, opening 2027 | not built |
| Mexico City - Querétaro, Querétaro - Irapuato, Saltillo - Nuevo Laredo (Tren del Norte) | under construction (2027 and later) | not built |
| Tren Turístico Puebla - Cholula | closed 2021 | not built (its OSM route has stops but no line comes of it) |
| Line 12's extension Mixcoac - Observatorio | not open | not built |

**No scheduled passenger train crosses to the United States or Guatemala today**, so there is
no border join and no `borders.EXTRA` point. Línea K reaches Ciudad Hidalgo on the Guatemalan
border, but its passenger trains (21 Nov - 28 Dec 2025) ran Ixtepec - Tonalá only, and that is
all the register keeps of it (`EXTENT`). The Geofabrik extract carries a few US pieces round El
Paso, Nogales, Laredo and Calexico (UP and BNSF track, Amtrak's Sunset Limited and Texas Eagle
routes); none is register track, and Amtrak's routes are named trains if they build at all.

## Sources

- **OSM** (Geofabrik `mexico-latest`, extracted into `data/proc/mx` 2026-10-03; ODbL).
- **Wikidata** (`python mx_register.py --fetch`, `data/raw/mx/wikidata_stations.json`, CC0): 471
  rows of Mexican station items with coordinates, line (P81), state of use (P5817), closure
  (P3999). Used only for railway stations and halts (Q55488, Q55678) in use or with no state,
  within 600 m of a register line's track, where OSM has no station record within 300 m and none
  of the same or a near-identical name (difflib ratio 0.9) within 25 km. The 25 km is because
  Wikidata puts several Tren Maya stations on their town (Xpujil 5.6 km, Nicolás Bravo 12 km
  from the station). Brackets are dropped from Wikidata names ("Candelaria (Tren Maya)"). 15
  taken: Chihuahua, Creel, Divisadero (Chepe); Cueyamil, La Loma, Teyahualco, Prados Sur, Cajiga,
  Xaltocan (AIFA branch); El Espinal, Reforma, Chahuites, Arriaga, Tonalá (Línea K); Pakal
  Ná-Palenque (Línea FA).
- **Published lengths** for the checks: Wikipedia (en and es), the SICT's own figure for the
  AIFA branch, Diario del Istmo for Línea FA (each in `check_model.REGISTER["mx"]`).

Tried and not used:

- **FRA NARN** (the US and Canadian register) covers Mexico: `COUNTRY='MX'`, 2,532 segments,
  20,158 km, owners FXE, CPKC, FSRR, FCCM, LFCD, CHP, TFVM... But no Mexican segment has a
  passenger code (`PASSNGR` is null on all 20,158 km), segments average 8 km (too coarse to
  draw), and there is no Tren Maya, El Insurgente or Suburbano in it. Queried 2026-10-03.
- **ARTF's Red Ferroviaria Nacional** (datos.gob.mx, CC BY 4.0, updated 10 Feb 2026: track,
  nodes, stations "origen-destino", structures, km posts; SHP). The CKAN metadata answers
  (`https://www.datos.gob.mx/api/3/action/package_show?id=red_ferroviaria_nacional`), but the
  files on `repodatos.atdt.gob.mx` answer 403 (Akamai "Access Denied") to a script; not retried.
  The track file is `https://repodatos.atdt.gob.mx/api_update/artf/red_ferroviaria_nacional/01_via_ferrea.zip`,
  stations `.../03_origen_destino.zip`, km posts `.../05_placa_kilometrica.zip`. With them the
  Chepe's missing stations (El Fuerte, La Junta, Témoris...) and real chainage could come in.
- **INEGI's** network is the same ARTF layer (Información de Interés Nacional).

## The line calls

- **Register lines are the legal track names**, as in Korea and the UK; services are OSM routes
  over them. Display names: the track's name where it is a name ("Tren Maya", "El Insurgente");
  "Tren Suburbano" for Ramal 1 plus the AIFA branch; "Chihuahua al Pacífico" (OSM's name for the
  Chepe route; the track is Línea Q, which runs on to Topolobampo and Ojinaga with freight only);
  "Ferrocarril del Istmo de Tehuantepec (Línea Z / FA / K)" for the Interoceánico's three.
- **Tren Maya is one line** (1,494 km built, 34 stations), not seven: the tramos are
  construction lots, and the trains (Cancún - Palenque, Mérida - Cancún, Cancún - Chetumal...)
  run across them. Its junctions are triangles at Escárcega, Cancún airport and Chetumal
  airport; the station on each sits on its own stub, so kr_register's search, which knows no
  direction, also paired the stations either side past it (Candelaria - Centenario 111 km,
  Leona Vicario - Puerto Morelos 54.5, Nicolás Bravo - Bacalar 65.7). `drop_junction_runs`
  removes a section whose track two other sections meeting at a third station cover (90%
  within 30 m). The stub to each station is then in two sections, so the line counts it twice
  (about 20 km over the three).
- **The AIFA branch is part of Tren Suburbano**, not a line of its own: its trains run from
  Buenavista, and its track leaves Ramal 1 500 m north of Lechería with no station there, so on
  its own it would begin at Cueyamil and lose the Lechería end. (Banobras runs it, not
  Ferrocarriles Suburbanos; the line's operator shows the latter.) The same junction rule took
  out Cueyamil - Tultitlán (7.0 km), a run past the Lechería junction.
- **El Chepe**: the Regional runs Chihuahua - Los Mochis several days a week and is the line's
  only all-stops service, so the line is a line (Anita's rule counts anything more often than
  about weekly); the Express is a named train over part of it (`rules/mx.py`). The Regional
  stops on request at many places, so every named railway=station on the track is taken as a
  stop (19 stations).
- **The Interoceánico's three lines are suspended**, not left out: Anita keeps closed lines
  greyed for riders who went before. OSM's route for Línea Z ("Ferrocarril del Istmo de
  Tehuantepec") would stay a running line beside the greyed one, so `rules/mx.py` makes it a
  named train as a stopgap (counts towards nothing); a way to grey an OSM line would be better.
- **Metros and light rail are register lines too** (unlike the US and UK), because OSM's own
  route relations are broken in two of the three cities: Guadalajara's lines 1-3 give their
  ways the role "route", which `build_model.assemble` reads as no track, so they built nothing;
  Metrorrey's lines 1-3 and Mexico City's Línea 4 list one direction's stops out of order, which
  gives a section end to end (Exposición - Talleres 18.7 km on a 18.6 km line). Their stations
  come from those same route relations (`route_lists`, gb_register's rule: a stop goes on the
  named line its route's own track runs on within 250 m), which reads way members whatever
  their role. Names: Mexico City's as OSM's route_masters ("Línea 1"), Monterrey's "Metrorrey
  Línea 1" (operator "Metrorrey", which norm_line_name strips, so OSM's Metrorrey "Línea 1" is
  matched to it and hands over its colour, not to Mexico City's), Guadalajara's "Mi Tren Línea
  1" as its route_masters. Mexico City's register operator is spelt "Sistema de Transporte
  Colectivo (Metro)" so that `same_operator` does not match it to Metrorrey's "Sistema de
  Transporte Colectivo Metrorrey".
- **Guadalajara's colours** are OSM's route colours, set in `LINE_INFO` (OSM's own lines 1-3
  never build to hand them over); Line 4's "orange" is written #f28c28.
- Mi Tren Línea 4 has 9 stations against the 8 opened: Acueducto (between Las Juntas and
  Jalisco 200 Años) is a SITEUR stop node on the track that the line's routes do not list;
  kept (a stop node on a line's own track is a station in kr_register), and may be an infill
  stop newer than the routes.

## Checks

`python check_model.py --region mx` (2026-10-03):

| line | built | published | ratio |
|---|---|---|---|
| Tren Maya | 1,494.0 | 1,554 (WP) | 0.96 |
| El Insurgente | 58.0 | 57.7 (es.WP) | 1.00 |
| Tren Suburbano | 48.9 | 50.7 (27 es.WP + 23.7 SICT) | 0.96 |
| Chihuahua al Pacífico | 652.6 | 668 (WP) | 0.98 |
| Línea Z (suspended) | 303.2 | 308 (WP) | 0.98 |
| Línea FA (suspended) | 327.7 | 329 (Diario del Istmo) | 1.00 |
| Mexico City Metro, 12 lines | | WP revenue lengths | 0.99 - 1.01 each |
| Metrorrey 1, 2, 3 | 18.6, 12.2, 7.3 | 18.8, 13.7, 7.5 | 0.99, 0.89, 0.97 |
| Mi Tren 1, 2, 3, 4 | 15.9, 8.6, 19.6, 20.1 | 16.5, 8.7, 21.5, 21.2 | 0.96, 0.98, 0.91, 0.95 |

- Tren Maya by tramo: I Palenque - Escárcega 226.0 / 228; II Escárcega - Calkiní 240.9 / 235;
  III Calkiní - Izamal 142.5 / 172 (the one far off; the track round Mérida is the line's, so
  either the published figure counts another alignment or OSM is short there); IV Izamal -
  Cancún 246.2 / 257; V Cancún - Tulum 112.6 / 121; VI Tulum - Chetumal 252.5 / 254; VII
  Chetumal - Escárcega 273.4 / 287.
- Metrorrey 2 (0.89) and Mi Tren 3 (0.91) are complete chains of all their stations (13 and
  18); the published figures likely include tail track past the end stations (OSM's own route
  relations measure 12.3-13.6 and 19.9).
- The Chepe has 19 stations where it should have about 30: OSM and Wikidata have no record of El
  Fuerte, La Junta, Témoris' station and others, so Los Mochis - El Descanso is one 161 km
  section and Chihuahua - Cuauhtémoc 133 km. ARTF's station layer would fill them.

## Commands

    python mx_register.py --fetch          # Wikidata stations, one SPARQL query (rate-limited: retries)
    python mx_register.py --names          # track names and km in the extract
    python build_model.py --region mx --register mx_register:data/raw/mx     # ~30 s
    python build_tiles.py --region mx                                        # ~15 s, 2.0 MB
    python check_model.py --region mx

## Open

- ARTF's files (403 to a script): the Chepe's missing stations and per-line chainage.
- OSM data errors worth fixing at source: Guadalajara's route roles ("route", "south"),
  Metrorrey's and Mexico City Line 4's stop order, the Chepe route's missing stops.
- A way to grey an OSM line (the Línea Z route), and the twin rule `TWIN_ON_STATIONS` proposed
  in `rules/mx.py` (the four broken OSM metro routes stay as "(as operated)" copies until then).
- Watch for: Metrorrey 4 and 6, the Interoceánico's return (set `suspended` off in
  `LINE_INFO`), AIFA - Pachuca and the 2027 lines, Campeche's light rail track in OSM.
