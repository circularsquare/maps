# Argentina register sources (built 2026-10-03)

What `ar_register.py` reads, what runs and what does not, the line calls and why, and how the
build checks out. Downloads are in `data/raw/ar/` (gitignored); this file is the record.
Nothing here needed a login, a key or an account.

## The short answer

- **Lines: the track of OSM's passenger route relations, grouped into lines by hand**
  (`LINES` in ar_register.py), read by kr_register's recipe the way mx_register reads
  Mexico's named track. OSM names 99.6% of Argentina's main-line track, but for its *network*
  ("FC Roca" is 5,137 km, from Constitución to Bariloche and Bahía Blanca, nearly all freight
  or closed), so named track alone cannot cut lines. Every passenger service does have a
  route relation, so a register line is the union of its routes' ways, each way going to the
  first line in `LINES` that lists a route over it.
- **23 register lines, 3,218 km** (2,871 km running; the Chaco's three lines, 347 km,
  suspended and greyed). 51 lines in all with OSM's, 574 stations.
- **Stations**: only those some OSM route stops at. Argentina maps hundreds of closed
  stations as railway=station on or beside track still in use (the Mar del Plata line passes
  dozens between its 11 stops), so a station record alone never makes a station.
- **Checks**: 13 lines against published lengths (0.97 - 1.05), and every section with a
  ministry km post at both ends (239 sections, median 0.997).

## What runs (checked 2026-10-03) and what this build does with it

| service | status | in the build |
|---|---|---|
| Línea Mitre: Retiro - Tigre, - Bartolomé Mitre, - José León Suárez; Villa Ballester - Zárate; Victoria - Los Cardales | running. The "Victoria - Capilla del Señor" branch's trains turn at Los Cardales (Trenes Argentinos, December 2023; OSM's route agrees) | register line "Línea Mitre" |
| Línea Sarmiento: Once - Moreno, Moreno - Mercedes, Merlo - Lobos | running | register line "Línea Sarmiento" |
| Línea Roca: Constitución - La Plata, - Ezeiza - Cañuelas, - Glew - Alejandro Korn, - Bosques (both ways round), Temperley - Haedo, Bosques - Gutiérrez, Korn - Chascomús, Cañuelas - Monte, Cañuelas - Lobos, La Plata - Policlínico (Tren Universitario) | running | register line "Línea Roca". Cañuelas - Lobos has no OSM route, so it is the FC Roca track between the two (`extent`), with Uribelarrea, and shares its last 3.7 km into Lobos with the Sarmiento |
| Línea San Martín: Retiro - Dr. Cabred | running | register line |
| Línea Belgrano Sur: Sáenz - González Catán - Marcos Paz - Villars - Lozano; Km 12 (Tapiales) - Marinos del Crucero General Belgrano | running (Lozano and Villars a few trains a day, 2026 timetable) | register line |
| Línea Belgrano Norte (Ferrovías): Retiro - Villa Rosa | running | register line |
| Línea Urquiza (Metrovías / Emova): Federico Lacroze - General Lemos | running | register line |
| Tren de la Costa: Avenida Maipú - Delta | running | register line |
| Constitución - Mar del Plata | six days a week (Tuesday's 303 dropped for works, El Cronista, 29 Aug 2026) | named train; its track past Chascomús is register line "Ferrocarril Roca: Chascomús – Mar del Plata" |
| Retiro - Rosario Norte | daily (El Cronista, Aug 2026) | named train; track past Zárate: "Ferrocarril Mitre: Zárate – Rosario" |
| Retiro - Junín | daily; new stops Manzanares, Cabred, Cortínez from 22 June 2026 | named train; track past Cabred: "Ferrocarril San Martín: Cabred – Junín" |
| Once - Bragado | Monday, Wednesday, Friday | named train; track past Mercedes: "Ferrocarril Sarmiento: Mercedes – Bragado" |
| Tren Patagónico, Viedma - Bariloche (Tren Patagónico S.A., Río Negro) | weekly (Friday out, Sunday back), sold for the winter and summer holiday seasons | named train; track "Ferrocarril Roca: Viedma – Bariloche", drawn as running (seasonal lines are drawn as running) |
| Retiro - Córdoba, Retiro - Tucumán | suspended since the derailment of 20 Sep 2025; reported cancelled outright in January 2026 (El Día, 2 Jan 2026); no date a year on (Diario Castellanos, 28 Sep 2026) | not built (their track past Rosario is no register line) |
| Buenos Aires - Bahía Blanca, - Mendoza, - San Luis, - Pehuajó, General Guido - Pinamar, Rosario - Cañada de Gómez, La Banda - Fernández, Córdoba - Villa María, Paraná's local service, Mercedes - Tomás Jofré | cancelled or suspended (enelsubte.com's list of 16 stopped services, 4 Sep 2026) | not built; OSM has route_masters for Córdoba - Villa María and Paraná with no routes, which build nothing |
| Tren de las Sierras and Tren Metropolitano (Córdoba / Alta Córdoba - La Calera - Cosquín - Valle Hermoso - Capilla del Monte) | running daily | register line "Tren de las Sierras"; OSM's "Tren Metropolitano" and the Sierras' patterns stay OSM lines over it |
| Tren Regional Salta: Salta - Güemes; Salta - Campo Quijano | Güemes running; Campo Quijano interrupted for rolling-stock maintenance, no date | one register line "Tren Regional Salta", running whole (a rolling-stock gap, not a closed line) |
| Tren del Valle, Cipolletti - Neuquén - Plottier | running (~25 a day; one day's stoppage on 1 Sep 2026); passes to Río Negro's Tren Patagónico in December | register line |
| Metrotranvía de Mendoza | running | register line |
| Tren al Desarrollo (Santiago del Estero - La Banda) | running daily (provincial) | register line |
| Tren Solar de la Quebrada (Volcán - Tilcara, Jujuy) | several departures a day (weekends and holidays at least, daily in season) | register line: a scheduled point-to-point service with intermediate stations, which a rider uses as a line |
| Posadas - Encarnación (Paraguay) | 23 a day each way, Monday - Friday | register line "Tren Binacional Posadas Encarnación", Posadas - Encarnación whole (below) |
| Chaco: Puerto Tirol - Resistencia, Cacuí - Los Amores, Sáenz Peña - Chorotis | stopped early September 2026, "temporarily", no date (enelsubte.com) | three register lines, `suspended` (greyed, out of completion) |
| Subte A, B, C, D, E, H; Premetro | running | OSM lines (clean route_masters, as the US and UK builds leave metros) |
| Tren del Fin del Mundo (Ushuaia), Tren Ecológico de la Selva (Iguazú) | daily, several departures / every half hour | OSM lines, not named trains (rules/ar.py) |
| La Trochita, Tren a las Nubes, Expreso Río Negro, Villa Elisa heritage train | excursions | named trains (rules/ar.py) |
| Tranvía Histórico (Buenos Aires, Rosario) | weekends | OSM lines (tram routes are never named trains in build_model) |

**Borders.** No passenger train runs to Chile, Bolivia, Brazil or Uruguay. The one crossing is
Posadas - Encarnación over the San Roque González bridge to Paraguay, which is not built:
the line runs whole to Encarnación (its route's stop node there has no tags, so
`extra_stations` puts "Encarnación" at the track's end, -55.85789, -27.36775). When Paraguay
is built, the section needs a border point in `borders.EXTRA` to be cut.

## Sources

- **OSM** (Geofabrik `argentina-latest`, extracted into `data/proc/ar` 2026-10-03; ODbL).
  `python ar_register.py --clip` then drops what lies in Chile, Bolivia, Paraguay, Brazil and
  Uruguay: 6 ways of the Antofagasta - Salta line at Socompa. religiondots' outlines are
  generalised (they put the Tren Ecológico's Garganta del Diablo end in Brazil), so only
  ground more than ~1.5 km beyond Argentina's outline is clipped (`HOME_BUFFER_DEG`).
- **Ministry of Transport, datos.transporte.gob.ar** (CC BY 4.0), dataset "Estaciones de
  Trenes y Servicios activos a 2022":
  - `estaciones_ffcc_serv_22.json`, 464 stations with line, operator and, for 261, the km
    post ("progresiva") from each line's terminus. Used for the chainage check
    (`python ar_register.py --chainage`). Fetched from
    `https://ide.transporte.gob.ar/geoserver/idera/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=idera:Estacion_ffcc_serv_22.view&maxFeatures=2000&outputFormat=json`
    (the host's certificate chain does not verify on this machine; fetched with curl -k).
  - `red_adifse_22.geojson`, ADIF's network by railway and ramal code (A, C14, 10...),
    "Activo" / "No operativo", with lengths. Not used: "Activo" is open to any train, not to
    passengers (FC Belgrano "Activo" alone is 5,435 km), and the ramales have codes, not
    names. Kept for reference.
- **Trenes Argentinos' Buenos Aires GTFS** (`trenes-gtfs.zip`, Buenos Aires city's open data
  portal, the Transitous source `Trenes-Argentinos-AMBA`): its calendar runs 1 Feb - 30 Apr
  2020, and it has neither the Urquiza nor the Belgrano Norte. Not used, and not proposed for
  gtfs_served: a 2020 window would close everything that has changed since.
- **Published lengths** for the checks: es.wikipedia (Tren Patagónico, Tren del Valle, Tren
  al Desarrollo, Tren Solar de la Quebrada, Metrotranvía de Mendoza, Estación Junín, Estación
  Bragado), the ministry's posts, and the 400 km given for Constitución - Mar del Plata.

## The line calls

- **Each Buenos Aires line is one register line with all its branches**, named as OSM's
  route_master ("Línea Roca"), so the OSM line is its twin and is dropped (TWIN_ON_STATIONS
  in rules/ar.py). Trenes Argentinos and every rider treat the línea as the line and its
  ramales as branches of it.
- **The long-distance trains are named trains** (a single train a day or a few a week over a
  long corridor; HANDOFF's rule). Their track past the end of the commuter services is a
  register line per railway and stretch, "Ferrocarril Roca: Chascomús – Mar del Plata", so
  it counts as the US build's NARN passenger subdivisions do. Where a long-distance train
  runs over another track of a commuter corridor than the commuter trains (Retiro's, Once's
  and Constitución's approaches, Palermo - Villa del Parque on the San Martín), that piece is
  left to no line (`PIECE_KEEP_KM`, 10 km): 0.4 - 6.6 km pieces, logged in the build.
- **The Tren Patagónico's line is drawn as running** although its train runs only in holiday
  seasons, weekly: Anita's rule draws seasonal lines as running.
- **Córdoba and Tucumán are not built**, not greyed: the government called them cancelled in
  January 2026, and Canada's build left out NARN track no train runs on in the same way.
  The Chaco's lines are greyed instead: stopped a month ago and called temporary.
- **Junction runs**: at a junction just past a station, kr's search pairs all three stations
  of the triangle and each section is covered by the other two. mx_register's rule dropped
  the longest, which at Belgrano R took out the real Belgrano R - Drago and kept Drago -
  Coghlan. `drop_runs` tries first the section no OSM route calls at both ends of one after
  the other: 11 sections dropped (Viedma - Valcheta past San Antonio Oeste 288.9 km, Alta
  Córdoba - Córdoba past the Mitre link 6.6, Lisandro de la Torre - 3 de Febrero 10.9...),
  every one with no route calling at both ends.
- **Station names**: a bracketed line after a name is dropped ("Zárate (Mitre)", "Haedo
  (Sarmiento)", "Retiro (San Martín)" show as "Zárate", "Haedo", "Retiro"); Retiro's three
  terminals are one complex, as kr_register merges one name within 500 m. Subte station
  records are left out of the register's stations ("Retiro (E)" had headed Retiro's
  complex). Belgrano Norte's stop positions are named by direction ("Tortuguitas a
  Retiro") and are read without it; "Manuel B. Gonnet" is read as the station record's
  "Manuel Bernardo Gonnet".
- **Colours**: Trenes Argentinos paints every line the same light blue, so the seven Buenos
  Aires lines' colours are picked (`colours/ar.csv`, marked `picked`); Tren de la Costa's and
  the Metrotranvía's are Wikidata's P465.

## Checks

`python check_model.py --region ar` (2026-10-03):

| line | built | published | ratio |
|---|---|---|---|
| Línea Sarmiento | 166.5 | 169.2 (posts: 36.4 + 61.6 + 71.2) | 0.98 |
| Línea San Martín | 72.1 | 72.3 (post) | 1.00 |
| Línea Belgrano Norte | 51.9 | 51.9 (post) | 1.00 |
| Línea Urquiza | 25.5 | 25.6 (post) | 1.00 |
| Tren de la Costa | 15.0 | 15.2 (post) | 0.99 |
| Chascomús – Mar del Plata | 281.7 | 283.5 (400 less Chascomús' post 116.5) | 0.99 |
| Cabred – Junín | 182.5 | 181.7 (es.WP 254 less 72.3) | 1.00 |
| Mercedes – Bragado | 111.0 | 111.0 (es.WP 209 less 98.0) | 1.00 |
| Viedma – Bariloche | 830.7 | 827 (es.WP) | 1.00 |
| Tren del Valle | 21.0 | 21 | 1.00 |
| Tren al Desarrollo | 8.4 | 8 (a round figure) | 1.05 |
| Tren Solar de la Quebrada | 41.6 | 42 | 0.99 |
| Metrotranvía de Mendoza | 16.5 | 17 | 0.97 |
| Subte A, B, C, D, E (OSM lines) | | es.WP | 0.99 - 1.00 |
| Subte H (OSM line) | 8.0 | 8.8 | 0.91 (OSM's routes measure 8.0 too) |

`python ar_register.py --chainage`: 239 sections have a ministry km post within 300 m of
both ends; built against the posts, median 0.997, 182 within 5%. Per line: Roca 62 sections
median 0.995, Mitre 52 / 0.997, Sarmiento 35 / 0.994, Belgrano Sur 21 / 0.999, San Martín
19 / 0.999, Belgrano Norte 18 / 1.004, Urquiza 22 / 0.999, Tren de la Costa 10 / 0.994. The
sections off by more than 5% are short ones where a post sits at the other end of a platform
from the station's node, or pairs whose posts count from different origins (Bosques -
Sourigues, Empalme Lobos).

Not checked against an outside figure: Línea Roca (355.9 km), Línea Mitre (172.7), Línea
Belgrano Sur (90.5), whose branches' posts start from different origins (their sections are
in the chainage check), Zárate – Rosario (221.4; no post at Zárate, and the 303 km often
given for Retiro - Rosario was not found in a source), Tren Regional Salta (86.9), Tren de las
Sierras (115.6; es.WP's 150.8 km counts something else: OSM's own routes measure 76.5 to
Valle Hermoso plus 33.9 on to Capilla del Monte), the binational train (3.0) and the Chaco's
three (suspended).

## Commands

    python extract.py --region ar --pbf data/raw/argentina-latest.osm.pbf     # ~1 min
    python ar_register.py --clip            # after every extract
    python ar_register.py --report          # each line's ways, km, pieces
    python build_model.py --region ar --register ar_register:data/raw/ar      # ~90 s
    python build_tiles.py --region ar                                         # ~15 s, 2.2 MB
    python check_model.py --region ar
    python ar_register.py --chainage

## Open

- Route ids are pinned in `LINES`: a route relation OSM replaces or splits drops out of its
  line with a log line ("route relation ... is not in the extract"); check the build log
  after each extract.
- The Tren de las Sierras' and Tren Metropolitano's patterns, the Roca's Temperley - Haedo
  route and the Tren Universitario stay OSM lines over their register lines (names that are
  not the register line's).
- Watch for: Córdoba and Tucumán's return (NCA's works), the Chaco's three (set `suspended`
  off), Tren del Valle's extension to General Roca (December 2026), the Belgrano Sur's
  extension to Constitución.
- Paraguay: a `borders.EXTRA` point on the San Roque González bridge when it is built.
