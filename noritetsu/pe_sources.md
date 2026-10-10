# Peru register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe). OSM's train routes give the track but list almost
no stops, so each register line lists its stations (`lists`, the operators' calls).

| line | built km | stations | published | ratio | runs |
|---|---|---|---|---|---|
| Ferrocarril Sur Oriente: Cusco – Hidroeléctrica (with the Urubamba - Pachar branch) | 130.8 | 10 | - | - | yes |
| Ferrocarril del Sur: Cusco – Puno (Wanchaq - Puno) | 385.2 | 5 | ~384 (survey) | 1.00 | yes |
| Ferrocarril Huancayo – Huancavelica (Chilca – Cuenca) | 55.7 | 8 | 57 (ProActivo) | 0.98 | yes |
| Ferrocarril Huancayo – Huancavelica (Cuenca – Huancavelica) | 70.8 | 6 | 128 - 57 | 1.00 | greyed |
| Línea 1, Lima (OSM) | 33.1 | 26 | 34.6 | 0.96 | yes |
| Línea 2, Lima, stage 1A (OSM) | 4.1 | 5 | 5 | 0.83 | yes |

4 register lines, 642 km (571 running, 71 greyed), plus Lima's two metro lines 37 km.
Cusco San Pedro - Machu Picchu Pueblo sums to 107.7 km over its sections (Poroy 15.4,
Huarocondo 23.3, Pachar 19.7, Ollantaytambo 6.2, Piscacucho 14.6, Km 104 22.0, Machu Picchu
6.5); the 112-113 km usually quoted takes in the switchbacks out of Cusco, which a shortest
path cuts through.

Calls:
- **Cusco - Machu Picchu - Hidroeléctrica is one line with the Urubamba branch**: PeruRail's
  and Inca Rail's trains; the Sacred Valley train (Urubamba - Pachar - Machu Picchu); San
  Pedro - Poroy because PeruRail's 2026 Cusco - Ollantaytambo train starts at San Pedro;
  Machu Picchu - Hidroeléctrica because PeruRail sells it (OSM's routes stop at Machu Picchu,
  so both ends are extents). Stations: San Pedro, Poroy, Huarocondo, Pachar, Urubamba,
  Ollantaytambo, Piscacucho (Km 82), Km 104, Machu Picchu Pueblo, Hidroeléctrica. The local
  trains' other km halts are left out (residents only).
- **Cusco - Puno**: PeruRail Titicaca, three a week each way: counts. Stations as OSM's route
  has them (Wanchaq, Santa Rosa, Pucará, Juliaca, Puno).
- **Huancayo - Huancavelica is two lines**: the Tren Macho runs Chilca - Cuenca (Mondays and
  Fridays), Cuenca - Huancavelica is closed for rebuilding and greyed (a whole-line
  `suspended` flag is the only greying a register line has). Cuenca is an unnamed station
  record at -75.03627, -12.42603 (`extra_stations`). The timetable's Paccha, Parco and
  Pilchaca have no OSM record and are left out.
- **Tacna - Arica is not built**: no train in 2026, and Peru's side has one station (Tacna),
  where a line needs two. Its route is clipped, the track drawn as track. When it returns: a
  line Tacna - border point with the point in borders.EXTRA (Chile meanwhile has it as a
  named train, rules/cl.py).
- **Not counted**: Juliaca - Arequipa (only Belmond's Andean Explorer cruise train), the
  Ferrocarril Central Andino (a few excursions a year; route clipped), Southern Copper's
  freight line (route clipped).
- Lima Línea 2: stage 1A only; OSM's routes still end at Mercado Santa Anita and no source
  shows stage 1B open (contract ties it to the Línea 1 interchange; first-half 2026 promise
  not met as far as found).

Commands:

    python extract.py --region pe --pbf data/raw/peru-latest.osm.pbf       # ~40 s
    python latam_register.py --clip pe
    python build_model.py --region pe --register latam_register:data/raw/pe   # ~35 s
    python build_tiles.py --region pe
    python check_model.py --region pe

## Survey (2026-10-08)

### What runs

| service | status | in the build |
|---|---|---|
| **Lima Metro Línea 1**, Villa El Salvador - Bayóvar, 34.6 km, 26 stations | running | line |
| **Lima Metro Línea 2**, stage 1A Evitamiento - Mercado Santa Anita, 5 km, 5 stations | running since 21 Dec 2023; stage 1B's three Ate stations (Vista Alegre, Prolongación Javier Prado, Municipalidad de Ate) were promised for the first half of 2026, no opening confirmed in the sources found | line, 5 stations (add 1B when OSM / news show it open) |
| **Ferrocarril Sur Oriente** (FCTSO, 914 mm), Cusco (San Pedro; Poroy for most trains; Wanchaq?) - Ollantaytambo - Machu Picchu (Aguas Calientes) - Hidroeléctrica, ~122 km | PeruRail and Inca Rail, many trains daily Ollantaytambo - Machu Picchu; Poroy - Machu Picchu daily outside the rainy season (bus-plus-train via Pachar for some, from 1 Sep 2025); PeruRail opened a direct Cusco - Ollantaytambo train in 2026; local trains for residents to Hidroeléctrica | register line Cusco - Hidroeléctrica |
| **Ferrocarril del Sur** (standard gauge), Cusco (Wanchaq) - Juliaca - Puno, ~384 km: PeruRail Titicaca | three a week each way (Cusco - Puno Wed, Fri, Sun 07:10/07:50, ~10.5 h; Puno - Cusco Mon, Thu, Sat 07:30; no passenger stop on the way, a photo stop at La Raya) | register line Cusco - Puno, two stations |
| Juliaca - Arequipa (part of Ferrocarril del Sur) | only Belmond's Andean Explorer, a two-night cruise train (Cusco - Puno - Arequipa) | not counted (a cruise excursion like Argentina's Tren a las Nubes); its track is no register line |
| **Huancayo - Huancavelica** ("Tren Macho", standard gauge since 2010, 128 km) | Chilca (Huancayo) - Cuenca, 57 km, Mondays and Fridays, free (MTC), since 20 Dec 2024; Cuenca - Huancavelica closed for rebuilding (concession for the whole line awarded Aug 2024) | register line Chilca - Huancavelica: Chilca - Cuenca running, Cuenca - Huancavelica `suspended` (greyed) |
| **Ferrocarril Central Andino**, Lima (Desamparados) - La Oroya - Huancayo, 346 km | excursions a few times a year only | not counted |
| Tacna - Arica (FCTA, 60 km) | no passenger train in 2026 (track rebuilt; cl_sources.md) | Peru's 36 km built greyed, or left out (decide with cl, which has it as a stopgap named train) |

### Sources

- Lima Metro: ATU and Línea 1 (lineauno.pe), en.wikipedia "Lima Metro" (lengths, station
  lists); Línea 2 news: La República (28 Sep 2024), peru-retail.com, energiminas.com (2024 -
  2026).
- PeruRail (perurail.com: routes, the 2026 Cusco - Ollantaytambo train, the Pachar bimodal
  change), Inca Rail (incarail.com).
- Tren Macho: ProActivo (20 Dec 2024, phase one Chilca - Cuenca), Energiminas and La República
  (timetable: Chilca 06:30, Viques, Paccha, Chanca, Retama, Ingahuasi, Huarisca, Parco, Manuel
  Tellería, Pilchaca, Cuenca 08:30).
- en.wikipedia "Rail transport in Peru" (FCCA 346 km, Cusco - Aguas Calientes 113 km, Huancayo
  - Huancavelica 148 km old metre-gauge length).
- Mobility Database: Peruvian feeds are buses (Aeroexpreso Cusco, Trujillo); no rail feed.
- **OSM**: Overpass summary in `data/raw/pe/survey/osm_summary.txt` (below).
- Geofabrik `south-america/peru-latest.osm.pbf`, 244 MB.

### OSM (Overpass, 2026-10-08; `data/raw/pe/survey/osm_summary.txt`)

- Lima: Línea 1 routes both ways (#228B22, operator AATE), Línea 2 routes both ways
  **Evitamiento - Mercado Santa Anita only** (#FFC300): OSM agrees 1B is not open. The rest of
  Línea 2 (31.5 + 12.5 km) and Línea 4 (14.1 km) are railway=construction.
- Train routes (PTv2, one each, no direction pairs): "Cusco - Machu Picchu" r5646859 and
  "Urubamba - Machu Picchu" r8275132 (PeruRail), "Cusco-Puno" r8266111 (PeruRail), "Ruta del
  Tren Macho" r3986378 (not PTv2), "Huancayo - Lima" r8277101 (FCCA excursions: drop),
  "Tacna - Arica" r8279775 (FCTA, not running: drop or grey), "Ferrocarril Southern Perú"
  r15537427 (Southern Copper's freight line: drop). No route reaches Hidroeléctrica.
- Track: the Cusco - Machu Picchu line is named "Ferrocarril Santa Ana" (89.3 km
  narrow_gauge usage=tourism + 47.1 km more narrow_gauge); "Ferrocarril Tacna - Arica" 52.2 km;
  the Ferrocarril del Sur and Central are mostly unnamed (1,133 km of unnamed usage=main).
  `route=railway` relations exist for Ferrocarril Central del Perú, Santa Ana, Tacna - Arica,
  La Oroya - Huancayo.

### Recipe

ar_register's recipe (the ways of hand-listed OSM passenger routes) for three lines:
Cusco - Machu Picchu (r5646859 + r8275132, plus an `extent` over "Ferrocarril Santa Ana" track
from Machu Picchu to Hidroeléctrica, since no route reaches it), Cusco - Puno (r8266111), and
the Huancayo - Huancavelica line (r3986378, its Cuenca - Huancavelica stretch `suspended`).
Stations from the routes' stops, checked against PeruRail's and the Tren Macho's stop lists.
Lima's two metro lines as OSM route relations. `--clip` drops Huancayo - Lima, Southern Perú
and (unless built greyed) Tacna - Arica.

Expected: 3 register lines, ~630 km (122 + 384 + 128), plus Lima's 2 metro lines (~40 km);
the Huancavelica line 71 km of it greyed.

### Open questions

- Lima Línea 2 stage 1B: no source says it opened, and OSM's routes still end at Mercado
  Santa Anita; check again at build time.
- PeruRail Titicaca's 2026 days: the pattern above is from tour sellers (busbud, aracari);
  PeruRail's own "Rail Operations Update" (5 Aug 2026) says timetables were revised "following
  authorization from the railway concessionaire" without listing them here. Weekly at least
  either way.
- Which Cusco station each service uses (San Pedro, Poroy, Wanchaq): the register line should
  start at San Pedro if any scheduled train leaves from there (the 2026 Cusco - Ollantaytambo
  train does, per PeruRail's notice).
