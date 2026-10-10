# Uruguay register sources

Built 2026-10-08 (latam agent). The survey below is the research record.

## Build (2026-10-08)

`latam_register.py` (ar_register's recipe): one register line, "Línea Rivera (Tacuarembó –
Rivera)", 117.8 km, 22 stations (AFE's 8 and its 14 request halts), running (Mondays and
Fridays). OSM's route_master is renamed by the clip so it drops as the twin.

**Check: the request halts are km posts.** OSM maps each as a railway=halt "km 457" ... "km
552", and they are listed as stations, so each section between two posts checks itself:
km 457 to km 552 builds 95.6 km for 95 (1.01); every post-to-post stretch is within 1 km
(457 - 469 12.1 for 12, 508 - 526 17.4 for 18, 539 - 548 9.5 for 9 ...). By the posts
Tacuarembó sits at about km 445 and Rivera station at about km 562 (the border is km 566.6,
FOCEM's track renewal). No whole-line figure is published (gub.uy rounds it to 100 km).

Calls:
- **The request halts are stations**: the train calls at them on request, so a rider can
  board there (the survey's open question).

Commands:

    python extract.py --region uy --pbf data/raw/uruguay-latest.osm.pbf
    python latam_register.py --clip uy
    python build_model.py --region uy --register latam_register:data/raw/uy
    python build_tiles.py --region uy
    python check_model.py --region uy

## Survey (2026-10-08)

### What runs

One passenger train: **AFE's Tacuarembó - Rivera**, one round trip on Mondays and Fridays
(timetable "a partir del 7 de abril del 2025", still the one on AFE's page on 2026-10-08):
Tacuarembó 07:00 - Rivera 09:10, back 16:15 - 18:25.

Stations in order, as AFE abbreviates them: **Tacuarembó, B. de Rocha (Bañado de Rocha),
P. del Cerro (Paso del Cerro), Laureles, B.C. de Rivera ("B.Civiles" in the fare table),
Tranqueras, P. Ataques (Paso Ataques), Rivera** (8 stations), plus
request halts named by km post (PDA. KM. 457, 469, 475, 484, 487, 496, 500, 504, 508, 526, 531,
539, 548, 552). The km posts run from Montevideo along the Línea Rivera, so the posts give the
chainage: the line is roughly km 450 - 566, about 115 km. AFE's table abbreviates names; spell
them from OSM.

Not running:
- **Montevideo suburban trains** (to Progreso / 25 de Agosto / Florida, Sudriers, San José):
  stopped since the Ferrocarril Central rebuild for UPM (2019). The rebuilt line opened to
  freight in 2023-24; passenger service is held up by its signalling (ETCS) and has no date
  (Railway Gazette, "ETCS delays hinder Uruguayan rail revival"; the page answers 403 to a
  script).
- Salto - Concordia (Argentina): promised for years, not running.
- Heritage excursions (AFE's Día del Patrimonio rides in Tacuarembó and Montevideo, Oct 2026):
  not scheduled service.

### Sources

- **AFE's passenger page**, `https://www.afe.com.uy/servicio-de-pasajeros.php`, saved as
  `data/raw/uy/survey/afe_servicio_de_pasajeros_20261008.html`: timetable, fare table, km posts
  of the request halts. No licence stated (facts only are taken).
- **OSM**: see the OSM section below (Overpass summary in `data/raw/uy/survey/osm_summary.txt`).
- Geofabrik `south-america/uruguay-latest.osm.pbf`, 53 MB.

### OSM (Overpass, 2026-10-08; `data/raw/uy/survey/osm_summary.txt`)

- **The train has PTv2 routes both ways**: r9205334 "Tacuarembó-Rivera" and r9209781
  "Rivera-Tacuarembó" (operator AFE, `interval=48:00`, stale but harmless).
- **AFE's legal lines are `route=railway` relations**: Línea Rivera (r1770791), Línea Río
  Branco, Línea Melo, Línea Minas, Línea a Rocha, a Colonia, a Mercedes, Algorta - Fray Bentos,
  a Artigas, Ramal a Bella Unión, Ramal al km 329. The track itself carries no names (1,388 km
  of usage=main, all unnamed). So Iran's recipe (`ir_register.py`, route=railway relations)
  would also give Línea Rivera, but only Tacuarembó - Rivera carries passengers, so the
  passenger route's ways are the simpler source (ar_register's recipe).

### Recipe

ar_register's recipe (the ways of a hand-listed OSM passenger route; `cl_register.py` is the
smallest example): one line, the two AFE routes, stations from the routes' stops, named
"Línea Rivera (Tacuarembó – Rivera)" as Chile names its legal-line stretches. AFE's km posts
give a check. One line does not justify a reader of its own; it fits a shared small-countries
reader (with Panama, Ecuador, the Dominican Republic, Puerto Rico, Costa Rica) built on
ar_register's code, as cl_register is.

Expected: 1 register line, ~115 km, 8 stations (plus any request halts OSM maps).

### Open questions

- Whether to include the km-post request halts as stations (they are stops the train calls at
  on request; HANDOFF's rule takes stops a rider can board at, so probably yes, where OSM has a
  node).
- Watch for: Montevideo - Progreso / Florida on the Ferrocarril Central, which would bring a
  second, much busier line.
