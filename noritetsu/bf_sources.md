# Burkina Faso (bf): sources

## Build (2026-10-08)

    python extract.py --region bf --pbf data/raw/burkina-faso-latest.osm.pbf --station-areas
    python wafrica_register.py --clip bf       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill bf
    python wafrica_register.py --convert bf
    python build_model.py --region bf --register wafrica_register:data/raw/rinf/bf
    python build_tiles.py --region bf
    python check_model.py --region bf

Built: **2 register lines, 495 km; 1 running (349 km), 1 greyed (145 km)**.
- Ouagadougou – Bobo-Dioulasso, running: Ouagadougou, Koudougou, Siby, Bobo-Dioulasso (the
  communiqué's stops), listed. Check: 349.2 against 345 (fr.WP: line 1,145 km, Bobo at PK
  800), 1.01.
- Bobo-Dioulasso – Niangoloko, greyed: the press says the weekly train runs on there, Sitarail's
  communiqué only to Bobo. Built so the line is visible; Bobo, Banfora, Niangoloko listed.
OSM's Abidjan – Ouagadougou route and its route_master are dropped by --clip (the through
train has not run since 2020).

## Survey (2026-10-08)

Research only; nothing built. One weekly train, Ouagadougou – Bobo-Dioulasso. Marginal
against the "about weekly" rule; I count it.

### What runs

- **Ouagadougou – Koudougou – Siby – Bobo-Dioulasso** (Sitarail, metre gauge). Passenger
  service resumed **17 Nov 2023** after the 2020 suspension, **one train a week each way**
  (Bobo-Dioulasso Tuesday 09:00, Ouagadougou Thursday 09:00), stops Bobo-Dioulasso, Siby,
  Koudougou, Ouagadougou (Sitarail's communiqué; leconomistedufaso.com/2023/11/16/ouaga-abidjan-communique-de-sitarail/,
  fr.allafrica.com/stories/202311180021.html, africasupplychainmag.com). Sitarail had
  announced the train would reach Niangoloko (the last station before Côte d'Ivoire); the
  press says it runs Ouaga – Niangoloko, the communiqué only to Bobo. Nothing reports a
  suspension since. 2026 shows the line in passenger use: special trains Ouaga – Bobo for
  the Semaine nationale de la culture (23 Apr 2026, ~400 passengers) and the "Train du
  tourisme" (3 July 2026) (wakatsera.com; burkina24.com). No 2026 timetable found.
- Not running: Bobo-Dioulasso – Banfora – Niangoloko – Côte d'Ivoire (no passenger train into
  Côte d'Ivoire since 2020; see `wafrica_survey.md`), Ouagadougou – Kaya (105 km, freight /
  disused). 

Decision: register line **Ouagadougou – Bobo-Dioulasso, running** (~350 km); Bobo – Niangoloko
– border greyed (or left out). Weakest evidence of any country here: if Anita prefers a
higher bar, Burkina Faso drops out.

### Line list with km

Ouagadougou – Bobo-Dioulasso ~ 350 km (the Abidjan – Ouagadougou line is 1,145-1,260 km in
all, 517 km in Burkina; fr.wikipedia "Ligne d'Abidjan à Ouagadougou" has a station diagram
with PK from Abidjan; use its PK differences). Fahrplancenter has Sitarail's old timetables
(fahrplancenter.com/BurkinaFasoEntry.html, Sitarail.html).

### OSM (Overpass, 2026-10-08, bbox 9.4,-5.6,15.1,2.4)

- route=train 8530177 "Abidjan - Ouagadougou" (Sitarail; the pre-2020 through train; its
  stops cover the Burkinabè part), route=railway 7184678 "Chemins de fer Abidjan-Niger".
- Track: 575 ways, **443 named "Chemins de fer Abidjan-Niger"**, metre gauge throughout.
  60 stations, 58 named.
- The best-mapped line in this survey for its size: a trace over named track, stops Siby and
  Koudougou from the communiqué (and the route's others if the train is an omnibus).

### Timetables / GTFS

None. 

### Recipe

One hand line traced by rinf.py in the shared reader (see `ng_sources.md`). Extract:
`africa/burkina-faso-latest.osm.pbf`, **80 MB**.

### Licences

OSM ODbL; facts from the press.

### Open questions

- Whether the weekly train still runs in late 2026 (no 2026 timetable found; only special
  trains reported). Re-check Sitarail's Facebook page or burkina24.com before a
  build.
- Does it run past Bobo to Niangoloko?
