# Qatar sources

## Survey (2026-10-08)

Research only; nothing built. Region code `qa`.

### What runs

All urban; Qatar has no main-line railway (the Qatar - Saudi - Bahrain links of the GCC
railway are not built).

| system | lines | km (published) | status |
|---|---|---|---|
| Doha Metro (Qatar Rail, run by RKH Qitarat) | Red (Lusail QNB - Al Wakra, with the Hamad International Airport T1 branch), Green (Al Mansoura - Al Riffa), Gold (Ras Bu Abboud - Al Aziziyah) | about 76: Red ~40, Green ~22, Gold ~14 (en.wikipedia "Doha Metro") | all running since 2019 |
| Lusail Tram (Qatar Rail) | Orange (Legtaifiya - Rawdat Lusail loop, extended 7.6 km in 2024), Pink (opened 8 April 2024), Turquoise (opened 6 January 2025, 11 stops, Fox Hills South - Lusail QNB) | about 28 for the planned network | running; the **Purple** line (Al Sa'ad Plaza - Lusail QNB) is still under construction: leave out |
| Education City Tram (Qatar Foundation) | Blue, Yellow (and Green) loops | ~11.5 | running, free, people-mover-like campus tram |
| Msheireb Tram | one 2 km loop downtown | 2 | running (free heritage-styled tram) |

Sources: en.wikipedia "Lusail Tram" and Railway Gazette ("Lusail tram extensions open",
"Turquoise tram route added to Lusail network"); the Purple line's status from the same pages.

### OSM (Overpass, 2026-10-08, `data/raw/qa/survey/osm_routes.json`)

- Route relations: 8 metro (Red both ways plus the airport branch both ways, Green and Gold
  both ways), 5 Lusail light_rail (Orange, Pink x2, Turquoise, and **Purple**, which is not
  open: drop it in `--clip`), 4 tram (Education City Blue, Yellow x2, Msheireb). Every metro
  relation carries `ref`, operator, network and colour (#E2251C, #009530, #F9B428).
- Track: the metro ways are named per line in Arabic ("الخط الأحمر" Red, "الخط الأخضر" Green,
  "الخط الذهبي" Gold): 79 of 81 subway ways named, so the named-track recipe works for the
  metro. Lusail and Education City track is mostly unnamed (10 of 70 light_rail ways named).
- **Data slips**: several metro relations have an Arabic `name` saying the other line
  ("الخط الذهبي" on a Red route, "الخط الأحمر" on Gold and Green routes). Use `ref` and `name:en`,
  never the Arabic route name, to tell the lines apart.

### Timetables

No open GTFS found: not in the Mobility Database catalogue, not in Transitous. Qatar Rail's
site (qr.com.qa) publishes headways only. Every system runs daily at metro frequencies, so
no feed is needed to say what runs.

### Licence

OSM (ODbL) only.

### Geofabrik

`asia/gcc-states-latest.osm.pbf`, 242 MB (Bahrain, Kuwait, Oman, Qatar, Saudi Arabia, UAE
in one file). One extract serves qa, ae and sa; clip each to its outline.

### Recipe

The same as sg and hk (a short operator list) but with no operator file to read: **one shared
Gulf reader** for qa, ae and sa (`gulf_register.py`, proposed), whose register lines are hand
lists of stations in order, traced over OSM track through rinf.py (nafrica_register's
recipe), with the station lists taken from OSM's own route relations (stop members in order,
as my_register does for KLIA Transit). Qatar's part: Red (with the airport branch as its own
section list), Green, Gold, Lusail Orange, Pink, Turquoise; Education City and Msheireb as OSM
lines (or register lines if Anita wants campus trams counted).

Expected: 6 register lines, about 100 km (76 metro + ~25 tram), about 37 + 25 stations.

### Open

- Is the Education City Tram a line or a people mover? It is a campus circulator, free, and
  runs on public streets; I would keep it as an OSM line, as Changi's Skytrain is kept.
- Lusail Orange line's route after the 2024 extension: OSM's relation should be checked
  against Qatar Rail's map before trusting its stop list.

## Build (2026-10-08)

No main line, so **no register**: `build_model.py --region qa` with no `--register` (all OSM
lines). Extract bbox `50.70,24.45,51.70,26.20` from gcc-states, then
`python mideast_register.py --clip qa` (drops the Lusail Purple Line's route, not open).
`tools/rebuild.py` needs a `"qa": None` entry for this (handoff_notes/mideast_build.md).

**9 OSM lines, 110 km, all running**: Doha Metro Red 38.4 km (with the airport branch; en.WP
40: 0.96), Green 22.3 (1.01), Gold 14.0 (1.00), Lusail Tram Orange 12.5, Turquoise 7.7, Pink
7.0, Education City Tram Yellow 3.9 and Blue 2.5, Msheireb Tram 1.8.

**Decisions**
- The metro's wrong Arabic route names do not show: the lines take their names from the
  route_masters, which are right (الخط الأحمر / الأخضر / الذهبي للمترو).
- **The Education City and Msheireb trams count**, against the managing session's call that
  they be drawn but not counted: build_model has no way to keep an OSM line out of completion
  (a tram is never a named train, and Mexico's stopgap of making a route a named train works
  for route=train only). They are ordinary OSM lines, 8.2 km together, until a hook for
  "drawn, not counted" exists.
- The Lusail Orange Line's OSM route (25 stops, Legtaifiya - Al Yasmeen and back) is taken as
  mapped.
