# Mauritius (mu): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs

Metro Express (Metro Express Ltd, state-owned), light rail, the island's only railway:

| line | status |
|---|---|
| Line 1, Place d'Armes (Port Louis) - Curepipe Central, ~26 km, 20 stops (Place d'Armes, Victoria, St Louis, Coromandel, Barkly, Beau Bassin, Vandermeersch, Rose Hill Central, Belle Rose, Quatre Bornes Central, St Jean, Trianon, Phoenix, Phoenix Mall, Palmerston, Vacoas Central, Sadally, Floréal, Curepipe North, Curepipe Central) | running, every few minutes |
| Line 2, Rose Hill Central - Réduit (Mahatma Gandhi), 3.4 km branch via Ébène | running; a trial of through trains Curepipe - Mahatma Gandhi at rush hour planned for late Aug/Sept 2026 |
| Extension towards Rose-Belle / the airport ("Phase 2B" in OSM) | not open |

Evidence: mapa-metro.com's Metro Express page (2026; 22 stations, 12.1 million riders in
2024-25); OSM's 27 named stations. 12 million riders a year: no doubt it runs.

### Line list

1. **Port Louis - Curepipe** (Place d'Armes - Curepipe Central): ~26 km.
2. **Rose Hill - Réduit**: 3.4 km (Rose Hill Central to Mahatma Gandhi; the branch leaves the main line at Rose Hill).

Expected: 2 register lines, ~30 km, 22 stations.

### Sources

- OSM: 63 km of `railway=light_rail` (both directions), 58 km named "Metro Express"; 27 named stations. The route relation query timed out on public Overpass today; read from the extract.
- GTFS: none found (Mobility Database has no Mauritius feed; Metro Express Ltd publishes timetables on its site/app only).
- Licence: OSM ODbL.
- Geofabrik: `africa/mauritius-latest.osm.pbf`, 9.0 MB.

### Recipe

Small enough for a two-line hand list in the shared eafrica reader (kind `light_rail`), or as sg/hk do, a
small `mu_register.py` from the operator's stop list; named track would also work (92% named).
I would add it as two entries in the shared reader.

### Open questions

- Whether the planned through service Curepipe - Mahatma Gandhi changes anything: no, the lines stay track pieces.

## Build (2026-10-08)

No register: the Metro Express is light rail, which rinf.py does not trace, and OSM's routes
are complete. `python eafrica_register.py --clip mu` (renames the two route masters, whose
names still say "Phoenix" behind a "⟷" build_model does not read as a direction mark), then
`build_model.py --region mu` with no `--register` (tools/rebuild.py: None, as Qatar).
**Result**: 2 OSM lines, 28 km: Port Louis - Curepipe 25.0 (19 stops), Rose Hill - Réduit 3.0
(3 stops). check_model KNOWN: 25.0 of 26 (0.96), 3.0 of 3.4 (0.89, platform to platform).
