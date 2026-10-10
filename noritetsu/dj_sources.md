# Djibouti (dj): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent. Same railway as Ethiopia's; see
`et_sources.md` for the service and the sources, which are shared.

### What runs

The Addis Ababa - Djibouti train's K1/K2 leg, Dire Dawa - Nagad, every second day each way
(seat61.com/Ethiopia.htm Nov 2025; ethiopiarailway.com July 2026). It calls at Ali Sabieh in
Djibouti. Nothing else: the old metre-gauge line is closed, Doraleh port station is freight only.

### Line list

1. **Addis Ababa - Djibouti railway, Djibouti part**: border (beyond Dewele, km ~667) - Ali Sabieh (690.7) - Holhol (711.3, passing loop, not a stop) - Nagad (743.9). About 77-80 km. Port station (756.1) freight only, not on the line.

Expected: 1 register line, ~80 km, 2 stations.

### OSM quality (Overpass, 2026-10-08)

123 km of track, 92 named ("Chemin de fer d'Addis-Abeba - Djibouti" in several spellings);
the train route 6281977 and infra relations 2732604, 10693686 cross into Djibouti. Geofabrik
`africa/djibouti-latest.osm.pbf`, 6.7 MB.

### Recipe

Build with Ethiopia in the same reader; one register line each side of a border point at
the Dewele/Guelile crossing. Too small to be worth a separate check beyond Nagad's chainage
(743.9 - 663.1 = 80.8 km Dewele - Nagad).

## Build (2026-10-08)

Built with Ethiopia's (eafrica_register.py). **Result**: 1 register line, border - Ali Sabieh -
Nagad 76.7 km, running (listed_only: Holhol is a passing loop). check_model 76.7 of 76.8 (en.WP
chainage Dewele 663.1 - Nagad 743.9, less the 4.0 km Dewele - border traced on Ethiopia's side).
The border point is OSM's own boundary crossing (et_sources.md "Build"); eafrica_register's
clip sides the track near it by OSM's boundary, not the app's outline, and puts back the way
that crosses it. The border point shows as "XDJET1" until `eXDJET1` is in borders.EXTRA.
