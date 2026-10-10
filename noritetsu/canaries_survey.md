# Canary Islands and Spain's other outlying networks: survey

Research for adding the Tenerife tram to Spain (es). es_sources.md is the es agent's; this file
does not change it.

## Survey (2026-10-08)

### What runs

The only rail service in the Canary Islands is **Tenerife's tram** (Metropolitano de Tenerife,
"Tranvía de Tenerife"), every few minutes, daily:

| line | stops | from GTFS | length (en.wikipedia) |
|---|---|---|---|
| L1 Intercambiador (Santa Cruz) - La Trinidad (La Laguna) | 21 | stops 01-21 | 12.5 km |
| L2 La Cuesta - Tíncer | 6 (La Cuesta, Ingenieros, Hospital Universitario, El Cardonal, San Jerónimo, Tíncer) | stops 22, 23, 14, 13, 24, 26 | 3.6 km |

L1 and L2 share track between Hospital Universitario and El Cardonal (the feed's transfer stops
13 and 14). The planned extensions (L1 to Tenerife Norte airport, L2 to La Gallega) and the
Tren del Norte / Tren del Sur and Tren de Gran Canaria are not built. No other rail: Gran
Canaria, Lanzarote, Fuerteventura, La Palma, La Gomera and El Hierro have none.

**Ceuta and Melilla**: no railway (es_sources.md agrees). **The Balearics** are already in es's
build (SFM, Palma metro, Sóller).

### Sources

| source | gives | licence | where |
|---|---|---|---|
| Metropolitano de Tenerife GTFS, http://metrotenerife.com/transit/google_transit.zip (redirects to https) | 2 routes (route_type 0), 25 stops with coordinates, stop order per line, shapes, frequencies | Mobility Database lists it as mdb-788 **inactive**; the file still downloads (5.4 kB). Its calendar runs 2024-09-08 to 2025-07-15, so stale for timetables, fine for stops and order. Licence not stated in the feed; Tenerife's open-data portal (datos.tenerife.es, Metropolitano de Tenerife's datasets, also on datos.gob.es "Paradas de tranvía en la isla de Tenerife") is the official source for the stops | `data/raw/ic/survey/metrotenerife_google_transit.zip` |
| OSM via Geofabrik `africa/canary-islands-latest.osm.pbf` (57 MB) | the tram's track (117 ways of `railway=tram` and the like; no `railway=rail` anywhere in the islands), four route relations: L1 both ways (20281633, 20281634), L2 both ways (1286854, 20281470), operator MetroTenerife; no route_master; stops are tram stops, not stations | ODbL | `data/raw/ic/survey/osm_route_relations.json` |

### Where it goes: es's build, not its own region

Add it to **es**. The tram is two OSM lines; it needs no register and no rules of its own,
Spain's outline (religiondots' country shape) already includes the islands, and a rider would
look for it under Spain, as the Balearics already are. A separate region would need its own
outline, regions.json entry and colours for 16 km of tram.

How, since extract.py takes one `--pbf`: either merge the two extracts first (pyosmium is
installed: `osmium.MergeInputReader`, or the osmium tool's `osmium merge spain-latest.osm.pbf
canary-islands-latest.osm.pbf -o es-merged.osm.pbf`) and extract es from the merged file, or
give extract.py a repeatable `--pbf` (a shared file: the managing session). The merge is the
smaller change. Nothing in es's RINF register is affected: Adif has no track in the islands.
The OSM lines come out as `light_rail`/`tram` kind with no register line, as Spain's other
trams do.

Check after building: L1 about 12.5 km with 21 stops, L2 about 3.6 km with 6; one station for
each of Hospital Universitario and El Cardonal shared by both lines.

### Open

- The feed's stale calendar means it cannot drive a timetable check; nothing needs one (trams
  have no register sections).
- es's agent owns es_sources.md: a line there pointing at this file once it is built.

## 2026-10-08 (asia agent): proposed, not built into dist

Extracted to `data/proc/ic` (the .pbf is deleted) and trial-built OSM-only into a scratch
folder: L1 12.4 km / 21 stops, L2 3.4 km / 6 stops, both as published. L2's route_master
(16267950) has no tags, so the line comes out nameless until a clip step or OSM names it. Two
ways onto the map, both needing shared-file changes, are in `handoff_notes/asia_build.md` §6:
its own region `ic` (borders.NAME/OUTLINE; `data/raw/ic/outline.geojson` is written) or the
merge into es on es's next rebuild (preferred for the rider).

## 2026-10-08 (es agent): built into es

Folded into es and shipped (es_sources.md "The Canary Islands"): `extract.py --region ic` from
canary-islands-latest, then `python -m rinf_countries.es --canaries` after the es extract (no
extract.py change, no `ic` region). L1 12.4 km / 21 stops, L2 3.4 km / 6 stops, named from
`CANARY_MASTERS` in es.py. check_model KNOWN["es"] carries both. data/raw/ic/outline.geojson
(written for an `ic` region) is no longer needed.
