# Puerto Rico register sources

Built 2026-10-08 (latam agent), as its own region `pr` (option 1 below). The survey below is
the research record.

## Build (2026-10-08)

OSM lines only (`build_model.py --region pr`, no register, as Qatar): Tren Urbano, 16.5 km,
16 stations (en.WP 17.2 with the tails: 0.96), running. `latam_register.py --clip pr` is a
no-op kept for the uniform recipe (no neighbours). Needs `build_regions.py` to appear in
regions.json (outline: religiondots' `pr` shape).

Commands:

    python extract.py --region pr --pbf data/raw/puerto-rico-latest.osm.pbf
    python latam_register.py --clip pr
    python build_model.py --region pr
    python build_tiles.py --region pr
    python check_model.py --region pr

## Survey (2026-10-08)

### Is it in the US build?

No. `dist/regions.json`'s `us` bbox stops at 18.906 N (Hawaii's south point), and Puerto Rico
lies at 17.9 - 18.5 N; nothing in `dist/data` names Tren Urbano or its stations. The US build
reads Geofabrik's `us-latest.osm.pbf`, and Geofabrik ships Puerto Rico as its own extract
(`north-america/us/puerto-rico-latest.osm.pbf`, 70 MB; the US Virgin Islands 3.0 MB, no rail).
FRA's NARN, the US register, has no Puerto Rico track either (us_sources.md lists none).

### What runs

**Tren Urbano** (San Juan), one heavy-rail metro line, 17.2 km, 16 stations, daily: Sagrado
Corazón, Hato Rey, Roosevelt, Domenech, Piñero, Universidad, Río Piedras, Cupey, Centro Médico,
San Francisco, Las Lomas, Martínez Nadal, Torrimar, Jardines, Deportivo, Bayamón. Operator
Alternate Concepts (ACI) for the Department of Transportation and Public Works (DTOP / ATI). No
extension planned; weekday ridership about 20,900 in Q2 2026 (en.wikipedia, Tren Urbano).
Nothing else on the island carries passengers (the sugar railways are gone; the Arecibo tourist
line closed long ago).

### Sources

- **OSM**: Overpass summary in `data/raw/pr/survey/osm_summary.txt` (below).
- No GTFS for Tren Urbano in the Mobility Database (none for PR at all, 2026-10-08).
- Published length 17.2 km (en.wikipedia), for `check_model.REGISTER`.

### OSM (Overpass, 2026-10-08; `data/raw/pr/survey/osm_summary.txt`)

- Two PTv2 subway routes, Bayamón => Sagrado Corazón and back (ref "Tren Urbano", colour
  purple, no operator, no route_master). Track "Línea del Tren Urbano", 33.9 km of ways (both
  tracks of a 17 km line). Everything else is abandoned railway (98 km) and a 0.3 km funicular
  way with no route.

### Recipe

The US and UK builds leave metros as OSM's route relations, and that is all Puerto Rico has:
build it as **OSM lines only** (no register reader), or, so it counts as a register line, a
one-line hand list through rinf.py. Two ways to ship it:

1. **Its own region `pr`** with Geofabrik's Puerto Rico extract (70 MB) and its own outline
   (religiondots' `country_shapes.geojson` has PR as a separate shape, as Natural Earth does).
   Simplest; matches how the app shows territories elsewhere (Abkhazia `xa` is its own region).
2. Fold it into `us`: a second extract merged into us's, and the us outline extended. Touches
   the managing session's shared files; not worth it for one line.

Recommendation: option 1, OSM lines only, plus `REGISTER["pr"]` with the 17.2 km.

Expected: 1 line, 17.2 km, 16 stations.

### Open questions

- None of substance; check after the extract that OSM's Tren Urbano route_master has both
  directions with all 16 stops.
