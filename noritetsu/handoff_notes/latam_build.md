# Latin America build (cr, pa, cu, pe, bo, ec, do, uy, ve, co, pr), 2026-10-08: shared-file changes for the managing session

All eleven are built in `dist/data/` (model, tiles, check_model run). Nothing below has been
applied; the country builds do not need it except for `tools/rebuild.py` batch rebuilds and
`build_regions.py` to put them on the map.

## 1. tools/rebuild.py: REGISTER entries

do and pr are metros only and build with no register (Qatar's `None`, already supported by
`run_country`).

```diff
     "qa": None,
     "pk": "pk_register:data/raw/pk",
+    # Latin America: ar_register's recipe for eleven countries in one module
+    # (latam_register.py). After an extract: `python latam_register.py --clip <cc>` (all
+    # eleven, do and pr included: it drops stale routes and renames route_masters).
+    # do and pr are metros only: no register.
+    "cr": "latam_register:data/raw/cr", "pa": "latam_register:data/raw/pa",
+    "cu": "latam_register:data/raw/cu", "pe": "latam_register:data/raw/pe",
+    "bo": "latam_register:data/raw/bo", "ec": "latam_register:data/raw/ec",
+    "uy": "latam_register:data/raw/uy", "ve": "latam_register:data/raw/ve",
+    "co": "latam_register:data/raw/co",
+    "do": None, "pr": None,
 }
```

(Gated: new keys only; no other country's build changes, no ab.py run needed.) The
register's `path` is only used for its last part, the country code; `data/raw/<cc>` exists
for all nine.

## 2. tools/build_regions.py

Run it to put the eleven in `regions.json`. Puerto Rico is its own region (`pr`; the `us`
build does not reach it); religiondots' `country_shapes.geojson` has a `pr` outline.

## 3. borders.EXTRA: nothing proposed

No passenger train crosses any of these borders in 2026: Bolivia - Argentina (Villazón / La
Quiaca, Yacuiba / Pocitos: no Argentine passenger train), Bolivia - Chile (Charaña /
Visviri), Bolivia - Brazil (Quijarro / Corumbá), Peru - Bolivia, Peru - Chile (Tacna -
Arica is not running; Peru does not build it, Chile keeps it as a named train), Uruguay -
Brazil (Rivera stops short of the border), Colombia - Venezuela, Costa Rica - Panama.
Neighbours need no rebuild.

## 4. A gap for later, not needed now

A register line can be greyed only whole (`suspended`, not_running.py): Peru's Huancayo -
Huancavelica is split into two register lines for that (Chilca - Cuenca running, Cuenca -
Huancavelica greyed). A per-section `suspended_sections` would let it be one line.

## Files the agent wrote (for the record)

New: `latam_register.py` (imports ar_register as cl_register does; reads ar_register's
`configure`, `build`, `clip`, `report`, `L` and its CFG keys `extra_stations`, `name_alias`,
`keep_routes`, `drop_routes`: a change to those would break these eleven), `rules/{cr,pa,cu,
pe,bo,ec,uy,ve,co}.py`, `colours/cr.csv`. Edited: `check_model.py` (a "Latin America" block
at the top of REGISTER and of KNOWN), `{cr,pa,cu,pe,bo,ec,do,uy,ve,co,pr}_sources.md`
("Build (2026-10-08)"), `latam_survey.md`. The eleven `.osm.pbf` extracts are deleted;
`data/proc/<cc>` holds what a rebuild needs.
