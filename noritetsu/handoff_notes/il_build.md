# Israel build (il), 2026-10-08: shared-file changes for the managing session

`il` is built in `dist/data/il/` and `dist/data/il.pmtiles` (model, tiles, check_model,
line_100_probe all passing). Nothing below has been applied; the il build does not need it
except for `tools/rebuild.py` batch rebuilds and the app finding il by panning.

## 1. tools/rebuild.py: REGISTER entry

```diff
     **{cc: f"asia_register:data/raw/rinf/{cc}" for cc in ("kh", "la", "ph", "mm", "mn", "np")},
+    # Israel: MOT GTFS stop lists laid on OSM track (il_register.py, my_register's recipe).
+    # After an extract (`--station-areas`): `il_register.py --clip` (drops OSM's stale IR
+    # train routes and the unopened Nofit light rail).
+    "il": "il_register:data/raw/il",
 }
```

and optionally `"il": 1` in `MINUTES` (build_model 35 s, build_tiles 2 s).

(Gated: a new key only, so no other country's build changes; no ab.py run needed.)

## 2. tools/build_regions.py

Run it to put il in `regions.json`. No neighbour needs a rebuild: no passenger train crosses
Israel's borders (Jordan, Egypt, Lebanon, Syria), so there are no border points.

## 3. Outline: no change needed (checked)

Anita, 2026-10-08: "if israel administers it we can draw it under israel". religiondots'
`country_shapes.geojson` already puts East Jerusalem (all of the Jerusalem light rail's Red
Line, Shuafat, Beit Hanina, Pisgat Ze'ev, Neve Yaakov) inside `il`. Three register lines have
track inside its `ps` shape:

| line | km inside `ps` |
|---|---|
| Tel Aviv - Jerusalem Railway (A1), near Mevo Horon / Canada Park | 6.6 |
| Eastern Railway, along the 1949 line near Tayibe and Kokhav Yair | 7.2 |
| Beit Shemesh - Jerusalem Malha (greyed), where the 1949 line follows the railway at Battir | 2.2 |

All three are register lines owning their ways, and ownership.py's abroad step (section 5)
only gives the neighbour ways *no register line owns*, so they stay il after build_regions
too. `ps` is not a built region and has no rail, so `split_at_borders` never runs there. No
`borders.OUTLINE["il"]` or `EXTRA_AREAS` entry is needed. If `ps` were ever built (it has no
passenger rail), this would need revisiting.

## 4. HANDOFF.md table row (suggested)

```
| il | MOT GTFS stop lists laid on OSM track (`il_register.py`; extract with `--station-areas`, then `--clip`) | 22 | 697 | `il_sources.md` |
```

## Files the agent wrote (for the record)

New: `il_register.py`, `rules/il.py` (no settings, a note), `colours/il.csv`,
`data/raw/il/gtfs/il_rail.gtfs.zip` (the 0.56 MB rail subset of the 145 MB feed, which is
deleted), `data/raw/il/malha_disused.json` (Overpass: the disused Beit Shemesh - Malha
track). Edited: `check_model.py` (REGISTER["il"]), `il_sources.md` ("Build (2026-10-08)").
`il_register` imports `Track` and `Polyline` from `my_register.py` and `dist_m` from
`kr_register.py`; a change to those signatures would break it. The extract
`data/raw/israel-and-palestine-latest.osm.pbf` is deleted (data/proc/il holds what a rebuild
needs).
