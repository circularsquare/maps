# Long lines from further out (tiles, 2026-10-09)

Anita (2026-10-09, eastern Canada at about z5-6): VIA's Ocean vanished through New Brunswick
when zoomed out, while Québec and Nova Scotia either side showed; "in general for really long
lines they should always show at farther zoom out".

## Cause

`build_tiles.MINZOOM` decides a way's first zoom from its kind and rank, and the rank comes
only from OSM's `usage` tag (`rank_of`). Through New Brunswick the Ocean runs on CN's Newcastle
Subdivision (ca register line `u20029675bf`, 277 km, Charlo - Rogersville) and the south end
of the Mont-Joli Subdivision (`u70c0d75456`). OSM tags 330 km of those ways `usage=branch`
(CN's freight goes Moncton - Edmundston instead), so they are rank 1, and rank-1 rail starts
at z6. Québec - Rivière-du-Loup's Montmagny Subdivision and Moncton - Halifax's Springhill
Subdivision are mostly `usage=main` (rank 0, from z0), so they showed. The ways are in
ways.json (VIA Rail Ocean `m10411280`, named train, 1,454 km; the Newcastle Subdivision),
not marked `n`, not dropped by drop_islands or simplification: only the minzoom kept them out.
The freight main Moncton - Edmundston (`usage=main`, no line, `n = 1`) did show at z5, in
the faint grey.

## The rule (build_tiles.py, `LONG_LINE_MINZOOM`, `promote_long_lines`)

A way's minzoom is also capped by the longest line in the model that runs over it (ways.json:
register lines, OSM lines and named trains), measured on the line's running sections (a
line's `closed` sections left out):

- 500 km or more: from z3 at the latest
- 150 km or more: from z5 at the latest
- otherwise as before (MINZOOM by kind and rank)

Only lowers a minzoom, never raises one. Track with `n = 1` (no line runs over it) is never
promoted. The merged chains below z10 are now merged per zoom over the ways live at that zoom
(`merged_chains` returns {z: chains}; a group whose live ways are unchanged is merged once).
A first version split chains by minzoom instead, and at z6-9 the new breaks showed as
round-cap dots and reshuffled which colour lay on top. The rank (`r`) is unchanged, so a promoted branch still draws at the branch's width
and held-back colour (0.8 width, RANK_MIX 0.867).

New flags for trials: `--out <file>` writes the archive elsewhere (the model is still read
from dist/data/<cc>/), `--no-long-lines` leaves the promotion out (an A/B baseline).

## Trial (z0-9 built both ways into a temp folder from the same model)

Track promoted (km of drawn track that now starts sooner), and what drives it:

| cc | track at z5 before | promoted | z6->z3 / z6->z5 (km) | mostly |
|---|---|---|---|---|
| ca | 26,187 km | 2,314 km | 1,655 / 659 | VIA Montréal - Jonquière/Senneterre, the Ocean, Polar Bear Express, Keewatin Railway, Tshiuetin |
| us | 158,594 km | 469 km | 215 / 251 | short pieces of Amtrak long-distance routes, the Downeaster, Glacier Discovery |
| ru | 123,102 km | 4,460 km | 1,796 / 2,664 | suburban diesel lines 150-340 km, long-distance trains' branch stretches |
| de | 45,161 km | 1,151 km | 188 / 951 | RE 87, RE 40 Freudenstadt - Karlsruhe, RE 6, RE 3, pieces of RINF 2651, 4000 |
| jp | 32,719 km | 1,181 km | 181 / 1,000 | Ban'etsu West Line, Sanriku Railway Rias Line, Yufu, Hachimantai Rapid |
| au | 30,006 km | 314 km | 194 / 119 | Riverina Rail Tour, Inlander, Spirit of Queensland |
| cn | 240,610 km | 3,253 km | 1,775 / 1,478 | Liupanshui - Hongguo, pieces of the Jinghu, Jingguang, Jingha lines |
| in | 97,687 km | 3,711 km | 1,007 / 2,534 (+170 narrow gauge z7->z5) | Rajdhani routes, Kangra Valley, Vijayawada loops, Bina - Katni |

Tile bytes, z0-9 (gzipped, all tiles of the zoom), before -> after:

| cc | z3 | z4 | z5 | z6-9 | z0-9 total |
|---|---|---|---|---|---|
| ca | 5.1 -> 5.4 KB | 7.9 -> 8.5 KB | 12.5 -> 13.9 KB (+11%) | identical | +0.5% |
| us | 34.0 -> 34.1 KB | 47.7 -> 47.9 KB | 67.6 -> 68.5 KB (+1.3%) | z6 +3 bytes, z7-9 identical | +0.1% |
| ru | 31.1 -> 31.5 KB | 50.7 -> 52.9 KB | 82.1 -> 96.3 KB (+17%) | identical | +0.9% |
| de | 15.7 -> 15.8 KB | 22.6 -> 22.7 KB | 39.2 -> 43.0 KB (+9.5%) | z6 +0.2%, z7-9 identical | +0.4% |
| jp | unchanged | +0.1% | 24.4 -> 25.3 KB (+3.9%) | identical | +0.2% |
| au | +0.6% | +1.8% | 13.2 -> 14.0 KB (+6.5%) | z6 +0.1%, z7-9 identical | +0.4% |
| cn | 35.0 -> 35.7 KB | 57.4 -> 59.5 KB | 87.7 -> 95.4 KB (+8.8%) | identical | +0.5% |
| in | 14.8 -> 15.0 KB | 22.3 -> 22.6 KB | 34.7 -> 38.1 KB (+9.8%) | z6 +0.3%, z7-9 identical | +0.6% |

z6 grows only where a z7 kind (narrow gauge) was promoted to z5. The largest single tile at
z5 goes from 11.3 to 13.2 KB (ru) and 21.6 to 23.5 KB (de). The z10-13 tiles, most of each
archive, do not change. With `--no-long-lines` the new code writes ca's z0-9 tiles
identical to the old code's.

Screenshots (headless, light theme, "View all lines", 1200x800, trial tiles swapped in by a
fetch rewrite): eastern Canada, the US west, Siberia and Germany at z4/5/6. Eastern Canada
gains the Ocean Mont-Joli - Campbellton - Bathurst - Moncton at z4 and z5, plus Montréal -
Jonquière/Senneterre and Sept-Îles - Schefferville (Tshiuetin). The US west and Siberia
barely change (their long lines are `usage=main` already). Germany at z5 gains short
branch stretches of 150 km+ RE lines; it was already dense. In the noritetsu root:
`review_tiles_ecanada_z5.png`, `review_tiles_ecanada_z4.png`, `review_tiles_germany_z5.png`
(before | after, new track ringed red).

Draw order: a tile's features come out in STRtree query order (`tile_features`, `hit`), so
adding any feature to a tile can reshuffle which of two overlapping coloured lines is on top
in that tile (seen in Germany at z5 and z6). Not new and harmless; `np.sort(hit)` would make
the order stable across builds (the source order), at the cost of a one-off reshuffle
everywhere. Not done.

**ca's tiles are rebuilt in dist** with this (compare_lines: 131 register lines, 0 differ).

## The page

No change needed: the track layers have no minzoom or rank filter, so whatever the tiles
carry at a zoom is drawn.

## Rebuilding every country's tiles

tools/rebuild.py has no tiles-only switch. A three-line diff (tools/ is the managing
session's):

```diff
@@ def run_country(cc, model_only):
-def run_country(cc, model_only):
+def run_country(cc, model_only, tiles_only=False):
     reg = REGISTER.get(cc, f"rinf:data/raw/rinf/{cc}")
-    steps = [["build_model.py", "--region", cc] + (["--register", reg] if reg else [])]
+    steps = [] if tiles_only else [["build_model.py", "--region", cc] + (["--register", reg] if reg else [])]
     if not model_only:
         steps.append(["build_tiles.py", "--region", cc])
@@ def main():
     model_only = "--model-only" in args
+    tiles_only = "--tiles-only" in args
@@
-        results = list(pool.map(lambda cc: run_country(cc, model_only), regions))
+        results = list(pool.map(lambda cc: run_country(cc, model_only, tiles_only), regions))
```

Then (every region with a pmtiles today, ca already done):

    python tools/slot.py 3 -- python tools/rebuild.py --tiles-only ae al am ao ar at au az ba bd be bf bg bo br by cd cg ch cl cm cn co cr cu cz de dj dk do dz ec ee eg es et fi fr ga gb ge gh gr hk hr hu id ie il in iq ir it jo jp ke kg kh kp kr kz la lk lt lu lv ma md me mg mk mm mn mu mw mx my mz ng nl no np nz pa pe ph pk pl pr pt qa ro rs ru sa se sg si sk sn th tj tm tn tr tw tz ua ug us uy uz ve vn xa xk za zm zw

(Leave `us` out if the US data agent's own rebuild, which already runs the new build_tiles,
lands after this change.) Without the diff, the same per country:
`python tools/slot.py -- python build_tiles.py --region <cc>`.

Time: the last logged tile builds of 80 countries add up to 36 min run one at a time (cn
5.2 min, us 4.6, de 3.6, ru 2.3, jp 2.1, fr 1.9, in 1.6, gb 1.5); the other 40 are small
(well under a minute each). Three at once: about 15-20 minutes. Tiles only, so
compare_lines shows nothing moving; no build_regions.py needed (regions.json does not read
the tiles).
