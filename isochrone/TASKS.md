# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-11-05, work resumed after a pause from 2026-10-26).** Walking and canoe both
work worldwide (L10 world, L3 and L1 view windows, 100 m in New York and Cluj). The canoe paddles
rivers, lakes and sea with currents and carries overland; water danger is on with K = 1000 h and
the hover shows survival. Corridors, a speeds panel, hill shading and river lines are on the page.
Our own 1 km walking surface is built but not on the page. Smoke passed 15/15 at the pause.

**Next up** (Anita, 2026-11-05): OSM roads at 1 km worldwide, fetched region by region since
the planet won't fit on C: (T-010), then our own surface on the page (T-028), which fixes rural
walking times 18-52% too short; the 100 m tiles by T-047's rule from the same fetch (T-057);
then land danger (T-042). 10,000 BC is for later. Open questions: `notes/for-anita.md`.

Milestones started from Anita's question, "if you just have a canoe in 10,000 BC, how far can
you get?": the canoe in today's world, roads and 100 m, survival; 10,000 BC moved to later
(Anita, 2026-11-05: "not really that necessary").

## In progress

(none)

## M1: canoe, today

- T-056 [P3] Corridor rays: over uniform land (Sahara, steppe) corridor trunks fan out as straight rays along the 8 grid directions, fainter ones dark red spokes (seen in t052_sahara_after.png). Needs a look at `js/corridors.js` (smoothing or tie-breaking of equal-cost paths), not a look tweak.
- T-031 [P2] River gradient south of 60N: `sgr_dk_rav` closes 24% of paddleable length (mostly Q < 10 streams over 5 m/km) and reads lowland Amazon streams as rapids (Manaus to the sea sums to 140 m, real ~20). Once canoe routes show on the page (T-022), compare a few basins with the GMTED drop-over-10 km used north of 60N (`rivers.py north_gradient`) and pick per region by eye; the Livingstone Falls stretch between OSM rapids points still runs (notes/T-020.md).
- T-036 [P3] Canoe: L1 river water no river step crosses (ponds and side channels HydroLAKES lacks, e.g. round the Danube mouth) has no canoe node, so it stays unreached and unpainted on a canoe start. Decide: still water when no step touches the patch, or leave (notes/T-022.md).
- T-065 [P3] Upstream creek-weaving: going up the Madeira or Solimoes the cheapest canoe route weaves between small rivers that share an L10 cell, paddling down some (6.9 km/h) and up others (3.9), since the main stem upstream is floored at 1.5 km/h (`upMin`); upstream corridors read 4-6 km/h. A side effect of joining rivers that share a cell (docs/canoe.md 8). Decide whether it needs fixing once Anita has seen it (notes/T-064.md).
- T-060 [P2] Built 2026-11-05, Anita picks: `?look=diag` (corner cuts, `DIAG_CUT`) and `?look=hex` (`HEX_SIZE`) in `js/paint.js`, default crisp; screenshots `t060_*`. Cell look at zoom 8-10: keep crisp square cells (Anita, 2026-11-05: no smoothing), and test painting band edges with cut diagonal corners (rounded steps) and a hex look painted from the square cells, screenshots side by side for Anita. Painting only; the grid stays square (docs/fine.md "Hexes"). After T-061.
- T-023 [P2] M1 doc tidy: spec against code, prune TASKS, fold notes. Last in M1.

## Roads and 100 m

- T-010 [P1] Roads as their own layer from OSM (DECISIONS 2026-10-26; research/friction.md 5, stage B), so "no roads" is a toggle and 10,000 BC has none. Stage 1 done 2026-11-05: tools/roads/, rule share3 (e = min(1, 3 x road share of the cell); 4% median off 100 m in the test tiles, MAP 22%), design in docs/roads.md for Anita's review (notes/T-010.md). Stage 2 is T-053.
- T-053 [P2] Roads world build (T-010 stage 2): fetch, extract and delete the 513 Geofabrik extracts of `data/roads/regions.json` one by one (84.6 GB), then `reduce` and a world GeoTIFF of the 1 km layer (docs/roads.md "Fetching the world"). A small driver (fetch the next while one is read) is the only new code. ~8 h on one core, ~4 h on two; peak disk ~4 GB. After Anita's review of docs/roads.md.
- T-028 [P2] Feed the page from our own surface: build_grid.py and build_tiles.py read `data/surface/walk_today.tif` (uint16 km/h x 1000, 0 = water) plus the roads layer (T-010) instead of MAP's tif; decide the water rule from `lc_water.tif` shares with T-013's small-water rule; smoke passes. After T-010 (docs/surface.md).
- T-057 [P2] Fine level world build: the 100 m tiles T-047's rule picks (~2.8 GB, docs/fine.md "Where the fine level goes", approved 2026-11-05), their `.roads` from the same region-by-region OSM pass as T-010's world build. Towns and water country first; mountain tiles gain mostly from 90 m slopes, so they wait for T-039. After T-010 stage 1.

## M2: survival

- T-006 [P2] Danger and survival: per-cell hazard for walking and canoe (research/friction.md 6, canoe.md 5), the search's cost becomes time + K x hazard (SPEC 3), survival along the path in the hover. Climate from CHELSA (stage E). Stage 1 done 2026-10-26: water hazard on canoe starts, K = 1000 kept after the check trips (notes/T-006.md). Next: the land band (T-042), then re-run the check trips with walking and carrying paying hazard (the Saharan walk, Cluj walking).
- T-042 [P2] Land hazard band: a uint8 log-coded rate per cell in every level's tiles and the L10 grid (heat, cold, altitude, distance to water, ice; CHELSA climate; research/friction.md 6), read by walking and carrying (x their time) and by getting in and out; walking starts leave the walking-only loop. After T-006 stage 1 (docs/canoe.md 3).
- T-043 [P3] Lake water temperature: lakes take the nearest sea's summer temperature for the cold factor of their wave hazard (docs/canoe.md 3), so a big cold lake far from a cold sea reads warm. Try ERA5 lake mix-layer temperature for lakes over ~100 km².

## Later

- T-012 [P3] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). Later at Anita's word (2026-11-05).
- T-039 [P3] Fine level stage 2: GLO-90 elevation in the 100 m tiles, directed slope per move with the street grade cap, a bucket queue in the search for every level, times in a real browser (docs/fine.md).
- T-050 [P3] Fine level stage 3: the canoe at L0.1 (LC100 water kinds, bridge cells as both land and canoe nodes, RiverATLAS steps at 1/1200 degree); until then a canoe start stops at L1 (docs/fine.md).
- T-051 [P2] Rural seams at L0.1 window edges: MAP's L1 walks a whole 1 km pixel at road speed when a road crosses it, so outside towns 100 m times are 18-52% longer than L1 (notes/T-047.md) and an L0.1 window's ring is seeded early. Check how it looks round Cluj at zoom 10-11; the real fix is L1 from our own surface with roads as a share of each cell (T-010, T-028).
- T-035 [P3] Canoe search speed: a canoe click at L10 is ~0.8 s with 8 moves past 20 km from shore (16 everywhere was 1.0-1.1 s), and canoe L3 windows are capped at 1.5 M cells. Try the bucket queue of T-039 and fewer tables read per sea edge, then go back to 16 moves at sea if it gets under 1 s (notes/T-022.md).
- T-011 [P3] 90 m terrain: stream Copernicus GLO-90 tile by tile, directed crossing costs per cell (research/friction.md 7, stage C); makes walking directed. 100-130 GB streamed, run detached overnight. After T-009.
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-011.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 135 MB (L1 117.5 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
