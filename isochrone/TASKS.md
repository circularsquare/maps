# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-11-05, evening).** Walking and canoe both work worldwide (L10 world,
L3 and L1 view windows, 100 m in New York and Cluj), on a globe, with hex cells. The canoe paddles
rivers, lakes and sea with currents (upstream 0.4 of the current, floor 2.5 km/h) and carries
overland at x2. Water and land danger are on with K = 1000 h; the hover shows survival from 0.5%
risk. Corridors are coloured by speed, drawn by trips, re-chosen over land and hoverable; a
speeds panel, hill shading and river lines are on the page. Our own 1 km walking surface and the
roads design (share3 rule) are built but not on the page. The world roads pass (T-053) is running
in the background. Smoke passes 15/15.

**Next up** (Anita, 2026-11-05): when the roads pass ends, reduce and build the world GeoTIFF,
then our own surface with roads on the page (T-028, which fixes rural walking times 18-52% too
short) and the 100 m tiles by T-047's rule (T-057, towns and water country first). Then 90 m
terrain (T-011), then T-039 and the mountain tiles. 10,000 BC is for later. M1 and M2 each wait
on a doc tidy (T-023 done; M2's is T-078). Open questions: `notes/for-anita.md`.

Milestones started from Anita's question, "if you just have a canoe in 10,000 BC, how far can
you get?": the canoe in today's world, roads and 100 m, survival; 10,000 BC moved to later
(Anita, 2026-11-05: "not really that necessary").

## In progress

- T-087 [P3] Road share factor on a farm grid: Iowa walks 1.34-1.47x slower than MAP, since share3 blends a 1.6 km road grid with 2.5 km/h cropland to ~3-3.5 km/h where a walker on the grid would make ~3.9 (notes/T-084.md). Add a farm-grid tile to the factor-3 check (docs/roads.md). Measured 2026-11-06 (notes/T-087.md): factor 3 fits dense and winding nets (2.5-3.3) but leaves straight grids 11-18% too slow (Iowa wants 5-6); a straightness rule fits all ten usable tiles to 3.9% mean; a 1 km mean of land cover is also 10-20% slower than a 100 m walker even without roads. Now prototyping L1 by 100 m crossing time (T-080's rule one level down), which would cover all three, with two prairie tiles. Agent, Sonnet.
- T-081 [P2] Coarse levels by crossing time (T-080's proposal): each L10 and L3 cell's speed is the best 1 km crossing of it, east-west and north-south, geometric mean; land/water rule and format unchanged, only `coarse()` in build_tiles.py. L1/L10 goes 0.81 -> 1.01, window seams go away; zoomed-out walks reach ~20% farther. Waits for Anita's yes (notes/T-080.md). Same rule for L1 from 100 m inputs later. If it lands, re-measure the canoe's `cfg.access` (0.4-0.6, notes/T-082.md).

## M1: canoe, today

- T-031 [P2] River gradient south of 60N: `sgr_dk_rav` closes 24% of paddleable length (mostly Q < 10 streams over 5 m/km) and reads lowland Amazon streams as rapids (Manaus to the sea sums to 140 m, real ~20). Once canoe routes show on the page (T-022), compare a few basins with the GMTED drop-over-10 km used north of 60N (`rivers.py north_gradient`) and pick per region by eye; the Livingstone Falls stretch between OSM rapids points still runs (notes/T-020.md).
- T-083 [P3] Desert river permanence: Cooper Creek and the Diamantina still paddle since WaterGAP says the Thomson never runs dry; try GIRES (Messager et al. 2021, CC BY 4.0) no-flow flags on hot-desert reaches. Glacier-fed desert rivers (Tarim, Hotan, Keriya) drop today since WaterGAP has no glacier melt; matters for 10,000 BC (notes/T-079.md).
- T-075 [P3] Island water: water round an island that no river line comes near stays unreached on canoe starts (25 cells in the Amazon window at Santarem): the water is reached only from the island and the island only from the water (notes/T-036.md). The one-pixel ponds in river water were fixed by T-088; the island case needs depth counted from canoe nodes first, which lets canoes reach banks across all wide river water, so measure it first (notes/T-088.md).
- T-065 [P3] Upstream creek-weaving: going up the Madeira or Solimoes the cheapest canoe route weaves between small rivers that share an L10 cell, paddling down some and up others (4.3 km/h at Q = 1), since the Amazon main stem upstream sits at the 2.5 km/h floor (`upMin`, T-071); Belem to Manaus still paddles ~78 h downstream on the way up. A side effect of joining rivers that share a cell (docs/canoe.md 8). Decide whether it needs fixing once Anita has seen it (notes/T-064.md, notes/T-071.md).
- T-073 [P3] Small rivers may be too fast both ways: upstream on a Q = 1 stream is 4.3 km/h, but shallow small rivers were often dragged or poled (Lewis and Clark up the Jefferson). Check small-river records and whether a depth or width limit belongs in the upstream rule (notes/T-071.md).

## Roads and 100 m

- T-057 [P2] Fine level world build: the 100 m tiles T-047's rule picks (~2.8 GB, docs/fine.md "Where the fine level goes", approved 2026-11-05), their `.roads` from the same region-by-region OSM pass as T-010's world build. Towns and water country first; mountain tiles gain mostly from 90 m slopes, so they wait for T-039. Towns and water country built 2026-11-06 (4,672 tiles, 0.52 GB, notes/T-057.md); partial town tiles need T-086 to show; mountain tiles wait for T-039.
- T-011 [P2] 90 m terrain: stream Copernicus GLO-90 tile by tile, directed crossing costs per cell (research/friction.md 7, stage C); makes walking directed and mountains show (Anita, 2026-11-05: queue after roads). 100-130 GB streamed, never all on disk, run detached overnight. After T-053.

## M2: survival

- T-006 [P2] Danger and survival: per-cell hazard for walking and canoe (research/friction.md 6, canoe.md 5), the search's cost becomes time + K x hazard (SPEC 3), survival along the path in the hover. Climate from CHELSA (stage E). Stage 1 done 2026-10-26: water hazard on canoe starts, K = 1000 kept after the check trips (notes/T-006.md). Land danger is done (T-042, 2026-11-05: design, world band, page; walking trips in checks/ktrips.mjs, notes/T-042.md and T-067.md). Left: close with M2's doc tidy (T-078).
- T-043 [P3] Lake water temperature: lakes take the nearest sea's summer temperature for the cold factor of their wave hazard (docs/canoe.md 3), so a big cold lake far from a cold sea reads warm. Try ERA5 lake mix-layer temperature for lakes over ~100 km².
- T-078 [P2] M2 doc tidy: spec against code, prune TASKS, fold notes (as T-023 did for M1). Last in M2, after T-006 and T-043 are closed or moved.

## Later

- T-012 [P3] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). Later at Anita's word (2026-11-05).
- T-039 [P3] Fine level stage 2: GLO-90 elevation in the 100 m tiles, directed slope per move with the street grade cap (until then 100 m walks hill roads slower than L1, which has the 10% cap since T-084), a bucket queue in the search for every level, times in a real browser (docs/fine.md).
- T-050 [P3] Fine level stage 3: the canoe at L0.1 (LC100 water kinds, bridge cells as both land and canoe nodes, RiverATLAS steps at 1/1200 degree); until then a canoe start stops at L1 (docs/fine.md).
- T-051 [P2] Rural seams at L0.1 window edges: MAP's L1 walks a whole 1 km pixel at road speed when a road crosses it, so outside towns 100 m times are 18-52% longer than L1 (notes/T-047.md) and an L0.1 window's ring is seeded early. Check how it looks round Cluj at zoom 10-11; the real fix is L1 from our own surface with roads as a share of each cell (T-010, T-028).
- T-035 [P3] Canoe search speed: a canoe click at L10 is ~0.8 s with 8 moves past 20 km from shore (16 everywhere was 1.0-1.1 s), and canoe L3 windows are capped at 1.5 M cells. Try the bucket queue of T-039 and fewer tables read per sea edge, then go back to 16 moves at sea if it gets under 1 s (notes/T-022.md).
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-011.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 135 MB (L1 117.5 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
