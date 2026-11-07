# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-11-07).** Walking and canoe both work worldwide (L10 world, L3 and L1
view windows, 100 m for towns, water country and mountains: 5,350 tiles), on a globe, with hex
cells. Walking is on our own surface with OSM roads and directed from 90 m GLO-90 terrain at every
level, L3 and L10 made by crossing L1 (T-011), and at 100 m by the same slope rule (T-039). The
canoe paddles rivers, lakes and sea with currents and carries overland at x2. Water and land
danger are on with K = 1000 h. Corridors are coloured by speed, drawn by trips and hoverable.
Smoke passes 18/18.

**Next up**: the search's bucket queue (T-093), then the
known coarse-level errors (T-007 fjords and rivers, T-075 islands) and M2's doc tidy (T-078).
10,000 BC is for later. Open questions: `notes/for-anita.md`.

Milestones started from Anita's question, "if you just have a canoe in 10,000 BC, how far can
you get?": the canoe in today's world, roads and 100 m, survival; 10,000 BC moved to later
(Anita, 2026-11-05: "not really that necessary").

## In progress

## M1: canoe, today

- T-031 [P2] River gradient south of 60N: `sgr_dk_rav` closes 24% of paddleable length (mostly Q < 10 streams over 5 m/km) and reads lowland Amazon streams as rapids (Manaus to the sea sums to 140 m, real ~20). Once canoe routes show on the page (T-022), compare a few basins with the GMTED drop-over-10 km used north of 60N (`rivers.py north_gradient`) and pick per region by eye; the Livingstone Falls stretch between OSM rapids points still runs (notes/T-020.md).
- T-083 [P3] Desert river permanence: Cooper Creek and the Diamantina still paddle since WaterGAP says the Thomson never runs dry; try GIRES (Messager et al. 2021, CC BY 4.0) no-flow flags on hot-desert reaches. Glacier-fed desert rivers (Tarim, Hotan, Keriya) drop today since WaterGAP has no glacier melt; matters for 10,000 BC (notes/T-079.md).
- T-075 [P3] Island water: water round an island that no river line comes near stays unreached on canoe starts (25 cells in the Amazon window at Santarem): the water is reached only from the island and the island only from the water (notes/T-036.md). The one-pixel ponds in river water were fixed by T-088; the island case needs depth counted from canoe nodes first, which lets canoes reach banks across all wide river water, so measure it first (notes/T-088.md).
- T-065 [P3] Upstream creek-weaving: going up the Madeira or Solimoes the cheapest canoe route weaves between small rivers that share an L10 cell, paddling down some and up others (4.3 km/h at Q = 1), since the Amazon main stem upstream sits at the 2.5 km/h floor (`upMin`, T-071); Belem to Manaus still paddles ~78 h downstream on the way up. A side effect of joining rivers that share a cell (docs/canoe.md 8). Decide whether it needs fixing once Anita has seen it (notes/T-064.md, notes/T-071.md).
- T-073 [P3] Small rivers may be too fast both ways: upstream on a Q = 1 stream is 4.3 km/h, but shallow small rivers were often dragged or poled (Lewis and Clark up the Jefferson). Check small-river records and whether a depth or width limit belongs in the upstream rule (notes/T-071.md).

## Roads and 100 m


## M2: survival

- T-006 [P2] Danger and survival: per-cell hazard for walking and canoe (research/friction.md 6, canoe.md 5), the search's cost becomes time + K x hazard (SPEC 3), survival along the path in the hover. Climate from CHELSA (stage E). Stage 1 done 2026-10-26: water hazard on canoe starts, K = 1000 kept after the check trips (notes/T-006.md). Land danger is done (T-042, 2026-11-05: design, world band, page; walking trips in checks/ktrips.mjs, notes/T-042.md and T-067.md). Left: close with M2's doc tidy (T-078).
- T-043 [P3] Lake water temperature: lakes take the nearest sea's summer temperature for the cold factor of their wave hazard (docs/canoe.md 3), so a big cold lake far from a cold sea reads warm. Try ERA5 lake mix-layer temperature for lakes over ~100 km².
- T-078 [P2] M2 doc tidy: spec against code, prune TASKS, fold notes (as T-023 did for M1). Last in M2, after T-006 and T-043 are closed or moved.

## Later

- T-012 [P3] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). Later at Anita's word (2026-11-05).
- T-093 [P3] Bucket queue in the search for every level (from T-039, fine level stage 2): a bucket queue in place of the heap was 1.3-1.6x faster in T-030's test; L0.1 windows at the cap take ~1.1-1.2 s in node with the slope per move (notes/T-039.md "Times"), canoe L10 ~0.8 s (T-035). Measure in a real browser (docs/fine.md "Times"). taking, 2026-11-07
- T-050 [P3] Fine level stage 3: the canoe at L0.1 (LC100 water kinds, bridge cells as both land and canoe nodes, RiverATLAS steps at 1/1200 degree); until then a canoe start stops at L1 (docs/fine.md).
- T-051 [P2] Rural seams at L0.1 window edges: MAP's L1 walks a whole 1 km pixel at road speed when a road crosses it, so outside towns 100 m times are 18-52% longer than L1 (notes/T-047.md) and an L0.1 window's ring is seeded early. Check how it looks round Cluj at zoom 10-11; the real fix is L1 from our own surface with roads as a share of each cell (T-010, T-028).
- T-035 [P3] Canoe search speed: a canoe click at L10 is ~0.8 s with 8 moves past 20 km from shore (16 everywhere was 1.0-1.1 s), and canoe L3 windows are capped at 1.5 M cells. Try the bucket queue of T-093 and fewer tables read per sea edge, then go back to 16 moves at sea if it gets under 1 s (notes/T-022.md).
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-011. Under T-011's crossing rule fjord and river country is too fast at L3/L10 (Sogndal L1/L10 1.32, Coari 1.32): a coarse cell crossed along its land joins its neighbours, so coarse walks cross water L1 goes round (notes/T-011.md stage 3).
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 735 MB since T-011 (l1 151, d1 314, e1 144, r1 66 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
