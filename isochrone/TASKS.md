# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-11-07).** Walking and canoe both work worldwide (L10 world, L3 and L1
view windows, 100 m for towns, water country and mountains: 5,350 tiles), on a globe, with hex
cells. Walking is on our own surface with OSM roads and directed from 90 m GLO-90 terrain at every
level, L3 and L10 made by crossing L1 (T-011), and at 100 m by the same slope rule (T-039). The
canoe paddles rivers, lakes and sea with currents and carries overland at x2. Water and land
danger are on with K = 1000 h, lakes with their own temperature. Corridors are coloured by speed,
drawn by trips and hoverable. M1 and M2 (survival) are closed, their doc tidies done (T-023,
T-078). Smoke passes 18/18.

**Next up**: the known coarse-level errors, T-007 stage 2 (fjords and rivers too fast at L3 and
L10, waiting on Anita's go) and T-075 (islands); a small one: T-095 (re-time on a quiet
machine). 10,000 BC is for later. Open questions:
`notes/for-anita.md`.

Milestones started from Anita's question, "if you just have a canoe in 10,000 BC, how far can
you get?": the canoe in today's world, roads and 100 m, survival; 10,000 BC moved to later
(Anita, 2026-11-05: "not really that necessary").

## In progress

## M1: canoe, today

- T-075 [P3] Island water: water round an island that no river line comes near stays unreached on canoe starts (25 cells in the Amazon window at Santarem): the water is reached only from the island and the island only from the water (notes/T-036.md). The one-pixel ponds in river water were fixed by T-088; the island case needs depth counted from canoe nodes first, which lets canoes reach banks across all wide river water, so measure it first (notes/T-088.md).
- T-102 [P2] Canoe level agreement after T-100: coarse river steps now count only each piece's main river (Kisangani to Kinshasa at L10 30.8 -> 20.8 days), so re-measure canoe L1/L10 and L1/L3 on T-092's 25 starts (its harness, notes/T-092/lib.mjs and measure.mjs, is in git at 5469ab95) and check `cfg.access` still holds (notes/T-100.md, notes/T-092.md).
- T-065 [P3] Upstream creek-weaving: going up the Madeira or Solimoes the cheapest canoe route weaves between small rivers that share an L10 cell, paddling down some and up others (flat small rivers 4.3 km/h up at Q = 1; rapids 2.5 since T-073), since the Amazon main stem upstream sits at the 2.5 km/h floor (`upMin`, T-071); Belem to Manaus paddles ~65 h downstream on the way up (68 h up, 2026-11-07). A side effect of joining rivers that share a cell (docs/canoe.md 8). Decide whether it needs fixing once Anita has seen it (notes/T-064.md, notes/T-071.md).

## Later

- T-096 [P3] Check FLake lake temperatures against satellite: T-043 gives lakes ERA5's SST, else FLake's `lmlt`, which are model values; high Tibetan lakes at 2-5 C in June-August may be too cold, Titicaca 16.7 C in December-February (just over the 15 C cold line) too warm. Compare big lakes with ESA CCI Lakes (open at CEDA; subset, the whole set is 768 GB) (notes/T-043.md).
- T-012 [P3] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). Later at Anita's word (2026-11-05).
- T-050 [P3] Fine level stage 3: the canoe at L0.1 (LC100 water kinds, bridge cells as both land and canoe nodes, RiverATLAS steps at 1/1200 degree); until then a canoe start stops at L1 (docs/fine.md).
- T-035 [P3] Canoe search speed: a canoe click at L10 is ~0.8 s with 8 moves past 20 km from shore (16 everywhere was 1.0-1.1 s), and canoe L3 windows are capped at 1.5 M cells. The bucket queue (T-093) gave the canoe L10 click only ~1.1x in Chrome and none in node: its cost is the sea edges. Try fewer tables read per sea edge, then go back to 16 moves at sea if it gets under 1 s (notes/T-022.md).
- T-007 [P3] Coarse cells across water: under T-011's crossing rule a coarse cell joins land L1 keeps apart, so fjord and river country is too fast at L3/L10 (Sogndal L1/L10 1.32, Coari 1.39). Stage 1 (design) done 2026-11-07: split coarse cells into their L1 land pieces, every piece a node with its own crossings and links (prototype: all 25 starts L1/L10 1.02 (0.93-1.21) -> 1.01 (0.93-1.13), L1/L3 to 97% within 10%; ~5 MB, search speed unchanged). Stage 2 (build_tiles.py, worker) waits on Anita's go (notes/T-007.md).
- T-095 [P3] Re-time the solves on a quiet machine (smoke.mjs and notes/T-093/chrome.mjs, when the core ledger is near empty) and restate SPEC.md 2's 0.3-0.7 s L10 click and ~0.8 s canoe click and docs/canoe.md's canoe times; T-093's bucket queue made them 1.1-1.4x faster but was measured under full load.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is ~750 MB since T-011 (l1 151, d1 314, e1 144, r1 66 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. A K slider and roads and era toggles in the mode panel (docs/canoe.md 3). OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
