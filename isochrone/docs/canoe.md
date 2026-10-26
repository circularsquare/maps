# Canoe

The canoe layer answers "if you just have a canoe, how far can you get?". Stage 1 is today's
world; 10,000 BC later swaps the land mask and opens dams (section 9), with the same files and
search. Model numbers and their sources: `research/canoe.md`. Working and measurements:
`notes/T-005.md`.

Anita's rules (2026-10-26): the canoe is **always with you**, carried overland at twice the
walking time; the search minimises **time + K x hazard** on every layer, one K, and shows survival
along the chosen path.

## 1. Graph

| Layer | Nodes | In which search |
|---|---|---|
| walking | every land cell | walking start (as now) |
| carrying | every land cell, walking time x 2 | canoe start |
| canoe | every sea and lake cell, plus a canoe node on each land cell a paddleable river or a strait passes through | canoe start |

- **A canoe start is canoe + carrying; a walking start is walking alone.** You never leave the
  canoe, so plain walking cannot be reached from a canoe start, and a walking start never meets a
  canoe. Walking and carrying are the same land cells at a different speed, so each land cell has
  one land node whose speed factor is 1 or 2 by start mode.
- **Node ids.** Ids 0..N-1 are the cells: a land cell's id is its land node, a sea or lake cell's
  its canoe node. Canoe nodes on land cells (river cells, and at L3 and L10 coastal and strait
  cells) get ids from N on, found through a per-cell table `canoeOf` (-1 where none): ~1 M extra
  at L10 (section 7).
- **Water** is as now: MAP's water value; a coarse cell is water when under a quarter of its L1
  pixels are land. Water is **sea** inside the ocean (Natural Earth 10 m ocean polygon for
  today, and water joined to it within 8 L1 pixels where NE's coast and MAP's differ), **lake**
  inside a HydroLAKES polygon, else **river water** (wide rivers MAP marks as water: the Amazon,
  the lower Danube). An L1 land pixel that NE's ocean touches keeps its land node and gets a sea
  canoe node too: MAP walks over some straits (the Bosporus, a gap in the Dardanelles), and this
  keeps them open. River water has no grid edges; the canoe uses only the
  river edges running through it, so a wide river is never a current-free channel beside its own
  current.

## 2. Edges and times

Paddling speed p = 5 km/h on still water. Clock: paddling hours, the same 8-hour day as walking.

| Edge | Between | Time | Directed |
|---|---|---|---|
| walk / carry | land node to its 16 neighbours | as now; x 2 when carrying | no (until T-011) |
| river down | canoe node to the next cell of a reach | length / (p + v) | yes |
| river up | the reverse | length / max(p - 0.6 v, 1.5 km/h) | yes |
| lake | canoe node to neighbouring lake canoe nodes | length / p | no |
| sea | canoe node to neighbouring sea canoe nodes | length / s, s from the current | yes |
| transfer | land node to the canoe node in its own cell or one of its 8 neighbours | 0.25 h + half the step at each side's speed | both ways |

- **River speed** v = 0.51 Q^0.11 m/s from mean discharge Q, from hydraulic geometry W = 7.2 Q^0.5,
  D = 0.27 Q^0.39 (Moody and Troutman 2002). A reach is paddleable at Q >= 1 m³/s.
- **River length** is the reach's real length spread over its cell steps, so meanders cost their
  length.
- **Sea current** c is the mean of the edge's two cells from the drifter climatology, x0.5 within
  5 km of shore. Along the edge direction d: s = c·d + sqrt(p² - (c×d)²); no edge when the
  cross-current exceeds p or s <= 0.
- **Ice**: sea cells with mean ice cover above 15% in the least-ice month (August north of the
  equator, February south) have no canoe node.
- **Moves on water**: 8 neighbours; the 8 knight moves only where all four cells involved are
  open water (walking needs all four on land). At L3 and L10 a canoe edge exists only where the
  two cells' L1 water touches across their shared border (`links`, section 5): a strait narrower
  than a cell (the Bosporus, the Danish straits, fjords) stays open at L10, and two waters a few
  km apart are not joined. Inside one coarse cell all its water counts as joined, so an isthmus
  narrower than a cell can be crossed at L10 where one cell holds water of both sides; the L1
  window corrects it. (Corinth is not: no L10 or L3 cell holds both gulfs.)
- **Portage has no edges of its own.** Unrunnable river steps are dropped: a fall, a dam, a weir,
  or a reach steeper than 5 m/km (2 m/km where Q >= 1000). The search gets off (0.25 h), carries
  along the bank at walking x 2 and gets back on (0.25 h), about the research's "1 h plus the
  carry" for a fall. Watersheds between rivers are crossed the same way. Upstream on a fast river,
  carrying along the bank can beat lining at 1.5 km/h; the search takes the quicker.
- **Rapids** of 1-5 m/km are run at the same speeds, with their own hazard class.

## 3. Time and danger

The search minimises **weight = time + K x hazard**, summed along the path, on every layer, with
one K. Hazard on an edge is its rate x real hours (research/canoe.md 5, friction.md 6); survival
along the chosen path is exp(-hazard sum). The painted time is the time of the chosen route, not
the fastest possible.

**Starting K = 1000 h**: 10 hours (1.25 days of travel) to cut the death risk by 1%. What it does,
worked by hand with research/canoe.md 5's rates, summer:

- **Brest to A Coruña.** Straight across the Bay of Biscay is 635 km, ~127 paddling hours
  (16 days), up to ~250 km from land in 1-2 m waves: hazard 0.4-1.1, survival 35-70%. Round the
  coast is ~1,150 km, ~230 h (29 days), hazard ~0.005, survival ~99.5%. The coast wins for any
  K above 100-300 h. At K = 1000 the map says 29 days and you arrive; time alone would have said
  16 days and a hover saying you probably drown.
- **Taiwan to Yonaguni** (110 km, no safer way round): the crossing is taken at any K, ~22
  paddling hours, survival ~90%. K never makes a place unreachable; it changes the route only
  where a safer one exists within K x (hazard saved) hours.
- **A 20 km strait** (hazard ~0.001) costs 1 h of weight: crossed unless the way round is
  under an hour longer.
- **Open ocean.** Past 600 km from land the supplies term (0.05 an hour) weighs each paddling
  hour as ~50, so the search hardly goes there unless it must.
- **Walking** barely changes: background 1e-6 an hour adds 0.1%; forest or swamp (1e-5) 1%;
  a hot desert with water or high altitude (1e-4) 10%, so a path skirts them when the detour is
  under 10%; above 8,000 m or a waterless desert (1e-2) weighs x11, so it is avoided.

**Tuning K.** A set of check trips in notes/T-005.md (Biscay round the coast, a 20 km strait
crossed, Taiwan-Yonaguni crossed, the Gulf of Lion, a Saharan walk) is run at K = 300, 1000,
3000 and 10000 once hazard exists; K is the value where they all read right. Above ~10,000 a
desert walk with wells goes far round, and every crossing over a day starts to look like a
detour candidate.

**A slider later.** K is one number in the worker; changing it re-solves (~0.3-0.8 s at L10),
with no new data. A slider in the mode panel (T-008), log steps from 0 ("fastest, whatever the
risk") to 10,000, costs nothing extra. It stays off the first panel until the checks settle K.

**Staged.** The search carries weight and hazard from T-022 on, with every hazard rate 0 until
T-006: time only, exactly the old map. Then the search needs from T-006:

- **water**: a table from what the canoe data already holds (kind, rapids class, distance to
  shore, summer wave height, sea temperature, exposure x3 past 20 km, supplies by distance) to a
  rate per hour. No new files; the worker computes it per edge.
- **land**: a hazard band in every level's tiles and the L10 grid (uint8, a log-coded rate per
  cell; heat, cold, altitude, distance to water, ice), since walking now pays for danger too.
  Roughly the size of the speed tiles at worst, likely far less (it is smooth).

Water hazard can land first: its inputs exist once T-020 and T-021 do.

## 4. Data per level

| File | Level | Holds | Size |
|---|---|---|---|
| `data/canoe/water.bin.gz` | L10, whole world | 3 bytes a cell: `kind`, `shore`, `links` | 1.1 MB |
| `data/canoe/l3/<r>_<c>.bin.gz` | L3, 5° tiles | the same 3 bands | 4.1 MB, 1130 tiles |
| `data/canoe/l1/<r>_<c>.bin.gz` | L1, 5° tiles | `kind`, `shore` (links implicit at L1) | 15.0 MB, 1141 tiles |
| `data/canoe/sea.bin.gz` | 0.25°, whole world | current U, V (int8, 0.02 m/s), summer wave height (uint8, 0.05 m), ice (uint8, %), sea temperature (int8, °C) | 0.6 MB |
| `data/canoe/rivers.bin.gz` | L10, whole world | ~1.1 M river steps | ~5 MB |
| `data/canoe/rivers/l3/`, `l1/` | 5° tiles | ~4 M and ~12 M river steps | ~15 MB, ~40-60 MB |

- Tiles share the speed tiles' grid and names (`docs/resolution.md`); `data/canoe/index.json`
  lists them, and the sea grid. A tile with no land and only sea is not written (the same tiles
  as the speed tiles): the worker treats it as open sea, `shore` sampled from the next coarser
  level (off by up to one coarse cell, only near a coast just outside the tile). Every file is
  gzip of whole bands one after another (band-sequential), each row-major from the north-west
  corner; byte layouts in notes/T-021.md.
- **kind** (uint8): 0 land, 1 sea, 2 river water, 3+ lake, coding the lake's wave height (fetch
  formula, U = 7 m/s, F = sqrt(area), research/canoe.md 2) in 0.05 m steps: Hs = (kind - 3) x
  0.05 m, 4 to 24 in the data. kind 1 or 3+ on a land cell (speed > 0) is its extra canoe node:
  at L3 and L10 any sea or lake pixel gives one, at L1 NE's ocean touching the pixel (section
  1). A coarse water cell is sea or lake when those pixels are at least its river-water pixels,
  sea when it has at least as many sea as lake pixels; a lake cell takes the largest lake code
  among its pixels.
- **shore** (uint8): great-circle distance from the pixel centre to the nearest L1 land pixel
  centre: 1-20 = km (rounded, at least 1), 21-78 = 20 + 10 x (code - 20) km, 78 = 600 km and
  beyond; 0 on land, 1 on a land pixel with a canoe node. A coarse cell stores the least over
  its water pixels (a coastal paddler hugs the coast). The grid ends at 60S, so Antarctica does
  not count as land.
- **links** (uint8, L3 and L10): bit k set when L1 sea or lake pixels of the cell and of
  neighbour k are 8-neighbours across their border (a diagonal: the two corner pixels), and both
  cells have a canoe node. k follows `js/search.js`'s moves as (dc, dr), dr down: 0 (1,0) E,
  1 (-1,0) W, 2 (0,1) S, 3 (0,-1) N, 4 (1,1) SE, 5 (1,-1) NE, 6 (-1,1) SW, 7 (-1,-1) NW. Bits are
  symmetric (a cell's bit k equals its neighbour's opposite bit) and wrap at 180°.
- **Sea grid**: 1440 x 580 cells of 0.25° from 180W, 85N. Local summer: June-August north of the
  equator, December-February south. Currents: the drifter monthly means over those months, else
  the annual mean, else 0 (40% of cells, mostly land). Wave height and sea temperature: the
  summer mean; ice: the August (north) or February (south) mean. Each is the mean of the four ERA5 points round the cell centre; cells with no ERA5 value (land, waves
  under ice) hold the nearest cell's, so a lookup anywhere returns a sea value.
- **River steps**: one record per downstream step, stored with the tile of the upstream cell, in
  columns: cell (uint32 within the tile), direction to the next cell (0-7), length (uint16, 10 m),
  v (uint8, 0.01 m/s), class (uint8: flat, rapids, unrunnable, fall, dam, weir, in a reservoir).
  The worker makes the upstream twin and drops the classes it closes, so a toggle (dams, era)
  needs no new data. Length and v, not time, are stored, so p stays a worker constant (a heavy
  dugout or a big crew later). If L1 comes out over 40 MB, T-020 stores reaches as chains (start
  cell, then 3-bit directions).
- **Rasterising.** Each reach with Q >= 1 is densified and walked through the cells of each
  level; back-and-forth steps along a cell border are folded (A B A becomes A; 5% of L10 steps);
  consecutive cells are 8-neighbours. A coarse step stores the total fine length and the
  length-weighted mean v. Reaches inside lakes and the sea are dropped (the water cells join
  inflow to outflow); reaches inside a reservoir are kept with that class.
- **Sources** (stage 1): RiverATLAS v1.0 (HydroRIVERS reaches with discharge, reach gradient
  `sgr_dk_rav`, and monthly minimum discharge for later; CC BY 4.0), HydroLAKES (local;
  `Lake_type` 2 = reservoir, its pour point = the dam), OSM `waterway=waterfall|dam|weir` by
  Overpass, the Natural Earth 10 m ocean polygon (public domain), the GDP drifter monthly
  climatology, ARCO-ERA5 sampled to a local-summer mean of wave height and sea temperature and a
  least-ice-month mean of ice,
  NSIDC median ice edges as a check. SWORD's obstruction flags are a later check (1.6 GB).

## 5. Search

- **One Dijkstra over the node ids**, as now, over each node's out-edges: grid moves of its
  layer, its river steps (a per-level index built at load from the records), and transfers.
  Directed edges need only that a cost is always worked out from u to v.
- **Arrays per node**: weight and hazard (Float32), heap and position (Int32), and the move it was
  reached by (Uint8: from-layer and move 0-16, for tracing the path). Time = weight - K x hazard.
- **Stop rule**: a node is not expanded once its time passes the 128-day limit or its survival
  falls below 1%.
- **Windows**: each layer's ring is seeded from the same layer of the coarser level (land from
  coarse land, canoe from the reached coarse canoe nodes round the point), with weight and
  hazard. A fine canoe node with no coarse canoe node near (a small river L10 lacks) is not
  seeded; the search reaches it inside the window by a transfer. Reuse, cancelling and caps stay
  as in `docs/resolution.md`.
- **Painting** a cell takes the least time of its nodes. On a canoe start sea and lakes are
  painted too, since where the canoe gets at sea is the answer; walking keeps water unpainted.

## 6. Hover and legend

- **Hover** on a canoe start: the time as now, then how it was spent, traced along the path:
  "3 days 5 h", "paddled 19 h, carried 6 h"; survival once hazard exists. The route shows as a
  thin line from the start to the pointer (traced through the finest window, then on through the
  coarser level from the seed it came in by).
- **Legend**: the same bands, nothing new.

## 7. Speed and memory

Measured in node on the L10 grid (sea at 5 km/h, carrying x 2, 0.25 h transfers, no currents or
rivers, 128-day limit; notes/T-005.md):

| Start | Walking | Canoe, 16 moves at sea | Canoe, 8 moves at sea |
|---|---|---|---|
| Cluj | 0.23 s | 0.23 s | 0.21 s |
| Lisbon | 0.07 s | 0.52 s | 0.42 s |
| Jakarta | Java only | 0.50 s | 0.29 s |

- The 128-day limit bounds the work: at 5 km/h a canoe covers ~5,000 km, ~1 M cells, not the
  5.3 M sea cells of the world. No coarser open-sea grid is needed.
- Currents (a square root per sea edge), river steps and the hazard array should add 20-50%:
  **~0.3-0.8 s per canoe click at L10**, against 0.2-0.3 s walking. K cuts the sea-heavy cases
  once hazard exists (the search stays near coasts).
- **16 moves at sea** keep bands round (8 moves run up to 8% long on diagonals). If a normal
  browser takes over 1 s, 8 moves past 20 km from shore is the fallback.
- **Windows** now solve their water too, so a coastal window costs up to twice what it did at the
  same size; the caps (2 M cells at L1, ~1-1.5 s) still bound it. T-022 measures and lowers the
  L1 cap for canoe starts if needed.
- **Memory** at L10: ~8.6 M nodes x 17 bytes plus `canoeOf` (4 bytes a cell), ~175 MB, against
  ~90 MB now.

## 8. Known approximations

- Distributaries and braids are missing (one channel per delta); tides are ignored; river flow is
  the annual mean (no low season); two rivers sharing an L10 cell are joined there.
- Window seams as for walking (`docs/resolution.md`), per layer.

## 9. 10,000 BC

The same model and files, built per era: `kind` from T-012's land mask (sea = below the era's sea
level and joined to the ocean; no canoe on ice), `shore` and `links` recomputed; river records
unchanged, with dams and weirs open and the steps inside reservoirs used (reservoirs are gone from
that era's land mask). Exposed shelves get no rivers in stage 1 (paleo rivers are later). Today's
currents, waves and ice are reused.

## 10. Build stages

| Id | Stage | Needs |
|---|---|---|
| T-020 | Rivers data: RiverATLAS to river steps at L1, L3, L10 (Q >= 1, v from Q, gradient classes), OSM falls, dams and weirs, HydroLAKES reservoirs | this review |
| T-021 | Lakes and sea data: water tiles (`kind`, `shore`, `links`) at all levels, `sea.bin.gz` from drifters and ERA5 summer | this review |
| T-022 | Search with layers: node ids, transfers, directed river and sea edges, weight = time + K x hazard (hazard 0 until T-006), seeding by layer, painting the sea; measure speed | T-019; T-020 and T-021 data (synthetic files meanwhile) |
| T-024 | Canoe hover: time by kind and the route line, traced across levels | T-022 |
| T-008 | Mode panel: walking or canoe start; later K, the other toggles and the era switch | T-022 |
| T-006 | Hazard rates: the water table first, then the land hazard band; then tune K on the check trips | T-022 |
