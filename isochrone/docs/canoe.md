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
| canoe | every sea, lake and river-water cell, plus a canoe node on each land cell a paddleable river or a strait passes through; at L3 and L10 one more node for each further piece of river or of sea or lake water in the cell that the cell's own does not join (section 4) | canoe start |

- **A canoe start is canoe + carrying; a walking start is walking alone.** You never leave the
  canoe, so plain walking cannot be reached from a canoe start, and a walking start never meets a
  canoe. Walking and carrying are the same land cells at a different speed, so each land cell has
  one land node whose speed factor is 1 or 2 by start mode.
- **Node ids.** Ids 0..N-1 are the cells: a land cell's id is its land node, a sea or lake cell's
  its canoe node. Canoe nodes on land cells (river cells, and at L3 and L10 coastal and strait
  cells) get ids from N on, found through a per-cell table `canoeOf` (-1 where none), then a
  cell's further nodes, river pieces and water units (`extra0`, `extraN`): 0.76 M and 0.44 M at L10.
- **Water** is the speed levels' water (our surface since T-028, `docs/resolution.md`: an L1 cell
  is water when LC100 has it half or more water, small water under 5 km² is land), read from
  `data/tiles` and the L10 grid, so canoe and walking agree cell for cell (T-085); a coarse cell is
  water when under a quarter of its L1 pixels are land. A walkable bridge (a road byte on a water
  cell) makes the cell land on the page but not in the canoe data: a land node over sea or river
  water, as a strait. Water is **sea** inside the ocean (Natural Earth 10 m ocean polygon for
  today, and water joined to it within 8 L1 pixels where NE's coast and ours differ), **lake**
  inside a HydroLAKES polygon, else **river water** (wide rivers LC100 has as water: the Amazon,
  the lower Danube). An L1 land pixel that NE's ocean touches keeps its land node and gets a sea
  canoe node too: the land mask closes some straits narrower than a cell (the Bosporus is cut
  where under half of a pixel is water), and this keeps them open. River water has no grid edges; the canoe uses only the
  river edges running through it, so a wide river is never a current-free channel beside its own
  current.
- **River water off the line** (T-036): a river-water cell no open river step crosses (the line
  runs in the next cell, the river's steps there are closed, a pond HydroLAKES lacks) is a dead
  end. Its depth is the number of 8-neighbour steps to the nearest land cell or other canoe node
  (1 when one is next to it). It is entered by getting in from the shore, from a canoe node beside
  it (depth 1), or from such a cell one depth less, at still-water speed with the flat-river
  hazard; it is left only by getting out to the shore, so only depth-1 cells lead anywhere. A sea
  or lake cell with no land, no water it is linked to, no river and no water link beside it (a
  one-pixel HydroLAKES pond inside river water, T-075) counts as river water off the line. No
  route paddles along it: a crossing goes in and out at the same cell (river water one cell wide),
  costing both transfers, so a chain of ponds is never a canoe route, and an unrunnable river's
  water is never paddled.

## 2. Edges and times

Paddling speed p = 5 km/h on still water. Clock: paddling hours, the same 8-hour day as walking.

| Edge | Between | Time | Directed |
|---|---|---|---|
| walk / carry | land node to its 16 neighbours | as walking (`docs/resolution.md`, "Directions"); x 2 when carrying | yes, by each cell's east, west, south and north speeds (T-011) |
| river down | canoe node to the next cell of a reach | length / (p + v) | yes |
| river up | the reverse | length / max(p - 0.4 v, 2.5 km/h); up rapids, length / 2.5 km/h | yes |
| lake | canoe node to neighbouring lake canoe nodes | length / p | no |
| sea | canoe node to neighbouring sea canoe nodes | length / s, s from the current | yes |
| transfer | land node to a canoe node in its own cell or one of its 8 neighbours | 0.25 h + half the step at each side's speed; the whole step carried when the canoe node is on a land cell (a river or strait node); within the cell at L3 and L10, the piece's access carried as well (a piece of river or of sea or lake) | both ways |

- **River speed** v = 0.51 Q^0.11 m/s from mean discharge Q, from hydraulic geometry W = 7.2 Q^0.5,
  D = 0.27 Q^0.39 (Moody and Troutman 2002). A reach is paddleable at Q >= 1 m³/s: a mean depth of
  0.27 m and width of 7 m, against a canoe's least 0.15 m and 1.2 m at a river's shallowest point
  (Hyra 1978; research/canoe.md 4). No depth limit beyond that, either way (T-073).
- **Hot desert** (T-079; the land hazard's mask, `docs/hazard.md` 2.1, one rule in
  `tools/hotdesert.py`): a reach with half its length or more in hot desert does not use its own
  WaterGAP discharge, which gives dry wadis hundreds of m³/s (Wadi ad-Dawasir 292, 98 in its driest
  month). Its Q is the flow carried in from upstream reaches outside the desert, mean and driest
  month, scaled down wherever WaterGAP's mean flow falls along the way and losing 3 cm a day over
  the wetted width W (evaporation and seepage) on every km. It is a river while that is at least
  1 m³/s mean and 0.1 m³/s in the driest month, and its speed v comes from that Q. WaterGAP also
  keeps rivers fed from wetter country running all year where they stop for months (it says the
  Thomson never runs dry), so a hot desert reach is dry, and carries nothing on, where GIRES
  (Messager et al. 2021) gives a probability of 0.6 or more that it stops flowing 30 days a year
  (`predprob30`, averaged over the reach and 25 km up and down its river, so one odd reach does
  not cut a big river; T-083, `tools/hotdesert.py`). So the Nile, Niger, Senegal, Euphrates,
  Tigris, Indus, Amu and Syr Darya, Colorado, Orange and Helmand run; wadis fed only by desert
  rain, small mountain streams that sink within a few hundred km, and the Channel Country's
  Cooper Creek, Diamantina and Georgina, the Finke and the Draa are left out. Outside hot desert
  every reach of Q >= 1 runs, seasonal ones too: the paddler picks the season.
- **Upstream** the canoe keeps to the bank, eddies and inside bends, where the current is slower
  than the section mean v: it meets 0.4 v. Where that leaves under 2.5 km/h (v over 6.25 km/h,
  Q over about 68,000 m³/s: only the Amazon from about Manaus down) it lines or poles along the
  bank at 2.5 km/h. So upstream on flat water runs from
  4.3 km/h on a small river (Q = 1 m³/s) to 3.8 on a big one (100) and 3.0 on a huge one (10,000),
  near what recorded upstream canoe travel made per moving hour (research/canoe.md 4).
- **Up rapids** (a step of class rapids, 1 m/km up to the closing gradient, or an OSM rapids on a
  river under 1000 m³/s) the canoe is poled, lined or waded at 2.5 km/h on any river: riffles too
  shallow and swift to paddle against are what made small rivers slow upstream, while their still
  meadow reaches were paddled (T-073, research/canoe.md 4). Downstream, rapids run at p + v.
- **River length** is the reach's real length spread over its cell steps, so meanders cost their
  length. RiverATLAS's `LENGTH_KM` is traced on a 15 arc-second network (HYDRO1k's 1 km north of
  60N) and loses meanders smaller than that, so each reach's length is `LENGTH_KM` x A x C (T-100,
  `rivers.py length_factor`). A is MERIT Hydro's 90 m flow length over RiverATLAS's, by Q and the
  reach's GMTED gradient: at Q = 3, 30, 300 and 3,000 m³/s, 1.23, 1.29, 1.18, 1.05 under 0.5 m/km,
  1.20, 1.22, 1.10, 1.05 at 0.5-2 m/km, 1.12, 1.10, 1.07, 1.05 from 2 m/km, times 1.17 and 1.28 at
  Q = 3 and 30 north of 60N (1 from Q = 300). C is what 90 m still misses, published river miles
  over MERIT's: 1.26 at Q = 4, 1.09 at 40, 0.96 from 300 (MERIT's own D8 steps on big rivers).
  Both are interpolated in log Q between those points and flat beyond. In the tropical forest
  tiles (section 4, gradient) a reach with a MERIT window takes its own MERIT length over that
  window x C instead of A x C. World-wide this makes Q 1-10 rivers ~1.5 x longer, 10-100 ~1.4 x,
  100-1000 ~1.15 x and bigger ones ~1.03 x. Hot desert water loss (above) still counts
  `LENGTH_KM`.
- **Sea current** c is the mean of the edge's two cells from the drifter climatology, x0.5 within
  5 km of shore. Along the edge direction d: s = c·d + sqrt(p² - (c×d)²); no edge when the
  cross-current exceeds p or s <= 0.
- **Ice**: sea cells with mean ice cover above 15% in the least-ice month (August north of the
  equator, February south) have no canoe node.
- **Moves on water**: 8 neighbours; the 8 knight moves only where all four cells involved are
  open water and linked along both two-step paths (walking needs all four on land); past 20 km
  from shore 8 neighbours only (section 7). At L3 and L10 a canoe edge joins two water units
  (section 4) only where their L1 water touches across the cells' shared border (`links` between
  the cells' own nodes, the listed water links between any others): a strait narrower than a
  cell (the Bosporus, the Danish straits, fjords) stays open at L10, two waters a few km apart
  are not joined, and neither are two lakes in one cell that an isthmus parts at L1. Knight
  moves at sea use the cells' own nodes only.
- **Portage has no edges of its own.** Unrunnable river steps are dropped: a fall, a dam, a weir,
  a reach steeper than its closing gradient (section 4), or OSM-mapped rapids where Q >= 1000. The search gets off (0.25 h), carries
  along the bank at walking x 2 and gets back on (0.25 h), about the research's "1 h plus the
  carry" for a fall. Watersheds between rivers are crossed the same way. Upstream on a big
  meandering river, carrying on a straighter line over open land or roads (2.4-2.5 km/h) can beat
  paddling the bends at 3 km/h; the search takes the quicker.
- **Rapids** from 1 m/km up to the closing gradient are run downstream at the same speed as flat
  water and poled upstream at 2.5 km/h, with their own hazard class both ways.

## 3. Time and danger

The search minimises **weight = time + K x hazard**, summed along the path, on every layer, with
one K. Hazard on an edge is its rate x real hours (research/canoe.md 5, friction.md 6); survival
along the chosen path is exp(-hazard sum). The painted time is the time of the chosen route, not
the fastest possible.

**K = 1000 h**: 10 hours (1.25 days of travel) to cut the death risk by 1%.

**Water hazard** (built, T-006 stage 1; `cfg.hz` in `js/config.js`). No new files: the worker
works it out from the canoe data when it builds a level's canoe layer. Rates are per real hour
(research/canoe.md 5); a paddling hour more than 20 km from shore counts as 3 real hours
(exposure: out there day and night), applied to every term below.

| Where | Hazard per real hour |
|---|---|
| river, flat (class 0) | 1e-5 |
| river, rapids (class 1) | 1e-4 |
| sea or lake, anywhere | 2e-5, plus waves and supplies |
| waves, by summer Hs: under 0.5 m / 0.5-1 / 1-1.5 / 1.5-2 / 2-3 / over 3 m | 0 / 2e-4 / 1e-3 / 3e-3 / 1e-2 / 3e-2, x min(d / 20 km, 1), x the cold factor |
| cold factor, by summer water temperature | x1 from 15 C, x2 at 5-15 C, x3 under 5 C |
| supplies, by distance from shore d | 0 to 250 km, rising linearly to 0.05 at 600 km and beyond |

- d is the cell's `shore` distance. At sea, Hs and temperature come from the sea grid at the
  cell. On a lake, Hs comes from the lake's `kind` code and the temperature from the sea grid's
  lake band there (T-043): ERA5's own summer temperature at the place, its SST where ERA5 has
  water there (on most lakes that is FLake's value; the Great Lakes read otherwise), else FLake's
  lake mix-layer temperature at that point; never a sea's from elsewhere. Every lake
  takes it, though only lakes over ~350 km² have Hs of 0.5 m or more, where the factor counts.
- A sea or lake edge's hazard is its time x the mean of its two cells' rates; a river step's is
  its time x its class rate. Carrying, walking and getting in or out pay the land band's.

**Land hazard** (`docs/hazard.md`): a band beside every level's speeds (uint8, a log-coded rate
per cell from thirst, cold, ice and altitude), paid at its rate x walking time; carrying the canoe
counts the same rate over its doubled time, and getting in or out 0.25 h at the land cell's rate.

**What K = 1000 does**: a canoe hugs coasts where a crossing is risky, short straits are crossed
as before, and the open ocean is not crossed. K never makes a place unreachable by itself; the
survival stop rule (section 5) does, in open ocean. The check trips and their figures are in
`notes/T-006.md`.

**A slider later.** K is one number in the worker; changing it re-solves (~0.3-0.8 s at L10),
with no new data. A slider in the mode panel, log steps from 0 ("fastest, whatever the risk")
to 10,000, would cost nothing extra (no task yet: TASKS.md "Someday").

## 4. Data per level

| File | Level | Holds | Size |
|---|---|---|---|
| `data/canoe/water.bin.gz` | L10, whole world | 3 bytes a cell: `kind`, `shore`, `links` | 1.1 MB |
| `data/canoe/l3/<r>_<c>.bin.gz` | L3, 5° tiles | the same 3 bands | 4.5 MB, 1122 tiles |
| `data/canoe/l1/<r>_<c>.bin.gz` | L1, 5° tiles | `kind`, `shore` (links implicit at L1) | 13.7 MB, 1140 tiles |
| `data/canoe/sea.bin.gz` | 0.25°, whole world | current U, V (int8, 0.02 m/s), summer wave height (uint8, 0.05 m), ice (uint8, %), sea and lake temperature (int8, °C) | 0.7 MB |
| `data/canoe/rivers.bin.gz` | L10, whole world | 1.25 M river steps, with slots and access | 6.8 MB |
| `data/canoe/rivers/l3/`, `l1/` | 5° tiles | 4.1 M (with slots and access) and 11.0 M river steps | 21.1 MB, 904 tiles; 40.2 MB, 911 tiles |
| `data/canoe/pieces.bin.gz` | L10, whole world | 222k water units, 65k water links (T-088) | 0.7 MB |
| `data/canoe/pieces/l3/` | L3, 5° tiles | 723k water units, 52k water links | 2.0 MB, 1058 tiles |

- Tiles share the speed tiles' grid and names (`docs/resolution.md`); `data/canoe/index.json`
  lists them, and the sea grid, and under `rivers` the river tiles (`{"1": {"tile", "path":
  "rivers/l1", "tiles"}, "3": ...}`, written by `tools/canoe/rivers.py`; a copy is
  `data/canoe/rivers/index.json`). A tile with no river steps is not written. A tile with no land and only sea is not written (the same tiles
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
- **links** (uint8, L3 and L10): bit k set when L1 sea or lake pixels of the two cells' own
  water units (slot 0, below) are 8-neighbours across their border (a diagonal: the two corner
  pixels), and both cells are sea or lake. `water.py` writes it cell to cell; `rivers.py build`
  rewrites it unit to unit, so it runs after `water.py`. k follows `js/search.js`'s moves as (dc, dr), dr down: 0 (1,0) E,
  1 (-1,0) W, 2 (0,1) S, 3 (0,-1) N, 4 (1,1) SE, 5 (1,-1) NE, 6 (-1,1) SW, 7 (-1,-1) NW. Bits are
  symmetric (a cell's bit k equals its neighbour's opposite bit) and wrap at 180°.
- **Sea grid**: 1440 x 580 cells of 0.25° from 180W, 85N. Local summer: June-August north of the
  equator, December-February south. Currents: the drifter monthly means over those months, else
  the annual mean, else 0 (40% of cells, mostly land). Wave height and sea temperature: the
  summer mean; ice: the August (north) or February (south) mean. Each is the mean of the four ERA5 points round the cell centre; cells with no ERA5 value (land, waves
  under ice) hold the nearest cell's, so a lookup anywhere returns a sea value. Lake
  temperature (section 3): the four-point SST where there is one, else the four-point summer mean
  of ERA5's lake mix-layer temperature, which ERA5 has at every point; nothing is filled from
  elsewhere.
- **River steps**: one record per downstream step, stored with the tile of the upstream cell.
  File: gzip of uint32 n, then columns cell (uint32[n]; row x 4320 + col at L10, local row x T +
  col in a tile), length (uint16[n], 10 m), direction to the next cell (uint8[n], 0-7 as `links`),
  v (uint8[n], 0.01 m/s), class (uint8[n]), and at L3 and L10 four more uint8 columns: the slot of
  each end (which canoe node of its cell, 0 the cell's own) and the access of each end's river
  piece (0.1 km); little-endian, sorted by cell then direction. The
  next cell may lie in a neighbouring tile or across 180°. Class: 0 flat, 1 rapids, 2
  unrunnable, 3 fall, 4 dam, 5 weir, 6 in a reservoir. A step holding several takes the first
  of fall, dam, weir, unrunnable, reservoir, rapids, flat: a dam goes before the gradient, since
  the DEM's drop at a dam is mostly the dam. The worker makes the upstream twin and drops the
  classes it closes, so a toggle (dams, era) needs no new data at L1; an opened dam, weir or
  reservoir step runs as flat. At L3 and L10 a closed L1 step inside a coarse cell parts two
  pieces and leaves no record, so opening dams there needs the coarse steps rebuilt with them open. Length and v, not time, are stored, so p stays a worker constant (a heavy
  dugout or a big crew later).
- **Rasterising** (L1). Each reach with Q >= 1 (in hot desert, each one that is a river by section 2) is cut into pieces of at most 0.002° and each piece
  goes to the cell of its midpoint; the reach's `LENGTH_KM` is spread over its pieces.
  A step holds the length of its upstream cell's run. A run back into a cell already on the reach
  folds the loop into that cell; a cell clipped only at a corner (under 0.3 of a cell's length)
  between two diagonal neighbours is folded into the one before, so the step goes diagonally.
  Consecutive cells are 8-neighbours. A reach's last cell runs on into its downstream reach's
  first step; where several reaches meet there, the one with the most discharge carries on. Two
  rivers on the same (cell, direction) keep one record: an open class first, then dam, weir or
  reservoir, then a closed one, the bigger river among equals. A step whose two cells are sea or lake water
  (`kind` 1 or 3+, so their grid edges join inflow to outflow) and half of whose length lies in a
  HydroLAKES lake is dropped; one with a river-water end is kept, as river water has no grid
  edges (T-085: dropping it cut river fragments off in the Sundarbans' channels); in a reservoir it is kept with class 6. On land cells the step is kept
  whatever it crosses.
- **L3 and L10 are the L1 river steps contracted** (T-082), so a coarse cell joins two rivers only
  where L1 does. A **piece** is a run of L1 river cells inside one coarse cell joined by open
  (flat or rapids) L1 steps. Each piece that an open step touches is a canoe node of its own,
  numbered per cell (its slot): pieces touching L1 sea or lake water share the cell's sea or lake
  node (slot 0); of the rest, the longest is the cell's own river node (slot 0) unless the cell has
  a sea or lake node, and the others get slots 1, 2, ... A coarse step is an L1 step between two
  coarse cells, from piece to piece, its class the L1 step's; its length is half of each piece's
  main river plus the step's (so a river's steps still add up to its length, less half a piece
  at its source and mouth) and its v the length-weighted mean over the same. A piece's main river
  is its L1 steps with v at least 0.9 x the piece's fastest (Q from ~0.4 x its largest), so the
  twigs of small tributaries joining a big river inside the cell are not paddled along it (T-100). One record per (cell,
  direction, slot pair), chosen as at L1. A piece's **access** is the mean straight distance from
  the coarse cell's L1 land pixels to the piece (0 on a sea or lake node). Getting in or out of a
  coarse river node within its own cell carries that whole distance at the cell's walking speed
  (`cfg.access`, 1.0): the land node stands for the whole cell, the river for a line through it.
  Measurements and why: notes/T-082.md, notes/T-092.md and, after T-100's main-river steps,
  notes/T-102.md.
- **Water units** (T-088, `tools/canoe/pieces.py` and `water_units` in `rivers.py`). A water
  piece is an 8-connected run of L1 sea or lake pixels inside one coarse cell. A cell is *split*
  when it holds two or more, or any while it is river water (kind 2). In a split cell a unit is a
  water piece with the river pieces that touch it (so a river joining two lakes inside the cell
  makes them one unit), or a dry river piece. The cell's own node (slot 0) is, in a sea or lake
  cell, the unit with the most pixels of the cell's kind; in a river-water cell the longest
  river, one touching water first. Other units with water or an open river take slots 1, 2, ...
  by size, water first. A cell that is not split keeps T-082's rule. A water unit's access is the
  mean straight distance from the cell's L1 land pixels to the unit's water (the least of its
  pieces' where a river joins several), carried at `cfg.access` like a river's.
- **Pieces files** (`pieces.bin.gz` at L10, `pieces/l3/<r>_<c>.bin.gz`, the `pieces` key of
  `index.json`): gzip of uint32 n, then cell (uint32[n]; local in a tile), slot, kind, shore and
  access (0.1 km) uint8[n]: the water units the bands do not give, which are those on further
  slots, slot 0 of a river-water cell, and every water unit on a land cell (for its access); then
  uint32 m, cell uint32[m], direction (0 E, 2 S, 4 SE, 5 NE), slot of the cell's unit, slot of the
  neighbour's unit, uint8[m]: the water links other than between two cells' own nodes, each once,
  stored with the tile of its first cell. A unit's kind is sea when it has at least as many sea as
  lake pixels, else its largest lake code; its shore the least over its pixels. Without the key
  the worker builds T-082's graph.
- **Gradient classes**: flat under 1 m/km, rapids from 1 m/km, unrunnable above the closing
  gradient: 5 m/km below Q = 1000 m³/s; from 1000 m³/s, 2.5 x sqrt(1000 / Q) m/km, at most 2 and
  at least 0.3 (a roughly constant stream power, about 100 W/m² over the width W: 2 m/km up to
  Q = 1560, 1 at 6,250, 0.4 at 39,000, the Congo below Kinshasa, and 0.3 from 69,000, the Amazon).
  A fall or an OSM rapids on a river of Q >= 1000 also closes the step (smaller rivers' OSM rapids
  are class rapids). The gradient is the channel's: a drop over the reach's real length (river
  length, section 2), so a meandering river is gentler than its valley. The drop is GMTED's
  (`data/surface/elev.tif`) over at least 10 km of RiverATLAS downstream of the reach's start,
  from the lowest of the 3 x 3 pixels round each reach end, made never to rise downstream
  (`rivers.py dem_gradient`, T-031: RiverATLAS `sgr_dk_rav` reads canyons as cliffs and short
  lowland reaches' 1 m steps as rapids). **In tropical forest** GMTED reads the canopy as drop, so
  in the 5-degree tiles with at least 20% of their land in LC100 closed evergreen broadleaf forest
  (132 tiles: Amazonia, the Congo basin, West Africa's forest, South-east Asia, New Guinea) the
  drop is MERIT Hydro's hydrologically adjusted elevation instead (T-101, `tools/canoe/h90.py`):
  from the reach's midpoint to the midpoint of the first downstream reach at least 10 km on, each
  snapped to the highest-accumulation MERIT cell within 360 m, kept when both are in one basin
  and Q over upstream area is within 4 x of the tile's median, else GMTED's.
- **Snapping** OSM points and reservoir pour points: to the nearest HydroRIVERS reach of any
  size, kept only if that reach has Q >= 1 and lies within half its width (W = 7.2 Q^0.5 m) plus
  0.5 km (falls, rapids), 0.3 km (dams, weirs; many are on ponds beside rivers) or 1 km (pour
  points); the piece of the reach nearest the point takes it.
- **Sources** (stage 1): RiverATLAS v1.0 (HydroRIVERS reaches with discharge, and driest-month
  discharge `dis_m3_pmn` for hot desert rivers; CC BY 4.0), GIRES v1.0 (Messager et al. 2021, no
  flow 30 days a year, hot desert reaches only; CC BY 4.0), CHELSA
  V2.1 for the hot desert mask (as the land hazard), HydroLAKES (local;
  `Lake_type` 2 = reservoir, its pour point = the dam), OSM `waterway=waterfall|dam|weir|rapids`
  and `natural=waterfall` (from QLever's OSM planet endpoint), GMTED2010 for the reach gradient,
  MERIT Hydro through Hydrography90m (Amatulli et al. 2022, CC BY-NC 4.0) for the length factor
  and the gradient in tropical forest, LC100 for those tiles, the Natural Earth 10 m ocean polygon (public domain), the GDP drifter monthly
  climatology, ARCO-ERA5 sampled to a local-summer mean of wave height, sea temperature and lake
  mix-layer temperature and a least-ice-month mean of ice,
  NSIDC median ice edges as a check. SWORD's obstruction flags are a later check (1.6 GB).

## 5. Search

- **One Dijkstra over the node ids** (`js/search.js`), over each node's out-edges: grid moves of
  its layer, its river steps, and transfers. Directed edges need only that a cost is always
  worked out from u to v. The queue is a bucket queue (T-093): buckets narrower than the
  lightest grid or transfer edge the level can have, so nodes leave it in the heap's order up to
  ties; river edges, some 0 km, may re-queue a node within its bucket. A walking start runs it too (land hazard, docs/hazard.md 4); with hazard off it runs
  a separate small loop (the old walking search), which the general one matches cell for cell.
- **Per level** (`js/canoe.js`, built when a level is loaded on a canoe start): `canoeOf` and the
  further nodes of a cell (`extra0`, `extraN`, from the steps' and the water units' slots), one 32-bit word a cell
  (water kind for grid edges, links, and flags: a land cell or a canoe node in the 3 x 3 round it,
  20 km or more from shore, river edges leave it), the current a cell (0.01 m/s, halved within
  5 km of shore), each river node's access, and the river steps as out-edges by node (each open
  step and its upstream twin, with its time). Further nodes have no grid moves of their own: their
  water links are edges in the same lists as river steps, both ways, coded as the grid move they
  stand for and costed as one (current at sea from each unit's own shore, still water on a lake,
  the hazard of each end's kind and shore; none to an iced sea unit). Steps of the closed classes are left out. A water cell the canoe data calls
  land (kind 0 where the speed is 0) is sea. Iced sea cells get no canoe node unless a river step
  ends there. Where the predecessor code leaves the node ambiguous (a cell with several canoe
  nodes), `canoeFrom` picks the one the search took, for routes, corridors and the check trips: for
  a grid move, a water link whose cost gives the node's weight exactly, else the cell's own node.
- **Arrays per node**: weight, time and hazard (Float32), and the move it was reached by (Uint8:
  1-16 a grid move of its own layer, 17-25 a transfer from the other layer at offset (dc, dr),
  26-33 a river step down and 34-41 up by move direction, 255 a window seed, 0 the start; the
  predecessor's cell is this cell minus the move). `cfg.hazard` is on for both starts; with it
  off, time is weight and both share one array.
- **K** is one worker constant, `cfg.K` = 1000 h.
- **Stop rule**: a node is not expanded once its time passes the 128-day limit or its survival
  falls below 1%. The search ends at the first weight past 128 days + K ln 100 (128 days while
  hazard is off).
- **Windows**: each layer's ring is seeded from the same layer of the coarser level (land from
  coarse land, canoe from the reached coarse canoe nodes round the point, the least weight of each
  cell's; every canoe node of a ring cell is seeded), with weight, time and
  hazard. A fine canoe node with no coarse canoe node near (a small river L10 lacks) is not
  seeded; the search reaches it inside the window by a transfer. Reuse and cancelling stay as in
  `docs/resolution.md`; caps in section 7.
- **Painting** a cell takes the least time of its nodes. On a canoe start sea and lakes are
  painted too, since where the canoe gets at sea is the answer; walking keeps water unpainted.
- **Start mode**: the mode buttons (walking, canoe) beside the `i` button; switching re-solves
  from the same start. The URL keeps it as `?mode=canoe`.

## 6. Hover and legend

- **Hover**: the time at once (the page's own cell times), then from the worker a second line
  with how it was spent, "paddled 19 h, carried 6 h" on a canoe start (a part under a minute is
  left out; getting in and out counts as carried), and "survival 97%" (rounded down) when `cfg.hazard`
  is on and the path's risk is 0.5% or more. A walking start shows the route and survival, no split.
- **Route**: a thin white line with a dark halo from the start to the pointer, solid where
  paddled or walked, dashed where carried. `js/route.js` traces it in the worker on request: from
  the hovered cell's node with the least time (as painted) in the finest solved level holding the
  point, back along the predecessor codes (section 5) to the start. At a window seed (`P_SEED`) it
  goes on in the next coarser solved level holding the point, from the reached node of the same
  layer in that cell (else the least weight in the 3 x 3 round it). Points inside straight runs
  of one kind are dropped. The page keeps one request in flight and sends the pointer's latest
  point when it returns; the worker answers between a solve's 30 ms slices, and none while a
  click's L10 solve is pending. The route is asked again when a window or a new start lands.
- **Legend**: the same bands, nothing new.

## 7. Speed and memory

Measurements: notes/T-022.md, notes/T-035.md (and the first estimates, notes/T-005.md).

- The 128-day limit bounds the work: at 5 km/h a canoe covers ~5,000 km, ~1.1 M nodes from a
  coast at L10, not the 5.3 M sea cells of the world. No coarser open-sea grid is used.
- **8 moves past 20 km from shore** (`cfg.sea8`), 16 nearer: a canoe click at L10 takes
  **~0.5-0.85 s** in Chrome (Jakarta, Lisbon, Cluj, Constanta the slowest), against ~0.2 s
  walking; 16 everywhere costs 10-25% more and takes Constanta and Cluj to ~1 s, the limit.
  About half the work is carrying over land and getting in and out, a sixth paddling at sea.
  Open-sea bands run up to 8% long on diagonals, and an open-water crossing on a diagonal can
  lose to the coast (Marseille to Barcelona 9.7 days, 8.2 with 16 moves). Water hazard
  makes no click slower: the survival stop rule leaves open ocean unexpanded, so an ocean-coast
  start does about half the work (0.5-0.7 M nodes against 1.1 M) and an inland or enclosed-sea
  start about the same (notes/T-006.md).
- **Window caps** on a canoe start: 1.5 M cells at L3 (3 M walking), 2 M at L1 as walking, each
  ~1.1-1.3 s at the cap. A 1400 x 900 view's windows take 0.1-0.35 s each.
- **Memory** at L10 on a canoe start: ~8.7 M nodes (0.44 M of them further nodes, T-082, T-088);
  building the canoe layer takes ~1.4 s once per session; the per-cell canoe tables ~130 MB, the
  search ~21 bytes a node while it runs (~175 MB) and 9 a node kept for seeding and tracing,
  against ~90 MB walking.

## 8. Known approximations

- Distributaries and braids are missing (one channel per delta); tides are ignored; river flow is
  the annual mean (no low season).
- WaterGAP has no glacier melt, so glacier-fed rivers in hot desert carry too little in from the
  mountains and sink: the lower Tarim, the Hotan and the Keriya are dry. Today the lower Tarim is
  dry below its reservoirs, but in 10,000 BC these run (T-012).
- The gradient is over 10 km: a cataract shorter than that is averaged with the water round it.
  Outside the tropical forest tiles it is GMTED's, a surface model; in boreal and temperate forest
  GMTED and MERIT agree (notes/T-100.md), but scattered tropical forest outside those tiles keeps
  some canopy drop.
- The length factor is a world average by Q and gradient; places differ round it: US and European
  rivers of Q 100-1000 are ~1.05 x RiverATLAS, tropical and West Siberian lowland ones 1.2-1.7 x.
  What 90 m still misses (C) rests on three US stretches (notes/T-100.md).
- A coarse canoe node on a land cell stands at the cell's centre for a transfer to or from a
  neighbouring cell, wherever its water lies in the cell, so L10 still finds chains of lakes and
  rivers a little cheaper than L1 does: ~10-15% fast round Jyvaskyla, New York and Utrecht
  (notes/T-088.md, T-092.md). On the Amazon (Coari) L10 runs ~15% slow, the floodplain start
  cell paying its mean water access; in fjord country (Sogndal) coarse walking crosses the fjord
  (T-007), which makes the canoe ~5% fast there where it would be ~13% slow.
- A window's ring seeds every canoe node of a cell from the coarse cell's best one, so at the
  ring a river or lake can take a neighbouring one's time.
- Window seams as for walking (`docs/resolution.md`), per layer.
- River water round an island that no river line comes near is unreached on a canoe start, and
  so is the island: depth counts from the island's shore too, so the water between the line and
  the island has no way down to it (25 cells in the Amazon L1 window at Santarem, T-075,
  notes/T-036.md).

## 9. 10,000 BC

The same model and files, built per era: `kind` from T-012's land mask (sea = below the era's sea
level and joined to the ocean; no canoe on ice), `shore` and `links` recomputed; river records
unchanged at L1, with dams and weirs open and the steps inside reservoirs used (reservoirs are gone
from that era's land mask), and L3 and L10 contracted again with them open (section 4). Exposed shelves get no rivers in stage 1 (paleo rivers are later). Today's
currents, waves and ice are reused.

## 10. Not built yet

- A slider for K and the other toggles in the mode panel, and the era switch (TASKS.md "Someday").
- The canoe at 100 m (T-050, `docs/fine.md`).
- 10,000 BC (T-012, section 9).
