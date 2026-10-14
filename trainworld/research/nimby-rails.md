# How NIMBY Rails handles a world-spanning network

Research for anitabuilder, 2026-10-14. NIMBY Rails is a C++ game by Carlos Carrasco (Weird
and Wry) on a map of the whole Earth, one save. Sources are the developer's devblog
(carloscarrasco.com), his 2023 retrospective, the Steam discussion forums, and the OSM wiki.
About 28 searches and fetches. Pages were read through a summarising fetch tool, so numbers below
are as that tool reported them; the two I would double check before relying on them are marked.

Tags: **[DEV]** the developer says so. **[PLAYER]** a player's claim or measurement. **[GUESS]**
my inference, not public.

Already covered elsewhere, not repeated in full: the threading and path cache details are in
`research/performance-factorio-nimby.md`, and the 1.12 demand design is summarised in
`research/competitors.md`. This file adds the map and coordinate side, the population data
source, the 1.13 quadtree change, and the player reasons demand disappoints.

Not found: no reddit (r/NIMBYRails) page came back from any search, so no reddit claims here.
`docs/world.md` (T-304) and `notes/T-303.md` did not exist when this was written; "What this
means" is tied to `research/world-sampler.md` (T-286/T-287) and the T-303/T-304 briefs as given.

## 1. One world

**Coordinates and precision**

- **[DEV]** Vector data is kept "in full precision longitude-latitude, not missing a single bit",
  and the renderer stores "the coordinates as a pair of 32 bit floats and using vector shader
  tricks", so no 64-bit floats on the GPU.
  https://carloscarrasco.com/nimby-rails-retrospective/
- **[DEV]** The same post says the tile data uses "6 bits quantization", "equivalent to having a
  max map resolution of 32 centimeters at the equator". The "6 bits" wording is odd (my
  arithmetic: 32 cm is about 2^27 steps around the equator, so it is not a 6-bit grid); read it as
  "fine quantization of lon/lat, 32 cm". Not verified further.
  https://carloscarrasco.com/nimby-rails-retrospective/
- **Not public**: the projection. Nothing says whether the map is drawn as Web Mercator, and no
  post discusses a floating origin, local frames, or jitter when zoomed far in. A search for it
  found nothing. [GUESS] lon/lat storage plus a shader view transform suggests the view is
  Mercator-like and the "tricks" are an origin offset done in the shader, but that is inference.
- **[DEV]** Tracks are saved only as control points plus parent links (parallel tracks, branch
  tips); the real geometry is generated from them. Rebuilding the same save on two machines once
  gave geometry differing by about 1e-6, "in the order of 1/1000 of a mm", which changed train
  timing. The fix was to use the slower exact rebuild path on load too. This is the only precision
  issue the developer has written about, and it was determinism, not rendering.
  https://carloscarrasco.com/page/32/ (2022-12)
- **[DEV]** The track pathfinder uses geodesic distance: it refuses paths over 100 km, or much
  longer than the straight-line geodesic between the endpoints (the longer limit wins). Reason: an
  impossible request makes it visit "every possible track segment", dropping sim speed below 10x.
  A long leg can be split by inserting a waypoint halfway. https://carloscarrasco.com/nimby-rails-may-2025/

**Tracks over long distances**

- **[DEV]** Nothing in public describes a special mechanism. Track is ordinary arcs and straight
  segments between control points; there is no chunking of track by region that anyone has
  written about. [GUESS] a single global database of segments with a geo index ("geo indexes and
  caches", mentioned but not described in https://carloscarrasco.com/page/21/).
- **[DEV]** Players do build continent-spanning networks. The 1.12 distance demand curve is
  explicitly built to "cover the full distance range" for "networks at continent and planet size".
  https://carloscarrasco.com/page/16/ (2024-04)
- **[DEV]** Scale limits: players report saves with about 400k track segments and 10k stations
  are "at the limit" (also noted in `research/competitors.md`; Steam thread
  https://steamcommunity.com/app/1134710/discussions/0/3190241086269986088, not re-fetched).

**Map tiles and data**

- **[DEV]** Source is raw OpenStreetMap PBF dumps cut by country. Sea polygons from a separate
  water-polygons set, SRTM elevation for low zoom, ESA GlobCover as a raster layer.
  https://carloscarrasco.com/nimby-rails-retrospective/
- **[DEV]** Output is one custom file ("NRM"), multi-LOD vector tiles with delta coding and
  quantisation, shapes already triangulated at build time. About 16 GB for the world, target
  was 20 GB; shipped so it works offline. Buildings were removed (size, patchy OSM coverage); the
  developer planned to generate them procedurally. https://carloscarrasco.com/nimby-rails-retrospective/
  and https://carloscarrasco.com/page/62/ (the demo map is 1/20 of the world file)
- **[DEV]** Build pipeline: OSM PBF into a libmdbx database (RAM need cut from 800 GB to about
  90 GB), then a flatbush-style spatial index, then a tile-cutting pass that takes a region, an LOD
  range and a filter. Run on a rented 32-core, 128 GB server with a very large swap file.
  https://carloscarrasco.com/page/47/ (2021-09). The player never builds tiles; it ships prebuilt.
- **[DEV]** Admin regions (OSM levels 6 to 11) are a layer; a station is assigned a city from
  level 8, falling back to smaller or larger levels. Same page.
- **[DEV]** Rendering is custom C++ on bgfx. Roads, rivers, and region limits are drawn as
  instanced 10-vertex line segments (about 6x less memory than duplicating vertices).
  https://carloscarrasco.com/page/47/
- **Not public**: how tiles are loaded as the camera moves (cache size, threads, prefetch). The
  map file is local, so it is a disk read, not a network fetch.

## 2. Demand worldwide

**Where population comes from**

- **[DEV]** Since v1.3 (2021-08) the population layer is real data: GHSL population (ghs_pop2019)
  at 250 m, combined with GHSL built-up (ghs_bu2019) at 30 m, merged into one layer. The 30 m
  built-up data was cut to one bit per pixel and compresses to about 200 MB (it would have been
  4 GB+). It replaced a first version that counted OSM road nodes in a grid.
  https://carloscarrasco.com/page/48/ (2021-08). The Oct 2020 post called that first density map
  "completely fake" (seen only in a search snippet, not fetched:
  https://carloscarrasco.com/nimby-rails-october-2020/).
- **[DEV]** In the map tiles the top level is a 320 x 320 texture per tile, combined
  buildings and population, about 15 m per pixel at the equator. https://carloscarrasco.com/page/18/
  (2024-02). Whether finer-than-250 m values are real detail or the built-up layer spreading the
  250 m population is not stated. [GUESS] it is the latter.
- **Not public**: any country rescaling. GHSL totals are used as they come, with no census
  correction mentioned. Jobs do not exist as a separate layer; there is no workplace population.
  (Players use points of interest, "POIs", with a made-up population to fake workplaces.)

**How trips are generated and assigned (v1.12, Feb to Apr 2024)**

- **[DEV]** Origins: each pixel of an "activated" tile (any tile touched by a station's catch area)
  is sampled by its share of the tile's population, through a cumulative distribution (CDF).
  Spawn positions are points, not stations. https://carloscarrasco.com/page/18/
- **[DEV]** Destinations: weighted by destination population and distance, "never to network
  design". Demand scales with how many destinations are reachable, so it "wakes up" as the
  network grows. The first version built a CDF from every activated tile to every other, which is
  quadratic memory (about 1 GB at 10k+ tiles). https://carloscarrasco.com/page/18/
- **[DEV]** Each pax then picks one of the stations whose catch area covers the point, at random at
  first, with an optional "best path" choice later (50x to 100x slower than random, brought down to
  2 to 3x with pruning). Catch areas overlap instead of Voronoi cells; radius 3 km at 1.12 launch,
  2.3 km after beta. https://carloscarrasco.com/page/17/ (2024-03), https://carloscarrasco.com/page/15/ (2024-05)
- **[DEV]** Distance weighting is a smooth curve (exponential ramp-down with a long tail) replacing
  the old three fixed buckets (under 10 km, 10 to 100 km, over 100 km). Time weighting is 168 hourly
  values per week, with separate origin and destination curves; players can define several curves
  and POIs. https://carloscarrasco.com/page/17/
- **[DEV]** 1.13 (2024-05) went back to a hybrid: demand redistributes when new destinations open
  (the 1.11 behaviour) and the overall rate rises "following a logarithm curve". A "demand tile
  rebuild" precomputes tile to tile relations, with quadratic cost that grows with activated tiles.
  The quadratic table was replaced by a quadtree where each tile groups distant tiles (fixed-size
  groups, then adaptive by population). RAM and CPU dropped "dramatically"; distance weighting
  became less precise because population is drilled down but distance is not.
  https://carloscarrasco.com/page/15/
- **Before 1.12 [DEV]** (2020): destination chosen from city populations, 90% from a normal
  distribution over reachable cities sorted by population, 10% random, pair demand scaled by both
  populations; the developer called city data "spotty" and city boundaries "coarsely
  geolocated". https://carloscarrasco.com/page/55/

**What players say**

Good:
- The 1.12 idea is called "absolutely incredible" by one player, and "much more realistic" than
  1.11 by the developer, because expanding a network creates demand elsewhere. Overlapping catch
  areas and adjacent stops now work. adlet found global demand 600% with 50% time demand gave about
  120% passengers-to-population, "a good number for a city that relies heavily on transit".
  https://steamcommunity.com/app/1134710/discussions/0/4352242595078331742 and
  https://steamcommunity.com/app/1134710/discussions/0/4335356121958516692

Bad (the reasons for Anita's dissatisfaction):
- **Ridership collapsed and feels empty.** After 1.12, one player went from about 100k to 20k
  weekday riders; others report stations with "two to three people at a time" and "passengers more
  or less doesn't seem to exist". The developer's own words: in 1.11 "every station was a firehose
  of pax"; 1.12 gives small or low-population saves less demand and offers no real aid for them
  (Steam thread above).
- **Demand does not respond to the network's quality.** [DEV] flows depend on origin and
  destination population and distance, "never to network design" (page/18). Frequency, speed, and
  fares change which path a passenger takes, not whether the trip exists. [GUESS] this is why
  trains feel like they run on a fixed schedule of crowds rather than on a market.
- **One global dial.** Only one global demand scale exists, "an option to double the demand for the
  entire simulation" (AngryDragon, thread 4335356121958516692). Per-station tuning is a POI
  workaround. Players need 600% to 900% to get believable numbers.
- **Opaque and complex.** "I would have settled for a simpler demand scheme" (same thread). Players
  reverse-engineer it by experiment; one estimate is population served x global demand x 0.25 =
  weekday passengers (adlet, a [PLAYER] rule of thumb, not official;
  https://steamcommunity.com/app/1134710/discussions/0/4417550166393249492/). The "population
  served" is the sum over all loaded tiles near stations, so adding one station raised it by 1.1
  million against the station's own 295,000.
- **Imbalance on long distance.** Inbound trains into London and New York overcrowd while outbound
  runs near empty, "even after matching destination demand to origin demand" (Spartyon). Stadium
  trips: departures far exceed arrivals because "your stadium is just one of the possible
  destinations" (TKR, thread 4335356121958516692).
- **No workplace layer.** Commutes depend on the player faking jobs with POIs; the population
  layer is residential and built-up area, so rush hours are weak without POIs.
- **Shortest-time routing** sends everyone to fast lines regardless of waiting (in
  `research/competitors.md`).
- **Bugs after the rewrite.** "A bug which massively reduced demand after any editing was done to
  curves or stations" was fixed in 1.12.3 (developer, thread 4352242595078331742).
- Overcounting is also reported: a Dutch network at 100% demand showed about twice the real
  boardings (Thepolwoman, thread 4417550166393249492).

## 3. Performance at world scale

Detail is in `research/performance-factorio-nimby.md`. New or sharpened here:

- **Everything is one global simulation.** No source describes simulating only the area in view.
  [GUESS] the whole save runs; the camera only decides what is drawn.
- **Activated tiles are the unit of load.** Only tiles touched by a station's catch area are
  simulated for passenger spawning, at "on the order of 100 MB per 1000 activated tiles", kept in
  RAM permanently. Saves with 5k+ stations or 10k+ activated tiles were the 1.12 problem cases.
  https://carloscarrasco.com/page/18/ and https://carloscarrasco.com/page/15/
- **Precomputed up front:** the tile to tile demand relations ("demand tile rebuild", redone
  when stations are edited), the 168-hour curves, the braking speed functions along each path, the
  built tiles (triangles, LODs). **On demand:** passenger paths (cached; cache holds only the
  currently dominant path; high train frequency makes it expire fast), best-path station choice,
  heavy passenger detail for UI windows. https://carloscarrasco.com/page/15/, /page/16/, /page/21/,
  /page/28/
- **Hot against cold state.** Train motion data is about 1 MB per 1000 trains and copied to the UI
  thread every frame; the database (100 MB+ in large saves) is edited only through commands.
  https://carloscarrasco.com/page/21/
- **Benchmark:** 5 sim frames inside a 6 ms graphical frame with 600 trains and 70k+ riders; 1000+
  passenger unloads in one frame caused spikes, so unloading is capped at 5 destinations per train
  per frame. https://carloscarrasco.com/page/28/
- **Caps against pathological input:** track pathfinding limited to 100 km or a multiple of the
  straight line (May 2025 post above).
- **Not public:** how much of the 16 GB map is held in memory, and the load policy as the player
  pans.

## 4. Other points relevant to a browser world game

- **Multiplayer and determinism** [DEV]: a fully shared deterministic simulation failed; Steam
  Relay allows about 128 KB/s per client. The shipped answer is a "private simulation" mode: only
  edited objects (tracks, lines) and train existence are shared, about 20 KB/s per client, while
  train motion and passengers run locally. https://carloscarrasco.com/page/32/ Relevant if
  anitabuilder ever shares worlds: sync the plan, not the simulation.
- **Buildings dropped** because OSM coverage is uneven; the same unevenness will show up as thin
  towns in tier 0 data (also in `research/world-sampler.md` section 4, Shanghai).
- **Player-made POIs** as a cheap way to tune demand is an admission that a global raster alone
  does not give jobs, events or destinations.
- **Admin-region layer** gives stations a city name and a place to attach statistics, from OSM
  levels 6 to 11. https://carloscarrasco.com/page/47/
- **Hand tuned curves** are the developer's own description of the time-of-day shapes
  (Oct 2020 post; snippet only).

## What this means for anitabuilder

Tied to the three current pieces of work. `docs/world.md` and `notes/T-303.md` were not on disk
yet, so these bullets assume only what the briefs and `research/world-sampler.md` say.

**docs/world.md (T-304, the one-world design)**

- NIMBY chose to store everything in lon/lat and project in the shader. Our current plan (world-
  sampler section 3) is one conic per region and one renderer origin per world, which is what the
  developer avoided, and is the cheaper path in a browser, where the GPU works in 32-bit floats
  too. If one save must hold London and Shanghai at once, NIMBY's route (global lon/lat, view
  transform with an origin offset done per frame in JS or the shader) is the proven one; a conic
  per region is only safe while each save stays inside one conic. The developer never published how
  he handles cross-world distances, so the design has to decide it.
- Store track as control points plus parent links and rebuild geometry on load, as NIMBY does,
  and make the rebuild path identical on load and in play: the 1e-6 mismatch there broke
  determinism. Worth a line in world.md if saves ever need to replay.
- Do not rely on a path search that can scan every segment: cap by distance (100 km or a multiple
  of straight line) and insert waypoints on long legs, as NIMBY did in 2025.
- Quantised, delta-coded, pre-triangulated tiles at 32 cm precision are enough for track; a
  browser world does not need more than that.

**T-303 (demand loading and responsiveness)**

- NIMBY's unit of cost is the "activated tile": only tiles near stations are simulated and held
  in RAM (about 100 MB per 1000). Our zone pack per city is the same idea at a coarser grain;
  loading demand lazily for the zones a line touches matches this.
- Its quadratic tile to tile table hit 1 GB at 10k tiles and had to become a quadtree (nearby
  detail, far tiles grouped). That is the shape for our outside-world demand too: exact inside the
  region, grouped beyond it, recomputed on station edits.
- Precompute the relations between tiles on edit, keep per-trip path choice on demand and cached,
  and expect the cache to be defeated by frequent service, as it was for him.
- Our demand responds to network quality (mode choice, speed, frequency); NIMBY's deliberately
  does not. His players hit the downside (empty trains, one global dial, no jobs). Keep that
  difference and surface per-zone scale controls, since his single global multiplier was a top
  complaint.

**research/world-sampler.md (T-287, tier 0 city adapter)**

- NIMBY's population layer is GHSL 250 m, no census rescaling and no jobs layer. Tier 0's plan
  (GHSL or WorldPop scaled to admin totals, plus a job proxy from GHSL non-residential and OSM)
  already goes beyond it on both counts, and those two gaps (totals and jobs) are exactly what his
  players complain about: unrealistic volumes needing 600 to 900% scaling, and weak commutes
  without hand-placed workplaces. Keep both in tier 0.
- Players had to invent POIs to get workplaces and events; tier 0's OSM destinations and
  schools/airports/universities are the equivalent, done from data.
- His 15 m population pixels over 250 m data look finer than they are [GUESS]; do not take that as
  a sign 100 m GHSL/WorldPop is too coarse for city cells.
- Calibration is the lesson from his 1.12 rewrite: demand volumes lost 4 to 20x on local lines
  overnight because nothing checked them against real ridership. Tier 0's plan to compare against
  LODES or StatCan on a rebuilt pack is the right check; add real ridership counts for a few
  systems (a 1.12 player compared a Dutch network to real boardings and found 2x over).
- Not found: how he handles non-urban or low-population saves; he says only that low-population
  saves get too little demand and "some kind of aid" is planned.
