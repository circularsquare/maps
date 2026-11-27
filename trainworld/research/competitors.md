# How the subway-building game and the track-drawing rail game work

Research report, 2026-10-08. **[DEV]** = the developer said it (site, docs, changelog,
devblog). **[PLAYER]** = player reports. **[UNVERIFIED]** = not confirmed.

## The subway-building game (the subway game's developer; beta Oct 2025, 1.0 Feb 2026, Steam Jul 2026)

### Demand

- **[DEV]** US homes and jobs from Census LODES on TIGER geography; UK from 2021 census
  workplace travel data on output areas. Homes are generated from that data, then "a
  distance-based gravity model" assigns each commuter a workplace. Departure times follow a
  realistic distribution. College students (federal data) and airport commuters (FAA data)
  are added separately.
  - Note: so the subway-building game's work destinations are gravity-assigned, i.e. what trainworld calls
    tier 1, not a literal OD matrix.
- **[DEV]** Every person is a pop (`size`, `residenceId`, `jobId`). Demand-point
  `residents`/`jobs` only size the bubbles. Missing driving times are estimated as
  straight-line x 1.3 at ~40 km/h.
- **[DEV]** Pop size 200; v1.0.0 raised it to 200 to cut pathfinding lag.
- **[DEV] Mode choice**: transit vs driving vs walking on perceived time. Weights: riding 1.0,
  walking 1.39, platform wait 1.37, waiting at home 0.4, congested driving 1.33, parking 1.6.
  Time valued by each commuter's income; fare vs fuel and parking. Varying incomes make
  ridership respond gradually.
- **[DEV] Pathfinding**: range-RAPTOR over the real timetable, every departure in a 30-minute
  window, with walks to, from and between stations.
- **[DEV] Constants**: walk 1 m/s straight-line (1.5 on real paths); max walk to/from a
  station 2700 s; max transfer walk 600 s; driving $0.65/km; parking $5 and 180 s; default
  fare $3; starting money $3B.
- **[DEV] Catchment**: per station type, a multiplier on a base 30-minute walk radius;
  transfer radius base 10 minutes.

### Scheduling and costs

- **[DEV]** Trains per hour per route; each train type has a `tphLimit`; grade crossings lower
  it; track capacity limits since v1.4; TPH by time of day since v1.7.
- **[UNVERIFIED]** the exact "high / medium / low demand" band names.
- **[DEV]** "Mediterranean" per-km prices. Track, station, car purchase, per-hour running,
  track and station maintenance. Shorter stations cheaper.
- **[PLAYER/wiki snippet]** Cost multipliers: deep bore 4.5x, tunnel 2.0x, cut-and-cover 1.0x,
  elevated 0.8x, at-grade 0.3x (cannot cross roads).
   (502 when fetched)

### Performance

- **[DEV] changelog fixes**: "15 minute lag spikes" removed by spreading commuter decisions
  over several minutes (v1.7.2); pathfinding rewrite ~35% on large maps (v1.5); "long freezes
  when drawing tracks or placing stations on large networks" (v1.4.5); slow cursor and track
  deletion with many tracks (v1.4.10). Implied causes: every pop re-deciding at once, the
  pathfinder, edit-time recomputation.
- **[PLAYER]** Laggy pan and zoom, slowdown at high speed, worse as the network grows,
  JavaScript-error crashes, poor on GPUs under 8 GB; some call demand "randomized".

### Tech

- **[DEV]** Electron, React, shadcn/ui, Recharts, MapLibre GL JS, online MVT tiles. Mods are
  JS via `new Function()`. Custom cities can use OSRM/Valhalla/GraphHopper for driving.
- **[DEV]** ~55 cities (US, UK, France, Ireland, Japan, Switzerland, NZ), one per map.

## The track-drawing rail game (the track-drawing game's developer)

### Demand (v1.12, 2024)

- **[DEV]** Pax spawn on map points sampled from a population texture (~15 m/px) by CDF. The
  destination CDF covers every other active tile (quadratic). Destination rate is proportional
  to destination population and distance, "never to network design". Exponential distance
  ramp, "not tunned". 168 hourly weekday curves; player POIs with own curves. Demand grows with
  available destinations up to a cap.
- **[DEV]** Catchments overlap; the station was picked at random within range at first, a
  best-path option came later.
- **[DEV] 2020 system**: local, regional and long-distance tiers; Voronoi + radius catchments;
  waiting pax "soft-instanced" as destination id + counter.

### Routing, signals, scale

- **[DEV]** Since v1.5, A* over fixed timetables; ~100x slower than v1.4; paths cannot be
  cached (stale within seconds). Only transfer stations as intermediate nodes; no wait over
  3 h.
- **[DEV]** Block and path signals; reservations as slices of track; signal checks
  single-threaded, eased by reading the previous frame's occupancy. Track pathfinding capped at
  100 km or 10x straight-line distance.
- **[DEV]** C++ engine on bgfx; 16 GB whole-world OSM vector tiles, quantised and
  pre-triangulated; float32 with shader tricks; SRTM and GlobCover. Buildings dropped because
  OSM coverage is uneven.
- **[DEV]** ~400k track segments and 10k stations is "at the limit". Editing a line flushes the
  track pathfinding cache.
- **[PLAYER]** Multiplayer saves tens of MB compressed. Overfull queues slow the sim to a crawl.
- No dev statement on off-screen simulation; all one global sim (inference).

### Criticisms

- **[PLAYER]** Shortest-time routing sends everyone to fast lines (wait 50 min for an HSR
  train that arrives 1 min earlier). After 1.12, local-line ridership fell 4-20x, 8-10x below
  real.
- **[DEV]** in 1.11 "every station was a firehose of pax"; 1.12 ties demand to destinations.

## Others

- **The 3D transport game [PLAYER/wiki]**: trips depend on destinations reachable in time and cost;
  wait time does not count, so frequency does not matter (a cited flaw); cars win unless
  transit is much faster. Its town "destinations" tab (share of trips on your network) is a
  good UI idea.
- The minimalist metro game, the city-builder, the indie rail game, the Soviet-era city sim: not researched.
