# One world: track, stations and trains anywhere on Earth (T-304)

T-304 stage 1, 2026-10-14. A design for Anita's review; nothing is built yet. Anita, 2026-10-14:
"one save for whole world. at least thats exactly how it should feel to the player." It replaces
the "one world per region" idea in research/world-sampler.md (its open call 1) and closes T-126 and
T-127 once built. Background: notes/T-125.md.

## What has to be true

- One save holds New York, London, Shanghai and Mumbai, with no seam the player can find. A line
  through the Channel Tunnel or from New York to Toronto is an ordinary line.
- Lengths, curve radii, speeds and costs stay right to about a metre anywhere, because they set run
  times, fleets and money.
- Track and trains draw centimetre-exact at the deepest zoom anywhere.
- No change to the demand packs: each keeps its own local frame, and the track model and the packs
  meet in longitude and latitude, as they do now.

Today none of this holds outside the Northeast. The world frame is a conic at New York, good to
0.4% within 5 degrees of latitude of 40.8 N: London is 10.7 degrees north, where its scale is
1.75% off (350 m on a 20 km line). The renderer's single origin is at New York and puts London's
vertices about 8,200 km (in view units) from it in Float32, which rounds them to half-metre
steps: one to two pixels of wobble at zoom 18-19.

## Recommendation

**One Mercator plane for the whole Earth, with lengths measured on the ground; a draw origin that
follows the view.**

1. **The track model stays planar, in one frame: spherical Mercator in metres** (the same sphere
   as today's frames, R = 6,371,008.8 m). Mercator is conformal everywhere, like the conic, so a
   circle in the plane is a circle on the ground and angles at junctions are true. Its scale
   depends only on latitude, `k = 1 / cos(lat)`, and the track model corrects for it: **every
   length along the track is a ground length**, and every metre figure (a radius, a platform, a
   ramp) is turned into plane units with `k` at that place. With that correction nothing depends
   on how far a place is from an origin, so there is no origin and no seam.
2. **The renderer draws relative to an anchor near the view.** Float32 offsets from the anchor stay
   small, so precision is the same everywhere. The anchor moves when the view travels far; that
   costs a rebuild of the draw buffers, which in Mercator is a subtraction, not trigonometry.

Mercator is also MapLibre's own plane, so the track's straight lines are straight on the basemap,
the render tiles are plain arithmetic, and the globe view (T-033) takes the same coordinates.

## Options weighed

| Option | For | Against |
|---|---|---|
| A. Local frames, one per region, city or zone; a node or edge belongs to one | Each frame is accurate with no scale correction | A line between two frames needs a frame of its own; crossings and junctions between frames need a common one; every id in the API gains a frame; the Channel Tunnel and New York-Toronto are seams. T-125 rejected this for two cities; it is worse for the world |
| B. Geographic or ECEF positions, with a local tangent plane per edge or per operation | Exact, no projection at all | Rewrites the planar geometry: every test between two edges (crossings, junctions, the grid) needs both in one tangent plane, and headings need turning between planes. Most of the track model's code changes |
| C. Per-tile or per-edge local coordinates | Small numbers everywhere | The same seam problem as A at every tile edge, many times over |
| D. One conformal plane for the world, ground lengths through the scale `k` (**chosen**, Mercator) | The track model stays one plane: grid, crossings, junctions, picking and saves unchanged in shape. Only the step that turns geometry into lengths, radii and costs learns `k` | `k` is 1.32 at New York, 1.61 at London, 2.0 at 60 N, so plane metres are not ground metres and that conversion must be done everywhere. Poles and the antimeridian are edges of the plane |
| Renderer: relative to eye (high and low Float32 halves per vertex) | Exact everywhere in one buffer | Doubles every position buffer and changes every shader |
| Renderer: anchor following the view (**chosen**) | One uniform; buffers as now | A rebuild when the anchor moves; every Float32 cache in view units has to follow it |

A conic with the same `k` correction would also work, but its scale grows fast away from its
latitude (south of the equator it is unusable), and its straight lines are curves on the Mercator
basemap. Mercator's `k` stays between 1.0 and 2.0 from 60 S to 60 N, where nearly all rail is.

## Design

### Track model (sim/src/track)

- `frame.rs` gets a third kind, `Merc`: `x = R lon`, `y = R ln tan(pi/4 + lat/2)`, scale
  `k = cosh(y / R)`. No origin: `x` and `y` are global, stored in f64 (a 1 mm step on a 20,000 km
  coordinate is far inside f64's precision). New games use it; `Equirect` and `Lcc` stay for old
  saves (below).
- **Ground metres along the track.** `geom::fit` and the pieces it makes stay in plane units, but
  each piece carries its ground length. A straight piece is a rhumb line, and its ground length
  has a closed form (`R` times the latitude change over the cosine of its bearing; plane length
  over `k` when it runs east-west), so it is exact at any length. An arc takes `k` at its
  midpoint; across a 10 km curve at London `k` varies by 0.1%, so that is centimetres. Every `s`
  outside the geometry (edge length, profiles, capacity, stops, platforms, hold points, ramps,
  costs, water spans, the run polylines the renderer samples) is ground metres, as it is now;
  `pos_at` takes a ground `s`. So `profile.rs`, `capacity.rs`, `service.rs`, `boarding.rs` and
  `cost.rs` keep their arithmetic.
- **Metre figures into the plane.** `fit` takes the scale at each vertex: a player's radius, the
  auto radius cap (`R_CAP`), `MIN_RADIUS` and `FIT_TOL` are multiplied by `k` there. Curve speed
  uses the ground radius (plane radius over `k`). `cross.rs` scales `SHARED_NODE_CLEAR`; station
  platforms and ramps are already measured along `s`. Radii in the save are ground metres, the
  number the player typed.
- **Shown straight is built straight.** A straight edge is a rhumb line, which is what a straight
  line on the Mercator basemap is, and it is charged its true ground length. (New York to Chicago
  as one straight: 1,140 km along the rhumb line, 1,139 km by great circle. The track runs where it
  was drawn, so its length is the right one.)
- `tile_of` is arithmetic (it calls trigonometry today). The water mask (`GeoMask`) already goes
  through longitude and latitude and needs nothing.
- **The plane's edges.** Track is refused above 85 N and below 85 S, and no edge crosses the
  antimeridian (180 degrees): no railway does. Decided, not open.
- **Running side.** `right_hand` is one flag per network and only `capacity.rs` reads it. Britain,
  France's main lines, India and Japan run on the left. It becomes a property of the place (the
  registry's region), read per junction. Decided; a small task of its own.

### Save format

- **TWT6**: TWT5 byte for byte, with `Merc` as the frame. The origin fields stay and are written as
  zero. Node positions are millimetre deltas in plane units as now; a jump from New York to London
  costs a few bytes once.
- **Old saves.** `save::convert_frame` already moves a network into another frame through
  longitude and latitude, keeping every place on the ground (round trip within 2.5 mm). Giving it
  the `Merc` target is the whole cost. TWT5 games (the conic) change by at most about 0.05% in
  length between Washington and Boston when converted, 0.13% at Toronto. Anita's legacy TWT4 save changes more: Philadelphia's track gets
  about 1.2% longer, Boston's 2.7%. Money already spent, fares and schedules do not move; quotes
  and run times do. Whether to convert on load is **open** (choice 2). A save left in its old frame
  keeps loading as it does now but can only build within the Northeast.

### Demand packs and the market

No change. Every pack keeps its own equirectangular frame at its own origin, and all conversions
(`worldToCity`, `cityToWorld`, `worldToMarket`) go through longitude and latitude, so they work
for a Mercator game unchanged once `frame.ts` knows the kind. London's pack (T-288) is `market: -1`
like Toronto's. A Europe market later is a second market entry in the registry (research/
world-sampler.md, open call 4); the registry's single `market` becomes a list then, not before.

Two places in the app measure distance in world units and need ground distance instead:
`cityReach.ts` (the 4.5 km reach of a station) and any Float32 array that holds world metres
(`commuterView`'s `cellsWorld`): in Mercator those are millions of metres, too large for Float32,
so they hold offsets from an anchor (next section).

### Clock worker, renderer and pick

- **One draw anchor, following the view.** `geo.ts`'s fixed `ORIGIN_LNGLAT` becomes the current
  anchor, with a version number. The main thread moves it when the view centre is more than about
  100 km of ground from it at zoom 10 or deeper (shallower zooms do not need it: a metre is under
  a pixel there). Every vertex on screen is then within a few hundred km of the anchor, which is
  centimetre-exact in Float32. The anchor snaps to a coarse grid so small pans never move it.
- **What follows the anchor.** The clock worker gets the anchor in a message and builds its draw
  buffers relative to it (`packLocal` becomes a subtraction and one constant multiply, from 174 ns
  of trigonometry per point today). An anchor change clears its draw caches like loading a game
  does, and sends a full state. Every main-thread Float32 cache in view units (demand discs,
  bubbles, the trip view, labels' positions, station shapes) rebuilds on the anchor's version,
  the way they already rebuild when the frame or the focused city changes (T-125). Workers never
  send anchored units; they send world positions or pack positions, and the main thread anchors
  them.
- **`MERC_K` stops being a constant.** Shaders and views that turn ground metres into view units
  (train and car lengths, disc sizes, catchments) use `k` at the vertex, from a `u_anchor_y`
  uniform plus the vertex's offset. Today a train at London would draw 18% short (1.32 against
  1.61).
- **Pick** works on the same anchored buffers with the camera matrix in f64, as now; nothing
  changes in its approach. Positions the player clicks go to the worker as longitude and latitude
  or world metres in f64.

### Globe view (T-033)

Mercator is MapLibre's input plane, so the globe needs our positions as Mercator 0..1, which is
the anchor plus the offset, scaled. MapLibre blends the globe into a flat Mercator view around
zoom 11-12; below that, Float32 Mercator 0..1 (about 2 m) is far below a pixel. So T-033 becomes:
at globe zooms, draw through MapLibre's `projectTile` with the whole-world tile; above them, the
anchored path. No precision trick is needed on the globe. Picking under the globe (pick.ts
assumes a flat affine camera) stays T-033's problem.

### Local time

One save across time zones raises a question the frame does not: London's morning peak comes five
hours before New York's. Today every city runs on one clock and nothing knows time zones (Toronto
shares New York's). Either every region's demand periods, service levels, weekend and boarding
hours follow its own local time while one game clock runs (shown in the focused region's time),
or the whole world shares one hour, so London's morning peak runs at New York's morning. This is
**open** (choice 1). Local time touches `periods.ts`, `days.ts`, the clock worker's hourly trip
lists and the demand workers' period mapping; Mumbai's +5:30 offset would be rounded to a whole
hour so period boundaries stay on the hour.

### NIMBY Rails

From research/nimby-rails.md (the developer's own posts; much of it is not public):

- **Supports this design.** NIMBY keeps one save for the whole Earth with no regions or chunks of
  track, stores positions in full-precision longitude and latitude, and draws them on the GPU as
  pairs of 32-bit floats "using vector shader tricks", with no 64-bit floats on the GPU. That is
  the same split as here: exact global positions on the CPU, small Float32 numbers on the GPU.
  Our Mercator metres are a fixed one-to-one function of longitude and latitude, so storing them
  is the same as storing longitude and latitude.
- **Not known.** How NIMBY gets its Float32 precision (an origin offset or a high and low split)
  and what plane its track geometry is laid out in are not published, so it neither confirms nor
  rules out Mercator with the `k` correction.
- **Determinism.** NIMBY saves only control points and rebuilds geometry, as we do. Once its load
  path and its play path rebuilt geometry differently (by about a micrometre) and train timings
  changed; the fix was one rebuild path for both. Stage 1a keeps that rule: `k` and the rhumb
  lengths are computed inside `derive`, the one path for load and edit, and the existing check
  that derived geometry hashes the same after a save and load covers it.
- **Long searches.** NIMBY caps its track route search at 100 km (or a multiple of the straight
  line) because an impossible request scanned every segment. Ours (`Network::find_path`, a line's
  path between two stops) has the same worst case, but it only reaches track connected to the
  start, and separate cities' networks are not connected, so it does not grow with the world. No
  cap now; worth one if continents get joined by track.

## Stages

Each stage is its own task, with ids given when Anita approves the design.

1. **London buildable in the same save as New York.** Two tasks, which can run in parallel; London
   is playable when both land.
   - 1a. Track model: the `Merc` frame (Rust and `frame.ts`), ground lengths per piece, metre
     figures through `k` in `fit` and `cross.rs`, TWT6, `convert_frame` to `Merc`, new games in
     it, the registry's world frame set to `Merc`, the 85-degree and antimeridian refusals.
     Checks: lengths against great circle and rhumb line at New York, London, Mumbai, Shanghai and
     60 N; curve speed at the right ground radius at those latitudes; a converted Northeast save
     against its TWT5 numbers; the T-125 fingerprint for TWT4 and TWT5 saves left unconverted;
     edit timings (`t125-edit-timing.mjs`).
   - 1b. Renderer: the moving anchor in `geo.ts`, the clock worker's anchored draw buffers, every
     Float32 view cache following the anchor version, `MERC_K` per vertex. Checks: headless
     screenshots at zoom 18 in New York and London with track, platforms and trains on the
     basemap's streets; panning New York to London at zoom 12; frame times unchanged.
   London's water and demand come with its pack (T-288); until then track there pays no water
   cost.
2. **Old saves** (after choice 2): convert TWT4 and TWT5 saves on load, or leave them. Small.
3. **Local time** (after choice 1): periods, weekends and boarding hours per region. Needed
   before London's demand (T-288) means much.
4. **Running side per region**: `right_hand` from the registry, read per junction. Small; before
   London's junctions carry traffic.
5. **World-scale loading**: water masks and boundaries for the zones on screen or reached, not
   every registered one (already T-302, folded into T-303's proposal).
6. **Globe** (T-033) on the Mercator frame.

## Open choices for Anita

1. **open** Local time: each region runs on its own local time under one game clock
   (recommended), or the whole world shares one hour?
2. **open** Old saves: convert your existing games to the world frame when they load
   (recommended: the conic saves move by about 0.05%, 0.13% at Toronto, the original New York save by 1.2% at
   Philadelphia and 2.7% at Boston, money unchanged), or leave them in their old frames for good,
   Northeast only? This settles T-129.
