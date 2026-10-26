# One world: track, stations and trains anywhere on Earth (T-304)

T-304, 2026-10-14; approved by Anita the same day ("i think this is a good way to handle it":
track lengths are true ground lengths, nothing distorts with zoom). Anita, 2026-10-14: "one save
for whole world. at least thats exactly how it should feel to the player." It replaces the "one
world per region" idea in research/world-sampler.md (its open call 1) and closes T-126 and T-127.
Built so far: the track model (T-306) and the renderer anchor (T-307). Background: notes/T-125.md.

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
  over `k` when it runs east-west; `geom::rhumb_ground` and `rhumb_plane`, stable for any
  bearing), so it is exact at any length. An arc's ground length is Simpson's rule on `1/k` over
  its ends and middle; across a 10 km curve at London `k` varies by 0.1%, so that is centimetres.
  A piece maps ground offsets to the plane (`Piece::at` takes ground metres, `at_plane` plane
  ones); in the conic and legacy frames the map is the identity and every result is bit for bit
  what it was. Every `s`
  outside the geometry (edge length, profiles, capacity, stops, platforms, hold points, ramps,
  costs, water spans, the run polylines the renderer samples) is ground metres, as it is now;
  `pos_at` takes a ground `s`. So `profile.rs`, `capacity.rs`, `service.rs`, `boarding.rs` and
  `cost.rs` keep their arithmetic.
- **Metre figures into the plane.** `geom::fit_in` (through `Network::fit`) multiplies a
  player's radius, the auto radius cap (`R_CAP`), `MIN_RADIUS` and `FIT_TOL` by `k` where the
  arc is (the arc's middle lies up to a kilometre inside the corner, so `k` is taken there, not
  at the PI). Curve speed uses the ground radius (plane radius over `k`). `cross.rs` works in
  plane offsets and hands back ground chainage; `SHARED_NODE_CLEAR`, the drawing tool's lead-in
  cap, the stroke joint leg and quad-track spacing and taper are scaled by `k`. Station platforms
  and ramps are measured along `s`. Radii in the save are ground metres, the number the player
  typed. A plane circle is a ground circle up to the rhumb line's own ground curvature
  (`tan(lat)/R`, 2e-7 per metre at London): a 1,500 m curve there measures 1,500.4 m through three
  points on the ground, sub-metre.
- **Shown straight is built straight.** A straight edge is a rhumb line, which is what a straight
  line on the Mercator basemap is, and it is charged its true ground length. (New York to Chicago
  as one straight: 1,140 km along the rhumb line, 1,139 km by great circle. The track runs where it
  was drawn, so its length is the right one.)
- `tile_of` is arithmetic (it calls trigonometry today). The water mask (`GeoMask`) already goes
  through longitude and latitude and needs nothing.
- **The plane's edges.** Track is refused above 85 N and below 85 S, and no edge crosses the
  antimeridian (180 degrees): no railway does. A node or PI there is `Issue::OutsideWorld`
  ("Track cannot be built within 5 degrees of a pole or across the 180th meridian."). Decided,
  not open.
- **Running side.** `right_hand` is one flag per network and only `capacity.rs` reads it. Britain,
  France's main lines, India and Japan run on the left. It becomes a property of the place (the
  registry's region), read per junction. Decided; a small task of its own.

### Save format

- **TWT6**: TWT5 byte for byte, with `Merc` as the frame. The origin fields keep the game's home
  origin (New York), which the Mercator frame does not use but a conversion back to the conic
  does. Node positions are millimetre deltas in plane units as now; a jump from New York to London
  costs a few bytes once.
- **Old saves.** `save::convert_frame` moves a network into another frame through longitude and
  latitude, keeping every place on the ground, and takes `Merc` as a target
  (`TrackApi::convert_save_frame(bytes, 2)`). A conic (TWT5) network's edges change by the conic's
  scale: 0.0007% near New York, -0.04% at Boston. Anita's real New York save (legacy frame,
  all within the five boroughs) converts with all 1,647 edges and all 109 lines intact: track
  4,500.62 km to 4,498.63 km (-0.044%), construction value -0.02%, round trips -0.02% in all, the
  most-changed line 0.3%. A legacy save with Philadelphia track changes more there (about 1.2%),
  since the legacy frame is 1.2% short at Philadelphia. Money already spent, fares and schedules do
  not move; quotes and run times do. Anita decided to convert on load, correcting anything that
  moves substantially where that is easy; built in T-308: see "Stages" below.

### Demand packs and the market

No change. Every pack keeps its own equirectangular frame at its own origin, and all conversions
(`worldToCity`, `cityToWorld`, `worldToMarket`) go through longitude and latitude, so they work
for a Mercator game unchanged once `frame.ts` knows the kind. London's pack (T-288) is `market: -1`
like Toronto's. A Europe market later is a second market entry in the registry (research/
world-sampler.md, open call 4); the registry's single `market` becomes a list then, not before.

App code that compares world metres with a metre figure goes through `groundScale` or
`groundDist` (`game/coords.ts`, T-306): station complexes (`buildComplexes` takes the scale),
links' 1 km limit, the nearest station. `cityReach.ts` and the walks already measure in longitude
and latitude. Float32 arrays that hold world metres (`commuterView`'s `cellsWorld`) are millions
of metres in Mercator, too large for Float32, so they hold offsets from an anchor (next section,
T-307).

### Clock worker, renderer and pick

Built (T-307, T-317): `map/geo.ts` (`anchor()`, `anchorWanted`, `kAtY`), `ClockClient.followView`
and the "anchor" message, the renderer drawing each network at its own anchor, `u_psi0` for k in
the train shader. The demand views follow the anchor the same way (T-317): every disc set (the
city's bubbles, a nearby zone's, a selection's near and far ends, longer trips, the lines to a
selection) is built relative to the anchor in force and carries it; `DemandOverlay` draws each set
with its own anchor's matrix, so an old set stays put while its replacement is built; the hover
finders get the pointer as longitude and latitude (`discAt(discs, pointer, anchor)`); the bubble
workers keep each cell's longitude and latitude, are told the anchor with their cells and again
whenever `anchorVersion` (`map/anchorVersion.ts`, set by the clock client when the new state
arrives) changes, and every answer names its anchor. Disc sizes use k at the disc.

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

Each region runs on its own local time, as NIMBY Rails does (Anita, T-309): London's morning peak
comes five hours of game time before New York's, and the clock shows the local time where the
player is looking.

- **One clock, world time.** The game clock counts game seconds in the registry's
  `clockUtcOffset`, UTC-5 (New York's standard time), so every save made before local time reads
  the same and the Northeast and Toronto run exactly as before. Game day 1 is a Monday in that
  time (its date: Calendar, below); a region's local day is the clock's day at its local time.
- **A region's offset** is its registry entry's `utcOffset`: a whole hour of standard time, no
  daylight saving (a seasonal hour would move every period twice a year for nothing the game
  models). Half-hour offsets round to a whole hour so period boundaries stay on the hour; India's
  +5:30 is +5, which puts the game's 6-10 morning peak on Mumbai's real 6:30-10:30, nearer its late
  office peak than +6 would (5:30-9:30). A place is in the nearest registered zone whose centre is
  within 120 km (`centreWithin`, the same rule in `game/localTime.ts` and `sim/src/track/localtime.rs`,
  which has no outlines; a place's zone for anything else is by the coarse outlines, `cityWithin`);
  a place in no zone takes its longitude's nominal offset, `round(lon / 15)`, until its region is
  registered (London 0, but Paris also 0 until it has its +1 entry).
- **What runs on local time.** Demand's five periods are abstract (a solve has no clock), so
  demand follows from the rest: each line's service periods (`Lattice::shift`: its departures laid
  out on its local hours), its boarding account (times of day, the busiest hours, the riders' kind
  of day, "today" from its local midnight), and money (each local time's lines settle by their own
  period and day: fares from their own income per period, running costs at their own demand level).
- **A line runs on the local time of its first stop's region.** A line from New York to Toronto is
  on one offset either way; a London to Paris line would be on London's. Simple, and a line's
  timetable has one clock, as a real one does.
- **The view.** The top bar's time, day and demand level, the demand views' period and their
  "Today" kind of day are the local time at the view's centre (`viewShift`, updated as the map
  moves). A line's inspector and the Lines tab show each line's level in its own local time.
- **Not yet local** (T-321): the money ledger's days and the world market's longer trips between zones of different offsets (London - Paris,
  Tokyo - New York), which take no period rule yet.

### Calendar

- **Dates** (T-861, T-862; `game/days.ts`): a 365-day year with no leap day. Game day 1 is Monday
  January 27 (`DAY1_OF_YEAR`). Weekdays run on from the day number, so each game year starts a
  weekday later.
- **Holidays** (T-862, T-262; `game/holidays.ts`, `holidays.json` from `pipeline/holidays_json.py`
  and the Python `holidays` package): each country's national public holidays with the substitute
  days its law gives (the package's observed days: Korea's substitute days, Japan's
  substitute and citizens' holidays, a US Friday for a Saturday holiday), plus Japan's New Year
  holidays (Dec 31, Jan 2 and 3) and Korea's Workers' Day from the package's bank category. A game
  year takes the holidays of the real non-leap year whose January 1 is the same weekday (Monday
  2029, Tuesday 2030, Wednesday 2031, Thursday 2026, Friday 2027, Saturday 2033, Sunday 2034;
  year 0 is 2031), so a holiday tied to a weekday keeps it and Easter moves between years. A
  holiday is a whole day of its country's local time. The top bar marks those in countries the
  player has built in, one entry for a holiday several keep that date ([ui-panels.md](ui-panels.md)).
  **China's** days off and worked weekend days come from `pipeline/china_holidays.json` instead of
  the package (T-1157): the State Council's notice for 2026 (国务院办公厅关于2026年部分节假日安排的通知,
  2025-11-04: Spring Festival Feb 15-23, National Day Oct 1-7, worked Saturdays and Sundays), and for
  the other real years the usual pattern, marked `"source": "pattern"` in the file until that year's
  notice is out (2027's comes in November 2026: replace the entry by hand). The pattern
  (`china_holidays_pattern.py`): Spring Festival from the eve for 4 days, Labor Day 2, National Day 3,
  the others 1; a statutory day on a weekend gives the next weekday off; a Tuesday festival bridges
  the Monday before (Saturday worked), a Thursday one the Friday after (Sunday worked); the long
  breaks add up to two swapped weekdays, the first for the nearest worked weekend day before, the
  second after (Spring Festival to 9 days at most, National Day 8, Labor Day to 5). Hong Kong, Macau
  and Taiwan keep the package's.
- **Weekends** (T-262): each country's weekend days from the package: Saturday and Sunday in most;
  Friday and Saturday in 18 (Saudi Arabia, Egypt, Israel, Iraq, Jordan, Kuwait, Qatar, Oman,
  Bahrain, Algeria, Sudan, Syria, Yemen, Afghanistan, Bangladesh, Djibouti, Mauritania, the
  Maldives); Sunday alone in 7 (Hong Kong, Paraguay, DR Congo, Niger, Anguilla, Saint Kitts and
  Nevis, Tokelau); Friday and Sunday in Brunei; Friday alone in Iran. A two-day weekend's day of
  rest runs as a Sunday and the other day as a Saturday: Friday is the day of rest where the
  weekend holds it (Brunei's Sunday then runs as its Saturday), Saturday in Israel, else Sunday. A
  one-day weekend runs as a Sunday, the other day as a weekday (`weeks` in holidays.json). A
  weekend day a country works in place of a day off (the package's weekend workdays, Hungary's in
  2026; China's make-up working days from its own table) is a weekday.
- **A country's calendar decides each local day's kind** (T-872, T-262, `game/holidayCalendar.ts`
  `kindOfIn`): a holiday is a Sunday, except the few most people work through (in the US and its
  territories: Columbus Day, Veterans Day, Martin Luther King Jr. Day, Washington's Birthday,
  Juneteenth; `working` in holidays.json), which keep their weekday; a weekend day it works is a weekday; others take the
  country's week. A line's country is its first stop's zone's (by the coarse outlines, so a Yuen Long line keeps Hong
  Kong's holidays, not Shenzhen's: T-1265); a line outside
  every zone has none and keeps Saturday and Sunday. On such days the line's boarding account reads
  that kind's riders (`TrackApi` `set_line_holidays`: the clock worker sends each line's days whose
  kind is not the day number's, with their kind, from yesterday to 8 days ahead), and its fares,
  rides and points settle at that kind's rates: money groups split by local time and country.
  Running costs do not depend on the kind of day. The demand views' Today follows the view
  centre's country, the line inspector its line's. The demand solve is unchanged: Saturday and
  Sunday are worked out from the weekday's ([demand-timing.md](demand-timing.md)).

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

Each stage is its own task.

1. **London buildable in the same save as New York.** Two tasks, run in parallel.
   - 1a, T-306 (done 2026-10-14). Track model: the `Merc` frame (Rust and `frame.ts`), ground
     lengths per piece, metre figures through `k`, TWT6, `convert_frame` to `Merc`, new games in
     it (the registry's `frame.kind` is `merc`), the 85-degree and antimeridian refusals, and app
     code comparing world metres with metre figures through `groundScale` and `groundDist`
     (`game/coords.ts`: station complexes, links, nearest station). Checks:
     `sim/src/track/merc_tests.rs` (rhumb lengths exact against a numerical integral at five
     latitudes; the same track near New York in the conic and in Mercator within 0.01% in length,
     cost and run time, with the same crossings; New York and London in one TWT6 save, London's
     curve and straights against the great circle, the same shape at both places within 0.003%;
     the refusals; a conic save converted), every earlier sim test unchanged, and a headless
     browser check (`../artifacts/trainworld/notes-outputs/T-306-browser.json`): a new game is
     Mercator, New York and London lines of one ground shape measure and run the same, the save
     loads back, the real New York save still loads in its own frame with 4,500.617 km, 109 lines
     and riders.
   - 1b, T-307 and T-317. Renderer: the moving anchor in `geo.ts`, the clock worker's anchored draw
     buffers, every Float32 view cache following the anchor version, `MERC_K` per vertex; the
     demand views (T-317) the same, with bubble workers told the anchor.
   London's water and demand come with its pack (T-288); until then track there pays no water
   cost, and the top bar names no zone there and nothing loads for the view (T-318, done).
2. **Old saves**, T-308 (done 2026-10-14): the clock worker converts any save whose network is not
   TWT6 (`trackFrameKind` in `saveFile.ts`, `TrackApi.convert_save_frame`) once, as it loads from
   the autosave or a slot, or imports from a file, then writes the slot again in TWT6; the old
   bytes are kept under the hidden key `old-frame-<slot>`. Money, fleet, fares and the ledger are
   copied. Nothing needed correcting: the real NYC save (legacy frame) moves 0.3% at most on any line's
   run time, no line's trains needed change, and no curve is newly refused (the 586 geometry
   refusals that save already carries are the same before and after, so a set radius, which
   `fit_in` reads as ground metres, behaves as it did). The check is the ignored test
   `merc_tests::t308_real_save_conversion_report`.
3. **Local time**, T-309 (done 2026-10-14): periods, weekends, boarding hours and money per
   line's region, the clock showing the local time in view ("Local time" above). Checks: the
   real New York save runs bit-identically with the registry's time zones (ignored test
   `boarding::tests::t309_real_save_runs_as_before`, a week of trips, boarding slices and today's
   counts); `a_london_line_runs_on_londons_hours`; a browser check of a London and a New York
   line in one new game. The rest is T-321.
4. **Running side**: dropped (T-319, Anita 2026-10-14). Every network runs on the right, worldwide;
   `right_hand` stays one flag per network and only junction capacity reads it.
5. **World-scale loading**: water masks and boundaries for the zones on screen or reached, not
   every registered one (already T-302, folded into T-303's proposal).
6. **Globe** (T-033) on the Mercator frame.

## Anita's choices (2026-10-14)

1. Local time: each region runs on its own local time and the clock shows the local time in view,
   as NIMBY Rails does (T-309).
2. Old saves: converted to the world frame on load, correcting what moves substantially where that
   is easy (T-308; settles T-129).
