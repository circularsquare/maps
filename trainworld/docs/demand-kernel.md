[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: the kernel (4.4)

How the commute and everyday trips are computed: gravity between zones, access, strategies, mode choice, loading and crowding.

### 4.4 Cost of computing it

Code in `sim/src/demand/`. Measurements: notes/T-005.md, T-019.md, T-020.md and
notes/demand-measurements.md; how this design was reached: notes/demand-kernel-history.md.

**Commuting does not enumerate cell pairs** (T-005: there are billions). Demand works on two
levels:

1. **Gravity between 2 km square zones, once per city.** Doubly constrained, commuting residents to
   commuting jobs (4.1), decay by distance: **f(d) = d^-0.8 x exp(-d / 30 km), x k between zones in
   different first-level regions** (the US packs' curve, carried in each pack's gravity block). The regions are US states (a cell's state is the one holding
   most of its census blocks' people and jobs); a square a state line cuts is one zone per state,
   both at the square's position. k is the pack's `border_factor`: **0.23 for the US packs**,
   fitted with the decay length on New York and Philadelphia LODES OD (T-162, T-198), and **0.45
   for Washington**, its own fit, since DC is a city counted as a state (T-181; `build_city.py`
   `CITIES[...]["border_factor"]` overrides `BORDER_FACTOR_US`). It stands for everything that
   keeps people working in their own state (separate labour markets, residency rules, tolls;
   T-159). A pack with no factor has 1. The fits and the alternatives tested (T-171): notes/T-159.md,
   T-162.md, T-171.md, T-181.md, T-198.md.

   **Borders outside the US** (Anita, 2026-10-12, T-181; reasons in notes/T-181.md). National
   borders get a factor; inside the Schengen area it is to be weaker (closer to 1) than across
   other national borders (not built: every national border takes `national_default` 0.05,
   [demand-worldwide.md](demand-worldwide.md) "Commutes between zones"). First-level regions
   (states, provinces, prefectures, Länder) inside a pack get none, factor 1, by default; a country
   whose own OD data shows an admin-1 effect can fit one, on its own commuting data when its first
   pack is built. Between packs the joint gravity takes its factors from `pipeline/borders.json`:
   the pack's own where one pack holds both regions, else the country's `admin1_default` (US 0.23,
   Japan 0.5), else `pack_line_default` 0.5 (T-584, T-1163), and across a national border
   `national` for the pair or `national_default`.

   **4 km blocks in thin country** (T-852, `zone_size`). A pack may join an aligned 2x2 block of
   squares (even `gx`, `gy` at its lower-left) into one zone when at least two of its zones of one
   region lie there and each makes under 1,000 trip ends a weekday (`kernel::coarsen`); towns stay
   at 2 km. A pair with a block takes the mean of the decay between their squares, so a block draws
   as its four squares would with its homes and jobs spread evenly over them; distances (the drive,
   the walk, trip length, distance bands, the ride's minimum) are between centres, a block's own
   pair 0.52 of its 4 km side apart, and its land cap is its 16 km². The everyday destinations'
   decay takes the same mean. The decay table holds three runs per row (two squares, a square and a
   block at their centres' half-square offset, two blocks; `kernel::table_layout`), three times a
   2 km pack's; the distance and walk's cost of a pair come from its table index
   (`Gravity::dist_km`), in every pack. A station's zone is the square's or the block covering it; near pairs (the walk
   between cells, walking with no rail) are those with squares within reach of each other; the
   routes' windows go by each zone's lower-left square. A pack without `zone_size` is all squares.

   **The curve** (T-198, from T-185's diagnosis): d^-0.8 exp(-d / 30 km), one value for every US
   pack, flatter at the near end so residents of job-rich places commute out to suburban jobs as
   LODES has them, undershooting as Anita asked (how it was fitted and its effect per pack:
   notes/T-185.md, notes/T-198.md). It does not depend on the network, so edits never rerun it.
   **The flat near end** (T-1091, Anita 2026-10-23): every other curve (fitted on borough,
   municipality or commune pairs, or on distance bands, or borrowed from those) is held flat inside
   2 km, f(d) = f(max(d, 2 km)) (`kernel::Curve`, `FLAT_NEAR_KM`), so a job in the home square (its
   own pair 1.04 km) weighs as one in the next square; pairs 2 km apart and more are as before. The
   US curve, fitted on LODES block pairs, keeps its near end (`kernel::flat_near_km`: none on the
   US curve, 2 km on any other). A pack carries its value in its gravity block (`decay_flat_km`;
   absent, none), and every reader of the curve takes it from there: the runtime's table
   (`DemandApi::new` copies it into `Params::decay_flat_km`; `kernel::gravity` with no shipped
   factors uses the same), the joint solve and `read_pack.check_gravity`. A pack is solved with
   today's rule when it is built (`pack_gravity`, `coarse_zones.py`) or joined (`joint_gravity`, for
   the packs it balances) or re-solved on its own zones (`pack_gravity --keep-zones`, for a pack
   that joins nothing), so packs solved before keep theirs until then. A 4 km block's pairs take
   the mean over their squares of the flat curve, so a block's own pair is 3/4 at 2 km and 1/4 on
   the diagonal. The pull cap is measured against the pack's own pulls under the same curve.
   The pipeline solves it (Furness, the sim crate's own code run natively) and ships a zone id per
   cell and two factors per zone in the pack (notes/T-004.md); opening a city rebuilds zone trips
   from them rather than solving. **The pull cap** (T-1000, `kernel::PULL_CAP`): every worker
   commutes (rows exact), and each zone's jobs are filled from homes, but a zone's pull (its column
   factor over its jobs: how much harder its jobs draw workers than their count and distance alone)
   is held within 20 times the pack's job-weighted geometric mean pull, either way. A zone whose jobs
   would need more pull keeps some unfilled; one with more workers than jobs nearby takes the
   surplus anyway. The zones inside the band share what the held ones leave or take over, all at
   one fill of their jobs (the header's `gravity.pull`: band, fill, share of jobs held). On China's
   countryside curve a pull of 20 is worth bringing the jobs about 50 km closer, so jobs past that
   draw no one in from the far side of a desert; ordinary places never reach the cap. Zone trips spread to cells by commuters at each end. **Joint
   gravity** (T-276, docs/zones.md 4): the shipped factors are those of one solve over every zone
   pack (`bin/joint_gravity.rs`, run by pipeline/intercity.py), so a pack's zone trips are the part
   of each commute that stays in it; `zone_home` and `zone_work` (per zone, the trips kept) let
   cells count their commuters per zone (`DemandApi::home_k`, `work_k`) instead of the pack's total
   over its people. Each pack's homes send as many commuters as it has jobs, scaled per pack as
   alone (scaled to the union, packs with net commuting from outside every zone would balance
   through their neighbours). The balance (T-297) runs per group of packs joined by any pair within
   the cut, to a tolerance on every column (rows exact) with Anderson mixing, the pull capped against
   each pack's own mean and each pack's fill its own, so a pack's net exchange stays nil. The joint
   solve also adds the long commutes (docs/zones.md 4): 1% of every zone's workers to jobs 50-150 km
   away, which the zone trips here leave out.
2. **Access per cell**: the 6 nearest stations by walk within the walk's 4 km bound (4.3), found by
   scattering each station over the cells near it. With routes (T-275) each station's walk is
   searched once per position, every station once per worker on opening; an edit searches only
   its new or moved stations (timings: notes/T-275.md).
3. **Access subzones**: the cells of one zone that share their nearest station and whose walk to it
   falls in the same 15-minute band (`band_min`, Anita, T-720; why: notes/T-720.md). Each carries
   its perceived walk to each of its stations, averaged over its cells as the mode choice weighs
   them (a logsum at its 0.10 scale: the walk's cost grows faster than its length, and the near
   cells carry most of the rail), used at both ends of a trip. Cells with no station within 4 km
   are in no subzone: their commuters drive or walk. Access and subzones together are rebuilt on
   every edit (counts and timings: notes/T-720.md, notes/T-090.md).
4. **Station-to-station times**: one reverse strategy search per destination station over station
   nodes plus one node per stop per direction per line (T-069). On a circle line (T-798,
   `Network::line_wrap`) each direction's last node rides on to its first, costing the stand at
   the first stop, and nobody boards at the last: a rider rides through the first stop the short
   way round without changing. At each station, useful lines
   share their departure frequency: boarding costs the wait (half the combined headway, in plain
   minutes: T-585) plus their frequency-weighted downstream cost.
   **The service span** (T-842, Anita 2026-10-21: "demand should know the schedule, roughly").
   Riders differ in how long they would wait for the next train rather than go another way: the
   tolerance is spread log-uniformly from 30 minutes to 6 hours (half of riders take up to 1.7 h;
   `track::span::TOL_MIN_S`, `TOL_MAX_S`). For each line, direction, stop and period the clock
   worker works out `phi`, the share of the period's riders (arriving by the boarding account's
   shape, at the line's local time) whose wait for the next departure in a steady day is within
   their tolerance, rounded to hundredths (`TrackApi.line_span`, worked out again only when a
   line's departure pattern or stop times change). It is 1 wherever a train is never more than
   30 minutes away, and the snapshot carries it only for lines where it is not
   (`SnapshotLine.span`; `DemandApi.set_span`, every population alike, riders between cities
   too). Boarding a route node with `phi` below 1 costs -ln(`phi`) / 0.10 perceived minutes more
   (`Graph::rn_span`, added with the wait for full trains: 0.9 adds 1.1, 0.5 adds 6.9, 0.1 adds 23,
   0.01 adds 46), the logit's own form for an option open to that share of riders; a stop whose
   `phi` rounds to 0 has no departures in the search. A period whose `phi` changed is solved again
   like one whose trains an hour did (T-124), and the cache key (T-328) includes it. The wait
   inside service stays half the period's headway.
   A change of trains costs a fixed 2 perceived minutes (`transfer_min`, Anita, T-717), its wait
   and the walk between the two platforms' centres (below); the first boarding pays neither the
   fixed cost nor a platform walk. A line joins only if it improves expected trip cost;
   wrong branches and off, slow, expensive or crowded alternatives add no useful frequency when
   they worsen that cost. Lines are offered to a station cheapest first by their cost from
   boarding, which includes the extra wait where full trains leave riders behind (step 7): a full
   train is weighed after the lines that beat it, never kept ahead of them (T-552). **A rider
   never changes onto the line they left**, in either direction (Anita, T-575): where full trains
   leave riders behind they wait at their station, and the trip explorer shows that wait in the
   boarding, rather than ride the other way to board upstream. A station whose strategy boards a
   line has a change node for riders reaching it off that line (by alighting or by a walk), with
   its own strategy over the station's other lines; a station with no other line has none, and
   the alighting there leads nowhere for this destination. A rider may still ride out on one line
   and back on another (another line from the stop upstream, or one parallel to the full line), as
   any change is allowed. A rider leaving
   a train may walk to another station within 300 m straight line and board there (walk at its
   weight). **A boarding rides at least one stop** (T-1055, T-200): a route node is offered to its
   station at the cost of riding on to the next stop, even where riders already on board do
   better to leave there, and the loading, the fare's km and the trip explorer send its boarders
   along that ride. So boarding only to walk to another station or platform, or to alight again,
   is never a strategy, and every rail trip rides a train.
   Snapshot stations within 100 m of each other are one demand station (side-by-side platforms of
   two lines; never two neighbours on one line). **Station complexes are transfer points** (T-161):
   the clock worker sends each constructed station's complex with the snapshot (game/complexes.ts
   over constructed stations only, with the player's links, T-197; `DemandApi.set_complexes`), and
   a rider leaving a train at one may walk to any other demand station of the same complex however
   far, at the same cost as any walking transfer: from the platform left to the other station's
   position, x 1.3 detour at 1.2 m/s, weighted 1.39, then the boarding's usual wait and the
   change's fixed 2 minutes. **Walks between platforms** (T-717): a demand station keeps each
   stop's platform (its snapshot station's position, `Network::stop_x`), and a change there walks
   from the platform arrived at (or the station's position, after a walk from another station) to
   the platform of the train taken, at the same speed and weight. Riders take the first useful
   train, so the search adds the expected walk over the departures pooled there, by frequency, to
   the arrival's cost (`platform_walk`), and which lines join the pool does not depend on the
   platform arrived at; the trip explorer shows each route's own walk. A change across one
   platform walks nothing, platforms 80 m apart cost 2.0 perceived minutes, 120 m 3.0 and a linked
   pair 1 km apart 25, each with the fixed 2. The members stay separate demand stations joined by walk edges rather
   than one merged station: each keeps its own catchment and position, a long link is not a free
   teleport, and transfers stay counted platform to platform for the complex map (T-176). The
   trip explorer shows such a change as a walk between the two stations. The 100 m merge is single
   linkage, so a chain of platforms is one demand station; complexes add walks only between
   members more than 300 m apart (long platforms end to end, a long chained complex the player
   builds, linked stations; the real New York save has none, notes/T-161.md).
   Destination strategies and boarding frequencies
   are kept for loading. Each destination's search is independent, so only stations someone can
   end a rail trip at are searched towards (those in the access lists of subzones with jobs,
   `egress_stations`): a population living near two stations of an 875-station network needs two
   searches. The tables are shared by the populations of a worker (`kernel::with_skims`, T-119):
   populations whose graphs are the same to the bit (a city's commutes and everyday layer; on the
   real save also New York and the regional market) search once per period and round. Each city
   converts station positions to its own frame, so cities can differ in the last bits and bring a
   graph each. A worker keeps two tables; when a third graph comes, the table with the fewest
   searched destinations gives way, and only to one with at least as many to search (a cheaper one
   is searched for that call alone), so the order populations are solved in does not change what
   is searched (T-405). A table is stations² costs and stations² walks (u16 metres, T-1059) plus, per searched destination, each node's next
   node and the settle order (u16 ids below 65,535 nodes, change nodes included), one bit per
   boarding route node for the departures accepted there (their trains an hour are the graph's),
   and each change node's station and a bit per boarding edge of that station for the departures
   it accepts: about 4 bytes a node and 16 a change node (T-398, T-581; sizes in
   notes/T-581.md). The search's queue holds each
   node at most once (a lower cost moves it up) and pops by cost, then id, as a queue of copies
   would (T-581).
5. **Mode choice per subzone pair.** The access logit factorises: each origin subzone gets one row
   over all stations, the logsum of its access options followed by every station-to-station cost
   (at most 6 options x stations multiply-adds); each pair then multiplies that row into its
   destination's egress weights, up to 6 multiply-adds, one ln and one logit per pair. Pairs whose
   two walks together pass `pair_walk_max` (80 perceived minutes, 4.3) are skipped without being
   visited: each zone's destinations are kept cheapest walk first, and the destination zones by
   their cheapest walk, so an origin stops at the first that is too far. Zone pairs carrying under
   0.1 trips in the period (`COMMUTE_CHUNK`; effect in notes/T-874.md; 1 trip in a zone opened coarse, T-1222) are dealt their trips in chunks (T-874, as the market's, T-780): per origin zone and block of
   8 x 8 destination zones, the block's small pairs in zone order are sampled by trips from a
   start fixed by the zone and block, each pair drawn carrying its draws' trips and the rest
   skipped. Each block's trips stay exact and the deal depends on trips alone, never on rail's
   costs, so crowding rounds and timetable changes deal the same pairs and an edit re-deals only
   the blocks whose zones' trips it changed. `?perfOff=chunks` solves every pair. The pair bound
   keeps the pair count near what a 2.5 km walk edge gave (counts: notes/T-090.md). What a pair
   reads about its destination is packed in visiting order, its egress list padded to 6 with
   weight-0 entries. An origin first lists its pairs within the bound, zone by zone, then takes
   them eight at a time in SIMD lanes (`lanes.rs`, T-036: wasm SIMD in the browser, SSE2
   natively), logsum, drive, logit and the spread of each pair's rail trips onto its egress
   stations in one pass. Each lane does the scalar expressions in their order, a pad adds exactly
   0, and every sum keeps the pairs' order, so the results are those of one pair at a time to the
   bit (the trip explorer's `pair_choice` computes one pair alone). Rail trips go back onto access
   stations once per origin subzone and are kept by station pair (over the stations in the city's
   access lists only, since no other station starts or ends a commute: a city a few stations reach
   keeps a few rows, not the whole network's, T-398), and by subzone and access
   station at each end for the demand views (T-084, notes/T-078.md). The views ask for them on
   demand, never pushed: commuters by mode per cell (`cellModes`), a station's catchment and far
   end (`stationRiders`, with the home riders split into those who walked and those who rode to
   it, commutes only: everyday trips and the regional layer are not split, T-374), and where the commuters living or working in any set of cells go, every
   mode (`flows`, T-098 for the bubbles: the gravity's zone trips from the selection, rail along
   the kept station pairs, walk and drive the rest; totals exact and equal to `cellModes`, far
   rail by place blurred like `stationRiders`; the far cells holding the last 1% left out as a cut
   total; notes/T-098.md). A selection's everyday
   purposes (`EverydayApi::selection_sparse_day`) visit only the selection's destination samples
   (a purpose's samples laid out by the selected end are read by range, the others scanned for
   selected cells before any is weighed) and answer only the far cells they reach; the main
   thread adds the workers' answers up over those cells (`workers/selectionMerge.ts`; T-285,
   timings in notes/demand-kernel-history.md).
   **Anita's walking rule** (T-1059, `Params::walk_rule`): a station pair is no rail option for a
   trip when the walk to its first station, the walks between stations and platforms on the way
   and the walk from its last together are longer than walking between the trip's ends. All in
   straight metres: an access entry's mean distance (none for a drive to the station), the
   strategy's expected walks (`Skims::walk`, u16 metres station to station, worked out from each
   destination's strategy as it is searched: a walk to another station from the platform left,
   and at a change the walk to the first train's platform, by frequency), and for the trip the
   distance its walk uses (commutes: between the subzones' centres; everyday: the pair's mean),
   times the pack's walk factor over water. The refused pairs leave the logsum, so rail costs
   more there, and a trip with none left has no rail (the trip explorer says walking is
   shorter). The commutes keep their lanes: each origin knows the most its trips can walk before
   their last station (an access walk and the most its station walks on to any destination,
   `Skims::walk_max`), and a lane whose distance is past that and its destination's longest
   egress walk is taken as before (98% of lanes on the real New York and Tokyo networks). Only the
   rest look at their station pairs one by one, scale the lane's logsum by the share kept, load as
   usual and take the refused pairs off again, or, where most are refused, load the kept pairs
   alone (f32 would lose too much in the taking off). The everyday pass, the pair choice
   (`walk_kept`) and the trip explorer skip the refused pairs in their own loops. Long trips between cities do not apply it (their
   walks are never close to the trip's length).
6. **Loading**: each destination's strategy is walked from farthest settled nodes towards it.
   Station flow (and a change node's) splits by accepted departure frequency, retaining each line's individual segment
   loads and boarding/alighting counts (T-069). Transfers are counted beside the loads (T-176,
   "Transfers" below).
7. **Crowding**: a penalty per segment from the loads, then the pass again, averaging the loads
   (method of successive averages), for the rounds below. Each round costs as much as the first pass. The riders projects worked
   this out; read memory `reference_transit_crowding_assignment.md` (in
   `~/.claude/projects/c--Users-anita-projects-maps/memory/`) first: the penalty never saturates,
   and averaging is what stops it ringing. It is 1 + 2x(1+x)/2, x from the larger of the seats and
   **70% of crush** to crush (Anita, T-710: riders mind crowding only near full; 3 at crush,
   5.4 at 120%; `crowd_start`). **At most three rounds**
   (T-022; one round leaves the segment that takes the overflow at its worst, notes/T-022.md).
   After them the riders are held to the trains' room (below). **Each period stops early once settled**
   (T-434, `kernel::settled`): a round is skipped when the round before moved the period's summed
   segment loads (every population's, rider weighted: the sum of the changes over the sum of the
   loads) by under 1%. The first crowding round always runs (notes/T-434.md). A round in which no segment is over its seats has the
   free-flow estimate's factors (all one), so it would solve the same pass again: it is skipped,
   with exactly the same result (T-119). A skipped round still counts as a round, so every period
   reports the same round numbers.

   **Riders left behind** (T-188, `kernel::left_behind`, `round_costs`). Each round also prices
   the wait for full trains. From the same frozen loads (every population's segment loads,
   boardings and alightings per route node; the worker sends all three), each direction of each
   line runs the boarding account's rule (docs/track-and-service.md 6.6) on the period's own
   trains: an even headway at the period's frequency, riders arriving at each stop by the
   account's shape for the period (fastest mid-period, the busiest hour at the busiest-hour
   factor), each train letting off the stop's alighting share and taking riders up to crush, the
   rest waiting a headway more for the next train of the same line and direction. A queue left
   at the period's end clears at the trains' average spare room. The rider-minutes waited
   beyond the first train, over the stop's boardings, are that boarding's average extra wait
   (capped at 60 minutes, `LEFT_WAIT_CAP_MIN`, for a line whose queue never clears), added to
   the line's cost at that boarding in plain minutes, as any wait (T-585), in the strategy search,
   so riders spread to other lines, other stations or the car. The free-flow estimate has none.
   It runs inside the solve, never read back from the clock worker's account: the same rule on
   the same trains, except that demand uses even headways, each period stands alone (no queue
   carried in from the period before) and nobody goes another way (the account's T-1067). Per period, not per hour; weekdays only, the weekend keeping
   the weekday's waits with its route and mode choices (4.2). Why, and the effect on the real New
   York save: DECISIONS 2026-10-13 (T-188), notes/T-188.md.

   **The trains' room** (Anita, T-1070, `kernel::capacity_fit`, `cap_with_external`). The
   averaging keeps a share of the free-flow pass (a quarter after three rounds), so the rounds
   alone leave an overloaded line far over what its trains carry, whatever the wait or the
   crowding curve (notes/T-1070.md). So once a period's last round is in, its final loads (every
   population's, as the rounds read them) are held to the room: no segment carries more in the
   period than its trains hold at crush in the busiest hour (trains an hour x crush x hours /
   the busiest-hour factor, the room `load_of_crush` measures 100% against). Each direction of
   each line is walked from its first stop as its trains run: the riders staying on keep their
   places, a stop's boardings get on up to the room left, and the rest go by car or on foot,
   each as the trip's other riders do (the walk takes its share back, `walk_lost`). Each
   population scales its own figures (`cap_loads`): the segment loads, boardings, alightings and
   transfers by route node; a station's entries by those turned away at their first train there;
   riders turned away at a change off every entry alike; the rail trips by place by the entries
   kept at the end they set out from (commutes: to work from the home end, home from the work
   end, by the period's shares; everyday trips per pair; the market's figures by its share kept
   overall). Their rides on the trains before a refused change stay counted. The weekend reads
   the capped weekday. A line within its room is left exactly as the rounds left it. Fares and
   the money read these loads, so income is the riders who fit. The trip explorer's shares do not
   apply it.
