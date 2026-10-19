[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: the kernel (4.4)

How the commute and everyday trips are computed: gravity between zones, access, strategies, mode choice, loading and crowding.

### 4.4 Cost of computing it

Measured in `sim/src/demand/` (method and all numbers in notes/T-005.md, T-019.md and T-020.md)
on the real New York pack (21.7M people, 10.2M jobs; 74,920 cells, and 202,337 since blocks are
spread over their area, T-013) and a synthetic city of 153k cells, with a hand-made 434-station,
25-line network.

**Commuting does not enumerate cell pairs.** 75k-150k cells make 5.6-23 billion pairs. Truncating by
distance does not save it: commutes average 15-20 km, and a cutoff that keeps them keeps billions
of pairs. Even pairing only cells within walk of a station is 90-180M pairs on this network
(1.3-2.6 s native per period) and grows with the square of the area the network covers. So demand
works on two levels:

1. **Gravity between 2 km square zones, once per city.** Doubly constrained, commuting residents to
   commuting jobs (4.1), decay by distance: **f(d) = d^-0.8 x exp(-d / 30 km), x k between zones in
   different first-level regions** (the US packs' curve, carried in each pack's gravity block). The regions are US states (a cell's state is the one holding
   most of its census blocks' people and jobs); a square a state line cuts is one zone per state,
   both at the square's position. k is the pack's `border_factor`: **0.23 for the US packs**,
   fitted with the decay length on New York and Philadelphia LODES OD (T-162: each city's own k
   came to 0.235 and 0.219, and 0.228 and 0.207 at T-198's curve), and **0.45 for Washington**, its
   own fit (0.454 at this curve, 0.486 at T-162's), since
   DC is a city counted as a state: with 0.23, DC residents working in Maryland or Virginia were
   3-4% of DC's commuters against LODES's 13.5% each (T-181; `build_city.py` `CITIES[...]
   ["border_factor"]` overrides `BORDER_FACTOR_US`). It stands for everything that keeps people
   working in their own state (separate labour markets, residency rules, tolls); the crow-fly
   gravity without it sent 2-4x LODES across state lines, land borders and rivers alike, while
   water inside one state was about right (T-159). A pack with no factor has 1. T-171 tested the
   alternatives on five packs' LODES (notes/T-171.md 2): per-state-pair factors tie it (they
   only separate DC from real states), a fade near the line is score-neutral, and a factor only
   between metros or a water or toll-crossing term in its place does clearly worse.

   **Borders outside the US** (Anita, 2026-10-12, T-181; no non-US pack exists yet). National
   borders get a factor; inside the Schengen area it is weaker (closer to 1) than across other
   national borders. First-level regions (states, provinces, prefectures, Länder) get none,
   factor 1, by default. The values are fitted on the country's own commuting data when its first
   pack is built. This does not contradict the US packs: there the first-level factor is a
   measurement, not a default, because LODES shows US states cut commuting to about a quarter
   (T-159, T-162). Elsewhere an admin-1 line is not assumed to: Tokyo's prefectures exchange
   millions of commuters, and many admin-1 layers have city-states (Berlin, Vienna, Mexico City)
   where a state-sized factor would be far too strong, as Washington shows. A country whose own
   OD data shows an admin-1 effect can still fit one. The mechanism is the same per-pack factor:
   a pack's `region` per cell becomes the country, and a pack spanning Schengen and non-Schengen
   borders would need a factor per region pair (not built). Inside a pack the default stays 1;
   the line between two packs of one country takes `pack_line_default` 0.4 (T-584, Japan's fit)
   unless the country has its own value.

   **The curve** (T-198, from T-185's diagnosis). T-006 fitted d^-1 on New York; T-162 kept that
   shape and moved its length from 29 to 40 km to give back the trip length the border factor
   removes. T-185 found the steep near end kept residents of job-rich places too close to home:
   in LODES they commute out to suburban jobs (reverse commuting) about as far down the list of
   nearest jobs as anyone else, and the model sent them 17-26% too short. The flatter
   d^-0.8 exp(-d / 30 km), one value for every US pack, lengthens job-centre residents' trips by
   0.6-1.3 km in all nine packs (still 9-19% short) and moves centre-to-suburb commuting from
   0.73-0.87x LODES to 0.88-0.97x, undershooting as Anita asked. Trips misplaced by length, county
   pair, nearest-job rank and centre/suburb class fall in eight packs (sum of the four 17-35% to
   12-27%); New York, the densest and most centre-heavy, is about neutral (16.2% to 16.7%: its
   own-county share falls from 42.0% to 40.1%, LODES 42.8%). Per pack: notes/T-198.md. It does not
   depend on the network, so edits never rerun it. The pipeline solves it (Furness, the sim crate's own code run natively, ~3 s)
   and ships a zone id per cell and two factors per zone in the pack (notes/T-004.md); opening a
   city rebuilds zone trips in ~2 ms in WASM instead of solving them for 8 s. Zone trips spread to
   cells by commuters at each end. **Joint gravity** (T-276, docs/zones.md 4): the shipped
   factors are those of one solve over every zone pack (`bin/joint_gravity.rs`, about 12 minutes,
   run by pipeline/intercity.py), so a pack's zone trips are the part of each commute that stays in
   it; `zone_home` and `zone_work` (per zone, the trips kept) let cells count their commuters per
   zone (`DemandApi::home_k`, `work_k`) instead of the pack's total over its people. Each pack's
   homes still send as many commuters as it has jobs (scaled per pack, as alone: scaled to the
   union, packs with net commuting from outside every zone would balance through their
   neighbours, and the solve barely converged). Furness stops at 500 iterations, max row error
   2e-4 (one Boston zone; the rest far below).
2. **Access per cell**: the 6 nearest stations by walk within the walk's 4 km bound (4.3), found by
   scattering each station over the cells near it. 7-13 ms native. With routes (T-275) each
   station's walk is searched once per position (0.6-1.2 ms a station: opening the real New York
   network's 867 stations costs 0.5-0.6 s native and 1.0 s in WASM, once per worker; an edit
   searches only its new or moved stations and costs what it did, 31 ms in WASM with or without).
3. **Access subzones**: the cells of one zone that share their nearest station. Each carries its
   perceived walk to each of its stations, averaged over its cells as the mode choice weighs them
   (a logsum at its 0.10 scale: the walk's cost grows faster than its length, and the near cells
   carry most of the rail), used at both ends of a trip. Cells with no station within 4 km are in
   no subzone: their commuters drive or walk. 2,750 subzones on the hand-made network, 5,032 on New
   York's real one (2,056 and 3,702 with the 2.5 km edge before T-090); access and subzones
   together are rebuilt on every edit in 16-27 ms on the hand-made network and ~57 ms on the real
   one in WASM.
4. **Station-to-station times**: one reverse strategy search per destination station over station
   nodes plus one node per stop per direction per line (T-069). At each station, useful lines
   share their departure frequency: boarding costs the wait (half the combined headway, in plain
   minutes: T-585) plus their frequency-weighted downstream cost.
   A change has no penalty of its own (Anita, T-557): it costs its wait and any walk between
   platforms (`transfer_min` 0). A line joins only if it improves expected trip cost;
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
   weight; walk edges start only after a ride, so walking between stations is never a rail trip).
   Snapshot stations within 100 m of each other are one demand station (side-by-side platforms of
   two lines; never two neighbours on one line). **Station complexes are transfer points** (T-161):
   the clock worker sends each constructed station's complex with the snapshot (game/complexes.ts
   over constructed stations only, with the player's links, T-197; `DemandApi.set_complexes`), and
   a rider leaving a train at one may walk to any other demand station of the same complex however
   far, at the same cost as any walking transfer: the centre-to-centre distance x 1.3 detour at
   1.2 m/s, weighted 1.39, then the boarding's usual wait. No fixed penalty, for a walk or a
   change: platforms of one demand station (side by side or stacked) cost no walk, a walk of 120 m
   costs 3 perceived minutes more than a change across the platform and a linked pair 1 km apart
   25 more. The members stay separate demand stations joined by walk edges rather
   than one merged station: each keeps its own catchment and position, a long link is not a free
   teleport, and transfers stay counted platform to platform for the complex map (T-176). The
   trip explorer shows such a change as a walk between the two stations. On the real New York save
   this changes nothing: the 100 m merge (single linkage, so a chain of platforms is one demand
   station) already makes its 855 complexes 873 demand stations, and the 20 pairs of demand
   stations sharing a complex are at most 269 m apart, inside the 300 m walk (Penn Station's 24
   platforms are 2 demand stations, Grand Central's 16 are 3, Times Sq's 10 are 2;
   `complexes_real_network`, notes/T-161.md). It matters for members more than 300 m apart: long
   platforms end to end, a long chained complex the player builds, and linked stations.
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
   is searched (T-405). A table is stations² costs plus, per searched destination, each node's next
   node and the settle order (u16 ids below 65,535 nodes, change nodes included), one bit per
   boarding route node for the departures accepted there (their trains an hour are the graph's),
   and each change node's station and a bit per boarding edge of that station for the departures
   it accepts: about 4 bytes a node and 16 a change node (T-398, T-581), 32 MB for New York's real
   network searched towards all its 873 stations in a peak period. The search's queue holds each
   node at most once (a lower cost moves it up) and pops by cost, then id, as a queue of copies
   would (T-581).
5. **Mode choice per subzone pair.** The access logit factorises: each origin subzone gets one row
   over all stations, the logsum of its access options followed by every station-to-station cost
   (at most 6 options x stations multiply-adds); each pair then multiplies that row into its
   destination's egress weights, up to 6 multiply-adds, one ln and one logit per pair. Pairs whose
   two walks together pass `pair_walk_max` (80 perceived minutes, 4.3) are skipped without being
   visited: each zone's destinations are kept cheapest walk first, and the destination zones by
   their cheapest walk, so an origin stops at the first that is too far. That bound is what holds
   the pairs where the 2.5 km edge had them: hand-made network 4.2M pairs (4.1M before T-090;
   with the 4 km reach and no pair bound 9.3M), real network 13.8M (13.6M; 31.8M). What a pair
   reads about its destination is packed in visiting order, its egress list padded to 6 with
   weight-0 entries. An origin first lists its pairs within the bound, zone by zone, then takes
   them eight at a time in SIMD lanes (`lanes.rs`, T-036: wasm SIMD in the browser, SSE2
   natively), logsum, drive, logit and the spread of each pair's rail trips onto its egress
   stations in one pass. Each lane does the scalar expressions in their order, a pad adds exactly
   0, and every sum keeps the pairs' order, so the results are those of one pair at a time to the
   bit (the trip explorer's `pair_choice` still computes one pair alone). Rail trips go back onto access
   stations once per origin subzone and are kept by station pair (over the stations in the city's
   access lists only, since no other station starts or ends a commute: a city a few stations reach
   keeps a few rows, not the whole network's, T-398), and by subzone and access
   station at each end for the demand views (T-084, notes/T-078.md). The views ask for them on
   demand, never pushed: commuters by mode per cell (`cellModes`), a station's catchment and far
   end (`stationRiders`, with the home riders split into those who walked and those who rode to
   it, commutes only: everyday trips and the regional layer are not split, T-374), and where the commuters living or working in any set of cells go, every
   mode (`flows`, T-098 for the bubbles: the gravity's zone trips from the selection, rail along
   the kept station pairs, walk and drive the rest; totals exact and equal to `cellModes`, far
   rail by place blurred like `stationRiders`; 10-20 ms in WASM for one cell or thousands, the
   far cells holding the last 1% left out as a cut total; notes/T-098.md). A selection's everyday
   purposes (`EverydayApi::selection_sparse_day`) visit only the selection's destination samples
   (a purpose's samples laid out by the selected end are read by range, the others scanned for
   selected cells before any is weighed) and answer only the far cells they reach; the main
   thread adds the workers' answers up over those cells (`workers/selectionMerge.ts`). A click on
   a bubble on the real New York save shows its far end 80-170 ms after the click when no solve
   is running (first click of a session about 300 ms); it took 0.5-0.9 s (1.5 s first) when every
   sample of the city was weighed and every worker sent three numbers per cell per purpose
   (T-285, notes/T-285.md).
6. **Loading**: each destination's strategy is walked from farthest settled nodes towards it.
   Station flow (and a change node's) splits by accepted departure frequency, retaining each line's individual segment
   loads and boarding/alighting counts (T-069). Transfers are counted beside the loads (T-176,
   "Transfers" below).
7. **Crowding**: a penalty per segment from the loads, repeat 4-6, average the loads (method of
   successive averages). Each round costs as much as the first pass. The riders projects worked
   this out; read memory `reference_transit_crowding_assignment.md` (in
   `~/.claude/projects/c--Users-anita-projects-maps/memory/`) first: the penalty never saturates,
   and averaging is what stops it ringing. It is 1 + 2x(1+x)/2, x from the larger of the seats and
   **70% of crush** to crush (Anita, T-710: riders mind crowding only near full; 3 at crush,
   5.4 at 120%; `crowd_start`). **At most three rounds**
   (T-022): the first round leaves the segment that takes the overflow at its worst (963% of crush
   on the real New York network against 310% before it). No crush cap: an overloaded line is the
   player's signal (notes/T-022.md). **Each period stops early once settled** (T-434,
   `kernel::settled`): a round is skipped when the round before moved the period's summed segment
   loads (every population's, rider weighted: the sum of the changes over the sum of the loads) by
   under 1%. The first crowding round always runs; the peaks keep all three, quiet periods stop
   after one or two (notes/T-434.md). A round in which no segment is over its seats has the
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
   Decisions (DECISIONS 2026-10-13):
   - **Inside the solve, not read back from the clock worker's account.** The account is worked
     out from a finished solve's riders on the main thread's schedule; feeding it back would need
     a second solve after every solve (and a third, to settle), and a new game would open on
     waits from no account. Run per period inside each round, it settles with the crowding by the
     same averaging (the waits come from the averaged loads, as the crowding factors do; no extra
     damping), costs nothing measurable (a loop over trains and stops), and needs no state at
     startup. The two are the same rule on the same trains; demand's differs only in using
     even headways and in each period standing alone (no queue carried in from the period before).
   - **Per period, not per hour.** Demand's periods are the solve's unit; the account's hours are
     not mapped back. The arrival shape gives the period its mid-period peak.
   - **Weekdays only.** The weekend is worked out from the weekday solve (4.2), keeping its
     route and mode choices, so it keeps the weekday's waits too; with fewer riders its trains
     are emptier, so this errs towards too few weekend riders on lines full on weekdays.
   On the real New York save (commutes and everyday purposes, native, three rounds;
   `left_behind_real_network`, notes/T-188.md): rail trips a day fall 0.1% (commutes 3,463,591
   to 3,460,098, everyday 4,831,084 to 4,825,736); riders left behind in the evening peak fall
   from 40,600 to 30,400 and in the morning from 5,600 to 4,500; the LIRR Hempstead Branch loses
   4.2% of its boardings and the Babylon Branch 2.3%; no busiest line moves more than 0.8%.

**If a bigger network makes it slow** (pairs grow with the square of the subzone count): add
workers first. Skipping zone pairs under 0.2-0.5 trips cuts pairs 1.1-4x but drops long trips
(segment loads 3.5-5.4% off, T-020). **If there is time to spare**,
10-minute walk bands are the accuracy lever: 2.4x the pairs, every error about a fifth smaller.
