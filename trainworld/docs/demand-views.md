[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: answers for the views (4.4)

Loads by purpose, transfers and the trip explorer; the interface side is in the ui-*.md files.

**Loads by purpose** (T-158, the map's purpose filter, [ui-demand-views.md](ui-demand-views.md)). Every period's result also
carries its segment loads, boardings and alightings split by purpose: work is the local city
populations together, then the everyday layer's six purposes, then the regional market as
longer trips. The everyday layer's purposes are kept apart inside its pass: its pairs are stored purpose by purpose, so one station-pair matrix is filled
by a purpose's pairs and then packed into that purpose's columns
(origin stations and trips, by destination), and the matrix reused for the next. One walk of each
destination's strategy then carries every purpose at once, a lane per purpose (T-433,
`assign_lanes`): each lane does the kernel's assignment arithmetic in its order, so a purpose's
loads are those of assigning it alone to the bit, and the total is the purposes' sum, which is
what crowding reads. Each purpose's loads are averaged across crowding rounds like the total, so
they still add up after every round. Nothing is solved again per purpose and mode choice never
sees the split. The kept purpose loads are 3 arrays per purpose per period (one float a route
node each), and each result message carries 8 purposes x 3 arrays. One purpose's local trip legs
per cell for the bubbles come from `EverydayApi.cell_modes_of(kind)`, asked of the solving
workers on demand (`purposeModes`) and answered only from a finished round, like stations; work
alone is the commute sums the bubbles already have.

**Transfers** (T-176, for the complex map, [ui-inspectors.md](ui-inspectors.md)). A station's flow is everyone there, so
loading alone forgets which line a rider arrived on. `kernel::assign_with_transfers` keeps it: when
a route node's riders leave the train at a station that is not the destination (at their own
station, or by a walking transfer to one within 300 m or in the same complex, T-161), the riders and the edge they arrived by
are noted on that station's list for this destination. A station is loaded after every node that
feeds it (the walk goes farthest first), so when it splits its flow over its useful departures the
list is whole, and each arrival splits the same way as everyone there: riders x that departure's
share, added to the slot of (arriving edge, departure). Slots are laid out once per graph
(`xfer_index`; every edge from a route node to a station, then that station's boarding edges in
order), so a slot is a pair of route nodes, line, direction and stop in and out: platform to
platform, merged demand stations included. The loads are `assign`'s to the bit; the transfers are
counted beside them. Every population does it (commutes, the everyday pass with all its purposes
in one array, the regional market), each round, averaged across rounds with the loads' weights.
The everyday walk notes an arrival's riders summed over its purposes' lanes, so its transfers
equal the purpose-by-purpose count to float rounding, not to the bit (T-433).
Check: at every station the transfers leaving equal its boardings less its first boardings, and
the commutes' transfers equal all boardings less entries.
Query: `transfers(sel)` on each API gives (route node in, route node out, riders) for every
pair with either end at a route node marked in `sel`, for each period held; the worker adds every
population's and period's by pair and answers the `transfers` message with the route nodes of one
complex, on request only (never pushed with results). The client adds the workers' answers and
names each end by station and line, both runs together (`complexTransfers(members)`).
Counting costs nothing measurable, so transfers are counted every round rather than on the last
only; loads with and without counting are bit-identical (`transfers_real_network`). Costs and
memory: notes/T-147.md.

**The trip explorer** (T-195, `sim/src/demand/trip.rs`, the interface in [ui-inspectors.md](ui-inspectors.md)). One trip
between two points, as the commute model decides it in each period: `DemandApi.trip(q, from, to)`
takes the cell nearest each point (within 600 m, else the point is outside the city), reads the
trip as a commute from a home in the first to work in the second, and answers for **those two
cells** as one subzone per cell would (Anita, T-720; `cell_subzones`): each cell's own walk to its
6 nearest stations, the home's rides those of its subzone, the drive between the cells, and
otherwise `mode_choice`'s expressions (4.4 step 5): the drive (real minutes, its parking, its perceived cost), the
rail logsum over every pair of access and egress stations and each pair's share of the trip's
rail riders, rail's utility against the road and the chance of walking (the whole way, by the
two zones' distance; shown as one number in a hover), rail and driving. The
expressions and their order are `mode_choice`'s, so the figures are the same f32 numbers, on the
period's own graph (the costs of its last crowding round; the published loads are the rounds'
average). Rail is a strategy, not one path: riders spread over the station pairs by the access
logit and, at each station, over the useful departures by frequency. So the answer also walks the
destination strategies `assign` loads, from each station pair carrying at least 0.5% of the
trip's rail riders, and lists every route (a sequence of boardings, with a walk between stations
where the strategy takes one, or between the platforms of one station, T-717) carrying at least
0.5%, at most eight, biggest first, each with its share, legs, waits (half the combined headway of
the useful trains at the station), in-vehicle minutes with and without crowding, distance fare
and each change's fixed cost. Strategies are searched towards
the destination's egress stations only (at most six searches, a few ms; each search is
independent, so they are the solve's). It changes no state and is answered between stages, like
the other queries: each solving worker answers for the periods it holds, only from a round that
all its periods hold (`trip` message, the cells' positions in world metres); a zone open bare is
answered by the aux worker from its copy solved on no network, every period (T-979). A trip whose end has
no subzone (no station within 4 km), or whose cheapest walks pass `pair_walk_max`, or with no
connecting service in the period gets no rail; its drive is priced between the two cells.
The answer also carries the subzones' own chances of walking, rail and driving and their cheapest
walk at the home end (the head's last four numbers): what the demand views count for the trip,
which the explorer's hovers give where it differs from the cells' answer.

Checks: the explorer's walk to a station is the home cell's own, and with one subzone per cell
its chances equal the subzones' to 1e-4 (`t720.rs`). Over every destination subzone, an origin's rail trips from these pair choices add up to
the solve's own for that origin subzone, and the station pairs' shares to its station-pair
matrix, within 1e-6 (f32 summation order) on the synthetic city, free flow and crowded
(`pair_choice_reproduces_mode_choice`), and within 6e-6 for the three origins of
`trip_real_network` on the real New York network, and the WASM worker's answers equal the native
ones. Timings and example answers: notes/demand-measurements.md.

An everyday purpose's trip (T-586, T-201): `EverydayApi.trip(kind, q, from, to)` answers in the
same layout as the everyday pass chooses, on its own period graph and parameters (its rail and
walking constants): the two cells' own distance and walks price the drive, the walk and whether
the home end may ride (the subzones' answer, beside it, takes the subzone pair's mean distance over
the purpose's samples), and walking is the two cells' own no-rail share less the share rail takes
(the views' rule, T-558). The subzone pair's rail chance equals the free-flow solve's to the bit
for one sample of every pair (`everyday_choice_reproduces_the_pass`). The answer adds the whole walk's real minutes
and the subzone pair's trips a day. The explorer's headline is not this answer but the selection's
far end summed over the two places (ui-inspectors.md); this answer is its representative trip.

**Car trips by pair of squares** (T-1435, the drives view, [ui-demand-views.md](ui-demand-views.md);
`sim/src/demand/api/drives.rs`, `everyday/drives.rs`). A zone's grid of 2 km zones in squares of
`side` zones (the smallest of 2, 3, 4, 6... with at most 800 squares), every unordered pair of two
squares one entry. Two parts, so a new result costs only the second. With no rail, needing no
network: `DemandApi.drive_base_day(day)`, the commutes' legs, their car legs (the zone pair's
walking with no rail left out) and those times each zone pair's train gain (`train_gain`: its drive
by `drive_min_f` between the zones' centres less 15 minutes and the straight line at 100 km/h);
`EverydayApi.drive_base_day(day)`, every sample's car legs and the same times its gain, the drive
between its two cells. Zone pairs under a hundredth of a leg a day are left out (0.25% of New
York's commutes). Over a worker's solved periods: `drive_rail_day(day)` on each, the car legs rail
takes, sparse. The everyday layer's are exact per sample (rail takes from walking and the car in
proportion, T-558). The commutes' rail is spread through the station pairs as `flow_rail` reads
them (each station's boarders over the squares they come from, its alighters over those they go
to): exact in total, blurred per pair; a pair's car legs are its car legs with no rail less its rail
times their share of its legs. The client (`driveArcs`) asks for the first part once per zone and
day and for the second with each demand result, and the sums worker makes the view's groupings
(`game/drives.ts`). Display only: demand is untouched.
