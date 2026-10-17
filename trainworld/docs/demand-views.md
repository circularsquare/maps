[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: answers for the views (4.4)

Loads by purpose, transfers and the trip explorer; the interface side is in the ui-*.md files.

**Loads by purpose** (T-158, the map's purpose filter, [ui-demand-views.md](ui-demand-views.md)). Every period's result also
carries its segment loads, boardings and alightings split by purpose: work is the local city
populations together, then the everyday layer's six purposes (four before T-142's airports and
medical visits), then the regional market as
longer trips. The populations were already separate; the everyday layer's purposes are kept apart
inside its pass. Its pairs are stored purpose by purpose, so one station-pair matrix is filled
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
on the real network the commutes' transfers equal all boardings less entries
(2,278,897 in periods 0 and 2; 593k of them walk between demand stations).
Query: `transfers(sel)` on each API gives (route node in, route node out, riders) for every
pair with either end at a route node marked in `sel`, for each period held; the worker adds every
population's and period's by pair and answers the `transfers` message with the route nodes of one
complex, on request only (never pushed with results). The client adds the workers' answers and
names each end by station and line, both runs together (`complexTransfers(members)`).
Cost on the real New York network (native, period 0): one commute assignment 36 ms, 42 ms counting
transfers (riders arriving at a station x its useful departures, about half of them, for each
destination); whole free-flow stages best of five, counted against not, commute 1,027 against
1,024 ms and everyday 337 against 339 ms, within the noise, so they are counted every round rather
than on the last only. Memory: 28,538 slots, 111 kB a period and population plus a 43 kB index
(about 0.5 MB a worker holding one period of New York commutes, its everyday purposes and the
market). The New York save's Penn Station answer is 886 pairs, 10.6 kB, 9 ms in the worker.
Loads with and without counting are bit-identical natively (`transfers_real_network`, two periods,
free flow and three rounds, commutes and everyday) and in the WASM worker (five periods, three
rounds, the market too; notes/T-147.md).

**The trip explorer** (T-195, `sim/src/demand/trip.rs`, the interface in [ui-inspectors.md](ui-inspectors.md)). One trip
between two points, as the commute model decides it in each period: `DemandApi.trip(q, from, to)`
takes the cell nearest each point (within 600 m, else the point is outside the city), reads the
trip as a commute from a home in the first to work in the second, and answers with the model's
own numbers for that pair of subzones (4.4 step 5): the drive (real minutes between the subzones' centres, its parking, its perceived cost), the
rail logsum over every pair of access and egress stations and each pair's share of the trip's
rail riders, rail's utility against the road and the chance of walking (the whole way, by the
two zones' distance; shown as one number in a hover), rail and driving. The
expressions and their order are `mode_choice`'s, so the figures are the same f32 numbers, on the
period's own graph (the costs of its last crowding round; the published loads are the rounds'
average). Rail is a strategy, not one path: riders spread over the station pairs by the access
logit and, at each station, over the useful departures by frequency. So the answer also walks the
destination strategies `assign` loads, from each station pair carrying at least 0.5% of the
trip's rail riders, and lists every route (a sequence of boardings, with a walk between stations
where the strategy takes one) carrying at least 0.5%, at most eight, biggest first, each with
its share, legs, waits (half the combined headway of the useful trains at the station),
in-vehicle minutes with and without crowding, and distance fare. Strategies are searched towards
the destination's egress stations only (at most six searches, a few ms; each search is
independent, so they are the solve's). It changes no state and is answered between stages, like
the other queries: each solving worker answers for the periods it holds, only from a round that
all its periods hold (`trip` message, the cells' positions in world metres). A trip whose end has
no subzone (no station within 4 km), or whose cheapest walks pass `pair_walk_max`, or with no
connecting service in the period gets no rail; its drive is priced between the two cells.

Checks: over every destination subzone, an origin's rail trips from these pair choices add up to
the solve's own for that origin subzone, and the station pairs' shares to its station-pair
matrix, within 1e-6 (f32 summation order) on the synthetic city, free flow and crowded
(`pair_choice_reproduces_mode_choice`), and within 6e-6 for the three origins of
`trip_real_network` on the real New York network; the WASM worker's answers for those three
trips equal the native ones to 1.7e-7 relative (rail and drive chances identical). On the real
save in Chrome a trip is answered in 3-14 ms. Park Slope to Midtown in the morning peak: train
91%, drive 9% (the drive is 36 minutes plus parking that counts as 43); the busiest route is the
F from 7 Av to 47-50 Sts-Rockefeller Ctr with 17% of the trip's train riders, the rest spread over
neighbouring stations at both ends and the 42 St stations. Jersey City (Journal Square) to Wall
St: train 62% (PATH to World Trade Center, then the 1 or a walk). Mineola to Hicksville: drive
63%, train 37%.

An everyday purpose's trip (T-586, T-201): `EverydayApi.trip(kind, q, from, to)` answers in the
same layout as the everyday pass chooses, on its own period graph and parameters (its rail and
walking constants): the subzone pair's mean distance over the purpose's samples (the two cells'
distance when no sample joins the two subzones) prices the drive, the walk and whether the home end
may ride, and walking is the two cells' own no-rail share less the share rail takes (the views'
rule, T-558). The pair's rail chance equals the free-flow solve's to the bit for one sample of
every pair (`everyday_choice_reproduces_the_pass`). The answer adds the whole walk's real minutes
and the subzone pair's trips a day. The explorer's headline is not this answer but the selection's
far end summed over the two places (ui-inspectors.md); this answer is its representative trip.
