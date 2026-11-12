[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: workers, zones and clusters (4.4)

Which packs a worker holds, bare zones, warm-up, the worker pool and clusters.

## Which packs a worker holds

**Which cities are open** (T-149). A worker holds only the city packs the game needs: the viewed
city and every city with a built station inside its footprint or within 4.5 km of it (the 3 km
walk plus a margin for edge cells and pack frames, `app/src/workers/cityReach.ts`). The reach
follows from the stations' positions alone, so an edit that moves no station (a timetable, a train
count) reuses the last one worked out, unless a zone's footprint failed to download then. The ride's
15 km does not widen it: a pack's trips run between two stations of its own pack and the work or
activity end walks, so a station only a ride from the pack carries none of its riders (T-394; the
regional market carries riders between zones). Any
other city has no subzone with a station, so all its trips walk or drive and it adds nothing to
any line's load, fare or crowding: leaving it unloaded changes no number elsewhere (checked
bit-identical, notes/T-149.md). Packs arrive in `open` messages after the
WASM starts, the viewed city's first; a solve covers the cities open when it starts, and the
client sends one only when every city its network needs is open in every worker. A city that
construction reaches later joins at the next solve, which the client sends at once; a city only
viewed or on screen opens bare instead (below). Opening a commute pack is cheap; the cost is each
city's everyday purposes (timings: notes/demand-measurements.md). A zone's **market pack** (T-377, `<id>.market.json`) opens in the
solving workers once the zone is reached, a built station lies within 30 km of its box (T-396's
rule, per zone), or the zone is selected (a zone selection, or a bubble selection in it); never for
viewing alone (T-801: tens of MB in every solving worker); before that no solve waits for it or covers it,
and a solve sent without a market pack open since is sent again. The workers compose the market
from the packs they hold (`MarketPacks.market`) whenever packs open or close: since T-1204 a
composition is 15-90 ms natively for one to seven packs (the first a worker composes 0.3 s more),
and the page still sends the
market packs downloaded together in one `open` once none is still downloading, and a worker opens
the market packs waiting for it together, one composition for them all (T-889), and city packs one
at a time (T-1533, "The zone in view first" below). **Packs close when
nobody needs them** (T-301): a zone that is
neither viewed, beside the view, reached by a station nor part of the load in hand closes in every
worker (bare or not) once it has been so for 60 s, or at once when more than two such zones are
open (the least recently needed first); a zone's market pack closes 60 s after the zone is in none
of those sets and no station is in its reach. A closed zone carries no rider on any line, so the shown
figures stand (a bare zone's city figures stay shown); the next solve leaves it out, and a worker
solving with it closes it when that solve ends. It reopens as it first opened (download, mostly
from the browser's cache, and opening). The bare copy of a zone that a shown solve covers closes at
once. WASM memory never shrinks, so closing bounds a worker's peak (the freed space takes the next
pack) rather than lowering it. `?perfOff=close` keeps everything open. A recomposition (a market
pack opened or closed) frees the old composition before composing, unless a market part still
holds it, and a part composed of the same packs serves again without composing (T-1198). A
composition holds its open places' pair rules and zones, 12-25 MB for a few places, and shares the
world's corridor figures (34 MB, worked out at the worker's first composition and kept, T-1204,
[demand-market.md](demand-market.md)). A new game
opens only New York; a save with construction in several cities pays
each of their openings once, in every worker. Only the cells with a budget for a purpose hold its
destination samples and their pair indices (T-395; about half of all cells x purposes have none),
and a cell holds one sample per 8 of its trips (T-397), a school or university one per 30, at most
100 (T-1874). A zone opened coarse (`DemandApi::at_level`,
T-1222, notes/T-1206.md) holds one per 64, 1 to 8 (airports, medical and attractions 1), and deals
its commute pairs under 1 trip a period in chunks; the same model at a lower sample density, its
rail on average its fine figure's. Which zones are fine is "Levels of detail" below. A sample's
pair index is held only once a network gives its layer a pair, written in place on the next
network, and the pairs are found through a 4-byte-a-slot table sized from the last network's count
(T-398). A city's everyday layer is opened beside its commutes (`EverydayApi::beside`, T-399) and
shares their cells, zone grid, gravity and routes; its own are the zones' sums of residents and
jobs, and the station access and subzones built on those weights (not the commute weights), so
each network sets up the two layers' subzones separately (memory: notes/T-399.md).
Zone centres sit on the zone grid, so the everyday destination gravity's decay comes from one
table per purpose by grid offset (T-172), and the long loops of the opening and of `set_network`
run in functions called once per zone or per 16k samples, because a browser optimises a
WebAssembly function only between calls (a single long call stays in the baseline compiler
throughout). Startup timings: notes/T-172.md, notes/T-174.md.

## Bare zones

**Bare zones** (T-325, T-336). A zone viewed or on screen beside the view (above zoom 7.5, T-871)
that no built station reaches opens **bare**: its pack goes to the auxiliary worker alone, which sets its commute and
everyday layers on an empty network and solves every period once. With no station in reach every
trip walks or drives, so these are exactly the figures any solve gives it. No solving worker opens
it and no solve is started or waited for, so a pan never re-solves. The `opened` reply carries
each period's city summary and purpose parts, from which the page builds the zone's city view as
from a solve's results. Bubbles ask the auxiliary worker (`bare`): commuters by mode, and the
everyday legs once per solving worker's periods (`cell_modes_mask_day`), added up in period order
as a solved zone's are. A bubble click asks it `bareFlows` (the commuters' far end, and per
solving worker's periods the work timing and everyday far end: `purpose_timing_mask_day`,
`selection_sparse_mask_day`) and `bareSquares` (the selection as the market's 600 m squares), which
go with `icSel` to the solving workers, holding the market, for longer trips and commutes between
zones. `checks/t325-bare-check.mjs` finds every one of these equal to the bit to the same zone
solved with no station in it.

- **At startup** (T-351) the viewed zone's pack downloads before the network's reach (the first
  network and the footprints) is known, and is sent once downloaded: bare if the reach is known by
  then and leaves the zone out, else to every worker. A new game (`?new=1`) has no station, so its
  reach is known at once: the viewed zone opens bare, the solving workers open only the market and
  solve it (under 0.1 s), and the loading card and held clock go when the bare figures arrive,
  which are final as they come (`holdDone`). The city figures and bubbles are identical to
  the bit to the full solve's; the network's own totals leave the bare zone out, as after a pan.
  A saved game's network arrives after its viewed pack has downloaded, so a save's viewed pack
  goes to every worker; waiting for the reach would cost every save that reaches its viewed zone
  more than it saves. In a game load's hold, a zone sent that way that the network, once known,
  neither reaches nor views (the last game's place, a save opened from a file) closes before the
  load's solve (`closeStrays`, T-2579), so the load's figures are the same however the page started;
  a cached result read ahead of the network leaves such a zone out of the zones it must cover (T-2588).
- **The first station** (T-351): with no station in the game, picking the track, station or
  blueprint tool opens the bare viewed zone in every solving worker as well (`prefetchViewed`),
  with no solve started for it, so a first line's solve need not wait. Timings: notes/T-351.md.
- A station built within reach downloads the pack again (the page keeps no copy) and opens it in
  every worker; the solve waits for it and covers it from then on. Its bare figures serve until a
  shown solve covers it. The auxiliary worker keeps its bare copy (WASM memory never shrinks).
  Memory and timings of a pan with bare zones: notes/demand-measurements.md.
- **Zones track crosses** (T-1047, `openCrossed`): a zone built track crosses (game/trackZones.ts)
  that no station reaches opens bare too, outside a load's card and only while no other bare
  opening is under way, one at a time, nearest the view's centre first. Nothing keeps it open: it
  closes as any spare zone does, its figures still shown (`bareKept`).
## Levels of detail

**Levels of detail** (T-1223, `workers/demandLod.ts`, design notes/T-1206.md). Each zone the solves
cover is open fine or coarse in the solving workers (the aux worker opens every zone fine; its
commute copies only map sums to cells). The page keeps a fine set, checked when the view settles,
on a click, on an edit, when a load's card goes and every 30 s:
1. the zone in view, once in view 2 s (a pan refines nothing);
2. the selected zones: a zone, a bubble selection (the zone in view), a station's zone, a trip's two ends;
3. zones edited in the last 15 minutes of wall time (stations added, removed or moved; a line's
   stops added or dropped, or its trains or cars changed: the zone in view if the line stops there);
4. other zones with bubbles on screen (`nearbyZones`);
5. every other zone reached.

Zones in 1-3 are always fine. A fine zone in none of 1-3 for 10 minutes goes coarse, whatever the
memory; past the budget the lowest fine zone goes at once (5 before 4, the least recently in 1-3
first); never one made fine within 5 minutes. The zones of the last two builds (a station placed or
a stop added; one edit's zones are one build; `builtKeep`, T-2005, T-2153) are spared both, however
long ago: they stay fine however long left alone, and go only when nothing else droppable is left
over the budget, the latest build's last of all. The save keeps them (`demandBuilt`), so a reload
spares the same zones. A coarse zone goes fine only by entering 1-3. While the player's edits
settle for the edit note (`editsSettling`, T-2180, [ui-map.md](ui-map.md) "Edit note") no check
runs and every fine zone counts as just looked at, so the held changes go at the next check after
and a zone's ten minutes left alone start from then. A zone
reached since the last check opens fine (a station was just built there). The budget is the zones'
memory a solving worker may hold, by T-1197's rate (`memoryCheck.ts`) with a coarse zone at 0.7 of
a fine one: 1.5 GB on a device reporting 8 GB, less in proportion below (a phone or tablet that
reports nothing counts as 4 GB, `deviceGB`). A game's first fine set is
the save's (`demandFine` in its header, kept current by the page), else every zone that fits the
budget, the zone in view first, then the zones on screen, then the nearest. A level change is a
close and reopen: the pack downloads again (the browser's cache) and each solving worker opens it at
its new level, one whose solve in hand covers the zone once that solve ends; the shown figures stay
until a solve covers it at its new level (`flush` sends one as for a zone newly opened), the zone
counting as open for the loading bar meanwhile. Zones changed in one check reopen together, so one
solve covers them. That solve is a **local refine** (T-1609, notes/T-1609.md):
in a solve without the barrier, a cluster whose network its other populations hold solved and
whose only new populations are zones opened at another level (at most half its populations, never
a market part) solves those alone, each period through its rounds and the capacity fit, crowding on
the other populations' loads round by round as their last solve was priced on (T-1733's `trail`;
without one, as that solve left them before its capacity fit) plus their own; the others keep
their figures, so lines carry the new zone's riders beside the others' as they were, and `plan`
names only the refined zones; when every cluster solved is one, the other zones' bubbles' sums are
the last solve's, sent again. Its periods are never kept as a whole solve's (T-124): the next edit
that touches the cluster solves it whole. Every solve records which of its zones were coarse (`Solve.coarse`); the zones
shown from a coarse solve are `roughZones` (no star or goal comes from them), and the zone in view
shown rough while fine is on its way is `refiningZone` (the status tag,
[ui-demand-views.md](ui-demand-views.md)). The demand cache's key marks each coarse zone's pack. `?lod=fine` opens
every zone fine and `?lod=coarse` every zone coarse, and neither changes them after.

## Warm-up and the worker pool

**Everyday warm-up** (T-173). Several workers opening a big everyday layer at once from the
browser's baseline code each take several times as long as one alone. So worker 0 opens a small
synthetic everyday layer first (`warm.rs`, 22,500 cells), and the client holds the other solving
workers' city packs until it is ready (`demandClient.ts` `held`). Compiled code is shared by a
page's workers, so they all open on optimised code. Timings in
notes/T-174.md and notes/demand-measurements.md.

**The pool and the bubbles' chain** (T-119). The pool is half the logical cores less one, at most
5, at least 1 workers that solve, and one more that holds no period ([architecture.md](architecture.md) 7). A solving worker that runs out of memory
(a failed WASM grow; `memoryShort`) halves the pool, the focus pool ends, and the same solve is sent again to the rest with
no figure changed; the smaller count is kept for the tab's session (`sessionStorage`, so the reload of "Play without
intercity trips" keeps it) and caps every later pool unless `?demandWorkers` says (T-2867, T-3072). Only the last solving
worker running out shows the out of memory notice. Periods go to workers
by index (worker i of n: periods q with q mod n = i), so five workers solve a stage of all five periods at once, unless
that puts two periods of a demand level on one worker (2 workers: high demand's three periods), when they are dealt
out level by level (high, then medium, then low: 0, 1, 2, 3 and 4 in that order, to workers 0, 1, 0, 1, 0), so an edit to one level's trains
has its periods solved side by side as far as the pool allows (T-124; `periodsOfWorker`).
**A market set apart** (T-1607, `marketsOfWorkers`, `?perfOff=share` off). Five periods do not deal
evenly over 2 to 4 workers, and on a big network a period's market is most of its work, so in a solve
with the barrier (a load, a zone opened; never a rough solve) a period's market may be solved by
another worker than its cities: the assignment that makes the busiest worker's round shortest, by
each period's time a round on all of Japan with the market 4/5 of it, used only where that round is
at least 11% shorter. In effect only with 3 workers: the evening's market goes to the worker holding
the evening peak alone (all of Japan's load 11% faster; 2 workers gained nothing, notes/T-1607.md).
The two sides solve their populations from the same
summed loads: after each round each sends its loads (segments, boardings, alightings on the whole
network's route nodes, `loads`) through the page to the other, which adds them in the order one
worker would (the cities', then the market's, a single population, so the bits are the same); each
sends its figures as a `result` part (`part`, `come`: the boardings `arrive` is worked out from) and
the page adds the two (`addParts`). The figures are those of one worker solving the period whole. A
worker drops any period of a population it does not solve (`forget_period`), so every answer counts
a period once (the market's answers by `Version.markets`). Such a solve solves every cluster with a
market and keeps none of its shared periods, and the next solve without the barrier solves those
clusters again, the shared periods whole on their cities' worker (the others kept as usual).
The bubbles ask for the
commuters by mode right after a round is published, when every worker is already part way into
the next round and cannot answer a question about the round before from its live state. So each
worker sends its subzone sums for a round (`sub_sums`, per city) unasked, from the moment all its
periods had that round and ahead of its last result for it; the client adds them up when the
round is published. Before a solve's last round only the zones on screen's go (`view`; T-2596: all 82
of a big save's took each worker about 4 s a round), at the last round every zone's, so a zone panned
into view mid-solve gets its bubbles a round later (`?perfOff=allSums`: every zone each round). They
are a weekday's; for the zone in view and the nearby zones (`view`, sent
by the page whenever they or the day shown change) the day shown's too (`sub_sums_day`, T-1904),
since the average day is the default and without them its bubbles waited a round behind the
figures (notes/T-1904.md). The commuters by mode per cell (`cell_modes`) read only those sums and the
network's subzones, so the worker with no periods answers them whatever round the others are in.
Stations, selections, transfers, trips and longer-trip queries read a period's state, so a worker
answers them only at a round all its periods hold; a selection's far end spread (`flows`) reads
only the sums it is handed and answers at any round. A worker answers one question at a time, so a
bubble selection's longer trips (`icSel`, seconds at most) are asked only once its far end's `flows`
is posted (at most 1 s later while a solve runs), and not at all once the selection is gone (T-2746). While the running solve's estimates are
shown (startup, a zone just opened), a cached result of the same network is, or nothing is shown
yet, the client asks about the round after the published one (round 0 before any), marked `wait`:
a worker short of that round keeps the question and answers it the moment it reaches the round,
just before that round's last result, so the client holds every answer when it publishes the
round. Such a solve is sent with `barrier`: a worker starts a crowding round only on `go`, which
the client sends to every solving worker as it publishes the round before, so a worker with fewer
periods (the default pool splits five periods 2, 2, 1) waits at the boundary holding its round
instead of running ahead past the round asked about (rounds counted per group of clusters, T-402:
"Clusters" below). That costs the final about 0.7 s and changes
no figure; solves after an edit run without it. A click then waits at most one stage, and its
figures are at most one crowding round ahead of the bubbles; the answer for the next round is
asked for at once, so each publish finds its answer waiting, and the final replaces it as always.
A worker already past the round asked about answers nothing and the question goes to the newer
round. After an edit questions about a zone being solved wait for its final (the workers hold a
network the map does not show yet); others are answered at once ("Clusters" below). T-327, notes/T-327.md.

## Clusters

**Clusters** (T-326, `workers/demandClusters.ts`). Each worker splits the network into clusters
that share no rider and solves each on its own sub-network: the cluster's stations (in snapshot
order) and its lines, with their own route nodes, mapped back to the whole network's in every
result and answer. A cluster is joined up from populations, stations and lines: a line joins its
stations; a station complex joins its stations; stations within 1 km join (merged platforms are
100 m, walking transfers 300 m); a city joins every station within 30 km of its cells' bounding box
(the 3 km walk and the 15 km ride from a home group's centre); the regional market, when open,
joins each of its own 6 km zones to every station within 30 km of the box of that zone's sites
(its rides reach about 17 km; `MARKET_REACH_M`, `IntercityApi.zone_boxes`), one node per zone, so
every station in a zone's access lists is in that zone's cluster, and two networks meet through
the market only where one 6 km zone reaches both (T-409). Clusters are
worked out afresh for every network, so a station or a population that reaches two of them merges
them. Each cluster the market reaches holds a part of the market of its own (`IntercityApi.beside`,
sharing the composition's sites and zones, kept by the cluster's market zones' world ids; a part
keeps the composition it was made from while its key lasts, so a pack opened elsewhere re-solves
nothing), set on the cluster's network alone: its pairs are those between zones its own stations serve, and a pair whose ends lie
in two clusters is in neither, its trips fixed in the pack's totals like any other pair with no
rail path. The market's figures are its parts' added up: rail from each, the trips, pairs and
corridor trips (which need no network) once. The corridors (place pairs with longer trips or
commutes) are a table no network or period changes: each pair's places, its longer trips and
commute legs a day (kept for the rows with any, a few in a thousand), and its shares by plane and by car
(game/demand.ts `CorridorTable`; place numbers 16-bit while the market has under 65,536 places, T-2485).
The worker holding period 0's market makes and sends a table once, by its key; after `corridorsAgain`,
any worker's next result with the market sends it again. Each period's result carries the
key, the period's factors of the trips and legs and the pairs with rail. A kind of day's corridors
are the table times the day's scales plus its pairs with rail, so no publish or view walks every
zone pair (T-789, T-1008). A market zone no station reaches is in no cluster;
with none reached the market is a cluster of its own with no lines, solved once and kept. Lines and
stations no population reaches are in no cluster and carry nobody. `?perfOff=split` joins the
market's zones into one node, as before T-409.
Every population of a cluster crowds on the cluster's own loads, which are the whole network's
there, since no other population puts a rider on its lines; so the figures are the same as one
solve of everything. A worker keeps what each population last solved, and a cluster whose
sub-network, populations and rounds are unchanged since keeps its result untouched: after an edit
only the clusters it touched are solved, and the others' figures are read from their populations
as they stand. A cluster whose lines moved only by noise keeps its result too (T-1862): the
same `shape` (the cluster's stations, complexes, upgrades, passages, bus stretches, fare multipliers,
stops, cars and hop km, its cities, market key, rounds, fares and frame), the same population
objects each holding this worker's periods from a whole solve (none from a local refine), every run
time within 0.5 s and trains an hour within 0.2% of the network the cluster was last solved on (not
the network before, so drift cannot add up), and its span exactly that network's. The settle moves
lines no edit touched by about that much (T-1850); one train more on a line is past 0.2%. Such a
cluster is re-keyed to the new network unsolved (`held`, `heldQ` and its capacity-fit ratios under
the new `sig`); one last solved with its market apart (T-1607) is re-keyed and then kept apart as
below. `?perfOff=noise` keeps a cluster only on exactly the network it solved. Every worker splits
the same way from the same inputs. `?perfOff=clusters` solves everything as one cluster, as before T-326.

## Each population on its reach

**Each population on its reach** (T-756, `reachView` in `workers/demandClusters.ts`). Within a
cluster a city's commutes and everyday layer are set on their own network: the stations within
30 km of the city's cells' box (`CITY_REACH_M`, the test that joins it to the cluster: past the
15 km ride from a home group's centre up to 9 km off its cell, and the work and activity ends walk),
and each of the cluster's lines with a stop among them from the stop before its first such stop to
the stop after its last, every stop between kept (a line that leaves the reach and comes back needs
no change of trains). The stops outside are those pads alone, where none of the city's riders
board. The regional market's part solves on the whole cluster, since its longer trips ride across
it. Every round each population is given the cluster's loads, all populations' summed on the whole
network's route nodes, on its own route nodes; at a pad its boardings are the riders already
aboard (the line's load leaving the pad), since the left-behind model runs each train from its
run's first stop. So a city's search, tables and passes grow with its own reach, not the cluster's.
Each population's crowding rounds stop when its own loads settle (T-434). `?perfOff=reach` sets
every population on the whole cluster, as before T-756.

**The market's pass by a window of zones** (T-1545, `IntercityApi::pass_window`). A zone pair's
logsum reads the route costs from its home stations to its destination zone's stations, and fills
those stations' columns of trips (a commute's or outing's trip home fills its home stations'). So
the pass goes through the destination zones in the order of their centres' x. A station's route
costs, the same with the planned first wait (T-1430) and its fare km (T-1286) are held from the
first zone that reads them to the last; its columns of trips (longer trips, their lean, commutes,
each everyday purpose) from the first zone that touches it to the last, when they are assigned
along its strategy and dropped. The strategy, searched at the first read, is kept to then within
192 MB (`WINDOW_STRATEGY_BYTES`, T-782, the stations assigned soonest first), else searched again. No
station x station table is held: on all of Japan (9,253 stations) at most 197 stations' route costs
and 1,954 stations' trips at once, the trips home of commutes making the long spans. A station's
loads are its column's, added in the window's order (rounding apart, the figures of one table of
every pair). On a small
cluster the strategies and route costs come from the table the cities share (`with_skims`,
T-119). When every destination's strategy would pass 256 MB (`MARKET_STRATEGY_BYTES`) and no
population on the worker has searched the period's graph (T-761), each destination is searched as
the window first needs it (`pass_searched`), the cities' cached tables dropped first
(the market solves last in its cluster). The longer trips' zone pairs are dealt in chunks
([regional.md](regional.md) 5.1, T-780), so the logsums and the far-zone table shrink with the
pairs; the far-zone table a station click reads (`station_flows`) holds only the entries a station's
zones can fill (`StPattern`, a tenth of all on Japan), and the zone pairs are counted before their
table is made, so it is never grown by doubling. While a pass runs, the table's start half (trips
starting at a station, by destination zone) is filled one zone's column at a time, in a second array
after the table (`StPattern::zoff`, `zst`, `zrow`, `zix`: each column's start stations and each start
entry's place in its column, made with the pattern once per network, 36 MB on Japan, plus 24 MB for
the second copy during a pass), and copied back to row order when the pass ends; every entry gets the same adds in the
same order, so the table is the same to the bit, and a zone's longer trips and commutes add into one
short column instead of a row per start station (T-2653). The commutes' and outings' trips home look
a station's entry up once a pair.

## Solving clusters

Within a cluster that is solved, a period whose own inputs are unchanged keeps its result too
(T-124). Periods share no rider and each reads only its own demand level's run times and trains an
hour (the trips home in a period go back by that period's own station pairs; the day's money, the
purposes and the weekend are worked out from the five periods' results afterwards), so a change to
one level's trains leaves the other levels' periods exactly as a fresh solve would give them. The
populations decide it themselves (`set_network`, `Service::same_periods`): with stations, lines, stops,
cars and complexes unchanged they keep station access and subzones, and every period whose level's
run times and trains an hour are unchanged; `set_fares` with the same fare keeps them as well. The
worker keeps a period only when every population of the cluster kept it and each holds it from a whole
solve of the cluster's period with the same inputs (`psig`, per period like `sig`), since the crowding
is the cluster's; the other periods are solved from free flow. A train-count change at one level
re-solves that level's periods (three at high, one at medium or low), a new station or a fare change all five.
`?perfOff=periods` re-solves every period of a cluster solved.

After an edit (a solve without the round barrier) the clusters to solve go one after another, the
zone in view's first (T-393), then the city view's zone's (T-2453), each to its last round; each but the last is sent as soon as it is
done, as one result per period with that cluster's figures alone (`result.cluster`), and the page
puts its lines, their stations, its zones' views and, if the market is in it, the regional market's
into the shown result at once. The network-wide totals wait for the solve's final: once every
cluster is done, the solve's result for every round is the final figures, read once (after an
edit only the final is shown, and new lines take round 0's, T-324). A cluster's network
is set up only when its turn comes, so until then it still holds the shown result's. A worker keeps
the clusters of its last six networks, and answers a question about any of them at once while the
asked zone's cluster holds that network solved to its last round, whatever it is solving
meanwhile: a click in a zone whose cluster is kept, or already published, needs no wait. Questions
about a published cluster's zones go to its solve. The top bar's faded split and "before your
last change" speak only for a zone still being solved (`plan`, `demandSolving`); the status tag shows
while any solve runs, its step (`demandStep`) the local solve then the whole one after an edit, the
rounds with the barrier (T-1790, docs/ui-panels.md "Status").
**The local solve after an edit** (T-1733, notes/T-1733.md): before those
whole solves, each cluster the edit touched is solved locally first when its market part and its
untouched cities still hold what they last solved: only the city populations whose view changed (a
station or stop of theirs added, removed or moved, their lines' trains changed; run times within 1%,
trains an hour within 5% and a stop's service span (the share of riders with a train due) within 0.06,
one train a day more or less, count as unchanged), through their rounds and the capacity fit. Each
round crowds on the others' loads as that round of their last solve was priced on (kept per
population, period and round, `trail`; a shared period's market loads per round as the other worker
sent them), plus their own, so with nothing changed it repeats the whole solve; the market's are
read through its old route nodes, carried to the new network's by line and station ids
(`solve.ids`). Each is sent as a cluster result marked `local`, its cities the touched zones, after
those zones' subzone sums (marked `local`, kept apart on the page until the whole solve's of the same
round are all in, T-2532), so their bubbles follow the figures: the page
puts its lines (not one shown at other fares, T-3303), their stations and those zones' views in (never the market's own figures) and
clears the lines' pending label ("calculating...", ui-inspectors.md). A worker then waits for the page's `whole`, sent 8 s after the
local results are shown, or at once when a worker solves nothing locally, so an edit following
within that time is solved locally too; the whole solves follow as above and replace everything.
**Stations on no line** (T-2410, `?perfOff=alone` off) are left out of the network demand solves
(`serviceFromSnapshot`; passages with an end among them too), so building stations ahead of their
line changes no key and starts no solve; their positions still reach zones, whose packs open ahead
of the line. The line then comes as one edit against the last solve, solved locally like any other.
**The market's pairs in a window** (T-1735, notes/T-1735.md): when the
edit's longest changed run (between a line's first and last changed stops: a stop added or dropped,
or a hop's run time past 1%, trains an hour not counted; the edit diffed against the network the
market part last solved on) is at least 20 minutes, the local solve also re-solves
the market's pairs with an end in a window of zones, in a population of their own beside the
frozen market (`IntercityApi.set_window`): the zones with a changed station in an access list, then
the zones whose nearest station is within that run's minutes by rail at free flow (at most 180),
nearest first, while the window's stations are at most 9% of those the market searches or 800,
whichever is more. Each pair is
searched towards its end in the window alone: a pair starting there and ending outside is priced on
the route costs the other way, a leg ending outside (a trip home, or the trip out of a pair kept
under its home) is assigned as the leg the other way turned round onto the same lines. It is solved
in one pass a period at the crowding the touched cities' last round ended on, then held to the
trains' room with them; the old market part gives the same pairs' old loads in one pass at its last
crowding (`window_loads`), and the result's loads are the frozen market's less those plus the
window's, each purpose to its own. A shorter edit (a city line) leaves the whole market frozen. The
window's pairs are taken from the market part's own tables when the new network serves the same
zones and joins them by track the same way (`set_network_ll_beside`, T-2878), else dealt from
nothing.
**An edit's solve runs to its end** (T-1734, notes/T-1734.md, `?perfOff=firm` off): from when the
page sends a solve without the barrier until its final is shown, a newer network (an edit, a zone
opened or at another level) is held on the page, not sent, and goes the moment that final shows,
solved locally on top of it. The one gap is those 8 s before `whole`: a network that arrives in them
goes at once, and one held since before them waits for the final, so edits coming faster than the
local solve never keep a whole solve from starting. A worker replaced meanwhile still re-solves at
once. So building steadily, a whole solve lands about every local solve plus whole solve (a line a
minute on 2 workers: all of China every 60-90 s, all of Japan every 19-26 minutes), and an edit made during one
shows its figures when it has landed and the local solve on top is done (the zone in view's sooner, from a
worker aside: "The zone in view aside" below). A solve with the barrier (a load, a zone just opened)
is still cut short by a newer network.
With the barrier
(startup, a zone just opened) the clusters to solve go in two groups when there are two or more and
the first ranked zone's is one of them (T-402; the zone in view's, else the city view's, "The zone
in view first" below; the city view's first while the focus pool solves the zone in view, T-2459): that cluster alone first, through every round, each
round sent as its own cluster result and shown where the solve's estimates are (a view of the
cluster alone when nothing else is shown, else only its final, put in as after an edit); then the
other clusters together, round by round, whose results are the solve's, with the first group's
figures read from its populations. Each group counts its own rounds: `go` names the group
(`plan.ahead` tells the page which cluster is first), and a worker tells which group a question
means by its zone, so a question about a zone in the second group waits for that group's round
and one about the first group's zone after its final is answered at once. Otherwise all clusters
step through the rounds together. The worker adds
cities up in the registry's order, never the order their packs opened in, so a network's figures
are the same to the bit however it was loaded. T-392, T-393, notes/T-326.md.

## The zone in view first

**The zone in view first** (T-1533, notes/T-1533.md). Every queue the zone in view passes through
takes it first. Its pack downloads at high fetch priority, ahead of the markets and the other zones
(`ensure`, `stationsAhead`); each worker opens it ahead of the packs waiting (`open.first`, and a
`first` message moves it up when the view reaches a zone whose pack is already queued); a worker
opens city packs one at a time and market packs together after them, and the page sends a game's
market packs only once its city packs are in, since composing China's 370 holds a worker for a
minute; the aux worker opens one pack at a time, so the bubbles' questions are answered in
between. While a whole solve waits for packs still downloading or opening, the zone in view, once
open in every worker and with no figures yet, is solved alone with no market (`solve.only`, a
**rough solve**, `?perfOff=rough` off): it starts ahead of the opens waiting, which wait for its
free-flow round, and its rounds show as estimates (never final: the card, the held clock and the
cache wait for the whole solve, whose figures replace them): over the whole view while nothing final
is shown, and over a final result in that zone's own view alone, no line's figures (T-2793: its riders
per line are that one zone's, so a long line would read near zero). A cached result shown beats it. During
a game load the focus pool (below) does this instead, to the zone's last round. After
an edit the zone in view's cluster sends its free-flow round as soon as it has it, which gives new
lines their first figures (T-324) without waiting for its last round.
**The city view's zone next** (T-2453, notes/T-2453.md, `?perfOff=selected` off). A zone selected in
the city view, when it is not the zone in view, comes second in each of those queues: its pack
downloads and opens right after the zone in view's (`firstZones`; a `first` message when it is
selected with its pack already queued), and a solve names it in `next`, so the worker orders the
clusters to solve the zone in view's, then its, then the rest, and with the barrier solves its
cluster alone first when the zone in view's needs no solving. **Its cluster goes ahead of the zone
in view's** (T-2459, `focused`, `?perfOff=focusAhead` off) when a solve with the barrier is sent while
the focus pool (below) has the zone in view in hand, solving or shown: the zone in view's figures come
from the focus pool to its last round meanwhile, so a selected zone in another cluster (an island, a
country beside a big one) shows from its own cluster's rounds instead of waiting for all the rest. The
loading card and the held clock then go when the focus pool's last round of the zone in view is shown
(or the solve's final, if sooner), as they go at the zone in view's cluster's final otherwise (T-421).
While a load's other figures are
shown (the focus pool's, the game before's), each round of a cluster solved ahead that has it
gives it that round's figures (`selectedRound`), and the zone in view the same when no focus pool
shows it (a small device, a focus pool that stopped; T-2476, `?perfOff=viewRound` off); the
cluster's lines and other zones still wait for its final. Its trips by length (the city view's graph, `tripLengths`) are asked where the views'
questions go (`target`): while a solve with the barrier steps through the rounds, of the round after
the one shown for a zone with no settled figures, whatever version the view is of; of the focus pool
for a zone shown from it. A worker past a round answers nothing of it, so the page asks for the city
view's zone at each round's start (`lengthsAhead`; a focus solve of that zone, of its round 0 as it is
sent, T-2476) and hands the graph the round shown's answer when it opens on it, with the next round's
asked at once (a question that waits for a round is kept by its solve and round, whatever view is shown
meanwhile; `?perfOff=lengths0` off); a round put in without the shown version or round
changing bumps `demandAskTurn`, and the graph and the top bar's hover ask again. On the Tokyo save
(Tokyo in view, Seoul selected, 2 workers) Seoul's figures and graph come with its cluster's round 0,
about 30 s after the load, where they came with the load's final at about 70 s. A zone selected while
a solve runs reorders the opens at once, but the solve itself only from the next one.
**The focus pool** (T-2045, `workers/demandFocus.ts`, notes/T-2045.md, `?perfOff=focus` off). A game
load with built stations starts a second pool, kept for that load alone (not on a phone, a tablet or a
device of 4 GB or less, `smallDevice`, where its workers would be a few hundred MB more in a tab of about
2 GB, T-2122):
one solving worker (`FOCUS_WORKERS`, `?focusWorkers=n` for more, never more than the main pool's;
each worker is about 0.1 GB during a load, and a second showed the zone's first figures no sooner,
T-2495), holding every period, and no aux worker of its own (`?perfOff=focusAux` gives it one): the
questions an aux worker answers go to its first solving worker, between its populations. It opens the zone in view's
pack (downloaded for it; T-2533: as soon as the save's stations are known to reach the zone, before the network
is in) and, once the network is in, solves that zone as a rough solve does (no market, the free-flow round from
one period in parts and estimates), but through every round and the capacity fit, kept as solved
(`solve.keep`). The main pool goes on solving and opening meanwhile, so the zone takes about its own
time, however big the save (holding the main pool's workers meanwhile settled the zone little
sooner and put the load's final much later, notes/T-2045.md). Each round shows as the zone's figures and its lines' (`onlyLines`):
alone while nothing else is shown, else put in the shown result as a cluster solved ahead is, and
the load's own estimates keep them wherever the focus solve is as far on or done (`withFocus`). Its
last round counts as final for the status tag (`focusZones`), not for the card or
the held clock, which wait for the zone in view's own final (its cluster's, or the whole solve's),
except where the city view's zone's cluster went ahead of it (T-2459, above). A zone newly in view during the load supersedes the
solve in hand; the pool holds at most three zones (`FOCUS_MAX_ZONES`), the oldest closed first.
Questions about a zone shown from it go to its workers (`ask`, `target`, `cellModes`; the cells too
while the main pool has not opened the zone). It ends (its workers terminated, questions waiting on them
answered empty) when the whole solve's final lands, the network changes or a worker of it fails; the
main pool makes no rough solve while it runs. **With a cached result shown** (T-2533, `?perfOff=cacheFocus`
ends the pool there) it shows nothing (the cache's figures are final) and only answers: its solve stops at round 0 (no further `go`, so
the main pool has the cores), and the zone in view's questions go to that round (`focusAsked`; a question asked
before the focus solve is sent waits for it). It ends once the main pool's solve of the same network is sent and
the focus round 0 has answered what was asked of it (1.5 s after its last answer), or at once if that solve has a
round of the zone (`retireFocus`): the pool beside the main pool's solve held 1.0-1.3 GB more on Anita's North
America save. A question after that waits for the main pool's round, as before. **It ends sooner**
(T-2495, `retireFocus`, `?perfOff=focusEnd` off) once the main pool's own estimates are the view and,
for every zone it solves or shows, it has shown a round of a zone those estimates cover and is no
further on than they are, or has shown the zone's last round: on the real save at the main pool's
round 0, about half the load's time. The zones it settled keep its figures until the final
(`withFocus`, `focusZones`), and their questions go to the main pool's next round; a zone newly in
view after that waits for the main pool, as without a focus pool. The figures it shows leave
out the regional market's longer trips and the riders other zones put on shared lines, so the whole
solve's final moves them a little (notes/T-2045.md); that solve itself is untouched and its figures
are the same to the bit.
**The zone in view aside** (T-2605, `refineAside`, `?perfOff=refineAside` off). Outside a load, while a whole
solve without the barrier runs (`firm`), anything newer for the zone in view waits for that solve's final: the
zone gone fine (it covers the zone coarse) or an edit that touches the zone (held, T-1734). On a big game that is
a minute or more. So a focus pool of one worker is started for that zone alone, solving it as during a load with
the newest network (fine), each round shown as the zone's figures and bubbles, and as the figures of lines
the shown result has none at all for (T-2793: new ones only; a line whose stops changed keeps its shown figures,
carried onto the new stops). Every other line and station keeps the shown figures (`withZone`, `aside`):
from the zone alone their riders read lower than the shown ones (on Anita's day-383 save, Salt Lake City's two
lines 50,400 a day against 67,800; a 1,068 km Hikari 36,600 a day against 589,000, fares 98% down, T-2793),
however the edit moved them.
Those figures stay over any solve that covers the zone coarse or is of an older network, its final included,
and the pool ends when a solve of the main pool of that network shows the zone (its local result or final), or
at a newer edit. They carry the same lie as during a load; the main pool's solves are untouched.
**One period first** (T-1538, `?perfOff=estimate` off). A worker holds two or three of the five
periods (the pool is at most 3), and on a big network each period's free flow takes seconds. So
in a rough solve, and in that free-flow round after an edit, a worker solves its first period
and estimates its others from it (`DemandApi::estimate`, `EverydayApi::estimate`, a few ms a
population): the period's own network and graph, the first period's mode choice, loads and places
scaled by the two periods' trips (commutes by the periods' shares, everyday trips purpose by
purpose); the market, which has no estimate, is solved (accuracy: notes/T-1538.md). After an edit the estimates go out as the preview, then the worker solves those periods, which
replaces them whole, so the cluster's rounds and final are the same to the bit. A rough solve
crowds on from its estimates, and is never kept for a whole solve to reuse.
**One period in parts** (T-1543). With two or more solving workers, both
free-flow rounds shown first are instead one period (the morning peak) solved by every solving
worker together, each a part of its origins:
the commutes' origin zones and the everyday layer's pairs dealt round-robin (`set_part(k, n)`,
`kernel::mode_choice_part`), every other period estimated from each part, and each part's figures
sent as a `result` with `part` for every period; the page adds a period's parts up (`addParts`).
Every figure is a sum over origins, so the parts add up to the whole period's: the trips and the
walking with no rail are the first part's, the market's figures (solved whole, in each worker's own
periods) its holder's, the local fare reach worked out again from the added sums (`fareRaw`), and
the bubbles' sums of a rough round sent by every worker over all five periods. Each worker then
drops the other periods and solves its own as before (a rough solve its first period and
estimates; after an edit every one), so the rounds after and the final are the same to the bit;
its own round-0 figures stand in for a period whose parts did not all come (a worker that kept a
period sends none) and change nothing once the parts are shown. Each part still finds every
station's strategies, so a part costs a little over half a whole period, not a third. While a solve steps through
its rounds, the bubbles' sums for a day not sent with the round or for one purpose, which only the
live state holds, are asked of the next round and answered on reaching it (T-327), and bubbles overtaken by a newer
round still show while the zone has none.
