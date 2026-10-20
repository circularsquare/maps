[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: workers, zones and clusters (4.4)

Which packs a worker holds, bare zones, warm-up, the worker pool and clusters.

**Which cities are open** (T-149). A worker holds only the city packs the game needs: the viewed
city and every city with a built station inside its footprint or within 4.5 km of it (the 3 km
walk plus a margin for edge cells and pack frames, `app/src/workers/cityReach.ts`). The ride's
15 km does not widen it: a pack's trips run between two stations of its own pack and the work or
activity end walks, so a station only a ride from the pack carries none of its riders (T-394; the
regional market carries riders between zones). Any
other city has no subzone with a station, so all its trips walk or drive and it adds nothing to
any line's load, fare or crowding: leaving it unloaded changes no number elsewhere. A worker
check solved a Midtown line with all nine packs open and with New York alone; every line load,
boarding, crowding figure, New York and market summary and corridor was bit-identical, and the
other eight cities had zero rail (notes/T-149.md). Packs arrive in `open` messages after the
WASM starts, the viewed city's first; a solve covers the cities open when it starts, and the
client sends one only when every city its network needs is open in every worker. A city that
construction reaches later joins at the next solve, which the client sends at once; a city only
viewed or on screen opens bare instead (below). Opening is cheap for a commute
pack (16 ms New York, 1-7 ms the others, natively); the cost is each city's everyday purposes
(natively New York 0.6 s, Washington 0.3 s, Philadelphia 0.2 s, Boston 0.2 s, Baltimore 0.1 s,
the four small ones 0.02-0.06 s, since T-397's sample counts; in Chrome, measured before T-397
with about twice the samples, New York 1.3-1.9 s in each of five workers, Philadelphia 1.1-1.2 s
and Washington 1.7-1.8 s). A zone's **market pack** (T-377, `<id>.market.json`) opens in the
solving workers once the zone is viewed, beside the view or reached, or a built station lies
within 30 km of its box (T-396's rule, per zone); before that no solve waits for it or covers it,
and a solve sent without a market pack open since is sent again. The workers compose the market
from the packs they hold (`MarketPacks.market`) whenever one opens or closes. **Packs close when
nobody needs them** (T-301): a zone that is
neither viewed, beside the view, reached by a station nor part of the load in hand closes in every
worker (bare or not) once it has been so for 60 s, or at once when more than two such zones are
open (the least recently needed first); a zone's market pack closes 60 s after the zone is in none
of those sets and no station is in its reach. A closed zone carries no rider on any line, so the shown
figures stand (a bare zone's city figures stay shown); the next solve leaves it out, and a worker
solving with it closes it when that solve ends. It reopens as it first opened (download, mostly
from the browser's cache, and opening). The bare copy of a zone that a shown solve covers closes at
once. WASM memory never shrinks, so closing bounds a worker's peak (the freed space takes the next
pack) rather than lowering it. `?perfOff=close` keeps everything open. A new game
opens only New York; a save with construction in several cities pays
each of their openings once, in every worker. Only the cells with a budget for a purpose hold its
destination samples and their pair indices (T-395; about half of all cells x purposes have none),
and a cell holds one sample per 8 of its trips (T-397): natively New York's everyday layer holds
2.3M samples, about 18 MB with their pair indices. A sample's
pair index is held only once a network gives its layer a pair, written in place on the next
network, and the pairs are found through a 4-byte-a-slot table sized from the last network's count
(T-398). A city's everyday layer is opened beside its commutes (`EverydayApi::beside`, T-399) and
shares their cells, zone grid, gravity and routes; its own are the zones' sums of residents and
jobs, and the station access and subzones built on those weights (not the commute weights), so
each network sets up the two layers' subzones separately. Natively a solving worker on the real
save holding three periods has about 156 MB of heap after a solve and 188 MB at its peak; with
all nine Northeast cities open, 282 and 318 MB (notes/T-399.md).
T-172 made the everyday opening cheap: zone centres sit on the zone grid, so the destination
gravity's decay comes from one table per purpose by grid offset instead of a square root and an
exp per zone pair (the sampled destinations agree with before to 7 significant figures in every
purpose's trips, walk share and trip length), and the long loops of the opening and of
`set_network` run in functions called once per zone or per 16k samples, because a browser
optimises a WebAssembly function only between calls (a single long call stays in the baseline
compiler throughout: in Chrome that made the opening 3-5x slower than natively). Startup
timings: notes/T-172.md, notes/T-174.md.

**Bare zones** (T-325, T-336). A zone viewed or on screen beside the view that no built station
reaches opens **bare**: its pack goes to the auxiliary worker alone, which sets its commute and
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
  which are final as they come (`holdDone`). New York's city figures and bubbles are identical to
  the bit to the full solve's; the network's own totals leave the bare zone out, as after a pan.
  Card gone 2.6-2.8 s after navigation (dev server, warm), against 6.6-7.1 s before; the bare
  opening is 2.0-2.2 s of it, 0.4 s more without the everyday warm-up, which stays. A saved game's
  network arrives after its viewed pack has downloaded (about 2.1 s against 0.65 s for the real
  New York save), so a save opens as before; waiting for the reach would cost every save that
  reaches its viewed zone about 1.4 s.
- **The first station** (T-351): with no station in the game, picking the track, station or
  blueprint tool opens the bare viewed zone in every solving worker as well (`prefetchViewed`,
  2.4-2.6 s for New York), with no solve started for it, so a first line's solve need not wait:
  its first estimate came 2.0-2.1 s after the build, as when the zone was open everywhere; built
  with no tool picked first, 4.4-4.6 s. notes/T-351.md.
- A station built within reach downloads the pack again (the page keeps no copy) and opens it in
  every worker; the solve waits for it and covers it from then on. Its bare figures serve until a
  shown solve covers it. The auxiliary worker keeps its bare copy (WASM memory never shrinks).
- **Memory and time** (real New York save, five solving workers, a 13-step pan): each solving
  worker stayed at 354 MB, against 855-955 MB when every zone on the way opened in every worker
  and was solved (10 solves); the auxiliary worker grew from 76 to 477 MB; the pool from about
  4.6 GB to 2.25 GB. A bare zone's bubbles came 1.5-3.6 s after arriving, against a final result
  16.6-22.6 s after. A click on a bare zone's bubble drew its far end in 155 ms. Opening bare took
  0.7-1.4 s (Toronto, Philadelphia) in Node.

**Everyday warm-up** (T-173). Even so, five workers opening New York's everyday layer at the same
time from baseline code took 7-8 s each instead of 1.3-1.8 s in most page loads (one worker alone
1.6 s, three 4.1 s each: as if they shared one core; never with Chrome's
`--no-wasm-dynamic-tiering`; not the module cache, efficiency mode or the tiering budget). So
worker 0 opens a small synthetic everyday layer first (`warm.rs`, 22,500 cells, about 150 ms;
ready 0.36-0.66 s after navigation, about when New York has downloaded), and the client holds the other solving workers' city packs
until it is ready (`demandClient.ts` `held`). Compiled code is shared by a page's workers, so
they all open on optimised code: New York 1.3-1.8 s in every worker in 10 of 10 page loads,
against 3.4-7.3 s in 7 of 10 without (`?perfOff=stagger`, same build).

**The pool and the bubbles' chain** (T-119). The pool is half the logical cores less one, at most
5, at least 1 workers that solve, and one more that holds no period ([architecture.md](architecture.md) 7). Periods go to workers
by index (worker i of n: periods q with q mod n = i), so five workers solve a stage of all five periods at once, unless
that puts both periods of a demand level on one worker (2 workers: both peaks), when they are dealt
out level by level (high, then medium, then low: 0, 1, 4 and 2, 3), so an edit to one level's trains
has its two periods solved side by side (T-124; `periodsOfWorker`, `?perfOff=spread` by index alone). The bubbles ask for the
commuters by mode right after a round is published, when every worker is already part way into
the next round and cannot answer a question about the round before from its live state. So each
worker sends its subzone sums for a round (`sub_sums`, per city) unasked, from the moment all its
periods had that round and ahead of its last result for it; the client adds them up when the
round is published. The commuters by mode per cell (`cell_modes`) read only those sums and the
network's subzones, so the worker with no periods answers them whatever round the others are in.
Stations, selections, transfers, trips and longer-trip queries read a period's state, so a worker
answers them only at a round all its periods hold; a selection's far end spread (`flows`) reads
only the sums it is handed and answers at any round. While the running solve's estimates are
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
network the map does not show yet); others are answered at once ("Clusters" below). `?perfOff=ahead` turns all of this off. T-327, notes/T-327.md.

**Clusters** (T-326, `workers/demandClusters.ts`). Each worker splits the network into clusters
that share no rider and solves each on its own sub-network: the cluster's stations (in snapshot
order) and its lines, with their own route nodes, mapped back to the whole network's in every
result and answer. A cluster is joined up from populations, stations and lines: a line joins its
stations; a station complex joins its stations; stations within 1 km join (merged platforms are
100 m, walking transfers 300 m); a city joins every station within 30 km of its cells' bounding box
(the 4 km walk and the 15 km ride from a home group's centre); the regional market, when open,
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
corridor trips (which need no network) once. A market zone no station reaches is in no cluster;
with none reached the market is a cluster of its own with no lines, solved once and kept. Lines and
stations no population reaches are in no cluster and carry nobody. `?perfOff=split` joins the
market's zones into one node, as before T-409.
Every population of a cluster crowds on the cluster's own loads, which are the whole network's
there, since no other population puts a rider on its lines; so the figures are the same as one
solve of everything. A worker keeps what each population last solved, and a cluster whose
sub-network, populations and rounds are unchanged since keeps its result untouched: after an edit
only the clusters it touched are solved, and the others' figures are read from their populations
as they stand. Every worker splits the same way from the same inputs. `?perfOff=clusters` solves
everything as one cluster, as before T-326.

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
re-solves two periods (one at the low level), a new station or a fare change all five.
`?perfOff=periods` re-solves every period of a cluster solved.

After an edit (a solve without the round barrier) the clusters to solve go one after another, the
zone in view's first (T-393), each to its last round; each but the last is sent as soon as it is
done, as one result per period with that cluster's figures alone (`result.cluster`), and the page
puts its lines, their stations, its zones' views and, if the market is in it, the regional market's
into the shown result at once. The network-wide totals wait for the solve's final: once every
cluster is done, the solve's result for every round is the final figures, read once (after an
edit only the final is shown, and new lines take round 0's, T-324). A cluster's network
is set up only when its turn comes, so until then it still holds the shown result's. A worker keeps
the clusters of its last six networks, and answers a question about any of them at once while the
asked zone's cluster holds that network solved to its last round, whatever it is solving
meanwhile: a click in a zone whose cluster is kept, or already published, needs no wait. Questions
about a published cluster's zones go to its solve. The status tag and the top bar's "before your
last change" speak only for a zone still being solved (`plan`, `demandSolving`). With the barrier
(startup, a zone just opened) the clusters to solve go in two groups when there are two or more and
the zone in view's is one of them (T-402): that cluster alone first, through every round, each
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
