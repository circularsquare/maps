[Back to the spec](../SPEC.md)

## 4. Demand: local trips

### 4.1 Where trips come from

A **doubly constrained gravity model** from homes to jobs with a distance-decay curve. Tiers say
how good the inputs are. **All nine registered Northeast cities have playable tier 1 packs;
global tier 0 is planned**:

| Tier | Inputs | Status |
|---|---|---|
| 0 | population grid + a jobs proxy (GHSL non-residential built-up, 100 m, global, free) | build |
| 1 | population + real jobs counts by small area | build |
| 2 | a real home-to-work OD matrix | not an input; used as a **check** on the tier-1 fit where it exists (LODES OD for the US) |

Subway Builder also uses gravity: LODES gives it homes and jobs, then a distance-based gravity
model picks each commuter's workplace (`research/competitors.md`). Tier 1 done well is at its level.

**Jobs data is real job counts at the place of work**, not a proxy, where it exists:

- US: LODES WAC (Workplace Area Characteristics), jobs per census block by sector and earnings.
  Covers wage and salary jobs under unemployment insurance plus federal civilian; leaves out the
  self-employed, military and informal work. Block figures carry deliberate noise, and some
  employers' jobs are placed by a model among their sites, so aggregate a little before trusting
  one block.
- Japan: Economic Census 2021 mesh, workers per 500 m square.
- England and Wales: Census 2021 workplace population by output area. **Caution: the census was in
  March 2021 lockdown**; many people working from home count at home. Usable for shape, biased
  against office districts.
- Australia: destination zones; Italy: 2011 census sections; EU: JRC ENACT 1 km (2011).
- Everyone else, China included: tier 0.

Full list with grain and access in `research/demand-data.md`. Population for the US: 2020
decennial census by block. **The home end of commutes is where job holders live**, not everyone:
in the US, LODES RAC (jobs by home block, the same jobs WAC counts at work), so a neighbourhood
of students, retirees or children sends fewer commuters than its population (T-054: county
totals now within 10% of ACS commuters, the Bronx was 23% over). Elsewhere, population until a
source by residence is found.

**People who work at their own home make no commute and are not counted** (Anita). LODES places
a remote worker at the employer's office, so commute trips are scaled down by the share working
from home (US: ACS "worked from home", by tract at the home end, by industry at the work end). Jobs whose work is at other people's homes
(home care aides) are real commutes and are spread over where people live (T-014).

**Everyday outings (T-066, NYC first).** All resident population generates home-based shopping/
errands (0.3 outings/person/day) and social/leisure (0.2), each with an outward and return leg.
These are partial, provisional activity budgets, not survey estimates or a commute-share target.
Shopping attracts trips by LODES retail and other-service employment (CNS07+CNS19). Social
destinations mix independently normalized food/recreation jobs (CNS17+CNS18) and population,
50/50, including visits to friends/family. Employees are relative draw, never customer capacity.
The pipeline spreads original sector locations by the same block land-area method as population;
office/headquarters commute corrections do not relocate these customer weights.

Destination gravity is origin-constrained: zone attraction times exp(-distance/scale), normalized
per origin zone, with scales 3 km errands and 6 km social; same-zone distance is 1.04 km. Sixteen
deterministic stratified destination samples per resident cell/purpose integrate that distribution,
then choose an actual res-9 destination cell by attraction weight. This is an approximation to
continuous demand, not sampled individual travellers. It is stable across networks and reloads.
**Everyday trips below 150 m cell-centre distance vanish entirely**, with no replacement or
denominator contribution, including walking. The cutoff is limited by cell grain; real within-cell
journeys are not separately represented. Existing commuting behavior stays unchanged.
Local weight tapers by the complement of the regional market's smooth 40–80 km onset; trips at
80 km or more disappear from this layer. Only within-pack everyday outings are generated here.

After station access is rebuilt, eligible cell edges aggregate into purpose-specific subzone
pairs, retaining weighted length and walking shares. Outward and return mode/route choices run
independently on each period's directed graph with the same fares, access, road costs, loading and
frozen combined crowding as all other demand. MSA retains direction-specific rail probabilities
for cell, selection and period queries. City mode splits, line loads and fare income include these
trips. Local bubbles count **trip legs**, twice the former commute-person count before adding
everyday travel; their purpose/timing queries reconcile to those same endpoint totals. Commute
station catchments remain explicitly commuter counts. Source-neutral cell-weight names allow
other countries' destination inputs later; runtime enables this first slice only for NYC.

**School attendance (T-140/T-150, NYC metro public schools).** Inside the five boroughs,
DOE school points and building enrollment retain 844,956 pupils across 1,213 matched buildings,
97.9% of positive source enrollment (report dated 2026-09-09). Shared buildings count once.
Outside the boroughs, NCES EDGE/CCD 2023-24 adds 3,199 public schools and 1,849,674 enrolled pupils,
for 2,694,630 pupils in total. Spatial exclusion of the DOE area prevents overlap; distinct
NCES school IDs at shared coordinates retain their distinct enrollment. Operational physical
schools with positive known enrollment are included; virtual/unknown categories are omitted.
All eligible NCES enrollment in this metro matched existing cells, with one 237 m snap.
Missing enrollment/sites contribute no invented pupils. Source dates differ deliberately.

Origins use 2020 block residents weighted by ACS 2023 tract ages 5–17 plus two fifths of
under-fives, approximating ages 3–4, spread by block land area across all 25 metro counties.
Missing shares use full-state county FIPS fallbacks, or the metro age share for Connecticut gaps.
The pack contains about 3.87 million estimated child-home weights. Each destination cell generates
0.9 attendance outings per enrolled pupil on a typical school day. This attendance factor is
provisional. Sixteen deterministic home samples per school cell use child weights and a 2.5 km
distance-decay scale. Unlike customer trips, school demand is destination-constrained by
enrollment. The same 150 m exclusion, local-distance taper and directional assignment apply.

School attendance joins bubbles, purpose/timing inspection, fares, loads and crowding under
School (metro public). Private schools, universities and school attendance in other city packs
are not modelled. Estimated homes are not observed student journeys or catchment/grade assignments.
There is no school calendar, escort demand, school-bus choice or student-fare policy yet.
Source selection and international/university follow-ups: [education data](../research/education-data.md).
Existing school staff commutes remain in Work. Airport passengers and patients still need
facility/visit inputs; those unmodelled purposes are labelled explicitly. Parks currently have
only the general leisure proxy. Activity chains, visitor origins and cross-pack everyday outings
remain later work. Research: [T-138](../notes/T-138.md); implementation and checks:
[T-066](../notes/T-066.md), [T-140](../notes/T-140.md).

### 4.2 Time of day

Five periods: morning peak 6-10, midday 10-16, evening peak 16-20, evening 20-24, night 0-6. They
are the player's scheduling periods too (section 6.3: high demand in the peaks, medium midday and
evening, low at night), defined once in `sim/src/track/params.rs` (`PERIOD_HOURS`,
`PERIOD_LEVEL`) for schedules and demand alike.

Each commuter makes a trip to work and a trip home a day. Of the day's trips to work, 70 / 12 / 6 /
4 / 8% start in the five periods; of the trips home, 2 / 18 / 58 / 16 / 6% (judgement for now). A
trip home goes back by the station pair the morning's trip used, so a period's rail trips are its
share of trips to work on the morning's station pairs plus its share of trips home on the same
pairs reversed. Crowding is judged at each period's busiest hour: 1.4 times the period's mean hour
in the peaks, 1.15 midday, 1.3 in the evening, 1.5 at night. Everyday outings share these combined-
load busiest-hour factors; their five-period departure shares are provisional:

| Purpose/direction | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Errands outward | 10% | 55% | 30% | 4% | 1% |
| Errands return | 6% | 49% | 36% | 8% | 1% |
| Social outward | 5% | 35% | 42% | 17% | 1% |
| Social return | 2% | 25% | 43% | 27% | 3% |
| School outward | 94% | 4% | 1.5% | 0.4% | 0.1% |
| School return | 1% | 82% | 15% | 1.8% | 0.2% |

The inspector uses actual purpose/mode/period selection results. Bars show departures per hour,
dividing period counts by 4/6/4/4/6 hours; outward and return rows remain separate.

### 4.3 Mode choice and catchment

Per OD pair and period, reserve the short-trip walking share first, then compare rail with
driving for the remaining trips by **generalised cost** in minutes. Buses are out of scope
(Anita, 2026-10-09).

- rail = walk to a station + wait (half the headway, capped) + ride + transfers x penalty + walk
  from station + fare / value of time
- driving = door-to-door road time (below; a few constants, real road travel times are not a
  priority) + its cost

Rail share = logistic in the difference (0.05 per perceived minute), plus a **per-city rail
propensity** constant instead of income: it absorbs car ownership, parking, habit and income in one
number, set so a realistic network gives roughly realistic mode share. New York's is 0 (T-078):
rail and driving are judged on perceived time and cost alone. **Catchment is soft** (T-090): there
is no circle the player sees. A farther station makes the walk, and so the whole trip, worse, and
the station loses riders where that trip gets worse than driving. Value of time is per country so
fares compare across countries. The US uses $20/hour (3 perceived minutes per dollar, a gameplay
judgement). The base fare is charged once per trip, including transfers; the per-km rate uses
actual track length. Distance fare affects route choice as well as mode choice, and crowding
multiplies ride time without multiplying fare (T-067).

**Walking.** A share of all trips is reserved for walking by distance:
`w = 1 / (1 + exp((d - 1.7 km) / 0.4 km))`; rail and driving then compete for `1 - w`.
Thus `walk = w`, `rail = (1 - w) * r`, and `drive = (1 - w) * (1 - r)`, where `r` is the
rail-versus-road result. This prevents attractive rail or expensive parking from taking the
walk share of very short trips. Commutes use the gravity's 2 km zone-centre distance, so a short
trip across a zone boundary can still inherit a coarse walking share. Everyday purposes use the
sampled cell-centre distance. The curve gives 84% walking within one 2 km zone, 32% to the next
zone and 6% two zones away; the remainder is offered to rail and driving. With no rail 5.9% of
New York's commutes walk (ACS: about 6%). Judgement until fitted. notes/T-153.md.

Weights (perceived minutes per real minute), from Subway Builder's published constants where they
exist: riding 1.0, the fast walk to and from stations 2.0 for its first 10 minutes and 8.0 after
(below), walking between platforms on a transfer 1.39, platform wait 1.37, waiting at home before a
known departure 0.4, congested driving 1.33, plus parking cost and ~3 min to park.

**Driving** (T-090; `place_min_per_km`, `drive_min` in `sim/src/demand/kernel.rs`). The road is 1.3x
the straight line. Its first and last 2 km (at most half the road each) are local streets, the rest
main roads. Each place has a speed for both, by its density (commuters living and working per km²
of land in its 2 km zone): local streets 40 km/h and main roads 80 km/h in open country, both
falling towards 15 km/h where it is dense, half way at 5,000 per km². The middle of a trip runs at
the mean of its two ends' main-road minutes a km. So a 25 km trip between suburbs takes about 27
minutes (it took 70 at the old flat 28 km/h), Brooklyn to Midtown (10 km) about 30, and 3 km inside
Manhattan about 13. On the gravity's commutes this gives 26-37 minutes in the four big boroughs
(ACS car commutes 34-38) and 36-47 in the suburbs (ACS 27-33; the gravity's suburban commutes are
too long, T-072). Parking is charged at the destination by its job density, as before.

Route choice is not all-or-nothing shortest path. NIMBY Rails' best-known complaint is riders
waiting 50 minutes for a fast train that arrives one minute earlier. Wait time counts at its
weight and the crowding loop (4.4) spreads riders.

**Access and egress.** Riders spread over the stations they could use by a logit on total cost
(scale 0.2 per perceived minute: a station 5 minutes worse gets 37% of the better one's weight),
and rail as a whole is priced at the logsum, so a second usable station helps a little and a
slightly better one does not take everyone.

**One way to reach a station: a fast walk** (Anita, 2026-10-09, as Subway Builder does). Getting
to and from stations is a single access mode at both ends, a walk at roughly bicycle speed and
willingness: **15 km/h along a path 1.3x the straight line (5.2 minutes a straight km); the first
10 minutes (1.9 km) weigh 2.0 each, every minute after that 8.0** (T-090: a 10-minute ride to the
station is taken readily, a 20-minute one rarely). So a station 1 km away costs 10 perceived
minutes, 2 km 23, 2.5 km 44, 3 km 65, 3.5 km 86. Against driving, a home 1 km out has 59% of the
odds of one beside the station, 2 km out 31%, 2.5 km 11%, 3 km 4%: a station's draw is strong for
a couple of km and fades out by about 3-3.5 km, sooner where driving is easy and later where it
is hard (into Manhattan). Between stations the logit above applies (a station 1 km further than
another within the first 1.9 km gets 12% of its riders, beyond that under 1%).

Two **computational bounds** keep the work down and are set where little rail is left: stations
are looked for within 4 km straight line (`walk_cutoff_m`), and a pair of places is offered rail
only if its walks at the two ends together cost at most 80 perceived minutes (`pair_walk_max`).
On the real New York network 0.1% of rail trips start 3.5-4 km from their station and 0.3% have
walks of 70-80 together; for a lone line in Manhattan, the worst case, 3% start 3-3.5 km out and
the bounds cost it about 4% of its riders (notes/T-090.md).

There is no drive, park-and-ride or feeder-bus leg to a station: it was most of a lone Manhattan
line's riders and doubled suburban rail on the real network (notes/T-034.md), and one mode is
easier to read. Driving is only the alternative for the whole trip (Anita, 2026-10-09). Roughly
calibrated on the real New York network (section 9, notes/T-078.md, T-090.md).

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
   commuting jobs (4.1), decay by distance only: **f(d) = d^-1 x exp(-d / 29 km)**, fitted to
   LODES OD for New York (T-006; 4,700-6,200 zones). It does not depend on the network, so edits
   never rerun it. The pipeline solves it (Furness, the sim crate's own code run natively, ~3 s)
   and ships a zone id per cell and two factors per zone in the pack (notes/T-004.md); opening a
   city rebuilds zone trips in ~2 ms in WASM instead of solving them for 8 s. Zone trips spread to
   cells by commuters at each end.
2. **Access per cell**: the 6 nearest stations within the walk's 4 km bound (4.3), found by
   scattering each station over the cells near it. 7-13 ms native.
3. **Access subzones**: the cells of one zone that share their nearest station. Each carries its
   perceived walk to each of its stations, averaged over its cells as the mode choice weighs them
   (a logsum at its 0.05 scale: the walk's cost grows faster than its length, and the near cells
   carry most of the rail), used at both ends of a trip. Cells with no station within 4 km are in
   no subzone: their commuters drive or walk. 2,750 subzones on the hand-made network, 5,032 on New
   York's real one (2,056 and 3,702 with the 2.5 km edge before T-090); access and subzones
   together are rebuilt on every edit in 16-27 ms on the hand-made network and ~57 ms on the real
   one in WASM.
4. **Station-to-station times**: one reverse strategy search per destination station over station
   nodes plus one node per stop per direction per line (T-069). At each station, useful lines
   share their departure frequency: boarding costs the weighted wait (half the combined headway:
   the first 5 minutes at 1.37, beyond that at 0.4) plus their frequency-weighted downstream cost
   and a transfer penalty, taken off once. A line joins only if it improves expected trip cost;
   wrong branches and off, slow, expensive or crowded alternatives add no useful frequency when
   they worsen that cost. A rider leaving
   a train may walk to another station within 300 m straight line and board there (walk at its
   weight; walk edges start only after a ride, so walking between stations is never a rail trip).
   Snapshot stations within 100 m of each other are one demand station (side-by-side platforms of
   two lines; never two neighbours on one line). Destination strategies and boarding frequencies
   are kept for loading. Each destination's search is independent, so only stations someone can
   end a rail trip at are searched towards (those in the access lists of subzones with jobs,
   `egress_stations`): a population living near two stations of an 875-station network needs two
   searches. The tables are shared by the populations of a worker (`kernel::with_skims`, T-119):
   Philadelphia's, New York's and the regional market's graphs are the same once crowding is
   set, and a search is done once per period and round, not once per population.
5. **Mode choice per subzone pair.** The access logit factorises: each origin subzone gets one row
   over all stations, the logsum of its access options followed by every station-to-station cost
   (at most 6 options x stations multiply-adds); each pair then multiplies that row into its
   destination's egress weights, up to 6 multiply-adds, one ln and one logit per pair. Pairs whose
   two walks together pass `pair_walk_max` (80 perceived minutes, 4.3) are skipped without being
   visited: each zone's destinations are kept cheapest walk first, and the destination zones by
   their cheapest walk, so an origin stops at the first that is too far. That bound is what holds
   the pairs where the 2.5 km edge had them: hand-made network 4.2M pairs (4.1M before T-090;
   with the 4 km reach and no pair bound 9.3M), real network 13.8M (13.6M; 31.8M). What a pair
   reads about its destination is packed in visiting order. Rail trips go back onto access
   stations once per origin subzone and are kept by station pair, and by subzone and access
   station at each end for the demand views (T-084, notes/T-078.md). The views ask for them on
   demand, never pushed: commuters by mode per cell (`cellModes`), a station's catchment and far
   end (`stationRiders`), and where the commuters living or working in any set of cells go, every
   mode (`flows`, T-098 for the bubbles: the gravity's zone trips from the selection, rail along
   the kept station pairs, walk and drive the rest; totals exact and equal to `cellModes`, far
   rail by place blurred like `stationRiders`; 10-20 ms in WASM for one cell or thousands, the
   far cells holding the last 1% left out as a cut total; notes/T-098.md).
6. **Loading**: each destination's strategy is walked from farthest settled nodes towards it.
   Station flow splits by accepted departure frequency, retaining each line's individual segment
   loads and boarding/alighting counts (T-069).
7. **Crowding**: a penalty per segment from the loads, repeat 4-6, average the loads (method of
   successive averages). Each round costs as much as the first pass. The riders projects worked
   this out; read memory `reference_transit_crowding_assignment.md` (in
   `~/.claude/projects/c--Users-anita-projects-maps/memory/`) first: the penalty starts where seats
   run out, never saturates, and averaging is what stops it ringing. **Three rounds** (T-022): the
   first round leaves the segment that takes the overflow at its worst (963% of crush on the real
   New York network against 310% before it), and a player-sized network moves under 1% by the
   third. No crush cap: an overloaded line is the player's signal (notes/T-022.md). A round in
   which no segment is over its seats has the free-flow estimate's factors (all one), so it would
   solve the same pass again: it is skipped, with exactly the same result (T-119; the real
   network is crowded in every period, so this only shortens small networks).

**Cost.** One period on the 202,337-cell New York pack and the hand-made network, one stage (the
free-flow estimate or one crowding round), one thread: 0.25-0.3 s native, 0.27 s in WASM (0.23-0.24
before T-090's wider reach: the extra origin subzones each build their station row and visit the
destination zones, and the drive costs a little more per pair; a day is 15% longer in WASM). An
edit reruns everything except gravity. Periods are independent, so the game runs them in a pool
of workers (section 7): with three workers in Chrome the mode split updated 0.49-0.55 s after a
new network reached them and the third crowding round landed by 1.9-2.0 s at T-078; T-090 adds
about 15% (not measured in Chrome). Each worker holds its own copy of the city: 18 MB of WASM
memory open, about 23 MB after solving (notes/T-026.md). New York's real 867-station network is
13.8M pairs, 1.0-1.7 s a stage in WASM, 24-25 s for a whole day on one worker (20 s before T-090).

**Accuracy of subzones** against one subzone per cell with the same model (T-090, hand-made
network, free flow): rail total within 0.2%, station entries and exits within 5.0% and 6.3% (sum
of absolute differences), segment loads within 3.0%. The logit roughly halved the station error
of all-or-nothing choice at any grouping, which is why the 5-minute walk bands are gone; what is
left is the grouping (a subzone's walk is one figure for all its cells). A plain mean of the
cells' perceived walks put rail 3.3% under, since the walk's cost is convex.

**If a bigger network makes it slow** (pairs grow with the square of the subzone count): add
workers first. Skipping zone pairs under 0.2-0.5 trips cuts pairs 1.1-4x but drops long trips
(segment loads 3.5-5.4% off, T-020). **If there is time to spare**,
10-minute walk bands are the accuracy lever: 2.4x the pairs, every error about a fifth smaller.

The Northeast recomputes both city populations and the separate regional market on service or
fare changes; per-city dirty filtering remains future work. At each period's crowding round, the
worker freezes their combined segment flows and feeds that same vector to all three populations.
Capacities and busiest-hour factors agree, so shared crowding affects every rider. The free-flow
estimate lands when all five periods agree on version and round; three refinements replace it the
same way. A faster worker cannot mix a later round with another worker's first estimate. A newer
network supersedes work after the population in progress (a pass is up to a second of one call).

Longer-trip queries (T-118, SPEC 5.2) come from the regional API, not the local ones. Each period
keeps rail trips per directed zone pair and, per station, rail trips starting and ending there by
far zone (T-120; one f32 per station, direction and zone, about 7 MB a period on 875 stations), both from the same
assignment pass that loads the lines. `selection_flows` sums the market's zone pairs for the 600 m
squares a selection of local cells lies in (a partly selected square counts its selected share;
homes by population share of the zone's departures, jobs by attraction share of its arrivals);
`station_flows` reads the per-station row. Both answer by market (the inspector's places) and by
far zone (all modes and rail, starts and ends; the bubbles), then every zone's own x and y, which
are the same on every worker and so are not added up by the client. Both answer over the periods a
worker holds, versioned and round-aware like `flows`. Tests: place and zone totals equal the
selection's trips, rail never above all, zone positions are the zones', station rows equal entries
and exits. A period's regional solve costs about 8 ms on the small real-pack profile
(3.7 ms before).

Each solving worker holds both local APIs and the regional API; stations transform from the loaded
game's world frame (T-125) through longitude and latitude into each pack's own frame (the regional
market's is the New York pack frame). Line and station loads and fares include all
populations. The top bar's denominator and the cell, flow and station-catchment queries remain the
focused city's local commutes. Query XY is returned in world coordinates; cell and zone ids remain
pack-local. Focus changes reset caches, layers and commuter selections, and reject late replies,
including a switch away and back.

**The pool and the bubbles' chain** (T-119). The pool is half the logical cores less one, at most
5, at least 1 workers that solve, and one more that holds no period (SPEC 7). Periods go to workers
by index, so five workers solve a stage of all five periods at once. The bubbles ask for the
commuters by mode right after a round is published, when every worker is already part way into
the next round and cannot answer a question about the round before from its live state. So each
worker sends its subzone sums for a round (`sub_sums`, per city) unasked, from the moment all its
periods had that round and ahead of its last result for it; the client adds them up when the
round is published. The commuters by mode per cell (`cell_modes`) read only those sums and the
network's subzones, so the worker with no periods answers them whatever round the others are in.
Stations, selections and longer-trip queries still read a period's state and are answered only
from a round that every period of the worker holds, so they wait for the solve to finish (T-119
leaves them; a selection made during the solve gets its far end when it ends).

**Cost on the real combined network** (T-119: the 109-line New York save plus one Philadelphia
line, 875 demand stations, 5,039 subzones, 13.9M subzone pairs a period; native, one period, free
flow or one round, three populations): 1.75 s before. Strategy searches took 0.28 s for each of
the three populations (Philadelphia's has 35 subzones but searched towards all 875 stations); New
York's mode choice 0.81 s; the market's pair loop 0.06 s. Now 1.25 s: one search (0.29 s), the
mode choice, and Philadelphia's 2 destinations. Results are identical to floating point noise
(max difference 2e-16 of each array's largest value over line loads, boardings, entries and the
summaries of all five periods; the market's own round-0 search is separate because its
free-flow graph leaves the distance fare out, see TASKS).

In Chrome on this machine (16 threads), the real network plus the Philadelphia line:

| | before (3 workers) | 3 workers | 5 workers |
|---|---|---|---|
| free-flow estimate published | 4.5 s | 3.2 s | 1.7 s |
| round three published | 17.2 s | 11.0 s | 6.1 s |
| bubbles drawn | 17.4 s (after round three) | 3.4 s | 1.9 s |
| WASM memory per worker | 129-132 MB | 120-134 MB | 120 MB |

Each period's stage in a worker takes about 1.5 s (2.1-2.3 s before). Historical single-city
measurements above retain their context.

**Memory follow-up** (T-123). The regional JSON pack now parses into flat numeric rows, skipping
metadata; temporary rows are consumed before constructing pairs. Immutable pair and lookup
vectors release their unused capacity. This preserves all trip parameters and results. On the
current full-network fixture, opening the five solving workers plus the auxiliary uses
360 MB against 596 MB before; each solving worker after the solve uses 127–128 MB against
137 MB. Estimated pool memory falls by 46 MB (6.4%) to about 668 MB, including the auxiliary's
opening allocation. These are WASM linear-memory measurements, not browser RSS. Live network,
strategy and period arrays still set the full-network peak. Comparison: notes/T-123.md.
