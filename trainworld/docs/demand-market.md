[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: the world market in the sim (4.4)

How the regional layer's market is held and measured; its rules are in regional.md.

**The world market** (T-377, [regional.md](regional.md) 5.1, docs/zones.md 4-5). Longer trips
follow T-376's rule: ground and tail departures, each over its own normaliser against the world
field, both stored per 6 km zone in the place's pack, so the departures to places with no zone
(`info()[4]`, by place `elsewhere(q)`, a selection's in its last two numbers) need no outside
world. The car's perceived minutes add 15 cents a road km and the destination zone's parking
(`site_park`: the local parking rule's mean over drivers at each site's 2 km zone, job-weighted), both halved for two occupants
(`routed_car`, a `Cost` of minutes and dollars). T-378: the plane (`air_cost`, [regional.md](regional.md) 5.1) is priced per pair of
places (`PairRule`: car or not, the flight's minutes and fare, the crossings, the routed drive, and
the plane's share of the trips not by rail over the departing place's income groups). T-1431: a
place's row carries its income groups (`fg`, from the app's `marketWorld`), and each zone pair's
choice is worked out per group with every mode's dollars at that group's minutes a dollar
(`PairRule::alt`), added up by the groups' trips; their rail fares need the stations' expected fare
km (`pair_km`) whenever a place has groups. **What is kept per
pair** (T-1204): a composition keeps the rules of the pairs with an open place at one end (each
open place's row and column, about 250 KB a place); any other pair's is worked out when asked,
which only `corridor_modes` does. The corridors' figures that depend on the places alone (each
place pair's longer trips and commutes a day, the departures to places with no zone, the totals:
`Corridors`) are worked out at a worker's first composition, about 0.3 s natively, and kept by
`MarketPacks` for every later one, since with the world's rows they are the same whichever packs
are open; the market's parts and compositions share them (`corridors_id`), so the worker makes the
corridor table once. Each period keeps its rail by corridor for the open places' corridors alone.
Before, every composition built all 6.6 million pair rules and dense corridor tables again: 450 MB
and 3-13 s in each solving worker, and 50 MB more for each period held. The figures are the same
to the bit (`t1204_signature` over four sets of places); a composition now takes 15-90 ms and
holds 12-25 MB beside the 34 MB kept. A served pair's rail share is `sum / (rest + sum)` against its zones' car time (or the
plane's, with no car), per income group, and the split of the rest by plane and car is the pair's
(`corridor_modes`), which the client applies to every figure it shows. Commutes between zones: in the home place's pack,
`commute_home` (its 6 km zone), `commute_place` and `commute_work` (the work place and its zone)
and `commute_trips`, from `bin/joint_gravity.rs --packs` by way of its group's
`data/work/<group>.joint_commutes.json` (pipeline/market_zones.py); at opening merged by zone pair, at `set_network` kept between served
zones (`cpairs`), and in each period's pass chosen
once per commuter on its home-to-work station pairs (rail against the car on the cost ratio as in
4.3, T-918; drive x 1.33 + parking at work; access weights at the damped logit's scale; the home
end with rides, [regional.md](regional.md) 5.3), outward legs at `TO_WORK[q]`, the trip home at `TO_HOME[q]` on the
same pairs reversed; their own assignment (`c_avg`) is the work purpose's share of the market's
loads (`commute_loads`; the worker moves it from longer trips to work), with their own corridors
(`commute_corridors`), parts (`part_summary`: longer trips and rail, then commute legs and rail)
and a tail on `selection_flows` (legs of commuters living in the selection by work zone, of those
working there by home zone). A station's far-zone rows hold both parts and take the longer trips'
weekend factor (a known approximation: commuters are counted 1.58x rather than 0.30x on a weekend
there, T-296). Measurements on the real New York network: notes/T-276.md.

**A superholiday's lean** (T-1373, [demand-timing.md](demand-timing.md) "Kinds of day"). On a
going-out day a pair of places' longer trips lean towards the smaller: those from place i to j are
times `1 + lean(i, j)`, `lean(i, j) = 0.4 (s_i - s_j) / (s_i + s_j)` (`LEAN`, `Market::lean`), and
j to i's times `1 - lean(i, j)`; a coming-back day takes the opposite sign. A place's size `s` is
its people times the square root of its people per 6 km zone (`Place::lean_size`), so a sparse
countryside zone or a small town counts smaller than its people alone (Anita: home trips by
population, with a boost towards rural areas, small towns and smaller cities; no birthplace
tables). A big city against a small town runs about 70/30 outward, two like places 50/50; a pair's
trips in all are kept. Each pass fills the longer trips' station to station trips times their
pair's lean beside their own (`od_r`, S x S floats, left out past 256 MB: `LEAN_OD_BYTES`) and
walks it on the same strategies as a fourth assignment (`r_avg`, averaged over rounds and cut by
capacity like the commutes'); `lean_loads(q)` gives it. Selections, a station's far zones
(the station's place: the zone that reaches it most) and the trip explorer weigh each pair by its
places' lean on those days.

**A composition opens without its zone pairs** (T-174, T-377). A market is the world (the
registry's frame, rule and every place's row) plus the packs of the places a worker holds
(`MarketPacks`, each parsed once). The figures that need all pairs (the day's trips, each
corridor's, the departures to places with no zone) come from the rows, the same in every
composition, so opening one parses its packs and visits no pair.
`set_network_ll` takes stations as longitude and latitude; each place's sites see them in the
place's own frame for access and rides (each place on its own, so its figures never depend on which
other packs are open), the graph in the first place's. Each zone end's access list holds up to 10
stations, its 8 best and then a slot each for track none of those is on (`access_list`,
T-1506, [regional.md](regional.md) 5.2), so the market's station-pair loop is at most 10 x 10 a
zone pair. Its rides reserve the same way one level down: past the 6 cheapest lines, up to 2 more
for track those miss (`access_options` with the stations' components, T-1511, regional.md 5.3). It builds pairs only between zones some
station serves whose stations share a track component (T-409: the rail graph's, every line and
walking transfer at any frequency; the commutes between zones follow the same rule) on one
landmass, in the composition's zone order, so every rail figure is the same sum; the longer trips'
pairs dealt in chunks (T-780, [regional.md](regional.md) 5.1), the commutes' all kept. All-mode trips
for a selection are computed row by row for its zones when asked, against every open zone and, by
place, every place not open (`selection_all_day`, over any periods: the worker asks it of the
composition of every pack it holds and adds each part's rail, `selection_rail_day`). Answers lay
out every place and every 6 km zone of the world (`zone_count`, the rows' zone counts in place
order); a zone not open has no figures of its own and sits at its place's centre. Positions the
app reads are in the market frame (the New York equirectangular one) whatever the place.

**The market's distances are on the ground** (T-128). Each zone centre goes through its pack's
frame to the unit sphere and zone pairs are measured by great circle (`zone_km`; the same as
`track::frame::great_circle`); far places by their centres'. The decay and the drive time follow.
An old single pack (`sites` with a market column: the Northeast's `data/packs/intercity.json`
from before T-1014, kept for the ignored tests that open it, and the tests' fixtures) opens as a
closed world over its own zones with the ground curve alone.

The Northeast recomputes every open city population and the separate regional market on service or
fare changes; per-city dirty filtering remains future work. At each period's crowding round, the
worker freezes their combined segment flows and feeds that same vector to all three populations.
Capacities and busiest-hour factors agree, so shared crowding affects every rider. The free-flow
estimate lands when all five periods agree on version and round; three refinements replace it the
same way. A faster worker cannot mix a later round with another worker's first estimate. A newer
network supersedes work after the population in progress (a pass is up to a second of one call).

Longer-trip queries (T-118, [regional.md](regional.md) 5.2) come from the regional API, not the local ones. Each period
keeps rail trips per directed served zone pair (T-174) and, per station, rail trips starting and ending there by
far zone (T-120; one f32 per station, direction and served zone, the only zones a rail trip reaches, T-398;
the market's whole period is about 7 MB on the real save), both from the same
assignment pass that loads the lines. `selection_flows` sums the market's zone pairs (each
selected zone's row computed when asked, T-174) for the 600 m
squares a selection of local cells lies in (a partly selected square counts its selected share;
homes by population share of the zone's departures, jobs by attraction share of its arrivals);
`station_flows` reads the per-station row. Both answer by market (the inspector's places) and by
far zone (all modes and rail, starts and ends; the bubbles), then every zone's own x and y, which
are the same on every worker and so are not added up by the client. Both answer over the periods a
worker holds, versioned and round-aware like `flows`. Tests: place and zone totals equal the
selection's trips, rail never above all, zone positions are the zones', station rows equal entries
and exits.

Each solving worker holds both local APIs and the regional API; stations transform from the loaded
game's world frame (T-125) through longitude and latitude into each pack's own frame (the regional
market takes longitude and latitude and puts them in each place's frame, T-377; its zone distances
are great circles, T-128). Line and station loads and fares include all
populations. The top bar's denominator and the cell, flow and station-catchment queries remain the
focused city's local commutes. Query XY is returned in world coordinates; cell and zone ids remain
pack-local. Focus changes reset caches, layers and commuter selections, and reject late replies,
including a switch away and back.
