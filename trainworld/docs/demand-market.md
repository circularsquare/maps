[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: the world market in the sim (4.4)

How the regional layer's market is held and measured; its rules are in regional.md.

**The world market** (T-377, [regional.md](regional.md) 5.1, docs/zones.md 4-5). Longer trips
follow T-376's rule: ground and tail departures, each over its own normaliser against the world
field, both stored per 6 km zone in the place's pack, so the departures to places with no zone
(`info()[4]`, by place `elsewhere(q)`, a selection's in its last two numbers) need no outside
world. The car's perceived minutes add 15 cents a road km and the destination zone's parking
(`site_park`: the local parking rule's mean over drivers at each site's 2 km zone, job-weighted), both halved for two occupants
(`car_min`). T-378: the plane (`air_min`, [regional.md](regional.md) 5.1) is priced per pair of
places when the market is built (`PairRule`: car or not, the plane's share of the trips not by
rail, the weight of every mode but rail against the reference, the reference's minutes with no
car); a served pair's rail share is `sum / (rest + sum)` against its zones' car time (or the
plane's, with no car), and the split of the rest by plane and car is the pair's
(`corridor_modes`), which the client applies to every figure it shows. Commutes between zones: in the home place's pack,
`commute_home` (its 6 km zone), `commute_place` and `commute_work` (the work place and its zone)
and `commute_trips`, from `bin/joint_gravity.rs` by way of `data/packs/intercity.json`
(pipeline/market_zones.py); at opening merged by zone pair, at `set_network` kept between served
zones (`cpairs`), and in each period's pass chosen
once per commuter on its home-to-work station pairs (logit 0.10 damped as in 4.3, drive x 1.33 + parking at work,
access weights at that scale, the home end with rides, [regional.md](regional.md) 5.3), outward legs at `TO_WORK[q]`, the trip home at `TO_HOME[q]` on the
same pairs reversed; their own assignment (`c_avg`) is the work purpose's share of the market's
loads (`commute_loads`; the worker moves it from longer trips to work), with their own corridors
(`commute_corridors`), parts (`part_summary`: longer trips and rail, then commute legs and rail)
and a tail on `selection_flows` (legs of commuters living in the selection by work zone, of those
working there by home zone). A station's far-zone rows hold both parts and take the longer trips'
weekend factor (a known approximation: commuters are counted 1.58x rather than 0.30x on a weekend
there, T-296). Measurements on the real New York network: notes/T-276.md.

**A composition opens without its zone pairs** (T-174, T-377). A market is the world (the
registry's frame, rule and every place's row) plus the packs of the places a worker holds
(`MarketPacks`, each parsed once). The figures that need all pairs (the day's trips, each
corridor's, the departures to places with no zone) come from the rows, the same in every
composition, so opening one parses its packs and visits no pair (all 20 places: 0.66 s natively).
`set_network_ll` takes stations as longitude and latitude; each place's sites see them in the
place's own frame for access and rides (each place on its own, so its figures never depend on which
other packs are open), the graph in the first place's. It builds pairs only between zones some
station serves whose stations share a track component (T-409: the rail graph's, every line and
walking transfer at any frequency; the commutes between zones follow the same rule) on one
landmass, in the composition's zone order, so every rail figure is the same sum. All-mode trips
for a selection are computed row by row for its zones when asked, against every open zone and, by
place, every place not open (`selection_all_day`, over any periods: the worker asks it of the
composition of every pack it holds and adds each part's rail, `selection_rail_day`). Answers lay
out every place and every 6 km zone of the world (`zone_count`, the rows' zone counts in place
order); a zone not open has no figures of its own and sits at its place's centre. Positions the
app reads are in the market frame (the New York equirectangular one) whatever the place.

**The market's distances are on the ground** (T-128). Each zone centre goes through its pack's
frame to the unit sphere and zone pairs are measured by great circle (`zone_km`; the same as
`track::frame::great_circle`); far places by their centres'. The decay and the drive time follow.
An old single pack (`sites` with a market column: the Northeast's `data/packs/intercity.json`,
which the joint commute gravity reads and writes, and the tests' fixtures) opens as a closed world
over its own zones with the ground curve alone.

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
and exits. A period's regional solve costs about 8 ms on the small real-pack profile
(3.7 ms before).

Each solving worker holds both local APIs and the regional API; stations transform from the loaded
game's world frame (T-125) through longitude and latitude into each pack's own frame (the regional
market takes longitude and latitude and puts them in each place's frame, T-377; its zone distances
are great circles, T-128). Line and station loads and fares include all
populations. The top bar's denominator and the cell, flow and station-catchment queries remain the
focused city's local commutes. Query XY is returned in world coordinates; cell and zone ids remain
pack-local. Focus changes reset caches, layers and commuter selections, and reject late replies,
including a switch away and back.
