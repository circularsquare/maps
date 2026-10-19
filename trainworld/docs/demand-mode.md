[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: mode choice and catchment (4.3)

The generalised cost, walking, driving, parking, access to stations and homes without a car.

### 4.3 Mode choice and catchment

Per OD pair and period, rail, driving and walking the whole way (below) compete in one logit by
**generalised cost** in minutes. Buses are out of scope
(Anita, 2026-10-09), except as part of "other", the choice of homes without a car (below).

- rail = walk to a station + wait (half the headway, capped) + ride + walks between platforms +
  walk from station + fare / value of time (a change costs its wait and walk, no penalty: T-557)
- driving = door-to-door road time (below; a few constants, real road travel times are not a
  priority) + its cost

Rail share = logistic in the difference (**`beta` 0.10 per perceived minute**: 10 minutes cheaper
is 73% of the trips, 20 minutes 88%; T-636), with **no rail propensity**: `asc_rail` is 0 in every
country, so rail and the car are judged on time and money alone. There is a rail constant for the
everyday layer apart from the commutes' (`everyday_asc_rail`, default 0, settable by `--set`), kept
at 0 as a tool for a later fit; there is no country constant (`api::COUNTRY_ASC_RAIL` is an empty
table). T-509 had fitted 0.4 (-0.7 for everyday trips, Japan +0.5) to six real networks; T-636
dropped the fit so that the game is sensible and building a line has a clear effect, and the real
cities are loose checks (notes/T-636.md).
**Cost damping** (T-495, `Params::mode_beta`): past 60 perceived minutes of
the car's trip T, the scale is 0.10 x sqrt(60 / T), so a minute counts for less on a long trip
(0.058 at 3 hours, 0.028 at 13); one rule for commutes, everyday trips and the regional layer
(docs/regional.md 5.1), T the car's perceived minutes door to door including parking, never the
rail's, so the scale does not change with the network. In the local layers only the choice
between modes is damped (the walks and station pairs behind rail's cost are weighed at 0.10, since
they are shared by every destination); the regional layer damps the whole rail trip, access
included. **Catchment is soft** (T-090): there
is no circle the player sees. A farther station makes the walk, and so the whole trip, worse, and
the station loses riders where that trip gets worse than driving. Value of time is per country so
fares compare across countries. The US uses $20/hour (3 perceived minutes per dollar, a gameplay
judgement). The base fare is charged once per trip, including transfers; the per-km rate uses
actual track length. Distance fare affects route choice as well as mode choice, and crowding
multiplies ride time without multiplying fare (T-067).

**Walking** (Anita, T-558) is a third mode in the same logit, at the same scale (damped on the
car's minutes, as rail): rail, the car and walking for those with a car (drivers parking free and
those paying for a space choose apart, as for rail), rail, walking and other for those without
(below). The walk's real minutes are its distance at 6 km/h along a path 1.3 times the straight
line (13 minutes a straight km); every minute weighs 1.5 (`walk_trip_w`, one rate), so 500 m costs
10 perceived minutes, 1 km 19.5, 2 km 39 and 3 km 58.5. Against the car (or other) a walk gets a
bonus of **10 perceived minutes** (`walk_bonus_min`; the logit's `asc_walk` is that times `beta`, so
1.0 at 0.10, and it follows `beta`): what a short drive costs beyond its road time, which the car
side does not otherwise carry (getting to the car, the walk from the space). One constant for
commutes and everyday trips alike, and for every country (T-636; before it the constant was 2.80 and
3.21, worth 56 minutes at 0.05, and the cost rose to 4 a minute after 10). Walks over 10 km straight
line are not offered (`WALK_REACH_KM`, a computational bound).

Commutes walk between cells (T-587, `Gravity::set_walk_fine`): each zone's homes and its jobs are
six cells (`walk_samples`), those where the zone's cumulative weight in pack order crosses 1/12,
3/12 ... 11/12, and a zone pair within 3 squares each way (`WALK_FINE_R`) is its 36 cell pairs,
each walking its cells' centre distance (a cell to itself 0.52 of its square's side). Its
commuters walk and take other with no rail at the cell pairs' mean shares; its walk in the choice
with rail is the one cost whose share against a driver parking free is that mean. Farther pairs,
whose cells are at least 6 km apart, use the zone-centre distance, and the drive between them
stays at the zone centres. Everyday purposes walk the sampled cell-centre distance (a pair of
subzones its samples' mean distance). Every walk's distance is times the walk factor of its ends' parts
of their zones where water lies between them (T-275, below), so a home across a river from its job is as far as
the walk round by a bridge. Walking with no rail is summed once (`Gravity::no_rail_by_zone` over
zone pairs for commutes, over the samples for everyday trips); a pair offered rail takes from walking rail's share times walking's share with no
rail, as the logit does (`ModeOut::rail_walk`, `Params::car_choice`). A good station near a short
trip can take walkers, and a long walk loses to the train.

Weights (perceived minutes per real minute), from Subway Builder's published constants where they
exist: riding 1.0, the walk to and from stations 1.5 (below), walking between platforms on a
transfer 1.39, waiting for a train 1.0 (half the headway, plain minutes: Anita, T-585), congested
driving 1.33 (also on a ride to a station), plus parking (below).

**Driving** (T-090, T-275; `place_min_per_km`, `drive_min_f`, `drive_factor` in
`sim/src/demand/kernel.rs`). The road is the straight line times the two zones' routed factor
(below) times 1.074, or 1.3x the straight line in a pack without routes. From each end it runs 2 km
on that end's local streets, then 8 km on that end's main roads, and anything left in the middle
is open highway at 100 km/h (T-370); a road too short for all that splits it evenly between its two
ends (at most half each for the local streets, then half the rest each for the main roads). Each
place has a speed for its streets and main roads, by its density (commuters living and working per
km² of land in its 2 km zone): local streets 40 km/h and main roads 80 km/h in open country, both
falling towards 15 km/h where it is dense, half way at 5,000 per km². So a dense end slows only its
own last 10 km: 3 km inside Manhattan takes about 13 minutes, Brooklyn to Midtown (10 km) about
30, 25 km between suburbs about 28, and 60 km from a suburb into a 30,000/km² centre about 71.

**Parking** (T-380, T-372; `Params::park_parts`). At the destination every driver pays `park_min`, 7
perceived minutes (T-636, T-637; the base of every car trip, with the ownership cost below), to park and any congestion charge there; a paid space costs `15 x ln(d / 3,000)` more per
one-way trip, capped at 50, with `d` the jobs per km² of land in the destination's 2 km zone (none
below 3,000: 18 at 10,000, 40 at 42,000, 50 from 85,000). A share `1 / (1 + d / 8,000)` of the
drivers to that zone park free (80% at 2,000, 50% at 8,000, 16% at 43,000). Rail's share is the mix
of two logit results, one against the free driver and one against the paying one, weighted by
that share, not one result at the mean price. Congestion charges (`kernel::CORDONS`, a place's
charging area and its charge, half the day's charge at 3 minutes a dollar) apply to the share of a
zone's jobs inside the area: Manhattan south of 60th Street 13.5 minutes ($9), central London 35.1
(£18); nowhere else. A ride to a station pays the mean over drivers there. Owning a car adds
`car_own_usd_trip` dollars a car trip (`car_own_min`, in `park_parts`' fixed part) plus
`car_own_usd_km` a km of the car's own road (`Params::car_drive`), at 3 minutes a dollar; a drive to
a station pays both on its own road, parked or dropped off; homes without a car and the longer
trips (`market_params`) pay none. The shipped values are $1.50 a trip and $0.10 a km (T-621,
T-636), so a typical trip pays $2.20-4.00 and a driver's day about $10. The same rule serves all
three layers: the commutes and the everyday layer call `park_parts` (the everyday layer takes the
mix of two logit results per pair, `rail_share_parked`); the regional layer's market pack holds each
site's mean over drivers (`site_park`), written by `pipeline/intercity.py park_by_cell`, which
mirrors `park_parts` and `cordon_by_zone` and is used for every zone's pack (packs built before
T-636 hold the earlier base of 3 minutes until they are next rebuilt).

**Routed walking and driving** (T-275, notes/T-275.md). Walks and drives go round water and
over bridges, from data that exists for any city: OpenStreetMap roads and bridges (Geofabrik
extracts) and the pack's own water mask. `pipeline/routes.py <city>` adds them to a finished
pack, in place, and must be rerun whenever the pack's zones change (`build_city.py`, a gravity
re-solve); `routes.py --check` lists the packs whose routes are missing or stale. A pack without
them, or with them made for other zones, has every walk and drive straight x 1.3, as before.
- **Walks.** A barrier raster at 50 m (`walk_row`, `walk_x`, header `routes.walk`): a pixel is a
  barrier when 3 of its 4 water-mask pixels are water, every OSM way a person may walk on that
  crosses as a bridge (footways, paths, streets; not motorways, trunk roads only with `foot=yes`)
  is land, and islets under 0.1 km² are water. Stations are the player's, so the walk to them is
  searched in the game: from each station over the raster's pixels within its 3 km bound
  (eight neighbours, never across a barrier's corner), and a cell's walk is its straight line
  times how much longer the search's path is than the free path between the same two pixels,
  exactly the straight line where nothing is in the way, and none (no access) where water cuts
  it off within the bound. The cells whose walk differs are kept per station position for the
  worker (`routes::station_walks`), shared by a city's commute and everyday populations, so an
  edit searches only the stations it moved or added. The walk to and from stations, the
  access subzones and the trip explorer all use these distances. For walking the whole way, each
  zone's cells split by bank into at most 4 parts (`walk_part`, T-700): a part is the cells its
  representative (the cell nearest the part's weighted centre) reaches by a walk of at most 1.5
  times the straight line plus 500 m, the next part starts from the cells left. The walk's factor
  between parts within 2 squares (`walk_zone`) comes from the same search run from each part's
  representative over the other part's cells, both directions averaged; a cell's walk takes the
  factor of the two parts its ends are in (a subzone the part most of its people are in), so a
  walk along one bank of a zone astride a river goes straight.
- **Drives.** Per zone, the routed factor (road metres over straight metres, commuter-weighted
  over the other zone's cells) to every zone within 6 squares (`drive_near`), and between
  supercells of 4 x 4 squares per region beyond (`drive_far`, `drive_super`). The roads are
  every public road but service ways; each place joins its nearest road point whose straight
  line crosses no water, at 1.3x that line. The speeds stay the density model's above: only the
  road's length comes from the routes. The factor is multiplied by `road_routed_scale`,
  1.3 / 1.210: shortest-distance routes come to 1.17-1.26x the straight line, commute- and
  distance-weighted (New York 1.210), while the density speeds were fitted on New York with 1.3,
  so New York keeps its fitted level and other places their difference from it (and the longer,
  faster roads drivers really take are covered). Hoboken to Midtown is now 7.5 km by road (5.0
  before), St George to Wall St 22 km (12), Stamford to Huntington 83 km (30).
- **Not routed**: walking transfers between stations (300 m and complexes) stay straight x 1.3,
  because every population must build the same graph from the same stations to share its
  strategy searches (T-119) and none of the real New York network's 80 pairs within 300 m has
  water between them; ferries are no walk or drive (T-280); the regional market's drives
  (T-276's redesign).
- **The map's walking minutes** (T-312, `app/src/game/walks.ts`): the closest-stations lists of
  the station inspector and the bubble hover walk the same barrier raster, read on the main thread
  from the pack's `walk_row` and `walk_x` (a Range request, about 250 KB a city) with a TypeScript
  copy of `WalkGrid::field` and `Field::eff` (same constants, same search), so a hover never waits
  for a demand worker mid-solve. Minutes = routed metres x `walk_detour` 1.3 / 6 km/h, the model's walk. It does
  not check `zones_check`: a pack whose routes are stale still has its barrier here. Change the
  search in `routes.rs` and `walks.ts` together.

**The interface** (for other work, T-276 included). Byte tables, factor = `exp(q / 64)`. A zone's
window slot is `(dx + W) * (2W + 1) + (dy + W)` for the other zone's square offset (dx, dy), W 6
for driving and 2 for walking; the pack stores, zone by zone, a byte for each zone of its window
that exists in slot order then region order (a square cut by a region border holds one zone per
region), and the sim expands them to slot x region-count + region. `drive_far` is
supercells x supercells, supercells numbered in order of their first zone. The header's `routes`
block holds the windows, the supercell size, the zone count and `zones_check` (a sum over zones of
position and region, `routes::zones_check`; the tables are used only when it matches).
**Lookup**: `Params::drive_factor(i, j)` (`Routes::drive_factor` x `road_routed_scale`) and
`Gravity::walk_cost_pair(z, i, j)`. Across pack edges nothing is stored; the pipeline's
`routes.drive_factors(lon, lat, extracts)` gives the same factor between any list of places from
the same roads (unscaled).

Route choice is not all-or-nothing shortest path. NIMBY Rails' best-known complaint is riders
waiting 50 minutes for a fast train that arrives one minute earlier. A wait counts its plain
minutes (T-585) and the crowding loop (4.4) spreads riders, from 70% of a train's crush load.

**Access and egress.** Riders spread over the ways they could reach a station (walking or riding
to each one) and the station to leave at, every pair of the two, by a logit on total cost, and
rail as a whole is priced at the logsum, so a second usable station helps a little and a slightly
better one does not take everyone. **The logit's scale follows the trip's length** (Anita, T-710):
what counts is a difference against the trip, up to a 60 minute trip; past that a minute counts as
on a 60 minute trip. The scale is `rel_gamma` 20 over the trip's rough rail length T = 25 + 3 x
its straight-line km (perceived minutes, `rel_t0`, `rel_tkm`), T held at 60 (`rel_tmax`), in four
bands of straight-line distance (`REL_EDGES`: zone centres for commutes, the pair's mean distance
for everyday trips): under 3 km 0.68 a perceived minute, 3-7 km 0.50, 7-12 km 0.37, 12 km and over
0.33. Of two options 1 minute apart the faster gets 66% under 3 km and 58% from 12 km; 5 minutes
apart 97% and 84%; 10 minutes apart 99.9% and 97%. Each band's exp table is offset by 60 / scale
perceived minutes, added back after the logsum, so every band keeps station-to-station costs from
0 to 129 / scale minutes in f32 (387 from 12 km). The regional layer keeps `station_theta` 0.2.

**Walking to a station** (Anita, 2026-10-14, T-371). A true walk at both ends: **6 km/h along a
path 1.3x the straight line, or round water by a bridge where one is in the way (above; 13 minutes
a straight km), each minute weighing 1.5** (`w_walk`, fitted on New York in T-372;
notes/T-372.md). So a station 500 m away costs 9.8
perceived minutes, 1 km 19.5, 2 km 39. Between walks the logit above applies (a station 500 m
further gets 0.1% of the nearer one's riders on a trip under 3 km, 4% from 12 km).

**Riding to a station** (Anita, 2026-10-14, T-295, T-371, T-389). Car, bus, bike or a lift are one
mode, the ride, called driving in the game. It is offered only at the home end of a trip of at
least 5 km (`ride_min_km`: shorter, the ride cannot beat driving the whole way), from the home
zone's home-weighted centre: for a commute, 5 km between its two zones' centres; for an everyday
trip, its own mean distance between home and activity cells, at the home end both ways (the trip
out boards there, the trip home leaves the train there). It costs **`p_ride` 15 + 1.33 x the drive
to the station + the cheaper of parking at the station and a drop-off, `d_drop` 10** perceived
minutes: the drive is the whole-trip drive's rule (below), from the home zone to the station's
zone, and parking is a destination's mean over drivers (Parking, above). In a suburb
a ride to a station 1 km off costs about 21, 3 km 26, 10 km 35, so past about 1 km it is cheaper
than walking and access flattens out: a long commute's rail cost grows slowly with the distance to
the station, as Anita's curve does. Which stations: within 15 km straight line (`ride_max_m`), the
cheapest station of each line, then those of the 6 cheapest lines (`ride_lines`), so a rider
skips the local stop for the line they want. Homes beyond every walk (3 km) are one subzone per
6 km square (3 x 3 zones) with rides only, from its home-weighted centre, keeping each zone's
trip total but taking the most populous zone's distance decay; the work or activity end walks
only. Car parks are automatic and unlimited. `p_ride`, `d_drop` and `w_walk` are fitted on the real
New York network (T-372: about 55-65% of commuter rail riders ride to their station).
The regional layer's rides (commutes between zones, inter-region trips at both ends) come from the
same function, `access_options` ([regional.md](regional.md) 5.3).

**Homes without a car** (Anita, 2026-10-16, T-511). Each cell carries the people living in a
household with no car (the pack's `nocar`, `pipeline/nocar.py`); a home subzone's share is their
share of its homes. In the commutes and the everyday layer that share of each origin's trips
cannot drive. It chooses among three by one logit (`Params::nocar_choice`), with the same
propensity on rail:
- **rail**, walking to stations at the home end too (no ride; where the home has rides, its rail
  is priced on the walks alone, and a home with no station on foot gets no rail for it);
- **walking the whole way**, at the walk's cost and constant as everyone's (Walking, above;
  T-558);
- **other** (Anita, 2026-10-16, T-513), standing in for bus, taxi and bike: the drive's road at
  12 km/h door to door (`other_kmh`, 5 minutes a km) plus a wait of 8 minutes
  (`other_wait_min`, plain minutes) plus `other_asc`, no parking. The speed and wait are
  fixed for every city (a city bus's average speed with stops, and half a 15-minute headway with
  some walk to the stop); `other_asc` is 10 perceived minutes (`kernel::OTHER_ASC`, set on New
  York's boroughs against ACS; T-509's joint fit of six cities kept it).

The scale is damped on other's minutes, as the car's on the car's. Rail's share of a pair's
trips is `(1 - c) r_car + c r_nocar`, c the home's share without a car; the people with a car
choose as above. Trips by other are counted with driving in every view (docs/ui-demand-views.md); the sim keeps
them apart (`summary` index 10, per layer: trips by other with no rail, from the zone pairs or the
everyday samples, less the no-car rail trips times other's share of the non-rail choice, which
in a logit is the same with or without rail).
The regional layer has no such group yet. Shares by country: the US by tract (ACS B08201),
England and Wales by LSOA (Census 2021 TS045), Scotland and Northern Ireland by data zone, Japan
by prefecture (AIRIA cars per household), elsewhere a national figure from a register, a household
survey or GDP per head (docs/demand-worldwide.md "Driving alternative"). A pack without `nocar` has everyone able to drive.

Two **computational bounds** keep the work down and are set where little rail is left: stations
are looked for within 3 km straight line of a walk (`walk_cutoff_m`), and a pair of places is
offered rail only if its access and egress together cost at most 80 perceived minutes
(`pair_walk_max`; a 3 km walk costs 59).

Driving is the alternative for the whole trip. Roughly calibrated on the real New York network
(section 9, notes/T-078.md, T-090.md, T-371.md).
