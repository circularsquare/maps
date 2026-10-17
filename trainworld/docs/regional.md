[Back to the spec](../SPEC.md)

# 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far [T-117, T-276, T-377]: **inter-region
trips** (visits, business, leisure, 25 km and more) within and between every zone of the world,
and **commutes between zones**. The design and its reasons are in [zones.md](zones.md) and
notes/T-362.md; the kernel's side (`sim/src/demand/intercity.rs`: compositions, served pairs,
queries) in [demand-market.md](demand-market.md) "The world market" and after. Every registered zone is a
**place** of the market (`place` in the registry); place ids are stable and new zones append;
Trenton's (2) stays with no sites since Mercer joined New York.

## 5.1 The total travel market, then rail's share

Generate **total** all-mode trips, then let the player's network earn rail's share.

- **Sites**: population- and job-weighted endpoint sites at 600 m grain from every zone pack,
  grouped into zones of 6 km, each zone's in its own pack, `data/packs/<id>.market.json`
  (`pipeline/market_zones.py`; the Northeast's in the market frame, every other zone's in its own
  pack frame).
- **Inter-region trips** (T-376's rule, constants in `pipeline/longhaul.json`, copied into the
  registry's `market.rule`): a resident makes two kinds of departures a weekday, each shared by its
  own normaliser over the **world field** (T-375: Kontur H3 cells with country, GDP per head and
  landmass), never over the zone list, so adding a zone moves only the trips that went where it now
  is:
  - *ground*: `0.107 x s` departures, weight `a_j x b x onset(25-50 km) x max(d, 25)^-2.37`;
  - *tail*: `0.00473 x s x u^0.503 / 1.2413` departures, weight
    `a_j x u_j^0.75 x min(y_j / y_i, 1) x b x onset(300-700 km) x max(d, 171)^-0.703`;

  with `s = min(y / y_US, 1)^0.7` (GDP per head at PPP), `u` the people of the site's H3 r5 cell
  over 100,000, `a_j` people times `s_j` (inside a pack its `population + 2 x jobs`, rescaled to
  the pack's people), and `b` 1 within a country, else `0.26 x min(y_i / y_US, 1)^1.5` (the
  origin's income). Each 6 km zone's two normalisers are computed offline at its centre.
- **Pairs**: places whose centres are under **250 km** apart pair 6 km zone to 6 km zone at their
  own distance; farther places pair at the distance between the places' centres, so their 6 km
  pairs add up to the places' trips exactly, each end's share by its zones' departures over the
  normalisers and its attraction (DECISIONS 2026-10-15, T-377). Opposite totals are averaged;
  pairs under 0.01 trips a day each way are pruned. Distances are great circles [T-128]. Local
  outings taper over 40-80 km, so the two layers meet without a gap or double count.
- **Registry rows**: each place's row (`marketRow`: its zones' sums, centre, country, income,
  landmass, parking, its airports' drive times and direct flights, and its trips with every place
  under 250 km, every zone pair added up) gives any pair of places its trips and modes with no pack
  open; `bin/market_pack.rs` writes them from every place's pack.
- **Commutes between zones**: the cross-pack part of the joint commute gravity
  ([demand-kernel.md](demand-kernel.md) 4.4, [zones.md](zones.md) 4), solved over each group of packs whose
  outlines come within 50 km of each other (T-539), spread from 2 km gravity zones to the
  market's zones and kept where at least one a day travels between two 6 km zones (each home zone's
  dropped commuters given to its kept pairs). They travel at the local commute's timing (70/12/6/4/8%
  to work, the return 2/18/58/16/6% on the same station pair reversed) and 0.30 of a weekday's at
  the weekend.
- **What is solved**: pairs are built only between 6 km zones some station serves, whose stations
  share a track component, with a car or a plane to compete with: on one landmass (T-375's, fixed
  links joined), or far places that both have airports. Every other pair's trips are fixed: counted
  in the totals, shown, never in the solve. The market's day and its corridors (place pairs) come
  from the rows, the same in every part of it.
- **Elsewhere**: a place's or a 6 km zone's departures less its trips with every place are its
  trips with places that have no zone; shown, never solved, their modes not modelled.

**Mode choice.** Inter-region trips choose rail, car and plane in one logit at 3 perceived minutes
a dollar [T-378], damped by the trip's length as every trip is ([demand-mode.md](demand-mode.md) 4.3,
T-495): 0.10 x sqrt(60 / T) per perceived minute, T the car's perceived minutes over the zone
pair's distance, even across a sea where no car goes (so rail against air never depends on it), so about 0.058 at 3 hours by car and 0.028 at 13. The scale covers the whole
rail trip: each zone's access weights (built at 0.10, 5.3) are raised to the power of the pair's
scale over 0.10, so a long walk to the station weighs less on a long trip as well. On the
market's own costs rail then holds about 93%, 73%, 48% and 28% of rail and air at 2, 3, 4 and 5
hours on the train.

- *Car*: 15 minutes plus 1.25 times the great-circle distance at 85 km/h, plus 15 cents a road km
  and the destination's parking (the local rule, docs/demand-mode.md 4.3, baked into the market pack's
  `site_park`), both split over two occupants; no tolls (T-270). Only within a
  landmass.
- *Plane*: between places whose centres are at least 250 km apart (the far pairs) and that both
  have airports. Door to door: the drive to an airport at each end (the car's drive time,
  people-weighted over the place's zones), 60 minutes at the airports, the flight at 750 km/h on
  the great circle between the places' centres plus 30 minutes, 60 more with no direct flight, and
  a fare of $50 plus 9 cents a km; the best pair of the two places' airports. Not built by the
  player, no capacity, not drawn.
- *Rail*: the player's network. Fares, service, transfers and shared crowding affect it.

The car's and the plane's costs are by place (the car's at the places' centres and their mean
parking), so a pair's split of its trips not by rail between car and plane is the same whoever
builds what, computed once, and kept (`corridor_modes`); the solve weighs every mode but rail
against the zone pair's own car time with that split. A place's airports (`pipeline/market_zones.py`):
OurAirports' large and medium airports with scheduled service within 150 km of its centre carrying
at least 1M passengers a year, the biggest eight; a route of 50,000 passengers a year is a direct
flight (both from the flights project's FR24 file, read offline only). Commutes between zones
choose rail or car as local commutes do: 0.10 per perceived minute damped the same way, the drive
weighted 1.33 plus the work end's parking; their loads are the work purpose's [T-158]. Inter-region trips' period
shares are 20/35/25/15/5%; their weekend factor 1.58.

Not modelled yet: coach, ferries, the Channel crossing's price, induced travel, routed drives
between zones (T-279; routes inside packs are T-275's).

## 5.2 Where a long-distance trip starts and ends

Origin access is population-weighted; destination access uses population and jobs. Each site offers
nearby stations through the local layer's walk and, where it applies, rides (5.3). Up to eight
weighted station options per zone end feed the common strategy graph. Local feeder lines, transfers and the
intercity service receive actual loads and boardings. No express-only route or station-city
affiliation bypasses the player's local network.

**Seeing them** [T-118]: the local views cannot show them (a Philadelphia bubble's trips end in New
York, off screen), so a selected bubble set and a selected station list them by place and draw
them by far zone ([ui-demand-views.md](ui-demand-views.md) "Inter-region trips"). A selection of
local cells is mapped to the market through its 600 m squares: the sites were built from the same
cells, so each selected cell falls in one site's square; a square counts the selected share of its
people (homes) or attraction (destinations), and a site's share of its zone's departures is its
population over the zone's (of arrivals, its attraction over the zone's). The numbers are the
market's own, summed from its zone pairs; a place whose pack is not open is counted by place only
(a far one exactly, a near one as the places' trips times the selected zones' shares of their
place's departures and attraction). A station's riders come from the same assignment that
loads the lines, so they add up to its entries and exits.

## 5.3 Park-and-ride

The regional layer rides to stations as the local layer does ([demand-mode.md](demand-mode.md) 4.3, T-371),
through the same `kernel::access_options` [T-373]: from each 2 km zone of the market's sites
(their homes and jobs, as a local zone's, give the drive's speeds and the parking), the cheapest
station of each of the 6 cheapest lines within 15 km, at `p_ride` 15 + 1.33 x the drive + at the
station a cost that depends on the end:

- **Commutes between zones**: at the home end, the cheaper of parking at the station and a
  drop-off (`d_drop` 10), when the two zones' centres are 5 km or more apart (`ride_min_km`); the
  work end walks.
- **Inter-region trips**: at the home end as a commute's; at the far end a drop-off only
  (`d_drop`), since the traveller has no car there.

A site's access at each end is its walks (the mean of their weights at 0.10 a minute, as
before rides) nested with its zone's rides at the station choice's scale (0.2 per minute), so a
ride dearer than the walk adds little, and a site beyond every walk is reached by its rides alone.
Zones that only a ride reaches are served (5.1), so a lone intercity station draws on everyone
within the ride's reach. Player-built feeder lines take part in regional routing as before.

## 5.4 Size

A place's market pack holds its sites, its 6 km zones' normalisers and its commutes to other
places (New York's 3.4 MB, Toronto's 1.1 MB). The solving workers open it with the zone (viewed,
beside the view, reached, or a station within 30 km of its box) and close it as they close the
zone; a worker composes the market from the packs it holds. It runs in the same worker pool as the
zone packs, periods divided among workers. Each period advances all populations from one frozen
aggregate flow vector per crowding round. Details and checks: [T-117](../notes/T-117.md),
[T-276](../notes/T-276.md), [T-377](../notes/T-377.md).
