[Back to the spec](../SPEC.md)

# 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far [T-117, T-276]: **inter-region trips**
(visits, business, leisure, 25 km and more) within and between every zone, and **commutes between
zones**. The design and its reasons are in [zones.md](zones.md); the kernel's side
(`sim/src/demand/intercity.rs`: the `derived` block, served pairs, queries) in [demand.md](demand.md)
4.4 "The market since T-276" and after. Market ids are stable and new zones append; Trenton's id
(2) stays with no sites since Mercer joined New York.

## 5.1 The total travel market, then rail's share

Generate **total** all-mode trips, then let the player's network earn rail's share.

- **Sites**: population- and job-weighted endpoint sites at 600 m grain from every zone pack,
  grouped into zones of 6 km.
- **Inter-region trips**: each resident makes 0.107 departures a weekday, shared by attraction
  `population + 2 * jobs` x an onset from 0 at 25 km to 1 at 50 km x `max(distance, 25 km)^-2.37`
  among every zone **and an outside world** (Kontur population living in no zone, at the zones'
  attraction per person), so a share of trips leaves the map and adding a zone later moves only the
  trips that went where it now is. Rate and curve are fitted to NHTS 2017 Northeast weekday trips by
  ground of 50-1,000 km (`research/intercity/market_fit.py`). Distances are great circles between
  zone centres [T-128]. Opposite zone-pair totals are averaged; tiny pairs are pruned. Local
  outings taper over 40-80 km, so the two layers meet without a gap or double count.
- **Commutes between zones**: the cross-pack part of the joint commute gravity
  ([demand.md](demand.md) 4.4, [zones.md](zones.md) 4), spread from 2 km gravity zones to the
  market's zones and kept where at least one a day travels between two 6 km zones (each home zone's
  dropped commuters given to its kept pairs). They travel at the local commute's timing (70/12/6/4/8%
  to work, the return 2/18/58/16/6% on the same station pair reversed) and 0.30 of a weekday's at
  the weekend.
- **Offline figures**: what needs every zone pair (each zone's normaliser with the outside world,
  the day's trips, each corridor's, the departures leaving) is computed offline by the game's own
  code into the pack's `derived` block [T-174]. The game builds pairs only between zones a station
  serves (the only ones that can carry a rail trip), and a selection's or far zone's all-mode trips
  row by row on demand. The block names its formula and is ignored if stale (derived at opening
  instead, same result); `pipeline/intercity.py --derive` refreshes it in seconds.

**Mode choice.** Inter-region trips compete with driving through a logit at 0.025 per perceived
minute. The car takes 15 minutes plus 1.25 times the great-circle distance at 85 km/h, plus 15
cents a road km and the local layer's parking at the destination, both split over two occupants,
at 3 perceived minutes a dollar; no tolls (T-270). Commutes between zones choose as local commutes
do: 0.05 per perceived minute, the drive weighted 1.33 plus the work end's parking. Fares, service,
transfers and shared crowding affect rail; the commutes' loads are the work purpose's [T-158].
Inter-region trips' period shares are 20/35/25/15/5%; their weekend factor 1.58.

Not modelled yet: air, coach, induced travel, routed drives between zones (T-279; routes inside
packs are T-275's).

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
market's own, summed from its zone pairs. A station's riders come from the same assignment that
loads the lines, so they add up to its entries and exits.

## 5.3 Park-and-ride

The regional layer rides to stations as the local layer does ([demand.md](demand.md) 4.3, T-371),
through the same `kernel::access_options` [T-373]: from each 2 km zone of the market's sites
(their homes and jobs, as a local zone's, give the drive's speeds and the parking), the cheapest
station of each of the 6 cheapest lines within 15 km, at `p_ride` 10 + 1.33 x the drive + at the
station a cost that depends on the end:

- **Commutes between zones**: at the home end, the cheaper of parking at the station and a
  drop-off (`d_drop` 10), when the two zones' centres are 5 km or more apart (`ride_min_km`); the
  work end walks.
- **Inter-region trips**: at the home end as a commute's; at the far end a drop-off only
  (`d_drop`), since the traveller has no car there.

A site's access at each end is its walks (the mean of their weights at the trip's logit scale, as
before rides) nested with its zone's rides at the station choice's scale (0.2 per minute), so a
ride dearer than the walk adds little, and a site beyond every walk is reached by its rides alone.
Zones that only a ride reaches are served (5.1), so a lone intercity station draws on everyone
within the ride's reach. Player-built feeder lines take part in regional routing as before.

## 5.4 Size

The regional pack holds its sites, the outside world's points and the commute list (about 14 MB).
It runs in the same worker pool as the zone packs, periods divided among workers. Each period
advances all populations from one frozen aggregate flow vector per crowding round. Details and
checks: [T-117](../notes/T-117.md), [T-276](../notes/T-276.md).
