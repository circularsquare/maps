# How demand works, in plain words

For Anita. One page, kept current: an agent who changes a rule here updates this page too.
Last updated 2026-10-17 (the T-636 model). The full rules are in demand.md; this is the gist.

## Who travels, and where to

- **Commutes.** Homes to jobs, from each country's census where it has commuting flows (New York,
  Tokyo, London...), or from a curve borrowed from the fitted cities where it has none (Lagos).
  Jobs come from official counts where they exist, otherwise from buildings, OSM places and
  population.
- **Everyday trips.** Errands, visits, school, university, medical, airports. How often people make
  them comes from each country's travel survey. How far they go is one rule for the whole world:
  the chance of going somewhere halves about every 0.9 km for errands, 1.0 km for
  visits and school, 1.4 km for medical, 2.4 km for university and 14 km for airports. Trips are
  kept short on purpose: real surveys have about half of everyday trips under 2 km, and short trips
  are the ones people walk.
- **Between cities.** A world market of longer trips by rail, car and air, plus commuters between
  neighbouring cities. Where two cities meet, commuting across the line will be damped (0.4).

## How each trip picks a mode

Every option gets a cost in **perceived minutes**: real minutes, some counted heavier than others.
The cheaper option wins more of the trips, smoothly. The curve is steeper than it was: an option
10 perceived minutes cheaper takes about 73% of the trips, 20 minutes cheaper about 88%. Past
60 minutes of driving a minute counts for less, so long trips are less clear cut.

There is no built-in liking for the train. Rail and the car are compared on time and money alone.

**By train**
- Walk to the station: each minute counts 1.5.
- Wait on the platform: half the time between trains, in plain minutes.
- Ride: 1 a minute, stretched when the train is crowded.
- Changing trains: the wait for the next train plus the walk between platforms. No extra penalty.
  Nobody changes back onto the line they just left: on a full line people wait at their station.
- If the train is full when it arrives, the wait for one with room counts too.
- Fare: in the US $1 counts as 3 minutes.
- Or drive to the station: 15 minutes to start, the drive (counted 1.33 a minute), and the cheaper of
  parking there or a 10 minute drop-off. Only for trips of 5 km or more.

**By car**
- Road time from the distance and the place: fast highways in the middle, slow streets at the ends
  where it's dense (3 km inside Manhattan about 13 minutes).
- Driving counts 1.33 a minute.
- Parking: 7 minutes to park on every trip, plus a price where jobs are dense (none in suburbs, up
  to 50 perceived minutes in the densest centres), plus congestion charges (Manhattan, London).
- Owning the car: $1.50 a trip plus $0.10 a km, at 3 minutes a dollar. A typical trip pays $2.20
  to $4.00, and a driver's day comes to about $10. Homes without a car pay none, and so do the
  long trips between cities.

**Walking the whole way**
- 13 real minutes a km, every minute counted 1.5. So 1 km costs 19.5 minutes, 2 km 39, 3 km 58.5.
  Walks over 10 km are not offered.
- Against the car, a walk gets a 10 minute head start: the car side leaves out getting to the car
  and walking from the space, and this stands for them. It is one number for commutes and everyday
  trips and for every country.
- Worked through: a 10 minute walk (0.77 km) against a 1 km drive in a dense district. The walk
  counts 15. The drive is 4 minutes at 15 km/h, counted 1.33, so 5.3, plus 7 to park, plus 4.5 for
  the $1.50, plus 0.3 for the $0.10 a km: 17.1. The walk is 10 ahead, so the difference is
  17.1 - 15 + 10 = 12 minutes, and 12 minutes is 77%. About 3 in 4 people walk, and 1 in 4 drive.
  At 0.5 km about 82% walk, at 1 km 72%, at 2 km 43%, at 3 km 18%. (With no train, free parking.)
  A train that is good for the trip takes its share from both.
- A commute's distance for walking is between actual homes and jobs (six of each per 2 km zone),
  not between zone centres, so a 1 km commute across a zone line walks like a 1 km one.

**Homes without a car** choose train, walking or "other" (bus, taxi, bike), never the car.

## Why rail is hard to win in the suburbs

The walk to and from the station. A train that is fast still has to be reached: a kilometre at each
end is 39 perceived minutes before the train has moved, more than most short drives cost. Where
parking is paid, rail wins easily (Hempstead to Midtown). The trip explorer shows any trip's real minutes and its effective time (the perceived minutes above) by train, car and walking, each with its sum in a hover.

## Anita's two tests

Both ask what rail could carry if a straight train ran between every pair of places. Walking
and "other" are left out of the comparison; the figure is rail out of rail plus car.
- **A**: 140 km/h, free, a 3 minute walk at each end, a train every 5 minutes. Target at least 85%.
  New York 86%, London 85.5%. As a share of all trips, walks and the rest included: 79% and 75%.
- **B**: 100 km/h, a 3 minute walk at each end, a fare of $1.50 plus $0.05 a km. Target at least 50%.
  New York 79%, London 79%. Of all trips: 71% and 67%.
- The real networks, for comparison: New York's rail takes 22% of rail plus car (17% of all trips),
  London's 31% (22%). The old constants gave 19% and 29% (14% and 19%). Short trips are where
  the real systems fall short: coverage, not the choice, holds them down.

Details and the settings that were tried: notes/T-636.md.

## Coming next

- T-584: commuting between neighbouring cities damped at 0.4.
- Deep calibration, later: whether the walk's 1.5 should be 1, and what the real cities say about
  the new numbers (the model carries about three times New York's station counts, twice London's).
