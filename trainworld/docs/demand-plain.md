# How demand works, in plain words

For Anita. One page, kept current: an agent who changes a rule here updates this page too.
Last updated 2026-10-19 (the T-636 model, with T-710's route choice and crowding, T-717's cost of
changing, T-718's route choice between cities and T-720's walking bands). The full rules are in demand.md; this is the gist.

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
  A longer trip into another country is damped too: to half inside the EU, Schengen and the UK
  (London - Paris about 30,000 a weekday), and much more elsewhere, the more so the poorer the
  traveller's country (0.26 for the richest).
- **Weekends and holidays.** Saturdays and Sundays have their own, smaller weekend demand. A public
  holiday counts as a weekend day for the lines in its country (a line's country is its first
  stop's), so Thanksgiving runs a New York line on weekend demand and fares that day only.

## How each trip picks a mode

Every option gets a cost in **perceived minutes**: real minutes, some counted heavier than others.
The cheaper option wins more of the trips, smoothly. The curve is steeper than it was: an option
10 perceived minutes cheaper takes about 73% of the trips, 20 minutes cheaper about 88%. Past
60 minutes of driving a minute counts for less, so long trips are less clear cut.

There is no built-in liking for the train. Rail and the car are compared on time and money alone.

**By train**
- Walk to the station: each minute counts 1.5. To keep the sums quick, homes are priced in groups:
  those in one 2 km square that share their nearest station and are within the same 15 minute
  band of walking to it (0-15, 15-30, over 30) share one walk, an average leaning to the nearer
  homes. Without the bands a short line's group could mix 2 and 28 minute walks, and rail came out
  7-21% too high. The trip explorer works out its own two places exactly.
- Wait on the platform: half the time between trains, in plain minutes.
- No train for hours: people differ in how long they would wait for the next train rather than go
  another way, from half an hour to 6 hours (half of them up to about 1 hour 45). At each stop and
  time of day the game works out what share of that period's riders have a train coming within
  their limit, from the timetable. A stop where everyone does (any line with a train at least
  every half hour) is priced as above. Where only some do (after a peak-only line's last train,
  overnight before the first) boarding counts extra: 1 minute if 90% have a train coming, 7 if
  half do, 23 if a tenth do, and with nobody, nobody boards there. The boarding account in the
  timetable spreads riders the same way, so nobody is shown waiting hours for a train.
- Ride: 1 a minute, stretched once the train is 70% full: at crush load a minute counts 3, at 120%
  it counts 5.4. Below 70% nobody minds the crowd.
- Changing trains: a fixed 2 minutes, plus the walk from the platform you arrive at to the one you
  leave from (centre to centre, at an easy walking pace, each minute counted 1.39), plus the wait
  for the next train. Platforms 80 m apart add 2 minutes; across one platform, nothing. Nobody
  changes back onto the line they just left: on a full line people wait at their station.
- If the train is full when it arrives, the wait for one with room counts too.
- Fare: in the US $1 counts as 3 minutes.
- Which stations: riders share the pairs of stations near each end by how much slower each option
  is compared with the trip's length. Of two options 1 minute apart the faster gets 66% on a short
  trip and 58% on one of an hour; 5 minutes apart, 97% and 84%; 10 minutes apart, nearly everyone
  takes the faster. Past an hour a minute matters less still, with the square root of the trip's
  length: on a 4 hour trip 5 minutes apart is 70%. The trips between cities choose their stations
  the same way, their length being the drive's. At a station, riders take the first
  train that helps.
- Or drive to the station: 15 minutes to start, the drive (counted 1.33 a minute), the cheaper of
  parking there or a 10 minute drop-off, and the parking at home (below). Only for trips of 5 km or
  more.

**By car**
- Road time from the distance and the place: fast highways in the middle, slow streets at the ends
  where it's dense (3 km inside Manhattan about 13 minutes).
- Driving counts 1.33 a minute.
- Parking: 7 minutes to park on every trip, plus a price where jobs are dense (none in suburbs, up
  to 50 perceived minutes in the densest centres), plus congestion charges (Manhattan, London).
- Parking at home: where people live densely, the car costs something to keep there too, 0.4 of
  what parking costs at work by the same density rule (none under 3,000 people a km², 7 minutes at
  10,000, 16 at 40,000). This is why dense neighbourhoods take the train more; there is no
  "households without a car" figure.
- Owning the car: $1.50 a trip plus $0.10 a km, at 3 minutes a dollar. A typical trip pays $2.20
  to $4.00, and a driver's day comes to about $10. The long trips between cities pay none.
- Everyone can drive. "Car" in the game means the car and every other way of getting about (bus,
  taxi, bike).

**Between cities** (the car and the plane against the train)
- Car: the drive at 85 km/h on a road 1.25 times the straight line, plus $0.15 a km and parking,
  shared by two people. Across the Channel it also pays $150 a car and an hour for the shuttle or
  ferry; to Hokkaido $170 and 4 hours for the ferry (the Seikan Tunnel takes no cars).
- Plane, between cities 250 km apart or more that both have airports: the drive to each airport,
  90 minutes at the airports, the flight at 750 km/h plus 30 minutes (an hour more with no direct
  flight), and a fare of $100 plus 5 cents a km, so $125 at 500 km and $150 at 1,000 km.

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

**Some always take the train, some never do.** Where a train serves the trip, 5% of the people
not walking take it whatever, while the train costs no more than driving; that fades to nobody
when the train costs three times the drive, so a useless detour line gets no free riders. The same
the other way: 5% always drive while driving costs no more than the train, fading to nobody when
the train costs a third of the drive.

## Why rail is hard to win in the suburbs

The walk to and from the station. A train that is fast still has to be reached: a kilometre at each
end is 39 perceived minutes before the train has moved, more than most short drives cost. Where
parking is paid, rail wins easily (Hempstead to Midtown). The trip explorer shows any trip's real minutes and its effective time (the perceived minutes above) by train, car and walking, each with its sum in a hover.

## Anita's two tests

Both ask what rail could carry if a straight train ran between every pair of places. Walking
is left out of the comparison; the figure is rail out of rail plus car.
- **A**: 140 km/h, free, a 3 minute walk at each end, a train every 5 minutes. Target at least 85%.
  New York 86%, London 85.6%, Paris 87.7%, Tokyo 87.8%; Houston, all suburb, 75.7%.
- **B**: 100 km/h, a 3 minute walk at each end, a fare of $1.50 plus $0.05 a km. Target at least 50%.
  New York 79%, London 78.5%.
- The real networks, for comparison: New York's rail takes 20% of rail plus car (14% of all trips),
  London's 29% (19%); before the route choice was sharpened (T-710) and changes cost 2 minutes
  (T-717), 23% and 32%. Short trips are where
  the real systems fall short: coverage, not the choice, holds them down.

Details and the settings that were tried: notes/T-636.md; the route choice, notes/T-710.md.

## Coming next

- T-584: commuting between neighbouring cities damped at 0.4.
- Deep calibration, later: whether the walk's 1.5 should be 1, and what the real cities say about
  the new numbers (the model carries about 2.4 times New York's station counts, 1.6 times London's).
