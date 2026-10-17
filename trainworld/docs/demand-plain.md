# How demand works, in plain words

For Anita. One page, kept current: an agent who changes a rule here updates this page too.
Last updated 2026-10-17. The full rules are in demand.md; this is the gist.

## Who travels, and where to

- **Commutes.** Homes to jobs, from each country's census where it has commuting flows (New York,
  Tokyo, London...), or from a curve borrowed from the fitted cities where it has none (Lagos).
  Jobs come from official counts where they exist, otherwise from buildings, OSM places and
  population.
- **Everyday trips.** Errands, visits, school, university, medical, airports. How often people make
  them, and how far they go, comes from each country's travel survey.
- **Between cities.** A world market of longer trips by rail, car and air, plus commuters between
  neighbouring cities. Where two cities meet, commuting across the line will be damped (0.4).

## How each trip picks a mode

Every option gets a cost in **perceived minutes**: real minutes, some counted heavier than others.
The cheaper option wins more of the trips, smoothly: 20 perceived minutes cheaper is about 73% of
the trips (a logistic curve).

**By train**
- Walk to the station: each minute counts 1.5.
- Wait on the platform: half the time between trains. (Counted 1.37 for the first 5 minutes and
  0.4 after; this is being dropped, so a wait will be plain minutes, T-585.)
- Ride: 1 a minute, stretched when the train is crowded.
- Changing trains: the wait for the next train plus the walk between platforms. No extra penalty.
  Nobody changes back onto the line they just left: on a full line people wait at their station.
- If the train is full when it arrives, the wait for one with room counts too.
- Fare: in the US $1 counts as 3 minutes.
- Or drive to the station: about 10 minutes to start, the drive, and parking there.

**By car**
- Road time from the distance and the place: fast highways in the middle, slow streets at the ends
  where it's dense (3 km inside Manhattan about 13 minutes).
- Driving counts 1.33 a minute.
- Parking: 3 minutes to park, plus a price where jobs are dense (none in suburbs, up to 50
  perceived minutes in the densest centres), plus congestion charges (Manhattan, London).
- No running or ownership cost yet; about $10 a day is planned at the deep calibration.

**Walking the whole way**
- 13 real minutes a km. The first 10 minutes count 1.5 each, the rest 4 each, so 1 km counts as
  27 minutes and 2 km as 79. Short trips walk; long ones don't.

**Homes without a car** choose train, walking or "other" (bus, taxi, bike), never the car.

## The constants

Some trips take the train more or less than their minutes say: habit, income, what a car is good
for. One number per kind of trip covers that:
- Commutes lean slightly to the train (+0.4); errands and visits lean away (-0.7), because people
  take the train for work far more than for shopping.
- Japan leans more to the train (+0.5 on top).
- Walking has its own constant, set so walking stays at the levels the surveys gave.

These were fitted on six cities' real networks (New York, London, Paris, Seoul, Tokyo, Sendai) so
that today's real systems carry about what they really carry.

## Why rail is hard to win in the suburbs

The walk to and from the station. Mineola to Hicksville, both on the LIRR: the car counts as 27
minutes, the train as 60, and 36 of those are the walks at each end. Rail gets 29%. Where parking
is paid, rail wins easily (Hempstead to Midtown 80%). With fast direct trains between every pair
of places, rail would take about half of all trips that don't walk (notes/T-551.md).

## Coming next

- T-585: waits as plain minutes.
- T-587: short commutes walk at their own distance, not 2 km zone centres.
- T-584: commuting between neighbouring cities damped at 0.4.
- Deep calibration, later: a car ownership cost (about $10 a day), whether the walk's 1.5 should be
  1, and each country's walking fitted to its surveys.
