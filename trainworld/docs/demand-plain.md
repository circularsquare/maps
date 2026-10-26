# How demand works, in plain words

For Anita. One page, kept current: an agent who changes a rule here updates this page too.
Last updated 2026-10-24 (the T-636 model, with T-710's route choice and crowding, T-717's cost of
changing, T-718's route choice between cities, T-720's walking bands, T-938's car speeds and
ownership cost, T-1000's cap on how far jobs pull and its set share of long commutes, T-1055's
rule that every rail trip rides a train, T-1059's that it never walks further than walking
there, T-1091's flat near end of the commute curve outside the US, T-1092's trips into
town, T-1070's cap at the trains' room as T-1255's 30-minute queue, and T-263's attractions). The full rules are in demand.md; this is the gist.

## Who travels, and where to

- **Commutes.** Homes to jobs, from each country's census where it has commuting flows (New York,
  Tokyo, London...), or from a curve borrowed from the fitted cities where it has none (Lagos).
  Jobs come from official counts where they exist, otherwise from buildings, OSM places and
  population. Outside the US a job in your own 2 km square counts the same as one in the next
  square: those curves were fitted on whole boroughs and towns, and taken at face value they kept
  two to three times too many people walking to work in their own square (London 18.5% walking
  against the census's 9.4%). The US curve was fitted block by block and already gets that right.
  Countries built before this take it at their next rebuild or join.
  Every job is filled from homes nearby, but no place's jobs pull workers more than 20
  times harder than usual for its area: where the job estimate puts far more jobs in one desert
  town than people live near it, some stay empty rather than drawing people from hundreds of km
  away. On top of that, a set 1% of everyone's workers commute 50-150 km to jobs in other towns.
- **Everyday trips.** Errands, visits, school, university, medical, airports. How often people make
  them comes from each country's travel survey. How far they go is one rule for the whole world:
  the chance of going somewhere halves about every 0.7 km for errands and visits, 1.0 km for
  school, 1.4 km for medical, 2.4 km for university and 14 km for airports. Trips are
  kept short on purpose: real surveys have about half of everyday trips under 2 km, and short trips
  are the ones people walk. But one in five errands and visits goes into town instead: there each
  place counts by how much there is to visit times how tightly it is packed, and the chance halves
  only every 4 km. So the corner shop draws its street, and the dense centre, where department
  stores, theatres and galleries are, draws people from across the city. Those are the long
  everyday trips a train can win.
- **Attractions.** Museums, zoos, theme parks, temples, castles and the like, about 12,900 places
  worldwide from Wikidata, each with its visitors a year: Wikidata's own figure where it has one
  (the Met 6.5M, Tokyo Disneyland 15.1M), otherwise 10,000 for every Wikipedia language that
  writes about the place. Each has that many visits a day, half from homes and half from where
  visitors eat and stay, from across the city: the chance halves only every 10 km. New York
  gets about 100,000 visits a day, London 195,000, Tokyo 130,000. Small next to the city's
  trips, but long trips to single spots, so a station at the Met gets riders it would not have.
- **Between cities.** A world market of longer trips by rail, car and air, plus commuters between
  neighbouring cities. Where two cities meet, commuting across the line is damped to half (0.5;
  the US keeps its own: none inside a state, 0.23 across one). Errands, visits, school and doctor's trips cross the line
  as if it were not there: a place right on the Guangzhou - Foshan line gets about a quarter of its
  visitors from the other side. These come with the next rebuild of the data.
  A longer trip into another country is damped too: to half inside the EU, Schengen and the UK
  (London - Paris about 30,000 a weekday), and much more elsewhere, the more so the poorer the
  traveller's country (0.26 for the richest).
- **Weekends and holidays.** Saturdays and Sundays each have their own demand: on a Saturday
  shopping is up by 40% on a weekday and commuting is down to a third; on a Sunday shopping is
  about a weekday's, outings are up by two thirds and commuting is a quarter. A line's country
  decides which days those are (a line's country is its first stop's): Saturday and Sunday in most
  countries; in Saudi Arabia, Egypt and much of the Middle East Friday runs as the Sunday and
  Saturday as the Saturday (in Israel Friday the Saturday, Saturday the Sunday); in Hong Kong only
  Sunday is a weekend day. A public holiday, with the substitute days its country gives, runs as a
  Sunday, so Thanksgiving runs a New York line on Sunday demand and fares that day only; a few US
  holidays most people work through (Columbus Day, Veterans Day and the like) run as weekdays.
- **Long holidays.** Spring Festival, Golden Week, Obon, Chuseok, Thanksgiving, Christmas, Eid and
  a few more make a run of days off around them a superholiday (3 days or more). While people are
  away, local outings are 0.7 of a Sunday's (many have gone home) and attractions 1.5 times a
  Sunday's (visitors in town); commuting and longer trips stay a Sunday's. The rush out and back
  comes later (T-1373).

## How each trip picks a mode

Every option gets a cost in **perceived minutes**: real minutes, some counted heavier than others.
The cheaper option wins more of the trips, smoothly, by how many times cheaper it is. When the
train is twice as dear as driving it gets 1 in 9 of the trips, three times as dear 1 in 28; at the
same cost, half; half as dear, 8 in 9. The car is judged the same way, so the two sides mirror each
other. Because it goes by times rather than minutes, 10 minutes matter a lot on a short trip and
little on a long one. Past 3 times as dear the train's share falls away, and from 4 times nobody
takes it, so a pointless detour line gets nobody (the same for the car the other way round).

Walking the whole way takes its share first, by minutes: a walk 10 minutes cheaper than the
alternatives takes about 73% of the trips. Rail and the car split the rest.

There is no built-in liking for the train. Rail and the car are compared on time and money alone.

**By train**
- Walk to the station: each minute counts 1.5. To keep the sums quick, homes are priced in groups:
  those in one 2 km square that share their nearest station and are within the same 15 minute
  band of walking to it (0-15, 15-30, over 30) share one walk, an average leaning to the nearer
  homes. Without the bands a short line's group could mix 2 and 28 minute walks, and rail came out
  7-21% too high. The trip explorer works out its own two places exactly.
- Wait on the platform: half the time between trains, in plain minutes; at a station with a
  waiting room (a station upgrade the player buys) each of those minutes counts 0.8, so a few
  more people take the train there.
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
  for the next train. Platforms 80 m apart add 2 minutes; across one platform, nothing. Between
  two stations the player joined by a passage the walk is the straight line at 2 m/s, moving
  walkways: 300 m takes 2.5 minutes rather than 5.4. Nobody changes back onto the line they just
  left to go back the way they came: on a full line people wait at their station. Where a line
  comes back through a station (a loop on a stalk), people may change there onto the same line
  heading somewhere else, like any change.
- If the train is full when it arrives, the wait for one with room counts too, minute for minute,
  up to an hour.
- No more riders than the trains take: once the rest is worked out, everyone who would ride if
  the trains had room comes to the platform, and stays only if the trains due in the next 30
  minutes have room for them; going along the line, riders already aboard keep their places. The
  rest go by car or on foot instead, and pay no fare. A line short of trains so has full
  platforms and full trains all through the peak.
- Every rail trip rides at least one train: nobody boards only to walk off to another station
  or platform of a complex, so walking between two stations is never a rail trip and pays no fare.
- No train that walks further than walking there: a pair of stations is no option for a trip when
  the walk to the first, the walks between stations and platforms on the way and the walk from the
  last add up to more than walking between the two ends (over water, each metre of it counts as
  3 metres of walking). A trip left with no such pair has no train. It takes about 3% of rail trips away, nearly
  all under 2 km, to walking and the car.
- Fare: a dollar of fare counts more minutes where people earn less: 4.4 in the US, 5.3 in the UK,
  12.2 in India (by GDP per head, to the power -0.5). The fare is the player's curve of trip length
  times the country's fare scale (2.8 in the US, 1.0 in India, where the curve is set, or the
  player's own), paid once a trip; choosing between routes sees only its first stretch. At the
  default scales a fare weighs the same everywhere; poorer countries just pay less per ride.
- Which stations: riders share the pairs of stations near each end by how much slower each option
  is compared with the trip's length. Of two options 1 minute apart the faster gets 66% on a short
  trip and 58% on one of an hour; 5 minutes apart, 97% and 84%; 10 minutes apart, nearly everyone
  takes the faster. Past an hour a minute matters less still, with the square root of the trip's
  length: on a 4 hour trip 5 minutes apart is 70%. The trips between cities choose their stations
  the same way, their length being the drive's. At a station, riders take the first
  train that helps.
- Or drive to the station: 15 minutes to start (5 at a station with a park and ride, a station
  upgrade the player buys), the drive (counted 2.66 a minute, twice a minute of
  driving the whole way, so a short drive to the station is common and a long one rare), the car's
  cost per trip (below), the cheaper of parking there or a 10 minute drop-off, and the parking at
  home (below). Only for trips of 5 km or more. A drive of 10 km to a station counts about
  64 minutes in a suburb. The game offers the train only where getting to and from the stations
  costs 80 minutes or less, and for that limit the drive still counts 1.33 a minute, so long
  drives are dearer but not ruled out where streets are slow. The trip explorer's hover for the effective time by train lists each of these parts.

**By car**
- Road time from the distance and the place: fast highways in the middle, slow streets at the ends
  where it's dense (3 km inside Manhattan about 12 minutes).
- Driving counts 1.33 a minute.
- Parking: 7 minutes to park on every trip, plus a price where jobs are dense (none in suburbs, up
  to 50 perceived minutes in the densest centres), plus congestion charges (Manhattan, London).
- Parking at home: where people live densely, the car costs something to keep there too, 0.4 of
  what parking costs at work by the same density rule (none under 3,000 people a km², 7 minutes at
  10,000, 16 at 40,000). This is why dense neighbourhoods take the train more; there is no
  "households without a car" figure.
- Owning the car: $2.00 a trip plus $0.13 a km, at 3 minutes a dollar. A typical trip pays $2.90
  to $5.25, and a driver's day comes to about $13. The long trips between cities pay none.
- Everyone can drive. "Walking" also stands for cycling and short bus rides (one reason its share
  runs above the census's), and "car" for driving, taxis and longer bus rides. The trip explorer's
  Walk and Drive rows say so in their hovers.

**Between cities** (the car and the plane against the train)
- Car: between places up to 1,500 km apart, the drive is routed over a world map of roads, hills
  and water (motorways fast, mountain roads slow, inlets driven round); farther, 15 minutes plus the
  drive at 85 km/h on a road 1.25 times the straight line. Plus $0.15 a km of road and parking,
  shared by two people. Over open water with no road (Helsinki to Tallinn) it takes a ferry: 30
  minutes plus the crossing at 35 km/h, $50 plus 50 cents a km a car. Across the Channel it pays $150
  a car and an hour for the shuttle or ferry; to Hokkaido $170 and 4 hours for the ferry (the Seikan
  Tunnel takes no cars). No tolls anywhere, borders included.
- Plane, between cities 250 km apart or more that both have airports: the drive to each airport,
  90 minutes at the airports, 25 minutes until the flight (the train's planned wait below at a
  flight every 2 hours), the flight at 750 km/h plus 30 minutes (an hour more with no direct
  flight), and a fare of $100 plus 5 cents a km, so $125 at 500 km and $150 at 1,000 km, its
  dollars counted at the starting country's minutes a dollar (the fare line above).
- Train: a trip between cities is planned, so its first train's wait is half the time between
  trains, but at most 5 minutes plus a sixth of it, and at most 30 minutes, and the "no train for
  hours" extra does not apply: 5 at a train every 10 minutes, 15 at an hourly train, 25 at one
  every 2 hours, 30 at anything sparser. A change of trains on the way waits as on any trip, and
  commutes and outings between cities wait as local trips do.

**Walking the whole way**
- 13 real minutes a km, every minute counted 1.5. So 1 km costs 19.5 minutes, 2 km 39, 3 km 58.5.
  Walks over 10 km are not offered.
- Against the car, a walk gets a 10 minute head start: the car side leaves out getting to the car
  and walking from the space, and this stands for them. It is one number for commutes and everyday
  trips and for every country.
- Worked through: a 10 minute walk (0.77 km) against a 1 km drive in a dense district. The walk
  counts 15. The drive is 3.6 minutes at 16.5 km/h, counted 1.33, so 4.8, plus 7 to park, plus 6
  for the $2.00, plus 0.4 for the $0.13 a km: 18.2. The walk is 10 ahead, so the difference is
  18.2 - 15 + 10 = 13 minutes, and 13 minutes is 79%. About 4 in 5 people walk, and 1 in 5 drive.
  At 0.5 km about 84% walk, at 1 km 74%, at 2 km 44%, at 3 km 18%. (With no train, free parking.)
  A train that is good for the trip takes its share from both.
- A commute's distance for walking is between actual homes and jobs (six of each per 2 km zone),
  not between zone centres, so a 1 km commute across a zone line walks like a 1 km one.

## Why rail is hard to win in the suburbs

The walk to and from the station. A train that is fast still has to be reached: a kilometre at each
end is 39 perceived minutes before the train has moved, more than most short drives cost. Where
parking is paid, rail wins easily (Hempstead to Midtown). The trip explorer shows any trip's real minutes and its effective time (the perceived minutes above) by train, car and walking, each with its sum in a hover.

The player's answer is a bus (since T-1464). Anyone within 400 m of a bus route walks to it (their
distance to the route plus 100 m), waits and rides at 18 km/h to a station, and the change to the
train is free: one fare for the whole trip on the same curve. A home 1.5 km from a station walks 29
perceived minutes to it, or about 18 by bus. A bus is just another line, so trips by bus alone
count as rail.

## Anita's two tests

Both ask what rail could carry if a straight train ran between every pair of places. Walking
is left out of the comparison; the figure is rail out of rail plus car.
- **A**: 140 km/h, free, a 3 minute walk at each end, a train every 5 minutes. Target at least 85%.
  New York 92%, London 92%, Paris 93%, Tokyo 93%; Houston, all suburb, 86%. (Trips shorter than
  the two 3 minute walks, about 460 m, get no train: T-1059.)
- **B**: 100 km/h, a 3 minute walk at each end, a fare of $1.50 plus $0.05 a km. Target at least 50%.
  New York 82.5%, London 83%.
- The real networks, for comparison: New York's rail takes 23% of rail plus car (17.5% of all
  trips), London's 32% (23%). Short trips are where the real systems fall short: coverage, not
  the choice, holds them down.

Details and the settings that were tried: notes/T-636.md; the route choice, notes/T-710.md; the
curve by times cheaper, notes/T-918.md; the car's speeds and ownership cost, notes/T-938.md; the
trips into town, notes/T-1092.md; the cap at the trains' room, notes/T-1070.md.

## Coming next

- T-1163: the everyday trips across a city line and the 0.5 damping, in the data at the next
  rebuild.
- Deep calibration, later: whether the walk's 1.5 should be 1, and what the real cities say about
  the new numbers (the model carries about 2.4 times New York's station counts, 1.6 times London's).
