# How demand works, in plain words

For Anita. One page, kept current: an agent who changes a rule here updates this page too.
Last updated 2026-11-06 (the T-636 model, with T-710's route choice and crowding, T-717's cost of
changing, T-718's route choice between cities, T-720's walking bands, T-938's car speeds and
ownership cost, T-1000's cap on how far jobs pull and its set share of long commutes, T-1055's
rule that every rail trip rides a train, T-2265's that it never takes longer than walking there,
T-1091's flat near end of the commute curve outside the US, T-1092's trips into
town, T-1070's cap at the trains' room as T-1255's 30-minute queue, T-263's attractions, T-2276's
station buildings and T-2278's parking by income). The full rules are in demand.md; this is the gist.

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
- **Attractions.** Museums, zoos, theme parks, temples, castles and the like, about 12,700 places
  worldwide from Wikidata, each with its visitors a year: Wikidata's own figure where it has one
  (the Met 6.5M, Tokyo Disneyland 15.1M), otherwise a share of a budget: everyone living within
  25 km makes 6 outings a year, and the world's tourists see 0.6 listed places a trip, both shared
  by how often the place's Wikipedia page is read, no one place taking much over 5% of its
  neighbours' outings (Kiyomizu-dera 2.2M, Nagoya Castle 2.7M). China's estimates are times 5 and
  Korea's times 2.5, since few people there read Wikipedia. A theme park's own staff don't also
  draw everyday visits as if they were restaurants. Each has
  that many visits a day, half from homes and half from where visitors eat and stay, from across
  the city: the chance halves only every 10 km. New York gets about 280,000 visits a day, London
  410,000, Tokyo 380,000. Small next to the city's
  trips, but long trips to single spots, so a station at the Met gets riders it would not have.
- **Between cities.** A world market of longer trips by rail, car and air, plus commuters between
  neighbouring cities. Where two cities meet, commuting across the line is damped to half (0.5;
  the US keeps its own: none inside a state, 0.23 across one). Errands, visits, school and doctor's trips cross the line
  as if it were not there: a place right on the Guangzhou - Foshan line gets about a quarter of its
  visitors from the other side. Airport trips cross too: an airport in a county zone outside its
  city (Wuhan's, Istanbul's) takes its passengers from the city, up to the airport's published
  passengers and never more. These come with the next rebuild of the data.
  A longer trip goes to a place by its people and jobs; in Japan, about 80 resort towns (Hakone,
  Atami, Kusatsu, Beppu and the like) also count 4 people for every visitor a day in their published
  tourism figures, so Tokyo's day trips and weekends away go to Hakone and Izu (Hakone about 9,100 a
  weekday each way, eight times before). These come with the next rebuild of Japan's data.
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
  Every day runs the same trains: each line's high, medium and low trains in their own hours,
  weekends and holidays too, and their crowding is the day's riders on those trains.
- **Long holidays.** Spring Festival, Golden Week, Obon, Chuseok, Thanksgiving, Christmas, Eid and
  a few more make a run of days off around them a superholiday (3 days or more). While people are
  away, local outings are 0.7 of a Sunday's (many have gone home) and attractions 1.5 times a
  Sunday's (visitors in town); commuting and longer trips stay a Sunday's. On the day before (the
  two days before a run of 5 days or more) people go out, and on the last day (last two) they come
  back: longer trips between cities are twice a weekday's, and lean towards the smaller place going
  out and back to the bigger one coming back, up to 70/30 for a big city against a small town. A
  place counts smaller for being sparse, so countryside and small towns draw more of the trips home
  than their population alone would. Touristy places (many attraction visits for their people:
  Kyoto, Nara, the zone round a theme park) draw up to half as many again going out, and send as
  many more home coming back, up to 90/10 in all. Everything else is a weekday's going out and a Sunday's coming
  back. So a line from a big city out to the country fills one way on the going-out day and the
  other on the way back.

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

Islands no road reaches (Sado, Salamis, Hong Kong's outlying islands, Liberty Island) have no car and no walk off
them: the player's ferry or train is the only way. It takes every such trip unless it costs over 3
times a drive over a bridge that isn't there, and none past 4 times. The rest are "no route":
counted in the totals, by no mode. Car ferries don't count as roads.

### By train

- Walk to the station: each minute counts 1.5. To keep the sums quick, homes are priced in groups:
  those in one 2 km square that share their nearest station and are within the same 15 minute
  band of walking to it (0-15, 15-30, over 30) share one walk, an average leaning to the nearer
  homes. Without the bands a short line's group could mix 2 and 28 minute walks, and rail came out
  7-21% too high. The trip explorer works out its own two places exactly.
- Wait on the platform: half the time between trains, in plain minutes; at a station with a
  waiting room (a station upgrade the player buys) or inside a station building the player drew,
  each of those minutes counts 0.8, so a few more people take the train there.
- No train for hours: people differ in how long they would wait for the next train rather than go
  another way, from half an hour to 6 hours (half of them up to about 1 hour 45). At each stop and
  time of day the game works out what share of that period's riders have a train coming within
  their limit, from the timetable. A stop where everyone does (any line with a train at least
  every half hour) is priced as above. Where only some do (after a peak-only line's last train,
  overnight before the first) boarding counts extra: 1 minute if 90% have a train coming, 7 if
  half do, 23 if a tenth do, and with nobody, nobody boards there. The boarding account in the
  timetable spreads riders the same way, so nobody is shown waiting hours for a train.
- Ride: 1 a minute, stretched once the train is 70% full: at 100% (160 a car) a minute counts 3,
  at 120% 5.4, at 150% (240 a car, as packed as a train gets) 10.8. Below 70% nobody minds the
  crowd.
- Changing trains: a fixed 2 minutes, plus the walk from the platform you arrive at to the one you
  leave from (centre to centre, at an easy walking pace, each minute counted 1.39), plus the wait
  for the next train. Platforms 80 m apart add 2 minutes; across one platform, nothing. Between
  two stations the player joined by a passage the walk is the straight line at 2 m/s, moving
  walkways: 300 m takes 2.5 minutes rather than 5.4. Inside a station building the player drew,
  the fixed 2 minutes count 1, and the wait counts as in a waiting room; the walk is as anywhere. Nobody changes back onto the line they just
  left to go back the way they came: on a full line people wait at their station. Where a line
  comes back through a station (a loop on a stalk), people may change there onto the same line
  heading somewhere else, like any change.
- If the train is full when it arrives, the wait for one with room counts too, minute for minute,
  up to an hour.
- No more riders than the trains take: once the rest is worked out, everyone who would ride if
  the trains had room comes to the platform, and stays only if the trains due in the next 30
  minutes have room for them (a train takes up to 150%, 240 a car; a bus 80, 150% of its 53; a boat 480); going along the
  line, riders already aboard keep their places. The
  rest go by car or on foot instead, and pay no fare. A line short of trains so has full
  platforms and full trains all through the peak.
- Every rail trip rides at least one train: nobody boards only to walk off to another station
  or platform of a complex, so walking between two stations is never a rail trip and pays no fare.
- No train that takes longer than walking there: a pair of stations is no option for a trip when
  its real minutes (the walk or drive to the first station, the waits, the rides, the walks
  between stations and platforms on the way, the walk from the last) are more than walking the
  whole way takes. Plain minutes on both sides: no fare, no crowding, no weights, no fixed minutes
  for changing (over water, each metre of the walk counts as 3). A trip left with no such pair has
  no train; a trip with a quicker pair keeps its whole share, however close to walking. Only
  trips walking reaches (10 km) are checked. It takes about 8% of New York's rail trips away to
  walking and the car, nearly all short everyday ones.
- Fare: a dollar of fare counts more minutes where people earn less: 4.4 in the US, 6.0 in the UK,
  8.4 in Japan, 16 in China, 51 in India (by GDP per head in US dollars, to the power -0.7). The
  fare is the player's curve of trip length times the country's fare scale (by income, so a new
  game's fares start at $2.50 in the US, $1.30 in Japan, $0.22 in India wherever it starts, or the
  player's own; older saves keep their fares), paid once a trip; choosing
  between routes sees only its first stretch. On high-speed lines the fare above the base is 1.25
  times as much (or the player's multiplier), by the share of the trip's km on them, so riders weigh
  the dearer fare. A transfer fare the player sets ($0 to start) is paid at each change of line,
  times the country's scale too; people weigh it when picking a route and the train, and the line
  they change onto takes it. Trips between regions change free. At the default scales a fare weighs the same
  everywhere; poorer countries just pay less per ride. A new game's curve is about what real trains
  charge: in New York $1.50 plus $0.10 a km to 20 km, then $0.25 a km, so $3.50 at 20 km, $23.50
  at 100 km and $98 at 400 km; Beijing to Shanghai about $75 (a real ticket is about $83).
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

### By car

- Road time from the distance and the place: fast highways in the middle, slow streets at the ends
  where it's dense (3 km inside Manhattan about 12 minutes).
- Driving counts 1.33 a minute.
- Parking: 7 minutes to park on every trip, plus a price where jobs are dense (none in suburbs, up
  to 50 perceived minutes in the densest centres), plus congestion charges (Manhattan, London).
- Parking at home: where people live densely, the car costs something to keep there too, 0.4 of
  what parking costs at work by the same density rule (none under 3,000 people a km², 7 minutes at
  10,000, 16 at 40,000). This is why dense neighbourhoods take the train more; there is no
  "households without a car" figure.
- In poorer countries both prices count less: times (GDP per head / $35,000) to the power 0.5
  where a country is under $35,000 a head, so 0.28 in India, 0.38 in Vietnam, 0.63 in Mexico and
  China, the full price in Japan, Korea, Europe and the US. Real parking prices rise about in step
  with income up to there (Colliers' city centre prices: Delhi about $25 a month, Tokyo $530) and
  not beyond; a poorer driver also feels each dollar more, by income to the power 0.5 as with
  fares, which leaves the 0.5.
- Owning the car: $2.10 a trip plus $0.1365 a km, at 3 minutes a dollar. A typical trip pays $3.05
  to $5.50, and a driver's day comes to about $14. The long trips between cities pay none.
- Everyone can drive. "Walking" also stands for cycling and short bus trips (one reason its share
  runs above the census's), and "car" for driving, taxis and longer bus trips. The trip explorer's
  Walk and Drive rows say so in their hovers.

### Between cities (the car and the plane against the train)

- Car: between places up to 1,500 km apart, the drive is routed over a world map of roads, hills
  and water (motorways fast, mountain roads slow, lakes and inlets driven round); farther, 15 minutes
  plus the drive at 85 km/h on a road 1.25 times the straight line. Each neighbourhood has its own
  detour to the other city, so from the end of a lake the car goes round a short way and from across
  its widest part a long way (Toronto to Buffalo). Plus fuel and tolls for each km of road
  (the two cities' countries' pump price at 8.5 litres a 100 km, and motorway tolls where they go by
  distance: about $0.10 a km in the US, $0.23 in Japan, $0.32 in France) and parking, shared by two
  people, the dollars counted as below. A car crosses water on a bridge or tunnel, or on a car ferry
  where OSM maps one (Helsinki to Tallinn): 90 minutes plus the crossing at 30 km/h, $80 plus 50
  cents a km a car, so a ferry is always slower and dearer than a bridge. Across the Channel it pays
  $150 a car and an hour for the shuttle or ferry; to Hokkaido $170 and 4 hours for the ferry (the
  Seikan Tunnel takes no cars). Belfast to Dublin is by road. No tolls at borders.
- Plane, between cities 250 km apart or more that both have airports: the drive to each airport,
  135 minutes at the airports (both ends together), 25 minutes until the flight (the train's planned wait below at a
  flight every 2 hours), the flight at 750 km/h plus 30 minutes (an hour more with no direct
  flight), and a fare of $100 plus 5 cents a km, so $125 at 500 km and $150 at 1,000 km, the
  same everywhere.
- Money on these trips: people are split into three equal groups by income, from each country's
  published income shares. Each group counts every dollar the same whichever way it goes, the
  car's fuel and ferries, the plane's fare and the train's fare: the starting country's minutes a
  dollar (the fare line above) times 0.4, since time on a long trip is worth more than on a local
  one; more for the poorest third, less for the richest (about 1.5 and 0.75 times in the US).
  Richer people take more long trips. Each group picks its own way, and the
  shares are added up. Which trains they take is worked out once for all three, at 0.4 (their
  average), so a high-speed line wins a long trip when it saves more minutes than its extra fare
  costs at that rate; commutes between cities share that choice of trains.
- Train: a trip between cities is planned, so its first train's wait is half the time between
  trains, but at most 5 minutes plus a sixth of it, and at most 30 minutes, and the "no train for
  hours" extra does not apply: 5 at a train every 10 minutes, 15 at an hourly train, 25 at one
  every 2 hours, 30 at anything sparser. A change of trains on the way waits as on any trip, and
  commutes and outings between cities wait as local trips do.

### Walking the whole way

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

The player's answer is a bus (since T-1464). Anyone within 300 m of a bus route walks to it (their
distance to the route plus 100 m), waits and rides to a station at 90% of what a car does on the road there, never under 30 km/h while moving (a bus lane), losing 28 seconds at each stop: about 19 km/h with stops every 400 m in a dense centre like Manhattan. The change to the
train is free: one fare for the whole trip on the same curve. Every time someone boards a bus,
including changing from a train to one, it costs them 6 perceived minutes on top of the wait. A home
1.5 km from a station walks 29 perceived minutes to it, or about 24 by bus. Trips by bus alone count
as rail. A bus is a third of a train car: riders mind its crowding from 45 aboard, it is at 100% with 53
and takes 80 at the most (a train car: 112, 160 and 240).

A ferry line (since T-2195) is a line like any other: people walk to its ports as to stations, wait
half its headway, ride at the boats' own speed, change to and from trains as anywhere, and pay the
same fare curve, its part above the base times the Ferry multiple (1.00 unless the player sets it).
There is no extra boarding cost as for a bus. A boat holds two cars' riders, 480 at the most. Where
the ferry is the quickest way across (Staten Island to Manhattan) it wins on these rules alone, and
its trips count as rail.

Ferry fans (since T-2196): a tenth of the people making a visit or going to an attraction find
ferries fun, so for them an hour aboard a ferry feels like 6 minutes. Everything else (the walk, the
wait, the fare, changes, crowding) counts as for anyone, and they still pick between the network and
the car as anyone does. Commutes, errands, school and the rest have no fans. Fans make no new trips:
they only move leisure trips that already go between the places a ferry links.

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
