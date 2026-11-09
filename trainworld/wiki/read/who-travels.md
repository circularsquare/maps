# Who travels

People have places to be. Every trip in the game starts as a person who wants to get somewhere, train or no train. Your network changes how they get there: [how people choose](how-people-choose.md).

**Commutes.** Homes to jobs. Everyone with a job heads to one somewhere nearby, and nearer is likelier. A place full of jobs pulls harder, up to 20 times the usual pull for its size. Past that some jobs stay empty, so a desert town with too many jobs on paper doesn't drag workers in from far away.

**Errands and visits.** These are mostly short. The chance of going somewhere halves about every 0.7 km. The corner shop gets its street.

**Into town.** About 20% of errands and visits head for the dense middle of the city instead, where the department stores and theatres are. There the chance halves only every 4.2 km. Those are the long everyday trips a train can win.

**School, university, the doctor, the airport.** Each has its own reach. School is close, at 1.0 km to halve. University is farther at 2.4 km, and airports farther still at 14 km.

**Attractions.** Museums, zoos, theme parks, temples and castles draw visitors from across the city. The chance halves every 10 km. There aren't many of these trips, but they are long and all go to one spot, so a station at the door helps.

**Between cities.** Longer trips go by train, car or plane, and neighbouring cities swap some commuters too.

How often people go out comes from each country's travel survey. How far they go is one rule for the whole world. Some of these trips happen more at certain hours: see [rush hours](rush-hours.md).

Exact rule: `docs/demand-kernel.md` "Gravity between zones" for commutes, `docs/demand-plain.md` "Who travels, and where to" for everyday trips and attractions, `docs/regional.md` "The total travel market" for trips between cities. Each kind has its own rule.
In the code: `sim/src/demand/everyday.rs` `DECAY`, `RATE`, `CENTRAL_SHARE` `sim/src/demand/kernel/gravity.rs` `gravity`, `PULL_CAP`
