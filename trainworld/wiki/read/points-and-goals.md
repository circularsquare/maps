# Points and goals

Each region gets a score for how good its network is, and goals to reach.

**Points.** A region's points are what it earned over the past day of game time. They are never a running total, so a region whose network gets worse loses points. A rail trip earns 20 and each passenger-km earns 1. They unlock nothing yet.

A trip is a rider's whole rail journey, however many lines it uses, and counts in the region it starts in. Passenger-km count where riders get on. A region whose riders all board in another region's stations scores no trips.

**Goals.** A region's goal is the share of its trips that go by train. The first goal is 10%, then 20%, 30% and 40%. The share is measured on the average day, so switching the day you are viewing earns nothing. Each goal is kept once it is reached, even if the share falls back again. The first one is the star.

**Where the money goes.** The Money tab also keeps fares, running costs and rides for each region, and the regions add up to the whole network. Fares are split by where riders board and by the km they ride, each stretch between two stops going half to each stop's region. Running cost is shared by route km. Construction goes to the regions the new track or stations are in.

Money is settled as the game runs, not once an hour, so cash moves smoothly. A day runs midnight to midnight in the start zone's local time.

Exact rule: `docs/track-and-service.md` "Regions, points and stars" for points and stars and the regional split of money, "Fares and the ledgers" for how a day and an hour settle.
In the code: **main** `app/src/game/points.ts` `POINTS_PER_TRIP`, `GOAL_SHARES`, `railTripsByRegion`, `pointsPast24h` `app/src/game/regionMoney.ts` `lineShares`
