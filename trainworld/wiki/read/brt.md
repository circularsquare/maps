# BRT

BRT gives a bus line its own lanes. Tick the BRT box on a bus line's panel.

**Faster in town.** A plain bus crawls in a dense centre, down to 25 km/h. A BRT bus gets 10 km/h more there, and never moves slower than that anywhere on its route.

**Paid only in town.** The lanes go where a plain bus would be slower than that. There they cost $5M a km of road and $21k a km a day, the same upkeep as street tram track. Out of town a BRT route costs what a plain bus route does. A constructed line pays when you switch it on. Switching it off gives nothing back.

**Lines share lanes.** Two BRT lines on the same road use the same lanes, whichever way each one runs. The second line pays only for road the first one doesn't cover, and the upkeep of a shared stretch is split between them. A plain bus line on that road stays in traffic.

**A lane fills up.** Each lane takes 60 buses an hour each way. Past that, every line on it runs fewer buses, each cut by the same share, the way trains share full track (see [line capacity](line-capacity.md)). The panel's "Busiest lane" row shows how close the line's fullest stretch is.

Exact rule: `docs/track-and-service.md` "6.7 Bus lines", its Money and Lane room items. Applies to commutes, everyday trips and trips between cities alike.
In the code: **main** `sim/src/track/bus/lanes.rs` `Lanes`, `Room`, also `sim/src/track/net/store.rs` `bus_cost`, `brt_split`, `brt_room`, `sim/src/track/capacity.rs` `fill`
