# BRT

BRT gives a bus line its own lanes. It's an upgrade on the bus line's panel, built like anything else: click it and the lanes go in the blueprint, then construct them. On the map the stretches getting new lanes are outlined in blueprint blue, and stretches with lanes already there are dashed blue, since they cost nothing. Built lanes get a thin dashed white border.

**Faster in town.** A plain bus crawls in a dense centre, down to 25 km/h. A BRT bus gets 10 km/h more there, and never moves slower than that anywhere on its route.

**Paid only in town.** The lanes go where a plain bus would be slower than that. There they cost $5M a km of road and $21k a km a day, the same upkeep as street tram track. Out of town a BRT route costs what a plain bus route does. You pay when the lanes are constructed. Taking them away gives nothing back.

**Lines share lanes.** Two BRT lines on the same road use the same lanes, whichever way each one runs. The second line pays only for road the first one doesn't cover, and the upkeep of a shared stretch is split between them. A plain bus line on that road stays in traffic.

**A lane fills up.** Each lane takes 100 buses an hour each way. Past that, every line on it runs fewer buses, each cut by the same share, the way trains share full track (see [line capacity](line-capacity.md)). The panel's "Busiest lane" row shows how close the line's fullest stretch is.

**BRT stations.** Pick BRT station in the Station tool's bar and click within 50 m of a road to place one beside it. Near a bus line it goes beside the line's route instead. It costs $2M and $5k a day, like a tram stop. Only BRT lines stop there; a plain bus drives past. A stop there takes 18 seconds instead of 28, with no dwell, and the waiting room upgrade makes waiting there feel shorter, as at any station. People walk and drive to it from as far as they would to any station, where a plain bus stop only draws people living near the route. That is where most of BRT's riders come from: a line stopping at stations carries three or four times as many.

Exact rule: `docs/track-and-service.md` "6.7 Bus lines", its Money, Lane room and BRT stations items. Applies to commutes, everyday trips and trips between cities alike.
In the code: **main** `sim/src/track/bus/lanes.rs` `Lanes`, `Room`, also `sim/src/track/bus.rs` `reach`, `profile_cut`, `sim/src/track/net/store.rs` `bus_cost`, `brt_split`, `brt_room`, `sim/src/track/capacity.rs` `fill`, `sim/src/track/wasm/links.rs` `construct_ops`
