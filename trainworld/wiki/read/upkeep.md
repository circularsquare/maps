# Upkeep

Built track and stations cost money every game day, whether trains run or not. A blueprint costs nothing until you build it.

A km of double track costs this much a game day:

| Track | Upkeep a day |
|---|---|
| Ground | $35k |
| Viaduct | $50k |
| Cut and cover | $100k |
| Bored tunnel | $130k |

A piece's upkeep is its length, times the rate of the dearest structure on it, times the track kind's factor, scaled to game money.

A piece of track pays the dearest thing it touches, so a ramp down into a tunnel makes the whole piece a tunnel. High-speed track is 1.5 times that. Quad track pays twice.

A station costs $50k a day at ground level or above, and $200k underground. Junctions and flyovers cost nothing on top.

A tunnel costs more to keep up as well as more to [build](levels-and-tunnels.md). Upkeep is the same in every country. The cost of running trains changes with the country.

Your lines share the bill by how many trains they run on each stretch, so a line's profit already has its share taken out. [Terrain](terrain.md) and [water](water.md) don't change upkeep.

Exact rule: `docs/track-and-service.md` "Blueprint, trains and running costs", the Upkeep item, which also has how it splits by line and region.
In the code: **main** `sim/src/track/cost.rs` `track_upkeep_km`, also `station_upkeep` `sim/src/track/wasm.rs` `edge_upkeep_km` `sim/src/track/wasm/money.rs` `upkeep`, `line_upkeep` `sim/src/track/params.rs` `UPKEEP_GROUND_KM`, `ECONOMY`
