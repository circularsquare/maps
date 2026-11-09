# Upkeep

Built track and stations cost money every game day, whether trains run or not. A blueprint costs nothing until you build it.

A km of double track costs this much a game day:

| Track | Upkeep a day |
|---|---|
| Ground | ${{sim/src/track/params.rs:UPKEEP_GROUND_KM*ECONOMY/1000|0}}k |
| Viaduct | ${{sim/src/track/params.rs:UPKEEP_VIADUCT_KM*ECONOMY/1000|0}}k |
| Cut and cover | ${{sim/src/track/params.rs:UPKEEP_CUT_KM*ECONOMY/1000|0}}k |
| Bored tunnel | ${{sim/src/track/params.rs:UPKEEP_BORED_KM*ECONOMY/1000|0}}k |

A piece of track pays the dearest thing it touches, so a ramp down into a tunnel makes the whole piece a tunnel. High-speed track is {{sim/src/track/params.rs:HSR_COST}} times that. Quad track pays twice.

A station costs ${{sim/src/track/params.rs:UPKEEP_STATION*ECONOMY/1000|0}}k a day at ground level or above, and ${{sim/src/track/params.rs:UPKEEP_STATION_UNDER*ECONOMY/1000|0}}k underground. Junctions and flyovers cost nothing on top.

A tunnel costs more to keep up as well as more to [[levels-and-tunnels|build]]. Upkeep is the same in every country. The cost of running trains changes with the country.

Your lines share the bill by how many trains they run on each stretch, so a line's profit already has its share taken out. [[terrain|Terrain]] and [[water]] don't change upkeep.

In the code: `sim/src/track/cost.rs` `sim/src/track/params.rs` `sim/src/track/wasm.rs`
