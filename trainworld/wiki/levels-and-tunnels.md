# Levels and tunnels

Track can be built on several levels, from deep underground to a tall viaduct. Each level is {{sim/src/track/params.rs:LEVEL_H|0}} m above or below the last. Where you put it sets the price.

Double track costs this much a km, in US$ millions:

| Level | Price a km |
|---|---|
| Up {{sim/src/track/params.rs:MAX_LEVEL|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[6]|1}} |
| Up {{sim/src/track/params.rs:MAX_LEVEL-1|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[5]|1}} |
| Viaduct, up {{sim/src/track/params.rs:MAX_LEVEL-2|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[4]|1}} |
| Ground | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[3]|1}} |
| Cut and cover, down {{sim/src/track/params.rs:MAX_LEVEL-2|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[2]|1}} |
| Bored tunnel, down {{sim/src/track/params.rs:MAX_LEVEL-1|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[1]|1}} |
| Deep tunnel, down {{sim/src/track/params.rs:MAX_LEVEL|0}} | {{sim/src/track/params.rs:BASE_COST_PER_KM*LEVEL_MULT[0]|1}} |

High-speed track is {{sim/src/track/params.rs:HSR_COST}} times that. Quad track is two lots of double.

**Ground isn't always cheap.** Ground-level track through a city has to knock things down, so it adds US${{sim/src/track/params.rs:ROAD_CROSSING_COST|0}}M for every road it crosses, more for a motorway. A viaduct or a tunnel goes past all of them, so past about {{sim/src/track/params.rs:BASE_COST_PER_KM*(LEVEL_MULT[4]-LEVEL_MULT[3])/ROAD_CROSSING_COST|1}} roads a km a viaduct is cheaper than the ground. Out in the country there's hardly a road to cross and the ground stays cheapest.

**Ramps.** Going up or down a level takes {{sim/src/track/params.rs:RAMP_PER_LEVEL|0}} m of track, and the ramp costs as much as the dearer of its two levels. Grades don't slow the trains down.

What the ground is like adds more: see [[terrain]]. Water does too: see [[water]]. Then the track costs money every day: see [[upkeep]].

In the code: `sim/src/track/params.rs` `sim/src/track/cost.rs` `sim/src/track/roads.rs`
