# Levels and tunnels

Track can be built on several levels, from deep underground to a tall viaduct. Each level is 8 m above or below the last. Where you put it sets the price.

Double track costs this much a km, in US$ millions:

| Level | Price a km |
|---|---|
| Up 3 | 150.0 |
| Up 2 | 112.5 |
| Viaduct, up 1 | 75.0 |
| Ground | 37.5 |
| Cut and cover, down 1 | 75.0 |
| Bored tunnel, down 2 | 120.0 |
| Deep tunnel, down 3 | 150.0 |

High-speed track is 1.5 times that. Quad track is two lots of double.

**Ground isn't always cheap.** Ground-level track through a city has to knock things down, so it adds US$10M for every road it crosses, more for a motorway. A viaduct or a tunnel goes past all of them, so past about 3.8 roads a km a viaduct is cheaper than the ground. Out in the country there's hardly a road to cross and the ground stays cheapest.

**Ramps.** Going up or down a level takes 200 m of track, and the ramp costs as much as the dearer of its two levels. Grades don't slow the trains down.

What the ground is like adds more: see [terrain](terrain.md). Water does too: see [water](water.md). Then the track costs money every day: see [upkeep](upkeep.md).

Exact rule: `docs/track-and-service.md` "Construction prices", and "Placing stations, curve speed and levels" for how levels and ramps fit.
In the code: `sim/src/track/params.rs` `LEVEL_MULT`, `level_mult`, `ROAD_CROSSING_COST` `sim/src/track/cost.rs` `track_cost_parts_t`, `track_factor` `sim/src/track/roads.rs` `charge`
