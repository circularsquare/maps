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

Each stretch of an edge costs its length, times the base price a km, times its level's multiplier plus whatever the ground or water adds, times a factor for the kind of track. Long edges are cut into stretches wherever any of those change.

High-speed track is 1.5 times that. Quad track is two lots of double.

**Ground isn't always cheap.** Ground-level track through a city has to knock things down, so it adds US$10M for every road it crosses, more for a motorway. A viaduct or a tunnel goes past all of them, so past about 3.8 roads a km a viaduct is cheaper than the ground. Out in the country there's hardly a road to cross and the ground stays cheapest.

**Ramps.** Going up or down a level takes 200 m of track, and the ramp costs as much as the dearer of its two levels. Grades don't slow the trains down.

**Changing track you've built.** Select it, pick a new level or Regular, High speed or Tram, then press Rebuild. It is the same track and the same stations, so lines and station names carry on. You pay the full price of the new track and get nothing back for the old. A piece with no room to ramp to the track beside it stays as it was, and shows in red before you confirm; the rest ramps to it. Rail rebuilt as tram turns its stations into tram stops, cutting long platforms down to a tram stop's longest, and its lines into tram lines. Tram and rail can't join, so all the track joined together has to change at once. Blueprint track switches between Regular, High speed and Tram the same way, at once and free.

What the ground is like adds more: see [terrain](terrain.md). Water does too: see [water](water.md). Then the track costs money every day: see [upkeep](upkeep.md).

Exact rule: `docs/track-and-service.md` "Construction prices", and "Placing stations, curve speed and levels" for how levels and ramps fit, and "Rebuilding constructed track" for changing built track.
In the code: **main** `sim/src/track/cost.rs` `track_cost_parts_t`, also `track_factor` `sim/src/track/params.rs` `LEVEL_MULT`, `level_mult`, `ROAD_CROSSING_COST` `sim/src/track/roads.rs` `charge`, `sim/src/track/wasm/rebuild.rs` `rebuild_pieces`
