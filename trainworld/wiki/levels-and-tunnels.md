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

Each stretch of an edge costs its length, times the base price a km, times its level's multiplier plus whatever the ground or water adds, times a factor for the kind of track. Long edges are cut into stretches wherever any of those change.

High-speed track is {{sim/src/track/params.rs:HSR_COST}} times that. Quad track is two lots of double, and regular, high-speed and tram track can all be quad. Single track costs {{sim/src/track/params.rs:SINGLE_TRACK_COST}} of double for every type.

**Ground isn't always cheap.** Ground-level track through a city has to knock things down, so it adds US${{sim/src/track/params.rs:ROAD_CROSSING_COST|0}}M for every road it crosses, more for a motorway. A viaduct or a tunnel goes past all of them, so past about {{sim/src/track/params.rs:BASE_COST_PER_KM*(LEVEL_MULT[4]-LEVEL_MULT[3])/ROAD_CROSSING_COST|1}} roads a km a viaduct is cheaper than the ground. Out in the country there's hardly a road to cross and the ground stays cheapest.

**Ramps.** Going up or down a level takes {{sim/src/track/params.rs:RAMP_PER_LEVEL|0}} m of track, and the ramp costs as much as the dearer of its two levels. Grades don't slow the trains down.

**Changing track you've built.** Select it, pick a new level or Regular, High speed or Tram, then press Rebuild. It is the same track and the same stations, so lines and station names carry on. You pay the full price of the new track and get nothing back for the old. Stations on the track go to the new level with it, even ones you didn't select. A station at the end of what you selected ramps on the track past it, just after the platform, and that bit is shown and paid for too. A piece with no room to ramp to the track beside it stays as it was, and shows in red before you confirm; the rest ramps to it. Rail rebuilt as tram turns its stations into tram stops, cutting long platforms down to a tram stop's longest, and its lines into tram lines. Tram and rail can't join, so all the track joined together has to change at once. Blueprint track switches between Regular, High speed and Tram the same way, at once and free.

**Single track to double, and back.** Select the track, pick the one-line or two-line icon (the same ones as when you lay track), then press Rebuild. The pieces and stations stay. Going double, the stations get platforms on both sides, and you pay the full price of double track and of the stations as double-track stations, nothing back for the single. Trains keep running and the stretch stops being shared between the two directions. Where the doubled stretch ends against single track, the two meet in a plain joint. Single track that runs one way stays single. Going single is free, and nothing is refunded either. The stations become single-track stations and the two directions share the track again. A station with passing tracks keeps its two tracks along the platform and a little past it, so it becomes the place where trains pass. Quad track is two lots of double, so each pair you pick goes single on its own. Track that needs both its tracks stays double, and the hover says why: a crossover, a flyover, a one-way track joined to one of its tracks. High-speed and tram track can go single too, and keep their type. Blueprint track changes either way at once, free, and can be undone.

What the ground is like adds more: see [[terrain]]. Water does too: see [[water]]. Then the track costs money every day: see [[upkeep]].

Exact rule: `docs/track-and-service.md` "Construction prices", and "Placing stations, curve speed and levels" for how levels and ramps fit, and "Rebuilding constructed track" for changing built track.
In the code: **main** `sim/src/track/cost.rs` `track_cost_parts_t`, also `track_factor` `sim/src/track/params.rs` `LEVEL_MULT`, `level_mult`, `ROAD_CROSSING_COST` `sim/src/track/roads.rs` `charge`, `sim/src/track/wasm/rebuild.rs` `rebuild_pieces`
