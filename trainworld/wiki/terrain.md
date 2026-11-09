# Terrain

Hills make track dearer. The map is cut into hexes and each one is flat, low hills, high hills or mountains.

Rough ground adds to the [[levels-and-tunnels|level's price multiplier]]:

| Ground | Added |
|---|---|
| Flat | nothing |
| Low hills | {{sim/src/track/params.rs:TERRAIN_SURCHARGE[1]}} |
| High hills | {{sim/src/track/params.rs:TERRAIN_SURCHARGE[2]}} |
| Mountains | {{sim/src/track/params.rs:TERRAIN_SURCHARGE[3]}} |

A bored tunnel doesn't care. It costs the same under flat fields and under mountains. So in the mountains a tunnel is the cheapest way through, then ground track, then a viaduct.

Stations, junctions and crossings pay nothing extra for the ground, and [[upkeep]] doesn't change. Steep ground never refuses track, and trains don't slow for it.

Over [[water]] the ground underneath doesn't count: only the water does.

A hex that looks like hills on the map is charged as hills.

In the code: `sim/src/track/cost.rs` `sim/src/track/params.rs`
