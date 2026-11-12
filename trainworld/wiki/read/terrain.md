# Terrain

Hills make track dearer. The map is cut into small squares and each one is flat, low hills, high hills or mountains.

Rough ground adds to the [level's price multiplier](levels-and-tunnels.md):

| Ground | Added |
|---|---|
| Flat | nothing |
| Low hills | 0.5 |
| High hills | 1.2 |
| Mountains | 3 |

The stretch's price is its length, times the base price, times the level's multiplier plus that surcharge.

A bored tunnel doesn't care. It costs the same under flat fields and under mountains. So in the mountains a tunnel is the cheapest way through, then ground track, then a viaduct.

Stations, junctions and crossings pay nothing extra for the ground, and [upkeep](upkeep.md) doesn't change. Steep ground never refuses track, and trains don't slow for it.

Over [water](water.md) the ground underneath doesn't count: only the water does.

A square that looks like hills on the map is charged as hills.

Exact rule: `docs/track-and-service.md` "Construction prices", the Terrain item, which also has how the hills are measured.
In the code: **main** `sim/src/track/cost.rs` `terrain_surcharge`, also `terrain_spans` `sim/src/track/params.rs` `TERRAIN_SURCHARGE`, `TERRAIN_MIN_LEVEL`
