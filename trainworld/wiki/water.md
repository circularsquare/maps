# Water

Track at ground level can't cross water. Draw across it at ground level and the game raises that stretch into a bridge for you, with ramps on land either side. If there's no room for the ramps the track is refused.

Over or under water, track adds {{sim/src/track/params.rs:WATER_SURCHARGE}} to its [[levels-and-tunnels|level's multiplier]], at any level. Stations, junctions and crossings over water cost {{sim/src/track/params.rs:WATER_COST|0}} times as much.

Water replaces [[terrain]]: what the ground would have been under it adds nothing.

Viaducts and cut and cover tunnels stop at {{sim/src/track/params.rs:WATER_SPAN_MAX/1000|0}} km of unbroken water. Bored tunnels have no limit, and a stretch of bored tunnel starts the count over.

Small creeks and ditches aren't on the water map, so track crosses them at ground level.

Exact rule: `docs/track-and-service.md` "Water, bridges and ramps" for how a bridge gets made, "Construction prices" for the price and the longest span.
In the code: `sim/src/track/bridge.rs` `bridge_long` `sim/src/track/cost.rs` `track_cost_parts_t` `sim/src/track/net.rs` `bridged_water_through` `sim/src/track/params.rs` `WATER_SURCHARGE`, `WATER_COST`, `WATER_SPAN_MAX`
