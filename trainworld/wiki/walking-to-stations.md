# Walking to stations

A train has to be reached first. For most people that's the hard part.

People walk at about {{sim/src/demand/kernel.rs:Params.walk_mps*3.6|0}} km/h, and streets don't run straight, so the path is about {{sim/src/demand/kernel.rs:Params.walk_detour}} times the straight line. A km on the map takes about {{sim/src/demand/kernel.rs:1000*Params.walk_detour/Params.walk_mps/60|0}} real minutes.

Each of those minutes counts {{sim/src/demand/kernel.rs:Params.w_walk}} ([[perceived-minutes]]). So a km from the station at both ends is {{sim/src/demand/kernel.rs:2*1000*Params.walk_detour/Params.walk_mps/60*Params.w_walk|0}} perceived minutes before the train has moved. More than most short drives cost. This is why rail is hard to win in the suburbs.

Stations more than about {{sim/src/demand/kernel.rs:Params.walk_cutoff_m/1000|0}} km away don't count as walkable. A train isn't offered at all if the two ends of the trip together cost more than {{sim/src/demand/kernel.rs:Params.pair_walk_max|0}} perceived minutes to reach.

**Driving to the station** is another way. It starts at {{sim/src/demand/kernel.rs:Params.p_ride|0}} minutes for the hassle, the drive counts extra, and it's for trips of {{sim/src/demand/kernel.rs:Params.ride_min_km|0}} km or more. A park and ride takes {{sim/src/track/upgrades.rs:PARK_RIDE_MIN|0}} minutes off that. Then there's [[parking]] at the station, or a drop-off counted as {{sim/src/demand/kernel.rs:Params.d_drop|0}} minutes if that's cheaper.

In the code: `sim/src/demand/kernel/subzones.rs` `sim/src/demand/kernel.rs`
