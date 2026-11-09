# Walking to stations

A train has to be reached first. For most people that's the hard part.

People walk at about 6 km/h, and streets don't run straight, so the path is about 1.3 times the straight line. A km on the map takes about 13 real minutes.

The first 10 minutes of a walk count 1.5 each. Minutes past that have their own weight, 1.5 ([perceived minutes](perceived-minutes.md)). So a km from the station at both ends is 39 perceived minutes before the train has moved. More than most short drives cost. This is why rail is hard to win in the suburbs.

Stations more than about 3 km away don't count as walkable. A train isn't offered at all if the two ends of the trip together cost more than 80 perceived minutes to reach.

**Driving to the station** is another way. It starts at 15 minutes for the hassle, the drive counts extra, and it's for trips of 5 km or more. A park and ride takes 10 minutes off that. Then there's [parking](parking.md) at the station, or a drop-off counted as 10 minutes if that's cheaper.

Exact rule: `docs/demand-mode.md` "Route choice and getting to the station", `docs/demand-kernel.md` "Access and station-to-station times". Driving to the station happens at the home end. The far end of a trip is a drop-off only. Trips between cities have their own version: `docs/regional.md` "Park-and-ride".
In the code: `sim/src/demand/kernel.rs` `walk_min`, `access_cost_min` `sim/src/demand/kernel/access.rs` `build_access` `sim/src/demand/kernel/subzones.rs` `access_options`, `ride_fixed`
