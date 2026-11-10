# Driving

Everyone in the game can drive. There's no count of households without a car. Dense places take the train more because [[parking]] there is awful.

Roads don't run straight. The game uses real road routes where it has them, and about {{sim/src/demand/kernel.rs:Params.road_detour}} times the straight line where it doesn't. Speed depends on where you are:

- in dense places, down towards {{sim/src/demand/kernel.rs:Params.road_city_kmh}} km/h
- on open country streets, about {{sim/src/demand/kernel.rs:Params.road_town_kmh|0}} km/h
- on main roads out of town, about {{sim/src/demand/kernel.rs:Params.road_far_kmh|0}} km/h
- on open highway in the middle of a long trip, {{sim/src/demand/kernel.rs:Params.road_open_kmh|0}} km/h

A drive is built in layers: the first and last stretch at each end's street speed, then main roads near each end, and whatever is left in the middle at highway speed. Dense ends slow the streets and the main roads, and the highway stays the same.

So a short trip in the city crawls and a long one is quick. Each minute behind the wheel counts {{sim/src/demand/kernel.rs:Params.w_drive}} ([[perceived-minutes]]).

**Owning the car** costs too: ${{sim/src/demand/kernel.rs:Params.car_own_usd_trip}} a trip plus ${{sim/src/demand/kernel.rs:Params.car_own_usd_km}} a km, turned into minutes. Long trips between cities pay for fuel and tolls instead.

"The car" also stands for taxis and longer bus trips. "Walking" also stands for cycling and short bus trips. There's no separate count for those.

Exact rule: `docs/demand-mode.md` "Weights, ferry fans, driving and parking" "Routed walking and driving". This page is commutes and everyday trips. Between cities the drive runs over a world road map with fuel and tolls, and pays no ownership cost: `docs/regional.md` "Mode choice".
In the code: **main** `sim/src/demand/kernel.rs` `drive_min_f`, also `place_min_per_km`, `drive_factor`, `car_drive` `sim/src/demand/intercity/costs.rs` `routed_car`
