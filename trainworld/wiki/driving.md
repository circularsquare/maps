# Driving

Everyone in the game can drive. There's no count of households without a car. Dense places take the train more because [[parking]] there is awful.

Roads don't run straight: the road is about {{sim/src/demand/kernel.rs:Params.road_detour}} times the straight line. Speed depends on where you are:

- in dense places, down towards {{sim/src/demand/kernel.rs:Params.road_city_kmh}} km/h
- on open country streets, about {{sim/src/demand/kernel.rs:Params.road_town_kmh|0}} km/h
- on main roads out of town, about {{sim/src/demand/kernel.rs:Params.road_far_kmh|0}} km/h
- on open highway in the middle of a long trip, {{sim/src/demand/kernel.rs:Params.road_open_kmh|0}} km/h

So a short trip in the city crawls and a long one is quick. Each minute behind the wheel counts {{sim/src/demand/kernel.rs:Params.w_drive}} ([[perceived-minutes]]).

**Owning the car** costs too: ${{sim/src/demand/kernel.rs:Params.car_own_usd_trip}} a trip plus ${{sim/src/demand/kernel.rs:Params.car_own_usd_km}} a km, turned into minutes. Long trips between cities pay for fuel and tolls instead.

"The car" also stands for taxis and longer bus trips. "Walking" also stands for cycling and short bus trips. There's no separate count for those.

In the code: `sim/src/demand/kernel.rs`
