# Driving

Everyone in the game can drive. There's no count of households without a car. Dense places take the train more because [parking](parking.md) there is awful.

Roads don't run straight. The game uses real road routes where it has them, and about 1.3 times the straight line where it doesn't. Speed depends on where you are:

- in dense places, down towards 16.5 km/h
- on open country streets, about 44 km/h
- on main roads out of town, about 88 km/h
- on open highway in the middle of a long trip, 110 km/h

So a short trip in the city crawls and a long one is quick. Each minute behind the wheel counts 1.33 ([perceived minutes](perceived-minutes.md)).

**Owning the car** costs too: $2.1 a trip plus $0.14 a km, turned into minutes. Long trips between cities pay for fuel and tolls instead.

"The car" also stands for taxis and longer bus trips. "Walking" also stands for cycling and short bus trips. There's no separate count for those.

Exact rule: `docs/demand-mode.md` "Weights, ferry fans, driving and parking" "Routed walking and driving". This page is commutes and everyday trips. Between cities the drive runs over a world road map with fuel and tolls, and pays no ownership cost: `docs/regional.md` "Mode choice".
In the code: `sim/src/demand/kernel.rs` `place_min_per_km`, `drive_min_f`, `drive_factor`, `car_drive` `sim/src/demand/intercity/costs.rs` `routed_car`
