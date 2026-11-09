# Full trains

A car holds 160 people at its rated load and 240 when it's as packed as it gets.

A ride counts one minute a minute until the train is 70% of the way to its rated load. After that each minute feels longer. At the rated load a minute counts 3, and at crush it counts far more than that. So a crowded line pushes people to other lines, or back to the car.

**Waiting for room.** If the train arriving has no space, the wait for one with room counts too, up to 60 minutes.

**No more riders than the trains can take.** People come to the platform and stay only if the trains due in the next 30 minutes have room for them. Everyone else goes by car or on foot and pays no fare. A line short of trains has full platforms and full trains all through the peak.

People already on board keep their places, so the squeeze falls on the stops down the line.

How busy it gets depends on the hour: see [rush hours](rush-hours.md). This all feeds into [perceived minutes](perceived-minutes.md).

Exact rule: `docs/demand-kernel.md` "Loading and crowding". These are train cars. A bus or a boat holds its own number.
In the code: `sim/src/demand/kernel/assign.rs` `crowd_factors` `sim/src/demand/kernel/stalls.rs` `left_behind`, `capacity_fit` `sim/src/demand/api.rs` `RATED_PER_CAR`, `CRUSH_PER_CAR`
