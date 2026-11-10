# Waiting and changing

**Waiting** is half the gap between trains, in plain minutes. A train every ten minutes means five minutes on the platform. At a station with a waiting room, or inside a station building, each of those minutes counts 0.8 instead.

**No train for hours.** People differ in how long they'd wait. On a line with long gaps, or at the end of the day when the last train has gone, only some people have a train coming soon enough. Boarding there counts extra, the fewer the people with a train coming, the more it counts. If nobody has one, nobody boards.

A boarding costs half the gap between trains, times the station's wait weight, plus the change cost when it is a change.

**Changing trains** costs 2 minutes for the bother, plus the walk from one platform to the other, plus the wait for the next train. Inside a [station building](station-buildings.md) the bother is 50% of that. The walk counts extra, as in [perceived minutes](perceived-minutes.md).

Stations within 300 m of each other are close enough to change on foot. A passage you build works over a longer distance, along a straight line at 2 m/s.

People don't get off a line to catch the same line back the way they came. On a full line they wait at their own station.

If the train that arrives is full, the wait gets longer: see [full trains](full-trains.md).

Exact rule: `docs/demand-kernel.md` "Access and station-to-station times". A longer trip between cities is planned, so its first wait is shorter, at most 30 minutes, and it has no extra for long gaps. Commutes and outings between cities wait as local trips do.
In the code: **main** `sim/src/demand/kernel/search.rs` `board_cost`, also `planned_board_cost`, `change_f` `sim/src/demand/kernel/subzones.rs` `wait_factor` `sim/src/track/upgrades.rs` `WAITING_ROOM_WAIT`, `PASSAGE_MPS`
