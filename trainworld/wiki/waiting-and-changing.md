# Waiting and changing

**Waiting** is half the gap between trains, in plain minutes. A train every ten minutes means five minutes on the platform. At a station with a waiting room, or inside a station building, each of those minutes counts {{sim/src/track/upgrades.rs:WAITING_ROOM_WAIT}} instead.

**No train for hours.** People differ in how long they'd wait. On a line with long gaps, or at the end of the day when the last train has gone, only some people have a train coming soon enough. Boarding there counts extra, the fewer the people with a train coming, the more it counts. If nobody has one, nobody boards.

**Changing trains** costs {{sim/src/demand/kernel.rs:Params.transfer_min|0}} minutes for the bother, plus the walk from one platform to the other, plus the wait for the next train. Inside a [[station-buildings|station building]] the bother is {{sim/src/track/upgrades.rs:BUILDING_CHANGE*100|0}}% of that. The walk counts extra, as in [[perceived-minutes]].

Stations within {{sim/src/demand/kernel.rs:Params.xfer_walk_m|0}} m of each other are close enough to change on foot. A passage you build works over a longer distance, along a straight line at {{sim/src/track/upgrades.rs:PASSAGE_MPS|0}} m/s.

People don't get off a line to catch the same line back the way they came. On a full line they wait at their own station.

If the train that arrives is full, the wait gets longer: see [[full-trains]].

Exact rule: `docs/demand-kernel.md` "Access and station-to-station times". A longer trip between cities is planned, so its first wait is shorter, at most {{sim/src/demand/kernel/search.rs:PLANNED_WAIT_CAP_MIN|0}} minutes, and it has no extra for long gaps. Commutes and outings between cities wait as local trips do.
In the code: `sim/src/demand/kernel/search.rs` `board_cost`, `planned_board_cost`, `change_f` `sim/src/demand/kernel/subzones.rs` `wait_factor` `sim/src/track/upgrades.rs` `WAITING_ROOM_WAIT`, `PASSAGE_MPS`
