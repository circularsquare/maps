# Waiting and changing

**Waiting** is half the gap between trains, in plain minutes. A train every ten minutes means five minutes on the platform. At a station with a waiting room, or inside a station building, each of those minutes counts 0.8 instead.

**Several lines.** Where more than one line goes your way, people turn up at any time. Three in four take whichever gets them there first, and the rest take the first train that comes. A faster express that comes less often still gets its riders: the ones who turn up just before it, or who can wait a bit for it and still get there sooner. A slower local keeps the people it gets there first, and some who just take it because it came. A line so slow it gets there after the other line's worst case gets nobody. Two lines with the same gap going the same way, but with different stops or platforms, mean waiting a third of that gap, not a quarter, since their trains aren't spread out evenly.

**Lines on the same track and stops.** The timetable spaces lines that share track, so two lines that run on the same track and stop at the same stations count as one line for people disembarking at the same stop. Two lines every ten minutes mean waiting two and a half minutes, and people split between them by trains an hour. A line that skips a stop, or stops at a platform of its own, stays separate. Where the lines split into branches, people going past the split wait for their own branch at their own station. They don't ride the other branch's train to the split and change there, since their own train is right behind it.

**No train for hours.** People differ in how long they'd wait. On a line with long gaps, or at the end of the day when the last train has gone, only some people have a train coming soon enough. Boarding there counts extra, the fewer the people with a train coming, the more it counts. If nobody has one, nobody boards.

A boarding costs the wait for the first train worth taking, times the station's wait weight, plus the change cost when it is a change.

**Boarding.** Getting onto your first train costs 2 minutes and off your last one 2, trams included, at any level. A tram stop on the street costs nothing, and neither do buses and ferries. A change pays its own bother instead.

**Buses.** A minute on a bus counts as 1.1 minutes. A bus wait counts at least 2 minutes, however often the buses come. People walk up to 500 m to a bus route.

**Changing trains** costs 2 minutes for the bother, plus the walk from one platform to the other, plus the wait for the next train. Inside a [station building](station-buildings.md) the bother is 50% of that. The walk counts extra, as in [perceived minutes](perceived-minutes.md).

Stations within 300 m of each other are close enough to change on foot. A bus stop reaches only 150 m. A passage you build works over a longer distance, along a straight line at 2 m/s. Changing on foot at a bus stop or a BRT station takes 1 minute more than the walk, unless a passage joins the two. A trip can also start or end with that walk, with no bother and no transfer fare, so the way home is the way out reversed.

**Boarding only and disembarking only.** A stop of a line can be kept to one or the other. At a boarding-only stop people get on and never off, so nobody rides the line to it, in either direction, and nobody changes off the line there. At a disembarking-only stop it's the other way around. Trains still stop and wait the usual time. Other lines at the same station aren't affected.

People don't disembark from a line to catch any line back the way they came. On a full line they wait at their own station.

A line that bends back close to itself can be changed onto across the bend. People disembark, walk over to its other stop and ride on from there when that beats riding around. A passage between the two stops lets them do it from further apart. They never walk to another of its stops to ride back towards where they disembarked.

If the train that arrives is full, the wait gets longer: see [full trains](full-trains.md).

Exact rule: `docs/demand-kernel.md` "Access and station-to-station times". A longer trip between cities is planned, so its first wait is shorter, at most 30 minutes, and it has no extra for long gaps; with several lines it takes whichever gets there first in the same way. Commutes and outings between cities wait as local trips do.
In the code: **main** `sim/src/demand/kernel/search.rs` `board_cost`, `pool_cost`, `build_graph`, also `planned_board_cost`, `wait_spread`, `even_classes`, `merge_even`, `change_f`, `trip_ends`, `walk_ban` `sim/src/demand/kernel/subzones.rs` `wait_factor` `sim/src/track/upgrades.rs` `WAITING_ROOM_WAIT`, `PASSAGE_MPS` `sim/src/track/span.rs` `line_span`, `stop_span` `sim/src/demand/network.rs` `walk_reach_m`, `connection_min`, `BUS_WALK_M`, `CONNECTION_MIN`
