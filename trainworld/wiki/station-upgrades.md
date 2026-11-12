# Station upgrades

A station has four things you can add, from the row of icons in its panel. You buy them with a blueprint station (they are built with it) or pay at once on a built one. They have no [[upkeep]]. Taking one away refunds nothing.

**Terminal.** Trains turning here, at the end of a line or reversing partway, use both platforms. Without one they use a single platform, and the next train waits for the one there to leave. It costs {{sim/src/track/upgrades.rs:TERMINAL_KM}} km of double track at the station's level, which is US${{sim/src/track/upgrades.rs:TERMINAL_KM*BASE_COST_PER_KM*LEVEL_MULT[3]|1}}M on the ground ([[levels-and-tunnels]]). See [[line-capacity]] for what it does to trains an hour.

**Passing tracks.** A second platform track each way. Trains that stop use the outer one, and trains that don't use the inner one, or a second train can stop. A slower line that stops here waits for faster ones to pass, so an express can overtake a local ([[slower-trains]]). It costs the platform plus {{sim/src/track/upgrades.rs:PASSING_EXTRA_KM}} km of double track at the station's level, more on high-speed track and over water. It only works on double track, at a station with platforms on both tracks.

**Park and ride.** A drive to or from the station counts {{sim/src/track/upgrades.rs:PARK_RIDE_MIN|0}} minutes less ([[walking-to-stations]]). It costs US${{sim/src/track/upgrades.rs:PARK_RIDE_COST|0}}M. A P shows after the station's name.

**Waiting room.** Waiting for a train here counts {{sim/src/track/upgrades.rs:WAITING_ROOM_WAIT}} of its minutes ([[waiting-and-changing]]). It costs US${{sim/src/track/upgrades.rs:WAITING_ROOM_COST|0}}M.

**Complexes.** Linked stations share park and ride and the waiting room. Building one in the complex view builds it at every station of the complex, at one price each. A station that joins later has none until you build it there.

**Passage.** A walkway between two stations so riders can change between them on foot. It costs US${{sim/src/track/upgrades.rs:PASSAGE_BASE|0}}M plus US${{sim/src/track/upgrades.rs:PASSAGE_PER_KM|0}}M a km of the straight line between them, and can be at most {{sim/src/track/upgrades.rs:PASSAGE_MAX_M|0}} m long. Riders cross it at {{sim/src/track/upgrades.rs:PASSAGE_MPS|0}} m/s. A drawn [[station-buildings|station building]] does a similar job for a whole cluster.

Exact rule: `docs/track-and-service.md` "Station upgrades"; for what riders feel, `docs/demand-kernel.md` "Access and station-to-station times".
In the code: **main** `sim/src/track/upgrades.rs` `price`, also `passage_price`, `pooled_turn_s`, `PARK_RIDE_MIN`, `WAITING_ROOM_WAIT`
