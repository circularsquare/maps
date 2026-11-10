# Station upgrades

A station has four things you can add, from the row of icons in its panel. You buy them with a blueprint station (they are built with it) or pay at once on a built one. They have no [upkeep](upkeep.md). Taking one away refunds nothing.

**Terminal.** Turnback sidings past the platform. A line that turns round here holds its platform only for the dwell, and turns in two sidings. It costs 1 km of double track at the station's level, which is US$37.5M on the ground ([levels and tunnels](levels-and-tunnels.md)). See [line capacity](line-capacity.md) for what it does to trains an hour.

**Passing tracks.** A second platform track each way. Trains that stop use the outer one, and trains that don't use the inner one, or a second train can stop. A slower line that stops here waits for faster ones to pass, so an express can overtake a local ([slower trains](slower-trains.md)). It costs the platform plus 0.4 km of double track at the station's level, more on high-speed track and over water. It only works on double track.

**Park and ride.** A drive to or from the station counts 10 minutes less ([walking to stations](walking-to-stations.md)). It costs US$20M. A P shows after the station's name.

**Waiting room.** Waiting for a train here counts 0.8 of its minutes ([waiting and changing](waiting-and-changing.md)). It costs US$5M.

**Complexes.** Linked stations share park and ride and the waiting room. Building one in the complex view builds it at every station of the complex, at one price each. A station that joins later has none until you build it there.

**Passage.** A walkway between two stations so riders can change between them on foot. It costs US$10M plus US$90M a km of the straight line between them, and can be at most 1,000 m long. Riders cross it at 2 m/s. A drawn [station building](station-buildings.md) does a similar job for a whole cluster.

Exact rule: `docs/track-and-service.md` "Station upgrades"; for what riders feel, `docs/demand-kernel.md` "Access and station-to-station times".
In the code: **main** `sim/src/track/upgrades.rs` `price`, also `passage_price`, `pooled_turn_s`, `PARK_RIDE_MIN`, `WAITING_ROOM_WAIT`
