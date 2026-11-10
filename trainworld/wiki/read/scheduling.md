# Scheduling

You tell a line how many trains to run, and the game works out the timetable.

**Three levels.** A line has a train count for high, medium and low demand. High runs in the morning peak, midday and evening peak, medium in the evening and low at night ([rush hours](rush-hours.md)). The same on every kind of day, weekends and holidays too. A new line starts at none at every level, so you set each one. A line given trains at high alone stops running when high ends.

**Counts or headways.** A count is a whole number of trains on the whole line, both directions together. The game turns it into a gap between trains from the round trip: running time, dwell at the stops, the turnaround at the ends and any waiting. A line in headway mode keeps a gap instead, from one minute up to 4 hours, and the game works out how many trains that takes. Zero means no trains.

If the round trip leaves time over, the extra trains wait at the ends, half at each. If the line asks for more trains than its track can carry, the extras wait at the ends too and the panel warns ([line capacity](line-capacity.md)). Congestion lengthens the round trip, so the same count gives wider gaps.

**A loop** runs an even number of trains, half each way.

**Departures.** The first train leaves about half a run before the level starts and then one every gap, until half a run before it ends. A train belongs to the level in force when it reaches the middle of the line. So stops near the ends change service a little early or late. A line uses the local time of its first stop.

**Spacing between lines.** Lines that share track or a platform are set a little apart, so their trains don't arrive together. Two lines at one gap on one track leave half a gap apart.

**Changing the count.** Trains join and leave at the ends. A new train enters at a terminal, and a removed one finishes its trip first. Trains already running carry on where they were. The edit takes effect a moment after you click, so old and new service swap on one frame.

**Cars.** Each line also sets its train length. A new line starts at 5 cars, or at the length you set last. A platform shorter than the train caps it ([full trains](full-trains.md)).

Exact rule: `docs/track-and-service.md` "Scheduling", "Trains on a running line" and "Train length".
In the code: **main** `sim/src/track/service.rs` `build_line` `sim/src/track/params.rs` `PERIOD_LEVEL` `app/src/game/world.ts` `HEADWAY_MAX_S` `app/src/workers/saveFile.ts` `NEW_LINE_CARS`
