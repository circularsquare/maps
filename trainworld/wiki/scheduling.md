# Scheduling

You tell a line how many trains to run, and the game works out the timetable.

**Three levels.** A line has a train count for high, medium and low demand. High runs in the morning peak, midday and evening peak, medium in the evening and low at night ([[rush-hours]]), every day, at the hours in the schedule window. A new line starts at none at every level, so you set each one. A line given trains at high alone stops running when high ends.

**Counts or headways.** A count is a whole number of trains on the whole line, both directions together. The game turns it into a gap between trains from the round trip: running time, dwell at the stops, the turnaround at the ends and any waiting. A line in headway mode keeps a gap instead, from one minute up to {{app/src/game/world.ts:HEADWAY_MAX_S/3600|0}} hours, and the game works out how many trains that takes. Zero means no trains.

If the round trip leaves time over, the extra trains wait at the ends, half at each. If the line asks for more trains than its track can carry, the extras wait at the ends too and the panel warns ([[line-capacity]]). Congestion lengthens the round trip, so the same count gives wider gaps.

**A loop** runs an even number of trains, half each way.

**Weekends.** Weekends and public holidays run their own trains: a percent of the weekday ones, {{sim/src/track/params.rs:WEEKEND_SHARE_MIN}}% to {{sim/src/track/params.rs:WEEKEND_SHARE_MAX}}%, or counts you set by hand in the line's schedule window. A new line runs the same trains every day. At a percent, counts round up and a headway rounds down to a whole minute, and changing the weekday trains moves the weekend ones with them. Counts set by hand stay put. One button sets every line to the same percent. Bus and ferry lines have weekends too, in buses and boats. A line owns enough trains for its busier kind of day. Long holiday travel days run the weekday trains. Weekend riders stay the same whatever runs, so fewer trains are more crowded.

**Departures.** The first train leaves about half a run before the level starts and then one every gap, until half a run before it ends. A train belongs to the level in force when it reaches the middle of the line. So stops near the ends change service a little early or late. A line uses the local time of its first stop.

**Spacing between lines.** Lines that share track or a platform are set a little apart, so their trains don't arrive together. Two lines at one gap on one track leave half a gap apart.

**Changing the count.** Trains join and leave at the ends. A new train enters at a terminal, and a removed one finishes its trip first. Trains already running carry on where they were. The edit takes effect a moment after you click, so old and new service swap on one frame.

**Cars.** Each line also sets its train length. A new line starts at {{app/src/workers/saveFile.ts:NEW_LINE_CARS|0}} cars, or at the length you set last. A platform shorter than the train caps it ([[full-trains]]).

Exact rule: `docs/track-and-service.md` "Scheduling", "Trains on a running line" and "Train length".
In the code: **main** `sim/src/track/service.rs` `build_line` `sim/src/track/params.rs` `PERIOD_ROW` `WEEKEND_SHARE_MIN` `WEEKEND_SHARE_MAX` `sim/src/track/world.rs` `weekend_counts` `settle_weekend` `app/src/game/world.ts` `HEADWAY_MAX_S` `app/src/workers/saveFile.ts` `NEW_LINE_CARS`
