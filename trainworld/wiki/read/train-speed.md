# Train speed

A train goes as fast as the slowest thing in front of it. That is its own top speed, the cap of the track it is on, or the bend it is in.

**Bends.** A curve of radius R can be taken at a speed that grows with the square root of R: top speed in m/s is the root of 1.1 times the radius in metres. A tight 50 m bend, the tightest the game lets you build, tops out at 27 km/h. A bend a kilometre in radius allows 119 km/h. Reaching the speed of an ordinary train takes a radius of about 1,796 m, and a high-speed train about 7,183 m.

**Two kinds of train.** An ordinary train tops out at 160 km/h. A high-speed train reaches 320 km/h, and only on track built for it. On other track it goes as fast as an ordinary one. You pick the kind per line.

**Speeding up and stopping.** Trains speed up in three bands. They pull hardest at low speed and ease off as they go faster, like a real electric train. Only a high-speed train reaches the last band. Braking is gentler than pulling away: 0.9 m/s². A train waits at each stop for about 30 seconds.

**Speed limit.** You can also cap a line at a speed of your own, anywhere below its train's top speed. Its trains then go no faster than that anywhere on the route, and the round trip grows to match.

So stations close together mean slow lines, because the train spends its time speeding up and braking. A fast train gains most on long gaps between stops.

**Joins.** The whole train has to be under a limit before it may speed up again, so a train brakes before the joint where fast track meets slower track, just as it does before a bend.

**Hills don't count.** Ramps don't slow a train. Only the bend, the track and the train do.

The line panel's hover on a stop's time says what limited the speed on that stretch. How long a trip takes feeds into the cost of a ride in [perceived minutes](perceived-minutes.md).

Exact rule: `docs/track-and-service.md` "Placing stations, curve speed and levels" for bends and caps, "Trains and lines" for the two kinds and how they speed up. Stop times and turnaround at the ends of a line are in "Scheduling".
In the code: **main** `sim/src/track/params.rs` `A_LAT`, also `MIN_RADIUS`, `V_TOP`, `V_HSR`, `ACCEL_LOW`, `ACCEL_HIGH`, `ACCEL_TOP`, `BRAKE`, `DEFAULT_DWELL_S`, `design_radius` `sim/src/track/profile.rs` `BANDS`
