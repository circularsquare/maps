# Train speed

A train goes as fast as the slowest thing in front of it. That is its own top speed, the cap of the track it is on, the bend it is in, or a ramp.

**Bends.** A curve of radius R can be taken at a speed that grows with the square root of R: top speed in m/s is the root of {{sim/src/track/params.rs:A_LAT}} times the radius in metres. A tight {{sim/src/track/params.rs:MIN_RADIUS|0}} m bend, the tightest the game lets you build on rail, tops out at {{sim/src/track/params.rs:sqrt(A_LAT*MIN_RADIUS)*3.6|0}} km/h. High-speed track needs at least {{sim/src/track/params.rs:HSR_MIN_RADIUS|0}} m. A bend a kilometre in radius allows {{sim/src/track/params.rs:sqrt(A_LAT*1000)*3.6|0}} km/h. Reaching the speed of an ordinary train takes a radius of about {{sim/src/track/params.rs:R_CAP|0}} m, and a high-speed train about {{sim/src/track/params.rs:V_HSR*V_HSR/A_LAT|0}} m.

**Two kinds of train.** An ordinary train tops out at {{sim/src/track/params.rs:V_TOP*3.6|0}} km/h. A high-speed train reaches {{sim/src/track/params.rs:V_HSR*3.6|0}} km/h, and only on track built for it. On other track it goes as fast as an ordinary one. You pick the kind per line.

**Trams.** A line that stops at a tram stop runs trams. Trams run on tram track and on regular and high-speed track too, never faster than {{sim/src/track/params.rs:TRAM_V_CAP_KMH|0}} km/h. They brake at {{sim/src/track/params.rs:TRAM_BRAKE}} m/s² and wait at a stop as long as a train does, so a tram loses as much time at each stop as a train going as fast. A line with no tram stop runs trains, and trains can't run on tram track.

**Speeding up and stopping.** Trains speed up in three bands. They pull hardest at low speed and ease off as they go faster, like a real electric train. Only a high-speed train reaches the last band. Braking is gentler than pulling away: {{sim/src/track/params.rs:BRAKE}} m/s². A train waits at each stop for about {{sim/src/track/params.rs:DEFAULT_DWELL_S|0}} seconds.

**Speed limit.** You can also cap a line at a speed of your own, anywhere below its train's top speed. Its trains then go no faster than that anywhere on the route, and the round trip grows to match. A piece of track can have a limit of its own too, as low as {{sim/src/track/params.rs:EDGE_LIMIT_MIN}} km/h, set for free in the bar at the top of the map when you click it, or on a whole stretch at once. Every train slows down for it there. Where a line limit and a track limit meet, the lower one wins.

So stations close together mean slow lines, because the train spends its time speeding up and braking. A fast train gains most on long gaps between stops.

**Joins.** The whole train has to be under a limit before it may speed up again, so a train brakes before the joint where fast track meets slower track, just as it does before a bend.

**Ramps.** A ramp from one level to the next is good for {{sim/src/track/params.rs:RAMP_KMH_PER_M}} km/h for every metre of it. A {{sim/src/track/params.rs:RAMP_PER_LEVEL|0}} m ramp holds trains to {{sim/src/track/params.rs:RAMP_PER_LEVEL*RAMP_KMH_PER_M|0}} km/h, and one twice as long to twice that. The track tool makes ramps long enough for the target max speed where there's room, and shorter, slower ones where there isn't, down to {{sim/src/track/params.rs:RAMP_FLOOR|0}} m ({{sim/src/track/params.rs:TRAM_RAMP_FLOOR|0}} m on tram track). Ramps on track from older saves don't slow trains. The ground itself doesn't count, only the ramps you build.

The line panel's hover on a stop's time says what limited the speed on that stretch. How long a trip takes feeds into the cost of a ride in [[perceived-minutes]].

Exact rule: `docs/track-and-service.md` "Placing stations, curve speed and levels" for bends and caps, "Trains and lines" for the two kinds and how they speed up. Stop times and turnaround at the ends of a line are in "Scheduling".
In the code: **main** `sim/src/track/params.rs` `A_LAT`, also `MIN_RADIUS`, `V_TOP`, `V_HSR`, `ACCEL_LOW`, `ACCEL_HIGH`, `ACCEL_TOP`, `BRAKE`, `TRAM_BRAKE`, `DEFAULT_DWELL_S`, `design_radius`, `RAMP_KMH_PER_M`, `RAMP_FLOOR`, `ramp_for` `sim/src/track/geom.rs` `limits`, `ramp_speed` `sim/src/track/profile.rs` `BANDS`
