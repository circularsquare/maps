# Slower trains ahead

A fast line that shares track with a slow one in the same direction catches it up and follows it. Its time saving melts away on the shared stretch, and it can end up no quicker than the local it trails.

**How much it loses.** Over the shared stretch the slow train takes some extra time D. The fast train closes the gap between trains, but has to keep a safe distance back, so it loses nothing until D passes the gap left after that distance. After that it loses nearly all of the rest. The gap is the slow line's headway, so more slow trains means a smaller gap, and a fast line behind a busy local line loses almost everything it would gain. A fast train behind several slow lines meets them at a random spot in their gaps. The worst of them counts.

**Timetabled together.** When one slower line shares the stretch, the timetable sends each fast train out as late as it safely can, just behind a slow train. A fast line gains something only where it gains more than that gap. Two lines whose headways differ by less than {{sim/src/track/capacity.rs:FOLLOW_TIMED_MIN_S|0}} seconds count as one pattern. Each fast train has its own times: one that would catch the slow train ahead slows down from the station before, just enough to stay a safe distance behind it, and one that never gets close keeps its speed.

**Quad track helps.** Quad track is two separate double tracks, so an express on its own pair keeps its speed.

**Passing tracks.** A station with [[station-upgrades|passing tracks]] changes who waits. The slow train that stops there waits for the fast ones to pass, and the fast line keeps its speed. A slow train leaves {{sim/src/track/overtake.rs:PASS_CLEAR_S|0}} seconds after the last fast train has passed. It waits no longer than the gap between two fast trains, since the next one would catch it just the same, and never so long that it holds up its own next train. Several passing stations in a row pass trains the same way, as often as each line's headways make them meet. Each slow train has its own times, so it waits only for the fast trains that would catch it, and one that no fast train catches does not wait at all. On weekends a line with its own weekend trains runs the average waits.

**Averages in the figures.** The round trip, the trains a line needs and the trip times riders see use the average wait and the average loss per train. The trains on the map and in the timetable chart run their own times. Where a slow line's own waits one way add up to more than its trains have to spare at the end, that direction runs the average waits. A fast train slows behind a slow one only as much as it has to spare at the end, so it still makes its next departure. Past that it catches the slow train and waits behind it on the map.

The loss counts like any other delay: it lengthens the round trip, so the line needs more trains for the same headway, and the average speed falls. It shows up in the trip times riders see ([[perceived-minutes]]). It does not fill track, so it is not a place trains queue ([[line-capacity]]).

Exact rule: `docs/track-and-service.md` "Delay and slower trains" has "Behind slower trains" and "Waiting for faster trains".
In the code: **main** `sim/src/track/capacity.rs` `follow_loss`, `follow_best`, `FOLLOW_TIMED_MIN_S` `sim/src/track/overtake.rs` `plan`, `PASS_CLEAR_S`, `train_waits` `sim/src/track/timing.rs` `retime`, `follow_one`, `floor`
