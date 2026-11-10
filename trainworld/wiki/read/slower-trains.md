# Slower trains ahead

A fast line that shares track with a slow one in the same direction catches it up and follows it. Its time saving melts away on the shared stretch, and it can end up no quicker than the local it trails.

**How much it loses.** Over the shared stretch the slow train takes some extra time D. The fast train closes the gap between trains, but has to keep a safe distance back, so it loses nothing until D passes the gap left after that distance. After that it loses nearly all of the rest. The gap is the slow line's headway, so more slow trains means a smaller gap, and a fast line behind a busy local line loses almost everything it would gain. A fast train behind several slow lines meets them at a random spot in their gaps. The worst of them counts.

**Timetabled together.** When one slower line shares the stretch, the timetable sends each fast train out as late as it safely can, just behind a slow train. A fast line gains something only where it gains more than that gap. Two lines whose headways differ by less than 30 seconds count as one pattern.

**Quad track helps.** Quad track is two separate double tracks, so an express on its own pair keeps its speed.

**Passing tracks.** A station with [passing tracks](station-upgrades.md) changes who waits. The slow train that stops there waits for the fast ones to pass, and the fast line keeps its speed. A slow train leaves 60 seconds after the last fast train has passed, unless the wait would hold up its own next train. Several passing stations in a row pass trains the same way, as often as each line's headways make them meet.

The loss counts like any other delay: it lengthens the round trip, so the line needs more trains for the same headway, and the average speed falls. It shows up in the trip times riders see ([perceived minutes](perceived-minutes.md)). It does not fill track, so it is not a place trains queue ([line capacity](line-capacity.md)).

Exact rule: `docs/track-and-service.md` "Delay and slower trains" has "Behind slower trains" and "Waiting for faster trains".
In the code: **main** `sim/src/track/capacity.rs` `follow_loss`, `follow_best`, `FOLLOW_TIMED_MIN_S` `sim/src/track/overtake.rs` `plan`, `PASS_CLEAR_S`
