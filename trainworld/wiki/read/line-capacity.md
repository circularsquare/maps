# What limits a line

A line can only run as many trains as its tightest spot allows. Every shared piece of the network is a spot that has to be free for each train for a little while. The more trains per hour, the fuller it is, and a spot that is nearly full starts to delay trains.

**Track.** Behind each train, the next one needs room to stop, and the faster the track, the more room. A train holds a stretch of double track for about 70 seconds plus its top speed over 0.6 m/s². At an ordinary train's top speed of 160 km/h that is 25 trains an hour each way, and at a high-speed train's 320 km/h it is 17. Fast track carries fewer trains.

**Single track.** One track serves both ways. A train holds the whole stretch between passing points for its run time plus 240 seconds, trains from both directions together. A line with one train, on a stretch no other line uses, holds it for the run time alone. That is a train every ten minutes each way over one kilometre, every twelve over two, every fourteen over five and every twenty over ten. Single track that runs one way is counted like double track: trains only go one way over it, so it takes the track rule above. It also runs one way by itself where its ends join double track on tracks that send trains the same way, like the two legs of a flyover. A branch to a station runs both ways.

**Platforms.** Each stopping train holds its platform for its dwell time plus 60 seconds. With the default stop of 30 seconds that is 40 trains an hour on one platform track. A train that doesn't stop holds it for 40 seconds.

**Turning.** A train that turns round at a station, where a line ends or where it reverses partway, holds one platform for its turnaround plus 60 seconds. With the default turnaround of 3 minutes that is 15 trains an hour. A [terminal](station-upgrades.md) turns them on both platforms: 30 trains an hour. A short turnaround doesn't help past a point: a turn is never counted as quicker than 240 seconds. A quad station counts each pair of tracks on its own.

**When it fills up.** Delay starts as a spot gets busy and grows fast near full. One line alone, with trains evenly spaced, has no delay until it is over what the spot can take. Several lines meeting at one spot delay each other, and more so the more evenly mixed they are. Past 90% full the delay keeps climbing, and over full a queue builds.

A line never runs more trains than its track can carry. The ones beyond that wait at the ends, and the panel warns. The fix is more track, a platform, a terminal, or fewer stops.

**Quad track.** A line runs on one pair of tracks, the one its stations are on. Put a station on the other pair beside every stop, close enough to be part of the same station, and the line's panel offers to run its trains on both pairs. Then half the trains run on each pair, taking turns, so each pair carries half the line and it can run about twice as many trains. If a stop loses its station beside it, the line goes back to one pair until it has one again.

A slower train ahead on shared track holds a faster one back. A fast line sharing track with a slow one gains less than its top speed suggests. Passing tracks at a station let it by.

Exact rule: `docs/track-and-service.md` "Resources and utilisation" for each spot's rule, "Delay and slower trains" for the delay and for following slower trains, "Both pairs of quad track" for a line on two pairs. Junctions, merges and crossings have their own rules in the same sections, not covered here.
In the code: **main** `sim/src/track/capacity.rs` `build`, `kingman` `sim/src/track/twin.rs` `twin_plan` `sim/src/track/params.rs` `SECTION_BASE_S`, `SIGNAL_BRAKE`, `PLATFORM_MARGIN_S`, `TERMINUS_MARGIN_S`, `KINGMAN_C0`, `RHO_KNEE` `sim/src/track/upgrades.rs` `pooled_turn_s`
