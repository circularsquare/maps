# What limits a line

A line can only run as many trains as its tightest spot allows. Every shared piece of the network is a spot that has to be free for each train for a little while. The more trains per hour, the fuller it is, and a spot that is nearly full starts to delay trains.

**Track.** Behind each train, the next one needs room to stop, and the faster the track, the more room. A train holds a stretch of double track for about {{sim/src/track/params.rs:SECTION_BASE_S|0}} seconds plus its top speed over {{sim/src/track/params.rs:SIGNAL_BRAKE}} m/s². At an ordinary train's top speed of {{sim/src/track/params.rs:V_TOP*3.6|0}} km/h that is {{sim/src/track/params.rs:3600/(SECTION_BASE_S+V_TOP/SIGNAL_BRAKE)|0}} trains an hour each way, and at a high-speed train's {{sim/src/track/params.rs:V_HSR*3.6|0}} km/h it is {{sim/src/track/params.rs:3600/(SECTION_BASE_S+V_HSR/SIGNAL_BRAKE)|0}}. Fast track carries fewer trains.

**Single track.** One track serves both ways. A train holds the whole stretch between passing points for its run time plus {{sim/src/track/params.rs:SINGLE_TRACK_MARGIN_S|0}} seconds, trains from both directions together.

**Platforms.** Each stopping train holds its platform for its dwell time plus {{sim/src/track/params.rs:PLATFORM_MARGIN_S|0}} seconds. With the default stop of {{sim/src/track/params.rs:DEFAULT_DWELL_S|0}} seconds that is {{sim/src/track/params.rs:3600/(DEFAULT_DWELL_S+PLATFORM_MARGIN_S)|0}} trains an hour on one platform track. A train that doesn't stop holds it for {{sim/src/track/params.rs:PASSING_S|0}} seconds.

**Where a line ends.** A train that turns round at a dead end holds the platform for its turnaround plus {{sim/src/track/params.rs:TERMINUS_MARGIN_S|0}} seconds, shared across the platform tracks that are built. With two tracks and the default turnaround of {{sim/src/track/params.rs:DEFAULT_TURNAROUND_S/60|0}} minutes that is {{sim/src/track/params.rs:3600/((DEFAULT_TURNAROUND_S+TERMINUS_MARGIN_S)/2)|0}} trains an hour. A short turnaround doesn't help past a point: a turn is never counted as quicker than {{sim/src/track/upgrades.rs:POOLED_TURN_MIN_S|0}} seconds. A terminal upgrade at a station adds sidings for turning.

**When it fills up.** Delay starts as a spot gets busy and grows fast near full. One line alone, with trains evenly spaced, has no delay until it is over what the spot can take. Several lines meeting at one spot delay each other, and more so the more evenly mixed they are. Past {{sim/src/track/params.rs:RHO_KNEE*100|0}}% full the delay keeps climbing, and over full a queue builds.

A line never runs more trains than its track can carry. The ones beyond that wait at the ends, and the panel warns. The fix is more track, a platform, a terminal, or fewer stops.

A slower train ahead on shared track holds a faster one back. A fast line sharing track with a slow one gains less than its top speed suggests. Passing tracks at a station let it by.

Exact rule: `docs/track-and-service.md` "Resources and utilisation" for each spot's rule, "Delay and slower trains" for the delay and for following slower trains. Junctions, merges and crossings have their own rules in the same sections, not covered here.
In the code: **main** `sim/src/track/capacity.rs` `build`, `kingman` `sim/src/track/params.rs` `SECTION_BASE_S`, `SIGNAL_BRAKE`, `PLATFORM_MARGIN_S`, `TERMINUS_MARGIN_S`, `KINGMAN_C0`, `RHO_KNEE` `sim/src/track/upgrades.rs` `pooled_turn_s`
