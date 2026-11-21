# Track, platform and station rework: design draft (T-3444)

A draft for us to talk through. Nothing is built, and none of the strings in it are final. Written 2026-11-21 by
agent trackdesign with seven helpers: four research (other games, player requests, today's code, real stations) and
three drafting.

## The summary

**What goes wrong today.** The game knows two kinds of track piece, single and double. Quad is two doubles that the
game finds again by measuring how far apart they are, in at least six places, each with its own distance. Passing
loops, passing tracks and turnback crossovers are drawn but are not really there. A station cuts the track in two.
Since October about 80 track fixes trace back to these, and to stops being worked out by distance. 56 of them landed in one week (11-10 to 11-18),
right after single track, one-way track and the two-lane double came in, so that week is a new feature's bug tail
more than the usual rate.

**The new model.**
- Every track is its own single track with its own direction. A double is two of them sharing one line on the map,
  saved together, as your notes put it. Quad is four. Any count up to 16.
- That group of tracks is what you draw, pick, move, price and delete. A second click picks one double or one track.
  A casual player never meets single tracks unless she wants to.
- Each track knows its partner: the track its trains come back on. AA/BB stores which tracks pair up.
- Where tracks meet, the game stores which track carries on into which, when you draw it. Nothing is guessed later.
- A platform is a strip beside a track, or between two tracks. It no longer cuts the track. Thin platforms that
  overlap stay, for ease. An island platform spreads the tracks apart on an S curve laid for the track's top speed:
  about 70 m each end at 80 km/h, 140 m at 160 km/h.
- A platform can sit between tracks that split apart, so it can taper. Drawn platforms come last.
- A station is a named group of platforms (below). Station complexes stay as they are, for changes and the circle.
- Simple stops stay exactly as now. Exact stops are an option on each line: a numbered circle for every track a
  train could stop on, pushed apart along its platform, tinted by level, as in your notes.

**What a station is.** Your four ideas, judged: a platform as a station makes every double-track station two. The
complex as the station loses the separate names players asked for. My pick is a mix of the other two: a station is
the platforms placed together, plus any later platform placed overlapping one of them side by side (at any level, so
125th St Lexington is one station), plus any you join to it by hand (NYC's offset platforms). The game decides
membership once, at placing, and never again. Real data works the same way: a station holds platforms, a complex
holds stations.

**In four releases** (hours of real agent time; past track work built in about half its first guess, then drew a
week of fixes):
1. The foundation, 60-84 h: quad as one thing, one track pickable, the station tool picking tracks, stations on
   built track showing up in under half a second instead of about 2 s on the big save, sandbox starts rebuilt.
   Nothing new to build yet.
2. More tracks, 23-34 h: any count, a fifth track, AA/BB and ABBA, real passing loops and passing tracks, the
   crossover tool.
3. Stations, 21-28 h: island platforms, the Spanish solution, tapering and drawn platforms, exact stops.
4. Real stations from OSM, 8-12 h: island or side platforms and track order where OSM has them.

About 110-160 h in all, plus 10-20 h of fixes after the foundation. Old saves are converted when they load, best
effort, with the old bytes kept.

**What players asked.** About 60 reports in 28 groups: AA/BB or direction-paired quad 5 times, three tracks or any
count, real passing stations, single and double joining cleanly. Other games: NIMBY Rails treats every track as its
own and a station as overlapping platforms; Subway Builder keeps it simple with fixed station types. Players there
complain both of too much hand assigning and of not knowing which platform trains use.

## Your decisions (my pick in brackets)

The model
1. Build the model above: every track its own track, grouped and saved together, stations that don't cut track? (yes)
2. A station is the platforms placed together, plus overlapping ones placed later, plus any you join; complexes stay
   above it? (yes)
3. Split one-way legs (your manual AA/BB from 11-13) stop working by distance: today a line's way back finds the
   station within 300 m by itself. After this, the two legs' stations must be one station, placed in one click on
   both legs or joined in the station panel. Old saves are joined for you. OK? (yes)
4. Foundation first, then more tracks, then stations? Exact stops can move up to right after the foundation. (as
   listed)

Drawing and stations
5. Price per track: a track with one beside it running the other way costs half a double, any other 0.6. That gives
   single 0.6, double 1, quad 2, AA/BB 2.2, three tracks 1.5? (yes)
6. The island's S is laid for the track's target max speed; flyover speed only where room is short? (yes)
7. Islands 6 m wide (reads as an island from about zoom 16), or 10 m (a zoom sooner, longer S)? (6 m)
8. An added track runs the same way as its neighbour and joins on at both ends? (yes)
9. Adding tracks to built track pays for the new tracks only? (yes)
10. Up to 16 tracks, odd counts with a two-way middle track? (yes)
11. A flying split with ramps switched off crosses flat? (yes)

Stops
12. Exact stops is a switch per line; going back to simple always works, with a notice of what changed? (yes)
13. Waypoints only on exact lines for now? (yes)
14. Zoomed out, a station's circles merge into one: a click adds by the simple rule, or zooms in when unclear? (yes)
15. A new simple stop goes on the tracks the line already uses, and a small track chip moves it? (yes)
16. On AA/BB in simple stops the chip picks a pair for both ways; a mix needs exact stops? (yes)

Trains
17. Turning back at a station stays as now: no crossing counted, the crossover only drawn? (yes)
18. The terminal stays an upgrade at 30 trains an hour, no stub track; real stub terminals can be built manually?
    (yes)
19. Old passing tracks become real track, each line stopping on one platform track, so busy ones wait a little more?
    (yes)
20. Old single-track stations become real passing loops: a little less waiting, and one with a terminal turns 30 an
    hour instead of 15? (yes)
21. A line over several tracks runs an equal share of its trains on each, as two pairs does now? (yes)
22. Getting-on and getting-off sides (the Spanish solution) change no numbers for now? (yes)

Riders
23. A cross-platform change stays an ordinary change, no new factor? (yes; if you want the flavour, 1 minute
    instead of 2 between the two sides of one island, about 2-3 h)
24. Stations you join that are over 100 m apart stay two places for riders, walk included? (yes)

Saves and real networks
25. Saves that must convert cleanly: the 2,647-station save, North America day 383 and the real NYC save; the rest
    best effort? (yes)
26. In the sandbox starts, a stop where local and express tracks run side by side becomes one station with four
    platforms, instead of today's two stations named alike. Separate lines under one name (Times Sq's IRT and BMT)
    stay separate stations in one complex. (yes)

Paused tasks
27. Free the street-gaps overlay (T-3036) now, since it doesn't touch track? (yes)

## The parts

- [01-model.md](01-model.md): tracks, bundles, nodes, platforms, stations, and two test tables (every layout you
  named; every old rule and the stage that removes it)
- [02-station.md](02-station.md): what a station is, four options and the pick
- [03-tools.md](03-tools.md): drawing and editing, the station tool, islands, crossovers, draft strings
- [04-stops.md](04-stops.md): simple and exact stops, the circles and how they move apart, draft strings
- [05-sim.md](05-sim.md): passing, turning back, crossovers, capacity, timetables, the checks against today's figures
- [06-demand.md](06-demand.md): what riders see; nothing moves on a save that uses nothing new
- [07-saves.md](07-saves.md): the save format and old saves
- [08-performance.md](08-performance.md): ten speed requirements with numbers, and the measurements to take first
- [09-real-networks.md](09-real-networks.md): sandbox starts, OSM and GTFS
- [10-build-plan.md](10-build-plan.md): 30 stages of 2-6 h, the releases, where each paused task goes
- research/: the four research reports (games.md, players.md, code.md, realworld.md)
