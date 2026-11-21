# Track, platform and station rework: design (T-3444)

The design you agreed on 2026-11-21, revised to your answers (`answers-2026-11-21.md`). Nothing is built, and none of
the strings in it are final. Written by agent trackdesign with seven helpers: four research (other games, player
requests, today's code, real stations) and three drafting.

## The summary

**What goes wrong today.** The game knows two kinds of track piece, single and double. Quad is two doubles that the
game finds again by measuring how far apart they are, in at least six places, each with its own distance. Passing
loops, passing tracks and turnback crossovers are drawn but are not really there. A station cuts the track in two.
Since October about 80 track fixes trace back to these, and to stops being worked out by distance. 56 of them landed
in one week (11-10 to 11-18), right after single track, one-way track and the two-lane double came in, so that week
is a new feature's bug tail more than the usual rate.

**The new model.**
- Players think in three cases: single track (the base unit), double track (the easy one), and more tracks (3 to 16
  laid together). The track bar is 1, 2 and More with a count. A second click narrows more tracks to one double, and
  a double to one single.
- Underneath, every track is its own single track with its own direction. A double is two of them sharing one line
  on the map, saved together. Each track knows its partner, the track its trains come back on.
- Where tracks meet, the game stores which track carries on into which, when you draw it. Nothing is guessed later.
  A single drawn from a double's end carries on one of its tracks, the one under the pointer; merging two tracks is
  something you do yourself.
- Any track's direction can be set, and it spreads along that track to the nearest junctions. A double click traces
  one track a long way, so its direction or purpose can be set in one go. A purpose says what a track is for: any
  line, passing, or only some lines.
- A crossover is just a short single track drawn between two tracks; the tool lays the S curve. Tracks you add beside
  others stay unconnected until crossovers are built, and when a line needs a track it can't reach, the game offers
  "Build crossovers?" with the cost, in one click. Where a line turns back on double track, its crossover is laid with
  the stop.
- A platform is a strip beside a track, or between two tracks. It no longer cuts the track. Thin platforms that
  overlap stay, for ease. The station tool has three templates: side platforms, island, single platform, and its tag
  shows which tracks the station covers and which way trains stop on each. Islands later spread the tracks 6 m apart
  on an S laid for the track's top speed (about 70 m each end at 80 km/h, 140 m at 160 km/h). Platforms can taper
  where tracks split, and can be drawn.
- A station is the platforms placed together, plus any later platform placed overlapping one side by side (at any
  level, so 125th St Lexington is one station), plus any you join to it. Complexes stay as they are.
- Simple stops stay as now: one click on a station makes a two-way stop, and the game picks each direction's
  platform at that click and remembers it, so your split one-way legs work in one click. Exact stops are an option
  on each line: a numbered circle for every track a train could stop on, as in your notes.

**How it lands.** On its own branch, so the game keeps shipping and hotfixes go out while it is half done. It merges
in four releases (hours of real agent time; past track work built in about half its first guess, then drew a week of
fixes):
1. The foundation, 70-97 h: the three cases, one track pickable, long selections, the station tool's templates,
   stations on built track showing up in under half a second instead of about 2 s on the big save.
2. More tracks, 31-46 h: any count, added tracks and the crossover offer, AA/BB and ABBA, real passing loops and
   passing tracks, track purpose, crossovers as single track.
3. Stations, 27-37 h: islands that spread the tracks, the Spanish solution, the 1-minute island change, tapering and
   drawn platforms, exact stops, quick-placed and copied station layouts.
4. Real stations from OSM, 10-15 h, low priority.

About 140-200 h in all, plus 10-20 h of fixes after the foundation. Old saves are converted when they load, with the
old bytes kept; your Amsterdam "test" save is tried first, then the Vienna community save, then North America day 383.

## What you decided

All 27 picks, with your additions: the station templates and the placing preview (2), one-click two-way stops (3),
islands 6 m wide (7), direction per track with long selections (10), real crossovers you can delete (17), the
1-minute island change as a later stage (23), the three reference saves (25), real-network stations at low priority
(26). Plus your general notes (three cases, crossovers as single track, no automatic merge) and your follow-up on 8
(added tracks stay unconnected, with the crossover offer).

**Your question on 8, "what exactly does end mean?"** It meant the two places where the stretch you add a track to
stops: a junction, a station's throat, or where that track meets other track. The idea was to connect the new track
to its neighbour there, so trains could get on it. You have since said to leave added tracks unconnected and offer
crossovers when a line needs them, so that is what the design does now.

## Still open for you (my pick in brackets)

1. Track purpose: a passing track is where a stopping train waits for a faster train of another line, and trains
   that don't stop there keep off it; "only for these lines" just keeps other lines' routes off the track? (yes)
2. A line that turns back on double track gets its crossover laid with the stop edit, its cost in the edit, without
   the "Build crossovers?" question? (yes)
3. Where no turnback crossover fits, the line still turns back as it does now, rather than the stop being refused?
   (yes)

Decided by us, for you to know: crossovers are laid for the flyover target speed (80 km/h by default, about 80 m
long); old double-track crossovers load as double track; a stop whose two directions use different platforms shows
as one row in the stop list, with the track chip.

## The parts

- [01-model.md](01-model.md): the three cases, tracks, bundles, nodes, platforms, stations, smart allocation, and two
  test tables (every layout you named; every old rule and the stage that removes it)
- [02-station.md](02-station.md): what a station is, the templates and the preview
- [03-tools.md](03-tools.md): drawing and editing, long selections, purpose, the station tool, islands, crossovers,
  draft strings
- [04-stops.md](04-stops.md): simple and exact stops, the circles and how they move apart, draft strings
- [05-sim.md](05-sim.md): passing, turning back, the crossover placer, capacity, timetables, the checks against
  today's figures
- [06-demand.md](06-demand.md): what riders see; nothing moves on a save that uses nothing new
- [07-saves.md](07-saves.md): the save format and old saves
- [08-performance.md](08-performance.md): eleven speed requirements with numbers, and the measurements to take first
- [09-real-networks.md](09-real-networks.md): sandbox starts, OSM and GTFS
- [10-build-plan.md](10-build-plan.md): the branch, 38 stages of 2-6 h, the releases, where each paused task goes
- [tasks-to-file.md](tasks-to-file.md): one task per stage, ready to file
- research/: the four research reports; saves/: the reference saves
