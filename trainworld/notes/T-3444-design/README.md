# Track, platform and station rework: design (T-3444)

The design you agreed on 2026-11-21, revised to your answers (`answers-2026-11-21.md`), and on 11-28 to Gemini's two
reviews and your 11-27 notes. Written by agent trackdesign with seven helpers; revised 11-28 by agent trackplan.

## Where it stands (11-28)

1. About half of release 1 is built, on its own branch (tw/track), 11-21 to 11-23: every track is its own single
   track, 1 to 16 side by side; one track can be picked, flipped and deleted; Add a track; long selections; stations
   and platforms as records; one-click two-way stops; the stop list's track chip. Stations that sit on the track
   without cutting it work in the simulation, but the station tool still cuts the track.
2. Paused 11-23 to 11-27. Master moved 575 commits meanwhile, and they are being merged into the branch now. From
   here the branch takes master after every batch that touches track, stations, lines or saves.
3. Left for release 1: stations placed without cutting the track (T-4037), old saves' stations converted (T-3961),
   station and platform ids of their own (T-4043, T-4517), the station tool's three templates (T-3963), Join to
   station (T-3964), a way to connect an added track (T-4067), lines drawn per track (T-3966, T-3967), and fixes from
   Gemini's read of the branch (T-4520).
4. Then, as you said on 11-27, one crash round near the release instead of crash hunting while building: the edit
   fuzzer on your saves and a Gemini adversarial read of the branch, your T-4116 panic first (T-4519). Plus an edit
   speed run on your Asia save against master; anything more than 10% slower comes to you (T-4521).
5. About 40-70 h of agent time for release 1 from here, plus 10-20 h of fixes after. Releases 2 to 4 are unchanged.
6. Gemini's reviews: of the plan's 15 points, 7 stand, 7 partly, 1 wrong; of the branch's 12 bugs, 6 are confirmed in
   the code, 2 partly, 4 likely. Each is in the plan or a task now ("Review 11-28" in 10-build-plan.md).

## Your call

- Once a station no longer cuts the track, a click picks the whole stretch from junction to junction, so "make just
  the bit between these two stations single" would need a joint first. Should a click still pick the stretch between
  the stations either side, with the game cutting the track there when you change it (T-4518)? I'd say yes: it is
  what players do today.

Settled already: the route question for lines that turn back at a middle stop was yours on 11-23 ((b): routes are
written on the track as stations cut it), and its first two steps are built on it. The three small calls this page
used to list were answered on 11-21 (no passing track purpose; turnback crossovers come with the stop edit; where none
fits, the line turns back as today).

Decided by us, for you to know: dragging a station slides it along its track (bending is done with the track's own
points); deleting track keeps the stations on it (your 11-23 call); on 3 or more tracks a station's platforms sit on
the outer two until the templates come.

## The design in short (11-21)

**What goes wrong today.** The game knows two kinds of track piece, single and double. Quad is two doubles that the
game finds again by measuring how far apart they are, in at least six places, each with its own distance. Passing
loops, passing tracks and turnback crossovers are drawn but are not really there. A station cuts the track in two.
Since October about 80 track fixes trace back to these, and to stops being worked out by distance. 56 of them landed
in one week (11-10 to 11-18), right after single track, one-way track and the two-lane double came in, so that week
is a new feature's bug tail more than the usual rate.

**The new model.**
- Players think in three cases: single track (the base unit), double track (the easy one), and more tracks (3 to 16
  single tracks side by side). The track bar is 1, 2 and More with a count. A second click narrows any pick to one
  track. The Select bar's **Add a track** puts one more track beside a run (it replaces Double up).
- Underneath, every track is its own single track with its own direction. A double is two of them sharing one line
  on the map, saved together. More tracks are singles too (your 11-21 call): two doubles side by side is one way to
  set four up (old quad loads as that), ABBA or AABB others, made by setting each track's direction. The game pairs
  tracks out of sight so a line comes back on the track beside it by default; a line's stops decide its tracks.
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

**How it lands.** On its own branch, merged into master in four releases: the foundation (single, double and
more tracks, stations off the track, the station tool's templates), more tracks (any count in one stroke, crossovers,
passing tracks, purpose), stations (islands, the Spanish solution, exact stops), and real stations from OSM. Old saves
are converted when they load, with the old bytes kept. Hours and order: 10-build-plan.md.

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
- [10-build-plan.md](10-build-plan.md): where it stands, the branch rules and the sync checklist, release 1's stages
  left, the gates before its merge, what master's changes mean for each stage, the 11-21 stages and releases, and
  "Review 11-28" (both Gemini reviews, finding by finding)
- [tasks-to-file.md](tasks-to-file.md): the 11-21 stage tasks as filed (TASKS.md has them as they are now)
- research/: the four research reports; saves/: the reference saves
