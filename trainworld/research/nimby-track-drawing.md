# How NIMBY Rails models and draws track and platforms (T-687)

Reference for T-658 (double track drawn as two tracks when zoomed far in) and T-657 (buildable ways
to turn trains faster at terminals). Public sources only: the NIMBY Rails wiki, the developer's
devblog, Steam forum threads and the store page. Researched 2026-10-19, about 25 searches. Where
something could not be found it says so; those gaps matter, because the zoom-level behaviour Anita
asked about is mostly undocumented in text.

## 1. How NIMBY models track

**Every track is a single track.** The developer wrote in May 2021 that "all the track in the game
is actually single track with a double track texture", and that the special two-train rules were
removed so the game is "truly single track"
([devblog 2021-04](https://carloscarrasco.com/page/52/)). Double track in NIMBY is two single
tracks side by side. The game's earlier "double track" was a drawing trick that was dropped.

**Track is control points, not a path.** The wiki describes a track object as one control point
linked to one or two neighbours (previous and next), which can carry signals and can be attached
to another track as a branch or a parallel track
([wiki: Track](https://wiki.nimbyrails.com/Track)). Curves were arcs joined to straights in the
early versions, then real circle arcs
([retrospective](https://carloscarrasco.com/nimby-rails-retrospective/)). Since 1.9 they are
Hobby splines with an editable tangent per node, which lets a dead-end track curve
([devblog 2023-06](https://carloscarrasco.com/page/26/)). Our model (straights and arcs between
PIs) is the same family as the pre-1.9 NIMBY one.

**Levels.** Ground, viaduct +1 to +3, tunnel -1 to -3, plus unlimited height and depth. Same-level
tracks intersect, different levels pass over or under
([wiki: How to build track](https://wiki.nimbyrails.com/How_to_build_track)). Close to our -3 to +3.

**Parallel track is a constrained child of another track.** From the 1.2 beta
([devblog 2021-05](https://carloscarrasco.com/page/51/)): parallel tracks "move and adapt
automatically to their parent track", placed "relative to their parent control point, given a
direction and offset". The wiki gives the offset as 0 to 1000 m from a control point, flippable to
the other side with F, or alternatively anchored to a point along the parent track with a sideways
displace of -1000 to +1000 m ([wiki: Track](https://wiki.nimbyrails.com/Track)). I could not
find the default offset in metres or the real track spacing; the track textures are drawn at
real scale and reproduce 1435 mm gauge ([devblog 2021-04](https://carloscarrasco.com/page/52/)).
The developer avoided snapping on purpose, because blueprint edits had to stay predictable.
The developer also called laying parallel track the main usability problem of going to true
single track ([devblog 2021-04](https://carloscarrasco.com/page/52/)).

**Double track mode.** The tutorial lists double track mode (key J) as the most common extra
setting in the track tools ([wiki: How to build track](https://wiki.nimbyrails.com/How_to_build_track));
the 2021 devblog calls it "2x" mode, on by default, in which continuations, branches and links try
to join existing parallel tracks, and Alt overrides the side chosen
([devblog 2021-05](https://carloscarrasco.com/page/51/)). Later patch notes say point mode was
disabled because "the parallel track system has some fundamental incompatibilities with point mode"
([1.9 patch notes](https://steamcommunity.com/app/1134710/discussions/0/3800524254163291827/)).
So the double-track convenience and the freehand tool did not fit together even for the developer.

**Switches and crossovers are built by hand from branches.** The Branch tool (B) makes
"junctions / switches". There is no crossover piece. A player asking how to build scissor crossovers
outside stations was told to branch two tracks out, join them (Ctrl swaps left and right), move
the segments into place and adjust the spacing; a video was linked
([Steam thread, Sep 2022](https://steamcommunity.com/app/1134710/discussions/0/3371531159453835625)).
A branch needs a parent track of some minimum length (a search summary of a Steam thread, which I
did not open, says the developer gave 30 m and a player measured 100 m).
Only in the station tool does the game build them for you: in 2x mode, platform creation "can
automatically create signals and scissors crossovers at the track ends"
([devblog 2021-05](https://carloscarrasco.com/page/51/)). The store page also says stations get
automatic double track and automatic entrance and exit signals, which can be turned off in the
track editor for manual control ([Steam store](https://store.steampowered.com/app/1134710/NIMBY_Rails/)).

**Signals affect drawing because the player places them.** Types now: one-way, platform stop hint,
block balise (a boundary that stops path reservation), path signal, no-way
([wiki: Signal](https://wiki.nimbyrails.com/Signal)). The old simple block signal was removed.
Players are told to put signals on every route into a junction or terminus, at passing loops and
alternative platforms, and on single-track entries, with balises on the way out
([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/4039228371164050216)). The
wiki's junction guide is a stub: path signals just ahead of the branch point (trains have an extra
30 m hitbox in front), balises one train length after; crossovers, terminals and diamond crossings
are empty headings ([wiki: How to signal a junction](https://wiki.nimbyrails.com/How_to_signal_a_junction)).
Signals are the main reason NIMBY is hard to build in, and we have none, so this is mostly a
warning, not a source of ideas.

## 2. How it draws track and platforms

**Zoom levels: little is written down.** What I found:

- "Track drawing was made to scale with real dimensions after certain level of zoom"; the track
  textures were redone "to replicate the look of a double track build in an abstract way"; plain
  textures with filtering tricks are used rather than polygons
  ([retrospective](https://carloscarrasco.com/nimby-rails-retrospective/)). So far out, track is a
  line of fixed screen width; past some zoom it becomes real width. That is the same trade as our
  minimum-width rule in T-658.
- The track texture has four layers (border, base, sleepers, rails) so branches and diamond
  crossings join cleanly ([devblog 2021-04](https://carloscarrasco.com/page/52/)). Tracks are drawn
  in four passes over four layers, with one stored geometry reused per pass
  ([devblog 2021-10](https://carloscarrasco.com/page/46/)).
- Lines on the map: a later visualisation stacks any number of lines side by side on a shared
  segment, "as real line maps do" ([retrospective](https://carloscarrasco.com/nimby-rails-retrospective/)).
- Terrain shading fades out as you zoom in ([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/3112518479597864319)).

I could not find a source for: the zoom at which the real width starts, whether platforms are
drawn at every zoom, whether the two tracks of a pair merge into one stripe when far out, or how
stations show when zoomed out. The wiki's Station and Line pages say nothing about drawing.
Screenshots and videos were not available to me, so none of this is described from the game's
actual screen.

**Platforms.** A platform is an ordinary pair of track segments flagged as a platform, "just your
standard 2 segment track piece, nothing more", with an invisible rectangular footprint; "touching
footprints between platforms groups them into a single station"
([devblog 2022-01](https://carloscarrasco.com/page/43/)). The retrospective says a platform models
"a real 2 track platform in a station". The footprint half width can be changed per platform since
1.13 ([devblog 2024-06](https://carloscarrasco.com/page/14/)). Platforms are "keep straight" by
default, so a station stretches when connected to a curve
([devblog 2022-01](https://carloscarrasco.com/page/43/);
[search summary of Steam threads](https://steamcommunity.com/app/1134710/discussions/0/3800524254163291827/)).
A station is "a set of one or more platforms whose footprints overlap", on any layer and in any
orientation ([wiki: Station](https://wiki.nimbyrails.com/Station)).

**Building a station.** Three routes exist. The station tool (T) creates stations or adds
platforms to existing ones; a first-release tool that placed a station on existing track with
automatic signals and scissors was dropped as too complex and replaced by "platform promotion",
which converts a valid pair of tracks under the cursor into a platform
([devblog 2022-12](https://carloscarrasco.com/page/32/)). Newer versions (1.15) removed the
invisible extension buildings so the player edits directly which tracks belong to a station
([SteamDB notes via search result; page itself returned 403](https://steamdb.info/patchnotes/16894454)).
Newcomers hit "track hits" and "can't find a path" because a station does not link itself to
track and needs at least two track sections between stations
([Steam thread, Apr 2021](https://steamcommunity.com/app/1134710/discussions/0/3128289322263949182)).
A reviewer's workflow advice is stations and platforms first, track second, lines third
([RailFans review, as quoted by a search result; page returned 403](https://www.railfans.ca/news/game-review-nimby-rails)).

## 3. Terminals and turnbacks

- Trains reverse only at a station or a dead end. The developer: "Trains do not teleport between
  platforms", so a terminus with a stop on each of two platforms makes the train reverse back out
  and in again; give a terminus one stop. A player's fix was "a small branch should be built for
  the train to make the maneuver" ([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/7221028806370441975)).
- Two platform tracks at a terminus must be set as alternative platforms or trains choose a long
  way round ([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/4039228371164050216)).
- A dead-end tail beyond two platforms that both reach it is used by players as a shared stop
  ([Steam thread, 2023](https://steamcommunity.com/app/1134710/discussions/2/6812310007736400955)).
- Two trains can share one platform if their stop points are at different ends and they fit; the
  game reserves track ahead of a stopped train, so same-end sharing fails unless a balise at the
  platform entrance overrides it, which puts the burden on the player's timetable
  ([wiki: How to have multiple trains share a platform](https://wiki.nimbyrails.com/How_to_have_multiple_trains_share_a_platform)).
- Turnaround time is the player's number, not a property of the layout. Players discuss 2
  minutes (a Japanese commuter service with a driver at each end) up to 5 to 10 minutes
  ([Steam thread](https://steamcommunity.com/app/1134710/discussions/0/604146154959017276)); one
  notes that two tracks at a terminus allow tight peak headways.
- Loops, crossover throats and tail tracks have no game support of their own; a player builds
  them from branches. I found no NIMBY source on loop terminals or on capacity per layout, and
  no wiki page on terminals. Real-world turnback capacity (geometry, signalling, rolling stock,
  dwell, crew changeover) is in
  [Gill's turn-back paper](https://witpress.com/elibrary/wit-transactions-on-the-built-environment/50/4189)
  (abstract only).

## 4. What players say about the building tools

Not much text exists; most of it is questions, not opinions.

- Praise: the store page sells automatic double track and automatic signalling for stations as the
  easy path, with the manual editor behind it ([store](https://store.steampowered.com/app/1134710/NIMBY_Rails/)).
  The game is rated about 88 percent positive; no review text was readable.
- Difficulty: a Steam reviewer called getting anything done frustrating and compared the tool to
  Illustrator's pen (quoted by a search summary; the review page itself was not opened). Players
  "have trouble making sense of which track node is which in complex junctions", which 1.10
  addressed by letting tracks be clicked anywhere along their length
  ([devblog 2023-07](https://carloscarrasco.com/page/25/)). Editing complex areas was slow
  because the conflict checker scaled quadratically with nearby track, fixed to linear in 1.9.
- Developer view: "the usability of laying parallel track" was the known cost of single-track
  everything ([devblog 2021-04](https://carloscarrasco.com/page/52/)), and he dropped a smart
  station tool partly because signals and automatic scissors made it too complex
  ([devblog 2022-12](https://carloscarrasco.com/page/32/)).
- Players call out the missing crossover piece indirectly: the scissors question above, and for
  another game ([Railway Empire thread](https://steamcommunity.com/app/503940/discussions/0/1700542332323158716))
  someone asking for X-intersections so double track need not be split and redoubled.

## 5. What we could borrow

Our default (one stroke draws a double track, junctions flat, crossovers as one-click X pieces,
no signals) is already easier than NIMBY's. The ideas below keep it that way; each lists what it
would cost the default case.

### For T-658, double track drawn as two tracks

1. **Width in world units past a zoom, a stripe before it.** NIMBY draws track at fixed screen
   width far out and real width after "certain level of zoom". We already have 16 m platforms drawn
   at true width from zoom 14. Do the same for track: one line (as now) at medium and far zoom,
   two thin tracks 4 m apart only from about zoom 16 or closer. Cost to default: none to build;
   a small draw cost, and a decision on the switch-over zoom so the line does not jump
   (NIMBY's switch point is not documented, so this is ours to choose).
2. **Keep double track as one piece with two lanes, not NIMBY's two parent and child tracks.**
   NIMBY's parent and child design is what made parallel building hard (its developer said so), and
   point mode could not coexist with it. Our one-piece double track only needs the drawing to
   offset each lane by 2 m from the centre line. Cost: none; it avoids copying the part of NIMBY
   players complain about.
3. **Draw junction and crossover geometry only when zoomed in.** NIMBY's four-layer track texture
   exists so branches and diamonds join cleanly. At the zoom where two tracks show, our crossover X
   and a junction's branch off one lane need to leave the right lane. Cost: drawing work only; no
   change to what the player clicks.
4. **A dead-end platform drawn with its tail visible.** NIMBY needs a dead-end extension to be
   built or hidden by the player. We could draw the over-run past the platform end on the
   existing track end. Cost: none.

### For T-657, faster turns at terminals

1. **Crossover tool as it is, plus a terminal throat preset.** NIMBY has no crossover piece and
   players build scissors by hand; our X piece already beats that. The terminal case is two
   platform tracks with a crossover beyond them (the Tube explainer calls this the commonest
   pattern: [how does the tube turnaround](https://tubenotifications.co.uk/insights/how-does-the-tube-turnaround)).
   Offer it as an optional button in the station inspector for a dead-end double-track station,
   not in the placement click. Cost to default: none if it stays out of placement; the click on
   a terminal still builds what it builds now.
2. **Tail track as a station setting: "tail track length" (0 default).** NIMBY players hide a
   dead-end extension past the platform so trains can clear the platform and reverse; the
   players' rule of thumb is that trains reverse only at a station or dead end (from a Steam thread
summary). A tail track fits
   our model as extra dead-end track beyond the platform that the turnaround rule treats as
   clear of the platform. Cost: nothing at 0; a stepper in the inspector, and a rule in
   `service.rs` for what a tail track does to turnaround time (our number, to be set: NIMBY
   does not give one, since it has no turnaround model, only player-set times).
3. **Loops are a separate, later item.** NIMBY has nothing built in and I found no source on
   how players use them. Our arcs allow up to 350 degrees, so a loop can already be drawn; the
   missing part is a rule that a train on a loop does not reverse. Cost: none to the default.
4. **Do not copy NIMBY's alternative platforms and stop-hint signals.** A terminus with one stop
   per platform makes trains reverse out and back in, which players find confusing. If
   turnback options arrive, keep one stop per line at a terminus so the question never comes up.
5. **Caution on scissors.** Real scissor crossovers on a terminus approach are space-saving but one
   fault cripples the service (a [forum comment](https://www.rmweb.co.uk/forums/topic/157152-first-layout-plan/page/2/),
   search summary only), and NIMBY needs signals to make them work. Without signals we should
   not offer a scissors piece unless the junction rules (6.2) handle the flat crossing between
   the diagonals, which our crossover already prices.

## 6. Not found

The zoom at which NIMBY draws real width; how platforms look at each zoom; the default parallel
offset in metres; any NIMBY text on loop terminals, tail tracks, or capacity per layout;
named player praise for the building tools (reviews were not readable, Reddit did not appear in
any result). SteamDB and RailFans returned 403 to fetches and are cited only through search
summaries. A watched video would settle the zoom questions.
