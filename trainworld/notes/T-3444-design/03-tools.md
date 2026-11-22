# T-3444 design, part 3: drawing and editing tools

Agent tdtools, 2026-11-21. Design for T-3444, revised the same day after Anita's answers
(`answers-2026-11-21.md`), and for her 11-21 call on single tracks and Add a track (agent quaddesign, T-4023); stages
2a-3a2 are built. Words as in `01-model.md` (track, bundle, slot, offset, node,
connection, platform, stop point, station, preset). Stage numbers are `10-build-plan.md`'s. Today's tools:
`docs/ui-building.md` "The track and station bars", "Drawing track", "Placing stations", "The Select tool".

## Three cases

Her words: "they should basically be able to think of 3 cases: single, double, and more. like a single is the base
unit, a double is the easy, supported, base unit if you dont wanna think about complex track, and more/bundle is for
grouping together ltos of tracks."

| Case | What it is | Panel title |
|---|---|---|
| Single track | one track, the base unit | "Single track" |
| Double track | a track and its partner running the other way | "Double track" |
| More | 3 to 16 single tracks side by side, each with its own way | "{n} tracks" ("4 tracks") |

Every tool and string uses these three. "Bundle", "slot", "pair", "partner" and "offset" are ours and never appear in
a string. Inside "more" every track is a single (her 11-21 call: "if people are building quad tracks they probably
want the flexibility of treating them as single tracks"); two doubles side by side is one layout, ABBA or AABB
others, all made by setting ways. The game pairs tracks for default return runs by 01's rule, out of sight.

A casual player meets nothing new: she draws a double with the Double track icon, click, click, and gets one bundle
of two tracks, ways by the running side. Nothing new shows on ordinary double: no chip, no picture choice, no new
tag line. Quad by direction is a layout inside More, never a top-level button.

## One rule all tools share: the pointer picks tracks

Given a bundle, a place along it, the pointer's offset across it and what is asked for (each track owns the band
halfway to its neighbours):

| Asked for | Tracks picked |
|---|---|
| a single | the track under the pointer |
| k tracks, fewer than the bundle has | the track under the pointer, then its neighbours, the nearer the pointer first, until k |
| the bundle's own count | all of them |

Partners never decide a pick: a partner need not be a neighbour (A A B B pairs 0-3). Zoomed out, where the bundle
draws under 8 px wide, a single is the track on the pointer's side of the centre line, k tracks the k nearest the
middle.
One sim function behind both the preview and the click (as `preview_station` is today), so the ghost and the click
cannot disagree. The extension, the station tool and the island tool all call it.

## The track bar (stage 6a for More)

| Control | What it does |
|---|---|
| 1, 2 icons | Single track, Double track, as today |
| **More** | a button showing its count ("4 tracks", landed in 2b at 4 only). From 6a its chevron opens a small panel: a count stepper, 3 to 16, and **Side by side** / **By direction**, presets of the tracks' ways. Both remembered per browser |
| Advanced row | the two target speeds, as today. No track counts and no crossover icons |

The More button replaces today's Quad track icon. With Tram picked More has side by side only.

Ways laid by each layout, slot 0 first (A and B are the two ways; which is which follows the running side):

| Count | Side by side | By direction |
|---|---|---|
| 3 | A, both, B | A, both, B |
| 4 | A B A B (paired 0-1, 2-3) | A A B B (paired 1-2, 0-3) |
| 6 | A B A B A B | A A A B B B |

**Price**: a track with a track beside it running the other way (a two-way track counts as the other way) costs half
a double; any other track costs 0.6 of a double. Single 0.6, double 1.0, 4 tracks side by side 2.0, A A B B 2.2,
A B B A 2.0, three tracks 1.5, six side by side 3.0. The price is set when track is laid or rebuilt; setting a built
track's way stays free, as setting one way is today.

## Drawing and extending from a bundle's end

| Start | What carries on | Ghost | Stage |
|---|---|---|---|
| Open ground | a new bundle of the count picked | each track at its offset | 3d |
| A loose end, same count | every track | the stroke runs on from the end | 3e |
| A double's end, Single picked | **the track under the pointer only** (zoomed out, the one on the pointer's side), never both | the single starts from that track's end; the other keeps its loose-end ring | 3e |
| A loose end of more, fewer picked | the tracks the shared rule picks; the others stay loose ends there | the new tracks start from the picked tracks' ends | 3e (side by side), 6b (by direction) |
| A double's end, More 4 side by side | the double carries on as one of the doubles; the new double starts beside it on the left with a turnout from each same-way track | all four, the new double leaving the old | 6a |
| A double's end, More 4 by direction | the double becomes the inner double; each outer track peels off its own side from the same-way inner track, so nothing crosses | as left | 6b |
| An A A B B end, Double picked, pointer on a side | the outer double heads off that side; its far track goes over the inner double on a flyover (the flying split) | the flyover leg as today's flyovers | 6b |
| Mid-way along a bundle (a branch) | the new tracks join by turnout the same-way track nearest them | the join dot on the picked track, as today | 3e |

There is no automatic merge: today's orange dot at a double's loose end, where both tracks merged into one single,
goes. To merge two singles, draw one into the other (a turnout). Old saves' merged joins load as they are.

Refusals are today's (curve, water, ramp room). A flying split with no room is built flat with today's "No room for
a flyover here" (reused). With the ramp switch off the far track crosses the inner double flat, as junctions do.

## Crossovers are single track (stage 8)

There is no crossover tool and no crossover record. A crossover is an ordinary single track drawn from one track to
another: track tool, Single, start on a track (the pointer picks it, the join dot shows which), end on the other.

- Where both ends lie on neighbouring parallel tracks of one bundle, the tool lays the crossover's S itself, at the
  flyover target max speed (80 km/h by default: about 85 m for 4 m across; open below). The ghost is the S, the tag
  today's single-track price and speed.
- It is a single-track bundle with a turnout at each end: priced as single track, picked, reshaped and deleted like
  any track. A scissors is two of them crossing (a flat crossing inside the bundle, allowed).
- Deleting one that lines run over reroutes them or is refused naming them: "{line} has no other way between its
  stations." (reused).
- Old T-338 crossover edges load as such singles; "A crossover needs double track." and the crossover kind in the
  track panel go.

**Turnback crossovers.** Until stage 8 the compile keeps internal turn moves where lines turn (today's figures). From
stage 8, where a line turns back on double track with no crossover there, the stop edit lays a real crossover single
as blueprint with the line's edit, priced as single track (old saves get theirs free at conversion). Capacity books a
turning line's moves over it as today's turn place; a through line using it pays the crossing as at any junction.

**Build crossovers?** When a line is set to use a track or platform it cannot reach for lack of crossovers (a stop
click, the track chip, exact stops, a route), the refusal says so and offers **Build crossovers**: one click lays
crossover singles nearest the need, clear of platforms and other nodes, shown as a blueprint with their price, and
the line edit goes through with them once confirmed. Before stage 8 the refusal only says so (draw a single between
the tracks). Part 4 has the stop-editing side.

## The second click and long selections (stages 3e, 3f)

- **Mouse**: click track that is already the whole pick again (a separate click). The pick narrows at once to the
  single under the pointer, then back to the whole, on any count (her 11-21 call; 2b's "4 tracks, then one double"
  changes at 3e).
- **Touch and everywhere**: the selection bar holds a small **track picture**, one short stroke per track, red one
  way and blue the other, grey for two-way, the picked ones bold. A click or tap on a stroke picks that single. On the
  phone's one row bar it is one 36 px button that opens it larger above the bar.
- **Double click**: on a bundle pick, the run of whole bundles (T-1680, as today). On a single picked by the second
  click, it **traces that track** along its continues out to the next branch points (junction nodes) both ways.
  Shift+click on track further along extends the trace there. A double click's first click narrows like any click,
  so nothing waits on a timer.
- The track panel only informs (T-3746); its title is the case picked: "4 tracks"; for a single, "Eastbound track"
  (or "Two way track") with "Part of double track" or "Part of 4 tracks" under it.
- The Delete tool takes what a first click picks: the whole run, all its tracks. One track goes by picking it with
  the second click and the bar's Remove.

What the selection bar does with one track or a traced track picked:

| Action | Control | Refused when |
|---|---|---|
| Delete one track | the bar's Remove, acting on the picked track only. Platforms serving only it go; a station left with none goes too, asked as a station removal | a line has no other way: "{line} has no other way between its stations." (reused) |
| Set its way | the bar's one way buttons ("Run trains one way", "Reverse direction", reused); off is both ways. One edit for the whole trace | the same, naming the line |
| Set its purpose | the Purpose choice (below) | |

**Way spreading** (3f, the T-3662 fix): a way set on one piece spreads along that track through its continues to the
nearest branch points both ways, so a one-way leg never turns two-way part way. The notice says how far: "Set one way
for 3.2 km, out to the junctions."

## Add a track (T-4024, after stage 4a1)

Her 11-21 call: "we basically maybe dont want a specific "double up" but rather we just want like "add a track"
instead". It replaces Double up (master) and the Select bar's "4 tracks" (2b, tw/track only).

- **Select bar**: 1 and 2 stay as today for runs of one or two tracks (greyed on more). **Add a track** sits beside
  them on any pick of runs, of any count up to 16: one undoable edit, one new track per run. It goes on the side with
  more room (away from the nearest other track), else the other side.
- **The track picture's +** (3e's picture): with the whole run picked, a + at each edge adds one track on exactly
  that side.
- **Its way**: the other way from the track beside it, so a double becomes A B A, then A B A B (two doubles, as
  Double up made). Beside a two-way track the run becomes a double, both tracks taking their ways by the running
  side; lines on the old track are rerouted or the edit is refused naming them, as a way change is. Any way can be
  set afterwards (3e, 3f). Its partner comes from 01's rule (A B A leaves the new track unpaired until a fourth).
- **Connections**: none along the run, no crossovers (her follow-up on decision 8: "this should mostly leave it an
  island unless they actually build crossovers"). At the run's end nodes it joins like a track drawn into them, to
  the nearest track running its way (recommended, her call: Double up joins its new pair there today, and loose ends
  would leave track no line can reach before 8b). A line that needs it part way gets the Build crossovers offer.
- **Stations**: it runs past them with no platform (platforms stop splitting track at 4a1); a platform is added in
  the station panel picture (4e).
- **Price** (the per-track rule above): the run's price after minus before, so a single to double costs 0.4 of a
  double, a double to three tracks 0.5. Blueprint changes at once; on built track it is new blueprint beside the
  built tracks (built state per track) until Construct pays for it.
- **Going down**: pick one track (second click) and Remove (3e); there is no "remove a track" button.

**Refusals.** Each names the reason the edit already knows; the general one is her own words.

| Why | Raised by | Text (draft) |
|---|---|---|
| The new track's curve is under the minimum radius (the inside of a curve) | `GeomIssue::RadiusTooSmall` on the new track (Double up's `tight` list) | "The curve is too tight for another track here." |
| Other track lies where the new one would run | `crossingOverlap` from the trial | "Other track is in the way." |
| Ground-level track would run into water | `groundOverWater` from the trial | "Water is in the way." |
| A track crosses the run on a ramp less than a level apart | `crossing` from the trial | "Tracks cannot cross on a ramp unless they are a full level apart." (reused) |
| A platform with width lies on that side (from 9a) | new check, 9a | "A platform is in the way." |
| The run has 16 tracks | new check, T-4024 | "16 tracks side by side is the most." |
| The pick is a crossover | `Issue::Crossover` | "A crossover stays a single track." |
| Anything else the trial refuses | the rest | "Additional track cannot fit here." (her words) |

A bridge, a tunnel or a level change gives no reason of its own: the new track takes the run's levels and ramps,
and the game has no structure width. `Alongside` ("This track is part of 4 tracks already.") and `Tracks` ("Track has
to be single or double.") go: any count may take another track. `tooManyPorts` is never raised by it (a node counts
bundle ends). With several runs picked and some refused: "Added a track on {done} of {total} runs. {why}".

## Track purpose (stage 7c)

From her notes ("designate each track at a station to a line with a purpose (like passing track)") and decision 10.
Set per track or for a traced track, in the selection bar's **Purpose** choice:

| Purpose | Meaning (exact rules open below) |
|---|---|
| Any line | the default |
| Passing track | a stopping train there waits for a faster one to pass; trains that don't stop keep off it |
| Only for {lines} | other lines' routes treat it as absent. Picking it opens the line list with checkboxes |

The track panel shows the purpose under the title ("Passing track", "Only for 4 and 5"). The station picture marks
a passing track with a small P at its end. Lines already on a track that a purpose closes to them are rerouted or the
edit is refused naming them: "{line} runs on this track."

## The station tool (stage 4d)

The station bar keeps Platform length and offers three **templates**:

| Template | Over single track | Over double or more |
|---|---|---|
| **Side platforms** (default) | a passing loop with a platform outside each track (today's two way station; real track from 7a) | a platform outside each of the two tracks the shared pick gives (the track under the pointer and its neighbour nearer the pointer); all of a double |
| **Island** | refused: "An island platform goes between two tracks." | one platform in the gap under the pointer, serving the tracks either side: zero width at 4d, with the tracks spread for a 6 m island from 9a |
| **Single platform** | one platform; trains can't pass | one platform on the track under the pointer, on its side nearest the pointer |

To cover all of 4 tracks, place twice; a platform placed overlapping a station's side by side joins it. Quick-place
layouts for more tracks and copying a station's layout come at stage 13.

| | |
|---|---|
| Ghost | today's blue boxes at true length, plus the **preview**: the tracks the station covers drawn highlighted under the ghost, each with a chevron for its way; tracks it passes without a platform dimmed |
| Tag | words only where the picture is not enough: "Westbound trains only" for a single platform on double, "Adds a platform to {station}" when it joins one |
| Click | places the station: one edit, one undo |
| Refused | today's reasons (reused: "There is already a station here.", length, ramp, joint, crossing). New: "These tracks have a platform here already." |

## The station panel's platform picture (stage 4e)

The station across its tracks, drawn as on the map with the tracks turned to run left to right, north side up (west
up for north-south track, the stop list's north-up rule).

- **Tracks**: red one way, blue the other (her T-3348 call), grey with both arrows for two-way; red is the way heading
  east (north on north-south track) at every station.
- **Platforms**: grey blocks. Every gap a platform could take is a dashed box. Click a block to remove it, a dashed box
  to add one (blueprint at once; built priced in its hover). The last platform can't be removed here ("A station needs
  one platform"); Remove station does that.
- **Lines**: each track's stopping lines as bullets at its right end.
- **Groups**: platforms on different bundles are one group each, labelled by level when levels differ, highest on top.
- **Faces** (9b): a mark on each platform edge; a click goes through boarding and getting off, boarding only, getting
  off only (the Spanish solution).
- **Join to station** (a button): the next click on a station within 300 m makes them one station, for the name; the
  tool notes show "Joining {station} to a station". Refused: "Stations join within 300 m." Lines need no join: a
  two-way stop already finds the platform beside (part 4).
- **Make a separate station** (a button by each group's label, with two groups or more): the group becomes a new
  station. Lines keep their stop points, so nothing is refused.
- Removing a platform a line needs re-picks that stop (part 4) or is refused naming it: "{line} has no platform on its
  track here." The panel's "Station type" row and "One way station ({way})" go: the picture shows both.

## Island platforms with spread (stage 9a)

Her words: "if we insert an island platform we should blueprint a reconstruct that shifts the tracks out a bit
(curvature determined by target max speed, or maybe even flyover maxspeed) and plops an island in between."

- The Island template, from 9a, spreads the two tracks on an S each end to a **6 m** island, the tracks outside them
  shifting with them. The S is laid for the bundle's target max speed, so trains passing never slow; where that does
  not fit before a node, for the flyover target max speed. `2 * sqrt(R * 3 m)`: about 70 m each end at 80 km/h, 140 m
  at 160, 290 m at 320; a 200 m island at 160 km/h takes about 480 m of track. The tag adds "Tracks spread over 140 m
  each end".
- Placed as **blueprint**, even on built track; trains run on the old track until Construct, which pays the full price
  of the re-laid stretch (today's Rebuild rule, nothing back) plus the station.
- Refused: "No room to spread the tracks here." where even the shorter S reaches a node, a junction or a platform.
- Today's box is 11.25 px wide from zoom 15, about 20 m at New York's latitude, so a 6 m island reads as one from
  about zoom 16.
- 9c: a change between the two edges of one island counts 1 minute instead of 2 (decision 23).

## Hovers and tags

- **Track hover**: the case's icon in the type's colour, no words (T-3714): 1 stroke, 2 strokes, and for more the
  count on a chip. No halos or rings.
- **Placing tags**: unchanged but for the station preview and the island's spread line. Which tracks carry on is shown
  by the ghost's strokes and today's join dot.

## Pieces and hours

| Piece | Stage | Hours |
|---|---|---|
| Shared pointer pick in the sim, with preview | 3e | 3-4 |
| Second click, track picture, delete and set way of one track, no auto-merge, extending | 3e | 6-8 |
| Tracing a track, Shift+click to extend, way spreading to branch points | 3f | 4-6 |
| Station templates, the tag's preview, join by overlap | 4d | 6-8 |
| Station panel picture, add and remove, join and separate | 4e | 6-8 |
| Add a track (Select bar and the picture's +), its way, ends and refusals, the price rule | T-4024 | 3-5 |
| More button's count and layouts | 6a | 3-4 |
| By direction drawing, peel-off, flying split | 6b | 6-8 |
| Track purpose (sim rules and the Purpose choice) | 7c | 6-9 |
| Crossover S between parallel tracks, turnback crossovers, Build crossovers offer | 8 | 7-10 |
| Island spread as blueprint | 9a | 8-12 |
| Faces, with boarding in the sim | 9b | 4-6 |
| Island change minute | 9c | 2-3 |
| Quick-place layouts for more, copying a station's layout | 13 | 4-6 |

## New and changed player text (drafts, none final)

| Where | Old -> new |
|---|---|
| Track bar quad icon (`toolBar.quadTrack`, `quadTrackTip`) | "Quad track", "Quad track (two parallel double tracks)" -> "4 tracks" (landed in 2b), then "{n} tracks" (6a) |
| More panel (6a) | new: "Tracks", "Side by side", "By direction" |
| Help answer (`help.aFourTracks`) | master: "Quad tracks are actually just two double-tracks side by side. For trains to run on both pairs ..." (2b: "4 tracks are actually just two double tracks side by side. ...") -> "4 tracks are four single tracks side by side, each running one way or both ways. Lines use the tracks their stops are on." (T-4024; 8b adds "The game offers to build crossovers where a line needs to change tracks.") |
| Select bar Double up (master; 2b's "4 tracks") | "Double up", hover "Double up to quad track for {price} (blueprint)" / "Double up to quad track on # runs for {price} (blueprint)" -> "Add a track", hover "Add a track for {price} (blueprint)" / "Add a track on # runs for {price} (blueprint)", then "Each run goes to the next junction or track end" and "Some cannot: {why}" (reused) |
| Its notices | "This track cannot be doubled up." -> the reason (table above), else "Additional track cannot fit here."; "{done} of {total} runs doubled up. {why}" -> "Added a track on {done} of {total} runs. {why}" |
| Its refusals | new: "The curve is too tight for another track here.", "Other track is in the way.", "Water is in the way.", "A platform is in the way." (9a), "16 tracks side by side is the most.", "A crossover stays a single track."; removed: "This track has a pair beside it already." (2b: "This track is part of 4 tracks already."), "Track has to be double or quad." (2b: "Track has to be single or double.") |
| Track panel titles (`trackInspector.name`) | "Crossover" -> removed; new: "{n} tracks" ("4 tracks" landed in 2b), "Eastbound track" (westbound, northbound, southbound), "Two way track", "Part of double track", "Part of {n} tracks" |
| Track picture + | new hover: "Add a track on this side", then the price |
| Way notice | new: "Set one way for {length}, out to the junctions." |
| Crossover refusal (`issues.singleCrossover`) | "A crossover needs double track." -> removed |
| Build crossovers offer | new: "{line} can't reach that track without a crossover." with buttons "Build crossovers" and "Cancel"; then the blueprint's "Build {n} crossovers for {price}" |
| Purpose | new: "Purpose", "Any line", "Passing track", "Only for {lines}", refusal "{line} runs on this track." |
| Station bar (`toolBar.oneTrackStation`, `bothTracksStation`) | "One way station" -> "Single platform"; "Two way station" -> "Side platforms"; new "Island" |
| Station tag | new: "Westbound trains only" (and the other ways), "Adds a platform to {station}" |
| Station refusals | new: "These tracks have a platform here already.", "An island platform goes between two tracks.", "No room to spread the tracks here." |
| Station panel (`stationInspector.platformTracks`, `oneWay`) | "Station type", "One way station ({way})" -> removed, the picture shows them |
| Station panel picture | new hovers: "Click to add a platform" (then the price), "Click to remove the platform", "A station needs one platform"; track hover "Eastbound" |
| Faces | new: "Boarding and getting off", "Boarding only", "Getting off only" |
| Station panel buttons | new: "Join to station", "Make a separate station"; notes "Joining {station} to a station"; refusal "Stations join within 300 m.", "{line} has no platform on its track here." |
| Island tag | new: "Tracks spread over {length} each end" |
| Reused as they are | "Single track", "Double track", "No room for a flyover here", "{line} has no other way between its stations.", "Run trains one way", "Reverse direction", "There is already a station here." |

## Still open for her

1. Track purpose's exact rules. (Passing track: a train stopping there waits only for a train of another line that
   is faster between the next two stations both serve, as T-3346's passing waits work today; trains that don't stop
   there avoid it while another track their way is free of purpose. Only for {lines}: routing only, so other lines
   route around it and are refused where they can't.)
2. Crossover speed: the S between neighbouring tracks is laid for the flyover target max speed (80 km/h by default,
   about 85 m long), or for the bundle's target max speed (160 km/h, about 170 m)? (flyover speed: trains slow at
   turnouts anyway, and a short crossover fits between stations)
3. Turnback crossovers at a line's ends come with the stop edit and its cost, without the Build crossovers offer?
   (yes: every double-track line needs them)
4. Add a track: does the new track join the junctions at the run's ends (as Double up's pair does today), or end
   loose there, a full island until crossovers are built? (join at the ends: otherwise no line can use it before 8b,
   and her "mostly leave it an island" still holds along the run)
