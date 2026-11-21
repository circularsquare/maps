# T-3444 design, part 3: drawing and editing tools

Agent tdtools, 2026-11-21. Design draft for discussion with Anita; nothing is built. Words as in `01-model.md`
(track, bundle, slot, offset, node, connection, platform, stop point, station, preset). Stage numbers are
`10-build-plan.md`'s. Today's tools: `docs/ui-building.md` "The track and station bars", "Drawing track", "Placing
stations", "The Select tool". Her calls behind this: T-3365 (11-12 answers), T-3348 (her evening calls).

## What a casual player meets

Nothing new. She draws a double the way she does today: Double track icon, click, click. The stroke lays one bundle
of two tracks, ways by the running side. Slots, offsets and pairs never appear in a string. There are only two ways
to reach single tracks: a second click on picked track, and the track bar's Advanced row (closed by default).

T-3348's four guards hold for every tool below:
- The station bar's 4 shows only with the pointer over a bundle of 4 or more.
- Quad by direction is a layout in the Advanced row and a choice in the Select bar, never a top-level button.
- Picking tracks with the pointer at a bundle's end only matters at ends with more tracks than the count picked.
  Ordinary double behaves as today, including T-3204's single-track pick at a double's end.
- Nothing new shows on ordinary double: no chip, no picture choice, no new tag line.

## One rule all tools share: the pointer picks tracks

Every bundle stores its **pairs**: which track is each track's partner (the nearest other-way track of its own
double). The preset sets them: side by side pairs slots (0,1), (2,3) ...; by direction pairs them nested, (1,2)
inside (0,3). 01-model stores them as each track's partner. Odd counts have a two-way middle track with no partner.

Given a bundle of n tracks, a place along it, the pointer's offset across it and a count k to pick (each track owns
the band halfway to its neighbours):

| k | Tracks picked |
|---|---|
| n | all of them |
| 1 | the track under the pointer |
| even | k/2 pairs: the pair holding the track under the pointer, then the pairs next to it nearer the bundle's middle |
| odd, 3 and up | as k - 1, plus the two-way middle track where the bundle has one, else the unpicked track nearest the pointer |

Checks: on a double, 2 is both and 1 the track under the pointer (today). On side by side quad, 2 is the pair under
the pointer. On A A B B, 2 on a side is the outer pair (T-3348's "side"), 2 on the middle the express pair. On 6
side by side, 4 is the pair under the pointer and its neighbour toward the middle. On 3 tracks, 2 is the pair (the
outer two) and 1 any one.

Zoomed out, where the bundle draws under 8 px wide, the pointer cannot tell tracks apart: it takes the pair nearest
the bundle's middle, or all tracks when the count equals the bundle's (today's T-3204 dead zone, generalised).

The rule is one sim function behind the preview and the click (as `preview_station` is today), so the ghost and the
click cannot disagree. The extension, the station tool, the island tool and the crossover tool all call it.

## The track bar

| Control | What it does | Stage |
|---|---|---|
| 1, 2, 4 icons | as today: single, double, quad. Quad lays one bundle of 4 | 2 |
| Advanced row, **Tracks** stepper | 1 to 16, any count. The icons and the stepper show the same choice: stepping to 4 lights the 4 icon, 6 lights none | 6 |
| Advanced row, **Side by side** / **By direction** | the layout for counts of 3 and up. Remembered per browser with the row | 6 |
| Advanced row, **Crossover** / **Scissors crossover** | turns the track tool into crossover placing (below) | 8 |

Ways laid by each layout, slot 0 first (A and B are the two ways; which is which follows the running side):

| Count | Side by side | By direction |
|---|---|---|
| 3 | A, both, B | A, both, B |
| 4 | A B A B | A A B B (locals outside, express pair inside) |
| 6 | A B A B A B | A A A B B B |

With Tram picked: the 1, 2 and 4 icons as today; the expander shows, and its row holds only the two crossover icons
(tram takes crossovers, T-3575). No counts past 4 and no by direction for tram.

**Price.** Today single is 0.6 of a double and quad 2.0. Proposed rule, one line: *a track with a track beside it
running the other way costs half a double; any other track costs 0.6 of a double.* It gives single 0.6, double 1.0,
quad 2.0, AA/BB 2.2 (her price), ABBA 2.0, six side by side 3.0. A two-way track counts as running the other way,
so three tracks (A, both, B) are 1.5. The price is set when track is laid or rebuilt; flipping a built track's way
stays free, as setting one way is today, so a built quad flipped to A A B B costs nothing more. Her call.

## Drawing and extending from a bundle's end

| Start | What carries on | Ghost | Stage |
|---|---|---|---|
| Open ground | a new bundle of the count picked | each track at its offset (from stage 3) | 2 |
| A bundle's loose end, same count | every track (continue) | the stroke runs on from the end | 2 |
| A bundle's loose end, fewer picked | the tracks the shared rule picks. The others stay loose ends there | the new tracks start from the picked tracks' ends; the left-behind ends keep their loose-end rings | 2 (side by side), 6 (by direction) |
| A double's end, 4 picked, side by side | the double carries on as one pair; the new pair starts on the left with a turnout from each same-way track (today's Double up side, T-255) | all four tracks, the new pair leaving the old | 2 |
| A double's end, 4 picked, by direction | the double becomes the middle pair; each outer track peels off its own side from the same-way middle track, so nothing crosses (T-3348) | as left | 6 |
| A by-direction end, 2 picked, pointer on a side | the outer pair heads off that side; the far outer track goes over the middle pair on a flyover (T-3348's flying split) | the flyover leg as today's flyovers | 6 |
| Mid-way along a bundle (a branch) | the new bundle's tracks join by turnout to the same-way track at the nearest offset (01-model) | the join dot on the picked track, as today | 3 |

Refusals are today's (curve, water, ramp room). One new case: a flying split with no room is built flat with
today's "No room for a flyover here" (reused). With the ramp switch off it crosses flat (T-3348 call 6, still open).

## The second click: one double or one track

- **Mouse**: click track that is already the whole pick again (a separate click, not a double click: a double click
  still picks the run, T-1680). The pick narrows at once: bundle, then the pair under the pointer (bundles of 3 or
  more), then the track under the pointer, then back to the bundle. A double click's first click narrows like any
  click, then the double click picks the run of whole bundles, so nothing waits on a timer. Works while plain
  selecting and with the Select tool.
- **Touch and everywhere**: the selection bar holds a small **track picture**: one short stroke per track, red one
  way and blue the other, grey for two-way (the station picture's colours), the picked tracks bold. Tap or click a
  stroke to pick that track. This is the visible control; the map's second click is the shortcut. On the phone's one
  row bar it is one 36 px button showing the picture, which opens it larger above the bar.
- The **track panel** only informs, as today (T-3746). Its title says what is picked: "Quad track", then for a pair
  "Double track" with "Part of quad track" under it, then for one track "Eastbound track" with the same second line.

What the selection bar does with a pick of one track (stage 6):

| Action | Control | Refused when |
|---|---|---|
| Delete one track | the bar's Remove, now acting on that track only. Platforms serving only it go; a station left with none goes too, asked as a station removal | a line has no other way: "{line} has no other way between its stations." (reused) |
| Flip or set two-way | the bar's one way buttons, as for single track today ("Run trains one way", "Reverse direction", reused) | the same, naming the line |
| Add a track | a **+** at each edge of the track picture, shown when the whole bundle is picked | at 16 tracks, the + is not shown |

An added track goes 4 m past the outermost on that side, running the same way as its neighbour. At each end it joins
the track its neighbour carries on into (a turnout), so trains can use it at once; at a loose end it ends loose. Its
price is the price rule's difference. It is blueprint even beside built track, so Construct builds only it (built or
blueprint is per track, 01-model). Adding a track over only part of a bundle is the
passing tracks preset (stage 7).

The delete tool still removes the whole bundle (one piece, red, as today). One track goes only through the pick.

## The Select bar's counts (stage 2, above 4 in stage 6)

- The bar shows the track bar's icons: 1, 2, 4, and while the track bar's Advanced row is open, its Tracks stepper
  and the two layouts. The **4 replaces Double up**: the Double up button and its four strings go.
- Going up adds tracks by the layout: side by side adds pairs on the left (as Double up did), by direction adds one
  outer track each side. No platforms are added (settled).
- Going down keeps the tracks lines run on; a line that cannot move is named and the edit refused: "{line} has no
  other way between its stations." (reused). Platforms on removed tracks go; stops move by the simple-stops rule
  (part 4) or the edit is refused: "{line} has no platform on its track here." (T-3348's draft).
- Blueprint changes at once and free; built is a Rebuild choice, as today's 1 and 2 (T-3321). With built state per
  track, a built pick going up pays for the new tracks only. Today's 1 to 2 Rebuild charges the full double; going up
  would then be cheaper than that rule. Her call.

## The station tool (stage 4)

The station bar keeps Platform length and gains count icons. The pointer picks which tracks by the shared rule.

| Icon | Shown | Over double | Over single |
|---|---|---|---|
| 1 | always | a platform on the track under the pointer (T-3332's one way station) | a station with no passing |
| 2 | always | platforms on both tracks (today) | a passing loop (today's drawing; real track in stage 7) |
| 4 | over a bundle of 4 or more | | |
| N | over a bundle of 5 or more, showing its count ("6") | | |

**Where the platforms go** (zero width, today's look), for any count:
1. Working from the bundle's middle outward, two picked neighbours running the same way share one platform between
   them (AA/BB's cross-platform change).
2. Every other picked track gets a platform on its side away from its partner. A two-way track goes on its side away
   from the bundle's middle; the middle track of an odd bundle on its north side.

| Layout, all tracks picked | Platforms |
|---|---|
| Double A B | two side platforms (today) |
| Quad side by side A B A B | four: one each side of each pair, two of them sharing the middle gap (as 02 and old saves) |
| A A B B | two islands, between the two A tracks and between the two B tracks |
| A B B A | one island between the B tracks, a side platform outside each A track |
| Three tracks A, both, B | a side platform outside each outer track, one on the middle track's north side |
| Six by direction A A A B B B | islands between the middle two A and the middle two B, side platforms outside |

| | |
|---|---|
| Click | places the station (one edit, one undo, as today) |
| Ghost | today's blue boxes, at true length, on the picked tracks' sides |
| Joins a station | a platform placed overlapping one of a station's side by side (02-station's rule) joins it. The tag's first line says "Adds a platform to {station}" |
| Refused | today's reasons (reused: "There is already a station here.", the platform length, ramp, joint and crossing ones). New: "These tracks have a platform here already." when every picked track already has one there |

## The station panel's platform picture (stage 4)

A small picture of the station across its tracks, drawn as seen on the map with the tracks turned to run left to
right, the north side up (west up for north-south track; the stop list's north-up rule).

- **Tracks** as lines with arrows: red one way, blue the other (her T-3348 call); grey with both arrows for a
  two-way track. Red is the way heading east (north for north-south track) at every station, so a line's colours
  agree from station to station.
- **Platforms** as grey blocks. Every gap a platform could take (outside each edge, between each two tracks) is a
  dashed box. Click a block to remove it, a dashed box to add one (blueprint at once; built is priced in its hover).
- **Lines**: each track's stopping lines as bullets at its right end.
- **Groups**: platforms on different bundles (stacked levels, offset platforms joined by the player) are one group
  each, labelled by level ("Level −1") when levels differ, the highest on top.
- **Faces** (stage 9): a small mark on each platform edge; a click goes through boarding and getting off, boarding
  only, getting off only (the Spanish solution).
- The panel's "Station type" row and its "One way station (eastbound)" go: the picture shows both.
- **Join to station**: a button under the picture. The tool notes show "Joining {station} to a station" with the
  usual cross, and the next click on a station within 300 m makes them one. Refused: "Stations join within 300 m."
- **Make a separate station**: a button beside each group's label, shown when the station has two groups or more.
  The group becomes a new station, named as new stations are. Refused when a line's simple stop uses both parts:
  "{line} stops on both parts. Take one out of its stops first."

Removing a platform a line needs is refused naming it: "{line} has no platform on its track here." The last
platform cannot be removed in the picture (its hover: "A station needs one platform"); Remove station does that.

## The island platform tool (stage 9)

Her words: "if we insert an island platform we should blueprint a reconstruct that shifts the tracks out a bit
(curvature determined by target max speed, or maybe even flyover maxspeed) and plops an island in between."

| | |
|---|---|
| Where | an **Island platform** icon in the station bar. Pointer between two tracks of a bundle (the gap nearest it) |
| Ghost | the two tracks spreading on an S each end to 6 m apart plus the island, the tracks outside them shifting with them. The tag adds "Tracks spread over 150 m each end", the S's length |
| Click | places the island and the spread as **blueprint**, even on built track. Trains run on the old track until Construct |
| Construct | the full price of the re-laid stretch (the S's and the platform's length, today's Rebuild rule: nothing back for what is there) plus the station |
| Joins | a station there by overlap, as the station tool does |
| Refused | "An island platform goes between two tracks." on single track or outside a bundle. "No room to spread the tracks here." where an S would reach a node, a junction or a platform. Today's ramp, length and crossing reasons (reused) |

The S comes from 01-model: `2 * sqrt(R * d)`, with d = 3 m each side. At 80 km/h (the flyover default) it is about
70 m each end, at 160 km/h about 140 m, at 320 km/h about 290 m. A 200 m island at 160 km/h takes about 480 m of
track. Which speed sets R is her call. Today's drawn box is 11.25 px wide from zoom 15, about 20 m at New York's
latitude, so a 6 m island inside it reads as an island only from about zoom 16 (6 m is 7 px there); further out it
looks like any station.

## The crossover tool (stage 8)

| | |
|---|---|
| Where | Crossover or Scissors crossover in the Advanced row (rail and tram). Descriptor "Placing crossovers" |
| Pointer | over a bundle of 2 or more tracks; the gap under the pointer, between two neighbouring tracks |
| Ghost | a 30-60 m diagonal (both for scissors) in the ghost look, between those tracks. One diagonal leans forward along the way of the track nearer the pointer |
| Click | places a node there (01-model: a split, continue for every track) with the crossover record |
| After | the crossover is picked whole, as today; the selection bar adds **Flip** for one diagonal |
| Refused | "A crossover needs double track." (reused) on single track. "There is already a crossover here." "No room for a crossover here." within 30 m of a node or a platform end. Today's ramp and crossing reasons |

The game's own crossovers where lines turn are not records: they are the sim's turn moves, drawn where a line uses
one and never priced as crossings (part 5, "Automatic"). They look like a placed crossover but cannot be picked or
removed; a player who wants a real one places it with this tool.

## Hovers and tags

- **Track hover**: the count icon in the type's colour, no words (T-3714). 1, 2 and 4 strokes as today; by
  direction a 4 icon with its outer strokes set apart; 3 and 5 or more as the number on a chip. No halos or rings
  (her 2026-11-18 call).
- **Track panel titles**: "Single track", "Double track" (reused), "Quad track", "Quad track by direction", "6
  tracks"; one track picked: "Eastbound track" (or "Two way track") with "Part of quad track" under it.
- **Placing tags**: unchanged except the station tool's "Adds a platform to {station}" and the island's spread line.
  Which tracks carry on is shown by the ghost's strokes and today's join dot, never by words.

## Pieces and hours

| Piece | Stage | Hours |
|---|---|---|
| Shared pointer pick in the sim, with preview | 2, by direction in 6 | 3-4 |
| Extension from bundle ends, side by side | 2 | 3-4 |
| By direction drawing, peel-off and flying split | 6 | 6-8 |
| Select bar 4 replacing Double up, pick narrowing by second click | 2 | 3-4 |
| Track picture in the selection bar; delete, flip and add one track | 6 | 6-8 |
| Advanced row counts and layouts, price rule | 6 | 2-3 |
| Station bar counts, platform placement rule, join by overlap | 4 | 5-7 |
| Station panel picture, add and remove, join and separate | 4 | 6-8 |
| Island tool with blueprint spread | 9 | 8-12 |
| Platform faces (with boarding in the sim) | 9 | 4-6 |
| Crossover tool | 8 | 3-5 |

## New and changed player text (drafts, none final)

| Where | Old -> new |
|---|---|
| Track bar quad hover (`toolBar.quadTrackTip`) | "Quad track (two parallel double tracks)" -> "Two double tracks side by side" |
| Advanced row stepper label | new: "Tracks" |
| Advanced row layouts | new: "Side by side", "By direction" |
| Advanced row crossover icons | new: "Crossover", "Scissors crossover" |
| Descriptor | new: "Placing crossovers" |
| Select bar Double up (four strings) | "Double up", "Double up to quad track for {price} (blueprint)" ... -> removed |
| Track panel titles | new: "Quad track", "Quad track by direction", "{n} tracks", "Eastbound track" (westbound, northbound, southbound), "Two way track", "Part of quad track" ("Part of {n} tracks") |
| Track picture + | new hover: "Add a track on this side", then the price |
| Station bar icons over double | "One way station" -> "Platform on one track"; "Two way station" -> "Platforms on both tracks" |
| Station bar icons over single | "One way station" -> "Trains can't pass here"; "Two way station" -> "Trains can pass here" |
| Station bar 4 and N | new: "Platforms on all four tracks", "Platforms on all {n} tracks" |
| Station tag | new: "Adds a platform to {station}" |
| Station refusal | new: "These tracks have a platform here already." |
| Station panel picture | new hovers: "Click to add a platform" (then the price), "Click to remove the platform", "A station needs one platform"; track hover "Eastbound" |
| Faces | new: "Boarding and getting off", "Boarding only", "Getting off only" |
| Station panel row (`stationInspector.platformTracks`, `oneWay`) | "Station type", "One way station ({way})" -> removed, the picture shows them |
| Station panel buttons | new: "Join to station", "Make a separate station"; notes "Joining {station} to a station" |
| Station panel refusals | new: "Stations join within 300 m.", "{line} stops on both parts. Take one out of its stops first.", "{line} has no platform on its track here." |
| Island | new: "Island platform", "Tracks spread over {length} each end", "An island platform goes between two tracks.", "No room to spread the tracks here." |
| Crossover | new: "Flip", "There is already a crossover here.", "No room for a crossover here." |
| Reused as they are | "No room for a flyover here", "{line} has no other way between its stations.", "Run trains one way", "Reverse direction", "A crossover needs double track.", "There is already a station here.", "Single track", "Double track" |

## Her calls

1. Price by track: a track with a track beside it running the other way costs half a double, any other 0.6 (single
   0.6, double 1.0, quad 2.0, AA/BB 2.2, ABBA 2.0, three tracks 1.5)? (yes)
2. The island's S is laid for the bundle's target max speed, so trains passing through never slow; the flyover speed
   only where that does not fit? (yes: target max speed, flyover speed as the fallback where room is short)
3. Islands are 6 m wide, which reads as an island only from about zoom 16? Or 10 m, which shows a zoom sooner and
   makes the S about 30% longer? (6 m)
4. An added track runs the same way as its neighbour and joins the next track at both ends? (yes)
5. Adding tracks to built track pays for the new tracks only, not today's full price of the whole rebuilt track? (yes)
6. Counts past 4 go to 16 in the Advanced row, any count, odd counts with a two-way middle track? (yes)
7. T-3348's open call 6: with the ramp switch off, the flying split's far track crosses the middle pair flat? (yes,
   as junctions do)

Decided and told rather than asked: two-way tracks in the station picture are grey with both arrows; red is the way
heading east (north on north-south track).
