# T-3444 design, part 4: stop assignment, simple and exact

Agent tdtools, 2026-11-21. Design for T-3444, revised the same day after Anita's answers
(`answers-2026-11-21.md`); nothing is built. Words as in `01-model.md`. Stage numbers are `10-build-plan.md`'s.
Today's stop editing: `docs/ui-building.md` "Making and editing lines" and "Clicks while drawing a route or editing
stops"; the stop list: `docs/ui-inspectors.md` "Stops". Her words: "we do definitely need to preserve simpel stop
assignment for simple 2track lines, for casual player ease", the numbered-circle paragraph in `../T-3444.md` (the
spec for exact stops), and decision 3: "we should also support an automatic smart allocation that makes sense ... and
just figures out a good way to put a two-way stop with a single click".

## Two ways to set a line's stops

- **Simple stops**, every line's default: click stations, as today. The game picks the track each way.
- **Exact stops**, a switch per line beside normal stop editing (T-3079): stop points in running order, out and back,
  with waypoints. One line at a time, not a mode of the game.

Which train uses which track between stops, and where trains turn, is part 5's. Per train waits for exact timetables
(T-3346). This part covers what is stored and how the player picks it.

## Simple stops

### What stays

All of today's editing: one round marker per station, click to add or remove, Shift click for a one way stop
(T-3711), right click for loops, the tiers and their hovers, the gap flash, the "+ 2 stations passed" row, stop
order set manually by dragging rows (T-2895), Sort stops. On a plain two-track line the player sees nothing new.

### Smart allocation: one click, a two-way stop (stage 4a)

A click on a station makes a stop served both ways. At the click the game picks **a stop point for each way** and
stores both on the stop (out and back):

1. A stop point of the clicked station the run reaches going its way: the one with the fewest crossing moves from
   the stop before (so a line stays on the tracks it is on), then the one nearest the pointer.
2. If the station has none for a way, the stop point of the nearest station within 300 m (its complex first) that the
   run passes on its own track going that way.
3. If neither, that way passes the station (today's one way station on double, T-3332).

The two picks are **stored, never re-derived**. An edit that removes a stored stop point re-picks for that stop only.
Shift click still makes a one-way stop (one stop point). Her split one-way legs 150 m apart work in one click with no
joining: the click on either leg takes the other leg for the other way. Joining stations is only for the name. T-3605
holds: two one-way stations listed as separate stops stay separate stops.

On a double each way has one stop point, so the picks are what today gives. On 4 tracks a line drawn along one double
stays on it: an express added on the inner double stops there, a local on the outer. On A A B B each way takes its
own track of the double the line is on.

### The track chip in the stop list (stage 4c)

Shown on a stop where either way had more than one stop point to choose from, or where the two ways stop at different
stations (her split legs). Never on ordinary double, never on single.

- **Looks**: after the station's name, a small picture of the station's tracks (as the station panel's, 14 px tall,
  one stroke per track), the tracks this line stops on in its colour, the rest grey.
- **Click**: a small list, one row per way ("Eastbound", "Westbound"), each with that way's choices as small pictures,
  the picked one ringed. Picking one changes that way only, so a local one way and an express the other is possible
  in simple stops. Hovers on a choice: "Stop on the outer tracks", "Stop on the inner tracks", or with 6 tracks and up
  "Track 3 from the north"; a choice at the platform beside, "Stop at {station}". Counting from the north is the
  station's numbering, never a bundle slot.
- **The edit**: one edit, one undo. That way's stop point changes and the path is found again through it.

### A track the line can't reach

When a stop click, a chip choice or a route needs a track the line cannot get onto for lack of crossovers (added
tracks are not connected until crossovers are built, part 3), the refusal says so and offers the fix:

> "{line} can't reach that track without a crossover." [Build crossovers] [Cancel]

Build crossovers shows the crossover singles the game would lay (nearest the need, clear of platforms and other
nodes) as blueprint with their price, "Build {n} crossovers for {price}", and the stop edit goes through with them
once confirmed. One undo takes both back. From stage 8, where crossovers are single track; before that the refusal
has no button. A track no crossover can reach (another bundle with no junction) gets today's "No track joins it to
this line" (reused). Turnback crossovers at a line's ends come with the stop edit and its cost, without this offer.

## Exact stops (stages 11a, 11b)

### Switching a line (11a)

A toggle in the Stops heading, beside related lines: **Exact stops**. Hover: "Choose each stop's track and the
order trains stop in". Both ways are one edit and one undo.

| Switch | What happens |
|---|---|
| Simple -> exact | the cycle is the line as it runs: each stop's out stop point in order, then each back stop point in reverse. Paths and timetable unchanged. Nothing to refuse |
| Exact -> simple | always works. If the cycle reads the same both ways station by station, it folds to an open line, each stop keeping both its stop points from the cycle, so no track changes. Otherwise it stays a line with those stops as a loop or turned line, as the 2026-11-10 rule. Waypoints are dropped; the notice says so: "Back to simple stops. 1 waypoint was dropped." |

Against DECISIONS 2026-11-10 (T-3079, T-3089, T-3109):

| Old rule | Here |
|---|---|
| Switching Exact edits off always works; a list that reads the same both ways folds, any other stays a normal line with those stops (line 1772) | kept, comparing stations; folding keeps the stop points |
| The route search turns back at a listed stop at no cost (1772) | kept for exact lines, at a stop point or a bare track end |
| A plain numbered list in running order, no related-lines drawing (1772) | kept |
| Cross, dot click and map number are one edit, one visit; a stop with two visits loses the first (1775) | kept: a chosen circle is one visit |
| An empty circle on the right of each pass over a station with no stop (1775) | replaced by a circle per stop point, which already is one pass's track |
| Numbers from zoom 12.5 (1775) | kept, but circles that do not fit merge first (below) |
| Exact edits and manual mode gone (1783) | the switch returns as "Exact stops"; saves' old manual lines already folded |

### What the map shows while an exact line's stops are edited (11b)

- **A circle per stop point** the line could use, with today's tiers (sizes 100%, 80%, 65%, a dashed ring where trains
  would reverse, greyer where unreachable). A track with platforms on both sides (the Spanish solution) gives one
  circle: a stop point is one per station and track.
- **Hover** on a circle: the station's name, its track ("Eastbound track, level −1"), then what a click does; the
  track lights on the map as hovered things do (`hovered.ts`), since a circle pushed along its platform can't show by
  its place which track it is on.
- **Not chosen**: white disc, grey ring, the whole circle tinted by the track's level (the track bar's level colours,
  mixed 25% into the white).
- **Chosen**: the same size and place; the middle in the line's colour with the visit's number (white or dark,
  whichever reads, `legibleOn`); the outer white ring tinted by level (30%), the dark rim outside. Two visits show
  both numbers, "3, 9". A small chevron for the way the visit runs (T-3139's).
- **Waypoints**: a diamond in the line's colour at 60% of a circle's size, no number.
- Text says "level", never "elevation".

### Clicks

| Gesture | On | Does |
|---|---|---|
| Click | a circle not chosen | adds a visit. On a pass of the route over it: after the stop before that pass, path unchanged. Elsewhere: where it costs least in the cycle, as `best_slot` does today. A track it can't reach for lack of crossovers gets the Build crossovers offer |
| Click | a chosen circle | removes that visit; with two visits, the first (1775's rule) |
| Click | a waypoint | removes it |
| Click | a merged circle, none chosen | adds the station by smart allocation: a visit on each pass over it, each on the stop point its way suits. Where a pass could take two (4 tracks) it zooms in until they fit instead (the camera jumps, no flight). Tag: "Click to zoom in and pick a track" |
| Click | a merged circle with chosen visits | removes the first of them; zoom in to pick another |
| Shift click | a circle | adds a waypoint there: trains pass through without stopping (T-3410) |
| Shift click | track | adds a waypoint on that track |
| Right click | as today | loops only (close, open, turn); "Can't loop back here" elsewhere |
| Drag a row | the list | reorders the cycle (T-2895) |
| Touch | | the strip's Loop button stays; a **Waypoint** button beside it makes the next tap a Shift click |

Shift click is a one way stop in simple stops; in exact stops every visit is already one way, so Shift is free for
waypoints, as for ferry waypoints on water (T-3809). Waypoints are free: no dwell, no riders, no price. A run may turn
back at a waypoint only where it could turn at a stop (a stop point or a bare track end).

### The list

Visits in running order, numbered, the closing row back to the first. Each row: number, station name, the track chip
(click: the station's other stop points the run could take there), time and boardings as today. Waypoint rows are
faint: "via {nearest station}", or "via track" with none within 1 km. No related lines.

### Laying out the circles

Marker sizes are today's (`stopMarkers.ts` SIZE): 9 px at zoom 9, 14 at 12, 20 at 14, 24 from 16. Gap 2 px.

1. **Anchor** each circle at its stop point on screen. Its **axis** is its track's direction there. It may slide
   along the axis only, its centre staying within its platform's length, and never across.
2. **Clusters**: circles whose anchors lie within one diameter plus the gap of each other, found on a grid.
3. A cluster whose axes agree within 15 degrees (one bundle, or stacked bundles on one plan line) lays out in **one
   row** along the axis, centred on its anchors' middle, ordered by level (lowest first) then from the north side,
   spaced one diameter plus the gap.
4. A row that does not fit within its circles' slide limits **merges** into one circle at the anchors' middle, drawn
   as a station marker with a pip on its ring per stop point; chosen visits show their numbers on it.
5. **Between clusters** (a complex's stations, crossing lines): up to 8 passes, each pushing overlapping circles apart
   along their own axes by half the overlap, within their limits, in a fixed order by stop point id. What still
   overlaps is drawn lower level first, so the tint tells them apart.
6. Positions are kept in ground metres for the zoom, so a pan moves nothing. They are worked out again when the zoom
   changes by 0.1 or the network or the line changes. Picking reads the moved centres, as `pick` does today.

How many circles a 200 m platform (today's default, 10 cars of 20 m) holds in one row, at New York's latitude, 3.6 m
a pixel at zoom 14 (`stationGeometry.ts`), half that each zoom in. A row of n fits when n x (d + 2) is at most the
platform's length on screen plus d + 2:

| Zoom | Platform on screen | Circle | Circles that fit | So |
|---|---|---|---|---|
| 12 | 14 px | 14 px | 1 | every station merged |
| 13 | 28 px | 17 px | 2 | double splits |
| 14 | 55 px | 20 px | 3 | three tracks |
| 15 | 110 px | 22 px | 5 | 4 tracks split |
| 16 | 220 px | 24 px | 9 | 6 and 8 tracks |
| 17 | 440 px | 24 px | 17 | Grand Central's 16 |

An 80 m tram stop holds 2 from zoom 15 and 4 from 16. The circles in view are a few hundred at most; the layout is a
grid pass and a sort, well under a millisecond.

### More tracks in exact stops

Each visit is a stop point, so the player clicks the circle on the track she wants, each way on its own. The route
search finds the crossing moves between; a visit no track reaches is refused as today ("No track joins {a} and
{b}.", reused), and one that only lacks crossovers gets the Build crossovers offer.

## What is stored

| Field | Simple stops | Exact stops |
|---|---|---|
| Mode | simple | exact |
| Stops | station id, out stop point, back stop point (either may be at the platform beside, or none for a way that passes), one way flag (T-3711), hand order (T-2895) | visits in a cycle: a stop point, or a waypoint (track id, chainage) |
| Path | track spans first to last stop, plus the stored return path where one is routed (01-model) | track spans for the whole cycle |
| Old saves (4b) | today's stop node is the out stop point; the back one is `way_station`'s pick where it had one, else the same station's other way | old manual lines already fold on load (`fold_manual`) |

**A stop point is stored as (station id, track id)**, plus an index along the track in the rare station with two
stop points on one track. So two platforms on one track (the Spanish solution) make one stop point, and moving or
resizing a platform changes no id a line holds. A platform placed elsewhere changes no line.

Removing a platform changes no stop while another platform still serves that track there. When a stop point goes (its
last platform or its track removed): in simple stops that stop re-picks by smart allocation; in exact stops each
visit on it moves to the station's other stop point running the same way on that run, when there is exactly one.
Otherwise the edit is refused naming the line: "{line} stops at this platform." A waypoint on a removed track is
dropped, and the removal question lists it: "{line} loses a waypoint."

## Pieces and hours

| Piece | Stage | Hours |
|---|---|---|
| Smart allocation in the sim, both stop points stored, re-pick on removal | 4a | 5-7 |
| Old stops converted to stop points | 4b | 2-3 |
| Demand reading the back stop points; the track chip per way | 4c | 4-6 |
| Build crossovers offer on stop clicks and the chip | 8 | 2-3 (with part 3's crossover work) |
| Exact switch, cycle storage, both switch directions, the list | 11a | 5-7 |
| Circles per stop point, tint, numbers, merging, layout, exact clicks, waypoints with touch | 11b | 9-12 |

## New and changed player text (drafts, none final)

| Where | Old -> new |
|---|---|
| Stop list chip | new: way rows "Eastbound" (and the other ways); choice hovers "Stop on the outer tracks", "Stop on the inner tracks", "Track {n} from the north", "Stop at {station}" |
| Unreachable track | new: "{line} can't reach that track without a crossover." with "Build crossovers" and "Cancel"; then "Build {n} crossovers for {price}" |
| Stops heading toggle | new: "Exact stops"; hover "Choose each stop's track and the order trains stop in" |
| Circle hover | new: "{way} track, level {level}" ("Eastbound track, level −1") |
| Switch notice | new: "Back to simple stops. {n} waypoints were dropped." |
| Merged circle tag | new: "Click to zoom in and pick a track" |
| Waypoint tags | new: "Shift click to pass through here", "Click to remove the waypoint" |
| Touch strip | new button: "Waypoint" |
| List rows | new: "via {station}", "via track" |
| Removal | new: "{line} stops at this platform.", "{line} loses a waypoint." |
| Reused as they are | "Click to add a stop", "Click to remove the stop", "Can't loop back here", "No track joins {a} and {b}.", "No track joins it to this line" |

## Still open for her

1. When the platform beside is at another station (her split legs), the stop list shows one row with the clicked
   station's name and the chip, or a row per station? (one row with the chip: the line stops "at" one place)
