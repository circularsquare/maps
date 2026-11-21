# T-3444 design, part 4: stop assignment, simple and exact

Agent tdtools, 2026-11-21. Design draft for discussion with Anita; nothing is built. Words as in `01-model.md`.
Stage numbers are `10-build-plan.md`'s. Today's stop editing: `docs/ui-building.md` "Making and editing lines" and
"Clicks while drawing a route or editing stops"; the stop list: `docs/ui-inspectors.md` "Stops". Her words:
"we do definitely need to preserve simpel stop assignment for simple 2track lines, for casual player ease", and the
numbered-circle paragraph in `../T-3444.md`, which is the spec for exact stops.

## Two ways to set a line's stops

- **Simple stops**, every line's default: stations, as today.
- **Exact stops**, a switch per line beside normal stop editing (T-3079): stop points in running order, out and back,
  with waypoints. Not a mode of the game; one line at a time.

Which train uses which track between stops, and where trains turn, is part 5's (runs assigned per run). Per train
waits for exact timetables (T-3346). This part covers only what is stored and how the player picks it.

## Simple stops

### What stays

All of today's editing: one round marker per station, click to add or remove, Shift click for a one way stop
(T-3711), right click for loops, the tiers and their hovers, the gap flash, the "+ 2 stations passed" row, stop
order set manually by dragging rows (T-2895), Sort stops. On a plain two-track line the player sees nothing new at all.

### The one rule (01-model)

> A run stops at a station on the stop point of that station its path reaches going its way. If its path can reach
> more than one, it takes the one on the track it is already on (no crossing move); if still more than one, the one
> on the track the player clicked. A run that can reach no stop point of the station going its way passes it.

**What "the track the player clicked" means.** Markers are one per station, and a quad station is now one station,
so a click does not land on a track. Proposed: at the click the stop gets a **preferred stop point**, the one the
line reaches from the stop before with the fewest crossing moves; on a tie, the one whose track is nearest the
pointer at the click. The chip (below) changes it. 01-model's line record now says the same.

### The track chip in the stop list (stage 4; AA/BB choices in stage 6)

Shown only on a stop where a run could use more than one platform track of the station going its way: quad and
bigger stations, AA/BB, a station joined from offset platforms. Never on ordinary double, never on single.

- **Looks**: after the station's name, a small picture of the station's tracks (as the station panel's, 14 px tall,
  one stroke per track), the tracks this line stops on in its colour, the rest grey.
- **Two choices** (quad side by side, AA/BB): a click switches to the other at once. The hover names what the click
  does: "Stop on the outer tracks", "Stop on the middle tracks", "Stop on the north pair", "Stop on the south pair"
  (west and east for north-south track, the stop list's north-up rule).
- **More choices** (6 tracks and up): a click opens a short list of the choices, each its own small picture, hover
  "Tracks 3 and 4 from the north". Counting from the north is the station's numbering, never a bundle slot.
- **The edit**: one edit, one undo. The preferred stop point moves, and the line's path is found again through it.
  Refused when no route gets there: "{line} can't get onto those tracks from {station}."

### Lines on quad: local or express

A line drawn along one pair stays on it: the rule takes the stop point on the track it is already on. So an express
added on the inner pair stops on the inner pair, a local on the outer. A line that should change pairs at one station
gets it from that stop's chip. The picks are per line and per stop (settled); every train of the line follows them.

### Per-direction stops on AA/BB (stage 6)

The chip picks a **pair**: outer or middle. Each run then stops on its own way's track of that pair: the local on
slot 0 one way and slot 3 the other, the express on slots 1 and 2. This needs the bundle's stored pairs (part 3). 01's
partner rule, "the nearest track running the other way", does not give it: at offsets -6, -2, +2, +6 with ways A A B B,
both A tracks have slot 2 as their nearest B, so a local's way back would land on the express track. A mix (local one
way, express the other) is exact stops' job.

Her split legs 150 m apart and T-3605's lone one-way stations work as 01-model says: one station when placed in one
click or joined, two stations otherwise; nothing in the editing changes.

## Exact stops (stage 11)

### Switching a line

A toggle in the Stops heading, beside related lines: **Exact stops**. Hover: "Choose each stop's track and the
order trains stop in". Both ways are one edit and one undo.

| Switch | What happens |
|---|---|
| Simple -> exact | the cycle is the line as it runs now: the out run's stop points in order, then the back run's. Paths, tracks and timetable stay the same. Nothing to refuse |
| Exact -> simple | always works. If the cycle reads the same both ways station by station, it folds to an open line; each stop keeps the out run's stop point as its preferred one, and the back run's come from the rule. Otherwise it stays a line with those stops as a loop or turned line, as the 2026-11-10 rule. Waypoints are dropped. The notice says what changed: "Back to simple stops. 2 stops changed track and 1 waypoint was dropped." |

Against DECISIONS 2026-11-10 (T-3079, T-3089, T-3109):

| Old rule | Here |
|---|---|
| Switching Exact edits off always works; a list that reads the same both ways folds, any other stays a normal line with those stops (line 1772) | kept, with stations compared rather than stop points, and the notice |
| The route search turns back at a listed stop at no cost (1772) | kept for exact lines, at a stop point or a bare track end |
| A plain numbered list in running order, no related-lines drawing (1772) | kept |
| Cross, dot click and map number are one edit, one visit; a stop with two visits loses the first (1775) | kept: a chosen circle is one visit, so this comes for free |
| An empty circle on the right of each pass over a station with no stop (1775) | replaced: a circle per stop point, which already is one pass's track |
| Numbers from zoom 12.5 (1775) | kept, but circles that do not fit merge first (below) |
| Exact edits and manual mode gone (1783) | the switch returns as "Exact stops"; saves' old manual lines already folded |

### What the map shows while an exact line's stops are edited

- **A circle per stop point** the line could use, with today's tiers (sizes 100%, 80%, 65%, a dashed ring where trains
  would reverse, greyer where unreachable). A stop point here is one per station and track (see "What is stored"),
  so a track with platforms on both sides (the Spanish solution) gives one circle, not two.
- **Hover** on a circle: the station's name, then its track ("Eastbound track, level −1"), then what a click does;
  the track lights on the map as hovered things do (`hovered.ts`). A circle pushed along its platform can't show by
  its place which of the tracks it is on, so the hover and the light say it.
- **Not chosen**: white disc, grey ring, the whole circle tinted by the track's level (the level colours of the track
  bar, mixed 25% into the white).
- **Chosen**: the same size and place; the middle in the line's colour with the visit's number (white or dark,
  whichever reads, `legibleOn`); the outer white ring tinted by level (30%), the dark rim outside it. Two visits show
  both numbers, "3, 9". A small chevron on the way the visit runs (T-3139's chevron).
- **Waypoints**: a diamond in the line's colour at 60% of a circle's size, no number.
- Text says "level", never "elevation".

### Clicks

| Gesture | On | Does |
|---|---|---|
| Click | a circle not chosen | adds a visit. On a pass of the route over it: after the stop before that pass, path unchanged. Elsewhere: where it costs least in the cycle, as `best_slot` does today |
| Click | a chosen circle | removes that visit; with two visits, the first (1775's rule) |
| Click | a waypoint | removes it |
| Click | a merged circle, none chosen | adds the station as simple stops would: a visit on each pass of the route over it, each on the one stop point its way suits (both ways on a plain double). Where a pass could take two (quad) it zooms in until they fit instead (the camera jumps, no flight). Tag: "Click to zoom in and pick a track" |
| Click | a merged circle with chosen visits | removes the first of them (1775's rule); zoom in to pick another |
| Shift click | a circle | adds a waypoint there: trains pass through without stopping (T-3410's "pass through a one-way station") |
| Shift click | track | adds a waypoint on that track |
| Right click | as today | loops only (close, open, turn); "Can't loop back here" elsewhere |
| Drag a row | the list | reorders the cycle (T-2895) |
| Touch | | the strip's Loop button stays; a **Waypoint** button beside it makes the next tap a Shift click |

Shift click means a one way stop in simple stops; in exact stops every visit is already one way, so Shift is free
for waypoints, as it is for ferry waypoints on water (T-3809). Waypoints are free: no dwell, no riders, no price.
A run may turn back at a waypoint only where it could turn at a stop (a stop point or a bare track end).

### The list

Visits in running order, numbered, the closing row back to the first. Each row: number, station name, the track chip
(click: the station's other stop points on the same run's way), time and boardings as today. Waypoint rows are
faint: "via {nearest station}", or "via track" with none within 1 km. No related lines.

### Laying out the circles

Marker sizes are today's (`stopMarkers.ts` SIZE): 9 px at zoom 9, 14 at 12, 20 at 14, 24 from 16. Gap 2 px.

1. **Anchor** each circle at its stop point on screen. Its **axis** is its track's direction there. It may slide
   along the axis only, its centre staying within its platform's length (so it overhangs the platform by half a
   circle at most), and never across.
2. **Clusters**: circles whose anchors lie within one diameter plus the gap of each other, found on a grid.
3. A cluster whose axes agree within 15 degrees (one bundle, or stacked bundles on one plan line) lays out in **one
   row** along the axis, centred on its anchors' middle, ordered by level (lowest first) then from the north side,
   spaced one diameter plus the gap.
4. A row that does not fit within its circles' slide limits **merges** into one circle at the anchors' middle,
   drawn as a station marker with a pip on its ring per stop point; chosen visits show their numbers on it.
5. **Between clusters** (a complex's stations, crossing lines): up to 8 passes, each pushing overlapping circles apart
   along their own axes by half the overlap, within their limits, in a fixed order by stop point id. What still
   overlaps is drawn lower level first, so the tint tells them apart (her "so that overlapping circles are easy to
   distinguish").
6. Positions are kept in ground metres for the zoom, so a pan moves nothing. They are worked out again when the zoom
   changes by 0.1 or the network or the line changes. Picking reads the moved centres, as `pick` does today.

How many circles a 200 m platform (today's default, 10 cars of 20 m) holds in one row, at New York's latitude, 3.6 m
a pixel at zoom 14 (`stationGeometry.ts`) and half that each zoom in. A row of n fits when n x (d + 2) is at most the
platform's length on screen plus d + 2:

| Zoom | Platform on screen | Circle | Circles that fit | So |
|---|---|---|---|---|
| 12 | 14 px | 14 px | 1 | every station merged |
| 13 | 28 px | 17 px | 2 | double splits |
| 14 | 55 px | 20 px | 3 | three tracks |
| 15 | 110 px | 22 px | 5 | quad splits |
| 16 | 220 px | 24 px | 9 | 6 and 8 tracks |
| 17 | 440 px | 24 px | 17 | Grand Central's 16 |

An 80 m tram stop holds 2 from zoom 15 and 4 from 16. The circles in view are a few hundred at most; the layout is a
grid pass and a sort, well under a millisecond.

### Lines on quad and AA/BB in exact stops

Each visit is a stop point, so the player clicks the circle on the track she wants: local or express, and on AA/BB a
different pair each way if she likes (local out, express back). The route search finds the crossing moves between;
a visit no route reaches is refused as today ("No track joins {a} and {b}.", reused).

## What is stored

| Field | Simple stops | Exact stops |
|---|---|---|
| Mode | simple | exact |
| Stops | station id plus preferred stop point, one way flag (T-3711), hand order (T-2895) | visits in a cycle: a stop point, or a waypoint (track id, chainage) |
| Path | track spans first to last stop, plus the stored return path where one is routed (01-model) | track spans for the whole cycle |
| Old saves | today's stop nodes map to their stations; the node's track is the preferred stop point | old manual lines already fold on load (`fold_manual`) |

**A stop point is stored as (station id, track id)**, plus an index along the track in the rare station with two
stop points on one track (offset platforms joined on one track). So two platforms on one track (the Spanish solution)
make one stop point, and moving or resizing a platform changes no id a line holds (01-model says the same). Ids are authored ones, so a platform placed elsewhere changes no line.

Removing a platform changes no stop while another platform still serves that track there. When a stop point goes
(its last platform or its track removed), each visit on it moves to the station's other stop point running the same
way on that run, when there is exactly one; otherwise the edit is refused naming the line: "{line} stops at this
platform." A waypoint on a removed track is dropped, and the removal question lists it: "{line} loses a waypoint."

## Pieces and hours

| Piece | Stage | Hours |
|---|---|---|
| Preferred stop point, the rule in the sim, old saves | 4 | 4-6 (with 01's simple-stops work) |
| Track chip, two choices and the list of more | 4, AA/BB pairs in 6 | 3-5 |
| Exact switch, cycle storage, both switch directions, the notice | 11 | 4-6 |
| Circles per stop point, tint, numbers, merging, layout | 11 | 6-8 |
| Exact clicks, list and chip, waypoints with touch | 11 | 5-7 |

## New and changed player text (drafts, none final)

| Where | Old -> new |
|---|---|
| Stop list chip hovers | new: "Stop on the outer tracks", "Stop on the middle tracks", "Stop on the north pair" (south, west, east), "Tracks {a} and {b} from the north" |
| Chip refusal | new: "{line} can't get onto those tracks from {station}." |
| Stops heading toggle | new: "Exact stops"; hover "Choose each stop's track and the order trains stop in" |
| Circle hover | new: "{way} track, level {level}" ("Eastbound track, level −1") |
| Switch notice | new: "Back to simple stops. {n} stops changed track and {m} waypoints were dropped." (each part only when it happened) |
| Merged circle tag | new: "Click to zoom in and pick a track" |
| Waypoint tags | new: "Shift click to pass through here", "Click to remove the waypoint" |
| Touch strip | new button: "Waypoint" |
| List rows | new: "via {station}", "via track" |
| Removal | new: "{line} stops at this platform.", "{line} loses a waypoint." |
| Reused as they are | "Click to add a stop", "Click to remove the stop", "Can't loop back here", "No track joins {a} and {b}." |

## Her calls

1. Exact stops is a switch per line in the Stops heading; going to exact keeps the line running as it does, and
   going back always works, with a notice when tracks changed or waypoints were dropped? (yes)
2. Waypoints only on exact lines for now (Shift click, a Waypoint button by touch), or on simple lines too? (exact
   only; simple keeps Shift click for one way stops)
3. Where a station's circles don't fit zoomed out, one merged circle: a click adds by the rule when only one track
   suits, else zooms in? (yes)
4. In simple stops a new stop goes on the tracks the line is already on, and the chip moves it? (yes)
5. On AA/BB in simple stops the chip picks a pair for both directions, and a mix (local one way, express the other)
   needs exact stops? (yes)
