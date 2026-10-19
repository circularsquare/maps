[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · **inspectors** ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the Lines and Stations tabs and the inspectors

The inspector, under the dock's tabs, shows whatever is selected: a line, a station complex (or
one station), a stretch of track, a junction, a train, bubbles, a city or zone, or a trip. Nothing
selected leaves it blank, with no text. An empty click on the map or Escape clears the selection.
One component per kind in `ui/inspector/` (`others.tsx` holds station, train, track, junction and
cell or zone). The track and junction inspectors are in [ui-building.md](ui-building.md) and
[track-and-service.md](track-and-service.md) 6.2; a bubble selection's and a city's (their
inter-region trips by train, car and plane) in [ui-demand-views.md](ui-demand-views.md).

## The past 24 hours, not a forecast

[T-293, T-446] The line and station panels and the Lines and Stations tabs count the past 24
hours, not a whole-day forecast, so nothing resets at midnight: riders, boardings, fares, running
cost, trains that left full, riders left behind. Labels say "Past 24 h" or "24 h";
hovers say "in the past 24 hours". They are counted by the clock worker from the boarding
account's departures (`TrackApi.past_day`: the rest of yesterday from the time of day now, on
yesterday's kind of day, then today so far on today's; a stop's boardings in a stretch are the
riders arrived by its last departure in it less those still waiting, less the same at the
departure before it). Fares are each line's hourly fare, the hour now by the share paid as the
cash is [T-284] and the hour a day ago by the rest; running cost is each hour's trains, the hours
at either end in part. Asked four times a wall second
while a panel shows them and the clock runs (`game/today.ts`); only the components showing them
re-render. The map's demand views stay demand's whole-day picture.

**Riders by purpose** [T-165, `ui/inspector/PurposeSplit.tsx`]: under a line's riders and a
station's (or complex's) boardings in the past 24 hours, a thin stacked bar in the purpose colours, in the
purpose filter's order, with no key (T-669); hovering a segment gives the purpose and its share
("Work: 48% of riders", "<1%" for a sliver). Purposes with none here have no segment. Under the
purpose filter the purpose drawn is ringed in the bar.

## Lines tab and the line inspector

- **The Lines tab** has New line above its list (actions go above lists of any length). The list
  sorts by clicking a column header [T-310]: Name, City (below), Riders (past 24 hours), Trains and Near. A click sorts by that
  column (names A to Z, figures biggest first, Near nearest first), a second reverses it, the sorted
  column has a small arrow, and each header's hover says what it sorts by. Names sort naturally, ties
  by name. Lines default to by name. The sort is kept while the page is open and re-runs each game hour.
  A double click on a row fits the map to the line's stops (a jump, as search does) after the
  single click selected it [T-569, `ui/locate.ts`]; the line inspector's locate button (cross
  hairs, hover "Show on the map", beside the pencil) does the same.
- **Trains to add** [T-472, T-491]: a row's train count (the trains its level needs now, the bare
  number under the "Trains" header [T-532]) is coloured by the crowding score of the line's period now, otherwise muted. The score is the
  average rider's extra wait for a train with room (rider-minutes spent on the platform after a
  full train left them, each waiting the gap to the next departure, over riders arriving in the
  period) divided by the square root of their average ride on the line (rider-minutes aboard over
  riders, at least 5 min): `crowding` in `game/today.ts`. Grades (`CROWD_GRADES`): yellow from
  0.1, orange from 0.3, red from 0.6, so 2 min on a 10 min ride is red and so is 4 min on a 40
  min ride; orange and red colour the count (`crowdClass`). Hover "Morning peak: riders wait 1.8
  min on average for a train with room", then "4/12 trains running now are full on their
  trip" (trains from leaving the first stop until they could leave again from the far end,
  `Boarding::under_way`). The figures end each `past_day` block (`Boarding::crowding`).
- **Scope and Near** [T-332; `game/listScope.ts`, `ScopeToggle` in `ui/tabs.tsx`]: beside New line
  (and above the Stations list) a toggle between the viewed place (a pin and its name, hover "Only
  <name>") and the whole world (a globe); one setting for both tabs, kept while the page is open, the
  place the default. The place is `viewedZone` (T-318); a station is in it when `placeAt` says so, a
  line when any stop is. A line with no stops and the selected row stay listed. Over unmodelled space
  the pin is disabled and the world is shown. The Near header is always there (beside "Name" until it
  is used); sorted by it, each row ends with its distance from the centre of the screen (a line's is
  its nearest stop's, hover "1.4 km from the centre of the screen"), rows without one last, and the
  column's width is given back to the names under any other sort. The centre is taken at each move end (`viewCentre`, set in
  `map/zoneFocus.ts`), so the order follows the view without moving while it is dragged.
- **City column** [T-670]: with the whole world listed and the lines' first stops in two or more cities, a
  City column follows Name (the first stop's city, named as the Cities list names it, `stationZone`;
  hover "The first stop is in Toronto"). It sorts like the others and takes room from the Riders and
  Trains columns.
- **New line** [T-109 to T-112] selects an empty zero-train line, enters stop editing
  ([ui-building.md](ui-building.md)) and focuses its unique `Line N` name, text selected. N follows
  saved creation order, copies included, skipping names in use. The N key does it from any tab
  [T-670]; the same on a selected track no line runs on (its inspector's "Lines on it" heading
  carries a plus button, hover "New line, starting at the stations on this track (N)") starts the
  line with the stations at the track's two ends, else empty.
- **The line editor** [T-670, T-672]: a double click on the badge, a click on it, a double click on the
  name or the pencil open one editor under the title: the name box (focused), a Symbol box and the
  colour picker (`ui/ColourPicker.tsx`, after languagedots' swatch editor: a saturation and brightness
  area, a hue strip, a hex box with the colour beside it, the palette swatches). Leaving the name box
  keeps the name and leaves the editor open; Enter in it, the check button (hover "Done") that
  replaces the pencil, a click anywhere outside the title and editor, or Esc closes it (Esc in the name box
  drops the typed name). A symbol with several characters widens the rounded badge; a blank symbol
  restores name-derived initials. Dragging in the picker changes the badge and the stop dots in the
  panel at once; the map's line follows by one colour edit every 0.2 s at most (longer on a slow
  machine: 6 times the last edit's round trip), and the final colour goes as one edit on release, a
  swatch click, an arrow key step or a typed hex.
- **Actions** at the top, under the title and the planned or broken notice: Edit stops (hover and key L), Copy line (duplicates route and settings, then opens
  naming and editing), and Delete line (asked first). While stops are being edited the Stops list (with a remove
  button on each stop) follows them there, above Schedule; otherwise it sits after the day's figures.
  Incomplete lines stay planned and run no trains.
- **Sections**, in order: Schedule (trains for each demand level, with the resulting headway;
  [track-and-service.md](track-and-service.md) 6.3; columns Demand, Trains and Headway, every
  cell one line, the Demand column cut with an ellipsis when short of room [T-692]; a Trains / Headway switch in the heading
  chooses what the player sets [T-625]: with Headway the headway column holds a stepper (30 s steps
  under 5 minutes, 1 minute under 15, 5 above; none, or 1 to 120 minutes; a click on the number types
  it in minutes) and the trains column the count it gives; each button's hover names the train
  count it leads to and, on a running line, the cars it buys at the car price, takes from the
  depot, or frees [T-631]; a line whose track holds it just over the headway set shows the headway
  it runs at, muted, left of the stepper, with a hover naming what limits it as the capacity alerts
  name it ("Terminus at West takes one train every 4m00s at most, so 8 trains run every 4m07s",
  from the tail of `TrackApi.line_held`) [T-679]; a headway shorter than that limit, or by hand more
  trains than fit, is red, with the same hover; an edit refused for cash shakes its control for
  300 ms and paints the level's train count red until the cash covers it or another edit to the
  line goes through (`refusedForCash` in `game/world.ts`) [T-625]; hovering High, Medium or Low gives that
  level's times of day in the line's local time, from the period table in `game/periods.ts`:
  "High demand: 06:00 to 10:00, 16:00 to 20:00"; weekends use the same hours) [T-406]; on a
  running line one dot before each level's name grades the level on today's kind of day by the
  average wait for a train with room (the level's periods averaged by riders; High is its two
  peaks), its colour running smoothly with that wait in minutes: green to 2, yellow at 3, orange
  at 10, red at 15 and a deep red from 20 (`crowdColour` in `game/today.ts`, fed `avg.wait` in
  `CrowdDots`; `Boarding::line_periods`) [T-692] [T-673] [T-477, T-491, T-515]; hover "Riders wait 4.0 min on
  average for a train with room", then "12/96 trains are full" (all the level's periods),
  and a last line naming the worst period and its wait when that period's grade is worse than the
  level's; the trains steppers take Shift-click for 5 and Ctrl-click for 20 (their hover says so)
  [T-516]; when full track holds trains back, a red box under the table names it ("Platform at
  Akabane is full", the station name a link that moves the camera there [T-669]) with the trains
  held per level;
  Today for a running line:
  riders by purpose,
  one Profit row (fares, a minus sign, then the running cost over the 24 hours in red; the hover
  gives fares, running cost and the net), how many trains left full as "x/y" of the trips that
  reached a stop in the 24 hours (the fourth number from the end of each `past_day` block) [T-481]
  and the riders left on the platform as "x/y", x those a full train left behind and y those who
  wanted a train (boarded plus left behind) [T-230, T-239, T-293, T-669]; the block's heading is
  "Past 24 hours" (no Yesterday view since T-669), with a rule above it and none between its rows
  (the only other rule is above Stops) [T-692]; the stops (above Schedule while being edited [T-670]) with their time from
  the start and boardings in the past 24 hours, each transfer stop (any station of its complex
  served by another line) showing the two other lines with most boardings there in the past
  24 hours as bullets and, when there are more, a "+" whose hover names the rest [T-549]; **flat junctions and
  flat crossings on the route** [T-629, `TrackApi.line_marks`] as small marks on the diagram's bar between the
  two stops they lie between, spread evenly in the gap (the diagram is not to scale): a white diamond for a
  junction with no flyover and no station, a cross for a flat crossing, each with a hover naming it and
  the lines that use or cross it ("Flat crossing near Harbour", "Crosses Cross"); the round trip split into running, dwell, turning and waiting,
  naming where it waits most [T-031]; and last the average speed, both ways over the round trip
  less the turning at the ends, so stops and waiting at busy track count (hover gives the
  distance and time) [T-549].
- A line's riders count every line actually boarded, transfers included; inter-region trips show
  only in its purpose split [T-182].

## Trains

- **The train inspector's "On board"** [T-083, T-164, T-230]: this train's own riders on the stretch
  it is running, from the boarding account ([track-and-service.md](track-and-service.md) 6.6),
  against seats and crush: a bar with a tick at the seats and a state ("Empty", "Seats free",
  "Standing", "Full"), beside the typical train this hour (the mean of its line's trains in the same
  direction drawn this hour, on the same stretch). A train is never over full: whoever does not fit
  waits on the platform. A line with no figures yet says "Calculating" [T-324].
- **A selected train is followed** [T-245, `map/trainFollow.ts`]: when its trip ends and the track
  model hands it to the next departure there (`Trip::next`), the selection moves to that trip as it
  leaves, so inspector and highlight carry on. A train leaving service stays selected where it
  finished ("This train finished its trip and left service").

## Stations tab and the station inspector

- **The Stations tab** has, above its list, the "Show station ridership" choice (None, Origins, Transfers,
  Total, each described in its hover) and, while circles are drawn, "One circle per station
  complex" [T-213, T-482; [ui-demand-views.md](ui-demand-views.md)]. The list sorts like the Lines
  tab's [T-310], by Name, 24 h (boardings in the past 24 hours) or Near, boardings by default. The scope
  toggle and distance column are the Lines tab's (above); the footer reads "N of M stations" when
  only the viewed place is listed. A row is a station complex [T-530]: the name of its member with
  the most lines, every line stopping at any member, boardings summed over the members, and a
  click selects the complex as a click on the map does and a double click also moves the map to
  that member, to zoom 14 or closer so its platform boxes show [T-569]; a station on its own is its own row, and
  M counts complexes. The boardings figure is a fixed column, right-aligned. The name comes first
  [T-530, Anita]: every bullet shows when the whole name fits beside them; otherwise as many as fit
  beside the whole name, never fewer than 3, and only then is the name cut with "...". Bullets left
  out sit behind a "+" whose hover lists their lines [T-521, T-530]. The bullets sit against the
  boardings column.
- **Line bullets** [T-521]: one component (`Chip` in `ui/widgets.tsx`), drawn the same everywhere.
  Hovering one gives the line's name and a click selects the line, except in the selected line's
  own panel (its title bullet opens the colour and symbol editor) and on a search row that is the
  line itself. The map's tags draw the same bullet as plain DOM (`lineBullet`). In both, the
  symbol is SVG text placed from its measured ink box (`ui/bullet.ts`, T-705), so the ink, not the
  font's line box, is centred in the circle, at any font and UI scale (within 1 px).
- **A click on a station selects its whole complex** [T-249]: complex view, whatever was selected
  before (there is no Station / Complex switch). A station on its own is a complex of one. Complex
  rules: [ui-map.md](ui-map.md) "Station complexes".
- **Complex view**, under the name and "Complex of 16 stations":
  - **Its lines** [T-246, T-250]: one row per line name, branches and patterns summed, with chip,
    name, riders waiting now (red when one of its platforms is full or its last train left riders
    behind) and boardings in the past 24 hours, busiest by the forecast first so the order holds through the day.
    The row's hover gives line, station and where its trains go; each number has its own hover. A
    click opens that line's busiest station there alone in station view (`openAlone`,
    `singleStation` in `game/complexView.ts`): its own inspector and outline, its
    subtitle ending with "Complex of 16 stations" as a link back. Any new selection, the same
    station included, shows the complex again.
  - **The complex map** (below), then the complex's boardings in the past 24 hours with their purpose split.
  - **Closest stations** [T-312, T-323; `ui/inspector/ClosestStations.tsx`, `game/walks.ts`]: the
    three other stations with the shortest walk, built or blueprint (blueprint muted and marked),
    each with the chips of its lines (six at most, then "+n") and walking minutes ("7 min", the
    metres on hover). A click selects it (and moves the map if it is off screen). Its own complex
    and stations of its own name are left out, and each name is listed once. The walk is the
    demand model's routed walk ([demand-mode.md](demand-mode.md) 4.3) x the 1.3 detour at the model's 6 km/h; a station
    water cuts off shows "no walk", only when fewer than three can be walked to within 10 km.
  - **Linked by you**, then the buttons **Trip from here** [T-221] (the trip tool with the station
    as the start) and **Link to another station** (below).
  - **Commuters**: "Live nearby" and "Work nearby" (commuters a day who live, or work, within its
    reach and ride from it); the larger is picked first, a click picks the other; then the top
    eight far-end places ("Where they work" or "Where they live"). Not drawn on the map [T-519,
    [ui-demand-views.md](ui-demand-views.md)]. Commuters and inter-region trips cover the whole complex, each of the demand
    model's stations counted once.
  - **How they arrive** [T-374; `AccessSplit` in `ui/inspector/StationRiders.tsx`]: under the two
    rows, a bar and key in the purpose split's style for the commuters who live nearby, walked
    (walk icon, the walk colour) against drove (car icon, the drive colour), shares in the key
    and counts in the hovers ("Drove, took a bus or cycled"). They are the model's commutes
    at the home end (`walked` and `rode` of `StationRiders`, from `DemandApi.station_riders_day`);
    everyday trips and the regional layer are not split. Not part of the past 24 hours' purpose split, which
    counts boardings.
  - **Inter-region trips**: rail riders starting and ending there, by place [T-118, T-120].
- **Station view** (one station opened alone) has its elevation under the name ("Elevation +1"), then in place of the
  complex's lines and map the station's own lines, platform settings, Construct and Remove, and
  **Waiting now** [T-164, T-238]: each line and direction ("To Norwood-205 St") with the riders on the
  platform now in red; under it, in red, how many full trains left riders behind this hour and the
  longest wait, or else the riders arriving this hour and the longest wait. The counts drop the
  moment a train leaves with whoever fitted (a timer set for the station's next departure) and grow
  with arrivals each game minute. The rest (closest stations, links, commuters, inter-region
  trips) is as in complex view, for the station alone.
- **Renaming** [T-534, `ui/inspector/NameEdit.tsx`]: a station's and a line's name both edit in
  place from a double click on the name or the pencil beside it (hover "Rename, or double click the
  name"), its text selected; Enter or leaving the box keeps it, Escape drops it, an empty name is
  ignored. A station's title also has the locate button [T-569] (cross hairs, hover "Show on the map",
  beside the pencil: the same move as a double click on its row) and the cycle button (two arrows, hover "Next suggested name"):
  each click renames it to the next of its suggestions, the cross street, the neighbourhood and the
  city or town, wrapping round, worked out then from where it stands (map/stationNames.ts,
  [track-and-service.md](track-and-service.md) "Placing stations"). The cross street needs the
  station on screen at zoom 13 or closer; with nothing else to offer it says "No other name is
  suggested here."

## The complex map

[T-147, `ui/inspector/ComplexMap.tsx`; T-176, T-246, T-250]

- The inspector's width, 120-240 px tall, north up, positions and headings true at one scale (at
  most 1.2 px a metre; a round scale bar in a corner).
- Each platform is drawn with its lines through it in their colours, each as wide as the riders on
  its trains there a day (each direction the busier of arriving and leaving loads, every period
  summed), 1.5 to 6 px by square root against the complex's busiest. Platforms on top of each other
  (parallel within 15 degrees, centre lines within 12 m) are one bundle, lines side by side in the
  order the platforms lie. Bundles have a thin dark outline, dashed for blueprint; the selected
  station's a heavier one. A line's hover gives station, line, boardings, riders on its trains
  there, and how many change to or from other lines there with its three biggest partners; a click
  opens that station alone.
- When the complex has more than one name, each name (the part before a slash) has one label beside
  its platforms wherever it is clear, else none.
- The player's links are dashed lines between the two stations, their hover giving the distance.
- **Now** [T-246, `game/complexLive.ts`]: after each station's lines its bundle has a 5 px platform
  strip, white between two lines of the platform blue, filling like the map's platform boxes (each
  direction's half from the middle out by its riders' share of the room, grey, or red once full); its hover
  gives the riders waiting (by line name when more than one stops there) and the room. Trains on their line's band in the darker train shade: one at the
  platform along it, one pulling in (stopping within 60 game seconds) paler, coming in from its end,
  further along the sooner it stops; hover gives line, destination and seconds. Read from the
  renderer's hour of the boarding account; worked out every 10 game seconds, re-rendered only when
  something shown changed.
- **Transfers** [T-176, `complexTransfers(members)`, [demand-views.md](demand-views.md) "Transfers"]: asked of the demand
  workers when the map shows and at each published round, the last answer kept meanwhile. Both
  directions of a line pair added. Changes between lines in different bundles are grey curves (ink
  at 55% over a thin halo of the land colour), across the gap where the two lines' bands come
  nearest, or middle to middle when they cross or come within 16 px: the busiest 8, none under 2%
  of the complex's changes, 1.5 to 8 px by square root. Each curve bows gently to one side; two
  middles at one spot get a small loop. Hover gives the riders per day and each direction. Changes
  within one bundle and walks to stations outside the complex are not drawn. Why line to line:
  notes/T-147.md.
- Two short notes under the map say what the widths and the curves are.

## Linking stations

[T-197, `game/links.ts`] The station inspector ends with "Link to another station". After a click
on it the next station clicked is linked with the selected one (the pointer a crosshair, a hand
over a station); a click on anything else, Esc, "Stop linking", another selection or a tool stops
it. Refused with a notice, still picking: the same station, one already in this complex, or one
more than 1 km away centre to centre. "Linked by you", above the commuters, lists each link
touching the complex with its two stations, their chips and Unlink. Only the player's links can be
undone: stations the distance rule groups cannot be split. Linking is not in the Ctrl+Z history
(a setting, like fares); removing a station drops its links. Blueprint stations can be linked;
demand counts a link once both are constructed.

## The trip explorer

[T-195, T-208, T-221, T-586, T-628; `map/tripView.ts`, `ui/inspector/TripInspector.tsx`,
`game/trip.ts`; the model's side in [demand-views.md](demand-views.md) "The trip explorer"]

The trips between two places as the view it was opened from counts them, homes at the start to
destinations at the end, and how the demand model's people would make one of them.

- **Starting it**: the T key from any tab [T-444], or Explore a trip at the bottom of Settings, turns on the trip
  tool (a crosshair; a second press or Esc leaves it). Clicks alternate between start and end, a
  third starting a new trip; the hint at the bottom left says which is next.
- **Stations as ends** [T-221]: a click on a station (a hand cursor over one) takes the station's
  own cell as that end, and the inspector names it ("From a home by Bergen St to work by Grand
  Central-42 St"). Its own cell, because the model's subzones group cells by nearest station, so
  the trip mostly boards there. A station's **Trip from here** starts it from that station.
- **From a selection** [T-208]: with a bubble selection, a click on a far-end bubble opens the
  trip for that pair (hovers end "Click to explore this trip"), homes to destinations. The
  selection's end is its busiest cell under the purpose filter, the far end the clicked bubble's
  cell with most of the selection's trips: the representative trip's two ends. The inspector then
  has a Back button to that selection (alone, with no line saying where it came from).
- The trip is a selection: it stays after the tool is left, until Esc or an empty click, and while
  selected panning does not change the focused zone. It is asked again at each published round,
  and when the purpose filter or the day changes.
- **The headline follows the view** [T-586]: under the purpose filter and the day shown, the trips
  a day between the two places (a line "124 trips per day, 2.5 km apart"; its hover the
  exact count and, under All purposes, each purpose's), then a bar and rows of their shares by
  train, walking and driving. Opened from a far-end bubble the places are the selection and every
  cell of that bubble, the same sums as the bubble's hover (`tripMix` over the selection's far end,
  `flows`), so the two agree; from the trip tool, the start's cell as homes and the end's cell. The
  walk row shows when walking the whole way takes under 30 minutes [T-628], else walking is only
  the bar's segment. When the view counts no trips between them, the headline is the representative
  trip's chances in the period shown.
- **The representative trip**: everything below the headline is one trip read by the model: under
  All purposes the purpose with most of the trips between the two places (work when none), else
  the filter's (school and university: the kind with more trips between the two areas); a commute
  through `DemandApi.trip`, an everyday purpose through `EverydayApi.trip`. Its period is the one
  with most of that purpose's trips, until a "Train by period" row is clicked; a new pair of places
  starts from the busiest again. "By train" names the purpose and period at its right, its hover
  saying why those. The sub line names the filter's destination ("to shops and errands by Union
  Sq").
- **The inspector**: under "By train" the main route as rows (walk to the first
  station, each line ridden with its chip and the station it goes to, each change and walk, the
  walk from the last station; each row's minutes its wait and ride; a ride access [T-374] reads
  "Drive to" with its drive minutes, its hover adding the ride's fixed cost and the station's
  parking or drop-off; the hovers give real minutes only), the fare, **Time** (real minutes; its
  hover the sum, "Walk 6.3 + wait 2.1 + ride 10.1 = 19 min") and **Effective time** [T-699, T-549]
  (the minutes the model weighs against the other methods; its hover is only the equation, the
  first route's terms in the rows' order, each real amount times its weight, "Walk 6.3 × 1.47 +
  wait 2.1 + ride 10.1 × 1.1 + change 2 + fare $1.92 × 3 − choice of routes 11.8 = 16 min", two
  changes or more as "changes 2 × 2" [T-717]; the
  last term is what the model's pooled figure, over every station and route people use, is below
  this route's sum, so the equation always adds to the row, and one plain line under the equation
  says choice of routes is the time saved by having several ways to go; the terms are the worker's
  perceived minutes, `routeTerms` in `game/trip.ts`), then one sentence with the share of the
  trip's train users on the route and the same trains from other platforms or walks added ("65% of
  the train users on this trip take this route, the rest use different routes."), with up to four
  "Other routes" and their shares, then a "Smaller routes" row with what is left [T-709]; the
  sentence's and the row's hover say how many pairs of stations the users spread over and what the
  busiest pair takes (in New York the main route has 5 to 12% and the rest is thin, notes/T-709.md). "By car" and "By walking" read the same
  way. Car: **Drive** (real minutes; its hover the straight line, the road length and the average
  speed of this drive, `DemandApi.trip`'s road km over its minutes), **Parking** (minutes, with the
  price of parking and any congestion charge counted as minutes), **Car ownership** (dollars) and
  Effective time ("Drive 17.6 × 1.33 + parking 67.3 + ownership $1.99 × 3 = 97 min"). Walking: one
  Effective time row [T-704] (its hover "Walk 52.4 × 1.5 = 79 min" holds the real minutes of
  walking the whole way); over 10 km it says walks are not offered. "Train by period": a row a period with the representative
  trip's chance of the train, a click showing that period. Without a train it says why (no station in reach, walks too long, no
  service in the period). Every row and bar segment has a hover with its figure and what it is made
  of. The trip is the two places' own (each one's walk to its stations, T-720); where the map's
  bubbles count it differently, grouped with the homes and places near each end, the period rows'
  hovers add that count ("Counted with the homes and places near here, as the map's bubbles count
  it: train 62%, drive 30%, walk 8%") and the first walk's hover the group's walk when it is a
  minute or more apart.
- **On the map**: a veil of the land colour at 60%; the start an open ring and the end a filled
  disc; the main route over the veil in its lines' colours along their track (straight between
  stations only when the drawn network no longer matches the solve's), with a dark casing and the
  boarding and leaving stations bigger; walks dashed and straight; other routes thinner and faint
  under it. Hovering a ride gives line, stations, stops, minutes after the wait, crowding's effect
  and the route's share; a walk its distance and minutes; a station what the route does there.
