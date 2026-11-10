[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · **inspectors** ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the Lines and Stations tabs and the inspectors

The inspector, under the dock's tabs, shows whatever is selected: a line, a station complex (or
one station), a stretch of track, a junction, a train, bubbles, a city or zone, or a trip. Nothing
selected leaves it blank, with no text. An empty click on the map or Escape clears the selection.
The title row (the bullet or glyph, the name and its buttons) is pinned to the top of the panel's
scroll area, on the panel's background, so it stays in view when the panel is scrolled [T-1812].
One component per kind in `ui/inspector/` (`StationInspector.tsx`, `TrainInspector.tsx`, `TrackInspector.tsx` and `JunctionInspector.tsx` hold station, train, track and junction). The track and junction inspectors are in [ui-building.md](ui-building.md) and
[track-and-service.md](track-and-service.md) 6.2; a bubble selection's and a city's (their
inter-city trips by train, car and plane) in [ui-demand-views.md](ui-demand-views.md), the city
view's own rows below.

## The city view

[T-1438; `ui/inspector/CityInspector.tsx` with `CityDetails.tsx`] With a city selected, a pair of cities under 10 trips a day (`MIN_PAIR_TRIPS`, `map/intercityView.ts`) has no bubble or line and no row in Top destinations [T-2770]. A city or zone picked by its
inter-city bubble zoomed out, or by a click on its row in the Cities tab (a double click there moves
the camera, as in the Stations tab). The home and go-to icons sit on the title's line, at its right. The home
icon (a house, `HomeButton.tsx`, T-2395) makes the zone the game's home, where the camera starts whenever the
game is opened ([ui-panels.md](ui-panels.md#start-menu)); it is filled while the zone is home, and a click on
it then clears the home. Hover "Set as home" with "Start here whenever you open this game" under it, or when home
"Home" with "You start here whenever you open this game". One home per game, kept in the save (`home`). Under the
title, rows: **Train share** (the
zone's, as the top bar gives it; hover the top bar's split hover, [ui-panels.md](ui-panels.md#the-top-bar)), and for a zone with a built station
**Stations** (built ones, the Cities tab's count), **Lines** (lines with a stop in the zone) and
**Train trips** ("12.3M/day", the Cities tab's Trips figure for the day shown, hover the whole figure
"per weekday" and so on) [T-1892], then **Points** (the past 24 game hours, hover as the Cities tab's), **Profit** ("$389M a day", the Cities
tab's figure and hover, losses red) and **Goals reached** (each goal reached, the star's as a small
star before "10%", then "20%" and on; hover when it was reached; "None" before the first)
[T-1437]. Then what a bubble click showed before ([ui-demand-views.md](ui-demand-views.md)): **Trips
within New York**, with the goal ticks over its mode bar (a grey No route part where some trips have no way, T-2215), **To
other places** (train, car, and planes in the muted grey). Then **By trip length** [T-2365,
`TripGraph.tsx`], for a zone with its own split: both rows' trips together by straight-line length, a log axis
from 0.125 km to the farthest trips (at least 100 km), labelled at powers of ten, then fives, then twos where they
fit ("0.5", "1", "2", "5", "10", "20", "50", "100", "1,000 km"). Each mode is stacked as trips a day per decade of
length, so a colour's area is its trips: train at the bottom, then walking, driving, No route, planes. The zone's
own trips go by the demand kernel's bins (`trip_lengths_day`, notes/T-2365.md), its trips to other places by the
distance between the two places' first zones' centres (their car trips with driving, their rail with the train);
the trips of 1,000 km and more are drawn together over 1,000 to 2,000 km. Each bin's trips are spread evenly over
its width on the axis and blurred by a normal curve of 0.05 decades, reflected at the axis's ends, so the curves
are smooth and keep every trip. Fills at 80% (90% in the dark theme), each layer's top a 1.3 px line in its colour
behind a thin gap in the panel's background (only where the layer is over 0.6 px thick), faint vertical lines at
the labels, no y axis. Under the plot a 10 px strip of each length's mode shares, where the length has trips. Hover
(the plot or the strip): the length band under the pointer shaded, and its trips ("Trips of 10-15 km per day",
then each mode's share and count, No route and Plane only where the band has some, "All", and "To other places"
where some are), the bands breaking at 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500 and
1,000 km ("Trips under 0.5 km per day", "Trips over 1,000 km per day"). Then the top three destinations. While the
zone still loads (its bar on the map, T-421), To other places and the destinations wait, "Calculating demand" in
their place; its Trips within and its graph show as soon as it has figures of its own (a round of its cluster
during a load, T-2453), "Calculating demand" alone before that [T-2500]. Then, for
a zone with a built station: **By purpose**, the purpose bar of the
inspectors (`PurposeSplit`, no key) over the boardings at the zone's stations on every line;
**Top lines**, the five lines with most boardings at the zone's stations today
(bullet, name, count with hover; the header's right says "boardings today"); **Top stations**, the five complexes (or lone stations) in the
zone with most boardings today, named after the member with most lines. A click on a
line or station row opens its view.

## The all cities view

[T-2395; `ui/inspector/AllCitiesView.tsx`] Opened by a click on the Cities tab's first row, All cities: that
row's sums (`allCitiesTotals` in `ui/Cities.tsx`) laid out as the city view's rows, totals only. The title "All
cities", then the home icon and the go-to icon (hover "Go to all cities": every built station in view, all
stations if none is built, as a home of all cities opens). Home here means the game opens zoomed out on all
cities; its hover "Set as home" with "Start zoomed out on all cities whenever you open this game" under it, or
"Home" with "You start zoomed out on all cities whenever you open this game". Rows: **Train share** (of the
cities' trips, the row's bar; hover its split "in all cities"), **Cities**, **Stations**, **Lines** (lines with
a stop at a built station in one of the cities), **Train trips** ("12.3M/day"), **Points** and **Profit** (the
row's figures and hovers, losses red).

## The past 24 hours, shown as today

[T-293, T-446, T-1893] The line and station panels and the Lines and Stations tabs count the past 24
hours, not a whole-day forecast, so nothing resets at midnight: riders, boardings, trains that left
full, riders left behind (the profit is the exception, below). They follow the schedule as it is, so they move at
once when it changes; the player is told "today" (Anita, T-1893: "24 h" was hard to keep
consistent): labels and headings "Boardings today" or "Today"; their hovers say "N riders in
past 24h" (Anita: today can mean the past 24 hours, and "past 24h" is the short form). The profit (the
line panel's Profit row, the Lines tab's Money) is the one figure that is not a count: the expected profit
for today, a whole day of the line's local day now (fares at every hour of it on its own kind of day,
minus a day's running cost and its upkeep share), its hover "Expected profit today: $x". It changes when
the day turns, with the network or the riders worked out again, never with the hour, so it does not drift
through a weekend as the past 24 hours would [T-2475, notes/T-2475.md]. It shows only once the line runs
trains and a demand result has landed for it, an entry for the line or none once its edit is no longer pending (`profitKnown`, game/today.ts) [T-2536]: before
that the Lines tab's Money cell is blank, the line panel's Profit row reads "calculating..." (muted) when it has trains and is
absent when it has none, and the line is left out of the group and Lines totals, so the upkeep alone is never
shown as the line's loss. Deleting a rail or bus line with no stops asks nothing; one with stops asks "Delete <name>? Track
and stations stay." [T-2535]. The line panel's block has no heading, only the rule above it. The points (Cities
tab, top bar, leaderboard) say "in past 24h" too: they are a sum of the last 24 whole game
hours, moving once an hour. They are counted by the clock worker from the boarding
account's departures (`TrackApi.past_day`: the rest of yesterday from the time of day now, on
yesterday's kind of day, then today so far on today's; a stop's boardings in a stretch are the
riders arrived by its last departure in it less those still waiting, less the same at the
departure before it). The profit's fares are each line's hourly fare over its local day now; its
running cost is each hour's trains over 24 hours, the same for any 24 of them. Asked four times a wall second
while a panel shows them and the clock runs (`game/today.ts`); only the components showing them
re-render. While the boarding account works through new riders the last whole answer stands, and a
running line it has no account for keeps its last counts while its riders are pending or demand has an entry for it, so a line's day never reads as today so
far [T-2793]; a line left with no riders shows none [T-2834]. A line's fares after a stop edit, until its riders land, are its figures carried onto its stops
(`incomeFor` in game/money.ts). The map's demand views stay demand's whole-day picture.

**Riders by purpose** [T-165, `ui/inspector/PurposeSplit.tsx`]: under a line's riders and a
station's (or complex's) boardings in the past 24 hours, a thin stacked bar in the purpose colours, in the
purpose filter's order, with no key (T-669); hovering a segment gives the purpose and its share
("Work: 48% of riders", "<1%" for a sliver). Purposes with none here have no segment. Under the
purpose filter the purpose drawn is ringed in the bar.

## Lines tab and the line inspector

### The Lines tab

- **The Lines tab** has New line above its list (actions go above lists of any length). The list
  sorts by clicking a column header [T-310]: Name, City (below), Riders (today), Money (the line's expected profit today, fares less running cost and its share of upkeep, in the money format with negatives red; its hover gives "Expected profit today", the fares, the running cost, the upkeep and, on a weekend or holiday, a line saying so) [T-735, T-1893], Trains and Near. A click sorts by that
  column (names A to Z, figures biggest first, Near nearest first), a second reverses it, the sorted
  column has a small arrow, and each header's hover says what it sorts by. Names sort naturally, ties
  by name. The name keeps at least about six characters; the Money column goes first when the
  dock is too narrow, then the City column beside Near [T-746]. Lines default to by riders, most first [T-905]. The sort is kept while the page is open and re-runs each game hour.
  A double click on a row fits the map to the line's stops (a jump, as search does) after the
  single click selected it [T-569, `ui/locate.ts`]; the line inspector's locate button (cross
  hairs, hover "Show on the map", beside the pencil) does the same. On a line's opening day
  [T-1434, [ui-map.md](ui-map.md) "Opening day"] its Riders cell reads 0 until its figure first
  shows, then counts up to it over 2.4 s, fast at first and slowing to the end; the hover always
  gives the real figure.
  **A line whose riders are being worked out again after an edit** keeps a riders figure
  with a small label above it [T-2392], "calculating..." [T-2678], both for a line that has just started running
  (its blueprint constructed, or new) and for one already running whose stops, service or times changed (a train
  count, say, or track moved under it [T-2622]). The figure is the last there is
  [T-2678, `pendingRiders` in game/today.ts]: its count now; else its last count while the line exists
  (a line stopped and built again; a game load forgets them); else the riders a day demand has for it
  (once solved but not yet counted by the boarding account, or a new line's first estimate). With none,
  for a line that never had riders, the label stands alone. A group's row takes the label of its
  lines, "calculating..." first, and adds up the figures its rows show. The line inspector's counts
  block shows the same figure with its other figures, the label at the right of its heading and the
  purpose split hidden. The label stays
  until the riders that count the change land: its cluster's local result after the
  edit (T-1733, intercity lines too; their longer trips as last solved, or for an edit of 20 minutes'
  run or more those in a window of zones round it re-solved, T-1735) or the whole final [T-1665,
  `game/lineFigures.ts`], and only a result solved for a network made after the change [T-2658]: each
  network demand is given carries the world version it was made at (`networkWorld`, and `finalWorld`
  for the whole final shown, in game/demand.ts), and the label stays while the result shown is older
  than the change, so an earlier edit's solve finishing first (the edit waits for it, T-1734) never
  reads as the change's figures. The label then stays while the result has riders for the line that the
  boarding account has not counted yet (no count, or 0; `pendingOf` in game/today.ts, which the rows and the
  panel read), so a figure never reads 0 before its riders show. An edit that brings demand no new network (nothing demand reads
  changed) drops the label after 8 s; demand stopping drops it at once; a game load shows none.
  What an edit did to riders and profit is not in this tab: it is on the map
  ([ui-map.md](ui-map.md) "Edit note").
- **Trains to add** [T-472, T-1677, T-2236]: a row's train count (the trains its level needs now, the bare
  number under the "Trains" header [T-532]) is coloured by the line's load in the period now on today's kind of
  day, otherwise muted (a 0 is muted and has no hover): the stretch mean of the period's intervals, as the line
  panel's dot grades a level (`periodLoad` in `game/today.ts`, from `linesPeriods` for each line's own kind of
  day) [T-2279, T-2335]; before those numbers come, the busiest hour's mean load on the line's fullest stretch
  (`period_peaks` in `sim/src/track/boarding.rs`, the last of `Boarding::window`'s numbers per line, `crowdLoad`).
  Grades (`CROWD_GRADES`, the colour scale's steps): past 100% from 100.5%, orange from 130%, red from 140%;
  orange and red colour the count (`crowdClass`). The hover is the line panel's dot hover for the same level on
  the line's own kind of day without the level's hours [T-1672, T-1677, T-2334]: "Peak load 132%", the bar of
  intervals (west to east, as the dot's), then one line per period of the level; one function (`crowdLevel` in `game/today.ts`) builds both,
  from the `line_periods` numbers the tab asks for every running train line in one clock query per kind of day
  (`linesPeriods`, each line's numbers after their count). A bus line's hover is its bus count.
- **Scope and Near** [T-332; `game/listScope.ts`, `ScopeToggle` in `ui/listParts.tsx`]: beside New line
  (and above the Stations list) a toggle between the viewed place (a pin and its name, hover "Only
  <name>") and the whole world (a globe); one setting for both tabs, kept while the page is open, the
  place the default. The place is `viewedZone` (T-318); a station is in it when `placeAt` says so, a
  line when any stop is. A line with no stops and the selected row stay listed. Over unmodelled space
  the pin is disabled and the world is shown. So it is while no line stops in the viewed place
  (`scopeZone`, `scopeEmpty`; the globe is lit and the pin's hover says "No lines in Toronto yet, so
  the whole world is shown") [T-827]; the same for the Stations and Money tabs. The Near header is always there (beside "Name" until it
  is used); sorted by it, each row ends with its distance from the centre of the screen (a line's is
  its nearest stop's, hover "1.4 km from the centre of the screen"), rows without one last, and the
  column's width is given back to the names under any other sort. The centre is taken at each move end (`viewCentre`, set in
  `map/zoneFocus.ts`), so the order follows the view without moving while it is dragged.
- **City column** [T-670]: with the whole world listed and the lines' first stops in two or more cities, a
  City column follows Name (the first stop's city, named as the Cities list names it, `stationZone`;
  hover "The first stop is in Toronto"). It sorts like the others and takes room from the Riders and
  Trains columns.
- **Line groups and dragging** [T-1791, T-1813, `game/lineGroups.ts`, `game/handOrder.ts`; the model in [track-and-service.md](track-and-service.md) 6.3]:
  a row is dragged with the mouse or a pen once the pointer has moved 6 px (touch scrolls the list
  as before). Let go over the middle third of a line's row and the two make a group; over the
  middle of a group's row or one of its lines, the line joins that group. Let go over the top or
  bottom third, and the line goes into the gap there: out of its group if it was in one, unless the
  gap is inside an open group (between its row and its first line, or between two of its lines),
  where it goes into that group at that place; the gap under a group's last line counts as inside
  for that group's own lines and outside for any other. A group's row is dragged the same way, its
  whole row only into gaps (top or bottom half). While dragging, the row's bullet and name follow
  the pointer, the dragged row is faded, the row it would join is outlined and a gap is a 2 px
  line in the accent colour; a drop that would change nothing shows nothing; Esc cancels. A group
  left with one line is no group. **The hand order**: a drop into a gap that moves a row puts the
  whole list, every line in every place, in the player's own order (the order shown, then the
  move), saved with the game; while it holds, no column header is lit and new lines go last. A
  click on a header sorts by it and drops the hand order. Inside a group, the hand order is also
  the group view's left-to-right order where its layout has no reason of its own. A group
  is one row: a chevron (hover "Show the group's lines", "Close the group"), the group's bullet
  (below; hover lists its lines), its name (the player's, else its oldest line's name; a
  double click on it opens a name box, Enter or leaving it keeps the name, Esc drops it, and
  typing the oldest line's name back clears it), and its lines' riders, money and trains added up,
  each hover listing the lines' own figures (riders "1.59M riders in past 24h in total", money
  "Expected profit today in total: $x"; a rider changing between two of its lines counts twice), the trains
  uncoloured. Only running lines count; a group with none reads Planned (or Broken). A group
  shows open by default [T-1818], its lines indented 24 px, each a line row as any other, sorted
  by the tab's sort (or the hand order); the chevron closes it, and what the player closed stays
  closed while the page is open. The riders, money and trains columns have fixed widths that
  never shrink, so every row, a group's and its lines' and the header's, has them in the same
  place at any dock width; a wide bullet or a long group name gives up room from the name,
  never from the columns. The group row sorts among the lines by its
  totals (Near by its nearest line, City and the tie by its oldest line, Name by its name), and
  shows, with all its lines, when any of them is in the viewed place. A click on it opens the group
  view (below), a double click fits the map to all its lines'
  stops. A group edit is one undo step and wakes no demand; the hand order is outside the undo
  history, as a sort is.
- **A group's bullet** [T-1813, `ui/GroupPill.tsx`]: a pill with a see-through body, a 2 px
  outline and bold text in one colour, reading the group's name with a trailing " line" or " main
  line" taken off, any case ("Tokaido Main Line" reads "Tokaido"; "Line 1" stays), and likewise a
  trailing 線, 线, 本線, ライン or 선 unless a number is left ("東海道本線" reads "東海道", "1号线"
  stays). Its colour is the one most of the group's lines share, the busiest line's (riders in the
  past 24 hours, then the oldest) breaking a tie; drawn, it is kept legible on the page (T-1817,
  `legibleOn`): a colour lighter than 0.45 relative luminance is mixed toward black on the light
  shade until it reaches it, one darker than 0.14 toward white on the dark. The text's capital height (its ink box in other scripts)
  is centred in the pill, the baseline placed from a measure of the rendered pill because the zoomed dock rounds the
  line box (`inkBox` in ui/bullet.ts, `GroupPill`) [T-1835]. Both can be set by hand in the group view's
  title; set ones are kept on every line of the group, so they join, copy and undo with it, and a
  line leaving loses them. It is the group's mark in the Lines tab's group row, the line panel's
  group row and the group view's title.

### New line and the line editor

- **New line** [T-109 to T-112] selects an empty zero-train line, enters stop editing
  ([ui-building.md](ui-building.md)) and focuses its `Line m` name, text selected [T-813]; after a finger's tap the name is not focused, so the on-screen keyboard stays down [T-1756]. m is the lowest number no line stopping in the
  same city has as its name (any stop in the city's zone counts, so an intercity line from elsewhere
  takes its number there too; a line with no stops yet counts in the city being viewed), so a renamed
  line frees its number. The line's colour
  is the `LINE_PALETTE` colour farthest from the lines near it, ties at random [T-856, T-1278]; a
  city's first four lines take only red, blue, green, orange, purple or yellow, and later ones
  favour those six (their distance counted 1.3 times) [T-1894]; once its stops are picked, a colour
  within 19 (CIEDE2000) of a line beside it is replaced by the default farthest from those beside it
  [T-2291, ui-map.md "Line colours"]
  (`nextLineName`, `nextFreeColour` in `game/lineActions.ts`). A copy is named "<name> copy". The N key does it from any tab
  [T-670]; the same on a selected track no line runs on (its inspector's "Lines here" heading
  carries a plus button, hover "New line, starting at the stations on this track (N)") starts the
  line with the stations at the track's two ends, else empty.
- **The line editor** [T-670, T-672]: a double click on the badge, a click on it, a double click on the
  name or the pencil open one editor under the title: the name box (focused), a Symbol box and the
  colour picker (`ui/ColourPicker.tsx`: a saturation and brightness area, a hue strip, a hex box with the colour beside it, the palette swatches). Leaving the name box
  keeps the name and leaves the editor open; Enter in it, the check button (hover "Done") that
  replaces the pencil, a click anywhere outside the title and editor, or Esc closes it (Esc in the name box
  drops the typed name). A symbol with several characters widens the rounded badge; a blank symbol
  restores the default: a name that is a short code (up to five capitals or digits, "6X", "HBLR") whole,
  "Line 7" as 7, else the name's first letter [T-330]. **A symbol typed by hand sticks** [T-1812]: the
  Symbol box keeps what is typed until it is left (or Enter) and then sends only the symbol, so a
  rename landing meanwhile cannot reset the box, and a symbol equal to the name's own letter still
  counts as set. A rename, a colour and a symbol each send only their own field (`lineLook` with the
  rest left out, filled from the worker's current line), so one edit never carries a stale value over
  another still on its way. A line with a symbol set (the save's `symbol`, "" for none) keeps it
  through renames; one without follows its name. Dragging in the picker changes the badge and the stop dots in the
  panel at once; the map's line follows by one colour edit every 0.2 s at most (longer on a slow
  machine: 6 times the last edit's round trip), and the final colour goes as one edit on release, a
  swatch click, an arrow key step or a typed hex. Opening the editor scrolls the panel to its top,
  since the editor sits under the pinned title (the group view's too) [T-1894].

### The line inspector

- **Under the title** a muted line, left aligned with the title's bullet, gives the stop count
  and length, "24 stops, 16.3 km", led by "High-speed, " on a line of high-speed trains [T-1280].
  Above it, for a line in a group [T-1791, T-1813, T-1818], a row reading "Part of [bullet]
  group", the group's bullet inline (a click opens the group view, hover "Open the group", led
  by the group's full name where the bullet reads it shorter; a long name is cut with an
  ellipsis), and a Leave group button at the right (an arrow out of a box, hover "Leave group"),
  there for touch, where dragging rows is not; a bus line's panel has the same row. The row
  starts 6 px below the title, which covers that much of whatever follows it.
- **Actions** at the top, under the title and the planned or broken notice: Edit stops (hover; while stops are being edited it reads "Stop editing" and shows pressed, the button as wide in both states so the row never wraps differently, T-2525, T-2601), Copy line (duplicates route and settings with no trains, T-1161;
  the copy joins its source's group, making one of the
  two when the source had none, T-1791, and the copy's own panel opens with its stops being edited, T-1894; its name and symbol are the next
  letter after a trailing number ("Line 1" and symbol 1 give "Line 1a" and 1a, then 1b), else
  "<name> clone N" (a copy of "X clone 1" is "X clone 2") with the old symbol's first letter and N (t or tm gives t1, then t2), N the lowest
  not taken, T-1872, `game/cloneNames.ts`), and Delete line (at once: an undo brings it back, and track and stations stay; only a constructed bus route, which is final, is asked about, T-2919). While stops are being edited the Stops list (with a remove
  button on each stop) follows them there, above Schedule; otherwise it sits after the day's figures.
  Incomplete lines stay planned and run no trains.

#### Schedule and trains

- **Sections**, in order: Schedule (trains for each demand level, with the resulting headway;
  [track-and-service.md](track-and-service.md) 6.3; columns Demand, Trains and Headway, every
  cell one line, the Demand column cut with an ellipsis when short of room [T-692]; beside the
  heading two small muted icon buttons, copy and paste [T-1293, `game/scheduleCopy.ts`]: copy keeps
  the line's trains or headways per level, train length and kind of train in the browser
  (`localStorage` "anitabuilder-schedule-copy", so it outlasts the game and the visit; hover: the
  settings, "Headways 3 min, 5 min and 10 min, 6 cars, high speed"), and paste puts them on the line
  shown as separate edits (kind, length, then the trains or headways; a refusal stops the rest), the
  length cut to what its shortest platform fits (the paste hover and a notice say so), a loop's
  counts rounded up to even; paste is off until something is copied, its hover naming the line
  copied from; a third, on a running line, opens the **timetable chart** (below, hover "Show timetable chart" /
  "Hide timetable chart", pressed while open); each level's count or headway is set on its own, a new line's included (T-2123 took
  back T-2092's fill of every level; the tutorial asks for trains at each level instead); a Trains / Headway switch in the heading
  chooses what the player sets [T-625]: with Headway the headway column holds a stepper (steps to
  the whole minute above or below, so 5.37 min goes to 6 or 5 and 5 min to 6 or 4; none, or 1 to 240
  minutes; from 60 minutes up a click moves to the multiple of 15 above or below; minus from none gives 5 min, plus gives 15, minus at 1 min gives none [T-1872, T-2519,
  `nextHeadway`]; Shift-click steps to the multiple of 5 minutes (7 goes to 5 or 10, 5 to 10 or the
  1-minute limit) and Ctrl-click to the multiple of 20, the hover saying "Shift-click for 5 min,
  Ctrl-click for 20 min" [T-1888]; a click on the number types it in minutes) and the trains column the count it gives, 10 px clear of the stepper [T-1159]; while a
headway or count is still on its way to the worker (an edit takes seconds on a big network), the other
cell shows what it gives (the headway's count by `autoTrainsFor` with the level's limit, a count's
headway over the round trip), and the muted running headway waits for the edit, so stepping fast never
shows an earlier edit's figures [T-1276]; each button's hover names the train
  count it leads to and, on a running line, the cars it buys at the car price, takes from the
  depot, or frees [T-631]; a line whose track holds it just over the headway set shows the headway
  it runs at, muted, left of the stepper, with a hover naming what limits it as the capacity alerts
  name it ("Terminus at West takes one train every 4m00s at most, so 8 trains run every 4m07s",
  from the tail of `TrackApi.line_held`) [T-679]; a headway shorter than that limit, or by hand more
  trains than fit, is red, with the same hover; an edit refused for cash shakes its control for
  300 ms and paints the level's train count red until the cash covers it or another edit to the
  line goes through (`refusedForCash` in `game/world.ts`) [T-625]; hovering High, Medium or Low gives that
  level's times of day in the line's local time, the same every kind of day (T-2335), from the period table in
  `game/periods.ts`: "High demand period: 06:00-20:00", "Medium demand period: 20:00-24:00", "Low demand
  period: 00:00-06:00" (hours as the clock writes them, `hourSpan`, T-2271; `levelTimes`) [T-406, T-1536,
  T-1797]; on a
  running line one dot before the name of each level that runs trains (none for a level with 0 trains or no headway, T-1783) grades the level on the kind of day the clock is on [T-1671], by the
  load past the rated load, as in Trains to add [T-2236]: each interval's load is the busiest hour's mean load
  of the trains on it (the fuller direction) over the level's periods (High is its three periods, 6-10, 10-16
  and 16-20) as a share of the rated load, 160 a car (`period_peaks` in `Boarding::line_periods`); the dot's
  fill is 100% plus the power-3 mean over every interval of its load past 100% (`stretchLoad`,
  notes/T-2279.md) and its 1 px ring 1 px outside it (a gap between) the fullest interval's load, so one
  packed interval among many reads yellow inside a crimson ring [T-2279, T-2335]; its colour
  running smoothly with that load: green to 70%, yellow-green at 80%, yellow at 100%, orange at 130%, red at
  140% and crimson at crush's 150% (`crowdColour` in `game/palette.ts`, fed by `crowdLevel` in `game/today.ts`;
  the Traffic view's crowding hover colours its load the same way, [ui-map.md](ui-map.md)) [T-477, T-515,
  T-2236]. The dot and the level's name share one hover (`LevelName`, T-2334): the level's hours as above,
  "Peak load 132%", a bar of the line's intervals (the run between two stops, the fuller way) in the crowding
  colour of each one's load over the level's periods, all green when nothing is crowded, with a station dot
  at each stop between them and at both ends; where the dots would leave an interval under 6 px, only the end
  dots stay and the intervals are thinner pills (`tipStrip` with stops in `ui/tip.ts`, `STOP_CELL_PX`);
  the bar reads **west to east** (`lineReversed` in `game/northUp.ts`: the line's west end on the left, a line within
  about 22 degrees of north-south its north end, a loop as built; the bar has no hover of its own to flip)
  [T-1926, T-1901, T-2334, T-2661], with the first and last station's names right under it, small and muted, one left and one
  right aligned, each ellipsized (`tipEnds`, `endNames`) [T-2772], then, for a level of more than one period with trains, a line for each in time
  order: "10:00-16:00 load: 118%" [T-1523, T-1672, T-1797, T-2271, T-2334]; in the tutorial, where at most half
  the intervals are past 100%, "Only a short stretch is crowded. More trains would run mostly empty, and may
  cost more than they earn." last [T-2263]; the dot's hover reaches 6 px round it (5 px on the
  name's side) [T-2271]; riders left at the platform by trains at crush are the
  station panel's and the trouble list's, not the dots'; the trains steppers take Shift-click for 5 and Ctrl-click for 20 (their hover says so)
  [T-516]; under the table a **Trains** row with a Normal / High speed switch [T-817, T-1784], each
  option's hover giving its top speed ("High speed trains run up to 320 km/h on high speed track
  or 160 km/h on normal track", "Normal trains run up to 160 km/h") and, on a running line, what
  switching buys ("Buys 270 cars for US$x", or "Uses N cars from the depot"), nothing else; then
  **Average speed**, both ways over the round trip less the turning at the ends, so stops and
  waiting at busy track count (no hover, T-1893) [T-549, T-1159]; then a **Train length** stepper in cars [T-733], 1 up to what the line's
  shortest platform fits (platform / 20 m; a click on the number types it, and a typed length past
  the cap goes to the cap and offers the platforms' expansion below for the length typed, at most 20
  cars [T-1894]), the buttons with no hover and the
  number of cars between them hovering "8 cars can carry 1,280 riders, 1,920 at most" (160 a car at
  the rated load, 240 at crush, T-2186) and
  "Running a train costs $4 per km, plus $2 per car" (the line's train kind, shown unit) [T-1893];
  the label's hover gives the cap ("The 160 m platform at East fits
  8 cars"; the schedule's rows, Trains, Average speed and
  Train length are one run of rows with no rules between, their controls 4 px apart [T-1159]); stepping up to the cap stores the default (the most
  that fits, `cars` 0), so the trains follow the platforms again; a + at the cap (under 400 m, 20 cars) refuses (the stepper shakes and the refusal knock plays) and adds a row under
  it, "2 stations are too short for 7-car trains." with a button "Expand all for US$15.0M" (`LongerPlatforms`; "Expand it" for one station, no price when they are still blueprint): it lengthens each short stop's platform to the wanted cars x 20 m with
  `stationProps` (final for a built station, so it clears the undo history; blueprint ones are free and each its own undo step), then steps the trains up. The price is the worker's `platformCost` query
  (`TrackApi::platform_cost`, each built station's own charge by trial); if the cash falls short of the platforms plus the cars the longer trains buy, the button shakes and knocks and nothing changes [T-751]. Any stepper refuses the same way when its `onStep` returns false (a limit: 1 car, 0 trains, headway ends,
  platform 60-400 m, Build's target speed, panel size) or a promise of false (a worker refusal, whose notice knocked already); a train length or count the cash cannot cover is refused before it is sent [T-1844]; when full track holds trains back, an alert under the table names it ("Platform at
  Akabane is full", the station name a link that moves the camera there and selects the station [T-669, T-1393]) with the trains
  held per level beneath it ("Medium demand: 1 train held out of service"), one alert for each
  full place. Alerts are a plain list, each with a red (bad) or yellow (warning) circled
  exclamation point (`Alert` in `ui/widgets.tsx`, `--bad`, `--yellow`), no box [T-748]; a broken
  line's "Track this line ran on is gone" is one too.

#### Today on a running line

Today for a running line:
  riders by purpose,
  one Profit row (the expected profit today, fares less running cost and the line's share of upkeep,
  in the alert red when negative; the hover gives "Expected profit today: $x", fares, running trains and upkeep share, T-867, T-1443,
  and when the line's day is not a weekday one more line, "Today is a weekend", "Today is a holiday
  (Thanksgiving Day)" (no name when the calendar has none) or "Today is a holiday travel day", so a loss
  on it does not alarm, T-1893 `dayNote` in `game/today.ts`; the running cost is at the trains and cars the fares' demand result was solved with, so after a train count change both move when the new result lands, T-1492), the **Average load** in a percentage [T-820] (rider-seconds
  aboard over the seconds each trip is under way times the riders a vehicle holds at its rated load, 160 a car,
  over every trip of today's kind of day from `linePeriods`, so able to pass 100%, at most crush's 150% [T-2186];
  one rule for trains, buses and boats [T-2373, T-2919]: a bus's rated load is 53 of 80 at crush and a
  boat's 320 of 480; `averageLoad` in `ui/inspector/schedule.tsx`; the hover says so),
  how many trains left full as "x/y" of the trips that
  reached a stop in the 24 hours (the fourth number from the end of each `past_day` block) [T-481]
  and the riders left at platform as "x/y", x those a full train left behind and y those who
  wanted a train (boarded plus left behind) [T-230, T-239, T-293, T-669]; the block has no heading
  (T-1893), only a rule above it and none between its rows
  (the only other rule is above Stops) [T-692].

#### Stops

The stops (above Schedule while being edited [T-670]),
  **north up** [T-1894, `game/northUp.ts`; anything drawn left to right reads west first instead, T-2661]: the end farther north on top, or for a line within about
  22 degrees of east-west (north-south distance under 0.4 of east-west) the west end; a loop as
  built; each with its time from the stop on top (its hover: "Trains reach 320 km/h" from the stop before, then what limits them there:
  the trains' top speed, normal track, curves or a crossover, and when the stops are too close
  to reach it [T-817]; dropped from a row whose name, time and bullets do not all fit, T-1391) and
  boardings in the past 24 hours, each transfer stop (any station of its complex
  served by another line, or by the line itself at a stop other than the next or the one before,
  so a line coming back through a station shows its own bullet there [T-1391]) showing the two lines with most boardings there in the past
  24 hours as bullets and, when there are more, a "+" whose hover names the rest [T-549]; a **loop**
  shows each station once, its bar going on past the first and last stops and curving round to
  the right at both ends (its last stop, the first again, stays while stops are edited, holding
  "Open the loop") [T-1391]; a line **turned round a loop** (T-3021) shows one row per station, (with related lines too, the side bend drawn over its columns): the stem straight, the first pass's time on each row, the loop's stations after it, the loop as a bend at the side leaving the turning station and coming back in at its last one with a white chevron, a hook past an end whose loop has no station; a stem with a loop past each end, each with stations, reads from the stop order alone and shows a bend above and one below (T-3050); no self-transfer bullet; boardings summed over the passes; a stop served one way has its dot filled on one half only (the loop side) [T-3025, T-3040, T-3041]; each stop is a white dot with a 2 px ring in map ink, 12 px across (14 at
  a transfer stop) as in the related lines view, centred on the 6 px bar [T-1886]; the plain list's bar and dots are painted half a pixel off the pixel grid (`--soft` in `style.css`) as the related view's SVG is, so both views have the same soft edges and ring weight [T-2623]; **while stops are
  edited** a toggle above the list, "Exact edits", puts a rail line in manual mode (6.3): the list shows every
  visit, numbered, each row dragging as now, with a cross that skips that visit only (none on the ends or a turning
  stop), a "+" that opens a "Pick a station on the map" row after it (the next station click on the map puts the stop in
  there, through `orderStops`, refusing like a drag), and no Direction row; the list runs in order (never flipped) with no related-lines side drawing; the toggle always switches back,
  keeping the stops as a normal line unless the list folds [T-3024, T-3079]; a click on a stop's dot (a 20 px target, ringed red on hover) takes it out keeping the path
  and Shift-click between the ends ends the line there, as in the group view (hover "Click to remove
  the stop at 225 St", "Shift-click to end the line here"; none on a stop the line needs: a line of
  two, a loop's ends, where trains turn back), beside the remove button [T-1894]; **flat junctions and
  flat crossings on the route** [T-629, `TrackApi.line_marks`] as small marks on the diagram's bar between the
  two stops they lie between, spread evenly in the gap (the diagram is not to scale): a white diamond for a
  junction with no flyover and no station, a cross for a flat crossing, each with a hover naming it and
  the lines that use or cross it ("Flat crossing near Harbour", "Crosses Cross"); **in the Traffic
  track display** [T-1272] (not while stops are edited) each stretch of the diagram's bar is as wide as
  its riders over the day shown (both directions, the map's purpose filter, `dailyRiders`; after an
  edit, until demand has riders for the new stops, the last figures carried onto them by station or
  complex, 0 for a new stop's stretch, blank when under half the stops remain, `carryLoads`
  [T-1419], as on the map's line widths and hover), from 2 px
  for none to 14 px for the line's own busiest stretch (`STRETCH_MIN_PX`, `STRETCH_MAX_PX`), so a quiet
  line varies as much as a busy one, with round ends under its stops' dots [T-1886]; a stretch the map marks as crowded (`Renderer.fullStretch`,
  read every half second) has its darkening and red edges 2 px each side, 1 px of the panel's colour
  between them and the stretch (T-1284), the edges from half
  strength at the map's faintest to full (`PANEL_MARK_FLOOR`; on the panel's background the map's
  15% cannot be seen); each stretch's hover: "3 Av to 14 St-Union Sq", "334,700 riders per day", and
  "Peak load 132% at 08:00" where marked (T-2236); **related lines** [T-1168,
  notes/T-1168.md, T-1391]: an icon after "Stops" (two lines side by side, a station oval between them, the right line curving away at the bottom, T-1802; hover "Show
  related lines" / "Hide related lines", one setting for every line, on at first and kept while
  the page is open, kept while stops are edited, the stations passed then between the rows) draws
  the stops with the lines related to the line (`TrackApi.line_related`,
  `game/relatedLayout.ts`, `RelatedStops.tsx`): the line straight down the middle of the drawing; each
  line running beside it (shared track, or its track within 100 m + 3.5% of the first 5 km of its own
  run + 2% of the rest of the line's between stations both reach; a line meeting only twice gets 2%
  throughout unless neither line stops between the two; a line whose stops are all on the line's
  route always) in a 2.5 px column (as thin as a crossing line's bar; 3 px before T-2623, strong
  beside the 6 px line), 2.5 px apart from the line and
  from each other whether on shared or its own track [T-1786], right for one stopping more often over the span and left
  otherwise, at most 3 a side and only as many as fit in 40% of the panel; where the other line goes
  on, its column goes straight on 13 px past the line's own first or last row, and elsewhere turns
  out to its side half a row past where it leaves (turns at one gap 5 px apart); where it ends, the
  column ends at its stop; a span over one row only, or a line with no column left, is drawn as
  crossings; beside a loop, a line running on across the loop's first stop goes on past the last
  row; rows are the line's stops (as in the plain list, bullets leaving out the lines drawn there,
  north up the same way), and only while stops are edited, the side lines' stops at a station the
  line passes (interlined: a station of the complex is on its path), at the nearest point of the
  line, names fainter than muted text (36% of the text colour, T-2602), a click on the row's mark adding the stop (hover "Click to add a stop at 96
  St"; such a station stays in the "+ N stations passed" row's count and Add, T-3007) [T-1894]; a
  side line's bullet shows after the crossing lines' at the first and last rows both stop at, and
  at each row both stop at next to one only one of them does (where an express's stops part from a
  local's); a white pill with the dark rim over each run of neighbouring columns stopping at a row
  (10 px at a row with crossings, else 8; never sized by riders), its rim at 25% strength where the
  line itself does not stop in that mark [T-1795, T-2602]; the crossing lines as 3 px bars across the drawing, 2 px
  apart: the two bulleted ones at a row with up to two, and at a row with more [T-1786] all of
  them up to 6, the row 38 px tall, its pill 10 px wide and stretched to (bars − 1) × 5 + 13 px with
  5 px corners so each bar meets its straight side, and its bullets (up to 6, in the bars' order, the
  rest behind the "+") in two rows; a side line's own row keeps two bars. Hovers: a pill only ("6, 6 and 6X stop at 23 St",
  "4 and 5 pass"); the columns, their ends and the bars have none (T-2020), and a bullet names its line as everywhere.
  A click on a column, bullet or bar selects that line.

#### The timetable chart

[T-2603, T-2660, T-2665, `ui/inspector/MareyChart.tsx`, `ui/inspector/mareyShare.ts`, notes/T-2603.md,
notes/T-2660.md] A Marey chart of the selected rail line, in a panel over the map at the dock's foot (on
a phone, just above the dock): the line's chip and name, the other lines toggle (below), a close button
(hover "Close"), and the chart. Time runs across, 600 px wide, on the line's own local clock (ticks
every 5, 10, 15, 30, 60, 120, 180 or 240 minutes, the first giving at most 8); the stops run down by
their distance along the line (run 0's offsets, run 1's mapped through the stops both share), north up
as the stops list (the top-down rule, `drawnReversed`, not T-2661's left-to-right one), 20 px a stop
between 120 and 340 px, a grey rule at each stop and its name at the left where it clears the name
above by 12 px (the ends always). The span is by default 2 round trips or 3 of the longest headway, at
least 2 hours; on a busy line no more than 12 of its shortest headways or half a round trip (one trip
end to end), whichever is longer, and at least half an hour; at most 8 hours; in whole quarter hours [T-2665].
The mouse wheel over the chart zooms it out (down) or in (up), 1.25 times a notch, from a quarter hour
to 24 hours, kept until another line is shown. A quarter of the span is before now; it moves with the
clock, the past under a half veil and now a line in the accent colour; not scrollable.
Each trip the map draws (`TrackApi.line_trips`, the same departures, ends and hand-overs as
`trips`; the timetable, without the display guard's lag) is a line in the line's colour, both
directions alike, curved as its run profile goes (speeding up, slowing, standing at stops), then a
dashed line at the terminal while its train stands for its next departure. Hover on a trip thickens it
and names its ends: "Leaves Ash 08:12", "Reaches Gum 08:28". It follows the selected line; it closes
with its button, the close button, Esc, or when no rail line is selected.

**Other lines** [T-2660]: every other rail line that runs over the line's track has its trains drawn
on the stretches they share, mapped onto this line's distance through the track both use (direction
aware, so a line that joins for a few km shows only there; the network's path table, `pathTable`), in
its own colour, 1 px at 55% under the line's own trains; hover thickens one and names its line and
ends ("Q", "Leaves Ash 08:12", "Reaches Gum 08:28"). **Flat junctions and flat crossings** on the
line's path that another line uses or crosses (`TrackApi.line_marks`; flyovers and stations are not)
are a dashed rule across the chart at their distance, with the line panel's diamond (junction) or
cross (crossing) at its right end and the line panel's hover ("Flat junction near Harbour", "Also used
by A and C"). Where another route's trains cross the line's path at grade (the train guard's
conflicts, `TrackApi.guard_conflicts`: a flat crossing, or a junction move that crosses one of the
line's), a dot in that line's colour on the rule marks each of its trains passing, hover "Q passes at
08:12": the line's trains near a dot must take turns with it there. The toggle (the related lines icon,
hover "Hide other lines" / "Show other lines", pressed while on; shown only on a line with other lines
to draw) hides the other lines' trains and dots but not the rules; on at first, remembered on this
device with the display settings (`chartOtherLines`). Their trips are asked for the same window as
the line's, one `lineTrips` query a line.

The trips are asked for a quarter span either side and again as now enters the next quarter, the span
changes or the network changes. The time ticks and every train are one SVG layer built once per
answer and moved by a CSS transform as the clock runs, so a new minute redraws only the time labels,
the veil and the now line; each profile's curve is one SVG path reused by every trip on it.

#### The round trip

The round trip split into running, dwell, turning and waiting,
  naming where it waits most [T-031]. Time following slower trains on shared track is its own row,
  "Behind slower trains", apart from "Waiting at busy track", and its places name the line ahead
  ("Behind J trains") [T-1924]. Time standing at passing tracks for faster trains is
  "Waiting for faster trains", its places "At E, for Express trains" (both directions summed, from
  `TrackApi.line_waits`) [T-2232]. Each of these three rows has its own places right under it, the
  largest three; places under 0.5 s are left out. Under the round trip is the
  heading "Line improvements" over buttons set apart by spacing, each its label with the price small
  on a line under it (none where the stations are blueprint; hover "At B, C, D, E and F", past 8
  stations the first 7 and "and 3 more"). "Add passing tracks", on a line that shares track with a
  faster or slower one, adds passing tracks at every station `TrackApi.passing_spots` names in one
  `stationUpgrade` edit; the price is the tracks' alone, and the notice adds the trains a slower
  line then buys for its headway ("Passing tracks built at 5 stations for $225M. New trains cost
  $150M."; blueprint: "Passing tracks added to the blueprint at 5 stations."); gone once none is
  missing [T-2232]. "Add parking at end stations" and "Add parking at all stations" give the
  parking lot upgrade to those of the line's stations that lack it, again one `stationUpgrade`
  edit, priced by the worker's `upgradeCosts` ("Parking built at 38 stations for $760M."); the
  second is left out when every station is an end station, and a button is disabled once none is
  missing, its hover saying so ("The end stations already have parking") [T-2692]. A refusal
  shakes the button and the notice says why. A **loop** [T-798] has a Direction row first, a segmented
  One way (hover "Round the way it was drawn") or Both ways [T-2981], then its lap (one way round, both ways the longer way's)
  in place of the round trip, split the same way, with no turning row for a circle and "Turning at
  the end" for a balloon; its average speed is once round over the lap (a balloon's less its
  turning); run both ways, its trains steppers move by 2 and the count's hover says how many run each way; the
  stop count under the name counts the first stop once.
  **Top speed** [T-2987]: a rail line's panel has, under the round trip block and above the line improvements, a
  "Top speed" row with the fastest its trains reach on the route (the most of `line_stretch_speeds`' reached
  figures) and a mini "Limit" toggle beside it. On, a "Speed limit" row appears with a stepper (steps of 10 km/h,
  from 40 up to the train kind's top speed, 160 or 320), starting at the current top speed rounded down; off, the line
  has no limit. The limit is saved with the line and copies with a train-to-train schedule paste (with the train
  length and kind). Bus and ferry lines have none.
- A line's riders count every line actually boarded, transfers included; inter-city trips show
  only in its purpose split [T-182].

#### The cross section

[T-2734, `ui/inspector/LineProfile.tsx`, `workers/lineProfile.ts`, `game/groundHeights.ts`, notes/T-2734.md]
Last in a rail line's view, under Line improvements, the heading "Cross section" (its pointer x is divided by the panel's CSS zoom, `--ui`: the rect is in screen px, the strip's width in its own, T-2769) over a strip 72 px
tall and as wide as the panel: the line from its first stop (left) to its last along its length, on the
ground's real height above sea level (the hillshade's Terrain Tiles, read at the zoom whose pixel is about
the line's length / 600). The ground is a faint filled area. Water is the map's water colour, from its
surface down a nominal 4 m to the ground drawn under it (`WATER_M`; at least 1.5 px on the strip): where the
zones' water masks (read every 20 m along the line by the clock worker) say water, its surface the tiles'
height there, or where no mask covers the place and the tiles go below 0 m, the sea at 0 m. The track is
a 2 px line coloured by its level (`LEVEL_COLOURS`), level n drawn n x 5 m from the ground or the water's
surface (the sim's 8 m a level scaled to 5, so ramps keep their shape; `PROFILE_LEVEL_M`). Heights are
stretched to fill the strip (a range of at least 40 m), with no figure for the stretch; sea level is a faint
dashed rule with "0 m" at its left, where it falls inside the strip (the close-up's too). Stops are white dots ringed in map ink on the track; other rail lines'
track crossing the line's (edges of their paths off its path, an intersection more than 3 m from the
other edge's ends, at more than about 12 degrees) is a 2.5 px dot in the first such line's colour at
that track's own height; crossings at one height within 60 m, or of a same line within 400 m, are one.
Nothing is asked until the page is idle, the clock worker works the query in 6 ms slices between its
other messages, and tiles are decoded when idle: until the line and its ground are in, the strip is empty.

Pointer over the strip (or a finger on it) shows a close-up in the hover tag's style above the strip,
at one scale for every line: 2 km of line over 320 px and 1 m of height a pixel,
centred on the track at the pointer, with a dashed centre rule; the 2 km is shaded on the strip. A stop or
crossing within 4 px of the pointer on the strip takes the centre. The close-up draws the ground, track
(3 px), water, crossing dots (4 px) and stops with their names, then lines: "2.4 km from Ash" (from the
first stop), "Ground 23 m above sea level" ("Water at sea level" or "Water 12 m above sea level" over water,
"Ground height not loaded" without a tile), "Track at −1, 5 m below ground" ("... above the water", "...
under the water" over water, "Track at ground level" at 0), and one line for each crossing within 250 m, at
most 3 ("[5] crosses 10 m above", "[A] [C] cross at grade", relative to this line's track, within 2.5 m
being at grade). Asked again (`lineProfile`, the clock worker) when the line or network changes.

## The group view

[T-1791 stages 2 and 3, T-1809; `ui/inspector/GroupInspector.tsx`, layout `game/groupLayout.ts`,
check `checks/t1791-layout.mjs`; notes/T-1809.md] A line group's panel, opened from its Lines-tab
row and the group name in a line panel. Stops are edited in its diagram (below); the
line panel's own remove button works as before.

- **Title**: the group's bullet, large, its name and Show on the map (all its lines' stops). A
  click on the bullet (hover "Edit the name, bullet and colour (double click also opens it)") opens
  the editor as a line's badge does: the name box (the oldest line's name typed back clears it),
  a Bullet box (its placeholder the text from the name; blank gives the name's text back; Enter or
  leaving it sends it) and the colour picker, with a "Back to the lines' colour" button (arrows in
  a circle) beside the box once a colour is set; a click away or Esc closes it. No row per line: a
  line's bullet in the diagram opens its panel. The diagram's lines are taken in the Lines tab's
  hand order where there is one (T-1813).
- **Stops header**: the all-stops switch (hover "Show all stops" / "Show fewer stops", kept while
  the page is open) and Copy line, which opens a row of the group's bullets; a click on one copies
  that line (hover "Copy 4") into the group.
- **Rows**: one place (station complex, `stopKey`) each. They start from the line with most stops in
  its path order, north up as the line panel's stops (T-1894); each other line is matched to them by place, turned round if it runs the other
  way, and its own places go in as one block: between the rows it leaves and rejoins, or above its
  first shared row or below its last, whichever leaves the shorter bar running beside them. So a
  branch hangs off the row where it leaves and its stops stay together. A line sharing no place goes
  below the rest. Then, in the default view, a tick row moves up into the first tick row after its
  lines' previous places where only lines apart from them are, so two branches' unnamed stops sit
  side by side on shared rows (named rows never move).
- **Named rows and ticks**: by default only key stops get a row of 18 px with the name: each line's
  termini; where two lines meet and where they part; the stops on either side of a stretch one of
  two lines skips (an express's stops beside a local); stops where lines outside the group stop.
  Other stops are a white tick across the bar, 4 px tall with a 1 px outline in the pills' outline
  colour, in an 8 px row with no name; a row every line passes is a bare 8 px row, where a ghost can
  add the stop. The switch names every row (a row every line
  passes muted).
- **Together and apart**: two lines run together from a stop both make to the next unless each
  makes stops the other does not between them (skip-stop: drawn apart). Lines together at a row
  are drawn 2 px apart (5 px bars), the rest 24 px apart; each line's x holds as long as it can,
  so a bundle bends in where lines meet and out where they part, an S-curve between rows with
  room to run at about 50 degrees. The left-to-right order is chosen row by row: lines together
  side by side; inside a bundle, the lines alike (same stops, still together) for longer are
  nearer each other, so the one that differs or leaves soonest is outside; apart, lines that will
  run alike soon are nearer; each crossing against a neighbouring row costs as much as 8 rows of
  likeness, so the order changes where that pays. The rows are ordered down, then up (so the order
  a meeting needs reaches back up the branches before it), then down again against both neighbours.
- **Marks**: in a named row a white pill where a line stops, one elongated pill over lines
  stopping side by side, a passed station's pill with its outline at 40%. Lines outside the group
  stopping there are drawn as the related lines view's crossing bars but 2 px, across only the
  lines with a place in the row and 10 px past them, up to 6 (past 2 the row and pill grow to fit),
  and their bullets after the name (busiest two, or up to 6 in two rows where
  the row grows, then "+" with a hover naming the rest; a bar's click opens its line). Each line's
  bullet sits above its first stop and below its last (side by side where several end at one
  place, moved out to the side where another line runs through); while a line's top bullet is
  scrolled away, it floats at the top of the visible panel over the line. A line with no trains at
  its level now is drawn at 40%, its bullets too (hover adds "No trains at high demand"). While the
  pointer is on a line the map draws that line full and every other line muted (ui-map.md "Lines
  and track"), as stop editing does. Names are cut with an ellipsis; a click on one selects the station.
- **Editing stops**, every edit keeping the line's path and a normal service edit (demand
  re-solves, one undo each; sim `drop_stop`, `end_line_at`, `stop_passed`, track-and-service.md).
  Each line has its own click target at each of its places, half way to its neighbours (at most
  11 px a side) and the row's height, ticks included: over a stop the hovered line's part of the
  mark shows its colour (a dot in a pill, the tick filled), and the hover shows the line's bullet
  inline ("Click to remove [4] stop at 14 St-Union Sq", the bullet drawn in the text; `tipBullet`
  in ui/tip.ts); the click takes that line's stop out and no other's. A
  stop between the ends leaves the track as it is, an end stop's removal shortens the path to the
  new end. Shift-click on a stop between the ends ends the line there, the stops past it toward the
  nearer end going (hover's second line "Shift-click to end [4] here"). Over a passed station
  the mark darkens (a named row's faded pill) or a ghost tick shows, and the click adds the stop
  (hover "Click to add [4] stop at 23 St"). A stop that cannot go has no click and the plain
  hover: a bus line's, a line's last two, a loop's ends, and where trains turn back (hover "[4]
  turns back at X"). One edit at a time; a refusal says "The line has to keep this stop."

## The bus line panel

A bus line's panel (T-1464, ui/inspector/BusInspector.tsx) is the line panel cut down: Edit route
in place of Edit stops (Save route while the route is open, which saves it; T-1514), Copy line (a planned copy) and Delete line (a constructed route refunds
nothing, so it is asked about; a planned line goes at once); while planned, its Construct price with the route and the buses in the hover. Its title,
schedule and counts block are the line panel's own parts (`LineTitle.tsx`, `schedule.tsx`, T-2373), in
buses: buses or a headway per demand level with the same hovers, the crowding dots, the schedule's copy and
paste buttons (between a bus or ferry line and any other line only the buses or headways are carried, the
hovers saying so: "Headways 8 min, 15 min and 30 min", "4, 3 and 2 buses"; no timetable chart), the waiting
spinner on a level with buses while demand loads, and the average speed; no track limits (T-2919). Stops: the ends and the
train stations met, the street stops between folded into one row each, in the line panel's order (north or
west end on top) with its stop times (the time from the stop on top after each name, "12m30s").
The counts block, headingless as the line panel's: riders by purpose, Changing to trains (riders a day changing between this bus and any rail
line at the stations it meets; hidden while the riders are worked out again, as the purposes), the one-figure profit,
average load (against 53 a bus, its rated load) and Buses leaving full; no riders left at
platform, since a street stop keeps no crowd. Under the stops the line panel's round trip block: running,
dwell at stops (the line's dwell at each train station between the ends and 20 s at each other stop, both ways)
and turning at the ends. No train kind, length, stop editor or related lines. The Money tab shows buses owned, needed and spare once there are any;
the Build tab's quote has rows for bus routes and new buses.

**A ferry line's panel** (T-2194) is the same panel in boats: Edit route opens the ferry tool on its
ports (ui-building.md, "Drawing a ferry line"); Delete line removes it at once (its ports stay, its
boats go to the depot, and it is undoable); while planned, Construct's hover gives the ports and the
boats. Schedule in boats. The ports list, headed Ports, is a rail line's stops strip (T-2275): a
port's dot, its name and the time from the port on top (north or west end on top), the bullets met and its
boardings today (T-2919); in the Traffic view each stretch as wide as its
riders against the line's busiest, its hover "A to B" and the riders a day, as a rail line's (T-2274).
With three ports or more, a click on a port's dot takes it out of the line (hover "Click to remove
the stop at X"; the port stays) and dragging a port's row moves it, a line between rows marking where;
the route is searched again through the ports and saved as one undoable edit, refused with "No water
path found between A and B." where a leg finds no water. Not while the ferry tool has the line. A
bus line's stops follow its route (its stations met and street stops), so its list has no editing:
the route is edited on the map. Its counts block is the bus line's
(T-2195): riders, Changing to trains (between this boat and any rail line at its ports' complexes),
profit, average load (against 320 a boat at the rated load, so up to 150% at crush's 480), Boats leaving full
and Riders left at platform (a port keeps a waiting crowd, T-2919); then the round trip block. A boat's panel is a train's,
named "boat", its On board against two cars (88 seats, 320 at the rated load, 480 at crush), "Typical
boat this hour" (a bus's "Typical bus this hour"). A port's panel is a station's: "Port" under its
name, no platform length or upgrades, its upkeep a port's (US$10k a game day), Construct port and
Remove port. The Money tab shows boats owned, needed and spare once there are any, and a **Ferry**
fare row under High speed once a ferry line exists (a stepper by 0.05, 1.00x by default, a button back
to 1.00x once set, saved with the game; a change solves demand again, T-2195). The Build tab's quote
has rows for ports, new boats and the ferry route (free).
## Trains

- **The train inspector's "On board"** [T-083, T-164, T-230]: this train's own riders on the stretch
  it is running, from the boarding account ([track-and-service.md](track-and-service.md) 6.6),
  against seats and crush: a bar running to crush (240 a car, 150%), with a tick at the seats and
  one at the rated load (160 a car, 100%), filled in the accent to 100% and in the warning orange (`--warn`)
  past it, and a state ("Empty", "Seats free", "Standing", "Crowded" past 100%, "Full" at crush)
  [T-2186]; a bus's bar runs to its 80, its 100% tick at 53 (T-2919). Beside it the typical train this hour (the mean of its line's trains in the same
  direction drawn this hour, on the same stretch). A train is never over full: whoever does not fit
  waits on the platform. A line with no figures yet says "Calculating" [T-324].
- **Next stops** [T-868, `ui/inspector/TrainStops.tsx`]: under On board, the train's next four stops
  (to the end of its trip; a circle's up to the stop before its first), each with the bullets of
  the other lines calling at it or any station of its complex after its name [T-886] (each with its
  name on hover, a click selecting it; past the row's width they are cut off), the minutes until
  it arrives ("<1 min" under one; hover "Arrives in 2m40s", from the train's own running profile)
  and the riders getting off there: its load arriving (the boarding account's) times the line's
  alighting share at that stop in the period it arrives in, as the account lets them off; everyone
  at the end of the line. The row's hover: "61 of the 1,440 riders aboard get off at Grand St, and
  40 get on". The list moves on at each stop and the minutes each game minute.
- **Random rider** (heading) [T-1436, `ui/inspector/OneRider.tsx`, `game/riderSample.ts`]: under Next stops, while
  riders are aboard, one rider drawn from them, with two icon buttons in its heading (Pick another
  rider, Open this trip in the trip explorer). The account holds no per-rider data, so the draw
  follows how it lets people on and off: where the rider got on is a stop at or behind the train,
  weighted by what boarded there in the period and is still aboard (stayed on through each later
  stop by the line's alighting shares, as Next stops uses them); where they get off, a stop ahead
  weighted by those shares; the purpose, by this stretch's riders by purpose. Rows: **From** and
  **To** (station names; hover: the homes and places around it, as a station picked in the trip
  tool stands for them), **Purpose** (its colour; hover: the stretch's top purposes by share),
  **By train** and **By car** (the trip explorer's Effective time in minutes, asked of the demand
  workers for the two stations' points as the explorer would, in the period of the clock; the lower
  one bold; hover with the people making the trip who take the train and who drive). Between two zones the minutes
  are the regional market's. The rider stays the same until the train passes their stop, then a new
  one is drawn; a pair with no answer is redrawn up to six times. The explorer button selects a trip
  between the same two stations (Back returns to the train), sets the purpose filter to the rider's
  purpose and the explorer's period to the rider's.
- **A selected train is followed** [T-245, `map/trainFollow.ts`]: when its trip ends and the track
  model hands it to the next departure there (`Trip::next`), the selection moves to that trip as it
  leaves, so inspector and highlight carry on. A train leaving service stays selected where it
  finished ("This train finished its trip and left service").

## Stations tab and the station inspector

- **The Stations tab** follows the map's station ridership row (on the map since T-1448;
  [ui-demand-views.md](ui-demand-views.md) "Station ridership circles"). The list sorts like the Lines
  tab's [T-310], by Name, Riders or Near, Riders by default. A row dragged into the gap above or
below another (mouse or pen, top or bottom half of the row) goes there and puts the list in a hand
order, kept and dropped as the Lines tab's is [T-1813]. Riders follows the ridership choice
  [T-905]: under Origins or Transfers it is the map circle's own figure (riders starting a rail trip,
  or changing lines, a day from the demand solve shown, `game/stationUse.ts`); under None or Total it
  is boardings in the past 24 hours. Its hovers, header and footer say which. The scope
  toggle and distance column are the Lines tab's (above); the footer reads "N of M stations" when
  only the viewed place is listed. The first 80 rows are drawn when the tab opens, and 160 more each
  time the list's end comes within 400 px of view as it scrolls [T-1200]. A row is a station complex [T-530] (the button "Show stations separately" beside the place toggle, off at each load, gives each station its own row, T-2788): the name of its member with
  the most lines, every line stopping at any member, riders summed over the members, and a
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
  circle and the symbol are one SVG (`ui/bullet.ts`, T-705, T-745): the symbol is SVG text placed
  from its ink box (measured on a canvas at 240 px and scaled), its baseline on a whole device pixel
  for the device pixel ratio and CSS zoom it is drawn at, and the circle moved with it, so the ink,
  not the font's line box, is centred in the circle at any font, UI scale and pixel ratio, within
  half a device pixel (`checks/old/t745-bullets.mjs` measures it on the rendered pixels).
  A one-character bullet is a 16 px circle (a wide letter such as W a little wider), a longer one a
  pill of the same height, 4 px each side of its text. Bullets in a row are 2 px apart whatever their
  widths, wherever they are listed (`.bullets` in style.css, or the row's own 2 px gap); in two rows
  each row is spaced on its own; in a group diagram bullets side by side at a line's end are 2 px
  apart and each is centred on its line or the run's mean (`chipW` in `game/groupLayout.ts`) [T-2016].
  A line with no trains at the demand level in force on its own clock (a train count of 0 or a
  headway of 0, or a planned or broken line; counts typed but not yet applied count) has its
  bullet faded lightly, everywhere it is drawn: 0.7 opacity and a little greyed (its column in a
  group diagram at 0.7 too), and its hover adds "Not running" (`game/lineRunning.ts`, one set read
  by `Chip` and `lineBullet`; `checks/t1792-idle-bullets.mjs`) [T-1792, T-1907]. The GPU map layers draw no symbol bullets.
- **A click on a station selects its whole complex** [T-249]: complex view, whatever was selected
  before (there is no Station / Complex switch). A station on its own is a complex of one. Complex
  rules: [ui-map.md](ui-map.md) "Station complexes".
- **Complex view** [T-1812: only these sections; no subtitle, no Trip from here, no commuters and no
  inter-city trips]:
  - **Its lines** [T-246, T-250]: one row per line name, branches and patterns summed, with chip,
    name, riders waiting now (red when one of its platforms is full or its last train left riders
    behind) and boardings in the past 24 hours, busiest by the forecast first so the order holds through the day.
    The row's hover gives line, station and where its trains go; each number has its own hover. A
    click opens that line's busiest station there alone in station view (`openAlone`,
    `singleStation` in `game/complexView.ts`): its own inspector and outline, its
    subtitle ending with "Complex of 16 stations" as a link back (complex view itself has no subtitle). Any new selection, the same
    station included, shows the complex again.
  - **The complex map** (below), then the complex's "Boardings today" with their purpose split [T-1095].
  - **Closest stations** [T-312, T-323; `ui/inspector/ClosestStations.tsx`, `game/walks.ts`]: the
    three other stations with the shortest walk, built or blueprint (blueprint muted and marked),
    each with the chips of its lines (six at most, then "+n") and walking minutes ("7 min", the
    metres on hover). A click selects it (and moves the map if it is off screen). Its own complex
    and stations of its own name are left out, and each name is listed once. The walk is the
    demand model's routed walk ([demand-mode.md](demand-mode.md) 4.3) x the 1.3 detour at the model's 6 km/h; a station
    water cuts off shows "no walk", only when fewer than three can be walked to within 10 km.
  - **Passages**: each passage touching the complex, its length (price on hover) and Remove.
  - **Station building** [T-2218]: the complex's building (below), if any.
    The panel has no Trip from here button (the T key starts a trip from any tab) and no
    commuters, "where they live or work" or inter-city trips sections [T-1812].
- **Station view** (one station opened alone) has its level under the name ("Level +1"), then in place of the
  complex's lines and map the station's own lines, platform settings, and
  **Waiting now** [T-164, T-238]: each line and direction ("To Norwood-205 St") with the riders on the
  platform now in red; under it, in red, how many full trains left riders behind this hour and the
  longest wait, or else the riders arriving this hour and the longest wait. The counts drop the
  moment a train leaves with whoever fitted (a timer set for the station's next departure) and grow
  with arrivals each game minute. The rest (closest stations, links) is as in complex view, for the
  station alone.
- **Buttons come first** [T-1134, Anita: never put stuff at the bottom after an arbitrarily long
  list]: every inspector's buttons sit in one row (`Actions` in `ui/widgets.tsx`) right under its
  header, above all rows and lists. A station has, in station view, Construct station (a blueprint
  only) in a row (what placing a passage asks, when picking, is under it), then the **upgrade row** under a "Station upgrades" heading
  [T-1281, T-1415, `ui/inspector/StationUpgrades.tsx`]:
  icons for the terminal (two sidings), parking lot (P), waiting room (a bench under a roof),
  passing tracks (a through track with a track bowing out each side; hover "Allow faster trains to
  pass slower trains that are stopped here.", Anita's wording; only where the station is on double
  track or already has them) [T-2232], passage (two platforms and a walkway) and station building (a hall round a platform,
  T-2218, below); in complex view park and ride, the waiting room, the
  passage and the station building, the first two for every station of the complex at once [T-1528]: built at all of them or
  none, a complex where only some have it shows as not built and a click builds it at the rest,
  priced one station's price at each station that has none, and a click on a built one removes it
  from all. A single station's Remove station is at the panel's bottom, in red text (Anita, T-1852:
  the Delete tool does the same). Each upgrade is outlined when
  not built, dashed in the blueprint, filled when built; its hover gives its name, what it does
  in a sentence, its price and what a click does ("Click to build" on a built station, no ", paid at once"; none of the click lines ends in a period) [T-1893]. A click adds it to the blueprint, or on a built
  station builds it at once for its price (a notice says so), or takes it away again (nothing
  refunded once built); the passage's click starts picking the other station (below). At a station
  building's member (in complex view, a complex whose every station is one) the waiting room is
  greyed unless already bought, its hover the name, what it does and "The station building already
  counts as a waiting room." [T-2308]. A track has Construct for its price (a blueprint
  only) and Remove track (Remove blueprint on a blueprint) [T-1669]; a junction has Construct a flyover (built) or Add a flyover (blueprint), with "Paid at once"
  or "Added to the blueprint" beside it, or, for a blueprint flyover, Remove the flyover; a city has an icon button (the crosshair, hover "Go to Philadelphia") [T-1439]; a trip
  opened from a selection has Back.
- **Renaming** [T-534, `ui/inspector/NameEdit.tsx`]: a station's and a line's name both edit in
  place from a double click on the name or the pencil beside it (hover "Rename, or double click the
  name"), its text selected; Enter or leaving the box keeps it, Escape drops it, an empty name is
  ignored. A station's title also has the locate button [T-569] (cross hairs, hover "Show on the map",
  beside the pencil: the same move as a double click on its row) and the cycle button (two arrows, hover "Next suggested name"):
  each click renames it to the next of its suggestions, the cross street, the neighbourhood and the
  city or town, wrapping round, worked out then from where it stands (map/stationNames.ts,
  [track-and-service.md](track-and-service.md) "Placing stations"). The cross street needs the
  station on screen at zoom 13 or closer; with nothing else to offer it says "No other name is
  suggested here." **A name the player typed stays** [T-2683]: a rename or a pick of the cycle button
  marks the station's name as the player's (saved), and nothing renames it afterwards (not adding a
  station or a line to its complex, a complex splitting or merging, loading). A complex goes by one
  name in the Stations tab row and the diagram: its member with a typed name, else the one with the
  most lines (`game/complexLead.ts`); the map still labels every distinct name. A line group's name
  typed in its title stays even when it equals its first line's name.

## The complex map

[T-147, T-1289, `ui/inspector/ComplexMap.tsx`; T-176, T-246]

- **Drawn by the map's own renderer** [T-1289, Anita: "could we just copy the view in the world"]:
  the complex as the map draws it close in, whatever the map is looking at: platform boxes, the
  lines and track through them, trains, platform crowds, riders walking between platforms, the
  player's links as bridges, the selection's outline, on the map's land colour (no roads, water or
  names of the base map). The network renderer draws it with a flat camera of its own into a corner
  of the map's overlay canvas before each map frame, and the panel's canvas copies it before that
  frame clears it (`Overlay.setSnapshot`, map/overlay.ts), so it moves whenever the map's trains do.
- The inspector's width, 120-240 px tall, north up, every platform's ends inside a 24 px margin, at
  zoom 14 (where the map's platform boxes begin) to 17.5; a round scale bar in a corner.
- **Hover** on a platform: the station, then each line stopping there with its boardings and riders
  on its trains there a day (each direction the busier of arriving and leaving loads, every period
  summed) and how many change to or from other lines with its three biggest partners (T-176,
  `complexTransfers(members)` asked at each published round, [demand-views.md](demand-views.md)
  "Transfers"), then the riders waiting now and the room (T-246, `game/complexLive.ts`). Platforms on
  top of each other (parallel within 15 degrees, centre lines within 12 m) share one hover listing
  them all. A click opens that station alone. The player's links have a hover giving the distance.
- When the complex has more than one name, each name (the part before a slash) has one label beside
  its platforms wherever it is clear, else none.
- No notes under the map [T-1095, T-1106].

## Passages: linking stations

[T-197, T-1281, `game/links.ts`] The passage icon in the station panel's upgrade row starts a
passage. After a click on it the next station clicked is joined to the selected one (the pointer a
crosshair, a hand over a station); a click on anything else, Esc, the icon again, another
selection or a tool stops it. Refused with a notice, still picking: the same station, one already
in this complex, or one more than 1 km away centre to centre. Between two constructed stations it
is paid at once (US$10M plus US$90M a km, a notice), else it is built with the second of them.
"Passages", after the closest stations, lists each passage touching the complex with its two stations,
their chips, its length and Remove (nothing refunded once built). Only the player's passages can be
removed: stations the distance rule groups cannot be split. Passages are not in the Ctrl+Z history
(a setting, like fares); removing a station drops its passages. Demand counts a passage once both
stations are constructed.

## Station buildings

[T-2218, `map/buildingDraw.ts`, `map/buildingFloors.ts`, `game/buildingView.ts`; the rules in
[track-and-service.md](track-and-service.md) "Station buildings"; design notes/T-2208.md 3]

- **The tool** starts only from the station building icon at the end of the upgrade row (no Build tab
  button or key). On a complex with no building it opens on the smallest rectangle round the
  complex's boxes, 15 m out from them, its first edge along one platform taken as straight (its
  drawn line's first point to its last): of the platforms within 15° of each platform's direction,
  the largest such group wins, the longer total platform length on a tie, and the rectangle follows
  the longest platform in it (`offeredRectangle`, T-2363); on one with a building it opens that
  outline to edit. Its hover: "Station building", what it does (it joins
  the stations it touches; "Changing trains inside it feels 1 minute shorter." and "Waiting in it for a train feels 20%
  shorter.", T-2220, T-2276), "Costs
  $2,000 per m²" ($186 per sq ft with US units), then "Click, then change its outline on the map",
  or "Built. Click to change its outline", or "Click to stop" while drawing.
- **A closed outline** shows its corners as round handles and each edge's middle as a small one.
  Drag a corner to move it; right click a corner to delete it (3 at least); drag an edge's middle to
  make a new corner there. The bar at the map's bottom left says "New station building: 4 corners"
  (or "Station building outline") with Cancel (Esc) and Construct, or Save outline, (Enter).
- **Drawing by hand**: Esc drops a new outline (a second Esc leaves the tool). Each click places a
  corner, a ghost edge follows the pointer, the outline so far is shaded closed through it; a click on
  the first corner (12 px), a double click or Enter closes it; Backspace or a right click takes back
  the last corner.
- **The snap** (Ctrl held: none): the first edge turns to the axis of a platform within 50 m of the
  first corner when within 10°; every later edge to the first edge's direction or a right angle to it
  when within 10°; a corner within 8 px of a line along those directions through the first corner
  lands on it. With three corners at a right angle the fourth shows as a ghost and Enter takes it. A
  dragged corner snaps both its edges where they meet within 8 px of the pointer, else only the one
  nearer its line, so a rectangle's corner can be dragged out of square.
- **The tag** by the pointer, from the track model's quote of the outline (closed, or through the
  pointer while drawing): its floor ("102,000 m²"), its price ("$204M"; editing, "$32.2M for the added
  floor") and "13 stations inside"; or the refusal, red: "A station building has 3 to 24 corners",
  "The outline cannot cross itself", "Keep the corners within 1 km of each other", "A station building
  needs at least 1,000 m²", "Station buildings cannot overlap", "Draw the building round a built
  station". Over a handle it says what dragging does. Construct or Save outline refused for any of
  these (or money) keeps the tool on with the reason as a notice.
- **Notices**: "Station building built at Times Sq-42 St for $204M.", "Times Sq-42 St station building
  changed for $32.2M." (or "... changed."), "Removed the Times Sq-42 St station building."
- **In the panel**, after Passages, "Station building": its name, renamed in place (`NameEdit`, free),
  "Area" and its floor, "Room" and the riders its hall holds once the platforms are full, to the
  nearest 100 ("36,700 riders"; while platform crowds are drawn and riders wait there, "4,200/36,700
  riders"; hover "Riders who can wait in the hall once the platforms are full") [T-2219], then Edit
  outline and Remove building. Remove building, and the delete tool
  on the floor ("Remove the Times Sq-42 St station building"), ask first: "Remove the Times Sq-42 St
  station building? Nothing is refunded." Neither is in the Ctrl+Z history.
- **On the map**: [ui-map.md](ui-map.md) "Station buildings on the map" (one shape with its
  members' boxes, shops, the hall's crowd, the name); the one being edited is left to the tool. The
  floor's hover: its name and "Station building", then "No station inside" when empty, or, while
  platform crowds are drawn and riders wait in the hall, "4,200 riders waiting in the hall" [T-2219];
  a click on it selects the complex, as a car park's does.

## The trip explorer

[T-195, T-208, T-221, T-586, T-628; `map/tripView.ts`, `ui/inspector/TripInspector.tsx`,
`game/trip.ts`; the model's side in [demand-views.md](demand-views.md) "The trip explorer"]

The trips between two places as the view it was opened from counts them, homes at the start to
destinations at the end, and how the demand model's people would make one of them. The panel's
title is "Trip" with no line under it naming the ends: the map marks them [T-1253].

### Opening it

- **Starting it**: the T key from any tab [T-444] (the Settings button was removed, T-1843; tapping two demand bubbles also opens it), turns on the trip
  tool (a crosshair; a second press or Esc leaves it). Clicks alternate between start and end, a
  third starting a new trip; the hint at the bottom left says which is next.
- **Stations as ends** [T-221]: a click on a station (a hand cursor over one) takes the station's
  own cell as that end (the title's hover then says a station end stands for the homes or places
  right around it). Its own cell, because the model's subzones group cells by nearest station, so
  the trip mostly boards there.
- **From a selection** [T-208]: with a bubble selection, a click on a far-end bubble opens the
  trip for that pair (hovers end "Click to explore this trip"), homes to destinations. The
  selection's end is its busiest cell under the purpose filter, the far end the clicked bubble's
  cell with most of the selection's trips: the representative trip's two ends. The inspector then
  has a Back button to that selection (alone, with no line saying where it came from).
- **Between groups** [T-865]: a Shift+drag box over far-end bubbles (ui-demand-views.md) opens the
  trip between the selection and every far bubble in the box. The headline is
  exact over both groups, as for one bubble. Below it is an average over up to 3 x 3 cell pairs
  (`GROUP_SAMPLE_SIDE`): the busiest cell of each of the 3 near bubbles with most trip legs, to the
  busiest cell of each of the 3 far bubbles with most trips with the selection, each pair weighted
  by the near bubble's legs times the far bubble's trips (`averageTrips` in game/trip.ts). By train
  is then the Effective time (the weighted mean over the pairs the train is offered for; hover
  names the pair count) and Busiest routes: the routes merged by their lines and last station,
  each pair's weighted by its train chance, the five biggest with their share of the train trips
  and the rest as Other routes; no single route's legs, fare or time. By car, By walking and Train
  by period are the weighted means. The map draws the busiest routes, each pair's walks from its
  own cells.
- The trip is a selection: it stays after the tool is left, until Esc or an empty click, and while
  selected panning does not change the focused zone. It is asked again at each published round,
  and when the purpose filter or the day changes. In a zone no station reaches (open bare) the aux
  worker's no-rail copy answers, every period at once [T-979]. A trip with an end in another zone
  opens the focused zone's market pack, as a bubble selection does. An empty answer on a final
  result is asked once more after 3 s (`RETRY_MS` in map/tripView.ts), then the panel reads "This trip
  could not be worked out." rather than waiting; while one trip is asked it reads "Working out this
  trip", while the solve runs "Calculating demand" [T-1242].
- **No train route** [T-1252]: once answered, when the period shown has no train route (or there is
  no answer), a dotted straight line in map ink on a casing joins the two ends, so the pair still
  shows. Hover: "No train route between these two points" (or "No train route at this time of
  day" when another period has one) and "4.2 km in a straight line".

### Between zones

- **Between zones** [T-866; `ui/inspector/MarketTrip.tsx`, `IntercityApi::market_trip` in
  `sim/src/demand/intercity/market_trip.rs`]: with a bubble selection, a click on an inter-city
  bubble (a whole place zoomed out, a 6 km area up close; hover ends "Click to explore these
  trips") opens the trip between the selection and it; so does a trip tool trip with an end
  outside the focused zone's cells (the cell of the end inside, as homes for the start or
  destinations for the end, to the far 6 km area with trips with it nearest the other end, within
  6 km, else the nearest such place within 60 km, else [T-1902] the two places whole, as when both
  ends are outside: the places of the zones the two ends are in, else those whose centres are
  nearest within 60 km). A city centre where few people live has almost no longer trips of its own,
  so its cell finds no far end and the two whole places answer. Every other pair of ends [T-885, T-881; the table in notes/T-885.md]: a
  Shift box over inter-city bubbles takes them as one far end (each asked, then merged:
  `mergeMarketPeriods`, trips added, times trip weighted, the five biggest routes kept); a zone
  selected in the city view, then another zone's bubble (or a Shift box over several) is the
  market between the two places, every 6 km area pair counted whole (`market_trip_between`);
  a zone selected, then bubbles of the focused zone zoomed in, is the market between them and that
  zone. It shows the regional
  market's inter-city trips: every 6 km
  area pair between the two ends, the selection's areas weighted as the bubbles weigh them,
  each pair's choice worked out as the solve does (every pair rail can serve, sampled or not),
  added up by its trips; and the commutes between zones and everyday outings between them, so the
  total is the inter-city bubble's count [T-880]. The panel: the trips a day and the distance
  (between two places both ways, as the city view's corridors count them; from a selection one
  way, as its bubbles count them [T-1148]; with commutes or everyday outings the count is all
  three added up, the hover giving each, and the inter-city trips get their own **Inter-city
  trips** head with their count), a Train / Plane / Car bar and rows (no Train when no period
  offers rail); then, where the two ends have commutes between zones [T-880], **Commutes** with
  its legs a day and a Train / Car bar and rows (`CPair`'s choice in the solve; from a selection
  the commuters living there, or working there, between two places both ways), their train
  riders in the routes below; then,
  where the two ends have everyday outings across a zone line [T-1229], **Everyday trips** with
  its legs a day and a Train / Walking / Car bar and rows (errands, visits, school and medical
  trips as the market chooses them, the walkers rail takes out; between two places both ways, from
  a selection the outings of its homes, or arriving at its activities), their train trips in the
  routes below with the inter-city ones, the block absent while no network reaches the market; from a
  selection or a 6 km area with one far place, **Between the two places**, the whole places'
  trips both ways as the solve split them (the corridor's, commutes left out), so a selection no
  station reaches is not read as the corridor; By train (the period with most trips, "Inter-city,
  midday") with the main route ("To Astor Pl" and "From Stratford" carry the areas' felt minutes
  to and from the stations, muted, since the market knows no single walk), Fare, Time (station to
  station), then [T-1902] "Trips possible by train" (the share of the trips on pairs with a
  station in reach at both ends; no hover), Effective time (the pairs' felt cost, averaged over the
  pairs the train serves and their riders' income groups; no hover, since an average over pairs has
  no one equation), then "Car on the same trips" (or "Plane", or "Car or plane" where the plane has
  0.5% of the trips or more: the other ways as the choice weighs them together, the reference minus
  ln(rest) over the scale, over the same pairs and riders; hover "Effective time by car on trips
  possible by train."), so the train's share reads as its share of the trips possible by train
  times that share; a far end not open in the game reads "No route."; the
  main route's share ("43% of train users between these cities take this exact route", "places"
  when an end is a 6 km area or a selection) and up to four other routes (routes under 0.5% of the riders are left out,
  but while any rail rider is counted the biggest route stays, so riders spread thin over many
  station pairs still show their trip [T-1500]); **By car** [T-1148]: Drive (15 minutes plus the
  road at 85 km/h), Running costs (each person's dollars), Parking, any Channel crossing or Ferry
  to Hokkaido (each person's fare and its minutes) and Effective time, whose hover is the sum; the
  parts follow from the sim's minutes and mean distance by the rule (`carPartsOf` in
  `game/marketRule.ts`), parking the rest; **By plane** where the pair flies (one far place): the
  drives to and from the airports, 135 minutes at them, the flight, the fare and the effective
  time, from the market rows; the car's and the plane's money at the travellers' mean minutes a
  dollar (`travelWeight`, the sim's `Place::trip_w`: the fare weight times the income groups'
  factors by their trips, so the longer trips' lower weight on money), as the choice weighs it
  [T-1902]; Train by period (only when rail is in play). The routes are drawn
  on the map as a local trip's, from the selection's busiest cell to the far end's centre. Rail
  is not worked out when a place's zone at either end is not open in the game [T-991]: the panel
  then shows only the straight-line distance between the two places' centres and By car from it
  (the model's car rules for that pair, or "No road joins the two ends"), and no trip counts, bar
  or Train by period. With no distance known, neither the distance nor By car is shown.
- **Between places with no station** [T-1148]: two whole places (the city view's path) where
  either end, or every far place, has no station within 40 km of its zone's box (`stationNear`;
  past the market's 15 km drive to a station) have no train in play, so the answer comes from
  the market rows at once with no worker (`rowsAnswer` in `game/marketRule.ts`, a mirror of the
  sim's place-pair rule held to it by `checks/t1148-market-rule.mjs`): the places' trips by the
  corridor's split, the car and the distance centre to centre. By train then reads "There is no
  station near Paris region." and the bar has no Train.

### The headline

- **The headline follows the view** [T-586]: under the purpose filter and the day shown, the trips
  a day between the two places (a line "124 trips per day, 2.5 km apart"; its hover the
  exact count and, under All purposes, each purpose's), then a bar and rows of their shares by
  train, walking and driving. Opened from a far-end bubble the places are the selection and every
  cell of that bubble, the same sums as the bubble's hover (`tripMix` over the selection's far end,
  `flows`), so the two agree; from the trip tool, the start's cell as homes and the end's cell. The
  walk row shows when walking the whole way takes under 30 minutes [T-628], or whatever it takes
  when walking has a share [T-992]. When the view counts no trips between them, the headline is the representative
  trip's chances in the period shown.
- **Between two land parts** [T-2215; T-2214's `apart`, slot 47 of `DemandApi.trip`]: when the two cells are on
  land parts no road joins, the headline's Walk and Drive rows go (their counted trips are No route: the far end's
  cells still split them as walking and driving, T-2256) and a **No route** row and bar part follow Train, in the No
  route grey, hover "No route: 38% of the 120 trips per weekday between the two places" and "No road joins these
  two places". By car and By walking each read "No road between these two places" in place of their rows; Train by
  period's hovers give "no route" in place of drive and walk. A sampled rider's By car (the train inspector's
  Random rider) reads "none" with that line as its hover.
- **Whole trips** [T-911]: counted shares are rounded to what whole trips allow: the trip count
  rounded (at least 1; under half a trip the line reads "Under 1 trip") is dealt among the modes by
  largest remainder (`wholePeople` in `game/trip.ts`), and each share is its trips over that count,
  so 1 trip at 30% train and 70% drive shows 100% drive. The bar, the rows and the market view's
  Train / Plane / Car (over the day's trips, and each Train by period row over its period's) are
  rounded so; the rows' hovers add the model's own share ("Model: 30.4%"). The route shares under
  By train (main route's sentence, Other routes, Smaller routes, Busiest routes) are dealt the same
  way among the train trips the headline counts (the market's: the period's train trips), the
  routes too small to list taking a leftover rider only after every listed route, and routes no
  one takes at that count left out; with no counted train trips they stay the model's. The
  representative trip's own chances (Train by period, a trip the view counts none of) are not
  rounded. Past a couple of hundred trips the rounding changes no whole percent.

### The representative trip

- **The representative trip**: everything below the headline is one trip read by the model: under
  All purposes the purpose with most of the trips between the two places (work when none), else
  the filter's (school and university: the kind with more trips between the two areas); a commute
  through `DemandApi.trip`, an everyday purpose through `EverydayApi.trip`. Its period is the one
  with most of that purpose's trips, until a "Train by period" row is clicked; a new pair of places
  starts from the busiest again. "By train" names the purpose and period at its right, its hover
  saying why those.
- **The inspector**: under "By train" the main route as rows (walk to the first
  station, each line ridden with its chip and the station it goes to, each change and walk, the
  walk from the last station; each row's minutes its wait and ride; a ride access [T-374] reads
  "Drive to" with its drive minutes; the hovers give real minutes only), the fare (with any
  transfer fares [T-1801], its hover the base fare and, where one is set, the transfer fare at each
  change), and on a route
  that starts with a drive to the station a **Car** row [T-915] (dollars: the car's cost per trip
  and per km on the road to the station plus any paid parking there, its hover the two apart),
  where full trains add anything a **Crowding** row (the rides' perceived minutes beyond their
  real ones) and a **Waiting for a train with room** row (minutes waited after full trains, T-188;
  T-1070), each with a hover saying how the sim counts it, **Time** (real minutes; its
  hover the sum, "Walk 6.3 + wait 2.1 + ride 10.1 = 19 min") and **Effective time** [T-699, T-549]
  (the minutes the model weighs against the other methods; its hover is only the equation, the
  first route's terms in the rows' order, each real amount times its weight, "Walk 6.3 × 1.47 +
  wait 2.1 + ride 10.1 + crowding 1.0 + change 2 + fare $1.92 × 3 − choice of routes 11.8 = 16
  min", the wait for a train with room its own term after the wait, the transfer fares a term
  after the fare, "transfer fare $1.00 × 3", where the route changes and one is set [T-1801], two
  changes or more as "changes 2 × 2" [T-717]; a drive to the station is its parts [T-915],
  "Starting the drive 15 + drive 9.2 × 2.66 + parking at the station 7 + parking at home 3 + car
  $2.85 × 3", or "drop-off 10" where that is cheaper than parking; the
  last term is what the model's pooled figure, over every station and route people use, is below
  this route's sum, so the equation always adds to the row, and one plain line under the equation
  says choice of routes is the time saved by having several ways to go; the terms are the worker's
  perceived minutes, `routeTerms` in `game/trip.ts`), then one sentence with the share of the
  trip's train users on the route and the same trains from other platforms or walks added ("65% of
  train users between these places take this exact route."), with up to four
  "Other routes" and their shares, then a "Smaller routes" row with what is left [T-709]; the
  sentence's and the row's hover say how many pairs of stations the users spread over and what the
  busiest pair takes [T-709].
  **A click on an other route opens it** under its row, one at a time, a second click closing it
  [T-870, `openRoute` in `game/trip.ts`]: its legs as the main route's rows, Fare, Time, Effective
  time (the route's own sum, its hover the equation without choice of routes) and "Against the
  main route" (the two own sums' difference in minutes, "+6 min" or "same"; its hover gives both
  sums and each term's difference, "Ride +5.8 min"). The opened route stays open through new rounds
  and periods while the trip is the same. "By car" and "By walking" read the same
  way. Car: **Drive** (real minutes; its hover the straight line, the road length and the average
  speed of this drive, `DemandApi.trip`'s road km over its minutes), **Parking** (minutes, with the
  price of parking and any congestion charge counted as minutes), **Car ownership** (dollars) and
  Effective time ("Drive 17.6 × 1.33 + parking 67.3 + ownership $1.99 × 3 = 97 min"). Walking: one
  Effective time row [T-704] (its hover "Walk 52.4 × 1.5 = 79 min" holds the real minutes of
  walking the whole way); over 10 km the whole walking section is left out [T-935]. "Train by period": a row a period with the representative
  trip's chance of the train, a click showing that period. Without a train it says why [T-910]: no
  station within a walk or a drive of the start, none within a walk of the end, no service in the
  period, or the way to and from the trains over `pair_walk_max` (80 felt minutes together), named
  by what holds it back: on a trip under `ride_min_km` (5 km) "too far to walk to, and the trip is
  too short to drive to one"; with no station within a drive, "none is within a drive of the start";
  else the bigger end: the end's, "No train trip found" [T-2144], or the start's, "The stations
  near the start are too far, even by car". Its hover gives the two ends' felt minutes and the limit.
  Where every station pair takes longer than walking there [T-2265]: "Walking there is quicker than
  the train.", its hover "By train at least 34 min, walking 28 min." and "Nobody takes a train that
  takes longer than walking." (real minutes, the quickest pair's, `DemandApi.trip` slot 44).
  **When the headline counts train trips anyway** [T-1902] (at least one whole trip; the headline
  counts the homes and places around the two points; the commuters' far end goes only to cells within
  the walk limit of the selection's nearest end [T-1920], but an everyday purpose's riders, a far
  subzone with no cell in reach and the walking rule's station pairs can still land train riders on
  cells the cells' own trip gives none), a trip whose walks are too long
  or whose station pairs all take longer than walking there shows the route read with those limits lifted
  (`DemandApi.trip` slot 46): one line "The stations are too far from these exact points. Trips
  from nearby take this route." (or "Walking is quicker from these exact points. ...", its hover
  "From these two points walking takes 28 min, less than any train. ..."), its hover
  the felt minutes and the limit, then the route's rows, fare, time and
  effective time as usual; its chances and Train by period stay the trip's own. With no train
  counted the reason stays and the lifted route is not drawn.
  **Station names** in the route's rows (the walk or drive to the first station, each ride's "to",
  a change's walk, the walk from the last; the market's "To" and "From" rows) are links: a click
  centres the map on the station, at zoom 12 or closer, and keeps the trip selected [T-1902]. Every row and bar segment has a hover with its figure and what it is made
  of. The trip is the two places' own (each one's walk to its stations, T-720); where the map's
  bubbles count it differently, grouped with the homes and places near each end, the period rows'
  hovers add that count ("Counted with the homes and places near here, as the map's bubbles count
  it: train 62%, drive 30%, walk 8%") and the first walk's hover the group's walk when it is a
  minute or more apart.
- **On the map**: a veil of the land colour at 60%; the start an open ring and the end a filled
  disc; the main route over the veil in its lines' colours along their track (straight between
  stations only when the drawn network no longer matches the solve's), with a dark casing and the
  boarding and leaving stations bigger; walks dashed and straight; other routes thinner and faint
  under it, except an opened one [T-870], drawn at full strength a little thinner than the main
  route, on top of it, with its stations marked. The main route's boarding, change and last
  stations are named beside their dots in map ink on a halo of the land, a name that would overlap
  one already drawn left out [T-1902]. Hovering a ride gives line, stations, stops, minutes after the wait, crowding's effect
  and the route's share; a walk its distance and minutes; a station what the route does there.

- The platform stepper reads in cars and is called station length (T-2785).
