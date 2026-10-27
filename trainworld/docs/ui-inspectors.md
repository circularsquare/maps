[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · **inspectors** ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the Lines and Stations tabs and the inspectors

The inspector, under the dock's tabs, shows whatever is selected: a line, a station complex (or
one station), a stretch of track, a junction, a train, bubbles, a city or zone, or a trip. Nothing
selected leaves it blank, with no text. An empty click on the map or Escape clears the selection.
One component per kind in `ui/inspector/` (`StationInspector.tsx`, `TrainInspector.tsx`, `TrackInspector.tsx` and `JunctionInspector.tsx` hold station, train, track and junction). The track and junction inspectors are in [ui-building.md](ui-building.md) and
[track-and-service.md](track-and-service.md) 6.2; a bubble selection's and a city's (their
inter-city trips by train, car and plane) in [ui-demand-views.md](ui-demand-views.md), the city
view's own rows below.

## The city view

[T-1438; `ui/inspector/CityInspector.tsx` with `CityDetails.tsx`] A city or zone picked by its
inter-city bubble zoomed out, or by a click on its row in the Cities tab (a double click there moves
the camera, as in the Stations tab). Under the title and the go-to icon, rows: **Train share** (the
zone's, as the top bar gives it; hover the split with counts), and for a zone with a built station
**Points** (the past 24 game hours, hover as the Cities tab's), **Profit** ("$389M a day", the Cities
tab's figure and hover, losses red) and **Goals reached** (each goal reached, the star's as a small
star before "10%", then "20%" and on; hover when it was reached; "None" before the first)
[T-1437]. Then what a bubble click showed before ([ui-demand-views.md](ui-demand-views.md)): within
the zone, with the goal ticks over its mode bar as on the top bar, to other places, the top three
destinations. Then, for a zone with a built station: **By purpose**, the purpose bar of the
inspectors (`PurposeSplit`, no key) over the boardings at the zone's stations on every line;
**Top lines**, the five lines with most boardings at the zone's stations in the past 24 hours
(bullet, name, count with hover); **Top stations**, the five complexes (or lone stations) in the
zone with most boardings in the past 24 hours, named after the member with most lines. A click on a
line or station row opens its view.

## The past 24 hours, not a forecast

[T-293, T-446] The line and station panels and the Lines and Stations tabs count the past 24
hours, not a whole-day forecast, so nothing resets at midnight: riders, boardings, fares, running
cost, trains that left full, riders left behind. Labels say "in past 24 h" or "24 h";
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
  sorts by clicking a column header [T-310]: Name, City (below), Riders (past 24 hours), Money (the line's fares less its running cost over the past 24 hours, in the money format with negatives red; its hover gives the fares, the running cost and the net) [T-735], Trains and Near. A click sorts by that
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
  The "+N daily rides" after a stop edit is not in this tab: it is on the map
  ([ui-map.md](ui-map.md) "Opening day").
- **Trains to add** [T-472, T-491, T-818, T-1255]: a row's train count (the trains its level needs now, the bare
  number under the "Trains" header [T-532]) is coloured by the crowding score of the line's period now, otherwise muted. The score is the
  average extra wait for a train with room of the riders at the platforms where trains come full
  [T-1255]: rider-minutes spent on the platform after a full train left them, each waiting the
  gap to the next departure, and 30 minutes for each rider a full platform turned away, over the
  riders arriving, counting only the platforms (a direction of the line at a stop) whose riders
  wait a minute or more on average in the period (`BUSY_PLATFORM_S` in
  `sim/src/track/boarding.rs`); every rider counts alike (T-818's fare weighting was dropped with
  T-1255): `crowding` in `game/today.ts`. Grades (`CROWD_GRADES`, the
  colour scale's steps): yellow from 6 min, orange from 9, red from 15; orange and red colour the
  count (`crowdClass`). Hover "Morning peak: where trains come full, riders face a wait of 9.4 min
  on average for a train with room" (or "riders get on the first train with room"), then "4/12 trains running now are full on their
  trip" (trains from leaving the first stop until they could leave again from the far end,
  `Boarding::under_way`). The figures end each `past_day` block (`Boarding::crowding`).
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
- **New line** [T-109 to T-112] selects an empty zero-train line, enters stop editing
  ([ui-building.md](ui-building.md)) and focuses its `Line m` name, text selected [T-813]. m is the lowest number no line in the same
  city has as its name (a line's city is its first stop's zone, as the City column; a line with no
  stops yet counts in the city being viewed), so a renamed line frees its number. The line's colour
  is a random `LINE_PALETTE` colour no line in that city uses, else a random one of the least used [T-856]
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
  "Line 7" as 7, else the name's first letter [T-330]. Dragging in the picker changes the badge and the stop dots in the
  panel at once; the map's line follows by one colour edit every 0.2 s at most (longer on a slow
  machine: 6 times the last edit's round trip), and the final colour goes as one edit on release, a
  swatch click, an arrow key step or a typed hex.
- **Under the title** a muted line gives the stop count and length, "24 stops, 16.3 km", led by
  "High-speed, " on a line of high-speed trains [T-1280].
- **Actions** at the top, under the title and the planned or broken notice: Edit stops (hover and key L), Copy line (duplicates route and settings with no trains, T-1161, then opens
  naming and editing; the copy keeps the symbol its line shows, T-1475), and Delete line (asked first). While stops are being edited the Stops list (with a remove
  button on each stop) follows them there, above Schedule; otherwise it sits after the day's figures.
  Incomplete lines stay planned and run no trains.
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
  copied from; a Trains / Headway switch in the heading
  chooses what the player sets [T-625]: with Headway the headway column holds a stepper (30 s steps
  under 5 minutes, 1 minute under 15, 5 above; none, or 1 to 120 minutes; a click on the number types
  it in minutes) and the trains column the count it gives, 10 px clear of the stepper [T-1159]; while a
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
  level's times of day in the line's local time, from the period table in `game/periods.ts`:
  "High demand: 06:00 to 10:00, 16:00 to 20:00"; weekends use the same hours) [T-406], then
  "How many trains run in this time interval. (Which time interval a scheduled train falls into is
  determined by when the train reaches the midpoint of its route.)" (the steppers have no such
  hover) [T-823, T-835, T-1242]; on a
  running line one dot before each level's name grades the level on today's kind of day by the
  average wait for a train with room at the platforms where trains come full, as in Trains to add
  [T-1255] (the level's periods averaged by those platforms' riders; High is its two peaks), its colour running smoothly with that wait in minutes: green to 0.2, yellow-green
  at 3, yellow at 6, orange at 9, red at 15 and a deep red from 25 (`crowdColour` in
  `game/palette.ts`, fed `avgWait` in `CrowdDots`; `Boarding::line_periods`; the Traffic view's
  full-stretch hover colours its wait the same way, [ui-map.md](ui-map.md)) [T-692] [T-673] [T-477, T-491, T-515, T-1160, T-1255]; hover "Where trains come full,
  riders face a wait of 11 min on average for a train with room" (or "Riders get on the first train
  with room"), then a line for each of the level's periods with trains, in time order: "10-16: 4 of
  10 trains full, 2k riders went another way" (the riders part only when any did) [T-1523],
  and "Most of the waiting is at Gare du Nord" (the stop with the most rider-minutes
  waited over the level's periods, from the tail of `line_periods`) when anyone waits [T-818]; the trains steppers take Shift-click for 5 and Ctrl-click for 20 (their hover says so)
  [T-516]; under the table a **Trains** row with an Ordinary / High speed switch [T-817], each
  option's hover giving its top speed ("High-speed trains run up to 320 km/h on high-speed track
  and 160 km/h on other track"), a car's price and the running costs, and on a running line what
  switching buys or takes from the depot and that the cars running now stay in the depot; then
  **Average speed**, both ways over the round trip less the turning at the ends, so stops and
  waiting at busy track count (hover gives the distance and time) [T-549, T-1159]; then a **Train length** stepper in cars [T-733], 1 up to what the line's
  shortest platform fits (platform / 20 m), each button's hover giving the length it leads to,
  the riders a train then holds at crush ("7 cars, 1,120 riders a train at most"), the cars it
  buys or takes from the depot on a running line, and the cap ("The 160 m platform at East fits
  8 cars"), which the label's hover also gives (the schedule's rows, Trains, Average speed and
  Train length are one run of rows with no rules between, their controls 4 px apart [T-1159]); stepping up to the cap stores the default (the most
  that fits, `cars` 0), so the trains follow the platforms again; when full track holds trains back, an alert under the table names it ("Platform at
  Akabane is full", the station name a link that moves the camera there and selects the station [T-669, T-1393]) with the trains
  held per level beneath it ("Medium demand: 1 train held out of service"), one alert for each
  full place. Alerts are a plain list, each with a red (bad) or yellow (warning) circled
  exclamation point (`Alert` in `ui/widgets.tsx`, `--bad`, `--yellow`), no box [T-748]; a broken
  line's "Track this line ran on is gone" is one too;
  Today for a running line:
  riders by purpose,
  one Profit row (the net over the 24 hours, fares less running cost and the line's share of upkeep,
  in the alert red when negative; the hover gives the net, fares, running trains and upkeep share, T-867, T-1443; the running cost is at the trains and cars the fares' demand result was solved with, so after a train count change both move when the new result lands, T-1492), the **Average load** in a percentage [T-820] (rider-seconds
  aboard over the seconds each trip is under way times the riders a train holds at crush, 160 a car,
  over every trip of today's kind of day from `linePeriods`; `averageLoad` in `LineInspector.tsx`; the hover says so),
  how many trains left full as "x/y" of the trips that
  reached a stop in the 24 hours (the fourth number from the end of each `past_day` block) [T-481]
  and the riders left at platform as "x/y", x those a full train left behind and y those who
  wanted a train (boarded plus left behind) [T-230, T-239, T-293, T-669]; the block's heading is
  "Past 24 hours", with a rule above it and none between its rows
  (the only other rule is above Stops) [T-692]; the stops (above Schedule while being edited [T-670]) with their time from
  the start (its hover: "Trains reach 320 km/h" from the stop before, then what limits them there:
  the trains' top speed, ordinary track, curves or a crossover, and when the stops are too close
  to reach it [T-817]; dropped from a row whose name, time and bullets do not all fit, T-1391) and
  boardings in the past 24 hours, each transfer stop (any station of its complex
  served by another line, or by the line itself at a stop other than the next or the one before,
  so a line coming back through a station shows its own bullet there [T-1391]) showing the two lines with most boardings there in the past
  24 hours as bullets and, when there are more, a "+" whose hover names the rest [T-549]; a **loop**
  shows each station once, its bar going on past the first and last stops and curving round to
  the right at both ends (its last stop, the first again, stays while stops are edited, holding
  "Open the loop") [T-1391]; **flat junctions and
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
  line varies as much as a busy one; a stretch the map marks as crowded (`Renderer.fullStretch`,
  read every half second) has its darkening and red edges 2 px each side, 1 px of the panel's colour
  between them and the stretch (T-1284), the edges from half
  strength at the map's faintest to full (`PANEL_MARK_FLOOR`; on the panel's background the map's
  15% cannot be seen); each stretch's hover: "3 Av to 14 St-Union Sq", "334,700 riders per day", and
  "Average 1.1 minute wait for a train with room" where marked; **related lines** [T-1168,
  notes/T-1168.md, T-1391]: an icon after "Stops" (two lines crossing at a station; hover "Show
  related lines" / "Hide related lines", one setting for every line, on at first and kept while
  the page is open, kept while stops are edited, the stations passed then between the rows) draws
  the stops with the lines related to the line (`TrackApi.line_related`,
  `game/relatedLayout.ts`, `RelatedStops.tsx`): the line straight down the middle of the drawing; each
  line running beside it (shared track, or its track within 100 m + 3.5% of the first 5 km of its own
  run + 2% of the rest of the line's between stations both reach; a line meeting only twice gets 2%
  throughout unless neither line stops between the two; a line whose stops are all on the line's
  route always) in a 3 px column (as thin as a crossing line's bar) 7 px out a step,
  3 px further for one on its own track, right for one stopping more often over the span and left
  otherwise, at most 3 a side and only as many as fit in 40% of the panel; where the other line goes
  on, its column goes straight on 13 px past the line's own first or last row, and elsewhere turns
  out to its side half a row past where it leaves (turns at one gap 6 px apart); where it ends, the
  column ends at its stop; a span over one row only, or a line with no column left, is drawn as
  crossings; beside a loop, a line running on across the loop's first stop goes on past the last
  row; rows are the line's stops (as in the plain list, bullets leaving out the lines drawn there)
  and the side lines' own stops at the nearest point of the line, names muted and nothing else; a
  side line's bullet shows after the crossing lines' at the first and last rows both stop at, and
  at each row both stop at next to one only one of them does (where an express's stops part from a
  local's); a white pill with the dark rim over each run of neighbouring columns stopping at a row
  (10 px at a row with crossings, else 8); the first two crossing lines as 3 px bars across the
  drawing. Hovers: a column ("1 runs beside 2 from Chambers St to 96 St", stops in common, stops of
  its own, "On the same track", its own-track km beside the line's and how far it strays at most), a
  pill ("6, 6 and 6X stop at 23 St", "4 and 5 pass"), a column's turn or straight end ("4 goes on to
  Woodlawn", "14 more stops", or "5 leaves here", "Back beside 2 at 149 St after 10 stops of its
  own"), a bar ("4 crosses at 149 St", and for a line met before with too wide a gap, where it left
  and how far its track strays; "Hook stops at Union Sq twice" for the line's own).
  A click on a column, bullet or bar selects that line; the round trip split into running, dwell, turning and waiting,
  naming where it waits most [T-031]. A **loop** [T-798] shows its lap (one way round, the longer way's)
  in place of the round trip, split the same way, with no turning row for a circle and "Turning at
  the end" for a balloon; its average speed is once round over the lap (a balloon's less its
  turning); its trains steppers move by 2 and the count's hover says how many run each way; the
  stop count under the name counts the first stop once.
- A line's riders count every line actually boarded, transfers included; inter-city trips show
  only in its purpose split [T-182].

## The bus line panel

A bus line's panel (T-1464, ui/inspector/BusInspector.tsx) is the line panel cut down: Edit route
in place of Edit stops (Save route while the route is open, which saves it; T-1514), Copy line (a planned copy) and Delete line (a constructed route refunds
nothing); while planned, its Construct price with the route and the buses in the hover. Schedule:
buses or a headway per demand level, as for trains, and the average speed. Stops: the ends and the
train stations met, the street stops between folded into one row each, minutes from the start.
The past 24 hours: riders, Changing to trains (riders a day changing between this bus and any rail
line at the stations it meets), profit, average load and buses leaving full. No train kind, length,
stop editor or related lines. The Money tab shows buses owned, needed and spare once there are any;
the Build tab's quote has rows for bus routes and new buses.
## Trains

- **The train inspector's "On board"** [T-083, T-164, T-230]: this train's own riders on the stretch
  it is running, from the boarding account ([track-and-service.md](track-and-service.md) 6.6),
  against seats and crush: a bar with a tick at the seats and a state ("Empty", "Seats free",
  "Standing", "Full"), beside the typical train this hour (the mean of its line's trains in the same
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
- **One rider** [T-1436, `ui/inspector/OneRider.tsx`, `game/riderSample.ts`]: under Next stops, while
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
  tab's [T-310], by Name, Riders or Near, Riders by default. Riders follows the ridership choice
  [T-905]: under Origins or Transfers it is the map circle's own figure (riders starting a rail trip,
  or changing lines, a day from the demand solve shown, `game/stationUse.ts`); under None or Total it
  is boardings in the past 24 hours. Its hovers, header and footer say which. The scope
  toggle and distance column are the Lines tab's (above); the footer reads "N of M stations" when
  only the viewed place is listed. The first 80 rows are drawn when the tab opens, and 160 more each
  time the list's end comes within 400 px of view as it scrolls [T-1200]. A row is a station complex [T-530]: the name of its member with
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
  half a device pixel (`checks/t745-bullets.mjs` measures it on the rendered pixels).
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
  - **The complex map** (below), then the complex's "Boardings in past 24 h" with their purpose split [T-1095].
  - **Closest stations** [T-312, T-323; `ui/inspector/ClosestStations.tsx`, `game/walks.ts`]: the
    three other stations with the shortest walk, built or blueprint (blueprint muted and marked),
    each with the chips of its lines (six at most, then "+n") and walking minutes ("7 min", the
    metres on hover). A click selects it (and moves the map if it is off screen). Its own complex
    and stations of its own name are left out, and each name is listed once. The walk is the
    demand model's routed walk ([demand-mode.md](demand-mode.md) 4.3) x the 1.3 detour at the model's 6 km/h; a station
    water cuts off shows "no walk", only when fewer than three can be walked to within 10 km.
  - **Passages**: each passage touching the complex, its length (price on hover) and Remove. The
    button **Trip from here** [T-221] (the trip tool with the station as the start) is in the row
    under the subtitle, above everything else [T-1134], and under it the upgrade row (below).
  - **Commuters**: "Live nearby" and "Work nearby" (commuters a day who live, or work, within its
    reach and ride from it); the larger is picked first, a click picks the other; then the top
    eight far-end places ("Where they work" or "Where they live"). Not drawn on the map [T-519,
    [ui-demand-views.md](ui-demand-views.md)]. Commuters and inter-city trips cover the whole complex, each of the demand
    model's stations counted once.
  - **How they arrive** [T-374; `AccessSplit` in `ui/inspector/StationRiders.tsx`]: under the two
    rows, a bar and key in the purpose split's style for the commuters who live nearby, walked
    (walk icon, the walk colour) against drove (car icon, the drive colour), shares in the key
    and counts in the hovers ("Drove, took a bus or cycled"). They are the model's commutes
    at the home end (`walked` and `rode` of `StationRiders`, from `DemandApi.station_riders_day`);
    everyday trips and the regional layer are not split. Not part of the past 24 hours' purpose split, which
    counts boardings.
  - **Inter-city trips**: rail riders starting and ending there, by place [T-118, T-120].
- **Station view** (one station opened alone) has its elevation under the name ("Elevation +1"), then in place of the
  complex's lines and map the station's own lines, platform settings, and
  **Waiting now** [T-164, T-238]: each line and direction ("To Norwood-205 St") with the riders on the
  platform now in red; under it, in red, how many full trains left riders behind this hour and the
  longest wait, or else the riders arriving this hour and the longest wait. The counts drop the
  moment a train leaves with whoever fitted (a timer set for the station's next departure) and grow
  with arrivals each game minute. The rest (closest stations, links, commuters, inter-city
  trips) is as in complex view, for the station alone.
- **Buttons come first** [T-1134, Anita: never put stuff at the bottom after an arbitrarily long
  list]: every inspector's buttons sit in one row (`Actions` in `ui/widgets.tsx`) right under its
  header, above all rows and lists. A station has Trip from here in one row (what placing a passage
  asks, when picking, is under it) and, in station view, Construct station (a blueprint only) and
  Remove station in a second, then the **upgrade row** under a "Station upgrades" heading
  [T-1281, T-1415, `ui/inspector/StationUpgrades.tsx`]:
  icons for the terminal (two sidings), park and ride (P), waiting room (a bench under a roof) and
  passage (two platforms and a walkway); in complex view only the passage. Each is outlined when
  not built, dashed in the blueprint, filled when built; its hover gives its name, what it does
  in a sentence, its price and what a click does. A click adds it to the blueprint, or on a built
  station builds it at once for its price (a notice says so), or takes it away again (nothing
  refunded once built); the passage's click starts picking the other station (below). A track has Construct for its price (a blueprint
  only) and Remove track; a junction has Construct a flyover (built) or Add a flyover (blueprint), with "Paid at once"
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
  suggested here."

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
"Passages", above the commuters, lists each passage touching the complex with its two stations,
their chips, its length and Remove (nothing refunded once built). Only the player's passages can be
removed: stations the distance rule groups cannot be split. Passages are not in the Ctrl+Z history
(a setting, like fares); removing a station drops its passages. Demand counts a passage once both
stations are constructed.

## The trip explorer

[T-195, T-208, T-221, T-586, T-628; `map/tripView.ts`, `ui/inspector/TripInspector.tsx`,
`game/trip.ts`; the model's side in [demand-views.md](demand-views.md) "The trip explorer"]

The trips between two places as the view it was opened from counts them, homes at the start to
destinations at the end, and how the demand model's people would make one of them. The panel's
title is "Trip" with no line under it naming the ends: the map marks them [T-1253].

- **Starting it**: the T key from any tab [T-444], or Explore a trip at the bottom of Settings, turns on the trip
  tool (a crosshair; a second press or Esc leaves it). Clicks alternate between start and end, a
  third starting a new trip; the hint at the bottom left says which is next.
- **Stations as ends** [T-221]: a click on a station (a hand cursor over one) takes the station's
  own cell as that end (the title's hover then says a station end stands for the homes or places
  right around it). Its own cell, because the model's subzones group cells by nearest station, so
  the trip mostly boards there. A station's **Trip from here** starts it from that station.
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
- **Between zones** [T-866; `ui/inspector/MarketTrip.tsx`, `IntercityApi::market_trip` in
  `sim/src/demand/intercity/market_trip.rs`]: with a bubble selection, a click on an inter-city
  bubble (a whole place zoomed out, a 6 km area up close; hover ends "Click to explore these
  trips") opens the trip between the selection and it; so does a trip tool trip with an end
  outside the focused zone's cells (the cell of the end inside, as homes for the start or
  destinations for the end, to the far 6 km area with trips with it nearest the other end, within
  6 km, else the nearest such place within 60 km; both ends outside: the zones' places whose centres
  are nearest each end). Every other pair of ends [T-885, T-881; the table in notes/T-885.md]: a
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
  station), Effective time (the pairs' felt cost, averaged over the pairs the train serves), the
  main route's share and up to four other routes (routes under 0.5% of the riders are left out,
  but while any rail rider is counted the biggest route stays, so riders spread thin over many
  station pairs still show their trip [T-1500]); **By car** [T-1148]: Drive (15 minutes plus the
  road at 85 km/h), Running costs (each person's dollars), Parking, any Channel crossing or Ferry
  to Hokkaido (each person's fare and its minutes) and Effective time, whose hover is the sum; the
  parts follow from the sim's minutes and mean distance by the rule (`carPartsOf` in
  `game/marketRule.ts`), parking the rest; **By plane** where the pair flies (one far place): the
  drives to and from the airports, 90 minutes at them, the flight, the fare and the effective
  time, from the market rows; Train by period (only when rail is in play). The routes are drawn
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
- **The headline follows the view** [T-586]: under the purpose filter and the day shown, the trips
  a day between the two places (a line "124 trips per day, 2.5 km apart"; its hover the
  exact count and, under All purposes, each purpose's), then a bar and rows of their shares by
  train, walking and driving. Opened from a far-end bubble the places are the selection and every
  cell of that bubble, the same sums as the bubble's hover (`tripMix` over the selection's far end,
  `flows`), so the two agree; from the trip tool, the start's cell as homes and the end's cell. The
  walk row shows when walking the whole way takes under 30 minutes [T-628], or whatever it takes
  when walking has a share [T-992]. When the view counts no trips between them, the headline is the representative
  trip's chances in the period shown.
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
  "Drive to" with its drive minutes; the hovers give real minutes only), the fare, and on a route
  that starts with a drive to the station a **Car** row [T-915] (dollars: the car's cost per trip
  and per km on the road to the station plus any paid parking there, its hover the two apart),
  where full trains add anything a **Crowding** row (the rides' perceived minutes beyond their
  real ones) and a **Waiting for a train with room** row (minutes waited after full trains, T-188;
  T-1070), each with a hover saying how the sim counts it, **Time** (real minutes; its
  hover the sum, "Walk 6.3 + wait 2.1 + ride 10.1 = 19 min") and **Effective time** [T-699, T-549]
  (the minutes the model weighs against the other methods; its hover is only the equation, the
  first route's terms in the rows' order, each real amount times its weight, "Walk 6.3 × 1.47 +
  wait 2.1 + ride 10.1 + crowding 1.0 + change 2 + fare $1.92 × 3 − choice of routes 11.8 = 16
  min", the wait for a train with room its own term after the wait, two
  changes or more as "changes 2 × 2" [T-717]; a drive to the station is its parts [T-915],
  "Starting the drive 15 + drive 9.2 × 2.66 + parking at the station 7 + parking at home 3 + car
  $2.85 × 3", or "drop-off 10" where that is cheaper than parking; the
  last term is what the model's pooled figure, over every station and route people use, is below
  this route's sum, so the equation always adds to the row, and one plain line under the equation
  says choice of routes is the time saved by having several ways to go; the terms are the worker's
  perceived minutes, `routeTerms` in `game/trip.ts`), then one sentence with the share of the
  trip's train users on the route and the same trains from other platforms or walks added ("65% of
  the train users on this trip take this route, the rest use different routes."), with up to four
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
  else the bigger end, "The walk from the nearest station to the end is too long" or "The stations
  near the start are too far, even by car". Its hover gives the two ends' felt minutes and the limit. Every row and bar segment has a hover with its figure and what it is made
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
  route, on top of it, with its stations marked. Hovering a ride gives line, stations, stops, minutes after the wait, crowding's effect
  and the route's share; a walk its distance and minutes; a station what the route does there.
