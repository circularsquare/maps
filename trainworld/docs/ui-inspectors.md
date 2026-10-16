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
cost, trains that left full (with when), riders left behind. Labels say "Past 24 h" or "24 h";
hovers say "in the past 24 hours". They are counted by the clock worker from the boarding
account's departures (`TrackApi.past_day`: the rest of yesterday from the time of day now, on
yesterday's kind of day, then today so far on today's; a stop's boardings in a stretch are the
riders arrived by its last departure in it less those still waiting, less the same at the
departure before it). Fares are each line's hourly fare, the hour now by the share paid as the
cash is [T-284] and the hour a day ago by the rest; running cost is each hour's trains, the hours
at either end in part. Asked four times a wall second
while a panel shows them and the clock runs (`game/today.ts`); only the components showing them
re-render. The map's demand views stay demand's whole-day picture.

**Yesterday** [T-408, `workers/clock.worker.ts` `closeDay`, `yesterday`]: at each line's local
midnight the clock worker keeps that line's whole day (`TrackApi.today` evaluated at the day's
last second, fares paid whole). A game loaded since the day ended has none kept, and shows
the day worked out again on the network as it is now (the account is the same for every day of a
kind). Lines built after the day ended show nothing. Kept days are not saved.

**Riders by purpose** [T-165, `ui/inspector/PurposeSplit.tsx`]: under a line's riders and a
station's (or complex's) boardings in the past 24 hours, a thin stacked bar in the purpose colours, in the
purpose filter's order, with a key of names and rounded shares ("<1%" for a sliver); hovering a
segment or key entry gives the share ("Work: 48% of riders"). Every purpose is listed, those with
none here muted. Under the purpose filter the purpose drawn is ringed in the bar and bold in the
key.

## Lines tab and the line inspector

- **The Lines tab** has New line above its list (actions go above lists of any length). The list
  sorts by clicking a column header [T-310]: Name, Riders (past 24 hours), Trains and Near. A click sorts by that
  column (names A to Z, figures biggest first, Near nearest first), a second reverses it, the sorted
  column has a small arrow, and each header's hover says what it sorts by. Names sort naturally, ties
  by name. Lines default to by name. The sort is kept while the page is open and re-runs each game hour.
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
- **New line** [T-109 to T-112] selects an empty zero-train line, enters stop editing
  ([ui-building.md](ui-building.md)) and focuses its unique `Line N` name, text selected. N follows
  saved creation order, copies included, skipping names in use.
- **The badge**: a click edits its colour (palette swatches or a custom colour) and a persistent
  symbol; several characters widen the rounded badge; a blank symbol restores name-derived
  initials.
- **Actions** above the stop list: Edit stops, Copy line (duplicates route and settings, then opens
  naming and editing), Express copy [T-255] ("<line> express" with stops moved to their twins on
  quad track's express pair, opening stop editing at once; [track-and-service.md](track-and-service.md)
  6.1), and Delete line (asked first). Incomplete lines stay planned and run no trains.
- **Sections**, in order: Schedule (trains for each demand level, with the resulting headway;
  [track-and-service.md](track-and-service.md) 6.3; hovering High, Medium or Low gives that
  level's times of day in the line's local time, from the period table in `game/periods.ts`:
  "High demand: 06:00 to 10:00, 16:00 to 20:00"; weekends use the same hours) [T-406]; Today for a running line:
  riders by purpose,
  fares, running cost so far, its fullest train so far ("100%" when one left packed full, else "N%
  full"), how many trains left full and when (spans under half an hour apart, at the stop where
  each is fullest) and riders left behind [T-230, T-239, T-293]; the block's heading is a Past 24 h /
  Yesterday switch (the Segmented pattern of Settings; the choice is kept while the page is open)
  and Yesterday gives the same figures for the last complete day in the line's local time, running
  cost over its 24 hours at the trains it has now [T-408]; the stops with their time from
  the start and boardings in the past 24 hours; the round trip split into running, dwell, turning and waiting,
  naming where it waits most [T-031].
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

- **The Stations tab** has, above its list, "Station demand circles" and "One circle per station
  complex" (the circles described in the checkbox's hover) [T-213]. The list sorts like the Lines
  tab's [T-310], by Name, 24 h (boardings in the past 24 hours) or Near, boardings by default. The scope
  toggle and distance column are the Lines tab's (above); the footer reads "N of M stations" when
  only the viewed place is listed.
- **A click on a station selects its whole complex** [T-249]: complex view, whatever was selected
  before (there is no Station / Complex switch). A station on its own is a complex of one. Complex
  rules: [ui-map.md](ui-map.md) "Station complexes".
- **Complex view**, under the name and "Complex of 16 stations":
  - **Its lines** [T-246, T-250]: one row per line name, branches and patterns summed, with chip,
    name, riders waiting now (red when one of its platforms is full or its last train left riders
    behind) and boardings in the past 24 hours, busiest by the forecast first so the order holds through the day.
    The row's hover gives line, station and where its trains go; each number has its own hover. A
    click opens that line's busiest station there alone in station view (`openAlone`,
    `singleStation` in `game/complexView.ts`): its own inspector, outline and catchment, its
    subtitle ending with "Complex of 16 stations" as a link back. Any new selection, the same
    station included, shows the complex again.
  - **The complex map** (below), then the complex's boardings in the past 24 hours with their purpose split.
  - **Closest stations** [T-312, T-323; `ui/inspector/ClosestStations.tsx`, `game/walks.ts`]: the
    three other stations with the shortest walk, built or blueprint (blueprint muted and marked),
    each with the chips of its lines (six at most, then "+n") and walking minutes ("7 min", the
    metres on hover). A click selects it (and moves the map if it is off screen). Its own complex
    and stations of its own name are left out, and each name is listed once. The walk is the
    demand model's routed walk ([demand.md](demand.md) 4.3) x the 1.3 detour at the model's 6 km/h; a station
    water cuts off shows "no walk", only when fewer than three can be walked to within 10 km.
  - **Linked by you**, then the buttons **Trip from here** [T-221] (the trip tool with the station
    as the start) and **Link to another station** (below).
  - **Commuters**: "Live nearby" and "Work nearby" (commuters a day who live, or work, within its
    reach and ride from it); the larger is picked first, a click picks the other. Drawn on the map
    as the station's catchment ([ui-demand-views.md](ui-demand-views.md)), with the top far-end
    places listed. Commuters and inter-region trips cover the whole complex, each of the demand
    model's stations counted once.
  - **How they arrive** [T-374; `AccessSplit` in `ui/inspector/StationRiders.tsx`]: under the two
    rows, a bar and key in the purpose split's style for the commuters who live nearby, walked
    (walk icon, the walk colour) against drove (car icon, the drive colour), shares in the key
    and counts in the hovers ("Drove, took a bus or cycled"). They are the model's commutes
    at the home end (`walked` and `rode` of `StationRiders`, from `DemandApi.station_riders_day`);
    everyday trips and the regional layer are not split. Not part of the past 24 hours' purpose split, which
    counts boardings.
  - **Inter-region trips**: rail riders starting and ending there, by place [T-118, T-120].
- **Station view** (one station opened alone) has its level under the name, then in place of the
  complex's lines and map the station's own lines, platform settings, Construct and Remove, and
  **Waiting now** [T-164, T-238]: each line and direction ("To Norwood-205 St") with the riders on the
  platform now in red; under it, in red, how many full trains left riders behind this hour and the
  longest wait, or else the riders arriving this hour and the longest wait. The counts drop the
  moment a train leaves with whoever fitted (a timer set for the station's next departure) and grow
  with arrivals each game minute. The rest (closest stations, links, commuters, inter-region
  trips) is as in complex view, for the station alone. A double click on the name renames it.

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
  direction's half from the middle out by its riders' share of the room, grey or red); its hover
  gives the riders waiting (by line name when more than one stops there), the room and whether the
  last train left riders behind. Trains on their line's band in the darker train shade: one at the
  platform along it, one pulling in (stopping within 60 game seconds) paler, coming in from its end,
  further along the sooner it stops; hover gives line, destination and seconds. Read from the
  renderer's hour of the boarding account; worked out every 10 game seconds, re-rendered only when
  something shown changed.
- **Transfers** [T-176, `complexTransfers(members)`, [demand.md](demand.md) 4.4]: asked of the demand
  workers when the map shows and at each published round, the last answer kept meanwhile. Both
  directions of a line pair added. Changes between lines in different bundles are grey curves (ink
  at 55% over a thin halo of the land colour), across the gap where the two lines' bands come
  nearest, or middle to middle when they cross or come within 16 px: the busiest 8, none under 2%
  of the complex's changes, 1.5 to 8 px by square root. Each curve bows gently to one side; two
  middles at one spot get a small loop. Hover gives the riders a day and each direction. Changes
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

[T-195, T-208, T-221; `map/tripView.ts`, `ui/inspector/TripInspector.tsx`, `game/trip.ts`; the
model's side in [demand.md](demand.md) 4.4 "The trip explorer"]

How the demand model's people would make one trip, read as a commute from a home at the start to
work at the end.

- **Starting it**: the T key from any tab [T-444], or Explore a trip at the bottom of Settings, turns on the trip
  tool (a crosshair; a second press or Esc leaves it). Clicks alternate between start and end, a
  third starting a new trip; the hint at the bottom left says which is next.
- **Stations as ends** [T-221]: a click on a station (a hand cursor over one) takes the station's
  own cell as that end, and the inspector names it ("From a home by Bergen St to work by Grand
  Central-42 St"). Its own cell, because the model's subzones group cells by nearest station, so
  the trip mostly boards there. A station's **Trip from here** starts it from that station.
- **From a selection** [T-208]: with a bubble selection or a station's catchment, a click on a
  far-end bubble opens the trip for that pair (hovers end "Click to explore this trip"), homes to
  destinations. A bubble selection's end is its busiest cell under the purpose filter, the far end
  the clicked bubble's cell with most of the selection's trips; a station's end its catchment cell
  with most riders, the far end the clicked zone's cell with most rail legs. The inspector then has
  a Back button to that selection.
- The trip is a selection: it stays after the tool is left, until Esc or an empty click, and while
  selected panning does not change the focused zone. It is asked again at each published round.
- **The inspector**: the period, the distance, a bar and rows with the chance of train and driving
  (walking too when at least 0.5%). Under "By train" the main route as rows (walk to the first
  station, each line ridden with its chip and the station it goes to, each change and walk, the
  walk from the last station; each row's minutes its wait and ride; a ride access [T-374] reads
  "Drive to" with its drive minutes, its hover adding the ride's fixed cost and the station's
  parking or drop-off), door to door, its drive, walk, wait
  and ride minutes, the fare and the perceived minutes against driving, then the share of the trip's
  train riders taking it, with up to four "Other routes" and their shares. Under "By car" the drive,
  its parking and perceived minutes. "Train by period": a row a period with the train's chance, a
  click showing that period. Without a train it says why (no station in reach, walks too long, no
  service in the period). Every row and bar segment has a hover with its figure and what it is made
  of.
- **On the map**: a veil of the land colour at 60%; the start an open ring and the end a filled
  disc; the main route over the veil in its lines' colours along their track (straight between
  stations only when the drawn network no longer matches the solve's), with a dark casing and the
  boarding and leaving stations bigger; walks dashed and straight; other routes thinner and faint
  under it. Hovering a ride gives line, stations, stops, minutes after the wait, crowding's effect
  and the route's share; a walk its distance and minutes; a station what the route does there.
