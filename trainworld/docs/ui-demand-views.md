[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · **demand views** · [building](ui-building.md)

# 8. Interface: demand on the map

[T-084, T-097; notes/T-084.md, notes/T-097.md; `map/demandViews.ts`, `map/demandLayers.ts`,
`map/commuterView.ts`, `map/bubbles.ts`, `map/intercityView.ts`, `map/demandOverlay.ts`,
`map/demandHover.ts`]

**Two kinds of figure.** Line widths, station circles and bubbles are demand's whole-day forecast
from the latest solve, for the day shown: the average day unless the demand bar's day choice says a
weekday or a weekend day ([ui-panels.md](ui-panels.md) "Demand shown for"); every hover says which. Train
fill and platform crowds are the boarding account ([track-and-service.md](track-and-service.md)
6.6): made from that forecast but counted train by train as the trains run, so they follow the
hour and the clock's own day ([ui-map.md](ui-map.md)).

**Data is asked for, not pushed** ([demand-views.md](demand-views.md)): each view asks again when the demand
result changes and keeps its last answer meanwhile. After an edit only the final round is shown
[T-232], so views rebuild once; at startup, a game load or a city just opened each estimate shows.

**Scope.** Bubbles and selections cover the focused zone's modelled local trips; a station
inspector's commuters are its commuters. Line widths, station totals and train fill include every open zone
and the regional layer. Changing zone clears cell-indexed selections and cached layers.

## Colours

**Mode colours**, Subway Builder's (Anita) [T-097, T-099; `game/palette.ts`]: driving red #e61e1e,
train blue #1e3ce6, walking green #1eaa3c, near-pure so a mix reads as RGB (all car red, all train
blue, half and half magenta). A bubble's mix is pulled towards a grey (190, 190, 190) as its smallest
mode share rises to a third (`modeMix`; fully grey at a third each, unchanged when a mode has none), so an
even three-way mix is not white [T-653]. Bars and swatches by mode keep the true colours.
Trips by "other" (bus, taxi and bike for homes without a car, T-513, [demand-mode.md](demand-mode.md) 4.3)
count as driving in every bar, split, bubble and hover, with no colour of their own; the sim
reports them apart only as a total (`summary` index 10).

**Purpose colours** [T-142, T-143, T-192]: work blue, school and university blue-green,
social/leisure purple, shopping/errands red, airport olive, medical pink (CIEDE2000 22 apart at the
closest pair), the same in the pie, bars, filter swatch and purpose split.

## The demand selector and purpose filter

- **None / Homes / Destinations** [T-132, `ui/DemandSelector.tsx`], always at the map's upper left,
  remembered per browser. Destinations is every activity end (work, shopping, social, school and
  university, airport, medical).
- **Track display** [T-703, `ui/DemandSelector.tsx`]: right below None / Homes / Destinations, in the
  same style, four buttons: Line, Traffic (line colours, thickness by traffic), Elevation and Max
  speed; the key of the last two (height colours, speed limits) sits beside them. Remembered per browser.
- **Day** [T-464, T-703]: a Settings choice under the bubbles' Opacity, not on the map; see "Demand
  shown for" in [ui-panels.md](ui-panels.md).
- **Readiness** [T-143]: while the focused zone has no published result, Homes and Destinations fade
  (their text only), keeping the stored choice, with "Calculating demand" on hover; while its pack
  is still loading the hover says "Loading Boston, 37%" [T-149]. Over dead space they say "Nothing
  modelled here" [T-318].
- **Purpose filter** [T-158, T-192, T-305; `game/purposes.ts`]: a dropdown at the end of the bar:
  All purposes (default), Work, School and university, Shopping and errands, Social and leisure,
  Medical, Airport. School and university are one shown purpose everywhere, though the model keeps
  them apart (`membersOf`). With one chosen, a swatch in its colour sits before the name, the
  control takes the accent border, and every demand drawing counts that purpose only: line widths,
  station circles, the bubbles and their hovers ("12,400 school and university trips per day at
  homes here"), a selection's own and far-end bubbles, the track tag. Scales stay as they are, so
  purposes stay comparable. Work is every zone's commutes (commutes between zones included); the
  others are each zone's everyday layer. Inter-region trips are not an option: they show only
  under All purposes. Train fill and crowding always count everyone.
  A selection's counts and far-end places follow the filter; its pie and timing charts keep every
  purpose. Switching purpose clears the bubbles until that purpose's are ready. Not remembered.

## Line width by riders

[T-227, T-267, T-274] In Line + traffic thickness, each stretch between two stops is as wide as
its riders over the whole day shown, both directions together, linear: 0.5 times the normal width
with no riders up to 6.6 times for the network's busiest stretch-day (`busiestDay`: the purpose
shown, the larger of the days, so a weekend's or the average's lines are thinner). The width does not
change as the periods pass. At zoom 15 the widest line is 44 px. Stations keep their own width
[T-239].

## Station use circles

[T-144, T-146, T-209, T-213, T-482; `game/stationUse.ts`] A choice at the top of the Stations tab,
"Show station ridership": None (default), Origins, Transfers or Total, remembered per browser. Large
translucent circles behind stations, area proportional to riders per day with no cap (the radius
is the zoom's circle radius, 18 to 56 px, times the square root of riders over 140,000, so 140,000
riders have that radius, the real New York save's biggest total (Penn Station's complex, 681,000)
is about 200 px across at zoom 13, and a circle with any riders is never under a tenth of that
radius) [T-722]: **origins** are riders
starting their rail trip at the station, **transfers** riders changing lines there, **total** every
boarding there (origins plus transfers). From the demand solve shown, for the day the demand bar
picks, every purpose whatever the purpose filter: boardings are the solve's per station, transfers
its transfer pairs over every station (`complexTransfers`) counted at the station the rider boards,
origins the rest. Origins and Transfers keep the last split while a new solve's is asked for. The
circle's hover and the station's say the figure and never the number of stations in a complex:
"12,400 entries per day" (origins), "464,000 transfers per day", "476,400 entries and transfers per
day" (total), "per weekday" or "per weekend day" for those days. They do not enlarge the station's click target. With "One circle per station
complex" (on by default, shown while circles are) a complex has one circle at the mean position of
its constructed stations, grouped exactly as the complexes are, the player's links included.

## Local trip bubbles

[T-066, T-097, T-099, T-101]

- **What**: every modelled trip leg at the chosen end (`cellModes`, [demand-views.md](demand-views.md)), the
  zone's res-9 cells summed into their H3 parent by zoom (resolution 9 from zoom 13.6 in, 8 from
  11.8, 7 from 10, 6 further out), each bubble at its trips' weighted centre.
- **Area is trip legs**, outward and return, at a fixed density: 50,000 legs/km² for homes and
  160,000 for destinations, times the Bubble size setting (0.5-2x width). No minimum size; nobody
  means no bubble.
- **Colour mixes the mode colours by share**, at 80% opacity times the Opacity setting (90% by
  default, T-470), biggest bubbles under smaller ones. Bubbles have no outline; only a selected
  one has its dark outline [T-549].
  Drawn over the network and capacity tags, under station names.
- **Hover** [T-323, T-494]: "422 trips per day from here" (Homes; "to here" for
  Destinations, a purpose's name before "trips" under a filter), then a thin **mode bar**
  (`map/modeBar.ts`): train blue, walking green, driving red, widths by share, no numbers. A far-end
  bubble adds "Click to explore this trip". Every demand hover with a mode split uses the bar (the
  airport badges too). A bubble's hover has no split by trip purpose (Anita, 2026-10-19, T-697).
- **Airports** [T-448, T-674, `map/airportMarks.ts`]: a small black plane (14 px, a thin white
  halo) on each airport's busiest terminal cell, so on its biggest destination bubble; no bubble
  of its own. Shown with Destinations under All purposes or Airport, for the focused zone and the
  nearby zones, hidden while a bubble is selected and in the city overview. Airports are the cells
  that receive the airport purpose's legs (500 a day or more), terminal cells within 2.5 km sharing
  one plane. Planes take no clicks (a click reaches the bubble under); the hover, within 8 px of
  the plane (the plane itself, so the job bubbles beside a terminal keep their own hover, T-681),
  says "228k trips per day to and from this airport" with the mode split, the whole airport's
  legs, read again with each demand result.
- There is no corner summary box: the zone's split is in the top bar, local splits in hovers and
  the inspector.

**The bubbles catch up over a game day** [T-156, `map/catchUp.ts`]. Riders, loads, fares and the
split follow a new result at once; the bubbles move to it over about one game day, as if each
person switches on their next trip: each cell's legs by mode are old x (1 - f) + new x f, where f
is the share of a day's trips that have left since the result landed, by the commute's period
profile (36/15/32/10/7% over the five periods, even within each). So they move fastest in the
peaks and are caught up after 24 game hours; paused, nothing moves. A result landing mid catch-up
starts a new one from what is shown; during startup's estimates the refined rounds carry on the
first estimate's. The first result, a change of purpose or zone, and a result landing while the
clock is held show at once. Rebuilt in their worker when f has moved 0.01, at most four times a
second. Inspectors, selections and the top bar are the current forecast at once.

## The focused zone and nearby zones

[T-276, T-318; [zones.md](zones.md) 6; `map/zoneFocus.ts`, `game/zones.ts`]

- After a pan the **focused zone** (the top bar's name, the shaded outside, the bubbles that can be
  selected) is the registered zone, city or not, whose boundary holds the view's centre or lies
  within 40 km of it; before boundaries load, the nearest registry centre within 120 km. Farther is
  dead space: no name, the split "–", nothing loads, and the focus stays on the last zone
  (`placeAt`, `viewedZone`). A bubble, station or trip selection keeps the focus where it is.
  Search picks the zone by the nearest boundary.
- **Nearby zones**: up to three more zones whose boundaries touch the visible map, nearest first,
  show their own bubbles with the same look, end, purpose and day, under the focused zone's, while
  Homes or Destinations is on and above zoom 8.5. Each has its own bubble worker and shows the
  current forecast (no catch-up); their hovers read as the focused zone's. A click on one focuses
  its zone, which takes over that zone's worker and bubbles, and selects the bubble. Selections,
  far ends, the catch-up and the inspector are the focused zone's only. The demand pool opens a
  nearby zone's pack like a viewed city's, but only after the startup load and outside a clock
  hold, with no top bar progress (only the bar over the zone, below); a solve never waits for it.
  A zone no station reaches (the usual nearby zone) opens bare in the auxiliary worker and starts
  no solve; one a station reaches opens in every worker and joins the next solve [T-325,
  [demand-runtime.md](demand-runtime.md) "Bare zones"].
- **A zone still loading is grey** [T-343, `map/zoneLoading.ts`, `zoneLoads` in `game/loading.ts`]:
  every zone fetched whose figures have not arrived (downloading, opening, bare or solved, or
  waiting for its first estimate), a nearby zone not opened yet, and (T-421) a loaded game's zone
  still without its final after the card went at the zone in view's (its figures, if any, estimates
  or the cache's), is grey inside its boundary.
  It is the one grey of "not the zone in view" [T-350]: the shade outside the focused zone
  (`map/cityBoundaries.ts`, `shadePaint`; #182536 at 7.5% in the light theme, black at 35% in the dark one, T-543) is the same colour and opacity, so a loading zone looks
  like an unselected one. Only the zone in view gets the loading fill (over the basemap, under the
  labels and the network, whether or not a demand view is on); every other loading zone already lies
  under the outside shade, and a second fill would darken it. A thin bar sits a little under the zone's centre, kept inside the zone's on-screen part,
  filling as it loads (hover "Loading Toronto, 40%"; the top bar's load for the zones in it, a
  loaded game's zone still solving its solve's rounds and time against its expected final, else
  the zone's own download and opening). A loading zone has no bubble in the zoomed-out overview, and the city inspector says "Calculating
  demand." for it [T-421]. The grey and bar go in one step when the figures arrive. A
  zone re-solving after an edit keeps its figures and is not grey. The bars hide while the startup
  card is up.

## Bubble selections

- **Selecting**: a click selects a bubble, Shift+click adds one, and holding the mouse still 0.4 s
  then dragging draws a box selecting every bubble whose centre is in it. Visible bubbles take
  priority over stations and trains for clicks, hover and starting a box, but not over dragging a
  blueprint point [T-474]. Escape or an empty click
  clears it.
- **On the map**: the other bubbles hide; the selected ones are outlined, and their far end
  (destinations for Homes, homes for Destinations) is drawn as bubbles sized among themselves (the
  biggest about its hexagon's area) and coloured by their own mix (`flows`) [T-098]. A click on a
  far-end bubble opens the trip explorer for the pair [T-208].
- **Lines to the selection** [T-520, on trial: `LINES_TO_SELECTION` in `map/commuterView.ts`
  turns them off]: from each far-end bubble's edge a 1 px line in that bubble's own colour, up to
  50% opaque, points at the selection's centre (its bubbles' centres weighted by area) and fades
  out over 6% of the map's longer side, stopping at the selected bubble's edge when there is one.
  Opacity scales with the source bubble's drawn radius: full at 8 CSS px or more, in proportion
  below, so a bubble too small to see has no visible line [T-524]. A selected zone's lines to the
  other zone bubbles are the same (in each zone bubble's colour, running from that
  bubble toward the selected one). Under every bubble, no hover (`setLines` in
  `map/demandOverlay.ts`).
- **The inspector** (`ui/inspector/CommutersInspector.tsx`): the selection's local trip legs and
  split; the top eight far-end places, each named by the nearest station within 3 km, else by
  distance and direction; for a single area the closest stations from its trip-weighted centre
  [T-312, [ui-inspectors.md](ui-inspectors.md)]; purpose counts and shares for Work, School and
  university, Shopping and errands, Social and leisure, Airport and Medical [T-142, T-192], with a
  purpose pie and legend following All modes / Train only (hover: "Work: 12,400 trip legs per day,
  48%"; a zero-trip scope shows an empty neutral disc); hovers state the scope (matched public
  schools and degree-granting colleges, commercial airports with connecting passengers and staff
  left out, OSM hospitals, clinics, doctors and dentists; private schools are not modelled).
- **Timing** [T-193]: per purpose one compact chart, a bar per period (6-10, 10-16, 16-20, 20-24,
  0-6), departures toward the destination drawn up from a baseline and departures home down, as
  departures an hour on one scale; each bar's hover gives count, hourly rate and share of that
  direction's day. Weekdays only.
- Then the selection's **inter-region trips** (below).

## A selected station

[T-519] Selecting a station leaves the demand views as they were: the bubbles, the zoomed-out
overview and the airport badges stay, and the bubbles are cut away round the station (T-189). Its
commuters and inter-region riders are in its inspector only ([ui-inspectors.md](ui-inspectors.md)).

## Inter-region trips

[T-118, T-120, T-135, T-136, T-139, T-222; the market in [regional.md](regional.md)]

- A bubble selection's and a station's inspector have an "Inter-region trips" section. A selection
  lists trips leaving its areas and trips arriving there, each by place with all modes, then by
  train, car and plane [T-378] (column heads are the modes' icons, `Icon.train`, `Icon.ride`,
  `Icon.plane` in `ui/widgets.tsx`; every number's hover gives the exact count and its share), then
  an "Elsewhere" row for trips with places that have no zone (all modes only; counted in the
  direction's total); its own zone's row reads "Other New York places". A station lists rail
  riders starting and ending there by place.
- On the map a bubble selection's are always drawn (no setting), except under a local purpose
  filter: the far end of its own end (Homes: where the trips go; Destinations: where they come
  from). A station's are not drawn [T-519]. Only zones other than the focused one are drawn;
  same-zone counts stay in the inspector [T-139].
- Above zoom 9.5 every far 6 km zone with trips is a disc at its position, on its own relative
  scale (the busiest covers 0.15 of a zone's area). At 9.5 and below they become one labelled bubble
  per other zone at its registry centre (the largest 19 px radius before the size slider), names and
  daily totals as labels that avoid each other and stay on screen, at most eight at once, the biggest
  first (T-493). Area is total travellers with no
  floor; colour is blue for train and red for the rest (car and plane), mixed by share. They lie under the local discs; hover names the place and the direction,
  with a mode bar (train blue, the rest red, plane included).
- A selection survives a pan to another zone.

## Zoomed far out: one bubble per zone

[T-186, T-276; `OVERVIEW_ZOOM` 8.5 in `map/intercityView.ts`] At zoom 8.5 and below, with Homes or
Destinations on and no bubble selection, every zone in the regional market, the focused
one included, is one bubble at its registry centre in place of the focused zone's own bubbles:
all its trips a day (its local trips, from the open zone's view or the pipeline's no-rail count in
`zones.json`, plus the regional layer's to and from it), coloured by `modeMix` of train, walking
and the rest [T-576: an open zone's walking is green, from its split; a zone not opened has no walk
split, so its walking counts in the red; regional trips have no walking], labelled
"Philadelphia · 187k trips/day" (at most eight labels on screen, the biggest first [T-493]). The hover
has two lines, each with a mode bar [T-494]: "1.6m trips per day within Yamanashi" (bar from
the zone's modelled split, none for a zone not opened) and "120k inter-region trips per day on
average to and from Yamanashi" (train blue, the rest red). One scale that depends on the zoom
alone, never on what is on screen [T-525, T-530]: 60,000 trips a day per px² of bubble area at
zoom 6.5 (`TRIPS_PER_PX2`, `SCALE_ZOOM`), before the size slider, the radius times
2^(0.35 x (zoom - 6.5)) (`ZOOM_GROWTH`), so bubbles shrink somewhat as the map zooms out: New
York's 71M is a 34 px radius at 6.5, 55 px at 8.5 and 21 px at 4.5, a 1.6M prefecture 5 px at
6.5. Homes and Destinations look the same. Under a local purpose the focused zone's own
bubbles stay instead. Toronto (`market: -1`) has none.

A click on one selects the zone (a `city` selection, kept at any zoom): it is outlined, with one
bubble per other zone for the regional trips between the two, on a scale of its own that changes
with the zoom the same way [T-525, T-530]: 1,400 trips a day per px² at zoom 6.5
(`SELECTED_TRIPS_PER_PX2`), so there the selected zone's own inter-region trips (New York's 1.57M)
are a 33 px radius, Philadelphia's 796k 24 px, a 15k partner 3 px and a few hundred a dot (the selected zone's hover: the inter-region line and its bar first, the local figure in brackets under it; the
others: "N trips a day between A and B" and a bar), a 1 px line in the network ink, 30% opaque,
from its bubble's edge to each other bubble's edge, under the bubbles with no hover, so the
selection shows from afar [T-520], and the city
inspector (`ui/inspector/CityInspector.tsx`, T-600) has two figures, each with a thin mode bar
under it (the game's mode colours; every segment and number has a hover with its count and share):
"Within Philadelphia" (train, walking, car from the open zone's split; the count alone for a zone
not opened) and "To other places" (the inter-region trips, both directions, train blue and the rest
red), then the three places with the most trips, name and trips a day, and
a "Go to Philadelphia" button that moves the camera to its centre and clears the selection
[T-220]. No table of every place and no purpose split. The overview catches up over a game day like the local bubbles; a selection's
inter-region bubbles do not.
