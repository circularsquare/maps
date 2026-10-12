[Back to the spec](../SPEC.md)

## 8. Look and UI

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars). The network is not styled by us: lines, trains and stations take the line colours.

- **Fixed layout, not floating windows.** One dock on the left with tabs (build, lines, stations,
  money, city) and an inspector area under the tab content. A top bar for the mode split, money,
  day and time, the demand level ("High demand"), pause and three speeds. At narrow map widths,
  the bar drops secondary labels and uses two rows below
  520 px (three below 300 px), with cash, time, train share and all speed controls available.
  Hovering the split or time retains the hidden mode shares, day and demand level (T-106).
  **The build tools
  (draw track, place station, delete, level, single track) live in the Build tab of the dock**,
  together with the blueprint total, undo/redo and Construct (Anita, 2026-10-10; T-100).
  Those controls stay at the bottom of the Build pane while its settings and cost breakdown
  scroll. The map keeps only tool hints and messages. Confirmations use a centered modal;
  there are no floating inspector windows. The player may
  drag the dock a little wider or narrower and move the divider between the tab area and the
  inspector (Anita, 2026-10-09), both remembered per browser; switching tabs still never moves the
  inspector. "Construct blueprints" builds every blueprint. While drawing, the tooltip gives the
  stretch's length, average cost multiplier and price ("1.5 km, cost 1.33x, $25.5M") instead of
  explanatory notes in the panel. A click where a train is drawn over a station selects the train.
  (The T-009 mock,
  `mock/`, is the approved starting point; `../../artifacts/trainworld/T-027.png` is the shell as built.)
- **Trains show who is on board** (T-083): the train inspector shows riders aboard now and its load
  against seats and crush (a bar with a tick at the seats; "Seats free", "Standing", "Over
  full"), from demand's riders on the segment it is running, divided among the trains the line
  runs in that period. Both the average and busiest-hour counts are listed; the gauge uses the
  Average / Busiest hour display choice, busiest by default (T-087).
- **Close-up trains** (T-103, T-107): from zoom 15, trains show the line's actual number of 20 m cars,
  each positioned on the route and angled along its own chord, so the consist snakes around
  curves. Cars have small gaps, rounded rectangular bodies, and windows/front cab when large
  enough. The whole consist stays on the route at terminals. Clicking any car selects the train.
  Cars and compact trains share the same zoom-dependent width, growing with physical width at
  very close zooms, so the switch never shrinks the train. Further out, trains use a compact
  capsule with length based on their car count; all positions
  remain GPU-derived from keyframes, with one instanced draw call for trains at either scale.
- **The inspector shows whatever is selected**: a line, a station, a stretch of track, a junction,
  a train, commuter bubbles, a cell or zone. Nothing selected means it is blank (no text at all), not a default line. **Its top edge
  never moves when the tab changes**: the tab content area has a fixed height and scrolls.
- **City pane** (T-143): follows the viewed city, with City centre and Whole region navigation.
  It shows that city's external intercity trips and the five largest connections ranked by
  all-mode trips, with compact 5 px blue train/red other bars and train/all trip counts such as
  `500/24k trips`. Full counts and percentages are in the row tooltip. Counts combine both
  directions and are labelled To and from. Same-city longer trips stay in selection inspectors.
  These are prototype estimates, not measured OD. The top bar follows the nearer full city when
  panning and shows its modelled local trip share. Line inspectors split daily boardings into
  Local trips and Longer trips; transfers count on every line actually boarded.
- **Confirmations** (T-113): a centered Are you sure? dialog with operation-specific text,
  cancel and action buttons. Cancel receives initial focus, Tab/Shift+Tab stay in the dialog,
  Escape cancels, focus returns when dismissed, and background map/tools are blocked.
- **Line management** (T-109 to T-112): the Lines pane has New line and Name/Passengers sorting
  above its list. Names sort naturally; passengers sort by daily boardings descending. Actions
  belong above lists of arbitrary length. New line immediately selects an empty zero-train line
  in the inspector, enters Edit stops and focuses its unique `Line N` name with all text selected.
  The number follows persisted creation order, including copies, and skips names already used.
  Copy line duplicates the route and all settings independently, then opens naming/editing.
  Clicking the inspector badge edits colour and a persistent symbol; multiple characters widen
  the rounded badge, with its text raised one pixel. Blank symbol restores name-derived initials.
  Edit stops, Copy line and Delete line sit above the stop list. Red removal crosses appear only
  while editing, including on one/two-stop lines. Delete line requires confirmation. Map clicks
  test every insertion position and choose the shortest valid route while keeping existing stop
  order; Enter or Escape finishes. Incomplete lines remain planned and do not run trains.
  The inspector uses Finances and boardings per day, without the redundant schedule sentence.
- **Demand readiness** (T-143): Homes/Destinations fade while the focused city has no published
  result, preserving the stored selection and its colour, with Calculating demand on hover.
  None remains available; ordinary recomputes retain the prior result and usable controls.
- **Display settings**: the commuter bubbles (off, homes, jobs), longer-trip bubbles on or off,
  the four track display modes,
  and show or
  hide trains, station labels, capacity markers, basemap labels, the riders drawn on the network
  (line width, station demand circles, how full trains are) and other layers as they appear.
  They live with the theme switch in a settings pane behind a gear button at the end of the tab
  row, shown in the tab area like a tab, so the inspector stays put. Remembered per browser. The
  same pane has the game's saves (section 11).
- **Track display** (T-105): four mutually exclusive choices: Line (uniform width), Line with
  traffic thickness (default, riders in the current period), Height, and Max speed. Height and
  Max speed colour the track itself with uniform width; trains keep their line colours. Max speed
  is the geometry's allowed speed at each section, including curve limits, capped at 160 km/h;
  it is not a train's instantaneous speed. A 40/80/120/160 km/h colour key runs from warm red
  through ochre and green to blue. Existing Line/width-checkbox preferences migrate to the
  corresponding mode, and the choice remains remembered per browser.
- **Height colours** (track coloured by level): warm above ground, grey at ground, cool below:
  +3 #c2410c, +2 #ea7a1a, +1 #f2b134, 0 #8c8c8c, -1 #5aa3d8, -2 #2f6fbf, -3 #23408e. Trains
  keep their line colour.
- **Picking on the map** follows visible drawing order: demand discs above trains, then stations,
  lines, junctions and track. Within a layer the last drawn hit wins, including overlapping
  bubbles; hidden layers do not intercept clicks. Network hits allow 7 px. Clicking a selection's
  destination or station catchment keeps its parent selection. Clicking the line already selected
  selects the track under it. Selected editing handles remain draggable above the demand view.
  An empty click or Escape clears the selection.
- **Tools** in the Build tab: draw track (key 1), station (2), delete (3), the level with
  its price, double or single track, platform length for new stations. The number keys work from
  any tab and open the Build tab (the level and platform settings are there); the key of the tool
  already on does nothing. Each button shows its key. **Delete** (T-096): hovering shows what a
  click removes (a stretch of track drawn red, a station or a flyover ringed, its name by the
  cursor); blueprint is removed at once (undoable), and anything constructed is removed only after
  a yes in the centered confirmation dialog ("Remove 1.2 km of constructed track? Nothing is
  refunded." with Remove and Keep; Esc is Keep). The inspectors' Remove buttons ask the same way.
  Lines are removed from the line panel. **Esc** undoes one thing at a time: a question waiting is
  answered Keep, a drag goes back, a half-drawn route is dropped (the tool stays on), then the tool
  is left for Select (drawing, stations, delete, making a line), then the selection clears. With
  select, the squares of a selected blueprint
  stretch (its corners) and any blueprint track end, junction or station can be dragged (6.1);
  the track inspector adds/removes blueprint corners and sets their individual levels (T-071).
  the cursor turns to a move cursor over a node that can be; a line is made from the Lines tab
  (New line, name it, then click stations) and its stops edited from the line panel. Ctrl+Z undoes,
  Ctrl+Y or Ctrl+Shift+Z redoes. There is no Select button: clicking the active tool again or
  leaving it with Esc returns to selecting. While drawing track, its placement ghost is solid
  with a dark casing: white in Line/Traffic views, or coloured by the selected Height/Max speed
  view using the preview's fitted geometry. An invalid ghost is red. Once placed, blueprint track
  is drawn dashed and faded. Build has the
  total and "Construct blueprints", including the new train cars that constructing starts,
  after spare cars. Its expandable cost breakdown shows track by priced level, single/double,
  land/water and ramps (charged at the dearer end), stations, junction switches, flyovers and
  flat crossings, with lengths/counts and prices from the track model. The fleet quote previews
  the constructed service, so starting several lines together includes their resulting delays
  without changing the game. Tool hints and a refused edit's reason show at the map's bottom left.
- **A new game** starts on day 1 at 07:00 in an empty New York with $6B. Opening the page
  continues the last game (section 11).
- **Money tab**: cash; yesterday and today (fares, running trains, new trains, construction,
  net); the fare curve (each ride, each km); a day at the current schedule; cars owned and needed.
  The line inspector shows a running line's fares and running cost a day.
- **Exact metropolitan focus and boundaries** (T-131, T-139): each registered pack's dissolved
  Census county footprint has a thin solid outline at regional zooms, fading between zoom 8.5
  and 10.5. The focused city retains a thin solid boundary at closer zooms, with a darker outside
  stroke and a lighter inside stroke. A subtle shade darkens the basemap outside its exact
  footprint, including excluded enclaves, leaving islands inside the city clear. These are
  metropolitan demand areas, not municipal limits. The layers sit beneath labels and game
  overlays, never intercept clicks, follow city focus and survive basemap style changes.
- **The goal number is always visible.** In a city: its mode split as one bar (e.g. driving 80%,
  train 16%, walking 4%); the player's job is to make the train share grow. Subway Builder does
  this well. It is the split of all modelled local trip legs (commutes plus NYC everyday outings),
  shown at 60% opacity while a newer network is being solved. Its three colours are the mode
  colours below, whatever the theme. Zoomed out to the world: one bubble per city or town for
  long-distance travel, sized by trips, showing rail's share.
- **Demand views** (Anita's pick, 2026-10-09; T-084, notes/T-084.md). These are steady-state
  forecasts from the latest demand solve, not observed passenger counters that advance with the
  animated trains. Line widths are selected by Track display; station size and train fill have
  checkboxes under "Demand forecast on the network":
  - **Line width by riders**: each stretch between two stops is as wide as its riders an hour in
    the period in force on the clock, both directions together: 0.45 of the normal width when
    empty, growing with the square root to 2.2 times at 30,000 an hour. Lines sharing track stay
    side by side at their own widths, trains ride their own stroke, and a station's dot spans the
    widened bundle. Only drawn in Line with traffic thickness mode (T-105).
  - **Station demand circles**: an explicit analytical setting, off by default, draws large
    translucent circles behind stations. Circle area grows with boardings plus alightings a day.
    The ordinary station stays a track-aligned platform box whose length follows the platform,
    with a readable screen minimum and a larger selected state. Blueprint boxes are faded and
    constructed boxes opaque. Demand circles do not enlarge the station's click target; selection,
    deletion, line-stop editing and the placement ghost use the visible box geometry (T-144).
  - **Train load forecast**: a train's core fills from its back like a gauge, in the dark shade
    of its line colour over a slight tint of it: filled is 48% of the line RGB, empty mixes only
    18% white into the line colour (T-103). The gauge spans the whole consist across car gaps;
    half full means every seat taken, full means packed
    to crush, and an over-full train is full with a red rim. The numbers are the train inspector's
    on the stretch it is running. The animated train's position selects the stretch; its riders
    are the steady-state forecast for that stretch. Settings choose Average or Busiest hour (the default), using
    the period's busiest-hour factor from the demand model. The train inspector lists both
    counts, and its gauge uses the same choice (T-087).

  **Local trip bubbles**, the main demand view (T-097/T-066, Anita's Subway Builder style view, notes/
  T-097.md): the always-visible selector at the map's upper left offers None, Homes or Destinations
  (T-132). Destinations includes work and NYC shopping/errands and social/leisure activity ends.
  The choice is remembered per browser; bubble size stays in settings. Every modelled trip leg at
  that end of the trip is in a bubble (`cellModes`, 4.4): the city's res-9 cells summed into their
  H3 parent by zoom (resolution 9 from zoom 13.6 in, 8 from 11.8, 7 from 10, 6 further out, about
  1.4 zoom levels a step, so bubbles keep about the same spacing on screen), each bubble at its
  trips' weighted centre. **Area is trip legs**, both outward and return, at a fixed density,
  50,000 legs/km² for homes and 160,000 for destinations, preserving the former commute bubble
  area when converting one commuter to two legs; the same at every level, times the "Bubble
  size" slider in settings (0.5 to 2 times the width, 1 by default, remembered per browser,
  T-099); no minimum size,
  and nobody means no bubble. **Colour mixes the three mode colours by share** (RGB: all car red,
  all train blue, half and half magenta), at 80% opacity, biggest bubbles under smaller ones.
  The bubbles are drawn over the network and its capacity markers, under station names.
  There is no corner summary box: the city mode split is in the top bar, and local splits are
  shown on hover and in the inspector (T-101, Anita).

  Hovering a bubble shows local trip legs there, both directions included, and their mode split.
  **A click on a bubble selects it**; a click
  with Shift adds it; **holding the mouse still for 0.4 s and dragging** draws a box and selects
  every bubble whose centre is in it. Visible bubbles take priority over stations and trains
  underneath, for clicks, hover and starting a box selection. A selection hides the other bubbles and shows the selected ones outlined and
  their destinations (or homes) as bubbles sized among themselves (the biggest about its
  hexagon's area) and coloured by their own mix (`flows`, T-098). The inspector shows the
  selection's local trip legs and split, and the top eight far-end places, each named by the station
  nearest it within 3 km, else by distance and direction. Escape or an empty click clears it.
  Purpose inspection shows daily counts/shares for Work and, in NYC, Shopping and errands and
  Social and leisure, plus School (metro public). A scope note identifies matched public schools
  across the NYC metro and estimated pupil homes; private schools and universities are not modelled.
  A purpose-share pie and labelled legend follow All modes / Train only. Work is blue, school blue-green,
  social/leisure purple and shopping/errands red, with matching departure-bar colours (T-143).
  A zero-trip scope shows an empty neutral disc. All modes / Train only changes this breakdown's scope. Separate To
  destination / Return home charts show actual period-specific departure counts per hour;
  tooltips include counts, hourly intensity and direction shares. Purpose sums match the bubble
  total for the chosen mode scope; unmodelled facility purposes are explicitly identified.

  **A selected station's commuters**: the station inspector lists commuters a day who live within
  its reach and ride from it ("Live nearby") and who work within its reach ("Work nearby"); the
  larger is picked first, a click picks the other. On the map, over the network, the chosen end's
  cells are opaque discs on top, the busiest a cell's area, and the places at the far end (up to
  20 gravity zones) filled discs at 55% opacity below them, the busiest half a zone wide; no
  minimum size; hovering either shows its riders. The inspector lists the top far-end places. A
  selected station hides the commuter bubbles.

  **Longer trips** (T-118; sim/src/demand/intercity.rs, SPEC 5.2) have a "Longer trips" section
  in the inspector of a bubble selection and of a station, and a "Longer trips" check in the
  settings pane (on by default) for their bubbles. A bubble selection lists trips that leave from
  its areas (homes first) and trips that arrive there, each by place with all modes and by train
  columns; the own region's row reads "Other New York region places" because the market has no trip
  under 40 km. On the map the selection's own end is drawn at the far end: for Homes, where the
  trips go; for Destinations, where they come from. A station lists rail riders who start and end there,
  by place, and draws where the ones starting there go (T-120). At every zoom, longer-trip bubbles
  are limited to markets outside the focused metropolitan area; same-city longer-trip counts
  remain in the inspector (T-139). Above zoom 9.5, every positive far
  6 km zone in those other markets is shown at its actual position. Longer trips have their own relative scale: the
  busiest disc covers 0.15 of a zone's area; every disc's area is proportional to total travellers.
  Smaller counts are no longer dropped. At zoom 9.5 and below, exact place totals become one
  labeled bubble per other city or town, anchored at the registered city centre (towns use their
  mean endpoint). The largest other-city circle has a 19 px radius before the size slider;
  area is proportional to total travellers with no radius floor or additive offset (T-136).
  Labels show names and daily totals, avoid each other and stay
  inside the viewport. Both modes use commuter opacity and the bubble size slider, blue for the
  everyone taking the train, red for nobody taking it, and the normal blended demand colour for
  the share in between (a station's riders are all blue). They lie under the
  commuter discs; hover names the city in the overview and states whether trips go to it or come
  from it. For Homes the map shows destinations; for Destinations it shows origins. The setting hides only the
  bubbles; the inspector keeps the numbers.
  The selection does not end when the map pans to another city (focus stays while a selection or
  a station is selected).

  Bubbles and selections cover the focused city's modelled local trips; station catchments still
  count its commuters explicitly. Changing
  city clears old cell-indexed selections and cached layers. Network widths, station totals and
  train fill include all registered cities and longer trips.
  Data is asked for, not pushed (4.4): each view asks again when the demand result changes and
  keeps showing its last answer meanwhile. Bubbles are rebuilt for a solve's first estimate and
  again when its crowding rounds are done.
- **Mode colours**, Subway Builder's (Anita, T-097), fixed and hand-tunable in `game/palette.ts`:
  driving red #e61e1e, train blue #1e3ce6, walking green #1eaa3c, near-pure so that a mix reads as
  RGB (Anita kept the mix, T-099). A selected
  station's own cells #c2185b, its far end #3a3f8f.
- **Square buttons**, 1 px borders, small line icons as inline SVG.
- **Few colour tokens**: background, text, one accent, as CSS variables so other themes are a
  token swap. Light and soft, no paper texture or "ink" statement look (Anita: too statementy).
  **Three themes, all kept**: pink (cream background, sakura accent), blue (cornflower), green
  (matcha); a dark theme later.
- No decoration that does nothing: section headings are plain text (the mock's small coloured
  squares before headings are dropped).
- **Line colours are a normal transit palette, fairly saturated** (strong reds, blues, greens,
  oranges, purples, as on real metro maps), not pastel. New lines take the next default; the player
  can set any line's colour in its inspector using palette swatches or a custom colour.
- **Zen Maru Gothic** (Anita's pick over M PLUS Rounded 1c) everywhere, including station names
  and the map's DOM controls. The map container explicitly inherits the project font rather
  than MapLibre's default Helvetica (T-101). Form controls inherit it too. The font on
  the basemap's labels is Regular only, faint grey, from glyph tiles we generate and host beside
  the app (`app/public/fonts/`, `pipeline/glyphs.py`), with Noto Sans merged in for scripts Zen
  Maru lacks. Kana and kanji on the basemap are drawn by the browser in the system font for now
  (T-057). notes/T-056.md. Few font sizes and weights (the mock's 3 sizes and 2 weights is the
  ceiling).
- Instant UI state changes, no CSS transitions. Trains moving is the only motion.
- **Space bar** pauses, and resumes at the speed that was running before the pause; it, the
  number keys and Esc are ignored while typing in a field.
- **Trains** are capsules further out and separate cars from zoom 15, in a darker shade of the
  line colour with a shared width at the transition (T-107) and a white rim blended into the
  core over a pixel, every edge antialiased in the shader, and turned along the chord between
  their two ends rather than the track sample under their centre (notes/T-047.md). With riders
  known the core is a gauge of how full the train is (demand views, above).
- **Lines sharing track are drawn side by side**, touching, each at its own width, like a metro map: each
  keeps its place along a shared stretch, a branch line sits on the side it leaves to, trains
  ride their own line's stroke, and a station's dot spans the lines through it (notes/T-062.md).
- Player-facing text follows the copy rules in Anita's global CLAUDE.md (no dash-joined phrases,
  sentence case, no softeners, no hype).
