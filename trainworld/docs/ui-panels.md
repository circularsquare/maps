[Back to the spec](../SPEC.md) · Interface: **panels** · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: look, dock, top bar and panels

## Look

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars); the network takes the line colours ([ui-map.md](ui-map.md)).

- **Nunito** (Zen Maru Gothic for kana and kanji, T-703) everywhere, station names, map DOM controls and form controls included (the
  map container inherits it rather than MapLibre's Helvetica) [T-101]. Few font sizes and weights:
  3 sizes and 2 weights at most (the T-009 mock's, `mock/`). The basemap's labels are Zen Maru
  Gothic, Regular only, faint grey, from glyph tiles hosted beside the app (`app/public/fonts/`, `pipeline/glyphs.py`,
  Noto Sans merged in for other scripts); kana and kanji are drawn in the system font for now
  [T-057, notes/T-056.md].
- **Few colour tokens**: background, text, one accent, plus `--bad`, `--warn` and `--yellow` for warnings, as
  CSS variables (`ui/style.css`); everything else is mixed from them, so a theme is a token swap.
  No paper texture. **Light or dark** [T-039], each with **three colours**: pink (cream, sakura
  accent), blue (cornflower), green (matcha); in dark the background is a near-black tinted towards
  the colour and the accent a little lighter. The map's own colours (`--map-land`, `--map-ink` for
  the map's tags and names, the basemap, the line casing, a selected train's halo, the shade outside
  the focused city) come from one table, `game/shades.ts`, and the stop editing markers' greys are
  CSS tokens (`--mk-*`);
  TS code that draws on the map takes its colours from there, never from `if dark` checks. Line,
  mode, level, speed and purpose colours are the same in both themes (`game/palette.ts`).
  Switching is instant and redraws only: no demand solve, no pack or style reload.
- **Square buttons**, 1 px borders, small line icons as inline SVG (`ui/widgets.tsx`). No
  decoration that does nothing: section headings are plain text.
- **No rules within a section** (Anita) [T-1237]: rows in a list, the ledger, the Cities total,
  purpose blocks, leaderboard rows and a slot's backups are spaced, never ruled: rows of plain text are 24 px high (the ledger's, the Lines, Stations and
  Cities lists'), rows holding a stepper, button or box 28 px [T-1802].
  A rule only separates sections about different things (the line panel's `.h.divided`, the
  inspector's top edge); a list's column headers keep their thin rule over the rows.
- **Spacing comes from the container** [T-3768]: Settings (and the start menu's settings) is one column with
  an 8 px gap between rows, a heading 14 px under the row above and 4 px over its first row; a row brings no
  margin of its own. `checks/t3768-panel-gaps.mjs` fails when two controls or lines of text in the main panels
  come within 3 px of each other, at 1400 and 390 px, unless they are one control (a Segmented, a stepper), and when
  one reaches past the panel's side or a label is cut to almost nothing; the one pair it lets through is a 28 px
  row of buttons right under a stepper row (2 px, from the row heights above). Where a heading's right part or a
  row's buttons do not fit beside their text, they go under it.
- **Motion where it helps** (Anita 2026-10-29, dropping the old "no motion but trains" rule): state
  changes stay quick and no control waits on an animation. Camera moves the game makes (search, Go to)
  jump with no flight [T-182]; the map's own gestures keep MapLibre's easing.
- **Panels show what can change** (Anita) [T-253]: labels, numbers and states. No explanatory
  notes or how-it-works sentences; a fact worth keeping goes in the hover of the label it explains.
  Icons with hovers over word labels where the icon is clear.
- **Live feel** (Anita, 2026-10-13): a figure that would change often if every train and passenger
  were simulated (money, riders, waiting counts) moves smoothly, not once an hour.
- Player-facing text follows the copy rules in Anita's global CLAUDE.md (no dash-joined phrases,
  sentence case, no softeners, no hype).

## Hover tags

- **One tag, never the browser's tooltip** [T-210, `ui/tip.ts`]: every hover outside the map
  canvas (panels, top bar, buttons, the complex and junction maps, DOM markers) is the map tags'
  look, shown at once under the pointer, below right of it and flipped to stay in the window; a
  press, scroll, key (not a modifier alone) or the pointer leaving the page hides it. When the
  window loses focus (Win+Shift+S for a screenshot) every hover, the map's too, stays up until the
  pointer next moves over the page, leaving ignored till then [T-483, `ui/blurHold.ts`]. Text lives in `data-tip` (SVG shapes
  too); no `title` attribute is written anywhere, and one met under the pointer (MapLibre's
  controls) is turned into the tag first. The tag is a manual popover, shown again each time it
  appears, so it sits over a modal dialog such as the saved games panel [T-1656].
  Besides text, a hover's line can be a chip (`tipBullet`; with no letter a plain 8 px square of its colour), a
  thin strip of colour cells (`tipStrip`; equal cells, or as wide as their weights for a split [T-2365]) or a small
  table (`tipGrid`, T-2365): the first column left, the rest right, a cell can span columns (`tipSpan`), a strip in
  it at least 130 px wide, a row of one empty cell a 4 px gap.
- **Every drawn quantity has a tooltip** (Anita) [T-194, notes/T-194.md]: bubbles,
  inter-city and city bubbles, line widths and track colours, trains, stations, crowds, demand
  circles, capacity tags, and in the panels the purpose split, pie and timing bars, the trip
  explorer's bars, the complex map's platforms, the junction map's routes and the
  train's load bar and speed dial.

## The dock

**Fixed layout, no floating windows** (the saved games panel, confirmations and the
[tutorial card](#tutorial-card) are the exceptions). One dock on the left [T-252, `ui/Dock.tsx`]:

- A row of icon tabs, names and keys in their hovers ("Build (B)"): **Build** (a wrench, T-2415), **Lines** (a route with stops),
  **Stations** (a platform under its canopy), **Finances** (a bill and a coin) [T-1819], **Cities** (a skyline) [T-749], sharing the row's width
  equally [T-532] (each at least 34 px); at the row's right end
  the saved games button (a floppy disk) [T-251] and the **Settings** gear (a toothed wheel outline with a hub circle) [T-523], which opens in the tab
  area like a tab. There is no City tab [T-220]; a browser that last had it open starts on Lines.
  A new game that shows the tutorial card opens on Build, where step 1's Track tool is [T-2092,
  `guideNewGame` in `game/guide.ts`, set by `main.ts`]; a saved game, or a new game with no card
  (a sandbox, a `?new=1` page, a browser that finished the guide), opens on the tab the browser last had.
- Under the tab content, the **inspector** ([ui-inspectors.md](ui-inspectors.md)). The tab area has
  a fixed height and scrolls, so the inspector's top edge never moves when the tab changes. By
  default it is 36% of the dock but at least 440 px (the inspector keeps 160 px), which holds the
  Build tab's tools and blueprint whole; only a blueprint's cost breakdown, the lists and Money and
  Settings scroll [T-1128].
- The player can drag the dock wider or narrower and move the divider between tab area and
  inspector; both are remembered per browser. Each tab keeps its own scroll position for the
  session, opening at its top the first time [T-215].
- The **tool notes** at the map's bottom left, 10 px in (`ui/ToolNotes.tsx`): first the click mode's
  descriptor ([ui-building.md](ui-building.md#click-modes-the-aura-and-the-descriptor)), then the stop
  editor's and trip tool's one-line hints (the build tools have none, T-1159), then a refused edit's
  reason or what an edit did.
- **Narrow screens** [T-951, T-1259, `style.css`]: at most 600 px wide and at least 560 px tall (a
  phone held upright) the dock goes under the map, full width and 40% of the height, so the map and
  top bar keep the whole width and the map column's container queries see the real width. A strip
  above the tab row (which has a rule on its top edge like the ones between and below its tabs, T-2549), a short line in the rule colour, drags the dock taller or shorter with a finger
  or the mouse, from the tab row and about two rows of the tab up to 58% of the screen (the map
  under the top bar keeps about 30%), remembered per browser [T-1884]. Under the tab row the tab
  area takes the dock's whole height. On every tab the inspector stands beside the tab area instead
  of under it, empty when nothing is selected [T-1852, T-1885], the tab area 40% of the width and
  the inspector the rest until the divider between them is dragged (30% to 70%, remembered per
  browser); a tab wider than its share scrolls sideways. At that width the Build tab's tools show
  their icons only, its button rows share the width, the Lines list drops its Near column, and the
  Money tab's ledger and fare rows put each name on its own line with the figures or stepper under
  it. The width handle is gone
  and the help panel covers the dock's width. Nothing changes above 600 px wide or on a short
  (landscape) screen. Every key hint is in the theme's accent (the letter on a button, the key in brackets in a hover, the Help panel's key column) [T-1835, T-1852]; on buttons and in hovers it is at 0.7 opacity, brackets and all, as a hint (the Help panel's list stays full) [T-2157].
  On a touch screen the buttons show no key hints, and a press held on a control
  for half a second shows its hover tag above the finger, the tap it ends in doing nothing [T-1260,
  `ui/tip.ts`]. The Select tool's selection bar leaves the top of the map for its bottom edge
  [T-1263, [ui-building.md](ui-building.md) "The Select tool"].

## The top bar

`ui/TopBar.tsx`, left to right, 44 px high, **every part centred on one line** [T-349, T-1853]: the
name, the split (the bar, then the train share), cash, the clock (the time and date), speed
buttons; the day and night picture is on the map, not in the bar [T-1868]. **The name, train share, cash and time are one size, 16 px, and one weight, medium (500)**
(Anita: equally important numbers, T-1853; one font style, T-1865); the date is 14 px regular and the
status small (12 px, regular, muted). **The clock and speed buttons have fixed widths**
[T-341, T-1136] and sit together at the right, so nothing moves them. The cash follows the train
share (after the name it would read as the city's, T-1865) in a box of fixed width, 4.7em (75 px, tabular
figures: it holds the widest figure written, "−$893.6M", and the usual 7 characters leave a little room),
so the mode bar (at least 60 px), which widens into the rest, does not change width as the money changes. The name takes what it needs up to 140 px (124 px under 760 px
of map column; "Ho Chi Minh City" is 122 px and fits on one line in both, T-2271), on two lines at most with an ellipsis (the whole name on hover, with the city's
points and points a day, T-684, T-694) [T-1108]. The train share is in tabular figures in a box of
fixed width, and the split bar is centred in the bar with nothing under it: the status is a tag
over the map (Status, below). The time and date are one line ("06:50 Mon May 5") [T-1136,
T-1853, Anita's sketch] behind a thin divider (the faint rule colour) after the cash, in a box 132 px
wide so a longer date moves nothing [T-1868].

### The goal number

- **The goal number**: the focused zone's mode split as one bar (driving, train, walking, in the
  mode colours whatever the theme), the split of all its trip legs within it, as the city view's Trips
  within (its modelled local legs: commutes, everyday outings, school and university; and the
  market's longer trips and long commutes between its own far-apart parts, T-3509), so the shares
  are of trips per day, not people [T-342]. The Distance row's km are the local legs' alone.
  The train share follows the bar ("18.5%"; no word "Train" [T-650] and no rides count [T-832]). The
  bar's hover [T-1107, T-2365, `splitTable` in `game/splitTip.ts`; the city view's Train share row has the same] is a
  title, "Per day in New York", then a small table: **Trip count**, a strip of the modes as wide as their trips and
  the total ("68.2M"); **Distance**, the same by the trips' straight-line km ("516.5M km", from the zone's trips by
  length, asked for while the pointer is on the bar and left out until they come); then heads "Train", "Walking",
  "Driving" (and "No route" where its share shows as 0.1% or more), each with a square of its colour, over the
  shares of the trip count and of the distance. The bar, its number, the goal ticks and the star stay on the share
  of trips (T-2364). **Transit by mode** [T-3695, `game/transitModes.ts`]: with Settings' "Show transit by mode" (off
  by default) every such bar (this one and the World's, the Cities tab's, the city view's two, the far-zoomed city's
  hover, both trip explorers') draws its transit part as ferry, tram, train, high speed and bus (high speed: a line
  of high-speed trains, T-817), teal to blueish purple, the same in both themes and every bubble scheme; the hovers
  add a line of their shares ("Train 15.3%, High speed 2.3%"), and the trip count's strip splits likewise (the
  distance's does not: no km by kind). A zone's split is its stops' boardings by kind of line on the day shown, so a
  trip changing kinds counts once on each (by leg count); trips within it count every purpose but the longer trips
  between places, its trips to other places those alone; a trip explorer shares its routes by ride length. Where
  no boarding is counted the bar keeps one transit part, as do a pair of zones, bubbles and their hovers, airport and
  attraction hovers and the trips-by-length graph. The demand bubbles keep three colours. The player's job is
  to make the train share grow. It shows "–" and an empty bar until the zone has figures of its own
  (a solve's, or the bare pass's for a zone no station reaches; not the no-rail split a pack gives
  before that), and over dead space [T-318]. The hover says "on a weekday", "on a Saturday", "on a Sunday" or "on average" for the
  day shown. The zone named is the focused zone ([ui-demand-views.md](ui-demand-views.md)).
  **The zone's goals** [T-1437, `ui/GoalTicks.tsx`; on the city panel's "Trips within" bar only, not on
  this one, T-1551] are at train shares of 10%, 20%, 30% and 40% (`GOAL_SHARES`); only 10% and 20%
  are drawn, as ticks over the bar [T-2145], laid over it so nothing moves, 1 px past the bar at either side: a reached goal's tick is 2 px in
  the text colour, an unreached one 1 px muted; under the 10% tick is the star, 7 px, filled in the
  accent colour once earned, an outline before. Each tick's hover names the goal and, once reached,
  when ("Goal: 20% train share, reached on Monday, day 71, at 14:08", the zone's local time). A
  goal is reached the first time the zone's train share on the average day reaches it in a final
  result, and kept from then (the save's `stars` for 10%, `goals` for the rest); 30% and 40% are
  still reached and kept, and shown in the city view's Goals reached row. **The real network**
  [T-2145, `game/realShare.ts`]: in the seven cities with a real network save (New York, London,
  Paris, Tokyo, Seoul, Toronto, Sendai) a 1 px line in the text colour through the bar with a 7 px
  pointer above it, at the train share that city's real network gets in this game's demand on the
  day shown (a holiday's days take the plain day they stand in for); its hover "Real world network:
  22.7% train share in Paris" (with " on a weekday" and the like off the average day). A fixed
  table, measured on each save in a new game with every zone open (notes/T-2145.md).
  **Zoomed out below zoom 7** (a country or a few regions on screen) [T-1865] the bar names "World" instead [T-1110]:
  the train rides summed over every city with a built station and a demand view, as the Cities
  tab's All cities row sums them (`game/worldTotals.ts`; the name's hover: the city and station counts),
  whatever lies under the view's centre. The figure is those train trips a day, "11.8M trips/day"
  (three significant figures) [T-1440]; the bar is their share of a fixed 25 billion trips a day in
  the world (8.2 billion people at about 3 trips each, `WORLD_RIDES`), so it is usually well under
  1% [T-1301], the walking and driving taking the rest in the cities' own proportions. No goal ticks.
  Its hover gives the train trips a day and in how many cities, the share of the world's rides and
  how they are reached, then the cities' own split as a city's hover gives it.

### Status and busy tags

- **Status** [T-349, `ui/StatusTag.tsx`]: a small tag (small type style, background, thin border)
  on the map's top row, to the right of the demand buttons and centred with them (`ui/DemandSelector.tsx`;
  the buttons stay at the top left). It appears and goes in one step and moves nothing; the whole
  text is its hover. On a map column under 520 px the tag (and the busy tag after it) goes under the
  day and night picture at the top right, right-aligned and as wide as its text (`MapSky`), taking no
  row of its own [T-1704, T-2202]; the demand row stops short of the day and night picture
  at the top right [T-1868], and the selection bar is under the rows
  (the find button's row included). **"Trips here being recomputed…"** shows whenever the trips of the zone in view are rough,
  early, from before an edit or from the last session's cache, or while its detailed solve is on the
  way [T-1851, T-1223, T-328, T-421]; when the solve doing it is a whole one over several zones (a
  game's load, an intercity edit's whole solve after its local one) it says **"Trips in 171 cities being
  recomputed…"** instead, the zones that solve has still to finish (`demandScope`, `demandSolving`),
  since that takes much longer on a big game than the zone in view alone [T-1904], except while a
  game's load solves the zone in view on its own first, which says "Trips here being recomputed…"
  [T-2045]; it goes once the zone in view's figures are final, though other zones still solve
  [T-392], and a zone solved on its own first counts as final at its last round [T-2045]. The tag
  shows all the while the clock is capped (below): when the zone in view is final and the cap still
  holds for an edit, it says **"Calculating trips after your change…"** [T-2622]. It says the same
  while an intercity edit's whole solve runs after its local result showed, until that solve's final: the
  local result keeps the longer trips and other zones' riders as they were before the edit (a line
  with no trains left can read tens of thousands) [T-3601]. A city edit's solve ends at its local result, and the
  tag with it; the idle refresh after it shows nothing unless an edit waits for it [T-2188, docs/demand-runtime.md]. While capped, the
  tag's hover says "Time capped at 4x while updating demand". After an edit the hover's first line is
  the time left [T-3599]: on "Calculating trips after your change…" to the edit's final, on the
  recomputing tag to the zone in view's figures; "About 5 min left" in whole minutes, "Less than a
  minute left" under 60 s, "Time left not known yet" before the solve has a plan, or while units whose
  populations mostly have no timed pass are left and none like them has ended (the first edit after some
  loads of a big game). Each solving worker times its units of the solve (a period of a local solve, a
  round of a period of a cluster solved whole, the zone in view's split free flow) by its populations'
  last passes, at the pace this solve and the last edit's ran against them, period by period, and a unit
  with no passes by its populations at this solve's time a population; the slowest worker's time, plus
  the page's wait between the local and the whole solve and the last edit's stretch from its last unit
  to its figures, is the estimate; with a newer edit waiting for that solve's final (T-1734), as long again
  as the running solve's whole time (not for one made in the first 10 s of a local solve, which likely goes
  at its local result, T-3616). A newer edit that stops the running whole solve (in its first 40%, T-3925)
  starts the time left and the tag's bar again from the new solve. It counts down and takes a new estimate as the loading card's time
  left does (never climbing by degrees).
  **"Calculation 1 day behind"** ("Calculation 3 days behind") takes the tag's place when a whole game
  day or more has passed since the oldest edit no solve's final covers yet (`demandBehindFrom`) while
  some figures in play still miss it: a line reading "calculating...", or the zone
  in view recomputing [T-2409]. Lines whose local result is in count for nothing [T-1733, T-2410]. While the
  edit changes the zone in view's figures the split is faded, the figures held [T-232]; a zone the edit
  does not touch, or whose cluster is already done, keeps its split unfaded. The split's hover ends
  "(trips being recomputed…)" in the same cases. A city loading later shows "Loading data" with a thin bar
  after it (Loading, below). Every status text has a thin bar after it and nothing before it
  (no bar on a failure) [T-2693, T-2728], past half only where bounded (Loading, "Every bar passes half only on a
  time left it can bound"): the load's own bar for "Loading data"; a zone's bar (`zoneLoads`) for a wait on a
  zone's load, for data or bubbles on their way and a click waiting on a zone; else, while a solve runs (a
  recomputing, behind or capped status, and data, bubbles or a click waiting behind it), the share of the wait by
  the time left its hover gives, bounded once the workers' units bound it, else, guesses, the solving workers'
  units done of all planned (an edit's solve, the unit in hand by its time against the stage's mean unit, to 90%
  of it, `demandWork`) [T-2729] or the rounds published of all (`demandStep`); terrain downloads finished of the
  most waiting at once (bounded); and where nothing measures the work, the share of the time the last such wait
  took that has passed, 3 s for the first (a guess) [T-4264]. Only the load's bar has a hover; the others pass
  the pointer to the tag. The figures change in one step when a detailed solve lands
  [T-1223, docs/demand-runtime.md "Levels of detail"]. No grey and no bar: those stay for a zone with no figures. A rough zone's city view,
  opened from elsewhere, and the Cities list mark nothing: rough figures are close enough (Anita,
  T-1206, 2026-10-28).
- **Busy** [T-1200, `ui/BusyTag.tsx`, `game/busy.ts`]: when the page is about to freeze for a
  while working something out, a tag in the status tag's style after it on the same row says what,
  in a word or a few, with no hover: "Drawing the network" (the network redrawn whole, at a new draw
  anchor) and "Updating the network" (a state after an edit). The tag is drawn and a frame painted
  before the work starts, and it goes when the work ends, in one step. Only work expected to take
  over 150 ms on this device shows it: each kind's time per unit (a state's edges and nodes) is
  learnt from its last runs in this page, so a fast desktop rarely sees it and a phone often. Not
  while the loading card is up.

### Cash and the clock

- **Cash**, four significant figures (to the dollar under a million), moving as fares are paid and
  running costs drain [T-284, T-722], in tabular figures in a box of fixed width after the train share,
  so nothing beside it moves as it changes [T-305, T-310, T-1865]. In a sandbox the box says
  "Sandbox" in the muted colour, with the money since day 1 (the cash minus the $10B start) and
  today's fares and running costs in its hover [T-728]. The hover ends with a line saying money
  rates are multiplied by 100 (`moneyTab.scaleTip`, T-3304).
- **The clock** [T-861], all of it the local time where the player is looking [T-309]: the time
  (figure style) and the date (normal style) on one line 6 px apart, "13:24 Mon Feb
  24" (hover "Monday, February 24, year 1, day 29"), [T-1868]. The date never carries the year; its hover
  does [T-2684, T-2731]. The day and night picture
  and its 2 px demand band are not in the bar: they sit at the top right of the map under the bar
  (`MapSky` in `ui/TopBar.tsx`, right 10 px and top 10 px of the map), right-aligned under the time
  and speed controls, on the wide bar and the two-row one. It is 250 px wide, shrinking to 150 px as
  the left-hand controls take the room (`--sky-w` in style.css; the sums assume the default UI scale,
  0.9), and below 460 px of map column the purpose filter ("All purposes") is hidden first, the picture
  going back to 250 px and shrinking again to 135 px at 360 px.
  The calendar is in [world.md](world.md) "Calendar". The scene
  (`ui/SkyBanner.tsx`, one canvas) shows 24 hours, 12 back and 12 ahead, now being its middle
  [T-896]. Its sky takes flat colours by the sun's height at the view's centre on that date
  (`game/sun.ts`): dark blue night with stars, a short pink and warm sunrise and sunset, light blue
  day [T-882]. Flat clouds sit at fixed game times, none to two a day as the day's number falls,
  white by day and dim blue at night [T-902]. Along the bottom is a track with sleepers, darker at
  night. At now sits the present-time marker: a red train (18 by 5 px, lit windows, nose
  to the right) on the track. The sun peeks in from the top edge at the middle of each day (the sun
  above -3 degrees) and a crescent moon at the middle of each night [T-1800], so they pass over the
  train as game time passes. Everything but the train is fixed to its time and slides left under
  it. It is redrawn on animation frames from the clock's `now()`
  whenever it has moved a fifth of a device pixel, never through Preact. The band under it has each
  hour's demand level (red high, yellow medium, blue low: `--dl-*` in style.css, both themes), so
  the colour under the train is the level now; its hover says "Low demand until 06:00", the time the level next changes
  [T-902, T-1242]. A public holiday in a country
  the player has built track or stations in (blueprint not counted) [T-862, T-1098] is a pink ribbon along the sky's top over its day (its
  country's local day), with a tick where it starts; the sky's hover lists those in view, and is the sky's only hover ("Tomorrow:
  Thanksgiving Day, United States", or "Tomorrow from 06:00: …" for a country on another local
  time), and on the day itself a pink dot sits before the date (its hover is the date's, which names the holiday). A holiday
  several of those countries keep on the same local date is one line [T-1098, `groupHolidays`]:
  the same name, or names `SAME` in `game/holidays.ts` counts as one (Labor Day, International
  Workers' Day, May Day and the like; Christmas; Boxing Day and Saint Stephen's Day; New Year's
  Day), shown by its one name or by `SAME`'s; the sky's hover names every country ("Today: Labor
  Day, Belgium, China, France and Germany"), the date's names two and counts the rest past three
  ("Labor Day, Belgium, China and 2 others"). A superholiday's going-out and coming-back days
  (T-264, docs/world.md "Calendar") are a paler pink ribbon with a small arrow at the start of
  each, right for out, left for back; its days away a pink ribbon. The sky's hover adds
  "Tomorrow: Golden Week, Apr 29-30, China", "Today: Golden Week, May 6-7, China" (the name and
  the dates of that part of the run, `RushSpan.dates` [T-1697]), and the same on a day away that is
  no holiday. **The dot** before the date [T-264, T-1476, T-1865, T-1888] is a button while a
  superholiday where the player has built is under way or goes out within 2 game days (otherwise
  it is a plain pink dot, there only on a holiday's day; there is no calendar icon). It is yellow
  (`--yellow`) while every such run is still ahead ("tomorrow", "in 2 days") and pink once one is
  under way ("now") or a holiday is today. Its hover gives each run within 21 days with the dates of its days
  away ("Golden Week, Oct 1-6, China: in 5 days") [T-1697], then the click's day by its part of the run,
  "Click to show demand on a going-out day, Sep 29-30" or "a coming-back day, Oct 7-8" (the days either side of the dates on the first line; they were a second
  unlabelled range, T-1888). A click sets the Day choice (Settings) to the run's
  going-out day [T-1373] (once those are past its coming-back day, then a day away), and a second
  click back to the day shown before. In the dark theme the
  scene is mixed 15% towards the background. Pause, then
  slower and faster buttons with the speed between them as text [T-3653]: one step along the slider's 20 stops (`SLIDER_STOPS` in `game/clock.ts`,
  real to 100x, spaced for the eye; "real" is 1 game second per second, 1/60 of 1x, and "4real" is four times that), each button off at its end; the
  comma key slows and the period key speeds up by one step (`clock.stepSpeed`, `main.ts`; listed in Help; held, they keep stepping every 90 ms at most [T-3717]). **Click the speed** for the slider [T-1664, T-3717];
  the speed stays until changed (not saved with the game). Replaced the 1x, 4x, 16x buttons; Settings > Map "Use legacy time speed controls" (remembered in the browser, T-3765; on for a browser with no stored settings and no saved game, off for one that has played before without choosing, T-3847) brings them back: pause, 1x, 4x and a third button showing the speed as text (a click runs it, a double click opens the slider), with the comma and period keys still stepping.
  **While the player's edits are not yet in the figures** the clock runs no faster than 4x [T-1665,
  `demandCapped` in `game/lineFigures.ts`]: after an edit, until every line it changed has its riders
  again (each reads "calculating..." meanwhile: its cluster's local result, or the
  solve's end), and after an edit no line's riders follow (fares, a station link or building, a line
  removed or stopped running, an undo), until the whole solve's final; and while a load the network waits on is going
  (a game's, a zone a station has just reached). The rule is the same wherever the camera is
  [T-2622]: nothing the view alone starts caps (a pack opened on a pan, the zone in view refined from
  rough, the zone in view solved first), and nor does a whole solve with no edit in it (a load's after
  the clock starts, a cached result's network solved again). The chosen speed's button is dimmed,
  its hover says the clock is held at 4x while demand updates and then returns to the chosen speed, and
  the cap lifts by itself; the status tag shows all the while it holds [T-1790].
  A new line with no riders yet ("calculating...", running trains that cost and earn nothing) holds
  the cap until its riders land, after quiet edits too [T-2409]. The cap is never lower
  than 4x (Anita, 2026-11-04).
  Demand that stopped (its notice) caps nothing. **A quiet edit does not cap** [T-1803,
  `game/editCap.ts`]: a train count or headway change at a level not in force on that line now, or by
  less than half of what it was (counted from before the edits since demand last settled, in trains
  for a count and trains an hour for a headway; from or to 0 is never small), or a stop edit on a line
  with no trains at any level. The status still shows "Trips here being recomputed…" (or the cities form) meanwhile, its hover without
  the clock line. One edit that would cap among quiet ones caps for them all; every other edit
  (track, stations, new lines, train length or type, fares, undo) caps as above. **Space** pauses and resumes at the speed that was running, even with a button focused.
  **A new game starts paused** [T-2142, `pauseNewGame` in `main.ts`], a sandbox and a real network's
  too (from the start menu or the saves panel; a `?new=1` page runs as before); a saved game opens running.
  While the tab is hidden the clock stops, unless "Run in background" is on (Settings), when it runs
  on at the chosen speed with fares, running costs and autosaves, and the map catches up when the
  tab is shown again [T-826].

### Narrow maps and Demand shown for

- **Narrow maps** [T-1853, T-1868, T-1883, Anita's sketch]: two rows below 700 px of map column, 56 px high.
  Row one has the name, the cash (the same box as the wide bar's), then the time and date at the row's end; row two has the mode bar
  with the train share under the name and cash, then the speed buttons under the clock. The same thin divider as the wide bar spans both
  rows, left of the clock and the speed buttons (`.bar-rule`). The rows are 1 px apart; the first row sits 4 px lower than it
  did and the name and the bar are 12 px from the left edge. On a short row the name
  gives way (onto two lines, whole words). Below 300 px the bar takes three rows (the name and cash; the split; the clock and speed, no divider),
  with cash, time, train share and all speed controls kept. The time and date end at the same right
  edge as the speed buttons under them (`.clock-text` right-aligned, T-1893). Hovering the split or the time gives
  what was dropped [T-106, T-1775].
- An **Autosave off** notice when this tab cannot autosave (saves, below); a click opens the saved
  games.

**Demand shown for** [T-260, T-266, T-464; `ui/DemandSelector.tsx`, display setting `dayShown`]: a
"Day" choice in Settings, right under the demand bubbles' Opacity [T-703]: Average (the default), Weekday, Sat or Sun [T-262], and while a superholiday is within 21 days its days as icons named on hover: a right arrow for a going-out day ("A day going out for Golden Week", T-1373), the calendar for a day away ("A day of Golden Week", T-264) and a left arrow for a coming-back day; each reads as the average once it has passed. The average is 5 weekdays, a Saturday and a Sunday over 7 ([demand-timing.md](demand-timing.md)
4.2). It sets the day every demand view shows, the split included, and does not follow the clock,
so nothing jumps when the clock enters a weekend. Hovers state the day beside every demand figure ("12,400 trip legs per day",
"per weekday", "per Saturday", "per Sunday": `shownDayPhrase` in `game/ui.ts`). A switch of day shows at
once. The trains, platforms, the
past 24 hours' riders and the panels' Today counts always follow the clock.

## Trouble list

[T-1433, `ui/TroubleList.tsx`; shown as the "Issues list", hovers "Show issues list" and "Hide issues list"] What is wrong on the network, in one list at the map's lower right,
10 px in from the bottom and the right, with the map's attribution button just left of it on the same baseline
(the attribution opens leftward, kept clear of the button's measured width) [T-2932]. Shut, it is a square button with an orange
alert mark (20 px) and the count in orange (`--warn`, T-3151: most items are not large problems), 15 px bold; open, the list stands above it (300 px wide, scrolling past 260 px or half
the window), and the button shows as on. Open or shut is remembered per browser
(`anitabuilder-trouble-open`); it never opens by itself and makes no sound. With nothing on the list
the button is gone.

- **Rows, worst first** (12 px type, one row each, the whole row's hover giving the figures):
  lines whose track is gone (its chip, a red alert mark); places where a line's reversal partway limits it (T-4178, as the map's reversal icons; an
  amber U-turn disc, the station or "Track end", up to three of the lines' chips); orange and red
  capacity tags, the map's tags at 90% and up (`NEAR_RHO`) in the demand period in force (T-3542; the hover's wait names
  it, "Each train waits 1m20s here in the morning peak") that are over capacity or
  make trains wait 2 s or more (a line alone queues only once over capacity, so a 99% tag with no wait is no problem) [T-2945], one row per
  kind by one station with the worst delay, red ones first ("Flat crossing near 14 St/6 Av", then
  the tag's "+3:40" or "104%"); lines leaving riders at the platform (a person and the count, both red; the row reads "{line} crowding") [T-2934, T-2944] where
  at least 3% of those who wanted their trains and 10 riders were left in the past 24 hours; lines
  whose trains left full at least a quarter of the time with fewer left behind (a train and
  "26/81", same label); a bus line's hovers say buses and a ferry's boats [T-2941]; last, lines with no stations (its chip, an alert mark, the hover "No stations"; a click
  selects the line). Within a kind the bigger figure goes first. A row's hover ends with a line starting
  "Try: " that names what to do about it (a few words per kind of capacity limit, `fix` in
  `game/capacityKinds.ts`; more trains for crowding; stations for a line with none; rebuilding the track
  for a broken line) [T-2933].
- **A click** moves the camera as search does: a line to its stops, a place to zoom 14 or closer.
  With the select tool it also selects the line, the reversal's first line, a capacity tag's
  junction or station (a tag on plain track selects nothing).
- **Cheap on big networks**: the capacity tags and reversals come with the world as the map draws
  them; the past 24 hours' counts are asked once a game hour and when riders are worked out again
  (`accountVersion`), not on the panels' refresh (up to four times a second, and no sooner than twice the last
  answer's time after it, so about once a second on the largest games) [T-3793].

## Tutorial card

[T-1822, T-2123, T-2263; `ui/Guide.tsx`, `game/guide.ts`; the step words are `WORDS` and the notes
`NOTES` in `game/guide.ts`] A flat
post-it at the map's upper right under the day and night picture, 240 px wide, not tilted: a pale yellow
(`--yellow` at 24% over the background; 14% in the dark theme), no shadow. Heading "Tutorial" (14 px,
bold) and an x (no hover; screen readers get "Close the tutorial"), then a row of 12 px type per step: a
12 px tick box, the step, its counter right in tabular figures (no hovers on the counters). Done steps
carry a tick; the step in hand (the first not done) is in the text colour, steps after it muted. Ticks
show at once. Some steps have a note under the step's words: in the muted colour while it is the step
in hand, and once the step is ticked it stays for the rest of the card in the same muted colour
[T-2473, T-2686: the card has only the text colour and one grey]; a step after the one in hand, not yet ticked, shows none. The search results open over it. Two cards, one after the other.

**The first card**, reading demand, then laying a line:

| Step | Counter | Counts | Note |
|---|---|---|---|
| Zoom into a city | none | only in a game started on the world map (Choose on map, or a location far from every zone): a zone in view at city zoom (not the zoomed-out bubbles) with its demand drawn | |
| Hover the red and green bar at the top of the screen to see how people travel | none | the pointer on the top bar's mode split in this page | Green is walking, red is cars and other modes, blue is your trains. |
| Click a red bubble to see where its trips go | none | a bubble selection | A bubble shows trips from homes there. Red bubbles mostly drive and green bubbles mostly walk. |
| Click one of its destinations to see that trip | none | the trip explorer with both ends | |
| Click Destinations to see where trips end | none | the demand view on Destinations [T-2685] | |
| Place 5km of track | 2.3/5.0 km | any track, blueprint or built, not switched off | |
| Place stations on the track (space them ~1km apart) | 1/3 | stations, blueprint or built, at least 500 m apart | |
| Construct your blueprints | none | at least 5 km of built track and 3 built stations 500 m apart | |
| Create a line | none | any line | |
| Assign stops for the line by clicking them on the map | 1/3 | stops of the line with most, goal the lesser of 3 and the built stations | |

**The second card**, running the line:

| Step | Counter | Counts | Note |
|---|---|---|---|
| Add trains to the line | 0/1 | any line with a train at any demand level | |
| Unpause the game | none | the clock running (a new game starts paused [T-2142]) | |
| Hover the dot by High to see if there are enough trains | none | the pointer on a line panel's crowding dot (any level's) in this page | If trains are crowded, they will show as >100% capacity, and people will be less likely to ride. To carry more, you can increase train length. |
| Add trains at high, medium and low demand | 1/3 | the demand levels with at least one train, on the rail line with most | If only a short stretch is crowded, adding more trains can end up costing more than they earn. |
| Show line traffic on the map | none | the track display on Traffic | Thicker lines represent more riders. Red borders indicate track at full capacity. |
| Reach 1% train share in New York (the zone in focus) | 0.6/1% | the zone's train share on the average day at 1% (`GUIDE_SHARE`, T-2689), or any region's star | |

A step ticks the first time its check holds and stays ticked (undo never unticks it); counters keep
following the world and cap at their goal. A later step can tick before an earlier one. Once every
step of the first card is ticked, a Continue button appears under the steps and opens the second
card; the second card's last tick gives a Done! button (closes the card for good in this game).

- **Who sees it**: a new Normal game in a browser that has not finished the first card once; not a
  sandbox, a real-network start or a `?new=1` page, and not a save made before it existed. Finishing
  (every step of the first card ticked) is remembered in the browser's UI settings (`guide`); a game started after it has
  no card. Switching a game to sandbox takes the card away.
- **Kept in the save**: the header's `guide` (ticked steps as bits, `GUIDE_BITS`; `hidden`, `closed`,
  `card` 2 for the second card, `zoom` for a world-map start), set by the clock worker on a new game
  and updated by the page's `guide` message. A reload picks up where it was.
- **Closing and back**: the x asks first (the confirmation: "You can restart the tutorial in
  Settings.", Keep going and Close tutorial), then hides it in this game; Game in Settings has a row "Tutorial" with
  Start again in every normal game: it clears the ticks and the hidden, closed and card marks and shows the
  first card, whether it was hidden, finished or never there (a game saved before the card) [T-1835]; steps
  the world already meets tick at once.
- **The view rows fold**: while the first card is up, the map's station ridership and track colour
  rows (ui-demand-views.md) show only their fold icons, as if folded, without changing the remembered
  folds; a row the player opens stays open for the page. The demand bubbles' row shows as remembered,
  since a new game with the card opens on Homes (any other new game opens on Destinations) [T-2685]. The second card shows every row as remembered.
- **Longer hovers** [T-2263]: while the card is up in a game (`guideVerbose` in `game/ui.ts`), the
  track tool's road chip reads "85 road crossings +$12.0M" (afterwards "85 crossings +$12.0M"), and the
  line panel's crowding dots and the Lines list's crowding add "Maximum crowding in the busiest
  hour of the busiest segment of the line." after the peak load, and, when
  at most half the line's stretches are past 100%, "Only a short stretch is crowded. More trains
  would run mostly empty, and may cost more than they earn." after the stretch count.
- **Out of the way**: it hides while a confirmation or the saved games panel is open, and with F2.
- **A phone held upright** (the dock under the map): 184 px wide, only the heading and the step in
  hand; a tap on the heading opens every step.

### The ripple

[T-1822 stage 2, T-2123, T-2263; `ui/ripple.ts`] Rectangles round the control the step in hand uses: 2 px in
the accent colour, each starting on the control's edge and growing 16 px out over 2.8 s while it
fades, one every second, on and on until the player presses that control. One fixed canvas over
everything with no pointer events, drawn by its own `requestAnimationFrame` loop (no CSS transition)
and placed from `getBoundingClientRect()` every frame, so a scrolling dock keeps it on target. Nothing
is drawn while the card is out of sight, the pointer is held or the stop editor is open on the stops
step. With reduced motion asked for: one still outline 8 px out, held until the press.

- **Targets** carry `data-guide="<step id>"`: zoom the find button, track the Track tool and
  stations the Station tool (Build tab), construct Construct blueprints, line New line (Lines tab),
  stops Edit stops (the line panel; the attribute is off while the stops are being edited), trains
  the stepper in the High row of the Schedule (the trains count or the headway, whichever the line's
  schedule uses), crowd the High row's crowding dot, levels the stepper of the first row with no
  train, play the top bar's Pause button (which also resumes), destinations the Destinations button, traffic the track display's
  Traffic button, and "end" the card's Continue or Done! button once every step is ticked. The split
  step points at the top bar's mode split by its class (`GUIDE_SPLIT`). The bubble, trip and train
  share steps point at nothing. The tab icons carry `data-guide-tab`, the view rows' fold icons `data-fold`.
- **When**: from 0.35 s after a step becomes the step in hand until its control is pressed; on a
  click on any row of the card (the step's own control, so a done or later step can be pointed at
  too), likewise until pressed. Once pressed, three rings 0.5 s apart come again after 20 s without a click, key or wheel while it is open (at most 3 times per step).
  A pulse that is due while the pointer is held (a drag) or the stop editor is open on the stops step
  waits for it to end.
- **Out of view**: a target whose tab is closed ripples the tab icon (Build for track, stations and
  construct, Lines for the line and line panel steps); the line panel steps with no line picked
  ripple the first row of the Lines list when that tab is open; Destinations or Traffic in a folded row
  ripples the row's fold icon. The ripple moves to the target itself as soon as it shows (the tab opened, the
  dot drawn), and back to the stand-in if it goes. A target scrolled out of its pane is scrolled into
  view (instantly) first, then ripples.

## Saved only in this browser dot

[T-1832, notes/T-1823.md fix 9; `ui/KeepDot.tsx`, `game/keepNudge.ts`, `game/keepState.ts`] A 10 px
yellow dot (`#f2c200`, a thin ring in the text colour) in a 20 px click area just right of the map's
search glass, level with it [T-1868]. Hidden while the search box is open. Its hover is the game's own tip, two lines:
"Saved only in this browser!" and "If your browser data is cleared, this game will be lost. You can
make an account to save it online, or download a copy locally." A click opens the account panel
(Sign in, or the account if signed in). No x, no buttons. Where accounts are off (no account server:
localhost until `?acct=live` or `?acct=local` is set, or the server cannot be reached) the second line
is "If your browser data is cleared, this game will be lost. You can download a copy to keep it." and a
click downloads the game, as the saved games panel's Download does.

- **When**: after an hour of real play in a game kept only here: the clock running, the tab visible,
  a saved slot this tab writes (a `?new=1` page has none, so never there). Counted in 5 s steps and
  kept per slot in localStorage `anitabuilder-keep` as
  `{ms, downloaded}`; nothing goes in the save file. `?keepMs=<ms>` shortens the hour for a check.
  The count stops once the game is downloaded. Entries of deleted slots are dropped.
- **Kept only here**: the slot is not an `account:` game, was never uploaded to an account (`cloud.acct`
  set and `rev` above 0, even if that account cannot be reached now) and is not tied to the signed-in
  one (`tiedTo`). Adding the game to the account (at the sign-in question or from the saved games
  panel) ends the dot.
- **Goes away**: when the game is on the account, or after any download of it (the saved games
  panel's Download or the file in Settings; `markDownloaded` in `downloadSave`). Never otherwise:
  there is no way to close it, and it is back after a reload while neither has happened.
- **Not on your account yet** [T-2614, notes/T-2613.md]: the same dot, ahead of the case above, when the
  playing game (a slot this tab writes, tied to the signed-in account, not given way to another device)
  has had its latest change fail to reach the account for 2 minutes (the slot's record holds the
  reason; `?keepMs` shortens the 2 minutes too). Hover: "Not on your account yet", then the reason
  with a full stop ("The server could not be reached.", "Offline.", "Your account is full. Delete a game
  to make room.", "This game is too big to keep on your account."), "This game is saved in this
  browser." and, for the first two, "Click to try again." A click saves by hand (as Ctrl+S), which
  sends it at once. Gone within 5 s of a send going through. The clock worker sends a failed one again
  every minute, paused or not, and at the browser's `online` event.
- **Out of the way**: hidden with F2 and while the start menu is up.
- **Check**: `checks/t1832-keep.mjs` (own server, `?keepMs=3000`); screenshots in
  `artifacts/trainworld/t1832/`.

## Search

[T-148, T-177, T-183, T-211, T-4004; `game/search.ts`, `game/placeSearch.ts`, `ui/Search.tsx`]

- A square button with a magnifying glass at the map's upper left, under the station ridership row
  (the last of the controls stacked there, in `.map-top-left`, its rows 4 px apart; the top right is the day and night
  picture) [T-1868]. The open box is 250 px wide and its results drop over the map below it. A click, `/`, Ctrl+K or Ctrl+F, from anywhere but a text
  field or the confirmation dialog, opens the input in its place, focused. Ctrl+F is kept from the
  browser only then. Up and down move through results, Enter or a click picks, Escape clears and
  closes it, as does leaving it empty; left with text, it stays open.
- It finds the save's stations, its lines (by name, or the badge symbol exactly), the registry's
  cities and zones, and the towns and neighbourhoods of every zone's OSM place list (a place's other
  OSM names count). Places are found by a worker (`workers/placeSearch.worker.ts`), so no query holds
  up the game [T-4004]: it loads the place lists of the zones within 30 km of the view when the box
  is first focused (while a search of three letters or more finds fewer than five, the next nearest
  zones' lists one at a time, the 48 nearest within 200 km at most) [T-400, T-3997], and for each
  query one shard of the world name index (`pipeline/place_index.py`: every word of every name, and a
  name of several words joined, keyed by its first three letters, two for CJK, kana and Hangul; towns down to villages first, then
  neighbourhoods and hamlets), so a town anywhere is found by a word of its name starting with the
  text; substrings and nicknames only in the lists near the view. One fetch at a time, at most 56
  lists and 24 shards kept, eight places posted back. Results show as they come; the row the player
  moved to stays under the cursor. Matching ignores case, accents and punctuation: exact names, then
  names starting with the text, a word starting with it, containing it, then clipped nicknames
  ("south philly": each word starts like a word of the name bar its last two letters, keeping at
  least four). Ties list cities, lines, stations (busiest first), then places: those within 200 km of
  the view first, then bigger kinds, then population. A query of one or two letters (one CJK
  character) finds a far place only by its whole name, after every near one. No addresses (no
  geocoder).
- Up to eight rows: a line's badge and "Line", a city "City", a place a hollow ring and its kind,
  a station its line badges (one per badge look, four at most), blueprint stations muted. Results
  sharing a name get a muted second line telling them apart: the city; within one city the lines
  only this one serves (three at most, a shared first word once) or else its compass direction
  from the others. A place more than 200 km from the view gets its city there too. Stations of one complex sharing a name are one result, picking the platform
  with the most lines [T-182].
- A pick moves the camera at once: a station to zoom 14 or closer, a line to its stops' bounds, a
  city to its registry centre and zoom, a place by its kind (town 12.5, suburb 13, neighbourhood
  14.5). With no build tool on it selects the station or line (a city clears the selection, a
  place clears a station or bubble selection); with a tool on it only moves the camera.

## Keys and focus

[T-183, `ui/keys.ts`] One rule for the game's keys (listed below) and search's `/`, Ctrl+K and Ctrl+F. A text field keeps every key (search's
own box gives Ctrl+K and Ctrl+F to selecting its text). Another form control keeps only its own
keys (arrows, Home, End, Page Up and Down, Enter, Space, and in a select a letter); the rest reach
the game. A select picked with the mouse gives up focus at once, so the next Space pauses; one
reached with Tab keeps it. Buttons show the browser's focus ring for keyboard focus only [T-169].

The key map [T-1819, numbers and letters of the tools back by T-1834; tables in `ui/toolKeys.ts`; letters read lowercase, so Caps Lock does not matter; none fires
with Ctrl, Meta or Alt held]. Each button's or tab's hover shows its key.

| Key | Does | From |
|---|---|---|
| 1, 2, 3, 4 | tool: draw track, place stations, delete, select; opens Build (`TOOLS`) | any tab |
| Shift+1 to Shift+7 | draw at elevation +3 down to -3 (Shift+4 is ground level); by key code, digit row only; with a blueprint pick in the selection bar, put the pick at that level instead [T-2285] | any tab |
| B, L, A, F, Y | open the Build, Lines, Stations, Finances or Cities tab (`TAB_KEYS`; a half-drawn route stays) | any tab |
| U, O | start a new bus line, a new ferry line (`BUS_KEY`, `FERRY_KEY`) | any tab |
| M | swap the map for the network diagram, and back (`DIAGRAM_KEY`, `diagram/view.ts`) | any tab |
| T | explore a trip; the tab stays | any tab |
| D | demand bubbles: None, Homes, Destinations, Total and then None again, once the view's demand is ready (`DEMAND_KEY`, T-2338); each of the four buttons' hovers names it | any tab |
| N, S, G, C | new line, edit the selected line's stops, add every station the selected line passes, construct blueprints (`ACTION_KEYS`) | any tab |
| E, Delete | E enables the picked blueprints when any is off and disables them when all are on (`SWITCH_PICK_KEY`, T-2338; the Enable hover, and the Disable hover while all are on, name it); Delete removes them | select tool |
| R | rename the open station: its name box opens (`RENAME_KEY`, T-4273); the pencil's hover names it | a station's panel open |
| Space, Esc, F1, F2, Ctrl+S/Z/Y, `/`, Ctrl+K | pause, back out, open the help panel, hide the interface, save, undo and redo, search | any tab |

C stays Construct's, so Cities has Y; A opens Stations and S is Edit stops (Anita's pick); add-passed-stations
has G (gaps). Anita wants 1234 kept for the tools (T-1834), so elevation is on Shift+1 to 7 again.

## Help

[T-908, T-3527, `ui/HelpPanel.tsx`] The plain "?" at the top of Settings [T-1099] (hover "Help (F1)") or F1 from
anywhere (`HELP_KEY`; the browser's own F1 help is prevented; nothing while the interface is hidden, or a dialog
or the start menu is up) opens a panel over the map beside the dock, under the tab row.

**Common questions** first (Anita, T-3527): a compact list of what players ask again and again, each a line with
a chevron that opens its answer under it, one open at a time, ruled off from the controls below. The answers name
the game's own controls through their keys ("Copy line", "Calculate intercity trips", the Select tool and its
key, "Rebuild"), so they follow those labels in every language. Now: a branching or short turning line ("Copy
line", then the new line's stops), an express line ("Copy line", then drop the skipped stations), a loop line
(right click, on a phone hold, the stop to loop back to while adding stops), no intercity riders (only while
the setting is off: turn on "Calculate intercity trips"), rebuilding sections of track (the Select tool and
its bar's Rebuild), editing or deleting built track (the Delete tool, or the Select tool's Rebuild or Remove),
four tracks using only two (new stations on the new pair and a new line), and the game recalculating after every
piece (pause and turn off "Recalculate trips while paused", which is on by default; an entry points to a setting
only when its default does not already fix the problem, Anita). New questions are proposed to Anita first.

**The controls** then list only those no button shows: turning the map (right drag or Ctrl drag, T-2031), the bubble
gestures (Shift+drag box, hold then drag, Shift+click, Shift+drag over far bubbles), building
modifiers (Shift+click, double click or Enter, Backspace, a double click on track adding a point
(T-2951, not on a phone), Caps Lock, right drag, E and Delete in the blueprint tool; a bus point's right click and drag, and a click on a bus route's end, T-1514), the keys (1 to 7, the tab and tool letters, T, D, N, S, G, C, U, Space, Ctrl+Z/Y, `/` and Ctrl+K, Esc,
the letters read from `TAB_KEYS`, `TOOL_KEYS`, `ACTION_KEYS`, `TRIP_KEY`, `DEMAND_KEY` and `BUS_KEY`; F2 hides the interface, below; F1 itself) and panel gestures (double click
a fare node, a name, a list row; Shift or Ctrl on the train stepper). Esc (before the game's own
Esc, so the selection stays), a press outside, F1 or the button again closes it. A control added or
changed that a button does not show gets its row here.

**Hiding the interface** [T-1491, `ui/hideUi.ts`]: F2 (`HIDE_UI_KEY`, any panel or field, with no
modifier) puts `ui-hidden` on the page, whose rule in `style.css` hides the dock, the top bar, the tool
notes, the map's control rows, search, blueprint bars, trouble list, loading card, touch bar, debug
panel, the day and night strip with its train (`.map-sky`, T-1893) and MapLibre's controls; the map, network, bubbles, station labels and hover tags stay, and the
map takes the room. F2 or Esc brings it back (Esc first, before every other Esc). Listed in the
help panel as "F2". While hidden, pressing F2 and then moving the mouse show "Press F2 to show the UI" at the top
centre; it disappears 3 s after the key or the mouse stops [T-1548, T-1893].

**Confirmations** [T-113]: a centred dialog headed "Confirm?" [T-1242] with operation-specific text, cancel
and action buttons. Cancel has initial focus, Tab stays inside, Escape cancels, focus returns when
dismissed, and the map and tools are blocked meanwhile.

## Settings

A search box heads the tab ("Search settings", T-3681, `ui/SettingsSearch.tsx`): typing hides every row whose shown
text (labels, options, icon buttons' names), section heading or keywords do not hold each typed word, ignoring case,
accents and full-width letters; a matched heading shows its whole section, a matched row its heading; "No settings
match" when none. Keywords are a short list per labelled row or heading (`ALIASES`), in English and the player's
language, both always searched; a new setting needs no entry to be found by its label. Esc or clearing the box
shows the tab as it was.

`ui/SettingsTab.tsx`, remembered per browser, in order: two icon buttons at the top, Help (a plain "?",
opens the [Help](#help) panel) and, when accounts are on, Account or Sign in (the person icon, the account
panel, [Account and leaderboards](#account-and-leaderboards)) [T-1099], with light text beside the icon,
"Logged in as alice" or "Not logged in" [T-1728]; then the Language row: a globe icon (hover "Language" over
"Non-English text is translated by AI") and a select of English, 日本語, 简体中文 and 繁體中文 (台灣), each in its own
name, switching at once and remembered on this device; a first visit takes the browser's language when offered ([localization.md](localization.md)
"Choosing a language") [T-2636]; Demand bubbles, first the Day choice (Average, Weekday, Sat, Sun;
"Demand shown for", below) [T-703], under this heading since it sets the day of every demand figure [T-2345], then the Purpose
choice (the map's purpose filter, the same list as the one on the map's demand row, which phones do not show; T-2548), then three sliders: Size
(0.25-2x, [T-099, T-2271]; the radius is the slider times `BUBBLE_BASE`, 0.75, so 1x is the old 0.75x, T-1852), Opacity (20-100% of the built alpha, default 70%, [T-470, T-1726]; for the
commuter and longer-trip bubbles, not station discs; ignored while bubbles are selected, T-936, but not while a whole zone is, T-1726) and Detail (4 steps from coarser to finer,
the third by default; its hover reads "How many bubbles are aggregated together"; [T-795, T-1838]), the Colors
choice (3-color by default [T-1724], the old Mode mix, 2-color [T-833], the old Train share, which shows two colour pickers labelled 0% train and
100% train, Purpose [T-2886] or Color blind [T-3005]; no hovers on any of it [T-804, T-1838]; both in [ui-demand-views.md](ui-demand-views.md) "Local trip bubbles"), "Show transit by mode" (off by default, no hover; "The goal number" [T-3695]), Station circles with a Size slider (0.5-2x, 1x by default, the station ridership circles' radius; [T-1565], [ui-demand-views.md](ui-demand-views.md) "Station ridership circles"), Show (trains, station names [off by default, T-1729], "Track at capacity" [the capacity markers; hover "Marks where stations or track cannot smoothly handle scheduled train frequencies", T-1843], "Crowded trains" [T-1152, T-1843, T-2236: the Traffic view's red edges, on by default; hover "Stretches where trains carry more than their rated load at the busiest hour"; [ui-map.md](ui-map.md) "Track display"], flat junctions and crossings [no hover, T-1843], "Lines side by side" [T-1886: lines sharing track as bands beside each other, on by default, T-2346; [ui-map.md](ui-map.md) "Lines sharing track side by side"], map labels, Real network (None, Passenger or All, then "Show stations", off by default, [ui-map.md](ui-map.md) "Real network" [T-3570]), Terrain [hover "Construction is more expensive in steep terrain", T-1843], then "Show levels" over the level buttons +3 to -3 [T-3420], with a small "Show all" button beside the label while any level is hidden [T-3593; its space is kept, so the row does not move]: all lit, a click hides that level's track, the lines on it, its stations, junction and ramp marks and the cars of trains on it, and picks pass over them, so pieces laid exactly over each other at two levels can be reached; a ramp fades toward a hidden level; no hovers; kept while the page is open, not remembered, so a reload never opens on a map missing a level; under them "Buses and ferries", ticked by default, unticked hides every bus and ferry line's route, stops and vehicles on the map and from picks, kept while the page is open as the levels are, then "Blueprints" [T-4003, remembered in this browser (localStorage)]: unticked, no blueprint is drawn or picked (track, stations, lines, BRT lanes, their marks and drag squares), Construct and the blueprints themselves unchanged, a viewed community network opens with them hidden whatever the setting, with a switch of its own while viewing, the Lines tab hides lines one by one, [ui-inspectors.md](ui-inspectors.md) "Hiding lines"), then Game (right after Show; below), Riders on trains
and platforms: no setting, since each train's fullness and each platform's filling up are always drawn [T-164, T-3579], Stations up close (Faces and Sound, both on by default, no hovers [T-1843], and a Volume slider while Sound is on, the one volume for every sound, building and fare-change sounds included [T-1725]; [ui-map.md](ui-map.md) Life at stations) [T-686], Map (Flat or Globe, flat by default; hover "Zoomed in to a city, the globe becomes the flat map"; [ui-map.md](ui-map.md) "Globe") [T-033], then two sliders: Line width (50-200% of the drawn width of lines, track, casings and trains, 100% by default; station shapes and picking follow it; once double track parts into its two tracks up close, those keep their width on the ground) [T-1248] and Base map (contrast, 0-100%, 100% by default: below it every basemap colour but water and the land, so roads, parks, woods, buildings, boundaries and airfields, is mixed that much toward the land colour, in both themes, the basemap's labels fade toward transparent by it instead, text and halo alike, so a faded name never leaves a pale patch on water [T-1266]; the real network is left as it is, T-1291; `map/basemap.ts`) [T-1249], and "Rotate map with right drag" (off by default, Anita 2026-11-02: with no tilt it adds little; off turns the map back to north; [ui-map.md](ui-map.md) "Globe") [T-2031], "Use legacy time speed controls" (off by default, T-3765; the top bar's older pause, 1x, 4x and custom-speed buttons, `Speed` in `ui/TopBar.tsx`), and "Show zoom buttons" (on by default: a plus and a minus at the map's right edge, halfway down, one zoom step a click with the map's own easing, `ui/ZoomButtons.tsx`; no hover, just aria labels) [T-3268], Theme (Light or Dark, light by default, over Pink, Blue or Green) [T-039], Panel and text size, Units (US units, Currency), Performance (Use more cores, Run in background, Recalculate trips while paused, Calculate intercity trips, Calculate trips in less detail, Memory), Default bullet shapes (below), then,
when accounts are on, Account [T-1139]: who is logged in ("Logged in as alice", "Logged in with
Google" or "Not logged in") and a button, Account settings or Sign in, that opens the same account
panel as the icon at the top and the start menu's (sign in, recovery email, change password, sign
out, delete; [Account and leaderboards](#account-and-leaderboards)), and Submit feedback at the bottom
[T-1673, `ui/FeedbackPanel.tsx`; the same button is on the start menu's foot and the failure screen, T-3630, the
form mounted in a root of its own from startup, `startFeedbackUi`, so it opens with no Settings tab; sent from
there, the "sent" note shows at the page's top centre]: a centred modal "Feedback" (the account panel's form styles) with Name (optional)
and Message, and Send feedback. Under the feedback row, a Ko-fi cup icon link (inline SVG, the cup in the theme's colour and its heart in Ko-fi's red #ff5e5b, https://ko-fi.com/anitamaps in a new tab) with Anita's line "please consider donating to keep trains world running!" beside it; both carry the hover "this game is free, but the cost to host it is starting to get expensive. please consider donating!", then About and Changelog [T-3340]. Feedback POSTs to the account server's `/trainworld/feedback`
(`sendFeedback` in `game/account.ts`, the same server choice as the account: live on anita.garden, `?acct=local` for
127.0.0.1:8083), with the build id as `version`; the server stores it and mails it on, so no address is in the page.
Beside the message the report carries the browser, memory, language, the last load failure, the currency, and
`state` [T-3313, `game/diagnostics.ts`]: the clock worker's oldest unanswered edit or read and the last hour of trips,
and the demand pool's newest solve, last step and oldest download, each with its age, so a report of figures that
stopped says where they stuck.
Screenshots [T-3326, `game/feedbackImages.ts`]: against a server whose config has `trainworldFeedbackImages` above 0, an "Attach screenshots" button, a count ("n/4") and thumbnails with a remove button sit under the message. Up to 4, from the picker, a paste (Ctrl/Cmd+V) or a file dropped on the form; PNG, JPEG or WebP only. Each is shrunk to 1920 px on its long edge and encoded as WebP (JPEG where the browser cannot) under the server's 1.5 MB, then sent with the message. Replies take none. The inbox page shows them as small thumbnails that open the full image.
Signed in, the session rides along so the report names the account [T-1719]. A send that works closes the panel and shows a passing note at the map's upper left under the edit note, "sent! thank you for feedback" (plain text as the edit note's, 5 s; `showPassingNote` in `ui/PassingNote.tsx`) [T-1709, T-2166]; a failure says why (no server for this copy,
not reachable, too many from here, or a server whose `/noritetsu/config` lacks `trainworldFeedback`: "Feedback can't be
sent yet") and keeps the text. Station ridership
circles are a row on the map instead, under the demand buttons [T-1448].

**Default bullet shapes** [T-3319, T-3329]: after the Performance section, a row per mode, Train, Tram, Bus and Ferry, each a row of the three bullet shapes,
the buttons touching in one compact grid (no gaps between rows or buttons)
(circle, square, diamond, outline; no hovers, aria labels only), the shape of that mode's lines that have none of their
own ([ui-inspectors.md](ui-inspectors.md) "Line shapes"). Train and Tram start on the circle, Bus on the square,
Ferry on the pill, as the bullets were before (the pill is not pickable, so no button is lit on Ferry until one is picked); a change redraws at once every line of the mode without its own.

**Panel and text size** [T-247]: 80% to 120% in 5% steps, 90% by default (T-703), no reset button. It
zooms the dock, top bar, map buttons, search, loading card, tool notes and dialogs, and sizes the
text of the map's DOM tags and the hover tag; the map itself is never scaled.

**Font** [T-698, T-703, T-1524]: Nunito, the only font and not a setting; a font choice stored
by an older version is dropped on load. Nunito is self-hosted (`public/fonts/`,
Latin and Latin extended subsets of the variable file, about 75 kB, SIL OFL) and has no Japanese, so
kana and kanji fall back to Zen Maru Gothic, which `index.html` loads from Google Fonts. The panels,
station names and map controls follow the `--font` variable (`style.css`; `game/fonts.ts` loads the
files), and station names and capacity tags are measured again once it has loaded. The map's own
labels are glyph tiles and are Zen Maru Gothic.

**Units** [T-381]: "US units" (off by default) shows every distance, length and
speed in miles, mph and feet instead of kilometres, km/h and metres: panels, inspectors, hovers, the
track, station and line tools' readouts, the placing rings, the scale bar of a station complex, and
prices per mile. All go through the distance functions in `game/format.ts`; the game itself stays
metric. The Currency row follows.

**Performance** [T-381, T-3329]: "Use more cores" (off by default) allows up to 5 demand workers instead of 3 at most; its
hover says it applies from the next load ([architecture.md](architecture.md) 7). "Calculate intercity trips" [T-2838,
T-3630], on every device, remembered in the browser, on by default on a computer and off on a phone or tablet: off,
no market pack opens, so there are no trips between zones and no demand cache (a solving worker about 174 MB lighter on
New York's real save). Ticked mid-game it applies from the next solve; unticked mid-game, the market packs already
open stay and keep their trips until the next load. The start menu's settings show the same box, and a game opened there has
intercity trips as it says; "Play without intercity trips" (a memory failure) and "Load without intercity trips"
(the start menu's memory warning) untick it. "Calculate trips in less detail" [T-4117], on every device, remembered in the browser, on by
default on a phone or tablet and off on a computer: every zone held coarse ([demand-runtime.md](demand-runtime.md)
"Levels of detail"); hover "Trips are calculated faster. Riders on each line can change by a few
percent, at small stations by more." It applies at once, mid-game too (the figures stay until the new ones land). "Run in background"
(off by default) keeps the clock running while the tab is hidden (the top bar's clock, above;
[architecture.md](architecture.md) "A hidden page") [T-826]. "Recalculate trips while paused" (on by default): off, edits made while
the game is paused start no demand solve; the first moment the game runs again, one solve covers them all
([architecture.md](architecture.md) "Editing while paused") [T-3427]. There is no Map detail choice: the map is always the full one
(the lighter map and its Auto are gone, DECISIONS.md 2026-11-12; a stored Low is dropped). A **Memory** row ends the section:
"1.4 GB / 8 GB", the memory the game takes now against the most it aims for here (`memBudget`: 8 GB, or 0.75 of a device
that reports less; 6 where the browser does not say), red above 85% of it, refreshed every 2 s while Settings is open; hover, two lines,
"About 0.5 GB more per 1,000 stations" and "8 GB limit is an estimate" [T-3329; `game/memoryEstimate.ts`]. The reading is from the demand
workers' WASM memory (`workers/liveMemory.ts`, T-3757): 1.17 GB, plus 2.69 times the largest solving worker's and the aux
worker's (what grows with the network), plus 1.56 times each further solving worker's; within about 20% of the browser's own
total from a fresh game to a 7,070-station game on 3 solving workers (notes/T-3757.md).

**Game** [T-728], right after the Show section [T-3579]: the game's mode, "Normal" or "Sandbox" (hover: what
it means). A normal game has "Switch to sandbox mode", which asks first ("Confirm?", then
"Money cannot be turned back on in this game.", with Stay in normal mode
and Switch to sandbox mode) [T-822]; a sandbox has no way back, so a later campaign's
or a hard start's achievements cannot be had for free. The mode is the game's, kept in its save,
not the browser's. First in the section, **Trains run on** Left or Right [T-3297; `ui/SideSwitch.tsx`]: picking
the other side switches the whole game (world.md "Running side"), at once on a network with no track, else after
a question ("Switch trains, trams and buses to the left? One-way track and flyover legs turn around to match, and
lines move onto them. Anything that still doesn't work is listed after. Undo history is cleared.", Switch side and
Keep this side). After a switch with anything to report, a list opens ("Switched to the left"): To fix (lines
that can't get through their stops, junctions now crossing flat, one-way track no train can use) and Fixed (one-way
track turned around, lines rerouted), each row a click that shows the place and selects the line, junction or track
and closes the list; under the row "3 places to fix after the switch" (or "Nothing left to fix after the switch")
with Show list, until another game opens. In a normal game a
row "Tutorial" with a Start again button (restarts the [tutorial card](#tutorial-card))
sits under the mode [T-1822, T-1835]. Then "Name new stations after their complex" [T-3759], off for a new player
(on for one who had settings stored before T-3814) and remembered in the browser, shown before a game is loaded too ([ui-inspectors.md](ui-inspectors.md) "Renaming"). Last in the section, "Buses obey one-way streets" [T-3480; moved here from Show, T-3579], off by default and remembered
in the browser: a bus route runs a one-way street only its way, with arrows on those streets in the bus tool. It shows before a game is loaded too.

## Finances tab (id `money`)

Cash (always the whole network's; in a sandbox "Since day 1" instead, the cash minus the $10B start,
T-728; the Cash row's label and the Yesterday and Today headings have the 100x hover, T-3304); then, in this order, yesterday and today, the fare controls, History, trains and the loan
[T-1660, T-1691]. Yesterday and today: fares, running trains, upkeep, "Loan" (the loan's payments,
hover "Paid back over 60 game days in equal payments")
while a loan is out or was paid yesterday (the network's only, never a region's) [T-2022], new trains,
construction, net (Upkeep's hover is seven lines: $35k per km of ground track per day, $50k on a viaduct, $100k in cut and cover, $130k in a bored tunnel, $50k per station per day, $200k underground, High speed rail 1.5x [T-1931, T-2619]; Running trains' hover reads "Trains cost $4 per km, plus $2 per train car"; T-867, T-1838), headed by the Lines and Stations tabs' place-or-world toggle (T-332, one setting
for all three): with the place on, these figures and History are that region's (T-531; the
place's hover adds how a region's money is counted), with the globe the network's. A region with
nothing in it shows $0. Money figures
in panels have three significant figures from a million up ("$22.4M", "$1.15B", "$233M"; thousands
as "$450k"; under $500 "$0") and a negative one is in the alert red (`ui/MoneyText.tsx`, `money` in
`game/format.ts`) [T-722]; the cash (top bar and this tab) has four ("$11.36B", "$893.6M"; to the dollar under a million) and is red
with its minus when negative; hovers may give more. History
[T-447, `ui/MoneyHistory.tsx`]: a 1d / 30d / All time switch [T-2271] (All time by default, kept while
the game is open, no hovers): 1d is the past 24 hours, hourly, 30d the past 30 days and All time every
day, daily; over two charts, each on a
linear scale from zero [T-2181], each title ending with its scope, the place's name or "all cities", since
the place switch is a scroll above [T-2487]. The first, "Revenue and expenses in New York" (revenue blue and expenses red in every
theme, the tokens `--revenue` and `--expense`, T-1111), has two bars side by side in each column
[T-2340, T-2444]: fares and running costs with upkeep and loan payments (T-867), the bars filling the column
less a 1.5 px gap before the next (the train rides' bars likewise). A pale red bar of the one-off costs (new
trains and construction; none when there are none) sits on top of the running costs bar, the same width
[T-2463]. Fares and running costs set the scale, the larger of the two;
the stack's one-off part is on the same scale, cut at the top with a red cap when it would go past, so a build day does
not flatten the others [T-452]; the top's figure at the right. Then
"Train rides in all cities" (or the place's) [T-565] as bars up from zero, the top's figure at the right (hover: it exactly), the
region's with the place on and the network's with the globe, from the same ledger entries (rides
are counted as in track-and-service.md 6.4). Days are the ledger's, one each in order (T-488), every
day kept (T-2143); while they span more than 90 days a column is a week (Monday to Sunday), past 90
weeks a calendar month, past 90 months a run of months, each column's figures and hovers then a day's
average over the days it covers ("$22.8M/day", the hover's name "Nov 10-16"), the day now counted as
far as it has gone, and no holiday ticks (All time only; 30d is always one a column). A young game's
few sit at the right of at least 7 days or 12 hours. The
day or hour now is drawn as it fills (paler bars) and moves as fares land.
Under the charts a tick under each day or hour that was a holiday or a superholiday's day where the player has built (the region's country with the place on), pink for a day off and a day away, paler for going out and coming back, its hover the day and one line each ("Thanksgiving Day, United States", "Golden Week, Apr 29-30, China") [T-264, T-1697]; then, when All time spans more than 365 days, a row of year labels ("Year 1", "Year 2") each at the column its year starts in, with a thin rule at its left (the last ones on the right edge end in it instead) and the hovers' dates ending ", year 2" [T-2684]; then the first and last dates ("Apr 14", "Apr 14, 04:00"). Hovering a column gives its
day or hour ("Mon Apr 14", "Mon Apr 14, 04:00", "so far" for now) with fares,
costs and their parts, or its train rides. Days run midnight to midnight in the start zone's
local time (the ledgers', [track-and-service.md](track-and-service.md) 6.4), so today's column turns as
the top bar's clock passes midnight there; hours are written in that local time [T-2271]. The
charts are saved with the game and are there after a reload [T-567]. A column's hover adds an
"Loan" line when it had loan payments, which the expenses bar includes [T-2022]. The fares, whatever the toggle,
are one country's [T-1277, T-1690]: the selected station's or city's, else that of the built station
nearest the view's centre, else (nothing built) the zone in view's, else the home city's. The graph and rows show the network's curve times that country's fare scale, and
an edit there sets the curve (divided back by the scale), so every country's fares move with it.
The fare curve (rule: [track-and-service.md](track-and-service.md), fares) is drawn as a graph of
the fare against trip length, both axes linear
[T-864, T-883, T-1112, T-1143, `ui/FareGraph.tsx`]. "Nice" below means 1, 2 or 5 times a power of
ten. The distance shown starts as the smallest nice one holding 1.2 times the last node's distance
with every node's fare below 90% of the fare shown (200 km for the default curve) and is kept while
the game is open; the wheel or two fingers over the graph zoom it smoothly, and the minus and plus
buttons beside the Fare heading step it to the next nice one, from 1 to 5,000 km (in the display
unit). The fare shown over the plot's height is always $0.15 a km of the distance shown, so the
line's slope on screen never changes with zoom ($30 at 200 km). The fare axis starts at $0, or higher
where the base fare would otherwise stand above a quarter of the height: then the base fare sits at a
quarter of the height. Fare ticks fall at every multiple of the smallest nice step of at least a
quarter of the fare shown, distance ticks every nice step of at least a fifth of the distance shown,
the last labelled with the unit. The Money tab sets three nodes; a curve of any other count (set in
"Advanced fares" below) still shows and drags here, and the rows under the graph give way to one line, "Set in
Advanced fares" [T-3739]. An "Advanced" button left of the graph's undo opens that panel on the tab's country.
The first node, the base fare, stays
at 0 km and drags up and down only; the other two drag either sideways (distance) or up and down
(fare), whichever way the pointer first moves 3 graph units, and only that way for the rest of the
drag. Hovering a node draws small grey arrows the ways it drags (two for the base fare, four for the
others); during a drag only the chosen way's two show, and the cursor shows it too. A drag snaps
distance to the nice step of at least 1/200 of the distance shown and fare to that of 1/250 of the
fare shown (at least $0.01), within the plot; each node stays at least 2% of the distance shown (and
0.1 km) beyond the one before and short of the one after, and its fare between theirs, so fares never
fall with distance. The scale holds still during a drag. A drag changes the graph and the figures
below it as the pointer moves, and sends the fare to the game once on release (each change restarts
demand), with the quiet pop of the build sounds [T-1490]; a release with no change is silent. While fare changes are the last edits sent, Ctrl+Z takes them back one
at a time, and Ctrl+Y or Ctrl+Shift+Z puts them back, in order with the track's own undo and redo; the
undo button left of the zoom buttons does the same, enabled only then [T-1130, `game/fareUndo.ts`:
the sim's edit history holds track, lines and schedules, so fare steps are kept on the main thread
and any other edit drops them]. A node past the plot is an arrow
on the plot's edge pointing toward it, with its hover; a click on the arrow zooms out to show it. A
hover anywhere on the graph names the fare at that distance, a node's hover its distance and fare
and the ways it drags. The axis labels are 11 px in the graph's own units. Below the graph, one row
per node [T-1201, T-1234], no rules between them and none wrapping at the dock's narrowest width,
their steppers lined up in one column: "Base fare" with its fare in a stepper, then each stretch as
`0 -` [20] `km` [T-1264], only its end distance in a box (typed, no steps), and its rate in a stepper,
[+$0.07] `per km`, the unit outside the box; the second "Past 20 km" with its rate only (following
the first's end; the last node's distance changes no fare, so it is only dragged). Then, once any
line runs high-speed trains, "High speed" with the high-speed fare multiplier in a stepper ("1.25x" by default,
by 0.05 from 0 to 5, a click types one), its hover the rule, and after a change an undo icon ("Back
to 1.50x") [T-1626]. Then "Transfer fare" with the fare each change of line pays in a stepper
("$0.00" in a new game, by $0.25 as the base fare, a click types one; at most the base fare, T-3689), shown times the country's
fare scale as the curve is [T-1801]. Last, one row "Japan fares" (the country above; no hover on the label, T-1852)
with its fares as a multiple of the reference country's (T-2679, docs/track-and-service.md "Fares by country").
With fares by city (T-3766) the fares and this row are a city's, picked as the country is (the selected station's
city, the selected city, the nearest built station's, the zone in view, the home city): "Osaka fares", no anchor
icon, its multiple read against the reference country's default fares.
The reference country's row shows a plain "1.00x" in the stepper's column, no stepper and no buttons (hover
"Reference country" over "Other countries' fares are set as a multiple of United States fares."). Any other
country's is a stepper, its default multiple until set (a US-started game's Mexico 0.40x, Japan 0.63x; hover
"Japan fares as a multiple of United States fares"), stepping to the next 0.05x up or down (0.63x to 0.65x or
0.60x) from 0.10 to 19.90 (a click types one, kept to 0.01x and acting as exactly that many times the
reference's fares, T-2021); after it an anchor icon ("Use as the reference country" over "Fares charged stay
the same.") makes that country the reference, and once the player has set one, an undo icon ("Back to 0.40x",
its default) returns to the default. The stepper sits in the same column as the rows above
[T-1277, T-1554, T-1584, T-1727, T-1906, T-2679]. The base fare
steps by $0.25; a rate to the next whole cent above or below ($0.138 to $0.14 or $0.13), or by a
cent of fare where a stretch is too short for that to move it. A click on any value opens a box to
type it, Enter or leaving the box applying and Escape cancelling [T-626, `TypedValue` in
`ui/widgets.tsx`]; a stretch's end is kept in order as a drag is. A row's change keeps every other
row's value, so a base fare or rate change moves that node's fare and every later one's by the same
amount, and a stretch's end keeps both rates; a rate never goes below $0. Fares show whole cents; a
rate shows two decimals when it is whole cents and three otherwise (typed or dragged). Dragging on
the graph sets fares, and the rows' rates follow. While a drag is under way, or from a fare edit until
demand has solved it, two more rows follow: "Riders a day" with the estimated change in rail trips
("-6%") and "Fares a day" with the estimated change in fare income ("+$1.2M"), each hover giving
the estimate and today's figure; otherwise neither shows [T-1300, `game/fareEstimate.ts`: each
layer's rail trips times exp(its fare sensitivity over its rail trips x the curve's change at its
riders' mean trip length, line by line), fare income as the riders kept pay the new fares]. Under
History, cars
owned (prices in its hover), cars the lines need and spare cars [T-631]; the value of the track and stations built. Last, "Loan"
(not in a sandbox) [T-1661, T-2022]: with none out, "Borrow" and three buttons, $1B, $5B and $10B, each hover
"Borrow $1.00B and pay back $22.2M per day for 60 days", or, disabled, why it is refused
[T-1906]: "Payments would be $111M per day. Average profit over the past 7 days was $32.0M per
day." (after one whole day "Profit yesterday was $32.0M per day."; profit is fares minus running
costs and upkeep), or before a whole day with fares "Until
there is a full day of fares, the most you can borrow is $1B."; with one out, "Borrowed" and its amount,
"Payment per day" (hover "Interest is 1% per day of what is left"), "Days left" (whole days, rounded up)
and a "Repay" button that pays off what is left (hover "Pay back the $999M left from the cash"),
disabled while the cash is under it (hover "Needs $999M in cash"). Default fares show rounded until the player changes the curve or that
country's scale: in a save from before T-1906 a country's fares are the curve times its scale,
rounded to a cent under $0.25, 5 cents under $10, 50 cents under $100, then a dollar; in a newer game
each country charges the round start curve it would have had as a new game's [T-1662, T-1906, `game/fareDefault.ts`]. Rules: [track-and-service.md](track-and-service.md)
6.4. The economy's 100x scaling is in the hover of "A day at this schedule".

### Advanced fares

[T-3739, `ui/AdvancedFares.tsx`, notes/T-3739.md] A centred modal like the schedule panel's, 600 px wide, its top
fixed (rows coming and going under the graph never move the graph under the pointer); closed by its x, Esc or a
click outside. Top to bottom: "Fares in [country]" (a picker of the listed countries, starting on the Money tab's)
with the graph's undo and zoom; the fare graph at 560 by 260 units, the Money tab's rules but for drags, which may
come right up to a neighbour (0.1 km) so a drag can make a step; the rows as the Money tab's, each stretch's with a
bin ("Remove point"; the base fare has none), their amount boxes one width (64 px, seven characters such as "+$55.50"
or "+¥1,500") with the figure centred [T-3749, T-3751]; "Add point" (pressed while armed; off at 12 points, hover "At
most 12 points") and, for a curve of other than three points, "Use simple curve", which asks "Go back to three points
at 0, 20 and 100 km? The other points are dropped." and keeps the fares at those distances; a thin rule under "Add point"; then the high-speed, Bus, BRT and boat size rows
as the Money tab shows them (each only while one of its lines exists, but Medium's "Ferry" with any ferry line, T-3915)
and the transfer fare: where the panel is
480 px wide or more and a ferry line exists, in two columns, high speed, Bus, BRT and the transfer fare on the left and
the boat sizes on the right, else one column, the transfer fare last; the estimate rows while a change is unsolved; the countries: the
reference country, then every country with a built station by name, each row its multiple (the reference a plain
1.00x), its base fare as charged and the anchor and undo buttons, under one row of column heads, "Fares by
[country | city]" (bold, over the names; the two words a switch, T-3766), "Multiple" and "Base fare" [T-3749]. With
"city" pressed the rows are every city with a built station by name, each its multiple, base fare and undo (no
reference row, no anchor), and the picker above lists cities ("Fares in [Osaka]"). Esc in a typed value's box cancels the typing and keeps
the panel. **Steps in the rows** (the panel's only; the Money tab's three rows stay rates) [T-3749]: a stretch of
0.5 km or less that is not the last reads as a step, one row
"10 km" with the fare's jump there, "+$2.50" with no "per km" (stepped by the base fare's step, typed, never below
$0; a change keeps the step's width and moves every later point's fare by the same amount); the stretch after it
reads from the step's distance ("10 - 40 km", "Past 70 km"). The step's distance is its row's typed box ("[10] km";
plain at 0 km), which moves the whole step, its width and every rate kept; the stretch before a step shows its end
as plain text ("0 - 10 km"), since it is the same number [T-3751]. **Add point**: pressed, the next click
on the graph adds a point there (the dock's graph never adds), and the button lets go; Esc lets go first, a second
Esc closes. While armed the points take no clicks, the curve with the new point is drawn dashed over the faded one
with a hollow dot where it would go, and the plot's hover reads "Click to add a point here" over its distance and
fare. The point's distance snaps as a drag's; within 8 graph units of a point sideways it goes 0.1 km after it when
the pointer is above that point's fare and 0.1 km before it when below, so a click over a point makes a step. Its
fare snaps as a drag's, is taken as a neighbour's within one snap step, and is held between its neighbours' fares;
where nothing fits (before the base, between two points 0.1 km apart) the click does nothing. Each add and remove is
one fare edit (one recalculation, Ctrl+Z, which the panel passes on). The drag estimate averages each line's fare
change over trips 0.4 to 1.6 times its mean length (`fareEstimate.ts`), so moving a step shows gradually. Room is
left below the fare zones for a fare per stop (T-3744).

**Fare zones** [T-3740, `AdvancedFares.tsx` `FareAreas`, the rule in track-and-service.md "Fares and the ledgers"]:
above the countries, the heading "Fare zones" (bold, over a rule, with a hover on the charging rule), then each fare area: its name in a box that reads
as plain text until hovered (typed over, Enter or leaving it renames; no recalculation), "Paint stations" with a
brush, which closes the panel and starts the Fare zones tool on that area, and, left of a bin ("Remove fare area", asking
"Remove {name}? Trips between its stations pay by distance again."; its stations leave it, not undone, as the
removal isn't), the game's usual checkbox with no text, ticked while the area is on (hover "Disable {name}" or "Enable {name}"; T-3783, T-3791; off, trips pay by distance as if the area did not exist, its paint, name and prices kept and its price rows dimmed; `FareArea.off` in the header's `fares.zones`, sent as a `fareZones` edit like removal, so not undone; `demandFares()` leaves an off area out); then one row per zone count, "1 fare zone", "2 fare zones" ..., as many as the area's
highest zone painted and at least three, each its price as charged in the panel's country (a hover on each: what rides it is for; the stored price times
that country's rate multiple, `countryFareScales`, as demand charges it; never below the base fare) in a stepper by the base fare's step, typed on a click. A price
change keeps prices from falling: the later ones rise to it, the earlier ones come down to it. Last, "New fare area",
which makes an area named "{city} fare zones" for the city in view, or the one holding most of the player's stations when none is in view ("Fare area 2" with none), "{city} fare zones 2",
3 ... when an area has that name already (T-3791: names never repeat, so an area made twice by mistake shows), its prices the curve's
fares at 5, 10, 15 km and so on, and starts the tool on it. A game with no fare area shows only the heading and
that button; nothing else in the game changes.

**The Fare zones tool** [`map/fareZoneTool.ts`, `game/fareZonePaint.ts`, `ui/FareZoneParts.tsx`], tool id `fares`
(no Build tab button; started from the panel): its bar at the map's top centre, as the station tool's, with the area
(a picker once there are two) and "Fare zone" with a stepper of the zone a click gives, 1 to 12 on its colour or
"none" (which takes stations of that area out of it, leaving other areas' stations alone); the click mode's descriptor "Painting fare zones" and the hint
"Click a station or drag around several." ("Tap a station or draw around several." on touch). Under them a second
row: "Set zones by distance" (rings, a toggle like "Add point"), which while on shows "Ring width" with a stepper
(0.5 to 30 km or mi, 5 km or 3 mi at first, kept for the session, not an edit) and the hint "Click the station at the
center." ("Tap ..." on touch); a left drag then pans the map, the station under the pointer previews its rings (each
ring filled faintly in its zone's colour, dashed lines between), and a click on a station gives every station or
complex whose middle is under 12 rings from that complex's middle (ground distance) zone floor(d / width) + 1 in the
picked area, one `stationFares` edit (one Ctrl+Z), turning the toggle off; stations past ring 12 stay as they are,
and complexes in another area ask the "Move stations" question as a loop does. Esc turns it off. A click on a station
(within 14 px, from zoom 9 where stations are drawn) gives it and its whole complex the zone; a left drag draws a loop (kept in map coordinates, so zooming or panning mid-drag leaves it on the ground), dashed and shaded in the
zone's colour, ringing the stations it would change (with "none", only those of the picked area), and on release every station inside (and their complexes) takes
it. A station is in one fare area at most (T-3791): when a click or loop takes in a complex shown in another area, the
game asks first ("Some of these stations are in another fare area. Move them to this one?", "Move stations" or
Cancel); Move stations moves them into this area, Cancel (or Esc) paints only the others. Linking stations (the passage) asks the same when their complexes are in different areas: Move stations puts both in the first station's area, Cancel makes no link; with an area on one side only the other joins it, no question (the link and the move are two Ctrl+Z steps; a link has none of its own, T-3796). Each click or loop is one
`stationFares` edit (one recalculation of trips, one Ctrl+Z), and none when no station would change. The right or middle button drags the map; on touch one finger draws the loop, a tap paints, two
fingers pan and zoom (MapLibre's touch pan held to two fingers while the tool is on). Esc drops a loop being drawn,
then leaves the tool. While the tool is on the ground near each painted station is tinted in its zone's colour (one
colour per zone number, `FARE_ZONE_COLOURS`), under the network: each painted complex (or lone station) owns its
cell, the ground nearer it than any other painted one (every area's in one diagram, so cells never overlap) within
650 m (at least 7 px on screen), drawn as vector shapes at 22% with a 2.5 px band at 80% inside each zone's edge (the
cell's arcs, and sides facing another zone or area; none between cells of one zone). A complex takes the zone of its
first painted station, as the station panel and demand do. Every painted complex or lone station has a disc with its
zone number beside it, above left first, else the first of seven other spots clear of the station names and the
numbers already drawn; where none is clear (stations crowded far out) it is left out. Off the tool nothing is drawn.
The station panel has a "Fare zone" row (a stepper, the same values) while the fares have an area: a station's own,
or in the complex view the complex's, stepping every station of the complex at once in its area (or the tool's when
it has none), the area's name in the label's hover when there are two or more.

## Cities tab

[T-550, T-749, `ui/Cities.tsx`] Whatever the Lines toggle says: a list of every region (the toggle's regions) with a built station, one row each, under a first
row "All cities" that sums them [T-2389], a faint divider after it [T-857, T-1110, T-1865] (its split bar weighs every region's trips, its rides, profit and points are the sums; hovers as a
row's, "in all cities"; the top bar names the same sum "World" when the map is zoomed far out, `game/worldTotals.ts`; it is not sorted; a click opens the
all cities view ([ui-inspectors.md](ui-inspectors.md#the-all-cities-view)) and tints the row, a double click also zooms out to every built station as that view's go-to does [T-2395]). Each row: its name
(hover: its station count), its mode split bar as the top bar draws it (the same colours and the
same hover), its train trips per day and its profit a day (the column is headed Profit), fares less running trains and upkeep (yesterday's, today so far
until the game has a first full day; hover: fares, running trains and upkeep, then new trains and
construction, left out of the figure, when there were any; losses red), and its points, its forecast
weekday's, changing when a solve answers with a whole final (the last final stays shown through an edit's solve), and a dash (hover "calculating trips…") from a load until its first final [T-3465, T-3644], to three significant digits ("512k", "4.56M", "103M"; `pointsShort`) [T-831] (hover: the exact figure "on a weekday", and the rule) [T-684, T-694]. A star after the name marks a region
whose train share has reached 10% (hover: when it was earned, "Monday, day 71, at 14:08" in the
region's local time) [T-685]. The top bar's city name has the same points in its hover. A narrow
list keeps about six characters of each name (60 px): below that the star goes first, then the
profit column, then (beside the city view on a phone) the points and the bar (container queries on
`.cities-box`) [T-693, T-1438]. Sorted by rides, largest first; a click on a header sorts by that
column (name, train share, rides, profit, points) as in the Lines tab. A click on a row opens the
city view ([ui-inspectors.md](ui-inspectors.md#the-city-view)) and a double click moves the camera
to the region as search does, as the Stations tab's rows do [T-1438]; the selected city's row is
tinted, or with no city selected the viewed region's. On a phone the inspector stands beside the
tab, as beside Lines and Stations. With no built station the tab reads "No stations yet." Above the
list (and above "No stations yet."), at the right, the medal and the count of achievement steps earned,
"11/27" (hover "Achievements"), opens the achievements view in the inspector and is lit while it is open
([Achievements](#achievements)); a sandbox has no medal. Before it, when accounts are on (a sandbox
too), a trophy (hover "Leaderboards") opens the leaderboards ([Account and leaderboards](#account-and-leaderboards)) [T-2179].

## Achievements

[T-2124, `game/achievements.ts`, `ui/Achievements.tsx`, `ui/inspector/AchievementsView.tsx`; design and
reasons in notes/T-2124.md] Goals for the whole game, each step kept once earned with the game second
it landed and never taken back (undo, a closed line or a fall in riders leaves it); the rings show the
figure now, so a fallen figure can sit beside a medal it no longer meets. A sandbox earns none, and a game made a sandbox
keeps what it had and earns no more.

- **Running** means a rail line with status running and trains set at some level; buses, blueprints and
  lines with no trains never count. **Demand figures** come from final results on the average day only,
  so the Day choice changes nothing, and a city's share from no rough zone's (as its goals); Trips per
  day counts coarse zones too, as the All cities row does. Network figures are checked as the
  world changes, demand ones on each final, the best line's profit as each game hour turns.
- **Tiered** (each tier a step): **Cities served** 2, 10, 25, 100 (regions, the Cities tab's rows, with a
  built station where a running line stops); **Trips per day** 1M, 10M, 100M, 300M, 1B (the All cities row's
  train trips a day); **Busiest station** 500k, 1M, 2M, 5M boardings a day at one station complex;
  **Busiest line** 500k, 1M, 3M, 10M riders a day on one line. A jump over several tiers records each
  and toasts the highest.
- **Firsts**, in this order: **Over the border** (a line with stops in two countries, each stop's zone's
  country), **Bicontinental** (two continents, `game/continents.ts`: UN M49, the Americas split at
  Panama, Russia at 60°E, Kazakhstan at 51.5°E, Egypt at Suez, Turkey at the Bosporus and Dardanelles),
  **Longest road** (a line 5,000 km long), **String of pearls** (a line 3,000 km long whose profit over the
  past 24 hours, as for Money train, is above zero), **Money train** (a line's profit over the past 24 hours, the
  Lines tab's Money figure, of $200M), **Bullet train** (a train reaching 300 km/h), **City circles**
  (a whole circle line of 10 stops: it closes on its first stop, turns back nowhere partway and stops
  at no station twice; a stem or a loop-back partway does not count), **Transfers world** (transfers per trip at least 1, over at least
  100k trips a day; its hover while unearned adds the figure now, "Now 0.56 transfers per trip"),
  **Trains city** (a region whose trips within it are 20% by train: its 20% goal),
  **Grounded** (a region whose trips to and from other places, both ways as the city view's "To other
  places", are 20% by train, over at least 100k such trips a day), **Sleeper** (riders who stay 12
  hours on one train: of the riders on a stretch, all but those who boarded since an earlier stop were
  aboard there, and at least 1 a day must be sure to ride that long), **trainsfan** [T-3520] (100 hours played in
  this browser: `game/hoursPlayed.ts` counts the seconds a game is open and the tab visible, as the plays ping
  does, in localStorage `trainworld.played`, across saves, from this release on; each save earns the step
  itself; its hover while unearned adds "23.4 hours played so far"). Sea crossing comes with T-2150.
  **An earned step stays earned** (Anita, 2026-11-10): the save's record is only added to, so a rule made
  stricter later takes nothing from a game that already has the step.
- **Hidden firsts** [T-3037]: not listed in the view or counted in its "n/N" until earned, then listed
  like the others (no rule hover, since none was shown). Gold medal, name, and a hover of what earned
  it. Two kinds. **Corridors** (`CORRIDORS` in `game/achievements.ts`): one network of running lines, joined
  where two lines share a station or a complex (the places a rider changes), has a built station at each
  end, worked out when the running lines or stations change. All Japan (Hokkaido to Shikoku by position
  inside Japan), Orient Express (the Paris zone to the Istanbul zone), Transcontinental (New York to San
  Francisco zones), Trans-Siberian (a station in western Europe, a list of countries, to one in Russia at
  or east of 104E), Trans-Andean (Santiago to Buenos Aires zones), Cape to Cairo (Cape Town to Cairo zones),
  The Ghan (Adelaide to Darwin zones), Silk Road (Istanbul to Xi'an zones). **Shares** (`NATIONS`,
  `EUROPE_SHARE`): the train rides a day in the built zones of a region (final results, the average day) over
  its people times 3 rides each (the world share's rule, T-1301): Grand Tour is 5% of Europe (by zone,
  `continentOf`: Russia west of 60E, Istanbul's zone counts), and one per country with the most play,
  each at its own share: Japan Railways 10, US 2, Spain 3, Britain 4, Italy 3, Germany 4, Korea 8, Taiwan 5,
  China 1, Indonesia 2, Mexico 2, Brazil 2, Australia 4, Turkey 2 (how they were set: notes/T-3037.md).
- **Medals** [T-2179]: a disc in the tier's metal on a ribbon in the achievement's colour. Tiers take
  bronze, silver, gold, platinum and emerald in order (four-tier ones stop at platinum); every first is
  gold. Colours (`achievements.css`, a light and a dark value each): Cities served blue, Trips per day
  red, Busiest station orange, Busiest line green, the firsts purple.
- **Players' share** [T-3038]: the hover of every listed step ends with "12% of players have this" once the
  server (shonei-server `/trainworld/achievements`) has answered with at least 20 players; none before.
  The game sends each key once per browser (`game/achievementCounts.ts`; design in notes/T-3038.md).
- **The view** (selection `achievements`): the medal and "Achievements"; each tiered one a row with its
  name at the top left, under it a place per tier holding the tier's medal once earned or blank space
  where it would go, no outline [T-2431], so the earned medals keep their places (hover the tier, and once earned when, "10 cities served, reached on Monday, day 406,
  at 14:08" on the game clock), and at the right a 54 px ring in the achievement's colour on a pale
  track of it, filling clockwise from the top from zero toward the next tier, the figure now in its
  middle and "/100M", the next tier, small under it; past the top tier the ring is full and the figure
  alone (hover, for the busiest station or line, which); the figure sits 2 px below the ring's middle [T-2431]. Then, after a gap, the firsts, earned ones first and the rest after them, each group in the
  list's order [T-2431]: a gold medal once earned, else an outline in its place, the name, "day 406" at the right; one not earned is muted with
  its rule in the hover, an earned one's hover says what earned it and when.
- **The toast**: at the map's bottom left, 10 px in, above the tool notes (`#tools`), at most 280 px
  wide: the step's medal (the highest tier's, or gold for a first; the folded one the accent's medal icon), the name in bold, one muted line of what was reached ("1M riders/day on
  Line 1a"). A 150 ms rise (none with reduced motion), 5 s held (the pointer over it holds it), a 1.2 s
  fade [T-2221], in real seconds. Up to three stacked, newest at the bottom, the next shown at least 1 s after the
  last; more than three waiting fold into one, "4 more achievements". A click opens the view. They wait
  while the start menu, the saved games, a confirmation, the leaderboards or the account panel is up, or
  the tab is hidden, and go with F2. A phone held upright shows one at a time, its line cut to one.
- **Saves**: the header's `achievements`, by key (`cities:10`, `trips:1000000`, `border`), each with
  `at` and what earned it (`of` a name, `v` the figure, `in` the countries or continents; a corridor's
  `of` and `in[0]` its two stations, a share's `of` the country code). An older
  save without the field works its out quietly once its first final result and the first profit answer
  are in: everything met then is recorded at that game second with no toast.

## Saved games

[T-068, T-241, T-251, T-320, T-1395; `ui/SavesPanel.tsx`; save format and slot rules in
[architecture.md](architecture.md) "Saves"]

### The panel and its cards

- The floppy disk button left of the gear, or the top bar's Autosave off notice, opens a centred
  panel over a dimmed page, closed by Esc, its close button or a click outside.
- **The head**: "Saved games", one icon, then Close [T-1395]. A door-and-arrow button at the right of the New game row while a game is open
  (hover "Save and quit to main menu"; it saves, lets the game go and reloads to the start page, `leaveToMenu`) [T-2850]. Where accounts are on, signed out a
  crossed cloud (hover "Kept in this browser only. Sign in to keep your games on your account."; a
  click opens Sign in), signed in a cloud (hover "Signed in as anita"; a click opens the account
  panel over this one). Where they are not (a localhost page, or no answer), no icon, except a
  crossed cloud with the hover "These saves can be lost if the disk fills up." while the browser
  refuses to keep the site's storage (`navigator.storage.persist`, asked once a slot is written,
  which Chrome grants or refuses without a prompt) [T-1214; `game/saveSafety.ts`].
- **Above the cards**, two buttons [T-1131, T-1395]: New game (accent border) shows the new game
  sheet (below) in the panel, headed "New game"; its Start closes the panel and opens the new game,
  its Back returns to the cards. Open a file (hover "Open a .save file as a new saved game") asks
  nothing: it closes the panel and opens the file's game in a new slot, the game being played saved
  first [T-743; the slot rules in architecture.md]. On a test page, whose game has no card, a copy
  icon (hover "Make a copy") follows them. Each of these, and Open and Open as a copy, reloads the
  page into its game ("Loading" below).
- One card per game [T-2037; architecture.md "Versions of one game"]: every slot of one origin,
  the account's games of it not in this browser and their backups are one card, showing the slot
  being played if it is one of them, else the version with the latest player change; the card is
  that version's (its name, picture, facts and buttons). Cards are ordered by their latest save,
  newest first. A card's picture is a 204 x 136 picture of its network, drawn by the clock worker when the
  slot is saved, at 480 x 320, on the land colour of the theme it was saved in [T-543]. The card
  shows it in the theme now set, whichever it was saved in: the page recolours the picture's land,
  rule and grey track to the other theme's and leaves the lines' colours, and the names over it use
  the current theme's ink and land (`ui/thumbShade.ts`, T-1852). The panel is 540 px wide.
  The picture is one panel per city with stations (each station in its zone by the coarse outlines,
  [zones.md](zones.md) "A place's zone", T-1265), the most stations first, at most four, leaving
  out a city with under 8% of the biggest one's stations (a fringe of its network) [T-894]:
  one fills it; two sit side by side; three are a large panel on the left (60%) and two stacked on
  the right; four are a 2 x 2 grid. Panels are split by a thin rule in the casing colour. Each panel
  frames its city's stations (at least 3 km across, room left at the top for the name) and draws
  all track in grey and each line in its colour that falls in it, so a line to another city runs
  off the edge; stations are not drawn [T-1100]. The city's name is text over the panel's top
  left in that theme's ink with a land-coloured halo, and with two or more panels its station
  count under it; the hover gives both. With more than three cities to show, the fourth panel is
  the whole network instead, framed on everything built and named "Whole network" with all its
  stations [T-2394]. A network with no station in a city is one unnamed panel
  over all of it. Then its name, "Autosaved 2 min ago"
  for the game being played ("Saved just now" after a save by hand, while it is the last write) or
  "Saved ..." for the others (exact time on hover), the game day, the cities its stations are in,
  lines, stations and cash. A slot with no stored picture shows "No picture yet" until it is next
  saved. A card of an account game whose picture is still being fetched shows "Loading..." in its place
  (a game with nothing built ends on "Nothing built"). Buttons sit at the card's bottom right, icons first, each icon with its hover [T-1395, T-2611]: under the picture and facts, on the row of the "3 other versions" line, at its right (on a row of their own when the game has one version):
  - **The Playing card**: a pencil after the name (hover "Rename"), the Playing mark, then copy
    (hover "Make a copy"), download (hover "Download as a file", `trainsworld-nyc-day-12.save`),
    bin (hover "Delete", asked first: `Delete "Tokyo" from this browser? The game ends and you go
    back to the start menu.`, from this browser and your account when tied) and Save [T-1852].
    Deleting the game being played removes the slot, its backups and the account's copy, stops
    autosave writing it again, and reloads the page to the start menu as a page opened with no game
    chosen (`ClockClient.leave`, worker `deletePlaying`).
  - **Every other card in this browser**: download (its last save in this browser, read from
    storage as it is, so also one that is damaged or from a newer version [T-4126]) and bin (hover "Delete", asked first), then Open (accent border; closes the panel and
    loads it).
  - **A card's bin deletes the game**: every version, after `Delete "Tokyo" and its 3 other
    versions from this browser and your account?` (or "from this browser", "from your account" as
    the versions are kept; the Playing card's adds "The game ends and you go back to the start
    menu."), with Delete (in the red of negative money) and Keep; the game being played goes last
    [T-2738; `deleteVersions` in `game/accountGames.ts`, shared with the start menu]. The account's
    copies are recorded as owed before anything is deleted here and leave the card at once; until the
    server takes a deletion, the account's list read in this browser leaves that game out, so a
    deleted version never comes back as a game only on the account. When the server cannot be
    reached, the panel (or the start menu, after the Playing card's delete) says "Deleted from this
    browser. Still on your account until the server can be reached." and the deletion is sent at the
    next load.

**Memory a game takes** [T-3329]: a line under the facts of every card with stations, just "2.5 GB" (hover "Uses about 2.5 GB of memory"), from the
station count (1.64 GB plus 0.8 GB per 1,000 stations plus, per solving worker past the first, 0.1 GB and 0.3 GB per 1,000
stations, as many as the pool's memory rule lets run and, past the first, only while they fit in the budget, T-3373,
T-3757; within about 20% on the real-network saves and players' 53-64 city games). Where the browser reports the device's memory (Chrome, Edge) or is 32-bit, a game over the budget
shows it in red and its Open (saves panel and start menu, not the versions list) first asks "This game needs about 5 GB of
memory and this device has about 4. It may not load." with Open anyway and Cancel. On a phone or tablet whose browser reports no
memory (every iPhone and iPad browser, iPadOS Safari told by its Mac user agent with touch; Firefox on Android) a game of 1,600
stations or more is red and asks "This game is big for a phone or tablet and may run out of memory while loading." [T-3940;
the shipped Tokyo start, 1,454 stations, loads; players' games of 1,646-2,090 died on iPhones, notes/T-3940.md]. Elsewhere (desktop
Safari and Firefox) the line is plain and nothing asks. On the start menu the saved game whose load last died in this browser
asks too, whatever its size: "This game ran out of memory the last time it loaded." (until it loads with its trips), and the
start menu's question has Open anyway and Cancel.
This is separate from the refusal [T-1197], which judges a save by its zones and still blocks a game no worker can hold.

### Card actions

- **Rename** [T-1247]: the pencil or a double click on a card's name (any card) puts a box in its
  place with the name selected; Enter or leaving the box renames, Escape puts the name back.
- **Make a copy** [T-1395]: the copy is made at once, named "<game> copy", and becomes the Playing
  card with its name box open and selected; Enter or leaving keeps what is typed, Escape keeps
  "<game> copy". The page does not reload.
- **Save** [T-1251]: the Playing card's Save, or Ctrl+S anywhere in the game, writes the slot now,
  autosave on or off; then the facts line reads "Saved just now" and the tool notes say "Saved
  9:41 PM". In a tab whose autosave is off because the slot is open
  in another, Save takes the slot over: this tab autosaves from then on, and the other says
  "Autosave is off here because this game was saved in another tab." until a save by hand there
  takes it back (architecture.md "One writer per slot").
- **Games on the account** [T-988; architecture.md 11 "Games on the account"], only while signed
  in to a server that keeps games: after the Playing card's "Autosaved" a small cloud, hover "On
  your account, saved 3:31 PM" (the time the account's copy was saved; "Oct 24 at 3:31 PM" after a
  day), or a crossed cloud in the warn colour, hover "Not on your account yet" plus the reason
  ("Offline", "The server could not be reached", "Your account is full. Delete a game to make
  room.", "This game is too big to keep on your account.") [T-1395]; nothing while another device
  has the game (the panel's callout and the top bar say "Autosave is off here because this game
  was opened on another device.", and Save takes it back, its hover saying autosave stops on the
  other device). Another card whose game another device opened or saved in the last 10 minutes
  says "Open on another device" (the time on hover). A game not tied to the account (kept here
  only, made signed out, or another account's) has a crossed cloud after its name (hover "Only in
  this browser"); on one tied to no account it is a button (hover "Only in this browser. Click to
  add it to your account.") that adds the game, which goes up at once [T-988 stage 3; the start
  menu's cloud is not a button]. A game whose versions are all on the account and none in this
  browser is a card of the server's picture and facts, a cloud after the name (hover "On your
  account. Open downloads it here."), download (hover "Download as a file": the account's copy as stored, also one
  that does not open; "This game could not be downloaded." on a failure, T-4126), bin (hover "Delete", asked first: "Delete "Tokyo" from your
  account?") and Open (downloads it into a new slot and opens it as Open does). Deleting a tied
  slot asks "Delete "Tokyo" from this browser and your account?".
- **Other versions** [T-2037, replacing T-1395's Earlier versions icon; `VersionsLine`]: a card
  with more than one version ends in a line "3 other versions" ("1 other version") with a chevron
  before it; a click opens or shuts the list under it (kept per game for the visit, in the panel
  and the start menu alike). The list is every other version, newest save first, one row each:
  "Saved today at 14:05 · Day 26 · 164 stations" (the day and stations where the version knows
  them), a bin for a slot or an account game (asked first in the row: `Delete this version from this
  browser?`, "from this browser and your account?", "from your account?", with Delete version (red)
  and Keep), and Open. The versions are the game's other slots in this browser (Open opens that slot,
  which stays as it was until changed), its account games not in this browser (a cloud after the
  text, hover "On your account. Open downloads it here."; Open downloads it into a new slot), each
  slot's backups but those that are the slot's own version (its save time, or its latest player change with only the clock run since; up to four tiers reaching back about 10 minutes, an hour, a day and a week, hover
  "A backup in this browser", and the one from before the account's copy last replaced the slot,
  hover "This browser's copy before your account's replaced it") and, signed in, the account's
  backups of each of its account games (time only; a cloud, hover "A backup on your account").
  Opening a backup makes a new slot under the game's name, another version of it. Nothing opened
  writes over another version. A failed download says "This game could not be downloaded." or
  "This backup could not be downloaded.", and one whose copy is damaged "This save is damaged.".
- **A damaged game** [T-3395; architecture.md "Damaged saves"]: its card's facts end in a red line,
  "This copy is damaged, so your account kept the last good one" when the account refused it, else
  "This save is damaged." (also the playing card's crossed-cloud hover for the first). Its other
  versions start open, the first row "Last undamaged copy · Saved Nov 14 at 15:31 · Day 26 · 164
  stations" with "Open a copy" (a new slot, as a backup opens); with none, a muted line under the
  card's foot, "No undamaged copy of this game is left".
- **A game that failed to open** [T-3773; architecture.md "No slot is written over"]: the page is back on the
  start menu, whose grey line says "Cut short could not be opened." once (a file, a
  backup or a viewed network: the reason alone), and the card's red line gives the reason until the game opens:
  "This save is damaged.", "This save could not be opened (<panic>).", or the device's out-of-memory line.
- **A game saved by a newer version** [T-3948, T-4014; architecture.md "Saves from a newer build"]: it does not
  open; the card's red line (and the start menu's line) is "This game was saved on a newer game version. Reload the
  page to update." The same line is the card's red line and the account mark's hover when the account holds the game
  from a newer version, so this build does not upload over it.
- **Leave page?** [T-1826]: closing or reloading the page asks first (the browser's own question)
  only while a change the player made could not be saved: autosave off in this tab for any reason
  above, or a failed write.
- A page opened with `?new=1`, `?synth=1` or `?demandTest=1` does not autosave: the top bar says
  "Autosave is off in this test page. Make a copy to keep the game." (the only place it is said,
  [T-971]), a `?new=1` page marks no card Playing, and the copy icon above the cards gives the game
  a slot of its own, which autosaves from then on.

### A new game

**A new game** starts on day 1 at 07:00 with $10B, empty or (a sandbox) from a real network, on its
start city's registry centre and zoom, or on the world map (centre 20° E 30° N, zoom 1.8: the
Americas, Europe and East Asia) with no start city; a start from the player's location is centred on
it (below). Opening the page shows the start menu (below), with "build 786" muted first in its foot: the
commit count of the build (`__TW_VERSION__`, vite.config.ts), "dev" on a dev server.
The address bar never carries the map position. The tab remembers its last view (sessionStorage), and
a reload starts the map at the registry centre and zoom of the city or zone that view is in, or over
New York with none. An incoming link with a `#zoom/lat/lng` hash starts the same way and the hash is
then removed from the address bar; other URL parameters are untouched [T-310, T-3044].

**The new game sheet** [T-728, T-731; `ui/NewGame.tsx`], the start menu's and the saves panel's:
- **Mode**: Normal (lit to begin with) or Sandbox, two buttons, each with a hover saying what it
  means. A sandbox is the normal game with nothing refused for money and the cash hidden; demand,
  fares, running costs, line money, the Money tab, points and stars are worked out as in normal.
- **Start in**, three ways separated by a small muted "or" [T-1132]. First New York, London, Paris,
  Guangzhou and Tokyo (`START_CITIES`, a fixed list of easy starts, T-2690) and Current location, one lit at most; New York
  is lit to begin with, so Start reads "Start in New York" [T-2123]. A click on a lit city keeps it
  (Choose on map is the way to no city); a second click on Current location unlights it. Current location shows only where the browser can give a location (a secure
  page with geolocation); the browser asks the player itself. While it waits the button says
  Locating; a refusal, an error or no answer in 15 s leaves the sheet as it was with the button
  saying "Location unavailable" (a click asks again; an answer after the 15 s still counts). A
  location starts in the zone whose coarse outline holds it, else the zone with the nearest centre
  among those whose box comes within 40 km (`ZONE_REACH_M`), the camera on the location at the zone's
  registry zoom; farther from every zone it is a start on the map centred on the location at zoom 4
  (`zoneForLocation`). Then "Find a city", which lists up to eight of
  the registry's cities and zones whose name has a word starting with the text (case and accents
  ignored, so "hague" finds Rotterdam [The Hague]), biggest by local trips first, each with its
  start tier in light lowercase text after the name, "easy", "medium" or "hard", whose hover says
  "Many (Some, Few) trips begin and end within 10 km of the centre" (`game/startDifficulty.ts`;
  zones.md "Start tiers") [T-2690], and its
  country (the registry's code, named by the browser). Up and down move, Enter or a click picks;
  a pick of one of the five lights its button. Then Choose on map, lit while nothing else
  is picked: the game starts on the world map (its tutorial card begins with "Zoom into a city") with Homes on and every realm shown (zones.md
  "Realms"), so the zones' zoomed-out bubbles show and take a click as anywhere (ui-demand-views.md
  "Zoomed far out").
- **Real world network** (a sandbox only): a list of every city with a real network, by name
  (`game/realNetworks.json`, written by `pipeline/realauto.py ship`; architecture.md 11), blank to begin
  with, so the player picks one on purpose [T-1302]. Picking a city makes it the start; picking another
  start in any other way blanks the list again [T-3225]. The game starts from that network, built and
  running; the hover "Start with Tokyo's real rail lines already built." shows once a network is picked.
- **No running side control** [T-3367; `game/runningSide.ts`]: the sheet never asks. The game's running side, for
  trains, trams and buses alike (world.md "Running side"), follows the start: the side its country's railways keep
  to, a real network's its metro's; on the world map (and a location outside every zone) the player's own country
  from the browser's time zone, else its language's region (`homeCountry`, no permission prompt), else Right. Settings'
  "Trains run on" changes it later. A real network started on the
  other side than its save is switched as it opens.
- **Start in Tokyo** (the zone picked or located) or **Start**, bigger than the other buttons (36 px
  high, bold, tinted) and centred in the sheet's last row, with Back at its left [T-1215]. If a real network cannot be downloaded, a
  line says so and the sheet stays.
- The slot is named after the start ("Tokyo", "Tokyo sandbox", "World", then " 2" and on).

## Account and leaderboards

[T-972, T-973, T-988; `game/account.ts`, `game/accountGames.ts`, `ui/AccountPanel.tsx`,
`ui/Leaderboard.tsx`; server routes in shonei-server's `trainworld.go`, `trainworld_games.go` and
`docs/trainworld-api.md`; notes/T-972.md, notes/T-988.md]

- **Optional.** Signing in is never needed to play. The account is noritetsu's (Google, or a
  username and password with an optional recovery email), kept under noritetsu's storage key, so on
  anita.garden one sign-in serves both pages. The server is market.anita.garden on anita.garden; on
  localhost none unless `?acct=local` (a shonei-server on 127.0.0.1:8083) or `?acct=live` was given
  once (remembered per browser, `?acct=off` forgets it). The icons show only once
  `/noritetsu/config` answers with `trainworld`; with no server, or an older one, there is no icon
  and no other request. Google's script loads when the account panel first opens.
- **Account panel**: the account button (top of Settings, start menu; not in the tab bar, T-1106), or Settings' Account row (T-1139), opens a centred modal like the
  saves panel. Signed out: Google's button, then username and password with Sign in, Create
  account and Forgot password; Privacy and Terms (`public/privacy.html`, `terms.html`). Signed in:
  "Signed in as alice" (or with Google); under it, while the server keeps games, "3 games on your
  account · 1.2 MB of 64 MB" (hovers: "Your account keeps up to 50 games", "Backups count too")
  [T-988 stage 3]; then, when the games list carries `played` (seconds), "3 h 20 min played" and, when `/trainworld/me` says
  `nameHidden`, a red "Your name is hidden from others. Pick another." [T-3230]; then the **flag and display name**
  (T-3269, T-3279, as City Circles' quiz; when `/noritetsu/config` says `trainworldFlags`): the flag, or an empty 4:3
  outline while there is none, then the leaderboard name and, once the account has a name and the server says
  `trainworldRename`, **Change name** at the row's end (T-3807): a field with the name in its place, Save and Back;
  the server allows two changes a day (a first name, set by Join or Share, is not one), and past that the row says
  "You can change your name twice a day. Try again in 5 hours.", as Join and Share do when they would change it. The flag opens a picker over the page (300 px wide, up to
  480 px tall, under the flag or above it when there is more room, free to reach past the panel): No flag (only while
  one is picked), the country the player is probably in, then every country, then other places (the EU, Amsterdam,
  Chicago, Washington, D.C., New York City and Japan's 47 prefectures, each with its country in grey), then the game's
  own few last (Train, Earth, Pride, Transgender, the UN, Pirate, Checkered); a gap between groups, each by name in the
  player's language. Typing filters (start of any word of the name or of a place's country, or the code); a pick is
  saved at once; Esc or a click outside closes it. The probable country is the browser's time zone's
  (`game/tzCountry.json`), else its first language's region. Codes from a newer flag list than the server's
  (`flags.json` `set` against config `trainworldFlagSet`, 1 when absent) are left out of the picker. The list is
  `game/flags.json`; the pictures `public/flags/`, all 4:3, and the server's list are written by `tools/flags.mjs`
  (sources in notes/T-3269.md). For a password account the recovery email ("No recovery
  email" in red when there is none) with Save email, Remove and Change password, then Sign out and
  Delete account (asked first: "Delete your account? Its games, its leaderboard entries and its
  noritetsu rides go too. Games in this browser stay."; the browser's slots are then tied to no
  account). Saving or removing the recovery email and deleting ask for the current password, and a
  Google account deletes through Google's button instead ("Confirm with Google to delete it.") [T-2917].
- **Signing out** [T-988 stage 3, T-1851] signs out at once; the account's games stay in the browser,
  tied to it, and sync again at the next sign-in. The "Remove games" choice (`signOutRemoving` in
  `game/accountGames.ts`) is no longer offered (Anita, 2026-10-30).
- **The sign-in question** [T-988]: signing in, or a load already signed in, with slots tied to no
  account (made signed out, never answered) opens the account panel on "Add the 3 games in this
  browser to your account?" ("Add Tokyo to your account?" for one) with Add (lit) and Don't add;
  Don't add asks "Are you sure? Games kept only in this browser are lost if its storage is
  cleared." with Keep them here only and Add them ("Keep it here only", "Add it" for one). Added
  games go up at once; kept ones are never asked about again and carry the crossed cloud.
- **Scores**: while signed in, each normal game in a browser slot sends its points (the forecast weekday's, the All cities row's figure, T-3465), its best 64 cities' points and (T-3879) its weekday profit in US$ (`dayProfit`: the weekday's fares minus running costs and upkeep, the edit note's profit figure; no new trains, construction or loan), with (T-3885) the same for each of those cities (`dayProfitByRegion`), when they change and at
  most every 2 wall minutes (at once on the first send and when the page is hidden), joined or not.
  A `?new=1` page or one whose slot is open elsewhere sends nothing. Each slot has a random game id
  in localStorage (`trainworld.score-games.v1`), or the id of its game on the account when it has
  one (T-988), so one id names a game's save and its figures. A game that turns sandbox sends that once and is
  never listed again; one that was always a sandbox sends nothing. A slot deleted in this browser
  takes its figures off (on the next signed-in load if signed out then).
- **Modded games** (T-3707, `game/modded.ts`): a game played in a page that is not the published game is marked
  modded in its save (`modded`: the reasons, never removed) and never ranks: the page sends the server's latch once
  (`{mode: "sandbox", modded: true}`, even with no figures sent before, so an older build can't list it either).
  Marked when a mod calls `trainsworld.markModded("<name>")` (always defined; the page's own mark goes on every
  game it plays after, the open one and any loaded or started), when the page is served from a site other than
  anita.garden or a local or home-network address, or when the build came from the public repo
  (`unofficial-build.txt`, which tools/opensource adds). A build stamp is never compared page side: 8800 and dev
  servers carry unpublished ones. The mark survives a file, Make a copy, a backup, the account and a share's copy.
- **Only the published game's scores** (T-3707): each normal send carries the build's stamp (`build`,
  `__TW_BUILD__`). The server keeps them only from anita.garden with a stamp on the site's `builds.json`, which
  `tools/deploy.py` writes at each publish; a refusal (426 `unpublished`) stops this page's sends for 10 minutes.
- **Leaderboards**: the trophy opens a centred modal. Against a server whose config says `trainworldProfit`, a
  Points / Profit switch sits at the top (T-3879, T-3885): Profit ranks weekday profit on the same places as points
  (the shown currency, short; "$1,904,120,812 profit per weekday" in the hover), and only a figure above 0 has a row; a
  city's is the game's profit split as the region ledgers split money, a country's its cities' summed, losses included.
  The switch keeps the place picked. Then a picker (World, then up to two countries in
  this game, then its cities by points; typing searches cities and countries, a country by its name in
  the player's language or in English), the top 50 (rank, name, points short, the
  exact figure in the hover; the person's flag after the name, 16 x 12 px, its name in the hover; a long name is cut, the flag never), the player's own row under them when lower, "Nobody here yet." on an
  empty board. At the bottom: Sign in to be listed (opens the account panel), a name (at most 24
  characters, the map games' name filter on the server) with Join, or Leave the leaderboards (the
  name is kept); in a sandbox game also "Sandbox games are not listed", in a modded one "Modded games are not
  listed", and after a refusal "Scores from this version are not listed. Reload the page for the newest.". Each person shows once per
  board. A game counts at the figures it last sent, not its best (T-3248), sent only from a whole final solve of the
  network as it is (no solve running or held, never the past 24 hours' stand-in, T-3639; the score on screen takes the same finals, `scoreSettled`, T-3644); on each board the person's
  row is their best game there. Anita's feedback inbox has a Leaderboards tab that sets a cutoff date:
  games last sent before it count on no board until played again. A country board ranks a game by its points summed over that country's
  cities (the same per-city figures the city boards use; the server maps city to country with a table
  made by `tools/country_table.mjs`) [T-3116].

## Community

[T-2887, T-3114, T-3115, T-3132, T-3144, T-3224; `game/community.ts`, `game/sharedSave.ts`, `ui/Community.tsx`, `diagram/shared.ts`;
server `trainworld_community.go`, its `docs/trainworld-api.md` "Community"; notes/T-2887.md]

- **Shown only when the server says so**: `/noritetsu/config`'s `trainworldCommunity`, or in preview its
  `trainworldCommunityPreview` on a page opened with `?community`. Otherwise no button, and a share link opens nothing.
- **Community** (people icon in the head of the saves panel and of the leaderboards, and in Settings between Help
  and the account button), headed "Community trains worlds", with the upload icon in its head to share the game
  being played. [Random] [Newest] [Biggest] [Most liked] [Bookmarked] in [place] (T-4005: Bookmarked has no place or search; each card and opened network has a bookmark button, kept in the browser and, signed in, on the account, merged once per account; a hidden or deleted network drops out of the list): Random is a shuffle (Shuffle for another); Newest (T-3652, shown only when the server config says `trainworldCommunityNewest`) is the newest network first, one per person, paged like Random; Biggest is
  by points in the place, one per person, never a sandbox game (a note under the list says "Sandbox games aren't
  counted"), top 50 ranked as the leaderboards are; Most liked is
  by likes, equals in random order (the panel's seed, kept while it is open, T-3258), one per person, top 50. The place is the world, a city, a country (as
  the country leaderboards have it) or a region: a realm (`game/realms.ts`, the names the map's realm buttons use)
  that is more than one country or part of one ("Benelux", "US west"). Typed, the picker finds every city, country
  and region; empty, it offers the world and the playing game's countries, regions and cities. A share is in a place
  when its game has points there, valued at their sum there. Twelve cards a page, each a shared schematic painted
  small with its title in bold and the player's name after it in grey, their flag after the name, smaller (12 x 9 px) to match it (a share from before titles shows the name alone, in bold, the flag after it at 16 x 12),
  its biggest city in the place, points, lines and stations (the game's own counts,
  bus and ferry included, as the saves list shows them). A card opens the drawing to pan and zoom, with the
  **Calculate trips** tick and View (both when the sharer let others open a copy), Copy link, Report and the heart.
- **Search** (T-3224): a box at the right end of the tabs' row (on a phone, beside the place picker), placeholder
  "Search", shown only when `/noritetsu/config` says `trainworldCommunitySearch`. Sent as the list's `q` once typing
  pauses 350 ms; matches the title or the sharer's display name, any case, accents folded, in the chosen tab and
  place. A search lists every matching share, not one per person, so a name finds all of that player's networks; a
  rank is still the share's place on the whole board, so only a player's best has one. Clearing it (or Escape)
  returns to the normal page; nothing found says "No networks match that search". Works signed out.
- **Likes**: every share starts with one, its player's own, which they cannot take back; on their own share the
  heart shows filled and disabled, hover "This network is yours.".
- **Share again from Community**: an opened share of the player's own has Share again. For the game being played
  it opens the Share dialog; for another game in this browser it opens that game, then the dialog once it plays
  (the wish rides the reload in sessionStorage); for one not in this browser it says "Open this game in Saved games
  to share it again.".
- **Share network** (people icon on the playing game's card and beside Save picture on the schematic, upload icon in
  Community's head; pink once shared, hover "Last shared <date time>"), signed in: "Anyone can see this network,
  your name and its points.", "This save was last shared on <date time>." once shared, how many shares are left
  today ("You can share 3 times today", "... 2 more times today", from the server's count), a name field when the
  account has no display name, a title field (starts as the game's name, 40 characters at most; blank sends the game's
  name; the server refuses a title or display name its word filter blocks), the **Let others open a copy** tick (on by default; a share again keeps the tick as
  it was last sent), Share network or Share again, Stop sharing, Copy link once shared, and the people icon to
  Community.
- **The link**: `<the game's address>?share=<id>` opens the game (start menu or not) with Community on that share;
  works with no account. The switch leaves the address bar at once. A hidden, banned or deleted share's link
  says "This network isn't shared anymore." over the Random page, the same for each.
- **View** [T-3257] downloads the share's save, checks it (`game/sharedSave.ts`: sizes capped before and while
  unpacking, the header walked with limits and each field's type checked, unknown fields dropped, the file
  re-encoded from what passed) and loads it as a **viewed game**, named "<title> - <their display name>" (either
  alone when the other is missing); the game in hand is saved first, and none of the viewer's games is written. A
  viewed game is in no slot: it plays and edits as any game, but nothing autosaves it, it is not on the account, and
  the saves panel shows it as its own card at the top (below). Save (the card's, the top bar notice's, or Ctrl+S)
  makes it a saved game under that name, the Playing card from then on. Until then, anything that would replace it
  (another game opened, a file, a backup, New game, View on another share, the main menu door) first asks (headed
  "Confirm?") "Guangzhou - kanocifer won't be saved." with Leave without saving (red) and Stay; closing or reloading the
  tab asks (the browser's own question) only once the player has changed something, as for any unsaved change. A
  save this build cannot open says "This copy doesn't open in this version of the game." (another world, a retired
  or newer format), a damaged one "This copy can't be opened."; a load that fails after that is reported as any
  loaded file's is, and the page falls back to the game it had.
- **A light view** [T-3851]: with the Calculate trips tick off (the default; the tick is remembered in this
  browser), View opens the network without its trips: the page reloads with `?trips=off`, no demand worker starts
  and no city pack is opened (only the clock worker's water masks and terrain come, as in any load), and the load
  ends once the network is in (notes/T-3851.md has the times).
  There is no memory check, since the demand workers are what a big game's memory goes to. Track, stations, lines
  and trains show and run (empty; money is frozen: no fares, running costs, upkeep or loan payments, so cash stays as
  saved, and stays so in a game saved from there; edits cost as usual), the heart under the day and night picture
  works, and the saved figures (the ledger, the Cities tab's profit) show as saved; riders, trips and points stay
  empty, with no loading rings or "Calculating" (a train's riders say "Not available"), and the map's demand
  bubble and station rows are hidden. The top bar says "Asia - kanocifer isn't saved.
  Trips aren't calculated." with Calculate trips and Save; saved from there, "Trips aren't calculated for
  this game." with Calculate trips. Calculate trips opens the same game again with its trips, the memory check
  first: a viewed network goes along with any edits (still viewed, its heart kept), a saved game is saved and
  reopened.
- **The viewed game's card** (saves panel, above the others): its picture and facts as any card, "Not saved" where
  the others say when they were saved, the mark Viewing where the game being played says Playing, then only
  Download (hover "Download as a file") and Save (hover "Keep it in your saved games (Ctrl+S)"). The top bar says
  "Guangzhou - kanocifer isn't saved." with Save. Its play time counts from its Save, as a slot's does.
- **A copy never ranks**: its header carries `copy` (the share's id), the saves list marks it Copy (hover "Never on
  the leaderboards"), it sends no figures and no achievement counts, and sharing it tells the server it is a
  sandbox, with the reason (copy), so it shows tagged Copy and only in Random (a real sandbox shows Sandbox). It plays as a normal game otherwise.
- **What goes up with the tick**: the save as the game writes it, minus the Getting started card's progress, and the
  game's name beside it (through the server's name filter on upload and on read, blanked if it fails; it names the
  copy and is shown nowhere else). Account and device ids, playtime and settings are never in a save. Line and group names are, and
  only there (the drawing has none): Anita's inbox can download a reported share's save to read them.

## Schedule panel

[T-3346 stage 1, T-3469, T-3475, T-3521, T-3817; `ui/inspector/SchedulePanel.tsx`, `TrainTable.tsx`, sim `TrackApi.station_departures`; notes/T-3346.md, notes/T-3521.md, notes/T-3817.md]

- **A line's**: opened from the bigger icon by a running line's schedule heading (a station timetable,
  hover "Detailed schedule"); a centred modal in the saved games panel's frame, up to 1320 px wide; closed by its
  close button, Esc or a click outside it. Heading: the
  line's bullet and name, "schedule" in grey, then the mode choice, drawn only once Detailed and
  Exact are built (stages 3 and 6; Simple alone shows no choice). A player who never opens it sees nothing else new.
- **Left**: the line panel's schedule table, its steppers, dots and hovers, under a "Set by" Trains / Headway
  switch; an edit here is the line panel's edit. **Periods** (T-3542; a bus or ferry line's window too, T-3767): under Set by,
  "Periods" 3 / 4 / 5 sets every line's rows for the whole game (lines with their own setting aside); "Set for this
  line" at the end of that row (hover "By default, periods are set globally. Click to set just for this line", T-3795)
  gives the line its own (at the rows it has), shown as a "This line" 3 / 4 / 5 under the game's with "Same as game"
  at the end of its row. A change of periods shows its rows before the trains and riders catch up (T-3795). Three rows are High, Medium, Low under "Demand" (as before); four are
  Rush (06:00-10:00 and 16:00-20:00), Midday, Evening, Night and five Morning, Midday, Afternoon, Evening, Night,
  under "Period", each name's hover its hours ("Rush: 06:00-10:00, 16:00-20:00"). **Hours** (stage 2): a column
  "Hours" after the names in the Weekdays rows, each row's hours in a box ("6-10", rush "6-10, 16-20", an evening
  past midnight "20-1"), typed in whole hours; the game's, or the line's own once it has its own periods. Hours that
  leave a row under two hours (high on three rows under six) are refused: the box shakes with the refusal's knock
  and a red line under the row says "2 hours at least" for five seconds. The row now (bold) and the names' hovers
  follow the hours; the window's left side is 330 px wide. A switch to fewer rows (the game's,
  this line's, or Same as game) first asks "Confirm?" with the rows it joins and what they run: "Morning, midday and
  afternoon will run 26 trains, the most of the three" (a headway line: "will run every 4 min, the shortest of the
  three"; the game's: "... will run the most trains of the three on every line without its own periods"), yes "Use 3
  periods" in red, no "Keep 5 periods". The line panel shows the line's rows (weekdays only) and nothing else of periods. The table has two sections (T-3521), each a bold heading row
  with room above it: **Weekdays**, the line panel's rows, and **Weekends** (hover "Weekends and public
  holidays"), a row for each weekday row with the same steppers on the line's weekend counts (under headways a headway
  stepper, the count beside it; the headway stepped to is kept with the counts and the weekends run it, T-3557),
  its headway the rate settled for weekends. Beside the Weekends heading the
  line's percent of its weekdays in a stepper (5% a click, typed 10-200), greyed with the hover "Weekend service
  set by hand" once a weekend count is stepped, and "Apply to all lines", which asks "Set weekends to 70% of weekdays on every line, and on new lines too?" first (new lines start on that percent, `SaveHeader.newWeekendShare`, T-3588). The level in force is bold in today's section only; each section's dots are its
  own kind of day's (today's, else a weekday's or a Saturday's). A stepper's hover counts the fleet over both
  kinds of day. Under the table, **Headways for new regular, tram, high-speed, bus or ferry lines** (the heading names the selected line's kind): the headways every new line of that kind starts with, "Default not set" until the player sets one (a new game has none, so a first rail line runs on train counts; without a default a rail line starts on counts, which give no fixed headway to show; a bus or ferry line shows its tool's headways until one is set, 8, 15 and 30 min for a bus, 20, 30 and 60 for a boat, `NEW_LINE`), a "Set as default" button that makes this line's headways the default for its kind (a line on counts is switched to Headway first at the headways its trains give, so it runs the same service; off while they already are) and "Use default", which gives the line its kind's default (off while there is none or the line already runs it). The row uses the table's 12 px text
  (T-3767: the list has one headway per row of the selected line; a default set from a line of three rows is high, medium and low, and read at four or five rows runs high's in the morning, at midday and in the afternoon; one set from a line of four or five rows keeps a headway per period, and a line of fewer rows reads each row as the shortest of its periods'; a new line takes one for each of the game's rows) (T-3543, T-3588, T-3703; saved with the game, `SaveHeader.newHeadway` {rail, tram, fast, bus, ferry}; the clock worker applies them to each new rail line after it is
  made, as a second edit, a line that cannot pay for the trains they ask staying on its counts, and makes a new bus or ferry line at them; a new bus or ferry line takes the weekend percent for new lines as a rail line does). Then the departures board.
- **A bus or ferry line's** (T-3703): the left side alone, in buses or boats, in a panel 360 px wide: the periods,
  the table with its Weekdays and Weekends sections and its hours, and the default headways, with no track limits and no departures board or
  timetable (the sim keeps no trip timetable for bus and ferry lines).
- **Right**: the timetable over the chart.
- **Departures**: headed "Departures" with a station picker (the line's stops in order, each station once, the first stop to start), then
  the board: one row per local hour of the line, from 4:00 through to 3:00, kept to the hours from the first train to
  the last, each direction's minutes in its own column headed by where its trains go ("to Gum"; on a loop, both
  ending where they start, the next stop that way, "via Birch"). The hours are the next 24 from the start of this
  one, so each row is the line's timetable as it stands now; the hour now is tinted and scrolled to the top. Only a
  run's leaving times count (a run's last stop is no departure), and a line on both pairs of quad track shows its
  hidden twin's trains too. Times come from the sim, each trip's own (`station_departures`: its `draw` plus its
  profile's `dep` at the stop), the trips the chart and map draw. A station the line keeps to boarding only or
  disembarking only (T-4013) has that named after the picker, muted with its icon ("Boarding only"); its departures
  stay, since trains still leave it.
- **Timetable** (T-3469), as a printed one: one direction at a time ("to Gum" / "to Ash", the way the chart reads
  top to bottom first), with the other lines toggle right after that switch, as tall as it (T-3819); stations down
  the side (a boarding-only or disembarking-only stop with the line panel's icon after its name, hover naming it, T-4013), one column per train under its line's bullet (a bullet too wide for a column becomes its letters down a
  strip of the line's colour), times in 10 px type as "742" / "1008" in the line's colour (`legibleOn`), the
  departure at each stop and the arrival at the last, "|" where a group's train runs through a station; a rule every
  5 rows, time marks along the top, a past train faded, the time now an accent line. It covers the hour before this
  one and the day after, opening with now a quarter of the way in. Each column stands at its train's first time.
- **Zoom** (T-3817): the wheel over the table or chart zooms both about the pointer (1.25 times a notch, coalesced to
  one step a frame), from the whole day across to a quarter hour; over the station names it scrolls the rows. A
  sideways wheel or Shift-wheel, a drag (mouse) or a swipe (touch) moves the view. It opens at the coarsest zoom
  level that draws at least 95% of the day's columns (no closer than an hour across) and keeps the
  player's zoom while the panel shows the same line. **Columns that fit**: only the columns that fit side by side
  at the zoom are drawn; the rest are left out. The choice is made over the whole day in zoom levels (columns at
  least 10 s x 1.41^n apart, n the finest level whose gap is a column's width): each level keeps the coarser one's
  columns and adds, left to right, each that clears them, so scrolling never changes the set and zooming in only
  adds columns. **Bullets**: every train's bullet is drawn, overlapping where they must, each with a 1 px rim in the
  panel's background (light in light mode, dark in dark) so a pile reads as many; the drawn columns' bullets lie in
  front. Hovering a bullet (the nearest within 9 px, so one under others is reached too) or a column lights that
  train over a veil: its bullet in front and its column, covering the drawn columns it lands on, whether or not it
  is drawn at the zoom; its path on the chart too.
- **Related lines** (T-3818, T-3846): while the other lines toggle is on (it shows whenever the line has any), the trains of
  each line the line panel's related lines draw beside it (`lineRelated`, kind beside; not its own quad twin, nor a
  group's own lines) that runs on the line's track at any point, or shares at least 3 stops with it (a complex counted
  once) that are at least 40% of either line's stops, get columns too, at the rows where both stop (a station's complex counts), "|" only within a
  stretch it runs beside. They never take a column from the line's own trains: at each zoom level they fill only
  the gaps the line's own columns leave, and the opening zoom counts the line's own columns alone. Their bullets sit
  2 px lower and behind the line's own (related behind, own behind, related drawn, own drawn; related bullets behind
  kept one per 3 px); their bullets and times at 50%, 25% once gone, 50% when lit. Hovering a bullet
  of another line names the line where its letters don't already. On the chart they are drawn as other lines, stop
  to stop where they have no shared track.
- **Chart**: the timetable chart ([ui-inspectors.md](ui-inspectors.md) "The timetable chart") under the timetable,
  its stops 240 px tall, scaled by distance, on the table's time axis: it zooms and scrolls with it. It draws the
  table's direction in front and the other direction dimmed behind, every line of a group alike (another line's way
  by most of the track it shares). Hovering a train, its column or its path, lights it in both and fades the rest (a
  veil over the table, the chart's trains at 35%).
- **A group's**: from the group view's timetable icon (the line panel's, hover "Detailed schedule"), for a group with two or more running rail lines. Heading the
  group's pill and name, then "schedule"; the body is the timetable and chart at the panel's width. The table's rows
  are every station its lines stop at, each once (a station's complex counting as the station), in the group view's
  order (T-4276, `groupRows`): the longest line's (by length) stops, each other line's own stations a block where it
  leaves and rejoins them (a branch), a line sharing none below. Every train of the group is a column under its
  line's bullet, its times in its colour. A train shows "|" at a station it passes, or where another line stopping
  at both its stops either side stops (an express beside a local), and nothing on another branch. The chart draws the
  longest line, the others at full strength in their colours where they share its track (T-2660's mapping); the
  other lines toggle adds lines outside the group on the longest line's rows, as the line chart does.

## Start menu

[T-677; `ui/StartMenu.tsx`; the startup order in [architecture.md](architecture.md) "Loading and readiness"]

- Every page load opens on the logo (below) over a card on the bare map, in the loading card's look:
  no tagline [T-1710], New game at the top with, at the right of its row, the account button when accounts are on
  [T-972] and Community; the new game sheet has the account button at the right of its heading; then "Saved games", one card per game as in the saves panel (T-2037: the version with the latest player change on top, the latest save first), each a button with the
  saves panel's picture (the same 240 x 160 with its names) and facts (name, when saved, day, cities, lines, stations, cash), and under it, joined to it, the saves panel's "3 other versions" line and list (above).
  Signed in (T-988), the slots are first settled against the account (a click on a card waits for
  that, 10 s at most), and a game only on the account is a card with a cloud (hover "On your
  account"); a click downloads it into a new slot and opens it. A backup opened from the list waits
  as a file does (`pending-load`) and opens in a new slot.
  With no saved game and no game played here before (a first visit) it opens straight on the new
  game sheet [T-1441, T-1612]. When browser storage keeps failing to list the saved games (four tries
  over about two seconds, as a slot's read at startup), the menu opens on New game with the grey line
  "Could not read your saved games from browser storage. Reload the page to try again." in place of
  the cards, never on the sheet as for a first visit [T-2705]. The dock and top bar are taken out of the page and the
  map controls hidden, so the map fills the page; it is centred around 10° E, 20° N one zoom closer
  than the zoom at which the world from 60° S to 75° N covers the page [T-927, T-1593;
  `worldCover`], with the zone outlines, outside shade and every basemap name hidden until the pick [T-974, T-1567]. The menu takes every key outside itself, and nothing loads behind it but the map.
- Over the card (and the new game sheet) the logo [T-1567; `ui/StartLogo.tsx`,
  `map/logoNetwork.ts`], after Anita's sketch: "trains" over "world!" in joined-up script slanted
  forward, the t turned over about its own middle so its lead-in is a bar across the top, a capital T (the r goes on from the stem's foot) [T-1812], its curves of one radius before the slant, switchbacks where a curve would be tighter,
  each stroke a line with stations at its ends and joins (boxes overlapping where lines join or
  cross, as a complex's do), and one train a line, a white pill, running with stops (the two dots and
  the t's bar are bare lines, no stations or train) [T-1593], the boxes 1 px smaller on every side
  than the platform so a parked train covers them, and 1 px shorter again at each end [T-1700, T-1711], the dots drawn as round circles at their placed centres, drawn by the game's own
  network renderer at zoom 15 in a canvas of its own straight over the map, scaled to the screen; the
  menu's dialog is named "trainsworld". The trains are the only motion; with reduced motion asked for,
  one still frame. `?logo=0` leaves the logo out: the card is then headed "Welcome to trainsworld!"
  and only city, town and village names are hidden.
- **Foot** [T-3630, after Anita's pick of the footer line, 2026-11-18]: one line at the bottom left, on the menu and
  the new game sheet alike, 12 px muted: "build 786", About this game, Submit feedback (opens the feedback form, as
  in Settings), the language picker (the Settings row's globe and select; here only, so a new player who lands on
  the sheet finds it in the same place, T-2753) and a gear (hover "Settings") that opens a small panel over the line
  with the settings that matter before a game loads: Calculate intercity trips, Calculate trips in less detail, Use more cores and Panel and text
  size, the same controls as in Settings; a press outside it or Esc closes it. The line wraps on a narrow screen;
  it and the update row stand in one column at the bottom, the row above, so neither covers the other.
- **Update row** [T-2810; `ui/updates.ts` is the one list of updates, newest first]: at the screen's bottom right above the foot (and so above the map attribution), a compact row framed in the theme colour, with 12 px muted text throughout, the newest update's one line, a Changelog button (a Modal listing each update's version, date and changes; each change has its area's icon in the area's colour, the list sorted by area with an area's bug fixes after its other changes and a little lighter, and a group such as "track bug fixes" lists its changes indented under it [T-3466]; one page per x.y version, v0.3 holding v0.3 to v0.3.5, picked with buttons under the title, the newest open [T-3468]) and an x that hides it for good in this browser. It shows for 2 days from the update's `at` (bumped at publish), not on the new game sheet. On narrow screens it can cover the saves; its close button stays visible.
- A click on a save opens that slot (a sandbox's name has the "Sandbox" mark). New game shows the
  new game sheet (above) in the card, headed "New game", with Back to the menu; its Start starts the
  game in a new slot, so no saved game is replaced. The slot is first written once the network has
  something in it (a real network at once): a new game left empty leaves no card.
- **Delete and Load from file** [T-1612; the same code as the saves panel's: `deleteVersions`,
  `deleteGroupQuestion`, `pickSaveFile`]. Every card has the panel's download and bin at its bottom right (hovers
  "Download as a file" and "Delete"). Download saves the card's top version as stored, from this browser or the
  account, also one that does not open (damaged, or from a newer version), for a player to send in [T-4126]. The bin asks under the card, `Delete "Tokyo" from this browser?` (from this browser and your
  account when the slot is tied to it; `Delete "Tokyo" from your account?` for an account-only card;
  `Delete "Tokyo" and its 3 other versions from ...` with more than one version, T-2037), with
  Delete (red) and Keep, and the card goes at once with every version, the account's copies with
  it (now, or at the next chance, as in the saves panel). A slot open in another tab says so and stays. Load from file (beside New game) is shown
  to a returning visitor only: one with a saved game here or on the account, or whose browser has played
  a game before (the remembered last slot, even once every save is deleted); a first visit sees only New
  game, and a returning visitor with no saves opens on the menu, not the sheet. A .save file goes to a
  new slot named after the file and opens with its stations in view, as Open a file does in the saves
  panel; one that is not a save, or too big for the device (T-1197), says why over the cards and nothing
  opens.
- A saved game with a home (set by the home button of a city view or the all cities view,
  [ui-inspectors.md](ui-inspectors.md#the-city-view); kept in the save, T-2395) opens there: a zone's registry
  centre and zoom, or for all cities every built station (all stations if none is built; the world map with
  none) fitted in view, padding 70 px, no closer than zoom 6.75, just under where the top bar turns to a city.
  A saved game with no home (older saves too) opens on the city view (its registry centre and zoom, as picking it in the Cities
  tab) of the zone with the most built stations (of all stations if none is built; the stations'
  zones by `placeAt`), as soon as the clock worker has read them from the save (`stationsAhead`,
  T-1539), and again at its first state (the same view), before its demand [T-1872]; stations in no
  zone are fitted in view instead (padding 70 px, no closer than zoom 12). A save with no stations opens on the zone it started in, else
  the world map. The view is not saved. A pick from this menu, a slot opened in the saves panel and
  an imported file fit; `?continue=1` and a raw `import` message (checks) keep the camera
  [T-895; `startOpenFit` in `ui/locate.ts`, `fitNextGame` in `workers/clockClient.ts`]. Once the
  player has moved the camera since the pick (a drag, the wheel, a pinch, a rotation, the arrow keys, a
  search pick or a Cities tab double click), the game being entered fits it no more, so the player can
  look around while it loads [T-2477; `enterGame` in `ui/locate.ts`].
- URL switches that already say which game skip it: `?new=1` (a new game owning no slot; normal
  in New York, or `&mode=sandbox` and `&city=<zone id>`),
  `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` (the game opened last).

## Loading

[T-149, T-172, T-180, T-212, T-232, T-243, T-318, T-793, T-884, T-1194 to T-1197; `game/loading.ts`, `game/loadBar.ts`, `ui/LoadingScreen.tsx`]

- Nothing is blocked: the map, inspectors and building work as soon as the network is drawn.
- **A loaded game** (the page opening, a file, a slot, a new game) holds only the clock until the
  zone in view has its final figures [T-347, T-421]: its fresh solve's last round, or that of the
  zone's own cluster, solved ahead of the rest (T-402: a Toronto view of a game with New York's
  network, about 6 s instead of 31 s); a new game until its viewed zone's figures, opened bare,
  arrive (T-351). **A game whose last final result is kept for its very network** (T-328: the
  same network and packs; after an update, the last version's result stands in [T-4202]) holds the clock only
  until that network is in, the result shown and the zone in view with figures [T-2560]: its figures are that
  network's final (a stand-in's the last version's, replaced at the fresh final), so the game plays on them while
  the workers solve the same network again. The status tag then says "Trips in 81 cities being recomputed…"
  until the zone in view's cluster has its fresh final (the tag's own rule in "Status and busy
  tags"); the clock is not capped for it [T-2622]. Zones still solving then show their own loading bar and grey ("A zone still loading
  is grey" in [ui-demand-views.md](ui-demand-views.md)) until their final lands. The time is
  greyed and its hover says "The clock starts when demand has loaded". Meanwhile a card in the
  middle of the map names the stage in hand over a filling bar [T-1194]: "Loading" (the game itself,
  until its network reaches demand), "Loading city data" (the packs downloading and opening in every
  worker) and "Calculating demand" (the solve to the zone in view's final), naming no cities and
  counting nothing (the packs open in one batch per worker, so no count could move). The percentage is
  only the bar's hover ("42% loaded"). The card has no Cancel (T-1236: a reload does the same). It
  takes no input but the bar's hover (a press elsewhere pans the map beneath) and goes when the
  clock starts.
  Loading is the one place with motion [T-647, T-677, T-2693, Anita; `ui/Spinner.tsx`]: on top of the card a
  drawn bunny rides a toy train in the theme's accent (104 px wide; wheels turn, the track's ties run
  by, the train chugs, the ears bob; each moving part a box of its own moved by a CSS transform, so its motion never
  repaints the page, T-3160; paused under reduced motion; the card alone has it, T-2693 follow-up); a pill drawn as the map draws a train (a rim in a darker shade of the theme's accent
  around a fill of it) runs around a 13 px ring leading the status tag's text and the diagram tag (a CSS
  turn of 1.1 s, nothing per frame; paused under reduced motion); a small ring turning (`.spin`, 1.2 s a turn, 8 px, grey) in place
  of each line's crowding dot per demand level in the line panel until the line has figures
  (while no final result is shown or a solve is updating it; not after a failure or over dead
  space); a line whose shown result has no riders for it as its stops are now (just given trains,
  its stops just changed) has the ring, never a dot, until one that does lands, since its trains run
  empty in the boarding account till then (`lineFiguresIn` in `game/demand.ts`, T-1279); and in a clicked bubble's panel in place of each line that waited ("Trip purposes", the far ends, "Inter-city trips") while they load.
  Nothing else animates. The figures behind it are the solve's estimates, or a cached result found
  before the game's network is in (T-1539).
- **Another game replaces this one by reloading the page into it** [T-1196]: Open, Open a file,
  Open as a copy and New game in the saves panel save the game in hand, let go of its slot, then
  reload with `?slot=<key>` or `?load=file|restore` (a file, a backup or a new game's spec waits in
  browser storage for the reloaded page's clock worker). New game reloads to the start menu
  (`?next=new`), which starts it as its own New game does, so a start that fails ends on the menu
  (T-3478); a new game whose spec is missing from storage still starts, never the game left. Leaving
  never waits on a download: a game still opening is let go as it is in storage, and an edit waiting
  for its ground is left out of the save. Demand's workers never give memory back
  while they live (about 2.5 GB each after all of China), so this is the one way the last game's
  memory goes. The switch leaves the address bar, so reloading from there opens the start menu.
- **A game too big for the device is refused** [T-1197; `game/memoryCheck.ts`]: before a saved game
  opens (the start menu, Open, Open a file, Open as a copy), its zones (the save's opened zones, else its
  slot's cities) give each solving worker's memory, 0.7 GB plus 1.2 GB per billion local trips a day
  of those zones, and the whole game's, 1.5 GB plus 15 MB a zone plus the workers'. A worker over
  3.5 GB (a browser worker cannot pass 4) or, on a device reporting under 8 GB, a game over three
  quarters of it is refused: the start menu shows the reason over the saved games, and in a game a
  notice says it ("Not enough memory. This save needs about 4.3 GB and your device has about 4 GB.
  This game works best on a computer.") [T-2545].
- **The bar is three stages, each with its share** [T-1195, notes/T-1195.md; `game/loadBar.ts`]: the
  game, the city data and the solve each get the share of the bar their expected times give them,
  and the stage in hand moves inside its share by what has landed (bytes downloaded, packs open in
  each worker, solve rounds and periods) or by its time against its expectation, slowing toward the
  end of its share when late and never passing it until the stage is done. Expected times are the
  slot's last load in this browser (each stage's time while it was the one in hand; by slot, the last
  12, a new game's under "new"), unless the network has since changed by over a fifth of its stops.
  With none remembered, until the network is in: 6% the game, 24% the data, 70% the solve, the game
  expected to take 4 s; then the game stage as it took, the data as modelled (each city pack 450 ms
  plus 220 ms per MB, market packs 20 ms per MB and 1.5 s for their composition, downloads at their
  rate; times how this browser's last load compared with the model), the solve 34 s plus 148 ms a
  zone it covers plus 7.88 ms times the square of its line stops, times how this browser's last game solve of 10 s or
  more compared with that [T-1479, T-2552]. A file or backup being opened has none of its own (never the last game's)
  and is remembered under the slot it goes into [T-2552]. The solve's first round and its later ones are timed apart: before the first lands the
  later ones are expected at this browser's last ratio of the two (0.5 with none), after it at that
  ratio times the first, from the second on at their own mean; a round running late stretches the
  rounds after it too. This model is a guess, so it fills half the bar at most (below).
- **Every bar passes half only on a time left it can bound** [T-4264, Anita 11-25; `barPlace` in
  `game/loadBarCore.ts`]: the card's, the status tag's and the zones' on the map. A guess (the model above, a
  remembered load, the last such wait) shows its fraction to the power 1.7 (the model ran ahead of the clock early:
  T-2638, notes/T-2638.md), its 90% at half the bar (her 11-22 call), and goes no further until the work is done or
  bounded. Bounded is a solve's share of its wait by its workers' time left (`editLeftOf`): every unit still to run
  timed at this solve's own pace (a unit of its stage has ended on that worker) and none in hand past its time at
  that pace, the card's other stages done; and counted downloads (terrain). A bounded share runs the bar from where
  it stands to the top in step with it; where the bound lapses (a unit over its time, a wait for the next unit
  longer than what the units have left, the stretch after the last unit until the figures show) the bar holds, and
  the time left adds a fifth of the overrun so far [T-4359]. The bar never goes back: when a bounded share falls (an
  estimate grows) it keeps its place and the rest is spread over the rest of it, moving again with the next work done
  [T-2552]; the status tag's wait ends at a final that covers every edit made, so a solve still running then (the same
  network solved again) starts its own bar from empty [T-4359]. It is never full before the card goes, except after a
  cached result [T-2574]: once the load's network is found in the cache (its zones, build, workers and flags match,
  its packs' digests still to come), the stages are the game, the packs' download and the result's showing (1.5 s
  once the network and packs are in), with nothing opened or solved; the bar keeps its place and runs from there, the
  time left drops to seconds, and when the result shows the card stays 0.4 s with the bar full and no time left
  before it goes. A read that then misses goes back to the full load's stages from where the bar stands. The pick and the game it
  opens are one wait. A game loaded in place while the last one's bar is still up (a check's direct
  load; the player's loads reload the page) starts the bar over from empty [T-840].
- **Under the bar, the share done and the time left** [T-1479, T-1654]: "42%" at its left, "About 2 min
  left" at its right, always a figure. The time left is each stage's expected time less its time so far;
  a stage past its expected time is guessed to go on a share of its time so far (a data stage 0.3 of it
  times the part not yet landed). The solve is the rounds still to come: the rounds so far give the
  rest (the remembered load's before the first lands); within a round, the periods land in waves, and
  the first one in gives the round's length (times the waves' ratio, which the round before measures; the
  first round's setup in the workers counted once, and a later round's first period in under half a wave's
  expected time, one that has converged, not counted) [T-2552];
  a round past that is guessed to go on as long per wave as it has so far. It shows once the game's
  network is in (from the start when the slot's last load is remembered). It counts down with the
  clock and drops to a lower estimate at once; it never climbs by degrees: an estimate over 1.4 times
  the count and 10 s more replaces it in one step, and under 5 s a higher estimate of 5 s or more does
  too. Under 5 s "A few seconds left", under 55 s "About 40 s left" in tens of seconds, under 90 s
  "About 1 min left", then whole minutes.
- A city opened later (a pan, construction) shows its estimate at once and the crowding rounds
  refine it with the clock running; the status says "Trips here being recomputed…" until the zone in view's cluster has its final [T-421].
- **After an edit** nothing changes until a final round: split, riders, bubbles and
  the boarding account keep the last finished forecast, the split faded and the status "Updating
  demand 1/2…" (Status, above), then each changes in one step [T-232]: the zones the local solve
  covers when it lands, their bubbles with them [T-2532], everything at the whole solve's final (docs/demand-runtime.md). Queries asked meanwhile (a station's riders, a bubble
  selection, a trip) wait for that round. A line new or changed by the edit has no figures to keep,
  so it takes the free-flow estimate's riders as soon as that lands, as do stations only such
  lines serve [T-324]; until a running line has any figures its trains are striped
  ([ui-map.md](ui-map.md)) and the train panel says "Calculating".
- **A city loading later** (a pan, construction reaching it) holds nothing and has no card: the
  status says "Loading data" with a thin bar after it, its hover naming the cities and step
  ("Loading Boston, 37%. Loading city data"); in a narrow bar the thin bar goes first, then the
  text ends in an ellipsis. It ends once the cities' packs are open, except a city with no figures on the map yet,
  which waits for the solve that gives it some; a city with figures (from a look at it: zones open bare first) is
  then in the calculation the status shows as calculating trips [T-4359]. A zone's own bar says "Loading city data"
  only until its packs are open, then "Calculating trips".
- **Terrain on its way** [T-2613, notes/T-2613.md]: after a city's "Loading data" and before every
  case below, the status says "Downloading terrain…" with the spinner once a terrain download has
  been on its way 1.5 s (the clock worker's terrain classes, water masks or world water, or the hill
  shading's tiles while Terrain is on), and "Terrain could not load" with no spinner while one of the
  clock worker's failed, hover "Check your internet connection. Click to try again, or wait up to 30 s."
  (it is: 2 s after the failure, doubling to 30 s, while the view or the blueprint still needs it). A click
  on the failed tag tries at once and starts the wait over at 2 s (T-3479). Both
  show also after demand stopped, and neither while the loading card is up. Construct waits for the
  ground under the blueprint and is refused while it could not load (track-and-service.md 6.4).
- **Bubbles on their way** [T-1493]: with Homes, Destinations or Total on, a zone in view that has its
  figures but whose own bubbles are not drawn yet (being asked for, waiting behind a solve, built)
  has the status "Loading data…", with no hover, or behind a running solve that solve's time left as the
  edit's hover words it ("About 4 min left") [T-4264]. An ask that ends with
  no bubbles is made again after 1 s, doubling to 8 s, until they come, so the bubbles of a zone in
  view always arrive. A zone come back to (the last 6 left) takes its cells and final legs from its last
  visit: at once for the same result, else shown until the result shown's come [T-2756].
- **Why a wait is slow** [T-2756]: the tag's hover adds "Your internet connection seems slow" while a
  city-data download has been open over 10 s, or terrain has been downloading 10 s past its tag; and
  "Recalculating after freeing up memory" while the workers restarted (or cut) to give back memory solve
  the game again (T-1205), which every question waits for.
- **A click that waits** [T-2523]: a bubble selection whose far end has not come, a trip the explorer
  is working out, or a city picked in the overview that is loading or has no trips to other places yet,
  puts the status (after the cases above, before "Calculation 1 day behind") on what it waits for:
  "Loading Boston…" while that zone's packs load, "Calculating trips in New York…" while a game load
  solves the zone in view on its own (the focus pool), "Calculating trips after your change…" while an
  edit's solve runs, "Calculating trips in 81 cities…" while a whole solve runs (the zones it still
  has to finish, as "Trips in 81 cities being recomputed…" counts them; the zone's name for one), and
  "Loading data…" with no solve running. Its hover has the time left as the loading card words it
  (the zone's load's, else the running whole solve's; none for the zone's own solve; after an edit, the
  edit's, as the capped tag's above), counted down each
  second, or "No progress for 1 min" after 90 s with nothing arriving, as a zone's bar has it
  ([ui-demand-views.md](ui-demand-views.md)). While the loading card is up it says the same in the
  middle of the map, and the status shows nothing.
- If demand cannot load, a screen says so and offers Try again (reload), Submit feedback (the feedback form,
  whose report carries the failure's text, T-3630) and Play without demand, and takes every key until one is
  pressed (the feedback form's typing excepted). If the game itself never started (the
  clock worker failed to start, or the page's setup threw before its end), it has Try again and Submit feedback
  alone [T-950]. A saved game that browser storage cannot read (four tries over about two seconds) is such a
  failure: the reason line says the save could not be read and nothing was written over it, and the game
  never opens as the Original game or a new one in its place [T-2703]. The box is centred, 400 px wide, the sentence balanced over its lines; when the address
  names a game (it has a query), a Back to start button sits beside Try again (it opens the address
  without the query, so the start menu). The small muted line under the sentence (`.load-why`) is the
  slot for the failure's reason text. The screen gives no reason, only "Ran out of memory. This game works best on a computer." when
  the failure was a failed allocation (`isMemoryFailure`, `game/loadText.ts`), else "Failed to load :( This game
  works best on a computer." [T-2545]; the cause goes to the console. The same two sentences come
  back for a game or a real network the clock worker could not load, and as the hover of the status
  tag's "Demand could not load". A page that finds the flag set in sessionStorage at its start
  (set when a game starts loading, cleared when the load ends, fails or the page is left) was
  reloaded by the browser mid-load, as iOS does on running out of memory (desktop Chrome's tab dies so too, and
  is reloaded by the player): if that load ran more than one solving worker at its most (after the memory
  estimate's planned cut) and its game was in a slot, the page opens that slot again, once, with half of them,
  at least 1 (the tab's pool cap, as `memoryShort`'s), and a note at the bottom of the map says "This game ran out of
  memory, so it loaded again with fewer cores." from the start of that load until 45 s after it [T-3933;
  `game/oomRetry.ts`, notes/T-3933.md]. A load whose last solving worker fails for memory in the page's sight is
  saved and reloaded into the same retry. Otherwise, or when the retry runs out too, it starts no game and shows the
  memory sentence with Try again (whose load then has the lower cap). A browser with no WebGL2 at all (no `WebGL2RenderingContext`;
  MapLibre cannot start; `webgl2State`, `map/webgl2.ts`, asked at the top of main.ts) gets its own screen
  before anything else starts: one sentence saying to turn on hardware acceleration or update the browser,
  and no button, since a retry or a way on cannot help [T-3307]. A browser that has WebGL2 but refuses the
  page a context gets the same screen with another sentence: the map's graphics stopped working, reloading
  won't fix it, close every browser window and open the browser again, and failing that check hardware
  acceleration (a browser can block a page's graphics after they were lost until it restarts). The browser's
  creation-error message goes to the console and the failure's reason. The same screen covers a game mid-play
  when a lost context has not come back after the graphics notice's 6 s and a fresh probe context is refused
  too; the game is paused and its autosave asked for, and the screen goes if the context comes back after
  all. A probe that works keeps the graphics notice, whose Reload will then work [T-3582]. A browser with no WASM SIMD (Safari before 16.4; `hasWasmSimd`,
  `map/wasmSimd.ts`, asked right after WebGL2) gets the same screen with its own sentence: no WebAssembly
  SIMD, worded "missing a feature the game needs", update the browser (Safari 16.4 or later), Edge, Chrome or
  Firefox may work better [T-3315].
- A card that sees nothing move (no demand worker message other than a query's answer, no pack
  bytes, no game or network from the clock worker) for 3 minutes, or 10 with a bar up, gives way
  to that screen ("No demand figures arrived in 3 minutes"); a hidden or frozen page counts no
  quiet [T-950].
- If demand stops later (a worker out of memory on a very big network, or an error) [T-757, T-2657],
  the clock is paused once (the player may run it again) and a notice with a red border sits at the
  top centre of the map: "Demand stopped, so the game is paused. Reload the page to start it again."
  ("Demand ran out of memory" for memory), with a Reload page button that writes the autosave and
  waits for it (at most 1.5 s) before reloading. An error's text is the small line under it (its
  hover the whole text). The status tag goes, the figures shown stay. `failDemand()` on `window.tw`
  (`?debug=1`) forces it.
- If the figures stop updating without a failure (a download the network dropped, a worker that never
  answers) [T-3314], the same kind of notice sits there once the game is up: "The game's figures stopped
  updating. Reload the page to start them again. Your last autosave from 24s ago is kept." (the age counts up
  while it shows; with nothing saved yet the last sentence is dropped) with a Reload button (it asks
  for the autosave, at most 1.5 s, then reloads). It shows when an edit or read the page sent to the clock
  worker has gone unanswered, or a network has waited for the demand pool with no download or worker step,
  for 3 minutes of visible time (`game/figuresStopped.ts`; time in a hidden tab does not count); it goes
  when the wait ends. Not shown while the demand-stopped or new-version notice is. `?figuresMs=N` sets the
  3 minutes for checks; `window.tw.figures` (`?debug=1`) has the longest quiets seen.
- A game whose view is over dead space and whose stations reach no zone loads nothing: no card, no
  status, the clock not held, the split "–" [T-318]; a change to its stations or lines is still
  solved, with no pack, so its lines get their figures (nobody aboard) [T-845].

- Triple click on a line name (Lines list) or a station name (Stations list) renames it (T-2786). Deleting a game of 100+ stations asks for its station count typed (T-2784).
