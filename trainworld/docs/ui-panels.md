[Back to the spec](../SPEC.md) · Interface: **panels** · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: look, dock, top bar and panels

## Look

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars); the network takes the line colours ([ui-map.md](ui-map.md)).

- **Nunito** (Zen Maru Gothic for kana and kanji; the font is a Settings choice, T-703) everywhere, station names, map DOM controls and form controls included (the
  map container inherits it rather than MapLibre's Helvetica) [T-101]. Few font sizes and weights:
  3 sizes and 2 weights at most (the T-009 mock's, `mock/`). The basemap's labels are Zen Maru
  Gothic whatever the font setting, Regular only, faint grey, from glyph tiles hosted beside the app (`app/public/fonts/`, `pipeline/glyphs.py`,
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
- **No motion but trains.** Instant state changes, no CSS transitions. Camera moves the game makes
  (search, Go to) jump with no flight [T-182]; the map's own gestures keep MapLibre's easing.
- **Panels show what can change** (Anita) [T-253]: labels, numbers and states. No explanatory
  notes or how-it-works sentences; a fact worth keeping goes in the hover of the label it explains.
  Icons with hovers over word labels where the icon is clear.
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
  controls) is turned into the tag first.
- **Every drawn quantity has a tooltip** (Anita) [T-194, notes/T-194.md]: bubbles,
  inter-city and city bubbles, line widths and track colours, trains, stations, crowds, demand
  circles, capacity tags, and in the panels the purpose split, pie and timing bars, the trip
  explorer's bars, the complex map's lines and transfer curves, the junction map's routes and the
  train's load bar.

## The dock

**Fixed layout, no floating windows** (the saved games panel and confirmations are the
exceptions). One dock on the left [T-252, `ui/Dock.tsx`]:

- A row of icon tabs, names in their hovers: **Build** (a wrench), **Lines** (a route with stops),
  **Stations** (a platform under its canopy), **Money** (a bill and a coin), **Cities** (a skyline) [T-749], sharing the row's width
  equally [T-532] (each at least 34 px); at the row's right end
  the saved games button (a floppy disk) [T-251] and the **Settings** gear (a toothed wheel outline with a hub circle) [T-523], which opens in the tab
  area like a tab. There is no City tab [T-220]; a browser that last had it open starts on Lines.
- Under the tab content, the **inspector** ([ui-inspectors.md](ui-inspectors.md)). The tab area has
  a fixed height and scrolls, so the inspector's top edge never moves when the tab changes. By
  default it is 36% of the dock but at least 440 px (the inspector keeps 160 px), which holds the
  Build tab's tools, elevation, track rows and blueprint bar whole; only a blueprint's cost
  breakdown, the lists and Money and Settings scroll [T-1128].
- The player can drag the dock wider or narrower and move the divider between tab area and
  inspector; both are remembered per browser. Each tab keeps its own scroll position for the
  session, opening at its top the first time [T-215].
- A refused edit's reason, what an edit did, and the stop editor's and trip tool's one-line hints
  show at the map's bottom left (`ui/ToolNotes.tsx`); the build tools have no hint there [T-1159].
- **Narrow screens** [T-951, `style.css`]: at most 600 px wide and at least 560 px tall (a phone held
  upright) the dock goes under the map, full width and 40% of the height, so the map and top bar keep
  the whole width and the map column's container queries see the real width. The tab area is 48% of the
  dock (at most 60%) until the divider is dragged; the width handle is gone and the controls panel
  covers the dock's width. Nothing changes above 600 px wide or on a short (landscape) screen.

## The top bar

`ui/TopBar.tsx`, left to right, 54 px high, **every part centred on one line** [T-349]: the name,
the split (the bar, then the train share), cash, the clock, speed buttons. **Three type
styles only**: normal (14 px, regular, text colour: name, cash), figure (18 px, bold: the
train share and the time) and small (12 px, regular, muted: status). **The cash, clock and
speed buttons have fixed widths** [T-341, T-1136] and sit together at the right with the usual gap
between each, so nothing moves them: the name and the split fill all the room left of them [T-1108]. The name takes
what it needs up to 104 px (88 px under 760 px of map column), on two lines at most with an ellipsis
(the whole name on hover, with the city's points and points a day, T-684, T-694), and the mode bar
(at least 80 px) widens into the rest, so a short name leaves no gap. The train share is in tabular
figures in a box of fixed width, and the split bar is centred in the bar with nothing under it: the
status is a tag over the map (Status, below). The time and date are one line ("06:50 Mon May 5")
[T-1136]: from 940 px of map column it stands beside the sky and band in a box 300 px wide, the sky
taking what the date leaves (so a longer date narrows the sky by a few pixels and moves nothing
else); narrower, it sits over the sky, 148 px wide [T-1109, T-861].

- **The goal number**: the focused zone's mode split as one bar (driving, train, walking, in the
  mode colours whatever the theme), the split of all its modelled local trip legs (commutes,
  everyday outings, school and university), so the shares are of rides per day, not people [T-342].
  The train share follows the bar ("18.5%"; no word "Train" [T-650] and no rides count [T-832]); the
  bar's hover gives all three shares with their counts and the total [T-1107]. The player's job is
  to make the train share grow. It shows "–" and an empty bar until the zone has figures of its own
  (a solve's, or the bare pass's for a zone no station reaches; not the no-rail split a pack gives
  before that), and over dead space [T-318]. The hover says "on a weekday", "on a Saturday", "on a Sunday" or "on average" for the
  day shown. The zone named is the focused zone ([ui-demand-views.md](ui-demand-views.md)).
  **Zoomed out below zoom 5** (a few countries on screen) the bar names "World" instead [T-1110]:
  the split and rides summed over every city with a built station and a demand view, as the Cities
  tab's World row sums them (`game/worldTotals.ts`; hover: the city and station counts), whatever
  lies under the view's centre.
- **Status** [T-349, `ui/StatusTag.tsx`]: a small tag (small type style, background, thin border)
  on the map's top row, to the right of the demand buttons and centred with them (`ui/DemandSelector.tsx`;
  the buttons stay at the top left). It appears and goes in one step and moves nothing; the whole
  text is its hover. On a map column under 520 px the tag takes a second row under the buttons,
  kept whether or not it shows, with the search glass, the demand stopped alert and the selection bar
  under it. While a solve runs after an edit that changes the zone in view's figures, "Updating
  demand 2/4…" and the split faded, the figures held [T-232]; a zone the edit does not touch, or
  whose cluster is already done, shows neither [T-392]; and the same while a reload shows the cached
  figures from the last time the game was open, with a hover saying so [T-328]; during startup's
  refinement "Estimate, refining 2/4…", which goes once the zone in view's own cluster has its final,
  though other clusters still solve, and the split's hover then stops saying "an early estimate"
  [T-421]; a city loading later "Loading data" with a thin bar after it
  (Loading, below).
- **Cash**, four significant figures (to the dollar under a million), moving as fares are paid and
  running costs drain [T-284, T-722]. A box of fixed width sized to "−$37.34B", the figure
  right-aligned in tabular figures with the $ right before it, so it always ends the usual gap before
  the clock and a changing figure never moves the rest of the bar [T-305, T-310, T-1136]; a longer
  figure ("−$999,949") reaches left into the gap before it. In a sandbox the box says
  "Sandbox" in the muted colour, with the money since day 1 (the cash less the $6B start) and
  today's fares and running costs in its hover [T-728].
- **The clock** [T-861], all of it the local time where the player is looking [T-309]: the time
  (figure style) and the date without the year (normal style) on one line 6 px apart, "13:24 Mon Feb
  24" (hover "Monday, February 24, day 29"), beside or over an 18 px scene and a 2 px demand band.
  The calendar is in [world.md](world.md) "Calendar". The scene
  (`ui/SkyBanner.tsx`, one canvas) shows 24 hours, 12 back and 12 ahead, now being its middle
  [T-896]. Its sky takes flat colours by the sun's height at the view's centre on that date
  (`game/sun.ts`): dark blue night with stars, a short pink and warm sunrise and sunset, light blue
  day [T-882]. Flat clouds sit at fixed game times, none to two a day as the day's number falls,
  white by day and dim blue at night [T-902]. Along the bottom is a track with sleepers, darker at
  night. At now sits the present-time marker: a red train (18 by 5 px, lit windows, nose
  to the right) on the track, with the sun, or a crescent moon at night, peeking in from the top
  edge above it. Everything but the train and the sun or moon is fixed to its time and slides left
  under them as game time passes. It is redrawn on animation frames from the clock's `now()`
  whenever it has moved a fifth of a device pixel, never through Preact. The band under it has each
  hour's demand level (red high, yellow medium, blue low: `--dl-*` in style.css, both themes), so
  the colour under the sun or moon is the level now; its hover says "Currently low demand until 06:00", the time the level next changes
  [T-902]. A public holiday in a country
  the player has built track or stations in (blueprint not counted) [T-862, T-1098] is a pink ribbon along the sky's top over its day (its
  country's local day), with a tick where it starts; the sky's hover lists those in view, and is the sky's only hover ("Tomorrow:
  Thanksgiving Day, United States", or "Tomorrow from 06:00: …" for a country on another local
  time), and on the day itself a pink dot sits before the date, whose hover names it. A holiday
  several of those countries keep on the same local date is one line [T-1098, `groupHolidays`]:
  the same name, or names `SAME` in `game/holidays.ts` counts as one (Labor Day, International
  Workers' Day, May Day and the like; Christmas; Boxing Day and Saint Stephen's Day; New Year's
  Day), shown by its one name or by `SAME`'s; the sky's hover names every country ("Today: Labor
  Day, Belgium, China, France and Germany"), the date's names two and counts the rest past three
  ("Labor Day, Belgium, China and 2 others"). In the dark theme the
  scene is mixed 15% towards the background. Pause and three
  speeds. **Space** pauses and resumes at the speed that was running, even with a button focused.
  While the tab is hidden the clock stops, unless "Run in background" is on (Settings), when it runs
  on at the chosen speed with fares, running costs and autosaves, and the map catches up when the
  tab is shown again [T-826].
- **Narrow maps**: secondary labels drop first; two rows below 700 px (name, cash, time and speed
  over the split), a different two below 520
  px (speed beside the split), three below 300 px, with cash, time, train share and all speed
  controls kept. Hovering the split or the time gives what
  was dropped [T-106].
- An **Autosave off** notice when this tab cannot autosave (saves, below); a click opens the saved
  games.

**Demand shown for** [T-260, T-266, T-464; `ui/DemandSelector.tsx`, display setting `dayShown`]: a
"Day" choice in Settings, right under the demand bubbles' Opacity [T-703]: Average (the default), Weekday, Sat or Sun [T-262]. The average is 5 weekdays, a Saturday and a Sunday over 7 ([demand-timing.md](demand-timing.md)
4.2). It sets the day every demand view shows, the split included, and does not follow the clock,
so nothing jumps when the clock enters a weekend. Hovers state the day beside every demand figure ("12,400 trip legs per day",
"per weekday", "per Saturday", "per Sunday": `shownDayPhrase` in `game/ui.ts`). A switch of day shows at
once. The trains, platforms, the
past 24 hours' riders and the panels' Today counts always follow the clock.

## Search

[T-148, T-177, T-183, T-211; `game/search.ts`, `ui/Search.tsx`]

- A square button with a magnifying glass at the map's upper right (under the demand selector
  when the map is narrower than 680 px). A click, `/`, Ctrl+K or Ctrl+F, from anywhere but a text
  field or the confirmation dialog, opens the input in its place, focused. Ctrl+F is kept from the
  browser only then. Up and down move through results, Enter or a click picks, Escape clears and
  closes it, as does leaving it empty; left with text, it stays open.
- It finds the save's stations, its lines (by name, or the badge symbol exactly), the registry's
  cities and zones, and the towns and neighbourhoods on each city's OSM place list (fetched the
  first time the box is focused, `game/placeList.ts`; a place's other OSM names count). Matching
  ignores case, accents and punctuation: exact names, then names starting with the text, a word
  starting with it, containing it, then clipped nicknames ("south philly": each word starts like a
  word of the name bar its last two letters, keeping at least four). Ties list cities, lines,
  stations (busiest first), then places (bigger kinds first, then population). No addresses (no
  geocoder).
- Up to eight rows: a line's badge and "Line", a city "City", a place a hollow ring and its kind,
  a station its line badges (one per badge look, four at most), blueprint stations muted. Results
  sharing a name get a muted second line telling them apart: the city; within one city the lines
  only this one serves (three at most, a shared first word once) or else its compass direction
  from the others. Stations of one complex sharing a name are one result, picking the platform
  with the most lines [T-182].
- A pick moves the camera at once: a station to zoom 14 or closer, a line to its stops' bounds, a
  city to its registry centre and zoom, a place by its kind (town 12.5, suburb 13, neighbourhood
  14.5). With no build tool on it selects the station or line (a city clears the selection, a
  place clears a station or bubble selection); with a tool on it only moves the camera.

## Keys and focus

[T-183, `ui/keys.ts`] One rule for the game's keys (the tool numbers, T, N, L, C, Esc, Space, Ctrl+Z/Y, the line tool's
Enter and Backspace) and search's `/`, Ctrl+K and Ctrl+F. A text field keeps every key (search's
own box gives Ctrl+K and Ctrl+F to selecting its text). Another form control keeps only its own
keys (arrows, Home, End, Page Up and Down, Enter, Space, and in a select a letter); the rest reach
the game. A select picked with the mouse gives up focus at once, so the next Space pauses; one
reached with Tab keeps it. Buttons show the browser's focus ring for keyboard focus only [T-169].

## Controls

[T-908, `ui/HelpPanel.tsx`] The "?" at the top of Settings [T-1099] (hover "Controls") opens a panel over the
map beside the dock, under the tab row, listing only the controls no button shows: the bubble
gestures (Shift+drag box, hold then drag, Shift+click, Shift+drag over far bubbles), building
modifiers (Shift+click, double click or Enter, Backspace, Caps Lock, right drag and D, E, Delete in
the blueprint tool), the keys (tool numbers, T, N, L, C, Space, Ctrl+Z/Y, `/` and Ctrl+K, Esc,
the letters read from `TOOL_KEYS`, `ACTION_KEYS` and `TRIP_KEY`) and panel gestures (double click
a fare node, a name, a list row; Shift or Ctrl on the train stepper). Esc (before the game's own
Esc, so the selection stays), a press outside or the button again closes it. A control added or
changed that a button does not show gets its row here.

**Confirmations** [T-113]: a centred "Are you sure?" dialog with operation-specific text, cancel
and action buttons. Cancel has initial focus, Tab stays inside, Escape cancels, focus returns when
dismissed, and the map and tools are blocked meanwhile.

## Settings

`ui/SettingsTab.tsx`, remembered per browser, in order: two icon buttons at the top, Controls (the "?",
opens the [Controls](#controls) panel) and, when accounts are on, Account or Sign in (the person icon, the account
panel, [Account and leaderboards](#account-and-leaderboards)) [T-1099]; Demand bubbles, three sliders: Size
(0.5-2x, [T-099]), Opacity (20-100% of the built alpha, default 90%, [T-470]; for the
commuter and longer-trip bubbles, not station discs; ignored while a bubble is selected, T-936) and Detail (4 steps from coarser to finer,
the third by default; its hover names the zoom the finest bubbles start at; [T-795]), the Colours
choice (Train share by default [T-833], which shows two colour pickers labelled 0% train and
100% train, or Mode mix; [T-804]; both in [ui-demand-views.md](ui-demand-views.md) "Local trip bubbles"), the Day choice (Average, Weekday, Sat, Sun;
"Demand shown for", below) [T-703], Show (trains, station names, capacity markers, full stretches [T-1152: the Traffic view's red edges, on by default, its hover saying so; [ui-map.md](ui-map.md) "Track display"], flat junctions and crossings, map labels), Riders on trains
and platforms ("How full each train is", "How full each platform is", both on by default)
[T-164], Stations up close (Faces and Sound, both on by default, and a Volume slider while Sound is on; [ui-map.md](ui-map.md) Life at stations) [T-686], Map (Flat or Globe, flat by default; hover "Zoomed in to a city, the globe becomes the flat map"; [ui-map.md](ui-map.md) "Globe") [T-033], Theme (Light or Dark, light by default, over Pink, Blue or Green) [T-039], Panel and text size, Font [T-698], Units and performance, Game (normal or sandbox, T-728), then,
when accounts are on, Account [T-1139]: who is signed in ("Signed in as alice", "Signed in with
Google" or "Not signed in") and a button, Account settings or Sign in, that opens the same account
panel as the icon at the top and the start menu's (sign in, recovery email, change password, sign
out, delete; [Account and leaderboards](#account-and-leaderboards)), and Explore a trip at the bottom [T-220]. Station demand
circles are at the top of the Stations tab instead [T-213].

**Panel and text size** [T-247]: 80% to 120% in 5% steps, 90% by default (T-703), no reset button. It
zooms the dock, top bar, map buttons, search, loading card, tool notes and dialogs, and sizes the
text of the map's DOM tags and the hover tag; the map itself is never scaled.

**Font** [T-698, T-703]: Nunito (default), Zen Maru Gothic, M PLUS Rounded 1c or Kosugi Maru (one
weight, so bold is synthesised), remembered per browser. Nunito is self-hosted (`public/fonts/`,
Latin and Latin extended subsets of the variable file, about 75 kB, SIL OFL) and has no Japanese, so
kana and kanji fall back to Zen Maru Gothic, which `index.html` loads from Google Fonts. The panels,
station names and map controls follow the `--font` variable (`game/fonts.ts`, `game/ui.ts`); the
other choices' stylesheets are fetched from Google Fonts (all SIL OFL) only when first chosen, and
station names and capacity tags are measured again once a font has loaded. The map's own labels are
glyph tiles and stay Zen Maru Gothic.

**Units and performance** [T-381]: "US units" (off by default) shows every distance, length and
speed in miles, mph and feet instead of kilometres, km/h and metres: panels, inspectors, hovers, the
track, station and line tools' readouts, the placing rings, the scale bar of a station complex, and
prices per mile. All go through the distance functions in `game/format.ts`; the game itself stays
metric. "Use more cores" (off by default) allows up to 5 demand workers instead of 3 at most; its
hover says it counts from the next load ([architecture.md](architecture.md) 7). "Run in background"
(off by default) keeps the clock running while the tab is hidden (the top bar's clock, above;
[architecture.md](architecture.md) "A hidden page") [T-826].

**Game** [T-728], last before Explore a trip: the game's mode, "Normal" or "Sandbox" (hover: what
it means). A normal game has "Switch to sandbox mode", which asks first ("Are you sure?", then
"Entering sandbox mode will disable money permanently and irreversibly.", with Stay in normal mode
and Switch to sandbox mode) [T-822]; a sandbox has no way back, so a later campaign's
or a hard start's achievements cannot be had for free. The mode is the game's, kept in its save,
not the browser's.

## Money tab

Cash (always the whole network's; in a sandbox "Since day 1" instead, the cash less the $6B start,
T-728); yesterday and today (fares, running trains, upkeep, new trains,
construction, net; Upkeep's hover gives the rule and the network's track, stations and upkeep a day, T-867), headed by the Lines and Stations tabs' place-or-world toggle (T-332, one setting
for all three): with the place on, these figures and History are that region's (T-531; the
place's hover adds how a region's money is counted), with the globe the network's. A region with
nothing in it shows $0. Money figures
in panels have three significant figures from a million up ("$22.4M", "$1.15B", "$233M"; thousands
as "$450k"; under $500 "$0") and a negative one is in the alert red (`ui/MoneyText.tsx`, `money` in
`game/format.ts`) [T-722]; the cash (top bar and this tab) has four ("$11.36B", "$893.6M"; to the dollar under a million) and is red
with its minus when negative; hovers may give more. History
[T-447, `ui/MoneyHistory.tsx`]: a Days / Hours switch (Days by default, kept while the game is
open) over three charts. The first, "Revenue and expenses" (revenue blue and expenses red in every
theme, the tokens `--revenue` and `--expense`, T-1111), has fares and running costs with upkeep
(T-867) as lines, scaled to the larger of the two, and one-off costs (new trains and construction)
as pale red bars on the same scale, cut at the top with a red cap when larger, so a build day does
not flatten the lines [T-452]; the top's figure at the right. Then net as bars up or down from zero
(gains blue, losses red). Then
train rides [T-565] as bars up from zero, the top's figure at the right (hover: it exactly), the
region's with the place on and the network's with the globe, from the same ledger entries (rides
are counted as in track-and-service.md 6.4). Days are the ledger's, one each in order (T-488), up to the past 30;
hours up to the past 48; a young game's few sit at the right of at least 7 days or 12 hours. The
day or hour now is drawn as it fills (a dashed last segment, a paler bar) and moves as fares land.
Under the charts the first and last dates ("Apr 14", "Apr 14, 04:00"). Hovering a column gives its
day or hour ("Mon Apr 14", "Mon Apr 14, 04:00", "so far" for now) with fares,
costs and their parts, or net, or its train rides. Days and hours are the game clock's (T-321). The
charts are saved with the game and are there after a reload [T-567]. Then, for the whole
network whatever the toggle: the fare curve (rule: [track-and-service.md](track-and-service.md),
fares), drawn as a graph of the fare against trip length, both axes linear
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
the last labelled with the unit. The curve has exactly three nodes. The first, the base fare, stays
at 0 km and drags up and down only; the other two drag either sideways (distance) or up and down
(fare), whichever way the pointer first moves 3 graph units, and only that way for the rest of the
drag. Hovering a node draws small grey arrows the ways it drags (two for the base fare, four for the
others); during a drag only the chosen way's two show, and the cursor shows it too. A drag snaps
distance to the nice step of at least 1/200 of the distance shown and fare to that of 1/250 of the
fare shown (at least $0.01), within the plot; each node stays at least 2% of the distance shown (and
0.1 km) beyond the one before and short of the one after, and its fare between theirs, so fares never
fall with distance. The scale holds still during a drag. A drag changes the graph and the figures
below it as the pointer moves, and sends the fare to the game once on release (each change restarts
demand). While fare changes are the last edits sent, Ctrl+Z takes them back one
at a time, and Ctrl+Y or Ctrl+Shift+Z puts them back, in order with the track's own undo and redo; the
undo button left of the zoom buttons does the same, enabled only then [T-1130, `game/fareUndo.ts`:
the sim's edit history holds track, lines and schedules, so fare steps are kept on the main thread
and any other edit drops them]. A node past the plot is an arrow
on the plot's edge pointing toward it, with its hover; a click on the arrow zooms out to show it. A
hover anywhere on the graph names the fare at that distance, a node's hover its distance and fare
and the ways it drags. Below the graph one row per node: "Base fare" with its fare, then the other
two as "20 km costs $3.50", distance and fare, each stepping with the buttons (distance by 0.5 under
10, 5 under 100, then 50; fare by $0.25), or a click on it opens a box to type it, Enter or leaving the
box applying and Escape cancelling [T-626, `TypedValue` in `ui/widgets.tsx`], kept in order as a drag
is; then what a 10 km or mile ride costs; cars
owned (prices in its hover), cars the lines need and spare cars [T-631]; the value of the track and stations built. Rules: [track-and-service.md](track-and-service.md)
6.4. The economy's 100x scaling is in the hover of "A day at this schedule".

## Cities tab

[T-550, T-749, `ui/Cities.tsx`] Whatever the Lines toggle says: a list of every region (the toggle's regions) with a built station, one row each, under a first
row "World" that sums them [T-857, T-1110] (a trophy after its name opens the leaderboards when accounts
are on, [Account and leaderboards](#account-and-leaderboards); its split bar weighs every region's trips, its rides, profit and points are the sums; hovers as a
row's, in the world; the top bar names the same sum when the map is zoomed far out, `game/worldTotals.ts`; it is not clickable or sorted): its name
(hover: its station count), its mode split bar as the top bar draws it (the same colours and the
same hover), its train rides per day and its profit a day (the column is headed Profit), fares less running trains and upkeep (yesterday's, today so far
until the game has a first full day; hover: fares, running trains and upkeep, then new trains and
construction, left out of the figure, when there were any; losses red), and its points, those of the
past 24 game hours, changing once a game hour, to three significant digits ("512k", "4.56M", "103M"; `pointsShort`) [T-831] (hover: the exact figure, points a day with the
riders now on the day shown, and the rule) [T-684, T-694]. A star after the name marks a region
whose train share has reached 10% (hover: when it was earned, "Monday, day 71, at 14:08" in the
region's local time) [T-685]. The top bar's city name has the same points in its hover. A narrow
list keeps about six characters of each name (60 px): below that the star goes first, then the
profit column (container queries on `.cities-box`) [T-693]. Sorted by rides, largest first; a click on a header sorts by that
column (name, train share, rides, profit, points) as in the Lines tab; a click on a row moves the camera to
the region as search does, and the viewed region's row is tinted. With no built station the tab
reads "No stations yet."

## Saved games

[T-068, T-241, T-251, T-320; `ui/SavesPanel.tsx`; save format and slot rules in
[architecture.md](architecture.md) "Saves"]

- The floppy disk button left of the gear, or the top bar's Autosave off notice, opens a centred
  panel over a dimmed page, closed by Esc, its close button or a click outside.
- One card per browser slot: a 240 x 160 picture of its network, drawn by the clock worker when the
  slot is saved, at twice that resolution, on the land colour of the theme it was saved in [T-543].
  The picture is one panel per city with stations, the most stations first, at most four, leaving
  out a city with under 8% of the biggest one's stations (a fringe of its network) [T-894]:
  one fills it; two sit side by side; three are a large panel on the left (60%) and two stacked on
  the right; four are a 2 x 2 grid. Panels are split by a thin rule in the casing colour. Each panel
  frames its city's stations (at least 3 km across, room left at the top for the name) and draws
  all track in grey and each line in its colour that falls in it, so a line to another city runs
  off the edge; stations are not drawn [T-1100]. The city's name is text over the panel's top
  left in that theme's ink with a land-coloured halo, and with two or more panels its station
  count under it; the hover gives both. A network with no station in a city is one unnamed panel
  over all of it. Then its name, "Autosaved 2 min ago"
  for the game being played or "Saved ..." for the others (exact time on hover), the game day, the
  cities its stations are in, lines, stations and cash. The game being played is marked Playing;
  every other card has Open (closes the panel and loads it) and Delete (asked first); every card
  has Rename. A slot with no stored picture shows "No picture yet" until it is next saved.
- Under the Playing card, its backups [T-744], newest first: "Backup 4 min ago · Day 26" (exact
  time on hover) and Restore as a copy, which closes the panel and opens the backup in a new slot.
- Above the cards [T-1131]: Save to a file
  (`anitabuilder-nyc-day-12.save`), Load a file, New game, then Save as a copy (with a name). Load a file asks nothing: it closes the
  panel and opens its game in a new slot, the game being played saved first [T-743; the slot rules
  in architecture.md]. New game shows the new game sheet (below) in the panel, headed "New game";
  its Start closes the panel and opens the new game the same way, its Back returns to the cards.
- A page opened with `?new=1`, `?synth=1` or `?demandTest=1` does not autosave: the top bar says
  "Autosave is off in this test page. Save a copy to keep the game." (the only place it is said,
  [T-971]), a `?new=1` page marks no
  card Playing, and Save as a copy gives the game a slot of its own, which autosaves from then on.

**A new game** starts on day 1 at 07:00 with $6B, empty or (a sandbox) from a real network, on its
start city's registry centre and zoom, or on the world map (centre 20° E 30° N, zoom 1.8: the
Americas, Europe and East Asia) with no start city; a start from the player's location is centred on
it (below). Opening the page shows the start menu (below).
For a loaded game the URL's view says only which city or zone; the map starts at that city's
registry centre and zoom, or over New York with none [T-310].

**The new game sheet** [T-728, T-731; `ui/NewGame.tsx`], the start menu's and the saves panel's:
- **Mode**: Normal (lit to begin with) or Sandbox, two buttons, each with a hover saying what it
  means. A sandbox is the normal game with nothing refused for money and the cash hidden; demand,
  fares, running costs, line money, the Money tab, points and stars are worked out as in normal.
- **Start in**, three ways separated by a small muted "or" [T-1132]. First New York, London, Paris,
  Tokyo and Toronto (`START_CITIES`, a fixed list) and Current location, one lit at most, a second
  click unlighting it. Current location shows only where the browser can give a location (a secure
  page with geolocation); the browser asks the player itself. While it waits the button says
  Locating; a refusal, an error or no answer in 15 s leaves the sheet as it was with the button
  saying "Location unavailable" (a click asks again; an answer after the 15 s still counts). A
  location starts in the zone whose coarse outline holds it, else the zone with the nearest centre
  among those whose box comes within 40 km (`ZONE_REACH_M`), the camera on the location at the zone's
  registry zoom; farther from every zone it is a start on the map centred on the location at zoom 4
  (`zoneForLocation`). Then "Find a city", which lists up to eight of
  the registry's cities and zones whose name has a word starting with the text (case and accents
  ignored, so "hague" finds Rotterdam [The Hague]), biggest by local trips first, each with its
  country (the registry's code, named by the browser). Up and down move, Enter or a click picks;
  a pick of one of the five lights its button. Then Find a city on the map, lit while nothing else
  is picked: the game starts on the world map with Homes on and every realm shown (zones.md
  "Realms"), so the zones' zoomed-out bubbles show and take a click as anywhere (ui-demand-views.md
  "Zoomed far out").
- **Real network** (ticked to begin with): only for a sandbox in New York, London, Paris, Tokyo,
  Seoul, Toronto or Sendai, the cities with a real network (`REAL_NETWORKS`; architecture.md 11). The
  game starts from that network, built and running.
- **Start in Tokyo** (the zone picked or located) or **Start on the map**, and Back. If a real network cannot be downloaded, a
  line says so and the sheet stays.
- The slot is named after the start ("Tokyo", "Tokyo sandbox", "World", then " 2" and on).

## Account and leaderboards

[T-972, T-973; `game/account.ts`, `ui/AccountPanel.tsx`, `ui/Leaderboard.tsx`; server routes in
shonei-server's `trainworld.go` and `docs/trainworld-api.md`; notes/T-972.md]

- **Optional.** Signing in is never needed to play. The account is noritetsu's (Google, or a
  username and password with an optional recovery email), kept under noritetsu's storage key, so on
  anita.garden one sign-in serves both pages. The server is market.anita.garden on anita.garden; on
  localhost none unless `?acct=local` (a shonei-server on 127.0.0.1:8083) or `?acct=live` was given
  once (remembered per browser, `?acct=off` forgets it). The icons show only once
  `/noritetsu/config` answers with `trainworld`; with no server, or an older one, there is no icon
  and no other request. Google's script loads when the account panel first opens.
- **Account panel**: the account button (top of Settings, start menu; not in the tab bar, T-1106), or Settings' Account row (T-1139), opens a centred modal like the
  saves panel. Signed out: Google's button, then username and password with Log in, Create an
  account and Forgot password; Privacy and Terms (`public/privacy.html`, `terms.html`). Signed in:
  "Signed in as alice" (or with Google), for a password account the recovery email (a red line when
  there is none) with Save email, Remove and Change password, then Sign out, Sign out everywhere
  else and Delete account (asked first; the account's leaderboard entries and noritetsu rides go
  too, games in this browser stay).
- **Scores**: while signed in, each normal game in a browser slot sends its points (the past 24
  game hours, the World row's figure) and its best 64 cities' points, when they change and at
  most every 2 wall minutes (at once on the first send and when the page is hidden), joined or not.
  A `?new=1` page or one whose slot is open elsewhere sends nothing. Each slot has a random game id
  in localStorage (`trainworld.score-games.v1`). A game that turns sandbox sends that once and is
  never listed again; one that was always a sandbox sends nothing. A slot deleted in this browser
  takes its figures off (on the next signed-in load if signed out then).
- **Leaderboards**: the trophy opens a centred modal: a picker (World, then the cities in this
  game by points, then other cities with rows, by name), the top 50 (rank, name, points short, the
  exact figure in the hover), the player's own row under them when lower, "Nobody here yet." on an
  empty board. At the bottom: Sign in to be listed (opens the account panel), a name (at most 24
  characters, the map games' name filter on the server) with Join, or Leave the leaderboards (the
  name is kept); in a sandbox game also "Sandbox games are not listed". Each person shows once per
  board, with their best game.

## Start menu

[T-677; `ui/StartMenu.tsx`; the startup order in [architecture.md](architecture.md) "Loading and readiness"]

- Every page load opens on a card over the bare map, in the loading card's look: "Welcome to
  anitabuilder!" (the account button at its top right when accounts are on [T-972]), New game, then "Saved games", one button per browser slot, the most recent first, each with the
  saves panel's picture (the same 240 x 160 with its names) and facts (name, when saved, day, cities, lines, stations, cash).
  With no saved game it is New game alone. The dock and top bar are taken out of the page and the
  map controls hidden, so the map fills the page; it shows the world from 60° S to 75° N covering
  the page (all of it on a wide screen, its centre around 10° E, 20° N on a narrow one) [T-927;
  `worldCover`], with the zone outlines, outside shade and the basemap's city, town and village names hidden until the pick [T-974]. The menu takes every key outside itself, and nothing loads behind it but the map.
- A click on a save opens that slot (a sandbox's name has the "Sandbox" mark). New game shows the
  new game sheet (above) in the card, headed "New game", with Back to the menu; its Start starts the
  game in a new slot, so no saved game is replaced. The slot is first written once the network has
  something in it (a real network at once): a new game left empty leaves no card.
- A saved game opens with its built stations all in view (every station if none is built), as soon
  as the game's first state arrives, before its demand: padding 70 px, no closer than zoom 12 so one
  station is not shown at street level. A save with no stations opens on the zone it started in, else
  the world map. The view is not saved. A pick from this menu, a slot opened in the saves panel and
  an imported file fit; `?continue=1` and a raw `import` message (checks) keep the camera
  [T-895; `startOpenFit` in `ui/locate.ts`, `fitNextGame` in `workers/clockClient.ts`].
- URL switches that already say which game skip it: `?new=1` (a new game owning no slot; normal
  in New York, or `&mode=sandbox` and `&city=<zone id>`),
  `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` (the game opened last).

## Loading

[T-149, T-172, T-180, T-212, T-232, T-243, T-318, T-793, T-884; `game/loading.ts`, `ui/LoadingScreen.tsx`]

- Nothing is blocked: the map, inspectors and building work as soon as the network is drawn.
- **A loaded game** (the page opening, a file, a slot, a new game) holds only the clock until the
  zone in view has its final figures [T-347, T-421]: its fresh solve's last round, or that of the
  zone's own cluster, solved ahead of the rest (T-402: a Toronto view of a game with New York's
  network, about 6 s instead of 31 s); a new game until its viewed zone's figures, opened bare,
  arrive (T-351). Zones still solving then show their own loading bar and grey ("A zone still loading
  is grey" in [ui-demand-views.md](ui-demand-views.md)) until their final lands. The time is
  greyed and its hover says "The clock starts when demand has loaded". Meanwhile a card in the
  middle of the map says "Loading data" over a filling bar and the current step (Downloading city
  data, Opening city data, Calculating demand), naming no cities; while it loads more than one city,
  the step counts the cities open of those it is loading ("Opening city data (15/120)", T-1161,
  `LoadProgress.done`); the percentage is only the bar's
  hover ("42% loaded"). The bar counts to that final, so on the real New York save it fills
  for 20 to 60 s, by the machine. A slot whose cities are all open shows the bar for its solve alone. The card
  takes no input but the bar's hover (a press pans the map beneath) and goes when the clock starts.
  Loading is the one place with motion [T-647, T-677, Anita; `ui/Spinner.tsx`]: on top of the card a
  drawn bunny rides a toy train in the theme's accent (104 px wide; wheels turn, the track's ties run
  by, the train chugs, the ears bob, steam puffs), and a 26 px copy of
  it before the status tag's text; a small ring turning (`.spin`, 1.2 s a turn, 8 px, grey) in place
  of each line's crowding dot per demand level in the line panel until the line has figures
  (while no final result is shown or a solve is updating it; not after a failure or over dead
  space), and in a clicked bubble's panel in place of each line that waited ("Trip purposes", the far ends, "Inter-city trips") while they load.
  Nothing else animates. The figures behind it are the solve's estimates, or the last
  final result kept beside the save when there is one (T-328; the split's hover ends "(from the
  last time this game was open)").
- **The bar is the share of the wait gone** [T-243, T-793, T-884, notes/T-884.md]: the time since the
  pick (or the load) against that time plus the time still expected, so it keeps an even pace while
  the expectation holds and slows, never stops, when a step runs late. A game's wait is first expected
  to last as long as its last one in this browser (by slot, the last 12; a new game's under "new"),
  unless its network has since changed by over a fifth of its stops; once the solve has shown its
  pace, that memory may move the step model's estimate only between 0.7 and 1.5 times, and past its
  length a quarter of the time so far is still expected. The step model, used alone with nothing
  remembered: downloads at their rate; opening at its rate once a quarter of it is in (each pack in
  every worker), before that 220 ms per MB of city pack, 20 ms per MB of market pack and 1.5 s for
  composing the market once for the load's market packs (T-889); the solve as the first estimate about 0.6 s plus 2.2 ms per line stop, scaled by this
  browser's last solve of the zone in view (a game of 1,000 stops is assumed with none), each
  crowding round about as long again (weights 1, 1.1, 1.05, 0.55), the rounds still to come at the
  pace measured so far, and a round running late expected to go on a further 15% of its time so far.
  The bar never goes back and is never full before the card goes. The pick and the game it opens are
  one wait: what loads before the game's network is known (the view's zone) stays in the same bar. A
  game loaded while the last one's bar is still up starts the bar over from empty [T-840].
- A city opened later (a pan, construction) shows its estimate at once and the crowding rounds
  refine it with the clock running; the status says "Estimate, refining 2/4…" (hover: the figures
  are an early estimate) until the zone in view's cluster has its final [T-421].
- **After an edit** nothing changes until the new solve's final round: split, riders, bubbles and
  the boarding account keep the last finished forecast, the split faded and the status "Updating
  demand 2/4…", then all change at once [T-232]. Queries asked meanwhile (a station's riders, a bubble
  selection, a trip) wait for that round. A line new or changed by the edit has no figures to keep,
  so it takes the free-flow estimate's riders as soon as that lands, as do stations only such
  lines serve [T-324]; until a running line has any figures its trains are striped
  ([ui-map.md](ui-map.md)) and the train panel says "Calculating".
- **A city loading later** (a pan, construction reaching it) holds nothing and has no card: the
  status says "Loading data" with a thin bar after it, its hover naming the cities and step
  ("Loading Boston, 37%. Opening city data"); in a narrow bar the thin bar goes first, then the
  text ends in an ellipsis.
- If demand cannot load, a screen says "Demand could not load" with the reason and two buttons, Try
  again (reload) and Play without demand, and takes every key until one is pressed. If the game
  itself never started (the clock worker failed to start, or the page's setup threw before its end),
  it says "The game could not start" with the reason and Try again alone [T-950].
- A card that sees nothing move (no demand worker message other than a query's answer, no pack
  bytes, no game or network from the clock worker) for 3 minutes, or 10 with a bar up, gives way
  to that screen ("No demand figures arrived in 3 minutes"); a hidden or frozen page counts no
  quiet [T-950].
- If demand stops later (a worker out of memory on a very big network, or an error) [T-757], a red
  alert under the demand buttons says "Demand ran out of memory" or "Demand stopped", its hover
  the cause and that a reload starts it again; the status tag goes, the figures shown stay.
- A game whose view is over dead space and whose stations reach no zone loads nothing: no card, no
  status, the clock not held, the split "–" [T-318]; a change to its stations or lines is still
  solved, with no pack, so its lines get their figures (nobody aboard) [T-845].
