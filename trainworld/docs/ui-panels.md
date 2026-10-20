[Back to the spec](../SPEC.md) · Interface: **panels** · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: look, dock, top bar and panels

## Look

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars); the network takes the line colours ([ui-map.md](ui-map.md)).

- **Nunito** (Zen Maru Gothic for kana and kanji; the font is a Settings choice, T-703) everywhere, station names, map DOM controls and form controls included (the
  map container inherits it rather than MapLibre's Helvetica) [T-101]. Few font sizes and weights:
  the mock's 3 sizes and 2 weights are the ceiling. The basemap's labels use it too, Regular only,
  faint grey, from glyph tiles hosted beside the app (`app/public/fonts/`, `pipeline/glyphs.py`,
  Noto Sans merged in for other scripts); kana and kanji are drawn in the system font for now
  [T-057, notes/T-056.md].
- **Few colour tokens**: background, text, one accent, plus `--bad` and `--warn` for warnings, as
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
- The T-009 mock (`mock/`) was the approved starting point for the look.
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
  inter-region and city bubbles, line widths and track colours, trains, stations, crowds, demand
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
  a fixed height and scrolls, so the inspector's top edge never moves when the tab changes.
- The player can drag the dock wider or narrower and move the divider between tab area and
  inspector; both are remembered per browser. Each tab keeps its own scroll position for the
  session, opening at its top the first time [T-215].
- Tool hints and a refused edit's reason show at the map's bottom left (`ui/Toolbar.tsx`).

## The top bar

`ui/TopBar.tsx`, left to right, 46 px high, **every part centred on one line** [T-349]: city name,
the split (bar, then its labels), cash, day and time, speed buttons. **Three type
styles only**: normal (14 px, regular, text colour: name, "Train", cash), figure (18 px, bold: the
train share and the time) and small (12 px, regular, muted: rides, walking and driving shares,
day, status). **Every part has a fixed width** [T-341], so no city name or status moves the split
bar: the name box is cut with an ellipsis (the whole name on hover, with the city's points and
points a day, T-684, T-694), the percentages and counts
are in tabular figures in boxes of fixed width, and the split bar is centred in the bar with
nothing under it: the status is a tag over the map (Status, below). The mode bar is the one
flexible part [T-406]: it takes the width the fixed parts leave, so its length follows the window
(at least 80 px) and never the numbers beside it, and grows on wide windows. Walking and driving
are the walk and car icons (the walk and drive mode colours) with the share beside them, each
with its own hover. They drop below 860 px of map column, the day below 1000, rides below 760;
below 700 px the bar takes two rows (name, cash, time and speed over the split).

- **The goal number**: the focused zone's mode split as one bar (driving, train, walking, in the
  mode colours whatever the theme), the split of all its modelled local trip legs (commutes,
  everyday outings, school and university), so the shares are of rides per day, not people [T-342].
  The labels after it give the train share, after a blue mark, with its rides per day ("18.5% 8.81M rides"; no word "Train" [T-650]) and the
  walking and driving shares as icons; the hover gives all three counts and the total, and an
  icon's own hover its share and rides. The player's job is
  to make the train share grow. It shows "–" and an empty bar until the zone has figures of its own
  (a solve's, or the bare pass's for a zone no station reaches; not the no-rail split a pack gives
  before that), and over dead space [T-318]. The hover says "on a weekday", "on a weekend day" or "on average" for the
  day shown. The zone named is the focused zone ([ui-demand-views.md](ui-demand-views.md)).
- **Status** [T-349, `ui/StatusTag.tsx`]: a small tag (small type style, background, thin border)
  on the map's top row, to the right of the demand buttons and centred with them (`ui/DemandSelector.tsx`;
  the buttons stay at the top left). It appears and goes in one step and moves nothing; the whole
  text is its hover. On a map column under 520 px the tag takes a second row under the buttons,
  kept whether or not it shows, with the search glass, the stations full alert and the selection bar
  under it. While a solve runs after an edit that changes the zone in view's figures, "Updating
  demand 2/4…" and the split faded, the figures held [T-232]; a zone the edit does not touch, or
  whose cluster is already done, shows neither [T-392]; and the same while a reload shows the cached
  figures from the last time the game was open, with a hover saying so [T-328]; during startup's
  refinement "Estimate, refining 2/4…", which goes once the zone in view's own cluster has its final,
  though other clusters still solve, and the split's hover then stops saying "an early estimate"
  [T-421]; a city loading later "Loading data" with a thin bar after it
  (Loading, below).
- **Cash**, to the dollar, moving as fares are paid and running costs drain [T-284]. Fixed width
  (room for "-$99,999,999,999") with the digits left-aligned after the $ in tabular figures, so a
  changing figure never moves the rest of the bar [T-305, T-310]. In a sandbox the box says
  "Sandbox" in the muted colour, with the money since day 1 (the cash less the $3B start) and
  today's fares and running costs in its hover [T-728].
- **The clock**: day of the week, day and time ("Sat 13", hover "Saturday, day 13 · High demand":
  the demand level is only in the hover [T-406]), the local
  time where the player is looking [T-309]; pause and three
  speeds. **Space** pauses and resumes at the speed that was running, even with a button focused.
- **Narrow maps**: secondary labels drop first; two rows below 700 px, a different two below 520
  px (speed beside the split), three below 300 px, with cash, time, train share and all speed
  controls kept. Hovering the split or the time gives what
  was dropped [T-106].
- An **Autosave off** notice when this tab cannot autosave (saves, below); a click opens the saved
  games.

**Demand shown for** [T-260, T-266, T-464; `ui/DemandSelector.tsx`, display setting `dayShown`]: a
"Day" choice in Settings, right under the demand bubbles' Opacity (it left the map's demand bar,
T-703): Average (the default), Weekday or Weekend. The average is 5 weekdays and 2 weekend days over 7 ([demand-timing.md](demand-timing.md)
4.2). It sets the day every demand view shows, the split included, and does not follow the clock,
so nothing jumps when the clock enters a weekend. Hovers state the day beside every demand figure ("12,400 trip legs per day",
"per weekday", "per weekend day": `shownDayPhrase` in `game/ui.ts`). A switch of day shows at
once (the bubbles' catch-up is for a new forecast, not another day). The trains, platforms, the
past 24 hours' riders and the panels' Today counts always follow the clock.

## Stations full

`ui/StationsFull.tsx` [T-231, T-284]: under the demand selector, a red-outlined button "5 stations
full" counts the station complexes where, now, a platform side holds as many riders as fit on it
(`platformRoom`, [track-and-service.md](track-and-service.md) 6.6) or the last train to leave a
platform was full and left riders behind (until the next train). Its hover lists them, most left
behind first, with riders waiting and left behind (twelve at most, then "and 3 more"). A click
selects the next one in that order and moves the map there at zoom 14 or closer, where crowds are
drawn. Hidden when there are none; updated each game minute. On a narrow map it sits beside the
search glass.

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

**Confirmations** [T-113]: a centred "Are you sure?" dialog with operation-specific text, cancel
and action buttons. Cancel has initial focus, Tab stays inside, Escape cancels, focus returns when
dismissed, and the map and tools are blocked meanwhile.

## Settings

`ui/SettingsTab.tsx`, remembered per browser, in order: Demand bubbles, two sliders: Size
(0.5-2x, [T-099]) and Opacity (20-100% of the built alpha, default 90%, [T-470]; for the
commuter and longer-trip bubbles, not station discs) and the Day choice (Average, Weekday, Weekend;
"Demand shown for", below) [T-703], Show (trains, station names, capacity markers, flat junctions and crossings, map labels), Riders on trains
and platforms ("How full each train is", "How full each platform is", both on by default)
[T-164], Stations up close (Faces and Sound, both on by default, and a Volume slider while Sound is on; [ui-map.md](ui-map.md) Life at stations) [T-686], Theme (Light or Dark, light by default, over Pink, Blue or Green) [T-039], Panel and text size, Font [T-698], Units and performance, and Explore a trip at the bottom [T-220]. Station demand
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
glyph tiles and stay Zen Maru Gothic. Stored settings from before T-703 (stored version below 3) lose
their saved font and size once, so they get Nunito and 90%.

**Units and performance** [T-381]: "US units" (off by default) shows every distance, length and
speed in miles, mph and feet instead of kilometres, km/h and metres: panels, inspectors, hovers, the
track, station and line tools' readouts, the placing rings, the scale bar of a station complex, and
prices per mile. All go through the distance functions in `game/format.ts`; the game itself stays
metric. "Use more cores" (off by default) allows up to 5 demand workers instead of 3 at most; its
hover says it counts from the next load ([architecture.md](architecture.md) 7).

**Game** [T-728], last before Explore a trip: the game's mode, "Normal" or "Sandbox" (hover: what
it means). A normal game has "Make it a sandbox", which asks first ("This game becomes a sandbox
for good...", Make it a sandbox or Keep it normal); a sandbox has no way back, so a later campaign's
or a hard start's achievements cannot be had for free. The mode is the game's, kept in its save,
not the browser's.

## Money tab

Cash (always the whole network's; in a sandbox "Since day 1" instead, the cash less the $3B start,
T-728); yesterday and today (fares, running trains, new trains,
construction, net), headed by the Lines and Stations tabs' place-or-world toggle (T-332, one setting
for all three): with the place on, these figures and History are that region's (T-531; the
place's hover adds how a region's money is counted), with the globe the network's. A region with
nothing in it shows $0. Money figures
in panels have three significant figures from a million up ("$22.4M", "$1.15B", "$233M"; thousands
as "$450k"; under $500 "$0") and a negative one is in the alert red (`ui/MoneyText.tsx`, `money` in
`game/format.ts`) [T-722]; the cash (top bar and this tab) has four ("$11.36B", "$893.6M"; to the dollar under a million) and is red
with its minus when negative; hovers may give more. History
[T-447, `ui/MoneyHistory.tsx`]: a Days / Hours switch (Days by default, kept while the game is
open) over three charts. The first has fares (accent) and running costs (grey) as lines, scaled to
the larger of the two, and one-off costs (new trains and construction) as pale bars on the same
scale, cut at the top with a dark cap when larger, so a build day does not flatten the lines
[T-452]; the top's figure at the right. Then net as bars up or down from zero (losses red). Then
train rides [T-565] as bars up from zero, the top's figure at the right (hover: it exactly), the
region's with the place on and the network's with the globe, from the same ledger entries (rides
are counted as in track-and-service.md 6.4; a day from a save made before T-565 has none). Days are the ledger's, one each in order (T-488), up to the past 30;
hours up to the past 48; a young game's few sit at the right of at least 7 days or 12 hours. The
day or hour now is drawn as it fills (a dashed last segment, a paler bar) and moves as fares land.
Hovering a column gives its day or hour ("Fri 5", "Sat 6, 04:00", "so far" for now) with fares,
costs and their parts, or net, or its train rides. Days and hours are the game clock's (T-321). The
charts are saved with the game and are there after a reload [T-567]. Then, for the whole
network whatever the toggle: the fare curve
(each ride, each km, and what a 10 km ride costs; each amount steps with the buttons, or a click
on it opens a box to type it, Enter or leaving the box applying and Escape cancelling, with
the stepper's rounding to the cent and floor of $0 [T-626, `TypedValue` in `ui/widgets.tsx`]); cars
owned (prices in its hover), cars the lines need and spare cars [T-631]; the value of the track and stations built. Rules: [track-and-service.md](track-and-service.md)
6.4. The economy's 100x scaling is in the hover of "A day at this schedule".
**The Cities tab** [T-550, `ui/Cities.tsx`; its own tab since T-749, no longer last in Money],
whatever the Lines toggle says: a list of every region (the toggle's regions) with a built station, one row each: its name
(hover: its station count), its mode split bar as the top bar draws it (the same colours and the
same hover), its train rides per day and its money a day, fares less running trains (yesterday's, today so far
until the game has a first full day; hover: fares and running trains, then new trains and
construction, left out of the figure, when there were any; losses red), and its points, those of the
past 24 game hours, changing once a game hour (hover: the exact figure, points a day with the
riders now on the day shown, and the rule) [T-684, T-694]. A star after the name marks a region
whose train share has reached 10% (hover: when it was earned, "Monday, day 71, at 14:08" in the
region's local time) [T-685]. The top bar's city name has the same points in its hover. A narrow
list keeps about six characters of each name (60 px): below that the star goes first, then the
money column (container queries on `.cities-box`) [T-693]. Sorted by rides, largest first; a click on a header sorts by that
column (name, train share, rides, money, points) as in the Lines tab; a click on a row moves the camera to
the region as search does, and the viewed region's row is tinted. It lives here and not in
Stations because the row's money is this tab's region ledger and a click on a row changes the
region the table above follows; a fifth dock tab button was ruled out by [T-532]. It sits at the
bottom so the figures above come first; there is no list until a station is built.

## Saved games

[T-068, T-241, T-251, T-320; `ui/SavesPanel.tsx`; save format and slot rules in
[architecture.md](architecture.md) "Saves"]

- The floppy disk button left of the gear, or the top bar's Autosave off notice, opens a centred
  panel over a dimmed page, closed by Esc, its close button or a click outside.
- One card per browser slot: a 192 x 128 picture of its network (track grey, each line in its
  colour, on the land colour of the theme the slot was saved in, drawn when the slot is saved, at
  twice that resolution) [T-543], its name, "Autosaved 2 min ago"
  for the game being played or "Saved ..." for the others (exact time on hover), the game day, the
  cities its stations are in, lines, stations and cash. The game being played is marked Playing;
  every other card has Open (closes the panel and loads it) and Delete (asked first); every card
  has Rename. A slot with no stored picture shows "No picture yet" until it is next saved.
- Under the Playing card, its backups [T-744], newest first: "Backup 4 min ago · Day 26" (exact
  time on hover) and Restore as a copy, which closes the panel and opens the backup in a new slot.
- Under the cards: Save as a copy (with a name), Save to a file
  (`anitabuilder-nyc-day-12.save`), Load a file, New game. Load a file asks nothing: it closes the
  panel and opens its game in a new slot, the game being played saved first [T-743; the slot rules
  in architecture.md]. New game shows the new game sheet (below) in the panel, headed "New game";
  its Start closes the panel and opens the new game the same way, its Back returns to the cards.
- A page opened with `?new=1`, `?synth=1` or `?demandTest=1` does not autosave: the top bar says
  "Autosave is off in this test page. Save a copy to keep the game.", a `?new=1` page marks no
  card Playing, and Save as a copy gives the game a slot of its own, which autosaves from then on.

**A new game** starts on day 1 at 07:00 with $3B, empty or (a sandbox) from a real network, on its
start city's registry centre and zoom, or on the world map (centre 20° E 30° N, zoom 1.8: the
Americas, Europe and East Asia) with no start city. Opening the page shows the start menu (below).
For a loaded game the URL's view says only which city or zone; the map starts at that city's
registry centre and zoom, or over New York with none [T-310].

**The new game sheet** [T-728, T-731; `ui/NewGame.tsx`], the start menu's and the saves panel's:
- **Mode**: Normal (lit to begin with) or Sandbox, two buttons, each with a hover saying what it
  means. A sandbox is the normal game with nothing refused for money and the cash hidden; demand,
  fares, running costs, line money, the Money tab, points and stars are worked out as in normal.
- **Start in**: New York, London, Paris, Tokyo and Toronto (`START_CITIES`, a fixed list), one lit
  at most, a second click unlighting it; then "Find a city or region", which lists up to eight of
  the registry's cities and zones whose name has a word starting with the text (case and accents
  ignored, so "hague" finds Rotterdam [The Hague]), biggest by local trips first, each with its
  country (the market row's code, named by the browser). Up and down move, Enter or a click picks;
  a pick of one of the five lights its button. Nothing picked starts on the world map.
- **Real network** (ticked to begin with): only for a sandbox in New York, London, Paris, Tokyo,
  Seoul, Toronto or Sendai, the cities with a real network (`REAL_NETWORKS`; architecture.md 11). The
  game starts from that network, built and running.
- **Start in Tokyo** or **Start on the map**, and Back. If a real network cannot be downloaded, a
  line says so and the sheet stays.
- The slot is named after the start ("Tokyo", "Tokyo sandbox", "World", then " 2" and on).

## Start menu

[T-677; `ui/StartMenu.tsx`; the startup order in [architecture.md](architecture.md) "Loading and readiness"]

- Every page load opens on a card over the bare map, in the loading card's look: "anitabuilder",
  New game, then "Saved games", one button per browser slot, the most recent first, each with the
  saves panel's picture (144 x 96) and facts (name, when saved, day, cities, lines, stations, cash).
  With no saved game it is New game alone. The dock, top bar and map controls are hidden under it,
  it takes every key outside itself, and nothing loads behind it but the map.
- A click on a save opens that slot (a sandbox's name has the "Sandbox" mark). New game shows the
  new game sheet (above) in the card, headed "New game", with Back to the menu; its Start starts the
  game in a new slot, so no saved game is replaced. The slot is first written once the network has
  something in it (a real network at once): a new game left empty leaves no card.
- URL switches that already say which game skip it: `?new=1` (a new game owning no slot; normal
  in New York, or `&mode=sandbox` and `&city=<zone id>`),
  `?synth=1`, `?demandTest=1`, `?paused=1` and `?continue=1` (the game opened last).

## Loading

[T-149, T-172, T-180, T-212, T-232, T-243, T-318; `game/loading.ts`, `ui/LoadingScreen.tsx`]

- Nothing is blocked: the map, inspectors and building work as soon as the network is drawn.
- **A loaded game** (the page opening, a file, a slot, a new game) holds only the clock until the
  zone in view has its final figures [T-347, T-421]: its fresh solve's last round, or that of the
  zone's own cluster, solved ahead of the rest (T-402: a Toronto view of a game with New York's
  network, about 6 s instead of 31 s); a new game until its viewed zone's figures, opened bare,
  arrive (T-351). Zones still solving then show their own loading bar and grey ("A zone still loading
  is grey" in [ui-demand-views.md](ui-demand-views.md)) until their final lands. The time is
  greyed and its hover says "The clock starts when demand has loaded". Meanwhile a card in the
  middle of the map says "Loading data" over a filling bar and the current step (Downloading city
  data, Opening city data, Calculating demand), naming no cities; the percentage is only the bar's
  hover ("42% loaded"). The bar counts to that final, so on the real New York save it fills
  for about 20 s. A slot whose cities are all open shows the bar for its solve alone. The card
  takes no input but the bar's hover (a press pans the map beneath) and goes when the clock starts.
  Loading is the one place with motion [T-647, T-677, Anita; `ui/Spinner.tsx`]: on top of the card a
  drawn bunny rides a toy train in the theme's accent (104 px wide; wheels turn, the track's ties run
  by, the train chugs, the ears bob, steam puffs), and a 26 px copy of
  it before the status tag's text; a small ring turning (`.spin`, 1.2 s a turn, 8 px, grey) in place
  of each line's crowding dot per demand level in the line panel until the line has figures
  (while no final result is shown or a solve is updating it; not after a failure or over dead
  space), and in a clicked bubble's panel in place of each line that waited ("Trip purposes", the far ends, "Inter-region trips") while they load.
  Nothing else animates. The figures behind it are the solve's estimates, or the last
  final result kept beside the save when there is one (T-328; the split's hover ends "(from the
  last time this game was open)").
- **The bar moves in proportion to time** [T-243, notes/T-243.md]: each step weighs as long as it
  is expected to take (a download by its rate, opening about 130 ms per MB of pack, the estimate
  about 0.6 s plus 2.2 ms per line stop, scaled by this browser's last load); inside a step the fill
  moves with time against that expectation until the step really ends, slowing if late, never
  going back.
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
  again (reload) and Play without demand, and takes every key until one is pressed.
- If demand stops later (a worker out of memory on a very big network, or an error) [T-757], a red
  alert under the demand buttons says "Demand ran out of memory" or "Demand stopped", its hover
  the cause and that a reload starts it again; the status tag goes, the figures shown stay.
- A game whose view is over dead space and whose stations reach no zone loads nothing: no card, no
  status, the clock not held, the split "–" [T-318].
