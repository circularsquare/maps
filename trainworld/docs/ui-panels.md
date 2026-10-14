[Back to the spec](../SPEC.md) · Interface: **panels** · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: look, dock, top bar and panels

## Look

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars); the network takes the line colours ([ui-map.md](ui-map.md)).

- **Zen Maru Gothic** everywhere, station names, map DOM controls and form controls included (the
  map container inherits it rather than MapLibre's Helvetica) [T-101]. Few font sizes and weights:
  the mock's 3 sizes and 2 weights are the ceiling. The basemap's labels use it too, Regular only,
  faint grey, from glyph tiles hosted beside the app (`app/public/fonts/`, `pipeline/glyphs.py`,
  Noto Sans merged in for other scripts); kana and kanji are drawn in the system font for now
  [T-057, notes/T-056.md].
- **Few colour tokens**: background, text, one accent, as CSS variables (`ui/style.css`), so a
  theme is a token swap. Light and soft, no paper texture. **Three themes**: pink (cream, sakura
  accent), blue (cornflower), green (matcha); dark later [T-039].
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
  press, scroll, key or the pointer leaving the page hides it. Text lives in `data-tip` (SVG shapes
  too); no `title` attribute is written anywhere, and one met under the pointer (MapLibre's
  controls) is turned into the tag first.
- **Every drawn quantity has a tooltip** (Anita) [T-194, notes/T-194.md]: bubbles, catchments,
  inter-region and city bubbles, line widths and track colours, trains, stations, crowds, demand
  circles, capacity tags, and in the panels the purpose split, pie and timing bars, the trip
  explorer's bars, the complex map's lines and transfer curves, the junction map's routes and the
  train's load bar.

## The dock

**Fixed layout, no floating windows** (the saved games panel and confirmations are the
exceptions). One dock on the left [T-252, `ui/Dock.tsx`]:

- A row of icon tabs, names in their hovers: **Build** (a wrench), **Lines** (a route with stops),
  **Stations** (a platform under its canopy), **Money** (a bill and a coin); at the row's right end
  the saved games button (a floppy disk) [T-251] and the **Settings** gear, which opens in the tab
  area like a tab. There is no City tab [T-220]; a browser that last had it open starts on Lines.
- Under the tab content, the **inspector** ([ui-inspectors.md](ui-inspectors.md)). The tab area has
  a fixed height and scrolls, so the inspector's top edge never moves when the tab changes.
- The player can drag the dock wider or narrower and move the divider between tab area and
  inspector; both are remembered per browser. Each tab keeps its own scroll position for the
  session, opening at its top the first time [T-215].
- Tool hints and a refused edit's reason show at the map's bottom left (`ui/Toolbar.tsx`).

## The top bar

`ui/TopBar.tsx`, left to right. **Every part has a fixed width** [T-341], so no city name or status
moves the split bar (the same length for every zone): the name box is cut with an ellipsis (the
whole name on hover), the percentages and counts are in tabular figures in boxes of fixed width, and
the bar's three lines (bar, labels, status) keep their height with nothing on the status line.

- **The goal number**: the focused zone's mode split as one bar (driving, train, walking, in the
  mode colours whatever the theme), the split of all its modelled local trip legs (commutes,
  everyday outings, school and university), so the shares are of rides a day, not people [T-342].
  The labels under it give the train share with its rides a day ("Train 18.5% 8.81M rides") and the
  walking and driving shares; the hover gives all three counts and the total. The player's job is
  to make the train share grow. It shows "–" and an empty bar until the zone has figures of its own
  (a solve's, or the bare pass's for a zone no station reaches; not the no-rail split a pack gives
  before that), and over dead space [T-318]. The hover says "on weekdays" or "at weekends" for the
  day shown. The zone named is the focused zone ([ui-demand-views.md](ui-demand-views.md)).
- **Status**, the third line under the split: while a solve runs after an edit, "Updating demand
  2/4…" and the split faded, the figures held [T-232], and the same while a reload shows the cached
  figures from the last time the game was open, with a hover saying so [T-328]; during startup's refinement "Estimate,
  refining 2/4…"; a city loading later "Loading data" with a thin bar after it (Loading, below).
- **Cash**, to the dollar, moving as fares are paid and running costs drain [T-284]. Fixed width
  (room for "-$99,999,999,999") with the digits left-aligned after the $ in tabular figures, so a
  changing figure never moves the rest of the bar [T-305, T-310].
- **The clock**: day of the week, day and time ("Sat 13", hover "Saturday, day 13"), the local
  time where the player is looking [T-309]; the demand level ("High demand"); pause and three
  speeds. **Space** pauses and resumes at the speed that was running, even with a button focused.
- **Narrow maps**: secondary labels drop first; two rows below 520 px, three below 300 px, with
  cash, time, train share and all speed controls kept. Hovering the split or the time gives what
  was dropped [T-106].
- An **Autosave off** notice when this tab cannot autosave (saves, below); a click opens the saved
  games.

**Demand shown for** [T-260, T-266; Settings, display setting `dayShown`]: Today (the default:
the clock's day, switching as the clock enters and leaves a weekend), Weekdays or Weekends. It
sets the day every demand view shows, the split included. There is no other mark of it. A switch
of day shows at once (the bubbles' catch-up is for a new forecast, not another day). The trains,
platforms and the panels' Today counts always follow the clock.

## Stations full

`ui/StationsFull.tsx` [T-231, T-284]: under the demand selector, a red-outlined button "5 stations
full" counts the station complexes where, now, a platform side holds as many riders as fit on it
(`platformRoom`, [track-and-service.md](track-and-service.md) 6.6) or the last train to leave a
platform was full and left riders behind (until the next train). Its hover lists them, most left
behind first, with riders waiting and left behind (twelve at most, then "and 3 more"). A click
selects the next one in that order and moves the map there at zoom 15 or closer, where crowds are
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

[T-183, `ui/keys.ts`] One rule for the game's keys (1-4, Esc, Space, Ctrl+Z/Y, the line tool's
Enter and Backspace) and search's `/`, Ctrl+K and Ctrl+F. A text field keeps every key (search's
own box gives Ctrl+K and Ctrl+F to selecting its text). Another form control keeps only its own
keys (arrows, Home, End, Page Up and Down, Enter, Space, and in a select a letter); the rest reach
the game. A select picked with the mouse gives up focus at once, so the next Space pauses; one
reached with Tab keeps it. Buttons show the browser's focus ring for keyboard focus only [T-169].

**Confirmations** [T-113]: a centred "Are you sure?" dialog with operation-specific text, cancel
and action buttons. Cancel has initial focus, Tab stays inside, Escape cancels, focus returns when
dismissed, and the map and tools are blocked meanwhile.

## Settings

`ui/SettingsTab.tsx`, remembered per browser, in order: Bubble size (only while Homes or
Destinations is on; 0.5-2x, [T-099]), Demand shown for, Track display (four modes with their key,
[ui-map.md](ui-map.md)), Show (trains, station names, capacity markers, map labels), Riders on trains
and platforms ("How full each train is", "How full each platform is", both on by default)
[T-164], the theme, Panel and text size, and Explore a trip at the bottom [T-220]. Station demand
circles are at the top of the Stations tab instead [T-213].

**Panel and text size** [T-247]: 80% to 120% in 5% steps, 100% by default, no reset button. It
zooms the dock, top bar, map buttons, search, loading card, tool notes and dialogs, and sizes the
text of the map's DOM tags and the hover tag; the map itself is never scaled.

## Money tab

Cash; yesterday and today (fares, running trains, new trains, construction, net); the fare curve
(each ride, each km, and what a 10 km ride costs); cars owned (prices in its hover) and cars the
lines need; the value of the track and stations built. Rules: [track-and-service.md](track-and-service.md)
6.4. The economy's 100x scaling is in the hover of "A day at this schedule".

## Saved games

[T-068, T-241, T-251, T-320; `ui/SavesPanel.tsx`; save format and slot rules in
[architecture.md](architecture.md) "Saves"]

- The floppy disk button left of the gear, or the top bar's Autosave off notice, opens a centred
  panel over a dimmed page, closed by Esc, its close button or a click outside.
- One card per browser slot: a 192 x 128 picture of its network (track grey, each line in its
  colour, drawn when the slot is saved, at twice that resolution), its name, "Autosaved 2 min ago"
  for the game being played or "Saved ..." for the others (exact time on hover), the game day, the
  cities its stations are in, lines, stations and cash. The game being played is marked Playing;
  every other card has Open (closes the panel and loads it) and Delete (asked first); every card
  has Rename. A slot with no stored picture shows "No picture yet" until it is next saved.
- Under the cards: Save as a copy (with a name), Save to a file
  (`anitabuilder-nyc-day-12.save`), Load a file, New game (after a confirmation; it replaces only
  the active slot).
- A page opened with `?new=1`, `?synth=1` or `?demandTest=1` does not autosave: the top bar says
  "Autosave is off in this test page. Save a copy to keep the game.", a `?new=1` page marks no
  card Playing, and Save as a copy gives the game a slot of its own, which autosaves from then on.

**A new game** starts on day 1 at 07:00 in an empty New York with $6B. Opening the page continues
the last game. The URL's view says only which city or zone; the map starts at that city's registry
centre and zoom, or over New York with none [T-310].

## Loading

[T-149, T-172, T-180, T-212, T-232, T-243, T-318; `game/loading.ts`, `ui/LoadingScreen.tsx`]

- Nothing is blocked: the map, inspectors and building work as soon as the network is drawn.
- **A loaded game** (the page opening, a file, a slot, a new game) holds only the clock until the
  first estimate of every city it needs has landed; the time is greyed and its hover says "The
  clock starts when demand has loaded". Meanwhile a card in the middle of the map says "Loading
  data" over a filling bar and the current step (Downloading city data, Opening city data,
  Calculating demand), naming no cities; the percentage is only the bar's hover ("42% loaded"). A
  slot whose cities are all open shows the bar for its solve alone. The card takes no input but the
  bar's hover (a press pans the map beneath), has no animation or spinner, and goes when the clock
  starts.
- **The bar moves in proportion to time** [T-243, notes/T-243.md]: each step weighs as long as it
  is expected to take (a download by its rate, opening about 130 ms per MB of pack, the estimate
  about 0.6 s plus 2.2 ms per line stop, scaled by this browser's last load); inside a step the fill
  moves with time against that expectation until the step really ends, slowing if late, never
  going back.
- The crowding rounds then refine the figures with the clock running; the status says "Estimate,
  refining 2/4…" (hover: the figures are an early estimate).
- **After an edit** nothing changes until the new solve's final round: split, riders, bubbles and
  the boarding account keep the last finished forecast, the split faded and the status "Updating
  demand 2/4…", then all change at once [T-232]. Queries asked meanwhile (a catchment, a bubble
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
- A game whose view is over dead space and whose stations reach no zone loads nothing: no card, no
  status, the clock not held, the split "–" [T-318].
