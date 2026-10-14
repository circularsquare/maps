[Back to the spec](../SPEC.md)

## 8. Look and UI

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars). The network is not styled by us: lines, trains and stations take the line colours.

- **Panels show what can change** (Anita, T-253): labels, numbers and states, no explanatory notes
  or how-it-works sentences; a fact worth keeping goes in a hover on the label it explains.
  notes/T-253.md lists every sentence taken out and where it went.
- **Fixed layout, not floating windows.** One dock on the left with tabs and an inspector area
  under the tab content. The tabs are icons, their names in the hover (T-252): Build a wrench,
  Lines a route with stops, Stations a platform under its canopy, Money a bill and a coin; at the
  row's right end the saved games button (a floppy disk, T-251) and the Settings gear. The saved
  games are the one floating panel (below). A top bar for the mode split, money,
  day and time, the demand level ("High demand"), pause and three speeds. At narrow map widths,
  the bar drops secondary labels and uses two rows below
  520 px (three below 300 px), with cash, time, train share and all speed controls available.
  The cash has a fixed width (room for "-$99,999,999,999", T-305): the minus and the $ stay in
  their places and the digits follow the $ directly (left-aligned, T-310) in tabular figures, so a
  figure that changes to the dollar never moves the rest of the bar and no gap grows after the $.
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
- **Trains show who is on board** (T-083, T-164): the train inspector's "On board" shows this
  train's own riders on the stretch it is running, from the boarding account
  (docs/track-and-service.md 6.6), against seats and crush (a bar with a tick at the seats;
  "Empty", "Seats free", "Standing", "Full"), and the typical train this hour (the mean of its
  line's trains in the same direction drawn this hour, on the same stretch, from the same account,
  T-230) for comparison; the heading's hover says it is counted train by train from the demand forecast. A train is never over full: whoever does not fit waits on the
  platform.
- **Riders waiting** (T-164): a station's inspector lists, under "Waiting now", each line and direction ("To Norwood-205 St") with the riders on the platform now, the
  number in red; under it, in red, how many
  full trains left behind this hour and the longest wait, or else the riders arriving this hour and
  the longest wait (a headway). Station view only; in complex view each line's row gives its
  riders waiting now instead (T-250). The counts drop the moment a train leaves one of
  the station's platforms with whoever fitted, the instant the boarding account boards it (T-238:
  the panel sets a timer for the station's next departure from the hour's events), and grow with
  arrivals each game minute in between, as the platform fill on the map does.
- **Today** (T-293, Anita: "lets not even show the whole day forecast cuz it shows stuff in the
  future. lets only show so far today, and call it 'today'"): the line and station panels count
  the day from midnight to now instead of forecasting a whole day. A line's "Today": riders
  (boardings on its trains, by purpose shares), fares taken, running cost so far, its fullest
  train so far ("100%" when one left packed full, T-239, else "N% full"), how many trains left
  full and when (spans under half an hour apart, at the stop where each is fullest; T-230), and
  riders left on the platform by full trains; each stop's boardings today in the stops list. A
  station's lines with their boardings today and its "Today" boardings with purpose shares; a
  complex's list has a "Today" column (T-250). The Lines tab shows riders today and the Stations tab
  boardings today, each a column that sorts (below); the sort re-runs each game hour. Counted
  by the clock worker from the boarding account's departures up to now (`TrackApi.today`: a
  stop's boardings since midnight are the riders arrived by its last departure less those still
  waiting, less the same at midnight), the account of today's kind of day; fares from each line's
  hourly fare paid as the cash is (T-284). Asked four times a wall second while a panel shows
  them and the clock runs (`game/today.ts`, about 5 ms a round trip on the real New York save),
  and only the small components showing them re-render. Demand's busiest-hour average train is
  not shown (docs/track-and-service.md 6.6 says why). The map's demand views (bubbles, traffic
  widths, station circles, the complex map's widths, the commuters and longer trips lists) stay
  demand's whole-day picture.
- **Stations full** (T-231, Anita: "a global alert in the top left under the demand buttons that
  says like n stations full"; `ui/StationsFull.tsx`): under the demand buttons, a red-outlined
  button "5 stations full" counts the station complexes (one count a complex) where, now, a
  platform side holds as many riders as fit on it (`platformRoom`, 1,250 a side on 150 m, T-219;
  the whole platform's room where trains use one side only, as drawn) or the last train to leave
  a platform was full and left riders behind (T-284: until the next train, not the hour's total,
  which made the list jump as each hour turned). Its
  hover lists them, most left behind first, with riders waiting and those the last full trains left (twelve at most,
  then "and 3 more"). A click selects the next one in that order (the member with the most
  waiting) and moves the map there at zoom 15 or closer, where crowds are drawn. Hidden when
  there are none; updated each game minute. On a narrow map it sits beside the search glass.
- **Close-up trains** (T-103, T-107): from zoom 15, trains show the line's actual number of 20 m cars,
  each positioned on the route and angled along its own chord, so the consist snakes around
  curves. Cars have small gaps, rounded rectangular bodies, and windows/front cab when large
  enough. The whole consist stays on the route at terminals. Clicking any car selects the train.
  Cars and compact trains share the same zoom-dependent width, growing with physical width at
  very close zooms, so the switch never shrinks the train. Further out, trains use a compact
  capsule with length based on their car count; all positions
  remain GPU-derived from keyframes, with one instanced draw call for trains at either scale.
- **A selected train stands out** (T-242, Anita: "highlight the train in an obvious way"): it is
  drawn over every other train, its colour, fill gauge and red packed edge unchanged, inside a
  halo round the whole consist: from the train out, a dark outline like a selected station's
  (T-189), a white band, a dark ring of the train's own shape 6 px out, and a soft glow beyond.
  The ring is what makes it readable at zoom 11-13 where the train is a few px. Two more instanced
  draws (the halo as one pill, then the train) only while a train is selected. A click tries the selected train first, as it is drawn on top.
- **A selected train is followed round** (T-245, Anita: "if it just turns around and goes back we
  should continue following it"): when its trip ends at a terminal and the track model hands its
  train to the next departure there (docs/track-and-service.md 6.3, `Trip::next`), the selection
  moves to that trip the moment it leaves, from the spot the train stood at, so the inspector and
  the highlight carry on with no gap. A train that leaves service at the terminal stays selected
  where it finished ("This train finished its trip and left service").
- **The inspector shows whatever is selected**: a line, a station, a stretch of track, a junction,
  a train, commuter bubbles, a cell or zone. Nothing selected means it is blank (no text at all), not a default line. **Its top edge
  never moves when the tab changes**: the tab content area has a fixed height and scrolls.
- **No City tab** (T-220, Anita: "redundant" now that the city bubbles exist; it was T-143's City
  pane). What it offered went elsewhere: a city's inter-region trips by place are the city overview's
  bubbles and the city inspector (T-186, "Zoomed far out" below); City centre is the city
  inspector's Go to button and search's city results; Whole region is zooming out, where the
  overview draws every city; Explore a trip is a button at the bottom of Settings (key 4 as
  before). Its top-five train/other bars were dropped: the inspector lists every place with all
  and train counts, and the overview colours each city by its train share. The tabs are Build,
  Lines, Stations, Money (icons since T-252) and the Settings gear; a browser that last had the City tab open starts
  on Lines. The top bar follows the zone under the view when
  panning (no name over dead space, T-318) and shows its modelled local trip share (its hover says "on weekdays" or "at
  weekends"); the clock shows the day of the week before the day number ("Sat 13", the hover
  "Saturday, day 13"). **Demand shown for** (T-260, T-266; Settings, above Track display, a
  remembered display setting `dayShown`): Today (the default, hover "The clock's day: weekdays,
  then Saturday and Sunday"), Weekdays or Weekends. It sets the day every demand view shows, the
  split included; on Today the views switch as the clock crosses into or out of a weekend. There
  is no other mark of the day shown: the clock says the day, and Weekdays or Weekends are a choice
  the player made. A switch of day shows at once: the bubbles' catch-up over a game day (T-156) is
  for a new forecast, not another day. The panels' Today counts (T-293) follow the clock's own
  day, not this setting. A line inspector's riders count
  transfers on every line actually boarded; the purpose split under them (Riders by purpose,
  below) is where inter-region trips show; there are no separate Local trips and Inter-region trips rows
  (T-182).
- **Confirmations** (T-113): a centered Are you sure? dialog with operation-specific text,
  cancel and action buttons. Cancel receives initial focus, Tab/Shift+Tab stay in the dialog,
  Escape cancels, focus returns when dismissed, and background map/tools are blocked.
- **Sorting the lists** (T-310): the Lines and Stations lists sort by clicking a column header,
  with no dropdown. Lines has Name, Riders (riders so far today) and Trains; Stations has Name and
  Today (boardings so far today). A click sorts by that column (names A to Z, figures biggest first),
  a second click reverses it, and the sorted column has a small arrow (up for ascending). Each
  header's hover says what it sorts by. Names sort naturally, ties fall back to the name. Defaults:
  Lines by name, Stations by boardings. The sort is kept while the page is open and re-runs each
  game hour (T-293).
- **Line management** (T-109 to T-112): the Lines pane has New line above its list. Actions
  belong above lists of arbitrary length. New line immediately selects an empty zero-train line
  in the inspector, enters Edit stops and focuses its unique `Line N` name with all text selected.
  The number follows persisted creation order, including copies, and skips names already used.
  Copy line duplicates the route and all settings independently, then opens naming/editing.
  Express copy (T-255) makes "<line> express" with its stops moved to their twins on quad
  track's express pair (docs/track-and-service.md 6.1) and opens stop editing on it at once, no
  naming step; its hover says what it does.
  Clicking the inspector badge edits colour and a persistent symbol; multiple characters widen
  the rounded badge, with its text raised one pixel. Blank symbol restores name-derived initials.
  Edit stops, Copy line, Express copy and Delete line sit above the stop list. Red removal crosses appear only
  while editing, including on one/two-stop lines. Delete line requires confirmation. Map clicks
  test every insertion position and choose the shortest valid route while keeping existing stop
  order; Enter or Escape finishes. Incomplete lines remain planned and do not run trains.
  **Stop editing mode** (T-233): while a line's stops are edited (Edit stops, New line, Copy line)
  the map hides the demand views (bubbles, a station's catchment, inter-region trip bubbles, the trip
  route, capacity tags, traffic line widths and station circles), draws the other lines and their
  trains at 30% (the edited line's trains stay full) and
  puts a round toggle marker on every station, over everything (`map/stopMarkers.ts`). A stop is
  a disc in the line's colour inside a white ring inside a dark one, with no check (T-239);
  other stations are hollow. Stations a stop can be added at without changing the line's path
  (on track the route passes over, or past either end on track the line can carry on over, which
  only lengthens the path, T-237) are drawn on top at 80% of a stop's size (T-239, Anita:
  "reachable stations a bit smaller"); stations the line can reach only by changing its route are
  65% and grey; stations it cannot reach at all (no track, or only by turning back) are the same
  size and greyer. The tiers come from the clock worker (`TrackApi::stop_tiers`), which tries
  each station with the same insertion search a click makes, so a marker always says what its
  click does; under 10 ms a line on the real New York save. Until the answer lands (a frame or
  two) an estimate from the track graph shows. A click toggles the stop; adding takes the shortest
  valid insertion (T-112), removing leaves the others in order. Hover says "Click to add a stop",
  "Click to add a stop and change the route", "Click to remove the stop" or "No track joins it to
  this line", under the station's name; a click on the last says why in its own tag for four
  seconds. The markers take no pointer events, so the wheel and dragging work over them. The panel
  at the map's lower left shows "Editing stops on <line>: N stops" and a Done button; Enter and
  Escape also leave.
  The line inspector has, in order, Schedule (trains for each demand level), Today for a running
  line (riders by purpose, fares, running cost and its trains so far, T-293), then the stops with
  their times and boardings today, and the round trip split into running, dwell, turning and
  waiting.
- **Demand readiness** (T-143): Homes/Destinations fade (their text; the buttons stay opaque, as
  the bar has no background of its own and so no strip between Destinations and the purpose
  filter, T-192) while the focused city has no published
  result, preserving the stored selection and its colour, with Calculating demand on hover.
  None remains available; ordinary recomputes retain the prior result and usable controls.
- **Purpose filter** (T-158): a dropdown at the end of the None/Homes/Destinations bar: All
  purposes (the default), Work, School and university, Shopping and errands, Social and leisure,
  Medical, Airport (T-142). Inter-region trips are not an option (T-305): they show under All
  purposes only. School and university are one shown purpose everywhere
  (filter, pie, purpose split, timing charts, tooltips) in school's blue-green, though the model
  keeps them apart, as their homes, places and timing differ (T-192, `game/purposes.ts`
  `membersOf`); choosing it draws both. With one purpose chosen, a swatch in its purpose colour (the purpose pie's, T-143;
  its colour) sits before the name and the control takes the accent border, and every
  demand drawing on the map counts that purpose only: line widths in Line with traffic thickness,
  the station demand circles (complex view too), the Homes/Destinations bubbles and their hover
  counts ("12,400 school and university trips a day at homes here"), a bubble selection's own and far-end
  bubbles, and the track tag. Widths, circles and bubbles keep their usual scales, so a small
  purpose (school and university) draws thin and purposes stay comparable. Work is every city's commutes; the six everyday purposes (five as shown) are each
  city's own everyday layer (every registered city since T-142);
  inter-region trips are the regional market, which has no purposes of its own, so their bubbles
  show only under All purposes (they still count in All), and a selected station's commuter catchment (commuters
  only) only under All purposes or Work; the station inspector then says the commuters are not
  drawn. Train fill and crowding always count everyone aboard: the filter changes what is drawn,
  never the solve. A bubble selection's inspector counts and far-end places follow the filter
  with a note; its purpose pie and departure charts keep every purpose. Switching purpose clears
  the city's bubbles until that purpose's are ready rather than leave another purpose's on show.
  Not remembered: each page load starts on All purposes. Like the bar's buttons it shows the
  browser's focus ring for keyboard focus only, round the whole control, never after a click (T-169).
  While the viewed city's pack is still loading the hover text is "Loading Boston, 37%" (T-149).
- **Riders by purpose** (T-165, `ui/inspector/PurposeSplit.tsx`): under a line's riders today
  and a station's boardings today (a complex's in complex view) a thin stacked bar splits riders
  by purpose in the purpose colours, in the filter's order, with a small key of names and
  rounded shares ("<1%" for a sliver) below it; hovering a segment or key entry gives the share
  ("Work: 48% of riders"; no forecast count since T-293). The shares are `DemandView.purposes`
  over the same stops the inspector counts. Every purpose is listed, those with no riders here in muted text. Under
  the purpose filter the split still shows every purpose; the one the map draws is ringed in
  the bar and bold in the key, so a thin line on the map can be read against its whole.
- **Loading** (T-149, T-172, T-180): nothing is blocked while demand loads. The map, the
  inspectors and building work as soon as the network is drawn, and the mode split shows "–"
  until the city has opened. At startup, and after any whole-game load (a file, a slot, a new
  game), the clock is held until the first estimate of the viewed city's demand, and that of every
  city with construction, has landed; the time is greyed meanwhile and its hover text says "The
  clock starts when demand has loaded". While it is held a card in the middle of the map says
  "Loading data" over a filling bar and the current step (Downloading city data, Opening city
  data, Calculating demand), naming no cities (Anita, T-212); the percentage is only the bar's
  hover ("42% loaded", T-243: Anita wanted a bar, not a number). A slot whose cities are all open
  shows the bar for its solve alone. The bar moves in proportion to time (T-243): each step weighs
  as long as it is expected to take (opening a pack about 130 ms per MB, the first estimate about
  0.6 s plus 2.2 ms per line stop, scaled by this browser's last load), and inside a step the fill
  moves with the time spent against that expectation until the step really ends, slowing near
  its end if the step runs late; it never goes back. notes/T-243.md. The card takes no pointer or
  key except the bar's hover: a press on it pans the map beneath. It has no transition, animation
  or spinner (the fill jumps to each new value), and it goes the moment the clock starts. The crowding rounds then refine the figures with the clock running, and
  while they do the top bar's status says "Estimate, refining 2/4…" (its hover: "These figures
  are an early estimate. They settle when the forecast is finished."). **Updating demand**
  (T-232, Anita: "when i edit a line minorly the ridership goes up from 19% to 20.5% then back to
  19% as it recalcs"): after an edit nothing changes until the new solve's final round: the
  split, line and station riders, bubbles and the boarding account keep the last finished
  forecast, the split faded and the status "Updating demand 2/4…" (hover: the figures are for the
  network before the last change), then all of them change at once. Estimates are shown only
  where there is nothing final to hold (startup, a whole-game load, a city just opened). Queries
  asked meanwhile (a station's catchment, a bubble selection, a trip) wait for that final round,
  since the workers have moved on from the shown solve. A city
  loading later (a pan to it, construction reaching it) holds nothing and has no card: the top
  bar's status says "Loading data" with a thin filling bar after it (T-243) where it otherwise
  says "Updating demand 2/4", and its hover names the cities and the step ("Loading Boston, 37%.
  Opening city data"); in a narrow map
  column the bar gives way first and the status then ends in an ellipsis. If demand cannot load a screen says "Demand could not load" with the reason and
  two buttons, Try again (reload) and Play without demand, and takes every key until one is
  pressed. A game whose view is over dead space and whose stations reach no zone loads nothing
  (T-318): no card, no status, the clock is not held, the mode split reads "–".
- **Settings** (`ui/SettingsTab.tsx`), in order: Bubble size (only while Homes or Destinations
  bubbles are on; the bubbles themselves are the selector at the map's upper left), the four track
  display modes, show or hide trains, station names, capacity markers and map labels, "Riders on
  trains and platforms", the theme, Panel and text size, and Explore a trip at the bottom.
  The station demand circles and "One circle per station complex" are not here but at the top of
  the Stations tab, above its list (T-213), what the circles show in the checkbox's hover. A "Riders on trains and
  platforms" section (T-164) has "How full each train is" and "How full each platform is" (both
  on by default), how the platform fills in its hover. The pane opens from the gear at the end of
  the tab row and shows in the tab area like a tab, so the inspector stays put. Remembered per
  browser.
  **Saved games** (T-241, T-251, SPEC 11) are not in Settings (Anita: "we should have a separate
  saves panel"): the floppy disk button left of the gear (and the top bar's Autosave off notice)
  opens a centred panel over a dimmed page, closed by Esc, its close button or a click outside it.
  No note on how saving works. One card per browser slot: a 192 x 128 picture of its network
  (track grey, each line in its colour, drawn when the slot is saved, at twice that resolution),
  its name, "Autosaved 2 min ago" for the game being played or "Saved ..." for the others (the
  exact time on hover), the game day, the cities its stations are in, its lines, stations and
  cash. The game being played is marked Playing in the accent; every other card has an Open
  button, which closes the panel and loads that game, and Delete (asked first); every card has
  Rename. Under the cards: Save as a copy (with a
  name), Save to a file, Load a file, New game. A slot saved before T-241 shows "No picture yet"
  until it is next saved. A page opened with `?new=1`, `?synth=1` or `?demandTest=1` (checks and
  the stress network, T-320) does not autosave: the top bar says "Autosave is off in this test
  page. Save a copy to keep the game.", no card is marked Playing in a `?new=1` page, and Save as
  a copy gives the game a slot of its own, which autosaves from then on.
  **Panel and text size** (T-247, Anita: the fonts felt a bit big): 80% to 120% in 5% steps, 100%
  by default, remembered per browser; no reset button (Anita). It zooms the dock, top bar, map buttons, search, loading
  card, tool notes and dialogs, and sizes the text of the map's DOM tags (station names, capacity
  and tool tags, inter-region trip labels) and of the hover tag; the map itself is never scaled.
- **Track display** (T-105): four mutually exclusive choices: Line (uniform width), Line with
  traffic thickness (default, riders in the current period), Height, and Max speed. Height and
  Max speed colour the track itself with uniform width; trains keep their line colours. Max speed
  is the geometry's allowed speed at each section, including curve limits, capped at 160 km/h;
  it is not a train's instantaneous speed. A 40/80/120/160 km/h colour key runs from warm red
  through ochre and green to blue. The choice is remembered per browser.
- **Height colours** (track coloured by level): warm above ground, grey at ground, cool below:
  +3 #c2410c, +2 #ea7a1a, +1 #f2b134, 0 #8c8c8c, -1 #5aa3d8, -2 #2f6fbf, -3 #23408e. Trains
  keep their line colour.
- **Station names and capacity tags share space** (T-184, `map/stationLabels.ts`,
  `map/capacityMarkers.ts`): names are placed first and never move or hide for a tag. Each
  capacity tag (docs/track-and-service.md 6.2) then takes the first of six spots around its
  resource clear of every shown name (3 px clearance) and of the tags already placed: above
  right, above left, below right, below left, right, left, a corner 5-6 px from the resource. A
  tag with no clear spot hides, like one losing to a worse delay. Tags follow names during a pan
  and when names are re-placed or switched off. While any tool other than select is on (track,
  station, delete, line, trip) the tags take no pointer events and show no hover, so a press,
  move or double click on a tag reaches the station or track under it (T-224). With the select tool a tag that opens nothing of its own (not a junction, no bubble above it) selects what is drawn under it, its station first, as a click on the map there would (T-256).
- **Tab scroll** (T-215): each dock tab keeps its own scroll position for the session (the Build
  tab its options list), opening at its top the first time, never at another tab's scroll.
- **Track tooltips** (T-157, `map/trackHover.ts`): with the select tool, resting the pointer on
  a line or track for 0.1 s shows a tag by the pointer (the demand views' tag style) saying what
  the track display draws there. Line: the line's name. Line + traffic thickness: the name and
  "41,200 forecast riders a day in both directions", the line's daily forecast on that
  stop-to-stop stretch, every period summed (the width itself is the current period's hourly
  rate); counts round to the nearest 100 above 1,000. Under the purpose filter (above) the count
  is that purpose's and a third line says so ("Shopping and errands trips only"). Height: "Tunnel at level −2", "Ground
  level" and so on, by the level the colour shows. Max speed: "Max speed 120 km/h". A planned
  line adds "Not running yet", a broken one "Not running"; bare track in the line views says
  "No line runs here"; before the first solve the traffic view says "Calculating demand". Once
  shown, the tag follows the pointer along the track and updates; leaving the track, pressing,
  dragging, zooming or picking a tool hides it, and the next one needs another rest. It follows
  the pick order: bubbles, capacity tags and editing handles above the network leave it hidden,
  and junctions say nothing.
- **Network tooltips** (T-194, `map/networkTips.ts`): the same tag (it wraps at 320 px, as wide as
  its text when shorter; T-305), after the same rest, for what
  is drawn on the network besides track. A train: "Line 1 train", its riders aboard ("340 riders
  aboard", the train inspector's count from the boarding account) and "352 seats, 1,280 packed
  full", the two marks of its fill gauge; a line the account has no riders for yet says "About 300
  riders on a train in the busiest hour", which the gauge then shows. A station: its name, then its
  lines by their symbol, or with none their name without a leading mode word ("subway 6X" is
  "6X", "lirr Port Washington Branch" "Port Washington Branch"), each once across the complex's
  stations (a station drawn as its own box lists its own lines only; T-305), with
  demand circles on its circle's "41,200 boardings and alightings a day" (in complex view "at the
  complex's 4 stations"), and where platform crowds are drawn a line per line and direction with
  anyone waiting ("120 waiting for subway 1 to Van Cortlandt Park-242 St", "after a full train left
  some behind" while that side is red) or "Nobody waiting on the platforms", then the fill drawn:
  "Platform 21% full on one side and 4% on the other (room for 1,250 a side)", or "Platform 21%
  full (room for 2,500)" where trains use one side only (T-219). Zoomed out each dot gives its own
  station's tag with its own lines (T-313, T-305), and on a
  dot the crowd lines are its tint's numbers instead: "1,257 waiting", "Fullest
  platform 45% full" (or "full"), and "A full train left some behind" while it is red (T-228). The crowd
  is inside the platform, so the station's own hover covers it. A demand circle away from any station, under the network: its busiest station's
  name and the count. The purpose filter's third line applies to circle counts. The tag refreshes
  each game minute while shown, as trains move and crowds change.
- **Every drawn quantity has a tooltip** (Anita, 2026-10-12; audit T-194, notes/T-194.md): the
  bubbles, catchments, inter-region trip and city bubbles (demandHover.ts), line widths and track
  colours (track tag), trains, stations, crowds and demand circles (network tag), capacity tags,
  and in the panels the purpose split, pie and timing bars, the trip explorer's bars, the complex
  map's lines and transfer curves, the junction map's routes and the train's load bar.
- **One tag, never the browser's tooltip** (Anita, 2026-10-12; T-210, `ui/tip.ts`, notes/T-210.md):
  every hover outside the map's canvas (panels, top bar, buttons, the complex and junction maps,
  the map's DOM markers) is the map tags' look, shown the moment the pointer is over the element,
  following the pointer below right of it and flipped to stay in the window; a press, scroll, key
  or the pointer leaving the page hides it. Elements carry their text in `data-tip` (SVG shapes
  too, never a `<title>`); no `title` attribute is written anywhere, and one met under the pointer
  (MapLibre's own controls) is turned into the tag before the browser's would show.
- **Search** (T-148, T-211, `game/search.ts`, `ui/Search.tsx`): most of the time a small square
  button with a magnifying glass at the map's upper right ("Find a station, line or place (/ or
  Ctrl+F)" on hover); when the map is narrower than 680 px it sits under the demand selector. A
  click on it, `/`, Ctrl+K or Ctrl+F, from anywhere but a text field or the confirmation dialog,
  opens the input "Find a station, line or place" in its place and focuses it, at once (no
  animation); Ctrl+F is kept from the browser's find only then, and in another text field it is
  the browser's. Up and down move through the results, Enter or a click picks, Escape clears it
  and closes it back to the button, as does leaving the box empty; left with text in it, the box
  stays open with the text. It finds the save's stations, its lines (by name, or the
  badge's symbol exactly), the registry's cities, and the towns and neighbourhoods on each city's
  OSM place list (T-177, `game/placeList.ts`, fetched the first time the box is focused, never at
  startup; a place's other OSM names count too), ignoring case, accents and punctuation:
  exact names first, then names starting with the text, a word starting with it, containing it,
  then clipped nicknames ("south philly" finds South Philadelphia: each word of the text starts
  like a word of the name, bar its last two letters, keeping at least four).
  Equal matches list cities, then lines, then stations (busiest first), then places (bigger kinds
  first, then by population). A place list's own entry for a registry city is left out. Up to
  eight rows: a line shows its badge and "Line", a city "City", a place a hollow ring and its kind
  ("Neighbourhood", "Town", "Village"), a station its line badges (one per badge look, four
  at most; "Station" if none), blueprint stations muted. Results sharing a name get a muted second
  line telling them apart (T-183): the city when the other is in another city; in one city
  (two complexes more than 50 m apart, like New York's Times Sq-42 St and its 8 Av platforms) the
  lines only this one serves, a first word they share written once and three at most ("subway A,
  C, E", "subway 1, 2, 3 and 7 more"), or, when it serves none of its own, the compass direction
  it lies in from the others ("northwest"); places get the city, or the direction within one city. Stations of one complex that share a name are one result: picking it
  selects the platform with the most lines (T-182), which shows its complex like any station
  click (T-249). A pick moves the camera at
  once (no flight): a station to zoom 14 or closer, a line to its stops' bounds, a city to its
  registered centre and zoom (the city inspector's Go to view), a place to a zoom by its kind (a town 12.5, a suburb 13, a
  neighbourhood 14.5); the viewed city follows. With no build tool on it selects the
  station or line (a city clears the selection, a place clears a station or bubble selection);
  with a tool on it only moves the camera, so a half-drawn route or a line's stop editing carries
  on there. Addresses are not searched (Anita, 2026-10-12: no geocoder; notes/T-148.md).
- **Trip explorer** (T-195, `map/tripView.ts`, `ui/inspector/TripInspector.tsx`, `game/trip.ts`):
  how the demand model's people would make one trip. Key 4, from any tab (the tab stays), or
  "Explore a trip" at the bottom of Settings (T-220), turns on the trip tool (a crosshair; a
  second press or Esc leaves it); its clicks alternate between where a trip starts and where it
  ends, so a third click starts a new trip. The hint at the bottom left says which click is next.
  **Stations as ends** (T-221, Anita: "explore a trip by clicking two stations"): a click on a
  station (the pointer turns to a hand over one) takes the station as that end: the end is asked
  about at the station's own position, so the model reads it as the station's own cell, and the
  inspector names it ("From a home by Bergen St to work by Grand Central-42 St"; the ring's
  hover too). The station's own cell, not the busiest cell of its catchment, because the model's
  subzones are the cells of a zone sharing their nearest station: the station's cell is in the
  subzone whose nearest station is this one, so the trip mostly boards there, while the busiest
  catchment cell can be nearer another station. A station's inspector has **Trip from here**:
  the trip tool on with the station as the start; the next click, a station or a place, ends it.
  **From a
  selection** (T-208): with a Homes or Destinations bubble selection, or a selected station's
  catchment, a click on one of its far-end bubbles opens the trip for that pair (their hovers end
  "Click to explore this trip"), homes to destinations whichever end is selected. A bubble
  selection's end is its busiest cell (most legs at that end under the purpose filter), the far
  end the clicked bubble's cell with the most of the selection's trips; a station's end is its
  catchment cell with the most of its riders, the far end the clicked 2 km zone's cell with the
  most rail legs at that end. One cell a side because the explorer reads one place pair, and the
  busiest because a bubble's drawn centre can fall where nobody is. The inspector then names
  where it came from with a Back button that returns to that selection. The trip is a
  selection: it stays in the inspector after the tool is left, until Esc or an empty click
  clears it, and while it is selected panning does not change the viewed city. It is read as a
  commute from a home at the start to work at the end, and the numbers are the model's
  (docs/demand.md 4.4, "The trip explorer"), asked of the demand workers again at each published
  round. The inspector shows the period, the distance, a bar and rows with the chance of
  train and driving (walking too when it is at least 0.5%); under "By train" the main route as
  rows (walk to the first station, each line ridden with its chip and the station it goes to,
  each change and walk between stations, the walk from the last station; each row's minutes are
  its wait and ride), door to door, its walk, wait and ride minutes, the fare, and what the
  train counts as against driving (the model's perceived minutes), then how much of the trip's
  train riders take that route, with "Other routes" (up to four, by the lines ridden and the
  station reached, routes that read the same added together) and their shares; under "By car"
  the drive, its parking and what it counts as; and "Train by period", one row a period with
  the train's chance, a click showing that period (the clock's period until one is picked).
  Without a train it says why (no station within reach of an end, walks too long, no service
  in the period). Every row and bar segment has a hover with its figure and what it is made of.
  On the map, while a trip is drawn, a veil of the land colour at 60% covers the map under it;
  the start is an open ring and the end a filled disc, the main route is drawn over the veil
  in its lines' colours (each ride along its line's track between the two stations, curves
  included, T-208; straight between the stations it calls at only when the drawn network no
  longer matches the solve's; with a dark casing, the boarding and leaving stations bigger) with
  dashed straight walks, and the other routes thinner and
  faint under it. Hovering a ride gives the line, its two stations, stops, minutes on the train
  after the wait, crowding's effect and the route's share; a walk its distance and minutes; a
  station what the route does there.
- **Picking on the map** follows visible drawing order: demand discs above trains, then stations,
  lines, junctions and track. Within a layer the last drawn hit wins, including overlapping
  bubbles; hidden layers do not intercept clicks. Network hits allow 7 px. Clicking a selection's
  destination or station catchment keeps its parent selection. Clicking the line already selected
  selects the track under it. Selected editing handles remain draggable above the demand view.
  An empty click or Escape clears the selection.
- **Tools** in the Build tab: draw track (key 1), station (2), delete (3), the level with
  its price, single, double or quad track (T-255), platform length for new stations. The number keys work from
  any tab and open the Build tab (the level and platform settings are there); the key of the tool
  already on does nothing. Each button shows its key. **Delete** (T-096): hovering shows what a
  click removes (a stretch of track drawn red, a station or a flyover ringed, its name by the
  cursor); a stretch is one piece of a drawn route, about one per click (T-244, 6.1); blueprint is removed at once (undoable), and anything constructed is removed only after
  a yes in the centered confirmation dialog ("Remove 1.2 km of constructed track? Nothing is
  refunded." with Remove and Keep; Esc is Keep). The inspectors' Remove buttons ask the same way.
  Lines are removed from the line panel. **Esc** undoes one thing at a time: a question waiting is
  answered Keep, a drag goes back, a half-drawn route is dropped (the tool stays on), then the tool
  is left for Select (drawing, stations, delete, making a line), then the selection clears. With
  select, the squares of a selected blueprint
  stretch (its corners) and any blueprint track end, junction or station can be dragged (6.1);
  the track inspector adds/removes blueprint corners and sets their individual levels (T-071).
  The cursor turns to a move cursor over a node that can be dragged. A line is made from the Lines tab
  (New line, name it, then click stations) and its stops edited from the line panel. Ctrl+Z undoes,
  Ctrl+Y or Ctrl+Shift+Z redoes. There is no Select button: clicking the active tool again or
  leaving it with Esc returns to selecting. While drawing track, its placement ghost is solid
  with a dark casing: white in Line/Traffic views, or coloured by the selected Height/Max speed
  view using the preview's fitted geometry. An invalid ghost is red. Once placed, blueprint track
  is drawn solid and opaque (T-311, Anita: the dashes "look quite confusing as the lines curve";
  opaque because segments overlap at every joint and see-through blueprint showed a bead there)
  like two parallel rails (T-322, Anita: light blue, "slightly wider" than built lines because
  blueprints matter more while building; grey vanished in the Lines view): a light blue middle
  (#a9cdf3) as wide as a line, with a dark blue rail (#2f62b3) 16% of a line width (at least 1 px)
  along each edge, in every track view. The rails are drawn for every blueprint segment before
  any middle, the way built lines' white casing is, so curves and junctions join cleanly. In
  Height and Max speed the middle keeps the view's colour, mixed 45% towards white, between the
  same rails; a selected blueprint's middle is the selection ink mixed the same way. A line planned
  along blueprint track fills the middle, in its colour mixed 45% towards white, with the rails
  either side. Lines that do not run stay see-through at 55%. **The station tool's ghost** (T-311, Anita: "theres no
  indicator for whether its okay to put a station here") is always the platform's full box at its
  true length, at every zoom (not a dot below zoom 15), and is blue (`#2f6fe0` fill, `#1c4fb8`
  outline) where a click would place the station and red (`#e03c3c`, `#b91f1f`) where it would be
  refused; fixed colours, not the theme accent, with a thin white halo. A refused spot also shows
  the reason in the tag by the pointer. The answer comes from the clock worker (one request in
  flight, asked when the spot, platform length or network changes; the last answer stays up
  meanwhile). **Placement help** (T-312, Anita: "distance from last station placed", "a circle
  around station we're currently placing of like 1km and 2km"): with the station tool on and the
  pointer snapped to track, the tag says how far the spot is from the station placed last in this
  session, as the straight line ("1.2 km from W 46th St"; nothing before the first placement, and
  the refusal reason takes the tag's place where the spot is refused), and two thin rings at 1 km and 2 km of ground
  distance, each labelled at its top, follow the spot under the ghost (`stationRings` in
  map/tools.ts, an SVG layer in the tool markers, not the renderer; fixed grey ink at 35%). Build has the
  total and "Construct blueprints", including the new train cars that constructing starts,
  after spare cars. Its expandable cost breakdown shows track by priced level, single/double,
  land/water and ramps (charged at the dearer end), stations, junction switches, flyovers and
  flat crossings, with lengths/counts and prices from the track model. The fleet quote previews
  the constructed service, so starting several lines together includes their resulting delays
  without changing the game. Tool hints and a refused edit's reason show at the map's bottom left.
- **A new game** starts on day 1 at 07:00 in an empty New York with $6B. Opening the page
  continues the last game (section 11). The map does not come back to the exact view it was
  left in (T-310): the URL's view only says which city or zone, and the map starts at that city's
  registry centre and zoom (zoomed out over it). A page opened with no view starts over New York.
- **Money tab**: cash; yesterday and today (fares, running trains, new trains, construction,
  net); the fare curve (each ride, each km, and what a 10 km ride costs); cars owned (the
  car prices in its hover) and cars the lines need; the value of the track and stations built.
  The line inspector shows a running line's fares and running cost so far today (T-293).
- **Exact metropolitan focus and boundaries** (T-131, T-139): each registered pack's dissolved
  Census county footprint has a thin solid outline at regional zooms, fading between zoom 8.5
  and 10.5. The focused city retains a thin solid boundary at closer zooms, with a darker outside
  stroke and a lighter inside stroke. A subtle shade darkens the basemap outside its exact
  footprint, including excluded enclaves, leaving islands inside the city clear. These are
  metropolitan demand areas, not municipal limits. The layers sit beneath labels and game
  overlays, never intercept clicks, follow city focus and survive basemap style changes. When the
  focus moves to another zone the shade swaps to the new footprint in one step, never through an
  empty frame (T-310); the footprints are made ready at startup.
- **The goal number is always visible.** In a city: its mode split as one bar (e.g. driving 80%,
  train 16%, walking 4%); the player's job is to make the train share grow. Subway Builder does
  this well. It is the split of all modelled local trip legs (commutes plus everyday outings, school and university),
  shown at 60% opacity while a newer network is being solved. Its three colours are the mode
  colours below, whatever the theme. Zoomed far out: one bubble per city for inter-region trips, sized
  by trips, showing rail's share (T-186, under Inter-region trips below).
- **Demand views** (Anita's pick, 2026-10-09; T-084, notes/T-084.md). Line widths, station
  circles and bubbles are steady-state forecasts from the latest demand solve. Train fill and
  platform crowds are the boarding account (T-164, docs/track-and-service.md 6.6): still made from
  that forecast, but counted train by train as the animated trains run, so a train after a gap is
  fuller and a full train leaves riders on the platform. Line widths are selected by Track display;
  station circles have a checkbox at the top of the Stations tab (T-213), train fill and crowds
  are under "Riders on trains and platforms" in settings:
  - **Line width by riders**: each stretch between two stops is as wide as its riders over the
    whole day shown (T-266's Today, Weekdays or Weekends), both directions together, in a straight
    line (T-227, Anita: "the scaling should be linear. the most used line should be a fixed width
    like 3x what it is now and the least used lines should be like 0.5x normal width"): 0.5 of the
    normal width with no riders to 6.6 times (three times T-084's 2.2) for the network's busiest
    stretch-day (`busiestDay` in map/demandLayers.ts: the purpose shown, the larger of the
    weekday's and the weekend's). The whole day, not the hour on the clock (T-274, Anita: "lets
    show demand for the whole day rather than the current hour"), so a width agrees with the
    track tag's riders a day and the panels, and does not change as the periods pass; platform
    crowds, dot tints and train fill still show the hour. At zoom 15 the widest line is 44 px.
    The white casing is not part of the width: it is a fixed margin drawn under the coloured
    stroke (T-267). Lines sharing track are
    drawn over each other (T-190); stations keep their own width, so a wide line can be wider
    than the station on it (T-239). Only drawn in
    Line with traffic thickness mode (T-105).
  - **Station demand circles**: an explicit analytical setting, off by default, draws large
    translucent circles behind stations. Circle area grows with boardings plus alightings a day.
    The station itself stays as drawn (Stations on the map, below). Demand circles do not enlarge
    the station's click target (T-144).
    With "One circle per station complex" on (the default) each station complex has one circle for all its stations' boardings
    plus alightings, at the mean position of its constructed stations (T-146), the player's links
    included. The circles group exactly as the complexes do (checked on every complex of the
    real New York save, T-209). Two complexes can share a name, as New York's Times Sq-42 St and
    its 8 Av platforms 290 m away do; they are two circles until the player links them.
  - **How full each train is**: a train's core fills from its back like a gauge, in the dark shade
    of its line colour over a slight tint of it: filled is 48% of the line RGB, empty mixes only
    18% white into the line colour (T-103). The gauge spans the whole consist across car gaps;
    half full means every seat taken, full means packed to crush. The numbers are the train
    inspector's: this train's own riders on the stretch its drawn position is on, from the
    boarding account, empty once it reaches its last stop. A train packed to crush is drawn full
    with a red rim 3.9 px wide (T-207, T-218; the white rim is 1.7 px), and its hover tag says
    "packed full": the account never loads a train
    past crush (whoever does not fit waits), so without the rim a full train looked the same as
    one at 85%. A train the account has no riders for yet (just after an edit changed its line's
    stops, until demand's next result) is drawn without a gauge and its tag says "Calculating
    demand": no forecast average stands in for it (T-230, Anita).
  - **Platform crowds** (T-164; drawn as the platform filling up since T-219, Anita: "people
    filling up the platform, sort of like how darkened space represents people filling up
    trains"): from zoom 15, where platforms are drawn (T-204), the white of each platform box fills
    with a dark warm grey (#77726d, kept clear of the rim, 1.5 px or as far as the blue platform
    edge where it is drawn, T-236, so the outline stays crisp) as
    riders wait, like a train's gauge. Each direction of travel has its half
    of the box along its length (trains running along the platform's drawn axis on one side,
    against it on the other); a platform whose trains all use one side (a terminal) fills as one.
    A side fills from the middle of the platform outwards, as far as its riders' share of the room
    to stand there (`platformRoom`, docs/track-and-service.md 6.6: 1,250 a side on a 150 m
    platform), full to both ends at that room or more. A side is red (the over-full red) instead of
    grey while the last train on it left anyone behind. The fill lies inside the box, so it never
    covers the lines beside it. A train standing in
    the box covers its middle (at zoom 15 a train is 9 px of the box's 12.6); from 16 the sides
    show beside it. The crowd grows between trains at the platform's arrival rate and drops when
    one leaves; it is worked out on the main thread at most once a game second, only while drawn.

  **Local trip bubbles**, the main demand view (T-097/T-066, Anita's Subway Builder style view, notes/
  T-097.md): the always-visible selector at the map's upper left offers None, Homes or Destinations
  (T-132). Destinations includes work, shopping/errands, social/leisure, school and university activity ends.
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

  **The bubbles catch up over a game day** (T-156, Anita 2026-10-12; `map/catchUp.ts`). Demand
  has no opening ramp: riders, loads, fares and the top bar's split follow a new network as soon
  as its result lands. The bubbles move to it over about one game day, as if each person makes
  their trips once a day and switches on their next one: each cell's legs by mode are the old
  result's times (1 - f) plus the new one's times f, so colour and size both move, where f is the
  share of a day's trips that have left since the new result landed. Trips leave by the
  commute's period profile (4.2: both legs together 36/15/32/10/7% over morning peak, midday,
  evening peak, evening and night, even within each period), so the bubbles change fastest in the
  peaks and hardly at night, and are caught up 24 game hours after the result. f follows the game
  clock: paused, nothing moves; at the fastest speed a day is 90 s. A result landing during a
  catch-up starts a new one from what the bubbles show then; where a solve's estimates are shown
  (T-232: startup, a game load, a new city), its refined rounds carry on the one its first
  estimate started. The first result, a change of purpose filter or city, and
  a result landing while the clock is held (a game loading) show at once. The bubbles are rebuilt
  in their worker when f has moved 0.01 and at most four times a second, only while catching up.
  Their hover tags read what the bubbles show (kept lean: no station list, T-323; a clicked bubble's
  panel has the closest stations, below). The inspectors, a bubble selection and its far
  end, a station's catchment and the top bar are the current forecast at once. The city bubbles
  zoomed far out (below) catch up the same way; a selection's inter-region trip bubbles do not.

  There is no corner summary box: the city mode split is in the top bar, and local splits are
  shown on hover and in the inspector (T-101, Anita).

  **The focused zone and nearby zones** (T-276, [zones.md](zones.md) 6; `map/zoneFocus.ts`,
  `game/zones.ts`). After a pan the focused zone (the top bar's name, the shaded outside, the
  bubbles that can be selected) is the registered zone, city or not, whose boundary
  (`<id>.boundary.geojson`) holds the view's centre or lies within 40 km of it; before the
  boundaries have loaded the nearest registered centre within 120 km. Farther than that is dead
  space (T-318): the top bar has no name, the split shows "–", the demand buttons say "Nothing
  modelled here", nothing loads and no card or status says "Loading data"; the focus stays on the
  last zone (`placeAt`, `viewedZone`). Search picks its zone by the nearest boundary. A
  bubble, station or trip selection keeps the focus where it is. Beside the focused zone's bubbles,
  up to three more zones whose boundaries touch the visible map, nearest the centre first, show
  their own bubbles with the same look, end, purpose filter and day, drawn under the focused
  zone's, so a zone line is not a cliff; only while Homes or Destinations is on and above the
  overview zoom (8.5). Each has its own bubble worker and always shows the current forecast (no
  catch-up). Their hover tags read as the focused zone's. A click on one focuses its zone, which
  takes over that zone's worker and bubbles, and selects the bubble in the same click; the old
  focused zone's worker and bubbles become a nearby zone's, so panning across a zone line rebuilds
  nothing. Selections, their far end, the catch-up and the inspector are the focused zone's only.
  The demand pool opens a nearby zone's pack like a viewed city's, but only after the startup load
  and outside a loaded game's clock hold, with no loading bar; a solve never waits for it, and
  once it is open the network is solved again so the shown result covers it. A pack once opened
  stays open (as a viewed city's does): New York's seam with Philadelphia added 90 MB of WASM
  memory to each of the five solving workers, Hartford's neighbours New Haven and Springfield 57
  MB each; bubbles came 2.3-4.5 s after the pan (2026-10-13).

  Hovering a bubble shows local trip legs there, both directions included, and their mode split.
  **A click on a bubble selects it**; a click
  with Shift adds it; **holding the mouse still for 0.4 s and dragging** draws a box and selects
  every bubble whose centre is in it. Visible bubbles take priority over stations and trains
  underneath, for clicks, hover and starting a box selection. A selection hides the other bubbles and shows the selected ones outlined and
  their destinations (or homes) as bubbles sized among themselves (the biggest about its
  hexagon's area) and coloured by their own mix (`flows`, T-098). The inspector shows the
  selection's local trip legs and split, and the top eight far-end places, each named by the station
  nearest it within 3 km, else by distance and direction. A click on a far-end bubble opens the
  trip explorer for the pair (T-208, "Trip explorer" above). Escape or an empty click clears it.
  Purpose inspection shows daily counts/shares for Work, School and university (one row, T-192),
  Shopping and errands, Social and leisure, Airport and Medical, in every city
  (T-142; Springfield has no airport, so its Airport row is zero). Names carry no source tags.
  Scope notes identify matched
  public schools and degree-granting colleges across the metro area, the metro area's commercial
  airports (connecting passengers and staff left out; for an arriving passenger the trip from the
  airport is the return) and OSM's hospitals, clinics, doctors and dentists, and estimated homes;
  private schools are not modelled.
  A purpose-share pie and labelled legend follow All modes / Train only; each slice and key entry
  gives its count on hover ("Work: 12,400 trip legs a day, 48%"). Work is blue, school and
  university blue-green, social/leisure purple, shopping/errands red, airport olive and medical
  pink (CIEDE2000 22 apart at the closest pair, medical and social), with matching bar colours
  (T-143, T-142, T-192).
  A zero-trip scope shows an empty neutral disc. All modes / Train only changes this breakdown's
  scope. **Timing** (T-193): each purpose has one compact chart, a bar per period (6-10, 10-16,
  16-20, 20-24, 0-6) with departures toward the destination drawn up from a shared baseline and
  departures home drawn down from it, both as departures an hour on one scale, in the purpose's
  colour. Each bar's hover gives the count, the hourly rate and its share of that direction's day
  ("To destination, 6–10: 12,400 trip legs (3,100 an hour), 45% of the day's"). Weekend days are
  not part of it (T-170). Purpose sums match the bubble
  total for the chosen mode scope; unmodelled facility purposes are explicitly identified.

  **A selected station's commuters**: the station inspector lists commuters a day who live within
  its reach and ride from it ("Live nearby") and who work within its reach ("Work nearby"); the
  larger is picked first, a click picks the other. On the map, over the network, the chosen end's
  cells are opaque discs on top, the busiest a cell's area, and the places at the far end (up to
  20 gravity zones) filled discs at 55% opacity below them, the busiest half a zone wide; no
  minimum size; hovering either shows its riders, and a click on a far-end disc opens the trip
  explorer for the pair (T-208). The inspector lists the top far-end places. A
  selected station hides the commuter bubbles.

  **Station complexes** (T-146, `game/complexes.ts`): stations whose platforms nearly touch are
  one complex. A platform is the straight segment of its length through the station along the
  track heading; two stations are linked when those segments come within 50 m, or their centres
  within 100 m (the demand model's own merge distance, 4.4), and a complex is everything linked
  directly or through other stations. Levels do not matter: a line crossing over or under
  another forms a complex with it. Complexes are worked out from the stations and the player's
  links whenever either changes; only the links are saved. A station on its own is a complex of
  one, and its inspector looks as before. Demand lets riders change between any two stations of a
  complex on foot (T-161, 4.4).

  **Linking stations** (T-197, `game/links.ts`): the station inspector (station and complex view)
  ends with "Link to another station". After a click on it the inspector says to click another
  station on the map, the pointer is a crosshair (a hand over a station), and the next station
  clicked is linked with the selected one, which stays selected; a click on anything else, Esc,
  "Stop linking", another selection or a tool stops it. Refused, with a notice and still picking:
  the same station, one already in this complex, or one more than 1 km away (centre to centre;
  real interchanges joined by passage are a few hundred metres, and 1 km keeps a complex a place).
  Linked stations are one complex everywhere a complex is used: the inspector, the demand circles,
  the members drawn selected, the name labels and demand. A "Linked by you" list above the
  commuters gives each link touching the station (in complex view, any member) with its two
  stations, their line chips and Unlink. Only the player's own links can be undone: stations the
  distance rule groups cannot be split (their platforms nearly touch, and stations within 100 m
  are one station for demand anyway). Linking is not undoable with Ctrl+Z (a setting, like fares);
  removing a station drops its links. Blueprint stations can be linked; demand counts a link once
  both are constructed. The complex map draws each link as a dashed line between the two stations,
  its hover giving the distance, with a note under the map. **A click on a station selects its
  whole complex** (T-249, Anita: "in general we should just pick the complex"; a zoomed-out dot
  standing for its own group's station is fine): the inspector opens in complex view, whatever
  else was selected before (there is no Station / Complex switch). Complex view shows,
  under the name and "Complex of 16 stations", first the complex's lines (T-246, T-250, Anita:
  "one row per line, not one row per line-direction"): one row per line name, the save's
  branches and patterns of it summed ("subway 6" to Parkchester and to Pelham Bay Park is one
  row), with its chip, its name, the riders waiting for it now (red when one of its platforms is
  full or its last train left riders behind) and its boardings there today (T-293), busiest by
  the forecast first, so the order holds through the day; the
  row's hover gives the line and station names and where its trains go ("subway 6 at Grand
  Central-42 St / To Chambers St/Brooklyn Bridge-City Hall, Parkchester, Pelham Bay Park"), each
  number its own hover. A click on a row opens that line's busiest station there on its own in
  station view (`openAlone`, `singleStation` in game/complexView.ts: its inspector, outline and
  catchment are the station's); its subtitle then ends with "Complex of 16 stations" as a link
  back, and any new selection, a click on the same station included, shows the complex again.
  Then the complex map (below) and the complex's boardings today with their purpose split.
  **Closest stations** (T-312, T-323, `ClosestStations` in ui/inspector/ClosestStations.tsx,
  `game/walks.ts`): below the station's lines, the three other stations with the shortest walk,
  built or blueprint (blueprint ones muted and marked), each with the chips of the lines stopping
  there (a chip per line name, six at most, then "+n") and its walking minutes ("7 min", hover: the
  metres).
  A click on a row selects that station (and moves the map to it if off screen). The station's
  own complex and stations of its own name are left out, and a name is listed once (the real
  save has a station per platform group). The walk is the demand model's routed walk (docs/demand.md
  4.3), so a station across the Hudson is never a neighbour; one that water cuts off shows "no
  walk", but only when fewer than three can be walked to within 10 km. Minutes are the routed
  metres x the model's 1.3 detour at a plain 4.8 km/h (the 15 km/h fast walk to stations stands
  in for bikes and buses and is not used). The panel of a selected demand bubble (one area; under the mode split) lists the same from the
  bubble's centre, its cells weighted by their trip legs; a selection of several areas has none.
  Commuters and inter-region trips cover the whole complex, each of the demand model's stations counted
  once, and the map draws the complex's catchment. "One circle per station complex" in the
  Stations tab (under Station demand circles, T-213; on by default since T-192) groups only the
  demand circles. In complex view every member is drawn selected (an outline and a glow, never larger),
  as the selected station is in station view (T-163, T-189). On the map, a complex's stations that share a name have one name label, at
  their mean position and oriented like the member with the most lines, ranked by the lines
  stopping at any of them; differently named members keep their own labels (T-169,
  `map/stationLabels.ts`).

  **The complex map** (T-147, `ui/inspector/ComplexMap.tsx`): complex view opens with a small
  map of the complex, the inspector's width and 120-240 px tall. North is up, and positions and
  headings are true at one scale (at most 1.2 px a metre; a round scale bar in the corner). Each
  platform is drawn with the lines stopping there through it in their colours, each line as wide
  as the riders on its trains at that platform a day: in each direction the busier of the
  arriving and leaving loads, every period summed, 1.5 to 6 px by square root against the
  complex's busiest. Platforms that lie on top of each other (parallel within 15 degrees, centre
  lines within 12 m; the real New York save gives each service its own station, so Grand
  Central's Metro-North lines are eight platforms in one place) are drawn as one bundle, their
  lines side by side in the order the platforms lie across. Bundles have a thin dark outline,
  dashed for blueprint; the selected station's lines a heavier one. Hovering a line gives the
  station, the line, its boardings and the riders on its trains there; a click opens that
  station on its own, as a click on a line's row above the map does. When the complex has more than one name, each name has one
  label (the part before a slash) beside its platforms, above, below, right or left, wherever it
  is clear of platforms, other labels and the scale bar, else none. A note under the map says
  what the widths are.

  **The complex now** (T-246, Anita: "show crowding, how many people are on each platform. and
  whether there are trains pulling in or in the station"; `game/complexLive.ts`). After each
  station's lines its bundle has a 5 px platform strip, white between two lines in the platform
  blue (as on the map, T-236), that fills like the map's platform boxes: each direction's half
  from the middle out by its riders' share of the room there (`platformRoom`), the whole strip
  where trains use one side, grey or red as on the map. Its hover gives the riders waiting (with
  each line name's, directions summed, when more than one line stops there; T-250), the room and
  whether the last train left riders behind. Trains are drawn on their line's band in a darker shade (the map's
  train colour): one standing at the platform along the platform, one pulling in (stopping there
  within the next 60 game seconds) paler, coming in from the end it arrives by, further along
  the sooner it stops; hover gives the line, where it goes and the seconds. Read from the
  renderer's hour of the boarding account and its trip buffer, the numbers the map draws; worked
  out again every 10 game seconds (checked every 150 ms of real time, re-rendered only when
  something shown changed). No note explains them (panels show what can change, CLAUDE.md).

  **Transfers on the complex map** (T-176). The map asks the demand workers for the complex's
  transfers when it shows and again at each published round (`complexTransfers(members)`, docs/
  demand.md 4.4: riders a day changing from one line at one member to another line at another or
  the same member, every period and purpose; a few kB), keeping the last answer drawn while the
  next is on its way. Both directions of a line pair are added together. The changes between lines
  in different bundles are drawn as grey curves (ink at 55%, over a thin halo of the land colour)
  over the platforms, across the gap from where the two lines' bands come nearest, or from the
  middle of one band to the middle of the other when they cross or come within 16 px (loops at a
  crossing made a knot): the busiest 8, none under 2% of all the changes inside the complex, 1.5
  to 8 px wide by square root against the busiest drawn. Each curve bows gently to one side, so
  two changes along one line of sight stay apart; two middles at one spot get a small loop. Hovering a curve gives the riders changing a day and each direction
  ("35k riders a day change between subway 1 at 34 St-Penn Station and njt NEC at Penn Station /
  16k from subway 1 to njt NEC, 20k the other way"). Changes within one bundle (across a
  platform, or between services sharing it) are not drawn; each line's hover adds how many change
  to or from other lines there a day and its three biggest partners. A walk to a station outside
  the complex is left out. A second note under the map says what the curves are. Why line to line
  rather than bundle to bundle: notes/T-147.md.

  **Inter-region trips** (T-118; sim/src/demand/intercity.rs, SPEC 5.2) have an "Inter-region trips" section
  in the inspector of a bubble selection and of a station, and their bubbles are always drawn
  (the settings pane's check to hide them is gone, Anita: "we always want that on", T-222), except
  while the map shows one local purpose. A bubble selection lists trips that leave from
  its areas (homes first) and trips that arrive there, each by place with all modes and by train
  columns; the own region's row reads "Other New York region places" because the market has no trip
  under 40 km. On the map the selection's own end is drawn at the far end: for Homes, where the
  trips go; for Destinations, where they come from. A station lists rail riders who start and end there,
  by place, and draws where the ones starting there go (T-120). At every zoom, a selection's or
  station's inter-region trip bubbles are limited to markets outside the focused metropolitan area
  (the city overview below draws the focused city too, T-186); same-city inter-region trip counts
  remain in the inspector (T-139). Above zoom 9.5, every positive far
  6 km zone in those other markets is shown at its actual position. Inter-region trips have their own relative scale: the
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
  from it. For Homes the map shows destinations; for Destinations it shows origins.
  The selection does not end when the map pans to another city (focus stays while a selection or
  a station is selected).

  **Zoomed far out, one bubble per city** (T-186, Anita 2026-10-12; `map/intercityView.ts`). At
  zoom 8.5 and below (`OVERVIEW_ZOOM`: where the metropolitan outlines are fully drawn, a New York
  region about half a 1,400 px map wide with Philadelphia and Hartford on screen), with Homes or
  Destinations on and no bubble or station selection, each of the nine cities, the focused one
  included, is one bubble at its registered centre in place of the focused city's own bubbles:
  area is its inter-region trips a day to and from every other place (both directions), colour its train share as above, labelled "Philadelphia · 187k trips/day". The busiest
  is 28 px in radius before the size slider, larger than a selection's 19 px city circles since
  nothing local is drawn beside it. Homes and Destinations look the same here: the market's
  corridors have no direction. A click on a city's bubble selects the city (a `city` selection):
  it is drawn outlined at 28 px, with one bubble per other city at that city's centre for the
  trips between the two, on the same scale, and the inspector lists its trips by place, all modes
  and by train, with a "Go to Philadelphia" button that moves the camera to the city's centre view
  (the search's) and clears the selection (T-220, the City tab's City centre button). A click on
  another city's bubble selects that one; Escape or an empty click clears it. A city selection keeps its bubbles at any zoom and hides the focused city's own.
  Hover gives "187k inter-region trips a day between Philadelphia and other places" (or "between
  Philadelphia and New York") and the train count. Trenton (Mercer County) is a market place but
  not a registered city: it has no bubble, but its trips count in its neighbours' totals and are
  listed in the inspector. The overview follows the purpose filter like the other inter-region trip
  bubbles: with a local purpose chosen, the focused city's own bubbles stay at every zoom.

  Bubbles and selections cover the focused city's modelled local trips; station catchments still
  count its commuters explicitly. Changing
  city clears old cell-indexed selections and cached layers. Network widths, station totals and
  train fill include all registered cities and inter-region trips.
  Data is asked for, not pushed (4.4): each view asks again when the demand result changes and
  keeps showing its last answer meanwhile. Bubbles are rebuilt when a result is shown: after an
  edit only the solve's final round is (T-232, "Updating demand" under Loading above), so once; at startup, a
  game load or a city just opened, each estimate too.
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
  oranges, purples, as on real metro maps), not pastel. Thirteen defaults (`LINE_PALETTE` in
  `game/palette.ts`): red, blue, green, orange, purple, yellow, brown, teal, then gray, yellow-green,
  pink, magenta and light blue (T-223, Anita), every pair at least 19 apart in CIEDE2000; chips on
  the light ones (yellow, yellow-green, pink, light blue) take dark letters. New lines take the
  first default no line uses; the player can set any line's colour in its inspector using palette
  swatches or a custom colour.
- **Zen Maru Gothic** (Anita's pick over M PLUS Rounded 1c) everywhere, including station names
  and the map's DOM controls. The map container explicitly inherits the project font rather
  than MapLibre's default Helvetica (T-101). Form controls inherit it too. The font on
  the basemap's labels is Regular only, faint grey, from glyph tiles we generate and host beside
  the app (`app/public/fonts/`, `pipeline/glyphs.py`), with Noto Sans merged in for scripts Zen
  Maru lacks. Kana and kanji on the basemap are drawn by the browser in the system font for now
  (T-057). notes/T-056.md. Few font sizes and weights (the mock's 3 sizes and 2 weights is the
  ceiling).
- Instant UI state changes, no CSS transitions. Trains moving is the only motion. Camera moves
  the game makes (search, City centre, Whole region) jump with no flight or ease (T-182); the
  map's own gestures keep MapLibre's easing.
- **Space bar** pauses, and resumes at the speed that was running before the pause, even with a
  button focused.
- **Keys and focus** (T-183, `ui/keys.ts`): one rule for the game's keys (1-4, Esc, Space,
  Ctrl+Z/Y, the line tool's Enter and Backspace) and search's `/`, Ctrl+K and Ctrl+F. A text field
  (search, a slot or line name) keeps every key (search's own box excepted for Ctrl+K and Ctrl+F,
  which select its text). Another form control (a select such as the
  purpose filter or the line sort, a slider, a checkbox, a colour) keeps only its own keys: the
  arrows, Home, End, Page Up and Down, Enter and Space, and in a select a letter (it jumps to that
  option); the rest reach the game, so `/`, Ctrl+K, 1-3 and Esc work with the purpose filter
  focused. A select picked with the mouse gives its focus up at once, so the next Space pauses;
  one reached with Tab keeps focus and its arrow keys.
- **Line widths** (T-191, T-205): a line is 0.85 px wide at zoom 9, 3.9 at 12 and 6.6 at 15 (linear between, held outside),
  bare track 0.4, 1.8 and 3, with a white casing 0.6-2.4 px wider; a selected line or track 0.5-3
  px wider still. A stroke thinner than a device pixel is drawn a pixel wide and that much fainter,
  so far-out lines stay whole. Traffic thickness multiplies the line width (0.5 to 6.6 times,
  T-227). Picking reaches 7 px from the centre line, or half the drawn width where a line is drawn
  wider than that (T-227), so the track tag and clicks hit anywhere on a wide line.
- **Zoomed far out** (T-205, Anita: "hide trains and stations when zoomed out";
  `STATIONS_ZOOM` in `map/stationGeometry.ts`): below zoom 11, where the screen holds more than
  a city such as Manhattan with its neighbours (about 30 km), only the lines are drawn. Stations
  and trains are neither drawn nor picked, so a click or hover there finds the line. A selected
  station (or complex) is still drawn and clickable. Demand circles, bubbles and capacity tags are
  unchanged.
- **Stations on the map** (T-189, T-204, `map/stationGeometry.ts`): from zoom 11 to 15 every
  station is its own dot at its platform's middle (white, dark rim a quarter thinner than a box's,
  1.5 px at zoom 13; T-313, Anita: "make station small dots bigger, maybe 65% bigger radius", then
  T-322: "20% smaller", "the black rim a little thinner": ordinary 4.3/5/7.4 px radius at zoom
  11/12/15, a station served by more than one line 5.6/6.6/9.5 px, 1.32 times T-189's, never
  narrower than the lines through it). A complex's dots are one shape the way its boxes are (T-313, Anita: "stop
  growing dots to fit all the platform centers within the complex dot ... their outlines merge in
  the same way that station boxes do"): every dark disc is drawn first, then every white inside its
  rim, so dots that overlap have one outline round them all. Dots that do not touch are joined by a
  connector: a straight band from one dot's centre to the other's with the dots' own rim along both
  sides and white 30% of an ordinary dot's white radius either side of its middle (about 5.6 px
  across at zoom 13; T-322 made it thinner), its dark drawn with the dots' dark and its white with their white, under it,
  so dots and connectors are one white shape with one outline (Anita: "it should look like its
  made of the same material"). Connectors join the stations the complex rule's distances group
  (game/complexes.ts without the player's links, `dotGroups` in game/complexView.ts), by the
  shortest set of joins linking all of a group's dots, and each of the player's links between
  two groups, from one linked station's dot to the other's (T-225); a join between dots that
  overlap is hidden inside them. (T-217 drew a group as one dot covering its platforms' middles,
  about 240 px across at zoom 14.2 for 5 Av/42 St-Bryant Pk linked with Grand Central.) Each dot is
  its own station: a click picks the dot whose centre is nearest among those under the pointer,
  and selects its complex as everywhere (T-249); the delete tool and the line tool take that
  station, and its tag is that station's, with its own lines (T-305). In complex view every dot
  and connector of the complex is drawn selected, with one outline and glow round them all, and
  the demand bubbles are cut away round that shape. A connector is faded when either of its
  stations is a blueprint. With platform crowds on, a dot's
  white is tinted by its platforms' crowd (T-228, Anita: "highlight the circle slightly"): towards
  the crowd grey by its fullest platform side's share of the room there (at most 45% of the way,
  at full), or a pale red (40% of the way to the over-full red) while a train at any of its
  platforms left riders behind; a connector stays white. From zoom
  15 (Anita: "turn into them sooner, around ~4x more zoomed in" than T-189's 13; a 150-200 m
  platform is then 80-110 px long) a station is its platform: a band along its track as long as
  the platform, following the track's own centre line (T-254, Anita: "ideally we actually fit the
  track"; one circular arc did not fit S-curves such as Wall St): the track is sampled at 13 evenly spaced points along the platform (`platformShape`'s
  `pts`, `PLATFORM_POINTS` in map/stationGeometry.ts, straight on past the end of an edge that
  stops short), and the band is the box width either side of that line, with flat ends, centred
  on the station node, or at a dead end running from the node along the
  track as trains stop at its middle there. Everything that draws or tests a platform takes that
  one line, in its frame at the platform's middle: the box, its blue edges, the crowd fill and the
  track stencil (STATION_VS/FS, `polyFrame` in SHAPE_GLSL: arc length along the line and signed
  distance across it), picking and the placement ghost (`stationShape`, `stationCoords`,
  `stationPath`), the selection's outline and glow (SEL_FS) and the cut-out in the demand
  bubbles (demandOverlay.ts). The box is drawn at one scale (T-204, Anita: the boxes
  "change proportions as we zoom in and out ... they end up looking stubby"): its width is fixed
  on the ground like its length, a quarter narrower than a plain line and its rim are at zoom 15
  (T-239, Anita: "a bit too fat still"): 9 px there, about 16 m at New York's latitude, wider than
  the true 12 m, so both double with each zoom level and its shape never changes. Every station
  has this width in every mode: in Line with traffic thickness a line can be wider than the
  stations on it, and dots do not grow for it either. It is never shorter than it is wide. Station
  names sit clear of a plain box at any zoom. Blueprint stations are faded (blueprint track is drawn as blue rails instead, opaque, T-311, T-322).
  Platforms drawn over each other are one white shape with one dark outline round the whole of
  it (T-217, Anita: "all the white rectangles overlapping seamlessly ... and then one continuous
  black outline surrounding the whole white shape"): every station's dark shape is drawn first,
  then every station's white inside its rim, so a rim lying inside another platform is painted
  over (a faded blueprint draws only its rim, so its see-through white is unchanged). Inside the
  white (T-236, Anita: "show the actual platforms as like little blue lines. and maybe the track
  on top of the station"), in
  this order over the crowd fill: the track and line strokes run through the box, drawn again
  where a stencil marks the white, as wide as on the map but no wider than 30% of the box's white
  (2 px at zoom 15, 5 at 16, the full 6.6 px line from 17) and without their casing, so the crowd
  shows either side and the box's outline stays whole; then each platform's two long edges as
  thin lines in the platform blue (`PLATFORM_BLUE` #8085f0 in game/palette.ts, a periwinkle 15
  CIEDE2000 from the line blue and the light blue, 38 from white), 14% of the white's half width
  thick (at least 1 px), as far in from the rim as 15% of it, coming in between zoom 15 and 15.5
  as the box grows wide enough to hold them. Platforms drawn over each other each show their own
  edges, so a complex reads as its platforms with the tracks between them. The player's
  links (T-197) add a bridge: a straight white band 3 px wide at zoom 15 (fixed on the ground like
  the boxes) with the 2 px rim along both sides, between the nearest points of the two platforms'
  centre lines, each at least 6 px (at zoom 15) inside its platform's ends, so it leaves each box
  on the side facing the other and the two outlines run on round it. Platforms that overlap get
  one too, inside both. Zoomed out a link is the connector between two dots (above). The
  complex rule's members that do not touch (up to 50 m apart) stay separate shapes. A selected
  station keeps its size and gets a 2 px dark outline and a soft dark glow 6 px beyond, round the
  union of the selected shapes (each pixel's distance is the nearest selected platform's or
  bridge's), drawn under them; selected stations are drawn after every other station, and the
  demand bubbles are cut away over them, their bridges and their outline, fading out over the
  glow, so they show above their own catchment and a train standing in one stays visible. In
  complex view every member of the complex and every bridge between two members is drawn
  selected. Clicks, the delete tool, the line tool and the placement ghost (the platform as it will be drawn, curved with the snapped track) all use
  the drawn shape, platform by platform (bridges are not picked) or dot; a click on a selected
  station picks it even over bubbles.
- **Trains** are not drawn below zoom 11 (T-205), pills from 11 and separate cars from zoom 15, in a darker shade of the
  line colour with a shared width at the transition (T-107) and a white rim blended into the
  core over a pixel, every edge antialiased in the shader. A train is 3.75 px wide at zoom 9, 6 at
  12 and 9 at 15 (linear between, held outside; its true 3 m only from about zoom 19), about 1.4
  times a plain line from zoom 12 (T-199); a pill is never shorter than 11, 17.6
  and 26.4 px at those zooms. A pill bends with its track along the
  circle through its two ends and its middle (T-189), so on a curve it follows the line rather
  than cutting the corner; each close-up car is short enough to stay straight and is turned along
  the chord between its two ends rather than the track sample under its centre (notes/T-047.md).
  With riders known the core is a gauge of how full the train is (demand views, above).
- **Lines sharing track are drawn over each other** (T-190, Anita: interlining is a hard problem,
  not represented for now): the line drawn last (highest id) is on top and wins a click there.
- Player-facing text follows the copy rules in Anita's global CLAUDE.md (no dash-joined phrases,
  sentence case, no softeners, no hype).
