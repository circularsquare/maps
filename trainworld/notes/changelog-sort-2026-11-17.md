# Changelog sort, 2026-11-17

Reordered the file by area (game order), features before fixes, biggest first; hidden items last in their old order. Count before 127 (49 hidden), after 133 (49 hidden): 6 new group items, nothing merged, hidden or reworded. Run by logsort through changelog.py's own read, update_item and write_file under its lock.

## Groups made (new items, ids c128-c133)

- added real world network features: (track, 4 children: c073, c076, c117, c118)
- changes for trams: (track, 3 children: c103, c114, c126)
- added scheduling tools: (scheduling, 3 children: c094, c095, c123)
- fixed map drawing problems: (schematic fix, 3 children: c014, c065, c098)
- fixed zones and places in some countries: (schematic fix, 5 children: c040, c032, c044, c080, c081)
- some things run faster: (performance, 3 children: c055, c091, c102)

Existing groups kept as they were: track bug fixes, train bug fixes, money bug fixes, small games use about 350 MB less memory.

## New order of shown items


### track

- added optional per-track segment speed limit
- added rebuild buttons to the track panel for a built piece's level, single or double track, and track type
- added real world network features:
    - can now snap to unused track in the real world network's All setting
    - added a real world network snap for tram track that follows real tram and light rail, and redrew the snap buttons as one set of icons, with snap to streets a straight street with a dashed center line
    - added real world stations to the real world network map, hidden by default
    - a station placed on a real world station takes its name by default
- changes for trams:
    - fixed single tram track continuing a double track end with snap to streets on, which joined both tracks instead of the one under the pointer
    - trams now pass each other at a tram stop on single track placed as a passing station, as trains do, so single tram lines can run more trams
    - tram stops can have a terminal and passing tracks, like rail stations
- ferries are faster, 45 km/h instead of 32
- one platform stations can go at the end of single track
- lines through two one-way tracks with a station on each stop at both, one each way, and riders board each way at its own station
- ferry ports can have a waiting room
- narrow rivers count as water across more of their width, so track over them is priced as a bridge for more of the crossing
- right click on the map now ends track drawing
- water border improved for track construction what is considered on water  [fix]
- fixed a bug where extending a line could break it  [fix]
- track bug fixes
    - fixed a sideways shift where a short single track joins one track of a double
    - fixed the delete preview showing single track two tracks wide
    - fixed a station turning one-way single track into two-way track
    - fixed lines running over flyover legs that were still blueprint; construct line now builds the way back too
    - high speed track drawn with clicked corners now keeps the speed you set
    - the network hover no longer shows over a bus route's path in edit route

### demand

- demand bubbles show intercity trips at all zoom levels
- the city view's trips by length graph can switch between mode and purpose
- waiting riders now show per direction in a station complex hover and its station list
- the intercity trip explorer's effective time for the train now has a hover that shows how it adds up
- a stop's hover in the line group view shows its boardings in the past 24 hours
- fixed long-distance trip bug where local lines were lowering intercity train usage, *intercity train usage increases significantly* in most cases  [fix]

### scheduling

- added scheduling tools:
    - added weekend scheduling! can run fewer trains on weekends and holidays with new schedule panel
    - added timetable in schedule panel
    - can set default headways for new lines, per train-type
- added an alert on a line where its trains pass the same track twice in the same direction
- bus boarding penalty is reduced, buses will be used slightly more often
- added a setting to keep buses to one-way streets
- train bug fixes
    - undoing or redoing a train count change no longer makes trains hop at a fast clock
    - trains no longer jump when a stop is added or taken off the end of a line
    - fixed a train line slowly losing trains an hour with every edit when it shares a full track with other lines

### fares

- rail line panels have a chart at the bottom with blue fare bars and red running cost bars for each of the last 30 days
- added per-line fare multiple, can set 0x-3x
- money bug fixes
    - the profit map now shows the other leg of single track and flyover splits
    - bus routes now split their money between regions by the route's own points
    - the money chart now runs on local time where you build first when the game has no start city

### schematic

- added hiding lines on the map from the lines list, and hiding track levels, buses and ferries in settings; a hidden level is drawn faint and can't be clicked, hovered or snapped to
- schematic draws disconnected distant regions separately
- a click on a station of the selected complex selects that station on its own; clicking it again goes back to the complex
- a station on single track with no passing loop is drawn with one smaller platform on the outside. tram stops are drawn 30% narrower
- fixed map drawing problems:
    - fixed lines and track disappearing when zoomed in on some computers
    - fixed long straight bars drawn across the map on the globe, in games with stations far apart
    - fixed double track looking narrower than the real gap between its tracks as you zoom in
- fixed zones and places in some countries:
    - fixed zone names, mostly in China and Japan
    - merged Arnavutkoy into the Istanbul zone, and moved Karamursel to Kocaeli and Altinova to Yalova
    - added the reclaimed land at Shenzhen Bay Port and Tide Park to Shenzhen, so it has homes and jobs
    - fixed Bucaramanga's zone sitting inside another one, and moved 29 Colombian zones onto DANE's 2025 population figures
    - fixed missing demand in central Lisbon and in Clydebank and Dumbarton by Glasgow

### community

- points now come from a forecast weekday and update as soon as you change your network
- preserve the last undamaged copy of a corrupted save
- sharing a very large network works  [fix]

### performance

- small games use about 350 MB less memory on desktop
    - the game uses about 170 MB less memory on desktop, most of it zone outlines now stored compactly
    - games need less memory, about 130 MB less on phones and 50 to 100 MB less on desktop: empty map layers no longer hold a screen-sized image, and the tutorial's rings cover only the button they point at
    - the game uses about 130 MB less memory on desktop once a city's trips to other cities are worked out
    - each solving demand worker holds about 60 MB less once the regional market is open, about 190 MB less in all on a desktop game's first line
- big games use less memory: demand runs on fewer cores for them, about 0.5 GB less on a New York sized network and 3 GB less on all of China, with loads up to a third slower
- some things run faster:
    - large networks load about twice as fast (thanks to peanutmow!)
    - deleting a station on a big network is faster
    - train count changes on a one-city game have rides settle ~10x faster
- added a Settings option to hold off recalculating trips until you unpause
- a saved game outside new york no longer downloads new york's city data first  [fix]

### other

- added Guatemala, Belize, El Salvador, Honduras, Nicaragua, Costa Rica, Panama and Venezuela
- selamat pagi! added indonesian
- added common questions with their answers above the controls list, opened with the ? in Settings or F1
- added more long hairstyles
- new game from the saves panel no longer sat in the old game for up to a minute while terrain would not load; it saves and goes by the start menu  [fix]

### bug

- a very long line group name no longer pushes its buttons out of the panel, so it can still be renamed  [fix]

Hidden items (49) follow in their old order, untouched.
