# Player requests sort, 2026-11-28

Sorted by a Sonnet agent from `board/requests.py sort-brief`: 98 reports (2026-11-24 21:21 UTC to 2026-11-28 03:33 UTC), 0 pastes. Result is in `notes/board-requests.json` (backup of the file before: scratchpad reqsort/board-requests.backup.json). Now 207 groups, 160 open; no report is left unsorted. TASKS.md and LINES.md untouched. Player text read as data only.

## Where the 98 went

- 41 reports into 23 existing groups (counts and summaries updated; group sizes now: g005 18, g084 14, g030 8, g009 12, g133 6).
- 57 reports into 36 new groups, g172 to g207. The biggest is g172, roads and water missing, 16 reports.

## Settled without her (state merged)

- g172 Roads and water missing from the map, 16 reports, reach many. Cause: tracker blocker lists (EasyPrivacy, NordVPN Threat Protection) block OpenFreeMap's tile address; fixed for most in 0.4.6.1 (T-4275, DECISIONS 11-25), five players wrote back that it works. Match T-4328 (own-address route, waits to see who is still blocked). One report (Map mode plain brown, satellite fine) is the case T-4328 already names. Two players found the cause themselves: the workaround is allowing trackers for anita.garden (or turning off the ad blocker).
- g180 Lines table remembers closed groups (done: T-4329, 11-25), g195 line panel shows another city's line (T-4373), g197 bus stop labels flicker (T-4378), g198 station circles missing on a phone (T-4375), g204 Ankara takes in Sincan (T-4562). All five are asks already filed; nothing to do.
- Left open although a closed task did it: g207 rename stations at a transfer. T-3759 (11-19) lets each station keep its name; the player may be on an old version. Open until someone checks the control is findable.

## Need her or the orchestrator

Bugs more than a few players could hit:
- g182 Long-distance rail gets few riders (2 reports, India and China): million-plus city pairs get 5k or fewer riders even at minimum fares, one says "after some update". Looks like a regression more than a taste question; worth a trip-count check on a Delhi-Mumbai and Kunming-Shenzhen build.
- g191 Crossing at another height fails (2 reports): the crossing button does nothing, and tracks one height apart cannot cross. Maybe T-4335's new track rules; try it.
- g175 Station buildings broken on a phone: no resize by dragging, long press removes one corner, a building cannot be demolished after its tracks go (the demolish part would trap a player).
- g205 Riders transfer between distant stations (French player: "did not do this before"; a closer station on the same line or a transfer tunnel is ignored). Route-choice regression, one report.
- g183 Headways from train count disagree between periods (French player, not all lines, "did not do this before"). Maybe T-4238; one report.

Design questions (her call, however many asked):
- g173 Faster buses (bus speed, highway cap 55 km/h against 110): changes how fast things go.
- g178 Station upkeep and parking costs in small towns (11-25 update raised them; asks to revert, remove tram parking lots, differ by track count): a cost change.
- g177 A downloadable app (2 reports, Chinese and Japanese): is an app planned.
- g048 driving side (now 4 reports): trains right and roads left, per-line running side.

## Other notes

- A few reports ask for several things; each went in its main ask's group: 1795588031-8fa2fe3b (real bus stops, rename, faster loading, import routes; g176), 1795689075-9706e925 lists (hide track, BRT mid-track, parking, SF6 trams, depots; g179), 1795644937-90c3cc31 (Otaru zone name, g128; its second ask, a station's yearly average boardings, is not filed anywhere), 1795696901-713ef342 (error screen g005; the Japanese text of "Blueprints" in Settings is not clear to that player, not filed), 1795739555-fba9976c (bus snap g030; planned-station build button and 30-minute timetable start are not filed).
- Unclear: 1795673563-316caf14 ("capacity is far too small", g110), 1795687939-22bb8e79 (greeting only, g154).
- Reach guesses are mine from the report text; none came with a version number except those quoting 0.4.6.
