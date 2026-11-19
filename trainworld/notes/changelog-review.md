## 2026-11-18, before v0.4.2

Reviewed: the 44 shown items among c001-c053. Nine were already hidden before this pass and not re-judged (c049, c027, c021, c017, c014, c010, c009, c004, c003).

Hidden in this pass:
- c051: the clear button in search boxes shows whenever there is text in them (tiny polish)
- c037: fixed old copies of the clock piling up in the top bar (rare, page-translation oddity)
- c030: moved the join control to the top of the leaderboards (minor layout move)
- c028: fixed community previews showing "Could not load this network" when switching tabs quickly (rare race)
- c020: fixed the new game city pick showing 0 intercity trips before they are calculated (minor display)
- c011: fixed the start menu covering the out of memory message after a game failed to load (rare failure path)

Reworded: none.

## 2026-11-19, v0.4.3

Ids restarted at c001 after the v0.4.2 publish, so the lists above are a different numbering. Reviewed: all 29 shown items c001-c029. Items marked (edited) by Anita were kept without re-judging.

Hidden:
- c028: stuck trip calculation restarts once on load (failure path)
- c024: crash at startup when the browser drops graphics (rare)
- c023: settings memory reading too high with several workers (diagnostic)
- c014: renamed station showing old name at some zooms (minor)
- c012: wrong fare on a bent fare curve (bug in a feature new in this release)
- c010: unused flat junctions and crossings no longer show a mark (tiny)
- c007: second tab unable to go back to the main menu (rare)
- c004: track ending just short of a bare track end refused (minor)
- c003: Copy shown instead of Sandbox on a kept copy (minor)

Reworded:
- c022: "fixed long games using more memory every game day; past about 2 years, money history older than 4 months is kept by month, so a 50 year game uses 1 GB less memory" -> "fixed long games using more memory every game day, so a 50 year game uses 1 GB less memory"

Areas: all items have one.
Proposed highlights: c021, c011, c015, c020, c005, c013
