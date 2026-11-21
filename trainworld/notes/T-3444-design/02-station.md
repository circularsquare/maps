# T-3444 design, part 2: what a station is

Agent trackdesign, 2026-11-21. Vocabulary as in `01-model.md`. Her question (notes/T-3444.md): "we might need some
sort of actual definition of what a "station" is, for display purposes and such ... not sure which of these makes
sense." Her four candidates are A-D below.

## What a station is used for in the game

Everything that today reads "a station" or "a complex", so a definition can be judged against it:

- **The name** and its label on the map; the station panel; the stations tab; a click on the map.
- **A line's stop** in simple stops (part 4): "this line stops at X".
- **Price and upkeep**, station upgrades (waiting room, park and ride), the terminal rule.
- **Demand**: a demand station is a point riders walk to (stations within 100 m merge, `station_merge_m`), changes
  within a complex cost 2 perceived minutes plus the walk (demand-kernel.md 4.4).
- **Ridership circles** and the station's figures: one per complex today (T-146, T-1449).
- **Real data**: OSM's `railway=station` plus `stop_area` (a station with its platforms and stop positions), GTFS's
  `location_type=1` station holding `location_type=0` platforms, NeTEx's StopPlace with Quays, the MTA's 496 stations
  grouped into 445 complexes (research/realworld.md). All of them have **two levels**: a station holding platforms,
  and a looser grouping of stations for changes (OSM `stop_area_group`, MTA complex). None of them has tracks.

## Options

### A. A platform is a station

Each platform the player places is its own station.
- For: the simplest record; NIMBY Rails' smallest unit is close to this.
- Against: the default case is wrong. A double-track station has two side platforms, so it would be two stations with
  two names, and a line's stop would have to name one of them. She saw this ("that is weird with side platforms").
  Every quad station would be four.

### B. Platforms built together are one station

What one click of the station tool places (one to N platforms across a bundle) is one station; later platforms are
new stations unless the player adds them to one.
- For: matches today exactly (one click, one station, twins named alike); no geometry rule at all.
- Against: a platform added later beside an existing one (the fifth track's island) is a new station unless the tool
  knows to join it, so the tool needs a rule anyway; offset and stacked stations come out as two stations.

### C. Mostly parallel platforms are one station

Platforms whose footprints overlap side by side, at any level, are one station; NIMBY Rails' rule ("a set of one or
more platforms whose footprints overlap", research/games.md).
- For: islands, AABB, ABBA, Grand Central's platforms and stacked 125th St Lexington all come out as one station by
  themselves, whenever they were built.
- Against: a geometry rule decides membership, the kind of rule that caused the pairing bugs (class B, 13 fixes);
  NYC's longitudinally offset platforms (one station to riders, platforms not overlapping) come out as two.

### D. The complex is the station

Everything within 50 m, joined by a passage or inside a station building, is one station with one name.
- For: one concept instead of two; one circle, one name, as players asked (inbox 1794584594, "complex = one name").
- Against: loses the distinction players also asked for (transfer stations keeping their own names: inbox 1795130903,
  1795163330, the setting added for it); Times Sq-42 St and 42 St-Port Authority would be forced into one name; a
  passage would rename stations.

## Recommendation: B plus C, with an explicit join, and the complex kept above

> **A station is a named group of platforms: the platforms placed together, any platform placed so that it overlaps
> one of them side by side (at any level), and any platform the player joins to it. Stations a rider walks between
> form a complex, as today.**

- **Geometry decides only at placement, once.** When the station tool places a platform overlapping an existing
  station's platform side by side (within 30 m across, overlapping along by at least a third of the shorter one, any
  level), the new platform joins that station; the ghost says so ("Adds a platform to Times Sq"). After that,
  membership is stored and never re-derived, so moving track later cannot split or merge stations by surprise (the
  failure mode of class B).
- **The player can join and split.** Station panel: "Join to station" (pick another station within 300 m, the change
  walk limit) and "Make a separate station" on one platform. This is how NYC's offset platforms become one station,
  and how her split legs 150 m apart become one station (part 1, simple stops rule).
- **The complex stays** (derived, 50 m, passages, buildings), with its one circle and its two naming settings. Times
  Sq is one complex of four or five stations, as in the MTA's data.
- **Two levels, like every real data source**: station holds platforms (OSM stop_area, GTFS station, NeTEx
  StopPlace), complex groups stations (stop_area_group, MTA complex). That makes real-network import a direct copy
  later (part 9).

What a player sees, case by case:

| Case | Result |
|---|---|
| Station on double track | One station, two side platforms (as today) |
| Quad, station on all four | One station, four platforms (or two islands) |
| An island added later beside an existing station's tracks | Joins the station (overlap) |
| AA above BB, 125th St Lexington | One station (overlap at any level) |
| Lexington Av/59th St (IRT and BMT) | Two stations, one complex |
| Offset platforms on a local and an express pair | Two stations in one complex, one station after "Join to station" |
| Her split one-way legs 150 m apart | Two stations until joined; placed in one click on both legs, one station |
| Times Sq | Several stations, one complex, as the MTA counts it |

## Old saves

Each station node becomes a station with its platforms (two zero-width side platforms on double, one on a one-track
station, both tracks of a single-track station's loop). Twins named alike on the two pairs of quad (`twin_at`) become
one station of four platforms. Pairs that today's per-direction rule pairs (`way_station`) become one station. Nicety,
not a migration project (CLAUDE.md "Saves are hers").

## Her decision

**Station is B plus C** (placed together, or overlapping side by side when placed, or joined by the player), with the
complex kept above it for changes and the circle? Recommended: yes. The alternative closest to it is D, if she would
rather players never meet the word "complex".
