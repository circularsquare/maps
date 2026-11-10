# Train flow where lines share track and at terminals (T-2974)

Background for notes/T-2974.md: how real railways and other games time lines through a shared trunk
and turn trains at a terminus. Searched 2026-11-09. Related: `passing-tracks.md` (platform choice
at stations, NIMBY and OpenTTD signals), `nimby-rails.md`, `performance-factorio-nimby.md`.

## Real railways

### Branches onto a trunk: one clock for all

- **Clock-face (takt) timetables.** A line repeats the same minutes past every hour. Transport
  associations set a base interval of 20 or 30 minutes and lay several lines over one another to
  give 10, 5, 7.5 or 15 minutes where they share track ([Wikipedia, Clock-face scheduling](https://en.wikipedia.org/wiki/Clock-face_scheduling)).
  So branches that join run headways from one family (the same, or a whole multiple), and are
  offset so their trains alternate on the trunk: two branches at 10 minutes, 5 minutes apart, give
  a train every 5 minutes on the trunk.
- **Munich S-Bahn**: every branch runs a 20 minute base interval, some branches 10 at the peak, and
  seven lines share the central tunnel (Stammstrecke) between them ([Wikipedia](https://en.wikipedia.org/wiki/Munich_S-Bahn),
  [muenchen.de](https://www.muenchen.de/verkehr/oeffentlicher-nahverkehr/strecken-fahrplaene-stoerungen-infos-zur-muenchner-s-bahn)).
  **Stuttgart**: clock-face on every branch, a 2.5 minute trunk ([Wikipedia](https://en.wikipedia.org/wiki/S-Bahn_Stuttgart)).
- **Elizabeth line**: 24 trains an hour in the core at the peak, 16 off-peak, fed by the Abbey Wood
  and Shenfield branches in the east (8 an hour each off-peak, 10 each on Saturdays from December 2022)
  and Heathrow and Reading in the west, where many trains turn at Paddington
  ([London Reconnections, May 2023 timetable](https://londonreconnections.com/crossrail-the-may-2023-final-timetable/),
  [November timetable](https://londonreconnections.com/the-crossrail-november-timetable/)). The
  branch frequencies are what the core divides into evenly.
- **The cost of mixing.** Planners avoid branches whose headways don't divide each other: the
  trunk then gets uneven gaps and the junction gets conflicts. Pedestrian Observations points out
  that a trunk limited to about 16 trains an hour limits two branches to 8 each, and that heavy
  interlining makes timetables less reliable ([The Limits of Regional Rail](https://pedestrianobservations.com/2020/08/04/the-limits-of-regional-rail/),
  [Systemic Investments in the NYC Subway](https://pedestrianobservations.com/2022/05/23/systemic-investments-in-the-new-york-city-subway/)).
- **Slots and pathing.** In a planned timetable each train gets a path (a slot) through the shared
  section, kept apart by the minimum headway and junction margins; planners plan one hour or half
  hour and repeat it all day ([NEC scheduling, Transit Costs Project](https://nec.transitcosts.com/scheduling),
  [Network Rail, the timetable](https://www.networkrail.co.uk/running-the-railway/the-timetable/)).
  A branch train's path is chosen at the junction and the rest of its run is fitted round it.

### Padding and recovery at the ends

- Timetables carry slack so a late train can recover: about 7% padding in Switzerland, the
  Netherlands and Sweden, up to 25% in Germany, plus station pads
  ([NEC scheduling](https://nec.transitcosts.com/scheduling)).
- **Layover / recovery time** at the end of a route is scheduled time between arriving and leaving
  again, used to recover delays and prepare the return trip
  ([UGPTI transportation glossary](https://www.ugpti.org/surcom/resources/transportation-glossary.php?id=287)).
  A line with a fixed fleet runs at the headway its round trip plus that layover allows, rounded to a
  clock interval; the leftover minutes are stood at the terminal. This is what makes "set trains"
  and "set headway" the same thing in practice: an operator picks the clock headway and the fleet
  follows, or the fleet is fixed and the headway is rounded up with the slack stood at the ends.
- That slack is also what lets the two directions be timed separately: a branch train can be fitted
  to its trunk slot inbound and to another slot outbound, the difference stood at the far end.

### Terminals

- A stub terminal with two platforms and a scissors crossover lets trains use both platforms in
  turn: one train arrives while the one before is still loading on the other platform. Capacity then
  comes from the turnaround split over the platforms (the game already counts it this way: 30 an
  hour at a 3 minute turnaround, 36 at best, the Victoria line's figure, `upgrades::pooled_turn_s`).
- With turnback sidings beyond the platform, a train unloads at the arrival platform, runs into a
  siding to change ends and comes out to the departure platform, so the platforms are held only for
  the dwell.
- Terminal layout is one of the main causes of unreliability: van Oort and van Nes compare terminal
  configurations, double crossovers included ([Impact of Rail Terminal Design on Transit Service
  Reliability](https://worldtransitresearch.info/research/3623)). TCRP Report 165 (the Transit
  Capacity and Quality of Service Manual, 3rd edition) has the rail capacity method
  ([front matter](https://onlinepubs.trb.org/onlinepubs/tcrp/tcrp_rpt_165fm.pdf)); its turnback
  chapter was not read.
- Platform choice decides much of the delay: Jiyugaoka dropped fixed platforms per train type in
  2023 so a train could enter as soon as either platform cleared (`passing-tracks.md`, Traffic News).

## Other games

- **NIMBY Rails.** Every train is simulated against path signals; a line's timetable is the
  arrival and departure times of each train at each stop, built from orders ("depart at", "arrive
  no later than"), repeating weekly, and recalculated on every edit to a line, station or train
  ([wiki, Timetable](https://wiki.nimbyrails.com/Timetable), [How to manual timetable](https://wiki.nimbyrails.com/How_to_manual_timetable)).
  Trains may share a platform from version 1.9; a run whose last stop matches the next run's first
  (on the main platform "or any secondary platform") turns there without a separate stop
  ([wiki, sharing a platform](https://wiki.nimbyrails.com/How_to_have_multiple_trains_share_a_platform)).
  Merging and spacing are left to the player's timetable and signals; who waits is decided by path
  reservation, not by priority (`passing-tracks.md`).
- **Transport Fever 2.** Vanilla has no timetables; trains follow fixed routes and wait for
  occupied blocks. A later update added **alternative terminals** for a line's stops, so an
  arriving vehicle can use another platform ([Steam patch notes](https://steamdb.info/patchnotes/8375959), read only
  through a search summary, not opened;
  a player's description on [Reddit](https://lr.psf.lt/r/TransportFever/comments/1gcb1pj/what_are_these_symbols_in_line_manager)).
  Timetables come from the **Timetables mod** (arrival/departure slots per station, "unbunch"), where
  a line can't run more trains than it has slots ([GitHub, TPF2-Timetables](https://github.com/Gregory365/TPF2-Timetables),
  [TransportFever.net tutorial](https://www.transportfever.net/thread/17500-advanced-timetabling-tutorial-to-enable-real-local-express-running-using-the-tim/));
  players report two trains leaving together from alternate platforms with unbunch on
  ([Steam workshop comments](https://steamcommunity.com/sharedfiles/filedetails/comments/2918585390)).
- **OpenTTD.** Timetables per order list; "autofill" learns the times, and ctrl-clicking the start
  date spreads the vehicles of a shared order list evenly over the cycle
  ([wiki, Timetable](https://wiki.openttd.org/en/Manual/Timetable)). Spacing that holds itself is a
  patch: timetable auto-separation in JGR's Patch Pack, and its scheduled dispatch (fixed departure
  slots), which forum posters say cannot be on together for one order list
  ([tt-forums, JGR's Patch Pack](https://www.tt-forums.net/viewtopic.php?p=1240659),
  [Scheduling patch thread](https://www.tt-forums.net/viewtopic.php?p=1244109),
  [FS#1128 auto-separation](https://bugs.openttd.org/task/1128)). At a terminus a path signal lets a
  train in when any platform is free (`passing-tracks.md`).
- **Subway Builder**, **Mini Metro**: nothing found on how they space lines through shared track
  or turn trains (searched 2026-11-09; no documentation or guide came up).

## What this says for the game

- The real-world answer to an uneven merge is a common headway family plus an offset, and enough
  slack at the ends to time each direction. The game can get that without simulating trains one by
  one, because its trains already run from repeating patterns (`Lattice`), as a clock-face timetable
  does.
- Every game that lets trains interact physically (NIMBY, OpenTTD, TF2) pays for it with per-train
  simulation and leaves the spacing to the player. The game's pattern model gives the spacing for
  free, so the thing to copy from them is the terminal: arriving trains take whichever platform is
  free.
