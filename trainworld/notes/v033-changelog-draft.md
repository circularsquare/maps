# v0.3.3 changelog, rough draft (maps-19, 2026-11-12, for Anita to cut and reword)

v0.3.1 and v0.3.2 were hotfixes off v0.3, so v0.3.3 carries everything landed on master since v0.3 (3a442a6e).
Held out: two pairs on quad (behind ?pairs), T-3346 stage 2, 350 km/h (her call pending).

update v0.3.3! one way stations, timetables, satellite map, flags...

new
- one way stations: a platform on one track stops trains going one way. on single track, stations where trains can't pass
- single track for high speed and tram (if it lands)
- single track on quad, and single track either side of a station with passing tracks
- turn single track into double, or back, from the Select bar
- schedule panel with a departures board for each stop and a timetable chart
- follow a train from its panel, or double click it
- satellite map (Settings, Map | Satellite)
- flags next to names on the leaderboards and in community, regions too
- line bullets can be circles, squares or diamonds
- trains run on the left or right, set from where you start, changeable in Settings
- 30 more real world networks in sandbox
- real world networks follow real tunnels and bridges, and start already running
- real world network setting: None, Passenger or All. S-Bahn and light rail now included
- drag a bus route to add a point
- zoom buttons
- the speed button shows its speed, and there are speeds down to real time
- memory use in Settings and on saved games
- attach up to 4 screenshots to feedback (needs the server deploy)
- zone names in Chinese and Japanese
- Japanese, Korean and Chinese browsers start in their own currency

map and demand
- new zones filling the gaps in Spain, Portugal, France, Romania, Bosnia, Greece and Albania, and North Korea
- universities, airports and long trips reworked; Japanese resort towns get their visitors
- schematic: cities keep their real direction on big maps, no made-up sea, more room in city centres
- terrain drawn as squares again

changes
- high speed fare default 1.25x (was 1.5x)
- money rates say they're x100

fixes
- trains on single track no longer drawn passing through each other
- trains on a line's way back drawn on the right track
- platforms lengthen past track joints
- a station complex shows one name
- riders can change back onto a line that bends back near itself
- a notice to reload when figures stop updating
- clearer messages for old Safari and browsers without WebGL2
