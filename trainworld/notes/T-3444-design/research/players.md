# players notes (T-3444). Raw inbox list: inbox.txt; full threads of 24 items: full.txt (same folder)

## Inbox (--hours 2000, 273 threads; about 55 on topic), grouped
Quad / AA-BB / three-track (7): 1794302077 (quad upgrade, lines don't use 4 tracks; Anita: place stations both sides, 'quad = 2 doubles'), 1794485386, 1794711765, 1794852386 (direction-paired quad, x3), 1794605160 (three-track or custom count; Anita 'planned as track rework'), 1794342410 (quad on high speed, 4-car platform bug, fixed), 1794752820 (up-up/down-down quad with one-direction stations).
Single/double joins (7): 1794536610 (two singles into one double, open), 1794498471 (single from single shifts; second single from double's other track refused, open task T-3353), 1794547102 (single at double's end shown joined, lines can't run), 1794752820, 1794498601 (trains pass through each other on single), 1794811284 (join track to station front, Hiroshima), 1794643454.
One-way / one-direction stations on single track (5): 1794643954, 1794894256 ('a station needs a track going through it'), 1794643454 (12 tph cap, can't place one-way station), 1794752820, 1795163330/1795061964 (rename joins all stations - complex names, fixed).
Upgrade single<->double (6): 1794586572, 1794617845, 1794494649, 1794923198 (tram), 1794536610; Anita added select-tool upgrade 11-17.
Passing / sidings / overtaking (3): 1795013423 (2-platform 4-track station so locals can be overtaken at a major stop), 1794707957 (passing janky at sidings; services should not have to stop), 1794642959 (signal station: no boarding).
Edit/delete/pick track under stations (5): 1794782422 (track under station can't be edited; layers; Anita added layer toggle), 1794157966, 1794158060, 1794274226 (add vertices), 1794496962 (lengthen platform too strict, fixed T-3286).
Station display/names/circles (7): 1795024290 (station circle size/shape), 1794584594 (complex = one name, done), 1795130903 and 1795163330 and 1795061964 (transfer stations keep own names, setting added), 1794635355, 1795190085.
Direction / handedness (1): 1794475501 (left/right running; added as game setting).
Other station/line wishes: 1794746390 right-click leaves track mode; 1794807176 free bus stop placement; 1794561921 group lines; 1794500378 single-line platforms and freight; 1795245615 train split/join at stations; 1794643979 service types on a line.

## board-requests.json groups (track topics, count = reports)
g003 AA/BB quad 5 open (T-3348); g010 single/double joins 5 open (T-3379); g011 upgrade single to double 3 dismissed + g083 1 merged (T-3052; 'people should rebuild'); g012 edit/delete built track 4 open; g015 one-way single track limits 4 open; g022 quad lines don't use new pair 2 open (T-3335); g021 passing locals leave before express 2 merged (T-3346); g046 trains pass through on single 2 merged (T-3264); g048 running side by line 2 open; g049 three tracks/custom counts 1; g099 passing station 2 platforms 4 tracks 1; g073 join track at station end 1; g108 loop line over split single 1; g123 place stations before tracks 1; g101 station circle size/shape 1; g047 signal stations 1; g040 service types on a line 1; g106 transfer stations keep names 2 dismissed; g036/g044/g031 fixed.
Totals: roughly 60 reports in about 28 groups; quad/AA-BB/3-track 8, single-double joins and one-way 15, upgrades 4, edit/delete 4, passing 5.

## Archive counts
{L:single-track}: 26 in ARCHIVE.md (0 in the pre-10-19 file), plus 4 open in TASKS (T-3662, T-1916, T-3354, T-3353). Other lines: track-kinds 0 archived/4 open; quad-track 2 archived/3 open; passing-tracks 3/5; station-placement 10/4; track-drawing 17/13.
single-track 26 split: one-way stations/legs, stop order and side platforms 15 (T-3711, 3700, 3672, 3651, 3647, 3605, 3595, 3565, 3429, 3411, 3386, 3101, 3673, 3676, 3677); single/double joins and drawing 7 (T-3609, 3618, 3541, 3538, 3379, 3380, 658); capacity, waits, doubling refused 4 (T-3526, 3562, 3383, 1681). Passing: T-3929 (express drawn through locals, 11-21), T-3346 stage 2.
Quad: T-3333 (both pairs, hidden twin line), T-3363 closed duplicate; open T-3335, T-3348, T-340 all {after:T-3444}.
Open after-rework: T-1948, T-3036, T-3353, T-3662, T-3354, B10.

## Anita's TODO.md
Nothing on these topics (file has only generic long-term items: campaign mode, weather, settings icons). Her wishes are in notes/T-3444.md and the inbox replies.

## DECISIONS a redesign must respect or revisit
2026-11-12 T-3365: quad is one remembered thing, second click for one double or one track; 'simple' (doubles) and 'advanced' (every track) are ways of thinking, no mode switch.
2026-11-12 T-3348/T-3363: two quad pairs, one-click quad station and AA/BB quad ship together as one package.
2026-11-12 T-3333: line on both quad pairs runs through hidden twin line; one pair's counts stored, halved.
2026-11-12 T-3356: a stop's station on the other pair = any station of same complex, 4-16 m beside.
2026-11-12 T-3332: one-track station on double/quad stops its way, passes the other; half dot; upkeep 0.6x (11-13).
2026-11-12 T-3262: single high-speed track forbidden.
2026-11-12 T-3297: left/right-hand running switches single-track one-way legs.
2026-11-12 T-3212: station split on single track shows only when double's tracks part (zoom ~17).
2026-11-13 T-3348 stage 1: per-direction stops; other way stops at nearest station of the complex or within 300 m.
2026-11-17 T-3605/T-3565: one-way station on one-way single track is a stop of its own, ordered by the run passing it; no pairing.
2026-11-17 T-3540/T-3564/T-3563: station on single track draws one platform to one side; with passing loop a platform each track; trams identical to trains.
2026-11-17 T-3526: two-way single track queues as one stream; cap trains that fit + 1. T-3569: short single line keeps cap.
2026-11-17 T-3609: single drawn at a double's split takes the double's track on its own side.
2026-11-18 T-3651/T-3672: one-way station picked by run reaching it without turning back more; single runs meeting at junction take their way together (general fix).
2026-11-18 T-3711: shift-click makes one-way stop, way that turns back fewer times.
2026-11-18 T-3690: route end within 5 m of a piece end slides to the joint (SPLIT_ROOM 5 m stays).
2026-11-17 T-3574/T-3575: tram stops get terminal and passing tracks; tram track takes crossovers.
2026-11-19 T-3346 ttport: locals' waits at passing stations worked out train by train.
