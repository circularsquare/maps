# Uganda (ug): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs (freshest evidence)

| service | status | evidence |
|---|---|---|
| Kampala - Namanve - Mukono commuter (URC), metre gauge on the old Uganda Railway main line | running weekdays; four trips in early 2026, more added 31 Aug 2026 (Mukono 06:30 - Namanve 07:05 - Kampala 07:40; Namanve 08:15 - Kampala; Kampala 13:30 and 17:30 out); halts Nakawa, Spedag, Kireka, Namboole | Matooke Republic 31 Aug 2026; rogerfarnworth.com Feb 2026 roundup (4,000 riders a day, restarted May 2024) |
| Kampala - Kyengera (Masaka road) and Kampala - Port Bell | **not built**: approved/planned, not running | Daily Monitor "New train routes approved as URC expands passenger services" |
| Kampala - Jinja weekend train | **not built**: planned "before the end of 2026" | ChimpReports |
| Tororo - Gulu - Pakwach, Kasese/Hima branch, Malaba - Kampala | freight or closed; no passenger | |
| SGR | not built | |

### Line list

1. **Kampala - Mukono** (Kampala, Nakawa, Spedag, Kireka, Namboole, Namanve, Seeta?, Mukono): about 22 km on the Uganda Railway main line. One register line; Mukono is the furthest stop in the 2026 timetable. Beyond Mukono the main line to Jinja/Malaba stays OSM track.

Expected: 1 register line, ~22 km. The smallest country in the set.

### Sources

- Stops: the 2026 timetable as reported (Matooke Republic, ChimpReports "Uganda Railways Launches New Mukono-Kampala Commuter Service"); OSM route relations 14620665/6 ("Kampala suburban train Kampala => Namanve" and back, with stops to Namanve only; Mukono needs adding from OSM's station or the timetable).
- Coordinates: OSM, 24 railway=station/halt, 22 named.
- GTFS: the Kampala paratransit feed (Mobility Database mdb-1813) has no rail.
- Licence: OSM ODbL.

### OSM quality (Overpass, 2026-10-08)

998 km of track, almost none named (14 km); infra relations "Uganda Railway" 7168563,
"Hima Branch", "Tororo - Pakwach (- Arua)". Geofabrik `africa/uganda-latest.osm.pbf`, 354 MB
(large for one 22 km line; consider clipping, or pairing with Kenya's extract is not possible
as Geofabrik has no East Africa bundle).

### Recipe

Hand list of one line traced by rinf.py in the shared eafrica reader, `own` = relation 7168563.
Or simply the OSM route as an OSM line, if one register line is not worth the reader entry; I
would still give it a register line so the country counts.

### Open questions

- Is "Seeta" a stop? Not in the reported timetable; the timetable names Mukono, Namanve, Kampala only. Use OSM's stations on the traced piece that the route stops at.
- Kyengera / Port Bell / Jinja: watch for opening.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"; `--fill ug` adds Mukono).
**Result**: 1 register line, Kampala - Namboole - Namanve - Mukono 23.7 km, running. No
published length to check against (the survey's "about 22 km" is an estimate); the timetable
agrees (Namanve - Kampala 35 min for 12.2 km, Mukono - Kampala 70 min). Mukono's halt is not in
OSM: placed where the main line passes closest to the town (32.7626, 0.3293), approximate.
Nakawa, Spedag and Kireka halts are not in OSM and not added (no coordinates). OSM's two
"Kampala suburban train" routes: rules/ug.py SKIP_ROUTES.
