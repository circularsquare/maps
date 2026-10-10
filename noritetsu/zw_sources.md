# Zimbabwe (zw): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs (freshest evidence)

NRZ suspended every passenger train in 2020 and brought two back on **17 October 2025**, each
weekly in each direction (NRZ on X, 17 Oct 2025: "plying these routes twice a week", meaning
one trip each way; The Herald "All aboard again! NRZ brings back Vic Falls and Mutare trains";
ZimLive):

| service | status | evidence |
|---|---|---|
| Bulawayo - Victoria Falls (via Dete, Hwange), 472 km | running, Fri 19:00 from Bulawayo, Sun 19:00 back | NRZ, Herald, ZimLive; seat61.com/Zimbabwe.htm (2 June 2026) confirms it running |
| Harare - Mutare (via Ruwa, Marondera, Rusape), 273 km | running, Fri 20:00 from Harare, Sun 20:00 back | NRZ, Herald, ZimLive (Oct 2025). seat61 June 2026 still lists it as suspended, which I take as not updated: it reports the Vic Falls train's return from the same NRZ announcement but not this one. Least certain line in the set |
| Bulawayo - Harare, Bulawayo - Chiredzi, Bulawayo - Chicualacuala, Bulawayo - Francistown, Bulawayo - Beitbridge | **not built**: suspended since 2020 (Beitbridge since 2014) | seat61 June 2026, NRZ's own passenger page ("all passenger trains are currently suspended", stale) |
| Bulawayo commuter (Cowdray Park - Bulawayo), Harare "freedom trains" | **not built**: suspended, no return found (a ZUPCO-train plan for Bulawayo and Gweru reported, not confirmed) | nrz.co.zw/passenger-services |
| Rovos Rail to Victoria Falls | named train, occasional | |

### Line list

1. **Bulawayo - Victoria Falls** (Bulawayo, Nyamandhlovu, Sawmills, Gwaai, Dete, Hwange, Thomson Junction, Victoria Falls; stops from OSM stations on the line, NRZ publishes no list). Published 472 km (seat61).
2. **Harare - Mutare** (Harare, Ruwa, Marondera, Macheke, Headlands, Rusape, Odzi, Mutare). Published 273 km (seat61).
3. Optionally Bulawayo - Gweru - Harare (486 km) greyed, so the trunk shows as a known but suspended line; I would include it greyed since it is the main intercity line and likely the next to return.

Expected: 2 running register lines, 745 km (+486 greyed).

### Sources

- km: seat61 (472, 273, 486); en.wikipedia "Beira–Bulawayo railway" for Mutare.
- Coordinates: OSM, 96 railway=station/halt, 90 named.
- GTFS: the Harare kombi feed (Mobility Database mdb-1960) has no rail.
- Licence: OSM ODbL.

### OSM quality (Overpass, 2026-10-08)

The relation and track queries timed out on public Overpass (both servers overloaded today); 96
stations, 90 named. Read the relations from the extract (`inspect_region.py`). Geofabrik
`africa/zimbabwe-latest.osm.pbf`, 171 MB.

### Recipe

Hand list traced by rinf.py in the shared eafrica reader. No border crossing (Victoria Falls
bridge has no scheduled passenger train; Mutare - Machipanda neither).

### Open questions

- Harare - Mutare: is it still running in late 2026? Only the October 2025 relaunch reports confirm it.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"). **Result**: 3 register
lines: running Bulawayo - Victoria Falls 470.5 and Harare - Mutare 267.0 (737.5 km); greyed
Bulawayo - Gweru - Harare 474.3 (managing session's call). check_model: 470.5 of 472, 267.0 of
273, 474.3 of 486 (seat61).

Decisions: every OSM station is a stop (NRZ publishes none). OSM's routes over these three:
rules/zw.py SKIP_ROUTES; the routes for trains that do not run (Chiredzi, Beitbridge,
Francistown, Chicualacuala, Masvingo, Chinhoyi, Shamva, Harare - Beira): NOT_SERVICE.
