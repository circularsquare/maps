# Ghana (gh): sources

## Build (2026-10-08)

    python extract.py --region gh --pbf data/raw/ghana-latest.osm.pbf --station-areas
    python wafrica_register.py --clip gh       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill gh
    python wafrica_register.py --convert gh
    python build_model.py --region gh --register wafrica_register:data/raw/rinf/gh
    python build_tiles.py --region gh
    python check_model.py --region gh

Built: **3 register lines, 128 km; 2 running (108 km), 1 greyed (20 km)**.
- Tema – Adome (GRDA, standard gauge), running, 74.7 km: Tema Harbour, Kpong, Adome.
- Adome – Mpakadan, greyed (no train reported past Adome), 19.8 km. Together 94.5 against
  the line's 96.7 (WP; built from Tema Harbour station).
- Accra – Tema (GRCL, Cape gauge), running, 33.5 km: Accra, Odo, Baatsona, Addogonno,
  Asaprochona, Tema Harbour (OSM's stations on the trace). No authoritative length.

Decisions: GRDA lists the line's stations as Tema Port/Harbour, Tema Industrial Area, Ashaiman,
Afienya, Shai Hills, Doryumu-Jokpanya-Kodiabe, Kpone, Adome and Mpakadan (graphic.com.gh,
Nov 2024). OSM has only Tema Harbour and Kpong (65.6 km out, likely GRDA's "Kpone"), so the
others are not stops. **Adome is placed by hand** (`--fill`) at 0.1222, 6.2299: the point
of the track nearest Adomi/Atimpoku, past the Volta bridge, between Kpong and Mpakadan. That
is a guess within a few km; it only decides where running turns to grey. Mpakadan is placed
at the railhead siding. Accra – Nsawam's OSM routes are dropped (no service reported in
2025-2026); Takoradi's lines are not built (survey).

## Survey (2026-10-08)

Research only; nothing built. Two services run, both around Tema; a third (Takoradi) is
uncertain.

### What runs (freshest evidence)

| line | gauge | operator | status | evidence |
|---|---|---|---|---|
| Tema – Afienya – Adorme (on the Tema – Mpakadan SGR) | 1435 | Ghana Railway Development Authority (GRDA) | **running since 1 Oct 2025**, 3 return trips a day Tema – Afienya; fares published to Adorme | graphic.com.gh/news/general-news/grda-activates-tema-mpakadan-rail-services.html (Oct 2025: stations Tema Harbour, Tema Industrial Area, Ashaiman, Afienya, Adorme, Mpakadan; fares Tema–Afienya GH¢15, Afienya–Adorme GH¢25); 3news.com "residents of Tema and Afienya call for more operational hours" (3 trips a day, Tema – Afienya) |
| Accra – Tema (Tema branch, Cape gauge) | 1067 | Ghana Railway Company (GRCL) | running: one return a day (morning in, evening out) in Oct 2024; "suspended for repair works, expected to resume Monday" in Oct 2025 so passengers connect to Mpakadan | trainstobeyond.com/ghana-railway-network/ (Oct 2024); the Graphic article above |
| Sekondi – Takoradi workers' train (Cape gauge) | 1067 | GRCL | uncertain: one return a day in Oct 2024 (trainstobeyond); nothing newer found | |
| Afienya – Mpakadan (the rest of the SGR, 97 km in all) | 1435 | GRDA | stations built, no evidence trains go beyond Adorme | |
| Accra – Nsawam (Eastern line), Takoradi – Tarkwa (resumed Feb 2020), Kumasi lines | 1067 | GRCL | no evidence of service in 2025-2026 | OSM still carries route relations 14435104/14435107 "Accra - Nsawan" |

Decision for a build: register lines **Tema – Afienya – Adorme running**, Adorme – Mpakadan
built and greyed; **Accra – Tema running**; Accra – Nsawam, Takoradi – Tarkwa greyed or
left out (Anita's call is "decide yourself": I would leave them out, since nothing has run
there since at least 2023). Sekondi – Takoradi: too short and too unclear; leave out unless a
2026 source turns up.

Tema – Mpakadan: 96.7 km, inaugurated 25 Nov 2024 (en.wikipedia "Tema-Mpakadan Railway Line"),
idle until Oct 2025 (graphic.com.gh, 20 May 2025: "inactive six months after opening").

### OSM (Overpass, 2026-10-08, bbox of Ghana; it also catches Togo's Lomé lines)

- Route relations: route=train "Accra - Nsawan" both ways (14435104, 14435107); route=railway
  "Eastern Line" (ref 1E, 11919949), "Takoradi - Tarkwa - Dunkwa - Kumasi" (11921649), one
  unnamed (11919538). **No route=train for Tema – Mpakadan or Accra – Tema.**
- Track: 191 `railway=rail` ways in the bbox, 98 named: "Eastern Railway Extension" (29 ways,
  standard gauge: the Tema – Mpakadan line), "Eastern Railway Line" (13), "Tema Branch Line"
  (4), "Kojokrom-Tarkwa Railway", "Sekondi-Tarkwa Railway", "Takoradi - Dunkwa Railway",
  "Dunkwa - Kumasi Railway", "Kojokrom Sekondi line". Gauge tagged on most (1067: 109, 1435: 54).
- 45 station/halt objects in the bbox, 43 named (some Togolese).
- So the two running lines have named track but no passenger route: a hand list is needed
  for stops.

### Timetables / GTFS

No Ghana feed in Transitous. The Mobility Database catalogue is behind Cloudflare for
scripts (share.mobilitydata.org/catalogs-csv; Anita can check in a browser). An Accra
trotro GTFS exists from 2015 (Accra Mobility, buses only), irrelevant.

### Recipe

A hand line list traced by rinf.py, in the shared West/Central Africa reader proposed in
`ng_sources.md`. Two or three lines:
- `Tema – Adorme` (stations Tema Harbour, Tema Industrial Area, Ashaiman, Afienya, Adorme),
  running; `Adorme – Mpakadan`, greyed. Trace over "Eastern Railway Extension".
- `Accra – Tema`, running, over "Tema Branch Line" / "Eastern Railway Line"; stations from OSM
  along the trace (`osm_stops`), as there is no stop list.

Expected about 2-3 register lines, ~130 km (~45 running). Extract:
`africa/ghana-latest.osm.pbf`, **110 MB**.

Checks: Tema – Mpakadan 96.7 km (WP, Railway Gazette "Ghana's Tema–Mpakadan railway
commissioned"); Accra – Tema about 30 km (no authoritative figure found).

### Licences

OSM ODbL; press facts only.

### Open questions

- Does any train run beyond Adorme to Juapong / Mpakadan? (Fares published only to Adorme.)
- Accra – Tema after the Oct 2025 repairs: resumed as announced? No later report found.
- Accra – Tema's stations: no published list; OSM's along the trace.
