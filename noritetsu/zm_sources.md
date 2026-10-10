# Zambia (zm): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent. TAZARA's Tanzanian half is in
`tz_sources.md`.

### What runs (freshest evidence)

| service | status | evidence |
|---|---|---|
| ZRL Livingstone - Lusaka - Kabwe - Kapiri Mposhi - Ndola - Kitwe (the "Zambezi Express", renamed Michael Chilufya Sata Express) | running, weekly each way (Mon from Livingstone, arr. Kitwe Wed; Fri from Kitwe, arr. Sun); the second train "Kafue" suspended | seat61.com/Zambia.htm (8 Apr 2026); en.wikipedia "Zambia Railways" (as of March 2026) |
| TAZARA New Kapiri Mposhi - Kasama - Nakonde (border), Mukuba Express | running, weekly each way since 10 Feb 2026 (Tue from Kapiri 14:00) | TAZARA notice, Railways Africa, seat61 |
| Livingstone - Mulobezi (the Mulobezi train, OSM route 8473287) | **not built / greyed**: Wikipedia says "occasional mixed freight and passenger"; no timetable found | |
| Lusaka commuter | **not built**: Wikipedia mentions "limited commuter services in the Lusaka area"; no timetable or route found | |
| Chipata line, Kitwe - Chingola, Lusaka - Chirundu | freight only | |
| Livingstone - Victoria Falls (Zimbabwe) | no scheduled train over the bridge | seat61 Zimbabwe |

### Line list

1. **Livingstone - Lusaka - Kapiri Mposhi - Ndola - Kitwe** (stops per seat61/OSM route 2277326 "Zambezi Train": Livingstone, Kalomo, Choma, Pemba, Monze, Mazabuka, Kafue, Lusaka, Kabwe, Kapiri Mposhi, Ndola, Kitwe-Nkana; intermediate list to confirm from OSM's route). Roughly 850 km (Livingstone - Lusaka 472, Lusaka - Kapiri ~200, Kapiri - Ndola ~120, Ndola - Kitwe ~60). Possibly cut at Kapiri Mposhi or Ndola for readability.
2. **TAZARA, Nakonde (border) - New Kapiri Mposhi**: 890.4 km (TAZARA chainage: Nakonde 971.0, Kasama 1,222.3, Mpika 1,424.0, Serenje 1,664.2, Kapiri Mposhi ~1,860). Stops: OSM stations; express between Kapiri and Kasama (major stations only), all stations Kasama - Nakonde.
3. Livingstone - Mulobezi: greyed if built at all.

Expected: 2 running register lines, ~1,740 km.

### Sources

- Stops: OSM routes (Zambezi Train 2277326, Mulobezi Train 8473287), seat61; TAZARA's chainage from en.wikipedia "TAZARA Railway".
- Coordinates: OSM (the station count query timed out on public Overpass; read from the extract).
- GTFS: none.
- Licence: OSM ODbL.

### OSM quality (Overpass, 2026-10-08)

2,502 km of track, only 122 named (TAZARA 79, Maamba colliery 38): named track does not work.
Infra relations: "Zambia Railways" 1789574, "Lusaka-Sakania" 1789561, "Mulobezi Railway",
"Choma–Masuku", "Tazara" 14060545, tracks "Mosetse–Kazungula–Livingstone" (Botswana link,
under construction), four unnamed. Geofabrik `africa/zambia-latest.osm.pbf`, 240 MB.

### Recipe

Hand list traced by rinf.py, `own` = relations 1789574 (ZRL main line) and 14060545 (TAZARA);
shared eafrica reader. Border point at Nakonde/Tunduma for TAZARA. The Zambezi Express and the
Mukuba Express as named trains (one train over the whole line each).

### Open questions

- The Zambezi Express's exact stop list: confirm from OSM's route or a ZRL notice.
- Lusaka commuter and the Mulobezi train: no evidence of a weekly timetable.

## Build (2026-10-08)

Built with `eafrica_register.py` (commands as ke_sources.md "Build"). **Result**: 3 register
lines: running TAZARA New Kapiri Mposhi - Nakonde - border 883.1 and Livingstone - Lusaka -
Kitwe 834.6 (1,717.6 km); greyed Livingstone - Mulobezi 163.7. check_model: TAZARA 883.1 of
889.7 (en.WP chainage: New Kapiri Mposhi 1,860 less the border at 970.3).

Decisions:
- Livingstone - Kitwe: the weekly train's stops (seat61; listed_only). TAZARA: every OSM
  station (the ordinary train's pattern).
- The Mulobezi train ("occasional mixed") built greyed so it can be marked; its OSM route is
  NOT_SERVICE. OSM's Zambezi Train route is the register line again: rules/zm.py SKIP_ROUTES.
- The Tunduma - Nakonde border point XTZZM1 (32.763516, -9.315615): where OSM's track (way
  200454396) crosses OSM's boundary (way 363623831, read from the OSM API's map call because
  Overpass answered 504/500). The app's outline crosses the track 2.5 km further west and put
  Nakonde station in Tanzania, so within 4 km of the point the clip sides by OSM's boundary
  segment instead (eafrica_lines.BORDER_LINES). Needs `eXTZZM1` in borders.EXTRA.
- No border point at Victoria Falls: no scheduled train crosses the bridge.
