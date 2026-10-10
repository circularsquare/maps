# Cambodia (kh): sources

Built 2026-10-08 (asia agent): see "Build (2026-10-08)" at the end. The survey below was the research for it.

## Survey (2026-10-08)

### What runs

Royal Railway (Royal Railway Cambodia, the concessionaire of the state's metre-gauge network)
runs two daily round trips (seat61 "Cambodia", updated 6 June 2026; Royal Railway's sales via
royalrailway.easybook.com and baolau.com):

| train | route | times | notes |
|---|---|---|---|
| Southern Line | Phnom Penh 07:00 - Takeo 08:40 - Kampot 10:40 - Sihanoukville 13:30; back 14:00 - 20:30 | daily | 263 km by seat61; also calls at Kep, Damnak Chang'aeur, Veal Renh |
| Northern Line | Phnom Penh 06:40 - Battambang 13:30; back 15:00 - 21:30 | daily | 273 km; calls at Pursat, Moung (Mong Russei) |

- **Battambang - Sisophon - Poipet**: no passenger train. It ran briefly from July 2024 as a
  separate Battambang - Poipet afternoon train (backpackmoments, "Trains in Cambodia"); seat61
  in June 2026 says suspended, and the Thai-Cambodian border has been closed since the 2025
  fighting. Recommendation: build it, greyed (`suspended`), as Kép - Cái Lân in Vietnam.
- **Poipet - Ban Khlong Luk (Thailand)**: the 2019 link, never had a regular passenger train,
  border closed. Greyed with the section above; Thailand's side is already built in th (its
  Eastern line ends at Ban Khlong Luk Border).
- **Phnom Penh airport shuttle** (old Pochentong airport branch): closed 2020, track being
  removed (en.wikipedia "Rail transport in Cambodia", read 2026-10-08). Leave out.
- No urban rail. No tram, metro or people mover open.

### Sources

| source | gives | licence | where |
|---|---|---|---|
| OSM via Geofabrik `asia/cambodia-latest.osm.pbf` (39.1 MB) | track, 19 station/halt nodes, two `route=railway` relations: "ខ្សែភាគខាងត្បូង" / Southern Line (17033173) and "ផ្លូវរថភ្លើងកម្ពុជាភាគខាងជើង" (17056375), operator ផ្លូវដែកកម្ពុជា | ODbL | Overpass sample `data/raw/kh/survey/osm_route_relations.json` |
| en.wikipedia "Rail transport in Cambodia", "Northern Line (Cambodia)", "Southern Line (Cambodia)" | station order, line lengths (Northern 386 km Phnom Penh - Poipet, Southern 266 km; network 612 km) | CC BY-SA | not saved |
| seat61 "Cambodia" (2026-06-06) | the timetable above | (reading only) | https://www.seat61.com/Cambodia.htm |

OSM counts (Overpass, 2026-10-08, inside Cambodia's boundary): 233 `railway=rail` ways, 162 of
them named, 163 tagged `usage=main`, 69 service ways; 19 stations and halts. Named track is
good (the two lines' names), so Thailand's and Vietnam's recipe fits.

No GTFS: the Mobility Database (feeds_v2.csv, 2026-10-08) has only Phnom Penh's city bus
(tld-5889, inactive). No open register; Royal Railway publishes no line or km list.

### Recipe

`th_register.py` / `vn_register.py` pattern (OSM named track, cut into the lines):

- Two register lines, Northern (Phnom Penh - Poipet, plus Poipet - border) and Southern (Phnom
  Penh - Sihanoukville, with the short port spur left out).
- Stations by proximity to the line's track, as vn does; extract with `--station-areas` in
  case stations are mapped as areas. 19 OSM stations is thin for ~650 km: Royal Railway's
  stops are about 10 on each line, so it may be enough; check that Pursat, Moung, Battambang,
  Takeo, Kampot, Kep and Sihanoukville each come through.
- Battambang - Poipet - border `suspended` (greyed).
- `check_model` REGISTER: Northern 386 (en.wikipedia), Southern 266 (or 254 depending on
  source; the build agent picks one and says why).
- Border point with Thailand at Poipet / Ban Khlong Luk (no trains; for the drawing only).

Expected: 2 lines, about 650 km, of which about 270 km of the Northern Line running.
Build time: seconds.

### Open

- Whether Battambang - Poipet runs at all in 2026 (seat61 says no; nothing found saying yes).
- Station names: OSM's are Khmer; `name:en` coverage not checked.

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region kh --pbf data/raw/cambodia-latest.osm.pbf --station-areas
    python asia_register.py --clip kh
    python asia_register.py --convert kh
    python build_model.py --region kh --register asia_register:data/raw/rinf/kh
    python build_tiles.py --region kh; python check_model.py --region kh

Reader: `asia_register.py` (lk_register.py's engine) with the list in `asia_lines.py` (`KH`);
settings `rinf_countries/kh.py`, rules `rules/kh.py`, colours `colours/kh.csv` (picked).

| line | km built | status | check |
|---|---|---|---|
| Northern Line, Phnom Penh - Battambang | 273.0 | running (daily) | seat61 273: 1.00 |
| Southern Line, Phnom Penh - Sihanoukville | 263.3 | running (daily) | seat61 263: 1.00 |
| Northern Line (Battambang - Poipet) | 111.1 | greyed | en.WP 386 less 273 = 113: 0.98 |

647 km of register line, 536 running, 111 greyed. No OSM lines (no urban rail).

Decisions:
- **Battambang - Poipet greyed** (seat61, June 2026: "temporarily suspended since the
  pandemic"; the 2024 revival did not last and the border is shut since the 2025 fighting). A
  line of its own, since rinf.py greys a whole line or none.
- **Poipet - border not drawn.** The list runs it to `xAranyaprathetPoipet`, but OSM has no
  track for ~470 m between Poipet's yard and the bridge stub at the border, and build_model
  drops a junction section no route runs over. Thailand's side ends at Ban Khlong Luk Border
  too, so nothing is lost; if OSM fills the gap the section comes back on a rebuild.
- **Stops**: `osm_stops: "all"`. OSM maps 20 stations; the trains call at the big ones (seat61
  lists Takeo, Kampot, Pursat) and the halts between are kept as stops (Bat Doeng, Tbeng Khpos,
  Kraing Skea, Phnom Thipdey on the Northern Line; Touk Meas, Kep, Prey Nob on the Southern).
  The Banan bamboo-train stops do not lie on the main line and were not taken.
- Phnom Penh - the junction 9.4 km out is shared track; ownership gives it to the Northern Line.
- No border point needed beyond borders.EXTRA's `xAranyaprathetPoipet` (already there).
