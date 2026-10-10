# Laos (la): sources

Built 2026-10-08 (asia agent): see "Build (2026-10-08)" at the end. The survey below was the research for it.

## Survey (2026-10-08)

### What runs

- **Laos-China Railway (Boten - Vientiane)**, standard gauge, opened 3 Dec 2021, operated by
  the Laos-China Railway Company (LCR). Several CR200J "Lane Xang" EMU and ordinary trains a
  day between Vientiane and Boten/Luang Prabang, plus the cross-border D887/D888 Kunming South -
  Vientiane daily since April 2023. OSM carries the trains as route relations (C82, C84
  Vientiane - Boten). Clearly running.
- **Thai metre gauge into Vientiane**: Nong Khai - Thanaleng over the Friendship Bridge, and
  Thanaleng - Khamsavath (Vientiane) since 2023. SRT runs 133/134 Bangkok - Khamsavath (since
  July 2024) and 147/148 Udon Thani - Khamsavath (in OSM as relations 17865058/17865059), plus
  the Nong Khai - Khamsavath shuttles. Daily. Thailand (th) already builds the Nong Khai line
  to the border (+2.7 km over the bridge, th_sources.md).
- No urban rail.

### Line list (Laos-China Railway)

en.wikipedia "Boten–Vientiane railway" (read 2026-10-08), passenger stations with km from Boten:

| station | km |
|---|---|
| Boten | 0 |
| Na Moh (Nateuy) | 28 |
| Muang Xay (Oudomxay) | 67 |
| Muang Nga | 113 |
| Luang Prabang | 168 |
| Kasi | 239 |
| Vang Vieng | 283 |
| Phon Hong | 342 |
| Vientiane | 406 |

Total 422 km (the line to Vientiane South / Thanaleng dry port, freight, makes up the rest;
leave that branch out unless a passenger train is found on it). The km look rounded to the
kilometre; good enough as `km_official`.

### Sources

| source | gives | licence | where |
|---|---|---|---|
| OSM via Geofabrik `asia/laos-latest.osm.pbf` (51 MB) | track (697 rail ways, 497 named, 501 `usage=main`), 21 stations and halts; `route=railway` 10168930 "ລົດໄຟລາວ-ຈີນ" (Vientiane-Boten Railway) and SRT's 17458406 "Thanon Chira Junction - Khamsavath railway line"; route=train 7750061 (the line), 14241472/3 (C82, C84), 17865058/9 (147/148) | ODbL | `data/raw/la/survey/osm_route_relations.json` |
| en.wikipedia "Boten–Vientiane railway" | the station table above | CC BY-SA | not saved |

No GTFS (Mobility Database 2026-10-08: nothing for LA). 12306 does not sell LCR's domestic
trains; LCR sells through its own "LCR Ticket" app (not looked at).

### Recipe

Small enough for the `th_register` named-track pattern, or a hand list through rinf.py as
`nafrica_register` does; the hand list is less work for two lines:

- **Laos-China Railway**: one register line Boten - Vientiane, the 9 stations above with their
  km as `chain`, traced over OSM's track; plus Boten - border (the Mohan tunnel), a section to a
  border point. `operator` LCR, `highspeed` false (160 km/h line).
- **Thanaleng line**: one register line, border (Friendship Bridge) - Thanaleng - Khamsavath,
  about 3.5 + 7.5 km, operator SRT (it runs the trains) or the Lao state; the build agent
  decides and writes it down.
- Border points: Boten / Mohan with cn (cn_sources.md says China's side is waiting for this),
  and the Friendship Bridge with th (th builds to the border already; same id on both sides).
- `check_model` REGISTER: 406 km Boten - Vientiane (en.wikipedia; 422 with the freight branch),
  Thanaleng - Khamsavath 7.5 km.

Expected: 2 lines, about 420 km. Extract `--station-areas` (Chinese-built stations are often
areas). Build time: seconds.

### Open

- Whether any passenger train uses the Vientiane South branch (assumed not).
- English station names: OSM's are Lao; LCR's own English names (Vientiane, Vang Vieng, Luang
  Prabang...) should go in via `name:en` or a hand map.

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region la --pbf data/raw/laos-latest.osm.pbf --station-areas
    python asia_register.py --clip la
    python asia_register.py --convert la
    python build_model.py --region la --register asia_register:data/raw/rinf/la
    python build_tiles.py --region la; python check_model.py --region la

Reader: `asia_register.py` (lk_register.py's engine) with the list in `asia_lines.py` (`LA`);
settings `rinf_countries/la.py`, rules `rules/la.py` (every route=train a named train),
colours `colours/la.csv`.

| line | km built | status | check |
|---|---|---|---|
| Laos-China Railway, border (Mohan) - Boten - Vientiane | 409.0 (406 by the km table + 3 traced to the border) | running | en.WP 406: 1.01; chainage median 1.000 |
| Thailand's Nong Khai line (`t4a13e49b73`), border - Thanaleng - Khamsavath | 9.8 | running | ~2.5 + 7.5: 0.98 |

419 km of register line, all running. OSM lines: the LCR's C82 and SRT's 147/148, both
named trains (147/148 has the same id as in Thailand, so it joins too).

Decisions:
- **LCR stops are the nine passenger stations** (`listed_only`): OSM's other stations on the
  line (Vang Khi, Pha Daeng, Phoukhoun, Xiang Ngeun, Huoay Han, Na Khok, Na Thong, Nateuy,
  Phon Soung, Vientiane North) are passing loops or not open to passengers. "Na Moh" in the
  table is OSM's Namor (28 km from Boten fits it; Nateuy is 12 km). The en.wikipedia km from
  Boten are the register's chainage (rounded to the km; `km_official`).
- **The Thanaleng line is Thailand's line continued.** SRT runs every train on it (133/134
  Bangkok - Khamsavath, 147/148 Udon Thani - Khamsavath, the Nong Khai shuttles), and OSM's own
  route=railway for it is "Thanon Chira Junction - Khamsavath". So asia_register (`JOIN`) gives
  it the id and name of th's สายชุมทางถนนจิระ–หนองคาย, read from dist/data/th/lines.json, as
  cn_register does for Đồng Đăng: Nong Khai -> Khamsavath is one ride. The operator stays SRT.
- **Vientiane South (freight) left out**: no passenger train found on the branch.
- **Boten - Mohan**: the LCR line ends at a new border point `xBotenMohan` (101.687244,
  21.179426, inside the Friendship Tunnel), where OSM's track (way 732797364) crosses OSM's
  boundary (way 1482078930). Proposed for borders.EXTRA, and China's side (磨憨 - border) as a
  cn_register CN_BORDERS piece under this line's id (handoff_notes/asia_build.md). Until then
  the point shows its id as its name.
