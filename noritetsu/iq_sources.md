# Iraq sources

## Survey (2026-10-08)

Research only; nothing built. Region code `iq`.

### What runs

**Iraqi Republic Railways (IRR)**, standard gauge, 2,272 km of network (en.wikipedia "Rail
transport in Iraq"); IRR has no website (Seat61), so what runs comes from news.

| service | route | frequency | evidence | decision |
|---|---|---|---|---|
| Baghdad - Basra night train | Baghdad Central - (Hillah / Babylon, Diwaniyah, Samawah, Nasiriyah) - Basra Al Maqal, 541-553 km | daily, 19:00 each way | Seat61 (updated 18 April 2026); resumed 12 April 2026 and again 7 August 2026 after the Arbaeen pause (2026 news via the search summary) | build |
| Baghdad - Samarra commuter | IRR Northern Line, Baghdad - Samarra, about 120 km | daily, plus a Friday pilgrim train | en.wikipedia "Rail transport in Iraq" (undated sentence) | build; weakest evidence of the three |
| Baghdad - Fallujah commuter | IRR Western Line, Baghdad Central - Fallujah, 65 km | daily (weekday 06:45 from Fallujah, 17:00 back) | resumed September 2023 (Shafaq News, 11 Sept 2023); no closure found since | build |
| Baghdad - Karbala | Karbala branch | pilgrim specials only (7 a day each way for Arbaeen from 26 July 2026) | Shafaq / Iraqi News 2026 | not weekly: track only, not a line |
| Basra - Umm Qasr | 56 km branch | pilgrim specials only | en.wikipedia | not a line |
| Baghdad - Mosul, Mosul - Turkey/Syria, Kurdistan, Basra - Shalamcheh (Iran) | | none (Northern Line north of Samarra under reconstruction; Shalamcheh link under construction) | | not built |

### Line list and km

- **en.wikipedia "IRR Southern Line"** has the station list with km from Baghdad
  International: Baghdad Mansur 9.1, Dora 17.2, Yusufiya 31.0, Mahmudiya 41.2, Iskandariya
  61.4, Musaiyeb 73.0, Hilla 109.7, Hashimiya 134.1, Diwaniya 185.5, Rumeitha 230.9, Samawa
  275.3, Nasiriya 381.3, Suq ash Shuyukh 408.6, ... Rumayla 504.7, Shoeyba 536.9, Basra
  Maqal 552.9 (the table also places Karbala at 97.6, which is the Karbala branch's
  station, not the main line). Use it as the hand list, with stops at the train's calling
  points (Hilla, Diwaniya, Samawa, Nasiriya and the larger halts; no published stop list).
- Northern Line Baghdad - Samarra and Western Line Baghdad - Fallujah: station names from
  en.wikipedia "IRR Northern Line" / "IRR Western Line" and OSM.

### OSM

Overpass (2026-10-08, whole country, counts only): 1,267 railway=rail ways (not service),
**831 named**; 11 route=railway/tracks relations; **no route=train relations**; 78 station
or halt nodes; 7 "subway" relations (probably the proposed Baghdad metro or a mistag;
ignore). Named track is promising but stations are thin (78 for the whole network), so the
trace needs the hand list's stations, with `--fill` for halts OSM lacks (nafrica's recipe).

The relations (bbox query, `data/raw/iq/survey/osm_routes_x.json`; the bbox also catches
Iran's, Türkiye's and Syria's) include **IRR line relations**: 16880309 "IRR Southern Line"
(ref 1), 18383033 "IRR Western Line", 17032672 "IRR Transversal Line" (ref 4),
18575814 "IRR Al Musayyib-Karbala Branch", 17299901 "Baghdad-Kirkuk-Erbil Railway",
17414112 Jalawla - Khanaqin, 18510018 Camp Taji Branch, 17416877 old metre-gauge
alignments; and one passenger route, 21409576 "IRR Southern Service" (ref 1, the Basra
train). No Northern Line (Baghdad - Samarra - Mosul) relation was seen; it may be one of the
unnamed ones.

Geofabrik `asia/iraq-latest.osm.pbf`, 86 MB.

### Timetables

None published. Seat61 (Baghdad - Basra), Shafaq News, Iraqi News. No GTFS.

### Licence

OSM (ODbL); Wikipedia for km.

### Recipe

Either **Iran's** (`ir_register.py`: OSM route=railway relations as the lines, a hand
timetable for what runs), since OSM has IRR's Southern and Western Line relations, or
**nafrica's hand list through rinf.py** (`nafrica_register.py` + `nafrica_lines.py`) if the
Northern Line has no usable relation. I would start with Iran's and fall back per line.
Three register lines, Baghdad - Basra (IRR Southern Line, 553 km, km posts from Wikipedia as
`chain`), Baghdad - Samarra (~120 km), Baghdad - Fallujah (65 km). Expected: 3 lines, about
740 km, about 25 stations. The Karbala and Umm Qasr branches as greyed track at most.

### Open

- Baghdad - Samarra and Baghdad - Fallujah: no 2026 timetable found; built on the last
  evidence. Re-check if a 2026 source turns up.
- Which halts the Basra train calls at.
- Northern Iraq (the Kurdistan Region): no railway, so no outline question arises.

## Build (2026-10-08)

Not Iran's recipe: OSM's one passenger route lists no stops and there is no Northern Line
relation, so all lines are hand lists through `mideast_register.py` (nafrica's code).

    python tools/slot.py 2 -- python extract.py --region iq --pbf data/raw/iraq-latest.osm.pbf --station-areas
    python mideast_register.py --clip iq
    python mideast_register.py --join iq
    python tools/slot.py 2 -- python build_model.py --region iq --register mideast_register:data/raw/rinf/iq

`--join`: OSM maps the Southern Line as two single tracks that each stop 4-39 m short of the
other near Rumaitha and Samawah (and ~90 such loose ends across the network), so no trace
passed Rumaitha - Samawah. It adds a two-node way from each loose end of running track to
running track in another connected piece within 40 m.

**Running: 3 lines, 733.1 km. Greyed: 2 lines, 81.0 km.**

| line | built km | published | stops |
|---|---|---|---|
| Southern Line: Baghdad – Basra | 551.6 | 552.9 (en.WP IRR Southern Line, Basra Maqal's post) | Baghdad Central, Hilla, Diwaniyah, Samawah, Nasiriyah, Basra Maqal (listed only) |
| Northern Line: Baghdad – Samarra | 119.3 | about 120 | every OSM station on it (8) |
| Western Line: Baghdad – Fallujah | 62.2 | 65 (Shafaq News 2023) | every OSM station on it (6; Kadhimiya too: the line leaves the Northern Line north of it) |
| Musayyib – Karbala branch (greyed) | 25.0 | | pilgrim specials only |
| Shuaiba – Umm Qasr branch (greyed) | 56.0 | | pilgrim specials only |

The Southern Line's stations check against en.WP's km posts too: Hilla 109.4 (109.7),
Diwaniyah 184.9 (185.5), Nasiriyah 380.2 (381.3). The two branches start at the main-line
station their trains come through (the junction is ~1 km beyond it), so their sections run
station to station and are kept while greyed. Baghdad - Mosul and the rest of the network:
not built (no passenger trains).

**Decisions**: the Basra train's stops are the survey's calling points (no published stop
list); the commuter lines stop everywhere OSM has a station. OSM's route 21409576 is left out
(`rules/iq.py`). Colours picked (`colours/iq.csv`).
