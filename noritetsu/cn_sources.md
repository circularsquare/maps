# Mainland China (cn): sources, commands, what is still off

Built 2026-09-30 from Geofabrik's china-260929 extract. The reader is `cn_register.py`; its
docstring explains the method. This file is where each input came from, how to rebuild, and the
faults.

## Run

```powershell
python cn_register.py --fetch                  # Wikidata -> data/raw/cn/wd_*.json, ~8 min
curl -L -A "noritetsu-rail-map/1.0" -o data/raw/china-YYMMDD.osm.pbf https://download.geofabrik.de/asia/china-YYMMDD.osm.pbf
python extract.py --region cn --pbf data/raw/china-YYMMDD.osm.pbf     # 2.5 min, 1.6 GB file
python cn_register.py --clip                   # 1 min; after EVERY extract
python build_model.py --region cn --register cn_register:data/raw/cn  # 6 min
python build_tiles.py --region cn              # 5 min
python check_model.py --region cn
```

The plain `china-latest.osm.pbf` URL redirect-loops (as Switzerland's did); take the dated file
from https://download.geofabrik.de/asia/china.html. Delete the .pbf once the extract is checked.

`--clip` keeps what lies inside OSM's own boundary of China (relation 270056,
`data/raw/cn/cn_boundary.geojson`, fetched from
`https://polygons.openstreetmap.fr/get_geojson.py?id=270056&params=0`) and outside Hong Kong
(`data/raw/hk/hk_boundary.geojson`, relation 913110). Geofabrik's extract runs well past the
border: North Korea's northern lines, Vietnam's Lao Cai and Dong Dang lines, the Russian Far
East, Laos, Mongolia and Kazakhstan all had track in it. Hong Kong is its own region (`hk`).
**Macau is in cn**: it is inside the extract and inside China's boundary, and its light rail
(氹仔線, 橫琴線, 石排灣線) is built as OSM lines. Taiwan is not in the extract.

`python cn_register.py --traffic <pbf>` writes `data/proc/cn/traffic_mode.pkl` from the .pbf;
it was a stopgap until extract.py kept `railway:traffic_mode` (landed 2026-09-30) and is not
needed any more.

## Sources

| what | where | used for |
|---|---|---|
| OSM, Geofabrik china-260929 | extract | track (named for its line), stations, route=railway relations (national line code, name:en), metros |
| 12306 station list | `https://kyfw.12306.cn/otn/resources/js/framework/station_name.js` -> `data/raw/cn/12306_station_name.js` (public JS, no login, fetched 2026-09-30) | which OSM stations are passenger stops (3,404 names) |
| Wikidata | `cn_register.py --fetch`: lines of China (P17 Q148, subclasses of Q728937), station adjacency (P197 + P81), stations with points; CC0 | English names of lines and stations; the stage-1 cross-check |
| zh.wikipedia line articles | infobox system_length / length_in_operation, `action=raw`, 2026-09-30 | published lengths in `check_model.REGISTER["cn"]` |

User-Agent for every request: `noritetsu-rail-map/1.0`. Nothing needed a login.

## Stage 1: why this recipe (measured 2026-09-30)

`probe_cn_infra.py` and `probe_cn_lines.py`, outputs in `data/raw/cn/probe_*.txt`:

- Heavy-rail main and branch track (build_tiles.rank_of < 2): 265,027 km, double track counted
  twice. In a route=railway/route=tracks relation 95.1%, in one with a four-digit national code
  79.2%; carrying a line name on the way itself 97.9%; either 98.0%.
- "Passenger-used" track in the task's sense (in a route=train relation) is only 18,541 km,
  because OSM China has just 134 route=train relations, most of them single trains. Of it 95.6%
  is in a relation, 99.7% named. The one big hole in the relations is 青藏线 (named, in none).
- The way names are China Railway's line names and the same strings as the relations'. 1,427
  names carry 5 km or more; 826 of them (185,689 km of track) have a relation of the same name
  with a national code.
- High-speed: 245 named lines are half or more highspeed=yes (97,367 km of track); 133
  infrastructure relations look high-speed by name (高速/高铁/客专/城际).
- Wikidata: 986 lines with station adjacency, 487 in one chain (7,775 stations, 85% found in
  OSM by name within 2 km), but only 298 of those lines map onto an OSM line name; the rest are
  metros and older line splits. Not enough to say which stations are on which line.
- Metros: OSM subway route relations in 45 cities (767 relations), trams, monorails and people
  movers in about 30 more places; 25,908 km of urban track, 95.6% in a route relation.

So: named track is the register (Korea's recipe), the relation gives the code, 12306 says which
stations are passenger stops, proximity to the line's own named track says which line.

## What was built

- 864 lines: **417 register lines, 121,894 km**, of which 142 lines (48,368 km) are mostly
  high-speed; 447 OSM lines (324 subway, 63 train, 30 light rail, 16 tram, 13 monorail, 1
  funicular; 39 of them named trains, 9,220 km).
- 10,638 stations; 3,319 on register lines (3,180 with an English name).
- Register lines: 270 have the national line code as `ref`, 340 an English name.
- `dist/data/cn/lines.json` 1.2 MB (Japan's is 1.2 MB), stations.json 1.2 MB, ways.json 3.4
  MB, geom/ one file per line, about 15 MB. `dist/data/cn.pmtiles` 36.6 MB (34.9 MiB), 80,826
  tiles to z13; Japan's is 10.5 MB.

## Check (check_model.py --region cn)

30 lines against zh.wikipedia's figures: 15 high-speed, 15 conventional. 26 are within 3%,
the four below 0.95 are explained in their REGISTER notes and here.

| line | built | published | ratio |
|---|---|---|---|
| 京沪高铁 | 1305.1 | 1318.0 | 0.99 |
| 沪昆高速线 | 2250.6 | 2266.0 | 0.99 |
| 徐兰高速线 | 1398.6 | 1395.0 | 1.00 |
| 兰新客专线 | 1780.9 | 1786.0 | 1.00 |
| 京广高速线 | 2088.4 | 2118.0 | 0.99 |
| 京哈高速线 | 1224.2 | 1240.0 | 0.99 |
| 郑渝高速线 | 1074.6 | 1065.7 | 1.01 |
| 合福高速线 | 848.4 | 850.0 | 1.00 |
| 贵广客专线 | 828.0 | 857.0 | 0.97 |
| 南昆客专线 | 724.9 | 715.8 | 1.01 |
| 西成客专线 | 635.0 | 683.0 | 0.93 |
| 银西高速线 | 541.8 | 543.0 | 1.00 |
| 沪宁城际线 | 299.8 | 301.0 | 1.00 |
| 京津城际线 | 161.2 | 165.0 | 0.98 |
| 海南东环高速线 | 308.0 | 308.0 | 1.00 |
| 京沪线 | 1447.0 | 1451.4 | 1.00 |
| 京广线 | 2239.8 | 2269.3 | 0.99 |
| 京九线 | 2278.3 | 2311.0 | 0.99 |
| 京哈线 | 1237.7 | 1249.0 | 0.99 |
| 陇海线 | 1709.2 | 1759.0 | 0.97 |
| 兰新线 | 2365.4 | 2413.0 | 0.98 |
| 焦柳线 | 1633.3 | 1639.0 | 1.00 |
| 包兰线 | 980.4 | 990.0 | 0.99 |
| 滨洲线 | 929.4 | 935.0 | 0.99 |
| 湘桂线 | 979.0 | 1013.0 | 0.97 |
| 沈大线 | 393.3 | 399.0 | 0.99 |
| 广深线 | 145.4 | 147.0 | 0.99 |
| 宝成线 | 610.4 | 676.0 | 0.90 |
| 青藏线 | 1833.9 | 1971.0 | 0.93 |
| 成昆线 | 1002.6 | 1100.0 | 0.91 |

## Still off, and why

- **A line whose named track stops short of its terminus loses the last stretch** when the
  last station is more than END_LAST_M (30 km) back: 宝成线 ends at 广汉北 (not 成都), 青藏线
  starts at 湟源 (not 西宁), 西成客专线 at 西安西 (not 西安北), 京广线 at 房山东 and 京九线 at
  北京大兴 (not 北京西). `extend_ends` adds 88 sections carrying a line on to a station close
  beyond its track (滨洲线 into 哈尔滨, 贵广客专线 to 广州南); the build log lists every one.
- **Gaps wider than join_pieces allows stay open**: 成昆线 is two pieces (成都南-花棚子,
  元谋西-昆明), 61 km apart as the crow flies, and 京港高速线 is two (the 庐山-南昌东 gap).
- **Stations by proximity, not by list.** No open per-line station list exists. A passenger
  station within 300 m of a line's named track is on that line, so where two lines run side by
  side a station can join both, and a high-speed line passing close to a conventional station
  would show it as a stop. The own-station count in the log is the check on this.
- **Mostly-freight lines with no traffic tag stay in**: 瓦日线 (1,213 km, 10 sections) and
  唐包线 (491 km, 6) are heavy-haul freight lines that OSM does not tag
  railway:traffic_mode=freight, and 12306 lists stations along them that no other line has,
  so they pass both tests. Whether passenger trains still call there was not checked; if not,
  drop them by name. The log's "no passenger station of their own" list (58 lines) is the
  other place to look; most of it is real passenger lines whose stations all sit beside a
  parallel line (石太客专线, 哈齐客专线) and short connecting lines.
- **Freight tags do not remove a line with passenger stations** (the rule since 2026-09-30,
  decided in the main session): a line with two 12306 stations placed on it is built whatever
  railway:traffic_mode says, and freight-tagged sections are only logged. Five lines are half
  or more freight-tagged and built on that rule, as (freight share, 12306 stations, km):
  胶济线 (0.65, 11, 365: 济南-淄博-潍坊-青岛北, a real passenger line), 南防线 (0.86, 14, 152),
  洛宜线 (0.69, 2, 7), 乐清湾铁路 (0.85, 2, 18) and 赤大白线 (0.97, 2, 242: one 乌丹-白音华南
  section, both stations on its own track). 浩吉线 and 广珠线 are left out by name
  (`FREIGHT_LINES`): each built as one section between two stray 12306 stations beside its
  track (乌审旗-万荣 438 km, with none of its 51 OSM stations on 12306's list; 江门-江高 122 km,
  the Guangzhou-Zhuhai freight railway). 大秦线 and 朔黄线 have no 12306 station on them and
  build no line.
- 54 of 12306's names have no OSM heavy-rail station (the Laos stations 万象, 磨丁; 广州西,
  深圳西, 清华园; some closed halts); the build log lists them.
- No line colours: China Railway publishes none for its lines, so there is no colours/cn.csv.
  Metro colours come from OSM (372 lines carry one).
