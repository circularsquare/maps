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

## Over the border (2026-10-04)

`CN_BORDERS` in cn_register.py: where passenger trains cross into a built neighbour, the line
here runs on from its last station to the border point (borders.py), over any heavy-rail
track (`Net`), listed in the line's `served_sections` so build_model keeps it (no OSM route
runs over either).

| point | section | km | joins |
|---|---|---|---|
| `xFutian` (borders.EXTRA; under the Shenzhen River) | 福田 - China – Hong Kong border, on 广深港高速线 (now 115.9 km) | 4.85 | Hong Kong's 香港西九龍 - border, which hk_register builds under this line's id (`c1d25d1a02a`), so 福田 -> 香港西九龍 is one ride |
| `xDongDang` (borders.EXTRA; Hữu Nghị Quan) | 凭祥 - China – Vietnam border (operator: 湘桂线's, China Railway's Nanning group) | 13.08 | a piece under the id of Vietnam's Đường sắt Hà Nội - Đồng Đăng (read from dist/data/vn/lines.json), so 凭祥 -> Đồng Đăng is one ride |

- **Why Dong Dang joins Vietnam's line and not 湘桂线**: MR1/MR2 run Nanning - Gia Lâm daily
  (since 2025-05-25, vn_sources.md), and the ride a rider enters is across the border. A
  piece of 湘桂线 would meet Vietnam's line at the border point with no line calling at both
  凭祥 and Đồng Đăng (the app never joins register lines of different ids). OSM names the
  track 湘桂线 to 2.6 km short of the border and leaves the rest unnamed. If Vietnam's build
  has no line ending at the point, the section goes on 湘桂线 instead (logged).
- **Crossings with passenger trains that still do not join** (the neighbour's side is not
  this reader's): Manzhouli - Zabaikalsk (an international train twice a week since
  2026-03-08) and Suifenhe - Grodekovo (daily since 2024-12-15) with Russia. Russia's build
  ends at Zabaikalsk station and at Grodekovo's tariff point, and borders.py has no point at
  either crossing; both need a point and Russia's side to it. Not joined, by design until
  their countries are built: Erenhot (Mongolia), Dandong (North Korea), Mohan - Boten (Laos).
  Kazakhstan: no passenger train crosses (casia_sources.md). Lào Cai - Hekou: freight only.
  **Macau** is inside cn, so its light rail at 橫琴 (Lotus) needs no border point.

## Still off, and why

- **Fixed 2026-10-04** (each point queried with its own latitude's cosine): trial build 81 ->
  1 sections with a straight end step over 1 km (渝熊线 1.1 km); 124 lines change, 52 new
  extend_ends sections reach their hubs (京广线 房山东 - 北京丰台, 西成客专线 西安西 - 西安北,
  滨绥线 into 哈尔滨); against check_model's published lengths 7 closer (贵广客专线 0.966 ->
  0.999, 西成客专线 0.930 -> 0.961, 京广线 0.987 -> 0.996), 20 unchanged, 3 further (包兰线
  0.990 -> 1.028 from a new 中宁 - 宣和 section, 南昆客专线 1.013 -> 1.027, 湘桂线 0.966 ->
  0.964). The old note follows. **`extend_ends` starts and ends its sections at the wrong
  track node.** It looks up the network node nearest a station in a k-d tree whose x is
  `lon * cos(lat)` of each node's own latitude, but queries it with the cosine of the named
  track's END's latitude, not the station's. At lon ~110, a few hundredths of a degree of
  latitude between the two move the query by kilometres: the nearest "node" comes back 2-24
  km off (measured at 福田: 6.1 km). In the shipped build 81 of the 88 extend_ends sections
  have a straight jump of over 1 km at one end of their drawn geometry (滨洲线 哈尔滨北-哈尔滨
  11.1 km, 石太线 榆次-太原东 23.5 km), and their km are off with it. The fix is to use each
  queried point's own latitude for k (two lines in `extend_ends`); `border_pieces` avoids the
  tree for the station end. Changes the 88 sections' km and geometry, so it wants its own
  trial and rebuild.

- **A line whose named track stops short of its terminus loses the last stretch** when the
  last station is more than END_LAST_M (30 km) back: 宝成线 ends at 广汉北 (not 成都), 青藏线
  starts at 湟源 (not 西宁), 西成客专线 at 西安西 (not 西安北), 京广线 at 房山东 and 京九线 at
  北京大兴 (not 北京西). `extend_ends` adds 88 sections carrying a line on to a station close
  beyond its track (滨洲线 into 哈尔滨, 贵广客专线 to 广州南); the build log lists every one.
- **Gaps wider than join_pieces allows** were 成昆线 (成都南-花棚子, 元谋西-昆明) and
  京港高速线 (the 庐山-南昌东 gap) among seven lines; since 2026-10-04 each is bridged, split
  or left whole ("Lines in pieces" below).
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

## Lines in pieces (2026-10-04)

Anita, 2026-10-04 ("yes, we can continue doing bridge over shared track"): the UK fix
(gb_sources.md "Lines in pieces") carried to China. A trip is entered station to station on a
line's strip diagram, so a register line whose sections do not all connect cannot be ridden
across its gap. cn_register now has the `split_pieces` hook build_model calls after
`drop_unridden_sections`, through the shared `pieces.py`: after `join_pieces` (still first,
inside build()), a gap is bridged over the track between the pieces where trains run across,
the `borrowed` sections crediting the line whose track it is; what cannot be bridged becomes
one line per piece (the biggest keeps the id, aliases.json `pieces` moves saved rides).

**Measured** on the build shipped 2026-10-03: 7 of 418 register lines in pieces, 1,872 km
outside each one's biggest piece. By what lies in each gap (the track found between the pieces
over all heavy rail in data/proc/cn):

| line | pieces (km) | gap | cause | done |
|---|---|---|---|---|
| 京港高速线 | 1,293.6 + 829.1 | 庐山 - 南昌东 (110 km crow-fly) | shared track: the line's own 南昌 - 九江 section (昌九高铁) is being built, and until then trains run 庐山 - 南昌 on 昌九城际线 ("昌九段暂由昌九城际铁路代替", zh.wikipedia 京港高速铁路; 南昌东 joined 昌赣段 in 2025, crecg.com) | bridged, 145.4 km in 5 borrowed sections, 庐山 - 德安 - 共青城 - 永修 - 南昌 over 昌九城际线 (87 km), then 南昌 - 南昌东 over 京九线 and 杭昌高速线 (45 km) |
| 成昆线 | 758.8 + 243.7 | 花棚子 - 元谋西 | OSM's 成昆线 is the old line: 成都 - 峨眉 - the mountain line to 攀枝花 and 花棚子, and 元谋西 - 昆明. The new line between, 峨广线 (峨眉 - 广通, 552 km), carries the through trains; the old line's middle is gone from OSM | split (`NO_BRIDGE`): the generic rule bridged 攀枝花 - 元谋西 over 121 km of 峨广线, and a ride 成都 - 昆明 entered on 成昆线 would then have credited the old mountain line |
| 沈佳高速线 | 452.9 + 370.2 | 永庆 - 牡丹江 (235 km crow-fly) | no track: the 白山 - 敦化 section is not built, the nearest track joining the pieces is 457 km round by 长珲城际线 and 图佳线 | split |
| 甬广高速线 | 362.4 + 230.2 | 厦门北 - 汕头 (193 km crow-fly) | the 漳州 - 汕头 section is being built; the track between is 杭深线, 200 km of another line, which is a different railway, not this line over a shared stretch | split |
| 川藏铁路 | 139.8 + 44.5 | 雅安 - 林芝 (837 km) | two railways under one name: 成都 - 雅安 and a stretch of the Lhasa - Nyingchi line | split |
| 龙龙高速线 | 98.4 + 63.4 | 武平 - 梅州西 (84 km crow-fly) | no track: the 武平 - 梅州 section is not built (the nearest track is 487 km round) | split |
| 青荣城际线 | 150.6 + 91.0 | 桃村北 - 牟平, at 烟台南 | OSM's track breaks at 烟台南 (121.38 E): the named track stops 0.8 km either side of the station and its unnamed station roads join neither side, so the nearest track joining the pieces is 284 km round by 桃威线 and 蓝烟线. Trains run through | left whole in pieces (`KEEP_WHOLE`): a split piece's id would vanish again once OSM joins the track |

China's settings beside the UK's (`cn_register.rules()`): OSM China has almost no train route
relations (134), so a bridge needs none under it and none is preferred (`ROUTE_SHARE` 0,
`UNROUTED_COST` 1); the line's own named track costs half and, for a mostly high-speed line,
track not tagged highspeed=yes four times its length (`OWN_COST`, `SLOW_COST`); `MAX_KM` 150
for 京港高速线's 145 km; `dense`, so a station on straight track with no OSM vertex within 150 m
joins the track graph; `NO_BRIDGE` and `KEEP_WHOLE` above. The other limits are the UK's (a
bridge at most 1.5 times the crow-fly plus 5 km), which refused 龙龙, 甬广 and 青荣 as
roundabout.

**Split lines** (biggest keeps the id; name_en from the end stops' English names):
沈佳高速线 `ca35071049a` (Shenyangdong – Yongqing) and `c0a996f6e11` (Mudanjiang – Jiamusi);
甬广高速线 `c1ce5b530e6` (Shantou – Xintangnan) and `cd9105d3189` (Xiamen North – Fuzhounan);
川藏铁路 `c50aa03014b` (Chengduxi – Ya'an) and `c7cd9e469c6` (Nyingchi – Milin); 龙龙高速线
`c5dc189a0db` (Meizhouxi – Longchuan West) and `c4442f5f579` (Gutianhuizhi – Wuping); 成昆线
`cbd359d71be` (Huapengzi – Chengdunan) and `c1b667de2bf` (Yuanmouxi – Kunming).

**Not certain**: the bridge's 45 km through Nanchang. G trains from 庐山 reach the 昌赣 line
through 南昌 (and 南昌西); OSM's 京港高速线 piece starts at 南昌东, so the bridge, the cheapest
track, runs 南昌 - 南昌东 over 京九线 and 杭昌高速线, which may not be the trains' track. The
昌九城际线 part is.

**Crediting**: 145.4 km of borrowed sections, all 京港高速线's; riding them credits 昌九城际线
(91.0 km), 京九线 (35.5) and 杭昌高速线 (15.9), the owners of the track; 4.9 km of ways beside
them have no other register line's section near and go to 京港高速线 (logged; the sum is over
both tracks, so it passes 145.4). The country's owned total (build_regions.owned_totals) goes
134,899.9 -> 134,905.1 km: the borrowed km count once, for their owners.

**Before -> after** (trial 2026-10-04): register lines 418 -> 423, 122,863.6 -> 123,009.0 km
(borrowed 145.4 among them); lines in pieces 7 -> 1 (青荣城际线, kept whole). check_model is
unchanged: its only affected row, 成昆线, sums the two pieces (0.91).
