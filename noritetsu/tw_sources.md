# Taiwan register sources (surveyed 2026-09-30)

What `tw_register.py` reads, where each piece came from, and what is wrong with it. The
downloads are in `data/raw/tw/` (gitignored); this file is the tracked record of them.
Nothing here needed a login, a key or an account.

## The short answer

- **Geometry: OSM's named track**, as in Korea. 98% of main and branch rail km in the
  Geofabrik extract (`taiwan-260929.osm.pbf`, data to 2026-09-29) carries its line's name
  (`python probe_kr_ways.py --region tw`); subway 99.8%, light rail 99.7%. Licence ODbL.
- **TRA lines, stations and km: TRA's own open data** (two JSON files below). The only
  register with per-section km, and it covers every TRA passenger line.
- **THSR and every metro and light-rail line: OSM's own route relations** for which stations
  a line has, and published line lengths to check against. No open file gives their
  station-to-station km without an account.
- **Alishan Forest Railway: zh.wikipedia's station table** (names and km).
- **TDX (MOTC's Transport Data eXchange) was not used.** It needs a free account for any
  real use. It is the one place that would give THSR's and the metros' station order with
  distances in one consistent form (its `StationOfLine` and line-shape endpoints). Nothing in
  this build needs it: every line is within a few percent of its published length without it.

## TRA (國營臺灣鐵路股份有限公司)

Licence for both: 政府資料開放授權條款-第1版 (Open Government Data License, version 1.0),
free, attribution.

### 鐵路里程, data.gov.tw 6999  `tra_mileage.json`

https://data.gov.tw/dataset/6999 ->
https://ods.railway.gov.tw/tra-ods-web/ods/download/dataResource/f0906cb8dcee4dfd9eb5f8a9a2bd0f5a

260 rows of `{lineName, fkSta (station code), staMil (km)}`, updated 2024-05-13.

- **Keyed by ticketing trunk, not legal line.** `西部幹線` is 基隆 to 屏東 by 山線 (0-420.8),
  `西部幹線 (海線)` is 竹南-彰化 by 海線, `東部幹線` is 八堵-臺東 (宜蘭線+北迴線+臺東線),
  and `南迴線` is 屏東-臺東, i.e. the south half of 屏東線 plus 南迴線 proper. `LEGAL` in
  tw_register.py cuts these into the legal lines by station code.
- `內灣線` starts at 北新竹; the legal line (and OSM's track name) starts at 新竹, 1.4 km
  earlier on 西部幹線.
- Codes the station file does not have (0995, 1045, 1105, 3155, 3355, 4230, 4240, 5115,
  5173, 5177, 5180, 5195, 5205, 5215, 5225, 6045 ... 7115, 7075) are signal stations,
  junctions and planned stops. 3355 is where 山線 and 海線 meet north of 彰化. 1001
  `臺北-環島` is a ticketing alias of 臺北.
- Freight and depot branches are in it too (臺中港線, 花蓮港線, 高雄港線, 中興一號/二號特種支線,
  樹林調車場出入庫線, 富岡/潮州基地側線) and the closed 舊山線 and 舊臺東線; none is built.
- `深澳線` runs on to code 7363 at 6.0 km, past 八斗子 (4.7); the passenger line ends at
  八斗子. `沙崙線` is 5.7 km here where zh.wikipedia says 5.3; the build measures 5.7.

### 臺鐵車站基本資料集, data.gov.tw 33425  `tra_stations.json`

https://data.gov.tw/dataset/33425 ->
https://ods.railway.gov.tw/tra-ods-web/ods/download/dataResource/0518b833e8964d53bfea3f7691aea0ee

245 passenger stations: code, Chinese and English name, address, phone, `gps` ("lat lon").
Every station in the mileage file that is a passenger stop is here. The points are good to
a few tens of metres; they decide which OSM station a listed name is (nearest the point),
which keeps TRA's 左營 (OSM `左營(舊城)`) off THSR's 左營 beside 新左營. English station names
come from here.

### 臺鐵營業里程及車站數, data.gov.tw 33387  `tra_km_by_year.json`

National passenger / freight km by year, but it stops at 2016 (1,057.6 passenger km). Not
used beyond that sanity figure; the TRA lines built here sum to 1,056 km (the 5 km 彰化
approach counted on both 臺中線 and 海岸線).

### Not used

- data.gov.tw 73226 臺鐵車站 (國土測繪中心 landmark points): points only, no lines.
- The TRA site (railway.gov.tw) timetable pages: dynamic, and the open files above are
  the same data.

## THSR, metros and light rail: OSM route relations

Which stations each line has comes from the stop members of OSM's route relations for it
(`ROUTES` in tw_register.py, by relation name). The counts match the operators' where I
had them (Taipei's five lines, 三鶯線 12, 安坑輕軌 9, 機場捷運 22, 臺中綠線 18) and the
newest stops are in (淡水信義線's 廣慈/奉天宮, opened 2026-08-30; 高雄捷運紅線's 岡山
extension; 三鶯線, opened 2026). What is wrong:

- Relations for the Danhai 藍海線 list the whole service, including the 綠山線 stops it runs
  over; only stops on the 藍海線's own track are taken.
- 中和新蘆線's track includes a link between its two branches (台北橋-三重國小) that no
  train uses; sections are kept only between stops some relation lists one after the other.
- 小碧潭支線's and 新北投支線's named track stops short of the junction platform (七張,
  北投); a stop beyond a dead end of the track still counts.

Published lengths (`check_model.REGISTER["tw"]`):

- metro.taipei 路網簡介, https://www.metro.taipei/cp.aspx?n=CCF30033E6ED8008 (2026-09-30):
  文湖線 25.2, 淡水信義線 30.7 (with 新北投支線), 松山新店線 20.7 (with 小碧潭支線),
  中和新蘆線 29.4, 板南線 26.5.
- Branch lengths, zh.wikipedia 新北投支線 / 小碧潭支線: 1.2 and 1.9.
- zh.wikipedia 臺北捷運, 新北捷運, 桃園捷運機場線, 臺灣捷運, 高雄捷運 (2026-09-30) for the rest.
  桃園機場捷運 51.33 is the operating length to 老街溪 (53.09 counts the unopened 中壢
  extension). 高雄捷運紅線 29.76 = 28.30 + 岡山路竹延伸 phase 1 1.46.
- THSR: thsrc.com.tw says "about 350 km", 南港-左營. zh.wikipedia gives chainage 南港 3.2,
  台北 6.1 ... 左營 345.2, which cannot be station-to-station (南港 to 台北 is about 9 km of
  track), so it is not used as km_official.

data.taipei has travel and dwell times between adjacent Taipei stations, not distances.

## Alishan Forest Railway (林業及自然保育署)

zh.wikipedia 阿里山林業鐵路 (2026-09-30), station table: 嘉義 0, 北門 1.6, 鹿麻產 10.8,
竹崎 14.2, 樟腦寮 23.3, 獨立山 27.4, 梨園寮 31.4, 交力坪 34.9, 水社寮 40.5, 奮起湖 45.8,
多林 50.9, 十字路 55.3, 屏遮那 60.5, 第一分道 62.7, 二萬平 66.8, 神木 69.6, 阿里山 71.6;
祝山線 阿里山 0, 沼平 1.3, 十字分道 2.9, 對高岳 4.9, 祝山 6.25. Whole line reopened
2024-07-06.

- 木屐寮 has a station in OSM but no row in the table, so 阿里山線 has no km_official.
- 鹿麻產 and 屏遮那 have no OSM station; the line runs through them without a stop.
- OSM tags all of it usage=tourism, and joins its two halves (奮起湖-多林) only through a
  0.66 km tunnel named "阿里山森林鐵路登山本線 (已崩塌舊線)" (collapsed old line). It is
  taken as the line, being the only track there.

## Line colours (`colours/tw.csv`)

English Wikipedia's `Module:Adjacent stations/<system>` colour tables, which follow each
operator's own line colours: Taipei Metro, New Taipei Metro, Taoyuan Metro, Taichung MRT,
Kaohsiung Metro, Taiwan High Speed Rail. TRA has no per-line colours: Wikipedia gives every
TRA line the one corporate blue 090980, which is left out, as Korea's was.

## Station and line English names

TRA stations from `tra_stations.json`; everything else from OSM `name:en`. Line English
names are the operators' usual ones (TRA's as English Wikipedia titles them), set in
`tw_register.LINES`.
