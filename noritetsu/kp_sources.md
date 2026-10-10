# North Korea (kp): sources

Built 2026-10-08 (kp agent), on Anita's call the same day: "north korea: if it runs passenger
service to the best of our knowledge, we should build it." The whole network is built; a line
is drawn running where the best evidence there is says scheduled passenger trains run on it,
greyed (`suspended`) where nothing says so. The evidence and how certain it is, line by line,
is in "What runs" below.

## Build (2026-10-08)

    python extract.py --region kp --pbf data/raw/north-korea-latest.osm.pbf --station-areas
    python kp_register.py --clip          # after every extract: China's, Russia's and South Korea's track out
    python build_model.py --region kp --register kp_register:data/raw/kp
    python build_tiles.py --region kp
    python check_model.py --region kp

The .pbf (Geofabrik `asia/north-korea-latest`, 2026-10-08, 93 MB) is deleted;
`data/proc/kp` holds everything a rebuild needs. Build ~45 s, tiles ~12 s.

**Result**: 58 register lines, 4,430 km (en.wikipedia's network figure is about 5,200 km
including the freight-only industrial lines left out here): **36 running, 3,754 km; 22
greyed, 676 km.** Plus 9 OSM lines (97 km: the Pyongyang Metro's two lines, Pyongyang's
trams T1-T3 and the Kumsusan tram, Ch'ŏngjin's tram, the Wŏnsan-Kalma resort tram, Hamhŭng's
Sŏho line) and the OSM numbered trains as named trains. 797 stations, 589 with an English
name (OSM `name:en`).

**Check** (`check_model.py --region kp`, en.wikipedia's line tables as published length):
every running line within 5% except 평남선 0.95 (P'yŏngyang - Pot'onggang is P'yŏngŭi Line
track here), 금골선 0.92 (OSM's stations end at Taesin, 14 km short of Muhak), 만덕선 0.93
(10 km line), 홍의선 0.67 (OSM draws Chŏkchi - Tumangang twice, as 홍의선 and 두만강선; the
two together are the line). Greyed: 백무선 0.86 (OSM in two pieces), 삼지연선 0.95, 평부선
(판문 - 개성) 1.06. The metro fails against en.wikipedia's "approximately 12 / 10 km"
(Chŏllima 8.7, Hyŏksin 12.3 km on OSM's track, 8 stops each, as published); I trust the
track over the round figures.

### The recipe (`kp_register.py`)

Thailand's (th_register.py) on kr_register's engine. The line unit is the legal line as OSM
names it (평의선, 평부선, 평라선 ...), which is the Ministry of Railways' naming that the
South Korean literature and both Wikipedias use. 75% of main-line track is named; every line
has a route=railway relation (infra.pkl). Per way: name (NAME_LINE), else the line relation
(REL_LINE), else its neighbours (`propagate`). Lines with trains and no OSM name or relation
are made from the trains' route relations (ROUTE_LINE): 백마선 (Paengma Line, 418/419), 천성
탄광선 (335/336), 서창선 (723/724; it includes the 4.2 km 형봉선, as OSM has no station at
Ch'ŏlgisan to cut it at), and missing bits of 금골선 and 고원탄광선. Stations: every OSM rail
station within 250 m of the line's track; a branch's junction station; two unnamed OSM
stations of 723/724 named 서덕천 and 형봉 (NAME_STOPS, en.wikipedia's distances). EXTEND runs
three lines on to their junction station over other track: 덕현선 to 남신의주, 청년이천선 to
세포청년, 천성탄광선 to 수양.

Left out, not register track: freight and colliery lines with fewer than two stations in
OSM (강계선 지선, 문천항선, 금야선, 운하선, 천내선, 청남선), the inter-Korean links south of
Kaesŏng/Kamho (경의선, 동해북부선: no train since 2008), the Yangdŏk hot-spring resort track,
산음 sidings. `rules/kp.py`: every route=train is a named train; the Mangyŏngdae and
Taesŏngsan amusement-park monorails are skipped (fairground rides).

## What runs

Evidence classes, strongest first:

- **A. Documented now** (news, 2025-2026). P'yŏngŭi Line: Beijing / Dandong - Sinŭiju -
  P'yŏngyang resumed 12 March 2026, Beijing - P'yŏngyang four a week, Dandong - P'yŏngyang
  daily (China Daily, 10 March 2026). Tumangang - Khasan: 645/646 Mon, Wed, Fri (Korea Times /
  iz.ru, June 2025); P'yŏngyang - Moscow through cars twice a month. The Pyongyang Metro
  (visitors have ridden every station since 2014).
- **B. OSM numbered trains**: ~75 route=train relations (1/2 P'yŏngyang - Hyesan, 5/6, 7/8,
  9/10 ... 931/932) compiled from en.wikipedia's line articles, whose "Services" sections cite
  Kokubu Hayato, 将軍様の鉄道 (Shinchosha 2007) and "the 2002 timetable". So this is the
  early-2000s timetable. Not checkable from outside; nothing newer is open. A section runs
  where such a train's track lies along at least half of it (`cut_running`).
- **C. en.wikipedia Services with no OSM relation** (same sources): RUNNING in
  kp_register.py.
- **D. A sentence, no train numbers** (lowest): also RUNNING, flagged here.

| line | built km | state | evidence |
|---|---|---|---|
| 평라선 P'yŏngra | 781 | runs | B (1/2, 3/4, 7/8, 9/10, 11/12, 13/14 and ~20 more) |
| 만포선 Manp'o | 304 | runs | B (15-18, 19/20, 134-137 ...); Ji'an border tail: C, one passenger car on the daily freight, for DPRK citizens and ethnic Koreans from China only |
| 북부내륙선 Pukpu | 248 | runs | B (Express 3/4 via Manp'o); WP: "a number of passenger trains", lightly used |
| 평의선 P'yŏngŭi | 226 | runs | A + B |
| 함북선 Hambuk, Ch'ŏngjin - Onsŏng | 183 | runs | B (25-28, 113/114, 9/10 to Komusan) |
| 함북선 Hambuk, Mulgol - Rajin | 42 | runs | A/B (7/8 and the Khasan trains, Rajin - Hongŭi) |
| 함북선 (온성 - 물골) | 103 | **grey** | nothing names a train Onsŏng - Hongŭi; WP mentions unnumbered "Ch'ŏngjin - Rajin" trains but the P'yŏngra Line is their shorter way. Uncertain |
| 평덕선 P'yŏngdŏk | 189 | runs | B (231/232, 236-239, 302-305, 781/782) |
| 평부선 P'yŏngbu (to Kaesŏng) | 186 | runs | B (142-145, 222-224 ...) |
| 평부선 (판문 - 개성) | 11 | **grey** | trains end at Kaesŏng |
| 강원선 Kangwŏn | 144 | runs | B (13/14, 117/118, 64-67) |
| 백두산청년선 Paektusan Ch'ŏngnyŏn | 139 | runs | B (1/2, 101/102, 104-111) |
| 청년이천선 Ch'ŏngnyŏn Ich'ŏn | 141 | runs | B (104-111, with times) |
| 평북선 P'yŏngbuk | 120 | runs | B (115/116, 200/201) |
| 은률선 Ŭnnyul | 117 | runs | B (219/220, 244-247, 138-141) |
| 금강산청년선 Kŭmgangsan Ch'ŏngnyŏn (to Kŭmgangsan Ch'ŏngnyŏn) | 101 | runs | B, but weak: one OSM relation with no train number; WP says only "in regular use" to Kŭmgangsan Ch'ŏngnyŏn |
| 금강산청년선 (감호 - 금강산청년) | 14 | **grey** | no train |
| 황해청년선 Hwanghae Ch'ŏngnyŏn | 91 | runs | B (15-18, 68, 240-243, 383/384) |
| 신흥선 Sinhŭng | 90 | runs | C (880/881 Hamhŭng - Sinhŭng) + D for the 762 mm part beyond Sinhŭng ("passenger trains also run", no numbers) |
| 평남선 P'yŏngnam | 81 | runs | B (146-149, 225/230, 226-229, 374/375, 391/392) |
| 허천선 Hŏch'ŏn | 80 | runs | C (551/556, 925/926) |
| 금골선 Kŭmgol | 77 | runs | B (11/12) + C (513/516 Kŭmgol - Muhak, 913/914) |
| 장진선 Changjin | 59 | runs | D ("significant for passenger transport in the area"); 762 mm with a cable incline. Least certain of the running lines |
| 무산선 Musan | 58 | runs | B (9/10) + C (662/663, 668/669) |
| 덕성선 Tŏksŏng | 51 | runs | C (261/262, 866/867) |
| 백마선 Paengma | 39 | runs | B (418/419) |
| 덕현선 Tŏkhyŏn | 37 | runs | D (three commuter pairs Sinŭiju - Tŏkhyŏn, no numbers) |
| 개천선 Kaech'ŏn | 28 | runs | B (19/20, 124-127, 250-253) |
| 서해갑문선 Sŏhae Kammun | 26 | runs | B (361/362) |
| 룡강선 Ryonggang | 18 | runs | B (733/734) |
| 고원탄광선 Kowŏn Colliery | 17 | runs | B (710-713 Kowŏn - Changdong; no WP services section) |
| 장연선 Changyŏn | 17 | runs | B (138-141) |
| 세천선 Sech'ŏn | 14 | runs | D (commuter trains Hoeryŏng - Sech'ŏn); Sech'ŏn - Chungbong (5.8 km) unknown, counted with it |
| 서창선 Sŏch'ang (+ 형봉선) | 13 | runs | B (723/724) |
| 천성탄광선 Ch'ŏnsŏng Colliery | 12 | runs | B (335/336, also in WP's P'yŏngra list), though its own WP article says freight only |
| 만덕선 Mandŏk | 10 | runs | B (931/932) |
| 두만강선 / 홍의선 Tumangang / Hongŭi | 9 + 6 | runs | A (Khasan trains, 7/8) |
| 송도원선 Songdowŏn | 2 | runs | B (64-67 to Songdowŏn) |
| 백무선 Paengmu | 166 | **grey** | no services listed; parts flooded |
| 삼지연선 Samjiyŏn | 61 | **grey** | the new standard-gauge line (2017 on): no service described |
| 배천선, 강계선, 옹진선, 안주탄광선, 대건선, 부포선, 비날론선, 룡성선, 송림선, 대안선, 도지리선, 은산선, 직동탄광선, 고참탄광선, 보산선, 후산선, 정도선 | 360 | **grey** | no services (freight-only per WP for 비날론, 대안, 후산, 은산, 안주탄광, 고참탄광, 보산; 송림선's infobox says passenger, no trains named; 옹진선 has commuter trains West Haeju - Haeju only, 7 km of 43, left grey) |

Lines WP gives passenger trains that are **not built**, because OSM names no track for them:
명당선 (Myŏngdang, 702-705), 청년팔원선 (Kujang - Kusŏng 94 km, 795/796), 강덕선 (601/604,
608/609), 다사도선 (Ryongch'ŏn - Tasado, five commuter pairs), 수풍선 (2.5 km, commuter).
Next step: find their track among OSM's unnamed ways and add them through ROUTE_LINE-like
geometry (no OSM train relation exists to take it from).

## Borders

Where OSM's track crosses OSM's admin_level 2 boundary (the extract's boundary ways, and
Overpass for Russia's, whose ways the extract lacks):

| point | lon, lat | way | passenger? |
|---|---|---|---|
| xSinuijuDandong | 124.392329, 40.115133 | 288979790 (Friendship Bridge) | yes, K27/28 and Dandong - P'yŏngyang |
| eXKPRUTUMANGANG | 130.641271, 42.415217 | 42510186 / 1475326893 (Friendship Bridge on the Tumen) | yes, 645/646 Khasan - Tumangang, Moscow through cars |
| xManpoJian | 126.273205, 41.154877 | 839022580 | one car on the daily freight (WP Manp'o Line); counted |
| Namyang - Tumen | 129.849287, 42.949015 | 199109778 | freight only; not offered |
| Ch'ŏngsu - Shanghekou | 124.879328, 40.458402 | 838948375 | freight only; not offered |

The proposed borders.EXTRA rows and the neighbours' sides are in `handoff_notes/kp_build.md`.
Until they land, kp_register.BORDERS supplies the three points and the lines run to them.

## Survey (2026-10-08), kept

- Beijing / Dandong - Sinuiju - Pyongyang: international trains resumed 12 March 2026 after
  six years (China Daily, https://www.chinadaily.com.cn/a/202603/10/WS69b02766a310d6866eb3d0d1.html).
  OSM: route=train 3838986 "K27/28".
- Khasan - Tumangang: train 645/646, Mon, Wed, Fri (Korea Times / iz.ru, June 2025).
  Pyongyang - Moscow through cars twice a month since 17 June 2025. OSM: 2129653, 8259408.
- Pyongyang Metro: OSM 1141949/7734277 Hyŏksin, 1141950/7734278 Chŏllima.
- No GTFS, no open DPRK timetable. en.wikipedia "List of railway lines in North Korea" is
  gone (404); Category:Railway lines in North Korea has the 136 line articles read.

| source | gives | licence |
|---|---|---|
| OSM via Geofabrik `asia/north-korea-latest.osm.pbf` | track, names, ~70 line relations, ~80 numbered-train relations, 799 stations | ODbL |
| en.wikipedia line articles | stations in order with km; Services (Kokubu 2007, the 2002 timetable) | CC BY-SA |
| China Daily, Korea Times, iz.ru | the international services | (reading) |
