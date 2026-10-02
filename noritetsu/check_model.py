"""Check built lines against published lengths and station counts.

    python check_model.py --region jp
    python check_model.py --region jp --find 山手

A line model can be self-consistent and still wrong: sections can be sliced out of the wrong
part of a route, a straight-line fallback can bridge two ends of a prefecture, and the total
still adds up to something.  The only check that catches that is an outside number.  These are
operating lengths (営業キロ) as published by the operators, which is also what the Japanese
line-completion hobby counts, so a ratio far from 1.00 is a real defect and not a definition.
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# (name fragment, operator fragment, published km, published stations or None, note)
#
# The operator fragment is not optional decoration: 東西線 is three different lines in Tokyo,
# Kyoto and Sapporo, and 中央 matches the Mizuho Shinkansen service via 鹿児島中央.
#
# OSM DOES NOT SPLIT LINES THE WAY THE OPERATORS DO, and the Chuo Main Line is the example:
# officially one 424.6 km line Tokyo-Nagoya, in OSM two route_masters split at Shiojiri where
# JR East hands over to JR Central. Both are checked separately here rather than pretending
# either should total 424.6. The remaining 27.7 km is the old route via Tatsuno.
# Register lines, as 国土数値情報 N02 names them and as the operators publish their 営業キロ.
# N02 writes 山陰線 where the timetable writes 山陰本線, and the register is what it means.
REGISTER = {
    "jp": [
        ("山陰線", "西日本", 673.8, "Kyoto-Hatabu, JR West"),
        ("東海道線", "東海", 341.3, "Atami-Maibara, JR Central"),
        ("東海道線", "西日本", 143.6, "Maibara-Kobe, JR West"),
        ("山陽線", "西日本", 534.4, "Kobe-Moji"),
        ("奥羽線", "東日本", 484.5, "Fukushima-Aomori"),
        ("日豊線", "九州", 462.6, "Kokura-Kagoshima"),
        ("根室線", "北海道", 362.1, "after the 2024 Furano-Shintoku closure"),
        ("山手線", "東日本", 20.6, "Shinagawa-Shinjuku-Tabata, the register line"),
        ("御堂筋線", "", 24.5, "Osaka Metro"),
        ("丸ノ内線", "", 24.2, "Tokyo Metro; Honancho branch is its own register line"),
        ("大江戸線", "", 40.7, "Toei"),
    ],
    # BAV Schienennetz km-lines, by their register name, against the length Wikipedia gives
    # for the line. The chainage check below covers every line; these are the outside numbers.
    "ch": [
        ("St. Moritz - Tirano", "RhB", 60.7, "Bernina line"),
        ("Brig - Visp - Zermatt", "", 44.0, "MGB, ex-BVZ"),
        ("Montreux - Zweisimmen", "", 62.4, "MOB; Zweisimmen-Lenk is its own km-line"),
        ("Zermatt - Gornergrat", "", 9.3, "Gornergratbahn, rack"),
        ("Brig - Andermatt - Disentis", "", 96.9, "MGB, ex-FO, via the Furka base tunnel"),
    ],
    # Register lines are named after the OSM track (kr_register.py), Seoul's numbered lines
    # prefixed 서울 지하철. Figures and their sources are in data/raw/kr/SOURCES.md:
    #   YB23 영업거리   2023 철도통계연보, part 1 "2. 역수 및 영업거리" sheet 2 (Korail/SR)
    #                  or part 2 "2. 운영현황" sheet 1 (metros), as of 2023-12-31
    #   YB22 광역       2022 철도통계연보, part 3 "2. 운영현황" sheet 1
    #   KRIC 18        레일포털 표준데이터 노선정보 노선연장
    #   KRIC 1294      the sum of 레일포털 역사정보's station-to-station km
    # A legal line's extent is not always what the OSM track name covers; where they differ
    # the note says which way the ratio should be off, rather than the line being dropped.
    "kr": [
        ("경부선", "", 441.7, "서울-부산, YB23 영업거리"),
        ("경부고속선", "", 398.2, "YB23 영업거리; = 거리표 시흥연결선 종점-부산"),
        ("호남고속선", "", 183.8, "오송-광주송정, YB23 영업거리; 거리표 182.4"),
        ("수서평택고속선", "", 61.1, "수서-평택분기, YB23 영업거리"),
        ("호남선", "", 252.5, "대전조차장-목포, YB23 영업거리"),
        ("전라선", "", 180.4, "익산-여수엑스포, YB23 영업거리"),
        ("경전선", "", 277.7, "삼랑진-광주송정, YB23 영업거리; 보성-임성리 is OSM 목포보성선"),
        ("장항선", "", 152.8, "천안-익산, YB23 영업거리"),
        ("충북선", "", 115.0, "조치원-봉양, YB23 영업거리"),
        ("중앙선", "", 332.2, "청량리-모량, YB23 영업거리; before the 2024-12 도담-영천 "
                              "realignment, so the built line may differ"),
        ("영동선", "", 188.9, "영주-청량신호소, YB23 영업거리"),
        ("태백선", "", 104.1, "제천-백산, YB23 영업거리"),
        ("경북선", "", 115.0, "김천-영주, YB23 영업거리"),
        ("동해선", "", 188.9, "부산진-영덕, YB23 영업거리; 영덕-삼척 opened 2025-01, so "
                              "expect about 1.6 if OSM carries it as 동해선"),
        ("경춘선", "", 80.7, "망우-춘천, YB23 영업거리"),
        ("경원선", "", 94.3, "용산-백마고지, YB23 영업거리"),
        ("경의선", "", 56.0, "서울-도라산, YB23 영업거리"),
        ("경인선", "", 27.0, "구로-인천, YB23 영업거리"),
        ("경강선", "", 177.7, "성남-여주 57.0 + 원주-강릉 120.7, YB23 영업거리; OSM names "
                              "both halves 경강선"),
        ("중부내륙선", "", 56.9, "부발-충주, YB23 영업거리; 충주-문경 opened 2024-12, "
                                "so expect about 1.7 if OSM includes it"),
        ("서해선", "", 38.5, "대곡-원시, YB23 영업거리; OSM also names the 2024 "
                              "홍성-서화성 intercity line 서해선, which is not in the figure"),
        ("분당선", "", 52.9, "왕십리-수원, YB23 영업거리"),
        ("수인선", "", 38.8, "수원-인천 less the 한대앞-오이도 section shared with 안산선, "
                              "YB23 영업거리; 거리표 51.6 with it"),
        ("안산선", "", 26.0, "금정-오이도, YB23 영업거리"),
        ("과천선", "", 14.4, "금정-남태령, YB23 영업거리"),
        ("일산선", "", 19.2, "지축-대화, YB23 영업거리"),
        ("교외선", "", 31.8, "능곡-의정부, YB23 영업거리; passenger service back 2025-01"),
        ("대구선", "", 26.1, "가천-영천, YB23 영업거리"),
        ("광주선", "", 11.9, "광주선분기-광주, YB23 영업거리"),
        ("정선선", "", 45.9, "민둥산-구절리, YB23 영업거리; trains stop at 아우라지 (38.7)"),
        ("용산선", "", 7.0, "용산-가좌, YB23 영업거리; OSM may run it on to DMC (8.6)"),
        ("인천국제공항선", "", 63.8, "서울역-인천공항2터미널, YB22 광역; KRIC 18 63.8"),
        ("신분당선", "", 33.5, "신사-광교: 17.3 + 13.8 YB22 광역 + 2.4 신사-강남 (its "
                                "footnote); KRIC 18 33.5"),
        ("수도권광역급행철도에이선", "", 64.7,
         "운정중앙-서울역 32.3 (KRIC 1294) + 수서-동탄 32.4 (SR 거리표); the 서울역-수서 "
         "middle (opened 2026) is not in it"),
        ("서울 지하철 1호선", "", 7.8, "서울역-청량리, YB23 영업거리"),
        ("서울 지하철 2호선", "", 48.8, "the loop, KRIC 18; YB23 60.2 includes both 지선"),
        ("성수지선", "", 5.4, "성수-신설동, KRIC 18"),
        ("신정지선", "", 6.0, "신도림-까치산, KRIC 18"),
        ("서울 지하철 3호선", "", 39.1, "지축-오금, YB23 영업거리; KRIC 18 38.2"),
        ("서울 지하철 4호선", "", 31.1, "불암산-남태령, KRIC 18; YB23 32.8 incl. 0.5 of depot"),
        ("진접선", "", 14.9, "불암산-진접, KRIC 18 14.892"),
        ("서울 지하철 5호선", "", 45.3, "방화-상일동, YB23 영업거리 stages (14.5+16.7+14.1)"),
        ("마천지선", "", 7.0, "강동-마천, YB23 영업거리"),
        ("하남선", "", 7.5, "상일동-하남검단산, YB23 영업거리 stages (4.6+2.9)"),
        ("서울 지하철 6호선", "", 36.4, "응암-신내, YB23 영업거리"),
        ("서울 지하철 7호선", "", 60.9, "장암-온수 46.9 서울교통공사 + 온수-석남 13.98 "
                                     "인천교통공사, YB23 영업거리"),
        ("서울 지하철 8호선", "", 17.7, "암사-모란, YB23 영업거리; 별내선 is its own track"),
        ("서울 지하철 9호선", "", 40.6, "개화-중앙보훈병원, YB23 영업거리"),
        ("우이신설선", "", 11.0, "YB23 영업거리"),
        ("신림선", "", 7.53, "YB23 영업거리; KRIC 18 7.76"),
        ("김포 골드라인", "", 23.5, "YB23 영업거리"),
        ("용인경전철", "", 18.1, "에버라인, YB23 영업거리"),
        ("의정부경전철", "", 10.6, "YB23 영업거리"),
        ("부산 도시철도 1호선", "", 39.9, "YB23 영업거리"),
        ("부산 도시철도 2호선", "", 45.2, "YB23 영업거리"),
        ("부산 도시철도 3호선", "", 18.1, "YB23 영업거리"),
        ("부산 도시철도 4호선", "", 12.0, "YB23 영업거리"),
        ("부산김해경전철", "", 23.2, "YB23 영업거리; KRIC 18 22.4"),
        ("대구 도시철도 1호선", "", 28.4, "설화명곡-안심, YB23 영업거리; 안심-하양 is its "
                                        "own track"),
        ("대구 도시철도 2호선", "", 31.4, "YB23 영업거리"),
        ("대구 도시철도 3호선", "", 23.1, "YB23 영업거리"),
        ("인천 도시철도 1호선", "", 30.3, "계양-송도달빛축제공원, YB23 영업거리; the 2025-06 "
                                        "검단 extension makes it about 37 (KRIC 1294)"),
        ("인천 도시철도 2호선", "", 29.2, "YB23 영업거리"),
        ("광주 도시철도 1호선", "", 20.5, "YB23 영업거리"),
        ("대전 도시철도 1호선", "", 20.5, "YB23 영업거리"),
    ],
    # Register lines are named as tw_register.LINES has them. Sources are in tw_sources.md:
    #   WP-TRA   zh.wikipedia 臺灣鐵路路線列表 (retrieved 2026-09-30), each line's 營業里程
    #   TRA-M    TRA's own 鐵路里程 file (data.gov.tw 6999), the km between the line's ends;
    #            it is also the km_official every TRA section is checked against
    #   TRTC     metro.taipei 路網簡介 (2026-09-30), whose trunk figures include the branch
    #   WP       zh.wikipedia pages of each system (新北捷運, 桃園捷運機場線, 臺中捷運,
    #            高雄捷運, 阿里山林業鐵路), retrieved 2026-09-30
    # A published metro length usually runs to the ends of the tail tracks; the build runs
    # station centre to station centre, so metros may read a few percent short.
    "tw": [
        ("縱貫線北段", "", 125.4, "基隆-竹南, WP-TRA; TRA-M 125.3"),
        ("縱貫線南段", "", 188.9, "彰化-高雄, WP-TRA; TRA-M 189.0"),
        ("臺中線", "", 85.5, "竹南-彰化 by 臺中 (山線), WP-TRA; TRA-M 85.6"),
        ("海岸線", "", 90.2, "竹南-彰化 by 大甲 (海線), WP-TRA; TRA-M 90.3"),
        ("成追線", "", 2.2, "成功-追分, WP-TRA; TRA-M 2.2"),
        ("屏東線", "", 61.3, "高雄-枋寮, WP-TRA; TRA-M 61.2"),
        ("南迴線", "", 98.2, "枋寮-臺東, WP-TRA; TRA-M 98.1"),
        ("宜蘭線", "", 93.6, "八堵-蘇澳, WP-TRA; TRA-M 93.5"),
        ("北迴線", "", 79.2, "蘇澳新-花蓮, WP-TRA; TRA-M 79.5"),
        ("臺東線", "", 150.9, "花蓮-臺東, WP-TRA; TRA-M 150.9"),
        ("平溪線", "", 12.9, "三貂嶺-菁桐, WP-TRA; TRA-M 12.9"),
        ("深澳線", "", 4.7, "瑞芳-八斗子, TRA-M; WP-TRA's 6.0 runs on past 八斗子 to the "
                           "freight end"),
        ("內灣線", "", 27.9, "新竹-內灣, WP-TRA; TRA-M 北新竹-內灣 26.5 + 新竹-北新竹 1.4"),
        ("六家線", "", 3.1, "竹中-六家, WP-TRA; TRA-M 3.1"),
        ("集集線", "", 29.7, "二水-車埕, WP-TRA; TRA-M 29.6"),
        ("沙崙線", "", 5.7, "中洲-沙崙, TRA-M; WP-TRA says 5.3"),
        ("台灣高速鐵路", "", 350.0, "南港-左營, THSR's own 'about 350 km' (thsrc.com.tw)"),
        ("阿里山線", "", 72.9, "嘉義-阿里山 71.6 + 阿里山-沼平 1.3, WP 阿里山林業鐵路"),
        ("祝山線", "", 4.95, "沼平-祝山: WP's 阿里山-祝山 6.25 less 阿里山-沼平 1.3"),
        ("淡水信義線", "", 29.5, "淡水-廣慈/奉天宮: TRTC 30.7 with the branch, less the "
                             "branch's 1.2 (WP)"),
        ("新北投支線", "", 1.2, "北投-新北投, WP"),
        ("松山新店線", "", 18.8, "松山-新店: TRTC 20.7 with the branch, less the branch's "
                             "1.9 (WP)"),
        ("小碧潭支線", "", 1.9, "七張-小碧潭, WP; OSM's branch track stops about 0.2 km "
                             "short of 七張's platform, so expect about 0.9 of it"),
        ("中和新蘆線", "", 29.4, "TRTC, both branches"),
        ("板南線", "", 26.5, "頂埔-南港展覽館, TRTC"),
        ("文湖線", "", 25.2, "動物園-南港展覽館, TRTC"),
        ("環狀線", "", 15.4, "大坪林-新北產業園區, WP 臺北捷運"),
        ("三鶯線", "", 14.29, "頂埔-鶯桃福德, WP 新北捷運; opened 2026"),
        ("安坑輕軌", "", 7.67, "雙城-十四張, WP 新北捷運; OSM's own track is only about 7.5 "
                           "a direction, so the figure likely counts tail and depot track"),
        ("淡海輕軌綠山線", "", 7.34, "紅樹林-崁頂, WP 新北捷運"),
        ("淡海輕軌藍海線", "", 2.21, "濱海沙崙-淡水漁人碼頭, WP 新北捷運 (phase 1)"),
        ("桃園機場捷運", "", 51.33, "台北車站-老街溪, WP 營運長度 (53.09 with the unopened "
                               "中壢 extension)"),
        ("臺中捷運綠線", "", 16.71, "北屯總站-高鐵臺中站, WP 臺灣捷運"),
        ("高雄捷運紅線", "", 29.76, "小港-岡山車站: WP 28.30 + 岡山路竹延伸 phase 1 1.46"),
        ("高雄捷運橘線", "", 14.4, "哈瑪星-大寮, WP 高雄捷運; OSM's own track is only about "
                               "13.4 a direction, so the figure likely counts tail track"),
        ("高雄環狀輕軌", "", 22.1, "the loop, WP 高雄捷運"),
    ],
    # Register lines are named as hk_register.LINES has them. Sources are in hk_sources.md:
    #   HyD   Highways Department, Shatin to Central Link page (retrieved 2026-09-30)
    #   WP    the "Line length" in each line's en.wikipedia infobox (not "Track length"),
    #         retrieved 2026-09-30; WP-MTR is the line table in en.wikipedia "MTR"
    # The build measures station centre to station centre; where a figure runs to the ends
    # of the track instead, the note gives OSM's own end-to-end track length for comparison.
    "hk": [
        ("東鐵綫", "", 46.0, "金鐘-羅湖/落馬洲, HyD 'approximately 46km'; WP: 紅磡-羅湖 34 + "
                           "落馬洲支綫 7.4; 馬場 (race days only) is not in MTR's list"),
        ("屯馬綫", "", 56.193, "屯門-烏溪沙, WP"),
        ("觀塘綫", "", 17.32, "黃埔-調景嶺, WP"),
        ("荃灣綫", "", 15.59, "中環-荃灣, WP"),
        ("港島綫", "", 14.93, "堅尼地城-柴灣, WP (track length 16.3)"),
        ("南港島綫", "", 7.4, "金鐘-海怡半島, WP; OSM's track end to end is 7.05, so the figure "
                            "counts the tail tracks"),
        ("東涌綫", "", 31.1, "香港-東涌, WP"),
        ("機場快綫", "", 35.2, "香港-博覽館, WP; WP-MTR 35.3"),
        ("將軍澳綫", "", 12.3, "北角-寶琳 and 康城, WP"),
        ("迪士尼綫", "", 3.3, "欣澳-迪士尼, WP-MTR (its own page says 3.5 and 3.8); OSM's track "
                            "end to end is 3.51, past both platforms"),
        ("輕鐵", "", 36.2, "the network, WP-MTR; the build counts one-way loops and both "
                        "streets of a one-way pair, about 2.5 km, see hk_sources.md"),
        ("香港電車", "", 15.9, "堅尼地城-筲箕灣 13.3 + 跑馬地 loop 2.6, WP Hong Kong Tramways"),
        ("山頂纜車", "", 1.364, "花園道-山頂 along the slope, WP Peak Tram; it climbs 369 m, "
                            "so about 1.31 km on the map; OSM's track is 1.26"),
    ],
    # Register lines as sg_register.LINES names them (LTA's names). Sources in sg_sources.md:
    #   LTA   lta.gov.sg rail_network/<line>.html "Length of rail" (2026-09-30), whole km
    #   WP    en.wikipedia line articles (2026-09-30)
    # A published length runs to the ends of the running track, tails and depot leads
    # included; the build runs station centre to station centre. "track" below is the line's
    # OSM running track (no yard, siding or crossover) halved for double track, 2026-09-30.
    "sg": [
        ("North-South Line", "", 45.0, "LTA, 27 stations; track 46.5 (0.7 of it past Marina "
                                      "South Pier)"),
        ("East-West Line", "", 57.0, "LTA 'approximately 57km' with the Changi Airport branch, "
                                    "35 stations; track 56.4"),
        ("North East Line", "", 21.6, "20 + 1.6 for the 2024 Punggol Coast extension (WP); "
                                     "LTA rounds it to 22; 17 stations; track 21.4"),
        ("Circle Line", "", 39.0, "LTA, with Stage 6 (opened 2026-07-12), 33 stations; "
                                 "track 39.4"),
        ("Downtown Line", "", 42.0, "LTA (WP 41.9), 35 stations; track 41.4, of which 1.5 is "
                                   "the lead from Bukit Panjang to Gali Batu depot, so expect "
                                   "about 0.95"),
        ("Thomson-East Coast Line", "", 40.6, "LTA, Woodlands North-Bayshore, 27 stations; "
                                             "track 40.9"),
        ("Bukit Panjang LRT", "", 8.0, "LTA, 13 stations; counts the Ten Mile Junction spur "
                                      "closed 2019-01; track 7.7, so expect about 0.95"),
        ("Sengkang LRT", "", 10.7, "WP, 14 stations; track 10.1 (halving undercounts any single track)"),
        ("Punggol LRT", "", 10.3, "WP (a 2003 design paper), 15 stations; track 9.3 and the "
                                 "relations' own ways 9.7, so the figure counts track no "
                                 "service runs over, perhaps the depot link"),
        ("Sentosa Express", "", 2.1, "WP, 4 stations; track 1.99, all of it built"),
    ],
    # Register lines as rinf.py names them (Infrabel's "L.36"). Sources in be_sources.md:
    #   WP    the LENGTE field of each line's nl.wikipedia infobox (retrieved 2026-09-30). It
    #         is uncited but agrees with the chainage in the article's own route diagram,
    #         except where a note says otherwise. Infrabel's network statement (annex D.1)
    #         lists every line but gives no lengths.
    #   RINF  the line's own section lengths in RINF, which check_model also compares
    #         every line against (km_official)
    # Lines where the article counts a closed or foreign stretch are left out here and
    # listed in be_sources.md; the RINF comparison still covers them.
    "be": [
        ("L.0", "", 3.8, "Brussel-Noord - Brussel-Zuid, WP; built station centre to centre"),
        ("L.1", "", 73.0, "HSL 1 Halle - Esplechin-Frontière, WP; RINF runs it from Y.Noord "
                         "Halle, 76.1, so expect about 1.04"),
        ("L.2", "", 64.7, "HSL 2 Leuven - Ans, WP; RINF 66.7"),
        ("L.3", "", 36.1, "HSL 3 Chênée - Y.Hammerbrücke, WP"),
        ("L.12", "", 32.1, "Antwerpen-Centraal - Essen-Grens: the Belgian part, by WP's route "
                          "diagram (the infobox's 64.4 runs on to Lage Zwaluwe)"),
        ("L.13", "", 6.6, "Kontich-Lint - Lier, WP"),
        ("L.15", "", 89.0, "Y.Drabstraat - Y.Zonhoven, WP"),
        ("L.16", "", 26.4, "Y.Nazareth - Aarschot, WP; its own route diagram says 22.1 and "
                           "RINF 24.5, so expect about 0.93"),
        ("L.19", "", 31.7, "Mol - Hamont-Grens, WP"),
        ("L.21", "", 28.6, "Landen - Hasselt, WP"),
        ("L.25", "", 47.6, "Brussel-Noord - Antwerpen-Luchtbal, WP; RINF 47.9"),
        ("L.26", "", 28.6, "Schaarbeek - Halle, WP; RINF's line starts at Y.Rue de Bruel "
                           "(24.8), so expect about 0.85"),
        ("L.27", "", 44.8, "Brussel-Noord - Antwerpen-Centraal, WP"),
        ("L.34", "", 54.8, "Hasselt - Liège-Guillemins, WP"),
        ("L.35", "", 53.8, "Leuven - Hasselt, WP"),
        ("L.36", "", 99.9, "Brussel-Noord - Liège-Guillemins, WP"),
        ("L.36C", "", 5.3, "Y.Zaventem - Y.Machelen-Noord (Brussels Airport), WP"),
        ("L.36N", "", 28.8, "Brussel-Noord - Leuven, WP; RINF also files the Leuven curves "
                            "under it, 29.8 in all"),
        ("L.37", "", 47.0, "Liège-Guillemins - Hergenrath-Frontière, WP"),
        ("L.42", "", 59.8, "Rivage - Gouvy, WP"),
        ("L.43", "", 62.0, "Angleur - Marloie, WP"),
        ("L.50", "", 55.6, "Brussel-Noord - Gent-Sint-Pieters, WP"),
        ("L.50A", "", 114.3, "Brussel-Zuid - Oostende, WP"),
        ("L.51", "", 14.9, "Brugge - Blankenberge, WP"),
        ("L.51A", "", 9.8, "Y.Blauwe Toren - Zeebrugge, WP"),
        ("L.51B", "", 13.9, "Y.Dudzele - Knokke, WP"),
        ("L.53", "", 64.1, "Schellebelle - Leuven, WP"),
        ("L.59", "", 55.8, "Y.Oost-Berchem - Gent-Dampoort, WP"),
        ("L.60", "", 27.5, "Jette - Dendermonde, WP"),
        ("L.66", "", 54.5, "Brugge - Kortrijk, WP; RINF's own sum is 58.8 because it gives "
                           "Torhout-Zedelgem 13.9 km for about 8 km of track"),
        ("L.69", "", 48.9, "Y.Kortrijk-West - Abele, WP; beyond Poperinge it is closed and "
                           "RINF stops there (42.0), so expect about 0.86"),
        ("L.73", "", 76.6, "Deinze - De Panne, WP infobox; its own route diagram says 70.5 "
                           "and RINF 72.5 to the buffer stop, so expect about 0.92"),
        ("L.75", "", 54.6, "Gent-Sint-Pieters - Mouscron-Frontière, WP; RINF 56.9"),
        ("L.75A", "", 15.6, "Mouscron - Froyennes, WP"),
        ("L.78", "", 39.0, "Saint-Ghislain - Tournai, WP"),
        ("L.89", "", 61.2, "Denderleeuw - Y.Zandberg, WP"),
        ("L.94", "", 77.8, "Halle - Blandain-Frontière, WP; RINF 80.6"),
        ("L.96", "", 84.7, "Brussel-Zuid - Quévy, WP infobox; its own route diagram says "
                           "74.9 and RINF 76.0, so expect about 0.89"),
        ("L.97", "", 20.2, "Mons - Quiévrain, WP; RINF 19.2"),
        ("L.112", "", 14.1, "Marchienne-au-Pont - La Louvière-Centre, WP infobox; its own "
                            "route diagram says 20.9 and RINF 20.2, so expect about 1.43"),
        ("L.116", "", 8.0, "Manage - Y.La Paix, WP"),
        ("L.117", "", 26.9, "Braine-le-Comte - Luttre, WP"),
        ("L.118", "", 18.8, "Y.Saint-Vaast - Mons, WP"),
        ("L.122", "", 28.5, "Y.Melle - Geraardsbergen, WP"),
        ("L.124", "", 55.9, "Brussel-Zuid - Charleroi-Central, WP"),
        ("L.125", "", 59.5, "Liège-Guillemins - Namur, WP"),
        ("L.125A", "", 11.0, "Y.Val-Benoît - Flémalle-Haute, WP"),
        ("L.130", "", 36.6, "Namur - Charleroi-Central, WP"),
        ("L.130A", "", 29.3, "Charleroi-Central - Erquelinnes, WP"),
        ("L.134", "", 5.5, "Mariembourg - Couvin, WP"),
        ("L.139", "", 29.0, "Leuven - Ottignies, WP"),
        ("L.140", "", 36.0, "Ottignies - Marcinelle, WP"),
        ("L.144", "", 14.4, "Gembloux - Jemeppe-sur-Sambre, WP"),
        ("L.161", "", 62.0, "Schaerbeek - Namur, WP"),
        ("L.161D", "", 4.4, "Y.Louvain-la-Neuve - Louvain-la-Neuve, WP; to the end of the "
                            "track, the build to the station, so expect about 0.9"),
        ("L.162", "", 146.8, "Namur - Sterpenich-Frontière, WP"),
        ("L.165", "", 81.9, "Libramont - Athus, WP"),
        ("L.166", "", 70.2, "Y.Neffe - Bertrix, WP"),
        ("L.167", "", 11.6, "Autelbas - Athus-Frontière, WP; RINF 11.7, but the last 1.4 km "
                            "into Athus fails its trace (be_sources.md), so expect about 0.86"),
    ],
    # Register lines as rinf.py names them (Wikidata's label for ÖBB's route number, else
    # "first - last"). Figures are Wikidata's P2043 length for the item carrying that route
    # number, which de.wikipedia's infobox supplies (retrieved 2026-09-30). Most Austrian
    # articles describe the historic railway (Südbahn, Ostbahn, Franz-Josefs-Bahn) rather
    # than ÖBB's route, so only lines whose article covers the same stretch are listed; the
    # comparison with RINF's own lengths covers every line. ÖBB-Infrastruktur's route list and
    # route descriptions (at_sources.md) give no lengths. Sources in at_sources.md.
    "at": [
        ("Salzburg-Tiroler-Bahn", "", 191.73, "101 03 Salzburg - Wörgl, WD; RINF's own is 179.8 "
                                              "(0.94), so the gap is between the two figures"),
        ("Pyhrnbahn", "", 104.2, "204 01 Linz - Selzthal, WD; RINF's own is 99.0 (0.95), so "
                                 "the gap is between the two figures"),
        ("Bahnstrecke Wels Hbf–Passau Hbf", "", 83.0, "205 01, WD to Passau Hbf; ÖBB's route "
                                                      "ends at the border, RINF 79.9"),
        ("Steirische Ostbahn", "", 80.235, "414 01 Graz - Szentgotthárd border, WD; RINF's own "
                                          "is 75.6 (0.94), so the gap is between the figures"),
        ("Summerauer Bahn", "", 67.656, "221 01 Linz - Summerau border, WD; RINF 65.2"),
        ("Innkreisbahn", "", 60.0, "207 01 Neumarkt-Kallham - Braunau, WD"),
        ("Mühlkreisbahn", "", 57.784, "258 01 Linz Urfahr - Aigen-Schlägl, WD"),
        ("Almtalbahn", "", 43.0, "252 01 Wels - Grünau im Almtal, WD"),
        ("Neue Unterinntalbahn", "", 40.236, "330 01 Kundl/Radfeld - Baumkirchen, WD; RINF 38.5, "
                                             "so expect about 0.93"),
        ("Radkersburger Bahn", "", 31.095, "462 01 Spielfeld-Straß - Bad Radkersburg, WD"),
        ("Bahnstrecke Herzogenburg–Krems", "", 20.308, "173 01, WD"),
        ("Bahnstrecke Gänserndorf–Marchegg", "", 18.17, "115 01, WD"),
        ("Bahnstrecke Absdorf-Hippersdorf–Stockerau", "", 17.08, "113 01, WD"),
        ("Übelbacherbahn", "Steiermärkische", 10.247,
         "Peggau - Übelbach (Lokalbahn Peggau-Übelbach), WD; the first 1.1 km from the ÖBB "
         "handover at Peggau is lost because that RINF point has no coordinate, so expect "
         "about 0.87"),
        ("Schruns - Bludenz Moos", "Montafonerbahn", 12.874,
         "Montafonerbahn Bludenz - Schruns, WD; the 1.2 km from the ÖBB handover at Bludenz "
         "is lost (no coordinate), so expect about 0.84"),
    ],
    # Register lines as rinf.py names them from ProRail's id ("Asd-Rtd" -> "Amsterdam -
    # Rotterdam"). Figures are the LENGTE in the infobox of the nl.wikipedia article
    # "Spoorlijn A - B" for the same two ends (retrieved 2026-09-30). ProRail's RINF line often
    # starts at a junction ("aansl.") a little outside the town the article starts at, which
    # the notes give where it matters. Sources in nl_sources.md.
    "nl": [
        ("Harlingen - Nieuweschans Grens", "", 127.6, "Harlingen - Nieuwe Schans, WP"),
        ("Amsterdam - Zutphen", "", 105.7, "WP; RINF 108.4"),
        ("Elst - Dordrecht", "", 92.8, "WP"),
        ("Roosendaal - Vlissingen", "", 74.4, "WP; RINF 75.0"),
        ("Maastricht - Venlo", "", 70.0, "WP"),
        ("Tilburg - Nijmegen", "", 65.9, "WP"),
        ("Breda - Eindhoven", "", 58.9, "WP; RINF 60.5"),
        ("Utrecht - Rotterdam", "", 55.9, "WP gives 52.3 / 55.9; the second, RINF 55.6"),
        ("Venlo - Eindhoven", "", 51.6, "WP"),
        ("Winterswijk - Zevenaar", "", 49.6, "WP"),
        ("Zaandam - Enkhuizen", "", 49.8, "WP"),
        ("Groningen - Delfzijl", "", 37.9, "WP"),
        ("Woerden - Leiden", "", 32.5, "WP; RINF 34.0"),
        ("Sauwerd - Roodeschool", "", 26.9, "WP"),
        ("Roosendaal - Breda", "", 22.7, "WP; RINF 23.4"),
        ("Stadskanaal Hoofdstation - Zuidbroek", "", 22.3, "Stadskanaal - Zuidbroek, WP"),
        ("Dieren - Apeldoorn", "", 21.7, "WP"),
        ("Haarlem - Uitgeest", "", 18.0, "WP; RINF 18.9"),
        ("Mariënberg - Almelo", "", 18.8, "WP"),
        ("Gouda - Alphen aan den Rijn", "", 17.6, "WP"),
        ("Breukelen - Harmelen aansl.", "", 8.3, "Harmelen - Breukelen, WP"),
        ("Haarlem - Zandvoort aan Zee", "", 8.2, "Haarlem - Zandvoort, WP"),
        ("Heerlen - Schin op Geul", "", 8.0, "WP"),
        ("Amsterdam - Rotterdam", "", 85.3, "the Oude Lijn, WP; ProRail files Amsterdam - "
                                           "Haarlem under Ass-Rtd (11.3 km), so RINF's Asd-Rtd "
                                           "is 69.8 and expect about 0.81"),
        ("Meppel - Groningen", "", 76.9, "WP; RINF's line is 70.8 from Meppel aansl., so "
                                         "expect about 0.92"),
        ("Utrecht - Boxtel", "", 60.3, "WP; RINF's own is 55.6, so expect about 0.92"),
        ("Deventer - Almelo", "", 38.5, "WP; RINF's line starts at Snippeling aansl. east of "
                                        "Deventer (35.8), so expect about 0.93"),
        ("Eindhoven - Weert", "", 30.0, "WP; RINF's line starts at Tongelre aansl. east of "
                                        "Eindhoven (27.0), so expect about 0.90"),
        ("Apeldoorn - Deventer", "", 14.8, "WP; RINF's line starts at Apeldoorn aansl. (13.9), "
                                           "so expect about 0.93"),
        ("Leeuwarden - Stavoren", "", 50.2, "WP; RINF's line starts at the Harinxmakanaal "
                                            "bridge south of Leeuwarden (47.9), so expect 0.95"),
        ("Gouda - Den Haag", "", 25.3, "WP; RINF's own is 23.9, so expect about 0.94"),
        ("Breda - Rotterdam", "", 49.4, "WP; RINF's own Bd-Rtd is 59.0, so expect about 1.17; "
                                        "what the extra 10 km is has not been worked out"),
    ],
    # Register lines as rinf.py names them (OSM's route=railway relation for IP's line number,
    # "Linha do Norte"). Figures are the line articles' infobox or text lengths on pt.wikipedia
    # (PT) or en.wikipedia (EN), retrieved 2026-09-30; Norte's and Tomar's cite IP's (then
    # REFER's) network statement, the rest cite nothing or old books. IP's current network
    # statement is not linked from its site in a form that could be fetched. RINF's own lengths
    # check every line besides. Sources in pt_sources.md.
    "pt": [
        ("Linha do Norte", "", 336.0, "Lisboa-Santa Apolónia - Porto-Campanhã, EN citing the "
                                      "Directório da Rede 2022 p.71"),
        ("Linha da Beira Baixa", "", 240.0, "Entroncamento - Guarda, PT"),
        ("Linha da Beira Alta", "", 202.0, "Pampilhosa - Vilar Formoso border, PT and EN"),
        ("Linha do Oeste", "", 197.9, "Agualva-Cacém - Figueira da Foz, PT article text (its "
                                      "infobox says 215.1, the line before the Meleças "
                                      "realignment); RINF 197.3, less Bifurcação de Lares - "
                                      "Amieira (2.7 km), which no OSM passenger route runs "
                                      "over, so expect about 0.98"),
        ("Linha do Douro", "", 160.0, "Ermesinde - Pocinho, PT and EN, 'cerca de'; RINF 163.1"),
        ("Linha do Leste", "", 140.692, "Abrantes - Spanish border, PT (its section table: "
                                        "64.404 + 65.573 + 10.715)"),
        ("Linha do Algarve", "", 139.5, "Lagos - Vila Real de Santo António, PT and EN"),
        ("Linha do Minho", "", 133.6, "Porto-São Bento - Valença border, EN; PT says 134"),
        ("Linha de Guimarães", "", 30.1, "Lousado - Guimarães, PT's line table"),
        ("Linha de Sintra", "", 27.2, "Lisboa-Rossio - Sintra, PT and EN"),
        ("Linha de Évora", "", 26.2, "Casa Branca - Évora, the open part (PT's line table: "
                                     "'26,2 km de 85,1 km')"),
        ("Linha de Cascais", "", 25.4, "Cais do Sodré - Cascais, EN"),
        ("Ramal de Braga", "", 15.0, "Nine - Braga, PT"),
        ("Ramal de Tomar", "", 14.8, "Lamarosa - Tomar, PT citing the Directório da Rede 2012 "
                                     "p.70"),
        ("Linha de Cintura", "", 10.5, "Alcântara-Terra - Braço de Prata, PT; the build adds "
                                       "RINF's 1.0 km link on to Alcântara-Mar, so expect "
                                       "about 1.05"),
        ("Ramal de Alfarelos", "", 16.5, "Alfarelos - Bifurcação de Lares, PT; RINF's own is "
                                         "14.7, so expect about 0.89"),
        ("Linha de Leixões", "", 18.7, "Contumil - Porto de Leixões, PT; the last 4.6 km from "
                                       "Guifões into the port is freight, dropped as unridden, "
                                       "so expect about 0.75"),
    ],
    # Register lines as rinf.py names them from MÁV's line number ("30-as vasútvonal").
    # Figures are the "Hossz" in the infobox of each line's hu.wikipedia article, as Wikidata
    # carries them in P2043 (retrieved 2026-09-30; lines 1 and 30 read off the articles too).
    # RINF's own length is no check here: MÁV's section lengths leave out the track inside
    # stations and read about 15% short (GYSEV's are full), see hu_sources.md. Only lines whose
    # article covers about the same stretch as MÁV's number are listed.
    "hu": [
        ("100-as vasútvonal", "", 337.4, "Budapest-Nyugati - Záhony, WP"),
        ("80-as vasútvonal", "", 266.2, "Budapest-Keleti - Sátoraljaújhely, WP"),
        ("20-as vasútvonal", "", 169.0, "Székesfehérvár - Szombathely, WP; GYSEV since 2025"),
        ("140-es vasútvonal", "", 118.0, "Cegléd - Szeged, WP"),
        ("29-es vasútvonal", "", 117.0, "Börgönd - Szabadbattyán - Tapolca, WP"),
        ("60-as vasútvonal", "", 115.276, "Murakeresztúr - Szentlőrinc, WP"),
        ("108-as vasútvonal", "", 103.0, "Debrecen - Füzesabony, WP"),
        ("17-es vasútvonal", "", 102.0, "Szombathely - Nagykanizsa, WP"),
        ("41-es vasútvonal", "", 101.0, "Dombóvár - Gyékényes, WP"),
        ("25-ös vasútvonal", "", 101.0, "Boba - Zalaegerszeg - Bajánsenye, WP"),
        ("35-ös vasútvonal", "", 100.0, "Kaposvár - Siófok, WP"),
        ("154-es vasútvonal", "", 96.0, "Bátaszék - Baja - Kiskunhalas, WP"),
        ("16-os vasútvonal", "", 94.0, "Hegyeshalom - Porpác, WP"),
        ("5-ös vasútvonal", "", 82.0, "Székesfehérvár - Komárom, WP"),
        ("147-es vasútvonal", "", 79.0, "Kiskunfélegyháza - Szentes - Orosháza, WP"),
        ("42-es vasútvonal", "", 79.0, "Pusztaszabolcs - Paks, WP"),
        ("102-es vasútvonal", "", 74.0, "Kál-Kápolna - Kisújszállás, WP"),
        ("10-es vasútvonal", "", 72.0, "Győr - Celldömölk, WP"),
        ("75-ös vasútvonal", "", 70.0, "Vác - Balassagyarmat, WP"),
        ("70-es vasútvonal", "", 62.9, "Budapest-Nyugati - Szob, WP"),
        ("116-os vasútvonal", "", 58.2, "Nyíregyháza - Vásárosnamény, WP"),
        ("111-es vasútvonal", "", 57.0, "Mátészalka - Záhony, WP"),
        ("23-as vasútvonal", "", 49.0, "Zalaegerszeg - Rédics, WP"),
        ("14-es vasútvonal", "", 37.0, "Pápa - Csorna, WP; GYSEV since 2025"),
        ("47-es vasútvonal", "", 18.7, "Godisa - Komló, WP"),
        ("12-es vasútvonal", "", 15.0, "Tatabánya - Oroszlány, WP; needs rinf's tol_abs 1.0 "
                                       "for Tatabánya - Bánhida"),
        ("1-es vasútvonal", "", 191.0, "Budapest-Déli - Hegyeshalom - Rajka, WP; MÁV's line 1 "
                                       "starts at Keleti via Ferencváros (+12.8 km) and runs to "
                                       "the Nickelsdorf border (+4.7), Rajka is line 1d (13.4), "
                                       "so the near match is partly luck"),
        ("30-as vasútvonal", "", 221.0, "Budapest-Déli - Murakeresztúr, WP; the article's own "
                                        "station table has Székesfehérvár at 66.9 (built 66.7), "
                                        "and 221 is where the build reaches Nagykanizsa, so the "
                                        "infobox looks to stop there: expect about 1.05"),
        ("8-as vasútvonal", "", 85.0, "Győr - Sopron, WP; the numbered line (GYSEV's) runs on "
                                      "5.4 km to the Austrian border past Sopron, so expect "
                                      "about 1.05"),
        ("120a-s vasútvonal", "", 100.0, "Budapest-Keleti - Újszász - Szolnok, WP; MÁV's line "
                                         "starts at Rákos, about 8 km out on line 100, so "
                                         "expect about 0.92"),
    ],
    # Register lines as rinf.py names them from PLK's number ("Linia kolejowa nr 1"). Figures
    # are the "długość" in each pl.wikipedia line article's infobox (retrieved 2026-09-30),
    # which the articles cite to PLK's own line list, Id-12 (D-29) "Wykaz linii" (editions
    # 2020-2026). RINF's own section lengths agree with Id-12 to within 1% on these lines
    # except where noted. Only lines with passenger service over their whole length are
    # listed; lines closed or being rebuilt in part (29, 97, 104, 201, 229, 309, 356...) build
    # only their open part, as they should. Sources in pl_sources.md.
    "pl": [
        ("Linia kolejowa nr 1", "", 316.066, "Warszawa Zachodnia - Katowice, Id-12 via WP"),
        ("Linia kolejowa nr 2", "", 214.227, "Warszawa Zachodnia - Terespol, Id-12 via WP"),
        ("Linia kolejowa nr 3", "", 475.583, "Warszawa Zachodnia - Kunowice, Id-12 via WP"),
        ("Linia kolejowa nr 4", "", 223.824, "Grodzisk Mazowiecki - Zawiercie (CMK), Id-12 via WP"),
        ("Linia kolejowa nr 6", "", 224.163, "Zielonka - Kuźnica Białostocka, Id-12 via WP; RINF's "
                                            "own is 218.2 (0.97), so the gap is between the two "
                                            "figures"),
        ("Linia kolejowa nr 7", "", 267.471, "Warszawa Wschodnia Osobowa - Dorohusk, Id-12 via WP"),
        ("Linia kolejowa nr 8", "", 317.164, "Warszawa Zachodnia - Kraków Główny, Id-12 via WP"),
        ("Linia kolejowa nr 9", "", 323.393, "Warszawa Wschodnia Osobowa - Gdańsk Główny, Id-12 "
                                            "via WP"),
        ("Linia kolejowa nr 14", "", 388.578, "Łódź Kaliska - Forst (Lausitz), Id-12 via WP"),
        ("Linia kolejowa nr 16", "", 71.027, "Łódź Widzew - Kutno, Id-12 via WP"),
        ("Linia kolejowa nr 18", "", 247.418, "Kutno - Piła Główna, Id-12 via WP"),
        ("Linia kolejowa nr 38", "", 241.534, "Białystok - Głomno (border), Id-12 via WP; RINF's own "
                                             "is 202.0, the stretch past Bartoszyce to the border "
                                             "is not in it, so expect about 0.83"),
        ("Linia kolejowa nr 61", "", 177.300, "Kielce - Fosowskie, Id-12 via WP"),
        ("Linia kolejowa nr 68", "", 177.512, "Lublin Główny - Przeworsk, Id-12 via WP"),
        ("Linia kolejowa nr 91", "", 258.974, "Kraków Główny - Medyka, Id-12 via WP"),
        ("Linia kolejowa nr 96", "", 145.916, "Tarnów - Leluchów, Id-12 via WP"),
        ("Linia kolejowa nr 101", "", 82.469, "Munina - Hrebenne, Id-12 via WP"),
        ("Linia kolejowa nr 131", "", 493.472, "Chorzów Batory - Tczew (coal trunk line), Id-12 "
                                              "via WP"),
        ("Linia kolejowa nr 137", "", 284.120, "Katowice - Legnica, Id-12 via WP"),
        ("Linia kolejowa nr 139", "", 113.695, "Katowice - Zwardoń, Id-12 via WP"),
        ("Linia kolejowa nr 202", "", 334.363, "Gdańsk Główny - Stargard, Id-12 via WP"),
        ("Linia kolejowa nr 203", "", 342.890, "Tczew - Kostrzyn, Id-12 via WP"),
        ("Linia kolejowa nr 207", "", 133.671, "Toruń Wschodni - Malbork, Id-12 via WP"),
        ("Linia kolejowa nr 271", "", 164.212, "Wrocław Główny - Poznań Główny, Id-12 via WP"),
        ("Linia kolejowa nr 273", "", 356.125, "Wrocław Główny - Szczecin Główny, Id-12 via WP"),
        ("Linia kolejowa nr 289", "", 38.869, "Legnica - Rudna Gwizdanów, Id-12 via WP"),
        ("Linia kolejowa nr 351", "", 213.500, "Poznań Główny - Szczecin Główny, Id-12 via WP"),
        ("Linia kolejowa nr 354", "", 93.025, "Poznań Główny POD - Piła Główna, Id-12 via WP"),
        ("Linia kolejowa nr 401", "", 100.716, "Szczecin Dąbie - Świnoujście, Id-12 via WP"),
        ("Linia kolejowa nr 405", "", 193.419, "Piła Główna - Ustka, Id-12 via WP"),
    ],
    # Register lines as rinf.py names them for Czechia: the timetable (KJŘ) number and ends from
    # OSM's route=tracks relation, "190 Plzeň – České Budějovice" (rinf_countries/cz.py). Figures
    # are Wikidata's P2043 on the item with that route number (P1671), which is the infobox
    # "délka" of the line's cs.wikipedia article; 040, 190, 200 and 113 checked against the
    # article itself (retrieved 2026-09-30). Only lines whose article covers the same stretch
    # as the timetable number; the comparison with RINF's own lengths covers every line.
    # Sources in cz_sources.md.
    "cz": [
        ("040 Trutnov – Chlumec nad Cidlinou", "", 101.9, "WP, checked"),
        ("086 Liberec – Česká Lípa", "", 59.0, "WD"),
        ("126 Most – Rakovník", "", 70.0, "WD"),
        ("137 Chomutov – Vejprty", "", 57.904, "WD, to the border"),
        ("145 Sokolov – Kraslice – Zwotental", "PDV", 27.452, "WD Sokolov - Klingenthal, "
                                                               "Czech part; PDV Railway"),
        ("160 Plzeň – Žatec", "", 107.0, "WD"),
        ("161 Rakovník – Bečov nad Teplou", "", 87.962, "WD"),
        ("183 Plzeň – Klatovy – Železná Ruda", "", 97.352, "WD"),
        ("190 Plzeň – České Budějovice", "", 136.0, "WP infobox; its km table gives 135.7 "
                                                    "(349.094 - 213.388) and RINF 133.1, so "
                                                    "expect 0.98"),
        ("198 Strakonice – Volary", "", 70.783, "WD"),
        ("200 Zdice – Protivín", "", 101.9, "WP, checked: km 101.911 to 0.022"),
        ("202 Tábor – Bechyně", "", 24.091, "WD"),
        ("203 Březnice – Strakonice", "", 49.117, "WD"),
        ("224 Tábor – Horní Cerekev", "", 69.414, "WD"),
        ("226 Veselí nad Lužnicí – Gmünd NÖ", "", 54.9, "WD České Velenice - Veselí"),
        ("227 Kostelec u Jihlavy – Slavonice", "", 53.5, "WD"),
        ("235 Kutná Hora – Zruč nad Sázavou", "", 35.865, "WD"),
        ("243 Moravské Budějovice – Jemnice", "", 20.775, "WD"),
        ("250 Havlíčkův Brod – Brno – Břeclav – Kúty", "", 191.0, "WD Havlíčkův Brod - Kúty; "
                                                                  "RINF 196.0"),
        ("252 Křižanov – Studenec", "", 33.832, "WD"),
        ("255 Hodonín – Zaječí", "", 37.492, "WD"),
        ("262 Chornice – Skalice nad Svitavou", "", 68.953, "WD Třebovice - Skalice"),
        ("300 Brno – Přerov", "", 90.1, "WD"),
        ("303 Kojetín – Valašské Meziříčí", "", 61.0, "WD"),
        ("346 Újezdec u Luhačovic – Luhačovice", "", 9.632, "WD"),
        ("113 Čížkovice – Obrnice", "AŽD", 34.817, "WP infobox now; Wikidata still 36.884 and "
                                                   "RINF 36.9, so expect 1.05"),
        ("199 České Budějovice – České Velenice – Gmünd NÖ", "", 52.0,
         "WD to Gmünd; RINF stops at the border (48.8), so expect 0.93"),
        ("180 Plzeň – Furth im Wald", "", 81.164,
         "WD to Furth im Wald; RINF stops at the border (73.5), so expect 0.90"),
        ("228 Jindřichův Hradec – Obrataň", "JHMD", 45.996,
         "WD; RINF 44.1, and the 1.3 km into Jindřichův Hradec is dropped (no OSM route; "
         "cz_sources.md), so expect 0.91"),
        ("229 Jindřichův Hradec – Nová Bystřice", "JHMD", 32.869,
         "WD; RINF 30.4, and the 3.6 km into Jindřichův Hradec is dropped (no OSM route), "
         "so expect 0.81"),
    ],
    # Register lines as rinf.py names them for Slovakia: the timetable (KCP) number and route
    # from OSM's route=railway relation, "120 Bratislava – Žilina" (rinf_countries/sk.py).
    # Figures are Wikidata's P2043 on the item with that route number (P1671), which is the
    # infobox "dĺžka" of the line's sk.wikipedia article; 120, 160, 170 and 180 checked against
    # the articles themselves (retrieved 2026-10-01). Only lines whose article covers the
    # same stretch as the built line; RINF's own lengths are no check here, since ŽSR's leave
    # out station track and read about 15% short (sk_sources.md). Sources in sk_sources.md.
    "sk": [
        ("113 Zohor – Záhorská Ves", "", 14.511, "WD; no passenger trains"),
        ("116 Kúty – Trnava", "", 67.46, "WD"),
        ("117 Jablonica – Brezová pod Bradlom", "", 11.669, "WD; no passenger trains"),
        ("120 Bratislava – Žilina", "", 203.0, "WP, checked; RINF ends 120 at Žilina "
                                               "predmestie, short of Žilina, so expect 0.97"),
        ("124 Trenčianska Teplá – Lednické Rovne", "", 17.271, "WD; no passenger trains"),
        ("126 Žilina – Rajec", "", 21.285, "WD"),
        ("127 Žilina – Mosty u Jablunkova", "", 37.3, "WD; the built line stops at the border"),
        ("128 Čadca – Makov", "", 26.172, "WD"),
        ("129 Čadca – Zwardoń", "", 21.0, "WD, to Zwardoń; built to the border, so expect "
                                                "about 0.95"),
        ("133 Galanta – Leopoldov", "", 43.878, "WD Galanta - Leopoldov 29.64 plus Trnava - "
                                                "Sereď 14.238, both timetable 133"),
        ("134 Šaľa – Neded", "", 18.939, "WD; no passenger trains"),
        ("140 Nové Zámky – Prievidza", "", 111.6, "WD; Nové Zámky - Šurany is built on 151, "
                                                  "which shares it, so expect 0.92"),
        ("143 Trenčín – Chynorany", "", 49.0, "WP infobox, checked; Wikidata has 52"),
        ("144 Prievidza – Nitrianske Pravno", "", 11.1, "WD; no passenger trains"),
        ("145 Prievidza – Horná Štubňa", "", 37.0, "WD"),
        ("152 Štúrovo – Levice", "", 52.0, "WD"),
        ("153 Zvolen – Čata", "", 106.0, "WD"),
        ("154 Hronská Dúbrava – Banská Štiavnica", "", 19.717, "WD"),
        ("160 Zvolen – Košice", "", 233.0, "WP, checked"),
        ("162 Lučenec – Utekáč", "", 41.194, "WD"),
        ("163 Katarínska Huta – Breznička", "", 9.96, "WD; no passenger trains"),
        ("165 Plešivec – Muráň", "", 40.926, "WD; no passenger trains"),
        ("166 Plešivec – Slavošovce", "", 24.0, "WD; no passenger trains"),
        ("167 Rožňava – Dobšiná", "", 26.06, "WD; no passenger trains"),
        ("168 Moldava nad Bodvou – Medzev", "", 15.35, "WD; no passenger trains"),
        ("170 Zvolen – Vrútky", "", 96.0, "WP, checked"),
        ("173 Červená Skala – Margecany", "", 92.579, "WD"),
        ("174 Brezno – Jesenské", "", 82.0, "WD; Brezno - Brezno-Halny is built on 172, "
                                            "which shares it, so expect about 0.95"),
        ("180 Žilina – Košice", "", 238.88, "WP, checked"),
        ("181 Kraľovany – Trstená", "", 56.45, "WD"),
        ("182 Štrbské Pleso – Štrba", "", 4.75, "WD, the rack railway"),
        ("183 Poprad-Tatry – Starý Smokovec – Štrbské Pleso", "", 29.1, "WD, Tatra electric "
                                                                         "railway"),
        ("184 Starý Smokovec – Tatranská Lomnica", "", 5.9, "WD, Tatra electric railway"),
        ("186 Spišská Nová Ves – Levoča", "", 12.656, "WD; no passenger trains"),
        ("187 Spišské Vlachy – Spišské Podhradie", "", 9.296, "WD; no passenger trains"),
        ("192 Trebišov – Vranov nad Topľou", "", 31.9, "WD; no passenger trains"),
        ("193 Prešov – Humenné", "", 60.437, "WD"),
    ],
    # RFN lines as fr_register names them (Wikidata's label for the six-digit code). Figures are
    # the "longueur" in each line's fr.wikipedia infobox (retrieved 2026-09-30), which the
    # articles source to SNCF Réseau; "RFN" is the line's own exploited PK extent in
    # lignes-par-statut. Sources in fr_sources.md. These are lines wholly inside
    # Ile-de-France, where the development extract reaches; the build runs station to station,
    # so a line that runs on past its last stop to a junction reads a little short.
    "fr": [
        ("Ligne des Invalides à Versailles-Rive-Gauche", "", 17.61, "977 000, RER C"),
        ("Ligne de Paris-Saint-Lazare à Versailles-Rive-Droite", "", 22.089,
         "973 000; the RFN's own PK extent is 22.8 (0.051 to Versailles), so WP's figure is "
         "likely 22.9 mistyped"),
        ("Ligne de Saint-Cloud à Saint-Nom-la-Bretèche - Forêt-de-Marly", "", 15.2, "974 000"),
        ("Ligne de Paris-Saint-Lazare à Ermont - Eaubonne", "", 14.888,
         "334 900; RFN extent 14.1, running 0.4 km past Ermont-Eaubonne, the last stop"),
        ("Ligne d'Ermont - Eaubonne à Valmondois", "", 15.0, "328 000"),
        ("Ligne de Montsoult - Maffliers à Luzarches", "", 11.116, "315 000; RFN extent 10.7"),
        ("Ligne de la bifurcation de Neuville à Cergy-Préfecture", "", 12.0,
         "326 000; RFN extent 10.8 (PK 28.249-39.092), which the build matches"),
        ("Ligne d'Esbly à Crécy-la-Chapelle", "", 9.945, "071 000, tram-train"),
        ("Ligne de Roissy", "", 15.0, "076 000, RER B; RFN extent 15.3"),
        ("Ligne de Choisy-le-Roi à Massy - Verrières", "", 16.25,
         "985 000; Wikidata has 15.037"),
        ("Ligne de Grigny à Corbeil-Essonnes", "", 10.7, "988 000"),
        ("Ligne de Corbeil-Essonnes à Montereau", "", 60.906, "746 000"),
        ("Ligne d'Achères à Pontoise", "", 15.0, "338 000"),
        ("Ligne de Paris-Saint-Lazare à Mantes-Station par Conflans-Sainte-Honorine", "",
         58.0, "334 000"),
        ("Ligne de Plaisir - Grignon à Épône - Mézières", "", 20.0,
         "396 000; RFN extent 18.5 (PK 33.510-52.036), which the build matches"),
        ("Ligne de Villeneuve-Saint-Georges à la bifurcation de Moisenay", "", 39.0,
         "752 100, LGV, no stations"),
        # Outside Île-de-France, added with the first whole-country build.
        ("Ligne de Paris-Lyon à Marseille-Saint-Charles", "", 862.0, "830 000, WP"),
        ("Ligne de Paris-Montparnasse à Brest", "", 622.4, "420 000, WP"),
        ("Ligne de Paris-Nord à Lille", "", 251.0, "272 000, WP"),
    ],
    # Register lines as cn_register names them: China Railway's own line names, which are
    # what OSM names the track (京沪线, 京沪高铁, 沪昆高速线). Figures are the length in each
    # line's zh.wikipedia infobox (system_length, else length_in_operation), retrieved
    # 2026-09-30; the articles cite China Railway and the NRA. A 铁路 article and a CR 线 can
    # differ in extent; where they do the note says so. The build runs station centre to
    # station centre over OSM track. Sources and faults in cn_sources.md.
    "cn": [
        ("京沪高铁", "", 1318.0, "北京南-上海虹桥, WP 京沪高速铁路"),
        ("沪昆高速线", "", 2266.0, "上海虹桥-昆明南, WP 沪昆高速铁路"),
        ("徐兰高速线", "", 1395.0, "徐州东-兰州西, WP 徐兰高速铁路"),
        ("兰新客专线", "", 1786.0, "兰州西-乌鲁木齐, WP 兰新客运专线"),
        ("京广高速线", "", 2118.0, "北京西-广州南, WP 京广高速铁路 linelength (its 2,291 "
                              "length_in_operation is the fare distance); OSM's line starts "
                              "at 北京丰台"),
        ("京哈高速线", "", 1240.0, "北京朝阳-哈尔滨西, WP 京哈高速铁路"),
        ("郑渝高速线", "", 1065.7, "郑州东-重庆北 (north route), WP 郑渝高速铁路"),
        ("合福高速线", "", 850.0, "合肥-福州, WP 合福高速铁路"),
        ("贵广客专线", "", 857.0, "贵阳东-广州南, WP 贵广客运专线"),
        ("南昆客专线", "", 715.8, "南宁-昆明南, WP 南昆客运专线"),
        ("西成客专线", "", 683.0, "西安北-成都东, WP 西成客运专线; the built line starts at "
                              "西安西, so 西安北-西安西 is not in it"),
        ("银西高速线", "", 543.0, "西安北-银川 over its own track, WP 银西高速铁路 "
                              "length_in_operation"),
        ("沪宁城际线", "", 301.0, "南京-上海, WP 沪宁城际铁路"),
        ("京津城际线", "", 165.0, "北京南-天津 and the 天津-滨海 extension, WP 京津城际铁路 "
                              "citing CR's 里程表; the built line has both"),
        ("海南东环高速线", "", 308.0, "海口-三亚, WP 海南东环铁路"),
        ("京沪线", "", 1451.4, "北京-上海, WP 京沪铁路"),
        ("京广线", "", 2269.3, "北京西-广州, WP 京广铁路; OSM's line starts at 房山东"),
        ("京九线", "", 2311.0, "北京西-深圳, WP 京九铁路; OSM's line ends at 北京大兴 and "
                             "东莞东"),
        ("京哈线", "", 1249.0, "北京-哈尔滨, WP 京哈铁路"),
        ("陇海线", "", 1759.0, "连云港-兰州, WP 陇海铁路"),
        ("兰新线", "", 2413.0, "兰州-乌鲁木齐-阿拉山口, WP 兰新铁路"),
        ("焦柳线", "", 1639.0, "焦作 (月山)-柳州, WP 焦柳铁路"),
        ("包兰线", "", 990.0, "包头-兰州, WP 包兰铁路"),
        ("滨洲线", "", 935.0, "哈尔滨-满洲里, WP 滨洲铁路"),
        ("湘桂线", "", 1013.0, "衡阳-凭祥, WP 湘桂铁路; the built line starts at 祁东北, so "
                             "衡阳-祁东北 is not in it"),
        ("沈大线", "", 399.0, "沈阳-大连, WP 沈大铁路"),
        ("广深线", "", 147.0, "广州-深圳, WP 廣深鐵路"),
        ("宝成线", "", 676.0, "宝鸡-成都, WP 宝成铁路; OSM's 宝成线 track stops at 广汉北, "
                            "so 广汉北-成都 is missing: expect about 0.90"),
        ("青藏线", "", 1971.0, "西宁-拉萨, WP 青藏铁路; the built line starts at 湟源, so "
                             "西宁-湟源 (about 50 km) is not in it: expect about 0.93"),
        ("成昆线", "", 1100.0, "成都-昆明 operating length, WP 成昆铁路; the built line is two "
                             "pieces, 成都南-花棚子 and 元谋西-昆明, whose 61 km crow-fly gap "
                             "cn_register.join_pieces leaves open: expect about 0.91"),
    ],
    # Register lines as rinf.py names them from CFL's line number and name ("Ligne 1
    # Luxembourg – Troisvierges-frontière", rinf_countries/lu.py). Figures are the "Distance"
    # on each line's sheet in annex 2A of CFL's network statement, Document de Référence du
    # Réseau 2026 (data/raw/lu_drr_2026_fr.pdf, from acf.gouvernement.lu). Left out: 4 (DRR
    # 16.2; only Luxembourg - Berchem, 7.4 km, carries passenger routes, the Berchem -
    # Oetrange freight bypass is dropped), 6e (DRR 2.7 runs on to Audun-le-Tiche station in
    # France; built to the border, 1.5, which is RINF's 1.53), and the freight lines 2b
    # (Ettelbruck - Bissen), 6d (Tétange - Langengrund) and 6k (Brucherberg - Scheuerbusch),
    # which build nothing. DRR counts 6g and 6h from Pétange (4.1, 5.2), but RINF files the
    # shared Pétange - Rodange (2.6) under 6j only, so those two are checked from Rodange,
    # DRR less its 2.6. lu_sources.md.
    "lu": [
        ("Ligne 1 Luxembourg – Troisvierges-frontière", "", 76.8, "DRR 2026"),
        ("Ligne 1a Ettelbruck – Diekirch", "", 4.1, "DRR 2026"),
        ("Ligne 1b Kautenbach – Wiltz", "", 9.0, "DRR 2026 (RINF 9.10)"),
        ("Ligne 3 Luxembourg – Wasserbillig-frontière", "", 37.4, "DRR 2026, less the 3.9 km "
                                                                  "Mertert-Port freight branch"),
        ("Ligne 5 Luxembourg – Kleinbettingen-frontière", "", 18.8, "DRR 2026"),
        ("Ligne 6 Luxembourg – Bettembourg-frontière", "", 16.6, "DRR 2026"),
        ("Ligne 6a Bettembourg – Esch/Alzette", "", 9.5, "DRR 2026"),
        ("Ligne 6b Bettembourg – Dudelange-Usines (Volmerange)", "", 7.0, "DRR 2026, to "
                                                                         "Volmerange-les-Mines"),
        ("Ligne 6c Noertzange – Rumelange", "", 5.9, "DRR 2026"),
        ("Ligne 6f Esch/Alzette – Pétange", "", 15.7, "DRR 2026"),
        ("Ligne 6g Pétange – Rodange-frontière (Aubange)", "", 1.5, "DRR 4.1 less Pétange - "
                                                                    "Rodange 2.6; RINF 1.48. The "
                                                                    "OSM station sits 135 m "
                                                                    "east of RINF's Rodange "
                                                                    "point: expect about 1.07"),
        ("Ligne 6h Pétange – Rodange-frontière (Mont St. Martin)", "", 2.6, "DRR 5.2 less "
                                                                           "Pétange - Rodange "
                                                                           "2.6; RINF 2.57"),
        ("Ligne 6j Pétange – Rodange-frontière (Athus)", "", 4.1, "DRR 2026, Pétange - border "
                                                                  "via Rodange"),
        ("Ligne 7 Luxembourg – Pétange", "", 20.4, "DRR 2026"),
    ],
    # Register lines as rinf.py names them from SŽ's line number ("Proga 50 Ljubljana–Sežana",
    # rinf_countries/si.py). Figures are the "dolžina" in the infobox of each line's
    # sl.wikipedia article (retrieved 2026-10-01), whose "oznaka" chainage agrees with them
    # (Ljubljana–Sežana 565.9-682.5); Prvačina–Ajdovščina cites SŽ-Infrastruktura's network
    # statement, Program omrežja 2025. Lines 40 and 44 have one article between them
    # (Pragersko–Središče, 51.9; built 40.2 + 11.5) and are not listed. Sources in si_sources.md.
    "si": [
        ("Proga 70 Jesenice–Sežana", "", 129.8, "WP; RINF's own is 124.8 (Plave - Solkan 6.7 "
                                               "for 10.7 km of track)"),
        ("Proga 80 Ljubljana–Metlika", "", 124.4, "WP, Ljubljana - Metlika d. m.; the last "
                                                 "0.6 km to the border has no OSM route"),
        ("Proga 50 Ljubljana–Sežana", "", 116.59, "WP, Ljubljana - Sežana d. m.; needs "
                                                 "extract.py's construction-track rule at "
                                                 "Preserje"),
        ("Proga 10 Ljubljana–Dobova", "", 114.75, "WP, Ljubljana - Dobova d. m."),
        ("Proga 30 Zidani Most–Šentilj", "", 108.27, "WP, to the border; Šentilj - border "
                                                     "(2.3 km) has no OSM route and is dropped, "
                                                     "so expect about 0.97"),
        ("Proga 34 Maribor–Prevalje", "", 82.1, "WP, Maribor - Holmec border"),
        ("Proga 20 Ljubljana–Jesenice", "", 70.36, "WP, Ljubljana - Jesenice d. m."),
        ("Proga 41 Ormož–Hodoš", "", 69.22, "WP, Ormož - Hodoš d. m."),
        ("Proga 82 Grosuplje–Kočevje", "", 49.1, "WP; RINF's own is 44.2 (Velike Lašče - "
                                                "Ortnek 3.1 for 7.0 km of track)"),
        ("Proga 31 Celje–Velenje", "", 37.97, "WP"),
        ("Proga 32 Grobelno–Rogatec", "", 36.50, "WP, Grobelno - Rogatec d. m."),
        ("Proga 62 Prešnica–Koper", "", 31.5, "WP, to Koper's port; Koper - Koper tovorna "
                                             "(the freight yard, 2.6 km) is left out, so expect "
                                             "about 0.92"),
        ("Proga 81 Sevnica–Trebnje", "", 31.35, "WP"),
        ("Proga 64 Pivka–Ilirska Bistrica", "", 24.41, "WP, Pivka - Ilirska Bistrica d. m."),
        ("Proga 21 Ljubljana Šiška–Kamnik Graben", "", 23.01, "WP"),
        ("Proga 72 Prvačina–Ajdovščina", "", 14.83, "WP citing Program omrežja 2025"),
        ("Proga 61 Prešnica–Podgorje", "", 14.72, "WP 'Divača–Podgorje–d. m.', the same "
                                                 "Prešnica - Rakitovec border stretch"),
        ("Proga 33 Stranje–Imeno", "", 14.24, "WP, Stranje - Imeno d. m.; Imeno - border "
                                              "(RINF 1.5 km) is rejected (traced 0.35), so "
                                              "expect about 0.90"),
        ("Proga 43 Lendava–d. m.", "", 5.2, "WP; RINF's own is 4.6, so expect about 0.88"),
    ],
    # Register lines as rinf.py names them for Bulgaria, NRIC's number: "Железопътна линия 2"
    # (rinf_countries/bg.py). NRIC's network statement 2026-2027 gives no per-line lengths, so
    # figures are bg.wikipedia's (retrieved 2026-10-01): the infobox "дължина" of the line's
    # article (WP-I), or the length its text gives (WP-T); 4 is Wikidata's P2043 (WD). Mostly
    # uncited; RINF's own section lengths agree with them within 2% except where noted, and
    # check_model compares every line with RINF as well. Freight lines (11, 12, 15, 51), the
    # razed 86 and lines with no single clean figure (3, 7, 24) are left out. bg_sources.md.
    "bg": [
        ("Железопътна линия 1", "", 355.0, "WP-T, Kalotina Zapad - Svilengrad. Built also has "
                                           "Svilengrad - Turkish border (18.9) and - Greek border "
                                           "(3.9), which OSM's Istanbul and Pythio trains ride, "
                                           "and Sofia - Poduyane (3.2); less Ihtiman - Verinsko "
                                           "(8.4, rebuilt, construction/razed in OSM): ~1.05"),
        ("Железопътна линия 2", "", 543.563, "WP-I, Sofia - Varna; RINF 543.6"),
        ("Железопътна линия 4", "", 400.0, "WD, Ruse - Stara Zagora + Mihaylovo - Podkova (it runs "
                                           "on line 8 between); RINF 401.0 for that. Built also "
                                           "has the Ruse pieces 4A2-4A4 and Ruse Razp. - Giurgiu "
                                           "border (~17 km) and Ivanovo - Dve Mogili traced 15.6 "
                                           "for RINF's 9.7 (14.2 crow-fly halt to halt): ~1.06, RINF's own "
                                           "comparison 1.00"),
        ("Железопътна линия 5", "", 207.670, "WP-T, Sofia - Kulata; RINF 209.1 to Kulata"),
        ("Железопътна линия 6", "", 136.0, "WP-T, Voluyak - Gyueshevo (Pernik - Radomir on "
                                           "line 5); RINF 134.3"),
        ("Железопътна линия 8", "", 292.492, "WP-T, Plovdiv - Burgas; RINF also files Plovdiv - "
                                             "Filipovo (5.7) under 8"),
        ("Железопътна линия 9", "", 142.0, "WP-I, Ruse - Kaspichan; RINF's 137.4 starts at Ruse "
                                           "Razpredelitelna, so expect about 0.97"),
        ("Железопътна линия 13", "", 10.9, "WP-I, Voluyak - Bankya"),
        ("Железопътна линия 19", "", 10.0, "WP-I, Krumovo - Asenovgrad"),
        ("Железопътна линия 23", "", 43.005, "WP-I, Yasen - Cherkvitsa"),
        ("Железопътна линия 26", "", 50.344, "WP-I, Shumen - Komunari"),
        ("Железопътна линия 28", "", 111.014, "WP-I, Povelyanovo - Razdelna - Romanian border; "
                                              "Kardam - border (5.1) has no OSM path and Razdelna "
                                              "- RP Razdelna (1.7) is an unridden junction "
                                              "section, so expect about 0.95"),
        ("Железопътна линия 42", "", 17.0, "WP-I, Tsareva Livada - Gabrovo"),
        ("Железопътна линия 52", "", 9.575, "WP-I, General Todorov - Petrich (RINF's 51B); RINF's "
                                            "own is 8.9, so expect about 0.94"),
        ("Железопътна линия 81", "", 71.111, "WP-I, Filipovo - Panagyurishte"),
        ("Железопътна линия 82", "", 60.0, "WP-I, Filipovo - Karlovo"),
        ("Железопътна линия 83", "", 61.170, "WP-I, Simeonovgrad - Nova Zagora"),
        ("Железопътна линия 91", "", 113.2, "WP-I, Samuil - Silistra"),
    ],
    # Register lines as rinf_countries/gr.py names them (OSE's ids grouped and named by their
    # ends). OSE's network statement gives no per-line lengths; figures are en.wikipedia's
    # (EN) and el.wikipedia's (EL) line articles, retrieved 2026-10-01, mostly uncited.
    # check_model compares every line with RINF's own section lengths as well. Left out: 09
    # (the old Tithorea - Lianokladi line, greyed) and 26 (Strymonas - Promachonas), which have
    # no published figure, and 20 (Thessaloniki - Idomeni), whose article copies another
    # line's 21.69. gr_sources.md.
    "gr": [
        ("Πειραιάς – Θεσσαλονίκη", "", 494.21, "EN Piraeus–Platy railway 456.60 + RINF's Platy - "
                                               "Thessaloniki 37.61 (lines 01 and 22)"),
        ("Θεσσαλονίκη – Αλεξανδρούπολη", "", 440.8, "EL infobox; the built line starts at the TX1 "
                                                    "junction, 5.6 km out of Thessaloniki on "
                                                    "line 01's track, and Kirki - port traces "
                                                    "0.6 under RINF: expect ~0.97"),
        ("Αλεξανδρούπολη – Ορμένιο", "", 178.5, "EN Alexandroupoli–Svilengrad railway, which "
                                                "takes in ~3.9 km in Bulgaria; built has Pythio "
                                                "- Turkish border (1.35, line 28): ~1.01"),
        ("Πλατύ – Φλώρινα", "", 151.2, "EN Thessaloniki–Bitola railway chainage, Florina 187.5 "
                                       "less Platy 36.3; RINF's own is 156.6, so expect ~1.04"),
        ("Αεροδρόμιο – Κιάτο", "", 135.0, "EN Athens Airport–Patras railway, Airport - SKA 30 + "
                                          "SKA - Kiato 105; built also has the Kato Acharnai - "
                                          "Zefiri and Kato Acharnai - Miden links and the Ano "
                                          "Liosia stub (line 06), 7.7 km: ~1.06"),
        ("Παλαιοφάρσαλος – Καλαμπάκα", "", 80.44, "EN Piraeus–Platy railway, branches"),
        ("Λάρισα – Βόλος", "", 60.76, "EN Piraeus–Platy railway, branches"),
        ("Λειανοκλάδι – Στυλίδα", "", 22.61, "EN Piraeus–Platy railway, branches"),
        ("Οινόη – Χαλκίδα", "", 21.69, "EN Piraeus–Platy railway, branches"),
    ],
    # fi.wikipedia line articles' infobox `pituus` (WP), retrieved 2026-10-01, and VR's
    # distances from Helsinki in fi.wikipedia's "Suomen rautatieliikenne" (VR, Matkahaku,
    # 2025), differenced (VR). Line extents are the articles'; see fi_sources.md.
    "fi": [
        ("Päärata", "", 187.0, "VR Helsinki - Tampere; the article's Päärata runs on to Oulu, "
                               "built as Tampere–Seinäjoki-rata and Pohjanmaan rata"),
        ("Rantarata", "", 195.8, "WP Helsinki - Turku satama; Helsinki - Pasila (3.2 km) is "
                                 "on the Päärata, so expect 0.98"),
        ("Savon rata", "", 357.8, "WP Kouvola - Iisalmi; RINF 354.6"),
        ("Karjalan rata", "", 316.0, "VR 482 - 166, Kouvola - Joensuu; WP says 325.8, which "
                                     "RINF (314.7) does not support"),
        ("Pohjanmaan rata", "", 334.8, "WP Seinäjoki - Oulu"),
        ("Tampere–Seinäjoki-rata", "", 160.0, "WP"),
        ("Riihimäki–Lahti-rata", "", 59.0, "WP"),
        ("Lahti–Kouvola-rata", "", 61.4, "WP"),
        ("Lahden oikorata", "", 74.0, "WP Kerava - Lahti; RINF's track number 007 is Kytömaa - "
                                      "Hakosilta (63.5), the ends into Kerava and Lahti being "
                                      "the Päärata's and the Riihimäki–Lahti-rata's track, "
                                      "so expect 0.84"),
        ("Oulu–Tornio-rata", "", 131.9, "WP"),
        ("Kolarin rata", "", 182.0, "WP Tornio - Kolari"),
        ("Iisalmi–Kontiomäki-rata", "", 108.4, "WP"),
        ("Oulu–Kontiomäki-rata", "", 166.0, "WP"),
        ("Iisalmi–Ylivieska-rata", "", 154.4, "WP"),
        ("Turku–Toijala-rata", "", 131.0, "WP"),
        ("Tampere–Haapamäki-rata", "", 114.0, "VR 301 - 187; WP says 106.5, RINF 112.5"),
        ("Haapamäki–Seinäjoki-rata", "", 117.8, "WP"),
        ("Orivesi–Jyväskylä-rata", "", 112.7, "WP article text (its infobox's 166 is "
                                              "Tampere - Jyväskylä)"),
        ("Jyväskylä–Pieksämäki-rata", "", 79.8, "WP"),
        ("Haapamäki–Jyväskylä-rata", "", 77.2, "WP"),
        ("Pieksämäki–Joensuu-rata", "", 181.7, "WP"),
        ("Vaasan rata", "", 74.0, "VR 420 - 346, Seinäjoki - Vaasa; WP's 78 runs on to the "
                                  "Vaskiluoto port"),
        ("Kotkan rata", "", 52.0, "WP Kouvola - Kotkan satama"),
        ("Hangon rata", "", 49.3, "WP Karjaa - Hanko"),
        ("Kehärata", "", 27.0, "WP Huopalahti - Hiekkaharju"),
    ],
    # Register lines as rinf_countries/lt.py names them. Figures are LTG Infra's line lengths
    # as lt.wikipedia's "Lietuvos geležinkelių transportas" lists them (WP, raw wikitext,
    # retrieved 2026-10-01); two are that list less a piece of RINF's own chainage, said so.
    # lt_sources.md.
    "lt": [
        ("Naujoji Vilnia–Turmantas", "", 139.0, "WP Naujoji Vilnia - Turmantas - border"),
        ("Šiauliai–Joniškis", "", 60.0, "WP Šiauliai - Joniškis - border"),
        ("Palemonas–Kazlų Rūda–Šeštokai–Mockava (Rail Baltica)", "", 120.0,
         "WP 1435 mm Palemonas - Kaunas - Kazlų Rūda - Mockava - border; RINF 125.1"),
        ("Palemonas–Gaižiūnai–Radviliškis–Šiauliai–Klaipėda", "", 312.3,
         "WP Palemonas - Gaižiūnai 25 + Kaišiadorys - Radviliškis - Kužiai 161 less RINF's "
         "Kaišiadorys - Gaižiūnai 23.2 + Kužiai - Kretinga 127 + RINF's Kretinga - Klaipėda 22.5"),
        ("Kyviškės–Kena", "", 19.0, "WP Naujoji Vilnia - Kena - border 27 less RINF's Naujoji "
                                    "Vilnia - Kyviškės 8.0"),
        ("Kazlų Rūda–Kybartai", "", 57.3, "WP Kaunas - Kybartai - border 94 less RINF's Kaunas - "
                                          "Kazlų Rūda 36.7; RINF's own Kaunas - border is 87.3, "
                                          "so expect about 0.88"),
        ("Lentvaris–Varėna–Marcinkonys", "", 107.0, "WP Lentvaris - Marcinkonys - border; RINF's "
                                                    "line ends at Marcinkonys (81.5), the rest "
                                                    "to Belarus is not in it: expect 0.76"),
        ("Senieji Trakai–Trakai", "", 3.0, "WP, a whole number; RINF 3.7"),
    ],
    # Register lines as rinf_countries/lv.py names them from LDz's line number. Figures are
    # Wikidata P2043 on each line's item (lv.wikipedia's infoboxes), retrieved 2026-10-01.
    # lv_sources.md.
    "lv": [
        ("Rīga–Jelgava", "", 43.0, "WD, line 14"),
        ("Jelgava–Liepāja", "", 180.0, "WD, line 15"),
        ("Jelgava–Meitene", "", 33.0, "WD, line 16, to the border"),
        ("Rīga–Lugaži", "", 166.0, "WD, line 17, to the border; RINF's sections add to 157.2, "
                                   "with Vangaži - Krievupe and Jāņamuiža - Cēsis short"),
        ("Torņakalns–Tukums II", "", 65.0, "WD, line 18; the built line starts at Rīga "
                                           "Pasažieru, 2.3 km before Torņakalns: expect 1.03"),
        ("Krustpils–Daugavpils", "", 88.4, "WD, line 04"),
        ("Daugavpils–Indra", "", 76.0, "WD, line 05, to the border; Indra - border (6 km) has "
                                       "no OSM route and is dropped as unridden: expect 0.91"),
        ("Rēzekne II–Zilupe", "", 55.0, "WD, line 08"),
    ],
    # Register lines as rinf_countries/ee.py names them. Figures are et.wikipedia's "Eesti
    # raudteetransport" line list (WP, raw wikitext, retrieved 2026-10-01); three lines are a
    # composite there less a piece of RINF's own chainage, said so. ee_sources.md.
    "ee": [
        ("Tapa–Tartu", "", 112.0, "WP"),
        ("Tartu–Valga", "", 83.0, "WP"),
        ("Tartu–Koidula", "", 87.0, "WP"),
        ("Tapa–Narva", "", 133.4, "WP Tallinn - Narva 211 less RINF's Tallinn - Tapa 77.6"),
        ("Keila–Paldiski", "", 21.1, "WP Tallinn - Keila - Paldiski 48 less RINF's Tallinn - "
                                     "Keila 26.9"),
        ("Lelle–Viljandi", "", 79.2, "WP Tallinn-Väike - Lelle - Viljandi 148 less RINF's "
                                     "Tallinn-Väike - Lelle 68.8"),
        ("Keila–Riisipere", "", 31.0, "WP Keila - Turba; Riisipere - Turba (6.5 km, reopened "
                                      "2020) is not in RINF and stays on Elron's R16: expect "
                                      "0.79"),
        ("Klooga–Kloogaranna", "", 3.0, "WP, a whole number; RINF 3.4"),
    ],
    # Register lines as rinf_countries/ro.py names them: the CFR timetable's number and route,
    # "Magistrala 300 ..." for the main lines and "Secția 202 ..." for the rest. Figures are
    # ro.wikipedia's (retrieved 2026-10-01): the infobox length of the line's article (WP; for
    # the magistrale the "Magistrala CFR N" article, MG), or the table in "Magistrale feroviare
    # în România" (MT); 125 is Wikidata's P2043 (WD). Mostly uncited; check_model compares every
    # line with RINF's own section lengths as well. Lines closed or unbuilt in part (107, 213,
    # 314), whose article covers a different stretch (102, 105, 119, 306, 701, 806), or whose
    # figure does not add up from its own sub-articles (400's 560, 600's "395 / 187") are left
    # out. Sources in ro_sources.md.
    "ro": [
        ("Magistrala 100 București – Timișoara", "", 533.0, "MG, București Nord - Timișoara "
                                                            "Nord"),
        ("Magistrala 200 Brașov – Curtici", "", 470.0, "MT; the MG infobox says both 450 and 500"),
        ("Magistrala 300 București – Brașov – Cluj-Napoca – Oradea", "", 647.0,
         "MG, București Nord - Episcopia Bihor; built also has Episcopia Bihor - border (7.1, "
         "CFR SA's 300A) and both ways round Ploiești (Brazi - Ploiești Triaj - Ploiești Vest, "
         "7.7, which the trains use, beside RINF 300's own through Ploiești Sud), less "
         "Valea Drăganului - Poieni (2.8, no OSM path): ~1.03"),
        ("Magistrala 500 București – Vicșani", "", 488.0, "MG, from București Nord; București - "
                                                          "Ploiești Sud (59 km) is 300's track in "
                                                          "the register, so expect about 0.88"),
        ("Magistrala 700 București – Galați", "", 229.0, "MG"),
        ("Magistrala 800 București – Mangalia", "", 268.0, "MG; built also has București Nord - "
                                                           "Băneasa (CFR SA's 301P, 5.2, on OSM's "
                                                           "800 relation), so RINF's own is 273.0, "
                                                           "and the trace is 5.8 longer than RINF "
                                                           "at Neptun and Mangalia: ~1.04"),
        ("Secția 101 București – Craiova", "", 250.1, "WP București–Pitești–Craiova"),
        ("Secția 103 Videle – Giurgiu", "", 67.0, "WP"),
        ("Secția 104 Titu – Pietroșița", "", 66.9, "WP"),
        ("Secția 106 Pitești – Curtea de Argeș", "", 38.4, "WP"),
        ("Secția 112 Craiova – Calafat", "", 107.0, "WP"),
        ("Secția 122 Timișoara – Stamora Moravița", "", 56.0, "WP"),
        ("Secția 125 Oravița – Anina", "", 33.4, "WD (Q12723301); RINF's own is 33.4 too, and "
                                                "Brădișoru de Jos - Dobrei traces 7.8 km for "
                                                "RINF's 6.3 on the line's own track: ~1.05"),
        ("Secția 126 Timișoara – Cruceni", "", 49.0, "WP, from Timișoara Nord; its first 3 km to "
                                                    "Ram. Modoș are 122's track, and Ram. Modoș "
                                                    "- Timișoara Vest (2.8) is an unridden "
                                                    "junction section: ~0.90"),
        ("Secția 201 Podu Olt – Piatra Olt", "", 163.11, "WP"),
        ("Secția 202 Simeria – Filiași", "", 202.0, "WP; Wikidata 204"),
        ("Secția 203 Bartolomeu – Zărnești", "", 27.86, "WP Brașov–Zărnești; Brașov - Bartolomeu "
                                                       "(about 4 km) is 200's track: ~0.85"),
        ("Secția 208 Sibiu – Copșa Mică", "", 45.0, "WP"),
        ("Secția 210 Alba Iulia – Zlatna", "", 42.0, "WP; Alba Iulia - Bărăbanț (about 5 km) is "
                                                    "200A's track: ~0.88"),
        ("Secția 212 Ilia – Lugoj", "", 83.0, "WP"),
        ("Secția 218 Timișoara – Cenad", "", 78.0, "WP"),
        ("Secția 304 Ploiești – Măneciu", "", 50.416, "WP"),
        ("Secția 317 Sântana – Brad", "", 144.0, "WP"),
        ("Secția 405 Deda – Războieni", "", 113.6, "WP Deda–Târgu Mureș–Războieni"),
        ("Secția 412 Carei – Jibou", "", 110.88, "WP Carei–Zalău–Jibou"),
        ("Secția 502 Ilva Mică – Suceava", "", 191.0, "WP Suceava–Vama–Floreni–Ilva Mică"),
        ("Secția 504 Buzău – Nehoiașu", "", 73.3, "WP"),
        ("Secția 511 Verești – Botoșani", "", 44.5, "WP"),
        ("Secția 801 București – Oltenița", "", 59.4, "WP Titan Sud–Oltenița"),
        ("Secția 802 Slobozia – Călărași", "", 44.0, "WP"),
        ("Secția 804 Medgidia – Tulcea", "", 143.8, "WP"),
    ],
    # Register lines as rinf.py names them for Croatia, HŽI's number and route: "M202 Zagreb
    # GK – Rijeka" (rinf_countries/hr.py). Figures are HŽ Infrastruktura's own constructional
    # line lengths, table 1.4 of "Statistika HŽ Infrastrukture za 2025" (hzinfra.hr, July
    # 2026), border stubs included; table 3.7 of the same report gives the per-section lengths
    # quoted for the misses. Lines with no passenger trains are listed too (greyed, not dropped).
    # Dropped wholly as unridden freight track, so not listed: M304 Metković - Ploče (no OSM
    # stations or routes; 137 seasonal passenger trains in 2025), the Zagreb yard lines M401-
    # M405, M407-M410, and M602, M603, L207, L211, L212. hr_sources.md.
    "hr": [
        ("M101 Savski Marof – Zagreb GK", "", 26.758, "HŽI, to the border"),
        ("M102 Zagreb GK – Dugo Selo", "", 21.198, "HŽI"),
        ("M103 Dugo Selo – Novska", "", 83.405, "HŽI"),
        ("M104 Novska – Tovarnik", "", 185.405, "HŽI, to the border; Tovarnik - DG (1.5) has no "
                                                "OSM route and is dropped: ~0.99"),
        ("M201 Botovo – Dugo Selo", "", 79.508, "HŽI, to the border; Novo Drnje - DG (3.9, 823 "
                                                "passenger trains in 2025) has no OSM route and is "
                                                "dropped: ~0.94"),
        ("M202 Zagreb GK – Rijeka", "", 227.871, "HŽI"),
        ("M203 Rijeka – Šapjane", "", 30.896, "HŽI, to the border"),
        ("M301 Beli Manastir – Osijek", "", 32.505, "HŽI, to the border"),
        ("M302 Osijek – Strizivojna-Vrpolje", "", 48.377, "HŽI"),
        ("M303 Strizivojna-Vrpolje – Slavonski Šamac", "", 23.298,
         "HŽI, to the border; Slavonski Šamac - DG (2.8, 15 trains in 2025) has no OSM route: "
         "~0.85"),
        ("M501 Čakovec – Kotoriba", "", 42.388, "HŽI, border to border; Kotoriba - DG (3.4, 14 "
                                                "trains in 2025) has no OSM route: ~0.92"),
        ("M502 Zagreb GK – Sisak – Novska", "", 116.791, "HŽI, M502-1 14.048 + M502-2 102.743"),
        ("M601 Vinkovci – Vukovar", "", 18.918, "HŽI"),
        ("M604 Oštarije – Knin – Split", "", 322.099, "HŽI"),
        ("M605 Ogulin – Krpelj", "", 6.153, "HŽI; its table 3.7 and RINF give 5.842: ~0.94"),
        ("M606 Knin – Zadar", "", 95.364, "HŽI; no passenger trains in 2025, greyed"),
        ("M607 Perković – Šibenik", "", 22.503, "HŽI; table 3.7 gives 21.449: ~0.95"),
        ("R101 Buzet – Pula", "", 91.140, "HŽI, to the border"),
        ("R102 Sunja – Volinja", "", 21.575, "HŽI, to the border; Volinja - DG (1.5, 16 trains "
                                             "in 2025) has no OSM route; table 3.7's Sunja - "
                                             "Volinja is 19.669: ~0.91"),
        ("R104 Vukovar-Borovo naselje – Erdut", "", 26.059, "HŽI, to the border; Erdut - DG "
                                                            "(3.7, 2 trains in 2025) dropped: "
                                                            "~0.84"),
        ("R105 Vinkovci – Drenovci", "", 50.939, "HŽI, to the border; Gunja - DG (2.0, no "
                                                 "trains) dropped: ~0.96"),
        ("R106 Zabok – Đurmanec", "", 27.198, "HŽI, to the border (hr.py adds RINF's missing "
                                              "border point)"),
        ("R201 Zaprešić – Čakovec", "", 100.714, "HŽI"),
        ("R202 Varaždin – Dalj", "", 249.842, "HŽI"),
        ("L101 Čakovec – Mursko Središće", "", 17.942, "HŽI, to the border"),
        ("L103 Karlovac – Kamanje", "", 28.799, "HŽI, to the border; no passenger trains in "
                                               "2025, greyed"),
        ("L201 Varaždin – Golubovec", "", 34.596, "HŽI"),
        ("L202 Hum-Lug – Gornja Stubica", "", 10.820, "HŽI"),
        ("L203 Križevci – Bjelovar – Kloštar", "", 62.047, "HŽI"),
        ("L204 Banova Jaruga – Pčelić", "", 95.752, "HŽI"),
        ("L205 Nova Kapela – Našice", "", 42.0, "HŽI's 60.493 less Čaglin - Našice cement "
                                               "(18.515, closed); built also lacks Našice Grad - "
                                               "Našice cement (1.9, freight): ~0.93"),
        ("L206 Pleternica – Velika", "", 24.955, "HŽI"),
        ("L208 Vinkovci – Osijek", "", 33.770, "HŽI"),
        ("L209 Vinkovci – Županja", "", 28.073, "HŽI"),
        ("L214 Gradec – Sveti Ivan Žabno", "", 12.520, "HŽI"),
    ],
    # Russia: tariff sections (ru_register.py), named by their two ends. check_model compares
    # every line with the tariff guide's own km (km_official) as well; these are the outside
    # figures, and there are few: of the 528 Wikidata line items named for a section's two
    # ends, 18 carry a length (P2043, WD) and 10 have a ru.wikipedia article, 7 of them with
    # an infobox length (WP; probe_ru_wplengths.py). Retrieved 2026-10-01. Many of these
    # figures may themselves come from the tariff guide. Left out: Пинозеро — Ковдор (WD 117)
    # and Первушино — Заволжск (WD 70), built as no line because no OSM passenger route runs
    # on them (build_model drops unridden junction-ended sections). ru_sources.md.
    "ru": [
        ("Зеленый Дол — Яранск", "", 196.0, "WP, WD"),
        ("Соблаго — Торжок", "", 165.0, "WP, WD; Торжок - Кувшиново (58 km) has no OSM "
                                        "passenger route (trains run Осташков - Кувшиново "
                                        "only) and is dropped: expect 0.65"),
        ("Алтайская — Бийск (эксп.)", "", 147.0, "WD"),
        ("Петрозаводск — Суоярви I", "", 139.0, "WD; tariff 140, of which Петрозаводск - "
                                                "Томицы (9 km) is the same pair of points on "
                                                "01-027 and counted there: expect 0.93"),
        ("Советск — Калининград-Пассажирский", "", 123.7, "WD"),
        ("Узуново — Рыбное", "", 68.1, "WP, WD"),
        ("Кривандино — Рязановка", "", 53.0, "WP, WD"),
        ("Калининград-Пассажирский — Мамоново", "", 49.6, "WD, to Mamonovo station; the "
                                                         "tariff section runs 6 km on to the "
                                                         "Polish border point, unplaced"),
        ("Адлер — Роза Хутор", "", 48.2, "WD Адлер — Красная Поляна (Q4057605); tariff 48"),
        ("Овинище II (пп) — Весьегонск", "", 42.0, "WP, WD"),
        ("Голутвин — Озеры", "", 40.0, "WD; tariff 39"),
        ("Калининград-Пассажирский — Багратионовск", "", 35.8, "WD, to Bagrationovsk station; "
                                                               "tariff 41 to the border point"),
        ("Енисей — Дивногорск", "", 31.0, "WP, WD; tariff 30"),
        ("Угловка — Боровичи", "", 30.0, "WP, WD"),
        ("Софрино — Красноармейск", "", 15.0, "WD; tariff 16, traced 16.6 (1.04 of it): the "
                                              "WD figure is low"),
        ("Вырица — Поселок", "", 7.0, "WD; tariff 7; the last point, Платформа № 4, has no OSM "
                                      "node, so the line ends 0.7 km short at Платформа № 3"),
    ],
    # Italy: RFI publishes no per-line length table that is open (the PIR's line list is on
    # the ePIR portal). CH = it.wikipedia's station table (Percorso) chainage, differenced
    # between the line's two ends as built; WP = it.wikipedia infobox `lunghezza`; WD =
    # Wikidata P2043 of the it.wikipedia article's item. Retrieved 2026-10-02. RFI's F lines
    # run node to node (Milano - Bologna is Rogoredo - Lavino), so their figures are the
    # chainage between those points, not the whole Wikipedia line. it_sources.md.
    "it": [
        # fundamental lines (CH)
        ("Brennero – Verona", "", 238.71, "CH Verona Porta Nuova 0 - Brennero 238.711"),
        ("Firenze – Roma (Direttissima)", "", 237.63, "CH Firenze Rovezzano 254.004 - "
                                                      "Settebagni 16.379"),
        ("Milano – Bologna", "", 199.23, "CH Milano Rogoredo 208.751 - PM Lavino 9.522; RINF "
                                         "also files the Rogoredo - Bivio Melegnano - "
                                         "Tavazzano pair (18.2 km) under F41-F42: ~1.09"),
        ("Bologna – Ancona", "", 193.09, "CH PM Mirandola-Ozzano 10.906 - Ancona 203.996"),
        ("Ancona – Foggia", "", 322.03, "CH Ancona 203.996 - Foggia 526.027"),
        ("Orte – Ancona", "", 211.63, "CH Orte 82.503 - Falconara 285.429 = 195.299, - "
                                      "Ancona 203.996"),
        ("Fiumetorto – Messina", "", 180.55, "CH Fiumetorto 43.219 - Messina Centrale 223.764; "
                                             "built has both the old coast line via Falcone "
                                             "and the Patti - Terme Vigliatore line (18 km), "
                                             "as RINF does: ~1.10"),
        ("Catanzaro Lido – Reggio Calabria", "", 177.55, "CH Jonica 294.720 - 472.270"),
        ("Sibari – Catanzaro Lido", "", 172.48, "CH Jonica 122.237 - 294.720"),
        ("Genova – Pisa", "", 147.70, "CH Genova Nervi 10.791 - La Spezia Centrale 86.162 = "
                                      "172.462 - Pisa San Rossore 100.133; built has both "
                                      "Vezzano - La Spezia routes (Migliarina, Cà di "
                                      "Boschetti) and Pisa Centrale: ~1.06"),
        ("Venezia – Trieste", "", 141.10, "CH Venezia Carpenedo 3.904 - 131.315 = 13.687 - "
                                          "Trieste Centrale 0"),
        ("Milano – Torino", "", 118.81, "CH Settimo 15.763 - Rho 134.571"),
        ("Torino – Arquata Scrivia", "", 110.10, "CH Trofarello 13.030 - Arquata Scrivia "
                                                 "123.132"),
        ("Verona – Bologna", "", 103.01, "CH Verona Porta Nuova 114.951 - PM Tavernelle "
                                         "11.941; built also has the Verona Porta Vescovo "
                                         "leg (6.8 km): ~1.06"),
        ("Bologna – Padova", "", 99.04, "CH San Pietro in Casale 23.879 - Padova 122.921"),
        ("Bologna – Firenze (Direttissima)", "", 97.0, "WD"),
        ("Alessandria – Piacenza", "", 96.51, "CH; built also has the Bressana Bottarone - "
                                              "Barbianello - Broni leg towards Pavia (13.3 "
                                              "km) and Alessandria Smistamento: ~1.15"),
        ("Messina – Catania", "", 94.77, "CH Catania Centrale 240.714 - Messina Centrale "
                                         "335.485"),
        ("Palermo – Fiumetorto", "", 43.22, "CH"),
        # high speed
        ("AV Torino – Milano", "", 125.0, "WD (WP 127)"),
        ("AV Bologna – Firenze", "", 78.5, "WD, the line proper; built runs from Bologna "
                                           "Centrale's underground AV station (WP 86): ~1.09"),
        # complementary lines and groups (WP, WD)
        ("Battipaglia – Potenza – Metaponto", "", 198.0, "WP"),
        ("Ferrara – Ravenna – Rimini", "", 123.0, "WD"),
        ("Lecco – Sondrio – Tirano", "", 105.0, "WD"),
        ("Lucca – Aulla", "", 90.0, "WP (WD 98)"),
        ("Domodossola – Novara", "", 89.56, "WD; Vignale - Novara (3.2 km) is RFI's C24, a "
                                            "line of its own: ~0.95"),
        ("Mantova – Monselice", "", 84.1, "WP, WD"),
        ("Terontola – Foligno", "", 82.0, "WD"),
        ("Avezzano – Roccasecca", "", 79.0, "WD"),
        ("Fortezza – San Candido", "", 73.06, "WP 65 to San Candido + RINF's San Candido - "
                                              "border 8.06"),
        ("Barletta – Spinazzola", "", 66.0, "WD"),
        ("Empoli – Siena", "", 63.0, "WD"),
        ("Cremona – Mantova", "", 62.0, "WD; Bozzolo - Mantova (26 km) has no track in the "
                                        "extract (railway=construction in OSM): ~0.59"),
        ("Vicenza – Treviso", "", 60.0, "WD"),
        ("Rovigo – Chioggia", "", 57.0, "WD"),
        ("Treviso – Portogruaro", "", 52.5, "WP, WD"),
        ("Novara – Biella", "", 51.0, "WD"),
        ("Asciano – Monte Antico", "", 51.0, "WD"),
        ("Savigliano – Saluzzo – Cuneo", "", 48.0, "WD (WP 49)"),
        ("Vairano – Isernia", "", 45.0, "WD"),
        ("Milano – Mortara", "", 44.0, "WD"),
        ("Lamezia Terme – Catanzaro Lido", "", 43.0, "WD"),
        ("Torino – Ceres", "", 42.0, "WP (WD 42.88)"),
        ("Castel Bolognese – Ravenna", "", 41.0, "WD"),
        ("Conegliano – Ponte nelle Alpi", "", 40.0, "WD"),
        ("Viterbo – Attigliano", "", 39.0, "WD"),
        ("Palermo – Punta Raisi", "", 37.0, "WP Passante ferroviario di Palermo (WD 38)"),
        ("Foggia – Manfredonia", "", 36.0, "WD"),
        ("Alessandria – Ovada", "", 34.0, "WD"),
        ("Pontassieve – Borgo San Lorenzo", "", 33.0, "WD"),
        ("Bolzano – Merano", "", 31.8, "WP, WD"),
        ("Seregno – Ponte San Pietro", "", 31.0, "WP Seregno-Bergamo (WD 40 is to Bergamo)"),
        ("Vicenza – Schio", "", 31.0, "WD"),
        ("Cecina – Volterra", "", 30.0, "WP (WD 37.5)"),
        ("Monza – Molteno", "", 29.0, "WD"),
        ("Santhià – Biella", "", 27.0, "WD"),
        ("Paola – Cosenza", "", 26.0, "WP (WD 27.65)"),
        ("Colico – Chiavenna", "", 26.0, "WD"),
        ("Giulianova – Teramo", "", 25.0, "WD"),
        ("Fossano – Cuneo", "", 25.0, "WD"),
        ("Pisa – Lucca", "", 23.0, "WP"),
        ("Lucca – Viareggio", "", 23.0, "WP (WD 22)"),
        ("Casarsa – Portogruaro", "", 22.0, "WP"),
        ("Treviglio – Bergamo", "", 22.0, "WD"),
        ("Montebelluna – Treviso", "", 20.0, "WD"),
        ("Carmagnola – Bra", "", 20.0, "WD (WP 25)"),
        ("Salerno – Mercato San Severino", "", 17.63, "WD"),
        ("Campiglia Marittima – Piombino", "", 16.0, "WD (WP 15)"),
        ("Palazzolo sull'Oglio – Paratico", "", 9.7, "WP"),
        ("Bari – Bitritto", "", 9.0, "WP (WD 11.9); RINF 9.3"),
        ("Fidenza – Salsomaggiore Terme", "", 9.0, "WD"),
        ("Nocera Inferiore – Codola", "", 4.0, "WP, rounded; RINF 4.2"),
        # other infrastructure managers
        ("Bari – Martina Franca – Taranto", "", 112.63, "WP, WD"),
        ("Brescia – Iseo – Edolo", "", 103.0, "WP (WD 105)"),
        ("Martina Franca – Lecce", "", 102.588, "WP"),
        ("Novoli – Gagliano del Capo", "", 74.194, "WP"),
        ("Lecce – Gallipoli", "", 53.812, "WP, WD"),
        ("Ferrara – Codigoro", "", 53.0, "WD"),
        ("Saronno – Laveno", "", 51.1, "WD"),
        ("Cancello – Benevento", "", 47.9, "WP"),
        ("Zollino – Gagliano del Capo", "", 46.502, "WP, WD"),
        ("Bologna – Portomaggiore", "", 45.0, "WD from Bologna Centrale; FER's RINF line "
                                              "starts at Bologna Roveri: ~0.93"),
        ("Bari Mungivacca – Putignano", "", 43.412, "WP, WD"),
        ("Parma – Suzzara", "", 43.0, "WD"),
        ("Santa Maria Capua Vetere – Piedimonte Matese", "", 41.245, "WP Ferrovia Alifana"),
        ("Saronno – Novara", "", 40.02, "WD"),
        ("Reggio Emilia – Guastalla", "", 29.0, "WD; built also has the Reggio San Lazzaro "
                                                "spur (2.1 km): ~1.05"),
        ("Reggio Emilia – Ciano d'Enza", "", 26.0, "WD"),
        ("Saronno – Como", "", 24.6, "WP, WD"),
        ("Casalecchio di Reno – Vignola", "", 24.0, "WD"),
        ("Reggio Emilia – Sassuolo", "", 22.494, "WP"),
        ("Casarano – Gallipoli", "", 22.003, "WP"),
        ("Milano – Saronno", "", 21.0, "WD"),
        ("Foggia – Lucera", "", 19.353, "WP, WD"),
        ("Modena – Sassuolo", "", 19.0, "WD"),
        ("Maglie – Otranto", "", 18.271, "WP"),
        ("Saronno – Seregno", "", 15.0, "WP, from Saronno; FERROVIENORD's RINF line starts at "
                                        "Saronno Sud (13.15): ~0.89"),
        ("Udine – Cividale", "", 15.0, "WP (WD 15.3)"),
    ],
    # Adif's line numbers, as rinf_countries/es.py names them ("100 Hendaya – Madrid-Chamartín-
    # Clara Campoamor"), against the catalogue length in es.wikipedia "Anexo:Líneas de la Red
    # Ferroviaria de Interés General" (Orden FOM/710/2015 and Adif's Declaración sobre la Red;
    # retrieved 2026-10-02). Lines whose build covers a different extent say so; es_sources.md.
    "es": [
        ("050 Límite ADIF-LFPSA – Madrid-Puerta de Atocha-Almudena Grandes", "", 752.4,
         "catalogue; Sants - Riells and Alcover - Camp de Tarragona added in es.py"),
        ("200 Madrid-Chamartín-Clara Campoamor – Barcelona-Estació de França", "", 699.7,
         "catalogue"),
        ("100 Hendaya – Madrid-Chamartín-Clara Campoamor", "", 640.9,
         "catalogue"),
        ("400 Alcázar de San Juan – Cádiz", "", 576.9,
         "catalogue"),
        ("300 Madrid-Chamartín-Clara Campoamor – València-Estació del Nord", "", 480.6,
         "catalogue; RINF's own is 492.0: ~1.03"),
        ("010 Madrid-Puerta de Atocha-Almudena Grandes – Sevilla-Santa Justa", "", 470.5,
         "catalogue"),
        ("822 Bifurcación Valorio – A Coruña", "", 436.3,
         "catalogue"),
        ("800 A Coruña – León-Aguja km 123,6", "", 428.2,
         "catalogue"),
        ("040 Madrid-Chamartín-Clara Campoamor – Valencia-Joaquín Sorolla", "", 397.6,
         "catalogue"),
        ("520 Ciudad Real – Badajoz", "", 336.7,
         "catalogue"),
        ("610 Sagunt – Bifurcación Teruel", "", 314.5,
         "catalogue"),
        ("982 Taboadela aguja km 234,0 – Bifurcación Medina", "", 313.9,
         "catalogue (to Taboadela); the build carries on over the mixed-gauge Taboadela - "
         "Ourense stretch RINF files under 982 (15.4 km): ~1.04"),
        ("790 Aranguren – Asunción Universidad", "", 310.0,
         "catalogue"),
        ("130 Gijón-Sanz Crespo – Venta de Baños", "", 306.1,
         "catalogue; RINF 303.8"),
        ("080 Burgos-Rosa Manzano – Madrid-Chamartín-Clara Campoamor", "", 304.0,
         "catalogue; Las Pajareras - Dueñas and Venta de Baños - La Vega added in es.py"),
        ("210 Miraflores – Sant Vicenç de Calders", "", 275.9,
         "catalogue"),
        ("740 Pravia – Ferrol", "", 269.0,
         "catalogue"),
        ("600 València-Estació del Nord – Cambiador de La Boella", "", 254.1,
         "catalogue"),
        ("410 Linares-Baeza – Almería", "", 240.8,
         "catalogue; Huércal-Viator - Almería is closed for works in OSM: ~0.97"),
        ("042 Bifurcación Albacete – Alacant-Terminal", "", 237.8,
         "catalogue"),
        ("160 Santander – Palencia", "", 217.2,
         "catalogue"),
        ("770 Santander – Oviedo", "", 216.0,
         "catalogue"),
        ("120 Villar Formoso – Medina del Campo", "", 201.0,
         "catalogue; Tejares - Barbadillo added in es.py"),
        ("220 Lleida-Pirineus – Bifurcación Vilanova", "", 181.7,
         "catalogue"),
        ("420 Bifurcación Las Maravillas – Algeciras", "", 179.6,
         "catalogue"),
        ("026 Plasencia – Bifurcación San Nicolás", "", 175.3,
         "catalogue; Peñas Blancas - Bif. La Isla added in es.py (RINF lacks it)"),
        ("270 Cerbère – Bifurcación Aragó", "", 162.1,
         "catalogue; built runs on to the border point: ~1.01"),
        ("030 Bifurcación Málaga-Alta Velocidad – Málaga-María Zambrano", "", 154.5,
         "catalogue"),
        ("222 La Tor de Querol-Enveitg – Bifurcació Aigües", "", 149.7,
         "catalogue; Montcada - La Garriga is railway=construction in OSM (doubling works, R3 "
         "runs La Garriga - Puigcerdà only): ~0.85"),
        ("320 Chinchilla de Montearagón-aguja km 298,4 – Murcia del Carmen", "", 146.2,
         "catalogue; Chinchilla - Hellín (51 km) has no OSM passenger route and ends at a "
         "junction, so it is dropped as unridden: ~0.65"),
        ("710 Altsasu – Castejón de Ebro", "", 139.2,
         "catalogue"),
        ("204 Bifurcación Canfranc – Canfranc", "", 138.5,
         "catalogue"),
        ("084 León – Bifurcación Venta de Baños", "", 127.9,
         "catalogue; Las Barreras - Vilecha (79 km) added in es.py"),
        ("036 Antequera-Santa Ana – Granada", "", 125.7,
         "catalogue; RINF's own is 114.6: ~0.91"),
        ("122 Salamanca – Ávila", "", 111.1,
         "catalogue"),
        ("440 Bifurcación Los Naranjos – Huelva", "", 109.1,
         "catalogue"),
        ("820 Zamora-aguja km 233 – Medina del Campo", "", 90.2,
         "catalogue"),
        ("276 Maçanet-Massanes – L'Hospitalet de Llobregat", "", 85.1,
         "catalogue"),
        ("082 Bifurcación A Grandeira aguja km 85,0 – Bifurcación Coto da Torre", "", 84.0,
         "catalogue"),
        ("330 La Encina – Alacant-Terminal", "", 78.3,
         "catalogue"),
        ("336 El Reguerón-aguja km 525,3 – Alacant-Terminal", "", 73.7,
         "catalogue"),
        ("240 Sant Vicenç de Calders – L'Hospitalet de Llobregat", "", 71.0,
         "catalogue"),
        ("522 Manzanares – Ciudad Real", "", 64.5,
         "catalogue"),
        ("342 Alcoi – Xàtiva", "", 63.7,
         "catalogue"),
        ("110 Segovia – Villalba de Guadarrama", "", 62.7,
         "catalogue"),
        ("416 Moreda – Granada", "", 56.7,
         "catalogue"),
        ("764 Trubia – Collanzo", "", 55.0,
         "catalogue"),
        ("344 Gandia – Silla", "", 50.8,
         "catalogue"),
        ("920 Móstoles-El Soto – Parla", "", 45.3,
         "catalogue"),
        ("804 Betanzos-Infiesta – Ferrol", "", 42.8,
         "catalogue"),
        ("436 Fuengirola – Málaga-Centro Alameda (apeadero)", "", 30.8,
         "catalogue"),
        ("910 Madrid-Atocha Cercanías – Pinar de Las Rozas", "", 28.0,
         "catalogue"),
        ("020 La Sagra – Toledo", "", 21.4,
         "catalogue"),
    ],
    # DB InfraGO's lines by VzG number, named "<number> <DB's Streckenkurzname>". Figures are
    # Wikidata's length (P2043) on the item carrying that one route number (P1671), which is
    # the de.wikipedia infobox ("WP"), taken only where the article covers the same extent as
    # the number (within 3% of RINF's own length); two high-speed lines with no such item use
    # DB InfraGO's Streckennetz CSV (de_sources.md). Retrieved 2026-10-02.
    "de": [
        ("2200 Wanne-Eickel – Hamburg", "", 355.0, "Bahnstrecke Wanne-Eickel–Hamburg, WP"),
        ("2550 Aachen – Kassel", "", 343.3, "Bahnstrecke Aachen–Kassel, WP"),
        ("1733 Hannover – Kassel – Würzburg", "", 327.0, "SFS Hannover–Würzburg, WP"),
        ("6100 Berlin-Spandau – Hamburg-Altona", "", 284.1, "Bahnstrecke Berlin–Hamburg, WP"),
        ("1720 Lehrte – Cuxhaven", "", 256.9, "Bahnstrecke Lehrte–Hamburg-Harburg, WP"),
        ("3900 Kassel – Frankfurt", "", 199.8, "Main-Weser-Bahn, WP"),
        ("1700 Hannover – Hamm (Westf)", "", 176.4, "Bahnstrecke Hannover–Hamm, WP"),
        ("2651 Köln Messe/Deutz – Gießen", "", 166.2, "Siegstrecke + Dillstrecke, WP"),
        ("2631 Hürth-Kalscheuren – Ehrang", "", 163.5, "Eifelstrecke, WP"),
        ("6383 Leipzig-Leutzsch – Probstzella", "", 160.0, "Saalbahn north + south, WP"),
        ("2630 Köln – Bingen", "", 152.0, "Linke Rheinstrecke, WP"),
        ("4250 Offenburg – Singen", "", 149.1, "Schwarzwaldbahn, WP"),
        ("3511 Bingen Hbf – Saarbrücken", "", 141.8, "Nahetalbahn, WP"),
        ("5903 Nürnberg Hbf – Schirnding", "", 140.6, "Nürnberg – Schirnding Grenze, WP"),
        ("5321 Treuchtlingen – Würzburg", "", 140.2, "Bahnstrecke Treuchtlingen–Würzburg, WP"),
        ("5500 München – Regensburg", "", 138.1, "Bahnstrecke München–Regensburg, WP"),
        ("5501 München – Treuchtlingen", "", 136.7, "Bahnstrecke München–Treuchtlingen, WP"),
        ("5634 Landshut – Bayerisch Eisenstein", "", 134.6, "WP"),
        ("6153 Berlin Ostbahnhof – Guben (DB-Grenze)", "", 132.0, "WP"),
        ("6325 Neustrelitz – Warnemünde", "", 127.0, "WP"),
        ("2610 Köln – Kranenburg (DB-Grenze)", "", 120.0, "Linksniederrheinische Strecke, WP"),
        ("5600 München Ost – Simbach (Inn)", "", 115.1, "Bahnstrecke München–Simbach, WP"),
        ("6899 Stendal – Uelzen", "", 107.5, "Amerikalinie, WP"),
        ("4500 Ulm – Friedrichshafen", "", 103.6, "Südbahn, WP"),
        ("3710 Wetzlar – Koblenz", "", 104.0, "Lahntalbahn, WP"),
        ("6212 Görlitz – Dresden-Neustadt", "", 102.1, "Bahnstrecke Görlitz–Dresden, WP"),
        ("1040 Neumünster – Flensburg", "", 101.5, "WP"),
        ("5850 Regensburg – Nürnberg", "", 100.6, "WP"),
        ("6132 Berlin Südkreuz – Halle Hbf", "", 161.6, "Anhalter Bahn, WP; DB's line starts "
                                                        "at Südkreuz, RINF 156.2"),
        ("1220 Hamburg-Altona – Kiel", "", 105.6, "WP"),
        ("5510 München – Rosenheim", "", 64.9, "WP"),
        ("1120 Lübeck – Hamburg", "", 62.8, "WP"),
        ("5503 München – Augsburg", "", 61.9, "WP"),
        ("3603 Frankfurt – Wiesbaden", "", 41.2, "Taunus-Eisenbahn, WP"),
        ("6605 Heidenau – Altenberg (Erzgeb)", "", 38.0, "Müglitztalbahn, WP"),
        ("5453 Tutzing – Kochel", "", 35.5, "Kochelseebahn, WP"),
        ("6773 Wolgaster Fähre – Seebad Heringsdorf", "", 34.9, "Usedomer Bäderbahn, WP"),
        ("1206 Heide – Büsum", "", 24.0, "WP"),
        ("5451 Murnau – Oberammergau", "", 23.7, "Ammergaubahn, WP"),
        ("4311 Denzlingen – Elzach", "", 19.3, "Elztalbahn, WP"),
        # S-Bahn on light_rail track in OSM (rinf.py light_rail_track)
        ("1244 Hamburg Hbf(S-Bahn) – Aumühle", "", 25.2, "Hamburg S-Bahn to Aumühle, WP"),
        ("1271 Hamburg Hbf SB – Hamburg-Neugraben", "", 22.0, "Harburger S-Bahn, WP"),
        # High-speed lines with no single-number Wikidata length: DB InfraGO Streckennetz
        ("4080 Mannheim – Stuttgart-Zuffenhausen", "", 98.7, "SFS Mannheim–Stuttgart, DB"),
        ("2690 Köln – Frankfurt am Main Stadion", "", 164.4,
         "SFS Köln–Rhein/Main, DB; WP's 180 km adds the Wiesbaden and Köln/Bonn airport "
         "branches, which have numbers of their own"),
        ("5919 Eltersdorf – Leipzig Hbf", "", 293.3,
         "VDE 8 Nürnberg–Erfurt–Leipzig, WP item Q136766768; DB's own list says 284.0 and "
         "RINF 284.2, so expect 0.96: the gap is between the published figures"),
    ],
}

KNOWN = {
    "jp": [
        ("山手線", "東日本", 34.5, 30, "JR East, loop"),
        ("東海道本線", "", 589.5, None, "Tokyo-Kobe"),
        ("中央線", "東日本", 222.1, None, "Tokyo-Shiojiri, JR East half"),
        ("中央線", "東海", 174.8, None, "Shiojiri-Nagoya, JR Central half"),
        ("銀座線", "", 14.3, 19, "Tokyo Metro"),
        ("丸ノ内線", "", 27.4, 28, "Tokyo Metro, 24.2 main + 3.2 Honancho branch"),
        ("大江戸線", "", 40.7, 38, "Toei"),
        ("御堂筋線", "", 24.5, 20, "Osaka Metro"),
        # Matched on 東京 alone, not the company name: Tokyo Metro is tagged 東京地下鉄 on
        # the Tozai line and 東京メトロ on the Namboku line. Operator strings are free text
        # in OSM and are not consistent even within one company.
        ("東西線", "東京", 30.8, 23, "Tokyo Metro"),
        ("南北線", "東京", 21.3, 19, "Tokyo Metro"),
        ("京浜東北線", "", 81.2, None, "JR East, operating pattern"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--find", default=None, help="print every line matching this fragment")
    ap.add_argument("--shared", default=None,
                    help="for the biggest line matching this, which lines share its track")
    ap.add_argument("--coverage", action="store_true",
                    help="how much of the network only a named train reaches")
    ap.add_argument("--station", default=None,
                    help="every station whose name contains this, and how far apart they are")
    args = ap.parse_args()

    d = ROOT / "dist" / "data" / args.region
    lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]
    stations = json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"]
    byid = {l["id"]: l for l in lines}

    if args.find:
        hits = [l for l in lines if args.find in (l["name"] or "")
                or args.find in (l["ref"] or "")]
        print(f"{len(hits)} lines matching {args.find!r}\n")
        for l in sorted(hits, key=lambda l: -l["km"]):
            print(f"  {l['id']:<10} {l['km']:>8.1f} km  {len(l['sections']):>4} sections  "
                  f"{len(l['display']):>4} display stops  {l['variants']} variants  "
                  f"{l['straight_sections']} straight")
            print(f"      {l['ref']:<6} {l['name']}  [{l['operator']}]  {l['colour']}")
        return

    if args.coverage:
        """How much of the network only a NAMED TRAIN reaches.

        Completion counts lines, not trains. But OSM does not map a line relation everywhere:
        rural Japan often has only the limited expresses. The San'in Main Line, 673 km of JR
        West main line, has no route relation at all and its track is not named either, so
        the only thing calling at Matsue is the Super Oki. Anything counted here is track a
        rider can ride and the totals cannot see.
        """
        svc_only_st, no_line_st = [], []
        for sid, s in stations.items():
            ls = [l for l in (s.get("l") or []) if l in {x["id"] for x in lines}] \
                if False else (s.get("l") or [])
            kinds = [byid[l] for l in ls if l in byid]
            if not kinds:
                no_line_st.append(sid)
            elif all(l["service"] for l in kinds):
                svc_only_st.append(sid)
        gid_service = {}
        for l in lines:
            for a, b, km, gid in l["sections"]:
                if gid not in gid_service:
                    gid_service[gid] = (l["service"], km)
        # A piece of track counts as service-only when no non-service line has a section on
        # it. Sections are per line, so group by the credit ranges instead: approximate by
        # asking whether any non-service line has a section between the same two stations.
        pair_real = set()
        for l in lines:
            if l["service"]:
                continue
            for a, b, km, gid in l["sections"]:
                pair_real.add((a, b) if a <= b else (b, a))
        svc_km = 0.0
        for l in lines:
            if not l["service"]:
                continue
            for a, b, km, gid in l["sections"]:
                if ((a, b) if a <= b else (b, a)) not in pair_real:
                    svc_km += km
        print(f"{len(stations)} stations")
        print(f"  {len(svc_only_st):>5} are served only by a named train, never by a line")
        print(f"  {len(no_line_st):>5} have no line at all")
        print(f"\n{svc_km:,.0f} km of track is reached only by a named train and so counts "
              f"towards nothing")
        print("\nexamples of stations only a named train reaches")
        for sid in svc_only_st[:15]:
            s = stations[sid]
            names = [byid[l]["name"] for l in (s.get("l") or []) if l in byid][:3]
            print(f"  {(s['e'] or s['n']):<26} {', '.join(names)}")
        return

    if args.station:
        import math
        hits = [(i, s) for i, s in stations.items()
                if args.station in (s["n"] or "") or args.station.lower() in (s["e"] or "").lower()]
        print(f"{len(hits)} stations match {args.station!r}\n")
        for i, s in sorted(hits, key=lambda kv: -len(kv[1].get("l", []))):
            print(f"  {i:<14} {s['n']:<12} {s['e']:<22} {len(s.get('l', []))} lines"
                  f"  ({s['y']:.5f}, {s['x']:.5f})")
        # How far apart, so the merge radius can be argued with rather than guessed at again.
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                sa, sb = hits[a][1], hits[b][1]
                dx = (sb["x"] - sa["x"]) * math.cos(math.radians(sa["y"])) * 111320
                dy = (sb["y"] - sa["y"]) * 110570
                d = math.hypot(dx, dy)
                if d < 2000:
                    print(f"    {hits[a][0]} to {hits[b][0]}: {d:.0f} m"
                          f"   {sa['n']!r} / {sb['n']!r}")
        return

    if args.shared:
        hits = [l for l in lines if args.shared in (l["name"] or "")]
        if not hits:
            print(f"nothing matches {args.shared!r}")
            return
        line = max(hits, key=lambda l: l["km"])
        # Riding a section credits its footprint (foot.json, ownership.py); another section
        # is ridden where its own footprint is, as the app works it out (creditState).
        import ownership
        foot = ownership.read(d / "foot.json")
        owner, sec_km = {}, {}
        for l in lines:
            for a, b, km, gid in l["sections"]:
                owner[gid], sec_km[gid] = l, km
        mine = {gid for a, b, km, gid in line["sections"]}
        footof = lambda g: foot.get(g, [[g, 0.0, 1.0, 0.0, 1.0]])
        ridden = {}
        for g in mine:
            for t, f, to, a, b in footof(g) + [[g, 0.0, 1.0, 0.0, 1.0]]:
                ridden.setdefault(t, []).append((min(f, to), max(f, to)))

        def union(iv):
            out = []
            for lo, hi in sorted(iv):
                if out and lo <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], hi)
                else:
                    out.append([lo, hi])
            return out
        ridden = {t: union(iv) for t, iv in ridden.items()}

        print(f"{line['name']} [{line['operator']}]  {line['km']:.1f} km, "
              f"{len(line['sections'])} sections")
        print("If you rode all of it, what else would that complete:\n")

        tally = {}
        for gid in owner:
            if gid in mine:
                continue
            got = []
            for t, f, to, a, b in footof(gid):
                lo, hi = min(f, to), max(f, to)
                for x, y in ridden.get(t, ()):
                    ix, iy = max(x, lo), min(y, hi)
                    if iy > ix and hi > lo:
                        p1 = a + (ix - f) / (to - f) * (b - a)
                        p2 = a + (iy - f) / (to - f) * (b - a)
                        got.append((min(p1, p2), max(p1, p2)))
            frac = min(1.0, sum(hi - lo for lo, hi in union(got)))
            if frac <= 0:
                continue
            o = owner[gid]
            name = o["name"] or o["id"]
            t = tally.setdefault(name, [0.0, 0.0])
            t[0] += frac * sec_km[gid]
            t[1] += sec_km[gid]
        print(f"{'km credited':>12}  {'of line':>9}  line")
        for name, (km, total) in sorted(tally.items(), key=lambda kv: -kv[1][0])[:15]:
            print(f"{km:>12.1f}  {total:>9.1f}  {name}")
        print(f"\n{len(tally)} other lines get some credit from riding this one")
        return

    print(f"{len(lines)} lines, {len(stations)} stations, "
          f"{sum(l['km'] for l in lines):,.0f} route-km\n")

    reg = [l for l in lines if l.get("src", "osm") != "osm"]
    if reg:
        print(f"{len(reg)} register lines, {sum(l['km'] for l in reg):,.0f} km"
              + (" (Japan's passenger network is about 27,300 km)" if args.region == "jp"
                 else "") + "\n")
        # A register that publishes its own chainage (km_official) can check every line, not
        # just the handful in REGISTER. It is the register's measure of the same track the
        # section walk used, so a ratio off 1.00 is the walk going wrong -- the double-track
        # doubling n02.py describes would show up here as 2.00.
        chk = [l for l in reg if l.get("km_official") and l["km"] >= 2]
        if chk:
            rs = sorted(l["km"] / l["km_official"] for l in chk)
            off = sorted((l for l in chk if abs(l["km"] / l["km_official"] - 1) > 0.05),
                         key=lambda l: l["km"] / l["km_official"])
            print(f"built against the register's own chainage, {len(chk)} lines of 2 km or "
                  f"more: median {rs[len(rs)//2]:.3f}, {len(off)} off by more than 5%")
            for l in off:
                print(f"  {l['km'] / l['km_official']:>5.2f}  {l['km']:>7.1f} of "
                      f"{l['km_official']:>7.1f}  {l['ref']:>8}  {l['name']}")
            print()
        print(f"{'register line':<16} {'built':>8} {'published':>10} {'ratio':>7}  note")
        worst_r = 0.0
        for frag, op, km, note in REGISTER.get(args.region, []):
            hits = [l for l in reg if frag == (l["name"] or "")
                    and (not op or op in (l["operator"] or ""))]
            if not hits:
                print(f"{frag:<16} {'NOT FOUND':>8}")
                worst_r = max(worst_r, 9.99)
                continue
            # With no operator given, SUM them: the register splits a line where it changes
            # hands, so 東海道線 is three companies and no one of them is the line.
            built = max(l["km"] for l in hits) if op else sum(l["km"] for l in hits)
            l = {"km": built}
            ratio = built / km
            worst_r = max(worst_r, abs(ratio - 1))
            flag = " <--" if abs(ratio - 1) > 0.05 else ""
            print(f"{frag:<16} {built:>8.1f}  {km:>9.1f}  {ratio:>6.2f}{flag}"
                  f"  {note}{'' if op else f'  [{len(hits)} operators]'}")
        print(f"\nworst register deviation {worst_r:.2f}\n")

    # OSM-derived objects only: with a register loaded, these names also match register
    # lines, and the two measure different things (the Yamanote LOOP against the Yamanote
    # register line) so comparing them to one published figure is meaningless.
    osm_lines = [l for l in lines if l.get("src", "osm") == "osm"] if reg else lines
    print(f"{'OSM line':<20} {'built':>8} {'published':>10} {'ratio':>7}  {'stops':>9}  note")
    worst = 0.0
    for frag, op, km, stops, note in KNOWN.get(args.region, []):
        hits = [l for l in osm_lines if frag in (l["name"] or "")
                and (not op or op in (l["operator"] or ""))]
        if not hits and reg:
            # An OSM line that was the register line twice over is dropped in the merge
            # (build_model.is_twin), so the register line is what there is to check.
            hits = [l for l in reg if frag in (l["name"] or "")
                    and (not op or op in (l["operator"] or ""))]
            note += " [register; OSM twin dropped]"
        if not hits:
            print(f"{frag:<20} {'NOT FOUND':>8}")
            worst = max(worst, 9.99)
            continue
        l = max(hits, key=lambda l: l["km"])
        ratio = l["km"] / km
        worst = max(worst, abs(ratio - 1))
        # Counted from the SECTIONS, not the display order: the display order is one variant
        # and so misses a branch's stations, which is how Marunouchi read 25 against 28.
        built = len({s for sec in l["sections"] for s in sec[:2]})
        s = f"{built}" + (f"/{stops}" if stops else "")
        flag = " <--" if abs(ratio - 1) > 0.05 else ""
        print(f"{frag:<20} {l['km']:>8.1f}  {km:>9.1f}  {ratio:>6.2f}  {s:>9}{flag}  {note}")
    print(f"\nworst deviation {worst:.2f}")


if __name__ == "__main__":
    main()
