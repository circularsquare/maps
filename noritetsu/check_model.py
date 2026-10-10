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
    # ---- the rest of Asia (asia_register.py; <cc>_sources.md; asia agent, 2026-10-08)
    "kh": [
        ("ខ្សែផ្លូវដែកភាគខាងជើង", "Royal", 273.0, "seat61: Phnom Penh - Battambang 273 km"),
        ("ខ្សែផ្លូវដែកភាគខាងត្បូង", "Royal", 263.0, "seat61: Phnom Penh - Sihanoukville 263 km "
                                                 "(en.WP 266 with the port spur)"),
        ("ខ្សែផ្លូវដែកភាគខាងជើង (បាត់ដំបង - ប៉ោយប៉ែត)", "Royal", 113.0, "en.WP Northern Line 386 km Phnom Penh - "
                                               "Poipet, less seat61's 273 to Battambang"),
    ],
    "la": [
        ("ທາງລົດໄຟ ລາວ-ຈີນ", "", 406.0, "en.WP Boten - Vientiane station table, plus Boten - "
                                         "border traced (~3 km, into the tunnel; 422 with the freight branch)"),
        ("สายชุมทางถนนจิระ–หนองคาย", "", 10.0, "Laos's part, under Thailand's line id: border "
                                            "- Thanaleng ~2.5 (crow-fly 2.6) + Thanaleng - "
                                            "Khamsavath 7.5 (en.WP)"),
    ],
    "ph": [
        ("South Main Line (Calamba - Lucena)", "", 77.0, "en.WP PNR South Main Line: the "
                                                          "Inter-Provincial Commuter, 77 km"),
        ("South Main Line (Naga - Legazpi)", "", 100.0, "en.WP Bicol Commuter Naga - Legazpi "
                                                        "about 100 km (greyed)"),
    ],
    # Mongolia: the main line is checked whole by asia_register's path check (border to
    # border 1,110 km, en.WP), not per half.
    "mn": [
        ("Салхит – Эрдэнэт", "", 164.0, "en.WP Erdenet branch, about 164 km"),
        ("Дархан – Шарын гол", "", 63.0, "survey (en.WP Rail transport in Mongolia), about 63"),
        ("Эрээнцав – Чойбалсан", "", 237.5, "en.WP 238 / ru.WP 237 from the border; built from "
                                             "Ereentsav station (greyed)"),
    ],
    # Myanmar: MR's mileposts as en.WP "List of railway stations in Myanmar" gives them (miles
    # from Yangon unless said), converted at 1.609 km.
    "mm": [
        ("Yangon–Mandalay line", "", 620.0, "en.WP Yangon–Mandalay Railway, 620 km"),
        ("Yangon–Mawlamyine line", "", 218.5, "Bago 46 1/2 - Mawlamyaing 182 1/4 mi"),
        ("Yangon–Pyay line", "", 259.1, "Pyay 161 mi"),
        ("Kyangin–Hinthada–Pathein line", "", 236.6, "Kyangin 174 1/4 - Hinthada 109 1/2 - "
                                                     "Pathein 191 3/4 mi (from Yangon by ferry)"),
        ("Thazi–Shwenyaung line", "", 157.7, "Thazi 306 - Kalaw 369 - Shwe Nyaung 404 mi"),
        ("Mandalay–Lashio line (Pyin Oo Lwin – Gokteik)", "", 65.2, "Pyin U Lwin 422 1/2 - "
                                                                   "Gokteik 463 mi"),
        ("Mandalay–Lashio line (Gokteik – Lashio)", "", 157.3, "Gokteik 463 - Lashio 560 3/4 mi"),
        ("Mandalay–Myitkyina line", "", 532.3, "Sagaing 392 - Myitkyina 722 3/4 mi"),
        ("Naba–Katha line", "", 24.1, "Naba 590 - Katha 605 mi"),
        ("Tanintharyi line", "", 307.0, "Mawlamyaing 182 1/4 - Dawei 373 mi"),
        ("Shwenyaung–Lawksawk line", "", 60.4, "Shwe Nyaung 404 - Lawksauk 441 1/2 mi"),
    ],
    "np": [
        ("जयनगर–जनकपुर–भंगाहा रेलमार्ग", "", 49.0, "en.WP Jaynagar - Bhangaha 52 km operational, "
                                                   "less ~3 km Jaynagar - border in India"),
    ],
    # ---- end the rest of Asia
    # ---- Latin America (latam_register.py; <cc>_sources.md; latam agent, 2026-10-08)
    # Costa Rica: INCOFER's GTFS shapes, each line's longest pattern.
    "cr": [
        ("San José - Heredia - Alajuela", "INCOFER", 20.83, "GTFS shape atlantico_alajuela"),
        ("San José - Cartago", "INCOFER", 24.62, "GTFS shape atlantico_paraiso"),
        ("Curridabat - San José - Pavas - Belén", "INCOFER", 21.9,
         "GTFS shape cfia_metropoli (runs CFIA - Belén)"),
    ],
    # Cuba: es.WP "Ferrocarriles de Cuba", the Línea Central La Habana - Santiago.
    "cu": [
        ("Línea Central: La Habana – Santiago de Cuba", "", 835.0, "es.WP"),
    ],
    # Peru (pe_sources.md).
    "pe": [
        ("Ferrocarril Huancayo – Huancavelica (Chilca – Cuenca)", "", 57.0,
         "ProActivo, Dec 2024: phase one Chilca - Cuenca 57 km"),
        ("Ferrocarril Huancayo – Huancavelica (Cuenca – Huancavelica)", "", 71.0,
         "en.WP 128 km for the whole line less Chilca - Cuenca's 57"),
    ],
    # Bolivia (bo_sources.md): Ferroviaria Oriental's sectors, es.WP.
    "bo": [
        ("Ferrocarril Santa Cruz – Puerto Quijarro", "", 651.0, "FO sector Este"),
        ("Ferrocarril Santa Cruz – Yacuiba", "", 539.0, "FO sector Sur, to Pocitos (~3 km "
                                                         "past Yacuiba)"),
    ],
    # Colombia (co_sources.md): en.WP "Medellín Metro". The Sabana train has no published
    # Usaquén - Zipaquirá length (built from Usaquén: OSM's track stops short of La Sabana).
    "co": [
        ("Tranvía de Ayacucho", "", 4.2, "en.WP, San Antonio - Oriente"),
    ],
    # Venezuela (ve_sources.md): es.WP "Sistema Ferroviario Ezequiel Zamora".
    "ve": [
        ("Sistema Ferroviario Ezequiel Zamora: Caracas – Cúa", "", 41.4, "es.WP"),
    ],
    # Uruguay: no published Tacuarembó - Rivera length (gub.uy's "100 km" is rounded). AFE's
    # request halts are km posts and are stations of the line: km 457 to km 552 builds 95.6
    # km for 95 (1.01), section by section within 1 km (uy_sources.md).
    # Ecuador (ec_sources.md).
    "ec": [
        ("Tren Nariz del Diablo", "", 12.5, "Primicias; built 9.2, ~0.74: the switchback's "
                                            "reversing tails are not on a shortest path "
                                            "(OSM's route ways are 11.2 km)"),
    ],
    # ---- end Latin America
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
                              "both halves 경강선, built as two lines (여주-원주 not built), "
                              "summed here"),
        ("중부내륙선", "", 56.9, "부발-충주, YB23 영업거리; 충주-문경 opened 2024-12, "
                                "so expect about 1.7 if OSM includes it"),
        ("서해선", "", 38.5, "대곡-원시, YB23 영업거리; OSM also names the 2024 "
                              "홍성-서화성 intercity line 서해선, which is not in the figure "
                              "(built as two lines, one operator, so summed: expect ~2.7)"),
        ("분당선", "", 52.9, "왕십리-수원, YB23 영업거리"),
        ("수인선", "", 51.6, "수원-인천 거리표, with the 한대앞-오이도 section shared with "
                              "안산선 (since 2026-10-04 a borrowed section, crediting 안산선); "
                              "YB23 영업거리 38.8 without it"),
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
        ("인천 도시철도 1호선", "", 37.0, "검단호수공원-송도달빛축제공원, KRIC 1294 (its "
                                        "km to the neighbour, summed); YB23 영업거리 30.3 is "
                                        "계양-송도달빛축제공원, before the 2025-06 검단 extension"),
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
        ("廣深港高速鐵路", "", 26.0, "香港西九龍-border (the HK section), en.wikipedia; to the "
                                     "platform ends"),
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
    # Register lines as my_register.LINES names them. Sources in my_sources.md:
    #   RD    en.wikipedia's route diagram {{KTM West Coast Line}}: chainage from Butterworth
    #         (Bukit Mertajam 10.2, Ipoh 181.0, KL Sentral 388.0, Gemas 562.2, JB Sentral
    #         756.8) and from Bukit Mertajam north (Padang Besar 157.8); it predates the
    #         2014 Ipoh - Padang Besar and 2025 Gemas - JB double-track realignments
    #   WP    en.wikipedia line infoboxes and text (2026-10-03)
    # The build runs station centre to station centre; metro figures run to the track ends.
    "my": [
        ("Laluan Pantai Barat", "", 906.1, "RD: Padang Besar - Bukit Mertajam 157.8 + Bukit "
                                          "Mertajam - JB Sentral 746.6, plus the build's "
                                          "1.7 km of border tails; realignments: ~0.99"),
        ("Laluan Pantai Timur", "", 526.0, "Gemas - Tumpat, WP and Wikivoyage; 527.75 also "
                                          "quoted"),
        ("Laluan Cawangan Butterworth", "", 10.2, "RD, Bukit Mertajam - Butterworth"),
        ("Laluan Cawangan Skypark", "", 10.9, "WP's KL Sentral - Terminal Skypark 26 km less "
                                             "the build's KL Sentral - Subang Jaya 15.1; WP's "
                                             "26 is rounded: ~0.83. Suspended since 2023-02"),
        ("Laluan Keretapi Barat Sabah", "", 134.0, "Tanjung Aru - Tenom, WP Sabah State "
                                                  "Railway"),
        ("Laluan Kelana Jaya", "", 46.4, "WP, 37 stations"),
        ("Laluan Kajang", "", 47.0, "WP, 29 stations"),
        ("Laluan Putrajaya", "", 57.7, "WP, 36 stations; tail tracks at both ends: ~0.97"),
        ("Laluan Shah Alam", "", 37.8, "WP, Bandar Utama - Johan Setia, 20 stations open of 25"),
        ("Laluan Monorel KL", "", 8.6, "WP, 11 stations"),
        ("Laluan Sri Petaling", "", 38.1, "Sentul Timur - Putra Heights: WP's network 45.1 "
                                         "less the Ampang Line's own Chan Sow Lin - Ampang "
                                         "(the build's 7.0); network built 43.9 of 45.1: "
                                         "~0.97"),
        ("KLIA Transit", "", 59.1, "WP's 57 km KL Sentral - KLIA T1 (built 56.1) plus the "
                                  "build's own 2.5 km on to KLIA T2: ~0.99"),
        ("Keretapi Bukit Bendera", "", 1.996, "WP, along the slope; it climbs about 700 m, so "
                                             "about 1.87 on the map: ~0.94"),
    ],
    # Israel (il_register.py; il_sources.md "Build (2026-10-08)"). en.wikipedia (WP) gives few
    # lengths, mostly rounded; a branch is built from where it leaves the line before it, so
    # the note says what the published figure counts beyond that.
    "il": [
        ("מסילת תל אביב–ירושלים", "", 56.0, "WP 'about 56 km' from the Ganot interchange; the "
                                          "build has Ganot - Navon 46.9 and the Modi'in branch "
                                          "6.8 apart: ~0.84 (0.96 with it)"),
        ("מסילת לוד–אשקלון", "", 50.0, "WP 'approximately 50 km' Lod - Ashkelon; the build "
                                     "starts 0.8 km past Lod, Lod - Ashkelon is 40 km crow-fly: "
                                     "~0.83"),
        ("מסילת אשקלון–באר שבע", "", 60.0, "WP 'approximately 60 km' Ashkelon - Be'er Sheva North "
                                         "(its table says 70)"),
        ("המסילה לבאר שבע", "", 87.0, "WP: the doubling project Lod - Be'er Sheva Center was "
                                    "87 km; the build's Lod - Na'an junction is the Jaffa - "
                                    "Jerusalem line's 10.6: ~0.87"),
        ("מסילת העמק", "", 60.0, "he.WP Haifa - Beit She'an 60 km; the build is from its "
                                "junction by HaMifrats Central: ~0.95"),
        ("מסילת עכו–כרמיאל", "", 23.0, "he.WP Acre - Karmiel 23 km; the build starts at the "
                                     "junction 2.5 km south of Acre: ~0.90"),
        ("מסילת יפו–ירושלים: בית שמש–ירושלים מלחה", "", 36.3, "WP's station table: Beit Shemesh 50.3, Malha 86.6 "
                                           "km from Jaffa (the Ottoman line, before the 2005 "
                                           "realignments): ~0.86. Greyed"),
        ("הרכבת הקלה בירושלים – הקו האדום", "", 22.5, "WP, 35 stations; built stop to stop, "
                                                     "no tail or depot track: ~0.89"),
        ("הרכבת הקלה בירושלים – הקו הירוק", "", 7.0, "WP, Malha - HaTurim, opened 2026-08-21, "
                                                    "13 stations"),
        ("הקו האדום", "תבל", 24.0, "WP, 34 stations incl. the Kiryat Arye branch"),
        ("כרמלית", "", 1.8, "WP, 6 stations"),
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
        ("成昆线", "", 1100.0, "成都-昆明 operating length, WP 成昆铁路; OSM's 成昆线 is the "
                             "old line's two pieces, 成都南-花棚子 and 元谋西-昆明, the new "
                             "line 峨广线 between; built as two lines since 2026-10-04 "
                             "(cn_register.NO_BRIDGE), summed here: expect about 0.91"),
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
    # Register lines as rinf_countries/se.py names them (Trafikverket's stråk names, split where
    # a stråk holds another named line). Figures are sv.wikipedia line articles' infobox
    # `längd` (WP), raw wikitext retrieved 2026-10-02. Where the built line is not the
    # article's extent, the note says by how much; se_sources.md has end-to-end checks along
    # the built lines (Stockholm C - Göteborg C 452.3 for 455, and so on).
    "se": [
        ("Västra stambanan", "", 455.0, "WP Stockholm C - Göteborg C; built also has the old "
                                        "line Flemingsberg - Södertälje - Järna (pendeltåg, "
                                        "31 km), Södertälje C and Hallsberg rangerbangård - "
                                        "Skymossen, so expect 1.06"),
        ("Södra stambanan", "", 483.0, "WP Katrineholm - Malmö"),
        ("Västkustbanan", "", 283.0, "WP Göteborg - Lund; built also has Ängelholm - Åstorp "
                                     "(14 km, stråk 3), so expect 1.04"),
        ("Kust till kust-banan", "", 406.0, "WP text: Göteborg - Kalmar ~350 + Emmaboda - "
                                            "Karlskrona 56; Göteborg C - Almedal is "
                                            "Västkustbanan's"),
        ("Ostkustbanan", "", 400.0, "WP Stockholm - Sundsvall"),
        ("Dalabanan", "", 265.0, "WP Uppsala - Mora"),
        ("Stambanan genom övre Norrland", "", 626.0, "WP Bräcke - Boden; Bräcke - Vännäs has no "
                                                     "passenger trains (WP, and none in the "
                                                     "feed), so expect 0.45"),
        ("Norra stambanan", "", 267.0, "WP Gävle - Ockelbo 38 + Ockelbo - Ånge 229"),
        ("Godsstråket genom Bergslagen", "", 311.0, "WP Storvik - Mjölby; Hovsta - Örebro "
                                                    "(8 km) is built as Mälarbanan"),
        ("Bergslagsbanan", "", 337.0, "WP Kil - Gävle; built also has Frövi - Ställdalen (63 km "
                                      "in RINF), so expect 1.18"),
        ("Norge/Vänerbanan", "", 300.0, "WP Göteborg - Kil and Skälebol - Kornsjö"),
        ("Värmlandsbanan", "", 202.0, "WP Laxå - Charlottenberg"),
        ("Jönköpingsbanan", "", 112.0, "WP"),
        ("Älvsborgsbanan", "", 133.0, "WP Uddevalla - Borås"),
        ("Mälarbanan", "", 187.0, "WP Stockholm C - Hovsta; built from Tomteboda to Örebro C, "
                                  "so expect 1.06"),
        ("Svealandsbanan", "", 115.0, "WP Södertälje - Valskog"),
        ("Nynäsbanan", "", 55.0, "WP"),
        ("Mittbanan", "", 358.0, "WP Sundsvall - Storlien"),
        ("Malmbanan", "", 473.0, "WP Luleå - Narvik; Riksgränsen - Narvik is Norway's "
                                 "Ofotbanen (43 km), so expect 0.92"),
        ("Botniabanan", "", 185.0, "WP"),
        ("Haparandabanan", "", 159.0, "WP Boden - Haparanda; Boden - Buddbyn is Malmbanan's"),
        ("Ådalsbanan", "", 175.0, "WP Sundsvall - Långsele; Västeraspby - Långsele has no "
                                  "passenger trains (WP), so expect 0.69"),
        ("Blekinge kustbana", "", 130.0, "WP Kristianstad - Karlskrona; Gullberna - Karlskrona "
                                         "is Kust till kust-banan's"),
        ("Bohusbanan", "", 180.0, "WP"),
        ("Kinnekullebanan", "", 121.0, "WP"),
        ("Stångådalsbanan", "", 235.0, "WP Linköping - Kalmar; built also has Berga - "
                                       "Oskarshamn (28 km, stråk 65), so expect 1.11"),
        ("Tjustbanan", "", 116.0, "WP Linköping - Västervik; Linköping - Bjärka-Säby (20 km) "
                                  "is Stångådalsbanan's, so expect 0.82"),
        ("Nyköpingsbanan", "", 109.0, "WP"),
        ("Skånebanan", "", 106.0, "WP Helsingborg - Kristianstad"),
        ("Ystadbanan", "", 113.0, "WP Malmö - Ystad via Citytunneln 67 + Österlenbanan "
                                  "Ystad - Simrishamn 46 (stråk 90 is both); Malmö C - "
                                  "Hyllie is Citytunneln's"),
        ("Fryksdalsbanan", "", 82.0, "WP"),
        ("Viskadalsbanan", "", 84.0, "WP"),
        ("Rååbanan", "", 45.0, "WP"),
        ("Vaggerydsbanan", "", 38.0, "WP"),
        ("Citybanan", "", 6.0, "WP tunnel"),
        ("Inlandsbanan", "", 1288.0, "WP Kristinehamn - Gällivare; Kristinehamn - Mora is not "
                                     "in RINF (no trains), Brunflo - Östersund is Mittbanan's"),
    ],
    # Register lines as no_register.py names them: Bane NOR's banestrekninger (Banenettverk's
    # banenavn). WP is no.wikipedia's infobox `lengde` (raw wikitext, 2026-10-03), JiN the
    # table in no.wikipedia's "Jernbane i Norge", BN Bane NOR's own chainage at the line's two
    # ends in Banenettverk (startposisjon/sluttposisjon, the register's kilometrering). Lines
    # are built stop to stop on Bane NOR's centre line, so a line whose end station is not at
    # its chainage end reads short by that much (noted).
    "no": [
        ("Nordlandsbanen", "", 729.0, "WP Trondheim S - Bodø"),
        ("Dovrebanen", "", 482.8, "BN km 70.9 - 553.7, Eidsvoll - Trondheim S (WP says 492)"),
        ("Sørlandsbanen", "", 549.0, "WP Drammen - Stavanger; Drammen - Hokksund's first km "
                                     "lie on Drammenbanen's centre line"),
        ("Rørosbanen", "", 382.0, "JiN Hamar - Støren (WP infobox 348)"),
        ("Bergensbanen", "", 380.6, "BN km 90.6 - 471.2, Hønefoss - Bergen; Hønefoss station "
                                    "lies on Roa-Hønefossbanen's centre line"),
        ("Østfoldbanen vestre linje", "", 170.0, "WP Oslo S - Kornsjø border"),
        ("Vestfoldbanen", "", 129.0, "WP Drammen - Skien (ca.)"),
        ("Gjøvikbanen", "", 123.8, "WP Oslo S - Gjøvik"),
        ("Raumabanen", "", 114.2, "WP Dombås - Åndalsnes"),
        ("Kongsvingerbanen", "", 113.4, "BN km 22.9 - 136.3, Lillestrøm - border"),
        ("Meråkerbanen", "", 71.5, "BN km 30.7 - 102.2, Hell - border (WP's 106 runs on to "
                                   "Storlien)"),
        ("Ofotbanen", "", 43.0, "WP Narvik - border; Narvik station lies about 4 km short of "
                                "the line's km 0 at the ore harbour, so expect about 0.9"),
        ("Flåmsbana", "", 20.2, "WP Myrdal - Flåm"),
        ("Arendalsbanen", "", 36.3, "BN km 281.7 - 318.0, Nelaug - Arendal"),
        ("Follobanen", "", 22.0, "BN km 1.0 - 22.9 with the approach to Oslo S; built from "
                                 "where it leaves Østfoldbanen's tracks, so expect about 0.92"),
    ],
    # Register lines as rinf_countries/dk.py names them (da.wikipedia's article titles).
    # Figures are da.wikipedia's infobox `linjelængde` (WP), raw wikitext retrieved 2026-10-03,
    # else Wikidata's length (P2043, WD). RINF's own Danish lengths leave out the station
    # areas, so there is no chainage check (rinf.py `km_floor`); these are the outside numbers.
    "dk": [
        ("Vestbanen", "", 111.0, "WP København - Korsør"),
        ("København-Køge-Ringsted-banen", "", 60.0, "WP; built from København H via "
                                                     "København G and Vigerslev (about 4 km) "
                                                     "with the Hvidovre curve, so expect 1.07"),
        ("Sydbanen", "", 142.3, "WP Ringsted - Rødby Færge and Gedser; built Ringsted - "
                                "Nykøbing F (Gedser is Gedserbanen, Rødby not in RINF), so "
                                "expect about 0.6"),
        ("Lille Syd", "", 61.4, "WD Roskilde - Køge - Næstved"),
        ("Nordvestbanen", "", 79.3, "WP Roskilde - Kalundborg"),
        ("Kystbanen", "", 46.0, "WP København H - Helsingør"),
        ("Lille Nord", "", 24.4, "WP Hillerød - Helsingør; Snekkersten - Helsingør (about "
                                 "3.5 km) is Kystbanen's, so expect about 0.86"),
        ("Svendborgbanen", "", 46.8, "WP"),
        ("Den fynske hovedbane", "", 88.57, "WP Nyborg - Fredericia"),
        ("Fredericia-Aarhus-banen", "", 108.0, "WP"),
        ("Aarhus-Randers-banen", "", 59.2, "WP"),
        ("Randers-Aalborg Jernbane", "", 80.7, "WP"),
        ("Vendsysselbanen", "", 80.7, "WP Aalborg - Frederikshavn; built 84.5 (1.05)"),
        ("Fredericia-Vamdrup-banen", "", 38.9, "WP"),
        ("Vamdrup-Padborg-banen", "", 71.7, "WD Fredericia - Padborg 110.6 less WP Fredericia "
                                            "- Vamdrup 38.9; built to the border"),
        ("Sønderborgbanen", "", 42.0, "WP text, Tinglev - Sønderborg"),
        ("Bramming-Tønder-banen", "", 67.9, "WP to the border; built to Tønder (Tønder - "
                                            "border is not in RINF), so expect about 0.95"),
        ("Den vestjyske længdebane", "", 146.0, "WP Esbjerg - Struer"),
        ("Langå-Struer-banen", "", 102.4, "WP"),
        ("Vejle-Holstebro-banen", "", 115.0, "WD"),
        ("Thybanen", "", 73.6, "WP Struer - Thisted"),
        ("Skanderborg-Skjern-banen", "", 111.9, "WP"),
        ("Varde-Nørre Nebel Jernbane", "", 37.6, "WP"),
        ("Hirtshalsbanen", "", 17.9, "WP"),
        ("Skagensbanen", "", 39.7, "WP"),
        ("Nærumbanen", "", 7.8, "WP"),
        ("Frederiksværkbanen", "", 39.7, "WP Hillerød - Hundested"),
        ("Hornbækbanen", "", 25.0, "WP Helsingør - Gilleleje"),
        ("Gribskovbanen", "", 50.6, "WP, both branches; Hillerød - Gilleleje and Kagerup - "
                                    "Tisvildeleje trace 42 km on OSM's track and RINF's points, "
                                    "and the article gives no split: expect 0.83 until a "
                                    "source says what the 50.6 counts"),
        ("Odsherredsbanen", "", 49.6, "WP"),
        ("Østbanen", "", 49.6, "WP, both branches"),
        ("Lollandsbanen", "", 50.2, "WP"),
        ("Nordbanen", "", 36.5, "WP København H - Hillerød (S-bane)"),
        ("Klampenborgbanen", "", 5.5, "WP Hellerup - Klampenborg (S-bane)"),
        ("Høje Taastrup-banen", "", 19.5, "WP sporlængde, København H - Høje Taastrup"),
        ("Køge Bugt-banen", "", 39.0, "WP sporlængde, København H - Køge"),
        ("Storebæltsforbindelsen", "", 17.0, "WP: the link is about 17 km; built Korsør - "
                                             "Nyborg station to station, so expect about 1.37"),
    ],
    # Register lines as rinf_countries/ie.py names them (en.wikipedia's article titles). The
    # figures are Iarnród Éireann's 2022 Network Statement as en.wikipedia's infoboxes quote it
    # (WP, raw wikitext retrieved 2026-10-03), miles converted. RINF's own Irish section
    # lengths are too often wrong to check against (rinf.py `no_chain`); these are the check.
    "ie": [
        ("Dublin–Cork line", "", 266.75, "WP Heuston - Cork Kent"),
        ("Dublin–Sligo line", "", 216.05, "WP Connolly - Sligo; built from Connolly Junction "
                                          "by Newcomen Junction with the 0.8 km Docklands spur"),
        ("Dublin–Rosslare line", "", 167.97, "WP 104 3/8 mi, Connolly - Rosslare Europort"),
        ("Dublin–Galway line", "", 141.46, "WP Portarlington - Athlone 63 + Athlone - Galway "
                                           "78.46"),
        ("Dublin–Westport line", "", 133.374, "WP Athlone - Westport; built from Athlone West "
                                              "Junction, 1 km on"),
        ("Dublin–Waterford line", "", 122.8, "WP Kildare - Waterford 119 + the Kilkenny spur "
                                             "3.8; built from Cherryville Junction, 3.9 km past "
                                             "Kildare, with the Lavistown triangle"),
        ("Limerick–Rosslare line", "", 123.1, "WP Limerick - Waterford, the operational part; "
                                              "built to Dunkitt Junction (the last 2.7 km are "
                                              "the Dublin–Waterford line's), so expect 0.98"),
        ("Limerick–Ballybrophy line", "", 84.49, "WP 52.5 mi"),
        ("Mallow–Tralee line", "", 98.97, "WP 61.5 mi, with the Killarney reversal"),
        ("Western Railway Corridor", "", 97.16, "WP 60 3/8 mi, Limerick - Athenry, the "
                                                "operational part"),
        ("Ballina branch", "", 33.19, "WP 20 5/8 mi, Manulla Junction - Ballina"),
        ("Great Northern Railway Main Line", "", 95.76, "WP: the border is at milepost 59 1/2 "
                                                        "from Connolly"),
        ("Glounthaune–Midleton line", "", 10.0, "WP Cork Suburban Rail: 10 km, opened 2009"),
        ("Dublin–Navan line", "", 7.5, "WP: 7.5 km from the junction west of Clonsilla to M3 "
                                       "Parkway"),
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
    # Ukraine: tariff sections (ua_register.py), named by their two ends as shown. The tariff
    # km is every line's own chainage (km_official); these are the outside figures, Wikidata
    # P2043 of the line's item (retrieved 2026-10-03, data/raw/ua/wd_lines.json). Nearly all
    # of Wikidata's 74 Ukrainian lengths are corridors or narrow gauge, not tariff sections.
    "ua": [
        ("Батьово — Королево", "", 68.0, "WD Q801907; tariff 68"),
        ("Стрий — Івано-Франківськ", "", 108.0, "WD Q15856795; tariff 108"),
        ("Антонівка — Зарічне (взк.)", "", 106.6, "WD Q801847, the narrow-gauge line; tariff 97"),
    ],
    # Georgia, Armenia, Azerbaijan (caucasus_register.py; tariff sections named by their ends).
    # de.WP = de.wikipedia's route diagrams' chainage, "Bahnstrecke Poti–Baku" and
    # "Bahnstrecke Tiflis–Jerewan" (2026-10-03); WD = Wikidata P2043. Lines ending at a border
    # point add Book 1's km from the station to its export code there.
    "ge": [
        ("სამტრედია-1 — ხაშური", "", 124.0, "de.WP, Samtredia-1 244.0 - Khashuri 120.0"),
        ("ხაშური — ნავთლუღი", "", 125.7, "de.WP, Khashuri 120.0 - Tbilisi 0 - Navtlughi 5.7"),
        ("სენაკი — სამტრედია-1", "", 28.1, "de.WP, Senaki 272.1 - Samtredia-1 244.0"),
        ("ნატანები — ოზურგეთი", "", 20.0, "WD Q135997954; tariff 18"),
    ],
    "am": [
        ("Գյումրի — Մասիս", "", 140.2, "de.WP, Gyumri 211.0 - Masis 351.2"),
        ("Այրում — Գյումրի", "", 143.1, "de.WP, Ayrum 71.9 - Gyumri 211.0, + 4 to the border"),
    ],
    "az": [
        ("Hacıqabul — Böyük Kəsik", "", 375.1, "de.WP, Beük-Kasik 46.1 - Hacıqabul 417.2, "
                                               "+ 4 to the border"),
        ("Baş Ələt — Hacıqabul", "", 42.9, "de.WP, Ələt 460.1 - Hacıqabul 417.2"),
        ("Yevlax — Balakən", "", 165.0, "WD Q141328728; tariff 163"),
    ],
    # Abkhazia (caucasus_register.py; the Abkhazian part of Book 1's 57-001, built from the
    # Psou bridge to Ochamchira, where OSM's track ends). ru.WP = ru.wikipedia's route diagram
    # in "Абхазская железная дорога" (2026-10-04): Psou 1998.0, Ochamchyra 2153.0. Psou
    # platform is 0.4 km east of the border point the line starts at.
    "xa": [
        ("Ԥсоу — Очамчыра", "", 155.0, "ru.WP, Psou 1998.0 - Ochamchyra 2153.0; tariff 157"),
    ],
    # Belarus: the tariff guide's Belarusian Railway sheet, as Ukraine (by_sources.md). Outside
    # figures: Wikidata P2043 of pl.wikipedia's line articles (retrieved 2026-10-03); the
    # Polish-border lines' figures run on to the border, where the build stops at the last
    # station before it.
    "by": [
        ("Гродна — Масты", "", 58.146, "WD Q73919142 (pl.WP); tariff 58"),
        ("Гродна — Брузгі", "", 21.755, "WD Q56315886 (pl.WP); tariff 23 to the border"),
        ("Ліда — Беняконі", "", 42.647, "WD Q79121736 (pl.WP); tariff 45 to the border"),
        ("Ліда — Баранавічы Цэнтральныя", "", 105.504, "WD Q79121797 (pl.WP); tariff 114"),
        ("Варапаева — Друя", "", 88.9, "WD Q109642867 (pl.WP); tariff 89"),
        ("Брэст-Цэнтральны — Высока-Літоўск", "", 48.408,
         "WD Q60863027 (pl.WP), to the border; built to Vysokaye station: expect ~0.85"),
        ("Брэст-Палескі — Хаціслаў", "", 57.568,
         "WD Q66331616 (pl.WP), to the border; built to Khotislav station: expect ~0.87"),
    ],
    # Moldova: the tariff guide's CFM sheet, as Ukraine (md_sources.md). No Wikidata or
    # Wikipedia length exists for any CFM section; CFM's timetable km (merstren.md, train
    # 826Г: Ungheni km 0, Chișinău km 107) is the one outside figure.
    "md": [
        ("Ungheni — Chișinău", "", 107.0, "CFM timetable km (826Г, merstren.md); tariff 107"),
    ],
    # Italy: RFI publishes no per-line length table that is open (the PIR's line list is on
    # the ePIR portal). CH = it.wikipedia's station table (Percorso) chainage, differenced
    # between the line's two ends as built; WP = it.wikipedia infobox `lunghezza`; WD =
    # Wikidata P2043 of the it.wikipedia article's item. Retrieved 2026-10-02/03. RFI's F lines
    # run node to node (Milano - Bologna is Rogoredo - Lavino); since 2026-10-03 the city nodes
    # are split into the lines inside them (rinf_countries/it.py NODE_SPLIT), so most F lines
    # now reach their city stations and their figures are the chainage between the built
    # ends. it_sources.md.
    "it": [
        # fundamental lines (CH)
        ("Brennero – Verona", "", 238.71, "CH Verona Porta Nuova 0 - Brennero 238.711"),
        ("Firenze – Roma (Direttissima)", "", 249.50, "CH Firenze Rovezzano 254.004 - Roma "
                                                      "Tiburtina 4.505 (Settebagni - Tiburtina "
                                                      "from the Nodo di Roma); built also has "
                                                      "the Chiusi and Valdarno interconnections "
                                                      "(15.8 km, kept by the timetable): ~1.06"),
        ("Firenze – Roma (linea lenta)", "", 314.0, "WP, Firenze SMN - Roma Termini (both city "
                                                    "ends from the Nodi di Firenze and Roma)"),
        ("Milano – Bologna", "", 214.54, "CH Milano Rogoredo 208.751 + Lambrate - Rogoredo "
                                         "5.79 (RINF, from the Nodo di Milano) - Bologna "
                                         "Centrale 0; RINF also files the Rogoredo - Bivio "
                                         "Melegnano - Tavazzano pair (18.2 km) under F41-F42: "
                                         "~1.08"),
        ("Bologna – Ancona", "", 204.0, "CH Bologna Centrale 0 - Ancona 203.996"),
        ("Ancona – Foggia", "", 322.03, "CH Ancona 203.996 - Foggia 526.027"),
        ("Orte – Ancona", "", 211.63, "CH Orte 82.503 - Falconara 285.429 = 195.299, - "
                                      "Ancona 203.996"),
        ("Fiumetorto – Messina", "", 180.55, "CH Fiumetorto 43.219 - Messina Centrale 223.764; "
                                             "built has both the old coast line via Falcone "
                                             "and the Patti - Terme Vigliatore line (18 km), "
                                             "as RINF does: ~1.10"),
        ("Catanzaro Lido – Reggio Calabria", "", 177.55, "CH Jonica 294.720 - 472.270"),
        ("Sibari – Catanzaro Lido", "", 172.48, "CH Jonica 122.237 - 294.720"),
        ("Genova – Pisa", "", 158.49, "CH Genova Piazza Principe 0 - La Spezia Centrale 86.162 "
                                      "= 172.462 - Pisa San Rossore 100.133; built has both "
                                      "Vezzano - La Spezia routes (Migliarina, Cà di "
                                      "Boschetti) and Pisa Centrale: ~1.06"),
        ("Venezia – Trieste", "", 141.10, "CH Venezia Carpenedo 3.904 - 131.315 = 13.687 - "
                                          "Trieste Centrale 0; built also has Bivio d'Aurisina "
                                          "- Villa Opicina (14.9 km, the Trieste - Ljubljana "
                                          "trains, kept by the timetable) and Mestre Olimpia - "
                                          "Carpenedo (1.9): ~1.13"),
        ("Milano – Torino", "", 118.81, "CH Settimo 15.763 - Rho 134.571"),
        ("Torino – Arquata Scrivia", "", 123.13, "CH Torino Porta Nuova 0 - Arquata Scrivia "
                                                 "123.132"),
        ("Verona – Bologna", "", 110.79, "CH Verona Porta Nuova 114.951 - PM Santa Viola ~4.16 "
                                         "(RINF Bologna Centrale - S.Viola 4.158); built also "
                                         "has the Verona Porta Vescovo leg (6.8 km): ~1.06"),
        ("Bologna – Padova", "", 122.92, "CH Bologna Centrale 0 - Padova 122.921"),
        ("Bologna – Firenze (Direttissima)", "", 97.0, "WD; built runs Bologna San Vitale - "
                                                       "Firenze SMN with the Castello - "
                                                       "Olmatello link (1.9 km): ~1.08"),
        ("Bologna – Porretta Terme", "", 54.31, "CH Porrettana, Santa Viola 127.676 - "
                                                "Porretta Terme 73.367 (the Nodo di Bologna's "
                                                "part of the Porrettana)"),
        ("Roma – Avezzano", "", 107.08, "CH Roma Termini 0 - Avezzano 107.080; built also has "
                                        "Tiburtina - Prenestina (2.9 km)"),
        ("Alessandria – Piacenza", "", 96.51, "CH; built also has the Bressana Bottarone - "
                                              "Barbianello - Broni leg towards Pavia (13.3 "
                                              "km) and Alessandria Smistamento: ~1.15"),
        ("Messina – Catania", "", 94.77, "CH Catania Centrale 240.714 - Messina Centrale "
                                         "335.485"),
        ("Palermo – Fiumetorto", "", 43.22, "CH"),
        # high speed
        ("AV Torino – Milano", "", 125.0, "WD (WP 127); built also has the Novara "
                                          "interconnection (4.0 km, kept by the timetable) and "
                                          "the Rho and Torino Stura links (4.4): ~1.07"),
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
        ("Milano – Mortara", "", 44.0, "WD; built also has the southern belt Milano Rogoredo "
                                       "- Romolo - San Cristoforo (9.8 km, from the Nodo di "
                                       "Milano), the S9's way into the city: ~1.23"),
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
         "catalogue; Chinchilla - Hellín (no OSM route) kept by the timetable check since "
         "2026-10-03 (es.py CUT_AT)"),
        ("422 Bifurcación Utrera – Fuente de Piedra", "", 113.4,
         "catalogue; Arahal - Bif. Utrera kept by the timetable check since 2026-10-03"),
        ("500 Bifurcación Planetario – Bifurcación Casa de la Torre", "", 322.7,
         "catalogue; Cañaveral - Bif. Casa de la Torre kept by the timetable check since "
         "2026-10-03"),
        ("984 Pola de Lena – Bifurcación Pajares", "", 49.3,
         "catalogue; the Pajares base tunnel, kept by the timetable check since 2026-10-03"),
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
    # FRA NARN subdivisions (us_register.py), against Wikipedia's lengths and mileposts (WP).
    # NARN's own KM is the chainage check above, for every line; these are the outside
    # numbers. The build log's "path check" lines test the network as a whole the same way:
    # Washington - Boston over NARN against the Northeast Corridor's 735 km (457 mi).
    "us": [
        # No operator, so its pieces are summed: the Michigan Line runs over CN's track through
        # Battle Creek, and us_register.split_pieces makes each side a line of its own.
        ("Michigan Line", "", 373.0, "Porter - Dearborn, WP 232 mi: Amtrak's 98 mi and "
                                     "MDOT's 135 mi, both filed AMTK in NARN"),
        ("Hartford Line", "Amtrak", 100.0, "New Haven - Springfield, WP 62 mi"),
        ("Keystone Corridor", "Amtrak", 168.3, "Philadelphia 30th St - Harrisburg, WP MP 104.6; "
                                               "NARN's line starts at Zoo, 3-4 km out"),
        ("New Haven Line", "Metro-North", 97.4, "Woodlawn MP 11.8 - New Haven MP 72.3, WP"),
        ("Main Line", "Long Island", 151.8, "Long Island City - Greenport, WP MP 94.3"),
        ("Port Jefferson Branch", "Long Island", 52.6, "Hicksville MP 24.8 - Port Jefferson "
                                                       "MP 57.5, WP"),
        ("Montauk Branch", "Long Island", 171.9, "Jamaica MP 9.0 - Montauk MP 115.8, WP; "
                                                 "NARN's branch starts at Jamaica"),
        ("Anchorage Subdivision", "Alaska", 573.0, "Anchorage - Fairbanks, WP 356 mi"),
        ("Seward Subdivision", "Alaska", 183.0, "Seward - Anchorage, WP 114 mi"),
    ],
    # FRA NARN subdivisions in Canada (ca_register.py), against Wikipedia's km posts (WP).
    # NARN's KM is the chainage check for every line; ca_register's path checks (build log)
    # test the network: Toronto - Montréal 539, Toronto - Vancouver 4,466 and others.
    "ca": [
        ("Wekusko Subdivision", "Hudson Bay", 219.0, "The Pas km 0 - Wabowden km 219, WP route "
                                                     "diagram (Template:Hudson Bay Railway)"),
        ("Thicket Subdivision", "Hudson Bay", 305.0, "Wabowden km 219 - Gillam km 524, WP"),
        ("Herchmer Subdivision", "Hudson Bay", 296.0, "Gillam km 524 - Churchill km 820, WP"),
        ("Island Falls Subdivision", "Ontario Northland", 299.0, "Cochrane - Moosonee, the "
                                                                 "Polar Bear Express, WP 186 mi"),
        ("Alexandria Subdivision", "VIA", 123.0, "Ottawa km 446 - Coteau Junction km 569, WP "
                                                 "route diagram (Template:Via Corridor routing)"),
        ("Kingston Subdivision", "Canadian National", 483.0, "Dorval - Pickering, WP "
                                                             "'approximately 300 miles'"),
    ],
    # Geoscience Australia's Foundation Rail Infrastructure line names (au_register.py),
    # against Wikipedia (WP). A line is built only where passenger trains run, so the figure
    # is the line's passenger extent. GA's own length is the chainage check above (it is the
    # length of GA's geometry, so it checks the tracing more than the register). The build
    # log's "path check" lines test the network as a whole: Sydney - Perth by the Indian
    # Pacific's shortest path against 3,961 km.
    "au": [
        ("Main Southern Railway", "", 617.8, "Cabramatta km 28.43 - Albury km 646.24, WP route "
                                             "table; GA's line starts at Cabramatta"),
        ("Main Northern Railway", "", 567.0, "Strathfield km 12 - Armidale km 579, WP; no "
                                             "passenger trains beyond Armidale"),
        ("North Coast Railway", "", 683.0, "Maitland km 193 - Queensland border km 876, WP"),
        ("Orange Broken Hill Railway", "", 801.0, "Orange - Broken Hill, WP 801 km"),
        ("Illawarra Railway", "", 153.0, "Illawarra Junction - Bomaderry, WP 153 km"),
        ("Blacktown Richmond Railway", "", 25.81, "Blacktown km 34.87 - Richmond km 60.68, WP"),
        ("Perth Kalgoorlie Railway", "", 653.0, "East Perth - Kalgoorlie, the Prospector, WP"),
        ("Perth Mandurah Railway", "", 70.8, "Mandurah line, Perth Underground - Mandurah, WP"),
        ("Belair Line", "", 21.5, "Adelaide - Belair, WP"),
        ("North Coast Line", "Queensland Rail", 1681.0, "Roma Street - Cairns, WP; QR's "
                                                       "deviations since have shortened it"),
    ],
    # India: Wikidata's IR line items (in_register.py), named by their English label, against
    # the en.wikipedia article's infobox length (WP, `tracklength`/`length`, main line only;
    # read 2026-10-03 into data/raw/in/wp_lengths.json) or Wikidata's P2043 (WD). Only lines
    # whose Wikipedia extent is the Wikidata chain's are listed; in_sources.md has the rest
    # and why they differ. The register's own km (the IR timetable's, km_official) checks
    # every line as well. Before OSM the timetable km came to 0.97-1.07 of these.
    "in": [
        ("Mathura–Vadodara Section", "", 852.0, "WP main line (article 'Mathura–Gangapur "
                                                "City–Kota section'); timetable 860"),
        ("Konkan Railway", "", 756.25, "WP, Roha - Thokur; timetable 757"),
        ("Jaipur–Ahmedabad line", "", 630.0, "WP"),
        ("Jodhpur–Bathinda line", "", 600.0, "WP"),
        ("Jabalpur–Bhusaval section", "", 551.0, "WP, the 2004 alignment; timetable 574 "
                                                 "(expect about 1.04)"),
        ("Agra–Bhopal section", "", 508.0, "WP main line"),
        ("Tatanagar–Bilaspur section", "", 468.0, "WP main line"),
        ("Bengaluru–Arsikere–Hubballi line", "", 469.0, "WP"),
        ("Pune–Miraj–Londa line", "", 468.0, "WP; timetable 479"),
        ("Guntakal–Vasco da Gama section", "", 457.0, "WP, WD"),
        ("Kanpur–Delhi section", "", 441.0, "WP (article 'New Delhi–Kanpur section')"),
        ("Viramgam-Okha line", "", 433.0, "WP"),
        ("Khurda Road–Visakhapatnam section", "", 424.0, "WP; timetable 440"),
        ("Bilaspur–Nagpur section", "", 414.0, "WP main line"),
        ("Bhopal–Nagpur section", "", 390.0, "WP"),
        ("Solapur–Guntakal section", "", 379.0, "WP main line"),
        ("Lumding–Dibrugarh section", "", 380.0, "WP (WD 376); the chain is carried on to "
                                                 "Dhamalgaon over the feed: expect about 1.07"),
        ("Duvvada–Vijayawada section", "", 350.0, "WP"),
        ("Pandit Deen Dayal Upadhyaya Nagar–Kanpur section", "", 346.0, "WP main line"),
        ("Lucknow–Moradabad line", "", 326.0, "WP main line"),
        ("Varanasi–Lucknow line", "", 324.0, "WP, via Jaunpur and Ayodhya"),
        ("Jammu–Baramulla line", "", 324.0, "WP; no Wikidata chain, laid over the feed by name"),
        ("Bilaspur-Katni line", "", 318.6, "WP main line"),
        ("Barkakana–Son Nagar line", "", 313.0, "WP; no Wikidata chain, laid over the feed"),
        ("Muzaffarpur-Gorakhpur main line", "", 309.72, "WP"),
        ("Delhi–Jaipur line", "", 305.0, "WP (article 'New Delhi–Jaipur main line'); "
                                         "timetable 317"),
        ("Secunderabad–Dhone section", "", 295.0, "WP (WD 379.69 is wrong)"),
        ("Cuttack–Sambalpur line", "", 284.0, "WP"),
        ("Moradabad–Ambala line", "", 274.0, "WP"),
        ("Ambala–Attari line", "", 273.0, "WP"),
        ("Bina–Katni railway line", "", 262.0, "WP"),
        ("Asansol–Gaya section", "", 267.0, "WP"),
        ("Nallapadu–Nandyal section", "", 256.91, "WP"),
        ("New Jalpaiguri–New Bongaigaon section", "", 252.0, "WP"),
        ("Mysuru–Bengaluru line", "", 138.25, "WP"),
        ("Ernakulam–Kollam line (via Kottayam and Kayamkulam)", "", 156.0, "WP"),
        ("Kalka–Shimla Railway", "", 96.6, "WP (WD 96); the chain ends at Barog in Wikidata "
                                           "and is carried on to Shimla over the feed"),
        ("Darjeeling Himalayan Railway", "", 83.9, "WP (WD 86)"),
        ("Kollam-Thiruvananthapuram trunk line", "", 65.0, "WP"),
        ("Nilgiri Mountain Railway", "", 46.0, "WP, WD"),
        ("Kolkata Circular Railway", "", 36.2, "WP (WD 38.8)"),
        ("Western Line", "", 123.78, "WD, Churchgate - Dahanu Road; no NTES train calls at "
                                     "most of it, so the register's km are crow-fly x 1.1"),
        ("Central Line", "", 180.0, "WP, Mumbai CSMT - Kasara and Khopoli; crow-fly km in "
                                    "the register as for the Western Line"),
    ],
    # Register lines are OSM's track names (gb_register.py). Figures: the en.wikipedia
    # infobox length (miles and chains converted), fetched 2026-10-03. OSM's name does not
    # always cover what Wikipedia's article does; the note says which way it should be off.
    "gb": [
        ("East Coast Main Line", "", 632.73, "WP 393 mi 13 ch, King's Cross - Edinburgh"),
        ("West Coast Main Line", "", 642.13, "WP 399 mi, Euston - Glasgow; OSM's name also covers "
                                             "the Liverpool and Edinburgh branches, and Rugby - "
                                             "Stafford is borrowed over the Trent Valley Line "
                                             "(high, ~1.1)"),
        ("Great Western Main Line", "", 190.28, "WP 118 mi 19 ch, Paddington - Bristol"),
        ("Chiltern Main Line", "", 180.33, "WP 112 mi 4 ch, Marylebone - Birmingham Snow Hill"),
        ("Cornish Main Line", "", 127.94, "WP 79.5 mi, Plymouth - Penzance"),
        ("Reading to Taunton Line", "", 166.79, "WP 103 mi 51 ch"),
        ("North Wales Coast Line", "", 169.78, "WP 105.5 mi, Crewe - Holyhead"),
        ("Welsh Marches Line", "", 135.80, "WP 84.38 mi, Newport - Shrewsbury"),
        ("Cotswold Line", "", 138.89, "WP 86.3 mi, Oxford - Hereford"),
        ("Settle-Carlisle Railway", "", 115.47, "WP 71.75 mi, Settle Jn - Carlisle"),
        ("Heart of Wales Line", "", 144.84, "WP 90 mi, Craven Arms - Llanelli"),
        ("Conwy Valley Line", "", 49.57, "WP 30.8 mi, Llandudno - Blaenau Ffestiniog"),
        ("Far North Line", "", 259.77, "WP 161 mi 33 ch, Inverness - Wick; Thurso branch apart"),
        ("Kyle of Lochalsh Line", "", 102.67, "WP 63 mi 64 ch, Dingwall - Kyle"),
        ("Highland Main Line", "", 190.08, "WP 118 mi 9 ch, Perth - Inverness"),
        ("Aberdeen to Inverness Line", "", 174.23, "WP 108 mi 21 ch"),
        ("Borders Railway", "", 56.83, "WP 35 mi 25 ch, Edinburgh - Tweedbank"),
        ("Newcastle and Carlisle Railway", "", 93.34, "WP 'Tyne Valley line' 58 mi"),
        ("Durham Coast Line", "", 63.57, "WP 39.5 mi, Newcastle - Middlesbrough"),
        ("Cumbrian Coast Line", "", 137.60, "WP 85.5 mi, Carlisle - Barrow"),
        ("Furness Line", "", 45.97, "WP 28 mi 45 ch, Barrow - Carnforth"),
        ("Lakes Line", "", 16.40, "WP 'Windermere branch line' 10 mi 15 ch"),
        ("Esk Valley Line", "", 56.33, "WP 35 mi, Middlesbrough - Whitby"),
        ("York to Scarborough Line", "", 67.71, "WP 42 mi 6 ch"),
        ("East Suffolk Line", "", 78.76, "WP 48 mi 75 ch, Ipswich - Lowestoft"),
        ("Bittern Line", "", 48.72, "WP 30 mi 22 ch, Norwich - Sheringham"),
        ("Breckland Line", "", 82.24, "WP 51 mi 8 ch, Ely - Norwich"),
        ("Fen Line", "", 66.93, "WP 41 mi 47 ch, Cambridge - King's Lynn"),
        ("Felixstowe Branch Line", "", 19.41, "WP 12 mi 5 ch"),
        ("Crouch Valley Line", "", 26.55, "WP 16 mi 40 ch, Wickford - Southminster"),
        ("Shenfield to Southend Line", "", 35.93, "WP 22 mi 26 ch"),
        ("Hertford Loop Line", "", 38.62, "WP 24 mi"),
        ("Marston Vale Line", "", 26.78, "WP 16 mi 51 ch, Bletchley - Bedford"),
        ("Gospel Oak to Barking Line", "", 22.09, "WP 13 mi 58 ch"),
        ("Medway Valley Line", "", 34.18, "WP 21 mi 19 ch"),
        ("Marshlink Line", "", 42.27, "WP 26 mi 21 ch, Ashford - Hastings"),
        ("Hastings Line", "", 52.93, "WP 32 mi 71 ch, Tonbridge - Hastings"),
        ("Sheerness Line", "", 12.31, "WP 7 mi 52 ch"),
        ("North Downs Line", "", 73.22, "WP 45 mi 40 ch, Reading - Redhill"),
        ("West Coastway Line", "", 99.86, "WP 62 mi 4 ch, Brighton - Southampton; OSM's name "
                                          "also has the Littlehampton and Bognor branches"),
        ("Wessex Main Line", "", 137.78, "WP 85 mi 49 ch, Bristol - Southampton"),
        ("Heart of Wessex Line", "", 140.41, "WP 87 mi 20 ch, Bristol - Weymouth; OSM's name is "
                                             "Castle Cary - Dorchester only (low)"),
        ("Tamar Valley Line", "", 22.53, "WP 14 mi"),
        ("Looe Valley Line", "", 13.5, "WD (no WP length)"),
        ("The Maritime Line", "", 18.91, "WP 11.75 mi, Truro - Falmouth"),
        ("Avocet Line", "", 18.11, "WP 11.25 mi, Exeter - Exmouth"),
        ("Tarka Line", "", 62.76, "WP 39 mi, Exeter - Barnstaple"),
        ("Dartmoor Line", "", 24.94, "WP 15.5 mi, Crediton (Coleford Jn) - Okehampton"),
        ("Island Line", "", 13.68, "WP 8.5 mi, Ryde Pier Head - Shanklin"),
        ("Ebbw Valley Line", "", 31.72, "WP 19 mi 57 ch"),
        ("Maesteg Line", "", 13.46, "WP 8 mi 29 ch, Bridgend - Maesteg"),
        ("Marlow Branch Line", "", 11.47, "WP 7 mi 10 ch"),
        ("High Speed 1", "", 109.9, "WP, St Pancras - Channel Tunnel portal; the register line "
                                    "goes on through the tunnel to the border (high, ~1.2)"),
    ],
    # Thailand: SRT's lines (th_register.py, th_sources.md). EN = en.wikipedia "Rail transport
    # in Thailand" (SRT's line list with chainage); NE = en.wikipedia "Northeastern Line
    # (Thailand)" km posts; WD = Wikidata P2043.
    "th": [
        ("สายเหนือ", "", 751.48, "EN Bangkok - Chiang Mai; built also has the 2025 Lop Buri "
                               "bypass (Ban Klap - Lopburi 2 - Khok Kathiam, 19 km beside the "
                               "old line) and the elevated Krung Thep Aphiwat approach: ~1.02"),
        ("สายสวรรคโลก", "", 29.007, "EN Ban Dara Jn - Sawankhalok"),
        ("สายชุมทางบ้านภาชี–อุบลราชธานี", "", 485.15, "NE: Ubon km 575.10 less Ban Phachi "
                                                     "Jn km 89.95"),
        ("สายชุมทางถนนจิระ–หนองคาย", "", 354.82, "NE: Nong Khai km 621.10 less Thanon Chira "
                                               "Jn km 266.28; built goes on 2.7 km over the "
                                               "Friendship Bridge to the border: ~1.01"),
        ("สายชุมทางแก่งคอย–ชุมทางบัวใหญ่", "", 249.887, "EN Kaeng Khoi Jn - Bua Yai Jn"),
        ("สายตะวันออก", "", 255.0, "EN Bangkok - Aranyaprathet; built runs Yommarat (2.2 km "
                                  "out, the trunk to Hua Lamphong is the Northern Line's) to Ban "
                                  "Khlong Luk Border (+5.7): ~1.01"),
        ("สายชุมทางฉะเชิงเทรา–สัตหีบ", "", 134.0, "EN Chachoengsao Jn - Chuk Samet (the Laem "
                                                "Chabang freight branch left out)"),
        ("สายใต้", "", 1144.16, "EN Thon Buri - Su-ngai Kolok; built also has Bang Sue Jn - "
                               "Taling Chan, the way in from Krung Thep Aphiwat: ~1.01"),
        ("สายสุพรรณบุรี", "", 78.09, "EN Nong Pladuk Jn - Suphan Buri"),
        ("สายน้ำตก", "", 130.989, "EN Nong Pladuk Jn - Nam Tok"),
        ("สายคีรีรัฐนิคม", "", 31.25, "EN Ban Thung Pho Jn - Khiri Rat Nikhom"),
        ("สายกันตัง", "", 92.802, "EN Thung Song Jn - Kantang"),
        ("สายนครศรีธรรมราช", "", 35.081, "EN Khao Chum Thong Jn - Nakhon Si Thammarat"),
        ("สายชุมทางหาดใหญ่–ปาดังเบซาร์", "", 45.0, "EN Hat Yai Jn - Padang Besar (Malaysia's "
                                                 "station, 0.5 km past the border)"),
        ("สายแม่กลอง (วงเวียนใหญ่–มหาชัย)", "", 31.22, "EN Wongwian Yai - Maha Chai"),
        ("สายแม่กลอง (บ้านแหลม–แม่กลอง)", "", 33.75, "EN Ban Laem - Mae Klong"),
    ],
    # Register lines are OSM's named passenger track (mx_register.py). Figures fetched
    # 2026-10-03; mx_sources.md has each source. Metro figures are revenue (in-service) lengths.
    "mx": [
        ("Tren Maya", "", 1554.0, "WP, the seven tramos; OSM's track measures ~4% short "
                                   "(tramo III Calkiní - Izamal 142 against 172)"),
        ("El Insurgente", "", 57.7, "es.WP, Zinacantepec - Observatorio"),
        ("Tren Suburbano", "", 50.7, "es.WP 27 km Buenavista - Cuautitlán + SICT 23.7 km "
                                     "Lechería - AIFA"),
        ("Chihuahua al Pacífico", "", 668.0, "WP, Chihuahua - Los Mochis"),
        ("Ferrocarril del Istmo de Tehuantepec (Línea Z)", "", 308.0,
         "WP, Coatzacoalcos - Salina Cruz (suspended)"),
        ("Ferrocarril del Istmo de Tehuantepec (Línea FA)", "", 329.0,
         "Diario del Istmo, Coatzacoalcos - Pakal Ná (suspended)"),
        ("Línea 1", "", 16.654, "WP, Observatorio - Pantitlán"),
        ("Línea 2", "", 20.713, "WP, Cuatro Caminos - Tasqueña"),
        ("Línea 3", "", 21.278, "WP, Indios Verdes - Universidad"),
        ("Línea 4", "", 9.363, "WP, Martín Carrera - Santa Anita"),
        ("Línea 5", "", 14.435, "WP, Politécnico - Pantitlán"),
        ("Línea 6", "", 11.434, "WP, El Rosario - Martín Carrera"),
        ("Línea 7", "", 17.011, "WP, El Rosario - Barranca del Muerto"),
        ("Línea 8", "", 17.679, "WP, Garibaldi - Constitución de 1917"),
        ("Línea 9", "", 13.033, "WP, Tacubaya - Pantitlán"),
        ("Línea A", "", 14.893, "WP, Pantitlán - La Paz"),
        ("Línea B", "", 20.278, "WP, Ciudad Azteca - Buenavista"),
        ("Línea 12", "", 24.110, "WP, Mixcoac - Tláhuac"),
        ("Metrorrey Línea 1", "", 18.8, "WP, Talleres - Exposición"),
        ("Metrorrey Línea 2", "", 13.7, "WP, Sendero - General Zaragoza"),
        ("Metrorrey Línea 3", "", 7.5, "WP, Hospital Metropolitano - General Zaragoza"),
        ("Mi Tren Línea 1", "", 16.5, "WP, Auditorio - Periférico Sur"),
        ("Mi Tren Línea 2", "", 8.7, "WP, Juárez - Tetlán"),
        ("Mi Tren Línea 3", "", 21.5, "WP, Central de Autobuses - Arcos de Zapopan"),
        ("Mi Tren Línea 4", "", 21.2, "WP, Las Juntas - Tlajomulco Centro"),
    ],
    # Brazil: the passenger track of the train lines (br_register.py), named as OSM's route
    # masters. en.WP = the line's en.wikipedia infobox; pt.WP SV = pt.wikipedia "SuperVia"
    # (lengths from Central do Brasil; a branch's register line starts where it leaves the
    # trunk, so the trunk is taken off); WD = Wikidata P2043. Fetched 2026-10-03.
    "br": [
        ("Linha 7 - Rubi", "", 62.7, "en.WP, Palmeiras-Barra Funda - Jundiaí, 17 stations as "
                                     "built; OSM's route also 56.9: the figure is likely from "
                                     "Brás (~0.91)"),
        ("Linha 8 - Diamante", "", 42.0, "en.WP, Júlio Prestes - Amador Bueno"),
        ("Linha 9 - Esmeralda", "", 39.1, "en.WP, Osasco - Varginha; OSM's route also 35.9 "
                                          "(~0.92)"),
        ("Linha 10 - Turquesa", "", 38.0, "WD; built from Palmeiras-Barra Funda (OSM names "
                                          "Barra Funda - Luz - Brás Line 10's track); en.WP has "
                                          "35 (~1.07)"),
        ("Linha 11 - Coral", "", 50.5, "en.WP 54.1 Palmeiras-Barra Funda - Estudantes, less "
                                       "Barra Funda - Luz 3.6 (Line 10's track)"),
        ("Linha 12 - Safira", "", 39.0, "en.WP, Brás - Calmon Viana"),
        ("Linha 13 - Jade", "", 12.2, "en.WP, Engenheiro Goulart - Aeroporto-Guarulhos; OSM's "
                                      "route and track 8.7-8.9, the ends 7.7 km apart as the "
                                      "crow flies (~0.72)"),
        ("Linha Deodoro", "", 23.0, "pt.WP SV, Central - Deodoro (the four-track trunk, both "
                                    "pairs one line)"),
        ("Linha Japeri", "", 38.75, "pt.WP SV 61.75 less the trunk 23: Deodoro - Japeri"),
        ("Linha Santa Cruz", "", 31.75, "pt.WP SV 54.75 less the trunk 23; built from Vila "
                                        "Militar, where the branch leaves the Japeri line "
                                        "(~0.96)"),
        ("Linha Paracambi", "", 8.26, "pt.WP SV, Japeri - Paracambi"),
        ("Linha Vila Inhomirim", "", 15.35, "pt.WP SV, Saracuruna - Vila Inhomirim"),
        ("Estrada de Ferro Vitória a Minas", "", 698.0, "Vale 664 Cariacica - Belo Horizonte "
                                                        "+ the Itabira connection 34 (OSM)"),
        ("Estrada de Ferro Carajás", "", 892.0, "Vale/WD, São Luís - Parauapebas"),
        ("Linha 1 do Metrô de Teresina", "", 16.8, "WD 13.5 Eng. Alberto Tavares Silva - "
                                                   "Itararé + the May 2026 branch Boa "
                                                   "Esperança - Todos os Santos 3.3 (OSM)"),
    ],
    # South Africa: za_register.py's track pieces. Hardly any South African line has a published
    # length of its own, so these are Metrorail's published route lengths less the stretch
    # into Cape Town another register line owns (measured on the build). The build log's
    # "path check" lines test the network the same way, unreduced: Cape Town - Worcester
    # 175.4 / 174, Cape Town - Malmesbury 79.3 / 79.4, Cape Town - Simon's Town 36.0 / 36,
    # Pretoria - Cape Town 1,588 / 1,600 (the Blue Train), Pretoria - Polokwane 289 / 284.4.
    # za_sources.md has the sources.
    "za": [
        ("Salt River–Simon's Town", "", 32.23, "WP Southern Line 36 km Cape Town - Simon's "
                                                "Town, less Cape Town - Salt River (3.77 built)"),
        ("Maitland–Heathfield", "", 16.33, "WP Cape Flats Line 23.8 km Cape Town - Retreat, "
                                           "less Cape Town - Maitland (5.87) and Heathfield - "
                                           "Retreat (1.6), both built"),
        ("Kraaifontein–Malmesbury", "", 47.4, "WP Malmesbury Line 79.4 km, less Cape Town - "
                                              "Kraaifontein (32.0 built)"),
    ],
    # Pakistan: pk_register.py's lines (pk_lines.py). Seven carry PR's km posts from
    # en.wikipedia's route diagrams (km_official, checked line by line above); these are the
    # outside figures: en.WP's list of lines and train articles' published distances. The
    # build log's "PK path check" lines test whole train routes over several lines the same
    # way (Jaffar Express 1,628 / 1,632, Khushhal Khan Khattak Express 1,504 / 1,512, Fareed
    # Express 1,251 / 1,250, Hazara Express 1,576 / 1,594, Kohat Express 176 / 177, Thal
    # Express 571 / 595). pk_sources.md has the rest.
    "pk": [
        ("Karachi–Peshawar Line", "", 1682.0, "WP ML-1 1,687 km Kiamari - Peshawar Cantt, "
                                              "less Kiamari - Karachi City (5, RDT)"),
        ("Quetta–Chaman", "", 142.0, "WP Chaman Passenger, Quetta - Chaman"),
        ("Rohri–Quetta", "", 384.0, "WP ML-3 526 km Rohri - Chaman, less Quetta - Chaman "
                                    "(142, the Chaman Passenger's)"),
    ],
    # West and Central Africa: wafrica_register.py's hand lists traced over OSM track
    # (<cc>_sources.md). The km are ours; these published lengths are the outside check.
    "ng": [
        ("Lagos – Ibadan", "Railway", 156.8, "NRC / WP Lagos–Ibadan SGR, Ebute Metta - Moniya"),
        ("Abuja – Kaduna", "Railway", 186.5, "NRC; WP 187, Idu - Rigasa"),
        ("Warri – Itakpe", "Railway", 326.0, "WP Warri–Itakpe Railway (with the Warri port "
                                             "extension; built Ujevwu - Itakpe)"),
        ("Port Harcourt – Aba", "Railway", 63.0, "nairametrics, NRC handover 2024"),
        ("Red Line", "Lagos", 27.0, "LAMATA's phase 1 figure; built Agbado - Oyingbo "
                                    "platform to platform, 24.4"),
        # No published length found for Iddo - Ijoko (built 28.7 km to where OSM's Cape
        # gauge track ends) or for either Abuja metro line alone (45 km for both: built
        # 26.5 + 17.8 = 44.3).
    ],
    "ga": [
        ("Transgabonais : Owendo – Franceville", "SETRAG", 670.0, "SETRAG's 2019 timetable, Owendo 0 - Franceville "
                                          "670 (fahrplancenter.com)"),
    ],
    "cg": [
        ("Pointe-Noire – Brazzaville", "CFCO", 512.0, "CFCO's PK, fr.WP line diagram"),
    ],
    "sn": [
        ("TER Dakar – AIBD", "SETER", 55.0, "WP Dakar - Diamniadio 36 + press Diamniadio - "
                                            "AIBD ~19 (Sept 2026)"),
    ],
    # Ghana: Tema - Mpakadan is 96.7 km (WP, Railway Gazette); built as Tema - Adome 74.7 +
    # Adome - Mpakadan 19.8 = 94.5, from Tema Harbour station. Accra - Tema has no
    # authoritative length (about 30 km; built 33.5).
    "bf": [
        ("Ouagadougou – Bobo-Dioulasso", "Sitarail", 345.0, "fr.WP: line 1,145 km, Bobo-"
                                                             "Dioulasso at PK 800"),
    ],
    "cm": [
        ("Douala – Yaoundé", "Camrail", 263.0, "seat61 / Camrail"),
        ("Yaoundé – Ngaoundéré", "Camrail", 622.0, "WP Transcamerounais"),
    ],
    # Angola: the railways' own km in their timetables (fahrplancenter.com's transcriptions,
    # CFL 2019, CFM 2019).
    "ao": [
        ("Luanda (Bungo) – Baía", "Luanda", 36.0, "CFL, Bungo 0 - Baía 36"),
        ("Baía – Malanje", "Luanda", 386.0, "CFL, Baía 36 - Malanje 422"),
        ("Zenza do Itombe – Dondo", "Luanda", 46.0, "CFL, Zenza 134 - Dondo 180 (greyed)"),
        ("Namibe – Lubango", "Moçâmedes", 246.0, "CFM, Namibe 0 - Lubango 246"),
        ("Lubango – Matala – Menongue", "Moçâmedes", 510.0, "CFM, Lubango 246 - Menongue 756"),
        ("Lobito – Huambo – Luau", "Benguela", 1344.0, "CFB, Lobito - Luau ~1,344 (the line "
                                                       "rebuilt 2006-2014 on shorter "
                                                       "alignments; no new figure found)"),
    ],
    "cd": [
        ("Train urbain : Gare Centrale – Ndjili", "SCTP", 25.0, "press: 'nearly 25 km', Tshenke - Gare Centrale; "
                                       "built Gare Centrale - Ndjili Aéroport"),
    ],
    # Bangladesh: bd_register.py's lines (bd_lines.py), lengths our own traces. Few lines have
    # a published length of their own; where the built line is one stretch BR's East Zone fare
    # list or the Information Book 2024 measures, that is the check. More, as paths over the
    # network, in the build log's "path check" lines (bd_lines.PATH_CHECKS). bd_sources.md.
    "bd": [
        ("Akhaura–Laksam–Chittagong line", "", 204.0, "BR fare list, Akhaura - Chattogram"),
        ("Akhaura–Kulaura–Chhatak line", "", 177.0, "BR fare list, Akhaura - Sylhet (Sylhet "
                                                     "- Chhatak not built: no track in OSM)"),
        ("Khulna–Mongla Port line", "", 63.82, "Information Book 2024, Khulna - Mongla "
                                               "(Phultala - Mongla)"),
        ("Pabna–Dhalarchar line", "", 78.8, "Information Book 2024, Pabna - Dhalarchar"),
        ("Chittagong–Cox's Bazar line", "", 149.0, "Information Book 2024, Dohazari - Cox's "
                                                   "Bazar 102 + Chittagong - Dohazari 47 "
                                                   "(en.wikipedia)"),
    ],
    # Sri Lanka: lk_register.py's lines (en.wikipedia's by-line station tables, km from
    # Colombo Fort), cut where Cyclone Ditwah's damage still stops trains. Wikidata's line
    # lengths (P2043) where the built line is the whole line, else the table's own km between
    # the two ends (that is also km_official, so those rows check the trace, not the source).
    # lk_sources.md has the rest.
    "lk": [
        ("Northern Line", "", 339.0, "Wikidata, Polgahawela - Kankesanturai (table 336.4)"),
        ("Batticaloa Line", "", 212.0, "Wikidata, Maho - Batticaloa (table 210.6)"),
        ("Coastal Line", "", 184.65, "Wikidata, Colombo Fort - Matara 157.9 + Matara - "
                                     "Beliatta 26.75 (table 185.2)"),
        ("Mannar Line", "", 106.0, "Wikidata, Medawachchiya - Talaimannar Pier; OSM's track "
                                   "ends at Talaimannar, 2.4 km short of the pier"),
        ("Trincomalee Line", "", 70.0, "Wikidata, Gal Oya - Trincomalee (table 69.8)"),
        ("Puttalam Line", "", 116.82, "table, Ragama 16.42 - Puttalam 133.24 (Wikidata's 133 "
                                      "is from Colombo Fort)"),
        ("Kelani Valley Line", "", 56.9, "table, Maradana 2.08 - Avissawella 58.98"),
        ("Main Line", "", 127.38, "table, Colombo Fort - Gampola (Wikidata 292 to Badulla)"),
        ("Main Line (Gampola - Nanu Oya)", "", 79.52, "table, Gampola 127.38 - Nanu Oya 206.9"),
        ("Main Line (Nanu Oya - Badulla)", "", 84.7, "table, Nanu Oya 206.9 - Badulla 291.6"),
        ("Matale Line", "", 27.64, "table, Kandy 119.5 - Matale 147.14"),
        # Matale Line (Peradeniya - Kandy), greyed, 5.9 km traced: the table's 4.16 (Peradeniya
        # 115.34 - Kandy 119.5) is shorter than the crow flies, so no number checks it.
    ],
    # Indonesia: id.wikipedia's line articles (id_register.py), named as the article less
    # "Jalur kereta api", against the article's infobox length (`linelength`/`tracklength`,
    # read 2026-10-03 into data/raw/id/idwiki_articles.json). Only lines whose infobox extent
    # is what runs; the note says where branches or closed ends make it differ. KAI's own km
    # posts (km_official) check every line as well. id_sources.md has the rest.
    "id": [
        ("Cikampek–Cirebon–Kroya", "", 293.0, "id.WP"),
        ("Cirebon–Semarang", "", 225.6, "id.WP, Cirebon - Semarang Tawang"),
        ("Gundih–Surabaya Pasarturi", "", 230.0, "id.WP"),
        ("Kertosono–Bangil", "", 215.5, "id.WP, via Kediri, Blitar, Malang"),
        ("Surabaya–Bangil–Kalisat", "", 214.4, "id.WP"),
        ("Bogor–Padalarang–Kasugihan", "", 388.0, "id.WP; Cipatat - Padalarang (16 km) closed "
                                                  "and left out: ~0.96, in two pieces"),
        ("Cilacap–Yogyakarta", "", 175.0, "id.WP main line; + YIA (5.5) and Karangtalun (3.4) "
                                          "branches: ~1.04"),
        ("Brumbung–Gambringan", "", 46.0, "id.WP"),
        ("Tegal–Prupuk", "", 38.5, "id.WP"),
        ("Anyer Kidul–Kampung Bandan", "", 147.0, "id.WP Merak - Kampung Bandan; KAI's posts sum "
                                                  "to 154.5 (Tanah Abang - Kampung Bandan by "
                                                  "Duri and Angke in): ~1.07"),
        ("Kereta Cepat Jakarta–Bandung", "", 142.3, "id.WP (en.WP 142.8), Halim - Tegalluar"),
        ("Medan–Tebing Tinggi", "", 80.5, "id.WP; + Kualanamu branch (~4.7): ~1.06"),
        ("Tebing Tinggi–Kisaran", "", 73.0, "id.WP; Kuala Tanjung freight branch left out"),
        ("Tebing Tinggi–Siantar", "", 48.0, "id.WP"),
        ("Kisaran–Rantau Prapat", "", 114.0, "id.WP"),
        ("Kisaran–Tanjungbalai", "", 20.7, "id.WP"),
        ("Belawan–Medan", "", 22.0, "id.WP"),
        ("Lubuk Alung–Naras–Sungai Limau", "", 27.8, "id.WP, Lubuk Alung - Naras"),
        ("Prabumulih–Kertapati", "", 78.0, "id.WP"),
        ("Prabumulih–Panjang", "", 324.0, "id.WP to Panjang; passengers end at Tanjungkarang, "
                                          "the goods line beyond left out: ~0.96"),
        ("Lubuk Linggau–Prabumulih", "", 227.2, "KAI km posts, Lubuk Linggau km 549.448 - "
                                                "Prabumulih km 322.295 (no infobox length)"),
        ("Makassar–Parepare", "", 109.0, "DJKA, Sulawesi's active km (2025); built Mandai - "
                                         "Garongkong without the Tonasa goods branch: ~0.93"),
    ],
    # Serbia: IŽS Network Statement 2026, Appendix 6 chainage (data/raw/rs_ns2026_appendix6.txt),
    # over the stretch built. Border stubs with no train are dropped, as in Croatia.
    "rs": [
        ("101 Београд Центар – Шид", "", 116.365, "IŽS, BC 0.000 - Šid 116.365; Šid - border "
                                                  "(5.6, no train) dropped: ~0.97"),
        ("102 Београд Центар – Ниш – Прешево", "", 392.309, "IŽS, BC - Preševo; BC - Rakovica "
                                                              "traces 5.8 for IŽS's 8.5: ~0.98"),
        ("103 Раковица – Мала Крсна – Велика Плана", "", 93.4, "IŽS by its passenger distances "
                                                               "(Rakovica - K1 3.0, K1 - Jajinci 1.6)"),
        ("104 Ћуприја – Параћин", "", 7.420, "IŽS"),
        ("105 Стара Пазова – Нови Сад – Суботица", "", 141.606, "IŽS, Stara Pazova 34.944 - "
                                                                "Subotica 176.550"),
        ("106 Ниш – Димитровград", "", 97.182, "IŽS, Niš 0.241 - Dimitrovgrad 97.423"),
        ("107 Београд Центар – Панчево – Вршац", "", 87.777, "IŽS distances via Pančevo Glavna"),
        ("108 Ресник – Пожега – Врбница", "", 287.013, "IŽS, Resnik 0.425 - border 287.438 (ŽICG "
                                                       "287+438.70); Štrpci's 9 km in BiH included"),
        ("109 Лапово – Краљево – Рудница", "", 161.322, "IŽS, Lapovo 0.666 - Rudnica 161.988"),
        ("110 Суботица – Сомбор – Богојево", "", 88.057, "IŽS, Bogojevo 43.815 - Subotica 131.872"),
        ("120 Карађорђев парк – Дедиње", "", 1.491, "IŽS; OSM's Dedinje junction is 0.6 km "
                                                    "nearer: ~0.6"),
        ("121 Инђија – Голубинци", "", 5.476, "IŽS passenger distance via Inđija TT"),
        ("201 Суботица – Хоргош", "", 24.018, "IŽS Subotica - Horgoš; + Horgoš - border 1.0 "
                                              "traced: ~1.04"),
        ("202 Панчево Главна – Зрењанин – Кикинда", "", 154.316, "IŽS to Banatsko Veliko Selo"),
        ("205 Банатско Милошево – Сента – Суботица", "", 78.045, "IŽS distances across Senta"),
        ("207 Нови Сад – Оџаци – Богојево", "", 73.614, "IŽS distances, Sajlovo - Bogojevo"),
        ("208 Нови Сад – Римски Шанчеви – Орловат", "", 53.030, "IŽS, Sajlovo - Perlez (OSM has "
                                                                "no station beyond)"),
        ("211 Рума – Шабац – Брасина", "", 99.998, "IŽS, Ruma 0.517 - Donja Borina 100.515"),
        ("213 Сталаћ – Краљево – Пожега", "", 135.733, "IŽS"),
        ("216 Смедерево – Мала Крсна", "", 10.929, "IŽS"),
        ("218 Мала Крсна – Пожаревац – Бор – Вражогрнац", "", 178.773, "IŽS 71.272 - 250.045"),
        ("219 Ниш – Зајечар – Прахово Пристаниште", "", 183.621, "IŽS, Crveni Krst - Prahovo "
                                                                 "Pristanište"),
        ("223 Дољевац – Прокупље – Куршумлија – Мердаре", "", 86.169, "IŽS via the Kuršumlija "
                                                                     "stub, which is no stop"),
        ("308 Доња Борина – Зворник Град", "", 5.654, "IŽS, Donja Borina - Zvornik"),
        ("309 Панчево Варош – Панчево Војловица", "", 2.346, "IŽS to Vojlovica's axis; OSM's "
                                                             "station node is past it: ~1.19"),
        ("501 Шарган Витаси – Мокра Гора", "", 15.440, "IŽS; the Šargan Eight's loops trace "
                                                       "2.8 short: ~0.81"),
    ],
    # Montenegro: ŽICG Network Statement 2017, Annex 4 (data/raw/me_zicg_izjava_o_mrezi_2017.pdf).
    "me": [
        ("Bar – Vrbnica", "", 167.408, "ŽICG, border 287+438.70 - Bar 454+847"),
        ("Podgorica – Nikšić", "", 56.215, "ŽICG, Nikšić 0+293 - Podgorica 56+508"),
        ("Podgorica – Tuzi – državna granica", "", 13.683, "ŽICG Podgorica - Tuzi; Tuzi - border "
                                                           "(11.1, freight) dropped"),
    ],
    # North Macedonia: Wikidata's lengths (P2043).
    "mk": [
        ("Табановце – Гевгелија", "", 214.9, "Wikidata Q3239944, with both border stubs "
                                             "(dropped): ~0.97"),
        ("Скопје – Волково – Блаце", "", 31.1, "Wikidata Q3239932"),
        ("Ѓорче Петров – Кичево", "", 102.6, "Wikidata Q3239602"),
        ("Велес – Битола – Кременица", "", 145.3, "Wikidata Q3239995 to Kremenica; built ends at "
                                                  "Žabeni (no OSM track beyond): ~0.96"),
        ("Велес – Кочани", "", 85.5, "Wikidata Q3239994"),
    ],
    # Bosnia and Herzegovina: ŽFBH's chainage and Wikidata.
    "ba": [
        ("11 Sarajevo – Čapljina", "", 177.7, "ŽFBH Sarajevo 0+000 - Čapljina 170+390 (its "
                                              "infrastructure page) + Čapljina - border 7.3 "
                                              "traced"),
        ("12 Šamac – Doboj – Sarajevo", "", 242.0, "Wikidata Q1279793; Šamac - Doboj (no train, "
                                                   "no OSM station or route) dropped: ~0.71"),
    ],
    # Albania: Wikidata's lengths.
    "al": [
        ("Elbasan – Pogradec", "", 78.0, "Wikidata Q31667925"),
        ("Fier – Ballsh", "", 25.0, "Wikidata Q130927911"),
    ],
    # Türkiye: register lines are OSM's track names (tr_register.py). Figures: TCDD's 2025
    # network statement, Ek-3.3 (data/raw/tr/sb2025_ek33_sections.csv: route-km per section,
    # whole km), summed over the line's extent; WD = Wikidata P2043; KHY = the 2026 decree's
    # public-service distances. A sum of whole-km rows is good to about 1%.
    "tr": [
        ("Ankara-Kars demiryolu", "", 1360.0, "KHY Ankara - Kars 1,360 (= Ek-3.3 sum); trains "
                                              "use the Tecer - Kangal variant, its own line"),
        ("İstanbul - Ankara demiryolu", "", 545.0, "Ek-3.3 Söğütlüçeşme - Gebze - Köseköy - "
                                                   "Arifiye - Eskişehir - Ankara; Köseköy - "
                                                   "Sapanca bridged in since 2026-10-04 "
                                                   "(its own track, cut at the YHT's joint)"),
        ("Ankara - İstanbul yüksek hızlı demiryolu", "", 414.0,
         "Ek-3.3 YHT rows 8-15; OSM has the YHT on the old line Karaköy - Yayla and "
         "Doğançay - Arifiye - Sapanca, bridged in since 2026-10-04 (borrowed where on the "
         "old line's track), so a little high (1.05)"),
        ("Ankara - Sivas yüksek hızlı demiryolu", "", 405.0, "tr.wikipedia, Ankara - Sivas"),
        ("Polatlı - Konya yüksek hızlı demiryolu", "", 224.0, "Ek-3.3 row 14"),
        ("Eskişehir-Konya demiryolu", "", 427.0, "Ek-3.3 rows 214-218"),
        ("İzmir-Afyonkarahisar demiryolu", "", 420.0, "Ek-3.3 Basmane - Afyon; WD 421.7"),
        ("İzmir-Alsancak-Eğirdir demiryolu", "", 430.0, "Ek-3.3 Alsancak - Goncalı - Karakuyu "
                                                        "- Gümüşgün (Eğirdir closed)"),
        ("Irmak-Zonguldak demiryolu", "", 414.0, "Ek-3.3 rows 72-82; tr.wikipedia 415.2"),
        ("Fevzipaşa-Kurtalan demiryolu", "", 413.0, "Ek-3.3 Malatya - Kurtalan only: no "
                                                    "passenger train Fevzipaşa - Malatya"),
        ("Samsun-Kalın demiryolu", "", 377.0, "Ek-3.3 rows 155-156; WD 377.8"),
        ("Yolçatı-Tatvan demiryolu", "", 373.0, "Ek-3.3 rows 177-185"),
        ("Manisa-Bandırma demiryolu", "", 276.0, "Ek-3.3 rows 103-108; WD 275.1"),
        ("Alayunt-Balıkesir demiryolu", "", 262.0, "Ek-3.3 rows 219-224"),
        ("İstanbul – Pythion demiryolu", "", 244.0, "KHY Halkalı - Uzunköprü 229 + Bakırköy - "
                                                    "Halkalı ~15; Uzunköprü - border no train"),
        ("Boğazköprü-Ulukışla demiryolu", "", 172.0, "Ek-3.3 rows 67-71"),
        ("Malatya-Çetinkaya demiryolu", "", 141.0, "Ek-3.3 rows 157-161"),
        ("Afyonkarahisar-Karakuyu demiryolu", "", 114.0, "Ek-3.3 rows 228-229; WD 114.2"),
        ("Ulukışla-Yenice demiryolu", "", 108.0, "Ek-3.3 rows 198-199"),
        ("Pehlivanköy – Svilengrad demiryolu", "", 68.0, "Ek-3.3 row 18 + to the border"),
        ("Toprakkale-İskenderun demiryolu", "", 58.9, "WD; Ek-3.3 row 204 58"),
        ("Torbalı-Ödemiş demiryolu", "", 62.9, "WD; OSM's line also has Ödemiş Gar - Şehir "
                                               "and both legs at Torbalı (high)"),
        ("Kars-Gümrü-Tiflis demiryolu", "", 55.0, "KHY Kars - Akyaka 55"),
        ("Gümüşgün-Isparta şube demiryolu", "", 28.0, "Ek-3.3 rows 233, 235; WD 13.4 from "
                                                      "Bozanönü"),
        ("Gümüşgün-Burdur şube demiryolu", "", 23.9, "WD; Ek-3.3 row 236 23"),
        ("Ortaklar-Söke şube demiryolu", "", 22.0, "WD 22.0; Ek-3.3 row 95 22"),
        ("Goncalı-Denizli şube demiryolu", "", 9.4, "WD; Ek-3.3 row 94 10"),
        ("Arifiye - Adapazarı şube demiryolu", "", 8.5, "tr.wikipedia; Ek-3.3 row 32 8"),
        ("Menemen-Aliağa demiryolu", "", 24.9, "WD; OSM's line has both legs of the Menemen "
                                               "triangle (high)"),
        ("Marmaray", "", 76.6, "WD, Halkalı - Gebze"),
        ("Başkentray", "", 36.0, "Ek-3.3 Sincan - Ankara 24 + Ankara - Kayaş 12"),
        ("Gaziray", "", 24.0, "Ek-3.3 rows 6, 208 Başpınar - Taşlıca; WD 25.5"),
    ],
    # Iran: RAI's railways as OSM's route=railway relations (ir_register.py, ir_sources.md).
    # RAI = RAI's station table with each station's km (fa.wikipedia "فهرست ایستگاه‌های
    # راه‌آهن ایران", data/raw/ir/wp/); EN = en.wikipedia "Rail transport in Iran", its table of
    # lines with lengths; WD = Wikidata P2043.
    "ir": [
        ("راه آهن تهران – مشهد", "", 923.0, "RAI Mashhad km 923.0 from Tehran by Garmsar; WD "
                                            "Garmsar - Mashhad 812"),
        ("خط راه‌آهن تهران – تبریز", "", 735.9, "WD 735.9; RAI Tabriz km 735.855 (by Maragheh)"),
        ("راه آهن تبریز – جلفا", "", 146.1, "RAI Jolfa 882.0 - Tabriz 735.9; EN 148"),
        ("راه آهن بافق – بندرعباس", "", 612.1, "RAI Bandar Abbas 1482.2 - Bafq 870.1"),
        ("راه آهن کرمان – زاهدان", "", 539.2, "RAI Zahedan 1658.7 - Kerman 1104.6, less Kerman "
                                             "- junction (14.9 km, Qom - Kerman's)"),
        ("راه آهن قم – کرمان", "", 825.0, "RAI by Nain: Kashan - Kerman 742 (Ardakan 582.4, "
                                         "Meybod - Yazd 60, Yazd - Kerman 351.8) + Mohammadieh "
                                         "- Kashan 80 + Kerman - junction 14.9 (built, "
                                         "unchecked); EN's Qom - Zarand 847 is longer"),
        ("خط ریلی بادرود – شیراز", "", 732.2, "RAI Shiraz 1074.5 - Badrud 342.3 (by Isfahan)"),
        ("راه آهن اصفهان – اردکان", "", 184.3, "RAI Meybod 692.75 - Sistan 508.5 (by Varzaneh, "
                                             "Aqda)"),
        ("راه آهن مشهد – بافق", "", 777.0, "RAI Bafq - Torbat-e Heydarieh 672.0 + Torbat - the "
                                          "Mashhad line at Kashmar ~105 (EN's table: Bafq - "
                                          "Torbat 800)"),
        ("راه آهن تربت حیدریه – خواف", "", 121.5, "RAI (from Bafq) Khaf 793.5 - Torbat 672.0"),
        ("راه‌آهن گرمسار – اینچه‌برون", "", 457.9, "RAI Garmsar - Bandar Torkaman 346.3, BT - "
                                                 "Gorgan 35.1, Yampi - Incheh Borun 58.0, + "
                                                 "the fork - Yampi 18.5 (built)"),
        ("راه‌آهن سراسری ایران (تهران – بندر امام خمینی)", "", 938.6,
         "RAI Tehran - Ahvaz 815.9 (by Parandak) + Ahvaz - Mahshahr 111.4, + the Robat Karim - "
         "Parand branch 11.3 (built); Mahshahr - Bandar Imam has no passenger train"),
        ("راه آهن اهواز – خرمشهر", "", 120.9, "RAI Khorramshahr 936.8 - Ahvaz 815.9; EN 121"),
        ("مسیر جدید میانه – تبریز", "", 173.6, "RAI Khavaran 612.8 - Mianeh 439.2"),
        ("راه آهن میانه – اردبیل", "", 174.0, "EN; built from the fork on the new Tabriz line "
                                             "north of Mianeh"),
        ("راه‌آهن مراغه – ارومیه", "", 183.0, "EN; WD 184"),
        ("راه آهن قزوین – رشت", "", 164.0, "EN"),
        ("خط ریلی یزد – اقلید", "", 271.0, "EN"),
        ("راه آهن همدان – سنندج", "", 151.0, "EN; RAI 148.0"),
        ("راه آهن اراک – کرمانشاه", "", 267.0, "EN; RAI Kermanshah 603.0 - Arak 320.3 = 282.7, "
                                              "of which Arak - the Shazand junction is the "
                                              "Trans-Iranian's"),
        ("راه آهن فریمان – سرخس", "", 175.0, "EN Mashhad - Sarakhs 165; RAI Fariman - Sarakhs "
                                            "162.0 + the Salam - Shahid Motahari link ~13"),
        ("راه آهن چابهار – زاهدان", "", 155.0, "EN Zahedan - Khash, the open part"),
        ("راه آهن صوفیان – رازی", "", 190.4, "RAI Razi 957.6 - Sufian 767.2 (suspended)"),
    ],
    # Morocco, Algeria, Tunisia, Egypt: nafrica_register.py's hand-written lists, traced over
    # OSM track (nafrica_sources.md). The km are ours, so these published lengths are the only
    # outside check.
    # Morocco: en.wikipedia / fr.wikipedia figures (ONCF publishes no line lengths).
    "ma": [
        ("LGV Tanger – Kénitra", "", 186.0, "en.WP; built platform to platform, with both "
                                            "approaches on their own track: ~1.04"),
        ("Casablanca – Rabat – Kénitra", "", 137.0, "en.WP Kenitra - Casablanca; built Casa "
                                                    "Voyageurs - Kénitra + Casa Port's 6 km"),
        ("Sidi Yahya – Mechraa Bel Ksiri", "", 45.0, "fr.WP; built from the junction 3.3 km out"),
        ("Tanger – Tanger Med", "", 45.0, "fr.WP Tanger - port; built from the junction 3 km out"),
        ("Taourirt – Nador – Beni Ansar", "", 110.0, "fr.WP; built from the junction 6.5 km west of "
                                        "Taourirt to Beni Nsar Ville"),
    ],
    # Algeria: SNTF's km in its timetables (fahrplancenter.com's transcriptions, 2017-2019,
    # data/raw/dz/fahrplancenter/), else the opening news or fr.WP's list.
    "dz": [
        ("Alger – Blida – Chlef – Oran", "", 419.0, "SNTF 302, Alger 0 - Oran 419"),
        ("El Harrach – Thénia – Bouira – Sétif – Constantine", "", 454.0,
         "SNTF 102, El Harrach 10 - Constantine 464"),
        ("Thénia – Tizi Ouzou – Oued Aïssi", "", 62.0, "fr.WP; SNTF 101 says 67"),
        ("Beni Mansour – Béjaïa", "", 88.0, "SNTF 103"),
        ("Constantine – Ramdane Djamel – Skikda", "", 86.0, "SNTF 205"),
        ("Ramdane Djamel – Azzaba – Annaba", "", 99.0, "SNTF 102, 532 - 631"),
        ("El Guerrah – Batna – Biskra – Touggourt", "", 419.0, "SNTF 203, El Gourzi 38 - "
                                                               "Touggourt 457; fr.WP 417"),
        ("Aïn Touta – Barika – M'Sila", "", 148.0, "SNTF 202, 151 - 299; fr.WP 145"),
        ("Bordj Bou Arreridj – M'Sila", "", 55.0, "fr.WP"),
        ("M'Sila – Boughezoul – Tissemsilt", "", 290.0, "opening news, 2022"),
        ("Boughezoul – Djelfa – Laghouat", "", 250.0, "opening news, 2023"),
        ("Annaba – Souk Ahras – Tébessa", "", 231.0, "SNTF 209"),
        ("Souk Ahras – frontière tunisienne", "", 53.0, "fr.WP; built to the outline"),
        ("Annaba – Sidi Amar", "", 13.9, "fr.WP"),
        ("Oued Tlelat – Sidi Bel Abbès – Béchar", "", 649.0, "SNTF 402, Béchar 0 - Oued Tlelat "
                                                             "649; fr.WP 648"),
        ("Tabia – Tlemcen – Maghnia – Ghazaouet", "", 185.0, "SNTF 403, Tabia 99 - Maghnia 219, "
                                                             "Akid Abbas 229 - Ghazaouet 284"),
        ("Moulay Slissen – Saïda – Frenda", "", 221.0, "SNTF 407 Moulay Slissen - Saïda 101 + "
                                                       "Saïda - Frenda 120 (opening, 2023)"),
        ("Oran – Aïn Témouchent", "", 70.0, "SNTF 401, Es Sénia 6 - Aïn Témouchent 76"),
        ("Oran – Arzew", "", 41.7, "fr.WP"),
        ("Mohammadia – Mostaganem", "", 47.0, "SNTF 404"),
    ],
    # Tunisia: SNCFT's own GTFS (shape_dist_traveled, median over trips) between the ends.
    "tn": [
        ("Tunis – Djedeida – Béja – Ghardimaou", "", 211.1, "SNCFT GTFS"),
        ("Djedeida – Mateur – Bizerte", "", 72.5, "SNCFT GTFS"),
        ("Tunis – Pont du Fahs – Dahmani – Kalaâ Khasba", "", 229.7, "SNCFT GTFS, Djebel Jelloud - Kalaâ Khasba"),
        ("Les Salines – Le Kef", "", 31.2, "SNCFT GTFS"),
        ("Ghraïba – Gafsa – Metlaoui – Tozeur", "", 234.1, "SNCFT GTFS"),
        ("Bir Bou Rekba – Hammamet – Nabeul", "", 17.1, "SNCFT GTFS"),
        ("Metlaoui – Redeyef", "", 44.1, "SNCFT GTFS"),
        ("Tabeddit – Om El Araies", "", 8.8, "SNCFT GTFS"),
    ],
    # Egypt: ENR's km per train as egypttrains.com lists them (data/raw/eg/enr_train_list.json).
    "eg": [
        ("القاهرة – طنطا – الإسكندرية", "", 208.0, "ENR Cairo - Alexandria"),
        ("القاهرة – أسيوط – الأقصر – أسوان", "", 879.0, "ENR Cairo - Aswan"),
        ("أسوان – السد العالي", "", 20.0, "ENR Cairo - High Dam 899 less 879; 19 listed"),
        ("الواسطى – الفيوم", "", 38.0, "ENR"),
        ("الزقازيق – أبو حماد – الإسماعيلية", "", 78.0, "ENR"),
        ("الإسماعيلية – القنطرة – بورسعيد", "", 79.0, "ENR"),
        ("الإسماعيلية – فايد – السويس", "", 92.0, "ENR"),
        ("القنطرة شرق – بئر العبد", "", 68.0, "ENR"),
        ("بنها – منيا القمح – الزقازيق", "", 35.0, "ENR"),
        ("طنطا – زفتى – ميت غمر – الزقازيق", "", 56.0, "ENR"),
        ("بنها – ميت غمر", "", 35.0, "ENR"),
        ("الزقازيق – السنبلاوين – المنصورة", "", 71.0, "ENR"),
        ("طنطا – المحلة – المنصورة – دمياط", "", 119.0, "ENR"),
        ("المنصورة – دكرنس – المنزلة – المطرية", "", 73.0, "ENR"),
        ("طنطا – قلين – كفر الشيخ", "", 63.0, "ENR"),
        ("دمنهور – دسوق – قلين", "", 43.0, "ENR"),
        ("دسوق – فوه – مطوبس – البوصيلي", "", 40.0, "ENR"),
        ("المعمورة – إدكو – البوصيلي – رشيد", "", 49.0, "ENR (40 for the trains that end short)"),
        ("السنطة – محلة روح – المحلة الكبرى", "", 32.0, "ENR"),
        ("بنها – منوف", "", 27.0, "ENR"),
        ("منوف – شبين الكوم – تلا – طنطا", "", 42.0, "ENR"),
        ("منوف – كفر الزيات", "", 50.0, "ENR"),
        ("بشتيل – وردان – إيتاي البارود", "", 115.0, "ENR"),
        ("الزقازيق – أبو كبير – فاقوس – الصالحية", "", 57.0, "ENR"),
        ("محرم بك – الحمام – العلمين – مرسى مطروح", "", 298.0, "ENR"),
        ("23 يوليو – الخانكة – شبين القناطر", "", 19.0, "ENR"),
    ],
    # New Zealand: KiwiRail's lines (nz_register.py), the stretches with passenger trains, by
    # KiwiRail's name, against en.wikipedia's infobox lengths (WP, read 2026-10-03). KiwiRail's
    # own km posts check every line as well (km_official).
    "nz": [
        ("North Island Main Trunk", "", 684.98, "WP, Wellington - Maungawhau through the City "
                                                "Rail Link; two short kms in the posts (km 274, "
                                                "357) make KiwiRail's own chainage 1.5 km long"),
        ("Main North Line", "", 348.04, "WP, Addington - Picton"),
        ("Midland Line", "", 212.0, "WP, Rolleston - Greymouth"),
        ("Johnsonville Line", "", 10.49, "WP Johnsonville Branch, Wellington - Johnsonville"),
        ("Wairarapa Line", "", 89.16, "WP: Masterton km 90.96; the line as built starts at "
                                      "KiwiRail's first post, km 1.8, where it leaves the NIMT"),
        ("Taieri Gorge Railway", "", 42.0, "WP: Pukerangi 19 km short of Middlemarch, the line "
                                           "60 km from the Taieri Branch's 4 km peg; built from "
                                           "Taieri (km 3)"),
    ],
    # Vietnam: VNR's lines as OSM names the track (vn_register.py, vn_sources.md). VI = the
    # vi.wikipedia line article's infobox; its station chainage (from VNR) where a line starts
    # at a junction, since OSM gives the shared trunk out of Hà Nội to one line only: Gia Lâm
    # km 5, Yên Viên km 11, Đông Anh km 21. The North-South line also has km_official from the
    # 175-station chainage table, checked section by section.
    "vn": [
        ("Đường sắt Bắc Nam", "", 1726.0, "VI, Hà Nội - Sài Gòn"),
        ("Đường sắt Hà Nội - Lào Cai", "", 283.0, "VI chainage Yên Viên km 11 - Lào Cai km 294 "
                                                 "(the 296 of the infobox is from Hà Nội)"),
        ("Đường sắt Hà Nội - Đồng Đăng", "", 166.5, "VI 162 Hà Nội - Đồng Đăng + the 4.5 km to "
                                                   "the border traced"),
        ("Đường sắt Hà Nội - Hải Phòng", "", 97.0, "VI 102 less Hà Nội - Gia Lâm (km 5): "
                                                  "DRVN's Gia Lâm - Hải Phòng"),
        ("Đường sắt Hà Nội - Quan Triều", "", 54.0, "VI 75 less Hà Nội - Đông Anh (km 21): "
                                                   "DRVN's Đông Anh - Quán Triều"),
        ("Đường sắt Kép - Cái Lân", "", 109.4, "en.WP Kép - Hạ Long 106 + Hạ Long - Cái Lân 3.4 "
                                              "traced (VI's infobox 126 counts more); greyed"),
        ("Đường sắt Diêu Trì - Quy Nhơn", "", 10.5, "VI; built from Diêu Trì station along the "
                                                   "branch's own track: ~0.94"),
        ("Đường sắt Bình Thuận - Phan Thiết", "", 10.0, "VI \"Ga Phan Thiết\": about 10 km from "
                                                       "Mương Mán (now Bình Thuận)"),
        ("Đường sắt Đà Lạt - Trại Mát", "", 7.0, "VI/en.WP 7 km; built from the station "
                                                "building's centre: ~0.93"),
        ("Tàu hỏa leo núi Mường Hoa", "", 2.0, "Sun World: \"2 km\" Sa Pa - Mường Hoa, a round "
                                              "figure; OSM's track is 1.69 km whole: ~0.84"),
    ],
    # Argentina: the track of the passenger routes, grouped by hand (ar_register.py,
    # ar_sources.md). "post" = the transport ministry's station km posts (Estaciones de Trenes
    # y Servicios activos a 2022, datos.transporte.gob.ar); `python ar_register.py --chainage`
    # checks every section that has one at both ends (239 sections, median 0.997). The Roca,
    # Mitre and Belgrano Sur have branches whose posts start elsewhere, so no line total here.
    "ar": [
        ("Línea Sarmiento", "", 169.24, "posts: Once - Moreno 36.382 + Moreno - Mercedes 61.649 "
                                        "+ Merlo - Lobos 71.206"),
        ("Línea San Martín", "", 72.308, "post, Retiro - Dr. Cabred"),
        ("Línea Belgrano Norte", "", 51.944, "post, Retiro - Villa Rosa"),
        ("Línea Urquiza", "", 25.612, "post, Federico Lacroze - General Lemos"),
        ("Tren de la Costa", "", 15.203, "post, Avenida Maipú - Delta"),
        ("Ferrocarril Roca: Chascomús – Mar del Plata", "", 283.5,
         "the 400 km given for Constitución - Mar del Plata less Chascomús' post 116.5"),
        ("Ferrocarril San Martín: Cabred – Junín", "", 181.7, "es.WP Junín 254 km from Retiro "
                                                              "less Cabred's post 72.3"),
        ("Ferrocarril Sarmiento: Mercedes – Bragado", "", 110.97, "es.WP Bragado 209 km from "
                                                                  "Once less Mercedes' post 98.03"),
        ("Ferrocarril Roca: Viedma – Bariloche", "", 827.0, "es.WP Tren Patagónico"),
        ("Tren del Valle", "", 21.0, "es.WP, Cipolletti - Plottier"),
        ("Tren al Desarrollo", "", 8.0, "es.WP, Forum - La Banda"),
        ("Tren Solar de la Quebrada", "", 42.0, "es.WP, Volcán - Tilcara"),
        ("Metrotranvía de Mendoza", "", 17.0, "es.WP, Gutiérrez - Avellaneda: ~0.97"),
    ],
    # Chile: the legal lines' passenger stretches, or the one service that runs a stretch
    # whole (cl_register.py, cl_sources.md). es.WP infoboxes of the services.
    "cl": [
        ("Línea Central Sur (Alameda – Chillán)", "", 397.6, "es.WP Tren Estación Central - "
                                                            "Chillán"),
        ("Línea Central Sur (Victoria – Pitrufquén)", "", 95.1, "es.WP Tren Victoria - Temuco "
                                                               "65.5 + Tren Pitrufquén - "
                                                               "Temuco 29.6"),
        ("Tren Laja – Talcahuano", "", 87.3, "es.WP"),
        ("Tren Talca – Constitución", "", 88.0, "es.WP"),
        ("Tren Limache – Puerto", "", 43.0, "es.WP"),
    ],
    # Central Asia: the tariff guide's sections (casia_register.py, casia_sources.md). Every
    # line is also checked against its own tariff km (km_official); outside figures are few.
    "kz": [
        ("Ақтоғай — Достық", "", 304.0, "WD Q12532221 Aktogay - Dostyk; tariff 310 to Dostyk"),
    ],
    "uz": [
        ("Angren — Pop-1", "", 123.1, "WD Q24088913 Angren - Pap; tariff 124"),
    ],
    "tm": [
        ("Türkmenabat demirýol menzili — Gazojak", "", 322.0,
         "railway.gov.tm timetable, train 609 Türkmenabat km 0 - Gazojak km 322"),
        ("Мары — Serhetabat", "", 316.0,
         "railway.gov.tm timetable, train 601 Mary km 343 - Serhetabat km 659; built short: "
         "Saryýazy - Sandykgaçy's trace is rejected (casia_sources.md)"),
    ],
    # Saudi Arabia (mideast_register.py; sa_sources.md). en.WP.
    "sa": [
        ("قطار الشمال: الرياض – القريات", "", 1242.0, "en.WP Riyadh–Qurayyat railway"),
        ("قطار الشرق: الرياض – الدمام", "", 449.0, "en.WP Dammam–Riyadh railway"),
        ("قطار الحرمين السريع", "", 453.0, "en.WP Haramain high-speed railway"),
    ],
    # Iraq (mideast_register.py; iq_sources.md).
    "iq": [
        ("الخط الجنوبي: بغداد – البصرة", "", 552.9,
         "en.WP IRR Southern Line, Basra Maqal's km post"),
        ("الخط الشمالي: بغداد – سامراء", "", 120.0, "about 120 (en.WP Rail transport in Iraq)"),
        ("الخط الغربي: بغداد – الفلوجة", "", 65.0, "Shafaq News 2023, 65 km"),
    ],
    # Jordan (mideast_register.py; jo_sources.md).
    "jo": [
        ("سكة حديد الحجاز: عمان – الجيزة", "", 37.3,
         "en.WP Hejaz railway's chainage: Amman km 222.4, Al-Jizah km 259.7"),
    ],
    # The UAE: no published length for Etihad Rail's passenger route (ae_sources.md); its
    # journey times check it instead (Al Dhaid - Fujairah 25 min for 59 km).
    # Kenya (eafrica_register.py; ke_sources.md).
    "ke": [
        ("Mombasa – Nairobi SGR", "", 472.0, "en.WP Mombasa–Nairobi SGR; built station to "
                                             "station (~0.98)"),
    ],
    # Ethiopia and Djibouti: en.WP's station table, the line's chainage from Sebeta (Furi-Lebu
    # 15.5, Dewele 663.1, Nagad 743.9); the border is 4.0 km past Dewele on OSM's track.
    "et": [
        ("Addis Ababa – Djibouti Railway", "", 651.6, "Furi-Lebu - Dewele 647.6 + 4.0"),
    ],
    "dj": [
        ("Chemin de fer Addis-Abeba – Djibouti", "", 76.8, "Dewele - Nagad 80.8 less 4.0"),
    ],
    # Mozambique: CFM's line pages and AIM (mz_sources.md).
    "mz": [
        ("Linha de Ressano Garcia", "", 88.0, "CFM, Maputo - Ressano Garcia"),
        ("Linha de Machipanda", "", 317.0, "AIM Nov 2023, Beira - Machipanda"),
    ],
    # TAZARA: en.WP's chainage from Dar es Salaam (Tunduma 969.6, Nakonde 971.0, New Kapiri
    # Mposhi 1,860); the border taken at 970.3.
    "zm": [
        ("TAZARA: New Kapiri Mposhi – Nakonde", "", 889.7, "1,860 less 970.3"),
    ],
    # Zimbabwe: seat61's distances (zw_sources.md).
    "zw": [
        ("Bulawayo – Victoria Falls", "", 472.0, "seat61"),
        ("Harare – Mutare", "", 273.0, "seat61 (~0.98)"),
        ("Bulawayo – Gweru – Harare", "", 486.0, "seat61 (~0.98); greyed"),
    ],
    # Tanzania (tz_sources.md). TAZARA: en.WP's chainage to the border, as Zambia's.
    "tz": [
        ("SGR: Dar es Salaam – Dodoma", "", 444.0, "TRC"),
        ("Central Line: Dar es Salaam – Kigoma", "", 1254.0, "en.WP Central Line (Tanzania); "
                                                             "built from Kamata"),
        ("Mwanza Line: Tabora – Mwanza", "", 378.0, "TRC"),
        ("Mpanda Line: Kaliua – Mpanda", "", 210.0, "TRC"),
        ("TAZARA: Dar es Salaam – Tunduma", "", 970.3, "Tunduma 969.6, Nakonde 971.0"),
    ],
    # Madagascar (mg_sources.md).
    "mg": [
        ("FCE : Fianarantsoa – Manakara", "", 163.0, "en.WP Fianarantsoa-Côte Est railway"),
    ],
    # Malawi: CEAR's km in its 2015 timetable (fahrplancenter.com; mw_sources.md).
    "mw": [
        ("Limbe – Balaka", "", 112.0, "CEAR"),
        ("Nkaya – Nayuchi", "", 99.0, "CEAR"),
    ],
    # Nairobi - Suswa SGR (greyed) has no length of its own: the 120 km published for phase
    # 2A runs to Naivasha's container depot, past the passenger station (built 99.9).
    # North Korea (kp agent, 2026-10-08; kp_sources.md): en.wikipedia's line tables (after
    # the South Korean literature). Running lines first, then the greyed ones.
    "kp": [
        ("평라선", "", 782.8, "Kalli - Rajin (P'yŏngyang - Kalli is the P'yŏngŭi Line's)"),
        ("평의선", "", 226.4, "Sinŭiju Ch'ŏngnyŏn 225.4 + ~1 km on to the border on the bridge"),
        ("만포선", "", 302.8, "Sunch'ŏn - Manp'o Ch'ŏngnyŏn 299.8 + ~3 km to the Ji'an bridge"),
        ("북부내륙선", "", 249.2, "Manp'o Ch'ŏngnyŏn - Hyesan Ch'ŏngnyŏn"),
        ("함북선", "", 222.0, "Ch'ŏngjin - Onsŏng 180.4 and Mulgol - Rajin 41.6, two pieces "
                             "either side of the greyed Onsŏng - Mulgol"),
        ("평덕선", "", 192.3, "Taedonggang - Kujang Ch'ŏngnyŏn"),
        ("평부선", "", 187.3, "P'yŏngyang - Kaesŏng"),
        ("강원선", "", 145.8, "Kowŏn - P'yŏnggang"),
        ("백두산청년선", "", 141.7, "Kilju Ch'ŏngnyŏn - Hyesan Ch'ŏngnyŏn"),
        ("청년이천선", "", 141.3, "P'yŏngsan - Sep'o Ch'ŏngnyŏn"),
        ("평북선", "", 120.5, "Chŏngju Ch'ŏngnyŏn - Ch'ŏngsu"),
        ("은률선", "", 117.8, "Ŭnp'a - Ch'ŏlgwang (the Sariwŏn spur not built)"),
        ("금강산청년선", "", 101.0, "Anbyŏn - Kŭmgangsan Ch'ŏngnyŏn, the part in use"),
        ("황해청년선", "", 91.4, "Sariwŏn Ch'ŏngnyŏn - Haeju Ch'ŏngnyŏn"),
        ("신흥선", "", 91.6, "Hamhŭng - Sinhŭng 41.0 + Sinhŭng - Pujŏnhoban 50.6"),
        ("평남선", "", 85.4, "P'yŏngyang - P'yŏngnam Onch'ŏn 89.3 less P'yŏngyang - "
                             "Pot'onggang 3.9, which is P'yŏngŭi Line track here"),
        ("허천선", "", 80.3, "Tanch'ŏn Ch'ŏngnyŏn - Honggun"),
        ("금골선", "", 83.4, "Yŏhaejin - Muhak; OSM's stations end at Taesin (68.9), so "
                             "expect about 0.92 or more where the track winds"),
        ("장진선", "", 58.6, "Yŏnggwang - Sasu"),
        ("무산선", "", 57.9, "Komusan - Musan"),
        ("덕성선", "", 51.7, "Sinbukch'ŏng - Sangri"),
        ("백마선", "", 39.6, "Yŏmju - South Sinŭiju"),
        ("덕현선", "", 37.3, "South Sinŭiju - Tŏkhyŏn"),
        ("개천선", "", 29.5, "Sinanju Ch'ŏngnyŏn - Kaech'ŏn"),
        ("서해갑문선", "", 26.7, "Ch'ŏlgwang - Sillyŏngri"),
        ("룡강선", "", 18.3, "Ryonggang - Mayŏng"),
        ("고원탄광선", "", 17.6, "Tunjŏn - Changdong"),
        ("장연선", "", 17.7, "Sugyo - Changyŏn"),
        ("세천선", "", 14.4, "Sinhakp'o - Chungbong"),
        ("서창선", "", 12.9, "Tŏkch'ŏn - Ch'ŏlgisan 8.7 + the Hyŏngbong Line 4.2"),
        ("천성탄광선", "", 11.5, "Sinch'ang - Ch'ŏnsŏng 9.2; built from Suyang, the trains' "
                               "way in, ~2.3 km more"),
        ("만덕선", "", 10.3, "Hŏch'ŏn - Mandŏk"),
        ("홍의선", "", 9.5, "Hongŭi - Tumangang; OSM draws Chŏkchi - Tumangang twice, as this "
                            "and 두만강선, so built ~0.7"),
        ("함북선 (온성 - 물골)", "", 103.1, "greyed"),
        ("평부선 (판문 - 개성)", "", 10.3, "greyed"),
        ("금강산청년선 (감호 - 금강산청년)", "", 13.8, "greyed"),
        ("백무선", "", 191.7, "greyed; Paeg'am Ch'ŏngnyŏn - Musan, OSM in two pieces"),
        ("삼지연선", "", 64.0, "greyed; the new standard-gauge line to Samjiyŏn Motka"),
        ("배천선", "", 56.6, "greyed; Changbang - Ŭnbit (64.4 from East Haeju)"),
        ("강계선", "", 56.8, "greyed"),
        ("옹진선", "", 43.5, "greyed"),
        ("부포선", "", 19.1, "greyed"),
        ("송림선", "", 11.3, "greyed; to Songrim Ch'ŏngnyŏn"),
        ("비날론선", "", 14.1, "greyed"),
    ],
}

KNOWN = {
    # The Pyongyang Metro (OSM lines; kp agent, 2026-10-08). en.wikipedia's round figures,
    # "approximately 12 km" and "approximately 10 km", do not fit the track OSM draws (8.7 and
    # 12.3 km station to station); the stops (8 each) do. Left failing until a better figure.
    "kp": [
        ("천리마선", "", 12.0, 8, "en.WP Puhŭng - Pulgŭnbyŏl, \"approximately\": ~0.73"),
        ("혁신선", "", 10.0, 8, "en.WP Kwangbok - Ragwŏn, \"approximately\": ~1.23"),
    ],
    # ---- the rest of Asia (OSM lines; asia agent, 2026-10-08). en.WP infoboxes.
    "ph": [
        ("LRT Line 1", "", 26.0, 25, "Fernando Poe Jr. - Dr. Santos (Cavite extension phase 1, "
                                     "Nov 2024); built station to station (0.95)"),
        ("LRT Line 2", "", 17.6, 13, "Recto - Antipolo; built station to station (0.94)"),
        ("MRT Line 3", "", 16.9, 13, "North Avenue - Taft Avenue"),
    ],
    "mm": [
        ("Yangon Circular Railway", "", 45.9, 38, "en.WP Yangon Circular Railway"),
    ],
    # ---- end the rest of Asia
    # ---- Latin America's metros and trams (OSM lines; latam agent, 2026-10-08)
    # Cochabamba's Mi Tren: Los Tiempos (Sep 2023, Apr 2024: Línea Verde 27 km to Suticollo).
    "bo": [
        ("Línea Verde", "Mi Tren", 27.0, None, "San Antonio - Suticollo"),
    ],
    # Quito's metro and Cuenca's tram: the operators' figures (ec_sources.md).
    "ec": [
        ("Metro Línea 1", "", 22.6, 15, "Quitumbe - El Labrador, with the tails: ~0.95"),
        ("Cuatro Ríos", "Cuenca", 10.7, None, "Río Tarqui - Parque Industrial: ~0.93"),
    ],
    # Santo Domingo Metro: en.WP / OPRET (2C open since 24 Feb 2026).
    "do": [
        ("Línea 1", "OPRET", 14.5, 16, "Mamá Tingó - Centro de los Héroes, with the tails: "
                                       "~0.90"),
        ("Línea 2", "OPRET", 21.0, 23, "Pablo Adón Guzmán - Concepción Bona, 2A + 2B + 2C"),
    ],
    # Puerto Rico's Tren Urbano (OSM line): en.WP.
    "pr": [
        ("Tren Urbano", "", 17.2, 16, "Bayamón - Sagrado Corazón, with the tails: ~0.96"),
    ],
    # Medellín's metro (OSM lines): en.WP. OSM has Poblado twice (a stop and "Estación del
    # Metro Poblado"), so Línea A shows 22 stations for 21.
    "co": [
        ("Metro Línea A", "", 25.8, 21, "Niquía - La Estrella"),
        ("Metro Línea B", "", 5.5, 7, "San Antonio - San Javier"),
    ],
    # Venezuela's metros (OSM lines): urbanrail.net / es.WP.
    "ve": [
        ("Línea 1 del Metro de Caracas", "", 20.4, 22, "Propatria - Palo Verde"),
        ("Metro Los Teques", "", 10.7, 5, "Las Adjuntas - Independencia (survey)"),
    ],
    # Lima Metro: en.WP.
    "pe": [
        ("Línea 1", "AATE", 34.6, 26, "Villa El Salvador - Bayóvar: ~0.96"),
        ("Línea 2", "Línea 2", 5.0, 5, "stage 1A Evitamiento - Mercado Santa Anita, news "
                                       "figure 5 km (rounded; takes in the tunnel past "
                                       "both ends): station to station 4.1, ~0.83"),
    ],
    # Panama Metro: en.WP; published lengths take in the tails, so station to station is short.
    "pa": [
        ("Línea 1", "Metro de Panamá", 18.1, 15, "Albrook - Villa Zaita: ~0.95"),
        ("Línea 2", "Metro de Panamá", 22.5, 18, "20.4 + the airport branch 2.1; the branch's "
                                                 "Corredor Sur makes 18 stations, ITSE and "
                                                 "Aeropuerto 2 more"),
    ],
    # ---- end Latin America
    # Almaty's and Tashkent's metros and Astana's LRT (OSM lines; casia agent). Published
    # lengths: Wikidata (Almaty Q484433, Astana Q779673, Chilonzor Q4515924), en.WP's
    # Tashkent Metro line table; published lengths take in depot tails.
    "kz": [
        ("Первая линия", "", 13.4, 11, "Almaty Metro, WD"),
        ("Astana LRT", "", 22.4, 18, "WD; opened 16 May 2026; all 18 stations, built station "
                                     "to station (0.94)"),
    ],
    "uz": [
        ("Chilonzor yoʻli", "", 23.7, 17, "WD/en.WP; all 17 stations, built station to station "
                                          "(0.91)"),
        ("Узбекистанская линия", "", 14.3, 11, "en.WP"),
        ("Юнусабадская линия", "", 10.5, 8, "en.WP (to Shahriston); all 8 stations, built "
                                           "station to station (0.90)"),
        ("Circle Line", "", 21.9, 14, "en.WP, after the March 2025 extension"),
    ],
    # Iran's metros (OSM lines; ir agent). en.WP infoboxes and line tables; published lengths
    # take in depot tails and unopened ends, so station-to-station runs short.
    "ir": [
        ("خط ۱", "Tehran", 92.0, 32, "Tehran L1 with the Parand and airport branches, en.WP"),
        ("خط ۲", "Tehran", 26.0, 22, "Tehran L2, en.WP"),
        ("خط ٣", "Tehran", 37.0, 25, "Tehran L3, en.WP (short: station to station)"),
        ("خط ۴", "Tehran", 26.0, 25, "Tehran L4, en.WP; + Mehrabad branch 2.8"),
        ("خط ۵", "Tehran", 69.0, 13, "Tehran L5 Sadeghiyeh - Hashtgerd, en.WP"),
        ("خط ۶", "Tehran", 32.0, 26, "Tehran L6, en.WP"),
        ("خط ۷", "Tehran", 28.0, 22, "Tehran L7, en.WP"),
        ("خط ۱", "Mashhad", 24.0, 24, "Mashhad L1, en.WP"),
        ("خط ۲", "Mashhad", 14.5, 13, "Mashhad L2, en.WP"),
        ("خط ۱", "شیراز", 22.5, 20, "Shiraz L1, en.WP"),
        ("خط ۱", "اصفهان", 20.2, 20, "Isfahan L1, en.WP"),
        ("خط یک قطار شهری تبریز", "", 17.2, 12, "Tabriz L1, en.WP (OSM lists 18 stops; WP: 6 "
                                                "intermediate not open)"),
    ],
    # Lahore Metro (OSM line; pk agent). en.WP infobox.
    "pk": [
        ("Orange Line", "", 27.1, 26, "Ali Town - Dera Gujran, en.WP; OSM's routes run "
                                      "station to station, 25.3 (~0.93)"),
    ],
    # Cairo Metro (OSM lines; nafrica agent). en.WP infoboxes.
    "eg": [
        ("الخط الأول", "", 44.3, 35, "Helwan - New El Marg, en.WP"),
        ("الخط الثاني", "", 21.6, 20, "Shubra El Kheima - El Mounib, en.WP"),
    ],
    # Buenos Aires' Subte (OSM lines). es.WP's table, commercial lengths.
    "ar": [
        ("Línea A", "", 9.7, 18, "Plaza de Mayo - San Pedrito"),
        ("Línea B", "", 11.8, 17, "Leandro N. Alem - Juan Manuel de Rosas"),
        ("Línea C", "", 4.4, 9, "Retiro - Constitución"),
        ("Línea D", "", 10.4, 16, "Catedral - Congreso de Tucumán"),
        ("Línea E", "", 11.9, 18, "Retiro - Plaza de los Virreyes"),
        ("Línea H", "", 8.8, 12, "Facultad de Derecho - Hospitales; OSM's routes 8.0: ~0.91"),
    ],
    # Santiago's Metro (OSM lines). es.WP's article text; published lengths take in tails.
    "cl": [
        ("Línea 1", "Metro S.A.", 20.0, 27, "San Pablo - Los Dominicos: ~0.95"),
        ("Línea 2", "Metro S.A.", 25.9, 26, "Vespucio Norte - Hospital El Pino: ~0.94"),
        ("Línea 4", "Metro S.A.", 24.7, 23, "Tobalaba - Plaza de Puente Alto: ~0.94"),
        ("Línea 6", "Metro S.A.", 15.0, 10, "Cerrillos - Los Leones: ~0.96"),
    ],
    # Vietnam's metros (OSM lines). en.WP; published lengths take in depot tails.
    "vn": [
        ("Tuyến số 2A", "", 13.05, 12, "Hà Nội 2A Cát Linh - Yên Nghĩa"),
        ("Tuyến số 3", "", 8.5, 8, "Hà Nội 3, the elevated Nhổn - Cầu Giấy open since Aug 2024 "
                                   "(the 4 km underground to Ga Hà Nội opens end 2027)"),
        ("Line 1 (Ho Chi Minh City Metro)", "", 19.7, 14, "HCMC 1 Bến Thành - Suối Tiên"),
    ],
    # KTM Komuter's lines (OSM lines over the register's KTM lines). WP infoboxes; RD as in
    # REGISTER["my"].
    "my": [
        ("Seremban Line", "", 135.0, 27, "Batu Caves - Pulau Sebang/Tampin, WP"),
        ("Port Klang Line", "", 126.0, 35, "Tanjung Malim - Pelabuhan Klang, WP"),
        ("Butterworth–Padang Besar", "", 169.8, 13, "WP; RD 168.0"),
        ("Butterworth–Ipoh", "", 181.0, 13, "RD; WP's infobox says 162: ~0.96 on RD"),
    ],
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
    # Bangkok's urban lines (OSM lines). Published lengths take in the track past the end
    # stations, so station-to-station runs a few per cent short.
    "th": [
        ("สายสีแดงเข้ม", "", 26.3, 10, "EN, Krung Thep Aphiwat - Rangsit; SRT's chainage, "
                                      "Bang Sue km 7.5 to Rangsit km 30.5, gives ~23: ~0.86"),
        ("สายสีแดงอ่อน", "", 15.26, 4, "Krung Thep Aphiwat - Taling Chan, EN"),
        ("รถไฟฟ้าเชื่อมท่าอากาศยานสุวรรณภูมิ", "", 28.6, 8, "Airport Rail Link, EN"),
        ("รถไฟฟ้าบีทีเอส สายสุขุมวิท", "", 54.25, 47, "EN, Khu Khot - Kheha with the depot "
                                                    "tails: ~0.94"),
        ("รถไฟฟ้าบีทีเอส สายสีลม", "", 14.0, 14, "EN: ~0.94"),
        ("รถไฟฟ้ามหานคร สายสีน้ำเงิน", "", 48.0, 38, "EN: ~0.97"),
        ("รถไฟฟ้ามหานคร สายสีม่วง", "", 23.0, 16, "EN, with the depot: ~0.91"),
        ("รถไฟฟ้าสายสีเหลือง", "", 30.4, 23, "EN: ~0.94"),
        ("รถไฟฟ้ามหานคร สายสีชมพู", "", 34.5, 30, "EN, main line: ~0.98"),
    ],
    # Kyiv's, Kharkiv's and Dnipro's metros (OSM lines; ua agent, 2026-10-03).
    "ua": [
        ("M1 line", "", 22.64, 18, "Kyiv Святошинсько-Броварська, WD"),
        ("M2 line", "", 20.95, 18, "WD"),
        ("M3 line", "", 23.86, 16, "WD"),
        ("Холодногірсько-Заводська", "", 17.3, 13, "Kharkiv, WD"),
        ("Олексіївська", "", 10.98, 10, "Kharkiv, WD"),
        ("Салтівська", "", 10.2, 8, "en.wikipedia"),
        ("Дніпровський метрополітен", "", 7.8, 6, "en.wikipedia; counts to the depot, built 7.0"),
    ],
    # The Minsk Metro (OSM lines; by-md agent, 2026-10-03).
    "by": [("Маскоўская лінія", "", 19.2, 15, "Minsk Metro 1, WD Q28604"),
           ("Аўтазаводская лінія", "", 18.1, 14, "Minsk Metro 2, WD Q2638932")],
    # Tbilisi's and Baku's metros (OSM lines; caucasus agent, 2026-10-03).
    "ge": [("ახმეტელი-ვარკეთილის ხაზი", "", 19.6, 16, "en.WP"),
           ("საბურთალოს ხაზი", "", 7.7, 7, "en.WP")],
    "az": [("Line 1", "Bakı Metropoliteni", 20.1, 13, "en.WP"),
           ("Line 3", "Bakı Metropoliteni", 6.1, 4, "en.WP"),
           ("Line 2B", "Bakı Metropoliteni", 2.3, 2, "en.WP")],
    # Jakarta's and Palembang's urban lines and KAI Commuter's (OSM lines). en.WP infoboxes;
    # published lengths take in depot tails, so station-to-station runs a little short.
    "id": [
        ("MRT North-South Line", "", 15.7, 13, "en.WP, Lebak Bulus - Bundaran HI"),
        ("LRT Jakarta", "", 12.2, 11, "en.WP, Kelapa Gading - Velodrome 5.8 + to Manggarai 6.4"),
        ("Jabodebek LRT Bekasi Line", "", 29.5, 14, "en.WP, Dukuh Atas - Cawang 11.1 + "
                                                     "Cawang - Jatimulya 18.4"),
        ("Jabodebek LRT Cibubur Line", "", 25.4, 12, "en.WP, Dukuh Atas - Cawang 11.1 + "
                                                      "Cawang - Harjamukti 14.3"),
        ("LRT Palembang", "", 23.4, 13, "en.WP"),
        ("Lin Tangerang", "", 19.3, 11, "KAI km posts, Tangerang km 19.297 - Duri 0.000"),
        ("Lin Rangkasbitung", "", 72.8, 19, "en.WP, Tanah Abang - Rangkasbitung"),
    ],
    # Riyadh Metro (OSM lines; mideast agent). en.WP's line table.
    "sa": [
        ("Blue Line", "", 38.0, 25, "Line 1"),
        ("Red Line", "", 25.3, 15, "Line 2"),
        ("Orange line", "", 40.7, 22, "Line 3"),
        ("المسار 4 - الخط الأصفر", "", 29.6, 9, "Line 4, over Line 6's track to KAFD"),
        ("Green Line", "", 12.9, 12, "Line 5"),
        ("Purple Line", "", 29.9, 11, "Line 6"),
    ],
    # Dubai Metro (OSM lines). en.WP; the Red Line's 67.1 takes in Route 2020 and the tails.
    "ae": [
        ("Red Line", "Dubai", 67.1, 35, "with Route 2020 (15 km) and the tails past both "
                                        "ends: station to station ~0.93"),
        ("Green Line", "Dubai", 22.5, 20, ""),
    ],
    # Doha Metro (OSM lines). en.WP.
    "qa": [
        ("الخط الأحمر للمترو", "", 40.0, 18, "with the airport branch"),
        ("الخط الأخضر للمترو", "", 22.0, 11, ""),
        ("الخط الذهبي للمترو", "", 14.0, 11, ""),
    ],
    # Addis Ababa's light rail (OSM lines; eafrica agent). en.WP.
    "et": [
        ("Addis Ababa LRT East–West", "", 17.35, 22, "Ayat - Tor Hailoch"),
        ("Addis Ababa LRT North–South", "", 16.9, 22, "Menelik II Square - Kality"),
    ],
    # Mauritius' Metro Express (OSM lines, no register; eafrica agent). mu_sources.md.
    "mu": [
        ("Metro Express: Port Louis – Curepipe", "", 26.0, 19, "Port Louis Victoria - Curepipe "
                                                               "Central"),
        ("Metro Express: Rose Hill – Réduit", "", 3.4, 3, "Rose Hill - Mahatma Gandhi; built "
                                                          "platform to platform (~0.89)"),
    ],
    # Tenerife's tram, folded into es from the Canary Islands extract (es agent, 2026-10-08;
    # canaries_survey.md). en.WP; built stop to stop, so a little short.
    "es": [
        ("Tranvía Línea 1", "MetroTenerife", 12.5, 21, "Intercambiador - La Trinidad"),
        ("Tranvía Línea 2", "MetroTenerife", 3.6, 6, "La Cuesta - Tíncer: ~0.95"),
    ],
    # Brazil's metros, light rail and the like (OSM lines; br_sources.md). WD = Wikidata
    # P2043, read 2026-10-03 (data/raw/br/wikidata_lines.json).
    "br": [
        ("Linha 1 - Azul", "Metropolitano", 20.4, 23, "WD, Jabaquara - Tucuruvi"),
        ("Linha 3 - Vermelha", "Metropolitano", 22.0, 18, "WD"),
        ("Linha 4 - Amarela", "ViaQuatro", 12.8, 11, "WD, Luz - Vila Sônia"),
        ("Linha 5 - Lilás", "ViaMobilidade", 19.9, 17, "WD, Capão Redondo - Chácara Klabin"),
        ("Metrô Rio Linha 1", "", 16.0, None, "WD (Q2333617), Uruguai - General Osório"),
        ("Metrô Rio Linha 2", "", 30.0, None, "WD (Q2333639), Pavuna - Botafogo"),
        ("Linha 1", "Trensurb", 43.4, 22, "WD, Mercado - Novo Hamburgo"),
        ("Linha Norte", "CBTU", 38.5, None, "WD, Natal's Linha Norte"),
        ("Trem do Corcovado", "", 3.824, 4, "WD, Cosme Velho - Corcovado"),
        ("Aeromóvel GRU", "", 2.7, None, "WD"),
        ("Linha 1 do VLT da Baixada Santista", "", 11.5, 15, "WD, Barreiros - Porto: ~0.90"),
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
