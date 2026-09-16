# Southern Kyushu + Okinawa: ferry passenger sources

Region: Kumamoto, Oita, Miyazaki, Kagoshima, Okinawa. Researched 2026-09-15.
Figures are in `kyushu_south_okinawa.csv`.

## Best sources

1. **Okinawa: 沖縄県「離島関係資料」chapter 3, table (6) 旅客定期航路事業輸送実績.**
   Covers every 一般旅客定期航路 in Okinawa, by operator and route: sailings,
   passengers and passenger-km. The table is copied from 沖縄総合事務局運輸部
   「運輸要覧」. Each year was taken from the newest edition that has it:
   - 令和8年3月 edition, FY2020–FY2024, p.80–83 (PDF 19–22):
     https://www.pref.okinawa.lg.jp/_res/projects/default_project/_page_/001/038/825/03.chapter3.r8.pdf
   - 令和6年3月 edition, FY2018–FY2022, p.80–83 (PDF 18–21):
     https://www.pref.okinawa.lg.jp/_res/projects/default_project/_page_/001/028/067/03_chapter3_r6.pdf
   - 運輸要覧 令和3年12月, FY2016–FY2020, p.71–73:
     https://www.ogb.go.jp/-/media/Files/OGB/Unyu/survey/R3unyuyoran_260414.pdf
   - 運輸要覧 平成29年12月, FY2012–FY2016, p.65–67 (used for FY2015 only):
     https://www.ogb.go.jp/-/media/Files/OGB/Unyu/survey/H29unyuyoran_250729.pdf
   - Lists of editions: 離島関係資料 https://www.pref.okinawa.lg.jp/kensei/tokei/1016451/index.html
     (H31 to R8); 運輸要覧 https://www.ogb.go.jp/unyu/unyu_youran (H16 to R7).
   - Check: for every year FY2015–FY2024 the transcribed passengers add up
     exactly to the table's 合計 row (FY2015 only after the two fixes listed
     under Oddities). Most passenger-km values also equal passengers × route km.
   - Route km come from table (4) 離島航路の現況（旅客定期）in the same chapter
     (R8 edition p.72–78), which is also in 運輸要覧 R7 p.24–26.
2. **Sakurajima Ferry: 鹿児島市船舶事業概要, 業務量の推移表 (p.26).** The
   令和2年度版 has FY2015–2019 and the 令和7年度版 has FY2020–2024. All editions
   and the annual accounts are listed at
   https://www.city.kagoshima.lg.jp/sakurajima-ferry/gaiyo/keiei.html
3. **天草市地域公共交通計画, p.20 fig. 26.** Three ferries from Amakusa to other
   prefectures, FY2017–2021: 牛深–蔵之元, 鬼池–口之津 and 中田–片側・諸浦.
   https://www.city.amakusa.kumamoto.jp/kiji00310511/3_10511_69480_up_7c888rsg.pdf
4. Single figures:
   - 九州運輸局「主要離島航路の状況」: FY2022 totals, in thousands, for three
     island groups (甑島, 種子・屋久, 奄美・沖縄).
   - 宮崎県離島振興計画: 島野浦 and 大島, FY2021 only.
   - 佐伯市地域公共交通計画 概要版: its two municipal routes combined, FY2022.

## Route lists (for judging coverage)

- **Okinawa:** 離島関係資料 R8 ch.3 table (4), p.72–78. Every route listed there
  appears in table (6), except 日東商船 知名–与論–塩屋 (licensed 2021, not yet running).
- **Number of 一般旅客定期 routes by prefecture, 2025-04-01:** Kumamoto 11,
  Oita 10, Miyazaki 3, Kagoshima 26. Source: 九州運輸局
  https://wwwtb.mlit.go.jp/kyushu/content/000369814.pdf p.2. Page 6 of the same
  file lists the main island routes with km (as of 2026-01-01).
- **Kagoshima operators and routes:** 鹿児島県地域公共交通計画 (2024-03) 表5-2,
  p.57: http://www.pref.kagoshima.jp/ac08/koutuukeikaku/documents/112595_20240329105535-1.pdf
- **Main Kagoshima island routes, with km and vessels:** 鹿児島運輸支局 業務概況
  令和5年版 p.24: https://wwwtb.mlit.go.jp/kyushu/content/000358245.pdf
- **Operators by area** (九州旅客船協会連合会 member maps):
  https://kyushu-ships.com/pages/10/ (Kumamoto/Amakusa), /12/ (Oita),
  /13/ (Miyazaki), /14/ (Kagoshima), /15/ (Kagoshima islands), /43/ (Ariake Sea)
- **Amami routes:** 奄美群島の概況 令和7年度 4-5 航路の現況, p.64:
  http://www.pref.kagoshima.jp/aq01/chiiki/oshima/chiiki/zeniki/gaikyou/documents/127876_20260423151440-1.pdf
- **Amakusa island routes, with km:** 熊本県離島振興計画 表6, p.16:
  https://www.mlit.go.jp/kokudoseisaku/chirit/content/001628433.pdf

## What was checked, by prefecture

### Okinawa
- Found: the table above. The 運輸要覧 R4–R7 editions were not opened
  separately, because 離島関係資料 R8 reproduces R7.
- Kudaka, Tsuken, Minna, Iheya, Izena, Tonaki, the Daito islands, Tarama,
  Ogami and all Yaeyama routes are in the table.
- 宮古フェリー and はやて (平良–佐良浜) closed in February 2015 when Irabu
  Bridge opened, so they have no rows from FY2015.
- The Kagoshima–Naha ferries (Marix, A-Line) come under the Kyushu bureau and
  are not in this table.

### Kagoshima
- Found: Sakurajima; 甑島商船 FY2022; two island-group totals; 天長フェリー
  (from the Amakusa plan).
- Checked, no per-route passenger figures:
  - 鹿児島県地域公共交通計画, ch.5: route list only.
  - 奄美群島の概況 R7, ch.4: ship routes are listed, but passenger counts are
    given only for air routes.
  - 鹿児島県離島振興計画 R5–14, island sections (獅子島 p.19, 甑島 p.51,
    種子島 p.85, 屋久島 p.117): timetables only.
  - 鹿児島運輸支局 業務概況 R5: prefecture totals only (scheduled routes FY2022:
    5,949 thousand), plus one total for its 7 operators / 8 subsidised routes
    (FY2019 449,978; FY2022 376,927), not split by route.
  - 垂水市地域公共交通計画 (2024): bus and taxi ridership only, no ferry figure.
  - Prefecture 船舶 page, 屋久島町 フェリー太陽II page, 十島村 site: no figures.
    The 三島村 site would not load (expired certificate).
- Not found: 鹿児島県統計年鑑 (two guessed URLs returned 404, and the web-search
  budget ran out before it could be located); 鹿児島市地域公共交通計画;
  薩摩川内市 甑島 data; 瀬戸内町 data.

### Kumamoto
- Found:
  - 牛深–蔵之元 and 鬼池–口之津, from the Amakusa plan.
  - 熊本県統計年鑑 R5 table 11-04 有明海自動車航送船事業実績. Excel only, so it is a
    needs_download row.
- No figures in:
  - 熊本県離島振興計画: routes and km only.
  - The Amakusa plan's island routes (御所浦): no ridership.
  - Other tables in the 熊本県統計年鑑 transport chapter: no other ship table.

### Oita
- Found: only 佐伯市's combined municipal-route figure.
- No figures in:
  - 大分県統計年鑑 R6 and H29, ch.11: only passengers per port (港別船舶乗降人員)
    and the prefecture-to-prefecture table.
  - 大分県離島振興計画: 姫島 p.8 and 豊後諸島 p.29–31 describe the services but
    give no counts.
  - 佐伯市's ship-routes page, 津久見市's 保戸島 page, 姫島村's homepage.
- Not read:
  - The full 佐伯市地域公共交通計画 (fetch failed, over 10 MB):
    https://www.city.saiki.oita.jp/kiji0038705/3_8705_up_qdq0npjl.pdf
    It may have per-route charts.
  - 大分運輸支局 業務概況 H30: downloaded, but its pages did not render:
    https://wwwtb.mlit.go.jp/kyushu/content/000358283.pdf

### Miyazaki
- Found: 島野浦 FY2021 (86,447) and 大島 FY2021 (about 8,300), both from
  宮崎県離島振興計画. Miyazaki Car Ferry is long-distance and covered by another agent.

### Kyushu-wide bureau documents (background, not per route)
- Passengers by transport branch office, including unscheduled services
  (旅客不定期):
  - FY2024 (000369814.pdf p.1): Kagoshima 6,542,366; Oita 1,199,820;
    Kumamoto 624,360; Miyazaki 283,114.
  - FY2019 (000162197.pdf): Kagoshima 7,208,082; Oita 1,143,597;
    Kumamoto 744,461; Miyazaki 294,654.
- Long-distance ferries FY2024 (000358745.pdf): 志布志–大阪 and 宮崎–神戸 are
  published only as one combined total, 320,445.
- 地域公共交通確保維持改善事業 evaluation summary, R7.2.28 (000347226.pdf):
  written assessments only, no ridership.
- 九州運輸局 年度別輸送実績 page: aggregates only.

## Known gaps (sizeable routes with no figure)

- **Kagoshima:**
  - 鴨池–垂水 (垂水フェリー), probably the region's second-busiest route.
  - 山川–根占.
  - Per-operator figures for 鹿児島–種子島/屋久島 (Toppy/Rocket jetfoils,
    Cosmo Line, 岩崎産業, 折田汽船).
  - Per-operator figures for 鹿児島–奄美–那覇, and 奄美海運.
  - 十島村 フェリーとしま2 and 三島村 フェリーみしま.
  - 屋久島町 フェリー太陽II (宮之浦–口永良部 and 宮之浦–島間).
  - 瀬戸内町 (加計呂麻, 請島・与路).
  - 獅子島汽船 幣串–水俣 and 共同フェリー 阿久根–大島.
  - 甑島 for any year other than FY2022.
- **Kumamoto:** 熊本港–島原 (九商フェリー, 熊本フェリー); 三角–松島/前島
  (シークルーズ); 御所浦 routes (共同フェリー); 湯島–江樋戸; 富岡–茂木.
- **Oita:**
  - 姫島村営フェリー 伊美–姫島.
  - Crossings to Shikoku and Honshu: 佐賀関–三崎 (国道九四フェリー),
    臼杵–八幡浜 (九四オレンジフェリー), 別府・臼杵–八幡浜 (宇和島運輸),
    竹田津–徳山 (周防灘フェリー).
  - 津久見–保戸島 and 地無垢島.
  - 大入島 routes.
  - 佐伯–大島 and 蒲江–屋形島・深島 as separate routes.
  - 大分空港 hovercraft.
- **Miyazaki:** only one year (FY2021) for its two island routes.

## Oddities and definitions

### Okinawa table
- Passengers are both directions combined. Sailings count a one-way trip as 0.5.
- Half-passenger values probably mean children are counted as 0.5. The source
  states that rule only for the unscheduled (不定期) table.
- FY2015 column (運輸要覧 H29):
  - The published total (4,039,142) leaves out the 八重山観光フェリー 大原/黒島
    row (1,362). That route stopped running on 2016-03-31.
  - I first read 安栄観光 石垣/大原 as 189,531. The correct value is 189,931: it
    matches passenger-km ÷ 30.85 km (the ratio in every other year), and with
    both fixes the column adds up exactly.
  - Passenger-km is left blank for 渡嘉敷 and 神谷観光 FY2015, because the small
    print did not agree with passengers × route km.
- Passenger-km values that look wrong as published: 久高 FY2021 works out to
  10.7 km per passenger (8.6 in other years); 船浮 FY2024 to 7.0 km (3.5 in
  other years). For these, passengers × route km is safer.
- 座間味 FY2017 has exactly the same passengers and passenger-km as FY2018,
  though the sailings differ.
- The 竹富町 routes are licensed as a single 航路 (石垣・大原・竹富・黒島・小浜・鳩間),
  but the table still splits them by leg. Legs between islands (小浜/竹富 etc.)
  have no route km in the route list.
- 石垣島ドリーム観光 suspended all routes on 2018-04-01. Its zero rows, and other
  all-zero years, are left out of the CSV.
- 第一マリンサービス runs along the main island (泊–本部, later via 名護), not to an
  island. Its FY2020 泊–本部 figure (1,651) and FY2022 泊–名護–本部 figure
  (4,240.5) appear only in the R6 edition; the R8 edition drops both rows and
  its totals for those years are lower by the same amounts.
- 浦内川観光 is a river boat on 西表島 carried mostly by sightseers and hikers.
- 粟国 FY2020–21 (1,496 and 1,864 passengers) is far below other years, as published.

### Sakurajima
- The figure is the whole ferry bureau's total. Up to FY2022 that is three
  routes: the 桜島–鹿児島 shuttle, よりみちクルーズ (scheduled) and 鹿児島湾内周遊
  (unscheduled), 57.8 km combined. From FY2023 it is two routes, 46.8 km.
- The shuttle carries most passengers but is not published separately, so no
  route km is recorded.
- Commuter passes are counted as a fixed number of trips per pass (footnote p.27).

### Other sources
- 九州運輸局 island-group figures are in thousands, FY2022 only. Two of the
  three are sums over several operators and should not be split.
- Amakusa plan (出典: 天草市資料):
  - Values were read from the chart's data labels.
  - For FY2020–21 each label was matched to its line by position; each year's
    three routes add up to the printed 3航路合計.
  - 天長フェリー FY2019 is a round 64,000.
- 大島 (日南市) is published as "約8,300人", an approximate figure.
- 佐伯市's 13,042 covers both municipal routes. The 屋形島・深島 route only became
  municipal in October 2022, so FY2022 may be a part year for it.
- The source does not name the operator of 鬼池–口之津 or 島野浦. The CSV takes
  them from the 九州旅客船協会連合会 member list and says so in the operator field.
