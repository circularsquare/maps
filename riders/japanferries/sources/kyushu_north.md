# Northern Kyushu (Fukuoka, Saga, Nagasaki) — ferry passenger sources

Researched 2026-09-15. The session's web-search allowance ran out partway
through, after which only known URLs could be fetched. Several leads below are
therefore listed as "not opened" rather than checked.

## What the CSV holds

- **12 individual routes with a figure**: 8 in Goto city (FY2015–FY2020 run),
  Tsushima municipal Aso Bay route (FY2024), Kitakyushu's Wakato and Kokura
  routes (2020, secondary source), Fukuoka city Genkaijima route (about 80,000,
  about FY2013, approximate).
- **Aggregates, flagged in `route`**: ship passengers per island for Iki,
  Tsushima, Kamigoto and Shimogoto (FY2018–FY2024), and Shin-Kamigoto town's
  total ferry users (FY2015–FY2021). These are not routes, but they are the only
  figures found for the Kyushu Yusen and most Kyushu Shosen corridors.
- **2 Excel files to download** (Nagasaki city port passenger counts).

## Best sources

1. **五島市地域公共交通計画（変更）** (R4.3, amended R6.8), pp.21–24 —
   https://www.city.goto.nagasaki.jp/s050/010/020/030/kokyokotukeikaku060806.pdf
   Per-route 利用者数 H26–R2 for every private route touching Goto city (Kyushu
   Shosen Goto–Nagasaki, Nomo Shosen Fukue–Hakata, Kiguchi Kisen, Goto
   Ryokakusen, Oshima Kaiun, Sagashima Ryokakusen) and two municipal routes.
   Every year checks against the total row. Footnote says the data follow the
   subsidy period October–September. Same tables as the original R4.3 plan, so
   nothing after FY2020.
2. **九州運輸局 九州運輸要覧〔3〕旅客航路事業の現況（資料）** —
   https://wwwtb.mlit.go.jp/kyushu/content/000369814.pdf
   - p.1: FY2024 passengers per bureau office (includes 不定期 services):
     Fukuoka 602,009; Wakamatsu 432,676; Saga 358,403; Nagasaki 2,861,086.5;
     Sasebo 1,973,530. Useful for checking coverage.
   - p.6 table (6) 管内主要離島航路一覧 (as of 2026-01-01): route km for the big
     island routes: Hakata–Hitakatsu 146.3; Hakata–Iki 65.8 / –Izuhara 135.3;
     Indoshi–Karatsu 41.9; Sasebo–Kamigoto 107.6; Nagasaki–Goto 96.5;
     Nagasaki–Arikawa 85.7; Fukue–Aokata–Hakata 225.6; Tainoura–Nagasaki 80.0.
     p.7 has jetfoil leg distances.
   - p.10 table (10): ship vs air passengers per main island (thousands),
     H10, H15, H20, H25, H30, R1–R6. Iki, Tsushima, Kamigoto and Shimogoto are
     in the CSV. **The source's R1 ship total (2,872) is not the sum of its
     island rows (2,701)**; the island values agree with the share columns, so
     the total row is the error.
3. **新上五島町地域公共交通計画** (R5.3), p.13 — town-wide 航路利用者 H24–R3
   in 万人 (one decimal).
   https://official.shinkamigoto.net/cmd/dlfile.php?entryname=benri&entryid=06214&fileid=00000016&disp=inline
   (the older k101ow01.town.shinkamigoto.nagasaki.jp host refuses connections).

## Route lists (for judging coverage)

- 長崎運輸支局「業務概況 令和6年版」p.11, 管内 一般旅客定期航路 一覧表 (as of
  2024-03-31): 32 numbered routes with operator, subsidy status and vessels,
  for the Nagasaki and Sasebo offices. The text says 27 operators and 41 routes;
  Iki/Tsushima routes of Fukuoka-based Kyushu Yusen are not in the list.
  https://wwwtb.mlit.go.jp/kyushu/content/000347124.pdf (R2 edition, same
  layout: https://wwwtb.mlit.go.jp/kyushu/content/000235443.pdf p.9)
- 九州運輸局 main island routes list (above, p.6).
- 九州運輸局 二次評価 of 離島航路運営費等補助 (R7.2.28), which lists the subsidised
  routes by council: https://wwwtb.mlit.go.jp/kyushu/content/000347226.pdf
  (Fukuoka: Munakata Jinoshima and Oshima, Shingu Ainoshima, Fukuoka city
  Genkaijima and Oronoshima, Itoshima Himeshima, Kitakyushu Ainoshima; Saga:
  Karatsu Madarashima and Kakarashima; Nagasaki councils on later pages). It has
  grades only, no ridership.
- Karatsu city's 7 island routes with operators: https://www.city.karatsu.lg.jp/page/1013.html

## Checked, no per-route figures

**Nagasaki**
- 長崎運輸支局 業務概況 R6 and R2: route list, and prefecture totals for
  一般旅客定期航路 in thousands (FY2013 6,023; FY2015 7,114; FY2019 4,740;
  FY2020 2,948; FY2023 4,369). Nothing per route.
- 長崎県統計年鑑 (58th ed. table list, https://www.pref.nagasaki.jp/doc/page-2364.html):
  only 表125 船舶乗降人員, presumably per port. Not opened.
- 長崎県離島振興計画 R5.4 (https://www.mlit.go.jp/kokudoseisaku/chirit/content/001619212.pdf):
  the PDF's text renders blank when read (fonts not extractable), so it could
  not be checked.
- Nagasaki prefecture 航路情報 and 国境離島航路運賃低廉化 pages: WebFetch returned
  only the page header.
- 佐世保市・佐々町地域公共交通計画 R7.3: bus and rail only (contents list and
  appendix pp.65–74 checked).
- 平戸市「地域公共交通の現状と課題」(2018): describes the Takushima and Oshima
  ferries, no counts.
- 対馬市地域公共交通網形成計画 概要版 H27: bus data only (pp.1–7).
- 壱岐市地域公共交通網形成計画: the PDF and its landing page both return 404.
- Kyushu Shosen and Kyushu Yusen company pages: no passenger figures.
- Wikipedia 平戸市営フェリー: no figures.

**Fukuoka**
- Fukuoka city ferry pages (市営渡船, notices, Hakata port profile and guide):
  no figures. The Hakata port statistics page
  (https://www.city.fukuoka.lg.jp/kowan/shinko/shisei/001_2.html) has monthly
  and annual reports with 船舶乗降人員; whether the annual reports split by route
  is **not checked**.
- Wikipedia 福岡市営渡船, 宗像市営渡船, 小倉航路: no figures.
- Munakata city 大島航路 page (fares only); Munakata talk at the R5 public
  transport symposium (000317686.pdf, pp.1–10): urban planning, no ferry data.
- Kitakyushu 若戸航路 and 小倉航路 pages: no figures. The only numbers found are
  in a Merkmal news article.
- Shingu town Ainoshima route H26 sheet (000014353.pdf p.1): 7.5 km, over 60%
  of users from off the island, no count.

**Saga**
- Karatsu city route page: operators and timetable PDF only. Karatsu subsidy
  evaluation sheets: grades only.
- Saga prefecture yearbook and the Karatsu city plan/statistics were **not
  reached** (search allowance gone).

## Known gaps (sizeable routes with no route-level figure)

- Kyushu Yusen: Hakata–Iki–Tsushima (ferry + jetfoil Venus), Hakata–Hitakatsu,
  Karatsu–Indoshi. Only island totals.
- Kyushu Shosen Sasebo–Kamigoto and Nagasaki–Arikawa; Goto Sangyo Kisen
  Tainoura–Nagasaki. Only the Kamigoto island total and the Shin-Kamigoto town total.
- Goto routes after FY2020.
- Nomo Shosen Nagasaki–Iojima–Takashima; Sasebo-area routes (Kurokami, Takashima,
  Aino–Mikuriya, Uku–Terajima municipal, Sakito Shosen, Kawachi–Sasebo, Saikai
  Engan Shosen Sasebo–Kamiura); Ojika town routes; Hirado (Takushima, Oshima);
  Saikai (Seto–Matsushima).
- Omura Bay high-speed boats (Yasuda Sangyo Kisen); Shimabara–Kumamoto (Kyusho
  Ferry, Kumamoto Ferry), Kuchinotsu–Oniike, Taira–Nagasu (Ariake ferry),
  Shimabara–Omuta (Yamasa Kaiun).
- All 7 Karatsu island routes.
- Fukuoka city municipal ferries (Shikanoshima, Noko, Oronoshima; Genkaijima
  only approximate and old), Umi-no-Nakamichi–Hakata, Munakata Oshima and
  Jinoshima, Shingu–Ainoshima, Itoshima Himeshima, Kanmon Kisen.

## Unopened leads worth trying (all need no search)

- Nagasaki city `66737.xlsx` (monthly 乗降船客数 by operator at Nagasaki port) and
  the yearbook workbook `14666.xlsx` (表38): both in the CSV as needs_download.
- Hakata port annual statistics (link above): may split domestic passengers by route.
- Tsushima's 2015 full 網形成計画 (only the summary was read); Ojika and Hirado
  town plans (Nagasaki branch office lists Ojika R5.4–R10.3 and Hirado
  H31.4–R7.3); Iki city's newer plan, if it has one.
- 九州離島航路経営改善ガイド (Kyushu bureau, H24.3), which has per-route case data,
  pre-2015: https://www.city.munakata.lg.jp/kiji0034186/3_4186_50_30-1-6.pdf

## Definition notes

- Goto city figures: subsidised routes are counted October–September. Unclear
  whether the Nomo Shosen Fukue–Hakata figure is the whole route or Goto-city
  passengers only, and whether Kyushu Shosen Goto–Nagasaki includes
  Nagasaki–Narao (Shin-Kamigoto) passengers.
- Kyushu bureau island totals come from the MLIT per-route operator reports
  and are passengers to/from each island. The split for Iki–Tsushima through
  traffic is not stated.
- Tsushima municipal figure 474.5 has half units as published (likely
  half-fare counting). The same sheet shows 271.0 for FY2025, labelled as a
  forecast.
- Kitakyushu figures are from a news article quoting the city. The Wakato
  crossing is a 渡船.
- No passenger-km found anywhere in this region. Route km exists only for the
  main island routes (bureau table) and Genkaijima (18.5), Ainoshima–Shingu
  (7.5) and the Tsushima municipal route (23.5).
