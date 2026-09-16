# Chugoku — ferry passenger sources

Region: Hiroshima, Okayama, Yamaguchi, Shimane, Tottori. Bureau: 中国運輸局
(western Yamaguchi — Shimonoseki, Ube, Sanyo-Onoda, Nagato — is under 九州運輸局
下関海事事務所 instead).

## Headline

Per-route passenger figures exist in this region **only for subsidised island
routes** (離島航路運営費等補助) plus Miyajima. Every large commercial route —
Hiroshima–Matsuyama, Hiroshima/Kure–Etajima, Onomichi's crossings, Uno–Naoshima,
Kanmon, Oki — publishes nothing per route that I could find. The CSV has 20
routes/series with at least one year; 12 of them are in Yamaguchi.

## Best sources

1. **中国運輸局 地域公共交通確保維持改善事業 事業評価 (離島航路)** — one slide per
   subsidised route per year, each with 【利用者数】, a route map with leg
   distances, vessel count and island population. This is the backbone of the CSV.
   - FY2019: https://wwwtb.mlit.go.jp/chugoku/content/000231990.pdf (Hiroshima 7,
     Okayama 3, Yamaguchi 12 routes)
   - FY2021: https://wwwtb.mlit.go.jp/chugoku/content/000259840.pdf
   - FY2023: https://wwwtb.mlit.go.jp/chugoku/content/000320173.pdf
   - FY2024: https://wwwtb.mlit.go.jp/chugoku/content/000347059.pdf
   - FY2025: https://wwwtb.mlit.go.jp/chugoku/content/000370191.pdf
   Year pages: 令和5年度 https://wwwtb.mlit.go.jp/chugoku/00001_01872.html,
   令和6年度 .../00001_02262.html, 令和7年度 .../00001_02653.html (the R1 and R3
   PDFs were found by search, not from an index page).
   **Coverage varies by year**: the routes are evaluated in two alternating
   groups, so no single year has all of them. FY2019 is the only year with all 12
   Yamaguchi routes.
2. **廿日市市 宮島来島者数** — the only per-operator figures in the region.
   - Monthly totals 1964–2026, all operators combined:
     https://www.city.hatsukaichi.hiroshima.jp/uploaded/attachment/96310.pdf
   - Split by vessel (JR / 松大汽船 / 宇品-宮島 / その他船舶), CY2006–2022:
     https://www.city.hatsukaichi.hiroshima.jp/uploaded/attachment/63413.csv
     (open-data item I-11; the series stops at 2022, the PDF continues to 2025)
3. **九州運輸局下関海事事務所 業務概況 令和6年度版** —
   https://wwwtb.mlit.go.jp/kyushu/content/000358238.pdf p.3: 旅客輸送実績 FY2019–FY2023
   for the whole Shimonoseki jurisdiction (4 operators, 5 routes) plus the
   subsidised subtotal. Aggregate only, but it bounds the Kanmon crossing:
   FY2023 一般旅客定期 799,583, of which 37,518 on the two 下関市 island routes,
   so Kanmon + the Omijima sightseeing route ≈ 762,000.

## Roster / completeness check

**運輸要覧（海事振興部編）令和7年版**, https://wwwtb.mlit.go.jp/chugoku/content/000319176.pdf
(chapter split of the bureau's yearbook, indexed at
https://wwwtb.mlit.go.jp/chugoku/00001_01864.html):
- p.3, p.6 — **95 一般旅客定期航路** run by 65 operators in the bureau area at
  2025-04-01, by office: 本局 23, 尾道 24, 因島 5, 呉 7, 鳥取 1, 島根 8, 岡山 6,
  水島 6, 山口 15. Of these 22 take national and 16 local subsidy.
- p.7 輸送実績総括表, FY2024: 一般旅客定期 **18,151,911 passengers and
  131,489,290 passenger-km** across 94 reporting routes (plus 42,611 on 特定旅客定期
  and 625,847 on 旅客不定期).
So the CSV's per-route rows cover perhaps 2 million of an 18 million passenger
region — the subsidised islands are a small fraction of the traffic, and the
per-route breakdown behind that 18.2 million is not published.
There is **no 航路別 passenger table** in the 運輸要覧: its "(3) 航路別" table only
counts routes by attribute (car-carrying, island, mail, etc.).

## Definitions worth carrying into the data

- **Subsidy year, not fiscal year.** 離島航路 subsidy runs **1 October – 30
  September** (stated on the MLIT scheme slide and spelled out on the Okayama
  sheets, e.g. "R6.10～R7.9までの利用者は25,684人"). I put `OctSep` in `year_basis`
  and labelled `fiscal_year` with the year the period ends, so 2025 = Oct 2024–Sep 2025.
  This is neither FY nor CY and shouldn't be merged with April–March rail years
  without noting the six-month offset.
- **Half people.** Many counts end in .5 (e.g. 132,431.5). Children are counted as
  0.5 in these returns.
- **Miyajima counts arrivals.** 来島者数 is compiled from operators' boarding
  counts to the island, so it is one direction; both-direction ridership is
  roughly double. The 2019 total of 4,657,343 is the familiar Miyajima visitor
  number.
- The Hiroshima sheets use **運航回数 (sailings) as their target**, so they carry
  no passenger figure at all — see gaps.
- Route legs, not route length, are what the sheets print. I summed legs into
  `route_km` only where the chain was unambiguous, and put the legs in notes
  otherwise.

## What was checked, by prefecture

**Hiroshima**
- Subsidy evaluation sheets for all 7 subsidised routes (阿多田～小方 9.65km,
  走島～鞆 7.0km, 常石～尾道, 細島～西浜 2.7km, 白水～契島, 斎島～久比, 三角～久比
  1.25km): sailings only, no passengers, in every year checked (FY2019, FY2021,
  FY2023, FY2024, FY2025). Solid negative.
- 広島県統計年鑑 第70回 (2025): the only maritime passenger table is 表100
  船舶乗降人員 — by port, Excel. Port-level, so out of scope here.
- 広島市統計書 令和7年版, chapter I 運輸及び通信 (ir7.xlsx): tables I-18-1/I-18-2
  船舶乗降人員（広島港）. Excel, not downloaded; unverified whether it splits by
  route. Row added with needs_download.
- 呉市地域公共交通計画 (令和7年3月): 概要版 read in full — no per-route ferry
  ridership, only mode-level cost/recovery indicators. 本編 (27.9MB) and 資料編
  (58.2MB) are over the fetch tool's 10MB limit and were not opened; the 資料編 is
  the most likely place for route ridership in Kure.
  https://www.city.kure.lg.jp/soshiki/28/chikoukeikaku.html
- 呉市定期航路情報 page: a useful route roster (Kure–Oyou, Kure–Akitsu,
  Tennou–Kirikushi, Aga–Nasakejima, Konaga–Akashi, Onaga–Temizu–Takehara,
  Kuhi–Mikado, Kuhi–Itsuki, Kure–Miyajima) but no figures.
- 広島県離島振興計画 (令和5–14年度), 尾道市 and 江田島市 statistics pages, Onomichi
  港湾振興課 pages: no passenger figures.

**Okayama**
- All three subsidised routes (三洋汽船 ×2, 大生汽船) have figures for FY2019,
  FY2021, FY2023 and FY2025, and the sheets also print the previous year's figure,
  which fills FY2018, FY2020, FY2022 and FY2024. This is the most complete run in
  the region.
- 岡山県離島振興計画 (令和5–14年度),
  https://www.pref.okayama.jp/uploaded/life/508849_3831142_misc.pdf — its
  statistics appendix has population and tourist numbers but no route ridership.
- 岡山県統計年報: the prefecture's statistics index at /site/toukei/ returns 404
  and, with the web-search budget gone, I could not locate the current URL.
- Uno–Naoshima / Uno–Shodoshima / Shin-Okayama–Tonosho: nothing found. The
  operators (四国汽船, 両備, 小豆島豊島フェリー) are Kagawa-facing, so the Shikoku
  agent may have better luck via Kagawa sources.

**Yamaguchi**
- All 12 Chugoku-side subsidised routes have FY2019 figures; subsets have FY2021,
  FY2023, FY2024 and FY2025 (see coverage note above).
- 山口県統計年鑑 令和7年刊, table 096 船舶乗降人員 (xls): port-level.
- 山口県離島振興計画 全文 (155014.pdf): route tables give operators, vessels and
  sailings per island group, no ridership.
- 周防大島町地域公共交通計画 概要版: confirms the roster (周防大島松山フェリー
  柳井港–伊保田港–三津浜港 4 round trips/day, 町営渡船 to 前島・浮島・情島,
  行政連絡船 to 笠佐島) and says ferry use was edging up until FY2019 then fell, but
  gives no route numbers. Its target table only quantifies bus lines.
- 下関: the Kyushu bureau report above. Per-route figures for 関門汽船 下関～門司,
  下関市 竹崎～六連島 and 蓋井島～吉見 are not published there.

**Shimane**
- 隠岐広域連合「隠岐航路」page gives network totals only, rounded to 0.1万人, for
  FY1998, 2007, 2016, 2019 and 2020. Recorded, flagged: these came through the
  fetch tool's text extraction of an HTML table (two consistent passes), not from
  a document I could read directly.
- 島根県「離島航路について」and「隠岐航路について」pages: vessels and capacities
  only, no ridership.
- 島根県統計書 令和5年版, 運輸・通信 chapter is a single Excel workbook
  (https://pref.shimane-toukei.jp/upload/user/00027717-dTqyss.xlsx). Not
  downloaded; **this is the most promising untried source for Oki per-route
  figures** and for the Dozen inter-island boats.
- The Oki routes do not appear in the Chugoku bureau's island-route evaluations at
  all, which is why there is nothing slide-level for them.

**Tottori**
- Only one 一般旅客定期航路 in the whole prefecture per the 運輸要覧 (境港 is a call
  of the Oki route). Nothing separate to collect.

## Known gaps — sizeable routes with no figure

- **Miyajima after 2022 by operator** — only the combined total continues.
- **Hiroshima–Kure–Matsuyama** (瀬戸内海汽船 / 石崎汽船, cruise ferry + superjet):
  nothing published; neither operator posts traffic.
- **Hiroshima/Kure–Etajima** (Ujina–Koyo, Kure–Koyo, Tennou–Kirikushi,
  Ujina–Nishinomi), **Hiroshima–Ninoshima**: nothing.
- **Onomichi's crossings** (尾道渡船, 駅前渡船, Onomichi–Mukaishima ferry) — high
  frequency, tiny distance, no counts found; the city publishes timetables only.
- **Mihara/Takehara–Osakikamijima/Omishima/Ikuchijima**, **Fukuyama–Tomonoura–Sensuijima**: nothing.
- **Uno–Naoshima / Shodoshima routes, Hinase–Shodoshima**: nothing from the
  Okayama side.
- **Kanmon (関門汽船 下関–門司)**: only inside the Shimonoseki aggregate.
- **Oki per route** (mainland–Dogo, mainland–Dozen, Rainbow Jet, 島前内航船
  いそかぜ/フェリーどうぜん): only the rounded network total.
- **防予フェリー 柳井–三津浜**, **周防大島松山フェリー**, **周防灘フェリー 徳山–竹田津**:
  nothing.

## Dead ends and blocks

- **中国運輸局 運輸要覧 (full editions)** — every edition on
  https://wwwtb.mlit.go.jp/chugoku/txt/toukei.html is 11–16MB, over the fetch
  tool's 10MB cap, so only the chapter-split版 could be read. The maritime chapter
  is the one that matters and it has no per-route passenger table.
- **呉市地域公共交通計画 本編・資料編** — 27.9MB and 58.2MB, same cap.
- **Web search budget for the session ran out** (200/200) partway through, so the
  later work was limited to following links from pages already in hand. Municipal
  transport plans for Etajima, Onomichi, Kasaoka, Hagi and the Oki towns were
  never located as a result; those are the obvious next places to look for the
  Hiroshima and Shimane gaps.
- 中国運輸局's 離島航路 page (kaijou02.html) links only an island-events PDF, not a
  route roster. The bureau's pages are Shift-JIS and come back as mojibake through
  the fetch tool; asking it for raw href values works around that.
