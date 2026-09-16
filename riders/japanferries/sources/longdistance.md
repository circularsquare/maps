# Long- and medium-distance ferries: per-route passenger sources

Research done 2026-09-15. Every figure in `longdistance.csv` was read off the page
images of the source PDFs, not from a fetch summary.

**Search budget.** The shared WebSearch budget ran out partway through the session
(200 calls, shared with the regional agents). After that only direct fetches of
official pages and links found on them were possible. Checks cut short by this are
marked **(not checked: search budget)** below.

## What the CSV holds

Two regional transport bureau (運輸局) publications give almost all the coverage.

1. **Hokkaido Transport Bureau (Hokkaido Un'yu-kyoku), 「数字でみる北海道の運輸」,
   section 2 (旅客輸送), table 11(3) 航路別旅客輸送人員の推移.**
   https://wwwtb.mlit.go.jp/hokkaido/content/000368329.pdf, PDF p.18.
   - Covers every Hokkaido–Honshu ferry route, FY2019–FY2024, in thousands (千人).
   - Routes: Tomakomai–Hachinohe, Tomakomai–Sendai–Nagoya, Hakodate–Oma,
     Hakodate–Aomori (both operators in one column), Muroran–Aomori (from 2023-10),
     Otaru–Maizuru, Tomakomai–Tsuruga, Tomakomai–Oarai, and Muroran–Miyako
     (discontinued).
   - Shin Nihonkai Ferry is grouped. "Otaru–Maizuru" includes Otaru–Niigata and
     Otaru–Tsuruga. "Tomakomai–Tsuruga" includes Tomakomai–Akita, –Niigata and
     –Maizuru. The table doesn't say whether legs that never touch Hokkaido
     (Niigata–Akita, or Sendai–Nagoya on Taiheiyo Ferry) are counted.
   - FY2019 discrepancy: the route columns add up to 1,849 thousand, but the printed
     total is 1,811. Every other year's row adds up exactly. The same document's
     table 1(2) gives 1,853 thousand ship passengers between Hokkaido and elsewhere
     in FY2019, which supports the column values rather than the total. One FY2019
     cell is probably misprinted; I can't tell which.
   - The source doesn't say how children are counted.
   - Earlier editions (FY2015–FY2018) aren't linked from the current statistics
     pages. An old per-route PDF at
     `wwwtb.mlit.go.jp/hokkaido/bunyabetsu/kaiun/cruise/cruise6/2011/2011ferry.pdf`
     (title 「航路別旅客輸送人員の推移（単位：千人）」) now returns 404. It is
     probably in the Wayback Machine.

2. **Kyushu Transport Bureau (Kyushu Un'yu-kyoku), 「令和6年度 長距離フェリー航路の
   輸送実績資料」 and 「令和5年度 長距離フェリー航路輸送実績資料」.**
   - FY2024: https://wwwtb.mlit.go.jp/kyushu/content/000358745.pdf
   - FY2023: https://wwwtb.mlit.go.jp/kyushu/content/000333221.pdf
   - Page 1 has a table of passengers and vehicles by **corridor**, not by route:
     - (A) Kitakyushu–Hanshin: Hankyu Ferry ×2 and Meimon Taiyo
     - (B) Central Kyushu–Hanshin: Sunflower Beppu and Oita
     - (C) Southern Kyushu–Hanshin: Sunflower Shibushi and Miyazaki Car Ferry
     - (D) Kitakyushu–Keihin: Ocean Trans and Tokyo Kyushu Ferry
   - Each table also gives change against the previous year and against FY2019.
     Page 2 (図1) is a stacked bar chart per corridor, FY2008–FY2024, labelled in
     万人. The CSV takes FY2015–FY2022 from those labels (rounded to 10,000) and puts
     the more exact FY2019 and FY2022 values implied by the ratios in `notes`.
   - Children are counted as 0.5 passengers.
   - Pages 3–4 of the same PDFs are the bureau's 「主要離島航路輸送実績資料」, same
     layout. Group (E) Mainland–Amami/Okinawa covers A-Line (Maruee Ferry), Marix
     Line and Amami Kaiun's Kagoshima–Kikai–China route. That group is in the CSV.
     Groups (A)–(D) on those pages (Iki/Tsushima, Goto, Koshiki, Tanegashima/Yakushima)
     belong to the Kyushu regional agents; pointer only.
   - Page 2 of the FY2023 PDF also gives the four-corridor total to 0.1 万人,
     FY2013–FY2023: 149.9, 142.2, 160.4, 160.9, 163.2, 163.7, 157.1 (FY2019), 72.3,
     93.3, 142.2, 170.8. Handy as a check.
   - Only the FY2023 and FY2024 editions were found; the bureau's listing page
     (`/kyushu/toukei/kankou/file04.htm`) comes back garbled (Shift_JIS). Their
     file names follow `content/000NNNNNN.pdf`, so older editions probably exist.

### Coverage of the roster (24 routes)

| Route | Figure? | How |
|---|---|---|
| Shin Nihonkai: Otaru–Maizuru, Otaru–Niigata | yes, grouped | Hokkaido "Otaru–Maizuru" column |
| Shin Nihonkai: Tomakomai–Tsuruga, Tomakomai–Akita–Niigata–Tsuruga | yes, grouped | Hokkaido "Tomakomai–Tsuruga" column |
| Sunflower Tomakomai–Oarai | **yes, own route** | Hokkaido table |
| Sunflower Osaka–Beppu, Kobe–Oita | yes, grouped | Kyushu (B) |
| Sunflower Osaka–Shibushi | yes, grouped with Miyazaki | Kyushu (C) |
| Taiheiyo Ferry Nagoya–Sendai–Tomakomai | **yes, own route** (unclear whether the Sendai–Nagoya leg is included) | Hokkaido table |
| Silver Ferry Hachinohe–Tomakomai | **yes, own route** | Hokkaido table |
| Hankyu Ferry Izumiotsu / Kobe–Shinmoji, Meimon Taiyo Osaka–Shinmoji | yes, grouped (3 routes) | Kyushu (A) |
| Ocean Trans Tokyo–Tokushima–Shinmoji, Tokyo Kyushu Ferry Yokosuka–Shinmoji | yes, grouped | Kyushu (D); the Tokyo–Tokushima leg is probably not in it |
| Miyazaki Car Ferry Kobe–Miyazaki | yes, grouped with Shibushi | Kyushu (C) |
| A-Line / Marix Kagoshima–Amami–Okinawa | yes, grouped with Amami Kaiun | Kyushu island group (E) |
| Tsugaru Kaikyo / Seikan Aomori–Hakodate | **yes, own route** (two operators combined) | Hokkaido table |
| Tsugaru Kaikyo Oma–Hakodate | **yes, own route** | Hokkaido table |
| Jumbo Ferry Kobe–Takamatsu | no | |
| Orange Ferry Osaka–Toyo, Kobe–Niihama | no | |
| Matsuyama Kokura Ferry | no | |
| Nankai Ferry Wakayama–Tokushima | no | |

Also in the CSV, outside the roster: Tsugaru Kaikyo Ferry Muroran–Aomori (FY2023–24)
and Kawasaki Kinkai Kisen Muroran–Miyako (FY2019–21, discontinued).

## Part 1: per operator, findings and dead ends

- **Japan Long Course Ferry Service Association (Nihon Chokyori Ferry Kyokai, JLC).**
  - Hearing deck https://www.mlit.go.jp/seisakutokatsu/content/001625058.pdf (10
    pages, all read): no passenger counts. It has the route map (9 companies, 15
    routes, 37 ships, June 2023) and trucks carried, about 1.27M in FY2022 on 37
    ships (p.4).
  - Website jlc-ferry.jp has no statistics page.
  - The association does compile corridor-level figures for all 12 MLIT
    "long-distance routes". The Japan Maritime Daily (Nihon Kaiji Shimbun) article
    「長距離フェリー23年度実績、全12航路で旅客2割増」
    (https://www.jmd.co.jp/article.php?no=295673) names corridors such as
    Chukyo–East Tohoku, Hanshin–Central Kyushu and East Shikoku–Kitakyushu. The
    article is paywalled; I recorded no figures from it. So MLIT's "12 routes" are
    most likely these JLC corridors, not single routes.
- **Japan Passengerboat Association (Nihon Ryokakusen Kyokai) hearing deck**
  https://www.mlit.go.jp/policy/shingikai/content/001634142.pdf, pp.1–10 read:
  - No per-route passengers.
  - p.2 lists the medium- and long-distance ferries: 15 companies, 19 routes (April
    2023). This confirms the roster, including Kawasaki Kinkai Kisen, Jumbo, Shikoku
    Kaihatsu (Orange), Matsuyama Kokura, Maruee and Marix.
  - p.3: 12 routes by 10 companies between Shikoku and the rest of Japan carried
    3.24M in FY2018, against 7.81M by rail across the Honshu–Shikoku bridges. One
    aggregate that mixes short routes in, not recorded.
  - p.4: Orange Ferry's Osaka route has capacity for about 1,000 passengers a day.
    Capacity, not ridership.
  - Pages 11 onward (COVID impact and so on) not read.
- **MLIT modal-shift load-factor release** (kaiji03_hh_000228). The fetch summary
  says it covers truck/cargo load factors for ferries, RORO and container ships, not
  passengers. The attachment URL it gave (`/maritime/content/002019301.pdf`)
  returned 404 and may have been invented by the summariser. **Not verified.**
- **Shin Nihonkai Ferry (SHK Line group).** Searches for 旅客数 by route and year
  and for Maizuru and Tsuruga port passenger trends found nothing. Figures come only
  from the Hokkaido table. Fukui prefecture's Tsuruga page covers cargo only
  (`pref.fukui.lg.jp/doc/kouwan/tsuruga/tsuruga05kamotu.html`, not opened).
- **Shosen Mitsui Sunflower.**
  - Searches on the Oarai route and Oarai port turned up nothing beyond timetables.
    Ibaraki prefecture's Oarai port pages have no passenger statistics in snippets.
  - Operator history: Oarai belonged to 商船三井フェリー and the Kansai routes to
    フェリーさんふらわあ; the two merged as 商船三井さんふらわあ in 2023.
  - Group IR (MOL) not checked.
- **Taiheiyo Ferry (Meitetsu group).** Searches found nothing. The Hokkaido table is
  the only source. Sendai and Nagoya port statistics **not checked: search budget.**
- **Kawasaki Kinkai Kisen / Silver Ferry.** Hokkaido table. Hachinohe port **not
  checked: search budget.**
- **Hankyu Ferry, Meimon Taiyo Ferry.** Searches found nothing per company. Kyushu
  (A) only. Kitakyushu port statistics **not checked: search budget.**
- **Ocean Trans, Tokyo Kyushu Ferry.** Kyushu (D) only.
- **Miyazaki Car Ferry.** The Miyazaki Nichinichi Shimbun reports its yearly
  results: FY2024 results (Yahoo copy, now 404), and FY2025 (article 2026-07-23,
  https://www.the-miyanichi.co.jp/today/topic/938906.html). Both give percentage
  changes but no absolute passenger count. The FY2025 piece says passengers rose
  11.4% year on year to a record. That was read via the fetch summary only; not
  recorded. Kyushu (C) is the only count.
- **A-Line (Maruee Ferry), Marix Line.** Searches found nothing. Kyushu island group
  (E) only.
- **Tsugaru Kaikyo Ferry, Seikan Ferry.** Searches found nothing per company. The
  Hokkaido table gives Hakodate–Aomori for both operators combined, plus Oma and
  Muroran separately. Aomori prefecture, Oma town and Hakodate port **not checked:
  search budget.**
- **Tomakomai port yearbook** (苫小牧港統計年報, jptmk.com). The statistics page
  lists a 2024 yearbook (15.8 MB), but the file URL from the summary returned 404.
  Search snippets say it has a monthly ferry usage table (フェリー利用状況月別表).
  Could give calendar-year per-route counts at Tomakomai; worth one more try from
  https://www.jptmk.com/020shisetsu/026toukei/index.html.
- **Jumbo Ferry, Orange Ferry, Nankai Ferry, Matsuyama Kokura Ferry.** No figure
  found. The Shikoku Transport Bureau's statistics and scheduled-route pages
  (`/shikoku/soshiki/toukei.html`, `/shikoku/soshiki/kaijyou/teiki.html`) came back
  empty or garbled. Operator news, Kobe and Osaka port statistics, and the Tokushima
  and Wakayama prefectures **not checked: search budget.**
- **Muroran Transport Branch report** 「令和4年度 旅客フェリー活性化事業 事業報告書」
  (https://wwwtb.mlit.go.jp/hokkaido/muroran/top/press/20230419.pdf). Downloaded,
  **not opened**. Probably about Muroran–Aomori.
- The Hokkaido Transport Bureau monthly 「北海道の運輸の動き」 has a ferry table, but
  it counts vehicles by strait vs medium/long-distance, not passengers by route
  (`/hokkaido/content/000380125.pdf`, p.13).

## Part 2: national sources

**a. MLIT headquarters island-route subsidy materials.** No national per-route list
with passengers found.
- Maritime Report 2025 (海事レポート2025), ch.2
  (https://www.mlit.go.jp/maritime/content/001912002.pdf, p.24): 272 island routes,
  127 subsidised (April 2025), domestic passenger-ship travellers 73.7M in FY2023,
  and one chart (図表2-5) of aggregate 輸送人員 and 欠損額 for subsidised routes,
  FY2004–FY2023. National totals only.
- Regional bureaus publish route lists without passengers. Example: Shikoku bureau
  「四国運輸局管内国庫補助航路一覧」 (https://wwwtb.mlit.go.jp/shikoku/content/000291368.pdf):
  map plus prefecture / route / operator for 22 routes, no passengers.
- The Kyushu bureau's 主要離島航路輸送実績資料 gives passengers by island group, not
  by route.
- Okuno's Kobe doctoral thesis (2013, https://hdl.handle.net/20.500.14094/D1005797,
  pp.17–21 read) has MLIT counts of subsidised operators and routes, and deficit and
  subsidy totals, 1952–2006. No per-route passengers.
- Hase Tomoharu's 「離島航路を巡る環境変化と政策」 (YMF, pp.45–50 read) is policy
  history with no route data.
- SPF Ocean Newsletter no.602 (Yukihira, 2026-02-20) only cites Maritime Report 2025
  totals.
- 事業評価 summaries under 地域公共交通確保維持改善事業: **not located (search
  budget).**

**b. Trunk ferry passenger flow survey on e-Stat** (幹線旅客流動実態調査, toukei
00600462).
- Two survey groups, labelled on e-Stat 「平成22年 幹線フェリー・旅客船旅客流動実態報告」
  (released 2014-05-02) and 「平成27年 …」 (released 2018-05-29).
- Each has exactly **one** file, titled 「報告書」: Excel, statInfId 000024699655
  (2010) and 000031703995 (2015). The table titles are inside the workbook, not in
  the listing, so I couldn't list them without downloading.
- e-Stat's description of the results: travellers by trip purpose, and prefecture
  flow tables between origin and destination and between boarding and landing
  ports, each for weekdays and holidays. That is **prefecture-level port pairs, not
  per route**. It is a survey-day sample feeding the national trunk passenger flow
  survey (全国幹線旅客純流動調査).
- The MLIT page http://www.mlit.go.jp/statistics/kansenferi.html came back
  garbled.
- To list the sheets: download `file-download?statInfId=000031703995&fileKind=0`.

**c. Regional passenger flow survey methodology** (旅客地域流動調査; e-Stat 00600460,
MLIT page `k-toukei/kamoturyokakutiikiryuudoutyousa_toukeinosakuseihouhou.html`).
Read via fetch summary, so wording not verified. Passenger-ship flows are built three
ways:
- Routes whose two ends are in different prefectures, with no intermediate calls:
  from the per-route passenger counts in 「内航旅客航路事業運航実績報告書」 (MLIT
  Maritime Bureau), assigned by the prefectures of the end ports.
- Routes with intermediate ports: from a separate 「旅客船旅客県間流動調査」.
- Other routes: into within-prefecture flows.
- Car ferries use the same method.
- No per-route appendix is mentioned.

So a table-2 cell holding one direct two-prefecture route is that route's reported
count. Multi-stop long-distance routes (Taiheiyo, Shin Nihonkai via Akita/Niigata,
Ocean Trans via Tokushima) are split by a survey and can't be read back as routes.

**d. Remote island statistics yearbook** (離島統計年報, Nihon Ritō Center).
- https://www.nijinet.or.jp/publishing/statistics/tabid/549/Default.aspx
- 2024 edition out. CD-ROM only since 2003, ¥7,260.
- 18 sections including (12) 港湾・航路現況 and (13) 空港・航空路現況.
- The page gives no field list and no sample page, so **whether it has per-route
  passengers is unconfirmed**. No table of contents was found elsewhere (search
  budget). Paid; not pursued.

**e. Japan Passengerboat Association and 「数字で見る海事」.**
- The association is at jships.or.jp (jpta.or.jp is a tennis association). Its
  statistics page (https://www.jships.or.jp/statistics.html) shows national MLIT
  aggregates only: operators, routes, ships, and FY2023 passengers and
  passenger-km, per the fetch summary.
- The 「旅客船」 magazine isn't linked from the site.
- 「数字で見る海事2025」 is on https://www.mlit.go.jp/maritime/maritime_fr1_000098.html.
  Its chapter 1 PDF (`/maritime/content/001912016.pdf`) is over 10 MB and couldn't
  be fetched. Earlier editions are national-level tables. Not useful for routes.

**f. Academic compilations.**
- CiNii Research OpenSearch for 「航路別 輸送人員」: 0 results.
- J-STAGE and other query wordings: **not checked (search budget).**
- The only thesis found (Okuno, Kobe 2013) has no per-route national table.

**g. Freedom-of-information release of the per-route operator reports.** **Not
checked.** The search budget was gone by then.

**Most useful national-level finding:** nothing national is per route. The practical
substitute is the regional bureau yearbooks. Hokkaido publishes a true per-route table
for every Hokkaido ferry, and Kyushu publishes corridor tables for every long-distance
route touching Kyushu. The other bureaus (Tohoku, Kanto, Hokuriku-Shinetsu, Chubu,
Kinki, Shikoku, Chugoku) may publish similar tables in their own 「数字でみる…の運輸」
yearbooks; their index pages came back without usable links.

## Known gaps

- No figures for Jumbo Ferry, Orange Ferry (both routes), Nankai Ferry or Matsuyama
  Kokura Ferry. The Shikoku and Kinki bureau yearbooks are the next place to look.
- Kyushu corridors (A), (B), (C), (D) and island group (E) need splitting into
  routes. Possible sources:
  - port statistics for single-route ports: Miyazaki (Miyazaki Car Ferry only),
    Shibushi, Beppu, Oita, Yokosuka (Tokyo Kyushu only), Izumiotsu (Hankyu only);
  - Shin Nihonkai and Hankyu figures in SHK Line / Kanpu group materials.
- The Shin Nihonkai groups need splitting: Maizuru port covers Otaru–Maizuru only,
  and Tsuruga port covers the Tomakomai routes only.
- Hokkaido table FY2015–FY2018 (older editions, Wayback), and the FY2019 row
  discrepancy.
- `route_km` and `passenger_km` are blank everywhere; none of these sources give
  them.
- All figures are fiscal year. Kyushu counts a child as half a passenger; Hokkaido
  doesn't say.
