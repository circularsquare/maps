# Kanto ferry passenger sources

Researched 2026-09-15. Nearly all usable figures are for the Tokyo islands. Mainland Kanto routes (Tokyo Bay Ferry, water buses, Seabass, Sarushima, Hakone) have no transcribed figure. The session's web-search allowance (200 calls) ran out partway through, so the mainland prefectures got fewer searches than planned. What was and wasn't checked is listed below.

## Best sources

1. **東京港港勢（概報）, table 7 外内航客船・船客乗降人員表** (Tokyo Metropolitan Port Bureau)
   - Passengers boarding + landing at Tokyo port, split by the island port they sailed to or from: 大島(元町), 大島(岡田), 利島, 新島, 新島諸港, 式根島 / 野伏, 神津島, 神津島諸港, 三宅島(三池), 三宅島諸港, 御蔵島, 八丈島(八重根), 八丈島(神湊), 小笠原(二見), plus 館山 in some years.
   - Calendar years 2016–2024, one PDF per year, all linked from https://www.kouwan.metro.tokyo.lg.jp/yakuwari/toukei (p.15; p.13 in the 2024 edition). The 2015 edition isn't online.
   - This gives the Tokyo end of every Tokai Kisen Izu route and Ogasawara-maru, by island.
   - Check: each year the island rows add up exactly to the yearbook's 伊豆諸島 / 大島 / 三宅島 / 八丈島 groups (table 4-20), and the 2019 table total matches (1,790,573).
2. **東京都統計年鑑, 港別乗降船人員（島しょ港湾）** (table 4-22; 4-21 in the 2024 edition)
   - Boarding + landing at every island port, all routes combined, CY2015–2024. Read from the yearbook CSVs (one edition per year, `tnenkan/20xx/tn..qv042200.csv`).
   - Aogashima and Hahajima (沖港) each have only one route, so their port counts are route counts for Hachijojima–Aogashima and Chichijima–Hahajima.
   - The yearbook's 東京港 航路別 table (4-20) has the same Tokyo-port data grouped coarser. Only 2015 is recorded from it, since it isn't in the port reports.
3. **東京都離島航路地域協議会, 地域公共交通確保維持改善事業・事業評価** (Port Bureau page https://www.kouwan.metro.tokyo.lg.jp/jigyo/announcement/chiikikotsujigyohyouka.html)
   - 旅客輸送人員 for each subsidised route: 東京～八丈島 (東海汽船), 神津島～下田 (神新汽船), 八丈島～青ヶ島 and 父島～母島 (伊豆諸島開発), and 東京～父島 (小笠原海運, 令和4年度 sheet only).
   - Read the 令和7年度 sheet (dated 2026-01-30) and the 令和4年度 sheet (2023-01-31).
   - The Tokyo–Hachijo text also gives the year-on-year change by section (Tokyo–Miyake / Mikura / Hachijo). From that, the prior year's total is derived and flagged DERIVED.
   - The page only links the latest sheet. The 令和4年度 one turned up through search. A guessed URL for 令和6年度 returned 404, and WebFetch can't reach the Wayback Machine.
4. **東海汽船 決算短信** (FY Dec)
   - Company-wide passenger totals only: 2018, 2019, 2022–2025.
   - Recorded as COMPANY TOTAL rows. What they cover changes from year to year (see the definitions section).

## Roster

No bureau list of scheduled routes was found. The Kanto bureau's second-stage evaluation PDF (wwwtb.mlit.go.jp/kanto/content/000322393.pdf) covers bus routes only.

The Tokyo Port Bureau's Izu port plan (伊豆諸島港湾整備計画 第2章, p.6) has a table of Izu services by operator and ship. It works as a roster for the Izu islands: 東海汽船 jet and large ships from 竹芝 / 熱海, 神新汽船 あぜりあ from 下田, 新島村 連絡船にしき (Niijima–Shikinejima), 伊豆諸島開発 あおがしま丸.

## Checked, by prefecture

**Tokyo**
- Found: the yearbook tables, the port reports, the evaluation sheets and the Tokai Kisen earnings reports (all above).
- 大島支庁 管内概要 令和4年版: over the 10 MB fetch limit, not read. The 三宅, 八丈 and 小笠原 branch offices' editions weren't found (no searches left).
- 東京宝島推進委員会「伊豆諸島・小笠原諸島の概況」: population and access map only.
- 伊豆諸島港湾整備計画 第2章 (pp.3–16): service reliability and ship details, no ridership.
- 伊豆諸島・小笠原諸島観光客入込実態調査: visitors by ship vs air, per island. It's only published as a zip (row with download needed).
- Ogasawara-channel fan site: per-sailing counts, no annual totals.
- Tokyo Cruise (東京都観光汽船) and Tokyo Mizube Line (東京都公園協会): nothing published turned up in two searches.

**Kanagawa**
- 横須賀市統計書 令和5年度版, table 79 定航路乗降人員・車輌台数 (2018–2022, almost certainly Tokyo Bay Ferry at Kurihama): the CSV came back as garbage through WebFetch and the chapter is Excel only (row with download needed).
- 横浜市統計書 ch.8 港湾: no passenger tables.
- Yokosuka city Kurihama port page: no figures.
- Odakyu fact book (Mar 2026): no Hakone sightseeing-boat ridership, only Hakone Free Pass sales.
- Wikipedia (citing Asahi, 2018-11-10) gives Tokyo Bay Ferry at about 2.8M (FY1994), 1.94M (FY1998) and 0.98M (FY2009). All pre-2015, not recorded.
- Sarushima: a Town News article (2025-07-04) reportedly gives about 230k island visitors in FY2024. Only seen through a WebFetch quote, so it's noted but not recorded.

**Chiba**
- 千葉県港湾統計年報 has a table of monthly ferry passengers at prefecture-managed local ports (probably 金谷港 = Tokyo Bay Ferry). Excel only; rows for 2024 and 2019 with download needed. Editions 2015–2024 are online.
- 富津市地域公共交通計画 概要版 and the March 2024 council paper: no ferry ridership. The full plan (21.6 MB) is too large to fetch.

**Ibaraki**
- The prefecture's list of scheduled routes shows only freight RORO and the Oarai–Tomakomai ferry, which another agent covers. No scheduled passenger boats. Kasumigaura has no scheduled service.

**Tochigi, Gunma, Yamanashi**
- Not searched (search allowance used up). As far as known, only sightseeing boats on lakes such as Chuzenji, Kawaguchi and Motosu, which are out of scope.

**Kanto Transport Bureau**
- 000322393.pdf: buses only.
- The 地域公共交通確保維持改善 page links an FY2018 evaluation PDF (koutuu_seisaku/kakuhoiji/30hyouka.pdf) that is over 10 MB and wasn't read.

## Known gaps

- **Oshima jetfoils from Atami, Ito, Inatori, Kurihama and Tateyama.** No per-route figure. The island-port count minus the Tokyo-end count gives a rough non-Tokyo remainder (Oshima 2019: 473,961 − 335,783), but the two tables differ in coverage.
- **Legs between the islands.** Tokyo–Oshima–Toshima–Niijima–Shikinejima–Kozushima and Tokyo–Miyake–Mikura–Hachijo. Leg loads can be built from the Tokyo-end counts by destination: a leg carries every passenger bound beyond it. This ignores island-to-island travel.
- **Niijima–Shikinejima connecting boat (新島村 にしき).** No figure.
- **Tokyo Bay Ferry (Kurihama–Kanaya).** The largest mainland route; needs the Chiba or Yokosuka Excel.
- **Yokohama Seabass, Tokyo water buses, Sarushima, Enoshima boat, Lake Ashi boats.** No figures found.
- **Route lengths and passenger-km.** No source gives them.

## Definitions and oddities

- **Port tables are calendar years and count boarding + landing.** A Tokyo-port row is "Tokyo ↔ that island port" for passengers who boarded or landed at Tokyo. Large ship and jetfoil are combined.
  - 諸港 = the island's other or alternate ports (e.g. Miyake's Sabigahama / Igaya, Kozushima's Takou-wan).
  - The same table has 湾内周遊 (Tokyo Bay round-trip cruises: 1.04M in 2019, 0.58M in 2024) and long-distance ferry rows (北九州, 徳島小松島). Both are left out; they are the yearbook's "その他" line.
- **The island-port table undercounts some islands.**
  - Alternate ports aren't listed. Its Kozushima and Niijima figures are below the Tokyo-end counts in some years (2019 Kozushima: 41,487 at the island vs 46,387 at Tokyo).
  - Before 2020 there is no Shikinejima figure: 式根島港 shows "-" and 野伏港 is only listed from 2020. The Tokyo-end table does have Shikinejima in every year.
  - For the Tokyo legs, prefer the Tokyo-port table.
- **Hachijo is lopsided.** At both 神湊 and Tokyo, about twice as many sail Tokyo→Hachijo as return by sea, presumably because many fly back.
- **Evaluation sheets use the subsidy year (assumed Oct–Sep)**, recorded as year_basis `Oct-Sep` with fiscal_year = the year it ends. Figures ending in .5 suggest children are counted as half.
- **Tokai Kisen totals change wording:**
  - 2018–19: 全航路の旅客数, including the Tokyo Bay summer cruise (納涼船).
  - 2022–23: 両航路 (Izu + Ogasawara via the subsidiary 小笠原海運).
  - 2024–25: 伊豆諸島航路 旅客部門 乗船客数, still including the cruise.
  - None of these is a route figure.
