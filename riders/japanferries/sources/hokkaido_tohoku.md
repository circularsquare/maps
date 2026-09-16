# Hokkaido + Tohoku — ferry passenger sources

Region: 北海道, 青森, 岩手, 宮城, 秋田, 山形, 福島. Researched 2026-09-15.
Figures in `hokkaido_tohoku.csv` were transcribed from the rendered PDF pages,
not from fetch summaries, and checked against total rows where the table has one.

## Best sources

| Source | Coverage | Years | Notes |
|---|---|---|---|
| 北海道「2025 北海道の交通の状況」2(3)(2)③ 航路別旅客輸送人員の推移, p.24 — [PDF](https://www.pref.hokkaido.lg.jp/fs/1/1/4/8/9/0/5/7/_/II%20%E8%BC%B8%E9%80%81%E3%81%AE%E5%8B%95%E5%90%91%E7%AD%89(3%20%E5%9B%BD%E5%86%85%E4%BA%A4%E9%80%9A1).pdf) | all 9 Hokkaido–Honshu ferry routes: 苫小牧～八戸, 苫小牧～仙台～名古屋, 函館～大間, 函館～青森, 小樽～舞鶴, 苫小牧～敦賀, 苫小牧～大洗, 室蘭～宮古, 室蘭～青森 | FY2010–FY2023 (CSV keeps FY2015+) | 千人; the column sums match the 計 row every year. Cites 北海道運輸局「数字でみる北海道の運輸」. The newer bureau edition (with FY2024) probably has the same table, but I did not find which of its section files holds it. |
| 北海道離島振興計画（令和5年4月）p.8 ○航路別状況 — [PDF](https://www.mlit.go.jp/kokudoseisaku/chirit/content/001619191.pdf) | every Hokkaido island leg: 稚内–鴛泊, 稚内–香深, 鴛泊–香深, 沓形–香深, 江差–奥尻, 瀬棚–奥尻, 羽幌–焼尻, 羽幌–天売, 焼尻–天売, with island-resident share | FY2019 only (chosen as the pre-COVID year) | Only per-leg island table found. Haboro legs sum to the bureau's Oct–Sep 年計, so at least Haboro is on an operator year. |
| 函館市統計書「航路別フェリー運輸状況」— [2020 ed. ch. M](https://www.city.hakodate.hokkaido.jp/docs/2021043000077/file_contents/2021043000077_hk_docs_2021043000077_files_m.pdf) (table 80), [2025 ed. ch. M](https://www.city.hakodate.hokkaido.jp/docs/2026032600054/file_contents/M.pdf) (table 79) | 函館～青森, 函館～大間 乗降人員, 出/入 split | CY2015–CY2024 | Also trucks, vehicles, tonnage. Definition is narrower than the bureau series (see oddities). Each yearly edition covers 5 years; the 2021–2024 editions are at /docs/2022051300043/, /2023032900061/, /2024032100017/, /2025032500012/. |
| 第3期 塩竈市交通事業会計経営健全化計画（令和7年1月）— [PDF](https://urato-island.jp/wp-content/uploads/2025/04/%E7%AC%AC3%E6%9C%9F%E5%A1%A9%E7%AB%88%E5%B8%82%E4%BA%A4%E9%80%9A%E4%BA%8B%E6%A5%AD%E7%B5%8C%E5%96%B6%E5%81%A5%E5%85%A8%E5%8C%96%E8%A8%88%E7%94%BB.pdf) | 塩竈市営汽船 塩竈～浦戸諸島 | FY2014–FY2023 | p.12 all riders; p.7 has a second series excluding season tickets (FY2008–2023), with tourists split out, and p.13 has per-sailing averages. |
| 酒田市船舶運航事業経営戦略（令和3年3月）— [PDF](https://www.city.sakata.lg.jp/sangyo/kotsu/teikisen/senpak_keieisenryak.files/keieisenryaku.pdf) | 酒田～飛島 定期船とびしま | FY2011–FY2020 | Children count 0.5. FY2020 is to end of Jan 2021 only. |

Smaller ones: the bureau's 羽幌航路 subsidy summary [000238003.pdf](https://wwwtb.mlit.go.jp/hokkaido/content/000238003.pdf) (monthly bars plus 年計, H30/R1); the bureau's 羽幌航路 study [000318906.pdf](https://wwwtb.mlit.go.jp/hokkaido/content/000318906.pdf) fig. 5 (2000–2021, rounded to thousands); 宮城県離島振興計画 [001619192.pdf](https://www.mlit.go.jp/kokudoseisaku/chirit/content/001619192.pdf) (網地島 FY2021 約57千人, Shiogama FY2017–21, 渡船 FY2020); 石巻市 third-sector sheet for 網地島ライン [12_1-9.pdf](https://www.city.ishinomaki.lg.jp/cont/10162000/600/0602/12_1-9.pdf) (FY2020–23, rounded to thousands).

Route lengths come from 東北運輸局「図で見る東北の運輸」令和6年版 ferry map [000346698.pdf](https://wwwtb.mlit.go.jp/tohoku/content/000346698.pdf) (R6.12). The map gives one-way km and sailings for 函館–青森 113, 函館–大間 40, 蟹田–脇野沢 21.3, 八戸–苫小牧 242, 青森–室蘭 204, 宮古–室蘭 353.9 (suspended), 苫小牧–仙台 560, 仙台–名古屋 770, 秋田–苫小牧 413, 秋田–新潟 224 and 網地島航路 41. The Haboro and Shiogama km come from their own documents.

No source here gives passenger-km, so every `passenger_km` cell is empty.

## Rosters

- 北海道「2025 北海道の交通の状況」p.22 ①定期フェリー航路一覧 (R7.3.31) and p.30 ①離島航路一覧 (R7.2) — same PDF family, [page](https://www.pref.hokkaido.lg.jp/ss/stk/sitetop1/191828.html). The island list names the 江差～奥尻 operator as オクシリアイランドフェリー(株); 瀬棚～奥尻 has been suspended since April 2019.
- 東北運輸局 [東北の離島航路](https://wwwtb.mlit.go.jp/tohoku/kj/kj-sub15.html) (シーパル女川汽船, 網地島ライン, 塩竈市営汽船, 酒田市) and [東北のフェリー航路](https://wwwtb.mlit.go.jp/tohoku/kj/kj-sub16.html) (シルバーフェリー, 太平洋フェリー, 津軽海峡フェリー, 青函フェリー, 新日本海フェリー).
- 宮城県 [離島航路](https://www.pref.miyagi.jp/soshiki/is-kouwan/ferry.html) (石巻港: 網地島ライン, シードリーム金華山汽船, 潮プランニング).
- Hokkaido island routes combined (same PDF p.30, 千人): FY2016 616, FY2017 627, FY2018 583, FY2019 565, FY2020 253, FY2021 260, FY2022 388, FY2023 445. The FY2019 per-leg table sums to 511k, about 54k less than this 565k. I could not explain the gap.

## Checked, per prefecture

**北海道** — Found: the island plan (FY2019 per leg), the prefecture's transport-status book (Hokkaido–Honshu routes), the bureau's Haboro sheets, the Hakodate yearbook. No per-route figures in: 数字でみる北海道の運輸 令和6年版 section 2 (passengers; ship totals only) and section 3 (freight); 奥尻町 過疎地域持続的発展計画 R3–7 (ferry described, no numbers); 利尻町 過疎計画 R3–7 (same); 利尻富士町 観光統計 H31 (tourists only); 羽幌町 商工観光概要 (index lists only up to the FY2016 edition, and that page now 404s). 礼文町 and 稚内市 statistics turned up nothing in search. The bureau's 利礼航路 and 奥尻航路 subsidy summaries (like the Haboro one) probably exist but I did not locate them; 000291500.pdf, cited as the Haboro one, now 404s.

**青森県** — 青森県統計年鑑 第13章 運輸 (2021 edition via open data): no ferry passenger table, only port cargo including フェリー tonnage. 青函 and 大間 are covered by the Hokkaido sources. むつ湾フェリー (蟹田–脇野沢, seasonal) and the former シィライン (青森–脇野沢, ended 2023-03-31 per Wikipedia): searched, no figures found.

**岩手県** — No island routes. 室蘭～宮古 (川崎近海汽船, FY2018–21 then 0) is in the Hokkaido table. 東北運輸局 ferry map shows nothing else.

**宮城県** — Found: Shiogama plan, 宮城県離島振興計画, 石巻市 third-sector sheets (the 2018 sheet has no numbers, the 2024 sheet has rounded FY2020–23). Not found: シーパル女川汽船 (女川–出島–江島; MLIT case-study PDFs 015/016_onagawa describe the route, no numbers); 金華山 boats (シードリーム金華山汽船, 潮プランニング, 金華山観光クルーズ; reservation-based, no figures); 気仙沼–大島 (大島汽船, closed April 2019 when the bridge opened; not pursued).

**秋田県** — 苫小牧–秋田–新潟–敦賀 (新日本海フェリー) calls at Akita. It has no separate row in the Hokkaido table, and I found no Akita-specific figure.

**山形県** — Found: Sakata 経営戦略 (to FY2020). Not found: FY2021+; the city ferry pages hold no ridership, and the search for 酒田市統計書 returned nothing useful.

**福島県** — No scheduled passenger routes on the 東北運輸局 rosters or ferry map.

## Known gaps

- **Rishiri/Rebun and Okushiri**: one year only (FY2019), nothing recent. These are the region's biggest island routes.
- **Haboro**: per-leg FY2019 only; whole route to 2021 rounded.
- **FY2024**: only the Hakodate CY2024 pair. The Hokkaido–Honshu table ends FY2023.
- **女川–出島–江島**: no figure. A search snippet said calls at 出島 ended 2025-03-30 when the bridge opened; not verified.
- **Sakata–Tobishima**: nothing after FY2020.
- **網地島ライン**: rounded to thousands; nothing before FY2020 except FY2021 in the island plan.
- **むつ湾フェリー**: no figure.
- The WebSearch budget for the session ran out (200 calls) before I could chase the Okushiri/Rishiri subsidy sheets and Sakata's recent years.

## Definitional oddities

- **Hakodate yearbook vs bureau, Tsugaru Strait**: the city's 乗降人員 is about 45% of the bureau figure on 函館～青森 and about 80% on 函館～大間 (CY2019 352k vs FY2019 638k; 98k vs 116k). Probably a narrower count, e.g. excluding drivers riding with vehicles, or one terminal. Use the bureau series for thickness; the city series gives CY2024.
- **函館～青森 has two operators** (津軽海峡フェリー, 青函フェリー). Both sources give one combined figure, so it is one row, not one per operator.
- **Haboro year**: the bureau sheet's 年計 runs October–September (the operator's year), and the island plan's "R1" legs add up to it. Recorded as FY with a note.
- **苫小牧～仙台～名古屋** is one figure for the whole route. It is unclear whether Sendai–Nagoya-only riders are inside it.
- **Half passengers**: Sakata counts children as 0.5 and the Haboro 焼尻–天売 leg has a .5. These are kept as published.
- **Shiogama**: the Miyagi plan's series (e.g. FY2019 101,538) is not all riders; the city plan's total for the same year is 158,808. The city's own "定期券を除く" series (FY2019 101,659) is close to the prefecture's but not identical. The CSV has both, with notes.
- Sightseeing: Sakata and Shiogama include tourists on the scheduled boat (the Shiogama plan splits them out). Pure 遊覧船 (松島, 気仙沼湾, 浄土ヶ浜, 仏ヶ浦, lakes) were skipped.
