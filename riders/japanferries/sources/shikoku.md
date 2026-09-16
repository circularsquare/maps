# Shikoku — ferry passenger sources

Region: Kagawa, Ehime, Tokushima, Kochi (四国運輸局). Researched 2026-09-15.

**Coverage is thin.** Three individual routes have per-route figures:

- Nankai Ferry, Wakayama–Tokushima, FY2019–2024
- Ogi–Takamatsu (雌雄島海運), FY2015–2025
- Sukumo municipal Okinoshima route, subsidy years 2017–2021

On top of that, five route-group totals for FY2015–16 come from a 四国運輸局 report. No passenger-km figures were found anywhere.

**The search budget ran out partway through.** The session-wide WebSearch cap (200 calls, shared with the other regional agents) was used up. After that only URLs already in hand, and pages linked from them, could be followed. Most municipal councils for the national-subsidy routes were never reached (see *Not reached* under each prefecture).

## Best sources found

| Source | Coverage | Years | Notes |
|---|---|---|---|
| 高松市離島航路確保維持改善協議会 (R8) 別紙「男木～高松航路の年間利用者数推移」 [pdf](https://www.city.takamatsu.kagawa.jp/kurashi/shinotorikumi/machidukuri/sogotoshikoutu/kakuho_iji/rito/r5ritoukouro.files/betusi.pdf) | Ogi–Megi–Takamatsu | FY2015–FY2025 (Apr–Mar) and Oct–Sep subsidy years H27–R7 | Chart data labels. Values with .5 as published. The Oct–Sep series matches the city's 事業評価 sheets for R1, R6 and R7 |
| 高松市 地域公共交通確保維持改善事業 事業評価 [page](https://www.city.takamatsu.kagawa.jp/kurashi/shinotorikumi/machidukuri/sogotoshikoutu/kakuho_iji/kotsu_kaizen.html) | Same route | H30–R7 sheets | Used as a cross-check only: R1 334,654; R6 218,571.0; R7 309,053.5 (Oct–Sep) |
| 南海電気鉄道「フェリー事業からの撤退について」2026-03-30 [pdf](https://www.nankai.co.jp/lib/company/ir/news/pdf/260330.pdf) | Wakayama–Tokushima | FY2019–FY2024 | In thousands. Includes drivers and car passengers. Withdrawal planned by March 2028 |
| 宿毛市定期船事業経営戦略 (R4.3) [pdf](https://www.city.sukumo.kochi.jp/fs/5/2/5/3/5/_/management_strategy.pdf) p.8 | Katashima–Ugurushima–Okinoshima | Oct–Sep 2016/17 to 2020/21 | Only a split into one-way and return-ticket persons. Unclear whether a return ticket counts one boarding or two |
| 四国運輸局「四国地方における運輸の動き30年」(H29.12) [pdf](https://wwwtb.mlit.go.jp/shikoku/content/unyu_30year.pdf) pp.11–21 | Route groups: 徳島～阪神, 香川～阪神, 高松～宇野, 西讃～中国, 愛媛～阪神, 愛媛～中国, 高知～阪神, 四国～九州, 四国～東京 | FY1987–FY2016 | Passengers plus vehicles. Groups do not list their member routes. 速報値. FY2015–16 rows put in the CSV only for groups inside scope. Long-distance groups (愛媛～阪神, 四国～東京) skipped |

## Rosters (for completeness checks)

- **四国運輸局管内国庫補助航路一覧**, 22 routes / 22 operators: [000291368.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000291368.pdf).
  - Tokushima (2): 伊島～答島, 牟岐～出羽島.
  - Kagawa (7): 本島～丸亀, 伊吹～観音寺, 男木～高松, 多度津～佐柳, 須田～粟島～宮の下, 宇野～土庄, 丸亀～広島.
  - Ehime (11): 魚島～弓削, 岡村～今治, 安居島～北条, 青島～長浜, 大島～八幡浜, 日振～宇和島, 津島～今治, 馬島～波止浜, 尾浦～宮窪, 三津浜～中島, 大島～黒島.
  - Kochi (2): 坂内～埋立 (須崎市), 沖の島～片島 (宿毛市).
- **一般旅客定期航路 route maps with operator tables, per prefecture.** Made by 四国運輸局 and hosted by 四国旅客船協会 at https://www.shikoku-ships.jp/route/.
  - Kagawa, 28 routes including Triennale-only services: [other01.pdf](https://www.shikoku-ships.jp/route/images/other01.pdf).
  - Ehime: other02.pdf and other03.pdf. Ehime/Kochi: other04.pdf. Tokushima/Kochi: other05.pdf, also at [4-1_tokusima.pdf](https://wwwtb.mlit.go.jp/shikoku/content/4-1_tokusima.pdf).
  - Only the Kagawa sheet was actually read.
- **四国対本州・九州間フェリーボート航路図** (as of 2022-04-01), 12 routes: [000290718.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000290718.pdf). Marks 高松～宇野 and 宿毛～佐伯 as 休止.
- **海上交通のサービス内容の現状と活性化方策に関する調査 報告書** (四国運輸局, R4.3) p.10 lists all 45 一般旅客定期航路 operators in the bureau area with their routes: [000301634.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000301634.pdf). The report says it compiled per-route yearly users from bureau data, but it publishes only port-statistics aggregates and qualitative hearing results.

## Aggregates for scale (not in CSV)

- **All routes under 四国運輸局** (一般 + 特定 + 不定期): [000329551.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000329551.pdf).
  - Passengers (千人): FY2017 9,872; FY2018 9,504; FY2019 9,986; FY2020 6,669; FY2021 6,811.
  - Passenger-km (千人キロ): FY2019 243,437; FY2021 133,685.
- **Shikoku to Honshu/Kyushu ferries**, same PDF (一般旅客定期航路 only). Passengers (千人): FY2017 2,856; FY2018 2,898; FY2019 2,706; FY2020 1,118; FY2021 1,289.
- **四国における旅客輸送の動き (R2年度)**, [000240677.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000240677.pdf) p.6. フェリー 四国～四国外: FY2018 324万, FY2019 306万, FY2020 138万. This matches the JPBA 3.24M figure for FY2018.
- **今治市地域公共交通計画 概要版 (R7–R11)**, [pdf](https://www.city.imabari.ehime.jp/kotsus/kokyokotu/keikaku/keikaku_gaiyou.pdf) p.1. A city-wide 航路 users line falls from 48万 to 29万 over about 11 years. The axis labels are too small to read, so it is not recorded.
- **今治市船舶運航事業経営戦略 (R3.3)**, せきぜん渡船 岡村～今治, [keiei.pdf](https://www.city.imabari.ehime.jp/kotsus/tosen/keiei.pdf). Header gives 年間輸送人員数 79千人 and 営業航路 26.2 km, but no year. The bar chart H22–R12 has no value labels. Not recorded.

## Checked, per prefecture

### Kagawa

- **Figures found:** 高松市 island-route council and 事業評価 (Ogi route only).
- **香川県統計年鑑 (R7):** chapter 11 運輸・通信 is Excel only. Table titles are not listed. CSV row with needs_download.
- **香川県離島振興計画 (R5–R14)** [pdf](https://www.pref.kagawa.lg.jp/documents/8225/kagawakenritousinkou2023.pdf): the 航路の現況 tables give distance, sailings and vessel only, no passengers. Checked Shodoshima p.16, Naoshima pp.53–55, Oshima pp.87–88, Ibuki pp.150–151.
- **第2期小豆島地域公共交通計画 (R8–R12)** and its 参考資料: bus ridership only. The ferry section (p.19) gives fares and sailings. Notes that 高松～草壁 is suspended from 2021-03-31 and 日生～大部 from 2023-12-01.
- **香川県 高松港 route list page:** no ridership.
- **Operator sites** (四国汽船 company page, 小豆島フェリー/四国フェリー info page): no ridership.
- **Wikipedia** (雌雄島海運, 本島汽船, 四国汽船): no ridership.
- **Not reached:** 丸亀市 (本島汽船, 備讃フェリー), 観音寺市 (伊吹), 多度津町 (佐柳), 三豊市 (粟島, 蔦島), 土庄町 (宇野～土庄), 直島町, 香川県地域公共交通計画. The first five are national-subsidy routes whose councils should publish 事業評価 sheets like Takamatsu's.

### Ehime

- **愛媛県 open-data statistics, 運輸 category:** only port-level 乗降人員 (港湾別). From the catalog listing; files not opened.
- **愛媛県離島振興計画 (R5–R14)** [pdf](https://www.mlit.go.jp/kokudoseisaku/chirit/content/001619208.pdf): 交通の現況 is text only (checked 魚島 p.19). Not all nine regions were read.
- **今治市:**
  - 地域公共交通計画: the 14.3 MB main plan exceeded the fetch limit. The summary version has only a city-wide aggregate.
  - せきぜん渡船 経営戦略: no labelled yearly figures.
  - せきぜん渡船 and さざなみ渡船 pages, and the せきぜん渡船航路改善協議会 page: no materials posted.
- **松山市 中島汽船:** MLIT case study [096_matsuyama2.pdf](https://www.mlit.go.jp/sogoseisaku/transport/pdf/096_matsuyama2.pdf) has timetables and route profit only. The 中島汽船 site failed with a certificate error.
- **Operators:** 国道九四フェリー company page, 宇和島運輸 company page, Orange Ferry: no ridership.
- **Wikipedia** (国道九四フェリー, 中島汽船, 今治市営渡船, 宇和島運輸): no ridership.
- **Not reached:** 上島町, 松山市 (安居島), 大洲市 (青島), 八幡浜市 (大島), 宇和島市 (日振), 新居浜市 (大島), 伊方町, 愛媛県地域公共交通計画.

### Tokushima

- **Figures found:** Nankai Ferry press release.
- **徳島県統計書 R6, 運輸・通信 chapter** (PDF, tables 106–117 read): no per-route ship passengers. Table 113 is port 乗降人員. Table 114 is ferry vehicle counts only (Ocean Trans and Nankai).
- **Wikipedia** 鳴門市営渡船: no ridership.
- **Not reached:** 阿南市 (伊島～答島), 牟岐町 (出羽島), 鳴門市 official pages.

### Kochi

- **Figures found:** 宿毛市 定期船 経営戦略.
- **高知県統計書 R7:** table 12-9 船舶輸送人員 is Excel only. CSV row with needs_download.
- **宿毛～佐伯 (宿毛フェリー):** the 宿毛市 page says 運休 (page dated 2020-01-15), and the bureau's 2022 map says 休止. Treat as not operating.
- **Wikipedia** 高知県営渡船: no ridership.
- **Not reached:** 須崎市 (坂内～埋立 巡航船), 高知県 official 県営渡船 pages.

### 四国運輸局 itself

- **Checked with no per-route figures:**
  - 離島航路 page and the subsidy overview [000302860.pdf](https://wwwtb.mlit.go.jp/shikoku/content/000302860.pdf).
  - 統計情報 pages. senpaku.html gives totals by route type only.
  - The R4.3 survey report (hearing section pp.77–84 is qualitative).
- **The 2024-04-26 release on operators' results since FY2020 returns 404.**
- **The latest 運輸の動き xlsx is listed but unopened** (CSV row with needs_download).

## Known gaps: sizeable routes with no figure

- **三崎～佐賀関 (国道九四フェリー)**, the busiest Shikoku–Kyushu crossing. Also 八幡浜～別府 and 八幡浜～臼杵 (宇和島運輸, 九四オレンジフェリー). These appear only inside the 四国～九州 group total, to FY2016.
- **The big Kagawa routes:** 高松～土庄 and 高松～池田 (小豆島フェリー, 国際両備), 高松～宮浦～宇野 (四国汽船), 宇野～豊島～土庄, 新岡山～土庄, 姫路～福田, 神戸～坂手～高松.
- **Ehime island routes:** 三津浜/高浜～中島 (中島汽船), 今治～大三島/岡村 (大三島ブルーライン, せきぜん渡船), 今治～上島町 (芸予汽船), 松山～宇品 (Chugoku agent).
- **Small national-subsidy routes:** all except Ogi–Takamatsu and Okinoshima.

## Definition oddities

- **Ogi–Takamatsu:** the CSV uses the April–March series. The same sheet has an Oct–Sep series used for subsidy accounting. Half values presumably count children as 0.5. Triennale years (2016, 2019, 2022, 2025) run 40–75% above the years either side.
- **Sukumo:** only the Oct–Sep subsidy year is published, so `year_basis` is `Oct-Sep` (not FY). `fiscal_year` is the year the period ends. The total is one-way plus return-ticket persons. It is unclear whether a return ticket is one boarding or two, so the figure may undercount boardings.
- **Nankai:** rounded to thousands, and includes vehicle occupants.
- **Route groups from the 30-year report:** 速報値, and the membership of each group is not stated. The operator names in those CSV rows are my inference.
