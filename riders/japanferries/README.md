# japanferries — Japan ferry passengers (layer for japanrail)

Research stage. The aim is an optional ferry layer on the japanrail map
([japanriders](../japanriders/)): each scheduled passenger route drawn with
thickness ∝ passengers per day, at the rail map's own width scale, and possibly
port bubbles. It would load the way the Korea overlay does, from
`../japanferries/data/`, fetched only when switched on.

## Why it takes assembling

No national per-route table is published. Every operator files a yearly
per-route report to MLIT's Maritime Bureau (内航旅客航路事業運航実績報告書), but
MLIT publishes only national totals from it. Per-route figures surface in
scattered places — prefecture yearbooks, island promotion plans, subsidy
evaluation sheets, municipal transport plans, operator reports — so coverage is
collected region by region.

## National sources

| Source | Gives | Years | Notes |
|---|---|---|---|
| 港湾統計（年報）第2表 船舶乗降人員表 | boarding + landing per port, domestic / international | 2010–2023 | e-Stat Excel. Mixes every route calling at a port (and probably sightseeing boats) |
| 旅客地域流動調査 表2 府県相互間輸送人員表 | prefecture × prefecture passengers, 旅客船 and 航送船 separately | FY2000–2023 | Built by MLIT from the per-route operator reports; a cell with a single route is that route |
| 国土数値情報 N09 定期旅客航路 | route lines, operator, sailings/day, capacity | FY2012 | Non-commercial licence; no passengers |
| ODPT ferry GTFS (MLIT standard format) | current geometry + timetables | current | ~30 operators on ckan.odpt.org |
| 幹線フェリー・旅客船旅客流動実態調査 | survey-day OD between port prefectures, weekday/holiday | 2011, 2015 | Not opened yet |
| MLIT 長距離フェリー航路の輸送実績 / 離島航路の旅客輸送実績 | national totals only | FY2019–2022 | Scale check: island routes 41.3M (FY2019), 28.5M (FY2021); 12 long-distance routes 2.06M (FY2022) |

## Per-route figures

Collected into `sources/<region>.csv` with notes in `sources/<region>.md`, one
column layout for every region:

```
prefecture,operator,route,ports,vessel,fiscal_year,year_basis,annual_passengers,passenger_km,route_km,needs_download,source_title,source_url,source_page,notes
```

Proven example: 沖縄県「離島関係資料」(令和6年3月) chapter 3, table (6)
旅客定期航路事業輸送実績 — passengers and passenger-km per operator and route,
FY2018–FY2022.

**Throughput measure.** Where a source gives passenger-km, passenger-km ÷ route
km ÷ 365 is the route's average daily throughput — the same 輸送密度 the rail
lines are drawn with. Otherwise annual passengers ÷ 365 for a point-to-point
route.

## Built so far

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `parse_ports.py` | `raw/port2024_koushu_t2.xlsx`, `raw/port2024_otsushu_t2.xlsx` (e-Stat 港湾統計 2024, 第2部/第3部 第2表) | `data/ports_2024.csv` | 375 ports (123 major, 252 minor). Port sums equal the published 総計 内 exactly |
| `geocode_ports.py` | the above + `raw/C02-14/` (国土数値情報 C02 港湾, 2014) | `data/ports_2024.geojson` | 374 placed. Three same-named ports resolved by municipality code in `OVERRIDES`. 野伏 (東京, 61,864) is not in C02 and is unplaced — probably 式根島's port, unconfirmed |
| `parse_flows.py` | `raw/flow_fy2024_t2.xlsx` (旅客船 sheet), `raw/flow_fy2024_t3.xlsx` (航送船) | `data/flows_fy2024.csv` | Prefecture level, thousands |
| `osm_ferries.overpassql` | Overpass | `raw/osm_ferries.json` | route=ferry ways + relations in a Japan bbox (Korean, Chinese, Russian lines included, filtered later). `raw/osm_ferries_mirror.json` is the same query from a mirror. Terminals: `nwr["amenity"="ferry_terminal"]; out center tags;` in the same bbox → `raw/osm_ferry_terminals.json` |
| `match_osm_ports.py` | OSM ways + terminals, ports | `data/osm_ferry_ways.geojson` | Joins ways into routes and ties route ends to statistics ports |
| `manual_sources.py` | `raw/dl/` | `sources/manual.csv` | Nagasaki port boardings + landings per route (CY2013–2017, CY2021–2025), Akashi–Iwaya (FY2015–FY2021), Tokyo Bay Ferry at 浜金谷 (CY2024, Chiba port yearbook) and 久里浜港's scheduled-route total (CY2018–2022, Yokosuka yearbook) |
| `merge_sources.py` | `sources/*.csv` | `data/route_figures.csv` | One table with a `region` column and parsed numbers; prints per-region coverage and suspect rows |

### Per-region coverage (after the first sweep, 2026-09-15)

`data/route_figures.csv`, 1,345 rows. "routes" counts distinct (prefecture, route) pairs
with at least one year; some are island or corridor totals rather than single routes.

| Region | rows | with passengers | routes | with passenger-km | to download | years |
|---|---|---|---|---|---|---|
| chugoku | 101 | 98 | 22 | 0 | 3 | 2015–2025 |
| hokkaido_tohoku | 135 | 135 | 26 | 0 | 0 | 2015–2024 |
| hokuriku_chubu | 119 | 119 | 29 | 0 | 0 | 2009–2025 |
| kanto | 281 | 275 | 44 | 0 | 5 | 2015–2025 |
| kinki | 55 | 49 | 18 | 0 | 4 | 2013–2025 |
| kyushu_north | 89 | 87 | 17 | 0 | 2 | 2013–2024 |
| kyushu_south_okinawa | 375 | 374 | 43 | 341 | 1 | 2015–2024 |
| longdistance | 97 | 97 | 14 | 0 | 0 | 2015–2024 |
| manual | 58 | 58 | 8 | 0 | 0 | 2013–2025 |
| shikoku | 35 | 32 | 8 | 0 | 3 | 2015–2025 |

Okinawa is the only region with passenger-km, from the prefecture's island data book
(every route, FY2015–FY2024, with route lengths). The Chugoku bureau's yearbook says the
region has 95 scheduled routes carrying 18.15M passengers and 131.5M passenger-km in
FY2024, and no published breakdown behind that total — a fair measure of how far
region-by-region collection still has to go.

Year bases are not uniform: subsidy evaluation sheets run October–September (recorded as
`OctSep` / `Oct-Sep`), port and city statistics are usually calendar years, bureau figures
fiscal. Kyushu and Nagasaki count a child as half a passenger. Miyajima's 来島者数 counts
arrivals only, so both directions are about double.

### Findings that shape the model

- **A port with one route gives that route's passengers.** Every passenger boards at one end and lands at the other, so a single-route port's boardings + landings equal the route's passengers. Akashi and Iwaya both show 611,739 in the 2024 port statistics; the operator's FY2021 figure is 611,871. 多比良 and 長洲 (the Ariake ferry) are also identical to each other (736,890).
- **The shortcut checks out where both ends are counted.** `match_osm_ports.py` writes
  `data/routes_from_ports.csv`: 87 routes whose passengers can be read straight off a
  single-route port. In 10 of them both ends are counted ports, so the same route is
  counted twice — median gap between the two counts 0.1%, and under 13% in every case
  except 子ノ口 / 休屋 on Lake Towada (84%), which is a sightseeing boat where riders
  return by road.
- **…but not uniformly for car ferries.** Against transport-bureau route figures, some single-route ports match (茨城/大洗 172,601 vs 苫小牧～大洗 169,000; 大間 116,883 vs 函館～大間 116,000; 宮崎 + 志布志 323,031 vs 南九州～阪神 320,445; 苫小牧 843,075 vs its routes' ~835,000) and some run far low (八戸 151,882 = 46% of 苫小牧～八戸 332,000; 青森 429,355 = 71% of 函館～青森 604,000; 函館 389,262 = 54% of its two routes). The low ones are freight-heavy crossings, so some port managers probably leave out people riding in vehicles. For car ferries, prefer operator or bureau figures and treat the port figure as a floor. Port years are calendar, bureau years fiscal.
- **The 旅客地域流動調査 is no check on single routes.** Its 旅客船 matrix includes sightseeing boats and disagrees with the port statistics in both directions: 神奈川 3.77M departures against 390k port boardings, 大阪 3.89M against 792k, but 新潟 76k against Sado's ~1M. National total 60.5M vs 43.2M port boardings. Whether 航送船 passengers are inside 旅客船 or on top of it is not settled by the numbers.
- **C02 port points sit far from piers in big ports** (新潟 10 km, 苫小牧 8 km, 鹿児島 3–7 km), so ferry ends are matched through OSM terminal names first.

## Layout

```
raw/        downloads, as fetched (raw/dl/ = files the source agents flagged)
data/       parsed and joined outputs
sources/    per-region research: CSV of route figures + notes
```
