# Japan Rail Flow Map — Project Brief

## Goal

A national map of Japan's passenger rail network where **line thickness corresponds to passenger throughput per segment**. Static first (web-publishable image), interactive later if it makes sense.

Nobody appears to have published this. The data exists; the assembly does not.

## The metric

**輸送密度 (yusō mitsudo)** = passenger-km per route-km per day per segment. This is the industry-standard flow metric — JR companies and MLIT both publish in these units. Map line width to `log(輸送密度)` (raw values span ~3 orders of magnitude, from 100/day rural lines to 1M+/day Yamanote).

## Data sources

### Primary: gtfs-gis.jp consolidated dataset

**URL:** https://gtfs-gis.jp/railway_tsukajinin/

Akira Nishizawa (U-Tokyo CSIS) maintains a unified national GeoJSON of segment-level 輸送密度, updated ~quarterly. Latest: 2026-01-17. License: CC-BY 4.0 / ODbL.

- **GeoJSON (with geometry):** `https://gtfs-gis.jp/railway_tsukajinin/data/tsukajinin_20260117.geojson`
- **Tab-separated WKT (UTF-8):** `https://gtfs-gis.jp/railway_tsukajinin/data/tsukajinin_20260117.txt`
- **CSV no-geometry (Shift-JIS):** `https://gtfs-gis.jp/railway_tsukajinin/data/tsukajinin_20260117_sjis.csv`

Schema (columns): ID, 事業者名 (operator), 路線名 (line), 起点駅 (start station), 終点駅 (end station), 営業キロ (km), 種別, 単線複線, 備考, then one column per fiscal year: 2018年度 ... 2024年度. Sentinel value `-9999` for unavailable.

Coverage and granularity:
- **True per-segment data** (multiple segments per line — what we want): JR Hokkaido, JR East, JR West, JR Shikoku, JR Kyushu. Latest year: FY2024. Source: each company's IR page.
- **Single per-line value only** (uniform thickness — less interesting): JR Tōkai, all private rail, Tokyo Metro, Toei, third-sector, trams. Latest year: FY2023. Source: MLIT 鉄道統計年報.

Geometry: from MLIT 国土数値情報 (鉄道データ), with manual corrections from 地理院地図 where needed. WGS84.

### Secondary: 大都市交通センサス (for per-segment data inside Tokyo/Nagoya/Osaka)

**URL:** https://www.mlit.go.jp/sogoseisaku/transport/sosei_transport_tk_000035.html

12th Census (2015), Excel files. Per-segment station-to-station passenger flows for all operators in the three big metro areas. This is the only way to get per-segment data for Tokyo Metro, Toei, Tōkyū, Odakyū, Keiō, Hankyū, etc.

Key files (replicate for 中京圏 and 近畿圏 from the page):
- `駅別発着・駅間通過人員表 首都圏`: https://www.mlit.go.jp/common/001178992.xlsx
- `線別駅間移動人員 首都圏`: https://www.mlit.go.jp/common/001179095.xlsx
- `鉄道事業者コード 首都圏`: https://www.mlit.go.jp/common/001179686.xlsx
- `鉄道路線コード 首都圏`: https://www.mlit.go.jp/common/001179688.xlsx
- `鉄道駅コード 首都圏`: https://www.mlit.go.jp/common/001179689.xlsx

13th Census (Dec 2021) also exists on e-Stat but it's COVID-affected and the schema is reportedly painful. Stick with the 12th for v1.

**Important methodology note:** The census reports passenger flows (人/day on a segment), not 輸送密度 (人-km/km/day). For a segment of length L with N daily passengers, the equivalent density is just N — divide by L if you wanted person-km but for a single segment it's the same. The data should be directly comparable to gtfs-gis.jp values after a sanity check (pick a JR East segment that appears in both — Yamanote Shinagawa-Tabata or similar — and compare).

## Suggested implementation

### v1: Static national map
1. Fetch GeoJSON from gtfs-gis.jp.
2. Filter to a target fiscal year (FY2024 for the per-segment JR data; values will fall back to FY2023 for everyone else).
3. Drop sentinel `-9999`.
4. Render with GeoPandas + matplotlib or QGIS. Equal-area projection — Japan Plane Rectangular CS or just Albers Equal Area centered on Honshu. Line width: `lw = a + b * log10(density)` with `density` clamped to [100, 2_000_000]. Tune `a` and `b` so the Yamanote feels appropriately fat without obliterating Tokyo.
5. Base map: just coastline (国土数値情報 administrative boundaries, or Natural Earth). Don't add labels in v1; the shape alone tells the story.

### v2: Stitch in metro census data
1. Parse the 駅間通過人員 Excel for each metro area.
2. Match each census (operator + line + start_station + end_station) tuple to a geometry. This is the hard part: station-name matching between MLIT census codes and gtfs-gis.jp's 起点駅/終点駅 fields. Build a translation table; expect manual fixes.
3. For operators where gtfs-gis.jp has only per-line data (Tokyo Metro, Toei, private rail in metro), replace with per-segment values from the census.
4. For JR lines that appear in both, the gtfs-gis.jp 2024 number is more current — keep it.
5. Re-render.

### v3: Interactive
Leaflet or MapLibre with the GeoJSON; line-width as a paint property. Click a segment → show operator, line, segment, all years. Year slider would show the COVID dip nicely (2020 column collapses on commuter lines, holds steady on rural lines).

## Notable pitfalls

- The CSV is Shift-JIS. Use the txt or GeoJSON for sanity. Even the GeoJSON has Japanese strings — use UTF-8 throughout.
- `種別` distinguishes 鉄道 / 軌道 (tram) / 索道 (cable car) / モノレール / etc. Filter or style-by-type as makes sense. Funiculars and cable cars will have very low density and will visually disappear with log scaling — fine.
- Some segments have been abandoned mid-period (e.g., JR Hokkaido 札沼線 北海道医療大学–新十津川 closed May 2020); 備考 column flags these. Drop or grey-out.
- New lines after the 西九州 Shinkansen opening (Sept 2022) have geometry digitized from 地理院地図 rather than MLIT — quality should be fine but worth knowing.
- Geometry segments and density segments are 1:1 in the gtfs-gis.jp file, but the geometry occasionally has multiple LineString pieces per row (`MultiLineString`). Handle accordingly.
- For the v2 census join: the 12th Census uses 2015 station/line codes. Some operators have since renamed lines, opened new sections, etc. Build the join table against a 2015-era reference, not current Wikipedia.
- License attribution: gtfs-gis.jp data is CC-BY 4.0 (credit Akira Nishizawa / 地域・交通データ研究所) and ODbL. MLIT census is government public-use. Geometry credit goes to 国土数値情報 (MLIT).

## Sanity-check targets

Once rendered, check:
- Yamanote Line (Shinagawa–Tabata): ~1.1M/day. Should be the fattest line on the map.
- Tōkaidō Shinkansen (Tokyo–Shin-Osaka): ~230-300k/day, uniform.
- Tōkaidō Main Line tapering: Tokyo–Atami ~450k, Shizuoka–Hamamatsu ~40k.
- JR Hokkaido 留萌本線, 花咲線 etc.: <500/day, should be barely visible threads.
- Anything labeled `-9999`: should be omitted, not rendered as zero.
