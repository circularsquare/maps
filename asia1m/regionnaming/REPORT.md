# asia1m — region naming CSV

Generated from the **main** layer of `asia1m.ase` (9500×8000). **1619 regions**.

`regions_all.csv` — one row per region. Fill the `name` column; feed `country`/`province`/`admin_divisions` + `lat,lon` to Gemini.

## Method

- Each region = one exact color. Same-color blobs merged when within ~300px (8px-coarse dilation). Color reuse across >300px is split into separate regions.

- Excluded colors: `#d9f6ff` water, `#535d8d` border, `#64a0d2` coast, and **`#eead75`** — a 100px grid overlay found on the main layer (every 100×100 cell held exactly 100px), not a region.

- Georeferencing reproduces `generateAsia.py`'s Albers projection (lon_0=95, parallels 6/42, extent 47.7–129.2E / -9–65N). Validated by overlaying an independent coastline on your drawn coast — pixel-aligned.

- Admin descriptions: each region's pixels are intersected with rasterized admin boundaries; positional qualifiers (`northern half of X`) come from where the region sits inside each unit. For dense Chinese city cores (region covers <30% of its main GADM county) the description switches to township (街道/镇) detail from `xiangzhen.shp`.

- **Hierarchy roll-up**: if a region covers ~all of a parent (prefecture/province) it's named as the parent (`Gifu`, `Sakha`) instead of listing children; if it covers all-but-one/two children it's written `{parent} minus {child}` (`Tonghua minus Huinan and Meihekou`). China prefectures come from a spatial join to the OCHA adm2 layer (GADM's own prefecture field is unreliable).

- **Ordering**: the units the region mostly consists of are listed first (as leads); smaller edge units are demoted to a trailing `(also …)` with bare names — so a naming model anchors on the prominent places, not the slivers. A unit leads if it's the single largest or ≥20% of the region.

## Admin sources & levels (matching generateAsia.py)

- **China** — CHN_adm3 (county/district, GADM) + xiangzhen townships for cities; Chinese names in 名 parens
- **Hong Kong / Macau** — Natural Earth admin-1 (18 HK districts; Macau = 1 SAR unit)
- **Russia** — rus adm2 (raion, far-east only)
- **Mongolia** — admin1 aimag (province-level only — coarse)
- **North Korea** — admin2
- **South Korea** — kostat municipalities (si/gun/gu)
- **Japan** — admin2 municipality (+JA names)
- **Taiwan** — WhosOnFirst region (county/city level)
- **Vietnam** — GADM adm2 district
- **Laos/Cambodia/Thailand** — admin2
- **Myanmar** — admin3 township
- **Malaysia** — admin2 district
- **Singapore** — WhosOnFirst borough (5 regions)
- **Philippines** — admin3 municipality
- **Indonesia** — admin3 kecamatan
- **East Timor** — admin1 municipality
- **Brunei** — Natural Earth admin-1 (4 districts)

## Region counts by country

- China: 678
- Indonesia: 285
- Japan: 124
- Philippines: 113
- Vietnam: 105
- Thailand: 69
- South Korea: 52
- Myanmar: 51
- Malaysia: 35
- North Korea: 25
- Taiwan: 24
- Cambodia: 18
- Russia: 9
- Laos: 8
- Hong Kong: 7
- Mongolia: 6
- Singapore: 6
- ?: 2
- East Timor: 2

Area (px) percentiles: {5: 30, 25: 218, 50: 1267, 75: 4538, 95: 15849}. Tiny city regions kept (smallest ~7px).

## Edge cases to review

**Color reused across >300px (4)** — split into separate regions; all are 1px strays, likely stray clicks:

- `#f3efe5` → 2 regions: (97.9752,1.9637) 13px; (107.5704,-6.4934) 1px
- `#6a5c85` → 2 regions: (126.6045,45.7513) 1px; (119.6199,33.3213) 1px
- `#6b798e` → 2 regions: (119.4021,47.3855) 22px; (111.1312,43.3747) 1px
- `#f1f1f1` → 2 regions: (112.7428,71.2419) 4px; (145.7733,63.9488) 39px

**Near-duplicate colors on adjacent regions (12)** — two shades differing by ≤3 in RGB, regions <250px apart. Kept as **separate regions** (exact RGB = a distinct region, always); listed only in case a pair is an unintended mis-click in the artwork:

- `#fc9da3` ~ `#fc9fa3` (RGB dist 2), regions ~23px apart
- `#f5a08f` ~ `#f8a08f` (RGB dist 3), regions ~31px apart
- `#7dc3c9` ~ `#7dc6c9` (RGB dist 3), regions ~41px apart
- `#68b2c8` ~ `#6ab2c8` (RGB dist 2), regions ~45px apart
- `#faef8d` ~ `#faf28d` (RGB dist 3), regions ~56px apart
- `#eaf0b1` ~ `#ecf0b1` (RGB dist 2), regions ~58px apart
- `#f19479` ~ `#f39579` (RGB dist 3), regions ~89px apart
- `#f5caa1` ~ `#f5cda1` (RGB dist 3), regions ~90px apart
- `#e7c9ac` ~ `#e7cbac` (RGB dist 2), regions ~95px apart
- `#ecaeb0` ~ `#ecafaf` (RGB dist 2), regions ~159px apart
- `#eda7af` ~ `#eea7af` (RGB dist 1), regions ~195px apart
- `#f0741d` ~ `#f2741c` (RGB dist 3), regions ~208px apart

(46 further near-dup pairs exist but are far apart = intentional palette reuse.)


**Uncovered / offshore (3)** — little/no admin polygon under the region (small islands, or admin gap):

- ? 13px (139.5092,43.5171) — (no admin found)
- ? 13px (97.9752,1.9637) — (no admin found)
- Vietnam 43px (106.6021,8.714) — western part of Côn Đảo (Con Dao)

**Other notes**
- 10 regions under 5px — possibly stray pixels (below your ~7px minimum); flagged `tiny<5px`.
- `#f4d79e` near the Ryukyus is a legitimate scattered island-chain region (low bbox-fill but real).
- GADM China (CHN_adm3, ~2012 vintage) occasionally mis-attaches a Chinese name (e.g. Zhanjiang's urban core labelled 浈江区). Positions are correct; verify obscure names — your Gemini prompt already does.
- Mongolia/Russia descriptions are province/raion-level (only data available) — coarser than elsewhere.
