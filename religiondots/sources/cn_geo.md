# China — boundaries and placement

Built 2026-09-05. `sources/cn_geo.py` writes both layers and the county index `sources/cn.py`
joins against. `data/` is gitignored, so this file is the record.

| output | what | size |
|---|---|---|
| `data/raw/cn/datav/*.json` | cached DataV payloads, ~370 of them | small |
| `data/raw/cn/datav/county_index.json` | 2,848 county-level units: adcode, name, parent city | — |
| `data/geo/cn/cn_counties.gpkg` | 2,848 county polygons keyed by adcode | — |
| `data/geo/cn/cn_grid_3km.gpkg` | 167,896 Kontur H3 r6 hexes carrying `unit` and `pop` | — |

## 1. geoBoundaries is unusable for China, and this is the worst instance of §8.1 yet

spec §8.1 already says geoBoundaries' vintage is **a per-country fact to check, not a property
of the dataset**, on the evidence of Mexico's 2012 ADM2 missing three Morelos municipios.
China is that failure an order of magnitude worse, and it is not only vintage. `CHN ADM2`
carries 2,391 units against the census's 2,859, and:

- **Counties abolished in the 1980s are still in it.** Xizang gets **78 polygons for 73
  counties**, and the extras are `Yanjingxian`, `Saxunxian`, `Tuobaxian`, `Shengdaxian` —
  names retired decades ago. `Tongxian` was renamed in 1997.
- **Polygons are duplicated.** `Huinongxian` twice in Ningxia; `Banmaxian` and `Geermushi`
  twice in Qinghai.
- **Units sit in the wrong province.** Gansu's `Maquxian` under Qinghai, Hebei's
  `Dachanghuizuzizhixian` under Beijing.
- **The romanisation is corrupted in a patterned way** — `Erminxian` for Emin, `Wenshuxian`
  for Wensu, `Zhaoshuxian` for Zhaosu, `Shihezhishi` for Shihezi, `Duinongdeqingxian` for
  Duilongdeqing.
- And every big city's districts are merged into one polygon.

It matched **59.9%** of census counties. **Its `boundaryYearRepresented` of 2017 is fiction**;
treat that field as a claim rather than a fact, which is the transferable finding.

There is a second, quieter problem with it: **CHN ADM2 carries no code and no province
column**, only `shapeName`, so even the province had to be assigned by centroid against ADM1
— whose 34 units include Hong Kong, Macau, Taiwan, a stutter (`Ningxia Ningxia Hui Autonomous
Region`) and a typo (**`Guangzhou Province`** for Guangdong).

## 2. DataV GeoAtlas is the answer, and the reason is the adcode

`https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json` returns a unit's children
with geometry, free, no auth, no key. Walking province → city → county is ~370 requests and
yields **2,848 county-level units with unique GB/T 2260 adcodes**.

The adcode is the whole point. Once a census row is resolved to one, everything downstream is
a **code join** rather than a name join, and `cn.py`'s override table can be validated against
the index instead of trusted.

**Scope.** Mainland only: Taiwan, Hong Kong and Macau are separate entries in DataV and are
excluded because the 2000 census does not cover them, not as a statement about anything. The
nine-dash-line feature DataV ships alongside the provinces (`100000_JD`) 404s on `_full` and is
skipped. The census lists the Paracel, Spratly and Macclesfield groups as name-only rows with
no population, so they contribute nothing and no polygon is sought.

## 3. The coordinate question, which had to be checked rather than assumed

Chinese web mapping normally carries the **GCJ-02** offset, which displaces WGS84 coordinates
by 300–600 m. A GCJ-02 boundary layer joined to Kontur's WGS84 hexes would be invisible in the
middle of a county and wrong along every boundary and coast.

**The check is the per-province Kontur-to-census ratio**, reported rather than asserted per
§9i — the two measure different things (a 2023 modelled surface against a 2010 enumeration),
so demanding equality would either fail on every honest difference or be loosened until it
detected nothing. What a correct join looks like is a tight band; what a shifted one looks
like is the **coastal** provinces sagging while the inland ones sit at 1.

| | ratio |
|---|---|
| Sichuan | 0.969 |
| Hubei | 0.975 |
| Gansu | 0.993 |
| … | |
| Beijing | 1.175 |
| Tianjin | 1.214 |
| Shanghai | 1.327 |
| Xizang | 1.540 |
| **national** | **1.063**, per-province median 1.048 |

Every province is inside a factor of two, and **the coastal provinces are the highest rather
than the lowest** — which is real growth since 2010, and the opposite of what a GCJ-02 shift
would produce. The polygons are WGS84.

## 4. Placement, and why not the §8.2 trick

spec §8.2 places dots by splitting a unit's dots equally over a finer layer, on the grounds
that agencies build such layers to a population target. China's counties are not that: they
are historical units averaging 3,400 km², and in the west they are enormous and nearly empty.
**Xinjiang's Ruoqiang alone is ~200,000 km², the size of Belarus, with its people in a handful
of oases.** Spread evenly its dots would cover the Taklamakan.

So the weight is Kontur's measured H3 r6 surface, per §8.2d, exactly as Russia does. r6 rather
than r8 for the same reason as Russia: China draws ~30,700 dots at 1:1,000 against 167,896
populated hexes, so the grid is nowhere near what limits the picture.

Two approximations, both the ones `ru_geo.py` and `de_grid.py` name: a hex belongs to the
county containing its **centre**, and is then **clipped** to that county so a dot cannot land
outside its own unit or in the sea.

### The one new failure mode: a county smaller than a hex

An r6 hex is ~36 km². **Six counties captured no hex centre at all** and would have had
nowhere to put their dots — Tianjin's Heping is 10 km², Shanghai's Jing'an 7 km². These are
exactly the dense old city cores where an urban Hui community sits, and the failure is silent:
the scatter reports "units in the data have no polygons" and moves on, 27,487 people lighter.

Each such county now gets **its own polygon as a single placement cell**, which is §8.2's
uniform fallback applied to a unit small enough that uniform is a good answer. This will recur
in any country placed on a coarse grid with fine urban units, and the tell is the scatter's own
"no polygons" line — worth reading rather than scrolling past.
