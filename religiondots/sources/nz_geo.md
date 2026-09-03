# New Zealand — boundaries: SA2 2023 for the data, SA1 2023 for the placement

Acquired 2026-08-30 into `data/geo/nz/`. No API key, no login, no bot protection —
`sources/nz.md` §4's note that datafinder wants a key and the ArcGIS feature services do not
still holds, and the services turned out to carry the *definitive* boundary products, not just
the census tables.

**Vintage: `SA22023_V1_00`, boundaries as at 1 January 2023, for every column.** `nz.md` §4 and
`spec.md` §8.1 both say why, and it is the mirror image of Connecticut: Stats NZ recodes the
2013 and 2018 census addresses *forward* onto the 2023 areas, so the older columns want the
*newer* geography. There is no case here for reaching for SA2 2018.

**Headline: the SA2 join is clean in both directions once the right layer is used.** 2,395
units in `data/normalized/nz.csv`; 2,395 records in the definitive layer; **0 polygons without
data**, and **16 data units without a polygon — all 16 non-digitised by Stats NZ, holding 24
people between them.** §4 is the whole story and the clipped layer is where the interesting
part is.

---

## 1. What was downloaded

| file | what | features | bytes |
|---|---|---:|---:|
| `sa2_2023_clipped.geojson` | SA2 2023, **clipped to the coastline**, + 2023 population | 2,311 | 71,179,431 |
| `sa2_2023_full.geojson` | SA2 2023, **definitive / full extent**, all codes | 2,395 | 102,070,457 |
| `sa1_2023_clipped.geojson` | SA1 2023, **clipped to the coastline**, + 2023 population | 32,817 | 140,373,163 |
| `sa1_2023_to_sa2_2023.csv` | SA1 → SA2 concordance, derived here (§5) | 32,817 | 750,141 |

Total ~314 MB, largest single request 140 MB, everything paged 500–1000 records at a time.

**Two SA2 files, on purpose.** The clipped layer is what the map should draw — it is the NZ
equivalent of choosing `cb_` over `tl_` in the US, and without it dots land in Wellington
Harbour. But clipping *deletes* the 84 SA2s that are entirely water, and 24 of those 84 have a
non-zero population. The full layer is kept so that the join can be closed and those 309 people
accounted for rather than silently dropped.

## 2. Exact URLs and how to re-fetch

All four come from Stats NZ's ArcGIS Online org `vKb0s8tBIA3bdocZ`. The org's service list is
at `https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services?f=pjson` (248 services);
`Statistical_Area_1_2023`, `Statistical_Area_2_2023`, `Statistical_Area_3_2023`,
`Meshblock_2023` and the annual 2018/2025/2026 siblings are all there.

```
BASE=https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services

# --- SA2 2023, clipped to the coastline (carries the census attributes too) ---
$BASE/2023_Census_totals_by_topic_for_individuals_by_SA2/FeatureServer/0/query
    ?where=1%3D1&outFields=SA22023_V1_00,SA22023_V1_00_NAME,VAR_1_3
    &returnGeometry=true&outSR=4326&f=geojson
    &resultOffset=<0,500,1000,...>&resultRecordCount=500

# --- SA2 2023, definitive (full extent), the clean 8-field boundary product ---
$BASE/Statistical_Area_2_2023/FeatureServer/0/query
    ?where=1%3D1&outFields=SA22023_V1_00,SA22023_V1_00_NAME,LAND_AREA_SQ_KM,AREA_SQ_KM
    &returnGeometry=true&outSR=4326&f=geojson&resultOffset=…&resultRecordCount=500

# --- SA1 2023, clipped to the coastline, with the 2023 population ---
$BASE/2023_Census_totals_by_topic_for_individuals_by_SA1/FeatureServer/0/query
    ?where=1%3D1&outFields=SA12023_V1_00,LANDWATER,VAR_1_3
    &returnGeometry=true&outSR=4326&f=geojson&resultOffset=…&resultRecordCount=1000
```

**Layer numbering, which `nz.md` §1 half-documented and is worth stating in full.** Both census
services carry four layers and the *even* ones are the clipped ones:

| layer | contents |
|---:|---|
| **0** | part 1, **clipped to coastline** — the religion columns, with geometry |
| 1 | part 1, unclipped (this is the attribute table `nz.py` reads) |
| 2 | part 2, clipped to coastline |
| 3 | part 2, unclipped |

`maxRecordCount` is 2000 on every service; smaller pages were used because the geometry is
full-resolution and the responses are large.

**`outSR=4326` was passed on every request**, so the service reprojected server-side — see §3.

## 3. CRS — reprojection was done, at the server

Everything Stats NZ publishes here is natively **EPSG:2193, NZGD2000 / New Zealand Transverse
Mercator 2000** — the service metadata reports `{"wkid": 2193, "latestWkid": 2193}` on all four
layers. All four files on disk are **EPSG:4326**, because `outSR=4326` was passed to the query
API and ArcGIS did the transform. No local `to_crs` step, and none needed: NZGD2000 and WGS84
are both ITRF-realised and agree to well under a metre for this purpose.

The extents make the clipped/full distinction visible without opening the geometry — in NZTM
metres, the full SA2 layer runs to `xmax = 2,523,320` and the clipped one stops at
`2,470,102`, the difference being open ocean.

## 4. The join report — both directions

Compared against the 2,395 distinct `geo_id` values in `data/normalized/nz.csv` at
`geo_level = sa2`, `source_id = nz_census_2023`.

| join | data units | polygons | matched | **data w/o polygon** | **polygon w/o data** |
|---|---:|---:|---:|---:|---:|
| SA2 2023 × **definitive / full** | 2,395 | 2,379 *(+16 null)* | 2,379 | **16** | **0** |
| SA2 2023 × **clipped to coastline** | 2,395 | 2,311 | 2,311 | **84** | **0** |

No duplicate codes in either file. **Nothing is unexplained in either direction**, and both
residuals are named below.

### The 16 that have no polygon anywhere

`Statistical_Area_2_2023` ships 2,395 records of which **2,379 are digitised and 16 have empty
or null geometry** — Stats NZ says so in the layer's own description, and the 16 are exactly
`400001` … `400016`. They are the "Area outside region" / oceanic pseudo-areas, including
`400004 Oceanic Oil Rig Taranaki`. Between them they hold **24 people (0.0005% of New
Zealand)**, 21 of whom are on the oil rig. There is no boundary set in which these get a
polygon; they are not a download problem and they should be dropped explicitly with a note
rather than lost.

### The 84 the clipping removes, and the 309 people in them

Clipping to the coastline removes every SA2 that is entirely water. 84 of the 2,395 go, holding
**309 people = 0.0062% of the country**; 60 of the 84 have zero population. **68 of the 84 are
recoverable from the full layer** (the other 16 are the null-geometry ones above). The
populated ones, largest first:

| SA2 | name | pop |
|---|---|---:|
| 192700 | Inlet Tauranga Harbour South | 51 |
| 100301 | Inlets Far North District | 45 |
| 303701 | Inlets Nelson City | 36 |
| 306401 | Marlborough Sounds Coastal Marine | 30 |
| 110600 | Oceanic Auckland Region East | 24 |
| 400004 | Oceanic Oil Rig Taranaki | 21 |
| 332601 | Inlet Port Lyttelton | 18 |
| 112001 | Inlets other Auckland | 15 |
| 108400 | Inlet Whangārei Harbour | 12 |
| 147300 | Bays Waiheke Island | 12 |
| 363400 | Oceanic West Coast Region | 6 |

…then thirteen more at 3 people each — `105001` Inlets other Whangarei District, `203300` Inlet
Ohiwa Harbour West, `258200` Oceanic Northland Region, `258500` Oceanic Waikato Region East,
`259100` Inlet Port Taranaki, `259500` Inlet Port Napier, `259600` Oceanic Wellington Region,
`259700` Inlet Wellington Harbour, `300401` Inlets Golden Bay, `344600` Inlet Port Oamaru,
`350801` Inlet Otago Harbour, `357100` Inlets Fiordland, `400001` New Zealand Economic Zone —
which is one FRR3 step and may be zero. These are `nz.md` §4's "inlets, harbours, oceanic areas
and forest parks": people who live on boats, on the rig, and in the handful of dwellings the
harbour SA2s pick up.

**Recommended handling:** draw from the clipped layer; for the 68 water SA2s with a real
polygon, either drop them with the residual counted, or fall back to the full-layer polygon and
accept dots on water for 309 people. Either is defensible; silently losing them is not.

### The population cross-check, which came out exact

The clipped SA2 layer carries `VAR_1_3` — the 2023 census usually resident population — so the
join can be checked on values as well as keys:

| | |
|---|---:|
| `sum(VAR_1_3)` over the 2,311 clipped polygons | **4,993,611** |
| `nz.csv` `Total` over the same 2,311 SA2s | **4,993,611** |
| `nz.csv` `Total` over all 2,395 SA2s | 4,993,920 |
| SA2s where the layer's population disagrees with `nz.csv` | **0** |

Exact agreement on all 2,311, and the 309-person gap is precisely the 84 SA2s above. This also
independently confirms `nz.py`'s handling of the `-999`/`-997` sentinels (`nz.md` §7): if the
sentinels had leaked into either side, this would not be zero.

## 5. SA1 2023 as the placement layer, and the concordance that makes it usable

`spec.md` §8.2's US trick is "allocate a county's dots equally across its tracts and the
geometry carries the weight". New Zealand does better than that: **the SA1 layer ships its own
2023 population**, so placement can be weighted properly instead of assumed.

- **32,817 SA1 polygons** in the clipped layer; the definitive layer has 33,164, so clipping
  removes 347. Measured over the clipped set, an SA1 holds a **median of 150 people, mean 152,
  IQR 120–183, max 1,038**, against SA2's median ~2,115 — a factor of fourteen finer, and much
  tighter than US census tracts (§8.2's IQR 2,818–4,043 between counties). 402 SA1s are empty.
- `LANDWATER` classifies each: **`12` Mainland 32,601** (pop 4,981,038), **`11` Island 145**
  (pop 12,270), **`21` Inland Water 71** (pop 6). Excluding `21` costs six people and keeps
  dots off lakes.
- `VAR_1_3` is the SA1's 2023 usually resident population. It sums to 4,993,314 — 297 below the
  SA2 total, which is the 347 clipped-away water SA1s plus FRR3 noise.

**Stats NZ publishes no SA1→SA2 lookup that is reachable without a datafinder key**, and
neither the SA1 boundary service nor the meshblock service carries an SA2 column — checked, the
fields are `SA12023_V1_00, LANDWATER, LANDWATER_NAME, LAND_AREA_SQ_KM, AREA_SQ_KM` and nothing
else. So the concordance was **derived spatially**, which is exact here because SA1s nest
inside SA2s by construction: each clipped SA1's `representative_point()` was joined `within`
the full-extent SA2 polygons.

    data/geo/nz/sa1_2023_to_sa2_2023.csv
    SA12023_V1_00, SA22023_V1_00, LANDWATER, POP_2023

- **0 SA1s landed in no SA2.** All 32,817 resolved.
- They hit **2,311 distinct SA2s** — exactly the clipped SA2 set, as they must.
- The 84 water SA2s have no SA1 in the placement layer, which is the same 309 people as §4 and
  needs the same decision, not a second one.

If an authoritative lookup is ever wanted, it is *Statistical Area 1 Higher Geographies 2023
(generalised)* on `datafinder.stats.govt.nz` — free, but behind a registration.

## 6. Licence — CC BY 4.0

From the portal item's own `licenseInfo` for `Statistical Area 2 2023`: **"Creative Commons
Attribution 4.0 International (CC BY 4.0)"**, `accessInformation` **"Stats NZ – Tatauranga
Aotearoa"**, and the same string as `copyrightText` on every layer used here. Identical to the
licence `nz.md` §2 records for the attribute data, which is expected — it is the same service.

Attribution required, no other restriction, commercial use permitted. Safe to ship.

## 7. Surprises, collected

1. **Clipping to the coastline deletes 84 SA2s, and 24 of them have residents.** The layer that
   is right for drawing is wrong for joining, and the difference is 309 people who live on
   water (§4).
2. **16 SA2s have no geometry at all, by publication** — Stats NZ ships them as records with
   null shapes and says so only in the layer description. `400004 Oceanic Oil Rig Taranaki`,
   population 21, will never have a polygon (§4).
3. **The definitive boundary services exist and are far better than the census layers for
   geometry** — `Statistical_Area_2_2023` is 8 fields against the census layer's 535 — but
   `nz.md` did not find them because they are not linked from the census hub pages. The org's
   `?f=pjson` service list is the way in (§2).
4. **The SA1 layer carries its own population**, so New Zealand does not need `spec.md` §8.2's
   equal-across-subunits approximation — it can weight placement exactly (§5).
5. **Nothing in the Stats NZ estate carries an SA1→SA2 column.** Not the SA1 service, not the
   meshblock service. The concordance had to be derived, and spatial nesting made that exact
   rather than approximate (§5).
6. **The population cross-check came out at exactly zero disagreements across 2,311 areas**,
   which is a stronger check on `nz.py`'s sentinel handling than anything in `nz.md` §7 — a
   single leaked `-999` would have shown.
