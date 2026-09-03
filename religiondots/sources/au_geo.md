# Australia — map boundaries (ASGS Edition 3, 2021)

Companion to `sources/au.md`. Acquired 2026-08-30. Files live in `data/geo/au/` (gitignored via
`religiondots/.gitignore:7 data/`).

**The join is clean in both directions.** 0 data codes without a polygon, 1 polygon without data
(`ZZZZZZZZZ Outside Australia`, expected). Detail in §5.

---

## 1. What was fetched

| layer | file | zip | unzipped | features |
|---|---|---:|---:|---:|
| **SA2** — the data layer | `SA2_2021_AUST_SHP_GDA2020.zip` | 50,678,443 B (48.3 MB) | 72.1 MB | **2,473** |
| **SA1** — the placement layer | `SA1_2021_AUST_SHP_GDA2020.zip` | 100,630,598 B (96.0 MB) | 193.7 MB | **61,845** |

```
mkdir -p data/geo/au

curl -L -o data/geo/au/SA2_2021_AUST_SHP_GDA2020.zip \
  https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip

curl -L -o data/geo/au/SA1_2021_AUST_SHP_GDA2020.zip \
  https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SA1_2021_AUST_SHP_GDA2020.zip

unzip -o data/geo/au/SA2_2021_AUST_SHP_GDA2020.zip -d data/geo/au/SA2_2021_AUST_GDA2020
unzip -o data/geo/au/SA1_2021_AUST_SHP_GDA2020.zip -d data/geo/au/SA1_2021_AUST_GDA2020
```

Direct, anonymous, no login, no captcha, no rate limit. The URL in `au.md` §2 uses the path
`…asgs-edition-3/jul2021-jun2026/…`; ABS 301-redirects that to
`…asgs/edition-3-july-2021-june-2026/…`. Both work with `curl -L`; the second is canonical.

Landing page: <https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files>

### Not taken, and why

- **GDA94 twins** (`…_SHP_GDA94.zip`, 47.5 / 94.4 MB) exist. GDA2020 is the current national datum
  and `au.md` names it; the two differ by ~1.8 m, irrelevant at dot-map scale but no reason to
  take the older one.
- **`ASGS_2021_MAIN_STRUCTURE_GPKG_GDA2020.zip`** — one GeoPackage holding MB/SA1/SA2/SA3/SA4
  together. Equivalent content; shapefiles were taken because the rest of `data/geo/` is
  shapefiles and geopandas reads both identically.
- **Mesh Blocks** (`MB_2021_AUST_SHP_GDA2020.zip`, 358,009 MBs) — one level finer than SA1. Not
  needed: SA1 already carries the population weighting (§4), and MB would be ~6× the geometry for
  no gain.
- **No generalised version exists.** ABS publishes only these full-resolution digital boundary
  files for ASGS Ed3 Main Structure. This is fine for the stated worry — see §6, they are
  coastline-clipped, so dots cannot land in the sea.

---

## 2. Edition and vintage — spec §8.1

**ASGS Edition 3, July 2021 – June 2026.** Exactly the vintage `data/normalized/au.csv` is
published on: the DataPack's column is `SA2_CODE_2021`, and the shapefile's is `SA2_CODE21`.

The file's own `CHG_FLAG21` / `CHG_LBL21` fields state the churn against ASGS 2016:

| flag | label | SA2s |
|---|---|---:|
| 0 | No change | 2,065 |
| 1 | New | 304 |
| 3 | Name change | 104 |

**Measured what a 2016 file would have done**, because §8.1 says the failure is silent. Downloaded
ASGS 2016 SA2 (`1270055001_sa2_2016_aust_shape.zip`, 2,310 features) and joined `au.csv` against it:

| | |
|---|---:|
| data codes that **would match** a 2016 file | 2,169 = **87.7%** — looks like a working join |
| data codes with **no 2016 polygon** | **303** |
| people in those 303 | **3,866,694 = 15.2% of Australia** |
| matched codes whose 2016 label is anything but "no change"/"name change" | 0 |
| matched codes whose polygon area moved >5% between editions | 5 |

So the 2016 failure here is *mostly* the loud kind after all — a seventh of the country silently
absent, concentrated in the growth-corridor and inner-city splits — plus five reused codes ABS
calls unchanged whose outlines nonetheless moved (`801041121 Taylor` ×1.34, `117011320
Banksmeadow` ×1.17, `701021017 East Arm` ×1.15, `315021405 Mount Isa` ×1.09, `801041117 Gungahlin
- West` ×0.93). `au.md` §2 anticipated "a few hundred areas with the wrong outline"; the real
shape of it is 303 areas *entirely missing* and 5 quietly wrong. Both are disqualifying. The 2016
file was used for this measurement only and is not in `data/geo/`.

---

## 3. Layer, field and CRS facts

Neither zip contains a `.cpg`; GDAL reports the DBFs as **UTF-8**, and all 2,473 SA2 names are
pure ASCII, so encoding is a non-issue.

### SA2 — `data/geo/au/SA2_2021_AUST_GDA2020/SA2_2021_AUST_GDA2020.shp`

- driver ESRI Shapefile · 2,473 features · geometry **Polygon** · 0 invalid geometries
- **CRS `EPSG:7844` (GDA2020, geographic lat/lon)** — `.prj` is `GEOGCS["GDA2020", …
  AUTHORITY["EPSG",7844]]`. Note this is *not* 4326; reproject before any metric work
  (`EPSG:3577` GDA94 / Australian Albers is the usual equal-area choice).
- bounds `96.817, -43.740 → 167.998, -9.142` (includes Cocos/Keeling and Norfolk Island)
- fields: `SA2_CODE21`, `SA2_NAME21`, `CHG_FLAG21`, `CHG_LBL21`, `SA3_CODE21`, `SA3_NAME21`,
  `SA4_CODE21`, `SA4_NAME21`, `GCC_CODE21`, `GCC_NAME21`, `STE_CODE21`, `STE_NAME21`,
  `AUS_CODE21`, `AUS_NAME21`, `AREASQKM21` (float), `LOCI_URI21`

> **The join column is `SA2_CODE21`.** It is a 9-character **string** and must be read as one —
> leading digits are significant and the first digit is the state, exactly as `geo_id` in
> `au.csv`. One value (`ZZZZZZZZZ`) is not numeric at all, so any read that coerces this column
> to int will fail outright — which is the good outcome; a read that coerces and drops is not.

`STE_CODE21` is the ASGS state code `1`–`9` and lines up with `au.csv`'s `state` rows (`1`–`8`;
`9` Other Territories has no state row in the data).

### SA1 — `data/geo/au/SA1_2021_AUST_GDA2020/SA1_2021_AUST_GDA2020.shp`

- 61,845 features · same CRS `EPSG:7844` · 0 invalid geometries
- fields: `SA1_CODE21`, `CHG_FLAG21`, `CHG_LBL21`, then the same `SA2_…`→`AUS_…` parent chain,
  `AREASQKM21`, `LOCI_URI21`
- `SA1_CODE21` is 11 characters, all rows; `SA2_CODE21` is carried on every SA1 row, so the
  SA1→SA2 rollup needs no separate correspondence file.

---

## 4. SA1 as the placement layer — spec §8.2

`au.md` §1 argued SA1 is the right *placement* layer and the wrong *data* layer. Measured against
the SA2 populations in `au.csv` (`level=grand_total`), the §8.2 trick holds:

| | Australia SA1 | (US tract, spec §8.2) |
|---|---|---|
| units | 61,845 across 2,454 real SA2s | 85,187 across 3,143 counties |
| people per unit, parent means | median **406**, IQR **359 – 447** | median 3,424, IQR 2,818 – 4,043 |
| p05 – p95 | 242 – 513 | — |
| log-log correlation, parent pop vs child count | **r = 0.923** | r = 0.98 |

Median 406 against ABS's ~400 design target, and an IQR of ±12% — **tighter than US tracts
relative to their target.** Scattering an SA2's dots equally across its SA1s is therefore already
a population weighting, and no population layer is read.

Usable SA1s per real SA2: min 1, p05 2, **median 24**, mean 25.2, p95 51, max 69.
**Every one of the 2,454 real SA2 polygons has ≥1 SA1 child with real geometry** — 0 exceptions —
so no SA2 falls back to uniform-over-the-polygon.

The tail worth knowing, since §8.2 says the within-unit error is the one that matters: eight SA2s
run over 600 people per SA1, led by `306031163 Yarrabah` (2,505 people, 2 SA1s), `702031059
Thamarrurr` (2,118 / 2) and `318011466 Palm Island` (2,098 / 2) — remote Indigenous communities
that ABS keeps as single large SA1s. In those the dots are coarser than elsewhere, but the SA2 is
itself small, so the absolute placement error stays small.

---

## 5. The join report

Loaded `SA2_2021_AUST_GDA2020.shp` (2,473 codes) and `data/normalized/au.csv` (2,472 distinct
`sa2` `geo_id`), and compared the code sets **both ways**, plus a third direction that only shows
up in Australia.

### A. Codes in the data with no polygon — **0**

Nothing in `au.csv` would silently vanish. This is the Connecticut check of §8.1 and it is clean.

### B. Polygons with no data row — **1**

| code | name | state | expected? |
|---|---|---|---|
| `ZZZZZZZZZ` | Outside Australia | Outside Australia | **yes** |

**This is the 2,473 vs 2,472 discrepancy `au.md` §2 left open** — "the boundary file will name the
extra one immediately", and it did. `ZZZZZZZZZ` is the ASGS Ed3 sink for records outside Australia;
the census is place-of-usual-residence and excludes overseas visitors, so it has no G14 row by
construction. It also has **no geometry**, so it cannot produce phantom dots even if joined.

### C. Third direction — codes that match but have empty geometry — **18**

Present in both files, so a code-set join calls them matched, and then they scatter nothing. These
are `au.md` §2's special-purpose SA2s:

| pattern | name | count |
|---|---|---:|
| `x97979799` | Migratory - Offshore - Shipping (NSW/Vic./Qld/SA/WA/Tas./NT/ACT/OT) | 9 |
| `x99999499` | No usual address (NSW/Vic./Qld/SA/WA/Tas./NT/ACT/OT) | 9 |

**Verified set-equal to the 18 `geo_id`s the data flags `pseudo_sa2_no_boundary`** — 0 in the
shapefile-empty set that the data does not flag, 0 flagged that the shapefile does fill. They hold
**52,920 people = 0.2082%**, matching `au.md` §2 to the person. They are real census units and
must be **dropped deliberately by the flag**, not lost in the geometry join. `ZZZZZZZZZ` is a
19th empty polygon, but it carries no data.

**`AREASQKM21 == 0` finds none of them.** Their area is **`NaN`**, not `0` — the 19 empty rows are
the only nulls in the column, and the smallest real SA2 is 0.4642 km². So filter on
`geometry.is_empty | geometry.isna()`, on `AREASQKM21.isna()`, or on the data's own flag — never
on `area == 0`, which matches nothing and would leave all 18 in.

### Summary

| direction | count | unexplained |
|---|---:|---:|
| data codes with no polygon | 0 | **0** |
| polygons with no data row | 1 (`ZZZZZZZZZ`) | **0** |
| matched but empty geometry | 18 (all `pseudo_sa2_no_boundary`) | **0** |
| matched with real geometry | **2,454** | — |

`data/normalized/au_sa2_allocated.csv` (the 148-category allocation) carries the same 2,472
codes: 0 with no polygon, 1 polygon (`ZZZZZZZZZ`) with no row. Identical result.

### Also confirmed absent, as predicted

- **37 SA2s with a real polygon and zero population** (`au.md` §2 note 3) — Wollemi, Centennial
  Park, reservoirs, airports, military camps. Counted exactly 37. An expected absence.
- **SA1 empty geometries: 34**, and every one of them has a parent in the 18 pseudo SA2s or
  `ZZZZZZZZZ` (3–4 per Migratory-Offshore-Shipping, 1 per No-usual-address, 1 for Outside
  Australia). No real SA2 loses an SA1 to this.
- **SA1 → SA2 nesting: complete.** 0 SA1 parent codes absent from the SA2 file, 0 data SA2 codes
  without SA1 children.

### One cosmetic difference

`208031188` is `"Highett (East) - Cheltenham "` in the DataPack metadata (`au.csv` `geo_name`) and
`"Highett (East) - Cheltenham"` in the shapefile — a trailing space in the ABS metadata workbook.
Codes are unaffected; if names are ever joined, strip them.

---

## 6. Coastline clipping

ASGS boundaries are built up from Mesh Blocks, which are clipped to the Australian coastline, so
**no generalised or clipped variant is needed — these already are the clipped ones.**

Checked rather than assumed: `AREASQKM21` over all 2,473 SA2s sums to **7,688,095 km²** against
ABS's official Australian land area of **7,688,287 km²** — a difference of 192 km², 0.0025%. A
file that included territorial waters would be millions of km² larger. Dots scattered inside these
polygons land on land.

The 96.8°E west edge is Cocos (Keeling) Islands and the 168.0°E east edge is Norfolk Island, both
in `9` Other Territories — genuine ASGS extent, not stray geometry. Any viewer that fits bounds to
the data will want to exclude STE `9` or it will render a mostly-empty Indian-to-Pacific strip.

---

## 7. Licence

**Creative Commons Attribution 4.0 International**, stated on the download page: *"Copyright
Commonwealth of Australia administered by the ABS. Unless otherwise noted, content is licensed
under a Creative Commons Attribution 4.0 International licence."* Same licence as the religion
data, so `au.md` §7's attribution covers both:

> Based on Australian Bureau of Statistics data.

Suggested citation for the boundaries specifically:

> Australian Bureau of Statistics (2021), *Australian Statistical Geography Standard (ASGS)
> Edition 3, July 2021 – June 2026*, Digital Boundary Files, ABS Website, accessed 30 August 2026.

No registration, no redistribution restriction, commercial use fine.

---

## 8. Notes for whoever builds the scatter

1. **Read `SA2_CODE21` and `SA1_CODE21` as strings.** `dtype={"SA2_CODE21": str}`. `ZZZZZZZZZ`
   will break an int coercion loudly, which is the safe failure.
2. **Drop by flag, not by geometry test.** The 18 pseudo SA2s carry `flag=pseudo_sa2_no_boundary`
   in `au.csv`'s `note`; dropping them there is deliberate and documented. Filtering on empty
   geometry gets the same 18 plus `ZZZZZZZZZ` and is equally correct, but the flag is the
   self-documenting version. **Do not filter on `AREASQKM21 == 0` — the empty rows are `NaN`
   there, not `0`, so that test matches nothing and keeps all 18.**
3. **Reproject before area or distance work.** Source is `EPSG:7844` (GDA2020 geographic), not
   Web Mercator and not WGS84/4326. `EPSG:3577` for equal area; `EPSG:4326` is close enough to
   7844 for web display (sub-metre) but declare the change rather than assume it.
4. **Placement:** pick an SA1 uniformly at random within the SA2, then a uniform point within that
   SA1's polygon. That is the §8.2 weighting; equal-per-SA1 is the whole point, so do not weight
   by SA1 area.
5. **The map shows counts, not shares** (§8.2), unless SA1 or SA2 populations get joined later.
   `au.csv`'s `level=grand_total` rows are the SA2 denominators if that is ever wanted — they are
   already in the file and cost nothing.
