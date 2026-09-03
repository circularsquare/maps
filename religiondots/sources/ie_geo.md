# Ireland — boundaries for `ie.csv` (Small Area, Census 2022)

`data/geo/ie/` · acquired 2026-08-30 · pairs with `sources/ie.md`

**Join is clean: 0 unmatched in both directions, 18,919 = 18,919.** Details in §5.

---

## 1. What was downloaded

| file in `data/geo/ie/` | bytes |
|---|---:|
| `CSO_Small_Areas_National_Statistical_Boundaries_2022_Ungeneralised.zip` | 125,944,598 |
| `smallareas2022/SMALL_AREA_2022.shp` | 181,127,788 |
| `smallareas2022/SMALL_AREA_2022.dbf` | 13,376,663 |
| `smallareas2022/SMALL_AREA_2022.shx` | 151,452 |
| `smallareas2022/SMALL_AREA_2022.prj` | 145 |
| `smallareas2022/SMALL_AREA_2022.cpg` | 5 (`UTF-8`) |

One layer, 18,919 features. **Ungeneralised**, per `ie.md` §3's instruction — the
generalised 20 m variant (`SMALL_AREA_2022_Genralised_20m_view`, INEGI-style typo theirs)
exists on the same server and was not taken.

## 2. Re-fetch recipe

```
curl -sSL -o data/geo/ie/CSO_Small_Areas_National_Statistical_Boundaries_2022_Ungeneralised.zip \
  "https://hub.arcgis.com/api/v3/datasets/70a33cbb8bd7406da0d571be28624721_0/downloads/data?format=shp&spatialRefId=4326&where=1%3D1"
unzip -o data/geo/ie/CSO_Small_Areas_National_Statistical_Boundaries_2022_Ungeneralised.zip \
  -d data/geo/ie/smallareas2022
```

Open, no login, no bot protection. `spatialRefId=4326` is what reprojects it out of the
service's native Irish Transverse Mercator — **omit it and you get EPSG:2157 metres**, which
will not overlay a web map.

The formats the Hub endpoint will emit for this layer are
`csv, shapefile, sqlite, geoPackage, filegdb, featureCollection, geojson, kml, excel`;
swap `format=shp` for `format=geojson` if a shapefile's 10-character field-name truncation
(§4) is a nuisance.

**Provenance of that opaque id.** `70a33cbb8bd7406da0d571be28624721` is the ArcGIS item id
behind data.gov.ie's *CSO Small Areas – National Statistical Boundaries – 2022 –
Ungeneralised*. If it ever rots, rediscover it as:

```
# service directory, Tailte Éireann's ArcGIS Online org
https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/arcgis/rest/services?f=json
# -> Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view
https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/arcgis/rest/services/Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view/FeatureServer/0?f=json
```

The FeatureServer is also queryable directly (`Query,Extract`, `maxRecordCount` 2000, so
10 pages of `resultOffset` for the whole country) if the Hub download route is down. Note
the **service item id is `ff1d7175d0a94b62a6e66e7d2c292ea0` and the Hub download endpoint
404s on it** — the Hub wants the *dataset* id above, not the service item id. That cost a
round trip.

Landing pages, for when filenames change:
- <https://data.gov.ie/dataset/cso-small-areas-national-statistical-boundaries-2022-ungeneralised1>
- <https://data-osi.opendata.arcgis.com/datasets/osi::cso-small-areas-national-statistical-boundaries-2022-ungeneralised/about>

## 3. Vintage and licence

**Vintage: Census 2022 Small Areas** — the exact vintage `ie.csv` is published on.
Confirmed three ways, not assumed:

- the layer carries `SA_PUB2022` / `SA_GUID_2022` as first-class fields;
- ArcGIS item `created 2023-06-09`, S3 `content-last-modified 2023-08-04T09:51:09Z`, which is
  the August 2023 amendment `ie.md` §3 mentions (the one that took CSO EDs 3,419 → 3,420 —
  and this file does carry **3,420 distinct `ED_GUID`**);
- shapefile members are dated 2023-09-19; item `modified 2024-12-05`.

**The 2016 vintage trap is live and measurable in this file.** `SA_PUB2016 != SA_PUB2022`
for **1,448 of 18,919 (7.7%)**, and `SA_GUID_2016` has only **18,287 distinct values against
18,919 rows** — i.e. 632 of the 2022 Small Areas are splits sharing a 2016 parent. Joined to
a 2016 boundary set those would collide or drop silently. `SA_CHANGE_CODE` counts the churn:

| `SA_CHANGE_CODE` | rows |
|---:|---:|
| 0 (unchanged) | 14,552 |
| 1 | 703 |
| 2 | 317 |
| 3 | 2,497 |
| 4 | 850 |

**Licence: CC BY 4.0.** `licenseInfo` on the ArcGIS item is *Creative Commons Attribution
4.0 International*; `accessInformation` is *"© Central Statistics Office & Tailte Éireann"*;
the layer's `copyrightText` is *"© Central Statistics Office, National Mapping Division of
Tailte Éireann."* Same licence as the CSO statistics, so one credit line covers both:

> **Central Statistics Office, Census of Population 2022, and Tailte Éireann (CC BY 4.0)**

## 4. Layer, fields, CRS

- **Layer name:** `SMALL_AREA_2022` (service layer id 0).
- **CRS as downloaded: EPSG:4326**, WGS 84 lon/lat. `.prj` reads `GEOGCS["GCS_WGS_1984"…]`.
  Native service CRS is **EPSG:2157** (IRENET95 / Irish Transverse Mercator, metres) — that
  is what you get without `spatialRefId`.
- **Bounds:** −10.66297, 51.41990 → −5.99628, 55.44658. Republic only; no Northern Ireland.
- **Geometry:** 18,701 Polygon + 218 MultiPolygon. **0 null, 0 empty, 0 invalid** under
  `shapely.is_valid`. Already clipped to the coastline, so unbuffered random points inside a
  Small Area will not land in the sea.

**Join columns — and shapefile truncation.** The DBF cuts field names at 10 characters, so
the names in the service metadata are *not* the names geopandas hands you:

| service field | name in the `.dbf` | what it is |
|---|---|---|
| `SA_GUID_2022` | **`SA_GUID__1`** | **the join key for `ie.csv` `geo_id`** — CSO GUID, e.g. `4c07d11e-0a4f-851d-e053-ca3ca8c0ca7f` |
| `SA_PUB2022` | **`SA_PUB2022`** | **the join key for `ie.csv` `geo_name`** — 9-digit published SA code, e.g. `017010016`, with `/01`-style suffixes on splits |
| `SA_GEOGID_2022` | `SA_GEOGID_` | `SA_PUB2022` with an `A` prefix (`A017010016`) — SAPS `GEOGID`. Not used |
| `SA_GUID_2016` | `SA_GUID_20` | previous-vintage GUID. **Confusable with the 2022 one — the truncation makes 2016 look like the primary field.** Do not join on it |
| `SA_PUB2016` | `SA_PUB2016` | previous-vintage code |
| `SA_CHANGE_CODE` | `SA_CHANGE_` | churn flag, §3 |
| `ED_GUID`, `ED_ENGLISH`, `ED_ID_STR` | same / `ED_ENGLISH` | Electoral Division, 3,420 distinct |
| `COUNTY_CODE`, `COUNTY_ENGLISH` | `COUNTY_COD`, `COUNTY_ENG` | 34 distinct — CSO's 34-county cut, **not** `ie.csv`'s 31 FY106 administrative counties. Do not join county-level data on this without a crosswalk |
| `SA_NUTS1/2/3` (+ `_NAME`) | truncated | NUTS |
| `SA_URBAN_AREA_FLAG`, `_NAME` | `SA_URBAN_A`, `SA_URBAN_1` | urban flag |

Both `SA_GUID__1` and `SA_PUB2022` are **18,919 distinct, 0 null** — either is a safe
primary key. Prefer `SA_GUID__1`, since that is `ie.csv`'s `geo_id` verbatim and needs no
string handling. `SA_PUB2022` is a string with leading zeros and embedded `/`; never let a
reader coerce it to a number.

## 5. Join report

`ie.csv` filtered to `geo_level == 'small_area'`: 94,595 rows, **18,919 distinct `geo_id`**,
18,919 distinct `geo_name`. `ie_small_area_allocated.csv`: 454,056 rows, **18,919 distinct
`geo_id`**.

| direction | count |
|---|---:|
| `ie.csv` `geo_id` with no polygon (vs `SA_GUID__1`) | **0** |
| polygons with no `ie.csv` row | **0** |
| `ie.csv` `geo_name` with no polygon (vs `SA_PUB2022`) | **0** |
| polygons with no `ie.csv` row | **0** |
| `ie_small_area_allocated.csv` `geo_id` with no polygon | **0** |
| polygons with no allocated row | **0** |

Nothing to explain in either direction, on either key. This is the vintage match `ie.md` §3
asked for; against a 2016 set the first row would have read ≈1,448.

`ie.csv`'s other levels are **not** covered by this file: `state` (1), `county` (31 FY106
administrative counties), `electoral_division` (3,420). The ED and county boundaries live in
sibling layers on the same server
(`CSO_Electoral_Divisions_National_Statistical_Boundaries_2022_Ungeneralised_view`, and the
administrative-county set) and were not taken, because per `ie.md` §3 Small Areas are the
placement layer and the finer layer subsumes the coarser ones for drawing.

## 6. No finer layer is needed

`ie.md` §3 settles this and the file bears it out: Small Areas are built to 80–120 dwellings
(the layer's own description says so), median population 259, none empty. Scattering a
unit's dots uniformly inside its Small Area polygon *is* the population weighting — spec
§8.2's free lunch, with no grid fetched. At a median of 259 people, a Small Area is well
under one dot at 1:1,000.
