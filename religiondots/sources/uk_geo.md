# United Kingdom — boundaries: four censuses, four boundary sets, three grids

Acquired 2026-08-30 into `data/geo/uk/`. Nothing needed a login, an API key, or fought bot
protection.

`sources/uk.md` §4 sets the rule this file executes: **the UK is not one geography.** Each of
the three agencies publishes its own census areas, on its own vintage, in its own projection,
under its own file conventions. The boundary sets are kept in one directory but never merged,
and each is joined only to the `source_id` it belongs to.

**Headline: every join is clean in both directions.** 188,880 / 46,363 / 3,780 units in
`data/normalized/uk.csv`, 188,880 / 46,363 / 3,780 polygons, 0 unmatched either way.

---

## 1. What was downloaded

| jurisdiction | level | product | vintage | features | CRS |
|---|---|---|---|---:|---|
| England & Wales | Output Area | **OA (Dec 2021) Boundaries EW BGC (V2)** | Dec 2021 | 188,880 | EPSG:27700 |
| Scotland | Output Area | **Census 2022 Output Areas — MHW** | 2022 | 46,363 | EPSG:27700 |
| Northern Ireland | Data Zone | **DZ2021** (ESRI shapefile) | 2021 | 3,780 | **EPSG:29902** |
| England & Wales | MSOA | **MSOA (Dec 2021) Boundaries EW BGC (V3)** | Dec 2021 | 7,264 | EPSG:27700 |
| UK | LAD | **LAD (Dec 2021) Boundaries UK BGC** | **Dec 2021** | 374 | EPSG:27700 |

The last two are not needed for placement — Output Areas and Data Zones are the placement
layer, ~125 and ~500 people each — but `uk.csv` carries TS031's 60 categories at `msoa` and
`ltla` and those rows are unusable without polygons of the **right vintage**. §5 is about the
`ltla` one and it is the trap `uk.md` §4 warned of.

**Generalisation and clipping, product by product:**

- **ONS BGC** = *Generalised (20m), Clipped to the coastline (Mean High Water)*. This is ONS's
  standard web-mapping product; BFE (full extent) and BFC (full clipped) are 3–8× larger and
  BSC (super-generalised) loses inlets. BGC is the right rung.
- **NRS MHW** = *"clipped to the Mean High Water (MHW) Mark with inland water removed"*, quoted
  from `Census2022_SpatialDatasetsFileSpecification.pdf`, which ships inside the zip. The
  alternative, EoR (Extent of the Realm), runs out to the territorial limit and would scatter
  dots into the Firth of Forth. NRS does not publish a generalised Scottish OA set — MHW is
  full-resolution, which is why 46,363 Scottish OAs cost as much on disk as 188,880 English
  ones.
- **NISRA DZ2021** is full-resolution and land-only; NISRA publishes **no generalised variant**
  and no clipped/unclipped distinction. 3,780 polygons at full detail is 28 MB and fine.

## 2. Exact URLs and how to re-fetch

```bash
# ---- England & Wales: ONS Open Geography Portal (ArcGIS Hub export API) ----
# Item 6beafcfd9b9c4c9993a06b6b199d7e6d = "Output Areas (December 2021) Boundaries EW BGC (V2)"
# The export API is asynchronous: the first call returns 202 + status ExportingData; poll the
# same URL until status == "Completed", then GET the resultUrl it hands back.  The resultUrl
# carries a build hash and WILL differ from the one below.
curl -sSL "https://hub.arcgis.com/api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d/shapefile?redirect=false&layers=0&spatialRefId=27700"
curl -sSL -o data/geo/uk/Output_Areas_2021_EW_BGC_V2.zip "<resultUrl>"

# MSOA BGC V3, item 6b282db29762450881ed5159259a6e4e
# LAD Dec 2021 UK BGC, item 7ceb69f99a024752b97ddac6b0323ab0   <- Dec 2021, NOT the current set
curl -sSL "https://hub.arcgis.com/api/download/v1/items/6b282db29762450881ed5159259a6e4e/shapefile?redirect=false&layers=0&spatialRefId=27700"
curl -sSL "https://hub.arcgis.com/api/download/v1/items/7ceb69f99a024752b97ddac6b0323ab0/shapefile?redirect=false&layers=0&spatialRefId=27700"

# ---- Scotland: National Records of Scotland, 2022 census geography products ----
# The /media/ path carries a content hash and WILL change; the current one is linked from
#   https://www.nrscotland.gov.uk/publications/2022-census-geography-products/
curl -sSL -o data/geo/uk/output-area-2022-mhw.zip \
  https://www.nrscotland.gov.uk/media/uwdpx4hn/output-area-2022-mhw.zip
curl -sSL -o data/geo/uk/outputarea2022_partremoved_mhw.zip \
  https://www.nrscotland.gov.uk/media/41sd3avo/outputarea2022_partremoved_mhw.zip

# ---- Northern Ireland: NISRA, published 21 Feb 2023 ----
#   https://www.nisra.gov.uk/publications/data-zone-boundaries-gis-format
curl -sSL -o data/geo/uk/geography-dz2021-esri-shapefile.zip \
  https://www.nisra.gov.uk/files/nisra/publications/geography-dz2021-esri-shapefile.zip
```

**Finding an ONS item id, since the service list is 3,905 entries long:** the org's services
are at `https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services?f=pjson`, and the
AGOL item behind a named service is
`https://www.arcgis.com/sharing/rest/search?f=json&q=<Service_Name>` — take the row whose
`type` is `Feature Service` and `owner` is `ONSGeography_data`. The service *name* embeds the
product code (`_BGC_`, `_BFE_`, `_BSC_`) and the vintage, so the name is the specification.

## 3. Files on disk

| path | bytes | what |
|---|---:|---|
| `Output_Areas_2021_EW_BGC_V2.zip` | 40,903,472 | as downloaded |
| `oa2021_ew_bgc/OA_2021_EW_BGC_V2.shp` + sidecars | 115,801,636 | unzipped |
| `ew_oa2021_bgc_4326.gpkg` | 109,580,288 | **reprojected, layer `ew_oa2021`** |
| `output-area-2022-mhw.zip` | 36,685,715 | as downloaded |
| `oa2022_scotland_mhw/OutputArea2022_MHW/` | 87,560,525 | unzipped (+ the file-spec PDF) |
| `sc_oa2022_mhw_4326.gpkg` | 92,127,232 | **reprojected, layer `sc_oa2022`** |
| `outputarea2022_partremoved_mhw.zip` | 306,383 | as downloaded — see §6 |
| `oa2022_scotland_partremoved_mhw/` | 581,213 | unzipped, 531 features |
| `geography-dz2021-esri-shapefile.zip` | 14,795,669 | as downloaded |
| `dz2021_ni/DZ2021.shp` + sidecars | 31,082,711 | unzipped |
| `ni_dz2021_4326.gpkg` | 30,916,608 | **reprojected, layer `ni_dz2021`** |
| `MSOA_December_2021_EW_BGC_V3.zip` | 7,886,927 | as downloaded |
| `msoa2021_ew_bgc/` | 18,308,324 | unzipped |
| `ew_msoa2021_bgc_4326.gpkg` | 21,442,560 | **reprojected, layer `ew_msoa2021`** |
| `Local_Authority_Districts_December_2021_UK_BGC.zip` | 4,726,345 | as downloaded |
| `lad2021_uk_bgc/` | 8,394,622 | unzipped |
| `uk_lad2021_bgc_4326.gpkg` | 8,826,880 | **reprojected, layer `uk_lad2021`** |

Total ~630 MB. Nothing was over 41 MB in a single request.

## 4. Layers, fields and CRS — **and yes, everything was reprojected**

**Three different national grids are in play, which is the thing to get wrong.**

| set | native CRS | note |
|---|---|---|
| ONS (OA, MSOA, LAD) | **EPSG:27700** OSGB36 / British National Grid | as expected |
| NRS (Scottish OA) | **EPSG:27700** | same grid — Scotland is on BNG, not its own |
| NISRA (DZ2021) | **EPSG:29902** TM65 / Irish Grid | **not 27700.** `DZ2021.prj` reads `PROJCS["TM65_Irish_Grid" … SPHEROID["Airy_Modified"]]` |

Reading the NI file as if it were British National Grid puts Northern Ireland in the Atlantic
several hundred kilometres south-west of itself, and does it silently — the numbers are in the
same range and nothing errors.

**Reprojection was done.** Every set has a companion `*_4326.gpkg` in **EPSG:4326**, written
with `geopandas.to_crs(4326)` (pyproj picks the OSTN15/OSGB36 and the TM65 datum shifts
itself), carrying only the join fields. The originals are kept unmodified alongside.

| GeoPackage | layer | fields kept |
|---|---|---|
| `ew_oa2021_bgc_4326.gpkg` | `ew_oa2021` | `OA21CD`, `LSOA21CD` |
| `sc_oa2022_mhw_4326.gpkg` | `sc_oa2022` | `code`, `Popcount`, `HHcount`, `council` |
| `ni_dz2021_4326.gpkg` | `ni_dz2021` | `DZ2021_cd`, `DZ2021_nm`, `LGD2014_cd` |
| `ew_msoa2021_bgc_4326.gpkg` | `ew_msoa2021` | `MSOA21CD`, `MSOA21NM` |
| `uk_lad2021_bgc_4326.gpkg` | `uk_lad2021` | `LAD21CD`, `LAD21NM` |

Full field lists in the source shapefiles:

- `OA_2021_EW_BGC_V2` — `OA21CD`, `LSOA21CD`, `LSOA21NM`, `LSOA21NMW`, `BNG_E`, `BNG_N`,
  `LAT`, `LONG`, `GlobalID`. **`OA21CD` is the join key** to `uk.csv` `geo_id` at
  `geo_level = output_area`, `source_id = uk_ew_census_2021`.
- `OutputArea2022_MHW` — `code`, `HHcount`, `Popcount`, `council`, `sqkm`, `hect`, `masterpc`,
  `easting`, `northing`, `Shape_Leng`, `Shape_Area`. **The key is `code`**, not `OA22CD`;
  NRS does not name its code column after the geography. `council` is the **2019** council-area
  code, not a 2022 one.
- `DZ2021` — `DZ2021_cd`, `DZ2021_nm`, `SDZ2021_cd`, `SDZ2021_nm`, `DEA2014_cd`, `DEA2014_nm`,
  `LGD2014_cd`, `LGD2014_nm`, `Area_ha`, `Perim_km`. **`DZ2021_cd` is the key**, and
  `LGD2014_cd` joins `uk.csv`'s `lgd` level for free.
- `MSOA_2021_EW_BGC_V3` — `MSOA21CD`, `MSOA21NM`, `MSOA21NMW`, `BNG_E`, `BNG_N`, `LAT`,
  `LONG`, `GlobalID`.
- `LAD_DEC_2021_UK_BGC` — `LAD21CD`, `LAD21NM`, `LAD21NMW`, `BNG_E`, `BNG_N`, `LONG`, `LAT`,
  `GlobalID`.

## 5. The join report — both directions, per jurisdiction

Compared against the distinct `geo_id` set of `data/normalized/uk.csv` per
`(source_id, geo_level)`.

| join | data units | polygons | matched | **data w/o polygon** | **polygon w/o data** |
|---|---:|---:|---:|---:|---:|
| E&W `output_area` TS030 × `OA_2021_EW_BGC_V2` | 188,880 | 188,880 | 188,880 | **0** | **0** |
| Scotland `output_area` UV205 × `OutputArea2022_MHW` | 46,363 | 46,363 | 46,363 | **0** | **0** |
| NI `data_zone` MS-B19 × `DZ2021` | 3,780 | 3,780 | 3,780 | **0** | **0** |
| NI `data_zone` MS-B23 × `DZ2021` | 3,780 | 3,780 | 3,780 | **0** | **0** |
| E&W `msoa` TS031 × `MSOA_2021_EW_BGC_V3` | 7,264 | 7,264 | 7,264 | **0** | **0** |
| E&W `ltla` TS031 × `LAD_DEC_2021_UK_BGC` | 331 | 374 | 331 | **0** | 43 *(expected)* |
| NI `lgd` MS-B20 × `LAD_DEC_2021_UK_BGC` (`N09*`) | 11 | 11 | 11 | **0** | **0** |

No duplicate codes in any file. Six of the seven joins are 0/0.

**The 43 is an expected absence, and naming it is the point of §8.1.** The LAD file is UK-wide
(374 districts); `uk.csv`'s `ltla` level is England and Wales only (331). The 43 extras are
exactly Northern Ireland's 11 `N09*` LGDs and Scotland's 32 `S12*` council areas — the two
places whose religion data sits at a different level in `uk.csv`. Nothing English or Welsh is
missing.

### The `ltla` vintage trap, measured

`uk.md` §4 flagged it; here is the number. `uk.csv`'s `ltla` rows are the **pre-2023
331-district set**, so the boundary file must be the **December 2021** LAD product, not the
current one. Verified against `LAD_DEC_2021_UK_BGC`:

- All **17 districts abolished on 1 April 2023** are present in both the data and the Dec-2021
  boundaries: Allerdale `E07000026`, Barrow-in-Furness `E07000027`, Carlisle `E07000028`,
  Copeland `E07000029`, Eden `E07000030`, South Lakeland `E07000031`, Craven `E07000163`,
  Hambleton `E07000164`, Harrogate `E07000165`, Richmondshire `E07000166`, Ryedale
  `E07000167`, Scarborough `E07000168`, Selby `E07000169`, Mendip `E07000187`, Sedgemoor
  `E07000188`, South Somerset `E07000189`, Somerset West and Taunton `E07000246`.
- The four successor unitaries — Cumberland, Westmorland and Furness, North Yorkshire,
  Somerset — are absent from both, as they should be.
- **Against a current LAD file those 17 would fail to match and 1,686,870 people — 2.83% of
  England and Wales — would vanish with no error**, taking Cumbria, North Yorkshire and
  Somerset off the map. That is §8.1's Connecticut, in England, and it is 47% larger than
  Connecticut was.

### The Scottish Output Area codes, as promised

`uk.md` §4's "safe case" holds: the 46,363 codes in `OutputArea2022_MHW` run `S00135307` to
`S00181669` and match the UV205 code set exactly. A 2011 boundary file would have matched
**nothing at all**, because the two vintages are disjoint — the loud failure, not the quiet one.

## 6. Two things in the Scottish file that are not join failures but will bite

**1. `Popcount` in the boundary file is *household* population and is 108,111 short.**

| | |
|---|---:|
| `sum(Popcount)` over the 46,363 polygons | **5,332,173** |
| `sum(All people)` over UV205 in `uk.csv` | **5,440,284** |
| difference | **−108,111 (−2.0%)** |

The file spec says why, verbatim: *"Popcount — 2022 Census **household** population count at OA
level."* Communal establishments — halls of residence, care homes, prisons, barracks — are
excluded. The gap is real and structural, not perturbation (perturbation on this file is
+0.068%, `uk.md` §5, three orders of magnitude smaller). **Use UV205's `All people` row as the
OA population; `Popcount` is only safe as a *relative* within-OA weight**, and even then it
under-weights every student hall in Scotland.

**2. 531 Output Areas have a part removed from the shapefile.**

From the same spec: the MHW file omits *"the parts of multipart Postcodes removed from an
Output Area"*, and ships them separately as `OutputArea2022_PartRemoved_MHW` (531 features,
531 distinct OA codes, downloaded here). The spec's own warning: *"the area held in 'sqkm' and
'hect' will not always agree with the area which users will be able to calculate from the
shapefile."* Each of the 531 still has its main polygon in the MHW file — the join is 46,363/0/0
either way — so nothing is lost from the *data*, but 531 OAs are geometrically incomplete and a
dot scattered inside one will never land in the removed sliver. Merge the part-removed file in
if exact OA area ever matters.

## 7. Licences — all Open Government Licence v3.0, all shippable

OGL v3.0 permits commercial reuse and adaptation with attribution. Required strings, verbatim
from each publisher:

- **England & Wales, and the UK LAD file** — ONS's licences page requires **both** lines:
  - *"Source: Office for National Statistics licensed under the Open Government Licence v.3.0"*
  - *"Contains OS data © Crown copyright and database right [year]"* — ONS writes the year as a
    placeholder; use the boundary product's own vintage, so **2021** for these files.
  (`https://www.ons.gov.uk/methodology/geography/licences`; the AGOL item's `licenseInfo`
  field is a bare link to that page.)
- **Scotland** — *"© Crown copyright 2024. Scotland's Census 2022, National Records of
  Scotland."* plus the Ordnance Survey line, boundaries being OS-derived.
- **Northern Ireland** — *"Contains Ordnance Survey of Northern Ireland information licensed
  under the Open Government Licence v3.0"*, quoted from the NISRA publication page.

Note the third attribution names **OSNI**, not OS — Northern Ireland's mapping agency is
separate, which is the same fact as the Irish Grid in §4 wearing a different hat.

## 8. Surprises, collected

1. **Northern Ireland is on the Irish Grid (EPSG:29902), not the British National Grid.** Three
   agencies, two projections, and the wrong one fails silently rather than loudly (§4).
2. **A current LAD file drops 1.69 million people, 2.83% of England and Wales.** The trap
   `uk.md` §4 predicted, measured (§5).
3. **NRS's `Popcount` is household population and is 108,111 short of the census** — a −2.0%
   error waiting for anyone who treats the boundary file's own population column as *the*
   population (§6).
4. **531 Scottish Output Areas are missing a piece of themselves**, shipped in a separate file
   NRS does not link prominently (§6).
5. **NRS names the Scottish OA code column `code`**, not `OA22CD` — and the council code in it
   is 2019 vintage inside a 2022 product.
6. **Scotland has no generalised Output Area boundary set.** 46,363 full-resolution Scottish
   OAs are 88 MB against 116 MB for 188,880 generalised English ones — four times fewer units,
   three quarters of the bytes.
7. **The ONS Hub export API is asynchronous and returns HTTP 202 with a progress percentage.**
   Treating the 202 body as the file gives you a 231-byte JSON named `.zip`.
