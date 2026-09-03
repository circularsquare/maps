# Mexico — boundaries for `mx.csv` (municipio, Censo 2020)

`data/geo/mx/` · acquired 2026-08-30 · pairs with `sources/mx.md`

**Join is clean: 0 unmatched in both directions, 2,469 = 2,469, and Morelos `17034`/`17035`/
`17036` are all present.** Details in §5. This is the file spec §8.1 asks for and
geoBoundaries is not.

---

## 1. What was downloaded

| file in `data/geo/mx/` | bytes |
|---|---:|
| `mg_2020_integrado.zip` | 257,527,530 |

Unzipped to `mg2020/`, 50 files, 446,452,408 bytes:

| `mg2020/conjunto_de_datos/` | features | `.shp` bytes | `.dbf` bytes |
|---|---:|---:|---:|
| `00ent` — áreas geoestadísticas estatales | 32 | 12,006,048 | 2,850 |
| **`00mun` — áreas geoestadísticas municipales** | **2,469** | 57,205,228 | 224,841 |
| `00a` — AGEB urbanas **and** rurales | 81,451 | 203,912,192 | 2,688,109 |
| `00l` — polígonos de localidades urbanas y rurales amanzanadas | 50,308 | 85,198,612 | 7,294,886 |
| `00lpr` — localidades puntuales rurales (points) | 295,779 | 8,281,912 | 47,324,930 |

Plus `mg2020/catalogos/` (9 catalogues in csv+pdf, incl. `municipios.csv`) and
`mg2020/metadatos/` (xml/txt/html/rtf/pdf).

## 2. Re-fetch recipe — and the URL trap `mx.md` warned about

```
curl -sSL -o data/geo/mx/mg_2020_integrado.zip \
  "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/mg_2020_integrado.zip"
unzip -o data/geo/mx/mg_2020_integrado.zip -d data/geo/mx/mg2020
```

Open, no login, no captcha. Verify `Content-Length: 257527530` before trusting it.

**The widely-circulated URL is dead and fails silently.** Every blog post, R package and
search result points at

```
.../productos/geografia/marcogeo/889463807469_s.zip        <-- WRONG, soft-404
```

which returns **HTTP 200 with a 2,263-byte HTML page** — `mx.md` §1's IIS trap, in geography
form this time. The live path has the UPC **twice**: once as a directory, once in the
filename. `889463807469/889463807469_s.zip` is the real 2.89 GB national file;
`889463807469/mg_2020_integrado.zip` is the 245.6 MB national subset taken here.
**Any fetcher must reject a download under ~1 MB rather than believe a 200.**

**How to rediscover the URLs if they move.** The ficha page is a JS shell with no links in
its HTML; its data comes from an undocumented JSON endpoint:

```
curl -sS "https://www.inegi.org.mx/app/api/productos/interna_v2/ficha/datos?upc=889463807469&lang=false"
```

`info.generales.formatos[].url.valor` is the national zip; **`info.multiarchivos[0].hijos[]`
is the full menu** — 32 per-state zips (31–216 MB each, e.g.
`889463807469/17_morelos.zip`) plus `mg_2020_integrado.zip`. That list is where the
integrado file was found; it is not linked from anywhere obvious.

Ficha (human): <https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463807469>

## 3. Vintage and licence

**Vintage: Marco Geoestadístico, Censo de Población y Vivienda 2020 — INEGI's own census
geography**, `edicion: 2020`, UPC `889463807469`, program `Marco Geoestadístico (MG)`,
cve_programa 3219. Server `Last-Modified: 2021-02-15`; files inside dated 2021-02-12.
`catalogos/contenido.txt` states the corte is **"el Cierre del Censo de Población y Vivienda
2020"** and enumerates exactly *"32 áreas geoestadísticas estatales, 2 469 áreas
geoestadísticas municipales"* — the same 2,469 `mx.csv` carries. The underlying cartographic
base is *Marco Geoestadístico, septiembre 2019*.

**"Integrado" is a documented subset, not a different vintage.**
`conjunto_de_datos/Bitácora_cambios_MG_2020-2020_Integrado.csv` lists what was dropped
relative to the full 2.89 GB product: **manzana, frentes de manzana, ejes de vialidad,
caserío disperso, and the three SIA/SIL/SIP service layers**; and it **merges the separate
`A` (urban AGEB) and `AR` (rural AGEB) layers into one `00a`**. Municipio, entidad, AGEB,
locality polygon and locality point are all full-resolution national coverage. If manzana
(block) geometry is ever wanted, it is only in `889463807469_s.zip` (3,106,988,366 bytes) or
the per-state zips.

**Why not geoBoundaries** — spec §8.1 and `mx.md` §2 both call it: `gbOpen/MEX/ADM2` is
`boundaryYearRepresented: 2012` with **2,457** units and no CVEGEO key. The three Morelos
indigenous municipios created 2017–19 are absent from it and present here (§5).

**Licence: INEGI Términos de Libre Uso de la Información** —
<https://www.inegi.org.mx/inegi/terminos.html>. Redistribution, adaptation, extraction and
commercial use all permitted; credit INEGI, disclose transformations, do not imply
endorsement. Same terms as the statistics in `mx.md` §7, so one credit line covers both:

> **Fuente: INEGI, Marco Geoestadístico, Censo de Población y Vivienda 2020.**

## 4. Layers, fields, CRS

**CRS — the one thing that will bite.** All five layers ship in **Lambert Conformal Conic,
metres**, not lon/lat:

```
PROJCS["MEXICO_ITRF_2008_LCC", ... DATUM["D_ITRF_2008", SPHEROID["GRS_1980",...]],
  PROJECTION["Lambert_Conformal_Conic"], lat_0=12, lon_0=-102,
  lat_1=17.5, lat_2=29.5, x_0=2500000, y_0=0, UNIT["Meter",1]]
```

This is **parameter-for-parameter EPSG:6372 (Mexico ITRF2008 / LCC)** — identical proj4
strings — but it is ESRI WKT1 with a non-standard datum name, so **`gdf.crs.to_epsg()`
returns `None`**. `to_crs(4326)` works fine from the WKT; only code an explicit
`set_crs(6372, allow_override=True)` if something downstream demands a numeric code.
Native bounds are metres (911,292 – 4,082,997 E). Reprojected: **−118.36511, 14.53210 →
−86.71041, 32.71865** (the west edge is Isla Guadalupe, not the mainland).

`.cpg` says `UTF-8` on every layer, so `NOMGEO` accents read correctly without a fallback
encoding.

### `00mun` — the join layer

4 fields, no truncation problems: **`CVEGEO`**, `CVE_ENT`, `CVE_MUN`, `NOMGEO`.

- **`CVEGEO` is the join key for `mx.csv` `geo_id`.** 5-character string = entidad(2) +
  municipio(3), e.g. `01001`, `17034`, `32058`. **2,469 distinct, 0 null, all length 5.**
- **Zero-padding: read it as a string.** `dtype=str` on the CSV side, and on the geo side the
  DBF field is already character — but any tool that infers types will turn `01001` into
  `1001` and destroy the join for the **289 municipios in entidades `01`–`09`**. If you ever have
  to rebuild it, `CVE_ENT.zfill(2) + CVE_MUN.zfill(3)`, never an integer concat.
- Geometry: 2,385 Polygon + 84 MultiPolygon. **0 null, 0 empty, 0 invalid.**

### Finer layers (§6)

| layer | key field | note |
|---|---|---|
| `00a` | `CVEGEO` | **13 chars = urban AGEB** (ent+mun+loc+ageb), **9 chars = rural AGEB** (ent+mun+`0000`+ageb, with `CVE_LOC = '0000'`). 81,451 distinct. Attribute is **`Ambito`**, mixed case — the documentation in `contenido.txt` says `AMBITO`, the DBF says `Ambito`. `00l` really does use `AMBITO`. |
| `00l` | `CVEGEO` (9) | 50,308 = 45,397 Rural + 4,911 Urbana, `AMBITO`, `NOMGEO` |
| `00lpr` | `CVEGEO` (16) | 295,779 points, `PLANO` ∈ {No 215,499 · Si 45,397 · Croquis 34,883} |
| `00ent` | `CVEGEO` (2) | 32, joins `mx.csv` `geo_level == 'entidad'` |

## 5. Join report

`mx.csv` filtered to `geo_level == 'municipio'`: 9,876 rows, **2,469 distinct `geo_id`**, all
5 characters. `mx_municipio_allocated.csv`: 56,787 rows, **2,469 distinct `geo_id`**.

| direction | count |
|---|---:|
| `mx.csv` municipio `geo_id` with no polygon in `00mun` | **0** |
| `00mun` polygons with no `mx.csv` row | **0** |
| `mx_municipio_allocated.csv` `geo_id` with no polygon | **0** |
| `00mun` polygons with no allocated row | **0** |

Nothing to explain in either direction. The named risk is explicitly cleared:

```
  Morelos 17034: PRESENT  Coatetelco
  Morelos 17035: PRESENT  Xoxocotla
  Morelos 17036: PRESENT  Hueyapan
```

Against geoBoundaries these three, and nine others, would have dropped with no error —
spec §8.1's Connecticut failure. `00ent` likewise matches `mx.csv`'s 32 `entidad` rows
(`CVEGEO` `01`…`32`, also zero-padded strings).

## 6. Placement layer — AGEB is Mexico's tract analogue, but weaker than the US one

Spec §8.2's trick is "allocate a unit's dots equally across a finer, population-equalised
layer." Mexico has one; it is **`00a` (AGEB)**, and it is **not as good as US census tracts**,
so here is the measurement rather than the assumption. Denominator is the classified base per
`mx.md` §6 (sum of the four ITER groups, 125,522,210), not `POBTOT`.

| finer layer | units | log-log r vs municipio population | people per unit, median (IQR) | worst unit |
|---|---:|---:|---|---:|
| **`00a` AGEB** | 81,451 | **0.787** | **1,077** (529 – 1,886) | 14,545 |
| `00l` locality polygons | 50,308 | 0.717 | 1,088 (653 – 2,080) | **1,833,433** |
| *(US census tracts, spec §8.2)* | 85,187 | *0.98* | *3,424 (2,818 – 4,043)* | — |

**AGEB wins and locality loses**, which is worth stating because locality is the intuitive
choice. Locality polygons have a catastrophic tail — one polygon holds 1.8M people (an entire
metro locality) while a rancho holds 60, so equal allocation across localities is badly
wrong. AGEBs are within a factor of ~3.5 across the interquartile range, the same order as
the US tract spread.

**The honest caveat is area, not count.** AGEBs are only population-equalised in the *urban*
half:

| | count | median area | 75th pct | max |
|---|---:|---:|---:|---:|
| urban AGEB (13-char) | 63,982 | **0.25 km²** | 0.48 km² | 36 km² |
| rural AGEB (9-char) | 17,469 | **93 km²** | 122 km² | 6,420 km² |

So a uniform scatter inside an urban AGEB is tight, and inside a rural AGEB spreads ~1,000
people over 93 km² of mostly empty land. The fix, if it matters, is already in this download
and costs no extra fetch: **within a rural AGEB, place dots at/near the `00lpr` locality
points** (295,779 of them, median 53 per municipio) rather than uniformly over the polygon.
`00lpr` covers 2,419 of 2,469 municipios; **the 50 with no rural point are entirely urban** —
ten of Mexico City's boroughs (Iztapalapa, Cuauhtémoc, Gustavo A. Madero, …), San Nicolás de
los Garza, Jaltenco, and small fully-nucleated Oaxacan and Chiapan municipios. An expected
absence, named here so it is not mistaken for a gap.

**Data quality:** `00a` has **22 invalid geometries of 81,451 (0.027%)** — 6 urban, 16 rural,
listed by `CVEGEO` in the scan; all self-intersections. `.buffer(0)` or `shapely.make_valid`
before any point-in-polygon work. `00mun`, the layer that actually matters for the join, has
**0 invalid**.

**Not fetched, and why.** Manzana (block) polygons would be the true tract-equivalent for
urban Mexico, but they only exist in the 2.89 GB national zip or the 32 per-state zips, and
AGEB is already fine enough that the limiting factor is the data, not the geometry:
`mx.csv` is four categories at municipio level (`mx.md` §3.2), so nothing below AGEB buys
extra truth. Per-state URLs are in the API response described in §2 if that changes.
