# Czechia — boundaries for `cz.csv` (obec + city district, Census 2021)

`data/geo/cz/` · acquired 2026-09-02 · pairs with `sources/cz.md`

**Join is clean: 6,254 of 6,254 data units matched. 4 polygons have no religion rows — the
four military districts, which is correct.**

---

## 1. What was downloaded

| file in `data/geo/cz/` | bytes | what |
|---|---:|---|
| `obce_gpkg.zip` | 6,672,841 | ČSÚ generalised **obce**, 2021 census vintage |
| `obce/csu_geodb_sde_CISOB_obyvatelstvo_etl_20210326.gpkg` | 13,197,312 | 6,258 features |
| `layer10044.zip` | 170,089 | ČSÚ generalised **městské části** |
| `l10044/csu_geodb_sde_CISMC_obyvatelstvo_etl_20210326.gpkg` | — | 142 features |

Built from those by `python sources/cz_geo.py`:

| file | what |
|---|---|
| `cz_finest.gpkg` | **the one `countries.py` reads.** 6,250 obce + 142 city districts = 6,392 polygons |
| `cz_replaced.csv` | the 8 obec codes that city districts replace |

## 2. Re-fetch recipe

```
curl -sSL -o data/geo/cz/obce_gpkg.zip \
  "https://geodata.csu.gov.cz/as/data/distribuce/Hosted/sldb2021_obyvatelstvo/FeatureServer/10043/gpkg.zip"
curl -sSL -o data/geo/cz/layer10044.zip \
  "https://geodata.csu.gov.cz/as/data/distribuce/Hosted/sldb2021_obyvatelstvo/FeatureServer/10044/gpkg.zip"
unzip -o -q data/geo/cz/obce_gpkg.zip   -d data/geo/cz/obce
unzip -o -q data/geo/cz/layer10044.zip  -d data/geo/cz/l10044
python sources/cz_geo.py
```

Open, no login, no key. `geojson.zip` and `gdb.zip` are served from the same paths if a
GeoPackage is inconvenient.

The obec layer is catalogued at
`https://data.gov.cz/zdroj/datové-sady/00025593/6becdd1d4cf9384df0800e3f86ab17af`
("Population for municipalities (2021 Census)"). **The city-district layer was not found in
the catalogue** — it was found by probing FeatureServer ids next to the obec layer's 10043.
Only 10043 and 10044 answer; 10040, 10041, 10042, 10045 and 10046 are all 404. So the id is
undocumented as far as this went, and if it moves, the catalogue is not where to look first.

`FeatureServer?f=json` returns **403**, so the layer list cannot be enumerated the normal
ArcGIS way. Probing is the method.

## 3. Vintage — correct by construction (spec §8.1)

Both files are stamped `etl_20210326` and carry a `datum` column of `2021-03-26`, the census
reference date. They are the boundaries the religion data was collected on, not the current
ones, which is what §8.1 requires. No Connecticut problem here.

Czechia's obec count is stable (6,254–6,258 depending on whether military districts are
counted), so this would have been a mild hazard anyway — unlike Ireland's 11% of Small Areas
changing code since 2016.

## 4. Two levels, and why they are merged into one layer

ČSÚ publishes the same 78 religion categories at both levels, and the 142 city districts
**subdivide** 8 obce rather than nesting as extra detail under them:

| city | obec code | districts |
|---|---|---|
| Praha | 554782 | 57 |
| Brno | 582786 | 29 |
| Ostrava | 554821 | 23 |
| Plzeň | 554791 | 10 |
| Opava | 505927 | 9 |
| Pardubice | 555134 | 8 |
| Ústí nad Labem | 554804 | 4 |
| Liberec | 563889 | 2 |

So the finest complete cover is *every obec that is not subdivided, plus every city district*
— 6,250 + 142 = 6,392 — and that is what `cz_finest.gpkg` holds. Using both levels unmerged
would draw those 8 cities twice.

**Why it is worth the step.** Czech obce have a median population of 435, finer than a US
census tract (3,424) and about the size of an Australian SA1 (406), so obec-as-placement is
already good by §8.2. The exception is the top of the distribution:

| | |
|---|---|
| median obec | 435 people |
| IQR | 214 – 941 |
| obce over 10,000 | 130, holding 51.4% of the country |
| obce over 100,000 | 6, holding 22.5% |
| **Prague, as one obec** | **1,301,432 — 12.4% of the country in one polygon** |

Uniformly scattering 1,301 dots across the Prague polygon is the worst case a dot map has.
At city-district level Prague is 57 units with a median of 6,189, which is not tract-fine but
is two orders of magnitude better.

**This is measured data, not an allocation.** ČSÚ publishes the counts at both levels, so
nothing is `derived` and every row may still become a ring (spec §3.10). That is the
difference between this and Canada's CSD figures.

## 5. The parent obec is derived spatially, not from a lookup

City districts nest exactly inside obce by construction, so a representative-point join is
not an approximation. `cz_geo.py` does that rather than fetching ČSÚ's territorial register
for one column — the same reasoning as Canada's dissemination-area join in `countries.py`.
It fails loudly if any district lands outside every obec.

Representative point rather than centroid: a district shaped around a river bend can have a
centroid outside its own outline.

## 6. Projection

Both layers are **EPSG:5514, S-JTSK / Krovak East North** — the Czech national grid, and a
sign-flipped one, so its coordinates are large negatives (`-904584, -1227295`). `scatter.py`
reprojects to EPSG:4326 and the result was checked against Czechia's real extent:

    lon 12.129 .. 18.834      lat 48.584 .. 51.022

which is the country. sources.md §5c's Northern Ireland warning is the general case — a
wrong assumed CRS still joins perfectly and puts everything hundreds of kilometres out.

## 7. The join, in three directions

- **Data units with no polygon: 0.** All 6,254 obce and all 142 city districts matched.
- **Polygons with no data: 4.** Libavá, Boletice, Hradiště and Březina — the military
  districts (vojenské újezdy). The ČSÚ boundary file is titled "municipalities **and
  military districts**"; the religion table covers obce only. Correct, not a loss.
- **Codes that match but carry no geometry: 0.**

## 8. Licence

ČSÚ open data via data.gov.cz, attribution to Český statistický úřad. Same terms as the
religion table itself — see `sources/cz.md` §8, and read them before shipping.
