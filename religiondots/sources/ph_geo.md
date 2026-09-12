# Philippines — boundaries

`sources/ph_geo.py` → `data/geo/ph/ph_barangays.gpkg` (42,042 polygons) and `ph_lookup.csv`.

The Philippines needs two different geographies and the census supplies neither as a file: a
**unit tier** matching the 117 rows PSA publishes religion on, and a **placement layer** fine
enough that a dot means something inside a province of a million people. One geodatabase
turned out to have both.

---

## 1. The file

| | |
|---|---|
| dataset | *Philippines Subnational Population and Housing Data Tables with Administrative Boundaries, Based on National Censuses* |
| publisher | **U.S. Census Bureau**, edition `202402` |
| on HDX | `philippines-subnational-population-and-housing-data-tables` |
| file | `philippines.gdb.zip`, 304 MB → `data/raw/ph/philippines_uscb.gdb.zip` |
| boundaries | admin 0–4, aligned to the State Department's Large Scale International Boundaries |
| licence | US federal government work; HDX lists it public, no click-through |

```
https://data.humdata.org/dataset/809dfb22-77f4-482c-8560-79b07d20fc15/resource/
f7835bf0-8168-4f4e-9961-108167c381a1/download/philippines.gdb.zip
```

Downloads clean over plain `curl`, unlike everything on `psa.gov.ph` (sources/ph.md §1).

### Layers used

| layer | rows | what it is |
|---|---|---|
| `PH_GEOG_ADM2_2020_uscb_202402` | 116 | **the census tabulation tier**, not the administrative one |
| `PH_GEOG_ADM4_2020_uscb_202402` | 42,042 | barangays, with 10-digit PSGC in `NSO_CODE` |
| `PH_AGE_SEX_2020census_uscb_202402` | 43,810 | `BTOTL` = total population, at every level including barangay |
| `PH_RELIGION_2020census_uscb_202402` | 134 | PSA's religion table, carried as data — used as a **check**, never as a source |

43,810 = 1 + 17 + 116 + 1,634 + 42,042. The population layer covers every tier exactly once.

---

## 2. Why not COD or geoBoundaries

Both were checked first and both fail, in different ways:

- **OCHA COD-AB `cod-ab-phl`** publishes admin4 with **11,920** features against the
  country's ~42,000 barangays — it is the Mindanao humanitarian subset and there is no
  national barangay layer in it. Worse, its admin2 is the **88 plain provinces**, which is
  the wrong tier: PSA tabulates religion on *province excluding any highly urbanised city
  inside it*, so a join on plain provinces silently double counts all 33 HUCs. Cebu means
  Cebu-minus-Cebu-City-minus-Lapu-Lapu-minus-Mandaue in the census and does not in COD.
- **geoBoundaries PHL** has no ADM4 at all; the API returns ADM0–ADM3 only.

The USCB file has the right tier because it was cut *for these tables* rather than adapted to
them. Its ADM2 `NSO_NAME` values are PSA's own row labels, parentheticals included —
`Basilan (excluding the City of Isabela)`, `Maguindanao (including the City of Cotabato)`,
`Davao de Oro (Compostela Valley)`.

---

## 3. The join is by name, and three checks say it is right

`ADM2.NSO_NAME` → `ph.csv geo_name`. Exact on 116 of 117, no folding, no fuzzy matching, no
override table — both strings are copies of the same PSA table. The codes are no use here
anyway: ADM2 carries an old NSO-style `PH19007` where the census keys on the 10-digit PSGC
`1500700000`.

sources/lk_geo.md is the standing reminder that a join can be silently wrong, so the name
agreement is not what the script trusts. These are:

1. **Every census unit has barangays and every barangay has a census unit.** 117 / 117 both
   ways, no orphans.
2. **Total population / census household population, per unit.** A quantity the names do not
   determine. Min 1.0004, median 1.0027, max 1.0469 — every unit just above 1, which is the
   institutional population of spec §3.7 and which a scrambled join cannot produce. **The
   maximum is the City of Muntinlupa**, and the reason is the New Bilibid Prison: the
   national penitentiary is 4.7% of the city and is exactly the population a household
   table excludes.
3. **The geodatabase's own religion table against `ph.csv`.** `RLG_HPOP` is the household
   population PSA tabulated the religion figures on, and `ph.csv` holds the same quantity
   read separately from the workbook. **They agree EXACTLY, to the person, on 115 of 116
   units.** This is the check that makes the name join safe, because exact agreement is not
   something a wrong pairing produces.

---

## 4. The 117th unit: rebuilding the BARMM Interim Province

The one unit with no ADM2 polygon is the **Interim Province** — the 63 barangays detached
from six Cotabato municipalities by the 2019 Bangsamoro plebiscite (sources/ph.md §2). There
is no polygon for it in any boundary set anywhere: it is younger than every published
Philippine admin layer, and PSGC splits it into eight "SGU" cluster codes rather than giving
it a shape.

The USCB put those barangays back in Cotabato and **said so in the file**. Its ADM1 layer
carries `63 Interim Province Barangays moved to Soccsksargen`, and every one of the 63 ADM4
rows carries the cluster it came from in `USCBCMNT`:

```
From BARMM, Interim Province, Pigkawayan Cluster     12
From BARMM, Interim Province, Pikit Cluster I         8
From BARMM, Interim Province, Pikit Cluster II        8
From BARMM, Interim Province, Pikit Cluster III       8
From BARMM, Interim Province, Carmen Cluster          7
From BARMM, Interim Province, Kabacan Cluster         7
From BARMM, Interim Province, Midsayap Cluster I      7
From BARMM, Interim Province, Midsayap Cluster II     6
                                                     --
                                                     63
```

So the province is reconstructed out of its parts, and **the two halves of check 3 are the
same fact seen from either side**: Cotabato is the single unit whose `RLG_HPOP` disagrees
with the census, and it disagrees by **exactly 215,348**, which is exactly the Interim
Province's household population. Nothing else in the country moves. The script asserts both
the count of 63 and that the discrepancy equals the Interim Province's population, and
refuses to write anything if either changes — because the failure mode is silent: 215,348
people would simply be drawn as Cotabato.

The 63 barangays total 215,433 people on the total-population basis against 215,348
household, a ratio of 1.0004, which is check 2's tightest unit in the country.

---

## 5. Placement is weighted by barangay population

spec §8.2 normally leans on placement layers a statistical agency **designed to a population
target**, so an equal share per polygon is already a population weighting. Philippine
barangays are the opposite: they are the political base unit, they are not sized to anything,
and they run from a few hundred people to well over a hundred thousand.

So `pop` travels in the gpkg and `_PhBarangayWeighter` in countries.py splits each unit's
dots by it. 42,042 barangays for 108.7M people is ~2,590 each, comparable to a US census
tract — but **that number is a placement grain, not a measurement grain**. The counts are at
province and HUC, 117 units, ~929,000 people each, and nothing measures which barangay a
given church's members live in. A cluster on this map means *this province, drawn where its
people are*.

The weight is TOTAL population (109,033,245) where the counts are HOUSEHOLD population
(108,667,043). The 0.34% difference is the institutional population and is irrelevant to a
weight normalised inside each unit; it is what check 2 measures.

---

## 6. Gotchas

- **`fiona` with geometry is 60× slower than `pyogrio` without it.** Reading 42,042 ADM4
  features through `fiona.open` for an attribute-only check ran past two minutes; the same
  read via `pyogrio.read_dataframe(..., read_geometry=False)` is under a second, and *with*
  geometry it is 2.2 s. Every attribute pass in the script uses the latter.
- **Assert the feature count, never the absence of an exception.** sources/cl_geo.md's
  geodatabase opened cleanly, reported the right CRS and returned zero features from every
  layer. `_read()` here exists to make that impossible to miss (spec §12).
- **`Interim Province` is not findable by code.** No `PH19999*` value appears in ADM4 —
  the USCB rewrote those barangays' codes to their Cotabato ones. `USCBCMNT` is the only
  handle, which is why the script keys on that string and asserts the count.
- **One of PSA's 129 categories is literally `None`**, and pandas turns it into `NaN` under
  default parsing. Every read of `ph.csv` in this project passes
  `keep_default_na=False, na_values=[""]`; without it 43,931 people who reported no religion
  vanish with no error anywhere.
