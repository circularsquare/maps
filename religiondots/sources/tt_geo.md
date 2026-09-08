# Trinidad and Tobago — boundaries and placement

Wired 2026-09-07. 15 municipality polygons, 4,610 Kontur hexes.

## What was downloaded

```
COD-AB   https://data.humdata.org/dataset/eed55f95-183c-48f7-adef-23dff31ec972/resource/
         218b72d0-35fb-4026-b8a6-152d87acea0d/download/tto_adm1_v2.zip             93,384 B
Kontur   https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/
         kontur_population_TT_20231101.gpkg.gz                                    334,548 B
```

The COD dataset ships `TTO_adm0`, `tto_adm1_v2` and `tto_adm_lines_v2` as separate
resources; only ADM1 is read, and its feature count is asserted after the read (§12, Chile).

## The join

**COD's ADM1 is the census's municipality tier exactly** — 15 polygons against Table 8's 15
drawn units. The tier is the country's local-government structure: nine regional
corporations, three boroughs, two cities and the Tobago House of Assembly area. It is
confirmed independently by CSO publishing **those same fifteen as separate
`... Individuals.xlsx` workbooks**, which is the same office agreeing with the boundary file
through a different product.

```
  census municipalities        15
  COD polygons                 15
  matched                      15
  census with no polygon        0
  polygons with no census       0
```

### Ten names differ, and all ten are punctuation

This is the country that justifies `fold()` dropping every non-alphanumeric character rather
than just collapsing spaces.

| census (Table 8) | COD `NAME_1` |
|---|---|
| `City of Port of Spain` | `Port of Spain` |
| `City of San Fernando` | `San Fernando` |
| `Borough of Arima` | `Arima` |
| `Borough of Chaguanas` | `Chaguanas` |
| `Borough of Point Fortin` | `Point Fortin` |
| `Couva/ Tabaquite/ Talparo` | `Couva-Tabaquite-Talparo` |
| `Mayaro/ Rio Claro` | `Mayaro/Rio Claro` |
| `Penal/ Debe` | `Penal-Debe` |
| `San Juan/Laventille` | `San Juan-Laventille` |
| `Tunapuna/ Piarco` | `Tunapuna/Piarco` |

Two systematic differences: Table 8 prefixes `City of` and `Borough of` on five units, and
the two sides disagree about whether a compound name is joined with `/`, `-` or `/ `. A
`HONORIFIC` regex strips the first and the alnum fold handles the second. **No alias table
is written** (§12: a frozen list of renames goes stale in silence at the next release).

Names on the drawn layer are taken from the **census**, not the boundary file (§12, Chile) —
so the map says `City of Port of Spain`.

### The p-code check

**CSO publishes no code at all in Table 8.** So `sources/tt.py` carries COD's `ADM1_PCODE`
against each municipality name by hand, and this file asserts the pairing from the boundary
side: 15/15. Nothing else would catch a transposition — every total in `tt.py` reconciles
whichever polygon a municipality is paired with (§9n's `TMA` lesson).

## Placement: Kontur, and the reason is the size *range*, not empty land

Unlike Belize or Malawi, Trinidad has no large uninhabited interior to keep dots out of. Its
problem is that the units are wildly unequal and **the small ones are the dense ones**.
Measured on the built layer:

| | km² | people | hexes |
|---|---|---|---|
| Borough of Arima | 13.1 | 33,404 | 18 |
| City of Port of Spain | 13.8 | 35,914 | 18 |
| City of San Fernando | 20.5 | 48,635 | 28 |
| Borough of Point Fortin | 28.5 | 20,161 | 40 |
| Borough of Chaguanas | 59.7 | 83,489 | 81 |
| … | | | |
| Couva/ Tabaquite/ Talparo | 742.4 | 178,160 | 832 |
| Mayaro/ Rio Claro | 787.5 | 35,649 | 414 |
| Sangre Grande | 931.0 | 75,605 | 687 |

**A 71-fold spread in area.** Uniform scatter within a polygon would be nearly harmless in a
13 km² borough and badly wrong in the big rural corporations: Sangre Grande's people are on
the coast road and the Eastern Main Road rather than spread over the Northern Range, and
Mayaro/Rio Claro's are coastal rather than in the interior forest.

**Tobago being its own unit makes the island split free** — no hex crosses between the
islands and the sea between them carries no weight.

```
  Kontur hexes                          4,788
  centroid outside every municipality     178   (14,590 people, 0.951%)  dropped
  kept                                  4,610
  Kontur 1,520,336 vs census 1,322,546 — ratio 1.150
```

For an island country the dropped ~1% is coastline disagreement between Kontur and COD, not
a land-border overrun.

### The per-municipality ratio

Range **0.95x to 1.31x** around a national 1.15x:

```
  San Fernando 0.95   Port of Spain 0.95   Diego Martin 1.09   San Juan/Laventille 1.09
  Arima 1.14   Tunapuna/Piarco 1.14   Siparia 1.15   Penal/Debe 1.17
  Point Fortin 1.17   Princes Town 1.17   Mayaro/Rio Claro 1.17   Tobago 1.18
  Couva/Tabaquite/Talparo 1.19   Sangre Grande 1.22   Chaguanas 1.31
```

**The 1.15x national level is a vintage artefact and is not a problem.** This is the widest
vintage gap on the map after Benin's — a 2023 modelled grid against a **2011** census, twelve
years — and the census figure here is additionally the *non-institutional* universe. Kontur
is used only as a within-municipality weight, so the level is irrelevant and only the shape
is load-bearing. Printed rather than asserted per unit (§9t).

The shape is tight, and the two ends both make sense: the **two cities read low** (0.95x)
because their resident populations have been falling while their daytime populations have
not, and **Chaguanas reads high** (1.31x) because it is the fastest-growing town in the
country and grew substantially after 2011.

### Eighteen hexes is thin, and it is thin where it matters least

Arima and Port of Spain hold 18 hexes each — a Kontur hex here measures **0.670 km²** (H3
r8). That is a coarse weight, and it is also a 13 km² fully-built borough where the true
distribution is close to uniform anyway.

This is worth stating because it is **the opposite of Saint Vincent's case** (§9ac), where
the grid was *coarser than the counting units* — 43 enumeration districts smaller than a
single hex, 78 of 219 getting none at all — and Kontur was measured, rejected and removed in
favour of uniform scatter. Here every unit has hexes, the largest units have hundreds, and
the units where the grid is thinnest are the ones that need it least. The rule holds in both
directions: **a population grid has to be finer than the counting tier to be worth using.**

## Not done

* **A finer tier.** CSO's 2011 Community Register has 3,619 community and enumeration-district
  rows with population and household counts, and would make an excellent placement layer —
  but it is a table, not geometry, and no matching boundary file was found. Kontur already
  does the job. Recorded in case a community shapefile ever surfaces.
