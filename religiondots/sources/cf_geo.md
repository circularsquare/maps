# Central African Republic — commune boundaries and the placement grid

**Built 2026-09-07.** `sources/cf_geo.py`. Counts: `sources/cf.md`.

Writes `data/geo/cf/cf_communes.gpkg` (177 units), `cf_hexes.gpkg` (34,651 Kontur hexes)
and `cf_lookup.csv`.

## 1. The join is free, and this is the sixth country that proves it

§11h's claim is that the USCB geodatabases remove the join entirely, because the boundaries
and the counts are in one file keyed on `GEO_MATCH`. Measured here rather than assumed:

```
ADM3 polygons          177
ADM3 count rows        177
GEO_MATCH in polygons only    0
GEO_MATCH in counts only      0
duplicate GEO_MATCH           0
```

`cf_geo.py` raises on any of those being non-zero. Every hard part of a dozen countries on
this map has been the join — China's cost the whole ingest, Ghana had an acronym collision,
Kenya had two files labelled ADM2 at different tiers, Mauritius needed a spatial check
against 183 French place names — and here there is nothing to verify beyond the assertion.

## 2. The vintage trap, and CAR is a cleaner case than Ethiopia

The geodatabase ships **two complete boundary sets**:

| set | vintage | ADM1 | ADM2 | ADM3 | used by |
|---|---|---:|---:|---:|---|
| `CF_GEOG1_*_2003` | 2003 | 17 | 72 | **177** | Age-Sex, Ethnicity, **Religion**, Language, Housing, Health, Agriculture |
| `CF_GEOG2_*_2021` | 2021 | 20 | 80 | 181 | Population, Displacement |

**The religion layer keys to GEOG1**, so counts and boundaries are the *same* vintage. That
is better than Ethiopia (§9u), where 2007 counts sit on 2021 woredas and 418 units carry a
`USCBCMNT` recording which census-era unit they came out of. Here only **four** communes carry
a comment at all — Boda *"Split from Lobaye"*, Berberati *"Split from Basse-Batouri"*, and two
more — and they are carried into `cf.csv`'s `note` regardless.

**Reading GEOG2 by mistake would look like a join failure rather than a vintage error**: 181
polygons against 177 count rows, four unmatched, and a plausible-looking "the boundary file
is missing some units" diagnosis. `cf_geo.py` asserts the layer name and the feature count
with that error message spelled out.

This is the same fact as `cf.md` §4's `Age-Sex` trap: **one workbook, two vintages, labelled
in the metadata and not in the data.**

## 3. Kontur is needed, and CAR is the strongest case on this map so far

| country | top-50 units by area | share of land | share of people |
|---|---|---:|---:|
| **CAR** | 50 of 177 communes | **73.0%** | **28.9%** |
| Ethiopia | 50 of 738 woredas | 38.4% | 5.3% |

Yalinga is **42,260 km² with 4,768 people** (0.11/km²) and Djémah **37,065 km² with 1,845**
(0.05/km²). Spread those uniformly inside their polygons and the whole east of the map fills
with an evenly spaced wash over country that is very nearly empty — the failure `ke_grid.py`
and `et_geo.py` exist to prevent, on a country where a larger share of the land is in the
biggest units than either of them.

```
kontur_population_CF_20231101.gpkg.gz   2,945,375 bytes -> 6,897,664
34,897 hexes, 5,821,233 people
246 hexes (66,096 people, 1.135%) have their centroid in no commune — border
overrun between Kontur's extract and the LSIB outline; dropped.
34,651 kept; every commune gets 4–1,148 hexes.
```

Joined on hex **centroids**, so no hex is split across two communes.

### The §8.2e floor is satisfied

The grid must be finer than the tier it weights (spec §8.2e, found on Saint Vincent, §9ac).
The smallest commune is **3.7 km²** against a 0.67 km² hex — about 5 hexes — and the measured
minimum is **4 hexes per commune**. Tight at the bottom end, and it holds everywhere.

## 4. THE FINDING: Kontur is not independent of this census, and the ratio band is not a check

`et_geo.py` and `ke_grid.py` both treat the Kontur/census ratio as a check that catches a
truncated download or a scrambled join. **On CAR it catches those and nothing else, and the
reason generalises.**

The national ratio is **1.500** — Kontur 2023 at 5,755,137 against the 2003 religion universe
of 3,836,736 — which is about right for twenty years of CAR's growth, and a value near 1.0
would have been the suspicious one. So far, ordinary.

The per-commune spread is the tell:

| | CAR (177 communes) | Ethiopia (738 woredas) |
|---|---:|---:|
| within ±1% of the median ratio | 29.9% | — |
| within ±2% | 55.4% | — |
| **within ±5%** | **78.5%** | **34.0%** |
| within ±10% | 88.7% | — |

**Independent modelling does not produce that.** Four fifths of CAR's communes sitting within
5% of one constant means Kontur's CAR extract is, at commune level, very close to a flat
rescale of the same 2003 table this map is drawing. Which makes sense: **CAR has had no census
since RGPH03**, so there was nothing newer for Kontur to have used.

Consequences, in order of how far they go:

1. **The ratio agreeing proves nothing about either source here.** It is not evidence the
   download is good, beyond the crude "the file is not empty and the join is not scrambled".
   `cf_geo.py` measures the ±5% concentration on every run and prints the warning itself,
   rather than leaving this as a claim in a docstring that nobody re-checks.
2. **The grid is still used, and is still doing real work.** The only thing it is used *for*
   is **where inside a commune the people are**, which comes from settlement and building
   footprints and is genuinely not in the census. The level is never read — Kontur is a
   within-commune weight and never a population — so the derivation cannot contaminate a
   count.
3. **And the general rule**: *a population grid's agreement with a census is only evidence
   when the grid had a different census to be built from.* Where a country's last enumeration
   is the one being drawn, the grid is downstream of it, and the check has to be understood
   as a smoke test rather than a corroboration. Worth checking on any country whose census is
   its most recent — which on this map is a growing set.

### The tail is real, and it is the war

The 1–99% range is **0.89–2.65** and the extremes are not errors:

| ratio | commune | prefecture |
|---:|---|---|
| 8.62x | Ouakanga | Mambéré-Kadéï |
| 2.67x | Basse-Batouri | Mambéré-Kadéï |
| 2.64x | Bangui-Ketté | Basse-Kotto |
| 2.51x | 1er Arrondissement | Bangui |
| 2.29x | Lobaye | Lobaye |
| 2.24x | 6e Arrondissement | Bangui |

CAR has had a displacement crisis since 2012; several hundred thousand people left the
country and several hundred thousand more moved inside it, and Bangui grew hard. The
arrondissements above 2x and the western communes are that. Since only the within-commune
shape is read, none of it affects a count — but it is the reason the band is wide and the
reason it is reported rather than asserted.

## 5. Inland water

Nothing needed. CAR is landlocked, has no large lake, and the Oubangui and Chari are river
boundaries rather than areas. `scatter.py` reports *"no ocean polygons over this country,
nothing to clip"*, and the 1.135% of Kontur that lands outside every commune is border
overrun between two different renderings of the same national outline, not water.
