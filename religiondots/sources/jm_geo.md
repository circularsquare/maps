# Jamaica — boundaries and placement

`sources/jm_geo.py` -> `data/geo/jm/jm_parishes.gpkg` (14 polygons) and
`data/geo/jm/jm_grid_400m.gpkg` (13,373 Kontur hexes). Counts: `sources/jm.md`.

---

## 1. The join is free, and it is the reason the USCB series is worth having

`GEO_MATCH` keys the counts layer to the boundary layer by construction — `JAM_GEO1_01` …
`JAM_GEO1_14` on both sides. **14 table keys, 14 geo keys, 14 matched, 0 unmatched.** No name
matching, no code bridge, no aliases, nothing to check by centroid.

That is worth stating plainly because of what it cost everywhere else. In the same week:

| country | what the join cost |
|---|---|
| **Jamaica** | nothing — one key column, exact |
| Bosnia (`ba_geo.md`) | four documented repairs to geoBoundaries before a name join could even be attempted: a polygon named `Republika Srpska` that is Višegrad, a duplicated `Novi Grad` 130 km apart, a mislabelled `Kupres` pair, a typo |
| China (§9r) | the whole ingest, 59.9% → 94.1% over four passes |
| Korea (§9s) | geoBoundaries omits Yeonggwang-gun entirely; rebuilt from 11 eup/myeon |

§11h's claim that these files join *"by construction"* is now measured on a fifth country and
it keeps holding.

## 2. The vintage pair has to be checked, not assumed

The geodatabase ships **two boundary generations**:

* `JM_GEOG1_ADM0/ADM1/ADM2_2011_uscb_202302` — cut for the 2011 census
* `JM_GEOG2_ADM2_2012_uscb_202302` — cut for the 2012 survey, and used by the poverty table

§11j's finding on the Central African Republic is that a country's own file may offer both and
**taking the newer layers because they are newer silently breaks the join**. The religion table
is `JM_ETHNICITY_AND_RELIGION_GEOG1_2011census_...`, so `GEOG1` is the correct partner.
`build_units()` asserts the layer name contains both `GEOG1` and `2011` rather than trusting
the constant, so a re-release that renames the layers stops the build instead of pairing the
wrong ones.

Total area comes out at **10,998 km² against Jamaica's 10,991** — 0.06% over, which is a
complete cover with ordinary polygon generalisation.

## 3. The ADM2 "Special Areas" exist, carry no religion, and are deliberately not used

The gdb carries `JM_GEOG1_ADM2_2011` — STATIN's *Special Areas*, which USCB rebuilt from **248
original shapefiles**. They are real, census-vintage and finer than the parish.

**§9p's lesson is that an extra level can hide inside the finest published one**, so the
religion sheet was checked for it rather than assumed. It has **15 rows: one country and 14
parishes**, and every other USCB table for Jamaica is `GEOG1` at ADM1 too. The Special Areas
are a *boundary* tier with no counts behind them.

They were then considered as a **placement** weight and rejected in favour of Kontur, because
an administrative subdivision is uniform inside itself and a population surface is not.
Recorded here so the next reader does not find the layer and assume it was missed.

## 4. Kontur, and the tightest ratio band on this map

**Why a grid for a country this small.** 14 parishes over 10,991 km² averages 785 km², and
Jamaica's people are not spread across that evenly — Kingston and Saint Andrew are roughly a
quarter of the country on about 2% of its area, while the Cockpit Country and the Blue
Mountains are close to empty. Uniform scatter would put Saint Andrew's 569 dots across a
parish that is mostly mountain. §8.2's argument at the small end.

`kontur_population_JM_20231101` is 926 KB gzipped, 13,683 H3 r8 hexes, 2,825,544 modelled
people. 310 hexes (42,238 people) fall outside the parish cover — Kontur's usual outward
rounding at a coastline — and are dropped; 1,045 edge hexes are clipped.

**The independent check (§9p) comes back tighter here than anywhere else on this map:**

```
median 1.04x, min 0.99 (SAINT MARY), max 1.07 (SAINT CATHERINE)
band 0.52-2.07x  ->  14/14 inside
```

A spread of **0.99 to 1.07 across all fourteen units**. Two reasons, and both are worth
carrying:

1. **The vintages are close and the population barely moved.** 2011 census against a 2023
   surface, in a country whose population changed by about 1% over that period. Compare
   Bosnia the same week, where the same check sits at a **median 0.94x** because BiH emigrated
   heavily between its 2013 census and the same 2023 surface — §9u's warning against
   inheriting another country's band, from the other direction.
2. **Fourteen units give a scrambled join nowhere to hide.** With this few, the band is set
   at median/2 rather than Bosnia's median/3 and **more than one outlier is treated as a
   failure**, because at 14 units an outlier is not noise.

The tightness is not a quality signal about Jamaica so much as confirmation that the
`GEO_MATCH` join did what it says.

## 5. What is left

* **Nothing on the boundary side.** The counting tier, the boundaries and the placement grid
  are all present, matched and checked.
* **A 2022 census would need new boundaries checked against it**, since parish boundaries are
  stable but the USCB file is a 2011 cut.
