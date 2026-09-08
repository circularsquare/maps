# Myanmar — boundaries and placement

`sources/mm_geo.py` writes `data/geo/mm/mm_states.gpkg` + `mm_lookup.csv`;
`sources/mm_grid.py` writes `mm_hexes.gpkg`.

## What was downloaded

| | |
|---|---|
| boundaries | **OCHA COD-AB Myanmar** (MIMU's own set, redistributed), `mmr_admin_boundaries.shp.zip`, 34 MB from HDX |
| layer | `mmr_admin1.shp` — **18 features**, EPSG:4326, `adm1_name` + `adm1_pcode` |
| placement | **Kontur population, MM extract**, `kontur_population_MM_20231101.gpkg.gz`, 15.3 MB → 212,821 hexes |

Shapefile and not geodatabase, on §12's Chile rule. Read with `engine="fiona"`, feature count
asserted either way.

## Eighteen polygons for fifteen counted units

**The count says it before any join is attempted, which is §12's Kenya rule working exactly as
intended.** The standard p-code set splits two of the census's rows:

    MMR007 Bago (East)   + MMR008 Bago (West)                        ->  Bago
    MMR014 Shan (South)  + MMR015 Shan (North) + MMR016 Shan (East)  ->  Shan

**MIMU's own religion sheet confirms this is the intended aggregation rather than a guess**: it
codes Bago `MMR111` and Shan `MMR222` — two codes that exist precisely because the census
reports those states whole. Those aggregate codes are what the drawn `unit` uses, so the id on
the map is the code of the thing actually counted.

**The dissolve is a rule, not a list** (§12): strip a trailing parenthesised qualifier from
COD's `adm1_name` and group on the remainder. A future release that splits a third state is
then handled rather than silently dropped, and one that renames a state fails loudly.

### Three checks on the dissolve, and the second is the one that would catch a wrong grouping

* **the 15 groups pair 1:1 with the census's 15 rows.** One romanisation difference: DOP writes
  `Ayeyawady`, MIMU and OCHA write `Ayeyarwady`, so the fold drops `r`. Uniqueness is asserted
  on both sides, which is what makes a fold that aggressive safe on a set of fifteen.
* **every multi-member group must be CONTIGUOUS** — its members have to touch, tested with a
  50 m buffer in UTM. Grouping on a stripped name is a string operation; this is the geometric
  fact that says the string operation grouped real neighbours rather than a coincidence of
  naming.
* **the 15 dissolved polygons must tile with no overlap.** Sum of areas 670,830 km² against
  the area of their union 670,830 km², difference 0.0 — which is what rules out §12's Korea
  trap, where overlapping ADM1 polygons hand a `keep="first"` centroid join the wrong unit.

### And the free independent check

**The census's printed row order reproduces COD's `ADM1_PCODE` order on all 15**, sorting each
group by its lowest member code. DOP's table position and OCHA's code attribute have different
origins, so agreement is evidence rather than a tautology — and a single transposed row would
break it while every total in `mm.py` still reconciled (§12's `TMA` lesson).

The minted census id is `MM-01`..`MM-15` and deliberately not a p-code (§12, Benin).

## Placement: Kontur, and a hole check the border made necessary

15 states for 51.5M is ~3.4M each and they are wildly uneven — Yangon 7.36M in 9,867 km²,
Kachin 1.64M in 88,978, Chin 479k in 36,018 of mountain — so an equal share per polygon would
wash the empty north and squash a seventh of the country into one speck. §8.2's Kenya case.

**2,770 hexes (740,531 people, 1.35%) have centroids outside every state**, which is more than
most countries here and is explained by Myanmar's 5,700 km of land border: Kontur's extract runs
over into Bangladesh, India, China, Thailand and Laos.

**1.35% is large enough that it needed distinguishing from a hole in the dissolve**, because a
hex inside the country and inside no state would mean the fifteen do not cover Myanmar — and
that would empty real places with no error anywhere. **ADM0 is the independent test**: of the
2,770, exactly **0** fall inside Myanmar's own ADM0 outline. Asserted, not eyeballed.

Every state gets hexes, 1,439 to 37,977 each. Kontur totals 54,054,541 against the drawn
universe of 51,486,253 — ratio **1.050**, which is right for a 2023 grid against a 2014 census.

## The ratio band measures the expulsion, which is not what it is normally for

| | |
|---|---|
| 14 states | **0.80 – 1.15** against the population drawn on them |
| **Rakhine** | **0.58** against the drawn population — and **0.89** against the enumerated count alone |

**That difference is the 2017 expulsion.** Rakhine's drawn population is 2,098,807 enumerated
plus 1,090,000 non-enumerated; roughly three quarters of a million Rohingya left for Bangladesh
three years after the census, and Kontur's 2023 surface does not contain them. Against the
enumerated count Rakhine is 0.89 and unremarkable. **A diagnostic that normally checks a join
is, for this one state, measuring the event the country is being drawn to show.**

**What it costs, stated plainly.** Placement inside Rakhine is weighted by where people lived in
2023, and the non-enumerated of 2014 lived disproportionately in the northern townships —
Maungdaw, Buthidaung, Rathedaung — which are exactly where the 2023 surface is emptiest. So
those dots sit south of where the people were. Nothing here can fix it: no source publishes the
non-enumerated below state level, a uniform spread would put them in the Arakan mountains
instead, and inventing a northern concentration would be §14.4. The country's own note says to
read it as composition and never as location, and this is the sharpest case of why.

## Both nulls discriminate

§12 asks for the band and the correlation to be measured against a shuffle control, and for the
file to say which is carrying the check. For Myanmar it is both — 15 very uneven units is enough
spread for a band and enough units for a correlation:

| | real join | shuffle control |
|---|---|---|
| **band** (factor of 1.8) | **0** states outside | median **9** of 15 outside; **0** of 2,000 shuffles pass cleanly |
| **correlation** (log–log) | **r = 0.9859** | best of 2,000 shuffles **0.8847**; **0** reach it |

Note the band admits Rakhine's 0.58 without being widened for it — 1/1.8 is 0.556 — so no
exception was needed and none was added.
