# Benin — boundaries and placement

Built 2026-09-07 with `sources/bj_geo.py` and `sources/bj_grid.py`. The join narrative and
the Cotonou decision live in `sources/bj.md` §5 and §6; this file is the file-level record.

## What was downloaded

| what | where | size | licence |
|---|---|---|---|
| **COD-AB Benin, shapefile bundle** | HDX `cod-ab-ben`, `ben_admin_boundaries.shp.zip` | 208 KB | OCHA COD, open |
| **Kontur population BJ, 2023-11-01** | `geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_BJ_20231101.gpkg.gz` | 5.8 MB | CC-BY 4.0 |
| geoBoundaries BEN ADM3 | GitHub `wmgeolab/geoBoundaries` @ `9469f09` | 969 KB | **ODbL**, source OpenStreetMap — **downloaded, measured, NOT USED** (`bj.md` §5) |

**The COD bundle is 208 KB and that is not a truncated download.** Benin is small and COD's
Benin geometry is coarse; `zipfile.is_zipfile` passes, the admin2 layer reads 77 features,
and the per-department split is right. Asserting a size floor here would have been the wrong
check — §5a asks for size *and type*, and the type check is the one that carries.

**The bundle has no ADM3.** `ben_admin0/1/2`, `ben_admincapitals`, `ben_adminlines`,
`ben_adminpoints` — that is the whole zip. This is why Cotonou is one polygon.

## COD ADM2 is the census tier exactly

77 polygons for 77 census columns, split 6/9/8/8/6/6/4/1/6/9/5/9 across the twelve
departments, which is the split the twelve booklets print. Both are asserted before the
join, so a boundary file cut to some other description of Benin fails on its own terms
rather than during the join.

`adm2_pcode` runs `BJ0101`..`BJ1209`, and **Cotonou is `BJ0800`** — `00`, not `01`, because
it is the Littoral's only commune. Anything slicing p-codes should not assume the last two
digits start at 01.

Read with `engine="fiona"`, per §12: pyogrio is geopandas' default when installed and is the
engine that has been seen to return zero features from a valid file without raising.

## geoBoundaries BEN ADM3 — measured and rejected

Kept here because the measurement is the useful part, and because the next person to want
Benin arrondissements should not have to redo it.

* 546 features, ODbL, `Boundary Source(s): uMap, Open Street Map`, representative of 2021.
* **Its `shapeName` field has lost every accent to a literal `?`** — `Tangbo-Dj?vi?`,
  `H?kanm?`. Not an encoding problem at the reading end: the bytes in the DBF are `0x3F`,
  and the GeoJSON in the same archive has them too. The mangling is upstream, in
  geoBoundaries. A name join against a French-language source would have to fold `?` to
  "any letter", which is not a fold.
* **There is no parent column**, so assigning 546 arrondissements to 77 communes needs a
  spatial join against a layer from a different vendor.
* **`Nème Arrondissement` is a name that repeats across four communes** — Cotonou,
  Porto-Novo, Parakou and Abomey-Calavi all have numbered arrondissements. §12's plain-place-
  name collision (Serbia's two Palilulas) wearing numerals.
* The Cotonou measurement is in `bj.md` §5. Short version: areas agree to 1.3%, IoU 0.729,
  and the census-vs-Kontur population check comes back at a 0.64–1.98 band with r = 0.81
  where the project's standard is a factor of two around a tight median.

## Placement: Kontur, and Benin needs it for water as much as for emptiness

`sources/bj_grid.py`. 76,525 hexes in, 574 dropped whose centroid falls outside every
commune (71,491 modelled people, 0.518% — the extract overruns into Nigeria, Togo, Burkina
Faso and Niger, and out to sea), **75,951 kept**.

* Every one of the 77 communes gets hexes, 78 to 4,725 each. Asserted: a commune with none draws nothing, silently.
* National ratio Kontur/census **1.372**. Expected to read high — a 2023 modelled grid against a 2013 census, over ten years at roughly 2.7%/yr — so the band is centred on that expectation rather than on 1.0.
* **The lagoons are inside the communes, not cut out of them.** Sô-Ava's polygon is largely Lac Nokoué. A population grid needs no clip for this and `water.py` is not involved.

### The two communes the model fits worst are the two built on water

Per commune, Kontur/census normalised by the national ratio:

```
Aguégués      0.30      islands in the Ouémé delta
Sô-Ava        0.32      Lac Nokoué — contains Ganvié
Toffo         0.50
Torri-Bossito 0.60
...
Toucountouna  2.38
```

**Ganvié is a town of roughly 30,000 people built on stilts over Lac Nokoué**, and Aguégués
is three delta islands. A population surface built from building footprints and night lights
finds least exactly there. This is spec **§8.2c-i** — *SOME PEOPLE LIVE ON THE WATER*,
raised by Anita on 2026-09-05 — answered from the other end: the question there was whether
clipping water would delete such people, and the answer here is that a footprint model
under-weights them by a factor of three before any clipping is considered.

`water.py` independently reached the same two units: *"2 unit(s) lost over 95% to the sea and
are left UNCLIPPED — stilt villages and the like"*. Two unrelated pieces of the pipeline
found the same two communes, which is the best kind of agreement.

The per-unit band is set at a factor of 4 to admit them, and they are named rather than
summarised (§12).

### And the band is not the check, which is worth knowing

Benin's communes are mostly 50,000–250,000 people and look alike, so **shuffling the census
populations across the polygons leaves all but ~11 of 77 inside the same factor-of-4 band**.
A band that a scrambled join mostly passes is not evidence about the join.

So the discriminating statistic is measured instead of assumed: log-log correlation of
census against Kontur population per commune is **r = 0.9052** for the join as built, against
a best of **0.3776** over 500 random pairings (median 0.0805). The assertion is that the real
join beats every shuffle, with an absolute floor of 0.80 as a second net.

**That control is the transferable part.** Every country here that joins by name asserts a
ratio band; none of them had ever checked whether the band could tell a right join from a
wrong one. For a country of uneven units — Serbia, Kenya — it can. For a country of
similar-sized units it cannot, and the band is decoration. **Two lines of shuffling say
which case you are in.**
