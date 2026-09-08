# Lithuania — boundaries and placement

Built 2026-09-05 by `sources/lt_geo.py`. One 3 MB download; the boundaries cost nothing.

| | |
|---|---|
| units | 60 municipalities, Eurostat **GISCO LAU 2021** |
| placement | 63,766 **Kontur H3 r8** hexes, `kontur_population_LT_20231101` |
| outputs | `data/geo/lt/lt_municipalities.gpkg`, `data/geo/lt/lt_grid_400m.gpkg` |

---

## 1. The easiest join in the project, and the one fact that makes it so

GISCO LAU 2021 — on disk since North Macedonia, and tabulated for all 34 countries in
`rs_geo.md` §1 — carries exactly 60 Lithuanian polygons. And **its `LAU_ID` is the
savivaldybė code Statistics Lithuania keys its cube on**: `11` Alytaus miesto, `13`
Vilniaus miesto, `41` Vilniaus rajono. `zfill(2)` on both sides and the join is done.

No names, no diacritics (`Šalčininkų`, `Švenčionių`, `Kėdainių` never touched), no
transliteration, no aliases, no collisions, no independent tiebreaker needed. Serbia, the
day before, needed a computed collision-resolver for two municipalities called Palilula and
an alias for `Belgrade`/`Beograd` — for exactly the same job.

**The difference is one fact: the source publishes codes and GISCO publishes the same
codes.** Romania, Ghana and Serbia all publish names only, and every trap in their `_geo.md`
files descends from that. When picking a country, "does the religion table carry a
geographic code" is worth asking early, because it decides whether the join is a dictionary
lookup or a day's work.

## 2. The independent check

GISCO's own `POP_2021` against the census total, as a band and never an equality (§9i):

    national 2,795,680 / 2,810,761 = 0.995x
    per unit: median 0.978, min 0.924 (Visagino sav.), max 1.100 (Klaipėdos r. sav.)
    every unit inside 0.85–1.2x

A 0.18-wide band across 60 units, tighter than Serbia's, which is what a code join buys.

## 3. Placement, and why Lithuania needs it more than Serbia did

Lithuanian savivaldybės average **1,088 km²** — more than twice a Serbian opština — over a
country that is forest, farmland and small towns. They are historical districts rather than
units built to a population target, so §8.2's equal share has nothing to be equal over. And
they have a shape that makes it worse: **six cities are their own municipality sitting as an
enclave inside the rajono municipality named after them**, so the rural unit is a doughnut
whose population is almost all pressed against its inner edge. Spread evenly, its dots would
land in the fields.

Kontur's country extract (`kontur_population_LT_20231101.gpkg.gz`, 3 MB, H3 r8, ~0.74 km²)
gives 64,308 hexes; 542 fall outside the 60 municipalities — the border strip Kontur rounds
outwards, and the Curonian Lagoon — leaving **63,766**, a median of **1,206 per
municipality**, against the 1 polygon §8.2 would have used. No Lithuanian municipality is
small enough to hold no hex centre, so the `de_grid.py` fallback never fires.

## 4. The outliers are a blurred surface, and there is a check that proves it

Seven municipalities sit outside 0.6–1.5× of the census, and they are not scattered:

| | census | Kontur | ratio |
|---|---|---|---|
| Šiaulių **m.** (city) | 100,653 | 48,365 | 0.48× |
| Šiaulių **r.** (ring) | 40,917 | 78,567 | 1.92× |
| Klaipėdos r. (ring) | 56,964 | 92,308 | 1.62× |
| Vilniaus r. (ring) | 96,295 | 149,512 | 1.55× |
| Alytaus r. (ring) | 25,581 | 39,288 | 1.54× |
| Elektrėnų sav. | 23,376 | 10,698 | 0.46× |
| Neringos sav. | 3,609 | 1,694 | 0.47× |

**Every one of the first five is half of a city/ring pair, the city low and the ring high.**
That is what a modelled population surface does to a dense city: Kontur is built from GHSL,
HRSL and building footprints, and Soviet-era apartment districts are denser than a
footprint-based model expects, so population leaks outward across the municipal boundary.

**A wrong join would look different** — it would put one unit's people somewhere unrelated,
and the pair would not close. So `report()` derives the pairs from the names (strip
`miesto`/`rajono`, group on the stem — `rs_geo.py`'s principle of computing rather than
listing) and checks each pair as one unit:

    Alytaus     78,308 / 82,554  1.05x        Vilniaus   652,785 / 607,010  0.93x
    Kauno      391,153 / 349,998  0.89x       Šiaulių    141,570 / 126,932  0.90x
    Klaipėdos  208,972 / 193,294  0.92x       Panevėžio  124,526 / 121,551  0.98x

All six close between 0.89× and 1.05×. The surface is blurred across an internal boundary,
not scrambled — and since the weights are relative **within** a unit, this moves dots inside
a municipality and never a count. **A city/ring split is worth checking as a pair wherever a
country has enclave cities**, which in Europe is most of the post-Soviet ones.

Elektrėnai and Neringa are not pairs and are genuine Kontur underestimates: a 1960s
power-station town and the Curonian Spit, both places whose built form is unusual.
