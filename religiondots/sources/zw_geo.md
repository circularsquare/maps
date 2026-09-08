# Zimbabwe — boundaries and placement

Built 2026-09-07 with `sources/zw_geo.py` and `sources/zw_grid.py`. The reading of the map
is in `sources/zw.md`; this is the file-level record.

## What was downloaded

| what | where | size | licence |
|---|---|---|---|
| **COD-AB Zimbabwe, shapefile bundle** | HDX `cod-ab-zwe`, `zwe_admin_boundaries.shp.zip` | 36.6 MB | OCHA COD, open |
| **Kontur population ZW, 2023-11-01** | `geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_ZW_20231101.gpkg.gz` | ~14 MB gz, 48.3 MB unpacked | CC-BY 4.0 |

The COD bundle carries admin0/1/2/3 plus lines and points — ADM2 is 63 districts and ADM3 is
the wards. **Only ADM1 is usable**, because ZIMSTAT publishes religion at province and
nowhere else (`zw.md` §2). The finer layers are downloaded and unused, which is worth
knowing: the geography is not the constraint here, the table is.

Read with `engine="fiona"` per §12 — pyogrio is geopandas' default when installed and is the
engine that has been seen to open a valid file, report the right CRS, and return zero
features without raising. The feature count is asserted either way.

## The join

Ten provinces, ten polygons, and the names agree **character for character**: no accents, no
transliteration, no abbreviation, no collision. This is the easiest join in the project and
it took no special handling of any kind.

**The independent check is Malawi's and here it holds exactly.** The two sides number the
provinces in completely different ways:

* on the CENSUS side there is no code at all — `sources/zw.py` mints `ZW01`..`ZW10` from the printed row position in Table 2.14;
* on the BOUNDARY side `adm1_pcode` is an attribute and runs **`ZW10`..`ZW19`**.

Bulawayo is printed first and coded `ZW10`; Harare is printed last and coded `ZW19`. The two
orderings agree on all ten, and since they have different origins that agreement is evidence
rather than a tautology. A single transposed row would break it and nothing else would —
every total in `zw.py` reconciles whichever polygon a province is paired with.

**The minted id is deliberately not shaped like the p-code** (`ZW01` against `ZW10`), which
is Benin's lesson: there, the same check failed on six of 77 and an id that looked like a
p-code would have invited the assumption that it was one.

## Placement: Kontur, and this is Kenya's case in its purest form

246,407 hexes in, 706 dropped whose centroid falls outside every province (45,375 modelled
people, 0.271% — the extract overruns into Zambia, Mozambique, Botswana and South Africa,
and across Lake Kariba), **245,701 kept**. Every province gets hexes, 510 to 38,005 each.

**Ten provinces for 15.2M people is the coarsest counting geography on the map, and the
provinces are wildly uneven.** Matabeleland North and South are 129,197 km² between them — a
third of the country, much of it Hwange National Park, the Zambezi escarpment and dry
ranching land — holding 1.59M people. Harare and Bulawayo are 872 km² and 479 km² holding
3.09M. An equal share per polygon would wash the empty west and squash a fifth of the
country into two specks.

**And the wash would say something false.** The two Matabeleland provinces have the least
typical religious mix in Zimbabwe — lowest Apostolic, highest *other Christian*, highest *no
religion* — so spreading them evenly would paint that distinctive combination across a third
of the map's Zimbabwe on ground where almost nobody lives, while under-drawing the
Mashonaland provinces where the people are. §8.2's argument with the loudest available
consequence.

**Lake Kariba is inside the provinces.** At 5,580 km² it is one of the largest reservoirs on
earth, and the Matabeleland North / Mashonaland West boundary runs out into it. A population
grid has no hexes on open water, so §8.2c's problem does not arise rather than being
patched — the same thing Malawi found with Lake Malawi and Ethiopia with the rift lakes, and
the reason `water.py` is not involved. (`scatter.py` confirms it from the other side: *"no
ocean polygons over this country, nothing to clip"* — Zimbabwe is landlocked.)

## Zimbabwe is Benin's lesson with the answer reversed

`bj_grid.py` found that the Kontur/census ratio band every name-joined country asserts
**could not tell a right join from a shuffled one** across Benin's 77 similar-sized communes,
and that the log-log correlation could. Zimbabwe is the opposite case, and both halves were
measured here rather than inherited:

**The band is tight and it discriminates.**

```
Harare                2,427,231   2,214,238   0.83
Bulawayo                665,952     621,401   0.85
Mashonaland West      1,893,584   2,075,595   1.00
Mashonaland East      1,731,173   1,947,322   1.02
Manicaland            2,037,703   2,301,885   1.03
Mashonaland Central   1,384,891   1,582,513   1.04
Matabeleland South      760,345     871,978   1.04
Matabeleland North      827,645     957,511   1.05
Midlands              1,811,905   2,096,234   1.05
Masvingo              1,638,528   1,998,982   1.11
```

0.83× to 1.11×, because the grid's vintage is **one year** off the census rather than
Benin's ten or Malawi's five. The assertion is a factor of 1.6, and the control says it
means something: shuffling the populations across the polygons puts a **median 4 of 10**
provinces outside that band, and only **12 of 2,000 shuffles** pass cleanly. Swapping any two
provinces moves a ratio by an order of magnitude, because Harare is 2.4M people in 872 km²
and Matabeleland North is 828k in 75,025.

**The correlation is the weak check here.** r = 0.9790 for the join as built, against a best
of **0.9758** over 2,000 shuffles — ten log-populations of similar size correlate by luck.
It is kept and reported, and it is explicitly not what the country is checked on.

**The transferable form, and it is the general lesson of the pair:** *measure both, use
whichever the country's own shape makes discriminating, and say in the file which one it
was.* A band is strong where units are uneven and weak where they are alike; a correlation
is the reverse. Neither is a check until the null has been measured, and measuring it is two
lines.
