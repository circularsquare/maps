# Mauritius — boundaries and placement

`sources/mu_geo.py` → `data/geo/mu/mu_units.gpkg`, `mu_lookup.csv`
`sources/mu_grid.py` → `data/geo/mu/mu_hexes.gpkg`

| | |
|---|---|
| units | **182 Municipal Council Wards and Village Council Areas**, from **OpenStreetMap** |
| join | by name in three stages: 177 folded-exact, 3 audited fuzzy, 3 structural repairs |
| check | **spatial** — every polygon must lie in the district the census printed it under. 182/182 |
| placement | Kontur 400 m hexagons, 1,870 cells |

---

## 1. This tier exists only in OpenStreetMap

**No humanitarian source has it.** COD-AB Mauritius (`cod-ab-mus`) stops at **ADM1 — 12
districts**, outer islands included, and has no ADM2 at all. geoBoundaries has **ADM0 and ADM1
only**; `MUS/ADM2` and `MUS/ADM3` both 404. Statistics Mauritius publishes religion at ward and
VCA and ships no geography whatsoever.

OSM has all of it, and is unusually complete for a small island state:

* **164 relations at `admin_level=8`** — the Village Council Areas, plus four towns as whole
  units;
* **35 at `admin_level=9`** — the Municipal Council Wards.

All 199 close into valid polygons with shapely alone (`linemerge` → `polygonize`, outer rings
minus inner), **zero failures**, and the level-8 set covers **1,942 km² against Mauritius's
2,040 km²** — the shortfall is coastal generalisation.

**Overpass needs a bbox, not an area lookup.** `area["ISO3166-1"="MU"]` 504s on the main
endpoint; two bboxes — the main island and Rodrigues 600 km east — return in seconds. And §5a
applies to Overpass specifically: **it answers a timeout with HTTP 200 and an HTML error
page**, so the fetch checks the body starts with `{` rather than trusting the status.

## 2. The four town relations are parents and are dropped

`Vacoas-Phoenix` at level 8 contains `Town of Vacoas-Phoenix, Ward 1`…`Ward 6` at level 9.
Keeping both double-counts 27% of the country — the same fact `sources/mu.py` handles on the
table side, arriving independently at the same four towns.

**There are four, not five, and Port Louis is the one missing.** That is not an OSM gap: Port
Louis is a district *and* a town, so its wards hang off `PORT LOUIS DISTRICT-Wholly Urban` one
level up and there is no town row beside them in either the table or OSM. The count is
asserted, so if OSM ever adds a Port Louis town relation the build fails loudly rather than
silently dropping seven wards.

## 3. The join, and the spatial check that is worth more than the join

**Stage 1 — folded exact, 177 units.** Accents, case, punctuation, the `VCA` and `Town of`
noise words, the census's cross-district parenthetical, and its abbreviations: `Riv.` →
`Rivière`, `Vac ` → `Vacoas`, `B-Bassin/R-Hill` → `Beau Bassin/Rose Hill`, `St` → `Saint`, and
Rodrigues's `Region 3 - St. Gabriel` → `Saint Gabriel`.

**Stage 2 — audited fuzzy at 0.86, 3 units.** Every pairing is printed with its ratio so it can
be read rather than trusted:

```
0.947  'Bois Chéri VCA'                        -> 'Bois Chérie VCA'
0.902  'Town of Beau Bassin/Rose Hill-Ward 6-South (North in P/L)'
                                               -> 'Town of Beau-Bassin / Rose Hill, Ward 6'
0.867  'Town of Curepipe-Ward 4-(East in Moka)' -> 'Town of Curepipe, Ward 4, West'
```

**Stage 3 — three structural repairs**, which are facts about the data rather than spelling:

* **Dubreuil.** The census prints one row, `Dubreuil VCA (East in Flacq & West in P/W)`; OSM
  splits the same VCA into `East`, `West` and `Part`. The census row carries no part-word,
  unlike every other split unit, so it is the whole VCA and the three are unioned.
* **Vacoas-Phoenix Wards 5 and 6-West.** **OSM has no polygon for either** — it carries Vacoas
  wards 1, 2, 3, 4 and `Ward 6, East` and stops. A gap in OSM, not a naming difference. The two
  are drawn together on the **remainder of the town polygon** after the five mapped wards are
  removed (17.9 km²), so they are one drawn unit rather than two: 35,664 people, 2.89%, and
  the cost is one internal boundary inside one town. `mu_lookup.csv` maps both `geo_id`s to it
  and `_mu_counts` sums them.
* **Rivière du Poste — and the spatial check is the only reason this was found.** See below.

### The spatial check, and what it caught

D6 has **no code column of any kind**; a unit's district is its position in the printed table.
`sources/mu.py` records that position, and `mu_geo.py` asserts that each matched polygon
actually lies in it, against COD-AB's ADM1. **182 of 182 pass.**

A name join on 183 French place names on a small island is precisely where §12's shape 2 lives
— *a key matches almost everything and pairs some units with the wrong polygon, leaving every
total intact* — and nothing else here would see it.

**It caught Rivière du Poste, which stage 1 had matched cleanly and confidently.** Both sides
split that VCA in two and both call the pieces East and West, so the names paired without a
murmur. **They are not the same split.** The census cuts it on the Grand Port/Savanne district
line; OSM cuts it somewhere else, and *both* OSM pieces sit mostly in Grand Port — `West` is
71.4% Grand Port / 28.6% Savanne, `East` is 99.8% Grand Port. Pairing them by name puts every
Savanne dot in the wrong district, and every reconciliation in the project still passes.

The repair rebuilds the census's own split: union the two OSM pieces back into the whole VCA
and intersect with each row's census district, giving 71.4% / 28.6%.

**The generalisable form: when two sources both split a unit and both name the pieces with
compass points, the names matching is not evidence that the splits are the same.** Check the
geometry against something the split is supposed to respect.

Three district names differ between the two sides and all three are abbreviation rather than a
different place: `R. DU REMPART`/`Riviere du Rempart`, `PLAINES WILHEMS`/`Plaine Wilhems`,
`RODRIGUES`/`Rodriguez Island`.

## 4. Placement, and the first country where Kontur is coarse

Mauritius needs a population grid **less than any country that has used one** — 182 units
averaging 11 km² and 6,800 people. Two things still go wrong without it:

* **The coastal VCAs own their lagoon.** Mauritius is ringed by reef and the VCA boundaries run
  out to it, so a seaside village's polygon is substantially water. 201 of 1,870 placement
  cells still needed clipping by `water.py` at 2.25% of their area.
* **The big rural units are mountain and cane.** Grande Rivière Noire 43.5 km² and Tamarin
  48.0 km² against a median of about 6 km², both largely gorge, forest and estate.

**But Kontur is coarse relative to this country and it is the first time that has bitten** —
see [[reference_kontur_resolution_floor]]. 2,072 hexes for 182 units is about eleven each, and
**three cross-district slivers of 0.09 to 3.6 km² contain no hex centroid at all**.

They are added to the placement layer as **one cell covering their own polygon**, and that is
not cosmetic: **the placement layer is the only geometry `scatter.py` sees**, so a unit absent
from it has nothing to put a dot in and `place_weight`'s equal-share fallback cannot fire —
there is nothing to share. Left out, their 890 people are reported as unplaceable and dropped.
Given the polygon they scatter uniformly inside their own boundary, which at that size is
exact enough. The build raises if any unit **larger than 10 km²** has no hex, because that
would be a broken join rather than a sliver.

205 hexes (37,700 people, 2.9%) have centroids outside every unit — the lagoon, the reef and
the sea between the islands — and are dropped.

**The band was measured, not copied (§9u).** Kontur is 2023, the census 2022, and Mauritius's
population is flat, so the two should be close: **1,262,853 against 1,233,097, ratio 1.024.**
That is a much tighter band than Malawi's 1.184 and the tolerance is set accordingly.

**Rodrigues is checked by name rather than trusted to the national ratio.** Its six regions are
3.5% of the country, so a Kontur extract that quietly omitted the island would still pass any
sane national test. They carry 39,603 modelled people against a census 43,604.

## 5. What was not used

* **COD-AB ADM1** is used, but only as the *check* — it is the 12 districts the spatial test
  asserts against, never a drawn tier.
* **`cod-ps-mus`**, UNFPA's 2018 population projections. Not needed; Kontur carries its own
  population and a projection would add a second vintage to a weight that only needs a shape.
* **OSM `admin_level=10`** — not queried. If Mauritius has a finer civil tier, the census does
  not publish religion at it.
