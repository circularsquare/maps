# Malawi — boundaries and placement

`sources/mw_geo.py` → `data/geo/mw/mw_districts.gpkg`, `mw_lookup.csv`
`sources/mw_grid.py` → `data/geo/mw/mw_hexes.gpkg`

| | |
|---|---|
| units | **32 districts**, OCHA COD-AB Malawi (`cod-ab-mwi` on HDX) |
| join | by name, 32/32, **one alias needed and none written** |
| check | COD's `adm2_pcode` against Table E5's print position, 32/32 |
| placement | **Kontur 400 m population hexagons**, 75,802 of them |

---

## 1. COD's ADM2 is the census's district tier exactly, cities included

The bundle is `mwi_admin_boundaries.shp.zip`, 54.8 MB, and its ADM2 layer has **32 features
split 7 / 10 / 15 across the three regions** — the same split Table E5 prints.

**That is not the expected outcome and it is worth checking rather than assuming.** Malawi is
usually described as having 28 districts, and a boundary file built to that description would
have no Mzuzu, Lilongwe, Zomba or Blantyre City and would silently absorb **2,115,867 people,
12.0% of the country**, into the four rural districts around them. Every total would still
reconcile. `mw_geo.py` therefore asserts the feature count *and* the per-region split before
it attempts the join, so a wrong-vintage file fails on its own terms rather than during the
name match — and it says so in the error: *"if this is 28 the file predates the four city
districts"*.

The shapefile bundle is taken rather than the geodatabase, on §12's Chile rule: GDAL's
OpenFileGDB driver has been seen to open a `.gdb`, list its layers, report the right CRS and
return zero features without raising.

**HDX's download URL 302s and the redirect must be followed.** The un-redirected request
returns HTTP 200 with a 1,513-byte body, which `is_zipfile` rejects — §5a's rule doing its
job, and the reason the fetch asserts zip-ness rather than status.

## 2. The join, and the check that is worth more than the join

32 census districts, 32 COD polygons, 32 matched, nothing spare on either side.

**One name differs and it is spacing:** NSO writes `Nkhata Bay`, COD writes `Nkhatabay`.
`fold()` strips non-alphanumerics, so it matches without an alias table, and no alias table is
written — a frozen list of renames goes stale in silence at the next release (§12, and the
call `ke_geo.py` made after testing that its own alias table changed nothing).

**The independent check is the p-code, and the two sides derive it in completely different
ways.** On the census side there is no code at all: `sources/mw.py` numbers each district
`MW<region><nn>` from its **position** in the printed table, region by region. On the boundary
side `adm2_pcode` is an attribute, `MW101`..`MW315`. If Table E5's print order is Malawi's own
district-code order — which is the assumption every dot's placement rests on — then matching
by *name* must reproduce the code on all 32.

It does, 32/32. A single transposed or inserted row would break this and **nothing else
would**: every total in `mw.py` reconciles whichever polygon a district is paired with, which
is §9n's `TMA` lesson in a new country.

## 3. Placement is Kontur, and Malawi's reason is water

**Lake Malawi is 29,600 km² and it is not a hole in the country.** The district boundaries run
out into the middle of it: Karonga, Rumphi, Nkhata Bay, Likoma, Salima, Nkhotakota and
Mangochi each own a slab of open lake. Spread dots evenly over those polygons and a large
share of Malawi's people are drawn onto water, in a band down the whole eastern side of the
country — and **Likoma**, a district of 14,527 people whose polygon is almost entirely lake,
would be a wash of dots over nothing at all.

This is §8.2c's inland-water problem in its worst form on this map, and Kontur removes it
rather than patching it: a population grid has no hexes on the lake, so the dots have nowhere
wrong to go. Ethiopia found the same thing (§9u) and `water.py` is not involved in Malawi at
all — `scatter.py` reports *"no ocean polygons over this country, nothing to clip"* and that
is correct rather than a miss.

It fixes the smaller problem too. The four cities are dense specks — Zomba City is 105,013
people — while Mangochi and Kasungu are large with big empty stretches. An equal share per
polygon would misplace both ends.

**The numbers.** Kontur's MW extract is 78,172 hexes carrying 21,129,984 modelled people.
2,370 hexes (336,607 people, 1.59%) have centroids outside every district — the Mozambican and
Zambian border overrun and the Tanzanian side of the lake — and are dropped. 75,802 remain.
Every district gets hexes: **22 for Likoma** (13,024 people against a census 14,527) up to
9,616 for Lilongwe.

**The join is on hex CENTROIDS**, deliberately: a hex on a district line belongs wholly to one
side, so no hex is split and no population is double-counted. The centroid is taken in
EPSG:3857, the CRS Kontur tiled in, and the *points* are reprojected afterwards — reprojecting
the polygons first and taking the centroid after moves it.

**The tolerance band is measured here and not copied from Kenya**, on §9u's rule. Kontur's
vintage is 2023, the census is 2018, and Malawi grew about 2.6%/yr over those five years, so
the grid **should** read high. It does: 20,793,377 against 17,563,749, a ratio of **1.184** —
which is roughly five years of growth plus modelling, and is the right shape rather than a
coincidence that passes. A band centred on 1.0 would have been the wrong check even though
Malawi would have squeaked through it.

Kontur is a model, not a census, and it is used **only as a within-district weight**, so its
level does not matter and its shape does. Nothing measures where Malawi's Anglicans sit inside
a district: an Anglican dot and a Muslim dot are spread identically. Read a cluster as *"this
district, drawn where Malawians live"*, never as sub-district detail.

## 4. What was not used

* **COD-AB ADM3** — 433 Traditional Authorities, and there is no religion table at that tier
  anywhere (`sources/mw.md` §2). The layer is in the same bundle if one ever appears.
* **The `_em` edge-matched twins** in the same bundle. Malawi is drawn alone and shares no
  drawn border, so edge-matching buys nothing here.
* **`cod-ps-mwi`**, UNFPA's 2023 subnational population projections. Not needed: Kontur
  carries its own population, and a projection would have introduced a second vintage into a
  weight that only needs a shape.
