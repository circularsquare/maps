# India — sub-district boundaries

Built by `sources/in_geo.py` into `data/geo/in/in_subdistricts.gpkg`.
**5,988 polygons covering all 1,210,854,977 people, nothing unmatched either way.**

## 1. The source, and why not the other three

**SHRUG (Development Data Lab) open-source polygons, PC11 vintage** — the 2011 Census
sub-districts, carrying the census's own `pc11_s_id` / `pc11_d_id` / `pc11_sd_id`. DDL
stitched them from SEDAC, Bharatmaps, Datameet and the Administrative Atlas of India.

Not downloaded from `devdatalab.org`, which gates the file behind a form. It is mirrored
verbatim as parquet in the **`yashveeeeeeer/india-geodata` GitHub release `census/2011`**,
a plain release asset:

    https://github.com/yashveeeeeeer/india-geodata/releases/download/census%2F2011/
        shrug-subdistrict-pc11.parquet     35MB   the layer
        shrug-village-pc11.parquet        309MB   only for 150 town polygons, see §3
        SubDistricts_2011.parquet          53MB   not used; it is the cross-check, see §2

**Licence: CC0 / CC-BY-NC-SA-4.0 — NON-COMMERCIAL.** The first restricted source on the
map. Fine for this project and worth knowing before anything here is ever sold; recorded in
`sources.md` §6.

The three rejected alternatives, all actually tried:

| candidate | why not |
|---|---|
| **Registrar General's `SubDistricts_2011`** | its codes split units into `(Pt)` parts, so **1,457 census units and 1,428 polygons fail to match and 15.1% of the population drops**. Kept anyway — see §2. |
| **geoBoundaries IND ADM3** | 6,836 units, **2018 vintage**, and **no census codes at all**. A name join across 6,836 Indian transliterations against a 2011 table: §8.1's vintage hazard and §12's "never guess a leftover" at a scale where neither is survivable. |
| **Datameet `2011_Dist.shp`** | districts only, one level too coarse. |

## 2. The join is exact, and it is checked against something that is not itself

`geo_id` is state(2) + district(3) + sub-district(5), which is exactly how `in.py` builds it
and exactly the widths SHRUG uses. **5,965 of 5,988 census units match on the first try**;
4 polygons have no census unit and are dropped (`0100000000`, `1932899998`, `1932999998`,
`2446800000` — strays at the wrong level or with retired codes).

A code join can match on both sides and still be wrong — Poland's `LAU_ID` matched 2,477
against 2,477 and joined **zero** (`spec.md` §12). Two independent checks:

- **Names:** 5,975 of 5,988 agree after folding case and punctuation — **99.8%**.
- **Population, from a different agency's file:** the Registrar General's
  `SubDistricts_2011` carries `Tot_pop` per sub-district. Of the 4,531 units whose codes
  line up with it, **4,501 agree with the census population EXACTLY — 99.3%**. A code join
  that also reproduces a separately published population cannot be agreeing by accident.

The three-way check of §8.1 is run in full: unmatched data, unmatched polygons, **and
matched codes with empty geometry** — zero of the last.

## 3. `Area not under any Sub-district` — 17.4 million people, and what to do with them

The interesting problem, and worth reading before adding any country with a nested urban
geography.

The census emits, in some districts, a unit with sub-district code **`99999`**, named *Area
not under any Sub-district*. 23 of them have no SHRUG polygon, and they are **17,374,936
people, 1.4% of India** — West Bengal 18 units and 16.7M, Tripura 4, Karnataka 1. Among
them is the whole Kolkata metropolitan fringe: **5.0M in North Twenty Four Parganas alone**,
1.6M in Haora, 2.3M in Barddhaman. Some of the densest inhabited ground on earth.

Three things were established before choosing what to do:

1. **There is no leftover geometry to give them.** SHRUG's sub-district polygons **tile
   each district completely**, so `district − union(sub-districts)` is empty to four decimal
   places for all 23. The urban ground is already inside the neighbouring rural polygons.
   The obvious fix does not exist.
2. **Spreading them over the district would be badly wrong, not merely coarse.** These are
   municipal corporations occupying a few hundred km² of districts several thousand km²
   across. Howrah's 1.6M smeared across rural Howrah is a visible falsehood in the part of
   the map where density matters most.
3. **The census says exactly what they are made of.** C-01's TOWN rows inside a `99999`
   unit sum to that unit's population — **100.0% exactly, in all three states** — and they
   are named municipal bodies: Kolkata (M Corp.), Haora, Asansol, Durgapur, Agartala,
   Hubli-Dharwad, BBMP. **All 150 have a SHRUG town polygon.**

So a `99999` unit's geometry is **the union of its own towns**, which is a fact read out of
the source rather than an estimate. The unit keeps its single set of religion counts; only
its shape comes from the towns. That is why `shrug-village-pc11.parquet` is a dependency at
all — 309MB for 146 polygons, taken because the alternative is misplacing the Kolkata
metropolitan fringe. Only the rows needed are read, via a parquet filter.

**These 23 polygons OVERLAP their neighbours, and that is inherent.** SHRUG's sub-districts
already tile the district including the urban ground, so a town-union polygon sits on top of
the rural sub-district that surrounds it. No people are double-counted — each unit scatters
only its own dots — but the same ground is covered twice, so a rural sub-district's dots can
land visually inside the city. Verified after the first build: all 23 units received dots
(17,373 expected), and a spatial join finds 20,089 inside them, the excess being exactly the
neighbours' dots falling on shared ground. The alternative was subtracting the towns from
their neighbours' geometry, which would silently shrink 40-odd sub-districts that the census
says nothing about.

**The free upgrade not taken:** the town rows carry the full eight categories, so these 23
units could be split into 150 town-level units instead of 23 town-shaped ones. Not done
because the count unit should stay the census's own; it is a half-hour's work if the Kolkata
region ever needs the resolution.

## 4. Placement, and India breaking the rule §8.2 is built on

`spec.md` §8.2 places dots by giving each polygon of a fine layer an equal share, justified
by statistical agencies designing fine units to a population target — US tracts ~3,400
people, Australian SA1s ~406, Irish Small Areas ~100 households.

**India has no such layer, and is the first country on the map where the trick does not
apply.** The finer geography that exists is 645,828 villages and 4,135 towns, and those are
natural settlements ranging from ten people to two million — administrative units, not
units engineered to a target. An equal share per village would weight a hamlet like a small
city, and there are a great many hamlets.

So sub-districts are the count layer **and** the placement layer, as in Poland, Romania and
Brazil. The cost, stated plainly:

|  | India | Brazil | Poland |
|---|---|---|---|
| median unit population | **~204,000** | ~38,000 | ~7,500 |
| median unit area | **551 km²** | 1,527 km² | 126 km² |

The population figure is the coarsest count unit on the map. The area figure is finer than
a Brazilian município's, so **at national and state zoom the grain is comparable to a
country already drawn; it is at city zoom that India is blockier than anywhere else.**

**The fix, recorded because it is specific and now unblocked:** `Census_Villages.parquet` in
the same release carries **645,828 village POINTS with `t_pop2011`**, summing to
828,886,066 — India's entire rural population — and the towns file supplies the urban half.
Weighting placement by that would put rural dots on actual settlements instead of spreading
them across a polygon.

`scatter.py` grew a **`place_weight`** hook for the US (spec §8.4) and it is exactly the
right shape: `place_weight(place)` returns a weighter, and `weighter.weights(node, idx,
count)` supplies a weight per placement polygon. So what is missing is not a capability but
two pieces of wiring:

1. a placement layer of villages and towns keyed to their sub-district — `pc11_s_id +
   pc11_d_id + pc11_sd_id` is already on every row of `shrug-village-pc11.parquet`;
2. a weighter returning each settlement's population, joined from `Census_Villages`
   (`t_pop2011`, rural) and C-01's own town rows (urban).

**India does not use `place_weight` today only because its placement layer IS its count
layer** — one polygon per unit, so there is nothing inside a unit to weight. India is the
obvious second customer for that hook after the US, and this is the single biggest
available improvement to how it looks.

## 5. Re-fetch

    python sources/in_geo.py --fetch     # ~397MB, then builds
    python sources/in_geo.py             # build from what is already in data/geo/in/
