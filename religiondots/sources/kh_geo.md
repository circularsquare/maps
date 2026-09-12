# Cambodia — boundaries and placement

`sources/kh_geo.py` writes `data/geo/kh/kh_provinces.gpkg` + `kh_lookup.csv`;
`sources/kh_grid.py` writes `kh_hexes.gpkg`.

## What was downloaded

| | |
|---|---|
| boundaries | **OCHA COD-AB Cambodia**, `khm_admin_boundaries.shp.zip`, 2.5 MB from HDX |
| layer | `khm_admin1.shp` — **25 features**, EPSG:4326, `adm1_name` + `adm1_pcode` |
| placement | **Kontur population, KH extract**, `kontur_population_KH_20231101.gpkg.gz`, 5.6 MB → 78,283 hexes |

The **shapefile** bundle and not the geodatabase, on §12's Chile rule — GDAL's OpenFileGDB
driver has been seen to open a `.gdb`, list its layers, report the right CRS and return zero
features while raising nothing. Read with `engine="fiona"`, and the feature count asserted
either way.

The bundle also ships ADM2 (districts) and ADM3 (communes). **Neither is usable**, because
NIS publishes religion at province and nowhere else — see `sources/kh.md` §2, which lists the
five places that were checked before accepting that.

## The join: two independent keys, and neither of them is a name

This is a better position than most countries here, and it is why the join is checked rather
than trusted:

* On the **census** side there is no code at all. `sources/kh.py` numbers each province
  `KH-01`..`KH-25` from its **position** in Tables 2.1.1 and 2.5.1.
* On the **boundary** side `adm1_pcode` runs **`KH01`..`KH25`**, which is NIS's own official
  province numbering.

**The pairing is made on NAMES**, and then the two orderings are compared. They agree on all
twenty-five: Banteay Meanchey printed first and coded `KH01`, Kep printed 23rd and coded
`KH23`, Tbong Khmum printed last and coded `KH25`. That is evidence rather than a tautology —
a printed row order and a code attribute have different origins, and **a single transposed row
would break it while every total in `kh.py` still reconciled**, which is §12's `TMA` lesson.

**The minted id is deliberately `KH-01` and not `KH01`** (§12, Benin). It is a position in a
printed table; the p-code is an official code; they happen to agree, and nothing downstream
should be able to assume that silently.

### Three names disagree, all by Khmer romanisation

| census | COD | what differs |
|---|---|---|
| Otdar Meanchey | Oddar Meanchey | `t` / `d` |
| Siem Reap | Siemreap | word break |
| Tbong Khmum | Tboung Khmum | `ou` / `o` |

So the fold strips non-alphanumerics, collapses `ou` to `o`, and maps `d` to `t`. **That is a
rule and not a hard-coded alias list** (§12 — a frozen list of renames goes stale in silence
at the next release; a rule fails loudly). It is aggressive for a national key and §12 says
so; the guard is that **every folded key is asserted unique on both sides and every match
required to be 1:1**, on a set of only 25. A future rename that collides stops the run rather
than pairing two provinces by luck.

Names are taken from the statistical source and not the boundary file (§12, Chile).

## Placement: Kontur, and the north-east is why

25 provinces for 15.55M people is ~622,000 each and they are wildly uneven — Phnom Penh is
2.28M in 679 km², Mondul Kiri is 92,213 in 14,288 km², Kep is 42,665 in 152. An equal share
per polygon would wash the empty north-east in evenly spaced dots and squash a seventh of the
country into one speck.

**It matters for the same reason it mattered in Zimbabwe: because of what the wash would
say.** Mondul Kiri and Ratanak Kiri are the two provinces whose composition is nothing like
the national one — 21.2% and 23.2% `Other` against 0.5% nationally — and they are also two of
the emptiest. An even spread would paint the map's only substantial non-Buddhist colour
across 25,000 km² of forest where almost nobody lives.

**And the Tonle Sap is inside the provinces.** The lake runs from ~2,700 km² in the dry season
to ~16,000 km² in flood, and the boundaries of Kampong Thom, Kampong Chhnang, Pursat,
Battambang and Siem Reap all run out into it. A population grid has no hexes on open water, so
§8.2c's problem does not arise rather than being patched — Malawi's and Zimbabwe's lesson, and
the reason `water.py` is not involved. The floating villages that really are on the lake keep
their dots, because Kontur models people there.

830 hexes (129,029 people, 0.757%) have centroids outside every province — the extract
overruns into Thailand, Laos and Vietnam — and are dropped. Every one of the 25 provinces gets
hexes, 165 to 7,638 each. Kontur totals 16,906,069 against the census 15,552,211, a ratio of
**1.087**: a 2023 modelled grid against a 2019 census whose universe excludes migrants
abroad, so a little above 1.0 is what it should read.

## Both nulls discriminate here, which is unusual

§12 asks for the band and the correlation to be measured against a shuffle control, and for
the file to say which one is carrying the check. **For Cambodia it is both**, because 25
uneven units is the shape where neither degenerates:

| | real join | shuffle control |
|---|---|---|
| **band** (factor of 1.8) | **1** province outside | median **16** of 25 outside; **0** of 2,000 shuffles reach ≤2 |
| **correlation** (log–log) | **r = 0.9241** | best of 2,000 shuffles **0.7227**; **0** reach it |

Compare Benin (77 similar communes, band useless) and Zimbabwe (10 similar log-populations,
correlation useless). Cambodia has enough units for a correlation null and enough spread for a
band.

## Kontur is 4.6× wrong about Pailin, and it is not the join

**Kontur models 374,607 people in Pailin against a census 75,112**, where every other province
sits between 0.62× and 1.72×. Every way this could have been the map's fault was ruled out:

* **the polygons tile cleanly** — every pairwise intersection is under 1 km² and the sum of
  the 25 areas equals the area of their union to within rounding, so this is not §12's Korea
  trap where overlapping ADM1 polygons hand a `keep="first"` sjoin the wrong unit;
* **COD's own `area_sqkm` matches each polygon's measured area at 1.00 on all 25**, so the
  Pailin polygon is not oversized in the file;
* **Pailin's hexes are inside Pailin's real bounding box** (lon 102.49–102.75, lat
  12.74–13.11, with Pailin town at 102.61/12.85), so they are not Thai border population
  leaking across;
* **the population is spread** — 779 hexes, median 176, and the ten densest hold only 13% of
  the total — so it is not one absurd cell.

So Kontur is simply wrong about Pailin, a small former Khmer Rouge stronghold on the Thai
border whose built-up footprint has grown far faster than its counted population.

**It changes no number on the map, and that is the point of §8.2.** The grid is a *within-unit*
weight: Pailin receives exactly NIS's 75,112 people, distributed across Pailin's hexes in
proportion to each hex's share of Pailin. Kontur's level is divided out and only its shape
inside the province survives. What to carry is that **Pailin's dots sit on the least
trustworthy placement surface in the country** — §12's instruction to name the units the grid
models worst rather than printing a min and a max.

The band is therefore asserted as **"at most 2 provinces outside"** rather than widened to
admit 4.6×. Widening it to pass would have made it decoration everywhere else, which is
exactly what §12 warns about; the shuffle control above is what shows the weaker assertion is
still strong.
