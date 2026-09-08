# Saint Vincent — boundaries, and the country where the placement grid was removed

`sources/vc_geo.py` -> `data/geo/vc/vc_eds.gpkg` (221 polygons, serving as both `units` and
`place`). Counts: `sources/vc.md`.

---

## 1. The join is free, and needs no download at all

`GEO_MATCH` keys the counts to the boundaries by construction: **221 table keys, 221 geo keys,
221 matched, 0 unmatched.** Sixth country in the USCB series, sixth exact join. This module
downloads nothing — the boundaries ship inside the geodatabase `sources/vc.py` already fetched.

## 2. The boundaries came out of a coastal engineering report

USCB's Metadata sheet, on the enumeration-district geometry:

> *"Geospatial data for the second-order administrative divisions of Saint Vincent and The
> Grenadines were downloaded from the **Feasibility Study & Environmental Assessment for
> Georgetown Coastal Defense, Figure 3: Census Districts in Saint Vincent**"*

The SVG Statistical Office does not publish its boundaries. The only public copy of the 221
enumeration districts is a **figure in a document about sea defences**, digitised by USCB.

That is the oddest provenance in this project and it is worth knowing, but it is not a reason
to distrust the file: the exact `GEO_MATCH` join and a total area of **384.6 km² against the
country's 389** (98.9%, ordinary generalisation) are what vouch for it. §11h's general point —
*ask who else publishes a country's census* — extends to its boundaries, and the answer can be
a civil engineer.

## 3. KONTUR WAS BUILT, MEASURED AND REMOVED

Every country drawn here since Kenya has been placed on Kontur's 400 m H3 grid, so it was the
default for this one too. **It was built, and it is the wrong tool.** Recorded because the
reason generalises — spec **§8.2e** now carries it as a rule.

**A Kontur r8 hex is about 0.16 km². The median enumeration district here is 0.66 km².** So the
grid is roughly four cells across a typical counting unit, and **43 of the 221 units are
smaller than a single hex**. Measured, 2026-09-07:

| | |
|---|---|
| Kontur hexes over the whole country | **509** (417 after centre-point assignment) |
| populated districts getting **no** hex | **78 of 219 — 36%** |
| per-district Kontur / census ratio | **p10 0.00, median 0.45, p90 2.68** |
| national Kontur / census ratio | 0.88× |

**A weighting absent for a third of the units and scattering over an order of magnitude for the
rest is noise, not a weighting.** Using it would mean two different placement rules applied
essentially at random across one island — Kontur where a hex happened to land, equal shares
where it did not — with no reason to think the first is better than the second.

So placement is **§8.2's uniform-within-unit**, and the units are fine enough for that to be
right rather than merely tolerable: a median district is a few hundred metres across, and at
any zoom this map reaches, uniform inside it is indistinguishable from correct.

**What that costs, stated rather than left to be discovered.** District areas are skewed by a
factor of 6,000 — 0.007 km² to **44 km²** — and the largest are the interior of Saint Vincent,
which is the Soufrière massif and close to uninhabited. Uniform scatter there puts dots on a
volcano. The bound is what makes it acceptable: at 1:1,000 a district of a thousand people
draws one dot, so this is **one or two dots up a mountain**, not a misread distribution.

**The check that should have been run before building the grid, and now is a rule:** divide the
median unit area by 0.16 km². Single digits means don't bother.

## 4. Water

`scatter.py` clipped **90 of the 221 districts** against the coastline, removing 0.56% of their
area — a high proportion of units touched and a tiny proportion of area, which is exactly what
you expect for a small island where almost every district reaches the sea. §8.2c's machinery,
doing its job without needing anything country-specific.

## 5. What is left

* **Nothing on the boundary side.** Counting tier, boundaries and placement are all present,
  matched and checked, from one download.
* **The `PARISH` column is carried into `vc.csv`'s note field** for anyone who wants the six
  parishes rather than the 13 census divisions; neither is drawn.
