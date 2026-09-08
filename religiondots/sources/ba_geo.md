# Bosnia and Herzegovina — boundaries and placement

`sources/ba_geo.py` -> `data/geo/ba/ba_municipalities.gpkg` (142 polygons) and
`data/geo/ba/ba_grid_400m.gpkg` (40,733 Kontur hexes). Counts: `sources/ba.md`.

---

## 1. BiH is the second hole in GISCO, and for Kosovo's reason

§9e established that *"the GISCO LAU file is the boundary answer for most of Europe"*, and
§11c/`xk_geo.md` found the first exception: Kosovo is absent because the EU has no agreed
status for it. **Bosnia is the second, and the cause is different but the effect
identical** — GISCO LAU 2021 covers the EU27 plus the *candidate* countries, and BiH had
not been granted candidate status when that file was cut. Albania, North Macedonia, Serbia,
Montenegro, Switzerland, Norway and Iceland are all in it. BiH is not.

So the fallback is the Kosovo one: **geoBoundaries `BIH ADM3`** for the units, **Kontur**
for the placement grid.

The rule worth carrying: *"boundaries are free in Europe" now has two named exceptions and
they are both in the western Balkans.* Check GISCO membership rather than assuming it.

## 2. THE FILE IS DIRTY IN FOUR WAYS AND REPORTS THE RIGHT UNIT COUNT ANYWAY

**geoBoundaries BIH ADM3 has 142 features. The census has 142 municipalities. They are not
the same 142.** That coincidence is what makes this dangerous: §9s's missing-county trap
(*"geoBoundaries is missing an entire county, and it is the one that mattered"*) presents
from the outside as a clean count match, and a naive fold-join scores 125/142 and looks
like an ordinary name-variation problem.

Each defect below was diagnosed **geometrically** — by centroid and area against the known
location of the municipality — not by the names looking similar. Each is repaired in
`repair()` *before* any matching runs, each with its test asserted beside it, so the join
itself stays a plain name join that either works completely or fails completely.

### 2.1 One polygon is named `Republika Srpska`, and it is Višegrad

An **entity** name in a municipality file. The census has `VIŠEGRAD` with no partner, and
the orphan polygon sits at **43.816 N, 19.301 E** with an area of **470 km²**, against
Višegrad's 43.782/19.293 and 448 km². It is Višegrad, mislabelled.

Renamed, with the centroid asserted inside a small box and the area inside 350–600 km². A
future release that fixes the name, or that puts a genuinely different polygon at that
index, stops the build rather than drawing Višegrad's 10,668 people somewhere else.

### 2.2 `Novi Grad` appears twice, 130 km apart, and one of them is huge

* **Novi Grad** (RS, formerly Bosanski Novi) — 45.023 N, 475 km², 27,115 people.
* **Novi Grad Sarajevo** — 43.862 N, 41 km², **118,553 people, the second largest
  municipality in the country**.

A plain name join takes whichever row pandas sees first and is wrong about one of them.
Getting Novi Grad Sarajevo wrong misplaces 3.4% of Bosnia — and, because both are majority
Muslim, it would not look obviously wrong on the map. Split by latitude, with both the
latitude gap and the ten-fold area difference asserted.

### 2.3 `Kupres` and `Kupres (BiH)` — and the labels read backwards

The two Kupres municipalities, split between the entities in 1995. geoBoundaries's `(BiH)`
suffix reads as "the state" and in fact marks the **Federation** one:

* `Kupres` — **57 km²**, the small RS remnant on the eastern side.
* `Kupres (BiH)` — **572 km²**, the Federation municipality.

`ALIAS` maps the census's `KUPRES - RS` to the small one and `KUPRES - F BiH` to the large
one, and the areas are asserted so a flip cannot pass. A ten-fold difference is not
something a bad guess survives.

`Trnovo (BiH)` / `Trnovo (RS)` is the same pair pattern and here the labels *are* right —
Trnovo FBiH is 319 km² and Trnovo RS 118 km² — so it needs an alias but no repair.

### 2.4 `Kupra na Uni` is a typo for `Krupa na Uni`

Confirmed by centroid (**44.906 N, 16.304 E**), not by the spelling being close — `Kupra`
and `Krupa` differ by a transposition, and Bosnia has a *Bosanska Krupa* 20 km away that a
fuzzy matcher would happily reach for instead. The alias carries the typo verbatim.

## 3. The join, after the repairs

**142 census units, 142 polygons, 142 matched, zero unmatched either way**, with 16 of them
going through `ALIAS`. Both sides' folded keys are asserted unique before the merge, so a
future rename collides loudly rather than merging two municipalities.

The other aliases are ordinary and are the kind §9e predicts for the region: geoBoundaries
translating (`Doboj East` for `DOBOJ-ISTOK`), dropping an administrative prefix
(`Mostar` for `GRAD MOSTAR`, `Brcko District` for `BRČKO`), using a seat name for the
municipality (`Ustiprača` for `NOVO GORAŽDE`, `Foča-Ustikolina` for `FOČA - F BiH`,
`Pale-Prača` for `PALE - F BiH`), or using the compound official name where the census uses
the short one (`Prozor-Rama` for `PROZOR`). Sarajevo's four city municipalities lose the
`SARAJEVO` the census appends: `Centar`, `Stari Grad`, `Novo Sarajevo`, `Novi Grad`.

Total area comes out at **51,475 km² against BiH's 51,197** — 0.5% over, which is
geoBoundaries' generalisation and is the expected sign of a complete cover rather than a
gap.

## 4. Kontur, and the band that had to be measured rather than inherited

**Why a population grid and not §8.2's equal share.** 142 municipalities over 51,200 km² is
360 km² each, and Bosnia is mountains with people in the valleys — the Sava plain, the
Bosna and Vrbas corridors, the Sarajevo basin. Uniform scatter paints the Dinaric karst,
which is close to empty.

`kontur_population_BA_20231101` is 2.9 MB gzipped, 41,334 H3 r8 hexes, 3,328,182 modelled
people. 601 hexes (35,717 people) fall outside the municipality cover — Kontur's usual
outward rounding at the border — and are dropped. 4,908 edge hexes are clipped to their
municipality.

**§9u warns against inheriting a ratio band from another country and this is the sharpest
case of it so far.** §9p's check — that a name join is confirmed by every unit's
*(independent population estimate / census count)* sitting in a tight band — assumes the
two numbers measure the same year. Here they do not: **the counts are 2013 and Kontur's
surface is 2023, and BiH emigrated heavily in between.** The ratio therefore sits at a
**median of 0.94×** and a band centred on 1.0 would be measuring emigration rather than the
join.

So the band is derived from the observed median — **median/3 to median×3, i.e. 0.31–2.81×**
— and what it tests is that the ratio holds *together*. A scrambled join throws units to 5×
and 0.1×; a correct one keeps them in a cluster wherever that cluster sits.

**141 of 142 units are inside. The single outlier is Usora at 3.05×**, a 6,500-person
municipality created in 1998 out of parts of three others, where a modelled surface and a
small unit with a complicated shape disagree. Reported, not asserted away.

**And the 0.94× is itself worth reading as a finding rather than noise**: it is the
independent confirmation that Bosnia has lost population since 2013, which is why
`sources/ba.md` §5 says to read the country for the shape of its religion rather than its
current size.

## 5. What is left

* **Sarajevo is four municipalities and the map treats them as four**, which is right —
  Centar, Stari Grad, Novo Sarajevo and Novi Grad are separately enumerated and separately
  drawn, so the capital is not one polygon holding a tenth of the country the way Zagreb is
  in Croatia.
* **The 2013 census has enumeration areas** — 24,319 of them, per the book's own
  introduction — and no boundary file for them is published. That would be a ~145-people
  tier and is not reachable.
