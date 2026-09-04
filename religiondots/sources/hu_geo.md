# Hungary — boundaries

`sources/hu_geo.py` writes `data/geo/hu/hu_settlements.gpkg` (3,177 polygons) and
`data/geo/hu/hu_lookup.csv`.

## 1. The census is FINER than the boundary file, which is the reverse of the usual problem

§12 says to watch for the capital arriving as one polygon. Hungary has that problem in
mirror image, and it is the more tractable direction:

```
GISCO LAU 2021       3,155 Hungarian units, Budapest as ONE polygon (1,723,836 people)
census WBS003        3,177 settlements,     Budapest as its 23 kerület
```

3,177 = 3,155 − 1 + 23. The census publishes religion for every one of Budapest's districts;
GISCO stops at the city boundary. **This is precisely Zagreb** (`sources/hr_geo.md`), where
the data supported the split, no boundary source was found, and 19.8% of Croatia is one
polygon as a result.

Hungary's districts exist, in **geoBoundaries HUN ADM2** — 198 units, which is the 174 járás
plus the 23 kerület plus one. So Budapest, 17.9% of the country, is drawn at census
resolution. That makes Zagreb the only capital still stuck, and narrows what was described
as "the cheapest capital fix outstanding" to one country.

## 2. The GISCO half needs no derivation and verifies perfectly

`LAU_ID` **is** the five-digit KSH settlement code, with no slicing, no offset and no alias
map — unlike Poland (slice down) and Germany (do not slice down). Printed both ways, as §12
insists:

```
matched                        3,154
census units with no polygon      23   <- exactly Budapest's kerület
polygons with no census unit       1   <- exactly Budapest
names agreeing on matched pairs  3,154 / 3,154
```

**Zero name disagreements over 3,154 units** is the independent verification §12 asks for,
and it is the strongest result of any country in the project — Poland managed 2,476 of
2,477, Romania needed diacritic folding and eight deductions. Nothing here needed either.
The names come from KSH's own `CL_TERUL_GEO5` codelist, so this is the source checked
against the boundary file rather than the boundary file against itself.

## 3. An empty residual is a proof, not a nuisance

KSH's geography codelist holds **3,178** five-digit codes, one more than the census has
settlements. The extra is `13578`, **`Budapest kerületre nem bontható adatai`** — "Budapest
figures that cannot be broken down by district".

It is the residual §12 warns a census usually publishes for units with no geography, and
here it carries **no religion rows at all**. That absence is worth an assertion rather than
a filter, because it is the proof that drawing Budapest as 23 kerület loses nobody: had KSH
put even one person in it, the districts would not sum to the city and the map would be
short by however many. `hu.py` asserts it empty; `hu_geo.py` drops it by name and says so.

Its code is also, exactly, GISCO's `LAU_ID` for Budapest-as-one-polygon. That is not a
coincidence — it is the same KSH code for the same object at a coarser grain — and it is why
`BUDAPEST_LAU` and the dropped residual are the same constant.

## 4. The districts are clipped, and the numbers say why

The two halves are different vintages: GISCO LAU 2021 against geoBoundaries ADM2, which is
2017 and OSM-derived. Their Budapest outlines agree remarkably well but not exactly:

```
GISCO Budapest          526.07 km²
23 districts, raw       526.08 km²   (+0.00%)
23 districts, clipped   522.58 km²   (−0.66%)
```

Raw agreement to two decimal places is better than the vintage gap deserves. But "the same
total area" is not "the same shape", and unclipped the districts overhang the city edge in
places by tens of metres, which would put Budapest dots inside Budaörs and Szentendre. The
intersection with GISCO's Budapest makes the union of the 23 exactly the parent, at the cost
of a 3.5 km² sliver ring at the city boundary that receives no dots. That is the right trade
for a dot map: a dot in the wrong municipality is a visible error, an unfilled 0.66% at the
edge of a polygon is not.

The district join itself is by Roman numeral — geoBoundaries writes `I. kerület`…`XXIII.
kerület`, KSH writes `Budapest 01. ker.`…`Budapest 23. ker.` — which is deterministic over
1–23, and checked by requiring each number exactly once on both sides.

## 5. Licence

geoBoundaries ADM2 for Hungary is **ODbL 1.0** (its ADM1 is CC0; the levels differ, which is
worth checking per level rather than per provider). That is the second non-permissive
boundary licence in the project after SHRUG's CC-BY-NC-SA for India, and unlike SHRUG's it
carries no non-commercial restriction — only attribution and share-alike on the boundaries.
It applies to 23 of 3,177 polygons.

## 6. Vintage

GISCO LAU 2021 against a 2022 census, with 3,154 of 3,154 codes and names agreeing, so §8.1
is satisfied by evidence rather than by argument. Budapest's district boundaries have not
moved since 1950, so geoBoundaries' 2017 vintage is not a risk for the 23 it supplies.
