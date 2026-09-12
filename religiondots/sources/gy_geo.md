# Guyana — boundaries and placement

`sources/gy_geo.py` → `data/geo/gy/gy_regions.gpkg` (10 regions),
`data/geo/gy/gy_grid_400m.gpkg` (5,773 Kontur hexes), `data/geo/gy/gy_lookup.csv`.

| | |
|---|---|
| boundaries | geoBoundaries `gbOpen` GUY ADM1, 10 features, ODbL, 2017 vintage |
| placement | Kontur population, H3 r8, `kontur_population_GY_20231101.gpkg.gz`, 499 KB |
| join | **ISO 3166-2:GY** — a published standard, not a name match |
| downloads | two, ~1 MB total |

## 1. The join is a published standard, and that has not happened before here

Every other country in this project has been joined on a name, a code the office publishes,
or a code derived from one. Guyana has **neither a name nor a code**: Table 2.19's columns are
`Region 1` … `Region 10`, and the compendium never prints `Barima-Waini` anywhere in its 66
pages. There is nothing on the census side to match on except the number.

geoBoundaries carries `shapeISO`, and **ISO 3166-2:GY is exactly the ten regions in
region-number order**:

| ISO | region | name |
|---|---:|---|
| `GY-BA` | 1 | Barima-Waini |
| `GY-PM` | 2 | Pomeroon-Supenaam |
| `GY-ES` | 3 | Essequibo Islands-West Demerara |
| `GY-DE` | 4 | Demerara-Mahaica |
| `GY-MA` | 5 | Mahaica-Berbice |
| `GY-EB` | 6 | East Berbice-Corentyne |
| `GY-CU` | 7 | Cuyuni-Mazaruni |
| `GY-PT` | 8 | Potaro-Siparuni |
| `GY-UT` | 9 | Upper Takutu-Upper Essequibo |
| `GY-UD` | 10 | Upper Demerara-Berbice |

So the mapping is a transcription of a standard rather than a guess, and the script asserts
the `shapeISO` set matches exactly and is unique before using it. **Worth carrying forward:
`shapeISO` is present on many geoBoundaries layers and is a code bridge nobody here had
looked for** — where a census numbers its units and a boundary file carries ISO 3166-2, the
join is free and needs no names at all.

## 2. And the boundary file's names are wrong, which is why they are not the key

geoBoundaries spells Region 1 **`Barina-Waini`** — an `n` where the Barima river has an `m`.
`gy_geo.py` prints the disagreement and uses the ISO spelling:

```
    1 region name(s) differ from ISO 3166-2 and the ISO spelling is used:
      geoBoundaries 'Barina-Waini' -> 'Barima-Waini'
```

This is `spec` §12's Chile rule — *take names from the statistical source rather than the
boundary file, because this is not the last boundary set with a bad label in it* — arriving in
the one situation the rule does not cover: **the statistical source publishes no names at
all.** The answer is the published standard's spellings, and the general form of the rule
becomes *prefer a standard to a file, and a file to a guess*.

Had the join been by name it would have failed on exactly one region out of ten and looked
like a vintage problem.

## 3. The independent check is Kontur against the census, per region

The ISO join determines which polygon is Region N. It does not determine how many people a
modelled population surface puts inside that polygon — so comparing the two is genuinely
independent, and it is the whole verification here, because the boundary file carries no
population column and the census carries no code.

```
  reg  region                             area km2    census    kontur  ratio
    1  Barima-Waini                         19,221    27,643    30,741   1.11
    2  Pomeroon-Supenaam                     5,778    46,810    45,336   0.97
    3  Essequibo Islands-West Demerara       3,464   107,785   113,073   1.05
    4  Demerara-Mahaica                      2,061   311,563   334,872   1.07
    5  Mahaica-Berbice                       3,891    49,820    52,367   1.05
    6  East Berbice-Corentyne               37,496   109,652   103,176   0.94
    7  Cuyuni-Mazaruni                      47,648    18,375    25,718   1.40
    8  Potaro-Siparuni                      20,555    11,077    11,861   1.07
    9  Upper Takutu-Upper Essequibo         54,483    24,238    32,460   1.34
   10  Upper Demerara-Berbice               16,129    39,992    41,194   1.03
```

National ratio **1.059** — Kontur models 790,798 against a 2012 census 746,955, which is
eleven years of near-flat population plus modelling. Every region sits between 0.94x and
1.40x. A swapped pair could not do that: putting Region 4's 311,563 people under
Potaro-Siparuni's polygon would produce a ratio near 0.04 and 25 respectively.

This is `sources.md` §9i's North Macedonia check — *assert the relationship, not the
equality* — and §9p's Serbia refinement of naming the units the model does worst rather than
printing a min and a max.

**The two loosest are Cuyuni-Mazaruni (1.40x) and the Rupununi (1.34x)**, and that is the
expected direction: both are interior regions of scattered mining camps and Amerindian
villages, which is where a building-footprint model over-predicts. It moves dots *within*
those two regions and never a count, so it is a caveat about placement rather than a bug — but
it means placement inside Guyana's two most Amerindian regions is the weakest on this country,
which is worth knowing given §8 of `gy.md`.

## 4. Guyana is the country Kontur exists for

More than any other country on this map:

- **Region 4 holds 41.7% of the population on 1.0% of the land** (2,061 km² of 214,969).
- **Region 8 is 11,077 people over 20,555 km²** — 0.54 people per km².
- 90% of Guyanese live on a coastal strip a few kilometres deep; the rest of the country is
  rainforest and savannah.

An equal share per polygon (§8.2's default) would have spread Region 9's 24,238 dots evenly
over 54,483 km² of empty savannah. Because Regions 1, 7, 8 and 9 are the Amerindian interior
and are 34–50% Roman Catholic, **that wash would have been one colour**, and it would have
been the loudest thing on the map — a solid Catholic interior three times the visual area of
the populated coast. Kenya's entry (`sources.md` §9o) makes this argument about Turkana; Guyana
is the sharper case.

5,773 hexes, a median of 246 per region. 169 hexes (2.84%, 25,811 modelled people) fall
outside every region and are dropped and reported — the coastal strip Kontur rounds outwards
past the shoreline. 416 boundary hexes are clipped to their region; the rest are left whole.

Every region gets hexes, which `gy_geo.py` asserts — at r8 over regions this large a region
with no hex centre would mean the join was wrong, not that the grid was too coarse, so the
script raises rather than falling back to the unit's own polygon the way `rs_geo.py` and
`de_grid.py` do.

**Inland water needs no clip**, for Kenya's reason: hexes exist only where people are, so the
Essequibo's islands are present and its water is not. Guyana has a large amount of inland
water and a coastline below sea level, and none of it needed handling.

## 5. Not done

- **COD/HDX was not checked.** geoBoundaries ADM1 has all ten regions with ISO codes and the
  Kontur check passes on all ten, so there was nothing to gain. If a finer tier is ever
  wanted, Guyana's 27 neighbourhood democratic councils are the level below and the census
  publishes no religion for them.
- **Vintage.** geoBoundaries GUY ADM1 is a 2017 file against a 2012 census. Guyana's ten
  regions were created in 1980 and have not changed since, so the vintage gap is nominal —
  but it is a gap, and the Kontur check is what stands in for proving it.
- **The Essequibo.** Regions 1, 2, 7 and 8 lie wholly or partly in the territory Venezuela
  claims. This map draws Guyana's own census on Guyana's own administrative regions, which is
  what it does for every country; nothing here takes a position on the claim.
