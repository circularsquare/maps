# Kenya — boundaries and placement

Built 2026-09-05 by `sources/ke_geo.py` (counties) and `sources/ke_grid.py` (placement).

| | |
|---|---|
| counties | OCHA **COD-AB Kenya**, `ken_admin1.shp` from HDX, CC BY-IGO, 47 features |
| join | by name, **47 of 47 both ways**, no spares, no ambiguity |
| placement | **Kontur Population 400 m H3 hexagons**, CC BY, 230,139 hexes weighted by population |

---

## 1. The counties are easy, and the interesting part is the check

Kenya's 47 counties were created by the 2010 constitution, took effect in 2013 and have not
changed since, so a 2019 census and a 2023-vintage boundary file are the same geography in
substance whatever the metadata years say. COD-AB has all 47 with `adm1_name` and
`adm1_pcode`; geoBoundaries KEN ADM1 also has 47 and is Public Domain, but carries no code,
which turns out to matter — see below.

**The SHP bundle, not the geodatabase**, on §12's Chile rule: GDAL's OpenFileGDB driver has
been seen to open a `.gdb`, list every layer, report the right CRS and return ZERO features
while raising nothing. The feature count is asserted after the read either way.

Two small traps in the bundle itself, both cheap and both real:

- **The file is `ken_admin1.shp`, not `ken_adm1.shp`.** A glob for `adm1` matches neither
  it nor anything else, and the script's first run died on "expected one adm1 shapefile,
  found []". The same bundle holds `admin0`, `admin2`, `adminlines` and `adminpoints`, so
  the pattern has to be `adm(in)?1\.shp$` and not a substring test.
- **The columns are lowercase here** — `adm1_name`, `adm1_pcode` — where COD ships
  `ADM1_EN` / `ADM1_PCODE` in other countries. Resolve the column name; do not assume it.

## 2. The independent check is the p-code, and it is checking the CENSUS side

§12: a 100% name join is also what a subtly wrong name join looks like. Kenya has no
population column in the boundary file, so the usual cross-check is unavailable — but the
p-code does something better here, because **what it verifies is not the join at all.**

`sources/ke.py` has no codes to work with: Table 2.30 prints county names and nothing else.
So it numbers the counties `001`..`047` **by their row position in the printed table**, on
the assumption that KNBS prints them in county-code order. That assumption is load-bearing —
it is the `geo_id` every downstream join uses — and nothing inside the census establishes it.

COD carries `ADM1_PCODE` as `KE001`..`KE047`. So: match by NAME, then require the matched
polygon's p-code to equal `KE` + the row-position code. It does, on all 47. A single
transposed or inserted row anywhere in the table breaks that and nothing else would — every
total reconciles either way, which is exactly the failure mode `TMA` produced in Ghana
(§9n).

**No hand-resolved names, and that was tested rather than assumed.** KNBS and COD do spell
five counties differently — `TAITA/TAVETA` vs Taita Taveta, `ELGEYO/MARAKWET` vs
Elgeyo-Marakwet, `MURANG'A` vs Murang'a, `THARAKA-NITHI`, `NAIROBI` vs Nairobi City — and
every difference is punctuation or case, which the fold already removes. An alias table was
written for them and then **deleted after checking that removing it changed nothing**: §12
says derive rather than hard-code, a frozen list of five renames goes stale in silence, and
a real rename should fail the build loudly instead of being pre-absorbed.

Names on the drawn layer come from KNBS, not from COD (§12, Chile).

## 3. Placement: the first use of Kontur, and Kenya is why it exists

**Kenya is the country that makes §8.2 load-bearing rather than tidy.** The counts are 47
counties, and the counties are wildly uneven in habitability: Turkana is 68,680 km² with
926,976 people, Marsabit 70,961 km² with 459,785, Wajir 56,686 km² with 781,263. Spread
those uniformly and the northern half of Kenya fills with an even wash of dots over ground
where nobody lives — and because Wajir, Mandera and Garissa are each 97-99% Muslim, that
wash is one colour and the most visually dominant thing on the map. It would not be a
rounding error, it would be the wrong picture.

`sources.md` §5 chose Kontur as the project's population grid long ago — "400m H3, on HDX,
vector hexagons out of the box" — and no country had yet needed it. This is the first.
16.5 MB gzipped, 231,360 hexes, each with its own modelled population.

**The two administrative alternatives were both tried and both are worse:**

- **COD ADM2** (290 constituencies) nests cleanly in the counties by p-code prefix, needs no
  spatial join, and is genuinely better than 47 — but it is still six blobs of even wash
  across Turkana instead of one.
- **Weighting those by sub-county population**, which is what the Philippines does, **does
  not work here.** HDX's `ken_admpop_2019.xlsx` has 345 ADM2 rows against COD's 290
  polygons, because Kenya's administrative **sub-counties** and its **constituencies** are
  different tiers with different names. Matched within county by folded name it is 182 rows
  of 345 and **40.5% of the population unplaced.** *Two files labelled "ADM2" are not two
  files at the same level* — the trap is that both are plausibly "the second tier" and
  neither file says which second tier it means.

Kontur sidesteps the whole question: a population grid needs no administrative alignment.

**Mechanics worth carrying forward.** The join is spatial, on hex CENTROIDS, so a hex on a
county line belongs wholly to one side and no population is split or double-counted. 1,221
hexes (305,197 people, 0.55%) have centroids outside every county — the ocean edge and the
Somali, Ethiopian and Ugandan border overrun — and are dropped and reported. Every one of
the 47 counties gets hexes, between 287 and 19,343 of them, and that is asserted: a county
with none would silently empty.

**Kontur is a MODEL and the census is a count, so the check is a relationship, not an
equality** (§12, North Macedonia). Kontur's Kenya total is 54,969,619 against the census's
47,564,296 — a ratio of 1.156, which is four years of ~2%/yr growth plus modelling
difference. It is used only as a *within-county* weight, so the level is irrelevant and only
the shape matters; the assertion is a 25% band, which would catch a wrong country or a
truncated download and nothing else.

## 4. Inland water solved itself

Ghana needed HydroLAKES subtracted because GSS runs district boundaries across Lake Volta
and uniform placement put 1.29% of the dots on open water (`gh_geo.md` §6). Kenya has Lake
Turkana (7,473 km²) and its share of Lake Victoria and needed **nothing**: Kontur only
carries populated hexes, so the lakes are absent from the placement layer by construction.
Measured after the build, **90 of 47,128 dots (0.19%) fall inside a HydroLAKES polygon**, and
those are mostly the inhabited islands of Lake Victoria — Mfangano, Rusinga — where people
really do live.

So the note in `gh_geo.py` that said "if a second country needs inland water, this is the
code that should move into `water.py`" has not been triggered, and the reason generalises:
**a population grid removes the inland-water problem as a side effect, because water has no
population.** A country placed on administrative polygons may need the clip; a country
placed on Kontur will not.
