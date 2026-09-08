# Grenada — 2021 census, via CSO's own preliminary results

`sources/gd.py` -> `data/normalized/gd.csv`. Boundaries: `sources/gd_geo.py`, placement:
`sources/gd_grid.py`. Taxonomy: `taxonomy/gd2021.py`.

**108,279 people on 8 census units drawn as 7, 25 named categories, 92.89% of that universe
drawn.** The most evenly divided country in the Caribbean on this map — its largest religion
is 31.5% — and one of the deepest category lists in the region.

| | |
|---|---|
| counting geography | **8 census units, drawn on 7 parishes** — 15,500 people each; see §3 |
| placement | Kontur H3 r8 hexes, 509 of them, snapped to the coastline |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2021, preliminary results |
| universe | non-institutional population in private dwellings, **99.3% of the country** |

---

## 1. THE DOCUMENT

`2021-National-Housing-Population-Census-Results-Latest-PRELIMINARY.pdf`, from CSO's own
WordPress media library at `stats.gov.gd`. The CARICOM mirror (`sources.md` §11v) serves the
same report under a longer filename and the two files are **byte-identical** — sha256 checked
— so the publisher's is read and the mirror is only how it was found.

**`Table 23. POPULATION BY RELIGON AND PARISH`** is one page, 26 categories by 8 units. (The
missing `I` is CSO's, in the table of contents; the table's own caption spells it correctly.)

**The 2011 National Report is not a substitute.** It is 230 pages and its `Table 2.3.1`
publishes religion **nationally only**, with 21 categories — a shallower list on no geography.
2021 is both newer and finer, which is not the usual way round in this region (compare Saint
Vincent, §11v, where the newer report is seventeen times coarser).

## 2. THE PDF SUBSTITUTES `Ǫ` FOR `Q`

The text layer emits **U+01EA, LATIN CAPITAL LETTER O WITH OGONEK**, where the page shows a Q:
`MARTINIǪUE`, `MARTINǪUE`. It is a font-encoding artefact, it is invisible on the page, and it
appears in the column headings — so any comparison against a hand-typed unit name fails on a
character nobody can see. `_fold()` maps it back; nothing else in the document is affected.

`MORMOM` and `INDEPENDENT BAPTISTE` are **not** artefacts — they are CSO's own spellings, and
they are left alone in `gd.csv` (§12). `taxonomy/gd2021.py` maps the strings the source
prints and says so, so that nobody "fixes" the data and silently unmaps two categories.

## 3. THE COUNTING TIER IS 8 UNITS AND THE MAP DRAWS 7

The census reports the **Town of St. George** — the capital, 2,681 people — apart from the
**Rest of St. George**. *Nobody publishes a boundary for the town:*

- **COD-AB**'s ADM1 is eight polygons: six parishes plus Carriacou and Petite Martinique
  separately. No town.
- **OpenStreetMap** has the six parishes and `Carriacou and Petite Martinique` at
  `admin_level=6`, and for the capital only a **`place=town` node**.

So the two halves are added back together into one St. George.

**What that hides is this country's sharpest number.** The town declined the religion
question at **15.8%** against 9.9% for the rest of the parish and 0.91% in St. Mark — an
urban refusal rate three times some parishes', in the 2,681 people the fold makes invisible.

**`gd.csv` carries both tiers**: the census's own 8 at `geo_level=census_unit` and the 7
drawn ones at `geo_level=parish`. Nothing is lost from the file, only from the map, and if a
town boundary ever appears the split is already there. `_gd_counts` in `countries.py` filters
on `parish` and asserts it finds 7, because reading the other tier would double-count
St. George.

**And Carriacou and Petite Martinique go the other way.** COD keeps them apart (GD01, GD08)
and the census publishes one figure, so `gd_geo.py` dissolves the two polygons. Splitting one
published number between two islands would be inventing a magnitude (§14.4). After both
moves the tiers agree exactly, 7 on 7 — which is also what OSM independently has.

> **Two tiers can disagree in both directions at once, and the fixes are not symmetric.**
> One is a fold on the *counting* side, because the geometry does not exist; the other is a
> dissolve on the *boundary* side, because the count does not exist. Doing either on the
> wrong side would have been wrong: folding Carriacou into Petite Martinique's polygon would
> lose an island, and dissolving St. George's boundary would lose nothing at all — there is
> nothing to dissolve.

## 4. THE TABLE RECONCILES TO THE PERSON

Both directions, every row and every column, exactly:

- the 8 units sum to each row's own `TOTAL` — 27 checks, exact;
- the 26 categories sum to each unit's own `TOTAL` — 9 checks, exact.

**Which Saint Lucia's does not** (`lc.md` §4, out by one to three, because its cells are
independently rounded weighted estimates) **and Cayman's does not** (§9at, one to five).
Grenada publishes a straight count and it adds up.

**An independent table pins the columns.** `Table 17. NON-INSTITUTIONAL POPULATION BY SEX AND
PARISH 2021 and 2011` gives each unit's population elsewhere in the report, and Table 23's
column totals reproduce it in order. All eight figures differ from each other, so this
identifies *which column is which unit* rather than merely showing that the arithmetic closes.
`Table 1` is asserted the same way for the universe ladder.

### The universe

    108,279   non-institutional, in private dwellings   <- Table 23, and this map
        690   institutional population
         52   homeless population
    109,021   total population, Census 2021

**99.3% of Grenada is inside the drawn universe**, the cleanest ladder of any Caribbean
source here — Cayman draws on 96.3% of its census count and Barbados on 81.4% of its own
estimate.

## 5. `NOT STATED` IS 7.1% AND IS NOT AN UNDERCOUNT

7,698 people were counted and declined the question. Marked, not filled (§3.5). Its geography
is eleven-fold — 10.2% of St. George against 0.91% of St. Mark — and finer than the map can
show, for §3's reason.

## 6. THE CATEGORY LIST, WHICH IS THE REASON TO DRAW THIS COUNTRY

Twenty-five named answers. Several are things almost nothing else in the region counts apart:

- **`SPIRITUAL BAPTIST`**, 1.70%, filed at `afrodiasporic.spiritualbaptist` — the node
  Trinidad's census created (`tt2011.py`). CSO offers this *and* `INDEPENDENT BAPTISTE` as
  two separate answers, so merging them would undo a distinction the source drew.
- **`MENNONITE` 0.26% beside `EVANGELICAL` 2.36%** — which is what makes this country the
  control for Saint Lucia's mislabelled row (`lc.md` §2). One form with both cells, in the
  same census round, in the same sea.
- **`PRESBYTERIAN` 0.43%, and it is one parish**: 3.17% of St. Mark against 0.02% of
  Carriacou. The Scottish mission on the west coast, still a congregation-sized signal.
- **`LUTHERAN`, `MORAVIAN`, `BUDDHIST`, `BAHAI`** — 41, 17, 24 and 15 people, each with a cell.
- **`ATHEIST` apart from `NO RELIGIOUS AFFILIATION`**: 0.05% against 5.95%, a **130-fold gap**,
  the widest this map has from a source that offered both boxes. Saint Lucia's form asks the
  same pair and gets 47-fold.

`CHURCH OF GOD` (3.60%) is filed at the **Holiness parent**, not a child: the name cannot
decide between the Cleveland (Pentecostal) and Anderson (Holiness) lines, and — unlike Cayman,
where the local body is identifiable — nothing external names which Grenada means. That is
`bb2010.py`'s call for the identical situation.

## 7. THE GRID

Kontur 124,520 against a census 109,021 — **ratio 1.14**, and both parts of the gap are
expected: the extract is the 2023-11-01 vintage against an April 2021 census, and Kontur
models everyone where the table holds only private dwellings. Per parish it runs **1.07x to
1.48x**, far tighter than Saint Lucia's 0.58–1.54.

It is here for two opposite reasons:

- **St. George holds 41% of the country on 65.8 km²**, its people on the south-west coast
  from the town through Grand Anse to Point Salines, its north-east a ridge. Uniform scatter
  would put a fifth of Grenada on a mountainside.
- **Carriacou and Petite Martinique are one unit on two islands 2.4 km apart.** Uniform
  scatter inside a multipolygon spreads by area, which would put far too many people on
  Petite Martinique's 2.4 km².

10.5% of the grid's population lands outside every parish before snapping, all of it within
1 km of a coastline — COD's outline against a 400 m hex. Four hexes holding ten people
between them stay outside and are dropped.

## 8. WHAT THE COUNTRY SHOWS

- **No religion reaches a third of Grenada.** Roman Catholic 31.53%, Pentecostal 19.93%,
  Seventh Day Adventist 12.32%, Anglican 7.31% — four traditions over 7%, which no other
  Caribbean country here does. France and Britain each left a church and the twentieth-century
  missions landed on top of both.
- **The Adventists have the north and the Pentecostals the south.** SDA is 24.09% of
  St. Andrew and 22.60% of St. Mark against 7.34% of St. George; Pentecostals are 25.88% of
  St. David and 23.03% of St. Andrew against 7.23% of Carriacou.
- **Carriacou is a different island religiously too.** **22.08% Anglican** — seven times
  St. David's 3.03% — and 40.55% Catholic, with Pentecostals at a third of the national
  share. Three centuries of Scottish and English settlement in the Grenadines.
- **`CHURCH OF GOD` is one parish**: 8.48% of St. Andrew against 0.66% of St. Mark.
- **`EVANGELICAL` is another**: 6.08% of St. Patrick against 0.61% of St. Mark.
- **Rastafari is 1.07%**, against 1.08% in both Jamaica and Saint Vincent — a third
  independently designed census landing on the same number.

## 9. §14, briefly

Grenada's census asks religion of everyone and publishes it by parish; there is no
minority-protection question here that §14 needs weighed. The one §14.4 point is in §3: the
Town of St. George's own figures are published and are *not* redistributed or modelled onto
a boundary that does not exist — they are added back to their parish, which is the only move
that invents nothing.
