# Liechtenstein — `sources/li.py`, `sources/li_geo.py`, `taxonomy/li2015.py`

Drawn 2026-09-07. **11 communes, 10 nodes, 36,393 people, 96.73% of the country.**

| | |
|---|---|
| counting geography | **commune, 11 units** — 3,420 people each |
| placement | Kontur H3 r8 hexes, 171 of them, clipped to the commune |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2015 |

The smallest country on this map by a wide margin — the next is Montenegro at seventeen times
the population — and among the finest per head. Anita's own reading was that one region would
have done for a country this size; **the 11 communes cost nothing extra**, because the
boundaries were already on disk.

---

## 1. It was found in Switzerland's catalogue

`ckan.opendata.swiss` carries the **Liechtenstein** Amt für Statistik alongside BFS, so the
single search that solved Switzerland (§9ad) returned this table too.

> **A national open-data portal may index a neighbour**, and a microstate is exactly the kind
> of country a sweep walks past. §11c and §11k between them probed thirty-odd European offices
> and neither mentions Liechtenstein.

`etab.llv.li` is an ordinary open PxWeb v1 endpoint — no key, no login, no wall, English
labels — and it holds **exactly one religion table** in the entire database.

## 2. The table

**`213.001e`** — *Permanent resident population by Reference date, Religion, Citizenship, Sex
and Municipality*.

| dimension | values |
|---|---|
| Reference date | 31.12.2010, **31.12.2015** |
| Religion | 11 categories + total |
| Citizenship | total, Liechtenstein nationals, foreign nationals |
| Sex | total, female, male |
| Municipality | Liechtenstein + **11 communes** |

**2015 is the last one.** There is no later reference date; Liechtenstein's subsequent
population statistics are register-based and carry no religion.

## 3. Two PxWeb traps, and neither announces itself

### The json-stat2 emitter is broken

    {"response": {"format": "json-stat2"}}

returns **HTTP 200**, a well-formed json-stat2 document declaring `size: [1, 12, 1, 1, 12]` —
144 cells — and a `value` array holding **one element**. No error, no status flag, nothing in
the envelope that looks wrong until the payload is checked against the shape the same document
declares.

`json-stat` (v1) and `csv` both answer correctly, so this is one broken serialiser rather than
a wall. **This module reads the CSV.**

> §5a, sharpened: *a 200 with a valid-looking envelope is not a download.* Check the payload
> against the shape the response itself declares — here the document carried its own proof of
> being wrong.

### `filter: "all"` is not honoured

`{"filter": "all", "values": ["*"]}` is accepted with a 200 and silently collapses the
selection. It has to be an explicit item list.

> **PxWeb selection semantics are not portable.** Two servers in two days: BFS drops the
> geography dimension entirely on an empty query (§9ad), and this one ignores the wildcard.
> The returned shape is the only thing worth believing, and both modules assert it.

## 4. The `-` cells are true zeros, and the partition proves it

14 cells are published `-`. That is Kosovo's case (§9w) rather than Lithuania's (§9q), and the
argument is cleaner than either: **there is no disclosure threshold here at all.** Planken
publishes a single person in `Other Christian communities`, and values of 1 and 2 appear
throughout — so a blank cannot be a suppression of something small.

`check()` asserts the partition rather than assuming the reading: the 11 categories sum to each
commune's own total, and the 11 communes sum to the national row category by category, both
with a gap of **zero**.

## 5. The communes are not contiguous

Liechtenstein divides its high alpine pasture among the valley communes as **exclaves**:

| commune | polygons |
|---|---|
| Vaduz | **6** |
| Schaan | 4 |
| Balzers, Planken | 3 |
| Eschen, Gamprin, Triesenberg | 2 |

Seven of the eleven are fragmented. **Nobody lives in the detached pieces** — summer grazing
above 1,500 m — so §8.2's equal share would scatter a third of Vaduz's dots onto an empty
mountainside. This is Switzerland's argument (§9ad) at a twentieth of the scale and sharper,
because the empty part is a *disjoint piece* rather than the thin end of one polygon. Kontur's
171 hexes fix it; the file is 17 KB.

**The boundary join is eleven exact strings.** GISCO LAU 2021 carries all 11 communes with the
names the census uses — no folding, no stemming, no aliases. Liechtenstein is absent from the
EU-27 correspondence *workbook* and present in the boundary *shapefile*, which is the same
distinction Switzerland turns on.

## 6. What it is worth

- **73.4% Roman Catholic, and it is the state church** — Article 37 of the constitution names
  it the Landeskirche, one of the last places in Europe where that is literally so.
  Disestablishment has been debated since 2012 and has not happened.
- **7.0% report no religion, about a fifth of Switzerland's share twenty kilometres away** —
  the lowest in Western Europe here. Two neighbouring Alpine countries asking the same kind of
  question and diverging that far is the most striking thing in the country.
- **The form separates Reformed from Lutheran**, which almost nothing else on this map does:
  two state-recognised Protestant churches, one Swiss-facing (2,365) and one Austrian (447).
- **Islam is 5.9%** — Turkish and Bosnian guest-worker migration, in the industrial communes:
  Eschen 11.4%, Gamprin 8.0%, against Planken 0.2%.
- **The citizenship split is the sharpest thing in the table** and this build does not draw it:
  Catholicism is 84.0% of citizens against 52.6% of foreign residents, Islam 2.2% against
  13.1%. A third of the country holds a foreign passport. `li.py`'s `check()` prints it every
  run.

## 7. Left undone

1. **The citizenship dimension is fetched and not drawn.** It is the most interesting variable
   in the table and would support a two-population treatment like Spain's (§9y) — except that
   here both halves are directly counted, so it would be better than Spain's rather than
   equivalent. Nothing in the schema currently carries "same place, two populations".
2. **2010 is on the same table** and would give a five-year change at commune level for a
   country where 128 people is a category.
3. **There is no Jewish cell on the form**, so `coverage.py` leaves Liechtenstein unlit for
   Judaism. That is correct and worth restating: it is not evidence of absence.
