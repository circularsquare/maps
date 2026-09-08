# Pakistan — boundaries and placement

`sources/pk_geo.py` → `data/geo/pk/`. 135 district polygons, 364,357 Kontur hexes.

| | |
|---|---|
| boundaries | `PK_GEOG1_ADM3_2017_uscb_202401`, a layer of the **same geodatabase as the counts** |
| join | **identity** — both tables are keyed on `GEO_MATCH`, 155 polygons against 155 rows |
| placement | Kontur `kontur_population_PK_20231101`, H3 r8 (~400 m), 380,718 hexes |
| outputs | `pk_districts.gpkg`, `pk_hexes.gpkg`, `pk_lookup.csv` |

---

## 1. The identity join, for the second time

Same as Ethiopia (`et_geo.md` §1): the geography and the counts are two layers of one file,
keyed identically, produced together. Nothing to match, nothing to transliterate, no match
rate to report — `pk_geo.py` asserts feature *counts* instead, because a match rate would be
meaningless.

This is now twice that the USCB seam has removed the step that cost the most everywhere else
on this map, and it is the argument in `sources.md` §11h for taking the remaining four
countries in the series.

## 2. Twenty districts are dropped and the hole is deliberate

Azad Kashmir's ten districts and Gilgit-Baltistan's ten have **polygons but no religion
data** — PBS did not publish the disputed regions (`pk.md` §2). They are excluded from
`units`, so they receive no dots and draw as empty ground.

`pk_geo.py` prints them grouped by ADM1 rather than dropping them silently, because "20
polygons have no counts" and "the two disputed regions have no counts" are very different
findings and only the second one can be checked against the metadata.

The consequence shows up in the hex join: **16,361 hexes (5.2 million people, 2.16%) have a
centroid in no drawn district**, and most of that is not border overrun — it is Azad Kashmir
and Gilgit-Baltistan, which are inside Pakistan and are deliberately not drawn. Ethiopia's
equivalent number was 0.249%. A reader of the log should not have to guess why one is ten
times the other, so the script says so.

## 3. Kontur, for Balochistan

The same §8.2 argument as Kenya and Ethiopia, and Pakistan supplies the largest empty quarter
of the three:

| | |
|---|---|
| Balochistan's share of Pakistan's land | **~44%** |
| its share of the population | **~6%** |
| its Muslim share | **99.28%** |
| Chagai district | 44,748 km², 226,508 people — under 6/km² |

Spread uniformly, nearly half the map fills with an even wash of one colour over the Makran
and the Kharan desert. 364,357 hexes weighted by hex population instead, joined on centroids
so no hex is split across two districts.

It is a **population** weight and not a religion one: nothing measures where Tharparkar's
Hindus sit inside Tharparkar, so every node's dots are spread identically. Read the map as
*"religion by district, drawn where Pakistanis live"*.

Every one of the 135 drawn districts gets hexes — 75 to 8,829 each — which is the assertion
that matters most, since a district with none would fall back to an equal share over nothing
and silently empty.

## 4. The ratio band is tight here, and that is the point

`et_geo.md` §4 argued that Kenya's ±25% Kontur tolerance **must not be copied**, because
Ethiopia's census is sixteen years older than the grid and a ratio near 1.0 would have meant a
bad download. Pakistan is the same argument in the other direction and confirms the rule:

```
Kontur 2023   236,374,181
census 2017   207,684,626
ratio               1.138          band [0.90, 1.45]
```

Six years, not sixteen, so the expected ratio is close to 1 and the band can be narrow again.
**The tolerance is a statement about the vintage gap and has to be re-derived per country.**

The per-district distribution is the check with teeth, and it is tight:

```
median 1.12     quartiles 0.96–1.24     1–99% 0.56–1.40
```

A scrambled join would pair Lahore's 11.1 million with Chagai's 226,508 and scatter the ratios
over orders of magnitude; §9i's North Macedonia test again. The largest outlier is Torghar at
1.91×, a small and mountainous KP district, and nothing in the tail looks like a mis-assignment.

## 5. Not done

- **`pk_districts.gpkg` is written but not read by `countries.py`.** Placement is on the hex
  layer, so the polygons only feed `pk_geo.py` itself — the same wiring as Kenya, Ethiopia,
  Serbia, Lithuania and China.
- **The ADM4 (tehsil) polygon layer is not extracted**, because the tehsil tier is not drawn
  (`pk.md` §3). It is in the geodatabase if §14.4's ceiling ever moves.
- **`PK_GEOG2_*_2010` is a second geography vintage** (7 ADM1, 147 ADM2) carrying the 2010
  agricultural census. Unused; no religion in it.
- **Inland water is not handled and barely matters here.** Unlike Ethiopia, this file does not
  cut lakes out as separate polygons — but Pakistan's inland water is Manchar, Keenjhar and the
  reservoirs, all small, and hexes exist only where people are, so the Kenya/Guyana reasoning
  applies: the Indus's water is absent because nobody lives on it.
