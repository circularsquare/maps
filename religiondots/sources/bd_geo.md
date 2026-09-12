# Bangladesh — upazila boundaries and the placement grid

`sources/bd_geo.py` writes `data/geo/bd/`. `data/` is gitignored, so this file is the record.

| output | what it is |
|---|---|
| `bd_upazilas.gpkg` | the 544 counted upazilas and thanas — `units` |
| `bd_hexes.gpkg` | 145,658 Kontur H3 r8 hexes with `unit` and `pop` — `place` |
| `bd_lookup.csv` | unit → names, census population, Kontur population, hex count |

---

## 1. The join is an identity, for the third time

`BD_GEOG_ADM3_2011_uscb_202107` and `BD_RELIGION_AND_ETHNICITY_2011census_uscb_202107` are two
layers of the **same geodatabase**, both keyed on `GEO_MATCH`.

```
544 polygons   544 counted units   544 matched   0 unmatched
```

No name matching, no code bridge, nothing to verify. This is the third country here to get its
boundaries free with its counts — Ethiopia (§9u), Pakistan (§9t), Bangladesh — and it remains
the single biggest reason these are cheap. Every hard part of the ten countries before them was
the join: China's cost the whole ingest, Ghana had an acronym collision, Kenya had two files
labelled ADM2 at different tiers.

## 2. The vintages match, which Ethiopia's did not

`ADM3_**2011**` against `**2011**census`. Spec §8.1 satisfied outright, no re-cutting anywhere.

Ethiopia is the contrast: a 2007 census re-cut onto 2021 woredas, 418 units carrying a
`USCBCMNT` lineage note, 70 census-era woredas split across two to four modern ones — all of it
USCB's own work and not independently verifiable. **Bangladesh's `USCBCMNT` is empty on all 617
rows**, and `bd.py` asserts that rather than skipping the column for being blank, because an
empty lineage column is the cheapest available evidence that the boundaries are the ones the
counts were published on.

## 3. Inland water is NOT free here, and Ethiopia is why that is worth saying

Ethiopia's file carries **756** ADM3 polygons of which 18 are lakes and parks — Lake T'ana,
Abaya, Chamo, the Gambella reserve — **cut out of** the woredas, so its water solved itself and
`et_geo.py` says so.

**Bangladesh has 544 polygons and 544 counted units.** The upazila cover is complete, which
means the rivers are *inside* it. In the largest delta on earth that is not a detail: the Jamuna
is braided and several kilometres wide, the Padma and lower Meghna wider still. A uniform
placement would put dots mid-channel in all of them.

Two things handle it and **neither is this module**:

1. **`water.py` runs for every country** and subtracts OSM tidal water, which in Bangladesh
   reaches a long way inland — the whole lower Meghna estuary and the Sundarbans creeks. On this
   build it clipped **1,084 of 145,658 placement polygons**, 0.18% of their area, and left 3
   entirely-water polygons whole. Seven units lost over 95% to the sea and are left unclipped —
   char and stilt settlements, which is exactly the case that rule exists for.
2. **Kontur is empty over open channel**, so the river beds take no dots even where the tidal
   layer does not reach.

**The general form: a complete polygon cover is not good news about water.** Spare polygons in
an administrative file usually mean the agency has already cut the lakes out; their absence
means it has not, and the check moves downstream to the placement layer.

## 4. Why Kontur is needed on the most uniform country here

Ethiopia and Pakistan need the hex weight because a few enormous desert units would otherwise
wash half the map in one colour. **Bangladesh is the opposite country** — 544 units averaging
258 km² at a national 1,027 people/km², about as even a counting grid as this map has — and it
needs the weight anyway, because **the units that are not uniform are the ones the country is
worth drawing for**:

| | km² | people/km² | why it matters |
|---|---|---|---|
| Thanchi | 1,079 | **22** | 36.4% Christian |
| Belai Chhari | 1,048 | **27** | 77.5% Buddhist |
| Baghaichhari | 1,617 | **60** | 70.0% Buddhist |
| Alikadam | 786 | 63 | Hill Tracts |
| Mongla | 1,345 | 102 | Sundarbans |
| Koyra | 1,373 | 141 | Sundarbans |
| Shyamnagar | 1,787 | 178 | Sundarbans, the largest unit in the country |
| Dacope | 797 | 191 | **56.5% Hindu**, the most Hindu upazila |

Against 1,027/km² nationally, Thanchi is a **47-fold** difference. Placed uniformly, every
Buddhist and tribal-Christian dot in Bangladesh — the most distinctive geography on the map —
smears evenly across empty forested ridge, and Dacope's Hindus wash out over mangrove.

So the argument here is not "some units are huge and empty" but **"the units that are huge and
empty are the ones carrying the minorities"**, which is a sharper version of the same problem
and easy to miss on a country that looks uniform in aggregate.

It is a **population** weight, not a religion one: nothing measures where Dacope's Hindus sit
inside Dacope, so every node's dots are spread identically.

## 5. The ratio band is derived, never copied

```
Kontur 2023  172,693,354  vs  2011 census  144,043,696   ->  ratio 1.199
band [1.00, 1.55]      per-upazila median 1.15, quartiles 1.05-1.25
```

Twelve years, 144.0M → roughly 171M, so ~1.19 is what the growth predicts and 1.199 is what came
back. Compare:

| | gap | ratio | band |
|---|---|---|---|
| Ethiopia | 2007 → 2023, 16 yr, near-doubling | 1.714 | [1.15, 2.10] |
| Pakistan | 2017 → 2023, 6 yr | 1.138 | [0.90, 1.45] |
| **Bangladesh** | **2011 → 2023, 12 yr** | **1.199** | **[1.00, 1.55]** |

Copying either of the others would be wrong in a different direction each time. §12's rule is to
re-derive per country; this is the third data point for it.

2,729 hexes (0.688%, 1.2M people) have centroids in no upazila — border overrun and the offshore
chars — and are dropped. That figure is *reported rather than assumed small*, because in a delta
a large value would mean the upazila cover has holes, which is the failure mode to watch.

## 6. The limit at the small end, which is new to this map

**A Kontur r8 hex is ~0.80 km². Central Dhaka's thanas are 0.8–3 km².** Bangladesh is the first
country here with counting units smaller than a few placement cells, and the centroid join —
deliberate, because it stops a hex being counted twice — then gives such a thana only the hexes
whose *centres* land inside it.

| thana | area | hexes | cover |
|---|---|---|---|
| Adabor | 2.29 km² | 1 | **35%** |
| Sutrapur | 2.19 km² | 1 | **36%** |
| Kalabagan | 1.32 km² | 1 | 61% |
| Kotwali | 0.76 km² | 1 | 105% |

Four units are under 60% covered — **0.41% of the country** — and their dots crowd into the
covered part. Most of the other small thanas land at 80–165%, which is fine.

**Not corrected, and the reason is that the correction would not be better.** The fallback is an
equal share over the whole polygon, and there is no evidence Kontur's one hex is worse than
that: it is a built-up-area model and central Dhaka is uniformly built up either way. The
displacement is a few hundred metres inside a unit of about 2 km², which is finer than anything
this map claims. `bd_geo.py` prints the table on every build so it stays a known limit rather
than a surprise.

**The general form, for the next country with small units:** the hex weight silently degrades
when the counting unit approaches the placement cell, and it degrades *without any check
failing* — every unit still has hexes, every total is still exact. Measure cover, not presence.

## 7. Re-fetching

```
python sources/bd.py --fetch        # workbook + geodatabase, ~74 MB
python sources/bd_geo.py --fetch    # Kontur BD, 9.7 MB gz -> 30 MB gpkg
python sources/bd_geo.py            # rebuild from what is on disk
```

Kontur: `kontur_population_BD_20231101.gpkg.gz` from the public S3 bucket, no account.
