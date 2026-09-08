# Nepal — boundaries and placement for the 753 local levels

Built 2026-09-07 alongside `sources/np.md`. Two modules: `np_geo.py` (the units) and
`np_grid.py` (the placement layer).

| | |
|---|---|
| units | OCHA **COD-AB Nepal** (`cod-ab-npl`, v02, `valid_on` 2024-03-14), shapefile bundle, 53.7 MB, one GET from HDX |
| tier | **ADM3 = local level (palika)** — 753 after the carve-out below |
| placement | **Kontur 400 m population hexagons**, `kontur_population_NP_20231101`, 7.1 MB gzipped → 104,129 hexes kept |
| join | **names, scoped to the district** — the census carries no code at all |
| result | **753 / 753 matched, 0 unmatched either way** |

---

## 1. The boundary file has 775 ADM3 and the census has 753, and the difference is parks

Nepal's **national parks and hunting reserves sit outside the local levels**, not inside
them, and COD carries each as its own ADM3 feature — several of them split across the
districts they touch:

```
  Chitawan  x4 (Chitawan, Parsa, Makwanpur, Nawalparasi East)   Khaptad  x4
  Parsa     x3 (Parsa, Bara, Makwanpur)                         Dhorpatan x3
  Koshi Tappu x3 (Sunsari, Saptari, Udayapur)                   Bardiya, Langtang,
                                                                Shivapuri, Shuklaphanta,
                                                                Lumbini Sanskritik
```

22 polygons, **4,569 km²**. They tile with the palikas rather than overlapping them:
4,569 + 143,084 = **147,653 km², exactly COD's own ADM0 area**. So the parks are carved out
of the local levels, the census attributes nobody to them, and **no dot will land in a
national park** — §8.2c's problem (administrative units owning territory nobody lives in)
solved by the boundary file instead of patched afterwards.

## 2. The carve-out is read off the p-code, and that is the check

An ADM3 p-code is `NP` + province + district + a **three-digit unit code whose first digit is
the unit type**. Filtering on it reproduces Nepal's federal structure exactly:

| digit | | count | Nepal actually has |
|---|---|---|---|
| 1 | metropolitan city (mahanagarpalika) | 6 | **6** |
| 2 | sub-metropolitan city (upa-mahanagarpalika) | 11 | **11** |
| 3 | municipality (nagarpalika) | 276 | **276** |
| 4 | rural municipality (gaunpalika) | 460 | **460** |
| 5 | protected area | 22 | — dropped |

**6/11/276/460 is a published fact about Nepal's 2017 federal restructuring that COD's code
attribute has no reason to reproduce unless the codes mean what they appear to mean.** That
makes it evidence rather than a restatement — §12's rule that a shared code is trustworthy
only as far up the hierarchy as it is independently verified, satisfied here at the tier
being used.

**It has to happen BEFORE the name join, and that is the trap in this country.** Four parks
share a name with a palika in the same district — **Shivapuri** in Nuwakot, **Dhorpatan** in
Baglung, **Shuklaphanta** in Kanchanpur and **Lumbini Sanskritik** in Rupandehi. A name-first
join pairs four local levels with a national park, and every total still reconciles: §12's
shape 2 exactly, and it would have put four palikas' dots into empty reserve land.

## 3. The join is on names because there is no code, and it is scoped to the district

`Religion_NPHC_2021.xlsx` has no geographic code of any kind (`sources/np.md` §3), so the
pairing is by name — the position §12 warns most about. Two things make it safe:

**Scoping to the district.** 753 names have real collisions nationally (four Madi's, among
others) and none inside a single district. Both sides are asserted collision-free under the
fold, inside each district, before anything is paired.

**Two derived rules rather than an alias list.**

1. *The unit-type word*, which the census appends and COD does not, in six spellings across
   two languages — `Gaunpalika`, `Nagarpalika`, `Municipality`, `Rural Municipality`, and
   NSO's own misspelling **`Metropolitian City`** / `Sub Metropolitian City`. Stripped
   repeatedly, because a few are doubled (`Madi Rural Municipality Municipality`).
2. *A leading district name*: `Manang Ngisyang Gaunpalika` against COD's `Ngisyang`.
   Stripped when the label starts with its own district's name — **applied symmetrically to
   both sides**, which is why `Gulmi Durbar` in Gulmi district still matches.

**752 of 753 match on that alone.** The last is `Melanchi` against COD's `Melamchi` — an n/m
nasal, one character. It falls through to a **unique edit-distance-1 match among that
district's unclaimed polygons**, which raises if it is ever ambiguous.

Collapsing n and m inside the fold was the alternative and was rejected: that is a
national-scale rule bought to fix one unit, and Cambodia's note (`sources/kh_geo.py`) is
about exactly that trade. A guarded fallback is narrower than a widened key.

**The independent check** is that every one of the 77 districts holds the same number of
units on both sides. The fold never consults the counts, so agreement is evidence that the
districts themselves were not crossed.

## 4. Why Nepal needs a population grid despite having 753 units

This looks like the country where §8.2's trick should apply — fine units make a population
layer unnecessary — and it is the country where it applies least. **The units are fine in
PEOPLE and wild in AREA**, because the 2017 federal map was drawn to equalise population
across the Terai, the middle hills and the Himalaya at once:

```
  Chandragiri (Kathmandu valley)      ~50 km²    ~90,000 people
  Namkha (Humla, on the Tibet border) 2,290 km²   ~2,500 people
```

A factor of 45 in area between two units of comparable rank. An equal share would put a
Himalayan rural municipality's dots evenly across glaciers and ridge lines — and the northern
units are exactly where `Bon` and the highest Buddhist shares are, so the wash would spread
Nepal's most distinctive colour over rock and ice and under-draw the valley and the Terai
where most Nepalis live.

**Both nulls discriminate, hard** (§12 — Benin measured one, Zimbabwe the other, Cambodia
both):

| | |
|---|---|
| national ratio | Kontur 30,859,146 / census 28,925,480 = **1.067** (2023 grid, 2021 census) |
| per-unit ratio | median **1.00**, quartiles **0.84–1.20**, p05 0.53, p95 1.56 |
| BAND | **10** of 753 outside a factor of 3, against a shuffled median of **224** |
| CORRELATION | r = **0.9055** on log populations, best of 2,000 shuffles **0.1350**, 0 reach it |

`MAX_OUTSIDE_BAND` is set to 20 against a measured 10 — headroom for a Kontur re-release, and
nowhere near the 224 that would make the check decoration.

### The ten outliers are two different things, and only one is a boundary effect

**A town smeared into its hinterland.** `Rohini Gaunpalika` reads 4.09 and the municipality
it wraps, `Siddharthanagar`, reads 0.26 — the same people on the other side of a line.
**Pool Rohini with its neighbours and it falls to 1.31**, which is what a boundary effect
looks like. Butwal (0.26), Dharan (0.35), Bhimdatta (0.32), Birendranagar (0.33) and Triyuga
(0.32) are the same shape seen from the town's side: Kontur's built-up model puts a Nepali
town's people further out than the municipal boundary does.

**A block of the Parsa Terai that Kontur simply over-models.** Kalikamai 8.14, Pakaha Mainpur
3.20, Pokhariya 2.43 — and it does **not** pool away: Kalikamai with all five neighbours is
still 2.88, Pakaha Mainpur still 3.18. A contiguous area near the Indian border where the
grid is wrong about the level. Cambodia's Pailin again.

**Neither changes a count** (§8.2). The grid is a within-unit weight: Kalikamai receives
exactly NSO's 23,480 people whatever Kontur thinks, and only the shape inside the unit
survives. What to carry is that dots in those ten units sit on Nepal's least trustworthy
placement surface.

**1.578% of Kontur's people fall outside every local level** and are dropped — the Indian and
Tibetan border overrun, plus whatever the model puts inside the 22 parks.

## 5. Vintage

COD-AB is `valid_on 2024-03-14` against a 2021 census, which §8.1 would normally flag.
**Nepal's 753 local units were created by the 2017 federal restructuring and have not changed
since**, so the 2024 vintage *is* the 2021 geography. The provinces were renamed in 2023
(Province 1 → Koshi, and so on) and both the boundary file and the census workbook use the
new names, so there is nothing to reconcile there either.
