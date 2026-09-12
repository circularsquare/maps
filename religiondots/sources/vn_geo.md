# Vietnam — boundaries and placement

`sources/vn_geo.py` -> `data/geo/vn/`. Counts: `sources/vn.md`.

| | |
|---|---|
| units | **63 provinces**, geoBoundaries VNM ADM1, pinned to commit `9469f09` |
| join | GSO administrative code -> ISO 3166-2:VN, hand-built and verified three ways |
| placement | Kontur Population 400 m H3, `kontur_population_VN_20231101` — 231,746 hexes drawn |
| output | `vn_provinces.gpkg`, `vn_grid_400m.gpkg`, `vn_lookup.csv` |

## 1. The vintage expired fifteen months ago

**On 1 July 2025 Vietnam merged its 63 provinces into 34.** spec §8.1 says boundaries must be the
vintage the data was published on, and this is the sharpest case of that rule the project has had:
a current ADM1 does not have a Hà Nam, a Bạc Liêu or a Ninh Thuận at all, and its An Giang is
An Giang plus Kiên Giang. It is not a subtly wrong file, it is a different country.

The pin is `9469f09`, the same commit Guyana, Hungary, Korea and Russia use, and it predates the
merger. `build_units()` asserts **64 features** before reading anything, and the error message
says what a count of 34 would mean.

## 2. The join: two code spaces that look alike and agree nowhere

The census identifies a province by GSO's administrative code — Biểu 7 prints `89. AN GIANG`.
geoBoundaries identifies it by `shapeISO`, which is ISO 3166-2:VN. **Both are two-digit numeric
and they agree on no province at all:**

| | GSO | ISO 3166-2 |
|---|---|---|
| `02` | Hà Giang | — |
| `VN-02` | — | Lào Cai |
| `04` / `VN-04` | Cao Bằng | Cao Bằng *(the only coincidence, and it is one)* |

So a numeric join produces 63 confident, silent, wrong assignments and every unit count and
national total still reconciles. `CODE_TO_ISO` is the bridge, and nothing about it is trusted:

1. **62 of the 63 are re-derived by folded name on every run** — census name against
   `shapeName`, diacritics stripped, `đ` folded — and must agree with the table.
2. **The 63rd is Ho Chi Minh City.** The census writes `Tp Hồ Chí Minh`; geoBoundaries romanises
   it as `Ho Chi Minh`. It is the *only* unmatched census province and `VN-SG` is the *only*
   unclaimed ISO code, so the pairing is forced by elimination rather than chosen — and that is
   asserted, not assumed.
3. **Kontur population against census population, per province.** The bridge decides which
   polygon is which province; it does not decide how many people a modelled surface puts inside
   one. A scrambled bridge pairs Ho Chi Minh City's 7.2M with Bắc Kạn's 294k and the ratios
   scatter over orders of magnitude. §9i's North Macedonia check, and here it is the only
   independent one available.

## 3. Sixty-four features, sixty-three ISO codes — and the duplicate is the fix

geoBoundaries carries **Côn Đảo** as its own polygon. Côn Đảo is not a province: it is a
*district* of Bà Rịa–Vũng Tàu, 200 km offshore, and the file correctly gives it the parent's
`VN-43`. So the duplicate ISO code is not an error, it is the instruction — **dissolving on
`shapeISO` reassembles the province**.

Worth naming because the obvious reading is the wrong one: a feature-count check alone says 64
against an expected 63 and sends you looking for a 64th province that does not exist, or worse,
to `drop_duplicates()`, which would silently discard either the islands or the mainland
depending on row order.

## 4. The census spells its own provinces two ways

Vietnamese admits two tone-mark placements on `oa` / `oe` / `uy`, and GSO uses both. Five names
differ between the two files:

| geoBoundaries | census |
|---|---|
| `Hòa Bình` | `Hoà Bình` |
| `Thanh Hóa` | `Thanh Hoá` |
| `Khánh Hòa` | `Khánh Hoà` |
| `Bà Rịa–Vũng Tàu` | `Bà Rịa Vũng Tàu` *(en-dash vs space)* |
| `Ho Chi Minh` | `Tp Hồ Chí Minh` |

Diacritic folding removes the first four. An exact-string join fails on all five and looks like a
vintage problem rather than an orthography one. **The census spelling is what ships** — §12's
Chile rule, names from the statistical source and not the boundary file.

`shapeName` also carries a **trailing tab** on Hà Nội, which whitespace folding removes and an
exact match does not.

## 5. Placement

Vietnam is 63 provinces for 86 million people — **1.36M per unit**, coarser than Kenya's county —
and the population sits in two deltas with a long thin middle. Equal-share placement inside a
polygon would put as many dots in the Central Highlands forest as in the Red River Delta, and
because the Highlands are the Catholic and Protestant end of the country and the deltas are the
Buddhist and Hòa Hảo end, that wash would be a *colour* rather than just a smear. Kenya's,
Ethiopia's and Guyana's argument, in a country shaped for it.

| | |
|---|---|
| Kontur hexes read | 234,595, 99,019,393 people |
| outside every province | 2,849 hexes (1.21%), 924,965 people — the coastal strip Kontur rounds outwards, and offshore islands geoBoundaries does not carry |
| drawn | 231,746, of which 7,733 clipped at a province boundary |
| provinces with no hexes | 0 |

**Kontur/census ratio: 0.73 to 1.88, national 1.14.** Kontur is a 2023 surface over a 2009
census in a country that grew 12% between them, so a ratio above 1 is what correctness looks
like here. The extremes are the interesting part rather than a problem:

- **lowest** — Hậu Giang 0.73, Quảng Ninh 0.77, Bắc Giang 0.85: provinces that have lost people
  to Ho Chi Minh City and Hà Nội since 2009, so a 2023 surface finds fewer than the census did.
- **highest** — Đắk Nông 1.88, Kon Tum 1.63, Cần Thơ 1.61, Hoà Bình 1.54, Điện Biên 1.44: the
  Central Highlands, which have grown fast, plus the two places a building-footprint model
  over-predicts scattered upland settlement. **Placement WITHIN those five is the weakest on
  this country**, and they are also where the Protestant and Catholic dots are, so it is worth
  knowing.

**Inland water needed no special handling**, for Kenya's reason: hexes exist only where people
are, so the Mekong's channels are absent and its islands are present. `water.py` still clips
0.33% of hex area as sea, and leaves **69 hexes that are over 95% water uncut** — the delta's
stilt and floating settlements, which are exactly the case that rule exists for.

## 6. Access

```
python sources/vn_geo.py --fetch    # 0.7 MB geoBoundaries + 16 MB Kontur
python sources/vn_geo.py            # needs sources/vn.py to have run first
```

Both are open, unauthenticated and CC BY. The Kontur extract is `_VN_` — the per-country file,
not the global grid, which is §9p's Serbia lesson: for a country this size the global r6 grid is
the wrong resolution and the country extract is 16 MB.
