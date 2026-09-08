# South Korea — boundaries and placement

`sources/kr_geo.py` → `data/geo/kr/kr_sigungu.gpkg` (229 units),
`data/geo/kr/kr_grid_400m.gpkg` (71,478 Kontur hexes), `data/geo/kr/kr_lookup.csv`.

| | |
|---|---|
| boundaries | geoBoundaries `gbOpen` KOR ADM1 (17), ADM2 (228), ADM3 (3,504, for one repair) |
| placement | Kontur population, H3 r8, `kontur_population_KR_20231101.gpkg.gz`, 5.4 MB |
| join | ISO 3166-2:KR for provinces; **romanised name inside a province** for districts |
| result | **229 of 229 matched, 0 unplaced, 0 spare** |

## 1. The province half is free and the district half is not

geoBoundaries KOR **ADM1** carries `shapeISO` = ISO 3166-2:KR — `KR-11` Seoul through `KR-49`
Jeju — so the seventeen provinces bridge to KOSIS's seventeen with no name matching at all.
That is Guyana's trick a second time, and it is now worth checking for on every geoBoundaries
ADM1.

**ADM2 carries no code. `shapeISO` is empty on all 228 features.** And the two sides are in
different scripts: KOSIS writes 종로구, geoBoundaries writes `Jongno-gu`. There is no shared
key of any kind, so the district half is a name join across a transliteration.

## 2. The romanisation is deliberately crude, and that is safe because of where it is used

`romanise()` decomposes Hangul syllables and maps jamo, and does **not** implement Revised
Romanization's inter-syllable assimilation. That is the hard half — 종로 is *Jongno*, not
*Jongro*; 중랑구 is *Jungnang-gu*, not *Jungrang-gu* — and implementing it wrong is worse than
not implementing it, because a wrong rule fails silently on the names it mangles.

Instead the fold absorbs it. Almost every case the assimilation produces is an l/r/n
alternation, so the fold **collapses `l`, `r` and `n` to one symbol**, strips administrative
suffixes and bracketed glosses, and de-doubles letters. `jungrang` and `jungnang` become the
same string without a single rule of Korean phonology being encoded.

**That fold is far too aggressive to be a national key, and it is never used as one.** It is
applied inside one province and every match is required to be 1:1 (`spec` §12, Sri Lanka). The
reason is §3: nationally 동구 and 중구 each name six different districts. A global match on a
fold this loose would pair one metropolis's Jung-gu with another's, and every total would
still reconcile.

**Matching is best-first, not first-come.** Taking each unit's best free polygon in list order
lets an early mediocre match consume a polygon a later unit needs exactly — that is how
영광군 lost `Yeonggwang-gun` to a 0.8-scoring neighbour and was then left choosing between
`Dong-gu` and `Buk-gu`. Scoring every pair and assigning the most confident one at a time
removes the ordering entirely.

**And the bracket is sometimes a translation and sometimes a rename.** `Jung-gu [Central
District]` glosses the name and the bracket is noise; `Michuhol-gu [Nam-gu]` carries **the
name the census uses**, because Incheon's 남구 became 미추홀구 in 2018, three years after this
census. Dropping the bracket loses the only string the two sides share. Both readings are
offered and the 1:1 requirement decides.

## 3. ADM1 is NOT used to assign districts to provinces, and that is the finding

The obvious approach — point-in-polygon of each ADM2 district against the ADM1 provinces —
fails, and then greatest-overlap fails too.

**The ADM1 polygons overlap each other.** Six metropolitan cities are enclaves carved out of
the province around them — Gwangju out of South Jeolla, Daegu out of North Gyeongsang, Busan
out of South Gyeongsang — and geoBoundaries draws the surrounding province *without cutting
the city out*. So a point in Gwangju's Dong-gu is inside both `Gwangju` and `South Jeolla`,
`sjoin` returns two rows, and taking the first hands whole metropolitan cities to the wrong
province. 14 units landed in a neighbour.

**Greatest overlap does not rescue it**, because the two layers are different vintages and
genuinely misaligned: 85 of 228 districts sit less than 90% inside their best province, and
Ganghwa-gun — an Incheon island — comes out 77% inside **Gyeonggi**. A geometric assignment
is only as good as the geometry, and this geometry is not good enough to carry the thing the
whole join rests on.

**So the province assignment is derived from the names instead**, in two passes:

1. A fold that is unique on **both** sides — once in ADM2, once across the whole census — is
   matched globally with no province at all, because there is exactly one candidate and no
   collision is possible. **199 of 229 anchor this way**, and each one tells us its polygon's
   province.
2. What is left is precisely the colliding gu names. Each remaining polygon takes the
   province of the nearest anchored polygon — **constrained by the counts the census already
   gives**, so a province that is full cannot take another's district.

Geometry is used only where names are ambiguous, and names only where they are unique.

**The count constraint is not decoration.** Nearest-anchor alone put Gwangju's Dong-gu and
Buk-gu into South Jeolla, whose districts ring the city and are nearer than Gwangju's own; the
census says Gwangju has five units, so filling short provinces cheapest-first fixes it.

## 4. geoBoundaries is missing an entire county

**영광군, Yeonggwang-gun in South Jeolla, 53,984 people, is absent from ADM2.** No polygon of
that name, nothing covering that ground. 228 polygons against 229 census units, and this is
the one.

It is rebuilt from its parts, which is `spec` §12's Philippines rule: ADM3 carries all eleven
of its eup and myeon, and they are exactly the eleven ADM3 units that fall inside **no** ADM2
polygon. `patch_hole()` takes the loose ADM3 units, requires those eleven to be among them,
dissolves them, and asserts the area — **481 km² against a published 475, 1% out**. Three
independent conditions, so a release that fills the hole or renames a myeon fails loudly
rather than drawing a wrong county.

The hole is patched **before** the join runs, so every count downstream is 229 against 229.

**Two things worth carrying forward.** The missing unit was not the one that looked obvious:
Sejong was the expectation — a special self-governing city that is both a province and its own
single sigungu, the natural candidate for a tier that does not exist — and it has an ADM2
polygon like everything else. Guessing wastes time; the join reports it.

And **one missing unit cascaded into a second, wrong-looking failure**. With Yeonggwang absent
South Jeolla was one polygon short, so the count-constrained assignment handed it Gwangju's
`Buk-gu`, and the visible symptom was **Gwangju** coming up short — 500 km from the actual
defect. Two units unplaced, one cause.

**And it matters more than a missing rural county usually would.** Yeonggwang is where
Sotaesan founded Won Buddhism in 1916, and at 3.91% it is the second most Won Buddhist place
on earth after Iksan. The hole was exactly where the map is most interesting.

## 5. The independent check

The name fold decides which polygon is which district; it does not decide how many people a
modelled surface puts there. So Kontur against the census, per unit, is genuinely independent
— and it is the only check available, since the boundary file has no population column and
the census has no code.

```
national Kontur/census ratio 1.044  (51,204,838 vs 49,052,389)
every unit inside [0.30, 3.5]       0 outside
```

A mispairing inside a province would show up immediately, because Korean city districts differ
in population by an order of magnitude — swapping a 525,000-person Gangnam-gu for a
95,000-person Jung-gu is a ratio of 5.5 against 0.18.

The loosest units are the ones a building-footprint model over-predicts: Hwaseong 2.01x,
Seogwipo 1.91x, Yangju 1.87x, Yongin 1.67x — all fast-growing exurbs built after 2015, which
is a caveat about placement *within* those units and never about a count.

## 6. Placement

71,478 Kontur r8 hexes, a median of about 190 per unit. 2,722 hexes (3.67%, 580,098 modelled
people) fall outside every unit and are dropped and reported — the coastal overrun. 7,332
boundary hexes clipped, 64,146 left whole. Every unit gets hexes, which the script asserts:
at r8 over Korean districts a unit with none would mean the join is wrong, not that the grid
is coarse, so it raises rather than falling back to the unit's own polygon.

Korea needs the grid less than Guyana or Kenya do — the units are small — but Gangwon's
mountain counties are large and nearly empty next to Seoul's gu, and ten island districts lose
over 95% of their area to the sea and are left unclipped by `water.py`.

## 7. Not done

- **Korean-language boundaries.** A file with Hangul names would remove the romanisation
  entirely. SGIS and `data.go.kr` both have one; SGIS needs a key and `data.go.kr` file
  downloads do not, so that is the route to try.
- **The 일반구.** No source found for the 35 general gu of the twelve large cities.
- **Vintage.** ADM2 is 2020 against a 2015 census. The one rename it caused (Nam-gu →
  Michuhol-gu) is handled; nothing else changed at this tier between those years.
