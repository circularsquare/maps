# New Zealand: the record

Drawn 2026-10-04 (agent d9e44929-nz) with 5 nodes. On 2026-10-05 (agent d9e44929-rulings) the
"Other" slot was split by Aotearoa Data Explorer's 17-language SA2 table, which Anita downloaded:
now 4,888,652 people on 32,413 SA1s, 16 nodes, 4,880 dots. Every row is `derived` (a
multi-answer table, spec §3.6).

## Source

Stats NZ, Census 2023, CC BY 4.0. The question is **languages spoken**: "In which language(s)
could you have a conversation about a lot of everyday things?", as many as the person likes. Not
a first language and not a home language: 95% name English. Counts are randomly rounded to base 3
per cell.

Route: Stats NZ Geospatial's ArcGIS Online feature services "2023 Census totals by topic for
individuals by SA1" and "... by SA2", part 1, layer 1 (unclipped), queried for attributes only,
no key (`sources/nz_census.py --fetch`, ~3 MB into `data/raw/nz/`). Fields are opaque
(`VAR_1_205`...) and are picked by their aliases. Two variables, level 1 only:

| variable | categories |
|---|---|
| Languages spoken (total responses) | English, Māori, Samoan, New Zealand Sign Language, Other, None (too young to talk), Not elsewhere included |
| Official language indicator (persons) | 13 classes by which of {Māori, English, NZSL, Other} a person named: English only, Māori and English only, English and Other only, ... , "Other combination" |

### The Other split: Aotearoa Data Explorer at SA2

The 2023 detail at SA2 is in **Aotearoa Data Explorer** (ADE): dataflow `CEN23_ECI_011`
"Languages spoken, ethnicity, and gender ... (RC, TALB, SA2, Health), 2013, 2018, and 2023",
codelist `CL_CEN23_LAN_001`, 20 codes: English, Māori, Samoan, NZ Sign Language, and eleven more
named inside level 1's Other (Northern Chinese, Hindi, Tagalog, Sinitic not further defined, Yue,
French, Panjabi, Afrikaans, Spanish, German, Tongan), then Other, None, Not elsewhere included,
Total stated, Total. `CEN23_ECI_007` and `CEN23_ECI_021`/`023` use the same list, so no ADE table
goes deeper at any geography; the full classification exists only nationally.

ADE's data API needs a subscription key (401 "missing subscription key"; structure queries answer
without one). **Anita downloaded the table by hand on 2026-10-05**:
`data/raw/nz/CEN23_ECI_011_sa2_2023.csv`, 50,560 rows = 2,528 areas x 20 codes, year 2023, Total
ethnicity (9999), Total gender (99), columns CEN23_GEO_002 (area code), Area, CEN23_LAN_001,
Languages spoken, OBS_VALUE, OBS_STATUS. The areas are the 2,395 SA2s (six-digit codes), the NZ
row (999999), and regions, TALBs, Auckland local boards and health areas (two to five digits),
which are dropped. 2,241 cells are confidential (OBS_STATUS `c`, empty value), in 120 small SA2s.

**Checks that it is the right table** (`sources/nz_census.py`, asserted, numbers from the run):

- Dataflow, year, ethnicity and gender are single-valued as above, and its SA2s are exactly the
  SA2 layer's 2,395, both ways.
- Its English, Māori, Samoan, NZSL, None, Total and Total stated equal the ArcGIS SA2 layer's
  **exactly** on every SA2 where both are published (max |diff| 0 on 2,275 to 2,395 SA2s;
  nationally English 4,749,969, Māori 213,861, Samoan 110,595, NZSL 24,657, Total 4,993,920).
  Same census, same random rounding.
- Per SA2, its 11 + Other mentions are at least the layer's level-1 Other (min -6, i.e. rounding;
  median +39). Level 1 counts a person once however many other languages they named, so ADE's
  detail gives 1.169 mentions per level-1 Other response nationally (1,025,208 vs 877,188 on
  the SA2s with all cells published). ADE's own residual Other is 391,425 mentions, 45% of
  level 1's Other.
- Against nz.csv as built: per SA2 (2,077 SA2s with no suppressed SA1), the SA1 mentions nz.csv
  holds sum to ADE's within random rounding (|diff| median 3, 99th 9-12, max 54 English, 18
  Māori); persons drawn per SA2 against ADE's Total stated minus None, median 0, 99th 27.

**The split.** Each SA1's Other persons (after the level-1 sharing below) go to the 11 languages
and the residual Other in proportion to its SA2's mentions of those 12, the same mention scaling
the script uses for Samoan against Other, and every new row is `derived`. Done at SA2 and applied
to each SA2's SA1s, because nothing below SA2 names these languages; the record and `grain` say
so. The 124 SA2s with a confidential split cell or no Other mentions keep their Other whole (20.7
people). Asserted: every SA1's Other persons are kept to 1e-13, and every English, Māori, Samoan
and NZSL row of nz.csv is identical to the 2026-10-04 build (74,224 rows, max diff 0).

Persons drawn after the split: English 4,216,768, Other 186,886, Māori 108,495, Samoan 61,644,
Northern Chinese 52,586, Hindi 37,542, Tagalog 28,543, Sinitic nfd 28,494, Yue 26,468, Panjabi
24,123, French 23,939, Afrikaans 23,217, Spanish 21,657, Tongan 18,943, German 18,111, NZSL 11,234.
Other falls from 490,178 to 186,886.

## Sharing each person (spec §3.6)

The official language indicator is the cross-table that gives the combinations, so the split
over English, Māori, NZSL and "some other language" is exact: a person in "Māori and English
only" gives 1/2 to each, in "Māori, English and Other" 1/3 to each. Two parts are not exact:

- **Class 52, "other combination"** (two or three of Māori, NZSL, Other without English; 972
  people nationally): shared over Māori, NZSL and Other by what each one's mentions leave after
  the named classes, assuming two languages a person.
- **The Other slot**, where a person counts once however many other languages they named, is
  split between Samoan and the rest of Other by their mentions in the SA1.

Result, nationally (mentions in brackets): English 4,213,880 (4,745,991), Other 490,178
(876,906), Māori 108,437 (213,789), Samoan 61,618 (110,796), NZSL 11,229 (24,687). Plain mention
scaling (over the SA1s with no suppressed cell) would have given English 3,933,000, Other
679,000, Māori 172,000, Samoan 82,000, NZSL 20,000: the minority languages are almost always
named beside English, so the exact split matters here.

## Units and placement

SA1 2023 (median 150 people), placed on religiondots' `data/geo/nz/sa1_2023_clipped.geojson`
read-only, unit = the SA1 code, one polygon per unit; weight = the layer's own population
(`VAR_1_3`). SA1s of inland water (LANDWATER 21, 71 units, 6 people) are dropped as religiondots
does.

- **Water SA1s**: 418 SA1s are not in religiondots' land lookup or are inland water (inlets 22,
  ocean 23, and a few 11/31 with no people): 726 people, the gap.
- **Suppressed cells**: under ~6 people Stats NZ suppresses every cell (-999), up to ~30 most
  indicator classes. 620 SA1s have a suppressed cell (421 entirely), 3,396 people; they are drawn
  on their SA2's persons-per-head from the SA2 layer (7 SA1s whose SA2 is suppressed too fall
  back to their SA2's other SA1s, then to NZ). 3,311 drawn (the rest are the too-young share).
- SA1 rather than SA2 because the SA1 layer has the same two variables at 14x the resolution;
  Māori and Samoan are concentrated (South Auckland, the East Cape) and SA1 places them.

## Checks (numbers from the run)

1. **Mentions equal the indicator classes holding them**, per SA1: English |diff| median 3, 99th
   percentile 6, max 12 (rounding); nationally 4,745,991 vs 4,746,252. Māori and NZSL also hold
   class 52 people: their mentions beyond the named classes, 1,326, lie inside 0 to 3 x 972.
2. **SA1s sum to their SA2** (2,279 SA2s with no suppressed cell): every language |diff| median 3,
   99th 9-12, max 15-54; nationally SA1 vs SA2 English 4,749,495 vs 4,749,969, Māori 213,888 vs
   213,879, Samoan 110,814 vs 110,595, NZSL 24,699 vs 24,657, Other 877,389 vs 877,188,
   population 4,994,034 vs 4,993,920. 84 SA2s have no land SA1 (water SA2s, 309 people); every
   lookup SA2 is in the SA2 layer.
3. **Persons drawn + gap = the indicator classes**, per SA1 to 1e-6: 4,885,341 + 104,847 =
   4,990,188. "Not elsewhere included" is zero in 2023 (Stats NZ imputed missing answers).
4. `tools/check_country.py nz`: ok. `scatter.py` (2026-10-05, after the Other split): 4,880 dots
   on 4,769 polygons, 0 rings; 8,652 people (0.18%) under one dot per language nationally draw
   none (the 4-language build: 4,886 dots, 0.05%). Dots per language equal the national persons
   / 1000, rounded down: Mandarin 52, Hindi 37, Tagalog 28, Chinese not named 28, Cantonese 26,
   Punjabi 24, French 23, Afrikaans 23, Spanish 21, Tongan 18, German 18, Other 186.
5. The Other split's own checks are under "The Other split" above.

## Calls

- **Drawn though the question is "languages spoken"**, per the brief (multi-answer: build, share).
- **"Other" on the root `other`**: even after the split it spans many families (Korean, Japanese,
  Dutch, Fijian, Cook Islands Māori, Gujarati, Arabic...), so no narrower node holds it (spec
  §3.2). It is 3.8% of the drawn people (10% before the split) and reads as a washed-out grey.
- **The Other split at SA2, applied to its SA1s.** SA1 stays the unit for the four level-1
  languages; the eleven follow their SA2's mix inside each SA1's Other share. The alternative,
  drawing the whole map at SA2, would throw away the SA1 placement of Māori and Samoan.
- **Northern Chinese on Mandarin, Yue on Cantonese** (no Yue group node exists; hk.txt has
  Cantonese and Sze Yap as siblings, and New Zealand's Yue speakers are mostly Cantonese).
  **Sinitic not further defined on the group `sinotibetan.sinitic`**, drawn as "Chinese, language
  not named", since those people named no variety. Panjabi on Punjabi (spelling). No new nodes:
  all eleven already existed; `tree.d/nz.txt` repeats them bare.
- **Colours not touched.** German (cz.txt, 0.66 0.13 240) and Afrikaans (na.txt, 0.62 0.11 220)
  are close and both are scattered thinly across the same suburbs here; both belong to other
  countries, and another agent is adjusting `tree.d/` colours, so left alone.
- **Exact indicator split rather than mention scaling.** Same as Poland's: the census's own
  combinations where it publishes them, scaling only inside the Other slot.
- **Suppressed SA1s on their SA2's rate**, not dropped: 3,396 people, small either way.
- **Too young to talk is the gap**, 104,847 (spec §3.6: a person who named no language).
- NZSL got a hand-picked pale mauve (`tree.d/nz.txt`) to stand apart from English's near-white.
- Year: 2023 only. The same layers carry 2013 and 2018 (fields `VAR_1_187`...), unused.
