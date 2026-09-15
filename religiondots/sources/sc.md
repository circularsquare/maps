# Seychelles (`sc`)

Drawn 2026-09-14 by `d743fc47-sc`. **Census 2010, 26 districts, 90,945 people, 56 labels on
34 nodes; 4,328 not stated (4.76%) not drawn.** Files: `sources/sc.py`, `sources/sc_geo.py`,
`sources/sc_grid.py`, `taxonomy/sc2010.py`, `countries/sc.py`, `other.sc` in
`taxonomy/branches.py`. Record in `sources.md §sc-2026-09-14`.

## 1. The source, and how it was found

`queue.md` carried Seychelles as *"§11w: not chased"*, priced off UNSD table 28's 2002 row
(11 categories, 81,755, national only). The oracle has nothing later.

- **NBS 2022 census report**, `nbs.gov.sc/downloads/1555-seychelles-population-and-housing-census-2022/download`
  (10,525,457 bytes, 301 pages). Table B4.1 (page 118): religion by region and district,
  2022, all households; Table B4.2 the same for conventional households. **Eight columns**:
  Catholic, Anglican, Islam, Hindu, Christian (other), Unable to classify, Other (Specify),
  Missing. 27 districts (Ile Perseverance is new) plus the individual islands inside `La Digue
  & Inner Islands` and `Outer Islands`.
- **NBS 2010 census supplement**, *Population and Housing Census 2010: Supplement Statistical
  Tables (ALL DISTRICTS)*, from the IHSN catalogue, `catalog.ihsn.org/catalog/4079/download/55082`
  (17,533,419 bytes, 313 pages). Thirteen tables repeated per district; **Table 3 is
  population by religion and sex, with every post-coded write-in, 57 labels**. This is drawn.
- **NBS 2010 census report**, `catalog.ihsn.org/catalog/4079/download/55081` (5,093,475
  bytes, 227 pages): Table 2.9 national religion (page 37), Table 2.3 population and area by
  district (page 28) with its island footnotes (page 29), the form (Annex 2, page 215).
- nbs.gov.sc's download category `36-data-acquisition-census` lists only 2022 volumes
  (checked 2026-09-14); the 2010 volumes are not linked there any more.
- Not checked: whether NBS has a 2002 district table (the 2002 form had seven boxes and two
  catch-alls, so it would be shallower anyway), and 2022 microdata.

## 2. Why 2010 and not 2022

2022 is newer and has one more district. It is not drawn because:

- it folds religion to six answers; `Other (Specify)` holds the Baha'is, the Buddhists and
  **no religion** in one cell (the report's own note to Table 3.2), which the no-religion rule
  would have to draw as `unknown` or not at all;
- **11,772 of 102,612 (11.5%) are `Missing`**, because institutional households got a short
  form without the question (report §3.2);
- NBS says in the same report that the census missed a significant part of the population
  (mid-2022 estimate 119,878 against 102,612 counted).

2010 has 57 labels and 4.76% not stated. `sources/sc.py::check_2022` uses B4.1 as a witness:
across the 23 districts whose boundaries did not change, the district shares of those who
answered correlate 2010 against 2022 at **Anglican rho 0.97, Catholic 0.71, Islam 0.67, Hindu
0.63**, all p ≤ 0.001 on 2,000 permutations. The pattern held; the levels moved (Hindu 2.4%
to 5.4%, Islam 1.6% to 2.4%, per the 2022 report's Table 3.2).

**What would reverse it:** a 2022 district table at the write-in level (the 2010 supplement
format) or the 2022 microdata. The 2022 questionnaire (report page 264) had separate boxes
for Jehovah's Witness, Baha'i, Adventist, Atheist and No religion, so the detail exists.

## 3. The parse and its checks

`sources/sc.py`, PDF text layer, one cell per line; a long label wraps to two lines.

1. Every row: female + male = total.
2. Every district: rows = Total row = **Table 1 (age) total** on the facing page, an
   independent table.
3. Sum of districts = **Table 2.9** in all 32 rows, after four folds the national table
   makes: `Jehovah Witness` is `Jehovah's Witness`; `Christian Community Fellowship` 241 =
   CCF 229 + `Christian Life Fellowship` 12; `Other Christian` 112 = 27 + twelve small
   Christian write-ins (85); `Other non-Christian` 108 = `Other` 30 + twelve small
   non-Christian write-ins (78). Each fold is the only subset that closes; all asserted.
   Table 2.9 is itself parsed off the page and asserted equal to the transcription.
4. Table 2.3 agrees with the supplement on 25 districts. **Its Other Islands row is wrong**:
   576 printed, 1,042 in the supplement, and Table 2.3's rows sum to 90,479 against its printed
   90,945, short by exactly 466. Asserted as that one misprint.

Quirks: Beau Vallon's Table 3 caption says `marital status` (the anchor is the header, not
the caption); `'7th Day Adventist` carries a stray apostrophe (stripped); Table 2 is printed
above Table 1 on some pages (the age total is anchored on `65+`).

## 4. Mapping calls (`taxonomy/sc2010.py` REVIEW has the full reasoning)

- `Roman Catholic` and `Latin Catholic` -> `christianity.catholic.latin` (the label says Roman).
- `Pentecostal Assembly` 1,333 and `Assembly of God` 831 -> `christianity.pentecostal.trinitarian`.
- `Born Again Christian` 605 -> `christianity.evangelical` (Ireland put the same label on
  `christianity.protestant`; argued in REVIEW).
- `Redeemed Christian Church` 394, `Deeper Life` 50, `New Testament Church` 27 -> `christianity.pentecostal`.
- `Christian Community Fellowship` 229 and `Nazarite Christian` 109 -> `christianity.other`:
  neither identified. A search summary names a Nazarite church among Seychelles' small
  Christian groups in State Department reports; the report pages refused the fetch.
- `End-Time-Bride Tabernacle`, `End Time Message`, `Peniel Tabernacle` -> `christianity.other`
  (Branham Message vocabulary; not Oneness Pentecostal denominations).
- `Grace and Peace` 52 -> `christianity.baptist`: the Baptist Union of Southern Africa lists
  `Grace and Peace Baptist Church (Seychelles)` as a member.
- `Tamil` 12 and `Padayachi` 2 -> `hinduism.tamil` (Mauritius's node); `Sai Baba devotee` -> `hinduism`.
- `Atheist`, `Agnostic` -> `secular`; `No Religion` (a printed box, no traditional-religion
  option) -> `unaffiliated`; `Pantheist` -> `esoteric`; `Other` and `Meditate` -> new `other.sc`.
- `Not stated` excluded, 4,328.

New node: `other.sc` only (34 people). No new top-level root, no legend row beyond it.

## 5. Geography (`sources/sc_geo.py`)

OCHA COD-AB `cod-ab-syc`, `syc_admbnda_adm3_nbs2010` (HDX: NBS's 2010 census districts, with
`Other Islands` added from GAUL); geoBoundaries SYC ADM3 is a copy. 27 features for 26
census districts. Joined on name **and** on the ISO 3166-2 number embedded in the pcode
(`SC1116ER` = SC-16); both keys must agree. No twins.

Two moves, both proved by the 2010 report's printed district areas (Table 2.3):

- **Perseverance Island -> English River.** COD draws the reclaimed island as its own feature
  (a district since the 2022 census). English River measures 1.38 km2 alone against 2.3
  printed (-40%) and **2.32 km2 with the island**; every other Mahé district is within 10%.
- **Silhouette, North, Félicité, Marianne, Grande and Petite Soeur, Cocos -> La Digue.** The
  report's footnote 6 lists them under `La Digue (& Inner Islands)`, and the supplement's La
  Digue has the same 2,761 people. COD's La Digue alone is 9.8 km2; with the 13 parts moved
  it is **36.37 km2 against 36.4 printed**. Frégate, Denis and Bird stay in Other Islands.

Not asserted: Grand Anse Praslin (15.4 km2 against 22.3 printed; the two Praslin districts
print 50.8 km2 for about 42 km2 of land, so the print is wrong) and Other Islands (211 km2
against 179.6; atoll outlines).

## 6. Placement (`sources/sc_grid.py`)

Kontur 2023 (489 populated hexes, 106,309 people) **cut by overlap** with the districts, not
assigned by centroid, because central Victoria's districts are 1.2-1.7 km2 against a 0.74 km2
hex. A coastal hex's population goes wholly to its land pieces; 6 hexes with no land under
COD's shoreline are snapped within 700 m (103 people), 2 dropped (2 people).

**Perseverance Island's hex is dropped** (1 hex, 1,507 people in 2023): the island counts
with English River in 2010 but was built up afterwards: the first houses were for the Indian
Ocean Games of August 2011, fewer than 600 stood by November 2011 (a Seychelles blog of that
month, read 2026-09-14; weak, but it agrees with a search summary saying the first 450
families moved in the following January), and the 2022 census counts 5,410 there.

Checks: every district has cells (median 20 hexes); Kontur/census 1.152 nationally; every
district but Other Islands within a factor of 2 after normalising (Anse Etoile lowest at
0.59, La Digue highest at 1.24); log-population r = 0.891, 0 of 2,000 shuffles reach it.

`water.py` clips 262 of the 636 pieces and leaves 3 unclipped under `KEEP_WHOLE_ABOVE`; its
"unit(s)" there are placement pieces (coastal slivers of the overlap cut), not districts.

`kontur_cap.py sc` does not check this layer (the overlap slivers read as densities above the
cap). No Seychelles hex can be capped: the cap is about 34,000 people a hex and no district
holds more than 4,876.

## 7. What is left open

- The 2022 write-in level, from NBS on request or in microdata (§2).
- `Christian Community Fellowship` and `Nazarite Christian`: identities not found.
- `Not stated` is 21.3% of Roche Caiman, 15.1% of Cascade, 12.3% of English River and 11.4%
  of Baie Sainte Anne against under 1% in nine districts. The 2010 report says an eighth
  enumeration zone covered institutional populations (construction barracks, convents,
  orphanages, the prison); whether they are these cells is not checked.

## 8. Review, 2026-09-14 (`d743fc47-rev2`, light pass)

`check_md` clean, `check_rollup sc` all measured, `check_no_religion` passes with the new
`CLASSIFIED` entry. Screenshot at the entry's `view`: dots on Mahé (Victoria and the east coast),
Praslin and La Digue, none in the sea.

- **`No Religion` classified `separate`: agreed.** The form prints it as its own box beside a
  write-in line, and Seychelles had no people before settlement in 1770, so there is no
  traditional practice for the box to hold.
- **`other.sc`** is routine under the `other.<cc>` rule. At 34 people it draws no dot at 1:1,000
  and does not show in the legend's default view.
- **`Born Again Christian`** is on `christianity.evangelical` here and on `christianity.protestant`
  in `ie2022.py`: one self-description on two nodes. REVIEW names it. I think this node is right
  and Ireland's is the odd one, but for the reason about the phrase (it is the evangelical
  conversion claim), not the one about Seychelles having no Protestant tradition outside the
  Anglican church. Not moved.
- **`note_public`, soft, not edited.** Three things on the fence for a public note:
  - "Counting the missing in the total, its national table puts Catholics at 61.3%, Hindus at 5.4%
    and Muslims at 2.4%, against 2.4% and 1.6% in 2010" gives Catholics no 2010 figure beside it,
    and its 61.3% has 11.5% missing in the total where the note's 76.2% has 4.8% not stated.
  - "twice their share anywhere else but Au Cap, Port Glaud and Roche Caiman" is hard to parse.
  - The closing sentence about Perseverance Island is placement detail that `note` already has.
