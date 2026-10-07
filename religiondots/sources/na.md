# Namibia: religion in the 14 regions from the pooled Afrobarometer, on the 2023 census

Built 2026-10-03 by `fafd1067-na`, from the negatives scout's row (`sources.md`
§scout-2026-10-03-negatives). Code: `sources/na.py` (survey), `sources/na_geo.py` (regions, census
populations, Kontur), `taxonomy/na2023.py` (mapping), `countries/na.py` (entry). Record section in
`sources.md`: §na-2026-10-03.

## 1. What Namibia publishes: no religion count

The 2001, 2011 and 2023 census forms carry no religion item (§11aq read the 2011 questionnaire and
PUMS dictionary, the 2001 Form A and the 2023 PUMS dictionary). The 2023 *Main Report* (122 pp) has
no occurrence of the word. Absent from the UNSD oracle.

Two open Demographic and Health Survey reports print religion nationally for women and men aged
15-49, in Table 3.1:

| | DHS 2013 (FR298, PDF p.52) | DHS 2006-07 (FR204, PDF p.53) |
|---|---|---|
| ELCIN (Evangelical Lutheran Church in Namibia) | 44.0% women, 43.4% men | inside "Protestant" (77.0%, 70.3%) |
| Roman Catholic | 19.6%, 25.9% | 20.9%, 26.3% |
| Protestant/Anglican | 21.2%, 12.7% | |
| Seventh-Day Adventist | 4.8%, 4.0% | |
| No religion | 1.1%, 1.8% | 1.4%, 2.4% |
| Other | 9.0%, 12.0% | 0.4%, 0.6% |

Pooled by weighted number: ELCIN 43.9%, Catholic 21.6% (2006-07: 22.4%), Adventist 4.5%, no
religion 1.3%. Both are re-read from the PDFs on every build (`na.py::dhs_witness`). The 2013 DHS
microdata would cross religion with region; it is behind a DHS account and was not requested.

## 2. The construction

    row margin      region populations      2023 census, Main Report Table 2.2      EXACT
    composition     each unit's own mix     Afrobarometer R4-R9, 7,144 answers      measured
                                            (Lutheran, Anglican: R5-R6 only)
    national level  neither                 computed

The Nigeria construction (ask/answered/010-ng), as Cameroon and Madagascar: nothing fitted, every row
`modelled`. Everyone the census counted is drawn, including the 145,395 non-Namibians (4.8%, Table 3.1;
61.5% of them Angolan) whom the survey, which interviews citizens, does not represent; they take the
mix of the region they live in.

### Thirteen survey units on fourteen regions

Rounds 4 (2008) and 5 (2012) sample Kavango whole and call Zambezi `Caprivi`; rounds 6-9 name the 14
regions. Round 5 has no location below the region, so Kavango cannot be split there, and
`cab.stability` refuses an empty (round, unit) cell. The survey is read at 13 units and Kavango East
and West both take the Kavango mix at their own census populations. Every round samples all 13; the
held-out population check passes in every round (r = +0.92 to +0.99, no random pairing of 20,000
reaches it).

### Populations: the census, not COD-PS

COD-PS Namibia 2023 is the US Census Bureau's projection from 2011 (2,777,232) and its adm1 sheet has
shifted total columns. The census's own Table 2.2 gives 3,022,401, re-read from the PDF.

## 3. The Lutherans, and why two rounds

`Lutheran` by round, weighted: 21.5, 42.8, 41.9, 22.2, 20.3, 23.5% (R4-R9). In the four Owambo
regions together it is 22.8, 63.9, 57.9, 28.6, 25.7, 34.1%, and the boxes that take up the difference
change by round:

| Owambo regions, % | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|
| Lutheran | 22.8 | 63.9 | 57.9 | 28.6 | 25.7 | 34.1 |
| Anglican | 20.9 | 12.9 | 14.4 | 11.1 | 30.5 | 18.7 |
| Evangelical | 17.7 | 1.6 | 1.2 | 14.7 | 6.6 | 6.2 |
| Christian only | 6.4 | 4.5 | 2.8 | 20.7 | 12.4 | 15.9 |
| Roman Catholic | 25.2 | 13.1 | 16.3 | 15.9 | 16.6 | 20.9 |

Round 4 codes ELCIN members `Evangelical` (the church's own name begins "Evangelical Lutheran"),
round 7 `Christian only` and `Evangelical`, round 8 `Anglican` (34% of Omusati and 30% of Oshikoto,
regions where every other round has Anglicans at 4-14%). Rounds 5 and 6 put all Lutherans at 42.8%
and 41.9%, against the DHS's 43.9% for ELCIN alone. So:

- **Lutheran and Anglican are taken from rounds 5 and 6**, as each unit's share of the pool they
  trade with (`Christian, other`: every Christian answer except Catholic, Adventist and Pentecostal).
  The pool's own level and geography come from all six rounds (split-half +0.854). Within rounds 5
  and 6, Lutheran ranks the 13 units alike (+0.725, p=0.007) and so does Anglican (+0.578, p=0.02).
- **Catholic, Adventist and Pentecostal come from all six rounds.** They hold their level by round
  (ranges 7.2, 2.0 and 2.6 points) and pass the split-half (+0.757, +0.573, +0.670). Catholic matches
  the DHS (22.6% drawn against 21.6%). Adventists are drawn at 2.9% against the DHS's 4.5%: short,
  and the shortfall is not fieldwork (their level holds), so it is the survey's against the DHS's.
  Pentecostals have no outside witness; Cameroon's rule (a church the probing does not move) is the
  reason they are drawn.

Within rounds 5-6 alone, Catholic and Adventist fail their single R5-against-R6 split (+0.24, +0.20)
while passing over six rounds; one halving of two rounds is a weak test, which is why only the two
churches that need those rounds take their shares from them.

Asserted: rounds 5 and 6 each within 3 points of the DHS's ELCIN (`LUTHERAN_DHS_GAP_MAX`), every other
round at least 15 points under it (`LUTHERAN_SWING_MIN`), Catholic, Adventist and Pentecostal within 8
points across rounds, pooled Catholic within 3 points of the DHS, Lutheran as drawn within 5 points
of ELCIN (40.5% against 43.9%; the pool's six-round level is under rounds 5-6's), and both carved
churches still passing their split-half.

## 4. None and traditional

The card offers `Traditional/ethnic religion` and `None` (and `Atheist`) in every round. Nationally
traditional runs 0.9, 3.4, 2.5, 0.0, 0.6, 0.1% and none 2.0, 0.0, 1.1, 3.3, 5.6, 3.5%. Kunene is 43%
and 38% traditional in rounds 5 and 6, then 16% and 18% none in rounds 7 and 9 with no traditional.
Tested apart each passes the split-half (+0.518, +0.535), but the levels swap, so Madagascar's rule
applies: placed as one box (+0.555) and split at one ratio. Rounds 4-6 put none at 33.5% of the pair,
rounds 7-9 at 94.5%; the DHS has no traditional box to witness either. The ratio used is all six
rounds pooled, 68.1% none, like every other level here. Result: none 2.7%, traditional 1.25%; Kunene
17.2% and 8.0%.

`None` is `unaffiliated` (step 2 of the draft procedure: the card offers traditional separately).

## 5. The tail

`Other` passes the split-half (+0.446) but 105 of its 108 answers are in rounds 7-9, most in Kunene,
Omaheke and Otjozondjupa, where the early rounds had traditional religion: not placed (Cameroon's
case). Muslims: 5 answers. The residual would draw Muslims at 4.56x in Kunene, so the tail is flat
(spec §12's 2x rule): both at their national share in every region. New node `other.na`.

## 6. As drawn

Lutheran 40.5%, Catholic 22.6%, other Christian 18.4%, Anglican 6.6%, Pentecostal 3.5%, Adventist
2.9%, none 2.7%, other 1.5%, traditional 1.25%, Muslim 0.06%.

| region | Lutheran | Catholic | Anglican | Adventist | other Christian | none + trad |
|---|---|---|---|---|---|---|
| Oshikoto | 67.6 | 14.6 | 7.5 | 0.3 | 6.5 | 0.9 |
| Omusati | 54.9 | 26.8 | 4.6 | 0.3 | 8.4 | 1.9 |
| Ohangwena | 53.7 | 9.2 | 30.6 | 0.0 | 1.3 | 2.2 |
| Oshana | 52.2 | 20.2 | 5.3 | 0.8 | 14.3 | 1.9 |
| Erongo | 46.4 | 19.9 | 1.7 | 1.4 | 20.0 | 6.7 |
| //Kharas | 40.1 | 33.1 | 2.5 | 1.9 | 19.0 | 1.0 |
| Kunene | 34.2 | 12.3 | 3.1 | 0.0 | 20.8 | 25.2 |
| Khomas | 33.9 | 17.9 | 6.1 | 1.1 | 30.2 | 4.3 |
| Kavango East, West | 21.3 | 47.2 | 0.5 | 3.5 | 22.6 | 1.2 |
| Zambezi | 4.7 | 20.2 | 0.0 | 41.0 | 24.0 | 1.7 |

Drawn at zero where rounds 5-6 or all rounds found none: Anglicans in Zambezi and Omaheke, Adventists
in Kunene.

## 7. Placement

COD-AB Namibia v01 (2020), 14 regions, joined to the census by name. The census's densities (region
profiles, PDF pp.14-27) contain COD-AB's area for 12 regions; for Kavango East and West they do not
(census about 24,000 and 24,650 km2, COD-AB 25,367 and 23,091) while the pair's sum agrees, so COD-AB's
line between the two sits about 1,400 km2 west of the census's. Both take one mix, so it moves no
religion. Kontur NA 2023-11: 628 hex centroids fall outside every region, all within 3.5 km of one;
263 inside a neighbour on Natural Earth (Angola, Zambia, Botswana, South Africa; 20,406 people) are
dropped, 365 in the sea or inside Natural Earth's Namibia short of the border river (19,900 people)
are snapped to the nearest region. Kontur against the census per region, over the national 0.868:
0.83 (Kavango West) to 1.13 (Khomas). No block at the density cap.

## 8. The §14 read

Nothing to raise. Religious practice is free, no group is targeted, and the units are regions of
100,000 to 495,000 people. The Himba's traditional religion in Kunene is drawn at region grain from a
survey answer.

## 9. What would improve it

- The 2013 DHS (and 2006-07) microdata: religion by region for 13,000 respondents, with ELCIN named,
  which would be a regional witness to the Lutheran share. Needs a DHS account.
- A census that asks; none of 2001, 2011, 2023 does.
- Church statistics (ELCIN, ELCRN, the Catholic dioceses of Windhoek, Keetmanshoop and Rundu) were not
  looked for; they would witness membership, not self-identification.

## Review, 2026-10-03 (`fafd1067-rev9`, full pass)

Checks clean (`check_md`, `built_countries --check`, `check_rollup na`: all 3,022,401 modelled,
nothing orphaned). Every note figure re-summed from `data/normalized/na.csv` and matches. Mapping
follows precedent (`christianity.catholic` bare as mg/zm, `indigenous.african` as tz/mg, `other.na`
as `other.mg`). Screenshot: dots in the Owambo belt, Windhoek, Rundu and along the Kavango, none in
the sea.

**The DHS's `ELCIN` is not "ELCIN alone".** The 2013 DHS questionnaire (FR298 PDF p.433, Q113)
offers six codes: Roman Catholic, Protestant/Anglican, ELCIN, Seventh-Day Adventist, No religion,
Other. ELCIN is its only Lutheran box, so ELCRN and German-church members had nowhere else Lutheran
to answer, and the 43.9% is best read as "Lutheran" with some of the ELCRN possibly in
Protestant/Anglican. That reading is also the one the construction needs: rounds 5-6's *all
Lutherans* (42.8%, 41.9%) can only "agree with" the DHS if the DHS figure is all Lutherans too. The
note said "in the Evangelical Lutheran Church in Namibia alone"; now "the only Lutheran church its
card named"; `tiles.py --refresh-meta` run. Not changed: the same "ELCIN alone" wording in §1/§3
above, in `taxonomy/na2023.py`'s Lutheran REVIEW reason ("3.4 points under ELCIN alone") and in the
`sources.md` section; read them with this correction. The drawn 40.5% stays a few points under the
DHS's Lutheran box either way.
