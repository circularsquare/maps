# Faroe Islands (`fo`): Census 2011, congregation association by district

Built 2026-09-15 by session `d743fc47-fo`, from the scout's record in `sources.md`
§scout-2026-09-15-europe. Code: `sources/fo.py` (tables), `sources/fo_geo.py` (placement),
`taxonomy/fo2011.py` (mapping), `countries/fo.py` (entry). Record section: `sources.md`
§fo-2026-09-15.

## 1. Source

- **Hagstova Føroya statbank, Census 2011** (`H2/MT`), open PxWeb v1 API, POST, no key:
  **MT325** congregation association by age, sex and district (drawn); **MT321** religion by age,
  sex and district (witness); **MT1** population by single year of age and district (universe).
  Queries saved beside the CSVs in `data/raw/fo/*.query.json`. The API table was last updated
  2026-02-06; `fo.py` pins every national and district cell against the scout's transcription.
- **The form.** UNSD's questionnaire archive has the English and Faroese forms
  (`unstats.un.org/unsd/demographic/sources/census/quest/FRO2011en.pdf`, 635,037 bytes, pinned).
  E23, voluntary: "With which Christian church, congregation or community are you associated?
  Select all that apply." Boxes: National Lutheran Church; Christian missionary movements
  (*Missiónshús, Meinigheitshús, Salvation Army, KFUM, KFUK, etc.*); Plymouth Brethren;
  Charismatic, evangelical congregations (*»Hvítusunnusamkomur«, etc.*); Seventh Day Adventist;
  Catholic Church; Orthodox Church; Jehova's Witness; Other; None. E22, voluntary, asks religious
  belief (Christian, Islam, Hindu, Buddhism, Judaism, Bahá'í, Sikh, other, no belief). E16 ends
  the form for anyone 14 or under.
- **The law.** Løgtingsmál 115/2010 (`logting.fo/files/casestate/10903/`), §3 stk.1 nr.4 "Trúgv og
  tilknýti til kristnar kirkur, fríkirkjur og samkomubólkar"; stk.3 makes answering compulsory
  except that item. Parents fill in for children (Til §3).
- **Table notes** (the PxWeb page for MT325): persons 15 years or older; "The associations are
  enumerated. Some persons have more than one association"; `Not queried` = "did not fill out the
  query form"; fewer than 3 shown as `...`. Faroese labels: `Samkomur nærri fólkakirkjuni` for the
  missionary-movements row, `Ikki svarað spurnablaði` for not queried.
- **UNSD table 28 has no Faroese row.** No census since 2011 (the one before was 1977). The only
  later religion series is `MM03010`, National Church membership by parish, one church only.

## 2. Universe and checks

Everyone aged 15 and over: MT1's population minus ages 0-14 equals MT325's persons in all seven
districts and nationally (37,965 of 48,346 residents). Responses + not stated + not queried =
persons in all 14 columns of both tables, and MT321's persons and not-queried cells are MT325's.
Districts, ages and sexes sum to the national cell in every unsuppressed row except the two
`Other` rows, which hold suppressed bodies (below). Drawn: 34,436, 71.23% of residents.

Gap: under-15s 10,381 (21.47% of residents), who were not asked; not stated 2,369 and not queried
1,160, together 3,529 (9.30% of the 15+, 7.30% of residents). `gap_share` 0.2877 is both, of
residents.

## 3. Multiple associations, and the rescale

MT325 counts ticks. Nationally the nine bodies and None sum to 39,558 against 34,436 answers:
5,122 extra ticks, 4,596 people with more than one association, so 526 third and fourth ticks.
The only pair printed is National Church and Brethren, 713.

A dot is a person, so `fo.py` scales each district's five body columns by (answers minus None) /
(ticks on bodies): Norðoyar 0.830, Eysturoy 0.799, N-streymoy 0.912, S-streymoy 0.907, Vágar
0.854, Sandoy 0.903, Suðuroy 0.901. None is a person count and stays as printed. This assumes the
extra ticks fall on each body in proportion to its size.

The reading that would have justified a different rule, that the unnamed overlaps are National
Church plus missionary movement, fails the arithmetic: the missionary movements are fewer than
the extra ticks left after the named pair in six of seven districts (Norðoyar 625 against 695).
Bounds for the call: the Brethren are 4,619 people after the rescale and at most 5,381, so 13.4%
to 15.6% of answers.

After the rescale, share of each district's answers:

| district | answers | Lutheran (NC) | missionary | Brethren | Pentecostal | other | none |
|---|---:|---:|---:|---:|---:|---:|---:|
| Norðoyar | 4,184 | 50.5% | 12.4% | 30.7% | 3.1% | 1.5% | 1.8% |
| Eysturoy | 7,782 | 63.9% | 16.6% | 12.7% | 3.5% | 1.3% | 2.1% |
| N-streymoy | 2,626 | 80.4% | 7.3% | 4.5% | 2.3% | 1.7% | 3.9% |
| S-streymoy | 13,234 | 70.6% | 6.0% | 12.5% | 3.5% | 1.8% | 5.6% |
| Vágar | 2,190 | 74.7% | 13.2% | 5.0% | 3.1% | 1.1% | 2.9% |
| Sandoy | 993 | 77.3% | 8.1% | 3.5% | 7.5% | 1.5% | 2.1% |
| Suðuroy | 3,427 | 74.9% | 8.5% | 12.5% | 0.9% | 0.8% | 2.3% |
| nation | 34,436 | 68.3% | 10.1% | 13.4% | 3.2% | 1.5% | 3.6% |

## 4. Suppression and the mapping

The Adventist (93), Catholic (167), Orthodox (93) and Jehovah's Witness (126) rows are `...` in
every district; each district's `Other congregations` holds them, and the districts' Other sums to
585, exactly those four plus national Other (106). MT321 folds its six named non-Christian
religions into `Other belief` the same way (273). Neither is split by national composition: under
one dot in total, with nothing at district level to check a split against.

Mapping (`taxonomy/fo2011.py`, all but the first in REVIEW with reasons): National Church and
missionary movements to `christianity.lutheran`; Brethren to `christianity.plymouth` (not `.open`:
no source read names the Faroese assemblies Open Brethren); Pentecostal box to
`christianity.pentecostal`; district Other to `christianity`; None to `unaffiliated` (it is None to
a question about Christian congregations, so it includes non-Christians, at most 273 people).
No new node.

## 5. Geography and placement

No district polygons exist (geoBoundaries has the outline only). **A district is a list of
villages**: the census codes 4100-4700 are Hagstova's register regions (IB01035), and the 119
villages of IB01031, assigned in `fo_geo.py::VILLAGES`, sum to the seven regions to the person in
November 2011 and November 2023. IB01035's November 2011 regions are 0.988 to 1.010 of the
census's district populations. N-streymoy is Kvívík and Vestmanna municipalities, Sunda's
Streymoy side, and Kollafjørður, Signabøur and Oyrareingir (Tórshavn municipality).

**Rule for a hex**: GADM 4.1 municipality of its centroid; Sunda split by island, using land cut
from OSM's sea polygons (GADM's Sunda polygon is one part across the Sundini sound: tried first,
it put all seven Streymoy-side villages in Eysturoy); Tórshavn municipality split by nearest named
village. Witness: all 115 villages with a GeoNames point land in their register district by the
rule. GeoNames' admin codes are wrong for several villages (Oyri, Saltnes, Kolbeinagjógv), so only
its coordinates are used; 16 names need an id (`GEONAMES_ID`); four villages with 14 people have
no point.

**Kontur alone was rejected as the weight.** Against the register of its own month (November
2023), by district over the national ratio: Norðoyar 0.953, Eysturoy 1.063, N-streymoy 1.158,
S-streymoy 0.921, Vágar 1.027, Sandoy 1.301, Suðuroy 1.036. The band set beforehand was 0.85-1.15;
N-streymoy and Sandoy fail it, and the band could not have caught Sunda's Streymoy side placed in
Eysturoy (N-streymoy 0.853). Within 2.5 km of named villages Kontur has Tórshavn at 7,289 against
13,999 and Hvítanes at 753 against 106: it spreads the town into the countryside. **So the weight
is the register**: each hex takes the nearest village in its district, and each village's November
2011 register population is shared over its hexes in Kontur's proportions (the Iran calibration,
one level finer). The calibration moves 6% (Sandoy) to 28% (S-streymoy) of a district's weight.
Every village with people has at least one hex.

## 6. Not checked, and where to reopen

- Whether the Faroese Brethren are all Open Brethren (would move the node to `.open`).
- The Faroese-language form (`FRO2011fo.pdf`, same archive) was found but only the English one read.
- MT326 and MT327 cross congregation with birthplace, arrival year, occupation and education;
  not read. They could tell whether the Catholic and Orthodox answers are recent arrivals.
- Hagstova's Census 2011 publications may describe what `Other` and the missionary movements
  held; not searched beyond the statbank and the census page.

## 7. Review, 2026-09-15 (session `d743fc47-rev12`)

- **The rescale has a direction: towards Lutheran, away from the free churches.** MT325 prints
  `More than one` and the National Church plus Brethren pair for every district. Take the pair and
  the missionary movements off each district's extra ticks and almost nothing is left: Norðoyar 70,
  Eysturoy 128, N-streymoy 9, S-streymoy 126, Vágar 5, Sandoy 9, Suðuroy 23 under; nationally 324,
  fewer than the 526 third and fourth ticks alone. §3's test shows only that the overlaps are not
  all National Church plus missionary movement. The reading the district columns fit is that nearly
  everyone with two ticks ticked the National Church and then a missionary movement or the Brethren.
  Under it (the missionary tick doubles inside the Lutheran node, the named pair split half each,
  the remainder spread in proportion) the Brethren are **14.4%** nationally against 13.4% drawn and
  **34.7%** in Norðoyar against 30.7% (36.4% if the pair counts as Brethren), Eysturoy 14.5% against
  12.7%, Pentecostal 3.6% against 3.2%, Lutheran with the movements 76.7% against 78.3%. The
  proportional rule shrinks every body whose people tick one box, so the gap is largest exactly
  where the Brethren are. Only T325's district columns go into this. Not rebuilt: the builder's
  call whether to switch, and the note's figures and its "in proportion to their size" sentence
  would change with it.
- Otherwise clean. check_md, built_countries, check_rollup (34,436 measured, nothing orphaned).
  Every note figure re-derived from `data/normalized/fo.csv`; `gap_share` 0.2877 is (10,381 +
  3,529) / 48,346. Mapping agrees with REVIEW; None sits below MT321's no-belief count in every
  district but by a handful, which supports `unaffiliated`.
