# Maldives (`mv`)

**Drawn 2026-10-03** by session `fafd1067-mv`, reopening the scout's block
(`sources.md` §scout-2026-09-14-asia-oceania, "Maldives (`mv`): blocked; only foreigners were asked")
on the Mauritania construction (`ask/RULINGS.md` 2026-09-15 and 2026-09-16). 21 units (20 atolls and
Malé), 402,071 people (2014 census, place of enumeration), 400,843 drawn. Code: `sources/mv.py`,
`sources/mv_geo.py`, `taxonomy/mv2014.py`, `countries/mv.py`, node `other.mv`. `sources.md`
§mv-2026-10-03.

## 1. Who was asked

- **2014 form** (read by the scout from the Wayback copy of `Census2014-Questionnaire_english.pdf`,
  not re-read this session): M6 nationality, "1- Maldivian" skips to M10; M7 religion (Islam,
  Hinduism, Buddhism, Christianity, other) for foreign residents only. No box for no religion.
- **Constitution of 2008**, Article 9(d) (Constitute Project, `Maldives_2008`, read 2026-10-03): "a
  non-Muslim may not become a citizen of the Maldives." Article 10: Islam is the state religion.
- **The tabulation codes every Maldivian Islam.** UNSD table 28's 2014 rows (below) have 1,228 `Not
  Stated` among 402,071 people; by sex and urban/rural every not-stated cell fits inside the foreign
  residents, and every Islam cell less the Maldivians is positive.
- **2022 census: no religion anywhere published.** Wayback CDX of `census.gov.mv/2022/*` (the live
  site answers 404 to everything since at least 2026-10-03) lists 180 documents; read for
  `religio|hindu|christian|buddhis`: the population report (two 84-page captures; the
  `_22724` capture is a 1 MiB fragment without `%%EOF`), the population summary and dynamics
  reports, the migration report and its summary, the resort census report and summary, and
  `Definitions-Population.xlsx`. The only hit is the migration report's "one language, one culture
  and one religion" (p.20). No 2022 form was found in the CDX list.
- DHS 2016-17 asks no religion (scout). HIES 2019 not checked.

## 2. The foreign residents' answers: UNSD table 28

`tools/oracle.py "Maldives"` (pinned in `sources/mv.py::UNSD`). `Urban` is Malé: 153,904, equal to
Table PP3's Malé row and PP2's. Less PP3's Maldivians (all coded Islam), the foreign residents:

| | men, Malé | men, atolls | women, Malé | women, atolls |
|---|---:|---:|---:|---:|
| all | 20,995 | 34,792 | 3,528 | 4,322 |
| Islam | 14,874 | 22,783 | 762 | 798 |
| Hindu | 3,025 | 5,145 | 884 | 1,109 |
| Christian | 1,326 | 2,572 | 1,019 | 1,486 |
| Buddhist | 1,329 | 3,095 | 533 | 380 |
| Other | 209 | 462 | 199 | 419 |
| Not Stated | 232 | 735 | 131 | 130 |

Foreign women are a third Christian (2,505 of 7,850) against 7% of foreign men, so sex is carried
through the spread (`playbooks/census_table.md`'s Saudi trap).

## 3. Spreading the atolls' answers (`sources/mv.py`)

Tables, Statistical Releases I and II (`statisticsmaldives.gov.mv/mbs/wp-content/uploads/2015/12/`):
PP3 (residents by nationality, sex, Malé, administrative islands per atoll, non-administrative
islands nationally; place of enumeration), MG4 (Maldivians by place of enumeration per atoll, all
islands), MG14 (foreign residents by country of birth, sex, atoll, administrative and other
islands; **place of usual residence**), MG15 (foreign residents by citizenship: India, Sri Lanka,
Bangladesh, others; sex; Malé, administrative, non-administrative; place of enumeration), PP5
(each administrative island). Every cross-table identity is asserted in step 1.

- **Fit.** IPF of sex x place (Malé, administrative, non-administrative) x citizenship x answer to
  MG15's sex x place x citizenship and the census's sex x (Malé, atolls) x answer. Seed: Pew 2020 per
  home country (India, Sri Lanka, Bangladesh), `others` from UN DESA 2015's 19 named origins for the
  Maldives (2,888 of the census's 6,555 `others`; DESA's 4,215 unnamed), families mapped to the
  census's answers (unaffiliated half Other, half Not Stated: the form has no "none"), floor 0.002.
  Fitted, men in the atolls: Indians 64% Hindu, 23% Muslim, 7% Christian; Sri Lankans 70% Buddhist;
  Bangladeshis 93% Muslim; others 58% Christian. Administrative and non-administrative islands get
  the same mix per citizenship and sex, since no margin tells them apart.
- **Per atoll.** Administrative islands: PP3's foreign residents per atoll by sex. Resort and
  industrial islands: MG14's per atoll (atoll row less administrative row; Kolhumadulu's and
  Gnaviyani's printed rows are blank, every printed one equals the difference) scaled from 20,289 and
  2,183 (usual residence) to MG15's 20,917 and 2,193 (enumerated). Citizenship inside each: MG14's
  countries of birth (born in the Maldives and not stated spread pro rata), raked to MG15.
  MG14's administrative islands differ from PP3's by -289 men in all (-64 to +52 per atoll); PP3's
  are used.
- **Rounding** keeps each atoll's foreign residents and the atolls' six answer totals exact.
- **Witness, how much the seed decides:** every citizenship seeded alike moves 795 of the atolls'
  14,668 non-Muslim foreign residents between atolls (Kaafu -382); `others` on Pew's All
  Asia-Pacific moves 94. Printed by the script; not a guard.
- **As drawn.** Foreign non-Muslims: Malé 8,524 (measured), Kaafu 3,878, Alifu Dhaalu 1,449, Baa
  1,225, Alifu Alifu 958; Vaavu and Gnaviyani 107 each.

## 4. Tiers

Malé's foreign answers are the census's count for Malé (`measured`); the atolls' are the count for
all atolls together, spread (`derived`, `fill=`); Maldivians are `modelled`. Malé's `islam` pair is
both, so it draws at the weaker tier.

## 5. Geography and placement (`sources/mv_geo.py`)

- **Units:** COD-AB Maldives (HDX `cod-ab-mdv`, valid from 2024-10-22; the `.gdb.zip` reads as six
  empty layers under pyogrio, the shapefile zip works): 21 first-level units and 1,556 islands. The
  census names atolls geographically with the administrative letters in brackets; `ATOLLS` pairs the
  letters with pcodes, each COD name asserted. Witness: each atoll's PP5 island names found under its
  own pcode (50% to 100%, folded) against at most 33% under any other. The bar written first (60%)
  failed Dhaalu at 50% on spelling and repeated names; the test kept is own >= 2.5x the next.
- **Island join:** all 187 PP5 islands join a COD island of their atoll one to one: by folded name,
  then difflib (cutoff 0.75; 31 matches, all printed, all spelling: `Kuburudhoo` / `Kun'burudhoo`),
  then one pin, Laamu's `Gamu` -> `Gan`.
- **Placement:** Kontur MV 2023-11-01 (1,019 hexes, 518,768 people) cut to the islands, each hex's
  people over its land pieces; 30 landless hexes (873 people) snapped within 1 km. **Kontur is wrong
  between islands**: 198,456 on Malé island against 128,767 counted, 4,347 on Hulhumalé against
  17,149, 4,627 on Funadhoo (a fuel-depot island beside Malé's harbour). So every counted island is
  scaled to its PP5 count (17 with no Kontur piece take their own polygon, Maavah 1,530 the
  largest), and each atoll's remainder (resort and industrial islands; Kaafu 13,072 on 217 pieces)
  is spread over its other islands on Kontur. Malé's harbours row and Hulhulé (367, the airport
  island, a Kaafu island in COD) are put on Malé island.
- Kontur per unit before calibration, share over census share: 0.70 (Vaavu) to 1.61 (Raa); rank
  witness +0.938 against a best shuffle of +0.709.
- `kontur_cap.py` skips the calibrated layer (densities reach 846,871/km2: a hex that is mostly sea
  puts its people on the sliver of island it touches, and the island's scaling keeps that share; the
  dots stay on the right island). The raw grid's 5 hexes at the cap are all Malé (62,500/km2 on
  2014's count for Malé island), real and left alone.

## 6. Not drawn

- 1,228 foreign residents who did not state a religion (`gap_share` 0.003054, `tools/gap_share.py`).
- Maldivians who are not Muslim: nobody counts them.
- **Foreign workers the census missed, unsized.** PP2's footnote says only that "attempts were made
  to include resident foreigners in 2014". A search summary (not evidence) says the national
  planning department called the census's 58,683 migrant workers far below the Immigration
  Department's figure and that about 63,000 were undocumented; the ILO page it cites
  (`apmigration.ilo.org`) no longer resolves, the Wayback Machine was offline on 2026-10-03, and the
  Statistical Yearbook's expatriate employment tables stop at 2011 (statistical archive Tables
  3.2-3.7; Yearbook 2015 Tables 5.3-5.6 return 404). So `gap` names them without a figure, as Saudi
  Arabia's does. Reopen with the Immigration Department's or the economic ministry's work-permit
  count for September 2014 (ask 033's rule).
- §14: drawn are foreign residents at atoll grain, modelled below the atolls' total, and no
  Maldivian on anything but Islam. No ask.

## 7. Reopen if

- A 2022 religion table appears (the 2022 base would double the foreign residents: 132,493).
- A work-permit count for 2014 is found (§6).
- Per-island foreign religion is wanted: PP5 has foreign residents per island, and the census
  microdata (`app.statisticsmaldives.gov.mv/census-data/`, a request form asking the requester's
  identity, Anita's to decide) would give religion by island.

## 8. Review, 2026-10-03 (fafd1067-rev7, light pass)

- **The fit's sums hold**, recomputed from `data/normalized/mv.csv` against `tools/oracle.py
  Maldives`: Malé's six foreign answers equal UNSD Urban less PP3's 129,381 Maldivians exactly;
  the 20 atolls' rows sum to UNSD Rural less the Maldivians (Islam 23,581, Hindu 6,254, Christian
  4,058, Buddhist 3,475, Other 881, not stated 865; 39,114); national 402,071 with 338,434
  Maldivians, every note figure as printed.
- **`check_rollup.py mv` reports 38,249 (9.5%) orphaned**: the atolls' spread answers. Honest
  (counted only for the atolls together), so nothing to wire. Comment added at
  `taxonomy/mv2014.py`'s `COLUMNS` saying it must not be attached as `roll`: rollup.py's table is
  per node for the whole country, so an identity roll would keep the atolls' answers on screen on
  the strength of Malé's count.
- Screenshot around Malé: dots on Malé, Hulhumalé and the resort islands, none in open sea.

## 9. Top text before the 75-word cut, 2026-10-03 (`fafd1067-top75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/mv.py`; `note_public` was not changed.

- `how`: census, 2014, which asked foreign residents only; Maldivians drawn as Muslim
- `fill`: from the census's count for all the atolls together, by citizenship and sex
- `grain`: atolls and Malé, 19,000 people on average
- `gap`: the 1,228 foreign residents who did not state a religion, 0.31% of residents; Maldivians who are not Muslim, whom no source counts; and foreign workers the census missed, whom nobody has counted
