# Kuwait (`kw`)

**Drawn 2026-10-03** by session `fafd1067-kw`, under a supervisor, reopening the row left blocked in
`sources.md` §11ao and §scout-2026-09-14-asia-oceania on Anita's priority line (`ask/RULINGS.md`
2026-09-15). 143 drawn units from the 2021 census's 157 areas, 4,381,139 people, every row
`derived`. Code: `sources/kw.py`, `sources/kw_geo.py`, `sources/kw_grid.py`, `taxonomy/kw2021.py`,
`countries/kw.py`. No ask filed. `sources.md` §kw-2026-10-03.

## 1. What was counted

- **PACI's register records a religion for every resident**, and its table builder
  (`stat.paci.gov.kw/englishreports/`, STR SuperVIEW) crosses it with nationality group and sex.
  The host still times out from here on 443 and 80 (2026-10-03). The Wayback Machine holds one
  data answer: `tableQuery?viewId=ColumnChartEduAge&wafers=mf_Year,sf_gender&rows=sf_relegion&
  columns=sf_nationality_TD`, capture 20141015093707, June 2014, 2,110 bytes of JSON (pinned in
  `kw.py`). Muslim / Christian / Other-Not Stated for Kuwaiti, Arabian, Asian, African, European,
  N.American, S.American, Australian, each by sex; 4,039,445 people; all sums close.
- Shares, both sexes: Kuwaiti 99.978% Muslim (255 Christians, 22 other); Arab 94.7 / 5.0 / 0.4;
  Asian 46.3 / 38.5 / 15.2 (women 30.8 / 56.4 / 12.8); African 35.3 / 55.6 / 9.0; European
  38.0 / 56.2 / 5.9.
- **Not published below the nation.** §scout-2026-09-14-asia-oceania read PACI's archived view
  configurations (English 2014, Arabic 2019): `sf_relegion`, `sf_governorate` and `sf_regions`
  exist and no view pairs them. The CDX of `stat.paci.gov.kw/*/tableQuery*` holds 9 data answers
  (6 English, 3 Arabic); the religion one above is the only one with religion. The tool would
  probably answer `rows=sf_relegion&columns=sf_governorate` (the browser query is in that
  section); not reachable from here.
- **2021 census** (`census.csb.gov.kw`, register-based, source line PACI): 118 tables, none on
  religion. Used: Table 1 (governorate x nationality x sex), Table 6 (governorate x sex x
  nationality group: Gulf, Arabic, Asian, African, European, North American, South American,
  Australian), Table 52 (157 areas x Kuwaiti/non-Kuwaiti x sex, Arabic and English names;
  4,578 `Not Stated`). Excel exports at `CensusData_EN?st_id=<n>&handler=ExportExcel` (4, 26, 72).
- **Compiler figures**: US State Department, *2023 Report on International Religious Freedom:
  Kuwait* (opened): PACI 2023, 4.8M people, 74.7% Muslim, 16.6% Christian, 8.7% non-Abrahamic;
  expatriates 62.7 / 24.5 / 12.8; 285 Christian citizens; citizens about 70% Sunni, 30% Shia
  ("NGOs and the media"); expatriate Muslims about 5% Shia; informal community estimates 250,000
  Hindus, 100,000 Buddhists, 25,000 Bohra, 10,000-12,000 Sikhs, 7,000 Druze, 400 Baha'is; "the
  two groups are generally distributed uniformly throughout most of the country"; 60% of the
  Bidoon Shia. Wikipedia's 31/12/2020 PACI figures (63.02% of non-citizens Muslim) were not
  traced to a PACI page.

## 2. The construction, decided

- **Kuwaitis**: 1,488,435, all Muslim. The 277 non-Muslim citizens of 2014 folded in (nothing
  places them). Originally all on `islam` (asks 040 and 043); since 2026-10-03 split at one
  national share, 263,299 `islam.shia` and 1,225,136 `islam.sunni` (§8).
- **Non-Kuwaitis per area, by sex and group**: the 2021 area count (Table 52) split by the area's
  December 2014 group mix (GLMM's copy of PACI's locality table, 183 localities; crosswalk
  `XWALK_2014`, hand-checked), then raked per governorate and sex to Table 6's group totals
  (Gulf less Kuwaitis joined to Arab, as PACI's 2014 `Arabian` holds them). 209 area-sex rows
  start from 2014, 105 from the governorate mix (`PRIOR_MIN` 100). The raking moves 61,882 of
  2,892,704 (2.1%) between groups.
- **Religion**: each group-sex at PACI's June 2014 shares. Muslim to `islam`, Christian to
  `christianity` (not split into churches: PACI's Asian Christians are about ten times what Pew's
  national shares give the named nationalities, so a nationality weighting has nothing to stand
  on). `Other-Not Stated` to `other.kw`, except the Asian part, split by the six Asian
  nationalities PACI's mid-2018 table names by sex (GLMM), each at its Pew 2020 count of
  non-Muslim non-Christians: men 92.3% Hindu, 4.6% Buddhist, 2.3% Sikh; women 77.6 / 18.5 / 1.8.
- **Tier**: all `derived`, `roll` to the PACI column (`taxonomy/kw2021.py` `COLUMNS`; the split
  rolls to `other.kw`); `may_ring` false.
- **Vintages**: religion 2014, population 2021. Accepted under Anita's mixed-vintage ruling (ask
  003): the method keeps each group's own shares and lets the 2021 group mix move.

**Witness** (asserted, 5 points): non-Kuwaitis as drawn 66.3% Muslim, 24.9% Christian, 8.8% other,
against PACI 2023's 62.7 / 24.5 / 12.8. The Muslim and other gaps run the way Indian (Hindu)
growth since 2014 would push them. **The split, printed only**: 216,323 Hindus, 20,379 Buddhists,
5,268 Sikhs against the informal 250,000 / 100,000 / 11,000. Buddhists are probably under-drawn:
India's weight is too large, because Pew's India row applied to 811,409 Indian men gives nearly
four times PACI's whole Asian `Other` for men. A fit with India as the residual was tried and
fails for women (2018's Sri Lankan and Nepali women at Pew already exceed 2014's Asian female
`Other`), so the proportional split was kept.

**Result**: 77.8% Muslim, 16.4% Christian. Muslim share by area from 61.1% (Al-Mahbula) and 64.5%
(Jleeb Al-Shuyoukh) to 90.7% (Taima, a Bidoon area) and 90.2% (Al-Sulaibiya); Kuwaiti suburbs
about 10% Christian (their foreign residents are mostly Asian, and 42-45% Asian women in Qurtuba,
Surra and Bayan in 2014). Al-Shuwaikh Medical is the most Christian area (30.6%).

## 3. Geography

- **OpenStreetMap areas**, `boundary=administrative`, `admin_level=6`, 192 relations (169 tagged
  `source=www.q8maps.com`) and the six governorates at level 4, by Overpass (overpass-api.de; the
  area query timed out twice and kumi/private.coffee returned nothing, so `kw_geo.py --fetch`
  takes the ids from a tags query and the geometry by id). ODbL. Not geoBoundaries KWT ADM2
  (OSM 2011, 137 areas, lacks Sabah Al-Ahmad, Jaber Al-Ahmad, Abdullah Al-Mubarak); no COD-AB
  below the governorate.
- **Join**: 115 of 157 areas on the folded Arabic name, 42 by hand (`BY_HAND`), each reason in the
  code; areas and polygons that touch form one unit (143). Governorate checked for every
  name match (two pinned, 16 and 8 people). Nested OSM areas (Sabah Al-Ahmad inside "South of
  Sabah Al-Ahmad City", Al Mitla, Wafra Residential, Sabhan Industrial) are cut out of the larger,
  smallest first. Each governorate's desert row takes the rural polygons nobody else claims.
- **Witness that neither key decides**: 13 OSM areas carry a `population` tag (most cite PACI,
  2020-2024); every one is 0.80-1.36 of its census area (Sabah Al-Salem 92,818 against 88,904;
  Al-Rai 2,616 against 2,616).
- 27 OSM polygons claimed by no census row (islands, the university, ports, South Khaitan,
  Shadadiya); Kontur's people in hexes wholly inside them are 5.6% of its total and drop out.

## 4. Placement: Kontur's counts are wrong in Kuwait

- Kontur KW (2023-11-01) per unit over the census share: p10 0.12, median 1.30, p90 15.6.
  Sabah Al-Salem 2,785 against 88,904, Mishrief 1,251 against 45,877, Al-Mahbula 13,886 against
  142,145; Wafra Farms 257,607 against 11,961, Al-Abdalli 103,751 against 10,169, Jahra desert
  150,757 against 2,752. 50.5% of its people in a different unit from the census. Its unit rank
  witness still beats every shuffle (+0.386 against a best of +0.345), and the OSM-tag witness
  carries the join.
- So only the **footprint** is used (`kw_grid.py`, `FOOTPRINT_MIN` 50 people/km2 of the whole
  hex), weighted by area, after cutting hexes to units as Malta. Median footprint share of a
  unit's area 1.00 (uniform in residential areas); the Jahra desert keeps 6% of its 9,682 km2,
  Ahmadi's 16%. Densities as weighted reach 53,446/km2 (Al-Farwaniya), so `kontur_cap` reads the
  layer as not raw Kontur and skips it, which is right.

## 5. Reopen if

- PACI's table builder answers from somewhere: `rows=sf_relegion&columns=sf_governorate` (or
  `sf_regions`) would make the religion measured by place and drop the nationality carry.
- A newer religion by nationality group and sex (2020 or 2023) is found as a table, not a quote.
- A Kuwaiti source gives Hindus and Buddhists apart, or nationality by sex for 2014.
- A self-identified sect figure for Kuwaiti citizens turns up (Pew Global Attitudes 2007 is the
  lead, §8), or one that places sect by governorate and Anita approves using it.

## 6. Review, 2026-10-03 (fafd1067-rev7, full pass)

- **Roll-up changed to NOWHERE.** §2's "Tier" line had every row `roll` to its PACI column, which
  `check_rollup.py` read as 4,381,139 rolling up: under `inferred dots: not shown` Kuwait would
  have kept every dot as if Muslim / Christian / other had been counted in each area. Spec
  §7a-i-1's rule is that the column must be measured at the same unit, and PACI's are national
  only (the Switzerland case, and Saudi Arabia, Oman and China empty the same way). `countries/kw.py`
  now writes `NOWHERE` (as `ao`, `ug`); `kw2021.COLUMNS` stays as a record, not attached. The
  published `counts.json` already had `roll: {}` for kw (nobody had run `rollup.py` since), so
  nothing shipped changes; `check_rollup.py kw` now reports 100% orphaned, which is the honest
  answer.
- **note_public range made true.** "runs from 61% in Al-Mahbula ... to over 90%" was not the
  range over units: Central Area (2,501) is 54.6%, Al-Shuaiba Industrial (15,902) 58.0%, Wafra
  Farms 60.7%, West Abdullah Al-Mubarak (1,257) 93.6%. Now "In the areas of more than 50,000
  people", where 61.1% to 90.7% holds. §2's "Al-Shuwaikh Medical is the most Christian area
  (30.6%)" has the same issue (Julaia Chalets 31.5%, a few tiny units higher); left, record only.
- Checked and fine: totals (1,488,435 + 2,892,704 = 4,381,139 = Table 1 less 4,578 not stated),
  non-Kuwaiti 66.31% / 24.87% as the note says, voice of the entry, no new nodes, §14 (nothing
  finer than the Gulf precedent). Screenshot: dots on the urban corridor, Jahra, Abdali and Wafra,
  none in the sea.

## 7. Top text before the 75-word cut, 2026-10-03 (`fafd1067-notes75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/kw.py`; `note_public` was not changed.

- `how`: population register, 2014, counted for the country by nationality group and sex
- `fill`: from the register's national count for each nationality group and sex, by the number of each living in the area
- `grain`: areas, 31,000 people on average
- `gap`: the 4,578 people (0.1%) whose area the 2021 census did not state

## 8. Sect split for Kuwaiti citizens, 2026-10-03 (`fafd1067-kwsect`)

**Ruling** (`ask/RULINGS.md`, 2026-10-03): Kuwaiti citizens may be split Shia/Sunni at one share
for the whole country ("its basically just a city state so one region fine"). Not decided: a
by-governorate split, so none is drawn. `sources.md` §kw-2026-10-03b.

**What exists, checked 2026-10-03:**

- **Arab Barometer wave III** (`data/raw/arabbarometer/ABIII_English.sav`, on disk), Kuwait,
  10 February to 14 March 2014. Technical report (`arabbarometer.org/wp-content/uploads/
  ABIII_Technical_Report.pdf`, p. 6, read): citizens 18+, frame the 2011 census, stratified by
  governorate and urban/rural (12 strata), 200 PSUs by probability proportional to size, 6
  interviews each, systematic household skip, Kish table, 1,021 interviews, partner Gulf Opinions.
  **No quota of any kind is listed, sect included.** The file carries a `wt` (mean 1.000, sd 0.55).
  `q1012`: 1,018 Muslim, 3 Christian; `q1012a` empty for Kuwait. `q2005kw`, *"In the opinion of
  the field team, the respondent is a member of what sect?"*: among Muslims, unweighted Sunni 677,
  Shia 149, cannot determine 192; **weighted Sunni 66.62%, Shia 14.32%, cannot determine 19.06%**.
  By governorate (Shia of those placed, weighted): Capital and Hawalli about 25%, Ahmadi 19%,
  Farwaniya 15%, Mubarak Al-Kabeer 14%, Jahra 9% (the expected order; not used, per the ruling).
- **Other waves**: Kuwait is in V (1,374), VII (1,228) and VIII (1,210). V has no `Q1012` answer
  for Kuwait at all; VII and VIII ask religion (all but a handful Muslim) and leave the sect item
  empty. II declares a Kuwait label with no rows; I, IV and VI never fielded Kuwait. So III is the
  only sect item for Kuwait in ten files.
- **Pew 2009** (*Mapping the Global Muslim Population*, on disk, pp. 10, 29, 40): Kuwait
  2,824,000 Muslims (~95%, from Pew Global Attitudes 2007), Shia 500,000-700,000, **20-25% of all
  Muslims**, citizens or not; Appendix B says the Shia ranges are ethnographic ascription.
  Pew's Shia methodology sheet (`Shiarange.pdf`) prints the range with no Kuwait source note.
- **US State Department 2023**: "about 70% Sunni and 30% Shia" of citizens, attributed to NGOs
  and the media; no method. Wikipedia's 2001 "525,000 Sunni and 300,000 Shia citizens" (36%) and a
  "2002 State Department 39%" were not traced to a source.
- **Lead, not opened**: Pew Global Attitudes Spring 2007 surveyed Kuwait, and Pew's 2009 report
  uses it for the Muslim share. Whether its religion item split Sunni and Shia was not checked;
  the dataset needs a Pew account (her identity), so it stays a lead. Its 2007 report's Middle
  East chapter gives no Kuwait result by sect.

**The call: 17.69% Shia**, the weighted Shia share of the Muslims the field team placed
(`sources/kw.py` `SECT_SHARE`, asserted to 4 places; `SECT_PIN` pins the counts). Every area's
Kuwaitis go to `islam.shia` and `islam.sunni` at that share, rounded inside each area's Kuwaiti
count so no rounding moves anyone across the citizen line (non-Kuwaiti cells byte-for-byte
unchanged, per-area totals unchanged). Non-Kuwaiti Muslims stay on `islam` (the State
Department's "about 5% Shia" among expatriate Muslims is national and unplaced).

Why this figure over the higher ones:

- It is the only one measured on a sample of Kuwaitis, with a stated design and no quota. The 30%
  has no method, and Pew's range is ascription over a denominator (95% Muslim) that PACI's own
  count (77.8%) contradicts.
- **The undetermined were apportioned, not left on `islam`.** That is the opposite of the Iraq
  playbook's `Just a Muslim` rule, on purpose: there the respondent declined a sect; here the
  interviewer could not tell. Checked whether the undetermined lean Shia: over every item with 800+
  answers, for each answer where the coded Shia and Sunni differ by 6+ points, the undetermined's
  position between the two gives the Shia share among them; median 0.21 (67 items), 0.17 (25 items
  at 8+ points), 0.22 (9 at 10+). So they resemble the placed mix (about 18%), not the Shia.
  Weak evidence: the items barely separate the coded groups at all (largest gap 15 points, on the
  Christian-Muslim relations item; relations with Iran 9 points). Jahra, the most Bedouin and
  Sunni governorate, also has one of the highest undetermined shares (20%), which a team failing
  mainly on Shia would not produce. Scratch scripts not kept.
- Post-stratified to the 2021 census's Kuwaitis by governorate it reads 17.67%; asserted within
  a point, so the 2011 frame does not move it.
- **Bounds**: 14.32% if every undetermined respondent is Sunni, 33.38% if every one is Shia. The
  State Department's 30% sits inside them, so the survey cannot rule it out; it only fails to
  support it.

**What could make it wrong**: a team that places Shia less readily than Sunnis, or calls some
Shia Sunni outright, which nothing here can test. Reversing it is one constant.

**Tier and roll**: `derived`, `roll` NOWHERE like every other kw row (§6). The split would
normally roll to `islam`; here `islam` is no more measured per area than the split, so Kuwait
still empties whole under `inferred dots: not shown`. `tools/check_rollup.py kw` reports
4,381,139 of 4,381,139 orphaned, `islam.shia` 263,299 and `islam.sunni` 1,225,136 among them,
which is the honest answer. `kw2021.COLUMNS` records both under `islam`.

**Estimate layer** (spec §15.3): kw now draws `islam.shia` and `islam.sunni`, so `estimates.py`
refuses the two Pew 2009 rows in `estimates_hand.py` on the next run (`refused`: "the country's
own source measures"). They stay in the file as the record and still count toward the world
total; noted in `estimates_todo.md`.

**Dots**: 1:1,000 edition 263 `islam.shia`, 1,225 `islam.sunni`; 1:10,000 26 and 122.
