# Unspecified splits: DRC, Kenya, Ethiopia, Malawi (scout, 2026-10-16)

**Built the same day (Anita approved Kenya and Malawi):** `sources/ea_christian.py` writes
`data/normalized/ke_split.csv` and `mw_split.csv`, read by countries/ke.py and mw.py through
`_swap_christian_split`. The records are `sources/ke.md` §9 and `sources/mw.md` §10. Malawi
differs from the proposal below in two ways: independents and `Christian only` are placed as one
box and split at R7-R9's national ratio, and the Other Christian Denominations residual now
files at bare `christianity`.

Research only; nothing in the project was edited. Scratch scripts are in the session scratchpad
(`eastafrica_scripts/`: `ab_load.py`, `ab_tab.py`, `gfs_ke.py`, `ke_split.py`, `ke_ab_region.py`,
`mw_buckets.py`), tabulating the Afrobarometer R4-R9 and GFS wave 1 files already on disk
(`data/raw/afrobarometer/`, `data/raw/gfs/`). WebSearch calls used: 6.

**Tier.** Every split below sits inside a column the source counted, so by spec §2.7a the rows are
`derived` (rolled back to their column by "inferred dots: not shown"), not `modelled`.
`bg_split.py` tags Bulgaria's rows `derived` too.

## Ranking

| rank | country | bucket now | best source | proposal | confidence |
|---|---|---|---|---|---|
| 1 | **Kenya** | `christianity.protestant` 15,777,473 (33.4%) | GFS wave 1 `REL3_Y1`, 47 counties, 10,192 Christians; KNBS enumerator manual for what the cell holds | split by county into Anglican, SDA, Presbyterian, Baptist, Methodist; rest stays | **high** |
| 2 | **Malawi** | `christianity.other` 6,311,166 (36.0%) = SDA/Baptist/Apostolic 1,644,829 + Other Christian Denominations 4,666,337 | Afrobarometer R4-R9, 3 regions, ~3,100 respondents in the two cells | split each cell at regional (or national) shares | **medium** |
| 3 | DRC | `christianity.protestant` ~32.8M (27.85%) | none open: no survey splits the ECC | hold | none |
| 4 | Ethiopia | `christianity.protestant` 13,661,588 (18.5%) | none: surveys don't name Mekane Yesus or Kale Heywet; church claims only | hold | none |

Visual payoff: Kenya is highest. Its Anglicans run on into Uganda's Church of Uganda (Busia, Teso)
and its Adventists into Tanzania's Mara and Mwanza (Kisii, Nyamira, Migori, Homa Bay are 72-97%
SDA inside the Protestant pool). Today both stop at the border. Malawi comes second: Zambia
(full church list) and Tanzania (SDA, Pentecostal, Anglican, Lutheran) already draw the same
churches, and Malawi's one 36% "other" colour is the largest unspecified Christian cell in the
region. DRC would pay most in principle (Angola's Baptists in the Kongo lands, Zambia's CMML and
Methodists in Katanga, Uganda's and Tanzania's Anglicans in the east), but there is nothing to
draw it from. Ethiopia borders Kenya's and Eritrea's `christianity.protestant`, the same node, and
South Sudan's bare `christianity`, so splitting it would make it stand out *more*, not less.

---

## 1. Kenya: `Protestant`, 15.78M, 33.4%

### What the cell holds: the census defines it, and ke2019.py's REVIEW has it partly wrong
The KNBS 2019 *Enumerator's Instructions Manual* (IPUMS copy,
`international.ipums.org/international/resources/enum_materials_pdf/enum_instruct_ke2019a.pdf`,
PDF pp.40-41, manual pp.35-36) defines the codes:

- **2 Protestant:** "Anglican Church of Kenya (ACK), Presbyterian Church of East Africa (PCEA),
  African Inland Church (AIC), Lutheran, Quakers (Friends), Methodists, Seventh Day Adventists
  (SDA), Salvation Army, Baptists, etc."
- **3 Evangelical Churches:** "CITAM, Redeemed, Full Gospel, Pentecostal, Deliverance, PAG, Kenya
  Assemblies, Winners, Apostolic church, Jehovah Witnesses, etc."
- **4 African Instituted Churches:** "Legio Maria, Israel, Church of Christ, Roho, African
  Independent churches, etc."
- **6 Other Christian:** "individuals who confess christian faith but are not affiliated to any of
  the above faiths, e.g. people who don't go to any church yet they consider themselves
  Christians."

`taxonomy/ke2019.py` gets three things wrong against this. It puts the Africa Inland Church and
the Baptists in `Evangelical Churches`; the manual puts them in Protestant. It also describes
`Other Christian` as Adventists, Witnesses, LDS and Quakers. In fact SDA and Quakers are in
Protestant, the Witnesses are in Evangelical, and `Other Christian` means Christians with no
church. That last point bears on the node too: by zm2022's rule, an "unaffiliated Christian"
answer belongs on bare `christianity`, not `christianity.other`. Not edited; flagged for a fix.

### Source: Global Flourishing Study wave 1 (CC BY, on disk)
- `COUNTRY` 10 = Kenya, 11,389 rows, 10,192 Christians. Only **1.6% named no church**, against the
  Afrobarometer's 10-38% "Christian only" (climbing by round, the Botswana problem), so GFS is the
  instrument for levels here, as it was for Tanzania.
- **`REGION1_Y1` 1001-1047 are the 47 counties in alphabetical order.** That reading comes from the
  pattern, not the codebook: 1016 Kisii 45% and 1034 Nyamira 62% SDA, 1036 Nyeri 29% Presbyterian,
  1026 Meru 15% Methodist, 1045 Vihiga 80% Pentecostal (PAG), 1039 Taita-Taveta 57% Anglican,
  1007 Garissa 13 Christians, 1046 Wajir 24, 1030 Nairobi the largest sample. **Confirm against
  the .sav value labels before building** (first 40 MB of OSF `eadfm`, as ng.md did). No
  respondents in Isiolo, Lamu, Mandera, Marsabit or Samburu.

### Checks run
The GFS boxes that the manual puts in code 2 are Anglican, Presbyterian, Lutheran, Methodist,
Baptist, Adventist and **Independent/Evangelical**. The last is the Africa Inland Church: it peaks
in Kitui 86%, Kericho 77%, Elgeyo-Marakwet 66%, Makueni 61% and Machakos 55% of the pool, which is
AIC country in Ukambani and among the Kalenjin.

| check | result |
|---|---|
| level: census Protestant vs the GFS pool, both weighted by census county population (42 counties) | **34.6% vs 34.7%** |
| county rank, census Protestant share vs GFS pool share (37 counties with 30+ GFS Protestants) | Spearman +0.33, p 0.046 (weak, but the right sign) |
| Evangelical column vs GFS Pentecostal + JW | +0.67, p 7e-6 |
| split-half on GFS respondents, each church as a share of the pool (200 halvings, 37 counties) | Anglican +0.88, Ind/Evangelical +0.88, Presbyterian +0.84, Adventist +0.81, Methodist +0.67, Baptist +0.65 |
| Afrobarometer R4-R7 by province, as a witness | Nyanza SDA 25% of respondents; Central Presbyterian 7.0%, highest; Eastern Methodist 7.5%, highest (Meru); Western Quaker 4.4% with Presbyterian 1.0% |

National composition of census Protestants, from GFS county shares weighted by census Protestants:

| GFS box | share | people | target node |
|---|---:|---:|---|
| Anglican | 26.3% | 4.15M | `christianity.anglican` |
| Adventist | 23.3% | 3.68M | `christianity.adventist.sda` (as tz, ug) |
| Independent/Evangelical (AIC, Salvation Army etc.) | 23.3% | 3.68M | **stays** on `christianity.protestant` |
| Presbyterian | 17.2% | 2.71M | `christianity.reformed.presbyterian` (but see the western caveat) |
| Baptist | 6.1% | 0.96M | `christianity.baptist` |
| Methodist | 3.2% | 0.51M | `christianity.methodist` |
| Lutheran | 0.4% | 0.06M | leave on parent (too small to place) |

### Western Kenya caveat (a judgement call)
GFS's list has no Quaker box. Kakamega and Bungoma are each **45% "Presbyterian"** inside the
pool, and Vihiga 31%. That is the Friends heartland, and the PCEA is weak there. The Afrobarometer
finds Western 4.4% Quaker and only 1.0% Presbyterian. So those ~0.66M "Presbyterians" are almost
certainly Friends. Options: (a) leave the Western counties' GFS Presbyterian share on
`christianity.protestant`, which is the safe choice and what I'd do; (b) send it to
`christianity.friends`, which already exists. The GFS card never offered "Friends", so (b) would
be our reading of the answer, not the respondents' own. Turkana (37%) and Baringo (36%)
Presbyterian are also high. PCEA has missions in both, but the samples are 59 and 42.

### Proposal
- For each county, the census Protestant count times the GFS share of each named church in that
  county's pool. Where a county has fewer than ~40 GFS Protestants (Garissa, Wajir, Tharaka-Nithi,
  Kwale, Vihiga) or none (Isiolo, Lamu, Mandera, Marsabit, Samburu), use the pool of the former
  province (`REGION2_Y1`, 8 codes) instead.
- Rows `derived`, `parent_column` = Protestant. Fail the build if the GFS pool level drifts more
  than ~3 points from the census Protestant share.
- **What could contradict it:** the census level (passes, 34.6 vs 34.7); the Afrobarometer's
  province pattern (passes for SDA, Anglican, Methodist; fails for Presbyterian in Western, hence
  the caveat); church rolls. ACK claims ~5M and PCEA ~4M. SDA rolls (East and West Kenya Union
  Conferences) were not fetched (adventiststatistics.org returns 403).
- **Caveats:** GFS samples adults (2022-23) and the census counts everyone in 2019, so the split
  assumes the same church mix among children. County samples are 3-257 Protestants.

Aside (not in this brief): the same GFS file would split Kenya's `Evangelical Churches`
(9.65M) the same way. GFS Pentecostal/Charismatic is 33.2% of Christians against the column's
23.9%, so some GFS Pentecostals sit in the census's AIC column. Worth its own look.

---

## 2. Malawi: `christianity.other`, 36.0% of the country, from two census cells

`mw2018.py` sends two cells to `christianity.other`: `Seventh Day Adventist/Baptist/Apostolic`
(1,644,829, 9.4%) and `Other Christian Denominations` (4,666,337, 26.6%). zm2022.py's note says a
residual like the second belongs on bare `christianity`, not `christianity.other`. That is a
separate fix.

### Source: Afrobarometer R4-R9 (on disk, open)
- 9,607 Malawian answers. `REGION` is only North, Central, South in every round; `DISTRICT` is
  filled in R4 only (110 of 801, 271 of 2,327 in the two cells), so **placement is 3 regions at
  best**.
- Level check, each answer mapped to its census cell: the SDA/Baptist/Apostolic answers are 6.9-9.2%
  of respondents by round (census 9.4%); the other-Christian answers 19.2-26.2% (census 26.6%);
  CCAP 15.5-19.3% (census 14.2%); Catholic 18.3-21.7% (census 17.2%). The cells line up.

### Inside `Seventh Day Adventist/Baptist/Apostolic` (801 respondents)

| | pooled | by round R4..R9 | North / Central / South |
|---|---:|---|---|
| SDA | **65.3%** | 42, 75, 71, 58, 59, 75 | 72 / 57 / 69 |
| Baptist | **31.3%** | 47, 26, 29, 42, 27, 25 | 13 / 43 / 28 |
| Apostolic | 3.4% | 12, 0, 0, 0, 13, 0 | |

People: SDA ~1.07M -> `christianity.adventist.sda`; Baptist ~0.51M -> `christianity.baptist`;
Apostolic ~56k stays. **Witness:** the Adventists' own count for the Malawi Union was 494,977
baptised members at mid-2016 and 627,797 in Q2 2021, about 540k in 2018 by interpolation. Members
are baptised, mostly adults, and under-15s are ~44% of Malawians, so 1.07M adherents is the right
size. Sources: General Conference SID map PDF 2017
(`documents.adventistarchives.org/Resources/Maps/2017/2017SID.pdf`) and SID statistics Q2 2021
(`sidadventist.org/secretariat/statistics/statistics-archives/statistics-2nd-quarter-2021/`).
**The SID PDFs give membership by conference (North, Central, South Malawi)**, which is a
regional witness for the SDA share as well.

### Inside `Other Christian Denominations` (2,327 respondents)

| answer group (Afrobarometer boxes) | pooled | North / Central / South | people at the pooled share | target |
|---|---:|---|---:|---|
| Independent, Zionist, Last Church (of God), African Abraham, African International, Utopia, Nationality | **30.6%** | 52 / 35 / 17 | 1.43M | `christianity.africaninstituted` |
| Christian only | 27.7% | 11 / 29 / 33 | 1.29M | stays (parent) |
| Church of Christ | **17.3%** | 13 / 13 / 24 | 0.81M | `christianity.restorationist` (as zm2022) |
| Evangelical | 8.4% | 4 / 8 / 11 | 0.39M | `christianity.evangelical` |
| Jehovah's Witness | 8.3% | 4 / 10 / 8 | 0.39M | `christianity.witnesses` |
| Lutheran | 2.7% | | 0.13M | `christianity.lutheran` (or leave) |
| New Apostolic | 2.2% (R5-R6 only) | | 0.10M | leave (box offered in two rounds only) |
| Orthodox, Methodist, other | 2.8% | | | leave |

By round, Church of Christ runs 8-22% and AIC 18-43% inside the cell. The R5 Christian-only share
fell to 6.5% and AIC rose to 43%, so the two boxes trade places. Pool all rounds, and test AIC
against Christian-only as one box before drawing it (the playbook's swap trap).

### Proposal
- Within each district, split the two census cells at the shares of its region (North, Central or
  South), pooled R4-R9. Draw only the groups with 180+ respondents (SDA, Baptist, AIC group,
  Church of Christ, Evangelical, JW); everything else stays on the parent.
- `derived`, `parent_column` = the census cell.
- **What could contradict it:** the cell levels (pass); SDA conference rolls (fetchable);
  JW publisher and Memorial figures (not fetched; ~0.39M adherents would need Memorial attendance
  in the 300-400k range to fit).
- **Caveats:** 3 regions only. The North's 52% "Independent" is unexplained: no named AIC is known
  to be that big in the north, and it may be Livingstonia-area independents or a coding habit.
  Afrobarometer is adults; the census year 2018 sits inside the 2008-2022 pool.
- Not checked: whether the 2008 census or DHS Malawi have a longer list. DHS final reports pool
  "Seventh Day Adventist/Baptist" the same way, as far as I recall, so they would only witness the
  cell.

---

## 3. DRC: `Protestant`, 27.85% of household heads, hold

- The cell is the Église du Christ au Congo: 62-64 member communities, regionally fixed by the
  colonial mission comity system. Baptists (CBFC) in Kinshasa, Kongo-Central and Équateur;
  Disciples (CDCC, Mbandaka) in Équateur; Presbyterians (CPC, about 2.5M members in 2006 by its own
  count) in the Kasaïs; United Methodists in Katanga, Kasaï and Sankuru; CEPAC Pentecostals in
  Sud-Kivu; CBCA Baptists in Nord-Kivu; CECCA (Africa Inland Mission) in Ituri. The Anglican
  province is in Ituri, Kisangani and Bukavu.
- **No open source splits it.** sources/cd.md §1 already established this: Enquête 1-2-3 has one
  Protestant code; EDS 2023-24 has one `Protestante` code, and its recode file is behind DHS
  registration (ruled out); MICS-Palu registration; Afrobarometer, WVS and GFS do not cover the
  DRC. Pew 2008-09 *Tolerance and Tension* covered the DRC, but no denomination topline turned up.
  State Department reports cite only Pew's 48.1% Protestant (including Kimbanguists).
- Church-reported membership exists community by community (WCC profiles: CEC, CDCC, CPK, CPC;
  mostly 2006 figures). Placing it would mean placing each community in its comity area by prose,
  which is what the proxy memory calls "placing it somewhere plausible".
- **Recommendation: hold.** Placement is the hard part, not the national split. A national
  ECC-community split painted flat across 26 provinces would put Kasaï Presbyterians in Ituri. If
  Anita wants it anyway, the only defensible version is a few communities whose comity areas are
  whole provinces (CBCA in Nord-Kivu, CEPAC in Sud-Kivu, CPC in the Kasaïs). It would rest on
  2006-era claims, `modelled` at best, and needs her call.

## 4. Ethiopia: `Protestant`, 13.66M (2007), hold

- The cell is P'ent'ay: Mekane Yesus (EECMY, Lutheran; 8.31M members claimed in 2016, 15M
  in 2025 per Wikipedia, sourcing unclear), Kale Heywet (SIM), Meserete Kristos (Mennonite), Mulu
  Wongel and the Pentecostals.
- Surveys don't separate them. Afrobarometer R8-R9 (4,778 respondents) has Ethiopian Protestants
  answer "Evangelical" (6-7%), "Pentecostal" (3-9%) or "Christian only" (8-9%); Lutheran is
  0.1-0.8%, Presbyterian 0.1-0.6%, Baptist 0-1.5%. EDHS has one Protestant code. GFS does not
  cover Ethiopia.
- Church rolls are synod-based (EECMY synods are geographic), but no synod table was found in
  one search. Kale Heywet publishes no regional figures I found.
- **Recommendation: hold.** The payoff is low or negative: Kenya's and Eritrea's Protestants are
  the same node, and South Sudan is bare `christianity`. A national Lutheran share painted over
  Wolaita (the Kale Heywet heartland) would be wrong where most Protestants live. Revisit only if
  Kenya's Protestant is split and the Moyale border starts to look odd. Few people live along it.

---

## Sources
- KNBS 2019 Enumerator's Instructions Manual (IPUMS): https://international.ipums.org/international/resources/enum_materials_pdf/enum_instruct_ke2019a.pdf
- ILO catalogue, KPHC 2019 P17 code list: https://webapps.ilo.org/surveyLib/index.php/catalog/8554/variable/FA_KEN_CENSUS_2019_FULL/VA24
- Global Flourishing Study wave 1 (OSF, CC BY): https://osf.io/download/vrejf/ (on disk)
- Afrobarometer merged R4-R9: https://www.afrobarometer.org/data/merged-data/ (on disk)
- SID Adventist statistics 2016/2017: https://documents.adventistarchives.org/Resources/Maps/2017/2017SID.pdf ; Q2 2021: https://sidadventist.org/secretariat/statistics/statistics-archives/statistics-2nd-quarter-2021/
- WCC/ECC profiles: https://oikoumene.org/id/node/15838 (CEC), https://oikoumene.org/id/node/15837 (CDCC), https://www.oikoumene.org/ko/node/15841 (CPK); CPC: https://en.wikipedia.org/wiki/Presbyterian_Community_in_Congo
- EECMY: https://en.wikipedia.org/wiki/Ethiopian_Evangelical_Church_Mekane_Yesus
- DRC Pew via State Dept IRF: https://www.ecoi.net/en/document/2073974.html
