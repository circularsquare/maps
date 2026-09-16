# Muslim branches and Buddhist schools: the sweep for sources, 2026-09-14

Anita asked whether anything could divide the Muslims and Buddhists the map draws on the bare
family node, *"ideally at at least like admin1 level"*, for the countries where they are most
numerous. Five scouts and the lead swept every built country holding 300,000 or more of either.
**This file is the record, including the negatives, so the sweep is not re-run from zero.**

## The answer

**No source divides Muslims by branch or school, or Buddhists by school, below national level in
any country swept, with one exception that cannot be used** (Pew 2021 India, six state-groups;
see India below). Nothing reaches a madhhab anywhere except Türkiye, which is already drawn.

What exists is national: **Pew's *The World's Muslims* (2012) Q31** for 20 built countries, Pew's
*Religion in India* (2021) for India's Muslims and Buddhists, and Pew's *Tolerance and Tension*
(2010) for four African ones. Those are candidates for the national estimate layer (spec §15),
not for dots. **Eleven rows were added on 2026-09-14** for Pakistan, Bangladesh, Egypt, Jordan, Kenya
and Ghana (`sources/estimates.md`); India's Pew 2021 split went into its dots instead.

## Decided, 2026-09-14 (Anita, after reading this file)

- **India is split from Pew 2021's six regions.** Sunni, Shi'a and Ahmadiyya go to their nodes;
  "some other sect", "no sect in particular" and "don't know" stay on `islam`. **Drawn
  2026-09-14** (`sources/in.md` §8): `islam` 172.2M to 73.2M, Sunni 90.5M, Shia 8.5M, with 7.2M
  left undivided where Pew interviewed nobody. Anita finds the step at the region edges (Bihar, the
  South) awkward and accepts it. **Ahmadiyya (1.36M) was drawn and then folded back into `islam`**
  the same day, on Anita's call after the check in the next bullet ("lets move ahmadiya back for
  now"); Sunni and Shi'a did not move.
- **India's Ahmadis, checked 2026-09-14.** No census of independent India counts them (C-01's
  119 write-ins, `in.md` §4). The only national figure with any standing is **about 150,000**, which
  the US State Department's religious freedom reports for 2021, 2022 and 2023 attribute to unnamed
  "media reports" (ecoi.net documents 2073984, 2091881, 2111881; the 2019 report omits Ahmadis).
  Journalism says "over a lakh" (The Tribune, 2018) and "1.5 lakh" (Ullekh NP, 2023), unattributed.
  **The "60,000 to 1 million" range traces to Amir Mir in Outlook (2022)**, whose same piece claims
  "the 2001 census counted roughly 20,000 Ahmadis in Qadian"; the census records no Ahmadis, and all
  of Qadian had 3,065 Muslims in 2011 (CPS India). Defensible range: about 100,000 to 150,000, both
  ends unattributed. **Pew's regional pattern fails against places whose answer is known**: East
  reads 0 where Odisha has organised Ahmadi villages (Kerang, Soro, Bhadrak); North's 0 is what
  sampling predicts anyway (Qadian's ~3,000 in 12.6M Muslims is 0.2 expected respondents); South's
  1.17M is 8 to 10 times the whole national estimate, and 15 respondents in one region reads as one
  or two sampling points in an Ahmadi locality. Pew publishes nothing to show the clustering, so that
  is an inference.
- **A large unspecified share is fine and expected** (spec §2.7a). The size of the parent's
  remainder is no longer a reason to withhold a split; the named share still has to mean what the
  card says (the Egypt test below).
- **Pakistan is rebuilt on the 2023 census tables, at district** (`sources/pk.md` §7b), keeping
  §14.4's ceiling though tehsil is printed. **Drawn 2026-09-14** (`sources/pk.md` §9): 136
  districts, 240,458,089 people, Sikhs and Parsis drawn for the first time. Still no sect.
- **Izady's Gulf/2000 maps were checked and give nothing usable** (section "The maps online"
  below): religion by ethnic ascription, no numbers below national, licence bars copying; 16 of 48
  maps list sources, none statistical for a built country; the one concrete Iran lead (Razmara,
  1949-53) Anita ruled too old.
- **National Sunni/Shia rows from Pew 2012** went on the estimate layer for pk, bd, eg, jo, ke and gh
  (Anita: "national estimates look good"). Kenya's volunteered Ahmadiyya 4 was left off.
- **Skipped**: the Ghana 2010 and IFLS-5 registrations.
- **Direction, not yet a decision: ethnic assignment.** Anita, after seeing that Pew's own 2009
  country estimates are ethnographic ascription (Appendix B, p. 38: consultants plus the World
  Religion Database's Sunni/Shia makeup for each of ~4,300 ethnolinguistic groups): *"the surveys
  aren't great. it seems like we will have to do the ethnic assigning, i dont think theres a way
  around it."* Nothing is built. Whoever designs it has to square it with spec §2.6 (a school is
  never assigned from outside the source), §14.5 (religio-ethnic derivation, and China's Dai as the
  precedent) and §14's safety rules, and should expect ascription to disagree with self-description:
  Azerbaijan is 65-75% Shia by Pew 2009's ascription and 37% Shia plus 45% "just a Muslim" when asked
  (Pew 2012). **Her shape for it, the same day:** a tier of its own beside `modelled`, which says *what*
  was assigned rather than carrying a share, e.g. *"assigned from ethnicity: Shia vs Sunni"*: *"we have
  data modelled 65% or whatever. we should also specify what we assigned based on ethnicity."* Note that
  China's §14.5 rows are `derived` today (`sources/cn.md` §7 table), so a new tier would move them too.

## Decided, 2026-09-15

- **Russia's Buddhists follow Mongolia's onto `buddhism.vajrayana`.** They were inconsistent:
  `mn2020.py` files the census's `Будда` on `buddhism.vajrayana` (its REVIEW note argues why, against
  the letter of spec §2.6), while `ru2012.py` files Arena's "I profess Buddhism" (655,196, 0.46%) on
  bare `buddhism`. Anita: *"we can match mongolia to vajrayana i think."* The heartland carries it:
  Tuva 208,050, Buryatia 193,760, Kalmykia 100,442 and Zabaykalsky 62,758 are 565,010, **85.7%** of
  the drawn subjects' 659,032. (Arena's national column is 655,196; this line first said 86.2% and
  90,186 against that, corrected the same day.) The other 94,022 are one or two respondents per
  subject at n~720 (Moscow 16,263, Moscow Oblast 10,656). **Revised the same day: only those four
  subjects go to `buddhism.vajrayana`; the other 22 subjects' Buddhists stay on `buddhism`** (14.3%,
  against WRD's 16.4% Mahayana share of Russia's Buddhists), using the survey's own geography rather
  than an even spread. Anita: *"we should use the surveys geography. why would we not?"* Russia and
  Mongolia both carry the "assigned from ethnicity" tier (spec §2.6a). **Not done yet**: no drawing
  on 2026-09-15 while rd-super runs.
- **Census Buddhists in Thailand, Myanmar, Cambodia, Sri Lanka, Laos and Bangladesh go to
  `buddhism.theravada`, less an unspecified remainder: spec §2.6a, reversing §2.6.** The remainder is
  for Chinese- and Vietnamese-descended Buddhists and is spread as one share per country, because
  nothing places them. Sized from WRD's shares (9.1% of Thai Buddhists). A per-country check found no
  drawn unit mostly non-Theravada except a 2,300-Buddhist village in Bokeo, Laos; Mae Hong Son,
  Ratanakiri and Preah Sihanouk run about a quarter and get a note only. **South Korea was added the
  same day** (census Buddhists to `buddhism.mahayana`, 0.7% remainder), and "assigned from
  ethnicity" is to be built as its own tier first. **Not built.**

## Where the undivided people are (counts.json, 2026-09-14)

Thousands of people, on the bare node. `total` is the family.

| islam | bare | total | | buddhism | bare | total |
|---|---:|---:|---|---|---:|---:|
| id | 207,176 | 207,176 | | th | 61,747 | 61,747 |
| pk | 200,362 | 200,553 | | mm | 45,185 | 45,185 |
| in | 172,245 | 172,245 | | lk | 15,196 | 15,196 |
| bd | 130,204 | 130,204 | | kh | 15,101 | 15,101 |
| ng | 103,852 | 103,852 | | in | 8,442 | 8,442 |
| eg | 101,173 | 101,173 | | kr | 7,619 | 7,703 |
| uz | 38,059 | 38,059 | | vn | 6,802 | 8,297 |
| et | 25,037 | 25,037 | | my | 6,066 | 6,066 |
| cn | 23,137 | 23,137 | | la | 4,193 | 4,193 |
| my | 20,610 | 20,610 | | us | 2,556 | 3,596 |
| kz | 13,297 | 13,297 | | np | 2,367 | 2,367 |
| ci | 12,453 | 12,453 | | id | 1,703 | 1,703 |
| jo | 11,770 | 11,770 | | sg | 1,074 | 1,074 |

World: 1,178m of 1,306m Muslims and 182m of 280m Buddhists on the bare node. The first six
Islam rows are 78% of the Muslim figure; the first four Buddhism rows are 75% of the Buddhist one,
and are the four countries spec §2.6 decided to leave undivided. China was not swept (§14.5-§14.16
own it).

## The rule this sharpens: a sect item that replicates is still not a sect geography

§11af found the Arab Barometer's sect item measured "which label a respondent reaches for", with
wholly-Maliki Morocco at 16% Maliki. **Every survey found here has the same shape**, and the
Global Flourishing Study adds the piece §11af could not show. Its `REL4` Sunni share in **Egypt**
replicates between random halves across 22 governorates at **+0.90** (bar +0.43), and Egypt's
Muslims are Sunni. A split-half test (§9bi) proves a regional pattern is not sampling noise; it
cannot prove the pattern is the thing named on the card. **For a sect or school item, check the
no-sect share and check the pattern against a country where the answer is known, before the
split-half.** Egypt is that country for Sunni; Morocco is it for Maliki.

## National figures that exist

### Pew, *The World's Muslims: Unity and Diversity* (2012), Q31

`https://www.pewresearch.org/wp-content/uploads/sites/20/2012/08/the-worlds-muslims-full-report.pdf`,
table on printed p. 30, re-read at source by the lead. Question: *"Are you Sunni (for example,
Hanafi, Maliki, Shafi, or Hanbali), Shia (for example, Ithnashari/Twelver or Ismaili/Sevener), or
something else?"* The schools are examples inside the Sunni option, so **no row reaches a school
(§2.6), and no Shia row reaches `jaafari`**. Microdata behind a Pew account. Asterisked rows are
re-tabulations of *Tolerance and Tension* (2008-09 fieldwork), not new interviews.

% of Muslims, built countries only:

| cc | Sunni | Shia | something else | just a Muslim | nothing/DK/ref |
|---|---:|---:|---:|---:|---:|
| jo | 93 | 0 | 0 | 7 | 0 |
| bd | 92 | 2 | 0 | 4 | 2 |
| tr | 89 | 1 | 5 | 2 | 4 |
| eg | 88 | 0 | 0 | 12 | 0 |
| th (5 southern provinces only) | 87 | 0 | 0 | 3 | 8 |
| pk | 81 | 6 | 1 | 12 | 0 |
| my | 75 | 0 | 0 | 18 | 7 |
| ke* | 73 | 8 | 4 | 8 | 7 |
| et* | 68 | 2 | 2 | 23 | 4 |
| gh* | 51 | 8 | 16 | 13 | 11 |
| iq | 42 | 51 | 0 | 5 | 1 |
| ug* | 40 | 7 | 4 | 33 | 16 |
| lr* | 38 | 9 | 10 | 22 | 21 |
| ng* | 38 | 12 | 5 | 42 | 4 |
| ba | 38 | 0 | 0 | 54 | 7 |
| ru | 30 | 6 | 0 | 45 | 19 |
| id | 26 | 0 | 5 | 56 | 13 |
| xk | 24 | 1 | 2 | 58 | 15 |
| kg | 23 | 0 | 0 | 64 | 12 |
| uz | 18 | 1 | 0 | 54 | 26 |
| kz | 16 | 1 | 0 | 74 | 10 |
| al | 10 | 0 | 13 | 65 | 12 |

Pakistan's sample excludes FATA, Gilgit-Baltistan, AJK and unstable parts of KP and Balochistan
(p. 125, 82% of adults), so its Shia 6% is a floor. Malaysia's Shia 0% is under a fatwa ban and is
not evidence of absence. Mozambique was withheld (footnote 9); the Philippines, Turkmenistan, Côte
d'Ivoire, Guinea, Benin and Malawi were not surveyed.

**If any of these become §15 rows**, the no-sect column is what decides it. Indonesia, Kazakhstan,
Kyrgyzstan, Uzbekistan, Albania, Kosovo and Bosnia have a majority on "just a Muslim" or no answer
and would put a minority figure on `islam.sunni` beside a parent that is really Sunni. Nigeria's
Shia 12% is above other published estimates and is a §14 question in its own right (`ng.md` §5
refused it subnationally).

### Pew, *Religion in India: Tolerance and Segregation* (2021)

Topline `https://www.pewresearch.org/wp-content/uploads/sites/20/2021/06/PF_06.29.21_India_topline.pdf`,
p. 23, re-read at source by the lead. Fieldwork 2019-20; excludes Kashmir Valley, Manipur, Sikkim
(report p. 16).

**QSECT, asked of Muslims**, % :

| | N | Sunni | Shi'a | other sect | no sect | Ahmadiyya (vol.) | DK/ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| India | 3,336 | 55 | 6 | 2 | 14 | 1 | 22 |
| Northeast | 512 | 32 | 5 | 4 | 20 | 0 | 38 |
| North | 655 | 80 | 7 | 1 | 6 | 0 | 6 |
| Central | 202 | 79 | 5 | 1 | 2 | 0 | 12 |
| East | 1,016 | 43 | 2 | 1 | 18 | 0 | 35 |
| West | 579 | 50 | 11 | 4 | 17 | 1 | 17 |
| South | 372 | 38 | 6 | 7 | 24 | 4 | 22 |

North is Chandigarh, Delhi, Haryana, HP, J&K, Ladakh, Punjab, Rajasthan (p. 123); East is Bihar,
Jharkhand, Odisha, West Bengal (p. 33); West is Goa, Gujarat, Maharashtra (p. 43); South is AP,
Karnataka, Kerala, Tamil Nadu, Telangana, Puducherry (p. 49). Central and Northeast lists not read.
**Not drawn** at the time of this sweep: no-sect plus DK runs from 12% in the North to 58% in the
Northeast, which is the Morocco shape again, and six state-groups of 200 to 1,000 Muslims is coarser
than `in`'s sub-districts by two orders. **Reversed the same day by spec §2.7a and drawn**; Central
and Northeast lists read off the p. 16 map; see `in.md` §8.

**QBUDDHIST, asked of Buddhists**, national only, N=719: Mahayana 2, Theravada 1, **Navayana 48**,
some other order 1, no order in particular 13, DK/refused 35. The only people-counting Buddhist
school figure found for any built country in the sweep. `buddhism.navayana` does not exist.

### Pew, *Tolerance and Tension* (2010), Q37/Q38

Report p. 21, topline p. 158 (read by the West Africa scout; the sect rows agree with the 2012
re-tabulation above). Adds **Ahmadiyya** and Sufi order membership. Ghana, 339 Muslims: Ahmadiyya
16, Tijaniyya 27, Qadiriyya 5. Nigeria, 818: Ahmadiyya 3, Tijaniyya 19, Qadiriyya 9, and no Izala
box. Uganda, 321: Sufi DK 59. Liberia, 279: Ahmadiyya 10, Tijaniyya 25.

## The Global Flourishing Study's `REL4` (on disk, `data/raw/gfs/`)

Found by the lead. `REL4_Y1`, *"Which of the following sects or schools do you most identify
with?"*, asked of every `REL2_Y1 = 2`: 1 Sunni, 2 Shi'a, 3 Sufis, 4 Bohra, 5 Ahmadiyya, 6 Khojas,
7 Quranists, 96 other, 97 no sect in particular (just Muslim). Labels from the codebook sheet linked
at `https://www.cos.io/gfs-wave-data`. Wave 1, weighted by `ANNUAL_WEIGHT_C1`, every country below
recruited face to face (`MODE_RECRUIT` 1; Indonesia mostly).

| country | Muslims | Sunni | Shi'a | other codes | just Muslim | regions with 30+ Muslims, split-half of Sunni |
|---|---:|---:|---:|---|---:|---|
| Egypt | 4,594 | 25.6 | 0 | | 72.7 | 22, **+0.90** (bar +0.43) |
| Indonesia | 6,364 | 6.0 | 0.4 | | 92.3 | 27, +0.48 (bar +0.38) |
| Nigeria | 2,940 | 48.7 | 2.1 | other 16.9 | 31.8 | 18, +0.77 (bar +0.48) |
| India | 1,022 | 49.5 | 1.8 | Quranists 8.5, Ahmadiyya 1.9 | 34.2 | 7, +0.88 (bar +0.80) |
| Kenya | 1,055 | 70.7 | 4.8 | | 22.5 | 5 |
| Tanzania (not built) | 2,936 | 24.9 | **21.4** | other 13.3 | 38.1 | 22, Shi'a +0.89 |
| Türkiye | 1,336 | 53.5 | 0.9 | code 8 (not in the codebook) 3.7 | 39.9 | 19, +0.05 |

**Not usable.** Indonesia and Egypt fail on the no-sect share. Nigeria, India and Tanzania replicate
and still fail: Tanzania's 21% Shi'a and India's 8.5% Quranists (27% in one region) are not
credible, and say the per-country answer cards were not coded alike. Egypt is the demonstration in
the section above. `REL9_Y1` is a Japanese sect list (Jodo, Jodoshin, Rinzai, Soto, Ji, Nichiren)
and is Japan's alone. Scripts: the lead's scratchpad, not kept; re-derive from the CSV.

## Country by country

### Pakistan
- **2023 census** (`https://www.pbs.gov.pk/wp-content/uploads/2020/07/National-Census-Report-2023.pdf`):
  no Shia, Sunni or sect anywhere; the only Muslim sub-category is Qadiani/Ahmadi. **The 2023
  district and tehsil religion tables are live on pbs.gov.pk**, which reverses `pk.md` §7; see
  `pk.md` §7b. No sect in them.
- **PDHS 2017-18** (`dhsprogram.com/pubs/pdf/FR354/FR354.pdf`): religion only on the fieldworker
  questionnaire (p. 531).
- **PILDAT, *Sectarian Conflict in Gilgit-Baltistan* (2011)**, pp. 12-13: valley shares as flat 100s
  (Hunza 100 Ismaili, Nagar 100 Shia, Baltistan 96-98 Shia), footnoted to a web article and an
  editorial. Compiler estimate, and GB has no census religion data to split anyway.
- Gallup Pakistan/Gilani, PIPS, PSLM: no sect-by-province table found (search only). The "Barelvi
  50%, Deobandi 20%" figures trace to commentary with no survey behind them.
- WVS wave 7 Pakistan (2018, 1,995 cases) has a country-specific denomination code, not seen; licence
  form.

### India
- **Drawn 2026-09-14** from Pew 2021 QSECT by region, on spec §2.7a: 90.5M Sunni, 8.50M Shi'a,
  73.2M left on `islam`; see `in.md` §8. Ahmadiyya (1.36M) was drawn the same day and folded back
  into `islam` on Anita's call, because Pew's regional pattern fails against known places (Decided,
  "India's Ahmadis").
- Census 2011 Annexure: write-ins, not a sect count (`in.md` §4, not redone).
- Lokniti: the Jharkhand 2024 state questionnaire has one Muslim code; NES data behind a CSDS request.
  Allie (Carnegie, 2024) surveyed ~2,000 UP Muslims with a sect item and printed no shares.
- NFHS, IHDS, NSS: not opened.

### Bangladesh
- Nothing below national. The US State Department's *"according to the 2022 census, Sunni Muslims
  constitute approximately 91 percent"* is the Department's gloss; the census asks no sect.
- 1911 Bengal tables: no Muslim sect table.

### British India censuses (historical only, and pre-partition)
- **1881**, Imperial Table IIIB (`ruralindiaonline.org` report scan, pdf p. 1093): province tier, Sunni
  46.8m, Shiah 0.81m, Farazi, Wahabi, no detail 5%.
- **1911 Punjab**, Table VI-A (archive.org `in.ernet.dli.2015.62718`, pdf pp. 47-49, OCR unusable,
  read off the scan): 29 districts and 22 states. Shia 247,529 (2.0%), Ahmadi 18,695, Ahl-i-Hadis
  39,083, and Hanafi written in by 482 people.
- **1911 Jammu & Kashmir**, Table VI (archive.org `in.ernet.dli.2015.449221`, pdf pp. 27-28): 13
  districts, Sunni/Shia, Shia including 24,910 Aga Khani Ismailis. Gilgit 11,987 Sunni / 11,088 Shia;
  Ladakh (then with Skardu and Kargil) 43,574 / 106,496. **The only self-reported sect count ever made
  for today's Gilgit-Baltistan**, which the 2017 census does not cover either.
- 1911 NWFP: five districts' Shia by census year in the report text, OCR garbled.
- No sect tables in the 1911 all-India, Bombay, Hyderabad or Baluchistan volumes, or Punjab and UP
  1921 (OCR search; can miss a garbled title).

### Indonesia
- SP2020 long form item 405 *Agama*: Islam is code 1, undivided.
- **Ormas affiliation is the only split with coverage, and it is national.** Alvara 2016 (1,626
  Muslims, 34 provinces): affiliated NU 50.3, Muhammadiyah 14.9; members NU 36.1, Muhammadiyah 6.3, no
  organisation 54.6 (via the founder's blog). LSI Denny JA 2023 (1,200): NU 56.9, Muhammadiyah 5.7
  (news report). No provincial tables. No node exists; NU affiliation is not `shafii` (§2.6).
- **IFLS-5** (RAND, 2014-15) Book 3A reportedly asks organisation affiliation, 13 provinces; questionnaire
  not read (rand.org 403), registration required.
- Shia and Ahmadiyya: national claims only, 80,000 to 500,000 Ahmadis and "about a million" Shia from
  interested parties. Balitbang Kemenag's 2016 city studies are anecdotal.

### Malaysia, Philippines, Thailand
- Malaysia: census Islam undivided; Shia claims run 2,000 (JAKIM, unverified) to 250,000-300,000
  (*Afkar*, sourced to shianumbers.com).
- Philippines: PSA 2020 has Islam as one of 129 rows; not in Pew 2012.
- Thailand: census Islam undivided; the Religious Affairs Department mosque roll puts 1-2% of 3,479
  mosques as Shia (roll, via ISEAS).

### Egypt, Jordan
- Egypt: Afrobarometer R5 98.9% "Muslim only"; R6 not asked. Madhhab by governorate (Shafi'i Delta,
  Maliki Upper Egypt) exists only as prose claims, unverified. Arab Barometer not redone (§11af).
- Jordan: no census religion table (`jo.md`).

### Central Asia and Russia
- **Central Asia Barometer wave 1 has a sect item** (`Muslim`, Sunni/Shia/Ismaili), which §11ak found
  only for Tajikistan. Kazakhstan, 919 Muslims in 7 macro-regions: DK 43.3, Sunni 42.7, **Ismaili 5.0**
  (44 answers, in regions with no Ismaili community); Kyrgyzstan, 1,363 in 9: DK 79.1, Sunni 17.3;
  Uzbekistan phone re-contact `TelMuslim`, 387: DK 77.5, Sunni 17.1. Waves 2-14 have no sect item. Not
  usable.
- LiTS III `q922`: one Muslim code. Turkmenistan: nothing anywhere.
- Russia: Sreda Arena 2012 already splits (Sunni 1.66%, Shia 0.21%, "neither" 4.66% of respondents,
  so ~71% stays on the parent). Levada November 2012: only ислам.

### Sub-Saharan Africa
- **Censuses with one Islam cell**: Ethiopia 2007, Kenya 2019 (Table 2.30), Mozambique 2017 (Quadro
  11), Côte d'Ivoire 2021, Guinea 2014, Uganda 2014, Benin 2013, Malawi 2018, Liberia 2022.
- **DHS final reports, Islam undivided**: Ethiopia 2016 (FR328, no sect word in 551 pages), Ghana 2022
  (FR387), Uganda 2016 (FR333), Liberia 2019-20 (FR362), Guinea 2018 (FR353), Malawi 2015-16 (FR319),
  Benin 2017-18 (FR350), Côte d'Ivoire 2021 (FR385).
- **Ghana is the near miss.** The 2010 form (P09 code 7) and the 2021 form (P09 code 6) both print
  **Ahmadi** as its own answer. Every published table folds it into Islam, and in the 2021 10% microdata
  (GSS catalogue 110, V1097) code 6 has **0 cases**. Whether the 2010 microdata kept code 7 is unknown:
  GSS catalogue 51 returns HTTP 500, DataFirst 856 an empty page, IPUMS is blocked.
- **Afrobarometer** (on disk), % of Muslims answering `Muslim only`, weighted:

| | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|
| ci | | 99.4 | 98.5 | 99.4 | 96.2 | 99.9 |
| gn | | 98.5 | 94.7 | 99.1 | 99.0 | 99.4 |
| gh | 94.1 | 88.2 | 77.3 | 95.3 | 91.9 | 93.7 |
| ug | 94.1 | 97.2 | 94.3 | 99.2 | 93.9 | 99.5 |
| bj | 99.7 | 98.2 | 99.7 | 92.6 | 98.7 | 99.3 |
| mw | 80.6 | 90.1 | 92.2 | 95.3 | 94.8 | 95.9 |
| lr | 94.4 | 99.4 | 96.3 | 96.7 | 98.5 | 100.0 |

  Kenya 80-99, Mozambique 86-99.6, Ethiopia R8 96.1 and R9 99.4, Egypt R5 98.9. **No round's card has
  an Ahmadiyya answer.** Against Pew's 13% "just a Muslim" and 16% Ahmadiyya for Ghana, this measures
  how hard the interviewer probed. Not usable anywhere; Nigeria's refusal (`ng.md` §5) generalises.
- Nigeria outside the Afrobarometer: NDHS 2018 undivided; NBS GHS-Panel variable pages would not load;
  no NOI poll on sect; no state-level count of Tijaniyya, Qadiriyya or Izala anywhere.

## Buddhism

**Nothing counts lay Buddhists by school below national level, and nothing new touches §2.6.**

- **The four §2.6 countries, short check, none found**: Thailand 2010 census (Buddhism one cell),
  Myanmar 2014 Vol 2-C (seven categories), Sri Lanka 2024 press note, Cambodia 2019 final report.
  Pew's 2023 South and Southeast Asia questionnaire (Cambodia, Sri Lanka, Thailand, Malaysia, Singapore,
  Indonesia) offers only "Buddhist".

### Thailand's 7.9% Mahayana, checked 2026-09-15

Gemini found it on ARDA's national profile (`thearda.com/world-religion/national-profiles?u=220c`),
which cites Zurlo ed., *World Religion Database* (Brill, accessed September 2025): 2025 Buddhist
86.87%, Theravada 78.98%, Mahayana 7.90%, "Lamaist" 0. Anita asked what it rests on, and for a second
estimate of the Chinese count and of Mahayana, because it is unclear whether Sino-Thai descendants
simply become Theravada. **Nothing here changes §2.6.**

- **WRD's split is ethnic ascription at a fixed ratio.** Its methodology note
  (`worldreligiondatabase.org/static/downloads/wrd_methodology.e10266f1a9be.pdf`, pp. 14-15) gives each
  ethnolinguistic people a religion breakdown, compiled for the World Christian Encyclopedia 2nd ed.,
  and assumes it holds wherever that people lives. The ARDA page's chart data put Mahayana at **9.08%
  of Buddhists in 1900, 1950, 1970 and 2000 alike** (8.246 of 90.859; 8.225 of 90.631; 8.190 of
  90.239; 7.903 of 87.084). A century of immigration and assimilation does not move it, so the figure
  records the assumption, not the population. National only; WRD holds provincial data only "where
  available from censuses and surveys".
- **No independent count of lay Mahayana exists.** COW WRP (`data/raw/estimates/WRP_national.csv`)
  never splits Thai Buddhists by school: 1955 has Theravada 82.0% plus "other Buddhist" 9.6% from an
  encyclopedia it rates low, and 2000-2010 codes every Buddhist Theravada from State Department
  reports. Religious Characteristics of States (RCS-Dem 2.0, ARDA/OSF) gives 0.94% Mahayana in 2010
  with 17.3% "unspecified Buddhists", no per-point source, and WCE and WRD as its main sources. The
  National Office of Buddhism's 2023 roll has the Chinese sect at 138 monks and 281 novices and the
  Annam sect at 143 and 138, against 256,219 monks (read through search summaries; the onab pages
  answer 403). Kataoka, "Religion as Non-religion" (*Southeast Asian Studies* 1(3), 2012, p. 464):
  *"the exact number of Theravada and Mahayana lay followers among Buddhists is not known"*;
  laypeople *"are never classified according to sects"* (p. 468).
- **Practice does not mark a school.** Pew's 2022 survey (topline p. 35, QFIGURESi, "Do you currently
  pray or offer your respects to ... Guanyin"): Thailand **33%** of adults (N=2,504), **36% of
  Buddhists** (N=2,228), four times WRD's Mahayana share. Skinner (1957:313, via Morita, Nagoya
  *Studies in Language and Culture* 28(2)) found Thai-born Chinese worshipping at both Chinese
  temples and wats; Skinner 1973:408, *"Chinese religious sentiment is eclectic and syncretic rather
  than exclusivistic."* Suengprasit 2022 (*J. Yanasangvorn Research Institute* 13(2)) asked 12
  worshippers at a Hat Yai Chinese temple; 3 knew it was a Buddhist temple. The scouting agent reports a
  Pew ethnicity item for Thailand (QETHTHA, Thai / Chinese / ...); the topline prints no ethnic cut
  and the microdata needs a Pew account. Not re-checked.
- **The Chinese count behind the ratio has one real vote, not five.** The 7-7.5M (10-11%) cluster,
  Taiwan's OCAC, Zhuang Guotu for the PRC Qiaoban, the PRC Overseas Chinese Blue Book and
  Suryadinata/ISEAS, is **Skinner's 1955 share of 11.3% applied to later Thai populations**: OCAC's
  2004 figure is 64,197,000 × 0.113 = 7,254,261 exactly, which assumes no assimilation after 1955.
  OCAC's open-data CSV (`data.gov.tw/dataset/6483`) has a percent column that is Thailand's share of
  world overseas Chinese, not of Thailand. Wikipedia's 9.5M and "11 to 14 per cent" carry no citation
  and are Skinner's 11% beside the CIA Factbook's unsourced 14% (1982 to about 2014 editions).
- **The independent figures, by definition**: Chinese nationals 3.8% (1919) falling to about 1% (1970),
  household registration then census; Chinese spoken at home 231,350 (2000) and 111,866 (2010), UN
  Demographic Yearbook language table; self-identified Chinese about 2% or less, Asian Barometer wave 4
  (search snippet only; data by application); Joshua Project 1.7M, 2.4%. So descent by Skinner's 1955
  definition is 10-11%, self-identification about 2%, home language 0.2-0.4%. WRD's 5.6M sits with the
  Skinner lineage.
- **Nothing gives Chinese by province.** The 2000 census language tables list Chinese only for a few
  southern provinces (Yala 3.0%, Nakhon Si Thammarat 2.1%), and the archived Bangkok table has no
  Chinese row at all. The one route to a regional self-ID cut is **WVS wave 7 Thailand** (2018,
  n=1,500): Q290 code 764002 "China", with region and province in the file, behind the WVS download
  form (name and email). Its religion item offers only "Buddhism".

### The other countries, 2026-09-14 sweep

- **India**: Pew 2021 QBUDDHIST above, national. Census 2011 ST-14 gives tribal Buddhists by district
  but pools every tribe; ST-14A splits by tribe at state level only; SC-14 is state level. All three
  are §14.5 derivations.
- **South Korea**: MCST *2018 Religions in Korea* order claims (Jogye 12M, Taego 6M, Cheontae 2.5M,
  Jingak 800k; all Buddhist claims 24.79M against the census's 7.6M), via Bulkyo21 and BBS; the PDF
  itself was not reached. Roll, national, and every major order is Mahayana, so it carries no split.
  Korea Gallup 1984-2021 asks Buddhist only. **Pew East Asia 2024, p. 31: "We did not ask respondents
  to say which of the three major strands of Buddhism they follow."**
- **Vietnam**: 2009 and 2019 censuses do not cross religion with ethnicity; the 2019 survey of the 53
  minorities may, not read (gso.gov.vn DNS failure).
  **Khmer counts are not a Theravada count, checked 2026-09-15.** Khmer by province is published
  (2019: Trà Vinh 318,231, Sóc Trăng 362,029), but against the map's 2009 Buddhist cells Trà Vinh has
  498,930 Buddhists, about 180,000 more than its Khmer (Kinh Mahayana too), and Sóc Trăng 332,392,
  fewer than its Khmer. The years differ; the gaps are too large for that. Without a religion by
  ethnicity table the Khmer count cannot weight anything. The office's 2019 census files now turn
  up under `nso.gov.vn` in search results, so the minorities survey is worth one retry there (not
  probed).
- **Nepal**: *Religions in Nepal* 2021 Table 5.5, Buddhists by caste/ethnicity, national (Tamang
  1,391,866, Magar 314,745, Gurung 296,124, Sherpa 128,341, Newar 120,812); a derivation, and national.
- **Malaysia**: 2020 key findings Table 6 is religion by state, no ethnic cross.
- **United States**: Pew RLS has no Buddhist children; ASARB's Buddhist rows are already drawn.
- **Laos, Indonesia, Singapore**: nothing below Buddhist.

## Gated routes, none taken

| route | what it might give | wall |
|---|---|---|
| WVS wave 7 microdata | **nothing for sect**: the open online tool shows pk, bd, id, ng, eg and my with one `Islam; nfd` code (read 2026-09-14, Pakistan section) | licence form |
| Pew datasets (2012, 2021 India) | the Q31 and QSECT items at the survey's own region codes | Pew account |
| Ghana 2010 census microdata | whether P09 code 7, Ahmadi, survived | GSS registration |
| IFLS-5 | NU / Muhammadiyah affiliation, 13 provinces | RAND registration |
| DHS microdata | nothing on sect in any report read | registration |

Expected yield is low for all of them: every survey sect item read in this sweep fails on the
no-sect share, and none of these is likely to differ. **Anita, 2026-09-14: skip the Ghana 2010 and
IFLS-5 registrations.**

## Groups with no node that the sources name

Ismaili, Bohra, Khoja, Noorbakhshi, Quranist, Ahl-i-Hadith, Barelvi, Deobandi, Izala, Tijaniyya,
Qadiriyya, Mouride, Nahdlatul Ulama, Muhammadiyah, Navayana. None has subnational data.

## §14 flags

- **Pakistan**: a Shia layer would mark Kurram, the Quetta Hazara, D.I. Khan and Gilgit-Baltistan,
  all repeatedly attacked. The 2023 tables print Ahmadis by tehsil.
- **Indonesia**: Shia (Sampang, 2012) and Ahmadis (Cikeusik, 2011; Kuningan's ban on Jalsah Salanah,
  December 2024). Kemenag's own studies locate Shia communities city by city.
- **Malaysia**: Shia teaching banned by fatwa and watched by Special Branch.
- **Nigeria**: Shia refused subnationally (`ng.md` §5); whether even a national Shia figure is
  acceptable is open.
- **Egypt**: Pew 2012 p. 89, *"In Egypt and Morocco, the prevailing view (52% and 51%) is that Shias
  are not Muslims."* No source offers a Shia placement, so nothing to decide.

## The maps online: Izady's Gulf/2000, checked 2026-09-14

Most online maps of Islam's branches colour each area by its plurality branch and carry no share,
and many trace to Michael Izady's Gulf/2000 maps at Columbia. The Austrian Interior Ministry's *Atlas
MENA* source notes (2017, ecoi.net `90_1487772308`) say so outright: p. 53 depicts Pakistan's
Sunni-Shia mixed areas "according to ... (Izady, 2016)", and p. 65 calls his maps "the main source"
for Arabia. Anita asked for them to be checked after reading this file.

- **The index** (`https://gulf2000.columbia.edu/maps.shtml`) has 43 religion maps, among them
  Pakistan, Iraq (2000, 2015, Central Iraq 2017), Jordan and the Levant, Nigeria, Afghanistan, Iran,
  Syria, Lebanon, Yemen, Sudan, South Sudan and western Libya, plus regional maps of Central Asia, the
  Caucasus, Kurdistan, West Africa and Islam's branches. Nothing for Egypt, Türkiye, India or
  Bangladesh alone. (The first version of this paragraph said nothing for South Asia or Africa; wrong.)
- **Opened**: Pakistan, Iran, Central Asia, Jordan, the world Islam branches map. **None prints a
  share below national, a provincial table, or a source list.** Central Asia's one table is the whole
  region in millions.
- **Method, in his words** (Central Asia map): *"Figures for religion can only be arrive at by knowing
  the religious affiliation of a group and then assertion their number from various ethnic
  statistics"*, counted *"by cultural criteria rather than confessional"*. The index promises
  methodology articles and bibliographies with each map; none is posted.
- **Licence**: *"NOT in public domain ... may not be used for reposting, reprinting or copying without
  a written permission of Dr. Izady"*.
- **Pakistan**: Sunni 76.3%, Shi'a 19.1% with the Zikris counted as Shia, undated (© 2007-2016,
  version 17). Its zones are plurality and population density, not shares.
- **Verdict: not for dots and not for the national layer.** It is ethnic ascription with no sources,
  its Shia figure is above Pew 2009's compiled 10-15% and three times Pew 2012's surveyed 6%, and the
  licence bars copying the zones.
- **Source boxes, every religion map read, 2026-09-14** (48 maps: the 43 religion maps and 5
  religion-and-ethnicity ones). **16 carry a source list and 32 do not**, so "no source list" above is
  true of the five first opened, not of the set. The Shia core, Middle East and Gulf summary maps share
  one list (1908-2006); the others with lists are Kurdistan, Afghanistan detailed, Levant 2010, Syria
  2010, Yemen and Southwest Arabia, Nigeria and West Africa, Holy Land, Golan 2020, Nineveh Plain,
  western Libya and South Sudan. **Pakistan, Jordan and both Central Asia maps have none**; Nigeria's
  cites histories, the 2006 census populations and DHS 2008, and tags each LGA Muslim, Christian or
  even before spreading population over the tags. The lists are colonial gazetteers (Lorimer 1908,
  the Iraq administration reports, *Les Tribus Arabes de Syrie* 1930), Soviet ethnography (Bruk 1960,
  *Atlas Narodov Mira* 1964), *Weltkarte Volkstum* (1943-44), compilers (Gabriel 1971, Müller 1969,
  Trench 1996), Joshua Project, and a few official volumes that supplied populations rather than
  sect. **Nothing behind them gives provincial sect shares for any built country.**
- **What the lists lead to, for unbuilt countries.**
  - **Iran, a real lead.** Razmara, *Farhang-i jughrafiya-yi Iran* (Iranian Army General Staff,
    1949-53, 10 volumes and index), is openly scanned with OCR on archive.org (item
    `05.krdstan-aylam-byjar-tvysrkan-sghz-snndj-shah-abad-ghsr-shyryn-krmanshah-mlayr`, uploaded 2023;
    rights line "Imperial Army, 1332"). Village and district entries give population, mother tongue and
    mazhab: a sect label per place, not shares, around 1950, noisy OCR, licence unclear. Separately,
    **Iran's 1986 census form** (IPUMS `census_forms/asia/ir1986ef_iran_enumeration_form.en.pdf`) has a
    "RELIGION & SECT" column whose codes 2 and 3 sit under the office-coding shading, so sect was very
    probably collected; no sect table or "secret edition" was found anywhere. **Anita, 2026-09-14:
    "1950 is too old to use"**, so Razmara is not a source for this map; the 1986 census stays a lead
    only if a sect table ever surfaces.
  - **Lebanon:** TAVO A VIII 7, *Libanon: Religionen* (Hartmann 1979, 1:350,000), paywalled e-book at
    Reichert. The Middle East sheet **TAVO A VIII 6** (Hartmann 1987, 1:8M, print €30) is described as
    carrying percentages only in Izady's own box; no description or scan anywhere.
  - **Syria:** the 1960 census reportedly counted Sunni and Alawi separately (unverified, search
    summary); no governorate table seen. No census since 1960 has asked religion, so Izady's 1970
    census citation probably supplied population only.
  - **Iraq 1970 Annual Abstract:** no copy found; the 1957 census has religion by liwa with no sect.
  - **Köy Envanter Etüdleri** (Türkiye, 1960s): religion recorded only as Muslim or non-Muslim in the
    volumes seen. Dead for sect.
- **§14, for whoever takes Iran or Lebanon:** Razmara labels every Iranian village's sect including
  Sunni and Ahl-e Haqq ones; TAVO A VIII 7 maps Lebanon's religions at 1:350,000; Izady's Nigeria map
  prints a table of Shias by LGA; his Syria, Levant and Golan maps draw Alawite, Druze, Ismaili and
  Yazidi areas.

## Pakistan's Shia by province, second pass, 2026-09-14

- **Fair, Malhotra and Shapiro's April 2009 survey** (6,000 respondents, four provinces), replication
  data on Harvard Dataverse `hdl:1902.1/17042`, CC0, no guestbook. Item `d19`, *"are you sunni or
  shi'ite?"*, two answers only. Weighted Shia: Pakistan 3.8%, Punjab 2.7%, Sindh 9.2%, NWFP 0.4% (5 of
  1,128), Balochistan 0.5% (4 of 876). **Fails against places whose answer is known** (Kohat, Hangu,
  D.I. Khan, Quetta's Hazaras), and the paper itself (p. 4) notes respondents can read an interviewer's
  sect from name and accent. Not usable; a national floor at most. Its file has a 50-district variable,
  which is a §14 matter if anyone ever tabulates it.
- **Pew 2012**: no provincial table. **USIP Special Report 354** (Kalin and Siddiqui, 2014, with
  Gallup's Pakistan affiliate) over-sampled Shias on purpose and gives no share. **1951 and 1961
  censuses**: Muslims undivided on the catalogue records; the tables themselves not seen.
- **World Values Survey wave 7, Pakistan 2018** (online analysis sample 3350): the online tool at
  `worldvaluessurvey.org/WVSOnline.jsp` needs no registration or terms, and offers Q289CS9 (detailed
  denomination) crossed by N_REGION_WVS. Read with headless Chrome over CDP, 2026-09-14. **The
  detailed list has no Muslim branch at all**: Pakistan (N 1,995; Punjab 1,139, Sindh 493, KP 272,
  Balochistan 91) is `Islam; nfd` 98.3%, Hindu 0.8%, `Other; nfd` 0.7%, no answer 0.2%, with no
  Sunni, Shia or Ahmadi code. Bangladesh, Indonesia, Nigeria, Egypt and Malaysia 2018 are the same,
  one `Islam; nfd` row each. The tool shows counts over N, unweighted, with no weight option. So the
  WVS route is closed for sect in all six, and the licence-form microdata almost certainly carries
  the same codes. Nigeria's list shows `Eastern Orthodox` at 4.4%, which looks like a coding slip
  worth knowing before anyone reads that row.

## Where the common Sunni/Shia geography comes from: Iran and Pakistan traced, 2026-09-16

Anita, after a Google AI overview listing Iran's Sunni provinces: *"people have a pretty good
qualitative understanding of shia vs sunni ... i'd like to look into where this comes from. trying to
trace anything to see if it juts ultimately comes from personal account or something."* Two
researchers followed the citation chains to their origins. Extracted texts were in session
`99a598aa`'s scratchpad (`ir_sect/`, `pk_sect/`), not kept.

**For both countries, no regional Shia or Sunni share traces to a count or a probability survey,
except British India's censuses.** What exists is compilers with no source, sect leaders' claims,
officials' guesses, one author's per-unit percentages, and travel accounts, with ethnic ascription
underneath.

### Iran
- **National 5-10% Sunni:** the CIA Factbook's, no source (the 1990 edition said 4%). The State
  Department's 2022-23 reports call it Iranian government data, and a 2025 Persian journal cites the
  Factbook. Circular.
- **Leaders' claims:** 20% and over a million in Tehran, Molavi Abdolhamid to DW, 10 November 2011
  ("according to our statistics", none published); 25%, Molavi Abdolaziz Mollazadeh to the Assembly
  of Experts, September 1979. The Atlantic Council's "officials say 7-9%" (2018) names nobody.
- **GAMAAN 2020** (5% Sunni): online opt-in, 39,981 answers, weighted by province among other
  things; national only, not a probability sample.
- **The only provincial table:** Masaili, *Atlas-e Towsifi-ye Ahl-e Sonnat-e Iran* (Amirkabir, 2023,
  p. 59), reprinted as Table 1 of Hosseinpour and Masaili, *Haft Aseman* 26(88), 2025
  (`haftasman.urd.ac.ir/article_218087_735cc0a11dd4bd43e466ede88c6286bc.pdf`, open). It is 2016
  census totals times the author's own percentages from "library and first-hand field research":
  Kurdistan 82, Sistan and Baluchestan 64, Golestan 38, West Azerbaijan 35, Hormozgan 35, Kermanshah
  26, South Khorasan 15, North Khorasan 10, Gilan 7, Razavi Khorasan 5, Bushehr 5, Fars 4, plus
  500,000 for Tehran and the centre; 7,608,500 in all. The paper says no census has published sect.
  Farmanian, *Mazaheb dar Iran* (2021, p. 22), same method, 8-11%.
- **Ascription and accounts for the rest.** Wikipedia's Baloch "70-76%" has no citation; Kurdistan
  cites a qualitative study (Ahmady); Golestan's Turkmen 34.2% cites an unfound "Ministry of
  Education estimate, 2006", and the AI overview's Sunni majority there contradicts Masaili's 38%.
  Hormozgan, rural Kermanshah, Larestan and the Talysh are the Atlantic Council author's interviews
  and travel. A 31-province Sunni and mosque table on Persian blogs comes from a Kurdistan Da'wat va
  Eslah site (2008) calling itself "unconfirmed, unofficial". Anonby et al. 2019 put Kurdish as
  possibly the mother tongue of a slight majority in West Azerbaijan, above Masaili's 35% if Kurds
  are taken as Sunni.
- **Surveys with no sect item:** WVS 2020 Iran has province codes and no Iran denomination list,
  so no Sunni code, and no Baluchi or Turkmen home-language answer. *Values and Attitudes of
  Iranians* wave 4 (2023, 15 provinces) shows no sect item in its Identity volume (garbled text
  layer; waves 1-3 not reached). No sect table of the 1986 census anywhere.
- **Institutions, not people:** 54,770 mosques in the National Mosque Planning Survey and about 900
  Sunni Friday-prayer sites (via Factnameh); a Shabestan News infographic of mosques by province and
  sect (August 2025, an image, not read); 18-22 Sunni MPs (snippet only).
- **§14:** a provincial Sunni layer marks the Baloch and Kurdish areas under the heaviest crackdown
  (Zahedan, September 2022), and the numbers are politically loaded inside Iran.
- **Masaili, vetted 2026-09-16.** Anita: *"do people generally think this feels trustworthy / what
  is methodology / does he seem strongly biased?"*
  - **Who.** Mehdi Masaili, born about 1984, a Shia cleric (Hojjat al-Islam) at the Isfahan
    seminary since 2002 and a teacher there, and a PhD student in the rapprochement (taqrib) track
    of Islamic denominations at the University of Religions and Denominations, Qom, where his
    co-author Hosseinpour is an assistant professor. His other titles, from book listings only,
    argue against anti-Sunni Shia practice (one against the 9 Rabi' festival; *Pishvayan-e Shia,
    pishgaman-e vahdat*). Amirkabir published the atlas; Fars News and Ketab News covered it.
  - **Method, in his words:** library and field research with field given priority, and "this
    research was conducted individually and therefore faced many research limitations" (Ketab News
    interview, `ketabnews.com/fa/news/20546`). The 2025 paper calls the total "a summation of the
    Sunni population of Sunni-inhabited counties and provinces", 9-10%, and the book names the
    Sunni-majority counties per province (Hormozgan: Bandar Lengeh, Parsian, Khamir, Bastak, Qeshm,
    Jask; Kermanshah: the Avroman counties). No per-county share, source list or interview list
    appears in anything read; the book itself was not seen.
  - **Framing.** The paper was written to answer social-media claims that Sunnis will soon equal
    Shia in number. It calls estimates of a third, a quarter or a fifth exaggerated, driven by
    "demands for Sunni rights or intensifying Shia-Sunni rivalry". Any lean runs low.
  - **Cross-checks.** Farmanian and Beheshti, *Mazaheb dar Iran* (2021), from the same institution
    and by the same method: 8-11% nationally, Kurdistan 75 against Masaili's 82, West Azerbaijan 40
    against 35. Not independent. **Soltani, *Ahl-e Sonnat-e Iran: forsat-ha va chalesh-ha*** (Aras,
    Sanandaj, 2015, from his Arabic PhD at Imam Ouzai University, Lebanon; a Sunni author;
    archive.org `abu-abdurahman-kurdi-f_barid_20171224_0042`, OCR text read): about 10.5 million
    from "statistics and personal and field research", by group Kurds 5M, Baluch 2.5M, Persians
    1.7M, Turkmen 1M, Talysh 250,000, Arabs 120,000, Turks 50,000. Against Masaili's 7.6M, Golestan
    (800,000 against 710,000) and Sistan and Baluchestan (about 1.7M on the 2011 census against
    1.78M) agree; the gap is the Kurds, 5M against roughly 3M in Masaili's three Kurdish provinces,
    and the scattered remainder. His other province figures are garbled in the OCR. Islahweb, the
    Sunni reformist Da'wat va Eslah's site, cites Masaili's atlas without objection and says no
    estimate has scientific credibility.
  - **Verdict.** A careful one-person estimate from the rapprochement camp, open about its limits,
    with a stated reason to lean low; about 28% below a Sunni author working the same way, mostly
    on the Kurds. The best province-level figures in print, and still not a measurement.
  - **Direction, not a decision (Anita, 2026-09-16):** *"maybe the best thing to do in these highly
    mixed areas is just to use already published estimates"*, rather than the project assigning
    sect by ethnicity itself.
  - **Decided 2026-09-16: draw Iran's split from Masaili** (*"lets draw masaili. he seems good."*),
    after the §14 flag above. The brief is `queue.md`, "Iran's Sunni/Shia split from Masaili".
    **Built 2026-09-15** (`ir_split.py`, `sources/ir.md` §9): 7,608,500 Sunnis `derived` inside the
    census's Muslim column, the rest left on `islam`, the 500,000 lump over Tehran and Alborz.
  - **The language lesson.** Anita: *"this 'gap' in estimates for religions in muslim coutnries are
    in part due to like language barrier more than they are actual gaps."* Every English chain here
    ended at the CIA Factbook; all three usable estimates (Masaili, Farmanian, Soltani) are Persian
    and none is cited in English. **For a sect or school estimate, search in the country's own
    language before concluding there is none**, and read the local university and seminary
    journals, not only the statistics office.

### Pakistan
- **National:** CIA 20% Shia (2009), 10-15% (2021), then sect dropped; no source in any edition. The
  State Department's 15-20% is "generally believed". Wikipedia's "25-50 million" misattributes CIA
  2009. DFAT 2019 credits the 2017 census, which asks no sect.
- **Gilgit-Baltistan 39/27/18/16** (Shia/Sunni/Ismaili/Noorbakhshi): IRB Canada 2024 copies DFAT,
  which copies ICG 2007, *Discord in Pakistan's Northern Areas*, footnote 120, which cites **Manzoom
  Ali, *Atlas of the Northern Areas* (Gilgit, 2004; revised 2016)**, by a geography lecturer at
  Government Degree College Gilgit, with district splits such as Skardu 87/10/3. Method unknown; no
  copy or catalogue record found. Wikipedia's 39.85/30.05/24/6.1 cites PILDAT 2011, which has no such
  numbers.
- **Kurram 42% Shia:** Jamestown 2008's "official figures", citing *The News*, 20 January 2008; no
  document named. DFAT's Upper Kurram 80-83% and Lower 35:65 have no source. The 1931 NWFP census
  report (para 297) has the superintendent's estimate that Kurram's Shia tribes "exceed 60 per cent";
  Kurram was not counted by sect.
- **Quetta's Hazaras**, 500,000 to under a million: Minority Rights Group 2018, DFAT 2019, UK Home
  Office 2022, none sourced; NCHR's 0.4-0.5 million cites a World Hazara Council article.
- **Punjab by division**, Shia 10.86% (Sargodha 17.5, Rawalpindi 16.75, Faisalabad 15.5): Ramzan,
  *Journal of Political Studies* 22(2), 2015
  (`pu.edu.pk/images/journal/pols/pdf-files/7%20-%20RAMZAN_v22_2_wint2015.pdf`), Table 1, "personal
  estimates of officials of provincial and local governments"; the author was Deputy Director
  (Research) at the Punjab Home Department. Table 3 counts madrassas by sect and division from police
  and intelligence records (Barelvi 5,584, Deobandi 4,636, Ahl-e-Hadith 779, Shia 195), and the paper
  also lists madrassas "linked with militancy" by division. **Not drawn (Anita, 2026-09-16):** it
  covers Punjab alone and rests on officials' guesses by a questionable method, so it stays a lead to
  use later beside other data.
- **Barelvi 50%, Deobandi 20%:** GlobalSecurity.org, "by one estimate", no source.
- **British India sect tables located, not transcribed:** NWFP 1931 report
  (`in.ernet.dli.2015.56040`) reprints district Shia counts for 1911, 1921 and 1931 (Kohat 10,576 /
  40,248 / 13,556) and calls them unreliable; NWFP 1911 `india.history.resource.92548`; NWFP 1921
  only on JSTOR, `saoa.crl.25430163`; Baluchistan 1921 and 1931 province-level only
  (`in.ernet.dli.2015.105669`, `.56021`); Jammu & Kashmir 1931, Subsidiary Table V by province, with
  Shia, Molayi (Ismaili) and Nur Bakshi in the Frontier Districts (`in.ernet.dli.2015.105679`, tables
  volume `.56050`); the Punjab 1931 report (`.56041`) has no "Shia" in its OCR.
- A Sindh Home Ministry madrassa survey, 512 Shia of 12,545 (via Friday Times 2013, not seen).
- **§14:** the Kurram, Parachinar and Hazara figures point at enclaves attacked again and again
  (Hazara Town, Mari Abad; three Parachinar attacks in 2017 killed over 120, per DFAT).

### What this means here
Every version of the common picture is ascription or a figure someone wrote down. The nearest
things to tables, Masaili for Iran, Manzoom Ali for Gilgit-Baltistan and Ramzan for Punjab, are one
person's or officials' per-unit percentages with no stated method: `estimate` in spec §3.1's terms,
and the same shape as the ethnic-assignment direction above.

### `taxonomy/origin_religion.py`'s sect shares have no source
Found the same day, on Anita's idea of placing Shia in 99%-Sunni countries through foreigners. The
foreigner model already does it: `MUSLIM` gives Iran 0.90 Shia, Iraq 0.62, Lebanon 0.50, Azerbaijan
0.75, Bahrain 0.60, Yemen 0.35, Pakistan 0.145, India 0.13, Afghanistan 0.15 and Türkiye 0.20 Alevi,
and **every origin not listed is 100% Sunni** (`DEFAULT_MUSLIM`). No source is recorded for any row.
Pakistan and India are about twice Pew's surveyed 6% (2012, 2021). The default makes Omanis wholly
Sunni (most Omani Muslims are Ibadi) and gives Saudis, Kuwaitis and Tajikistan's Ismailis no split.
It feeds the foreign half of `fr`, `es`, `it`, `gr`, `be`, `se`, `no`, `dk`, `fi` and `lv`. Not
changed: a shared-code change moves drawn countries, and it is Anita's to schedule.
