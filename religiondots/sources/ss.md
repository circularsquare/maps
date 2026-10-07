# South Sudan (`ss`)

Session `cb8b206e-ss`, 2026-09-15, under a supervisor. Reopened from `queue.csv`'s blocked row on Anita's
priority of 2026-09-15 (`ask/RULINGS.md`, top line), which lifted her deferral of 2026-09-14. `sources.md`
§ss-2026-09-15. Ask 042. Nothing built, no code.

## 0. Outcome

**Drawn 2026-10-03 (session `fafd1067-ss`), six of ten former states, 6,886,052 people; §5 is the
build.** Jonglei, Unity, Upper Nile and Warrap are blank. §1-§4 are the 2026-09-15 scouting record and stay
as written.

*2026-09-15:* **Parked at checkpoint A.** The one source with religion below the nation is the World Bank's High
Frequency South Sudan Survey (HFSSS), waves 1 and 2, whose files need a free World Bank Microdata Library
login (Anita's, brief §3). Every open route checked on 2026-09-15 is national only or has no religion item
(§2). §14 is heavy; ask 042 carries the evidence and the recommended grain (§3).

## 1. The route: HFSSS waves 1 and 2

- **Wave 1.** World Bank Microdata Library catalog 2778, *South Sudan - High Frequency Survey 2015, Wave 1*,
  IDNO `SSD_2015_HFS-W1_v02_M` (DOI 10.48529/bn2b-8q88; IHSN 6969); Utz J. Pape (World Bank) with the
  National Bureau of Statistics. File **`hhq`**, 3,550 cases, 269 variables: `state` ("State of Residence"),
  `stratum` ("group(state urban)"), `ea`, `urban`, `weight` ("Population weight based on listing scaled to
  Census"), `C_8_hhh_tribe`, `C_8_1_hhh_tribe_spec`, `C_9_hhh_religion1`, `C_9_hhh_religion2`,
  `C_9_1_hhh_religion_spec`. Member file `hhm`, 23,004 cases. The UNHCR Microdata Library lists the same
  study (catalog 504) and sends the download back to the World Bank.
- **Wave 2.** Catalog 2777, *High Frequency Survey 2016, Wave 2*, IDNO `SSD_2016_HFS-W2_v02_M` (DOI
  10.48529/xz60-7w58; IHSN 6964). File **`hhq`**, 1,189 cases, 275 variables: `state`, `ea`, `weight_x`
  ("Cross_sectional population weight based on listing scaled to Census"), the same `C_8` and `C_9`
  variables. Member file `hhm`, 8,575. **`hhq_w1_w2` (2,332 cases) has no religion variable**, so it is not
  the file to ask for.
- **Access.** "Public Use Dataset": sign in, accept the terms (statistical and research use only, no
  redistribution, no attempt to identify respondents, cite the study).
- **The item** (§11aq, read from the dictionaries). Module C, *Please specify which religion [household
  head] belongs to*, multi-select: Christianity, Islam, Traditional African Religion, Judaism, Buddhism,
  Hinduism, Agnostic, Atheism, Other. Unweighted, wave 1: 3,492 answered (Christianity 3,117, Traditional
  231, Islam 132); wave 2: 1,156 (1,073, 31, 50). It is the head's religion, drawn for the whole household
  (`playbooks/dhs_mics.md`, first trap).
- **Coverage.** Wave 1: Western, Central and Eastern Equatoria, Northern and Western Bahr el Ghazal, Lakes,
  urban and rural. Wave 2 adds Warrap, urban only (§11aq). **Jonglei, Unity and Upper Nile are in neither.**
  On COD-PS 2022 (`ssd_admpop_adm1_2022_v2.csv`, 12,394,970): those three 4,677,661 (37.7%), Warrap
  1,294,050 (10.4%), the six fully sampled states 6,423,259 (51.8%).
- **Later waves.** Wave 3 (catalog 2914, IHSN 7268) lists the religion variables with 0 valid answers
  (§11aq). The 2017 wave 4 and Crisis Recovery Survey analysis file `hhm_analysis` (catalog 3392, 57,544
  cases) has no religion variable; its `state` is labelled "State/Camp".
- **Geography, if built.** COD-AB `cod-ab-ssd` v03: 10 states, 79 counties, 512 payams (CC BY-IGO; valid
  2022-12-19, reviewed 2024-10-09). COD-PS `cod-ps-ssd` (NBS and UNFPA): 2022 by state, 2024 and 2025 by
  county, no Abyei. USCB's HDX geodatabase has the 2008 census's 10 states, 79 counties and 514 payam
  polygons as a second boundary file.

## 2. Open routes, checked 2026-09-15

None has religion below the nation.

- **UNSD oracle**: South Sudan absent.
- **USCB on HDX**, `south-sudan-subnational-boundaries-and-tabular-data`: `South_Sudan_uscb_201812.xlsx`
  downloaded and every sheet listed. Twelve tables (Age-Sex, Education, Employment and Labor, Poverty and
  Consumption, Disability and Health, Mortality and Domestic Violence, Households, Access to Services,
  Agriculture and Livelihood, Pop Est and Food Insecurity, Migration, People in Need); no header in the top
  rows of any sheet mentions religion. §11j's negative, read from the dataset notes, holds at sheet level.
  USCB's only other South Sudan dataset is a 2017 gridded population raster.
- **HDX search** `religion south sudan`: 576 datasets; the top 50 titles read, none on religion.
- **World Bank Microdata Library variable search**, `api/catalog/search?vk=religion&country=SSD&ps=50`: 31
  South Sudan studies. Apart from the HFSSS waves they are programme-monitoring, energy, enterprise,
  finance, nutrition and refugee-camp surveys. The *Forced Displacement Survey 2023* (catalog 6639, UNHCR,
  about 3,000 households) samples refugees and hosts in Pariang and Maban counties and refugees in Central
  and Western Equatoria and Jonglei, so it is no state composition; its variables were not read.
- **2008 census**: the Presidency removed religion in March 2007 (*Southern Sudan Counts*, p.20; §11aq).
- **Sudan Household Health Survey 2006**, ERF's harmonised copy (`erfdataportal.com` catalog 103, licensed):
  all ten southern states at 1,000 households each; the household file (24,527 cases, 88 variables) has no
  religion variable. The individual, women's and children's files were not read. The 2010 round has no
  religion item (§11aq).
- **IRI, *Survey of South Sudan Public Opinion*, 6-27 September 2011** (2,225 adults, all ten states), PDF
  p.67, "What is your religion?": Christian 93%, Muslim 2%, Other 1%. National only.
- **IRI, *Survey of South Sudan Public Opinion*, 24 April to 22 May 2013**, PDF p.67: Christian
  (unspecified) 38%, Catholic 27%, Protestant 13%, Episcopal Church of Sudan 9%, Traditional 7%, Muslim 2%,
  non-religious 1%, Seventh-day Adventist 1%, Jehovah's Witness 1%, Africa Inland Church 1%, Evangelist and
  Baptist under 1%. National only; the sample size was not read. `iri.org` answers WebFetch with 403; curl
  with a browser user agent downloads both PDFs.
- **South Sudan Law Society and UNDP, *Search for a New Beginning*** (June 2015): 1,525 respondents in 11
  locations in six states and Abyei (Table 1, p.14); Table 2 is ethnicity; no religion table or figure in
  its lists.
- **Catholic dioceses** (catholic-hierarchy.org, Annuario Pontificio figures): Juba archdiocese 904,000
  Catholics of 1,174,000 (2023, 77.0%); Rumbek 238,103 of 1,158,000 (2023, 20.6%), but 212,000 of 1,805,000
  in 2022. gcatholic.org's national line: 7,724,000 of 14,746,000 (2023, 52.4%). One church, and diocesan
  populations that jump a third in a year: a witness at most. The other six diocese pages were not opened.
- **Pew Research Center 2020** (`data/raw/estimates/pew.zip`, as read by the Sudan build, `sources/sd.md`
  §1): Christians 60.5%, other religions 32.8%, Muslims 6.2%. National only.
- **Not in any round of** Afrobarometer, Arab Barometer, WVS, DHS or the Global Flourishing Study (§11aq,
  the playbooks).
- **Not checked**: Gallup World Poll (paid); NBS's current site; IOM's Displacement Tracking Matrix (terms
  forbid derivative works, RULINGS 2026-09-16); the other SHHS 2006 files; FDS 2023's variables.

**For the estimates layer.** Every instrument that asks people puts traditional religion far below Pew's
32.8% "other religions": HFSSS wave 1 6.6% of heads (231 of 3,492, unweighted, six states), IRI 2013 7%
(national). A level taken from Pew would draw four to five times the traditional share any survey records.

## 3. §14, in ask 042

- The war that began in Juba in December 2013 "quickly spread throughout the three states of the Greater
  Upper Nile region" (SSLS and UNDP 2015, executive summary): Jonglei, Unity and Upper Nile, the states the
  HFSSS never sampled. In that survey 63% of respondents said a close family member had been killed, and the
  report names the Dinka, Nuer and Shilluk as the groups most associated with the conflict.
- UNHCR Refugee Data Finder API (`year=2024&coo=SSD`, read 2026-09-15): 2,290,622 South Sudanese refugees
  abroad and 32,841 asylum seekers, 944,631 internally displaced, 404,744 refugees returned.
- **Recommended grain**: the former states (COD-AB v03's ten), drawing only those the survey sampled, and
  leaving Jonglei, Unity and Upper Nile empty as never measured (the Galápagos line, RULINGS `ec`
  2026-09-08) rather than filling them from the sampled states. A state holds 662,897 to 2,031,777 people on
  COD-PS 2022; the survey's enumeration areas are no county design, so nothing finer. Warrap's urban-only
  sample is the builder's call.
- **Refugees in South Sudan** (ask 033): UNHCR 2024 counts 514,794 refugees (487,652 from Sudan), 2,677
  asylum seekers and 18,000 stateless, 535,471 in all, 4.141% of COD-PS 2022 plus them, for `gap`. Their
  states were not read here.

## 4. REOPEN

- **The download** (ask 042): both `hhq` and both `hhm` files into `data/raw/ss/`; then `sources/ss.py` with
  heads weighted to persons, a split-half on EAs within states, COD-AB v03 admin1.
- **For that build**: whether `weight` already counts persons or needs household size from `hhm`; Warrap's
  urban-only sample; heads naming two religions on the multi-select; IRI 2013's 7% traditional and the
  dioceses as witnesses.
- **Not read**: SHHS 2006's individual file, FDS 2023's variables, six Catholic diocese pages.
- A post-war census, if one is held and asks religion.

## 5. Calls someone might reverse

1. Parked at A rather than drawn now at one national share (Pew or IRI): Anita refused one-unit draws for
   large countries (ask 014), and Pew's traditional level disagrees with every survey that asks.
2. The recommendation to leave the three unsampled states empty rather than fill them.
3. The Catholic diocese figures kept as a witness only.

## 5. The build, 2026-10-03 (`fafd1067-ss`)

Files: `sources/ss_geo.py`, `sources/ss.py`, `taxonomy/ss2015.py`, `countries/ss.py`; `data/raw/ss/` holds
Anita's two zips (licence: research use, no redistribution; nothing leaves but state shares), COD-AB,
COD-PS and Kontur.

### 5.1 What the files hold, checked

- The household key is `(state, ea, hh)`; `hh` alone repeats. `hhsize` equals the `hhm` roster lines in
  all 3,550 (wave 1) and 1,189 (wave 2) households. `weight` / `weight_x` is constant inside each EA, a
  household weight, so persons = weight x hhsize: 4,394,838 in wave 1's six states, 722,045 in wave 2.
- The handoff's warning holds: `hhq_w1_w2` has no religion; the per-wave `hhq` files do.
- **Wave 2 is towns only**: no `urban` variable, no rural EAs, 101 EAs in seven states (Warrap 15). Wave 1
  is 50 EAs per state, urban and rural. So wave 1 is drawn and wave 2 is a replicate of its urban half.
- Card (`lreligion`): Christianity, Islam, Traditional African Religion, Judaism, Buddhism, Hinduism,
  Agnostic, Atheism, Other. `religion1` is card order, not preference: 32 heads ticked two boxes and every
  one has Christianity first. Each is split equally between its answers. 58 heads (47 in rural Eastern
  Equatoria, a few EAs) have no answer; kept as `Not recorded`, mapped to nothing.

### 5.2 Checks

- **Decode**: the files' own `lState` labels matched to COD-AB names letter for letter; both waves decode
  the same. Held-out (no religion): survey state shares against the 2025 estimate, r +0.805, 9 of 719
  other orderings reach it. Printed only; six units cannot carry a join (spec §12, São Tomé), and the
  decode is by label.
- **Wave 2 against wave 1, towns only**, person-weighted, six states: Islam r +0.992 (mean gap 1.9
  points), traditional r +0.973 (0.5 points). Western Bahr el Ghazal's towns 19.7% Muslim in 2015, 14.0%
  in 2016; Northern Bahr el Ghazal's 7.2% and 7.3% traditional.
- **Split-half**: `lits.stability`, 300 EAs, median of 400 random EA halves against a 400-draw null that
  deals the EAs into states (family 5 in `sources/stability.py`), with its chi-square and one-EA vetoes.
  Christianity +0.943, traditional +1.000, Not recorded +0.898, Islam +0.829 (all p 0.0025); Atheism
  +0.920 (p 0.005, 8 heads); Buddhism +1.000 (p 0.027, 3 heads, all Central Equatoria) carry. Judaism
  (2 heads) fails, Agnostic (1) is untestable; both flat at the national share. `CARRIES` asserted.

### 5.3 As drawn (six states, 2025 estimate)

| state | heads | Christian | traditional | Muslim | not recorded |
|---|---|---|---|---|---|
| Central Equatoria | 600 | 98.6 | 0.0 | 1.3 | 0.0 |
| Eastern Equatoria | 569 | 75.1 | 10.6 | 0.3 | 12.8 |
| Lakes | 598 | 93.9 | 6.1 | 0.0 | 0.0 |
| Northern Bahr el Ghazal | 600 | 75.0 | 23.4 | 0.8 | 0.8 |
| Western Bahr el Ghazal | 585 | 84.6 | 2.6 | 10.5 | 1.2 |
| Western Equatoria | 598 | 99.6 | 0.0 | 0.4 | 0.0 |

Six states: Christianity 88.77%, traditional 6.74%, not recorded 2.48%, Islam 1.62%, Atheism 0.33%,
Buddhism 0.03%, Judaism 0.01%. Traditional 6.7% sits beside IRI 2013's 7% (national) and a fifth of Pew
2020's 32.8%.

### 5.4 Geography and population

- COD-AB v03: 10 states; admin2 has 78 counties plus `SS0001 Abyei` under a parent `SS00` that admin1
  lacks (dropped by pcode, as Sudan drops its SD19).
- **Population: the 2025 county estimates** (`SSD_2024_population_estimates_data.xlsx`, "displacement
  adjusted", cleared with NBS, IMWG-adopted), 13,297,196 without Abyei's 145,358, rather than the 2022
  COD-PS (12,394,970): newer, and by county, which the grid calibration needs. Every state is 1.03x-1.11x
  its 2022 figure. The workbook files Uror (SS0311) under SS04; parent taken from COD-AB, pinned in
  `MISFILED_PARENT`.
- Kontur `SS` 2023: 0.830 of the estimate overall, 2.21x in Eastern Equatoria (Kapoeta East, Torit and Budi
  read 2.7x-4.8x their county totals) and 0.51x in Western Bahr el Ghazal, so every hex is scaled to its
  county's 2025 total (`cd_geo.py`'s construction). No county is a Kontur hole (Nagero 4.2x is the
  largest factor; `HOLE_FACTOR` 10). No raw block reaches the cap (densest 42,059/km2). 1,537 hex
  centroids fall outside every county, 179,655 Kontur people: 1,028 hexes (31,062) in Abyei, 394 within
  2 km of the border (115,328), 91 more than 20 km out (the Kafia Kingi area, which Sudan administers). Dropped;
  the calibration puts each county's total back on its own hexes, so only border placement moves.
  Calibrated densest hex 44,694/km2 in Pochalla (Jonglei, which draws nothing).

### 5.5 The not-drawn part (`gap_share` 0.5146)

(6,411,144 in the four blank states + 171,087 not recorded + 535,471 refugees and asylum seekers, UNHCR
2024) over (13,297,196 + 535,471). Refugees are counted outside the 2025 estimate, which adjusts for
internal displacement and returns; if the estimate does hold them, the share is 53.5% and part of the
refugee figure (Maban, Jamjang, Renk are in Upper Nile and Unity) overlaps the blank states. Their
states were not read: UNHCR's portal answered with a JavaScript bot wall on 2026-10-03.

### 5.6 Calls someone might reverse

1. **Warrap left blank.** Its only sample is wave 2's towns (149 heads; 91.8% Christian, 6.3%
   traditional), and wave 1's town/countryside gap on traditional religion is 7.2 against 24.8 in
   Northern Bahr el Ghazal and 2.1 against 13.4 in Eastern Equatoria. Drawing it from towns would put
   the state at a third of its likely traditional share; modelling the countryside from neighbours was
   not tried because Northern Bahr el Ghazal (24.8) and Lakes (4.4) disagree too much to borrow from.
   Reversing: add 81 to the drawn set in `ss.py` from wave 2.
2. **Wave 1 alone**, wave 2 as witness, rather than pooling the towns of both waves.
3. **Buddhism and Atheism drawn where found** on 3 and 8 heads, because they pass the stated test and
   both vetoes; Judaism kept on its node at 0.01% flat rather than dropped.
4. **The 2025 county estimates** as the base, not the endorsed 2022 COD-PS.
5. Refugees added outside the estimate in `gap_share` (§5.5).

### 5.7 Not done

- The roster in `hhm` has no religion item; nothing prices the head-for-household assumption here.
- The six states' refugee counts (UNHCR portal) and the IRI 2013 sample size.
- Wave 3 (catalog 2914, religion variables empty) and the 2017 Crisis Recovery Survey were not reopened.

## 6. Review, 2026-10-03 (`fafd1067-rev2`, full pass)

Checks clean (check_md, built_countries, check_rollup: 6,714,965 all modelled, nothing orphaned).
The normalized CSV reproduces every figure in the note and in §5.3 (six states 6,886,052; the four
blank states 48.2% of 13,297,196; `gap_share` 0.5146 recomputed). Mapping matches precedent (bare
`christianity` for a one-box card, Atheism and Agnostic on `secular`). The grain is the one ask 042
recommended, and Warrap's blanking is argued in §5.6. Screenshot: Jonglei, Unity, Upper Nile and
Warrap empty, dots along the Nile and in Aweil, Wau, Rumbek, Juba and Yambio, none in the sea.

- **`gap` reworded:** "about 535,000 refugees and asylum seekers" read as South Sudanese abroad (2.3
  million, the better-known figure); they are UNHCR's refugees *in* South Sudan, mostly from Sudan
  (§3). Now says so. refresh-meta run.

## 7. Top text before the 75-word cut, 2026-10-03 (`fafd1067-notes75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/ss.py`; `note_public` was not changed.

- `how`: household survey, one round in 2015
- `grain`: former states, 1.1 million people on average
- `gap`: 51.5% of residents: Jonglei, Unity, Upper Nile and Warrap, which the survey did not sample or sampled in towns only (6.4 million); households whose head's religion was not recorded (171,000); and about 535,000 refugees and asylum seekers from other countries living in South Sudan, most from Sudan (UNHCR, 2024)
