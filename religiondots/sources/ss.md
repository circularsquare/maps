# South Sudan (`ss`)

Session `cb8b206e-ss`, 2026-09-15, under a supervisor. Reopened from `queue.csv`'s blocked row on Anita's
priority of 2026-09-15 (`ask/RULINGS.md`, top line), which lifted her deferral of 2026-09-14. `sources.md`
§ss-2026-09-15. Ask 042. Nothing built, no code.

## 0. Outcome

**Parked at checkpoint A.** The one source with religion below the nation is the World Bank's High
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
