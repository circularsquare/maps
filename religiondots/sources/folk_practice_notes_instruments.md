# Instruments asking the same religion or practice question in mainland China, Taiwan and Hong Kong

Research notes, 2026-09-15. No repo file edited. Figures marked (unverified) come from memory or from a
search-result snippet that was not opened; everything else was read from the source named.

Working files in this scratchpad: `grep_pdfs*.py` and `grep_out*.txt` (text pulled out of the PDFs
below with PyMuPDF). The PDFs themselves sit in the session `tool-results` folder.

---

## 0. Headline

- **No instrument found asks one folk-practice item in all three places.** Every usable item covers
  a pair, and Taiwan is in every pair:
  - **China + Taiwan**: the ISSP religion items (CGSS 2010 and **CGSS 2018**; TSCS 2008/2018), and
    the EASS ritual block (2008: auspicious days only; **2018: grave visits, praying for luck, charm,
    auspicious days**).
  - **Hong Kong + Taiwan**: Pew's *Religion and Spirituality in East Asian Societies* (2023): ancestor
    offerings, home altar, deity veneration, temple visits, graves, fortune-telling.
  - **All three, one questionnaire**: only AsiaBarometer 2006 (items not checked) and the WVS/Asian
    Barometer identity questions (no folk-practice items).
- **EASS 2008 adds nothing folk-practice beyond ISSP.** Its "East Asian rituals" block is two items:
  counting strokes when naming a child, and caring about auspicious days. No ancestor, incense or
  temple item.
- **CGSS 2018 is the key China file, and it is not open.** Its questionnaire carries both the
  EASS 2018 ritual block and the ISSP 2018 religion items (home shrine, temple visits, ancestors'
  powers). Harvard Dataverse returns 0 for `cgss2018` and `"CGSS 2018"`.
- **Hong Kong has never fielded the ISSP wording.** Its practice items come from Pew 2023, the
  HK Political Culture Survey 2021, and the 1988/1995 Indicators of Social Development surveys.

---

## 1. Instrument table

| instrument | societies | year | item & threshold | figures | sub-national geography | access |
|---|---|---|---|---|---|---|
| **Pew, Religion & Spirituality in East Asian Societies** | HK, TW (+JP, KR, VN); not CN | Jun-Sep 2023 | ancestor rites past 12 months: burned incense / offered food, water or drinks / flowers or candles / money or goods for afterlife | incense HK 57, TW 81; food HK 48, TW 77; flowers HK 58, TW 47; money HK 44, TW 70 | none published; CATI random-digit-dial; weights age/gender/education (region only in VN) | Pew dataset needs free account (DOI 10.58094/5jv2-m279); ARDA `fid=EASTASIA` listed, page did not render |
| same | HK, TW | 2023 | "there is an altar in your home" | HK 25, TW 49 (unaffiliated 21 / 37; Buddhists 56 / 58; Christians 11 / 15) | none | same |
| same | HK, TW | 2023 | pray or offer respects to Guanyin | HK 30, TW 69 | none | same |
| same | HK, TW | 2023 | "generally go to" temples or pagodas | by group only: unaffiliated HK 5, TW 53; Buddhists HK 50, TW 90; Daoists TW 85 | none | same |
| same | HK, TW | 2023 | have a family gravesite / visit it at least yearly (share of all adults) | have HK 31, TW 28; visit HK 25, TW 24 | none | same |
| same | HK, TW | 2023 | religion "What is your religion, if any?" (list includes Daoist, Confucian, Local/Indigenous religions) | HK none 61, Buddhist 14, Christian 20, Daoist 1, other 3; TW none 27, Buddhist 28, Christian 7, Daoist 24, other 12 | none | same |
| **EASS 2008 Culture & Globalization** | CN (CGSS 2008, n 3,010), TW (TSCS 2008 module II, n 2,067), JP 2,160, KR 1,508; no HK | 2008 | B1 counted strokes when naming own child (yes/no, skip if no child); B2 care about auspicious or ominous days for wedding, moving, funeral (4 points) | TW unweighted: B1 yes 43.9% of all (908), no 24.3%, skipped 31.7%; B2 very 32.2, somewhat 41.5, not very 13.9, not at all 12.3. CN not found | EASS 2008 integrated file geography not checked; TSCS 2008 has residence county/township | TSCS 2008: SRDA `C00156_2`, direct download for SRDA members; EASS integrated: ICPSR 34607 (403 to fetcher), EASSDA (captcha) |
| **EASS 2018 Culture & Globalization** | CN (CGSS 2018), TW (TSCS 2018 module II, n 1,961, fielded 2018-19), JP, KR; no HK | 2018 | V8 auspicious days (same as 2008); N1 visits to family graves past year (0 to 5+); N2 went to a place to pray for good luck past year (optional item); N3 has charm or amulet for good luck | CN (CGSS 2018, via Pew 2023): grave at least once 75% (never 24, once 38, twice 23, 3+ 14); pray for good fortune 24%; charm 8%; care somewhat/very about auspicious days 62%. TW not found | harmonized `cn_reg` = province; `tw_reg` = county/city (Keelung, Taipei city, Taipei county...) | questionnaire + harmonized codebook open: `eassda.org/down/2018/Standard Questionnaire for 2018 Culture Module.pdf` (288 pp, 8.2 MB); data ICPSR 38489 (403), EASSDA (captcha); TSCS 2018 module II on SRDA |
| **ISSP religion items in CGSS 2018** | CN | 2018 | C21 home shrine/altar/religious object for religious reasons; C22 frequency of visiting temple, Daoist temple, church or mosque for religious reasons; C11-5 belief in ancestors' supernatural powers | not found published; Pew notes 2018 religion module asked of about 1/3 (~4,500) | province (`s41`-type) | **not open**: Dataverse 0 hits; CNSDA application |
| **ISSP religion items in CGSS 2010** | CN | 2010 | ISSP 2008 items (per coordinator: `na` home shrine, `nb` temple visits, `n187` ancestors, `n361` fortune-telling, `n362` feng shui) | coordinator computing | 31 provinces | on disk |
| **ISSP in TSCS** | TW | 2008 (ISSP), 2018 (ISSP); plus 2014 | same home-altar item (per coordinator: 2014 `v27`, 2018 `v55`; 2018 `v49f` ancestors) | coordinator computing | county | ARDA open (TSCS142, TSCS181) |
| **HK Political Culture Survey 2021** (Cai & Hung, *China Quarterly* 259, 2024, CC BY) | HK only | May-Sep 2021 | practices (multi-answer): burning incense for dead family members; drawing a fortune stick / evaluating fengshui; fortune teller; self fortune-telling. "No religion but practising" = no affiliation AND at least one practice | incense for dead family 75.01% (2,807); fortune stick/fengshui 40.60; fortune teller 20.24; palm/Tarot 9.97. Beliefs (agree): good to be worshipped by posterity 70.37; select auspicious date 70.37; soul survives 49.76; fengshui first when buying a flat 46.39. No religion but practising 56.07; no religion, not practising 9.76 | 72 of 452 electoral districts sampled; no district figures | paper open; microdata not public |
| **Indicators of Social Development, HK** (Cheng & Wong 1997, as tabled by Cai & Hung) | HK | 1988 (n 1,644), 1995 (n 2,275) | religion list including folk religion; 1995 ancestor worship | folk religion 23.0 (1988), 15.3 (1995); Buddhism 6.6 / 11.6; Protestant 7.2 / 8.4; Catholic 4.9 / 4.5; none 58.3 / 60.2; 1995 ancestor worship 55.2% | not known | book (CUHK HKIAPS), not seen |
| **AsiaBarometer 2006** | CN, TW, HK, JP, KR, SG, VN; adults 20-69 | Jun-Aug 2006 | one English master questionnaire, back-translated; religion/practice items **not checked** | none found | not known | openICPSR 163441, free ICPSR login (page 403 to fetcher) |
| **Asian Barometer** | CN, TW, HK (waves 1-5) | 2001- | religion + attendance (unverified); no folk-practice items known | Pew GRL 2012 drew HK composition from ABS 2001 | not checked | application form |
| **WVS** wave 7 | CN 2018 (n 3,036, 29 provinces), HK 2018, TW 2019 | 2018-19 | denomination, attendance, prayer, importance, belief in God/afterlife; no altar or ancestor items (unverified) | CN 2018: zongjiao Buddhism 8.8 (Pew); Pew 2025 uses WVS 2018 for HK 2020 composition | CN province | WVS download form; online tool not tried |
| **Spiritual Life Study of Chinese Residents** (Horizon) | CN only | 2007, n 7,021 (Pew GRL) or 6,861 (Pew 2023), 16+ | worship gods/spirits at sites, home, workplace; temple incense; belief items | basis of Pew 2012 China folk 22% | sampled cities/towns/villages; no west, far northeast, south coast | ARDA `SPRTCHNA` (page did not render) |

CFPS-derived, recorded only as published by Pew 2023: 2016 CFPS burn incense to worship Buddha or
deities a few times a year 26% (11% monthly); 2018 CFPS believe in Buddha/bodhisattva 33%, Taoist
immortals 18%, fengshui 47%, ghosts 10%.

---

## 2. Item wording (source language)

**Pew 2023 East Asia** (English master; translations of the god item only are in the report):
- Religion: "What is your religion, if any? Buddhist; Catholic, Protestant or other Christian; Muslim;
  Daoist; Confucian; Local religions/Indigenous religions; No religion; or Some other religion."
- Ancestors: "In the past 12 months, have you done each of the following to honor or take care of your
  ancestors?" burned incense; offered food, water or drinks; offered flowers or lit candles; offered
  money or other things they may need in the afterlife.
- God: Cantonese 您信唔信神？; Mandarin (HK) 您是否相信神？; Mandarin (TW) 請問您信不信神？;
  Hokkien 請問你有信神或是無信神？

**EASS 2008, Taiwan wording (TSCS 2008 report, Appendix 1, p54):**
- B1 當您在為自己的小孩取名字時，是否有算過筆劃? (01 是 / 02 否)
- B2 請問在特殊的情況如結婚、搬遷或喪禮時，您會不會在意吉日或凶日? (非常在意 / 有點在意 / 不太在意 / 毫不在意)

**EASS 2018, China wording (CGSS 2018 questionnaire, B block, p43-44):**
- B2 请问在特殊的情况如结婚、搬迁或丧礼时，您会不会在意吉日或凶日？
- B3 在过去的一年中，您去过多少次已故家庭成员的墓地？ (没有 / 一次 / 两次 / 三次 / 四次 / 五次或以上)
- B4 在过去的一年中，您有多少次去过一个地方祈求好运（学术和商业成功、健康等）？
- B5 您是否随身戴着祝好运的符咒和/或护身符？ (有 / 没有)

**ISSP 2018, China wording (CGSS 2018 questionnaire, C block):**
- C21 您家里是否有出于宗教原因而设的神龛、祭坛，或者摆放宗教物品如观音像、神位、神符等？
- C22 不包括您在自己所属的宗教场所参加日常性宗教活动，您多长时间会出于宗教信仰的原因去一次寺庙、道观、教堂或清真寺等宗教场所？
  (从来没有去过 / 一年不到1次 / 一年大概1到2次 / 一年几次 / 大概一月1次或更多)
- C11-5 您是否相信以下事物？ 祖先显灵 (完全相信 / 也许相信 / 也许不相信 / 完全不相信)
- A5 religion list includes 民间信仰（拜妈祖、关公等）.

ISSP English source (GESIS): "For religious reasons do you have in your home a shrine, altar, or a
religious object on display such as an icon, retablos, mezuzah, menorah, or crucifix?"; "How often do
you visit a holy place for religious reasons, such as going to a shrine, temple, church or mosque?"

**Wording gap to carry into any bridge:** Pew asks whether *there is an altar in your home*; ISSP asks
about a shrine, altar *or religious object* kept *for religious reasons*. The ISSP version should run
higher where objects count and lower where people deny "religious reasons". Pew is phone (CATI), CGSS
and TSCS are face-to-face.

---

## 3. Consistent measure across all three: what exists

**Nothing measures one item the same way in all three.** Two routes to a consistent layer, both
bridged through Taiwan:

1. **ISSP home altar (China + Taiwan measured, Hong Kong bridged).** CGSS 2010/2018 and TSCS
   2014/2018 carry the same ISSP item with province and county geography. Hong Kong has no ISSP
   item. Its only home-altar figure is Pew 2023 (HK 25, TW 49), so an HK value would be
   TW(ISSP) x 25/49, assuming the Pew-to-ISSP wording effect is the same in both places.
2. **Ancestor offerings (Hong Kong + Taiwan measured, China bridged).** Pew gives HK and TW directly.
   China has no same-wording item. The nearest are CGSS 2018 grave visits (75%), HKPCS 2021 incense
   for dead family (HK 75%) and CFPS-derived incense to deities (26%). None of these is a like-for-like
   ancestor-offering item.

**Middle-ground shares on the items that exist (percent of adults):**

| practice | China | Taiwan | Hong Kong |
|---|---|---|---|
| burned incense for ancestors, past year | no item | 81 (Pew 2023) | 57 (Pew 2023); 75 "for dead family" (HKPCS 2021) |
| altar in home | ISSP item in CGSS 2010/2018, coordinator computing | 49 (Pew); ISSP `v27`/`v55` computing | 25 (Pew) |
| visited family grave, past year | 75 (CGSS 2018, EASS item) | EASS 2018 item exists, figure not found; Pew: 24 visit yearly | Pew: 25 visit yearly |
| went somewhere to pray for luck, past year | 24 (CGSS 2018) | EASS 2018 item, not found | fortune stick/fengshui 40.6 (HKPCS, different item) |
| care about auspicious days | 62 (CGSS 2018) | 73.7 (TSCS 2008, unweighted) | 70.4 agree (HKPCS, agree scale) |
| charm or amulet | 8 (CGSS 2018) | EASS 2018 item, not found | no item |
| pray to Guanyin | no item | 69 (Pew) | 30 (Pew) |
| self-identified folk / Daoist / Buddhist | folk 2.7, Daoist <0.5, Buddhist 3.8 (CGSS 2018, Pew weights) | Pew 2023: Daoist 24, Buddhist 28, other 12; TSCS long card folk 48 | Pew 2023: Buddhist 14, Daoist 1; HKPCS: Buddhist 13.7, Daoist 4.0 |

Taiwan (Pew) against Taiwan (TSCS long card) shows how much the answer set moves identity. Pew offers
no "folk religion" box and gets Daoist 24 and other 12. TSCS's interviewer-coded long card gets folk 48.

**Two things would close most of the gap:**
- **EASS 2018 integrated file** (ICPSR 38489): Taiwan's grave, pray-for-luck and charm figures on
  exactly China's wording, with `cn_reg` provinces and `tw_reg` counties. ICPSR EASS sets usually
  need a free MyData login (unverified). The Taiwan half alone is SRDA's TSCS 2018 module II.
- **AsiaBarometer 2006** (openICPSR 163441): the only questionnaire found fielded in CN, TW and HK
  together. Its religion items are unchecked. An hour with the codebook settles whether it has any
  practice item.

---

## 4. Provenance of compiler figures

### CIA World Factbook (site retired 2026-02-04; read from the `factbook/factbook.json` GitHub mirror)

- **China**: "folk religion 21.9%, Buddhist 18.2%, Christian 5.1%, Muslim 1.8%, Hindu < 0.1%, Jewish
  < 0.1%, other 0.7% (includes Daoist (Taoist)), unaffiliated 52.1% (2021 est.)"; note "officially
  atheist". **These are Pew GRL 2012's 2010 figures digit for digit** (Pew: unaffiliated 52.2). The
  "2021 est." label is the Factbook's; the numbers are Pew's 2010 values.
- **Taiwan**: "Buddhist 35.3%, Taoist 33.2%, Christian 3.9%, folk religion (includes Confucian)
  approximately 10%, none or unspecified 18.2% (2005 est.)". Traces (per a US State Department
  snippet, not opened) to the 2006 ROC Government Information Office Yearbook, which quotes the Ministry
  of the Interior Religious Affairs Section: 35% Buddhist, 33% Taoist. That is a ministry estimate,
  not a survey (unverified in detail).
- **Hong Kong**: "Buddhist or Taoist 27.9%, Protestant 6.7%, Roman Catholic 5.3%, Muslim 4.2%, Hindu
  1.4%, Sikh 0.2%, other or none 54.3% (2016 est.)"; note "many people practice Confucianism". The
  shares match gov.hk *Hong Kong: The Facts, Religion* body self-estimates divided by ~7.3M. For
  example, 390,000 Catholics gives 5.3%, 300,000 Muslims 4.1%, 100,000 Hindus 1.4%. That match is my
  arithmetic, not a stated source (`sources/hk.md` §2 already rejects these body counts).

### Pew Global Religious Landscape 2012 (2010 figures), table p45-50, Appendices A and B

| | Christian | Muslim | unaffiliated | Hindu | Buddhist | folk | other | Jewish |
|---|---|---|---|---|---|---|---|---|
| China | 5.1 | 1.8 | 52.2 | <0.1 | 18.2 | 21.9 | 0.7 | <0.1 |
| Taiwan | 5.5 | <0.1 | 12.7 | <0.1 | 21.3 | 44.2 | 16.2 | <0.1 |
| Hong Kong | 14.3 | 1.8 | 56.1 | 0.4 | 13.2 | 12.8 | 1.5 | <0.1 |
| Macau | 7.2 | 0.2 | 15.4 | <0.1 | 17.3 | 58.9 | 1.0 | <0.1 |

- **China** (App. A p63-64): Muslims from 2000 census ethnicity; Christians from multiple sources
  (Global Christianity 2011, App. C); Hindus and Jews from WRD. Buddhists, other religions, folk and
  unaffiliated come from Pew's analysis of the **2007 Spiritual Life Study of Chinese Residents**.
  - **Folk religionist**: someone who "did not identify with one of the other religious groups and
    they did report that they worshiped gods or spirits at conventional religious sites, at home or
    in the workplace; or if they attended formal temple services or prayed or burned incense in
    temples; or if they believed in the existence of gods or spirits, evil forces or demons, heaven,
    hell, the afterlife or reincarnation."
  - **Buddhist**: from a question on belief in "Buddhism or Buddha" (Pew 2025 p121).
  - CGSS 2010 was "not publicly available at the time".
- **Taiwan** (App. B p79): "Estimates based on 2009 Taiwan Social Change Survey, adjusted to account for
  underrepresented religious groups."
- **Hong Kong** (App. B p74): "Estimates based on 2001 Asian Barometer, adjusted to account for
  underrepresented religious groups." Pew 2025 says folk-religion undercounts were adjusted "using data
  from the World Religion Database and other sources", so HK's 12.8 folk is probably WRD-influenced
  (inference).

### Pew, How the Global Religious Landscape Changed 2010-2020 (2025)

- Folk religion is no longer a category; it goes into "other religions". Pew no longer adjusts for
  undercounts (p123-124).
- **China** now uses the CGSS 2010 and 2018 zongjiao identity question, reweighted to census ethnic
  shares. The revised 2010 figures replace 52.2/21.9/18.2 (p121).

| (App. B, 2025) | Christian | Muslim | unaffiliated | Buddhist | Hindu | Jewish | other |
|---|---|---|---|---|---|---|---|
| China 2010 | 2.3 | 1.5 | 87.4 | 5.6 | <0.1 | <0.1 | 3.1 |
| China 2020 | 1.8 | 1.8 | 89.6 | 3.7 | <0.1 | <0.1 | 3.0 |
| Taiwan 2010 | 4.9 | 0.3 | 22.8 | 20.9 | <0.1 | <0.1 | 51.1 |
| Taiwan 2020 | 5.5 | 0.5 | 23.1 | 19.2 | <0.1 | <0.1 | 51.7 |
| Hong Kong 2010 | 18.6 | 0.1 | 68.2 | 11.9 | 0.1 | <0.1 | 1.2 |
| Hong Kong 2020 | 18.7 | 0.3 | 71.4 | 8.4 | 0.3 | <0.1 | 0.9 |

Sources (App. A table, five columns per country; I read them as 2010 composition, 2010 detail, 2020
composition, 2020 detail, switching — column headers not seen, so treat as unverified):
- China: CGSS 2010, 2010, 2018, 2018, 2018.
- Taiwan: TSCS 2009/2010/2012 for 2010; TSCS 2019 and 2020 for 2020; Pew 2023 East Asia for switching.
- Hong Kong: back-projection of WVS 2014 for 2010; WVS 2018 for 2020; Pew 2023 East Asia.

### World Religion Database (via ARDA national profiles, "Gina A. Zurlo, ed., WRD, Brill, accessed September 2025")

- China: Chinese folk-religionists 29.56% (418,617,805), Buddhists 15.85, Christians 8.54,
  Muslims 1.86, agnostics 31.69, atheists 6.82.
- Taiwan: folk 42.50, Buddhists 26.46, Daoists 12.61, new religionists 6.75, Christians 6.49,
  agnostics 4.27, Muslims 0.39.
- Hong Kong: folk **42.50** (3,143,055). Macau 54.63, Singapore 37.50, Malaysia 18.80, Vietnam 1.00.
- **Taiwan and Hong Kong are both exactly 42.50%**, which reads as an assigned share rather than a
  measurement.
- WRD definition of the category: "Followers of indigenous religions of China, representing an
  amalgamation of beliefs and practices that can include: universism…, worship of
  ancestors/gods/goddesses/spirits, divination, sacrifices, and elements from Taoism, Confucianism,
  neo-Confucianism, and/or Buddhism."
- Method for China not found. Johnson & Grim 2013, *The World's Religions in Figures* ch. 12
  "Estimating China's Religious Populations", is Wiley 403.

---

## 5. Academic comparisons

- **Yang & Hu 2012**, JSSR 51(3):505-521, *Mapping Chinese Folk Religion in Mainland China and
  Taiwan*. Three surveys; communal, sectarian and individual folk religion. Abstract only; Wiley
  paywall, ResearchGate not tried. Pew 2012 cites it beside the 2007 Spiritual Life Study. Which
  Taiwan survey it used, and its figures, are unverified (probably TSCS 2009 plus the 2007 Spiritual
  Life Study).
- **Zhang, Lu & Sheng 2021**, *Chinese Journal of Sociology*, *Exploring Chinese folk religion:
  popularity, diffuseness, and diversities*. The PKU-hosted PDF fetch failed. Data source not
  confirmed (possibly CFPS; if so, cite only published figures).
- **Hu Anning**: 2014 Taiwan folk religion and giving (TSCS); 2025 family rituals from the Chinese
  Cultural Values Survey (mainland only); 2018 *Demographic Research* on ancestor worship and family
  formation (mainland). No cross-strait same-item paper found.
- **Pew 2023 *Measuring Religion in China***, the best compiler treatment:
  - It shows the Buddhist identity share drops when a "folk religion" box is offered (CGSS 3.8-5.8
    against WVS/CFPS/CLDS 6.8-9.2).
  - It shows 2021 CGSS folk religion (0.2%) collapses mainly because nine higher-folk provinces went
    unsampled: 6% against 1% in 2018 (p118-119). This bears on `sources/cn_cgss.md`'s 2021 fall.

---

## 6. Dead ends (what was tried)

- ICPSR study pages 34607 and 38489, and openICPSR 163441: HTTP 403 to the fetcher.
- `eassda.org` home: Korean captcha ("please prove that you are human"). The direct file URL
  `eassda.org/down/2018/Standard Questionnaire for 2018 Culture Module.pdf` works.
- JGSS `english/research/codebook/EASS2008_Codebook.pdf`: 404. JGSS Japanese EASS 2018 codebook:
  over 10 MB, not fetched.
- GESIS pages (ISSP 2018 overview, ZA7570 search): 403. `dbk.gesis.org`: DNS failure. Whether China
  is in the ISSP 2018 integrated file stays unverified; CGSS 2018 carries the items either way.
- Pew HTML pages (practices, measuring religion, Wiley abstracts): 403. The Pew PDFs downloaded fine.
- ARDA `data-archive?fid=EASTASIA`, `SPRTCHNA`, `TSCS182`: render only the archive menu, no metadata.
- **CGSS 2018 and CGSS 2008 open copies**: Harvard Dataverse file search `cgss2018` 0, `cgss2008` 0,
  dataset search `"CGSS 2018"` 0. Figshare "Chinese general social survey data, CGSS." (28568735)
  gave 403, wave unknown, not resolved.
- SRDA metadata for TSCS 2018 module II: ID not found (guessed `C00345_2`, a search page).
- AsiaBarometer 2006 religion items: nothing online. Inoguchi et al. 2009 sourcebook is print.
- HKPSSD religion practice figures: none published found; data is application-gated.
- WVS online analysis tool: not attempted (JS app). WVS wave 7 CN/HK/TW denomination tables not
  collected.
- Yang & Hu 2012 full text and Johnson & Grim ch. 12: paywalled; not retried.
