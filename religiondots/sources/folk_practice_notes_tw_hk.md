# Taiwan and Hong Kong folk-religion practice: working notes (2026-09-15)

No repo file was edited. Two scratch scripts read the ARDA TSCS Stata files already in
`religiondots/data/raw/tw/` (read-only): `crosstab_tscs.py` and `composite_tscs.py` in this scratchpad.
Text extracts of every PDF used are beside this file (`*.txt`).

Tags: **[orig]** = original measurement read from the source itself; **[mine]** = my computation from
open microdata; **[via X]** = figure seen only in compiler X; **[unverified]** = snippet or memory, not
checked against the source.

---

## 1. TAIWAN

### 1.1 TSCS religion card: how the 48% folk is built [orig]

Source: 傅仰止、林本炫、蔡明璋 主編, 《台灣社會變遷基本調查計畫 第七期第四次調查計畫執行報告》
(2018 round; tables head the column "七期四次2019"), Academia Sinica Institute of Sociology.
Open PDF, 481 pp: https://www2.ios.sinica.edu.tw/sc/cht/download-tscs18.pdf . Frequency tables in
附錄四 carry 2009 / 2014 / 2018 side by side. 2014 report: https://www2.ios.sinica.edu.tw/sc/cht/download-tscs14.pdf (454 pp).

Q29 「請問您目前的宗教信仰是？（請依受訪者答案勾選）」 open answer, interviewer codes.
Interviewer instructions (report p.103-104, same in 2014):

- (021) 自己認為的: 主動說明自己是屬於『民間信仰者』 (volunteers "folk religion")
- (022) 拜神的: 只是表明自己是拜神明、尊敬神明、什麼都拜、拿香的、**自認無宗教信仰但會跟著家人一起拜**…等的一般信仰者
  (says they worship/respect the gods, worship everything, hold incense, **or says they have no religion but worship along with family**)
- (023) 沒有清楚認定: not clear, but by (022)'s description obviously folk
- (024) 其他

So part of the long card's folk share is people who first said "no religion". Self-identified folk is tiny:

| code (unweighted %) | 2009 | 2014 | 2018 |
|---|---:|---:|---:|
| 010 沒有宗教信仰 none | 12.9 | 10.3 | 13.2 |
| 02 民間信仰 folk total | 42.8 | 48.3 | 49.3 |
| 021 自己認為的 self-identified | 3.1 | 2.5 | 1.5 |
| 022 拜神的 worships the gods | 35.9 | 42.7 | 46.4 |
| 023 沒有清楚認定 not clearly specified | 3.6 | 3.1 | 1.3 |
| 024 其他 | 0.2 | 0 | 0.1 |
| 03 佛教 Buddhism (031 信佛、拜佛 alone: 16.6 / 11.3 / 11.1) | 19.7 | 14.9 | 14.0 |
| 040 道教 Taoism | 13.5 | 15.6 | 12.4 |
| 060 基督教 Protestant | 4.0 | 4.3 | 5.5 |
| 072 一貫道 | 1.7 | 2.0 | 2.1 |

n = 1,927 / 1,934 / 1,842. Weighted (WR_19_5 in 2014 and 2018, WEIGHT 2009) [mine]: 2018 folk 48.8,
Buddhism 13.5, Taoism 12.8, none 13.8, Protestant 5.5, Yiguan Dao etc. 2.3, Buddhism+Taoism/three
teachings 1.1, Catholic 1.3, other 1.0.

### 1.2 TSCS practice items, national frequencies [orig, 2018 report 附錄四]

| item (exact wording) | threshold | 2009 | 2014 | 2018 |
|---|---|---:|---:|---:|
| Q28 「請問您目前有沒有祭拜祖先？」 Do you currently worship ancestors? Instruction: subjective; any act of worship counts, including worship while tomb-sweeping; 追思 (remembrance only) does not | yes | 86.6 | 88.2 | 83.1 |
| Q55 「為了宗教上的目的，在您家中有沒有神龕、神明桌或是擺放宗教物品（譬如，八卦、符咒、神像、避邪鏡、金錢豹、蟾蜍、門神、劍獅、照壁、照片、真言、咒語、護身符、十字架、經典等）」 Home shrine, god table, or any religious object (bagua, talisman, statue, mirror, cross, scripture...); counts whoever put it there | yes | 72.5 | 69.5 | 71.3 |
| Q56 「為了宗教上的目的，您多久到寺廟、道場或教堂去祈求、禱告或從事其他的儀式活動？(請不要包括您例行參與法會或禮拜的次數)」 How often do you go to a temple, dao chang or church to pray or do rituals for a religious purpose (not routine services) | monthly+ | 10.4 | – | 12.4 |
|  | several times a year+ | 28.0 | – | 31.4 |
|  | never | 39.9 | – | 29.0 |
| Q52 「您多常祈禱或向神祈求？」 How often do you pray or ask the gods for something. Instruction: 拜拜 at home counts if asking a god; asking ancestors does not | monthly+ | 43.3 | – | 40.1 |
|  | never | 22.2 | – | 16.8 |
| Q57 「去年一年內…您有沒有去過神壇？」 Visited a 神壇 (private shrine/medium's altar) last year | often / sometimes | 1999: 6.0 / 30.4; 2004: 5.8 / 33.3 | – | 5.0 / 26.2 |
| Q58 進香 pilgrimage last year | yes | 17.7 | 13.9 | 20.1 |
| Q32 「請問您常不常參加宗教活動(例如：進香、禪修、做禮拜、靈修聚會、宗教志工服務)」 religious activities (pilgrimage, meditation, services, volunteering) | monthly+ | – | – | 16.8 |
| 2014 V24 「請問您目前大約多久去一次寺廟、神壇或教會？」 how often go to a temple, shrine or church | monthly+ | 2009 (fqtmalch): 34.3 | 39.6 [mine, weighted] | – |

**Label trap in ARDA's English files:** 2018 `V28` is labelled "Do you worship the gods or ancestors?" but
the Chinese question is ancestors only (祭拜祖先); 陳杏枝 says the 2004 wording was 有無拜神、拜祖先.
2018 `V32` is labelled "How often do you go to a temple, an altar, or a church?" but the Chinese Q32 is
參加宗教活動 (religious activities). Anyone using the ARDA labels should check the Chinese.

### 1.3 Cross-tabs by religion answer [mine, weighted, ARDA OSF Stata files]

2018 (TSCS181, WR_19_5), % within each answer:

| answer | pop % | n | 祭祖 ancestors | shrine/objects | prays to gods monthly+ | ...several/yr+ | temple for religious purpose monthly+ | ...several/yr+ | MID | WIDE | ancestors & shrine & WIDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| folk | 48.8 | 907 | 91.6 | 72.5 | 34.3 | 59.4 | 8.0 | 28.5 | 35.6 | 63.9 | 49.3 |
| Buddhism | 13.5 | 256 | 85.8 | 79.0 | 50.3 | 65.8 | 15.3 | 31.7 | 53.9 | 70.5 | 53.4 |
| Taoism | 12.8 | 228 | 91.3 | 73.8 | 49.4 | 76.4 | 17.9 | 44.0 | 51.9 | 79.7 | 55.4 |
| none | 13.8 | 244 | 63.9 | 47.2 | 14.9 | 29.1 | 2.9 | 13.4 | 16.8 | 33.6 | 14.3 |
| Protestant | 5.5 | 102 | 23.7 | 72.2 | 83.5 | 93.0 | 32.0 | 52.4 | 83.5 | 94.1 | 13.5 |
| ALL | 100 | 1,842 | 83.0 | 70.7 | 40.2 | 61.1 | 12.4 | 31.5 | 42.2 | 65.3 | 44.8 |

MID = prays to/asks gods at least monthly OR temple for religious purpose at least monthly.
WIDE = the same at "several times a year or more".

**Population share of non-Christian practitioners (2018)**: MID **36.7%** (folk 17.3, Buddhism 7.3,
Taoism 6.6, none 2.3, Yiguan Dao etc. 1.5, other 0.8, Bud+Tao 0.7); WIDE **59.1%** (folk 31.2, Taoism
10.2, Buddhism 9.5, none 4.6, Yiguan Dao 1.7, other/multi 1.8); ancestors & shrine & WIDE 43.4%.

2009 (TSC09, WEIGHT): MID non-Christian **39.3%**, WIDE 61.5%. Folk label 42.7% → 48.8% and Buddhism
19.6% → 13.5% between 2009 and 2018 while the practitioner share barely moved (39 → 37): the relabelling
is on the card, not in practice.

Folk subcodes 2018: 022 worships the gods (n 855) ancestors 91.8, shrine 72.5; 021 self-identified
(n 27) 82.4, 62.4.

2014 (TSCS142): ancestors folk 94.2 / none 71.4 / all 88.0; shrine folk 69.5 / none 46.2; temple/shrine/
church monthly+ folk 34.4 / none 14.5 / Buddhism 46.3 / Taoism 51.8; never: none 14.0, folk 1.0;
pilgrimage last year folk 15.8 / none 5.1; festival or ceremony last year folk 43.3 / none 20.4.

### 1.4 Chu Hai-yuan's 1985 behavioural re-sort [orig]

瞿海源 1988〈台灣民眾的宗教信仰與宗教態度〉, in 楊國樞、瞿海源編《變遷中的台灣社會》pp.239-276,
中研院民族所; reprinted as ch.2 of his collected 台灣宗教研究 volume. Open PDF:
https://www2.ios.sinica.edu.tw/people/hyc/essay/9ROS/0102台灣民眾的宗教信仰與宗教態度.pdf

First TSCS (1985), "四千多個" respondents, 20+:
- self-identified Buddhist 47%, Taoist ~7%, folk ~29%, Catholic 1.7, Protestant 3.5, none ~9%, Muslim 0.1, other 2.7.
- Only 6% eat vegetarian, chant sutras or attend Buddhist 法會. Excluding those who worship Mazu, Guangong,
  Tudigong, "at most about 15%" are true Buddhists; two thirds of self-called Buddhists are folk.
- Folk incl. self-called Buddhists worshipping non-Buddhist gods: "應在65%以上" (65% or more).
- 51.4% of men and 42.2% of women rarely or never go to temples; Chu says many of them are probably
  close to no-religion.

### 1.5 Chen Hsing-chih, ancestor belief [orig]

陳杏枝〈祖先信仰的變遷〉, TSCS 11th conference 「台灣的社會變遷1985～2005」(stage 1; year ~2006
[unverified]). https://www2.ios.sinica.edu.tw/sc/cht/files/conf11-1/D1.pdf

- TSCS 1990: 93% worship ancestors; at festivals 82.8%, death anniversaries 50.8%, birthdays 16.9%, daily 12.4%.
  2.2% in a 祭祀公業, 5.7% in a clan association.
- 住家中供奉祖先牌位 (ancestral tablet at home): 61% in 1994 and 1999 (Hokkien 66/65, Hakka 42/50 because
  tablets stay in ancestral halls, Mainlanders 55/44).
- Fengshui done for ancestral graves 64% (1994, 1999).
- "Has worshipped ancestors", under-65 sample: 0.93 (1990), 0.90 (1994), 0.87 (1999), 0.85 (2004, wording 有無拜神、拜祖先).

### 1.6 Chang & Lin: the "no religion" follow-up [via Zhang & Lu 2020 and 2018]

張茂桂、林本炫 1992〈宗教的社會意象：一個知識社會學的課題〉《中研院民族學研究所集刊》74:95-123
(original not read; survey year and n not seen). Of self-declared no-religion respondents, 60% said they
believe in gods (信神); of the 40% who did not, 70% worship gods (拜神); only **6.3% of respondents** were
truly non-religious. Also Soong & Li 1988: 62% of Taiwan's self-declared non-religious believe in fengshui,
a third in auspicious days.
Via: Zhang Chunni & Lu Yunfeng 2020, "The measure of Chinese religions: Denomination-based or
deity-based?", Chinese Journal of Sociology 6(3):410-426, doi 10.1177/2057150X20925312 (open PDF on
isss.pku.edu.cn); Chinese original 张春泥、卢云峰 2018《社会》38(5), https://html.rhhz.net/society/html/20180505.htm

### 1.7 Pew 2023 Taiwan [orig]

Pew Research Center 2024, *Religion and Spirituality in East Asian Societies*, 17 June 2024.
https://www.pewresearch.org/wp-content/uploads/sites/20/2024/06/PR_2024.06.17_religion-in-east-asia_report.pdf
Taiwan n = 2,277, adults 18+, CATI (landline + mobile RDD), Hokkien and Mandarin, 2 June - 17 Sept 2023,
MoE ±2.64.

Identity (p.30): none 27, Buddhist 28, Daoist 24, **Christian 7**, some other religion 12 (Muslim,
Confucian, local/Indigenous, combination). The "10%" in one web summary is Vietnam's Christian figure.
Local/Indigenous about 5% [unverified: search snippet of Pew's short read, which returned 403].
Question: "What is your religion, if any? Buddhist; Catholic, Protestant or other Christian; Muslim;
Daoist; Confucian; Local religions/Indigenous religions; No religion; or Some other religion."
Belief-in-god translation in Taiwan: 「請問您信不信神?」 [via China-Zentrum].

| item | all | unaffiliated | Buddhists | Christians | Daoists |
|---|---:|---:|---:|---:|---:|
| altar in home (p.77) | 49 | 37 | 58 | 15 | – |
| burned incense for ancestors, past 12 months (p.81, 83) | 81 | 72 | 87 | 29 | 96 |
| offered food/water/drinks to ancestors, past 12 m | 77 | 69 | 83 | 31 | 87 |
| offered money/goods for the afterlife, past 12 m | 70 | 61 | 72 | 25 | 87 |
| currently pray or offer respects to Guanyin (p.67, 70) | 69 | 43 | 89 | 17 | 81 |
| ... Buddha | 46 | 23 | 71 | 12 | 42 |
| ... Mazu | "two-thirds" | 43 | 79 | 17 | 82 |
| ... Guandi | "over half" | 35 | 61 | 11 | 69 |
| generally go to temples or pagodas (p.75) | – | 53 | 90 | 22 | 85 |
| generally go to shrines | – | 40 | 72 | 17 | 81 |
| pray daily (p.67) | 16 | 3 | 21 | 43 | 14 |
| told ancestors about life events, past 12 m (p.85) | 38 | 26 | 44 | 34 | – |

### 1.8 Registered temples (venue density) — not reached

全國宗教資訊網 (religion.moi.gov.tw) refused connections twice (ChartReport page and `Report/temple.xml`).
data.gov.tw dataset 8203 lists the temple roll as XML with 行政區 and 教別 fields. tw.md §8.3 already
refused a building layer.

---

## 2. HONG KONG

### 2.1 Hong Kong Political Culture Survey 2021 [orig]

Cai Yongshun & Hung Sin Yu, "Religion and Trust in Hong Kong", *The China Quarterly* **259** (2024),
611-628, doi 10.1017/S0305741023001844, CC BY 4.0.
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4E270D68DABEEB3C68A05D5C4359CC76/S0305741023001844a.pdf/religion_and_trust_in_hong_kong.pdf
**Citation mismatch:** `religiondots/sources/hk.md` cites volume 257, pp. 609-628; the PDF's own header
and "Cite this article" line say 259, 611-628.

Design: May-Sept 2021; 3,744 interviewed (tables 3,740); 16+; 72 of 452 electoral districts at random,
52 per district, quota-matched to the 2016 census on sex, age, education, income, housing; ~30-minute
tablet interviews.

Table 1:

| | 2021 n | 2021 % | 1995 % | 1988 % |
|---|---:|---:|---:|---:|
| Buddhism | 513 | 13.72 | 11.6 | 6.6 |
| Taoism | 151 | 4.04 | – | – |
| Folk religion | – | – | 15.3 | 23.0 |
| Hinduism / Sikhism / Islam | 22 / 2 / 89 | 0.59 / 0.05 / 2.38 | | |
| Protestant | 341 | 9.11 | 8.4 | 7.2 |
| Catholic | 160 | 4.28 | 4.5 | 4.9 |
| No claimed religion | 2,462 | 65.83 | 60.2 | 58.3 (conclusion says 59.3) |
|   practising religious activities | 2,097 | 56.07 | | |
|   not practising | 365 | 9.76 | | |
| total n | 3,740 | | 2,275 | 1,644 |

Source line: "Authors' Hong Kong Political Culture Survey 2021; Cheng and Wong 1997, 301."

**What "practising" means.** Text: "About 56 per cent of the respondents considered themselves to be
non-religious but practised at least one of the above-mentioned folk religious activities". The
"above-mentioned" list is a prose sentence: temple festivals for patron deities, temple visits for
blessings, attending church services, burning incense, ancestor worship, fortune telling, fengshui. No
frequency or time window is given, and no Chinese wording is published. So the threshold is "any one,
ever". 85.2% of the no-religion group practise [mine: 56.07/65.83].

Table 4, all respondents, multiple answers:
- burning incense for dead family member(s) 75.01%
- drawing a fortune stick, evaluating fengshui, etc. 40.60%
- finding a fortune teller 20.24%
- palm reading / computer / tarot by yourself 9.97%
- beliefs: good to be worshipped by posterity 70.37; choose auspicious date 70.37; soul survives 49.76; fengshui first when buying a flat 46.39

Table 3, no religion but practising, by age: 16-24 41.4, 25-44 58.6, 45-64 59.5, 65+ 53.9; no religion and not practising: 28.4, 11.1, 4.7, 5.0.
1995 survey: 55.2% practised ancestor worship (Cheng & Wong 1997).

### 2.2 The 1988 and 1995 affiliation surveys (card offered folk religion)

- 1988: Hui, C. Harry 1991, "Religious and supernaturalistic beliefs", in Lau Siu-kai, Lee Ming-kwan,
  Wan Po-san & Wong Siu-lun (eds.), *Indicators of Social Development: Hong Kong 1988*, HKIAPS, CUHK, 103-143.
- 1995: Cheng, May & Wong Siu-lun 1997, "Religious convictions and sentiments", in *Indicators of Social
  Development: Hong Kong 1995*, HKIAPS, CUHK, 299-330.
- Both print only; no online copy found. Question wording not seen.

### 2.3 HKCC and CUHK Divinity School 2021: folk religion offered as an answer [orig, web summary]

Hong Kong Christian Council + CUHK Chung Chi College Divinity School, 香港市民對基督教觀感調查報告2021.
Fieldwork by HKIAPS Telephone Survey Research Laboratory, RDD, 5 July - 31 Aug 2021, n = 2,013, 18+,
Cantonese or Mandarin. https://www.theology.cuhk.edu.hk/tc/surveyreport2021.html

No religious belief 58.6; Protestant 17.2; Buddhism 10.5; **拜神或傳統中國民間信仰 (worshipping gods or
traditional Chinese folk religion) 6.9**; Catholic 5.2. Taoism not listed separately in the summary.
Full report PDF (76 pp, `Survey Report 2021_SECURED.pdf`) downloaded, but its Chinese text extracts as
garbage (secured, custom font encoding), so the exact question and any Taoism row were not recovered.
Response rates 41.2% / 34.9% (landline / mobile, read from the garbled text; treat as uncertain).

### 2.4 Pew 2023 Hong Kong [orig]

n = 2,012, 18+, CATI, Cantonese / English / Mandarin, MoE ±2.74. Identity: none 61, Buddhist 14,
Christian 20, Daoist 1, other 3. Religion very important 11%.

| item | all | unaffiliated | Buddhists | Christians |
|---|---:|---:|---:|---:|
| altar in home | 25 | 21 | 56 | 11 |
| burned incense for ancestors, past 12 m | 57 | 62 | 84 | 21 |
| offered food/water/drinks to ancestors | 48 | 51 | 68 | 23 |
| offered money/goods for the afterlife | 44 | 49 | 59 | 18 |
| pray or offer respects to Guanyin | 30 | 22 | 82 | 9 |
| ... Buddha | – | 15 | 80 | 8 |
| ... Mazu | – | 11 | 53 | 6 |
| ... Guandi | – | 9 | 36 | 6 |
| generally go to temples or pagodas | – | 5 | 50 | 3 |
| never pray | – | 65 | 32 | 9 |
| told ancestors about life events | 26 | 24 | 29 | 29 |

Temple-goers overall ≈ 0.61×5 + 0.14×50 + 0.20×3 ≈ 10.7%, plus at most a few points from Daoists and
others [mine, rough].

### 2.5 Other Hong Kong leads

- Global Flourishing Study HK (Huang et al. 2025, *International Journal of Wellbeing* 15(3), n = 3,012):
  article PDF at ro.ecu.edu.au returned 403; the IJW supplement (43 pp) reports outcomes by affiliation
  but no affiliation distribution was found in its text.
- gov.hk / Hong Kong Yearbook "over 1 million Buddhists / Taoists": religious bodies' own estimates
  (already argued in hk.md). LegCo fact sheet FS01/17-18 (religious facilities) failed on a TLS certificate error.
- Liu Tik-sang 2003, "A nameless but active religion", CQ 174:373-394: ethnographic, no survey figures;
  cites Stevens 1980's survey of 450 temples in Hong Kong and Macau.

---

## 3. MAINLAND CHINA, for comparison only

All CFPS figures below are **CFPS-derived**, from published reports; no CFPS microdata touched.

- CGSS 2018 [via Pew 2023, *Measuring Religion in China*, https://www.pewresearch.org/wp-content/uploads/sites/20/2023/08/PF_2023.08.30_religion-china_REPORT.pdf]:
  75% visited a family gravesite in the last year (14% three or more times); 24% visited a site,
  typically a temple or shrine, to pray for good fortune in the last year (10% twice or more); 62% care
  somewhat or very much about auspicious days; 4% Buddhist by affiliation; 7% any religion in CGSS 2021.
- CGSS affiliation card offers "Folk religion (e.g., Mazu, Guangong)": 3.5% (2012), 2.2% (2013),
  1.8% (2015) [via Zhang & Lu 2018 summary; not checked against CGSS].
- CFPS 2016 [CFPS-derived, via Pew 2023 p.42 and Appendix B]: 「您烧香/拜佛的频率有多高」 (how often do
  you burn incense / worship Buddha): a few times a year or more **26%**, at least monthly **11%**. Not
  asked of Christians or Muslims; Pew counts them as never.
- CFPS 2018 [CFPS-derived]: 「您是否相信佛或菩萨」 believe in Buddha/bodhisattva 33% (32.4% of 16+);
  Taoist immortals 18% (Pew) / 19.6% (Zhang, Lu & Sheng 2021 Table 1 via China-Zentrum); ghosts 10%;
  fengshui 47%; at least one deity or ghost 40%. Lu & Sheng 2023: "narrow" Buddhists (believe, name no
  other institutional religion, and do not say "never" to incense in 2016) 8.6%.
- CFPS 2014 deity card [CFPS-derived, Zhang & Lu 2020 Table 2, panel n 20,644]: Buddha 16.2, Taoist
  deity 1.1, ancestors 5.3, multiple 0.8, none 73.0 (the same people said none 89.3% on the 2012 card).
- CFPS 2010 [CFPS-derived, Hu & Tian 2018, *Demographic Research* 38(1)]: visited ancestors' gravesite
  70.79%; family genealogy 22.54%.
- CSLS 2007 [via China-Zentrum 2025]: 18% of 16+ self-identify Buddhist, 17.3 million took refuge;
  12 million Daoist, 173 million with some Daoist practice. ">70% did at least one ancestor-worship
  activity" [unverified, abstract snippet of Hu's 2016 paper].

---

## 4. SIDE-BY-SIDE ON ONE MEASURE

| measure | mainland China | Taiwan | Hong Kong |
|---|---|---|---|
| ancestor rite in the last year / currently | 75% visited a family grave (CGSS 2018) | 83% currently 祭拜祖先 (TSCS 2018); 81% burned incense for ancestors (Pew 2023) | 57% burned incense for ancestors (Pew 2023); 75% have burned incense for dead family (HKPCS 2021, no time window) |
| temple for a religious purpose in the last year | 24% to pray for good fortune (CGSS 2018) | 57% went at least once or twice a year to pray or do rituals (TSCS 2018 Q56) | ~11% "generally go to temples" [mine from Pew groups]; 41% ever drew a fortune stick or had fengshui done (HKPCS) |
| incense or prayer to deities at least monthly | 11% incense/拜佛 (CFPS 2016) | 40% pray or ask gods; 35% non-Christian (TSCS 2018 Q52) | no item found |
| ...a few / several times a year | 26% (CFPS 2016) | 61%; 55% non-Christian (TSCS 2018) | no item found |
| home altar | no national figure found | 49% (Pew); 71% any religious object (TSCS) | 25% (Pew) |
| venerate Guanyin now | (33% "believe in" Buddha/bodhisattva, CFPS 2018; different verb) | 69% (Pew) | 30% (Pew) |
| folk religion when offered as an answer | 1.8-3.5% (CGSS 2012-15) | 31% short card 1994, 37% short card 2015; 48% long card 2014/18 (TSCS) | 23.0% (1988), 15.3% (1995), 6.9% "拜神或傳統中國民間信仰" (HKCC/CUHK 2021) |
| Buddhist + Daoist by identity | ~4% + <0.5% (CGSS) | 28 + 24 (Pew); 14 + 13 (TSCS long card) | 14 + 1 (Pew); 13.7 + 4.0 (HKPCS) |
| no religion AND no practice | – | 6.3% (Chang & Lin, ~1990) | 9.8% (HKPCS 2021) |
| unaffiliated who burned incense for ancestors | – | 72% (Pew) | 62% (Pew) |
| unaffiliated who never pray | – | 22% (Pew) | 65% (Pew) |

Note the wording gaps: Taiwan's Q52 「祈禱或向神祈求」 counts asking a god at a home altar and is
broader than CFPS's 「烧香/拜佛」; CGSS asks about praying for good fortune, TSCS Q56 about any
religious purpose.

---

## 5. A "MIDDLE GROUND"

**Taiwan.** The long card's 77% (folk 48.8 + Buddhism 13.5 + Taoism 12.8 + multi 1.1) is everyone who
is not Christian and does not flatly say "no religion", and the folk code explicitly takes in people who
say "no religion but I worship with my family". Practice thresholds on the same 2018 respondents:

- at least monthly prayer to gods or temple visit: **~37%** of adults (39% in 2009)
- several times a year or more: **~59%** (61% in 2009)
- Pew's identity card (Buddhist 28 + Daoist 24 + local ~5 ≈ 57%) lands on the several-times-a-year line.
- Chu Hai-yuan's 1985 behavioural re-sort put folk at 65%+ and true Buddhists at ≤15%.

So Taiwan's practitioner share is about 55-60% at a "several times a year" bar and ~37% at a monthly bar.
Inside the 59%, TSCS splits it folk 31, Taoism 10, Buddhism 9.5, none 4.6, other Chinese 3.5.

**Hong Kong.** Identity with a Chinese religion is 15-18% on every 2021 card (HKPCS 17.8; Pew 15; HKCC
Buddhism 10.5 + folk 6.9 = 17.4). Practice runs 25% home altar, 30% Guanyin, ~11% temple-goers. The 56%
"no religion but practising" uses an "any one activity, ever" bar that includes incense for dead
relatives (75% of everyone), so it is a floor on non-practice rather than a practitioner count. A
middle ground at the altar / deity-veneration bar is **~25-30%** of adults, i.e. roughly 10-15 points
of the 65.8% none on top of the 17.8% Chinese-religion identifiers.

**Mainland China.** 11% burn incense to deities monthly, 26% a few times a year, 24% went to a temple
for luck last year; identity with Buddhism, Taoism or folk religion is ~6-8%.

**On the one measure that runs through all three** (religious practice several times a year, not ancestor
rites): China ~26%, Hong Kong ~25-30% by proxy (no frequency item), Taiwan ~55-60%. Ancestor rites alone
are near-universal in all three (57-83%) and do not separate practitioners from anyone else.

---

## 6. DEAD ENDS

- **Yang & Hu 2012, JSSR 51(3):505-521** (doi 10.1111/j.1468-5906.2012.01660.x): Wiley 403; ResearchGate
  request-only; Semantic Scholar has no open PDF; Fudan sociology page abstract only; NCCU lecture-notes
  PDF (sino-college.ccstw.nccu.edu.tw) failed with an SSL WRONG_VERSION_NUMBER; Chinese search found the
  three-type framework but no figures. Its measures and numbers were not obtained.
- **Zhang, Lu & Sheng 2021** (shehui.pku.edu.cn PDF): fetch failed with no output; one figure via China-Zentrum.
- **Hu Anning 2014** "Gifts of Money and Gifts of Time" (TSCS 2009 + REST 2009): abstract only.
- **Pew Taiwan short read** (pewresearch.org/short-reads/2024/07/29/...): 403.
- **MOI 全國宗教資訊網**: connection refused twice.
- **HK Indicators of Social Development 1988 / 1995 chapters**: print only.
- **Global Flourishing Study HK article**: 403; supplement has no affiliation table.
- **HKCC/CUHK 2021 full report**: text layer unreadable.
- **LegCo FS01/17-18**: TLS certificate error.
- **Chang & Lin 1992 original** and **Chen Hsing-chih's conference year**: not verified.
- **Pew's "local/Indigenous 5%" for Taiwan**: snippet only.
- WebFetch cannot read these PDFs directly; saving the binary and extracting with PyMuPDF worked
  (`pdf2txt.py` in this scratchpad).
