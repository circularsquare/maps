# Mainland China: how many people practise folk religion, Buddhism, Daoism

Research notes for religiondots, 2026-09-15. No repo file was edited. One local file was read
read-only: `religiondots/data/raw/cn/slsc2007.dta` (ARDA, already in the repo). Every figure below is
% of adults unless stated. **CFPS-derived** marks anything that ultimately comes from China Family
Panel Studies; nothing from CFPS microdata was downloaded or touched.

---

## 0. Bottom line

- The two extremes are both real measurements of different things. "Names a religion (zongjiao)" is
  10-12% (CGSS 2018 10%, CLDS 2016 12.2%). "Did any ancestor/deity/fengshui thing or holds any such
  belief" is 70-75% (CSLS 2007 grave veneration 72%; CGSS 2018 grave visit 75%; CFPS 2018 any of
  eight beliefs 75%, CFPS-derived).
- **A practice threshold of "worshipped deities at a temple or a home altar in the past year" lands
  at about 25-30% of adults, from four independent instruments:**
  - CSLS 2007: worshipped gods/spirits at a religious site or at home in the past year, 27.2%
    (weighted, tabulated here from the ARDA file);
  - CSLS 2007: has a Buddhist image, god-of-wealth image, other deity image or ancestral tablet at
    home, 28.0%;
  - CFPS 2016: burns incense to worship Buddha/deities a few times a year or more, 26%
    (CFPS-derived, via Pew 2023);
  - CGSS 2018: went somewhere to pray for good luck in the past year, 24% (via Pew 2023).
  - ECNU 2005 (Tong/Liu, 16+): 31.4% "religious", of whom about two-thirds Buddhist, Daoist or
    worshippers of folk deities, so roughly 21% of adults. Original never found.
- Pew's 2012 figures (folk 21.9%, Buddhist 18.2%) are **the CIA Factbook's figures**, and they add to
  40%, above that cluster, because the Buddhist 18% is a *belief* item and the folk 22% is a
  belief-or-practice composite. Pew abandoned both in 2025.
- Inside the 25-30%, the split is roughly Buddhist-flavoured 10-11%, god of wealth / earth god /
  other folk deities 8-9%, ancestral tablets 12%, Daoist ~1% (all CSLS 2007, overlapping, see §2.1).
  No survey separates "folk" from "Buddhist" practice cleanly; incense items are asked as 烧香/拜佛.
- **No survey anywhere gives this middle-ground measure by province.** The only provincial practice
  figures found are CFPS 2010 grave-visiting and genealogy bands (Hu & Tian 2018, CFPS-derived), plus
  registered-venue geography (NRAA Buddhist/Daoist temples by province; Zhejiang folk temples 2013 by
  prefecture).

---

## 1. Summary table

| source | year | measure & threshold | China figure | geography | access |
|---|---|---|---|---|---|
| CSLS / SLSC (Horizon Research, Yang Fenggang, Purdue) | 2007 | worshipped gods/spirits at religious site or home, past year | **27.2%** (w), 25.1% unw | national, 56 sampling points, no provincial cut | ARDA free; in repo |
| same | 2007 | deity/Buddhist object or ancestral tablet at home | **28.0%** w | same | same |
| same | 2007 | prayed/worshipped/burned incense *in a temple* (Buddhist, Daoist, ancestral, other), past year | 15.9% w | same | same |
| CFPS 2016 via Pew 2023 (**CFPS-derived**) | 2016 | 您烧香/拜佛的频率有多高？ a few times a year or more | **26%**; once a year+ 35%; monthly+ 11% | national | Pew PDF open |
| CGSS 2018 via Pew 2023 | 2018 | 在过去的一年中，您有多少次去过一个地方祈求好运 at least once | **24%** | national (28 provinces sampled) | Pew PDF open; microdata needs CNSDA application |
| ECNU (Tong Shijun, Liu Zhongyu) | fieldwork 2005, reported Feb 2007 | "religious believers", 16+; wording unknown | 31.4% (~300M); ~2/3 of believers Buddhist/Daoist/folk-deity (~200M) | national, 4,500 respondents | press reports only |
| Pew GRL 2012 = CIA Factbook | data 2007, published 2012 | folk: belief-or-practice composite; Buddhist: believes in "Buddhism or Buddha" | folk 21.9%, Buddhist 18.2%, unaffiliated 52.2% | national | open PDF |
| Zhang, Lu & Sheng 2021 (**CFPS-derived**) | 2018 | 您是否相信… 8 forced-choice yes/no items; "folk" = 2+ beliefs or single belief in ancestors/ghosts/geomancy | 70% folk, 5% institutional, 25% none | national | open PDF |
| Hu & Tian 2018 (**CFPS-derived**) | 2010 | visited ancestors' grave; family has a genealogy | provincial bands, see §4 | **province (25)** | open PDF |

---

## 2. Sources in detail

### 2.1 Chinese Spiritual Life Survey 2007 (CSLS; ARDA title "Spiritual Life Study of Chinese Residents")

- **Who/design.** Horizon Research Consultancy (零点研究咨询集团) fieldwork May 2007; designed with
  Yang Fenggang (Purdue Center on Religion and Chinese Society); ARDA deposit
  `thearda.com/data-archive?fid=SPRTCHNA`. 7,021 face-to-face interviews, adults 16+, multi-stage,
  56 locales: Beijing, Shanghai, Chongqing; 6 provincial capitals (Guangzhou, Nanjing, Wuhan, Hefei,
  Xi'an, Chengdu); 11 regional cities; 16 small towns; 20 administrative villages. AAPOR response rate
  28.1%. Pew: "No major cities in the west, the far northeast or on the south-central coast were
  surveyed" (Pew GRL 2012 App. A fn 37).
- **Original measurement.** The figures below are my own tabulation of the ARDA file, weighted by
  `weight` (file label: `WEIGHT1 * WEIGHT2`). WEIGHT2 takes three values (0.477, 0.517, 2.548) and
  moves villages from 24.8% of the sample to 63.1% of the weight, which is why rural practices
  (graves, ancestral tablets) rise when weighted. Scripts and full output in the scratchpad:
  `slsc_items.py/.txt`, `slsc_multi.py/.txt`, `slsc_exist.py/.txt`. Question text is ARDA's English
  translation; the Chinese questionnaire was not found.
- **Checks against published CSLS numbers.** Published: 185M Buddhists = 18% of 16+ → my BELIEVE1
  Buddhism 18.1% w. Published: 754M did some ancestor worship in the past 12 months → my ACTIVTY
  "venerate ancestral spirits by graves" 72.4% w. Published: 17.3M took refuge → convbudd ceremony
  1.7% w. The tabulation reproduces the headline numbers.

**Identity and belief**

| item | wording (ARDA English) | weighted | unweighted |
|---|---|---|---|
| religblf | Do you have any religious belief? — yes | 17.8 | 15.8 |
| BELIEVE1 | Regardless of whether you have been to churches or temples, do you believe in any [religion]? — Buddhism | 18.1 | 16.6 |
| | — Daoism (first mention; +0.6 second mention) | 0.6 | 0.3 |
| | — nothing | 77.1 | 78.1 |
| EXIST16 | Do you think [god of wealth] exists? yes | 13.8 | 12.3 |
| EXIST17 | [ancestral spirits] | 20.3 | 16.0 |
| EXIST8 | [Buddha] | 17.6 | 16.2 |
| EXIST2 | [god of heaven] | 12.4 | 9.8 |
| EXIST13 | [gods, spirits] | 5.8 | 5.2 |
| EXIST14 | [ghosts] | 5.4 | 5.4 |
| EXIST15 | [fate, fortune] | 25.4 | 25.4 |
| EXIST10 / 18 | [karma] / [karma in personal relationships] | 20.5 / 42.1 | 21.2 / 43.2 |
| any of god of heaven, gods/spirits, god of wealth, ancestral spirits | | 29.1 | 24.1 |

EXIST slots were mapped to items by matching each slot's "yes" count with the non-missing count of
its follow-up INFLxxx question (asked only after a yes): all 18 agree within 1-4%.

**Practice**

| item | wording | weighted |
|---|---|---|
| WORSHIP (multi) | Have you worshipped God or gods/spirits in the following settings in the past year? — conventional religious settings (churches, Buddhist/Daoist temples) | 17.4 |
| | — at the grave of a deceased family member or an ancestral temple | 54.1 |
| | — at home | 15.0 |
| | — "I never worship" | 35.2 |
| ACTIVTY (multi) | In the past year, which activities did you take part in? — venerate ancestral spirits by their graves | 72.4 |
| | — pray, worship and/or burn incense in Buddhist temples | 10.7 |
| | — same in Daoist temples | 1.3 |
| | — same in other temples | 4.1 |
| | — same in ancestral temples | 1.6 |
| | — none | 21.2 |
| PAST12 (multi) | Done in the past twelve months: venerate ancestors 29.2; burn incense 12.7; make and fulfil a vow 6.4; worship the Buddha 5.2; pray 5.2; vegetarian for Buddhist reasons 3.2; none 59.2 | |
| frqburn | burn incense regularly / occasionally | 2.8 / 9.8 |
| frqvener | venerate ancestors regularly / occasionally | 1.7 / 27.2 |
| ITEMS (multi) | Items at home: ancestral tablets 12.1; Mao image 11.5; Buddhist objects 9.9; god of wealth 9.3; other gods such as the earth god 8.8; Christian 2.4; Daoist 0.2; none 61.7 | |
| WRKITMS | At the workplace: god of wealth 4.5 | |
| WEARACC | Worn in the past year: Buddhist objects 6.0; other protective talisman 3.5; Daoist 0.7 | |
| RDONE | Ever done or asked others to: fortune telling 7.8; feng shui 3.1; astrology 1.7 | |
| RDONEL | Talismans at the house entrance 9.7; fireworks against bad luck 16.0; auspicious wedding day 7.3 | |
| WHOMPRY | Prays to: Buddha 6.3; ancestral spirits 2.9; god of wealth/land/kitchen 1.7; Jade Emperor 0.7 | |
| donated | Donated money or goods to religious organisations | 20.6 |

**Composites (my construction, % of all adults, weighted)**

| threshold | % |
|---|---|
| burns incense regularly | 2.8 |
| burned incense in past 12 months | 12.7 |
| worshipped in a Buddhist/Daoist/ancestral/other temple, past year | 15.9 |
| self-identifies as having a religious belief | 17.8 |
| worshipped at temple/church or at home, past year AND (deity object at home OR incense) | 17.3 |
| worshipped at temple/church or at home, past year | **27.2** |
| deity object, Buddhist object or ancestral tablet at home | **28.0** |
| either of the previous two | 40.5 |
| venerated ancestors at graves, past year | 72.4 |

Christians and Muslims are not separated out of the composites, but they are about 3-4% of the
sample and mostly fall under "church" or no item, so the effect is small. Not checked.

**Published CSLS-based analyses**
- Yang Fenggang & Hu Anning 2012, "Mapping Chinese Folk Religion in Mainland China and Taiwan",
  *JSSR* 51(3): 505-521. Paywalled (Wiley); Fudan page is abstract only. Per Zhang et al. 2021 it
  counts "no religion but believes in gods", plus fengshui, fortune telling, god of wealth and
  amulets, as folk; result 55.5%.
- Hu Anning 2016, "Ancestor worship in contemporary China: an empirical investigation", *China Review*
  16(1): 169-186. Abstract: ancestor-worship participants are "over 70 percent of the adult
  population" (CSLS). Paywalled.
- CSLS numbers as relayed by Wenzel-Teuber (China-Zentrum, RCTC 2023 no. 2, citing Wenzel-Teuber
  2012): 185M Buddhists (18% of 16+); 17.3M took refuge; 173M "have exercised some Daoist practices";
  about 12M Daoists (1.17%, her own calculation).
- Fang Wen, 《中国宗教图景上的浮尘》, 宗教社会学 vol. 2. PKU PDF, but the CJK text doesn't render; the
  visible numbers are CSLS 16+: atheists 15%, 85%, Buddhists 18%, Christians 3.2% / 33M, citing Yang
  et al. 2010 "Quantifying Religion in China".

### 2.2 CFPS-based papers (all **CFPS-derived**)

**Lu Yunfeng (Lu Yunfeng / 卢云峰) 2014.** 《当代中国宗教状况报告——基于CFPS(2012)调查数据》,
*世界宗教文化* 2014(1). CNKI `CJFDTOTAL-RELI201401004`; PKU IR `ir.pku.edu.cn/handle/20.500.11897/201151`.
- Question (CFPS 2012 M601): 请问您属于什么宗教？ ("Which religion do you belong to?") with no folk option.
- ~10% religious; Buddhist 6.75%; Protestant 1.9% (~26M).
- The five oversampled provinces, from a search-engine summary of the article (full text not read):
  Liaoning and Guangdong have the fewest religious respondents; Shanghai's are mostly Buddhist;
  Henan Protestant 5.6%; Gansu Muslim 3.4%.
- Full text unreachable (see §6).

**Zhang Chunni & Lu Yunfeng 2018.** 《如何在社会调查中更好地测量中国人的宗教信仰？》, *社会* 38(5): 126-157.
Open at `html.rhhz.net/society/html/20180505.htm` and `shehui.pku.edu.cn/upload/editor/file/20181214/20181214152131_3345.pdf`.
- CFPS 2012 (请问您属于什么宗教？) against CFPS 2014 (M601A 您信什么？(可多选): 佛/菩萨, 道教的神仙,
  安拉, 天主教的天主, 基督教的上帝, 祖先, 以上都不信).
- Same panel respondents, 2012 → 2014: Buddha/bodhisattva 4.4 → 16.2; Daoist immortals 0.1 → 1.1;
  ancestors 0.2 → 5.3; none 89.3 → 73.0.
- 2014 typology: organised active believers 0.8%, unorganised active believers 18.3%, non-believers
  76.6%. Relayed through a WebFetch summary; the exact definitions were not read.

**Zhang Chunni, Lu Yunfeng & Sheng He 2021.** "Exploring Chinese folk religion: popularity, diffuseness,
and diversities", *Chinese Journal of Sociology* 7(4): 575-592, DOI 10.1177/2057150X211042687. Open
PDF at `shehui.pku.edu.cn/upload/editor/file/20230120/20230120144957_3012.pdf`.
- CFPS 2018, 29,996 respondents aged 16+.
- Question: 8 forced-choice yes/no items on whether the respondent believes in 佛/菩萨, 道教的神仙,
  真主安拉, 天主, 耶稣基督, 祖先, 鬼, 风水.
- Table 1, % of total (weighting not stated):

  | belief | single | multiple | total |
  |---|---|---|---|
  | none | | | 25.2 |
  | Buddha/bodhisattva | 2.1 | 31.3 | 33.4 |
  | Daoist deities | 0.4 | 19.2 | 19.6 |
  | ancestors | 14.5 | 43.4 | 57.9 |
  | ghosts | 0.3 | 10.0 | 10.3 |
  | geomancy | 6.5 | 40.4 | 46.9 |

- Threshold: folk = believes 2+ items, or a single belief in ancestors/ghosts/geomancy. Result 70%
  folk, 5% institutional, 25% none.
- Latent classes: non-believers and single-belief 46.1%; "believers of geomancy" 30.3%; "diffused
  Buddhism and Daoism" 20.3%; "embracing all beliefs" 3.3%.
- Fig. 1 CGSS self-id folk: 2.8% (2006), 0.2 (2008), 2.3 (2010), 1.9 (2011), 2.0 (2013), 1.7 (2015),
  2.1 (2017).
- **This is the broad extreme, not a middle ground.**

**Min Li 2022** (Sichuan University), relayed by China-Zentrum RCTC 2023 no. 2, fn 9.
- CFPS 2018: Buddhism 27.03%, Daoism 15.89%, ancestors 46.89%, ghosts 8.33%, geomancy 37.9%.
- Denominator unclear: 37,356 respondents, possibly including under-16s.

**Pew 2023 compilation of CFPS 2016/2018** (weighted by Pew).
- 2018 believes in (相信) Buddha/bodhisattva 33% (16% only Buddha, no other deity); immortals 18%;
  ghosts 10%; fengshui 47%; any deity item 38%; none 61%.
- CFPS 2016 M602A 您烧香/拜佛的频率有多高？ (从不 / 一年一次 / 一年几次 / 一月一次 / 一月两三次 / 一周一次 /
  一周几次 / 几乎每天), asked only of non-Christian, non-Muslim respondents; Pew counts the rest as
  "never".
  - once a year or more 35%; a few times a year or more 26%; monthly or more 11%.
  - By sex: women 30%, men 21% (few times a year+).
  - Unaffiliated 21% (few times a year+), 8% monthly+; affiliated 55%, 34%.
- CFPS 2014 "您信什么" Buddha/bodhisattva 17% (the word 信 is stronger than 相信).

**Hu Anning & Felicia F. Tian 2018.** "Still under the ancestors' shadow? Ancestor worship and family
formation in contemporary China", *Demographic Research* 38(1). Open PDF at
`demographic-research.org/volumes/vol38/1/38-1.pdf`.
- CFPS 2010 adult sample, 25 provinces (excluding Xinjiang, Qinghai, Inner Mongolia, Ningxia, Hainan).
- Two items: family keeps a genealogy; visits ancestors' gravesite.
- Figure 1 is a **province-level map** in bands, raw data. See §4.

### 2.3 CGSS 2018 practice items (via Pew 2023, Appendix B)

- A5 您的宗教信仰是什么？ (single choice in 2010, 2018, 2021; multi-select 2012-2017), with option
  13 民间信仰（拜妈祖、关公等）. Affiliated 10%; folk ~3%; Daoist <0.5%; Buddhist 4%.
- A6 您参加宗教活动的频繁程度是 — a few times a year or more: 11% (2012), 8 (2013), 7 (2015),
  6 (2017), 6 (2018), 3 (2021).
- B2 请问在特殊的情况如结婚，搬迁或丧礼时，您会不会在意吉日或凶日？ ("On occasions like weddings, moves
  or funerals, do you care whether a day is auspicious?") — somewhat or very much 62%.
- B3 在过去的一年中，您去过多少次已故家庭成员的墓地？ ("How many times in the past year have you visited
  the graves of deceased family members?") — at least once 75%; three or more times 14%
  (women 11, men 16); never 24%.
- B4 在过去的一年中，您有多少次去过一个地方祈求好运（学术和商业成功，健康等）？ ("How many times in the
  past year have you gone somewhere to pray for good luck, e.g. study, business, health?") — at
  least once **24%** (women 27, men 21); unaffiliated 21%, affiliated 45%.
- B5 您是否随身戴着祝好运的符咒和/或护身符？ ("Do you carry a lucky charm or amulet?") — 8%.
- CGSS 2017+2018 pooled self-id by ethnicity: Han — none 91, Buddhist 4, folk 2, Christian 2;
  Zhuang — folk 8; other minorities — folk 5, Buddhist 5.
- **Provincial:** CGSS 2018 carries province, but Pew publishes none of these by province, and no
  published provincial analysis of B3/B4 was found (Chinese and English searches). The repo notes say
  2018 microdata needs a CNSDA application.

### 2.4 East China Normal University survey (Tong Shijun, Liu Zhongyu)

- **Found:**
  - 31.4% of Chinese aged 16+ are religious believers, ~300M.
  - The five main religions are 67.4% of believers.
  - "About 200 million are Buddhists, Taoists or worshippers of legendary figures [Dragon King, God of
    Fortune], accounting for 66.1 per cent of all believers."
  - Christians 12% of believers (~40M).
  - 62% of believers aged 16-39; 9.6% aged 55+.
  - Sources: China Daily 2007-02-07 "Religious believers thrice the estimate" as quoted by Global
    Voices (2009-02-18) and Al Jazeera (2007-02-07).
- **Sample:** 4,500 respondents (Al Jazeera). Results also in 瞭望东方周刊 (Oriental Outlook), Feb 2007.
- A search-engine summary (source page not identified) says fieldwork was summer 2005, run by ECNU's
  中国现代思想文化研究所 as part of an MOE major project 《中国人的精神生活》. **Unverified.**
- **Not found:** question wording, threshold, sampling design, full breakdown, any provincial data. No
  paper or report located.
- Relation: "about 200M Buddhist/Daoist/folk-deity" is ~20% of the 16+ population then. That sits
  inside the 25-30% practice cluster once you allow for a self-report threshold. Original measurement,
  but only reported secondhand.

### 2.5 Yu Tao 2008 rural survey (six provinces) — compiler chain only

- Rural populations of Jiangsu, Sichuan, Shaanxi, Jilin, Hebei, Fujian. Survey scheme led by the
  Center for Chinese Agricultural Policy (CCAP) and Peking University.
- Folk religion 31.9% (Wikipedia article) or 31.09% (Wikipedia template), a discrepancy; Buddhist
  10.85%; Christian 3.93% (Protestant 3.54, Catholic 0.39); Daoist 0.71%; none 53.41%.
- Wikipedia cites "Yu Tao (2012), ECRAN presentation, academia.edu". No paper found; wording unknown.
- **Might carry a six-province split in the original.** Rural only.

### 2.6 Yao Xinzhong 2005 urban survey — compiler chain only

- Wikipedia template "Religion in China surveys":
  - "cults of gods/ancestors" 23.8%; Buddhist 23.1%; 51.8% "not members of religions";
  - folk practices (fengshui, celestial beliefs) 38.5%; convinced atheists 32.9%.
- Source: Yao 2007, "Religious Belief and Practice in Urban China 1995-2005", *J. Contemporary
  Religion* 22(2): 169-185 (paywalled).
- Related book: Yao & Badham 2007, *Religious Experience in Contemporary China* — 3,196 respondents,
  10 sites, three regions, 2004-07. An open review has no religion shares.
- Not verified at source.

### 2.7 Other surveys

- **China Values Survey 2004** (Shanghai Academy of Social Sciences + Peking University). Li Yaojun
  et al. 2008, "Who are the believers in religion in China?", in *Religion and the Individual*
  (Ashgate). Open at `pure.manchester.ac.uk/ws/files/32297712/FULL_TEXT.PDF`.
  - 3,267 respondents aged 18+; 65 counties in 23 provinces.
  - Question 你信教吗？ ("Do you believe in religion?") — yes 11.4%; Buddhism 4.4%; Daoism <1%.
- **Ruan Rongping & Wang Bing 2011**, 《差序格局下的宗教信仰和信任——基于中国十城市的经验数据》, *社会*
  31(4): 195-217. Public Life Attitudes Survey 2005-07; 1,650 respondents in 10 cities (Beijing,
  Shijiazhuang, Baoding, Taiyuan, Taigu, Chengdu, Luojiang, Mianyang, Shehong, Ziyang).
  - Five institutional religions 15%; "inclusive" (adds ghosts, immortals, souls) 17%.
- **WVS 2018 China** (Pew): a religious person 16%, not religious 49%, atheist 34%. Religion very or
  rather important 13%.
- **Buddhist Association of China 2017**: >100M Buddhists (不完全统计, "incomplete statistics"),
  about 9% of adults (via Pew).
- **Li Xiangping 2011 Yangtze Delta survey** (长三角地区信仰与宗教信仰调查, ECNU). Residents aged 18-70,
  3,000 questionnaires, PPS/Kish sampling; Shanghai, Nanjing, Hangzhou, Ningbo, Suzhou, Wuxi and 10
  other prefectures. **Results not found online.** It would be a regional practice source for
  Shanghai, Jiangsu and Zhejiang if located.

### 2.8 Venue counts (geography, not magnitude)

- **NRAA registered sites** (宗教活动场所基本信息 database at sara.gov.cn, via Pew 2023 and China-Zentrum
  2023):
  - Buddhist 34,090 (Han 28,528; Tibetan 3,857; Theravada 1,705); Daoist 8,349 (Zhengyi 4,338;
    Quanzhen 4,011).
  - Share of Han Buddhist temples by province: Zhejiang 14%, Fujian 12%, Hunan 11%, Jiangxi 11%,
    Hubei 7%, Anhui 5%, Sichuan 5%, Guangdong 5%.
  - Share of Zhengyi Daoist temples: Zhejiang 25, Jiangxi 20, Fujian 20, Hunan 16.
  - Share of Quanzhen Daoist temples: Gansu 15, Hubei 14, Zhejiang 14, Shaanxi 11, Henan 8, Hunan 6,
    Sichuan 5, Hebei 5.
  - A full provincial table is in Wenzel-Teuber 2016, RCTC, p. 27, Table 1 (not fetched).
  - The NRAA database does **not** cover folk-religion sites.
- **Zhejiang folk temples, 2013 survey by the provincial ethnic and religious affairs commission**.
  叶涛《浙江民间信仰现状及其调研述略》, read on the chinesefolklore.org.cn mirror
  (`chinesefolklore.org.cn/web/index.php?NewsID=6991`); original mzw.zj.gov.cn 2013-05-03 page now 404.
  - Total 33,678: Hangzhou 1,253; Ningbo 4,058; Wenzhou 8,579; Huzhou 2,000; Jiaxing 753; Shaoxing
    1,691; Jinhua 5,000; Quzhou 88 (looks like an undercount); Zhoushan 682; Taizhou 5,686; Lishui 3,888.
  - Another search summary gives 34,880 for the same survey.
  - Registration (登记编号) began 2014-15, target 5,000 in 2015. Sites under 50 m² get no number and
    are to be demolished, merged or converted.
- **Zhejiang 2020: ~17,000** (Pew 2023, linking zj.gov.cn 2021-01-29). Source page returned 405; the
  People's Daily 2017 page for the "35,000 in 2013" figure failed TLS. Not verified at source. The
  halving is a policy effect, not a change in practice.
- **Fujian: "约2万多座" (about 20,000+) folk venues.** Search-engine summary only; unverified.
- **Guangdong / Hunan:** registration thresholds only.
  - Guangdong: ≥500 m² land, ≥300 m² building, or ≥1,000 participants per event.
  - Hunan: main hall ≥50 m² or ≥1,000 participants.
  - No counts found.
- **CLDS 2014 neighbourhood-committee survey (via Pew 2023)** — share of committees with at least one:

  | venue type | rural | urban | all |
  |---|---|---|---|
  | 寺庙 (temple) | 30% | 15% | 25% |
  | 土地祠/神龛 (folk temple or shrine) | 27% | 7% | 19% |
  | 宗祠/祠堂 (ancestral hall) | 16% | 8% | 13% |
  | 道观 (Daoist temple) | 1% | <0.5% | 1% |
  | any of these | 45% | 17% | 35% |

  - Pew extrapolates ≥165,000 folk temples/shrines, ~192,000 temples with a Buddhist connection and
    >102,000 ancestral halls.
  - The repo's CLDS 2016 community file has the same kind of items with province (non-commercial
    licence).

---

## 3. What "no religion" contains

- **Pew 2023, among adults with no zongjiao affiliation** (source survey in brackets):
  - visited family graves in the past year 75% (CGSS);
  - care about auspicious days 61% (CGSS);
  - believe in fengshui 45% (CFPS);
  - believe in Buddha/bodhisattva 30% (CFPS);
  - burn incense a few times a year or more 21% (CFPS 2016);
  - went somewhere to pray for good luck 21% (CGSS);
  - believe in Taoist deities 16% (CFPS);
  - ghosts 8%, heaven 8%, afterlife 8%, hell 7% (WVS);
  - wear a charm 7%;
  - attend zongjiao activities a few times a year 2%.
- **WVS 2018:** only 34% of all adults call themselves atheist.
- **CSLS 2007:** 17.8% say they have a religious belief. 64.8% worshipped somewhere in the past year,
  most of them only at graves; 27.2% at a temple or at home.

---

## 4. Provincial breakdowns found

1. **Hu & Tian 2018 (CFPS 2010, CFPS-derived), province map in bands, raw data**, 25 provinces.
   - Grave visiting over 80%: Shandong, Guangxi, Yunnan, Jiangxi, Zhejiang, Fujian; several others
     over 70%.
   - Genealogy over 40%: Shandong, Anhui, Jiangxi, Hunan; over 30%: Guangdong, Guangxi.
   - Exact values not in text.
   - Ancestor practice only; does not separate deity worship.
2. **Lu Yunfeng 2014 (CFPS 2012, CFPS-derived)**, five oversampled provinces, self-id only; see §2.2.
3. **NRAA registered Buddhist and Daoist temple shares by province** (§2.8). Geography of formal
   temples, strongly southeastern; matches the CGSS Buddhism gradient already in the repo.
4. **Zhejiang folk temples by prefecture, 2013** (§2.8).
5. **CGSS pooled self-id folk by province** is already in the repo (`sources/cn_cgss.md`, Guangdong
   22.5%).
6. **Not found:** any province-level estimate of the 25-30% practice measure, CGSS 2018 B4 by
   province, or a national folk-venue count by province.

---

## 5. Provenance: CIA Factbook and Pew

- **CIA World Factbook, China, Religions** (read from the factbook.json mirror on GitHub; cia.gov now
  shows a sunset notice dated 2026-02-04): "folk religion 21.9%, Buddhist 18.2%, Christian 5.1%,
  Muslim 1.8%, Hindu < 0.1%, Jewish < 0.1%, other 0.7% (includes Daoist (Taoist)), unaffiliated 52.1%
  (2021 est.)".
- **Pew, The Global Religious Landscape (Dec 2012), table "Religious Composition by Country", China
  2010:** Christian 5.1, Muslim 1.8, unaffiliated 52.2, Hindu <0.1, Buddhist 18.2, folk 21.9,
  other 0.7, Jewish <0.1. **The CIA row is Pew's 2012 row**, with unaffiliated 52.1 instead of 52.2.
  "2021 est." is a date stamp, not a new estimate.
- **How Pew built the 2012 row** (GRL 2012 Appendix A, "China"):
  - Buddhists, folk, other and unaffiliated: Pew staff analysis of CSLS 2007.
  - Buddhist 18%: from "believe in Buddhism or Buddha" (CSLS BELIEVE1; my tabulation 18.1% w).
  - Folk 22%: "conservative criteria". Fn 40: classified as folk if not in another group AND
    worshipped gods or spirits at religious sites, home or workplace; OR attended formal temple
    services or prayed/burned incense in temples; OR believed in gods or spirits, evil forces or
    demons, heaven, hell, the afterlife or reincarnation.
  - Pew's alternatives: World Religion Database 30%; Yang & Hu 55%. Fn 39 cites the CASS *Blue Book
    of Religions* (2010), pp. 170-171 and 175, for treating folk belief as a religion, not for a number.
  - Muslims: 2000 census ethnicity. Christians: adjusted up for underreporting (Global Christianity
    2011, App. C).
- **Pew 2025, "How the Global Religious Landscape Changed From 2010 to 2020":**
  - Dropped the custom China method; uses the CGSS zongjiao item only (2010 and 2018 waves),
    reweighted so Muslim-ethnic shares match the census. No undercount adjustments. Folk religion now
    sits inside "other religions".
  - Revised China 2010: Christian 2.3, Muslim 1.5, unaffiliated 87.4, Buddhist 5.6, other 3.1.
  - China 2020: Christian 1.8, Muslim 1.8, unaffiliated 89.6, Buddhist 3.7 (53.4M), other incl. folk
    3.0 (43.1M).
  - Column order checked against Chile's row.
- **Chinese Wikipedia 宗教人口列表** ("list of religious populations"): folk 30%, Buddhism 18.2%,
  no religion 47%. The 30% matches Pew's WRD citation. WRD is gated; not traced further.

---

## 6. Dead ends (what was tried, what came back)

**The ECNU survey**
- Washington Post 2007-02-08: 403 to WebFetch.
- China Daily original `chinadaily.com.cn/china/2007-02/07/content_802994.htm`: 404.
- Chinese searches found no primary report:
  - 华东师范大学 童世骏 刘仲宇 宗教信仰 调查 31.4%;
  - 瞭望东方周刊 2007 … 4500 31.4%;
  - "中国人的精神生活" 华东师范大学 问卷调查.
- cn.bing.com redirected to bing.com and returned an empty page.
- sinoss.net PDF (Chen Qinjian & Yi Xiaolong, ECNU) is a literature review with no numbers.
- ccj.pku.edu.cn 经济学(季刊) 2014 PDF (matched "31.4%"/"16岁以上" in search) is a JBIG2 image scan;
  unreadable by WebFetch, not OCR'd.

**Lu Yunfeng 2014 full text**
- cssn.cn copy: 404. pacilution.com: DNS failure. iwr.cass.cn and iwr.cssn.cn: connection refused
  (several URLs). CNKI: abstract only. PKU IR record not fetched.

**Other publications**
- Yang & Hu 2012: Wiley paywall; Fudan page abstract only; ResearchGate not tried.
- Hackett & Tong 2025 (*Review of Religious Research*): Sage 403.
- Pew web pages: 403 to WebFetch. The same URLs downloaded fine with plain curl, so not a bot wall.
- CIA factbook: cia.gov page is a sunset notice; web.archive.org blocked for WebFetch; GitHub mirror used.
- Qiu Yonghui 2021 (iwr.cssn.cn): connection refused. The sohu mirror of the Blue Book ten-year review
  has no folk-religion numbers.
- Blue Book (宗教蓝皮书) folk-adherent estimates: none found online.
- ARDA SLSC web page: navigation only; local file used instead.
- Yu Tao 2008 original: no paper found in several searches; the academia.edu presentation Wikipedia
  cites was not tried.
- Yao 2007 JCR: paywalled.
- "Revival of Folk Religion" PDF (Asian Ethnology): it is Pui-lam Law 2005, a Dongguan ethnography
  with no survey figures.
- Fang Wen PKU PDF: CJK fonts don't render.
- Li Xiangping 2011 Yangtze Delta results: not online. Zhejiang Social Sciences 2011 PDF at
  sociology.cssn.cn: connection refused.
- Kuang & Liang 2021 Baidu-index paper: national only, no provinces.

**Venue counts**
- Zhejiang 2020 zj.gov.cn page: HTTP 405. People's Daily 2017 page: TLS certificate mismatch.
- Fujian ethnic and religious affairs department folk-religion column: article list only, no totals.
- National folk-venue count: not found. The NRAA database is Buddhist/Daoist only.
- China Religion Survey 2015 (Renmin University, venue survey): nsrc.ruc.edu.cn https→http redirect
  refused; cnsda article URL 404.

**Other**
- CGSS 2018 provincial practice papers: searched in Chinese (CGSS2018 烧香拜佛 求好运 祭祖 省份) and
  English; none.

---

## 7. Files in the scratchpad

- `slsc_items.py` / `slsc_items.txt` — all SLSC variables and single-item frequencies.
- `slsc_multi.py` / `slsc_multi.txt` — multi-mention batteries (worship settings, home items,
  activities, amulets, divination).
- `slsc_exist.py` / `slsc_exist.txt` — belief battery with slot-to-item check, plus composites.
- `pew_china_2023.pdf` — the report; Appendix B at pp. 136-158 has the Chinese question wordings.
- `pew_grl2012_appA.pdf`, `pew_grl2012_full.pdf`, `pew_grl_2025.pdf` + `.txt`.
- `zhang_lu_sheng_2021.pdf`, `chinazentrum_2022.txt`, `hu_tian_2018.txt`, `manchester_believers.txt`.
