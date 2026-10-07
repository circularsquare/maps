# Japan (jp): record

Drawn 2026-10-05 (session edd42a8c-jp). No census or survey asks Japan's residents a language;
the 2020 census asks nationality. Built under Anita's 2026-10-05 ruling for countries with no
language question (AGENT_BRIEF §2): Japanese, plus regional languages from regional surveys and
immigrant languages proxied by nationality, Saudi Arabia's home-mix method (`sources/sa.md`).
126,146,099 people, 1,896 municipalities and wards (1,895 with people; Futaba, Fukushima, had
nobody in 2020), 214 nodes, **every row `derived`**. 126,072 dots at 1:1,000, 119 rings.

Files: `sources/jp_census.py` (the build; docstring has every step), `sources/jp_geo.py`
(placement layer), `taxonomy/jp2020.py` (identity mapping), `taxonomy/tree.d/jp.txt` (own nodes
by hand in the script's `OWN_NODES`, borrowed block written by the script), `countries/jp.py`,
`data/raw/jp/` (every source below), `data/normalized/jp.csv`, `data/geo/jp/`.

```
python sources/jp_census.py
python taxonomy/build.py
python sources/jp_geo.py
python tools/check_country.py jp
python scatter.py --country jp
```

## 1. Census: table 44-1

2020 Population Census, 人口等基本集計 第44-1表 "男女，国籍別人口－全国，都道府県，市区町村",
e-Stat Excel `statInfId=000032142708` (no key; the e-Stat API needs an appId, the file download
does not). Columns: total, foreign, 12 named groups (Korea incl. North Korea, China,
Philippines, Thailand, Indonesia, Vietnam, India, Nepal, UK, US, Brazil, Peru) and Other
(incl. stateless and country not stated), Japanese, Japanese/foreign not stated. Units: wards of
the 20 designated cities and Tokyo's 23 special wards, plus every other city, town and village:
1,896. Checks (all stop the build): units sum to Japan in every column (126,146,099), units to
each prefecture, wards to their designated city, nationality columns to unit totals.

**Nationality not stated** (2,202,484, 1.7%; non-response, concentrated in big cities) is spread
over each unit's known nationalities pro rata, so units keep their census totals.

## 2. Splitting "China" and "Other"

Immigration Services Agency, 在留外国人統計 December 2020, 第4表 (prefecture x nationality,
e-Stat `statInfId=000032104295`; the download is an xlsx despite the csv kind). ISA names to
ISO2 via babel's Japanese territory names plus 19 by hand (`ISA_NAMES`); continent subtotal
columns dropped; prefectures + 未定・不詳 sum to the national total. Each municipality's "China"
and "Other" take its prefecture's ISA mix.

**Is Taiwan in the census's "China"?** The census classifies 195 countries Japan recognises and
the 2015 detailed nationality table has no Taiwan row, so taken as yes (China split into China
and Taiwan by ISA). The census counts 71-87% of ISA's figure for every named group; with Taiwan
in China, China is 0.81 and Other 1.20; with Taiwan in Other, 0.87 and 1.00. Other's high ratio
either way is the census's own "country not stated" filed there. Reversing costs ~45,000 people
moving between Taiwan's and the other origins' mixes.

## 3. Nationality to language

- Korea and North Korea: Korean. Brazil: Portuguese. Peru, Bolivia: Spanish (Nikkei families
  from Lima and Santa Cruz; the home mix would add Quechua). France: French.
- Home mix (this map's own `counts()` for that country, languages of 1%+ kept and scaled to
  100%, as sa): China, Taiwan, Vietnam, the Philippines, Nepal, Indonesia, Thailand, India,
  Myanmar, Sri Lanka, Pakistan, Bangladesh, Cambodia, the US, Malaysia, Canada, Australia,
  Russia, Turkey: the drawn multilingual origins with 5,000+ residents. China's mix is the
  whole country's, though Chinese in Japan come disproportionately from the north-east and
  Fujian; nothing counts them by province.
- Everyone else: `fr_build.COUNTRY_LANG`'s main language; stateless on `other`.

**Retention** (share drawn as Japanese): Aichi Prefecture, 外国人県民アンケート調査 (March 2022),
Q25 "what language do you speak with your children" (parents of children under 18, N = 746),
table 25-2 by nationality, in `data/raw/jp/aichi_gaikokujin_2022_part2.pdf`. The prefecture's
site is behind an Incapsula wall; all parts came from the Wayback Machine (2022-06-09).
Measure: "always Japanese" / answered, the analogue of France's TeO "French only with the
children" (`sources/fr.md` §2b); mixed answers stay on the language.

| nationality | always Japanese | answered | share |
|---|---:|---:|---:|
| Korea | 50 | 62 | 80.6% |
| Philippines | 53 | 117 | 45.3% |
| China | 54 | 183 | 29.5% |
| Brazil | 25 | 203 | 12.3% |
| Vietnam | 5 | 55 | 9.1% |
| all respondents (everyone else) | 213 | 723 | 29.5% |

Rows under 40 respondents (Nepal 10, Peru 20, Indonesia 10, Thailand 7, US 10) take the total.
Special permanent residents (the Zainichi families, almost all Korean) answered 90.7% always
Japanese (n = 43), consistent with Korea's row. The survey's nationality rows are not labelled
in the PDF text; they were matched by their N to the prose (Korea 79.4%, Philippines 44.2%,
Brazil 50.7% always another language) and order. Weakness: one prefecture, and only parents.
Moved onto Japanese: 860,388 of 2,454,234 foreign residents (after the not-stated spread).

## 4. Ryukyuan languages

**Okinawa.** Okinawa Prefecture, しまくとぅば県民意識調査 報告書, 令和5年度 (fieldwork Jan-Feb 2024,
n = 1,028) and 令和6年度 (Jan-Feb 2025, n = 1,043), 18+, stratified random sample, question on
how much one uses shimakutuba, by region (北部, 中部, 南部, 宮古, 八重山, その他の離島). Measure:
"mainly use" + half of "as much as standard Japanese" (France's "both counts half"), the two
years pooled by respondents. Overall 2024: mainly 3.4%, as much 13.0%; 70+ mainly 11.1%; under
40 nobody mainly.

| region | rate | language(s) drawn |
|---|---:|---|
| north (Okinawa Island) | 10.9% | Kunigami: Nago, Kunigami, Ogimi, Higashi, Nakijin, Motobu; Central Okinawan: Onna, Ginoza, Kin |
| central | 9.1% | Central Okinawan |
| south | 8.2% | Central Okinawan |
| Miyako | 23.9% | Miyako (Miyakojima, Tarama) |
| Yaeyama | 6.9% | Yaeyama (Ishigaki, Taketomi), Yonaguni (Yonaguni) |
| other islands | 14.3% | Kunigami: Ie, Iheya, Izena; Central Okinawan: Kume, Tokashiki, Zamami, Aguni, Tonaki |

Applied to each municipality's Japanese nationals 18+ (census 2020 table 3-3 for Okinawa,
e-Stat `statInfId=000032148629`, 5-year bands; 18-19 = 2/5 of 15-19; age not stated pro rata;
then the nationality not-stated factor). Under-18s drawn as Japanese. The Daito islands (settled
from Hachijo and Okinawa in 1900) are left Japanese. "Other islands" is not defined in the
report; taken as the remote islands outside Miyako and Yaeyama. Regional samples are small
(Miyako 30 and 35: "mainly" was 2.9% one year and 20.0% the next), hence the pooling.
Drawn: Central Okinawan 86,758, Miyako 10,396, Kunigami 8,970, Yaeyama 2,830, Yonaguni 94;
7.4% of Okinawa's population. Ethnologue's Yonaguni figure is 400; the survey's Yaeyama rate
gives 94.

**Amami Islands (Kagoshima).** No survey found (Japanese web searches for an Amami or shimaguchi
language-use survey turned up none). Ethnologue's speaker figures as quoted by Wikipedia (18th ed., dated 2004):
Northern Amami-Oshima 10,000 (Amami city, Tatsugo, Yamato, Uken), Southern 1,800 (Setouchi),
Tokunoshima 5,100, Okinoerabu 3,200, Yoron 950, spread over their municipalities by Japanese
population. Kikai's 13,000 ("cited 2000") exceeds the island's population, so Kikai takes the
others' speakers-per-resident ratio (21.6%): 1,427. These are speaker counts from 2004, a looser
and older measure than Okinawa's; today's first-language speakers are probably fewer.
Matsumoto & Tabata (2012, Gengo Kenkyu 142) and Shimoji & Pellard (eds., *An Introduction to
Ryukyuan Languages*, 2010) were read: both say fluent speakers are mostly over 50-60, neither
counts them.

**Ainu: not drawn.** Hokkaido Ainu living conditions survey 2023 (北海道アイヌ生活実態調査, table
78): 0.8% of 472 respondents can hold a conversation in Ainu, all 60+; that is a few dozen
people at the survey's population, under one dot, and not first-language.

## 5. Geography (`sources/jp_geo.py`)

Polygons: MLIT N03 (2021-01-01) as simplified 1% by SmartNews SMRI's japan-topography (GitHub),
joined on the JIS code: every census unit found; the only N03 codes with no unit are the six
Northern Territories villages, plus four unassigned-land polygons with no code. religiondots'
Japan layer is prefecture-level, so not reused; its Kontur extract was copied. Hexes by
centroid; the simplification drops coastline, so 7,581 hexes (1.2M people) fell outside every
polygon and were snapped to the nearest municipality within 15 km; 8 hexes (90 people) dropped.
Kontur / census normalised: p10 0.92, median 1.07, p90 1.25; 5 units outside a factor of 3, all
in Fukushima's evacuation zone (Okuma 12.2, Namie 9.9: Kontur predates the census's empty
towns) or Samegawa; lowest are central wards (Hiroshima Naka 0.39), where coastal snapping may
have moved hexes to a neighbouring ward. Log correlation 0.993 against a best of 0.083 over 500
shuffles. Placement inside a unit is by population for every language.

## 6. Results

Japanese 98.63%, Mandarin 0.24%, Vietnamese 0.22%, Portuguese 0.13%, Central Okinawan 0.07%,
Korean 0.06%, English 0.05%, Spanish 0.04%, Tagalog 0.04%, Wu 0.03%.

## 7. Calls someone might reverse

- Retention from one prefecture's survey of parents (Aichi 2022); France's TeO shares were the
  alternative the brief allowed, rejected because Japan's own survey exists.
- Okinawa measure (mainly + half of equally) and its region-to-language assignment; the
  Onna/Ginoza/Kin line between Kunigami and Central Okinawan.
- Amami at Ethnologue's 2004 counts; Kikai imputed.
- Taiwan inside the census's China.
- The 1% cut in home mixes (sa's convention).

## 8. Room for improvement

- A survey of Amami islanders' language use like Okinawa's.
- Full-resolution N03 instead of the 1% simplification would end the coastal snapping.
- ISA status of residence by prefecture would separate Zainichi Koreans (90.7% Japanese) from
  newcomers per place, instead of one Korea share everywhere.
- China by province of origin for Chinese residents.
