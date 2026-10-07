# South Korea (kr): record

Drawn 2026-10-05 (session edd42a8c-kr). No Korean census or survey asks a language (the 2020
census items are nationality and entry date; queue scout note). Built under Anita's 2026-10-05
ruling for countries with no language question (AGENT_BRIEF §2): the national language, a
regional language from a cited estimate, immigrant languages proxied by nationality. 2020,
51,829,136 people, 229 si/gun/gu, 118 nodes, **every row `derived`**. Placed on religiondots'
Kontur 400 m hexes by population. 51,782 dots at 1:1,000; 50 rings.

Files: `sources/kr_census.py`, `taxonomy/kr2020.py`, `taxonomy/tree.d/kr.txt`, `countries/kr.py`,
`data/raw/kr/mois_foreign_residents_2020.xlsx`, `data/normalized/kr.csv`.

```
python sources/kr_census.py --fetch
python taxonomy/build.py
python tools/check_country.py kr
python scatter.py --country kr
```

## 1. The table

Ministry of the Interior and Safety (MOIS), *2020 지방자치단체 외국인주민 현황* (foreign residents
by local government, as of 2020-11-01), statistical workbook on mois.go.kr (board
BBSMSTR_000000000014, article 88648; `FileDown.do?atchFileId=FILE_00105361PZHQd7S&fileSn=5`).
Open download, no login, a plain browser User-Agent. KOSIS itself bot-blocks downloads
(religiondots `countries/kr.py`), so this workbook is the route; it carries the census figures.

- Sheet 1-2: 총인구 per si-gun-gu = the 2020 census population, 51,829,136, foreigners
  included; 한국국적을 가지지 않은 자 (foreign nationals) 1,695,643.
- Sheet 4-2: foreign nationals by nationality per si-gun-gu, 36 columns, with **중국(한국계)**
  (Korean-Chinese, 541,337) and **러시아(한국계)** (Russian Koreans, 19,961) apart.
- Sheet 5-2-2: the same for marriage migrants (173,756); 5-4-2 for 외국국적동포 (ethnic Koreans of
  foreign nationality, 345,110).
- Cells under 5 print `*`: each unit's starred cells share the residual of their printed
  subtotal (or the unit total) equally.

Checks (all in the script, all pass): national rows equal 51,829,136 and 1,695,643; si-gun-gu
populations sum to the national total; per unit, sheet 1-2's foreign nationals equal sheet 4-2's
합계; nationalities sum to each unit's total within 1% after star filling.

**Join**: by Hangul name within province to religiondots' `kr_lookup.csv` (229 units, 2015
boundaries), asserted 1:1 both ways. General gu rows (수원시장안구...) dropped, their city row
kept, as religiondots draws the cities. One rename: Incheon's 미추홀구 (Michuhol-gu) is the 2015
남구 (Nam-gu, renamed 2018). Gunwi stays in North Gyeongsang (moved to Daegu only in 2023).

## 2. Korean nationals: Korean

Census population minus foreign nationals: 50,133,493, naturalised citizens (199,128) and foreign
residents' children (261,646) included, all on Korean. Their own home languages are not counted
anywhere (`gap`).

## 3. Jejueo

`koreanic.jejueo`, Glottolog jeju1234 (Koreanic, its own language); UNESCO critically endangered
(2010). Speaker figure: the Jejueo Project, University of Hawai'i at Manoa (William O'Grady,
Changyong Yang; sites.google.com/a/hawaii.edu/jejueo): "Jejueo has at most 5000 to 10,000
speakers ... largely by elderly speakers". Drawn: **7,500**, the midpoint, taken from Korean
nationals in Jeju-si and Seogwipo-si in proportion to their Korean nationals (out of 670,858
residents, about 1.1%). No source places speakers more finely; the elderly rural villages are
probably over-represented relative to Jeju city, which this does not capture.

## 4. Foreign nationals: nationality to language

| nationality (4-2) | 2020 | drawn as |
|---|---:|---|
| 중국(한국계) Korean-Chinese | 541,337 | Korean (Joseonjok mostly speak Korean at home) |
| 중국 China | 207,764 | China's drawn mix (Mandarin, Wu, Cantonese, Min Nan, Jin, Gan...) |
| Vietnam, Thailand, Philippines, Indonesia, Cambodia, Myanmar, Malaysia, Laos, Timor-Leste, Taiwan, Sri Lanka, Pakistan, Bangladesh, Nepal | | each country's drawn mix |
| Uzbekistan 58,000, Kazakhstan 27,035, Kyrgyzstan 5,199 | | 외국국적동포 of that nationality (5-4-2, the Koryo-saram) on Russian; the rest at the country's drawn mix |
| Russia; 러시아(한국계) | 21,724; 19,961 | Russian |
| Japan, Mongolia | | Japanese, Mongolian |
| US, Canada, UK, Oceania | | English |
| Latin America | 3,478 | Spanish |
| Africa | 16,658 | `africa_other` |
| the 기타 remainders (other SE Asia, other South-West Asia 9,506, other Central Asia, Asia other, other Europe 16,101, other) | | `other` |

Home mixes: each origin's own `counts()` on this map, summed, languages of 1%+ kept and scaled to
100% (the Saudi method, `sources/sa.md` §3). The Japan agent's record had not landed when this
was built; nothing here depends on it (Japan is a single-language origin).

**Retention (call).** No Korean survey publishing the share of immigrants who speak only Korean
at home was found (the 2021 National Multicultural Family Survey reports falling bilingual use,
not a rate; searched 2026-10-05). France's TeO2 share speaking only the host language with their
children (`fr_build.TEO_FRENCH`, by TeO2 region; `sources/fr.md` §2b) is applied **to marriage
migrants only** (sheet 5-2-2 by nationality and unit), moving 52,236 onto Korean (Vietnamese
17,233, Mandarin 7,766, Japanese 3,870). Workers (E-9 and the like), students and other
foreign nationals keep their origin language whole: most are temporary residents without Korean
households, unlike TeO's settled immigrant families, and applying TeO to all of them moved
410,283, including 70,000 Thai workers. Marriage migrants have Korean spouses, so TeO2's shares
(drawn from immigrant families in France) probably understate their shift to Korean.

## 5. Results

Korean 97.86%, Vietnamese 0.33%, Thai 0.31%, Mandarin 0.26%, English 0.16%, Russian 0.12%, Uzbek
0.09%, Khmer 0.08%, Mongolian 0.07%, other 0.06%, Jejueo 7,500. Highest non-Korean shares: Pocheon 9.8%, Eumseong
9.5%, Yeongam 8.0% (the Daebul shipyards), Jincheon 7.2%, Yongsan 6.7%, Seoul Jung-gu 6.5%. The
Korean-Chinese districts (Ansan Danwon-gu, Yeongdeungpo, Guro) do not stand out, because
Joseonjok are drawn as Korean.

## 6. Calls someone might reverse

- Korean-Chinese on Korean (the brief's instruction); Koryo-saram and Russian Koreans on Russian.
- Retention for marriage migrants only, at TeO2's French shares.
- Jejueo at 7,500 (the midpoint of "at most 5,000 to 10,000"), spread by population across Jeju.
- China at its full drawn mix; Chinese in Korea are disproportionately from Shandong and the
  north-east, so Mandarin is probably undercounted against Wu and Cantonese.
- Remainders on `other` (other South-West Asia is largely Indians and Iranians; not guessed).

## 7. Room for improvement

- Placement inside si-gun-gu by foreign residents: sheet 1-3 gives foreign residents per
  eup-myeon-dong (3,500 units); joining it to dong boundaries would put immigrant dots in the
  right neighbourhoods (Daerim-dong, Wongok-dong) instead of following all population (§4.4 of
  the brief; not done).
- A Korean figure for marriage migrants' home language (the Multicultural Family Survey
  microdata on data.go.kr, 15114756) would replace TeO2.
- Jejueo by age and village, if a survey ever counts it.
