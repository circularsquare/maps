# Saudi Arabia (sa): record

Drawn 2026-10-05 (session edd42a8c-sa). No census or survey asks Saudi residents a language.
Built under Anita's 2026-10-05 ruling for countries with no language question (AGENT_BRIEF §2):
the national language for citizens, immigrant languages proxied by citizenship. 2022 census,
32,175,224 people, 13 regions, 230 nodes, **every row `derived`**. Placed on religiondots' Kontur
400 m hexes by population. 32,071 dots at 1:1,000; 48 rings.

Files: `sources/sa_extract.py` (copies religiondots' parsed census tables), `sources/sa_census.py`
(the build), `taxonomy/sa2022.py`, `taxonomy/tree.d/sa.txt`, `countries/sa.py`,
`data/raw/sa/sa_regions.csv`, `data/raw/sa/sa_nationality.csv`, `data/normalized/sa.csv`.

```
python sources/sa_census.py --fetch     # --fetch re-runs sa_extract.py
python taxonomy/build.py
python tools/check_country.py sa
python scatter.py --country sa
```

## 1. The census tables

religiondots built Saudi Arabia on the same census (`../religiondots/sources/sa.py`, record
`../religiondots/sources/sa.md`), with the fetch, parsers and checks already done, so
`sa_extract.py` loads that module by path in its own process (its taxonomy folder's module names
collide with ours) and writes two CSVs. Read-only on religiondots. What it carries over:

- Saudis 18,792,262 and non-Saudis 13,382,962 per region (GLMM's mirror of the census portal),
  equal to the GASTAT report's Figure 11 shares to 0.05 points.
- Non-Saudis by nationality and sex, national only: GLMM's four tables (Arab; non-Arab Asian;
  European; Sub-Saharan), which reproduce the report's continent prose (Asia 76.16 vs 76.2%,
  Africa 23.23 vs 23.2%); the 42,140 in none of them are the Americas, mostly US.
- Non-Saudi men and women per region, from the report's Figure 12 sex ratios raked to the
  national sexes (Riyadh within 0.013% of the Royal Commission for Riyadh City's own count).

No nationality is published by region. The sexes carry different nationalities (1,181
Bangladeshi men per 100 women; 61 Filipino men per 100), so each region's non-Saudi men take the
national language mix of non-Saudi men and its women that of non-Saudi women, as religiondots does.
That still assumes the same mix everywhere within a sex.

## 2. Citizens: Saudi Arabic

All 18,792,262 on `afroasiatic.saudi_arabic`, a new node beside `arabic` as Iraq's, Sudan's and
Morocco's are. It stands for Najdi (najd1235), Hijazi (hija1235), Gulf (gulf1241) and the southern
Yemeni-type dialects of Asir and Jazan; no source counts them apart, so they are not split by
region (that would be guessing the map). Not drawn: citizens with another first language, such as
Mehri and other Modern South Arabian speakers on the Yemeni border, Faifi, and naturalised
families; no source counts them (`gap`).

## 3. Non-Saudis: nationality to language

Every nationality gets a language or a mix (`sources/sa_census.py`):

| nationality | non-Saudis | drawn as | source |
|---|---:|---|---|
| Bangladesh | 2,116,192 | Bengali | `fr_build.COUNTRY_LANG` |
| India | 1,884,476 | state mix, below | KMS 2023 + MEA clearances + census 2011 |
| Pakistan | 1,814,678 | province mix, below | BEOE + census 2023 |
| Yemen | 1,803,469 | Yemeni Arabic (new) | Sanaani, Ta'izzi-Adeni, Hadrami |
| Egypt | 1,471,382 | Egyptian Arabic (new) | egyp1253 |
| Sudan | 819,575 | Sudan's drawn mix | `countries/sd.py` |
| Philippines | 725,893 | the Philippines' drawn mix | `countries/ph.py` |
| Syria, Jordan, Palestine, Lebanon | 836,211 | Levantine Arabic (new) | nort3139 |
| Nepal, Indonesia, Ethiopia, Afghanistan, Uganda, Kenya, Sri Lanka, Nigeria, Mali, Niger | | each country's drawn mix | `countries/<cc>.py` |
| Myanmar | 163,717 | Rohingya | religiondots `sources/sa.py`: GASTAT's Myanmar nationals are the Rohingya (Refugee Law Initiative 2023; the 2017 special residency permits) |
| Iraq | 6,400 | Iraqi Arabic | |
| France | 5,163 | French | (France is not in its own COUNTRY_LANG table) |
| the Gulf states, Libya, Tunisia, Chad | | Arabic | COUNTRY_LANG |
| everyone else | | COUNTRY_LANG's main language | |
| GASTAT `Other` rows | | `africa_other` (Sub-Saharan table) or `other` | narrowest node |
| not in the four tables (Americas) | 42,140 | English | mostly US nationals |

**Home mixes** ("drawn mix") are each country's own `counts()` on this map, summed nationally,
for multilingual origins with 20,000+ people in Saudi Arabia. **Every mix (these and India's and
Pakistan's states) keeps languages of 1%+ of its people and scales them back to 100%**: drawn at
full length, the state and country tails put 830 languages on the map at a handful of people each,
a precision the proxy does not have. With the cut, 230 nodes.

**India (1,884,476).** Nothing publishes Indians in Saudi Arabia by state. Built in two parts:

- Keralites: Kerala Migration Survey 2023 (IIMAD draft report, 2024): 2,154,275 emigrants
  (Table 3.1), 16.9% in Saudi Arabia (Table 3.7) = 364,072, **19.3%** of the census's Indians
  (the Government of India's own estimate of Indians in Saudi Arabia is 2.47M, so this share may
  be a little high). Drawn at Kerala's 2011 census mix (Malayalam).
- Everyone else (80.7%): the ILO's *India Labour Migration Update 2018*, Figure 4, MEA emigration
  clearances 2011-17 by state, the ten states named: UP 31, Bihar 15, Tamil Nadu 11, West Bengal 8,
  Rajasthan 7, Punjab 7, Andhra Pradesh 7, Telangana 2, Odisha 2 (Kerala's 10 dropped, the survey
  stands for it). Telangana joins Andhra Pradesh (one state in 2011). Each state at its 2011 census
  mother-tongue mix (`data/normalized/in.csv`, state rows, through `in2011.resolve`).
- Result: Hindi 28.8%, Malayalam 19.0%, Tamil 9.2%, Telugu 7.7%, Bhojpuri 6.5%, Bengali 6.4%,
  Punjabi 5.9%, Urdu 3.8%.
- Weaknesses: clearances are flows to all 18 ECR countries, only of workers without ten years of
  school; the states' mixes are for all their people, but Indians in Saudi Arabia are mostly
  Muslim (religiondots: 19.7% Hindu), so **Urdu is very probably undercounted** against Hindi in
  UP, Bihar and Telangana. Bihar's "Others under Hindi" sit on Indo-Aryan, as in India's own map.

**Pakistan (1,814,678).** Ministry of Overseas Pakistanis and HRD, *Year Book 2020-21*, Table 14:
BEOE registrations by province, FY 2019-20 and 2020-21 summed (699,871; all destinations, Saudi
Arabia about half of all registrations since 1971). Each province at its 2023 census mix
(`data/normalized/pk.csv`); Azad Kashmir (not in the census) on Pahari-Pothwari, Gilgit-Baltistan
on Shina, the former tribal areas on Pashto. Result: Punjabi 37.0%, Pashto 31.3%, Saraiki 12.3%,
Urdu 5.7%, Sindhi 4.6%, Pahari-Pothwari 3.5%, Hindko 3.1%. BEOE's own statistics page is behind a
Cloudflare wall; the cumulative 1971-2025 split quoted in the press (Punjab ~50%, KP 25.7%, Sindh
9.3%) is close to the two years used.

## 4. Results

Saudi Arabic 58.4%, Bengali 7.0%, Yemeni Arabic 5.6%, Egyptian Arabic 4.6%, Levantine Arabic
2.6%, Sudanese Arabic 2.5%, Punjabi 2.4%, Pashto 1.9%, Hindi 1.7%, Malayalam 1.1%. Saudi Arabic
by region: 52% in Riyadh and Makkah, 74% in Al Bahah and Al Jawf. Arabic varieties together about
74% of residents.

## 5. Calls someone might reverse

- One Saudi Arabic node, and separate Yemeni, Egyptian and Levantine nodes (the iq/sd/ma
  convention). Folding all of them into `arabic` is a mapping edit in `sa_census.py`.
- India via KMS + clearances; the alternative was India on Hindi (would draw 1.9M on Hindi).
- The 1% cut in home mixes.
- No placement inside regions by nationality: dots follow Kontur population. religiondots keeps
  non-Muslims out of Mecca's haram; nothing here does, so a few Filipino (largely Christian) dots
  can fall in Mecca. Small, left.

## 6. Room for improvement

- A survey asking Saudi citizens' home language (WVS wave 4, 2003, has a Saudi sample; not
  checked) could test the all-Arabic citizens and split dialect regions.
- Nationality by region (GASTAT has it in the census portal's detailed tables if they ever come
  back; `portal.saudicensus.sa` no longer resolves) would replace the same-mix-everywhere assumption.
- India's Urdu share: a Gulf-migrant survey by religion and mother tongue.

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. Riyadh region is split by RCRC's census table into "Riyadh and Ad Diriyah" (52.1% non-Saudi) and the rest of the region (30.5%); each keeps the region's non-Saudi language mix. Placement layer is now `data/geo/sa/sa_hexes.gpkg` (religiondots' hexes re-keyed). Non-Saudi share, before -> after: Riyadh city 48.3 -> 52.2%, rest of Riyadh region 48.3 -> 30.5%, Jeddah 48.2 -> 57.2%, Dammam-Khobar 42.4 -> 50.7%, outside the five big cities 37.6 -> 32.5%. This replaces §5's "no placement inside regions by nationality"; nationality itself is still one national mix per sex.
