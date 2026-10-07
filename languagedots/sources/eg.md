# Egypt (eg): record

Drawn 2026-10-05 (session edd42a8c-eg). No census or survey counts Egypt's languages. Built on
the supervisor's estimate route: cited speaker figures placed on their home governorates, the
rest of each governorate on Egyptian or Sa'idi Arabic, refugees by UNHCR governorate counts and
nationality (Saudi Arabia's home-mix method, `sources/sa.md`). 108,528,518 people (CAPMAS's
own estimate for 2026-01-01, religiondots' `eg_lookup.csv`), 27 governorates, 29 nodes.
Refugee rows `derived`, every other row `modelled`. 108,517 dots at 1:1000, 4 rings.

```
python sources/eg_build.py
python taxonomy/build.py
python tools/check_country.py eg
python scatter.py --country eg
```

Files: `sources/eg_build.py`, `taxonomy/eg2026.py` (identity), `taxonomy/tree.d/eg.txt`,
`countries/eg.py`, `data/normalized/eg.csv`, `data/raw/eg/` (UNHCR fact sheet PDF and a 300 dpi
render of its governorate map).

## 1. What exists

- **Census.** None asks language (2017: religion and nationality only; scout 2026-10-05). The
  1960 Nubian count was not traced to a table.
- **Surveys** (religiondots' downloads, read-only): Afrobarometer R5 (2013, 1,190: all
  "Arabic") and R6 (2015, 1,198: all "Egyptian Arabic"; the card has no Sa'idi box); Arab
  Barometer II (2011, 1,219: all Arabic), III (2013, 1,196: 1,195 Arabic, 1 Nubian), IV (2016,
  1,200: all Arabic); VII (2021-22) asks ethnicity, 2,043 of 2,044 "Arab". R7-R9 and ArB V, VI,
  VIII have no Egyptian language question. Every Afrobarometer interview was in Arabic by an
  Arabic-speaking interviewer. WVS not run: 6,003 answers at 99.98% Arabic already settle the
  big share, and no WVS wave samples Siwa, Sinai or Matrouh either.
- **The one useful survey signal**: Arab Barometer III's 20 Aswan respondents (one sampling
  point, it looks like) are 1 Nubian first language and 9 Arabic-first, Nubian-second. So half of
  that cluster speaks Nubian, and nine of ten name Arabic first in an Arabic interview. The draw
  puts 17% of Aswan on Nobiin/Kenzi as first language; the survey neither confirms nor refutes
  that level, but it shows the minority is there and that the surveys' card pushes it to Arabic
  (Sudan's interview-language trap, `sources/sd.md` section 2, again).
- Matrouh, New Valley and both Sinais have 0-8 respondents per round: the surveys cannot speak
  to Bedouin, Siwi or the oases at all.

## 2. How the counts are made (`sources/eg_build.py`)

Per governorate, carve-outs from CAPMAS's population, then the rest:

| language | node | figure | where | source |
|---|---|---:|---|---|
| refugees | home mixes | 1,101,744 | all 27, UNHCR's own counts | UNHCR Egypt fact sheet June 2026 (as of 31 May 2026), map "Main residential areas"; the governorates sum to 1,101,744 against the headline 1,101,700 |
| Nobiin, Kenzi | `nilosaharan.nobiin`, `.kenzi` | 502,000 + 35,000 | 300,000 Aswan; 237,000 Cairo, Giza, Alexandria by population | Ethnologue (27th ed., via Wikipedia's Nobiin and Kenzi pages, read 2026-10-05); "about 300,000 speakers ... around Kom Ombo and Aswan" (Wikipedia, Languages of Egypt) |
| Beja | `afroasiatic.cushitic.beja` | 88,000 | Red Sea 14,346 (half the people south of 24N), Aswan 73,654 | Wikipedia, Beja language: "as of 2023 ... 88,000 Beja speakers in Egypt" |
| Siwi | `afroasiatic.berber.siwi` | 21,000 | Matrouh, Siwa/Qara box | Ethnologue 27th ed. via Wikipedia |
| Domari | `indoeuropean.indoaryan.domari` | 10,000 | a quarter each: Dakahlia, Cairo, Alexandria, Sa'idi governorates | Ethnologue 2016 via Joshua Project (people group 11597, location Dakahlia); places from Wikipedia, Doms in Egypt |
| Libyan Arabic (Awlad Ali) | `afroasiatic.libyan_arabic` | 85% of Matrouh = 504,804 | Matrouh outside Siwa | Huesken, "The practice and culture of smuggling in the borderland of Egypt and Libya", International Affairs 93(4), 2017: Awlad Ali "represent the majority (85 per cent) of its population" |
| Bedawi Arabic | `afroasiatic.bedawi_arabic` | 70% of North Sinai = 332,732; South Sinai 69,531 | Sinai | Aziz, Brookings 2017 ("Seventy percent of Sinai residents are Bedouin"); South Sinai: 38,000 in 13 tribes (Senri Ethnological Studies 55, 2001, Tribal Affairs Department figures, late 1990s; 70% of the 1996 census's 54,495), grown at Egypt's rate since (59,312,914 in 1996 to 108.5M), 58% of the governorate |
| Sa'idi Arabic | `afroasiatic.saidi_arabic` | the rest: 24,752,752 | Minya, Asyut, Sohag, Qena, Luxor, Aswan, New Valley | Ethnologue's region "Al Minya Governorate and south to Sudan border"; Glottolog puts the oases' Western Desert Egyptian Arabic (west2939) under Sa'idi |
| Egyptian Arabic | `afroasiatic.egyptian_arabic` | the rest: 81,110,955 | the other 20 | |

**Refugee mix** (fact sheet's top four, rounded): Sudan 852,000 at Sudan's drawn mix (which is
all Sudanese Arabic after the 1% cut), Syria 98,100 Levantine Arabic, South Sudan 56,000 at its
drawn mix, Eritrea 45,000 Tigrinya, the other 50,600 (57+ nationalities, mostly Ethiopians,
Somalis, Yemenis and Iraqis by UNHCR's other releases) on `other`. One national mix in every
governorate: UNHCR does not publish nationality by governorate. Refugees are carved out of
CAPMAS's population, not added to it.

**Checks.** Sa'idi drawn 24.75M against Ethnologue's 27M (2024): 8% short, the expected
direction, since Ethnologue also counts Upper Egyptians in Cairo and on the Red Sea, which this
map leaves on Egyptian Arabic. Wikipedia's Ethnologue-based infobox shares (Sa'idi 24%, Western
Bedawi 0.8%, Eastern Bedawi 1.1%, Nobiin 0.4%, Beja 0.07%, Kenzi 0.03%, Siwi 0.02%) agree in
order with what is drawn (22.8%, 0.47%, 0.37%, 0.46%, 0.08%, 0.03%, 0.02%); Bedouin are drawn
lower because only their home governorates are placed. Every governorate sums to CAPMAS's
figure (asserted).

**Placement inside a governorate** (`countries/eg.py`, `ZONES`): Siwi on the Siwa/Qara oases
box (25-27E, 28.8-29.8N), Libyan Arabic on Matrouh outside it; Nobiin, Kenzi and Aswan's Beja
south of 24.6N (Kom Ombo, Nasr al-Nuba, Daraw, Aswan city; not Edfu); the Red Sea's Beja south of
24N. Everything else by Kontur population. Counts never move.

## 3. Calls someone might reverse

- The estimate route itself (supervisor's brief; Sudan's ask 019 asks Anita the same question
  for Sudan). Every minority figure is Ethnologue-type, not a count.
- Sa'idi from Minya south plus New Valley; Beni Suef and Faiyum on Egyptian Arabic.
- Nubian: 300k Aswan, 237k cities. Putting all 537k in Aswan would make it 31% Nubian.
- Beja: Red Sea gets half the people south of 24N (Ababda and Rashaida, Arabic speakers, share
  that district); the rest of Ethnologue's 88,000 in Aswan. Least certain split on the map.
- Domari 10,000, not Ethnologue's current ~0.3% (about 350,000, three times the estimated Dom
  population of 100,000, so likely an ethnic figure).
- Refugees: UNHCR registrations (1.1M), not IOM's 2022 estimate of 9 million migrants (4M
  Sudanese, 1.5M Syrians), which Ethnologue's Sudanese Arabic 3.5% and Levantine 1.8% seem to
  follow. IOM publishes no governorate split.
- Bedouin outside Matrouh and Sinai (Beheira's and Alexandria's western fringe, Sharqia, the
  Eastern Desert's Ma'aza, Faiyum) are not split out: no source places them.
- Western Egyptian Bedawi drawn on `libyan_arabic` (Glottolog's parent), not a node of its own.

## 4. Room for improvement

- A survey that interviews in Nubian and Beja, or samples Siwa, Sinai and Matrouh.
- UNHCR nationality by governorate (the monthly statistical reports may carry it; unhcr.org is
  behind a bot wall, ReliefWeb's API needs an approved appname).
- Ethnologue's own Egypt page (403 to us) would give the dated Eastern Bedawi and Domari
  figures directly.

## Terms

Ethnologue figures are cited as quoted on Wikipedia (CC BY-SA). UNHCR fact sheet: public. Arab
Barometer, Afrobarometer: free, citation requested. Glottolog CC BY. Kontur CC BY 4.0. CAPMAS
population via religiondots' `eg_lookup.csv`.

## Moved from note_public (2026-10-06 sweep)

- Ethnologue's 27 million Sa'idi speakers are close to the 24.8 million drawn here (the Sa'idi
  governorates, Minya south to Aswan plus the New Valley).
