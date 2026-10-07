# Jordan (jo): record

Drawn 2026-10-05 (session edd42a8c-arab). Jordanians by pooled Arab Barometer first language
(and ethnic group for Circassian), everyone else by the 2015 census nationality on its home
language or mix; 2015 shares per governorate applied to the DoS end-2025 governorate
populations: 11,937,000 people, 12 governorates, 71 nodes. Jordanian rows `modelled`,
non-Jordanian rows `derived`. Placed on religiondots' Kontur 400 m hexes. 11,909 dots.

Files: `sources/jo_build.py`, `sources/ab_firstlang.py`, `taxonomy/jo2015.py` (identity),
`taxonomy/tree.d/jo.txt` (all borrowed), `countries/jo.py`, `data/normalized/jo.csv`,
`data/raw/jo/` (Tables 3.1, 8.1 PDFs).

```
python sources/jo_build.py [--fetch]
python taxonomy/build.py
python tools/check_country.py jo
python scatter.py --country jo
```

## 1. Sources

| source | item | grain | used |
|---|---|---|---|
| Census 2015 (DoS) | nationality, no language | governorate | Tables 3.1 + 8.1 |
| Census 2026 | in the field Oct 2026, form not seen | - | re-check when out |
| Arab Barometer II 2011 (1,188), III 2013 (1,795), IV 2016 (1,500) | first language | 12 governorates | yes |
| Arab Barometer VII 2021-22 (2,399) | ethnic group | 12 | Circassian only, and Arab as Arabic |
| WVS 4-7 Jordan | language at home | regions | not fetched |
| Rannut 2009, JMMD 30:4 | Circassian school pupils: 72% Arabic only at home, 6.5% Circassian only, 15% both | - | retention 21.5% |

Table 8.1 parsed from the PDF text (9 numbers per row; every row's sexes and urban/rural add
up; every group total equals its rows; national section = sum of the 12 governorates per
nationality; 3.1's non-Jordanians per governorate = 8.1's sum, all asserted).

## 2. Jordanians

4,483 first-language answers: Arabic 4,469, English 5, German 2, French 1, "Serbo-Croatian" 4,
missing/refused 2. The four "Serbo-Croatian" are in Jerash (2) and Mafraq (2); Table 8.1 has no
Bosnian or Serbian nationals in either, and Jerash is a Circassian settlement, so the label is
not readable as printed: on `other`. Arabic is drawn as Levantine Arabic (sa.txt's node,
nort3139 with South Levantine sout3123 as its dialect).

Circassian: AB VII ethnic group, 7 of 2,398 (Amman 3, Irbid 3, Zarqa 1) x 21.5% home use =
**1,968 people**. Published estimates of ethnic Circassians run 30,000-170,000 (Wikipedia's
Circassians in Jordan), so the survey finds too few; said in `note_public`. Chechen: no survey
answer, and the only figures found (8,000; 30,000 in 2020) are uncited Wikipedia lines, so not
drawn (Dweik 2000 says the language is well kept).

## 3. Non-Jordanians

2015: 2,918,125 of 9,531,712. Mixes via `sources/gulf_mix.py` `origin_mix` (sa.md method):
Syrians 1,265,514, Palestinians 634,182, Lebanese -> Levantine; Egyptians 636,270 -> Egyptian
Arabic; Iraqis 130,911 -> Iraqi Arabic; Yemenis -> Yemeni Arabic; Libyans 22,700 -> plain
`arabic` (Libya not drawn); Saudis, Gulf nationals on their drawn nodes; India and Sri Lanka
forced to their drawn mixes (gulf_mix's 20,000 floor would put Sri Lankans on Tamil); Canada
forced to English, Brazil to Portuguese (COUNTRY_LANG says French / nothing). 10,480 in
nationalities under 500 or "Other" rows -> `other`.

Drawn 2025 totals: Levantine 89.3%, Egyptian 6.7%, Iraqi 1.4%, Yemeni 0.3%, English 0.3%.

## 4. Calls someone might reverse

- 2015 nationality shares applied to 2025: Syrian returns since Dec 2024 (UNHCR registered
  ~0.5M in 2025-26 against the census's 1.27M) not reflected.
- Circassian from 7 ethnic answers x a school-survey retention; Chechen not drawn.
- "Serbo-Croatian" answers on `other`.
- Palestinian refugees with Jordanian nationality are inside "Jordanians" (all Levantine
  anyway); UNRWA camp counts not used since they change no node.

## 5. Room for improvement

- The 2026 census (if it asks language). WVS Jordan waves (4 rounds) via the online tool.
- A Circassian / Chechen count by locality (Wadi al-Seer, Sweileh, Zarqa, Jerash, Azraq).

## Terms

DoS census tables: public. Arab Barometer: free, citation requested. Glottolog CC BY. Kontur
CC BY 4.0.
