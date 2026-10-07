# Language coverage sweep, 2026-10-03/04

Question: how much of the world could a first-language dot map (a sister to religiondots)
draw from census data? Seven research agents each covered a region, under `BRIEF.md`.
Each region has a CSV and a short write-up beside it. `coverage.csv` is all 222 rows merged,
with a tier and a 2024 World Bank population; `python merge.py` rebuilds it, and the
tier list is at the top of that script.

This is a first pass. About a third of the rows are `reported` or `guess`, not `checked`, and
small countries got little effort. Negatives record what was tried, not verdicts.

## Share of world population by tier

| Tier | Meaning | Countries | Population | Share |
|---|---|---|---|---|
| A | census language data at district level or finer | 64 | 3.2 bn | 39% |
| B | real language data, but coarse, multi-answer, or old | 41 | 1.0 bn | 12% |
| C | no question, but ~95%+ speak one language | 49 | 0.5 bn | 6% |
| D | ethnicity only, fine enough to crosswalk | 27 | 2.2 bn | 26% |
| E | nothing usable from the census | 41 | 1.3 bn | 15% |

A + B + C, the part that needs no proxy, is about 57%. China alone is 17 points of tier D.

- **A:** India, US, Pakistan, Brazil, Ethiopia, Mexico, South Africa, Great Britain, most of
  Latin America, Canada, Australia, and Central/Eastern Europe.
- **B:** Indonesia, Russia, Germany, Thailand, Morocco, Angola, Uzbekistan, Mozambique.
- **C:** Japan, Egypt, the Koreas, Yemen, Madagascar, Somalia.
- **D:** China, Bangladesh, Philippines, Vietnam, Kenya, Myanmar.
- **E:** Nigeria, DR Congo, Iran, Turkey, Tanzania, France, Italy, Sudan.

## The best sources

- **India 2011 C-16:** about 270 mother tongues by sub-district (about 5,900 units), xlsx
  per state. The mother-tongue rows separate Bhojpuri, Rajasthani etc. from Hindi.
- **Nepal 2021:** about 125 mother tongues for all 753 local levels in one xlsx.
- **Pakistan 2023 Table 11:** about 15 languages by tehsil. The office's PDFs now 404; the
  CRAN package PakPC2023 carries the table.
- **Canada 2021:** mother tongue by dissemination area.
- **England and Wales 2021:** main language by output area, 106 categories, OGL.
- **Australia 2021:** home language by SA1, CC BY.
- **Latin America via REDATAM:** Peru (district), Guatemala, Bolivia 2024 (municipality)
  mother tongue. Mexico, Colombia, Argentina and Chile ask indigenous languages only, by
  municipality, and Spanish is the remainder.
- **Central/Eastern Europe:** mother tongue by municipality, open tables (Romania and
  Slovakia checked).
- **Zambia 2022:** language group for all 1,853 wards in one PDF.
- **South Africa 2022:** home language by ward (SuperWEB2, free account).
- **Ethiopia 2007:** mother tongue by woreda, US Census Bureau file on HDX, CC BY.
- **Morocco 2024:** Darija/Tachelhit/Tamazight/Tarifit/Hassania by commune. Multi-answer;
  shares add to about 117%.
- **Laos 2015:** 10 ethno-linguistic groups for about 8,500 villages (open ArcGIS server).

## Multi-country finds

- **CLEAR Global (formerly Translators without Borders) on HDX, CC BY-SA:** district
  language shares for Mali, Senegal, Guinea, Sierra Leone, Benin (from IPUMS census samples)
  and DR Congo (2016 admin reports); Afrobarometer-based ones for several more. Shares, not
  counts. Reuse terms of the IPUMS-derived ones need a check. Benin's may be ethnicity.
- **Afrobarometer:** open download, home language with a region variable, 21 of 24 West
  and Central African countries plus much of East/South Africa and the Maghreb.
- **US Census Bureau on HDX:** language tables for Indonesia 2020 (but only
  Indonesian/regional/foreign), Pakistan 2017, Ethiopia 2007, Ukraine 2001, CAR 2003.

## The hard gaps

- **China:** no census has ever asked about language. Minority languages crosswalk from minzu
  (maps/chinaethnicity has the county tables). Han dialect groups exist only as one label per
  county: the Language Atlas of China coded to 1990 counties (Harvard Dataverse, non-commercial
  academic use only) or an unlicensed GitHub classification. Breaks in migrant cities.
- **Nigeria** (no question since 1963) and **DR Congo** (no census since 1984): Afrobarometer
  for Nigeria; CLEAR's territory file for DR Congo.
- **Indonesia:** detailed 2010 languages by province only; regency needs IPUMS (gated).
- **Western Europe:** France, Belgium, the Netherlands, Italy, Spain, Portugal, Greece and the
  Nordics (except Finland) ask nothing. Spain's Basque Country (Eustat) is the one regional
  first-language source.
- **Turkey** (only 1965, by province), **Iran** (a 2015 ministry survey, percentages only),
  **Algeria** (1966), **Sudan**, **Iraq** (2024 census dropped language on purpose),
  **Afghanistan** (no census), **Tanzania**, **Cameroon**.

## Caveats that apply across regions

- The question differs: mother tongue, home language, main language, ex-USSR "native
  language". Native language leans towards identity: Belarus 2019 has Belarusian as 54%
  native but 26% at home.
- The US asks only about languages other than English, ages 5+, and only 12 groups at
  tract level.
- Balkan labels (Serbian/Croatian/Bosnian/Montenegrin, Moldovan/Romanian) follow identity.
- Several "C" calls are judgement: Egypt, Yemen, Portugal, Japan count as one language here.

## Things that didn't work (2026-10-03)

- UNSD Demographic Yearbook table 27 (population by language): data.un.org now serves the
  UN System Data Commons app for every URL, including the old DownloadHandler route that
  worked for religion table 28. Not chased further.
- IPUMS-I per-variable pages for `LANG` and `ETHNIC` return 404; the group page
  `/international-action/variables/group/ethnic?page=1` works.
- UNECE database and Eurostat 2021 census hub: no language tables.
- Blocked or down: Ukraine and Albania statistics sites, Hungary census site (403), Ecuador INEC
  (403), Guatemala (bot wall), Mozambique INE (timeout), SPC PopGIS and Pacific Data Hub.
  India, Nepal and Senegal offices have broken TLS chains (`curl -k`).
