# Dominican Republic (do): record

Drawn 2026-10-05 (session edd42a8c-carib). ENHOGAR-MICS6 2019, mother tongue of the household
head (HC1B), shares per province applied to the 2022 census province populations: 10,771,505
people in 5 answers over 32 provinces, every row `modelled`. Placed on religiondots' Kontur hexes
(read-only).

Files: `sources/do_enhogar.py`, `taxonomy/do2019.py`, `taxonomy/tree.d/do.txt` (bare repeats, no
new nodes), `countries/do.py`, `data/normalized/do.csv`. Raw data are religiondots'
(`../religiondots/data/raw/do/mics6_2019_hogares.csv` and `.sav`), read only.

## 1. What exists, and why this source

- **No census asks language**: neither 2010 nor 2022 (scout 2026-10-05; religiondots checked the
  2022 microdata codebook, P25-P67, for religion and found self-identification by colour only).
- **ENHOGAR-MICS6 2019** (ONE with UNICEF), household questionnaire item HC1B: "Cual es el idioma
  materno del jefe o la jefa del hogar, es decir, el que hablaba cuando era nino(a)?" Answers
  Espanol / Creol / Ingles / Frances / Otro idioma. 31,488 completed households, every one
  answered, all 32 provinces (HH7A), 450-2,180 households a province. religiondots already holds
  the open microdata for its religion item (HC1A) beside it.
- **Why not ENI-2017** (the lead in the brief): ENI counts people born in Haiti (497,825) and
  people born in the country to Haitian parents (253,255), but by origin, not language, and its
  published tables are by region rather than province. HC1B is a language question asked in every
  province, so it is the better source; ENI serves as the check below.

## 2. Method and checks

- Each household weighted `hhweight x HH48` (members): a province's share is the share of its
  people living in a household whose head's mother tongue is that language. Applied to the 2022
  census population (`do_lookup.csv` `pop_2022`), as religiondots does with HC1A.
- National: Spanish 93.5%, Haitian Creole 6.3% (673,975), English 0.08%, French 0.07%, other
  0.06%. Asserted: household count, no blank answers, 32 provinces, sum equals the census total
  (exact), Creole 5-8%.
- **ENI-2017 check**: Haitian origin (born in Haiti, or born in the country with a Haiti-born
  parent) 751,080, about 7.4% of the 2017 population. The Creole-head share (6.3%) sits below it,
  as it should: some of the descendants live in Spanish-headed households (81,590 have one
  Dominican-born parent). Sources: ONE/UNFPA, *Segunda Encuesta Nacional de Inmigrantes*
  (one.gob.do/media/orcb0gse/eni-2017-finalweb.pdf) and its descendants volume.
- Highest Creole shares: Independencia 25.7%, Monte Cristi 20.1%, Elias Pina 18.3%, Pedernales
  17.1% (the border), El Seibo 15.9%, La Altagracia 15.8%, La Romana 10.9% (the east's sugar and
  tourism economy). The pattern is what the ENI and religiondots' no-religion map would predict.

## 3. Mapping (taxonomy/do2019.py)

Creol on Haitian Creole (in the Dominican Republic "creol" is Haitian Kreyol; French is a separate
answer). Ingles on English: some is Samana English and some the *cocolo* Anglophone Caribbean
descendants in the east; the survey does not tell them apart. Otro idioma on `other`. No colour
changes: Spanish and Haitian Creole's lilac are far apart.

## 4. Calls someone might reverse

- The head's mother tongue applied to the whole household. It overstates Creole for children of
  Haitian parents raised in Spanish, and understates it for Creole speakers living in
  Spanish-headed households; said in `note_public`.
- 2019 shares on the 2022 population. Haitian migration has been heavy since 2021 (and
  deportations heavy since 2024), so today's share may differ in either direction.
- English, French and other rest on 1-8 households per province and are drawn where they were
  found rather than pooled to a national rate; together they are 0.2%.

## 5. Room for improvement

ENHOGAR rounds since 2019 (if they repeat HC1B) could be pooled to steady the small answers.
A person-level language question would fix the household-head step.

## 6. Scatter

10,769 dots at 1:1000 over 4,746 hexes; 2,505 people (0.02%) under one dot per language.
