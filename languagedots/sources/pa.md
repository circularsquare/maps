# Panama (pa): record

Drawn 2026-10-05 (session edd42a8c-amer). 2023 census groups x MICS 2013 mother tongue by group,
4,064,780 people on 74 district units, every row `modelled`. Spanish 85.7%, Ngäbere 375,208,
Kuna 100,333, Embera 35,490, English 12,271, Wounaan 9,656, Buglere 3,449, Bribri 766, Naso
316, other 43,100. 4,061 dots, 3 rings.

Files: `sources/pa_censo.py`, `sources/pa_mics.py`, `sources/pa_geo.py`, `taxonomy/pa2023.py`,
`taxonomy/tree.d/pa.txt`, `countries/pa.py`, `data/raw/pa/` (5 programs + outputs),
`data/normalized/pa.csv`, `pa_units.csv`, `pa_mics_shares.csv`, `data/geo/pa/`.

## 1. Sources

- **INEC, XII Censo de Población y VIII de Vivienda 2023**, REDATAM www.inec.gob.pa/panbin, base
  LP2023 (open). P08 grupo indígena, P09 grupo afrodescendiente; no language question (VARLIST
  read). One joint code 10*P08+P09 per corregimiento (699). An AREALIST of ~100 columns returns
  HTTP 404, so FREQUENCY ... AREABREAK is used. Checks pass: sum 4,064,780 = crosstab Total;
  joint code = national P08 x P09 crosstab cell by cell; P08 and P09 margins = separate AREABREAK
  tables on every corregimiento; corregimientos rebuild the 13 provinces. Code labels recovered
  by matching national totals one to one.
- 2010 census indigenous report (INEC P6571) checked: no language table.
- **MICS 2013 Panama** (religiondots' raw copy, hl.sav, 42,568 people, hhweight): HC1B "lengua
  materna / idioma nativo" for every household member, with HC1E indigenous people and HC1G
  Afro group. The retention source.

## 2. Retention (MICS shares, weighted)

Inside / outside the three comarcas: Ngäbe 93.5% / 69.1% Ngäbere; Kuna 95.8 / 83.8; Emberá 98.1 /
59.0; Wounaan 97.4 / 73.0; Buglé 13% Buglere, 43-53% Ngäbere, 28-37% Spanish. Naso/Teribe (n=41)
4.2% Naso, 88% Spanish. Afro-Antillean 8.0% English (Bocas 18.8%, Colón 9.8%); "Negro" 1.9%;
Afrocolonial 0.3%. No group or afro: 98.4% Spanish, 1.4% other.

Calls: comarca split only for the five big groups; Bokota takes Buglé's vector (MICS's 23
"Bokota" sit in the Emberá comarca speaking Emberá, a miscode); Bri Bri 100% Bribri (MICS n=1);
"Otro grupo indígena" (45,498) takes the no-group vector (MICS "Otro" n=42, 65% Spanish); for
indigenous groups, cross-codes other than plausible neighbours dropped and renormalised (6%
"Kuna" among Buglé, 5% among Naso, look like keying slips). Census Afrodescendiente and
Afropanameño take MICS "Negro"; Moreno and "otro afro" the no-group vector. Afro-Antillean
"Inglés" drawn on English, as the survey labels it, not on a creole node. Naso's 4% rests on 41
people; a reviewer could prefer a higher figure from the literature.

## 3. Geography

COD-AB 2021 (594 corregimientos, 76 districts, alphabetical province codes) predates ~110 of the
census's 699 corregimientos. Units: census corregimiento names matched by fold within province
(583 of 699; names found in several COD districts tie only if nothing else does); a tie links
the census district to the COD district; connected groups are units. 74 units, 7 multi-district
(e.g. Almirante 0104 with Bocas and Changuinola; Jirondai 1208 with Kankintú; Santa Catalina
1209 with Kusapín). Kontur log r = 0.993 vs 0.349 shuffled, 0 of 74 outside a factor of 3.

## 4. Room for improvement

2023 corregimiento boundaries from INEC would give the 699-unit grain. A 2023 MICS or a census
language question would replace 2013 shares.

## 5. Immigrant languages (2026-10-05, session edd42a8c-latn)

Anita's priority: non-indigenous minority languages in Latin America. Method and retention rule:
`sources/mx.md`, "Immigrant and settler languages" §2 (`sources/latam_immig.py`). Now: Spanish
3,501,587 (86.1%), English 16,526, Hakka 6,650, Portuguese 2,076, Mandarin 1,355, Cantonese 929,
then Hindi, Levantine Arabic, Italian, French, German. 4,048 dots, 209 rings.

Files: `sources/pa_imm.py`, `data/raw/pa/pa_nat_naci{cod,lab}.htm`, `pa_corr_paisx0-4.htm`
(+ programs), `data/normalized/pa_imm.csv`; `countries/pa.py`, `taxonomy/tree.d/pa.txt`
(borrowed-node block), `sources/origin_mix.py` (one override) changed.

- **Country of birth**: P_NACIO = 3 (born abroad, 249,476) and P_NACI_COD, an INEC code with no
  labels; RP05_NACI prints the names in the same order, and the two frequencies pair 189 of 189
  rows with equal counts. 161 non-Spanish-speaking countries hold 42,414 (China 12,420, US 8,511,
  India 3,719, Brazil 2,396, Taiwan 1,921, Italy 1,452, Canada 1,451, France 1,081). Colombians
  (66,689), Venezuelans (60,290) and Nicaraguans (30,761) stay Spanish. "Otros de Europa / Asia /
  África / Oceanía" and "país no declarado" (226) stay Spanish. Checks: per country,
  corregimiento sums equal the national frequency; each chunk's columns sum to 4,064,780.
- **MICS's "other" now goes to Spanish** (43,100 people), and so does the 0.1% English of people
  of no indigenous or Afro group (3,168): both were mostly immigrant languages, which country of
  birth now draws. Afro-Antillean English (MICS, 8%) is unchanged.
- **China-born: Hakka 80 / Cantonese 10 / Mandarin 10**, an `origin_mix` override for Panama
  from Wikipedia's "Chinese people in Panama" ("around 80% of this population are of Hakka
  origin"); its reference was not checked. Taiwan-born (1,921) keep Taiwan's home mix (Mandarin,
  Min Nan). The Panama-born descendants of the older Chinese community are Spanish here.
- India-born (3,719) take India's home mix (Hindi first). Panama's Indians are widely described
  as largely Gujarati; no figure was found, so no override.
