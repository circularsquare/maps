# Equatorial Guinea (`gq`)

**Drawn 2026-10-03** by session `fafd1067-gq`. 7 provinces, 1,225,377 people (2015 census), every
row `modelled`, one national mix. Rulings: `ask/RULINGS.md` 2026-09-15 (priority holes; ask 034, the
two preliminary files) and 2026-09-16 (Mauritania: draw on the best survey or compiler figure, with
the method said). Code: `sources/gq_geo.py`, `sources/gq_grid.py`, `sources/gq.py`,
`taxonomy/gq2011.py`, `countries/gq.py`; node `other.gq` in `taxonomy/branches.py`. `sources.md`
§gq-2026-10-03. Earlier records: `sources.md` §11aq, §scout-2026-09-15-negatives, §gq-2026-09-15.

## 1. The census asked and published nothing on it

- **Form**: 2015 *Cuestionario IV Censo General*, Bloque V question 9 *¿Qué religión profesa?*: sin
  religión, católica, protestante, Islam, otra, no sabe/no contesta (UNSD archive
  `quest/GNQ2015esHh.pdf`, read 2026-09-15).
- **The three volumes, every page viewed, no religion and no ethnic table in any:**
  - *Resultados definitivos* (72 pp, phone scan; read 2026-09-15, `sources.md` §gq-2026-09-15).
  - *Resultados preliminares del IV Censo de Población 2015* (Anita's download for ask 034,
    `data/raw/gq/RESULTADOS-PRELIMINARES-DEL-IV-CENSO-DE-POBLACION-2015.pdf`, 3,677,801 bytes,
    SHA-256 `fad1e606...ea11`, 16 pp, no text layer, rendered and read 2026-10-03). Tablas 1.1
    region, 2.1 province, 3.1 sex, 4.1-4.2 nationals and foreigners by province and sex, 5.1
    urban/rural, 6.1 density, 7.1 households. Nothing else.
  - *Síntesis de los resultados preliminares del censo 2015*
    (`data/raw/gq/SINTESIS-DE-LOS-RESULTADOS-PRELIMINARES-DEL-CENSO-2015.pdf`, 357,289 bytes, SHA-256
    `b25d2e92...8e2b`, 3 pp, text layer). Population by province and sex, nationals and foreigners,
    a few national indicators. Nothing else.
- Not checked, as before: the 1983 and 1994 forms; whether the 2015 microdata is deposited anywhere.

## 2. What the map draws

**DHS EDSGE-I 2011**, final report FR271 (`data/raw/gq/dhs_fr271.pdf`, from
`dhsprogram.com/pubs/pdf/fr271/fr271.pdf` with a browser user agent; WebFetch gets 403). Cuadro 3.1
(printed p.30, PDF p.60), women and men 15-49, weighted; read back from the text layer on every run.

| | women % | women n (wtd) | men % | men n (wtd) | drawn, of answers |
|---|---:|---:|---:|---:|---:|
| Cristiano | 96.3 | 3,442 | 91.9 | 1,432 | 94.87% |
| Musulmana | 1.9 | 68 | 5.4 | 84 | 3.81% |
| Animista | 1.0 | 34 | 0.6 | 10 | 0.79% |
| Sin religión | 0.0 | 2 | 0.4 | 6 | 0.23% |
| Otro | 0.2 | 7 | 0.4 | 6 | 0.30% |
| Sin información | 0.6 | 22 | 1.3 | 20 | out of the base |

3,575 women and 1,612 men interviewed (unweighted). The card (questionnaires PDF pp.365, 437) has
one Christian code; the questionnaire at PDF p.365 prints code 3 as *Presbiteriana* (with *Sans
religion*) where the one at p.437 and the table have *Animista*, read as a draft left in the annex.

- **Sexes combined** at the census's 53.3% men (preliminary Tabla 3.1: 651,820 men, 570,622
  women). Unweighted pooling would have given the men's half the weight of its subsample.
- **Christians split 88:5** at the government estimate for 2015 that the US State Department's
  *2023 Report on International Religious Freedom: Equatorial Guinea*, Section I, quotes (read
  2026-10-03): "88 percent of the population is Roman Catholic, 5 percent Protestant, and 2 percent
  Muslim. Most of the Muslims are Sunni and expatriates from other West African countries. The
  remaining 5 percent ... animism, the Baha'i Faith, Judaism, or other beliefs." The CIA Factbook
  (mirror `factbook/factbook.json`, `africa/ek.json`) has the same figures marked "(2015 est.)". No
  method is named. `sources/gq.py` stops if the estimate's Christian total (93%) is more than 4
  points from the survey's (94.87%). Reasoning: Anita's Bulgaria reversal (the survey sets the total,
  the estimate only divides it, and the rest of central Africa is drawn with the split). Fallback:
  bare `christianity` for all of it, one line in `taxonomy/gq2011.py`.
- **As drawn:** Catholic 1,099,978 (89.77%), Protestant 62,499 (5.10%), Muslim 46,632 (3.81%),
  traditional (`indigenous.african`) 9,720, `other.gq` 3,676, unaffiliated 2,872.
- **Witnesses, not drawn:** Pew 2020 (everyone resident) 88.68% Christian, 4.05% Muslim, 4.99%
  unaffiliated, 2.13% other; its 2010 and 2020 shares are identical to six decimals, so it is one
  estimate carried forward. The 2015 government estimate's Muslim 2% is below the survey's 3.81%.

## 3. Muslims and foreign residents: not placed

The preliminary census counted 209,611 foreign residents (17.1%; 77.1% men): Litoral 70,663 (19.3%
of the province), Bioko Norte 54,570 (18.2%), Wele-Nzas 40,190 (21.0%), Kié-Ntem 28,233, Centro Sur
13,055, Bioko Sur 2,597, Annobón 303. No nationality. UN DESA's 2024 migrant stock for Equatorial
Guinea is the census figure itself (209,611 in 2015) with 200,865 of it under `Others`, so there is no
origin mix to borrow. The DHS 2011 sample was 4.1% foreign among women and 8.2% among men
(Cuadro 3.1's *Extranjero* ethnicity), well under the census's 17.1% four years later.

Solving the two sexes' marginals for "Muslim share among foreigners" gives about 0.84 and a
slightly negative share among nationals, which says the survey's Muslims are nearly all foreign
but is two marginals and 128 foreign men; not a basis for a layer. So Muslims are drawn at the
national share everywhere and `note_public` says they are probably undercounted and more
concentrated in the three provinces than drawn. Reopen with the DHS microdata (ask) or any table
of religion by nationality.

## 4. Geography

- **COD-AB** `cod-ab-gnq` v01 (OCHA ROWCA, valid 2021-07-15, reviewed 2025-10-30),
  `gnq_admin1.geojson`: 7 provinces GQ198-GQ204, no Djibloho (made a province in 2017; its Ciudad
  Nueva Oyala is a Wele-Nzas municipality in COD's admin2, where the 2015 census counted it).
  Admin2 is 32 municipalities, not the census's 18 districts, and is not used. No COD-PS
  (`cod-ps-gnq` 404 on 2026-10-03).
- **Base**: *Resultados definitivos* Tabla 2.1 (printed p.18), transcribed from the scan:
  Bioko Norte 300,374; Bioko Sur 34,674; Annobón 5,314; Litoral 367,348; Centro Sur 141,986;
  Wele-Nzas 192,017; Kié-Ntem 183,664; 1,225,377. Checked against Tabla 1.1 (insular 340,362,
  continental 885,015), Tabla 3.1's 18 districts (printed p.19, each province's districts sum
  exactly) and the preliminary figures (all within 0.35% but Annobón, +1.57%).
- **Area witness**: preliminary population over Tabla 6.1's whole-number density, against COD:
  0.885 (Bioko Norte) to 1.067 (Centro Sur), Wele-Nzas 0.894. Bar 15%. With one national mix a
  misdrawn province line moves same-coloured dots only.

## 5. Placement

- **Kontur `GQ`** (2023-11-01): 3,816 hexes, 1,719,138 people; 1.397x the 2015 count (the US
  government's 2023 figure is 1.7 million). 186 hexes (63,015 people) have centroids outside every
  province: 141 within 2 km snapped (coasts), 45 beyond dropped (6,804 people, 0.40%, along the
  Cameroon and Gabon borders).
- Per province raw Kontur over its census share: 0.90 (Centro Sur) to 1.03 (Bioko Norte); the rank
  witness is the true ordering, 1 of 5,040. 1.3% of Kontur's people sit in a different province
  from the census. Every hex scaled to its province's count.
- **GeoNames places of 5,000+**: Luba (0.17 within 5 km, 0.40 within 10) and Abaamang (0.15, 0.67)
  low, none under the 0.10 bar; Mongomo and Rebola far above GeoNames' old figures.
- **One block at Kontur's cap, left**: 10 hexes, 191,402 raw people at (8.8148, 3.7316), 3-4 km
  south-east of GeoNames' Malabo point, around which Kontur holds only 11,073 people within 2 km.
  Kontur draws Malabo's population on its south-eastern neighbourhoods rather than the old centre.
  Capping to the 3 km ring's median would take the city to about 10,000 weight and the province
  scaling would spread it over Baney and the forest, below the census's urban 272,249 for Bioko
  Norte. Left, as `af`'s Kabul and Herat; `gq_grid.py` asserts the scaled block (133,284) stays
  under that urban count. Not checked against a second grid.

## 6. Reopen if

- The DHS 2011 microdata (`GQIR`/`GQMR`, `v130` by the four sample domains: Malabo urban, rest of
  Bioko, Bata urban, rest of the mainland; Appendix A Cuadros A.4-A.6) is ever registered for. It
  would place Muslims between the two cities and the rest, not by province. Ask filed.
- INEGE publishes a 2015 table on religion (question 9) or nationality by province.
- A source with a method for the Catholic and Protestant split.

## 7. Review, 2026-10-03 (`fafd1067-rev6`)

Full pass, nothing changed. Checks clean, rollup clean. `data/normalized/gq.csv` is the same mix in
all 7 provinces and re-sums to the note's figures (89.77 / 5.10 / 3.81 / 0.79 / 0.30 / 0.23) on
1,225,377; the sex weighting reproduces the Muslim 3.81% from Cuadro 3.1's 5.4 and 1.9. No `gap`,
matching `cu`'s one-national-mix entry; the undersampled foreign residents are in the note instead,
which is the honest place for a bias the build cannot correct. On the map Malabo and Bata carry the
weight and the mainland follows its roads; nothing in the sea.
