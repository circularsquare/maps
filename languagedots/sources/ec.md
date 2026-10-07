# Ecuador: Censo 2022, languages spoken, canton

**Drawn** 2026-10-05 (agent d9e44929-ec2, after d9e44929-ec was stopped with nothing on disk).
16,611,076 people on 221 cantons, 17 nodes, 16,603 dots at 1:1000, 7 rings. Every row
`derived` (a multi-answer question, spec §3.6), but the per-person split is counted, not scaled.
Redrawn the same day (d9e44929-svec) with the foreign-language shares drawn as Spanish, Anita's
ruling below.

## Source

- INEC, VIII Censo de Población y VII de Vivienda 2022, tabulado
  `2022_CPV_Autoidentificacion_Cultura.xlsx` ("Autoidentificación según cultura y costumbres"),
  `https://www.censoecuador.gob.ec/wp-content/uploads/2024/02/2022_CPV_Autoidentificacion_Cultura.xlsx`.
- Question: "Idiomas o lenguas que habla o se comunica", everyone aged 1 and over, several
  answers: an indigenous language (one, named), Castellano, a foreign language (not named),
  Ecuadorian Sign Language. So `how` is "languages spoken, several allowed".
- **Access.** `www.censoecuador.gob.ec` answers 403 (Apache, not a challenge page) to every path
  from this machine on 2026-10-05, browser User-Agent included; so do `redatam.inec.gob.ec` and
  `anda.inec.gob.ec`. `www.ecuadorencifras.gob.ec` answers. The workbook came from the Wayback
  Machine's 2025-01-14 capture (found by CDX over `censoecuador.gob.ec/*`; it arrives
  gzip-encoded), and is byte-identical to religiondots' copy fetched live on 2026-09-08
  (`../religiondots/data/raw/ec/`). `sources/ec_censo.py --fetch` copies religiondots' file
  first, then tries live, then Wayback (which rate-limited with a 429 on the second try).
- **Why canton and not parish.** The coverage sweep's lead said "parish (open microdata
  download)". The microdata sit behind the 403 (ANDA), and so does REDATAM; the census site's
  other route to parish figures is a Power BI embed (`cubos.inec.gob.ec/AppCensoEcuador/`). The
  tabulado's finest geography is the canton. Parish would be worth having for the Andes, where
  Kichwa parishes sit beside Spanish ones inside one canton; a session that can reach ANDA from
  an Ecuadorian-looking network, or Anita in a browser, could get it.

## Tables used and the split

| table | what | per canton |
|---|---|---|
| 5.1 | people 1+ by combination of the four classes: 4 singles, 6 pairs, "three or more", "does not speak" | yes |
| 7.1 | mentions per class | yes |
| 10.1 | indigenous speakers by language: 14 named + "Otras Lenguas Indigenas" | yes |
| 1.1 | whole population (for the under-1s) | yes |

One class: 1 to it. A pair: 1/2 to each. "Tres o más idiomas" (7,594 people) is solved
exactly from 7.1: with r_l the class mentions not used by singles and pairs, n4 = Σr − 3·n3
named all four (101 nationally) and n3 − r_l named the three without l, so class l gets
n4/4 + (r_l − n4)/3. The canton's indigenous share is then divided among its languages by
10.1's proportions. **The one assumption:** within a canton, speakers of each indigenous
language are equally likely to be bilingual. Nothing published crosses language with
combination. It matters most where a monolingual community (Waorani, some Achuar) shares a
canton with a mostly bilingual one (Kichwa).

The split gives nationally (people, then mentions): Spanish 15,967,282 (16,469,637); Kichwa
307,329 (538,449); foreign 247,998 (472,520); Shuar 35,165 (58,770); other indigenous 15,109
(25,566); sign 14,573 (19,992); Cha'palaa 9,916 (14,154); Achuar 5,888 (9,162); the other ten
under 2,000 each. Not drawn: 241,750 under-1s and 86,160 "No habla/No se comunica" (62,048 of
them aged 1-4), 1.9% together.

**The foreign-language shares are drawn as Spanish** (Anita, 2026-10-05: "ideally we don't use
English like this, to be consistent with neighbours", said of El Salvador; Ecuador had the same
build). Colombia and Peru, like the rest of the region, draw only the indigenous languages as
measured and everyone else as Spanish (spec §3.5), while the first build here drew 247,998 people
in `other`, two thirds of them in Quito, Guayaquil and Cuenca, from a question that counts second
languages. So `countries/ec.py` moves every "Idioma extranjero" share to Spanish and leaves the
indigenous and sign shares exactly as counted (an indigenous + foreign speaker is still half
indigenous). The 5.1 combinations behind the foreign share, nationally:

| combination | people | foreign share |
|---|---|---|
| Castellano + foreign | 438,704 | 219,352 |
| foreign only | 25,949 | 25,949 |
| indigenous + foreign | 264 | 132 |
| foreign + sign | 234 | 117 |
| three or more, with foreign | (7,369 mentions) | 2,448 |
| total | 472,520 mentions | 247,998 |

The 25,949 who named only a foreign language, no Spanish, are drawn as Spanish too, as a
neighbour's remainder would be; that is the call most open to reversal (put them back on `other`
by folding only the shares of people who also named Castellano, which 5.1 allows per canton).

Drawn now: Spanish 16,215,280 (97.6%), Kichwa 307,329, Shuar 35,165, other indigenous 15,109, sign
14,573, Cha'palaa 9,916, Achuar 5,888, the other ten under 2,000 each. `countries/ec.py` asserts
the drawn shares equal each canton's population less its under-1s and no-habla, all 221.

## Checks (`sources/ec_censo.py`, all asserted, all pass)

- 221 cantons in 24 provinces; 5.1, 7.1 and 1.1 have the same set; 10.1 has 217 (four cantons
  have no indigenous speaker; 7.1 agrees, 0).
- 5.1's twelve combinations sum to the row's 1+ total on every unit; "No habla" is equal in 5.1
  and 7.1; 10.1's fifteen languages sum to its own total, which equals 7.1's indigenous
  mentions, on every unit; 1.1 population ≥ 5.1's 1+ population.
- The combinations reproduce 7.1's mentions: 0 ≤ r_l ≤ n3 and 0 ≤ n4 ≤ min r on all 221
  cantons. This is the second-table check: two tables of the same census agreeing per unit.
- Cantons rebuild provinces, and provinces the national row, on every cell of all four tables.
- National pins: population 16,938,986 (the census count religiondots also uses); 1+
  16,697,236; indigenous 659,361, which is 3.95% of 1+, INEC's own headline; Castellano
  16,469,637; foreign 472,520; sign 19,992; Kichwa 538,449; Shuar 58,770.
- Each canton's drawn shares + no-habla + under-1s equal its 1.1 population exactly.
- The population includes the ~759,000 people INEC imputed for occupied dwellings whose
  residents were absent (VOPA, note 2 of every sheet); their language is imputed too.

## Geography (`sources/ec_geo.py`)

- Units: COD-AB Ecuador 2024 ADM2, 221 cantons, read in place from religiondots' zip. The
  tabulado prints names only; folded names match all 221 inside their province with no alias,
  asserted one to one.
- Witness 1, order: INEC lists cantons in DPA code order and COD's pcode is EC + that code; the
  joined pcodes rise down the sheet in all 24 provinces.
- Placement: religiondots' `ec_hexes.gpkg` (plain Kontur EC 2023-11-01, outside hexes already
  dropped) re-keyed from province to canton by hex centroid in EPSG:3857. 113,005 hexes, 0 in no
  canton, 0 landing in a canton outside their religiondots province; every canton has populated
  hexes (22 at least).
- Witness 2, Kontur: 1.071x the census nationally; per canton, normalised, p10 0.85, median
  0.96, p90 1.16; lowest Isabela (Galápagos) 0.63, Daule 0.67, Samborondón 0.68 (Guayaquil's
  fast-growing suburbs; Kontur's 2023 grid is under them), highest Colta 1.52, Quijos 1.46;
  none outside a factor of 3. Log correlation 0.993 against a best of 0.231 over 500 shuffles.
- Placement is `pop_weight` on Kontur alone. Scatter: no Kontur cap block stopped it, so nothing
  added to `kontur_cap.csv`.

## Mapping calls (`taxonomy/ec2022.py`, `taxonomy/tree.d/ec.txt`)

- Every named label has its own node. New: Shuar, Shiwiar (Chicham); Andoa, Záparo
  (Zaparoan); Cha'palaa, Tsafiki (Barbacoan); Waorani (isolate); Ecuadorian Sign Language.
  Glottolog files Shiwiar as an Achuar dialect; the census prints it apart, so it is Achuar's
  sibling (a child would make Peru's Achuar a group node).
- Reused: Kichwa → `quechuan.kichwa` (pe.txt; the same name and the same Quechua II-B north of
  the border and in Peru's Loreto), Achuar, A'ingae → `isolate.cofan`, Awapit →
  `barbacoan.awa_pit`, Siapedee → `chocoan.embera.eperara` (Glottolog Epena, epen1239, CO/EC/PA).
- **Bai Coca → Siona, Paaikoka → Secoya.** Ecuadorian Siona is called Baicoca; Paicoca is the
  Secoya's name. The census geography agrees: Bai Coca's largest canton is Cuyabeno (254 of
  379), Paaikoka's is Shushufindi (426 of 580), where the Secoya's San Pablo de Kantesiya is.
  Easy to reverse if wrong: swap two lines.
- "Otras Lenguas Indigenas" → `americas_other`, kept apart from "Idioma extranjero" → `other`
  (the mapping still says what the label means; since 2026-10-05 `countries/ec.py` draws that
  share as Spanish).
  Its biggest cantons are Quito (5,589) and Guayaquil (4,330), then Taisha (1,054), a Shuar and
  Achuar canton, so some of it is likely unrecognised Chicham names. Not split.
- One oddity left as published: 524 Cha'palaa speakers in Pedro Moncayo (Pichincha), a Kichwa
  Kayambi canton far from Esmeraldas. Could be flower-plantation migrants, could be miscoding;
  nothing to check it against.
- Colours: Shuar hand-picked a light blue-violet (away from Achuar's magenta and Peru's Awajún);
  Shiwiar pale pink; Cha'palaa cyan beside Awa Pit's green and Épera's ochre; Tsafiki teal;
  Waorani a rose, away from Cofán's and Achuar's magentas. Kichwa (amber) against Spanish
  (muted salmon) is the main edge and already reads.

## Corroboration of the remainder

Not needed: the question went to everyone aged 1+, so nobody is a derived Spanish speaker in
the §3.5 sense, except the foreign-language shares folded into Spanish (247,998, 1.5%), which
are a choice of what to draw, not a gap in the question.

## Immigrant languages (2026-10-06, session 5d7dac7e-br)

`python sources/ec_immig.py` -> `data/normalized/ec_immig.csv`; `countries/ec.py` `_immigrants`
turns it into languages with `sources/latam_immig.py`, out of the canton's Spanish.

- **Source**: INEC tabulado `2022_CPV_Migracion.xlsx` (Wayback capture 2025-01-15, gzip; the
  live site is the same 403 as above), `data/raw/ec/`. Table 1.1: "En otro país" per canton
  (425,045 nationally) is the count. Table 7: foreign-born by country of birth (189 columns)
  per province is the mix, applied to each of the province's cantons; no canton-by-country
  table exists (1.2 has parishes, but birthplace only as here/elsewhere in Ecuador/abroad).
  560 people under "Otras Naciones ..." and "Zonas No Especificadas" spread over the
  province's named countries. Note 1 of every sheet: migration was not imputed for the VOPA
  dwellings (~759,000 people), so the foreign-born count is of enumerated people only.
- **Checks** (all pass): 221 cantons, 24 provinces; cantons sum to provinces and provinces to
  425,045; table 7's province totals equal 1.1's exactly; table 7's countries sum to its
  totals; every canton joins `ec_lookup.csv` one to one by folded name.
- **Rule**: the northern pass's (sources/mx.md §2): the census asked everyone about indigenous
  languages, so indigenous-American languages are dropped from origin mixes (`DROP_ROOTS`);
  Spanish-speaking origins (`HISPANIC`, Spain included) stay Spanish whole. Venezuela 232,007,
  Colombia 97,933, Spain 20,609, Peru 14,859, Cuba 10,782 -> Spanish.
- **Result**: 23,114 moved from Spanish: English 11,066, Italian 2,335, Mandarin 1,266, German
  1,175, Portuguese 1,122, French 743. 16,596 dots, 235 rings.
- **Calls**: the 14,448 US-born are drawn on the US mix (English), though many are probably
  children of returning Ecuadorians (mx found that for its US-born; Ecuador's table has no age
  by country); Spain-born kept Spanish for the same reason (also the HISPANIC rule).
