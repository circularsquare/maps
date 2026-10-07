# Honduras (hn): record

Drawn 2026-10-05 (session edd42a8c-amer). 2013 census, no language question: each self-identified
people read as its language (AGENT_BRIEF §2, ethnicity rule). 7,657,684 people on 298 municipios,
every row `derived`. Spanish 98.3%, Miskito 75,864, Garifuna 36,524, Bay Islands English 11,178,
Pech 5,472, Tawahka 2,563, Tol 1,167. 7,654 dots.

Files: `sources/hn_censo.py`, `sources/hn_geo.py`, `taxonomy/hn2013.py`, `taxonomy/tree.d/hn.txt`,
`countries/hn.py`, `data/raw/hn/` (5 REDATAM programs and outputs), `data/normalized/hn.csv`,
`hn_units.csv`, `data/geo/hn/hn_hexes.gpkg`.

## 1. Source

INE, XVII Censo de Población y VI de Vivienda 2013, REDATAM server 181.115.7.199/binhnd, base
CPVHND2013NAC (open). The PERSONA entity has P05 self-identification (Indígena, AfroHondureño,
Negro, Mestizo, Blanco, Otro) and, for the first three, P06 pueblo (Maya-Chortí, Lenca, Miskito,
Nahua, Pech, Tolupán, Tawahka, Garífuna, Negro de habla inglesa, Otro). No language question
(the full VARLIST was read). The base holds the 7,657,684 enumerated; INE's published 8,303,771
and pueblo figures (Lenca 453,672) include an omission adjustment that is not a single factor, so
there is no printed witness to reproduce; FACTORVI weighting was not pursued.

Checks (all pass): codes sum to 7,657,684; 298 municipios rebuild the 18 departments; the derived
GRP equals separate AREABREAK tabulations of raw P06 and P05 on every municipio; municipio sums
equal the national P06 x P05 crosstab.

## 2. Retention: the calls

- **Miskito, Tawahka, Pech, Garífuna, Negro de habla inglesa: 100% on their language.** No
  usable retention share exists. Carlos Palacios, "Pueblos indígenas y negros de Honduras" (UNAH,
  historia.unah.edu.hn dmsdocument 1455) says each still keeps its language, qualitatively.
- **ENDESA-MICS 2019 was looked at and set aside.** Its household and women's files (religiondots'
  raw copy) ask the respondent's "idioma nativo" (Spanish, English, Miskito, Garífuna only) beside
  the head's ethnic group. Women 15-49: Miskito households 415 of 682 Miskito (Gracias a Dios
  407/599); Garífuna 27/311 (Colón 18/115, Atlántida 0/55); Negro inglés 2/134 (Bay Islands 2/101).
  "Negro de habla inglesa" is defined by speaking English, and Garífuna villages of Atlántida
  speak Garífuna; the item looks defaulted to Spanish by interviewers. A reviewer could instead
  apply its Miskito share (~61%) and its Garífuna share (~9%).
- **Tolupán: Tol in Orica (900) and Marale (267) only**, the Montaña de la Flor, where Palacios
  places the language's stronghold; the Yoro Tolupán (most of 18,411) on Spanish. No speaker count
  was found.
- **Lenca, Nahua, Maya-Chortí, Otro pueblo: Spanish.** Lenca extinct by ~1900 (Palacios after
  Herranz); Nahua "no conservan su lengua"; Chortí speakers in Honduras "muy pocos", mostly from
  Guatemala.
- **Bay Islanders who said white or mestizo are on Spanish** (Roatán 19,350 mestizo, 4,507 white;
  Utila 2,906 / 496). Many speak Bay Islands English; nothing counts them. The map under-draws
  English on the islands.

## 3. Mapping and nodes

New: `jicaquean` root (Jicaquean jica1245, colour 0.62 0.15 300) with `jicaquean.tol` (toll1241);
`misumalpan.tawahka` (a Mayangna variety in Glottolog; own node since the census names it);
`chibchan.pech` (pech1241); `creole.english_based.bay_islands` (not separate in Glottolog).
Miskito and Garífuna reuse ni.txt / gt.txt nodes.

## 4. Geography

COD-AB admin2 (2016), 298 polygons, from religiondots' raw download (read-only; religiondots draws
Honduras by department). **Joined on name within department, not code**: COD numbers municipios
alphabetically in Colón, Gracias a Dios and Santa Bárbara, so 24 codes point at a different
municipio (INE 0903 Juan Francisco Bulnes is COD HN0904). Four spellings pinned. Kontur hexes:
log r = 0.993 against 0.183 shuffled, 0 of 298 outside a factor of 3.

## 5. Room for improvement

Any source with an indigenous-language question by municipio (a future census, or a survey with a
credible native-language item) would replace the 100% retention. Bay Islands English for white
and mestizo islanders needs a source.

## 6. Immigrant languages (2026-10-05, session edd42a8c-latn)

Out of the Spanish remainder, `derived`: Spanish 7,524,916 -> 7,521,826. English 1,826, German
194, Mandarin 166, Portuguese, Italian, French, Belize Kriol, Japanese and smaller. 7,652 dots,
55 rings. Method and retention rule: `sources/mx.md`, "Immigrant and settler languages" §2
(`sources/latam_immig.py`).

Files: `sources/hn_imm.py`, `data/raw/hn/hn_nat_pais.htm`, `hn_mun_paisx.htm` (+ programs),
`data/normalized/hn_imm.csv`; `countries/hn.py`, `taxonomy/tree.d/hn.txt` (borrowed-node block).

- P08C_PAIS, country of birth (INE's codes, printed with names): 33,878 foreign-born, of whom
  El Salvador 6,833, Nicaragua 6,227, Guatemala 4,457 and the other Spanish-speaking origins stay
  Spanish; regional remainders and "Ignorado" (1,176) too. 47 non-Spanish-speaking countries:
  US 7,483, China 362, Belize 304, Canada 272, Germany 237, Italy 193, Brazil 174. Checks pass:
  298 municipios; per country, municipio sums equal the national frequency; every person once.
- **US-born under 18 are Spanish**, as in Mexico: 5,339 of the 7,483 (71%), children of returning
  Honduran families. The 2,144 adults take the US mix with TeO2 retention.
- Bay Islanders who said white or mestizo are still on Spanish (§2); this pass does not change
  that, and the US-born on the islands are the only new English there.
