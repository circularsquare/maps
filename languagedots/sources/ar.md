# Argentina: the record

Drawn 2026-10-04 (agent d9e44929-ar). 45,736,662 people drawn on 527 departamentos; 382,697
speakers of an indigenous language, the rest drawn as Spanish.

## Source

INDEC, Censo Nacional de Población, Hogares y Viviendas 2022, tabulated on INDEC's REDATAM
server (redatam.indec.gob.ar, base CPV2022, Redatam 7, open, no login). INDEC asks that output
be cited as "elaboración propia con base en datos del INDEC. Censo Nacional de Población,
Hogares y Viviendas 2022, procesado con Redatam 7".

- The pueblo questions live in a separate database, **CPV2022Afro** (menu "Pueblos
  originarios, afrodescendientes e identidad de género", ITEM=PROGVIVAFRO), private dwellings
  only, whose geography stops at PROV and DPTO. The main database (CPV2022.dic) goes to fracción
  and radio but has no pueblo variable. So the floor is the departamento.
- The question: P22 self-identifies as indigenous or descended from indigenous peoples; if yes,
  P23 the pueblo (coded into about 75 categories) and P24 "¿Habla y/o entiende la lengua de ese
  pueblo indígena u originario?" (sí / no / ignorado). Not asked in collective dwellings or of
  people on the street. The language is implied by the pueblo, as in Colombia.
- Pueblo codes and labels: `redarg/CENSOS/CPV2022/Docs/CPV2022-afro.xlsx`, copied into
  `sources/ar_censo.py`.

`python sources/ar_censo.py --fetch` runs four Redatam programs (saved beside their output in
`data/raw/ar/`) and writes `data/normalized/ar.csv` and `ar_pueblos.csv`.

## Checks (numbers from the run)

1. Every departamento's categories sum to its printed Total; the Afro database totals
   45,618,787, the private-dwelling population.
2. Per departamento, the Afro database's total equals the main database's person count, all
   527: two databases of one census agree.
3. Private 45,618,787 + collective and street 273,498 = 45,892,285, INDEC's 2022 definitive
   total. The only departamento with no private dwellings is 94028 Antártida Argentina (81
   people at the bases), not drawn.
4. Per pueblo label, speakers summed over departamentos equal the national P23 x P24 crosstab
   exactly (69 labels with speakers); speakers 382,697, non-speakers 768,491, ignorado 155,542
   equal its column totals. Self-identified indigenous 1,306,730.

## Calls

- **Pueblo to language**, one node per pueblo as in Colombia, families from Glottolog. The full
  reasoning is in `taxonomy/ar2022.py`'s docstring. In short: INDEC's own subdivisions (14x
  Diaguita, 24x Kolla, 28x Mapuche) draw on their pueblo's node, except Huiliche (Huilliche,
  its own language), Diaguita Quechua and Kolla Quechua (under Quechuan), Ava Guaraní
  (Chiriguano) and Tupí Guaraní. Aoniken merges with Tehuelche (the same people's own name).
  Kolla sits under Quechuan, labelled "Kolla". Guaycurú (a family name) on the Guaicuruan group
  node; "Sin información" (82,156 speakers whose pueblo was not recorded) on `americas_other`.
- **Guaraní (171) split by province**: Salta and Jujuy (9,007) drawn as Ava Guaraní
  (Chiriguano), everywhere else (38,563, mostly Buenos Aires, the city, Corrientes, Misiones)
  as Paraguayan Guarani. A place-dependent label, like India's Pahari.
- **Extinct or unattested languages drawn as recorded**: Diaguita 14,947, Omaguaca 4,954,
  Tonokoté 4,834, Huarpe 2,034, Atacama (Kunza) 1,828, Tehuelche 1,598, Comechingón 1,369 and
  smaller ones. Probably heritage or revival knowledge, or another language; in Santiago del
  Estero very likely Santiago del Estero Quichua. Said in note_public, not redrawn.
- **Spanish** = not indigenous + indigenous non-speakers + collective dwellings and street,
  tier `derived`. **Not drawn**: 155,542 indigenous people with P24 ignorado (`gap`), as
  Colombia's "no informa".
- **Second source for the Spanish remainder**: none checked. The 2010 census asked pueblo only;
  INDEC's ECPI 2004-05 survey asked indigenous people's languages but not everyone's. Not
  corroborated.
- **Geography**: IGN's `departamento` WFS layer, joined on INDEC's five-digit code, asserted
  both ways (only 94021 Islas del Atlántico Sur unmatched, not enumerated). religiondots'
  Argentina layer is six survey regions, so not reusable; its Kontur file was copied, not
  re-downloaded. Hexes by plain Kontur population: nothing below the departamento says where
  indigenous people live. Kontur check numbers are in `sources/ar_geo.py` (Corrientes under-
  counted evenly at 0.25; Tolhuin 5.9x, phantom forest population). Both are Kontur's, not the
  join's.
- **Colours**: new roots Matacoan (cyan, 195), Chonan (purple, 310), Charruan (225), Huarpean
  (120). Kunza and Comechingón re-picked away from Kolla and Tonokoté. Not changed, not mine:
  Spanish (us.txt) and Quechua (br.txt) are close (OKLab 0.073), and they share NOA and the
  Bolivian and Peruvian migrant districts of Buenos Aires.

## Scatter

45,717 dots, 35 rings; 19,662 people (0.04%) in languages under one dot nationally draw no dot.
Spot check of median dot positions: Wichí -62.3, -23.2 (Chaco salteño and Formosa), Qom -59.5,
-26.2, Kolla -65.4, -22.9 (Jujuy), Mapuche -68.8, -39.0 (Neuquén), Mbya -54.8, -26.8
(Misiones), Ava Guaraní -63.8, -23.1 (Orán and San Martín).

## Files

`sources/ar_censo.py`, `sources/ar_geo.py`, `taxonomy/ar2022.py`, `taxonomy/tree.d/ar.txt`,
`countries/ar.py`, `data/raw/ar/` (4 tables + programs, 1.5 MB), `data/normalized/ar.csv`,
`ar_pueblos.csv`, `data/geo/ar/ar_departamentos.gpkg`, `ar_hexes.gpkg`,
`data/processed/dots_ar.geojson`, `rings_ar.geojson`.

## Immigrant languages (2026-10-05, session edd42a8c-lats)

Anita's priority: non-indigenous minority languages in Latin America. Before this, everyone not
an indigenous speaker was Spanish.

- **Source**: the same census, main database (CPV2022, private dwellings), PERSONA.PAISNAC
  country of birth per departamento (`sources/ar_immig.py --fetch` -> `data/raw/ar/
  ar_dpto_paisnac.htm`, `data/normalized/ar_immig.csv`). AREALIST and per-area FREQUENCY refuse
  PAISNAC ("Too many categories"), so a recode keeps the 84 countries with 100+ people and folds
  the rest into one category per continent, split back at national shares. Codes are paired with
  labels by two national frequencies of identical counts (asserted). Checks: each departamento's
  categories sum to its Total; per category the departamentos sum to the national frequency (90
  categories); the 527 departamentos are ar.csv's; Totals = 45,618,787. Foreign-born 1,933,463;
  "Ignorado" 179,239 spread over each departamento's known countries.
- **Languages**: `origin_mix.mix(iso, "ar")` (Paraguay 63% Guarani / 35% Spanish / 2% Portuguese,
  Bolivia 79/14 Quechua/8 Aymara, Peru 85/14/2, Italy at its regional mix, Brazil Portuguese...).
  The Spanish share stays Spanish. **Retention**: the rest kept at France's TeO2 rate for the
  origin's region (`sources/latam_immig.py`, `fr_build.TEO2`: Americas 77.4%, Spain-Italy 62.4%,
  other EU 66.7%, China 66.3%, Middle East 74.5%); the remainder Spanish. No Argentine survey
  gives home-language retention by birth country.
- **Double counting**: Paraguayan Guarani, Quechua and Aymara are also counted among the
  self-identified indigenous speakers (38,563 Guarani outside the NOA, mostly in Buenos Aires).
  Per departamento only the estimate above the measured count is added: 69,937 of the estimate
  dropped this way. Measured rows untouched.
- **Result**: estimate 532,712, added 462,775, taken out of the Spanish remainder (never below
  zero; nothing capped). Paraguayan Guarani +239,589 (now 278,152; Venezuelans' 1.1% Wayuu since moved to Spanish by a new origin_mix override), Portuguese 56,483, Italian
  37,645 (+ Neapolitan, Sicilian... ~12,000), Quechua +32,917, Aymara +14,925, English 13,806,
  Mandarin 9,883, Korean 4,392, German 4,201. Spanish 45.35M -> 44.89M. 621 nodes (origin mixes'
  tails); `tree.d/ar.txt` regenerated with `origin_mix.py --fragment ar` (one label fixed: Wu).
  Scatter 45,693 dots, 329 rings. Placement by plain population inside the departamento.
- **Not drawn, no count exists**: Argentine-born children of immigrants (Spanish). Welsh in
  Chubut was listed here too until 2026-10-06; now drawn, see "Welsh in Chubut" below.
- Room for improvement (immigrants): placement by the radio-level foreign-born count (PAISNAC goes to radio
  in the main base); a survey of Paraguayans in Buenos Aires by home language for a local
  retention rate.

## Welsh in Chubut (2026-10-06, session 5d7dac7e-wl)

Anita asked for Patagonian Welsh (Y Wladfa). No census or survey counts it (the 2022 census asks
language only of self-identified indigenous people), so it is drawn on the estimate route (ask
019, as `sources/eg.md`): rows `modelled`, `countries/ar.py` `_welsh()`.

**The figures found** (read 2026-10-06):

| figure | what it counts | source |
|---:|---|---|
| about 1,500 | "Patagonian Welsh speakers", Chubut province's own estimate | Western Mail, 27 Dec 2004, as cited by Wikipedia "Y Wladfa" (ref. 14); the article itself not read |
| up to 5,000 | "people in Chubut today who still speak Welsh" | Prof. E. Wyn James (Cardiff), BBC News "Viewpoint: The Argentines who speak Welsh", 16 Oct 2014 (read via Wayback) |
| "several thousand" | "some knowledge of Welsh with varying degrees of fluency" | Ó Néill 2005: 429, via Sleeper 2015 (UCSB thesis, Contact effects on VOT in Patagonian Welsh, p. 7) |
| 15% of Gaiman | spoke Welsh regularly, 1973 | Jones 1984: 240, via Sleeper 2015 |
| 1,411 (2019), 1,106 (2024-25) | learners on Welsh courses, mostly children | British Council, Welsh Language Project in Chubut annual report 2019 p. 8 (via Wayback; the live site times out); Wikipedia for 2024-25 |
| none | "No reliable figures are available for the number of Welsh speakers in the Wladfa" | same report, p. 5 |

Not read: Johnson 2009 (IJSL 195, "How green is their valley?", a 2004 questionnaire of 369
people in seven places; ResearchGate and Cardiff's ORCA both 403) and Ethnologue's Argentina
figure (paywalled). Neither would give a count; Johnson's is a vitality survey.

**Call: 1,500.** No source separates first-language or home speakers from learners. The 5,000
explicitly takes in anyone who speaks Welsh, and since the late 1990s a large share of those are
course learners (1,400 a year on courses by 2019, most of them children in the bilingual
schools). The province's 1,500 is the low end and the one least swollen by learners, so it is
the nearest to home use on offer; it is still a speaker count, not a first-language count, and
may well be high for people who speak Welsh at home. Said in `how` as "about 1,500 speakers".

**Split by a stated rule**: the Welsh Language Project's 2019 class count per area (report p. 8):
Gaiman incl. Dolavon 68, Trelew (incl. Puerto Madryn and Comodoro) 23, the Andes (Esquel and
Trevelin) 23, of 114. Classes rather than learners, because learner numbers in Gaiman are pushed
up by compulsory Welsh in the first three years of Coleg Camwy (423 teenagers). Areas to
departamentos: Gaiman -> 26042 Gaiman (holds Dolavon) 894.7; Trelew -> 26077 Rawson (holds
Trelew and Rawson) 302.6; the Andes -> 26035 Futaleufú (Esquel, Trevelin) 302.6. Trelew's
classes in Puerto Madryn (Biedma) and Comodoro (Escalante) are not given their own share: the
historic community is the lower Chubut valley, and the report does not count them apart.
Each departamento's Spanish row loses exactly what Welsh gains (asserted at least 10x larger),
so every total is unchanged (check_country: 45,736,662). Inside each departamento the dots
follow plain Kontur population.

**Node**: `indoeuropean.celtic.welsh`, uk.txt's, so the colour matches Wales; repeated in
`tree.d/ar.txt` above the origin_mix block (without colour) so a regenerated fragment keeps it.

**Scatter**: at 1:1000 the 1,500 make one dot, in Gaiman (-65.13, -43.29); Rawson's and
Futaleufú's 303 each round to none. 45,691 dots, 334 rings.

Room for improvement: Johnson's 2004 questionnaire tables (fluency by place); any Chubut survey
asking home language.
