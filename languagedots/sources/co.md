# Colombia: the record

Drawn 2026-10-04 (agent d9e44929-co). 44,131,075 people of the 44,164,417 the 2018 census
counted, 1,122 municipios; 862,420 speakers of their own people's language, 43,268,655 drawn as
Spanish. 119 language nodes. 44,086 dots at 1:1000, 71 rings.

## Source

DANE, Censo Nacional de Población y Vivienda 2018, through DANE's own open REDATAM webserver
over the full person file (base `CNPVBASE4V2`, `https://systema59.dane.gov.co/bincol/`). No
published table gives language by municipio; the server runs any Redatam+SP program.
`python sources/co_cnpv.py --fetch` runs eighteen programs (saved beside their output in
`data/raw/co/*.program.txt`) and writes `data/normalized/co.csv` and `co_pueblos.csv`.

The question: only people who identify with an ethnic group are asked (`PA1_GRP_ETNIC` 1
indigenous, then `PA11_COD_ETNIA`, one of 124 pueblo codes; 2 Rrom; 3 Raizal; 4 Palenquero).
They are asked `PA_HABLA_LENG`, "habla la lengua nativa de su pueblo", and `PB_OTRAS_LENG`,
whether they speak other native languages (a count, never names). So the census names a people,
not a language, and asks about speaking, not first language. Afro-Colombians (grp 5) and
everyone else are not asked.

## Checks (numbers from the run)

1. 1,122 municipios; in all fifteen municipal tables every row sums to its printed total, the
   totals sum to 44,164,417, and across the tables each person is counted exactly once.
2. Per pueblo, municipal speakers summed equal the national pueblo x speaks crosstab (a second
   engine path, `AS FREQUENCY ... BY`) for all 124 codes. 838,356 indigenous speakers.
3. Rrom 1,608, Raizal 19,504, Palenquero 2,952 speakers equal the group x speaks crosstab.
4. 17,988 who speak another, unnamed native language but not their own, and 15,354 with no
   answer, equal their cells of the speaks x other-languages crosstab. Both are `gap`.
5. Geography: COD-AB adm2 (OCHA from DANE's MGN, dated 2018-01-01) has 1,122 polygons and joins
   the census's MUPIO codes one to one both ways. Kontur over the census per municipio, p10
   0.63, median 0.89, p90 1.40 (national ratio 1.176); log r = 0.946 against a best of 0.107
   over 500 shuffles. 18 municipios outside a factor of 3 (Recetor CO85279 at 52, a Kontur
   artefact), but see placement: every zone's count is the census's.

## Server traps (for the next REDATAM country)

- A defined variable may declare at most about 250 categories ("Too many categories"; 1-250
  ran, 1-499 did not), hence five windows of 200 pueblo codes per zone.
- `AREALIST ... UNIVERSE` is "too many parameters"; filter inside a `DEFINE ... AS SWITCH`.
- SWITCH has no `NOT`; `DEFAULT` takes a constant only (a variable came back 0 for everyone);
  the web form strips `<` as a tag, so `<=` silently vanishes. Use ordered INCASEs with `=`
  and `>=`. `ASSIGN x + -199` is a 500; write `x - 199`.
- Higher entities by their entity name where they have no alias: `Clase.UA_CLASE1`, not
  `CLASE.`. `VIVIENDA.` and `PERSONA.` aliases work.
- The text route drops a connection now and then; the fetch retries.

## Calls

- **Each pueblo code is a node.** Embera, Embera Katío, Chamí, Dobidá and Eperara Siapidara
  are leaves under a `chocoan.embera` group; the small Guahibo peoples (Amorúa, Wipiwi,
  Yamalero, Tsiripu, Chiricoa, Mapayerri, Masiguare) each a leaf beside Sikuani. Codes whose
  language Brazil already drew reuse Brazil's node (list in `taxonomy/co2018.py`). Guariquema
  = Guarequena, on Brazil's Warekena. Kichwa on Brazil's Quíchua; Otavaleño its own node.
- **Peoples whose language is no longer spoken are drawn as answered.** 40,777 Zenú (13% of the
  people), 14,039 Pastos, 6,822 Pijao, 4,503 Yanacona, 2,870 Mokaná, 1,665 Muisca, 1,453
  Totoró, 1,023 Quillacinga, 753 Kankuamo and a few hundred more said yes although Glottolog
  marks the language extinct (AES 6) or has no entry. Drawn on their own nodes, as Brazil's
  `unclassified` draws Tupinambá and Pankararú; under the Glottolog family where there is one
  (Muisca, Kankuamo, Tairona Chibchan; Coconuco, Polindara, Ambaló, Quizgo Barbacoan; Dujos =
  Tama Tucanoan; Betoye, Andakíes isolates), else `unclassified`. `note_public` says they are
  probably heritage or revival speakers. Reversing: map those codes to Spanish in co2018.py.
- **Families**: Glottolog's. New roots chibchan, chocoan, barbacoan, guahiboan, saliban,
  pebayaguan (Yagua), plus `creole.spanish_based` (Palenquero). Kakua, Nukak and the census's
  generic "Maku" go under Brazil's `nadahup` (which already holds Puinave, IBGE's call), the
  Colombian Makú-Puinave convention, so "Maku" (24) has one narrowest node; Glottolog makes
  Kakua-Nukak a family. Nasa, Kamëntšá, Cofán, Andoque, Pumé, Tinigua under `isolate`.
  Unsure and tiny: Je'eruriwa (15) under Tucanoan; Judpa (17) its own Naduhup leaf.
- **Remainders**: "Indígenas Ecuador/Perú/Venezuela/México/Brasil/Panamá/Bolivia", "Maya
  (Guatemala)" and 999 "Indígena sin información" (6,060) on `americas_other`, the node us.txt
  added for indigenous languages of the Americas not named (repeated in co.txt, same label).
- **Rrom** on a new `indoeuropean.indoaryan.romani.vlax` leaf, "Vlax Romani (Romanés)": uk.txt
  made `romani` a group, so a named label cannot sit on it. **Raizal** on San Andrés Creole
  under `creole.english_based` (us.txt's group; uk.txt files Caribbean creoles under Germanic
  instead, a split for whoever owns the tree).
- **Spanish = everyone else, `derived`** (spec §3.5): not asked (98% of the country), or asked
  and not a speaker. Second source for the remainder: none checked.
- **Placement, three zones per municipio.** The census counts each person as in a resguardo
  (`VIVIENDA.UVA_ESTATER` 1 and `UVA1_TIPOTER` 1), else in the cabecera (Clase 1), else the
  rest; 67.5% of speakers are in a resguardo, 13.2% in a cabecera, 19.3% elsewhere. Hexes:
  `r` = centroid inside the Agencia Nacional de Tierras' "Resguardo Indígena Formalizado"
  (984 polygons, updated 2026-09-07, CC BY-SA 4.0; `data/raw/co/`); `u` = the municipio's
  densest remaining hexes up to the census's cabecera share; `x` the rest. Each zone gets the
  census's own counts for it. Without this, Arhuaco dots sat in Valledupar's streets; now
  Arhuaco's median dot is at -73.63, 10.52 (Sierra Nevada south slope), Kogui -73.56, 11.02,
  Wayuu -72.29, 11.61 (Alta Guajira), Nasa -76.32, 2.79 (Tierradentro), Zenú -75.47, 9.14.
- **Where ANT's polygons cannot hold the census's resguardo people** (over 400 people/km2 spread
  evenly over the in-polygon hexes), those people go to the `x` zone and the polygon's hexes
  join it: 79 municipios, 399,887 people, mostly the colonial resguardos of Cauca and Nariño
  (Caldono: 23,965 counted in resguardos, 4 hexes in ANT's polygons). Kontur also sees far
  fewer people inside the polygons than the census counts (Cauca 2.7% against 19.1%), so in an
  `r` zone the weight is half Kontur, half even per hex (`countries/co.py::_CoWeighter`).
  The 400 is a judgement (the uniform density over `r` hexes had median 105, p90 740).
- **Kontur cap blocks**: Manaure, Tiquisio, Pinillos and Sahagún are already in
  `kontur_cap.csv` as `unreviewed` (from religiondots' review) and draw as Kontur has them.
  Each now sits in a cabecera zone holding only that town's census count, so the worst case is
  a town's own few dots on its dense block; Manaure's block is 19 km from the town and may put
  Manaure's few cabecera dots in the wrong place. Not acted on.

## Colours

Hand-picked in `tree.d/co.txt` for neighbours: Sierra Nevada (Arhuaco blue 265, Kogui lavender
322, Wiwa violet 295, Kankuamo 280) against Wayuu's olive; Nasa magenta against Namtrik light
green and Totoró green (moved off teal, which sits near Spanish's built colour #2fa8c5); Embera
orange family against Wounaan pale peach; Sikuani yellow against Piapoco green; Inga orange
against Kamëntšá lilac. Spanish's colour is not set here (uk.txt and us.txt set it).

## Files

`sources/co_cnpv.py`, `sources/co_geo.py`, `taxonomy/co2018.py`, `taxonomy/tree.d/co.txt`,
`countries/co.py`, `data/raw/co/` (18 REDATAM outputs with their programs, ANT resguardos),
`data/normalized/co.csv`, `co_pueblos.csv`, `data/geo/co/co_hexes.gpkg`, `co_zones.csv`,
`data/processed/dots_co.geojson`, `rings_co.geojson`.

## Immigrant languages (2026-10-05, session edd42a8c-lats)

- **Source**: the same REDATAM base, PERSONA.PA3_PAIS_NAC (country of birth, asked when
  PA_LUG_NAC = 3, 963,492 people), per municipio (`sources/co_immig.py --fetch` ->
  `data/raw/co/co_pais.htm`, `co_mupio_pais.htm`, `data/normalized/co_immig.csv`). Labels carry
  ISO codes ("862_Venezuela_VE_VEN"). The ~250-category limit: countries with 30+ people kept
  (folded small ones, 1,133 people, split at national shares). Checks: rows sum to Totals,
  1,122 municipios summing to 44,164,417, per-country municipal sums equal the national
  frequency, foreign-born 963,492 (8,942 with no valid country spread over each municipio's
  known countries; 15 people in municipios with only those lost).
- **Venezuela 837,900 (87%)**: Spanish (new uncited override in origin_mix §2b, their home mix
  had 1.1% Wayuu). Then US 20,124, Ecuador 18,111, Spain 14,954.
- **Languages and retention**: `sources/latam_immig.py`, as Argentina (TeO2 regional rate;
  Spanish shares stay Spanish). Per municipio, shared over its zones (resguardo, cabecera,
  rural) in proportion to each zone's Spanish remainder.
- **Result**: added 37,750 out of the Spanish remainder: English 16,616, Portuguese 3,605,
  French 2,297, German 1,941, Italian 1,603, Mandarin 753, and the origins' tails (788 nodes;
  `tree.d/co.txt` borrowed block regenerated). Scatter 44,071 dots, 364 rings.
- Not drawn: no long-settled non-indigenous community with a count (San Andrés Creole is
  already measured, as Raizal).
