# Venezuela: the record

**Drawn 2026-10-05** (agent d9e44929-ve2), as a model: Anita's ruling on ask 007 (2026-10-05),
option A, "because it names more languages". The 2011 census's pueblo for every indigenous
person per parroquia, with the share who speak their pueblo's language scaled to INE's published
state shares. 27,212,694 people drawn, 1,111 units, 41 languages; 466,498 drawn as speakers of
an indigenous language, every one of them `modelled`. Parked 2026-10-04 by d9e44929-ve at
checkpoint B, waiting on that ruling; the first three sections are that session's work.

## What the censuses asked, and what is reachable

**2011 (XIV Censo).** Everyone born in Venezuela was asked whether they belong to an indigenous
pueblo and which; indigenous people were then asked "Que idioma(s) habla?" (their pueblo's
language, Spanish, another, which other) and whether they read and write the indigenous one
(INE, Boletin "La Poblacion indigena de Venezuela. Censo 2011", Oct 2013, p. 1; the variables
IDIOMAMULT, IDIOMAINDI, IDIOMACAST, OTROIDIOMA, CUALOTROID, ALFABETIIN).

- Published language results: national shares only (10.2% only their pueblo's language, 54.1%
  both, 33.1% Spanish only, 1.1% not declared, 1.5% other combinations) and one "speaks their
  language" share for each of eight states (Amazonas 75.2, Anzoategui 17.1, Apure 75.8, Bolivar
  87.3, Delta Amacuro 89.4, Monagas 33.3, Sucre 2.5, Zulia 67.5), in
  `ine.gob.ve/wp-content/uploads/2025/09/POBLACION-INDIGENA-CENSO-2011.pdf` pp. 33-34. No
  pueblo x language table anywhere: not in that PDF, not in INE's 227-page
  `EmpadronamientoIndigena.pdf`, not in `Tabulados_Poblacion_Indigena.xls`, not in the 2013
  bulletin.
- REDATAM: `www.redatam.ine.gob.ve` no longer resolves, **the bare `redatam.ine.gob.ve` does**
  (XAMPP front page; `/redatam/` and `/Censo2011/` frame the portal). Deployment `/vencgibin/`
  serves bases CPV2011 and CPV2011SEGMENTO, full person files, **without** the language
  variables ("Identificador de variable desconocido"). Deployment `/vencgibin2/` (behind
  `/redatameval/`) has a CPV2011SEGMENTO whose dictionary lists all six language variables, but
  every program on it fails: "Al menos uno de los archivos requeridos no fue encontrado:
  G:\redatam_eval\RCenso2011\vencgibin\RpBases\CENSOS\AppCp11Segmento\Base\Ve110001.ptr". So
  the data exist at INE and are not served. Other paths tried: `/vencgibin2011/` (404),
  `/venbin/` (404), `/cgibin/` (2001 only), base names CPV2011IND, CPV2011INDIGENA, INDIGENA2011,
  CPV2011PERSONA, AppCp11Segmento (none). CELADE's redatam.org only links to INE's server.
  **If INE ever serves them, the model should be replaced by the measured answers.**
- INE's old `/documentos/` tree is gone from the live site (redirects to the WordPress home);
  the Wayback Machine has it. The state Sintesis Estadistica zips carry 2001 tables only.

**2001 (XIII Censo).** `/cgibin/` base AppCp01 is live and has PBLOINDI (pueblo) and IDIOMAIN
(speaks an indigenous language) per person, to parroquia. But it holds the ethnic answers of
only the 327,998 indigenous people counted on the general questionnaire (193,860 speak, 108,026
do not, 26,112 no answer). The 178,343 counted by the separate 2001 census of indigenous
communities (506,341 - 327,998, INE's published total less this base) are in the base's
population but coded "No indigena": Amazonas has 70,464 people in the base, 15,490 of them
indigenous, against INE's published 53,748. Nothing tested (dwelling type, tenure, area) picks
them out. INE's 2001 community-census tables (Sintesis Estadistica, "Poblacion en comunidades
del pueblo X por municipio") give sex and age, no language. Ask 007's option B; not used.

## Step 1: the table (checkpoint B)

`python sources/ve_censo.py --fetch`: five REDATAM programs (windows of 200 pueblo codes; the
server refuses a variable over about 250 categories), AREALIST over every parroquia, programs
saved beside their output in `data/raw/ve/`. Writes `data/normalized/ve.csv` and
`ve_pueblos.csv`. Without `--fetch` it re-runs every check from the saved files in seconds.

Checks (numbers from the run):

1. 1,128 parroquias; every window's rows sum to their Total, the windows' Totals agree, they sum
   to 27,227,930 (the 2011 count), every person counted once across windows.
2. 724,592 indigenous people (INE's figure). 79 pueblo codes in the parroquia tables and 79
   labels in the national frequency of CUALINDIGE (second engine path), counts equal in order;
   that pairing is what gives each code its label.
3. Aggregated to state and to INE's 52 published pueblos, equal to INE's own table "22.-Pueblos
   por entidad" (`Tabulados_Poblacion_Indigena.xls`, 2014) in all 1,325 cells.
4. Indigenous per state equal to INE's Cuadro 4 for the ten states it names.

INE's presentation (Cuadro 15, p. 30) prints Yanomami 9,569 and Pume 9,479: the two figures
swapped. Microdata and table 22 both say Yanomami 9,479, Yaruro/Pume 9,569.

## The model (countries/ve.py `model()`)

`python countries/ve.py` prints the per-state table and the national totals.

- **Living-language pueblos** are every pueblo code whose node is a language, plus 999 "Otro"
  (on `americas_other`). **Pueblos whose language is gone** (taxonomy/ve2011.py `EXTINCT`) are
  drawn as Spanish, never as speakers: Glottolog aes 5 or 6, or no entry and generally described
  as extinct. 64,858 people: Kumanagoto 20,876, Añú/Paraujano 20,814, Chaima 13,217, Baré 5,044,
  Waikerí 1,985, Gayón 1,033, and Yabarana 440, Mapoyo 423, Sáliva 344, Timote 228, Ayaman 214,
  Píritu 121, Kaketío 56, Jirajara 34, Arutani 20, Sapé 9. Ask 007 named 63,631 of them; the
  last four aes-5 pueblos (Yabarana, Mapoyo, Sáliba, Arutani, 1,227) fall under the same rule and
  were not in its list.
- **In each of the eight states INE prints a share for**, the speakers are share x the state's
  indigenous people whose pueblo is known (998 "No declarado" left out of both sides, assuming it
  speaks at the same rate), spread over the living-language people at one rate:

  | state | INE share | pueblo known | living-language | speakers | rate |
  |---|---|---|---|---|---|
  | Amazonas | 75.2 | 75,843 | 70,075 | 57,034 | 0.814 |
  | Anzoátegui | 17.1 | 32,832 | 11,793 | 5,614 | 0.476 |
  | Apure | 75.8 | 11,358 | 11,342 | 8,609 | 0.759 |
  | Bolívar | 87.3 | 54,182 | 53,801 | 47,301 | 0.879 |
  | Delta Amacuro | 89.4 | 41,289 | 41,254 | 36,912 | 0.895 |
  | Monagas | 33.3 | 17,397 | 9,678 | 5,793 | 0.599 |
  | Sucre | 2.5 | 21,938 | 16,444 | 548 | 0.033 |
  | Zulia | 67.5 | 439,457 | 418,733 | 296,633 | 0.708 |
  | the other 17 | (national) | 15,060 | 11,378 | 8,052 | 0.708 |

- **The other seventeen states** (15,060 indigenous people, 2.1%) take the national 64.3% turned
  into a rate over living-language people the same way nationally: 0.643 x 709,356 / 644,498 =
  0.708. A residual (national speakers less the eight states') was tried first and came out at
  -2,330: the eight published shares and the national one do not add up exactly (rounding, and
  perhaps a different denominator), so it cannot carry a rate.
- **Tiers.** Speakers and the indigenous non-speakers and extinct-language pueblos drawn as
  Spanish are `modelled` (709,356). Everyone born in Venezuela and not indigenous (25,346,760)
  and everyone born abroad (1,156,578, never asked the pueblo question) are Spanish `derived`,
  spec §3.5. 998 (15,236) is not drawn, in `gap`.
- **Weakness, said in note_public.** One rate per state for every living-language pueblo, so
  Zulia's Wayuu and Yukpa, or Bolívar's Pemon and Kariña, read the same. No published figure
  splits them. In 2001 (general questionnaire only, so not the Amazonas or Delta communities)
  the rate did differ by pueblo, and a future session could test whether 2001's per-pueblo
  ratios improve on the flat rate; I did not, because the 2001 base misses most of the peoples
  that would matter.
- **No witness exists** for the per-pueblo figures. The state totals equal INE's by
  construction; the national total, 466,498 speakers, is 64.3% of 724,592 within the 1.1% "not
  declared" and the rounding.

## Language mapping (taxonomy/ve2011.py, tree.d/ve.txt)

Keyed by CUALINDIGE code; one node per pueblo label except spelling variants and
exonym/endonym pairs (Guajiro/Wayuu, Panare/Eñepa, Makiritare/Yekwana, Yaruro/Pumé, Hoti/Jodi,
Wótüja/Piaroa, Guajibo/Sikwani/Jiwi, Chase/Piapoko, Arawako/Lokono, Kapón/Akawayo, Curripaco/
Kurripako, Kuiva/Cuiba, Sanema/Sanüma, Ñengatú/Yeral), the pairs INE's table 22 also groups.
Amorúa (91) keeps Colombia's node beside the Jivi; Shiriana (251) goes on Brazil's Xiriana
beside Yanomami. Pemón (140) is its own leaf `cariban.pemon` beside the three named Pemon
peoples (Arekuna, Kamarakoto, Taurepang, Brazil's nodes), not on a group node, since it is a
named answer.

New nodes: `arawakan.baniva` (Baniva de Maroa; Glottolog keeps it apart from Brazil's Baniwa
of the Içana, which Colombia's Baniba uses), `arawakan.lokono`, `cariban.pemon`,
`cariban.japreria`, `saliban.mako`, `isolate.jodi` (Hoti: Glottolog has no family). All other
pueblos reuse Brazil's, Colombia's or Argentina's nodes (Kariña on Brazil's Galibi Kali'na,
Matako on Argentina's Wichí, Tunebo on Colombia's U'wa, Kechwa (20) on Quechua). Colours
hand-picked for the six new nodes against their ground neighbours (the fragment's header says
which); Pemon a cyan because everything around it in the Gran Sabana is yellow or green.

## Geography (sources/ve_geo.py)

- COD-AB Venezuela adm3 (INE, valid 2021-02-23, 1,135 parroquias), read inside religiondots'
  zip; its pcode is "VE" + the census code. 1,121 census parroquias join by code. In 13
  municipios the two code lists differ (Amazonas's autonomous municipios gained parroquias after
  2011; Simón Rodríguez in Anzoátegui split; Rojas and Sosa in Barinas gained one each; Anaco,
  Julio César Salas, Francisco Javier Pulgar and the Dependencias Federales differ the other
  way), so there both sides merge to the municipio: 30 census parroquias, 37 polygons, 13 units.
  1,111 units in all, the bijection asserted, the merged list pinned (`MERGED`).
- Kontur VE 2023 read from religiondots' raw folder, re-keyed by centroid. religiondots' own
  `ve_hexes.gpkg` could not be re-keyed as the handoff planned: it holds no hexes in Amazonas or
  Delta Amacuro (religiondots does not draw them). 1,317 hexes (250,719 Kontur people) fall
  outside every unit, the extract's overrun across borders and coasts, and are dropped.
- Witness: log correlation of Kontur against census per unit r = 0.907, against a best of 0.262
  over 500 shuffles **within each state**, so the parroquia pairing carries information beyond
  the state. Kontur/census normalised p10 0.37, median 0.89, p90 1.63, 107 of 1,111 units
  outside a factor of 3 (2023 grid against a 2011 census, with the emigration between; printed,
  not asserted).
- **Known flaw: COD has no polygon for Arapuey** (census VE141001, 12,143 people, 50 indigenous):
  Julio César Salas municipio is only Palmira's 65 km2 in COD, so Arapuey's people are drawn on
  Palmira. The Sur del Lago parroquias of Mérida all read very low against Kontur (Caracciolo
  Parra Olmedo 0.05, Tulio Febres Cordero 0.08-0.10), which looks like COD's Mérida-Zulia line
  sitting south of the real one. Almost all Spanish dots; left.
- Kontur cap: religiondots' five ve rows apply through rdlink. One new block, registered in
  languagedots' kontur_cap.csv as `isolated`: Cubagua island, one hex of 29,067 people where the
  whole parroquia counts 24,416 and no hex within 5 km is populated; lowered to 46.

## Calls someone might reverse

- The aes-5 pueblos beyond ask 007's list (Yabarana, Mapoyo, Sáliba, Arutani) drawn as Spanish,
  by the ask's own rule; they then draw no language dots at all.
- The other seventeen states at the national rate per living-language person (0.708), not 64.3%
  flat and not a residual.
- 998 not drawn; born-abroad drawn as Spanish (spec §3.5), though some speak Portuguese,
  Italian, Arabic or Chinese.
- Thirteen municipios merged rather than splitting polygons by formation law.

## Files

`sources/ve_censo.py`, `sources/ve_geo.py`, `sources/ve.md`, `data/raw/ve/`,
`data/normalized/ve.csv`, `ve_pueblos.csv`, `data/geo/ve/ve_units.csv`, `ve_hexes.gpkg`,
`taxonomy/ve2011.py`, `taxonomy/tree.d/ve.txt`, `countries/ve.py`, `kontur_cap.csv` (one row),
`ask/007-ve.md` (handoff/ve.md was removed by `claim.py done`).

## Immigrant languages (2026-10-05, session edd42a8c-lats)

- **Source**: CPV2011 on INE's REDATAM (`/vencgibin/`), PERSONA.ENCUALPAIS (18 named countries
  + "Otro país" 27,192) by PARROQUI; "Otro país" split at the national composition of its 144
  countries from CODOTROPAI (codes paired with labels by two identical-count frequencies,
  asserted; 338 in unnamed "other countries of X" spread over the rest). `sources/ve_immig.py
  --fetch` -> `data/raw/ve/ve_*pais*.htm`, `data/normalized/ve_immig.csv`. Checks: 1,128
  parroquias, rows sum to Totals, per-country sums equal the national frequency, never above
  ve.csv's born-abroad row. 1,031,103 with a country of the 1,156,578 born-abroad row; the
  ~125,000 with no country stay Spanish.
- **Languages and retention**: `sources/latam_immig.py` (TeO2 regional rate), taken only from
  the `derived` Spanish rows, never the modelled indigenous ones. Colombians 721,791 Spanish.
- **Result**: added 103,981: Portuguese 28,238, Italian 15,376 (+ Italy's regional languages),
  Levantine Arabic 12,141, Mandarin 6,717, English 6,254, Guyanese Creole 4,056. 604 nodes;
  `tree.d/ve.txt` borrowed block regenerated. Scatter 27,174 dots, 267 rings.

## Moved from countries/ve.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- All of these dots disappear when inferred dots are turned off.
