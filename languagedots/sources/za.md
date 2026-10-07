# South Africa: 2011 census, home language by small area

Drawn 2026-10-04 (session d9e44929-za). 50,925,657 people with a language, 84,783 small areas
with people (84,907 in all), 13 nodes from 13 drawn census categories. 50,918 dots at 1:1000,
no rings.

## Source

- **Table.** Statistics South Africa, Census 2011, persons by language per small area (SAL),
  from the Census 2011 Community Profiles DVD, as extracted by Adrian Frith:
  `https://stuff.adrianfrith.com/sal-lang.csv`, published with his linguistic-diversity code
  (github.com/afrith/language-diversity-index). Fetched into `data/raw/za/`.
- **Polygons.** Stats SA's own 2011 Small Area Layer, `SAL_APRI` (DBF last updated 2013-04-18),
  from the same host's `Census2011-GIS/` folder. 84,907 features.
- **Terms.** Stats SA: use freely with acknowledgement of Stats SA as the source of the basic
  data and that the processing is our own; no sale. Frith waives his own rights (CC0 on the
  sibling repo). Nothing gated, no account.
- **Question.** Questionnaire A, P-06 LANGUAGE: "Which two languages does (name) speak most
  often in this household?", codes 01 Afrikaans to 12 Xitsonga with 09 Sign language
  (`data/raw/za/Census2011_q_A.pdf`, page 2). The small-area table is the first of the two,
  which Stats SA tabulates as "first language". A home-use question, not mother tongue.
- **Grain.** Small area, 600 people on average with a language. The finest unit Stats SA
  publishes.

## Why 2011 and not the 2022 census

The 2022 census asked the same kind of question, but below the nine provinces its answers are
only on SuperWEB2 (`superweb.statssa.gov.za`), which needs an account and answered our fetch
with an Incapsula bot wall. Searched and empty for 2022 language below province:

- Provincial profiles (Report 03-01-7x 2022): province table only (KZN's Table 2.11 checked).
- Census 2022 Municipal fact sheet (03-01-82) and Provinces at a Glance: no language table.
- The census dashboard's API (`disseminationapi-a2f6fff8f7a3f3ff.z01.azurefd.net/api/`, found in
  census.statssa.gov.za's JS bundle): `DsLanguages/getDsLanguagesReport/<timeseries>/<level>/<code>`
  returns language for 2011 (timeseries 2) at national, province and municipality, and `[]` for
  2022 (timeseries 1) at every level.
- The Census 2022 Special District Layer (police, education, health districts; Stats SA post of
  2025-08-05, via Wayback) is also SuperWEB-only; only its boundary zips are open.
- The 2022 10% sample (DataFirst catalogue 982, CC BY) needs a DataFirst login and reaches only
  the municipality. Anita has a DataFirst account (religiondots used catalogue 611); if she
  wants 2022, that file would give 213 municipalities against 84,907 small areas here.

2011 small areas over 2022 municipalities is the call: on a language map the edges are inside
cities (Cape Town's Xhosa townships, Afrikaans Cape Flats and English southern suburbs are one
municipality), and a municipality-grain map would spread each city's languages evenly over it.

## Checks (`python sources/za_c11.py`)

All pass.

1. **The same 84,907 SAL codes** in `sal-lang.csv` and the layer, both ways.
2. **A second table of the same census, per SAL:** Frith's `sal-pop.csv` (total population from
   the same DVD, github.com/afrith/sal-population-estimates). National 51,734,562 against
   51,764,951 (-0.06%); per SAL |difference| median 2, p99 10, max 17. The DVD's small-area cells
   are perturbed by Stats SA for confidentiality, which is this size. The bar set before reading
   (10 people or 3%) failed on 51 SALs, all at most 17 people out; widened to 20 people, which
   still stops a shifted column or a wrong SAL.
3. **Stats SA's published 2011 tables** (the dashboard API above, cached in `data/raw/za/api2011/`):
   - national, 12 categories (the API drops Sign language): within 0.31% (isiNdebele 1,086,813
     against 1,090,223), every one below the published figure.
   - 9 provinces x 12: the 52 cells of 100,000 or more within 0.52%. **The small areas run short
     on thin languages, never long:** Sepedi in the Eastern Cape 13,335 against 14,299 (-6.7%),
     isiNdebele there -4.6%, Sepedi in KwaZulu-Natal -4.1%. All 24 cells past the first bar
     (0.5% everywhere) were a language far from home and all were short. That is the
     perturbation zeroing cells of one or two speakers. The bar is now 1% at 100,000+, 10% from
     10,000, and never more than 0.5% above the published figure (3 of 88 cells are above, at
     most +0.12%).
   - 199 municipalities on 2011 codes: totals within -0.29% to +0.19%; the 117 cells of 100,000+
     within 0.14%. **The API's 2011 municipal figures are on the 2011 municipalities**, under
     whichever codes survived into 2016: EC101 is old Camdeboo alone (a 2016-boundary link read
     +57.8%), LIM344 is Makhado before Collins Chabane was cut out. So the check groups the small
     areas by the layer's own 2011 `MN_MDB_C`. The 13 codes new in 2016 carry a different label
     set (Sign language and a `KHOI/ NAMA AND SAN LANGUAGES` row where the others have OTHER;
     2,014 people in Enoch Mgijima, i.e. Komani, which is the Other cell mislabelled) and are not
     used; EC129 returns nothing.

## Not drawn

`Not applicable`, 808,905 (1.56%): in 6,254 small areas, 53% of them in small areas that are at
least half not-applicable. The largest: Johannesburg Prison (8,586 of 9,741), Castle Rock SP in
Cape Town (5,469 of 5,481), Marapong SP2 at Lephalale (5,295 of 5,295), Grootvlei Prison, Wildebeesfontein and Modderfontein mine hostels. People in collective
quarters, recorded without a language. `Unspecified` is 0 everywhere in this file. Both are in
`gap`.

## Labels (`taxonomy/za2011.py`, `taxonomy/tree.d/za.txt`)

Every census category has its own node except the two not drawn.

| census | node | note |
|---|---|---|
| IsiZulu, IsiXhosa, SiSwati, IsiNdebele | nguni.zulu, .xhosa, .siswati, .ndebele_za | uk.txt's nodes; IsiNdebele is Southern (South African) Ndebele |
| Sepedi | sotho_tswana.sepedi | Stats SA's name for Northern Sotho as a whole |
| Sesotho, Setswana | sotho_tswana.sesotho, .setswana | new group, Sotho-Tswana (S.30) |
| Xitsonga | tswa_ronga.tsonga | new group, Tswa-Ronga (S.50) |
| Tshivenda | bantu.venda | Venda is in neither group in Glottolog |
| Afrikaans | germanic.continental.afrikaans | us.txt's node |
| English | germanic.english | |
| Sign language | signlanguage (root) | code 09 names no sign language |
| Other | other | everything outside the twelve codes, unnamed (826,707) |

Groups checked against Glottolog: Sotho-Tswana (S.30) soth1248, Nguni (S.40) ngun1276,
Tswa-Ronga (S.50) tswa1254, Venda vend1245. No glottocodes stored.

Two things another session should know:

- **Setswana has two nodes.** uk.txt hangs Setswana directly under Bantu
  (`nigercongo.bantu.setswana`); here it is under Sotho-Tswana with Sepedi and Sesotho, the
  conventional level. The fix is one line in `taxonomy/uk2021.py` and dropping uk.txt's line.
- **The Nguni languages' colours cannot be set from this fragment.** uk.txt defines them first
  without a colour, and `build.py` keeps the first definition and silently ignores a colour on a
  repeat. They keep their generated oranges. In Gauteng, Zulu (0.76 0.13 52), Ndebele
  (0.78 0.13 76) and Sesotho (0.86 0.11 95, set here) read as three similar yellow-oranges, and
  Swati's generated salmon (0.72 0.13 16) sits near Sepedi's crimson in Mpumalanga. Worth a hand
  pick for Ndebele and Swati in `HAND` or in uk.txt.

Colours set here: Limpopo's three are pulled far apart (Sepedi crimson 355, Xitsonga teal 165,
Tshivenda violet 300), Sesotho pale yellow against Setswana olive (130) across the Free State
and North West. Looked at in a quick render: Cape Town's Xhosa, Afrikaans and English areas and
Limpopo's three read clearly.

## Geography (`python sources/za_geo.py` -> `data/geo/za/za_hexes.gpkg`)

- **Kontur hexes cut to the small areas**, the playbook's Malta rule: a small area is mostly
  smaller than a 0.74 km² hex, so a centroid join would leave thousands with none. 808,704
  hex x small-area pairs; 798,781 pieces kept, 9,930 slivers under 500 m² dropped. Each hex's
  people are shared over its pieces by area, divided by the whole hex (the playbook's rule where
  a hex's remainder is another country's land; it also keeps every piece at or below Kontur's
  density, so the Kontur cap check stays valid; the two registered Johannesburg blocks are
  `real`).
- **The small areas do not tile the country, and that is right for 2011.** They cover 1,149,551
  km² of 1,220,813. Stats SA dissolved them from the 2011 enumeration areas
  (`SAL_APRI.XML`: `Dissolve EA_SA_2011_080413_MunicChange ... SAL_CODE1`) and left the frame's
  Vacant EAs (9,097, 109,071 km²) out. 4.45M of Kontur's 60.5M people (7.35%) fall outside every
  small area: Intsika Yethu (EC135) has 2,107 km² of Vacant EAs against a 1,916 km² gap and 38%
  of its Kontur people there. Kontur is 2023 and models built-up land; the 2011 census put
  nobody there, so no dot goes there. (Checked with the EA attribute table, `EA_SA_2011.dbf`,
  from the same host; not kept.)
- **7 small areas touch no Kontur hex** (531 people with a language): their own polygon at pop 0,
  so equal shares over the small area.
- **Kontur against the census**, log correlation and the best of 200 shuffles: small areas 0.410
  against 0.010 (noisy at 600 people, as expected), main places (13,869) 0.922 against 0.023,
  2011 municipalities (234) 0.974 against 0.180.
- **Grid floor warning, accepted here**: median 3 pieces per small area, 13,316 in one piece.
  The warning is about weighting inside big units with too few cells; here the pieces are the
  small areas' own shape and the unit is already finer than a hex. Uniform placement would put a
  rural small area's dots over its empty grazing land. `sources/geo_checks.csv` is
  religiondots' file, so the `accepted` row is not written; the scatter prints the warning.
- Water: 2,068 pieces clipped (0.01% of their area was sea); 15 units left unclipped by
  `water.py`'s over-95% rule.

## Calls someone might reverse

- 2011 small areas over 2022 municipalities (the 10% sample, needs Anita's DataFirst login).
- `Sign language` on the root rather than a South African Sign Language node.
- Sotho-Tswana and Tswa-Ronga as groups under Bantu.
- `note_public` says the vintage and the first-of-two question, nothing else.
