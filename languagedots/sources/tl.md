# Timor-Leste: Census 2015, mother tongue

Drawn 2026-10-05. 1,172,926 people on 13 municipalities, 38 nodes, 1,155 dots and 11 rings.
Modules: `sources/tl_census.py`, `taxonomy/tl2015.py`, `taxonomy/tree.d/tl.txt`, `countries/tl.py`.
Placement is religiondots' `data/geo/tl/tl_hexes.gpkg` (Kontur 400 m, COD-AB p-codes), read-only.

## 1. Which census, and why 2015

INETL (then the Direcção-Geral de Estatística) asked mother tongue in 2010, 2015 and 2022.

| census | question | published geography |
|---|---|---|
| 2010 | mother tongue, one answer | district table (Volume 2 table 13); per suco only as an unlabelled % bar chart in 442 `Sensus Fo Fila Fali` PDFs |
| 2015 | mother tongue, one answer | **municipality, workbook** (Volume 2, `4_2015-V2-Language.xls`, table 12) |
| 2022 | "What languages did <Name> learn as a child?", **one or two** (item E55, main report p.192) | municipality only as a bar-chart image in each `em Números 2022` volume; no table, national or otherwise |

**2015 is drawn.** It is a single answer per person, in a workbook, at the finest grain any of the
three publishes as numbers. 2022 counts mentions: Ermera's chart prints 117,108 Tetun Prasa and
63,481 Mambai answers for 137,750 people, Aileu's 48,891 and 30,100 for 54,324
(table 4.03). It also cannot be reconciled for Atauro (no national language table exists in the
2022 main report) and it would mean hand-reading thirteen chart images. AGENT_BRIEF §2: a
single-answer table from the same office beats a multi-answer one.

**Nothing finer than the municipality exists as numbers.** The 2015 suco volume has no language.
The 2010 suco reports (mof.gov.tl `download-suco-reports`, one PDF per suco, now only in the
Wayback Machine) have a "Lia inan" page, but it is a percentage bar chart without labels; reading
442 of them off vector bar lengths is possible in principle and was not attempted. Williams-van
Klinken and Williams (2015, Dili Institute of Technology, "Mapping the mother tongue in Timor-Leste:
who spoke what where in 2010", tetundit.tl) did map them per suco, as images.

## 2. The table and its checks

`4_2015-V2-Language.xls`, sheet 2.12: "Table 12 Population by mother tongue, age, urban/rural
location and Municipality". 38 categories: 32 Timorese names, Portuguese, Indonesian, English,
Malay, Chinese, Other. 1,179,654 people; the private-household population is 1,178,340 (Table 1.a),
and per municipality the two agree within 0.5% (Manatuto +0.47% is the widest). The 3,989 people of
the 1,183,643 counted who are not in the table are outside private households.

All asserted in `sources/tl_census.py`:

1. Per municipality the categories sum to the printed total; per category the municipalities sum to
   the national figure; urban + rural = total. All exact.
2. **Second table of the same census.** Sheets 2.13 and 2.13a-m (mother tongue by five-year age, one
   per municipality, compiled separately) agree with table 12 in all 545 cells.
3. Table 1.a's private-household population, as above.
4. **Second census.** 2010 Volume 2 table 13 (district by mother tongue, parsed off the PDF, its own
   districts summing to its 1,053,971): **21 languages** over 2,000 speakers in both censuses have
   their largest municipality in the same place in 2010 and 2015. This is the join check that
   population cannot do: Manatuto and Manufahi have near-equal populations (46,588 and 53,685; Kontur
   53,905 and 53,269), so a swapped column would pass any population test, but Galoli/Idate and
   Lakalei cannot be swapped without failing this.
5. Glottolog's point for seven regional languages (Fataluku, Baikenu, Tokodede, Galoli, Lakalei,
   Waima'a, Idaté) falls in the p-coded unit where 2015 has most of their speakers. Naueti's point
   lands a few km over the line in Lautém's Iliomar while 13,898 of its 16,507 speakers are in
   Viqueque, so it is left out of the check; Mambae's and Bunak's sit on borders.

The join is an identity on p-codes (`MUNIS` in `sources/tl_census.py`, the same codes religiondots'
`sources/tl.py` uses). Nothing joins on a name.

## 3. Atauro

In 2015 Atauro was an administrative post of Dili; it became a municipality for 2022, and
religiondots' hex layer splits it out as `TL0604`. Here the two are folded back into Dili (`place_unit`).

But Dili's column holds the island's own languages, and drawing them across Dili by population would
put almost all of them in the city. So `countries/tl.py` places them inside Dili by where they are
spoken, a within-unit placement only (§4.4 of the brief; counts unchanged):

* Rahesuk 2,287, Raklungu 1,808, Resuk 3,120, Adabe 65, Atauran 44, Dadu'a 35: **7,359** people,
  on the island's hexes only.
* Atauro had **9,274** people in 2015 (Volume 2 table 3), about 9,220 in this table's universe, so
  about **1,861** of Dili's other-language speakers live on the island. Every other language in Dili
  is weighted on the mainland plus the island's hexes scaled so the island receives that many
  (0.69% of Dili's other languages). In the build, the island gets 6 Atauro-language dots and 2
  Tetun Prasa dots.

Tetundit (2010) says the Atauro languages are "spoken only on Atauro island"; 2010 table 13 has
4,845 of Rahesuk, Raklungu and Resuk in Dili and 81 elsewhere. Some will live in Dili city; the
placement ignores that.

## 4. English in Ermera and Manufahi, withheld

Table 12 has **7,271** people with English as their mother tongue, and **4,774** of them are in
Ermera and **1,954** in Manufahi; nationally 6,734 of the 7,271 are rural. They are not English speakers:

* their age profile is the municipality's own: Ermera's English speakers are 14.3% aged 0-4 against
  13.8% of all Ermerans, and 0.8% are over 75 against 1.3%;
* the 2010 census has 47 in Ermera and 23 in Manufahi (773 nationally, 600 of them in Dili);
* the 2022 charts for both municipalities list no English at all.

So the 6,728 are some Timorese answer mis-keyed as English, and which one cannot be told: Ermera is
Mambai and Kemak, Manufahi Mambai, Tetun Terik and Lakalei, and Aileu, the most Mambai of all,
has none. They are left off the map and counted in `gap`, rather than drawn as English or guessed
onto Mambai. English everywhere else (543) is drawn as counted. `ENGLISH_WITHHELD` asserts both
cells, so a changed table stops the build.

## 5. Labels and nodes

`taxonomy/tl2015.py` has the reasoning per label. In short:

* Two new groups (`tree.d/tl.txt`): `austronesian.timoric` (Timoric; Glottolog splits it between
  timo1265 and west2945, readers know it as one) and `papuan.timor_alor_pantar` (timo1261: Bunak,
  Fataluku, Makasae, Makalero, Sa'ani) under au.txt's `papuan` root.
* Tetun Prasa and Tetun Terik are two new leaves inside Timoric. uk.txt's bare `austronesian.tetun`
  ("Tetun", used by uk, au and pl) is left alone: giving it children would make it a group, and
  those countries' named answers would draw washed out.
* Every label is a leaf, including Hull's cover terms Idalaka (211) and Kawaimina (41), the Atauro
  dialect names, Lolein and Nanaek (dialects to Glottolog, answers here), and Makuva (121, scattered
  over all thirteen municipalities, drawn as counted).
* Other (617) on `other`.

**Labels that move between censuses, worth knowing before reading anything into one municipality:**

* Dadu'a: 2010 has 1,656 in Dili (Atauro) and 1,400 in Manatuto; 2015 has 35 and 1,863. Over the
  same years Rahesuk in Dili went 985 to 2,287. Atauro's answers moved between the island's dialect
  names. Glottolog's Dadu'a point is in Manatuto.
* Ermera: Tetun Prasa 56,277 (48%) in 2010 and 23,701 (19%) in 2015, Mambai 40,946 to 76,785. Ermera
  is the municipality the English miscode sits in, and the swing suggests its coding was unsteady in
  both censuses. Drawn as 2015 has it.
* A few "Atauro" answers turn up far from the island: Adabe 144 in Covalima, Atauran 54 in Ainaro.
  Drawn as counted (they are rings, being under a dot nationally).

## 6. Colour

`tree.d/tl.txt` lists the hand-picks and the municipality each was chosen against. Checked by
OKLab distance between every pair of languages over 1,000 speakers sharing a municipality; what is
left under 0.10 is Dili's minor mixes (Makasae/Bunak) and Atauro names against mainland languages,
whose dots never share ground. Tetun Prasa was first a pale blue, 0.009 from English's near-white;
it is a pale cream.

## 7. Worth coming back for

* **The 2010 suco charts**, if anyone wants suco grain: 442 PDFs on the Wayback Machine
  (`mof.gov.tl/wp-content/uploads/2011/10/<district>-<suco>-fo-fila-fali-tetum-FINAL.pdf`, list
  pages under `.../sensus-fo-fila-fali/download-suco-reports/<district>-suco-reports/`), page 7,
  bar lengths as vector paths. 2010 vintage and a chart, so a derived tier.
* **The REDATAM host** INETL links (`20.6.104.113`), off in September 2026 (religiondots' `tl.md`):
  2022 microdata would give the two-answer question by suco, and the first answer alone might be
  usable as a single answer.

## Moved from countries/tl.py text (2026-10-06 sweep)

Cut from the reader-facing text and not recorded above; verbatim from the old `note_public`.

- The census lists two of Geoffrey Hull's group names, Idalaka (for Idaté and Lakalei) and Kawaimina (for Kairui, Midiki, Waima'a and Naueti), beside the languages they group, and a few hundred people gave them; they are drawn as answered.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Baikenu (Oecusse) is under Uab Meto, Glottolog's dialect of it, with Indonesia's Uab Meto. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
