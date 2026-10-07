# Central African Republic: RGPH03 2003, language commonly spoken, by commune

Drawn 2026-10-04 (session d9e44929-cf). 3,344,358 people, 177 communes, 78 nodes from 78 census
categories. 3,307 dots at 1:1000, 27 rings.

## Source

- **Table.** ICASEES, *Troisième Recensement Général de la Population et de l'Habitation*
  (RGPH03, enumerated 8-22 December 2003, published 2005), "Langue couramment parlée". Read as
  transcribed by the U.S. Census Bureau: "Central African Republic Subnational Population and
  Housing Data Tables with Administrative Boundaries" (HDX, CC BY, release `uscb_202303`), sheet
  `Language`. The same workbook religiondots draws CAR's religion from; fetched separately into
  `data/raw/cf/` (one GET, 401,471 bytes).
- **The office's own route** is ICASEES's REDATAM server
  (`http://108.60.219.85/redbin/RpWebEngine.exe/Portal?BASE=RGPH03FRA`), which USCB cites. It timed
  out on 2026-10-03 (the coverage sweep) and again on 2026-10-04. Live, it would answer language
  by age directly and settle the Fulfulde question below; worth one more try in a later session.
- **Question.** The language each person commonly speaks, one answer each (the 78 columns
  partition the total to rounding). Not mother tongue: `how` says "language commonly spoken".
- **Vintage.** 2003 is the last census. RGPH4 (cartography 2021) has not been enumerated.
- **Grain.** 177 communes (ADM3), 22,000 people on average. The workbook also has prefecture
  and sous-préfecture rows; only communes are read.
- **Geography.** religiondots' `cf_hexes.gpkg` (34,651 Kontur hexes keyed by USCB `GEO_MATCH`
  on the 2003 boundary set; `religiondots/sources/cf_geo.md`). The Language sheet uses the same
  177 ids; `cf_uscb.py` asserts the same id set in the Language, Ethnicity and Religion sheets,
  and `check_country.py` found all 177 units in the layer. Read-only.

## Checks (`python sources/cf_uscb.py`)

All pass:

1. 78 language columns, each with exactly one census field name in the Data Dictionary, all
   distinct.
2. No negative cells; 1 country, 17 prefectures, 72 sous-préfectures, 177 communes.
3. Categories minus total per row: -7..+4, mostly negative (USCB: "small summation errors that
   were not corrected"). Communes minus national per category: largest 11.
4. The Ethnicity sheet's national total is 3,895,139, the published RGPH03 population (also on
   page 4 of ICASEES's "La RCA en chiffres").
5. The Fulfulde floor (below): 35 clean communes, the collapsed communes inside their range,
   and the implied under-threes 61% of the census's 0-4 year olds.

**Coverage.** The language total is 3,726,684, 95.7% of the census. Most communes are 96-99%;
seven in Ouham and Ouham-Pendé are far lower (Bah-Bessar 20%, Mia-Pendé 34%, Nana-Barya 38%,
Bédé 45%, Bakassa 48%, Nana-Markounda 61%, Paoua 79%). That north-west was in rebellion in 2003;
those communes' ethnicity and religion tables are complete, so language answers specifically
were lost there. Reported in `gap`, not filled.

## The Fulfulde column holds the children under three

USCB's metadata: "the data collected on language spoken excluded household members under the
age of 3." Yet the language total is 95.7% of the census, where leaving out the under-threes
would give about 85%. They are in the `Fulfuldé` column:

- Fulfulde is 9-12% of every commune. In the 35 communes where under 2% of people are Muslim and
  under 1% are of Haoussa/Muslim ethnicity (CAR's Fulfulde speakers are Peul and Mbororo, all
  Muslim), it is 7.2-11.9% of the census, median 10.3% outside Bangui and 9.4% in Bangui's three
  such arrondissements (lower urban fertility).
- **The decisive test:** in the five non-Muslim communes where the language table collapsed
  (34-61% answered), Fulfulde is still 10.6-11.6% *of the whole census population*. The floor
  scales with who was counted, not with who answered, which is what children entered by default
  look like and not what speakers look like.
- The implied under-threes, 382,004, are 61% of the census's 627,118 aged 0-4 ("La RCA en
  chiffres" §2.9.1). Three single years of five, less infant deaths, is about 62%.

**The call.** Each commune's under-threes = the floor share (Bangui 0.0944, elsewhere 0.1026,
measured every run) × its census population, capped at the published column. Fulfulde drawn =
column − that, never below zero: 75,087 people (from 457,091), zero in 57 communes, rows
`tier="derived"`. The under-threes are not drawn (they were not asked) and go in `gap`.
`data/normalized/cf_under3.csv` has the per-commune numbers.

What could contradict it: the floor's noise is about ±1 point of a commune's population (q1-q3
of the clean communes is 9.9-11.2%). As a check, the estimate is compared with each commune's
Muslims: it exceeds them in 9 communes, by 916 people in all, which is the size of the noise.
The alternatives were worse: drawing the column as published puts 380,000 phantom Fulfulde
speakers evenly over the whole country (it would read as CAR's third language, 12%), and
dropping it loses a real language of ~75,000 people (plus the separately printed Mbororo 22,543,
Fulata 12,801 and Peulh 1,926) in its north-western heartland (after the correction, of those
answering: Koui 35%, Groudrot 29%, Binon 23%, Bocaranga 18%; 33,550 in Ouham-Pendé, 13,074 in
Nana-Mambéré, 8,078 in Mambéré-Kadéï). The census's separate count of the Mbororo people, 38,589,
fits the order of magnitude.

## Labels (`taxonomy/cf2003.py`)

Mapped from ICASEES's French names, not USCB's ISO renamings. USCB got these wrong, and each one
was checked against where the speakers live:

| census | USCB | drawn as | why |
|---|---|---|---|
| Mandjia | Mangbetu (mdj) | Mandja | Mangbetu is in DR Congo; Mandja is Glottolog's Manza |
| Kara | Kara (CAR), Central Sudanic | Gbaya Kara | 76% in Bocaranga, in the census's Gbaya block |
| Yaka | Yaka (CAR) = axk, the Aka language | Yakpa (Banda) | 63% Basse-Kotto, 32% Ouaka, in the Banda block; Aka is printed separately in the Lobaye |
| Aka | Aka of Sudan (soh) | Aka (Bantu) | 99% Lobaye and Sangha-Mbaéré |
| Issongo | Manza (mzv) | Mbati | Isongo is the Mbati's own name; 81% Lobaye |
| Mondjombo | Mbangala (mxg) | Monzombo | Ubangian, Mundu-Baka |
| Kaka | (none) | Kako (Bantu) | 97% Mambéré-Kadéï, 85% Basse-Boumbé; ends the census's Gbaya block, but Gbaya has no Kaka variety and Kako is spoken there |
| Baba | Grassfields Baba | Banda: Baba | 93 people in Kémo and Ouaka, in the Banda block |
| Irri | Birri | Birri | Glottolog's "Irri" is an Edoid dialect of Nigeria |

Not in Glottolog, placed by the census's own ethnic block: Barar (Sara block, so Central
Sudanic), Buli, Bokaré, Budigiri, Gbaguiri, Gbadok, Tongo, Bouar, Boda, Mboundjia (Gbaya block,
4,000 people together), Bidjori (Banda block, 40). Each is a leaf of its own.

`Banda` is a named answer (281,499, 8.4%) and gets a leaf, `ubangian.banda.banda`, not the
group. `Fulfuldé`, `Peulh`, `Mbororo` and `Fulata` are four answers and four sibling nodes.
`Autres langues locales` (56,755) is on `africa_other`; `Langues non centrafricaines` (301,955,
54% in Bangui, 73% of Bangui's 1er arrondissement) on `other`. French is the likely bulk of it,
but the census does not say, so it is not guessed.

## Tree (`taxonomy/tree.d/cf.txt`)

New groups: Ubangian (Banda, Ngbandi, Zande-Nzakara, Ngbaka-Mba, plus Sere and Kpatili on the
group), Gbaya-Manza-Ngbaka (its own branch, as in Glottolog and Moñino), Adamawa (Mbum, Kare,
Pana, Talé), Central Sudanic and Maban under Nilo-Saharan, and Bantu leaves. Sango is reused
from `ca.txt` (`nigercongo.sango`, directly under Niger-Congo); it belongs in Ngbandi with Yakoma,
but moving it changes Canada's node, so it stays and only gets a colour here.

Colours hand-picked for the big languages and neighbours (see the fragment's header), then
checked: every pair of languages above 2% in a shared prefecture is at least 0.07 apart in
OKLab. Five were moved after the first build (Mpiemo/Babango, Mandja/Ngam, Suma/Kresh,
Kare/Mbum, Bokoto/Sango).

## What it shows

Sango, the national language, is a third of the answers and half of Bangui's: this is a
"commonly spoken" question, so Sango wins wherever people use it day to day, most of all in
Bangui, Ombella-Mpoko and the Lobaye. Behind it: Gbaya across the west (Ouham, Mambéré-Kadéï,
Nana-Mambéré), Mandja in Kémo and Nana-Grébizi, Banda in Ouaka and the Kottos, the Mbum languages
(Pana, Kare, Talé, Mbum) in Ouham-Pendé, Ngbugu in Basse-Kotto, Yakoma and Nzakara in Mbomou,
Zande in Haut-Mbomou (78%), Runga, Arabic and Sara in Vakaga.

## Limits

- **2003.** The war from 2013 displaced much of the Muslim population of the west and centre,
  which is where Fulfulde, Arabic and Hausa speakers were. `note_public` says so.
- **Placement** is by population inside each commune; nothing says where a language's own
  speakers sit inside it. religiondots found Kontur's CAR extract close to a rescale of this same
  census at commune level (`religiondots/sources/cf_geo.md` §4); only its within-commune shape is
  used.
