# Guinea: RGPH 2014, main national language, by région, placed by prefecture

Drawn 2026-10-05 (session edd42a8c-gn). 9,436,465 people aged 3 and over, 8 régions, 24 nodes
from 24 table rows, every row `measured`. 9,424 dots at 1:1000, no rings. Inside each région the
dots are placed by prefecture (34) using CLEAR Global's shares from the IPUMS sample of the same
census.

```
python sources/gn_rgph.py --fetch     # copies religiondots' PDF (digest pinned), fetches CLEAR's CSVs
python sources/gn_place.py            # religiondots' hexes + each hex's prefecture
python taxonomy/build.py
python tools/check_country.py gn
python scatter.py --country gn
```

## Source

- **Table.** INS Guinée, RGPH 2014, *État et structure de la population* (122 pp), **Tableau
  5.08** (PDF p89): 24 rows x 8 régions + Total, one decimal, with an `Effectif` per région.
  Tableau 5.07 (p88), the same rows by urban/rural x sex, is the national check. The same volume
  religiondots draws Guinea's religion from (`religiondots/sources/gn.md`, whose §6 already
  noticed 5.08 and called it "the language table").
- **Route.** `https://www.stat-guinee.org/images/Documents/Publications/INS/rapports_enquetes/RGPH3/RGPH3_etat_structure.pdf`,
  no login (Wayback copy 2020-09-21). `--fetch` copied religiondots' verified download
  (3,762,432 bytes, sha256 `76e06c3b...`).
- **Question** (definitions, PDF p27): "Langue nationale parlée : la langue nationale
  habituellement parlée par l'individu même s'il parle d'autres langues", with examples. One
  answer, people aged 3 and over (5.07's title and total). Only national languages could be
  named, plus "Aucune" and "Autre langue nationale"; French is not an answer. **Main language**
  in `how`.
- **5.08's title says "plus de 15 ans" and is wrong**: its Effectif total is 5.07's 3+ total,
  9,439,468, to the person, and its Total column is 5.07's Total column exactly.
- **Grain.** 8 régions. religiondots checked all eighteen RGPH 2014 volumes for prefecture
  tables (for religion) and found none finer than région; the language table is the same.
  The microdata is only in IPUMS (account blocked).

## Kindia's Effectif is misprinted

5.08 prints Kindia's Effectif as `140 044`; the régions then sum to 8,179,028 against the printed
9,439,468. The residual, **1,400,484**, is used. Each région's Effectif over its Tableau 2.07
resident population: Boké 0.899, Conakry 0.910, Faranah 0.907, Kankan 0.877, **Kindia 0.899**,
Labé 0.905, Mamou 0.911, N'Zérékoré 0.899. The printed figure is the residual's digits with one
dropped and the rest shuffled, so no digit was guessed.

## Placement inside régions: CLEAR Global's prefecture shares

**CLEAR Global's Guinea dataset** (HDX `guinea-languages`, CC BY-SA 4.0, created 2025-03-05):
"main language spoken in the household" proportions for the country, the 8 régions and the 34
prefectures, which CLEAR made from the IPUMS International 10% sample of this census (IPUMS
variable LANGGN2, extract ipumsi_00292). Its pcodes are COD-AB's. Its 21 named languages carry
Glottolog names and codes for 21 of INS's 24 rows; it has no code for "Tomamania", "Autre langue
nationale" or "Aucune", which sit in its `Unknown` with children under 3.

**Counts are INS's; CLEAR only places them.** Inside a région, a language's dots go to each hex
in proportion to its Kontur population times CLEAR's share of that language in the hex's
prefecture (AGENT_BRIEF §4.4: a proxy that moves people only inside the counted unit needs no
ask, and this one is the same census). The région totals per language do not move.

- **Manya ("Tomamania")** is weighted by each prefecture's `Unknown` above the median of the 34
  prefectures (0.099). Macenta, where Toma-Manian is spoken, has 22.8% Unknown against 8.5-12%
  everywhere else except Kérouané's 14.4% (which borders Macenta); its excess, 13 points, is
  where most of N'Zérékoré's 63,800 Manya speakers go (about 49,000 are placed there; the rest
  in Beyla and Lola, whose excess is 1-2 points). Macenta's excess over the median is only about
  30,000 people, so CLEAR probably folded some Manya into its Toma or Maninka too; the weight
  says where, not how many.
- **"Aucune" and "Autre langue nationale"** are placed on population (16 région rows).
- `sources/gn_place.py` gives every religiondots hex the COD-AB ADM2 pcode its centroid falls
  in; all 91,048 land in a prefecture of their own région, all 34 are hit.

What the placement gives, per prefecture (top languages, % of the dots placed there): Boffa
Susu 80; Boké Fula 37, Susu 31, Landuma 9, Jahanka 9; Koundara Fula 72 with Jaad, Wamey and
Bassari 5-6 each; Télimélé Fula 90 inside Susu-majority Kindia; Kissidougou Kissi 31, Kuranko 26;
Kérouané Maninka 44, Konyanka 30; Beyla Konyanka 70; Guéckédou Kissi 72; Lola Kono 36, Kpelle
22; Macenta Loma 33, Maninka 20, Manya 16; N'Zérékoré Kpelle 77, Mano 12.

## Checks (`sources/gn_rgph.py`, `sources/gn_place.py`, all pass)

| check | result |
|---|---|
| file | 122 pages, 3,762,432 bytes, sha256 pinned, %%EOF |
| 5.08 parse | 24 rows x 9 columns off p89 = the transcription |
| columns | each sums to 100 within 0.2 (99.8-100.1) |
| Effectif | page digits = the 8 printed figures + 9,439,468; Kindia = residual 1,400,484; Effectif / resident 0.877-0.911 |
| 5.08 Total vs 5.07 Total | identical, all 24 rows |
| régions weighted by Effectif vs Total | worst 0.067 points |
| join | the 8 régions = religiondots' gn_hexes units, both ways |
| CLEAR | 34 prefectures = COD-AB gin_admin2 pcodes, both ways; shares sum to 1 |
| CLEAR régions vs 5.08 | within 3.2 points for every language both name; worst Susu in Boké 35.4 vs 32.2, Maninka in Faranah 39.1 vs 36.3 (CLEAR's shares leave out its three missing rows, so run a little high) |
| hexes | 0 of 91,048 needed the nearest-prefecture fallback; population = religiondots' to the person |

Drawn total 9,436,465 against 9,439,468: one-decimal cells printed 0,0 are not drawn and
rounding. A cell is good to about +-800 people (0.05% of a région of 1.7 million).

## Mapping calls (`taxonomy/gn2014.py`, `taxonomy/tree.d/gn.txt`)

- **Reused:** Fula (Pular), Maninka (ci/ml; Eastern Maninkakan), Soninke (Sarakolé/Maraka),
  Yalunka (Djalonké), Jaad (Badiaranké), Tenda Bassari and Wamey (Bassari, Koniagui), Susu (pl),
  Kpelle, Loma and Mano (au/pl).
- **Toma on `loma`.** Glottolog's Toma (toma1245, GN;LR) is the language Liberia calls Loma; one
  node.
- **Tomamania = Manya** (many1261, Mande), the Toma-Manian of Macenta. A Manding language, not
  Loma, so not merged into Toma.
- **Koniaka = Konyanka Maninka** (kony1250), its own node beside Maninka because the census
  prints it apart.
- **Kono = Kono of Guinea** (kono1267, Southwestern Mande, a Kpelle relative), node
  `kono_guinea` so that Sierra Leone's unrelated Kono (Mande, Vai-Kono) can take a plain `kono`.
- **New:** Konyanka, Manya, Kono (Guinea), Kuranko, Lele, Jahanka (Diakanka), Mikhiforé (Mande);
  Kissi, Baga, Landuma, Nalu (Atlantic), all flat as sn.txt does. Kissi, Baga and Landuma are
  Mel languages, but Temne already sits flat under Atlantic (au, pl), so no Mel group.
- **Baga** is one census answer for several Baga languages (Glottolog's Northern Mel: Baga
  Koga, Sitemu, Manduri, Sobané, Pukur); one leaf, not a group, since it is a named answer.
- **"Autre langue nationale" on `africa_other`** (a Guinean language not printed; indigenous
  remainder kept off `other`). **"Aucune" on `other`**: the person usually speaks no national
  language, so French or a foreign language, unnamed. 0.2%.

**Colours.** Susu (golden yellow), Kpelle (light cyan), Loma (ochre) and Mano (blue) were defined
bare by au and pl, where they are a few migrants, and are coloured here, where they are big.
Landuma, Jahanka and Mikhiforé were generated too close to Susu or to `africa_other` in Boké and
are hand-picked. Closest pairs at 1%+ in one région (OKLab): Baga/Mikhiforé 0.111 (Boké),
Fula/Loma 0.123, Kissi/Kono 0.123 (N'Zérékoré), Maninka/Konyanka 0.127.

## Not escalated

Forest Guinea has had clashes along ethnic lines (religiondots/sources/gn.md §5), and language
there follows ethnicity closely. Not escalated: the counts are INS's own published région table,
and the prefecture placement is CLEAR Global's public humanitarian dataset, no finer than
long-published language maps of the country.

## Terms

INS's report is a public download with no licence text. CLEAR Global's dataset is CC BY-SA 4.0
(attribution: CLEAR Global, from IPUMS International, Minnesota Population Center, and INS
Guinée); it is used only as a placement weight and its file is not republished. IPUMS's terms
allow publishing aggregate figures, which is what CLEAR did. COD-AB Guinea CC BY-IGO; Kontur
CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Maninka, Jahanka and Konyanka are in a Manding group with Mali's Bambara. Kuranko (Mokole) stays beside it. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
