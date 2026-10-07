# Mali: RGPH5 2022, mother tongue, by région

Drawn 2026-10-05 (session d9e44929-ml). 19,144,628 people (residents of ordinary households aged
3 and over), 20 units, 27 nodes from 28 table categories. 19,133 dots at 1:1000, 4 rings (the
foreign languages, which are `derived`).

```
python sources/ml_rgph.py --fetch     # copies religiondots' PDF, digest pinned
python taxonomy/build.py
python tools/check_country.py ml
python scatter.py --country ml
```

## Source

- **Table.** INSTAT, RGPH5 (2022), *Caractéristiques culturelles de la population* (70 pp),
  annex **A06** "Répartition de la population résidente par langue maternelle selon la région
  de résidence" (PDF pp56-57): 23 rows x 20 régions + Total, shares to two decimals, and an
  `Effectif` row. Annex **A03** (p53): national counts by sex for all 28 codes and `ND`.
  Tableau **3.01** (p39): national counts with ND spread, the second table. The same volume
  religiondots draws Mali's religion from (`religiondots/sources/ml.md`).
- **Route.** `https://www.instat-mali.org/laravel-filemanager/files/shares/rgph/rapport-caracteristiques-culturelles-population-rgph5_rpgh.pdf`,
  no login; also on the NADA (`microdata.instat.ml`, catalog 95). `--fetch` copied
  religiondots' verified download (17,708,018 bytes, sha256 `57a1967d...`).
- **Question** (Tableau 1.02, pp26-27): "Quelle est la langue maternelle de [Nom] ?", population
  aged 3 and over, one answer, 28 codes (19 named Malian languages, "Autre langue du Mali",
  "Autre langue africaine", six named foreign languages, "Autre langue non africaine"). The
  volume's definition (p25): the first language learnt in early childhood, the one spoken to
  the child at home. **Mother tongue**, so `how` says that.
- **Not drawn: "principale langue parlée"** (annex A07, same shape). The form asked the three
  languages each person speaks most; that is use, not first language. It is where Bambara's
  vehicular reach shows (57.83% against 49.90%), and `note_public` says so in one sentence.
- **Vintage.** 2022, the latest. The coverage sweep's lead was the 2009 census (Tableau S-3, 9
  old régions) and CLEAR Global's cercle proportions from IPUMS; RGPH5 is newer, on the current
  20 régions, and is INSTAT's own table, so neither was fetched.
- **Grain.** 20 régions, 957,000 people on average. Nothing finer: A01-A07 are national or
  régional (religiondots found no cercle volume). Ask 018 in religiondots (Anita) settled
  drawing Mali at its 20 régions.

## Universe and gap

A06's `Effectif` sums to 19,144,632 (printed 19,144,633; Tableau 3.02 prints the same total).
That is residents of ordinary households aged 3 and over: the ordinary-household population is
21,347,587 (religiondots, Tableau 2.9 of the structure volume), and each région's Effectif is
0.90 of it (Kayes 0.897, Bamako 0.902, Ménaka 0.903). `gap`: children under three, about 2.2
million; 941,335 in areas not enumerated for insecurity (modelled from building footprints);
106,567 in collective households or homeless. Population de droit 22,395,489.

## How the counts are made, and the one real call

Each région's count = A06 share x A06 Effectif, as printed. No rake.

**Why no rake.** A06's shares are of those who answered; Effectif includes the 68,582 who did
not (ND, 0.36%). The régions' shares weighted by Effectif do not reproduce A06's own Total
column exactly, and the misses have a direction: Tamasheq 3.989 against 3.85, Songhay 4.607
against 4.58, Hassaniya 0.932 against 0.92 (high); Bambara 49.778 against 49.90, Maninka,
Soninke, Senufo, Mamara (low). Drawn minus A03's answered count, per language:

```
Bambara +12,429   Tamasheq +28,937   Songhay +9,682   Dogon +6,095   Fula +4,050
Hassaniya +2,804  Soninke +1,998     Bozo +1,676      Arabic +1,127  (rest within +-850)
total +71,628 = ND 68,582 + the 3,046 by which Effectif exceeds A03's total
```

Every language's excess is above the rounding bound (-957) and they sum exactly to the
non-response. Tamasheq, 4.0% of the people, takes 40% of it: the non-response sat in the north
(paper forms were used in the north and centre, religiondots/sources/ml.md §3). A bounded
least-squares fit of per-région non-response to A03's counts puts 72,700-73,000 of it (against
71,628) mostly in Tombouctou, Kidal and Ménaka, but which of those is not identifiable
(Kidal and Ménaka trade off), so it is not used.

Raking to A03 (religiondots' move for religion) would spread ND evenly over the country: its
Tamasheq margin is 3.4% under share x Effectif, so the rake takes that off Tamasheq in every
région, Kidal and Ménaka included, and puts it on Bambara in the south. Keeping the shares as printed spreads each région's non-response
over that région's answers, which is what INSTAT's percentages already do. Every row is
`measured` except the split below.

**The foreign languages.** A06 prints one row, "Autre langue étrangère"; nationally it is A03's
Français 18,218, Anglais 1,974, Allemand 65, Russe 85, Chinois 69, Espagnol 63 (20,474; A06's
Total 0.11% = 20,474 / 19,073,005 to rounding). Each région's count is split among the six in
those national proportions, `tier="derived"`: 20,863 people, 89% French, 55% of them in Bamako.
Drawing them on `other` would have shown 18,000 French speakers as unnamed.

## Checks (`sources/ml_rgph.py`, all pass)

| check | result |
|---|---|
| file | 70 pages, 17,708,018 bytes, sha256 pinned, %%EOF |
| A06 parse | 23 rows x 21 columns; labels in order; every column sums to 100.00 (forced, as religiondots found for 6.13) |
| A06 Effectif | page digits = the 20 transcribed figures + 19,144,633; they sum to 19,144,632 |
| A03 | M + F = total on every row (4 off by one, rounding); rows sum to 19,141,586, printed 19,141,587 |
| A06 Total vs A03 / answered | 20 of 23 within rounding; Maninka 7.105 printed 7.10, Songhay 4.574 printed 4.58, Autre non africaine 0.071 printed 0.08 (3.01 prints 7.11 and 4.57: A06's Total column is closed by hand) |
| A06 régions weighted vs Total | within 0.139, the non-response pattern above |
| Tableau 3.01 vs A03 with ND prorated | the 20 named rows within 0.5 people |
| join | A06's 20 régions = religiondots' `ml_hexes` units (A06 prints Taoudénit and Dioila) |
| excess per language | >= -957, sums to 71,628 |

**Tableau 3.01's "Langue étrangère" is wrong.** It prints 69,350; A03's four remainders
prorated are 55,697, and 69,350 is exactly that plus "Autre langue non africaine" a second
time. Not used.

## Mapping calls (`taxonomy/ml2022.py`, `taxonomy/tree.d/ml.txt`)

- **New nodes:** Songhay group + leaf (`nilosaharan.songhay.songhay`, Nilo-Saharan as most
  readers know it; Glottolog makes Songhay a family), Dogon group + leaf (`nigercongo.dogon.dogon`,
  same reasoning), Khassonké, Konabéré, Marka (Dafing), Duungooma (Samogo), Bozo under Mande;
  Mamara and Bomu under Gur; Hassaniya beside Arabic; Tamasheq under Berber.
- **Mamara beside Senufo, not under it.** Mamara Senoufo is a Senufo language, but
  `nigercongo.gur.senufo` is the leaf Côte d'Ivoire draws its whole Senufo answer on; a child
  would make CI's answer a group node. Mali's "Sénoufo/Syenara" goes on that same leaf.
- **Kunabere = Konabéré** (Northern Bobo Madaré, nort2819, Mande). Not under that name in
  Glottolog; identified from the MPI numerals database ("Konabéré / Northern Bobo Madare") and
  its régions (San 0.09%, Koutiala 0.05%, on the Burkina border). 2,438 people. So "Bobo/Bomu"
  is the Bwa's Bomu (Gur), not Bobo Madaré.
- **Fula on `nigercongo.atlantic.fulah`**, not cf.txt's `peulh` (a separate CAR answer).
- **"Autre langue du Mali" and "Autre langue africaine" both on `africa_other`.** Both are other
  African languages and neither is a foreign language on `other`; the rule that keeps
  indigenous remainders off `other` is met. Reversible by giving the Malian remainder its own
  node. "Autre langue du Mali" is 10.29% in Ménaka (20,900 people), very likely Dawsahak
  (Idaksahak, Northern Songhay), but the census does not name it, so it is not guessed.
- "Autre langue non africaine" on `other`. `ND` is carried in ml.csv's national rows only.

**Colours.** Bambara (golden yellow), Soninke (light green) and Fula (strong red) were defined
bare by ca/fi/cf/us and are coloured here, where they are big; the rest are new. Closest pairs
of languages at 2%+ in one région (OKLab): Hassaniya/Khassonké 0.079 (Nioro, both small),
Fula/Senufo 0.095 (Sikasso), Tamasheq/Bambara 0.102 (the north, Bambara small there),
Bomu/Bambara 0.109 (San, Koutiala). Everything else 0.12 or more.

## Placement

religiondots' `data/geo/ml/ml_hexes.gpkg` (COD-AB v03 régions, Kontur 2023 400 m hexes, 141,250
cells), read-only, `pop_weight`. The two Bamako cap blocks are registered `real` in
religiondots' kontur_cap.csv and scatter found them. Kontur against the census is 0.27x in
Ménaka and 2.5x in Douentza (religiondots/sources/ml.md §6): a within-région weight only.

## Terms

INSTAT's report is a public download. COD-AB Mali CC BY-IGO; Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Bambara, Maninka, Khassonké and Marka are in a Manding group with Guinea's Maninka and Burkina's and Côte d'Ivoire's Dyula; Tamasheq is in a Tuareg group with Niger's Tamajaq. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
