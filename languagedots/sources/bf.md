# Burkina Faso: RGPH 2006, main language spoken, by province

Drawn 2026-10-05 (session d9e44929-bf). 12,413,963 people (residents aged 3 and over who
answered), 45 provinces, 36 nodes from 38 table categories. 12,398 dots at 1:1000, 5 rings.

```
python sources/bf_rgph.py --fetch     # copies religiondots' PDF, digest pinned
python taxonomy/build.py
python tools/check_country.py bf
python scatter.py --country bf
```

## Source

- **Tables.** INSD, RGPH 2006, *Thème 2: État et structure de la population* (181 pp), the
  volume religiondots draws Burkina Faso's religion from (`religiondots/sources/bf.md`).
  **Tableau A5.3** (PDF pp159-162): province x language, counts, three blocks of 15 provinces.
  **Tableau A5.2** (pp157-158): région x language, counts, with the foreign languages printed
  apart. **Tableau A5.1** (p156): national by milieu and sex.
- **Route.** Not on insd.bf's current site. Three Wayback captures on retired paths (in
  `WAYBACK` in the script); `--fetch` copies religiondots' verified copy (2,348,972 bytes,
  sha256 `722497af...`). The Wayback Machine was refusing this machine as a bot on 2026-10-05,
  so the fallback is untested today.
- **Question** (p37, p43): "la principale langue parlée par un individu, qui peut être une
  langue locale ou étrangère", one answer, asked of every member of the household (1985 and 1996
  asked one language per household). Tables cover residents aged 3 and over. **Main language
  spoken**, so `how` says that, and `note_public` says it need not be the first language.
- **Vintage: 2006, not 2019.** The 2019 census (5e RGPH) asked "principale langue couramment
  parlée", but its final report (`insd.bf/sites/default/files/2022-07/Rapport resultats definitifs
  RGPH 2019.pdf`, live, Tableau 11 p48) is national only, 18 categories (Grusi as one
  "Gourounsi", no Kasem/Nuni/Lyélé). The coverage sweep found the regional monographs carry
  language by urban/rural only. The *Volume des tableaux statistiques* (June 2024) may have a
  région table; the Wayback copy could not be fetched today (bot wall) and was not opened. Even
  so it would be 13 régions against 2006's 45 provinces with 26 named languages, exact counts.
  2006 is drawn; `note_public` gives the 2019 national shares beside 2006's.

| | 2006 (drawn) | 2019 national |
|---|---|---|
| Mòoré | 50.5% | 52.9% |
| Fulfulde | 9.3% | 7.8% |
| Gourmanchéma | 6.1% | 6.8% |
| Dioula | 4.9% | 5.7% |
| French | 1.3% | 2.2% |

(2006 shares of the 12,606,887 aged 3+, ND included.)

## Universe and gap

A5.3's Total is 12,606,887 residents aged 3 and over; 0.878 (Gnagna) to 0.922 (Kadiogo) of each
province's whole population (Tableau A 3.1 bis, 14,017,262). `gap`: children under three,
1,410,375; and ND, 192,924 (1.53%, Tableau 1.3 prints 1.5% for language), carried in bf.csv and
not drawn. The 2006 census reached every province; the insurgency came later.

## How the counts are made

- The 26 named national languages and "Autres langues nationales": A5.3 as printed, `measured`.
- **The foreign languages.** A5.3 prints only "Langues Africaines" (30,367) and "Langues Non
  Africaines" (174,486) per province. A5.2 prints their ten members per région: Ashanti, Djerma,
  Haoussa, Ouolof, Autre langue africaine; Français, Arabe, Anglais, Russe, Autre langue non
  africaine. Each province's group is split in its région's proportions (largest remainder),
  `derived`. Kadiogo is the Centre région alone, so its split is exact and `measured`. Because
  A5.3 by région equals A5.2's group sums (check 4), the split reproduces A5.2's régional counts
  within a third of a person per province. French is 170,047 of the 174,486 (97.5%, as p89 says);
  119,668 of them in Kadiogo.

## Checks (`sources/bf_rgph.py`, all pass)

| check | result |
|---|---|
| file | 181 pages, 2,348,972 bytes, sha256 pinned, %%EOF |
| A5.3 parse | 31 rows x 45 provinces + Burkina; labels in order; every province's 30 rows sum to its Total; Burkina = sum of 45 on all 31 rows |
| A5.2 parse | 39 rows x 14; every row's 13 régions sum to Burkina Faso; every column sums to its Total |
| **A5.3 by région = A5.2** | all 31 rows x 13 régions exact (the two foreign groups against their five members); also proves the province -> région table |
| A5.1 national total column | = A5.2's Burkina Faso, all 39 rows |
| aged 3+ / population | 0.878-0.922 per province, 0.899 nationally |
| foreign split by région | = A5.2 within 0.33 person per province |
| join | 45 provinces = religiondots' `bf_hexes` units, both ways, no alias needed (A5.3 spells Komandjoari and Balé as the hex layer does) |

**Parsing gotcha.** A5.2 and A5.1 use space thousands separators and the text layer usually gives
each cell its own line, but some lines hold two or three cells (`149 209 086` is 149 | 209,086;
`961 809 574 125 110 562` is three). `solve_row` enumerates the cuts and keeps the one where the
régions sum to the national cell (A5.1: M + F = T three times and urban + rural = total); every
row has exactly one. A5.3 has no separators.

## Mapping calls (`taxonomy/bf2006.py`, `taxonomy/tree.d/bf.txt`)

- **New nodes.** Gur: Gourmanchéma, Dagara, Bwamu, Lyélé, Nuni, Kasem, Winyé ("Ko"), Kusaal
  ("Koussassé"), Cerma ("Gouin"), Sisaala ("Sissaka"), Gurunsi. Mande: Bissa, Bobo (Bobo
  Madaré), San (Samo), Sembla (Seeku). `isolate.siamou` (Glottolog makes Sɛmɛ an isolate;
  older sources say Kru). `nilosaharan.songhay.zarma`, `nigercongo.kwa.asante`.
- **Reused:** Mòoré (coloured here), Fula, Dyula, Senufo, Lobi (ci.txt), Marka, Mamara, Dogon,
  Songhay, Tamasheq (ml.txt), Hausa, Wolof, Arabic, French, English, Russian.
- **"Gurunsi" is a leaf**, labelled "Gurunsi (Grusi, language not named)" (40,298). It names the
  Grusi peoples without the language, but it is a printed label and Kasem, Nuni, Lyélé and
  Sisaala are printed apart, so a Grusi group node over them would draw named languages under a
  washed parent. Reversible.
- **"Sissaka" = Sisaala.** No language has that name; 207 of 332 are in Sissili, Sisaala country.
- **"Ko" = Winyé** (Kõ, Kols): 7,205 of 10,301 in Balé, where Winyé is spoken.
- **Bwamu beside ml.txt's Bomu, not on it.** Mali's leaf holds Mali's own "Bobo/Bomu" answer;
  Burkina's "Bwamu" is the whole cluster (Buamu, Cwi, Láá Láá, Bomu). A near colour (peach next
  to Mali's apricot) keeps the border reading as one people.
- **Remainders.** "Autres langues nationales" (633,565, 5.0%) and "Autre langue africaine"
  (10,718) both on `africa_other`, as Mali does: both African, neither foreign-non-African. The
  national remainder is very likely Kurumfé (Soum 44,130), Karaboro, Komono and Turka (Comoé
  105,474), Birifor (Bougouriba, Noumbiel, Poni), Yaana (Koulpelogo 110,497) and Nankana
  (Nahouri 34,118), but nothing printed names them, so they are not guessed. A Burkinabè-only
  node would be a reversible call. "Autre langue non africaine" (418) on `other`.

**Colours.** Closest co-located pairs at 2%+ in one province (OKLab), after the change below:
africa_other (washed) with Nuni 0.078 and Gourmanchéma 0.080; Mòoré/Winyé 0.082 (Balé, Winyé
3.8%); Fula/Kusaal 0.090 (Boulgou); Tamasheq/Mòoré 0.091 (the Sahel, Mòoré small there);
Bobo/Marka 0.094 (Banwa). **ci.txt's Lobi was changed** from brick (0.60 0.15 30) to brown
(0.52 0.12 45): it sat 0.047 from Fula's red across the south-west (Poni, Bougouriba, Noumbiel),
and in Côte d'Ivoire it stays clear of Koulango's peach and Senufo's orange. Lobi is used only
by ci and bf.

## Placement

religiondots' `data/geo/bf/bf_hexes.gpkg` (geoBoundaries ADM2 provinces, Kontur 2023-11 400 m
hexes, 136,760 cells), read-only, `pop_weight`. Kontur against the 2006 census runs 0.93x
(Sourou) to 3.23x (Kompienga) (religiondots/sources/bf.md §7): a within-province weight, so the
shape inside a province is 2023's while the counts are 2006's. No cap stops.

## Terms

INSD's reports are public PDFs (here via religiondots' Wayback copy). geoBoundaries and Kontur
CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Dyula and Marka are in the Manding group; Tamasheq in the Tuareg group. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
