# Angola: RGPH 2024, mother tongue, municipality

**Drawn.** 34,345,274 people aged 2 and over on 326 municipalities (Moxico Leste's nine at
province grain), 23 nodes, 34,336 dots at 1:1000, 2 rings.

## Source

- INE Angola, Recenseamento Geral da População e Habitação 2024, Resultados Definitivos.
  Quadro 6.2 `População por município e área de residência, segundo a língua materna, 2024` in
  each of the 21 provincial volumes (January and February 2026), and Quadro 6.2 of the national
  volume (20 November 2025), the same table by province.
- The question is língua materna, "the first language a child learns" (national volume p.54),
  one answer per person, asked of the population aged 2 and over (34,492,888 of 36,175,745).
- The PDFs are the 22 religiondots already downloaded for its religion table
  (`../religiondots/data/raw/ao/`, read in place; `sources/ao_rgph.py --fetch` would put any
  missing one in `data/raw/ao/`). File names and URLs are in the script.
- The queue's lead (2014 census, home language, province) is superseded: 2024 publishes mother
  tongue by municipality.

## What the tables print

- Provincial volumes, by municipality: Português, Kimbundu, Umbundu, Cokue (Chokwe/Kioko),
  Kikongo, Olunyaneka (Nhaneka), Nganguela, Oxikwanhama (Kwanhama), Ifyoti (Fiote), Muhumbi,
  Luvale, Khoisan, Línguas estrangeiras, Outras línguas, Não sabe. Cuanza Sul, Huíla, Namibe
  print the foreign column as nine languages instead; Malanje prints no foreign column.
- National volume, by province: the same with the foreign column as nine (Mandarim, Inglês,
  Francês, Espanhol, Alemão, Russo, Árabe, Lingala, Criolo).
- Footnote 5, national volume p.54: INE folded some languages into others before printing
  (Kimbundu includes Mbongala, Songo, Ngoya; Umbundu includes Mukubale; Cokue includes Lunda;
  Olunyaneka includes Humbi, Handa, Mucilengue, Sela, Kimbali; Nganguela includes Luchazi, Mbunda,
  Ukumbi; Kwanhama includes Herero and Mudimba; everything else, Gestual, Kisumbe, Kiswahili,
  Mwalabi, Sikabunda, Muko, Lucumai, is in Outras).

## The parse and its repairs (sources/ao_rgph.py)

The table reader is religiondots' Quadro 7 parser adapted (columns matched to the printed header
by a global assignment, figures placed by their right edge). The previous session wrote it and
was stopped before running it to the end; this session verified it and found five problems in
the source, each repaired in the script with the evidence beside it:

1. **Table titles.** Namibe prints its second panel as `Quadro 6. 3`, Cubango's second page has
   an en dash; both were missed, so half their columns were. The title pattern now takes any
   Quadro 6.x whose title says língua materna (Moxico Leste's two 6.x titles both say ethnic
   groups, so it still does not match there).
2. **Bengo prints its foreign column doubled.** Every Bengo row summed over its universe by half
   its foreign figure (Dande 405 foreign, 203 over; Panguila 666 and 332), and the column totals
   1,658 against the national volume's 829, exactly twice. Halved.
3. **Uíge prints the last three columns of three pairs of rows swapped.** Ambuíla -467 / Negage
   +324, Puri -5,594 / Maquela do Zombo +5,660, Cangola -2,295 / Uíge +2,327 (universe minus
   categories). Swapping foreign, other and não sabe within each pair brings Negage to +1,
   Maquela to 0, Uíge to 0, Puri +66, Cangola +32, and makes the figures plausible (the
   provincial capital gets 1,450 foreign speakers instead of 5; Maquela do Zombo on the DRC
   border 5,566 instead of 131). Ambuíla stays 144 over (0.7%), accepted as INE's.
4. **Malanje has no foreign column.** Its Outras (7,550) equals the national volume's Malanje
   other 6,721 + foreign 830 = 7,551, asserted. Written under its own label and drawn on
   `other`.
5. **Moxico Leste's volume has no language table.** Its Quadro 6.1 and 6.2 are both the ethnic
   table (p.81, checked); the only language text is a chart on p.43. The national volume's
   province row (382,707, categories 382,705) is shared over the nine municipalities by each
   one's own population aged 2+, largest remainder, `derived`.

Two Malanje municipalities had a printed row that does not add up (Kunda dya Baze 494,052) and
Cunene's Cafima the same; their communes are used where they match the religion table (13,741
and 17,788), as religiondots does.

## Checks (all pass, printed by the script)

| check | result |
|---|---|
| each municipality's categories sum to its universe, within 120 | 325 of 326 after the repairs; Ambuíla 144 |
| each province's municipalities sum to its own volume's province row, per category | all 21 |
| 326 municipalities, joined one-to-one to religiondots' ids by (province, name) | 326 |
| language universe equals religiondots' religion universe | 322; the other 4 below |
| foreign split re-aggregates to each municipality's measured total | exact |
| municipal foreign totals against the national volume's nine, per province | ratio 0.989-1.037 |

Four municipalities' language universe differs from the religion table's, in two pairs whose
totals agree: Malanje's Cacuso 56,342 / Ngola Luiji 18,601 (religion 51,218 / 23,725) and
Moxico's Luena 340,536 / Lucusse 16,469 (religion 337,755 / 19,250). The ethnic table of the same
volume (Quadro 6.1, Malanje p.109-110, Moxico p.97) prints the language table's figures, so the
religion table is the odd one out and the language figures are used.

National sums against the national volume, every language: Português 15,697,172 / 15,697,180,
Umbundu 5,881,184 / 5,881,181, Kimbundu 3,728,002 / 3,728,001, Cokue 2,374,098 / 2,374,104,
Kikongo 2,351,529 / 2,351,518, Olunyaneka 1,473,571 / 1,473,568, Kwanhama 1,003,782 / 1,003,783,
Nganguela 700,870 / 700,878, Fiote 390,845 / 390,858, Muhumbi 235,053 / 235,063, Luvale 179,508 /
179,517, Khoisan 11,394 / 11,406, Não sabe 147,468 / 147,511. Province populations against the
national volume differ only where the provincial volumes revise it (Luanda hands 144,745 to Icolo
e Bengo, Uíge 7,052 to Cabinda), as religiondots found for religion; the provincial volumes are
the later word and are drawn.

## Geography

religiondots' Angola layer, read-only: its normalized `ao.csv` gives each municipality's geo_id,
`data/geo/ao/ao_lookup.csv` its unit, `ao_hexes.gpkg` the 213,689 Kontur hexes with `pop`
(Lei 14/24's 326 municipalities; religiondots' `sources/ao_geo.py` has the boundary story). The
scatter met religiondots' `unreviewed` cap block in Mussulo (AO0516, one hex, 43% of 14,437
people, about 6 dots) and drew it as Kontur has it; not reviewed here.

## Mapping calls (taxonomy/ao2024.py, taxonomy/tree.d/ao.txt)

- Each printed label is drawn as the language it names, INE's folded-in languages with it:
  Cokue → Chokwe (with Lunda), Nganguela → Ngangela (Glottolog Nyemba, nyem1238; with Luchazi and
  Mbunda) under zm.txt's Chokwe-Lunda group, Oxikwanhama → Kwanyama (with Herero), Kimbundu
  (with Songo), Umbundu.
- **Kwanyama is its own node, not under na.txt's Oshiwambo**, which is a language node; making it
  a group would wash Namibia's Oshiwambo out. Coloured nearly the same orange, so the border reads
  as the one people it is.
- **Fiote** (Cabinda's Kongo varieties: Woyo, Vili, Yombe) is a node beside Kongo, not under it,
  for the same reason.
- **Nyaneka and Nkhumbi** (Glottolog nyan1305, nkhu1238) under a new Nyaneka-Nkhumbi (R.10) group.
  INE says Olunyaneka includes Humbi and Handa, which Glottolog files as Nkhumbi dialects, yet
  prints Muhumbi separately; both drawn as printed.
- **Khoisan → the `khoisan` root** (!Xun and Khwe, two families), washed out.
- **Criolo → `creole.portuguese_based`**: Cabo Verde's, Guinea-Bissau's and São Tomé's creoles
  are all Portuguese-based and the label does not say which.
- **Outras línguas → `africa_other`.** By the footnote it is Angola's other national languages
  plus Gestual (sign language), which cannot be taken out; mostly indigenous, so kept apart from
  `other` (Anita, 2026-10-04). Malanje's, which holds its foreign speakers too, → `other`.
- **The foreign languages** are shared out within each province by the national volume's
  provincial mix (largest remainder, `derived`) except in Cuanza Sul, Huíla and Namibe, where
  they are measured. Both margins are INE's; only the split inside a province is assumed. Of
  120,517 foreign speakers drawn, 84,444 are Lingala, most in Luanda, Lunda Norte, Uíge, Zaire and
  Cabinda.

## Colours

Hand-picked in the fragment: Umbundu red, Kimbundu magenta, Kikongo green, Fiote pale yellow-green,
Kwanyama orange (Oshiwambo's), Nyaneka gold, Nkhumbi dark green, Ngangela teal, Lingala light
cyan. Kimbundu, Umbundu, Kongo, Kwanyama and Lingala were defined bare by fi.txt and pl.txt (a few
immigrants each), so colouring them here changes nothing that matters there. Portuguese is
br.txt's (moved to a blue-violet while this was built, so Kikongo was moved off violet to green).
OKLab distance between neighbours on the ground is 0.10 or more, except Nyaneka against zm.txt's
pale Chokwe (0.074), which barely meet; Chokwe/Luvale (0.103) are zm.txt's.

## Not done

- A second source for the municipal pattern (the 2014 census asked language spoken at home).
- Moxico Leste below province: nothing published.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Kongo draws on 'Kongo (variety not given)' in a Kongo group with the DRC's varieties and Fiote; Kwanyama sits in an Oshiwambo group with Namibia's Oshiwambo. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
