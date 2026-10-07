# Suriname: Census 7 (2004), language most spoken in the household, 62 ressorten

Built 2026-10-05 (session d9e44929-sr). 484,363 of 492,829 people drawn (98.3%), 14 nodes, every
row `derived`.

| | |
|---|---|
| source | ABS, `census-profile-on-ressort-level.xls`, sheet POPULATION BY RESSORT, block 11 "Most Spoken Language in the household" |
| question | household-level: "the language usually spoken by the household's members to one another" (Census 7 Volume 4 definitions) |
| unit counted | **households**, 123,463; turned into people (below) |
| geography | 62 ressorten, religiondots' COD-AB ADM2 hexes (`RD_GEO/sr/sr_hexes.gpkg`), read-only |
| script | `sources/sr_census.py [--fetch]` -> `data/normalized/sr.csv` |
| raw | `data/raw/sr/` (xls, Census 7 Vol. 4, Census 8 Vol. 3 and the three district presentations) |

## Why 2004

Census 8 (2012) asked the same question but has no whole-country sub-national table: national
only in six rows (Vol. 3, HWG-06a: Dutch, Sranan, Sarnami, Javanese, Maroon pooled, rest pooled);
full district tables for Paramaribo and Wanica only (Districtsresultaten Vol. I, 11 rows); a
chart for Nickerie, Coronie, Saramacca, Commewijne, Para (Vol. II, text layer has only each
district's top language); nothing for Marowijne, Brokopondo, Sipaliwini (census8dis3.pdf). Census
9 (2024-25) has published no results (religiondots enumerated ABS's whole media library). 2004
is finer in both directions: 62 units and 15 rows, with the three Maroon languages and the two
indigenous ones named separately.

## Households into people

The ressort table counts households. Precedent (Paraguay, Namibia) draws everyone in a household
in its language, but those tables were already in people. Here each household is weighted by the
national mean size of households of its language, from **Census 7 Volume 4, Tabel 02**
(households by size 1..9, 10+ and language), and each ressort is then scaled to its census
population:

    people[r,L] = hh[r,L] * size[L] * pop[r] / sum_L(hh[r,L] * size[L])

- Tabel 02 pools Arawak+Carib ("Inheemse taal") and the three Maroon languages ("Marrontaal"),
  so each pool carries one mean.
- The open "10 & meer" bin gets the single value (11.72) that makes the table's people equal
  Census 7's non-institutional population, 486,907 (Vol. 1 p.16; Vol. 4 gives 123,463 households,
  mean 3.94).
- Means: Dutch 3.93, Sranan 3.62, Sarnami 4.09, Javanese 3.55, indigenous 4.21, Maroon 4.31,
  Chinese 4.09, Portuguese 5.21, English 3.78, French 3.11, Other 3.99, Unknown 2.84.
- Against plain household shares the weighting moves 10,154 people (2.1%) between languages.
- Scaling to `pop[r]` means the 5,922 people outside private households (institutions and
  special groups, Vol. 1) are drawn in their ressort's household mix: there is no per-ressort
  count of them to take out. 1.2% of the country; not put in `gap` for that reason.
- Unknown (3,075 households, ~8,466 people after weighting) is scaled with the rest and then not
  drawn: that is `gap`.

## Checks (all asserted in the script)

1. Every ressort's 15 rows sum to its own printed "Number of Households"; national 123,463.
2. The 62 ressorten sum to the national column, row by row. Exact, integers.
3. **Second publication:** Vol. 4 Tabel 01 and Tabel 02 (pdf p.32, parsed from the text layer)
   print national households per language; every row total equals the xls national column.
4. Tabel 02's size columns sum to its printed Totaal row.
5. The 62 populations sum to 492,829.
6. The xls's 62 column headers equal religiondots' `RESSORTEN` order (district, name, p-code),
   which religiondots derived from names and asserted (its `sources/sr_geo.md`).

National result (people, derived): Dutch 47.3%, Sarnami 16.6%, Sranan 8.5%, Ndyuka 7.6%,
Saramaccan 7.3%, Javanese 5.0%, English 2.1%, Unknown 1.7%, Chinese 1.1%, Other 1.0%,
Portuguese 0.9%, Pamaka 0.5%, Carib 0.2%, French 0.1%, Arawak 0.1%.

Ressort sanity, households: Boven-Suriname 4,804 of 5,162 Saramaccan; Tapanahony 2,293 of
3,407 Ndyuka and 413 Pamaka; Westelijke Polders 1,506 of 2,275 Sarnami; Lelydorp 1,085 of
4,088 Javanese; Galibi 121 of 151 Carib; Coeroeni 261 of 310 Other. All where expected.

## Calls

- **Sarnami** -> new leaf `indoeuropean.indoaryan.bihari.sarnami`, Glottolog Caribbean Hindustani
  (cari1275; Sarnami Hindustani sarn1238 its dialect) under Bihari > Bhojpuric. Not merged into
  Bhojpuri: the census names it and it is a Bhojpuri-Awadhi koine.
- **Javanese** -> existing `austronesian.javanese`, though Glottolog separates Caribbean Javanese
  (cari1276). The census label is "Javanese".
- **Saramaccaans, Aucaans, Paramaccaans** -> three new leaves under `creole.english_based`, beside
  Sranan. Pamaka is a Glottolog dialect of Aukan, made a sibling so Ndyuka is not washed out as
  a group.
- **Arowaks -> arawakan.lokono, Caraib -> cariban.galibi_kali_na**, nodes from ve.txt and br.txt.
- **Chinese -> sinotibetan.sinitic** (as bo2024, py2002, pl2021).
- **Other -> `other`, not split by place.** ABS (Census 8 Vol. 3, note under HWG-06a) says the
  smaller Maroon groups' languages are filed in "other"; the ressorts show it also holds the
  unnamed indigenous languages (Coeroeni 261/310 households, the Trio area; Tapanahony 273;
  Boven-Saramacca 207; Kwakoegron 49). Since it mixes indigenous, Maroon and foreign answers
  and the census cannot separate them, it goes on the narrowest node holding all of them. A
  reversible call: Coeroeni's cell could plausibly go on `americas_other` as a place-dependent
  label, but that rests on ethnography rather than anything ABS printed.
- **Second language not drawn.** Vol. 4 Tabel 01 has it (Sranan is the second language of 45,634
  households, 37%); only the first is drawn, as everywhere on this map.

## Colours (taxonomy/tree.d/sr.txt)

New: Sarnami 0.74 0.16 50 (orange, near Bhojpuri), Sranan 0.78 0.14 128 (lime; pl.txt's node had
no colour), Saramaccan 0.68 0.15 95, Ndyuka 0.62 0.12 170, Pamaka 0.86 0.08 85. OKLab distances
afterwards, nearest pairs: Javanese ~ Portuguese 0.052 (they do not meet: Portuguese is the
interior gold fields), **Dutch ~ Javanese 0.081** (they do meet, in Commewijne and Wanica),
Sranan ~ Lokono 0.096. Dutch (cz.txt/us.txt) and Javanese (id.txt) are other countries' colours
and were left alone; Dutch/Javanese is the one worth a look on the map.

## Placement

Religiondots' 400 m Kontur hexes on the 62 ressorten, plain `pop_weight`. Every language's dots
in a ressort are spread by population alone. Religiondots' record covers the grid's vintage gap
(2023 grid on a 2004 census, ratio 1.27) and Galibi's near-empty 4 hexes. Scatter at 1:1000:
478 dots, 2 rings; 6,363 people (1.3%) sit under one dot per language nationally.

## Not done

- Census 9 (2024-25) supersedes this the moment ABS publishes language by district or ressort.
- A finer people-weighting (per-district household sizes by language) does not exist in the
  2004 publications; Vol. 4 Tabel E3 has sizes by district but not by language.
