# Slovenia: Popis 2002, mother tongue

Built 2026-10-05 (session edd42a8c-si). Rebuild:

```
python sources/si_popis2002.py [--fetch]   -> data/raw/si/, data/normalized/si.csv
python taxonomy/build.py
python tools/check_country.py si
python scatter.py --country si
```

Drawn: 1,909,994 people on 192 municipalities, 12 labels on 12 nodes, 1,904 dots, no rings.
Not drawn: 52,316 (2.66%) "Neznano" (unknown), and 1,726 (0.09%) in withheld cells.

## 1. The table

Statistični urad Republike Slovenije (SURS), Popis prebivalstva, gospodinjstev in stanovanj 2002,
on the SiStat PxWeb API (no key, no login, browser User-Agent):

- `05W1007S.px` "Prebivalstvo po maternem jeziku, občine, Slovenija, popis 2002": 193 rows
  (SLOVENIJA + 192 municipalities) x 14 columns: total, Slovenski, Italijanski, Madžarski, Romski,
  Albanski, Bosanski, Hrvaški, Makedonski, Nemški, Srbski, Srbsko-hrvaški, Drugi, Neznano.
  **The drawn table.**
- `05W1607S.px` "Prebivalstvo po maternem jeziku, Slovenija, popisa 1991, 2002": national only,
  33 named languages. A check level (`geo_level=country_detail`), never drawn.
- `05W0405S.px` population by settlement: only for the official municipality codes (below).

Why 2002: the 2011 and 2021 censuses are register-based and carry no language (coverage sweep,
confirmed by the catalogue: no language table after 2002 except adult foreign-language surveys,
which are not first language). `05W2211S.px` has "pogovorni jezik v družini" (language spoken in
the family) but only by 12 statistical regions; mother tongue is at municipality, so mother
tongue is drawn.

Grain: municipality is the finest published. No settlement-level language table exists in the
catalogue.

## 2. Reading it, and the join

Same API and same trap as religiondots' religion table (`religiondots/sources/si.py`): the
`OBČINA` dimension codes are an alphabetical sequence where 001 is SLOVENIJA, not the official
municipality codes. The official codes come from `05W0405S.px`'s municipality labels (`001
AJDOVŠČINA`); the join is by name with the same two aliases religiondots needed (Destrnik spelt
DESTERNIK; Sveti Jurij ob Ščavnici under its pre-2001 name SVETI JURIJ), and **every one of the 192
is confirmed by population to the person**. The resulting codes are exactly religiondots' hex
`unit` ids (192 both ways, check_country ok).

Status flags read by text: `z` (withheld) is not drawn and not reconstructed; `-` is a true zero.
416 cells withheld, 474 true zeros.

## 3. Checks (all pass)

- national total 1,964,036, as published; the 13 categories partition it.
- municipality totals sum to the country exactly; per category the municipality sum falls short of
  the national row only where cells are withheld (1,726 people in all, 0.088%; Slovenian and
  Neznano are never withheld).
- `05W1607S` (2002) matches the drawn national row on all 12 shared cells to the person, and its 21
  further languages (4,652) plus its own Drugi (1,588) equal the drawn table's Drugi (6,240).

National, 2002: Slovenian 1,723,434 (87.75%), Croatian 54,079 (2.75%), Serbo-Croatian 36,265
(1.85%), Bosnian 31,499 (1.60%), Serbian 31,329 (1.60%), Hungarian 7,713, Albanian 7,177, Other
6,240, Macedonian 4,760, Romani 3,834, Italian 3,762, German 1,628, Unknown 52,316 (2.66%).

## 4. Mappings (`taxonomy/si2002.py`)

All onto existing nodes; no new nodes.

| label | people | node | why |
|---|---:|---|---|
| Slovenski | 1,723,434 | slavic.south.slovenian | |
| Hrvaški | 54,079 | slavic.south.croatian | |
| Srbsko-hrvaški | 36,265 | slavic.south.serbocroatian | printed apart, so its own leaf (as hr2021) |
| Bosanski | 31,499 | slavic.south.bosnian | |
| Srbski | 31,329 | slavic.south.serbian | |
| Madžarski | 7,713 | uralic.hungarian | |
| Albanski | 7,177 | albanian.albanian | |
| Makedonski | 4,760 | slavic.south.macedonian | |
| Romski | 3,834 | indoaryan.romani.romani | variety not stated |
| Italijanski | 3,762 | romance.italian | |
| Nemški | 1,628 | germanic.continental.german | |
| Drugi | 6,240 | other | see below |
| Neznano | 52,316 | not drawn | the gap |

**Drugi** is several families: nationally Russian 766, Montenegrin 462, Czech 421, Ukrainian 399,
English 345, Slovak 294, Polish 267, Romanian 251, Turkish 226, Chinese 216, French 206, Bulgarian
159, Arabic 130, Spanish 129, Croato-Serbian 126, Dutch 74, Vlach 45, Rusyn 42, Greek 40, Swedish
34, Danish 20, and 1,588 left as other. So `other`, whole. It is not split by the national detail:
nothing published says which municipality any of those languages is in, and it is 0.3%. Nothing
indigenous is in it (the autochthonous Italian and Hungarian and the Romani have their own
columns), so `other` rather than a regional remainder.

## 5. Geography and placement

religiondots' `data/geo/si/si_hexes.gpkg` (read only): Kontur 400 m hexes on GISCO's communes 2001,
the 192 municipalities of the 2002 enumeration (Slovenia now has 212; no crosswalk needed).
`unit` = official code, matching the join above. Inside a municipality, plain population.

Not done, and the obvious improvement: the bilingual areas are legally fixed lists of settlements.
Lendava, Moravske Toplice, Šalovci and Koper contain both bilingual and Slovene-only settlements, so
placing Hungarian and Italian on the bilingual settlements' hexes would be a within-unit proxy the
brief allows without an ask. It needs settlement polygons (GURS RPE) to key the hexes, which
religiondots does not have; not pursued.

## 6. Colour

Serbo-Croatian was generated (#63e6be) and sat 0.072 OKLab from Serbian and 0.076 from Slovenian,
the three meeting in Jesenice, Velenje and the Zasavje towns. `taxonomy/tree.d/si.txt` pins it to
`0.64 0.16 125`, a mid green: 0.116 from Croato-Serbian (hr), 0.121 from Bosnian, further from the
rest. It is also drawn in hr, cz, fi and us; in hr it sits beside Croato-Serbian and Serbian, both
still clear. Slovenian and Macedonian are 0.051 apart; Macedonian is 4,760 people (about five dots)
scattered in towns, and both colours belong to other countries, so left.

## 7. Facts behind note_public

Hungarian: Hodoš 59.0%, Dobrovnik 55.5%, Lendava 42.1%, Šalovci 10.9%, Moravske Toplice 6.9%;
those five hold 6,365 of the 7,430 placeable. Italian: Piran 7.0%, Izola 4.3%, Koper 2.2% (3,160 of
3,531 in the top five). Bosnian: Jesenice 16.7%, Velenje 6.3%. Serbo-Croatian: Velenje 5.7%,
Jesenice 5.2%, Postojna 5.1%. Croatian highest along the border: Metlika 13.3%, Rogatec 11.3%.
Unknown highest in Dobrna 7.4%, Piran 7.1%, Izola 4.7%, Ljubljana 4.1%.
