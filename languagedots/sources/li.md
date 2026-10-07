# Liechtenstein: Volkszählung 2020, main language

Built 2026-10-05 (session edd42a8c-li). Rebuild:

```
python sources/li_census.py --fetch     -> data/raw/li/213.011d.json (one POST), data/normalized/li.csv
python taxonomy/build.py
python tools/check_country.py li
python scatter.py --country li
```

Drawn: all 39,055 residents of 31.12.2020 on 11 Gemeinden, 32 filled categories on 28 nodes,
36 dots and 27 rings. Nothing is left out: the table has no "not stated" row.

## 1. Source

Amt für Statistik, eTab (PxWeb) table **213.011d** "Ständige Bevölkerung nach Stichtag,
Hauptsprache, Geschlecht und Gemeinde", reference dates 31.12.2020, 2015 and 2010:
`https://etab.llv.li/PXWEb/api/v1/de/eTab/Bevölkerung/Bevölkerungsstruktur/213.011d.px`.
No key, no wall. It is the only language table on the server (a search for "Sprache" and
"Hauptsprache" finds only it).

- **Use the German tree.** The English one (213.011e) was last updated 2017 and stops at 2015;
  213.011d was updated 2022-12-15 and carries 2020.
- Server quirks (religiondots `sources/li.py` found them on the religion table): positional value
  indices, explicit item lists only, json-stat2 broken. The fetch reads json-stat v1 and checks the
  value count against the declared shape (420 of 420 for one date; 1,260 for three).
- The coverage lead said "pdf". The 2020 table is machine-readable on eTab; no PDF needed.
- statistikportal.li (the census landing page) is behind a Cloudflare challenge, so the
  questionnaire wording was not read. The category list is BFS's Swiss list
  (taxonomy/ch2000.py) almost verbatim, and the table carries one language per person, so this is
  read as the same single-answer "main language" question.

## 2. Checks (sources/li_census.py check())

- 32 filled categories sum to each Gemeinde's total exactly, and the 11 Gemeinden sum to the
  national row category by category exactly.
- National total 39,055 (the resident population of 31.12.2020).
- Cell status `-` is a true zero (no disclosure threshold; single people are published, e.g. 1
  Rhaeto-Romance speaker in Schellenberg). Status `..` occurs only for "Übrige westeuropäische"
  and "Übrige osteuropäische Sprachen" in 2020, in every cell; the remaining categories still
  partition every total, so those two are simply unfilled (their speakers sit under "Übrige
  Sprachen" or elsewhere), not lost.
- Earlier rounds from the same table, for plausibility: German 94.5% (2010), 91.5% (2015),
  92.4% (2020); the next languages are the same five each time (Italian, Portuguese, Turkish,
  Spanish, Serbo-Croatian).

## 3. Calls

- **2020, not 2015**: newest, same table, same categories.
- **Mapping follows ch2000** so the two sides of the Rhine read the same. "Deutsch" is German and
  includes the local Alemannic dialect (not counted apart); `swiss_german` is not used because
  the census does not name it. Regional remainders that cross families go on `other`: Übrige
  nordeuropäische (3), Westasiatische (28), Indoarische und drawidische (24), Ostasiatische (179),
  Übrige Sprachen (4). "Afrikanische Sprachen" (38) on `africa_other`. "Serbisch und Kroatisch"
  (288) on Serbo-Croatian.
- **Geography**: religiondots' `data/geo/li/li_grid_400m.gpkg` (171 Kontur hexes keyed by
  Gemeinde name, built for its Volkszählung 2015 religion table), read-only. The census's
  Gemeinde names are the grid's `unit` strings exactly (check_country: 11 units, all placed).
  The grid matters because seven Gemeinden own empty alpine exclaves. Placement is plain
  population within each Gemeinde; no sub-Gemeinde origin or citizenship grid was looked for, at
  one dot per 1,000 people it would move nothing.
- **No new nodes, no colour changes.** Only German draws more than ring-sized counts per
  Gemeinde, so neighbouring-colour checks are moot here.

## 4. Not done

- The coverage note mentions a 2020 "language at home" item with Liechtenstein dialect vs
  standard German (73% dialect). eTab carries no such table; it is likely in the PDF census
  report on statistikportal.li, which is behind Cloudflare. It would only split the German node,
  which the tree does not do for Switzerland either.
