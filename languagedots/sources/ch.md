# Switzerland: Volkszählung 2000, main language

Built 2026-10-04 (session d9e44929-ch). Rebuild:

```
python sources/ch_vz2000.py --fetch     -> data/raw/ch/ (3 GETs, ~2 MB)
python sources/ch_geo.py                -> data/geo/ch/ch_units.gpkg, ch_hexes.gpkg
python sources/ch_vz2000.py             -> data/normalized/ch.csv (needs ch_units.gpkg)
python sources/ch_geo.py                (again: prints the census check on the layer)
python taxonomy/build.py
python tools/check_country.py ch
python scatter.py --country ch
```

Drawn: 7,287,357 people (everyone; the census imputed non-response) on 2,198 communes of the
1 January 2020 state, 36 labels on 29 nodes, 7,273 dots, 1 ring. Nothing is left out.
Since 2026-10-06 German is split into Swiss German and Standard German (section 5a): 30 nodes.

## 1. Which source, and why the old one

Two candidates from the Federal Statistical Office (BFS):

| | Volkszählung 2000 | Strukturerhebung (since 2010) |
|---|---|---|
| who | everyone, full count | a sample of ~200,000 a year, people in private households |
| question | one main language | up to three main languages |
| finest published grain | 2,896 communes | 143 districts, 2022-2024 pooled (T40.02.01.08.09) |
| categories at that grain | 36 | 6: German, French, Italian, Romansh, English, other |

The coverage sweep's lead ("commune, pooled 5-year structural survey") does not hold: BFS's asset
catalogue (dam-api, prodima 01.08.01, every language asset listed) has nothing below the district
for the survey; the commune-level series it does have ("Anteil der Wohnbevölkerung mit deutscher
Hauptsprache seit 1920") is the Swiss cities' statistics, cities only.

**Chosen: the 2000 census.** Single answer (spec §1), full count, every commune, and 36 categories
against 6. The survey's district table would need the multi-answer scaling of spec §3.6 on top of
sampling error (Romansh at district level carries ±25-90% intervals) and would put 20% of
mentions in one "other". AGENT_BRIEF §2: an old vintage is fine, say the year. The cost is real
and is said in `note_public`: since 2000 the country grew from 7.3M to 9.0M and English, Albanian
and Portuguese grew most. Not chosen either: religiondots' move for the same census (2000 commune
structure fitted to current canton totals), which is a proxy and so Anita's call; the survey's
canton table would anyway only re-weight six categories.

**Corroboration, national, 2000 census against the 2024 survey** (survey shares are of mentions
per resident and can sum over 100%): German 63.7% / 61.1%, French 20.4 / 22.6, Italian 6.5 / 7.8,
Romansh 0.48 / 0.43, English 1.0 / 6.5, Albanian 1.3 / 3.4, Portuguese 1.2 / 3.4, Spanish 1.1 / 2.5,
Serbian-Croatian 1.4 / 2.1 (asset 36399478, "Die häufigsten Hauptsprachen, 2024"). The national
languages' shares hold; the immigrant languages roughly doubled and English sextupled.

## 2. The table

`px-x-4003000000_123`, *Wohnbevölkerung 2000 nach Wohnsitztyp, Kanton / Bezirk / Gemeinde,
Staatsangehörigkeit (Kategorie) und Hauptsprache*, found through ckan.opendata.swiss
(`wohnbevolkerung-nach-wohnsitztyp-region-hauptsprache-und-staatsangehorigkeit-kategorie`), as
religiondots found its religion table. 3,107 geographic rows x 2 residence types x 3 nationality
cells x 37 language cells.

- **The json-stat API returns 403 for this cube.** Every commune x nationality x language is
  345,000 cells, over PxWeb's cell limit (religiondots' religion cube, 59,000 cells, passes). The
  whole .px file is one GET instead: `https://www.pxweb.bfs.admin.ch/DownloadFile.aspx?file=px-x-4003000000_123`
  (1.7 MB, CHARSET "ANSI" = cp1252). `sources/ch_vz2000.py` parses it.
- **Civil-law residence** (`Zivilrechtlicher Wohnsitz`), the first block, as religiondots uses for
  the same census; its national total, 7,287,357, is the published one.
- The question: "Welches ist die Sprache, in der Sie denken und die Sie am besten beherrschen?"
  One answer. The form offered no Swiss German box, so Swiss German is inside German.

## 3. Checks (all pass)

- the 36 categories partition the national total; in every commune Swiss + foreign = total and
  the categories sum to the total;
- **a second table of the same census**: every one of the 2,896 communes' totals equals the
  religion table's (px_122, religiondots/data/raw/ch/, read only) to the person;
- after moving onto 2020 communes, every category's commune sum equals its national figure, all
  2,896 communes land on a unit, and all 2,198 units receive people.

## 4. Geography

The units are religiondots' 2,197 commune polygons (GISCO LAU "2021", really the 1.1.2020 state,
lakes and comunanze dropped) **plus Verzasca**, and the hexes are religiondots' `ch_grid_400m.gpkg`
unchanged plus Verzasca's own. Two things religiondots' Swiss geography gets wrong, both found
here and both NOT fixed in religiondots (read only):

1. **Verzasca (BFS 5399) is missing from religiondots' layer.** Its polygon is in the LAU file, but
   religiondots' `sources/ch_geo.py` drops every feature BFS's 1.1.2020 register does not list,
   and Verzasca was created on 18 October 2020; so it went out with the lakes ("5399 holds 802
   people" in that file). Brione (Verzasca), Corippo, Frasco, Sonogno and Vogorno, 746 people in
   2000, have nowhere to land there. Here `sources/ch_geo.py` adds the LAU polygon (218 km²,
   no overlap with the others) and 161 Kontur hexes cut religiondots' way (representative point
   inside, clipped). religiondots' own ch.csv resolves them to 5399 and its rescale then has no
   polygon for them.
2. **Castel San Pietro is empty in religiondots.** BFS's correspondence API, even with territory
   exchanges excluded, gives four Muggio-valley communes two successors each (Castel San Pietro,
   Casima, Monte, Caneggio: the 2004 merger into Castel San Pietro and the 2009 creation of
   Breggia are chained). A plain `dict(zip(...))` keeps the last row, which sends Castel San
   Pietro itself (1,720 people in 2000) and Casima and Monte (154) to Breggia and leaves 5249
   empty; religiondots' ch.csv and ch_commune_rescaled.csv indeed have no 5249 row.
   `SPLIT` in `sources/ch_vz2000.py` sends each to the commune that holds its people today, and
   the script stops if another multi-successor commune appears.

Rebuilding the whole layer with `sources/_grid.py` was tried first and is worse here: by plain
centroid it left 176,700 people in lake-shore hexes outside every commune and six small communes
with no hex, which religiondots' builder had already solved.

Placement weights: Kontur 2023 population inside each commune, the same for every language
(nothing says where in a commune its speakers live). Verzasca's Kontur total (3,170) is far above
its census (746), but only the shares inside it matter.

## 5. Mapping calls (taxonomy/ch2000.py has them in full)

- German is split into Swiss German and Standard German (section 5a).
- "Rätoromanisch" (35,097) on the existing `rhaetoromance` node ("Rhaeto-Romance", the literal
  English of the term; uk and pl put theirs there). In Switzerland it is all Romansh. Coloured a
  deep rose in `tree.d/ch.txt` (was a periwinkle next to German's blue in Graubünden).
- BFS's own merges, drawn on the named language: Spanish with Catalan and Galician, English with
  Scots, Turkish with the other Turkic languages, Russian with Belarusian and Ukrainian (Lüdi and
  Werlen, *Sprachenlandschaft in der Schweiz*, BFS 2005, p. 11 note 2; the px residuals are too
  small to hold them).
- Cross-family regional remainders on `other` (uk2021's rule): other West, North and East
  European, other European, West Asian, "Indo-Aryan and Dravidian" (28,101, mostly Tamil but two
  families), East Asian, other answers. 79,859 people in all, 1.1%.
- "Other Slavic" on Slavic; "African languages" (9,202) on `africa_other` (Arabic is printed
  apart).

## 5a. Swiss German and Standard German (2026-10-06, session 5d7dac7e-ch)

Anita's ruling: draw Swiss German as its own language in Switzerland; Bavarian, Alemannic and
the rest stay German in Germany and Austria. Node: `indoeuropean.germanic.continental.swiss_german`,
which already existed (uk2021, us2024, ca2021) as a **sibling** of
`indoeuropean.germanic.continental.german`; repeated in `tree.d/ch.txt` with us.txt's colour
(0.80 0.09 245, #8cc4f4, pale sky blue against German's #359bd9). A different colour would
have to be changed in us.txt too (the build refuses a node coloured two ways).

**The source.** The main-language question has one "Deutsch" box. The same 2000 census also
asked the languages spoken at home / with relatives, with Schweizerdeutsch and Hochdeutsch as
separate boxes (since 1990). BFS never put that question on PxWeb (every VZ2000 cube,
px-x-4003000000_101 to _182, was listed: none holds it); it is published by language region x
nationality in Lüdi & Werlen, *Sprachenlandschaft in der Schweiz*, BFS 2005, Tabellen 18 and
20-22 (`data/raw/ch/luedi_werlen_2005.pdf`, pp. 37-39):

| region | nationality | Hochdeutsch only | Schweizerdeutsch only | both | drawn Standard German |
|---|---|---|---|---|---|
| German | Swiss | 1.3% | 90.8% | 5.4% | 4.1% |
| German | foreign | 13.8% | 29.1% | 6.8% | 34.6% |
| French | Swiss | 31,467 | 82,837 | 19,438 | 30.8% |
| French | foreign | 13,961 | 2,527 | 1,699 | 81.4% |
| Italian | Swiss | 6,722 | 24,493 | 4,054 | 24.8% |
| Italian | foreign | 3,167 | 1,143 | 307 | 71.9% |
| Romansh | Swiss | 272 | 9,188 | 545 | 5.4% |
| Romansh | foreign | 334 | 219 | 74 | 59.2% |

Standard German share = (only + both/2) / (all three): "both" is split evenly, since nothing
says which is the person's own.

**The model.** px_123 already splits each commune's German main-language speakers into Swiss
and foreign citizens, so each 2000 commune's two counts are split at its region's rates, summed
onto the 2020 units and rounded (German per unit is kept exactly; Swiss German is the
remainder). The language region is the commune's majority main language among the four
national ones, BFS's own definition; it gives 1,672 / 892 / 268 / 64 communes against BFS's
published 1,669 / 892 / 269 / 66 (three communes differ; for Swiss citizens the rates of the
regions involved differ by about a point). One override: Bosco/Gurin (5304), the Walser village
in Ticino, is Italian-majority in 2000 (40 Italian, 31 German) but its German speakers are
Walser, so it takes German-region rates (`REGION_OVERRIDE`); 1,673 / 892 / 267 / 64 with it. Rows are `modelled`. The substitution being made: a
rate measured on home-language German speakers is applied to main-language German speakers.

The commune nationality split does most of the placing a canton rate would do, at a finer
grain: the foreign share is what drives the canton differences BFS's 2017 study reports (Basel,
Zug, Zürich low on Swiss German). German citizens alone are not in px_123 (only Swiss/foreign),
so foreign German speakers stand in for them.

**Result.** 4,639,870 German main-language speakers: 4,285,603 Swiss German (58.8% of the
country), 354,267 Standard German (4.9%). Zürich city 9% Standard German, Bern 7%, Geneva 46%,
Lausanne 43%. Check against the source: home-language Standard German by the same formula,
summed over the four tables, is about 367,000 people; the model draws 354,000.

**Checks.** Every non-German row of ch.csv is identical to the pre-split file (31,660 commune
rows), German per unit and unit totals unchanged, so Romansh, Italian and French are untouched.
Walser German stays inside Swiss German, as it is Alemannic and the census does not name it:
the Valais and Graubünden Walser communes are German-majority (German-region rates), and Bosco/
Gurin is overridden to them.

**Not used.** BFS 2017, *Schweizerdeutsch und Hochdeutsch in der Schweiz* (ESRK 2014,
`data/raw/ch/bfs_2017_schweizerdeutsch_hochdeutsch.pdf`): it has cantons (chart G10/G11), but
for "regularly used" languages, which include reading and media (97% of German Switzerland
"uses" Standard German), on a sample with wide intervals, 14 years from the census. The 2000
family-language tables are the same census and the same kind of question (spoken at home).

## 6. Not done

- Tamil, Kurdish, Chinese and Thai have national figures in Lüdi and Werlen but not by commune;
  a finer 2000 table by commune was not found (the per-canton VZ2000 workbooks on opendata.swiss,
  `su-d-vz18-k-02`, were not opened; worth a look if Tamil matters).
- Liechtenstein's office publishes main language by its 11 communes (opendata.swiss,
  `standige-bevolkerung-nach-hauptsprache-geschlecht-und-gemeinde`, etab.llv.li PxWeb), a lead
  for `li` if it is ever queued.
