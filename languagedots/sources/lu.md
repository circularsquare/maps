# Luxembourg: census 2021, main language

Built 2026-10-05 (session edd42a8c-lu). Rebuild:

```
python sources/lu_census.py --fetch     -> data/raw/lu/ (DF_B1625, the PDF, 102 GetFeatureInfo calls, ~12 min)
python sources/lu_census.py             -> data/normalized/lu.csv
python taxonomy/build.py
python tools/check_country.py lu
python scatter.py --country lu
```

Drawn: 563,092 people (everyone who answered) on the 102 communes, 59 labels on 58 nodes, 540
dots, 33 rings. Placement: religiondots' Kontur hexes cut by commune and scaled to the census
(`religiondots/data/geo/lu/lu_grid_400m.gpkg`, read-only).

## 1. The question and the national tables

Census of 8 November 2021, "Quelle est la langue dans laquelle vous pensez et que vous connaissez
le mieux ?", one answer: the *langue principale*. Six boxes (Luxembourgish, Portuguese, French,
English, Italian, German) and a write-in. A second, multi-answer question asked the languages
usually spoken at home and at school or work; it is not used (spec §3.6: a single-answer
question from the same census beats it, and only 74% answered it).

National figures: STATEC, RP2021 "1ers résultats" n°8, *Une diversité linguistique en forte
hausse* (Fehlen, Gilles, Chauvel, Pigeron-Piroth, Ferro, Le Bihan, University of Luxembourg,
2023; `data/raw/lu/rp08-03-02-fr.pdf`).
- Tableau 1, p. 4: Luxembourgish 275,361, Portuguese 86,598, French 83,802, English 20,316,
  Italian 20,021, German 16,412, other 60,582; total 563,092 of 643,941 residents. 10.4% did not
  answer and 2.2% were marked "not of an age to speak" (p. 3): 80,849 people, the `gap`.
- Tableau 3, p. 6: the 52 write-in languages with more than 100 speakers, 57,146 in all (the
  printed total; the script asserts it). The remaining 3,436 are on `other`.

## 2. The commune data: geoportail.lu, not LUSTAT

The coverage lead said "commune maps in RP2021 No.8; table-level data unverified". LUSTAT (the
.Stat Suite at lustat.statec.lu) has **no language dataflow**: every DSD_CENSUS flow was listed on
2026-10-05 and none carries language. The commune data is on geoportail.lu instead, as WMS layers
of the public map service `wms.geoportail.lu/public_map_layers/service`:

| layer | content |
|---|---|
| 2735-2739 | share of the commune with main language Luxembourgish, French, German, Portuguese, English, 2021, one decimal |
| 2740 | share with none of the three national languages (allophones), 2021, unrounded |
| 2610 | share of Italian citizens, 2021 (religiondots' nationality layers) |
| 1609-1615 | 2011 census, the seven categories with shares **and counts** (witness only) |

There is no 2021 Italian or "other" layer. GetFeatureInfo at a point inside each commune
(religiondots' route: WMS 1.3.0 reads EPSG:2169 northing first; `text/plain`). The commune names
returned match STATEC's DF_B1625 at every code, after two spellings religiondots already found
and two that differ between layers (`Redange/Attert`, `Lac de la Haute Sûre`), in `GP_ALIAS`.

Shares are of each commune's respondents (p. 13), whose number is not published.

## 3. How the counts are built, and the calls

1. **The five languages with a layer**: share x commune population (DF_B1625), then scaled per
   language to Tableau 1. The factors (Tableau 1 / sum of share x population) are Luxembourgish
   0.892, French 0.848, German 0.862, Portuguese 0.878, English 0.840, allophones 0.862: all near
   the 87.4% response rate, Luxembourgish highest, as the publication's note on response by
   migration background predicts (97.6% for Luxembourg-born citizens, 63.8% for foreigners born
   in Luxembourg). Assumption: within a commune, non-response does not depend on the language,
   beyond what the national factor absorbs. `measured`.
2. **Italian + other** per commune = allophones - Portuguese - English (never negative; scaled to
   80,603). Split by iterative proportional fitting: commune totals fixed, national Italian
   (20,021) and other (60,582) fixed, seed Italian = the commune's Italian citizens (layer 2610),
   seed other = the commune's Italian+other. AGENT_BRIEF §4.4 placement proxy (the specific
   citizenship): counts are the census's at both margins. `modelled`.
   **Witness**: the 2011 layers carry counts; their Italian sums to 13,896, exactly 2011's Tableau
   1 figure. The 2021 Italian placement correlates with 2011's Italian speakers by commune at
   r = 0.984; half the L1 distance between the two distributions is 0.137.
3. **The 52 write-ins and the remainder**: each commune's "other" split at the national
   proportions of Tableau 3. A national count placed inside the country (§4.4), `modelled`; the
   viewer hides them with inferred dots off. A better placement (Spanish by Spanish citizens etc.)
   would need nationality by commune beyond the five layers geoportail has.

Mapping calls (taxonomy/lu2021.py has them all): the five BCMS names and "Yougoslave" each on
their own node (`yugoslav` is new, tree.d/lu.txt); "Créole" on the `creole` group (the
publication says it covers several regions' creoles; Cape Verdean is printed apart); "Persan" and
"Farsi" merged on Persian; "Pular" on Fula as gn2014; "Chinois" on the Sinitic group, as other
countries do.

Colour: Luxembourgish was generated #348dcf beside German #359bd9 and Portuguese #6b8acf;
hand-picked teal `0.58 0.10 200` (#008c92) in tree.d/lu.txt. Its only other users are au, fi, pl
with a few hundred people.

## 4. Not done

- Placement of the write-ins by their own nationality (geoportail has only Portuguese, French,
  Italian, Belgian and German citizens per commune).
- Cross-border workers (about half the workforce) live abroad and are not counted; said in
  `note_public`.
