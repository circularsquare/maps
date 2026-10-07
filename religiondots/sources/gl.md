# Greenland — SLiCA 2003-2006, the Greenland-born, one national mix

Built 2026-10-03 by `fafd1067-gaps` (the spatial-gaps scout), under a supervisor. Closed on
2026-09-15 (`queue.md` "Old negatives re-checked, 2026-09-15", `sources.md`
§scout-2026-09-15-negatives) because the only religion figure found was the church roll; that record
named the missing check, "whether any Greenlandic survey asks religion", and this is it.

## 1. The source

**SLiCA, the Survey of Living Conditions in the Arctic**, Greenland component. Statistics Greenland
interviewed a random sample of people born in Greenland, aged 15 and over, drawn from the population
register, December 2003 to August 2006: 1,197 interviews, 83% participation. Eight municipalities and
their main towns chosen in advance (Nanortalik, Qaqortoq, Paamiut, Nuuk, Aasiaat, Ilulissat,
Upernavik, Tasiilaq), villages at random inside them, settlements under 500 people weighted up (Kruse et
al., "Design and methods in a survey of living conditions in the Arctic", Int J Circumpolar Health
2012, PMC3417679). The data went to Ilisimatusarfik; no public microdata file was found.

**The question is self-identification**: "Consider self to be Christian". SLiCA Results Tables
(Poppel, Kruse, Duhaime, Abryutina; ISER, University of Alaska Anchorage, March 2007), Cultural
Continuity section:

| table | Greenland |
|---|---|
| 162, by country | **98% yes**, 2% no, estimated total 35,969 |
| 163, by region | Sydgrønland 99 (5,037), Midgrønland 97 (15,594), Diskobugten 99 (8,453), Nordgrønland >99 (4,733), Østgrønland 99 (2,153) |

Tables 167-171 ("consider indigenous spiritual beliefs part of life") exist too; they are not an
affiliation and are not used. The PDF is `data/raw/gl/slica_results_tables_2007.pdf`, from Wayback
20180612024209 of `iseralaska.org/static/living_conditions/images/SLICA_Results_Tables.pdf` (live URL
404 since 2023). `sources/gl.py::check_slica` reads both tables off the PDF on every run.

**Checked and not usable for this**: Statistics Greenland's population health survey
(Befolkningsundersøgelsen i Grønland, SDU/SIF; 1993-94, 1999-2001, 2005-10, 2014, 2018, 2024-26):
the published topic lists (community, nature, family, food, language) name no religion item, and the
2005-07 report was not searched page by page. ESS and ISSP Denmark exclude Greenland. Compilers: ARDA
(WRD 2010) and Pew 2011 give 95.5% Lutheran, 2.3% agnostic, 0.8% Inuit spiritual beliefs, which is the
roll in compiler form and is not used.

## 2. The roll versus the survey, and Denmark

Denmark is drawn from ESS self-identification and prints its church roll beside it undrawn
(`countries/dk.py`). Greenland now matches: drawn from what SLiCA's respondents said, with the roll
used only to split the self-identified Christians (spec 3.1: a roll may split, never add).

Statistics Greenland `BEXKIRK` (PxWeb `bank.stat.gl`, BE/BE01/BE0120), born in Greenland:

| 1 January | members (incl. congregation) | Greenland-born | share | outside the church |
|---|---|---|---|---|
| 2012 | 49,092 | 50,340 | 97.5% | 943 |
| 2026 | 47,887 | 49,685 | **96.38%** | 1,506 |

96.38% < 98%, so the Church of Greenland's members fit inside the self-identified Christians:
96.38% `christianity.lutheran`, 1.62% `christianity`, 2% `unknown` (taxonomy/gl2006.py REVIEW has
each). The roll's fall since 2012 says self-identification has probably drifted down since SLiCA too;
note_public says so. Of the 7,030 people born outside Greenland, about half (3,583) are members.

## 3. Construction

- **One national mix.** SLiCA's five regions are the pre-2009 ones and cut across today's
  municipalities (Ilulissat moved from Disko Bay to Avannaata; Sermersooq holds Nuuk and East
  Greenland), and 97-99% is inside sampling noise for regions of a 1,197 sample. Every municipality
  takes 98%, the Cuba/Eritrea/Comoros construction (Anita's rulings 2026-10-03).
- **Universe the Greenland-born**, as SLiCA's was: 49,721 on 1 January 2026 (`BEXSTD`, by locality).
  The 7,019 born outside (Denmark 3,904, Asia 2,001, other Nordic and Faroese 528, rest of Europe 377,
  Americas 149, Africa 54, Oceania 6; `BEXST8G`) are `gap`, gap_share 0.1237, hand-written because
  `tools/gap_share.py` refuses (nothing excluded in the mapping). Children are drawn at the adults' mix.
- **Units**: geoBoundaries GRL ADM1 (gbOpen 9469f09), the five municipalities and Northeast Greenland
  National Park, keyed on ISO (GL-KU, GL-SM, GL-QE, GL-QT, GL-AV, GL-UO). Statistics Greenland's
  "outside municipalities" is Pituffik (25 Greenland-born, drawn in GL-AV, whose ground it is on) and 2
  unplaced people (GL-UO, at Daneborg).
- **Placement by locality, not Kontur.** Kontur holds ~20,000 of 56,740 people. Each of 78 towns and
  settlements with Greenland-born residents gets a 1.5 km disc weighted by them (`sources/gl_geo.py`),
  geocoded from GeoNames GL on the primary name only (the alternate-name field matched dozens of places
  and snapped settlements onto their town; caught because two settlements landed at Ilulissat's
  coordinates); namesakes are told apart by distance to their district's town; aliases Tiilerilaaq =
  Tiniteqilaaq, Kangerluk = Diskofjord, Naajaat = Naajat; Innaarsuit and Isertoq by hand.
- **geoBoundaries ADM1 leaves out Disko Island**: Qeqertarsuaq and Kangerluk sit 36-40 km off
  Qeqertalik's polygon; their discs are unclipped. Island settlements elsewhere sit up to ~20 km off
  the generalised coast. `EDGE_KM` = 50.

## 4. Result

49,721 people: 47,923 `christianity.lutheran`, 805 `christianity`, 993 `unknown`. 47 dots at 1:1,000
(16 at Nuuk), 4 at 1:10,000; `christianity` and `unknown` are under 1,000 and draw as rings.
check_mapping, check_md, built_countries --check and coverage clean.

## 5. Calls someone might reverse

1. Drawn at all on a 2003-2006 survey. Luxembourg was closed in 2026-09 for being one unit on ESS
   2002-2004; Greenland differs in that its 2026 roll still corroborates the level (96.4% members).
2. The roll split inside the Christians, rather than all 98% on `christianity.lutheran` or all on
   `christianity`.
3. The born-outside 12.4% left undrawn rather than drawn by birthplace (Denmark-born at Denmark's ESS
   mix, Asia-born through Pew), which would mix in a nationality model.

## 6. REOPEN

- SLiCA microdata (Ilisimatusarfik) with a region variable that maps to municipalities.
- Any later Greenlandic survey with a religion item (the 2024-26 population survey's questionnaire,
  once published).
