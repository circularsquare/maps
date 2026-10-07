# Mongolia (mn): record

**Drawn 2026-10-05**, 3,196,973 people on 21 aimags and Ulaanbaatar. Ethnic group read as
language (AGENT_BRIEF §2, Anita's 2026-10-05 ethnicity ruling). Session edd42a8c-mid.

## Table and vintage

2020 Population and Housing Census, English national report (`Census2020_Main_report_Eng.pdf`,
already downloaded by religiondots under `../religiondots/data/raw/mn/`, read-only). The
questionnaire (report p202) asks ethnicity (Q6) and no language question; the only language item
is literacy "in any language".

- p210, appendix table 1.1: resident population by aimag (counts). Parsed totals sum to 3,197,020
  and each equals the sum of its 15 age groups (that is how the space-grouped digits are split).
- p230, table 4.1: resident population by ethnic group, national counts. 31 groups + "other
  foreign / Mongolian citizen" (3,979) sum to 3,174,565 Mongolian citizens, as printed.
- p222-226, tables 3.6 (each aimag split by group, row %) and 3.6a (each group split by aimag,
  column %), one decimal each, 26 columns. 3.6a folds Tsakhar, Khorchin, Khalimag, Tumed, Sunud,
  Tuved, Balba and "other" (143 people) into "Other ethnic groups / Mongolian".
- p59, table 3.7: foreign citizens by aimag (counts, sum 22,418); table 3.6 their citizenship,
  national % only (China 41.7, Russia 11.6, Korea 9.9, USA 10.6, other 26.2).

`sources/mn_census.py` reads all of these straight from the PDF.

## Method and checks

Each aimag x group cell is printed twice. The seed takes whichever table is finer for that
cell (row % when the aimag is smaller than the group nationally: 63 cells; column % otherwise:
509), then a rake to aimag citizens (table 1.1 minus 3.7) and national group counts (4.1).
- The rake moved 1,023 people (0.03%).
- 23 of 572 cells end outside one of the two printed rounding intervals, worst by 100 people
  (Bayankhongor Khalkh); a two-margin rake cannot hit every interval.
- Aimag sums within 40 of table 1.1 (rounding plus the 37 stateless people, not drawn).
- The single-table version (3.6a only) failed the row-% check by 0.9 points (Bayan-Olgii's
  ~930 Khalkh print as 0.0% of all Khalkh), which is why both tables are used.

Foreign citizens: each aimag's count (3.7) split by the national citizenship shares; rows
`modelled`. Ethnic rows `derived`.

## Calls

- **Mongol subgroups by dialect group, not all on Mongolian.** Oirat (new leaf
  `mongolic.oirat`, Glottolog kalm1243): Durvud, Bayad, Zakhchin, Torguud, Uuld, Myangad,
  Khoshuud, Uriankhai, Khoton, Darkhad (Glottolog files Darkhat as Kalmyk-Oirat). Buryat:
  Buriad, Barga. Khamnigan Mongol its own leaf (kham1281). Everyone else on Mongolian. 289,000
  Oirat, 46,000 Buryat. Reversal is a one-line change per group in `taxonomy/mn2020.py`.
- Census "Uriankhai" is mostly the Altai Uriankhai (Oirat); Khovsgol's share are Darkhad-area
  Uriankhai, also drawn Oirat. Not split by place.
- Tsaatan (Dukha) on a new leaf `turkic.dukha` (dukh1234, under Tofa in Glottolog); 208 people.
- The 143-person unnamed Mongol remainder sits on `mongolic` (drawn "not named").
- Ulaanbaatar is one unit; religiondots' düüreg-keyed hexes are folded to MN11 in
  `countries/mn.py`. Nalaikh's Kazakhs are therefore spread over the whole city.

## Retention

No source found. The 2020 and 2010 censuses asked no language question; one web search
(2026-10-05) found no survey crossing ethnicity with home language. Nobody is moved onto
Mongolian. Kazakh in Bayan-Olgii is well kept (Kazakh-medium schools); Oirat and Buryat are
under heavy Khalkha influence, especially in the capital, so those dots are an upper bound
(said in `note_public`).

## Room for improvement

A home-language question, or any survey crossing ethnicity x language, would let urban Oirat,
Buryat and Kazakh speakers who now speak Khalkha be moved. Soum-level ethnicity (aimag volumes
other than Bayan-Olgii's, which prints none) would place Khovd's Kazakhs and Bayan-Olgii's
Tuvans and Uriankhai properly. Foreign citizens by aimag x citizenship would replace the
national split.
