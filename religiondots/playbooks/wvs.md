# World Values Survey playbook

WVS wave 7 (2017-2022): one national round of about 1,000 to 3,000 adults per country, religion in `Q289` on
a country card with a harmonised detail code `Q289CS9`, and two place columns, `N_REGION_ISO` and
`N_REGION_WVS`. It is a route only where nothing better asks and the region column is a real tier. One
country is built from it, so most of this playbook is the Puerto Rico build plus recorded negatives.

## Used by
- `pr` Puerto Rico: drawn, WVS-7 2018, 1,127 adults, 6 regions with 3 municipios sampled in each; Católico
  and Otros at region share, the other four answers in each region's remainder.
- `ad` Andorra: drawn, WVS-7 2018, 1,004 adults, one unit, every answer at its national share, read
  from the IHSN catalogue's frequency pages with no download (`sources/ad.md`).
- Checked and not used: `jp` (WVS-7 2019: `N_REGION_ISO` is prefecture, 45 of 47, `N_REGION_WVS` 5 blocks,
  no Shinto code; `ask/answered/014-jp`); `tr` (the wave 7 denomination list has no split, `sources/tr.md`);
  `pk bd id ng eg my` (the online tool shows one `Islam; nfd` code each, no sect, `sources/branches.md`);
  `uy` (national only).
- Listed and not opened: `bo` (2017, 9 departments), `ve` (2021, 22 states), `co` (2018, 26 codes), all in
  sources.md §11ap; `ir` wave 7 (a province code may exist, but about 1,500 over 31 provinces would not clear
  a split-half); `az` wave 6; Puerto Rico's waves 3 (1995) and 4 (2001).
- Checked and not used: `lb` Lebanon, wave 7 (2018), 1,200 citizens, Anita's Stata download. The sample design
  sets every cluster's sect (Statistics Lebanon Ltd., the Arab Barometer's firm), so the sect mix by governorate
  is an allocation (`sources/lb.md` §9, `sources/lb_wvs.py`).

## Loading it
- Country files come from the WVS wave 7 download page: non-profit use, publications cited and reported to
  the WVSA (terms as read for Japan, `ask/answered/014-jp`). Anita made Puerto Rico's download; sources.md
  §11ap treated the form as hers.
- The online analysis tool `worldvaluessurvey.org/WVSOnline.jsp` needs no registration and crosses `Q289CS9`
  with `N_REGION_WVS` in unweighted counts, which prices a country before any download. It is
  JavaScript-driven; read it with headless Chrome over CDP (`sources/branches.md`, Pakistan section). Its
  frame is `AJOnline.jsp?WAVE=7&COUNTRY=<ISO numeric>`, a chain of JSP form posts with no single crosstab
  URL, and curl needs `-k` (missing intermediate certificate).
- Puerto Rico: `data/raw/pr/F00013157-WVS_Wave_7_Puerto_Rico_Csv_v5.1.zip`, one semicolon-delimited CSV
  with a BOM. The codebook `F00011055-WVS-7-Codebook-Variables-report` has the code lists (the `N_REGION_ISO`
  annex, `N_REGION_WVS`, `Q289CS`). The survey team's national report carries the design, the region map,
  the questionnaire and a national religion table (`sources/pr.md` §1); find the equivalent before building.
- **The IHSN catalogue prints each WVS-7 country file's unweighted frequencies, with no form.**
  `catalog.ihsn.org/catalog?sk=World+Values+Survey+<Country>` finds the entry; each
  `/catalog/<id>/variable/F1/V<n>?name=<VAR>` page has the category table, and the V numbers
  differ by country (Andorra's Q289 is V338, Romania's Q289 V337), so take them from
  `/catalog/<id>/data-dictionary/F1?offset=0&limit=600`. The entry's related materials carry the
  national questionnaires, methodology report and sample design. One variable at a time, so a
  national share and a region's interview count, never religion by region. For a one-unit country
  that is the whole source (`sources/ad.py::read_page`, which asserts the page's variable and file).
- There is no shared reader. `pr.py::load`, `attach_geography` and `stability` are the only WVS code; a
  second WVS country should lift them into a module rather than copy them a second time.
- Run `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/pr.py`.

## Traps
- **A trailing delimiter shifts every column by one, and every column still looks plausible.** The header
  has 404 names and each row 405 fields; pandas takes the first field as an index, so `A_YEAR` reads 630 and
  `Q289` reads the eight-digit codes. Compare the header's field count with a row's, read with
  `index_col=False`, and assert columns that can only hold themselves (`doi`, `A_YEAR`, `B_COUNTRY`).
  Caught by: `pr.py::load`. Detail: spec §12 "A TRAILING DELIMITER SHIFTS EVERY COLUMN BY ONE"; `sources/pr.md` §2.
- **No weights, no PSU, no interviewer.** `W_WEIGHT`, `S018` and `PWGHT` are constant, `I_PSU` is 0 or -4,
  and `D_INTERVIEW` is a serial that does not carry the municipio. The finest resampling unit is the
  municipio in `N_REGION_ISO`. Caught by: `pr.py::load` stops if `W_WEIGHT` is ever not 1. Detail:
  `sources/pr.md` §1.
- **`N_REGION_ISO` is neither ISO nor FIPS.** For Puerto Rico it is 630000 plus the municipio's place in the
  alphabetical list of 78. Caught by: `pr_geo.py::main` asserts it for all 18. Detail: `sources/pr.md` §3.
- **The regions may exist only as a picture in the team's report.** Puerto Rico's six are one colour map on
  p.16, and no official six-region grouping exists (the Planning Board's has 11 functional areas). Read the
  border municipios at native resolution, then pin the reading with the report's interview count per region
  and per municipio (six distinct region totals), the file's second region column, and geography that uses
  no names. Caught by: `pr.py::attach_geography` (counts; `N_REGION_WVS` against the municipio's region with
  exactly the one known slip in `REGION_SLIPS`); `pr_geo.py::main` (westernmost in Oeste, easternmost in
  Este, Norte north of Sur, Metropolitana densest). Detail: `sources/pr.md` §3, §10.
- **Sample share against population does not pin the join.** The allocation is called proportional, but
  Centro holds 1.54x its adult share (the report prints the same 187 interviews), and with four regions of
  near-equal size 37 of 720 orderings reach the observed r. Caught by: `pr.py::held_out` asserts that
  pattern, so a changed file is noticed. Detail: `sources/pr.md` §3.
- **`Q289CS9` harmonises away the write-ins.** Puerto Rico's `Otros (escribir)`, 20.3% of answers, is coded
  80000000 `Other Christian; nfd` with the text withheld, and the card is the WVS template with no
  evangelical box. It goes to the `christianity` root, not `christianity.other` and not evangelical. Every
  Muslim-majority country gets one `Islam; nfd` code. Caught by: `pr.py::load` asserts `Q289` one to one
  with `Q289CS9` and the frequencies against the report's Tabla 80; the node is a review call
  (`taxonomy/pr2018.py` `REVIEW`). Detail: `sources/pr.md` §5, §10, §11.
- **Clusters nest inside the drawn units, so the null regroups clusters.** ESS's within-round label shuffle
  has no analogue here. Halve each region's municipios one against two over every distinct halving
  (23,328), and for the null deal the 18 municipios into random groups of three (95th percentile +0.49 to
  +0.54, against the formula bar's +0.877, which is printed only). Caught by: `pr.py::stability`, with
  `CARRIES` asserted. Detail: spec §12 "WHERE THE SAMPLING UNITS NEST INSIDE THE DRAWN UNITS, THE NULL
  REGROUPS THEM"; `sources/pr.md` §6.
- **A small answer can sit in one municipio.** `Budista` passed at p 0.007 with 3 of 5 respondents in San
  Juan. Refuse where one municipio, or one interview day in one municipio (the nearest the file has to an
  interviewer), holds half an answer. Caught by: `pr.py::stability` (`CELL_CAP` 0.5). Detail:
  `sources/pr.md` §6.
- **A carried answer can rest on two clusters.** Centro's 64% Catholic is Corozal 71% and Naranjito 76%
  against Cayey 40%, and dropping any one of four municipios takes Católico's between-region F test over p
  0.05. Do not quote a region figure in `note_public` that two municipios make. Caught by: Not checked yet
  (a drop-one-municipio print belongs in `pr.py::stability`). Detail: `sources/pr.md` §10, §11.
- **One round leaves most answers failing, and the tail construction has a rule.** Failing answers share
  each unit's remainder at national proportions unless that draws one at 2x its national share or more in
  a unit where the survey found none; then all go flat. Puerto Rico is at 1.20x. Caught by: Not checked yet
  in `pr.py` (`pr.py::compose` only prints `ABOVE NATIONAL WHERE NONE FOUND` at 1x; `tz.py::compose` is the
  pattern that tests 2x and asserts `TAIL_FLAT`). Detail: spec §12 "SMALL CATEGORIES GO IN THE RESIDUAL
  UNLESS IT DRAWS ONE AT 2x WHERE THE SURVEY FOUND NONE".
- **The card's write-in is harmonised differently by country.** Puerto Rico's `Otros (escribir)` is
  `Q289CS9` 80000000 `Other Christian; nfd`; Andorra's code 8 `Altra, quina?` is held as `Q289` code 9
  and `Q289CS9` 90000000 `Other; nfd`, so it goes to `other.<cc>`, not the Christianity root. Read
  `Q289CS9` before mapping Other. Caught by: `sources/ad.py::check_survey` (`CS9_OF`, one to one);
  the node is a review call (`taxonomy/ad2018.py` `REVIEW`). Detail: `sources/ad.md` §3.
- **Interviews per region can depart from the design note's allocation.** Andorra's note says
  proportional to parish population; `N_REGION_ISO` puts La Massana and Ordino at 8.2% of interviews
  against 20.0% of residents. With no weight, the national share carries any difference between
  parishes. Print interviews against population per region before trusting an unweighted national
  share. Caught by: `sources/ad.py::check_survey` prints it; not asserted. Detail: `sources/ad.md` §4.
- **A stratum can be a sect, and one round cannot show it by agreement.** Lebanon's Sample Design (an IHSN
  related material, a table image on pp.3-4) gives every cluster a sect; the file's 120 `I_PSU` clusters of 10
  are each one community and equal that table in all 23 kadaa, while Christian denominations mix inside them.
  Lebanon came as Stata, with real `I_PSU` values and `N_TOWN` as the caza, unlike Puerto Rico. Before trusting
  a region's composition, read the country's sample design and methodology report (Q14 "profile required",
  Q15 quota controls, Q20 stratification factors), and where `I_PSU` is filled, count clusters pure on each
  religion answer against a finer answer that mixes. Caught by: `sources/lb_wvs.py::psu_purity` and `::main`
  (the design table asserted). Detail: `sources/lb.md` §9; spec §12 "ONE ROUND CANNOT SHOW A QUOTA BY AGREEMENT".

## Shared code
Nothing WVS-specific is importable yet.
- `sources/lits.py::held_out` (exhaustive on few units) and `lits.lean` fit a WVS country as they stand.
- `pr.py::stability` today: municipio halvings within region over every distinct halving, median Spearman
  across regions, a null that deals municipios into random groups of the same sizes 400 times, the spatial
  chi-square under 0.05 as a veto, `CELL_CAP` on municipio and municipio-day, and Honduras's standout test at
  95% of halvings. It reads module globals, so copy it and name the source; its null, p-value, chi-square and
  standout pieces are `sources/stability.py`'s (`cluster_null`, `permutation_p`, `chi2_p`, `top_both_halves`).

## Rulings
- None on the WVS itself. Whoever downloads accepts its terms, and the Puerto Rico download was Anita's.
- Open, not a ruling: `ask/019-pr`, Puerto Rico's two adjacent pale yellows. The palette is Anita's; do not
  recolour it.
