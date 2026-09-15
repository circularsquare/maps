# DHS and MICS playbook

Two household-survey programmes with the same shape. **MICS** (UNICEF) asks `HC1A` or `HC1`, the religion
of the household head, on the household questionnaire; offices field it under their own names (ENHOGAR-MICS6,
ENDESA-MICS, EMNA, LSIS). **DHS** asks `v130`, usually of women and men 15-49 and in some countries on the
household roster, with `v024` region. The route where an office's own MICS copy is open, or a DHS is
representative at a useful tier. Two MICS countries are built; **no DHS has ever been built here**, so the
DHS half of this playbook is a priced route, not a tested one.

## Used by
- `do` Dominican Republic: drawn, ENHOGAR-MICS6 2019, ONE's open microdata with `hhweight`, 32 provinces.
- `hn` Honduras: drawn, ENDESA-MICS 2019, INE's open zip with no weights (rebuilt from the report), 18
  departments.
- `la` Laos: drawn from the census; LSIS 2011-12 and 2017 (`HC1A`, with `Animist` and `No religion`
  separate) put the census "No religion" box at about 99.8% animist, unweighted. Redraw ruled, listed in
  `queue.md`.
- `pa` Panama and `cr` Costa Rica: drawn from LAPOP. MICS 2013 (`HC1.A` asked of every household member)
  and EMNA 2018 (MICS6, representative at 7 provinces) would replace them; Anita downloaded Panama's file on
  2026-09-09 and neither is built.
- DHS, priced and not built: `cd` (26 provinces, the only person-level route, `queue.md` §D); `pg` (DHS
  2016-18, 22 provinces, ages 15-49, 21.3% `Other Christian church`, ranked below IPUMS); `ht`, `bi`, `lr`
  gated. `ng`, `tz`, `ug`, `mz`, `gn`, `zm` were drawn without it. No religion item in Türkiye 2018
  (`sources/tr.md`), Jordan's JPFHS (`sources/jo.md`), Morocco 2003-04 or Mauritania 2019-21; the DHS API
  lists no Eritrean datasets at all.
- `uz` uses the five DHS 1996 survey regions as a coarse design grouping, not DHS data.

## Loading it
- **Look for the office's own open MICS copy first.** `do`: `data/raw/do/mics6_2019_{hogares,miembros}.csv`.
  `hn`: `python sources/hn.py --fetch` unpacks `BasesdatosENDESA2019.zip` into `data/raw/hn/endesa/Bases de
  datos/` (`hh.sav` households, `hl.sav` roster). Read `.sav` with pyreadstat and check the value labels
  against the country's `CATEGORY` before anything else.
- **Free views that need no microdata.** A NADA catalogue's per-variable summary gives a weighted national
  distribution (`cr`: `/api/catalog/IDD-CRI-INEC-EMNA-2019/variable/V23`); World Bank microdata variable
  pages give unweighted counts (`la`: `catalog/1911/variable/V45`, `catalog/3401/variable/V774`). The final
  report's questionnaire annex shows whether the item exists.
- **UNICEF's MICS copies** (with `hhweight`, PSU and stratum) sit behind `mics.unicef.org`. Anita has the
  account (`ask/answered/006-pa`) and does the downloading. Keep files under `data/raw/<cc>/`, which is
  gitignored with all of `data/`, and never pass them on.
- **DHS recode files** need a DHS Program registration with a project title and an analysis description,
  reviewed in 24-48 hours. Individual researchers can register, and the "applications suspended 2025-02-07"
  claim is false (checked 2026-09-09, sources.md §11ag item 4). Redistribution is barred. The registration
  is Anita's (`AGENT_BRIEF.md`: anything needing an account). IPUMS-DHS is blocked with IPUMS.
- There is no shared module. Copy `sources/hn.py` for an open copy without weights, `sources/do.py` for one
  with `hhweight`. Run `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/<cc>.py`.

## Traps
- **It is the household head's religion, drawn for everyone in the household.** A Catholic head's
  unaffiliated children are drawn Catholic. The Dominican Republic priced it with ENHOGAR 2018's `AD118`
  (women 15-19, their own religion): Catholic 49.8% as drawn against 37.1% self-reported, no religion 21.6%
  against 34.8%, evangelical within 0.06 points; a level shift with the ranking kept (r 0.87 over 32
  provinces). Look for an individual-religion module in the same series, cut the comparison to its universe,
  and say it in `note_public`. Caught by: Not checked yet (`do`'s figure was a review run; nothing in
  `hn`'s files can price it). Detail: spec §12 "A HOUSEHOLD-HEAD RELIGION QUESTION CAN BE PRICED";
  `sources/do.md` §7.1; `sources/hn.md` §1.
- **An office's copy can have no weight, PSU or stratum.** Rebuild household weights from the report's
  weighted and unweighted households by domain (Honduras Tabla SR.3.1) and frame areas by domain and area
  (Tabla SD.1), with one national rural factor, and test on SR.3.1 rows the fit never saw. Assert the
  unweighted column against the microdata first, and measure what the weights can move (at most 2.3 points
  on any Honduran department share). Caught by: `hn.py::rebuild_weights` (transcriptions against `hh.sav`,
  SD.1's own totals, witness rows better than no weights). Not checked: `HH6` is not the design stratum.
  Detail: spec §12 "A MICS REPORT PRINTS ENOUGH TO REBUILD THE WEIGHTS"; `sources/hn.md` §2, §8.
- **The report prints the question and no table of it.** Panama's MICS 2013, Costa Rica's EMNA 2018,
  Honduras and the Dominican Republic all asked and tabulated nothing. DHS final reports put religion in
  Table 3.1 beside region, never crossed (Nigeria FR359: one of 734 captions), and the DHS API's 4,655
  indicators hold no religion composition. Grep the questionnaire annex for `religi` before calling a
  country closed. Caught by: a search rule. Detail: `sources/pa.md` §1 "MICS 2013"; `sources/cr.md` §1
  "EMNA 2018"; sources.md §11ag item 4.
- **DHS universes differ by country.** Papua New Guinea asks women and men 15-49; Nigeria asks on the
  household roster. Only a roster item is a population composition. Appendix A of the final report gives
  the tier and the questionnaire gives the universe and the card, both free. Caught by: Not checked yet (no
  DHS loader exists; the assertion belongs there). Detail: spec §12 "A GATED INSTRUMENT'S CATEGORY LIST AND
  ITS TIER ARE BOTH FREE TO READ".
- **Put the card's residual beside the census's before valuing the route.** Papua New Guinea's DHS card has
  11 codes and no Baptist code, and its `Other Christian church` is 21.3% against the census's 9.7%. Caught
  by: Not checked yet. Detail: the same spec §12 entry.
- **Completed households, roster size and city domains.** `HC1` must be answered for exactly the completed
  households (`HH46`), `HH48` must equal the roster lines in `hl.sav`, and Honduras's `HH7` codes 19 and 20
  are city domains to merge into their departments. Caught by: `hn.py::load`; `do.py::load` (completed
  interviews equal answers, every person has a head's answer, and HC1A against the file's recoded `religion`).
  Detail: `sources/hn.md` §1.
- **The card changes between offices and rounds.** Honduras names Mormons; Costa Rica's EMNA folds
  evangelical, Pentecostal and Mormon into one box; LSIS offers `Animist` beside `No religion`. Caught by:
  `hn.py::check_labels`, `do.py::check_labels` stop on a changed label set. Detail: `sources/cr.md` §1.
- **A one-round split-half is held to a bar built for one correlation.** `hn.py::stability` takes the median
  of 400 cluster halvings drawn inside each department and compares it with `1.96/sqrt(17)` = +0.475;
  `do.py::stability` uses one cluster-parity split against `1.96/sqrt(N_PROVINCES - 1)`. One split is a draw
  (parity alone moved the Dominican Witnesses by 0.18), and a median has a narrower null than that bar
  (Kyrgyzstan +0.450 against +0.693), so both err strict in the way `ask/answered/007-cr` ruled against.
  Split on clusters, never persons. Caught by: Not checked yet (a built null belongs in
  `sources/stability.py`). Detail: spec §12 "ONE SPLIT-HALF IS A DRAW"; `sources/lits.py` docstring.
- **The vetoes beside the rank test.** Chi-square across units on unweighted households (in `do` it is a
  print, not a licence: persons, clustering ignored); no single cluster holding half of an answer; a failing
  answer keeps its share in a unit that tops both halves in 95% of halvings (Islas de la Bahía's
  Adventists, 400 of 400). Caught by: `hn.py::stability` (`CLUSTER_CAP`, and `CARRIES` and `STANDOUTS`
  asserted). Detail: spec §12 "A RANK TEST CANNOT SEE ONE UNIT STANDING APART"; `sources/hn.md` §4.1.
- **Failing answers take the residual or go flat, by the 2x rule.** Honduras's residual drew Latter-day
  Saints at 2.96x their national share in Gracias a Dios, where the survey found none, so every failing
  answer went flat with the carried shares scaled. Test after the standouts are taken out. Caught by: Not
  checked yet in `hn.py` (the 2.96x was a read-only re-run; `tz.py::compose` is the pattern that asserts
  `TAIL_FLAT`). Detail: spec §12 "SMALL CATEGORIES GO IN THE RESIDUAL UNLESS IT DRAWS ONE AT 2x";
  `sources/hn.md` §4.2, §9.
- **A MICS split source is unweighted, about heads, and not proportional.** LSIS's animist share of heads
  (37-38%) runs above the census's 31%, and its `Other religion` may also sit inside the census cell. A
  weighted split needs the UNICEF file; LSIS III 2023 is unchecked. Caught by: Not checked yet. Detail:
  `sources/la.md` §10.

## Shared code
Nothing DHS- or MICS-specific is importable.
- `hn.py::stability` and `do.py::stability` are the split-half as it stands (above); `hn.py::rebuild_weights`,
  `hn.py::held_out` and `hn.py::population_witness` are the weight and join checks. All read module globals:
  copy them and name the source. `sources/stability.py` lists both as different methods and left them in place; `hn.py` uses its
  `chi2_p` and `top_both_halves`.
- `sources/lits.py::held_out` suits a small unit count; `tz.py::compose` has the 2x tail rule.

## Rulings
- **006-pa** (Anita, 2026-09-09): she registered with UNICEF MICS. The terms bar editing, distributing or
  sharing the datasets in any form, and bind use to the registered objective, a public dot map of aggregate
  counts with the data on her machine and UNICEF/MICS credited. A new purpose, such as a sold print edition,
  needs an email to `mics@unicef.org` first. Does not decide whether `pa` or `cr` get rebuilt, and does not
  cover DHS.
- **Laos** (Anita, 2026-09-14 night, `queue.md`): draw the census "No religion" box as traditional religion
  on LSIS's ~99.8% animist, cite LSIS in the mapping `REVIEW`, and follow the draft procedure. Does not
  decide the weighted split or whether `indigenous.laos` is revived or replaced.
- **Draft "no religion" procedure** (`WORKFLOW_PLAN.md`, written at Anita's request): names MICS/LSIS and DHS
  as split sources; a box is drawn as one reading when that reading is 80% or more of it. A draft, not yet
  in spec.
- **DHS**: no ruling. The registration needs her identity (`queue.md` §D).
