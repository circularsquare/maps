# LiTS playbook

The EBRD Life in Transition Survey, round III (late 2015 to early 2016): one wave of about 1,500 adults in
75 PSUs of 20 interviews per country, across 32 transition economies plus Germany and Italy, with religion in
`q922` and place in `region_name`. It is the route for a post-Soviet country whose census asks no religion
and whose office holds no table, once its regional coverage is checked. LiTS IV has no religion question, so
the sample can never grow.

## Used by
- `kg` Kyrgyzstan: drawn, 9 oblasts of 9; Muslim and Orthodox at oblast share, six answers at the national
  rate inside each oblast's remainder.
- `uz` Uzbekistan: closed on LiTS (10 of 14 regions sampled), later drawn from the Central Asia Barometer
  (`playbooks/cab.md`).
- `tj` Tajikistan: closed; 5 of 5 regions, but zero Orthodox and a one-respondent Buddhist cell
  (`sources/tj.md` §3).
- `tm` is not a LiTS country. `kz` is drawn from its census; do not take LiTS for it (`queue.md` §B).
- `lits.held_out` is also run by `uz`, `tm` and `bo`, and `lits.lean` by `uz`.

## Loading it
- `data/raw/lits/lits_iii.dta` (170,487,994 bytes, 51,206 rows, 1,290 columns): an open EBRD download with
  no account, cookie or terms gate (URL in sources.md §11ag). `lits_iv_csv.zip` sits beside it and has no
  religion item.
- `lits.load(country_match)` reads `country`, `PSU_number`, `region_name`, `urban`, `weight_population`,
  `weight_sample` and `q922` with labels, matches `country` case-insensitively (Kyrgyzstan's label is
  `Kyrgyz Rep.`), and adds `region`, `code` and `w` (`weight_population`). If nothing matches it stops and
  lists the file's country labels.
- `district_l1`, `district_l2` and PSU coordinates are in the file and not read: about 20 respondents a
  district.
- `region_name` holds abbreviated Russian (`И-КУЛЬСКАЯ`, `Д-АБАДСКАЯ`, the two cities as `горкенеш`), so
  the decode is a written dict (`kg_geo.LITS_REGION`) with a witness (below).
- Copy `sources/kg.py` and `sources/kg_geo.py`. Run
  `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/kg.py`.

## Traps
- **The weights do not show a coverage hole.** Uzbekistan's 75 PSUs miss Fergana, Kashkadarya, Andijan and
  Surkhandarya, 37.8% of the country, and `weight_population` still sums to a plausible 20.4 million.
  Tabulate sampled units against the official list before anything else. Caught by: `lits.coverage` prints
  and returns the unsampled units; `kg.py::main` stops if there are any. Detail: spec §12 "A PORT SCAN THAT
  SAYS EVERY PORT IS OPEN IS A GEO-FENCE" (its last paragraph); `sources/uz.md` §5.
- **The card has eight substantive codes, not six.** The printed questionnaire (p41) shows six; the
  delivered `q922` labels add `BUDDHIST` and `CATHOLIC`, both used in Kyrgyzstan, and `Refusal` is -99.
  Caught by: `lits.load` (stops unless the `q922` labels are exactly those nine, in the file's order). Detail: `sources/lits.py`
  docstring.
- **LiTS IV is not a second round of the question.** Its only religious content is the identity battery
  `q811` (in Uzbekistan 274 of 1,006 tick "Muslim" as an identity that matters). Do not read it as
  affiliation or pool it. Caught by: `lits.load` reads only `lits_iii.dta`. Detail: `sources/uz.md` §5.
- **A bar for one Spearman correlation is the wrong test for a median of 400 halves.** `1.96/sqrt(n-1)` is
  +0.693 on nine units; Kyrgyz Islam (89% of the country) came in at +0.548 and would have failed, while the
  built null's 95th percentile is +0.450 (p 0.020). The null is per category and stricter where a category
  is thin (`OTHER` +0.750, `JEWISH` +1.000). Caught by: `lits.stability` builds the null. Detail:
  `sources/lits.py` docstring; `sources/kg.md` §4.1, §9.1.
- **`lits.stability` applies no chi-square veto and no largest-PSU check.** It predates Sweden's rule (a
  mostly-zero column can pass on how its ties break) and Uzbekistan's cluster rule. Kyrgyzstan's two passes
  are dense (1,331 and 107 respondents). Caught by: `lits.stability` (since 2026-09-14 it stops, rather than
  vetoes, on a pass whose spatial chi-square is not under alpha or with over half its respondents in one
  PSU, so no verdict moved). Detail: spec §12 "A
  RANK TEST CAN BE PASSED BY A COLUMN THAT IS MOSTLY ZERO", "A CHI-SQUARE CANNOT VETO A CLUSTER".
- **A category in one PSU gets no test and is drawn anyway.** Tajikistan's single rural `BUDDHIST`
  respondent would put more Buddhists on the map than Christians of any kind: `lits.stability` prints "no
  test possible" and `lits.build` spreads the category at its national rate. Caught by: `lits.stability` (stops
  on an untested category the caller does not list, with its reason, in `untested=`, and on a listed one
  that became testable; `kg.py::UNTESTED` lists Kyrgyzstan's one Catholic). Detail: `sources/tj.md` §3.
- **A zero where a known community lives is the instrument, not the country.** LiTS III returns 0 Orthodox in
  Tajikistan against 107 in Kyrgyzstan and 467 in Kazakhstan on the same design; at n=1,510 even 0.5%
  expects about eight. For each category the country is known to have, compute the count the smallest
  outside estimate implies; a zero where eight or more are expected condemns the source's whole minority
  half. Caught by: Not checked yet (belongs in the country file, beside `lits.national`). Detail: spec §12 "A
  CATEGORY THAT IS EXACTLY ZERO IN A SURVEY CELL IS A COVERAGE FAILURE".
- **A keying artefact looks like a small category.** Kyrgyzstan's 16 `BUDDHIST` answers sit in Osh (6) and
  Batken (4), the most uniformly Muslim oblasts, and 1 in Bishkek. Run the test on every category, not only
  the odd-looking ones, and give the national figure as a ceiling. Caught by: `lits.stability` (median
  -0.151, so national rate); `kg.py::main` asserts `CARRIES`. Detail: `sources/kg.md` §4.2.
- **The held-out check cannot separate units that sit within sampling error, and says so.** Issyk-Kul
  (7.49%) and Batken (8.17%) are 0.69 points apart against ±1.33 on n=1,500, and one of 362,879 orderings
  beats the truth. `lits.held_out` forgives a beating ordering only when every unit it moves lands within
  1.96 standard errors of its own population share (on the whole sample, not the unit's interviews), prints
  the pairs, and stops on anything else. Caught by: `lits.held_out`. Detail: its docstring; `sources/kg.md` §6.
- **The witness has to be on the survey's label decode, not on a neighbouring join.** Kyrgyzstan's SOATE-code
  and Russian-name witnesses pin the office to COD; swapping `И-КУЛЬСКАЯ` and `БАТКЕНСКАЯ` in the written
  dict passed the held-out check more cleanly than the truth (+0.9870 against +0.9866). Require each LiTS
  label to abbreviate exactly one COD Russian name, token by token. Caught by: `kg_geo.lits_decode_witness`
  (Kyrgyzstan only; a new country needs its own). Detail: `sources/kg.md` §9.5.
- **The split is on PSUs, so PSUs must nest in units.** Splitting rows would split a 20-household cluster
  and count it twice. Caught by: `lits._matrices` stops if a PSU spans two units.
- **Adults only, and the lean is measurable.** Kyrgyzstan's under-18 share against drawn Orthodoxy is r =
  -0.85 (leave-one-out -0.96 to -0.74), so an adult composition over all ages overdraws Orthodoxy. Report it
  with its range; never correct it. Caught by: `kg.py::age_lean` prints it from COD-PS age bands, not
  asserted. Detail: `sources/kg.md` §7, §9.6.
- **The refusal cell is one respondent.** A single non-zero unit gives no correlation. Caught by: `lits.lean`
  says no lean is measurable instead of printing a number. Detail: `sources/kg.md` §7.

## Shared code
Import these, do not copy them.
- `sources/lits.py`: `load`, `coverage`, `national`, `held_out`, `stability`, `build` (carried categories at
  their unit share, the rest sharing each unit's remainder at national proportions), `lean`.
- `lits.stability` today: 400 random halves of the PSUs, median Spearman of weighted shares across units (a
  unit missing from one half drops out of that halving only), a null that shuffles the PSU-to-unit labels
  400 times, pass at p < 0.05. Since 2026-09-14 it stops on a pass the chi-square or one PSU does not back,
  and on an untested category missing from `untested=`. `cluster_null` and `permutation_p` are
  `sources/stability.py`'s.
- `sources/spearman_null.py` is the exact null for one correlation that `lapop` and `arabbarometer` use. It
  is not the null for `lits.stability`'s median and was deliberately not adopted there.
- `kg.py::age_lean` and `kg_geo.lits_decode_witness` are Kyrgyz: copy them and name the source.

## Rulings
- **007-cr** (Anita, 2026-09-09): the split-half is a real 95% test, the exact null in place of
  `1.96/sqrt(n-1)`. It was about the single-correlation bar in `lapop` and `arabbarometer`; `lits.stability`
  already builds its own null for its own statistic at alpha 0.05, and the ruling neither changes that nor
  allows a looser level.
- No ruling is specific to LiTS. The Kyrgyz office's population figures are CC BY-NC-SA 4.0
  (`sources/kg.md` §5, §9.9), which would bind a sold print edition; that call is Anita's when it comes up.
