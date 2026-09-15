# European Social Survey playbook

ESS (rounds 1-11, 2002-2024) has an open GraphQL API that cross-tabulates religion by region
server-side, so no microdata moves. It is the route for a European country whose census does not
ask religion, once the office's commissioned-table shelf has come back empty.

## Used by
- `be` Belgium: drawn, rounds 5-11 citizens at 11 provinces plus a census x Pew foreign half.
- `se` Sweden: drawn, rounds 5-8 at 21 län and all six rounds at 8 riksområden, per category.
- `no` Norway: drawn, rounds 5-9 at 7 regions plus 4 shared units, rescaled to rounds 10-11.
- `dk` Denmark: drawn, rounds 5-7 and 9 at 5 regions, round 9 recoded and on `dweight`.
- `lv` Latvia: drawn, rounds 4 (`regionlv`), 9, 11 at 6 regions, non-citizens from ESS, rescaled.
- `ua` Ukraine: drawn, rounds 2-6 for all residents at 26 oblasts, round 11 a witness; ask 016.
- `gr` `fi` `fr` `it`: drawn before the split-half existed; it runs on them as a report only.
- `de` Germany: ESS splits the register's residual by Land (`de_ess.py`, `countries.py::_de_split_ess`).
- `nl` Netherlands: priced for ESS, drawn from a CBS maatwerk table. `lu`: closed, rounds 1-2 only.

## Loading it
- **Office first.** Look for a commissioned-table shelf and on the customer's site (Sweden's church
  PDF was SCB's work). queue.md "Europe via ESS" lists the shelves; `nl` was won that way.
- **API.** `https://api.nsd.no/graphql`, anonymous. Template `no.py::_TAB` (`%s` is the weight
  clause): `frequencyTabulationByVariables`, `byVariables:["cntry"]` (without it `region` hits
  `E204TooManyCategoriesInVariable`), `includeMissing:true`, `breakVariables` as names typed
  `[String!]!`. Take your country's response by `by[0].value`. `sources/gr.md` §1.
- **Datafiles** are `(id, version)`: `be.py::ESS_ROUNDS` (5-11), `ua.py::ESS_FILES` (2-6, 11),
  `lv.py::WITNESS_FILES` (round 3's separate `ess3lv` file). `search.searchDatafiles` is gone;
  `search.seriesMetadata` lists every round (spec §12 "ESS's `searchDatafiles` IS GONE").
- **Per round, two passes** of one break (region variable, `ctzcntr`, `rlgblg`, card):
  `data/raw/<cc>/ess_r<N>_n.json` unweighted and `ess_r<N>_w.json` on `pspwght`. Also the card
  against `rlgdnm`, and region against `domicil` and home language, for the checks below.
- **Foreign half.** Eurostat `cens_21ctz_r3` (EU-wide; `geo` holds every NUTS level, read the
  leaves) x Pew 2020 compositions. Needs a current `certifi` (`gr.py::_eurostat`).
- **Run** `python sources/<cc>.py --fetch`, then `python sources/<cc>.py`. `sources/dk.py` is the
  simplest one-level loader to follow, `sources/no.py` the two-level one.

## Traps
- **`table.path` indexes `codeList`; it is not a code value.** Read as codes, every category whose
  code is not a valid index goes to zero silently (Finland lost Islam). Caught by: the both-ways
  category check in `fi.py`/`be.py`/`se.py::_citizen_shares`, `no.py`/`dk.py`/`lv.py::_assert_source`;
  `gr.py`, `fr.py`, `it.py`, `ua.py` check one way only. Detail: `sources/fi.py::_ess_table`.
- **Pool `region` on its code, everything else on its label.** Region labels change alphabet
  (Greece round 5 Latin, 10-11 Greek) and punctuation (France), and pooled on labels every unit
  splits in two. Ukraine's `regionua` has names only. Caught by: `be.py::_check_regions`,
  `se.py`/`no.py`/`dk.py`/`lv.py::_check_level`, `ua.py::_unit_of`. Detail: `sources/gr.md` §2.
- **Codes change NUTS vintage between rounds.** Recode to one: `gr.py::GR_TO_EL`,
  `fr.py::FR10_TO_16`, `fi.py::RECODE` (three vintages). Caught by: `fi.py::_check_recode` (same
  label both sides); `gr.py::_ess_table` (round 5's GR labels against NUTS2's names for the EL codes,
  every round inside NUTS2); France's `FR10_TO_16` is not checked.
- **Two vintages can cross rather than nest.** Norway's rounds 5-9 (7 regions) and 10-11 (6) share
  only 4 units, where the null's 95th percentile is +0.8. Count shared units before planning two
  levels. Caught by: `no.py::_check_level` per round; the crossing itself Not checked yet. Detail:
  spec §12 "TWO NUTS VINTAGES CAN CROSS".
- **`regunit` changes level inside one country.** Italy NUTS 2 in rounds 6, 8 and NUTS 1 in 9-11;
  Sweden NUTS 3 in 5-8 and NUTS 2 in 9, 11. Pooled, they average. Caught by: `se.py::_check_level`,
  `it.py::_pooled_shares`. Detail: queue.md "Europe via ESS".
- **Rounds 1-4 have no `region` but can have `region<cc>`.** Latvia's `regionlv`, Ukraine's
  `regionua`, older vintages, recode by label. `gr`, `se`, `be`, `dk` closed rounds 1-4 unprobed.
  Caught by: Not checked yet (probe in each `fetch()`). Detail: spec §12 "ESS ROUNDS 1-4".
- **A round can publish right codes on wrong regions.** Denmark round 9's DK01-DK05 are Danmarks
  Statistik's 1081-1085 order and `pspwght` was raked to them, so only unweighted sample shares
  show it. Compare each round's unweighted share and `domicil` per region with the others, recode,
  assert both ways, use `dweight`. Caught by: `dk.py::_check_recode`, `lv.py::_check_labels`,
  `ua.py::_check_labels`; missing in the other eight. Detail: spec §12 "ONE ROUND CAN PUBLISH".
- **A round's region can carry no geography.** Latvia round 10: all regions alike, nothing to
  recode; use it nationally, assert it still fails. Caught by: `lv.py::_check_labels`.
- **Card variables vary by country and round.** `rlgdnfi`/`rlgdnse` do not exist (`rlgdnafi`,
  `rlgdnase`); `rlgdnbe` is rounds 5-9, `rlgdndk` round 5; `rlgdnua` (4-6) and `rlgdnaua` (11)
  are different cards. A missing variable or a UUID is HTTP 400 `E201VariableNotFound`. Tabulate
  both cards for a round, then choose. Caught by: `be.py::_check_be_card`,
  `no.py`/`dk.py`/`lv.py`/`ua.py::_check_card`, `be.py::N_ROUNDS`, each loader's `N_CITIZENS`.
  Detail: spec §12 "A survey variable can be revised under the same name".
- **Other variables rename by round.** Latvia: `scrlgblg` in round 10 (both "No, ..." answers are
  No), `lnghoma`/`lnghom1`, `ctzshipb`/`ctzshipd` (alien's passport `65`/`6500`), `dweight` only in
  round 4. Keep per-round maps (`lv.py::REGION_VAR`, `BLG_VAR`, `WEIGHT_VAR`, `CTZSHIP_VAR`). Caught
  by: `lv.py::_check_level` (answers exactly Yes/No), `lv.py::_noncitizens` (alien label).
- **Card `Not applicable` is everyone who belongs to nothing, and is flagged `isMissing`.** Filter
  on `isMissing` and the unaffiliated vanish. Build from `rlgblg` x card (`be.py::_category`) or map
  it to `unaffiliated` (`taxonomy/gr2024.py`). Caught by: the both-ways check above.
- **Weighted cells are not respondents.** `pspwght` totals n; counts, chi-squares and the
  split-half take `_n`, shares take `_w`. `gr`, `fr`, `it` fetched only `_w`. Caught by: Not checked
  yet, and a whole-count assertion cannot do it: the API returns weighted cells as whole numbers too
  (no fractional cell in any cached `_w` file of ten countries, 2026-09-14), so only the file name
  tells them apart.
- **ESS under-reaches non-citizens and language minorities.** Greece: 3% non-citizen against 7.2%,
  no Muslim citizens in rounds 10-11 beside a Thracian minority of about 110,000. Draw citizens
  (`ctzcntr = Yes`) plus the foreign half; an unreached minority is an authored split
  (`gr.py::THRACE_MINORITY`); Germany does not rescale Muslims to BAMF (`de_ess.py` docstring).
  Caught by: the construction. Detail: `sources/gr.md` §3.
- **Eurostat `FOR` can include the country's own non-citizens.** Latvia's 190,544 `RNC` are inside
  it; scaling to `FOR` multiplies Russia fourfold. Target `FOR - RNC`, draw `RNC` from `ctzship*`.
  Caught by: `lv.py::_foreign_half`; `be`, `dk`, `gr`, `no` and `se` `::_foreign_half` stop on any
  `RNC` and on named citizenships outside Latvia's coverage band. `fi`, `fr` and `it` still only
  print (their `RNC` is 0, 2026-09-14). Detail: spec §12 "EUROSTAT'S `FOR`".
- **Countries and units drop out of rounds.** Sweden has no round 10, Denmark no 8, 10, 11, Ukraine
  none of 7-10, and Ukraine's 2-5 each skip 2-4 oblasts. A unit empty in one half drops that
  halving in `stability.median_rho`, and different ones from the null. Assert who each round sampled
  and test on units in every round. Caught by: `no.py::_stability`, `be.py::_stability` and
  `se.py::_stability` (stop on an empty (round, unit) cell), `ua.py::_check_regions`
  (`EXPECT_ABSENT`, `TEST_UNITS`). Detail: spec §12 "A ROUND THAT SKIPS A UNIT".
- **The null permutes unit labels per round.** One global relabelling is the identity: p = 1 for
  every category. Print the null's 95th beside the statistic. Caught by: `stability.py::wave_null`,
  which relabels per round for every ESS caller (`tools/test_stability.py::check_wave_null` tests it).
  Detail: spec §12 "A PERMUTATION NULL".
- **One halving is a draw.** Sweden's church scored +0.125, +0.434, +0.458 on the three halvings of
  four rounds; take the median over all. With an odd count every halving is distinct, and the old
  `if 0 in a` filter kept only some. Caught by: `stability.py::halvings`, which every ESS caller
  uses (`tools/test_stability.py` checks it against brute force). Detail: spec §12 "ONE SPLIT-HALF
  IS A DRAW", "EVERY DISTINCT HALVING".
- **A mostly-zero column can pass the rank test.** Sweden: n = 21 and 23 passed at p 0.018 and
  0.022 with chi-squares of 0.32 and 0.40. Require both at 0.05; no size floor (`se.py` comment).
  `be.py::_stability` has no chi-square. Caught by: `no.py::_stability`, `se.py::_stability`,
  `tools/ess_split_half.py::run`. Detail: spec §12 "A RANK TEST CAN BE PASSED".
- **Few units hide one unit standing apart.** Denmark's Islam: p 0.32, chi-square 4e-05, the
  capital alone. Keep the rule unless an independent witness orders the units (Norway's roll,
  Latvia's round 3); then `OVERRIDE` with the reason printed. Caught by: `EXPECT_PASS` (`dk`, `lv`),
  `EXPECT_FINE_PASS` (`no`), `EXPECT_OBLAST_PASS` (`ua`); `ua.py::_standouts`. Detail: spec §12 "A
  RANK TEST OVER FEW UNITS", "A RANK TEST THAT MISSES ON FEW UNITS".
- **The national-rate residual can reverse a geography.** Latvia drew Latgale 3.9% Orthodox against
  11.0% measured. Print each residual category beside the survey's share. Caught by: printed, not
  asserted, in `lv.py::_compose`, `dk.py::_compose`, `ua.py::_citizen_like_shares`. Detail: spec
  §12 "THE RESIDUAL CONSTRUCTION CAN REVERSE".
- **With nested levels the big category stays the residual.** Fixing Sweden's `No religion` at
  riksområde share sends the tail negative in 6 of 21 län; it goes in `KEEP_AS_RESIDUAL`. Caught by:
  each `_compose` prints the negative count and exits on no room. Detail: spec §12 "NESTED UNITS".
- **A pool spanning fast change draws a stale level.** Scale each drawn category to the late
  rounds' national share, weighted by census citizens per unit, when it moved over 3.5 points
  (Norway's church 44.1% to 32.2%) and the late rounds reach every unit (Ukraine's round 11 cannot).
  Caught by: `no.py::_compose`, `lv.py::_rescale`, `ua.py::_late_check` (prints `DRIFT_BAR`);
  coverage Not checked yet (belongs in `lv.py::_rescale`). Detail: spec §12 "A SURVEY POOL THAT
  SPANS A FAST CHANGE", "A LATE ROUND THAT CANNOT REACH".

## Shared code
Import these, do not copy them.
- `sources/no.py::_stability`: the current test (`stability.py`'s `halvings`, `median_rho`,
  `wave_null`, `permutation_p` and `chi2_p`, at `be.py::STAB_ALPHA`/`STAB_PERM`/`STAB_SEED`),
  used by `dk`, `lv`, `ua`. `se.py::_stability` computes the same through `sources/stability.py`;
  `be.py::_stability` does too but has no chi-square, so do not copy it.
- `sources/no.py::_TAB`, `_save`, `_eurostat`, `ESS_API`, `PEW_ZIP`.
- `sources/be.py::_ess_soft`, `ua.py::_ess_try`: return None on `E201VariableNotFound`.
- `sources/ua.py`: `_standouts` (spec §12 Honduras), `CLUSTER_REFUSE` (refuse a pass with over half
  its respondents in one round-and-unit cell, spec §12 Uzbekistan) and the 2x-rule exit.
- `taxonomy/origin_religion.py::composition`, `PEW_BY_ISO`, `REGIONAL`, `FAMILIES`. `_census` and
  `_foreign_half` have no shared home yet; take `lv.py`'s, which has the `RNC` guard.
- `tools/ess_split_half.py` (`tidy`, `splits_for`, `run`): report only, the build reads nothing.

## Rulings
- **Ask 012 (answered):** split-half plus chi-square on `gr`, `fi`, `fr`, `de`, `it` as a report in
  each `sources/<cc>.md`; no dots move. Does not decide whether they are ever rebuilt on it.
- **Ask 013 (answered, deferred):** Greece stays as built, dots and `note_public`. Does not pick
  one of its three options; not for an unprompted session.
- **Ask 016 (open):** Ukraine's Orthodox as one node, occupied oblasts from pre-war rounds, no
  rescale to round 11. Leave `ua` as built until answered.
- **queue.md, 2026-09-14 evening:** probe `region<cc>` in rounds 1-4 for `gr`, `se`, `be`, `dk`,
  report first, not high priority. Does not approve a rebuild.
- **Sweden (runlog 2026-09-14):** the two-level rebuild was approved and Jews stay at riksområde
  level. Not a general rule for thin passes.
- **Greece's Thracian split and Italy's three resolutions** were her calls, per country.
- **Ask 015 (LAPOP keeps its pooled level):** LAPOP only. Does not undo Norway's or Latvia's
  rescale, and does not say whether a new ESS country rescales.
