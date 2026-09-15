# Afrobarometer playbook

Afrobarometer's merged rounds 4 to 9 (2008-2023): about 30 religion answers, cut by `REGION`
(ADM1, named), in open files. It is the route for an African country whose census never asked
religion or never published it below the nation: the survey gives each unit's mix, and a
population table gives the people.

## Used by
- `ng` Nigeria: drawn, 37 states, R4-R9 on COD-PS 2022; each state at its own mix, no column
  margin, small categories at the national rate.
- `lr` Liberia: drawn, 15 counties; the county pattern fitted by IPF to two 2022 census margins.
- `tz` Tanzania: drawn, 30 regions, R4 and R6-R9 on the 2022 census; R4 placed by district, R5
  left out; the first build on `cab.stability`.
- `cm` Cameroon: drawn, 12 units (the regions with Yaoundé and Douala apart), R5-R9 on COD-PS
  2025; the first build to draw churches (Presbyterian, Baptist), on level by round, the census's
  Protestant total and the unnamed share where each church lives (`sources/cm.md` §4).
- `mz` Mozambique: drawn from the census; Afrobarometer was the witness that put `Sem religião`
  at 93-98% no religion (`sources/mz.md` §6).
- `mg` Madagascar: drawn, 22 regions, R5-R7 and R9 on the 2018 census; Catholic, FJKM and Lutheran
  as churches (`Christian only` 0.3-2.3% by round, both DHS surveys witnessing the
  Catholic/Protestant level); None and traditional placed as one box and split at one national
  ratio (`sources/mg.md` §5); R4 left out (no Betsiboka district sampled).
- `tg` Togo: drawn, 6 units (Lomé apart), R5-R9 fitted by IPF to the 2022 census's national rows
  (UNSD table 28) and unit populations, as Liberia. With a census margin the churches' levels come
  from the census, so Pentecostal and Presbyterian are drawn as patterns (`sources/tg.md` §4).
- Also queued with an Afrobarometer route or witness: `sn`, `bf` (`queue.md`, "Africa swept
  a second time"). Mauritania is not asked the question.

## Loading it
- Six merged `.sav` files, ~280 MB, plain links on `afrobarometer.org/data/merged-data/`, no
  account. Citation required; the geocoded extracts are gated and not used. File, URL, religion
  column and weight column per round: `afrobarometer.py::ROUNDS`.
- On disk at `data/raw/afrobarometer/`, shared by every country. Fetch once with
  `python sources/afrobarometer.py --fetch`.
- `afrobarometer.py::load(country, expect_rounds=[...], regroup=True, extra=[...])` gives one row
  per answering respondent: `round`, `category` (the answer's own wording), `geo_raw` (that
  round's `REGION` label), `geo_code`, `w`, and any `extra` column (`DISTRICT`, `LOCATION.LEVEL.1`).
- Build with `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/<cc>.py`.
  Copy the order of `sources/tz.py::main`: load, raw crosstab, `Christian only` by round, card,
  group, units, held-out per round, quota, split-half, standouts, compose, level, witness.

## Traps
- **The religion column changes name.** Q90, Q98A, Q98A, Q98, Q98A, Q95 for R4-R9, and R7's
  `Q98A` is a different question, so a fallback lookup reads the wrong column. Caught by:
  `afrobarometer.py::load` (asserts the variable label). Detail: `sources.md` §11ai.
- **The weight changes at round 8.** `withinwt` becomes `withinwt_hh`; `Combinwt` is
  cross-country and would re-level units by country size. Caught by: `afrobarometer.py::load`
  (the weight must average 1 over the country). Round 6 is not UTF-8: `_read` falls back to LATIN1.
- **`COUNTRY` is spelled differently by round.** Côte d'Ivoire three ways, `Swaziland`/`eSwatini`,
  `Cape Verde` in R4-R6; the pool loses whole rounds with no error. Caught by: `load` through
  `COUNTRY_ALIASES`, raising on an accent-only miss. An ordinary miss is only printed, so read
  the `no <country> rows in R` line and always pass `expect_rounds`. Detail: `sources/lr.md` §9.
- **`Christian only` is set by the fieldwork.** The share naming no denomination swings 48.9
  points by round in Liberia, 27.9 in Nigeria and 65.7 in Botswana, by country and round
  together, so there is nothing to divide out. Group to Christian, Muslim, traditional, none and
  other; draw a denomination only with an outside witness to its level (Madagascar swings 2.0
  points, Mali 1.3). Caught by: `lr.py::main`, `ng.py::main` (the swing must stay at 15 points or
  more), `tz.py::main` (10); nothing shared. Detail: `sources.md` §11ai.
- **Two other boxes can trade places between rounds.** Madagascar's `Traditional/ethnic religion`
  runs 8.5, 4.5, 1.5, 1.3% by round while `None` runs 8.2, 4.0, 13.0, 12.7%, and per region the same
  places move (Melaky 45% traditional and 0% none in R5-R6, 0% and 12% in R7 and R9). Tested apart,
  traditional fails the split-half and None's level trips Norway's 3.5-point bar, which reads as a
  real change and is not one. Print each small box per unit, early rounds against late, before
  testing it alone; where two swap, test and place them as one and split them at the late rounds'
  national ratio, with an outside witness to that ratio (Madagascar: both DHS reports). Pooling fixes
  the level and keeps the early answers; it did not make the ranking steadier (+0.57 against +0.59).
  Caught by: `mg.py::swap_table` (prints), `mg.py::none_fraction` (asserts the ratio against the DHS);
  nothing shared. Detail: `sources/mg.md` §5.
- **Every answer is grouped by name, and a box can be renamed.** An answer missing from the group
  table is dropped silently. `Shia only` (R4-R5) and `Shia` (R6-R9) are one box that does not
  fold together, and a zero in the crosstab means nobody chose it, not that it was off the card.
  Caught by: `GROUP` (`ng.py`, `tz.py`) and `CENSUS_CATEGORY` (`lr.py`) raising on an unmapped
  answer, then `assert_one_wording` on the grouped column (load with `regroup=True`);
  `tz.py::report_card` asserts `None` is on every drawn round's card, `ng.py::report_card` prints.
  Renames are not checked yet (a shared card reader belongs in `afrobarometer.py`). Detail: spec
  §12 "A ZERO IN A CROSSTAB IS A FACT ABOUT RESPONDENTS".
- **`REGION` is a per-round label set.** Liberia's 15 counties arrive as 33 strings, Nigeria's 37
  states as 78; the codes are a cross-country range re-cut between rounds, so never pool on them.
  Harmonise before any test. Caught by: each module's `NORM` raising on an unmapped label or on
  two labels for one unit in a round; `stability` raising when fewer units are in both halves.
- **A round can carry codes and no labels.** R8 has Tanzania's codes 740-770 and no value labels;
  `load` used to print `0 REGION labels` and return blanks. Decode by code only where every labelled
  round gives each code one unit. Caught by: `afrobarometer.py::load` (stops on a REGION code with no
  label unless the round is named in `unlabelled_rounds=`, as `tz.py` names R8, and on a named round
  that has its labels), then `tz.py::decode_codes`, `check_r8`. Detail: spec §12 "A MERGED
  SURVEY FILE CAN DROP ONE COUNTRY'S REGION LABELS".
- **Rounds fielded before a boundary re-cut.** Tanzania's R4 and R5 use the 26 pre-2012 regions.
  Place a round by `DISTRICT`, assert each district lands in its old region's successors, drop a
  district divided across the new line (Magu), and leave out a round with nothing finer (R5).
  Caught by: `tz.py::decode_r4`, `check_locations` (district against region label in R6, R7, R9).
  Detail: spec §12 "A ROUND FIELDED BEFORE A REGIONAL RE-CUT".
- **`afrobarometer.py::stability` is out of date.** It takes one early-against-late halving
  against `1.96/sqrt(n-1)`, with no quota test and no chi-square veto. The rule now: the quota
  test first, then the median Spearman over every halving against a per-round permutation null,
  with the spatial chi-square as a veto. Caught by: `cab.assert_not_quota` and `cab.stability` as
  `tz.py::main` calls them (`ng` and `lr` never ran the quota test). `cab.stability` computes through
  `sources/stability.py`; this module's own copy stays as it is. Detail: spec §12 "ONE SPLIT-HALF IS A DRAW", "A RANK TEST CAN BE PASSED
  BY A COLUMN THAT IS MOSTLY ZERO", "A SPLIT-HALF CANNOT SEE A QUOTA".
- **A round that misses a unit.** Nigeria's R6 has no Adamawa, Borno or Yobe. `cab.stability`
  refuses any empty (round, unit) cell, which is why Tanzania dropped R5 rather than keep it for
  21 regions. Test on the units present in every round (`ua.py::EXPECT_ABSENT`) or drop the round,
  and say which. Caught by: `cab.stability` raises. Detail: spec §12 "A ROUND THAT SKIPS A UNIT".
- **A column margin from the survey's own national shares.** With a population row margin the fit
  undoes the reweighting; it moved Nigeria 4.6 points. Fit columns only to a count of the same
  people (Liberia's census Table A13); otherwise compose per unit. `held_out`'s thinnest and
  fullest line (Yobe 0.63x, Bayelsa 1.37x) is the early warning. Not checked yet (an assertion
  belongs beside any IPF call). Detail: spec §12 "FITTING A COLUMN MARGIN", `sources/ng.md` §3.
- **The tail.** A unit where everyone gave a carried answer has no residual (Grand Cape Mount,
  seven Nigerian states). Caught by: `afrobarometer.py::build` raises. The default is the
  residual; go flat (national shares, carried shares scaled to fill) when the residual draws a
  category at 2x or more in a unit where the survey found none (Tanzania's traditional, 4.03x).
  Caught by: `tz.py::compose` with `TAIL_FLAT`. Detail: spec §12 "SMALL CATEGORIES GO IN THE
  RESIDUAL".
- **Zero cells.** No Muslims found in Abia, Cross River or Ebonyi, no Christians in three Zanzibar
  regions: drawn at zero (§3.5), and `note_public` says the survey found none. Where an outside
  estimate implies 8 or more expected respondents, the zero is the instrument. Not checked yet
  (modules only print zero cells). Detail: spec §12 "A CATEGORY THAT IS EXACTLY ZERO".
- **Held-out strength.** Run it per round as well as pooled. Caught by:
  `afrobarometer.py::held_out` (checks every ordering below 50,000 and raises if any reaches the
  observed r); under seven units it prints that it cannot carry the join alone, and does not raise.
- **A 14-year pool can be stale.** Compare the drawn level with the last two rounds recomposed the
  same way. Caught by: `tz.py::main` (`LEVEL_GAP_MAX`, 3.5 points). Detail: spec §12 "A SURVEY POOL
  THAT SPANS A FAST CHANGE".

- **`Christian only` is uneven across units, so a church that holds its level can still be drawn
  short where it lives.** Cameroon: 7% of Christians in Ouest, 35% in Adamaoua; Lutherans hold
  2.1-2.4% by round and live in the north. Test the unnamed share averaged over the church's own
  respondents against the national one. Caught by: `cm.py::unnamed_where_they_live`; nothing
  shared. Detail: `sources/cm.md` §4.
- **A box can be abandoned between rounds while it stays on the card.** Cameroon's `Other` has 58
  answers in R5-R7 and none in R8-R9 (Anglican, Methodist and Coptic vanish too), and passes the
  split-half on which rounds it is in. Caught by: `cm.py::main` (asserts the zero; `NOT_PLACED`).
- **REGION codes can shift between rounds with the labels intact** (Cameroon R8, every code off by
  one against R6, R7, R9), and R6's labels arrive as mojibake (`Centre-YaoundÃ©`) because `_read`
  falls back to LATIN1. Decode by label after undoing the mojibake, and check against the district
  column. Caught by: `cm.py::gkey`, `cm.py::check_locations`.
- **`held_out` stops a correct six-unit decode.** Six units allow 720 orderings, and a sample frame
  from an older census moves the capital's share (Togo: Lomé sampled at 1.40x its 2022 share, 14
  orderings reach the observed r). Where REGION labels are names, make the location column the
  witness and say so in code. Caught by: `tg.py::check_locations` (asserts the one known
  disagreement); nothing shared. Detail: `sources/tg.md` §4.
- **A box can stay a value label after a country stops using it.** `Assembly of God` is a label in
  every round and Togolese chose it only in R5 and R6: the labels are the merged file's, not one
  country's card. Read the country's own counts by round before assuming a box was offered. Caught
  by: `tg.py::report_card` (prints). Detail: `sources/tg.md` §4.

## "No religion" boxes
The draft procedure is at the foot of `WORKFLOW_PLAN.md`.
- **As the source.** The card offers `Traditional/ethnic religion`, `None`, `Atheist` and
  `Agnostic` separately, so `None` is `unaffiliated` (step 2; Tanzania, `taxonomy/tz2022.py`
  `REVIEW`). Confirm the traditional box is on every pooled round's card (`tz.py::report_card`).
- **As the witness for a census box that lumps them** (step 4): weighted over the country's
  rounds, traditional against none plus atheist plus agnostic. Mozambique R4-R9, 10,467 adults:
  0.17% against 7.6%; with Pew 2009, 93-98% none, so `unaffiliated`. It measures adult
  self-description, not practice; say so. Not checked yet (computed in scratch scripts; a
  function belongs in `afrobarometer.py`). Detail: `sources/mz.md` §6.

## Shared code
Import these, do not copy them.
- `afrobarometer.py`: `fetch`, `load`, `country_spellings`, `fold`, `assert_one_wording`,
  `report_wordings`, `national`, `held_out`, `build` (float counts: round once, at the end),
  `ROUNDS`, `AB_DIR`, `COUNTRY_ALIASES`, `ELIGIBLE_FLOOR` (1%), and since 2026-09-14
  `round_within_rows` and `compose(df, nat, units, cats, carried)` (tz.py's, with the category list
  passed in and no standouts; `cm.py` uses them, `tz.py` and `ng.py` keep their copies).
- `cab.py`: `assert_not_quota(df, country, waves)` and `stability(df, cats, units, label)`. They
  expect columns `wave` and `code`; rename as `tz.py::main` does. `cab.stability` computes through
  `sources/stability.py`; look there first.
- Copied between country modules today; lift them rather than copy again: `key`,
  `round_within_rows`, `report_card` (`ng.py`, `tz.py`); `splits`, `standouts`, `compose` (`tz.py`).
- An independent sample at the same units: Global Flourishing Study wave 1,
  `tz.py::gfs_witness` (`python sources/jp_gfs.py --fetch`).

## Rulings
- `ask/answered/010-ng` (2026-09-11): Nigeria's state Christian/Muslim balance ships as built,
  with its note. Not decided: whether Christians in the Sharia states or Zanzibar are a rule-2
  group, or zones against states elsewhere; Nigeria's Shia refusal was the builder's call. Her
  follow-up, to revisit countries drawn as basically Christian and Muslim, is low priority and not
  to be started unless a session is told to (`queue.md` "Revisit").
- Zanzibar (2026-09-14 night): one Christian share across all five Zanzibar regions; as of
  2026-09-14 `tz.py` still draws them separately. Not decided: Mbeya and Songwe, round 5, small
  units in other countries.
- Mozambique (2026-09-14 night): `Sem religião` is `unaffiliated` on the Afrobarometer and Pew
  split, and the draft procedure stays as written. Not decided: anything about practice.
- Madagascar (2026-09-14 night): before building, check the card for a separate traditional or
  ancestral answer, then apply the draft procedure.
- `ask/answered/007-cr` (2026-09-09): the split-half bar is the exact 95% null. It was applied to
  `lapop.py` and `arabbarometer.py`; `afrobarometer.py` kept `1.96/sqrt(n-1)` by the
  implementer's choice (`sources.md` §9ct), not Anita's.
- `ask/answered/015-bo`: the pooled level stays as displayed, for LAPOP countries only.
- Not a ruling: putting Tanzania's round 5 back into its 21 unchanged regions is a low-priority
  supervisor item, waiting on a split-half that handles an empty (round, unit) cell.
