# LAPOP playbook

The AmericasBarometer: about 1,500 interviews per country per wave, religion asked as `q3c` (`q3cn` in
some waves) on a Latin American card, 2010-2023, placed by `prov` at ADM1. It is the route for a Latin
American country whose census asks no religion, **after** checking whether the office runs a household
survey that does (MICS `HC1A` included); Uruguay, the Dominican Republic and Honduras all had one
(spec §12 "AN INTERNATIONAL SURVEY'S COUNTRY LIST IS NOT A LIST OF WHAT THOSE OFFICES HAVE").

## Used by
- `gt` Guatemala: drawn, merge 2010-2023, 22 departments, `prov - 200`; `Ninguna` on an override.
- `sv` El Salvador: drawn, merge, 14 departments, joined on the name (COD pcodes are alphabetical).
- `ec` Ecuador: drawn, merge 2010/2012/2016/2023, 20 provinces placed, 3 at the national rate, Galápagos blank.
- `pa` Panama: drawn, merge, 10 units; Guna Yala and Emberá-Wounaan blank, Panamá Oeste merged.
- `cr` Costa Rica: drawn, merge, 7 provinces on INEC's 2022 estimate.
- `co` Colombia: drawn, merge, 5 waves, 26 of 33 departments; the current stability test and one-round rule.
- `bo` Bolivia: drawn from LAPOP's single-country files 2010-2023, 9 departments, decoded by municipality name.
- `hn` Honduras: drawn from ENDESA-MICS 2019; LAPOP 2012/2014/2018/2023 is the cross-check.
- `do` Dominican Republic, `uy` Uruguay: drawn from office surveys; LAPOP is the cross-check (`do.py::cross_check`, `uy.py::cross_check`).
- `ht` Haiti: drawn from ECVMAS 2012; LAPOP refused, its `prov` does not decode (r=+0.23 against COD-PS).
- `ve` Venezuela: queued; single-country files 2010-2016/17, `PROV` 16xx state codes, expect thin states.

## Loading it
- **Grand merge**: `data/raw/lapop/Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta` (1.12 GB,
  free click-through on lapopsurveys.org, walk-through in `sources/gt.md`). `lapop.fetch()` slims the ten
  `lapop.USECOLS` to `data/raw/lapop/lapop_slim.feather`; `lapop.load(pais, expect_waves)` returns one
  country with `code`, `prov_code`, `w` (`weight1500`) and `wave`. Religion waves are 2010, 2012, 2014,
  2016, 2018, 2023; 2021 is the phone round with neither religion nor place.
- **The slim file has no `municipio`, `upm`, `cluster` or `estratopri`**, and the decode and the stability
  test need them. Extract them per country as `co.py::fetch` does (`data/raw/co/lapop_co.feather` plus a
  labels CSV) and check the extract against `lapop.load` (`co.py::load`).
- **Countries the free merge drops after 2008** (Bolivia, Venezuela): single-country Stata files, Free
  Tier, `?lp_download=<id>` on LAPOP's data directory. Venezuela's ids are 2010 1582, 2012 1970, 2014 2020,
  2016/17 2195 (sources.md §11ap). `bo.py::fetch` and `bo.py::read_wave` are the pattern
  (`data/raw/bo/lapop_bo_<year>.dta`).
- Four files per country: `sources/<cc>_geo.py`, `<cc>_grid.py`, `<cc>.py`, `taxonomy/<cc>2023.py`. Copy
  `sources/co.py` for a merge country and `sources/bo.py` for single-country files; they carry the current
  tests, which `gt`, `sv`, `ec`, `pa` and `cr` predate. The population base is a geography question
  (`playbooks/geography.md`); three of the seven drawn LAPOP countries (`ec`, `cr`, `bo`) replaced COD-PS with an office count.
- Run `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/<cc>.py --fetch` once,
  then without `--fetch`. The rest is `COMMANDS.txt`.

## Traps
- **`prov` means something different in every country, and can change between waves.** Guatemala's
  `prov - 200` is the COD pcode; El Salvador's code join mispairs 12 of 14; Panama's puts Panamá in Kuna
  Yala. The merge prints Honduras's 2023 labels on every wave, while 2012-2018 codes are LAPOP's own order
  (404 labelled Copán is Cortés; the labels misplace 59.6%). Bolivia's files use LAPOP's order to 2018 and
  a province code `10DDPP` in 2023. `pais=41` is Canada, not Haiti. Decode each wave from `municipio`
  names (per `prov` code, the departments its names can belong to must intersect in exactly the labelled
  one), or from `upm` where a wave has no `municipio` (Colombia 2010 holds DANE codes), then pin it with
  sample share against population. Caught by: `co.py::check_labels`, `bo.py::decode_wave`,
  `hn.py::lapop_decode` with `hn.py::population_witness`; the pcode join by `<cc>_geo.py::check_code_join`.
  `lapop.load` reads no labels. Detail: spec §12 "A SURVEY'S SAMPLING-UNIT COLUMN CAN CARRY THE OFFICE'S
  PLACE CODES" and "LAPOP'S SINGLE-COUNTRY FILES KEEP THEIR OWN ORDER"; sources.md §11ap (Honduras table).
- **`lapop.held_out` fails correct decodes on small or lopsided countries.** It requires 0 of 20,000
  sampled orderings to reach the observed r. Below about ten units that is unattainable (Costa Rica, 7);
  one dominant unit lets hundreds of wrong orderings reach it (Panamá is 51%, 468 of 3.6 million); a shelf of
  equal-sized units cannot be ordered at all. Old rounds' weights carry their own year's population, so a
  check against a recent census can fail too. Use `lits.held_out` (exhaustive when small, forgives swaps
  within sampling error), report rather than assert where weights are old, and let the names decide. The
  mean-age comparison has no power (F below 1) and must stay a print. Caught by: `lapop.held_out`,
  `lits.held_out` (as in `bo.py::held_out_waves`), `pa.py::held_out_exact`, `cr.py::held_out`. Detail:
  `lapop.held_out` docstring; spec §12 "LAPOP'S SINGLE-COUNTRY FILES".
- **`municipio` is a PSU list, not a finer tier.** About 50 sampled municipalities a wave, and many
  departments are one municipality in most waves (Colombia's Huila is Neiva), so a department's extreme can
  be one city. Do not build below ADM1; quote in `note_public` only units sampled in two or more
  municipalities. Caught by: nothing needed, it is a rule. Detail: sources.md §11ad; `sources/co.md` §8.
- **The answer card changed after 2016.** Code 77 `Otro` is exactly zero in 2010-2014 in every country;
  codes 6, 10 and 12 (Mormons, Jews, Witnesses) are exactly zero in 2018 and 2023. Pooled, those three are
  floors and `Otro` is inflated late; `Otro`'s split-half is undefined because the box did not exist.
  Bolivia's files differ (2016/17 still offers 6 and 12 beside 77). Say it in the taxonomy `REVIEW` text,
  in `taxonomy/ec2023.py`'s words. Caught by: `lapop.py::check_waves`, run by `lapop.load` (`CARD_ABSENT`
  must hold exactly; it did in all 22 countries of the merge on 2026-09-14). Not in `bo.py::read_wave`. Detail: `taxonomy/ec2023.py` docstring.
- **A wave can arrive with shifted answer codes.** Honduras 2016: code 3 (Eastern religions) is 9.4% over
  29 of 51 municipalities and evangelicals fall to 13.5%; pooled, Eastern religions come out 2.2% against
  0.38%. Tabulate each code's weighted share by wave before pooling, as `sources/co.md` §4 does, and drop a
  wave that jumps. Caught by: `lapop.py::wave_flags`, run by `lapop.load` (per code and wave, unweighted,
  against the median share of the other waves on that card: a zero where 8 are expected, or 3x or a third
  of expected at Poisson p under 1e-6; a flag not in `WAVE_FLAGS` stops). On 2026-09-14 it flagged `gt`,
  `sv`, `ec`, `pa`, `cr` and `do`, listed there as not judged; El Salvador 2016's code 3 is Honduras's shape
  (113 respondents against about 1 expected, and `sv` draws that answer at 1.42% nationally). Colombia's
  flags are judged in `sources/co.md` §4. Not in `bo.py`. Detail: sources.md §11ap, "Two waves cannot be
  used as they stand".
- **`lapop.stability` is one halving, and one halving is a draw.** It splits the waves once by date with
  the exact Spearman null and a 1% size gate, and has no chi-square, so a mostly-zero column can pass. The
  current test is `co.py::stability` and `bo.py::stability`: unweighted counts, median Spearman over every
  distinct halving (`stability.py::halvings` keeps all ten of five waves), 2,000-draw per-wave permutation null, pass
  only with the spatial chi-square also under 0.05; `bo.py` also refuses an answer when one (wave, cluster)
  cell holds half of it and runs Honduras's which-unit-tops-both-halves test. Run it at the ADM1 units and
  at `estratopri`, which nests them, and assert the verdicts as `CARRIES`, `REFUSED`, `COARSE`,
  `STANDOUTS`. It needs every unit in every wave (it stops on an empty cell). It computes through
  `sources/stability.py`. Caught by: `co.py::main` and `bo.py::main` stop when a verdict changes. Detail:
  spec §12 "ONE SPLIT-HALF IS A DRAW", "A RANK TEST CAN BE PASSED BY A COLUMN THAT IS MOSTLY ZERO",
  "EVERY DISTINCT HALVING MEANS ALL OF THEM", "A CHI-SQUARE CANNOT VETO A CLUSTER".
- **Units missing from waves cannot be ranked, and each kind is treated differently.** Tabulate waves per
  unit first. Never offered a code or never sampled (Galápagos has no code 920; Colombia's seven): blank,
  in `gap=`. Sampled in one round: its design region's shares when the region passes a leave-one-out
  (`co.py::region_fallback`), else the national rate. A neighbour average lost to the national rate in
  Ecuador. Where many units are unmeasured (Dominican Republic, 10 of 32), do not build from LAPOP alone.
  Caught by: `co.py::main` asserts `ONE_ROUND`, `ON_REGION`, `NOT_DRAWN`; `ec.py::main` asserts
  `NATIONAL_RATE`, `NOT_DRAWN`. Detail: spec §12 "A UNIT MEASURED IN ONE ROUND TAKES ITS REGION'S SHARES";
  `sources/co.md` §6.
- **The tail construction degenerates in two cases.** `lapop.build` spreads each unit's remainder over the
  failing answers at national proportions. With exactly one failing answer that is its own measured share
  (and a division by zero where it is 0): set it flat and rescale the carried shares (`do.py::main`). And
  if the residual draws any small answer at 2x its national share or more in a unit with zero unweighted
  respondents of it, after standouts are taken out, switch every failing answer to flat. Print each
  national-rate answer as drawn beside its own unit share and look for a reversal. Caught by: Not checked
  yet (both belong in `lapop.build`; `tz.py::compose` implements the 2x test, `bo.py::compose` prints the
  reversal). Detail: spec §12 "A SINGLE FAILING CATEGORY CANNOT USE THE RESIDUAL CONSTRUCTION", "SMALL
  CATEGORIES GO IN THE RESIDUAL UNLESS", "THE RESIDUAL CONSTRUCTION CAN REVERSE A GEOGRAPHY".
- **The non-Christian tail is set by the card, and it misreads in both directions.** No Maya, Andean,
  Winti or Vodou box. Suriname's traditional cell read 0.21x its census with a near-perfect ordering; the
  missing people answered Christian. Guatemala's 0.22% (43.6% indigenous), Ecuador's 0.06% and Panama's
  0.43% are floors named in `note_public`; El Salvador's 0.03% is plausibly right; Haiti's 4.4% is high
  against its census's 2.11%. Map code 7 to `indigenous` only where it leans to indigenous areas (a unit's
  share of code 7 over its share of the sample: Quiché 4.8x); Colombia's leans to Bogotá (2.8x) and went to
  `other.co`. Caught by: Not checked yet (the lean ratio belongs beside `lapop.national`). Detail:
  sources.md §11ad "what limits this source is its ANSWER SET"; `sources/co.md` §10.
- **Codes 2 and 5 trade respondents between waves.** Traditional Protestant and evangelical swap boxes
  from round to round. Test the union and the Spearman between their unit shares; swapping by place would
  make it negative. Both countries that tested kept them apart (Colombia +0.08, Bolivia +0.03). Caught by:
  `bo.py` runs the union through `stability` as `UNION = 25`, never drawn; Colombia's was a session run
  (`sources/co.md` §5).
- **`Ninguna (cree en un Ser Superior)` moves people across the Catholic line.** Against Uruguay's office
  survey LAPOP ordered non-Catholic Christians at +0.86 and Catholics at +0.34, not significant. A
  no-religion geography resting on code 4 claims less than it looks. Caught by: `uy.py::cross_check`,
  reported. Detail: spec §12 "AN INTERNATIONAL SURVEY'S COUNTRY LIST".
- **Single-country files can carry `wt` = 1 in a disproportionate design.** Bolivia from 2016/17, with
  Beni at twice its population share. Post-stratify each (wave, unit) to its population share of 1,500.
  Caught by: `bo.py::read_wave` (stops on a constant `wt` in a wave not in `WT_CONSTANT`, and on a listed
  wave whose `wt` varies); `bo.py::poststratify` corrects the listed waves.

## Shared code
Import these, do not copy them.
- `sources/lapop.py`: `fetch`, `load`, `valid`, `national`, `CATEGORY`, `RELIGION_WAVES`, `build`;
  `held_out` above about ten units with no dominant unit. `stability` is the older single-halving test.
- `sources/lits.py::held_out` for small unit counts; `sources/spearman_null.py` (`critical_rho`, `exact_p`).
- `co.py::stability`, `region_fallback` and `bo.py::stability`, `compose`, `decode_wave`, `poststratify`
  read their module's globals (`WAVES`, `STAB_PERM`), so they cannot be imported as they stand. The
  statistic inside both `stability` functions is `sources/stability.py`'s (`halvings`, `median_rho`,
  `wave_null`, `permutation_p`, `chi2_p`; `bo` adds `halves`, `top_both_halves`): import those, copy
  the rest and name the source in the docstring.

## Rulings
- **007-cr** (2026-09-09): the split-half bar is the exact Spearman null at 0.05, not `1.96/sqrt(n-1)`.
  Does not decide a stricter level, and never licenses moving a bar to make an answer pass.
- **015-bo** (2026-09-14): LAPOP countries keep the pooled 2010-2023 level, new ones (Venezuela) too; no
  recent-level rescale. Does not decide the note wording; Bolivia's and Colombia's notes say the map is
  more Catholic than the recent rounds.
- **Guatemala override** (2026-09-08): `Ninguna` drawn under the bar on a chi-square of 3.6e-16. An
  `OVERRIDE` is Anita's call per answer, never an agent's, and needs a significant chi-square first.
- **Ecuador units** (2026-09-08): a unit measured once is assumed, a unit nothing measured is blank
  (Galápagos). Dominican Republic: "probably cant do DR without additional data".
- **One-round rule** (supervisor, 2026-09-14; Anita deferred, leaning "use the most granular thing
  available"): `co.py::region_fallback`. **Carchi (`ec`) switches to Sierra's shares** (Anita, 2026-09-14
  night); Pastaza and Orellana stay national. Not yet applied: `ec.py` still lists `EC04` in `NATIONAL_RATE`.
- **The tail** (Anita, 2026-09-08, sources.md §11ad): supplement non-Christian religions from another
  source; a country built from LAPOP alone goes on `queue.md`'s refinement list. Does not name the source.
