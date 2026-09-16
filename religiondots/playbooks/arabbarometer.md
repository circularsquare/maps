# Arab Barometer playbook

Arab Barometer waves I to VIII (ten files, to 2024): `Q1012` "What is your religion?", cut by `Q1`
governorate, in open files. It is the route for an Arab country whose state asks religion and
publishes none of it: the survey gives each governorate's mix, and a population table the people.

## Used by
- `eg` Egypt: drawn, 24 of 27 governorates, waves III, IV, V, VII on CAPMAS's own estimates;
  wave II deliberately out (`arabbarometer.py::OMITTED`).
- `jo` Jordan: drawn, 12 governorates, waves II to VIII on DOS end-2025.
- `iq` Iraq: drawn, 18 governorates, waves V, VI-3, VII, VIII on the 2024 census, the religion
  answer composed with the sect follow-up.
- `lb` Lebanon: closed on this source; the per-governorate mix is a fieldwork quota (`sources/lb.md`). WVS wave 7
  closes the same way: the same firm, Statistics Lebanon Ltd., sets each cluster's sect there too (`sources/lb.md` §9).
- `ye` Yemen: drawn, 21 of 22 governorates (Socotra unsampled), wave V on the Population Task
  Force's 2025 estimate, the logged sect item composed in; wave III is the replication witness.
- `dz` Algeria: drawn, 48 wilayas, waves V to VII on the RGPH 2008 count; the non-Muslim share at
  two units (Kabylie, the rest) on Anita's 2026-09-16 ruling, split at the national mix; II-IV
  out on the card (`sources/dz.md`).
- `ma` Morocco: drawn, 73 units, waves V to VIII on RGPH 2024 Moroccans; the non-Muslim share
  split urban and rural (tested in this survey and in the Afrobarometer), with foreign residents
  by nationality; III-IV out on the card and the pre-2015 regions (`sources/ma.md`).
- `tn` Tunisia: drawn, 24 governorates, waves V to VIII on RGPH 2024 at one national share and mix
  (no governorate, Greater Tunis or urban stratum passes); II-IV out on the card; wave VI decoded on
  the code (`sources/tn.md`).
- `ly` Libya: drawn, 22 districts, waves V to VII on BSC's 2020 estimate of Libyans at one
  national share and mix (7 non-Muslim answers); III out on the card; wave VII's district labels
  fail the allocation check and set the level only (`sources/ly.md`).
- `sd` Sudan: drawn, 18 states, waves V and VII pooled with Afrobarometer R5-R9 on COD-PS 2022 at
  one national share; II and III out on the card; eight wave V answers contradicting the follow-up
  dropped (`sources/sd.md`). It had been closed on the pooled survey (`sources.md` §11af). Morocco,
  Tunisia and Libya were
  reopened on 2026-09-16 (`queue.md` "Maghreb reopened").
  Saudi Arabia, Mauritania and Bahrain have no `Q1012` answers at all.
- `cab.py::assert_not_quota` wraps this module's quota test for other surveys.

## Loading it
- Ten zipped `.sav` files, ~46 MB, at `www.arabbarometer.org/wp-content/uploads/`. The download
  page shows `href="#"` behind a form; the real names are in its HTML and in
  `arabbarometer.py::WAVES`. Free, no redistribution clause. The site's certificate chain fails
  verification from here, so `_ctx()` skips it and `fetch` checks the zip magic.
- On disk at `data/raw/arabbarometer/` (zips and extracted `.sav`), shared, all ten present. The
  module has no command line: `python sources/eg.py --fetch` runs `ab.fetch` and `ab.unzip`.
- `arabbarometer.py::load(country, expect_waves=, waves=, omit={wave: reason}, recode={raw: new},
  extra={col: (alias, ...)}, raw={col: (alias, ...)}, blank_weights={wave: reason})` gives `wave`,
  `wave_no`, `category` (decoded through that wave's own labels), `geo_raw`, `geo_code`, `w`, each
  `extra` column, and each `raw` column undecoded (a PSU: wave V `psu`, wave III `bid`).
- Build with `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/<cc>.py`.
  `sources/jo.py` is the plain case; `sources/iq.py` adds a second answer column.

## Traps
- **Answer codes are re-used between waves.** Code 3 is `Other` in three waves and `Jewish` in V;
  code 4 is `Jewish`, `Atheist` or `No religion`. Pooled on the code, three answers merge with every
  total intact. Caught by: `load` (decodes each wave through its own labels, raises on an
  unlabelled code). Detail: `sources/eg.md` "The religion codes".
- **Wave II puts the code inside the label.** `8. Jordan`, `1. muslim`, `99999. declined to
  answer`; the whole wave was missing from every country until 2026-09-08. Caught by:
  `country_key` for the country; `fold` strips the ordinal, so `assert_one_wording` raises on
  `1. muslim` beside `Muslim`; merge with `recode=` (`jo.py::RECODE`). Detail: spec §12 "The
  shapes of failure that cost the most", `sources/jo.md` §3.
- **A pool narrower than the files.** Caught by: `load` (`wave_coverage` reads the files first; a
  wave they offer that is not in `waves=` must be in `omit=` with a reason; `expect_waves` and
  stale `omit` entries are checked against the files). Detail: `sources/jo.md` §10.1.
- **One country under two names.** `17. Saudi Arabia` and `Kingdom of Saudi Arabia`. Caught by:
  `assert_no_near_miss` over every declared label; name both in `COUNTRY_ALIASES`.
- **Rows with no answer, and wave I.** Saudi Arabia and Mauritania have governorates, weights and
  an empty `Q1012`; wave I asks `q711` and has no subnational column. Caught by: `load` raises on
  an empty pool and on a wave with no `Q1012`; leave wave I out with `waves=`.
- **The card changes by wave.** `Atheist` is on Egypt's wave V card only and `No religion` on VII;
  Iraq's sect item is absent in II and empty in III. A share pooled over waves whose card lacked
  the box measures the questionnaire: drop the answer (`eg.py::DROPPED`, `iq.py::DROPPED`) or omit
  the waves (`iq.py::OMIT`). Caught by: `dz.py::card` (reads each wave's `Q1012` labels and asserts
  which waves offer a no-religion box against the pool: none in II-IV, `Atheist` in V, `No
  religion` in VI-VII, where Algeria's non-Muslim answers are half that box); lifted 2026-09-15 as
  `arabbarometer.py::card(pool, omit)`, first called by `ly.py`, with the copies in `dz.py`, `ma.py`
  and `tn.py` still to be replaced when each is next edited. The labels are the whole file's, not
  one country's card.
- **Two spellings of one answer.** `Other`/`other` (Lebanon), `refused`/`Refused to answer`
  (Jordan, Iraq). Caught by: `assert_one_wording` at the foot of `load`; merge with `recode=`, whose
  keys must exist. A curly apostrophe got past it until 2026-09-14 (`Ja’fari` against `Ja'fari`).
  Caught by: `arabbarometer.py::fold`, which now reads U+2019 as `'`; `iq.py::RECODE` still merges
  Iraq's by hand.
- **One spelling of two answers.** Composing one category from two columns: `Other` is a box on the
  religion card and on the sect card. Caught by: nothing automatic. List both answer sets,
  intersect them, label each side (`iq.py::COMPOSED`, `compose`), then run `assert_one_wording`
  again. Detail: spec §12 "A COMPOSED CATEGORY CAN COLLIDE WITH ITSELF".
- **A sampling quota passes every other check.** Lebanon's waves V and VII return the same
  Christian count in all eight governorates; the split-half and the held-out check pass it at
  their strongest. Caught by: `assert_not_quota` (exact tie probabilities, Bonferroni, bar
  `QUOTA_P_BAR` 1e-3), called inside `stability` on the column passed in only. Run it on every
  religion column the file offers. Detail: spec §12 "A SPLIT-HALF CANNOT SEE A QUOTA".
- **Weights used to default silently.** `load` took the first of `wt`, `weight`, `weight1500`, used
  1.0 when none existed and filled a blank weight with 1.0. Caught by: `arabbarometer.py::load`
  (stops on no weight column, a blank weight on an answered row, or a mean outside 0.98-1.02 over the
  country's rows; Egypt, Jordan and Iraq measure 0.997-1.001).
- **Governorate labels drift and translate.** Egypt's 27 arrive as 45 labels: `The Lake` is
  Beheira, `Eastern` Sharqia, `The capital` Amman, `Diwaniyah` Al-Qadisiyyah. Unharmonised,
  Egypt's split-half failed with 13 of 27 units in both halves; harmonised, it passed. Caught by:
  each module's `NORM` raising on an unmapped label or two labels for one governorate in a wave;
  `stability` raising when the overlap is not `n_units`. Egypt's `The West Bank` is dropped, not
  guessed (`eg.py::BOGUS`).
- **The `Q1` code is a witness, never the key.** Jordan's codes mean three things across waves;
  Iraq's are `70000+n` and `7000+n`. Caught by: `jo.py::code_witness`, `iq.py::code_witness`
  (each respondent's name against its code, where the code means something).
- **The sample is mostly citizens.** About three in ten of Jordan's DOS denominator are not; wave
  IV's `q1020jo` has 303 Syrians and no Christian among them. Nothing is corrected, and the drawn
  Christian share is a ceiling. Caught by: `jo.py::origins` (prints). Ask it of every new country.
  Detail: `sources/jo.md` §6.
- **The sect item measures sect only where few decline it.** `Q1012A` "Just a Muslim" is 45-82% in
  North Africa and about a quarter of Iraq, its level moving 17-38% by wave: draw it as itself on
  `islam`, never apportioned, and fold madhhabs into their branch (`iq.py::SECT_FOLD`). Caught by:
  `iq.py::sect_geography`, `SHIA_OF_NAMED_BAND`. Detail: `sources/iq.md` §3-4.
- **The split-half is one halving.** `arabbarometer.py::stability` compares early waves with late
  ones once, against `spearman_null.critical_rho`, with no chi-square veto. The rule now is the
  median over every halving against a per-wave permutation null, with the chi-square as veto
  (`cab.stability`); `eg`, `jo` and `iq` were built on the single halving. `cab.stability` refuses
  an empty (wave, unit) cell, and Egypt's Matrouh is in wave V only: test on the units in every
  wave (`ua.py::EXPECT_ABSENT`). It stays in this module; `sources/stability.py` lists it as a different method and did not move it. Detail: spec §12
  "ONE SPLIT-HALF IS A DRAW"; its line that the barometer modules "split by PSU" is not true here.
- **Thin passes lean on one unit.** Jordan passes at +0.617 against +0.5035, and reads +0.509
  without Balqa. Caught by: `jo.py::leave_one_out`, `iq.py::leave_one_out` (print only).
- **Held-out at few units.** Caught by: `held_out` checks every ordering below `EXACT_PERM_MAX`
  and raises if one reaches the observed r; under seven units it prints that it cannot carry the
  join alone, and does not raise.
- **One label on two governorates in one wave.** Wave III labels Yemen's 10503 (Amanat al-Asimah)
  and 10513 (Sana'a governorate) both `Sana'a`; a name join merges the capital into its governorate
  with every total intact. Caught by: `ye.py::decode_iii` (decodes on the CSO-order code, names and
  wave II's labels as witnesses). Not checked for other countries; `NORM`'s one-label-per-unit check
  cannot see two units under one label.
- **Two codes can share one label while a governorate's own label is missing.** Tunisia's waves
  VI-1, VI-2 and VI-3 label both 21009 and 21010 `Jendouba` and have no `Kef` label, in a code order
  that is not INS's (V, VII and VIII are). Decoded by name, Le Kef merges into Jendouba with every
  total intact. Decode such a wave on the code, with the names as witness everywhere else, and settle
  the pair on sample size against the waves that label both (23 against 37 respondents, Le Kef 56 to
  Jendouba's 88 elsewhere). Caught by: `tn.py::decode` (`VI_CODES`, `VI_MISLABELLED`, the swap
  asserted worse). Check for it: print each wave's code-to-label table and look for a label on two
  codes or a unit with no label. Detail: `sources/tn.md` §3.
- **A wave's district labels can fail the population while its weights average 1.** Libya's wave
  VII ranks its weighted respondents per district against BSC's estimate at Spearman +0.853, where
  V and VI read +0.99: Ajdabiya 3.7 times its share, Benghazi 0.32, Tripoli 0.41, mean weights near
  1 in each. Swapping three pairs of labels halves the misfit and does not remove it, so no swap is
  applied. `held_out` pools waves and passed Libya at +0.999 with VII left out; with it in, the
  wave's misfit is diluted, not caught. Caught by: `ly.py::allocation` (per-wave Spearman, VII
  pinned in a band); not shared yet, and it belongs beside `held_out`. Use such a wave for the
  level only. The same wave also re-uses VI's codes 11001-11022 for other districts. Detail:
  `sources/ly.md` §3.
- **Wave V's sect item is interviewer-logged** (`DO NOT READ, LOG ANSWER`) on one cross-country
  list. A branch the list lacks is logged under another code (Yemen's Zaydis on code 14 `Alawi`;
  Libya's 561 `Alawi` with no Maliki code, where VII records Maliki 286 and no Alawi),
  and a whole governorate can have nobody in a box (Ta'iz 0 of 260 `Just a Muslim`). Caught by:
  `ye.py::zaydi_geography` for the first; nothing for the second (`E2001B`, the interviewer, is
  blank, and a PSU split inside a unit replicates a team's habit). Detail: `sources/ye.md` §4-5.
- **Blank weights on answered rows.** Yemen wave V has 32, all with no recorded gender. Caught by:
  `load` (stops); `blank_weights={wave: reason}` fills a named wave's from its PSU mean.
- **Two waves on two cards is not a split-half.** Caught by: `ye.py::replication` (wave against
  wave on the one quantity both cards measure, exact bar plus a permutation for tied zeros) and
  `ye.py::psu_test` (`lits.stability` on the drawn wave's PSUs).
- **`build` finds no room for a tail when the carried answers are all of a unit** (Yemen, 15 of
  21). Sub-floor answers on the same node as a carried one go in at their unit shares:
  `ye.py::SAME_NODE`, asserted against the mapping.
- **A booster sample can sit inside one wave.** Palestine's wave V is 8.78% Christian unweighted and
  0.95% weighted, with 12 Bethlehem PSUs where all nine interviews are Christian. Counts pooled
  unweighted, or a split-half on counts, sees a Christian level the weights remove. Caught by: Not
  checked yet (a per-wave weighted-against-unweighted print belongs in `load`). Detail: sources.md
  §scout-2026-09-14-asia-oceania.
- **Palestine's wave III `wt` averages 0.9755**, so `load`'s weight guard stops the country. Read
  the wave III codebook before passing it. Caught by: `arabbarometer.py::load`. Detail: sources.md
  §scout-2026-09-14-asia-oceania.
- **A weight can be left un-normalised for one country in one wave.** Algeria's `WT` in all three
  parts of wave VI averages 0.840, 0.780 and 0.849 while every other country in the same files
  averages 0.99-1.00. Caught by: `load` stops; `rescale_weights={wave: reason}` divides the wave by
  its mean over the country (`dz.py::RESCALE`). Detail: `sources/dz.md` §3.
- **The religion answer can contradict the denomination follow-up in the same interview.** Algeria
  wave V: all three `Jewish` answers give Shafi'i, Sunni or Orthodox, and three `Christian` answers
  give Shia, Sunni or `Just a Muslim`; drawn as recorded, that is about 60,000 Algerian Jews. Drop
  the contradictions like refusals. Caught by: `dz.py::contradictions` (the list asserted; a
  non-Muslim answer with an unclassified follow-up stops). Not shared yet; it belongs in
  `arabbarometer.py`. Detail: `sources/dz.md` §3.
- **Write an outside level check before building.** Egypt 5.0-7.0% Christian (1986 census
  5.7-5.8%) and Cairo 6-11%; Jordan `CHRISTIAN_BAND`; Iraq `SHIA_OF_NAMED_BAND`. Caught by: those
  assertions in `eg.py::main`, `jo.py::main`, `iq.py::main`.

- **Wave VI has no urban and rural stratum.** `Q13` Urban/Rural is in V, VII and VIII; VI-1 has
  nothing, and VI-2 and VI-3 carry `Q13A`, a self-reported "big city, village or town, rural area"
  (Morocco VI-3: 947 of 1,201 "big city"), which is not the sampling stratum. Test an urban and rural
  contrast on the waves with `Q13` and take the level from the whole pool. Caught by: `ma.py::main`
  (`extra={"urban": ("q13",)}` prints `not asked` for VI). Detail: `sources/ma.md` §4.
- **A share of 0.3% cannot carry a region but can carry urban and rural.** Morocco's 36 non-Muslims
  stand apart in no region, and are urban beyond the sample in both this survey and the
  Afrobarometer. Pre-register the bar, require it in both instruments, and lay the two shares on the
  census's own urban and rural counts per unit. Caught by: `ma.py::urban_test` (`URBAN_BAR`,
  `URBAN_EXPECTED`); nothing shared. The card check is now copied in `dz.py::card` and
  `ma.py::card`: lift it into `arabbarometer.py` next time either is edited.

## "No religion" boxes
The card offers no traditional religion, so the draft procedure in `WORKFLOW_PLAN.md` seldom
arises. What bites is the wave: `Atheist` and `No religion` are on some cards only, so never pool
them over waves that lacked the box. Egypt and Iraq dropped two respondents each; Jordan asserts
that nobody chose it in the five waves that offered it (`jo.py::note_public_figures`). In Egypt a
count is also a floor, because saying it to an interviewer carries risk.

## Shared code
Import these, do not copy them.
- `arabbarometer.py`: `fetch`, `unzip`, `load`, `wave_coverage`, `country_key`,
  `assert_no_near_miss`, `fold`, `assert_one_wording`, `national`, `held_out`, `quota_agreement`,
  `assert_not_quota`, `stability`, `build` (rounds and absorbs drift; `small` may be empty),
  `WAVES`, `WAVE_NAMES`, `OMITTED`, `COUNTRY_ALIASES`, `ELIGIBLE_FLOOR` (1%).
- `spearman_null.py`: `critical_rho`, `exact_p`, `ties_note`.
- `cab.py`: `stability` and `assert_not_quota`. Pass the wave list: `quota_agreement` stops on a
  frame wave missing from its order. `cab.stability` computes through
  `sources/stability.py`; look there first.
- Copied between `jo.py` and `iq.py`; lift them rather than copy again: `key` (Arabic and Latin
  fold), `code_witness`, `leave_one_out`.

## Rulings
- `ask/answered/001-eg` (2026-09-08): Egypt drawn at governorate ("governorates are pretty big"),
  which is also the instrument's ceiling. Not decided: Egypt's wave II.
- `ask/answered/020-eg` (2026-09-14): Egypt's wave II stays out ("a very marginal improvement");
  the pin in `OMITTED` stays.
- `queue.md` §A (2026-09-08): build `lb`, `iq`, `jo` and `ye`, not high priority.
  `ask/answered/021-jo` (2026-09-14): Jordan and Iraq confirmed at governorate as built, and Yemen
  is built the same way. Not decided: sect geography in Egypt; placing Iraq's minorities.
- `ask/answered/007-cr` (2026-09-09): the split-half bar is the exact 95% null; Egypt and Jordan
  did not move. Not decided: any stricter level, which stays Anita's.
- `ask/answered/015-bo`: the pooled level stays as displayed, for LAPOP countries only.
