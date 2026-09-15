# Central Asia Barometer playbook

The Central Asia Barometer: about 1,500 interviews per country per wave, twice a year since 2017, in
Kazakhstan, Kyrgyzstan, Tajikistan, Uzbekistan and (waves 4-6, 10-14) Turkmenistan, with religion in
`Religion_M` and region in `Region_M`. It is the route for a Central Asian country whose census asks no
religion, after checking per country and per wave that the item was asked, and whether face to face or by
phone. LiTS, the other Central Asian survey, has its own playbook (`playbooks/lits.md`).

## Used by
- `uz` Uzbekistan: drawn, waves 1-6 (9,000), 14 regions; religion read within three ethnic groups and laid
  on the 2026 census's ethnic groups per region.
- `tm` Turkmenistan: drawn, waves 4-6 (4,500), 6 velayats; religion read within four nationality cells and
  laid on the 2022 census's nationality counts; wave 14 (2023, phone) is a witness only, never pooled.
- `tj` Tajikistan: closed; the item is withheld from Tajikistan by design in every wave (`sources/tj.md` §4).
- `kg` Kyrgyzstan: drawn from LiTS. The barometer asks Kyrgyzstan too (wave 14's `DD13` label names KGZ) and
  nobody has scanned its Kyrgyz files. `kz` is drawn from its census.

## Loading it
- All 14 waves are on disk in `data/raw/cab/`: `CAB-Survey-Wave-<N>-All-Countries-And-Files-<year>-<season>.zip`
  for waves 1-9 and `.rar` for 10-14, from the open Discuss Data mirror (dataset
  `1d10e56e-540b-4751-96b6-885309cb4b1d`, no form; `ca-barometer.org` itself wants one). Each archive holds,
  per country, Stata, SPSS and Excel files, the questionnaires and a methods report.
- `cab.load(country, waves)` reads each wave's Stata file straight out of the zip (`country` as spelt in the
  file name, `uzbekistan`) and returns `wave`, `region_code`, `region`, `answer_code`, `code` (the answer's
  label, which mappings key on) and `w` (`totwt`, mean 1.0 a wave, post-stratified on region x urban/rural,
  age and sex).
- `Ethnic_M`, `SamPt` (sampling point) and `IntCode` (interviewer) are in waves 1-6 and `cab.load` does not
  return them. Read them as `uz.py::survey_extras` or `tm.py::extra_columns` do, and assert the rows line up.
- Wave 14 renames the columns (`DD13` religion, `MM10` region, `FinalWgt1` weight), and `cab._dta` refuses
  `.rar`. Extract with `tar -xf <rar> -C <dir>` and read the `.dta` directly (`tm.py::wave14`).
- The methods report gives the frame year, the strata and the PSUs per region (wave 4, Table 6). Read it first.
- Copy `sources/uz.py` (the census counts ethnic groups by region) or `sources/tm.py` (few units). Run
  `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python sources/<cc>.py` (`uz` about two
  minutes, `tm` under one).

## Traps
- **The item is not asked of every country in every wave.** Tajikistan: no wave asks it (wave 14's `DD13`
  is code 95 `Not Asked` for all 1,500), only a phone re-contact in waves 1-2. Uzbekistan: none in waves
  7-13, and wave 14 codes every Uzbek `Not Asked`. Scan every country's file for a religion variable and a
  place variable, and write the result down per country, before closing the file set. Caught by: `cab.read_wave`
  (stops on a missing `Region_M` or `Religion_M`, and on any `Not Asked` religion answer; it stops on
  Uzbekistan's wave 7 and Tajikistan's wave 1). Scanning the other countries is still a rule. Detail: spec §13 "A SURVEY SERIES OPENED FOR ONE COUNTRY HAS TO
  BE SCANNED FOR ALL OF ITS COUNTRIES"; sources.md §11ao; `sources/tj.md` §4.
- **Wave 7 moved to mobile phones, and the item and the level moved with it.** Waves 1-6 are face to face.
  Turkmenistan's wave 14 reads Ashgabat 8.44% Christian against 35.32% face to face, with the same top and
  bottom velayat. Never pool across the change; a phone wave can witness an ordering, not a level. Caught
  by: `cab.load` (stops on a wave list mixing 1-6 with 7 on, and on phone waves unless
  `phone_witness=True`); `cab._dta` still refuses the `.rar` waves 10-14. Detail: sources.md §11ao "What generalises"; `sources/tm.md` §4.
- **Wave 1 breaks pandas' label conversion.** `TypeProb2` has a duplicated value label, so
  `convert_categoricals=True` raises on the whole file, and `read(columns=...)` loses the label list.
  Caught by: `cab.read_wave` reads integer codes through each column's own label set; `cab.load` stops if a
  code means different words in two waves. Detail: `sources/cab.py` docstring.
- **An old frame puts the wrong people in each region.** Turkmenistan's sample was allocated and weighted on
  1995 figures: 6.4% Russian, Ukrainian or Armenian against 1.87% in the 2022 census (Ashgabat 35.6% against
  7.71%), and 92.9% of that cell answers Christian, so regional shares would have drawn Ashgabat 35.3%
  Christian. Uzbekistan's Tashkent city sample is 23.05% Russian against the census's 9.29%, although it is
  post-stratified on region. Compare `Ethnic_M` with the census's nationality table per unit before using
  regional shares; where they differ, take religion within group from the survey and group sizes from the
  census. Caught by: printed in `uz.py::main` and `tm.py::main`, not asserted (Not checked yet; belongs in
  `cab.py`). Detail: spec §12 "A SURVEY DRAWN ON AN OLD FRAME CAN BE RIGHT ABOUT HOW PEOPLE ANSWER" and "A
  SURVEY'S CAPITAL SAMPLE CAN CARRY A MINORITY AT 2.5 TIMES THE CENSUS".
- **The groups have to exist on both sides.** Uzbekistan's census has no Ukrainian row and the survey no
  Turkmen code (its Turkmens answer `Other (vol.)`). Turkmenistan's `other` cell rate, 4.66% Christian on 7
  respondents, is applied to 85,384 census Balochi and draws about 4,000 of Mary's Christians. Put each
  census group where its religion belongs and print what the mismatch moves. Caught by: `uz.py::main` and
  `tm.py::load` stop on an `Ethnic_M` label with no group; the rate mismatch is Not checked yet. Detail:
  `sources/uz.md` §9; `sources/tm.md` §11.
- **A held-out check against a current census can fail a correct decode.** Ashgabat took in part of Ahal in
  2013, so six of 719 orderings beat the truth against 2022. Run `lits.held_out` against the frame the
  methods report prints, and pin the decode with something that uses no names. Caught by:
  `tm.py::frame_witness` (Table 6's PSUs per named region, ten interviews each); `uz_geo.cab_decode_witness`
  (codes 4001-4014 in the office's row order, each label name-matching its own pcode). Detail: spec §12 "A
  SURVEY DRAWN ON AN OLD FRAME"; `sources/tm.md` §3.
- **The chi-square cannot see a cluster or an interviewer.** Uzbekistan's `Other (vol.)` passed the
  split-half (p 0.043) and the chi-square (6e-21) with 11 of 18 answers from wave 4 Bukhara, all recorded by
  one interviewer. Turkmenistan's `A non-believer` passed on 7 answers, 4 in one wave-velayat cell. Placed
  answers had at most 22-28% in their largest cell; the refused ones 57-61%. Caught by: `cab.stability`
  (stops on a pass with over half its respondents in one (wave, unit) cell unless the caller names it in
  `cell_refused`, as `tm.py` passes `REFUSED`). Sampling point and interviewer are still only printed, in
  `tm.py::concentration` and `uz.py::main`, and `uz.py::OVERRIDE` refuses by hand. Detail: spec §12 "A CHI-SQUARE CANNOT VETO A CLUSTER"; `sources/uz.md` §8.
- **The quota test passes without testing anything unless it is told the waves.**
  `arabbarometer.quota_agreement` used to pair only Arab Barometer wave names, so it compared nothing and
  passed. Run it on the whole answer column, don't-knows included. Caught by:
  `cab.assert_not_quota(df, country, waves)` stops unless every pair was compared. Detail: `sources/cab.py`
  docstring; spec §12 "A SPLIT-HALF CANNOT SEE A QUOTA".
- **Six units give the split-half no power.** Turkmenistan's Muslim and Christian both land at +0.600
  against a null 95th of +0.600 (p 0.0665). The witnesses there are the census nationality table and an
  independent wave's ordering, with an exact p over all 720 orderings. Caught by: `tm.py::main` asserts
  `EXPECTED_PASS`; `tm.py::exact_rank` prints the ordering witness, not asserted. Detail: `sources/tm.md` §4, §5.
- **Sparse minorities cannot take a split-half; test them on two units.** Uzbekistan's Russians leave 56 of
  84 (wave, region) cells empty. Test capital against rest, shuffling whole sampling points within wave, with
  the chi-square as a veto. Caught by: `cab.stability` stops on an empty (wave, unit) cell;
  `uz.py::two_unit_test`. Detail: spec §12 "A SURVEY'S CAPITAL SAMPLE"; `sources/uz.md` §9.
- **A card with the code can still miss the community.** In Gorno-Badakhshan, seat of the Ismailis, the
  barometer recorded 40 of 51 Muslims as Sunni and 3 as Ismaili. Check that the category with the sharpest
  known geography comes out highest in its home unit. Caught by: `uz.py::main` (Christian highest in
  Tashkent city), `tm.py::main` (in Ashgabat). Detail: spec §12 "A CARD THAT HAS THE CODE CAN STILL FAIL TO
  SEE THE COMMUNITY".
- **Adults only.** Respondents are 18 and over and the census counts all ages, and the minorities that answer
  Christian are older. Caught by: Not checked yet in `uz` or `tm` (`kg.py::age_lean` is the pattern).
  Detail: `sources/uz.md` §9 "Not done"; `sources/tm.md` §10.
- **One `Christian` code holds the Orthodox and the converts.** The pressured groups (Protestant and
  Jehovah's Witness converts from Muslim families) cannot be located, and a Turkmen Christian is drawn at one
  national rate everywhere. Say so in the record. Caught by: a rule. Detail: `sources/tm.md` §8.

## Shared code
Import these, do not copy them.
- `sources/cab.py`: `load`, `read_wave`, `national`, `assert_not_quota`, `stability`, `shares`, `compose`
  (a fine level, a coarse level and a tail at national proportions, closing every unit), `counts`.
- `sources/lits.py`: `held_out` (exhaustive up to a million orderings, forgives swaps within sampling error)
  and `lean` (§3.5 with leave-one-out).
- `cab.stability` today resamples waves (a `SamPt` re-run in `uz` kept every verdict): unweighted counts,
  median Spearman over every distinct halving (an odd wave count keeps all of them since 2026-09-14), a null
  that permutes unit labels independently within each wave (2,000 draws), and the spatial chi-square under
  0.05 as a veto. It stops on an empty (wave, unit) cell, and on a pass with over half its respondents in
  one (wave, unit) cell that the caller has not named in `cell_refused`. It computes through
  `sources/stability.py`.
- `uz.py::two_unit_test`, `tm.py::concentration` and `tm.py::exact_rank` read module globals: copy them and
  name the source.

## Rulings
- **Turkmenistan** (Anita, 2026-09-14, `queue.md` "Anita's rulings, 2026-09-14 evening"): build it, Christian
  geography included, against a split-half with no power on six units: *"given the state of data in
  turkmenistan in general we're unlikely to be able to do any better than this."* Does not decide that a
  powerless split-half is enough for another country, or the Balochi rate in `sources/tm.md` §11.
- **Uzbekistan by ethnic group** (Anita, 2026-09-14, `sources/uz.md` §9): where a census counts a group whose
  religion differs sharply, draw by group, as the citizenship splits do. Does not decide the adult-only lean
  or any placement question.
- No ruling on access is needed: the Discuss Data mirror has no account or form.
