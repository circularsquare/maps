# Afrobarometer playbook

Afrobarometer's merged rounds 4 to 9 (2008-2023): about 30 religion answers, cut by `REGION`
(ADM1, named), in open files. It is the route for an African country whose census never asked
religion or never published it below the nation: the survey gives each unit's mix, and a
population table gives the people.

## Used by
- `ng` Nigeria: drawn, 37 states, R4-R9 on COD-PS 2022; each state at its own mix, no column
  margin, small categories at the national rate. Since 2026-10-03 Catholics from R4-R6 (four NDHS
  reports as witness) and Anglicans levelled by GFS; Pentecostals tested and not drawn.
- `lr` Liberia: drawn, 15 counties; the county pattern fitted by IPF to two 2022 census margins.
- `tz` Tanzania: drawn, 30 regions, R4 and R6-R9 on the 2022 census; R4 placed by district, R5
  left out; the first build on `cab.stability`. Since 2026-10-03 five churches on the mainland,
  levels from the GFS 2023's church item, places from it and R4, R6, R8, R9 together
  (`sources/tz.md` §9).
- `cm` Cameroon: drawn, 12 units (the regions with Yaoundé and Douala apart), R5-R9 on COD-PS
  2025; the first build to draw churches (Presbyterian, Baptist), on level by round, the census's
  Protestant total and the unnamed share where each church lives (`sources/cm.md` §4); since
  2026-10-03 Catholics from R5-R6, the rounds matching the DHS 2011 and 2018 (§9).
- `mz` Mozambique: drawn from the census; Afrobarometer was the witness that put `Sem religião`
  at 93-98% no religion (`sources/mz.md` §6).
- `mg` Madagascar: drawn, 22 regions, R5-R7 and R9 on the 2018 census; Catholic, FJKM and Lutheran
  as churches (`Christian only` 0.3-2.3% by round, both DHS surveys witnessing the
  Catholic/Protestant level); None and traditional placed as one box and split at one national
  ratio (`sources/mg.md` §5); R4 left out (no Betsiboka district sampled).
- `tg` Togo: drawn, 6 units (Lomé apart), R5-R9 fitted by IPF to the 2022 census's national rows
  (UNSD table 28) and unit populations, as Liberia. With a census margin the churches' levels come
  from the census, so Pentecostal and Presbyterian are drawn as patterns (`sources/tg.md` §4).
- `sd` Sudan: drawn, 18 states, R5-R9 pooled with Arab Barometer V and VII on COD-PS 2022 at one
  national share; R5 is 15 pre-2012 states, R6-R8 six regions, R9's `LOCATION.LEVEL.1` the 18
  states (`sources/sd.md`).
- `ga` Gabon: drawn, 9 provinces, R6-R9, citizens only on the RGPL 2026 count (foreigners, 34.1%,
  in `gap`); the split-half passes nothing, two standouts drawn, Christians split at the DHS
  2019-21 national ratio (`sources/ga.md`).
- `na` Namibia: drawn, 14 regions (13 survey mixes, Kavango whole in R4-R5), R4-R9 on the 2023
  census; Lutheran and Anglican from R5-R6 only, the rounds that match the DHS 2013's ELCIN share,
  as a share of the Christian pool they trade with (`sources/na.md` §3).
- `ls` Lesotho: drawn, 10 districts, R4-R9 on the 2016 census; Catholic, Anglican, Methodist,
  Pentecostal and Zionist/independent placed, the LEC (two boxes in R9) in the residual, both DHS
  reports as the level witness; round 6's northern traditional block dropped (`sources/ls.md`).
- `bi` Burundi: drawn, 17 provinces of 2008, R5-R6 seeding a three-way fit to the 2008 census's
  religion rows by urban and rural, province populations by urban and rural, and collective
  households (`sources/bi.md` §0).
- `bw` Botswana: drawn from the 2011 census by locality; since 2026-10-03 Catholics and Adventists
  are split out of its Christians at R4-R8 unit shares, `derived` inside the counted column, the
  two churches that keep their share while `Christian only` climbs (`sources/bw.md` §10).
- `km` Comoros: drawn, 3 islands at one national mix from round 10's summary of results (2025),
  its first round; the data set is not released, so no island split (`sources/km.md`).
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
- **A round's REGION labels can be the first N official names, not the units sampled.** Algeria R5:
  codes 1420-1455 carry wilayas 1 to 36 in official order, so Ghardaïa (47) is missing, Algiers has 30
  respondents and Tamanrasset 128, and the 10 Ibadi answers land in "Boumerdes". Compare each label's
  respondents with its population share before decoding by name. Not caught by anything yet. Detail:
  `sources.md` §maghreb-2026-09-16.
- **`held_out` stops a correct six-unit decode.** Six units allow 720 orderings, and a sample frame
  from an older census moves the capital's share (Togo: Lomé sampled at 1.40x its 2022 share, 14
  orderings reach the observed r). Where REGION labels are names, make the location column the
  witness and say so in code. Caught by: `tg.py::check_locations` (asserts the one known
  disagreement); nothing shared. Detail: `sources/tg.md` §4.
- **A box can stay a value label after a country stops using it.** `Assembly of God` is a label in
  every round and Togolese chose it only in R5 and R6: the labels are the merged file's, not one
  country's card. Read the country's own counts by round before assuming a box was offered. Caught
  by: `tg.py::report_card` (prints). Detail: `sources/tg.md` §4.
- **One region can carry a round's whole `None` box.** Sudan R8: 31 of 32 `None` answers are in
  Darfur (30 rural), 7.2% of its interviews there, against 3 in about 2,400 Darfur interviews in the
  other rounds and the Arab Barometer; R8 has no location below the region. Tabulate each small box
  by region and round before pooling, and drop a block no other round or instrument reproduces,
  counted and asserted. Caught by: `sd.py::load_afro` (`R8_DARFUR_NONE`); nothing shared. Detail:
  `sources/sd.md` §3. The same shape in Lesotho R6: `Traditional` at 10-25% in three northern
  districts and under 4% in every other (round, district) cell, with the Zionist and `Independent`
  boxes empty there that round (`ls.py::drop_r6_north_traditional`, `R6_NORTH_TRAD`).
- **One church can split across two boxes in a single round.** Lesotho's Evangelical Church is
  `Evangelical` 17-22% through R8; in R9 its members choose `Calvinist` (13.1%) and `Evangelical`
  (9.6%) in every district, so either box alone reads as the church halving. Sum candidate boxes by
  round before calling a level unstable, and check both appear in every unit. Caught by:
  `ls.py::levels_by_round` (the grouped level's range); nothing shared. Detail: `sources/ls.md` §3.
- **A church can be coded right in some rounds and not others, and an open DHS report can say which.**
  Namibia's `Lutheran` runs 42.8% and 41.9% in R5-R6 and 20-24% in R4 and R7-R9, where ELCIN members
  went to `Evangelical` (the church's name), `Christian only` or `Anglican`, a different box each round.
  The DHS final report's Table 3.1 often names the big church (ELCIN 43.9%) even when the dataset is
  gated. Where some rounds match the witness, take that church's unit shares from those rounds only, as
  a share of the pool it trades with, and the pool from all rounds; assert both the match and the
  other rounds' gap. One halving of two rounds is a weak split-half (Namibia's Catholics fail it there
  and pass over six rounds), so take only the churches that need it from the subset. Caught by:
  `na.py::levels_by_round`, `na.py::carved_shares`; nothing shared. Detail: `sources/na.md` §3.
- **Where `Christian only` is large, the Global Flourishing Study can level the churches.** GFS wave 1
  (on disk, CC BY) asks every Christian `REL3_Y1`, the church they most identify with (codebook wave
  2, OSF 285w7, p.21: Catholic, Orthodox, Anglican, Presbyterian, Lutheran, Methodist, Baptist,
  Pentecostal/Charismatic, Independent/Evangelical, LDS, Jehovah's Witness, Adventist, African
  Initiated, other, none), by `REGION1_Y1`. Tanzania: 0.5% named none against the Afrobarometer's
  21%, and the unnamed turned out to be mostly Pentecostal (61.5%) and not Catholic, so spreading
  them at the named proportions would have halved Pentecostals. Test the two surveys as the two
  halves (rank over units) and check each GFS level lies between the Afrobarometer's named share and
  that plus `Christian only`. African GFS countries: Kenya, Nigeria, South Africa, Tanzania. Caught
  by: `tz.py::church_shares` (both assertions); nothing shared. Detail: `sources/tz.md` §9.
- **Test a church on the basis it is drawn on.** Nigeria's Pentecostals rank the states alike in
  GFS and Afrobarometer as a share of ALL Christians (+0.40, p 0.02) and not as a share of the
  non-Catholic Christians they are carved from (+0.20, p 0.15): the first is mostly the Catholic
  pattern inverted. Where one church is carved first, test the next on what is left. Also: GFS
  wave 1's Catholic level can disagree with every DHS (Nigeria 27.9% against 19-24%; prefer the
  DHS), GFS small boxes can be one team's coding (Plateau 87% `Orthodox`), and its `REGION1_Y1`
  labels are not in the codebook PDF but in the `.sav`'s value labels (the first 40 MB of OSF
  `eadfm` suffice). Caught by: `ng.py::church_split` (`CHURCHES_CARRIED`). Detail: `sources/ng.md`.
- **A DHS report that names only `Catholic` and `Other Christian` is still a Catholic witness**, and
  four of them (Nigeria 2008-2024) agree within 4.5 points. Compare Catholics as a share of
  Christians, sexes pooled by weighted n, round by round; never spread `Christian only` over the
  named churches without checking it against that share (Nigeria's named Catholics are 29-66% of
  named Christians, the DHS's 19-24% of all). The Church's diocesan figures summed by cathedral
  region are a rank witness, never a level (`sources/gcatholic.py`; Nigeria +0.83, Cameroon +0.64).
  Caught by: `ng.py::catholic_fraction`, `cm.py::carve_catholic`. Detail: `sources/ng.md`, `sources/cm.md` §9.
- **Where `Christian only` climbs by round, test which churches keep their share of ALL
  respondents.** Botswana's unnamed run 5% to 57% of respondents over R4-R8; rounds 7-8 against 4-5,
  named Christians keep 0.45, Catholics 0.78, Adventists 0.81, Lutherans 0.63, Pentecostals 0.57. A
  church that keeps its share is one whose members go on naming it, so its named share is near its
  level and can be drawn as a floor, with the unnamed left on the parent; run the split-half on the
  church as a share of ALL the unit's Christians, the quantity drawn (the ZCC passes on named
  Christians and fails on all). The card can also drop a big church's box between rounds (UCCSA
  after R5). The Pew Forum's 2008-09 Africa survey (*Tolerance and Tension*, p.23) prints churches
  per country but its Botswana Catholics (22%) are off by a factor of three to four. Caught by:
  `bw_churches.py::main` (`HOLD_MIN`, asserted both ways); nothing shared. Detail: `sources/bw.md` §10.
- **Two standouts that are complements break the standout residual.** `tz.py::compose` fixes each
  standout at the other units' pooled share everywhere and fills the rest with the tail. Gabon's
  standouts are Christian (Woleu-Ntem) and None (Nyanga): in Nyanga 82% + 28% > 100%, and the
  residual goes negative with no error from the arithmetic itself. Start every unit from one base
  mix (each category pooled where it is not a standout), let a standout unit keep its own share and
  scale the rest to fill. Caught by: `ga.py::compose` (asserts no negative share); `tz.compose`
  has no such assertion. Detail: `sources/ga.md` §3.
- **A census urban/rural religion table is a third margin, and a carried survey pattern counts the
  town twice unless it is split first.** Burundi prints religion only nationally, by urban and rural
  (Muslims 14.3% urban, 1.3% rural). Fitting province x urban/rural x religion with the survey's
  province share seeded flat across both milieus put 20.0% Muslims in Bujumbura Mairie (survey 12.6%,
  n=175) and 4.9% in every other town, because the province shares already contain their towns.
  Split each carried province share by the survey's own national urban and rural multiples (keeping
  the province total), then fit: Mairie 14.4%, other towns 14.1%, census 14.3%. Rows not carried
  are seeded flat and take the census's milieu shares. Caught by: `bi.py::main` (`TOWN_BAND`;
  `--flat-milieu` reruns the rejected seed and stops on it), `bi.py::ipf3`; nothing shared. Detail:
  `sources/bi.md` §0.3.
- **The survey samples citizens; the census may count a third of a country as foreign.** Gabon's
  2026 count is 34.1% foreign residents; Afrobarometer's frame is citizens 18+. Draw the citizens
  and put the foreigners in `gap` (Libya ruling); where foreigners are counted by province only in
  an older census, fit the province split to the new national margin (`ga_geo.py::split_citizens`).
- **A country new in round 10 is published as PDFs before its data set.** Comoros R10 (fieldwork
  May-June 2025) has a *Résumé des résultats* (weighted religion line in the sample description,
  and the religion question in full) and a codebook (unweighted counts per code) on its country
  page, while `afrobarometer.org/data/data-sets/?select-countries[]=comoros` lists nothing
  (2026-10-03; the same filter lists 13 for Namibia, so the empty answer is real). The summary
  gives a national mix at one decimal and no `REGION` cut, so the build is one national mix until
  the file appears; re-check the data-sets filter before building. Caught by: `sources/km.py`
  (both PDFs read back and asserted against each other). Example: `km`.

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
