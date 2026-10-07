# Nigeria — a state pattern from the pooled Afrobarometer, on a projected population

**Drawn 2026-09-09.** 37 states, 5 categories, 216,798,930 people, every row `modelled`. The
largest country on this map by population, and the only one of the ten largest whose state
publishes no religion figure at all.

- `sources/ng_geo.py` -> `data/geo/ng/ng_states.gpkg`, `ng_lookup.csv` (COD-AB ADM1 + COD-PS
  2022 populations, joined on pcode and asserted on the names as well)
- `sources/ng_grid.py` -> `data/geo/ng/ng_hexes.gpkg` (Kontur 400 m, 635,561 hexes keyed to
  state)
- `sources/afrobarometer.py` -> `data/raw/afrobarometer/*.sav` (six merged rounds, ~280 MB,
  shared with every other country drawn from that survey)
- `sources/ng.py` -> `data/normalized/ng.csv`
- `taxonomy/ng2022.py` -> the mapping; `countries.py` `"ng"` -> the wiring
- sources.md **§9cv** is the write-up, **§11ai** assesses the Afrobarometer for Africa, and
  **§11p** is the continental sweep whose flat *"Nigeria does not ask"* this does not
  overturn.

```
python sources/ng_geo.py  --fetch
python sources/ng_grid.py --fetch
python sources/ng.py      --fetch      # once; the six .sav files are shared
python sources/ng.py
```

## 1. What Nigeria publishes, and it is nothing

**Nigeria has published no religion figure from a census since 1963, and it is not for want of
asking once.**

| census | religion | what happened |
|---|---|---|
| 1963 | asked, published by region | the last public figure of any kind |
| **1973** | **asked** | annulled in 1975 with nothing released, amid allegations that the returns had been falsified |
| 1991 | not asked | |
| **2006** | **not asked** | 140,431,790 people, gazetted 2009-02-02 (Extraordinary Gazette No. 2, Vol. 96) |
| 2023 | announced, postponed, not held | NPC has said it would not carry a religion question |

**Nigeria is ABSENT from the UNSD oracle** (`python tools/oracle.py Nigeria`), which is
consistent with the above and proves only that no tabulation was ever forwarded.

The reason is not obscure and is the same one every source gives: revenue allocation and the
federal-character rule share national resources and national offices out by population, so a
count of Christians and Muslims is also a count of who is owed what. §3 of `sources.md` has
carried the row *"census does not ask, deliberately, and the Christian/Muslim balance is
politically explosive"* since the file was started, and it is still right.

## 2. What was searched, and where each route ended

| route | date | outcome |
|---|---|---|
| UNSD Demographic Yearbook table 28 | 2026-09-09 | Nigeria absent |
| **NDHS 2018 final report FR359** (748 pp, `dhsprogram.com/pubs/pdf/FR359/FR359.pdf`, open, no account) | 2026-09-09 | **religion appears in exactly one table**, Table 3.1 on p. 51, as a row block beside the Age block and the State block rather than crossed with either. 41,821 women and 11,867 men aged 15-49, national only. The one other caption mentioning religion is Table 18.9, on opinions about female circumcision. §11ai said this and it is confirmed to the page. |
| DHS recode microdata (`v130` religion x `v024` region) | | **registration-walled**, institutional, with a stated research purpose. §11ag's assessment stands and it is Anita's, not an agent's. |
| DHS open API / STATcompiler | | §11ai measured it: 4,655 indicators, none of them a religion composition, and `breakdown=background` offers no Religion category for Nigeria 2018. There is no way to get the cross out of it. |
| IPUMS `ng2010` (the 2010-11 General Household Survey, 72,191 persons, state in practice) | | **blocked**, `[[reference_ipums_account]]`. §11 called this *"still the best subnational religion evidence for Nigeria that exists"* and it remains unreachable. |
| Pew Research Center, *5 facts about religion in Nigeria*, 2026-11-11 (carrying the June 2025 global report's 2020 estimates) | 2026-09-09 | national only: Muslims **56.1%** and about 120 million, Christians **43.4%** and about 93 million, everything else 0.6%. A synthesis, not a count, and Pew says so. Used as a comparison, never as a margin. |
| NBS `nigerianstat.gov.ng`, *Religion and Related Activities Statistics* (`/download/27`, 6 pp) | 2026-09-09 | **read, and it is a PROPOSAL rather than a publication.** The document sets out a data structure NBS would like to exist: ISIC division 94, twenty-five proposed items and about 500 variables, of which `4301 Registered Members of Christian Religion by States` with 37 details is one. Its own concluding remarks say *"presently, virtual nothing is being done in terms of data collection by the various agencies concerned with managing religious matters in the country"*, and §4 says the datasets that do exist are *"kept in disparate locations as hard copies in files"* at the Corporate Affairs Commission, the Ministry of Internal Affairs and the Pilgrims Welfare Boards. **So there is no NBS religion table, and there is a published NBS statement that there is none.** It would have been a `roll` and a different basis from the survey (§3.1) even if it existed. |
| — the access note, which is a second sighting | 2026-09-09 | `nigerianstat.gov.ng` **closes the connection on a bare urllib GET** (`RemoteDisconnected`) and its e-library page times out at 60 s, both of which read as a dead host. It answers 200 to `requests` with a **complete browser header set** (Accept, Accept-Language, Accept-Encoding, Sec-Fetch-*), which is §9cp's Costa Rica finding again: the wall is keyed on header COMPLETENESS and not on the User-Agent, so the usual UA retry does not open it. |
| **NGA Human Geography Information Survey**, `Nigeria_Religion_Points` / `Nigeria_Religion_Areas` on ArcGIS Online (§11f) | | presence, not counts. A §4 location source and no use for magnitude. |

**So the only open sub-national religion data for Nigeria is a survey, and the Afrobarometer
is the one that needs no account.** Access, terms and citation are `sources/afrobarometer.py`'s
and were read before anything was downloaded.

## 3. The construction, and the one thing about it that is not obvious

    row margin      state populations       COD-PS 2022 (UNFPA/NPC)      EXACT
    the composition each state's own mix    Afrobarometer R4-R9 pooled   measured, n=11,909
    the national level                      neither                      computed

Each state is drawn at the Christian-to-Muslim ratio its own respondents gave, at its COD-PS
population, with the three small categories at the national rate everywhere. **Nothing fits a
column margin**, which is the whole difference from `sources/lr.md` on the same instrument:
Liberia had a census religion total and Nigeria has nothing.

### THE NATIONAL BALANCE IS 51.4 / 47.9 AND THE SURVEY'S OWN IS 56.0 / 43.3

**This is the finding that generalises and it nearly went the other way.** The first version
fitted the state-by-religion table to two margins by IPF, on Liberia's pattern, with the column
margin taken from the Afrobarometer's own pooled national shares. That is wrong, and the reason
is worth stating in full because it looks harmless:

> With a population table as the row margin, fitting the columns back to the survey's own
> national total **undoes the reweighting the row margin just did.** The pool's state mix is
> not the population table's, so those are two different weightings of the same states, and
> the fit bends every state's measured share to reconcile them.

Two mechanisms put the pool's state mix out of line with COD-PS 2022, and both are documented
rather than suspected:

- **Round 6 has no Adamawa, Borno or Yobe.** It was in the field in December 2014 and January
  2015, when Borno and Yobe were substantially outside federal control. That removes about 14
  million mostly-Muslim people from one of the six rounds.
- **Pooling fourteen years averages over a period in which the north grew fastest.** Against
  2006, COD-PS 2022 has Katsina at x1.79 and Bauchi at x1.79 while Akwa Ibom is x1.28 and Osun
  x1.30, so a pool whose rounds were each allocated against their own year's population
  under-weights the north relative to 2022.

`held_out` reports the consequence without touching the religion column: Yobe is sampled at
**0.63x** its COD-PS population share and Bayelsa at 1.37x. Recomposing per state moves the
drawn national balance 4.6 points, from 56.0% Christian to **51.4%**.

**`ab.build` already does it this way**, so no country already drawn from LAPOP, the Arab
Barometer or the Afrobarometer is affected; the IPF was this file's own deviation. Liberia's
IPF is a different and correct thing, because both of its margins are census totals over the
same 5,250,187 people.

### WHY IT IS NOT FITTED TO THE NDHS OR TO PEW

Both were considered and both were refused, and the reasons are not the same.

- **Pew 2020** is `estimate` in spec §3.1's own table, and §3.1 lets another basis *split* a
  category but never set its magnitude. Worse, Pew's Nigeria figure is a synthesis over the
  DHS and this same survey, so fitting this survey's geography to it would be laundering one
  witness through the other.
- **NDHS 2018** is the more tempting one: same basis, 53,688 respondents against 11,909, and
  state-representative by design. Its universe kills it. Table 3.1 covers ages 15 to 49 and
  nothing else, so it sees neither the under-15s (about 43% of Nigeria) nor anyone over 49,
  and the row margin here is a whole-population state table, so fitting to it would *require*
  the scale-up **§3.4 refused for Brazil** when the 2022 census asked only the 10-and-overs.
  §11ah priced exactly this gap for Haiti against a census that crossed religion with age;
  Nigeria has no such census, so here it cannot be priced at all.

So all four figures are printed on every build and all four reach the reader:

| | Christian | Muslim |
|---|---|---|
| **this map, as drawn** | **51.4%** | **47.9%** |
| Afrobarometer R4-R9, its own pooled weighting | 56.0% | 43.3% |
| NDHS 2018 FR359 Table 3.1, women 15-49 | 46.0% | 53.5% |
| Pew Research Center, 2020 | 43.4% | 56.1% |

An 8.2-point spread on the Muslim share, which over 217 million people is 18 million of them.
Ask 010 (`ask/answered/010-ng-should-the-map-publish-a-state-level-christi.md`) put the §14
question about publishing any of this to Anita; the decision taken
meanwhile is to draw it and to say all four numbers.

## 4. The checks, and what each is worth

- **The decode.** `held_out`: r = +0.923 across 37 units against COD-PS, and **0 of 20,000
  random pairings** of the same units reach it. Nothing in it touches the religion column.
- **The geography.** Split-half across round halves (§14.16): Christianity **+0.960**, Islam
  **+0.952**, against a bar of +0.327. The three small categories are under the 1% eligibility
  floor and are never tested at all.
- **That bar is the OLD one, and §9ct says so on purpose.** Anita's ruling on `ask/007-cr`
  replaced `1.96/sqrt(n-1)` with the exact permutation critical value in `sources/lapop.py`
  and `sources/arabbarometer.py` and **deliberately left `sources/afrobarometer.py` alone**,
  because switching a bar under already-drawn countries needs a before-and-after list first.
  Nigeria does not care: the exact bar at 37 units is lower than +0.327, both drawn categories
  clear either by more than half a unit of correlation, and the three that do not carry a
  geography are excluded by the eligibility floor before any bar is applied. Liberia, the only
  other Afrobarometer country, is the same case. So this file changes nothing and inherits the
  open item.
- **The outside evidence, and it is a legal record rather than a table.** Twelve northern
  states extended the Sharia penal code to criminal matters between 1999 and 2001 and
  twenty-five did not. The survey makes **every one of the twelve** Muslim-majority without
  having been told which twelve they are; a random assignment of the fifteen Muslim-majority
  labels it produces would cover a named twelve about once in four million. The second leg is
  a median-to-median gap, 93.4% against 9.2%, and `check_sharia()` says plainly that this one
  is **a smoke test on the decode rather than corroboration of the fine geography**.
- **What this check is NOT.** Two sharper forms were written first and both failed on this
  data, and neither was tuned to pass: *"Kaduna is the least Muslim of the twelve"*, which its
  LGA-by-LGA application implies, fails because Gombe comes in 1.1 points below it; *"the
  twelve are separated from every non-adopting state with no overlap"* fails because Adamawa,
  on 200 pooled respondents, comes in at 68.0% against Gombe's 66.3%. Both are recorded in
  `sources/ng.py` next to the constant.

## 5. What is deliberately not drawn

- **Every Christian denomination but the Catholic Church** (Catholics drawn since 2026-10-03,
  see "Catholics, 2026-10-03" below). The card names about twenty and Nigerians fill nearly all
  of them, Roman Catholic at 8.6% of respondents and Pentecostal at 4.8%. The share answering
  `Christian only` rather than naming one runs 19.4% to 47.3% between rounds with no trend, so
  a pooled denominational share measures the fieldwork (§11ai). Anglican and Pentecostal were
  re-tested on 2026-10-03 and stay folded (below).
- **Sunni and Shia, and the brotherhoods.** Same reason, plus a second: `report_card()` reads
  each round's value-label set and finds **Izala on three of the six cards** and the Shia box
  renamed between rounds (`Shia only` in R4 and R5, `Shia` in R6 to R9). A zero in a crosstab
  is a fact about respondents; this is a fact about the questionnaire, and it is now read
  rather than inferred.
- **Shia specifically is also a §14.4 rule 2 refusal.** 58 respondents over fourteen years, in
  a country where the Islamic Movement in Nigeria has been proscribed since 2019 and its
  members killed in numbers. Neither reason needs the other.

## 6. Three states are drawn with no Muslims

Abia, Cross River and Ebonyi. None of the 198 to 256 people interviewed in each of them across
six rounds was Muslim, and §3.5 drops rather than invents. Every large Nigerian town has a
northern trading quarter, so `note_public` tells the reader to read those as states where the
survey found none rather than as states with none. This is `sources/lr.md`'s River Cess, three
times over and on much bigger states.

## 7. What would improve it

1. **The DHS recode microdata.** `v130` x `v024` on NDHS 2018 would give a 40,000-household,
   state-representative religion cut, which is strictly better than this on every axis. It
   needs an institutional registration with a stated research purpose, so it is Anita's, and
   §11ag's unconfirmed claim that DHS suspended new applications on 2025-02-07 should be
   checked before she spends any time on it.
2. **MICS 2021** (NBS and UNICEF) is the same shape and the same kind of wall.
3. **The 2010-11 GHS** via IPUMS, blocked on the account.
4. **Below the state.** Nothing measures religion at LGA level in any open source. COD-AB
   ships 774 LGAs and 9,000-odd wards in the same bundle already downloaded, so the geography
   is free the day a source appears; the Afrobarometer's geocoded extracts are gated and are
   the obvious first place to look.
5. ~~The NBS e-library~~ is **closed with a citable negative**, see §2: NBS's own religion
   document is a proposal for a database that does not exist, and says so.

## Placement: blocks at Kontur's density cap, 2026-09-14 (session `f95259a4-kontur`): Gombe capped, Okene and others listed

Kontur limits every hex to 46,200 people/km², and a block of hexes at that limit is either a real
dense core or a false concentration (spec §12, "KONTUR'S DENSITY CAP"). **Gombe state (NG016) had
two blocks in farmland east of Gombe town: 7 hexes with 6 at the limit holding 229,617 people
(6.3% of the state's placement weight), and 5 hexes with 4 at the limit holding 162,179 (4.4%)**,
6.5 and 7.2 km from the centre. Kontur within 5 km of the town's own centre never exceeds
4,859/km².

Both are `capped` in `kontur_cap.csv`, and `scatter.py` now lowers each hex to the median density
of the populated hexes within 3 km, 354/km² and 153/km², which leaves the blocks 1,786 and 553
people. Counts did not move: dots per node are identical before and after, 216,796 at 1:1,000
and 21,678 at 1:10,000.

**Listed as `unreviewed` and not changed.** Every scatter warns about these and draws them as
Kontur has them:

| unit | where | hexes (at the limit) | share of the unit's placement weight |
|---|---|---|---|
| NG023 | Okene, 6.7 km southwest of the centre | 12 (11) | **6.8%**; Kontur at the centre peaks at 11,811/km² |
| NG004 | 10 km from Onitsha | 11 (1) | 3.2% |
| NG003 | no town within 20 km | 6 (4) | 2.5% |
| NG033 | 17 km from Choba | 6 (3) | 1.6% |
| NG012, NG023 | single hexes west of Okene | 1 (1) each | 0.6% each |
| NG023 | single hex 3 km from Okene | 1 (1) | 0.6% |
| NG033 | single hex 16 km from Choba | 1 (1) | 0.3% |

Lagos, Kano and the other blocks at the limit are registered `real`. To act on an `unreviewed`
block: set its `status` to `capped` and re-scatter.

## Catholics and Anglicans, 2026-10-03 (session `fafd1067-chwa`)

On Anita's 2026-10-03 ruling (re-search churches for ng, cm, tz, bw). Two churches are now their own
nodes: **Catholic (`christianity.catholic`) 10.21% of Nigerians, 22,132,249 people, 19.9% of
Christians**, from Afrobarometer rounds 4-6 with four NDHS reports as the level witness; and
**Anglican (`christianity.anglican`) 4.18%, 9,064,122, 8.1% of Christians**, levelled by the Global
Flourishing Study. `christianity` keeps 37.02%. Code: `sources/ng.py` (`dhs_witness`,
`catholic_fraction`, `church_split`, `diocese_witness`), the new shared `sources/gcatholic.py`,
`taxonomy/ng2022.py`, `countries/ng.py`. Nothing else moved: the Christian/Muslim balance, the
tail and every state total are as before (asserted: the three Christian rows sum to the old
`Christian` per state, within rounding).

### The witness: four open NDHS final reports

Each names `Catholic` and `Other Christian` in Table 3.1 (women and men 15-49), no other church.
All four are now in `data/raw/ng/` and re-read on every build.

| NDHS | file, PDF page | Catholic, women / men | Other Christian | Catholics as % of Christians |
|---|---|---|---|---|
| 2008 | FR222 p.63 | 11.5 / 11.6 | 42.1 / 42.1 | 21.5 |
| 2013 | FR293 p.59 | 11.1 / 11.6 | 35.7 / 35.6 | 23.9 |
| 2018 | FR359 p.91 | 10.4 / 11.3 | 35.6 / 34.5 | 23.1 |
| 2024 | FR395 p.93 | 8.2 / 7.6 | 33.7 / 33.2 | 19.4 |

FR395 is the 2023-24 NDHS final report (October 2025); its Islam row is 57.6% of women and 58.2% of
men, against 53.5% in 2018. Not used for the Christian/Muslim balance, which is unchanged; worth a
line in §3's table on the next rebuild.

### Which rounds

Catholics as a share of the survey's Christians: R4 21.2%, R5 18.9%, R6 19.1%, then R7 8.4%, R8 4.3%,
R9 12.5%, against an NDHS mean of 22.0%. Rounds 4-6 sit within 3.1 points of it; rounds 7-9 are 9.5
to 17.6 points under, as `Christian only` swells (32, 19, 28% of respondents in R4-R6; 45, 47, 41%
in R7-R9). That is Namibia's case (`sources/na.md` §3), so each state's Catholics are its Catholic
share of Christians in rounds 4-6 times its Christian share from all six rounds. Asserted both ways
(`CATH_DHS_GAP_MAX` 4 points, `CATH_SWING_MIN` 8).

- **Split-half**, Christians in rounds 4-6, the 27 states with Christian respondents in all three:
  **+0.753** against a null 95th of +0.265 (p 0.0005). Over all six rounds it is +0.818, so the
  geography holds while the level moves.
- **Thin states**: Kano 4, Jigawa 5, Yobe 5, Zamfara 6, Katsina 7, Kebbi 7, Sokoto 14 and Adamawa 14
  Christian respondents in rounds 4-6. They take the national 19.7% (`MIN_CHRISTIANS` 30; every
  other state has at least 40). Adamawa's own figure was 18.8%, so it barely matters there.
- **Zero cells**: Kwara (0 of 40 Christian respondents) and Osun (0 of 84) are drawn with no
  Catholics. Osun's diocese claims 3.2% of its people; on the survey's Osun Christian share that is
  about 6 expected respondents, under the playbook's 8, so the zero stands as the survey's.
- **Level as drawn**: 19.9% of Christians, 2.1 points under the NDHS mean (`CATH_DRAWN_GAP_MAX` 4).
  10.2% of everyone, inside the NDHS's 7.6-11.6% of respondents.

### How `Christian only` is handled, and why not Togo's way

Togo (`sources/tg.py`) spreads `Christian only` over the named churches of its own unit. Here that
would make Catholics 29-66% of Christians (their share of NAMED Christians by round) against the
NDHS's 19-24% of ALL Christians: the people who name no church are mostly not Catholic. So Catholics
are a share of all Christians, and `Christian only` stays in `Christian` with every other church.
Cameroon's second test (`cm.py::unnamed_where_they_live`): in rounds 4-6 the unnamed share is 33.1%
of Christians averaged where Catholics live, against 46.0% nationally, so nationally they are not
drawn short. **Per state they are**, where the unnamed share is high: Lagos 71%, Oyo 73%, Ogun 66%,
Rivers 58%. Lagos is drawn 5.8% Catholic; its archdiocese claims 25%. Said in the note.

### The Church's own figures as a geography witness

`sources/gcatholic.py` (new, shared) reads every diocese page on gcatholic.org (robots.txt allows it;
one request a second) into `data/raw/ng/gcatholic_dioceses.csv`: 59 territorial dioceses, mostly
2022 figures, summed by the state the cathedral stands in (31 states). **Spearman +0.833** against
the drawn Catholic share (`DIOCESE_RHO_MIN` 0.60). The Church claims 14.6% of its own population
figure (35,097,000 of 222,205,000 at the country level for 2023), far above the NDHS: a count of the
baptised, never a level. Biggest disagreements: Lagos (25/6), Cross River (27/7), Rivers (19/7),
Ekiti (14/3) where the survey's Christians name no church; Delta and Edo the other way (7/23,
11/23), not looked into.

### Anglicans: the Global Flourishing Study as the level

The GFS wave 1 file on disk (`data/raw/gfs/`, CC BY, the one `tz` and `jp` use) has Nigeria as
`COUNTRY` 12: 6,827 respondents, 2023, Christians 50.9% and Muslims 48.4% weighted (this map:
51.4/47.9). `REL3_Y1` asks each Christian the church they most identify with; 1.2% of 3,849 name
none, against the Afrobarometer's 46% `Christian only` in rounds 4-6. `REGION1_Y1` labels were read
from the value labels of the GFS `.sav` (OSF `eadfm`, only its first 40 MB needed for the header),
since the wave-2 codebook PDF (OSF `285w7`) does not list regions; they are in `ng.py::GFS_REGION`.
Adamawa, Borno, Taraba and Yobe have no GFS respondents; the weighted state shares track COD-PS at
r = +0.68, Bauchi and Gombe sampled at 2.6x.

GFS shares of Christians naming a church: Pentecostal 39.9%, Catholic 27.9%, Anglican 7.3%, Orthodox
5.8%, Presbyterian 5.0%, Independent/Evangelical 3.7%, Baptist 3.4%, Methodist 2.6%, the rest under
1.1% each. Two things to know: its Catholic share is above all four NDHS reports (19.4-23.9%), so
it is not used for Catholics; and it codes 87% of Plateau's and 72% of Bauchi's Christians
`Orthodox` and 42% of Gombe's `Independent/Evangelical`, which in those states can only be COCIN and
ECWA. Its small boxes are not used.

Construction (`church_split`, Tanzania's, on the non-Catholic Christians): the GFS level as a share
of non-Catholic Christians, recomposed on the drawn ones; the Afrobarometer's named share in rounds
4-6 plus `Christian only` spread at one national proportion that meets that level; each state the
average of the GFS and Afrobarometer shares weighted by non-Catholic Christian respondents.

| | GFS level | AB named (floor) | AB ceiling | share of unnamed given | two-survey rank, 27 states | |
|---|---:|---:|---:|---:|---:|---|
| Anglican | 10.0% | 8.9% | 66.0% | 1.9% | +0.550, p 0.0014 | drawn |
| Pentecostal | 55.9% | 15.1% | 72.2% | 71.5% | +0.201, p 0.153 | not drawn |

(shares of non-Catholic Christians; `CHURCHES_CARRIED` asserts the verdicts.) Anglican as drawn:
Anambra 22.1%, Enugu 15.6%, Imo 12.8%, Ekiti 12.2%, Delta 12.1%, Bayelsa 11.4%, Rivers 10.0%; under
1% across the far north.

### Re-tested and still folded

- **Pentecostal** (above): the level is plausible and the place is not checkable. As a share of
  ALL Christians the two surveys do agree (+0.40, p 0.019), but that is mostly the Catholic pattern
  inverted. Within the Afrobarometer alone it runs 0.3-21.4% of Christians by round and fails its
  rounds 4-5 split-half (+0.137). Rounds 8-9 also carry a second box, "Pentecostal (e.g., Born
  Again ...)" (33 answers), beside the plain one; `key()` folds both to the same answer. The first reversal to try if someone wants it drawn
  flat inside the non-Catholic Christians at the GFS level.
- **Baptist, Methodist, ECWA, Aladura** (`Independent`): each under 3% of Christians in every round;
  `Evangelical` (where ECWA members might answer) is 1.8-3.1% in R4-R6 and near zero after.

### Routes checked this time, for the record

- **NDHS 2023-24 final report** FR395: open, read (above).
- **General Household Survey / NLSS**: NOT CHECKED this session. IPUMS `ng2010` (the 2010-11 GHS)
  stays blocked on the account; whether the World Bank Microdata Library's GHS-Panel or NLSS
  2018-19 files carry religion, and whether they need a login, is the next thing to open.
- **Pew 2010 sub-Saharan Africa survey** (*Tolerance and Tension*): NOT CHECKED this session,
  neither the report's appendix tables nor the terms of its data set.
- **Catholic diocesan statistics**: read, used as a rank witness (above).
