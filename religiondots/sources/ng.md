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
`ask/002-ng` puts the §14 question about publishing any of this to Anita; the decision taken
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

- **Every Christian denomination.** The card names about twenty and Nigerians fill nearly all
  of them, Roman Catholic at 8.6% of respondents and Pentecostal at 4.8%. The share answering
  `Christian only` rather than naming one runs 19.4% to 47.3% between rounds with no trend, so
  a pooled denominational share measures the fieldwork (§11ai).
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
