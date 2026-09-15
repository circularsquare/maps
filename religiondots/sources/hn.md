# Honduras — INE's ENDESA-MICS 2019, 18 departments

Built 2026-09-14 by session `f95259a4-hn`. `sources.md` §11ap is the scouting that found the
source and proved the LAPOP decode; §9dl is the write-up; this file is the working record.

    sources/hn_geo.py    COD-AB ADM1 (18 departments) + INE's 2024 projection via COD-PS + the join
    sources/hn_grid.py   Kontur 400 m hexes, the placement layer
    sources/hn.py        the weights, the checks, the construction, data/normalized/hn.csv
    taxonomy/hn2019.py   seven answers -> seven nodes, one new (`other.hn`); NO RESPONDE excluded

---

## 1. The source

INE's `bases-de-datos` page links the full ENDESA-MICS 2019 microdata as a plain zip,
`https://ine.gob.hn/wp-content/uploads/2025/02/BasesdatosENDESA2019.zip` (13,437,168 bytes,
seven SPSS files). The household file `hh.sav` carries `HC1`, *"Religión del jefe del hogar"*:
CATOLICA, EVANGELICA, TESTIGOS DE JEHOVA, MORMON, ADVENTISTA, OTRO(ESPECIFIQUE), NINGUNA
RELIGION, NO RESPONDE.

- **20,669 completed households** (`HH46 == 1`), exactly the ones with `HC1`; **80,439 people**,
  and `HH48` equals the member roster's line count in `hl.sav` for every household.
- **1,221 clusters** (`HH1`), none spanning two domains; SD.1 says 1,226 were sampled.
- `HH7` has 20 codes: departments 1-18 in INE's order, 19 San Pedro Sula and 20 Distrito
  Central, both urban-only and merged back into Cortés and Francisco Morazán.
- Fieldwork June to December 2019.
- No other file asks anyone's religion: `wm.sav` and `mn.sav` carry only "discriminated against
  for religion" and "asked a religious leader for help". So **the head-of-household ceiling
  cannot be priced from this survey**, and the Dominican measurement of it (sources/do.md §7.1,
  -12.7 points Catholic among women 15-19) is not transferred.

Census negatives, from §11ap and not re-derived: the 2013 census and the Censo 2026 form ask
none; ENDESA 2011-12 is DHS-account and national; ENDESA 2005-06 has no religion question.

## 2. The weights, rebuilt

None of the seven files carries a weight, PSU or stratum. `rebuild_weights()`:

1. **Transcribes Tabla SR.3.1** (printed p.70, PDF p.86 of the UNAH mirror) and **Tabla SD.1**
   (printed p.765, PDF p.781). All 20 unweighted domain counts and both area counts in SR.3.1
   equal the microdata; every SD.1 row and column adds to its printed total.
2. Splits each department's SR.3.1 weighted households between urban and rural in proportion
   to its **frame** enumeration areas, with one national factor for how many more households a
   rural area holds. The factor is fitted so the national urban share equals SR.3.1's 47.47%;
   it comes out **1.315**. The two cities take their own SR.3.1 totals out of their
   department's urban part.
3. Household weight = stratum total / completed households in the stratum; person weight =
   household weight x `HH48`.

**Witnesses the fit never saw**, households misplaced against SR.3.1's weighted column:

    household size           rebuilt  83   no weights   377
    sex of the head          rebuilt  29   no weights    36
    ethnicity of the head    rebuilt 227   no weights 1,944

**What the weights move**: at most 1.63 points on any department's Catholic share, 1.50 on
evangelical, 1.00 on no religion. Department totals come from the projection, not the survey.

**One known flaw.** `HH6` is not the design stratum. Counting clusters by department and `HH6`
does not reproduce SD.1's sampled urban/rural split (Cortés plus San Pedro Sula has 107 urban
and 41 rural clusters against SD.1's 130 and 21), so some areas sampled as urban are coded rural
at interview. The reconstruction treats `HH6` as the stratum. The witnesses say it works anyway.
The MICS copy with `hhweight` (World Bank `HND_2019_MICS_v01_M`, behind `mics.unicef.org`, the
account `ask/006-pa` already asks for) would replace all of this.

## 3. Geography and population

COD-AB `hnd_admin1` is the 18 departments. `HN%02d` of `HH7` equals the name join for all 18,
and both are asserted (Costa Rica's situation, not El Salvador's). The population is COD-PS
2024, whose source field is INE: **9,892,674**, equal to the adm0 row. There is no count since
2013 to check it against per department. Held-out, survey share of people against the
projection: **r = +0.9971**, 0 of 20,000 permutations (Gracias a Dios 0.87x, Valle 1.145x).

Kontur 2023 reads 1.066x the projection; 0.96% of its people fall outside every department
(coast and cays) and are dropped.

## 4. Which categories carry their own geography

Median Spearman over 400 random cluster halvings drawn inside each department, bar +0.475;
Sweden's chi-square on households; the largest single cluster's share of the answer:

    category              national   hh  median parity  pass%   chi2 p  top cl
    CATOLICA               41.67%  8,694  +0.907 +0.930  100.0  1e-233   0.00  own
    EVANGELICA             41.40%  8,252  +0.886 +0.763  100.0  1e-106   0.00  own
    NINGUNA RELIGION       14.82%  3,119  +0.841 +0.897  100.0   1e-56   0.00  own
    TESTIGOS DE JEHOVA      0.72%    143  +0.522 +0.679   63.5    2e-5   0.03  own, narrowly
    ADVENTISTA              0.67%    205  +0.358 +0.320   19.8  8e-122   0.04  standout
    MORMON                  0.47%     89  +0.263 +0.306    4.5    8e-5   0.03  flat
    OTRO(ESPECIFIQUE)       0.25%     76  +0.122 +0.284    0.5    4e-20  0.05  flat

(survey-weighted national shares; as drawn on the projection they move by a few tenths)

**The Witnesses pass on the rule as written.** Parity alone would have passed them more
comfortably (+0.679); the median is the statistic and it clears by 0.047.

### 4.1 The Adventist standout, and why it is not an override

No coarser published tier exists (ENDESA's domains are the departments and the two cities), so
`sources/se.py`'s two-level splice has nothing to splice. What exists is a different kind of
coarse partition: **one department against the rest.** Islas de la Bahía is the highest Adventist
department in **both halves of 400 of 400 halvings**: 75 households in 27 of its 42 clusters,
8.28% against 0.61% across the other seventeen. The rank test fails because the other seventeen
cannot be ordered at 205 households nationally, not because nothing is there.

The rule in `sources/hn.py` is general and asserted: a failing category that passes the
chi-square and cluster requirements keeps its measured share in a department that is its top in
both halves in at least 95% of halvings, and takes its share across the other departments
everywhere else. Only the Adventists meet it:

    MORMON               Islas de la Bahía   23.0%   flat
    ADVENTISTA           Islas de la Bahía  100.0%   kept there
    OTRO(ESPECIFIQUE)    Gracias a Dios      24.5%   flat (top PAIR with the Bay Islands: 84.5%)

It claims less than an OVERRIDE would: the other seventeen departments' Adventist shares, which
the survey cannot order, are not drawn. The 95% was set before the other two were looked at, and
it is the obvious level; `OTRO`'s pair at 84.5% is the case it keeps off the map.

### 4.2 The flat categories are at their national share, not in the residual

§9bi's construction was run first. Its reversal check (spec §12, Latvia) failed on this data:

    MORMON      drawn 0.15% to 3.85%; measured 0.00% to 1.26%; Bay Islands +2.59 pt
    ADVENTISTA  drawn 0.21% to 5.49%; measured 0.00% to 8.28%; Bay Islands -2.79 pt
    OTRO        drawn 0.08% to 2.07%; measured 0.00% to 2.29%; Gracias a Dios -1.79 pt

and it drew Latter-day Saints at 0.94% of Gracias a Dios, where the survey found none. The two
departments with a large tail have an Adventist tail and an `OTRO` tail, not a national mix. So
Mormon and `OTRO` take their national share everywhere and the four carried categories are scaled,
in their measured proportions, to fill what is left: by 0.991 (La Paz) to 1.027 (the Bay Islands).

## 5. The LAPOP cross-check

`data/raw/lapop/lapop_hn_municipio.feather` is a Honduras extract of the grand merge with `prov`,
`municipio` and their labels (17,213 respondents; `sources/lapop.py`'s slim file has no
`municipio`). `lapop_decode()` derives the 2012-2018 department of each of 18 `prov` codes from its
municipality names and COD-AB's 298 municipalities alone, and **the result equals §11ap's table**;
every 2023 municipality sits in `prov - 400`. The printed labels would misplace **59.6%** of the four
waves used (§11ap's 63.3% included 2016). Waves 2012, 2014, 2018, 2023, 6,225 respondents:

    Catholic                 r = +0.84      0 of 20,000   ENDESA 41.7%   LAPOP 41.9%
    non-Catholic Christian   r = +0.78      1 of 20,000   ENDESA 43.3%   LAPOP 43.9%
    no religion              r = +0.63     38 of 20,000   ENDESA 14.8%   LAPOP 12.2%

2010 (no `municipio`) and 2016 (answer codes shifted, §11ap) are left out and were not examined
further. LAPOP has no Adventist box, so it cannot witness §4.1.

## 6. What was drawn

9,892,674 people on 18 departments; 46,205 (0.47%) in `NO RESPONDE` are not drawn. Of the people
drawn: evangelical 41.87%, Catholic 41.12%, no religion 14.87%, Witnesses 0.74%, Adventist 0.67%,
Latter-day Saints 0.47%, other 0.25%. The survey's own weighted national figures have Catholics
ahead (41.67 against 41.40); laid on the 2024 projection the order flips. **Neither is ahead by
more than the design can see**, so `note_public` calls them level.

Catholic from 68.7% (Intibucá) and 61.4% (Lempira) to 28.3% (Cortés) and 16.7% (Islas de la
Bahía); evangelical 60.3% in Gracias a Dios, 52.2% Cortés, 51.4% Atlántida; no religion 24.5% in
the Bay Islands, 2.6% in Gracias a Dios.

## 7. Open

- **The head of household is the ceiling and it is unpriced here.** Nothing in the seven files asks
  a person their own religion.
- **`OTRO` in Gracias a Dios is 17 households, all with a Misquito head**, and the card has no
  Moravian or Protestant box. The specify text is not public. If INE or UNICEF's copy has it, it
  would say whether this is the Moravian Church.
- **The MICS copy with real weights** would replace §2; it needs the account in `ask/006-pa`.
- **ENDESA 2011-12's DHS recode** (women and men, `v130` by department) would give a second
  department-level round and a real split across rounds; DHS account.

## 8. Review, 2026-09-14 (session `f95259a4-hnrev`)

Re-run from `hh.sav` and the LAPOP extract, not from §2-§5. Nothing rebuilt, no output changed.

- **The witnesses test SR.3.1's printed domain totals, not the rebuilt urban/rural split.** Weights
  from the domain totals alone (no SD.1, no factor) misplace size 67, sex 85, ethnicity 227,
  against the rebuilt 83, 29, 227. The ethnicity row's whole gain over no weights is the printed
  totals. Only the sex row favours the split, on small numbers, and it swings with the factor
  (k = 1.0 gives 220, 1.5 gives 146).
- **No department share hinges on the weights.** Across no weights, domain totals only and k from
  0.8 to 1.7, the largest move against the drawn weights is 2.3 points (Choluteca Catholic at
  k = 0.8). The Bay Islands' Adventist share stays 8.24-8.30%. No department moves more than one
  rank on Catholic or no religion, or two on evangelical; the Witnesses up to three. The national
  Catholic/evangelical order does flip between schemes (42.7/40.6 unweighted, 41.4/41.8 at
  k = 1.0), which the note's "almost exactly level" already covers. Rest-of-Francisco-Morazán's
  urban part goes negative above k of about 1.9.
- **The standout rule holds against a null.** With departments shuffled across clusters (200
  shuffles, 100 halvings each) no small category reached 95% in any shuffle; the highest was 72%
  (Mormon), 95th percentiles 29-44%. Runner-up Adventist department: Atlántida 1.69% against
  8.28%; the 75 households are in 27 clusters, the largest holding 8. The rule was written after
  the Adventists were seen at 400 of 400, but any cut from about 30% up gives the same answer
  here, so the level decided nothing.
- **It would not have changed the Dominican Republic**: on `sources/do.py`'s frame its one
  uncarried category, the Witnesses, tops both halves in 1.8% of 400 halvings. `sources/bo.py`
  adopted the rule the same day and it selected nothing. Wave-halved countries not checked.
- **Flat over §9bi is right here.** §4.2's residual figures reproduce (Mormon 0.93% of Gracias a
  Dios, 3.85% of the Bay Islands). Taking the Adventist standout out of the tail first is worse
  (Mormon 1.49% of Gracias a Dios). Note that `sources/bo.py` pairs the same standout test with
  §9bi's residual, and spec §12 only says the flat construction "can" be used.
- **Head of household is applied as in `do.py`**: every roster member at the household weight.
  Here that is `HH48` times the weight, and `HH48` equals `hl.sav`'s line count everywhere. No
  file's labels carry a person's own religion or a weight.
- **`lapop_decode()` is not independent of §11ap's table.** Both read the same `municipio` labels
  the same way, so it proves the transcription, not the premise. What pins the decode without
  names or religion is §11ap's population witness, which `hn.py` does not run: LAPOP sample share
  against the 2024 projection, r = +0.939 decoded (0 of 20,000) against +0.300 on the printed
  labels; +0.87 to +0.91 per early wave. Without it, ENDESA-LAPOP agreement is cited as proof of
  both the decode and ENDESA. 174 early-wave respondents under two codes (Santa Rosa de Copán
  401405 under 409, Intibucá 400907 under 410) have a `municipio` prefix that is not their
  `prov`; the names agree with `prov`, so they are placed right.
- `note_public` figures match `hn.csv`; no edit. No screenshot, the Kontur fix was running.

## 9. Two additions, 2026-09-14 (session `f95259a4-house`)

**The population witness from §8 is now asserted on every build.** `population_witness()` in
`sources/hn.py` compares LAPOP's weighted share of respondents per department (`weight1500`, all
6,451 respondents of the four waves) with INE's 2024 projection. It reads no municipality name and
no answer, so unlike `lapop_decode()`'s check against §11ap's table it does not re-read what §11ap
read. The build stops if any of 20,000 random pairings reaches the decoded r, or if the printed
labels do as well as the decode, pooled or in any 2012-2018 wave. It runs whenever the LAPOP extract
is on disk, as the decode itself does.

    pooled   decoded +0.939 (0 of 20,000)   printed labels +0.300
    2012             +0.872                                +0.043
    2014             +0.910                                +0.051
    2018             +0.905                                +0.095
    2023             +0.977                                +0.977   the labels are right in 2023

No count changed: `hn.csv` was rebuilt and is byte-identical to the file before the edit (sha256
`c1506862fbfd...`).

**The flat construction complies with spec §12's small-category rule, written the same day.** With
the Adventist standout taken out first, §9bi's residual would draw Latter-day Saints at 2.96x their
national share in Gracias a Dios (1.40% against 0.47%), where the survey found none; the rule's bar
is 2x, so flat stays. §4.2's version of the residual, with the Adventists still in the tail, gives
1.97x there and would pass. The rule tests the residual after the standouts, which is the one this
build would have shipped.
