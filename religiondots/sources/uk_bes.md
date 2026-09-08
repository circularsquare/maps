# British Election Study — how many English people belong to each denomination

`sources/uk_bes.py` → `data/normalized/uk_bes.csv` (9 rows).

The anchor half of England's denominational split. `sources/uk_churches.md` says where the
denominations are; this says how big each one is. Neither can do the other's job, and the
reason is the gap between attendance and identity.

---

## 1. Why a survey at all, and why this one

Church attendance is not identity. Only about **12% of England's Christians were in church on
any given Sunday in 2005**, and the ones who were are not a random twelfth — Catholics attend
far more per head than Anglicans. The two sources disagree violently about the same country:

| leg | church census (attendance) | BES (identity) |
|---|---:|---:|
| anglican | 31.3% | **64.6%** |
| catholic | 34.1% | 18.2% |

Reading denominational shares off attendance draws an England with more practising Catholics
than Anglicans, which is true of churchgoing and false of the population. Spec §3.1 forbids
adding a `roll` to a `self_id`; what it permits, and what this is, is using one basis to
**split** another.

**Why the BES and not a better survey.** Understanding Society and British Social Attitudes
both carry a deeper denominational list, and both sit behind a UK Data Service registration.
The BES panel needs a **free account only**, and at n=126,840 across 31 waves it is far larger
than either — **25,630 English respondents in wave 21 alone**. Its religion question is a
profile variable asked every wave, which is why the sample is that size.

> `p_religion`: *"Do you regard yourself as belonging to any particular religion, and if so,
> to which of these do you belong?"*

**Wave 21, fielded May 2021**, is the wave closest to census day (21 March 2021). The waves are
deliberately **not pooled**: the same people answer every wave, so pooling would count
individuals repeatedly and narrow the confidence interval on a sample that had not grown.
Waves 22, 23 and 31 are read anyway, by `check()`, only to show the shares are stable.

---

## 2. The eleven Christian options

Deeper than the census's for England, which has none, and deeper than Scotland's, which names
two:

> Church of England/Anglican/Episcopal · Roman Catholic · Presbyterian/Church of Scotland ·
> Methodist · Baptist · United Reformed Church · Free Presbyterian · Brethren · Orthodox
> Christian · Pentecostal (*e.g. Assemblies of God, Elim, New Testament Church of God,
> Redeemed Christian Church of God*) · Evangelical – independent/non-denominational
> (*e.g. FIEC, Pioneer, Vineyard, Newfrontiers*)

All nineteen options are present in every wave from 17 on — checked, so nothing here depends
on an option appearing mid-panel.

The last two matter for the join. BES names the same bodies the English Church Census calls
**Pentecostal** and **New churches**, with examples on both sides, which makes those the
cleanest category matches in the whole construction.

**England, wave 21, share within Christians:**

| leg | respondents | share |
|---|---:|---:|
| anglican | 6,970 | 64.56% |
| catholic | 1,776 | 18.19% |
| methodist | 515 | 4.65% |
| newchurch | 279 | 3.11% |
| pentecostal | 156 | 2.54% |
| baptist | 234 | 2.43% |
| reformed | 227 | 2.17% |
| orthodox | 162 | 2.17% |
| other | **16** | 0.19% |

---

## 3. It disagrees with the census about how many Christians there are, by thirteen points

BES puts Christians at **36.1% of English adults**; the census puts them at **49.2%** of the
same population (21,994,389 of 44,715,447). Two structural reasons: an online panel people opt
into skews younger and more secular, and **there is no "Christian, no denomination" option on
this list**, so the many people who would answer a census with a bare "Christian" must pick a
denomination, pick "Other", or say none.

`uk_split.py` reads only the **ratio between** the Christian legs and takes every magnitude
from the census. The absolute Christian share emitted here is a survey artefact and is never
drawn.

### The largest assumption in England's split lives in that gap

BES classifies **73.4% of the census's adult Christians**. Applying its ratios to the whole
Christian total assumes the other **26.6% — about 5.8 million adults** — divide the same way as
the people who did pick a denomination. Untestable from this survey, and two plausible stories
run in opposite directions: someone who says only "Christian" may be a loosely-attached
Anglican, which would push the truth further towards `anglican`; or may be deliberately
non-denominational, which would push it towards `newchurch`. **Stated, not corrected.**

---

## 4. The Scotland check, which runs on every build

Scotland is a free test, because there a census publishes exactly the quantity this survey is
being asked for. NRS Census 2022 (UV205), share within Christians:

| | national church | Catholic | other Christian |
|---|---:|---:|---:|
| **NRS 2022 — the truth** | **52.5%** | **34.3%** | **13.2%** |
| BES W21 (n=1,034) | 49.3% | 28.4% | 22.2% |
| BES W22 (n=925) | 51.1% | 25.1% | 23.9% |
| BES W23 (n=1,066) | 53.2% | 24.1% | 22.7% |
| BES W31 (n=1,087) | 50.4% | 28.1% | 21.5% |

**The national-church leg lands within sampling error in all four waves.** Catholicism comes
out 6–10 points short and "other Christian" 9–11 points long, consistently and in one
direction.

Most of that is the child assumption below — Catholics are much younger than the national
church in both countries, so an adults-only survey finds proportionally fewer of them than an
all-ages census. The rest is a long option list attracting people whom a two-option census form
would have collected under its residual.

**A wave that matched NRS exactly, or missed it the other way, would be the thing worth
investigating.**

---

## 5. The child assumption, and why it is cheap here

BES surveys adults; the census counts everyone; `uk_split.py` applies these adult shares to all
26,167,899 of England's Christians. So **15.9% of the drawn people are children whose
denomination nobody measured.**

Anita's call, 2026-09-07, over the alternative of leaving 4.17M children on a bare
`christianity` node beside their own denominations: *"id rather not have a bare christianity
group in england, i feel like that'd be confusing."*

Unlike `us_rebase.py`, where the same assumption scales a survey by 1.28× and is called
load-bearing, here it is small and **measurable**. English Christians aged 30–49 — the people
whose children these are — split 55.7 Anglican / 23.8 Catholic against the all-adult 64.6 /
18.2. Giving every child their parents' generation's split instead would move the England-wide
anchor by:

    anglican −1.42   catholic +0.90   orthodox +0.37   pentecostal +0.20
    newchurch +0.19  methodist −0.10  reformed −0.10   baptist −0.05

**At most 1.4 points, on the largest leg.**

---

## 6. `other` is understated and is drawn anyway

Sixteen respondents, all of them Brethren, put the leg at **0.19% of England's Christians** —
about 42,000 people — while the matching churches in the English Church Census are **7.2% of
English churchgoing, 2,075 congregations**. The Salvation Army alone claims around 30,000 UK
members and the Adventists about 35,000, so the true figure is some multiple of this one.

The cause is the flip side of §2's long list: BES has no tick box for the Salvation Army,
Quakers, Adventists or Lutherans, so those people land in its generic "Other" (2.40% of adults)
pooled with non-Christians and unseparable.

Anita's call, 2026-09-07, over dropping the leg or inferring a size from the gap between BES's
"Other" and the census's non-Christian "Other religion": **use the residual category the map
already has.** `christianity.other` is what Scotland's *Other Christian*, Northern Ireland's
*Other Christian denominations*, Canada's and Australia's all resolve to, so England gains a
category its neighbours already display rather than a bespoke one, and the number stays
traceable to a source instead of being estimated into existence.

The undercount is real, one-directional, and belongs in `note_public`.

*(As of 2026-09-07 `other` is inside the unplaced 8% anyway, because `uk_churches.py` cannot
locate those bodies either. It becomes a drawn leg when the census-proxy placement lands.)*

---

## 7. Re-fetch

**Needs a free account** — email registration, nothing institutional. Anita downloaded it,
2026-09-07.

- <https://www.britishelectionstudy.com/data-objects/panel-study-data/> → *British Election
  Study Combined Wave 1–31 Internet Panel*, February 2014 – June 2026, N = 126,709.
- SPSS (225 MB zipped, **2.09 GB unzipped**) or Stata (324 MB). The SPSS file is what
  `uk_bes.py` reads, from `data/raw/uk/BES2024_W31_Panel_v31.05.sav`.
- Also at the UK Data Service as SN 8202, which needs a different registration.

**`pyreadstat` is required and does not build from source on Windows** — the current release
wants `iconv.h`. `pip install --only-binary=:all: "pyreadstat<1.3"` gets a working wheel.

The file has **13,381 columns**. Read it with `usecols`; a naive `read_sav` will take the whole
2 GB into memory for the sake of five columns.

---

## 8. Licence

**BES data is subject to the BES terms and conditions of data use.** Academic and
non-commercial use with citation; the terms are linked from the download page and should be
read before anything drawn from this reaches a print. Cite as:

> Fieldhouse, E., J. Green, G. Evans, J. Mellon, C. Prosser, J. Bailey, R. de Geus,
> H. Schmitt, C. van der Eijk, J. Griffiths & S. Perrett (2026) *British Election Study
> Combined Wave 1–31 Internet Panel*. DOI: 10.5255/UKDA-SN-8202-4
