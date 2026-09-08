# English Church Census 2005 — how big a congregation is, and nothing else

`sources/uk_ecc.py` → `data/normalized/uk_ecc.csv` (1,083 rows).

**`count` is a mean congregation size, not a number of people.** Every consumer multiplies it
by a church count from `sources/uk_churches.py`. A reader who sums this column has computed
nothing.

---

## 1. What it is

The fourth English Church Census, **Sunday 8 May 2005**, organised by Peter Brierley for
Christian Research: a postal census of every Trinitarian church in England. **37,051
approached, 18,633 returning usable data, a 50.02% response.** Deposited by David Voas at the
UK Data Archive as SN 6409 and redistributed free by the ARDA, which is the copy read here.

Comparable censuses ran in 1979, 1989 and 1998, and this file carries the same churches'
1989 and 1998 attendance (`TAGCAL89`, `TAGCAL98`) — unused, but there for a trajectory.

**Scope.** All Trinitarian churches: Free, Protestant, Anglican, Roman Catholic and Orthodox.
**Excluded by the census itself**: Jehovah's Witnesses, Mormons, Christian Scientists,
Christadelphians and Unitarians. Those people are inside the census tick-box `Christian` and
outside this structure, so they land on `uk_split.py`'s unplaced remainder.

---

## 2. What is used, and what was thrown away

Only **mean attendance per church**, by county and by county × settlement class.

The totals are not used and must not be. `sources/uk_churches.md` §1 has the full argument;
the short version is that the census took bulk data from ten Church of England and eight Roman
Catholic dioceses, so its county totals are partly a fact about English religion and partly a
fact about which bishop's office had a spreadsheet. Measured against the CofE's own register
the Anglican response rate runs **25.9% in Norfolk and 92.5% in Greater Manchester**. A mean
survives that; a total does not.

The per-denomination response rates the user guide publishes are kept in `RESPONSE`,
**unapplied**, because they document the bias rather than fix it:

> Baptist 67 · smaller denominations 57 · Anglican 55 · Catholic 54 · Independents 50 ·
> New Churches 49 · United Reformed 44 · Methodist 37 · Pentecostal 30 · **Orthodox 7**

Orthodoxy's 7% is not carelessness: **8 May 2005 was the Sunday after Orthodox Easter** and
many Orthodox churches were shut.

**Churches with no service that Sunday are excluded from the mean.** 537 of them. The user
guide explains why they are not zeroes: linked rural benefices held one service between three
churches, and all the people were recorded at the church where it happened. A zero there is a
fact about the rota, not about the congregation.

---

## 3. The geography, and it was abolished before the census was taken

`cntycde` is the **1974–1996 county set** with two later edits: London split Inner and Outer,
and Herefordshire under its post-1998 unitary name. So **Avon, Cleveland and `Hereford UA and
Worcester` are all present in 2005 data**, and the first two had been gone for nine years.
There is no postcode and no local authority, despite the UKDA metadata claiming Local
Authority Districts as the spatial unit.

`uk_split.py`'s `ECC_COUNTY` rebuilds all 47 from April 2023 districts. The unions are exact:

| ECC county | modern districts |
|---|---|
| Avon | Bristol · Bath and North East Somerset · North Somerset · South Gloucestershire |
| Cleveland | Hartlepool · Middlesbrough · Redcar and Cleveland · Stockton-on-Tees |
| Hereford UA and Worcester | Herefordshire + the six Worcestershire districts |
| East Yorkshire (combined) | East Riding of Yorkshire + Kingston upon Hull |

Humberside's south bank (North Lincolnshire, North East Lincolnshire) goes to Lincolnshire,
where the census puts it. **Rutland goes to Leicestershire**, which is where it was in 1974–96;
the census has no Rutland row. Somerset and Gloucestershire must exclude the Avon districts,
and Durham and North Yorkshire the Cleveland ones, or those four counties silently absorb
areas the census counted separately.

**Inner vs Outer London uses ONS's statistical definition** — 14 boroughs including the City,
Haringey and Newham. Brierley does not say which he used; the 12-borough reading would move
Haringey and Newham. It matters: Inner London is 913 of the census's churches.

---

## 4. The settlement axis, and why it is two classes and not eight

`envnmcde` classifies each church's address into eight settlement types, and the
denominational difference across them is nearly as large as the difference across counties —
Anglicans are 57% of rural attendance and 24% on council estates; Catholics run the opposite
way, 14% to 54%.

**Only two of those eight survive, and ONS forced it.** The 2021 Rural Urban Classification is
the only one keyed on `OA21CD`, and it has six categories that are three settlement sizes
crossed with proximity to a major town. ONS describes the change from 2011 as "a streamlining
of the taxonomy", and what it streamlined away is exactly the settlement detail this file would
have used. The 2011 RUC does carry major conurbation / minor conurbation / city and town, but
only for 2011 Output Areas, and joining it forward is `uk.md` §4's live trap.

What two classes still buy, in mean congregation size:

| leg | England | urban | rural |
|---|---:|---:|---:|
| anglican | 81 | 135 | 45 |
| catholic | 377 | 426 | 207 |
| methodist | 48 | 73 | 26 |
| baptist | 95 | 111 | 75 |

A rural Anglican congregation is a third the size of an urban one; a rural Catholic one is half.
That is real signal about where people are, not just where buildings are, and it is the reason
this source is still in the pipeline at all.

---

## 5. The trap inside the settlement code

**`envnmcde` is missing for 37% of churches and the missingness is denominational.** Coded:
anglican 68%, methodist 73%, baptist 71%, reformed 73% — but catholic 54%, newchurch 38%,
pentecostal 35%, **orthodox 6%**.

This no longer biases anything, because a mean is a within-leg quantity and the coding rate
cancels. It used to: reading a cell's *mix* off the coded churches made every English
conurbation about twelve points more Anglican and ten points less Catholic than the census
found it. That was a second, independent bug on top of the response-rate one, and it is worth
knowing the shape of it in case this file is ever read a different way.

What the coding gap still decides is **how many cells get their own mean** rather than falling
back to their county's or England's. `uk_split.py` prints the ladder on every run.

---

## 6. Re-fetch

Free, no registration, no bot protection. The ARDA hosts the UKDA deposit on OSF:

```
# Stata (used here), 6.3 MB
curl -sSL -o data/raw/uk/ecc05.dta https://osf.io/download/tbg3y
# SPSS equivalents, if ever needed
#   https://osf.io/download/r7wdn   (.sav)
#   https://osf.io/download/mvnt3   (.por)
# User guide, which is where the response rates and the bulk-diocese sentence live
curl -sSL -o data/raw/uk/ecc05_userguide.pdf \
  "https://www.thearda.com/ARDA/pdf/originalCodebooks/Engish%20Church%20Census%202005%20User%20Guide.pdf"
```

Note the ARDA's own typo in that filename — `Engish`, not `English`. The ARDA dataset page is
`thearda.com/data-archive?fid=ENGCC05`; it is server-rendered but the download hrefs point at
OSF, so the URLs above are stable and the page is not needed.

**pandas cannot `read_stata(convert_categoricals=True)` on this file.** `cmscomp` carries two
identical value labels and pandas raises rather than accepting a non-unique categorical.
`_read()` applies the three label sets it needs by hand.

---

## 7. Licence

**© D. Voas and Christian Research.** This is *not* an open licence — it is a UK Data Archive
deposit redistributed by the ARDA, and the UKDA's standard terms are research use with
citation, not free redistribution or commercial reuse. Cite as:

> Voas, D. and Brierley, P.W., *English Church Census, 2005* [computer file]. Colchester,
> Essex: UK Data Archive [distributor], March 2010. SN: 6409,
> http://dx.doi.org/10.5255/UKDA-SN-6409-1

**This is the one non-open source in England's split** and the constraint that matters if any
of this reaches a print — see `reference_poster_commercial_licences`. What is drawn from it is
a set of mean congregation sizes rather than any of its records, which is a weaker derivation
than redistribution, but the question should be asked before it ships commercially rather than
after.
