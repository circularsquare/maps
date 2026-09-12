# Thailand — 2010 census, nine categories at region met with 76 provinces

Drawn 2026-09-07. `sources/th.py` rebuilds `data/normalized/th.csv`; `allocate.py` turns it
into `th_province_allocated.csv`. Boundaries and placement: `th_geo.py`, `th_grid.py`.
Taxonomy: `taxonomy/th2010.py`, `taxonomy/th_parents.csv`. `data/` is gitignored, so this
file is the record.

| | |
|---|---|
| source | *สำมะโนประชากรและเคหะ พ.ศ. 2553* — 2010 Population and Housing Census, NSO |
| categories | **9** — Buddhist, Muslim, Christian, Hindu, Confucian, Sikh, other, none, unknown |
| geography | **province (จังหวัด), 76 drawn units**, mean 868,000 people |
| basis | `self_id` |
| tier | **98.51% `measured`, 1.49% `derived`** (spec §7) |
| licence | Thai government work; OCHA COD-AB is CC BY-IGO, geoBoundaries CC BY |
| drawn | **65,977,881 of 65,981,658 — 99.99%** |

## 1. The office moved hosts, and that is the whole find

`sources.md` had Thailand as **"data existed and the server is gone"**, with a fallback plan
of breaking a substitution-cipher Thai font in 105 archived PDFs for a day to get the 2000
census. That was right about the hosts it checked and wrong about the office:

- `statbbi.nso.go.th`, `web.nso.go.th`, `service.nso.go.th`, `popcensus.nso.go.th` — **all
  dead**, no DNS or refused connections. Confirmed again 2026-09-07.
- **`www.nso.go.th` is alive.** It answers **418** to a bare `curl` and **200** to a request
  with a browser `User-Agent`, which is why a status check reads as a dead host.
- Its `/sites/2014/Documents/` tree now 404s — but **the Wayback Machine archived it
  wholesale** in 2018–2022, and that is where every census file here comes from.

**The generalising lesson is the one Nepal already taught (`§9ao`): a statistics office that
looks dead has usually moved.** What Thailand adds is that a *bot wall can look exactly like
a dead host* — 418 is not a status anybody expects — and that the CDX API is the way to find
the old tree once the new one has dropped it.

The live site's new catalogue is real and was checked: **`catalog.nso.go.th` is an open CKAN
with no key** and a bulk CSV companion at `catalogapi.nso.go.th/api/index?table=<T>&format=csv`.
It carries religion only as `OS_04_0001_01`, a **6-region, 3-religion** table from the Social,
Cultural and Mental Health Survey. Strictly worse than the census on both axes; not used.

## 2. No published table crosses religion with province, in either census

Checked rather than assumed, because the whole build depends on it:

| census | religion table | cut by | geography |
|---|---|---|---|
| 2010 (2553) | **Table 4** | sex, municipal/non-municipal | region, and the nation |
| 2000 (2543) | **Table 5** | age group, sex, area | **national only** |

**Table numbering is not stable between censuses** — 2000's Table 4 is marital status — so the
2000 file was found by scanning every sheet of every workbook for the religion words rather
than by trusting a number. Each of the five 2010 regional volumes carries its own Table 4 at
the same cut, and **none of that volume's other 21 tables crosses religion with changwat**;
that was checked one table at a time.

The province cut did exist, as one file per changwat on the dead hosts
(`..._C-pop_2553_000_<PROV>_00400.xls`). **The archive holds 2 of 77.**

## 3. So the two halves come from different documents — spec §3.10

| | geography | categories | file |
|---|---|---|---|
| fine geography | **76 provinces** | 3 | `2553/kpi_stat/<Province>_T.pdf` |
| fine categories | 5 regions | **9** | `2553/3/<region>/Table4.xls` |
| denominator | 76 provinces | — | `2553/3/<region>/Table1.xls` |

`allocate.py --hierarchy parent --parent-file taxonomy/th_parents.csv --within 1`.

**`--within` is not optional here.** Christianity is 3.05% of the North and 0.35% of the
Northeast; a pooled national share would take the hill churches of Chiang Mai and Mae Hong Son
and scatter them across Isan. §3.10c found this with India and Thailand is the second
customer.

**The result is a much better tier split than an allocation usually buys: 98.51% measured.**
Canada's is 71.3% derived. The reason is not that the method is better but that the two
categories NSO publishes per province are the two that hold 98.5% of Thailand.

### Table 1 gives the province→region map for free

Which volume a province appears in *is* its region, so nothing here asserts the assignment.
Table 1 also supplies exact province populations, which keeps the KPI sheets' rounding out of
the denominator.

## 4. What to distrust

**The percentages are one decimal place.** A province's Buddhist figure carries about ±0.05%
of its population — ±300 people in a 600,000-person province — and the residual, being
100 − Buddhist − Muslim, carries twice that. The counts are the product of a rounded share
and an exact total, which is arithmetic on two published figures rather than an estimate, so
the rows are `measured` — but the band is real and small categories live inside it.

**`a` is not a number.** NSO prints `a` for *"less than half the last digit shown"* and `na`
for not available, in the same column position a value would occupy. A parser that collects
"the first three numeric lines after the label" walks straight through them into the next
indicator: **Lampang's Muslim share came out as its 90.2% household-registration rate**, and
Buddhist + Muslim then exceeded the province population, which is the only reason it was
caught. `th.py` anchors on the English label line instead, which always carries the 2010 value
as its first token.

**Kanchanaburi has no sheet.** It is the one province of 76 with no captured `<Province>_T.pdf`
anywhere in the archive — checked against every capture of the `kpi_stat` directory, not just
the collapsed listing. Its Buddhist and Muslim shares are its region's, its rows say
`shares_from=region` in `note`, and `th.py` fails loudly if that set ever grows. 848,000
people, 1.3% of the country.

**Bangkok is one unit of 8.3 million**, the coarsest on this map after Île-de-France. Its own
Table 1 enumerates its fifty districts and religion is published for the city as a whole, so
a district geography would assert nothing the city total does not already say while looking
like a measurement. §8.2's placement grid spreads its dots by where people live, which is the
honest version of the same thing.

**Bueng Kan was created seven months after the census.** The 2010 tables have 76 changwat and
geoBoundaries has 77, so `th_geo.py` dissolves Bueng Kan back into Nong Khai — spec §8.1,
boundaries at the vintage the data was published on. NSO published a back-computed KPI sheet
for it, which `th.py` skips; drawing both would double-count.

**0.07% of Thailand answers `no religion`, the smallest share on this map**, and that is a
fact about the question. `ไม่ทราบ` (unknown, 0.013%) is excluded as non-response per §3.5.

## 5. The reconciliation, and it is the only real check this country has

The KPI sheets and Table 4 are different documents that never reference each other. Summing
each region's province residuals and comparing with that region's own Table 4 non-Buddhist
non-Muslim total is therefore a genuine test that they describe the same population:

| region | Table 4 residual | provinces summed | |
|---|---|---|---|
| Bangkok | 236,811 | 232,546 | −1.8% |
| Central | 204,850 | 203,238 | −0.8% |
| North | 373,413 | 388,276 | **+4.0%** |
| Northeast | 112,371 | 114,843 | +2.2% |
| South | 48,444 | 48,310 | −0.3% |

The North is the loosest and is where the residual is largest and most concentrated, so it is
also where the one-decimal rounding has the most room to accumulate. `th.py` prints this table
on every run and flags anything past 25%.

**Placement**: Kontur's 400 m grid, 419,176 hexes, reproducing the census at 1.085× nationally
with a per-province median of 0.90 and **not one of the 76 outside a factor of two** — against
**38 of 76** for a shuffled null, which is what says the join is real.
