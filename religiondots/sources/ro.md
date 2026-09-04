# Romania — INS, Recensământul Populaţiei şi Locuinţelor 2021

Ingested and drawn 2026-09-03. Rebuild with `python sources/ro.py --fetch`.

23 religions at **UAT** level — municipiu, oraş, comună — 3,181 units for 19.05M people,
about 6,000 a unit. Fine geography, coarse categories, and the reason for the coarseness
is unlike anything else in the project.

---

## 1. The file

One XLSX, 463,240 bytes, no login, no bot protection, plain HTTPS that verifies.

```
https://www.recensamantromania.ro/wp-content/uploads/2023/06/Tabel-2.04.1-si-Tabel-2.04.2.xlsx
```

Two sheets:

| sheet | what | used as |
|---|---|---|
| `Tab 2.4.1` | religion × macroregion / development region / **judeţ** | `country`, `judet` |
| `Tab 2.4.2` | religion × **judeţ / municipiu / oraş / comună** | `uat` — the drawn level |

The table list is at
`https://www.recensamantromania.ro/rezultate-definitive-caracteristici-etno-culturale-demografice/`,
where the 28 files are named `Tabel-2.NN` with **no titles in the link text**, so finding
the religion one means opening them. 2.4 is religion by place; 2.5 is religion × ethnicity;
2.8, 2.11, 2.13, 2.16, 2.25 and 2.26 cross religion with other variables at national level.

## 2. The category list is a statute

**Romania publishes its list of state-recognised cults and nothing else.** There are 18
recognised cults plus the Metropolis of Bessarabia, and INS names almost exactly those,
sweeping everything else into `Alta religie (asociatii religioase sau grupari religioase)`
— 23,956 people.

This is a different granularity ceiling from every other source here. Australia's 148 and
Poland's 216 are limited by what respondents wrote and what the agency chose to tabulate;
Romania's 23 is limited by **which bodies the state has recognised**. A Romanian Mormon,
Buddhist or Hindu is in `Alta religie` no matter how many of them there are, because the
category does not exist to be ticked.

What the list does buy is three things almost nothing else has:

- **`Crestina de Rit Vechi`, 28,362** — the Lipovans, Russian Old Believers who settled
  the Danube delta after the Nikonian reforms of 1653-66. **The largest Old Believer
  population any census in the world publishes.** It is the second user of
  `christianity.orthodox.oldbeliever`, which Poland introduced the same day.
- **`Unitariana (Biserica Unitariana Maghiara)`, 47,992** — the Hungarian Unitarian Church
  of Transylvania, continuous since the Edict of Torda in 1568 and the oldest Unitarian
  body anywhere.
- **Two Lutheran churches kept apart** — `Evanghelica Lutherana` (Slovak and Hungarian)
  and `Evanghelica de Confesiune Augustana` (Transylvanian Saxon). Same confession, two
  churches, separated by ethnicity rather than doctrine. `branches.py` has one Lutheran
  node, so the map loses this; recorded rather than solved.

Full mapping and the arguable calls are in `taxonomy/ro2021.py`.

## 3. `*` is a suppression marker inside a numeric column

INS writes `*` for a confidential cell and `-` for a true zero, **in the same columns as
the counts**. This is New Zealand's `-999` problem in a different costume (spec §3.2), and
it is worse in one way: `*` is a string, so a naive `pd.read_csv`-style read makes the
column `object`, and `pd.to_numeric(errors="coerce")` turns every suppressed cell into NaN
that sums to nothing. The suppressed people disappear and no error is raised.

`ro.py` reads the cells one at a time and classifies each as a number, `*` or `-`, and
raises on anything else so that a new sentinel cannot slip in unnoticed:

```
cells: 23,357 numeric, 10,224 suppressed (*), 40,571 true zero (-)
```

Suppressed rows are **dropped, not estimated**. The cost is measurable exactly, because
the totals are not suppressed:

| level | categories sum to | of total | shortfall |
|---|---|---|---|
| uat | 19,037,322 | 19,053,815 | 16,493 (0.087%) |
| judet | 19,053,559 | 19,053,815 | 256 |
| country | 19,053,815 | 19,053,815 | 0 |

So 16,493 people are in some category somewhere and in no row of the drawn level. That is
the entire cost of suppression and it is under a tenth of a percent.

## 4. County headers are indistinguishable from communes, and the obvious rule is wrong

On `Tab 2.4.2`, municipii and oraşe carry a type prefix (`MUNICIPIUL ALBA IULIA`,
`ORAȘ ABRUD`) but **communes and county headers are both bare all-caps names**. Nothing in
the row says which it is.

The obvious rule — "the first row bearing a county's name is that county's header" — is
**wrong**, because two county names are also commune names elsewhere, and the communes
sort first:

```
CĂLĂRAȘI     3,285 (a commune) ... 283,458 (THE COUNTY) ... 1,883 ... 5,195
SATU MARE    1,995 (a commune) ... 330,668 (THE COUNTY) ... 3,232
```

Taking the commune as the header shifts every following county boundary. It cost 6
misfiled rows and 600,861 double-counted people, and the only reason it was caught is that
two different counties' `Păuleşti` happened to collide into one key and tripped the
"categories exceed the unit total" check. Had they not collided, the run would have passed.

**The rule that works: a row is a county header when its name is a county's AND its total
equals that county's total on sheet 2.4.1.** Bucharest still needs a seen-once guard,
because it appears on two consecutive rows with identical numbers — once as its own county
and once as the single UAT that fills it.

## 5. Reconciliation

```
uat       3,181 units    19,053,815   (expected 3,181 units)
judet        42 units    19,053,815
country       1 unit     19,053,815
```

Exact at all three levels against INS's published 19,053,815. 3,181 UATs is also exactly
the Romanian LAU count in GISCO, which is the first sign the geography will join.

## 6. The dominant caveat: 14% have no religion recorded

`Informatie nedisponibila` is **2,658,165 people, 13.95%**.

This is **not** a refusal like Poland's 20.5%. Romania conducted RPL 2021 substantially
from administrative registers, and religion is in none of them, so for one person in seven
the variable is simply absent. It is excluded from the dots, leaving 16.4M of 19.1M drawn.

The difference matters for how it should be read: Poland's missing fifth chose not to
answer, which is itself information about them. Romania's missing seventh were never asked.

## 7. What is drawn

23 categories → 19 taxonomy nodes → **16,369 dots and 1 ring** at 1:1,000.

Romania is 85.7% Orthodox among those with a religion recorded, so the map's interest is
the Hungarian belt of Transylvania — Reformed, Roman Catholic and Unitarian — the
Greek Catholics of the north-west, the Pentecostals of the Banat (403,672, an unusually
large Pentecostal population for Europe), the Lipovans in Tulcea, and the Muslim Tatars
and Turks of Dobruja.

One new node was added for Romania: `other.ro`, for `Alta religie`.

## 8. Not done

- **`Ateu` and `Agnostic` are collapsed.** INS asks `Fara religie`, `Ateu` and `Agnostic`
  as three separate answers — 71,430 / 57,229 / 25,485 — and `ro2021.py` sends the last
  two both to `secular`. The distinction survives in `source_category` if it is ever
  wanted.
- **The two Lutheran churches are one node.** See §2.
- Sheet 2.5 crosses religion with ethnicity at national level. Transylvania's religious
  map is largely an ethnic one, and that table would say so, but it has no geography.
