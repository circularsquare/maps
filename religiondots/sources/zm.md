# Zambia — the 2022 census religion volume, at 156 constituencies

**Drawn 2026-09-14.** 156 constituencies, 34 source categories on 29 nodes, 18,340,343 people
(the de facto count), every row `derived`.

- `sources/zm_geo.py` -> `data/geo/zm/zm_constituencies.gpkg`, `zm_lookup.csv` (COD-AB admin 3)
- `sources/zm_grid.py` -> `data/geo/zm/zm_hexes.gpkg` (Kontur 400 m, keyed to constituency)
- `sources/zm.py` -> `data/normalized/zm.csv` (two ZamStats PDFs in `data/raw/zm/`)
- `taxonomy/zm2022.py` -> the mapping; `countries.py` `"zm"` -> the wiring; new node `other.zm`
- sources.md **§9db** is the write-up; spec §12 carries the header-relabel trap.

```
python sources/zm_geo.py  --fetch
python sources/zm.py      --fetch      # needs zm_geo.py's lookup for the join
python sources/zm_grid.py --fetch      # prints a Kontur shape check once zm.csv exists
```

## 1. What ZamStats publishes

| release | religion | tier |
|---|---|---|
| **2022 CPH, Series B: Religion Descriptive Tables** (April 2026, 39 pp.) | B.1 ten religions; B.6/B.9/B.10 23 named churches + None + Other, rural and urban; B.4/B.5 Christianity vs everything else, by sex and by rural/urban | **province** for the categories; **province, district, constituency** for the two-way split |
| 2022 CPH, *National Analytical Report* (Aug 2025), §3.4 | five categories in charts (Christianity, Islam, African Traditional, Other, None); denominations by residence and sex | national, province for Christian vs rest |
| 2022 CPH, *Summary Report Part 2* (Sept 2024) | none; Table 5.2 is de jure population to ward | used for the gap and a join check |
| 2010 CPH, provincial *Descriptive Tables*, Series A-D (2013) | Table B5: religion x 5-year age x sex x **district**, eight categories (Catholic, Protestant, Muslim, Hindu, Buddhist, Bahai, Other, None) | district (72 then) |
| UNSD table 28 | 2010 national, eight categories | national |
| 2010 census microdata (IHSN catalog 4124) | religion presumably present | gated: written authorisation from the Director |

**The two earlier negatives were each about one release.** §11p read the 2022 *Analytical
Report*'s chart and wrote "5 categories, Christianity undivided at 98%, do not build"; §11w
corrected it from the oracle's 2010 row (eight categories). The Series B volume came out in
April 2026, after both, and it has 25 Christian rows.

## 2. Why 2022 at constituency, and not 2010 at district

2010 is the only release with **the Catholic/Protestant split measured below the province**
(72 districts). 2022 has **23 named churches at province** and the Christian/non-Christian
split measured at 156 constituencies, rural and urban. The churches are the country's content
(Adventists 42.1% of Southern, New Apostolic 38.0% of Western, the Reformed Church 68.5% in
Eastern, CMML 46.2% in Luapula, ECZ 74.3% in North-Western and Western), and none of them is
visible in 2010's `Protestant` 75.3%. So 2022 is drawn.

The 2010 district volumes are the obvious **witness** for how much within-province variation
the fill misses, at least for Catholics. Not built: only some of the ten provincial A-D
volumes are linked from the ZamStats census page (Central, Lusaka, North-Western, Northern,
Southern, Western), the district set changed from 72 to 116 in between, and it would not move
a dot. The Lusaka volume was read (Table B5, pp. 120-138) and is the source of the 2010 sex
ratios quoted below.

## 3. The allocation

`--within` by province AND residence (spec §3.10c). For constituency *c* in province *p*:

    count(c, church d)    = chr_rural(c) * B9(p, d) / B9(p, Total) + chr_urban(c) * B10(p, d) / B10(p, Total)
    count(c, religion k)  = nonchristian(c) * B1(p, k) / sum over the nine non-Christian columns of B1(p)

Every province x category re-aggregates to B.6 and B.1 within 0.003 people. B.1 has no
residence split for the religions, so those use the province mix alone.

**Every drawn row is `derived`**, and the Christian rows carry `parent_column=Christianity`,
which `COLUMNS` maps to `christianity`. The non-Christian parent is every non-Christian at once
and has no node, so it has no roll and those dots go under `inferred dots: not shown`
(ao2024.py's reasoning).

## 4. Two of B.1's column headers are wrong

Table B.1 heads its columns Christianity, Islam, **Judaism**, Hinduism, Buddhism, Bahai Faith,
Sikhism, **African Traditional Religion**, **Non-Religious**, **Other Religious Groups**. The
figures under those last and third are 233,260 and 30,502; under ATR and Non-Religious, 463 and
9,238.

The National Analytical Report (§3.4, Figure 3.13, and its key findings) gives **Christianity
98.0, Islam 0.5, African Traditional 0.2, Other 0.1, None 1.3** and says *"1.3 percent reported
no religious affiliation"*.

| reading of B.1 | Christ. | Islam | ATR | Other | None |
|---|---:|---:|---:|---:|---:|
| the report | 98.0 | 0.5 | 0.2 | 0.1 | 1.3 |
| B.1 with `Judaism`=ATR, `Other Religious Groups`=None, the six small = Other | 98.0 | 0.5 | 0.2 | 0.1 | 1.3 |
| B.1 as printed | 98.0 | 0.5 | 0.0 | 1.5 | 0.1 |

Independent of the arithmetic:

- Figure 3.14 puts the report's `None` at **1.8% of men and 0.8% of women**. In the 2010 census,
  Lusaka Province's `None` was 26,603 men to 8,676 women and its `Other` 23,182 to 24,077. The
  233,260 column has the no-religion profile.
- Figure 3.15 puts `African Traditional` at 0.3% rural and 0.1% urban, and the `Judaism` column
  is **15,625 of 30,502 in Eastern Province**, 1,037 on the Copperbelt.

So the mapping files `Judaism` -> `indigenous.african`, `Other Religious Groups (B.1 column)`
-> `unaffiliated`, and the printed `African Traditional Religion` and `Non-Religious` ->
`other.zm` (both are inside the report's `Other` under its reading; if the mislabel is a
symmetric swap the 463 is the Judaism column, and nothing says so). `check_relabel()` asserts
the report's five figures come out under this reading and do not under the printed one.

**What was not done**: filing them as printed. That would draw 30,502 Jews concentrated in
rural Eastern Province and 463 traditionalists, against the office's own prose.

**Also recorded, not acted on**: the 335,271 Christians who named no denomination (B.6 `None`)
are **68.4% men** against 48.3% for Christians overall (B.12), and rank the provinces like the
no-religion column does (North-Western first on both, Luapula second). Some of them are very
likely people with no religion recorded as Christian. Filed at `christianity` as printed (§2.7).

## 5. The join, and three naming traps

On district + constituency name, since district names are unique in both files (asserted).
Five constituencies by an explicit alias: `Mwembeshi`/`Mwembezhi` and `Petauke
Central`/`Petauke` (spellings), and **Solwezi West, Solwezi East and Solwezi Central**, which
COD-AB names for their districts Kalumbila, Mushindamo and Solwezi; asserted to be the sole
constituency of their district in both files.

- **Chirundu is in Southern Province in the census and Lusaka in COD-AB.** COD-AB's boundaries
  were created 2020-11; the census uses the December 2021 revision. The polygon is the same
  district; the allocation uses the census's province.
- The religion volume spells one district three ways (B.4 `Ikelenge`, B.5 `Ikeleng'i`, COD-AB
  `Ikeleng'i`) and B.5 writes `Northwestern` where B.1 writes `North Western`.
- Summary Report Table 5.2 switches label style at Luapula, from `CENTRAL PROVINCE` / `CHIBOMBO
  DISTRICT` / indented name to `PROVINCE LUAPULA` / `DISTRICT KAWAMBWA` / `Constituency Liuwa`,
  and writes `Mpongwe Central` for B.5's `Mpongwe`.

## 6. Suppressed cells, all recoverable

B.5 prints `*` in 15 cells, always non-Christians in the urban part of a small-town constituency
(Chilubi, Zambezi West, Gwembe, Luampa, Mwandi, Nkeyema, Shangombo, Sikongo, and their district
rows). The same row prints the urban total and urban Christians, and all non-Christians and the
rural ones, so each cell is recovered two ways and they agree. 70 people in total.

## 7. The universe: de facto, 6.9% below the census count

Every religion table is de facto, **18,340,343**. The census total is de jure, **19,693,423**
(Summary Report 2.1), which adds absent usual members, 47,941 in institutions and 4,063 homeless,
and drops visitors. The 1,353,080 difference is not broken down anywhere. Per constituency the
ratio runs **0.812 (Lupososhi) to 0.989 (Vubwi)**, median 0.941, with Mandevu in Lusaka at 0.823.
`gap` states it and `gap_share` is 0.0687, hand-written because the hole is not a column any
table printed.

## 8. Placement

Kontur 2023-11, 245,547 hexes, 1.045x the de jure count. Per constituency against the religion
tables the ratio runs 0.41x (Milenge) to 2.66x (Mansa Central). The high tail is dense Lusaka and
Copperbelt wards, where Kontur runs high everywhere, plus a **Luapula cluster** (Mansa Central
2.66, Chembe 2.37, Bahati 2.10 against Milenge 0.41, Mambilima 0.57), which suggests COD-AB's
constituency lines around Mansa and Mwense do not match the census's. It moves no count, only
where inside a polygon the dots go.

## 9. What would improve it

- **The 2010 district volumes as a witness** for the Catholic geography inside provinces (§2).
- **The 2022 microdata**, which would put the churches on the constituency directly. ZamStats
  gates microdata behind the Director's authorisation (IHSN 4124 for 2010); not requested.
- A re-issue of Series B with corrected B.1 headers breaks `check_relabel()` on purpose.

## 10. Review, 2026-09-14 (f95259a4-zmrev): the relabel holds, on evidence the build does not read

Tested §4 against three documents: the 2022 household questionnaire, UNSD table 28's 2010 row,
and the analytical report's charts read as images. The relabel stands. Two things to add.

**B.1's header is the questionnaire's code list, word for word.** P21 on the household form
(`wp-content/uploads/2023/12/2022-Census-Population-Housing-Qre_FV_5June2022-1.pdf`, page 4) codes
religion 1 Christianity, 2 Islam, 3 Judaism, 4 Hinduism, 5 Buddhism, 6 Bahai Faith, 7 Sikhism,
8 African Traditional Religion, 9 Non-Religious, 10 Other Religious Groups (specify), in B.1's
order. So B.1 is not a mistyped header row. Its columns are codes 1 to 10 under the paper form's
labels, and the question is which answer each code actually holds. The Figure 3.13 arithmetic
cannot settle that alone: 0.1% is anything from 9,170 to 27,510 people, so `Non-Religious`
(9,238) fits in the report's None or its Other and all five figures still come out. The response
profiles settle it:

| | people | men | rural / urban |
|---|---:|---:|---|
| 2010 `None` (UNSD table 28) | 224,295, 1.79% | 65.1% | 2.14% / 1.27% |
| 2010 `Other` | 253,621, 2.02% | 48.7% | 1.87% / 2.25% |
| 2022 code 10, B.1's `Other Religious Groups` | 233,260, 1.27% | about 68% (Fig. 3.14: 1.8% of men, 0.8% of women) | 1.5% / 0.9% (Fig. 3.15) |

Code 10 has 2010's no-religion profile on sex and on residence, and not 2010's `Other`. The sex
half does not need the report: B.2 prints non-Christians as 231,327 men and 142,639 women, so if
the other 140,706 non-Christians were half men, code 10 is 69% men; for code 10 to be as balanced
as 2010's `Other`, the rest would have to be 84% men. Code 3's geography (§4) is the other half.

Under the form's code order the simplest account is a symmetric swap, codes 3 and 8 exchanged and
codes 9 and 10 exchanged, which makes the 463 the Judaism answer and the 9,238 other religions.
The 9,238 is not the same answer as the 233,260 in any case: as a share of its province's code 10
it runs from 1.4% in Luapula and North-Western to 21% in Muchinga. Filing both at `other.zm` is
right. The 463 could go to `judaism` under the swap, but its geography is not a Jewish one either
(135 in Eastern, 78 in Lusaka), and it is half a dot at 1:1,000; left alone.

**The questionnaire also confirms B.6's `None` row.** P22 (denomination) is asked only when P21
is 1 (`IF 2 TO 10 SKIP TO P23`), and its code 24 is `NONE`. So the 335,271 are people recorded as
Christian who named no church, as zm2022.py files them.

Pass checks: `check_md` clean; both editions built; `check_rollup` lists the 373,966
non-Christians as gone under the toggle, which the COLUMNS comment explains (ao2024.py). No
change made to the mapping, the note or the build.
