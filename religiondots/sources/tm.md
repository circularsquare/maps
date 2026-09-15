# Turkmenistan: DRAWN 2026-09-14 from the Central Asia Barometer on the 2022 census, 6 units, every row `modelled`

Built by session `f95259a4-tm` on the route §11ao opened, after Anita approved it on 2026-09-14
knowing the split-half has no power at six units (*"given the state of data in turkmenistan in
general we're unlikely to be able to do any better than this"*). `sources.md` §9do is the
write-up.

## 0. The build

| | |
|---|---|
| survey | Central Asia Barometer waves 4-6, face-to-face, autumn 2018 to autumn 2019, 1,500 a wave |
| files | `data/raw/cab/CAB-Survey-Wave-{4,5,6}-*.zip`, the Turkmen `.dta` read out of each zip |
| columns | `Region_M` (5001-5006), `Religion_M`, `totwt`, `Ethnic_M`, `SamPt`, `IntCode` |
| witness only | wave 14 (autumn 2023, mobile phone): `DD13`, `MM10`, `FinalWgt1`, copied out of the `.rar` to `data/raw/cab/central-asia-barometer-survey-wave-14-stata-turkmenistan-2023-autumn.dta` |
| population | 2022 census (17 December 2022), section 1 table 1.3, **7,057,841** |
| nationality | 2022 census, section 4 tables 4.1 and 4.3-4.8, sixteen rows per unit |
| boundaries | Kontur Boundaries `kontur_boundaries_TM_20220407.gpkg` (OSM), `admin_level` 4, 6 units |
| placement | Kontur 400 m hexes, 2023-11 (`sources/tm_grid.py`) |
| modules | `sources/tm_geo.py`, `sources/tm_grid.py`, `sources/tm.py`, `taxonomy/tm2019.py`, node `other.tm`; shared `sources/cab.py` |

Census PDFs are cached in `data/raw/tm/` as `tm_census2022_results_en_<n>.pdf`, all with a `%%EOF`
trailer; the index is `stat.gov.tm/en/population-census` and the files are
`stat.gov.tm/population-census-pdfs/results/en/{introduction,1..11,definition}.pdf`. The census
publishes no religion table in any of the eleven sections.

## 1. The population, and the press figures not to use

Table 1.3 gives each unit's total, urban and rural population. **The figures in press coverage
(Turkmenportal, THE AsiaN) are different and also sum to 7,057,841**: Mary 1,616,246, Dashoguz
1,552,725, Lebap 1,446,857, Ashgabat 1,030,445, Ahal 882,230, Balkan 529,338. Each is the rounded
percentage on the report's page-1 map times the national total (22.9% x 7,057,841 = 1,616,246).
Table 1.3's counts are Ashgabat 1,030,063, Ahal 886,845, Balkan 529,895, Dashoguz 1,550,354, Lebap
1,447,298, Mary 1,613,386. `tm_geo.py` asserts urban + rural per row, the columns against table
1.1, every urban share against the map to two decimals, and every figure against the PDF's text
layer. A search result carried the press figures; the PDF corrected them.

**Arkadag city** was split from Ahal with velayat status in 2023. The census counts it inside
Ahal (table 1.5: 567 people), and Kontur's 2022-04 boundaries have no Arkadag unit, so nothing
needs merging. Kontur's 2023-06 release is also on HDX with six level-4 units.

## 2. Boundaries

No OCHA COD-AB (`cod-ab-tkm` is a 404). geoBoundaries gbOpen `TKM-ADM1` (Wikimedia, "2007") holds
five polygons, not the six its API reports: no Ashgabat, and Ahal spelt `Ahai`. Kontur's OSM
extract has six, with Ashgabat at 979 km2 (the city after its 2013 expansion into Ahal).

**Join witnesses:** (1) OSM's English and Turkmen names, the census rows and the barometer's labels
pair one-to-one, worst margin 0.500; (2) Ashgabat is the smallest polygon, and Balkan, Lebap,
Dashoguz and Mary have the westernmost, easternmost, northernmost and southernmost centroids; (3)
the survey decode below.

## 3. The survey decode, and a held-out check that has to use the survey's own frame

The wave 4 methods report (p. 14 and Table 6, p. 16) gives the Turkmen design: target 18+,
frame "National Statistic Agency 1995", 11 strata (five velayats x urban/rural, Ashgabat all
urban), settlements as PSUs, ten interviews each, random route. Table 6 allocates **Ahal 21,
Ashgabat 19, Balkan 14, Dashoguz 31, Lebap 30, Mary 35** PSUs. All six numbers differ, and wave 4's
file holds exactly ten interviews and that many distinct `SamPt` values under each label. That
pins the decode with no names involved (`tm.py::frame_witness`).

`lits.held_out` against the **2022 census fails**: r = +0.960, Ashgabat 0.82x and Ahal 1.18x its
population share, and six of 719 orderings beat the truth. That is the 1995 allocation and
Ashgabat's 2013 expansion into Ahal, not a wrong decode. Against the **1995 frame** Table 6 prints
it passes: r = +0.994, none of 719 orderings reaches it. The census comparison is still printed.

**Quota test:** all three wave pairs compared, no identical free cell, p = 1.

## 4. Why the survey's regional shares are not what is drawn

The survey reaches a 1995 country. Weighted, waves 4-6, against the 2022 census:

```
                 Turkmen            Uzbek              Russian/Ukr/Arm    other
    Ahal       88.33 vs 98.60     0.76 vs  0.20      3.47 vs 0.26       7.44 vs 0.94
    Balkan     92.15 vs 93.71     0.47 vs  0.55      3.33 vs 3.18       4.04 vs 2.56
    Dashoguz   94.72 vs 67.48     3.33 vs 31.57      0.07 vs 0.20       1.88 vs 0.75
    Lebap      78.28 vs 89.28    14.73 vs  9.43      3.70 vs 0.86       3.29 vs 0.42
    Mary       91.57 vs 92.09     0.85 vs  0.41      2.33 vs 1.08       5.26 vs 6.41
    Ashgabat   56.79 vs 89.86     0.12 vs  0.50     35.57 vs 7.71       7.52 vs 1.92
    national   84.89 vs 86.72     4.11 vs  9.10      6.37 vs 1.87       4.64 vs 2.31
```

And how each nationality answers, all velayats pooled:

```
    cell                            n    Muslim  Christian  non-believer  other answers
    Turkmen                     3,942    99.88%     0.03%      0.03%         0.05%
    Uzbek                         142   100.00%     0.00%      0.00%         0
    Russian, Ukrainian, Armenian  248     3.99%    92.93%      3.08%         0
    other                         168    95.03%     4.66%      0.00%         0.31%
```

Christian is an ethnic answer, and the survey has three to four times the census's share of the
people who give it (in Ashgabat, 35.6% against 7.7%). The wave 5 methods report says the
official ethnic figures for Turkmenistan are not reliable and the 1995 census is the best the
project had (p. 17, footnote 3). **The 2022 census's nationality table is the office's own count and
twenty-seven years newer than the frame, and it is used.** Russians at 1.62% are a plausible
figure after three decades of emigration; nobody outside the office can check it.

**The construction:** Christian share of a velayat = sum over the four cells of the census's cell
share there times the survey's national Christian share in the cell. The within-cell rate is
national because 248 Russian/Ukrainian/Armenian respondents split six ways gives a few dozen
per velayat. Muslim is the residual. The four small answers share it at their national
proportions, themselves post-stratified on nationality.

```
    velayat     waves 4-6   per-velayat cells   DRAWN   wave 14   census RUA share
    Ashgabat      35.32%          8.04%          7.28%    8.44%        7.71%
    Balkan         3.33%          3.08%          3.11%    7.74%        3.18%
    Mary           1.75%          1.11%          1.34%    4.80%        1.08%
    Lebap          3.49%          0.82%          0.85%    2.12%        0.86%
    Ahal           3.47%          0.25%          0.32%    1.65%        0.26%
    Dashoguz       0.07%          0.22%          0.24%    0.49%        0.20%
    national       6.16%                         1.87%    3.50%        1.87%
```

"Per-velayat cells" uses each velayat's own within-cell rate where a cell has 30 or more
respondents; it is printed and not drawn, and it lands within 0.8 points of the drawn figure.

**Witnesses on the drawn ordering** (exact one-sided p over all 720 orderings): wave 14, never
pooled, rho **+1.000, p = 0.0014**; waves 4-6's own regional shares +0.543, p = 0.149 (Lebap
and Ahal read high there against the census's nationality mix; Lebap's includes the wave 5
cluster §11ao noted).

**Wave 14 is not pooled.** It is a different mode (interviewers dialling mobile numbers from
home, methods report pp. 7, 37) and weighted to the same 1995 targets. Its national 3.50%
Christian sits between the drawn 1.87% and the face-to-face 6.16%; its Ashgabat 8.44% is close
to the drawn 7.28%.

## 5. The split-half, which decides nothing here

Three waves, all three 1-against-2 halvings (after the `cab.stability` fix below), 2,000-draw
per-wave permutation null over six velayats, spatial chi-square:

```
    answer                         n   median rho  null 95th      p     chi2 p
    Muslim                     4,245     +0.600     +0.600     0.0665   1e-200    fails
    Christian                    245     +0.600     +0.600     0.0665   2e-185    fails
    A non-believer                 7     +1.000     +0.775     0.0285   1e-09     passes, REFUSED
    another faith, no particular faith, Other (vol.): one respondent each, no test
```

Six units cannot do better than this; §11ao predicted it. `A non-believer` passes on seven
respondents, four of them wave 6's Ashgabat interviews (largest wave-velayat cell 57%, against
Christian's 28%), which is spec §12's Uzbekistan cluster rule. It is refused in `tm.py`'s
`REFUSED` and drawn at the national rate. The three single answers are all wave 4, Ashgabat.

Christian's 245 respondents sit in 93 sampling points (largest 2.9%) and 49 interviewers (largest
7.3%), so it is not one cluster.

## 6. What is drawn

```
    97.986%   6,915,669  Muslim
     1.870%     132,010  Christian
     0.085%       5,991  A non-believer
     0.030%       2,140  A believer of another faith
     0.022%       1,519  A believer of no particular faith
     0.007%         512  Other (vol.)
```

Christians by velayat: Ashgabat 75,021 (56.8%), Mary 21,564, Balkan 16,472, Lebap 12,315,
Dashoguz 3,782, Ahal 2,856. Nobody in waves 4-6 refused or said they did not know, so there is no
`gap` and no §3.5 lean.

## 7. Kontur placement layer: no capped block

Checked against runlog 2026-09-14 `kontur scan` with `plateau_scan2.py tm`: 32,455 hexes, highest
density 29,960 per km2, **no hex at the 46,200 cap**. The one dense block is 55 hexes holding
1,029,874 people (57.2% of Ashgabat's Kontur weight), peaking 4.1 km from the city point with the
centre's own density equal to the peak, so it is the city and not a displaced block. Nothing to
record for the cap decision.

Kontur against the census by unit reads Dashoguz 0.35x, Ahal 0.78x, Mary 0.79x, Lebap 0.90x,
Balkan 1.72x, Ashgabat 1.75x. That only moves dots inside a velayat, not its totals. Ashgabat's
1.75x is Kontur putting about 1.8 million people in a city the census counts at 1.03 million;
Dashoguz's 0.35x is the reverse. Neither was chased.

## 8. §14, as recorded

Turkmenistan registers and restricts religious groups, publishes nothing on religion, and is
about as closed to polling as any country on this map (wave 4's report records four interviewers
detained and their tablets confiscated during fieldwork, p. 44). How candidly anyone answers a
stranger about religion there is not something the survey can show. The pressured groups
(Protestant and Jehovah's Witness converts from Muslim families) sit inside one `Christian` code
with the Orthodox, and **a Christian of Turkmen nationality is drawn at the same national rate
(0.03%) in every velayat**, so the map does not locate them. Units average 1.2 million people.
Anita approved the build on 2026-09-14.

## 9. Shared code changed

- **`cab.stability` enumerates every halving on an odd wave count** (spec §12, Colombia). The
  filter is now `if n_w % 2 or 0 in a`. Uzbekistan, the only other caller, was rebuilt to a
  scratch file before and after: sha256 `2cbce1d1dfea4a07f1b354aa83dcffea031904175cb9ac241f339d9b0c006d5d`
  both times.
- `cab.assert_not_quota` was called with `WAVES = [4, 5, 6]` and reports 3 of 3 pairs compared.

## 10. Not done

- **The adult-only lean.** The survey is 18 and over; the nationality counts are all ages, which
  is what the dots are. Russians are older than Turkmens, so the within-cell rates are adults'
  and the cell shares everyone's; no age-by-nationality table was pulled.
- **A PSU split-half.** `SamPt` is in all three files (150 per wave). Uzbekistan's review found it
  agreed with the wave split; at six units it would still have little to rank.
- **Kontur's Ashgabat and Dashoguz ratios** (section 7).

## 11. Review, 2026-09-14 (session `f95259a4-tmrev`), read from the raw files

- **The census nationality counts are read correctly.** Tables 4.1 and 4.3 re-read from
  `tm_census2022_results_en_4.pdf`: `tm_geo.NATIONALITIES` is in the PDF's row order (Russians
  third, Armenians sixth, Ukrainians twelfth), and Ashgabat's 68,188 + 9,761 + 1,460 give the
  7.71% used.
- **The within-nationality rate is at a level the sample holds.** The Russian/Ukrainian/Armenian
  cell is 248 respondents and 236 Christians (218 Russian, 17 Armenian, 1 Ukrainian); 95.1%,
  94.4% and 90.1% Christian by wave; 176 of the 248 in Ashgabat and 1 to 24 in each other
  velayat. National is the right level.
- **Two figures in `tm.py`'s docstring were wrong and are fixed:** the survey is 6.4%
  Russian/Ukrainian/Armenian weighted (it said 5.4%), and it has two Turkmen Christians, both
  in Ashgabat (it said nine; nine is every Christian outside that cell).
- **The `other` cell repeats the frame problem on a small scale.** Its 4.66% Christian rate is
  7 respondents (4 `Other (vol.)`, 2 Tatar, 1 Azerbaijani; 6 in Ashgabat). The survey's cell
  holds 6 Balochi of 168, none Christian; the census's holds 85,384 Balochi of 162,787, 83% of
  them in Mary. So about 4,000 of Mary's 21,564 drawn Christians are that rate applied to
  Balochi, and the cell as a whole draws 7,586 Christians (5.7% of the map's), mostly among
  Balochi, Kazakhs and Karakalpaks. The velayat ordering does not change. Not rebuilt. If
  `tm.py` is touched again, giving the census's Balochi, Kazakhs, Karakalpaks, Persians, Afghans
  and Kurds the survey's Azerbaijani/Uygur/Balochi rate (99 respondents, 1 Christian) removes
  most of it.
- **Wave 14 is independent data, and thin.** It is a separate 2023 phone sample, never pooled, so
  the ordering check does not reuse waves 4-6. But rho +1.000 rests on 14 Christian respondents
  (1 to 4 a velayat) and on `FinalWgt1`, which weights Russian, Kazakh, Azerbaijani and
  `Other (vol.)` respondents 3 to 7 times a Turkmen. Balkan's 7.74% is two Russians; Ashgabat's
  8.44% is four people who are 1.47% unweighted. Unweighted, Mary and Balkan swap (rho +0.943).
  Shuffling the 14 answers across respondents still gives p = 0.001 weighted and 0.011
  unweighted, so the ordering holds; dropping the one Lebap `Other (vol.)` Christian takes rho
  to +0.829. The Ashgabat level match (8.44% against 7.28% drawn) is the weights, not
  corroboration. The note_public sentence citing wave 14 was cut on this.
- **Boundaries and population line up.** Kontur 2022-04's Ashgabat (979.3 km2) is the same
  polygon as 2023-06's, whose level-6 units are table 1.4's four etraps (Kopetdag, Buzmeyin,
  Bagtyyarlyk, Berkararlyk, 980 km2 together); 2023-06's Ahal holds table 1.5's seven etraps
  plus Arkadag. Table 1.3's counts sit on the polygons drawn. The 2013 change breaks only the
  1995-frame comparison in section 3.
- **The `cab.stability` halving fix is intact and does not move Uzbekistan.** Current `cab.py`
  (16:45, with uz2's `SamPt` docstring correction also in it) uses 3 halvings on 3 waves and 10
  on 5, the same sets as a brute-force list, and unchanged sets on 2, 4, 6 and 8. `sources/uz.py`
  run into the scratchpad with the fix and with it reverted gives sha256
  `dd82f2d3b4b150a6fbb89deb3f1bfbd95ba04732004eabfa3212aeb24eb29dbc` both times, equal to
  `data/normalized/uz.csv` as written at 16:42; both of its `stability` calls see all six waves.
- **note_public:** the wave 14 sentence is cut, and "Most of the difference is Ashgabat" now gives
  the two figures (35% in the survey, 7.3% drawn). `tiles.py --refresh-meta` not run, because
  other sessions hold shared files. §14 wording is factual and short. Screenshot: dots follow
  the oases and the Kopetdag foothills, none in the Caspian.
