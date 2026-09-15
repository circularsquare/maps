# Belarus: LiTS III for how many, the Catholic Church's dioceses for where Catholics are

Built 2026-09-14, session `d743fc47-by`. `sources/by.py`, `sources/by_geo.py`, `sources/by_grid.py`,
`sources/lits.py`, `taxonomy/by2016.py`, `countries/by.py`. 9,056,080 people, 7 units, 8 answers,
basis `self_id`, every row `modelled`.

## 1. What exists

- **Census: no religion item.** 2019 Form 2N, 25 questions (scout record, `sources.md`
  §scout-2026-09-14-taiwan-belarus-gabon). Not re-read.
- **LiTS III** (`data/raw/lits/lits_iii.dta`, open): 1,504 adults, 75 PSUs, all 7 units, 181-263 each.
- **EVS 2017**: region `v275b` BY01-BY07 and a 13-denomination card; the data need a GESIS login.
  Not used; it is the obvious second instrument (section 7).
- **Catholic Church, Annuario Pontificio**, via catholic-hierarchy.org (`dgrod`, `dmins`, `dvitb`,
  `dpins`, read 2026-09-14, raw HTML checked): Catholics per diocese. Grodno 548,125 (2023),
  Minsk-Mohilev 652,300 (2023), Vitebsk 167,516 (2022), Pinsk 54,140 (2023). Sum 1,422,081. The
  Church's population denominators are stale (Vitebsk 1,412,489 against Belstat's 1,060,687) and
  are not used.
- **2019 census nationality by oblast**: Belstat bulletin *Национальный состав населения Республики
  Беларусь* (`belstat.gov.by/upload/iblock/df5/df5842f32b1b8a711043f8f54856f5c8.pdf`, 26 pp.), the
  age-group table by oblast, pp. 17-19. Poles 287,693 nationally; Grodno 223,119 (21.73%), Minsk
  city 19,397, Minsk oblast 15,785, Brest 14,893, Vitebsk 9,806, Gomel 2,572 (0.19%), Mogilev 2,121.
- **Population**: Belstat, *Численность населения на 1 января 2026 г. по областям и г.Минску*,
  9,056,080, read from the page's HTML table.
- Searched and empty for a regional Catholic share: Nasha Niva's 2014 "map of religiosity" is
  registered communities per head, not believers; the Grodno oblast executive's confession page
  gives community counts only; a NAS/BISI survey presented September 2026 has no religion item.

## 2. LiTS III has no oblast geography in Belarus

`lits.stability`, 400 PSU halves, shuffled-label null, refusals out:

```
    category                                   national  median rho  null 95th       p
    ORTHODOX CHRISTIAN                          73.545%      +0.107     +0.500  0.4065
    ATHEISTIC / AGNOSTIC / NONE                 14.589%      +0.214     +0.464  0.2818
    CATHOLIC                                     9.208%      +0.368     +0.430  0.1022
    BUDDHIST                                     1.017%      +0.555     +0.470  0.0374  passes
    OTHER                                        0.755%      +0.147     +0.500  0.4514
    OTHER CHRISTIAN, INCLUDING PROTESTANT        0.499%      +0.278     +0.500  0.2743
    JEWISH                                       0.273%      -0.516     +0.500  1.0000
    MUSLIM                                       0.115%      one respondent, no test
```

Coverage 7 of 7. Held-out population check r = +0.986; five of 5,039 orderings reach it, all
swaps of units within the survey's error (Grodno/Mogilev, Brest/Gomel, Gomel/Minsk region), so
forgiven by `lits.held_out`'s rule. The decode is pinned by the PSU coordinates instead (section 5).

**BUDDHIST passes and is overridden to the national rate** (`by.py::OVERRIDE`). Thirteen
respondents in ten PSUs, one each in five Minsk city PSUs and three in one Postavy PSU; about
92,000 people drawn, in a country whose EVS card has no Buddhist box. Kyrgyzstan's builder read
the same card's Buddhist cell as a keying slip (code 2 beside code 1, none); with seven tests at
0.05, a false pass somewhere is about a one-in-three event. Kept as a ceiling rather than excluded,
to match `kg`.

## 3. The Catholic geography is fieldwork, and the ethnicity item shows it

The scout's trap, explained. Weighted Catholic share: Gomel 18.1%, Brest 14.2%, Grodno 11.2%, Minsk
city 7.5%, Minsk region 6.0%, Mogilev 4.2%, Vitebsk 2.5%. The same respondents' ethnicity (`q923`,
`Option 3` = `Pole` in `q923_ethnicity`) against the 2019 census:

```
    unit              LiTS   census
    Brest            1.69%    1.10%
    Gomel            8.17%    0.19%
    Grodno           3.92%   21.73%
    Minsk region     0.23%    1.07%
    Minsk city       0.99%    0.96%
    Mogilev          2.51%    0.21%
    Vitebsk          0.37%    0.86%
    rank correlation over the 7 units: -0.21
```

A census is a count, so this is the survey failing, not Gomel hiding 100,000 Poles. Of Gomel's 41
Catholic respondents, 17 also answer Pole, and 35 of the 41 come from five interviewers (79: 9 with
6 Poles; 76: 8/4; 91: 6/3; 72: 6/0; 70: 6/0). Grodno's two Ostrovets PSUs and Volkovysk return no
Catholic at all, and Grodno city's two PSUs three of 40. Whether it is keying, substitution or
invention cannot be told from the file; any of them makes the survey's minority geography unusable.
The religion item shares the fault, so Catholic geography is set aside on the witness, whatever
the split-half said (it failed anyway, p 0.10). `by.py::ethnicity_witness` asserts the rank stays
under +0.5.

The survey's own Catholics by diocese against the Church's distribution: Grodno 13.1% against
38.5%, Minsk-Mohilev 33.1% against 45.9%, Vitebsk 3.2% against 11.8%, **Pinsk 50.6% against 3.8%**.

## 4. The construction

Spec §3.5a's shape (a survey's total, a church's structure), for one answer:

1. National Catholic share from LiTS, **9.21%**, 833,851 people.
2. Split across dioceses by the Annuario counts, then a uniform share inside each diocese.
   Grodno 32.9%; Vitebsk 9.3%; Minsk-Mohilev 8.7% of Minsk city, Minsk oblast and Mogilev alike;
   Pinsk 1.2% of Brest and Gomel alike.
3. Every other answer at its national share inside each unit's non-Catholic remainder.

**Floor, asserted** (`by.py::poles_floor`): drawn Catholic share at least the census's Poles in
every unit. Tightest is Brest, 1.22% against 1.10%. Brest's Catholics are surely more than its
Poles and Gomel's fewer, but no source here separates the two oblasts, and the grain row says
oblasts; the note says the diocese covers both.

**Level.** 9.2% self-identified against the Church's 15.7% members. Kept the survey's, because
`basis` is self-identification. The Gomel fieldwork may inflate the survey's national Catholic
figure (Gomel's weighted Catholics are about 238,000 of 834,000) while Grodno's shortfall deflates
it; nothing here nets the two.

## 5. Geography

- **COD-AB** `cod-ab-blr` v01 (valid 2022-07-27), 7 ADM1 units.
- **COD's Minsk City is 86.8 km²**; the city is 353.64 km² (ru.wikipedia infobox; the 2012 decree
  gave 348.84). Five of nine city PSUs fall outside it. Replaced by OSM relation 59195 (Nominatim,
  353.0 km²) clipped to COD's Minsk pair, joined with COD's own city polygon, which COD runs
  1.02 km² past OSM's line at 27.69 E, 53.87 N (kept, since nothing says which is right). Drawn
  354.1 km²; Minsk oblast 39,894 km² against the 39,854 km² ru.wikipedia gives.
  geoBoundaries was no alternative: CIESIN 2005, smallest unit 217 km².
- **Witnesses** (`by_geo.py`): Belstat's Russian labels and LiTS's English ones each match exactly
  one COD name; 74 of 75 PSU coordinates fall inside their decoded unit, all 9 city PSUs inside the
  city polygon and no Minsk region PSU inside it. PSU 28 `Vitebski Senno` is 6.9 km over the Mogilev
  line and allowed by name.
- **LiTS `region_name` trap**: `Gomel'  region` (double space), `Grodno region ` and `Minsk ` beside
  the clean spellings. `by_geo.LITS_RAW` asserts the nine raw strings.
- **Kontur** BY 2023-11: 113,421 hexes, 9,491,612 inside the units against Belstat's 9,056,080
  (1.048). Per unit 0.94 (Brest) to 1.17 (Minsk region), banded 0.80-1.25 so a Minsk city/oblast
  population swap would stop the grid.

## 6. Calls someone might reverse

- Catholics by diocesan counts rather than the national rate. The national rate would draw Grodno
  at 9%.
- Uniform share within the Pinsk and Minsk-Mohilev dioceses.
- BUDDHIST kept at the national rate as a ceiling rather than excluded.
- Minsk City as OSM plus COD's patch.

## 7. What would improve it

- **EVS 2017** (GESIS login, not tried): region on the file and a card naming Orthodox, Roman
  Catholic, Greek Catholic, Old Believer and four Protestant bodies. The one instrument that could
  test the Orthodox and none geography and the Catholic level.
- **Registered Roman Catholic communities per oblast** (the Commissioner for Religious and
  Nationalities Affairs) would split Pinsk and Minsk-Mohilev between their oblasts; a location
  count, so as a split weight only. Not searched.
- Not checked: EVS 2008, WVS 1990-2011 for Belarus, Pew's 2015-16 Central and Eastern Europe file
  (account), GGS-II 2017 (registered).

## 8. Review, 2026-09-15 (d743fc47-rev4)

Checks clean: `check_md`, `built_countries --check`, `check_rollup by` (every row modelled, nothing
orphaned). Screenshot at the entry's `view`: dots follow the cities, none outside the border, and
the legend totals match `by.csv`.

- The Catholic construction recomputes off `by.csv`: 833,852 people, 9.21%; Grodno 32.9%, Vitebsk
  9.26%, Minsk-Mohilev 8.67% in all three of its units, Pinsk 1.22% in both. The diocese totals
  reproduce the Annuario ratios exactly.
- The mapping matches `kg2016.py` box for box, and `other.by` follows `other.kg`. The note follows
  Kyrgyzstan's structure and wording. Nothing to change.
- **The national Catholic level is the one number the fieldwork fault can push in one direction,
  and §4 does not size it.** Taking §3's surveyed shares on the drawn populations: put Gomel alone
  (the oblast the Poles witness covers) at its diocese's 1.22% and the national share is 6.8%; put
  Brest there too (surveyed 14.2%, in a diocese the Church itself counts at 2.1% Catholic) and it is
  4.9%. Raising Grodno to its census Poles (21.7%) adds back 1.1 points, so about 6% to 8%. Gomel's
  17 Catholic Poles, in an oblast the census puts at 0.19% Polish, read as records that should not
  exist rather than as Catholics moved from somewhere else, so the drawn 9.2% looks like the top of
  the range rather than the middle. Not moved: the level is the builder's call, and each correction
  above is a guess at which records are wrong. `note_public`'s "The survey can say how many" is
  firmer than §4. A level check that needs no account: Pew's published 2017 report on Central and
  Eastern Europe prints Belarus's national shares from 2015-16 fieldwork (not fetched here).
