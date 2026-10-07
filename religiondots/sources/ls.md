# Lesotho: religion in the 10 districts from the pooled Afrobarometer, on the 2016 census

Built 2026-10-03 by `fafd1067-ls`, from the negatives scout's row (`sources.md`
§scout-2026-10-03-negatives). Code: `sources/ls.py` (survey), `sources/ls_geo.py` (districts, census
populations, Kontur), `taxonomy/ls2016.py` (mapping), `countries/ls.py` (entry). Record section in
`sources.md`: §ls-2026-10-03.

## 1. What Lesotho publishes: no religion count

The 2016 census dictionary (161 variables) and IPUMS 1996 and 2006 carry no religion item (§11aq).
The 2026 census was enumerated (reference night 11 April 2026) and on 2026-10-03 has published
nothing: `bos.gov.ls/census.htm` says data processing, and MISA Lesotho's fact-check of 27 July 2026
says no results are out. REOPEN on its district tables, in case the form asked.

Two open DHS reports print religion nationally in Table 3.1, women and men aged 15-49, re-read from
the PDFs on every build (`ls.py::dhs_witness`), pooled by weighted number:

| | DHS 2014 (FR309, PDF p.70) | DHS 2023-24 (FR391, PDF p.82) |
|---|---|---|
| Roman Catholic | 39.3% | 35.8% |
| Lesotho Evangelical Church | 17.3% | 15.3% |
| Anglican | 7.4% | 6.3% |
| Methodist | (in Other Christian) | 1.3% |
| Pentecostal | 23.1% | 15.4% |
| Other Christian | 9.1% | 20.1% |
| No religion | 2.3% | 3.7% (men 8.3%) |
| non-Christian | 1.4% | Islam 0.3%, other 0.6% |

The DHS's `Pentecostal` is wider than the survey's: neither DHS card names the Zionist or apostolic
churches, so they are in its Pentecostal or Other Christian. Microdata (religion by district) is
behind a DHS account and was not requested.

## 2. The construction

    row margin      district populations    2016 census, Key Findings Table 2.1.2   EXACT
    composition     each unit's own mix     Afrobarometer R4-R9, 7,102 answers      measured
    national level  neither                 computed

The Nigeria construction (ask/answered/010-ng), as Namibia and Madagascar: nothing fitted, every row
`modelled`. All 10 districts are sampled in every round; the held-out population check passes every
round (r = +0.970 to +0.998; no random pairing of 20,000 reaches it). The census counted 6,564
non-citizens (0.3%); they take their district's mix.

**Why 2016 and not COD-PS**: COD-PS Lesotho is a 2022 projection from the 2016 census; the census's
own count is the base (Ecuador's rule). The survey pool is centred on 2015.

## 3. The churches

They hold their level by round, which is the Madagascar case, and `Christian only` stays at 0.5-4.3%:

| % | R4 | R5 | R6* | R7 | R8 | R9 |
|---|---|---|---|---|---|---|
| Roman Catholic | 43.7 | 38.8 | 42.4 | 42.2 | 41.5 | 39.1 |
| LEC (Evangelical + Calvinist) | 18.1 | 22.5 | 19.0 | 18.2 | 17.0 | 22.8 |
| Anglican | 11.5 | 9.2 | 11.2 | 8.1 | 6.9 | 6.0 |
| Zionist + Independent + Apostolic | 9.6 | 14.1 | 6.9 | 9.3 | 13.4 | 12.0 |
| Pentecostal | 4.3 | 4.7 | 5.7 | 7.3 | 7.5 | 10.4 |
| Methodist | 2.7 | 2.9 | 1.5 | 1.7 | 1.7 | 1.1 |

*after the round 6 drop (§4). Asserted: Catholic, LEC and Anglican within 6-7 points across rounds
and each pooled within 5 points of the DHS 2014; every placed category's pooled level within 3.5
points of rounds 8-9 recomposed (Anglican +2.4, Pentecostal -2.4).

- **LEC is two boxes.** In round 9 Basotho choose `Calvinist` (13.1%) beside `Evangelical` (9.6%) in
  every district; `Calvinist` is a value label from round 5 (11 answers there). Grouped as one.
- **Zionist and independent are one box.** Nobody chose `Zionist Christian Church` in round 4 though it
  is a value label (the merged file's labels, not Lesotho's card, Togo's trap); round 4 has 9.6%
  `Independent` instead (Butha-Buthe 42%). `Apostolic church` is chosen only in round 8 (26 answers).
  Together 9.6-14.1% by round. Node `christianity.africaninstituted`.
- **The survey's adults run more Catholic and LEC than the DHS's 15-49s**, by about as much as the
  DHS itself moved from 2014 to 2023-24. Drawn: Catholic 41.2%, LEC 19.6%, Anglican 8.8%.

## 4. Round 6's traditional religion in the north: dropped

Round 6 (May 2014) has `Traditional/ethnic religion` at 25.3% in Butha-Buthe, 17.9% in Leribe and 10.3%
in Berea; every other (round, district) cell is 3.9% or less (Mokhotlong and Qacha's Nek, late rounds),
and those three districts are 1.7% or less in every other round. In the same round and districts there
are fewest Zionists and no `Independent`, and more `Christian only` and `None`. Neither DHS has a
traditional row (2014: all non-Christians 1.4%). So the 62 answers are dropped, counted and asserted
(`R6_NORTH_TRAD`, `TRAD_ELSEWHERE_MAX`), as Sudan's round 8 Darfur `None`. They are probably Zionist or
independent church members; they are not reassigned.

## 5. Split-half

Median of 10 halvings over six rounds against a per-wave permutation null, with the chi-square veto:

| | rho | null 95th | verdict |
|---|---|---|---|
| Roman Catholic | +0.758 | +0.400 | placed |
| Anglican | +0.812 | +0.400 | placed |
| Methodist | +0.770 | +0.394 | placed |
| Zionist and independent | +0.745 | +0.394 | placed |
| Pentecostal | +0.461 | +0.382 | placed (p=0.02) |
| LEC | +0.267 | +0.412 | residual |
| Other Christian | +0.270 | +0.418 | residual |
| None | +0.212 | +0.394 | residual |
| traditional, Muslim, Other | | | residual |

The quota test passes (15 of 15 wave pairs). The LEC fails although it is the second-largest church:
its district shares move round to round (Mafeteng 28.5% in round 6, 8.6% + 19.3% in round 9), so it is
drawn at one proportion of what the placed churches leave. The tail is the residual (worst
small-category multiple: Muslim 1.13x in Mafeteng).

## 6. As drawn

Catholic 41.2%, LEC 19.6%, Zionist and independent 11.0%, Anglican 8.8%, other Christian 7.0%,
Pentecostal 6.6%, none 2.6%, Methodist 2.0%, other 0.6%, traditional 0.5%, Muslim 0.1%.

| district | Catholic | LEC | Anglican | Zionist/indep. | Pentecostal | Methodist |
|---|---|---|---|---|---|---|
| Thaba-Tseka | 58.3 | 16.6 | 3.1 | 8.5 | 3.8 | 0.5 |
| Maseru | 46.4 | 19.5 | 8.3 | 6.2 | 7.8 | 1.1 |
| Berea | 43.2 | 21.6 | 8.7 | 7.5 | 6.0 | 1.0 |
| Mohale's Hoek | 43.2 | 17.3 | 6.0 | 15.1 | 7.0 | 1.8 |
| Mokhotlong | 41.3 | 19.9 | 5.1 | 13.5 | 5.2 | 4.0 |
| Qacha's Nek | 41.1 | 16.3 | 14.9 | 11.2 | 3.0 | 4.5 |
| Leribe | 37.3 | 19.5 | 13.7 | 11.0 | 6.3 | 1.3 |
| Mafeteng | 34.2 | 22.1 | 9.7 | 12.7 | 6.4 | 2.7 |
| Quthing | 29.8 | 19.2 | 11.8 | 16.5 | 7.2 | 4.9 |
| Butha-Buthe | 25.3 | 20.4 | 3.1 | 26.3 | 9.8 | 4.0 |

Nothing drawn at zero.

## 7. Placement

COD-AB Lesotho (FAO and the Ministry of Local Government 2016), 10 districts, pcodes LSA-LSK, joined to
the census by name (`Botha-Bothe` is COD's `Butha-Buthe`). Kontur LS 2023-11: 222 centroids outside
every district, 94 in South Africa (4,362 people, dropped), 128 inside Natural Earth's Lesotho (7,285,
snapped within 5 km). Kontur is 1.164x the census nationally and 0.99-1.24 of that per district except
**Berea, 0.257**, and Mokhotlong, 1.389. Berea's loss is flat (median hex 20 people against Leribe's
78; Teyateyaneng about its town share once scaled), and neither neighbour is inflated, so it is
Cuba's Granma case: each district's dots are its census count and Kontur only places them inside it.
Both ratios are pinned (`ls_geo.py::OUT_OF_BAND`). No block at the density cap.

## 8. The §14 read

Nothing to raise. Religious practice is free, no group is targeted, and the units are districts of
75,000 to 519,000 people.

## 9. What would improve it

- The 2026 census, if its form asked religion (unknown on 2026-10-03).
- The DHS 2014 and 2023-24 microdata: religion by district for about 9,000 respondents each, with the
  LEC named; a district witness for the LEC, which the survey cannot place. Needs a DHS account.
- The Afrobarometer's round 10, when released.

## Review, 2026-10-03 (`fafd1067-rev9`, full pass)

Checks clean (`check_md`, `built_countries --check`, `check_rollup ls`: all 2,007,201 modelled,
nothing orphaned). Every note figure re-summed from `data/normalized/ls.csv` and matches. Mapping
follows precedent (LEC on `christianity.reformed` as mg's FJKM; Zionist/independent on
`christianity.africaninstituted` as sz/za/zw; `other.ls` as `other.mg`). The round 6 drop and the
LEC's flat draw are the builder's calls and are recorded with their tests; nothing to change.

**On the map, Lesotho is blank in Auto mode. The data is fine; this is the viewer.** Standing the
camera on Lesotho, `countryAt()` in `index.html` returns `za`: it tests outer rings only (by design,
its comment says, so that Lesotho "would be told South Africa" back when Lesotho was not built) and
returns the first match, and South Africa comes before Lesotho in `SHAPES` (index 683 against 1742).
Auto then selects South Africa, the single-country view draws only South Africa's dots, and
Lesotho shows as an empty hole surrounded by dots. With `setCountry('ls')` forced, Lesotho draws
correctly: 2,002 dots, Maseru cluster, the lowland belt, sparse highlands. The fix belongs in the
viewer (prefer the smallest containing shape, or honour holes now that the enclave is built) and
was not made in this review. Lesotho is the first built country entirely enclosed by another;
any later enclave (San Marino, Vatican) will hit the same.
