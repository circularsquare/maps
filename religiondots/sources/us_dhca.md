# United States: Sikhs and Yazidis from the 2020 Census Detailed DHC-A

Built 2026-09-15 by session `d743fc47-ussikh`, from the scout `sources.md §scout-2026-09-15-us-small-religions`.
`sources/us_dhca.py` rebuilds `data/normalized/us_dhca.csv`; `countries/us.py` draws it inside spec §3.5a.
Cite as `sources.md §us-2026-09-15`.

**Scattered 2026-09-15:** 68 Sikh dots of 326,701 at 1:1,000 and 6 of 32,594 at 1:10,000; one Yazidi ring
in each edition. Built by the supervisor's build tail the same day (`runlog.md`).

## 1. Re-fetch recipe

```
python sources/us_dhca.py --fetch
```

Two files into `data/raw/us_dhca/`, both keyless and size-pinned (Content-Length read 2026-09-15):

| file | URL | size |
|---|---|---|
| `2020-ddhc-a.zip` | `www2.census.gov/programs-surveys/decennial/2020/data/detailed-dhc-a/2020-ddhc-a.zip` (Last-Modified 2024-01-10) | 41,121,420 |
| `iterations.xlsx` | `.../technical-documentation/complete-tech-docs/detailed-demographic-and-housing-characteristics-file-a/2020-census-hispanic-origin-and-race-iterations-list.xlsx` (2023-09-20) | 113,686 |

The reader streams `ddhca_t01001.csv` out of the zip (about a minute) and keeps five iterations at USA,
STATE, COUNTY and TRACT. Output: 1,738 rows, `level, geoid, iterid, label, node, count, suppressed`.

## 2. What the table is, and its traps

- **The census asks no religion; its race write-in codes two religions as groups.** "Sikh" is a detailed
  Asian group (codes 4305-4309) and "Yazidi" a detailed Middle Eastern and North African group.

  | ITERID | label (iterations list) | USA | node |
  |---|---|---|---|
  | 3845 | Sikh alone or in any combination | 70,697 | `sikhism` |
  | 3788 | Sikh alone | 48,321 | witness |
  | 1207 | Yazidi alone or in any combination | 630 | `yazidism` |
  | 1096 | Yazidi alone | 444 | witness |
  | 3839 | Asian Indian alone or in any combination | 4,768,846 | witness, USA only |

  Both Sikh figures equal the Sikh Coalition's quotation of the release (*Updated Census Figures Severely
  Undercount U.S. Sikhs*, 2023-09-28). Every code is looked up by label in the iterations list at run time.
- **Noise and a threshold.** A detailed group is printed below the state only at a noise-infused count of 22
  or more, and noise is added to each cell separately (summary file technical document, Tables 1 and 3). So
  counties sum to 68,483 of 70,697 (96.9%) and states to 70,708, and tracts do not sum to their county. The
  reader checks that no county's printed tracts exceed it by more than 11 per tract, and that "alone" never
  exceeds "alone or in any combination" by more than two margins (566 Sikh cells, 0 inversions).
- **A withheld cell** reads `-888888888` with `ANN` = `X`; `classify()` accepts that and plain digits and
  raises on anything else.
- **DC's county row is its state row**, so it sits under the threshold (Sikh 19, Yazidi 5). Allowed by name
  only when it equals the state cell.
- **Joins.** All 213 Sikh and 5 Yazidi counties join `cb_2020_us_county_500k`, all printed tracts join
  `cb_2020_us_tract_500k` (the 2020 vintage the US build places on). No Puerto Rico rows.
- **Not drawn by construction:** 2,214 Sikhs in counties under 22 (they stay in `other.us`), and 4 printed
  Sikh tracts (95 people) and 3 Yazidi tracts (66) whose county is not printed.

## 3. Why it is drawn as counted and never scaled

The count holds people who wrote the word. A Sikh who ticked Asian Indian and wrote nothing is not in it.
Two outside figures, neither with geography:

- The Sikh Coalition estimates "more than 500,000 Sikhs in the United States", from its work with over 350
  gurdwaras (same post).
- Pew, *Religion Among Asian Americans* (fielded 2022-07-05 to 2023-01-27, 7,006 adult respondents,
  national only): Indian Americans are "Hindu (48%) ... Christian (15%), Muslim (8%) or Sikh (8%)". 8% of the
  census's 4,768,846 Asian Indians is about 381,500, for scale only, since Pew's share is of adults.

So the write-in is a fifth or less of the community. Scaling it up fails `[[feedback_proxy_residual_nameable]]`:
where the missing Sikhs live is published nowhere, and the gurdwara check below shows agreement, not a
correction table. It is drawn as counted, `measured` (a count read at the county it was printed for), and
the rest stay in `other.us`, which is spec §2.7a's construction.

Checks carried from the scout, not re-run here: 88.5% of county-printed Sikhs live in a county with an ASARB
gurdwara (group 416); Spearman +0.53 over the 117 counties with both; +0.61 by state against Pew's
other-world-religions line. ACS 2020-24 B02018 row 026 finds 45% of the census count in the same counties
(Spearman +0.70), a sample superseded by the census.

Yazidis hold on the same terms: the label and code check, the national pin, and 476 of 551 county Yazidis
in Lancaster County NE (Lincoln), the community's known centre. No outside count was looked for.

## 4. Inside §3.5a: what it takes out of the survey residual

`taxonomy/us_pew2024.py` maps Pew's `other-world-religions` line to a set of roots, and the residual is the
line minus everything drawn under any of them (`us_rebase.compute`). The census county rows join the
measured frame the residual is taken against (`countries/us.py::_us_measured`), exactly as the Bahá'í roll
does. `yazidism` was added to the set (REVIEW there): a Yazidi answering Pew has nowhere else to be coded,
and leaving it out would draw the 551 twice.

Computed in `us_rebase.compute` itself, as built against with the census rows (scratch `overflow.py`):

| | as built | + Sikhs | + Sikhs + Yazidis |
|---|---|---|---|
| `other.us` residual | 1,124,212 | 1,058,891 | 1,058,368 |
| `unaffiliated` residual | 60,554,819 | 60,551,657 | 60,551,629 |
| overflow on the lump | 19,736 in 16 states | 22,899 in 17 | 22,927 in 17 |
| country drawn total | 326,813,748 | 326,813,748 | 326,813,748 |

Where the census count exceeds what Pew's line leaves after the Bahá'í roll, §3.5a charges the difference
to that state's unaffiliated. That is **3,190 people in 10 states**, against the scout's approximation of
4,156 (which used ASARB's state population and no county frame):

| state | census drawn | `other.us` change | `unaffiliated` change | Pew line respondents |
|---|---|---|---|---|
| Indiana | 1,596 | 0 | -1,596 | 0 |
| Maryland | 978 | -144 | -834 | 1 |
| North Carolina | 289 | 0 | -289 | 1 |
| Kansas | 165 | 0 | -165 | 0 |
| Missouri, Mississippi, Utah, Tennessee, New Mexico, DC | 22 to 68 each | 0 | -22 to -65 each | 0 or 1 |

(Tennessee's 23 Yazidis and DC's 5 add 28.) The rule was applied as written and not asked about: it is
0.2% of the 1.63M §3.5a already charges, every state absorbs it, and 7 of the 10 states are a survey zero
against a census count (the case §3.5a names) while the other 3 rest on one respondent.

**The unchurched pool is still ASARB's** `population - adherents`; the census's 69,034 people are not taken
out of it. They move no county's share of a state's spread by more than a rounding.

## 5. Placement

`countries/us.py::_UsCensusTracts` wraps `us_weights.Weighter`. For a `sikhism` or `yazidism` county row
(measured only; a `plain` row passes through), the dots go first to the tracts the census printed, in
proportion to their counts, and the county's remainder goes over its other tracts on the inner weight: for
Sikhs the authored tie (Punjabi at home 1.0, ACS Sikh 1.0, South Asian 0.2), for Yazidis tract population.
Checked on the weights (scratch `verify.py`):

| county | count | printed tracts | on printed | on the rest |
|---|---|---|---|---|
| Sutter CA 06101 | 2,812 | 17 of 21 | 2,788 | 24 |
| Fresno CA 06019 | 5,297 | 52 of 225 | 4,916 | 381 |
| Queens NY 36081 | 4,456 | 49 of 724 | 3,497 | 959 |
| King WA 53033 | 2,900 | 32 of 494 | 1,884 | 1,016 |
| Marion IN 18097 | 369 | 5 of 253 | 245 | 124 |

Yazidis draw no dot at 1:1,000 (551 people), so they are one ring, at Lancaster County, which does not use
the weights.

## 6. Calls someone might reverse

- `yazidism` inside Pew's other-world-religions set (REVIEW in `taxonomy/us_pew2024.py`).
- The overflow charged to unaffiliated rather than allowed to raise the country total.
- County grain with tracts as placement only. Tracts are printed for 55% of county Sikhs; drawing them as
  their own units would lose the rest of each county or double it.
- The 2,214 Sikhs in unprinted counties left in `other.us` rather than spread from the state rows.

## 7. Not done

Jains, Zoroastrians, Daoists and Shinto still draw nothing (congregations only in ASARB 2020). Open leads,
unchanged from the scout: the 2010 US Religion Census's Zoroastrian adherents by county (ARDA `RCMSCY10`)
and FEZANA's 2012 demographic table (403 to WebFetch, a browser job).

## 8. Review, 2026-09-15 (session `cb8b206e-rev2`)

A full review found nothing to change. `check_rollup.py us` is clean: no derived rows, and none orphaned. A
screenshot at the country's `view` looks normal, with Sikhism at 68k in the legend. `sikhism` and `yazidism` were
already on the tree, so there are no new legend rows. The note's figures match §2 to §4. §14 raises nothing: the
census itself prints both counts by county and tract, the map places nothing finer than that, and a county
ring for Lincoln's Yazidis only shows what is already public. The "waiting for the supervisor's build tail"
line at the top is out of date; the tail ran on 2026-09-15 (runlog). One thing is untested and has no finding
behind it: whether the share of Sikhs who write the word in the race question differs between long-settled
and newer communities. If it does, the drawn Sikhs lean toward the communities that write it most, and nothing
published measures that.
