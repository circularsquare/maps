# Palestine — PCBS census 2017, religion by governorate

**Drawn 2026-09-15** (session `d743fc47-ps`). 16 governorates, 3 categories, 4,663,917
Palestinians drawn of 4,665,426 counted, every row `measured`. 4,662 dots at 1:1,000, 465 at
1:10,000.

- `sources/ps.py` -> `data/normalized/ps.csv` (the PDF is `data/raw/ps/book2364-1.pdf`, pinned
  at 22,646,657 bytes)
- `sources/ps_geo.py` -> `data/geo/ps/ps_governorates.gpkg`, `ps_hexes.gpkg`, `ps_lookup.csv`
  (COD-AB `cod-ab-pse` v01 admin 2, Kontur PS and IL, CBS settlement counts from the Israel build,
  OCHA community points as a witness)
- `taxonomy/ps2017.py` -> the mapping; `countries/ps.py` -> the entry; `taxonomy/branches.py`
  `other.ps` is the one new node
- sources.md **§ps-2026-09-15** is the summary; **§scout-2026-09-14-asia-oceania** was the scout's.

```
python sources/ps.py     --fetch
python sources/ps_geo.py --fetch     # needs data/normalized/il.csv and data/geo/il/ on disk
```

## 1. What PCBS publishes

| release | religion | tier |
|---|---|---|
| **Census 2017, *Preliminary Results* (Feb 2018, `pcbs.gov.ps/Downloads/book2364-1.pdf`, 81 pp)** | **Table 3, governorate x Islam / Christian / Other / Not Stated, counts** (p.35, PDF index 33) | **16 governorates** |
| same book | Table 1 (national totals and under-coverage), Table 2 (everyone counted by governorate and sex), Table 25 (population by locality, counted plus estimate) | no religion |
| Census 2017 and 2007 microdata (PCBS NADA, English catalog 663 licensed, Arabic 641) | no religion variable (scout) | |
| Census 2007, 1997 | the forms ask religion; no table below the nation found (scout); UNSD table 28 holds both national rows | national |

**Not checked, so not a negative:** the 2017 *final results* population reports (a Princeton
DataSpace record of the *Detailed Report* answered 401), the per-governorate 2017 census booklets
other than Rafah's (the scout read Rafah's, `book2429.pdf`, with no religion table), and
`info.wafa.ps/ar_page.aspx?id=2116` (HTTP 500, a browser job). A religion-by-locality table would
most plausibly be in the final-results series.

## 2. Three nested totals, and why Jerusalem closes

| table | universe | Palestine | Jerusalem |
|---|---|---:|---:|
| 3 | Palestinians counted | 4,665,426 | 392,835 |
| 2 | everyone counted (Table 1's "actual counted population") | 4,705,601 | 414,786 |
| 25 | counted plus PCBS's post-enumeration estimate (Table 1: 75,377, 1.7%) | 4,780,978 | 435,483 (J1 281,163, J2 154,320) |

The scout recorded J1 + J2 = 435,483 against Table 3's 392,835 as unresolved. It is the nesting:
Table 25 is the largest universe, and its footnote says it includes the uncounted estimate. J2
alone is 154,320 even with the estimate, so Table 3's Jerusalem row must include J1. PCBS lists
J1's 21 localities under Table 25 (Kafr A'qab to Umm Tuba). `check()` asserts each step, per
governorate: Table 3 never exceeds Table 2 anywhere.

The 40,175 counted people outside Table 3 are 21,951 in Jerusalem, 7,110 in Ramallah & Al-Bireh
and 2,856 in Bethlehem, then small numbers elsewhere. PCBS does not say who they are beyond "not
Palestinian". They are not the Israeli settlers, whom this census does not enumerate at all.

## 3. The form

Form 25 PHC, *Household and Housing Conditions Questionnaire* (IPUMS `enum_form_ps2017a.pdf`,
read 2026-09-15). The person block has a column group headed **For Palestinians only**; Religion
is in it with `1. Muslim 2. Christian 3.Other`. No code for no answer and no box for no religion.
So `Not Stated` (1,509) is blank or unreadable, and a Palestinian with no religion could only
leave it blank or answer Other. Nothing goes to `unaffiliated`.

## 4. The mapping (`taxonomy/ps2017.py`)

Islam -> `islam`; Christian -> `christianity` (one box; the churches are not separated anywhere
PCBS publishes); Other -> new `other.ps`; Not Stated and Total EXCLUDED. Nablus's 361 Other may be
largely the Samaritans of Kiryat Luza on Mount Gerizim; the book does not say, and the mapping
does not assert it.

Christians: Bethlehem 23,165 (10.9%, 49.4% of all Christians), Ramallah & Al-Bireh 10,255 (3.3%),
Jerusalem 8,558 (2.2%), Jenin 2,699 (0.9%), Gaza 1,082; Gaza Strip 1,138.

## 5. Boundaries and the join (`sources/ps_geo.py`)

COD-AB `cod-ab-pse` v01 admin 2, sixteen governorates, valid 2023-10-19. It is the dataset whose
admin 0 `il_geo.py` cut Israel on, so the two entries share one line. Six of sixteen names differ
between PCBS and COD, so the join is an authored pcode table, witnessed three ways:

1. COD's pcodes run in Table 3's order and COD's admin 1 matches each row's territory.
2. OCHA's 893 community points (`palestiniancommunities_wb_gs.zip`, pop2017 from PCBS) joined to
   COD by location and summed against Table 2: rho +0.988, summed |log ratio| 0.715, and none of
   20,000 random pairings reaches 3.861. Fifteen governorates within 25% (Jerusalem 1.125, which
   carries J1's estimate). **Dier Al-Balah is 0.716, and the reason is OCHA's file:** it has eight
   of the governorate's eleven Table 25 localities, missing An Nuseirat (54,851), Al Bureij
   (15,491) and Al Maghazi (9,670) while keeping their camps, and those three are exactly the
   80,012 gap. Excepted by name in `COMM_BAND_EXCEPT`.
3. OCHA's 25 East Jerusalem communities: 23 inside COD's Jerusalem; the other two, Khirbet Khamis
   and Umm al-'Asafir, have no population figure, are not on PCBS's J1 list, and sit just south of
   Beit Safafa inside COD's Bethlehem. Pinned in `EJ_OUTSIDE_KNOWN`.

## 6. Placement: two Kontur extracts, settlements taken out

**Kontur's PS extract has no hexes in J1.** Inside COD's Jerusalem it holds 375,091 people and the
IL extract adds 350,549. Both are read and de-duplicated on `h3` (410 shared, identical).

**The settlements are in Kontur's weights and not in the census.** Before: Ramallah & Al-Bireh
1.85x Table 2, Jericho 1.92x, Jerusalem 1.75x, Bethlehem 1.69x, Salfit 1.55x. CBS's 2022 census
units that `il_geo.py` dropped beyond the Green Line (267, 1,097,156 people, 723,899 Jews and
Others) are rebuilt with `il_geo.build_units()`, and each unit's non-Arab count is taken off the
hexes it overlaps, in proportion to hex population times overlap, floored at zero.

**CBS draws 119 small localities as 0.008 km2 placeholders** (219,586 people; Talmon, Shilo,
Kiryat Arba and Beit El are single 0.008 km2 polygons in `statareas2022.geojson` itself). They
are replaced by a disc sized at 4,000 people per km2. With area overlap alone, Hebron lost 96
people against CBS's 21,950.

Result, 477,068 removed of 723,899 asked (hexes floor at zero):

| governorate | removed | within 1.5 km of an OCHA community, before / after | Kontur / Table 2, before / after |
|---|---:|---|---|
| Jerusalem | 209,887 | 0.889 / 0.961 | 1.75 / 1.24 |
| Ramallah & Al-Bireh | 89,037 | 0.788 / 0.870 | 1.85 / 1.57 |
| Bethlehem | 80,326 | 0.980 / 0.990 | 1.69 / 1.32 |
| Salfit | 37,596 | 0.868 / 0.947 | 1.55 / 1.04 |
| Qalqiliya | 25,505 | 0.933 / 0.946 | 1.21 / 0.97 |
| Hebron | 15,419 | 0.752 / 0.752 | 1.25 / 1.23 |
| Nablus, Jericho, Tulkarm, Tubas, Jenin | 1,666 to 6,792 each | rise or hold | little change |

Kontur is 2023 and the count 2017, so about 1.13x is the expected level. **What is left:**
Ramallah & Al-Bireh at 1.57x and Jericho at 1.78x still carry weight that is probably settlement
(small outposts CBS does not list, and discs that miss) or Kontur error; Tubas at 0.59x is Kontur
reading low. Some Palestinian dots will still fall in settlements, fewer than before. The
assertion is that removal never moves weight away from the community points.

## 7. What the map does not show

- **Israeli settlers.** Not on this entry: the census does not count them, and Israel's entry
  drops everyone beyond the Green Line (Anita's decision, 2026-09-07, `sources/il.md` §7).
  About 720,000 in CBS's 2022 units beyond the line. Since 2026-09-15 they are drawn as their own
  entry, `xs`, part of neither country (ask 028's ruling, `sources/xs.md`); `gap` and
  `note_public` point there.
- **The 40,175 non-Palestinians counted**, never asked religion; with the 1,509 not stated, 0.89%
  of the counted population (`gap_share` 0.0089, hand-written; the tool confirms the 0.03% part).
- **The 75,377 PCBS estimates the count missed** are in no table by governorate and religion.
- **Gaza since October 2023.** The dots are where people lived in 2017. `note_public` says so.

## 8. §14

Filed as ask 028: whether Palestine should be drawn as built. Decision taken: drawn at the 16
published governorates. The Gaza Strip's 1,138 Christians (1,082 in Gaza governorate) are about
one dot, placed across a governorate of 640,314; their churches' location is public, and rule 2
(no finer than the state publishes) is met. Reversing costs removing `ps` from `ORDER`.

## 9. Review, 2026-09-15 (session `d743fc47-rev8`)

Full pass. `check_md.py` clean, `built_countries.py --check` ok, `check_rollup.py ps` clean
(4,663,917 measured, nothing derived).

- **Figures.** Every `note_public` figure recomputes off `ps.csv`: Islam 98.9%, Christian 1.0%,
  Bethlehem 23,165 (49.4% of Christians, 10.9% of the governorate's Table 3 total), Ramallah &
  Al-Bireh 10,255, Jerusalem 8,558, Gaza Strip 1,138. `gap`'s 0.89% is (40,175 + 1,509) /
  4,705,601; `grain`'s 292,000 is 4,665,426 / 16.
- **Mapping.** Christian on the parent, Islam with no branch, Other on `other.ps` as `other.cg`,
  `other.sl` and `other.bn`: agreed.
- **Against ask 028's ruling.** Built as ruled. The Jerusalem row includes J1: PCBS's own list
  (book2364-1.pdf, the Table 25 footnote, PDF index 80) names 21 localities, Beit Safafa and
  Sharafat among them. Gaza is drawn as counted in 2017 with the sentence. The settlers are on
  `xs`: `counts.json` carries exactly one `"territory": false` entry, and
  `country_shapes.geojson` has `il` and `ps` and no `xs`.
- **Nobody is drawn twice across `il`, `ps` and `xs`.** Three boundaries:
  1. `il` / `xs`. CBS's 267 dropped units hold 1,097,156 people, which is `xs`'s 723,899 Jews and
     Others plus 373,257 Muslims and Christians exactly. `_il_counts` removes all 267;
     `_xs_counts` stops on any unit not on the list.
  2. `xs` / `ps`. Table 3 is Palestinians only, and PCBS does not enumerate the settlements.
  3. `il` / `ps`, which no assertion covers: Palestinians PCBS counts living in a unit Israel's
     entry keeps. CBS's units were rebuilt uncut with `il_geo.build_units()` and measured
     against OCHA's oPt polygon (scratchpad script, not kept). Jerusalem's 370,471 Muslims
     split 359,478 in dropped units and 10,993 in kept ones. Beit Safafa, which the Green Line
     crosses and PCBS lists in J1, is dropped: no kept Jerusalem unit holds more than 707
     Muslims. Every kept unit with 100 or more Muslims, Christians or Druze has its centroid
     outside COD's governorates. Among the 30 largest, four Jerusalem units reach 5-28% into
     the oPt (3000_1336+1332, 3000_512+511, 3000_521, 3000_1412+1411), with 1,925 Muslims and
     Christians together in near-equal counts, 0.04% of the drawn total. The other 50 kept units
     in that list hold under 290 Muslims and Christians each and were not printed one by one,
     so any overlap with J1 is of that order. The Arab towns Israel keeps along the line (Kafr Qasim, Tayibe,
     Baqa al-Gharbiyye, Jatt, 0.2-4.7% east) are not PCBS localities. Nothing to change.
- **One `note_public` phrase no longer holds.** "Israel's entry on this map leaves out East
  Jerusalem and the West Bank, so each place is drawn once" was written when the settlers were
  on neither entry. With `xs`, East Jerusalem and the West Bank are drawn on two entries (for
  different people), so "each place is drawn once" is now wrong and "so nobody is drawn twice"
  would be right. Not edited here; a note wording fix for the supervisor or Anita.
- **Screenshot** (all-countries view, framed on Palestine): West Bank and Gaza dots on land and
  green, settlement clusters blue inside the West Bank, nothing in the sea, nothing blank.
