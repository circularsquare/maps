# Iceland: `sources/is.py`, `sources/is_geo.py`, `taxonomy/is2024.py`

Drawn 2026-09-15, session `d743fc47-is`, from sources.md §scout-2026-09-15-europe's row. **2 units,
357,390 people, 99.52% of the 2021 census.** sources.md §is-2026-09-15 is the summary. Hagstofa's
faith-body register is printed beside the map, not drawn (§1).

| | |
|---|---|
| counting geography | **NUTS 3: IS001 Höfuðborgarsvæði (capital area, 7 municipalities) and IS002 Landsbyggð (62)** |
| citizen composition | the same two units; ESS has nothing finer in any round |
| placement | Kontur r8 hexes, each municipality scaled to Hagstofa's 1 January 2021 population (MAN02005) |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout** |
| vintage | ESS rounds 6, 8, 10, 11 (round 6 interviewed 2012-13, round 8 2016-17); census 2021; Pew 2020 |

---

## 1. The office check, which finds a register

Hagstofa Íslands tabulates Registers Iceland's record of the faith or life-stance body each
resident is registered in. PxWeb folder `Samfelag/menning/5_trufelog`, open API, no key:

| table | what | geography |
|---|---|---|
| `MAN10001` | population by registered body, 1998-2026 (61 bodies plus `Önnur trúfélög og ótilgreint` and `Utan trú- og lífsskoðunarfélaga`), with 0-17, 18+, share, fee payers | national only |
| `MAN10289` (2023) back to `MAN10302` (2010) | population by parish, clergy district and deanery: under 16, 16 and over, in and not in the Church of Iceland | 254-345 parishes |
| `MAN10303`-`10310` | Church of Iceland members by parish, 2002-2009 | parish |

That is the Nordic `roll` shape (one church by place, every body only nationally), and like
Norway's (sources/no.md §1) and Denmark's (sources/dk.md §1) it is not drawn: a membership count
beside neighbours drawn from what people say would put a 25-30 point step at the comparison, and it
cannot say who holds no religion. The register is also moving away from belief on its own terms:
`Önnur trúfélög og ótilgreint` grew from 5.33% (2012) to 20.69% (2026). Why it grew was not read;
Hagstofa's table notes were not opened.

Register, share of the population in the table's year (printed by `sources/is.py`):

| | 2012 | 2016 | 2020 | 2021 | 2023 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| Church of Iceland | 77.36% | 72.11% | 64.57% | 63.46% | 59.95% | 56.30% |
| Catholic Church | 3.26% | 3.72% | 4.04% | 3.99% | 3.86% | 3.95% |
| Free Church, Reykjavík | 2.87% | 2.88% | 2.79% | 2.76% | 2.62% | 2.52% |
| Free Church, Hafnarfjörður | 1.86% | 1.97% | 2.02% | 2.04% | 1.98% | 2.01% |
| Óháði söfnuðurinn | 0.99% | 1.00% | 0.90% | 0.89% | 0.83% | 0.74% |
| Ásatrúarfélagið | 0.61% | 0.95% | 1.31% | 1.40% | 1.50% | 1.56% |
| Siðmennt (humanist) | 0.00% | 0.44% | 0.98% | 1.13% | 1.43% | 1.62% |
| other bodies and unspecified | 5.33% | 7.44% | 13.04% | 13.61% | 17.10% | 20.69% |
| outside every body | 4.89% | 5.71% | 7.19% | 7.62% | 7.78% | 7.67% |

Þjóðskrá (Registers Iceland) was not searched beyond the scout's note (monthly national counts).

## 2. Rounds, region, and round 9

Probed 2026-09-15 against every integrated file from round 2 to 11:

| round | Iceland | `region` | card |
|---|---|---|---|
| 2 | 579 | `regionis`, one value (Iceland) | not probed |
| 3, 4, 5, 7 | absent | | |
| 6 | 752 | IS001 469, IS002 283 | `rlgdnis` |
| 8 | 880 | IS001 522, IS002 358 | `rlgdnis` |
| 9 | 861 | **IS001 684, IS002 177** | `rlgdnais` |
| 10 | 903 | IS001 551, IS002 350, 2 `Not available` | `rlgdnais` |
| 11 | 842 | IS001 560, IS002 282 | `rlgdnais` |

**Round 9's `region` is not the NUTS 3 split.** Domicile by region (unweighted, all respondents):

| round | IS001 share of sample | IS001 big city or suburbs | IS002 town | IS002 village or farm |
|---|---:|---:|---:|---:|
| 6 | 62.3% | 61.5% | 47.7% | 45.9% |
| 8 | 59.2% | 58.2% | 61.5% | 34.9% |
| **9** | **79.4%** | **44.8%** | **21.5%** | **77.4%** |
| 10 | 61.2% | 61.6% | 56.4% | 37.8% |
| 11 | 66.5% | 56.7% | 57.7% | 36.7% |

The capital area is 64.2% of the census. In round 9 the towns outside it (Akureyri, Reykjanesbær,
Selfoss and the rest) were coded IS001 and IS002 holds villages and farms. Weighted, round 9's IS001
is 82.6%, so `pspwght` does not correct it. With two codes there is nothing to recode (Denmark's
round 9 was a permutation of five, sources/dk.md §3), so round 9 is in no pool. `_check_round9`
asserts every pooled round has IS001 at least 50% big city or suburbs and IS002 at least 40% town,
and that round 9 as published fails, so the build stops if ESS reissues the file. No ESS release
note was read on it.

Round 9's national shares are printed beside the others and do not stand out (Church of Iceland
35.10%, no religion 60.10%).

## 3. The cards

`rlgdnis` (rounds 6, 8) and `rlgdnais` (9-11) are Iceland's own. Each round's cross-tab against the
harmonised `rlgdnm`, all respondents, unweighted:

| answer | r6 | r8 | r9 | r10 | r11 | `rlgdnm` |
|---|---:|---:|---:|---:|---:|---|
| Þjóðkirkjunni (Church of Iceland) | 277 | 359 | 333 | 344 | 292 | Protestant |
| Fríkirkjunni (the Free Church) | 11 | 13 | 14 | 14 | 16 | Protestant |
| Öðru kristnu trúfélagi innan lúthersku (other Lutheran body) | 16 | 20 | 13 | 12 | 9 | Protestant |
| Kaþólsku kirkjunni | 5 | 8 | 12 | 24 | 28 | Roman Catholic |
| Rússnesku rétttrúnaðarkirkjunni (6, 8) / Rétttrúnaðarkirkju (9-11) | 1 | 1 | 0 | 3 | 1 | Eastern Orthodox |
| Öðru trúfélagi utan lúthersku (other body, not Lutheran) | 1 | 0 | 0 | 5 | 5 | **r6 Eastern Orthodox**, r10-11 Other Christian |
| Félags múslima á Íslandi (6, 8) / Íslamstrú (9-11) | 1 | 1 | 1 | 0 | 4 | Islam |
| Austrænum trúarbrögðum | 4 | 4 | 2 | 2 | 2 | Eastern religions |
| Ásatrúarfélaginu | 3 | 5 | 6 | 10 | 7 | Other Non-Christian |
| Öðrum trúarbrögðum utan kristni | 4 | 1 | 3 | 4 | 6 | Other Non-Christian |
| Öðrum (6, 8 only) | 1 | 1 | | | | Other Non-Christian |

Two answers changed label and are harmonised to the later one (`HARMONISE`). **Round 6's one
`Öðru trúfélagi utan lúthersku` is in `rlgdnm` Eastern Orthodox**, where rounds 10-11 put all ten in
Other Christian: ESS's recode, not the respondent's answer, so the Icelandic label is what is mapped
and `CARD_NEST` allows both for that answer. `rlgdnis` was not read for its unused codes 6, 8, 10.

## 4. Which answers carry the two units

**3,222 answered citizens** in rounds 6, 8, 10, 11 (IS001 2,005, IS002 1,217); 0.53% declined
(weighted).

**The rank split-half has nothing to rank on two units.** It is run (`no.py::_stability`) and
printed: every median rho is +1 or -1 and every p is 0.48 or more, whatever the chi-square.

**The deciding test is Uzbekistan's two-unit test** (spec §12, `uz.py::two_unit_test`): the absolute
difference in a category's unweighted share between the units, against 2,000 draws with each round's
region labels shuffled among its respondents, and the 2 x 2 chi-square, both at 0.05. Uzbekistan
shuffles sampling points; Iceland shuffles respondents, because its ESS sample is nearly unclustered:
the ESS8 Sample Design Data File user guide (Lynn, 2019, Table 1) gives Iceland's 880 respondents
705 PSUs in 4 strata. Rounds 6 and 10-11 were not checked for design.

| answer | n | IS001 | IS002 | perm p | chi² p | same side | drawn at |
|---|---:|---:|---:|---:|---:|---|---|
| No religion | 1,749 | 56.11% | 51.27% | 0.0060 | 8.4e-03 | 3 of 4 | residual (would pass) |
| **Þjóðkirkjunni** | 1,262 | 36.01% | 44.37% | 0.0005 | 2.9e-06 | 3 of 4 | **two units** |
| Öðru kristnu trúfélagi innan lúthersku | 53 | 1.95% | 1.15% | 0.0835 | 1.2e-01 | 3 of 4 | residual |
| **Fríkirkjunni** | 54 | 2.49% | 0.33% | 0.0005 | 6.8e-06 | 4 of 4 | **two units** |
| Kaþólsku kirkjunni | 33 | 0.95% | 1.15% | 0.5802 | 7.1e-01 | 2 of 4 | residual |
| Ásatrúarfélaginu | 24 | 0.75% | 0.74% | 1.0000 | 1.0e+00 | 2 of 4 | residual |
| Öðrum trúarbrögðum utan kristni | 14 | 0.55% | 0.25% | 0.2879 | 3.2e-01 | 3 of 4 | residual |
| Austrænum trúarbrögðum | 11 | 0.50% | 0.08% | 0.0665 | 9.8e-02 | 4 of 4 | residual |
| Öðru trúfélagi utan lúthersku | 10 | 0.25% | 0.41% | 0.5267 | 6.4e-01 | 2 of 4 | residual |
| Rétttrúnaðarkirkju | 5 | 0.20% | 0.08% | 0.6787 | 7.2e-01 | 3 of 4 | residual |
| Íslamstrú | 5 | 0.15% | 0.16% | 1.0000 | 1.0e+00 | 1 of 4 | residual |
| Öðrum | 2 | 0.10% | 0.00% | 0.5312 | 7.1e-01 | 2 of 4 | residual |

The Church of Iceland is level in round 6 (37.9% against 37.3%) and apart in 8, 10 and 11. The Free
Church is apart in every round and sits where its congregations are (Reykjavík, Hafnarfjörður).

**The witness is the church's own parish roll**, summed to the two units (`_register_parish`).
Deaneries 01 and 02 are all capital area, 04-09 all outside it, and deanery 03 (Kjalarnes) is split
by parish name into Hafnarfjörður, Garðabær, Mosfellsbær, Kjalarnes and Kjós against Grindavík,
Suðurnesjabær, Reykjanesbær and Vogar. The assignment is checked by population, which the church
counts do not determine: the parishes put the capital area at 64.04% of Iceland (1 December 2021)
and 63.65% (2023), against the census's 64.23%. Members of those 16 and over: **capital area 56.8%,
the rest 69.2%** (2021); 53.0% and 64.2% (2023). The survey orders the two the same way.

**No rescale.** The Church of Iceland is 34.80% of the pool and 32.87% in rounds 10-11, 1.92
points, under spec §12's 3.5 (ask 022's rule). Catholic moved +0.62 and is in the residual.

## 5. Geography

**Units by code.** GISCO LAU 2021 has 69 Icelandic municipalities with Hagstofa's numbers; the first
digit is the region (0, 1 the capital area). `CAPITAL` lists the seven codes with their names and
the build asserts the digit rule agrees. **Witness:** Hagstofa MAN02005 (1 January 2021, on the 2026
municipality map, seven 2021 municipalities merged forward by `MERGE_2026`, asserted to give exactly
its 62 codes with no merge crossing a region digit) summed by that rule against Eurostat's own NUTS 3
totals: IS001 229,647 against 230,657, IS002 128,651 against 128,465, relative 0.998 and 1.004. The
LAU shapefile's `POP_2021` is zero for Iceland.

**Kontur alone put Landsbyggð's dots on summer houses.** 22,176 hexes, 375,249 people; 792 hexes
with centres just offshore snapped within 500 m (17,007 people); 54 hexes, 203 people, in no
municipality. Uncalibrated, Kontur holds the capital area at 0.892 of its census share and
Landsbyggð at 1.193. Per municipality against the 2021 register, over the national ratio:

| municipality | Kontur | register 2021 | relative |
|---|---:|---:|---:|
| Skorradalshreppur | 600 | 60 | 9.55 |
| Grímsnes- og Grafningshreppur | 3,377 | 474 | 6.81 |
| Árneshreppur | 254 | 41 | 5.92 |
| Kjósarhreppur | 1,073 | 235 | 4.36 |
| Dalabyggð | 2,038 | 607 | 3.21 |
| Bláskógabyggð | 3,433 | 1,062 | 3.09 |
| ... | | | |
| Reykjanesbær | 16,106 | 18,751 | 0.82 |
| Mosfellsbær | 10,426 | 12,341 | 0.81 |
| Vestmannaeyjabær | 3,405 | 4,248 | 0.77 |
| Seltjarnarnesbær | 3,496 | 4,590 | 0.73 |

33,165 more people than the register in the municipalities Kontur over-reads outside the capital
area. The cottage districts of the south and west are the top of the list. **So each hex keeps
Kontur's share of its municipality and the municipality takes the register's count** (factors 0.100
Skorradalshreppur to 1.313 Seltjarnarnes), the Faroes' and Iran's construction at municipality
level. Inside a municipality the summer-house hexes still take Kontur's share; Hagstofa's
settlement tables (`2_byggdir/Byggdakjarnar`) would fix that and were not read.

`kontur_cap.py is`: no stops. Hexes are clipped to their unit where the centre was inside (2,156
edge hexes). The grid is `is_grid_400m.gpkg` because `kontur_cap.py` only recognises
`*_grid_<n>m` and `*_hexes` names.

## 6. The mapping, the arguable calls

Full reasons in `taxonomy/is2024.py::REVIEW`.

- **Church of Iceland, the Free Church and "other Lutheran body" all to `christianity.lutheran`.**
  The Free Churches are Lutheran congregations outside the national church; "innan lúthersku" is the
  card's own word. A Pentecostal who read the third box loosely is misfiled there.
- **Ásatrú to `paganism`**, not `other.is`: its own box, 1.6% of the register, and `paganism` is the
  node for reconstructed pre-Christian religion. No Iceland-only node.
- **Islam to `islam.sunni`**: the register's Muslim bodies name no Shia body.
- **Orthodox unsplit**: the register's Orthodox bodies are 98.0% Eastern.
- **`other.is` is new** (branches.py): Eastern religions, other non-Christian and other, 27 citizens.

## 7. What the finished country looks like

| unit | Lutheran | unaffiliated | Catholic | Islam | Orthodox | pagan | foreign citizens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Höfuðborgarsvæði | 30.80% | 54.62% | 8.48% | 1.11% | 1.39% | 0.88% | 13.11% |
| Landsbyggð | 36.33% | 49.24% | 9.35% | 0.66% | 1.33% | 0.79% | 13.11% |

Nationally: unaffiliated 52.69%, Lutheran 32.78%, Latin Catholic 8.75% (19,267 Polish citizens in
the foreign half), `christianity.protestant` 1.19% (foreign half), paganism 0.85%.

The foreign share is 13.11% in both units, which is arithmetic, not a bug (30,237 of 230,657; 16,836
of 128,465).

## 8. What the build cannot do

- **Two units.** Akureyri and Grímsey draw the same citizen composition.
- **Everything except the Church of Iceland and the Free Church is at the national rate** inside each
  unit's remainder among citizens, Catholics and Ásatrú included.
- **Round 9 is out** and rounds 10-11 have no interview-year variable under `inwyys`, `inwyr` or
  `inwyye`, so `how` says "from 2012" and not an end year.
- **The citizen half is 2012-2024 pooled; the population is the 2021 census.** Iceland has grown
  about 10% since (register 358,298 in 2021, 394,324 in 2026), mostly by immigration.

## 9. Numbers to check a rebuild against

```
ESS: Iceland in rounds 2, 6, 8, 9, 10, 11; region NUTS 3 (IS001, IS002) in 6-11; round 9 region fails domicile
pooled rounds 6, 8, 10, 11; 3,222 answered citizens (IS001 2,005, IS002 1,217); 0.53% declined (weighted)
two-unit test passes Þjóðkirkjunni and Fríkirkjunni; drift 1.92 points, no rescale
cens_21ctz_r3: 359,122 = 311,972 NAT + 47,073 FOR + 48 STLS + 29 UNK
drawn 357,390, 99.52%; citizen half 310,317; foreign half 47,073
unaffiliated 52.69%  lutheran 32.78%  catholic.latin 8.75%  paganism 0.85%
MAN02005 1 Jan 2021: 358,298 over 62 municipalities; IS001 229,647
Kontur IS 2023-11-01: 22,176 hexes, 375,249 people; calibrated grid 22,122 hexes
parish roll, members of 16+: 2021 capital 56.8% rest 69.2%; 2023 53.0% and 64.2%
scatter 1:1,000: 349 dots, 42 rings; 1:10,000: 32 dots, 48 rings; kontur_cap.py is: no stops
```

## 10. Not checked

- Hagstofa's register table notes (why `ótilgreint` grew), Þjóðskrá's own releases.
- EVS 2017 Iceland (GESIS login), ISSP Iceland, Gallup Iceland's religion polls.
- ESS round 2's card; rounds 10-11's fieldwork dates; round 6, 10, 11 design files.
- Hagstofa's settlement tables for placement below municipality.

## 11. Review, 2026-09-15 (`d743fc47-rev13`)

Full pass. `check_md.py` clean, `built_countries.py --check` ok, rollup clean (357,390, all
`modelled`). Every `note_public` figure, the grain and `gap_share` recompute off `is.csv`,
`is_foreign.csv` and §4: Church of Iceland 31.6% of citizens in the capital area and 40.0% outside
it, the Free Church 2.2% and 0.3%, 310,316 citizens plus 47,073 foreign, 1,732 of 359,122 = 0.48%.

- **Round 9 left out: agreed.** §2's domicile table makes the case (IS002 77% village or farm,
  IS001 82.6% weighted against the census's 64.2%), and round 9's national shares sit among the
  others, so leaving it out costs sample and nothing else.
- **The two-unit test: agreed.** `uz.py::two_unit_test` exists and spec cites it. Uzbekistan
  shuffles sampling points and this shuffles respondents; the Church of Iceland and the Free Church
  pass at the permutation floor (0.0005) with chi-square p of 3e-6 and 7e-6, so clustering in the
  rounds whose design file was not read could not flip either.
- **Mapping: agreed**, box for box with `dk2024.py` and `no2024.py`; Ásatrú on `paganism` and
  `other.is` are routine.
- **`coverage.py`: nothing else moved.** HEAD's `coverage()` against the working tree's, all 181
  countries: only `is` (+43 nodes) and `ma` (+44, Morocco's own line and comment) differ, and only
  by additions. `python coverage.py` passes, 3,108 pairs.
- **One wording point, not edited.** The note says every other citizen answer is divided in national
  proportions "so the map says nothing about where those citizens live". That includes no religion,
  which passes the two-unit test on its own (§4) and is drawn at 60.6% of citizens in the capital
  area and 54.7% outside it through the remainder, the same order as the survey. "Every other
  answer except no religion" would be exact.
- **Catholic level, for the record.** Latin Catholic is drawn at 8.75% of Iceland, 27,173 of its
  31,273 people from the foreign half at Pew's shares for their nationality; the Catholic Church
  was 3.99% of the register in 2021. The register's "other bodies and unspecified" line (13.61%)
  may hold registered immigrants, which was not checked, so the register does not settle the
  level. Norway, Denmark and Sweden are built the same way, so this is not an Iceland defect.

Screenshot clean: Reykjavík, Keflavík, Selfoss and Akureyri, nothing offshore.
