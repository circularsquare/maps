# The Gambia — 2013 Population and Housing Census, religion by Local Government Area

**Drawn 2026-09-15** (session `d743fc47-gm`). 8 LGAs, 4 categories, 1,856,211 of 1,857,181 people
drawn (the 970 left blank are not), every row `measured`. 1,854 dots at 1:1,000, 184 at 1:10,000.

- `sources/gm.py` -> `data/normalized/gm.csv` (the PDF is
  `data/raw/gm/census_2013_spatial_distribution_report.pdf`, pinned)
- `sources/gm_geo.py` -> `data/geo/gm/gm_lgas.gpkg`, `gm_hexes.gpkg`, `gm_lookup.csv`
  (COD-AB v01 ADM1, geoBoundaries ADM1 as the join witness, Kontur 400 m)
- `taxonomy/gm2013.py` -> the mapping; `countries/gm.py` -> the entry; `taxonomy/branches.py`
  `other.gm` is the one new node
- sources.md **§gm-2026-09-15** is the summary; **§11aq** was the scout's row.

```
python sources/gm.py     --fetch
python sources/gm_geo.py --fetch
```

## 1. What the Gambia Bureau of Statistics publishes

| release | religion | tier |
|---|---|---|
| **Census 2013, *Spatial Distribution Report*** (373 pp) | **Annex H, Tables H.1-H.63 (PDF pp.119-181): age x religion for the nation and each LGA, both sexes, male, female, and urban and rural where an LGA has them, in counts**; section 2.1.4 (p.16) national shares | **8 LGAs** |
| Census 2013, *Preliminary Results* (IHSN study 6065, document 74258, 23 pp) | none | |
| Census 2024, *Preliminary Report* (GBoS; UNFPA Gambia's copy) | none; Table 12, population by sex and LGA, is used here as a Kontur witness | |
| UNSD Demographic Yearbook table 28 | absent | |
| `microdata.gbosdata.org` | timed out (scout, 2026-09-14) | |

**Not opened this session, so not a negative.** gbosdata.org lists fifteen 2013 volumes
(`/downloads/158-2013-population-and-housing-census`, read 2026-09-15): Access to ICT; Children;
Compounds and Buildings Structures; Directory of Settlement; National Disability; Economic
Characteristics; Education Characteristics; Fertility; Gender; Housing and Household
Characteristics; National Migration; Mortality; Elderly; Youth; and the Spatial Distribution Report.
None is titled for religion or culture. A religion table by district (COD-AB's 49 ADM2 units)
would most plausibly sit in an annex of the Gender, Children or Youth reports, which cross social
characteristics. Whether the 2024 census asked religion was not checked: its preliminary report
mentions religion only as a building use (scout).

## 2. The construction

No arithmetic beyond addition. Each LGA's four religion cells and `Not stated` are the Total row of
its both-sexes table: H.4 Banjul, H.7 Kanifing, H.10 Brikama, H.19 Mansakonko, H.37 Kuntaur, H.46
Janjanbureh, H.55 Basse. **Kerewan is H.29 + H.30** (male plus female), because H.28 is a misprint
(§3). The tables cover the whole population, 1,857,181, which is Table B.1's 2013 total: Form B
enumerated group quarters and the floating population with the same four religion codes, and the
annex does not separate them.

## 3. The checks (`sources/gm.py::check`)

| check | result |
|---|---|
| PDF pinned | gbosdata.org 2026-09-15, digest `R4BTYNYMIW5GCJ426JRHKDZFVVWKS43P`, 8,702,036 bytes, 373 pages; four Wayback captures 2024-03-13 to 2026-01-11 share the digest |
| all 63 tables parse | every row sums to its Total and every column's age rows to its Total row |
| both sexes = male + female, cell by cell | all 21 triples except H.28 |
| **H.28 (Kerewan both sexes)** | **equals H.31 (Kerewan urban) in every cell**, 50,188 people; pinned |
| LGA = urban + rural, cell by cell | all six LGAs with rural parts, both sexes, male and female (Kerewan both sexes from H.29 + H.30) |
| Kerewan | H.29 + H.30 = H.31 + H.34 = 220,080, cell by cell |
| eight LGAs sum to the nation | H.1 in all 20 rows x 6 columns |
| Annex B | every LGA total = Table B.1 and B.3's 2013 population; B.3's areas sum to its 10,679.28 km2 |
| section 2.1.4's 96.0 / 3.8 / 0.1 / 0.1 per cent | shares of the 1,856,211 who stated a religion, not of the whole population (Christians are 3.750% of everyone, 3.752% of those who stated) |

The text layer's quirks, each pinned so a different file announces itself: H.18 is captioned
*Brikama-Rural-Male* and holds the female figures (H.17 + H.18 = H.16); H.39 prints the age label
`10-15` and H.54 `4-9`; H.33, H.40-42 and H.49-51 print no `Not stated` age row; H.38, H.40-42 and
H.44 (Kuntaur male, urban and rural male) print no `Traditional` column; five headers spell
`Tradition`; H.6 breaks `Age group` over two lines. Missing rows and columns read as zero, and the
identities above confirm it. H.28 and H.31 were rendered and are the same table in print.

## 4. The questionnaire and the codes

Form A (household) Part 2, column 7, *What is your Religion?* (IHSN study 6065, document 74247,
rendered and read): **1 Islam, 2 Christianity, 3 Traditional, 4 Other.** Form B (group quarters and
floating population, document 74248) has the same four. The enumerator's manual (document 74250,
para 8.38): "Record the appropriate code number for the religion professed by the respondent. There
is no need to probe to ascertain the authenticity of the claim... 'Traditional' refers to the
traditional African religion and is indigenous to the respondent. For those claiming to belong to
religions that fall under other categories, record code 4 for other religions and specify, e.g.
Hindu."

So there is **no code for no religion and none for no answer**. `Not stated` (970) is a blank field;
anyone who said they had no religion went to code 4 or was left blank, and nothing measures which.
The no-religion rule of 2026-09-14 has no box to apply to (`tools/check_no_religion.py` lists no
Gambian row). No church, no Muslim order and no Ahmadiyya code exists, so Islam and Christianity
are drawn bare.

**`Other` holds a block.** 1,106 of Kanifing's 1,996 `Other` are in its age-not-stated row, beside 37
Muslims and 53 Christians and nobody else; Kanifing's other age rows hold 890. They look like one
batch of records with neither age nor a named religion (an institution, or forms coded wrong), not
a congregation. Drawn as printed; `taxonomy/gm2013.py` REVIEW says so. Brikama's age-not-stated row
has the matching shape for `Not stated` (191 of its 447).

## 5. Geography

**COD-AB Gambia v01** (`cod-ab-gmb`, valid from 2022-09-01, reviewed 2025-10-30, source NDMA) has
the eight LGAs as ADM1 under region names; `gm_geo.py` maps them by pcode (GM01 Banjul City Council,
GM05 Kanifing Municipal Council, GM03 West Coast = Brikama, GM08 Lower River = Mansakonko, GM06
North Bank = Kerewan, GM07 Central River North = Kuntaur, GM04 Central River South = Janjanbureh,
GM02 Upper River = Basse). **geoBoundaries gbOpen GMB ADM1** (World Bank, 2020) names the same eight
by LGA and is the witness that the pcode mapping does not decide: each COD unit overlaps its
LGA-named feature best (IoU 0.974 to 0.998 for the six rural LGAs, Banjul 0.995, Kanifing 0.564,
nothing else above 0.000).

| LGA | Table B.3 km2 | COD | geoBoundaries | Kontur 2023 / 2024 count / national ratio |
|---|---:|---:|---:|---:|
| Banjul | 12.23 | 9.3 | 9.4 | 1.20 |
| Kanifing | 75.55 | **93.7** | **52.9** | 1.09 |
| Brikama | 1,764.25 | 1,761.7 | 1,764.4 | 0.92 |
| Mansakonko | 1,608.00 | 1,573.0 | 1,539.0 | 0.99 |
| Kerewan | 2,255.50 | 2,253.0 | 2,208.0 | 1.11 |
| Kuntaur | 1,466.50 | 1,520.6 | 1,494.1 | 0.98 |
| Janjanbureh | 1,427.75 | 1,509.4 | 1,472.7 | 0.95 |
| Basse | 2,069.50 | 2,049.1 | 2,027.7 | 1.10 |

**The two files disagree on Kanifing, and it moves nobody.** The census prints 75.55 km2; COD draws
93.7 and geoBoundaries 52.9, on a line through the built-up land between Serekunda and Brikama.
Kontur holds 479,694 people in COD's Kanifing and 479,679 in geoBoundaries', so the 40 km2 between
them is empty (wetland), and COD is drawn. Kontur is compared with the 2024 preliminary count
(Table 12: Banjul 26,461, Kanifing 379,348, Brikama 1,151,128, Mansakonko 90,624, Kerewan 248,475,
Kuntaur 118,104, Janjanbureh 147,412, Basse 261,160; 2,422,712), the nearest count to its date:
**Kontur is 1.161x that count nationally**, and every LGA is 0.92 to 1.20 of the national ratio.
Banjul's 1.20 fits a city that has lost people every census since 1983.

**Placement.** Kontur GM 2023-11, 6,135 hexes, 2,811,983 people; on COD, 43,541 people snapped to an
LGA within 500 m and 15,039 (0.53%) dropped, which is hexes whose centroids fall in Senegal. 6,078
hexes placed. `kontur_cap.py gm` found no block at the cap. `scatter.py`: 79 hexes clipped by the sea
(0.55% of their area), 2 left unclipped. Median LGA about 1,500 km2, so no grid-floor concern.

## 6. What the table shows

Islam is 96.0% of everyone. Christianity is 7.70% of Kanifing, 4.84% of Banjul and 4.82% of Brikama,
and those three hold 91.5% of the country's Christians (Brikama 47.6%, Kanifing 41.7%); Kerewan is
1.26%, and Kuntaur, Mansakonko, Janjanbureh and Basse are 0.48-0.70%. Traditional religion is 1,028
people: 602 in Brikama, 309 in Kanifing, and no more than 36 in any other LGA. It is a floor, one
code per person. `Other` is 0.53% of Kanifing (0.24% without the block in §4), 0.21% of Banjul and
0.08% of Basse.

Against the State Department's 2023 religious freedom report (Muslims about 96.4%, Christians about
3.5%, "the majority of whom are Roman Catholics", Ahmadis claiming about 50,000), the census level
agrees and says nothing about churches or the Ahmadiyya.

## 7. §14 was considered and no ask was filed

The State Department's *2023 Report on International Religious Freedom: The Gambia* (read via
ecoi.net document 2111867) records "tensions between Muslims and Christians in Bakau and Tallinding
stemming from reported vandalism of churches", from media accounts, and that the Supreme Islamic
Council says Ahmadis do not belong to Islam and has barred them from Muslim cemeteries since 2015.
Both places are inside Kanifing. Not escalated: the map draws Christians at the LGA, Kanifing is
377,134 people, the figure is GBoS's own publication (on gbosdata.org since at least March 2024 by the
Wayback CDX, and archived from the old `gbos.gov.gm` by January 2018 according to the scout), and nothing is placed more finely than GBoS already prints; the report
describes vandalism and tension, not attacks on people. The Ahmadiyya has no code, so nothing
locates Ahmadis. This follows asks 017 and 018 (units that big were cleared in Chad, Burkina Faso and
Mali); a session that finds attacks on people should reread it.

## 8. Gotchas

- **H.28 reprints H.31 and closes on itself.** Every row and column adds up; only the sex and
  urban-rural identities, or Table B.1's 220,080, find it.
- **Section 2.1.4's shares are of people who stated a religion.** Christians are 3.8% there and 3.75%
  of everyone.
- **Kuntaur's male and urban tables drop the `Traditional` column** rather than print zeros, and seven
  small tables drop the `Not stated` age row. A parser that expects six columns reads the wrong
  cells without failing an in-table check if the column count is not taken from each header.
- **COD-AB names the LGAs by region**; Central River North is Kuntaur and South is Janjanbureh.
- **COD-AB's Banjul is 9.3 km2 against the census's 12.23**, and geoBoundaries agrees with COD
  (IoU 0.995), so the census area probably includes water.

## 9. Reopen when

- A 2013 volume (§1's unopened list) turns out to print religion by district, or the 2013
  microdata opens: COD-AB's 49 districts would be the next tier.
- The 2024 census publishes religion (whether it asked was not checked).
- Anything splits Christians by church or Muslims by Ahmadiyya below the nation (§14 would then
  need a new look, starting from §7).

## 10. Terms

GBoS's reports are public PDFs on its own site with no licence text. OCHA COD-AB is CC BY-IGO;
geoBoundaries gbOpen GMB is CC BY 4.0 (World Bank data catalog 0038034); Kontur Population is CC BY
4.0. The IHSN questionnaires and manual were read, not redistributed.

## 11. Review, 2026-09-15 (session `d743fc47-rev10`, light pass)

- **Checks.** `check_md.py` clean, both editions present, `check_rollup.py gm` all measured with
  nothing orphaned. The H.28 reprint and the Kerewan rebuild were not re-parsed; the identities in §3
  close on Table B.1, which is the check that would catch a wrong rebuild.
- **Mapping agreed.** Four answers mapped as `sl2015` maps the same four (Islam and Christianity
  bare, Traditional on `indigenous.african`, Other on the country's residual). `other.gm` is routine.
- **The 1,106-record block drawn on `other.gm` is agreed.** It is about one dot at 1:1,000, the node
  note says what it looks like, and nothing says it is a non-answer rather than code 4.
- **§14: the decision not to file an ask holds.** The State Department report describes vandalised
  churches and tension in Bakau and Tallinding, not attacks on people. Christians are drawn only at
  the LGA, GBoS's own grain; Kanifing is 377,134 people, and dots inside it follow Kontur, not where
  Christians live, so neither town is picked out. Asks 017 (Chad: killings, drawn at 22 régions of
  about 500,000) and 018 (Burkina Faso and Mali: drawn at provinces and régions of similar size) were
  ruled on worse facts at units of the same order. Ahmadis have no code and are not located. §9's
  reopen line (a district tier, or a split of Christians or Ahmadis) is the right trigger.
- **Screenshot.** Dots on land, densest in the Kombos, thin along the river; none in the sea or over
  Senegal.
