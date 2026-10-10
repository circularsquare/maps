# Gulf placement: citizens and foreign residents apart (sa, ae, qa, kw, om, bh)

2026-10-06, session 5d7dac7e-gulf, on Anita's note that the Gulf read as an even rainbow: every
unit's citizens and foreign residents were spread on the same population weight. Placement only
(AGENT_BRIEF §4.4), plus two finer tables that split units. Code: `sources/gulf_osm.py`,
`sources/gulf_place.py`; each `countries/<cc>.py` has `_rows()` and `_weight()`.

```
python sources/gulf_osm.py --fetch      # OSM industrial land + labour camps per country, admin regions
python sources/gulf_place.py --layers   # data/geo/{sa,ae}/<cc>_hexes.gpkg, split units
python sources/gulf_place.py --fit      # the search below
python scatter.py --country <cc>        # sa ae qa kw om bh
```

## 1. The rule

Each hex i of unit u gets a foreign share `logit s_i = a_u + 0.4 ln(density_i) + 10 industrial_i`,
`a_u` solved so the unit's foreign total is exact. Foreign dots weigh `pop_i s_i`, citizen dots
`pop_i (1 - s_i)`; together still population. A row mixing both (Gulf Arabic also carries other
GCC nationals) blends by its citizen share. `industrial_i`: share of the hex under OSM
`landuse=industrial` or a feature named like a labour camp (`gulf_osm.py`; ODbL; 50 camps in the
UAE, under 10 elsewhere, so it is mostly industrial land: SA 2,149 km2, KW 1,717 (oil fields),
AE 1,033, OM 476, QA 365, BH 34).

## 2. The fit

Fitted on the only tables with citizens and foreigners below a drawn parent: Kuwait census 2021
areas within governorates (raw Kontur hexes keyed to religiondots' 143 units, since its own layer
is flat within an area) and Oman's 2024 register wilayat within governorates; hex pop scaled to
each child's count first, so only placement inside the parent is tested. Loss: people-weighted
squared error of each child's foreign share. Best BETA 0.375; GAMMA kept improving to 14, the
largest tried; rounded to 0.4 and 10. A U-shaped term (foreigners also in the sparsest hexes)
was tried and did worse. Errors at 0.4 / 10:

| set | rms error, even spread | rms, fitted | |
|---|---:|---:|---|
| Kuwait areas (fit) | 0.254 | 0.252 | no gain at any BETA |
| Oman wilayat (fit) | 0.129 | 0.089 | r 0.59 |
| Riyadh region's governorates (witness) | 0.082 | 0.001 | Riyadh gov. 52.2% foreign, placed 52.2%; rest 30.5%, placed 30.3% |
| Abu Dhabi's regions (witness) | 0.067 | 0.058 | Al Ain 70.5% foreign, placed 79.0%: density does not find Emirati Al Ain |

**Kuwait is the warning.** Its areas run from 98% foreign (Farwaniya, Jleeb, Hawalli, Salmiya,
20-50k people/km2 on the census) to 31-40% (the Kuwaiti suburbs, 6-12k), but Kontur's 400 m hex
density does not separate them, so density can only be a weak weight. Higher BETA overfits the
Riyadh witness and does nothing for Kuwait. Kuwait and Oman already have fine units; in Kuwait the
layer is flat inside each area, so only industrial land moves anything there.

## 3. Units split by finer tables (counts move between sub-units, not in total)

- **sa**: Riyadh region -> "Riyadh and Ad Diriyah" (3,405,270 Saudis, 3,699,684 non-Saudis) and
  the rest (1,033,940 / 452,854): RCRC open data, census 2022 by citizenship and governorate
  (religiondots' raw copy). Ad Diriyah is not split off alone: OSM's Ad Diriyah polygon holds
  508,635 of Kontur's people against the census's 95,834 (it takes in Riyadh's north-west
  suburbs); together the two are 7,033,099 in Kontur against 7,104,954 counted. Each sub-unit's
  non-Saudis keep the region's language mix.
- **ae**: Abu Dhabi emirate -> its three regions (OSM admin_level 6, relations 13249272-4).
  People at the 2023 census's regional totals (SCAD release via the Abu Dhabi Media Office,
  2,495,925 / 1,009,735 / 284,205) scaled to the drawn 4,135,985; Emiratis (665,515) at SCAD's
  mid-2016 regional split (Statistical Yearbook of Abu Dhabi 2017, Table 3.1.6, Wayback copy in
  `data/raw/ae/scad_syb2016_population.pdf`: 293,860 / 226,285 / 31,390). Result: Abu Dhabi
  Region 87.0% non-Emirati, Al Ain 75.2%, Al Dhafra 87.8% (was 83.9% everywhere). Kontur put
  1.08M in Abu Dhabi Region against 2.72M drawn there, so the split also moves dots onto the
  capital from Al Ain and Al Dhafra.

## 4. Before and after (expected non-citizen share, no dot sampling)

| place | before | after |
|---|---:|---:|
| Riyadh city (40 km) | 48.3% | 52.2% |
| Riyadh region outside the capital | 48.3% | 30.5% |
| Jeddah (30 km) | 48.2% | 57.2% |
| Dammam-Khobar (30 km) | 42.4% | 50.7% |
| Saudi Arabia outside the five big cities | 37.6% | 32.5% |
| Dubai city (25 km) | 91.8% | 93.0% |
| Sharjah city (12 km) | 90.0% | 92.9% |
| Abu Dhabi island + Mussafah (20 km) | 83.9% | 90.3% |
| Al Ain city (15 km) | 83.9% | 79.3% |
| Doha's Industrial Area (4 km) | 90.3% | 95.8% |

People living in hexes 90%+ foreign: SA 0 -> 4%, AE 34 -> 45%, OM 0 -> 6%, BH 0 -> 6%. Kuwait
unchanged (its 143 areas already carry the split). Qatar's municipalities barely move.

## 5. What this does not fix, and the data searched

The UAE's cities stay a rainbow because Dubai and Sharjah are 89-93% foreign and nationality is
published nowhere below the federation (one UN DESA mix for all). Searched 2026-10-06:
- Saudi Arabia: no governorate or city table by citizenship besides RCRC's Riyadh (religiondots
  `sources/sa.md` §2 lists the rest); GLMM's "governorate" table is the 13 regions.
- Dubai Statistics Center: blocked to scripts (religiondots `sources/ae.md` §1); its community
  tables are by sex, not nationality, as far as known.
- Abu Dhabi: SCAD 2016 regions used; the 2023 census release gives regions without citizenship.
- Sharjah 2015 census: not found open in this session.
- Qatar 2020 census: Qataris (10+) by municipality only; zones by sex only.
- Bahrain: data.gov.bh's 45 census 2020 datasets, governorate at finest.
- Kuwait (PACI areas) and Oman (NCSI wilayat): already the drawn units.

Room for improvement: a building-height or villa/apartment layer (GHSL built-up height, OSM
`building=` tags where complete) would find the villa suburbs that Kontur density cannot; Qatar's
zone sex ratios could locate the labour areas (done for qa and ae in §6).

## 6. Families and single men apart (qa, ae; 2026-10-07, session fix-gulf)

Anita, 2026-10-07: the UAE and Qatar still read as one even mix of Indian, Filipino and Arabic
languages everywhere. Placement only (AGENT_BRIEF §4.4): no unit's counts move; checked with
`tools/check_country.py` (ok) and the scatter's dot totals (qa 2,794, ae 11,243, as before).
Code: `node_sex_pools`, `row_pools`, `PoolWeighter`, `fit_sex` in `sources/gulf_place.py`;
`_weight()` in `countries/qa.py` and `countries/ae.py`.

```
python sources/gulf_place.py --fit-sex   # the DELTA / EPS search on Qatar's zones
```

**Rule.** Each nationality's migrants (UN DESA by sex: Qatar 2020, UAE 2024) split into three
pools: *family*, min(men, women) of each sex; *single men*, the men beyond that; *single women*,
the women beyond that (live-in domestic workers). Each (unit, language) row's foreign people are
shared over the pools at its languages' national sex make-up; Qatar's rows at each
municipality's own foreign men and women (census Table 1 less the Qatari estimate), pairs being
min(men x P/M, women x P/W). Then, inside the unit:
- single men: a hex share `logit t_i = b + 10 industrial_i + 0.5 (ln density_i - ln 1000)` of
  the hex's placed foreigners, `b` solved to the pool's total;
- family: the rest of the foreign weight, `foreign_i (1 - t_i)`;
- single women: with households, `citizens_i + family_i`.
- **Qatar** also has census Table 2, 87 zones by sex. Each zone's foreign men and women are its
  men and women less the citizens placed there (at the municipality's Qatari sex ratio); the men
  beyond the women are that zone's single men, and `b` is solved per zone to that count.

**Fit** (`--fit-sex`): single men placed by the hex rule alone within each municipality, against
each zone's own single-men count; foreigner-weighted rms of the zone's single-men share. Even
spread 0.254; DELTA 10 / EPS 0.5 0.137 (best 0.135 at 14 / 1.0, flat past 8; r 0.22). Most of
the gain is the labour zones: Industrial Area (57) 99.6% single men, placed 95.1% (even 52.5%);
Rawdat Rashed (82) 99.9%, placed 90.9%. Among the residential zones the rule hardly beats even.

**Before and after** (expected shares, no dot sampling; South Asian = Indo-Aryan incl. Bengali
and Nepali, plus Dravidian):

| place | South Asian | Bengali | Arabic (not Gulf) | Austronesian |
|---|---|---|---|---|
| Doha Industrial Area (zone 57) | 66.5 -> 71.4% | 13.4 -> 17.4% | 14.7 -> 13.0% | 9.4 -> 6.8% |
| West Bay (zones 61, 66) | 57.1 -> 52.1% | 11.5 -> 7.6% | 12.6 -> 14.2% | 8.1 -> 10.5% |
| Mussafah / ICAD (5 km) | 60.4 -> 70.3% | 14.6 -> 21.0% | 18.3 -> 12.8% | 9.6 -> 7.2% |
| Al Quoz industrial (3 km) | 60.7 -> 70.4% | 14.7 -> 20.9% | 18.5 -> 13.1% | 9.7 -> 7.4% |
| Dubai Marina / JLT (2 km) | 57.4 -> 56.9% | 13.9 -> 13.6% | 17.4 -> 17.7% | 9.1 -> 9.2% |

**Limits.** The contrast is modest, for two reasons. (1) DESA's Qatar split by sex is nearly flat
across origins (most at 2.7-3.1 men per woman; Nepal 3.1, which origin-country permit records
would put far higher), so Qatar's family and single-men mixes differ little; the UAE's split
varies more (Bangladesh 3.9, Indonesia 0.5, Egypt 1.6). (2) Nothing separates nationalities
*among families*: Arab, Indian and Filipino neighbourhoods are real (Dubai's Karama and Bur Dubai
against Al Nahda; Doha's Najma against West Bay), but no table places them, so every residential
district keeps the national family mix.

**What would do more** (not fetched, URLs for Anita):
- Dubai Statistics Center, population by community and sex (2022 estimates, 200-odd
  communities; Statistical Yearbook "Population and Vital Statistics"). The DSC site now forwards
  to https://data.dubai/services (an app; `dsc.gov.ae/Report/...` paths 404 to scripts). With it,
  Dubai gets the zone treatment Qatar has. Drop the Excel in `data/raw/ae/`; community polygons
  also needed (Dubai Municipality's community layer on the same portal).
- Origin-country records of women's share by destination (Nepal DoFE labour permits, Bangladesh
  BMET, Sri Lanka SLBFE, Philippines DMW) would replace DESA's flat Qatar sex split; that moves
  municipality counts by sex, so it is Anita's call, not placement.

**Data note.** `data/normalized/qa.csv` (and possibly `ae.csv`) predates later home-mix changes
(Bangladesh's Chittagonian and Sylheti, Indonesia's Cirebonese, `afroasiatic.arabic.*` ids):
`resolve_pools` matches pool ids to the CSV on the last id component and leaves out the three
nodes the CSVs lack, and a row node the pools lack takes the country's overall sex ratios.
Rebuilding the two CSVs (`sources/qa_build.py`, `sources/ae_build.py`) would pick up the new
mixes; not done here (counts).
