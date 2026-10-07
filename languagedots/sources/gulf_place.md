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
zone sex ratios could locate the labour areas.
