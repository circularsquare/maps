# Luxembourg (`lu`): record

Built 2026-10-03 by `fafd1067-lu`. Drawn: 643,941 people (census, 8 November 2021) at one national
mix on the census's 102 communes, every row `modelled`. Code: `sources/lu.py`, `sources/lu_geo.py`,
`taxonomy/lu2021.py`, `countries/lu.py`. Section in `sources.md`: §lu-2026-10-03.

## 1. What was looked for, and what came back

No census has asked religion since 1970 (AHA's 2022 release: the census has been barred from asking
since 1979; STATEC, Regards 03/23: "n'a actuellement pas de mandat"). Searched 2026-10-03:

| source | years | covers foreigners | place below the country | status |
|---|---|---|---|---|
| ESS rounds 1-2 (API, `api.nsd.no`) | 2002-04 | yes, 30.5% non-citizens sampled (census 2001: 36.9%) | **no**: `regionlu` has the one value `Luxembourg` in both rounds; `region` does not exist (E201) | witness |
| ESS rounds 3-11 | | | | Luxembourg absent (probed every main file) |
| EVS 1999, 2008 | 1999, 2008 | yes ("population résidante", 1,610 adults in 2008) | unknown: GESIS login, not opened | national 2008 shares printed in CEPS/INSTEAD cahier 2011-02 Tableau 1: Catholic 68.7, Protestant 1.8, other Christian 1.9, other non-Christian 2.6, none 24.9 |
| EVS 2020/21 (STUDIALUX, University of Luxembourg; fielded by TNS Ilres) | late 2020 to early 2021 | yes, weighted to the resident population | none published | **not in the EVS 2017 integrated release nor the joint EVS/WVS v5.0** (its 92-country list, `worldvaluessurvey.org/documents/countriesJointv5_0.png`, has no Luxembourg). National figures in STATEC Regards 03/23 (`data/raw/lu/statec_regards_03_23.pdf`). Used. |
| TNS Ilres for AHA | 9-18 March 2022 | yes, 515 residents 16+, online panel and phone | none published | national table in AHA's release of 21 June 2022 (`data/raw/lu/aha_umfrage_2022.pdf`). Used. |
| ISSP | | | | not opened: search results list 48 members without Luxembourg; issp.org's members page names no countries inline. Not checked: the members PDF |
| Eurobarometer | | EU residents | Luxembourg is one NUTS 2 | GESIS login; national only, not opened |
| CEPS/INSTEAD 2011 cahier | 2008 | | | religiosity by nationality only, no affiliation by nationality or region |
| Vatican / archdiocese | 2022 | | | 271,000 Catholics, 41% (a diocesan estimate, not self-identification; via search results, not read) |

STATEC itself warns about EVS 2020/21: "la qualité de l'enquête EVS de 2021 que nous avons utilisée
laisse à désirer malgré les repondérations ... La taille réduite de l'échantillon ne permet pas
d'analyses détaillées" (p. 5). It does not print the sample size.

## 2. What is drawn

Each survey normalised to 100 over the five categories both print, then the mean:

| | EVS 2020/21 | TNS Ilres 2022 | drawn |
|---|---:|---:|---:|
| Catholic | 40.94 | 51.26 | **46.10** |
| Protestant (EVS: Protestants + evangelicals) | 3.22 | 1.93 | 2.58 |
| Muslim | 1.30 | 2.90 | 2.10 |
| other religion | 2.54 | 2.90 | 2.72 |
| no religion | 52.00 | 41.00 | **46.50** |

EVS: 48% belong (Graphique 4); of them Catholic 85.3%, Christian 92%, Muslim 2.7% (p. 2); the text's
41% Catholic and 44% Christian of everyone are reproduced and asserted. Graphique 5's Jewish,
Buddhist and other bars carry no numbers, so they are the remainder (5.3% of those who belong).
TNS: belong 59 (Catholic 53, Muslim 3, Protestant 2, other 3, which sum to 61 by rounding and are
scaled to 59); none 41 (23 former Catholics, 1 former other, 18 never).

Every commune takes these shares; children are drawn at the adults' shares; nobody is left out
(stateless and not-stated citizenship are residents both surveys sample). No `gap`.

## 3. Why not the foreign half

The brief's route for a country whose survey misses foreigners. Both late surveys sample residents
of every nationality, so it is not needed for coverage; it was still built, because nationality is
the only thing that varies by commune and it would have given the map a geography. It fails
Luxembourg's own totals (`sources/lu.py::_foreign_half_test`, printed every build, pinned):

| | target, everyone | foreigners on Pew origins | citizens would need |
|---|---:|---:|---:|
| Catholic | 46.10 | 49.64 | 42.96 |
| Protestant | 2.58 | 7.69 | **-2.00** |
| Muslim | 2.10 | 9.40 | **-4.42** |
| other religion | 2.72 | 10.79 | **-4.49** |
| no religion | 46.50 | 22.48 | 68.00 |

The foreign half's Muslims alone are 4.43% of the country; Pew's own 2020 estimate for Luxembourg
is 1.83%. ESS 2002-04 shows where Pew's origin rows go wrong (Pew 2020 vs ESS respondents of that
nationality living in Luxembourg, %):

| | n | Christian | none | Muslim |
|---|---:|---|---|---|
| Portugal | 357 | 83.2 vs 85.1 | 16.0 vs 13.8 | 0.0 vs 0.4 |
| France | 122 | 48.4 vs 46.5 | 46.7 vs 42.6 | 2.5 vs 9.1 |
| Italy | 109 | 80.7 vs 80.5 | 14.7 vs 13.3 | 0.0 vs 4.4 |
| Belgium | 95 | 56.8 vs 51.0 | 41.1 vs 39.0 | 2.1 vs 6.8 |
| Germany | 50 | 62.0 vs 56.2 | 38.0 vs 36.1 | 0.0 vs 6.5 |

On Christian and none, Pew's origin rows fit Luxembourg's EU residents well. On Muslims they do not:
the French, Belgians and Germans who move to Luxembourg are not their home countries' Muslims in
proportion. Rule 2 of `origin_religion.py` would allow a documented correction, but the ESS cells are
20 years old and small, and the other failing categories (Protestant, other) have no such witness,
so nothing was corrected and the method was dropped. The data it needs is on disk and the code
stays, so a later session can try a correction against the late totals.

Geoportail sweep, for that later session: `wms.geoportail.lu/public_map_layers/service` layers
2608-2612 are each nationality's share per commune from the 2021 census, one decimal. **WMS 1.3.0
reads an EPSG:2169 BBOX northing first**; an easting-first box north of the capital returns an empty
FeatureCollection with HTTP 200. `INFO_FORMAT=text/plain` is 200 bytes against 1.2 MB of polygons in
JSON. Two names differ from STATEC's (`Préizerdaul`, `Rosport-Mompach`). Commune shares x census
totals reproduce Eurostat's national counts within 0.27% for all five.

## 4. ESS 2002-04, the witness

`regionlu` is one value, so the split-half the brief asked about cannot run. 3,187 respondents
(1,552 + 1,635); pspwght raises non-citizens from 30.5% to 32.7% (census 2001, STATEC DF_B1753:
162,285 of 439,539, 36.9%). Citizens belonged 72.3%, non-citizens 73.9% (pspwght).
**`Other Christian denomination` is 16.2% of all answers** and 17% of Luxembourgers, Portuguese and
Italians alike, low among ex-Yugoslavs; EVS 2008 printed 1.9% other Christian for the same
population. It is a card artefact (most likely Catholics answering "Christian"), and one more reason
not to draw the 2002-04 mix. No country card variable exists (`rlgdnlu`, `rlgdnalu` E201).

## 5. Geography

GISCO LAU 2021, 102 communes, the census's own vintage (Bous and Waldbredimus, Grosbous and Wahl
merged in 2023). STATEC DF_B1625 codes `LU00<cc><nn>` end in the LAU code; every code pairs and
carries the same name in both files. GISCO POP_2021 (1 January 2021 register) against the census:
0.924 (Saeul) to 1.028 (Schifflange) of the national ratio. Kontur LU 2023 (3,306 hexes, 676,380
people): 188 centres fall outside the 1:1 million LAU line, all within 1 km; Natural Earth puts 78
(9,328 people) in Luxembourg, snapped and kept whole, and 110 (12,267) in Germany, Belgium or
France, dropped. The rest are cut by commune and each hex's people shared over its pieces by area
(Malta's method), then scaled to the census per commune. The first build clipped hexes and kept
their people, which put 1.56 million per km2 on a sliver; after the cut the densest piece is
10,676/km2 in Luxembourg City and `kontur_cap.py lu` finds nothing. Kontur over the census per
commune ran 0.74 (Schieren) to 3.01 (Kiischpelt) of the national ratio before calibration.

## 6. Calls someone might reverse

- The mean of the two surveys, equally weighted. EVS's sample size is unprinted and STATEC doubts
  its quality; TNS's is 515 and commissioned by an advocacy group. Either alone moves Catholic by
  about 5 points and none by about 5.5.
- One national mix rather than a nationality model (section 3).
- `Other religion` on `other.lu` (new node); Orthodox Christians are probably inside it.
- Muslims on `islam.sunni`, as the ESS countries file them.

## 7. Leads

- EVS 2020/21 microdata: held by the University of Luxembourg (STUDIALUX, Philippe Poirier); not
  deposited with GESIS as of the joint v5.0. A canton or region variable there would be the first
  sub-national figure. Needs a request.
- EVS 2008 (ZA4800) at GESIS: login; its Luxembourg region variable is unchecked.
- A nationality model with Pew's Muslim rows corrected for Luxembourg's EU residents, raked to
  the late totals per category (section 3).
