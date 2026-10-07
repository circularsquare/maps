# Comoros (`km`): record

Session `fafd1067-km`, 2026-10-03, under a supervisor. Reopens `queue.md`'s row from
§scout-2026-10-03-negatives (one national mix on Pew 2020) on the lead from the Djibouti builder
(Afrobarometer's country list now names Comoros). Drawn at 3 islands, one national mix, from
Afrobarometer round 10's summary of results, on the 2017 census's island counts. No ask filed.

## 1. What was checked, 2026-10-03

| source | what came back |
|---|---|
| Afrobarometer data sets (`afrobarometer.org/data/data-sets/?select-countries[]=comoros`) | **no data set**: "There are currently no search items". The same filter for Namibia lists 13, so the filter works. Round 10 files are out for 10+ other countries (Mozambique 2 Oct 2026, South Africa, Mauritania, Togo, ...). |
| Afrobarometer Comoros country page | R10 questionnaire (9 Dec 2025), codebook (25 Aug 2025), *Résumé des résultats* (30 Sep 2025, revised 1 Nov 2025). Fieldwork 13 May to 5 June 2025, 1,200 citizens 18+, Co.Fin.Co; frame INSEED's 2025 projections from the 2017 census, stratified by région (préfecture) and urban/rural. |
| R10 *Résumé des résultats* | p.5 sample description, weighted: Chrétiens 0.3, Musulmans 99.6, Autre 0.1, Refus 0.0. p.7 Q97 Total: Aucune 0.1, Chrétien seulement 0.0, Mormon 0.2, Musulman seulement 96.2, Sunnite seulement 1.0, Ismaélite 2.4, Refus 0.0. Cut by urban/rural and sex only; **no island or préfecture cut** of any question. |
| R10 codebook | Q97 unweighted: Aucune 1, Mormon 3, Musulman seulement 1,154, Sunnite 12, Ismaélite 29, refused 1 (= 1,200). `REGION` has 18 préfectures; `LOCATION.LEVEL.1` the islands (Mwali 83, Ndzuwani 488, Ngazidja 629). |
| RGPH 2017 (INSEED NADA `DDI-COM-RPGH-2017`, variable list via `api/catalog/DDI-COM-RPGH-2017/variables`) | 351 variables, **no religion**; nationality is B10. Agrees with §11aq. |
| MICS 2022 report (UNICEF, 704 pp., every `relig` hit read) | no religion item; only "felt discriminated against for religion or belief" (Tableau EQ.3.1W). |
| DHS 2012 (EDSC-MICS II, FR278, open report) | Tableau 3.1, women and men 15-49: Musulmane 99.0 / 99.3, Catholique/Protestante 0.3 / 0.3, Manquant 0.6 / 0.4. National only. Microdata off the table (asks 047/048). |
| EHCVM 2020 and 2024 (NADA) | 2020 asks `s01q12`, licensed (§11aq). 2024 not opened. |
| UNSD oracle | Comoros absent. |
| Pew 2020 | Muslim 98.30, Christian 0.51, unaffiliated 0.13, other 1.06 (2010: other 0.03; the jump is not explained). |

## 2. The construction

`sources/km.py`. The weighted religion line of the sample description: Muslim 99.6, Christian
0.3, `Autre` 0.1, refused 0.0 (dropped). `Autre` is Q97's `Aucune` (asserted: the only non-refusal
answer outside the two religions), so it is `unaffiliated`. The same three shares on each island's
2017 census count (INSEED, annex Tableau 1 of *Cartographie de vulnérabilité au COVID-19*, 9 July
2020: Ngazidja 379,367, Ndzuwani 327,382, Mwali 51,567, total 758,316; each island's préfectures
asserted to sum to it). Every row `modelled`; `may_ring` False (spec §3.10, as Bahrain).

Drawn: 755,284 Muslim, 2,275 Christian (two dots at 1:1,000), 757 no religion (under one dot,
drawn nowhere and not ringed).

Why Afrobarometer over Pew: it is self-identification, measured in 2025, and its Christian share
agrees with DHS 2012's 0.3% exactly (asserted within 0.5 points). Pew's 0.51 Christian and 1.06
"other" are a witness, as for Eritrea.

## 3. Calls someone might reverse

- **Islands, not préfectures, as units.** The mix is national, so the units only decide how many
  dots each place gets; the census prints 18 préfectures and 54 communes, but COD-AB adm2 has 17
  units that do not match (no Moya; Mitsamiouli and Mboudé merged; a Kartala summit unit). The
  préfecture counts are used as a witness instead (§4). Cost of reversing: a commune or préfecture
  layer rebuilt from COD adm3, an hour.
- **No foreigner layer.** UN DESA 2024 has 12,449 migrants in Comoros, 9,569 born in Madagascar,
  on a smooth decline from 14,079 in 1990: a back-projection, and most Madagascar-born residents
  are likely Comorians (the 1976 Majunga expulsions). Running them through Pew's Madagascar row
  would invent several Christian dots. Children and foreigners take the adult citizens' mix.
- **No Muslim branch.** Sunni 1.0% and Ismaili 2.4% are answers, but nothing places them (Oman,
  Saudi and Tajikistan rulings). The Ismaili share is not checked against any other source.
- **The three Christian respondents are coded Mormon** in the codebook; drawn on bare
  `christianity`, since three answers cannot name a church.
- **No §14 ask.** A national mix places nobody. Proselytising restrictions on Christians are a
  reason not to place them, and they are not placed.

## 4. Geography and placement

`sources/km_geo.py`. COD-AB Comoros adm1 (valid 2019-12-05), joined by p-code (KM1 Ndzuwani, KM2
Ngazidja, KM3 Mwali), names asserted. Mayotte is not in the file and is drawn with France.

**Kontur KM has Anjouan's and Mohéli's people swapped.** Raw, per island over the national ratio
(1.140): Ngazidja 1.05, Ndzuwani 0.18 (67,617 people, no hex above 1,529, Mutsamudu included),
Mwali 5.84 (343,284; Fomboni hexes of 37,211 and 33,945, at the density cap). Kontur's Mwali over
the census's Ndzuwani reads 0.92 and Kontur's Ndzuwani over the census's Mwali 1.15: a swap of the
island totals with the footprints left in place. Each island's hexes are scaled to its census count
(densest hex afterwards 9,267/km2), and `kontur_cap.py km` finds nothing. Witness of the shape
inside each island, calibrated Kontur's share of its island per COD adm2 over the census
préfectures' (Moya counted with Sima): 0.79 (Mbadjini Est) to 1.16 (Hambou, Mitsamiouli-Mboudé);
Kartala holds 3 people. 104 coastal hexes (51,413 raw people) snapped within 1 km; none dropped.

## 5. Reopen if

- Afrobarometer publishes the R10 Comoros data set: `LOCATION.LEVEL.1` gives the island split
  (Ndzuwani 488 respondents, Ngazidja 629, Mwali 83), and `REGION` the 18 préfectures. With
  Christians at 3 respondents only Muslim/other is worth testing, so the redraw would change little.
- INSEED publishes EHCVM tables with religion by island, or the 2024 EHCVM opens.
