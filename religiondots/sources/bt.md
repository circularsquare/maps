# Bhutan (`bt`)

Drawn 2026-10-03 by `fafd1067-bt`, reopened on Anita's priority list (`ask/RULINGS.md` 2026-09-15)
and the Mauritania ruling (2026-09-16: a country no census asks is drawn on the best survey or
compiler figure, method disclosed). 20 dzongkhags, 727,145 people (the 2017 census), every row
`modelled`. Ask 046 (spec §14) is open on the Hindus' placement. Code: `sources/bt_geo.py`,
`sources/bt_grid.py`, `sources/bt.py`, `taxonomy/bt2010.py`, `countries/bt.py`.

## 1. What is asked, and what is published

| instrument | asks religion? | published |
|---|---|---|
| PHCB 2005 census | **yes**, everyone: form PHCB-2C item "Religion", 1 Buddhism, 2 Hinduism, 3 other (IHSN catalogue 1374 DDI, `q4ca`; also `q6ca_rec`, language mainly spoken, 14 codes incl. Lhotshamkha) | **nothing**: no table in the 502-page report (NSB download 5058, Wayback 2022-03-19), the factsheet (5050), *Socio-Economic and Demographic Indicators 2005* (5056) or the online table set `nsb.gov.bt/GIS/phcb/tables/` (sections 3-8, Wayback 2011). Microdata on written application with a signed undertaking enforceable in Bhutan's courts (DDI `dataAccs`) |
| PHCB 2017 census | no (the 2017 metadata and report, `sources.md` scout 2026-09-14) | non-Bhutanese by dzongkhag, no nationality |
| BLSS 2007, 2017, 2022; BMIS 2010 | no (scout 2026-09-14) | |
| GNH surveys 2010, 2015, 2022 (Centre for Bhutan & GNH Studies) | **yes**, Q12 "What is your religion?" Buddhism / Hinduism / Christianity / Others; Bhutanese 15+, ~7,100-11,000 respondents, designed representative by dzongkhag | national only. Weighted 2010: "Eighty-one per cent of Bhutanese are Buddhists, 18% are Hindus, and 1.2% are Christians" (*An Extensive Analysis of GNH Index*, 2012, p.153). 2015 and 2022 print only the unweighted sample (2022 report Table 1 p.12; 2015 report Table 3 p.54). Microdata not open (none on the Centre's site; GHDx lists 2010 as tabulations only) |

The 2026-09-14 scout closed Bhutan "at the variable list" because the 2005 *topics* listed no
religion and the form was not seen. The DDI's variable list has it. Lesson in
`playbooks/census_table.md`.

The unweighted samples: 2010 7,139 (13.1% Hindu, 1.2% Christian), 2015 7,152 (14.5%, 2.0%), 2022
11,052 (12.2%, 2.1%, plus 18 other and 60 none). In 2010 weighting moved Hindu from 13.1% to 18%,
so the later raw shares are not levels. The 2010 estimate has three categories (the 2010 sample
recorded no other or none); the 2022 sample's 0.7% other and none sit inside the Buddhist remainder.

## 2. Level

GNH 2010 weighted: Hindu 18%, Christian 1.2%, Buddhist the remainder (80.8%; 81% printed, and the
three printed figures sum to 100.2). Applied to the 2017 census's 681,720 Bhutanese (Table 2.6).
Children are drawn at the 15+ shares; Lhotshampa fertility is higher, so the Hindu share of all
ages is if anything understated. Vintage: a 2010 level on 2017 people; the raw samples show no
trend (13.1, 14.5, 12.2).

## 3. Placement: Hindus by Nepali mother tongue (spec §14.10, §14.12)

*A Compass Towards a Just and Harmonious Society: 2015 GNH Survey Report* (2016; Wayback copy of
`grossnationalhappiness.com/wp-content/uploads/2017/01/Final-GNH-Report-jp-21.3.17-ilovepdf-compressed.pdf`),
Table A1.5, pdf pp.300-301: mother tongue by dzongkhag, weighted ("All analysis results are sample
weighted unless otherwise stated", p.51). Nepali (Lhotshamkha) 18.69% nationally (Table 65); the
p.187 prose repeats five southern rows and is asserted against the table read.

    hindu_d = 0.18 * N_d * L_d / L,   L = sum(N_d L_d) / sum(N_d) = 18.28% on the 2017 census

so each dzongkhag's Hindu share is 0.985 x its Nepali share: Samtse 55.4%, Tsirang 43.1%, Sarpang
39.2%, Chhukha 38.1%, Dagana 33.7%, Thimphu 13.8%, the east under 2.5%.

§14.12's three conditions: (1) X by unit published, yes; (2) religion x X nationally, **no**: the
coefficient is an assumed identity, supported only by the two national figures (18% Hindu 2010,
18.69% Nepali 2015; the 2015 report also says in passing, printed p.9, that its "deeply happy"
group is 83% Buddhist and the remainder Hindu, "which mirrors the proportion of each group in the
population at large"); (3) a
second cut to check against, **no**. Said in the note. What it misses: Lhotshampa who are Buddhist
or Kirat (Tamang, Gurung, Sherpa, Rai, Limbu; many give their own language and sit in A1.5's
`Others`, 24% of Tsirang, 20% of Dagana, 15% of Sarpang) and Hindus with another mother tongue.

Christians are drawn flat at 1.2%: nothing places them, and they are the group the state restricts
now. `HINDU_PLACEMENT = "national"` in `sources/bt.py` is the reversal (ask 046).

## 4. Non-Bhutanese

45,425 by dzongkhag (PHCB 2017 Table 2.8; excludes 8,408 tourists in hotels and 16,057 day
workers). No nationality tabulated. UN DESA International Migrant Stock 2024 file (`data/raw/mr/`),
2020 column, destination Bhutan: 53,612, of which India 46,974, China 949, Nepal 751, `Others`
4,098; the 24 named origins (49,514) are the mix, `Others` assumed alike. Each through Pew 2020 via
`taxonomy/origin_religion.py`, islam.* folded to `islam`. Result: hinduism 34,778, islam 6,689,
unaffiliated 1,004, sikhism 933, Christians 1,248, buddhism 546, other.bt 107. Known skew left as
is: origin_religion's India `Other` split is 85% Sikh, written for Punjabi migrants in Spain and
Greece; Bhutan's Indians are mostly from West Bengal, Assam and Bihar, so a few hundred Sikhs here
are probably others. Not worth a Bhutan-only split.

Check on the level: 21.66% of residents drawn Hindu against Pew 2020's 22.51% for everyone living in
Bhutan (Pew is close to the 2005 figure the CIA Factbook carries, 22.1%, source not stated).

## 5. Geography and placement

COD-AB `cod-ab-btn` v01 (20 dzongkhags, 205 gewogs), pcodes pinned to the census names in
`sources/bt_geo.py`. Census Table A2.8 swaps Lhuentse's and Monggar's areas (COD's Lhuentse is the
northern one) and prints three dzongkhags high (Trashigang 3,066 km2 against COD 2,202; Sarpang,
Thimphu); its 20 areas sum to 40,454 against its own 38,394. 17 of 20 agree with COD within 1%.

Kontur BT 2023-11-01 calibrated to each dzongkhag's 2017 total. The extract holds 116 hexes
(72,617 people) whose centroid is in India or China, 52,177 of them at Jaigaon beside
Phuentsholing; they are dropped, not snapped (Bhutan has no coast). Kontur/census 1.106 nationally;
Spearman +0.950 over 20 (0 of 20,000 shuffles); Chhukha 1.31 and Samdrup Jongkhar 1.28 still lean
to their border towns. No block reaches the density cap.

## 6. Not done, and why

- **PHCB 2005 microdata** (religion and language by gewog): needs Anita's identity and a signed
  undertaking, and would publish what the state withheld. Not requested; ask 046 says so.
- **GNH microdata**: not open. If the 2022 file is ever released, religion by dzongkhag from 11,052
  respondents would replace the language model and give the check §14.12 wants.
- **Gewog grain**: the dzongkhag reports print population by gewog (Table A2.1), but nothing about
  religion or language is finer than dzongkhag, so it would only move Kontur's weight.

## Review, 2026-10-03 (`fafd1067-rev5`)

Full pass. Fixed: three `Â§` (a double-encoded section sign) in `countries/bt.py`, two in the
internal `note` field and one in `_bt_counts`'s docstring; `note` does not reach `counts.json`, so
nothing to refresh. Drawn shares recomputed from the entry's own `counts`: Hindu 21.66% of all
residents, highest in Samtse, Tsirang, Sarpang and Chhukha, under 6% in every eastern dzongkhag
with foreign residents included, matching the note. No shared code touched by this build. Map glance: dots inside the border, the
Jaigaon hexes gone, Hindus visibly in the south-west.
