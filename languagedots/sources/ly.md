# Libya (ly): record

**Drawn 2026-10-05** (session edd42a8c-mid). Three survey rounds with a home or first language
question pooled per district, plus cited estimates where the surveys plainly miss (Nafusa towns,
Tuareg, Tebu). 6,872,674 Libyans (BSC 2020), 22 districts, 6 nodes, every row `modelled`.
6,870 dots at 1:1000.

```
python sources/ly_surveys.py [--fetch]
python taxonomy/build.py
python tools/check_country.py ly
python scatter.py --country ly
```

Files: `sources/ly_surveys.py`, `taxonomy/ly2020.py`, `taxonomy/tree.d/ly.txt`, `countries/ly.py`,
`data/normalized/ly.csv`, `data/raw/ly/` (five WVS online-tool pages).

## 1. What exists

| source | item | grain | used |
|---|---|---|---|
| Census 2006 | no language item | - | no |
| **WVS 7, 2022** (1,196) | Q272 home language: Arabic 1,161, Berber 30, Other 5 | 22 districts | yes |
| **WVS 6, 2014** (2,131) | V247: Arabic, Berber 94, Tahaggart Tamahaq 3 | 22 districts | yes; the tool prints WEIGHTED percentages for this sample, so counts are fractional |
| **Arab Barometer III, 2014** (1,247) | q1019_1 first language: Arabic 1,206, Amazigh 28, English 2 | 22 districts | yes |
| Arab Barometer VI-3 (2021), VII (2022) | Q1012B ethnicity: Amazigh 166, Tuareg 40, Kouloughli 39, Toubou 13 | districts | check only |
| Arab Barometer V, VIII | no language or ethnicity | - | no |

WVS Libya: AMIDS 434, samples 3776 (w7) and 2228 (w6); region crosses 2437884 (w7), 43783 (w6),
as Iraq's. Every WVS interview was in Arabic (V257, S_INTLANGUAGE: Arabic only).

## 2. Method (`sources/ly_surveys.py`)

Per district, all usable answers pooled unweighted (WVS 6 at its printed weighting), don't know
and missing dropped, shares x BSC 2020 Libyans (religiondots' `ly_lookup.csv`, the base
religiondots uses). Then three corrections, each where the surveys are visibly short:

- **Ghat, WVS 7**: the card had no Tamahaq; Ghat's 4 "Other" of 10 read as Tamahaq (WVS 6's
  Ghat answered Tamahaq 31%).
- **Nafusi top-up**: pooled Berber is 227,085 against Ethnologue's 300,000 Nafusi (27th ed.,
  2020, via Wikipedia). The surveys found Nalut 71%, Zuwara 25%, Tripoli 3.4% Berber, but
  Jabal al Gharbi only 1.4% (3 of 214; their points look to be Gharyan and Zintan). The
  72,915 shortfall is added to Jabal al Gharbi (19.4% of it, cap 35% unused) and placed in a
  Yafran-Kikla-al-Qalaa box; Zuwara's Berber is placed in Zuwara town.
- **Tuareg and Tebu**: no card had Tedaga, and the southern samples answered Arabic (Murzuq
  50 of 50 non-Tebu). Wikipedia's infoboxes: Tuareg in Libya 100,000-250,000, Toubou
  50,000-85,000 (Shoup). The low ends (the base is citizens only), placed by this build:
  Tamahaq Ghat 20,000, Ubari 50,000, Sabha 15,000, Murzuq 10,000, Wadi al Shati 5,000; Tedaga
  Murzuq 20,000, Kufra 15,000, Sabha 8,000, Ubari 7,000. The district's other shares fill the
  rest.

Result: Libyan Arabic 93.4%, Nafusi 4.4%, Tamahaq 1.5%, Tebu 0.7%, English 3,131 (two Arab
Barometer first-language answers), other 1,278.

## 3. Check: it does not corroborate

Pooled Berber + Tamahaq share against Arab Barometer VI-3 + VII's Amazigh + Tuareg share, 22
districts: **Pearson +0.008**. Nalut 71% vs 7.5%, Zuwara 25% vs 1.7%. The ethnicity rounds put
their Amazigh in Jafara (64), Zawiya (28) and Misrata (17) and almost none in Nalut or Zuwara;
religiondots already found AB VII's district labels unreliable (Spearman +0.853, Ajdabiya
3.7x). Nationally the two agree (ethnicity 5.6% Amazigh; drawn 4.4% Nafusi). Kept the language
pool; reported, not used.

## 4. Calls someone might reverse

- The Tuareg / Tebu district split (the totals are cited; the split is mine).
- Nafusi top-up to Ethnologue's 300,000, all of it in Jabal al Gharbi.
- Ghadames' and Awjila's counts (section 6: Ethnologue's, carved by me), Zuwara as its own leaf.
- Two English first-language answers drawn as English.
- AB III "Bahariya" read as Ajdabiya (Al Wahat).

## 5. Room for improvement

Any source with Tebu on the card, or a southern sample interviewed in Tamahaq or Tedaga.
A town-level population (Yafran, Zuwara, Ghadames) for the placements. Non-Libyans by district
and nationality (~827,000; not in the base).

## 6. Berber split by place (2026-10-19, session eca0185b-ber)

Anita, "berber in africa". The Berber answers were all on `nafusi`, which also held Zuwara's,
Ghadames' and Awjila's speakers (section 4 had said no source splits them). Split now in
`countries/ly.py` `_split_berber()`; the survey file and `ly_surveys.py` are untouched.

- **Zuwara** (district LY0215, 85,758 pooled Berber answers): its own leaf `zuwara` "Zuwara (Zuwara
  Berber)". Glottolog files Zuwara's speech with Tunisia's as Tunisian-Zuwara Berber (tuni1262), not
  with Nafusi; kept a leaf of its own rather than Tunisia's `tunisian_berber` because the two
  places are 200 km apart and readers know it as Zuwara's. Placed in Zuwara town as before.
- **Ghadames** (ghad1239, gha): 17,000 speakers, Ethnologue 27th ed. 2020 as quoted on Wikipedia
  (13,100 in 2016 incl. 2,000 emigrants; the article warns it counts all inhabitants), carved out
  of Nalut's Nafusi (80,632 -> 63,632), placed in the Ghadames oasis box. Nalut's Nafusi now sits
  in a box over Nalut, Kabaw and Wazin, not the oasis.
- **Awjila** (awji1241, auj): 2,700 native speakers, Ethnologue 2020 (Brenzinger 2008 says
  2,000 to 3,000), carved from Ajdabiya's Libyan Arabic (no survey Berber there), placed in the
  Awjila oasis box. Three dots.
- **Sokna** (Sawknah-Fogaha, sawk1238): extinct since about 1960, not drawn.
- **Tamahaq at Ghat** was already its own node (Ghat 20,000, Ubari 50,000, ...); unchanged.

Result: Libyan Arabic 6,414,601 (was 6,417,301), Nafusi 196,220 (was 298,978), Zuwara 85,758,
Ghadames 17,000, Awjila 2,700, Tamahaq 101,986, Tebu 50,000. The 300,000 Nafusi target of
section 2 now covers the three together (196,220 + 85,758 + 17,000 = 298,978, the old Nafusi
total). Colours (OKLab, hand-set in `tree.d/ly.txt`): Zuwara `0.86 0.16 62` (Nafusi 0.200,
Tunisian Berber across the border 0.157), Ghadames `0.58 0.15 125` (Libyan Arabic 0.112, Nafusi
0.143), Awjila `0.90 0.12 70` (Libyan Arabic 0.346, Siwi 0.120).

## 7. Coverage gap: non-Libyans (2026-10-19, session eca0185b-gap)

Anita, "coverage gaps". The base is Libyans only (BSC 2020), so the migrants were not drawn. Now
added, `modelled`: Libya is 7,457,183 people (6,872,674 Libyans + 584,509 migrants). Built on the
other agent's Berber split (section 6), which is untouched. Files: `sources/ly_migrants.py`
(writes `data/normalized/ly_migrants.csv`, `ly_migrant_sites.csv`), `data/raw/ly/dtm_ly_migrants_r32.xlsx`,
`countries/ly.py` (`_counts`, `_BoxWeighter`), `tree.d/ly.txt` (generated borrowed-node block).

**Source.** IOM DTM Libya, Migrants Baseline Assessment **Round 32** (interviews July-August 2020;
HDX `libya-migration-data-migrants-baseline-assessment-iom-dtm`, resource 5b23eecf-8550-4c50-ada6-
40d02b94a419; open). 584,509 migrants in 575 muhallas (neighbourhoods) of 100 baladiyas in the 22
mantikas; sheet "Nationalities" gives each muhalla's migrants by 24 named nationalities and
"Other" (4,158, drawn on `other`). The "Muhalla Main" sheet gives the same totals (asserted equal for
every muhalla) and coordinates for 565 of 575 neighbourhoods. Round 32 is the nearest round to the
2020 base with a nationality sheet (R28-R31 and R33 have totals only; R34+ sit in 2021-26). Later
rounds count more (R49, 2023, about 700,000); the UN's 827,000 is a stock estimate for another date.
**Used as counted, not scaled to 827,000**: nothing says where the other ~240,000 are or who. Largest
groups: Niger 118,094, Egypt 97,247, Chad 91,690, Sudan 80,585, Nigeria 41,614, Ghana 27,220, Mali
26,631, Bangladesh 19,167, Syria 18,937. Tripoli district 83,140, Ajdabiya 71,615, Misrata 56,980,
Zawiya 41,835, Murzuq 41,337.

**Languages.** `origin_mix.mix(iso, "ly")` for each nationality: its country's drawn home mix with
languages under 1% cut (Niger: Hausa 56, Zarma 21, Tamajaq 9; Egypt: Egyptian Arabic 77, Sa'idi 23;
Sudan: Sudanese Arabic 89; Chad: Shuwa Arabic 19, Sara languages, Tubu...). No override: no survey of
migrants' home languages in Libya was found. Libyan Arabic is not used for any migrant. The 183 nodes
mostly hold a handful of people; Hausa 81,000, Egyptian Arabic 75,000, Sudanese Arabic 71,000 lead.
Niger's Tamajaq and Libya's Ghat Tamahaq stay separate nodes, as in their own countries.

**Placement.** Each district's migrants go around that district's DTM neighbourhood points: hex
weight = Kontur population x (a Gaussian of 2 km around each neighbourhood, weighted by its migrants,
plus a 2% floor); neighbourhoods without coordinates (5,816 migrants) share their weight on
population. All nationalities in a district share one weight. `_BoxWeighter` recognises a
(district, language) row with migrants by its count and mixes the Libyans' weight (Berber boxes or
population) with the migrants' in proportion.

**Calls someone might reverse:** Round 32 unscaled; home mixes for languages (a Niger migrant in
Libya may be Toubou or Tuareg rather than Hausa); one weight for all nationalities in a district.
**Room for improvement:** a survey of migrants' home languages (DTM's flow monitoring survey asks
nationality only); nationality-specific placement (Egyptians in the coast, Nigeriens and Chadians in
the south); refugees (UNHCR) beyond the DTM count.

## Terms

WVS online tool: free, citation requested. Arab Barometer: free download, citation requested.
IOM DTM data on HDX: CC BY (HDX licence field), open. Population via religiondots (BSC). Wikipedia CC BY-SA; Ethnologue figures as quoted there.
Glottolog CC BY. Kontur CC BY 4.0.
