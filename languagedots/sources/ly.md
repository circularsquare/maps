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
- Ghadames and Awjila inside Nafusi (no source splits them).
- Two English first-language answers drawn as English.
- AB III "Bahariya" read as Ajdabiya (Al Wahat).

## 5. Room for improvement

Any source with Tebu on the card, or a southern sample interviewed in Tamahaq or Tedaga.
A town-level population (Yafran, Zuwara, Ghadames) for the placements. Non-Libyans by district
and nationality (~827,000; not in the base).

## Terms

WVS online tool: free, citation requested. Arab Barometer: free download, citation requested.
Population via religiondots (BSC). Wikipedia CC BY-SA; Ethnologue figures as quoted there.
Glottolog CC BY. Kontur CC BY 4.0.
