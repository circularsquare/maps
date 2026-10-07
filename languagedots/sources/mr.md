# Mauritania (mr): record

Drawn 2026-10-05 (session edd42a8c-mono4). 4,927,531 people (RGPH 2023), 15 wilayas. Nationals'
rows `modelled`, foreigners' `derived`. 4,915 dots at 1:1000, 18 rings.

```
python sources/mr_afro.py
python sources/origin_mix.py --fragment mr
python taxonomy/build.py
python tools/check_country.py mr
python scatter.py --country mr
```

Files: `sources/mr_afro.py`, `taxonomy/mr2024.py`, `taxonomy/tree.d/mr.txt`, `countries/mr.py`,
`data/normalized/mr.csv`, `data/raw/mr/` (Afrobarometer R10 .sav, MICS 2015 report). Read-only
from religiondots: `data/normalized/mr.csv` and `mr_foreign.csv` (only their per-wilaya sums,
which are RGPH 2023's Mauritanians and foreigners), `data/geo/mr/` (lookup, hexes).

## 1. What exists

- **RGPH 2013** asked mother tongue; results withheld (coverage sweep). **RGPH 2023**: none of
  the four thematic reports religiondots holds mentions a language; nothing found online.
- **DHS 2019-21** (FR373): no language or ethnicity item in the report.
- **MICS 2015** (`mics5_2015_rapport.pdf`, Tableau HH.3, p.50): language of the household head,
  national only: Arabic 82.0, Pulaar 13.0, Soninke 2.6, Wolof 1.7, other 0.7. Microdata gated.
- **Afrobarometer R10** (2024; Mauritania's first round; country file on afrobarometer.org,
  survey-resource "mauritania-round-10-data-2025"): 1,200 adult citizens, Q2 "langue parlée dans
  le ménage", REGION = 15 wilayas, Q83A ethnic group. The source. Q115 ("première langue
  parlée") was looked at and not used: it is a list of languages spoken, and 93 of 134 Pulaar-
  at-home respondents put Arabic first.

## 2. How the counts are made

- Mauritanians per wilaya x the wilaya's weighted Q2 shares (respondents per wilaya: 16 in
  Adrar and Inchiri to 168 in Nouakchott-Nord and -Sud).
- **French** (16 answers, 1.5%): moved to the respondent's Q83A ethnic group's language (9);
  the 7 who gave only a national identity are dropped before the shares. Nigeria's English rule.
  **2026-10-05, ask 018 closed** (Anita: lingua francas at Afrobarometer R7's mother-tongue
  question, Q2A): Mauritania is not in R7 (R10 is its only round), so there is no Q2A. This
  reading stands: French 0 among Mauritanians (its answers on the ethnic language).
- **Foreigners** (125,933, Tableau 16.6 per wilaya): 46,800 refugees (Tableau 16.5) in Hodh
  Chargui on Mali's 2022 census mix for Tombouctou, Mopti and Ségou weighted 0.897/0.065/0.036
  (UNHCR 2018 origin map, religiondots' weights); the rest at Tableau 16.2's national groups,
  Mali, Senegal, Morocco, Algeria through `origin_mix`, other Arab countries on
  `afroasiatic.arabic`, other African on `africa_other`, Europe and the rest on `other`.
- Every wilaya sums to RGPH 2023 (asserted).

**Check against MICS 2015** (household heads): drawn among Mauritanians Hassaniya 83.6%,
Pulaar 12.2%, Soninke 3.5%, Wolof 0.5%; MICS 82.0 / 13.0 / 2.6 / 1.7. Wolof is drawn low (R10
found 4 Wolof households; its own ethnic question gives 0.6%).

## 3. Calls someone might reverse

- French folded to ethnic language (above).
- Refugees on northern Mali's census mix: Mbera's people are mostly Tuareg and Arab, so the
  Songhay share (38% of refugees, 18,000) is probably too high; religiondots made the same call.
- Wilaya shares from very small samples in the north (Adrar, Inchiri 16; Tagant, Tiris
  Zemmour 24): all Hassaniya there, which is plausible.
- Nouakchott-Nord: 1 non-Hassaniya respondent of 168, against 17% in Nouakchott-Sud. As surveyed.

## 4. Room for improvement

The RGPH 2013 or 2023 mother-tongue table, if ever published, at moughataa level.
