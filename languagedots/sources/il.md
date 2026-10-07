# Israel (il): record

Drawn 2026-10-05 (session edd42a8c-il). CBS Social Survey 2021 native language (age 20+), by
sub-district and population group, applied to the CBS 2022 census population of each statistical
area inside the Green Line plus the Golan: 8,418,491 people, 2,968 units, 10 nodes, rows
`modelled`. 8,413 dots on religiondots' statistical-area polygons (uniform inside each, as
religiondots measured the Kontur grid to be coarser than the units).

Files: `sources/il_social.py`, `taxonomy/il2021.py`, `taxonomy/tree.d/il.txt`, `countries/il.py`,
`data/raw/il/social2021_native_{subdistrict,district,age}.json`, `data/normalized/il.csv`.

## Sources

| source | item | result |
|---|---|---|
| census 1995, 2008 forms; 2022 register-based | no language item | population only |
| census 1983 | main language, age 15+ | IPUMS microdata only, gated: skipped |
| **CBS Social Survey 2021** (table generator) | native language, 9 codes; language at home, 6 | **used**: sub-district x population group, weighted estimates with SE |
| CBS Social Survey 2011 | native language, language at home | same generator; older, not used |
| religiondots `il.csv` (CBS 2022 census, register religion) | population by statistical area | the population base |
| religiondots `bycode2023.xlsx` | locality sub-district and population group | Arabs vs Jews and others per locality |
| religiondots `sa2022.csv` (CBS 2022 statistical-area profile) | age 0-19, origin by continent, main country of birth | children; placement inside sub-districts |

The generator (boardsgenerator.cbs.gov.il, linked from cbs.gov.il "Social Survey Generator") is
an Angular page over two JSON handlers that answer plain posts with no login: the docstring of
`sources/il_social.py` has the routes and variable ids. Only 2011 and 2021 carry a language item
(checked 2002-2025). `lang=en` breaks it (the WAF serves an error page); `lang=English` works.

Survey national, Jews and others 20+ (4,801,909): Hebrew 67.9%, Russian 15.5%, English 2.6%,
French 2.1%, Yiddish 1.6%, Spanish 1.3%, Amharic 1.1%, Arabic 1.3%, other 6.7%. Arabs 20+
(1,112,117): Arabic 99.2%. Aged 20-24, Jews and others: Hebrew 83.1%, Russian 7.5%.

## Method and checks

- **Territory follows religiondots exactly.** The 267 units in religiondots'
  `dropped_units.json` (West Bank settlements and East Jerusalem) are left out and `counts()`
  refuses them; the survey's Judea and Samaria sub-district is not used. East Jerusalem's
  Palestinians are on Palestine's entry (`sources/ps.md`, Jerusalem governorate includes J1).
  Settlers are on neither: they are the `xs` entry, as in religiondots (`sources/xs.md`, 2026-10-05).
- **Groups per locality from bycode2023** (Arabs / Israelis, 2023 provisional) times the 2022
  census total, placed on each area's Muslims + Christians + Druze, any surplus on the rest.
  Religiondots' register rows cannot be the groups: its "Other religions" allocation puts 3,587
  "Muslims" in Bat Yam (1,389 Arabs) and some Nazareth residents on Jews. Jerusalem is capped at
  its drawn Muslims + Christians + Druze, since bycode's row includes East Jerusalem. Result:
  Arabs 1,578,530, Jews and others 6,839,961; 96.7% of the register's Muslims, Christians and
  Druze counted Arab.
- **Shares**: Jews and others at their sub-district's shares shrunk to the district's by
  est / (est + 100,000) (Tzfat 0.30, Golan 0.13, Tel Aviv 0.91); Arabs at national shares.
  bycode2023's sub-districts 25 (second Jezreel) and 52/53 (Tel Aviv) folded into 23 and 51.
- **Children** (sa2022 `age0_19_pcnt` per area) at their group's national 20-24 shares.
- **Raking** per sub-district: unit totals and sub-district language totals held (200 IPF
  rounds, unit error asserted < 1 person), seed weighted by origin profile (docstring).
- Checks passed: sub-district and district rows sum to the national row per group and language;
  every row's languages sum to its total; all 2,968 units matched sa2022 and a sub-district;
  output sums to 8,418,491.

Spot checks (share of all residents): Ashdod Russian 23.9%, Haifa Russian 20.5% and Arabic
14.2%, Bene Beraq Yiddish 9.8%, Qiryat Malakhi Amharic 4.4%, Nazareth Arabic 98.9%, Bet Shemesh
English 6.8%.

## Calls someone might reverse

- Survey (2021, adults) instead of the gated 1983 census; tier `modelled`, not `measured`: the
  survey gives weighted estimates of adults, not counts of the population drawn.
- Native language rather than language at home (home has only 6 codes, no Yiddish or Amharic).
- Under-20s at the 20-24 cohort's shares; likely still overstates Russian among children.
- Arabic among Jews and others on plain `afroasiatic.arabic` (as other countries' plain "Arabic"),
  which is a group node with children elsewhere and so draws washed; not guessed to Judeo-Arabic.
- Arabs at national shares, not their sub-district's (cells of a few hundred respondents).
- Placement weights inside sub-districts are origin proxies (Russian by European origin, etc.):
  counts per sub-district are the survey's, positions borrowed.
- Hebrew hand-coloured (0.70 0.12 260, a mid blue): this recolours Hebrew's few dots in bo, cz,
  fi, fr and others.
- Settlers: built as `xs` on religiondots' units with the survey's Judea and Samaria shares
  (and Jerusalem's for East Jerusalem); `sources/xs.md`.

## Terms

CBS table generator: public, no login. CBS 2022 census figures via religiondots, read-only.
