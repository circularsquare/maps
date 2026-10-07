# Iran (ir): record

Drawn 2026-10-05 (session edd42a8c-ir). World Values Survey waves 5 (2005, 2,667 adults) and 7
(2020, 1,499 adults), language at home, pooled per province and applied to the 1395 (2016) census
province populations: 79,926,270 people, 31 provinces, 13 nodes, all `modelled`. Placed on
religiondots' Kontur 400 m hexes (calibrated there to the census's 429 county totals).

Files: `sources/ir_wvs.py` (fetch + counts), `taxonomy/ir2020.py`, `taxonomy/tree.d/ir.txt`,
`countries/ir.py`, `data/normalized/ir.csv`, `data/raw/ir/` (four WVS online-tool pages).

```
python sources/ir_wvs.py --fetch
```

## 1. What exists, and why this source

| source | language item | grain | open? | used |
|---|---|---|---|---|
| Census 1395 (2016) | none (religiondots `sources/ir.md` read the tables) | - | - | no |
| **WVS wave 7, 2020** (1,499) | `Q272` language at home, 7 answers; `Q290` ethnic group | `N_REGION_ISO`, 30 provinces (no Kohgiluyeh and Boyer-Ahmad) | online tool, no registration | **yes** |
| **WVS wave 5, 2005** (2,667) | `V222` language at home, 12 answers incl. Balochi | `V257`, 30 regions (Tehran still held Alborz) | online tool | **yes** |
| WVS wave 4, 2000 (2,532) | `V219`, online tool collapses it to Azerbaijani / Persian / other | 28 regions | online tool | fetched, not used |
| Values and Attitudes of Iranians, wave 3, 2015 (Ministry of Culture's Office of National Plans with the Interior Ministry; 14,906 face to face, 31 provinces) | "which language do you speak at home": Persian vs "local or ethnic language or dialect" only | province | report PDF (`ircud.ir/Media/PDF/1400/04/12/637608720832709393.pdf`, 718 pp., Table 19-30: 50.9% local, 49.1% Persian of 14,686); microdata not public | **check only** |
| GAMAAN 2022 (16,850 online, literate 19+, weighted) | language at home, 13 answers incl. Mazandarani, Laki, Tati, Turkmen | province asked, but only national shares published (report p. 28); no microdata | report only | no |
| Household Expenditure and Income Survey | no language variable | - | - | no |

**Why WVS.** It is the only open probability sample with a home-language item and a province
code. Its cost is size: 10 to 50 interviews in most provinces, and one or two clusters in each,
so a province's mix is rough. Pooling the two waves that name languages roughly doubles each
province. GAMAAN names more languages but publishes no province table, and its sample is online and
literate. The 2015 government survey is far larger but published only Persian against "local".

**The route.** `worldvaluessurvey.org/WVSOnline.jsp` needs no registration. Its frame is a chain of
JSP form posts (AJOnlineCountries -> AJOnlineIndex -> AJOnlineQtn, then AJOnlineQtn again with
`MACRUCE1`, and `MACRUCE2` for a three-way). Plain `requests` with a browser UA does it; no headless
browser needed (religiondots' playbook thought one was). Iran's sample ids: wave 7 `3466`, wave 5
`461`, wave 4 `482`; cross variables: wave 7 region `2437884`, ethnic group `2415047`; wave 5
region `1512`; wave 4 region `49506`. The pages print column percentages to 0.1% and each column's
N. Whether the tool weights is not stated; every percent x N lands within rounding of a whole number,
so these are unweighted counts.

## 2. Checks (all in `sources/ir_wvs.py`, all pass)

| check | result |
|---|---|
| province counts summed = IHSN catalogue national frequencies (catalog/11583 V308; catalog/8974 V721) | both waves, every answer, to the person |
| percent x N within 0.08 of a whole number; each column sums to its N | all 60 columns |
| 3-way (x ethnic group) inside the 2-way | yes; exactly the 13 with no ethnic answer drop out |
| population base = religiondots' `ir_lookup.csv` 1395 census totals | 31 units, 79,926,270 |
| Spearman, pooled Persian share vs Values and Attitudes 2015 (FDD's map of its microdata) | **+0.878** over 31 provinces |

The online tool prints 1 of 2,667 as 0.0%, so the national column is rebuilt from the provinces.
It also relabels wave 5's file codes: "Azari" (458) prints as Turkish, the file's own "Turkish" (7)
as Turkmen, "Asirien" as Assyrian Neo-Aramaic; the IHSN check pins that reading.

**Largest gaps against the 2015 survey** (WVS Persian share minus 2015's): Hormozgan +56,
Chaharmahal and Bakhtiari +40, Mazandaran -39, North Khorasan -25, Sistan and Baluchestan +23,
Semnan +22. Hormozgan and Semnan are mostly definitional: 2015's "local language or dialect"
holds Bandari/Larestani and Semnani, which WVS respondents filed as Persian (Yazd's 4% "local" shows
the 2015 answer took dialects). Mazandaran is the survey's own bias (section 3).

## 3. Pooling, and the two exceptions

Per province: wave 5 + wave 7 counts, unweighted, "No answer" (25) left out, shares x census
population, largest remainder so each province sums exactly.

- **Mazandaran and Golestan use wave 7 only.** Wave 5's list had no Mazandarani answer and its
  Mazandaran sample answered Persian (96.7% of 120). Wave 7's "Gilaki" took Mazandarani speakers:
  all 41 Mazandaran interviews gave it. The cost: Mazandaran draws **0% Persian** where the 2015
  survey has 39%, so Mazandarani (3.7M) is overdrawn there. Pooling wave 5 would instead have put
  Mazandarani at 25%, below 2015's 61% "local". Neither is right; the 2020 answer at least names
  the language.
- **Wave 5's Tehran** (540, Tehran and Karaj) is shared between Tehran and Alborz by 2016
  population.
- **Kohgiluyeh and Boyer-Ahmad** has wave 5 only (25 interviews, Luri 80%).

## 4. Mapping (`taxonomy/ir2020.py`), and the result

| answer | node | people | % |
|---|---|---:|---:|
| Persian | `indoeuropean.iranian.persian` | 48,375,384 | 60.5 |
| Azerbaijani (w7) + "Turkish" (w5 "Azari") | `turkic.azerbaijani` | 12,725,104 | 15.9 |
| Kurdish; Yezidi | `indoeuropean.iranian.kurdish` | 4,925,036 | 6.2 |
| Gilaki, in Mazandaran and Golestan | `mazandarani` (new) | 3,732,099 | 4.7 |
| Lurish; Luri; Bakhtiari | `luri` (new leaf) | 3,490,079 | 4.4 |
| Gilaki, elsewhere | `gilaki` (new) | 2,011,509 | 2.5 |
| Arabic | `afroasiatic.arabic` | 1,597,070 | 2.0 |
| Balochi (w5) + w7 Other from ethnic Baluch | `balochi` | 1,510,268 | 1.9 |
| Other (w7) | `other` | 1,362,287 | 1.7 |
| "Turkmen" (w5 file "Turkish") | `turkic.turkmen` | 121,647 | 0.2 |
| Armenian, Assyrian, Zoroastrian (w5, 2, 1, 1 answers) | existing / `zoroastrian_dari` (new) | 75,787 | 0.1 |

National comparisons: GAMAAN 2022 weighted Persian 63.2%, Azerbaijani/Turkic 12.8%, Kurdish 5.6%,
Luri 6.5%, Gilaki 1.4%, Mazandarani 2.1%, Arabic 1.6%, Balochi 1.4%, Turkmen 0.3%. The 2015 survey:
Persian 49.1% (its "local" includes Persian dialects).

- **Kurdish varieties are not told apart.** Both waves have one answer; no variety is guessed
  from the province (Sorani in Kurdistan, Southern Kurdish in Kermanshah and Ilam, Kurmanji in
  North Khorasan and northern West Azerbaijan would be the guess).
- **Luri** is one answer for the Luric languages (Northern Luri, Bakhtiari, Southern Luri; Glottolog
  luri1252), so one leaf. Laki had no answer.
- **Azerbaijani holds Iran's other Turkic speech** (Qashqai in Fars, Khorasani Turkic, Afshar):
  no answer of its own in either wave.
- **Wave 7's "Other"**: 24 of Sistan and Baluchestan's 25 came from self-described Baluch and are
  drawn as Balochi. The rest stay on `other`: Golestan 25 (half its sample; 18 of them gave ethnic
  group "Other", and Turkmen was on neither card, so these are very likely Turkmen but nothing in
  the file says so), Qazvin 10 (one cluster, ethnic "Other": Tati or Alamuti speech, unknown),
  Hormozgan 3. Golestan draws about 0.95M on `other` as a result.
- **Oddities left as printed**: wave 5 North Khorasan has 6 of 30 answering Gilaki (implausible
  for Bojnurd; perhaps Kurmanji misfiled), Chaharmahal and Bakhtiari 18% "Turkish" (Qashqai or
  Bakhtiari Turks plausible), Kerman's 10% Azerbaijani in wave 7.

## 5. Geography

religiondots' `data/geo/ir/ir_hexes.gpkg` (read-only), `unit` = English province name, the same
names `ir_lookup.csv` uses; asserted that the 31 units equal the survey's mapped provinces. Its
`pop` is Kontur calibrated to the census's 429 county totals, so dots avoid Kontur's false cities
(religiondots `sources/ir.md` section 5). Every language is placed by population within a
province: nothing says where inside one its speakers live. So in mixed provinces (West Azerbaijan,
Khuzestan, Golestan, North Khorasan) the languages are spread evenly over the province's people.
Scatter: 79,922 dots at 1:1,000.

## 6. Colours

Generated, none hand-picked: Persian green #8ac191, Azerbaijani magenta #b74d95, Kurdish olive-sand
#abba77, Luri salmon #f9af9f, Gilaki rose #e49c8d, Mazandarani dark olive #869453, Arabic mint,
Balochi tan #b38a50. Neighbours on the ground (Kurdish/Luri, Kurdish/Azerbaijani, Arabic/Luri,
Gilaki/Mazandarani, Persian/Balochi) differ in hue or lightness. Persian vs Mazandarani is the
closest pair that meets (Tehran/Mazandaran edge); left for Anita's palette pass.

## 7. Room for improvement

- WVS microdata (the form download) would give `W_WEIGHT`, settlement and PSU, so Golestan's and
  Qazvin's "Other" could be read by town and the cluster structure tested. The online tool is what
  was usable without the form.
- The Values and Attitudes 2015 microdata (14,906, all 31 provinces, ethnicity x language; FDD
  clearly had it) would replace WVS outright. Not public; the Ministry's RICAC site lists the study.
- GAMAAN province-level language tables, if ever published.
- A county-level placement of the minorities (Kurdish, Turkic, Arabic within their provinces)
  would need a source that says where; none found.
