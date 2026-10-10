# Iraq (iq): record

**Since 2026-10-09 (session 32a047f0) Iraq is drawn from MICS6 2018 microdata** (§0). Sections
1-7 are the survey pool it replaced, kept as the comparison (`sources/iq_surveys.py`, now
writing `data/normalized/iq_surveys.csv`).

## 0. MICS6 2018 (`sources/iq_mics6.py`, `taxonomy/iq2018.py`, `countries/iq.py`)

Why: a viewer said Iraqi Turkmen were underrepresented (Kirkuk especially; also Baghdad's Raghiba
Khatun, and Ottawa's diaspora). The pool had Kirkuk 71% Arabic, 11% Turkmen. Anita made a
UNICEF MICS account (mics.unicef.org) and downloaded `IRQ_2018_MICS6_v01_M.zip`; its SPSS files
are in `data/raw/iq/mics6/` (gitignored: research use, no redistribution, and the readme asks
that copies of reports and publications based on the data go to the CSO and UNICEF Iraq). The
published report (washdata.org/report/iraq-mics-2018-sfr) tabulates no language at all.

Item: HC1B, language of the household head (Arabic / Kurdish / Turkman / Asserian / others),
read as every member's: hl.sav members x hhweight, per governorate (HH7). 20,214 interviewed
households, 131,394 members. Kurdish is split by HH16 (native language of the household's
respondent, with Kurdish Surani / Kurdish Badinani apart) inside each governorate's
Kurdish-headed households; where the respondent answered something else the Kurdish stays on
the parent `kurdish` (171,374 people, mostly Baghdad and Diyala, plausibly Feyli). Shares x the
2024 census governorate populations, largest remainder; 46,118,793 people, 46,114 dots.

HH16 alone was drawn first (same day) and dropped: where the interview was in Arabic it slides
to Arabic. Of Kurdish-headed households 107 answer HH16 Arabic (Baghdad 8 of 8; Diyala has 74
Kurdish heads and 20 Kurdish respondents); of Turkmen-headed, 36. Every Baghdad interview (HH15) was in Arabic. WM14 (women 15-49,
own native language) sits between the two.

| | Arabic | Sorani | Badini | Kurdish | Turkmen | Assyrian | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iraq | 81.94 | 11.57 | 3.24 | 0.37 | 1.98 | 0.18 | 0.71 |
| Kirkuk (pool was 71 Ar / 18 Ku / 11 Tk) | 37.8 | 28.6 | 0 | 1.2 | 30.8 | 1.6 | 0 |
| Nineveh (pool 85 / 4 / 11) | 87.0 | 0.7 | 0.6 | 0.4 | 4.0 | 0 | 7.2 |
| Salah al-Din (pool 0 Turkmen) | 96.6 | 0.2 | 0 | 0 | 3.3 | 0 | 0 |
| Diyala | 96.4 | 0.7 | 0 | 2.3 | 0.6 | 0 | 0 |
| Baghdad | 99.5 | 0 | 0 | 0.5 | 0 | 0 | 0 |

So: national Turkmen was about right (pool 1.6%, MICS 2.0%) but sat in the wrong place. Kirkuk is
the "roughly a third each" of the usual estimates, and Nineveh's Turkmen fall from 469,000 to
about 170,000 (Tal Afar was retaken from ISIS only in August 2017, so displacement may still show
in the 2018 fieldwork). National Kurdish 15.2%, inside 15-20%.

**Baghdad's Turkmen** (the viewer named Raghiba Khatun): none of Baghdad's 2,153 households has a
Turkmen head or respondent, and none of its 3,261 women. But the sample is 180 clusters, so an
enclave holding 0.25% of Baghdad (about 25,000 people) is missed entirely 64% of the time, 0.5%
41%, 1% 16%: the zero says Turkmen speakers are under about 1-2% of Baghdad, not that there are
none. Arab Barometer VI-3 + VII's ethnicity question found 2 Turkmen of 767 Baghdad respondents
(0.26%). **Anita, 2026-10-09: draw Baghdad's Turkmen from that ethnicity share** (`ETHNIC_FILL`
in `iq_mics6.py`): 2 of 764 valid answers, 25,603 people (about 26 dots), taken pro rata from
Baghdad's other rows; national Turkmen 2.04%. Inside a governorate dots follow population, so
they spread over all of Baghdad, not gather in Raghiba Khatun (Adhamiyah).

Mapping: Sorani -> new `indoeuropean.iranian.kurdish.central` (cent1972), Badini -> new
`...kurdish.northern` (nort2641, Kurmanji); every other country's Kurdish stays on the parent.
Colours in `tree.d/iq.txt` (Badini nearly the parent, so Duhok runs on into Turkey and Syria).
Nineveh's 7.2% "others" go on `other`: HC1B's "others" (mostly Nineveh) are 95% Muslim by
HC1A, so plausibly Shabak (usually put at 200,000-250,000; this is about 307,000), but MICS
does not name it. Shabaki is no longer drawn.

Still weak: placement inside a governorate is by population (Tal Afar's and Tuz Khurmatu's
Turkmen spread over Nineveh and Salah al-Din); Yazidis are few (HC1A Yazidi 25 households) and
Sinjar barely sampled; Feyli Kurds in Diyala and Baghdad hardly appear.

## The survey pool (2026-10-05 to 2026-10-09, superseded)

Drawn 2026-10-05 (session edd42a8c-iq). Five survey rounds with a first or home language
question, pooled per governorate and applied to the 2024 census governorate populations:
46,118,793 people, 18 governorates, 6 nodes, every row `modelled`. Duhok, which no language
question reached, is drawn from the ethnic group asked in 2020-22. Placed on religiondots' Kontur
400 m hexes. 46,116 dots at 1:1,000.

Files: `sources/iq_surveys.py` (fetch + counts), `taxonomy/iq2018.py`, `taxonomy/tree.d/iq.txt`,
`countries/iq.py`, `data/normalized/iq.csv`, `data/raw/iq/` (eight WVS online-tool pages).

```
python sources/iq_surveys.py --fetch
python taxonomy/build.py
python tools/check_country.py iq
python scatter.py --country iq
```

## 1. What exists

| source | language item | grain | open? | used |
|---|---|---|---|---|
| Census 2024 | none: language, ethnicity and ancestry left out on purpose (disputed territories) | - | - | no |
| Census 1997 | language spoken (IPUMS `LANGIQ`) | the three Kurdistan governorates not enumerated | IPUMS, gated (account blocked) | no |
| Census 1957 | mother tongue by liwa (Kirkuk 48.3% Kurdish, 28.2% Arabic, 21.4% Turkmen) | 14 liwas | tables not found online | context only |
| MICS6 2018 | household head's mother tongue; respondent's native language with Sorani / Badini apart | 18 governorates, ~20,000 households | UNICEF registration with identity | no (gated) |
| **WVS wave 7, 2018** (1,200) | `Q272` language at home | 9 governorates | online tool | **yes** |
| **WVS wave 6, 2013** (1,200) | `V247`, card Arabic / Other only | 9 governorates | online tool | **yes**, Other split (section 3) |
| **WVS wave 4, 2004** (2,325) | `V219` | 16 governorates (no Duhok, Nineveh) | online tool | **yes**, Kirkuk left out |
| WVS wave 5, 2006 (2,701) | `V222` | 18 governorates | online tool | **no**: codes scrambled (section 3) |
| **Arab Barometer II, 2011** (1,234) | `q10191` first language (Arabic, Kurdish, Turkmen, Shabaki) | 10 governorates | religiondots' .sav | **yes** |
| **Arab Barometer III, 2013** (1,215) | `q1019_1` first language | 9 governorates | religiondots' .sav | **yes** |
| Arab Barometer VI-3 (2020-21, 1,016), VII (2022, 2,460) | ethnicity `Q1012B` only | 18 governorates | religiondots' .sav | Duhok only, and the check |
| Arab Barometer V, VIII | no language or ethnicity | - | - | no |

WVS route and parser as `sources/ir.md`: Iraq is `AMIDS` 368, samples 3323 (w7), 2229 (w6), 462
(w5), 481 (w4); region cross ids 2437884 (w7), 43783 (w6), 1512 (w5), 49506 (w4). Counts are
unweighted (percent x N integral). Arab Barometer counts unweighted too, for one rule across the
pool.

## 2. The interview-language check (the Sudan trap)

Sudan's surveys interviewed in Arabic and missed the minorities. Iraq's did not miss the Kurds:

- Every round that sampled Erbil and Sulaymaniyah found them 96-100% Kurdish; Arab Barometer
  VI-3 and VII found Duhok 100% Kurd or Yazidi (64).
- WVS 6's `V257` interview language is Kurdish for exactly the 77 Erbil and 90 Sulaymaniyah
  interviews, and Arabic everywhere else (asserted against V247's Other there).
- WVS 7's `S_INTLANGUAGE` says Arabic for all 1,200, including the 167 Erbil and Sulaymaniyah
  respondents who all answered Kurdish at home: the field is not informative there.
- Arab Barometer's Iraq files carry no interview-language field. Wave VII has a Kurdistan-only
  module (`Q915A_KRI`), so the Region had its own fieldwork.

**National Kurdish share drawn: 15.4%**, inside the 15-20% "Kurdish" of the usual ethnic
estimates (CIA World Factbook). The Kurdistan Region's three governorates alone are 14.1% of the
2024 census.

**Check against a different question:** pooled Kurdish share per governorate against Arab
Barometer VI-3 + VII's Kurd share (ethnicity, 2020-22), 17 governorates: Pearson **+0.999**,
Spearman +0.625 (held down by the ten southern governorates tied near zero on both). Largest gaps:
Kirkuk +5.6 points, Diyala -2.0, Erbil -2.0.

Where the minorities ARE undercounted, said in `note_public`:
- **Nineveh** 3.6% Kurdish, 0.3% Shabaki, no Syriac: the samples sit around Mosul; Sinjar's and
  Shekhan's Yazidis (Kurmanji speakers, many still in Duhok's camps), the Nineveh Plains'
  Christians and Shabak (usually put at 200,000-250,000; drawn 12,000) are barely reached.
- **Diyala** 0% Kurdish: Khanaqin's Kurds (Feyli / Southern Kurdish and Sorani) not reached.
- **Kirkuk** 71% Arabic, 18% Kurdish, 11% Turkmen. The 1957 liwa was 48% Kurdish and most
  current estimates give the three groups roughly a third each, with Arab settlement since 2017;
  Arab Barometer VII's ethnicity gives Kirkuk 81% Arab, so the surveys agree with each other and
  plausibly over-sample Arab Kirkuk.

## 3. How the pool is made (`sources/iq_surveys.py`)

Per governorate, every usable respondent counts once, unweighted; no answer / missing dropped;
shares x 2024 census population (religiondots' `iq_lookup.csv`, 46,118,793), largest remainder.
Respondents per governorate: 64 (Duhok) to 1,572 (Baghdad); Muthanna, Maysan and Wasit have only
WVS 4 (102-114 each).

- **WVS 6's "Other"** is shared over the governorate's non-Arab ethnic answers (`V254`): Nineveh's
  59 Other are its 59 ethnic Turks (Tal Afar); Kirkuk's 18 over 11 Turk and 8 Kurd; Erbil's 77
  over 74 Kurd and 3 Turk; Sulaymaniyah's all Kurdish. The tool cannot cross three ways here.
- **WVS 4's Kirkuk is left out**: 59 of 114 answered Other against 34 ethnic Turks, so the Other
  mixes Turkmen with Arabs and Kurds and cannot be read. Kirkuk keeps three other rounds (225).
  WVS 4's other Others (Salah al-Din 10, Baghdad 9, a few elsewhere) stay on `other` (188,000
  people; Salah al-Din's are likely Tuz Khurmatu Turkmen but nothing says so).
- **WVS 5 (2006) is left out**: its Iraq codes are scrambled. It puts Basra 15% and Muthanna 22%
  Kurdish, Sulaymaniyah 57% Arabic, Babil 23% Kurdish, and 127 of 529 Baghdad at no answer.
- **Duhok** from Arab Barometer VI-3 + VII ethnicity (34 + 30): Kurd 63 and Yazidi 1, read as
  Kurdish (Yazidis speak Kurmanji). AGENT_BRIEF section 2's ethnicity rule; the retention check is
  Arab Barometer II's own ethnicity x language in Iraq, which match almost one to one (Arab 1,030
  / Arabic 1,032; Kurdish 180 / 177; Turkman 22 / Turkmen 21).
- Pooling years 2004-2018 assumes the language geography held; the large moves since (ISIS
  displacement 2014-17, Kirkuk 2017) fall mostly in the places already undercounted.

## 4. Mapping and tree (`taxonomy/iq2018.py`, `taxonomy/tree.d/iq.txt`)

| answer | node | people | % |
|---|---|---:|---:|
| Arabic | `afroasiatic.iraqi_arabic` (new) | 38,066,651 | 82.5 |
| Kurdish | `indoeuropean.iranian.kurdish` | 7,108,411 | 15.4 |
| Turkmen | `turkic.iraqi_turkmen` (new) | 727,977 | 1.6 |
| Other | `other` | 188,330 | 0.4 |
| Assyrian Neo-Aramaic (2 answers) | `afroasiatic.assyrian` | 15,264 | 0.03 |
| Shabaki (2 answers, Nineveh) | `indoeuropean.iranian.shabaki` (new) | 12,160 | 0.03 |

- **Iraqi Arabic**: Mesopotamian Arabic (Glottolog meso1252 Gilit, nort3142 North Mesopotamian /
  qeltu). One node beside `arabic`, the Darija / Sudanese Arabic precedent; the surveys do not
  tell Gilit from qeltu, and no split is guessed from the governorate.
- **Kurdish not split.** No card named a variety. Sorani (Sulaymaniyah, Erbil, Kirkuk), Badini /
  Kurmanji (Duhok, Nineveh) and Feyli (Diyala, Baghdad) are well known by place, but per the brief
  a variety gets a leaf only where the source names it, and Iran was drawn the same way. MICS6
  has the split if it is ever fetched.
- **Iraqi Turkmen**: a leaf of its own, not `turkic.turkmen` (Turkmenistan's language); Glottolog
  has Iraq's Turkmen as the Kirkuk dialect (kirk1242) of South Azerbaijani.

## 5. Geography and colours

religiondots' `data/geo/iq/iq_hexes.gpkg` (read-only), `unit` = IQG01-IQG18 as `iq_lookup.csv`,
COD-AB boundaries (religiondots chose them over geoBoundaries for Baghdad's extent). Halabja is
inside Sulaymaniyah, as the census tabulates. Scatter warned of religiondots' unreviewed Kontur cap
block 18 km from At Taji (0.8% of Baghdad); religiondots left it as under 5% of its unit, and so
does this.

Colours: Iraqi Arabic hand-set mint #97cda5 (0.80 0.08 152, one shade off Sudanese Arabic and
Darija); Kurdish keeps its olive-sand #abba77; Iraqi Turkmen generated pink #fc7b9d; Shabaki
generated dark green. The three that meet in Kirkuk and Nineveh differ in hue.

## 6. Calls someone might reverse

- Pooling five rounds over 2004-2018, unweighted, one respondent one vote.
- Dropping WVS 5 entirely and WVS 4's Kirkuk.
- Splitting WVS 6's Other by the governorate's ethnic marginals.
- Duhok from ethnicity (64 respondents).
- Kurdish not split into Sorani / Badini / Feyli by governorate.

## 7. Room for improvement

- MICS6 2018 microdata (Sorani / Badini apart, 18 governorates, ~20,000 households) would replace
  all of this; it needs a UNICEF registration with identity.
- WVS microdata (the form download) would give weights and a true three-way for wave 6.
- A source for Nineveh's minorities (Yazidi, Shabak, Syriac by district) would fix the largest
  undercount.

## Terms

WVS online tool: free, citation requested. Arab Barometer: free download, citation requested.
Census populations via religiondots (Central Statistical Organisation). Glottolog CC BY. Kontur
CC BY 4.0.
