# Bahrain (`bh`)

**Drawn 2026-10-03** by session `fafd1067-bh`, under a supervisor, from the scout's re-read
(`sources.md` §scout-2026-10-03-negatives; earlier §11ao and the 2026-09-14/15 Gulf re-checks).
4 governorates, 1,501,635 people (2020 census, everyone), Muslim / Others rows `derived`, the
split of foreign `Others` `modelled`, roll `NOWHERE`. Code: `sources/bh.py`, `sources/bh_geo.py`,
`sources/bh_grid.py`, `taxonomy/bh2020.py`, `countries/bh.py`. No ask filed.
`sources.md` §bh-2026-10-03; the Gulf rule applied since §bh-2026-10-03b (§6 below); Bahraini
Muslims split Shia / Sunni by governorate since §bh-2026-10-03c (ask 055, §7 below).

## 1. What was counted

All from `www.data.gov.bh` (Opendatasoft; `api/explore/v2.1/catalog/datasets/<id>/records`, which
answers; §11ao found `where=search()` returning 400, it worked on 2026-10-03). Raw JSON in
`data/raw/bh/`.

- `population-by-religion-nationality-and-sex-census-2020`: Muslim / Others x Bahraini /
  non-Bahraini x sex, national. 1,501,635; Muslims 1,111,533 (UNSD table 28's 2020 row, to the
  person). Bahraini Others 938 men, 1,357 women; non-Bahraini Others 267,628 men (46.0%),
  120,179 women (57.9%).
- `population-by-governorate-nationality-and-sex-census-2020`: 4 governorates x nationality x sex.
- `population-by-governorate-nationality-groups-and-sex-census-2020`: 4 x 8 groups (Bahraini, Gulf
  Co-operative Countries, Other Arabs, Asian, African, European, North American, Others) x sex.
  Agrees with the previous table cell for cell; both close on the religion table. The scout did not
  list this one; it is what lets each governorate's foreigners differ.
- The form codes Christian and Jewish (2010 QN 1.09; 2001 item 64), but 2010 and 2020 print only
  Muslim / Others. The 2001 census printed Christian 58,315 and other 63,896 (UNSD), used as a
  witness. Other portal religion sets: births by religion, and population by age and religion
  (national); neither has a place.
- Bahraini Sunni/Shia by governorate: found by the sect-register scout and drawn on Anita's ruling
  (ask 055), §7.

## 2. The construction, decided

- **Bahrainis**: each governorate's men and women at the national Bahraini shares by sex (99.74%,
  99.61% Muslim). The 2,295 Others on `other.bh`, not folded into Islam as Kuwait's 277 were: a
  real count, about two dots.
- **Non-Bahraini Muslim / Others**: each census group and sex starts from its UN DESA 2024 origins
  (Bahrain column; 33 named, `Others` unused; Gulf origins on Islam; Somalia and Sudan in Other
  Arabs, Türkiye in Asian; the census `Others` group takes the European and North American origins
  pooled) through Pew 2020. Prior national non-Muslim share: men 54.0% against the census's 46.0%,
  women 49.3% against 57.9%. One logit shift per sex on the Asian and African groups only (men
  -0.359, women +0.505) closes it exactly; Arab, European and North American groups keep Pew's
  shares (Other Arabs 4.3% non-Muslim). Reason for shifting only those two: DESA's worker streams
  are where it is worst (African women 2,027 in DESA against 15,826 in the census).
- **Others split** (changed 2026-10-03, §6): per group and sex, the non-Muslim families of its
  origins through Pew, then the Gulf rule as the UAE and Oman (`origin_religion.gulf_christian_hindu`):
  Christians / (Christians + Hindus) of the layer raked from 0.258 to Pew 2020's Bahrain 0.550 by
  moving 102,870 Indian non-Muslims (of 264,476) from Hindu to Christian, inside the Asian rows.
  Every other family stays at the DESA x Pew prior. Christian 49.9% / Hindu 40.9% of non-Muslims
  (prior 23.4% / 67.4%; Pew's row 52.6% / 43.1%); Buddhists 12,404, unaffiliated 10,952.
  `Other_religions` resolved per origin by `origin_religion.OTHER` (Sikhs 6,750, Jains 555).
  Christians bare `christianity`, Muslims bare `islam`.
- **Tier**: Bahraini rows and non-Bahraini Muslim `derived` (counted nationally, carried by
  governorate x group x sex); the Others split `modelled`. `other.bh` holds both and is
  `modelled` on the pair. **Roll** `NOWHERE`: Muslim and Others were counted for the country
  only (the kw review's fix). `check_rollup.py bh` reports everything orphaned, the honest answer.

**Witness** (asserted, 10 points): Christians 49.6% of all non-Muslims drawn, against 47.7% in the
2001 census. **Result**: 74.0% Muslim; Christians 193,586, Hindus 158,651. Muslim share by
governorate: Capital 63.9%, Southern 74.2%, Muharraq 78.1%, Northern 85.5%.

## 3. Geography and placement

- **geoBoundaries gbOpen BHR ADM1** (OSM, Wambacher, 2017): the four post-2014 governorates
  (Capital 71 km2, Muharraq 60, Northern 146, Southern 484). COD-AB `cod-ab-bhr` is FAO GAUL 2008,
  before the Central Governorate was abolished; not used. Join on the four distinct names.
  Witness: each lies 100% inside the same-named governorate of Kontur Boundaries BH 2023 (OSM,
  with territorial sea). Same lineage, so this checks vintage and labels, not tracing.
- **Kontur BH** (2023-11-01), uncalibrated: 1,485,697, 0.989 of the census. Per governorate over
  the national ratio: Capital 1.09, Southern 1.18, Muharraq 0.87, Northern 0.82 (band 0.67-1.5);
  6.8% of its people in a different governorate. 249 hexes outside the 2017 lines (reclaimed land,
  coast; 56,121 people), 238 snapped within 2 km, 11 dropped (1,192 people, 0.08%).
  `kontur_cap.py bh`: no stops.

## 4. Reopen if

- A census table crosses religion with governorate or block (the form codes region and block), or
  prints Christian apart again; the 2001 Arabic book and CIO `Census/Population/2.pdf`-`7.pdf`
  were never opened (§11ao).
- A count of foreign residents by nationality (country) and sex, to replace DESA's mix inside the
  groups (LMRA's work-permit tables cover workers only).
- A survey of Bahraini citizens that asks sect and records the governorate (the 2017 Washington
  Institute poll's data is "available on request"; Gengler's 2009 file has no region). That would
  replace the mosque registers as the geography (§7).

## 5. Review, 2026-10-03 (`fafd1067-rev11`)

Full pass: checks clean, `check_rollup.py bh` shows all 1,111,531 Muslims orphaned (the NOWHERE
roll, as intended), note figures re-summed from `bh.csv`, map shot fine (dots on land, Capital and
Muharraq densest).

- **Note wording**: "The form also coded Christian and Jewish" read as the 2020 form, which §1 does
  not cite (only 2010 QN 1.09 and 2001 item 64). Now "The 2010 census form also coded Christian and
  Jewish, and neither census printed them apart". `other.bh`'s description in
  `taxonomy/branches.py` still says "coded on the form"; left, it does not name a year.
- **The Buddhist inconsistency with `ae`/`om` is worth fixing, in this file's next touch.** Pew's
  Bahrain row is the same regional template as its other Gulf rows (§gulf-2026-10-03: Buddhists
  0.6% of non-Muslims everywhere); only its Muslim share is the census's. Raking every family to
  it throws away the origin signal the Gulf rule deliberately keeps elsewhere. Measured read-only
  with `bh.py`'s own functions: under the Gulf rule (rake only Christians / (Christians + Hindus)
  to Pew's 0.550, every other family at the DESA x Pew prior) the foreign `Others` would be
  Christians 193,587 (now 204,119), Hindus 158,650 (167,280), Buddhists 12,403 (2,383),
  unaffiliated 10,952 (4,778), `Other_religions` 11,859 (9,034) before `origin_religion.OTHER`
  resolves it. About 19,600 people change family, 1.3% of Bahrain; Christians would be 49.6% of
  non-Muslims, nearer the 2001 census's 47.7% than the 52.3% drawn. The fix touches `bh.py`'s
  column target, `NOTE`, the note's four figures and "52%", `taxonomy/bh2020.py`'s `REVIEW`, and `other.bh`'s count
  in `branches.py`; then a rescatter. Not done here (review rebuilds nothing). Done in §6.

## 6. The Gulf rule, 2026-10-03 (`fafd1067-bhfix`)

`sources/bh.py` now applies `origin_religion.gulf_christian_hindu` in place of raking every family
to Pew's Bahrain row (`gulf_rule()`; the IPF is gone). Indians sit inside the census's Asian rows,
so their part of each (Asian, sex) row is India's weight in that row's DESA mix times India's
non-Muslim share, over the row's; the shared function sees one key for Indians (both sexes, one
composition) and one per row for everyone else, and one fraction of Indian non-Muslims (38.9%) is
moved in both sexes. Row totals, so the census's Muslim / Others closure and every governorate's
Muslim share, are unchanged; asserted with `GULF_MATERIAL` (the move is 6.9% of the country).

Measured, to the reviewer's read-only estimate within rounding: Christians 204,119 -> 193,586,
Hindus 167,280 -> 158,651, Buddhists 2,383 -> 12,404, unaffiliated 4,778 -> 10,952, Sikhs 5,881 ->
6,750, Jains 484 -> 555, Jews 215 -> 357, foreign `other.bh` 2,560 -> 4,322 (`branches.py`
updated). 19,162 people change family (the reviewer's 19,600 was before rounding). `note_public` rewritten for the paragraph on the
split; `REVIEW` for christianity, hinduism and buddhism. `sources.md` §bh-2026-10-03b.

## 7. Bahraini Muslims split Shia / Sunni, 2026-10-03 (`fafd1067-bhsect`)

Anita's ruling, ask 055 ("ok, we can do bahrain"); sources from `sources.md`
§scout-2026-10-03-sect-registers. `sources.md` §bh-2026-10-03c. Code: `sect_split()` and its helpers
in `sources/bh.py`.

- **Level**: Arab Barometer wave I, Bahrain (January-May 2009, Justin Gengler's team with the Bahrain
  Center for Studies and Research), `q711`, read from `ABI_English.sav` and asserted: 249 Shia, 183
  Sunni, 3 "Muslim" of 435 citizens, unweighted (no weight column). Shia 57.2% of all answers,
  57.6% of those naming a sect. The 3 "Muslim" (0.7%) stay on bare `islam` in every governorate.
- **Geography**: each governorate's Shia share of mosques, Ja'fari Endowments 2016 (Capital 332,
  Muharraq 44, Northern 344, Southern 33) against Sunni Endowments about 2022 (jami' + masjid:
  92, 168, 89, 130; the 32 unplaced masjid dropped, which moves only the level and the level is
  the survey's). Prior 78.3 / 20.8 / 79.4 / 20.2%; one logit shift over all four, **+0.055**,
  closes the Shia total on the survey. Drawn Shia share of Bahraini Muslims: Capital 79.2%,
  Muharraq 21.7%, Northern 80.3%, Southern 21.1%. Islam.shia 406,452, islam.sunni 298,717, bare
  islam 4,898 (plus the 401,464 foreign Muslims).
- **Sensitivity, printed by the build**: counting ma'tams with the Ja'fari mosques gives prior
  87 / 41 / 86 / 33%, and at the same level Capital 79.2%, Muharraq 27.3%, Northern 77.4%,
  Southern 21.3%. Mosques alone were kept: a ma'tam is a hall for Muharram gatherings, often
  several per village and with separate women's halls (153 of the Capital's 306), so it is less
  like one congregation than a mosque is.
- **Witnesses, asserted**: OSM, fetched by `bh.py` (843 Muslim places of worship, 599 tagged Shia,
  16 Sunni, 223 untagged, 2026-10-03), puts Shia-tagged mosques at 0.86 / 0.66 / 0.74 / 0.88 of the
  register's count per governorate (band 0.6-1.25), so an independent source places the Shia
  mosques where the Ja'fari register does. OSM's Sunni tags are too few to check the other side.
  The Washington Institute's 2017 poll of 1,000 citizens (Pollock; multi-stage probability sample)
  found 62% Shia; drawn 57.6% of named, within the 10-point band.
- **The quota check could not run as the playbook has it, and was done from the documents.**
  `assert_not_quota` compares per-unit compositions across waves; Bahrain is in wave I only and the
  file has no region. Instead: the wave I technical report (`arabbarometer.org/wp-content/uploads/
  ABI_Methods_Report.pdf`, read 2026-10-03) gives Bahrain "Stratified area probability sample",
  frame from the Central Informatics Organization, and no strata line, while Lebanon's page in
  the same report says "Strata: Governorates and sect". Bahrain's census and frame carry no sect,
  so there was no official figure to set a quota to, and the answers come back as unround
  counts (249 / 183 / 3 of 435). Taken as no quota. Not read: Gengler's dissertation (Michigan,
  2011), which should describe the sample in full.
- **A sampling skew, recorded and not corrected**: 277 men, 151 women, 7 blank. Shia respondents
  are 72% men (173 of 242 with a sex), Sunni 55% (101 of 183). Weighting by sex to 50/50 would give
  54.4% Shia of named. Not applied: sect belongs to the household and the Kish grid picks one
  adult per household, so the sex skew says who answered, not which households were reached, and
  the household-size weight that would correct for one adult per household is not in the file,
  so its direction is unknown. The range
  54-62% across these readings is stated here, not on the map.
- **Tier and roll**: the Shia and Sunni rows are `modelled` (nobody counted sect at any level); the
  bare-islam remainder `derived`. Roll `NOWHERE`, as every other row: the brief asked for a roll
  back to `islam` as Iran and India have, but those two counted Muslims at the drawn unit and
  Bahrain counted them for the country only, so a roll to `islam` would put Bahraini Muslims by
  governorate on the map under `inferred dots: not shown` while the foreign Muslims beside them
  vanish. Modelled rows never roll anyway (`tools/check_rollup.py`, §7b). Bahrain still empties.
- **Not split**: foreign Muslims (their sect is not measured; the Gulf foreigner rule), and the
  Shia into schools (`islam.shia.jaafari` exists; nothing measured it, though Bahrain's Shia are
  Ja'fari) or the Sunni into Maliki and Shafi'i.
- Text: `note_public`'s Bahraini paragraph rewritten (names both endowments and the survey, says
  nobody counted sect); `source`, `basis`, `how` and `note` extended; top text about 71 words.
