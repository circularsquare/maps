# Libya (`ly`)

Session `d743fc47-ly`, 2026-09-15, under a supervisor. Built on Anita's Maghreb rulings
(`ask/RULINGS.md`, 2026-09-15 and 2026-09-16) and the scouting in `sources.md`
§maghreb-2026-09-16. Code: `sources/ly.py`, `sources/ly_geo.py`, `sources/ly_grid.py`,
`taxonomy/ly2020.py`, `countries/ly.py`. No ask filed.

## 0. Outcome

Drawn at 22 districts, **6,872,674 Libyans** (the Bureau of Statistics and Census's estimate for
2020), at **one national share and mix in every district**, because seven non-Muslim answers place
nothing. Every row `modelled`. Non-Libyans are not in the base and are not drawn.

| | Libyans | share |
|---|---:|---:|
| Muslim (`islam`) | 6,865,655 | 99.898% |
| Christian (`christianity`) | 5,786 | 0.084% |
| No religion (`unaffiliated`) | 1,233 | 0.018% |

## 1. Sources

- **Arab Barometer**, waves V (1,962 Libyans, 2018-19), VI-1 (720), VI-2 (1,008), VI-3 (1,002) and
  VII (2,505, 19 February to 4 April 2022), `data/raw/arabbarometer/`. `Q1012` religion, `Q1`
  district, `WT`, `Q13` urban/rural (V, VII), `PSU` (V, VII), `Q1012A` / `Q1012A_MUSLIM` /
  `Q1012A_CHRISTIAN` follow-ups. No Libya wave has a nationality or citizenship column. Libya is
  not in wave VIII. Wave III (1,247) is read and left out (§3).
- **Arab Barometer Wave VII Technical Report** (`arabbarometer.org/wp-content/uploads/
  AB7_Technical_Report.pdf`, p.8, opened): Libya's target population is "Citizens aged 18 and
  above", sampling frame "Libyan Center for Documentation and Statistics 2012", 38 strata, 22
  governorates, 299 PSUs, response rate 44%. The V and VI reports were not opened.
- **BSC, *تقدير السكان الليبيين حسب المناطق لسنة 2020*** ("estimate of the Libyan population by
  region for 2020"), one page, linked from `bsc.ly`'s front page
  (`bsc.ly/wp-content/uploads/2024/02/…-2020.pdf`; the page's map links a 2023/06 copy of the same
  name, not compared), `data/raw/ly/bsc_libyans_by_region_2020.pdf`. 22 regions and a total,
  6,872,674. The front page's map repeats all 22 figures (read 2026-09-15, equal), and its tile
  "تقديرات السكان, 2020" says 6,875,635, 2,961 more, unexplained.
- **U.S. Census Bureau, Libya workbook** (HDX `libya-subnational-boundaries-and-tabular-data`,
  `Libya_uscb_202304.xlsx`, CC BY), `data/raw/ly/`. `Population`: the same 2020 estimate as
  `POP_TTL_20` and the 2012 National Population Survey by age; `Households`: the 2006 census's
  Table 1 by mahalla, summed to districts; `Nationality`: the 2012 survey's non-Libyans by district
  in continent groups. No religion or ethnicity anywhere (as `sources.md` §11af found).
- **COD-AB `cod-ab-lby` v01** (OCHA ROMENA; valid 2018-05-07, reviewed 2024-12-19),
  `data/raw/ly/lby_admin_boundaries.geojson.zip`. ADM2 is the 22 districts, p-codes LY0101-LY0322,
  which are BSC's codes (USCB's `NSO_CODE`); ADM1 is three regions.
- **Kontur population `LY` 20231101**; **GeoNames `LY`** (CC BY 4.0).
- **Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), Libya
  2020, everyone living in the country: Muslims 98.995%, Christians 0.525%, Buddhists 0.258%,
  Hindus 0.090%, other 0.082%, unaffiliated 0.049%, Jews 0.002%. Its 2010 row has Christians at
  2.632%.

## 2. The rulings applied

- 2026-09-15: draw the 99.x% Muslim countries with the best evidence for where non-Muslims are.
- 2026-09-16: foreigners only where the state publishes a count by province, on a national
  nationality mix (§7: not available); no IOM DTM; no presence rings for Jews or Ibadis.
- §14: leaving Islam carries legal and social risk in Libya. Nothing is placed below the national
  level, so the map points at no community; not escalated, as Tunisia.

## 3. The survey: pool, card, labels

**The card** (`ab.card`, lifted into `arabbarometer.py` from the Maghreb copies). Wave III's card
is Muslim, Christian, Other, `Jewish (Yemen only)`, don't know, refuse: no box for having no
religion. Its 1,247 Libyans answered Muslim (1,235) or don't know (12), so it adds nothing but a
different questionnaire; out. V offers `Atheist`, VI-1 to VII `No religion`, merged.

**Spellings.** V's `refused` merges into `Refused to answer` (5 dropped in all); V's `Atheist`
into `No religion` (1). No `Other` or `Jewish` answer in the pool (asserted).

**Pool as drawn:** 7,196 answered, 7,191 after refusals; 26 gave no district (VI's `Don't know`
and `Refused`, all Muslim), kept for the level. **7 non-Muslim:** V no religion (Tripoli); VI-2
Christian (Al Jabal al Akhdar, Benghazi, Misrata); VI-3 Christian (Misrata, Al Jabal al Gharbi);
VII Christian (labelled Jafara, follow-up `Just a Christian`). Weighted non-Muslim share by wave:
V 0.065%, VI-1 0.000%, VI-2 0.317%, VI-3 0.211%, VII 0.031%; pooled **0.1021%**, mix Christian
82.5%, no religion 17.5%. `NOTE`, asserted.

**Follow-ups.** No non-Muslim carries a Muslim follow-up (`tn.contradictions`). **Wave V's sect
item** (interviewer-logged) records 561 of 1,961 Libyans as `Alawi` and none as Maliki, beside
Sunni 716 and Just a Muslim 611; VII, whose card offers Maliki, records Maliki 286 and no Alawi.
Read as Malikis logged on a code the list did have (Yemen's Zaydi trap); nothing is drawn from it.
**Ibadis:** VII `Ibadi` 4 (Jafara 2, Murqub 1, Zuwara 1), V `Mozabite` 1 (Tripoli); none in Al
Jabal al Gharbi or Nalut, the Nafusa, where Ibadis are said to live.

**Districts.** Decoded by name in every wave (`LABELS`); each wave's code-to-label table is 1:1.
VI labels Al-Wahat (1, 6, 5) and Ejdabia (24, 24, 23) apart; both are BSC's "Ajdabiya and the
Oases" (LY0105). V has no Jufra respondent. **VII's codes 11001-11022 mean different districts
from VI's 11001-11023** (VI's 11001 is Tobruk, VII's is Derna), so neither is decoded on code.

**Wave VII's district labels do not fit the population** (`allocation`). Weighted respondents per
district against BSC 2020, Spearman: V +0.992, VI-1 +0.997, VI-2 +0.997, VI-3 +0.994, **VII
+0.853**. VII's weighted share over population share: Ajdabiya 3.70, Ghat 3.47, Jafara 2.28, Wadi
al Shati 1.75, Zawiya 1.57; Murzuq 0.29, Benghazi 0.32, Tripoli 0.41, Misrata 0.53, Sabha 0.63;
mean weights 0.9-1.1 in each, so the weights do not compensate. Swapping Tripoli with Jafara,
Benghazi with Ajdabiya and Murzuq with Ghat takes the summed absolute log misfit from 9.58 to 3.22
and leaves Misrata and Wadi al Shati off, so it is not a clean label swap and none is applied. PSU
ranges per label run consecutively except Misrata's (205-301). VII's answers and weights set the
level and mix; its labels are used for nothing (`GEO_WAVES` is V and VI). Trap added to
`playbooks/arabbarometer.md`.

**Held-out** (V and VI): r = +0.999 between weighted respondents and BSC 2020 over 22 districts,
best of 20,000 pairings +0.834. **Quota test:** one comparable pair (VI-2 against VI-3, 4 free
cells), p = 1.

## 4. What stands apart: nothing

- **Districts** (`standouts`, halves V+VI-1 and VI-2+VI-3): only Misrata has two non-Muslims, 0
  against 0.11 expected then 2 against 0.47 (P 0.072). None passes.
- **Towns** (`urban_test`, V and VII): 2 urban of 2 against 1.58 expected, P 0.614. Morocco's bar
  also needs the Afrobarometer, which has no Libyan round.

## 5. Units and population

`ly_geo.py`. The key is BSC's district code, COD's `adm2_pcode`. Witnesses: the PDF's 22 rows sum
to its printed total, each extracted line carrying the expected Arabic name (`PDF_ORDER`; the text
layer garbles سرت and مصراته); COD's and USCB's names under each code are the expected ones;
**USCB's copy of the estimate equals the PDF in 21 districts, and has Al Marj at 286,045 against
227,658**, the whole of USCB's 58,387 national excess (pinned). With USCB's figure Kontur read Al
Marj at 0.79 of its share; with the PDF's, 0.98. The rank witness is in §6.

BSC's names: "المنطقة الغربية" (the Western Region; its footnote lists Zuwara, Sabratha, Al
Ajaylat, Riqdalin and Al Jumayl) is COD's Zwara; "اجدابيا والواحات" is Ejdabia; "وادي الحياة" is
Ubari. The map prints Zuwara, Ajdabiya and Ubari.

COD's area against the 2006 census's land area (Households sheet): the eight north-western
districts, Sabha and Ubari within 3%; Derna 0.67, Ghat 0.74, Al Jabal al Akhdar 0.75, Al Marj 0.78,
Murzuq 0.79, Jufra 0.83, Sirte 0.87, Wadi al Shati 1.13, Benghazi 1.19, Ajdabiya 1.82. The desert
and eastern lines were drawn differently in 2006 and hold few people; the 2020 estimate matches
COD's polygons (§6).

**Why the 2020 estimate.** It is BSC's latest by region and within two years of every wave. No
census since 2006; searched 2026-09-15: bsc.ly's front page (the 2020 estimate, MICS7 2025 results
and a 2022/23 household budget survey, neither by population), UNFPA Libya's data page (a national
demographic survey "at municipality and national levels" in preparation, a 2020 census planned
and not held), one web search for regional estimates 2023-2025 (nothing). The 2012 survey
(5,363,369, all residents) and the 2006 census (5,657,692) are older and include non-Libyans.

## 6. Placement

`ly_grid.py`. Kontur `LY` 56,929 hexes, 6,891,138 people; 409 centroids outside every district
(45,523 people, the coast), 405 snapped within 2 km (42,784 within 0.5 km), 4 hexes and 112 people
dropped. Kontur over the estimate 1.003. **Join witness:** Spearman over 22 districts +0.999, 0 of
20,000 shuffles reach it (best +0.827). Per district over the national ratio: Tobruk, Derna,
Murqub, Al Jabal al Gharbi and Al Marj 0.98 to Kufra 1.06 and Sabha 1.05. Kontur models everyone
in Libya and the estimate counts Libyans, so the small excess in Kufra, Sabha and Jufra (11.5%,
10.6% and 10.0% non-Libyan in the 2006 census) is the expected direction.

**The seat check needed a gate.** GeoNames gives Libyan seats their municipality populations
(Tripoli 1,302,947, Al Khums 201,943), and Kontur spreads Tripoli's 842 km2 district near its
average density, so within 5 km Al Khums read 0.10 (0.20 within 10 km), Tripoli 0.10 (0.33),
Benghazi 0.11 (0.31), Misrata 0.25 (0.52) and Derna 0.26 (0.45) of GeoNames, all in districts at
0.98-1.02. A seat counts as a hole only in a district under half its share (Morocco's gate). No
hole. `kontur_cap.py ly`: no block at the cap.

## 7. Foreign residents: not drawn

The base is Libyans and so is the survey, so non-Libyans are outside both. **`gap` and `gap_share`
carry them since the ask 033 sweep** (§11): 826,537 in mid-2020 by UN DESA's *International
Migrant Stock 2024*, `gap_share` 0.10735. The figures that disagree with it: the 2006 census's
359,540 (6.35% of residents), the 2012 survey's 187,372 (3.5%), and IOM's 2026 baseline of 943,748
migrants (not used; `sources.md` §maghreb-2026-09-16). UN DESA's is an estimate, not a count.

The ruling's construction needs a state count by province and a national nationality mix. What
exists: the 2006 census's non-Libyans by mahalla, with no nationality in the USCB workbook; the
2012 survey's non-Libyans by district in continent groups (Arab 152,749, African 28,282, Asian
4,903, European 1,339, American 59; Tripoli's 88% Arab, Sabha's 66% African). No table found gives
nationality by country, for the country or any district, so Pew's per-country compositions have
nothing to multiply, and a continent average stands for no one: Libya's African residents are
mostly from its Sahel neighbours, whose religion is not their continent's. And the 2012 stock predates the migration of the
2010s. **Not drawn.** Pew's 0.525% Christian for everyone in Libya, against 0.084% drawn for
Libyans, is in `note_public` so the reader sees the size of what is missing.

## 8. Not drawn, and REOPEN

- **Foreigners by nationality and district.** BSC's planned demographic survey; the 2006 census's
  nationality tables (bsc.ly, not searched past the front page); the 2012 survey report's tables
  6.1-6.3 and its nationality table (USCB cites them; not opened).
- **MICS7 Libya (2025, BSC with UNICEF).** Whether its household questionnaire asks religion or
  nationality was not checked.
- **Ibadis.** 5 answers, none in the Nafusa. The Libyan Tmazight Congress's 300,000 to 400,000
  (via Human Rights Watch, 2017) is an advocacy figure, not opened. No rings.
- **Wave VII's labels.** The Arab Barometer's own Libya VII report or codebook might explain the
  allocation; not opened.
- **The V and VI technical reports**, for whether those waves also sampled citizens only.

## 9. Calls someone might reverse

1. One national share everywhere; nothing passes at district or urban.
2. Pool V-VII, with wave VII in the level and mix but not in the geography tests.
3. III out on the card.
4. The base is Libyans (BSC 2020), non-Libyans a stated gap sized by UN DESA's mid-2020 estimate
   (826,537, `gap_share` 0.10735), not its mid-2024 one (897,751), so the hole and the base share a
   year (§11).
5. Al Marj from BSC's PDF (227,658), not USCB's copy (286,045).
6. The seat check gated on the district ratio.

## 10. Review, 2026-09-15 (cb8b206e-rev1)

Full pass. `check_md` clean, `built_countries --check` ok, rollup clean (all modelled).

- **Figures.** Every `note_public` figure recomputes off `ly.csv`: 6,872,674, 0.102%, grain
  312,394. The CSV's mix is 82.4 / 17.6 after rounding counts, against the weighted pool's 82.5 /
  17.5; that difference is fine.
- **Mapping.** Same as `dz`, `ma` and `tn`, with no new node. Wave V's `Alawi` read as Malikis,
  with nothing drawn from it; agreed.
- **§14.** No ask, agreed.
- **The shared `ab.card()`.** Only `sources/ly.py` calls it. `dz`, `ma` and `tn` still use their
  own copies, so it changes nothing for any other country.
- **Screenshot.** Clean: dots on the coast and the Fezzan oases, none in the sea or on the sand
  seas.
- **`gap` against ask 033, already queued.** Anita's ruling (`runlog.md` 637) came in after this
  build. It asks for an estimate of today's non-Libyans, with a `gap_share`.
  - The entry still gives the 2006 count with no `gap_share`, and §7 and call 4 still argue
    against having one.
  - Not edited here: the ruling leaves open which estimate to use. The supervisor's non-national
    `gap_share` sweep has Libya first (`runlog.md` 654).
  - When the sweep lands, §7, call 4 and the note's 2006 sentence go stale with it.
- **Soft, not edited.** The note bolds **82.5%** Christian, a mix that rests on 6 answers against
  1. The note already says "6 Christian and 1 atheist", so a bolded decimal claims more precision
  than seven answers carry. Dropping the bold, or the decimal, would read truer.

## 11. Non-Libyans sized for the bar, ask 033 sweep (cb8b206e-gap, 2026-09-15)

On Anita's ruling (`ask/RULINGS.md` 2026-09-15); the sweep's record is `sources.md`
§gapsweep-2026-09-15.

- **Source.** UN DESA Population Division, *International Migrant Stock 2024*,
  `undesa_pd_2024_ims_stock_by_sex_and_destination.xlsx` (un.org, CC BY 3.0 IGO, opened 2026-09-15),
  Table 1, Libya: **826,537 at mid-2020**, 897,751 at mid-2024. Type of data `C R`: built from data
  on foreign citizens, with UNHCR's refugees added. So these are non-citizens, not the foreign-born.
  The workbook has no country note saying which Libyan figures the model starts from. Table 3 gives
  11.73% of UN DESA's own 2020 population (7,045,399), which is not BSC's base.
- **Share.** 826,537 / (6,872,674 + 826,537) = **0.10735**: the bar adds the estimate to BSC's
  Libyans. Mid-2020, because the base is 2020; mid-2024 on the same base would be 11.6%. The note
  gives both figures.
- **Why not the others.** IOM's 943,748 (2026) is barred by IOM's terms (Anita, 2026-09-16); the 2006
  census and the 2012 survey predate the migration of the 2010s.
- **Note.** The 2006 sentence is replaced by the UN estimate. The bolded **82.5%** Christian is now
  "83% Christian and 17% with no religion", without bold (§10's soft point). §7 and call 4 are
  updated. Nothing rebuilt; the next tail carries the note and the bar.
