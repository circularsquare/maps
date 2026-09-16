# Mauritania (`mr`)

**Drawn 2026-09-15** by session `cb8b206e-mr`, resumed from `d743fc47-mr`'s checkpoint-A handoff. 15
wilayas, 4,927,531 people, every row `modelled`. Rulings: `ask/RULINGS.md` 2026-09-15 and 2026-09-16
(draw it on a compiler's figure, foreigners by wilaya, no rings). Code: `sources/mr_geo.py`,
`sources/mr_grid.py`, `sources/mr.py`, `taxonomy/mr2023.py`, `countries/mr.py`. `sources.md`
§mr-2026-09-15.

## 1. Nothing asks religion

- RGPH 2013 form: no religion item (`sources.md` §11aq).
- RGPH 2023 (RGPH-5): the form is not published. ANSADE's media library
  (`admin.ansade.mr/wp-json/wp/v2/media?search=questionnaire`, 2026-09-15) returns nothing. Its
  sixteen thematic reports (`?search=Theme`) are population, age and sex, marriage, fertility,
  mortality, education, socioeconomic characteristics, agricultural households, households,
  housing, children and youth, women, disability, the elderly, migration and foreigners; none is
  on religion. Thème 16 §16.2.2 lists what was asked of everyone (Q07 nationality; marital status,
  education, health, occupation) and the four questions for foreigners 15 and over. Tome 1 does not
  reproduce the form.
- Arab Barometer VII and VIII: 3,200 Mauritanian rows, every `Q1012` empty (`queue.md` Closed row).
  Afrobarometer does not put the question in Mauritania; DHS 2019-21 has no item. MICS not checked.

## 2. Compiler figures, as read

**Pew Research Center, Religious Composition 2010-2020** (`data/raw/estimates/pew.zip`), base
everyone living in the country. Column order confirmed against the unrounded counts' header.

| year | population | Muslims | Christians | unaffiliated | other religions | Jews |
|---|---:|---:|---:|---:|---:|---:|
| 2010 | 3,390,965 | 99.125% | 0.271% | 0.102% | 0.499% | 0.003% |
| 2020 | 4,600,131 | 99.185% (4,562,658) | 0.234% (10,754) | 0.107% (4,899) | 0.472% (21,717) | 0.002% (102) |

**CIA World Factbook**, discontinued 2026-02-04; the `factbook/factbook.json` GitHub mirror
(`africa/mr.json`) reads *"Muslim (official) 100%"*. Anita's 99.9% was not found. Not read: the US
State Department's 2023 religious freedom report (WebFetch 403) and Constitute Project's
`Mauritania_2017` (404), so `note_public` quotes neither.

## 3. Foreigners: ANSADE, RGPH-5 *Thème 16*

`admin.ansade.mr/wp-content/uploads/2026/01/Theme-16-Population-etrangere-vivant-en-Mauritanie.pdf`,
text layer, parsed by `sources/mr.py` (every figure below is asserted there).

- **Tableau 16.1:** 4,927,532 counted; Mauritanians 4,801,600; foreigners 125,933. The wilaya rows of
  Thème 1 give 4,927,531, so nationals as drawn are 4,801,598.
- **Tableau 16.2**, checked against A.5 (urban/rural): Mali 83,681; Senegal 22,906; other African
  11,660; rest of the world 2,920; other Arab 2,280; Morocco 1,602; Europe 600; Algeria 284.
  Nationality is national only.
- **Tableau 16.5:** refugees and asylum seekers 46,800 (also the indicators page).
- **Tableau 16.6, per wilaya:** Hodh Chargui 59,555; Hodh El Gharbi 1,876; Assaba 1,256; Gorgol
  2,347; Brakna 1,828; Trarza 3,940; Adrar 230; Dakhlet Nouadhibou 6,671; Tagant 385; Guidimakha
  2,963; Tiris Zemmour 2,199; Inchiri 1,281; Nouakchott-Ouest 18,638; Nouakchott-Nord 8,946;
  Nouakchott-Sud 13,818.
- Thème 16 (p.15) puts Hodh Chargui's figure down to the refugees there; **Thème 15** (Tableau 17,
  p.39, 116,288 "immigrants" by wilaya, a different measure) to the Mbera camp's Malian refugees.
  Neither report crosses nationality with wilaya.
- **Stated undercount (§16.2.3):** homeless migrants, mobile gold miners, foreigners declaring
  themselves Mauritanian during a residence-card campaign. Unquantified.

## 4. The construction, decided

Pew's shares include foreigners, so Pew on the country plus a foreigner layer counts foreign
non-Muslims twice. **Drawn: every national on `islam`, every non-Muslim dot a foreign resident.**

- Why not Pew's residual on nationals: no source says anything about nationals who are not Muslim,
  and the residual would be mostly Pew's 21,717 `other religions`, a cell nothing explains. The
  Factbook's entry is 100%.
- **Witness, band written before the run (0.5-2.0):** the foreigner layer alone gives 6,136
  Christians, 0.125% of the census, against Pew's 0.234%: **0.53**. Drawn overall 99.790% Muslim.
  (Before §7: 7,031 Christians, 0.61, 99.742%.)
- Reversing it: change the nationals block in `sources/mr.py`, rerun it and the scatter.

**Foreign residents** (`sources/mr.py`):

- The 46,800 refugees are placed in Hodh Chargui as Malians first (Hodh Chargui's 59,555 and Mali's
  59,876 rural both hold them); the other 79,133 take the national mix without them in every
  wilaya. Without this step 47% of the country's Europeans would be drawn in Hodh Chargui. The
  refugees take northern Mali's census shares, not Pew's Mali row (§7).
- **Inside the census's `other` groups, UN DESA International Migrant Stock 2024** (CC BY 3.0 IGO,
  `data/raw/mr/undesa_pd_2024_...xlsx`), Mauritania as destination, where the named origins reach
  half the census group (`COVER_BAR`, set before computing any religion figure):
  - other African 11,660: Guinea 3,650, Guinea-Bissau 1,415, Benin 647, Côte d'Ivoire 542,
    Cameroon 460, Ghana 240, Niger 174, Togo 168, DR Congo 104, Gabon 97, Chad 58 (7,555, 65%);
  - other Arab 2,280: Syria 651, Saudi Arabia 357, Tunisia 290, Egypt 263, Palestine 188, Libya 161,
    Lebanon 138, Iraq 131, UAE 78, Kuwait 35 (2,292, 101%);
  - Europe (297 named, 49.5%) and the rest of the world (709, 24%) fall under the bar and take
    Pew's All Europe, and All Asia-Pacific + Latin America-Caribbean + North America summed.
  - DESA's 32,947 from **Western Sahara** are more than the census's two `other` groups together, so
    they are not in the census's foreign count under either and are left out.
- Pew 2020 per nationality through `taxonomy/origin_religion.py`; Muslim branches folded to
  `islam` as Morocco; unplaced `other religions` on the new `other.mr` (636 people; 991 before §7).
- **As drawn** (since §7): islam 115,588; Latin Catholic 3,601; unaffiliated 2,497; Protestant 2,358;
  Hindu 638; other.mr 636; Buddhist 178; African traditional 148; Afro-diasporic 103; the rest under
  60 each.
- **The ruling's test** (non-Muslim foreigners few or predictably placed): 10,345 people, 0.21% of
  the country. With the refugees placed first, 58.5% of them are in Nouakchott's three wilayas and
  Dakhlet Nouadhibou and 19.2% in Hodh Chargui. It holds, so no ask. (Before §7: 12,711, 47.6% and
  34%, Hodh Chargui's share swollen by Pew's Mali row on the refugees.)

## 5. Geography and placement

- **Base:** Thème 1 (`.../2026/04/Theme-1-...RGPH2023.pdf`) p.3 urban, rural, nomad and total per
  wilaya; its rounding is pinned (four wilayas off by one, columns by up to 2). Tableau 1.5's
  areas sum to 1,036,000 km2 against its own national row's 1,030,700. Tableau A.1.4 gives 63
  moughataas; three wilaya totals differ from p.3 by one. `ansade.mr/wp-content/...` returns the
  site's HTML shell; the files are under `admin.ansade.mr`.
- **Boundaries:** COD-AB `cod-ab-mrt` `mrt_admin1.geojson` (15) and `mrt_admin2.geojson` (63). The
  `_em` layers are not used; their ADM2 names Maal `0`. Area per wilaya against Tableau 1.5, over the
  national ratio: 0.955 (Tiris Zemmour) to 1.061 (Guidimakha). Moughataas join on pinned names
  (Nouakchott-Nord prints them in the opposite order to COD's pcodes).
- **Kontur `MR`** (2023-11-01): 1.015 of the census; rank witness over 63 moughataas +0.898, best of
  20,000 shuffles +0.519; no seat lost. **13.9% of Kontur's people sit in a different moughataa from
  the census**, most of it Nouakchott: Tevragh Zeina 4.09x, Ksar 2.73x, Sebkha 2.41x against El Mina
  0.42x, Dar Naim 0.46x, Riyad 0.51x. So every hex is calibrated to its moughataa's census count.
- **One raw block at Kontur's cap**, 10 hexes and 244,404 people, 86% of Toujounine's Kontur and 31%
  of Dar Naim's. Left as Kontur has it: capping to the 3 km ring's median (2,378/km2) left the block
  17,989 people and calibration then pushed Toujounine's edge hexes to 76,322/km2. Toujounine is one
  commune, so nothing finer says where its 303,882 live. `kontur_cap.py` skips the calibrated layer
  (densities above Kontur's limit), so the raw scan in `mr_grid.py` is the only one.
- **The Mbera camps** (33.4% of Bassiknou moughataa, Thème 1 printed p.13): of Bassiknou's communes
  (Bassiknou 21,252, Elmegve 15,232, Vassala 79,508, Dhar 7,345) only Vassala (Fassala) can hold
  them. **GeoNames' `Mbera` point (58,985) lies in COD's El Megve** and cannot be the camps as
  counted. Kontur holds 184,907 in COD's Vessale against the census's 79,508, so after moughataa
  calibration about 30,000 of Bassiknou's weight too many sit there, within 60 km of where the
  census counted them. Not fixed; commune calibration would need COD's 238-commune `_em` layer
  joined to Tableau A.1.4's communes.

## 6. Reopen if

- A census or survey asks Mauritanians their religion, or ANSADE publishes nationality by wilaya.
- Commune-level calibration (above), if Bassiknou or Nouakchott's placement is ever questioned.
- UNHCR publishes areas of origin with figures, for Mbera or for the 2023-24 arrivals outside it (§7).

## Review, 2026-09-15 (cb8b206e-rev3)

Full pass. `check_md` clean, `built_countries --check` ok, `check_rollup mr` clean (all modelled),
`coverage.py` verify ok over 186 countries with the `other.mr` line in. Every `note_public` figure
recomputes (125,933 is 2.6%; the eight groups sum; 12,711 non-Muslims; 99.742% Muslim). Nationals on
`islam` and not on Pew's residual, the witness band, DESA inside the `other` groups, `other.mr` and
the moughataa calibration: agreed. §14: nothing; the refugees are drawn at wilaya grain and the camp
is public. Screenshot clean.

- **The refugees take Mali's national mix, and they are from northern Mali.** Pew 2020's Mali
  shares on the 46,800 put **2,746 non-Muslims** in Hodh Chargui (5.87% of them), **21.6% of the
  12,711** drawn in the country: 1,580 of its 3,652 unaffiliated, 1,211 of 4,070 Latin Catholics,
  910 of 2,830 Protestants (counted off `mr_foreign.csv`, taking the other 12,755 foreigners there at
  the 12.59% non-Muslim mix every other wilaya has). Mali's own 2022 census, already drawn here
  (`data/normalized/ml.csv`), has Tombouctou 99.67% Muslim, Kidal 99.64%, Gao 99.51% and Mopti
  99.10%; UNHCR describes Mbera's refugees as from northern Mali and publishes an *Areas of origin
  of Malian refugees living in Mbera camp* map (data.unhcr.org/en/documents/details/79788, not
  opened). Whichever of those régions they come from, they give 154 to 421 non-Muslims, not 2,746.
  The error is one way and sits in one wilaya, so §4's "34% in Hodh Chargui (Mali's row). It holds"
  passed the ruling's test on a mix that is wrong for these people. Put back in `queue.md` (the
  Maghreb table's `mr` row); not rebuilt.
- **The census's refugees are about half of UNHCR's, and ask 033's rule is not applied.** Tableau
  16.5 has 46,800 refugees and asylum seekers. UNHCR's Mauritania factsheets give close to 93,000
  Malian refugees in Hodh Chargui in September 2023 and, on 29 February 2024, 99,000 in Mbera camp
  and about 181,000 Malians in the wilaya after more than 55,000 arrivals in 2023. Some of the
  difference may be §16.2.3's foreigners who said they were Mauritanian, already drawn on `islam`.
  The gap sweep skipped `mr` while its builder was live (runlog, supervisor line after rev1), so
  `gap` names only non-Muslim Mauritanians and there is no `gap_share` for foreigners the count
  missed. In the same queue note.
- `taxonomy/mr2023.py`'s docstring said 4,801,599 nationals; the wilaya rows and the note give
  4,801,598. Fixed, docstring only.

## 7. Refugees' religion: northern Mali's census, not Pew's Mali row (cb8b206e-mr2, 2026-09-15)

On the review above and the supervisor's call in `queue.md`'s Maghreb table.

- **Origin.** UNHCR, *Areas of Origin of Malian refugees living in Mbera camp as of 17 Sep 2018* (map
  `mli_refugee_origins`, `data.unhcr.org/en/documents/details/79788`, file at
  `.../documents/download/79788`, opened 2026-09-15). A map with no table: Mali's nine régions
  before the 2023 reform in four classes of "% of total population living in M'Bera", each
  labelled by its upper bound. Tombouctou is alone in "<= 89.7%", Mopti alone in "<= 6,5 %", Ségou
  alone in "<= 3,6 %"; Kidal, Gao, Kayes, Koulikoro, Bamako and Sikasso are "<= 0,1%". The three
  single-région bounds sum to 99.8%, so they are read as those régions' shares, and the last 0.2%
  is split equally over the six (`ORIGIN_2018` in `sources/mr.py`). The map is five years older
  than the count, and 55,000 more arrived in 2023, mostly outside the camp (§8).
- **Shares.** `data/normalized/ml.csv` (RGPH 2022, 20 régions) pooled into each old région by
  `sources/ml_geo.py`'s `PARENT`: Tombouctou with Taoudenni 99.69% Muslim, Mopti with Douentza and
  Bandiagara 96.98%, Ségou with San 90.85%. Nodes as `taxonomy/ml2022.py` maps them, Catholique on
  Latin Catholic and `other.ml` on `other.mr`; `christianity.other` added to `mr`'s line in
  `coverage.py`, since `origin_religion.nodes()` never emits it.
- **Result.** 379 of the 46,800 not Muslim (0.811%), against 2,745 (5.866%) on Pew's Mali row. The
  country: 10,345 non-Muslims (was 12,711), 6,136 Christians (7,031), 2,497 unaffiliated (3,652),
  99.790% Muslim (99.742%). The witness moves from 0.61 to 0.53, still inside the band written
  before the first run. Mali's other 36,881 residents keep Pew's Mali row.
- **Calls.** Pooling each old région takes in Bandiagara's and San's Christians, who are not the
  likeliest people to be in Mbera; the same-named 2022 régions alone give about 190 non-Muslims
  instead of 379. Pooled, because the map names the old régions and nothing says which part of
  each. Not used: the March 2024 factsheet's *Region of origin (as identified by key informants)*
  inset (§8's factsheet), four classes (51+, 11-50, 1-10, 0) with Tombouctou in the top one, for
  the 2023-24 arrivals and with no shares.

## 8. Refugees the census missed, ask 033 (cb8b206e-mr2, 2026-09-15)

On Anita's ruling (`ask/RULINGS.md` 2026-09-15, ask 033), which the gap sweep (`sources.md`
§gapsweep-2026-09-15) skipped here while the builder was live.

- **The count.** RGPH-5 enumerated from 25 December 2023 to 8 January 2024 (Thème 1 §1.2.3, PDF
  p.11). Tableau 16.5: 46,800 refugees and asylum seekers, every nationality. Thème 1 (printed
  p.13): the Mbera camps are 33.4% of Bassiknou moughataa's 123,337, so about 41,200 people of
  every nationality.
- **UNHCR, end of 2023.** Refugee Data Finder API,
  `api.unhcr.org/population/v1/population/?year=2023&coa=MRT&coo_all=true&cf_type=ISO` (read
  2026-09-15; without `cf_type=ISO` it returns no rows): 112,549 refugees and 5,927 asylum seekers
  in Mauritania from 32 origins, **118,476** together; Mali 110,997 + 3,521. End 2022 106,370, end
  2024 162,277.
- **UNHCR, the camp.** *Mauritania Factsheet, March 2024* (`data.unhcr.org/en/documents/download/107472`,
  opened): more than 55,000 arrivals in 2023, 41,000 of them "in over 90 locations in the Hodh
  Chargui region" and 14,000 in Mbera camp, which "bring the camp's population to almost 100,000
  people"; at 29 February 2024, 109,000 registered Malian refugees in Hodh Chargui and 19,000
  registered refugees and asylum seekers elsewhere, and an estimated 181,000 Malians in Hodh
  Chargui, 99,000 in the camp and 82,000 outside it. The September 2023 factsheet (ReliefWeb, with
  the review's "close to 93,000") returned 403 and was not read.
- **Undrawn, not drawn as Mauritanians.** A refugee in the camp who told the census they were
  Mauritanian is still inside the camp's census population. That population is about 41,200 where
  UNHCR had almost 100,000, so most of the difference is people the census did not count, in
  neither the base nor the dots. Outside the camp the census cannot separate them: some of 2023's
  41,000 who settled in villages may be among the nationals drawn on `islam` (§16.2.3's
  residence-card reason), and the 8,000 Mauritanian returnees are nationals anyway. So the figure
  below is a little high, by an amount nothing measures.
- **Figure.** UNHCR's end-2023 registered total, dated inside the count, less the census's 46,800:
  **71,676**; share 71,676 / (4,927,531 + 71,676) = **0.01434**. Registered, not the 181,000
  estimate, which adds the unregistered and the 19,000 who arrived in January and February 2024.
  Türkiye's `gap` is a registration count too (`sources.md` §gapsweep-2026-09-15).
- Written into `gap`, `gap_share`, `note_public`'s last paragraph and the internal `note`.
