# Cuba (`cu`): NORC 2016's national shares in every province

Built 2026-10-03 by session `fafd1067-cu`, reopening a country closed on its census forms
(sources.md §11ap, §scout-2026-09-15-negatives) under Anita's rulings of 2026-09-15 (Cuba is a
priority visual hole) and 2026-09-16 (a country no census asks is drawn on the best survey or
compiler figure, method disclosed). Code: `sources/cu.py`, `sources/cu_geo.py`,
`sources/cu_grid.py`, `taxonomy/cu2016.py`, `countries/cu.py`. Ask 044 (the Santería node).

## 1. Why this route

- **No census asks.** Forms 1899-2012 read (1970 and 1981 unread; nothing suggests either asked).
- **The survey with microdata.** NORC at the University of Chicago, *A Rare Look Inside Cuban
  Society: A New Survey of Cuban Public Opinion*. The project page
  `norc.org/research/projects/2016-cuban-public-opinion.html` links a **public use file and
  codebook with no login or form**: `NORC Cuba Public Use Files and Codebook.zip` (1,519,069
  bytes, fetched 2026-10-03, in `data/raw/cu/`; csv, dta, sav, sas). §11ap had recorded "reopen
  on NORC's microdata" without finding this link. No terms of use are printed in the codebook or
  on the page.
- **The other routes, and why they lose.** Pew Research Center's 2020 estimate (compiler;
  Christian 60.7%, unaffiliated 21.6%, other religions 17.4%) has no Catholic/Protestant split
  and no Santería row, and its basis for Cuba is not stated; it is printed as a witness.
  Bendixen & Amandi's 2015 poll for Univision and The Washington Post (1,200 interviews, 17-27
  March 2015) has no microdata; as reported, 44% not religious and 27% Catholic (Splinter's
  "Top 25 findings", read 2026-10-03; the Washington Post article returned 403). A search summary
  gave its Santería share as both 13% and 27%, so no Santería figure from it is used anywhere.
  CIPS's 1988-90 religiosity study (about 20,000 interviews) measured levels of religiosity, not
  religion named; Díaz Cerveto's summary (CLACSO, *15DP060.pdf*) prints no provincial table.

## 2. The survey

840 in-person interviews, adults 18+, national random-route sample stratified by three regions
(west, centre, east) and settlement size, main fieldwork 3 Oct to 26 Nov 2016 (April pilot kept).
Weighted (`finalwt`) to the 2012 census by age, sex, urban/rural. **Areas of eastern Cuba with
about 15% of the population were not sampled** after Hurricane Matthew. Question Z10, "What is
your religion, if any?"; the PUF variable `religion` merges the card's Evangelical, Protestant and
Christian (other) into code 3 and keeps DK/refused as blanks (5).

Weighted, of 835: Catholic 28.28% (n 223), Santería 16.91% (156), other Christian 6.47% (60),
believe in God but no religion 21.84% (184), atheist 1.13% (9), none of the above 24.37% (193),
other 1.01% (10). Every share is within a point of the published topline, whose base includes the
5 blanks (asserted).

**No region variable.** The file has `sector` (urban/rural) and nothing finer. Urban/rural test:
753 urban, 82 rural; chi-square 9.70 on 6 df, p 0.138. Weighted permutation per answer: Santería
+9.4 points urban, p 0.039 (0.27 after Bonferroni over 7); other Christian p 0.065; none of the
above p 0.113. **Not drawn**, so no proxy by urbanisation either (Sudan's call, `sources/sd.md`).
One national mix in every province.

## 3. Mapping (`taxonomy/cu2016.py`)

Catholic -> `christianity.catholic.latin` (as Venezuela); Santería -> **`afrodiasporic.santeria`,
new** (ask 044; fallback `afrodiasporic`); other Christian -> `christianity`; believer ->
`unchurched`; atheist -> `secular`; none of the above -> `unaffiliated`; other -> **`other.cu`,
new**. Reasons in REVIEW.

## 4. Population and boundaries (`sources/cu_geo.py`)

- **ONEI's own count, not COD-PS.** *Anuario Demográfico de Cuba 2024* (July 2025), Tabla 1.5,
  *población efectiva* at 31 Dec 2024: **9,748,007** in 16 units. onei.gob.cu fails on an
  expired certificate and returns 500; the gender portal `genero.onei.gob.cu/static/documents/
  informes/00-anuario-demografico-2024.pdf` serves the same file. COD-PS 2024 (UNFPA, ONEI's
  2015-2050 projection from the 2012 census) gives 11,306,203, 1.16x, because the projection
  assumed emigration would fade; rejected.
- **COD-AB `cod-ab-cub` v01 is GADM.** Province areas within 9% of ONEI's (area = Tabla 1.5
  people / Tabla 1.6 density) except **La Habana, 819 km2 against 728**, pinned; Artemisa and
  Mayabeque are 125 and 70 km2 short. Its municipal lines in Havana are about 2 km east of the
  real ones (COD's La Habana Vieja spans -82.339 to -82.286, over the harbour and Regla). With one
  national mix this moves only same-coloured dots between provinces.
- **Municipalities, for the placement witness only.** ONEI's *Tablero municipal* workbook
  (Wayback 20250606173952 of `publicaciones/2025-05/3-tablero-municipal.xlsx`), sheet Base, rows
  `Municipios`, 2023: 167 municipalities summing exactly to Tabla 1.5's 2023 column per province;
  Isla de la Juventud from Tabla 1.5. Joined to COD admin2 by folded name within province, five
  spellings pinned, and **two Mayabeque rows mislabelled in ONEI's workbook**: 2407 "Güines"
  (18,555) is San Nicolás and 2408 "Alquizar" (56,925) is Güines, by ONEI's DPA order and by
  Kontur (24,394 and 76,752 people).

## 5. Placement (`sources/cu_grid.py`)

- Kontur CU 2023, 62,620 hexes, 11.19M people. **Guantánamo Bay Naval Base dropped** (centroid in
  Natural Earth `USG`: 62 hexes, 3,583 people; spec §14.18), and not mentioned in `gap` because
  ONEI does not count it. 695 offshore hexes snapped within 2 km.
- **Kontur has lost most of the east, uniformly.** Raw Kontur over ONEI 2024, over the national
  ratio: Granma 0.17, Santiago de Cuba 0.39, Guantánamo 0.41, every other province 1.14-1.23.
  Per municipality it is uniform inside each (Granma 0.16-0.19, Santiago 0.37-0.44, Guantánamo
  0.32-0.44), so the hexes are **scaled to each province's ONEI count**, not each municipality's
  (Havana's misdrawn municipal lines would move people onto the wrong polygons). Raw Kontur had
  15.2% of its people in a different province from ONEI. Municipal rank witness: Spearman +0.795
  over 168, no shuffle of 20,000 reaches +0.283.
- **Two raw blocks at the cap, both capped** (`BLOCKS`): east of Havana harbour (5 hexes, 106,329
  people, to 4,549) and Marianao/La Lisa (9 hexes, 162,318, to 15,640). Centro Habana, the real
  dense core, does not reach the cap in Kontur.
- Calibrated densest hex 38,532/km2 in Songo-La Maya: a raw Kontur peak in Santiago province
  scaled by about 2.6. Several small eastern towns (Guisa, Campechuela, Jobabo) have single hexes
  over 20,000/km2 after scaling. Known, left: it is the shape Kontur gives, scaled.

## 6. Not drawn, and the gap

No `gap`: NORC sampled everyone living in Cuba, not citizens only; the 5 blanks are left out of
the base like Sudan's refusals; the unsampled east is drawn at the national mix (said in the
note). Children drawn at the adults' shares. The emigration since 2016 is not measured by religion.

## 7. Reopen if

- A Cuban census asks (the next has no date or form).
- NORC's restricted file has the three sampling regions (not asked for; would need a request).
- Another public-microdata survey inside Cuba appears (the Cuba Study Group and CubaData polls
  were not checked).

## Review, 2026-10-03 (`fafd1067-rev5`)

Full pass, nothing to change. Shares recomputed from `norc_cuba_2017.dta` independently of
`sources/cu.py`: identical to two decimals, and `cu.csv` carries them in all 16 provinces. The PUF
has no hidden region: `su_id` is a plain sequence 1-843, and no variable label names a place. The
hurricane exclusion is on codebook p.3 as the note puts it. Mapping matches precedent (the
believer box is `unchurched` in eleven other countries). One claim cannot be rechecked from disk:
the card's Jewish, Muslim and Buddhist boxes "which nobody chose" and the asserted `TOPLINE`
come from `Cuba Topline_FINAL.pdf`, which is not in `data/raw/cu/`. Map glance: dots on land,
Havana dense, the east populated.
