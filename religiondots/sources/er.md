# Eritrea (`er`)

**Drawn 2026-10-03** by session `fafd1067-er`, under a supervisor. 6 zobas, 3,291,271 people (UN WPP
2024 for 2020), every row `modelled`, one national mix. Rulings: `ask/RULINGS.md` 2026-09-15 (priority
holes) and 2026-09-16 (Mauritania: draw on the best survey or compiler figure, with the method said).
Ask 053 (§14). Code: `sources/er_geo.py`, `sources/er_grid.py`, `sources/er.py`, `taxonomy/er2010.py`,
`countries/er.py`; `christianity.oriental.eritrean` promoted from an ASARB leaf to a branch in
`taxonomy/branches.py`. `sources.md` §er-2026-10-03. Earlier records: `sources.md` §11aq (closed,
"EPHS 2010 and EDHS 1995/2002 ask and never cross religion with zoba"), §scout-2026-10-03-negatives
(reopened at one national mix from EDHS 2002).

## 1. What exists

- **No census, ever.** The State Department's 2023 report: "Reliable population data on the country
  is difficult to gather. There are no reliable figures on religious affiliation."
- **Three surveys ask religion**, each of women 15-49 and in 2002 and 2010 of men too; each report
  prints it once, nationally, in Table 3.1/3-1, beside a zoba and an ethnic-group block it never
  crosses with religion. Read 2026-10-03, every page matching `religi|muslim|orthodox`:
  - **EPHS 2010** final report (`data/raw/er/ephs2010_final_report_v4.pdf`, 574 pp, NSO and Fafo,
    from `afro.who.int/sites/default/files/2017-05/ephs2010_final_report_v4.pdf`; ReliefWeb 403s).
    Table 3-1 (PDF p.70): Women CORE 10,238, Women ALL 30,224 (core plus the maternal mortality
    module), Men 15-49 4,299, weighted. Women ALL: Orthodox 55.8, Catholic 4.2, Protestant 0.8, Muslim
    39.0, traditional 0.2. Men: 60.0, 4.7, 1.0, 33.9, 0.2. Card (PDF pp.460, 524, 555) has six codes,
    the sixth `OTHER (SPECIFY)` with no row printed (CORE 3 and ALL 7 weighted women short of their
    totals). The only other religion hits are FGC and family-planning reasons.
  - **EDHS 2002** (`data/raw/er/dhs_fr137.pdf`, DHS FR137): Table 3.1 (PDF p.57), 8,754 women:
    Orthodox 57.7, Catholic 4.6, Protestant 0.7, Muslim 36.5, traditional 0.4, other 0.1, missing 0.1.
    Appendix A: the frame was the Ministry of Local Government's 2000 list of villages and towns.
  - **EDHS 1995** (`data/raw/er/dhs_fr80.pdf`, FR80): downloaded, not re-read; §11aq read its Q118/Q124.
- **No microdata**: the DHS API lists no Eritrean dataset (§11aq). No Afrobarometer, Arab Barometer
  or other survey round (§11ah).

## 2. Why the survey and Pew disagree by 15 points

Pew 2020 is Muslim 51.67%, Christian 46.67%, and its 2010 row is 51.28/47.07. **Pew's 2025 Appendix A
("Sources", p.14) names the World Religion Database as its only source for Eritrea's composition in
both 2010 and 2020**, with UN WPP 2024 for age and sex and "Data unavailable" for switching. The State
Department quotes the same WRD 52/47. The WRD ascribes religion by ethnic group
(`[[reference_arda_wrd_ascription]]`), so Pew's figure is the ethnic shares times assumed religions,
not anyone's answer. (Pew's 2012 report is remembered as about 63% Christian, close to EDHS 2002,
which would make the change a change of source; not re-read this session.)

The surveys are self-identification, two rounds eight years apart agree (Muslim women 36.5% then
39.0%), and they are the only measurement. Reasons they could run low, none closing 15 points:
- **Ages.** Women and men 15-49 only; 47% of the household population is under 15 (Table 2-1).
  Fertility is higher outside Maekel (TFR by zoba, Table 4-2: Anseba 5.7, Gash-Barka 5.4, Semenawi
  Keih Bahri 5.4, Debub 5.0, Debubawi Keih Bahri 4.2, Maekel 3.4), but no table gives it by religion.
- **Coverage.** The frame is the zobas' own 2010 lists; the ethnic shares sampled (Table 3-1, women
  ALL: Tigrinya 61.2, Tigre 23.2, Saho 4.4, Nara 3.4, Bilen 3.2, Afar 2.1, Kunama 1.3, Hedareb 1.0,
  Rashaida 0.1) sit against the Factbook's 2021 estimate (Tigrinya 50, Tigre 30, Saho 4, Afar 4,
  Kunama 4, Bilen 3, Hedareb 2, Nara 2, Rashaida 1). Pastoral Rashaida and Afar look under-listed.
- **Men.** Men are 33.9% Muslim against women's 39.0%; the report itself says migration moves both
  the sex ratio and the religious mix.

**Call: draw the survey, print Pew as a witness.** EPHS 2010 over EDHS 2002 because it is newer,
three and a half times larger and asks men. Reversal: change the mix in `sources/er.py`.

## 3. The construction (`sources/er.py`)

Women ALL and Men 15-49 shares, each over its own answers, combined at the survey's de facto household
population aged 15-49 (Table 2-1: 18,398 men, 31,710 women, 36.72% men). At an even split Muslims
would be 36.51% instead of 37.18%. Drawn: Orthodox 57.39%, Muslim 37.18%, Catholic 4.36%, Protestant
0.87%, traditional 0.20%. The same mix in every zoba. Read back from the text layer each run; the
note's figures are asserted. EDHS 2002's women are asserted within 3.5 points of 2010's.

Mapping (`taxonomy/er2010.py`): `Orthodox` to `christianity.oriental.eritrean` (et2007's reasoning;
promoted to a branch so a non-ASARB source can reach it; same label, no new legend row); `Catholic` to
bare `christianity.catholic` (the Eritrean Catholic Church is Ge'ez-rite Eastern Catholic, but the
card has one box); `Protestant` to `christianity.protestant` (the Lutheran church plus unregistered
churches); `Muslim` to `islam`; `Traditional believer` to `indigenous.african`.

## 4. Population (`sources/er_geo.py`)

- **National**: UN WPP 2024 for 2020, 3,291,271, from Pew's file (which names WPP 2024). The US
  government's estimate is 6.3 million (mid-2023, State Department; the Factbook's 6,416,435 for 2025).
  The note gives both. Not a count either way.
- **Zoba shares**: EPHS 2010 Table 2-16, weighted de jure population (150,297): Debubawi Keih Bahri
  1.52%, Maekel 21.64%, Semenawi Keih Bahri 11.03%, Anseba 14.84%, Gash-Barka 23.26%, Debub 27.71%.
  Asserted within 1 point of the 2010 frame's household shares (Table A-3 column 2). COD-PS
  `cod-ps-eri` is the NSO's 2001 figures (2,908,795), printed as a gross-error witness (6 points):
  Semenawi Keih Bahri 16.49% in 2001, Gash-Barka 19.95%, Maekel 19.00%. Not used: nine years older.
- **Boundaries**: COD-AB `cod-ab-eri` v01 (valid 2020-04-27), 6 zobas, joined on p-code with names
  asserted. No office area table to witness against.

## 5. Placement (`sources/er_grid.py`)

- **Kontur `ER` (2023-11-01) rejected.** 3,877,037 people. Per zoba against the survey's share: Maekel
  0.45, Debub 0.82, Anseba 0.88, Gash-Barka 0.96, Debubawi Keih Bahri 1.83, Semenawi Keih Bahri 2.68.
  Three blocks at the 46,200/km2 cap: Massawa (12 hexes, 268,920), Ghinda (6, 140,872), Karora (3,
  94,801, a border village). Within 5 km of GeoNames towns it holds Keren 290,550, Massawa 308,810,
  Barentu 114,575, Ak'ordat 58,689.
- **Meta's 2020 high-resolution population grid** (Data for Good at Meta and CIESIN, HDX
  `highresolutionpopulationdensitymaps-eri`, `eri_general_2020.csv`, CC BY 4.0), 283,923 points and
  3,546,847 people in the zobas. Its zoba split is no better (Maekel 0.67, Semenawi Keih Bahri 2.26),
  so it places only: binned to 0.008-degree cells (median 0.76 km2), cut to the zoba, scaled to each
  zoba's population. After scaling, people within 5 km: Asmara 332,894, Keren 66,932, Af'abet 54,399,
  Massawa 51,376, Mendefera 48,590, Dek'emhare 41,972, Barentu 34,328. 10% of Meta's people lie
  outside every Kontur hex, so Kontur's geometry was not reused.
- **Borders.** The CSV runs into Ethiopia and Sudan. Points outside the zobas within 2 km are snapped
  only where Natural Earth puts them in no neighbour (42,264 people, the coast); those inside Natural
  Earth's Ethiopia (54,330) and Djibouti (1,118) are dropped (playbook rule, Bhutan and Namibia). A
  cell whose cut piece is under a quarter of its square keeps the square less neighbours' land; 26
  of 3,288 dots sit just outside COD-AB's line (up to 1.8 km) inside Natural Earth's Eritrea.
- **Town witness pins**: Edd (GeoNames 11,259; both grids about 1,000, a village). Himora (46,100) is
  Humera, the Ethiopian town across the Tekeze, filed under ER in GeoNames.

## 6. §14

Ask 053. Only four bodies are registered (Eritrean Orthodox, Sunni Islam, Catholic, Evangelical
Lutheran); the State Department's 2023 report counts more than 500 Christians from unregistered
churches and 36 Jehovah's Witnesses in detention. A national mix places nobody. Built and shipped;
the ask is whether to keep it.

## 7. Reopen if

- Any table of religion by zoba or by ethnic group, or EPHS/EDHS microdata, becomes available.
- Eritrea holds a census.
- A third gridded population with zoba counts behind it appears; both grids here guess the split.

## 8. Review, 2026-10-03 (`fafd1067-rev12`, full pass)

Nothing to change in the build. Re-derived the five shares from the pinned Table 3-1 weighted counts
and Table 2-1's household bands (men 36.72%; Orthodox 57.39, Muslim 37.18), the zoba totals in
`er.csv` (sum 3,291,271) and the mapping; `christianity.oriental.eritrean` was already an ASARB
node, so the promotion adds no legend row. Note voice is clean. Screenshot at the country and at
z7.6: dots on land, densest on the highlands and Asmara, no blank zoba. Fixed the stale `queue.csv`
source cell (it still named EDHS 2002 Table 3.1).
