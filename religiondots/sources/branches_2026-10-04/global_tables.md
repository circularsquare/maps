# Muslim branch shares, global compiled sources, one row per country

Compiled 2026-10-04 for the religiondots Muslim-branch sweep. Every figure is **% of Muslims** unless
marked "of pop". Abbreviations in cells: **S** Sunni, **Sh** Shia, **Ib** Ibadi, **Ah** Ahmadiyya,
**Alw** Alawite, **X** WRD "Islamic schismatics", **else** "something else", **just** "just a Muslim",
**none** nothing in particular + don't know/refused, **n/s** not surveyed / not given.

## The sources, their basis, and where each figure is

| # | Source | Basis | Exact location | Notes |
|---|---|---|---|---|
| 1 | Pew Forum, *Mapping the Global Muslim Population* (Oct 2009) | **Ascription.** Appendix B (printed p. 38): "based primarily on data gathered via ethnographic and anthropological studies": 20+ consultant demographers, WRD's Sunni/Shia makeup of ~4,300 ethnolinguistic groups, and a review of IRF/CIA figures. No margin of error possible. | `https://www.pewresearch.org/wp-content/uploads/sites/20/2009/10/Muslimpopulation-1.pdf` (local copy `religiondots/data/raw/estimates/pew_muslim_population_2009.pdf`). "Estimated Percentage Range of Shia by Country", printed pp. 39-41 (PDF pp. 42-44); headline table "Countries with More Than 100,000 Shia Muslims" printed p. 10 (PDF p. 13). | Shia only; Sunni is the remainder. Shia **includes Alevis, Alawites, Ismailis, Zaydis** (p. 9). Ibadis ("Kharijites in Oman"), Druze and Nation of Islam are not separated, just left in the Muslim total (p. 9). Kosovo is "--". |
| 2 | Pew, *The World's Muslims: Unity and Diversity* (Aug 2012), Q31 | **Self-ID survey.** "Are you Sunni (for example, Hanafi, Maliki, Shafi, or Hanbali), Shia (for example, Ithnashari/Twelver or Ismaili/Sevener), or something else?" Ahmadiyya, Alevi, Bektashi, Aliran Kepercayaan, "just a Muslim" were volunteered answers. | `https://www.pewresearch.org/wp-content/uploads/sites/20/2012/08/the-worlds-muslims-full-report.pdf`. Summary table printed p. 30; full Q31 with every volunteered category in the Appendix D topline, PDF p. 128 (no printed number on that page). Sample notes printed pp. 119-127. | Asterisked rows (*) are re-tabulations of *Tolerance and Tension* (2010; fieldwork 2008-09). Exclusions that matter: Pakistan 82% of adults (no FATA, GB, AJK); Azerbaijan 85% (no Nakhchivan, Karabakh, Kalbajar-Lachin); Lebanon excludes "areas of Beirut controlled by a militia group" (i.e. likely Shia areas), n=551; Niger excludes Agadez; Palestine excludes Bedouin. (Iraq is missing from one later question "due to an administrative error"; Q31 is not affected.) |
| 3 | World Religion Database (Zurlo ed., Brill, accessed Sept 2025) via ARDA | **Ethnic ascription** per people group (WRD methodology note pp. 14-15). Figures are 2025. | Rankings pages (one table of every country): Muslims `https://www.thearda.com/world-religion/np-sort?var=ADH_495`, Sunnis `...?var=ADH_496`, Shias `...?var=ADH_505`, Islamic schismatics `...?var=ADH_512`. National profiles `https://www.thearda.com/world-religion/national-profiles?u=<n>c` (codes in the table below). Read 2026-10-04. | Only three sub-variables exist (ADH_497-504, 506-511, 513-514 return nothing). **"Islamic schismatics"** per ARDA (`thearda.com/world-religion/world-maps/?var=ADH_512`): "Followers of Islam, in other than its 2 main branches of Sunni or Shia. Islamic schismatics include Kharijite and other orthodox sects; reform movements (Sanusi, Mahdiya), also heterodox sects (Ahmadiya, Druzes, Sabbateans)." So X mixes Ibadi, Ahmadi, Druze and Sanusi. Shares here are counts ÷ the Muslim count; S + Sh + X = 100.0 in every row. |
| 4 | Correlates of War, World Religion Project v1.1, national file, 2010 | **Compiled blend, not self-ID** (codebook says so); reliability-weighted mix of sources incl. Barrett's WCE, smoothed. | `https://correlatesofwar.org/wp-content/uploads/WRP_national.csv`, local copy `religiondots/data/raw/estimates/WRP_national.csv`; columns `islmsun, islmshi, islmibd, islmnat, islmalw, islmahm, islmothr` ÷ `islmgen`, year 2010. Script: scratchpad `wrp_table.py`. | A 0 in a sub-column means "not split". Kosovo and Brunei are 100% `islmothr` = unsplit. Alawites have their own column (Syria). No Palestine row (COW has none). Reliability level shown where not "Medium". |
| 5 | CIA World Factbook, "Religions" field | **Unsourced compiler.** Several lines copy Pew 2009 or WRD (see flags). | **The Factbook is retired**: `cia.gov/the-world-factbook/field/religions/` now redirects to `cia.gov/stories/story/spotlighting-the-world-factbook-as-we-bid-a-fond-farewell/`. Read from the Wayback copy of the field's data file, `https://web.archive.org/web/20251231223225id_/https://www.cia.gov/the-world-factbook/page-data/field/religions/page-data.json` (snapshot 2025-12-31). | Converted to % of Muslims where the line gives % of population; original wording kept in brackets when it matters. "—" = the line gives no sect split. |

## The table

ARDA `u` code in brackets after the country. Pew 2012 cells read S / Sh / else / just / none, with
volunteered Ahmadiyya, Alevi or Bektashi named inside "else".

| cc | Pew 2009 Sh (ascribed) | Pew 2012 Q31 (self-ID) | WRD 2025 via ARDA | COW WRP 2010 | CIA Factbook (Dec 2025) |
|---|---|---|---|---|---|
| pk (172c) | 10-15 | 81 / 6 / 1 / 12 / 0 | S 89.83, Sh 7.87, X 2.30 | S 84.8, Sh 15.0, oth 0.2 | S 85-90, Sh 10-15 (2020 est.) |
| id (109c) | <1 | 26 / 0 / 5 / 56 / 13 | S 99.98, Sh 0.02, X 0 | S 98.7, Sh 1.0, Ah 0.2 | — |
| bd (19c) | <1 | 92 / 2 / 0 / 4 / 2 | S 100.00, Sh 0, X 0 | S 99.8, Sh 0.2 | — |
| eg (73c) | <1 | 88 / 0 / 0 / 12 / 0 | S 99.69, Sh 0.31, X 0 | S 99.5, Alw 0.5 | "predominantly Sunni" |
| sd (211c) | <1 | n/s | S 99.93, Sh 0.07, X 0 | S 100 (reliability Low) | "Sunni Muslim", no % |
| uz (236c) | ~1 | 18 / 1 / 0 / 54 / 26 | S 99.58, Sh 0.42, X 0 | S 98.9, Sh 1.1 | "mostly Sunni" |
| ma (155c) | <1 | 67 / 0 / 0 / 30 / 3 | S 99.55, Sh 0.45, X 0 | S 100 | "virtually all Sunni, <0.1% Shia" (of pop) |
| af (1c) | 10-15 | 90 / 7 / 0 / 3 / 0 | S 88.51, Sh 11.49, X 0 | S 80.4, Sh 19.1, oth 0.6 | S 84.7-89.7, Sh 10-15 (of pop, Muslim 99.7; 2009 est.) |
| dz (4c) | <1 | n/s | S 100.00, Sh 0, X 0 | S 100 | "predominantly Sunni"; Ahmadi, Shia, Ibadi inside "other <1%" of pop |
| sa (196c) | 10-15 | n/s | S 89.56, Sh 9.94, X 0.50 | S 93.3, Sh 6.7 | citizens S 85-90, Sh 10-12 |
| my (139c) | <2 | 75 / 0 / 0 / 18 / 7 | S 99.81, Sh 0.08, X 0.10 | S 90.0, Sh 10.0 (High) | — |
| ml (141c) | <1 | 20 / 0 / 3 (Ah 2) / 55 / 21 * | S 100.00, Sh 0, X 0 | S 98.9, Sh 1.1 (High) | — |
| sy (217c) | 15-20 | n/s | S 84.33, Sh 14.76, X 0.91 | S 82.3, Alw 17.7, Sh 0 | S 85.1, "Alawi, Ismaili, and Shia" 14.9 (74% and 13% of pop, Muslim 87); Druze 3% of pop listed outside Islam |
| so (205c) | <1 | n/s | S 98.26, Sh 1.16, X 0.58 | S 99.0, Sh 1.0 | S 98.2, Sh 1.2, schismatic 0.6 (= WRD) |
| ne (165c) | <1 | 59 / 7 / 11 (Ah 6) / 20 / 3 | S 99.99, Sh 0.01, X 0 | S 100 | — |
| kz (120c) | <1 | 16 / 1 / 0 / 74 / 10 | S 99.29, Sh 0.71, X 0 | S 94.9, Sh 5.1 | — |
| tn (225c) | <1 | 58 / 0 / 0 / 40 / 2 | S 97.99, Sh 0.10, X 1.91 | S 99.0, Sh 1.0 | "Sunni"; Shia inside "other <1%" of pop |
| jo (119c) | <1 | 93 / 0 / 0 / 7 / 0 | S 97.44, Sh 2.14, X 0.42 | S 97.9, Sh 2.1 | "predominantly Sunni" |
| az (16c) | 65-75 | 16 / 37 / 0 / 45 / 2 | S 31.18, Sh 68.82, X 0 | S 15.0, Sh 85.0 | "predominantly Shia" |
| tj (218c) | ~7 | 87 / 3 / 0 / 7 / 2 | S 90.04, Sh 9.96, X 0 | S 94.4, Sh 5.6 | S 96.9, Sh 3.1 (95% and 3% of pop, Muslim 98; 2014 est.) |
| gn (101c) | <1 | n/s | S 99.98, Sh 0.02, X 0 | S 99.1, Sh 0.9 | — |
| ae (232c) | ~10 | n/s | S 85.04, Sh 8.99, X 5.97 | S 83.9, Sh 16.1 | S 85.0, Sh 9.0, other 5.9 (63.3/6.7/4.4 of pop; = WRD) |
| bf (36c) | <1 | n/s | S 100.00, Sh 0, X 0 | S 98.4, Sh 1.6 | — |
| tm (227c) | ~1 | n/s | S 98.84, Sh 1.16, X 0 | S 100 (High) | — |
| ly (132c) | <1 | n/s | S 95.16, Sh 0.16, X 4.68 | S 100 (High) | "virtually all Sunni"; note: native Ibadis <1% of pop |
| kg (126c) | <1 | 23 / 0 / 0 / 64 / 12 | S 99.37, Sh 0.63, X 0 | S 100 | "majority Sunni" |
| td (45c) | <1 | 48 / 21 / 4 (Ah 4) / 23 / 4 * | S 100.00, Sh 0, X 0 | S 97.8, Sh 2.2 | — |
| sl (200c) | <1 | n/s | S 99.82, Sh 0.17, X 0.02 | S 97.3, Sh 2.7 | — |
| mr (146c) | <1 | n/s | S 100.00, Sh 0, X 0 | S 100 | — |
| ps (114c) | <1 | 85 / 0 / 0 / 15 / 0 | S 88.24, Sh 0.11, **X 11.66** | no row | West Bank and Gaza both "predominantly Sunni" |
| om (171c) | 5-10 (Ibadi not separated) | n/s | S 53.07, Sh 7.34, X 39.58 | S 32.2, Sh 7.8, **Ib 60.0** | citizens: Ibadi ~45, Sunni ~45, Shia ~5 (of citizens, ≈ 47 / 47 / 5 of Muslim citizens) |
| ba (28c) | <1 | 38 / 0 / 0 / 54 / 7 | S 99.62, Sh 0.38, X 0 | S 100 | — |
| gm (87c) | <1 | n/s | S 99.56, Sh 0.06, X 0.38 | S 98.9, Sh 1.1 | — |
| xk (250c) | -- (no figure) | 24 / 1 / 2 (Bektashi 2) / 58 / 15 | S 100.00, Sh 0, X 0 | unsplit (High) | — |
| al (3c) | <5 | 10 / 0 / 13 (Bektashi 13) / 65 / 12 | S 94.16, Sh 5.84, X 0 | S 95.2, Sh 4.8 (High) | — (Bektashi 2.1% of pop listed beside Muslim 56.7%) |
| dj (68c) | <1 | 77 / 2 / 0 / 8 / 13 * | S 100.00, Sh 0, X 0 | S 100 | Sunni 94% of pop; Shia among the foreign-born "other 6%" |
| km (56c) | <1 | n/s | S 100.00, Sh 0, X 0 | **Sh 100** (known error) | "overwhelmingly Sunni", small Shia and Ahmadiyya |
| gw (100c) | <1 | 40 / 6 / 2 (Ah 2) / 36 / 16 * | S 99.45, Sh 0.32, X 0.22 | S 97.5, Sh 2.5 | — |
| qa (183c) | ~10 | n/s | S 94.91, Sh 4.86, X 0.23 | S 78.7, Sh 21.3 | — |
| mv (140c) | <1 | n/s | S 99.89, Sh 0.11, X 0 | S 93.9, Sh 6.1 | "Sunni Muslim (official)" |
| bn (34c) | <1 | n/s | S 99.78, Sh 0.22, X 0 | unsplit (High) | — |
| *calibration, already split on the map* | | | | | |
| tr (226c) | 10-15 (Alevis counted Shia) | 89 / 1 / 5 (Alevi 5) / 2 / 4 | S 84.61, Sh 15.39, X 0 | S 100 | "mostly Sunni" |
| ir (110c) | 90-95 | n/s | S 17.20, Sh 81.81, X 0.99 | S 9.1, Sh 90.9 (High) | — |
| iq (111c) | 65-70 | 42 / 51 / 0 / 5 / 1 | S 36.27, Sh 62.87, X 0.86 | S 34.2, Sh 65.8 | Sh 61-64, S 29-34 of pop, Muslim 95-98 (≈ Sh 62-67, S 30-36; 2015 est.) |
| ye (244c) | 35-40 | n/s | S 44.34, **Sh 55.06**, X 0.60 | S 55.6, Sh 44.4 (High) | S 65, Sh 35 |
| lb (129c) | 45-55 | 52 / 48 / 0 / 0 / 0 | S 41.96, Sh 49.51, X 8.54 | S 50.0, Sh 50.0 (known placeholder) | S 47.1, Sh 46.0, rest Alawite/Ismaili (31.9/31.2 of citizen pop, Muslim 67.8); Druze 4.5% of pop outside Islam |
| kw (125c) | 20-25 | n/s | S 85.96, Sh 14.03, X 0.01 | S 76.1, Sh 23.9 | — |
| bh (18c) | 65-75 | n/s | S 43.76, Sh 56.24, X 0 | S 33.0, Sh 67.0 (High) | — |
| sn (197c) | <1 | 55 / 0 / 7 (Ah 1) / 27 / 12 * | S 99.98, Sh 0.02, X 0 | S 98.9, Sh 1.1 | — ("most adhere to one of the four main Sufi brotherhoods") |

Pew 2012 also surveyed Cameroon* (27 / 3 / 17 incl. Ah 12 / 40 / 14), DR Congo*, Ethiopia*, Ghana*,
Kenya*, Liberia*, Nigeria*, Tanzania*, Uganda*, Russia, Thailand (five southern provinces): see
`religiondots/sources/branches.md`. Not surveyed by Pew 2012 among the target list: Sudan, Algeria,
Saudi Arabia, Syria, Somalia, Guinea, UAE, Burkina Faso, Turkmenistan, Libya, Sierra Leone,
Mauritania, Oman, Gambia, Comoros, Qatar, Maldives, Brunei (and Iran, Yemen, Kuwait, Bahrain).

## Where the sources disagree by a lot

**Effectively one branch by every compiler** (Pew 2009 <1, WRD Shia under ~1% and no big X, WRP under
~2%): Morocco, Algeria, Egypt, Sudan, Mauritania, Somalia, Djibouti, Bangladesh, Indonesia,
Uzbekistan, Kyrgyzstan, Turkmenistan, Bosnia, Kosovo, Brunei, Guinea, Gambia, Burkina Faso, Mali,
Sierra Leone, Senegal, Jordan (WRD 2.1). Maldives too, except WRP's 6.1. The only other dissent in these rows is
Pew 2012's 2% Shia in Bangladesh and Djibouti.

1. **Sub-Saharan self-ID Shia far above every compiler.** Pew 2012 (Tolerance and Tension fieldwork):
   **Chad 21% Shia** against Pew 2009 <1, WRD 0.00, WRP 2.2; **Niger 7%** against WRD 0.01 and WRP 0;
   **Guinea-Bissau 6%** against WRD 0.32; Djibouti 2% against 0. No compiler knows a Shia community
   of that size in any of them; the same item gave Nigeria 12% and Tanzania 20%. Reads as label
   confusion in the answer, the same shape branches.md found for Maliki in Morocco. Do not use as Shia.
2. **Azerbaijan, ascription against self-ID.** Pew 2009 65-75, WRD 68.8, WRP 85.0 Shia; Pew 2012
   asked: 37 Shia, 16 Sunni, **45 "just a Muslim"**. (Already in branches.md.)
3. **Turkey: whether Alevis are Shia.** Pew 2009 10-15 and WRD 15.4 count Alevis as Shia; WRP 0; Pew
   2012 self-ID 1 Shia plus 5 volunteered Alevi. A definition split, not a measurement one.
4. **Albania: Bektashis.** WRD 5.8 and WRP 4.8 "Shia" are almost certainly the Bektashis; Pew 2012 has
   0 Shia and 13 volunteered Bektashi; the Factbook lists Bektashi (2.1% of pop) outside "Muslim".
   Kosovo: Pew 2012 2 Bektashi, every compiler 0.
5. **Oman: the Ibadi share.** WRP 60.0 Ibadi, WRD 39.6 X (Ibadi is the bulk), Factbook ~45 of citizens,
   Pew 2009 does not separate it. WRD and WRP are whole-population Muslims, so South Asian and other
   expatriate Sunnis dilute the Ibadi share; the Factbook's is citizens only. None says where its figure
   comes from.
6. **WRP runs high on Shia across the Gulf and Asia**: Qatar 21.3 (WRD 4.9, Pew 2009 ~10), UAE 16.1
   (WRD 9.0), Afghanistan 19.1 (others 7-15), Pakistan 15.0, **Malaysia 10.0** (WRD 0.08, Pew 2012 0,
   Pew 2009 <2), Kazakhstan 5.1 (others <1), Maldives 6.1 (WRD 0.11). Malaysia's 10% is not credible.
   And the opposite error: **Comoros 100% Shia** (every other source Sunni), Lebanon an exact 50/50,
   Turkey and Libya 100% Sunni.
7. **Yemen: WRD makes Shia the majority (55.1)**, against Pew 2009 35-40, Factbook 35, WRP 44.4.
   Calibration only (ye is already split), but it says WRD's ethnic ascription can be 15+ points off
   on Zaydis.
8. **Pakistan, 6 to 15.** Pew 2012 self-ID 6 (a floor: 82% sample, no GB/FATA/AJK), WRD 7.9 (+2.3 X,
   which will be the Ahmadis), Pew 2009 and Factbook 10-15 (the Factbook copies Pew 2009), WRP 15.0.
9. **Tajikistan, 3 to 10.** Pew 2012 3 (sample covered all four oblasts incl. Gorno-Badakhshan),
   Factbook 3.1, WRP 5.6, Pew 2009 ~7, WRD 10.0. All the Shia here are Pamiri Ismailis.
10. **Saudi Arabia, 6.7 to 15**: WRP 6.7, WRD 9.9, Factbook 10-12 of citizens, Pew 2009 10-15.
11. **Gulf calibration rows sit lower in WRD**: Bahrain WRD 56.2 vs 65-75 elsewhere; Kuwait WRD 14.0 vs
    20-25; Iran WRD 81.8 Shia vs 90-95. WRD is total population including expatriates, which pulls
    Shia shares down in the Gulf.
12. **Unexplained WRD "schismatics"**: **Palestine 11.66%** (no Ibadi, Druze or Ahmadi community of
    that size is known there; Pew 2012 self-ID is 85 Sunni / 15 just a Muslim); UAE 5.97; Libya 4.68
    (Ibadi Berbers of Jebel Nafusa and Zuwara, perhaps also Sanusi, which WRD's definition includes);
    Tunisia 1.91 (Djerba Ibadis, presumably). Lebanon's 8.54 is the Druze, whom WRD counts as Muslim.
13. **Lebanon's Pew 2012 Shia 48** is probably low: the sample excluded "areas of Beirut controlled
    by a militia group" (n=551).

## Circular figures to discount

- **Factbook Afghanistan "Sunni 84.7-89.7%, Shia 10-15%" and Pakistan "Sunni 85-90%, Shia 10-15%"
  are Pew 2009's ranges.** Afghanistan's is dated "2009 est." and the Sunni bound is 99.7 minus the
  Shia range.
- **Factbook Somalia (98.1 / 1.2 / 0.6 schismatic) and UAE (63.3 / 6.7 / 4.4 of pop) are WRD's
  shares**, down to WRD's own "Islamic schismatic" label.
- Pew 2009's ranges themselves lean on WRD (Appendix B), so Pew 2009 and WRD agreeing is partly one
  vote. Pew 2012 is the only independent measurement in this file, and it is self-ID with large "just
  a Muslim" shares.
- Factbook Tajikistan Shia 3 (2014 est.) matches Pew 2012's 3; it may be copied but the Factbook line
  does not say.

## Files in the scratchpad

`wrp_table.py` (WRP 2010 shares), `pew2009_shia.py` (Pew 2009 Shia ranges from the local PDF),
`pew2012.pdf` + `pew2012_q31.py` (Q31 with volunteered categories), `arda_495/496/505/512.html` +
`arda_parse.py` (WRD rankings), `cia_pagedata.json` + `cia_extract.py` (Factbook, Wayback 2025-12-31).
