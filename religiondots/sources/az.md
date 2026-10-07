# Azerbaijan (`az`): an ethnicity model on the 2019 census

Drawn 2026-10-03 by session `fafd1067-az`, reopening a country closed twice (sources.md §11ao,
§scout-2026-09-14-asia-oceania) on Anita's priority line (ask/RULINGS.md 2026-09-15) and the
Mauritania ruling (2026-09-16: a country no source asks is drawn on the best figure there is, method
disclosed). `sources.md` §az-2026-10-03 is the summary. Ask 045 (Karabakh) is open.

Files: `sources/az_geo.py` (units), `sources/az_grid.py` (placement), `sources/az.py` (counts),
`taxonomy/az2019.py`, `countries/az.py`. Raw files in `data/raw/az/`.

## 1. What asks religion, and what it found

| instrument | place | religion | what it says |
|---|---|---|---|
| 2009 census form (UNSD `AZE2009azIn.pdf`) | rayon | no item | nationality Q8, mother tongue Q9 |
| 2019 census (Volumes A and B, stat.gov.az zips) | rayon | no item | nationality printed for the country only (Vol B Table 28); Vol A has no nationality |
| DHS 2006 (report FR195, `data/raw/az/dhs2006_FR195.pdf`, Table 3.1.1 p.64) | 11 regions | Q118 Muslim / Christian / none / other | women 15-49 99.2% Muslim, 0.7% "Christian/no religion/other"; national only in the report; microdata behind a DHS account (not requested) |
| LiTS III (2016, `data/raw/lits/lits_iii.dta`) | 8 regions, 75 PSUs | `q922` 8 codes | **1,510 of 1,510 MUSLIM**; `q923` Azeri 1,459, Talysh 30, Lezgian 17, Russian 3, all Muslim. Expected about ten Orthodox at 0.7% Russian; zero is the instrument (playbooks/lits.md) |
| EVS 2017 (ZA7500) | 8 regions, rayon | Muslim, Orthodox, Jewish ... | 1,704 of 1,714 Muslim among belongers; behind GESIS (not requested); Pew 2020's source |
| WVS 6 (2011), CRRC 2012, Caucasus Barometer 2013, Pew 2011 | various | see §11ao and the scout section | not reopened; none places a minority |

So the surveys answer for the majority (every Muslim-heritage respondent answers Muslim) and see no
minority. **Pew 2020's sources** (Pew 2025, Appendix A, p.10, `data/raw/az/pew2025_appendix_a.pdf`):
composition EVS 2017, age structure a projection of DHS 2006, switching Pew's 2011-12 survey.
Pew 2020: 10,181,730 people, Muslim 94.73%, unaffiliated 484,645 (4.76%), Christian 42,730, Jewish
8,580.

## 2. The route taken: §14.12's ethnicity model, census nationality by unit

Condition 1 (X published by unit): the 2009 census's nationality by rayon, *XIX cild* (2011). The
volume is not online; Tim Bespyatov's transcription (`pop-stat.mashke.org/azerbaijan-ethnic2009.htm`,
saved as `data/raw/az/mashke_azerbaijan_ethnic2009.htm`) is checked two ways: every unit total
equals the Committee's table 1.17 2009 column, and the national row equals table 1.11's 2009
column for every group used. Condition 2 (religion by X): no Azerbaijani source; Kazakhstan's 2021
census religion-by-nationality (`sources/kz_model.py`) for Russians, Ukrainians and Tatars, and
religio-ethnic identity (spec §14.5) for Georgians, Udins, Jews and Armenians. Condition 3 (a
second cut to check against): none in Azerbaijan; the note says so and names the weakest cell.

The groups, their 2019 counts and their religion are in `sources/az.py`'s docstring. Each group's
2019 national count is spread on its 2009 distribution over units populated in 2019.

**Why Kazakhstan's coefficients and not Pew's Christian total.** Pew's 42,730 Christians come from
a survey of 1,800 adults that meets about a dozen Russians; the census counts 71,046 Russians and
13,947 Ukrainians. Kazakhstan's census measured the same nationalities' religion in a post-Soviet,
Muslim-majority country, and the Orthodox share is ancestry-shaped (spec §14.12: held out to within
3%). Refusals (7.5% of Kazakhstan's Russians) are removed rather than drawn; Catholic, Protestant,
Judaism, Buddhism and other (0.6% of Russians together) are dropped and the rest renormalised.
Russians 92.73% Orthodox, 2.13% Muslim, 5.14% non-believer; Ukrainians 90.23 / 2.83 / 6.94; Tatars
28.72 / 63.62 / 7.66. Kazakhstan's figures for its own Azerbaijanis (27% refused) are NOT applied
to Azerbaijan's majority, whose own surveys answer.

**Why Pew's unaffiliated is not drawn.** EVS's 4.76% is "belongs to no denomination"; DHS 2006's
self-description item gives 0.7% for Christian, none and other together, and LiTS gives zero. No
source places it. Drawing it at a flat national share would put 485,000 secular dots on a country
whose two other probability surveys find almost none. Named in the note and in `gap`, not drawn.

Result (`data/normalized/az.csv`, 66 units): islam 9,830,318 (98.86%), christianity.orthodox
83,549, Georgian Orthodox 8,445, unknown 6,849, secular 5,975, judaism 5,099,
christianity.oriental 3,540, Armenian Apostolic 183. Christians 95,717 (0.96%) against Pew's 42,730.
Non-Muslims are 4.0% of Baku and 15.5% of Gakh.

## 3. Calls someone might reverse

- **Ingiloys (1,817) and other nationalities (5,039) on `unknown`.** Ingiloys are Muslim or
  Christian and the census box does not say which; "other" is mixed.
- **Udins on `christianity.oriental`**, not Armenian (the Albanian-Udi community does not call itself
  Armenian) and not a new node.
- **Molokans** (Ismayilli's Ivanovka, Gadabay's Slavyanka; 2009 Russians 2,024 and 133) drawn
  Orthodox; no node, and the census does not separate them.
- **Tatars at Kazakhstan's 28.7% Orthodox** rather than §14.5's Muslim: Baku's Tatars, like
  Kazakhstan's, are urban and intermarried, and consistency with the Russians' coefficient won.
  About 5,100 people.
- **No Sunni/Shia split** (§6).

## 4. Units and population

COD-AB `cod-ab-aze` v01, `aze_admin1.geojson`, 74 units (rayons plus the cities of republican
subordination; Baku and Ganja whole). Population is Volume A Table 3's **existing** column (de
facto), not the headline permanent one (de jure), because the permanent column counts the people
displaced from the districts outside government control in their district of origin: Kalbajar
71,039 permanent, every one temporarily absent, 0 existing. Joined on Table 3's English names
(`az_geo.ALIAS`), witnessed against table 1.17 (2019 permanent, 0.1 thousand: four units differ by
59 to 76 people, a revision after the volume). 8 units have no existing population; 66 are drawn,
151,000 people on average.

## 5. Placement

Kontur AZ 2023-11-01, 400 m hexes, centroid join, snap within 2 km unless the centroid is in a
neighbour on Natural Earth (101,428 people snapped; 12,535 dropped inside Armenia, Georgia, Iran or Russia and 3,820 more than 2 km out). **Masked:**
every hex inside Natural Earth **4.1.0**'s `Nagorno-Karabakh` (2018, public domain, 11,923 km2, the
1994-2020 line), because Kontur models built-up area and puts 2.7x Aghdam's census people inside
Aghdam (most of them on the ruins east of the line) and 79,000 in Jabrayil, where the census found
420. After the mask: Kontur/census 0.988 nationally, Spearman +0.955 over 66 units (0 of 20,000
shuffles), Aghdam 1.43 and Fuzuli 1.56 of the national ratio (the 1:10m line is generalised),
Nakhchivan city 1.84, Naftalan 0.26. Three of 9,940 dots fall inside the mask (hexes straddling
it). Today's Natural Earth has only the 2020-2023 remainder (`Artsakh`), the wrong polygon for 2019.
A digitised line of contact on GitHub (`mkudamatsu/data_karabakh-map`) has no licence and was not
used.

## 6. Sect

Not drawn, on the om/sa ruling (2026-09-15): Pew 2011 (37% Shia, 16% Sunni, 45% just Muslim) and
CRRC 2012 (Shia 10, Sunni 4, Islam 85) are national, and WVS 6's use of its Shia/Sunni codes for
Azerbaijan is still unchecked (the online tool, playbooks/wvs.md). An ethnicity-derived Sunni layer
for the Lezgins (167,570), Avars (48,636), Tsakhurs (13,361) and Turks was considered and refused: it
would mark the north's Sunni minorities and leave the Sunni Azerbaijanis of Shaki-Zagatala and
Guba-Khachmaz on the parent, which reads as "the north's Azerbaijanis are not Sunni".

## 7. Karabakh (ask 045)

Spec §14.18 (de facto administration) puts the whole of Karabakh in Azerbaijan; the 2019 vintage
leaves the area outside government control empty. Not drawn anywhere: the Karabakh Armenians, never
counted by Azerbaijan's census, of whom "more than 100,000" arrived in Armenia in September 2023
(UN News, 29 September 2023, quoting UNHCR's Filippo Grandi), after Armenia's October 2022 census.
Drawn in their 2019 places: the people resettled since 2021, "more than 48,000" per
president.az/en/greatreturn (read 2026-10-03, undated, no district figure). Caspian News reported
15,781 returned in 2025 from a Cabinet report; that page was 403 and is not cited.

## 8. Not checked, the next places to look

- WVS 6 `V144` by `V256` for Azerbaijan (Shia/Sunni codes), the online tool.
- DHS 2006 microdata (`v130` religion by region, Christians by region): needs a DHS account
  (Anita's), and would be the only survey to place the minority half, at about 60 Christian women.
- The 2009 *XIX cild* itself, to replace the transcription (checked, but a transcription).
- A district split of the Great Return resettlement.
- Armenia's registration of the 2023 refugees by marz (ask 045, option b).

## 9. Review, 2026-10-03 (`fafd1067-rev2`, full pass)

Checks clean (check_md, built_countries, check_rollup: 9,943,958 all modelled, nothing orphaned).
Screenshot: draws, Karabakh empty, nothing in the sea. Karabakh not looked at (ask 045).

- **Molokans against Armenia's precedent.** `taxonomy/am2022.py` files Armstat's `Molokai` on
  `christianity.other` and says in so many words that they are not Orthodox and must not be filed
  there. Here Ismayilli's and Gadabay's Russians (2009: 2,024 and 133; about 1,200 and 80 at the
  2019 scale) are drawn Orthodox, and the note says so. "No node exists" is true of Armenia too;
  the neighbour's answer is the existing `christianity.other`. About one dot, so not rebuilt here.
  At the next az touch: split those two units' Russian Orthodox share onto a `Molokan (Russians of
  Ismayilli and Gadabay)` category on `christianity.other`, reword the note's Molokan sentence,
  re-scatter az.
- **Note fixed (refresh-meta run):** it said "Three surveys asked religion and recorded where people
  were interviewed", but WVS 3 (1996, 8 areas) and WVS 6 (2011, 10 regions and 46 districts) did too
  (sources.md, "Azerbaijan stays closed, now on the surveys"). Now "Surveys that ... include" the
  three; the "In all three" sentence is unchanged and still true of those three.
- Seen, not changed: the only `secular` dots are Russians', Ukrainians' and Tatars' (5,975, from
  Kazakhstan's coefficients), while the 4.76% Pew finds among everyone else is left on Islam, so the
  little secular there is reads as Slavic. Six dots; the note names both halves.
