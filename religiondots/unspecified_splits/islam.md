# Islam's undivided countries: Pakistan, Afghanistan and Europe by origin (research, 2026-10-16)

**Built 2026-10-16 on Anita's approval:** Germany (§3, `sources/de_mld.py`, `sources/de.md` §13) and
England and Wales's first step (§4, Pakistani and Bangladeshi only; `sources/uk_islam.py`,
`sources/uk.md` §13). Not approved, left as they are: Pakistan, Afghanistan, and borrowing German
shares for the Netherlands, Austria or Switzerland. The rest of this file is the research as written.

Research only at the time of writing. Read with `sources/branches.md` (the 2026-09-14 and 2026-10-04 sweeps), spec §2.6b,
§2.7a and §14.

Anita's aim: neighbouring countries should share a palette, so Islam should be split into Sunni,
Shia (and Ahmadi, Alevi where nodes exist) in more places.

## Corrections to the brief, found on the way

- **Pakistan is no longer drawn from 2017.** Since 2026-09-14 it is the 2023 census, PBS Table 9,
  136 districts, plus Azad Kashmir's ten districts since 2026-10-03 (`sources/pk.md` §9, §10). Ahmadis
  are already their own cell, by district. **Gilgit-Baltistan is not drawn at all** (no religion
  table found, `pk.md` §10.3), so the one Shia-majority region of Pakistan is outside the drawn map.
- **Germany's census has no religion question.** Zensus 2022 reads the church-tax register (Catholic,
  EKD, everything else). Germany's 3.47M Muslims on the map are `modelled` from pooled ESS rounds,
  one share per Land, placed inside each Gemeinde by a Turkish/Bosnian citizenship grid
  (`sources/de.md` §6, §9, §10; `taxonomy/de2022.py`).
- **The Survey of the Afghan People has no sect item.** Checked in the 2019 report's questionnaire
  (ACBAR copy, `acbar.org/upload/1575432931309.pdf`): demographics run D-1 to D-21 with ethnicity
  (D-14) and languages (D-11) and nothing on religion or sect. This agrees with `sources/af.md` §1.
  The microdata (ADA Dataverse, doi:10.26193/VDDO0X) sits behind a bot wall and a download form.
- **France's Shia and Alevi dots are passport-holders only.** `fr2024.py` sends every citizen
  Muslim to `islam.sunni`; only the foreign half is split, by `taxonomy/origin_religion.py`. Belgium,
  Sweden, Norway, Denmark, Finland, Italy, Greece, Spain, Luxembourg and Iceland do the same.
- **`origin_religion.py`'s `MUSLIM` rows still have no source** (`branches.md`, "origin_religion.py's
  sect shares have no source"). Anything below that reuses them inherits that; the German source
  below can source several of them.

## Ranking by visual payoff

| rank | where | Muslims drawn on bare `islam` | proposal | tier | readiness |
|---|---|---:|---|---|---|
| 1 | Pakistan | ~240M | national Sunni assignment, Kurram-type districts whole | `assigned` | cheap; a §14 and remainder call |
| 2 | Germany | 3.47M | per-Land split from BAMF MLD 2020 | stays `modelled` | cheap; numbers computed below |
| 3 | England and Wales | 3.87M | per-LTLA split by ethnic group x origin sect shares | `derived` | moderate work |
| 4 | Afghanistan | 34.9M | national Sunni assignment, Hazarajat whole | stays `modelled` | cheap; a §14 call |
| 5 | Netherlands, Austria, Switzerland | ~0.85M, 0.34M, 0.31M | the German method by transfer | as drawn | not sourced well enough yet |

Pakistan and Afghanistan come first on area and people, but what they can honestly get is a
national split. Germany and England can get a split that varies by place.

---

## 1. Pakistan

**Drawn now.** 2023 census, 136 districts plus AJK's ten; `islam` undivided except
`islam.ahmadiyya` from the census's Qadiani/Ahmadi column. GB undrawn. spec §2.6b names Pakistan as
*not* assigned: "real minorities nothing places".

**Sources, all re-checked against `branches.md`; nothing new turned up.**

| source | year | universe and sample | geography | what it gives |
|---|---|---|---|---|
| Pew, *The World's Muslims* Q31, p. 30 | 2011-12 | adults, n about 1,450; excludes FATA, GB, AJK, unstable KP and Balochistan (82% of adults) | national | Sunni 81, Shia 6, something else 1, just a Muslim 12 |
| Pew 2009 *Mapping the Global Muslim Population* | 2009 | ascription | national | Shia 10-15% |
| Fair, Malhotra, Shapiro (Dataverse `hdl:1902.1/17042`, CC0) | 2009 | n 6,000, four provinces | province | Shia 3.8%; KP 0.4%, Balochistan 0.5%. Fails known places (Kohat, Hangu, D.I. Khan, Quetta Hazara) |
| Ramzan, *J. Political Studies* 22(2) | 2015 | officials' estimates | Punjab divisions | Shia 10.86%; Anita declined it 2026-09-16 |
| Jamestown 2008 citing *The News* | 2008 | "official figures", no document | Kurram | 42% Shia |
| Manzoom Ali, *Atlas of the Northern Areas* via ICG 2007 | 2004 | method unknown | GB districts | moot, GB undrawn |
| WVS 7, PSLM, PDHS, census 2023 | | | | no sect code |

Not followed up (lead only): Pew Global Attitudes Pakistan waves (n about 1,200 a year, free Pew
account) may carry a sect code with a province variable. It would have the same exclusions and the
same problem Fair et al. found, that respondents can read an interviewer's sect, so expect it to
fail the known-places test too.

**Proposal.** spec §2.6b's shape, not §2.7a's:

- every drawn district's census Muslims (less the census Ahmadis, who stay where they are) split
  into `islam.sunni` at one national share, rolled to `islam` as the `assigned` tier, the rest left
  on `islam`;
- **the Sunni share is the decision.** Pew 2012's surveyed 81% leaves 19% on `islam` (6% Shia, 13%
  just a Muslim or other). Pew 2009's ascribed 10-15% Shia would leave 85-90% Sunni. Pew's 6% is a
  floor because its frame left out the most Shia areas;
- **no Shia dots anywhere.** Drawing 6% Shia evenly over every district would place Shia in places
  nothing says they live, and the places where they are concentrated are the ones repeatedly
  attacked (`branches.md` §14 flags). The Shia sit inside the `islam` remainder, as India's
  Ahmadis do;
- **whole units, as Ghardaïa and Gorno-Badakhshan:** districts where any source puts a large
  non-Sunni share stay wholly on `islam`. Candidates: Kurram (Jamestown's 42%; the 1931 superintendent
  "exceed 60 per cent"), Orakzai, Hangu. Every candidate rests on journalism or a colonial estimate,
  so the list itself is a call.

**What would contradict it.** Nothing below national can. The national 81% is Pew's own figure.

**Confidence.** High that most of Pakistan's Muslims are Sunni; low on the exact share.

**Anita decides:** (a) whether to assign Pakistan at all after §2.6b left it out; (b) Pew 2012's 81%
or a WRD/Pew 2009 figure; (c) whether any district is kept whole, and which; (d) §14: even a
remainder-only split makes no Shia placement, but the whole-unit list would mark Kurram as "not
Sunni" on the map.

## 2. Afghanistan

**Drawn now.** NSIA 2025-26 estimates, 34 provinces, 34.9M, every row `modelled` on bare `islam`
(`sources/af.md`). The supervisor's brief at build time was "no Sunni/Shia split by province".

**Sources.**

| source | year | sample | geography | gives |
|---|---|---|---|---|
| Pew *World's Muslims* Q31 | Nov-Dec 2011 | 1,509, all 34 provinces, nomads excluded | national | Sunni 90, Shia 7, just a Muslim 3 |
| WRD 2022 via State Dept IRF 2023 | | ascription | national | Sunni 89, Shia 11 |
| Izady via State Dept | | ascription | national | Shia up to 29% |
| State Dept IRF 2023, citing community leaders | | | regions only | about 90% of Shia are Hazara; "25 percent or more of Hazaras are Sunni" |
| Survey of the Afghan People | 2004-2019 | about 16-18k a year, all 34 provinces | province | **no sect item** (checked 2019); ethnicity only |
| Pew 2011 microdata, ARDA `WDMUS12` | | about 106 Shia respondents | possibly province | about 3 Shia a province; not opened |

**Ethnic assignment does not work here.** Ethnicity by province (Asia Foundation D-14) would draw
Hazaras, not Shia: the State Department's sources put a quarter or more of Hazaras as Sunni, and the
Ismailis are Tajik and Pamiri. spec §14.5's religio-ethnic test fails, as `af.md` §3 already says.

**Proposal.** The Pakistan shape: `islam.sunni` at Pew's 90% (or WRD's 89%) in every province,
the rest on `islam`, still `modelled`, no Shia dots. **Whole units:** Bamyan and Daykundi, the two
provinces every account calls Hazara and Shia heartland. No figure exists for either, so the
whole-unit list rests on prose (State Dept "central provinces", WikiShia). Without whole units the
map would draw Bamyan 90% Sunni, which every reader who knows the country would see is wrong.

**Confidence.** Medium nationally. Low for any province.

**Anita decides:** whether to split at all (§14: Hazara Shia under Taliban rule; even the whole-unit
list says "these two provinces are not Sunni"); Pew 90% or WRD 89%; whether to keep Bamyan and
Daykundi whole.

---

## 3. Germany: per-Land split from BAMF *Muslimisches Leben in Deutschland 2020*

**Drawn now.** `islam` 3,471,376, ESS-modelled per Land (`de_ess.py`), placed within Gemeinden on
the citizenship grid. `de2022.py`'s note refuses `islam.sunni` because of the Alevis and Shia.
That reasoning stands; this proposal answers it with a source instead of assigning.

**Source.** Stichs, Pfündel et al., *Muslimisches Leben in Deutschland 2020*, BAMF Forschungsbericht
38 (2021), `bamf.de/SharedDocs/Anlagen/DE/Forschung/Forschungsberichte/fb38-muslimisches-leben.pdf`.
Telephone survey, 2020; 5,120 respondents aged 16 and over (4,538 with a migration background from
23 Muslim-majority countries); 11,243 household members; weighted. Projected onto Mikrozensus 2019
and the AZR.

- **Table 23** (p. 38): Muslims with a migration background, 2019, middle estimate in thousands:
  Middle East (Afghanistan, Bangladesh, Iran, Pakistan) 481; Near East (Iraq, Jordan, Lebanon,
  Syria, Yemen/Saudi/UAE) 1,050; North Africa 417; South-East Europe 1,046; Türkiye 2,462;
  total 5,457.
- **Table 32** (p. 59): sect by origin region, % of Muslims (question v201, n 3,398):

  | | Middle East | Near East | North Africa | SE Europe | Türkiye | all |
  |---|---:|---:|---:|---:|---:|---:|
  | Ahmadi | 10.9 | 0.2 | 0.2 | 0.3 | 0.4 | 1.3 |
  | Alevi | 1.1 | 0.6 | 0.1 | 1.9 | 16.9 | 9.5 |
  | Shia | 27.3 | 5.5 | 0.5 | 2.8 | 1.2 | 4.4 |
  | Sunni | 55.7 | 89.3 | 90.6 | 54.9 | 71.9 | 71.7 |
  | other, unspecified, don't know | 5.0 | 4.4 | 8.7 | 40.2 | 9.7 | 13.2 |

- **Table 26** (p. 53): each origin region's Muslims spread over the Länder, % (household
  members, n 11,243). Bremen and Hamburg are printed together, as are Brandenburg and
  Mecklenburg-Vorpommern.

**Method.** For each Land, Muslims by region = Table 23 × Table 26; sect shares = the
Muslim-weighted mean of Table 32's rows. Apply those shares to the Land's ESS Muslim count, which
stays `modelled`, and keep the current placement inside the Land. "Other, unspecified, don't know"
stays on `islam` (§2.7a). Every input is a published BAMF number, so this passes
`feedback_proxy_residual_nameable`: the non-matching part, the non-Sunni share of each origin
region, is printed in Table 32.

**Computed** (`scratchpad/islam/scripts/de_land.py`; % of each Land's Muslims):

| Land | MLD Muslims (k) | Ahmadi | Alevi | Shia | Sunni | rest |
|---|---:|---:|---:|---:|---:|---:|
| Baden-Württemberg | 931 | 0.7 | 8.9 | 3.2 | 68.8 | 18.4 |
| Bayern | 649 | 1.0 | 7.6 | 4.1 | 69.5 | 17.8 |
| Berlin | 275 | 0.9 | 10.7 | 3.6 | 73.5 | 11.4 |
| Bremen + Hamburg | 270 | 2.8 | 7.4 | 8.3 | 70.6 | 10.9 |
| Hessen | 604 | 2.4 | 7.9 | 6.8 | 70.8 | 12.1 |
| Niedersachsen | 364 | 1.4 | 7.0 | 5.5 | 74.2 | 12.0 |
| Nordrhein-Westfalen | 1,710 | 0.8 | 9.3 | 3.2 | 73.3 | 13.5 |
| Rheinland-Pfalz | 234 | 1.5 | 8.0 | 5.1 | 71.7 | 13.8 |
| Saarland | 77 | 0.8 | 5.9 | 5.0 | 79.7 | 8.6 |
| Schleswig-Holstein | 134 | 2.0 | 5.7 | 7.1 | 73.8 | 11.5 |
| Brandenburg + MV | 49 | 1.6 | 2.5 | 7.3 | 77.5 | 11.1 |
| Sachsen | 32 | 1.9 | 3.3 | 7.5 | 78.2 | 9.1 |
| Sachsen-Anhalt | 69 | 1.5 | 0.7 | 7.9 | 82.5 | 7.3 |
| Thüringen | 64 | 3.7 | 2.7 | 11.8 | 73.7 | 8.1 |
| **Germany** | 5,462 | 1.3 | 8.2 | 4.6 | 72.0 | 14.0 |

Each Table 26 column sums to 99.9-100.3. The national row reproduces MLD's own 1.3 / 9.5 / 4.4 /
71.7 within about a point. Alevi comes out at 8.2 against 9.5 because Table 32's last column weights
respondents, not projected people.

**Possible refinements, not needed for a first build.** (a) Table 23 has per-country counts, so
Iran (69k) and Afghanistan (278k) could be separated from Pakistan within "Middle East" using
outside sect shares. Table 32 is by region only, so that would bring the unsourced
`origin_religion.py` rows back in. (b) Within-Land placement could move Shia dots towards
Iranian/Iraqi/Afghan citizens, but `de.md` §10 says the Zensus grid publishes none of those
countries, so no.

**What would contradict it.** No other source gives sect by Land. Zensus 2011's Frage 8 had
Sunni/Shia/Alevi boxes but was voluntary, routed past members of public-law bodies, and is
unusable (`de.md` §2). MLD 2008 (Haug et al.) gives the earlier national row: Sunni 74.1, Alevi 12.7,
Shia 7.1, Ahmadi 1.7. That is a drift check, not a Land check.

**Confidence.** Good on the national level. The Land differences rest on Table 26's household
sample, and its small cells (Thüringen's 4.3% of Middle-Eastern Muslims, the eastern Länder at
n of a few dozen) are noisy. Pooling Brandenburg+MV and Bremen+Hamburg is the source's own choice.

**Anita decides:** **where the Alevis go.** `alevism` is its own family, outside Islam
(`branches.py`). MLD counts them as Muslims, but only 50% of MLD's Alevis say they are Muslim and 41%
say they are not (a Bundestag Drucksache, 21/4291 or 20/6673, quoting MLD; seen in a search
summary only, not opened), so some German Alevis are probably in ESS's
"other" or "none" answers rather than its Islam cell. The options are to move MLD's Alevi share to
`alevism` (consistent with France, Sweden and England, where Alevis sit on `alevism`) or to leave
it on `islam`. Also: Ahmadi on `islam.ahmadiyya` at 0.7-3.7% (Ahmadiyya Muslim Jamaat has public-law
status in Hessen and Hamburg, consistent with the higher shares there).

**Side benefit.** Table 32 sources part of `origin_religion.py`. Türkiye-origin Muslims in Germany
are 71.9 Sunni, 16.9 Alevi and 1.2 Shia, against the file's unsourced 0.80 / 0.20 (Shia: none).
The Middle East and Near East rows bound Iran, Iraq, Afghanistan and Pakistan as groups.

## 4. England and Wales: per local authority, from the 2021 census

**Drawn now.** `uk2021.py`: `Muslim` → `islam`, 3.87M at Output Area. Alevis are already their own
node (25,657 write-ins → `alevism`). Scotland (NRS 2022) and Northern Ireland are also bare `islam`.

**What ONS publishes, checked live 2026-10-16** (open, no key, through the Cantabular API
`api.beta.ons.gov.uk/v1/population-types/UR/census-observations`):

- **religion × ethnic group (20), every LTLA:** `?area-type=ltla&dimensions=religion_tb,ethnic_group_tb_20b`
  returned 330 of 331 LTLAs. Muslims nationally: Pakistani 1,470,766 (38.0%), Bangladeshi 593,142
  (15.3%), Black African 378,209 (9.8%), Arab 277,736 (7.2%), any other 255,760 (6.6%), Indian
  246,969 (6.4%), Other Asian 237,253 (6.1%), Other White 131,428 (3.4%), White British 90,926 (2.4%).
  (Also the pre-built table RM031.)
- **religion × country of birth: real, but blocked for most areas.** The 60-country version at LTLA
  returned 61 of 331 LTLAs (270 blocked by disclosure rules); at MSOA 42 of 7,264. The 190-country
  version is open at region level for only 4 of 10 regions. Where it is open: **48% of Muslims
  were born in England**, then Pakistan 13.6%, Bangladesh 7.7%, Somalia 3.0%, India 2.2%, Türkiye
  2.2%, Afghanistan 2.0%, Iraq 1.7%, Iran 1.0% (four regions, 2.43M Muslims). ONS has no pre-built
  religion × country-of-birth table (dataset list checked).
- three-way religion × ethnicity × country of birth: refused even at nation level (HTTP 400).

So country of birth cannot carry the split alone: half of Muslims are UK-born, and the LTLA table is
mostly blocked. Ethnic group is complete.

**Proposal.**
1. At each LTLA, Muslims by ethnic group (complete).
2. Groups that name one origin take that origin's sect shares: Pakistani → Pakistan, Bangladeshi →
   Bangladesh. Together these are 53% of Muslims. The weights have to be sourced: Pew 2012 Q31 for
   both. Note the documented skew: Britain's Pakistanis are largely from Mirpur in AJK, a Sunni area
   outside Pew's frame, so Pew's 6% Shia is if anything high for them.
3. Mixed groups (Arab, Other, Other Asian, Black African, Other White, Indian) take the origin mix
   of foreign-born Muslims at the finest open level (an LTLA where it is not blocked, else the
   region or nation, 190 categories), weighted by origin sect shares. Indian Muslims need care:
   East-African-born Muslims (Kenya 12.6k, Uganda 6.3k, Tanzania 6.1k) are heavily Khoja (Ismaili
   and Twelver) and Bohra. That is a known skew without a published number, so it is better left on
   `islam`.
4. Tier `derived` (spec §2.7a: survey shares inside a counted column), rolling back to `islam`,
   placed at LTLA and spread over the existing OA placement.

**The weak joint.** Step 2's "UK-born Pakistani-ethnicity Muslims are like Pakistan" is the
ethnicity proxy `feedback_proxy_residual_nameable` warns about. It passes the test only because the
correction (Pakistan's non-Sunni share) is a published number. For "Arab" and "Other" there is no
published number for which origins they hold, so step 3 borrows the foreign-born mix, which is an
assumption. A defensible first build stops at step 2: assign Pakistani and Bangladeshi Muslims
(53%) by origin, leave the rest on `islam`. That alone turns most of Bradford, Birmingham, Tower
Hamlets and Luton Sunni-coloured.

**Check.** No survey of British Muslims publishes a sect split that I found. Policy Exchange/ICM
2016 (n 3,040) prints none; its PDF was read. The independent placement witness is Mehmood
Naqshbandi's MuslimsInBritain.org mosque register (*UK Mosque Statistics*, 16 Sep 2017,
`muslimsinbritain.org/resources/masjid_report.pdf`): 1,975 UK masjids by "theme", with Shi'a
Twelver, Ismaili and Bohra listed separately (Shi'a about 3% of masjids). The per-mosque directory
on the same site gives each one's town. As with Bahrain (`branches.md`), a register ranks places and
does not count people; use it to check that Shia-weighted LTLAs (Brent, Ealing, Westminster,
Kensington, Harrow) come out on top.

**Confidence.** Medium-high for the Sunni assignment of Pakistani and Bangladeshi Muslims. Low for
Shia placement from mixed groups.

**Anita decides:** stop at step 2 (assign 53%, rest on `islam`) or go on to step 3 with the
foreign-born mix. And whether Scotland gets the same (NRS's flexible table builder was not checked
for religion × ethnicity).

## 5. Other drawn European countries

Checked from the taxonomy files:

| cc | source drawn | Muslims now on | origin data for a split | verdict |
|---|---|---|---|---|
| fr, be, se, no, dk, fi, it, gr, es, lu, is | ESS or census, plus the foreign half | citizens `islam.sunni`, foreigners by `origin_religion.py` | already split | consistent palette; only Shia and Alevi among foreigners |
| bg | 2021 census | census Sunni/Shia | | done |
| nl | CBS EBB 2010-14 by gemeente | `islam` | CBS StatLine migration background by gemeente (Turkey, Morocco, Suriname groups), open; sect shares by origin would come from MLD's Germany rows or SCP *De religieuze beleving van moslims in Nederland* (2018), not opened | cheap if SCP prints Alevi/Shia by group; MLD's German rows transferred to NL are a borrowed number, not a Dutch one |
| at | 2001 census by Gemeinde | `islam` | 2001 census religion × citizenship not checked; VID/ÖIF *Demographie und Religion in Österreich* (Goujon 2017) has Muslims by origin and Land, not checked | the German method transfers only by borrowing MLD's rows |
| ch | 2000 census by commune | `islam` | 2000 census religion × nationality not checked | same |
| ie, pt, cy | census | `islam` | small numbers (ie 81,930; pt about 0.4%) | low payoff |
| uk Scotland, NI | NRS 2022, NISRA 2021 | `islam` | see §4 | with England |

**Cheapest consistent step for nl, at, ch** if Anita wants the palette now without a sect source:
the existing ESS-country shape, `islam.sunni` with the minority on the parent (§2.6b-style assigned
row and remainder). The German MLD shares are the best published remainder for Turkish-origin
communities, which are the biggest group in all three. Flag: that is a German number applied
abroad.

## Downloads and scripts (scratchpad, not kept in the repo)

`scratchpad/islam/dl_ons/`: ONS API pulls (religion × country of birth 60/190, religion × ethnic
group by LTLA), `fb38.pdf` (MLD 2020), `acbar_af2019.pdf` (Survey of the Afghan People 2019),
`pex.pdf` (Policy Exchange 2016). `scratchpad/islam/scripts/`: `ons_sum.py`, `de_land.py`. About
65 MB in all.

## URLs

- BAMF MLD 2020: https://www.bamf.de/SharedDocs/Anlagen/DE/Forschung/Forschungsberichte/fb38-muslimisches-leben.pdf
- Bundestag Drucksachen quoting MLD (not opened): https://dserver.bundestag.de/btd/21/042/2104291.pdf , https://dserver.bundestag.de/btd/20/066/2006673.pdf
- ONS Cantabular API: https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations?area-type=ltla&dimensions=religion_tb,ethnic_group_tb_20b
- ONS custom dataset, country of birth (extended) and religion: https://www.ons.gov.uk/datasets/RM108/editions/2021/versions/1/filter-outputs/1c282a29-a9fb-4990-891b-c2fe6a7280e4
- Survey of the Afghan People 2019 report: https://www.acbar.org/upload/1575432931309.pdf ; data https://ada.edu.au/survey-of-the-afghan-people/
- Pew, *The World's Muslims* (2012): https://www.pewresearch.org/wp-content/uploads/sites/20/2012/08/the-worlds-muslims-full-report.pdf
- Policy Exchange, *Unsettled Belonging* (2016): https://policyexchange.org.uk/wp-content/uploads/2016/12/PEXJ5037_Muslim_Communities_FINAL.pdf
- MuslimsInBritain.org mosque statistics (2017): https://www.muslimsinbritain.org/resources/masjid_report.pdf
- SCP 2018 summary: https://english.scp.nl/documents/2018/06/07/summary---the-religious-experience-of-muslims-in-the-netherlands
