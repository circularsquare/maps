# United Kingdom: main language, three censuses

Built 2026-10-04 (session d9e44929-uk). `python sources/uk_fetch.py` (about 8 minutes, most of
it the Output Area download) then `python sources/uk_census.py`. Mapping `taxonomy/uk2021.py`,
new nodes `taxonomy/tree.d/uk.txt`, entry `countries/uk.py`. Open Government Licence v3.0
throughout; attribution lines as religiondots' `sources/uk.md` §8.

All three censuses asked "What is your main language?" of everyone aged 3 and over, with
English as a tick box and a write-in. Under-3s ("Does not apply", "No code required") are the
`gap`: 1,893,279 in England and Wales, about 142,000 in Scotland (the OA total of UV212 is
5,294,681 against 5,436,600 people), 66,548 in Northern Ireland.

## 1. England and Wales, ONS Census 2021

The finest categories and the finest geography are never published together (religiondots
spec §3.9):

| table | categories | geography | route |
|---|---|---|---|
| TS024 Main language (detailed) | 95 leaves, hierarchical | 331 LTLAs and up | Nomis bulk `census2021-ts024.zip` (no OA/LSOA/MSOA file exists) |
| `main_language_detailed_26a` | 26 | **188,880 Output Areas** | ONS custom-dataset API, `census-observations` |
| `main_language_23a` | 23 | (not used) | same API |

The API refuses a whole-country OA query ("Too many rows returned"), so `uk_fetch.py` asks
for 300 OA codes at a time (`area-type=oa,E00000001,...`), 630 requests, three at once, about
8 minutes; no area came back blocked. The OA list and the OA -> district key come from ONS's
"Output Area (2021) to LSOA to MSOA to LAD (December 2021) Exact Fit Lookup in EW (V3)" (ArcGIS
item `b9ca90c10aaa4b8d9791e9859a38ca67`), whose 331 districts are TS024's. religiondots'
`oa_lad.csv` is LAD 2023 (318 districts after Cumbria, North Yorkshire and Somerset merged) and
would not join.

**Checks** (printed by `uk_census.py`):
- 188,880 OAs, each with 26 rows; the OA set equals the lookup's.
- OA sums against the same 26 categories fetched at district: 32,839 people of 59.6M differ
  (0.055%), largest cell 54. Each geography is perturbed separately (cell-key perturbation).
- TS024's leaves grouped into the 26 categories against the 26 at district: 2,526 of 57.7M
  (0.0044%), largest 8. **This is the check that the grouping in `ts024_group()` is right**:
  every residual group is the TS024 leaves it claims. Nationally the groups match TS024 to the
  person (e.g. "Any other European languages (EU)" 1,224,532 = TS024's EU 1,815,515 minus
  Polish 590,983).
- TS024's leaves sum to each district's total exactly.

**Allocation.** Fourteen of the 26 categories name one language (English, French, Portuguese,
Spanish, Polish, Russian, Turkish, Arabic, Panjabi, Urdu, Bengali, Gujarati, Tamil, BSL) plus
two single sign categories; they stay `measured` at OA. The ten residual groups ("Any other UK
languages", "Other European language (EU): Any other European languages", "European languages
(non-EU)", "West or Central Asian languages", "Any other South Asian languages", "Mandarin,
Cantonese and other Chinese", "Any other East Asian languages", "African languages", "Any other
languages") are shared among their TS024 members in the OA's district's proportions, tier
`derived`: 2,453,898 people, 4.25% of the drawn total. Only 2 people sat in a group their
district's TS024 has none of; they take England and Wales's mix. The allocation keeps every OA
total exactly.

What the allocation assumes, and where it is weakest: a group's mix is the same in every OA of a
district. That is fine for Romanian or Lithuanian in a district where they are the bulk of their
group, and weakest for small languages inside a big group (Tigrinya beside Somali in the African
group of one London borough). Those dots carry `t` = derived in the dot files; the viewer does
not yet treat derived dots differently (no inferred switch as religiondots has).

**Wales.** The Welsh form had one tick box, "English or Welsh". ONS prints it as "English
(English or Welsh in Wales)", 2,917,644 people in TS024 (2,917,681 summed over the 26a OAs),
and no main-language table splits it. It was first drawn on `indoeuropean` as "language not
named"; **Anita's ruling on ask 004 (2026-10-04) was to split it, a proxy allowed.** Split
2026-10-04 by session d9e44929-uk2, in `uk_census.py`'s `split_wales()`:

- **Proxy:** the same census's `welsh_skills_speak` ("can you speak Welsh?", the TS033
  question), everyone aged 3 and over, which is the main-language universe, at each of the
  10,275 Welsh Output Areas (ONS custom-dataset API, `ew_oa_welsh_speak.csv`). Can speak
  538,326, cannot 2,479,878.
- **Rule, per OA:** Welsh = min(can speak Welsh, box); English = box - Welsh. Both rows tier
  `derived`, under the new keys "English or Welsh (Wales): split as Welsh" / "... as English"
  in `uk2021.py`. No OA hit the cap (speakers never exceeded box answers), so Welsh is exactly
  the 538,326 speakers and English 2,379,355. Asserted: the two parts sum back to the box in
  every OA, neither is negative, every OA's total is unchanged, and no Welsh unit keeps the
  unsplit label (`uk2021.resolve()` raises if one does).
- **Why a count, not a share.** Applying the OA's speaking share to the box (box x
  speakers / all 3+) would put Welsh speakers into other main languages in proportion. They
  are not there: ONS publishes the cross-tab main_language_11a x welsh_skills_speak for 15 of
  Wales's 22 districts (it blocks the other 7 and nearly every MSOA, LSOA and OA, so it
  cannot be used directly), and in those 15, 383,542 of 385,161 Welsh speakers (99.58%) gave
  "English or Welsh" as main language. Summed by those districts, the count rule gives
  385,168 against the cross-tab's 383,542 (abs diff over districts 1,626, worst 269); the
  share rule would give 377,576 (abs diff 5,966). The speak table's 3+ total and the main
  language 3+ total agree per OA to 0.25% (7,430 of 3.02M, worst 7; independent perturbation).
- **Why "can speak" and not a stricter measure.** The census's stricter option is "can speak,
  read and write Welsh" (`welsh_skills_all_6a`, 429,310 in Wales, 80% of speakers). It was
  not used: literacy filters the wrong people for main language. It drops 3- and 4-year-olds
  in Welsh-speaking homes who cannot read yet and older first-language speakers schooled in
  English, while keeping school learners in Cardiff and Newport who read and write Welsh but
  use English most. The 2021 census has no fluency or frequency question. The Welsh Language
  Use Survey 2019-20 measures fluency, daily use and childhood home language (35% of Welsh
  speakers only or mainly spoke Welsh at home as children; 85% of speakers in the north-west
  and 57% in the south-east grew up with a parent who spoke some Welsh), but by region, on its
  own speaker base (it finds far more speakers than the census), and none of its measures is
  "main language"; weighting by it would trade one assumption for another, so it is cited in
  `note_public` as context only.
- **What it means.** The split is an upper bound on Welsh as a main language: everyone whose
  main language is Welsh can speak it, but not everyone who can speak it uses it most. The
  overstatement is surely largest in the south-east and among school-age learners. Reversal
  costs: switch `split_wales()` to the 6a "speak, read and write" category (one line, Welsh
  ~430k), or drop the call to return to the Indo-European remainder.
- **Since 2026-10-06** only the daily speakers among these are drawn as Welsh (306,466; §10).

## 2. Scotland, NRS Census 2022

UV212 Main language at 46,363 Output Areas, from NRS's "2022 output area data" zip
(`Census-2022-Output-Area-v1.zip`, 73 MB; the media path carries a hash and moves): English
5,002,046, Other language 272,820, Scots 13,470, Gaelic 3,536, Sign Language 2,682. NRS perturbs
every cell, so categories sum to 5,294,554 against the "All people aged 3 and over" column's
5,294,681 (abs gap 41,875 summed over OAs, 0.79%); the categories are drawn as published.

**There is no finer table, anywhere I could find.** NRS codes main language into 22 categories
(`Main_language_cat_p`, metadata page) and has a 605-code write-in list, but publishes only the
five. Searched: the OA topic zip (UV212, UV212b by age), the 2025 multivariate OA zip
`OutputArea.zip` (47 tables; the only language one is MV205, country of birth by English
skills), the "search the census" API (`/search-the-census/api/topics?year=2022&CategoryId=1`
lists UV212/UV212a/UV212b only, the same five categories), the topic report PDF, the
council-area and Scotland-level document pages. Since 2026-10-06 "Other language" is split
by country of birth (§9); before that it was drawn on `other`, 273 grey dots.

"Gaelic" here is Scottish Gaelic (NRS's usage) and goes to `indoeuropean.celtic.scottishgaelic`.

## 3. Northern Ireland, NISRA Census 2021

The Flexible Table Builder (`build.nisra.gov.uk`, Cantabular) has `MAIN_LANGUAGE_1000` (20
categories: English, Polish, Lithuanian, Irish, Romanian, Portuguese, Arabic, Bulgarian,
Chinese (not otherwise specified), Slovak, Hungarian, Spanish, Latvian, Russian, Tetun,
Malayalam, Tagalog/Filipino, Cantonese, Other languages, No code required) at Data Zone
(3,780), and `MAIN_LANGUAGE_AGG11` / `AGG3`. Variable names are not guessable: they are listed
in `https://build.nisra.gov.uk/en/metadata/dataset.json?d=PEOPLE`. MS-B12 (the same 20, by
settlement, ward and district) and MS-B13 (100 languages, Northern Ireland only) are the
spreadsheets on the "Census 2021 main statistics language tables" page.

Checks: 3,780 Data Zones, 1,903,157 people (NISRA's 1,903,175). Data Zones against districts
differ by 163 people in all (each level perturbed). MS-B13's 81 rows beyond the 19 named sum to
13,577 against "Other languages" 13,605 in the Data Zones; each Data Zone's "Other languages" is
shared among those 81 in Northern Ireland's proportions, tier `derived`. Tetun (1,573, East
Timorese in Dungannon) is measured at Data Zone and drawn.

## 4. Geography

religiondots' `data/geo/uk/uk_units.gpkg`, read-only: 239,023 polygons with `unit` = OA21CD,
NRS OA code, or DZ2021 code, which are exactly the units here (the scatter found every unit).
No placement layer and no population weight; religiondots draws the UK the same way. The sea
clip reused religiondots' cache. The four code prefixes (E00, W00, S00, N20) are asserted
disjoint in `uk_census.py`.

## 5. Calls

- Wales's "English or Welsh" split per OA by "can speak Welsh" (ask 004 ruling; §1), rather
  than by the stricter "speak, read and write" or a survey weighting.
- Regional remainders whose members cross families go on `other`: "Any other South Asian
  language" (31,028), "Any other African language", "Any other Nigerian language", "Any other
  West African language", "Any other West or Central Asian language", "Any other East Asian
  language", "Oceanic or Australian language", "North or South American language", "Any other
  Eastern European language (non EU)", "Any other European language (EU)" (can hold Basque),
  "Other language". ONS does not publish their members. (Scotland's "Other language" was here
  until 2026-10-06; §9 splits it.)
- "Northern European language (non EU)" (7,876) on North Germanic: non-EU northern Europe is
  Norway, Iceland and the Faroes; ONS does not list its members.
- "All other Chinese" (118,278, mostly people who answered "Chinese") and NISRA's "Chinese (not
  otherwise specified)" on `sinotibetan.sinitic`, drawn as "Chinese, language not named".
- "Bengali (with Sylheti and Chatgaya)" on Bengali: ONS's own merge; Sylheti cannot be shown.
- "Pakistani Pahari (with Mirpuri and Potwari)" on the existing Pahari-Pothwari node.
- "Tagalog or Filipino" on Tagalog; "Romany English" on Angloromani under a new Romani group, and
  "Any Romani language" on that group; "Irish Traveller Cant" on a new `indoeuropean.shelta`.
- Creoles follow the `creole` root br.txt and us.txt set up: "English-based Caribbean Creole" on
  English-based creoles (a group label), Krio a leaf under it, "Any other Caribbean Creole" and
  NISRA's "Creole (Not otherwise specified)" on the root.
- Celtic is flat (us.txt put Irish directly under Celtic), so "Gaelic (Not otherwise specified)"
  sits on Celtic, not on a Goidelic level.
- Sign: BSL, Irish Sign Language and Makaton have nodes; "Any other sign language", "Any sign
  communication system", "Sign Language (Not otherwise specified)" and Scotland's "Sign
  Language" sit on the root.

## 6. Colours

Most UK languages' nodes and colours come from us.txt and tree.txt. Mine: Scots, Ulster Scots,
Welsh, Scottish Gaelic, Tetun, Romani, Shelta, Chadic, Nguni, Kartvelian, the three sign nodes.
Measured over the UK's 40 largest, the closest pairs in OKLab that share London boroughs are all
other files' colours: Turkish (us.txt) against Albanian (us.txt) 0.036 and Panjabi (tree.txt)
0.036, Romanian (us.txt) against Turkish 0.048 and Gujarati 0.052. Turkish, Albanian, Romanian,
Bulgarian and Panjabi/Gujarati live side by side in north and west London, and four of them sit
in red-pink. Not changed here because they are another country's colours; flagged in the report.

## 7. Result

64,836,714 people, 239,023 units, 113 nodes (114 before the Wales split took
`indoeuropean` out); 64,795 dots; 41,714 people under one dot per language nationally. Welsh
545,762 people (538,326 from the split, the rest written in in England, NI and the "Any
other UK language" allocation), about 540 dots, all derived. No rings: every language that reaches no dot is a derived (allocated)
row, and only measured rows may ring, so Cornish (572), Manx, Ulster Scots and the other
small allocated languages draw nothing.

## 8. The grey wedge (2026-10-06, session 5d7dac7e-oth)

Anita: the grey "other languages" segment is big in London. Measured on the dots inside
Greater London's box: the viewer's pie fold (7 languages kept, `index.html` PIE_K = 8) takes 13%
(62 languages; Newham 16%), almost all of them named. The data's own unnamed rows in London are
0.3% (`other`: ONS's "any other South Asian / African / East Asian ... language" residuals,
which TS024's 95 leaves do not split further), plus Arabic 0.8% and Chinese 0.4%, named but on
group nodes (washed in their family colour, not grey). No finer England and Wales table exists.
**Not changed.** Scotland's 272,820 "Other language" is the UK's one large unnamed block; a
country-of-birth split is queued in `followups.md` (done 2026-10-06, §9).

## 9. Scotland's "Other language" split by country of birth (2026-10-06, session 5d7dac7e-sco)

Anita: split the 272,820 by country of birth per area, "ideally with some sort of national level
estimate to fit". `sources/uk_scot_other.py` (docstring has the method), called from
`uk_census.py`'s `split_scotland_other()`. Every row `derived`, keyed "sc:Other language:
<2011 label>" in `uk2021.SC_OTHER`.

**What was drawn before, and what "Other language" holds.** UV212's tick boxes were English,
Scots, Scottish Gaelic, British Sign Language and "Other, please write in". Gaelic (3,536) and
Scots (13,470) were already drawn, as main language only, on their own nodes; the ability
questions (UV208 Gaelic: 69,655 speak it; UV209 Scots: 1,507,996 speak it) are not main language
and stay undrawn (said in `note_public`). "Other language" is every other main language,
Polish included: no immigrant language is a tick box in 2022, so it is all of them, plus Welsh,
Irish and Shelta. NRS still publishes nothing finer (§2).

**Sources** (all OGL v3.0):
- 2022 UV204 country of birth (79 categories, 26 single countries) by 355 electoral wards and
  nationally: NRS's SuperWEB2 extracts on the UK Data Service CKAN (`ukds-ckan.s3...`). Ward
  names only, non-ASCII printed as "?", the two duplicate names given their council in brackets;
  joined to the 2022 ward codes of NRS's Census 2022 Index by a normalised name, asserted 355 to
  355. OA to ward from the same Index's `OA_TO_HIGHER_AREAS.csv`.
- 2022 UV204b, 14 birth regions by Output Area (the OA zip), for placement inside a ward.
- 2011 AT_002_2011 "Language used at home other than English (detailed)", Scotland, 180
  labels, and AT_003_2011 "Country of birth (detailed)": NRS additional tables, gone from NRS's
  current site, fetched from the Wayback Machine (`web.archive.org/web/2016id_/...`). Other
  additional tables are listed by Wayback CDX (`scotlandscensus.gov.uk/documents/additional_tables/*`,
  465 files); the archive refused connections after eight downloads, so only these two were taken.
- England and Wales 2021: TS024 (95 main languages, the `ctry` file, England and Wales rows only:
  it also carries a combined row) and ONS `country_of_birth_190a` at `ctry` (census-observations
  API, one call).

**The 2011 table is not a main-language table.** Scotland 2011 asked "Do you use a language
other than English at home?" (one write-in). Anita's prompt took it for main language; it is
not, and it matters: projected straight to 2022 it predicts 445,292 people against the 272,820
counted, and inflates the languages people in Scotland use at home beside English (French r
1.81, Italian 1.68, Urdu 7.79). So the 2011 table gives the label set and an upper bound, and
England and Wales 2021, which asked the 2022 question, gives the main-language ratio:
- r(language) = speakers / (sum over countries of birth of born x `origin_mix.mix(c, "uk")`
  share). Mix nodes go to the 2011 label whose node is the same or the nearest ancestor (Arabic
  varieties to Arabic); English, Scots, Scottish Gaelic, sign and nodes with no 2011 label
  (Saraiki) are dropped, so the r folds in children born here, English speakers and how people
  name a language.
- 67 of the 138 languages a birthplace explains (96.3% of their 2011 people) have an England
  and Wales ratio; each takes min(r_EW, r_Scotland2011). Nine hit the cap: Gujarati 3.92 to
  0.71 (England's Gujaratis are largely East African Asians, counted under Kenya and Uganda,
  whose mixes lack Gujarati; Scotland had 878 Gujarati users in 2011), Serbo-Croat 8.34 to
  0.92, Mirpuri 2.96 to 1.27, Kurdish, Lingala, Pashto, Romanian, Tamil, Ukrainian. The rest
  take r_2011 x 0.568 (the median r_EW / r_2011).
- 2022's grouped birth categories ("Other EU member countries", "Other Middle East", "South
  America"...) are split into countries in their 2011 proportions. "Accession countries March
  2022" is read as the EU candidates then other than Turkey (Albania, Montenegro, North
  Macedonia, Serbia): its 3,684 matches no other reading of the table's totals.
- **The check: the estimate before any scaling is 259,920 against 272,820 counted (ratio
  1.050).** With the 2011 ratios alone it was 445,292 (0.613). Scaled to 272,820.
- Languages no birthplace explains (Welsh, Irish, Shelta, Romany, "Other languages" 1,921) keep
  their 2011 counts x 0.923, placed flat over the wards' "Other language".

**Fit and placement.** Wards: seed = r x expected speakers from the ward's 2022 births (+2%
flat), fitted to each ward's "Other language" (sum of its OAs) and the national estimate;
worst row gap 0.005 people. OAs within a ward: each language's ward count spread by where the
ward's people born in the regions that feed it live (UV204b; +1% flat), fitted to each OA's own
"Other language" count. Every OA keeps its UV212 total exactly (asserted). Cells under 0.02
people are dropped and refitted: 1,860 people move between languages (Polish +529), 161 of the
164 labels survive. 34,252 OAs, 1,066,034 rows.

**Result** (national, people): Polish 65,261, Spanish 16,364, Punjabi 14,060, Arabic 13,553,
Urdu 12,508, Chinese not named 10,490, Romanian 10,136, Italian 8,862, French 7,890, Lithuanian
6,024, Cantonese 5,844, Russian 5,452, Portuguese 5,296, Greek 5,294, Hungarian 5,214, Bengali
4,456, Tamil 4,343, German 4,315. Polish is 0.87 of Poland-born (England and Wales: 0.82).
Spot checks: Glasgow's 68,898 are Polish 10,696, Punjabi 6,202, Arabic 5,301, Urdu 4,901;
Pollokshields ward Punjabi 1,003, Urdu 639; Edinburgh's Leith Walk Polish 959, Spanish 697,
Italian 394; Aberdeen's Torry/Ferryhill Polish 967, Lithuanian 148; Na h-Eileanan Siar 352,
Polish 65.

**Weak points.** (1) "Chinese (Not otherwise specified)" stays at 10,490 because in 2011 most
China-born wrote "Chinese"; England and Wales's "All other Chinese" agrees (r 1.60). (2) Urdu
against Punjabi: Scotland 2011 had them level (23,394 / 23,230); England's ratios give Urdu
12,508, Punjabi 14,060, so Scotland's Pakistanis are assumed to name main languages as
England's do. (3) Ukrainians mostly arrived after census day (20 March 2022); their few are in
"Other European countries (Non EU)" at 2011's weights. (4) Syria-born, resettled after 2015,
sit in "Other Middle East" at 2011 weights; their language is Arabic either way.

**Reversal**: drop the `split_scotland_other()` call in `uk_census.main()` (one line) and the
grey block returns. New nodes: Kachchi, Runyakitara (`tree.d/uk.txt`); the fragment's
borrowed-node block was regenerated with `origin_mix.py --fragment uk` (138 nodes).

**Result after the split**: 180 nodes (113 before); 64,789 dots; 100 rings, all on derived
rows (the scatter now rings those, placed by area).

## 10. How many Gaelic speakers to draw (2026-10-06, session 5d7dac7e-gd)

Anita: the map's Gaelic is far below most published figures; is it trustworthy? Researched
first (below), then Anita: "home-use rule for uk/ireland would be good". **Implemented** as one
rule for the UK's indigenous languages, `sources/uk_home_use.py` (end of this section).

**What each figure measures** (Scotland, people aged 3 and over unless said):

| figure | people | year | what it is |
|---|---|---|---|
| main language Gaelic (UV212, drawn) | 3,536 | 2022 | one answer, the language used most |
| uses Gaelic at home (AT_002: "Gaelic (Scottish)" 10,443 + "Gaelic (Not otherwise specified)" 14,531) | 24,974 | 2011 | "Do you use a language other than English at home?", one write-in; it presumes English and asks for the other |
| can speak Gaelic (UV208: the three "speaks" columns) | 69,655 | 2022 | ability |
| can speak Gaelic | 57,375 | 2011 | ability |
| any Gaelic skill, including understanding or reading only | about 130,000 | 2022 | ability |
| vernacular community (Soillse, *The Gaelic Crisis in the Vernacular Community*, Ó Giollagáin et al. 2020) | about 11,000 | 2015-17 survey on 2011 census | speakers in the Western Isles, Staffin and Tiree, where Gaelic is still a community language; the figure is as reported, not checked against the book |

Bòrd na Gàidhlig quotes the census ability figures; it has no count of its own. NRS's
*Scotland's Census 2011: Gaelic Report* part 1 (`scotlandscensus.gov.uk/media/cqoji4qx/report_part_1.pdf`):
"25,000 people aged 3 and over (0.49 per cent) reported using Gaelic at home"; 40.2% of Gaelic
speakers did, 73.7% in Na h-Eileanan Siar, 41.5% in Highland, 33.4% in Argyll and Bute, 23.6%
in the other 29 councils. Its AT_261a gives home use by council area (not fetched).

**Closest to the map's definition.** The map draws first or home language. Main language is the
strictest reading: of 69,655 speakers 3,536 (5%) named Gaelic, because nearly every Gaelic
speaker also has English and most use English more, even in the Western Isles (1,746 main
language against 11,463 speakers). The 2011 home-use answer is the nearer match to "home
language"; it is the same kind of question Ireland's Q15 is, and `ie` draws Q15 answers as home
languages. It is 2011, though, and counts people who use Gaelic at home beside English.

**How the neighbours are drawn** (same three measures, where they exist):

| language | drawn | main language | home or daily use | can speak |
|---|---|---|---|---|
| Welsh, Wales | 538,326, ability | not asked apart from English | (Welsh Language Use Survey only) | 538,326 (2021) |
| Irish, Ireland | 71,968, daily outside education | not asked | 71,968 (2022) | 1,873,997 |
| Irish, Northern Ireland | 5,971, main language | 5,971 (2021, MS-B13) | 43,557 speak it daily (2021, NISRA frequency of speaking Irish; may include school) | 126,740 (sum of the frequency answers) |
| Gaelic, Scotland | 3,536, main language | 3,536 (2022) | 24,974 at home (2011) | 69,655 |
| Scots, Scotland | 13,470, main language | 13,470 (2022) | 55,817 at home (2011) | 1,507,996 |
| Ulster Scots, NI | 385, main language | 385 (2021) | none | 190,600 any ability (60% understand only) |

So Welsh is drawn at 100% of speakers (a forced call: the form had one English-or-Welsh box),
Irish in Ireland at a use measure, and Gaelic, Scots, Irish in Northern Ireland and Ulster Scots
at the strict main-language measure. Gaelic is not wrong as a count of main language; it is the
measure that looks lowest beside the other two Celtic languages people will compare it with.

**Option (b), worked out.** 2022 Gaelic speakers per OA x the 2011 home-use share of speakers in
the OA's council group (the four above) gives 24,646 nationally, within 1.3% of the 2011 count,
so the rule can be fitted to 24,974 with little distortion: Western Isles 8,448, Highland 4,994,
Argyll and Bute 1,057, the other 29 councils 10,146 (Glasgow and Edinburgh's learners carry the
low 23.6%). Floor at each OA's own main-language Gaelic (631 OAs have more main-language Gaelic
than the estimate), take the rest from English in the same OA, rows `derived`. It assumes the
2011 shares held to 2022; Soillse's decline in the islands says the Western Isles figure is if
anything high.

**Recommendation** (before Anita's ruling): (b) is defensible and a better match for "home
language" than main language for a language whose speakers all have English, and it brings
Gaelic into line with how Irish is drawn in Ireland; but the same argument moves Scots and
Irish in Northern Ireland, so decide it as one rule. She did.

### The rule as built (2026-10-06)

`sources/uk_home_use.py`, called from `uk_census.main()` after the Wales and Scotland splits.
Each language keeps its main-language count as a floor in every unit; the extra people come out
of the same unit's English (in Wales, out of the "split as Welsh" row into "split as English").
New keys `sc:Gaelic: used at home, beyond main language`, `sc:Scots: used at home, beyond main
language`, `ni:Irish: speaks daily, beyond main language` in `uk2021.py`. Every changed row is
`derived` (this makes most of Scotland's English derived, since Scots speakers are in nearly
every OA); every unit keeps its total (asserted). Reversal: drop the `uk_home_use.apply()` call.

| language | before (main language, or Welsh ability) | after | measure |
|---|---|---|---|
| Welsh, Wales | 538,326 | 306,466 | census speakers x APS daily share, by local authority |
| Scottish Gaelic, Scotland | 3,536 | 24,974 | 2011 home use, placed on 2022 speakers |
| Scots, Scotland | 13,470 | 55,817 | 2011 home use, placed on 2022 speakers |
| Irish, Northern Ireland | 5,973 | 28,817 | 2021 census, speaks Irish daily, less full-time students (below) |
| Cornish, Manx, Ulster Scots | 572, 8, 399 | unchanged | main language; no use measure exists |

- **Welsh.** The Welsh Language Use Survey 2019-20 publishes daily use only nationally (56% of
  its speakers; it could not be analysed by local authority after COVID cut the sample) and has
  no current home-language measure (its "home" figure is childhood home language). So the share
  is the Annual Population Survey's, which asks the same frequency question of its speakers:
  StatsWales "Annual Population Survey - Frequency of speaking Welsh by local authority and
  year" (dataset `f66b78dd-e166-4e8e-9b61-d2a61d0c39e3`, paged from the HTML table; its
  download form 404s to scripts), saved as `data/raw/uk/wales_aps_welsh_frequency_la.csv`.
  Counts pooled over the years ending 31 March 2020, 2021 and 2022 (centred on census day);
  share = daily / (daily + weekly + less often + never). Wales 53.5% (WLUS 56%), Gwynedd
  86.9%, Cardiff 48.2%, lowest Torfaen 31.8%. Applied to the census's speakers per OA, not to
  the APS's own counts: the APS finds far more speakers than the census (daily alone 478,510
  in the year to March 2021), and the census is the count base everywhere else on the map. The
  APS speaker base includes weak speakers, so its daily share may be low for the census's
  (fewer, more fluent) speakers; Welsh is if anything understated now, where it was overstated.
- **Gaelic.** 2022 UV208 speakers per OA x 2011 council-group home-use share gives 24,646;
  fitted (x0.987 with the main-language floor) to 24,974. Western Isles 8,334.
- **Scots.** No council breakdown of 2011 Scots home use was found, so one national share
  (3.34% of 2022 UV209 speakers, fitted to 55,817 with the floor). Scots home use is likely
  more concentrated in the north-east than speakers are; this spreads it evenly.
- **Irish in Northern Ireland.** NISRA Flexible Table Builder, `table.csv?d=PEOPLE&v=DZ21&v=
  IRISH_SKILLS_SPEAK_FREQUENCY` (43,541 daily at Data Zone; NISRA's headline 43,557), saved as
  `ni_irish_speak_frequency_DZ21.csv`. **The daily count includes school use**: the question
  ("How often do you speak Irish?", daily / weekly / less often / never) has no education
  exclusion, unlike Ireland's. By `IN_FULL_TIME_EDUCATION` (`ni_irish_speak_frequency_x_
  student_LGD14.csv`): 14,880 full-time students or schoolchildren, 27,505 others, 1,170 not
  coded (under school age). First drawn whole (43,565); **since 2026-10-06 (session
  5d7dac7e-nii, Anita) the students are taken out** to match Ireland's "daily outside
  education", below.

  **School use out of NI's Irish (2026-10-06).** The Flexible Table Builder crosses
  `IN_FULL_TIME_EDUCATION` x `IRISH_SKILLS_SPEAK_FREQUENCY` at DEA14 (80, complete: 14,886
  daily-speaking students), SDZ21 (850; 123 of 850 daily-student cells blank, known 14,736) and
  DZ21 (3,780; 2,145 blank, known 12,041), saved as `ni_irish_speak_frequency_x_student_
  {DEA14,SDZ21,DZ21}.csv`. No age or economic-activity split was needed. DZ -> SDZ -> DEA
  lookup: the attribute table of religiondots' NISRA DZ2021 shapefile (read-only), saved as
  `ni_dz2021_lookup.csv` (SDZs nest in DEAs, asserted). Placement, top down: each level keeps
  its published cells (scaled down where they exceed the parent) and the blank units share
  the parent's remainder by daily speakers. 14,820 placed (60 lost where a DEA's remainder had
  no blank unit or a DZ's estimate exceeded its daily speakers). Drawn = daily minus students
  per DZ, floored at main-language Irish; the students go back to English in the same DZ.
  Result 28,817 (43,541 - 14,820 = 28,721, plus 96 from the main-language floor). The 1,170
  daily speakers "not coded" (under school age) stay in.
  **Call someone might reverse:** this also removes Irish-medium pupils who speak Irish at home,
  which Ireland's measure would keep (its question asks about use outside education, not about
  being a student). No NI table separates them; the full removal matches Anita's instruction and
  errs low. Reversal: `northern_ireland()` in `uk_home_use.py`, set `want = daily`.
- `data/normalized/uk.csv` (the published tables) does not carry these new inputs; they are
  read from `data/raw/uk/` directly.

Result: check_country ok; 64,788 dots, 101 rings (all on derived rows). After the NI student
removal: check_country ok, 64,789 dots, 101 rings.
