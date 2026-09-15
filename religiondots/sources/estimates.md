# The national estimate layer's inputs — spec §15

Everything under `data/raw/estimates/` is gitignored with the rest of `data/`, so this file is the
re-fetch recipe. Read by `tools/scan_estimates.py`, which is a scout and ships nothing.

## Files

| file | from | notes |
|---|---|---|
| `pew.zip` | `https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/Religious-Composition-2010-2020-dataset.zip` | copied from `data/raw/gr/pew.zip` on 2026-09-14, so the layer and Greece, Spain, France read the same vintage. README last updated 2026-02-12. Four CSVs: rounded counts, unrounded counts, percentages, diversity statistics |
| `WRP_national.csv` | `https://correlatesofwar.org/wp-content/uploads/WRP_national.csv` | 870,713 bytes, 84 columns, 1945–2010 in five-year steps, 194 states in 2010 |
| `wrp-codebook-bibliography.pdf` | `https://correlatesofwar.org/wp-content/uploads/wrp-codebook-bibliography.pdf` | 843 KB. Table 1 is the category list, Table 2 the column layout |
| `COW-country-codes.csv` | `https://correlatesofwar.org/wp-content/uploads/COW-country-codes.csv` | `StateAbb, CCode, StateNme`, 243 rows. The WRP file carries only `state` (the numeric code) and the abbreviation |

Natural Earth 10m admin 0 (`data/geo/ne_10m_admin_0_countries.geojson`, already on disk for
`country_shapes.py`) is the join hub.

## What the two sources are

**Pew is self-identification.** Its methodology page (2025-06-09): *"We rely on how people describe
their own religious identity."* Seven families from more than 2,700 censuses, surveys and registers,
fitted to UN World Population Prospects 2024. The World Religion Database is its primary source only
for places holding 3% of the world (Cuba and North Korea are named); China comes from CGSS. Counts are
published rounded to 10,000, with an unrounded file Pew says to use "with caution". `Other_religions`
holds Baha'is, Daoists, Jains, Shintoists, Sikhs, Wiccans, Zoroastrians and folk or traditional
religions.

**The World Religion Project is not.** Its codebook: self-identification *"does not apply to our
project, since we had to rely mostly on secondary data"*. Each country-year is a reliability-weighted
mean of whatever sources existed, Barrett's *World Christian Encyclopedia* among them, then
interpolated where a year was missing, smoothed where shares jumped, and forced to the population
total using `othrgen` as the balancing column. `datatype` records which of those happened, per row.
`dualrelig` marks rows whose adherents exceed the population; Japan 2010 is 211M on 127M.

The category list is Table 1, verified 2026-09-14: Christianity (Protestant, Catholic, Eastern
Orthodox, Anglican, other), Judaism (Orthodox, Conservative, Reform, other; the codebook says no data
was found for these), Islam (Sunni, Shia, Ibadi, Nation of Islam, Alawite, Ahmadiyya, other),
Buddhism (Mahayana, Theravada, other), and single columns for Zoroastrian, Hindu, Sikh, Shinto,
Baha'i, Taoism, Jain, Confucianism, syncretic, animist, non-religious and other. Rastafari, Druze and
Samaritans were dropped by its expert survey. **Table 2 swaps the labels of `jaingen` and `confgen`;
the data does not.** India's 4.9M Jains are in `jaingen`.

## Joins

- **Pew to ISO**: `Countrycode` is ISO 3166 numeric and matches Natural Earth's `ISO_N3` (falling back
  to `ISO_N3_EH`). Kosovo's 412 is aliased. 195 of 201 join; the six that do not are the Channel
  Islands and France's five overseas regions, which Natural Earth folds into France.
- **WRP to ISO**: `state` to the COW name, then an exact normalised name match against Natural Earth,
  with `COW_ALIAS` in the scan for the rest. All 194 join, none ambiguous, none duplicated.
- **The check is population**, WRP 2010 against Pew 2010 within a factor of 1.33. Three fail and all
  three are real source disagreements rather than wrong twins: Cyprus (814k against 1.13M, whole
  island or not), Equatorial Guinea (698k against 1.19M) and Eritrea (5.23M against 2.95M).
- Eleven WRP states are below Pew's 100,000 threshold and have no Pew total to scale.

## UN Demographic Yearbook table 28, checked for this layer on 2026-09-14

Already on disk as `data/raw/unsd/dyb_table28_values.zip` (fetched 2026-09-08; the URL is in the
UNSD oracle note in memory and `sources.md`). Columns: `Country or Area, Year, Area, Sex, Religion,
Source Year, Value, Value Footnotes`. Filter on `Sex = Both Sexes` and `Area = Total`.

- **117 countries; 90 are already built.** The unbuilt ones are mostly territories, plus Bahrain 2020,
  Brunei 2021, Burkina Faso 2006, Burundi 2008, Guinea 2014, Guinea-Bissau 2009, Maldives 2014,
  Mozambique 2017, Niger 2012, Qatar 2004, Seychelles 2002, Togo 2022 and Zambia 2010.
- **Seven names do not join to Natural Earth exactly** and need aliases: Iran (Islamic Republic of),
  Viet Nam, Micronesia (Federated States of), State of Palestine, Falkland Islands (Malvinas), Saint
  Helena ex. dep., Tokelau.
- **Small religions in each country's latest tabulation**: Baha'i 31 countries, Sikh 13, Jain 4,
  Zoroastrian 4 (Iran 2016 is the only unbuilt one), Confucian 4, Daoism 3, Shinto 3, Ahmadiyya 3,
  Sunni 2, Shia 1, Alevi 0. Nearly all in built countries. So this table cannot supply the small
  religions for unbuilt countries; it can supply a national figure where a built country's drawn table
  is coarser than its national one.

## Shia and Sunni (`islam.shia`, `islam.sunni`): the Gulf and Yemen, 2026-09-14

The rows are in `estimates_hand.py`. Yemen is self-identification from the Arab Barometer; Saudi
Arabia, Kuwait, Qatar, the UAE and Oman are Pew's 2009 compiler ranges, because no self-identified sect
covers any of them; Bahrain is parked. Iran was left for the next batch.

### Inputs

| file | from | notes |
|---|---|---|
| `data/raw/arabbarometer/*.sav` | already on disk; `sources/arabbarometer.py --fetch` has the URLs | waves I to VIII |
| `pew_muslim_population_2009.pdf` | `https://www.pewresearch.org/wp-content/uploads/sites/20/2009/10/Muslimpopulation-1.pdf` | 8.4 MB, 62 pages, `%%EOF` present. Shia table printed p. 10; Appendix B (method) p. 38; Shia by country pp. 39-41; Muslim population by country p. 29 |
| `ABV_Questionnaire_ENG_v2.pdf` | `https://www.arabbarometer.org/wp-content/uploads/ABV_Questionnaire_ENG_v2.pdf` | not saved; Q1012a is on printed p. 39 |

### What the Arab Barometer holds for these seven, read off the rows

Shares are weighted and of Muslim respondents unless marked. A refusal of the sect follow-up stays in
the denominator, as in `sources/iq.md` §5.3.

| cc | wave | item | asked of | result |
|---|---|---|---|---|
| ye | III, Nov-Dec 2013 | `q1012a`: Sunni, Shia, Druze and Christian denominations, no way to name no sect | 1,198 Muslims, all 20 governorates | Sunni 80.06%, Shia 19.02%, refused 0.92% |
| ye | V, 2018-19 | `Q1012A`, volunteered and logged against a master list | 2,399 Muslims, 21 governorates | Shafi'i 35.25%, Just a Muslim 24.37%, Sunni 21.69%, **Alawi 18.15%**, refused 0.41% |
| bh | I, 2009 | `q711` religion, whose card has `shiite muslim (lebanon & bahrain)` and `sunni muslim (lebanon & bahrain)` | 435, no weight column, no subnational variable, all Muslim | Shia 57.24%, Sunni 42.07%, Muslim 0.69% |
| sa | II, 2010-11 | `sa1012` Denomination, a Saudi-only column; `q1012` is empty | 1,394 Muslims in 6 regions | Sunni 76.56%, Hanbali 18.11%, Shi'ite 3.69%, Shafi'i 1.48%, Jaafari 0.16% |
| kw | III | `q2005kw`, *"In the opinion of the field team, the respondent is a member of what sect?"*; `q1012a` empty | 1,021, share of all respondents | Sunni 66.68%, cannot determine 19.00%, Shia 14.32% |
| kw | V, VII, VIII | sect column present and empty for Kuwait | | |
| qa | IV, V | sect column present and empty for Qatar | | |
| ae, om | none | not in any wave | | |

Arab Barometer's methodology page says interviews are with citizens aged 18 and over (read through a
search summary, not re-read at source). That matters little in Yemen and a great deal in the Gulf.

**Yemen's wave V `Alawi` is the Zaydi answer.** The wave V questionnaire marks Q1012a *"DO NOT READ, LOG
ANSWER"* against one master list for every country: Maronite, Orthodox, Catholic, Armenian, Sunni, Shia,
Hanbali, Shafi'i, Ja'fari, Druze, Ahmadiyya, Mozabite, Just a Muslim, Alawi, Other. There is no Zaydi
box, no Yemeni was logged `Shia`, and `Alawi` has the Zaydi highlands' geography: Sa'dah 62.5%, Amran
51.3%, Dhamar 39.9%, Raymah 36.5%, Hajjah 33.8%, Amanat al-Asimah 28.2%, and 0% in Aden, Lahij, Taiz,
al-Bayda, ad-Dali, Shabwah, Abyan and al-Mahrah. Wave III's `Shia` has the same shape: Saada 98.9%,
Sana'a 53.3%, Dhamar 42.3%, the south 0%. Whether the Arabic card printed Zaydi in slot 14 or
interviewers filed Zaydis under Alawi was not checked; the geography is the evidence either way.

**Yemen's range against Pew.** Pew 2009 puts Yemen at 35-40% Shia of Muslims, from its consultants and
the World Religion Database's ethnic ascriptions (Appendix B). The survey reads 18-19%, and wave V's
24.4% `Just a Muslim` sits heavily in the north (Sana'a 46.5%, Amanat al-Asimah 43.9%, Hajjah 36.3%),
so the survey is a floor on Zaydi heritage. The brief puts self-identification first, so the row is the
survey's and Pew's range is in the note, not in `low`/`high`. Anita, 2026-09-14, asked whether to span
both: *"yemen: self identified fine."*

**Saudi Arabia is not "nothing" in this survey, correcting `sources/arabbarometer.py`'s docstring in
part.** `q1012` is empty, as it says, but wave II carries `sa1012` for every Saudi respondent. It still
gives no national figure: the six regions are Riyadh, Mecca, Eastern Region, Asir, al-Jouf and Jazan,
so Najran (the Ismailis) and Medina are unsampled. Eastern Region reads 13.4% Shi'ite and 0.5%
Jaafari, the highest of the six, and Asir 7.0%.

### Pew 2009

Appendix B, p. 38: *"the Sunni-Shia estimates presented in this report are based primarily on data
gathered via ethnographic and anthropological studies, necessitated by the fact that many Muslims either
cannot or will not identify themselves as Sunni or Shia."* So every row from it is `basis="estimate"`.
The same page: *"sectarian differences among Muslims were simplified into two categories: Sunni and
Shia"*, which is why a Sunni row is the rest of Muslims. Not for Oman: p. 9 names *"Kharijites in Oman"*
among groups *"difficult to classify as either Sunni or Shia"*. `islam.ibadi` was added on 2026-09-14
(Anita's call) and ships no Oman row, because the Ibadi shares found for Oman are of citizens: the State
Department's reports put Ibadis at 45% and cite other estimates up to 75%. `www.state.gov` returns 403 to
a script and the Wayback copy was unreachable, so those figures are as a search summarised the 2023
report, not read at source.

| cc | Shia, % of Muslims | page | Pew 2009 Muslims (p. 29) | Pew 2020 Muslims |
|---|---|---|---:|---:|
| sa | 10-15 | 41 | not extracted | 28,726,928 |
| ye | 35-40 | 41 | 23,363,000 | 36,085,366 |
| kw | 20-25 | 40 | not extracted | 3,530,671 |
| bh | 65-75 | 39 | 642,000 | 1,097,795 |
| ae | ~10 | 41 | 3,504,000 | 6,891,776 |
| om | 5-10 | 40 | 2,494,000 | 3,698,828 |
| qa | ~10 | 41 | 1,092,000 | 2,127,531 |

Every row uses `of="islam"` (added to `estimates.py` for this batch), so the share multiplies Pew's 2020
Muslims. **That assumes the 2009 share survived the growth**, and for Qatar and the UAE the Muslim count
roughly doubled, mostly by migration whose sect nobody measures. The notes say so.

### Bahrain, left off (Anita, 2026-09-14)

**Pew's share cannot be carried to 2020.** 65-75% of Pew 2020's 1,097,795 Muslims is 713,567 to 823,346
Shia. Bahrain's 2020 census counts **712,362 Bahrainis** in 1,501,635 residents (CIO/iGA, via the Gulf
Labour Markets and Migration table *Population by nationality group at dates of census*), so even the
low end is every citizen. Pew's 2009 base was 642,000 Muslims, before the 2010 census found the migrant
population, and its share describes that smaller base.

**The self-identified figure is citizens only.** Wave I's 57.24% Shia, on a sample that is all Muslim,
is about 408,000 Bahraini Shia, 27% of residents, with the sect of the 53% who are not citizens
unmeasured. Offered the choice between shipping that as a labelled floor and leaving Bahrain off, Anita
left it off: *"only citizens is pretty different from what we're trying."* That is now a rule in
`estimates_todo.md`.

## Shia, Sunni and Ahmadiyya from Pew's sect question: Pakistan, Bangladesh, Egypt, Jordan, Kenya, Ghana, 2026-09-14

Anita, 2026-09-14: *"we can add the national sunni/shia figures for sure."* The rows are in
`estimates_hand.py`, and `sources/branches.md` has the sweep that found them. Every row is `self_id`,
`of="islam"`, and a single figure (low == high). "Just a Muslim", nothing and don't know get no row and
stay on `islam` (spec §2.7a), and a 0 gets no row, so Egypt and Jordan have no Shia row.

### Inputs

| file | from | notes |
|---|---|---|
| Pew 2012, *The World's Muslims: Unity and Diversity* | `https://www.pewresearch.org/wp-content/uploads/sites/20/2012/08/the-worlds-muslims-full-report.pdf` | 164 pages, `%%EOF` present, not saved. Sect table p. 30; Appendix C (method) pp. 117-127, Muslim sample sizes p. 120, per-country sample design and fieldwork pp. 121-127; topline Q31 p. 128 |
| Pew 2010, *Tolerance and Tension: Islam and Christianity in Sub-Saharan Africa* | `https://www.pewresearch.org/wp-content/uploads/sites/7/2010/04/sub-saharan-africa-full-report.pdf` | 331 pages, `%%EOF` present, not saved. Sect table p. 21; sample sizes p. 17 and p. 66; method pp. 65-70; the topline is appended, Q37's wording on p. 134 and each country's religion by sect (shares of all adults) on pp. 135-153 |
| Pew 2010 topline alone | `https://www.pewresearch.org/wp-content/uploads/sites/7/2010/04/sub-saharan-africa-topline.pdf` | 254 pages, the same topline pages. The download broke off once (curl exit 18); a retry got the whole file |
| Pew 2009, *Mapping the Global Muslim Population* | the Gulf section's table above | Pakistan's Shia on p. 40 |

### The question

Pew 2012 Q31 and *Tolerance and Tension* Q37 have the same wording: *"Are you Sunni (for example, Hanafi,
Maliki, Shafi, or Hanbali), Shia (for example, Ithnashari/Twelver or Ismaili/Sevener), or something
else?"* The 2010 questionnaire adds a show card and tells the interviewer to probe for the one identity
a respondent holds most when they name several. The schools are examples inside the Sunni answer, so no
row reaches a school (§2.6) and no Shia row reaches `jaafari`. Pew 2012's sub-Saharan rows are
*Tolerance and Tension*'s answers reprinted (p. 30's footnote), not new interviews.

### The figures, % of Muslims, read at source

Pew 2012 p. 30, checked against its topline (p. 128) read by word position. The topline has a column
for a volunteered Ahmadiyya answer, which p. 30 folds into "something else".

| cc | Sunni | Shia | Ahmadiyya | something else | just a Muslim | nothing, DK, refused | Muslims asked | fieldwork | shipped, of Pew 2020's Muslims |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| pk | 81 | 6 | 0 | 1 | 12 | 0 | 1,450 | Nov. 10-30, 2011 | Sunni 180m, Shia 14m |
| bd | 92 | 2 | 0 | 0 | 4 | 2 | 1,918 | Nov. 21, 2011 to Feb. 5, 2012 | Sunni 140m, Shia 3m |
| eg | 88 | 0 | 0 | 0 | 12 | 0 | 1,798 | Nov. 14 to Dec. 18, 2011 | Sunni 92m |
| jo | 93 | 0 | 0 | 0 | 7 | 0 | 966 | Nov. 3 to Dec. 3, 2011 | Sunni 9.8m |
| ke | 73 | 8 | 4 | 0 | 8 | 7 | 340 | Dec. 18-27, 2008 | Sunni 4.2m, Shia 460k |
| gh | 51 | 8 | 16 | 0 | 13 | 11 | 339 | Jan. 17-30, 2009 | Sunni 3.2m, Shia 510k, Ahmadiyya 1m |

*Tolerance and Tension* p. 21 prints Ahmadiyya as its own column and agrees for both: Ghana 51, 8, 16, 0,
13, 11; Kenya 73, 8, 4, 0, 8, 7. Its topline's shares of all adults fit: Ghana (p. 141) Muslim 11, of
whom Sunni 6, Shia 1, Ahmadiyya 2; Kenya (p. 143) Muslim 11, Sunni 8, Shia 1, Ahmadiyya 0.

**Kenya's 4% Ahmadiyya has no row.** The batch as approved named Ghana's Ahmadiyya only, and p. 30
shows Kenya's as "something else", so it was not seen until the topline was read. It belongs to the
`open` Ahmadiyya item in `estimates_todo.md`.

### Coverage, and what each row's note says

- **Pakistan's Shia 6% is a floor.** The sample left out FATA, Gilgit-Baltistan, Azad Jammu and Kashmir
  and unstable parts of Khyber Pakhtunkhwa and Balochistan, and represents 82% of adults (p. 125). Pew
  2009 gives 10-15% of Muslims (p. 40), compiled from ethnographic sources rather than asked. Under the
  Yemen ruling that range goes in the note and not in `low`/`high`. No Ahmadiyya row: Pakistan's census
  counts Ahmadis, and Pew's topline reads 0.
- **Bangladesh**: all seven divisions. The State Department's "91 percent" Sunni is its gloss on a census
  that asks no sect (`sources/branches.md`).
- **Egypt**: 24 of 29 governorates; the five frontier ones, 2% of the population, were left out (p.
  122). No Shia row: 0% in Pew 2012. The Global Flourishing Study's sect item reads 25.6% Sunni and 72.7%
  just a Muslim in the same country, which is how far the answer card moves the named share.
- **Jordan**: all 12 governorates. No Shia row: 0% in Pew 2012. The fieldwork predates most Syrian
  refugee arrivals, which Pew's 2020 Muslims include.
- **Kenya and Ghana**: small samples and 2008-09 fieldwork. Each is a national sample of 1,300 plus 200
  extra interviews with Muslims, margin of error ±7 points among Muslims (p. 66), and the Muslim samples
  are 56% or more male (p. 65).
- **Every row assumes the survey's share held until 2020**, since it multiplies Pew's 2020 Muslims.

### Why these six and not the rest of Q31's built countries

Q31 covers 20 built countries (`sources/branches.md` has the table). Anita's call left off:

- **Indonesia, Kazakhstan, Kyrgyzstan and Uzbekistan**: most Muslims said just a Muslim or gave no
  answer (Indonesia 56 and 13, Kazakhstan 74 and 10, Kyrgyzstan 64 and 12, Uzbekistan 54 and 26). A row
  would put a minority figure on `islam.sunni` beside Muslims who are overwhelmingly Sunni.
- **Malaysia**: the Shia 0 sits under a fatwa ban on Shia teaching, and is not evidence of absence.
- **Nigeria**: its Shia 12% is above other published estimates and is a §14 question in its own right;
  `ng.md` §5 refused Shia below national.

Türkiye, Iraq and Russia need nothing: their own sources already draw Sunni and Shia (Türkiye at the
schools), so §15.3 would refuse a row, as it would India's, which draws Sunni, Shia and Ahmadiyya.
Pakistan draws `islam.ahmadiyya` from its census, which is the other reason it has no Ahmadiyya row.
Ethiopia, Uganda and Liberia were not in the batch as approved, and nothing here rules them in or out.
Thailand's sample is Muslims in five southern provinces (p. 120), so it is not national. Albania, Kosovo
and Bosnia have a no-sect majority too.
