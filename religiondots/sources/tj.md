# Tajikistan: drawn 2026-10-03 as an ethnicity model at five regions; the census asks and publishes nothing

**Drawn 2026-10-03** by session `fafd1067-tj`, from `sources.md` §scout-2026-10-03-negatives, on
Anita's rulings of 2026-09-15 (priority holes) and 2026-09-16 (a country nothing places is drawn on
the best figure there is, method disclosed). `sources.md` §tj-2026-10-03 is the summary. The new
sections are §7 (the model), §8 (Gorno-Badakhshan's Ismailis, ask 051) and §9 (what reopens it).
Code: `sources/tj_geo.py`, `sources/tj_grid.py`, `sources/tj.py`, `taxonomy/tj2020.py`,
`countries/tj.py`. Raw files in `data/raw/tj/`.

§1-§6 below are the closing record of 2026-09-08 and stand as written; §5's objection was to
drawing LiTS III's composition, and the model in §7 does not use LiTS for anything.

---

## The 2026-09-08 record: NOT DRAWN. The census asks and publishes nothing; three surveys ask and none of them can see the country

Closed 2026-09-08, `sources.md` §11ak. **No `sources/tj.py`, no `taxonomy/`, no row in
`countries.py`.** The queue sent Tajikistan to the Life in Transition Survey with a warning
that it would be *"a five-unit map that is 99.5% one colour"*. It is worse than that, and it is
worse in a way that is worth the file: the state's own census **does** ask religion, the answer
has never been published at any level, and every open survey that reaches Tajikistan either
skips the question by design or returns a composition that is checkably wrong.

| | |
|---|---|
| office | Agency on Statistics under the President of the Republic of Tajikistan, `www.stat.tj` |
| censuses since independence | **2000, 2010, 2020** |
| religion question | **YES, question 7 of Form 2 in 2020**, and for the first time ever |
| published religion | **none. Not one figure, at any geography, including the nation** |
| enumerated instead | 3,484 media-library files, 417 live data-portal indicators, 252 reporting forms, 29,385 archived URLs |
| licence | **CC BY 4.0**, footer of every `stat.tj` page, and `data.stat.tj` repeats it |
| surveys tried | LiTS III (drawable coverage, unusable composition), Central Asia Barometer waves 1-14 (question excluded from Tajikistan by design) |

---

## 1. THE 2020 CENSUS ASKS RELIGION, AND HERE IS THE FORM

`[[reference_census_questionnaire]]` in its most useful direction. Kyrgyzstan's questionnaire
(§9co) closed that country by showing no religion item exists. Tajikistan's shows the opposite.

**Form 2, ПЕРЕПИСНОЙ ЛИСТ**, the individual census questionnaire, *Утверждена Приказом
Агентства по статистике при Президенте Республики Таджикистан от "6" августа 2019 года, № 30*:

```
    5. Национальность     1 таджик  2 узбек  3 русский  4 киргиз  5 другое (указать)
    6.1 Родной язык       1 таджикский 2 узбекский 3 русский 4 киргизский 5 другое
    7. ВЕРОИСПОВЕДАНИЕ    1 ислам   2 христианство   3 неверующий
                          4 отказ от ответа          5 другое (указать)
    8. Гражданство ...
```

Ten forms in the packet; Form 1M is the list of temporarily resident foreign citizens and
Form 2 is the personal sheet. **Question 7 is the whole religion programme**: a five-box card
with a write-in, no school of Islam, no denomination of Christianity, and *refusal* printed as
a code rather than left as item non-response.

The form is at `http://stat.ww.tj/pages/Переписные формы 2020 года_рус.pdf`, which now returns
403 from that host; the Wayback capture `20220127073818` is the whole file, 5,639,924 bytes,
ten pages of scanned images with no text layer. **The earlier `20211208061656` capture is
truncated at exactly 1,048,576 bytes** and PyMuPDF opens it with `page_count = 0`, which is
`[[reference_pdf_truncated_at_source]]` exactly; the Tajik-language sibling capture is
truncated at the same 1 MiB. Always take the largest capture the CDX API reports.

Two contemporary reports confirm the change was deliberate and name the official who announced
it. Fergana, 9 July 2019, and Radio Ozodi both quote Kiyomiddin Davlatzoda of the Agency:
*"во время предыдущих переписей населения вопросы о вероисповедании в анкетах отсутствовали"*.
A deputy, Abdulhalim Gafforov, described the purpose as establishing how many followers Islam
and Christianity have and how many non-believers there are. **The news is the lead; the printed
form above is the evidence.**

## 2. AND THE OFFICE HAS PUBLISHED NONE OF IT, ENUMERATED FOUR WAYS

§11aj's rule: *a country closed on an enumerated database stays closed until the database
changes; a country closed on documents was never closed at all.* Tajikistan has no SDMX
endpoint and no indicator API, but it has four things that enumerate.

**(1) The census release itself.** `stat.tj/ru/perepis-naseleniya-i-zhilishhnogo-fonda-2020/`
lists the final results as **ten volumes** and links 75 PDFs. Volumes I, II, IV, V, VI, VII,
VIII, IX and X are all there with their tables. **Volume III — *Национальный состав, владение
языками и гражданство* — is an empty `<details>` element**: the heading renders, the table list
is `<p></p>`. There is no religion volume at all. CISSTAT holds the Agency's own 33-page
dissemination presentation for the same release, which walks every volume's contents; the
strings `religio`, `faith` and `confess` occur zero times in it, and its Volume III page names
nine tables, **every one of them national** rather than by region.

**(2) The WordPress media library.** `stat.tj` is WordPress; `[[reference_wordpress_media_api]]`.
`wp-json/wp/v2/media` reports `X-WP-Total: 3486` and pages out **3,484 attachments** in full,
no `search=` filter. Grepped over titles and file URLs for
`религи|вероиспов|конфесс|мусульман|ислам|правосл|христиан|буддис|атеис|мазҳаб|эътиқод|миллат|национальн|этнич`:
**one hit, and it is the word `динамика` inside a report on the middle class.**

**(3) The data portal, including the pages nothing links to.** `data.stat.tj` is a .NET
application titled *Data portal*, licensed CC BY 4.0, with fifteen subject links
`/Home/index2/<1-15>`. **All fifteen return the identical document** — the filtering is
client-side — so the linked catalogue is **265 indicators**, ids 31 to 495. The viewer route
`/Home/show/<int>` takes a bare integer, so it was swept 1 to 700
(`[[reference_cms_download_id_sweep]]`): **417 ids return a live indicator page, so 152 of them
are linked from nowhere.** Not one of the 417 is religion, and none is ethnicity either; the
unlinked 152 are oil and gas extraction, life expectancy, poverty rate and the rest of the SDG
tail.

**(4) The forms catalogue.** `catalog.stat.tj`, *Каталог форм*, is the register of statistical
reporting forms: **252 forms**, ids 1 to 999, and sweeping `/Home/Form/<int>` from 1 to 1,100
returns exactly the 252 that are linked, so there is no hidden tier here. **Two mention
religion and both are about buildings and organisations, not people**: form 93, *Шакли 1-ИД
ҷамъбастӣ (нимсола), Ҳисоботи ҷамъбастӣ оиди иттиҳодияҳои динӣ*, and form 197, *Шакли 1а-ИД
(нимсола), Ҳисобот оиди фаъолияти иттиҳодияҳои динӣ* — the consolidated and the activity report
on **religious associations**. Same shape as Kyrgyzstan's four CKAN packages (§9co), and
`[[feedback_proxy_residual_nameable]]` rules it out for the same reason: the part that does not
match is not a published number anyone can weight by.

**And the Wayback CDX for `stat.tj*` returns 29,385 distinct captured URLs**, plus 819 for
`old.stat.tj*`. Unquoting each and grepping the **filename** for the same vocabulary returns no
document. So this is not "the file has not been found on the current site".

### The dead and half-dead hosts, for whoever comes back

- **`old.stat.tj`** answers on every path with a Laravel *Whoops! There was an error* 500. It
  was live as recently as mid-2025 and its archived census page is where the questionnaire link
  was found, so **the Wayback copy is the working version of that site**.
- **`stat.ww.tj`** and `oldstat.ww.tj` are the predecessor hosts that actually served the census
  documents. `stat.ww.tj` now returns 403 to a browser user-agent on a URL the archive proves
  was public.
- **`nada.stat.tj`** is linked from the current homepage and 404s at `/`, `/index.php`,
  `/catalog` and `/index.php/catalog`. A NADA microdata catalogue is the single most valuable
  thing that could appear at that name; **it is worth re-probing on any future pass.**
- **`52.54.6.130/portal/`**, an AWS address carrying the main menu's *Официальная статистика*
  link, times out on port 80 rather than refusing.
- `pricetool.stat.tj`, `wef.stat.tj` and `office.stat.tj` answer but are a price tool, an
  expired certificate and a one-page respondent login.

**No BI engine anywhere.** Fetched as raw HTML rather than through WebFetch, which strips
iframes (§9cd): `stat.tj` in three languages, the census page, `data.stat.tj` and
`catalog.stat.tj`. The only `iframe` token on any of them is a JSON config key for a
scroll-to-top plugin, and there is no Qlik, Power BI, Tableau, Superset, Metabase or ArcGIS
string on the domain.

## 3. LiTS III HAS THE COVERAGE AND CANNOT MEASURE THE COMPOSITION

`data/raw/lits/lits_iii.dta`, read through `sources/lits.py`. Coverage is genuinely fine:
1,510 respondents, 75 PSUs, **5 of 5 regions**, against Uzbekistan's ten of fourteen.

```
    region      PSUs    n        religion answers
    Khatlon       27   543       543 MUSLIM
    Sughd         25   502       499 MUSLIM, 2 OTHER, 1 OTHER CHRISTIAN
    RRP           16   323       321 MUSLIM, 1 BUDDHIST, 1 OTHER CHRISTIAN
    Dushanbe       4    82        79 MUSLIM, 1 CATHOLIC, 2 Refusal
    GBAO           3    60        60 MUSLIM
```

Weighted, that is **99.46% MUSLIM** and six substantive non-Muslim respondents in the whole
country. It is not the thinness that closes this; it is what the thinness is made of.

### THERE ARE ZERO ORTHODOX CHRISTIANS IN IT, AND THAT IS A COVERAGE FAILURE RATHER THAN A FINDING

**The same instrument, the same round, the same 75-PSU design and the same 1,500 respondents
returned 107 Orthodox in Kyrgyzstan and 467 in Kazakhstan. In Tajikistan it returns none.**
Tajikistan has a Russian Orthodox diocese with functioning parishes in Dushanbe, Khujand and
Qurghonteppa; every outside estimate of the country's Christians is in the tens of thousands.
At n=1,510 a population of even 0.5% has an expected count near eight. Zero is not a
measurement that Tajikistan has no Orthodox Christians; it is the survey not reaching them.

A country whose largest non-Muslim group returns an exact zero cannot have its non-Muslim half
drawn from that survey, and its non-Muslim half is the entire content of the map below `islam`.

### AND WHAT WOULD BE DRAWN INSTEAD IS AN ARTEFACT THE PROJECT HAS ALREADY NAMED

Run through `lits.build`, the six substantive answers become national-rate shares:
`buddhism` 0.083%, `christianity.catholic` 0.064%, bare `christianity` 0.062%, `other.tj`
0.148%. **That would put more Buddhists in Tajikistan than Christians of any kind**, on the
strength of one rural respondent in the Districts of Republican Subordination — which is the
identical signature §9co diagnosed as a keying artefact in Kyrgyzstan, where sixteen `BUDDHIST`
answers landed in the two most rural and most uniformly Muslim oblasts. There the split-half
caught it and demoted it. **Here it cannot**: a category present in one PSU has no split, so
`lits.stability` returns *no test possible* rather than a verdict, and the artefact would be
drawn at face value. `[[reference_check_needs_power]]`.

## 4. THE CENTRAL ASIA BAROMETER HAS AN ISMAILI CODE, AND IT STILL CANNOT SEE GORNO-BADAKHSHAN

This is the part worth carrying out of the country. **New source for this project**, opened
here and recorded in `sources.md` §11ak: the Central Asia Barometer, run twice a year since
2017 in Kazakhstan, Kyrgyzstan, Tajikistan, Uzbekistan and (waves 4-6) Turkmenistan, about
1,500 respondents per country per wave.

`ca-barometer.org` gates its downloads behind a name/email/institution form. **The Discuss Data
mirror does not**: dataset `1d10e56e-540b-4751-96b6-885309cb4b1d`, *CAB Survey Waves 1-14
v1.1*, Open Access, and `/dataset/<uuid>/files/<file-uuid>/` returns the archive body directly
with no account, no cookie and no form. All fourteen are on disk at `data/raw/cab/`, about
167 MB, waves 1-9 as `.zip` and 10-14 as `.rar`. Each archive carries, per country, a Stata
file, an SPSS file, an Excel file, an English and a local-language questionnaire and a methods
report. **That is 22,050 Tajik interviews with a region variable in every wave.**

### The question is excluded from Tajikistan by design, and the instrument says so out loud

Scanning every wave's Tajik Stata file for any variable whose name or label mentions religion:

```
    W01 2017 spring   TelReligion_M  531 answered   TelMuslim  518 answered
    W02 2017 autumn   TelReligion_M  441 answered   (no Muslim-school follow-up)
    W03 .. W14        NOTHING
```

The wave 1 English master questionnaire heads the block **D-4a. *(Excluding Tajikistan)* What
is your religious affiliation?** and **D-4b. *(Excluding Uzbekistan and Tajikistan)* Are
you… Sunni / Shia / Ismaili Muslim**. Nine waves later the exclusion is still there and is now
printed in the delivered data: wave 14's Stata label for `DD13` is literally *"[KAZ, KGZ, TUR]
What is your religious affiliation? Do you consider yourself"*, and all 1,500 Tajik rows hold
code 95, **Not Asked**. `MM10 Region` is populated in every wave, so it is the religion question
that is withheld from Tajikistan and not the geography.

The only Tajik religion data in the whole series is a **telephone re-contact module** attached
to waves 1 and 2. Those `Tel*` columns are the *same respondents*: `IDNO` matches `TelIDNO` for
all 531 rows, `Educ` matches `TelEduc` in 99.4% and the `EducLevel_M` × `TelEducLevel_M`
cross-tab is perfectly diagonal, while the month and interview length differ. **So `Region_M`
does legitimately belong to the phone answers** — worth checking rather than assuming, because
a `Tel`-prefixed duplicate variable set is exactly the shape of a positionally-stacked second
sample (`[[reference_pooled_survey_labels]]`).

### And here is what it measures

Wave 1, the one wave with the school follow-up, `TelMuslim` by region:

```
                                Sunni  Shia  Ismaili  Don't know  Refused   Muslims asked
    Gorno-Badakhshan               40     3        3           5        0        51
    Khatlon                       126    14        6          36        1       183
    Sughd                         133     6        6          24        2       171
    RRP                            75     7        1          15        3       101
    Dushanbe                       10     1        0           1        0        12
    ------------------------------------------------------------------------------
    all Tajikistan                384    31       16          81        6       518
```

**Gorno-Badakhshan is overwhelmingly Ismaili** — it is the seat of the Nizari Ismaili community
in Central Asia, the Aga Khan's foundations are its largest civil institution, and the Pamiri
peoples who make up almost the whole of its population are Ismaili. **This survey, with an
Ismaili box on the card, records 40 of its 51 Badakhshani Muslims as Sunni and three as
Ismaili.** Nationally it puts fourteen Shia in Khatlon and seven in the Districts of Republican
Subordination — regions with no Shia population anyone describes — against three in the one
region where Shia Islam is the majority faith, and 15.6% of the Muslims asked answer *Don't
Know*.

**The Uzbekistan agent's finding was that the LiTS card has no Ismaili code. This is the
stronger version of the same fact: the code exists, and the answers still come back wrong.**
Whether that is a face-to-face-then-telephone interview in a country where the state has
prosecuted religious expression, or interviewer coding, or *Sunni* being heard as *ordinary
Muslim*, is not something this data can settle. What it settles is that no weighting of these
answers produces a religious geography of Tajikistan.

## 5. Why the country is not drawn, stated plainly

Three instruments ask, and none of them can carry the map.

- **The census** asks the right question of 9.5 million people and has published no answer, at
  no geography, in five years. Its card would not separate Ismailis either — question 7 has one
  `ислам` box — but its `христианство` and `неверующий` counts by region would be a first-class
  source for four of this map's nodes.
- **LiTS III** has the regional coverage and returns a composition with an exact zero where the
  country's largest minority faith should be, and a Buddhist cell that the project has already
  learned to read as an artefact.
- **The Central Asia Barometer** excludes the question from Tajikistan in twelve of fourteen
  waves, and in the one wave that asked the school of Islam it measures Gorno-Badakhshan as
  three-quarters Sunni.

Drawing Tajikistan from LiTS would put roughly 8,000 Buddhists and zero Orthodox Christians on
a map of a country that has the reverse, and would render Gorno-Badakhshan identical to
Khatlon. §14.3's rule — never put people on a node the source cannot see — is satisfied by
mapping `MUSLIM` to bare `islam` the way `kg2016.py` does, but §14.3 does not license the other
half, which is drawing four minority nodes whose levels are set by six respondents and one
known-zero cell. **A blank Tajikistan says nothing. A drawn one says something false.**

`[[feedback_no_granularity_floor]]` says coarse geography is never a reason to skip a place, and
it is not the reason here: five units would have been fine. The reason is that the composition
inside those units is not measured.

## 6. WHAT REOPENS IT, AND THE FIRST ONE IS A PUBLICATION AND NOT A DISCOVERY

- **The 2020 census religion table.** It was collected from the whole population on a printed
  form. The Agency has published nine of ten volumes with their tables and left Volume III's
  list empty, so the release programme is visibly unfinished rather than closed. **A single
  region × religion table turns Tajikistan into a census country with five categories** —
  Islam, Christianity, non-belief, refusal and a write-in — which is a better card than several
  countries already on this map. Ask the Agency's dissemination unit; the licence is CC BY 4.0
  and nothing about the request is unusual.
- **`nada.stat.tj`**, if it ever answers. That hostname is linked from the Agency's own
  homepage and is the conventional address of a NADA microdata catalogue. Census microdata with
  question 7 in it would settle the country outright.
- **Volume III when it lands**, even without religion. Nationality by region would let
  Tajikistan be scored as a §9aq ethnicity model — but note the ceiling: the census's own
  nationality card is таджик / узбек / русский / киргиз / другое, with no Pamiri code, and in
  CAB wave 1 all 150 Gorno-Badakhshan respondents are coded ethnically **Tajik** while 37 of
  them give **Shughni** as the language spoken at home. **The language question is the only
  place in any of these instruments where the Pamiri population is visible.**
- **A survey that asks the school of Islam face to face and is not the Barometer.** The Life in
  Transition round after III would need the religion question back; §11aj established LiTS IV
  does not have it.

---

## 7. THE BUILD, 2026-10-03: AN ETHNICITY MODEL, AS AZERBAIJAN

§6's third bullet, reached by another route: the 2020 Volume III is still empty, but **the 2010
Volume III is on the Wayback Machine** and is a 537-page PDF with a text layer.

| | |
|---|---|
| 2010 nationality | *Национальный состав, владение языками и гражданство населения Республики Таджикистан*, Том III (Agency, 2012), `web.archive.org/web/20131014054442/http://www.stat.tj/ru/img/526b8592e834fcaaccec26a22965ea2b_1355501132.pdf`, 13,682,802 bytes, saved as `census2010_vol3.pdf`. Linked from ru.wikipedia's 2010 census article. |
| what it has | pp.7-11 every nationality (92 rows plus India/Pakistan, other, not stated), national only; pp.108-115 seven groups (Tajiks, Uzbeks, Russians, Kyrgyz, Turkmens, Tatars, Kazakhs) by the five regions. **Nothing below region**: every regional table in the volume is the same five. |
| 2020 nationality | shares only, on the Agency's own slides: UNECE workshop, September 2023 (`unece2023_WS10RizoevENG.pdf`, slide 27, via Wayback; unece.org is behind Cloudflare): Tajik 86.1, Uzbek 11.3, Kyrgyz 0.4, Russian 0.3, other 1.9. The CISSTAT deck of July 2023 says 81 nationalities and 86.1% Tajik. |
| 2020 population | Volume I table 1 (`census2020_vol1_table1.pdf`), permanent population by region and district, 9,657,005; it also prints the 2010 column, which equals Volume III's region totals exactly. |
| pop-stat.mashke.org | `tajikistan-ethnic2010.htm` transcribes the same regional table (seven groups plus other, with urban and rural); it agrees with the PDF and is not used. |

**The construction** (`sources/tj.py` docstring has the table): Russians and Tatars on their own
2010 regional counts; Ukrainians, Belarusians, Germans, Armenians, Georgians, Jews and about forty
other non-Muslim-heritage nationalities (Koreans, Chinese, Ossetians, Moldovans ...) on the
Russians' regional distribution, each region's 2010 `other` column asserted to hold them; all scaled
by 28,971/34,838 = 0.832 (the Agency's 0.3% Russian in 2020, whose rounding alone spans 24,000 to
34,000); religion from Kazakhstan's 2021 census for Russians, Tatars, Ukrainians, Belarusians and
Germans (refusals and the small answers out, as `az.py`); Armenians, Georgians, Jews religio-ethnic;
the rest `unknown`; every Muslim-heritage nationality, the nine Uzbek tribal rows (Lakai 65,555,
Kongrat 38,078 ...) included, on `islam` as the residual of each region.

**Checks:** the 92 national rows sum to 7,564,502; the seven regional groups sum to their national
rows and to p.7; Volume I's 2010 column equals the regions; the build reproduces the 2020 count per
region exactly.

**Result** (`data/normalized/tj.csv`): islam 9,622,434 (99.64%), christianity.orthodox 29,641,
unknown 2,464, secular 2,000, Armenian 361, Georgian 76, judaism 29. Not on Islam: Dushanbe 1.94%,
Sughd 0.33%, Districts of Republican Subordination 0.13%, Khatlon 0.12%, Gorno-Badakhshan 0.03%.
9,655 dots at 1:1,000.

**Pew is the witness, and it is three times higher.** Pew 2020: 9,749,311; Muslim 98.91%,
Christian 97,515 (1.00%), unaffiliated 7,268. Appendix A p.24 sources it to Pew's Survey of the
World's Muslims, 2011-12, with UN WPP for the population. 1.0% of about 1,500 respondents is about
fifteen people; the census counted 34,838 Russians in 2010 (0.46%), and Kazakhstan's coefficients
make about 92% of them Orthodox. The difference (about 67,000) is not drawn: no source places it.
Christian converts of Muslim-heritage nationality, however many there are (no figure was opened),
are not visible to a nationality model; `gap` names them, without a figure.

**Calls someone might reverse.** Germans on Kazakhstan's German row (89.1% Orthodox after refusals
and Catholics are out) rather than `unknown`; Tatars at Kazakhstan's 28.7% Orthodox, as `az`;
Koreans on `unknown` rather than Kazakhstan's Korean row (Protestant-heavy, which the kept columns
would drop); the scaling of every minority by the Russians' fall, when the Chinese rose from 24 in
2000 to 801 in 2010.

**Population base.** Permanent population. The migration volume (table 6,
`census2020_migration_table6.pdf`) counts 470,973 temporarily absent, 352,681 of them outside the
republic, and 246,133 of all the absent gone to work. They are residents and are drawn at home; a
de facto base would take 3.7% off, mostly men of working age, evenly enough that it moves no
region's religion.

**Geography.** COD-AB `cod-ab-tjk` has no polygons (the workbook and the p-code PDF only).
geoBoundaries gbOpen TJK ADM1 (OSM via Wambacher, 2017, ODbL) has the five units, Sughd with its two
exclaves in Kyrgyzstan. **Its Dushanbe is 366 km2 against the city's own 203.18 km2**
(`dushanbe.tj` city passport, 1 July 2019); OSM's current relation 7328360 is 197 km2. The old
polygon's extra 180 km2 is suburb the census counts in the Districts (Kontur holds 260,000 there).
Dushanbe is redrawn on today's OSM line (`polygons.openstreetmap.fr`; Overpass timed out on two
mirrors) and the rest goes to the Districts. Kontur TJ 2023 (37,960 hexes, 10.28M): ratio to the
census 1.059; per region over the national ratio, Khatlon 0.84, Districts 1.04, Sughd 1.00,
Gorno-Badakhshan 1.18, Dushanbe 1.46 (pinned: Kontur's capital is high, 1.72 on the old polygon).
348 hexes in Afghanistan, Uzbekistan, Kyrgyzstan or China dropped (60,748 people), 361 snapped
within 2 km (74,561). `kontur_cap.py tj`: no blocks at the cap. Five units are too few for a rank
witness; the band is the check.

## 8. GORNO-BADAKHSHAN'S ISMAILIS (ask 051)

Not drawn: every Muslim is on `islam`. Drawing the Pamiri districts (Khorog, Shughnon, Roshtqal'a,
Ishkoshim, Rushon, 154,623 people in 2020; Vanj, Darvoz and Murghob 73,293, mostly Sunni) as
Ismaili would be ascription by district: the census has one `ислам` box and no Pamiri nationality,
and the Central Asia Barometer's one reading (§4) is not credible. `taxonomy/cn2000.py` keeps
Taxkorgan's Ismaili Tajiks on plain `islam` for the same want of a measure. §14 because of the
state's pressure on the region since the May 2022 Khorog protests; the ask recommends leaving it.

## 9. WHAT REOPENS IT

- The 2020 Volume III (nationality), even national only: it replaces the 0.3% scaling with counts.
  Check `stat.tj/ru/perepis-naseleniya-i-zhilishhnogo-fonda-2020/` for the `<details>` list.
- Any 2020 religion table (§6, first bullet): it replaces the whole model.
- `nada.stat.tj` (§2), if it ever answers.

## 10. Review, 2026-10-03 (`fafd1067-rev10`, full pass)

Checks clean (`check_md`, `built_countries --check`, `check_rollup tj`: all 9,657,005 modelled, none
orphaned). Every note figure re-summed from `data/normalized/tj.csv` and matches: 30,078 Christians
(29,641 Orthodox, 361 Armenian, 76 Georgian), 2,464 unknown, 2,000 secular, Dushanbe 1.94% not
Muslim, 99.64% Muslim. No new node; mappings follow `az2019`/`kz2021`. Screenshot: dots in all five
regions, Gorno-Badakhshan sparse, along the Panj and around Khorog, nothing over the border.

One wording fix in `note_public`: *"about **28,971** people"* is 0.3% of 9,657,005, a share the
Agency printed to one decimal (§7: its rounding alone spans 24,000 to 34,000), so five digits read as
a count. Now *"about **29,000**"*; `refresh-meta` run.
`queue.csv`'s row still carried the scout's plan (Pew 2020, one national mix); set to what was built.
Ask 051 not re-opened.

