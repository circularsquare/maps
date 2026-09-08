# South Africa — Stats SA, Community Survey 2016 provincial profiles, tables 2.10a and 2.10b

**Drawn 2026-09-08.** Nine provinces, 25 source categories, 54,946,360 people, from the nine
CS 2016 provincial profile reports. `sources/za.py`, `sources/za_geo.py`,
`sources/za_grid.py`, `taxonomy/za2016.py`, and the `za` entry in `countries.py`.

The headline: **`christianity.africaninstituted` gains 14,158,453 people**, taking the node to
25,899,969 across seven countries. That is **1.21x everything it held before** (11,741,516:
Zimbabwe 6,112,503, Kenya 3,292,573, Angola 1,099,234, Benin 676,032, Eswatini 420,690, Côte
d'Ivoire 140,484) and **2.32x Zimbabwe's cell**, which was the largest any one country had
contributed; South Africa alone is now 54.7% of the node. An earlier draft of this line said
"more than doubling" and that was wrong. South Africa is where the African Independent Church
movement began, and this is the first source on the map that counts it at its own scale.

Every figure in this file reproduces from `data/normalized/za.csv`. Percentages described as
*of Christians* are of that province's fourteen denominations plus its not-reported residual;
all others are of that province's drawn answers. **That is NOT the denominator Stats SA
prints** — the reports compute against table 2.10b's own total, which excludes the residual —
so every figure here sits a few tenths of a percent relatively below the printed one
(Limpopo's African Independent row is 50,9 in Report 03-01-15 and 50.77 here). The wider base
is deliberate: it is the same denominator in all nine provinces, and North West's printed
total is defective (§4.1), so computing on it would make that province incomparable.

---

## 1. Why the survey and not the census

`queue.md` had `za` closed on *"Christianity undivided; behind a DataFirst account"*, and
sources.md §11ag had already corrected half of that: **Census 2022 is not walled.**
Statistical release P0301.4 section 2.9 table 2.10 gives religion for all nine provinces
over eleven categories, openly downloadable. What it does not give is any split of
`Christianity`, which is 83.6% of the country in one cell.

**Community Survey 2016 is a different release and it has both tables.** Each of the nine
provincial profiles carries:

- **table 2.10a**, *Distribution of population by religious affiliation* — eleven
  categories: Christianity, Islam, Traditional African religion, Hinduism, Buddhism,
  Bahaism, Judaism, Atheism, Agnosticism, No religious affiliation/belief, Other.
- **table 2.10b**, *Distribution of population by Christian denomination* — fourteen:
  Catholic, Anglican/Episcopalian, Baptist, Lutheran, Methodist, Presbyterian,
  Pentecostal/Evangelistic, African Independent Church/African Initiated Church, Jehovah's
  Witness, Seventh Day Adventist, Mormon, Reformed church, Just a Christian/non-
  denominational, Other. Several carry an exemplar list, and the African Independent row's
  names *Zion Christian Church; Apostolic Church; African Nazareth Baptist Church/Shembe*.

So 24 published categories against the census's 12, at the same nine units. The cost is six
years and a survey rather than a census. §3.1 forbids mixing them and §3.9 says category
detail and spatial detail trade off inside one source; this map has taken categories over
vintage every time it has been asked — Benin, Trinidad, Paraguay, Türkiye.

**§3.4's Brazil construction was considered and rejected**, and it is worth saying why,
because it looks made for this case: take the 2022 provincial totals, keep the eleven
categories, and split only the Christian cell by CS 2016's within-province denominational
mix. The reason not to is §3, immediately below.

---

## 2. Nine provinces is the open ceiling, and that was established rather than assumed

Six million people per unit is the coarsest counting geography on this map, four times
Zimbabwe's 1.5M, so the finer tier was worth real effort. Everything below was opened and
read, not inferred from a search result.

| what | verdict |
|---|---|
| **Report 03-01-84, *Cultural dynamics in South Africa*** — named in §11ag as the likely home of a finer cut | **Province-only and COARSER.** Religion is chapter 4; table 4.1 is *by religious denomination and province*, censuses 1996/2001/2022, over **8** categories (Christianity / Islam / Traditional African Religion / Hinduism / Jewish Faith-Hebrew / Other beliefs / No religious affiliation / Undetermined). Its §4.3.1 is headed *"Regional trends and patterns"* and "regional" means the nine provinces. Its own chapter 2 goes finer for language (table 2.3 is by metropolitan area), so the report had the geography and did not use it for religion. |
| **Census 2011** | **Asked no religion question at all.** Report 03-01-84 p.48 says so in terms: *"The Census 2011 questionnaire did not include a question on religious belief/affiliation"*. That closes Wazimap and every Census 2011 municipal product in one line, and it is why the UNSD oracle jumps 2001 → 2022. |
| **Census 2022 provincial profiles** (Report 03-01-74 KZN, 03-01-75 North West, and the rest) | Religion is one table, **province total only**, 11 categories, Christianity undivided. |
| **Census 2022 Municipal Fact Sheet** | 41 pages, zero matches for religion, Christian, Islam or Hindu. |
| **Census 2001 *Census in Brief*** | 111 pages, text extracts cleanly, zero matches for religion, Christian, Zion, Muslim or Hindu. |
| **Stats SA's Census 2022 dissemination API** — `disseminationapi-a2f6fff8f7a3f3ff.z01.azurefd.net`, keyless, no login | **Live and useful, and religion is not on it.** `/api/GeoLevels/getAll/` returns National, Province, District, Municipality and **Main Place**, and it serves 24 topics (population group, age, language, marital status, education, income, employment, water, toilets, energy, dwelling, internet and more) down to main place. Grepping the whole 19 MB front-end bundle for `religio` gives **0 matches**. Language goes to main place; religion is not served at any level. |
| **HDX**, **USCB country geodatabases** (`[[reference_uscb_country_gdb]]`) | No South African religion dataset. USCB's CKAN listing has no South Africa row at all. |
| **Nesstar** | Discontinued by Stats SA. |
| **SuperWEB2**, `superweb.statssa.gov.za` | Stats SA's own cross-tab tool, geography reportedly to ward. **Unverified**: the host is behind the Imperva wall for both curl and WebFetch, and it requires a registered account with no documented guest route. A search snippet claimed religion sits on its *Census 2022 Special District Layer*; that could not be confirmed and is **not** recorded as fact. |

**The pattern is the finding.** In the Census 2022 provincial profiles essentially every
other variable is tabulated *"by district and local municipality"* — population, density,
sex ratio, age, population group, marital status, birthplace, education, dwelling, tenure,
water. Religion is the single variable published province-only. That reads as a decision
rather than an oversight, and it means no amount of further searching in the published
reports will turn up a finer open table.

### 2.1 The finer data exists and it needs an account

**CS 2016 microdata, DataFirst catalogue 611.** Person file 3,328,867 records × 99
variables. `ReligionBelief` carries the same 13 codes as table 2.10a (11 plus *Do not know*
and *Unspecified*), and a separate `Christianity` variable carries **15** denominations —
the fourteen above plus *Do not know*. Geography is in the same file: `DC_MDB_C_2016`
(district/metro), `MN_CODE_2016` (local municipality), `PR_CODE_2016`. CS 2016 was
*designed* to be representative at local-municipality level, so this is not over-reach: it
would take South Africa from 9 units to roughly 234 with the **same 24 categories**. The
metadata is browsable without a login; the files are not. It needs a free registration plus
a signed confidentiality declaration, so it needs Anita's identity. **That is the one ask
filed for this country.**

Census 2022's 10% sample (DataFirst 982 / World Bank 8218) is the fallback if currency ever
matters more than denominations: `P09_RELIGIOUS_AFFILIATION`, Christianity undivided,
`Municipality` with 214 categories, EA withheld, and the same account wall. IPUMS
International has RELIGION for South Africa 1996, 2001 and 2016 and is separately dead for
this project (`[[reference_ipums_account]]`).

**One route deliberately not taken.** Stats SA accepts custom tabulation requests and
religion by district plainly exists in their database. That is an email rather than a
download, and §3 of `AGENT_BRIEF.md` puts anything needing her identity on Anita's side.

---

## 3. The two Stats SA releases disagree past what six years explains

This is the most important thing in this file for anyone who later wants to update the
country, and it is why the §3.4 rescale was rejected rather than merely not attempted.

The category *labels* match verbatim between CS 2016 table 2.10a and Census 2022 table 2.10,
which makes them look interchangeable. The distributions do not:

| category | CS 2016, % of answers | Census 2022, % of answers |
|---|---|---|
| Christianity | 79.0 | 85.3 |
| **No religious affiliation/belief** | **10.9** | **2.9** |
| **Traditional African religion** | **4.5** | **7.8** |
| Other | 2.7 | 1.0 |
| Islam | 1.6 | 1.6 |
| Hinduism | 1.0 | 1.1 |

Islam and Hinduism, the two categories a respondent is least likely to be unsure about, are
stable to a tenth of a point. The three that move are the ones whose boundary is a matter of
how the question is put: no religion falls by a factor of nearly four, traditional religion
rises by three quarters, and the residual halves. **That is an instrument difference, not a
trend**, and it is §3.1a's finding — two sources can share a basis and still be
incomparable because their answer sets behave differently.

So a Brazil-style rescale would have been putting 2016 shares underneath 2022 magnitudes
whose *own* top-level split disagrees with the 2016 one by eight points on Christianity.
Everything is on one basis instead: CS 2016, drawn at its own year, with the year on every
figure. Census 2022 figures appear in this project only as the contrast above, in
`taxonomy/za2016.py`'s REVIEW entries and in one sentence of `note_public`.

---

## 4. What the parse had to survive

Nine reports, produced by nine provincial teams, and no two are laid out quite the same.
Every item below is a **silent** failure — a wrong number, no exception.

1. **The table number is not stable.** Free State, KwaZulu-Natal, Mpumalanga, North West and
   Northern Cape use 2.10a/2.10b; Eastern Cape and Gauteng use 2.9a/2.9b; Limpopo uses
   2.7/2.8 with no letter suffix at all; Western Cape uses 2.11a/2.11b. Gauteng writes its
   own caption as `Table 2.9 a:`, with a space inside the number, which defeats a regex on
   `2\.9a`. **Anchor on the caption text, never the number.**
2. **The total row is usually labelled with the province name.** Eastern Cape,
   KwaZulu-Natal, Limpopo, Mpumalanga and North West end their tables with a row reading
   `Eastern Cape 6 928 088 100,0` rather than `Total`. Parsed as data it **doubles the
   table** and halves every share.
3. **Northern Cape prints Bahaism as a bare dash** in both columns. Skipped, that province
   silently has ten categories where the rest have eleven; read as a zero, the answer set
   stays a partition. It is emitted with count 0 and `scatter.py` correctly draws nothing.
4. **Two header cells glue onto the first row's label.** Western Cape's is
   `Christian domination` (the report's own typo) and Northern Cape's denomination table is
   headed with the bare word `Christianity` — which is *also* a real row label in the other
   table. The header set therefore has to be **per table**: one shared set either loses
   Northern Cape's Catholic row or deletes every province's Christianity row.
5. **Spelling variants.** Gauteng writes `Buddism` (which is the CS 2016 codebook's own
   spelling, not one report's slip) and KwaZulu-Natal and Limpopo write `Seventh-Day
   Adventist`. Folded, and the fold is asserted to leave exactly 11 and 14 categories.
6. **Both tables have a row called `Other`** and they mean different things. Every category
   is emitted prefixed `Religion: ` / `Christian: `, which is `sources/sz.py`'s convention.
   A bare join on the label would move 1,482,210 people of other faiths into Christianity.
   `_za_counts` asserts the prefixes are still there.
7. **The report numbers are not guessable and are read from the PDFs.** They do not run in
   province-code order: Western Cape is **03-01-07** and Mpumalanga is **03-01-13**, which
   is the number a code-order guess hands to Western Cape. `read_province` checks each
   against the running header of the file it opened. Three of nine were guessed wrong on
   the first pass here, so this check is not theoretical.

### 4.1 North West's table 2.10b does not add up

**Report 03-01-11's fourteen denomination rows sum to 3,072,039 against its own printed
total of 3,408,521**, and its printed percentages sum to 90.1 rather than 100.0. So 336,482
people, 9.9% of that province's Christians, are in no denomination row.

The printed total is not the problem: 3,408,521 is exactly table 2.10a's Christianity cell
(3,430,406) less that table's own footnote exclusions (`Do not know` 21,873 + `Unspecified`
12). It is a row that is wrong.

**Which row, almost certainly.** The `Other` cell reads **21 873** — character for character
the `Do not know` figure in the same table's own footnote. And North West's `Other` comes
out at 0.64% of its Christians where the other eight provinces run 4.43% (KwaZulu-Natal) to
17.25% (Northern Cape). Restoring the missing 336,482 to `Other` would put it at 10.5%,
squarely inside that range.

**That is a good story and it is not acted on.** The 336,482 go into the same
`Christian: Denomination not reported` row every province already carries, where a reader
can see them, rather than being assigned to a denomination on the strength of a coincidence.
North West therefore reads 10.45% not-reported against 0.20–0.91% elsewhere, which is
visible in the map and explained in `taxonomy/za2016.py`.

`sources/za.py` **asserts the defect is still present**. If Stats SA reissues the report
with the row fixed, the build fails and says so, rather than quietly changing 336,482
people's category.

### 4.2 The `Denomination not reported` row is the reconciliation, not an invention

Table 2.10b excludes its own `Do not know` and `Unspecified` and prints both figures. The
gap between table 2.10a's Christianity cell and table 2.10b's total reproduces those two
numbers **exactly in four provinces** (Northern Cape 7,989 against 7,759 + 230, Free State
8,998 against 8,936 + 62, KwaZulu-Natal 18,874, Limpopo 8,346), **to within one person in
three more** (Eastern Cape +1, Gauteng −1, Mpumalanga −1), **unverifiably in one** because
Report 03-01-07 prints no exclusion note under Western Cape's table at all, and **not at all
in North West**, which is §4.1. So Western Cape's 46,055 is inferred from table 2.11a rather
than confirmed against a footnote, and it is the largest residual of the eight sound
provinces. Dropping the gap would silently shrink
Christianity in every province. It resolves to bare `christianity`, which is at2001.py,
au2021.py, bd2011.py, bg2021.py and bs2022.py's call for the same shape.

Nationally it is 567,039 people, 1.31% of Christians, and North West is most of it.

---

## 5. Access — the wall is real and it is not where §11ag put it

`statssa.gov.za` and its subdomains sit behind Imperva. **HTML pages are unreachable**:
`?page_id=` and `?p=` listings return a ~1 KB `_Incapsula_Resource` stub for curl *and* for
WebFetch, and a browser User-Agent does not help. So the publications listing cannot be
read from here, and finding a document means finding its PDF URL some other way.

**But the PDFs themselves are not walled, and §11ag's account of why curl fails is wrong.**
What stops a plain `curl` on `cs2016.statssa.gov.za` is the **TLS chain**, not the bot wall:
the host presents a self-signed intermediate and curl exits 60 before it sends the request,
which reads exactly like a dead host. Relax the certificate check and the same URL returns
the real PDF at full size. `sources/za.py --fetch` does exactly that and then checks the
`%PDF` magic and the `%%EOF` trailer, because a 200 is not a download
(`[[reference_pdf_truncated_at_source]]`).

WebFetch also retrieves these PDFs, but its summariser cannot read them (compressed
streams) — it saves the binary and returns nothing useful, so extract locally with PyMuPDF.

The nine profiles are static WordPress uploads at
`https://cs2016.statssa.gov.za/wp-content/uploads/2018/07/<Province>.pdf`, where `<Province>`
is `WesternCape`, `EasternCape`, `NorthernCape`, `FreeState`, **`KZN`** (not
`KwaZulu-Natal`, which 404s), `NorthWest`, `Gauteng`, `Mpumalanga`, `Limpopo`.

---

## 6. Boundaries and placement

OCHA COD-AB South Africa from HDX, the shapefile bundle rather than the geodatabase on
§12's Chile rule, read with `engine="fiona"`.

**COD-AB misspells one of the nine.** `ADM1_EN` reads **`Nothern Cape`**, no `r`. An
explicit alias handles it rather than a fuzzy matcher, because a matcher loose enough to fix
this is loose enough to pair two genuinely different provinces and never say so
(`[[reference_name_join_wrong_neighbour]]`).

**The join's independent check is AREA, not the name and not the p-code.** Zimbabwe's check
was print order against p-code order; that is unavailable here, because Stats SA's province
codes 1–9 and COD's `ZA1`..`ZA9` come from the same statute and agreeing proves nothing.
Areas are genuinely independent and South Africa's provinces are unusually well separated on
them — Northern Cape 372,889 km² to Gauteng 18,178 km², a factor of twenty. The computed
polygon areas reproduce the published figures **to within 0.05% on all nine** and reproduce
the published rank order exactly.

Free State and Western Cape are 129,825 and 129,462 km², 0.3% apart, so the rank test cannot
separate them; they are checked on longitude instead (Western Cape's polygon at 22.2°E,
Free State's at 26.6°E). The Free State figure is not remembered — it is quoted from that
province's own profile, Report 03-01-12 p.10.

Why it matters that this check exists at all: **every province's figures reconcile against
its own report whichever polygon it is paired with**, so a transposition would move eleven
million people and no totals test anywhere in the pipeline would notice.

**Placement is Kontur's 400 m population grid** (`kontur_population_ZA_20231101`), and this
country needs it more than most. Nine polygons over 1.22M km²: Northern Cape is 30.5% of the
land and 2.2% of the people, Gauteng 1.5% of the land and 23.9% of the people, a density
ratio near 470 to 1. With nine units nothing averages out. 1,760 hexes (124,999 people,
0.21%) fall outside every province — the border overrun plus, mostly, **the Lesotho enclave
and the Eswatini salient**, which South Africa surrounds but does not contain. Dropped and
reported. Kontur holds 60,364,560 people inside the nine against 54,946,360 drawn answers, a
ratio of 1.099, which is the expected direction: a November 2023 model against a 2016 survey
whose non-answers are not drawn.

---

## 7. What the map shows

79.0% of answers are Christian, and the fourteen-way split is the point.

| | national | of Christians | low | high |
|---|---|---|---|---|
| African Independent Church | 14,158,453 | 32.6% | Western Cape 15.8% | **Limpopo 50.8%** |
| Pentecostal/Evangelistic | 8,483,677 | 19.5% | KwaZulu-Natal 16.0% | Limpopo 25.7% |
| Catholic | 3,778,332 | 8.7% | Limpopo 2.9% | Northern Cape 13.6% |
| Other Christian | 3,509,156 | 8.1% | *North West 0.6%, see §4.1* | Northern Cape 17.3% |
| Methodist | 2,777,937 | 6.4% | Limpopo 1.0% | Eastern Cape 15.2% |
| Just a Christian | 2,501,384 | 5.8% | Eastern Cape 2.8% | Western Cape 8.6% |
| Reformed church | 2,350,853 | 5.4% | **KwaZulu-Natal 1.4%** | Western Cape 12.3% |
| Anglican/Episcopalian | 1,765,287 | 4.1% | Mpumalanga 1.0% | Western Cape 8.3% |

**The African Independent Churches are the country.** A quarter of everyone who answered,
a third of Christians, and half of Limpopo's Christians — where the Zion Christian Church
has its headquarters at Moria, which draws the largest Easter gathering in Africa.
KwaZulu-Natal is 43.0%, which is Shembe's Nazaretha country, and Mpumalanga 39.4%. Against
that, Western Cape is 15.8%: the AIC line is the old-homeland/settler line as much as
anything else.

**The mission denominations are a nineteenth-century map.** The Reformed churches (the NG
Kerk family) run 12.3% of Western Cape's Christians and 11.3% of Free State's against 1.4%
in KwaZulu-Natal, a factor of nine. Methodists are 15.2% of Eastern Cape's Christians and
1.0% of Limpopo's, which is the sharpest gradient of any named denomination here on both
measures (15.3x and 14.2 points, against the Reformed row's 9.0x and 10.9); the Scottish
Presbyterian mission at Lovedale shows up in that same Eastern Cape at 3.5% against 0.4% in
Mpumalanga.

**Pentecostal/Evangelistic is the flattest large category**, 16.0% to 25.7% of Christians,
where everything else moves by three to ten times. A national religion rather than a
regional one.

**The non-Christian tail is two cities and two coasts.** Hinduism is 4.0% of KwaZulu-Natal's
answers and 0.02%–0.8% everywhere else, which is the Durban population descended from the
indentured labourers brought to the Natal sugar estates from 1860. Durban is often called the
largest Indian city outside India; South Africa is **not** the largest Indian-descended
population outside Asia, which an earlier draft of this file claimed, because the United
States and the United Kingdom are both larger. Islam is 5.6% of Western Cape against 0.3% in
Free State and
Limpopo, and the two ends are different communities the survey cannot separate: Shafi'i Cape
Malay in the west, Hanafi South Asian in Gauteng and KwaZulu-Natal. Judaism is 0.22% of
Western Cape and 0.21% of Gauteng and essentially absent from the other seven.

**No religious affiliation is 10.9%, and its geography is not the one a reader expects.**
Limpopo is the highest in the country at 17.25% and Northern Cape the lowest at 1.67%, with
Gauteng 13.8%. Limpopo is also the most African-Independent province, so this is not a
secularisation gradient. A plausible reading is that where church membership is the dominant
idiom, someone outside a particular church answers `none` rather than naming an ancestral
practice — Benin's `Aucune` warning in another country. Nothing here resolves it.

**African traditional religion is 4.5% and is a floor**, for the reason sources.md §11b
gives for the continent: the box is exclusive of the Christian ones, and consulting a
*sangoma* or honouring the *amadlozi* commonly accompanies church membership rather than
replacing it. South Africa is a sharper case than most, because the African Independent
Churches counted separately at six times the size grew out of exactly that overlap.

### 7.1 One new taxonomy node

`other.za`, *Other religion (South Africa)* — Stats SA's `Other` in table 2.10a, 1,482,210
people, 2.70%. A genuine tail rather than a store cupboard: Islam, Hinduism, Judaism,
Buddhism, Bahaism and Traditional African religion all have their own boxes. **Unusually
flat** at 3.87% (Gauteng) to 1.18% (North West), where this map's residuals commonly run ten
or twenty to one and point straight at what is inside them (§9r's Chittagong rule), so
nothing is inferred from its geography.

Nothing else was added. All 24 target nodes already existed.

---

## 8. Ethics (§14)

No §14 question was raised and none is filed. Religion is asked on South Africa's own census
and community survey and published by Stats SA itself, so §14.4 rule 2's ceiling is the
state's own publication and this build sits at exactly that ceiling — nine provinces, which
is coarser than the state holds and not finer. No group here is drawn at a resolution its
own government has not already published, and no category on the card identifies a
persecuted minority at a locatable grain.

---

## 9. Not done

- **The district / local-municipality build.** CS 2016 microdata, DataFirst catalogue 611,
  same 24 categories at ~234 local municipalities. Account-walled; the ask is filed. This
  would be the single largest improvement available to any country currently on the map.
- **The 1996 and 2001 censuses split the African Independent Churches and nothing here uses
  it.** UNSD's oracle carries 2001 at **26** categories including *Zion Christian Church*
  4,971,932, *Other Apostolic Churches* 5,609,070, *Other Zionist Churches* 1,887,147,
  *Ethiopian Churches* 880,414, *Ibandlalama Nazaretha* 248,824 and *Other African
  Independent Churches* 656,644; and **1996 at 64 categories**, which is the richest religion
  card any African census has ever run. Both are national-only in the oracle, and §2 above
  found no open subnational table for either. If one is ever found, a §3.4 rescale splitting
  the 2016 AIC cell by 2001 shares would be the most interesting thing this country could
  become — and note that the answer sets are close enough to make it tractable, unlike the
  2016/2022 pair in §3.
- **SuperWEB2** could not be reached and its Census 2022 religion content is unverified
  either way. It is the one open question in §2's table.
- **Whether CS 2016's `no religion` or Census 2022's is closer to the truth.** Neither is
  checkable from here, and the gap is a factor of four. Anyone updating this country to 2022
  is changing that number by 4.2 million people and should say so in the note.

---

## 10. Review, 2026-09-08

A second pass, not the builder's. `check_md.py`, `built_countries.py --check`,
`check_rollup.py za` (54,946,360, all measured, nothing derived, nothing orphaned) and
`check_mapping.py za` are all clean, and the screenshot is clean: dots inside the border,
Gauteng dense and Northern Cape sparse, Lesotho and the Eswatini salient blank as §6 says
they should be, nothing in the sea.

**The node arithmetic in §7 and in `taxonomy/za2016.py` reproduces exactly**, recomputed from
`data/normalized/*.csv` through each country's own `resolve()` rather than from this file.
`christianity.africaninstituted` and its children: za 14,158,453, zw 6,112,503, ke 3,292,573,
ao 1,099,234, bj 676,032, sz 420,690, ci 140,484. The six-country prior is 11,741,516, the
node is 25,899,969, za is 54.67% of it, 1.206x the rest and 2.316x Zimbabwe. All four printed
figures are right. *A trap for whoever recomputes this next: `ke` carries `country` and
`county` rows and `bj` carries three tiers, so a naive sum over the normalised file
double-counts Kenya and triples Benin. Group by `geo_level` first.*

**The fourteen denominations agree with every precedent that shares a body.** Against
`sz2017.py`, which is the closest card and landed the same day: Catholic, Anglican, Lutheran,
Methodist (parent, not `.african`), Jehovah's Witness, Pentecostal, Seventh Day Adventist
(parent, not `.sda`) and `Other` all land on the same nodes, and Zionists/Apostles and
African Independent Church both land on `christianity.africaninstituted`. `zw2022.py` and
`ke2019.py` agree on Catholic, Pentecostal, Other Christian, the AIC cell and every
non-Christian row. Zimbabwe's `Protestant` has no South African counterpart because Stats SA
names the bodies individually; that is §2.4 working, not a divergence. `Mormon` ->
`christianity.latterday` matches bb, bs, gd, ee, fi, lc and pe; `Reformed church` ->
`christianity.reformed.continental` matches au2021 and ca2021 and the node's own definition.
**The AIC call is not close and the exemplar list settles it**: the printed row reads *Zion
Christian Church; Apostolic Church; African Nazareth Baptist Church/Shembe*, and the
Pentecostal row separately prints *Apostolic Faith Mission*, so the indigenous Apostolic
bodies and the Pentecostal mission body each have their own box. That is exactly the
reasoning `sz2017.py` gives for reading Eswatini's `Apostles` as the indigenous family.

**One divergence, and it is Eswatini's rather than South Africa's.** `sz2017.py` sends
`Christian: Not Stated` to `christianity.other`; this country sends `Denomination not
reported` to bare `christianity`, with at2001, au2021, bd2011, bg2021, bs2022 and ie2022
behind it. Bare `christianity` is the better call and Eswatini's cell is thirteen people, so
nothing is worth changing; recorded so the next reader does not read the pair as a rule.

**§4.1's assertion does fire on a reissue.** `sources/za.py` raises when North West's fourteen
rows come within two people of the printed total, which is the reissue case, and the residual
row carries the shortfall in its own `note` column so the province does not read as a finding
on the map. The treatment matches how residuals are handled elsewhere. Two limits worth
writing down: it only runs under `--fetch`, so a stale CSV is never re-checked, and a reissue
that changed the figures without closing the sum would not trip it. Both are acceptable.

### 10.1 The one thing to change: there is a published gap and the entry does not carry it

**707,294 people, 1.27% of CS 2016, answered no religion question at all and are not drawn.
`countries.py` has no `gap` and no `gap_share` for `za`.**

Table 2.10a is a table of *answers*, not of people: its printed total excludes `Do not know`
and `Unspecified`, and **eight of the nine reports print both figures under the table**. Free
State's footnote reads *Excludes 'Unspecified' (133) and 'Do not know' (25 846)*; Gauteng's
reads *Total excludes 269397 Do not know and 927 Unspecified*. Only Report 03-01-07 (Western
Cape) prints no note, which is the same silence §4.2 already records for its table 2.11b.

Two independent routes agree to one person, which is the rounding in the weighted totals:

- the eight printed footnotes sum to **640,083**, and Western Cape by difference against
  table 2.1's 6,279,730 is **67,212**, giving **707,295**;
- table 2.1's national CS 2016 population of **55,653,654** less the nine table 2.10a totals
  (54,946,358) is **707,296**.

So `gap_share=0.0127` is a published figure and not an estimate, and `gap` wants to say
something like *"the 1.3% who answered no religion question"*. Note that `basis` currently
reads *"self-identification, whole survey population"*, which is true of the question (it was
asked at every age, unlike Peru's or Botswana's) but reads, with no `not drawn` row beside
it, as though the dots are everybody.

**And §3.5 asks which way the hole leans, which nothing here has asked.** It leans, and at
nine units it is still significant. Non-response by province runs from Northern Cape 0.417%
to Gauteng 2.017%, and against the drawn shares:

| | r | exact permutation p, n=9 |
|---|---|---|
| vs `No religious affiliation/belief` share | **+0.72** | 0.028 |
| vs Christianity share | −0.63 | 0.068 |

That is Serbia's direction (§3.5's own worked example): the people dropped are
disproportionately from the least religious provinces, so **every share drawn for South
Africa is slightly more religious than South Africa is**, and the 79.0% Christian and 10.9%
no-religion figures in `note_public` are both biased in the same direction by it. Small
against the 4.2-point instrument gap in §3, but it is the one correction §3.5 asks for by
name and it is free.

*Not applied here.* `countries.py` moved 71 lines under this session while it was being read,
so another agent is in it; and the wording of a `gap` and of a §3.5 lean sentence is the
builder's or Anita's, not a reviewer's. It is a two-line change plus `check_md.py` and
`tiles.py --refresh-meta`; no re-scatter and no build tail.

### 10.2 The `africaninstituted` node note is now the stalest thing about this country

`taxonomy/branches.py`'s note for `christianity.africaninstituted` is a legend row every
reader sees, and it still says **"It exists because Kenya counts it and almost nobody else
does"**, **"AND IT IS NOW MOSTLY ZIMBABWEAN"**, and **"three countries supply it, each with
one undivided cell"**. Seven countries supply it; Zimbabwe is 23.6% of it and South Africa
54.7%; and four of the seven cells are named churches, three of them Angolan children of this
node plus Côte d'Ivoire's Harrist.

**This is not South Africa's fault alone** — the note was last revised when Zimbabwe landed,
and ci, ao and sz have each been added to the node since without touching it. But South
Africa is the addition that makes it wrong rather than merely incomplete, and it is the one
place on this map where a reader is told the opposite of what the dots show. Flagged rather
than rewritten: it is a shared file with live builders in it, the rewrite needs
`taxonomy/build_tree.py` after it, and the sentence about what the node *is* now is worth
Anita writing.
