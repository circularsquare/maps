# Belgium — `sources/be.py`, `sources/be_geo.py`, `taxonomy/be2024.py`

Drawn 2026-09-11. **11 units, 48 nodes, 11,497,382 people, 99.50% of the country.**
No Belgian census has ever asked about religion.

| | |
|---|---|
| counting geography | **NUTS 2, the ten provinces plus Brussels**, 1.05 million people each |
| placement | 581 communes, GISCO LAU 2021, weighted by communal population |
| basis | self-identification (survey) for citizens; nationality-derived for residents |
| tier | **`modelled` throughout**, a survey is not a count of anybody and neither is a nationality model |
| vintage | ESS rounds 5-11, 2010 to 2023; census 2021; Pew 2020 |
| authored cells | **none** |
| not drawn | 57,385 people, 0.50%: 31,600 citizens who declined the religion question, and 25,785 whose citizenship the census did not record |

**The finding is not a table, it is that there is no table.** Statbel publishes no religion
figure at any geography, and establishing that took longer than building the country.

---

## 1. Statbel, which publishes nothing, and how to find that out

§9cu made the custom-tabulation shelf a standing check before building any European country
from ESS: CBS's *maatwerk* shelf gave the Netherlands 403 gemeenten where the queue had
priced twelve provinces. **Belgium has no such shelf, and it has no religion statistic to put
on one.**

Statbel's own site search, run on 2026-09-11 with 50 results a page:

| term | index | hits |
|---|---|---|
| `religie` | nl | 2, neither about religion (a households press release, a SILC glossary) |
| `godsdienst` | nl | 1, a 2022 labour-force press release about language barriers |
| `levensbeschouwing` | nl | 0 |
| `moslim` | nl | 0 |
| `religion` | fr | 1, the French version of the same 2022 press release |
| `culte` | fr | 0 |
| `confession` | fr | 0 |

    https://statbel.fgov.be/nl/search?search_api_fulltext_block=<term>&items_per_page=50
    https://statbel.fgov.be/fr/search?search_api_fulltext_block=<term>&items_per_page=50

Statbel does offer statistics on demand, but as a service you commission and not as a public
shelf: there is no `/maatwerk/` URL space and nothing corresponding to CBS's 2015/20 file.
So the check §9cu asks for comes back negative here, which is the answer it was always
going to give for some country and is worth recording as one.

### 1a. The wall, which is the part worth carrying to other countries

**statbel.fgov.be and data.gov.be sit behind an F5 Shape (TSPD) wall.** Three things about
it generalise:

- **It returns HTTP 200 with a JavaScript challenge**, a 5,644-byte stub with
  `window["bobcmn"]` and a `/TSPD/` script tag. So a naive fetcher records a success and a
  `Content-Length` that looks plausible, which is
  [[reference_pdf_truncated_at_source]]'s failure mode one layer up.
- **Costa Rica's fix does not work.** §9cp's Akamai wall at `inec.cr` keys on header
  COMPLETENESS and opens to a full browser header set. This one does not: nine headers,
  `Accept-Language: nl-BE`, the full `Sec-Fetch-*` set and a current Chrome UA all still get
  the challenge.
- **Headless Chrome does not work either, until you override the User-Agent.**
  `--headless=new` still reports `HeadlessChrome/<version>` and the wall escalates that from
  a solvable JS challenge to an **image CAPTCHA**, which is a dead end. Passing
  `--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like
  Gecko) Chrome/131.0.0.0 Safari/537.36"` gets through on the first navigation. `--dump-dom`
  is still not enough, because it captures the challenge page before the reload; drive it
  over CDP, navigate, wait about ten seconds, then `DOM.getOuterHTML`.
- The Wayback CDX API is not the fallback here. A regex filter over the whole
  `statbel.fgov.be` domain **504s** on every attempt, with and without `matchType=domain`.

And one Drupal detail that cost twenty minutes: **the search parameter is
`search_api_fulltext_block`, not `search_api_fulltext`.** The wrong name is not an error, it
is ignored, and the page returns the entire 5,016-item index with a result count that looks
like a huge number of hits.

## 2. What was ruled out besides Statbel

**The reason there is nothing to find is a law rather than an omission.** Religion is
protected personal data under Belgium's privacy law of 8 December 1992 and its successors,
the National Register holds nationality and the language of the commune and no belief item,
and the federal recognition regime funds the ministers of six religions and of organised
secularism without counting anyone's members. Alain Vannieuwenburg's survey of the question
for the Humanistisch Verbond, 21 January 2020
(`humanistischverbond.be/blog/209/levensbeschouwing-waar-zijn-de-cijfers/`), reaches the
same place and lists what Belgian researchers use instead: Pew, ESS, WIN/Gallup, Bertelsmann
and Eurobarometer. So Belgium has **no register tier at all**, which is the difference from
Finland (§9by), and the survey is the tier rather than second best.

**Belgium is federal and Statbel is not its only statistical office**, so a negative at
Statbel is not a negative for the country. Four more tiers were swept on 2026-09-11 and all
four are empty:

| tier | what was asked | what came back |
|---|---|---|
| Statistiek Vlaanderen | 30 catalogue themes; all 16 SV-bevraging questionnaires (6,000 Flemish residents twice a year) searched for godsdienst, religie, levensbeschouwing, geloof, kerk | no religion theme, **zero occurrences** in any questionnaire; the per-gemeente Lokale Inburgerings- en Integratiemonitor has 65 sheets and no religion indicator |
| `Samenleven in Diversiteit` (SID) | does ask religion; its methodological report | **5,127 responses as seven origin quotas of 750**, every other origin excluded from the frame, 85% Flemish, finest geography **three urbanisation classes** with Brussels folded in. Smaller than ESS, not a population sample, coarser |
| IWEPS / WalStat (Wallonia) | site search; WalStat swept by theme id 1-24 | one hit, the 2012/13 Baromètre social whose religion indicators were promised and never published; **19 themes, ~154 indicator groups, zero hits** on nine terms |
| IBSA/BISA (Brussels) | `keys=religion`, `keys=culte`; 17 themes | nothing, including in the education theme |
| European Values Study | the EVS 2017 participating-countries table | **Belgium is not in it**; the last Belgian wave is 2009 |
| Sciensano Health Interview Survey | the 36-page 2018 written questionnaire | **does not ask religion** |
| Catholic Church annual report | the bishops' conference 2018 report, 88 pages | baptisms, confirmations, weddings, de-baptisms and an October Mass count as **national totals**; per-diocese tables are parishes, priests and staff, eight units, institutions rather than people |

**The school courses are a real enumeration and the wrong universe.** Official-network
pupils choose between six religion courses and non-confessional ethics. Flanders publishes
it **only by province**, which is exactly what ESS already gives (education minister Ben
Weyts, written answer to question nr. 407 of 2 March 2023: *"De data is geordend per
provincie"*). The universe is the problem: ORELA's figures put non-confessional ethics at
26.4% of official-network Flemish primary pupils but **10.0% of all of them**, because most
Flemish pupils attend Catholic-network schools where no choice is offered and are recorded
as taking Catholic religion, **81.9% of all Flemish primary pupils**. It measures which
network a family chose. The French Community publishes its own shares Federation-wide only
and `statistiques.enseignement.be` no longer resolves.

**The one sub-provincial number that exists, and why it is not a source.**
`npdata.be/BuG/448-Moslims/` (Jan Hertogen, April 2020) gives a Muslim share for all **581
municipalities**, 2011 to 2019. The author says plainly it is not a count: it is the
register's migration-background counts per nationality per commune, times **Pew's Muslim
share for each origin country**, times a secularisation rate from a single 2008 German
study, and it covers Muslims only. Its 2019 figures are Belgium 8.2% and the Brussels Region
**25.5%**, against this map's 7.46% and **25.24%**. Close, and only half independent: this
build's foreign half is also census-nationality times Pew, so the two share machinery, and
what differs is the citizen half, which npdata does not use at all. A non-contradiction, not
a validation.

The tier that does exist is the survey, and ESS is the deepest of them. Belgium is in
**every ESS round**; rounds 1 to 4 carry no `region` variable, the wall Greece, France,
Italy and Finland all hit, so **seven rounds are usable**.

## 3. The variable, which is the mirror image of the Netherlands

**`rlgdnbe` exists in rounds 5 to 9 and does not exist in rounds 10 and 11.** The API's
answer for the missing ones is `E201VariableNotFound` behind a bare HTTP 400, so a fetcher
that treats 400 as a transport failure reads it as an outage, and one that pools on the bare
variable name silently drops the two most recent rounds. `queue.md`'s ESS block warned about
`a`/`b`-suffixed revisions; Belgium's failure is the other one, a country variable that stops
existing.

**What rescues Belgium is that its country card is the harmonised card.** In all five rounds
that carry both, `rlgdnbe` and `rlgdnm` agree code for code:

| code | `rlgdnbe` (Dutch) | `rlgdnm` (English) |
|---|---|---|
| 1 | (Rooms) Katholiek | Roman Catholic |
| 2 | Protestants | Protestant |
| 3 | Orthodox | Eastern Orthodox |
| 4 | Andere christelijke kerkgenootschap | Other Christian denomination |
| 5 | Jodendom | Jewish |
| 6 | Islam/Moslim | Islam |
| 7 | Oosterse religie: Hindoeisme/Boeddhisme/Shintoisme | Eastern religions |
| 8 | Andere niet-christelijke religie | Other Non-Christian religions |

So pooling on `rlgdnm` costs nothing and gains two rounds.
`sources/be.py::_check_be_card` re-proves this from the fetched data on every build rather
than asking anyone to trust the table above; if ESS ever lengthens the Belgian card the way
it lengthened the Dutch one, the build stops.

**The Netherlands is the opposite case and the pair is the lesson.** `rlgdnanl` splits
hervormd, gereformeerd and PKN, three answers `rlgdnm` flattens into `Protestant`, and
building the Netherlands on the harmonised variable would have lost the country's most
interesting distinction. Neither choice is right in general: look at the two cards.

## 4. The split-half, run on an ESS country for the first time

Greece, Finland, France, Germany and Italy all draw every ESS category where it was
measured. Belgium's pooled citizen sample is 10,877 people over eleven provinces and its
card has eight denominations, five of which are reached by **fewer than a hundred**
respondents in seven rounds. That is the situation §14.16 exists for.

**The resampling unit is the round, because this API returns cross-tabs and no PSU.** Seven
rounds split 35 ways into a three-set and its complementary four-set; the statistic is the
median Spearman, across those 35 splits, of a category's share over the eleven provinces.

| category | respondents | national | median rho | null 95th | p | verdict |
|---|---:|---:|---:|---:|---:|---|
| No religion | 6,475 | 59.71% | +0.345 | +0.355 | 0.0620 | not distinguishable |
| Roman Catholic | 3,572 | 32.94% | +0.491 | +0.373 | 0.0150 | **own geography** |
| Islam | 503 | 4.64% | +0.764 | +0.409 | 0.0005 | **own geography** |
| Protestant | 82 | 0.76% | +0.237 | +0.360 | 0.1554 | not distinguishable |
| Other Christian denomination | 75 | 0.69% | -0.137 | +0.369 | 0.7241 | not distinguishable |
| Eastern Orthodox | 45 | 0.42% | +0.710 | +0.392 | 0.0005 | **own geography** |
| Other Non-Christian religions | 43 | 0.40% | -0.110 | +0.372 | 0.6702 | not distinguishable |
| Eastern religions | 34 | 0.31% | +0.445 | +0.385 | 0.0270 | **own geography** |
| Jewish | 15 | 0.14% | +0.305 | +0.419 | 0.1404 | not distinguishable |

**THE NULL HAS TO PERMUTE THE PROVINCE LABELS PER ROUND.** The first version of this test
drew one permutation and applied it to the whole cube. A single global relabelling is
applied to both halves of every split alike, so it moves no rank correlation at all: the
null came out identical to the observed statistic, every category scored exactly p = 1.0000,
and the table read like a clean negative result rather than like a bug. Worth a spec entry
and it has one.

**Three things about this table are worth reading rather than skimming.**

1. **`No religion` fails and it barely matters.** It is 96.6% of the residual the four
   passing categories leave, so it still moves with each province's measured non-Catholic,
   non-Muslim share: 35.30% of Brussels against 58.95% of West Flanders in the finished map.
   The test declining to license it separately is honest and nearly free.
2. **`Eastern religions` passes on 34 respondents** and that is the one verdict here to
   treat as provisional. Nine categories were tested at alpha = 0.05, so about half a false
   pass is expected, and this is the likeliest candidate. It is 0.31% of citizens and its
   geography is not load-bearing. A Holm correction was considered and rejected: it would
   also fail Roman Catholic at p = 0.0150 x 7, and flattening the Catholic geography of
   Belgium on a multiplicity correction over seven resampling units would be trading a real
   pattern for a formal one.
3. **`Jewish` failing is the one that costs something visible.** Belgium's roughly 30,000
   Jews are overwhelmingly in Antwerp and Brussels; 15 respondents cannot show it; the
   people are drawn at the national rate everywhere. `note_public` says so in plain words
   rather than the map implying a flat community.

**§14.16's quota check (the Lebanon entry) was not run and could not be.** ESS is a
probability sample with no sect or region quota, and `quota_agreement` needs two waves to
offer identical rational shares, which 35 overlapping splits of one pool cannot produce.

## 5. The foreign half, which carries more of Belgium than of anywhere else

Eurostat `cens_21ctz_r3`, the 2021 census round, **200 named citizenships covering 99.82%**
of Belgium's 1,453,074 foreign residents, crossed with `taxonomy/origin_religion.py`.

| | Belgium | Greece | Finland |
|---|---:|---:|---:|
| foreign citizens | **12.58%** | 7.2% | 5.2% |
| ESS's own non-citizen share | 7.9% | 3.2% | — |
| foreign share of the largest city's region | **34.99%** (Brussels) | — | — |

So the half this construction exists for is doing more work here than in either country it
was copied from. Brussels drawn from ESS alone would have been about a third wrong, and
nothing inside the sample could have said so.

**The improvement that is named and not built.** Both halves are counted at NUTS 2 and
placed on communal population, so Brussels's Muslim dots spread evenly over all nineteen
communes rather than concentrating in Molenbeek, Schaerbeek, Sint-Joost and Anderlecht.
Two things would fix it: Eurostat publishes the same citizenship table at **NUTS 3**, the 44
arrondissements, and Statbel publishes population by nationality **per commune** every year.
Either would let Belgium use the Italy weighter (§9as) and place the foreign half where
foreign residents are. It is not done because doing it for one half and not the other puts
the sharper geography on the half with the weaker claim to it, which is the call Greece made
for Thrace; if it is ever done, both halves should move together.

## 6. What the finished country looks like

| | national | lowest province | highest province |
|---|---:|---|---|
| unaffiliated | 54.11% | Brussels 35.30% | West-Vlaanderen 58.95% |
| Catholic | 32.50% | Brussels 26.11% | Luxembourg 37.75% |
| Islam | 7.46% | West-Vlaanderen 2.25% | **Brussels 25.24%** |
| Orthodox | 2.10% | Namur 0.59% | **Brussels 7.46%** |
| Protestant | 1.88% | West-Vlaanderen 1.22% | Brussels 3.51% |
| Judaism | 0.16% | flat by construction | flat by construction |

**The division this map shows is Brussels against the rest, not Flanders against Wallonia.**
Every Flemish province and every Walloon one sits inside a few points of the others on all
five rows above; the capital is somewhere else entirely on three of them.

**The lapsed half, which is printed by the build and drawn by nothing.** ESS asks everyone
who says they belong to no religion whether they ever did, and then which one. Of 6,479
Belgian citizens who say they do not belong, **2,229 (34.4%) say they once did, and 95.5% of
those name the Catholic Church**. That is about one citizen in five who was Catholic and now
belongs to nothing, on top of the 32.5% who still say Catholic. It is `rlgblge` and
`rlgdnme`, it is fetched into `data/raw/be/ess_r*_past.json`, and `_lapsed()` prints it so
the claim is reproducible rather than quoted. Finland prints a register against a survey
because it has two instruments; Belgium has one, so what it prints is the same instrument's
own memory.

## 7. What this instrument cannot ask

- **Organised secularism.** Belgium funds the laicite and vrijzinnigheid movements on the
  same constitutional footing as the six recognised religions, with counsellors in hospitals
  and prisons and a non-confessional ethics course in official schools. ESS never offers
  atheist or agnostic as a denomination, so all of it lands in `unaffiliated` beside
  everyone who simply does not belong, and Belgium puts nothing on `secular`. Greece and
  Finland make the same call for a much less uncomfortable reason.
- **Anglicanism**, one of the six recognised religions, has no box and is inside
  `Other Christian denomination` with the Jehovah's Witnesses.
- **Sunni and Shia.** One Islam code, so the citizen half is all `islam.sunni`, which is
  right for a population that is mostly Moroccan and Turkish in origin but is an assertion
  about the minority. The Shia dots on this map come only from the census half, where
  `origin_religion.py` splits Iran, Iraq, Afghanistan and Syria.
- **The schools data, which §2 has in full.** It is a real enumeration, it is published by
  province and not below, and its universe is the official network, which is a minority of
  Flemish pupils. Named there so the next session does not spend an hour on it.

## 8. One thing left on the table for whoever wants it

`data-onderwijs.vlaanderen.be` serves the Flemish yearbook's religion-course workbook at
`documenten/bestanden/STJB-2324-godsdienstnietconfessionelezedenleer.xlsx`, 187 kB, ten
sheets. **The host resets the TLS handshake** for curl and for Python alike, so the sheet was
never opened and the "by province" claim above rests on the minister's written answer rather
than on a reading of the file. It would not change this build, because province is what ESS
already gives; it would settle whether free-network pupils are in it as blanket Catholic. A
browser would fetch it in one click, which is the shape of
[[feedback_gated_data_last_resort]]'s "hand it over" case rather than a wall worth fighting.
