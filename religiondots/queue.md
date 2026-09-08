# religiondots — countries to work on

**This file is for Claude, unlike `todo.txt`, which is Anita's and Claude does not edit.**

**If you were spun up to add a country and nobody is watching, read `AGENT_BRIEF.md` first.**
It says which calls are yours (almost all of them), the short list that is Anita's, and when to
stop — park with a handoff at a checkpoint rather than running out of context mid-mapping.
`python tools/claim.py` lists parked countries; **take one of those before a fresh one.**

Two or three agents work this directory at once (spec §12). Before starting a country:

```
python tools/claim.py                        # what is claimed, what is free
python tools/claim.py take <cc> --id <sid> --note "what you are doing"
```

`<sid>` is your session id — the last path component of the scratchpad directory in your
system prompt. Release it with `done` when you finish, and take the row out of the table
below (or move it to *Drawn*).

**The claim is advisory. It is a note on the door, not a mutex.** What actually prevents the
accident is spec §12's habit: **check before you write.** On 2026-09-08 one session spent an
hour on Peru that another had already finished, and overwrote its `sources/pe.py` with a
`Write` to a path it had not checked. This file makes that check one command.

**Do not trust this table about what is drawn** — `claim.py` derives that from `countries.py`
and from the dots on disk, and says so. A row here is a *candidate*, and it goes stale.

---

## Drawn

| code | country | people | units | where it stands |
|---|---|---:|---:|---|
| `fi` | Finland | **5,511,092** | **19 maakunnat** | **Built 2026-09-08, §9by. §11k's Nordic closure was right about the register and wrong about the country.** The register half is now confirmed much harder than §11k could: **all twelve databases at `pxdata.stat.fi` walked in full, 5,634 nodes, nine needles** including `kirkko`, `seurakun`, `luteril` and `ortodoks`, and the only religion tables in the whole system are the national `vaerak/11rx.px` plus parish payroll and a leisure survey's *"read religious or devotional books"*. `Kuntien_avainluvut`, Paavo and the experimental-statistics database were checked specifically because a **Norway-style *share belonging to the church* key figure would live there — Norway publishes one and Finland does not**, which splits two countries §11k had treated as one pattern. **So the register closes on what it measures rather than on reachability**, and Finland is where you can see the size of that: survey 45.00% Evangelical Lutheran against the register's **62.24%** at end-2024, a 17.25-point gap that is people who never filed the resignation form. **The two rows that go the OTHER way are the ones to carry** — Islam 1.67% here against the register's 0.48%, Orthodoxy 1.86% against 1.03% — because the register only sees members of a *registered congregation*, so it undercounts exactly the groups a naive reading would trust it most on. **The tier nobody tested was the survey**: Finland is in every ESS round with `region` at **NUTS 3 in all seven usable ones**, never dropping a level the way Italy's does, 12,741 citizens interviewed over rounds 5-11 at a median 455 per maakunta. §11ai listed the deferred Nordic ESS route for eight countries and **Finland is absent from it, because a country recorded as closed is not one anyone looks for a route to** — the generalisable lesson, and the fifth closure overturned on 2026-09-08 by the same move. Traps: **`table.path` in the ESS API is a list of INDICES into `values`, not code values**, so reading it as codes silently zeroes every category above code 9 and Finland would have shipped **with no Muslims and no error**; `rlgdnfi` does not exist and the variable is `rlgdnafi` (worth chasing, since harmonised `rlgdnm` files 45% of the country under `Protestant`); and **three NUTS vintages in seven rounds**, where the label-assertion refused its own first run on `FI181 Uusimaa -> FI1B1 Helsinki-Uusimaa` because Itä-Uusimaa was merged in 2011 and the region renamed. Also `11rx.px` is the id and `statfin_vaerak_pxt_11rx.px`, which the web UI shows, returns 400. Content: **no religion is the largest answer at 46.94%**, just ahead of Lutheran 45.00%; Lutheran runs 36.75% (Helsinki-Uusimaa) to 59.54% (Etelä-Pohjanmaa) and unaffiliated 32.89% to 53.06% (Lappi), so the axis is the Ostrobothnian west coast against the capital and the far north; and **Pohjois-Karjala comes out 6.22% Orthodox against 1.86% nationally**, the part of Karelia that stayed Finnish in 1944, which nothing told the model to expect. Open, both named rather than hidden: **Conservative Laestadianism cannot be drawn at all** (its members answer the national church's code, so `christianity.lutheran.laestadian` stays empty for Finland), and the placement is a plain population weight when `vaerak/11rq` would give foreign citizens per municipality and move Finland onto the Italy weighter for one PxWeb call. Boundaries cost nothing: GISCO LAU 2021 already on disk, 310/310 both directions. |
| `am` | Armenia | **2,932,731** | **11 marzes** | **Built 2026-09-08, §9bx.** §11o closed this with *"`armstat.am` census pages carry no religion table"* and that was wrong twice over. **The 2022 results page is a GIF image map whose eleven `<area>` polygons ALL point at the same national volume**, so clicking any marz returns the same file and the page reads as if only a national volume exists; the eleven marz volumes are one nid each (`945`-`953`, `956`, `957`) reachable only from the left nav. **And under `/en/` all eleven are a bare heading over *"Information is not available in English"* while the same nids under `/am/` carry nine section archives apiece.** The rule that generalises: an office's language versions are separate trees that can differ in what exists at all, so record which tree a negative came from. **The newer census is not the finer one here** (2011 and 2022 both stop at marz; 2022 taken for being fresher and longer), which is the opposite of Moldova and cost nothing to check. **The reconciliation check had to be the right shape or it would have rejected a correct read**: a marz prints only the columns it has people in, so Syunik names five religions and Yerevan fourteen and a rare answer with no column sits in that marz's residual. **483 of Armenia's 515 Muslims are printed and the other 32 are inside `Other` across seven marzes**, so a category's marz sum is a floor; what holds to the person is that each marz's columns close on its own population and the eleven totals sum to 2,932,731. The comparison then finds **8 people folding cannot explain** (Refused +6, Evangelical +1, JW +1), two Armstat publications disagreeing by 177 people, 0.0060%, bounded rather than waved through. Also: the header is **two rows** and `No religion`/`Refused` are printed one row HIGHER than the religions, which silently loses 66,854 people and still balances. 95.2% Armenian Apostolic, so the content is the last 5%: the Armenian Catholic north (Lori 3.15%, Shirak 2.89%, 77% of all Catholics), the Molokans of Lori (1,578 of 1,982), the Assyrians of Ararat (346 of 479). **The Yazidi finding is the one to carry and is NOT drawn**: the box is `Շարֆադինական`, Sharfadin, the community's own word for its religion where Australia, Georgia and the UK all use the ethnonym; and the national table crosses religion with ethnicity, so of **31,079 ethnic Yezidis, 13,256 answered Sharfadin and 9,939 answered Armenian apostolic**, with 1,672 in `Pagan` against 237 ethnic Armenians there. Those rows are carried in `am.csv` at `geo_level=country_by_ethnicity`, checkable and undrawn; reassigning them was refused under §14.5. **Open: `ask/004-am`** — `christianity.oriental` now carries 2,793,041 Armenians against Georgia's 109,041, so an `.armenian` child is arguable, and taking it re-points six mapping files and re-scatters six drawn countries. |
| `za` | South Africa | **54,946,360** | **9 provinces** | **Built 2026-09-08, §9bw. The corrected row was right that CS 2016 beats the census on categories, and the district tier does not exist openly for any vintage.** Drawn from the nine Community Survey 2016 provincial profiles: table 2.10a is 11 religions and table 2.10b is 14 Christian denominations, 24 published categories against Census 2022's 12. **`christianity.africaninstituted` gains 14,158,453 people, 1.21x everything the node held before (11,741,516 across six countries) and 2.32x Zimbabwe's cell, the largest any one country had contributed**, 25.8% of everyone who answered and 32.6% of Christians, running 15.8% of Western Cape's Christians to **50.8% of Limpopo's**, where the ZCC is headquartered at Moria. **The finer tier was hunted properly and is not there**: Report 03-01-84 *Cultural dynamics in South Africa* is province-only and COARSER (8 categories); **Census 2011 asked no religion question at all** (03-01-84 p.48), which kills Wazimap and every 2011 municipal product; and Stats SA's keyless Census 2022 dissemination API serves 24 topics down to **Main Place** with religion on none of them. Religion is the single variable the Census 2022 provincial profiles publish province-only while tabulating everything else by district and municipality. **AND THE TWO STATS SA RELEASES ARE NOT INTERCHANGEABLE EVEN WHERE THEIR LABELS MATCH VERBATIM**: no-religion is 10.9% in CS 2016 and 2.9% in Census 2022 while traditional African religion moves 4.5% -> 7.8% the other way, and Islam and Hinduism are stable to a tenth of a point; that is an instrument difference, so a §3.4 Brazil-style rescale was considered and REJECTED. Parser traps: the table number differs in four ways across the nine reports, five of them label the total row with the PROVINCE NAME, Northern Cape prints Bahaism as a dash, and **both tables have a row called `Other`** meaning different things. Report 03-01-11 (North West) is defective at source, its 14 rows summing to 90.1% of its own total; the 336,482 go to `denomination not reported` rather than being assigned on a guess, and the defect is asserted so a reissue fails the build. **Open: `ask/002-za` — a free DataFirst account (catalogue 611) would give the same 24 categories at ~234 local municipalities.** Also open: the 2001 census split the AICs six ways (ZCC 4,971,932, Other Apostolic 5,609,070, Shembe 248,824) and 1996 ran **64 categories**, both national-only so far. |
| `md` | Moldova | **2,409,207** | **901 UATs** | **Built 2026-09-08, §9bv.** The queue row was the **2014** census; the **2024** one, final results out 2025, beats it on every axis but two category names. 2014 publishes religion at **raion level only** (35 units; its commune sheet is sex and age), enumerated 2,804,801 against BNS's own estimate of 2,998,235 so about one in fifteen was never reached, and 6.88% did not answer. 2024 publishes **14 affiliations at UAT level, 901 units, ~2,700 people each**, and 0.75% did not answer. What 2024 loses is `Iudaism` (584 in 2014) and the Lutheran church (2,291), both now inside `Alte religii`, so **no Jewish dot is drawn in Bessarabia**. **The geography nearly went badly and the finding generalises**: geoBoundaries, HDX COD-AB and Kontur all stop at the raion, an OSM name join was built and got to 898 of 901, and then **BNS's own GIS server** turned up (`gis.statistica.md`, 212 hosted FeatureServers) with a 897-polygon commune layer keyed on the census's own CUATM statistical code **and carrying `p_distrib`, the census population, which matches this project's total to the person on all 896 joined units**. Chişinău's five sectors are drawn separately (567,038 people, 23.5% of the country) from OSM `admin_level=7` clipped to the city. Everything reconciles exactly: 901 UATs to the national figure in all 16 columns, and to each of the 35 raion rows. Open: sheet 5.35 crosses religion with ethnicity nationally and is not read; nothing splits the two Orthodox metropolises, which is Moldova's live religious question. |
| `bw` | Botswana | **1,384,276 aged 12+** | **485 localities** | **Built 2026-09-08, §9bu.** §11p's *religion x LANGUAGE only* is a true statement about the **2022** census, which asks the question and publishes no subnational table of any kind (Volumes 1-5, the technical report, four dissemination papers, the 2024 conference set, the 2025 Gender Monograph and the `/census-2022-data` CSV were all read). **The 2011 census is published the other way round**: its national volumes are geography-free too, but Statistics Botswana issued a separate *Selected Indicators* booklet per census district and **all eighteen print a religion table by named village**, 424 locality rows, ungated on `statsbots.org.bw`. Drawn on COD-AB ADM3, which tiles the country exactly; the name join is constrained within the district because **eight ADM3 names occur twice** and Kontur tests it at log-log r=0.744 against 0.174 for the best of 500 shuffles. **The booklets are eighteen typesetters**: the row-total column is first in ten, last in six and absent in one, and the captions lie outright (Serowe/Palapye labels both halves of its pair `(%)`; the Cities and Towns religion table is captioned *"marital status"*), so everything is detected arithmetically. **No religion is 14.8%, the highest in sub-Saharan Africa here** and rural rather than urban, but the 2022 census puts the same cell at 6.9% and does not explain the halving, which `note_public` carries. Open: **Central Boteti and Central Bobonong have no booklet** (129,312 people, 6.4%; 26 filenames and the whole Wayback CDX say they were never posted) and they are the more traditional half, since Badimo draws to only 88.2% of its national figure against 93.7% overall. They could be filled from the 2011 urban/rural split if anyone wants 100%. |
| `mn` | Mongolia | **2,067,841 adults** | 20 of 22 aimags | **Built 2026-09-08, §9bt.** The oracle says ABSENT and that is only about what was forwarded: the 2020 census asks `DO YOU HAVE A RELIGION?` (question 29, six answers) of everyone aged 15+ **in a 10% sample**, and has since 2010. **The national report has a whole religion chapter with NO geography in it** (sex, age, ethnicity, national only) and so does 2010's, which is the trap: the aimag figures live in **twenty-two separate per-aimag volumes** on the index-less static host `downloads.1212.mn`. Nineteen follow a filename pattern; the other three were found by sweeping the Wayback CDX of the whole `1212.mn` domain for the retired `BookLibraryDownload.ashx?url=` links, whose `url=` parameter is the live static filename. **No volume prints a single absolute figure**, so every count is two chained percentage tables times the 15+ population from the English national report's appendix 1.1. **Nothing is found by its caption**: twenty-two provincial offices agree on no table number, no caption wording, no column order, and six merge the two tables while one transposes its own, so each table is located by an ARITHMETIC IDENTITY (shares summing to 100.0) which a wrong page fails and a wrong caption match does not. Licensed by the national reconstruction landing within **0.15 points** of NSO's published shares on all six categories. First **measured** `buddhism.vajrayana` on the map; China's 6.3M are derived from nationality. Open: **Darkhan-Uul and Dundgovi are not drawn** (148,869 people, 4.7%) because their volumes have the tables as scanned IMAGES, needing OCR or six hand-read numbers each, §7 of `sources/mn.md`. **Do not chase a finer geography**: a 10% sample cannot support 339 soums and every volume stops at aimag for religion alone. |
| `cy` | Cyprus | 923,381 | **396 municipalities/communities** | **Built 2026-09-08, §9bs.** The queue's *answers on cystat.gov.cy* was half right. CYSTAT-DB's six census tables split exactly down the middle: **three cut LANGUAGE by district and three cut RELIGION by citizenship group, birth-country group and sex**, so the twelve national counts are exact and match UNSD table 28 to the person while nothing anywhere crosses religion with a place; the 2011 branch has none either. Drawn as CYSTAT's own arithmetic, P(religion | citizenship group) from table 1891632E applied to each community's four groups from 1891213E, reconciling to **0.0011 of a person** and using no coefficient from outside Cyprus. **The API prefix is the trap**: every other PxWeb here is `/pxweb/api/v1/`, this one is `/api/v1/`, and the wrong prefix returns a 500 whose body is a plausible error page. **The join is an integer Cyprus hands over** - PxWeb's community value CODES are the LAU codes, so the 396 units join GISCO LAU 2021 on the code and all 396 Latin names then agree as a free check. **The finding worth carrying, and it is not about Cyprus: an allocation reproduces exactly the concentrations that live in the dimension it allocates on, and is blind to every concentration inside one of its cells.** Testable here because 2001's Table 29 is the only religion-by-district tabulation any Cypriot census ever published: it measures Armenians at **1.88x** and Maronites at **2.07x** their population share in Lefkosia where the allocation returns a flat 1.0x, while Orthodoxy is reproduced across all five districts and the British Anglicans of Pafos are half found (3.37x measured, 1.86x allocated). Six dots' worth of error, accepted and named. Open: **17.3% did not answer** (the question was optional, which no other census here was); Keryneia has no census code and 5,846 km2 are drawn of 9,249; the TRNC's only census asks no religion question. The one upgrade that would matter is religion by COUNTRY of citizenship rather than the four-way group. |
| `py` | Paraguay | 3,892,603 | 229 distritos | **Built 2026-09-08.** §11t's negative was about the PUBLISHED output and it was right: INE's 2002 library has religion in exactly one table (CUADRO P11), national with an urban/rural split and four categories. The MICRODATA TABULATOR has the variable itself, `P17`, at 229 districts with **54 categories, the longest list on this map**. Route: `prod.redatam.org/redpry/` names its own cgi dir, then POST `binpry/RpWebStats.exe/Frequency` with `ROW=PERSONA.religion&AREABREAK=DISTRITO`. **What hid it was `<iframe>` vs `<frame>`** - the portal looks like a 4,611-byte empty shell to a frame-walker written for the older R+SP servers. Exact partition, three ways. Open: Asuncion is 6 census districts on 1 polygon, and 26 post-2002 districts are dissolved to a parent by shared boundary (11 close calls, none of which moves a count). |
| `bg` | Bulgaria | 6,519,789 | 265 obshtini | **Built 2026-09-08.** Both hosts §11o named are dead ends: `content/6704` 200s with the NSI *homepage*, and `censusresults.nsi.bg` is the 2011 census at oblast level. The route is `nsi.bg/search` then `statistical-data/151/1349`, and sheet 4 of the ethnocultural workbook is religion by municipality. Placement is the census's own 1 km grid. Christianity and Islam are undivided in that table and are **split by `bg_split.py`** from the 2011 oblast composition raked to the 2021 national totals, all of it `derived` and rolling back to `christianity` / `islam`. Open: a finer 2021 table would make it measured. `infostat.nsi.bg` serves the homepage on every path; the EU Census Hub is unchased. |
| `vu` | Vanuatu | 293,963 | **66 area councils** | **Built 2026-09-08, §9bg.** The queue said *6 provinces* and was out by a factor of eleven: Table 3.5's `Region` column is the whole census hierarchy, printing the same fourteen categories down to the **64 rural area councils** plus Port Vila and Luganville. No Fiji-style trade-off; VNSO publishes the deep list and the fine geography in one table. COD-AB ADM2 joins **66/66 on name** and is witnessed by OCHA's ADM1 against the census's own province grouping (64/64). **`indigenous.vanuatu` is new** for `Customary beliefs`, 3.1% nationally and **30.3% of South West Tanna**, which is John Frum country. The published table rounds each cell separately and misses its own totals by up to 2, verified on the page. Open: the 1999 census publishes religion **by island**, and 2020 also crosses it with sex and age. |
| `tr` | Türkiye | **85,279,553** | 12 İBBS-1 regions | **Built 2026-09-08, §11ac. §11r's *"Türkiye's own publication of religion is nothing since 1965"* was WRONG**, and the reason generalises: it asked TÜİK and asked the census, and religion here is published by the **Diyanet İşleri Başkanlığı**, with TÜİK's own design and fieldwork behind it. `Türkiye'de Dinî Hayat Araştırması` (Ankara 2014), 21,632 respondents, Table 4 on page 42 = sect by İBBS Düzey 1. **First source on this map to enumerate a madhhab at all**: `islam.sunni.hanafi/shafii/maliki/hanbali` and `islam.shia.jaafari` are new nodes, and Ortadoğu Anadolu is 48.7% Shafi'i against 45.7% Hanafi. State publication, so §14.4 rule 2 holds by construction. Every row `modelled` (§7b, Guatemala's test). **The open item is Alevis**: the questionnaire has no Alevi box, the word appears zero times in 293 pages, and they are inside the 10.4% drawn on the parent `islam`. KONDA's whole archived library, WVS (Türkiye's denomination list is Islam/Orthodox/other/none) and ESS were all checked and **none publishes an Alevi share by region** — the national figure is 5.02% (KONDA 2006) and is stated in the note rather than drawn. Nişanyan is closed by Anita's call. |
| `gt` | Guatemala | 17,843,132 | 22 departamentos | **Built 2026-09-08, §11ad, and the first country here drawn from the AmericasBarometer.** Guatemala's census has never asked; §11x closed it on the oracle and IPUMS together and this does not reopen that. Pooled LAPOP shares on COD-PS 2024 populations, **every row `modelled`**. The finding worth carrying: **which categories may carry a geography is decided by a SPLIT-HALF across waves, not by size** — a 4% size cut would have drawn `Protestante Tradicional` (5.43% of the country) whose split-half rank correlation is **-0.04**. Catholic +0.57 and Evangelical +0.50 clear the +0.43 bar. **`Ninguna (creyente)` at +0.21 is UNDER the bar and drawn anyway, on Anita's call of 2026-09-08** — the departments differ at chi-square p=3.6e-16, so the split-half says the ordering is not pinned rather than that the variation is fake, which is §14.16's China at +0.17. The bar was not moved; `sources/gt.py`'s `OVERRIDE` names the one category and prints the reason every run. The rest are drawn at the national rate inside each department's residual, so nobody is deleted and only the claim to know where they are is withdrawn. Open: the non-Christian tail wants a second source. |
| `sv` | El Salvador | 6,350,969 | 14 departamentos | **Built 2026-09-08, §11ad and §9bl, the second AmericasBarometer country.** The best-sampled of the nine: 647 respondents a department against Guatemala's 405, so the split-half comes back much stronger and **THREE categories carry their own geography rather than two** — Catholic +0.88, Evangelical +0.82, and `Ninguna (creyente)` at **+0.74** where Guatemala's failed at +0.21. First country in this set whose no-religion geography is a measurement. **Two findings worth the visit.** The code join is Guatemala's INVERTED: LAPOP's `prov` is the official west-to-east number and COD's `SV` pcodes are ALPHABETICAL, so `prov-300` mispairs twelve of fourteen while every total still reconciles; joined on name, with the mispairing asserted. And **the age held-out check was withdrawn from BOTH countries** after being measured: its between-unit variance is below its sampling variance (F=0.36 here, 0.88 in Guatemala), so Guatemala's +0.52 was luck. Replaced by a permutation test. Open: `Protestante Tradicional` misses the bar by 0.02; the non-Christian tail wants a second source. |
| `ec` | Ecuador | 16,910,403 | 23 of 24 provincias | **Built 2026-09-08, §11ad and §9bn, the third AmericasBarometer country and the most Catholic of them at 75.5%.** Three findings worth the visit. **THE ANSWER CARD CHANGED AFTER 2016 and it affects `gt` and `sv` too**: `Otro` is exactly zero in 2010/2012/2014 across all 28 countries and Mormons, Jews and Jehovah's Witnesses are exactly zero in 2018/2023, so those three are floors and `Otro` is inflated at the late end. **A UNIT can fail the split-half, not just a category** — Carchi, Pastaza and Orellana are in the 2010 wave only, so they are assumed at the national rate (2.76%), while **Galápagos is drawn EMPTY** because LAPOP has no code 920 and nothing measured it at all; Anita's line is whether anything measured the place, not how thin the sample is. A neighbour-average fallback was tested and lost. And **the population is INEC's 2022 census, not COD-PS**: the 2020 projection is 3.4% high and uneven (Loja −6.9%, Manabí +2.0%) and carries a 25th row with no polygon. Held-out r=+0.994 over 23 provinces, the strongest in the set. Catholic runs 94.0% in Loja against 61.7% in Sucumbíos. Open: Galápagos needs any Ecuadorian source (try the 2006 Galápagos census); INEC's own 2012 religion module is located, open and **unwired** — five cities, urban, and its `Otra` coding looks unstable. |
| `sb` | Solomon Islands | 720,956 | **183 wards** | **Built 2026-09-08, §9bh.** The queue said *9 + Honiara*; Vol 2's Table P8.3 is religion by **WARD**, 3,940 people each, the finest tier of any Pacific country here. **The only table on this map that reconciles to the person**: 183 wards, 10 provinces and every ward's own row all sum exactly, no tolerance anywhere. Joined on **SINSO's own ward id** (183/183) because a name join matches only 154 — prenasalised stops are written both ways, and Isabel ward 02 is `Baolo` to the census and `Havulei` to COD, which is a rename a name join could not see. New nodes: **`christianity.melanesianindependent.cfc`** (Silas Eto's Christian Fellowship Church, 77.2% of Kusaghe ward) and **`indigenous.solomon`**. The South Sea Evangelical Church, 17.3%, went to `christianity.evangelical` on Kenya's Africa Inland Church precedent and is in REVIEW. Open: 2009 at the same ward tier; P8.4 ethnicity; `solomons.popgis.spc.int` has a `p11_religion` dataset behind a disabled endpoint. |
| `to` | Tonga | 99,408 | **156 villages** | **Built 2026-09-08, §9bj.** The queue's unit count was `?`; TSD publishes religion **by village** in a spreadsheet, 637 people per unit, the finest tier on this map by population per unit. The media API is open and is the WRONG DOOR (1,396 files, 53 PDFs, no census table) - the tables are on WP File Download and its URLs appear only in PAGE BODIES, so one `wp/v2/pages` call gets the whole library. **Closes five ways**, the fifth being UNSD table 28 on all 22 categories to the person, which is the only check that is not a copy of the same typesetting. G 20 has NO TIER MARKER and G 19 is the key: it prints divisions and districts alone, so walking it gives every G 20 row's tier. **53.4% of the country is Methodist in four churches** and `christianity.methodist.tongan` is new with four children - flagged for Anita against her todo note on single-country categories. The 1946 Niuafo'ou eruption resettled seven village NAMES onto 'Eua and the twins are unalike ('Esia is 71.4% Catholic there, 62.7% Free Wesleyan here), so the join is district-qualified; the five that do not pair are **witnessed by OSM point-in-polygon**, which caught COD labelling two 'Eua polygons `Ohonua` with no Ta'anga at all. Open: 2016 at the same tier, the 1996-2021 national series, and a 2026 census now in the field. |
| `hk` | Hong Kong | 7,413,070 | 18 DC districts | 8 | **Built 2026-09-08, §14.24, as mainland China's sibling and in the same two layers.** Hong Kong, Macau and Taiwan are all excluded from the mainland by `cn_geo.py`'s `NOT_MAINLAND` and none had ever been considered; `hk` appeared nowhere in spec.md, sources.md or queue.md before today. **No HK census has ever asked about religion** (the 2021 round publishes its 46 topics and it is not among them) and HK is absent from the oracle, so: §14.5's ethnic derivation at 18 districts plus a `self_id` survey for the territory, carved out of the `unknown` residual, which is `_cn_counts`'s arithmetic unchanged. **The finding worth carrying: a derivation coefficient can come from this map's OWN drawn countries.** Indonesia 87.51% `islam`, Pakistan 96.47% `islam`, the Philippines 78.88% `christianity.catholic.latin`, each recomputable from that country's entry here. It cannot drift away from the rest of the map and it inherits every later correction to the source country. **But it only fixes the PROVENANCE of the coefficient, not whether it applies** - Anita: *"probably pretty bad for large origin countries cuz the people migrating are probably skewed in some way"*, which is §14.12. spec §14.24 turns it into a permission with three conditions (origin share near 1; direction of selection known and stated; an independent check on the result), at least one of which must hold, and refuses the derivation otherwise - which is what happens to the Indians and Nepalese here. **gov.hk's fact sheet is refused with evidence**: every figure on it is the religious body's own claim, it says 1,040,000 Protestants in Jan 2026 where the same office said 480,000 in Jul 2022 and where the churches' own 2024 census counted 197,935 at weekly worship, down 26% in five years, and its Muslim and Hindu numbers are each about twice what the census's ethnic counts and the survey independently agree on. **The rule that decides it: when two sources with no lineage in common agree and a third does not, the third is the one to leave out.** Second finding: **a migrant derivation's geography can be an EMPLOYMENT geography rather than a settlement one** - the Muslim share runs 3.00% (Wan Chai) to 1.51% (Kwun Tong) and is highest in the wealthiest districts, because ~93% of the Indonesians and Filipinos are live-in domestic workers, while the sharp enclave geography belongs to the Nepalese and Indians, who are refused as §14.12 cases. Islam comes from the census and not the survey, China's rule. **HK measures §14.22's gap in one instrument**: 56.07% of the whole sample practise folk religion while 65.83% claim no affiliation, so nothing is drawn on `chinesefolk` and §3.1 is why. **THE THING TO FIX NEXT IS SPATIAL, and it is flagged for whoever picks it up** (Anita: *"hong kong being 1 spatial component is pretty bad"*): 31.8% of every district is the territory-wide survey and varies not at all, the derived layer moves only between 1.51% and 3.00% Muslim, so the 18 districts do almost no work and Hong Kong reads as a national pie chart with a population-weighted scatter. Routes, best first: **a survey with district-level religion** (HKPSSD or Asian Barometer, both behind an application, so §11b's order applies), the 2024 Church Survey's district tables (paid report, worth an email to the HKCRM), the Catholic diocese's parish statistics, and only alongside those the 452 constituency areas on data.gov.hk. sources/hk.md §8 has it in full. Also open: **Macau and Taiwan are the same gap and Taiwan is much the larger prize.** |
| `ws` | Samoa | 205,557 | **25 districts** (read at 339 villages) | **Built 2026-09-08, §9bk.** §11aa's *"worth an hour before committing"* was right, and the hour is spent. The DATA is the cleanest on this map: 26 categories summing to the total at **every one of 395 place rows**, four indentation tiers nesting exactly on all 27 columns, **no `Not stated` column at all**, so Samoa is 100% drawn. The GEOMETRY is the worst in the Pacific: no COD-AB, and **GADM level 2 is the SAME 43 polygons as geoBoundaries** (one Pacific Data Hub source, not two). The 43 are a DIFFERENT CUT from the census's 51, not a coarser one - `Vaimauga 1..4` against `Vaimauga East/West`, neither nesting, stem alone settling 23.1% of people. **The fix is that both are cuts of the same 25 traditional districts**, so both fold onto those 25 by name, asserted set-against-set. The 43-unit route via OSM village points reached only 85.2% and was REJECTED, with the diagnosis written down. New nodes: **`...congregational.cccs`** (55,411, and `.ekt`'s own note had already predicted it) and **`christianity.adventist.sisdac`** (Aso Fitu, counted by no other census on earth). Open: the 43 once geocoding is closed; the census's 51 need a boundary file nobody has; `data.sbs.gov.ws` is an unexplored .Stat/SDMX instance. |
| `ki` | Kiribati | 110,136 | **24 islands** | **Built 2026-09-08, §9bl.** Three candidate sources and the newest is the wrong one: 2020 publishes religion NATIONALLY ONLY and its Census Atlas map of it is a **raster** (66 chars of text on the page); the per-island workbooks reach **village** but are 2005 with column sets that differ island to island. **2015 report Vol 1 Table 6** is the newest with a geography and the longest list. Islands sum to national exactly on 15 columns, and UNSD table 28 agrees on all 14 categories to the person. **TWO TRANSFERABLE FINDINGS.** (1) The oracle's LABELS for this country are wrong while its numbers are right - `KPC` is expanded to `Kempsville Presbyterian Church` and the 1995 row is mangled outright - so pair on an alias table, never on the string. (2) **The antimeridian check has to be INVERTED**: Kiribati's bbox legitimately spans 351°, so the country-level assertion fires on good data; check per polygon instead. Fiji (§9bd) needed the opposite. New node **`...congregational.kpc`**, the fifth and last of the Pacific Congregational set. Open: the 2005 village tables; the 2020 island figures exist inside KINSO unpublished; `Te Ran` (86 people, a printed cell nobody can identify). |
| `sg` | Singapore | 3,459,094 | **31 planning areas** | **Built 2026-09-08, §9bp.** The queue said *exceptional category depth*; nine categories is middling here (the Philippines maps 129, Vietnam 28) and the shape is what is unusual, Sikhism named at 12,051 people while 411,674 non-Catholic Christians stay in one cell. Source is **TableBuilder CT/17592 via data.gov.sg**, wide open, whole country rebuilds in under a minute; the printed release tabulates religion seven ways and **not by geography**, so the PDF is a check and not a route. **The finding worth carrying: a population grid can be fine enough and still be the wrong instrument.** Kontur clears the resolution floor easily at 400m over 31 units and is still refused, because resolution was never the problem, the UNIVERSE was: it counts everyone present, and **1,641,590 people here are non-residents the religion table excludes** (Tuas has 70 residents, Sungei Kadut 750). Placed instead on the census's own 332 subzone resident populations, same census, same office, same universe; r=0.99857. Second finding: **`Changi- Total` is printed with no space before the hyphen**, so a positional parse silently hangs Changi's three subzones on Central Water Catchment and 54 planning areas parse where URA has 55 - [[reference_name_join_wrong_neighbour]] in positional form. New node `other.sg`. Two calls argued in REVIEW and confirmed by a second agent: `Taoism` to **`chinesefolk`** on the source's own footnote (Hong Kong's goes to `daoism`, deliberately), `Other Christians` to the **`christianity` root** on `lk2024.py`'s precedent. **Universe is 60.8% of the island** (residents 15+; 585,117 children and 1.64M non-residents excluded, neither scaled up) and it is the flattest religious geography on this map, because the Ethnic Integration Policy has capped each group's share per housing block since 1989. Open: **non-residents are published by pass type, not nationality and not by planning area**, so a Hong Kong style derivation has no denominator; Rochor (Little India, Kampong Glam) is inside the 25-area `Others` row and a finer TableBuilder cut would fix it. |
| `sz` | Eswatini | 1,093,238 | **4 regions** | **Built 2026-09-08, §9bq.** §11w's top African lead and its `no reachable host` was right about the OFFICE and wrong about the country: `eswatinistats.org.sz` still times out and its 61 Wayback captures still hold no PDF, but the census volumes are Joomla articles on **`www.gov.sz` under `/images/FinanceDocuments/`**. Route: one CDX sweep of the WHOLE government domain, grep `census`, read the archived article's hrefs. **`Zionists` are 367,290 people, 33.60%, so `christianity.africaninstituted` is a country's plurality for the first time** — and unlike Zimbabwe the CSO counts them beside eleven named mission bodies rather than as one cell next to `Protestant`. **Religion by region is a CHRISTIAN-ONLY table** (13 denominations, Table 3.2.4); the 9 top-level religions are national. Each region's non-Christian total is still a count (its Table 5.2.2 population minus its Christians) and only the split is carried down, so 89.25% `measured` / 10.75% `derived`. Six tables read and one drawn, because every identity inside 3.2.4 survives a column permutation; the check that pins it is each region's Christian share against its population, which **exactly 1 of the 24 orderings passes** (enumerated, not sampled). **KONTUR IS WRONG ABOUT THIS COUNTRY** — 0.38x Hhohho, 2.24x Lubombo, 43% of the country dumped in Lubombo against a census 19%, because it is built from OSM building footprints and the Lowveld sugar estates are mapped while the Highveld homesteads are not. Placed on WorldPop constrained instead (0.95-1.04), with a second WorldPop release of the census year as a control. New node `other.sz`. Open: the CSO holds religion at **tinkhundla** and published only maps of everything else (Volume 2 has 29 tinkhundla maps and no religion), so a table request or a microdata route would be 14x finer; `No religion` at 7.4% has no geography here and Zimbabwe's runs 4.5-13.5% across provinces. |

| `at` | Austria | 8,032,926 | **2,380** (2,357 Gemeinden + 23 Wiener Bezirke) | **Built 2026-09-08, §9br.** The queue's **31 categories** were NATIONAL: every subnational table in the 2001 publications carries **ten**, and Tabelle 15 (the Bundesland cross-tab, where depth would live) carries the same ten crossed by age. So this is ten on a very fine geography, ~3,400 people a unit, and not the deepest list in the world. **THE TRAP IS THAT THE SOURCE NUMBERS ITS COLUMNS IN A DIFFERENT ORDER FROM THE ONE IT PRINTS THEM IN**: the eight Länder volumes head Tabelle 4 `1 2 3 5 4 6 7 8 9 10 11`, so Orthodox is printed fourth and numbered 5 while Evangelisch is printed fifth and numbered 4 — and the **Wien** volume numbers the same columns in print order. Keying on the number swaps Orthodoxy and Protestantism in eight volumes of nine and **nothing catches it**, because the swap is consistent inside each volume and every total still reconciles. Columns are identified from the header LABELS by x-position, with the order asserted; Vorarlberg's separately published `.xls` of the same table prints the same anomaly, so it is the source's. That `.xls` also checks one whole Bundesland against a machine-readable original (101 rows, 1,111 figures, all equal). **Boundaries are GISCO's `Communes 2001`**, the census's own Gebietsstand, so Styria's 542→287 merger of 2015 never arises and no crosswalk was written: 2,357 of 2,358 join on the Kennziffer, names agreeing 99.4% as a check. **The fourteen Statutarstädte are printed once at the Bezirk tier and never with a Gemeinde code** — 1,044,429 people, 13.0% of the country and all its big cities, silently absent unless minted. **UNSD's 31 rows decompose the ten drawn columns EXACTLY** (six reproduce a column, the other 25 partition the remaining four with no row used twice), which is the strongest parse check here and also proves a split that is **deliberately not applied**: those figures exist at one geography, so a split would give all 2,380 units an identical mix and turn 88,977 measured people derived for no spatial gain. Drawn at 2001 as published; §3.4 refused because the only recent source is a **Mikrozensus sample** whose publisher marks cells under 3,000 uninterpretable, and whose categories do not crosswalk (no `Unbekannt`; its `Christentum` residual is 5× the 2001 column). Open: **Vienna at Zählbezirk** (all 245 are parsed and in `at.csv`, undrawn, because the current OGD layer has 250 and the vintages do not correspond); `pages/402/Religion.ods` sheet A2 has ~60 national denominations, deeper than the 31. |

## Verified this session, data seen with my own eyes

*(empty — Samoa was the only entry and it is drawn, §9bk.)*


## Oceania — the region nothing in `sources.md` had ever mentioned (§11aa)

Offices probed 2026-09-08: fourteen of seventeen answer a plain GET. HDX has COD-AB for all
of these except Samoa. **PDH.stat has no religion dataflow** — do not re-check the hub.

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `pg` | Papua New Guinea | **10.19M** | 89 districts | 13 | **Scouted to the end 2026-09-08, §11ab — BLOCKED ON IPUMS, not on PNG.** The office is wide open (same WP File Download plugin as Fiji) and the whole 287-file library was swept: **religion is published nationally and nowhere else**, in 2000, 2011 and the 2022 SDES alike. The provinces get one line — *main religion* — a plurality, not a partition. But **IPUMS holds the 2000 microdata at DISTRICT level** (520,609 records, RELIGION universe = all persons), so this is one extract away from being better-resolved than the queue assumed. Dead ends retired: `png.popgis.spc.int` is a 530 with zero Wayback captures, and the `dotstat` post type is an **unused plugin**, not a .Stat instance. Population is 2024-census; the old ~7.3M was 2011. |
| `fj` | Fiji | ~885k | provinces | 11 | **The most religiously plural country in the Pacific** — Methodist / Hindu / Muslim — which is the one thing Samoa and Tonga cannot show. Oracle row is 2007; the 2017 religion release was not located. |

## Microstates — DRAWN 2026-09-08, all nine, from the oracle alone

`sources/micro.py` + `sources/micro.md`. Nine countries, 311,888 people, **no office
contacted**: `tools/oracle.py` has UNSD table 28 *with its counts*, and eight of the nine
partitions close to the person. `pw` `ck` `tv` `nu` `ms` `bm` `ag` `dm` `mh`.

New nodes: **`modekngei`** (a root), and the three London Missionary Society national churches
under `christianity.reformed.congregational`. Niue and Montserrat draw **zero dots** and are
rings only, which found a crash in `buffers.py` and a silent drop in `tiles.py`.

**Four of the eleven remain**, because the oracle has no row for them: `fm` Micronesia,
`nr` Nauru, plus `pg` and `sb` from the Oceania table below. Those still need their offices —
except **`pg`, whose office has now been opened and emptied (§11ab): it publishes religion
nationally only, and the route left is IPUMS.**

## Microstates — unlocked by Anita's call, 2026-09-08 (four left)

*"for the really small island countries we might not even need any divisions. like for
instance if we do palau, it has 17000 people so itll just be 17 dots."* At 1 dot = 1,000
people the placement carries no claim, so **a national-only source is buildable** — one
polygon, Kontur placement. This retires the third floor after §3.9b (unit count) and §3.9c
(variety). §11v had explicitly parked the first three of these as *"a decision rather than a
build"*; this is that decision.

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `fm` | Micronesia | 107,008 | 1 | 8 | `fsmstatistics.fm` answers. |
| `nr` | Nauru | ~11k | 1 | ? | `nauru.prism.spc.int` answers. |

## Europe — what is left after four sweeps (§11o, §11aa)

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `si` | Slovenia | 1,964,036 | ? | 14 | §11c and §11k both closed it — correctly — on the **2021 register census, which does not ask**. The oracle's row is **2002**, which did. Nobody has looked at the 2002 tables. |
| `al` | Albania | 2,402,113 | 61 bashki | 10 | **`instat.gov.al` did not resolve on 2026-09-06, -07 or -08** — connection layer, not a 403. The only source anywhere that counts **Bektashi** apart from Sunni. Its oracle row is now **2023**, a newer census than §11o priced. Browser job. |

## Elsewhere, ranked by what they would add

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `cv` | Cabo Verde | 491,233 | ? | 16 | §11w: open office, not chased past the catch-all. |
| `rw` | Rwanda | 13,246,394 | 5 provinces | 11 | §11p: coarse, but §3.9b removed the floor. |
| `bi` | Burundi | 8,053,574 | ? | 8 | §11w: not chased. |
| `st` | São Tomé and Príncipe | 178,739 | ? | 13 | §11w: not chased. |
| `sc` | Seychelles | 81,755 | ? | 11 | §11w: not chased. |
| `lr` | Liberia | 3,476,608 | ? | 5 | §11w: not chased. |
| `tl` | Timor-Leste | 1,341,737 | ? | 6 | Oracle backlog. |

---

## Latin America from LAPOP — §11ad, THREE BUILT (`gt` §9bi, `sv` §9bl, `ec` §9bn), six left

The AmericasBarometer grand merge is on disk at `data/raw/lapop/`, 301,156 respondents, and it
carries religion at ADM1 for nine countries this map does not have. **Six of them are the ones
§11x closed for having no census religion question at all**, so this is the only route known.

Its provincial cut was validated against three censuses already on the map (Mexico +0.918
Catholic, Peru +0.776 evangelical, Suriname **+0.981 Hindu** on a district-concentrated
minority). **It is sound for anything above roughly 1% of a province and noise below that.**

### HOW TO BUILD ONE, now that three exist

**Read §11ad, then §9bi (Guatemala), §9bl (El Salvador) and §9bn (Ecuador), then copy
`sources/sv.py`.** It is the shortest and the one whose join is hard. Four files per country:

    sources/<cc>_geo.py    ADM1 boundaries + populations, and THE JOIN
    sources/<cc>_grid.py   Kontur 400 m hexes, the placement layer
    sources/<cc>.py        ~120 lines: the decode, the pop table, and calls into lapop.py
    taxonomy/<cc>2023.py   the same eleven answers every time, plus an `other.<cc>` node

**`sources/lapop.py` holds everything that is the same in every country** — the category list,
the wave set, `load`, `national`, `stability`, `held_out`, `build`. Do not re-implement any of
it; its docstring says what must stay per-country and why.

**READ THIS ONE FIRST — THE ANSWER CARD CHANGED AFTER 2016 AND IT AFFECTS ALL NINE (§9bn).**
Code 77 `Otro` is EXACTLY ZERO in 2010, 2012 and 2014 across all 28 countries, on ~29k answers
a wave, and appears from 2016. Codes 6, 10 and 12 — **Mormons, Jews and Jehovah's Witnesses**
— are EXACTLY ZERO in 2018 and 2023, on 15,107 and 25,649 answers. Zero Witnesses among 25,649
Latin Americans is a withdrawn box, not a measurement: the named small denominations were
folded into `Otro`. So in every pooled build **those three cells are floors and `Otro` is
inflated at the late end**. Nothing leaves the partition and the country total stays right;
what moves is the attribution of about half a percent between two small nodes. Ecuador states
it in `taxonomy/ec2023.py` rather than correcting it, because correcting it means deciding how
the 2023 `Otro` decomposes. **`gt` and `sv` were built before this was known and neither says
so.** Say it in the fourth country, and decide deliberately whether to go back.

**FIVE THINGS §11ad DID NOT KNOW**, all of them found by building:

1. **The `prov`→polygon decode is different in every country and is the dangerous step.**
   Guatemala's COD pcodes are the official department numbers, so it joins on the code.
   El Salvador's are ALPHABETICAL, so `prov-300` mispairs twelve of fourteen while every total
   still reconciles. **An identifier scheme that worked in the last country is a hypothesis.**
   Join on the name, and assert what the code join would have done.
2. **Which categories carry their own geography is decided by a SPLIT-HALF, not by size.**
   Rank the units on the early waves, rank them on the late waves, correlate; the bar is
   1.96/sqrt(n-1). Guatemala's `Protestante Tradicional` is 5.4% of the country and returns
   -0.04. `lapop.stability()` does this and asserts the answer against `CARRIES`.
3. **The held-out check is a PERMUTATION test.** `r = +0.968` means nothing on fourteen skewed
   units; "none of 20,000 random pairings reaches it" is the evidence. An age-structure check
   was tried, asserted on, and withdrawn from both countries when it turned out to have no
   power (§9bl). Do not reinstate it.
4. **Expect a country to differ from its neighbour in what it may CLAIM.** Guatemala's
   traditional-religion cell is named a floor in `note_public`; El Salvador's is not, because
   its 2007 census counted 0.2% indigenous. Same instrument, same cell, opposite write-up.
   Ecuador's IS a floor: 0.06% against a country 7.69% indigenous by its own 2022 census.
5. **A UNIT can fail the split-half too, and COD-PS may be the wrong population (§9bn).**
   Ecuador's Carchi, Pastaza and Orellana appear in the 2010 wave and no other, so the
   split-half never ranks them, yet the naive build still draws them on their own shares —
   plus Galápagos, which LAPOP does not even offer a code for. **The line Anita drew is
   whether ANYTHING measured the place**, not how thin the sample is: the three that were
   measured once are assumed at the national rate (2.76% of Ecuador), and **Galápagos is
   drawn EMPTY into `gap=`** because nothing measured it at all. Check the per-unit wave
   table before trusting any unit's shares. A neighbour-average fallback was tested against
   the national rate and lost, so do not re-propose it.
   **AND BEWARE THE NATIONAL-RATE FALLBACK AT SCALE.** It is not neutral — it asserts that a
   place resembles its country, so it is most wrong exactly where a region is distinctive,
   and distinctive regions are the prominent ones. Ecuador survives it at 2.76% across three
   obscure provinces. **The Dominican Republic would not: 10 of its 32 provinces are
   unmeasured** (3 never sampled, 7 in one wave-half only), and at that point a
   province-level map is mostly inference wearing the same colours as measurement. Anita,
   2026-09-08: *"probably cant do DR without additional data."* **Do not build `do` from
   LAPOP alone.** Panama is milder than it looks — LAPOP does sample Comarca Ngäbe-Buglé
   (n=138); what is missing is Guna Yala and Emberá-Wounaan, small but exactly the
   distinctive case, plus Panamá Oeste which is a 2014 boundary split rather than an
   unsampled place.
   **And check whether the country has counted since COD-PS's vintage** — Ecuador's COD-PS is
   a 2020 projection against a 2022 census that came in 3.4% lower and unevenly (Loja −6.9%,
   Manabí +2.0%), plus a 25th row with no polygon, so `ec` is drawn on INEC's census instead.
   COD-PS is a projection with a date on it, not the neutral choice.

The rest of §11ad's gotchas still stand and are not guessable: Honduras is disqualified as it
stands, two codes have blank labels, and the Mexico name join needs five aliases.

| code | country | n pooled | ADM1 | Catholic | Ev+Pent | none+ath | where it stands |
|---|---|---:|---:|---:|---:|---:|---|
| `do` | Dominican Republic | 8,904 | 29 of 32 | 53.2% | 20.2% | 18.0% | **DO NOT BUILD FROM LAPOP ALONE (§9bn, Anita 2026-09-08: *"probably cant do DR without additional data"*).** Ecuador's rule applied here leaves **10 of 32 provinces unmeasured** — three never sampled, and seven (Barahona, Monseñor Nouel, Monte Cristi, Elías Piña, Santiago Rodríguez, Hato Mayor, Dajabón) in one wave-half only, several on n under 20. A third of the units at the national rate is inference wearing the same colours as measurement. Needs a Dominican source first; ONE publishes census tabulations. |
| `cr` | Costa Rica | 5,903 | 7 of 7 | 63.1% | 13.8% | 10.6% | Clean, and only seven units, so §3.9b applies rather than a unit floor. |
| `uy` | Uruguay | 4,318 | 19 of 19 | 37.0% | 8.8% | **48.4%** | **Nothing else in the Americas is half grey.** Only three waves (2010–2014), so the level is fifteen years old. Blank label on `1407`, which is Flores. |
| `pa` | Panama | 6,105 | 10 of 15 | 64.4% | 21.1% | 7.2% | Two comarcas unsampled and Panamá Oeste still inside Panamá. §11x calls Panama's own geography *"a perfect placement layer attached to nothing"*, and this is something to attach to it. |
| `ht` | Haiti | 7,252 | 10 of 10 | 49.0% | 6.7% | 8.6% | **26.9% traditional Protestant, which no other country here approaches.** Vodou at 4.4% is a self-identification floor and Anita has flagged it as needing a second source before it is drawn. |
| `hn` | Honduras | 9,293 | **broken** | 45.5% | 29.7% | 10.6% | **Do not draw from `prov` in the merged file.** 22 codes for 18 departments, four labels duplicated, and the sample sizes do not match the country. Needs the per-wave releases. |

## Drawn from one survey alone — the standing refinement list

*Anita, 2026-09-08: "for any coutnries that just lapop, we should mark that as countries that
would probably benefit a lot from further refinement."*

**A country enters this list when it is built from LAPOP alone and leaves it only when a second
source is actually wired in** — not when one is found, and not when one is proposed. What every
one of them needs is the same thing: **the non-Christian tail**, which LAPOP's answer card does
not offer and which §11ad measured going missing at a factor of five in Suriname.

| code | built from | what would fix it |
|---|---|---|
| `gt` | LAPOP 2010-2023 pooled, on COD-PS 2024 | **The non-Christian tail.** `Religiones Tradicionales` is 0.22% in a country that is 43.6% indigenous, and §11ad measured the same instrument reading 0.21x a census on exactly that cell in Suriname. Nothing has been searched for yet; the USCB per-country geodatabases were checked and Guatemala is not among the 34. Also wanted: anything at all below the department, since LAPOP's `municipio` is a PSU list rather than a partition. |
| `sv` | LAPOP 2010-2023 pooled, on COD-PS 2024 | **The non-Christian tail**, as for every LAPOP-only country: `other.sv` is 2.35% and nearly two thirds of it is the Eastern-religions box, with El Salvador's Palestinian and Lebanese communities the obvious thing to chase. Nothing searched yet. NOT the traditional-religion cell — 0.03% is plausibly right here, unlike Guatemala (§9bl). Also wanted: `Protestante Tradicional` clears +0.54 and gains a fourth measured layer if a later wave lands. |
| `ec` | LAPOP 2010-2023 pooled, on **INEC's 2022 census** | **Galápagos draws nothing at all** and needs any Ecuadorian source; LAPOP will never supply one, because the province is not on its card. INEC's 2006 Galápagos census (`censo_galapagos_2006.zip`, open) has not been checked for a religion question and is the first place to look. **And a second national source is already located and not yet wired: INEC's own `Filiación Religiosa`, ENEMDU 2012, 13,211 respondents, microdata open at `bdd_filiacion_religiosa.zip`.** Its card names **Islámica, Budismo, Judaísmo, Espiritismo and Religiones Afroamericanas** separately, which is exactly the tail LAPOP cannot see. Three cautions before using it (§9bn): five cities only and 100% urban, so it is a national cross-check and not a placement layer; 2012, the wrong end of a twelve-point fall in Catholic identification; and **its `Otra`/`Ateos` coding looks unstable between cities** — `Otra` is 19.4% in Cuenca against 0.2% in Machala while `Ateos` runs the other way. Also wanted: the 2.93% of the country drawn at the national rate (Galápagos, Carchi, Pastaza, Orellana), which any Ecuador-specific source would fix. |

## Closed, with the reason — do not re-derive these

Kept short; the long version is the `sources.md` section named.

- **Morocco, Algeria, Tunisia, Libya, Mauritania** — every office reachable, none publishes the
  question. **The ministry re-test this row asked for was RUN on 2026-09-08 (§11af) and the row
  holds.** Morocco's Habous ministry, Algeria's Ministère des Affaires Religieuses (moved to
  `marw.gov.dz`) and Tunisia's are **mosque-administration bodies** with fatwa banks, sermons
  and Quranic-education sections and **no statistics arm at all**; Türkiye's Diyanet was
  unusual in having a research directorate and TÜİK behind it. The **Arab Barometer** was then
  pooled across every wave for all five: 99.2-99.7% Muslim, and the largest non-Muslim cell in
  any of them is **19 respondents**. The **Ibadis** of the M'zab, Djerba and the Nafusa are the
  region's real sect story and pooled AB sees **4, 7 and 4 of them**; no source anywhere gives
  an Ibadi magnitude by region, and that stays open. **Do not build a madhhab layer off AB's
  sect item** — 45-82% answer "just a Muslim" and wholly-Maliki Morocco returns 16% Maliki.
- **Iraq, Saudi Arabia** — the other half of §11r's row, and **the ministry re-test has NOT been
  run on these two.** Saudi Arabia's Islamic Affairs ministry is still unknocked. On §11af's
  evidence expect little: the door only pays when the religion body publishes numbers about
  people.
- **Egypt** — **REOPENED 2026-09-08 (§11af) and now blocked on §14 ALONE; see `ask/001-eg`.**
  Pooled Arab Barometer (waves III, IV, V, VII, 6,840 respondents, 457 Christians) **passes the
  split-half at +0.495 against a +0.418 bar** over 23 governorates, puts Sohag, Minya and Asyut
  top, and agrees with the last census that published: **5.93% against 1986's 5.7%**, and Cairo
  **8.50% against the census's 8.57% (1996)**. Two of §11d's four reasons are dead — CAPMAS ran
  a 544-study NADA catalogue at `censusinfo.capmas.gov.eg` (now DNS-dead, fully archived, and
  its dictionary does carry a religion variable), and IPUMS is no longer the only route. What
  has not moved is §14.4 rule 2, and that is the whole question.
- **Sudan** — closed on a document rather than an inference: **the Presidency deleted the
  religion question from the 2008 census**, in UNSD's own country report's words, because
  *"ethnicity and religion are causes of conflicts in Sudan"*. `cbs.gov.sd` SERVFAILs at both
  public resolvers. The USCB geodatabase carries no religion or ethnicity sheet (§11af).
- **Togo, Uganda, Mozambique, Guinea-Bissau, Zambia (2022), Namibia** — religion published
  nationally or not at all (§11p, §11w).
- **Panama, Ecuador, Guatemala, Honduras, El Salvador, Costa Rica** — no religion variable in
  any census; Panama across five censuses and 252 tables (§11x). Strongest negatives here.
- **Venezuela, Colombia** — closed 2026-09-08, from each office's OWN variable dictionary
  rather than by reputation (§11ac, `sources/py.md` §4). Venezuela: `redatam.ine.gob.ve` is
  live, base `CPV2011`, **67 person variables and none is religion**, corroborated by INE's
  24-page person metadata. It was marked *unchecked* in `sources.md` for the whole project.
  Colombia: **CNPV 2018 has 116 variables, Censo 1985 has 71, GEIH 2023-26 has 760**, none
  religion-shaped; the only religion variable in DANE's entire catalogue is in a Uniandes
  panel survey. **South America is now exhausted** apart from Paraguay, which is drawn.
- ~~**Laos** — 2015 census, national only~~ — **DRAWN 2026-09-08 at VILLAGE level, §9bk.**
  The report is national only and always was; LSB publishes the same census on `k4d.la`, the
  2008 socio-economic atlas's data platform, as 400-odd ArcGIS services. **8,499 villages,
  763 people each.** The 31.4% "no religion" is drawn as `indigenous.laos` on the atlas's own
  gloss, the report's own "no religion *or animist*", Pew's `<0.1%` unaffiliated, and a
  geography that runs 96.5% in Dakcheung against 5.9% in Vientiane Capital. **The lesson to
  carry: the sweeps looked for tables and this is a map server.** Any country whose census
  was mapped by a European development institute in the 2000s deserves the same look.
- **Thailand's 2010 province tables** — the server is gone; DRAWN from an allocation instead
  (§9as).
- **Japan** — the roll counts buildings and catchments, not people (§11q). Deferred, not closed.
- **Kazakhstan's 218 rayons** — available and deliberately refused; the model draws 17 (§9aq).
- **PDH.stat** (SPC's Pacific hub) — 127 dataflows, **no religion**. The offices have it; the
  hub does not (§11aa).

---

## Angola-shaped work: UPGRADES to countries already drawn — added 2026-09-08 after `ao`

Anita, when Angola came in: *"are there other countries that you think might need a similar
workflow?"* Angola was **two separate misses at once**, and both are cheap to check on any
country here. Neither needs a new country: these are countries already on the map that are
drawn from less than their office published.

**Nobody had looked since the country was ingested.** Angola's 2024 census was published
20 November 2025 and January-February 2026; the map she brought was the 2014 one. The check
is ONE request to the office's publications listing, filtered for anything newer than the
`how=` vintage in `countries.py`. Run `python -c "import countries..."` (the vintage table is
easy to print) and work down it.

**And the national report was the wrong document.** Angola's national volume has religion by
PROVINCE; the 21 provincial volumes have it by MUNICIPALITY. A country drawn at a coarse tier
may only be coarse because nobody opened the provincial series. Ask, for every coarse country:
*did this office publish a per-province or per-district volume of the same round?*

### A. Newer round than the one drawn — check the office's listing

Ranked by what a newer round would add. **Verify before building; these are leads, not
findings.** Several may turn out not to ask religion, or to publish it nationally only.

| cc | drawn from | worth checking | why it matters |
|---|---|---|---|
| `id` | census, **2010**, six permitted answers | SP2020, and the Dukcapil civil registry, which publishes religion by kabupaten annually | 270M people, the largest stale entry on the map by a factor of four |
| `bd` | census, **2011**, upazilas | BBS ran a census in **2022** and published religion | 265,000-person upazilas, a whole new round |
| `vn` | census, **2009** | the **2019** census | vn is at provinces (1.4M) AND limited to state-recognised organisations; a newer round may not fix the second problem |
| `th` | census, **2010**, provinces | the **2020** round | 868,000 per unit is among the coarsest here |
| `br` | **2022 totals with 2010 denominations** | whether IBGE has released Censo 2022's religion tables | would replace a 2010 denominational structure on 32,000-person municípios |
| `nz` | **2023 totals with 2018 denominations** | same shape, Stats NZ | as above |
| `fj` | census, **2007** | the **2017** round | fj is at 14 provinces; 2017 may go finer |
| `jm` `tt` `gy` | 2011, 2011, 2012 | the 2022-2023 Caribbean round | three countries, one regional round |
| `py` | census, **2002** | the **2022** census | CAREFUL: py's 2002 REDATAM tabulator gives **54 categories on 229 districts**, the longest list on the map. A 2022 round is only an upgrade if it is at least as deep — check before replacing |
| `ch` | census, **2000**, resized to 2024 totals | the annual structural survey / cantonal registers | 26 years, and §11k already looked at the register tier |
| `cf` `ni` `et` `mm` | 2003, 2005, 2007, 2014 | whether a later round exists at all | all four are known-hard; `mm` ran a census in 2024 under a government whose coverage claims cannot be taken at face value |

### B. Same round, finer volumes — the provincial-series check

Countries already drawn where the office very likely published a per-province or per-district
series carrying the same table. **This is the Angola workflow exactly**: 21 volumes instead
of 1.

| cc | drawn at | ask for | prize |
|---|---|---|---|
| `zw` | **10 provinces, 1.5M each** | ZIMSTAT 2022 PHC **provincial reports**, and whether they carry Table 2.14(c) at district | **The top pick.** The coarsest African unit on the map, and its `Apostolic Sect` is 40.3% of the country, the largest single religious answer there. 10 → 63 districts |
| `ke` | 47 counties, 1.0M | KNBS 2019 county reports, sub-county tables | Kenya supplies most of `christianity.africaninstituted`'s non-Zimbabwean half |
| `ci` | régions, 887,000 | ANStat RGPH 2021 at département | §9az already found the office half-shut; the série régionale is the thing to ask for |
| `kh` | provinces, 622,000 | NIS 2019 district (srok) volumes | |
| `ph` | provinces and cities, 930,000 | PSA 2020 city/municipality special release | |
| `ge` | regions, 334,000 | GEOSTAT 2014 municipal tables | |

### C. And the boundary half, which is separate

Angola's tier did not exist in any boundary file because **Lei 14/24 redrew the country
between the fieldwork and the publication**. Whenever an office publishes on a tier COD-AB
does not have, see [[reference_agol_statute_boundaries]] and `sources/ao_geo.py`: the
statute's own boundary text is usually traced into a public ArcGIS Online feature service,
and it can be checked against the statute, the census's own count-per-province table, and
Natural Earth. **Do not plan a dissolve on a stale file's attributes without reading them** —
COD-AB Angola's commune parentage is wrong in Luanda, which is where the people are.

---

## From the lead triage — added 2026-09-08, §11ag

Anita ran a lead list past Gemini; §11ag triages all eight rows against the record. **Five rows
were already decided and are not repeated here** — the Maghreb plus Libya and Sudan (§11af,
closed on pooled Arab Barometer itself), Egypt (`ask/001-eg`, and this section does not
duplicate it), the `Q602` sect item (§11af, refused), Pew (no geography in the public-use file),
and IPUMS (blocked, `[[reference_ipums_account]]`). **Three leads closed outright on the checks
run for §11ag** and must not be re-derived: **Belarus** is not in ESS, confirmed both ways (the
string does not occur on the participating-countries page and `rlgdnby` returns nothing), and
has no route of any kind; **Turkmenistan** is in neither LiTS round; **Iran** closes twice over,
since the **Iran Social Survey** is stratified by all 31 provinces but its data link is an empty
placeholder and it has simply never been released, while **GAMAAN** is an online opt-in
sample with an effective n of 1,911 and is national, on top of §11n's §14 finding.

What follows survives. `sources/lapop.py` already implements the construction every row here
needs — pool the waves, weight, cut by ADM1, gate each category on `stability()`, take the
magnitude from COD-PS — and was written country-agnostic on purpose. **Reuse it.**

### A. Arab Barometer east of Egypt — §11af swept North Africa and stopped there

**All four are BLOCKED behind `ask/001-eg` and must not be built before Anita rules on it.**
§14.4 rule 2, no resolution finer than the state's own publication, is exactly what stopped
Egypt and it bites at least as hard here. Access is settled and clean (§11af): the real file
URLs are in the download page's own HTML at `www.arabbarometer.org/wp-content/uploads/`, no
retrieval or redistribution clause, nine waves. **Nothing is on disk** — `data/raw/` has no
`arabbarometer` directory — so all four cost one re-fetch. The governorate variable is
confirmed for Egypt (23 units, §11af) and **unverified for these four**.

| code | country | people | units | where it stands |
|---|---|---:|---|---|
| `lb` | Lebanon | ~5.4M | governorates (8) | **The one to want.** No census since **1932**; §2 still carries it as *"needs a note of its own"*. Sect geography here is the most-wanted unmapped religious distribution in the region. Whether AB's sect item behaves better than it did in the Maghreb is genuinely open and different: sect is a salient public identity in Lebanon and an unmarked default in Morocco, which is exactly why §11af's 16%-Maliki result may not transfer. §14 is the whole question. |
| `iq` | Iraq | ~44M | 18 governorates | Sunni and Shia by governorate for the first time on this map. §11r closed Iraq on the **census and the office**, never on AB, and its ministry re-test was never run either. Yazidis and Christians make §14.4 rule 2 acute. Absent from the oracle. |
| `jo` | Jordan | ~11M | 12 governorates | Thinner prize, comes free with the same file. Absent from the oracle. |
| `ye` | Yemen | ~34M | governorates | Thinnest of the four and the hardest boundaries; take last. Absent from the oracle. |

### B. Central Asia via LiTS — the one wholly untouched instrument

**`LiTS`, `EBRD` and `Life in Transition` return zero hits across the whole project.** None of
these is in the oracle. Only Kazakhstan has ever been scouted here (§11u: religion is
national-only, four independent ways; the ethnicity route is Anita's call and unspent).

**RESOLVED and fully open — no auth, no cookie, no terms gate.** HEAD on the LiTS IV CSV zip
returns 200 and 23,580,948 bytes. The LiTS III CSV header carries `country, PSU_name,
district_l1, district_l2, region_name, urban, latitude, longitude, weight_population,
weight_sample`, so there are **two subnational levels plus PSU coordinates**. **LiTS III's
questionnaire p41 asks "What is your religion?"** in six codes: Muslim / Orthodox Christian /
Other Christian including Protestant / Jewish / Atheistic-agnostic-none / Other. Thin, but the
right shape for Central Asia, where the line that matters is Muslim against a Russian Orthodox
remnant.

**Two things to settle before building.** (1) **It is NOT confirmed that LiTS IV kept the
religion question** — its stated scope mentions religion only as a ground of discrimination,
which is a different item. **LiTS III (2015-16) is the confirmed religion round**; check LiTS
IV's own questionnaire before planning to pool, and pooling is the only way the sample gets
respectable. (2) **The sample per region, which is worse than the target implies.** LiTS III
fielded 1,500 per country; **LiTS IV achieved 1,006 in Uzbekistan, 1,034 in Tajikistan and
1,002 in the Kyrgyz Republic.** Over a dozen Uzbek regions that is ~75-110 per unit, **thinner
than anything drawn from a survey on this map** (Greece ~600, El Salvador 647, Guatemala 405).
If ADM1 will not clear the split-half, the honest answer is the coarser design stratum or
nothing — LAPOP's `estratopri` lesson.

**And it reaches Kazakhstan, which is not one of the countries the lead named and may be the
best use of the file.** LiTS IV achieved **1,028 in Kazakhstan**. §11u established four ways
that Kazakh religion is national-only and left a §14.5/§14.10 ethnicity model as Anita's call —
and **§14.10 requires an independent check on that model's output**, which nothing currently
supplies. A small self-id survey with a region variable is exactly that check.

| code | country | people | units | where it stands |
|---|---|---:|---|---|
| `uz` | Uzbekistan | ~35M | 12 regions + Karakalpakstan | Largest of the three and the one worth doing first. In LiTS III. |
| `tj` | Tajikistan | ~10M | 4 regions | In LiTS III. Ismaili Gorno-Badakhshan is the interesting cell and is also the one a 1,500-household sample is least likely to see. |
| `kg` | Kyrgyzstan | ~7M | 7 oblasts + 2 cities | In LiTS III as *Kyrgyz Republic*. |

### C. Europe via ESS — a named, priced, deliberately-deferred route rather than a new lead

**ESS is already this map's most productive survey instrument** — Germany (§9g), France (§9ab),
Greece (§9z) and Italy are drawn from it, `api.nsd.no/graphql` is open and keyless and
cross-tabulates server-side, and every convention and trap is in `sources/gr.md` §1-§3. §11o's
own closing line already named it as *"the only route"* to these countries and then chose France
as *"the one worth the trouble"*. §11k's *"do not re-scout the Nordics"* is about the **register**
tier (Norway one denomination for 891 municipalities, Denmark flows not stock, Iceland's only
geographic table is 1981, Sweden nothing) and does not reach ESS.

**Round coverage is now established** and is in the `units` column below. What is still per-country
is **the NUTS level, because it varies by round within one country** — Italy is NUTS-2 in rounds
6 and 8 and NUTS-1 in 9-11 and the later files simply do not contain the finer level (§9bp).
`regunit` names it; assert it.

**THE ONE THAT WILL BITE.** All five `rlgdn<cc>` variables exist as spelled, but several
countries carry **`a`/`b`-suffixed revisions with different category lists** in the later
rounds: `rlgdnanl`, `rlgdnase`, `rlgdnaua`, `rlgdnapl`/`rlgdnbpl`, `rlgdnask`/`rlgdnbsk`,
`rlgdnacy`. **Pooling on the bare variable name silently drops the later rounds** — no error,
just a smaller n and an older category card. **Match on the prefix and assert the round count
you expected.** The Netherlands, Sweden and Ukraine would all have hit this.

| code | country | people | units | where it stands |
|---|---|---:|---|---|
| `nl` | Netherlands | ~17.9M | **R1-R11, all rounds**; NUTS-2 (12 provinces) to verify | **Best of this block, and the deepest sample in it.** `rlgdnanl` exists and gives the Netherlands its own Protestant taxonomy, which is the whole point in the one country where the bevindelijk gereformeerde geography is the story (Staphorst, Urk, Rijssen, Barneveld). §11k closed CBS `82904NED` as national-only; that is the census/register tier, not this. |
| `be` | Belgium | ~11.7M | **R1-R11, all rounds**; NUTS-1 (3) or NUTS-2 (11) to verify | If only NUTS-1, three units is still not a floor (§3.9b) but is barely more than the Flanders/Wallonia/Brussels split §2 already calls thin. Check the level before committing. |
| `se` | Sweden | ~10.5M | **R1-R11, all rounds**; NUTS-2 (8) to verify | SCB carries nothing at all, so ESS is the only route. |
| `no` | Norway | ~5.5M | **R1-R11, all rounds**; NUTS-2 (7) to verify | Pairs against table `12025`'s Church of Norway share for 891 municipalities as an independent check on one category, which is the sort of witness §11ac liked. |
| `dk` | Denmark | ~5.9M | **R1-R7, R9**; NUTS level to verify | **Denmark's last round is R9 and there is a gap at R8**, so it has been out of ESS for several rounds. Buildable, but it is a historical picture and `grain` must say the vintage rather than implying a current one. |
| `lv` | Latvia | ~1.9M | **R3, R4, R7, R9, R10, R11**; NUTS level to verify | Six rounds, irregular. The lead's "rounds 10/11" is right as far as it goes but understates it; pool all six. |
| `ua` | Ukraine | ~38M | **R2-R6, then R11**; oblasts to verify | A long gap and then one recent round. **The pre-war and post-invasion halves are not the same country to sample** and pooling across that gap needs an argument, not a default. Wartime boundaries are their own problem. Take last of this block. |
| ~~`lu`~~ | ~~Luxembourg~~ | ~660k | **R1-R2 only** | **EFFECTIVELY CLOSED.** Two rounds, 2002-2004, nothing since — twenty years stale on top of being one unit, which makes it a national pie chart drawn from a survey older than most of the map. Do not chase. |

### D. Sub-Saharan Africa via DHS — named in §8 since the project started, never opened

**Institutionally walled, and it needs Anita.** Downloading the recode microdata requires an
account plus a **research project title and a description of the proposed analysis**, reviewed
by DHS staff, normally 24-48 hours — and the registration is **institutional**, not a personal
email signup, so it is dearer than §11s's *"one free registration away"*. **Redistribution is
barred outright**: no DHS file may be committed to this tree, and §6b's *"cite the origin, not
the badge"* applies. **IPUMS-DHS at `idhsdata.org` is blocked separately**
(`[[reference_ipums_account]]`).

**CHECK BEFORE ANITA SPENDS ANY TIME ON THIS.** A claim circulates that **DHS suspended new
user applications on 2025-02-07**. It could not be confirmed against DHS's own pages and is
**unconfirmed** — but if it is true the whole block below is closed rather than walled, so
verify it first.

**Two negatives established for §11ag, so nobody walks them twice.** The open keyless API
(`api.dhsprogram.com/rest/dhs/indicators`) lists **4,655 indicators, of which 10 mention
religion and none is a composition indicator** — they are other things broken down by religious
group, never "what share of this region is Muslim" — and **`breakdown=background` offers no
Religion category at all** for Nigeria 2018 or India. So **STATcompiler and the REST API can
never supply a religion distribution.** And a DHS **final report is not a shortcut**, checked on
the biggest one: **Nigeria's FR359 has 734 table captions, exactly one mentions religion**, and
there religion is a **row block inside Table 3.1 sitting parallel to the Zone and State blocks
rather than crossed with either**. The `v130` × `v024` cross exists only inside the microdata,
which is why the wall is the whole cost.

§14.16's test applies unchanged before drawing any of these, and each category gates on the
split-half rather than on its size (§9bi).

| code | country | people | units | where it stands |
|---|---|---:|---|---|
| `ng` | Nigeria | ~230M | 37 states | **The prize, and the largest single addition available to this map anywhere.** §11b already priced NDHS 2018: ~40k households, **state-representative**, religion on the household roster, *"far better than modelled"*, and MICS 2021 is the same shape. The census *"does not ask"* (§11p) and Nigeria is absent from the oracle, so this is the only route to the most consequential unmapped religious boundary on earth. |
| `tz` | Tanzania | ~67M | regions | Religion dropped after **1967**; absent from the oracle. DHS is the only route of any kind. |
| `ug` | Uganda | ~46M | regions | §11b: the 2024 census has **the best unclaimed category list in Africa**, published national-only. DHS supplies geography for categories that already exist. The UBOS microdata email (`ubos@ubos.org`) is the competing route and may be better. |
| `mz` | Mozambique | ~33M | 11 provinces | §11b's *"most interesting untried source"*: the 2017 census has **Sião/Zione at 16.3%**, which would be the second source ever to count the Zionist churches. Microdata is a request on `mozdata.ine.gov.mz`; DHS is the second route to the same country. |
| `zm` | Zambia | ~20M | 10 provinces | 2010 census reported 8 categories, 2022 published 5 (§11w). DHS supplies geography. |
| `cd` | DR Congo | ~105M | 26 provinces | **§11h already REFUSED the other route** — *Enquête 1-2-3*'s cells are 31,755 *heads of household sampled*, not people. DHS is a genuine person-level source where the refused one was not. Big country, nothing drawn. |
| `gn` | Guinea | ~14M | 8 regions | Oracle says the 2014 census has 6 categories; §11w found **no religion volume** in the RGPH-3 series. DHS supplies geography. |

### E. Argentina — the CONICET survey, at six regions and not 23 provinces

| code | country | people | units | where it stands |
|---|---|---:|---|---|
| `ar` | Argentina | ~46M | **6 regions** | **The LAPOP half of the lead is false**: §11ad, Argentina appears only in the pre-2010 waves and **carries no religion at all**. The census half is closed twice (absent from the oracle; §11ae's *"South America is exhausted"*). **CONICET is genuinely new to this record.** *Segunda Encuesta Nacional sobre Creencias y Actitudes Religiosas en la Argentina* (CEIL-CONICET 2019, second edition after 2008), openly published at `conicet.gov.ar` and in the repository at `ri.conicet.gov.ar`, no account. **n = 2,421 over AMBA, Centro, Cuyo, NEA, NOA and Patagonia** — regionally representative by design, and NOT the 23 provinces plus CABA the lead claims. Catholicism 62.9%. Six units for 46M is 7.7M per unit, almost exactly Türkiye's grain as built (§11ac), and §3.9b removed the unit floor. **The real constraint is ~400 respondents per region**, thinner than any LAPOP country drawn (El Salvador 647, Guatemala 405), and with one wave the split-half has to run across some other split. Say the grain honestly. **Six regions is the CEILING, checked and not assumed: the breakdown is on p18, there is no province table anywhere in the report, and no microdata is offered.** So Argentina is buildable at six units or not at all, and nobody should go hunting for a provincial cut. Closed on tier, wide open on access. |
