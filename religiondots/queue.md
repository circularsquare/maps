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
| `fi` | Finland | 5,533,793 | ? | 18 | §11k's register tier, which "fails the same way four times". Left as recorded. |

## Elsewhere, ranked by what they would add

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `md` | Moldova | 2,804,801 | ? | 12 | Oracle backlog, never chased. |
| `am` | Armenia | ? | ? | 11 | The Apostolic/other split is unlike anything drawn. |
| `za` | South Africa | 62,027,503 | ? | 12 | Christianity undivided; behind a DataFirst account (§11b). The largest undrawn country in Africa. |
| `cv` | Cabo Verde | 491,233 | ? | 16 | §11w: open office, not chased past the catch-all. |
| `bw` | Botswana | 2,359,609 | ? | 9 | §11p: religion × language. |
| `rw` | Rwanda | 13,246,394 | 5 provinces | 11 | §11p: coarse, but §3.9b removed the floor. |
| `bi` | Burundi | 8,053,574 | ? | 8 | §11w: not chased. |
| `st` | São Tomé and Príncipe | 178,739 | ? | 13 | §11w: not chased. |
| `sc` | Seychelles | 81,755 | ? | 11 | §11w: not chased. |
| `lr` | Liberia | 3,476,608 | ? | 5 | §11w: not chased. |
| `mn` | Mongolia | 3,174,565 | ? | 6 | Oracle backlog. |
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
