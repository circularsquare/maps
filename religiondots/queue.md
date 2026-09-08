# religiondots — countries to work on

**This file is for Claude, unlike `todo.txt`, which is Anita's and Claude does not edit.**

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
| `py` | Paraguay | 3,892,603 | 229 distritos | **Built 2026-09-08.** §11t's negative was about the PUBLISHED output and it was right: INE's 2002 library has religion in exactly one table (CUADRO P11), national with an urban/rural split and four categories. The MICRODATA TABULATOR has the variable itself, `P17`, at 229 districts with **54 categories, the longest list on this map**. Route: `prod.redatam.org/redpry/` names its own cgi dir, then POST `binpry/RpWebStats.exe/Frequency` with `ROW=PERSONA.religion&AREABREAK=DISTRITO`. **What hid it was `<iframe>` vs `<frame>`** - the portal looks like a 4,611-byte empty shell to a frame-walker written for the older R+SP servers. Exact partition, three ways. Open: Asuncion is 6 census districts on 1 polygon, and 26 post-2002 districts are dissolved to a parent by shared boundary (11 close calls, none of which moves a count). |
| `bg` | Bulgaria | 6,519,789 | 265 obshtini | **Built 2026-09-08.** Both hosts §11o named are dead ends: `content/6704` 200s with the NSI *homepage*, and `censusresults.nsi.bg` is the 2011 census at oblast level. The route is `nsi.bg/search` then `statistical-data/151/1349`, and sheet 4 of the ethnocultural workbook is religion by municipality. Placement is the census's own 1 km grid. Christianity and Islam are undivided in that table and are **split by `bg_split.py`** from the 2011 oblast composition raked to the 2021 national totals, all of it `derived` and rolling back to `christianity` / `islam`. Open: a finer 2021 table would make it measured. `infostat.nsi.bg` serves the homepage on every path; the EU Census Hub is unchased. |
| `vu` | Vanuatu | 293,963 | **66 area councils** | **Built 2026-09-08, §9bg.** The queue said *6 provinces* and was out by a factor of eleven: Table 3.5's `Region` column is the whole census hierarchy, printing the same fourteen categories down to the **64 rural area councils** plus Port Vila and Luganville. No Fiji-style trade-off; VNSO publishes the deep list and the fine geography in one table. COD-AB ADM2 joins **66/66 on name** and is witnessed by OCHA's ADM1 against the census's own province grouping (64/64). **`indigenous.vanuatu` is new** for `Customary beliefs`, 3.1% nationally and **30.3% of South West Tanna**, which is John Frum country. The published table rounds each cell separately and misses its own totals by up to 2, verified on the page. Open: the 1999 census publishes religion **by island**, and 2020 also crosses it with sex and age. |
| `sb` | Solomon Islands | 720,956 | **183 wards** | **Built 2026-09-08, §9bh.** The queue said *9 + Honiara*; Vol 2's Table P8.3 is religion by **WARD**, 3,940 people each, the finest tier of any Pacific country here. **The only table on this map that reconciles to the person**: 183 wards, 10 provinces and every ward's own row all sum exactly, no tolerance anywhere. Joined on **SINSO's own ward id** (183/183) because a name join matches only 154 — prenasalised stops are written both ways, and Isabel ward 02 is `Baolo` to the census and `Havulei` to COD, which is a rename a name join could not see. New nodes: **`christianity.melanesianindependent.cfc`** (Silas Eto's Christian Fellowship Church, 77.2% of Kusaghe ward) and **`indigenous.solomon`**. The South Sea Evangelical Church, 17.3%, went to `christianity.evangelical` on Kenya's Africa Inland Church precedent and is in REVIEW. Open: 2009 at the same ward tier; P8.4 ethnicity; `solomons.popgis.spc.int` has a `p11_religion` dataset behind a disabled endpoint. |

## Verified this session, data seen with my own eyes

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `ws` | Samoa | 205,557 | 43 or 339 | 26 | §11aa. `CensusTablesEXCELFiles.xlsx` Table 2, **exact partition**, four tiers nested by indentation. **The blocker is geometry**: no `cod-ab-wsm` on HDX, OSM has 11 admin relations for the whole country, geoBoundaries ADM2 = 43 against the census's 51 districts. Resolve 43↔51 by name, or find a village layer. |

## Oceania — the region nothing in `sources.md` had ever mentioned (§11aa)

Offices probed 2026-09-08: fourteen of seventeen answer a plain GET. HDX has COD-AB for all
of these except Samoa. **PDH.stat has no religion dataflow** — do not re-check the hub.

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `pg` | Papua New Guinea | **10.19M** | 89 districts | 13 | **Scouted to the end 2026-09-08, §11ab — BLOCKED ON IPUMS, not on PNG.** The office is wide open (same WP File Download plugin as Fiji) and the whole 287-file library was swept: **religion is published nationally and nowhere else**, in 2000, 2011 and the 2022 SDES alike. The provinces get one line — *main religion* — a plurality, not a partition. But **IPUMS holds the 2000 microdata at DISTRICT level** (520,609 records, RELIGION universe = all persons), so this is one extract away from being better-resolved than the queue assumed. Dead ends retired: `png.popgis.spc.int` is a 530 with zero Wayback captures, and the `dotstat` post type is an **unused plugin**, not a .Stat instance. Population is 2024-census; the old ~7.3M was 2011. |
| `fj` | Fiji | ~885k | provinces | 11 | **The most religiously plural country in the Pacific** — Methodist / Hindu / Muslim — which is the one thing Samoa and Tonga cannot show. Oracle row is 2007; the 2017 religion release was not located. |
| `to` | Tonga | 99,408 | ? | 22 | Deep list on a small country. The **Free Wesleyan Church** is the state church and nothing here counts it. `tongastats.gov.to` answers; `wp/v2/search` is disabled, so walk its publications pages. |
| `ki` | Kiribati | 110,136 | ~23 islands | 14 | **Confirmed**: religion is a tabulated census topic and the Census Atlas 2022 maps it. `nso.gov.ki/download/<id>/` is an open Download Monitor catalogue — **the id is the catalogue**, and nobody has swept it. |

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
| `at` | Austria | 8,032,926 | Gemeinde | **31** | **The deepest undrawn religion list in the world.** Both hosts answer. The objection is staleness alone — 2001 is the last census that asked, and §11d found zero religion hits in `data.statistik.gv.at`'s catalogue. §11j's test (was the category itself the target of something that moved it) passes. |
| `cy` | Cyprus | 923,381 | ? | 13 | **Answers** on `cystat.gov.cy` (the Greek default; `/en/` 302s). Not chased further. |
| `si` | Slovenia | 1,964,036 | ? | 14 | §11c and §11k both closed it — correctly — on the **2021 register census, which does not ask**. The oracle's row is **2002**, which did. Nobody has looked at the 2002 tables. |
| `al` | Albania | 2,402,113 | 61 bashki | 10 | **`instat.gov.al` did not resolve on 2026-09-06, -07 or -08** — connection layer, not a 403. The only source anywhere that counts **Bektashi** apart from Sunni. Its oracle row is now **2023**, a newer census than §11o priced. Browser job. |
| `fi` | Finland | 5,533,793 | ? | 18 | §11k's register tier, which "fails the same way four times". Left as recorded. |

## Elsewhere, ranked by what they would add

| code | country | people | units | cats | where it stands |
|---|---|---:|---:|---:|---|
| `sz` | Eswatini | 1,093,238 | ? | **20** | §11w: **Zion Christian Church 33.6%** would be `christianity.africaninstituted` as a country's *plurality*, which nothing on this map has. **No reachable host** — `eswatinistats.org.sz` times out, `swazistats.org.sz` does not resolve, and the Wayback CDX has zero archived PDFs for either. Wants a route nobody has found. |
| `sg` | Singapore | 3,459,093 | ? | 10 | Exceptional category depth on a tiny geography. Oracle backlog, never chased. |
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

## Latin America from LAPOP — scouted 2026-09-08, §11ad, none built

The AmericasBarometer grand merge is on disk at `data/raw/lapop/`, 301,156 respondents, and it
carries religion at ADM1 for nine countries this map does not have. **Six of them are the ones
§11x closed for having no census religion question at all**, so this is the only route known.

Its provincial cut was validated against three censuses already on the map (Mexico +0.918
Catholic, Peru +0.776 evangelical, Suriname **+0.981 Hindu** on a district-concentrated
minority). **It is sound for anything above roughly 1% of a province and noise below that.**

**Read §11ad before starting one of these.** The geography gotchas are not guessable: Honduras
is disqualified as it stands, two codes have blank labels, and the Mexico name join needs five
aliases.

| code | country | n pooled | ADM1 | Catholic | Ev+Pent | none+ath | where it stands |
|---|---|---:|---:|---:|---:|---:|---|
| `gt` | Guatemala | 8,919 | 22 of 22 | 52.0% | 34.5% | 5.5% | Cleanest of the nine. The evangelical share is the largest in the file and is what makes it worth drawing. |
| `sv` | El Salvador | 9,063 | 14 of 14 | 46.7% | 29.1% | 12.8% | Clean. |
| `do` | Dominican Republic | 8,904 | 29 of 32 | 53.2% | 20.2% | 18.0% | Three provinces unsampled; five codes have n under 50 pooled. |
| `cr` | Costa Rica | 5,903 | 7 of 7 | 63.1% | 13.8% | 10.6% | Clean, and only seven units, so §3.9b applies rather than a unit floor. |
| `ec` | Ecuador | 7,387 | 23 of 24 | 75.5% | 11.0% | 6.6% | Galápagos never sampled. INEC's own religion module (~2012, unverified) would be the better source if it exists. |
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
| *(empty — none of the nine is built yet)* | | |

## Ready to build, and it needs one decision first

| code | country | people | units | where it stands |
|---|---|---:|---:|---|
| `tr` | Türkiye | 85.7M | **12 İBBS-1 regions** | **Reopened 2026-09-08, §11ac, and §11r's *"Türkiye's own publication of religion is nothing since 1965"* is WRONG.** The **Diyanet İşleri Başkanlığı with TÜİK** published `Türkiye'de Dinî Hayat Araştırması` (Ankara 2014): 21,632 respondents, TÜİK design and fieldwork, and **Table 4 on page 42 is sect by İBBS Düzey 1**, all twelve regions, partitioning to 100. Hanefi / Şafi / Maliki / Hanbeli / Caferi / Diğer / Hiçbiri / Bilmiyorum. **Ortadoğu Anadolu is 48.7% Shafi'i against 45.7% Hanafi**, and no other country on this map enumerates a madhhab at all. State publication, so §14.4 rule 2 is satisfied by construction; open PDF, no licence problem. **The one blocker is a §14.2 judgement, not a sourcing one: the questionnaire has no Alevi box** (Q11 offers the four Sunni schools, Caferi and Nusayri, and the word Alevi appears zero times in 293 pages), so Türkiye would draw with no Alevi dots. Proposed mitigation in §11ac: `Hiçbiri` stays on the parent `islam` and `Bilmiyorum` goes to `unknown`, so ~9% draws as Muslim-with-no-school-stated and the note says why. **Chase KONDA's regional tables first, since KONDA does have the Alevi box.** Nişanyan's *Index Anatolicus* is **closed by Anita's call** (its ToS bans systematic retrieval; she declined to ask) and its coverage ran inverse to the variable anyway.

## Closed, with the reason — do not re-derive these

Kept short; the long version is the `sources.md` section named.

- **Morocco, Algeria, Tunisia, Iraq, Saudi Arabia** — every office reachable, none
  publishes the question (§11r). Egypt likewise collected and withheld: §14 objects.
  **Türkiye moved out of this row on 2026-09-08** — the state side still reads exactly as
  §11r left it, but a non-state source exists; see the table above and §11ac.
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
- **Laos** — 2015 census, national only, and its 31.4% "no religion" is explicitly "no
  religion *or animist*" (§9an).
- **Thailand's 2010 province tables** — the server is gone; DRAWN from an allocation instead
  (§9as).
- **Japan** — the roll counts buildings and catchments, not people (§11q). Deferred, not closed.
- **Kazakhstan's 218 rayons** — available and deliberately refused; the model draws 17 (§9aq).
- **PDH.stat** (SPC's Pacific hub) — 127 dataflows, **no religion**. The offices have it; the
  hub does not (§11aa).
