# religiondots — source inventory

Claude-managed, alongside `spec.md`. One row per candidate source. **Nothing here has been
downloaded yet.** The `status` column is the only defence against a spec that reads as if it has:

- **confirmed** — checked this session, the claim in the row is verified.
- **likely** — I am confident from prior knowledge but have not checked the current release.
- **to verify** — plausible, needs someone to look.

When a source is actually ingested, its row gains a `sources/<id>.py` and moves to §9 with the
gotchas found while ingesting it. That is where the real value of this file will end up.

**The headline finding, and the reason this project is a lot of work:** there is no global
subnational religion dataset. Searching for one returns Pew, ARDA/WRP and Correlates of War, all
country-level, and gridded work exists for population and GDP but not for religion. The
subnational layer has to be assembled country by country from censuses, and the sect layer from
institutional statistics. That assembly *is* the project.

---

## 1. Global backbones — country level, complete coverage

Used for totals, for the countries with nothing better, and as the check that a country's
assembled subnational figures sum to something sane.

| source | categories | year | basis | access | status |
|---|---|---|---|---|---|
| **Pew, Religious Composition by Country 2010–2020** | 7 (Christian, Muslim, Hindu, Buddhist, Jewish, other, unaffiliated) + a Religious Diversity Index added Feb 2026 | 2010, 2020 | estimate, from 2,700+ censuses and surveys | free download, 201 countries | **confirmed** |
| **World Religion Database** (Brill) | ~18 top categories, and much deeper for Christianity; province level "where available" | 1900–2050 | estimate + roll | **subscription** — needs university access | **confirmed** it is paywalled; subnational depth unverified |
| **ARDA / World Religion Project** | ~30, national + regional + global | 5-yearly to 2010 | estimate | free, thearda.com | likely |
| **Correlates of War World Religion Data v1.1** | as WRP | to 2010 | estimate | free | **confirmed** exists; too old to lead |
| **Joshua Project** | 16,382 people groups × religion, with lat/lon, 238 countries | rolling | estimate | free API with key; bulk download | **confirmed** available. Evangelical missionary source — usable for *presence* of small groups, not for totals. Treat as a lead generator. |

Pew is the backbone. It is only seven categories, so it can never satisfy R2 on its own — it is
the denominator everything else is fitted into.

## 2. Subnational with real category depth — the valuable tier

These are the countries where R2 is actually achievable.

| country | source | geography | categories | year | status |
|---|---|---|---|---|---|
| **United States** | US Religion Census (ASARB) | **county** | **372 religious bodies** (adherents for 217, congregations only for 155) | 2020 | **confirmed** — Excel download at usreligioncensus.org, also on ARDA (RCMSCY20). The single best source on the map. `roll` basis, not self-id. |
| **Philippines** | PSA 2020 CPH | province | detailed — Roman Catholic, Islam, Iglesia ni Cristo, SDA, Aglipay, IFI, Bible Baptist, UCCP, JW, Church of Christ… | 2020 | **confirmed** at national level with province tables referenced; exact province × denomination table to verify |
| **Brazil** | IBGE Censo 2022, Religiões | **município** | top level only so far — Catholic / evangelical / etc. **IBGE has not released the evangelical denominational breakdown**, citing data quality, and is still evaluating whether it can | 2022 | **confirmed** (released 6 June 2025). The §3.4 case: 2010 census for structure, 2022 for totals. 2010 municipal detail **to verify**. |
| **Japan** | 宗教統計調査 / 宗教年鑑, Agency for Cultural Affairs | prefecture × 系統 (Shintō/Buddhist/Christian/other); separate national tables **by umbrella corporation, i.e. by sect** | annual, current | **confirmed** on e-Stat (tstat 000001018471). Prefecture and sect are in *different* tables and joining them is real work. `roll` basis — national total ≈180M vs 125M population; the §3.1 example. |
| **Mexico** | INEGI Censo 2020 | municipio | detailed Protestant/evangelical split, JW, LDS, Adventist | 2020 | likely |
| **Australia** | ABS Census 2021 | SA2 (SA1 via TableBuilder) | ~130 denominations | 2021 | likely |
| **Canada** | StatCan Census 2021 | CSD / CT | detailed | 2021 | likely — the [[project_canadadots]] pipeline already knows how to pull Census Profile |
| **India** | Census 2011, C-1 + appendix | **district** | 6 main + a long "other religions and persuasions" tail (Sarnaism etc.) | 2011 | likely. Pre-2015 and unavoidable — 18% of humanity, and the 2021 census has not happened. §3.4, with no recent total to rescale to. |
| **Indonesia** | BPS Sensus 2020 | kabupaten | 6 official religions + aliran kepercayaan; the official list is itself the granularity ceiling | 2020 | likely |
| **South Korea** | KOSTAT Census 2015 | sigungu | Buddhist, Protestant, Catholic, Won Buddhism, Cheondogyo… | 2015 | likely. Religion was dropped after 2015, so this is the last one. |
| **Kenya, Ghana, Uganda, South Africa** | national censuses 2019 / 2021 / 2024 / 2022 | county / district | detailed denominations in several of them | various | to verify. Africa's coverage is very uneven and **Nigeria does not ask**. |
| **Ireland** | CSO Census 2022 | electoral division | detailed | 2022 | likely |
| **New Zealand** | Stats NZ Census 2023 | SA2 | level-4 religious affiliation classification | 2023 | likely |
| **United Kingdom** | ONS / NRS / NISRA 2021–22 | output area | **top level only** at fine geography; write-in detail (Pagan, Jain, Alevi, Zoroastrian, Rastafarian) at LA level. Scotland has a denominational split, NI more | 2021–22 | likely. The write-in tail is exactly R2's target and it exists at a coarser geography than the main table — a §3.4 case within one country. |
| **Poland** | GUS NSP 2021 + *Wyznania religijne* yearbook | voivodeship; the yearbook lists ~180 registered churches nationally | 2021 | to verify |
| **Russia** | Sreda "Arena" Atlas | federal subject | unusually good: ROC / Orthodox-unaffiliated / Old Believer / Sunni / Shia / Tengrist… | **2012** | likely. Old, and there is no successor. |
| **Nepal, Vietnam, Sri Lanka, Bangladesh, Pakistan** | national censuses 2021 / 2019 / 2012 / 2022 / 2023 | district | moderate — Vietnam has Hòa Hảo and Cao Đài as separate categories | various | to verify |
| **Israel** | CBS | locality | Jewish / Muslim / Christian / Druze; Haredi share is modelled separately | current | to verify |
| **Switzerland, Germany, Austria, Nordics, Netherlands** | structural survey; church membership registers; Zensus 2022 | canton / Kreis / kommune | varies wildly — register countries give exact denominational membership, survey countries give broad categories | various | to verify. Germany's Landeskirche/diocese statistics are `roll`, the census question is `self_id`; §3.1 says pick one. |

## 3. Not asked, or unusable

| where | situation | what we can do |
|---|---|---|
| **China** | no religion question; CFPS/CGSS and the Chinese Spiritual Life Survey are the evidence base; temple and mosque registries exist | modelled tier (§7), one `china.folk` node (§3.3), heavily desaturated |
| **Nigeria** | census does not ask, deliberately, and the Christian/Muslim balance is politically explosive | modelled, and the about panel should say why |
| **France** | the census does not ask, and collecting religion in official statistics is tightly restricted; INSEE's Trajectoires et Origines survey is the usual source | modelled, region level at best |
| **Gulf states, Lebanon, Iraq** | no usable census; Lebanon's last was 1932 | modelled; Lebanon needs a note of its own |
| **United States (federal)** | the Census Bureau is barred from asking religion on a mandatory basis (PL 94-521), so there is no census religion question — which is *why* §2's ASARB study exists | covered better than almost anywhere, on a `roll` basis |

## 4. Locations by religion — the sect and small-group layer

Everything above answers *given this place, which religions*. This section is the inverted shape —
*given this religion, where is it* — and it is where R2 and R3 actually get satisfied. See
`spec.md` §4.4: **these sources feed rings, one ring per group per unit, never a ring per
building.**

Mostly `roll` basis, so by §3.1 they cannot be added to census figures; they are used to **split**
a census category, and to place presence rings.

| source | what it gives | status |
|---|---|---|
| **OpenStreetMap** `amenity=place_of_worship` | global point/polygon coverage with `religion=*` (treated as mandatory on the tag) and `denomination=*` — protestant, roman_catholic, greek_catholic, adventist and a long tail | **confirmed** that the tagging scheme carries denomination. Volume and per-country completeness unchecked. Extractable via Overpass or a Geofabrik/planet pass. The single widest-coverage location source available, and the one most exposed to the mapping-effort bias. |
| **Wikidata** | `P140 religion or worldview` + `P625 coordinate location` over churches, monasteries, synagogues, temples; also religious orders as items | **confirmed** the properties are used this way. Coverage follows Wikipedia's, so it is strong on the notable and blank on the ordinary — which is fine, because notability and smallness correlate here. |
| **Annuario Pontificio** / catholic-hierarchy.org / GCatholic | dioceses, religious institutes, members per order, houses | to verify what is machine-readable. catholic-hierarchy is scrapable; the Annuario itself is a printed book. |
| **GeoNames** | feature codes for church / mosque / temple / monastery, worldwide, with coordinates | likely — but **no denomination**, so it can only place a ring at family level. Weaker than OSM for this purpose. |
| **Orthodox jurisdictions** | monasteries and their communities; Mount Athos as a case of its own | to verify |
| **Buddhist sangha statistics** | Thailand's Sangha Supreme Council, Sri Lanka's nikāyas, Japan's per-sect figures (§2) | to verify, very uneven |
| **Denominational yearbooks and congregation directories** | most large Protestant bodies publish membership, and many publish congregation lists with addresses | per-body, slow, high value in Europe and for the US bodies ASARB has congregations-only |
| **Jewish community statistics** | JPR / Berman Jewish DataBank, by metro | to verify |
| **Small-group registries** | Poland's GUS list, Germany's REMID, national registers of religious associations | good source of *names* for the taxonomy even where the counts are weak |

**The general shape of the small-group problem:** a group of 200 people has no census row anywhere,
and does have a website, a registry entry, an OSM node or an encyclopedia article. So the small end
of this map is built from location lists one entry at a time, and each entry is a presence ring at
a place rather than a population estimate. `spec.md` §4.3 was designed around that being true.

**The bias to hold in mind while ingesting any of these:** they are dense where mapping and
documentation effort has gone, not where religion is. The one-ring-per-group-per-unit rule throws
away the count — the biased part — and keeps the presence, which mostly survives. Do not be tempted
back into density by how good the OSM coverage looks in Germany.

## 5. Geography and population

| | choice | status |
|---|---|---|
| admin boundaries | **geoBoundaries**, CC BY, ADM1 + ADM2 | preferred over GADM on licence |
| population grid | **Kontur Population**, 400m H3, on HDX — vector hexagons out of the box, built from GHSL + HRSL + Microsoft building footprints | **confirmed** available, incl. 3km and 22km versions for low zoom |
| alternative | **GHS-POP** 100m raster (JRC) | **confirmed** available; finer, but raster and needs sampling |
| basemap | OpenFreeMap dark, as cityhistory | confirmed in use elsewhere in this repo |

### United States geography — downloaded 2026-08-27

| file | what |
|---|---|
| `data/geo/cb_2020_us_county_500k.zip` | **the one to use.** 3,234 counties, 2020 vintage; all 3,143 ASARB counties match |
| `data/geo/cb_2020_us_tract_500k.zip` | 2020 tracts, the population weight for placement |
| `data/geo/cb_2024_us_*_500k.zip` | downloaded first, **wrong vintage**, kept only as the evidence for spec.md §8.1 |

Cartographic boundary (`cb_`) rather than TIGER (`tl_`): pre-clipped to the coastline, so dots do
not scatter offshore. Same reasoning as ancestrydots' `download_shapefiles.py`, which is also
where the AREAWATER trick for inland water lives if it turns out to be needed.

**The 2024 files delete Connecticut** — the state replaced counties with planning regions in 2022
and every ASARB Connecticut FIPS fails to join. See spec.md §8.1; the general rule is that
boundaries must be the vintage the data was collected on.

**Tract populations: not needed.** Decided 2026-08-27 — tracts are built to ~4,000 people each, so
allocating a county's dots equally across its tracts is already a population weighting and no
population figure is read at all (spec.md §8.2). County population, where it is wanted for the
residual, is already a column in ASARB's own summaries workbook.

`fetch_tract_pop.py` is kept for the day exact weights are wanted. It needs a free
`CENSUS_API_KEY` — the Census API now rejects unkeyed requests, returning a "Missing Key" HTML
page rather than JSON where it used to serve modest volumes without one. Nothing depends on it.

## 5a. A download that returns HTTP 200 is not a download

Found on INEGI 2026-08-27, and general enough to be worth stating once. The widely circulated URL
for Mexico's Marco Geoestadístico **soft-404s: HTTP 200, `Content-Type` fine, and 2,263 bytes of
HTML where a 257MB zip should be.** Nothing in the status code says so. The real URLs come from an
undocumented endpoint (`/app/api/productos/interna_v2/ficha/datos?upc=…`) whose `multiarchivos`
list is the only place the filename appears.

So every fetch is checked on **size and content**, not status: a zip that does not open, or a file
two orders of magnitude smaller than expected, is a failed download whatever the server said. The
same habit catches Cloudflare interstitials, login walls returned as 200, and truncated transfers.

## 5b. Placement layers, per country (spec §8.2)

How good is "equal dots per fine polygon" as a stand-in for population? Measured per country
against whatever population figure was already free:

| country | placement layer | units | median people | correlation |
|---|---|---|---|---|
| Australia | SA1 | 61,845 | **406** (IQR 359–447) | r = 0.923 |
| United States | census tract | 85,187 | 3,424 (IQR 2,818–4,043) | r = 0.98 |
| Canada | dissemination area | ~57,900 | ~500–700 by design | not yet measured |
| Mexico | AGEB | 81,451 | 1,077 (IQR 529–1,886) | r = 0.787 |
| Ireland | Small Area *is* the count geography | 18,919 | ~330 | n/a — no allocation needed |

Australia is the best case and Mexico the weakest: rural AGEBs have a median area of 93 km², so
dots there are spread over large empty ground. INEGI's rural locality *points* (`00lpr`, 295,779 of
them) are the obvious refinement if Mexico ends up looking wrong.

Ireland is the case that needs nothing at all — its religion data is published at Small Area, which
is already the finest unit, so there is no "inside the unit" problem to solve.

## 5c. Three things about boundary files that are not the join

Found across the geography pass, 2026-08-27. None of these is caught by checking codes match.

- **Do not assume a country has one national grid.** Northern Ireland is on the **Irish Grid
  (EPSG:29902)** while the rest of the UK is on British National Grid (27700). Assuming BNG for NI
  puts everything several hundred kilometres out — and the codes still join perfectly, so nothing
  reports an error. Everything is reprojected to EPSG:4326 with the bounds verified afterwards.
- **"Clipped to coastline" and "definitive" are different files and lose different things.** New
  Zealand's clipped SA2 layer has 2,311 polygons against the definitive 2,395: joining the data to
  the clipped version leaves 84 units homeless, of which 24 are populated (309 people). The
  definitive layer leaves 16, all published with null geometry and holding 24 people between them —
  21 of whom are on an oil rig. Clipped is the right choice for a dot map (dots must not land in
  the sea) but the loss should be counted, not discovered.
- **A population column in a boundary file may be a different universe.** Scotland's OA shapefile
  has a `Popcount` that is the *household* population and runs **108,111 (−2.0%)** short of the
  census table it looks like it should match — spec §3.7's household-versus-total distinction,
  arriving this time inside a geometry file.

The `ltla` trap in England and Wales is now measured too: against current local-authority
boundaries, the 17 districts abolished in April 2023 take **1,686,870 people, 2.83% of E&W**, with
them.

## 6. Licensing notes

Every row needs a licence before it ships, not before it is explored. Known so far: Pew is free
for reuse with attribution; geoBoundaries is CC BY; Kontur on HDX is CC BY; WRD is a paid
subscription and its figures **cannot** be redistributed. Census microdata via IPUMS International
has a use agreement that forbids redistribution of individual records — aggregates are fine, and
aggregates are all this project wants.

## 7. Two systematic biases to keep in view

1. **`roll` sources inflate and `self_id` sources deflate institutional religion**, in opposite
   directions, and by a lot. Japan's 180M-vs-125M is the extreme; European church tax registers
   and Latin American Catholic baptismal counts run the same way. Any figure that arrives already
   summing past the population is a `roll` figure that has met §3.1.
2. **Missionary-sourced datasets are systematically better on small evangelical bodies and
   systematically worse on folk and syncretic practice**, because their reason for existing is to
   enumerate the first. Joshua Project and the WCD both have this shape. Useful, with the tilt
   named.

## 8. Search leads not yet followed

- IPUMS International — census microdata with a religion variable and harmonised ADM1 geography
  for 100+ countries. **Confirmed** that it holds religion and first-level harmonised geography
  with shapefiles; which countries have the religion variable is unchecked and is potentially the
  single biggest coverage win in this file.
- DHS / MICS surveys — religion by survey region for much of Africa and South Asia. Sample-based,
  so wide intervals, but they cover the countries §3 says are dark.
- Afrobarometer, Arab Barometer, Latinobarómetro, WVS/EVS — same shape, useful for the modelled tier.

## 9a. The six acquired 2026-08-27

Acquired in parallel, one agent per country. Each has a `sources/<cc>.py` that rebuilds
`data/normalized/<cc>.csv` from `data/raw/<cc>/`, and a `sources/<cc>.md` with the exact URLs,
re-fetch recipe, full category list and per-source gotchas. **All of `data/` is gitignored** —
252MB of normalized CSV alone — so the `.md` files are the version-controlled record.

Nothing here is mapped to the taxonomy yet: `source_category` is verbatim, per §2.4.

| country | source | finest geography | categories there | rows | basis | reconciliation |
|---|---|---|---|---|---|---|
| **Philippines** | PSA CPH 2020 | province / HUC (117 units, exact partition) | **129** | 17,550 | self_id | exact to the person |
| **Australia** | ABS Census 2021 | SA2 (2,472) | 34 — *150 nationally* | 84,282 | self_id | −111 (−0.0004%) |
| **Mexico** | INEGI Censo 2020 | municipio (2,469) | 4 — *24 at state* | 10,644 | self_id | exact |
| **Canada** | StatCan Census 2021 | CSD (5,161), CT (6,247) | 25 — *168 at province/CMA* | 321,757 | self_id, 25% sample | matches *The Daily* exactly |
| **New Zealand** | Stats NZ Census 2023 | SA2 (2,395) | 11 — *163 nationally, 2018* | 29,975 | self_id | 3 people of 4,993,923 |
| **Ireland** | CSO Census 2022 | Small Area (18,919) | 5 — *25 at county* | 112,495 | self_id | exact at all four levels |
| **UK** | 4 separate systems | Output Area / Data Zone | 8–60 | 1.83M | self_id | +0.0006% to +0.068% |

**The single loudest pattern: every one of the seven splits category depth from spatial depth**
(§3.9). Not one publishes its finest categories at its finest geography. Australia 150 vs 34,
Canada 168 vs 25, New Zealand 163 vs 11, Mexico 24 vs 4, Ireland 25 vs 5. This is now the expected
shape rather than a surprise, and it is the central obstacle to R2.

### Per-country notes worth carrying forward

- **Philippines** — `psa.gov.ph` is behind a Cloudflare interstitial; raw files came byte-identical
  from the Wayback `id_` endpoint (recipe in `ph.md`). **Aglipay and Iglesia Filipina Independiente
  are the same church offered as two answer options** (818,916 + 640,076 = 1.46M) — a taxonomy
  decision, not a data error. Household population excludes 368,300 institutional residents (§3.7).
- **Australia** — ASGS Edition 3; SA2 count went 2,310 → 2,473 since 2016 with ~8% reshaped, so a
  2016 join "would silently succeed and be wrong". `Pentecostal, nfd` is **89% of all Pentecostals**
  — the residual *is* the branch. Yezidi went 63 → 4,123 (refugee resettlement). SA1 is Australia's
  analogue of the US tract trick for placement (§8.2). Attribution must read "Based on Australian
  Bureau of Statistics data".
- **Mexico** — **Orthodox Christians are filed under *otras religiones***, not Christianity: source
  categories sit in different *places* in the tree, not just under different names. INEGI codes 46
  denominations and publishes 24 — Mennonites, Lutherans, Buddhists, Hindus are in the database and
  in no released table. One municipality (La Magdalena Tlaltelulco) is 92% unclassified with no
  flag. geoBoundaries MEX ADM2 is 2012 vintage — see §8.1.
- **Canada** — base-5 random rounding, verified: 0 of 321,757 counts is not a multiple of 5, so
  parent/child disagree by ±5–25 by construction (§3.8). **241 CSDs publish religion on ≥50%
  long-form non-response** — StatCan dropped quality suppression in 2021 (§7). DGUIDs state their
  own vintage, so no §8.1 hazard.
- **New Zealand** — **`-999` is an in-band "Confidential" sentinel in the count column**; summed as
  delivered Islam comes to −36,753 (§3.2). 15.6% of 2023 answers are carried forward or imputed and
  the not-stated residual is pre-filled to zero in all 2,395 SA2s. Multiple affiliation measured at
  +0.18%/+0.70% (§3.3). 2023 fine tables need a free key from `portal.apis.stats.govt.nz` — the
  biggest available upgrade for NZ.
- **Ireland** — CSO publishes **two national totals**, 5,149,139 (SAPS/FY106) and 5,084,879
  ("usually resident and present", used for the headline percentages); mixing them loses 64,260
  people. **Atheist, Agnostic and Lapsed Catholic sit inside *Other religion*, not *No religion***.
  11% of Small Areas changed code since 2016. The religion question is *not* voluntary here;
  not-stated is 6.70% with a 4× county spread.
- **UK** — four statistical systems, kept unmixable by `source_id`. **England & Wales publishes no
  Christian denominations at all**, while carrying 50 minority write-ins (Alevi 25,672, Jain
  24,991, Pagan 73,733, Zoroastrian 4,090, Yazidi 413, and *Church of All Religion* at 24) — more
  granularity on minorities than on the majority, the inverse of the usual pattern. Scotland splits
  Church of Scotland / Roman Catholic / Other Christian and has **Pagan as a tick box**. Scotland's
  perturbation drift is +0.068%, about 100× the ONS figure. The E&W `ltla` level is the pre-2023
  331-district set and will silently drop Cumbria, North Yorkshire and Somerset (§8.1). NI's
  brought-up-in question is a separate source — see spec §3.1.

## 9. Ingested sources

### US Religion Census 2020 (ASARB) — downloaded and characterised 2026-08-27, not yet ingested

Two workbooks, free, no login, direct download:

- `data/raw/2020_USRC_Group_Detail.xlsx` (8.5MB) — sheets: group × nation (374), × state (7,564),
  **× county (80,680)**, × metro (39,837)
- `data/raw/2020_USRC_Summaries.xlsx` (0.5MB) — nation / state / county (3,145) / metro totals

Columns on the county sheet: `FIPS, State Name, County Name, Group Code, Group Name,
Congregations, Adherents, Adherents as % of Total Adherents, Adherents as % of Total Population`.
Group codes are mixed-format strings — `081`, `500`, but also `FD3`, `FHU`, `C51` — so they must be
read as text, and the numeric ones need zero-padding to 3.

**National totals:** population 331,449,281 · congregations 356,642 · adherents 161,224,088 =
**48.6% of population**. The residual is the largest single slice on the US map.

What was found on first read, all of it now in `spec.md`:

| finding | spec |
|---|---|
| 155 bodies report congregations but **no adherent count** — 27,005 congregations, 7.6% of the national total, on the order of 13M people | §4.3 |
| those 155 are **not all small**: UPCI has 4,549 congregations in 1,692 counties, EFCA 1,602 → the congregation-to-adherent conversion is **required**, not optional | §4.3, §4.4 |
| Sikh, Jain, Zoroastrian, Shinto, Tao, Vedanta and Unification bodies are all congregations-only → non-Christian minorities are systematically the unquantified ones | §4.3 |
| 50 bodies report under 1,000 adherents nationally; smallest is Reformed Congregations of North America, **26 adherents, 1 county** | §4.3 |
| the small-body tail is overwhelmingly Anabaptist — dozens of Old Order Mennonite, Amish, Hutterite and Brethren groups in the hundreds, individually enumerated | §4.3 |
| group 267 is named "Muslim Estimate"; 890/891/892/895 are compiler estimates too → **basis is per row, not per source** | §3.1 |
| 30 counties report more adherents than residents — King County TX **452%** — because a roll is attributed to the congregation's county | §3.6 |
| categories mix denominations, whole traditions, building types and one practice (*Hindu Yoga and Meditation*, 437k) | §2.3 |
| "Orthodox" spans six unrelated families in this file alone → no name-based mapping | §2.3 |
| distinct groups per county: median 20, mean 25.7, **max 171 (Los Angeles)** | R1, §4.2 |

**Mapped to the taxonomy 2026-08-27.** All 372 bodies now carry a `path` in
`taxonomy/usrc_groups.csv`; the mapping itself is `taxonomy/usrc2020.py` and the tree it hangs
from is `taxonomy/branches.py` (spec.md §2.4). Adherents roll up to 160,786,973 against ASARB's
161,224,088 — the 437,115 difference is exactly the one held-out category, so the mapping neither
loses nor duplicates anyone.

24 placements are flagged in `REVIEW` with their reasoning. The ones most worth a second opinion:
Messianic Judaism under Christianity; Church of God and Saints of Christ as its own Hebrew
Israelite family; Full Gospel Baptist Church Fellowship under Pentecostal rather than Baptist;
four former-Mennonite bodies (Missionary Church, Bible Fellowship, and the two Fellowship of
Evangelical bodies) kept under Anabaptist on origin rather than present identity. Two need actual
checking rather than a judgement: group **F13 "Church of Christ"** (19 congregations) and **F53
"Church of God"** (204) are bare names that several unrelated bodies use.

**Licence:** the workbooks carry a suggested-citation sheet (Grammich et al., *2020 U.S. Religion
Census*). Terms need reading before anything ships.
