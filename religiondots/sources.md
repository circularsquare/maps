# religiondots — source inventory

Claude-managed, alongside `spec.md`. One row per candidate source. The `status` column is the only
defence against a spec that reads as if everything in it had been downloaded:

- **drawn** — wired into `countries.py` and on the map. **Fourteen** as of 2026-09-03: the US
  and Canada, plus Czechia and Brazil (§9b), Australia, Ireland and Mexico (§9c), New Zealand
  and the United Kingdom (§9d), Poland, Romania, Estonia and Croatia (§9e), and **India
  (§9f), which alone is more people than the other thirteen together**.
- **ingested** — there is a `sources/<cc>.py` that rebuilds it and a `sources/<cc>.md` recording
  what went wrong, but it is not yet wired into `countries.py`. **Only the Philippines is left
  here**, and it needs boundaries downloaded before it can be drawn.
- **confirmed** — checked, the claim in the row is verified, but nothing has been downloaded.
- **likely** — confident from prior knowledge, current release not checked.
- **to verify** — plausible, needs someone to look.

When a source is actually ingested, its row gains a `sources/<cc>.py` and moves to §9a/§9b with the
gotchas found while ingesting it. That is where the real value of this file ends up.

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
| **United States** | US Religion Census (ASARB), inside Pew RLS 2023-24 totals | **county** | **372 religious bodies** (adherents for 217, congregations only for 155) | 2020 | **confirmed** — Excel download at usreligioncensus.org, also on ARDA (RCMSCY20). The single best source on the map. `roll` basis; since spec §3.5a it supplies the structure inside a `self_id` total, see §9. |
| **Philippines** | PSA 2020 CPH | province | detailed — Roman Catholic, Islam, Iglesia ni Cristo, SDA, Aglipay, IFI, Bible Baptist, UCCP, JW, Church of Christ… | 2020 | **confirmed** at national level with province tables referenced; exact province × denomination table to verify |
| **Brazil** | IBGE Censo 2022 + 2010, SIDRA 9537 / 2094 | **município** (5,570 / 5,565) | **9** for 2022; **56** for 2010, incl. named Pentecostal and mission-Protestant bodies | 2022 + 2010 | **INGESTED 2026-09-02**, see §9b and `sources/br.md`. The 2010 municipal detail is confirmed real. |
| **Czechia** | ČSÚ Sčítání 2021 (SLDB), open data | **obec** (6,254), and 142 city districts below it | **78**, no suppression, no rounding | 2021 | **INGESTED 2026-09-02**, see §9b and `sources/cz.md`. The best source in the project after ASARB. |
| **Japan** | 宗教統計調査 / 宗教年鑑, Agency for Cultural Affairs | prefecture × 系統 (Shintō/Buddhist/Christian/other); separate national tables **by umbrella corporation, i.e. by sect** | annual, current | **confirmed** on e-Stat (tstat 000001018471). Prefecture and sect are in *different* tables and joining them is real work. `roll` basis — national total ≈180M vs 125M population; the §3.1 example. |
| **Mexico** | INEGI Censo 2020 | municipio (2,469), AGEB for placement | **23** — Protestant/evangelical split, JW, LDS, Adventist; and it separates believers-with-no-affiliation from the irreligious, which most censuses do not | 2020 | **DRAWN 2026-09-03**, see §9c |
| **Australia** | ABS Census 2021 | SA2 (2,423), SA1 for placement | **148** — the deepest list after ASARB. Keeps Eastern Orthodox, Oriental Orthodox and the Church of the East apart, which almost nothing else does | 2021 | **DRAWN 2026-09-03**, see §9c |
| **Canada** | StatCan Census 2021 | CSD / CT | detailed | 2021 | likely — the [[project_canadadots]] pipeline already knows how to pull Census Profile |
| **India** | Census 2011, C-01 + Appendix | **sub-district (5,988)** — one level finer than expected | **8** at sub-district; **83 more** named at state level in the Appendix, nearly all Adivasi — Sarna 4.96M, Gondi 1.03M, Donyi-Polo, Sanamahi, Niam Khasi | 2011 | **DRAWN 2026-09-03**, see §9f and `sources/in.md`. 1.21bn people. The six religions are measured at sub-district; only the 0.66% in `Other religions and persuasions` is derived. No 2021 census exists, so §3.4 has no recent total to rescale to. The Annexure's "sects" are a trap — see §9f. |
| **Indonesia** | BPS Sensus 2020 | kabupaten | 6 official religions + aliran kepercayaan; the official list is itself the granularity ceiling | 2020 | likely |
| **South Korea** | KOSTAT Census 2015 | sigungu | Buddhist, Protestant, Catholic, Won Buddhism, Cheondogyo… | 2015 | likely. Religion was dropped after 2015, so this is the last one. |
| **Kenya, Ghana, Uganda, South Africa** | national censuses 2019 / 2021 / 2024 / 2022 | county / district | detailed denominations in several of them | various | to verify. Africa's coverage is very uneven and **Nigeria does not ask**. |
| **Ireland** | CSO Census 2022 | **Small Area (18,919)** — the finest geography on the map | 24 at county, 5 at Small Area | 2022 | **DRAWN 2026-09-03**, see §9c |
| **New Zealand** | Stats NZ 2023 totals + 2018 structure | SA2 (2,395), SA1 for placement | **159** after allocation — Ratana and Ringatu counted where they were founded; Jedi is 22,605 | 2023/2018 | **DRAWN 2026-09-03**, see §9d |
| **United Kingdom** | ONS 2021 + NRS 2022 + NISRA 2021 | **Output Area / Data Zone (239,023)** — the finest on the map | 56 + 13 + 32. E&W publish **no Christian denomination at all** for 27.5M people; NI names four kinds of Presbyterian | 2021–22 | **DRAWN 2026-09-03**, see §9d |
| **Poland** | GUS NSP 2021, *Przynależność wyznaniowa* | **gmina (2,477)**, powiat, voivodeship | **216 nationally, 139 at gmina** — named churches, and the tail is individual congregations | 2021 | **DRAWN 2026-09-03**, see §9e |
| **Romania** | INS RPL 2021, Tabel 2.4 | **UAT (3,181)** — municipiu / oraş / comună | **23** — the list of state-recognised cults, incl. the Lipovan Old Believers and the Hungarian Unitarians | 2021 | **DRAWN 2026-09-03**, see §9e |
| **Estonia** | Statistics Estonia Rahvaloendus 2021, PxWeb `RL21452` | **municipality (79) + 8 Tallinn linnaosad** | **21**, and 44 nationally — the only census anywhere that names Maausk and Taarausk | 2021 | **DRAWN 2026-09-03**, see §9e. Universe is **15+**, and everything is rounded to base 10. |
| **Croatia** | DZS Popis 2021, `gradovi_opcine.xlsx` sheets 2 and 5 | **555 municipalities + 17 Zagreb gradske četvrti** | **12** drawn; **54 named churches** in sheet 5, not yet ingested | 2021 | **DRAWN 2026-09-03**, see §9e. Sheet 5 is the biggest single upgrade outstanding anywhere. |
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

**"Modelled" is the floor, not the plan — see §12.** Every country in this table has a subnational
workaround of some kind, and China's is good enough to be worth real effort. §12 is the per-country
starting list.

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

- ~~IPUMS International~~ — **answered 2026-09-02, see §10, and followed up 2026-09-03 in §10a.**
  It carries religion for ~63 countries at second-level administrative geography. This was the
  biggest open question in the file and the answer is yes. §10a is the one to read: it says which
  variable to actually extract and which countries are worth it.
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

### Pew Religious Landscape Study 2023-24 — ingested 2026-09-03, `sources/us_pew.md`

The second US source, and the first time any country has two. It is not a rival to ASARB and
does not replace it: spec §3.5a re-bases the United States on self-identification, so this
supplies the **root totals** and ASARB goes on supplying the **structure** inside them. That
is §3.1's permitted split rather than its forbidden addition, and §3.4's "structure from the
detailed source, totals from the recent one" with `basis` where Brazil has `year`.

The number that forced it: ASARB's 161.2M adherents are 48.6% of the country, so **171 million
Americans are currently drawn as nothing at all**, and because a roll's residual means "on no
roll" the map cannot draw the American non-religious while Canada draws 34.6% of itself that
way.

`basis` is `self_id`, n = 36,908 adults, fielded July 2023 – March 2024. Everything about the
fetch, the two traps in it and the known gaps is in `sources/us_pew.md`; the one worth
repeating here is that **Pew's public-use file carries no geography at all**, so the 51
published state pages are the only public route to a state-level number.

The mapping is `taxonomy/us_pew2024.py` and it is shaped unlike every other one in the project.
Because §3.5a takes the survey's totals at the root and nowhere else, it maps **28 of Pew's 149
categories** — a *cut* across their tree at the point where each piece reaches one of our roots,
not a category-by-category match. `southern-baptist-convention` is in the data, is not in the
mapping, and is not supposed to be. `tools/check_pew_mapping.py` proves the cut partitions all
51 states exactly.

One correction against the first write-up, made 2026-09-03: **an absent Pew cell is a true zero,
not a suppression.** The state trees sum to the national tree exactly, category by category, so
nothing is withheld between the levels — Muslim appearing in 38 states of 51 means the weighted
estimate is zero in the other 13, which an n=36,908 survey cut 51 ways will do.

**Drawn 2026-09-04**, by `us_rebase.py`. The United States now draws **326,813,748 of
331,449,281 people, 98.6%**, against 48.4% when ASARB was the whole story; the 1.4% still absent
is the non-response the mapping excludes, as Czechia's 30% and Ireland's 6.7% are. The residual
is 166.2M people — a little over half the American map. It is recorded as a `derived` row under
§7 but **nothing on screen distinguishes it**: the desaturation built for it was removed the same
day, because every colour on the map has to be a colour in the legend. Every ASARB number is untouched: its
372 bodies keep the county figures they always had. See spec §3.5a for the overflow record and
the three things the build settled.

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

---

## 9b. The two acquired 2026-09-02

Same shape as §9a: a `sources/<cc>.py` that rebuilds `data/normalized/<cc>.csv` from
`data/raw/<cc>/`, and a `sources/<cc>.md` carrying the URLs, re-fetch recipe, full category list
and gotchas. `data/` is gitignored, so the `.md` is the record. Nothing is mapped to the taxonomy
yet; `source_category` is verbatim per §2.4.

| country | source | finest geography | categories there | rows | basis | reconciliation |
|---|---|---|---|---|---|---|
| **Czechia** | ČSÚ SLDB 2021 | **obec (6,254)**, + 142 city districts | **78** | 531,196 | self_id | exact, and an exact partition in all 6,724 units |
| **Brazil** | IBGE Censo 2010 (SIDRA 2094) | município (5,565) | **56** | 311,640 | self_id, sample | ±34 by construction, see below |
| **Brazil** | IBGE Censo 2022 (SIDRA 9537) | município (5,570) | 9 | 55,700 | self_id, sample | ±34 by construction |

### Czechia breaks the §9a pattern, and it is the first country to

§9a's "single loudest pattern" was that every country splits category depth from spatial depth
(§3.9). **Czechia does not.** All 78 categories are published at the finest geography, so there is
no `allocate.py` step and every row may become a ring (§3.10) rather than being `derived`.

It is also the only source so far with **no suppression and no rounding**: the smallest published
figure in the country is 1 person, and `cz.py` verifies that categories partition the unit total
in all 6,724 units with zero mismatches. Against Canada's base-5 rounding, New Zealand's `-999`
sentinel and the ONS/NRS perturbation, that is a different régime.

Three things worth carrying forward:

- **A geography level finer than municipality is hiding in the same file.** `uzemi_cis` 44 is 142
  městské části subdividing the statutory cities — Prague is 57 units there against 1 as an obec,
  which is the difference between a usable and an unusable dot map for the capital. It is an
  *alternative* to its parent obec, not a child of it for drawing. Levels 44 and 72 cover only
  the statutory cities, so they do not sum to the national population and look like missing data
  if checked against it.
- **Tradition and institution are separate overlapping rows.** `islám` 5,132 against `Ústředí
  muslimských obcí` 112; `judaismus` 1,427 against `Federace židovských obcí` 474; `katolická
  víra (katolík)` 235,834 against `Církev římskokatolická` 741,019. Summing them into one node
  needs a decision about what the two rows mean — §2.3, arriving as two rows in one column.
- **Jedi is the 13th largest category at 21,023 people in 2,512 municipalities**, ahead of
  Jehovah's Witnesses. Sith 516, pastafarianism 2,696. ČSÚ tabulated them because people wrote
  them in, they are large enough to be visible, and what to do with them is a taxonomy decision
  this file does not make.

And the dominant fact about the Czech map: the question was **voluntary, 30.05% did not answer**,
and the not-stated share runs 11.4%–81.3% between municipalities (median 30.3%, ≥500 people). The
write-in residuals — "křesťanství", "katolická víra", "protestantská víra", "věřící" ×2 — come to
1,359,840, **larger than every named church combined.**

### Brazil is §3.4's worked example, and adds a new reconciliation rule

2010 has 56 categories and is fifteen years stale; 2022 has the totals and 9 categories. IBGE said
on release (6 June 2025) that the 2022 evangelical breakdown is withheld over data quality and may
never appear, so 2010 is not a stopgap — it is the only municipal denominational data Brazil has.

- **The 2022 universe is persons aged 10+**, so it totals 176,600,150 against a population of
  203,080,756. The 26.5M gap is children who were never asked and must not be drawn as a
  not-stated residual. 2010 asked everybody.
- **`Outras religiosidades` means different things in the two years** — an 11,307-person leftover
  in 2010, a 7,079,101-person catch-all in 2022 holding Judaism, Islam, Buddhism, JW, LDS,
  Hinduism and Orthodoxy, all of which have their own 2010 rows. Same label, 626×. Joining the two
  years by category name produces nonsense.
- **Umbanda + Candomblé went 588,797 → 1,849,824**, a 3.1× rise against 6% population growth. The
  largest proportional move between the censuses, and municipal in both years.

**New general rule — a sample tabulation does not sum across geographic levels.** SIDRA's own
national row disagrees with the sum of SIDRA's own municipal rows by up to 34 people, because IBGE
expands the sample independently at each level. Under 1 ppm, and the same species as Canada's
base-5 rounding (§3.8), but a reconciliation written to demand equality fails on every category —
which is what the first Brazil run did. Where a figure is a *universe* count rather than an
expansion, equality still holds exactly and is worth checking exactly: Brazil's 2010 grand total
is exact and is what would catch a genuinely missing município.

Two transport gotchas, both of which cost a run and both of which generalise:

- **SIDRA answers HTTP 400, not a partial result, above ~50,000 values per request.** Minas Gerais
  is the only state big enough to trip it (853 municípios × 66 categories). `br.py` halves the
  category list on refusal rather than hardcoding which states are too big.
- **`servicodados.ibge.gov.br` gzips unconditionally and ignores `Accept-Encoding: identity`**,
  while `apisidra.ibge.gov.br` never compresses. `urllib` decompresses neither, and the symptom is
  a `UnicodeDecodeError` on byte 0x8b that reads like a charset bug. Check the magic number, not
  the header.

### Both are wired and drawn — 2026-09-03

`countries.py` now has `cz` and `br`; boundaries are in `sources/cz_geo.md` and
`sources/br_geo.md`; the taxonomy mappings are `taxonomy/cz2021.py` and `taxonomy/br2010.py`,
both **branch-level like Canada's** (spec §2.4), and `build_tree.py` now validates both — every
target must exist, every category must have a decision, and for Brazil no internal node of IBGE's
tree may be mapped.

| | Czechia | Brazil |
|---|---|---|
| drawn | 7,361,627 of 10,524,167 | 190,513,957 of 190,755,799 |
| what is missing | the 30% who did not answer | 241,944 "don't know" / "no declaration" |
| units | 6,388 (6,246 obce + 142 city districts) | 5,565 municípios |
| nodes | 45 | 31 |
| dots at 1:1,000 | 7,346 | 190,498 |
| rings | 59,625 | 56,914 |
| year | 2021 | **2010** — see below |

Both are ring-heavy against the US, and for the same reason in each case: the count units are
small enough that most (unit, group) pairs fall under the 1,000-person floor. Czech obce have a
median of 435 people, so 88% of present pairs are sub-floor. That is the data being honest rather
than a fault, but it is worth knowing before the ring layer is styled.

**Brazil is drawn at 2010, not 2022.** spec §3.4's rescale — 2022 municipal totals split by 2010
municipal shares — is the intended end state and is not built. Drawing 2022 instead would mean
nine categories with 47.4M evangelicals in one lump, which is the thing 2010 exists to avoid. The
`note_public` says which year it is and why.

**10 new branches for Czechia, 6 for Brazil**, each earned by a countable category (spec §2):
`christianity.hussite`, `christianity.protestant`, `esoteric`, `unchurched`, `parody`,
`rastafari`, `scientology`, `other`/`other.cz`/`other.ca`; then `afrodiasporic` with `.umbanda`
and `.candomble`, `japanesenew`, `spiritualism.kardecist`, `other.br`. Two of them are worth
naming: `unchurched` (believing without belonging — 960,201 Czechs, 9.1% of the country, and
neither `unaffiliated` nor `secular`), and `parody`, because Jedi at 21,023 is too large to drop
silently and too false to file under a religion. `other.ca` also **fixed a live bug**: `ca2021.py`
had mapped to that id since Canada was ingested but the node was never declared, so it was missing
from `religions.json` and the viewer did not know it.

### Three findings from wiring them, in order of how general they are

1. **A source that publishes a complete partition emits its absences as rows, and a zero must
   never become a presence ring.** ČSÚ writes a row for every category in every municipality
   whether anyone is there or not — **417,083 of 494,066 municipal rows, 84% of the file, are
   zeros**. They cost nothing in dots, but a ring asserts presence (§4.3), so left in they turned
   into 277,987 rings against 7,346 dots: a near-solid mask of claims that a body is in a village
   it is not in. Dropping them takes Czechia to 59,625 rings. IBGE does the same thing with its
   `-` sentinel. **ASARB and the other seven list only what they found, so nothing before these
   two had to think about it** — and the symptom is not an error, it is a map that looks busy.
2. **A nested classification published at every level is a triple-count waiting to happen.**
   IBGE's classification 133 is three deep and *all three levels are at município*, so summing the
   file as delivered counts most Brazilians three times — once as `Evangélicas`, once as
   `Evangélicas de origem pentecostal`, once as `Igreja Assembléia de Deus`. Only the leaves may
   be counted, and `br2010.py` derives which those are from the parent column rather than
   hand-listing, so a category IBGE adds later cannot silently double count. Verified exhaustive
   first: on every branch the children sum to the parent within the §9b drift.
3. **"The codes match" is not evidence the vintage is right, and the Brazilian case fails
   silently where the American one fails loudly.** Connecticut's 2022 planning regions delete the
   old FIPS, so a wrong-vintage join errors. Brazil's five post-2010 municípios were *split off*
   existing ones and no old code was retired — so a 2010-data-on-current-mesh join matches all
   5,565, reports nothing, and draws five parents over territory that is no longer theirs. The
   asymmetry in `br_geo.py`'s two-way report is the only evidence: 2010 data has 0 units with no
   polygon, 2022 data has exactly 5.

Plus two smaller ones worth having written down:

- **`geoftp.ibge.gov.br` serves an incomplete TLS chain** — curl and Python both fail with
  "unable to get local issuer certificate", and **certifi does not help**, because the missing
  piece is an intermediate the server should send and does not. Browsers hide it by fetching the
  intermediate themselves. Plain `http://` works and is what `br_geo.py` uses. This looks exactly
  like a local CA problem and is not one. The malhas API is not a way round it: `periodo=2010`
  returns HTTP 500 and it only serves the current mesh.
- **IBGE ships two lakes as municípios.** Lagoa Mirim (4300001) and Lagoa dos Patos (4300002) are
  in the municipal mesh with codes of their own, so the raw merge is 5,567 rather than 5,565.
  Harmless — they carry no census rows — but it makes the count disagree with the expected number
  for no stated reason, which is how a real problem gets dismissed.

### The one that is a gap rather than a finding

**Brazilian placement is the weakest in the project.** Czechia needs no placement layer at all
(median obec 435 people, finer than a US tract) and its 142 city districts fix the only bad case,
Prague's 1.3M in one polygon. Brazil has none of that: dots are spread uniformly across the
município, which in Amazonia means spreading a town across an area the size of England. The fix is
IBGE's **setor censitário**, ~310,000 units designed to a household target — the §8.2 shape — and
it has not been downloaded. §5b's placement table should gain a Brazil row when it is.

---

## 9c. Australia, Ireland and Mexico wired — 2026-09-03

All three were already normalised and allocated, and their boundaries downloaded, on
2026-08-30. What was missing was the same two files Czechia and Brazil needed: a
`taxonomy/<cc><year>.py` mapping and a `countries.py` entry. `build_tree.py` now validates
all three the way it validates Czechia — every target must exist in `branches.py`, and every
category in the **allocated** file must be mapped or explicitly excluded.

| | Australia | Ireland | Mexico |
|---|---|---|---|
| source | ABS Census 2021 | CSO Census 2022 | INEGI Censo 2020 |
| unit | SA2 (2,423) | **Small Area (18,919)** | municipio (2,469) |
| placement | SA1 (61,845) | the Small Area itself | AGEB (81,451) |
| categories | **148** | 24 | 23 |
| nodes drawn | 63 | 19 | 19 |
| drawn | 23,571,427 | 4,803,974 | 125,522,210 |
| dots at 1:1,000 | 23,511 | 4,794 | 125,512 |
| rings | 0 | 0 | 0 |

All three come out with **zero rings**, because under the current rule a ring means "this
religion is in this country and reaches no dot anywhere in it", and every mapped node in all
three reaches at least one dot nationally. Australia additionally reports *12 sub-dot nodes
are derived-only and get no ring*, which is §3.10 working: those categories are below a dot
AND allocated, so nothing may assert they are present.

### What Australia's 148 categories did to the tree

Australia is the deepest list in the project after ASARB's 372, and it forced seven new
branches — more than any source since the US. Four of them are things **no other source has
counted**:

- **`christianity.churchofeast`** — and this is the one that matters. ABS keeps groups 221,
  222 and 223 apart: Oriental Orthodox, Assyrian Apostolic, Eastern Orthodox. Those are
  **three** communions, not two. The Assyrian Church of the East separated in 431, twenty
  years before Chalcedon divided the other two from each other, and it is neither. Almost
  every source folds it into one or the other, which is wrong twice over. 18,932 in
  Australia.
- **`mandaeism`** — 9,182. Australia now holds one of the largest Mandaean communities in
  the world, larger than what is left in Iraq.
- **`yazidism`** — 4,125. Its own religion, not a branch of Islam and not Zoroastrianism,
  though sources file it as both.
- **`caodaism`** — 677 here, and millions in Vietnam, which §11 already wanted for other
  reasons. spec §3.3's case: the syncretism *is* the tradition.

Plus `christianity.united` (the Uniting Church, 673,383, Australia's third largest Christian
body — filing it under Methodist because Methodism was its largest strand would erase the
Presbyterians and Congregationalists equally in it), `christianity.maori` (Rātana, which New
Zealand will need far more than Australia does) and `indigenous.australian`.

**The contrast with Ireland is the point.** CSO publishes one row reading `Orthodox (Greek,
Coptic, Russian)` — Eastern and Oriental welded into a single number that cannot be taken
apart afterwards. Same continent-scale communions, two agencies, one of which preserved the
distinction and one of which destroyed it. Neither is a data error; both are classification
decisions, and §2.3 is about exactly this.

### The nodes added for Czechia and Brazil paid for themselves immediately

`unchurched`, `christianity.protestant`, `esoteric`, `afrodiasporic`, `japanesenew`,
`rastafari` and `scientology` were all created on 2026-09-02 for two countries. All seven are
used by these three, which is the evidence that they were categories rather than one-offs:

- **`unchurched`** is the strongest case. Czechia's *věřící - nehlásící se k žádné církvi*
  (960,201), Mexico's *Sin adscripción religiosa (creyente)* (3,103,464) and Australia's *Own
  Spiritual Beliefs* (27,328) + *Theism* (5,411) are the same answer in three languages.
  **Folding Mexico's into `unaffiliated` would overstate Mexican irreligion by a third** —
  9.5M becomes 12.6M — and INEGI is explicit that they are different people.
- **`christianity.protestant`** absorbs Mexico's *Cristiana* (6.8M) and *Evangélica* (2.4M),
  Ireland's *Protestant* / *Evangelical* / *Born Again Christian*, and Australia's *Other
  Protestant, nfd*. Every one is a Protestant answer that names no body, and there is no
  Protestant super-node for them to hide under (there is deliberately no such node — the
  Protestant families are siblings, not children).

### Two small things worth having written down

- **Ireland's Small Area shapefile has two GUID columns**, `SA_GUID_20` and `SA_GUID__1`, and
  only the second joins. Using the first matches 17,837 of 18,919 and loses 1,082 Small Areas
  holding 286,262 people — a 5.6% shortfall that reports as a warning rather than an error.
  `ie_geo.md` §5 names the right one; this is a note to read it first.
- **Australia loses 34,034 people (0.14%) on purpose.** 34 SA1s carry *empty geometry* — the
  per-state "Migratory - Offshore - Shipping" and "No usual address" pseudo-units, plus
  "Outside Australia" — and they belong to 19 SA2s, 14 of which have no other SA1. These are
  people with no place to be drawn, and dropping them is correct. It is also §8.1's third
  direction working as designed: *codes that match but carry no geometry*, which two-way
  matching misses entirely.

---

## 9d. New Zealand and the United Kingdom wired — 2026-09-03

The last two of the nine normalised sources. Both needed an `allocate.py` run first, and both
pushed on `allocate.py` itself rather than just using it.

| | New Zealand | United Kingdom |
|---|---|---|
| source | Stats NZ 2023 totals, 2018 structure | ONS 2021 + NRS 2022 + NISRA 2021 |
| unit | SA2 (2,395) | **Output Area / Data Zone (239,023)** |
| placement | SA1 (32,817), median 150 people | none — the counts are already there |
| categories | **159** after allocation (was 13) | 56 E&W + 13 Scotland + 32 NI |
| dots at 1:1,000 | 4,634 | 62,959 |
| drawn | 4,660,389 responses | 62,980,011 |

**The UK's 239,023 units are the finest geography on the map by a factor of twelve** — an
England-and-Wales Output Area is about 130 households. Ireland's 18,919 Small Areas held the
record until now.

### The UK is three censuses and the tree can see the difference

`sources/uk.md` opens with "The UK has no census. It has four" and mapping it makes that
concrete. The three that are drawn disagree about what a religion classification is *for*:

- **England and Wales publish no Christian denomination at all, at any geography, for 27.5
  million people.** ONS asks one tick-box question and the write-in box is only reached by
  people who tick "Any other religion" — so all 50 write-in categories are *outside*
  Christianity (Pagan 73,723, Alevi 25,657, Jain 24,992, Ravidassia 9,583, Yazidi, Vodun,
  Thelemite, Eckankar) while `Christian` stays one undifferentiated node. **This is the
  largest single unresolved block on the map** and §3.11's irreducible floor made literal.
- **Scotland** names the Church of Scotland and the Roman Catholics and stops — 13 categories,
  no allocation needed because NRS publishes them at Output Area directly.
- **Northern Ireland**, where the denomination is the political fact, names twenty-two
  Christian bodies including **four kinds of Presbyterian**, and is the only agency on this
  map that publishes `Mixed Catholic / Protestant` as a category. That is a statement about
  Northern Ireland rather than about religion, so it goes to `other.uk` rather than being
  forced onto either side.

The three code namespaces are disjoint — E00/W00, S00, N20 — so `sources/uk_geo.py`
concatenates the three boundary files into one `unit` column without a prefix, and *checks*
that rather than assuming it.

### New Zealand does §3.4 and §3.9 at once, and counts responses rather than people

Stats NZ published 166 level-3 categories nationally in **2018** and 13 level-1 columns by SA2
in **2023**, so the fine categories are five-year-old shares on current totals — spec §3.4 —
*and* a category/geography split — §3.9 — in the same source. No other country does both.

Two consequences worth carrying:

- **A New Zealand dot is a response, not a person.** The census takes up to four religions
  per person, so the categories sum to 5,003,112 against an SA2 population of 4,993,920.
  About 9,000 people are drawn twice. 0.19%, and it is the only country on the map where a
  dot is not a person.
- **Stats NZ tabulates political ideologies as religions** — Socialism, Marxism, Maoism,
  Libertarianism, 47 people between them — because respondents wrote them in. Not `secular`,
  which is for stated non-theistic *positions*; they go to `other.nz`.

Also: **Jedi is 22,605 in New Zealand**, second only to Sikhism among its "other religions"
and larger than Bahá'í, Jain, Taoist and Zoroastrian combined. Church of the Flying Spaghetti
Monster is 4,705. Czechia's `parody` node has now been earned three times over.

And Rātana (45,582) and Ringatū (12,832) — the churches founded by Māori prophets — are
counted here where they were founded, which is why `christianity.maori` existed before New
Zealand arrived: Australia's 3,246 Rātana created it a day earlier.

### `allocate.py` grew three flags, each for a real reason

- **`--tolerance`**, because a source whose structure and totals are different YEARS can miss
  its own column by however much the group grew. NZ's Hindu column is −14.7% and its Muslim
  column −18.2% against 2023 totals; both are immigration, both were rejected at the default
  10% band, and both are correct. The flag says in its help text that it is not a way to force
  a broken mapping through.
- **`--source-id`**, because `uk.csv` holds England-and-Wales AND Scotland at `geo_level ==
  'output_area'`. Allocating without it silently mixes two countries — no error, no warning,
  just a wrong answer.
- **`--out`**, because one source file now produces two allocations.

### Five new branches

`indigenous.maori` (traditional Māori religion, as against the prophetic churches),
`other.nz`, `other.uk`, and two from the England-and-Wales write-in tail:

- **`alevism`** — 25,657. Usually filed as a branch of Shi'a Islam, a placement many Alevis
  reject; given its own family for the same reason as Druze.
- **`ravidassia`** — 9,583. Declared itself separate from Sikhism in 2010, so the boundary is
  live rather than settled.

### One correction

`sources/uk.md` carried a warning that `data/normalized/` was not gitignored and a `git add
-A` would try to commit a 162 MB `uk.csv`. **That was wrong.** `religiondots/.gitignore`
ignores `data/` — the whole directory, one line, with a comment saying it is deliberately the
whole directory for exactly that reason. `git check-ignore -v` confirms it. The warning has
been replaced with the correction rather than deleted, since someone who read it once may go
looking for the fix.

---

## 9e. Poland, Romania, Estonia and Croatia wired — 2026-09-03

All four from §11's Central and Eastern European cluster, all drawn the same day, and no
two of them fail in the same way: Poland has the deepest category list in Europe on a
middling geography, Romania has one of the finest geographies on the map with a category
list fixed by statute, Estonia is tiny, rounded, and enumerates things nobody else does,
and Croatia is drawn shallow while holding a much deeper table in reserve.

| country | source | finest geography | categories there | drawn | basis | reconciliation |
|---|---|---|---|---|---|---|
| **Poland** | GUS NSP 2021 | **gmina (2,477)** | **139** — *216 nationally* | 30,138 dots | self_id, voluntary | exact at all four levels |
| **Romania** | INS RPL 2021 | **UAT (3,181)** | **23** | 16,369 dots | self_id + registers | exact at all three levels |
| **Estonia** | Stat. Estonia 2021 | **79 municipalities + 8 Tallinn districts** | **21** — *44 nationally* | 958 dots | self_id, **15+ only** | base-10 rounded; drift within bounds |
| **Croatia** | DZS Popis 2021 | **555 municipalities + 17 Zagreb districts** | **12** — *54 churches in an uningested sheet* | 3,716 dots | self_id | exact at every level |

Details in `sources/pl.md`, `sources/pl_geo.md`, `sources/ro.md`, `sources/ro_geo.md`,
`sources/ee.md`, `sources/ee_geo.md`, `sources/hr.md`, `sources/hr_geo.md`.

### Croatia is the first source deliberately drawn below its own ceiling

DZS publishes two religion tables at the same geography. Sheet 2 is a 12-category
partition; **sheet 5 names 54 individual churches** — four Orthodox jurisdictions kept
apart, eleven separate Jewish communities by city (for a national Jewish population of
573 people), two Old Catholic churches, a dozen Pentecostal and Baptist bodies.

Sheet 5 is not ingested, and the reason is worth recording because it will recur: **it
refines two of sheet 2's residual categories rather than replacing the partition.** Its
largest column is `Katolička crkva` at 157,388 — people whose *religion* answer was "other
Christian" but who named the Catholic Church as their *community*. Those are disjoint from
sheet 2's 3,057,735 `Katolici`; adding them would be wrong, and drawing them as a separate
node would be baffling. That is a design decision rather than a parsing one, so it was
deferred rather than guessed. `sources/hr.md` §4 has the full shape.

**Croatia is therefore the only country on the map whose drawn depth understates its
source**, and any comparison of category counts across countries has to say so.

### Poland is the second country that does not make the §3.9 trade

Czechia was the first source that publishes its finest categories at its finest geography.
Poland is the second, and it is more interesting because it *nearly* makes the trade and
then doesn't: 216 categories nationally, 139 at gmina, and the 77 that never reach gmina
are **1,648 people in total**. Measured, not assumed:

| level | categories | share of the affiliated they reach |
|---|---|---|
| gmina | 139 | 99.787% |
| powiat | 198 | 99.970% |
| voivodeship | 204 | 99.997% |
| country | 216 | 100% |

So `allocate.py` is not run for Poland — it would move a rounding error. Two of eleven
countries now need no allocation step, and both are Central European.

### The Polish tail is congregations, which no other source reaches

About 60 of the 216 categories are **single congregations** that registered as religious
associations: `Zbór Ewangeliczny "Betel" w Warszawie` (2 people), `Kościół Jezusa Chrystusa
w Werbkowicach` (5), `Warsaw International Church` (24). Everything else in the project
stops at the denomination.

This breaks a comparison the spec has been making implicitly. **"148 categories" for
Australia and "216 categories" for Poland are not the same measurement** — Australia's are
denominations and Poland's bottom out at individual churches. Any future summary that
ranks sources by category count has to say so.

### Romania's ceiling is a statute, not a question design

INS publishes the **list of state-recognised cults** and sweeps everything else into
`Alta religie`. So Romania's 23 is not a tabulation choice that could be revisited, it is a
legal list: there is no Mormon, Buddhist or Hindu category because those bodies are not
recognised cults. Every other source's granularity is limited by what respondents wrote and
what the agency chose to publish; this is a fourth kind of ceiling and §2.3 should carry it.

It buys three things nothing else has: the **Lipovan Old Believers** (28,362 — the largest
such population any census in the world publishes), the **Hungarian Unitarians of
Transylvania** (47,992, continuous since 1568), and the Saxon and Hungarian Lutheran
churches counted apart.

### Three findings that generalise

1. **A unit-count check can pass while the join fails completely.** GISCO gives Poland a
   13-digit `LAU_ID` and GUS a 7-digit TERYT; both sides have exactly 2,477 units and
   joining them as delivered matches **zero**. The id embeds TERYT at fixed offsets and
   drops the gmina *type* digit. §5c already says the unit count is not the join; this is
   the sharpest example so far, because the reassuring number was exactly right.

   The **names** are what proved the derived key was correct rather than lucky: 2,476 of
   2,477 agree, and the single disagreement (`gm. w. Nowiny` vs `Sitkówka-Nowiny`) is a
   real 2021 rename. A wrong offset rule cannot produce that.

2. **A source can suppress and still reconcile.** INS writes `*` for a confidential cell
   and `-` for a true zero **in the numeric columns**, which is New Zealand's `-999` in a
   different costume. Read with `errors="coerce"` the suppressed cells become NaN and the
   people vanish silently. Because the *totals* are not suppressed, the cost is measurable
   exactly: 16,493 people, 0.087% of Romania. Suppression is not a reason to reject a
   source; unmeasured suppression would be.

3. **The obvious parse rule was wrong in a way that only just got caught.** Romanian county
   headers and communes are both bare all-caps names. "First row bearing a county's name is
   the header" misfiled 6 rows and double-counted 600,861 people, because `CĂLĂRAȘI` and
   `SATU MARE` are commune names too and the communes sort first. It was caught only
   because two different counties' `Păuleşti` collided into one key and tripped the
   categories-exceed-total check. **Had they not collided, the run would have passed.**
   The rule that works keys on the county's own total from the higher-level sheet.

### Two new tools and three new branches

`tools/check_mapping.py <cc>` answers the three questions a taxonomy mapping gets wrong
quietly: does every source category resolve or is it explicitly EXCLUDED, does every node
it points at exist in `branches.py`, and where do the people land. `build_tree.py` has
always validated `usrc2020.py` this way and no other country's mapping had anything.

Branches added: `christianity.orthodox.oldbeliever` (Poland named two Old Believer
churches, Romania named the Lipovans the next hour — spec §R2 lists Old Believers as a
test case), `christianity.biblestudent` (GUS's `nurt badaczy Pisma Świętego` holds four
bodies, one of which is the Witnesses and three of which are not), plus `other.pl` and
`other.ro`.

### Estonia is worth its size for three things, and the capital nearly sank it

1.6% of the drawn population, and it earns its place anyway: it names **Maausk and
Taarausk** (`Earth Believer` 3,860, `Taara Believer` 1,770), which **no other census on
earth enumerates**; it names **Old Believers** for the third time in a day; and at 58.4%
reporting no affiliation it anchors the irreligious end of the map, which had nothing past
Czechia's 47%.

Two caveats that make an Estonian dot mean something different from every other dot:
**the universe is persons aged 15+** (§3.7 — no Estonian child is drawn) and **every figure
is rounded to base 10** (§3.8 — Canada's problem one size larger, so nothing reconciles
exactly and nothing is supposed to).

**Tallinn is 33.05% of Estonia in one 159 km² polygon** — three times Bucharest, seven
times Warsaw, worse than Prague. Statistics Estonia publishes the 8 linnaosad, so it was
fixable. That gives a clean four-way comparison now that the region is well covered:

| capital | share in one unit | split published? |
|---|---|---|
| Tallinn | 33.1% | **yes**, 8 linnaosad |
| Prague | 12.4% | **yes**, 57 city districts |
| Bucharest | 9.8% | no |
| Warsaw | 4.7% | no |

Central and Eastern European capitals are single municipalities, and whether the map can do
anything about it depends entirely on whether the office publishes below that level. Two of
four do; the other two stand as one polygon because subdividing would invent structure
(§3.10).

### Estonia was the easiest acquisition in the project, and the hardest to download

The data came from a **standard PxWeb API** — a POST with `"filter":"all"` and json-stat2
back, no scraping, no login, no bot protection. That is the first time in the project. The
lesson for §11's remaining leads is to look for `/api/v1/<lang>/<db>/` before anything else;
it took three minutes against three days of hunting for Slovakia.

The BOUNDARIES were the opposite. `linnaosa_shp.zip` looks like the obvious download and
returns **HTTP 200 with a 282-byte PNG** — a picture of an error message. Not a 404, not an
error status, not even text. §5a says a download that returns 200 is not a download, and
this is its most literal instance yet: the only defence is checking the bytes are a zip.

Also, four Estonian **EHAK codes were retired between the census and the 2024 boundary
release** (Antsla, Narva-Jõesuu, Sillamäe, Valga). The old codes appear nowhere in the
current file, so this is a genuine §8.1 vintage difference rather than a level mix-up; they
are re-joined by name where exactly one candidate remains on each side, derived rather than
hard-coded so the next release cannot make a frozen alias list stale in silence.

### The GISCO LAU file is the boundary answer for most of Europe

One 98MB download — Eurostat GISCO LAU 2021 — carries 98,188 municipal polygons across 34
countries, and the companion **LAU–NUTS correspondence workbook** carries `NUTS3 | LAU CODE
| LAU NAME NATIONAL` for each. Together they solved both countries' geography, and they
already hold Slovakia (2,928 obce) and Hungary (3,156) whose counts are the only thing
still missing. Both are preferred to geoBoundaries here, whose POL ADM3 is **2017 vintage**
— the §8.1 hazard exactly.

**The `LAU_ID` format is per-country and Poland's rule is not general.** Romania's `LAU_ID`
*is* the SIRUTA code with no decoding at all. Each new country needs its own check.

### Romanian `ş` is not Romanian `ș`

`ş` U+015F (cedilla) and `ș` U+0219 (comma-below) are different codepoints, as are `ţ` and
`ț`. INS writes comma-below, Eurostat writes cedilla, **for the same place names**. Without
folding both to ASCII about a third of Romanian names miss, and the failure looks exactly
like a boundary-vintage mismatch. Worth expecting anywhere in the region.

---

## 9f. India wired — 2026-09-03. Bigger than everything before it put together

**1,210,854,977 people on 5,988 sub-districts**, against 730M for the thirteen countries
already drawn. Full write-up in `sources/in.md` and `sources/in_geo.md`; this section is
what generalises.

### The source went one level finer than the file said it would

`sources.md` had India down as **district** level, on the strength of the C-01
documentation. C-01 is actually published at **India / state / district / sub-district /
town**, so the map got 5,988 units instead of 640 — a 9× improvement discovered by opening
the file rather than by reading about it. §12 already says *"look at a sample of the actual
table"*; this is the first time that instruction paid nine to one.

The mirror-image correction is that the **Appendix went one level coarser** than the
catalogue claims. Its NADA description says the tail is published "at India, state and
District levels"; the published file has `Distt. Code == 000` throughout and is
state-level only. `in.py` asserts that rather than trusting either statement, so a reissue
that adds district detail fails loudly instead of being ignored.

### A table can be a perfect partition and still be unusable — the Annexure

**The most important finding here, and the one most likely to be met again.** The C-01
Annexure is arithmetically flawless: for every state and every religion, `Religion:X` =
unspecified remainder + named sects, to within a few hundred people nationally. Every
structural test `allocate.py` applies would pass it.

It names **573 Shia Muslims** among 172.2M, and **8,399 Catholics** among 27.8M Christians.

What it counts is people who wrote a *sect* where the form asked for a *religion* — a
measure of insistence, not of membership — and every figure in it undercounts the real
community by one to three orders of magnitude. Drawing it would have put figures on the map
wrong by 100× in a direction no confidence marking can express.

**The general rule this earns:** *arithmetic consistency is not evidence of meaning.*
`allocate.py`'s reconciliation checks that a mapping is structurally right and cannot check
that a category means what its label says. The only thing that caught this was reading five
numbers and knowing roughly how many Catholics India has. **Every allocation source needs
one sanity check from outside the data**, and it should be a number a person already knows.

The trap is specifically that the *large* entries look fine: Lingayat is 2.66M and 99%
Karnataka, which would have drawn beautifully and been wrong by a factor of four.

### Allocation had to become per-coarse-unit, and it converts guesses into measurements

`allocate.py` pooled every coarse unit into one national composition. That is right for
Australia (`--coarse nation`) and harmless for Canada. It would have destroyed India:
Sanamahi is 100% Manipur, Niam Khasi 100% Meghalaya, Donyi-Polo 98% Arunachal Pradesh, and
pooling would have put Manipuri and Arunachali religions into every sub-district in the
country in proportion to its `Other` count — §3.10a's "Yezidi and Paganism come out with an
identical ranking" failure, at national scale.

**`--within N`** allocates inside each coarse unit, matching a fine unit to the coarse unit
whose `geo_id` is its first N characters. Each religion then reproduces its published state
distribution exactly.

The unexpected benefit is worth more than the fix: **the single-child test becomes
per (coarse unit, column), so 245 of India's (state, column) pairs have exactly one named
religion and are `measured` rather than `derived`.** Splitting the coarse geography does not
merely make better estimates, it converts estimates into measurements wherever a state has
only one answer. Any source with a many-unit coarse table should use it.

India's derived share ends up at **0.66%** — the 7.94M in `Other religions and persuasions`
— against six religions measured on all 5,988 units. That is by far the best measured/derived
ratio of any allocated country.

### The other four things that generalise

- **The same category, spelled two ways in two tables of one census.** C-01 writes `Other
  religions and persuasions`, the Appendix writes `Other Religions and Persuasions`. The
  parent went unrecognised and every state's bucket total was added as a named religion —
  15.7M against a 7.9M bucket. **Match a parent on its code, never on its label**, wherever
  the source gives codes.
- **Excel type inference differs between two files of the same release.** State codes are
  text (`"00"`) in the C-01 files and numbers (`0`) in the Appendix, so `str(cell)` gives
  different keys for the same code. India's own row became a 36th state. Normalise every
  code through one zero-padding helper at the point of reading.
- **A nested geography can hide a second universe.** India's town rows are urban-only
  subsets of their sub-district, so summing the file as delivered counts urban India twice.
  They happen to carry only `Urban` and never `Total`, which makes the obvious filter work
  by luck; `in.py` asserts the property instead of relying on it. This is §12's "universe
  rows are not categories" in geographic rather than categorical form, and it is easier to
  miss.
- **A source with a publication floor needs its remainder emitted as a category.** The
  Appendix lists a religion only at 100+ adherents nationally, so 1.9% of the bucket is
  unnamed. Without an explicit row for it the shares normalise over the named religions and
  inflate every one of them by ~2%.

  **And then that new category has to be mapped, which is where §12's warning about
  `check_mapping.py` earned itself.** The remainder was emitted correctly, resolved to
  nothing in `in2011.py`, and `countries.py` dropped it in silence — 149,668 people gone,
  with every reconciliation in `in.py` and `allocate.py` still passing, because both of them
  are upstream of the taxonomy. Nothing in the build failed. What caught it was
  `tools/check_mapping.py in` reporting one unmapped category, and the arithmetic then being
  unambiguous: 1,207,838,006 reached `countries.py` where 1,207,987,674 should have, and the
  gap was exactly the remainder. **Run it after adding a category, not only after writing
  the mapping.**

### And one that is about boundaries

**17.4M people were in units with no polygon, and the census said what they were made of.**
India's `Area not under any Sub-district` (code `99999`) is the big municipal corporations —
Kolkata, Haora, Asansol, Agartala, BBMP. The sub-district polygons tile each district
completely, so there was no leftover geometry to give them, and spreading them over the
district would have smeared the Kolkata metropolitan fringe across rural Bengal.

The repair came from the source: **C-01's town rows inside a `99999` unit sum to that unit's
population exactly — 100.0%, in all three states — and all 150 towns have polygons.** So the
unit's shape is the union of its own towns, which is read rather than estimated.

**Look for the sub-level before accepting that a unit has no geography.** A census that
publishes a residual usually publishes its parts somewhere.

### The §8.2 assumption has its first exception

§8.2 places dots by equal share per fine polygon because statistical agencies design fine
units to a population target. **India has no such layer**: its finer geography is 645,828
villages and 4,135 towns, natural settlements from ten people to two million. So
sub-districts are both count and placement layer, and India's median unit holds ~204,000
people — the coarsest on the map, though its median *area* of 551 km² is finer than a
Brazilian município's.

The fix exists and is named in `sources/in_geo.md` §4: `Census_Villages.parquet` carries
645,828 village points with population summing to India's entire rural population.
`scatter.py`'s `place_weight` hook (the US's §8.4 weighter) is the right shape for it, so
what is left is wiring rather than capability — a village-and-town placement layer keyed to
sub-district, and a weighter returning settlement population. India does not use the hook
today only because its placement layer is its count layer, one polygon per unit.

### What it did to the palette

`tools/check_palette.py`, run after the re-tile as §12 requires:

- **India's own view is clean** — 12 families with dots, every pair over 20 dots at least
  ΔE 25 apart. Which is the binding test, since the viewer draws one country at a time.
- **The pooled all-countries view is not, and India is why two of the pairs now show.**
  India raises `jainism` to 4,488 dots and `indigenous` to 7,869, so pairs that were
  previously under the size threshold now clear it: `jainism / afrodiasporic` at **ΔE 9.7**
  is the worst pair anywhere in the project, and `indigenous / japanesenew` at ΔE 12.8 and
  `indigenous / paganism` at ΔE 13.3 follow. These are two of the twenty-one families
  sharing §6.3's indigo→magenta wedge, and they are exactly the "4° apart is the honest
  limit of the idea" case that section admits to.

Nothing is broken by this — no country puts both members of any of those pairs on screen
together — but it is the first time the wedge's crowding has been visible in a real tally
rather than in principle, and it is worth knowing before a 31st root is ever proposed.

### Access notes

- **`censusindia.gov.in` serves an incomplete certificate chain** — identical to
  `stat.gov.pl`. Not a bad URL, not a proxy, not bot protection. Verification off for the
  one host, payload validated structurally instead (§5a).
- **The NADA catalogue is scrapable and the resource ids are not derivable.** Each file's
  download URL carries a per-file numeric id unrelated to the state code, so the catalogue
  page is fetched per file. The catalogue block for C-01 is 11361–11398.
- **SHRUG's own download is form-gated; the `india-geodata` GitHub release is not**, and
  mirrors the same parquets. **Licence is CC-BY-NC-SA — non-commercial**, the first such
  source on the map. See §6.

---

## 10. IPUMS International — the §8 question, answered 2026-09-02

**Confirmed: the RELIGION variable exists for ~70 countries, and GEOLEV2 — second-level harmonised
administrative geography, i.e. ADM2 — exists for 86.** This was flagged in §8 and in `todo.txt` as
"the biggest single coverage question in the file", and the answer is that it covers most of what
§3 calls dark.

Countries with RELIGION, by region:

- **Sub-Saharan Africa** — Benin 1992/2002/2013, Botswana 2001/2011, Burkina Faso 1996/2006,
  Cameroon 2005, Côte d'Ivoire 1988/1998, Ethiopia 1984/1994/2007, Ghana 2000/2010, Guinea
  1983/1996/2014, Kenya 2019, Liberia 2008, Malawi 1998/2008/2018, Mali 2009, Mauritius
  1990/2000/2011, Mozambique 2007/2017, Rwanda 1991/2002/2012, Senegal 1988/2002/2013, Sierra
  Leone 2004/2015, South Africa 1996/2001/2016, Togo 1970/2010, Uganda 1991/2002/2014, Zambia
  2000/2010
- **Asia** — Bangladesh 1991/2001/2011, Cambodia 1998–2019, Indonesia 1971–2010, Iran 2006, Israel
  1972–2008, Laos 1995/2005/2015, Malaysia 1970–2000, Mongolia 2010/2020, Nepal 2001/2011,
  Pakistan 1981/1998, Philippines 1990/2000/2010, Thailand 1970–2000, Vietnam 1999/2019
- **Americas** — Brazil 1960–2010, Canada 1852–2011, Chile 1960–2002, Haiti 1971/1982/2003,
  Jamaica 1982/1991/2001, Mexico 1960–2020, Nicaragua 1995/2005, Paraguay 1962/1992/2002, Peru
  1993/2007/2017, Saint Lucia 1980/1991, Suriname 2004/2012, Trinidad and Tobago 1970–2011,
  Uruguay 2006
- **Europe and elsewhere** — Armenia 2011, Austria 1971–2001, Egypt 1986/1996/2006, Fiji
  1966–2014, Germany 1819/1970/1987, Iceland 1901/1910, Ireland 1901–2016, Netherlands 1960/1971,
  Norway 1865–1910, Papua New Guinea 1980/1990/2000, Portugal 1981–2011, Romania 1992/2002/2011,
  Slovakia 1991/2001/2011, Sweden 1880–1910, Switzerland 1970–2011, United Kingdom 2001

**Two things to check before relying on it:**

- It lists **Nigeria 2010**, which contradicts §3's "the census does not ask, deliberately". That
  is almost certainly the General Household Survey panel rather than a census, and if so it is a
  survey-tier source, not a census one. Worth confirming — it is the difference between Nigeria
  being dark and being modelled from a real sample.
- It lists **India 1983/1987/1993/1999/2004**, which are **NSS rounds, not the census**. India's
  district-level route is still Census 2011 C-01 and is unchanged by this.

**Access and licence:** free but not instant — it needs an account and an approved extract request,
not a download. §6 already records that the use agreement forbids redistributing individual
records while aggregates are fine, and aggregates are all this project wants. The practical shape
is: request an extract per country with RELIGION + GEOLEV2, aggregate to ADM2 locally, keep the
aggregate. IPUMS also publishes harmonised boundary files, which is what makes GEOLEV2 joinable —
worth confirming they cover every sample we would use.

**Caveat that matters for R2:** `RELIGION` is harmonised to ~7 broad categories, which is Pew-depth
and cannot satisfy R2 on its own. `RELIGIOND` is the detailed version that preserves the source
country's own categories, and it is the one to extract. How deep `RELIGIOND` actually goes per
country is unchecked and is the obvious next question.

## 10a. IPUMS follow-up — the four §10 questions answered, 2026-09-03

§10 left four things open: how deep `RELIGIOND` goes, whether Nigeria 2010 is a census, whether
the boundary files cover every sample, and what the access mechanics actually are. All four are
answered, and two of the answers change the plan.

**`RELIGIOND` is shallower than §10 hoped, and it is the wrong variable.** IPUMS's own note on the
variable is that detailed codes are *"restricted to the Christian and 'Other' major groupings"* and
that *"little integration of codes has been attempted after the first digit."* So `RELIGIOND` splits
Christianity — Catholic, Orthodox, Protestant, Anglican, Baptist, Pentecostal, Evangelical — and
leaves Islam, Buddhism, Hinduism and everything else at family level. For a project whose whole
point is sect granularity that is only half the map.

**The depth is in the source variables instead.** IPUMS keeps the *unharmonised* per-sample
variables — the original census codes as the country submitted them, e.g. `AT2001A_RELIGD` for
Austria 2001 — and they are extractable alongside the harmonised ones. IPUMS documents these as
carrying subgroups the harmonised variable throws away (Catholic and Orthodox subgroups in the
Austrian sample is their own example). **So the extract shape is: source religion variable for the
categories, `RELIGION` alongside it as the harmonisation key back to `taxonomy/`.** There is also a
`RELIGION2`, which carries more detail in the pre-1950 samples.

**Geography: take the year-specific variable, not `GEOLEV2`.** `GEOLEV2` is harmonised *across
census years within a country*, so any unit whose boundary moved between censuses is merged with its
neighbours to keep the series consistent. This map is a single-year snapshot per country and pays
that cost for nothing. The year-specific `GEO2_<CC><YYYY>` variables are finer, and IPUMS ships
matching year-specific shapefiles — 140+ countries and territories, 1960–2020, zipped shapefiles,
first through third administrative level. That answers §10's "worth confirming they cover every
sample": the year-specific series is the one to join against, and it is broader than `GEOLEV2`.

**The hard resolution floor is 20,000 people.** Units below 20,000 population are aggregated for
confidentiality, in both the geography variables and the boundary files. IPUMS is therefore an
ADM2-shaped source and can never be an ADM3 one — it will not produce anything like the Czech obec
or the UK Output Area. **Sampling is the second floor:** these are 5–10% samples, so a small group
in a small unit can be absent by chance. Fine for the large categories, useless for the §4 sect
layer, and the reason IPUMS complements the §2 census route rather than replacing it.

**Nigeria 2010 is the General Household Survey, as §10 suspected — confirmed.** All five Nigerian
samples are surveys, not censuses: 2006-07 through 2010-11, sample fractions 0.5–0.7%. The 2010-11
one that carries `RELIGION` is 72,191 persons in 4,851 households. LGA is the smallest geography,
but 72k persons over 774 LGAs is ~93 people each, so it is **usable at state level only** — ~1,950
persons per state. That is still the best subnational religion evidence for Nigeria that exists, and
it moves Nigeria from §3's dark list to a survey-tier ADM1 estimate.

**Access: an application, then an API.** Registration is an individually reviewed application asking
for a description of the proposed research and institutional affiliation, and it expires after a
year. Once approved, every sample is available. The licence forbids redistributing the microdata and
re-identification, and restricts use to *"scholarly and educational purposes… commercial use of the
data is prohibited"* — aggregates and maps are fine, which is all this project wants, but the
non-commercial clause should be read before anything is deployed. **The extract loop is scriptable:**
the IPUMS microdata extract API supports IPUMS International under collection code `ipumsi`, with
`ipumspy` as the Python client and CSV output, rectangular-on-person. Custom sample sizes are not
supported for `ipumsi`. There is also an **SDA online tabulator** that cross-tabs geography ×
religion in the browser and skips microdata entirely — worth trying first for one country, with the
caveats that not every sample is in SDA and large tables hit a cell limit.

**Per-country sample checks, 2026-09-03** — spot-checked against what §11 says is blocked:

| country | sample | fraction | persons | smallest geography | what it changes |
|---|---|---|---|---|---|
| **Indonesia** | 2010 census | **10%** | 22.9M | **regency (kabupaten)** | **The biggest single win available.** §11 says BPS publishes per-regency PDFs and to expect real work; this is the same geography as a clean extract. 270M people. |
| **Kenya** | 2019 census | 10% | 4.72M | **division** (below county) | §2's "to verify" Africa row, answered |
| **South Africa** | 2016 Community Survey | 5.8% | 3.33M | municipality | religion is in 1996 / 2001 / **2016** — the 8.5% 2011 census sample does **not** carry it |
| **Slovakia** | 1991 / 2001 / 2011 census | 10% | ~540k each | **region (8 kraje)** | **IPUMS does not rescue Slovakia.** §11's obec-level problem still needs the ŠÚ SR download; 8 units is not worth wiring. |
| **Nigeria** | 2010-11 GHS | 0.5% | 72k | LGA nominally, state in practice | see above |

**Two counts in §10 need an eyeball before they are quoted.** Read on 2026-09-03, the `RELIGION`
availability list names **63** countries, not "~70", and `GEOLEV2` is given as **76** countries, not
86. The 2026-09-03 read also shows **Guatemala 1964**, which §10 missed, and does **not** show the
Fiji samples §10 lists. Neither discrepancy changes any decision, but the per-country list in §10
should be re-read from the site rather than trusted, before an extract is requested against it.

**Applied 2026-09-03.** Registration submitted at `account.ipums.org/ipumsi/user/new`. Declared as
independent / non-academic, field Geography, with `anita.garden/ancestrydotsna/` as the personal
website, and a research description naming the eighteen countries below plus an explicit undertaking
to publish aggregated proportions only and redistribute no individual-level records: Indonesia 2010,
Kenya 2019, Ethiopia 2007, Ghana 2010, Uganda 2014, Malawi 2018, Zambia 2010, Mozambique 2017,
Senegal 2013, Rwanda 2012, South Africa 2016, Nepal 2011, Bangladesh 2011, Vietnam 2019, Cambodia
2019, Laos 2015, Iran 2006, Pakistan 1998. **Registration expires after a year, so this is also the
renewal record.** If a country outside that list is wanted later, the description is what was
approved against — worth checking whether it needs amending. API key, once approved, lives at
`account.ipums.org/api_keys`.

**The recommendation.** IPUMS is worth doing, and it is the largest coverage jump left in the file —
but as a **second tier below §2**, not as a replacement for it. Take it for the countries where the
national statistical office is blocked, absent, or publishes only PDFs: Indonesia first by a wide
margin, then Kenya, Ethiopia, Ghana, Uganda, Malawi, Zambia, Mozambique, Senegal, Rwanda and the
rest of the Sub-Saharan block, then Nepal, Bangladesh, Vietnam, Laos, Cambodia, Iran and Pakistan.
Do **not** take it for anything already in §9a–§9e: the drawn sources are censuses at finer geography
with deeper categories, and IPUMS would be a downgrade for all thirteen. The one prerequisite is the
account, and it is an application with a review, so **it should be applied for before the extract
work is planned, not during it.**

---

## 11. Leads verified 2026-09-02, not yet ingested

Checked far enough to know the data exists and roughly what shape it is. None downloaded.

| country | source | geography | categories | status |
|---|---|---|---|---|
| ~~**Poland**~~ | GUS NSP 2021 | gmina | 216 / 139 | **DRAWN 2026-09-03 — see §9e.** Was "small file, easy win"; it was. |
| **Slovakia** | ŠÚ SR SODB 2021 | **obec** | detailed; RC 3.04M, Evangelical 287k, Greek Catholic 218k | **still not found, looked again 2026-09-03.** `scitanie.sk` renders its tables client-side and its JS exposes no data endpoint; `/otvorene-udaje` and `/na-stiahnutie` are both 404. `data.statistics.sk/api/v2/collection` is the general VBD catalogue (677 datasets) and contains **no** SODB or religion table. `datacube.statistics.sk/api/v1/...` is not PxWeb and 404s. `data.gov.sk`'s CKAN API returns an SPA shell, not JSON. The remaining routes are the DATAcube UI's own XHR (needs a browser session to observe) and the per-obec PDF/XLSX products. **Boundaries are already solved** — SK is in the GISCO LAU file with 2,928 obce — so only the counts are missing. **IPUMS does not rescue this one** (§10a): its three Slovak samples are 10% of the census but identify only the 8 kraje. |
| **Hungary** | KSH Népszámlálás 2022, census database tables **WBS003** + **WBS008** | **settlement (3,403 units incl. Budapest's 23 districts)** for 11 categories; **NUTS3 county (20)** for 29 | **29** | **ACQUIRED 2026-09-04, not yet ingested — files in `data/raw/hu/`, see §11a.** A clean §3.9 split: the county table's categories aggregate exactly into the settlement table's. |
| ~~Hungary, the earlier note~~ | | | | **The 403 is GONE — re-checked 2026-09-04.** `nepszamlalas2022.ksh.hu` answers 200 to curl with a browser User-Agent; the whole `/en/results/tables` index and its workbooks download without trouble. The 2026-09-03 note called it bot protection and that is no longer the obstacle. **The obstacle is depth:** the only religion table published there, `nsz2022-1.1.7-eng.xlsx` ("1.1.7. Religion"), is **national only** — 9 categories × sex × 1930/1949/2001/2011/2022, no geography whatsoever. Fine geography, if it exists, is behind the **Database** card (`/en/database`), a 1.6MB JavaScript app that builds its exports client-side; no static endpoint was found in the bundle, so it needs a human in a browser. **Whether religion goes below county there is the open question, and it decides whether Hungary is worth ingesting at all** — at county level it is 20 units and Pew-depth. Boundaries are already solved either way: HU is in the GISCO LAU file with 3,156 units. |
| ~~**Romania**~~ | INS RPL 2021 | UAT (3,181) | 23 | **DRAWN 2026-09-03 — see §9e.** The ~15% unknown was real: 13.95%. |
| **Nepal** | NSO/CBS Census 2021 | **district** (77) | ~10 | `censusnepal.cbs.gov.np/results/downloads/census-dataset` |
| **Switzerland** | BFS Strukturerhebung / ESRK | canton, commune | detailed | **it is a sample survey**, ~200k persons/year cumulated over 5 years, not a census — so fine geography is limited and figures carry intervals. Different tier from a census, and §3.1 applies. |
| **South Korea** | KOSIS, Census 2015 | **시군구** (sigungu) | Buddhist / Protestant / Catholic / Won Buddhism / Cheondogyo… | KOSIS has an OpenAPI needing a free key. Religion was dropped after 2015, so this is the last one. |
| **Indonesia** | BPS Sensus 2020 | kabupaten | 6 official religions | exists but BPS publishes largely as per-regency PDFs; expect real work. **Go via IPUMS instead — see §10a:** the 2010 census sample is 10% / 22.9M persons and identifies regency directly. |

**The Central and Eastern European cluster was the best-value block not yet taken, and four
of it are now done** — Czechia, Poland, Romania and Estonia, 69M people between them.
Slovakia and Hungary are what is left of the original five, both with their boundaries
already in hand (§9e). **They are no longer the same kind of stuck, and the row above
corrects it:** Slovakia is still blocked on finding the counts at all, while Hungary's
site is now perfectly reachable and the question has become whether KSH publishes religion
below the national level anywhere. A "blocked on the download" note that goes stale reads
as a dead end long after it has stopped being one, which is the argument for re-checking a
403 before believing it.

Checked 2026-09-03 while looking for more of the same shape:

| country | finding |
|---|---|
| **Estonia** | **DRAWN** — clean PxWeb API, see §9e |
| **Croatia** | **DRAWN** — see §9e. `podaci.dzs.hr` is a shell with no PxWeb; the files are linked from the OLD site, `dzs.gov.hr/naslovna-blokovi/u-fokusu/popis-2021/88` |
| **Lithuania** | `osp.stat.gov.lt`'s REST API returns 403 to scripted clients; `osp-rs.stat.gov.lt/rest_xml/` 404s |
| **Bulgaria** | the NSI religion page carries **no data links at all** — the 2021 results live behind `census2021.bg`, unexamined |
| **Serbia** | `data.stat.gov.rs` answers, but with a ~1MB SPA shell rather than an API |

**The lesson from Estonia is to try `/api/v1/<lang>/<db>/` first.** PxWeb is the Nordic and
Baltic standard and, where it exists, acquisition takes minutes rather than days. Latvia,
Finland, Sweden, Norway and Denmark all run it — though of those only Latvia and Lithuania
plausibly ask about religion at all.

Serbia, Bosnia, Montenegro, North Macedonia, Slovenia, Latvia, Georgia and Armenia remain
unchecked. Slovenia's 2021 census was register-based and **did not ask religion**, so it
should be moved to §3 rather than kept as a lead.

---

## 12. The modelled tier is not actually empty — leads found 2026-09-03

The countries in §3 are dark *officially*. Every one of them has something subnational. This
section is the starting point for whoever takes one of them on.

**Status for the whole section is `to verify`.** None of it has been checked; it is prior
knowledge written down so the search does not start from zero.

Three mechanisms recur, and they are more useful as patterns than as per-country facts:

1. **Ethnicity or citizenship at census geography.** Measured geography, derived religion — §7
   `derived` at best. Carries China's and Russia's Muslim and Buddhist maps almost single-handed,
   and carries the *immigrant* religions of every Western European country below.
2. **Diocesan and denominational registers.** Fine geography, `roll` basis, so §3.1 forbids mixing
   them with anything self-id.
3. **One national survey large enough to break down.** Family-level categories, wide intervals,
   but real subnational structure.

### China — the one worth special effort

1.42B, 17% of the map, and the workaround is unusually good for two of the four big groups.

- **Census ethnicity at county level is the best Muslim and Tibetan Buddhist map in the world, and
  it is the place to start.** Hui, Uyghur, Kazakh, Dongxiang, Salar, Kyrgyz, Tajik, Uzbek, Bonan,
  Tatar ≈ 23M and effectively 100% Muslim; Tibetan, Mongol, Tu, Yugur, Monba, Pumi → Tibetan
  Buddhist; Dai → Theravada. ~35M people at census geography with near-certain religion.
- **CFPS** (Peking University, China Family Panel Studies) — religion module 2012/14/16/18, 25
  provinces, ~40k individuals; the best sample source. **CGSS** is the annual, smaller alternative.
  **CSLS 2007** (Purdue/Horizon, Fenggang Yang, 7,021 respondents, 56 sites) is the classic and is
  old. Christianity's province gradient — Henan, Zhejiang, Anhui, Fujian high — comes from these,
  with wide intervals.
- **The registered-venue registry** (五大宗教, with addresses) is a §4.4 site layer, and it is blind
  to house churches and unregistered folk practice — i.e. blind to most of what is there.
- Han folk religion stays one modelled node (§3.3). No source fixes that.

### The rest

- **Nigeria** — **NDHS 2018**: ~40k households, **state-representative**, religion on the household
  roster. MICS 2021 is the same shape. Both are far better than modelled. The IPUMS "Nigeria 2010"
  flagged in §10 is the LSMS-ISA GHS panel — ~5k households, zone level — so §10's suspicion was
  right and it is a survey-tier source.
- **Russia** — **Sreda Arena 2012**: 56,900 respondents, all 79 federal subjects, separating ROC /
  Orthodox-not-ROC / Old Believer / Sunni / Shia / Tengrist. No successor exists. Pair it with 2021
  census ethnicity at rayon for finer geography (Tatar, Bashkir, Chechen, Avar, Dargin, Kumyk,
  Lezgin → Muslim; Buryat, Kalmyk, Tuvan → Buddhist).
- **Egypt** — **the strongest single item in §10.** Religion is in the census microdata for
  1986/1996/2006 even though CAPMAS never published a tabulation, so an IPUMS RELIGION + GEOLEV2
  extract should give governorate or markaz Christian share and the real Upper Egypt concentration
  (Minya, Asyut, Sohag). The 5%-vs-15% Coptic dispute is not resolvable — publish the microdata
  figure with a note.
- **Turkey** — hardest of the group. **KONDA**'s large surveys give the Alevi share (~4–5%) at
  NUTS-1/2 and nothing finer. Non-Muslims are ~0.2% and Istanbul-concentrated: a rings case, not a
  dots case. Language is a weak proxy — Kurdish splits Sunni Shafi'i / Alevi.
- **France** — **TeO2** (INED/INSEE 2019–20, 27k, religion of respondent *and* parents) at region
  for self-id; **diocesan statistics** at diocese (~95 units ≈ départements) for roll. §3.1 says
  pick one. IFOP polls frequently and occasionally finer.
- **Spain** — **CIS asks religion in every monthly barómetro** (~2,500 each), microdata free, so a
  pooled year is ~30k: enough for autonomous community, possibly province. Family-level only. The
  R2 half is the **Observatorio del Pluralismo Religioso** directory — ~9,000 geocoded non-Catholic
  places of worship tagged by body. That is §4.4 pre-built.
- **Italy** — ISTAT publishes **foreign residents by citizenship at comune level**; crossed with
  origin-country religious composition that is a fine-geography derived layer for ~5M people
  (Romanian→Orthodox, Moroccan/Albanian/Bangladeshi/Pakistani→Muslim, Filipino/Peruvian→Catholic).
  This is what Caritas/Migrantes does, and **it works identically for France, Spain, Belgium, the
  Netherlands and Greece.** Separately, **CESNUR's *Le religioni in Italia*** is a taxonomy source —
  several hundred named groups with locations.
- **Netherlands** — **probably does not belong in §3 at all.** CBS publishes religion from pooled
  survey years down to **gemeente**; KASKI has membership for essentially every Dutch denomination;
  and **SGP vote share per municipality** is a near-perfect proxy for the bevindelijk gereformeerde
  population, available every election. Staphorst, Urk, Rijssen and Barneveld would render right.
- **Belgium** — federal subsidy data gives recognized parishes/communities per religion per commune
  (a site layer); the Catholic Church publishes an annual Mass-attendance count by diocese. Below
  the Flanders / Wallonia / Brussels split, thin.
- **Greece** — genuinely poor, and it barely matters. One national number plus four known
  concentrations covers everywhere that number is wrong: the Muslim minority of Western Thrace
  (Rodopi ≈ half, Xanthi, Evros) and the Catholic pockets of the Cyclades (Syros, Tinos).

**The two claims here I am least sure of**, and the ones to check before building on them: whether
CBS's gemeente-level religion release is current, and whether Egypt's and Turkey's DHS carry a
religion item at all (I believe neither does, which is why both fall back to other routes above).

**And the finding behind all of this:** religion is dropped from censuses precisely where it is
contested — Nigeria 1963, Tanzania 1967, Lebanon 1932, Sudan 2008, Iraq 2024, Egypt collected and
withheld. The map is blindest exactly where religion matters most, and that is not
missing-at-random. §7's confidence rendering is the only defence, and the about panel should say
it outright.
