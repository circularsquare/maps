# Burundi: drawn 2026-10-03 at 17 provinces (2008), census totals on an Afrobarometer pattern

**Drawn by `fafd1067-bi` on 2026-10-03** (`sources.md` §bi-2026-10-03), reopened from §11ah's close
by the survey ruling (§scout-2026-10-03-negatives). §0 below is the build; §1-§6 are the 2026-09-08
scouting record, still correct, with §4 and §6 updated.

## 0. The build

| | |
|---|---|
| level | the 2008 census's religion rows by urban and rural (Tableau 1.13, ordinary households, 7,964,078) |
| geography | the 17 provinces of 2008; 2008 census populations by urban and rural (Tableau 1.5) and collective households (Tableau 1.4) |
| pattern | Afrobarometer R5 (Nov-Dec 2012) and R6 (Sep-Oct 2014), 2,395 adults, 17 provinces both rounds |
| fit | three-way IPF, province x urban/rural x religion, to three exact census tables (`sources/bi.py::ipf3`) |
| carried | Catholic (+0.639), Protestant (+0.659), Muslim (+0.467); R5 against R6, nulls +0.42-0.43 |
| tier | every row `modelled` |
| files | `sources/bi.py`, `sources/bi_geo.py`, `taxonomy/bi2008.py`, `countries/bi.py`, node `other.bi` |

**0.1 Did the 2008 census print religion below the nation anywhere? No; checked again 2026-10-03.**
- USCB's HDX geodatabases: 34 datasets (`package_search?fq=organization:us-census-bureau`), no Burundi.
- IHSN catalog 4167 (RGPH 2008): questionnaire and the population répartition only, no microdata.
- **The thematic volume §4 could not open is intact at USAID.** USAID's Burundi mission hosted the
  2011 analysis series, and the Wayback Machine holds it:
  `web.archive.org/web/20170202013857id_/https://www.usaid.gov/sites/default/files/documents/1866/2008%20Burundi%20Population%20Survey_Status%20and%20Structure%20of%20Population.pdf`
  (88 pages, opens cleanly). Its §4.2 Religion is Tableau 4.4, religion by urban/rural and sex only,
  and says *"la différenciation par rapport à la religion se traitera dans les autres rapports
  thématiques"*. The same directory (CDX `usaid.gov/sites/default/files/documents/1866/*`) has
  *Birth and Fertility* (religion against completed fertility, national) and *Marriage and Marital
  Status* (marital status and polygamy by religion, national), both read; *Household and Housing*,
  *Literacy*, *Mortality* (lifetable.de holds its copy) and *Poverty* are listed and were not opened.
  None of the read ones crosses religion with geography.
- PARIS21's `BURUNDI-population-2008.PDF` is an 8-page population summary.

**0.2 Geography.** COD-AB Burundi v02 (July 2026) is the 2025 map (5 provinces, 42 communes) and
HDX keeps no older edition. geoBoundaries gbOpen BDI has the 18 provinces of 2015 and the 119
pre-2025 communes; the 116 outside the Mairie join the census's own commune list one to one inside
their province (four aliases, ten-letter truncation), Rumonge's five go back to Bururi (Burambi,
Buyengero, Rumonge) and Bujumbura Rural (Bugarama, Muhuta), and the Mairie's three are one unit.
R6's commune column puts all 1,199 of its respondents in their REGION province. **The census prints
`MAKAMBA` twice in Makamba province and no Mabanda; the first is Mabanda** (alphabetical order;
Kontur 1.34x and 1.20x that way round, 0.66x and 2.46x the other). Dots are placed on Kontur 2023
calibrated to each commune's 2008 count, hexes cut to the national outline.

**0.3 Why three dimensions, and the seed.** Liberia's and Togo's two-margin fit uses the census's
national rows and unit populations. Burundi's table also has the urban and rural split, and it is
large (Muslims 14.3% of urban Burundi, 1.3% of rural), so it is a third margin. Seeding the carried
rows flat across urban and rural double counts the town: the survey's province shares already
contain it, and the fit put 20.0% Muslims in the Mairie (survey 12.6%, n=175) and 4.9% in every
other town. Splitting each carried province share by the survey's own national urban and rural
multiple (Muslim 3.26x / 0.59x) and letting the fit move it to the census's gives the Mairie 14.4%
and the other towns 14.1%, beside 14.3% for all urban Burundi (asserted, `TOWN_BAND`). The two-margin
fit with no urban dimension gave the Mairie 9.3%.

**0.4 The survey against the census.** Survey over census, national: Catholic 0.91x, Protestant
1.34x, Muslim 1.52x, Adventist 1.03x, None 0.14x, Other 0.30x. `Christian only` is 4.2% of
Christians nationally and 10-16% in Bujumbura Rural, Gitega, Mwaro and Karuzi, with no order that
repeats between rounds (+0.028); it is spread over the two carried Christian rows of its own
province, which leaves their balance untouched (`UNNAMED_RANGE_MAX` raised to 0.17 with the reason).
The survey's Adventists are 15.7% of Cibitoke and fail the split-half, so Cibitoke is drawn at 2.7%.
Muslims are drawn at zero in Muramvya, Mwaro and Rutana, where the survey found none (about 2
expected in each).

**0.5 What is not used.** The 2024 census (RGPHAE) publishes on the five new provinces only and has
no religion table; it is the reopen trigger (§2). The dots are the 8,053,574 people of August 2008.
Rescaling to 2024 would need the 2024 counts on the old provinces, which only the commune tables
could give (new communes merge old ones), and was not attempted.

---

*The scouting record of 2026-09-08 follows, as written then.* It was closed on what the offices
publish, not on reachability. Every host below answered. The 2008 census asked religion of every
household member and published the answer at **national and urban/rural only**; no subnational
religion table exists anywhere in either office's catalogue. `sources.md` §11ah is the long version.

| | |
|---|---|
| census | RGPH 2008 (3rd), 8,053,574 residents |
| question | P14 *"Quelle est la religion de … ?"*, asked of **all household members** |
| answer set | 8 codes: Catholique, Protestante, Musulmane, Adventiste, Témoin de Jéhovah, Traditionnelle, Autre religion, Sans religion |
| published tier | **national, and urban against rural. Nothing finer, in any product.** |
| newer census | RGPHAE 2024, enumerated Aug 2024; socio-demographic results **not yet published** |
| offices | INSBU (`insbu.bi`), BCR (`bcr.bi`) |

## 1. The office moved and its old domain is now an SEO squatter

`isteebu.bi` resolves and returns HTTP 200, and the body is a 5.6 KB page selling SEO services
from a Romanian agency ("We are the Nova Network"). The institute was renamed and is now the
**Institut National de la Statistique du Burundi (INSBU)** at `insbu.bi`. Anything citing
`isteebu.bi` is citing a lost domain, and a scripted check that only looks at the status code
records the office as alive.

**INSBU is a React SPA and its whole catalogue is one keyless GET.** The bundle at
`/static/js/main.*.js` names the API:

```
https://api.insbu.bi/api/publications?page=N        # 496 documents, 50 per page at ps=10
```

Every row carries `document_url` under `https://api.insbu.bi/storage/documents/<ULID>.pdf`, no
key, no referer check. That is the complete list of what INSBU has ever put online, and it was
swept in full. **The RGPH 2008 is not in it at all** — no census volume of any kind.

The same bundle names three platforms in its footer, and each was tested:

* **`insbu.bi/nada/`** — a NADA microdata catalogue, live, correct version, **and empty**.
  `index.php/api/catalog/search?ps=200` returns `found: 0, total: 0`. Installed, never
  populated.
* **`http://41.79.224.140/redbdi/`**, labelled *IMIS Burundi*, a REDATAM information system —
  **unreachable from outside Burundi**. Port 80 times out from three independent network paths
  (curl, a headless-browser reader service, and WebFetch); port 443 returns ECONNREFUSED, so
  **the host is up and port 80 is filtered** rather than the machine being dead. This is the
  one lead here that is blocked rather than answered, and it is the only place a
  religion-by-commune tabulation of the 2008 census could plausibly still be run.
* **`burundi.opendataforafrica.org`** — the AfDB/Knoema portal, **behind a Cloudflare
  interstitial** for scripted clients including its `/api/1.0/meta/dataset` route. A browser
  job. Low expected value: these portals mirror the national yearbook, whose only religion
  table is the national one reproduced in §3.

## 2. The census bureau is a separate agency on a separate domain and a non-standard port

`bcr.bi` 302s to **`https://app.rusansuma.bi:8405/`**, the Bureau Central du Recensement's own
Next.js site for the **RGPHAE 2024** (Recensement Général de la Population, de l'Habitat, de
l'Agriculture et de l'Élevage). Searching for "Burundi statistics office" never reaches it.

What it publishes, all checked:

* **16 publications** — the preliminary-results report plus Volumes I to VIII, which are the
  **community module** and the **agricultural modules** only. There is no population volume.
* **`/donnees/rgph`, 60 statistical tables**, filterable by tier: 29 national, **47 provincial,
  26 communal**. Themes are Population & Démographie, Structure par Âge, Répartition
  Géographique, Ménages & Habitat, Organisations & Communautés, Évolution Historique. **No
  religion theme, and no religion table.**
* **`/donnees/rgph/croisement`**, a cross-tabulation tool — it crosses variables *within* those
  same published tables, so it adds no variable that is not already listed.
* **`/plateforme-dissemination`** — *"Quelque chose d'important arrive bientôt … L'ouverture
  officielle sera annoncée sur nos canaux."* **Not open.**

The 152-page preliminary-results report contains the string "religi" **zero times**. Its
objectives do say the census will establish population structure by *"la nationalité et les
caractéristiques socio-économiques et socioculturelles"*, which is the usual francophone
wrapper for religion, but the questionnaire is not published and nothing confirms it.

**This is the reopen trigger.** If RGPHAE 2024 asked religion, its dissemination platform is
where a provincial or communal table will appear, and Burundi becomes a good country rather
than a closed one. Worth re-checking `app.rusansuma.bi:8405/plateforme-dissemination` and
`/donnees/rgph` every few months.

## 3. What the 2008 census actually published, and where it is

The old ISTEEBU WordPress is gone but the Wayback Machine has its `/rgph-2008/` page and every
file it linked. `data/raw` holds none of this; the URLs are:

```
https://web.archive.org/web/20220327050202id_/https://www.isteebu.bi/wp-content/uploads/2020/10/Chapitre-1.xlsx
   ... Chapitre-2.xlsx, -3.xlsx, -4.xls, -5.xlsx, -6.xlsx, -7.xls, -8.xlsx
https://web.archive.org/web/20211229085013id_/https://www.isteebu.bi/wp-content/uploads/2020/05/R%C3%A9partition-de-la-population-issue-du-RGPH-2008.pdf
```

Eight workbooks, 130-odd sheets, and the same tables again as a 173-page PDF. **Religion
appears in exactly six of them and not one is subnational:**

| table | what it crosses |
|---|---|
| 1.13 | religion × sex × urban/rural |
| 2.13, 2.14, 2.15 | religion × marital status, total / urban / rural |
| 7.8 | households and mean size by religion of head × urban/rural |
| 7.14 | households by size × religion of head |

Chapter 6 has a per-province sheet for *migration*, chapter 8 has per-province sheets for
*disability*, and chapter 1 has per-commune sheets for sex, household type and residence. So
the census's own table programme does cut by province and commune, repeatedly — **it just
never cuts religion that way.** That is a statement about the published programme, not about
what the microdata could do.

**Table 1.13 in full** (ordinary households, 7,964,078 people), which is the drawable content
if anyone ever finds the geography:

| | urban | rural | total |
|---|---:|---:|---:|
| Catholique | 396,334 | 4,545,499 | 4,941,833 |
| Protestante | 180,951 | 1,541,088 | 1,722,039 |
| Aucune religion | 28,486 | 462,612 | 491,098 |
| Autre religion | 14,910 | 246,171 | 261,081 |
| Musulmane | 109,748 | 90,761 | 200,509 |
| Adventiste | 11,012 | 174,349 | 185,361 |
| Témoin de Jéhovah | 5,253 | 20,201 | 25,454 |
| Traditionnelle | 256 | 2,491 | 2,747 |
| ND | 21,263 | 112,693 | 133,956 |
| **total** | **768,213** | **7,195,865** | **7,964,078** |

The *Annuaire Statistique* reprints the same table for the whole resident population
(8,053,574) rather than for ordinary households only; its `ND` is 223,452, exactly 133,956 plus
the 89,496 people outside ordinary households. It also prints a `Sans religion` column of zero
beside `Aucune religion`, which is a spreadsheet artefact: the questionnaire has one such code,
`0. Sans religion`, and `Aucune religion` is it.

**The UNSD oracle's Burundi row is arithmetically wrong and the office's own table is what
shows it.** The oracle gives `Other Religions 494,533`; the true figure is 261,081 + 223,452 =
**484,533**, so its eight categories overshoot its own printed total by exactly 10,000 for both
Burundi Total and Burundi Rural. `oracle.py` reports this as *"does NOT sum to the total"*,
which reads as a real residual category and is a transcription error.

## 4. The thematic analysis volumes, and the one file that is broken at source

ISTEEBU commissioned a thematic series off the 2008 census and put twelve volumes on
`isteebu.bi/images/rapports/`: *état et structure de la population*, *alphabétisation
scolarisation et instruction*, *caractéristiques économiques de la population*, *caractéristiques
ménages et habitations*, *état matrimonial et nuptialité*, *mobilité et migration*, *mortalité*,
*natalité et fécondité*, *personnes âgées*, *personnes avec handicap*, *situation
socioéconomique des enfants*, *évaluation de la qualité des données*. **There is no
socio-cultural volume**, which is where a francophone census usually puts religion.

*Personnes âgées* (59 pp) and *caractéristiques économiques* (95 pp) open cleanly and contain
**no religion tabulation**. **`etat et structure de la population.pdf` cannot be opened**, and
it is the one that would carry religion if any of them does:

* Wayback has **one capture**, 2018-09-18, and it returns
  `x-archive-orig-content-length: 1572822` — **the same size the origin server sent**, so this
  is `[[reference_pdf_truncated_at_source]]`: the file is damaged as ISTEEBU published it, not
  clipped by the archive.
* PyMuPDF and pypdf both report zero pages. A hand-rebuilt xref finds `/Type /Catalog` at
  object 306 but only **177 of 357 objects** are present, so the page tree cannot be closed.
  Inflating every stream by brute force yields image and font data, not text.
* It is mirrored nowhere findable.

So this is the single genuinely untried thing about Burundi. It is a coin flip, it needs a
better PDF surgeon than was available here (no qpdf, mutool or pikepdf on this box), and even
intact it would analyse the same national tables the workbooks carry.

*Settled 2026-10-03 (§0.1): an intact copy is on USAID's site via the Wayback Machine, and its
religion table is national by urban and rural only.*

## 5. What was checked and came back empty

Written out so nobody repeats it.

* **All 496 INSBU publications**, by title and description. The `psep` series —
  *Profil socio-économique de la province X, édition 2019-2020* — is exactly the shape that
  paid off for Rwanda and Botswana, one booklet per unit filed away from the census. **Two were
  opened in full (Bujumbura 212 pp, Bururi 175 pp) and neither has a religion table**; the only
  hits are faith-run clinics and schools. The series also covers only **10 provinces of 18** —
  Bujumbura, Bururi, Cankuzo, Cibitoke, Gitega, Karusi, Kayanza, Kirundo, Makamba, Muyinga —
  so it could not have carried the country anyway.
* **Annuaire Statistique** editions 2015 and 2024, **Tableau de Bord Social** 2009 and 2023.
  The 2015 yearbook reprints the national census religion table (§3) and that is the only
  religion table in any of them.
* **DHS.** Burundi has EDSB-III 2016-2017, EDSB-II 2010 and 1987. Religion is asked, and in the
  252-page *Tableaux des résultats définitifs* it is a **row characteristic** in table 3.1 and
  in the crosstabs, never a column against province. Both *Analyse secondaire* reports
  (INSBU ids 327, 328) mention religion only in prose. The **DHS API is keyless and has no
  religion indicator** for Burundi — `countryIds=BU` returns 3,336 indicators and the only two
  matching "relig" are a bednet source and a violence-help-seeking item. The microdata would
  answer this at province level and needs a DHS account, so it is
  `[[feedback_gated_data_last_resort]]` territory and Anita's call, not a build.
* **World Bank Microdata Library**, `country[]=Burundi`, 62 studies: **no RGPH 2008 of any
  vintage.** `BDI_2019_SES` is a UNHCR refugee survey, not the national EICVMB.
* **HDX** — `Burundi religion` returns zero packages.
* **ArcGIS Online** — `arcgis.com/sharing/rest/search` for Burundi census and commune content
  returns boundary services (esri_dm, geoBoundaries mirrors), student exercises and two UNFPA
  StoryMaps about machine-learning census cartography. **No Burundi statistical organisation
  owns an AGOL org**, unlike Rwanda's NISR, and there is no religion layer at any tier.
* **The 2008 questionnaire** is at
  `https://unstats.un.org/unsd/demographic/sources/census/quest/BDI2008fr.pdf` and it is what
  fixes the answer set in the table above.

## 6. If Burundi is ever wanted before RGPHAE 2024 publishes

*2026-10-03: this urban/rural composition is now one of the three margins of the drawn build (§0.3),
with the Afrobarometer supplying the provincial pattern on top, under the survey ruling of
2026-09-15. As written on 2026-09-08:*

There is a buildable fallback and it is **a model, not counted geography**, which is why it was
not taken.

Chapter 1 table **1.5 gives the urban and rural population of every commune**, and table 1.13
gives the religion profile of urban and of rural Burundi separately. Those two compose into a
per-commune estimate: give every commune's urban residents the national urban profile and its
rural residents the national rural profile. The urban/rural contrast is real and large —
**Muslims are 14.3% of urban Burundi against 1.3% of rural**, Catholics 51.6% against 63.2% —
so the resulting map would not be flat.

It would also be flat in every other respect: two cells, one dimension, and spec §14's
Kazakhstan finding says a layer that is uniform because the model cannot see the variation is
indistinguishable from one that is uniform because the country is. Burundi is 8.05M people and
that is far above the microstate tier where a national table stands on its own (§3.9b), so this
would be a modelled country of real size resting on a single binary. That is a shared-rule call
and it belongs to Anita, not to a build agent; it is written down here so the option is costed
rather than rediscovered.

## 7. Review, 2026-10-03 (`fafd1067-rev10`, full pass)

Checks clean (`check_md`, `built_countries --check`, `check_rollup bi`: 7,830,122 modelled, none
orphaned). Province shares re-computed from `data/normalized/bi.csv` match every note figure
(Bururi 48.6% and Makamba 46.8% Protestant, Gitega 81.9% Catholic and 6.7% Protestant, Mairie 14.8%
Muslim, Muyinga 4.6% the next highest, zero in Muramvya, Mwaro, Rutana); national rows reproduce
Tableau 1.13 to rounding, and the zero cells follow the afrobarometer playbook (§3.5, about 2
expected, well under the 8 that would make the zero the instrument). Mappings follow `rw2022`;
`other.bi` is the routine node. Screenshot: Burundi fully covered, nothing in Lake Tanganyika.

Two note fixes. *"Muslims were 14.3% of the towns"* sat under *"Of the people whose religion was
recorded"*, but 14.3% is of all urban residents including `ND`; of those recorded it is 109,748 /
746,950 = **14.7%** (rural stays 1.3%). And *"The dots are the 8.05 million people counted"*: the
dots are 7,830,122, the 2.8% in `gap` being undrawn; the figure is dropped from that sentence.
`refresh-meta` run. `tools/check_no_religion.py` did not recognise `Aucune religion` as a
no-religion box, so bi was never linted; the pattern is added, and bi passes (`Traditionnelle` is
separate, step 2, `unaffiliated`).

Not raised, for whoever looks next: the census's `Aucune religion` is 6.2% and 94% rural while the
Afrobarometer's None and Atheist are 0.85%, a seventh. Step 2 is applied as written and the REVIEW
entry gives a reason; if Burundi's rural "no religion" is in fact Kiranga practice with no church
(`Traditionnelle` is only 2,747), the box is closer to step 3's lumped case. Nothing found measures it.

## 8. Top text before the 75-word cut, 2026-10-03 (`fafd1067-top75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/bi.py`; `note_public` was not changed.

- `how`: census totals by town and country, 2008, given a provincial pattern by a pooled survey
- `grain`: provinces as of 2008; 474,000 people on average
- `gap`: 2.8%: 1.7% whose religion the census gives as not declared, and 1.1% in collective households (barracks, boarding schools, prisons), whom the religion table leaves out
