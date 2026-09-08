# Mongolia — sources

Status: **in progress**, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-mn`, 2026-09-08.

## 1. The question exists, and the office asks it directly

Mongolia's **2020 Population and Housing Census** (Хүн ам, орон сууцны 2020 оны улсын ээлжит
тооллого, enumerated January 2020) carries religion as **question P29, `Та шашин шүтдэг үү?`**
— *do you practise a religion?* — asked of the population **aged 15 and over**. That is read
off the NSO's own DDI for the census, not off a search summary:

    http://web.nso.mn/nada/index.php/catalog/ddi/175      3,647,800 bytes, HTTP 200

whose variable `V98 P29_SP10_RELIGION` has exactly six answer categories, in this order:

| code | label | gloss |
|---|---|---|
| 1 | `Шүтдэггүй` | does not practise |
| 2 | `Будда` | Buddhist |
| 3 | `Христ` | Christian |
| 4 | `Ислам` | Muslim |
| 5 | `Бөө` | Shamanist |
| 6 | `Бусад` | other |

**The oracle is wrong-by-omission here.** `tools/oracle.py mn` reports Mongolia ABSENT from
UNSD's table 28, which per §11r's standing reading proves only that no tabulation was
forwarded to New York. The census asks the question and has done since 2010.

**The frequencies in that DDI are a sample and must not be used.** Study 175 is a public-use
microdata *sample*: V98's six categories sum to 23,221 people and V3 (AIMAG) sums to the same
order, against a census population of 3,296,866. The DDI is evidence for the *category list*
and for the *codes*, and for nothing else. Microdata itself is gated (`get_microdata`), and
per `[[reference_ipums_account]]` that route is not open to this project.

## 2. Geography is solved, and the join is by code

**COD-AB Mongolia** (`cod-ab-mng`, OCHA FISS, refreshed 2026-01-26, CC BY-IGO) — one GET from
HDX, 5,757,562 bytes, no wall:

    https://data.humdata.org/dataset/a9b0a8a6-cb14-448e-b35c-aa5eb51b0557/resource/
      2ec00922-5b9b-47fc-aa36-0a8b59a877df/download/mng_admin_boundaries.shp.zip

It ships `mng_admin1.shp` (**22** aimags and the capital) and `mng_admin2.shp` (**339** soums
and Ulaanbaatar's 9 düüregs), both with English and Cyrillic names.

**The pcodes ARE the Mongolian official aimag codes, so there is no name join at all.** COD's
`adm1_pcode` is `MN` + the two-digit aimag code, and every one of the seventeen aimag codes the
census DDI happens to print (11 Улаанбаатар, 21 Дорнод, 22 Сүхбаатар, 23 Хэнтий, 41 Төв, 42
Говьсүмбэр, 43 Сэлэнгэ, 44 Дорноговь, 45 Дархан-Уул, 46 Өмнөговь, 48 Дундговь, 61 Орхон, 62
Өвөрхангай, 63 Булган, 64 Баянхонгор, 65 Архангай, 67 Хөвсгөл) matches COD's code **and its
Cyrillic name, exactly, with zero mismatches**. `adm2_pcode` is `MN` + aimag + soum, its first
four characters always equal the parent `adm1_pcode`, and all 339 are unique.

This is the one thing that usually goes wrong here
(`[[reference_name_join_wrong_neighbour]]`) and in Mongolia it cannot: the source and the
boundary file share a numeric key.

## 3. Placement: Kontur, and Mongolia needs it more than almost anywhere

    https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/
      kontur_population_MN_20231101.gpkg.gz          10,588,936 bytes, HTTP 200

**121,265 400 m hexagons totalling 3,452,426 people**, against a 2020 census 3,296,866 — a
ratio of 1.047 for a 2023 grid over a 2020 census, which is what Mongolia's growth rate
predicts and is well inside the tolerance `kz_grid.py` uses.

Mongolia is 1.56 million km² with 3.3 million people, a density of **2.1/km²** — a third of
Kazakhstan's, which `kz_grid.py` calls the strongest case for §8.2's grid in the project. Half
the country lives in Ulaanbaatar, and Ömnögovi is 165,000 km² holding about 70,000 people. An
equal-share-per-polygon wash would paint the Gobi in dots of one colour. Kontur is not optional
here.

## 4. What the 1212.mn statistical database does NOT have, and how its API works

The database at `www.1212.mn` is a Next.js SPA over PxWeb `.px` tables. Its internal API,
recovered by grepping the JS bundle (`[[reference_spa_hidden_apis]]`):

    /api/sectorname?lng=en
    /api/subsectorname?subsectorname=<sector>&lng=en
    /api/sectortablename?sector=<s>&subsector=<ss>&lng=en      -> DT_NSO_####_###V#.px ids
    /api/catalogue?list_id=<id>
    /api/download?info=<subsector>&lng=en&type=report|reportSector|updatereports
    POST /api/elastic_search   body {"values": "<term>"}

**`1212.mn` and every `*.nso.mn` host serve a broken TLS chain** — a missing intermediate, the
same failure as `stat.gov.pl` and Ghana's `statsbank`. `curl` needs `-k`, `requests` needs
`verify=False`. A cert error on these hosts is not the host being down. `web.nso.mn` is
HTTP-only and refuses HTTPS outright. `opendata.1212.mn` (the old documented open-data API host)
no longer resolves, and `www2.1212.mn` (the legacy ASP.NET database) times out or returns 500.

**Its religion tables are about institutions, not people**, and are the wrong quantity for this
map: `СҮМ ХИЙДИЙН ТОО` (temples and monasteries by religion type and aimag), `ХУРЛЫН ЛАМ НАРЫН
ТОО` (monks), `ШАШНЫ СУРГУУЛЬ ДАЦАНД СУРАЛЦАГЧДЫН ТОО` (religious-school pupils). Searching the
database for the census wording (`шашин шүтлэг`) returns nothing: the census religion
tabulation is not in the statistical database.

The census landing page `https://www.1212.mn/mn/statistic/fun-statistic/census2020` is **only a
Tableau embed** — workbook `PopulationMongolia2022/Dashboard1` on `tableau.1212.mn`, loaded
through a trusted-authentication ticket minted server-side, so there is no static payload to
read and the workbook is a population dashboard rather than a religion one.

## 5. Ethics (§14): nothing here needs a flag

Mongolia's state publishes religion itself, in its own census, and the one strongly
geographically-concentrated minority — the Kazakh Muslims of Bayan-Ölgii — is a openly counted,
uncontroversial and constitutionally protected population whose concentration is published by
the office and is not secret. COMMANDS.txt's §14 gate ("a country whose state does not publish
religion, or whose religious minorities are persecuted") does not catch this country.

## 6. Open question: which geography the tabulation reaches

See §7 below once resolved.
