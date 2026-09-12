# Ireland — CSO Census 2022 religion

`sources/ie.py` → `data/normalized/ie.csv`. **Reconciles exactly at all four geography
levels** (script prints the check and refuses to write if it fails).

`source_id` on every row: **`ie_census_2022`**. `basis` on every row: **`self_id`** —
it is a census self-declaration throughout, no roll or estimate figures anywhere in
the file.

---

## 1. What was taken, and the geography/category trade

Ireland is a clean instance of spec.md §3.4's *structure from one source, totals from
another* — with the unusual luxury that **both sources are the same census, same year,
same universe**, so nothing is interpolated. Only the geography changes.

| geo_level | units | categories | source table |
|---|---|---|---|
| `state` | 1 | **25** (24 religions + `All religions`) | PxStat **FY106** |
| `county` | **31** administrative counties | **25** | PxStat **FY106** |
| `electoral_division` | **3,420** | **5** (4 religions + `Total`) | SAPS 2022 Theme 2 Table 4 |
| `small_area` | **18,919** | **5** | SAPS 2022 Theme 2 Table 4 |

112,495 rows.

**The finest geography published is Small Area, and it carries only four categories.**
Catholic / Other religion / No religion / Not stated. The 24-way split exists no finer
than administrative county. So the fine layer supplies the *totals* and the county layer
supplies the *structure*, exactly the §3.4 shape.

**And there is no older SAPS to borrow structure from.** Checked: `SAP2016T2T4CTY` carries
the *same* four categories (`CA`, `OR`, `NR`, `NS`, `T`), so SAPS 2016 was no finer. Small
Area religion in Ireland has never been published at more than four categories. The county
layer is the only structure available at any vintage.

### The two sources nest exactly

Verified in `ie.py`'s `check()`, and this is the property the §3.4 split depends on:

```
SAPS Catholic        == FY106 "Roman Catholic"                   3,540,412
SAPS No religion     == FY106 "No religion"                        755,455
SAPS Not stated      == FY106 "Not stated"                         345,165
SAPS Other religion  == the other 21 FY106 categories, summed      508,107
                                                       total     5,149,139
```

**Gotcha worth flagging loudly:** FY106's `Atheist` (966), `Agnostic` (2,949) and
`Lapsed (Roman) Catholic` (3,279) fall inside SAPS's **`Other religion`** bucket, *not*
inside `No religion`. So at Small Area, "Other religion" is not a religion category —
it contains about 7,200 explicitly non-religious people. Anyone mapping the 4-category
SAPS split to the taxonomy without the county detail will get this wrong.

---

## 2. Exact URLs and how to re-fetch

All four downloads are open, no login, no bot protection encountered.

```
# SAPS 2022 — Small Area, the finest geography (40 MB, 795 columns, 18,920 rows)
curl -sSL -o data/raw/ie/SAPS_2022_Small_Area_UR_171024.csv \
  https://www.cso.ie/en/media/csoie/census/census2022/SAPS_2022_Small_Area_UR_171024.csv

# SAPS 2022 — Electoral Division
curl -sSL -o data/raw/ie/SAPS_2022_CSOED3270923.csv \
  https://www.cso.ie/en/media/csoie/census/census2022/SAPS_2022_CSOED3270923.csv

# SAPS 2022 — administrative county (kept for cross-checking; ie.py does not read it)
curl -sSL -o data/raw/ie/SAPS_2022_county_270923.csv \
  https://www.cso.ie/en/media/csoie/census/census2022/SAPS_2022_county_270923.csv

# SAPS 2022 glossary — the only place the column codes are defined
curl -sSL -o data/raw/ie/Glossary_Saps_2022_REVISED_21102024.xlsx \
  https://www.cso.ie/en/media/csoie/census/census2022/Glossary_Saps_2022_REVISED_21102024.xlsx

# FY106 — Population by Religion x Administrative County, 24 categories
curl -sSL -o data/raw/ie/FY106.csv \
  "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/FY106/CSV/1.0/en"
curl -sSL -o data/raw/ie/FY106.json \
  "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/FY106/JSON-stat/2.0/en"

# F5051 — same 24 categories x County and City x Sex x 2011/2016/2022.
# NOT used by ie.py (different universe, see §5); kept for the time series.
curl -sSL -o data/raw/ie/F5051.json \
  "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F5051/JSON-stat/2.0/en"
```

The SAPS landing page, if the filenames change (they carry a release date, e.g. the Small
Area file was revised 17 Oct 2024):
<https://www.cso.ie/en/census/census2022/census2022smallareapopulationstatistics/>

**Finding every religion table in PxStat**: `ReadCollection` returns all 12,985 CSO tables
as one 45 MB JSON-stat document; filter on a dimension whose label matches `religio` and
whose `CensusYear` index contains `2022`. That is how FY106 was found — its *label* is
just "Population", so a title search for "religion" misses it entirely.

```
curl -sSL -o collection.json \
  https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadCollection
```

The 2022 tables carrying a Religion dimension are: `FY030 FY031 FY032 FY106 F5008 F5050
F5051 F5070 F5071 F5088 F5134 F5135 F5138 F5141 F6011 F9027 TNLIA05` plus the eight
`SAP2022T2T4*` geography cuts (all 4-category).

---

## 3. Geography level and **vintage** (spec.md §8.1)

The data is on **2022 vintage CSO boundaries** and must be joined to 2022 boundaries.

| level | boundary set | key in `geo_id` |
|---|---|---|
| small area | **CSO Small Areas 2022** | `SA_GUID_2022`-style GUID (SAPS `GUID` column) |
| electoral division | **CSO Electoral Divisions 2022** | GUID |
| administrative county | **Administrative Counties 2022** | GUID |

`geo_id` is CSO's own GUID at every level, because that is the key that is shared between
SAPS and PxStat: FY106's `2ae19629-1433-13a3-e055-000000000001` is Dublin City, and the
same string is the `GUID` of the `DC` row in the SAPS county file. `geo_name` carries the
label, which for Small Areas is the published area code (`017001001`) since Small Areas
have no names.

**The 2016 → 2022 Small Area change is large and would fail silently.** Of the 18,919 Small
Areas in 2022, **2,081 (11.0%) carry a changed code**: a split shows as a suffix
(`017010012/01`, `017010012/02`) and a merge shows as two codes joined
(`017008002/017008001`). Joined against 2016 Small Area boundaries, all 2,081 drop out with
no error — the Connecticut failure of §8.1 in Irish form. Check the join in **both**
directions and report both sides.

**Small Areas are population-equalised by design, which is spec.md §8.2's free lunch
again.** Measured on this file: min 37, median **259**, max 2,777, and **no Small Area has
zero population**. They are built to hold 50–200 dwellings, so scattering a unit's dots
uniformly *within* a Small Area needs no population grid at all — the geometry already
carries the weight, exactly as US census tracts do. At a median of 259 people a Small Area
is well under one dot at 1:1,000, so Ireland is essentially a per-Small-Area placement
problem, not a within-unit one.

Boundary files are published by Tailte Éireann / CSO on data.gov.ie and the Tailte Éireann
ArcGIS Hub ("CSO Small Areas National Statistical Boundaries 2022", "CSO Electoral
Divisions 2022"). **Not downloaded in this pass** — `data/geo/` is shared and another
agent owns it. Take the *ungeneralised* set if dot placement matters, and note the
Small Areas are already clipped to the coastline.

---

## 4. Full category list with national counts

FY106, Census 2022, `All religions` = 5,149,139.

| category (verbatim) | count | % |
|---|---:|---:|
| Roman Catholic | 3,540,412 | 68.76 |
| No religion | 755,455 | 14.67 |
| Not stated | 345,165 | 6.70 |
| Church of Ireland, England, Anglican, Episcopalian | 126,658 | 2.46 |
| Orthodox (Greek, Coptic, Russian) | 105,827 | 2.06 |
| Islam | 83,272 | 1.62 |
| Christian (Not Specified) | 38,408 | 0.75 |
| Hindu | 33,827 | 0.66 |
| Presbyterian | 23,597 | 0.46 |
| Other stated religion (nec) | 22,163 | 0.43 |
| Apostolic or Pentecostal | 13,632 | 0.26 |
| Buddhist | 9,285 | 0.18 |
| Evangelical | 8,859 | 0.17 |
| Jehovah's Witness | 6,445 | 0.13 |
| Methodist, Wesleyan | 5,355 | 0.10 |
| Protestant | 5,237 | 0.10 |
| Baptist | 4,262 | 0.08 |
| Pagan, Pantheist | 3,868 | 0.08 |
| Lutheran | 3,706 | 0.07 |
| Spiritualist | 3,350 | 0.07 |
| Lapsed (Roman) Catholic | 3,279 | 0.06 |
| Born Again Christian | 3,162 | 0.06 |
| Agnostic | 2,949 | 0.06 |
| Atheist | 966 | 0.02 |
| **All religions** | **5,149,139** | 100 |

SAPS 4-category, same total: Catholic 3,540,412 · Other religion 508,107 ·
No religion 755,455 · Not stated 345,165.

**Category kinds are mixed, per spec.md §2.3.** This list contains denominations
(*Presbyterian*, *Baptist*), a bare tradition (*Islam*, *Hindu*, *Buddhist*), an
unresolvable catch-all (*Christian (Not Specified)*, *Protestant* — 5,237 people who
wrote "Protestant" and nothing more), two non-religions (*Atheist*, *Agnostic*), a
*former* affiliation (*Lapsed (Roman) Catholic*), and a residual (*Other stated religion
(nec)*). `Pagan, Pantheist` at 3,868 is one of the few places any European census
enumerates modern paganism, so it is worth a taxonomy node. `Orthodox (Greek, Coptic,
Russian)` fuses Eastern and Oriental Orthodoxy into one row and **cannot** be split —
its 65% growth since 2016 is largely Ukrainian and Romanian arrivals, so it is
predominantly Eastern, but the source will not say so.

Ireland's census religion question is a **write-in-backed tick box**, so the long tail
above is coded from free text. There is no published sub-tail beyond
`Other stated religion (nec)`.

---

## 5. The universe trap — two CSO tables, two denominators

**This bit costs you 64,260 people if you get it wrong.**

| table | universe | total |
|---|---|---:|
| **FY106**, SAPS (used here) | Population (usually resident) | **5,149,139** |
| **F5051**, F5070, and the Profile 5 release text | Population usually resident **and present** in the State | **5,084,879** |

Same 24 categories, same county names, different base. CSO's own headline percentages
come from the *smaller* one — the release says "No religion 736,210, 14%", which is
F5051's figure, while FY106 and SAPS say **755,455**. Likewise Roman Catholic
3,515,861 (F5051) vs **3,540,412** (FY106/SAPS). The 64,260 difference distributes across
all four SAPS buckets and is what makes them reconcile.

`ie.py` uses the FY106 / SAPS pairing throughout, because it is the pair that nests
exactly and because the Small Area layer is only available on that base. **Do not mix a
Profile 5 figure into this file.**

---

## 6. Is the question voluntary? — no, and the non-response rate anyway

Unlike England, Wales and Scotland (see `sources/uk.md`), the Irish census carries **no
voluntary question**. Completion of the whole form is a statutory obligation under the
Statistics Act 1993, and CSO does not mark the religion question as optional on the form
or in the release.

**Non-response is nonetheless 345,165 = 6.70%**, and it is the third-largest category in
the country. It is very uneven — a **4× spread** across counties, and not in the direction
you would guess:

| highest | | lowest | |
|---|---:|---|---:|
| Dublin City | 13.50% | Dún Laoghaire-Rathdown | 3.43% |
| Longford | 11.95% | Monaghan | 3.97% |
| Galway City | 11.49% | Cork County | 4.22% |
| Limerick City & County | 9.24% | Meath | 4.35% |

So it is *not* a simple urban/rural gradient: Dublin City is the highest in the country and
Dún Laoghaire-Rathdown, its immediate neighbour, is the lowest. It must be carried as its
own node and never redistributed — spec.md §3.2's residual rule covers it, and at 6.7%
nationally and 13.5% in the largest city it is far too big to absorb quietly.

---

## 7. Licence

**CC BY 4.0.** CSO: "The statistics and other information provided on the CSO site are
accessible free of charge and licensed under Creative Commons Attribution (version 4.0
cc-by). Reproduction is authorised subject to acknowledgement of the source."
data.gov.ie's standard attribution string is *"Contains Irish Public Sector Data licensed
under a Creative Commons Attribution 4.0 International (CC BY 4.0) licence"*.

Suggested credit line for the map: **Central Statistics Office, Census of Population 2022
(CC BY 4.0)**.

---

## 8. Surprises, collected

1. **The finest geography and the finest categories are 18,919 units apart**, and no
   vintage of SAPS has ever closed the gap — 2016 has the same four categories as 2022.
   Ireland has excellent geographic resolution on a four-way split and nothing better.
2. **FY106's label is just "Population"** — searching PxStat table titles for "religion"
   does not find the county-level detailed table. Search the *dimension* labels.
3. **Two published national totals for the same question**, 5,149,139 and 5,084,879 (§5),
   and CSO's own commentary uses the one that does *not* match SAPS.
4. **Atheists and agnostics are inside "Other religion"** in the SAPS 4-way split (§1).
5. **11% of Small Areas changed code between 2016 and 2022** (§3) — the §8.1 vintage trap,
   live in this dataset.
6. **`Lapsed (Roman) Catholic` is a real published category** with 3,279 people. It is a
   statement about a *former* affiliation and is not a group anyone belongs to; treat it
   the way spec.md §2.3 treats *Hindu Yoga and Meditation* — hold it out and let Anita
   decide, rather than folding it into either `catholic` or `none`.
7. **Cork appears merged in F5051 (30 units) and split in FY106 (31 units).** FY106 is the
   one used, so `county` here is the full 31 administrative counties with Cork City and
   Cork County separate.
8. The SAPS Small Area file ships a whole-State row inside it (`GEOGID == "Ireland"`), so
   a naive column sum comes to exactly twice the population. `ie.py` drops it.
