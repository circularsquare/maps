# Cyprus — CYSTAT Census of Population and Housing 2021

Built 2026-09-08. `sources/cy.py`, `sources/cy_geo.py`, `sources/cy_grid.py`,
`sources/cy_2001.py`, `taxonomy/cy2021.py`. sources.md §9bs is the short version.

**923,381 people, 396 municipalities and communities, 12 religion categories, and not one
published figure that crosses religion with a place.** That last clause is the whole country.

---

## 1. What Cyprus publishes, and the shape of the hole

CYSTAT's online database, **CYSTAT-DB**, carries the 2021 census. Under *Population / Census of
Population and Housing 2021 / Population / Population - Language, Religion, Ethnic Religious
Group* there are exactly six tables, and the split between them is the finding:

| table | what it crosses | geography |
|---|---|---|
| 1891610E | Language x sex x district x urban/rural | **district** |
| 1891613E | Language x sex x citizenship group x district | **district** |
| 1891616E | Language x sex x district | **district** |
| 1891632E | Religion x sex x citizenship group | none |
| 1891635E | Religion x sex x country-of-birth group | none |
| 1891642E | Ethnic/religious group x citizenship group x sex | none |

Language is published by district three separate ways. Religion is published by district zero
ways. The 2011 branch of the same database holds three tables (age/municipality, citizenship by
district, living quarters) and no religion at all.

The twelve religion counts are:

| | |
|---|---|
| Christian Orthodox | 688,075 |
| Not recorded / Not stated | 159,835 |
| Muslim | 19,534 |
| Roman Catholic | 13,860 |
| Anglican/Protestant | 9,621 |
| Atheist/No Religion | 9,591 |
| Buddhist | 7,868 |
| Other Religion | 4,545 |
| Maronite church | 4,486 |
| Sikh | 2,260 |
| Armenian church | 2,025 |
| Hindu | 1,681 |

All twelve match **UNSD Demographic Yearbook table 28** to the person, from the return CYSTAT
forwarded rather than from this database, and they partition 923,381 exactly. UNSD splits the
not-stated cell further, into 146,943 *not specified* and 12,892 *not stated*.

## 2. The route, and the one thing that hides it

CYSTAT-DB is classic ASP.NET PxWeb and its JSON API is **open, keyless and undocumented on the
site**. The trap is the prefix:

```
https://cystatdb.cystat.gov.cy/pxweb/api/v1/en/      -> HTTP 500
https://cystatdb.cystat.gov.cy/api/v1/en/            -> the API
```

Every other PxWeb on this map (`ee`, `lt`, `mk`, `ge`, `gh`, `li`, `xk`, `ch`) lives under
`/pxweb/api/v1/`, so the natural first guess fails, and it fails with a **500 whose body is the
"your saved query can not be found" page** rather than a 404 — the server has read
`api/v1/en` as a saved-query path. That reads like a broken portal and is not one.
[[reference_spa_hidden_apis]] in its mildest form.

Two more hostnames appear in search results and neither is the answer: `cystatdb20.cystat.gov.cy`
404s at the root (a stale search-engine artifact), and `cystatdb23px.cystat.gov.cy` is the same
data. `www.cystat.gov.cy` redirect-loops on `/en/` and `/el/` without a cookie jar, and `gov.cy`
returns 403 to a plain fetcher.

Folder listing and table metadata are `GET`; data is a `POST` of a `{"query": [...],
"response": {"format": "json-stat2"}}` body to the `.px` URL, the same as Estonia's.

## 3. What is drawn: CYSTAT's own arithmetic

    count(religion, community) = SUM over the four citizenship groups of
                                 P(religion | group) x N(group, community)

`P` comes from table **1891632E**, `N` from **1891213E** (citizenship group by
municipality/community, 396 units). Same census, same office, same universe, same reference
day, and **no coefficient from outside Cyprus anywhere**. The national totals reconcile to
0.0011 of a person.

The four groups are far from alike, which is why this buys anything at all:

| | Cypriots | other EU | non-EU |
|---|---|---|---|
| N | 719,252 | 93,540 | 107,168 |
| Christian Orthodox | 84.0% | 63.7% | 22.4% |
| Muslim | 0.45% | 0.52% | 14.7% |
| Roman Catholic | 0.29% | 3.70% | 7.73% |
| Buddhist | 0.03% | 0.03% | 7.07% |
| Anglican/Protestant | 0.11% | 0.53% | 7.76% |
| not stated | 13.7% | 28.8% | 29.3% |

So across the 396 communities the allocation produces Orthodoxy from **62% to 82%**, Islam from
**0.9% to 7.8%**, Buddhism from 0.3% to 3.6%, Roman Catholicism from 0.6% to 4.6%. That is real
geography, and it is the geography of who moved to Cyprus.

This is a **stronger footing than the nationality derivations in `gr`, `es`, `fr` and `it`**,
which have to assume what a Romanian resident of Greece believes. Cyprus counted.

## 4. The join is an integer, and Cyprus hands it over

PxWeb's *value codes* for the community dimension are **Cyprus's own LAU codes** — `1000`
Lefkosia, `1010` Agios Dometios — and Eurostat's GISCO **LAU 2021** layer carries the same code
in `LAU_ID`. The bundle is already on disk as a shared asset (`data/geo/lau2021/`), used by
Greece, Austria and a dozen others.

So the join is made on the code, and the `LAU NAME LATIN` column of
`EU-27-LAU-2021-NUTS-2021.xlsx` is then asserted equal to CYSTAT's English label on all 396,
which it is, character for character. That order matters and is not decoration: Cyprus repeats
village names across districts (`Agios Theodoros`, `Kalo Chorio`, `Pyrgos`, `Kellaki`) and both
lists disambiguate them the same way only because they are the same list.
[[reference_name_join_wrong_neighbour]].

GISCO has **615** Cypriot LAUs; the census enumerates **396**. The 219 left over split cleanly
and the split is asserted rather than assumed: **182** have population `n.a.` in Eurostat's own
workbook, which is the area under Turkish Cypriot administration, and **37** have population
**0**. The 37 are not a defect — they are the Turkish Cypriot villages of Pafos and Larnaka
emptied in 1974 (Vretsia, Fasli, Melandra, Sarama, Evretou, Trimithousa, Kios, Zacharia,
Lapithiou, Foinikas, Maronas, Livadi and the rest) plus the uninhabited Troodos summit, and
nobody was enumerated in any of them.

**Keryneia has no code at all.** Cyprus's six districts are numbered 1 to 6; the census's own
district list is `1, 3, 4, 5, 6`. All 47 of Keryneia's communities are in GISCO and none is in
the census. Ammochostos keeps 9 of its 98 and Lefkosia 109 of its 174. **5,846 km² are drawn of
the island's 9,249**, 63.2%.

### The placement grid, and Akrotiri

`cy_grid.py` clips Kontur 400 m hexagons to the 396 communities. 14.8 km² per unit against a
0.14 km² hexagon clears [[reference_kontur_resolution_floor]] by two orders of magnitude. It
earns its place on the Troodos and the Pafos hinterland, where a community is a mountain valley
with everybody in one village at the bottom; Cyprus draws about 920 dots in total, so one dot on
a ridge is a visible error rather than a rounding one. Kontur inside the drawn area comes to
971,577 against the census's 923,381, **1.052x**, which is what a 2023-11 vintage against a 2021
census should look like and is a useful corroboration that the clip is right.

**One community has no populated hexagon: 5200 Akrotiri**, 931 people, inside the Western
Sovereign Base Area, where Kontur models nobody. This is not harmless. `countries.py` points
`place` at the hex layer, so a community missing from it is not placed on its polygon instead —
it is not placed at all, and `scatter.py` carries its people into *other* units of the same
node, putting them in the wrong village rather than in none. `cy_grid.py` therefore appends the
LAU polygon of any such community at its census population. That invents no geometry and no
population, and the count of communities in the placement layer is asserted at 396.

## 5. The check: 2001 is the only religion geography Cyprus has ever published

`sources/cy_2001.py`. **Table 29 of the 2001 census Volume 1**, *Population by sex, religion,
district and urban/rural area*, at
`library.cystat.gov.cy/Documents/Publication/CENSUS%20OF%20POPULATION%202001-VOL.1.pdf`
(5.0 MB, 329 pp, ungated), PDF pages 253-254. Nine categories, five districts, 689,565 people.
The five districts sum to the printed total on every column, and all nine totals match UNSD's
2001 return to the person.

**It is not drawn.** Cyprus was **94.8% Orthodox in 2001 and is 74.5% now**, on a population a
third larger, and the whole of that difference arrived after 2001. What it is for is to put a
number on the one assumption `cy.py` cannot test from 2021 alone: that religion does not vary
between places *within* a citizenship group.

The answer is clean, and it is the predicted one. Concentration is stated as a multiple of the
district's own population share, so 1.00 means "spread exactly like the population":

| | Lefkosia 2001 | Lefkosia 2021 alloc | verdict |
|---|---|---|---|
| Orthodox | 1.00 | 1.03 | reproduced |
| **Armenians** | **1.88** | **1.01** | **missed entirely** |
| **Maronites** | **2.07** | **1.03** | **missed entirely** |

and at the other end of the island:

| | Pafos 2001 | Pafos 2021 alloc | verdict |
|---|---|---|---|
| Anglican/Protestant | 3.37 | 1.86 | direction found, about half the magnitude |
| Roman Catholic | 0.85 | 1.72 | reversed, and see below |
| Muslim | 0.94 | 1.77 | reversed, and see below |

**The rule this establishes, and it generalises past Cyprus: an allocation reproduces exactly
those concentrations that live in the dimension it allocates on, and is blind to every
concentration inside one of its cells.** Orthodoxy is reproduced across all five districts
because it is the modal answer of every group. The Anglicans of Pafos are half-reproduced
because they are mostly British and Britain is non-EU, so the citizenship dimension can see
them — but non-EU also holds Syrians, Filipinos and Sri Lankans, and averaging over all of them
flattens the peak. The Armenians and Maronites are **invisible**, because both sit inside
`Cypriots` and that group has one profile, and the allocation returns a flat 1.0x where the
measurement says 1.9x and 2.1x.

The two rows that *reverse* are not failures of the method: Pafos's Roman Catholics and Muslims
in 2021 really are migrants (Polish and Filipino Catholics, Syrian and Bangladeshi Muslims) in a
district that had almost none in 2001, so a change of sign between the columns is the country
changing rather than the allocator lying. **The 2001 column is not ground truth for 2021 and a
gap here is an upper bound on the error, not a measurement of it.**

Cost of the two missed rows: the Armenian church and the Maronite church are 6,511 people,
about **six dots at 1:1,000**, drawn flat when they belong disproportionately in Lefkosia. The
alternative was to redistribute them off the language table (1,067 Armenian speakers by
district against 2,025 Armenian-church members, so the coefficient would have been invented),
and for six dots that is a worse trade. Recorded in `taxonomy/cy2021.py`'s REVIEW and said in
the country note.

### Parsing note

The 2001 volumes are typeset in a legacy Greek symbol font, so every Greek label extracts as
Latin lookalikes (`ĬȇǾȈȀǼȊȂǹ` for ΘΡΗΣΚΕΥΜΑ). Digits and English captions extract correctly, so
the parse anchors on English. **Two of the nine English column headers still cannot be matched
literally**: `Orthodox` extracts as `Ȅrthodox`, because its capital O is a Greek omicron glyph,
and `Roman catholic` is typeset over two header lines so it is never one string. Both are
matched on the surviving fragment. A header assertion that fails on a page plainly containing
the word is worth knowing about before blaming the extractor. Thousands separator is `.`.

## 6. Below district there is only 1960

`POP_CEN_1960-POP(RELIG_GROUP)_DIS_MUN_COM-EN-250216.pdf` — the 1960 census, religious group
**village by village**, whole island including the north, pre-partition. Table V is *Population
of Nicosia district, by place of enumeration, race and sex*, with columns Greeks / Maronites /
Armenians / Turks / British / Gypsies / Other.

Dead on the live site (404 on `library.cystat.gov.cy` and `www.cystat.gov.cy`). Working copy:

```
https://web.archive.org/web/20190301013415if_/http://www.mof.gov.cy/mof/cystat/statistics.nsf/
  All/1240A557C7D9F399C2257F64003D0D54/$file/POP_CEN_1960-POP(RELIG_GROUP)_DIS_MUN_COM-EN-250216.pdf
```

1.86 MB, 20 pp, **scanned, no text layer**, so it needs OCR. Two other Wayback snapshots of the
same file (`20220107111854`, `20211027214653`) return exactly **1048576 bytes** and are
truncated: `page_count` 0, no `%%EOF`. Use the `20190301` capture.
[[reference_pdf_truncated_at_source]].

A 1946 equivalent exists and was not opened
(`POP_CEN_1946-POP(RELIGION)&HH_DIS_MUN_COM-EN-121017.pdf`, snapshots `20211024144917` and
`20191207091752`), and there is an 1881-1931 series whose filenames do not claim religion.

**The whole pre-2015 CYSTAT tree at `www.mof.gov.cy/mof/cystat/statistics.nsf/*` is dead and
lies about it**: every path 301/302s to the cystat homepage, so a request "succeeds" and returns
3.2 MB of homepage. It is Wayback-only. Enumerate it with

```
http://web.archive.org/cdx/search/cdx?url=mof.gov.cy/mof/cystat/statistics.nsf*
  &fl=original,timestamp&collapse=urlkey&filter=original:.*POP_CEN.*&limit=500
```

[[reference_dead_stats_office]].

## 7. What is not drawn

**17.3% of the country did not answer.** 159,835 people. The religion question was **optional**
in the 2021 census, which no other census on this map made it, and the refusal is concentrated
where the map is already weakest: 13.7% of Cypriot citizens against **28.8% of other EU
citizens and 29.3% of non-EU citizens**. Excluded rather than scaled up, so Cyprus draws
763,546 of 923,381.

**Northern Cyprus.** The census covers the government-controlled area. The 2011 TRNC census, the
only one ever held there, is published at
`istatistik.gov.ct.tr/TEMEL-İSTATİSTİKLER/NÜFUS-SAYIMLARI/Nüfus-Sayımı-2011` as eleven files —
nine population tables and two housing — and **none of them asks about religion or ethnicity**.
Its geography goes down to *mahalle*, and the only compositional variable it carries is
*tabiiyet*, citizenship (TRNC / Türkiye / other). So there is no religion table to draw and no
proxy that this map's rules would accept; the office moved off `devplan.org` and the old
hostname is a dead end. Not closed harder than that: the TRNC 2011 questionnaire at
`unstats.un.org/unsd/demographic/sources/census/quest/CYP2011tr.pdf` would say whether the
question was ever asked, and was not opened.

**No Jewish category.** The religion question's twelve answers do not include one, although the
same census counts a Jewish community in its ethnic/religious-group table (1891642E) and 885
Hebrew speakers in its language table. Those people are inside `Other Religion` and therefore
inside `other.cy`.

## 8. What to fix next, best first

1. **A finer citizenship cut.** 1891215E is citizenship group by **postal code** and 1891221E is
   **country of citizenship** by district. Neither adds a religion cross, so neither helps
   directly — but if CYSTAT ever publishes religion by *country* of citizenship rather than by
   the four-way group, the whole allocation gets far sharper at a stroke, because the origin
   countries are named (Greece, Romania, UK, Russia, Philippines, India, Sri Lanka, Nepal,
   Syria, Vietnam) and several are drawn on this map already.
2. **The three Cypriot minorities.** Nothing published since 1960 places Armenians, Maronites or
   Turkish Cypriots below the national level except 2001's five districts. The Maronite villages
   are a matter of public record (Kormakitis, Asomatos, Karpasia, Agia Marina, all four in the
   north) and the Armenian community is Nicosia and Larnaca; none of that is a *published table*
   and §14.4 is why it is not used.
3. **1960 by village**, if anyone wants to OCR twenty scanned pages. It would give the only
   village-level religion Cyprus has, for the whole island including the north, sixty-six years
   stale. Worth it as a historical layer, not as a source for this map.
4. **Ask CYSTAT.** The database is a published subset; a census that tabulated religion against
   citizenship, birth country, age and sex has the variable on the record and could cross it
   with district. That is a request, not a lead, so it needs Anita.

## 9. Review, 2026-09-08

Second pass, `rd-review`. The build is sound: `check_md.py` clean, both editions present,
`check_rollup.py` reports Cyprus 100% orphaned and `taxonomy/cy2021.py`'s docstring already
explains why that is the correct answer rather than a missing `COLUMNS` entry. The seven `REVIEW`
calls were spot-checked against the precedents they name and all hold: `au2021` and `nz2023` do
file "Maronite Catholic" to the eastern-Catholic node, and `bg2021`, `ee2021`, `ge2014`, `pl2021`
and `ro2021` do all use `christianity.oriental` for their national Armenian churches. The
screenshot is clean, with dots on land, the four cities where they should be and the north empty.

**The one thing to fix is the spread figures in §3 above and in `note_public`.** They do not
reproduce from `data/normalized/cy.csv`, and they are presented as a range when they are a
trimmed band.

§3 says the allocation produces "Orthodoxy from 62% to 82%, Islam from 0.9% to 7.8%, Buddhism
from 0.3% to 3.6%, Roman Catholicism from 0.6% to 4.6%", and `note_public` repeats the Islam and
Buddhism pair. Recomputed across the 396 communities in the shipped CSV:

| | note says | share of drawn population | share of all enumerated |
|---|---|---|---|
| Christian Orthodox | 62 to 82 | 31.69 to 97.36 | 22.41 to 84.01 |
| Muslim | 0.9 to 7.8 | 0.52 to 20.74 | 0.45 to 14.67 |
| Buddhist | 0.3 to 3.6 | 0.03 to 10.00 | 0.03 to 7.07 |
| Roman Catholic | 0.6 to 4.6 | 0.33 to 10.92 | 0.29 to 7.73 |

Two separate problems, and the second is the one that reaches the reader.

**The quoted numbers are a percentile band, and the band is not the same one twice.** Orthodoxy's
62 to 82 is close to a p5 to p95 on the all-enumerated denominator (59.48 to 82.54); Islam's 0.9
to 7.8 is close to a p5 to p95 on the drawn denominator (0.83 to 7.06). No single definition
produces all four rows, so whatever was computed was not computed the same way each time.

**Trimming the tails was the right instinct.** Every extreme in the table above comes from a
community of one to seven people: Petrofani, population 7, is the 20.74% Muslim and the 10.00%
Buddhist, and Pitargou, population 1, is the 97.36% Orthodox. Quoting those as the range would be
worse than what is there now. The defect is that "runs from 0.9% of a community up to 7.8%" reads
as a floor and a ceiling and is neither, and the ceiling is passed by 16 communities holding 9,745
people, three of them substantial: Pegeia at 10.2% Muslim, Tala at 8.7%, Pissouri at 8.1%. Those
are exactly the coastal places a reader would think to check, and the map will not agree with the
note when they do.

The arithmetic behind all of this is fine and the direction of the error is conservative, so the
argument the figures are making survives intact and is in fact stronger than stated. Worth noting
that because each community's share is a convex combination of the three citizenship-group shares
in §3's table, the true community min and max are pinned to that table's own columns: Orthodoxy's
22.41 and 84.01 on the all-enumerated denominator are the non-EU and Cypriot entries exactly. So
the honest phrasing is available without any new computation, either as "across the middle 90% of
communities" with one denominator stated, or by naming the group shares the spread is bounded by.

Not fixed here, because choosing the band and the denominator is a wording call that belongs to
whoever owns the note rather than to a reviewer, and `note_public` is asserted at import.
