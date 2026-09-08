# Tonga — Tonga Statistics Department, 2021 Census, General Table G 20

Built 2026-09-08. `sources/to.py`, `sources/to_geo.py`, `sources/to_grid.py`,
`taxonomy/to2021.py`. 99,408 people, **156 villages**, 22 categories, nothing derived.

The queue priced this as *"deep list on a small country"* with the unit count unknown and the
note that *"the Free Wesleyan Church is the state church and nothing here counts it"*. Both
halves were right and the unit count turned out to be the story: TSD publishes religion **by
village**, at 637 people per unit, which is the finest tier on this map by population per unit.

---

## 1. Access — the office is open and the report PDF is a decoy

`tongastats.gov.to` answers a plain GET and is WordPress. The queue's note that `wp/v2/search`
is disabled is correct, but it does not matter, because **`wp/v2/media` and `wp/v2/pages` are
both open** ([[reference_wordpress_media_api]]).

The media library is 1,396 files and sweeping it is a **dead end for this question**: 875 JPEGs,
433 PNGs and only 53 PDFs, none of them a census table. `Census Report Vol1 2021` is in there
as a *cover thumbnail*, which is the tell — the reports themselves are served by something else.

That something else is **WP File Download**, the same plugin Fiji, PNG and the Solomon Islands
run ([[reference_wpfd_sweep]]), on `/download/<cat>/<slug>/<id>/<file>`. Its URLs are not in the
media API at all; they are in the **page bodies**, so the route is `wp/v2/pages?per_page=100`
(56 pages, one request) and then a regex for `href`. That yields the whole publication library
in two calls, and the Census Tables page alone carries nineteen general tables:

    https://tongastats.gov.to/download/266/general-tables/7664/4-religion.xlsx

**A whole topic workbook, 95 KB, no PDF parsing anywhere in this country.** Compare the Solomon
Islands, which needed 13 MB of PDF and a column-count anchor to get the same thing.

## 2. What is in the workbook — three tables, and the third is the one

`4-religion.xlsx` has four sheets: a contents page and G 18, G 19, G 20.

| sheet | what it is | shape |
|---|---|---|
| G 18 | religion x division x **sex** | 22 rows, 1 + 3x6 columns |
| G 19 | religion by **district** | 5 divisions + 23 districts, 23 columns |
| G 20 | religion by division, district and **village** | 185 rows, 23 columns |

G 20 is *Population religious affiliation by division, district and village*. **156 villages,
every one of the 22 categories printed at every one of them.**

## 3. The table closes five ways and needs no tolerance

This is the strongest reconciliation on the map after the Solomon Islands, and it has one thing
the Solomons do not: an **external** witness.

1. Each district's villages sum to the district total, on all 23 columns.
2. Each division's districts sum to the division, on all 23 columns.
3. The divisions sum to the printed national row.
4. **G 19 and G 18 are typeset separately** in the same workbook and reproduce G 20's district
   and division figures cell for cell.
5. **UNSD Demographic Yearbook table 28 reproduces the national row again**, all 22 categories,
   to the person — from the return Tonga forwarded to the UN, not from this workbook.

`sources/to.py` asserts all five before it writes a line. Point 5 is worth the trouble because
it is the only check here that is not a copy of the same typesetting: it catches a
misread column as well as a misread row. [[reference_unsd_religion_oracle]]

Four categories are spelled differently in the two sources (`Gospel Church` against
`Full Gospel Church`, `Tokaikolo/Maamafo'ou` against `Tokaikolo Christian Church`, and the DYB's
`Islam ` carries a trailing space). The check pairs on an explicit alias table rather than on
the string, so **a disagreement about a number is never hidden by a disagreement about a name**.
The `Full Gospel` spelling is also what makes that category safely Pentecostal.

## 4. Parsing G 20 — there is no tier marker, and G 19 is the key

G 20 prints divisions, districts and villages **in one column, with no indentation, no code and
no marker of which tier a row is**. Worse, districts are named for their largest village, so
`Pangai` is a Ha'apai district *and* a village inside it, and `Kolofo'ou` is a Tongatapu
district, a village inside it, and a village on Niuafo'ou.

An indentation heuristic would fail and a name lookup would fail. What resolves it is **G 19**,
which prints the same figures for the divisions and districts *alone*: walking G 19 gives the
expected tier of every G 20 row in print order, and each district's villages are then read until
they sum to that district's own printed total. A row that is not where G 19 says it should be,
or a village block that does not close, raises.

One district is titled differently in the two tables — `Nomuka` in G 20 against `Mu'omu'a` in
G 19, which is the district named for its largest village in one and by its own name in the
other. The **figures are identical**, which is what settles the pairing; the parser prints it as
a note rather than failing.

## 5. Half the country is Methodist, counted as four churches

    Free Wesleyan Church            33,953  34.16%
    Free Church of Tonga            11,244  11.31%
    Church of Tonga                  6,782   6.82%
    Constitutional Church of Tonga   1,152   1.16%
    -----------------------------------------------
                                    53,131  53.44%

All four descend from the one Wesleyan mission of 1826, all four are a printed census cell, and
all four are still here. Add the two later revival breakaways — Tokaikolo (1,455) and Mo'ui
Fo'ou 'ia Kalaisi (688) — and **55.6% of Tonga descends from that mission**.

**Nothing else on this map divides a single Protestant tradition this far**, which is why
`christianity.methodist.tongan` was added with four children rather than putting all of them on
`christianity.methodist`. On the branch they would draw as one colour over half the country and
Tonga's actual geography would vanish. The precedent is the Solomon Islands'
`christianity.melanesianindependent.cfc` (§9bh), a node added for one 16,179-member church.

**The history is contested in its detail and the notes say only what the sources agree on.** The
1885 break under King George Tupou I and Shirley Baker, Queen Sālote Tupou III's 1924 reunion,
and a further separation in the later 1920s are common ground; *which* body continues *which* is
not, and the Wikipedia articles for the Free Church of Tonga and the Church of Tonga disagree
with each other about it (and give 1928 and 1929 for the same event). Nothing in
`branches.py` asserts past that.

**Tokaikolo and Mo'ui Fo'ou are in REVIEW**, on `christianity.pentecostal.charismatic`. Their
*descent* is Methodist — Senituli Koloi was a Free Wesleyan minister and took congregations with
him in 1978 — but §2.1's containment is a fact about people now, and they are revival bodies in
practice. The census's own alternative name for Tokaikolo, `Maama Fo'ou`, is not a Methodist
name. If the call is wrong it is one line in `to2021.py` and no geography moves.

## 6. The state church is the one body with no geography

| division | Free Wesleyan | Church of Tonga | Roman Catholic |
|---|---:|---:|---:|
| Tongatapu | 34.1% | 5.9% | 14.4% |
| Vava'u | 34.6% | 6.2% | 11.0% |
| Ha'apai | 33.1% | **20.1%** | 5.4% |
| 'Eua | 36.3% | 7.8% | 15.1% |
| Ongo Niua | 30.7% | 3.8% | **36.3%** |

The Free Wesleyan Church varies by **six points across the whole kingdom**. The Church of Tonga
swings by sixteen and the Catholics by thirty-one. Everything else here has a stronghold and the
national church does not, which is the sort of thing only a village-level table can show.

Where the others are:

- **Church of Tonga is the outer islands.** 37.1% of Lulunga district, 29.8% of Ha'ano, 44.3%
  of Ha'afeva village — the small islands scattered through Ha'apai.
- **Catholics are the far north and one old village.** 42.8% of Niuatoputapu, 300 km beyond
  everything else; and **71.0% of Lapaha**, which was the seat of the Tu'i Tonga.
- **Latter Day Saints, 19.65%, is the largest share in the UN's table.** Of the 67 country-years
  in DYB table 28 that count Latter Day Saints separately, Tonga 2021 is first and Samoa 2016 is
  second at 16.9%. 33.7% of Hahake district; 59.6% of Matahau village.

## 7. The join — village names are not unique, and the duplicates are a volcano

COD-AB Tonga's **ADM3 village** layer is the statistics department's own: it carries `TDOS_VID`
beside the OCHA pcode, and its 23 ADM2 districts and 5 ADM1 divisions are the census's own tiers.
166 polygons against the census's 156 villages.

**Seven village names are printed twice in G 20, 900 km apart.** Niuafo'ou was evacuated after
the 1946 eruption and most of its people were resettled on 'Eua, where they gave the new
villages the names of the ones they had left: **'Esia, Sapa'ata, Fata'ulua, Mata'aho, Mu'a,
Tongamama'o and Petani** each exist in both 'Eua Fo'ou and Niuafo'ou. Kolofo'ou, Hihifo, Pangai,
Houma and Eueiki repeat for ordinary reasons.

**The twins are not alike, so this is not a cosmetic risk.** 'Esia on 'Eua is 71.4% Catholic;
'Esia on Niuafo'ou is 62.7% Free Wesleyan. Petani on 'Eua is 49.4% Free Wesleyan and Petani on
Niuafo'ou is 33.8% Catholic. A name join would have swapped real congregations **and every total
would still have balanced**, which is precisely [[reference_name_join_wrong_neighbour]]. The join
is therefore on the district and the village together, and 151 of 156 pair on the folded name.

### The other five are witnessed by OpenStreetMap, not assumed

Four are a rename rather than a respelling and one is an error in COD, so each was checked by
putting an OSM `place` node through point-in-polygon against the layer:

| census | COD | the witness |
|---|---|---|
| `Nukunukumotu` (Kolofo'ou) | `Nukumotu` TO1103 | a contraction; the Siesia node falls inside it |
| `Pangai` (Pangai, Ha'apai) | `Lifuka` TO3101 | Pangai town is **on** Lifuka island. OSM's Pangai node is inside TO3101 and neighbouring Hihifo is inside TO3102, so the pair is not simply shifted by one |
| `Ha'atu'a / Kolomaile` | `Ha'atu'a` TO4105 | the census prints two villages as **one row**; OSM's Kolomaile node is inside COD's Ha'atu'a, so that polygon already holds both |
| `Ta'anga` ('Eua Motu'a) | `Ohonua` TO4106 | **COD labels two polygons `Ohonua` and has no Ta'anga at all.** OSM's Ta'anga node falls inside TO4106 and 'Ohonua town's falls inside TO4101, which settles which is which |
| `Sapa'ata` (Niuafo'ou) | `SapaataNf` TO5203 | the Niuas suffix, unspaced |

COD disambiguates the resettled names with its own `Nf` and `Ntt` suffixes, which are stripped
inside the two Niuas districts and nowhere else. Its district spelling differs once: `'Eua
Prope`, a DBF field truncated at ten characters from `'Eua Proper`.

**The witness on the other 151 is the division.** COD files each village under an ADM1
independently of the census, and the two organisations agree on all 156.

**Ten COD polygons have no census row and are not units**: uninhabited islets, six of them in
the Vava'u lagoon (Foeata, Vaka'eitu, Mounu, Eueiki, Mala, Fofoa) plus Onevai, 'Ataa, Fukave and
Tapana. Note there are **two Eueiki islands** — the census's Eueiki, 61 people, is in Lapaha on
Tongatapu and joins correctly; the Vava'u one is the uninhabited islet.

## 8. Placement — Kontur, snapped, and it is a third check on the join

Village polygons are land allotments, not settlements: a Tongatapu village runs from the shore
back across its bush allotments with the houses in a band at one end, and `Ha'atu'a` on 'Eua is
43.5 km² of forested plateau with its people on the west coast road. Weighting by area would put
the dots in the plantations, so `to_grid.py` clips Kontur 400 m hexes to the villages.

- **16% of Kontur's people fall outside every village and are snapped, not dropped**, on
  Vanuatu's rule (§9bg §9). Tonga is 171 islands, so the whole country is coastline and the loss
  would be entirely seaward; 98.8% of the strays are within 700 m. This also catches whatever
  the grid puts on the ten uninhabited islets. 0.21% remains unplaced.
- **Eight villages are smaller than one 400 m hex** and are given their own polygon as a single
  cell (§8.2) — mostly the resettled Niuafo'ou villages on 'Eua at about 0.11 km² each.
  [[reference_kontur_resolution_floor]]
- **r = 0.842 over 148 villages**, against a best of 0.256 over 2,000 random pairings, none of
  which reaches it. A modelled 2023 grid sharing no lineage with TSD's census or with OCHA's
  boundaries agrees about how many people are in each of 156 villages — which is the check that
  would have caught a Niuafo'ou village paired with its 'Eua twin.

Six villages sit outside a factor of 6 and are 0.91% of the counted population; four of the six
are the tiny Niuafo'ou villages (9 to 74 people), where a building-footprint model at 400 m has
nothing to work with.

The antimeridian is 300 km away and Tonga does not cross it, but the shapefile is projected on a
**150°E Mercator**, so a bad unprojection would put the country on the far side of the Pacific
rather than tearing it. Both `to_geo.py` and `to_grid.py` assert a span and a bbox.
[[reference_antimeridian]]

## 9. What the map shows

90 dots at 1:1,000 and 6 at 1:10,000 — Tonga is 99,408 people, so this is the microstate scale
that Anita's 2026-09-08 call covers (*"it has 17000 people so itll just be 17 dots"*). Nine
religions draw no dot anywhere and are rings.

The dot bbox stops at 18.6°S because the Niuas villages are too small to draw one, so the entry
carries an explicit `view` reaching 15°S. Without it the camera would cut off a third of the
country, including the most Catholic district in it.

**Tonga is 99.88% drawn.** 119 people in the whole country refused the question.

## 10. What else is on this server, unused

- **G 18 crosses religion with sex** at division level, which nothing here uses.
- **2016 census** report volumes 1 and 2, and the DYB has 2016, 2006 and 1996 religion rows, so
  there is a four-census national series, and it moves in one direction. Over 1996, 2006, 2016
  and 2021 the Latter Day Saints run **13.8% → 16.8% → 18.6% → 19.7%** and the Free Wesleyan
  Church runs **41.4% → 37.3% → 35.0% → 34.2%**, while the Free Church of Tonga and the Church
  of Tonga sit still at about 11.5% and 7%. The 2016 table has 17 categories against 2021's 22.
  Nothing on this map draws a time series yet.
- `2-ethnicity.xlsx` and seventeen other general tables in the same library.
- The **2021 census questionnaires** are published ([[reference_census_questionnaire]]) and were
  not read; the category list is unambiguous enough that they were not needed, but they would
  settle what `Other Pentecostal` was read out as.
- **A 2026 census is in the field** — enumerator recruitment was posted in July and August 2026.
