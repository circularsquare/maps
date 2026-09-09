# the Netherlands — `sources/nl.py`, `sources/nl_geo.py`, `sources/nl_grid.py`, `taxonomy/nl2014.py`

Drawn 2026-09-09. **394 gemeenten, 10 nodes, 16,794,120 people, 99.79% of the country.**
The last Dutch census to ask about religion was in 1971.

| | |
|---|---|
| counting geography | **gemeenten, 42,600 people each**, the 403 units of the 2014 classification |
| placement | 49,868 Kontur 400 m hexes, 2023, weighted by hex population |
| basis | self-identification, sample survey |
| tier | **`modelled` throughout**, a survey is not a count of anybody |
| vintage | Enquête Beroepsbevolking pooled 2010–2014, published 2015 |
| authored cells | **none** |
| not drawn | 35,169 people, 0.21%: nine gemeenten CBS suppressed for thin samples |

**The finding is one shelf, not one table.** §11k closed the Netherlands on StatLine
`82904NED` and was right about that table. CBS's *maatwerk* shelf, custom tabulations made
for somebody else and left up afterwards, carries the country's religion geography and is
not in the OData catalogue.

---

## 1. What was already ruled out, and why it did not close the country

§11k's finding stands and is not disturbed here: **`82904NED` *Religieuze betrokkenheid;
persoonskenmerken* is national only**, and its categories change at 2019 when CBS stops
splitting Protestants. The OData catalogue was re-swept on 2026-09-09 with a title filter on
`eligi` and it returns four tables and no fifth:

| id | what it is | geography |
|---|---|---|
| `82904NED` | Religieuze betrokkenheid; persoonskenmerken | **national only** |
| `83288NED` | Religieuze betrokkenheid; kerkelijke gezindte; regio; 2010-2015 | **landsdeel and provincie**, 12 units, stopped 2020 |
| `70794ned` | Religie; naar regio; 2000/2002 of 2003 | regional, twenty years stale |
| `82868ENG` | Caribbean Netherlands; religious denomination | Bonaire, Saba, Sint Eustatius |

So StatLine's best offer for the European Netherlands is **twelve provinces**, which is the
same twelve the European Social Survey's `rlgdnanl` would have given at NUTS 2 and is what
`queue.md`'s ESS block expected to build. That is not what was built.

## 2. The maatwerk shelf, which is the whole finding

**`Religie en kerkbezoek naar gemeente 2010-2014`**, CBS maatwerk 2015/20:

    https://www.cbs.nl/nl-nl/maatwerk/2015/20/religie-en-kerkbezoek-naar-gemeente-2010-2014
    https://www.cbs.nl/-/media/imported/documents/2015/20/
        religie-en-kerkbezoek-naar-gemeente-2010-2014.xls?sc_lang=nl-nl

110 kB of `.xls`, no login, no form. Its sheet is titled *Kerkelijke gezindte en kerkbezoek
in 403 gemeenten, 2010/2014 (18 jaar en ouder)* and it prints, per gemeente, the monthly
service-attendance rate, the share belonging to any denomination, and **nine denominations**:
Katholiek, Hervormd, Gereformeerd, PKN, Islam, Joods, Hindoe, Boeddhist, Anders.

**Why a country of 17 million can be cut 403 ways.** The religion question rode on the
**Enquête Beroepsbevolking**, the labour force survey, from 2010 to 2015. Its samples are
enormous by survey standards: 108,463 people in 2010, 130,529 in 2012, and about **460,000
adults** over the five years this table pools. A gemeente of 40,000 people therefore gets on
the order of a thousand respondents. Nothing else on this map gets that: Greece's ESS pool is
7,885 people over 13 regions and Finland's is 12,741 over 19.

**How to find a shelf like this.** The three routes that worked, in order of cost:

1. The OData catalogue's title filter, which found the four tables in §1 and no maatwerk.
2. `cbs.nl/nl-nl/zoeken?q=...&type=maatwerk`, which is where the file actually is.
3. The CBS paper that draws maps from the same data, *De religieuze kaart van Nederland,
   2010–2015* (2016/51), which publishes **only maps** and no table. It is the reason to
   believe the underlying figures exist; it is not itself a source.

Generalising: **ask a European statistical office's custom-table shelf before concluding it
publishes religion nationally only.** This is §12's SPA rule one shelf up, and it cost about
forty minutes here against a country that was queued as a twelve-province ESS build.

## 3. The three Reformed answers, which are why the depth matters

CBS's card offers `Nederlands hervormd`, `Gereformeerd` and `PKN` separately. They are not
three names for one thing and their maps do not resemble each other:

| answer | national | strongest three |
|---|---:|---|
| Katholiek | 25.86% | Simpelveld 88.2, Gulpen-Wittem 85.8, Nederweert 83.3 |
| Hervormd | 7.30% | Staphorst 47.5, Putten 41.7, Twenterand 39.6 |
| PKN | 5.85% | Dongeradeel 32.4, Ferwerderadiel 32.4, Grootegast 27.6 |
| Islam | 4.63% | Leerdam 17.1, 's-Gravenhage 14.8, Rotterdam 13.4 |
| Anders | 4.43% | Urk 12.9, Elburg 11.8, Nunspeet 11.7 |
| Gereformeerd | 3.58% | Urk 52.2, Bunschoten 51.5, Reimerswaal 28.6 |
| Hindoe | 0.62% | 's-Gravenhage 4.6, Rotterdam 3.1, Zoetermeer 3.0 |
| Boeddhist | 0.36% | Lingewaal 1.9, Beemster 1.8, Cuijk 1.7 |
| Joods | 0.13% | Amstelveen 2.9, Heemstede 1.3, Noordwijkerhout 1.2 |
| Geen | 47.23% | Landsmeer 80.1, Menterwolde 79.8, Pekela 78.2 |

Three new taxonomy nodes were added for the Reformed three
(`christianity.reformed.continental.{hervormd,gereformeerd,pkn}`) and **that is the one thing
put to Anita**, `ask/009-nl-three-dutch-reformed-nodes-hervormd-gereform.md`: three legend
rows no other country uses. The country ships with them. The alternative, all three on
`christianity.reformed.continental`, would put one Reformed colour over the country that
invented the distinction, and the same subtree already carries thirteen leaves that are the
American emigrant branches of exactly these three answers.

**`Anders` is not what it looks like.** It correlates **+0.49** with PKN, **+0.49** with
gereformeerd, **+0.48** with hervormd and **-0.66** with Katholiek across the 394 gemeenten,
and its top of the list is Urk, Elburg, Nunspeet, Oldebroek, Staphorst and Barneveld. So the
biggest thing inside it is the Protestant free-church end and not the urban one; the four big
cities are above the national rate (Rotterdam 6.9%, Amsterdam 6.1%) but nowhere near the top.
`other.nl`'s node description records this as a reading of a residual.

## 4. The three arithmetic decisions, all of them small

**Adults' shares are applied to everybody.** The table's universe is 18 and over. Scaling up
rather than leaving about a fifth of the country undrawn is
[[feedback_leave_children_out]]'s call, and `note_public` says which universe the figures are
from.

**`Geen` is the complement of the published religious total, and the nine parts are scaled to
fill that total.** Both quantities are CBS prints and they disagree by rounding: the nine
one-decimal parts miss the separately rounded total by up to 0.50 points (Laren, worst case;
median 0.10). Taking `Geen` as 100 minus the sum of the parts would push the entire rounding
error into the largest category in the country, so it is absorbed proportionally across the
nine instead.

**Population is CBS's own, 1 January 2014**, from StatLine `70072ned` filtered to
`2014JJ00`. The table holds every gemeentecode CBS has ever issued and leaves the dead ones
null, so the 403 live codes fall out of it without a classification file, and the live rows
sum to 16,829,289 exactly.

## 5. The geography, and the two traps in it

**PDOK serves a WFS per year and the year is a path component.**

    https://service.pdok.nl/cbs/gebiedsindelingen/2014/wfs/v1_0
        ?service=WFS&version=2.0.0&request=GetFeature
        &typeName=gebiedsindelingen:gemeente_gegeneraliseerd
        &count=2000&outputFormat=application/json

403 features, EPSG:28992, `statcode` = `GM0003`, which is the maatwerk sheet's `Gemcode`
zero-padded. 2014, 2015, 2016, 2021 and 2025 all answer. **Only `_gegeneraliseerd` exists for
2014**; `gemeente_niet_gegeneraliseerd` is on the 2023 service and 400s on the 2014 one.

**Why the vintage had to be 2014 rather than rolled forward.** The Netherlands went from 403
gemeenten in 2014 to 352 in 2021, and the merged-away units are not a random eighth:
Molenwaard, Ferwerderadiel, Dongeradeel, Menterwolde, Graft-De Rijp and Zederik are all named
in CBS's own write-up of this table, and four of them are among the extremes in §3's table.
Austria (§9br) solved the same problem the same way.

**The province column is a second witness and is used as one.** The maatwerk sheet prints a
province beside every gemeente; `nl_geo.py` derives one from a spatial join to the 2014
province layer. The two agree on all twelve counts (Noord-Brabant 67, Zuid-Holland 65,
Gelderland 56, Noord-Holland 53, Limburg 33, Utrecht 26, Overijssel 25, Friesland 24,
Groningen 23, Zeeland 13, Drenthe 12, Flevoland 6), which is what rules out a row read off the
wrong line ([[reference_name_join_wrong_neighbour]]).

**KONTUR'S NL EXTRACT CARRIES A SLAB OF BELGIUM.** 2,525 hexes with 492,914 people fall
outside every gemeente, and **371,100 of those people are nearest to Sluis and 5 to 25 km
away from it**: Brugge, Knokke and Zeebrugge, across the border from Zeeuws-Vlaanderen. A
plain nearest-join would have hung Bruges on a Dutch gemeente of 24,000 people and quietly
moved every Sluis dot to the Belgian coast. `nl_grid.py` separates the two populations with a
**300 m cap**: 948 hexes and 94,337 people are within it and are snapped, per
[[reference_archipelago_grid_snap]]'s rule that a coastal rim must be snapped rather than
dropped because the loss is seaward; the remaining 1,577 hexes are dropped as not Dutch.

**The nearest-join must be done in EPSG:28992.** In 4326 the distances are degrees, every cap
passes, geopandas emits a `UserWarning` and nothing else goes wrong visibly. The first run of
this check reported "all 2,525 hexes within 200 m", which is how it was caught.

## 6. What has happened since, printed rather than asserted

The religion question left the EBB after 2015. CBS's live instrument is **Sociale samenhang
en welzijn**, and its current release is the maatwerk `Religie naar regio, 2021/2025`
(2026/11, published March 2026): province and most COROP regions, ages 15 and over, and
**four categories** (Rooms-katholiek, Protestants, Islam, Ander geloof) where this table has
nine. `sources/nl.py` fetches it and prints the comparison at the end of every build:

| province | 2010/2014 EBB, 18+ | 2021/2025 SSW, 15+ | change |
|---|---:|---:|---:|
| Limburg | 77.2% | 57.9% | -19.3 |
| Noord-Brabant | 66.2% | 47.4% | -18.8 |
| Gelderland | 56.4% | 44.2% | -12.2 |
| Overijssel | 58.2% | 47.5% | -10.7 |
| Utrecht | 48.6% | 38.6% | -10.0 |
| Drenthe | 42.0% | 33.5% | -8.5 |
| Friesland | 45.3% | 38.0% | -7.3 |
| Zuid-Holland | 50.0% | 43.6% | -6.4 |
| Zeeland | 55.1% | 49.3% | -5.8 |
| Noord-Holland | 39.1% | 34.9% | -4.2 |
| Groningen | 36.3% | 33.7% | -2.6 |
| Flevoland | 44.8% | 45.0% | +0.2 |
| **Netherlands** | **52.8%** | **42.9%** | **-9.9** |

**Nothing is re-levelled onto this.** The two are different surveys with different age bases
and different wordings, and CBS treats them as different series; the 2023 longread
*Religieuze betrokkenheid in Nederland* compares them side by side rather than splicing them.
Re-levelling would also be impossible below `Protestants`, which is the split the whole
country turns on. What the comparison is for is the size of the caveat in `note_public`.

**The obvious question this raises, for whoever comes back to it.** SSW's own questionnaire
does ask Christians which denomination, from about thirteen options including PKN, CGK and
the various Reformed churches, and its pooled 2012-2022 sample is 85,000. That is a live
instrument with the right category list at COROP. If CBS ever publishes it crossed with
region, the Netherlands should be rebuilt on it: 40 units instead of 394 is much coarser, but
the vintage would be current, and a maatwerk request is how this table came to exist in the
first place.

## 7. What is not here

**No foreign-resident half.** Greece (§9z), Finland (§9by) and Italy (§9bp) split their
countries into a survey half for citizens and a nationality-derived half for foreign
residents, because ESS undersamples non-citizens. The EBB does not: it is a household survey
of residents, its Turkish- and Moroccan-origin respondents are in it, and Islam comes out at
**4.63%** nationally with **17.1%** in Leerdam and **14.8%** in The Hague, which is the right
order of magnitude against any external estimate. Adding a Eurostat `cens_21ctz_r3` layer on
top would double-count.

**No church membership register.** The Netherlands has no Germany-style Kirchensteuer roll.
KASKI at Radboud publishes membership counts for essentially every Dutch denomination and is
the obvious `roll`-basis source, but spec §3.1 says pick one basis and the survey is the one
with the geography. KASKI was not opened here.

**No SGP vote share.** §11's own scouting note suggested SGP results per municipality as a
proxy for the bevindelijk gereformeerde population. It is not needed: `Gereformeerd` is a
published column at the same geography, and a vote is not a religion
([[feedback_proxy_residual_nameable]]).

---

## Review, 2026-09-09, session `f95259a4-nlrev`

A second pass, read against `data/normalized/nl.csv` rather than against the sections above.
Nothing structural was found; three small things were.

**Every figure in `note_public` and in §3's table recomputes exactly from the normalized
CSV**, checked independently: the ten national shares, the per-gemeente top threes for
Hervormd, Gereformeerd, PKN, Katholiek, Islam, Hindoe and Geen, the 16.73% Reformed total,
Urk at 84.0% combined and 98.1% religious, and the 394 units / 16,794,120 people. `Geen` at
47.23% is the largest answer and the least religious three (Landsmeer 19.9, Menterwolde 20.2,
Pekela 21.8) are the peat-colony and north-of-Amsterdam pattern the note describes.

**One correction made to `note_public`.** Its list of the eight highest combined-Reformed
gemeenten skipped Dongeradeel, which ranks 8th at 62.41%, and gave Rijssen-Holten (9th,
61.80%) as the eighth. Dongeradeel was inserted in rank order, so the sentence is now a
correct top nine. Ranks 10 to 12 are Twenterand 61.2, Grootegast 60.8, Nunspeet 59.4, so the
cut is a real one wherever it is drawn.

**The legend cost in ask 009 is smaller than the ask states, and this is worth knowing before
ruling on it.** `christianity.reformed.continental.{hervormd,gereformeerd,pkn}` are L4 nodes.
The viewer opens at L2, where the Netherlands contributes a single `Reformed` row at 2.8m;
the three appear only when a reader expands Reformed and then Continental Reformed. So they
are not three rows everyone sees, which is the shape the `todo.txt` Jewish-categories worry
and the Tonga flag both have. Confirmed on a screenshot of the country at z8.6 with the
Netherlands selected: the legend shows eight rows, `Catholic`, `Reformed`, `Islam`, `Judaism`,
`Hinduism`, `Buddhism`, `No religion` and `Other`.

**Checks and a look at the map.** `check_md.py` clean, `built_countries.py --check` clean,
`check_rollup.py nl` clean (nothing derived, so nothing to roll up), `check_mapping.py nl`
clean, `check_palette.py` puts every nl family pair over dE 25. Two screenshots: dots follow
the towns, the Randstad and Brabant read as dense against an empty north-east, and **Belgium
is blank right up to the Zeeuws-Vlaanderen border**, which is the visible confirmation that
§5's 300 m cap kept the Bruges slab out.

**Fixed silently.** Four references to `ask/nl-reformed-nodes.md`, which does not exist, in
`countries.py` (twice), `taxonomy/branches.py` and `taxonomy/nl2014.py`. They now point at
`ask/009-nl-three-dutch-reformed-nodes-hervormd-gereform.md`.

**Precedent checked and not disturbed.** `Islam` to the `islam` root is what 110 other
mappings do; `other.nl` is one of 115 `other.<cc>` nodes; `Geen` to `unaffiliated` rather
than `secular` matches gr2024, fi2024 and ge2014 as the REVIEW entry says.
