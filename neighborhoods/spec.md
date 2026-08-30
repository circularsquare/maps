# neighborhoods — how it works, and why

A browser for the neighbourhoods of world cities: name, shape on a map, population, and
a notability score. A quiz ("here is a neighbourhood, name the city") is a filtered view
over the same data, built later — see §7.

**This file is Claude-managed.** It records the state of the pipeline and the reasoning
behind it, especially the measurements that settled a decision and the approaches that
were tried and rejected, so the same ground is not re-walked. `todo.txt` is Anita's and
is not mine to edit. Section numbers here are what `todo.txt` points at.

Run it with `python serve.py`; see §8.

---

## 0. State

**Browsable, with shapes, notability and de-duplication.** All **56 cities** surveyed,
**30,098 units** in kept levels resolving to **25,257 distinct places** (§4.6). Geometry:
**15,991 outlines**, 53% of units, down to 23.6 MB of display GeoJSON. Notability joined:
97% of QIDs resolve to a Wikipedia article, sitelinks everywhere and pageviews filling in.

The unit count went **down** by 2,842 against the previous build and that is the headline
fix: 3,137 of the old units were **boundary line-work rather than places** (§1.5), and
removing them is what makes London's `admin=8` the 33 boroughs instead of 87 units
dominated by River Thames segments. `MIN_UNITS` came down 8 → 4 at the same time (§3.3),
which adds back the borough-scale levels the corpus previously could not represent — New
York's five boroughs among them.

**The roster is now 301 cities, of which 56 are fetched.** `roster.py` added 245 from
Wikidata across 125 countries (§1.6); `cities_seed.csv` holds them, and nothing downstream
has run for them yet — `pick_levels` skips a city with no survey cache, so the build stays
consistent while the roster runs ahead of the data. Fetching the other 245 is §8's
three passes and several hours of Overpass.

**Population coverage tripled**, 5.7% → **19.1%** of distinct places, from Wikidata P1082
(§1.7) — 201 requests and three minutes, run alongside the views pass without disturbing
it. Where it overlaps the OSM tag the two figures are identical (median ratio 1.000).

In flight: the pageviews pass (~6.5 h), and a Who's On First stage for the three cities
borrowing cannot reach.

Not built: the quiz (§7). §5 ranks what is left.

## 0b. The browser and the quiz want different things

This is the organising decision and everything else follows from it.

| | **browser** | **quiz** |
|---|---|---|
| unit of selection | a whole **level** | an individual **unit** |
| rule | all communes or none — §3 | notability cutoff on top — §7 |
| size test | population floor, 10k+ | **5–30 per city** by pageviews |

The browser is exhaustive within a level because a half-populated level is not a way
anyone divides a city. The quiz is ruthless because **most people recognise 0–10
neighbourhoods in a city they have never been to** — 325 New York neighbourhoods is a
browsing corpus, not a quiz deck.

So the level rule (§3) is the browser's contract, and the quiz is a *view* over its
output. Nothing in the level rule should be weakened to make the quiz smaller, and
nothing in the quiz filter should be pushed back up into the level rule.

## 1. Sources

| | |
|---|---|
| **Divisions** | **OpenStreetMap via Overpass**, per city, cached. `place=borough\|suburb\|quarter\|neighbourhood\|city_block` plus `boundary=administrative` at `admin_level` 6–11. |
| **Population** | OSM `population` tag where it exists — rarer than assumed, §4.2. **Wikidata P1082** on top of it, §1.7, which triples the coverage. The division estimate (§3.1) for the level rule. Kontur (§1.2) for the units neither reaches, not yet integrated. |
| **Notability** | Wikipedia sitelinks + pageviews, joined on the `wikidata` tag OSM carries on the unit itself. `fetch_wiki.py`; sitelinks done, views ~10%. See §1.4. |

### 1.1 Why OSM and not Overture

Considered and rejected, for reasons worth keeping because Overture looks like the
obvious choice:

- Overture's divisions theme is **conflated OSM plus geoBoundaries**, ODbL. Not an
  independent source — same polygons, one release stale.
- Its headline value-add is a **normalised subtype vocabulary** (borough / macrohood /
  neighborhood / microhood). That is the one thing we must not have. A German
  `admin_level=9` and a Japanese one denote different things, and the level rule is only
  sound *because* it never compares across countries. Normalising erases the distinction
  the rule is built on.
- Polygon coverage would be identical: Overture only has a polygon where OSM has a
  boundary relation.
- `population` and `wikidata` are **OSM tags**; Overture only lifts them. We read them
  directly and get the same values.
- Practically it would mean duckdb and remote GeoParquet scans for data we can fetch per
  city and cache in a few hundred KB.

**When to revisit:** if the near-duplicate level problem (§4.6) turns out to need a real
division hierarchy rather than spatial reasoning, Overture's `parent_division_id`
becomes genuinely attractive and worth the dependency.

### 1.2 Population via Kontur — built, and what it actually turned out to be

`fetch_kontur.py`, tested end-to-end on five countries. The optimistic reading of the
docs was half right and the decisive half was wrong.

**Licences differ between the two Kontur datasets, and an earlier version of this file
had it wrong.** Kontur *Population* is CC BY 4.0. Kontur *Boundaries* is **ODbL 1.0**,
because it is OSM geometry — share-alike, and attribution to OpenStreetMap contributors
and Kontur is required. Both strings are emitted into `kontur_pop.json` so they travel
with the data.

**True:** Kontur Boundaries carries `osm_admin_level` as the raw OSM tag, so `admin=8`
maps to `osm_admin_level == "8"` with no vocabulary in between. Confirmed across NL, KR,
BR, TW, TH.

**False, and it changes the design:** the file has **no OSM element id and no wikidata
QID**. Six fields only — `admin_level`, `osm_admin_level`, `name`, `name_en`,
`population`, `hasc`. Nothing in it is a foreign key, so this is *not* a join. Identity
must be reconstructed from name plus geometry, which is a heuristic and needs a
provenance marker on every figure. It also does **not** replace the geometry pass: it
contains only `boundary=administrative`, so every `place=*` node is simply absent. Nor is
it complete within a level — Kontur's NL has 293 `osm_admin_level=8` rows against ~342
Dutch municipalities.

Three rules, in decreasing trust, each recorded as `via` on the figure it produced:

| `via` | rule | accuracy vs OSM's own `population` tags |
|---|---|---|
| `kontur-admin` | `admin=N` unit ↔ row with same `osm_admin_level` and name | median 1.24, 85% within 2× |
| `kontur-name` | `place=*` unit ↔ same-named row at any level containing its point | median 0.96, 94% within 2× |
| `kontur-hex` | no name match but the unit has a polygon → area-weighted 400 m hex sum | median 0.79, 69% within 2× |

The `admin` figure of 1.24 is one city: Bangkok's *khet* come out 2.5–2.7× their OSM tag
because the *tag* is a registered-household count for a city whose real population is
roughly double. Excluding Bangkok it is 1.04 with everything inside 2×.

`kontur-name` is deliberately **refused** to `admin=N` units. Kontur's NL has no level-8
Amstelveen, so `admin=8` Amstelveen (a municipality, ~91k) would silently have been given
the level-10 *woonplaats* of the same name, 78,882 — wrong unit, plausible number,
invisible error.

**Coverage: 52% of 8,370 candidate units over six cities — but the split is the story.**
91% of units that have a polygon, **18% of units that are a bare node**. A polygon can be
measured; a node can only be recognised by name. Bangkok is the pure failure case at 0% of
nodes, because Thai `place=*` names do not repeat the *khwaeng* names the boundaries carry.

**Kontur Population is a MODELLED surface**, not a census — GHSL plus building footprints,
distributing people by built volume. At borough scale it is excellent (Amsterdam 918,353
vs 921,000 actual; Gangseo-gu 557,530 vs ~568,000). At neighbourhood scale it flattens the
real density gradient badly: Amsterdam's Westpoort, a container port OSM tags at 664
people, comes out at **115,203**; the Jordaan gets 8,201 where CBS counts ~19,000. **The UI
must never present these as resident counts.** Area-weighting the hexes is mandatory — an
earlier centroid-assignment implementation gave the Jordaan 5,264 from a single hex.

**Cost: `cache/kontur/` is 881 MB for five countries; all 23 surveyed cities would be
~2.5 GB** (Brazil alone is 717 MB, France boundaries 350 MB). `python fetch_kontur.py
--plan` prints the per-country breakdown before downloading anything.

### 1.7 Population from Wikidata P1082 — the cheapest source there is

`fetch_wiki.py --pass pop`. **201 requests, ~3 minutes, and it triples the population
coverage**: 5.7% of distinct places → **19.1%**, +3,235 units. The one to run before
considering anything else.

| level | n | OSM tag | +P1082 | combined |
|---|---|---|---|---|
| `admin=6` | 485 | 227 | **132** | 74% |
| `admin=8` | 2,926 | 451 | **1,164** | 55% |
| `admin=9` | 2,872 | 134 | **922** | 37% |
| `admin=7` | 974 | 230 | 71 | 31% |
| `admin=10` | 2,700 | 28 | **443** | 17% |
| `place=suburb` | 4,848 | 219 | 391 | 13% |
| `place=quarter` | 2,591 | 49 | 22 | 2.7% |
| `place=neighbourhood` | 6,738 | 45 | 87 | 2.0% |

**Where both sources exist they are the same number.** Over the 949 units carrying an OSM
tag *and* a P1082 claim the median ratio is **1.000** with **99% inside 2×** — not two
estimates agreeing, but two copies of one census, since OSM's tag was frequently imported
from the same national release. That is the whole argument for trusting the 3,235 units
where only Wikidata has a figure, and it is a far stronger result than Kontur's 1.24 /
0.96 / 0.79 (§1.2).

**It is one request for 50 QIDs and has no language fanout**, which is why it costs 201
requests where the views pass costs 78,235: a pageview count exists per-wiki, a population
claim exists once per item however many Wikipedias carry it.

**Picking among an item's claims is the whole job.** A commune routinely holds a dozen
P1082 values, one per census. `best_p1082` drops `deprecated` rank outright (a value known
wrong, not merely old), then prefers `preferred` rank — the editors' own statement of
which figure is current — then the latest `P585` point in time. An undated claim sorts
below every dated one rather than being discarded, because on small places it is often the
only figure there is.

**The year is not optional to display.** Figures range 2010–2025 (mode 2021, then 2024)
and 241 of 4,202 are undated. A 2010 count and a 2025 one must not share an axis.

**It could not reuse the p31 pass, and this is the trap worth recording.** Both want the
identical `wbgetentities&props=claims` request, so merging them looks free — but
`p31.jsonl` already holds every corpus QID, so `--pass p31` resumes to an empty worklist.
A combined pass would have reported success and fetched nothing. Separate ledger
(`cache/wiki/pop.jsonl`), separate output (`data/pop.json`), separate early return, all
following §1.4's p31 precedent.

**Safe to run during the views pass**, and it was. Different host — `www.wikidata.org`
against `wikimedia.org` — so the two are not sharing a rate limit, and the early return
means `data/wiki.json` is never touched. Two 429s over the run, absorbed by the pacer.

**Carried as `wp` (+ `wy` for the year), never merged into `p`.** Same rule as §1.6's `e`
flag on borrowed geometry: two figures with different provenance and different dates must
not become one field the UI cannot tell apart. `p` stays the OSM tag.

`popLine` in the browser renders it with the provenance always attached — `osm tag` or
`wikidata · 2021`. OSM leads only because it is the tag on the object being shown; the
Wikidata figure appears as a **second line only when the two disagree by more than 10%**,
so a disagreement is a signal rather than a wall of duplicated numbers. Both routes into
the card go through it (hover and the pinned panel), so the rule cannot drift apart
between them.

**It does not feed the level rule.** The rule rests on the division estimate (§3.1) and
folding 3,235 new real populations into it would move level verdicts corpus-wide — a
separate decision, not a side effect of a display field. §4.2's framing holds: this is
what the browser shows, not what picks the levels.

**What it does not reach.** Coverage tracks whether a country's neighbourhoods have
Wikidata items at all, and splits the corpus in two. Melbourne 99%, Rome 92%, Sydney 90%,
Prague 88%, Madrid 66%, Istanbul 63%. Against Bogotá 0.3%, Delhi 0.6%, Mumbai 0.8%,
Beijing 1.1%, Johannesburg 2.0%. The binding constraint is unchanged and is **the join,
not the source**: 58% of the corpus carries no QID at all, and 65% of those are bare nodes
with no polygon — the same 91%-vs-18% wall §1.2 hit. A per-country statistical release
still has to attach to a named point with no identifier.

### 1.4 Notability — `fetch_wiki.py`

Three resumable stages writing append-only JSONL ledgers to `cache/wiki/`, so a kill loses
at most the in-flight request: `--pass sitelinks` (`wbgetentities`, 50 QIDs/request),
`--pass views` (one request per language per article), `--pass merge` (rebuilds
`data/wiki.json` with zero requests).

**Sitelinks: done.** 9,826 distinct QIDs; **9,536 (97%) resolve to at least one Wikipedia
article**; 86,386 articles total, median 4 per QID. 17 QIDs no longer exist on Wikidata —
stale OSM tags.

**Views: ~10% done, and far more expensive than estimated.** 78,235 requests remain at a
sustained 3.3 req/s ≈ **6.5 hours**. The earlier ~35,000 estimate was made when the corpus
was 21 cities; it doubled. `--pass views --limit N` takes the most-linked QIDs first, so
the top of the deck can be bought cheaply. Wikimedia gives less than expected: 10 req/s
drew a 429 about every 300 requests, so the default is 5 req/s with an adaptive pacer.
Connection reuse via `http.client` measured 19.6 req/s against `urllib`'s 5.4 and asks the
endpoint for one TLS handshake rather than 86,000.

**All-languages was the right call, and the data confirms it.** Copacabana totals 635,898
views of which English is 141,197 — enwiki alone would have seen 22% of its readership and
mis-ranked it badly against English-speaking neighbourhoods.

**Per city the ranking is already quiz-ready.** Prague: Staré Město, Malá Strana,
Hradčany, Josefov, Žižkov. San Francisco: Castro, Fisherman's Wharf, Chinatown,
Haight-Ashbury. Rio: Copacabana, Ipanema, Botafogo, Leblon. Also right for Delhi, Nairobi,
Johannesburg, Lagos, Mumbai, Warsaw, Madrid, Hong Kong, Shanghai.

**Globally it is not, and the reason is not the metric.** The global top 20 is 13
municipalities and 10 self-references (§4.9). The top 100 holds 76 `admin=8` and exactly
one `place=neighbourhood`. Sorting the whole corpus by fame surfaces de-duplication and
self-reference problems, not bad scores — so **rank within a city, never globally**.

**It is also a free QID-error detector.** Singapore's "Peng Siang" (57 sitelinks) points at
the article *Common year*; NYC's "Bronxdale" (100) carries The Bronx's QID. The report
prints the article title whenever it differs from the OSM name, so these read at a glance.
And it flags mis-picked levels: Vienna's kept level tops out at 9 sitelinks and Hanoi's at
6, which is what a sub-neighbourhood level looks like when no ranking can rescue it.

### 1.5 A tenth of the corpus was boundary LINE-WORK, not places

The survey query is `nwr[boundary=administrative]`, and OSM routinely repeats the boundary
tags on the **member ways** of a relation as well as on the relation itself. Such a way is
named after whatever the border runs along, so it arrived as a unit called `River Thames`,
`Pedro Gil Street`, `Bergenline Avenue`, `중랑천`, `神田川`.

Confirmed geometrically against `cache/geom/` — a way whose first node ≠ last node is a
line, not an area. **3,137 units, 9.5% of the corpus**, and concentrated exactly where it
does the most damage:

| | units | line-work | |
|---|---|---|---|
| `admin=6` | 1,215 | **689** | 57% |
| `admin=8` | 4,642 | **1,840** | 40% |
| `admin=7` | 1,055 | 123 | 12% |
| `admin=9` | 3,301 | 323 | 10% |
| `admin=10` | 3,995 | 158 | 4% |
| all `place=*` | 18,732 | **2** | 0.0% |

616 of them were two-node ways — a single line segment. **All 4,545 way-units carried
`poly: 1` and 3,502 were being drawn as polygons**, so they also inflated the shape
coverage figure. Fourteen kept city-levels were ≥50% junk (Santiago `admin=6` 98%,
Istanbul `admin=6` 94%, Moscow `admin=8` 90%, Bogotá `admin=8` 77%).

It decided level picks, which is why this is not a cosmetic bug. London `admin=8` was 87
units of which 53 were Thames/Wandle/Beverley Brook segments; the 34 that remain are
London's 33 boroughs. London `admin=6` was 15 units, 14 of them named `River Thames`, and
was being kept as a level.

**The test is the `place` tag, not the geometry** (`is_boundary_linework`). The survey pass
is `out tags center` and has no node list to close a ring with, and re-running 56 Overpass
surveys to learn what a tag already implies is not worth the donated CPU. Scored against
the 4,252 ways the geometry cache *can* adjudicate: **99.9% recall (2,837 of 2,841) for 65
false drops**, and those 65 are mostly closed street loops (`Pedro Gil Street`,
`Elliptical Road`) that are junk anyway. Legitimate closed ways are untouched — 95% carry a
`place` tag against 0.1% of the line-work.

Relations are deliberately exempt: a `boundary=administrative` relation is an area by
construction, and checking every relation in `cache/geom/` found no line-work among them.

This **generalises the water rule** that preceded it, which caught only the 813 borders
that follow rivers. Water stays a separate test because it also catches nodes and
relations, which the way rule does not.

**After the fix:** way-units 4,545 → 1,351, all closed areas; 4 stragglers survive because
they carry a `place` tag and are therefore exempt by design.

### 1.3a Shape coverage is uneven by NATIONAL CONVENTION, and borrowing fixes half of it

Measured over 56 cities: **50% of units have a polygon**, distributed nothing like evenly.

**Final, after the full geometry pass over all 56 cities and donor borrowing: 16,399
outlines, 50% of units, 822 of them borrowed.** Only three cities remain under 15%.

| | shapes | borrowed | note |
|---|---|---|---|
| Copenhagen | 2% | 0 | no same-named polygon exists at any level |
| Hong Kong | 8% | 0 | |
| Mumbai | 8% | 0 | 17 twins of 781 nodes |
| São Paulo | **38%** | **291** | was 13% — donor levels worked |
| Shanghai | 54% | 126 | |
| Bogotá | 47% | 92 | |
| Tokyo | 51% | 51 | |
| Manila | 84% | 46 | |
| Paris, Singapore | 82–83% | 0 | |
| Stockholm, Jakarta, Tel Aviv | 100% | 0 | |
| **total** | **50%** | **822** | |

Borrowing's whole yield is 822 shapes, not the 3,355 a naive count suggested — see the
correction below. Copenhagen, Hong Kong and Mumbai are unreachable this way and are what
Who's On First (§1.3) is for.

The spread is a fact about **mapping conventions, not cities**: Danish mappers put
districts on `place=suburb` nodes, German and Russian mappers draw polygons. Copenhagen
is not less divided than Hamburg.

**Borrowing.** The same place is frequently mapped twice at different levels (§4.6), so a
node-only `place=suburb` often has an `admin=9` twin that carries the outline. Where it
does, the outline is attached to the node's record.

**But most of the headline gain was an illusion, and the measurement that showed it is
worth keeping.** A first implementation borrowed from any same-named containing polygon
and reported 2,976 borrowed outlines, apparently 50% → 65% coverage. Fingerprinting the
shapes showed **2,912 of them were byte-identical to an outline already on screen**: the
twin was itself a unit in the corpus, so the "new" shape was the same polygon drawn
twice. Borrowing had not found shapes, it had found duplicates.

So a candidate whose id is already a corpus unit is **not** borrowed from. The pair is
recorded as `dup` on the node — a de-duplication fact for §4.6, not a shape — and the node
stays a pin. Real borrowing is therefore only possible from **donor levels**, whose units
are by definition not in the corpus. Currently 20 cities have donor levels (24 in total:
Tokyo `admin=10`+`admin=8`, São Paulo `admin=10`+`place=neighbourhood`, Manila, Dubai,
Berlin, Milan…), and **2,870 dedup candidates** have been identified as a byproduct.

The lesson generalises: *a coverage number that counts shapes rather than distinct places
will happily count the same polygon twice.* Fingerprint before believing a gain.

- **Gated on containment, not just the name.** A same-named polygon that does not contain
  the node is a different place — every large country has several Santa Cruzes — and
  matching on name alone would confidently draw the wrong one. The node lying inside the
  polygon is what turns a name coincidence into evidence. Name matching itself is
  casefold + strip-accents and deliberately **not** fuzzy: borrowing asserts two records
  are the same place, and edit distance is not evidence for that.
- **Donor levels.** Geometry is only fetched for kept levels, so a city can look
  shapeless while the same places sit as polygons one level over. `find_donors()` spots
  rejected levels holding outlines for node-only kept units and adds them to the geometry
  fetch. São Paulo has **445** such donors at `admin=10`/`admin=9` against 109 inside its
  kept levels. A level needs `MIN_DONOR_HITS` (5) matches to be worth a fetch.
- Because the geometry cache is keyed by city but its contents depend on the level set,
  each cache file records `_levels`; a city is refetched when its donor levels grow.
  Without that, a city fetched before donors existed would be skipped forever.
- Borrowed outlines are marked `b: 1` on both the unit and the feature. A borrowed extent
  is a weaker claim than a surveyed one and **must not be drawn identically**.

**What borrowing cannot fix.** Copenhagen has **zero** same-named polygons anywhere in
its survey; Bangkok 2 of 578 nodes, Mumbai 17 of 781, Istanbul 24 of 737. OSM simply has
no outline for those places at any level. They need an external source (§1.3) or stay
pins.

**Shape availability must never feed back into level picking.** The rule answers "how is
this city divided"; preferring polygon levels because they look better would bias the
corpus toward whatever each country happens to draw. Borrowing keeps the two separate —
the rule picks, then shapes are attached wherever they can be found.

### 1.3b Who's On First — built, and it does NOT rescue the cities it was fetched for

`fetch_wof.py`. 427 MB of per-placetype bundles (the per-country SQLite builds are the
*admin* repos — India alone 1.5 GB — and carry countries and localities we already have).

**Result: 2,690 matches, of which 1,690 are new. Coverage 50% → 55%.**

**It fails on its target cities**, which is the finding that matters: Copenhagen 2% → 14%
(7 shapes), Mumbai 8% → 12% (31), Hong Kong 8% → 8% (0), Bangkok unchanged (1 match of 585
shapeless units). The binding constraint is that **349,709 of WOF's 413,374 neighbourhood
records have no polygon at all** — they are points. Only ~63,000 worldwide carry an
outline, and their distribution is US/NL/DE/ES-shaped, which is the same gap OSM has.

Where it genuinely pays is somewhere else entirely: **London +307, New York +176, Istanbul
+175, Chicago +121, Los Angeles +95, São Paulo +69, Prague +66, Seoul +63, San
Francisco +60.**

**`src:geom` predicts duplication, and that is the reusable lesson.** Where WOF's geometry
came from a municipal or national open-data release, OSM imported the *same official file*
— so WOF is not an independent source there, it is the same source twice:

| `src:geom` | matched | share new |
|---|---|---|
| quattroshapes | 729 | **98%** |
| mz (Mapzen), pedia, zolk, sfgov | 657 | 94–99% |
| whosonfirst | 790 | 34% |
| esp-aytomad, ssuberlin, esp-cartobcn, sg-sggov | 355 | **2%** |

So a city's value here is predictable from provenance rather than from coverage. Matches
at IoU ≥ 0.9 against an outline the corpus already has are dropped at build time
(`WOF_DUP_IOU`).

**Matching discipline held.** 497 units were rejected for failing containment, and **6,499
units sat inside a WOF polygon that did not share their name** — that second number is the
size of the mistake a name-blind rule would have made.

**Vintage: better than §1.3 assumed.** Bundles built 2025-10-30, and `wof:lastmodified`
clusters in 2023, not 2018. But treat that sceptically: a bulk reprocess bumps timestamps
without anyone re-examining a boundary, and the `src:geom` table says the outlines are
still largely Quattroshapes/Mapzen-era.

**Licence: linking back to the licence is REQUIRED** (crediting is merely recommended).
WOF is part original work, part modification of ~312 other datasets; the matched set draws
on 22 distinct `src:geom` values, all emitted into `wof_shapes.json` because that is the
list an attribution string has to cover.

### 1.3c Rivers are not neighbourhoods

Rendering the WOF output surfaced four London units named **River Thames** adopting a
21 km² WOF polygon of the same name. Name and containment both genuinely held; the river
was being drawn as a district.

The cause is a tag-level fact, not a naming coincidence: **borders follow rivers**, so
OSM's Thames ways carry `boundary=administrative` and `admin_level=6/8` *and*
`waterway=river` at once. They are boundary lines wearing a place's clothes. **813
elements corpus-wide** are both an administrative boundary and a water feature.

`is_water_feature()` in `pick_levels.py` excludes them, which removed 181 units from the
corpus. A tag test rather than a name blocklist, so it needs no per-city maintenance and
works in every language.

### 1.6 Per-city official sources — what to do when the generic fallback is spent

Who's On First (§1.3) is the generic answer and it does not reach the cities that need it:
measured, it gave **Cairo 0 shapes, Nairobi 0, Johannesburg 0, Lima 0, Hong Kong 0 and
Copenhagen 7**, because its 1,682 useful shapes went to cities that already had coverage.
So `fetch_external.py` names sources one city at a time. It does two things with each:
attaches a polygon to a unit the corpus already has, below; and promotes the polygons no
unit claimed to units of their own, which is **§1.7** and is where a city OSM has barely
mapped actually gains anything.

**There is no world source at this granularity, and that is the finding.** Overture's
divisions theme is OpenStreetMap plus geoBoundaries, so it cannot beat what we hold.
GADM forbids redistribution outright — verbatim, *"Redistribution or commercial use is not
allowed without prior permission"* — and draws Zamalek, a Nile island, as a pentagon.
What exists instead is national and municipal open data, in four formats with four naming
conventions.

| city | source | licence | shapes | coverage |
|---|---|---|---|---|
| Copenhagen | DAWA `/steder`, Klimadatastyrelsen | CC BY 4.0 | 53 | 2% → **96%** |
| Cairo | HDX/OCHA COD-AB, *kism*, from CAPMAS | CC BY 3.0 IGO | 25 | 11% → **59%** |
| Kuala Lumpur | DBKL *Sempadan Taman* | not stated | 25 | 24% → **53%** |
| Mumbai | Greater Mumbai revenue villages | CC BY 4.0 (asserted) | 62 | 3% → **16%** |

**Match rates are limited by GRANULARITY, not by the join.** Every source is a district
layer for the city proper, while the corpus spans the metro and goes finer. Kuala Lumpur's
misses are Bukit Jelutong, Subang and Kota Damansara — all Selangor, outside DBKL's writ by
design. Mumbai's are `place=neighbourhood`; it matches ~36% of `place=suburb` and 1% of
`place=neighbourhood`. Cairo's are sub-district: Kit Kat, Abbasiya, Mit Okba. Copenhagen
hits 96% precisely because its 56 units already sit at district scale.

**The join is name + containment, never name alone.** Denmark has seven Frederiksbergs and
none is Copenhagen's; `Amager` as a *bydel* exists only in Jutland. The same rule as
§1.3a's borrowing. Where nothing contains the point — an OSM node placed at a station just
outside its own polygon — a *single* candidate within ~2 km is accepted and two or more is
refused, which is what keeps the decoys out. Worth 39 units.

**A size guard copied from the WOF stage was wrong here and cost Copenhagen 13 matches.**
Rejecting anything over 12× the median polygon is right for a world gazetteer, where a
neighbourhood can match a country. It is wrong for a curated municipal layer in which
every polygon is a legitimate unit: the median is dragged down by the many small ones, so
Nørrebro (5.5 km²), Østerbro, Valby and Gentofte were all thrown out as "oversize". The
guard is now a fraction of the **city's own area**, which tests the failure that actually
exists — a unit named after its city adopting the whole outline.

**Licences are mixed and that was a deliberate call.** Copenhagen and Cairo are cleanly
licensed. DBKL states no licence (Malaysia's national open-data terms are permissive and
the service is deliberately public, but that is inference). Mumbai's CC BY 4.0 is the
publishing repository's assertion over data of unstated provenance. Two better-fitting
sources were **rejected on licence**: Tindak Malaysia's KL polygons (BY-NC-SA) and a Cairo
*shiakha* layer of 823 bilingual full-resolution polygons that sits on an individual's
ArcGIS account republishing copyrighted CAPMAS data.

**Rendered as its own claim** — `e` on the unit, a 4-2 dash, between borrowed and Who's On
First. The flag is `e` and **not `x`, because `x` is the unit's longitude**; an earlier
version silently rewrote the coordinate of every externally-shaped unit.

### 1.7 Units that are not in OSM at all, and why this is not a hole in the level rule

§1.6 attaches a source's polygon to a unit OSM already knows. The complementary question
is the one that actually reaches a city OSM has barely mapped: **does the source name a
district the corpus does not have at all?** For Cairo it names 26 — Abdeen, Ezbakeya,
El-Darb El-Ahmar, El-Sahel — none of which is a `place=*` node in the survey. So the
unclaimed polygons are promoted to units of their own.

| city | layer | in radius | already OSM's | **new units** |
|---|---|---|---|---|
| Copenhagen | DAWA `bydel` + `by` | 122 | 54 | **68** |
| Mumbai | revenue villages | 123 | 52 | **71** |
| Cairo | CAPMAS `kism` | 49 | 23 | **26** |
| Kuala Lumpur | DBKL `Sempadan Taman` | 333 | 25 | *rejected, see below* |

**The level rule is APPLIED, not bypassed.** This was the one thing worth getting right,
because §0b makes the level rule the browser's contract and every other level is chosen by
measurement. A hand-declared level would be a real departure. It turns out none is needed:
§3.1's division estimate — `city_pop / core units` — works on these layers unchanged, and
it discriminates on the first four sources tried.

| city | layer | core units | est. per unit | verdict |
|---|---|---|---|---|
| Cairo | kism | 33 | 306,060 | `keep-est` |
| Mumbai | revenue village | 104 | 119,230 | `keep-est` |
| Copenhagen | bydel | 41 | 16,097 | `keep-est` |
| Kuala Lumpur | taman | 327 | 5,504 | **below the 10k floor** |

**Kuala Lumpur being rejected is the result that justifies the whole approach**, because
the rejection is *correct* on inspection: among its 333 "taman" are `Pasar Borong Kuala
Lumpur`, a wholesale market, and `Kwsn Industri Batu Caves`, an industrial estate. It is a
housing-estate register, not a way the city divides itself, and the same 10k floor that
rejects OSM's over-fine levels rejects it without being told anything about Malaysia. The
source stays configured, with its verdict recorded, so that "we looked and said no" does
not read as "nobody tried".

**The denominator is the WHOLE layer, not the promoted remainder.** Copenhagen's estimate
divides by 122, not the 68 new units, because §3.1 asks how many parts the layer cuts the
city into and that number does not change because OSM happens to name 54 of them too.
Dividing by the promoted count would divide the city's population by the source's failure
to be joined — §3.4's "dividing by units nobody can see", with the sign flipped.

**Only the unclaimed polygons become units.** Emitting all 122 and marking the 54
duplicates was the alternative, and `mark_dupes` (§4.6) is the natural home for it, but it
would draw every matched polygon twice — once under the OSM record that already carries
that exact geometry via §1.6, once under a new record — for no information. The layer is
complete on screen either way; it is simply split across two level keys.

**Three gates, each earning its place:**

- **The radius**, the same `city.radiusKm` every OSM unit is held to. DAWA's mandatory
  bbox is bigger than Copenhagen: of 296 kept polygons only 122 are within 15 km, and
  without this 174 Zealand villages become Copenhagen neighbourhoods.
- **A per-source name filter**, for a layer that mixes two scales the way §3.4's Delhi
  does. Cairo's file puts urban قسم and rural مركز on one admin level, and `norm_arabic`
  strips both prefixes so that the *matching* can work — so only the raw name separates
  them. 15 markaz dropped: Ausim, El-Badrashein, El-Khanka, farmland administered from
  Giza rather than districts of Cairo.
- **An IoU backstop** at 0.9 against the city's own OSM outlines, the same threshold and
  reasoning as the Who's On First duplicate test, because a name *miss* is not proof of
  distinctness. It fires zero times on these four, which is the expected result for a
  working name join and is worth keeping as the thing that would catch a transliteration
  the join fumbles.

**The synthetic id is hashed from the raw name, never from `cfg["norm"]`.** §6 makes `i` a
join key that must survive a refetch, and the normalisers exist to make a *loose* match —
Mumbai's deletes compass words so that "Borivali East" reaches "BORIVALI" — which is the
one property an identity key must not have. Keyed on the normalised name, `PAHADI
GOREGAON-WEST` and `PAHADI GOREGAON-EAST` collapse to one id: two polygons, one record,
and one of the two outlines silently dropped. Measured, and now guarded by an assertion.
The key is the source's own id only where the source has a genuinely stable one — DAWA's
UUID and CAPMAS's `adm2_pcode` — and **not** DBKL's `objectid`, an ArcGIS row number that
renumbers on republish, nor Mumbai's file, which has no id property at all.

**The stage reads `base.json` and writes into it, so it has to be idempotent.** The units
created here are written back by `build.py`; left in the match set on the next run they
match the very polygons they were made from, mark them claimed, and nothing new is
created — the stage would work exactly once and then quietly stop. `src` is what excludes
them.

**Two flags, because they are two different claims.** `e` says the *outline* came from the
city's own source and is set on OSM units too; `src` says the whole *record* did and there
is no OSM object behind it. `src` is what explains a unit with no QID, no population and no
tags, and it names the source for the attribution CC BY requires — shown in the level
panel, not a credits block.

**Tiering is by hand, keyed on the source.** P31 cannot reach these units: there is no OSM
object to have carried a `wikidata` tag, so every rule in `assign_tiers` that reads a QID
reads nothing and the layer falls through to `unclassified` and is hidden. `SOURCES` names
the class the way `tier_map.json` names Q123705's, and the tier *number* is still derived
(§9.1) — Cairo's kism order against its OSM levels as `division-1`. Copenhagen's bydel are
declared `informal`, not `official`: DAWA is the national place-*name* register, so they
are the districts Copenhageners use rather than a kommune's administrative split.

**`known` is not measurable over these units, and that is different from being zero.**
§9.5 hides an informal tier below 50% `known`, the share of units with a Wikipedia article.
An external unit can never carry a QID, so counting it as unknown measures the absence of
an OSM join, not the obscurity of the place — Cairo's kism all have articles. This is
exactly the trap §9.5 already records in its other form, where New York's boroughs read 20%
because the sitelinks pass had not reached them. `known` is now computed over units that
*could* have carried a QID and is `None` when there are none, and the test that needs the
number does not fire without it.

**One consequence to eyeball: giving a city a division can hide its informal layer.**
§9.5's "something beats nothing" rescue turns on an informal tier only for a city with no
official division. Cairo had none, so its 69 `place=suburb` — 20% known, and the layer
holding Zamalek, Maadi and Heliopolis — were rescued and shown. Now `ext=kism` is a
`division-1` and the rescue no longer applies, so they start hidden behind a checkbox. That
follows the rule as written and may still be the wrong default; widening the rescue to
"no visible neighbourhood-scale tier" would change behaviour for all 181 cities and has not
been done.

### 1.3 Node-only levels: pins for now. Not Voronoi.

Many of the best levels have no polygons at all — Seoul's `place=quarter` and
`place=borough` are **0%** polygons, every unit a bare node; NYC's `place=neighbourhood`
is 15%.

**Voronoi cells were proposed and rejected.** They would make the units polygonal and
thereby unlock hex population, which is tempting, but a Voronoi cell is a claim about
extent that nobody ever made. Drawing one as a neighbourhood boundary is inventing data
and presenting it as surveyed. Not worth it.

So: **pins in the first iteration**, which the browser must support anyway. Real
polygons for these come later from **Who's On First**, whose `neighbourhood` placetype is
strongest in precisely the US cities where OSM is node-only. Low priority; it is a second
gazetteer to name-align and it has been unmaintained since Mapzen shut down in 2018.

Whatever supplies a shape, the UI must never render an inferred or borrowed extent
identically to a surveyed one.

### 1.6 The roster, 56 → 301, and the four ways Wikidata offers the wrong city

`roster.py` proposes seed cities from a WDQS query for everything that is a `Q515` (city)
subclass with a population and coordinates. It **never writes `cities_seed.csv`** — the
data contract (§6) keeps that file hand-maintained. It writes `data/roster_proposal.csv`
to be reviewed and pasted.

The query is the easy half: 1,297 cities at ≥400k. Choosing 245 of them is where the
decisions are.

**Breadth beats size, so the pick is round-robin by country, not a population sort.** A
straight sort down the list spends its first 310 picks on Chinese prefecture-level cities
before it reaches Dublin. Since the roster feeds a *name the city* quiz, the pass takes
each country's largest, then every country's second, and so on, capped by `--cap`
(default 6, already-seeded cities counted). At 245 that reaches **125 countries**, and
because the first pass is one-per-country it is essentially the world's capitals first.

**A candidate is rejected when its centre falls inside an already-accepted city's box.**
Not a spacing aesthetic: `fetch_osm` assigns each unit to its *nearest* seed city, so two
overlapping boxes do not yield two cities, they split one city's neighbourhoods down the
middle. This is what keeps Soweto out (20 km inside Johannesburg), and Kawasaki, Giza,
Quezon City and New Taipei City. It also costs real answers — **Yokohama at 29 km is
inside Tokyo's 30 km box** — and those are printed on every run so the losses are visible
and can be hand-added with a tightened radius if wanted.

**Then four distinct ways Wikidata hands you something that is not the city.** Each needed
its own answer; they are not one problem.

| | what it looks like | the answer |
|---|---|---|
| agglomeration | Greater Mexico City, New York metropolitan area | P31 class list (`Q1907114` and kin) |
| administrative twin | City of Belgrade beside Belgrade | name pattern **plus proof the twin is here** |
| not a settlement | Waqooyi-Bari, a federal member state of Somalia | flagged for review, rejected by hand |
| the number is simply wrong | Veracruz carrying 8,062,579 | rejected by hand |

`ADMIN_ONLY` matches a **deaccented** copy of the name, and on the stem `metropol` rather
than a list of spellings. Both were paid for: "Bordeaux Métropole" — an agglomeration of 28
communes — reached the roster and kept no levels at all, because `metropolis` does not
match `Métropole`. The next spelling would have been `metropolitana`. It was replaced by
Bordeaux the commune, which is below the 400k candidate floor and so had to be added by
hand.

The **administrative twin** is the one worth explaining, because the obvious rule is
wrong. Dropping every "City of …" and "… Municipality" loses Valencia, Spain, whose only
Wikidata record with a population is *City of Valencia*. So the test is containment plus
proximity — drop the administrative name only when the place it duplicates is also in the
pool within 25 km. "City of Belgrade" sits on top of "Belgrade" and goes; "City of
Valencia" is 7,000 km from the only other Valencia and stays. Names like
"… Metropolitan Municipality" are exempt from the proof requirement, because no city is
called that and their twins (eThekwini → Durban, Ekurhuleni → Germiston) share no text to
match on.

**The last two rows cannot be fixed by a rule and should not be.** `wdt:` is already the
truthy path, so the population is taken over best-rank statements only — where an item
marks one preferred, as Jeddah does among seven, that is the value used. But Veracruz *is*
the city item and its 2020 `P1082` is the state's 8,062,579; Masvingo (~90k) carries its
province's 1,638,528 as its only statement. Latest-date and best-rank both pick the wrong
number just as happily. They live in `roster.py`'s `REJECTED` dict with a reason each,
next to the Somali state and a Vietnamese province. **This matters beyond a wrong label**:
`pop` is the band denominator (§3), so an inflated figure keeps levels that are far too
coarse.

What survives is flagged, not silently trusted: any pick carrying none of the settlement
classes is printed as `CHECK THESE BY HAND`. That list is mostly false positives (Surabaya,
Incheon, Wrocław carry none of them) but it is what surfaced Waqooyi-Bari.

**A footgun, guarded.** `--propose` counts every row in `cities_seed.csv` as already
covered, so re-running it against a file that already holds a pasted proposal asks for the
leftovers — 245 additions come back as 68. `read_seed` now warns when it sees the paste
marker.

### 1.6a `--lint`: when `pop` and `radiusKm` describe different places

Both feed the division estimate — `pop` is the numerator, and `radiusKm` decides which
units are counted underneath it. When the two describe different areas the estimate is
wrong by exactly that factor, and it is **invisible**, because each column looks perfectly
reasonable on its own.

`python roster.py --lint` makes it visible by comparing the query box against a disc of the
city's `P2046` area. The ratio is an area ratio, so 8× means the box covers eight times the
city the population figure describes.

It is deliberately loose, because a box is *supposed* to overshoot the administrative city
and reach the built-up area around it. **Paris at 6.7× is correct** — its `pop` is the 20
arrondissements while the box has to reach the Petite Couronne. What the threshold is
hunting is the order-of-magnitude case.

Of 281 checkable rows, 33 flag. The two directions fail differently:

| | example | ratio | what it means |
|---|---|---|---|
| box ≫ city | Manila 1.85M over 25 km², box 18 km | **40.7×** | `pop` is the city proper, the box holds a metro of 14M |
| box ≪ city | Chongqing 32M over 82,403 km², box 30 km | **0.0×** | `pop` is a province-sized municipality the box cannot see |

The first understates unit size and drops levels; the second overstates it and keeps levels
that are too fine. Neither is fixable by the tool — the resolution is per-city, either a
metro population to match the box or a smaller box to match a city-proper population — so
the lint reports and stops.

**Its real value is the 245 new rows.** Nobody hand-checked those; `radius_for()` derived
every one of them from population alone, which is precisely the assumption this catches.
Where `levels.json` exists the lint also prints whether the city's boundary survived, since
a city with a good boundary is largely protected — `core%` re-restricts the denominator and
absorbs the mismatch. **Manila and Khartoum are flagged *and* rejected**, which is the
combination that actually hurts.

**`--lint` also reports cities too thin to browse** (`MIN_BROWSABLE_UNITS`, 15). Ten of the
built cities hold fewer than that — Blantyre and Córdoba have **four units each**. The
level rule cannot catch this: `MIN_UNITS` judges one LEVEL, and a 4-unit level is correct
where it is a city's borough scheme sitting beside richer levels (§3.3, New York's five
boroughs are exactly that). What is wrong here is the city TOTAL, which only a city-level
check can see. Four dots on a map is not a way a city is divided; it is what OSM happened
to have.

## 2. Pipeline

```
roster.py --fetch --propose     data/roster_proposal.csv   candidates, for pasting (§1.6)
   |
data/cities_seed.csv   hand-maintained roster (301 cities). The population denominator.
   |
   |  fetch_osm.py --pass survey     cache/survey/<qid>.json    `out tags center`
   |  fetch_osm.py --pass boundary   cache/boundary/<qid>.json  the city's own outline
   v
   |  pick_levels.py                 data/levels.json           which levels per city
   v
   |  build.py                      data/base.json             what the browser draws
   v
index.html  (python serve.py -> localhost:8766)

   later:  fetch_osm.py --pass geom   cache/geom/<qid>.json    `out geom`, kept levels only
           notability, de-duplication, real per-unit population
```

**The browser runs on the survey pass alone.** `out tags center` gave every unit a point
— a real centroid for ways and relations, the node itself otherwise — so all 20,131 units
are already mappable. Shapes are an upgrade to the same records (`poly` marks which units
will get one), not a prerequisite for looking, and looking is what tells us whether the
level picks are sane.

**Two Overpass passes over the divisions, and the cheap one runs first.** The level
decision needs names, levels and populations, never geometry, and geometry is ~1000× the
bytes. Overpass is donated infrastructure; the wasteful ordering would work and someone
else would pay for it.

**The geometry cache's staleness check must read the cache's CONTENTS, never the current
keep-list.** `cache/geom/` is keyed by city but its contents depend on which levels were
asked for, so each file records `_levels`. Files written before that field existed fell
back to "assume it held the kept levels of the day" — implemented by reading *today's*
keep-list, which is only correct while keep-lists never change. The moment the level rule
moved (`MIN_UNITS` 8 → 4, §3.3) the fallback silently asserted every old cache already
contained the newly-kept levels, so `--pass geom` reported **"0 to fetch"** while New
York's boroughs sat at zero shapes. It now derives the level set from the elements in the
file via `level_key`, which cannot go stale, and treats "cannot tell" as needing a fetch —
paying for a fetch is the safe direction, silently skipping one is not.

Politeness is duty-cycle based (sleep 2× the query's own runtime, floor 5 s) because
Overpass rations CPU-seconds, not requests. Timeouts are never retried — a deterministic
query that hit the server deadline will hit it again; the fix is a smaller `radiusKm`.
429 is honoured via `Retry-After`. **Do not run two of these jobs concurrently**; doing
so during development produced exactly the 429s the pacing exists to avoid.

## 3. The level rule

**Filter whole levels, not individual units.** If a level's typical unit clears the floor,
keep **all** of them; otherwise keep none.
Per-unit filtering would return a ragged set — Shibuya in, the next ward out because it
happens to sit under 10k — which is not a thing anyone would recognise as "how this city
is divided".

- **Floor: 6,000+, no ceiling** (10,000 → 6,000 on 2026-08-29, §3.4). Large divisions are
  wanted.
  What bounds the top end instead is `MIN_UNITS` — a level of 8+ named parts averaging
  10k+ people each is a reasonable definition of "a way this city is divided", and needs
  no ceiling.
- **Level key is raw**: `admin=9` or `place=suburb`, straight off the tags.
- **A unit with both** `boundary=administrative` and `place=*` counts as `admin` only.
  Counting it twice would let one set of polygons pass the rule under two names.
- **Guards:** ≥8 units to be judged at all (below that it is a borough scheme or a
  mapping artefact); ≥5 population-bearing units covering ≥25% of the level before a
  *reported-population* verdict.
- **Multiple levels per city is expected.** Paris keeps `admin=8`, `admin=9` and
  `place=suburb`. Overlap is acceptable; near-duplicates are a real problem, §4.6.
- `data/levels_override.json` replaces the keep-list for a named city. The rule is a good
  default, not an authority, and Anita is happy to hand-judge the handful of cities where
  it misfires.

### 3.1 The division estimate

When a level has no reported populations — most of the levels worth having — the fallback
is `city_pop / n_units`: if a level divides the city into *n* parts, the average part holds
that many people. No per-unit data needed.

It estimates the **average**, never any particular unit, so it can only support a
level-wide verdict — exactly the shape of the rule. It answers "is this the right *size*
of division for this city", which is the actual question.

Its verdict is **`keep-est`**, deliberately distinct from `keep`, because it is a weaker
claim and the report and override file should both see which levels rest on what.

**The denominator must be core units only.** The first version divided by every unit in
the query radius while the numerator was the city's own population. Those describe
different areas, and the error is exactly the factor by which the radius overshoots the
city — 5.5× for São Paulo, whose 30 km box covers the metro but whose population figure is
the *município*'s. Consequences were not subtle:

| | old (all units) | fixed (core units) | reality |
|---|---|---|---|
| Paris `admin=10` | 4,393 → rejected | **14,583** → kept | ~80 quartiers of ~26k |
| Tokyo `admin=9` | 8,914 → rejected | **11,731** → kept | 1,566 units, 94% wikidata |
| Amsterdam `place=quarter` | 9,387 → rejected | kept | the *wijken* |

The two "just missed the floor" near-misses of §4.5 were this bug, not a bad threshold.
They were fixed without moving 10,000.

Two gates, and both earn their place:

- `MIN_CORE_UNITS` (5) — enough core units for a ratio to mean anything.
- `MIN_CORE_FRAC` (0.15) — enough *of the level* must be core, or it is not this city's
  division at all. São Paulo's `admin=10` has 417 units of which 10 are inside the
  município; that is not the distrito level (~96 of those), it is the surrounding
  municipalities' own subdivisions, and dividing São Paulo's population by 10 of them
  yields a nonsense 1,150,000 per unit.

**`estPop` is a size class, not a population.** Never show it as one.

### 3.4 The estimate decides where it can; reported population only where it cannot

The precedence used to run the other way — *reported populations beat the estimate wherever
they exist* — which sounds unarguable, since a measurement should beat a division sum. It
was reversed on 2026-08-29 because **OSM's `population` tags are not a sample of a level.**
They are whatever somebody happened to tag, and that skews small.

**Aleppo is what the old rule cost.** `place=suburb` has 46 units for a city of 2,003,671,
an average of 58,931. Twelve carry a population tag, median 4,583, and on that dozen the
level was rejected — so Aleppo kept nothing at all and was absent from the browser. The
arithmetic settles it: at 4,583 each, all 46 suburbs would house 211,000 people, a tenth of
Aleppo, leaving 1.8 million living nowhere. A quarter of a level (`MIN_POP_FRAC`) was
enough to decide it, and a biased quarter beat sound arithmetic.

**But the gate could not simply be deleted**, and removing it outright is what proved why.
The estimate divides city population by units *inside* the city, so a level made of the
surrounding municipalities has no denominator — `MIN_CORE_FRAC` switches the estimate off
and `meanPop` is None. That is not an edge case; it is §4.4's deliberate design:

| city | level | units | core% | tagged | median |
|---|---|---|---|---|---|
| Paris | `admin=8` | 116 | 3% | 115 | 31,392 |
| New York | `admin=8` | 54 | 0% | 52 | 19,519 |
| Athens | `admin=7` | 41 | 2% | 41 | 61,308 |

Those are the Petite Couronne, Hoboken and Jersey City, the demoi around Athens — among the
best levels in the corpus, and a first attempt at this change deleted all three.

**The two failure modes are opposites, and the rule now matches that.** Aleppo's problem is
a thin, biased sample beating good arithmetic. Paris's is that there is no arithmetic to be
had. So the estimate goes first, and reported data speaks when the estimate is silent.

**One consequence in `mark_sparse`.** That check defends the estimate, so it may only
overturn a verdict the estimate produced; a level whose own reported populations clear the
floor is exempt. The old code got this free by testing `verdict == "keep-est"`, a name that
no longer exists. Without the exemption it fires on Paris `admin=9` — the 20
arrondissements, 22/22 tagged, 100% over the floor — because the `admin=8` above them is
the surrounding communes and outnumbers them. The two levels do not cover the same ground,
so comparing their unit counts is meaningless there.

**`keep-est` and `below-floor` are gone as verdict names.** Every keep now rests on the
estimate unless it says otherwise, and the strong/weak distinction moved to `estBasis`
(§3.1a) — core, disc or radius — which is a real statement about how much to trust a
number. "Somebody tagged a dozen of these" was not.

### 3.4a Lowering the floor to 6,000

At 10,000 the floor was cutting off real neighbourhood-scale divisions, and the tell was how
many cities sat just underneath it. **Dublin's `place=suburb` missed by 122 people and
Tunis's by 282**, and because it was each city's only candidate, both kept nothing and
vanished from the browser over a rounding error.

**Why 6,000 and not 9,000.** Dropping to 9,000 rescues Dublin and Tunis, and no further city
is rescued at any lower value — so the empty cities were never the real question. The
question is what sits in the 6k–10k band in cities that *already* keep something, and that
is **33 levels which are not marginal at all**:

| city | level | units | est | wiki | poly |
|---|---|---|---|---|---|
| Budapest | `admin=10` | 209 | 8,598 | 90% | 100% |
| Toronto | `place=neighbourhood` | 352 | 8,480 | 46% | 3% |
| Stockholm | `place=suburb` | 270 | 8,448 | 68% | 1% |
| Amsterdam | `place=neighbourhood` | 172 | 6,478 | 59% | 54% |

**Amsterdam closes the argument.** §4.5 records it keeping nothing because "its stadsdelen
average 123k against a 92k ceiling while its wijken sit just under the 10k floor" — it fell
out of the band at both ends at once. Dropping the ceiling fixed one end; this fixes the
other, and its buurten finally qualify.

A 6,000-person division is a neighbourhood by any ordinary reading, so this is less a
loosening than a correction of where the line was drawn.

**Combined effect of §3.4 and §3.4a, measured over 188 surveyed cities: 46 levels gained,
0 lost, cities keeping nothing 6 → 3 (Aleppo, Dublin and Tunis rescued), none newly empty.**
Strictly additive.

#### 3.1a Without a boundary, a disc of the city's Wikidata area

The denominator has three sources, best first — `core` (units inside the real boundary),
`disc` (units inside a disc of the city's `P2046` area), `radius` (every unit in the box,
the biased fallback). `estBasis` records which one a level used.

The **disc** exists because §4.4a rejects 8 boundaries outright and, at 301 cities, the
survey pass runs far ahead of the boundary pass — so "no boundary" is common, and without
a fallback every one of those cities reinstates the São Paulo error above.

**The insight is that the error is one of *scale*, not shape.** The numerator describes the
city, the denominator describes the box; correcting that needs the city's SIZE, and
Wikidata's `P2046` gives it for free. No shapefile, no download.

Measured against the real boundary on the 48 cities that have one, comparing the count a
disc yields against the true core count: **median 1.01×, 40 of 48 within 1.5×, 46 within
2×.** Worst cases São Paulo 2.27× and Dubai 0.05×, against a status quo whose error reaches
5.5×.

**It is a denominator only, and must never populate `coreFrac`.** The same measurement's
recall column shows a disc selects a materially different *set* of units even where the
count matches — Cairo 52%, Tel Aviv 44%. "This unit is in the city proper" is not something
a circle may assert; that is the objection that rejected Voronoi cells in §1.3.

Two properties that make it safe to have turned on:

- **It clears the same gates as `core`** (`MIN_CORE_UNITS`, `MIN_CORE_FRAC`), and when it
  cannot it falls through to `radius` rather than to no estimate. So the change is a strict
  improvement: corrected where the disc works, unchanged where it does not. **Dubai is why
  this matters** — its `P2046` is the historic core, its disc holds 5% of units, and it is
  discarded rather than believed.
- **Verified no regression.** Rerunning every surveyed city: boundary-having cities changed
  in *nothing*, and no verdict flipped anywhere. What moved was 17 estimates, all in
  boundary-less cities, led by **Manila `admin=7` 59,677 → 370,000 (6.2×)** and
  `place=suburb` 34,259 → 185,000 (5.4×).

That last result is worth stating plainly: **the fix corrects numbers that were badly
wrong, and today changes no decision.** Manila's estimates were 5–6× low but still far
above the 10k floor, so the verdicts held either way. It buys correct `estPop` in the
browser, and insurance for the 245 new cities where a level sitting near the floor will
eventually turn on it.

#### 3.1b P2046 is a quantity, and quantities have units

`wdt:P2046` returns a bare number. Harare's is **960,600,000** — square metres. Taking
`MAX()` over raw values across a roster therefore picks whichever statement used the
smallest unit, and the first version of `city_area.json` was full of cities the size of
continents. Units must be read (`psv:P2046` → `wikibase:quantityUnit`) and converted
before anything is compared.

Two further traps, both found by looking at the extremes of implied density:

- **Restrict to `wikibase:BestRank`.** Munich carries a stray 0.86 sq mi statement beside
  its preferred 310.71 km²; a naive minimum picks the stray.
- **Then take the max.** Among best-rank statements the small readings are sub-areas
  rather than the city — George Town has 305.77 km² beside a 109 ha.

An unrecognised unit is **skipped, never guessed**: a wrong factor produces a confident
number nobody would think to check. Cities with no usable area are written as `null` so
the cache can tell "no area exists" from "not fetched", and `--lint` need not refetch 301
rows to rediscover the 20 that have none.

### 3.3 `MIN_UNITS` was excluding the first division on purpose, and that was wrong

Lowered **8 → 4** (2026-08-29). The old comment said the quiet part out loud: a level of a
handful of units "is not that city's neighbourhood scheme — it is its **borough scheme**
(NYC has 5)". Under the four-tier framing (§9) a borough scheme is precisely the first
division a browser wants, so the guard was rejecting the tier, not the artefact.

New York's five boroughs were in the survey **three times over** and all three were
dropped here:

| level | n | wikidata | polygons | core |
|---|---|---|---|---|
| `admin=7` | 5 | 100% | 100% | 100% |
| `admin=6` | 6 | 100% | 100% | 83% |
| `place=suburb` | 5 | 100% | 0% | 100% |

Corpus-wide, **15 levels across 14 cities** sat at 4–7 units with ≥80% wikidata *and* ≥80%
polygons — Amsterdam, Delhi, Hong Kong, Kuala Lumpur, Lisbon, Madrid, Mumbai, New York,
Rome, San Francisco, Singapore, Sydney, Toronto, Warsaw. That is not a set of mapping
artefacts.

4 rather than 2, because a two-unit "division" (an east/west half, a river split) really is
an artefact, and because with the ceiling gone (§4.5) this guard is still the only thing
bounding the top end.

**Verified after the change:** New York now keeps `admin=7` = Manhattan, Brooklyn, Queens,
The Bronx, Staten Island, and `admin=6` = the six counties, correctly as a separate level.

**No `admin_level=5` widening is needed** and this was checked rather than assumed — the
boroughs were never outside the 6–11 window, only under the unit floor. Level 5 does
already appear for Singapore, whose five Regions arrive through the `place=borough` clause
and are then keyed `admin=5` by §3's "counts as admin only" rule, which is the rule working
as intended.

### 3.4 One `admin_level` can also mix two SCALES, and only a name rule reaches it

§9.4 splits a level whose units are two different *kinds* of thing. Delhi is the case that
neither that nor anything else general reaches: its `admin=9` holds 161 units that are 45
real Delhi areas (Vikaspuri, Karol Bagh, Rohini, Vasant Kunj, Saket, Najafgarh) and 116
numbered planned-colony sectors, most of them Noida's and Gurgaon's, at a completely
different scale.

**Three general signals were tried and all three fail**, which is why the blunt instrument
is justified:

- **`coreFrac` cannot**, because Dwarka's and Rohini's sectors genuinely *are* in Delhi —
  28 of the 69 core units are named "Sector N".
- **P31 cannot**, because the level is 3% wikidata; there is nothing to read.
- **Area spread cannot.** Tested as `p90/p10` of unit area across all 71 measurable tiers:
  Delhi's `admin=10` scores 37×, below Sydney's local government areas at **34,000×**,
  Stockholm's city districts at 64× and Hong Kong's at 45×, all of which are correct.
  Legitimate divisions vary in area enormously, so the metric has no usable threshold.

So `levels_override.json` gained `excludeNames`, a list of case-insensitive regexes
matched against the OSM name, **scoped to one city on purpose** — `^Sector` would be
catastrophic in Singapore or Warsaw, and the filter leaves the 39 "Sector *" units in
Lima, Madrid, Mumbai and Santiago alone. Applied at **ingest**, in both `pick_levels` and
`build`, so the level statistics and the browser agree; filtering at display only would
leave the division estimate (§3.1) dividing by units nobody can see.

Delhi also **drops `admin=10` rather than filtering it**, and the distinction matters:
stripping sectors still leaves 1,106 units called `DDA Flats`, `Pkt-B Sec-A`, `B-5`,
`Silver Oak Apartments`. Those are housing blocks, not areas, and no name rule rescues
them. The filter is for a level that is *mostly* right; a level that is wrong all the way
down needs dropping.

Result: Delhi 2,247 units → 824, and its `place=suburb` improved as a side effect, 284
units at 25% known → 181 at 40%, because the sectors were polluting that level too.

### 3.2 Sparse levels — the estimate's failure mode, and the cheap defence

The estimate assumes the level is *completely* mapped inside the city. Where OSM has only
some of the units, the divisor is too small and unit size is overstated. This was
originally written up as a mild bias "in the safer direction". Measuring against Kontur
showed it is nothing of the sort:

| Seoul level | estimate | Kontur | error |
|---|---|---|---|
| `admin=8` 행정동 | 21,709 | 19,549 | 1.1× ✓ |
| `place=borough` 구 | 361,538 | 348,679 | 1.04× ✓ |
| `admin=10` 통 | 34,686 | **451** | **77×** |
| `admin=11` 반 | 522,222 | **82** | **6,370×** |

OSM holds 285 of Seoul's ~13,000 통 and 18 of its ~100,000 반. Both levels were kept.

**`mark_sparse` detects this with nothing but unit counts.** `admin_level` is a *nesting*
hierarchy — every `admin=10` unit lies inside an `admin=8` one — so a complete finer layer
must have **at least as many units** as the coarser one above it. Seoul has 552 at
`admin=8` but 285 at `admin=10` and 18 at `admin=11`, which is arithmetically impossible
unless under-mapped. `place=*` gets the same treatment as an independent family, ordered
by the size ranking OSM documents: borough > suburb > quarter > neighbourhood >
city_block. The two families are never compared with each other.

Such a level becomes verdict **`sparse`** rather than being silently dropped, so the report
says why. Caught on the first run: Seoul `admin=10`, `admin=11` and `place=neighbourhood`
(57 units estimated at 522,222 each), and New York `admin=10` (13 units at 638,461).

This lives here rather than in the Kontur stage deliberately — **it needs no downloads and
covers every city**, including the ~34 with no Kontur data. Kontur found the problem; the
fix does not depend on it.

## 4. What the survey showed

21 cities surveyed. Full table via `py pick_levels.py --report`.

### 4.1 The rule works where populations exist

Paris `admin=9` — 22 units, median 98,742, **96% in band** → `keep`. The arrondissements.
Paris `admin=8` — 205 units, median 31,392, 92% → the Petite Couronne communes. Both are
defensible answers to "what is a Paris neighbourhood" and keeping both is intended.

### 4.2 Reported population is scarce, and the estimate is what rescues it

| city | level | units | with population |
|---|---|---|---|
| Paris | `admin=10` | 478 | **0** |
| Paris | `place=neighbourhood` | 273 | **0** |
| Seoul | `place=quarter` | 661 | 24 |
| New York | `place=neighbourhood` | 325 | 14 |

The levels most worth having are exactly the ones OSM has no populations for. Before the
division estimate they were all `unknown` and New York kept **nothing**. After it, New
York keeps `place=neighbourhood` (est. 25,538 each) and Seoul keeps `place=quarter` — the
661 *dong* — and `place=borough`, the 33 *gu*.

**This substantially reduces what Kontur is needed for.** It is no longer required to
pick levels; it is required to show a real population per unit in the browser. That is a
display feature, not a blocker.

### 4.3 Polygon coverage is very uneven, so the pin fallback is load-bearing

`polyFrac`, the share of a level's units that are ways or relations rather than nodes:

- Paris `admin=8/9/10`: **100%**.
- NYC `place=neighbourhood`: **15%**.
- Seoul `place=quarter` and `place=borough`: **0%** — every one a node.

Shape availability varies per *level*, not per unit, so it can be stated honestly in the
UI at the level being browsed. See §1.3 for why the answer is pins and not Voronoi.

### 4.4 Neighbouring municipalities are kept, and `core%` is how they are told apart

An earlier version rejected any unit outside the city's own administrative boundary. That
removed Hoboken and Jersey City from New York — and it was **wrong for this project**. A
compact core ringed by separately-incorporated municipalities that everyone treats as
neighbourhoods is a common pattern (the Petite Couronne, the comuni around Milan, much of
the Americas) and those municipalities are good quiz targets.

So membership is the **radius test only**, and the boundary became an annotation:
`coreFrac` per level, the share of units inside the city proper. Nothing filters on it.
It reads clearly:

- NYC `admin=8`: **0% core** — all 57 are New Jersey and Long Island.
- Paris `admin=8`: **2% core** — the surrounding communes.
- Paris `admin=9`: **86% core** — the arrondissements.
- NYC `place=neighbourhood`: **91% core**.

The radius is still doing real work: it keeps the Ogasawara Islands out of Tokyo, whose
administrative boundary is a prefecture reaching 1,000 km into the Pacific, and most of
Beijing's farmland out of Beijing. `coreFrac` also gates the division estimate (§3.1).

**Open question this raises:** if Hoboken is in New York's deck, the quiz answer for it is
"New York" — which is right in a metro sense and wrong in a civic one. Worth deciding
before the quiz is built.

### 4.4a Matching a city to its boundary by QID fails 25% of the time, systematically

Fetching each city's outline with `relation["wikidata"=<roster QID>]` returned **nothing
for 14 of 57** cities: Mumbai, Athens, Copenhagen, Stockholm, Lima, Santiago, Sydney,
Melbourne, Johannesburg, Nairobi, Istanbul, Cape Town, Cairo, Lagos.

That is not sloppy tagging, it is a category difference, and it is worth knowing about
generally because the same trap will appear in any OSM↔Wikidata join:

> **The roster's QID names the city as a *place*. OSM's boundary relation carries the QID
> of the *administrative entity*. Those are different Wikidata items.**

Measured directly with `is_in`:

| city | roster QID | what actually encloses the centre |
|---|---|---|
| Sydney | `Q3130` | `Q1094194` Council of the City of Sydney (lvl 6); `Q110046497` "Sydney" (lvl 9) |
| Cairo | `Q85` | `Q30805` Cairo Governorate (lvl 4) |

Tokyo (`Q1490` → 東京都) and New York (`Q60` → the lvl-5 relation) matched only because
their place item happens to be tagged on the administrative relation.

**The fix is geometric, not textual.** `is_in(lat,lon)` + `relation(pivot)` asks which
admin relations contain the city centre, which does not care how anything is tagged. The
QID query still runs first because when it hits it is the most precise answer; the
fallback runs only on a miss. Of the enclosing relations the **deepest admin_level wins**
as the most specific, restricted to levels 4–8 (9+ is sub-city, 3− is country-sized).
Tags are fetched first and geometry only for the chosen relation — a level-4 relation can
be an entire Australian state, and pulling four of those to discover we wanted the
level-6 would be gratuitous.

The chosen relation's name, level and QID are printed, because this rule will sometimes
pick something arguable and it should be visible when it does.

**Outcome over the 14: mostly right, three clearly wrong.** Good — Stockholm →
Stockholms kommun, Copenhagen → Københavns Kommune, Athens → Δήμος Αθηναίων, Cairo →
Cairo Governorate, Johannesburg and Cape Town → their metropolitan municipalities. Wrong,
all because "deepest" grabs whatever small unit the centre coordinate happens to land in:

| city | picked | what it actually is |
|---|---|---|
| Istanbul | Cankurtaran Mahallesi (lvl 8) | one neighbourhood in Fatih |
| Lagos | Shomolu (lvl 6) | one LGA of twenty |
| Nairobi | CBD division (lvl 7) | the central business district |

**So a bad boundary must not be allowed to act.** `MIN_BOUNDARY_TRUST` (10%): a boundary
containing almost none of the city's own candidate units is discarded and the city falls
back to radius-only. This matters beyond a mislabelled `coreFrac` — a too-small boundary
starves the division estimate's denominator and silently drops levels that should have
been kept. Verified: Istanbul and Sydney are the two surveyed cities it rejects. Sydney is
the interesting one — its City of Sydney council is a genuinely correct relation that is
simply too small to stand for a metro of thirty councils, so radius-only is the honest
answer rather than marking 95% of Sydney "neighbouring".

Anita's plan is a later case-by-case pass over these, which is the right call — this is
per-city knowledge, not a rule.

**The rejection is now recorded, not just acted on.** "No boundary was fetched" and "a
boundary was fetched and thrown away" are the same `hasBoundary: false` downstream, but
only the second can be explained to someone reading the map — and in all 8 surveyed cities
that warn, it is the second. So `trustworthy()` returns `(shape, reject)`, where `reject`
carries the discarded relation's name and level and the share of units it actually held,
and `pick_levels` puts it in `levels.json` as `boundaryReject` for `build.py` to pass on.

| city | rejected relation | held |
|---|---|---|
| Istanbul | Cankurtaran Mahallesi (admin 8) | 0.2% |
| Lima | Lima (admin 8) | 0.9% |
| Santiago | Santiago (admin 8) | 2.2% |
| Nairobi | CBD division (admin 7) | 3.7% |
| Melbourne | City of Melbourne (admin 6) | 5.9% |
| Lagos | Shomolu (admin 6) | 6.5% |
| Sydney | Council of the City of Sydney (admin 6) | 7.4% |
| Manila | Manila (admin 6) | 9.4% |

Manila is the near-miss worth noting — at 9.4% against a 10% floor it is the one city
where the guard is a judgement call rather than an obvious catch, and like Sydney the
relation it rejects is *correct* and merely too small for the metro around it.

The browser spends this on a hover note naming the relation and what it held (§6a.5).

**What a rejected boundary actually costs, since it is easy to overstate.** Membership is
unaffected: `locate()` decides `belongs` by the radius test alone, for every city, with or
without a boundary — that is §4.4's deliberate choice and the reason Hoboken is in New
York. Two things change. `core%` becomes unavailable, which is a label. And the division
estimate loses its denominator: with a trusted boundary it divides city population by
**core units**, and without one by every unit in the radius — the São Paulo error of §3.1,
understating unit size and pushing levels under the floor. **§3.1a is the answer to that
second one**: a disc of the city's Wikidata area now stands in, so a rejected boundary
costs `core%` and some accuracy rather than a broken estimate.

**"No boundary" is four states, not one**, and `boundaryState` separates them: `ok`,
`rejected` (fetched and discarded here), `not-fetched` (the pass has not reached this city
— the common case while the roster runs ahead), and `empty`. They are identical in
`hasBoundary` and must never read the same to a person.

### 4.5 The relative ceiling was dropped

There used to be a ceiling at 10% of city population. Because the floor was absolute and
the ceiling relative, the usable range scaled with city size: **9.2× for Amsterdam**
(10k–92k), **94× for Seoul**, **219× for Beijing**. That made the rule strict on small
cities and nearly toothless on large ones — backwards.

It showed at both ends. Seoul passed everything from an 18-unit level to a 661-unit one.
**Amsterdam kept nothing at all**: its *stadsdelen* average 123k against a 92k ceiling
while its *wijken* sit just under the 10k floor, so it fell out on both sides at once.

The rule is now a floor only, 10,000+. Amsterdam keeps its 17 *stadsdelen* (median
123,085, 94% wikidata, 88% core) and every one of the 21 surveyed cities now keeps at
least one level.

**The floor is now the only arbitrary edge, and two good sets sit just under it:**

| city | level | units | est. per unit | wikidata |
|---|---|---|---|---|
| Amsterdam | `place=quarter` (*wijken* — Jordaan, De Pijp) | 98 | 9,387 | 65% |
| Tokyo | `admin=9` | 1,566 | 8,914 | **94%** |

Both are ~6–11% under. Tokyo's `admin=9` in particular is 100% polygons and 94%
wikidata-tagged, which is not what mapping scaffolding looks like. Whether 10,000 is
right, or whether notability should be allowed to rescue a level that narrowly misses,
is worth revisiting once pageviews are in.

### 4.6 Near-duplicate levels

Seoul's *gu* appear as both `place=borough` (33) and `admin=6` (48); its *dong* as
`place=quarter` (661), `admin=8` (552) and `admin=10` (285). These are not overlapping
*different* divisions, which is fine — they are the same places twice, which is not.

De-duplication runs at build time (`mark_dupes`), keyed on the `wikidata` tag where
present and on name plus containment where not, unioned into connected components so the
answer does not depend on dict order. **30,098 units resolve to 25,258 distinct places.**

**A shared QID is not proof when the names disagree.** OSM sometimes tags a whole
district's children with the district's own QID, and the union is then catastrophic
rather than merely wrong: Mexico City's `Q6091026` sits on **49 differently-named
colonias** — Santa Isabel Tola, San Pedro Zacatenco, Coltongo — and merged all 50 records
into one "place". Singapore's `Q12684211` does the same to Boon Lay, Samulun, Shipyard and
Liu Fang. So a QID carried by ≥3 distinct names is refused and printed as a bad tag; the
cut is measured — 9,363 QIDs carry one name and 503 carry two (legitimately: Tokyo's
大字栄和 and 栄和 are one place with and without the 大字 prefix), and only 6 reach three.
Failing toward "leave separate" is the safe direction, since a bad merge deletes distinct
places while a missed one only leaves a duplicate dot. Largest component: 50 → 6.

**The browser draws duplicates; the quiz is what collapses on `dupOf`.** A version that
collapsed them on screen was tried and removed. In a tool whose job is to show what OSM
holds, a missing unit reads as missing *data*, and the reader cannot tell whether the gap
is OSM's or the browser's — a cost paid every time anything looks absent. The gain is also
smaller than it looks now that tiers exist, because duplicate records of one division
usually land in the same tier and switch off together (§9.2). The quiz has the opposite
requirement — one place must not be asked twice — and it is a different consumer.

Worth keeping because it cost real confusion: Toronto's `admin=8` shows 5 former
municipalities and **no Old Toronto**, which looks exactly like a de-duplication bug and
is not one. OSM has no `admin=8` Toronto at all; the city is `relation 324211` at
`admin_level=6`, `place=city`. The other 7 `admin=8` relations in the survey (Mississauga,
Brampton, Markham…) are outside the 22 km radius. Check the survey cache before blaming
the pipeline.

### 4.9 Self-reference: 15 units that ARE their own city

Found by sorting the corpus by sitelinks — the top of the list was not neighbourhoods.
Paris's `admin=8` commune (289 sitelinks) is a unit *of Paris*; likewise Los Angeles, São
Paulo, Barcelona, Milano, Manila, 大阪市, Montréal, תל־אביב–יפו, Casablanca, Santiago,
Lima at `admin=8`, and Sydney and Melbourne at `admin=9`/`place=suburb`. Vatican City
appears as a Rome `place=suburb`.

Detected as `wikidata == the city's own QID`, or a name equal to the city's.

**Marked `self: 1`, not dropped**, and the distinction is the §0b split doing its job. The
Paris commune is a genuine member of `admin=8` alongside Boulogne-Billancourt, so removing
it would make the level incomplete, which the browser must not be. Sydney's and
Melbourne's same-named units really are those cities' CBD suburbs, so a name match is not
even reliable grounds for deletion. **The quiz filters on this flag; the browser keeps
them.**

No unit turned out to be a *different* seed city, so §4.4's deliberate inclusion of
neighbouring municipalities is unaffected — Yokohama and Jersey City are kept on purpose
and are not seed cities.

### 4.7 `wikidata` coverage is a good free quality signal

The share of a level's units carrying a `wikidata` tag tracks "these are real, notable,
named places": NYC `place=quarter` 95%, Seoul `place=borough` 94%, NYC
`place=neighbourhood` 71%, Paris `admin=9` 100% — against 0–13% for levels that are
mapping scaffolding. Already collected as `wdFrac`. It is also the join key to pageviews,
so quiz-worthiness and joinability arrive together.

### 4.8 Coverage extremes

Tokyo returns **40,089** elements — Japan maps down to *chōme*, far below neighbourhood
scale. Beijing returns **1,867** for a city of 22 million. San Francisco returns 133.
Neither extreme is a bug; they are the ends of OSM's coverage range, and the level rule is
the mechanism meant to absorb exactly that. All three want eyeballing.

## 5. Open problems, ranked

1. **Eyeball the level picks in the browser.** It exists now and this is what it is for.
   Everything below is easier to judge afterwards.
2. **De-duplicate near-identical levels (§4.6).** The main blocker on a clean corpus.
   Measured: 7,839 units carry only ~6,000 distinct QIDs, so the duplication is real and
   countable.
3. **Notability: pageviews summed across all languages** (§7), joined on the `wikidata`
   tag. Needed for the quiz's 5–30 per city.
4. **Geometry pass.** Written, not yet run; run it once the keep-lists settle.
5. **Real per-unit population** via Kontur Boundaries (§1.2) — a display feature now, not
   a blocker.
6. **Case-by-case boundary fixes** (§4.4a) — the 8 cities the trust guard rejects. The
   browser now names the rejected relation (§6a.5), so each case states its own fix.
7. **Who's On First polygons** for node-only levels (§1.3). Low priority.
8. ~~A thin reported-population sample can outvote a sound estimate (Aleppo).~~ **FIXED,
   §3.4** — the estimate now decides wherever it can, and reported population only where
   the estimate is structurally silent.
9. ~~The floor is absolute, so small cities fail it everywhere (Dublin, Tunis).~~ **FIXED,
   §3.4a** — floor lowered 10,000 → 6,000.
10. **Cities too thin to browse.** Ten built cities hold fewer than 15 units, Blantyre and
    Córdoba four each. Reported by `roster.py --lint` (§1.6a); no decision taken on whether
    to drop them from the roster.
11. **Kuwait City has neither a boundary nor a `P2046` area**, the only city with nothing
    to scale its estimate against, so it divides across the whole 20 km box. Costs one
    level (`admin=7`, 382 units, est 7,824). A hand-entered area would fix it.

## 6. Data contract

| file | who writes it | tracked |
|---|---|---|
| `data/cities_seed.csv` | hand | **yes** |
| `data/levels_override.json` | hand | **yes** |
| `data/city_candidates.json` | `roster.py --fetch` | no |
| `data/roster_proposal.csv` | `roster.py --propose` | no |
| `data/city_area.json` | `roster.py --areas` | no |
| `cache/survey/<qid>.json` | `fetch_osm.py --pass survey` | no |
| `cache/boundary/<qid>.json` | `fetch_osm.py --pass boundary` | no |
| `cache/geom/<qid>.json` | `fetch_osm.py --pass geom` | no |
| `data/levels.json` | `pick_levels.py` | no |
| `data/external_shapes.json` | `fetch_external.py` | no |
| `data/base.json` | `build.py` | no |
| `data/units/<qid>.json` | `build.py` | no |
| `data/city_shapes.json` | `build.py` | no |
| `data/report.txt` | `pick_levels.py --report` | no |
| `cache/wiki/pop.jsonl` | `fetch_wiki.py --pass pop` | no |
| `data/pop.json` | `fetch_wiki.py --pass pop` | no |

Population fields on a unit in `base.json`, and they are **three different claims** that
must never be collapsed into one:

| field | source | |
|---|---|---|
| `p` | OSM `population` tag | 7% of units |
| `wp` | Wikidata P1082, §1.7 | 18% |
| `wy` | the P585 year of `wp` | absent where the claim is undated |

`estPop` on a *level* (§3.1) is a fourth thing and is a size class, not a population.

The root `.gitignore` excludes `data/` as a directory and git will not descend into an
excluded directory, so the local `.gitignore` un-ignores `data/`, re-excludes `data/*` as
children, then rescues the two hand-maintained files. Same three-line pattern as
`citybrowser` and `cityhistory`. Verify with the **exit code**, not the output — `-v`
prints the matching rule even when that rule is a negation:

```
git check-ignore -q neighborhoods/data/cities_seed.csv   # must exit 1 (tracked)
git check-ignore -q neighborhoods/data/base.json         # must exit 0 (ignored)
```

`external_shapes.json` is the one input that is also downstream of its own output:
`build.py` writes the units it creates (§1.7) into `base.json`, which is
`fetch_external.py`'s input on the next run. The stage skips units carrying `src` for
exactly that reason — without it the pass works once and then silently creates nothing.

`cities_seed.csv`'s `pop` is the **city-proper** figure because it is the band
denominator; a metro figure would make the 10% ceiling meaningless.

## 6a. The browser

`index.html`, served by `serve.py` on :8766. MapLibre with the OpenFreeMap basemaps,
dark and positron, one per theme (§6a.7) — the dark one is what `cityhistory` uses.

**Four payloads. Only the city list is loaded up front; everything about a city arrives
when it is chosen.**

| file | holds | when loaded |
|---|---|---|
| `data/base.json` | the INDEX — city and level stats, plus `levelKeys` | at startup |
| `data/city_shapes.json` | one outline per city, for the zoomed-out view (§6a.6) | at startup |
| `data/units/<qid>.json` | that city's units, as points | on city selection |
| `data/geom/<qid>.json` | that city's outlines, GeoJSON | on city selection |

Bundling outlines into the index would turn a startup into a ~180 MB one to draw shapes
for cities nobody has clicked. The join key is `i`, the OSM type-initial plus id
(`r123456`), so a geometry file stays valid however the level rule later changes — it
carries only ids, and level and colour are joined in the browser.

### 6a.8 Units moved out of the index, and why the colour list had to move in

`base.json` used to carry **every unit of every city**. That was survivable at 56 cities
and is not at 287: measured, **13.58 MB blocking the first paint** (2.17 MB gzipped, and
`serve.py` does not gzip at all, so locally it was the full 13.58 MB).

**The map only ever draws one city.** `visibleUnits()` returns `[]` with nothing selected
and otherwise filters `u.c === city`, so of the 84,019 units shipped before anything could
be drawn, the median city used **27 KB** of them. Splitting per city, exactly as the
geometry files already were:

| | before | after |
|---|---|---|
| upfront | 13.58 MB | **~350 KB** (36 KB gzipped) |
| per city on select | — | median **27 KB**, largest 614 KB (Tokyo, 3,174 units) |

**`levelKeys` had to move into the index, and this is the part worth remembering.** Level
colours are assigned in sorted key order so the same kind of division is the same colour
in every city — that is what makes two cities comparable by eye. The order used to be
derived by scanning all units at startup. With units loaded per city there is nothing to
scan, and deriving it from whichever city happened to open first is precisely the
instability the sorted warm-up existed to prevent. So `build.py` precomputes it.

Two smaller savings were measured and **not** taken, because the split makes them
pointless:

- **Omitting null/false fields** (`p` is null on 94% of units, `sl` on 85%, `core` on 74%)
  shrinks the units payload 23% raw but only 2.18 → 2.10 MB gzipped — gzip already
  collapses a repeated `"core":null,` to nearly nothing.
- **Trimming coordinates** from 5 dp (~1 m) to 4 dp (~11 m) saves ~150 KB raw.

Both are rounding errors beside a 97% cut, and each costs a field the data no longer
states plainly.

**One ordering hazard the split creates.** Geometry and units are now two independent
fetches for the same city, and geometry can land first — decorated against an empty
`byId`, every outline taking the fallback colour. `loadUnits` therefore re-runs `decorate()`
after populating the map. Both loaders also keep the `if (qid !== city) return` guard, so a
slow fetch for an abandoned city cannot overwrite the current one.

**`build.py` tolerates a half-written geometry cache.** A geometry fetch runs for hours and
a build during one is normal; `fetch_osm` writes each city in a single non-atomic
`write_text`, so a read can land mid-write. That city is skipped with a message rather than
taking the whole build down.

Outlines are simplified at `SIMPLIFY_DEG` = 0.0001° (~11 m), which is well under a pixel
at city zoom and cuts ~13 KB per polygon to ~650 bytes. **Display only** — anything
measuring these shapes must read `cache/geom/`, not `data/geom/`.

Drawing conventions that carry meaning and should not be casually restyled:

- **Solid dot = OSM has a real polygon; hollow ring = node only, will always be a pin.**
  §1.3 is why that distinction has to be visible rather than buried in the hover card.
  Between them sit the three third-party shapes at 0.45/0.40/0.35 — borrowed, city
  source, Who's On First — each with its own dash, because an inferred or second-hand
  extent must never render identically to a surveyed one (§1.3a). A unit from a city's
  own source (§1.7) has `poly: 0`: the field means *OSM* has a polygon, and the shape
  it does have arrives through `e`.
- Fill is faint (0.13) because several levels are drawn at once and overlap is expected.
- Hovering a dot outlines that unit in `LIT` — white on the dark map, near-black on the
  light one — so the card and the shape are unambiguously the same feature when districts
  overlap. Clicking pins that outline and opens the details panel (§6a.3).
- Level colours are assigned by level *key* and warmed in sorted order at startup, so the
  same colour means the same kind of division in every city. What is assigned is the
  *index*; the theme picks the palette it reads from (§6a.7).
- **Radius carries notability, and the unlinked floor is 0.6 of the base radius.** The
  `FAME` multiplier ramps `sqrt(sitelinks)` from 0.6 to 2.8 and is spliced into every
  output stop of the zoom ramp — never wrapped around it, which is a style validation
  error (§6a.1). The floor is 0.6 rather than 1.0 because 59% of units carry no
  `wikidata` tag at all: at full size that majority set the visual weight of the whole
  map, and the dots that mean something had nothing to stand out from. It is a floor and
  not zero because absence of a link is not evidence of obscurity, only of an untagged
  OSM object — an invisible dot would read as missing data.

### 6a.1 The load-order bug, and why the fix looks the way it does

`map.on('load', …)` was registered **inside** the `fetch` callback. `base.json` is ~3 MB,
so the map fired `load` first, and a `load` handler attached after the event has already
fired never runs — the map stayed completely empty with no error anywhere. It was caught
by screenshotting the page, not by reading it.

Two further things were learned fixing it, both worth not rediscovering:

- **`map.loaded()` is not a "has it fired yet" test.** Gating on it deadlocks: it can
  return false long after `load`, so the fallback listener never fires either.
- **The sidebar must not wait for the map.** The first fix made the city list depend on
  map readiness, which meant no basemap ⇒ no UI at all. The level panel and stats table
  are the point of this tool and need no WebGL, so data now renders the sidebar and
  selects a city immediately, and the map draws whenever it becomes ready. `draw()` and
  `applyShapes()` therefore guard on the sources existing.

**Headless Chrome CAN render the map here, with two flags that are easy to get wrong.**
The earlier claim in this file — that this laptop has no headless WebGL — was a flag
problem, not a fact about the machine. What works:

```
chrome --headless=new --enable-unsafe-swiftshader --remote-debugging-port=9333 \
       --user-data-dir=<a temp dir of its own> about:blank
```

- `--enable-unsafe-swiftshader` is the one that matters: without it there is no WebGL
  context at all. **Do not also pass
  `--disable-gpu --use-gl=angle --use-angle=swiftshader`** — that combination silently
  produces a black map, which is exactly what "no WebGL" looked like.
- **`--screenshot` together with `--virtual-time-budget` does not work for MapLibre.**
  Virtual time starves the render loop and the shot comes out black even when WebGL is
  fine. Drive the page over CDP and call `Page.captureScreenshot` instead.
- Always a dedicated `--user-data-dir`, so it is a separate instance that exits on its
  own and no window of Anita's is ever touched.

This matters more than the screenshot does: **MapLibre validates every paint expression
when the layer is added, and reports a bad one only to the console**, silently dropping
the layer. A CDP run reading `Runtime.consoleAPICalled` catches what
`tools/lint_map_expressions.js` can only approximate. Worth checking after any layer
change: `map.loaded()`, that every expected layer id is present, and
`map.queryRenderedFeatures({layers:['dots']}).length`. A 404 for `/favicon.ico` is
expected and is the only console error the page produces.

**The way round it: render the built outlines directly.** Plotting `data/geom/<qid>.json`
with matplotlib checks the thing that actually matters underneath the map — that the
polygons are real, correctly placed and correctly shaped — and needs no browser at all.
Done for Paris, New York, Tokyo and Seoul: the arrondissement spiral inside the Petite
Couronne, Manhattan through to the Rockaways, Tokyo Bay with `admin=9` dense across the
23 wards and `admin=7` reaching into Tama, and the Han River separating Seoul's *dong*
from its *gu*. Worth repeating whenever the geometry pipeline changes; it is a stronger
check than a screenshot of the map would have been.

### 6a.2 Hover must be coalesced to one frame, and highlight must not use a filter

Hovering a dot lights that unit's outline. Written the obvious way — `setFilter` on
`shape-hi` with the hovered id — it visibly **lagged the cursor**, replaying every
district the pointer had already crossed and reaching the right one only after the
pointer stopped. Two independent causes, and both had to go:

- **`setFilter` is not a cheap call.** It re-filters the whole source and rebuilds its
  tile buckets, on a source holding up to ~1,600 polygons. The fix is `feature-state`:
  `shape-hi` carries no filter, its `line-width` and `line-opacity` are a `case` on
  `["feature-state","hi"]` / `["feature-state","sel"]`, and a hover writes one attribute
  for one feature. **Feature ids must be numbers** — unit ids are strings like `r123456`,
  so `loadGeom` numbers the features 1..n as it decorates them and keeps
  `featId: unitId -> number` alongside. `promoteId` is deliberately not used; a promoted
  string id is not reliably addressable.
- **MapLibre delivers one `mousemove` per pointer sample**, and rendering each in order
  is what built the queue. `flushHover` runs at most once per animation frame and draws
  only the LAST sample of that frame. Everything in between is a frame nobody would have
  seen, so it is dropped rather than queued.

`setData` clears every feature-state on a source, so `applyShapes` re-applies the lit and
pinned outlines after calling it — otherwise toggling a level silently un-highlights
whatever is pinned.

The paint expressions in `shape-hi` are written out longhand rather than built by a
helper, because `tools/lint_map_expressions.js` `eval`s each `addLayer` literal on its
own and treats a layer it cannot evaluate as a failure. That is the right call: an
unchecked paint expression is the exact bug that script exists for.

### 6a.3 The details panel, and why it is not the hover card

The hover card cannot hold a link. It is `pointer-events:none` — it has to be, it sits
under the cursor — so the pointer would dismiss it on the way to one, and it is rebuilt
every frame, so it has to stay cheap. But a QID and an article title are only worth
showing if they can be followed. So a **click** pins a second panel that does take
pointer events and stays until dismissed, carrying three real links:

| link | built from | coverage |
|---|---|---|
| wikipedia | `wiki.json`'s `title` map, edition chosen by the rule below | 94% of QID-bearing units |
| wikidata | `q`, the OSM `wikidata` tag | 41% of all units |
| openstreetmap | `i`, whose first letter is the element type | every unit |

**Which Wikipedia to open**: English if there is one; failing that the edition whose
title *is* the OSM local name, which is the local-language article and nearly always the
fullest; failing that the largest edition present, by a fixed rank list. The row names
the edition it chose (`wikipedia · ja`), because which one you are being sent to is part
of what the link means.

**Exactly one article is linked, even where 37 editions exist** (Tokyo's 銀座). Listing
the rest was tried and removed: a wall of two-letter chips is not multi-language support,
it is the appearance of it, and doing this properly — which edition a reader wants, and
what the other editions are evidence *of* — is its own piece of work rather than a
by-product of the details panel.

**All three links are ordinary `dt`/`dd` rows in the same list as `osm level` and
`population`.** They were boxed callouts first, which made three identifiers look like
three actions. They are not actions; they are more of what is known about this unit, and
they read better sharing a list with the rest of it.

**`wiki.json` (3 MB) and `p31.json` (280 KB) are fetched lazily, once, on the first
click.** Neither belongs in front of the map's first paint: `base.json` already carries
the sitelink *count*, which is all the map itself draws, and the titles are wanted only
by a panel most readers never open. A second click while a fetch is in flight owns the
panel and the older one must not paint over it — hence the `selId !== p.i` guard.

Two things the panel shows that the card does not: `p31.json`'s "instance of" labels,
which are the same source the tier names come from (§9), and `wt` — the mis-tagged
article warning (§1.4) — as a plain sentence rather than a pill, because this is the one
place with room to say what it actually means.

**A unit with no `wikidata` tag simply has no wikipedia and no wikidata row.** An earlier
version explained the absence, with the corpus-wide share to say it was normal. That is
59% of units, so the explanation was on screen more often than not, saying the same thing
every time about the commonest state there is. A missing row already says it.

Because 45% of units are pins with no outline to light, and those are exactly the ones
whose panel a reader is likely to open, the selection also draws a white ring on the dot
itself (`dot-sel`). That layer's filter changes on a click, not on a pointer sample, so
`setFilter` is fine there.

### 6a.4 The map may hand the selection back to the sidebar

Panning past a city's own search radius, at z8 or closer, selects whichever city now
holds the view centre. Three things make that behave:

- **It is sticky, not nearest-wins.** Radii overlap heavily — 56 cities at 12–35 km, with
  several pairs inside each other's circle — so "closest centre" would swap the whole
  sidebar back and forth on a small pan near a boundary. Only *leaving* the current
  city's radius hands the selection on; while the centre is still inside it, nothing
  moves.
- **It does not re-frame the map.** `selectCity(qid, fly=false)`. The reader is already
  looking at where they want to be, and fitting the new city's bounds under them would
  undo the very pan that triggered the switch.
- **Our own camera moves must not trigger it.** `fitBounds` and `flyTo` end in a
  `moveend` indistinguishable from the reader's, so every programmatic move calls
  `quiet(ms)` first and `autoSelect` ignores `moveend` until that expires. Without it,
  flying *to* a city can select a different one.

Below `CITY_ZOOM` the map is showing whole cities as single shapes and one is picked by
clicking it (§6a.6), so a pan across a continent must not rewrite the sidebar. The two
rules share the constant deliberately: the zoom at which a city stops fitting on screen is
the same zoom at which "the city under the centre" stops being a thing anyone is looking
at. Open ocean claims nothing, and the current selection simply stands.

### 6a.5 Explaining a missing boundary, and why it is not an alert

There used to be a ⚠ beside such cities in the list. On its own it was a symbol with no
legend, and the card said only *No usable city boundary — `core` unavailable*, which names
a consequence and none of the cause.

**The icon is gone; the explanation stayed.** A rejected boundary is usually not a fault in
the city. Sydney's City of Sydney council and Manila's `admin=6` are the *correct*
relations, merely too small to stand for the metro around them — a city whose city-proper
limits are small is a normal fact about cities, not a defect. Eight permanently-flagged
rows read as "these cities are broken" when mostly they are not, and the genuinely wrong
picks (Istanbul's single *mahalle*, Nairobi's CBD division) cannot be told from the correct
ones by coverage alone: Lagos is wrong at 6.5% and Sydney is right at 7.4%. That
separation is per-city knowledge, which §4.4a already resolved to handle case by case.

The hover note went the same way and for the same reason. What is left is **one quiet line
in the city card**, next to the levels it actually bears on. Nothing in the city list marks
these cities at all.

`boundaryReject` (§4.4a) is what makes that line worth reading. Istanbul gets

> No usable boundary: OSM gave "Cankurtaran Mahallesi (admin 8)", only 0.2% of units.
> No `core`, and the size estimate runs over the whole radius.

Naming the relation is the point. "No usable boundary" is a dead end; *Cankurtaran
Mahallesi* tells you at a glance that OSM handed back one **mahalle** of Fatih, and it is
what a per-city override in `levels_override.json` would have to replace.

Two details that are deliberate:

- **The relation name is a raw OSM tag**, so it goes through `esc` like every other
  external string in the file.
- **The "none found" branch is written even though no city currently takes it.** Every
  such city today is a rejected boundary, but one whose relation is genuinely missing is a
  fetch away, and a line that silently said the wrong thing would be worse than no line.

An earlier draft of this note said units were *gathered by distance from the centre alone*.
True, but it implies a contrast that does not exist — that is how every city is gathered,
boundary or not. The real cost is the estimate's denominator (§3.1), and the difference
between naming that and naming the radius is the difference between a note that explains
and a note that misleads.

### 6a.6 Zoomed out, the map is a map of cities

Only one city's units are ever loaded, so zoomed past that city the map used to show an
unreadable smear of its dots and nothing at all about the other 180. That is backwards
from the question someone zooming out is asking, which is *which cities are in here, and
which one should I go to next.*

So `CITY_ZOOM` = 8.5 splits the map in two, by `minzoom` and `maxzoom` on the layers
themselves rather than by a zoom handler, which would show both sets or neither during a
gesture:

| | below 8.5 | at 8.5 and above |
|---|---|---|
| drawn | one shape + one marker + one label per city | the selected city's dots and outlines |
| hover | names the city and its unit count | the unit under the cursor (§6a.7) |
| click | selects that city and flies to it | pins the unit and opens the details panel |
| pan | changes nothing | hands the selection over (§6a.4) |

**The outline is the city clipped to its own search radius**, built by `build_city_shapes`
into `data/city_shapes.json` (125 KB for 181 cities, simplified at `CITY_SIMPLIFY_DEG` =
0.003°, which is about a pixel at the top of the range). The clip is not cosmetic: Tokyo's
`admin=4` relation reaches 1,000 km into the Pacific and Beijing's is mostly farmland, the
same oversized boundaries `locate` and §3.1 already have to defend against, and drawn raw
they would claim a corpus that stops at 25 km covers an ocean. The disc is exactly the
area that was surveyed, so the intersection is exactly the claim we can make.

**Cities with no trusted boundary get the convex hull of their own units, padded.** That is
131 of 181 today, because the boundary pass has reached 59 cities and the survey has
reached all of them. A hull is a blobbier shape, but it answers the same two questions:
where is this city, and is there anything here yet. `src` on each feature records which of
the two it is.

**A marker dot carries the low zooms, because the outline cannot.** At zoom 4 a 25 km city
is two pixels wide, which marks nothing a reader could aim at or click. The dot is drawn
from the city centre `base.json` already holds, is always big enough to hit, and fades out
between z6 and z8.4 as the outline grows into something worth looking at. Hover tests the
dot first and the outline second, in that order, for the same reason.

**The basemap's own settlement labels are pushed above `CITY_ZOOM`.** Otherwise every city
in the corpus is named twice a few pixels apart, once by us and once by OpenFreeMap. Ours
cannot be the one to give way: the basemap names every city there is, and the entire point
of this view is which of them the corpus *has*. So below 8.5 the only settlement names on
the map are the 181, while country and state labels stay, being context rather than
competition. Both basemaps are OpenMapTiles, so those layers are the `place` source-layer
with ids naming what they hold (`place_city`, `label_town`); anything unrecognised is left
alone, which degrades to the duplicate labels rather than to a blank map.

### 6a.7 Light and dark, and turning the dots off

Two changes with one motive: the first version read as an instrument panel, and this is a
map of where people live.

**The theme swaps four things at once**, because half of a theme is the basemap. The panel
variables, the level palette, the OpenFreeMap style (`dark` / `positron`) and the overlay
colours all move together; there is no useful state where the sidebar is light and the map
is not. It is remembered in `localStorage`, defaulting to dark.

- **`setStyle` is a teardown, not a recolour.** It discards every source and layer added
  to the old style, so the swap re-adds all of them and hands back what the map was
  holding: the city outlines, the dots, the selected city's shapes, the two feature-states
  that light one of them up, and the `dot-sel` filter. It passes `diff: false` so that is
  certainly what happens, rather than sometimes surviving in the old theme's colours.
- **Level colours are assigned as an INDEX, not a colour.** `keyIndex` maps a level key to
  a number once, warmed in sorted order at startup; the theme picks which of the two
  palettes that number reads from. A level therefore cannot change colour by being looked
  at in a different theme, and the light palette can be genuinely different ink rather
  than the dark one dimmed — a colour that carries on near-black does not carry on paper.
- **The shapes carry their colour baked into their properties**, so the swap redecorates
  them from the geometry cache (`decorate`). No refetch.
- **The `addLayer` calls keep literal hex** and are corrected immediately afterwards by
  `applyThemePaint`. That is not tidiness: `tools/lint_map_expressions` evaluates each
  layer literal in isolation and reports one it cannot evaluate as a failure by design
  (§6a.1), so a layer whose colours came from a lookup would be a layer it silently
  stopped checking. Same reason `CITY_FONT` is a top-level const array.
- **The city label's font is read off the basemap at runtime.** A `text-font` the style
  has no glyphs for renders as nothing at all, with only a console warning. Not the first
  stack found, either: positron's first is `Noto Sans Italic`, its water labels.

**Turning the dots off leaves the divisions as areas.** The dots stay the default, because
45% of units have no outline at all and the dot is the only thing that can stand for
those. With them off, the pointer resolves through `shape-fill` instead, and the rule is
**the smallest shape under the cursor wins**: levels are drawn together and a borough
contains its neighbourhoods, so of the shapes stacked at a point the smallest is the
specific answer and every larger one is context. Areas are a planar shoelace over each
outline in square degrees, computed once per city in `decorate`; they are only ever
compared within one city, where the missing cos(latitude) factor cancels.

Where there is no shape there is nothing to say, and the card closes rather than reaching
for the nearest dot. A pin-only unit has no extent, and pretending its centroid covers
this spot would invent the one fact §1.3 says we do not have.

> **`queryRenderedFeatures` takes a Point or an ARRAY, and anything else is options.**
> It tests `g instanceof Point || Array.isArray(g)`; a plain `{x, y}` is read as the
> *options* argument with the geometry defaulting to the whole viewport. It does not
> throw. It returns every rendered feature on screen, so the smallest-shape rule picked
> the smallest shape in London rather than the one under the cursor, and the card said
> *Chinatown* from anywhere in the city. `pickAt` passes `[x, y]`.

**Typography: one family, four sizes, sentence case.** There were two families, nine sizes
and uppercase letterspaced micro-labels. Each is legible on its own; together they are the
register of a control panel. The monospace column went with the rest — raw tags like
`place=suburb` are exact strings, but the sidebar shows each one beside its own
plain-English gloss, so the second typeface carried nothing the row did not already carry.
Colour lives only in `:root`, since a literal in a rule is a colour that stays dark on a
light page.

## 7. The quiz (not built)

A filtered view over the browser's corpus — see §0. Downstream on purpose: the browser is
what makes the data legible enough to know whether the quiz is worth building.

- **5–30 units per city**, cut by pageviews. Three tiers: top ~500 / top ~2500 / all.
- **Pageviews are summed across ALL language editions** (decided 2026-08-29). The audience
  is, as Anita put it, "the people that use Wikipedia" — so global readership is the right
  measure, and enwiki-only would badly under-rate Shibuya (mostly `ja` traffic) and
  Copacabana (mostly `pt`). Sitelink count comes free in the same Wikidata query and is
  worth storing alongside as a cheap prefilter.
- Type the city; drop a pin for partial credit, scored by distance decay from the
  centroid.
- **Name collisions.** Centro, Chinatown, Old Town, Downtown and Santa Cruz recur across
  dozens of cities. A name in **>3 cities is dropped**; a name in **2–3 cities accepts
  any of them**. Computed over the final deck, after the level rule, not over raw OSM.
- Reveal the shape after each guess — the payoff, and why §4.3 matters more here than in
  the browser.
- Unresolved: what the correct answer is for a neighbouring municipality (§4.4).

## 9. The four tiers, and why they come from Wikidata rather than the tags

The browser's user-facing scheme, Anita's: **cities/municipalities · first division ·
second division · neighbourhoods**, each named with the term that city actually uses.
`build.py assign_tiers` derives it; `data/tier_map.json` is the hand-maintained input.

**The OSM tag family cannot supply this.** `admin=N` vs `place=X` is a mapping
convention, not a statement about a place. Berlin's twelve Bezirke are `place=borough`,
Mexico City's sixteen alcaldías likewise, Amsterdam's stadsdelen are `place=suburb` —
while Tokyo's wards are `admin=7` and Paris's arrondissements `admin=9`. A first
implementation classified on the tag family plus `coreFrac` and mis-tiered roughly a
third of the cities checked by hand, Berlin, Amsterdam, Mexico City and Barcelona among
them.

**Wikidata `P31` does supply it**, and it supplies the label in the same breath:

| | P31 |
|---|---|
| Tokyo `admin=7` | special ward of Japan |
| Berlin `place=borough` | borough of Berlin |
| Amsterdam `place=suburb` | borough of Amsterdam |
| London `place=suburb` | area of London *(informal)* |
| Los Angeles `place=neighbourhood` | neighborhood *(informal)* |
| Seoul `admin=6` | city of South Korea *(a NEIGHBOURING city)* |

`fetch_wiki.py --pass p31` collects it in ~200 requests (§1.4), into its own ledger and
`data/p31.json` so it never races the 7 h views run.

### 9.1 Classes are mapped by hand; tier NUMBERS are derived

`tier_map.json` maps a type QID to `municipality` / `official` / `informal` / `skip`. It
does **not** store a tier number, and that is deliberate: Q123705 "neighborhood" is
Amsterdam's `place=quarter`, sitting *under* the stadsdelen, and also Los Angeles's
`place=neighbourhood`, which is the only tier LA has. The same type is a different tier
in different cities, so first/second division is computed per city by ordering that
city's `official` levels coarse-to-fine.

`byCity` overrides the global class for one city, because a type can be official in one
country and vernacular in another — Q188509 "suburb" is state-gazetted in Australia and
informal in Johannesburg, Cape Town, Nairobi and London; Q4286337 "city district" is
Warsaw's dzielnice but Toronto's municipalities dissolved in 1998.

### 9.2 Four rules, each found by being wrong first

- **Specific beats popular.** Giving a level its commonest type picks the *generic* one.
  London's boroughs carry both `London borough` (31) and `unparished area` (33) — and
  "unparished area" is a **negative property** meaning that part of England has no civil
  parish, not a kind of division at all. Stockholm's `city district of Stockholm` (11)
  lost the same way to a generic multi-country `quarter` (18). Ranking candidates by how
  few *cities* carry the type fixes both: a type that exists for one city is describing
  that city.
- **Levels sharing a type AND the same places are one tier.** New York's `admin=7` and
  `place=suburb` are both "borough of New York City"; numbering them division-1 and
  division-2 would re-create on screen the duplication tiers exist to remove. But sharing
  a *type* is not sufficient evidence on its own, and the counter-example is in the same
  city: New York's 19 macro-neighbourhoods (Harlem, Midtown) and its 325 finer ones are
  both Wikidata "neighborhood" and are **nested, not duplicated** — collapsing them loses
  a layer. Warsaw's 142 `admin=10` and 147 `place=quarter` are the same MSI zones twice
  and must collapse. Only the dedup components (§4.6) separate those two cases, so the
  merge needs the type to match *and* the places to coincide ≥60%.
- **An untyped level inherits from a same-division twin.** A level can be untyped merely
  because its units are thin on QIDs. Sydney's `place=suburb` is 8% wikidata but is the
  same 400 places as its `admin=9`, which is 99%. Where the dedup components (§4.6) agree
  ≥60%, the typed level names both. Worth 7 levels.
- **Count distinct places, not summed level counts.** A merged tier otherwise
  double-reports: Sydney's 400 suburbs, held twice, read as 804.

### 9.3 What it does not reach

**Of 190 levels: 120 tiered from P31, 43 inferred informal by position, 27 unclassified.**

The positional fallback closed most of the gap. Wikidata simply has no type for São
Paulo's 1,030 bairros, Tokyo's 1,569 `place=quarter` or London's 782
`place=neighbourhood`, but a `place=*` level with almost no QIDs sitting finer than every
official division the city has is a vernacular layer, and saying so costs nothing. It is
marked `inferred` and never given a label, because the label would be a guess where the
class is a deduction.

**`place=*` only**, and that restriction is the safety of the rule: an `admin_level` tag
is at least a *claim* to administrative standing — Paris's `admin=10` quartiers
administratifs really are official — while a `place=*` tag claims nothing. That is why 24
of the 27 survivors are `admin=*` levels (Delhi `admin=10`, Mexico City's colonias, Paris
`admin=10`) and are left alone rather than guessed at. The `place=*` survivors are ones
*coarser* than the finest official level, which the rule also declines: Sydney's 7-unit
`place=borough` is not a neighbourhood layer, and Istanbul's 736 `place=suburb` sit just
above its 791 mahalle.

**Reach: 37 of 56 cities have a first division, 43 have a neighbourhood tier.**

### 9.4 One `admin_level` can hold two tiers, so levels split on P31

`admin_level` is a national numbering, not a semantic one, and several countries put two
kinds of thing on one number. Tokyo's `admin=7` is the 23 special wards **and** the Tama
and Saitama municipalities, because a special ward ranks with a whole city; Seoul's
`admin=6` is its 25 *gu* plus Gyeonggi's *si*. Taking one type per level named each after
whichever group was larger and buried the other — Tokyo's wards lost to the cities and
surfaced only through a 31-unit `place=suburb`.

**Neither more sampling nor `coreFrac` can find this.** The mixing is categorical, not
geographic: Komae-*shi* sits inside Tokyo Metropolis, so it is a core unit exactly like
Shibuya-*ku*. Only P31 separates them.

So a level whose units divide into two classes, each ≥20% of the level and ≥5 units,
becomes two tiers. **5 levels split**, and the results are the point:

```
Tokyo   admin=7 -> 46 cities (狛江市, 立川市, さいたま市)  +  23 special wards (杉並区, 新宿区)
Seoul   admin=6 ->  7 Gyeonggi si (안양시, 고양시)        +  34 districts (gu)
Mexico  admin=6 -> 19 State of México municipalities     +  16 alcaldías (Iztapalapa)
Lisbon  admin=8 -> 51 freguesias                         +  25 former freguesias
```

Untyped and minority units join the largest group rather than being stranded; the error
is bounded, because a level only splits when its QID coverage is good enough for two
groups to clear the guards. **Splitting is presentation, not membership** — every unit
still appears, so the level rule's "all of a level or none" (§0b) is untouched.

**The tier is stamped on the UNIT (`tr`), not named on the level**, because a split level
has units in two tiers and a level-keyed mapping cannot say so. The browser therefore
carries two independent filters: `off` over level keys for the raw panel, `offTier` over
tier indices for the tier panel.

**The blocker this exposed:** `tier_map.json` was built from types that appeared as a
level's *mode*, so "special ward of Japan" and "city of South Korea" — minorities within
their level — were absent, and the split could not fire. Splitting reads types per unit
and needs the wider vocabulary; the map went from 87 to 116 types.

**Two counts are junk and marked `skip`** rather than classified: `unparished area`, and
`county of New York` — the five NY counties are coextensive with the boroughs and have no
separate government, so classing them `official` gave New York two identical tiers.

### 9.5 Which tiers start visible, and the one signal that decides it

Showing every layer of every city at once is the state that made the browser unreadable.
The right default is per city and cannot be a global level rule: Chinese `place=*` layers
are spotty and tiny and want hiding, while London, Paris, New York and Johannesburg all
want their informal layer on.

**The criterion is recognisability** — `known`, the share of a tier's units with a
Wikipedia article at all. It separates the cases cleanly with nothing else added:

| on | | off | |
|---|---|---|---|
| London *area of London* | 89% | Beijing `place=quarter` | 0% |
| New York *neighborhood* | 95%, 71% | Shanghai `place=quarter` | 0% |
| Paris *quartiers* | 50% | Tokyo `place=quarter` | 22% |
| | | Seoul, Chicago | 13%, 18% |

The rule: **every official division is always on; only informal layers are hidden.**
Municipality and every `division-N` start visible however deep, because an actual
administrative division of the city is never the thing a reader wants suppressed — that
includes Tokyo's 1,497 *chōchō*. Neighbourhood tiers start on only if `known` ≥ 50%.

**Unclassified levels follow the tag.** With no P31 type we do not know what they are, but
the OSM tag is itself a claim: a `boundary=administrative` level asserts administrative
standing (Paris's 379 `admin=10` really are quartiers administratifs) while a `place=*`
tag asserts nothing. So unclassified `admin=*` is shown and unclassified `place=*` is not
— the same asymmetry the positional fallback rests on.

**But "untyped" and "judged not a division" are different states, and conflating them
regressed a fixed bug.** New York's `admin=6` counties are typed, and that type is
deliberately `skip` (§9.3) because they are coextensive with the boroughs. Once
unclassified `admin=*` started showing, they came back on screen beside the boroughs —
the exact duplication that had already been noticed and fixed. A level is therefore marked
`skipped` when its dominant type maps to `skip`, and a verdict is never re-read as missing
evidence. Same fix hides Toronto's `admin=9`, which is federal electoral ridings.

Two clauses earn their place, both from a city that broke the simple version:

- **Something beats nothing.** Johannesburg's informal layer is only 13% known, but its
  only other tier is a set of units called *Ward 34*. A city with no official division
  shows its informal one regardless.
- **…but only the best one.** Toronto also has no official division and *three* informal
  tiers; rescuing every one turned on a 4%-known layer of 24 units beside the good one.
  The rescue applies to the highest-`known` tier only.

Third and deeper divisions start off because they are below neighbourhood scale — Tokyo's
1,497 *chōchō* are 97% known and still not what anyone means by a Tokyo neighbourhood.
Unclassified starts off because it is a holding pen, not a claim.

Result: **109 of 169 tiers on, 1.9 per city.** Nothing is dropped; every tier keeps its
checkbox and the raw level panel is untouched.

**`known` is only measurable over units that could have carried a QID.** A unit promoted
from a city's own source (§1.7) has no OSM object behind it and so can never have a
`wikidata` tag; counting it as unknown measures the absence of an OSM join, not the
obscurity of the place — Cairo's *kism* all have Wikipedia articles and would have scored
0%. `known` is therefore computed over non-external units only and is **`None`** where
there are none, and a test that needs the number does not fire without one. Same trap as
the paragraph below, in its second form.

**Note the knock-on: giving a city its first official division can hide its informal one.**
Cairo's 69 `place=suburb` (20% known — Zamalek, Maadi, Heliopolis) were on only through the
"something beats nothing" rescue. `ext=kism` is a `division-1`, so the rescue no longer
applies and they start behind a checkbox. That is the rule as written; whether it is the
right default is open, and widening the rescue to *no visible neighbourhood-scale tier*
would move all 181 cities, so it has not been done.

**This depends on sitelinks being complete, which is a trap.** `known` read 20% for New
York's boroughs — Manhattan and Brooklyn had simply never been fetched, because the
`MIN_UNITS` change (§3.3) admitted 291 new QIDs after the sitelinks pass had run. A
criterion calibrated on that would have been calibrated on a coverage gap. `load_wiki`
now supplements `wiki.json` from the append-only sitelinks ledger, so newly admitted units
are never silently unknown, and `fetch_wiki.py --no-merge` tops the ledger up without
touching the file the ~7 h views pass owns.

## 8. Running it

**`python` is correct again as of 2026-08-29** — plain `python`, `pip` and `py` all now
resolve to `AppData\Local\Python\pythoncore-3.14-64`, the interpreter that has geopandas,
shapely, pandas and cartopy. Verified.

Kept as a record because the failure was silent and could recur if anything reinstalls
python.org Python. There were two Python 3.14.3 installs: the python.org MSI at
`C:\Python314`, on the **machine** PATH with nothing in `site-packages` but pip, and the
Python Install Manager runtime on the **user** PATH. Windows searches machine PATH first,
so bare `python` and bare `pip` both hit the empty one — `pip install X` reported success,
wrote into `C:\Python314`, and `py` still could not import it. Fixed by uninstalling
`Python 3.14.3 (64-bit)` and `Python Launcher` from Installed Apps.

**The check, if imports ever start failing inconsistently:**

```
python -c "import sys; print(sys.executable)"
```

must print a path under `AppData\Local\Python`. If it prints `C:\Python314`, the decoy is
back.

```
python roster.py --fetch                   # WDQS candidates -> data/city_candidates.json
python roster.py --propose 245 --cap 6     # -> data/roster_proposal.csv, to paste (§1.6)
python roster.py --areas                   # P2046 -> data/city_area.json (§3.1a)
python roster.py --lint                    # pop vs radiusKm sanity (§1.7)
python fetch_osm.py --plan                 # zero requests, prints the query
python fetch_osm.py --only Q90,Q60         # smoke test
python fetch_osm.py                        # survey pass, resumable, cached
python fetch_osm.py --pass boundary        # city outlines, for coreFrac
python pick_levels.py --report             # the survey table, writes nothing
python pick_levels.py                      # data/levels.json
python build.py --stats                    # data/base.json + coverage summary
python serve.py                            # the browser, localhost:8766
python fetch_osm.py --pass geom            # geometry for kept levels (not yet run)
```

Standard library only so far, plus shapely (via `osmgeom.py`). The browser stage will add
geopandas. Both are already installed.

**Checking the browser without opening it.** Two tools, and they answer different
questions:

```
node tools/lint_map_expressions.js         # every layer's paint/layout expression, instantly
node tools/screenshot.js <url> out.png [waitMs] [probe.js] [w h scale]
```

The map half used to be unverifiable here — headless Chrome fell back to no WebGL, so
MapLibre never initialised and every screenshot was a black rectangle. It renders fine
with `--headless=new --enable-unsafe-swiftshader`, on software GL, slowly. The other half
of it is that `chrome --screenshot --virtual-time-budget` captures while tiles are still
in flight, so `screenshot.js` drives an already-running Chrome over the DevTools protocol
and waits in real time instead; its header has the launch command.

Its `probe.js` argument is evaluated in the page and its value printed, so it drives the
map *and* asserts against it — this is how the smallest-shape rule (§6a.7) was checked,
by sweeping a grid of screen points and confirming `pickAt` returned the smallest of the
stacked shapes at all 277 points where more than one was under the cursor.

**Do not run two Overpass passes at once** — every fetch is resumable and skips what is
cached, so an interrupted run is restarted by re-issuing the same command.

**Cost of the 245 new cities (§1.6), measured on Belgrade, Zagreb and Dublin.** The pacer
sleeps `max(5s, 2 × query time)` after each request, so wall-clock is roughly three times
query time:

| pass | per city | 245 cities |
|---|---|---|
| survey | 4–8 s query, ~18 s with the sleep | ~1.2 h |
| boundary | **see below — the estimate here was wrong** | |
| geom | heaviest; scales with kept units | several hours |

**The boundary estimate above was badly wrong, and the mistake is instructive.** It was
sampled from Belgrade, Zagreb and Dublin at 1–3 s each, which all matched their QID
directly. A city that misses costs **three** requests, not one — the QID query, the global
`is_in()` + `relation(pivot)` lookup, and a second fetch for the chosen relation — plus a
`MIN_SLEEP` between the last two. Measured over 225 fetched boundaries, **85 (38%) take
that path**, and on the post-expansion roster it is the common case rather than the
exception, because a QID match is far likelier for European cities than for the rest of the
world.

Wall-clock per city is roughly **3× its reported query-seconds** (the query, then a 2×
duty-cycle sleep), so a 45 s fallback city is over two minutes. Read the rate off the
running log rather than trusting a figure here.

**The fallback is worth its cost, which is worth stating since it is what makes the pass
slow.** Over the same 225:

| path | cities | boundary trusted | requests |
|---|---|---|---|
| direct QID hit | 140 | 139 (99%) | 1 |
| geometric fallback | 85 | 57 (67%) | 3 |

Two thirds of the expensive path yields a usable boundary — 57 cities that would otherwise
fall back to radius-only. The 28 failures are §4.4a's "deepest enclosing relation" picking
a sub-unit the centre happened to land in, and `MIN_BOUNDARY_TRUST` catches them.

Overpass returned a 504 on 1 of the 3 test cities and the built-in retry absorbed it, so
expect the survey pass to take longer than the arithmetic and to need re-issuing. Nothing
is lost when it does.
