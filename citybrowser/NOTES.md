# citybrowser — notes

Reference and decision log. **Anita skims this; she does not edit it.**
Claude maintains it: when something is decided, learned the hard way, or would
otherwise be rediscovered painfully, it goes here.

Her own task list is `TODO.md` — **do not edit that file.**

Contract for the data layers is in `SCHEMA.md`.

---

Interactive world city map, ~11k bubbles, hover card per city.
Part scraped, part hand-curated over months.

## ARCHITECTURE CHANGE: Wikidata is the roster, GHS is enrichment

Supersedes the earlier "GHS roster + Wikidata labels" plan. Two of Anita's
constraints — **primary key is Wikidata** and **nothing gets merged away** —
are the same decision, because GHS urban centres are density-contiguous blobs
and merging is intrinsic to how they are built. Wikidata already holds the
cities separately:

```
Guangzhou Q16572 18.7M · Shenzhen Q15174 17.5M
Dongguan  Q59218 10.5M · Foshan   Q34412  9.5M      GHS: one 43M "Guangzhou"
```

So: **one point per Wikidata settlement.** Key is the QID. GHS attaches to it
where a match exists, and contributes area, density, the 1975–2030 history, and
a nice card line ("part of a 43M contiguous urban area") — which is genuinely
interesting *as* a fact rather than a lie about population.

The Coyah/Conakry problem mostly evaporates: Conakry is `Q3733`, it is a point,
done. GHS blob 673 either attaches to it or does not.

### Consequences, all resolved

- **Bubble size** = metro population where it exists, else city proper. It is
  **not a population map** — size is a rough *importance* signal, so the
  city-proper/metro inconsistency does not matter.
- **Population floor: 10,000.** Refetched at this floor.
- **Elevation** deferred → see Someday.
- Wikidata is sparse in DRC / Somalia / Sudan (measured: 0.5–0.7 settlements
  per urban centre vs 6–60 normal). No fix available.

### The P31 filter (this one matters — it decides what is a map point)

Non-settlements are now *in the roster*, not merely candidates: `Flanders`,
`Benelux`, `Roman Catholic Diocese of Roermond` all carry population + coords.

A direct P31 test is not sufficient. In the Netherlands only **14** rows carry
`Q486972` directly, while **190** carry `Q1852859` ("cadastral populated place"
— real towns) and **501** carry `Q2039348` ("municipality of the Netherlands" —
an administrative division). Only the subclass tree separates them.

Solution: fetch the subclass closure of `Q486972` **once** (0.4s, 2,706 types),
then filter locally by set membership. Rows store their raw P31 values so a bad
closure never costs a refetch. Verified it keeps Amsterdam, Tokyo, NYC, Lagos,
Conakry, Guangzhou, Shenzhen, Hong Kong, Singapore, Vatican City, Monaco,
São Paulo — and drops municipalities, islands and neighbourhoods-as-admin-units.

Still open: `Q123705` (neighborhood) **is** in the closure, so "Haarlem-Centrum"
would become a point. Decide whether neighbourhoods stay.

## Matching = suggestion, never silent truth

Nothing suspicious is ever shown as a confident match. Below the bar the city
keeps its own identity and carries a visible "not confidently matched" state
with the suggestion attached — Anita reviews it. A wrong confident match is
worse than no match, because grey reads as "not done" while a wrong match reads
as "done".

Bar (all must hold): name agrees (exact / alias / normalised) **and** population
within **3x** **and** within a distance cut. Coyah/Conakry fails on name and so
lands in review, correctly.

## Edit-mode tools needed

Beyond field editing — none are blocking, build as they become possible:

- [ ] **delete** a city (wrong place / not real)
- [ ] **move** a city (bad coordinate)
- [ ] **create** a city absent from both sources — needs its own id namespace
- [ ] **manually match** to a GHS centre, or unmatch ← **now the blocking one.**
      `match_ghs.py` has produced a real review queue — 1,656 `near` guesses and
      1,110 `centre + low` population mismatches — and there is currently no way
      to accept or reject any of them. The card shows the warning and that is
      all it can do. A queue nobody can action is the same failure as no queue.

## Someday / low priority

- [ ] **Elevation.** Wikidata P2044 is only 16%. GHS `GE_ELV_AVG` is 100% but
      per-blob, so it cannot serve a Wikidata-keyed roster. Needs a DEM sampled
      at each point — same shape as the CHELSA climate sampling, do it then.

## GHS matching: DONE — `match_ghs.py`

`ghsConf` was `none` on all 61,866 rows until now; `assemble_base.py` only ever
set the default and no stage filled it. So area, density, the 1975-2030 history
and the member lists were all absent. They are in.

### The key that unlocked it: `GC_UCN_LIS_2025`

Every urban centre ships a semicolon-separated list of the places inside it.

```
id 10933 "Guangzhou" 43.0M / 6454 km2
LIS = Shenzhen; Guangzhou; Foshan; Dongguan; Jiangmen; Shunde; Zhongshan; ...
```

**Membership is stated outright, by name.** It does not have to be inferred from
geometry — which matters, because we have centroids and areas but no footprints
(those are only in the 1.69 GB GeoPackage), so an "is it inside" test would have
been a circle-approximation guess. This is not a guess.

That gives three roles, which `ghsConf` alone could not express. See SCHEMA.md;
the short version is `centre` (this city IS it) / `member` (this city is IN it)
/ `near` (a suggestion for review). Population is checked for `centre` only — a
member being 1/50th of its blob is normal, and being named in the list is far
stronger evidence than any population ratio.

### Results

```
centre + high    5,077     the city IS the centre
centre + low     1,110     name agrees, population does not -- the Fort Worth class
member + high    1,857     named inside someone else's blob
near   + low     1,656     unclaimed centre, best guess, awaiting review
none            52,166
```

6,187 of 11,422 centres (54%) have a city matching their main name, against the
51.5% NOTES measured on English labels alone — the alias pass is what closed the
gap. Chittagong now matches GHS's "Chattogram", which was a named miss.

**15.7% of the roster looks low and is not.** Only 11,422 urban centres exist,
and a centre needs 50k people at 1,500/km2. Most of the 61,866 Wikidata
settlements are small towns that are in no urban centre at all. The ceiling here
is roughly 11k + members, and we are at 9,700.

### `near` had to run BACKWARDS, and this is the real lesson

Asked forwards — every city looks for a blob it might be in — the answer is
junk: **29,833** suggestions at a 25 km cut. Tightening to one blob radius gave
9,931, still useless, *and* dropped Conakry, the one case NOTES demands be in
review (GHS names that 2.99M blob "Coyah", after a town on its edge).

Asked backwards — this centre has nobody claiming it, which nearby city is it? —
there is at most one answer per centre, and each is worth a look. 1,656 of them,
gated on population within 3x. Conakry lands: blob 673, 25.6 km, 2.99M against
1.67M.

A review queue of 29,833 is not a review queue.

### What is left unmatched, and why it is not a matching bug

3,663 of 11,422 centres have nothing attached — 12.8% of urban-centre
population. Measured, not assumed:

```
India   1217/1925 (63%)   Nigeria  301/425 (71%)   DRC      144/189 (76%)
China    810/1974 (41%)   Somalia   28/39  (72%)   Sudan     54/80  (68%)
59% of the unattached are under 100k
```

Three causes, none of them fixable by better matching:

- **The blob is not a city.** GHS "Maharajganj" is 2.84M over 1,120 km² with
  **zero** roster settlements within 30 km. It is contiguous dense-rural
  Gangetic plain caught by the Degree of Urbanisation rule, named after a small
  town on it. Nothing should match it. Same for Ponnani, Bibhutpur, Marhaura.
- **Wikidata sparsity**, already measured here at 0.5–0.7 settlements per centre
  in DRC/Somalia/Sudan against 6–60 normal. No fix available.
- **Roster gaps** — see Kaohsiung below. Upstream of this stage.

**Suffix stripping was tested and is not worth it.** Both directions:

```
strip " City"/" District" from GHS names   ->    3 new matches
strip " Municipality"/" Town" from ours    ->   59 new matches
```

and most of the 59 are Chinese `X Town` townships matching township-level GHS
members, which is below city scale anyway. 62 matches against the false
positives a fuzzy-suffix rule invites is a bad trade. Do not re-litigate this.

### ⚠ Kaohsiung is missing from the ROSTER — same class as the China bug

`Q170251` is not in base.json. Zero roster entries contain "kaohsiung". A
2.29M city, and only its *districts* (Fongshan, Sanmin, Zuoying, Nanzih) made
it in — so the type filter kept the subdivisions and dropped the city, which is
exactly the `Q1070990` county-level-city failure documented above, in a
different country.

This is a `kinds.py` / closure problem, **not** a matching problem, and the
check that found the Chinese one applies unchanged: list every dropped type
whose English label contains "city". Puducherry (1.06M) looks like the same
shape. Worth a pass over Taiwan's and India's settlement types.

## Review queue: BUILT — `js/review.js`, settings → Review

The 2,766 items `match_ghs.py` cannot decide (1,656 `near`, 1,110 `centre+low`)
now have somewhere to go. One at a time, map flown to the city, keyboard-driven
(**a** accept · **r** reject · **s** skip · **esc**) because a queue this long
that needs the mouse will never be finished. Biggest cities first.

Hidden in the settings panel deliberately: it is a batch job you sit down to do,
not something to trip over while curating a city.

**The two cases are asked as different questions**, because they are:

- `near` → "This urban centre has no city claiming it. Is *Conakry* it?"
- `centre+low` → "GHS names this centre *Chongqing* too, but its population is
  **3.8× smaller**. Is it still the same place?"

Answering the second as though it were the first is exactly how a 102k "Fort
Worth" gets confirmed as the real one.

### Two things this needed, both non-obvious

**1. Review must NOT set `_touched`.** Accepting a match is review, not
curation. If it counted, working the queue would light the map up with "curated"
rings for thousands of cities nobody has written a single fact about — and
"faded = not yet curated" is the entire progress signal. `REVIEW_FIELDS` in
**both** `js/data.js` and `build.py` (they must stay in step — SCHEMA.md).

**2. A rejection is `ghsConf = "none"` and nothing else.** The obvious version
also nulls `ghs`, and it is a trap: in the PATCH API `value: null` means *clear
this override and revert to base*, not *set the field to null* (`serve.py`
`do_PATCH`). Sending it would **delete the rejection** and the item would return
to the queue forever. Verified against the live server.

So `ghs` keeps pointing at the centre that was turned down, which is worth
having as the record of what was rejected — and everything downstream therefore
gates on `ghsConf`, never on `ghs` being set. `data.js` skips the centre join
and `card.js` skips both blocks when conf is `none`.

### Traps found

- **The themes disagree on encoding.** GENERAL_CHARACTERISTICS and GEOGRAPHY
  (V1-0) are real UTF-8; **GHSL (V1-2) is cp1252** and blows up a UTF-8 read at
  the first Mexican row (`México`). `open_csv()` sniffs and says which it found.
  Do not "fix" this with `errors="replace"` — that is how `Klaip?da` shipped.
- **The 1975-2030 population series is in the GHSL theme, not the two we had.**
  80 MB zip, `GH_POP_TOT_1975..2030`, twelve 5-year steps. 2025 and 2030 are
  **projections**; `charts.js` draws them dashed, because a projection rendered
  identically to an observation is a quiet lie.
- **One city can claim only one centre.** Wikidata's city/municipality
  duplicates (Amsterdam is Q727 *and* Q9899, populations within 0.5%) both come
  out as `centre + high` otherwise, and the card claims the centre twice. 381
  demoted to `member`.
- **River basin names are dated.** `GE_MRB_MAI_2025` says "Zaire" for the Congo,
  "Hwang Ho" for the Huang He, "Si" for the Xi. 189 distinct, and 4,575 of
  11,422 centres have none. Worth overriding by hand where it shows.

## Tile basemap: DONE, raster, on by default

`js/tiles.js`. Settings panel → Basemap. The coarse zoomed-in coastline is
fixed: at Tokyo you now get the real bay, the reclaimed islands and the Sumida
instead of one straight-edged polygon.

**Why it was cheap.** The map's projection already *is* the XYZ tile grid —
`data.js` stores `nx=(lon+180)/360` and `ny=mercY(lat)`, which is normalised Web
Mercator, so tile `(z,X,Y)` covers exactly `nx ∈ [X/2^z,(X+1)/2^z]`. Nothing to
reproject. Level is `round(log2(k/256))` where `k` is world-width in CSS px.

**Raster, not vector.** Vector tiles mean MapLibre, and MapLibre wants to own
the canvas — which takes the draw loop, the density thinning and the hit-testing
with it. Raster is one `drawImage` per tile into the canvas `map.js` already
owns, and all three of those stay untouched.

**The geojson layer stays** as the under-layer. It is local, so it paints
instantly and works with no network — that is what stops a slow or absent tile
fetch from showing bare water. `tiles.covered()` skips it once every visible
tile is decoded, since it is then invisible anyway and costs ~50k path ops on
every mousemove of a drag.

Details worth not rediscovering:

- **Seams.** Round the *shared edge* of adjacent tiles (`round(sx(x/n))` and
  `round(sx((x+1)/n))`), never a position and a width independently — that way
  both neighbours agree on the boundary by construction and no hairline gaps
  appear in the fill.
- **Parent fallback.** A missing tile is drawn from whatever ancestor is already
  cached (up to 5 levels), so a zoom reads as "sharpening" rather than "blank,
  then a flash". The parent level is also prefetched.
- **No longitude wrap**, deliberately. The city dots do not repeat, so a
  repeating basemap would show land with no cities on it and read as a bug.
- `@2x` tiles when DPR > 1; `MAX_INFLIGHT` caps a fast zoom at 16 GETs.
- Attribution is a **licence condition** on every source here, not decoration.

**Style choice: `light_nolabels` is the one to use.** Positron's greys sit under
the population ramp without competing. Tested and rejected: **Voyager**, whose
amber roads are the *same hue* as the mid-size-city dots — the roads read as
cities. **With labels** works but the basemap's own city names fight the
bubbles, which are the entire point of the map.

Known cosmetic break: Positron water is grey (`#d4dadc`), the palette's is light
blue (`#d8e6f2`). Only visible at the edges and before tiles load. Left alone —
tinting tiles per-frame is not worth it.

Default is **on** (`light_nolabels`). Off was the first default and it was the
wrong one: with tiles off the zoomed-in coastline is visibly wrong, so off is
the worse first impression rather than the safer one. Offline is unaffected —
the geojson layer underneath is local and paints whenever no tile arrives.
Choice persists in `localStorage`; `?basemap=<id>` overrides it, the same way
`?city=` and `?q=` do, which is also how it gets screenshot-tested.

## Decided

- Scope: all ~11k GHS-UCDB urban centres. Hand-added historical / isolated cities later.
- Two-layer DB: `base.json` (generated, never hand-edited) + `overrides.json`
  (field-level hand patches, each recording the value it replaced) → `cities.json`.
- Own stable ids (`c00001`), crosswalk to GHS id + Wikidata QID so ids never shift.
- Edit mode is local: `serve.py`, open map at `?edit=1`, PATCH writes `overrides.json`.
  Curation lives in git. No hosting, no auth.
- Untouched cities render gray, so progress is visible on the map itself.
- No lead paragraph, no page thumbnail. Facts are read and written by hand.
- Not optimising for speed anywhere. Slow and polite is fine.

## Open questions

## GDP per capita: settled, but coverage is thin

**Hard pipeline rule: never divide a source's GDP by GHS population.** Always
take GDP *per capita* directly from one consistent boundary. Mixing numerator
and denominator across definitions gives 2–4x errors on merged Chinese/Indian
blobs and on fragments like Fort Worth. Within one boundary the error is 5–25%
and defensible. The build should refuse to compute a ratio across sources.

Fallback chain, all CC BY 4.0:

1. **OECD Functional Urban Areas** — 562 FUAs, 33 countries, GDPpc direct in
   USD PPP, latest 2020–2023. SDMX bulk CSV:
   `https://sdmx.oecd.org/public/rest/data/OECD.CFE.EDS,DSD_FUA_ECO@DF_ECONOMY,/all?format=csvfile`
2. **Eurostat metro regions** (`met_10r_3gdp`, ~243) then **NUTS3**
   (`nama_10r_3gdp`, 1,343) — per-inhabitant direct. DBnomics mirrors it.
3. **OECD TL3** (2,612 small regions) / **DOSE v2.14** (1,667 admin-1 regions,
   83 countries, Zenodo 20035157) — best reported non-modelled global set.
4. **Kummu et al. 2025** — note this ships **GPKG/CSV at admin-2, 43,501 units**
   (Zenodo 13943886), not just a raster. Join UCDB centroid to the admin-2
   polygon; that sidesteps the raster-footprint ambiguity entirely. This is the
   global backstop.

**Coverage reality:** ~900 centres confirmed today, 2,000–3,500 optimistic, out
of 11,686. OECD countries hold only 1,685 centres (14.4%). China (1,974) and
India (1,925) are 33% of the roster with essentially no dedicated city GDP, and
nor do Nigeria (425), Pakistan (299), Bangladesh (239), Egypt (214), DRC (189).
So GDP is a sometimes-field. The card must look right with it absent.

### National statistical offices: surveyed, and mostly not worth it

The dedicated-source path has a sharp cost curve. Where to stop:

**Worth doing — one download each, genuinely sub-provincial:**
- **Brazil** IBGE PIB dos Municípios — 5,570 municípios, 2023, per capita
  direct, **no API key**. `apisidra.ibge.gov.br/values/t/5938/n6/all/v/37/p/2023`
  (table 5938; table 21 is discontinued). Bulk is over `ftp://`, not https.
- **Indonesia** BPS PDRB — 514 kabupaten/kota, **2025** (6-month lag, fastest
  anywhere), per capita direct, free key. Note `www.bps.go.id` 403s fetchers;
  use `webapi.bps.go.id`.

**Not worth it for this project:**
- **China** — the City Statistical Yearbook is a ¥358 print book, no free
  download. `data.stats.gov.cn` carries GDP for just **36 cities** and 403s from
  outside the mainland. The real route is **31 provincial bureau scrapers**.
- **Japan** — 市町村民経済計算 is not published nationally (0 hits on e-Stat).
  47 prefectural sites, Excel/PDF, one prefecture suspended since 2018.
- **India** — no city or municipal GDP from any official source, at all. Only
  26 of 36 states compile district product, with differing methodologies.
- **Korea** 17 provincial KOSIS orgIds; **Mexico** has no municipal GDP (VACB is
  establishment-based, not GDP); **Nigeria** has 22 states, 2013–2017 only.
- Turkey / Russia / South Africa / Vietnam are **province-level** — no finer
  than what OECD/DOSE already give us.

**⚠️ US changed under us:** BEA **discontinued all metro-area statistics** with
the Feb 2026 release. `MAGDP*` now error. Only GDP-by-county survives (3,127
units), per capita is no longer published, so an MSA figure means aggregating
counties over an OMB CBSA crosswalk and dividing. OECD FUA already covers US
metros with per capita direct — use that instead and skip BEA.

**So the GDP plan is: OECD FUA + Eurostat met (~700 rich-world metros, two
CSVs) → optionally Brazil + Indonesia later → Kummu admin-2 for everything
else, flagged as modelled.** Do not build country scrapers for this.

**Do not use the Oxford Economics GCFS sample** (900 cities) found in a
third-party GitHub repo — it is proprietary IP redistributed with no licence
grant. Brookings GMM18 (300 metros, 2016) states no licence either and its
underlying data is also Oxford Economics; cross-check only, never ship.

**Wikidata is not a GDP source**: only 712 items carry P2131 at all, of which
**6** are instances of city. That path is dead.

**GHS-UCDB ships its own GDP** (`SC_GDP_SUM_*`, from Kummu) — convenient, but
it is national GDP redistributed over grids, so it misses the urban
productivity premium: it puts Bangkok above London. Same shape of problem as
population. Fine as a consistent fallback, wrong as a displayed figure.

## GHS re-export: done

Fetched directly from the JRC open-data FTP — no manual download needed. The
themed sub-packages are small and public; the 1.69 GB GeoPackage is only needed
if we ever draw footprints.

```
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_THEME_GLOBE_R2024A/
  GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A/V1-0/...zip               7 MB
  GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A/V1-0/...zip 6 MB
  GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A/V1-2/...zip          31 MB
```

- Mojibake **gone**: 0 bad names in the fresh export vs 329 in the local copy.
- Comma-delimited, real UTF-8, no thousands-separator trap.
- **Coordinates found** in the GeoPackage's `UC_centroids` table as
  `PWCentroidX/Y` — *population-weighted* centroids, better than geometric.
  ESRI:54009 Mollweide metres, reprojected to WGS84 → `data/ucdb_centroids.json`.
  Validated: Quito 4.8 km, Reykjavík 4.7 km, Tokyo 3.7 km from city centre.
  Guangzhou is 44.9 km off, which is the merge blob doing exactly what we
  expect. Confirms the plan: GHS centroid for **matching**, Wikidata P625 for
  **display**.

**Schema changed — do not mix with the old copy.** Fresh export keys on
`ID_UC_G0` with 11,422 rows; `cityhistory/data/ghs_ucdb.csv` keys on
`ID_MTUC_G0` with 11,686. Different vintage, different ids.

- [ ] Gitignore `citybrowser/data/` — source zips and derived intermediates,
      all regenerable. See [[project_maps_untracked]].

## Population: GHS-UCDB is NOT the displayed number

Checked against the raw file. It merges and splits inconsistently, so there is
no offset a reader could learn (ratios vs Wikipedia swing 0.6x–3.2x):

| GHS row | pop | area km² | what it actually is |
|---|---|---|---|
| "Guangzhou" | 42,987,704 | 6454 | Shenzhen + Guangzhou + Foshan + Dongguan + … |
| "New Delhi" | 31,422,508 | | Delhi + Ghaziabad + Gurugram + Noida |
| "Fort Worth" | 102,173 | 60 | a fragment |
| "North Richland Hills" | 523,625 | 268 | ← the actual Fort Worth blob |
| "Atlanta" | 602,708 | 292 | dense core only, metro is ~6M |
| Hong Kong | 4,807,599 | | split 3 ways, official is 7.5M |

**UN WUP 2025 is not an escape hatch.** As of the Nov 2025 revision the UN
dropped national administrative definitions and adopted the GHSL Degree of
Urbanisation rule (contiguous ≥1500/km², total ≥50k) — the *same* rule GHS uses.
So WUP 2025 and GHS-UCDB are the same family, not independent sources, and
`cityhistory/data/stadester/wup2025.json` inherits that. UN WUP 2025 puts New
York at 13.9M, down from 18.8M in WUP 2018, against an MSA of 19.9M.

**Resolution: separate the two jobs, because they have opposite requirements.**

- **Bubble size** needs cross-city *consistency*. Nobody eyeballs a bubble and
  says "that's 6M too small." → **GHS urban centre.**
- **The displayed number** needs to match what the reader would find if they
  googled it. → **Wikidata/Wikipedia, with its definition stated.**

So the card shows a small labelled set — city proper, metro, urban centre —
which is what a Wikipedia infobox does anyway, and no single figure has to carry
the weight. GHS's member-city list goes on hover so "Guangzhou 43M" explains
itself. Any of them can be overridden.

GHS keeps three jobs it is genuinely good at: the 11,686-row **roster**,
**area/density**, and the **1975–2030 recent history**. Merge/split/relabel
mistakes get fixed through the override layer as she curates.

- [ ] Re-export GHS from source — the local CSV is mojibaked (329 names with
      `?`/`�`: `Klaip?da`, `Timi?oara`) **and** has no coordinates. One fresh
      download fixes both. Note: raw values use `.` as thousands separator.

## Climate: GHS shipped, CHELSA still needed for the band

The GHS **CLIMATE** theme is in (19 MB zip) and applied per blob, which for
climate is a far smaller compromise than for elevation — annual mean temperature
over a 300 km² blob really is close to uniform.

What it gives: **Köppen class**, annual mean temperature, annual temperature
range, annual precipitation. Plus Köppen projected to 2040 and 2070 under seven
SSP scenarios; the card uses **SSP2-4.5**, the middle one, so it does not
implicitly argue a best or worst case.

**2,021 of 7,756 centres change Köppen class by 2070** — Los Angeles Csa→BSh
(Mediterranean to hot semi-arid), Paris Cfb→Cfa, Seoul Dwa→Cwa, Dhaka Aw→Am.
That is the most striking single fact GHS has handed us.

**What it does NOT give: a monthly min/max band.** There is no monthly series
anywhere in the theme, only annual bioclim aggregates. So the card's "climate
band (monthly min–max)" still needs CHELSA — the plan below stands unchanged.
The card labels this block "annual" so the two do not get confused when the band
lands beside it.

### The Köppen codes ship with no legend

`CL_KOP_*` are bare integers. They are Beck et al. 2018's 30-class order,
**verified** rather than assumed: Singapore 1=Af, Cairo 4=BWh, London 15=Cfb,
Moscow 26=Dfb, Rome 8=Csa, Chicago 25=Dfa, Bangkok 3=Aw, Lima/Phoenix 4=BWh,
Vancouver 15=Cfb. 10 of 14 exact. The four that differ from Wikipedia (Mumbai
Am/Aw, Nairobi Cfb/Cwb, Anchorage Dsc/Dfc, Beijing BSk/Dwa) are all genuinely
borderline cities — a wrong offset would have put Singapore in a polar class,
not one step along a boundary. Table in `match_ghs.py`, labels in `js/koppen.js`.

## Elevation: GHS applied as a fallback, and it doubles as an error detector

`GE_ELV_AVG_2025` is now used wherever Wikidata's P2044 is absent, labelled
"centre avg" on the card so a per-blob average never reads as a surveyed figure.

Validated against the **6,828 cities that have both**:

```
median disagreement   13 m       within 50 m   82.5%
p90                   88 m       within 100 m  91.3%
```

**Blob area does not predict the error** — median 12 m for blobs under 50 km²,
16 m for those over 1,000 km². That was worth measuring, because gating on size
was the obvious precaution and it turns out to buy nothing.

Coverage gain is modest: **52.3% → 57.0%**. GHS elevation only reaches matched
cities, and most of those already had a Wikidata figure.

**The tail is more interesting than the coverage.** Where the two disagree by
more than 500 m (~1%), it is usually *Wikidata* that is wrong:

| city | Wikidata | GHS | reality |
|---|---|---|---|
| Facatativá | 3 m | 2,594 m | on the Bogotá savanna, ~2,600 m |
| Oyem, Gabon | 3,000 m | 655 m | Gabon's highest point is 1,575 m |
| Muhanga, Rwanda | 5,945 m | 1,846 m | Rwanda's highest is 4,507 m |
| Ciudad Nezahualcóyotl | 3 m | 2,305 m | Valley of Mexico, ~2,240 m |

So the card flags the disagreement (`≠ 2594`) rather than silently preferring
either source. `ELEV_CONFLICT_M` in `js/data.js`. This is a curation queue in
its own right if it ever seems worth working.

## Climate: the monthly band (still to do)

**CHELSA V2.1, 30 arc-sec, 1981–2010.** Beats WorldClim on every axis that
matters here: CC0 vs CC BY-NC-SA, 1981–2010 vs 1970–2000, better precipitation,
and half the bytes at the same resolution.

30 arc-sec rather than ~5 km because the whole content of the chart is the band
height, and a 5 km pixel in steep terrain carries ~1.6–3.3 °C of terrain error
vs ~0.4–0.8 °C at 1 km. La Paz, Quito, Kathmandu would be visibly wrong.

```
https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/{var}/CHELSA_{var}_{MM}_1981-2010_V.2.1.tif
```
`var` = tasmin / tasmax / pr · `MM` = 01–12 · tasmin+tasmax ≈ 2.9 GB, pr ≈ 2.9 GB
Scaling: °C = `v * 0.1 - 273.15` · mm = `v * 0.01`

Try `/vsicurl/` range reads first (we need ~0.003% of the bytes); fall back to
downloading tasmin+tasmax only if that thrashes.

Caveat to carry onto the card: tropical-mountain precipitation is unreliable in
both datasets (~50% underestimate of peaks). Temperature is sound.

## Smoke test: passed, after one real failure

First attempt hung: 10 minutes, zero countries written. Cause was the type
filter `?city wdt:P31/wdt:P279* wd:Q486972`. Measured on Gabon (26 settlements):

| query shape | time | rows |
|---|---|---|
| `P31/P279*` subclass walk | **502 @ 43s** | died |
| no type filter | 0.6s | 26 |
| + aggregation | 0.8s | 25 |
| + label service | 0.8s | 25 |
| explicit `P31` VALUES list | 3.1s | **only 9** |

So the walk is ~70x slower, and the "cheap" enumerated-type alternative
silently drops two thirds of the data — a quiet correctness bug, not a loud
one. **No type filter.** The label service is free *because* it wraps an
aggregated subquery. `diag_query.py` re-runs this if WDQS behaviour shifts.

Re-run: 5 countries, 764 settlements, ~6s total query time. Field coverage —
name 100%, pop 100%, coords 100%, admin 96%, **elevation 16%**.

**Consequence of dropping the type filter:** the pool contains non-settlements
— "West Africa" (429M), "Guinea" the country, "Kindia Region". Mostly harmless,
but a region's coordinate can sit near its namesake city and shadow it. Fix
costs nothing: stage 3 calls `wbgetentities` for aliases anyway, and that
returns P31, so **filter by type locally there** — zero extra requests.

## Altitude: use GHS, not Wikidata

Wikidata P2044 is only **16%** populated, useless for a per-city colour scale.
`GE_ELV_AVG_2025` in the GEOGRAPHY theme is **100%** (11,422/11,422) and
validated against ten known cities — La Paz 3868 vs 3640, Quito 2801 vs 2850,
Lhasa 3658 vs 3656, Amsterdam 1 vs 2. All within tolerance.

Same theme also carries `GE_ECO_CLA_2025` (ecoregion, e.g. "Samoan tropical
moist forests") and soil class — possible extra card fields, free.

## Full run: 80,633 settlements, 274 requests, 1,770s query time

No partials, no rate limiting. A few transient 502/504s absorbed by backoff;
Russia and Germany timed out and split into population bands successfully.

**False alarm:** 46 countries returned zero. Not a bug — tiny territories,
duplicate-ISO twins (the real `SA` is `Q851` with 181; `CY` is `Q229` with 37),
and dependencies whose cities carry the parent's P17 (Hong Kong files under
China, Réunion and French Guiana under France). Riyadh, Mecca, Kowloon, Cayenne
are all present. **Consequence: country must come from GHS's per-city field,
never from which country file a city landed in.**

**Real bug:** the Netherlands returned **7** settlements against 43 urban
centres. The ISO code sits on `Q29999` "Kingdom of the Netherlands" but Dutch
cities carry P17 = `Q55` "Netherlands". Found by ratio, not by eye — every
healthy country yields 6–60 Wikidata settlements per urban centre, so 0.2 was
the outlier. DRC/Somalia/Sudan are also low (0.5–0.7) but that is genuine
Wikidata sparsity, not a wrong entity.

Fixed generally rather than by special-case: the country list is now the
**union of P297 holders and the entities cities themselves declare via P17**.
That second query costs 8.2s and adds 218 entities. Historical states (USSR,
Czechoslovakia, Austria-Hungary) come along harmlessly — they dedupe by QID.

## Matching (stage 2): 51.5% on English labels alone

Against the 11,422-centre roster, within 50 km:

| | |
|---|---|
| exact name or GHS alias match | 5,885 (51.5%) |
| candidates nearby, no name match | 5,220 (45.7%) |
| no candidate within 50 km | 317 (2.8%) |

The misses are **not** missing data. Spot-checked: Chattogram/**Chittagong**,
Ahwaz/**Ahvaz**, El Mansura/**Mansoura** — all present in the pool under the
other romanisation. So the alias pass should lift this a lot, and matching
should combine name + population + distance rather than name alone.

The rest are GHS naming pathologies: "Coyah" (2.99M) is the Conakry blob;
"Maharajganj", "Ponnani", "Bibhutpur" are Indian blobs named after small
constituent places. Those need the population/distance heuristic, not names.

## Two-layer database: BUILT and tested

Contract in [SCHEMA.md](SCHEMA.md) — read that before touching any stage.

```
data/base.json        generated   assemble_base.py
data/overrides.json   curated     serve.py PATCH only
data/cities.json      built       build.py only
```

Round-trip verified end to end: patch a field, add a curated-only field, create
a city absent from both sources (`x0001`), tombstone a junk row, rebuild. Then
the important one — change `base` under an existing override and rebuild:

```
STALE 1 overridden fields whose source value has since changed
  _stale=['name']    name still the curated value: 'Tokyo (edited)'
```

The correction is **kept and flagged**, never silently dropped or silently
trusted. That is what patches buy over whole-record copies.

All four edit tools have working endpoints: `PATCH /api/city/<key>`,
`DELETE /api/city/<key>`, `POST /api/city`, `POST /api/build`.

**Correction to an earlier note:** Wikidata elevation is **38.3%** across the
full pool, not the 16% quoted earlier — that figure came from a 5-country
sample. Still too sparse for a per-city colour scale, so the DEM plan stands.

## UI: light mode, modular, edit mode working

**Light palette** — water `#d8e6f2`, land `#f6f4ef`, dark text. Departs from the
dark nycriders house style; layout language (one control bar, ghost buttons,
hairline panels, no transitions) still follows it.

Ramps were **rebalanced for the light ground**, not just recoloured: yellow
disappears on pale backgrounds, so the population ramp runs medium-green →
**amber** → orange → red → purple.

**Curation state is deliberately subtle.** Untouched cities fade only 30% toward
grey, and curated ones are picked out by a **dark ring**. First attempt faded
62% and the map looked muddy and dead — which is the wrong default when <1%
curated is the normal state for months. Progress should show by rings appearing,
not by everything else looking drab.

### Split (see SCHEMA.md for ownership)

```
index.html        shell only, ~35 lines
css/app.css       palette + layout
js/colors.js      4 ramps + language palette. Pure, no deps.
js/data.js        load, index, typed arrays
js/map.js         canvas, thinning, pan/zoom, hit-test
js/card.js        hover card. Pure renderer.
js/edit.js        edit panel, PATCH, tools
js/main.js        wiring only
```

Verified end to end against the live server: patch → `POST /api/build` →
`cities.json` carries the curated fields with `_touched`. Unicode round-trips
(東京). `data/overrides.json` currently holds **one demo entry on Tokyo** that I
added while testing — delete it whenever.

## Roster is final: 61,357 cities

Fetch at floor 10,000 done — 102,134 raw rows, 322 countries, 1,540s query time.
Type filter applied → **61,357 cities**. Continents, empires and dioceses gone;
largest entries are now Greater Tokyo Area, Chongqing, Jabodetabek, Delhi.

### Two bugs found on the way, both silent

**1. One bad coordinate killed two whole countries.** India and Iran failed with
`ValueError: substring not found` — `parse_point` doing `wkt.index("(")` on a
P625 value that was not WKT. Exactly **one** malformed row in 5,244 took all of
India with it. `parse_point` now returns None for anything unparseable, skips
the row, and reports the count. Recovered India (5,244) and Iran (1,981).

**2. The type filter was deleting Madrid.** This is the important one.

```
Madrid     Q2807   types=['Q2074737']   "municipality of Spain"    DROPPED
Marseille  Q23482  types=['Q484170']    "commune of France"        DROPPED
Seville    Q8717   types=['Q2074737']                              DROPPED
```

A city's only P31 is often its national municipality type, which Wikidata files
under *administrative territorial entity*, not *human settlement*. São Paulo
survived only by accident — it also carries `Q515` "city". So the closure now
walks **two roots**, `Q486972` (human settlement) ∪ `Q15284` (municipality):
3,111 types, +16,240 rows rescued.

Still correctly excluded: French electoral cantons, Chinese subdistricts /
townships / counties, Tanzanian wards, Catholic dioceses, Iranian districts,
tambon, gram panchayat, UK parliamentary constituencies.

Verified against 43 major cities across the awkward countries (Italy comune,
France commune, Brazil/Mexico/Spain municipality, Thailand, China, Peru,
Tanzania, Ukraine) — **all present**.

This is precisely why the fetch **stores raw P31 and filters downstream**: a
wrong closure cost one re-assemble, not a 100k-row refetch.

### Safeguard added

`assemble_base.py` now reports **orphaned overrides** — curated cities missing
from the new base — loudly. Same failure class as the cities.json staleness bug:
the work is on disk but nothing shows it. Currently 4 overrides, all resolve.

## Point kinds, settings, and the China gap

`kinds.py` classifies every point as **city / aggregate / admin**; settings
panel toggles them, cities only by default, choice persisted in localStorage.
Card shows the type ("city", "borough of New York City", "prefecture of Japan").
`fetch_type_labels.py` pulls English labels for all 1,449 P31 types in use.

**Aggregates are a curated list, NOT a keyword match.** Matching labels on
"urban area" / "metropolitan" / "greater" is tempting and deletes real cities:

| type | why a keyword filter breaks it |
|---|---|
| `Q12813115` urban area in Sweden | *tätort* — how Swedish towns ARE counted |
| `Q15092344` urban area in Norway | same |
| `Q448801` Greater district town | a German town class |
| `Q200250` metropolis | carried by New York City itself |
| `Q2716259` metropolitan municipality in Turkey | Istanbul's actual government |
| `Q1530824` metropolitan municipality in SA | Johannesburg, Cape Town |
| `Q482821` metropolitan city of South Korea | Busan, Incheon |

An aggregate marker is **decisive** (NY metropolitan area carries `megacity`
too); an admin marker only counts when nothing else is claimed (Manhattan is
both a borough and a consolidated city-county). Verified on 14 cities + 6
aggregates. Default view's top cities: Chongqing, Delhi, Shanghai, Beijing,
Chengdu, Guangzhou, Shenzhen, Dhaka, Istanbul, Mumbai.

### China: was a filter bug, not a data gap

Anita noticed sub-prefecture Chinese cities in only ~a third of provinces.
Cause: `Q1070990` "county-level city of China" is outside the P279* closure, so
**366 of 396 fetched county-level cities were silently deleted**.

```
                                 before      after     official
prefecture-level city of China   286         286       ~293
county-level city of China        30         396       ~394
```

Found by listing every dropped type whose English label contains "city" — a
check worth repeating whenever the closure changes. It caught 9 more of the
same class (`subprefecture-level city`, `city municipality`, `district with
city status`, `city of Bosnia and Herzegovina`, …), now in
`kinds.EXTRA_SETTLEMENT`. Roster 61,357 → **61,866**.

**Still genuinely patchy:** the 6,507 `town of China` rows (township level) are
unevenly imported into Wikidata, dense in Hebei/Shaanxi/Guizhou and thin
elsewhere. That is a real upstream gap, not ours — but it is below city scale,
so the city layer is now complete.

## Build

- [x] `fetch_wikidata.py` — reviewed, rewritten, smoke-tested, full run done.
- [ ] `fetch_entities.py` — stage 3, `wbgetentities` 50 at a time: aliases in
      all languages **and P31**, so the non-settlement filter costs no queries.
      Only for QIDs that are candidates for some centre, not all 80k.
- [ ] `match.py` — stage 2, name + population + distance.
- [ ] `fetch_geonames.py` — alt-name candidate pool (not auto-selected; feeds the edit UI)
- [ ] `sample_climate.py` — monthly tmin/tmax/prec at each city point
- [ ] `build.py` — merge base + overrides → `cities.json`, flag stale overrides
- [ ] `serve.py` — local edit server
- [ ] `index.html` — map, bubbles, hover card

## Card fields

name · up to 3 alt names · country · subdivision · coords · altitude ·
population + sparse history (log line) · gdp per capita · languages (pie) ·
climate band (monthly min–max) · up to 3 facts · photo slot · religions (maybe)

## Colour

Coloured text throughout, and the same value gets the same colour in every city.

- **languages** — fixed colour per language, identical across all cards
- **altitude** — hypsometric: green → yellow → orange-red → pink / pale grey
- **gdp per capita** — dark blue → red rainbow
- **population** — green (low) → yellow (mid) → orange → red → purple (highest)

House style is `riders/nycriders/index.html`. No transitions.



extras to maybe include 
- river / body of water 
- regional food 
- etymology

- 
---

## Running it

```
python C:\Users\anita\projects\maps\citybrowser\serve.py
```

Then open **http://localhost:8765/** — edit mode is always on.
Ctrl-C to stop. `python serve.py 9000` for a different port.

Works from any directory; the script resolves paths relative to itself.
If it says the port is in use, an older server is still running — stop that one
first. (It binds exclusively on Windows precisely so this is a loud error rather
than two servers silently sharing the port, which happened once and made a new
endpoint 404 with no error anywhere.)

### Rebuilding data

```
python assemble_base.py     # cache/* -> data/base.json   (safe, fast, rerun freely)
python build.py             # base + overrides -> cities.json  (export only)
```

`assemble_base.py` is the one to rerun after changing filters or adding a
source. It never touches `overrides.json`, and it reports any orphaned
overrides.

### Fetch stages (slow, cached, resumable — rarely needed again)

```
python fetch_wikidata.py      # the roster        ~30 min
python fetch_entities.py      # aliases + wiki    ~20 min
python fetch_type_labels.py   # P31 labels        ~30 s
python fetch_countries.py     # country languages ~3 min
python match_gdp.py           # OECD GDP -> cities
python match_ghs.py           # GHS urban centres -> cities   ~2 min, local only
```

GHS theme downloads (once; all live under the same FTP path as the re-export
above, and `data/` is gitignored so they cost nothing in the repo):

```
GHS_UCDB_THEME_GHSL_GLOBE_R2024A/V1-2/...zip      80 MB   population 1975-2030
GHS_UCDB_THEME_CLIMATE_GLOBE_R2024A/V1-2/...zip   19 MB   Koppen + annual climate
```

Unzip the `.csv` from each into `data/` and rerun `match_ghs.py`.

**A JS syntax check is worth running after touching `js/`** — the modules are
only ever exercised by a browser, so a duplicate export or a bad import shows up
as a silently blank page with no error text anywhere. Copy `js/*.js` to `.mjs`,
rewrite the import extensions, and `node --check` each. That is how a second
`export function place` (colliding with the card's positioner) was found, having
produced nothing but an empty map.

`match_ghs.py` needs no network. It reads the four GHS theme CSVs in `data/`
and writes `data/ghs_matched.json` (merged by `assemble_base.py`) plus
`data/ghs_centres.json` (joined in the browser). It ends with a fixed spot check
over Guangzhou/Shenzhen, Tokyo/Yokohama, Fort Worth, Conakry and friends — every
one a case this file calls out as a trap, so a rule change that breaks one shows
up immediately rather than on the map three weeks later.
