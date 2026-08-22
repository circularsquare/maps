# cityhistory — how it works, and why

Interactive world-city population map, 3500 BC → 2025, annual, sublinear bubbles, viridis
(big = bright). `build.py` turns two source datasets into `data/cities.json`; `index.html`
draws it; `validate.py` re-runs the six checks that found each class of defect.

**This file is Claude-managed.** It records the current state of the pipeline and the reasoning
behind it — especially the calibration evidence and the approaches that were tried and failed,
because those are what stop the same ground being re-walked. `todo.txt` is Anita's and is not
mine to edit.

Build is ~10s, validate ~30s; run both directly, no need to hand off.

---

## 1. Sources

| | |
|---|---|
| **Historical** | Stadestér (`data/stadester/stadester_cities.json`), which is itself populstat (Lahmeyer) + Chandler–Modelski + Buringh fused per city. Antiquity → ~2002. See `FINDINGS.md`. |
| **Modern** | A stack, highest first — see §3.5c. **US Census MSA** (`tools/us_metro.csv`, annual 2000–2024, 373 metros, public domain) → **EC/OECD functional urban areas** (`prep_fua.py` → `fua2025.json`, 7,956 metros, EC reuse licence) → **UN WUP 2025** (`prep_wup.py` → `wup2025.json`). WUP is 13,191 agglomerations ≥50k, annual 1975–2025, official population-weighted centroids, CC BY 3.0 IGO. Source csv.gz and cbsa csvs are gitignored. |
| **Basemap** | [OpenFreeMap](https://openfreemap.org/) `styles/dark`, OpenStreetMap data — the same tiles japanrail uses. Replaced a Natural Earth 50m coastline drawn on its own 2D canvas: that existed because one big geojson fill would not render as a maplibre layer, which a *tiled* style sidesteps. Dropping it also removed `ne_50m_land.geojson` (1.6MB) from what has to ship. |

Attribution for WUP is in the About panel. Adding the literal "CC BY 3.0 IGO" string would be
stricter.

### The two source habits that cause most defects

Nearly every rule below exists because of one of these:

1. **Carry-forward.** populstat repeats one figure verbatim, year after year — sometimes for
   centuries. Its pre-modern series is effectively a **step function**: one benchmark held
   across an interval, then a new benchmark.
2. **Straight-line fill.** Where two real figures are far apart it interpolates linearly and
   stores every intervening year as if it were data. A filled stretch has a dead-constant
   per-year slope (Yuzhou: +438/yr for 2,300 years).

Neither is a measurement. Telling them apart from real returns is most of what `build.py` does.

---

## 2. Adjusted time

The time slider is warped: 27.8% / 22.2% / 25% / 25% of the track for pre-AD1 / AD1–1400 /
1400–1900 / 1900–now, so 1400 sits at the halfway mark and 1900 at three-quarters. Play advances
by slider fraction, so recent years pass slower.

That same warp is the project's unit for measuring **data gaps** (`ADJ_EDGES` / `adj()` /
`unadj()`), normalised so 1 adjusted year = 1 real year before AD 1. A 700-year hole is ordinary
sparsity in 2500 BC and a genuinely unrecorded city in 1750; adjusted years make the two
comparable. `segY()`/`segS()` in `index.html` and `ADJ_*` in `build.py` must stay in step.

---

## 3. build.py, in pipeline order

### 3.1 Entry filtering

- `DROP_MARKERS` — parenthetical metro variants dropped outright. Also does double duty as
  `index_variants()`' variant *finder*, so it cannot simply be widened.
- `DUP_MARKERS` — a **second** tuple doing only the "never draw this" job, added because
  widening `DROP_MARKERS` changed which entries got auto-spliced. 142 entries had a plain base
  entry and were being drawn twice at full agglomeration size on top of the real city (La Habana
  2.19M, Kano 2.17M, Ibadan 1.84M, San Salvador 1.80M). Those go. The other 112 have no plain
  entry at all — "Zagreb (municip.)" at 850k is the *only* Zagreb in the source — so they are
  kept and merely lose the parenthetical.
- **A `MERGE_INTO` donor is exempt from both marker drops.** The drops run first, so without
  the exemption the donor never reaches the fold and the merge reports "donor not present".
  Three cities had their *entire* pre-modern record inside a parenthetical entry: Sparta
  (−430..AD 200 in `Sparti (agglomeration)`, plain entry starts 1861), Pyongyang (−1000..2002,
  plain entry starts 1890) and Benin City (1600..1900, plain entry starts 1901). A sweep of all
  144 variants that begin >50y before their base found only six pre-1700, so this is a closed
  set — the other three are Alexandria (already spliced), Kazan, and **Cairo, which is still
  missing its entire medieval history** (variant from 700, plain entry from 1859).
  `index_variants()` also skips donors, so the splice and the merge cannot both fire.
- **`prefer_agglomeration` structurally cannot do that job**, which is why the hand table has to.
  It returns early on `if not V or not S or not G` — no WUP centre, no splice at all — and its
  cost model prices only the modern switch step against the break the splice opens. A variant
  whose entire worth is 2,300 years before the seam is invisible to it. Sparta's modern self is
  19,100, under WUP's 50k floor, so `G` is empty outright.
- `DROP_KEYS`, `CLIP_BEFORE`, `DROP_YEARS` — hand tables for individual bad entries/points.
- **`DROP_YEARS` entries must be RANGES.** Dropping a bare anchor promotes the adjacent
  straight-line fill into a new anchor and manufactures a fresh defect (dropping Beijing's 1914
  exposed the 1920 fill at 1,145,091).
- `SYNTHETIC` — **where a city is created from a shipped source**, injected into `raw` before
  anything reads it so every rule downstream treats it as a source entry. Three members:
  **Cahokia**, **Chan Chan** and **Mayapán**. Taking Cahokia as the pattern — its figures
  are a verbatim `chandlerV2.csv` row (`AD_1100 = 40,000, AD_1400 = 4,000`) that the fusion filed
  onto `Saint Louis-Missouri`, where it appears as 39,999 held from 1100 to 1830 and `NA_CLIP_BEFORE`
  correctly deletes it — on *that* entry it really is a placeholder. It cannot be a `RENAME`
  either: St Louis has its own history from 1764 and the mounds are 12 km away, so it is the
  Danapur/Pataliputra shape except that the ancient half had nowhere to go. `NA_CLIP_KEEP` holds
  the clip off the new entry, and is the single exception to that rule's premise that Anglo North
  America had no cities before 1700. **North America went from zero cities at every pre-colonial
  year to one.** The bar is the strictest in the file: verbatim figures from a shipped source, no
  entry already carrying them, and coordinates on the site rather than Chandler's geocode.
- `ARCHAEOLOGICAL` — the same injection, **one evidentiary tier down, and opened deliberately**
  for the pre-Columbian Americas because `SYNTHETIC`'s bar makes that region unfixable. The audit
  is GAPS.md §2.7/§2.8: `chandlerV2.csv` holds **three** rows with any value before AD 600 and
  stadester **seven** entries, four of which are a flat Modelski 100,000 stamped back to 900 BC.
  Chandler compiled in 1987 from the historical record, and for Mesoamerica and the Andes there
  essentially is no historical record before the Spanish — the evidence is excavation and survey,
  published after him and never folded in. So the region is not thin, it is absent, and no rule
  change reaches it. Eight members: **Cuicuilco, El Mirador, Kaminaljuyú, Moche, Wari, Calakmul,
  Copán, Chichén Itzá.** The relaxed rule is *a published estimate from the archaeological
  literature, with the source named and the range given in the comment where the field
  disagrees* — plus SYNTHETIC's rules 2 and 3 unchanged (nothing on the map carries the series;
  coordinates are the site). Two consequences elsewhere: they carry `type: "archaeological"` and
  get their own provenance character **`a`** (§3.12), and they are **exempt from the New World
  ramp** — `nw_cap` suppresses *unattested* compiler stamps, and an entry that cites its source
  is the case it was never aimed at. Without the exemption the ramp would delete most of the
  table, since it drops everything before `NW_RAMP_START` outright. Sites are excluded when they
  fall under the era display floor rather than when they are doubtful: Caral, Chavín, San
  Lorenzo, La Venta and Palenque are all out on that test, and the comment says so.
- `SITE_COORDS` — **the one place a lat/lon is typed by hand**, and the coordinate analogue of
  `CENSUS`. The `coord_fixes.json` pipeline repairs a bad coordinate by matching the entry's name
  to a WUP urban centre and taking its centroid, which cannot work for an archaeological site
  because there is no modern centre to resolve *to* — and those are exactly the entries a
  geocoder gets wrong, since the fallback it reaches for is a modern place-name that happens to
  match. Five entries, each needing (1) a series that is not in doubt, (2) a current coordinate
  that is demonstrably a different place, (3) a named destination site: **Anyi** (the Wei capital
  at Xia County, Shanxi, drawn 800km away in Jiangxi), **Caracol** (drawn 0.1km from Chichén
  Itzá, because the geocoder matched *El Caracol*, the observatory building there — 440km from
  the real Caracol in Belize), **Djenné** (filed by Chandler itself in Senegal, 700km west
  of the Niger inland delta), and two **lost minus signs** — **El Tajín**, filed at longitude
  +97.3782 and so drawn in Myanmar, and **Hamilton, Ontario** at +79.8661, drawn in Xinjiang.
  The sign flips are a distinct class: the digits are correct, so no name-matching repair ever
  fires, and `country` is right so `in_americas()` treats them correctly — only the *dot* is on
  the wrong continent. The general test is `country in AMERICAS and not -170 < lon < -30`, which
  returns exactly three entries; the third, a duplicate **Richmond, Virginia**, is in `DROP_KEYS`
  instead, because Richmond is already drawn correctly with a longer record.
- `RENAME` — modern town on ancient ruins → historical name (Pi-Ramesses, Thebes, Memphis,
  Babylon, Nineveh, Vijayanagara…). Without it the antiquity leaderboard read "Faqus, biggest
  city in the world".
- `coord_fixes.json` — repaired coordinates for geocoder-fallback entries (`tools/`).
- `drop_fallback_stacks()` — deletes what `coord_fixes.json` could not repair. 375 entries on 54
  country centroids, drawn as piles of bubbles nowhere near themselves (22 in central Siberia,
  20 in Morocco, 18 in the Netherlands). `propose_coords.py` re-homes an entry by matching it to
  a **WUP centre**, which fixes everything WUP knows about — 32 of the 55 on Russia's centroid —
  and structurally cannot reach the rest, because WUP's floor is 50,000 and the leftovers are
  all under it (Kandalaksha 48.5k, Kurchatov 48.2k, Gryazi 48.3k). A city drawn 3,000km from
  itself is worse than one absent, so they go; §7's GeoNames line is what would bring them back.
  - **Exact coordinates**, where validate check D rounds to 2dp. Reporting can be loose,
    deleting cannot: every genuine fallback resolves to one point to the metre — a geocoder
    returns a centroid, not a scatter — while a ~1km cell also catches Senglea / Vittoriosa /
    Isla, Malta's Three Cities, which really are 500m apart.
  - **Two rescues**, because a crowded point is not always a centroid. Where a geocoder collapses
    a city and its own districts onto the city's *real* location it looks identical to a
    fallback. WUP settles most of it — an entry that belongs where it sits has a same-**named**
    centre on top of it (Panamá City, 704,100, under two of its own corregimientos). Name
    agreement is required rather than proximity, because Spain's fallback point is Madrid
    itself, so a distance test would keep Erandio and Galdakao on the strength of Madrid's
    centre. The second rescue covers merged twin cities, where WUP has nothing at all
    (Villingen-Schwenningen is 85k and has no centre within 40km): a hyphenated name whose
    parts are exactly the other entries at that point. Deliberately narrower than plain
    containment, which would rescue Greece's centroid off "Eleftherios" / "Eleftherios
    Venizelos".
  - Check D's expected steady state is therefore **3 points / 9 entries**, all intentional:
    Sekondi + Takoradi + Sekondi-Takoradi, Villingen + Schwenningen + Villingen-Schwenningen,
    and the Malta trio. Those are duplicate entries, a different defect; do not "fix" them here.
- `MERGE_INTO` — `{donor key: target key}` for one city split across two entries. The donor is
  not drawn; the years it covers **outside** the target's own range are folded in, and nothing
  else. That outside-only rule is the safety: the two entries are usually on different
  definitions where they overlap, so mixing them inside the covered span would manufacture the
  very unit flips the rest of the pipeline removes. Extending the record at either end is safe
  because there is nothing there to contradict.
  - dedup cannot do this job — it *picks* the richer entry and discards the other.
  - **Athens** is the motivating case. `Gazi-Greece` is a neighbourhood 1.6km from the centre
    whose `chandler_modelski_key` is literally `Athens-Greece`; it carries 155,000 at −430, the
    Periclean city, and is the **only** ancient Athens in the source. The real `Athínai-Greece`
    entry starts at 1750. Dropping Gazi — the obvious move, since it draws as a separate 3.1M dot
    on top of Athens — would have deleted Athens from antiquity. Same trap as Copenhagen. Merged,
    Athens runs −430 → 2025 as one line. Note dedup would have kept *Gazi* (2,400 Buringh points
    beats 250 populstat ones) and thrown away Athínai's modern detail, its spliced agglomeration
    variant and its WUP graft.
  - The remaining duplicate-orphan cases in §6.5 (Karâchi/Karachi, Miensk/Minsk,
    To`skent/Tashkent, the second Minneapolis) are candidates for this table.

**Two hand-table entries that look wrong and are not.** Both are load-bearing; do not "tidy"
them away.

- **Copenhagen was missing from the map entirely.** Both "København [stad]" (890k) and
  "København [agglom.]" (1.44M) have `coords: null` in the source, so they drop as no-coord —
  and `propose_coords.py` never saw them, because it can only move an entry that *has* a
  coordinate. Standing in for the capital was the **Frederiksberg** entry, an enclave 1.8km away
  carrying Copenhagen's whole series (12,000 in 1101 → 1,268,867 in 1974; Frederiksberg itself
  never passed ~105k) with genuine Frederiksberg census rows interleaved. Fixed by renaming that
  entry to "København (Copenhagen)" plus `DROP_YEARS` for the interlopers (1921, 1927–1969, 2003)
  and the 1970–73 fill between the two cities. It keeps the deep history [stad] lacks (that one
  starts at 1750). Costs Frederiksberg as its own dot. Only Copenhagen was significant among the
  21 no-coord entries.
- **Memphis (Egypt)** showed an 846,000-person city from 1974 to 2001, on a site that has been
  ruins for two millennia. The entry is **two sources fused** — its `chandler_modelski_key` is
  "Memphis-Egypt", so Chandler–Modelski supply ancient Memphis (−2500..−300) and populstat
  supplies the modern town of Al-Badrashayn (1974..2002); Stadestér glued them and filled the
  2,274-year hole at a dead-constant +327.91/yr. The AD 0 value of 198,374, which reads as data
  on the chart, is a point *on* that line. And 845,672 is bad, which the dataset settles by
  itself: of 98 Egyptian entries carrying both a 1974 and a 2002 figure, **95 grew**; Badrashayn
  is the only one falling by more than 3x, and it falls 14.5x. Its Giza-province neighbours were
  39,184 (Awsîm), 64,192 (Al-Hawamidiyah), 110,826 (Warraq al-'Arab) in 1974 — exactly the tier
  its own 58,500 in 2002 sits in. So: `DROP_YEARS` −299..2001 (a range, per the rule above —
  dropping only 1974–2001 promotes 1973's 845,344 into a fresh anchor), plus
  `CF_KEEP ("Badrashayn, Al--Egypt", -1100)` so Chandler–Modelski's 100,000-through-−300
  survives. Without that second half Memphis faded out from −950, eight centuries early and
  through its own Saite revival.

### 3.2 Regional placeholder clips

Both tests are driven by the source's own **`country` field** via `in_americas()` /
`in_anglo_na()`, not by lat/lon. See "why the country field" below.

- **New World ramp** (`nw_cap`, `AMERICAS`): Chandler–Modelski stamp Classic-era peaks back into
  deep antiquity (Teotihuacan 150k at 800 BC; Tikal/Caracol/Tula/Tiwanaku flat ~100k from
  900 BC). A time-varying cap ramps from `NW_RAMP_START` −200 to `NW_RAMP_FULL` 400, so American
  cities fade in instead of popping in full-size. Region-based because a gap/shape filter cannot
  discriminate — the Old World giants share the same sparse shape.
- **North America pre-1700 clip** (`ANGLO_NA`): Anglo/French North America had no cities before
  ~1700, so all pre-1700 data there is fabricated (Cincinnati held flat 10k from 1425, Saint
  Louis 40k from 1100). Drop pre-1700, then strip a flat leading run held a century or more.
  Latin America is excluded — real cities from the 1500s — and so are Puerto Rico and the US
  Virgin Islands, US territories but Spanish-colonial cities from the 1520s. "Boston" in the
  source is Boston **England**, correctly untouched.

**Why the country field and not a shapefile.** The boxes (`lon < −30`; `lat > 32` and
`lon ∈ −130…−55`) misfired in both directions. Alaska and Hawaii fell *outside* the NA box, so
neither the clip nor the leading-flat-run strip ever ran there — which is why "Barrow" held
56,000 from 1900 to 1989. That figure is Barrow-in-Furness, England (the source has it at 57,010
in 1900) filed against Utqiaġvik, real population ~4,000. The lon box also swept in 140 Pacific
island entries (French Polynesia, Wallis and Futuna, American Samoa, Tonga) as "the Americas".

`country` is populated on all 24,219 entries with no blanks, 280 distinct values, and US entries
carry the **state** — 1,936 of them across all 50 states, which is a finer signal than any box.
`Georgia` (80 entries) is the one genuinely ambiguous string, US state vs Caucasus country, split
on longitude; none of the Caucasus ones has pre-modern data. Unknown values default to Old World,
which fails safe.

Point-in-polygon was evaluated and rejected **on accuracy, not cost**: PIP reads the coordinate,
and the coordinate is the field that is wrong in precisely the cases where the two disagree —
the source has `Hamilton-Canada` geocoded into India, `El Tajin-Mexico` at lon 97.38,
`Los Angeles-Equatorial Guinea` at LA's real coordinates. What these rules encode is a
*provenance* fact — "populstat's US/Canada pages contain fabricated pre-1700 numbers" — and
`country` records that directly where a coordinate only proxies it. (Cost also disqualifies it:
ray-casting 24,198 points against the repo's own vendored land geojson measures 2.8s at 110m and
23.9s at 50m, on a 10s build. And no box can express the rule anyway — Hawaii's latitude range
sits inside Cuba's, Puerto Rico's and Mexico's.)

Effect: exactly two cities change, Barrow and Nome, and validate is unchanged on all six checks.
**The one real cost is Nome**, whose 12,500 at 1900 is a genuine gold-rush census; the strip
drops the whole leading run including its first point, so Nome now starts at 1990. It cannot be
fixed by "keep the run's first point" the way `strip_carry_forward` does, because Barrow's first
point is the bogus one — that change would un-fix Barrow.

A `region check:` line prints every build, counting entries whose *coordinate* is west of −30 but
whose country is not in `AMERICAS`. The standing 140 are Pacific islands plus a few mis-geocoded
entries; a **new** name appearing there is the signal that a source update needs the table
extended.

Latent, not live: Spanish Florida, Louisiana and New Mexico are now inside the clip's scope,
where `NA_LAT_MIN = 32` had deliberately excluded St Augustine. Measured, 0 of the 269 newly
scoped entries have any pre-1700 data, so nothing changes today — but the clip's rationale no
longer quite matches its scope.

### 3.2a Chandler's AD 100 benchmark, filed at 100 BC

`data/chandlerV2.csv` carries both a `BC_100` and an `AD_100` column, and **the `BC_100` column
is Chandler's AD 100 table**. Three independent lines of evidence, none needing a judgement call:

1. The columns are **complementary, not overlapping** — 27 cities in `BC_100`, one in `AD_100`,
   and only Cadiz in both. Real benchmark columns do not partition like that.
2. **Rome is the sole occupant of `AD_100`** (450,000) and is **absent from `BC_100`**. No 100 BC
   world-cities table omits Rome; no AD 100 table is Rome alone. Read together the two columns
   are one coherent AD 100 list: Rome 450k, Seleucia 250k, Antioch 150k, Anuradhapura 130k,
   Carthage 100k, Smyrna 90k, Athens 75k, Chengdu 70k … London 30k.
3. The contents are **impossible as 100 BC**. London 30,000 — Londinium was founded c. AD 47.
   Lyon 50,000 — Lugdunum was founded in 43 BC. Nîmes 44,000 — colonia from c. 28 BC. Corinth
   50,000 and Carthage 100,000 — both razed by Rome in 146 BC and derelict until the 40s BC.
   Ostia 30,000 — its boom is the Claudian and Trajanic harbours, AD 42 on. Every figure is right
   for AD 100 and impossible for 100 BC.

Stadestér absorbed the error, so it arrives as a −100 anchor and is then straight-line-filled
forward for centuries: a city drawn at full Roman size through the two centuries before it
existed, and a fade or gap that starts 200 years early (Ostia faded out at AD 26, sixteen years
before its harbour was dug).

`fix_chandler_ad100()` runs as a pre-pass on the raw dict, before the New World cap, the
carry-forward strip and the fades, so everything downstream sees corrected dates. **It has to
take the stale fill with it** — the fill runs forward to the next real anchor and was computed
from a point that is about to move, so leaving it manufactures a cliff (Istanbul would read
36,000 at AD 100 against 254,400 at AD 200). The whole span from −100 to the next real anchor
goes; deleting fill costs nothing, since `dp_simplify` would have collapsed it anyway.

15 entries move: Ankara, Dunhuang, Ostia (`Fiumicino`), Athens (`Gazi`), Corinth, London, Lyon,
Milan, Gortyn (`Mires`), Nîmes, Oxyrhynchus, Teotihuacan, Thessalonica, Petra (`Wâdî Moosa`),
Zafar (`Yarîm`). Note the correction generally makes a gap **longer**, so it feeds §3.10b rather
than competing with it.

**Five are deliberately not shifted** — Carthage (`Al Marsâ`), Chengdu, Cadiz, Smyrna (`Izmir`)
and Capua all hold a real anchor inside the window a shift would have to clear, so the mis-filed
figure duplicates a value the series already has at the right date and moving it would collide.
Deleting it instead is a separate call, and Carthage in particular wants one: razed 146 BC,
refounded 29 BC, so its real shape is a trough rather than either figure. Cadiz is the one city
in both columns (62,000 and 65,000) — either the single genuine `BC_100` row, or the duplication
that gives the whole error away. Istanbul is left alone too: its Chandler key is misspelled
`Instanbul-Turkey`, and 36,000 for Byzantium in 100 BC is defensible on its own.

Listed by hand rather than derived, because build.py does not otherwise read `chandlerV2.csv`.
`--ad100` prints what moved and main() reports the count, so a renamed key goes loud.

### 3.2b Year 0 does not exist

1 BC is followed by AD 1 in the Julian and Gregorian calendars alike, so no source can have
observed a year 0 and every year-0 row is a processing artifact. Measured across the 102 entries
that have one: **68 (67%) are an exact duplicate of the entry's previous negative year**, and the
other 34 all sit on the straight-line fill between their neighbours (Chengdu's `0 = 248,218` is
precisely 100 years along the line from `-100:70,000` to `1:250,000`). Not one is a datum.

What dropping them buys is the **AD 1 cliff**. populstat holds the 100 BC benchmark verbatim onto
year 0 and then starts the AD table at year 1, so three top-12 cities fell off a one-year edge —
Alexandria 1,000,000 → 400,000, Patna 200,000 → 100,000, Taxila 150,000 → 100,000. Total drawn
urban population dropped **6.16M → 5.61M (−9.0%) in a single year**, ~87% of it Alexandria, and
Alexandria handed the world's-largest slot to Rome overnight — at a year everybody scrubs to.

No check could see it: whipsaw needs 8× (this is 2.5×), oscillation needs a return inside 40y (it
never returns), and despike is up-only.

With year 0 gone the same decline is spread over the 101 years from −100 to AD 1, which is a
claim the source is actually making rather than an artifact of its table layout. Totals now run
4.94M (−50) → 5.56M (−1) → 5.80M (1) → 5.40M (50), and Rome overtakes Alexandria around 20 BC.

**The filter is needed in two places**, which is easy to miss: the agglomeration-variant loader
(§3.3) reads `raw` directly rather than through the main year loop, so Alexandria — whose entire
series arrives via its `(agglomeration)` entry — kept its year-0 row and its cliff after the
base-loop fix alone.

### 3.3 Agglomeration variants

`index_variants()` pairs each "(agglomeration)" entry to its base; `prefer_agglomeration()`
splices one in only when it **lowers total discontinuity** — the switch step it saves must beat
the new break it opens at the variant's first year. Unconditional splicing of all 552 pairs
introduced 552 fresh mid-series breaks (median 1.37x).

`VARIANT_MIN_ANCHORS = 5` is the term the cost model was originally missing: it priced the step
saved and the break opened, but destroyed *detail* cost it nothing. The variant must carry at
least five points surviving DP (i.e. not fill) over the years it would overwrite. Manila has 7
and splices; Manchester, Liverpool and Newcastle have 3 and do not — they had been running
1901 → 1970 as one straight line with every intervening census gone.

`VARIANT_MAX_GAP = 30` / `VARIANT_GAP_ANCHORS = 4` is the *distribution* half, and counting was
not enough without it. Lisbon's variant has 7 anchors — comfortably past the floor — but **139 of
the years it overwrites (1820–1959) contain none of them**, while the base holds 15 real censuses
across exactly that span (1911: 435,400 = INE's 435,359). The splice was trading a century and a
half of Portuguese census data for a straight line. So: refuse when the variant leaves a hole
longer than 30 years *anywhere the base has ≥4 real anchors of its own*. The test is deliberately
conditional on the base — a hole is only a loss if there was something in it, and where neither
series has data the splice still runs. 30y is where the data splits cleanly: every legitimate
splice is under it (Manila's worst hole is 28y) and the losses run 50–140y. It recovered **Lisboa
(139y), Berlin (89y), Yerushalayim (78y), Jekaterinburg (72y), Glasgow (69y), Charkiv (69y), La
Paz (69y), Odesa (54y)** and more — 9–15 real censuses each. Splices 119 → 52, no check moved.

This was first written as a *ratio* against the base's anchor count, and that was subtly wrong:
**DP counts shape complexity, not data density**, so a clean census run is nearly collinear and
scores low. Cleaning Newcastle's conurbation rows out of the base left 266,600 / 275,000 /
283,200 for 1911/1921/1931, DP folded those to two points, the base looked sparse, and the splice
fired and re-hollowed the city the fix had just repaired. An absolute floor on the variant cannot
be moved by improving the base.

`PREFER_VARIANT` is the hand version, for a base that interleaves two definitions year by year.
**London is the only member**, because its variant has 8 real anchors. Manchester was removed
from it — the hand table bypasses the guard entirely — and its three conurbation interlopers
went to `DROP_YEARS` as ranges instead.

**Do not "fall back to the base where the variant has no anchors."** It sounds like the obvious
repair for London's empty 1947–1996 stretch, and it is wrong: in those years the base is still
the *other unit*. London would drop from 8,322,100 (1950, Greater London) to 3,343,075 (1950,
County of London) — a fabricated 60% collapse — then ramp to 10,427,084 by 1974. The base is not
the same city with fewer readings there; being a different measurement is why the variant was
preferred in the first place.

That fallback in fact already happens for years *outside* the variant's range, and it was a live
bug: the variant stops at 1997, the base holds its 1974 figure of 10,427,084 to 1999, so **London
was drawn at 10.4M in 1998–99**, between 7.19M and 7.07M. No check saw it — an exactly-equal pair
gives despike 10.4/10.4 = 1.0 on one leg, and check F no flip. Fixed via `DROP_YEARS`.

The remaining gap was worse than a fill: the variant **holds 8,322,100 verbatim from 1950 to
1996**, 46 years of carry-forward (well under `CF_MIN_SPAN`), then reports 7,187,300 for 1997. So
London sat at its 1950 peak for half a century and the entire postwar decline was absent. Filled
from the `CENSUS` table below.

### 3.3a Hand-entered censuses (`CENSUS`)

The only place a population is typed by hand — see `SITE_COORDS` and `SYNTHETIC` for the
coordinate and whole-entry equivalents, which came later and carry their own bars. Everything
else deletes, clips or reassigns what the sources say. So the bar is four things, all required:

1. a **named census**, not a recollection or a round number;
2. a geographic definition **matching what the surrounding series measures** — a correct figure on
   the wrong unit is worse than none, because it manufactures exactly the definition oscillation
   the rest of the pipeline exists to remove;
3. a real gap, i.e. the source has no measurement of its own there;
4. a span of source fill/hold to **clear**, so the new anchors are not fighting a held value.

Format is `key -> (clear_from, clear_to, {year: population})`, applied last in pass 1 so nothing
upstream can overwrite or re-hold it.

**A second, separate use of the same mechanism** is marked off inside the table: restoring a
figure already in a source we ship that the Stadestér fusion lost. That is not typing a
population, so the four conditions do not apply; its own three are that the figure is a verbatim
row from a file in the repo, that no entry on the map carries it, and that the span being cleared
is fabrication rather than data. One member: **Tula**, whose source entry is 100,000 held verbatim
from −300 to 1999 — 2,300 years of one number, the largest single block of fabrication in the
corpus — while `chandlerV2.csv` holds `Tollan / Tula AD_900 = 50,000` at those same coordinates
and nothing carried it. So the map drew a 100,000-person city for the twelve centuries before Tula
existed, and blanked it for the whole period it did. Now `900: 50,000`, a `DISAPPEARED` span at
`(1180, 1950)` for the Toltec collapse, and the modern town's real 2000 census.

**Ordering is load-bearing: this runs AFTER dedup.** Clearing a span removes points, dedup ranks
co-located duplicates by (point count, peak), and every US city here is filed *twice at identical
coordinates* — `Cleveland-Ohio` (city proper, peak 922,900) beside `Cleveland-United States`
(metro, peak 2,134,395). Run in pass 1 it shrank the city-proper entry until the metro one won,
so filling Cleveland with its real censuses silently swapped the map onto a 2.1M metro series.
A `CENSUS` key that is not on the map now prints a note rather than failing silently — which
immediately caught a proposed Seoul entry aimed at `Sõul (agglomeration)`, an entry `DROP_MARKERS`
drops. (Seoul needs nothing: the entry actually drawn already carries 1935: 444,000 = the Keijō
census 444,098, plus 1938, 1949, 1959, 1965.)

Applied: **London**, **Roma / Torino / Genova** (ISTAT comune census-day counts — Rome's 1981 peak
of 2.84M and Turin's 1.17M → 963k deindustrialisation were both missing), **Tashkent** (All-Union
censuses; the line read +21% in 1959) and **Amsterdam** (gemeente; the 1959 peak was held verbatim
to 1970, hiding the suburbanisation trough — +19% in 1985).

**The US rust belt is applied**, and getting there needed a prior fix. Every large US city is
filed **twice at identical coordinates** — `<city>-<State>` and `<city>-United States` — with the
same name, type and particulars, agreeing exactly at 2000 but diverging through the 20th century.
232 such pairs exist. The one carrying a `chandler_modelski_key` is the larger in 49 of the 51
pairs where only one has it (Boston 2.73x, San Francisco 2.68x, Albany 2.17x): that is
Chandler–Modelski's urban-agglomeration figure fused in, against populstat's city proper.

Which twin got drawn was decided by dedup ranking on (point count, peak), so the metro one usually
won and US cities showed at metro size for a century. Scored against the real city-proper counts
at 1900/1950/1990 (Census WP-27), the **state** entry wins all eight rust-belt cities decisively —
Milwaukee 7.7% mean error vs 38.5%, Pittsburgh 33.3% vs 144.1% — so eight `MERGE_INTO` entries
pin the state entry as the survivor. `MERGE_INTO` rather than `DROP_KEYS` because the metro twin
sometimes starts earlier and the outside-only rule folds exactly that in: Philadelphia gains
1700–1749, Baltimore 1775–1789, Pittsburgh 1800–1809.

Only then can `CENSUS` be aimed reliably. Keying those figures to `-United States` was the earlier
mistake and would have put city-proper counts onto an agglomeration series.

Result: every one of the eight now peaks where it should. **Cleveland 915,000 at 1950** — it
previously had no peak at all, because the source's own 1906–1925 rise (460,300 → 922,900) was
deleted by the terminal-unit trim and its backward ramp removal, leaving 1900 → 2000 as one line
reading 52% low at mid-century. Detroit 1,850,000 at 1950, St Louis 857,000, Pittsburgh 677,000,
Milwaukee 741,000 at 1960. Checks: **B 91 → 73, C 22 → 17, E 4 → 3**, A/D/F flat.

### 3.3b The 1950 US hole (`tools/us1950.csv`)

populstat has **no 1950 row for US cities at all** — its anchor years run 1930, 1940, then jump
to 1959, 1969, 1979, 1989, 1999. The viewer calls a year measured only if a real control point is
within ±5 years after 1900, so decennial anchors normally tile exactly; the 1940 → 1959 jump left
1946–1953 unmeasured and **~91% of US cities dimmed at precisely 1950**. The dimming was correct —
it was reporting a real hole — but it landed on a round year everyone scrubs to.

Filled from US Census Bureau Working Paper 27 (Gibson 1998), table 18, "Population of the 100
Largest Urban Places: 1950". Tables 17 (1940) and 19 (1960) were fetched too and used **only to
verify the unit**, never copied in. 91 of the 100 landed; the other 9 could not be matched or
verified and were omitted rather than guessed.

Kept as a CSV rather than 91 more dict literals: the provenance, the unit check and the per-row
error belong beside the numbers. Each row records which basis verified it (1940, or 1959-vs-the-
1960 census where the entry has no 1940 value) and the twin it must outrank.

**Two ordering traps, both hit and both fixed:**

- A 1950 city-proper figure is only safe on an entry that is *itself* city proper, so each row
  carries a `merge_from`. But `MERGE_INTO` runs after dedup, and dedup had already thrown the
  verified twin away — 27 of 91 keys were "not on the map". Dedup itself now treats "is a us1950
  target" as the first term of its rank, ahead of point count and peak. Richness is the wrong
  tiebreak for these pairs: the metro entry is bigger *because* it is the metro, so it won every
  time and US cities drew at agglomeration size for the whole 20th century.
- A `CENSUS` key that is not on the map prints a note instead of failing silently. That is what
  surfaced the 27, and earlier a proposed Seoul entry aimed at a dropped variant.

Result at 1950: US cities ≥250k **73% dim → 11%**, ≥100k **87% → 30%**. Values land on the census
(New York 7,891,957, Los Angeles 1,970,358, Boston 801,444, San Francisco 775,357). No check
moved. The long tail is untouched and left deliberately — WP-27 stops at the top 100, and filling
the ~620 remaining cities ≥20k needs a bulk source, not a code change.

### 3.3c The dead US modern era (`tools/us_modern.csv`)

populstat stops about 2000, and WUP maps **agglomerations ≥50k rather than incorporated places**,
so only 388 of 1,790 US cities got a modern graft. The other ~1,300 had nothing at all after 2000:
build.py held their last value flat and the viewer correctly dimmed it, so **77–82% of the US read
as no-data from 2010 on**.

Filled from the US Census Bureau's Population Estimates Program (SUB-EST), `SUMLEV` 162/157 —
incorporated places and CDPs, which is the unit these series are already on:

| year | column | file |
|---|---|---|
| 2010 | `CENSUS2010POP` | `sub-est2019_all.csv` |
| 2020 | `ESTIMATESBASE2020` | `sub-est2024.csv` |
| 2024 | `POPESTIMATE2024` | `sub-est2024.csv` |

Unlike WP-27 this covers *every* place, not a top-100 sample: 1,594 matched and verified, 1,127
actually applied.

**Applied only to cities with no WUP graft**, and that condition is why it runs after the graft
principal is decided rather than with the other `CENSUS` entries. A grafted city already ends on
an agglomeration tail, and a city-proper 2020 on top of it would manufacture the definition
oscillation the rest of the pipeline exists to remove — New York's WUP 2025 is ~18M against a
city proper of 8.2M. The ungrafted set is precisely the set that was dead, so the constraint costs
nothing. This is not a break in the "agglomeration via the graft" rule either: that rule only
bites where an agglomeration exists, and a town of 30,000 that is not part of one *is* its own
city-proper figure.

Unit check as for us1950: a row is kept only if the entry's own last 1985–2005 value is within
**40%** of the 2010 census. 25% was the first cut and it was too tight — it rejected exactly the
most interesting cities, because Detroit really did lose a quarter of its people that decade. A
metro twin is wrong by 200–300%, not 40%, so the looser bound still separates them.

Result, US cities ≥20k drawn as no-data: **2010 82% → 40%, 2020 77% → 24%, 2025 77% → 24%**.
No check moved. What remains is 425 entries whose name is not in the census list and ~250 Canadian
and Mexican cities that fall inside the lat/lon box and are correctly excluded.

**London** is the first entry: the Greater London decennial census counts for 1951 (8,196,978),
1961 (7,977,178), 1971 (7,452,346), 1981 (6,608,598) and 1991 (6,679,699), clearing 1951–1996.
1981 and 1991 are the *unrevised* usually-resident counts — both were later revised upward, to
roughly 6.81M and 6.83M, for census undercount — and the unrevised ones are used to keep the
whole series on one basis. London now runs 8.24M (1947) → 6.61M (1981) → 7.19M (1997) → 10.4M.

### 3.4 Carry-forward strip

`strip_carry_forward()` — an exactly-flat run of ≥`CF_MIN_SPAN` 150 years at ≥`CF_MIN_VAL` 20k
is repetition, not data. Keep the run's **first** point (the real estimate), drop the rest.
Genuine data essentially never repeats a value to the byte across 150 years.
`CF_EPS` is relative because Stadestér's spline leaves float dust.

**The strip can eat real benchmarks, and `CF_EPS` being relative is why.** "Keep the first point"
is right when the run is populstat holding one estimate forward. It is wrong when the *compiler*
asserted the same value at several successive benchmark years and the fill that follows echoes it:
the measured half and the repeated half then fuse into one run and only the first survives. Gao is
the case — Chandler states 75,000 at 1550, 1575, 1585 **and** 1591, four separate benchmarks, and
Stadestér's 1600–1930 hold sits 5e-16 away, so the strip saw one 380-year run, kept 1550 and deleted
three benchmarks. The fade then landed at 1574 and Gao — the largest city in sub-Saharan Africa at
1575 — was invisible. **validate.py check G** finds these by joining each run to its Chandler row *by coordinate*
(the shipped source has no `chandler_modelski_key`) and counting benchmark years inside it. 206
found, 63 followed by a ≥2x cliff — and the cliff is what matters, because a run followed by a flat
continuation redraws to nearly the same line while one followed by a collapse turns an asserted
plateau into a centuries-long slide.

**The year must be a Chandler benchmark AND Chandler must roughly agree with the run's value**
(`GBM_VALUE_TOL`, a loose 2x band). Without the value test the check reports disagreements
*between the two compilers* as deleted data. Guangzhou is the case that showed it: Stadestér holds
200,000 from 700 to 1000, but Chandler's row **starts** at `AD_1000 = 40,000` and carries no
700/800/900 benchmark at all — so the "lost benchmark" was populstat contradicting Chandler
five-fold, not a figure worth restoring. The band is loose rather than an equality test because
Stadestér rescales: Timbuktu's genuine case arrives as 21,250 against Chandler's 25,000. Adding it
took the check from 206 to 170, and the cliff subset from 91 to 63.

Two fixes, both existing tables:

- **`CF_KEEP[(key, run start)]`** where the whole plateau is real. Istanbul, Memphis, Dali.
  Baghdad was added here in error and has been withdrawn — see §3.4a, which is the cautionary
  case for reading check A's output without checking Chandler's row.
- **`CF_END`** — whose `end` field already means "last year of genuine data", so setting it
  *forward* of the run's start keeps the measured half. This is the mirror of its original use
  (setting it back when the run's own first value is already too late). It must be keyed
  **`(key, run start)`** in this direction: a bare key applies to *every* run in the entry, which
  is what a backward entry wants and which un-stripped Gao's separate, genuinely spurious
  800–1300 plateau when tried. Either direction forces a fade, so a city that did not die also
  needs a `DISAPPEARED` entry.

- `CF_KEEP` — keyed by (source key, run's first year), because a city can have one genuine
  plateau and one bogus one. Istanbul is exactly that: its 944–1200 plateau is real, but a
  key-wide exemption would also have protected 750,000 held 1690–1790.
- `CF_END` — where the run's own starting value is already too late (the city was finished
  before it). Researched deaths: Kaifeng −225, Corinth 400, Tbilisi 1226, Sagaing 1364…

**`CF_MODERN = 1800` — which runs earn a *fade*.** The flat-run test is a good detector of
repetition and a bad detector of *abandonment*: for a step-function series every interval trips
it, so the strip fired on all of them and `plant_fades` then blanked the city across each one.
The two families separate on where the run **ends**:

- **dead** — the run ends *at* the first modern census, next datum a year or two later.
  Vijayanagara 1550..1890 → 1891, Kamakura 1250..1940 → 1946, Nara 709..1870 → 1877,
  Samarra 889..1840 → 1843, Bergama 100..1900 → 1901, Mari −1800..1980 → 1981.
- **alive** — the run ends centuries earlier and another *pre-modern* benchmark follows, i.e.
  the compiler is still tracking the city. Roma 900 → 970, Istanbul 900 → 944, Dimashq
  1100 → 1150, Guangzhou 1000 → 1067, Kunming 1600 → 1694.

1800 sits in a real valley: only 3 of 386 runs end in 1700–1749, the alive family tops out at
Changsha 1637 and Kaifeng 1750, the dead family starts at Luoyang 1810. `CF_END` keys always
fade regardless. **The gate is on the fade only** — the repeated points still go, so a step is
drawn as one interpolated segment between benchmarks (quarter weight, half opacity) rather than
as a hole in the city's existence.

### 3.4a Baghdad, and how a `CF_KEEP` exemption came to invent a collapse

Recorded at length because it is the only case so far where a hand exemption made the map
*worse* than the rule it overrode, and because the mistake is cheap to repeat.

Until 2026-08-22 the map drew Baghdad **932: 1.1M → 1100: 1.1M → 1150: 10,000 → 1250: 100,000**:
a 110× collapse across fifty years with no event behind it, followed by a tenfold recovery. That
is the largest unexplained cliff on the medieval map, and every part of it was ours.

**What the sources actually say.** Stadestér holds 1,100,000 at 932, 1000 *and* 1100 — float dust
apart, so `strip_carry_forward` reads one 168-year run and keeps only 932. `chandlerV2.csv`'s
Baghdad row says something different: **932: 1,100,000 · 1000: 125,000 · 1100: 150,000**. One
benchmark at 1.1M, not three. `provenance.py` agreed all along — it labels 932 `chandler exact`,
1000 `fill`, 1100 `populstat default`, which is the signature of a hold, not of a repeated
assertion.

**Why check A is not enough on its own.** Check A reports the run and the cliff after it and says
nothing about the compiler, so its line reads as a lost plateau. Check G *is* the compiler test
and it **never reported this run**: `GBM_VALUE_TOL = 2.0` exists for exactly this — the Guangzhou
guard above — and 1.1M against Chandler's 125,000 is 8.8× out. Check G's real Baghdad hit is a
different run, 1250..1400, eating Chandler's 1350 and 1400 behind a 1.1× cliff. The exemption
was written as though check G had found the 932 run; it had not, and its value test is the thing
that would have said so.

**What the exemption cost.** It protected populstat's carry-forward *at the Abbasid peak* and
handed it straight to Chandler's 1150: 10,000 — so the two halves of the cliff were a fabricated
plateau on one side and an outlier on the other. It also masked the run from every later check:
`CF_KEEP` short-circuits both `strip_carry_forward` and check G, so once written the entry made
itself invisible.

**The repair, in three tables.**

| table | what it does |
|---|---|
| `CF_KEEP` | the `("Baghdád-Iraq", 932)` entry withdrawn, with the reasoning kept in place |
| `CENSUS` | Chandler's own 1000/1100/1200/1250/1300/1350/1400 restored verbatim over 933–1400 |
| `DROP_YEARS` | Chandler's 1150: 10,000 deleted — the third editorial deletion of a benchmark |

Only the last of those is a judgement call. 1150 sits between Chandler's own 1100: 150,000 and
1200: 100,000, so his row asserts a 93% loss and a tenfold recovery inside a century either side
of a year in which nothing happened to Baghdad — al-Muqtafi's restored caliphate withstood a
Seljuk siege in 1157. Read as a missing digit (100,000, the figure he gives at 1200, 1250 and
1350) it stops being anomalous. Nothing in the pipeline can reach it: `despike()` is up-only by
construction and check F needs a return inside `OSC_SPAN = 20` years. This is the Roma 361 call
again; deleting the one `DROP_YEARS` line restores it.

**What changed on the map.** Baghdad now falls from the Abbasid peak across the Buyid and Seljuk
periods, is 100,000 at 1250, **40,000 at 1300** — Hulagu took the city in 1258 — recovers to
100,000 under the Jalayirids by 1350 and is 90,000 at 1401, the year of Timur's second sack. All
six figures are Chandler's. The AD 1000 leaderboard changes with it: Baghdád goes from 1.1M and
first to 125,000 and fifth, behind Istanbul, Kaifeng, Kyôto and Merv. Chandler's 125,000 is at
the *low* end of the literature for Baghdad in 1000 just as his 932: 1,100,000 is at the high
end, and both are carried as he states them — the same rule Cahokia's `SYNTHETIC` note sets out.

`analyze_jumps.py` loses both Baghdad entries from its worst-60; the dataset's `10x+` bucket goes
from 25 jumps to 22. Baghdad's largest remaining jump is 2.2× at 957–1000, which is Chandler's
own assertion about the Buyid takeover.

**The events file depended on this.** `data/events.json` carries a 1258 "Mongols sack Baghdad"
note whose curator note recorded the problem before the cause was known: at 1258 the bubble was
~100,000 and *rising*, so an `im: −3` collapse colour sat over a growing dot. With the repair the
curve under that note falls 100,000 → 40,000 across 1250–1300 and the note describes what the map
does.

### 3.5 Graft: joining a city to its modern WUP centre

This is the one place the pipeline could invent a fact rather than repeat a source's, so it is
deliberately dumb. `match_centre()`: a city joins a centre if the **names agree within
`GRAFT_NAME_KM` 40km**, or the **centroids are within `GRAFT_TIGHT_KM` 15km**. Otherwise it
joins nothing and the city simply ends when its data ends, which is the honest outcome. **No
population term anywhere.**

Radii are tuned against damage: 30km lost Guangzhou (pop-weighted centroid 31.6km off the old
town); a 5km tight radius lost every romanisation pair (Bangalore/Bengaluru 6.0km, Taegu/Daegu
6.6, Jiddah/Jeddah 5.5, T'aipei/Taipei 10.3). `TIGHT_MIN_FRAC = 0.2` stops a big city whose own
centre is farther off landing on a satellite named after a neighbourhood (Kobe → "Hinomine
3-chome", Fort Worth → "River Oaks", Perm → the village of Kondratovo) — killed all 27 with no
false positive.

The predecessor scored every centre within 50km by `peak_pop / distance`: 6.6% of its matches
were name-mismatched *and* >5km away (Moyo took Nimule's centre across a border), and scoring by
size also failed the other way — Brazzaville's own centre went to Kinshasa across the river.

**Principal.** Two cities can land on one centre and only one carries the modern series. Rank is
`(GRAFT_PRINCIPAL_WINS, names_agree, peak)`. The name test is right for Dongguan/Shenzhen (WUP
has no Dongguan centre and Dongguan's history outweighs Shenzhen's) and wrong wherever WUP labels
a centre after a *district*: Cleveland lost to Parma OH and got no modern tail at all, Venezia
lost to Mestre, Antioch lost to "Hatay". `GRAFT_PRINCIPAL_WINS` is an 18-key hand list. See
§6 for why no general rule is available.

### 3.5c The modern source stack: MSA → FUA → WUP

**The problem WUP alone has.** WUP 2025 measures the Degree-of-Urbanisation *urban centre* —
contiguous cells above 1,500/km². That rule works where cities are dense and fails where they
are not, and the **United States is the worst case on the map**. American suburbia sits nearer
1,000/km², fails the density test, and the metro shatters into disconnected cores: WUP has
**357 US centres, 25 of them ≥1M**, and puts Chicago at 3.68M (metro 9.5M), Dallas at 1.52M
(8.1M), Boston at 1.64M (4.9M), Atlanta at 458k (6.3M). Measured at the seam, US cities peaking
≥100k had a **median switch step of 0.80× and 25% of them lost more than half their population
in a single year**, against 1.08× and 8.5% worldwide.

It is also a *fragmentation* problem, not only a scaling one. GHSL names each fragment after
whatever populated place landed in it, so name-matching lands on the wrong piece: Fort Worth's
real blob is called **"North Richland Hills" (404,680)**, Nashville's is **"Castlegate Estates"
(132,507)**, Oklahoma City's is **"Moore" (204,392)**.

**The fix is a different delineation, not a different population raster.** Every alternative
built on the same DEGURBA rule — GHS-UCDB, OWID — relabels the seam without fixing it (§7).

**Layer 1: eFUA (`prep_fua.py`).** GHS-FUA R2019A draws 9,031 functional urban areas worldwide
— an urban centre plus the zone sending ≥15% of its residents into it, which is the same
concept as a US metropolitan statistical area. It ships one epoch, so a series is constructed:
take the WUP centres whose centroid falls inside the polygon, sum their annual series, and
scale by `k = FUA_p_2015 / (that sum in 2015)` so the curve passes through the FUA's own
measured population. Result: **7,956 centres**, median k 1.24. New York 19.8M, Chicago 9.02M,
Dallas 8.61M, Philadelphia 6.44M, London 14.1M, Tokyo 36.8M.

Three guards, each with a case behind it:
- **`k` is clamped at 1**, and this is load-bearing rather than defensive. eFUA is a 2015
  delineation and WUP 2025 a 2025 one; in ten years a lot of ring became core, so for 1,081
  FUAs the WUP centres inside already sum to more than the whole 2015 polygon (WUP's single
  Jakarta centre is 36.1M against a 2015 Jakarta FUA of 29.8M). An FUA is a superset of its
  centres by construction, so where the cores have caught up the honest uplift is none.
- **Coverage ≥ `COV_MIN` 0.22**, coverage being `sum(WUP members) / eFUA's own UC_p_2015`.
  This asks "did the join find this FUA's cores", which a threshold on `k` cannot: **high k is
  also the signature of the thing being fixed.** Charlotte's k is 10.2 and entirely real (FUA
  2.02M against an MSA of 2.7M); Jakarta's satellite polygon has k 28.5 because its only member
  is an unrelated 58k town. It separates cleanly — every bad join is ≤0.20, the lowest genuine
  US case (Davenport) is 0.23. The bar rises to 0.6 when no member is even *named* after the
  FUA, which is what caught a second "Jakarta" polygon handing Purwakarta 3.8M.
- **The principal is name-preferred but size-bounded (`NAME_MIN_FRAC` 0.5).** Name preference
  is needed because the fragment carrying the city's name is not always the biggest (Tangail
  444k vs a fragment called Elenga at 460k) and `match_centre` joins on *name*, so the override
  must land on the centre the city will actually claim. Unbounded it hands a conurbation to a
  junior partner: eFUA calls the western Ruhr "Dortmund" (5.84M) and Dortmund's own centre is
  753k inside it against Essen's 2.85M, so Dortmund took the whole Ruhr and drew a 6.3M dot —
  GRAFT_DENY's Essen problem rebuilt in a new place. Bounded, it falls to Essen, which
  `GRAFT_DENY` already refuses, so the Ruhr stays unbuilt. Which is right.

**Layer 2: US Census MSA (`tools/make_us_metro.py` → `us_metro.csv`).** eFUA *models* an MSA;
for the US the Census Bureau publishes the real one, annually, so where both exist there is no
reason to prefer the model — and where the model is bad it is very bad (eFUA's Boston is 2.79M
against 4.90M). 373 MSAs, matched to entry keys through the stadester key list itself so it
cannot invent an entry. Two traps found the hard way: CBSA **codes change between vintages**
(Los Angeles is 31100 for 2000–2009 and 31080 after; Cleveland 17460 then 17410), so series
must be merged across codes or each metro silently keeps half its years; and only the **first
state** in a title may be used, or "Kansas City, MO-KS" hands the identical 2.25M to two real
entries five kilometres apart.

**What the stack does NOT fix, and must not be expected to.** The seam. populstat's American
tail is **city proper** — its 2000 figures are the census place counts almost exactly (Atlanta
416,000 vs 416,474; Indianapolis 792,000 vs 791,926) — so handing over to any metropolitan
figure is a real change of unit. US cities ≥500k now step **up** by a median 2.55×. The old
0.80× median looked better and was two wrong numbers agreeing by coincidence. What went away
is the *collapse*: cities losing more than half at the seam fell from 22.4% to 0.8%.

So `report_switch_steps()` gets worse on purpose (median 1.08× → 1.40×) and is no longer a
pure honesty number — it now measures a definition change the map is deliberately making.
`validate.py` is the better witness: **C (graft collapse) 17 → 11** and **H (never regains)
84 → 79**, against B (whipsaw) 67 → 75.

**Open: 228 of 373 MSAs never reach a graft**, because their entry matched no WUP centre or a
co-located duplicate won it. Those US cities fall through to `us_modern.csv` city-proper
figures instead, so the US is not internally consistent. Biggest miss is San Jose (2.0M).

### 3.5d Africapolis: built, and deliberately not switched on

`prep_africapolis.py` → `africapolis.json`, enabled with `--africapolis`, sitting above the FUA
layer for the 1,899 African centres it matches. It is complete and it works; the default is a
judgement, and it goes against the argument that put `us_metro` above eFUA in the States.

**The case for it.** Africapolis is the purpose-built African urban record — national censuses
plus imagery, rebuilt for African settlement patterns rather than inherited from a global model
— with nine observed epochs. By the "a real national measurement beats a global model" rule it
should win, and one number agrees: the seam **improves**, 30% → 33% within 0.8–1.25× and 34% →
29% beyond 2×, because populstat's African tail shares Africapolis's census lineage.

**The two costs, both measured.**
1. **It changes nothing where the map is read and shrinks everything else.** Against the FUA
   layer the big cities already agree — Lagos ×0.97, Kinshasa ×1.06, Abidjan ×1.03, Kano ×0.98,
   Kampala ×1.02, Casablanca ×0.95. But the median across all 1,638 matched centres is **1.50×**
   (p90 3.73×), so mid-size African cities would be drawn about a third smaller than their
   non-African peers. That is the cross-region definition break §3.5c exists to remove, put back
   in a new place.
2. **A continent dimmed at the present day.** Africapolis's observed record stops at 2020 —
   2025 onward are projections, which this pipeline does not draw as data — so every African
   city ends in a flat `hx` hold. African cities ≥200k held forward go from **37 of 656 to 311
   of 541**, i.e. 58% of significant African cities drawn dimmed for the last five years while
   the rest of the world runs to 2025.

**Known rough edge if it is ever switched on:** a big agglomeration rejected by the
re-delineation filter leaves its WUP centre free for a small neighbour to claim, which is how
Onitsha goes from 6.31M to 363,000 ending in 1995. The match would need a size guard against
the centre it is replacing, the way `TIGHT_MIN_FRAC` guards `match_centre`.

**What the source actually is, because it reads as corruption and is not.** Africapolis is
*morphological* — contiguous built-up plus ≥10,000 people — and in continuously-settled rural
belts that produces regions. The 15.5M "Kisumu" is really `Kisumu/Mbale/Busia/Sirari`, the Lake
Victoria basin across three borders, with a built-up area of **18,694 km²** against Nairobi's
1,746. Cairo reads 38.4M. These are genuine published values: an earlier pass judged Digital
Earth Africa's geoserver mirror corrupt on exactly these numbers, and the official workbook
carries them identically — the mirror was faithful and the inference was wrong.

**Workbook traps, all of which cost a pass to find:**
- `MergedTo` is documented as a merge destination and behaves as a **cluster assignment** —
  Kisumu points at Nairobi 250 km away, and 1,908 rows point at themselves. Filtering on it
  deletes Cairo, Lagos and Kinshasa.
- `Population_2025` onward are **projections**, identifiable as the non-integer columns.
  Nairobi's 2025 is 15.95M against a 2020 of 7.57M; its 2050 is 57M, more than Kenya.
- `0` means "did not exist / below the 10,000 threshold", not zero people.
- Africapolis **re-delineated at 2015**, so 108 rows step >2× inside their own series (Mbale
  88k → 2.23M, Embu 175k → 2.05M). Those are excluded by our own one-definition-per-city rule.

**Growth-rate audits do not work on this data and the attempt is recorded so it is not
repeated.** Any threshold strict enough to catch something also fires on real demography:
Monguno, Dikwa and Gajram multiply because they are Borno State garrison towns holding the
region's displaced, and Angola's jumps follow the 2014 post-war census, its first in 44 years.
`audit()` therefore tests the one thing with no judgement in it — a country's agglomerations
cannot sum past its national population — which is the check that catches the real failure
mode, and did: before the re-delineation filter, Kenya summed to 74% urban against a true ~28%.

### 3.5b China: administrative-area tails

`trim_admin_tail()`, and it runs **before** the terminal-unit trim so that one sees the city's
own figures rather than a county's.

China's late populstat rows report the county, prefecture-level city or municipality rather than
the settlement. Chongqing is the clearest: 4,644,814 held 1984–1994 (the real urban core), then
21,834,938 in 1996 and 30,979,100 in 2001 — the municipality, a province-sized unit created in
1997. It was the largest city in the world on the map for those years.

It is not a handful of entries, and it is specific to China. populstat's last figure against
WUP's first, measured on the finished build:

| | n | ≥2x | ≥3x | ≥5x |
|---|---|---|---|---|
| China, before the rule | 375 | 41.6% | 25.9% | 9.3% |
| China, after | 375 | 29.9% | 16.3% | 6.4% |
| everywhere else | 5,528 | 4.2% | 1.3% | 0.3% |

**Why this is scoped to China and is not a general rule.** The worst non-Chinese cases on the
same measure are Jacksonville, Nashville, Charlotte, Fort Worth and Indianapolis — and there the
disagreement has the *opposite* cause: WUP's density-defined core undercounts a sprawling city
while populstat is right (§6.2). Identical numbers, inverted meaning. A global "trust WUP when
they disagree by 3x" would fix China and wreck the American South.

Two thresholds, because one is not enough. Trigger at `CHINA_ADMIN_RATIO = 2.5`, then walk back
until the series is within `CHINA_ADMIN_ACCEPT = 1.5` of WUP again. With a single threshold the
walk stops at the first point under it and leaves the tail still on the wrong unit — Harbin
dropped 9,411,000 and landed on 8,241,248, which is still the prefecture (WUP says 3,570,000).
populstat *ramps into* the administrative figure, so the intermediate fill is administrative too.

85 tails trimmed. Chongqing 30,979,100 → back to 4,644,814, and its seam becomes 1.26x. No large
Chinese city is damaged: Beijing 1.07x, Shanghai 0.87x, Wuhan 1.29x, Hangzhou 1.19x are
untouched, and Guangzhou (0.35x) and Shenzhen (0.15x) are the inverse case, correctly left alone.
validate B 112 → 91, C 29 → 22, **F 27 → 10** — most of the definition-oscillation remainder in
§6.3 was this.

**Residual: 82 entries trigger the rule and are refused**, because trimming cannot fix them.
`CHINA_ADMIN_MIN_KEEP = 2` stops the walk gutting a city entirely, and these hit it — their
series is administrative more or less throughout, so there is no city-level figure to fall back
to. Two shapes: ones that *start* administrative (Wuwei starts 804,000 against a WUP of 66,928;
Dongying starts 512,000 against 55,484), which are unfixable here; and ones that start at city
scale and ramp (Yushu 57,200 → 1,185,800), which lowering `MIN_KEEP` from 3 to 2 already
recovered 19 of.

### 3.6 Terminal unit switch

`trim_terminal_unit_switch()`. populstat's late series often carries two definitions: a metro
figure held flat for decades, then one final census at the administrative city. New York holds
16,150,000 from 1974 to 1999 then reports 8,008,300 for 2000. Both are real; they measure
different things, so whichever the switch lands on, the other becomes a cliff.

We do not guess. **WUP arbitrates** — it is a third measurement of the same city, so whichever
level its handover value is closer to in log space is the one on the same footing as the modern
layer, and the other is deleted. It cuts both ways, which is the sign it reads the data rather
than applying a prejudice: New York keeps the 16.15M plateau, Detroit keeps the 951k census.

Deleting the plateau alone leaves the trough, because the fill *before* it survives — those
points interpolate toward a figure we just rejected, so they go too. The backward walk stops at
the previous real anchor, bounded by `TRIM_WALK_MAX_YEARS = 20`. It is bounded in **years**
because that is the unit the fill is measured in; the old 40-*point* limit was 170 years in a
decadal series and took every Cleveland census from 1830 to 1920.

**When WUP never matched the entry**, there is no third measurement, and this used to return
immediately — leaving the largest jumps on the map untouched (Funza ×90.8, Brebes ×15.9), and
worse than the grafted ones, because the contradicted final row is also what §3.11 then freezes
to 2025. Two fallback arbiters, in order:

- **The entry's own earlier record.** Sidoarjo runs 38,700 (1963) → 53,500 (1979) →
  1,070,000 held 1989–2001 → 76,900. Growing to 76,900 is a city; to 1,070,000 is a kabupaten.
  `base` is the **minimum** over `TRIM_LOCAL_LOOKBACK` = 40 years before the plateau, not the
  nearest anchor, because the straight-line fill ramps monotonically *up* into the plateau, so
  every fill point sits between the anchor and the plateau and a nearest-point rule reads the
  fill. The minimum cannot — the fill never goes below where it started. Klaten is the case: 29
  years of fill from 32,700 to 1,020,000, where a walk bounded at `TRIM_WALK_MAX_YEARS` lands on
  a fill point at ~339,000 and reads the plateau as continuous. Fires when the plateau is
  `TRIM_LOCAL_MARGIN` = 0.5 log10 further from `base` than the final row is.
- **A short whole-record stub.** Cibinong, Cikampek, Ciparay, Majalaya and Japeri have *no* rows
  before the plateau at all — ~1.8M held 1990–2001, then 142,000, and nothing else. Nothing
  supports the plateau but its own repetition, and a carry-forward is not a measurement (§1).
  Bounded by `TRIM_LOCAL_WHOLE_MAX` = 25 years, because dropping this plateau deletes the city
  for the span it covered: Islington, Tower Hamlets and Brent each carry one borough figure from
  1901 to 2001, and unbounded the rule erased their entire 20th century. Majalaya and Ciparay
  give the game away by carrying the *same* 1,909,500 four km apart — one Kabupaten Bandung
  figure stamped on both.

It is deliberately **one-directional** — it can drop the plateau, never the final row — and that
asymmetry is the safety argument. Funza has the same terminal shape but is Bogotá's series top to
bottom, with one final row at 49,900 that is the only genuinely-Funza figure in it; its plateau
is *continuous* with its record, so the margin test refuses it. Were the rule symmetric it would
delete that 49,900 and hold a 4.5M phantom to 2025, far worse than the cliff.

`TRIM_LOCAL_TAIL_FLOOR = 3` guards the other end. Closer-than-the-plateau is not enough on its
own: Sefton's record runs back to a 4,636 village figure, so its junk final row of **1,000** —
for a borough that had just reported 305,813 — still scored as the closer of the two, and
dropping the plateau in its favour froze 1,000 people on screen until 2025. A final row below a
third of even the *lowest* figure in the window is a bad row, not a unit switch, and neither
level is then trusted. Sefton is left to validate check E, which is where it belongs.

182 entries are arbitrated this way. Terminal-class jumps in `analyze_jumps` fell 34 → 19, and
every validate check improved or held (B 112 → 73, C 29 → 17, F 27 → 10, E 4 → 3).

Gate constants: `TRIM_PLATEAU_MIN = 2` and `TRIM_DROP = 1.3`, made safe by `TRIM_BREAK_MAX = 3`
— the disagreement must appear within 3 years. Every real case is a **one-year** fall (Nantes
540k → 270k across 1999→2000, Lens 499k → 36k) and a genuine catastrophe that size does not land
on the last row of every French and German city at once. Note it is *not* "delete the year
2000": only 939 of the 7,662 series ending in 2000 dip, so a year-keyed rule would destroy 88%.

### 3.6b Denying a graft

`GRAFT_DENY` — a hand table of entries where the modern layer is *worse* than the historical one,
so the graft is refused and the city ends when its own data ends, held forward and drawn dimmed
like any unmatched entry. Three failure modes — in the first two the centre is the right one and
its *number* is not the city's; in the third the centre is the wrong place outright:

- **Conurbation.** GHS merges the western Ruhr — Essen, Duisburg, Oberhausen, Bochum,
  Gelsenkirchen, Mülheim, Bottrop — into one 2.72M urban centre (wup 675), while keeping
  Dortmund (748k) and Düsseldorf (907k) separate. Essen won it on its name, so the map drew a
  2.72M "Essen" on top of Duisburg (515k), Bochum (391k), Gelsenkirchen (279k) and Oberhausen
  (222k), all still drawn from their own populstat records — the conurbation counted once whole
  and once in parts.
  It was self-reinforcing, which is why it needs a table and not a threshold: populstat's Essen
  entry carries *both* definitions (3,609,289 held through the 1990s, then 595,100 for 2000),
  and §3.6 arbitrates that split against the WUP handover — which, being the conurbation, kept
  the Ruhrgebiet plateau and deleted the one correct figure in the entry. Denying the graft
  breaks the loop: with no WUP, §3.6's local arbiter weighs the plateau against Essen's own
  earlier record, drops it, and the entry ends on 595,100. Essen now reads 595k, not 2.72M.
- **Absent.** WUP's Noril'sk centre (wup 20872, 1.0km away, unambiguously the right one) runs
  144,964 in 1975 down to 56,272 in 2010 and then stops — it falls under the 50k threshold and
  leaves the dataset. The real city was 174,673 at the 1989 census and 182,701 in 2021. GHS's
  density segmentation does not cope with an Arctic industrial city (the centre 17km away shows
  the same decline). populstat's 143,100 is wrong by 20%; the graft was wrong by 70%.
- **Wrong place.** `TIGHT` matched **Soweto** to wup 3855 **Lenasia**, a separate township 6.9km
  away, so the map handed Soweto's 596,600 (1991 census) to Lenasia's 57,272 (2000) — a 10.4×
  drop at the handover — and then drew Lenasia's growth to 214,770 under Soweto's name. Soweto is
  ~1.3M.
  Nothing was going to match, and that is the honest answer rather than a failure: WUP has no
  Soweto centre because Soweto is contiguous with Johannesburg and falls inside wup 3664.
  `TIGHT_MIN_FRAC` should have caught it and did not — Lenasia reaches 36% of Soweto's peak by
  2025, over the 20% gate — because the gate compares **all-time peaks** and these two series
  barely overlap in time. The diagnostic quantity is the step *at the handover*: for a name-
  mismatched `TIGHT` match a 10× drop there is not the seam, it is a different place. Check C
  misses it too, for the mirror reason — it scores peak against the post-2000 *maximum*, and
  Lenasia's own growth brings that back to peak/2.8, inside the peak/5 gate. Left as a table
  entry, not a rule; see GAPS.md for the check-C variant it suggests.
  Denied, Soweto ends on its 1991 census held forward. That is ~53% below its 2011 count of
  1,271,628 and still much the better answer: right place, right unit.

Deliberately **not** a rule. "Deny when the centre is much bigger than the entry" *is* the seam
(§3.7) and would fire on every US city; "deny when the centre shrinks" would fire across the
whole post-Soviet and Rust Belt map, where the decline is real.

### 3.7 Hard switch

`merge_series()` — populstat runs to its own last year, WUP takes over the year after. The
per-city seam is emitted as `sw`.

This replaced a geometric blend across 1975–2000, which hid the seam by *spreading* it: the step
became a 25-year trend, and where populstat was metro-wide and the WUP centre only the dense
core, that trend ran against real growth. 358 cities were drawn **declining** through the late
20th century when the source says they grew — and drawn *solid*, because blended points are
dense enough to pass the "real control point nearby" test.

`report_switch_steps()` prints the step distribution every run. It cannot reach 1.0 — they are
different measurements. Currently **median 1.40x, 30% within 0.8–1.25x, 34% beyond 2x**, up
from 1.08x / 36% / 26% before the modern stack of §3.5c.

**It got worse on purpose and is no longer a pure honesty number.** The modern side is now
metropolitan and the historical side is often city proper, so the step measures a definition
change the map is deliberately making, not an error. Read `validate.py` C and H instead — both
improved. The number to watch here is the *downward* tail: cities losing more than half at the
seam, which is what a bad graft looks like, and which fell from 22.4% to 0.8% for US cities
≥500k.

### 3.8 DP simplification

`dp_simplify()` runs in **linear** space with a relative eps (`DP_EPS_REL` 0.01), not log,
because the fill is linear — so linear-space DP collapses it back to its real anchors and the
viewer's log-linear interpolation reconnects them *geometrically* (a plausible growth curve
instead of a smear). The same change stopped the modern era being over-simplified into fake
20-year gaps (Tokyo 3 → 10 control points).

### 3.9 Despike

`despike()` deletes a single row that jumps **up** off both neighbours and straight back down —
one row measured on a different geographic unit. Runs *after* DP, never on the raw series: the
fill means raw neighbours are interpolated **from** the spike, so no amplitude test can see it
there (Berlin's raw 1910 is a fill between 1905 and the 1914 spike).

- `OSC_SPAN = 20` years. Deliberately **not** widened for antiquity, which is the opposite of
  intuition: every unambiguous catch is within 18y because a definition flip is a census-*row*
  artifact and rows are years apart. Past ~20y the shape stops being diagnostic and becomes
  history — all ten pre-1500 candidates with a 40–400y span were real (Thebes' Ramesside peak,
  Carthage before the Punic Wars, Timur's Samarkand, Mansa Musa's Mali, Delhi before Timur's
  sack, Naples either side of the Black Death). A wide ancient window would delete exactly what
  the map exists to show.
- `OSC_AGREE = 1.35`, `OSC_AMP = 1.5`, or `OSC_AMP_HUGE = 4.0` with no agreement requirement
  (no trend explains a 13x round trip).
- **Up only.** A switch to a bigger unit inflates; catastrophes deflate. Deleting downward
  excursions would erase Hiroshima 1945.
- `OSC_SEAM_AMP = 6.0` when the right-hand witness is grafted WUP rather than populstat. A WUP
  value is not a witness against a census — the rule works by believing both neighbours, and
  when the graft is wrong they are wrong *together*, so it deleted the one correct row
  (Henderson NV really did hit 175,381 in 2000). But a flat "never trust WUP" gate silently
  restored the whole Chinese prefecture family, because there the spike **is** the last
  historical year. The two groups separate cleanly: prefecture rows are 6–19x, every suburb
  false positive was 1.5–2.8x.

Families killed: the 1990/1994 Chinese censuses reporting county seats at prefecture level
(Xuanwei 70k → 1,174,700 → 78k, plus ~20 more), "Greater" rows in 1914 Germany, a 1925
agglomeration row across Europe, the 1984 Libyan census, Mazar-i-Sharif 1997 (a wartime
displacement estimate, not the city). `build.py --spikes` prints every deletion with its
justification.

### 3.10 Fades

**`bracket_lone_anchor()`** — a city whose entire record is ONE point was drawn in exactly one
frame of the timeline and in none of the other ~5,500, because `popAt()` returns 0 outside
`[p[0], p[-1]]`. That is data we hold and never show, and **29 pre-1800 cities were in that
state** — Miletus, Troy, Ani, Khajuraho, Prambanan, Istakhr, Dvin, Vallabhi, Tamralipti, Anbar,
Loango, Dongo, Surame, Dzibalchaltun, Chan Chan. They now get the same shoulders the fade
machinery plants everywhere else, `FADE_YEARS` adjusted years either side. That is not an
invented population — it is exactly the claim the record supports, *this size at this date,
nothing either side* — and it draws as a bubble rising to the benchmark and falling away.

The `LONE_ANCHOR_BEFORE = 1800` gate is load-bearing: the **other 639** single-point entries are
20th-century census rows for city districts and English boroughs (Vale Royal, Thamesdown,
Warringah, Landhi Korangi), which are the duplicate/sub-district problem of §6.5, not this one.
Bracketing those would put 639 spurious dots on the modern map.

`fade_long_gaps()` inserts floor points across a gap so the city fades **out** after the earlier
anchor and back **in** before the later one, instead of the viewer smearing a line across
centuries of no data. Absence of data across a millennium means "not a recorded major city", not
"draw a line".

- `FADE_GAP = 3000` **adjusted** years — 3,000 real before AD 1, 1,500 to 1400, ~480 to 1900,
  ~120 after. Calibrated on the cities either side of the line: must clear Kaifeng 1232–1751
  (2,547 adj) and Roma 622–1377 (1,510), both continuously inhabited and merely under-recorded,
  while still catching Kannauj (3,322), Kamakura (4,609), Handan (6,586), Memphis (7,813),
  Yuzhou (8,138). At 700 adjusted it blanked Rome 675–1350 and Baghdad 1000–1150 — that is how
  sharp the cliff is. (Baghdad no longer has a hole there at all; see §3.4a.)
- `FADE_YEARS = 150` adjusted years of ramp at each end. Measuring the ramp in adjusted time is
  what stops it being an eyesore: a fade-in ending at a 1940 census used to rise out of nothing
  across 1790–1940, a seventh of the visible timeline. Now ~6 real years there, still literally
  150 in antiquity, which is the look worth keeping.
- `plant_fades()` is triggered by the carry-forward strip rather than by gap length, so it skips
  `FADE_GAP` — but it needs `FADE_STRIP_MIN_REAL = 300`, or it blanks cities across gaps that are
  just early-modern sparsity. It plants 130 fades and the floor suppresses 53, among them Sakai
  1582–1877, Constantine 1515–1808, Shizuoka 1600–1877 and Samarkand 1575–1834. (Baghdad's
  218-year 932–1150 hole set this constant and no longer exists; see §3.4a.) It is also gated by
  `CF_MODERN` (§3.4).

**The fade cannot simply be deleted.** Turned off for one build, the medieval leaderboard
collapsed immediately — Yuzhou #1 in 1000/1200/1400/1600, Memphis #2. The graph's quarter-weight
is honest about interpolation but the **bubbles** are not.

#### 3.10a `FADE_GAP` is saturated, and a real-year second arm does not work

`FADE_GAP` cannot catch the **cross-era smears** — a single ancient anchor joined to a medieval
one by a straight line, so the city is drawn at full Roman size through centuries it spent
derelict. London is the type case: after §3.2a its only anchors are AD 100 (30,000) and 1199
(40,000), and the interpolation puts ~35,000 Londoners on the map in AD 500.

The threshold cannot be lowered to reach it. London's gap measures **2,198 adjusted years** and
Kaifeng's legitimate 1232–1751 hole measures **2,547** — they want opposite verdicts, they are
14% apart, and the band just below is packed with 1900–2000 pairs that must not fade at all.

The obvious second arm is **raw years** (London 1,099, Kaifeng 519). It was implemented at
`FADE_GAP_REAL = 900`, measured, and **rejected**. Of the 21 gaps it newly fades, about half are
cities that were continuously inhabited and merely under-recorded, and it asserts their absence:

| faded | reality |
|---|---|
| Halab −1700..622 | Aleppo, among the longest continuously occupied sites anywhere |
| Vārānasi −430..622 | continuously inhabited; Xuanzang finds it dense and prosperous c. 635 |
| Samarqand −400..622 | Afrasiab occupied c. 500 BC–1220 AD without a break |
| Trabzon 260..1204 | a Byzantine port and theme capital throughout |
| Changsha −200..1077 | a Chinese prefectural seat throughout |
| Messina −200..1223 | Roman, then Byzantine, then Norman |
| Memphis −2500..−1360 | the Old and Middle Kingdom capital — the largest city on earth |

The failure is not the threshold, it is the **question**. Gap length measures how sparse the
record is; it cannot distinguish *derelict* from *under-recorded*, and those are the two
hypotheses. No function of `(y0, y1, v0, v1)` separates Aleppo from London, because the data
genuinely looks the same in both — the difference lives entirely outside the dataset. Same
lesson as "the span must not widen for older history" above. `FADE_GAP_REAL = None` is left in
place as a named constant so the idea is not re-derived from scratch.

#### 3.10b `DISAPPEARED` — the case-by-case half

Because a fade **asserts absence**, and a wrong-but-present interpolation is the cheaper error,
the cross-era smears are handled one at a time. `plant_disappearances()` runs after both generic
rules and **overrides** them for the cities named: every `FADE_FLOOR` point is stripped first,
then the table's spans are planted. An entry mapping to `[]` therefore reads "this city was
continuous, take the fade off" — which turned out to be the commonest correction, because the
generic rules were blanking six cities that never went away.

13 cities, all individually researched:

- **No rule catches these** (the smear is one straight line): London `(450, 670)` — Londinium
  abandoned by 457, Lundenwic from the 670s. Athens `(650, 900)` — Herulian sack 267, Slav sack
  582, the Agora abandoned 7th–10th c. Corinth `(-146, -44)` — hidden *inside* the −430→AD 100
  line: razed by Mummius, refounded by Caesar. Chengdu `(1650, 1670)` — Zhang Xianzhong burned
  it in 1646 and the source holds 120,000 verbatim from 1649 to 1720 straight across the event.
- **Right collapse, wrong dates** — every one had its fade-out planted at 175, which is
  `fade_pts`' mechanical ramp off an AD 100 anchor rather than a claim: Ostia `(550, 1930)`,
  Petra `(650, 1985)`, Zafar `(550, 1900)`, Cádiz `(600, 1500)`, and **Smyrna `(1402, 1580)`**,
  whose collapse is medieval (Timur, 1402) not ancient — the old floor blanked the entire
  Byzantine city. **Mosul `(-600, 640)`** had it backwards: Nineveh fell in 612 BC, but the rule
  blanked 1325–1813, when Mosul was battered and never emptied and Chandler has it at 34,000 in
  1800.
- **False abandonments, fade removed**: Vārānasi (was floored 697–1616), Balkh (1043–1769,
  though Chandler has 30,000 at 1150 inside it), Ankara (175–1501, which blanks a 30,000-class
  late-Roman city — Ancyra really did contract to a citadel of 2,000–3,000 in the 650s–660s, so
  `(650, 900)` is defensible, but the reading is contested and `[]` asserts nothing instead).

Two sharp edges. Spans must land **strictly inside** one gap or they are not planted and the
build reports `NOT APPLIED` — that means a real anchor sits inside the span, so either the dates
or the anchor is wrong and the answer is to look. And the strip cannot tell a planted floor from
**Buringh's literal 1,000**, its nominal value for a town too small to model (Nîmes carries a
real 1,000 at 1399, Drapetsona opens on one at 1300) — so every entry is checked against its
built series by hand, and the table must stay short enough that that keeps happening.

Keyed on the **post-merge** record: ancient Athens arrives on `Gazi-Greece` and is merged into
`Athínai-Greece` (§3.1), and this runs after the merge. `CHANDLER_AD100` keys the same city as
`Gazi-Greece` because it runs before it.

### 3.11 Hold recent cities forward

`EXTEND_FROM = 1990` / `YEAR_NOW = 2025`, emitted as `hx`. The map used to empty out after 2000
(17,512 cities visible in 1990, 6,372 in 2025), which is not "we stopped knowing" — it drew the
towns as ceasing to exist, and made every aggregate read of the modern era wrong in the same
direction. 79% of the cities that stop are under 50,000, i.e. below WUP's threshold, so no
modern figure exists for them in any source we have.

Held **flat**, not grown: a growth factor is an assumption applied to 14,000 towns at once, and
plenty of the world is flat or shrinking. Flat says "this is the last measurement", which is what
it is. Honesty is carried by existing machinery — `winsFor()` skips control points past `hx`, so
the held run is drawn at quarter weight and dimmed, like any other stretch with no data behind
it. 13,884 cities held; visible at 2025 rises 6,372 → 19,440 and the count now increases
monotonically 1950 → 2025.

### 3.12 Per-point provenance (`provenance.py`, emitted as `s`)

Stadestér is four datasets in a trench coat and labels only the **entry**, not the year — so one
line on the chart can be a Chandler benchmark, a Buringh model value and a populstat census in
consecutive segments with nothing to say so. `provenance.py` recovers it in three tiers, and
build.py emits the answer as `s`: **one character per control point, parallel to `p`**.

- **Tier 0 — anchor vs fill. Free and exact.** Stadestér fills gaps with straight lines in
  linear space, so a point on the chord between its neighbours is fill. Same Douglas–Peucker as
  §3.8 but at `ANCHOR_EPS = 1e-6` instead of 1%: build.py wants the biggest simplification that
  still looks right, this wants the smallest set that regenerates the series *exactly*. Median
  share of a series that is real: **populstat 15.8%, buringh 27.0%, chandler_modelski 60.0%**
  (needs `MIN_SERIES = 3`; C-M's median series is 3 points, so short entries are trivially 100%).
  **Only 19.1% of the corpus is a data point.** This is the most valuable output here — the
  viewer currently cannot tell a measured century from an invented one.
- **Tier 1 — Chandler, exact.** 1,497 entries carry `chandler_modelski_key`; matching the value
  against `chandlerV2.csv` at 0.5% identifies **78.6%** of the anchors that land on a Chandler
  year for their city (3,994 points). Two gotchas: the CSV is **cp1252, not UTF-8** (`A Coruña`
  at byte 6429 kills a UTF-8 read), and `BC_100` must be read as year 100 per §3.2a — the
  lookup accepts −100 as an alias so it matches both before and after the shift. Of the 187
  entries typed `chandler_modelski`, **none carries the key**, so tier 1 can never speak for
  them and their fallback is chandler/`default`.
- **Tier 2 — the Buringh year grid, derived not hardcoded.** Anchor-rate likelihood ratio,
  buringh entries vs populstat entries, same year: 1550 397×, 700 363×, 1100 337×, 1000 335×,
  1200 334×, 1650 283×, 1300 278×, 1400 261×, 1740 171×, 1790 105×, 1840 25×.
  **The ratio alone gives the wrong grid**, and this is the load-bearing detail: 1500 (57×),
  1600 (39×) and 1861 (33×) clear any threshold that keeps 1840 (25×), but none is a Buringh
  benchmark — only 19%, 14% and 16% of buringh entries anchor there against 82% at 1550. Their
  ratios are high because the *populstat* rate is low. Hence a second gate,
  `GRID_MIN_SHARE = 0.25`. Without it London's **1861 census reads as Buringh**. Final grid is
  13 years: 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1550, 1650, 1740, 1790, 1840.
  `REFERENCE_LR` pins the measured ratios and raises if any drifts >30%.

de Vries is 12 entries and gets no separator; it folds into the populstat default.

Codes, in precedence order: **`f`** planted fade floor (ours, not data — and it must not read as
a measurement of 1,000 people, see §3.10b), **`a`** an `ARCHAEOLOGICAL` entry (§3.1), **`w`**
past `sw`, so UN WUP 2025, which `provenance.py` has never seen, **`c`/`b`/`p`/`u`** attributed,
**`i`** a control point provenance considers fill. `a` is **assigned from the table, not
derived**: provenance.py recovers which *compiler* a year came from, and those entries predate
every compiler's coverage, so it would report them as `u`. It earns its own character because a
survey estimate is a different kind of claim from a census or a benchmark — and because
`sourceSpans()` then marks the join wherever one meets a Chandler row, which is exactly what
Monte Albán needs (archaeological to AD 700, Chandler's benchmark at 800). `i` is not a contradiction of §3.8: DP keeps points at 1% while
provenance splits at 1e-6, so DP legitimately keeps points on a fill line, and merge/trim/hold
add their own. Calling those `i` is the honest answer — they are not measurements.

Distribution over 211,857 control points: p 67.1%, w 14.5%, i 7.9%, b 7.0%, c 1.9%, f 1.6%.
Cost is ~3.0s on a ~10s build (2.4s to derive the grid, 0.6s for 24k `classify` calls, anchors
cached by the identity of each population dict). Payload 4.41 → 4.78 MB.

**Three records can contribute years to one series and all three must be classified**, in the
order the pipeline applies them, so the last writer of a value is the one that names it:
`donor_entries` (§3.1) → `entry` → `variant_entry` (§3.3). Both of the compound cases were found
by a top-tier city reading as pure interpolation, and both are cheap to get wrong and very
visible:

- **Athens** — its entire ancient record arrives on the `Gazi-Greece` donor, so classifying only
  the surviving entry reported all of it as fill. The target winning any overlap is exact rather
  than a tie-break, since a merge only takes donor years *outside* the target's range.
- **Alexandria** — its whole series arrives via its `(agglomeration)` variant, which *overwrites*
  the base across the years it covers, so the variant is applied last. Before this it read `i`
  from −300 to 1999 — a top-3 city for the entire classical era, reported as invented.

---

## 4. index.html (viewer)

- Dark map, year scrubber + play, sublinear bubbles (`BASE` 0.21, exp 0.29), live "bubble size"
  and "big-city contrast" sliders. `?year=N` URL param. The map's floor is `mapFloor()` (§4.3);
  `MINPOP` 5,000 stays the *graph's* floor and the level at which a control point counts as a
  city for the data windows.
- Viridis on an absolute scale (5k → 20M) so pre-modern "big" reads green; colour floor raised to
  40/256 so the smallest read blue, not violet.
- Fast projection: precomputed mercator + affine, which fixed pan lag from per-frame
  `map.project`.
- Top-10 largest-cities panel. Cache-buster on the `cities.json` fetch, so a rebuild shows up
  without a hard refresh.
- Slider starts at 3500 BC (Eridu + Larak + Uruk), not the lone-Eridu 3700 BC frame.
- **Interpolation honesty, two tiers.** A run is full weight only where a real control point is
  within ±5yr (post-1900) / ±20yr (1400–1900) / ±100yr (pre-1400); quarter weight elsewhere
  (floor 0.5px). Dashes were tried first and read as *breaks in the data* rather than as lighter
  line. On the map the same claim is half opacity (.325 vs .65). Shares `winsFor()`/`isInterp()`
  so the two cannot disagree. Half opacity = "we are interpolating this size"; missing entirely
  = "no data for thousands of adjusted years".
  - NB post-1900 the tolerance is ±5yr and ~34% of cities read as interpolated in 1960; that is
    DP-collapsed WUP data, guaranteed within 1% of the real series. If it feels overstated the
    fix is `solidTol`'s 1900 tier — one number, and it moves graph and map together.
- **Selection.** Click a bubble or line to trace it at 3.2px over a dark halo, in the city's own
  colour lightened 65% toward white (plain white read as a foreign element). Selecting mutes the
  field to 0.55x, including the current top 10. Hover priority follows what is drawn on top — the
  selected line gets a 26px radius and the top-10 preference is switched **off** while something
  is selected, or hovering your own selection hands you a neighbour's tooltip.
- `drawGraph()` early-outs on an unchanged (year, colorMode, hover, selection, size) key, so
  panning no longer repaints it. Path geometry depends only on canvas size and the scrub track,
  so `Path2D`s are cached per layout (`gLayoutKey`) and reused while scrubbing.
- **Canvas overlay on a real basemap.** The bubbles stay on their own 2D canvas above the GL
  map, glued to it by redrawing on maplibre's `render` event — the only way to stay in step with
  the transform mid-animation. With the old empty style that fired only while panning; a tiled
  basemap also fires it for every tile fade-in, and `draw()` costs ~27–42ms in 2025 (20,614
  bubbles), so those extra frames were pure waste because the transform is not moving during a
  fade. `redraw()` now compares a signature of centre/zoom/size against the last frame drawn and
  returns early when it is unchanged. Year changes call `draw()` directly and are unaffected.
- **Attribution** is `AttributionControl({compact: true})`, added by hand with
  `attributionControl: false` on the map. The default full bar sits straight across the expanded
  timebar's scrub track and its 1900 tick. `compact` alone is not enough — it still renders
  *open* on first load and collapses only once the "i" is clicked — so the `maplibregl-compact-show`
  class is removed on `load`. Sources are credited in full in the About panel either way.
- **Expand/shrink** (`setGraphBig`, the link at the panel's top right) toggles the graph between
  142px and `min(40vh, 100vh − 150px)`, **and widens the timebar with it** to
  `calc(100vw − 124px)`. The two must move together: `positionGraph()` pins the plot to the scrub
  track, so widening the graph alone would slide it out of step with the timeline directly under
  it, and widening the timebar alone would leave the plot narrow. 62px of clearance each side
  clears the only two things sharing that row — the info button bottom-left (16px + 31px) and
  maplibre's zoom control bottom-right (10px + 29px). The mobile media query overrides `.big`
  back to the full-bleed width, or the expanded bar would come out *narrower* than the default
  on a phone. `resizeGraph()` clears `gKey`, and `gH` is in `gLayoutKey`, so the cached paths
  rebuild at the new scale.
  - The panel's left edge sits ~110px inside the timebar's, because the scrub track starts after
    the year readout and play button. That gap is the alignment being correct, not a layout bug.

### 4.0b Source changes, in growth mode

Growth mode draws a segment **white** where the two control points either side come from
different datasets — the payoff for §3.12's `s` string, and the answer to "which of these
dramatic changes are real?".

The justification is the mode's own: `growthAt()` already refuses to measure across the WUP
handover, because a change of definition annualised reads as explosive growth (Shenzhen's
one-year 8.6× step came out at +24%/yr and saturated the scale). That reasoning was only ever
applied to one of the four sources; this applies it to the other three, which were invisible.
**Rome is the case to look at:** it falls 999,000 → 150,000 between 300 and 361, which reads as
a catastrophe and is populstat handing over to Chandler — two compilers who disagree 2.2× about
how big Rome ever was. White is used because it sits outside the ramp entirely, so it cannot be
misread as "very fast" at either end, which any in-gamut choice could be.

**`SOURCE_CHANGE_MIN_RATE = 0.015` is what makes it readable, and it is not optional.** Ungated,
**25–54% of every pre-modern frame goes white**: a buringh entry's grid years (1550, 1650, 1740,
1790, 1840) interleave with populstat's census rows, so nearly every segment is technically a
handover, and most are a percent or two and distort nothing. Gating on the segment's own
annualised rate clears the wash — 1800 goes 54% → 12% — while keeping every case worth seeing,
including the WUP seam, which peaks at ~13% of the map around 2001 and is the one moment where a
white flash is the correct answer. 1.5%/yr is also where the growth ramp has clearly left its
flat yellow-green band, so below it there is no misleading colour to suppress.

Fill and fade points make no provenance claim, so they never open or close a change: `i` inherits
the source of the last attributed point to its left (the question is which compiler the *segment*
belongs to), and `f` is ours, not a source.

### 4.1 Smooth line, and the handover join

Two drawing-only changes. Neither touches `popAt()`, the readout, the tooltip, the bubbles or
hit-testing, all of which still use the values `build.py` emitted.

**`smoothInto()` — monotone cubic instead of a polyline.** Every control point used to be a
corner, and at this scale most cities are mostly corners, so the whole graph read as spiky. The
curve passes exactly through the same vertices; only the shape *between* them changes, and that
was an arbitrary straight line anyway. Monotone (Fritsch–Carlson) specifically, **not**
Catmull-Rom: the monotone tangent rule cannot overshoot, so the curve never rises above the
higher of two neighbouring points nor dips below the lower one. A plain spline would invent
peaks the data does not have, indistinguishable from the defects the pipeline exists to remove.
At a local maximum the tangent is forced to zero, so a spike keeps its exact height and merely
stops being a needle. One `bezierCurveTo` per segment (the exact Hermite→Bézier conversion), so
it is no more expensive than the `lineTo` it replaced.

**`HANDOVER_YEARS` — the seam drawn as a join.** The handover is one year wide, and at under two
pixels per year that is a vertical cliff. `mkLine()` now emits the seam as its own run
(`d === 2`) spanning `sw` → `sw + 8`, dropping the vertices inside it, and `stroke()` draws that
run **dashed at half weight** — the dashes already carry "not a measurement", and at the quarter
weight the interpolated runs use they broke the join up so far that it stopped reading as a
connection. Wuwei stops reading as a city that lost 84% of its population and starts reading as
a source change.

This is **not** the geometric blend §3.7 rejected — that one rewrote the values across 1975–2000
and drew the result solid, so 358 cities were *asserted* to be declining. Here the values are
untouched and the join is visibly not a measurement. But the distinction is one of degree, and
the span is the knob: over those 8 years the drawn line is a straight ramp, so where the seam
step is large it understates the city (Shenzhen's 1.32M → 11.3M ramp puts it near 3.9M in 2005
against WUP's ~8M). 8 years is ~14px — wide enough to read as a diagonal, short enough that the
misstatement stays inside the dashed segment. Widening it re-creates §3.7's problem in a new
place. If the large steps become a nuisance, the better lever is to gate *which* seams get the
ramp on the size of the step, leaving the dramatic ones as an honest cliff.

### 4.2 Bubble size: smoothed inside the record, ramped at its ends

Two mechanisms, deliberately separate — they want different shapes and different lengths.

#### `sizeLog` — the size itself

The radius is drawn from the **mean of log10(pop) across a window centred on the current year**,
not from `popAt()` at the year itself, so any *step* in the data — a graft seam, a census that
jumps, the rise out of a fade — is spread across the window instead of snapping between two
frames. `logInt()` — the cumulative integral the growth colour already builds — makes that two
binary searches, so it costs about what the `Math.pow(pop, expo)` it replaced did (24.8 → 26.1 ms
over 19,948 cities in a synthetic loop). `sizeWindow()` converts the window bounds once a frame
rather than once a city, the same trick `growthWindow()` uses and for the same reason.

The window is **clamped to the city's own record**, so this is smoothing and nothing else: it says
nothing about the ends, which `edgeRamp` owns.

**Width is in track fraction** (`SIZE_TRACK` 0.010), for the reason `GROWTH_TRACK`'s comment
gives: it has to be a constant amount of *playback*, or it is invisible in 3000 BC and enormous
in 2020. 0.010 lands on ±5yr after 1900 and ±20yr across 1400–1900 — **exactly `solidTol`'s two
modern tiers** — and ±63yr to 1400, ±126yr before AD 1, straddling its ±100. That is the
justification as much as the arithmetic: the bubble is sized from the population averaged over
the era's own dating tolerance, the same window inside which the map already refuses to claim
precision. Drawing a razor-sharp size off a figure we only date to ±100 years was the less honest
of the two.

Mid-life distortion is nil — the median drawn/true radius ratio is 1.000 at every year sampled —
because `log10(pop)` is piecewise linear, so a window sitting inside one segment averages to its
own centre. Only kinks and steps move, which is the entire point. Widening past `solidTol` would
start flattening real peaks, so that is the ceiling on this number. Ur at −1988 is drawn at 0.57×
because its record really does fall 100,000 → 10,000 between −2000 and −1900; that is the
mechanism, not a defect.

#### `edgeRamp` — the two ends

A city's first control point is the first time somebody wrote a number down, not a founding, and
drawing it at full size made cities materialise out of nothing: 4,357 arrived as a disc 4px or
wider, and Shuicheng, Tianmen and Xiantao each appeared as a ~13px blob between one frame and the
next. So the bubble grows in — **ahead of the first record, reaching full size exactly at it** —
and the mirror at the far end for a city whose record stops before the timeline does.

**Ahead of the record, not after it.** Running the ramp after the first point was the first
attempt and it is wrong twice over. It contradicts the record it is drawn from (the figure says
100k at that year and we drew 19k), and because 1900 is the single biggest census year in the
sources, the default paused frame came up full of pinprick cities that the data says are cities —
211 of the 2,347 on the 1900 map are first recorded *at* 1900. Ahead-of-the-record invents
presence instead, which is the softer claim of the two — a city with a first census in 1900 was
there in 1898 — and `interpW` already dims a bubble with no figure near it to the most-interpolated
opacity for free, so the ramp arrives pre-labelled as a guess. On the 1900 frame, cities drawn
under 0.75× of their true radius fall from 211 to **2**.

**Shape is ease-out, `1-(1-u)²`.** Mean-log alone came out slow-then-sudden, because averaging in
log space and then exponentiating is exponential in `u` — Roma sat at 1.1px for 40 of its 63 ramp
years and then leapt. Ease-out leaves 0 at twice the linear rate (u=0.25 → 0.44 of full, u=0.5 →
0.75) and lands on full size with **zero slope**, so there is no corner where the ramp hands over
to the record.

**`BIRTH_TRACK` 0.005, half of `SIZE_TRACK`** — 63yr before AD 1 down to 2.5yr after 1900. The
ramp reads as a bloom and wants to be quick; the smoothing wants to stay at the era's tolerance.
There is no reason those two numbers should be the same one.

**Capped at the record's own span**, which is not a detail. 659 cities have a single control point
and were previously undrawable — `popAt()` only returns a value if the year lands exactly on it —
and 229 of those are post-1900 and large: Landhi Korangi 551k, Kázimiyah 521k, Chaoxian 740k, the
orphans and district-duplicates of §6.5. An uncapped ramp flashes each of them onto the modern map
as an ~11px bubble for five years. Nothing may be extended past its record by more than the length
of that record, so a one-point city stays exactly as invisible as it was (verified: 0 drawn across
301 sampled years).

**A ramping bubble is drawn and nothing else.** It is not ranked, not in the top-10 panel, not in
`curTopSet`, and not pushed to `projected`, so it is not hoverable — there is no figure behind it
to put in a readout, and it must never displace a real city out of the ten. `popAt()` still feeds
the ranking, the colour, the panel, the tooltip, the graph and hit-testing, so no readout ever
disagrees with what `build.py` emitted.

**Ends measured.** A city alive in 2025 is drawn at 0.99× and does not shrink at the end of the
timeline (its record runs to `yearMax`, so no ramp applies). One that stops earlier holds full size
*through* its last point and fades after it — Ctesiphon (record ends 622, abandoned for Baghdad)
is at 0.94× by 630 and 0.11× by 653. That is a claim, and the precedent is `plant_fades`, which
already fades a city out after its last real estimate; the ramp is short enough (2.5yr post-1900)
that a record ending in 1981 fades rather than asserting a decline.

**Result.** Over a 1× playthrough, the worst consecutive-frame radius change for a bubble present
in both frames is **1.17px**, against 14.5px before any of this. Continuous motion costs 0.0064px
per city-frame against 0.0172 before. What remains is the floor crossings of §4.3 — 3,052
appear/vanish events over 442,889 city-frames, mean radius 1.7px at the cut — which is two thirds
of all remaining radius change and is not something either mechanism here touches.

### 4.3 The map's display floor rises with the era (`mapFloor`)

5,000 is a city in the Bronze Age and a suburb in 1900, and the sources thin out in exactly the
opposite direction: at `MINPOP` there are 19,145 bubbles in 2025 and the small end is a grey wash,
on top of being most of `draw()`'s ~42ms. So the floor is 5,000 → **10,000 after AD 600** →
**20,000 after 1400**. Bubbles drawn: 1800 1,964 → 504, 1900 6,016 → 2,347, 2025 19,145 → 14,663.

**Ramped, not stepped.** A hard cut at 1400 takes 103 of the 292 bubbles on screen out between
two frames; log-linear over 600→700 and 1400→1500 they thin out across a century (~3s of play)
instead. This is the objection `GROWTH_TRACK`'s comment already records against `solidTol` being
a step function: a threshold every city shares becomes a visible instant the moment it moves. The
600 tier is free either way — no city on the map that year is between 5k and 10k.

Hides bubbles and nothing else. The data is untouched, `MINPOP` still governs the graph and the
data windows, and a city under the floor still draws its full line if you select it.

**Not fixed:** a city *crossing* the floor still pops, now at ~3.7px rather than ~1–2.5px. That
is pre-existing and far larger in volume than anything §4.2 addresses — the flat 5,000 floor
already popped ~1,300 cities/second in across 1800–1900 — and the raised floor roughly halves the
count (1,326 → 635/s there, 151 → 44/s across 1500–1800) while making each one bigger. The fix if
it ever matters is an alpha ramp over a narrow band just above the floor; it was left off because
it dims a large share of the small end for a problem the map has always had.

---

### 4.4 History notes (`data/events.json`)

The map shows that Vijayanagara dies between 1560 and 1570 and says nothing about why. History
notes are hand-curated one-liners that pop up at a point on the map as the run passes their year:
a coloured headline, an optional grey detail line, a leader to the spot. On by default; the
settings checkbox and `?notes=0` turn them off. A missing or unparseable `events.json` is
survivable — they are commentary on the map, not part of it, and the fetch is separate.

**Fired by crossing, then timed in wall clock.** A note starts its two seconds when the run
crosses its year and lives in milliseconds from there — `NOTE_IN`/`NOTE_HOLD`/`NOTE_OUT`
(120/2400/600 ms), all three stretched by the note's own `w`. Three seconds rather than two costs
density: at 2.1s the default level leaves the screen full 15% of the run, at 3s it is 35%, and
what a fuller screen buys is notes silently not placed.

The obvious alternative, a *window* around the note's year, was tried first and fails at both ends
of the run in either unit. Measured in years, a fixed window is a different number of seconds
everywhere, because the timeline gives 1400–1900 the same track width as 1900–2025. Measured in
scrubber fraction it is a fixed number of seconds — but then the year span balloons in the deep
past, and the Black Death is legible from 1206 to 1488. And both forms leave the 2022 Ukraine note
hanging permanently on the AD 2025 screen, which is where the run stops and where most people end
up sitting. Firing on the crossing gives every note the same two seconds whether it was passed at
1x, at 5x, or by dragging the scrubber, and then takes it away again.

Two cases the crossing model has to handle explicitly. **Landing** on a year rather than
travelling to it (first load, `?year=`) fires anything within half a second of playback of it,
scaled through `yearsPerSec` so "close enough that the run would have shown it" means the same in
1500 BC as in 1980. **A jump** — dragging the scrubber crosses centuries between two frames —
fires one note, the best-ranked one it passed, rather than everything in the interval, which
strobes. Frames come from the play loop while it is running; a note fired while paused, or still
fading when playback stops, drives its own `requestAnimationFrame` chain (`noteFrames`).

Drawn last in `draw()`, on the bubble canvas rather than in the DOM, so it inherits the world-copy
tiling and the halo trick the selected-city label already uses. Two departures from how a bubble
is drawn: a label is rendered **once**, at the world copy nearest the middle of the viewport
(a bubble is drawn per copy), and it is skipped entirely if that copy is off-screen.

**Placement happens once, at the moment a note fires, and never again** (`placeNote`, stored on
the note as an offset from its own anchor). The first version re-solved the layout every frame,
best `pri` first — correct at every instant, and unreadable in motion, because a note you were
halfway through reading got shoved aside the moment a higher-ranked one fired. Holding still
matters more than being optimally packed. Storing an *offset* rather than a screen position means
the label still rides with the map when you pan or zoom, which it must, since it is pointing at a
place; two notes can therefore collide under a zoom, and that is accepted.

The other half of holding still is **refusing to place a note that does not fit**. A new note is
laid out around whatever is already on screen, plus the panels (opaque, over the canvas, so
anything anchored in India or China was otherwise placed under the leaderboard column and simply
lost). It may be pushed `NOTE_STACK` = 2 slots from its natural spot to clear something; past
that it is dropped rather than parked somewhere its leader no longer explains. A note nobody
reads costs nothing; one sitting 150px from the city it is about is worse than absent.
`NOTE_MAX` = 5 caps it regardless, counted over everything still fading, not just what fires now.

**Coloured by what it did to the population, not by what kind of event it was** — and on the
bubbles' own scale, because `NOTE_IMPACT` is built by indexing `GROWTH_STOPS`. A note about a
city emptying is the same blue as the bubbles emptying under it. Seven steps: `-3.2%/yr` is not a
judgement anyone can make by hand about the Black Death, and the yellow stop is skipped so the
steps stay far enough apart to tell apart in 13px text. The key in Settings paints itself from
`NOTE_IMPACT` rather than from CSS, so it cannot drift from what the map draws.

| `im` | colour | |
|---|---|---|
| −3 | indigo | abandoned, never recovers |
| −2 | blue | heavy loss, recovers later |
| −1 | green | mild or slow decline |
| 0 | yellow-green | a marker: nothing on the map changes yet (Columbus, Dammam No. 7) |
| +1 | orange | steady growth |
| +2 | red | strong growth |
| +3 | magenta | explosive growth |

**Schema.** `y` year (negative = BC, no year 0), `la`/`lo` anchor, `t` headline (≤34 chars),
`im` impact; optional `d` detail (≤66 chars), `pri` 1 headline / 2 strong (default) / 3 flavour,
`w` life multiplier 0.5–3, `p` a curator's note that is never rendered. Anchoring is by
coordinate, not by city name: a note often belongs to a region rather than a city, and the best
ones (Columbus at San Salvador, Dammam No. 7 in 1938) point at a part of the map where nothing is
drawn *yet*.

**The file holds more notes than one run has room for, on purpose.** A playthrough at 1x is
`1/PLAY_SPEED` = 62 seconds and each note holds the screen for 2.1 of them, so the average number
on screen is `2.1 × N / 62`. At the file's 138 notes that is 4.6, and simulated at 1x the screen
is full 60% of the run with 17 notes that never once get drawn. So `noteLevel` — the
**key / more / all** selector in Settings, and `?notes=key` / `?notes=all` — picks how much of the
file to use, and the default is `more` (`pri` ≤ 2, 77 notes): 15% full, and everything in it
appears. The filter is applied at *firing*, not at drawing, so a note below the level never takes
a slot and never blocks one that would have been drawn.

Measured, not guessed. `check_events.py` runs the viewer's own loop — steps `s` at `PLAY_SPEED`,
fires what each step crosses, applies the same sort and `NOTE_MAX` cap — and prints which notes
never appear and which get under a second. Counting neighbours instead, which is what it did
first, called 67 notes "crowded" and said nothing about which ones a viewer would actually see.
The simulation keeps ticking after `s` passes 1, because `setPlaying(false)` hands the fade to
`noteFrames()` and the last notes of the timeline do get their full two seconds.

`pri` also keeps the level reversible: nothing has to be deleted to thin the map, and
`live.sort` gives the slots to the better-ranked note first, so an over-full file degrades by
dropping flavour rather than by being cut. `check_events.py` prints the average and the worst
moment for each tier.

`tools/check_events.py` validates the file against the viewer's own constants — it ports
`mapFloor`, `regionOf` and the year→`s` warp — and catches what reading the file cannot: a note
pinned outside the timeline, a coordinate with nothing on the map within 400km (a typo, unless it
is deliberate), text that will run off the screen, and notes that pile up more than `NOTE_MAX`
deep in the same two seconds of playback. `--coverage` prints a century × region table, which is
how you see what is missing.

---

## 5. validate.py

Seven checks, each named for the defect class that motivated it. Run after every build; a count
going the wrong way is the signal.

| | check | current |
|---|---|---|
| A | carry-forward (dead city frozen at its peak) | 10, of which 8 deliberate |
| B | whipsaw (one entry holding several units) | 74 (+38 at seam) |
| C | graft collapse (modern tail lost its agglomeration) | 17 |
| D | coord stacking (geocoder fell back to a centroid) | 3 points / 9 entries |
| E | terminal break (last point collapses) | 3 |
| F | definition oscillation | 10 (+175 at seam) |
| G | strip ate a benchmark | 170, 63 followed by a cliff |

**Checks B, E and F now ignore planted floor points** (`real_points()`, which reads the `f`
codes in `s`). A fade shoulder is ours, not a measurement, but all three checks measure ratios
between consecutive points, so a floor beside a real anchor read as a 50x collapse that nobody
introduced. Measured on one identical `cities.json`, filtering took **E from 11 to 2 and B from
85 to 69** — i.e. it removed pre-existing false positives as well as the ones
`bracket_lone_anchor` would have added. C is deliberately left alone: none of its rows are
floor-ended, and it measures peak-vs-tail rather than a step.

**Check A now marks its own successes.** A hunts "flat run then cliff", which is also the exact
shape of a `CF_KEEP` plateau and of a forward `CF_END`, so 8 of its 10 rows are decisions rather
than defects and are marked `*`. `deliberate_plateaus()` builds that set from build.py's tables,
resolving a MERGE_INTO donor to the label its target is drawn under.

Two things worth knowing about the numbers:

- **B and F were mostly re-reporting our own seam.** populstat's terminal figure is often
  metro/county while the WUP centre is the dense core, which is exactly the dip-and-recover shape
  both checks hunt. `sw` is emitted per grafted city and validate skips flips straddling it,
  reporting them separately. F went 296 → 26. C is deliberately **not** seam-skipped — it is a
  check *about* the graft.
- **Rank impact is charged only in the years a defect spans.** It used to flag an entry and then
  charge it in every sampled year, so Chengdu's 1936 flip counted against its year −50 ranking.
  Honest numbers: carry-forward 2% (entirely the deliberate `CF_KEEP` entries), oscillation 0%,
  coords 0%.

**Reference leaderboards** (regression canaries — these must reproduce):

- 1000 — Istanbul / Kaifeng / Kyôto / Mary / Baghdád   (Baghdád is 125,000 and 5th from
  2026-08-22; it read 1.1M and 1st while the withdrawn `CF_KEEP` entry stood — see §3.4a)
- 1200 — Istanbul / Hangzhou / Cairo / Fès
- 1400 — Nanjing / Beijing / Vijayanagara
- 1600 — Beijing / Istanbul / Âgra
- 800 — Baghdád / Xi'an / Istanbul / Dimashq / Luoyang

### 5.0b Checks E and H (2026-08-21)

**E was structurally blind to 62% of the map.** It compared `p[-2]` to `p[-1]`, and §3.11 holds a
city's last value flat to `YEAR_NOW` and marks the join `hx` — so for a held city those two
points are equal *by construction* and the ratio is 1.00 whatever the record did. 13,809 of
22,144 cities end on a flat segment, i.e. E could only ever see the grafted minority. Walking
back to `hx` first takes it 3 → 5 and surfaces **Codru** (653,000 → 11,500) and **Harburg**
(113,000 → 5,000 — whose series is Harburg near Hamburg while its coordinate is Harburg in
Bavaria, 600 km away).

**H, "never regains"** — a fall ≥2.5× within 15 years from ≥50,000 that never returns to 90% of
the pre-drop level. 79 hits, none of which B, E or F can see: B needs 8× and these are 2.5–6×,
F needs the series to come *back* inside 40 years and the whole point is that it never does, E
only looks at the final point, and despike is up-only.

Recovery time is the discriminator because the two hypotheses make opposite predictions. A
catastrophe is followed by a recovery — Hiroshima is back inside a decade, so are the 1945 war
troughs — because the place is still there. A definition change is not, because the smaller unit
was never going to reach the bigger one's number. Precision bears that out: of the hits only five
are pre-1900 and three of those five are real history (Amarapura 1820, Srirangapatna 1870,
Matsumae 1870); after 1900 it is essentially all district-or-borough figures falling to the town.
Rank impact 0%, so it is a shape defect rather than a wrong ranking. The graft seam is excluded
for the usual reason — past `sw` the neighbour is WUP and the step is `merge_series`', already
measured by `SWITCH_STEPS`.

### 5.1 tools/analyze_jumps.py

`validate.py` answers "is a known defect class getting worse". This answers the different
question of **where is the drawn line visibly discontinuous, and is that our fault** — the
worklist for both hand-fixing entries and deciding what to smooth.

Two choices make the output readable:

- **Speed is measured in adjusted years** (§2), so one threshold means the same thing to the eye
  everywhere: the default 50-adjusted-year window is ~2 real years after 1900, ~8 across
  1400–1900, ~25 across AD 1–1400, 50 before AD 1. Without the warp a 40-year Bronze Age
  doubling outranks a 2-year one in 1995, and the list is all antiquity.
- **Big for its time**, via a top-500 threshold curve recomputed from the built data (2k in
  AD 1000, 70k in 1900, 1.06M today), taken at the *larger* end of the jump so collapses are
  judged by what the city was. The curve is printed every run because of the one place it does
  nothing: before ~AD 600 the dataset holds fewer than 500 cities at once, so the filter is
  inert and the ancient counts are not comparable with the modern ones.

The window search is exact rather than sampled: log-pop is linear in real years between control
points and adjusted time is linear in real years within an era, so with the era edges inserted
as breakpoints, `L(t+w) − L(t)` is piecewise linear in `t` and every extremum sits on a
breakpoint. Overlapping windows describing the same jump are collapsed to the strongest, and a
rise and a fall sharing an instant stay two events.

Each jump is tagged with whichever of *our* mechanisms it sits on, so the artifacts sort
themselves out of the way — `seam` (§3.7), `fade` (a planted ramp, §3.10), `terminal` (the jump
lands on the last real row, which §3.11 then freezes to 2025), `step` (a long flat run either
side: populstat changed benchmark, not the city), and `other`. Two of those tests are less
obvious than they look: a fade has to be tested as an *interval*, because a window inside a
ramp has no floor point near its own ends (Yuzhou's 200→275 descent read as a seam step), and
the seam test has to be strict on the right, or a drop that merely *ends* at the seam is blamed
on it.

`segments` (control-point intervals spanned) separates a single straight step, which the eye
reads as a corner, from a steep but continuously-anchored climb, which is a real curve.

It also prints the seam step for **all** grafted cities, not just the ones clearing the filters
— the number the smoothing decision turns on — and looks for identical runs of three points in
three or more cities. That last one is ranked by size, not by how many cities share the block:
Buringh's medieval European towns share round modelled values by the hundred and it means
nothing, but four sites at exactly 43,100 is one curve copied (§6.10).

---

## 6. Known defects, open

Each of these has been investigated; the evidence is here so it is not re-derived.

### 6.1 The trim's drop branch can delete a city's real peak

**Confirmed case: Cleveland, and so far only Cleveland.** Its 922,900 held 1925–1999 is the 1950
census (914,808) repeated, and the branch deletes the whole run including that first real point,
so Cleveland runs 394,000 (1900) → 478,000 (2000) and never peaks. `strip_carry_forward`'s policy
— keep the run's first point, drop the repetition — is the right one and the trim does not follow
it.

Do **not** keep the plateau's first point unconditionally: Philadelphia-United States' starts at
4,853,649 (1974), which *is* the metro figure we rejected, and keeping it restores the trough.

Two separators tried, both failed:

- *"the dropped level exceeds the surviving peak"* — 142 of 265 drop-branch cases hit it, and the
  top of that list is Tembilahan (4.88M), Montevideo (10.9M) and a long tail of Indonesian
  **regency** figures, i.e. the trim working as designed. Cleveland is buried at 1.93x.
- *"does the deleted run jump off the previous point, or continue it smoothly"* — 212 of 247 read
  as smooth, because the first deleted point is the first point of the *fill*, which rises gently
  by construction. Chicago 1959:3,889,405 → 1960:3,995,239 is 1.03x while Cleveland 1900:394,089
  → 1906:460,300 is 1.17x, so Cleveland looks *less* smooth than the cases we want to drop.

The real difference is whether the plateau sits at the top of a ramp toward a bigger unit or is a
census held down the years; neither test measures that. 265 cities in this branch.

### 6.2 The modern layer manufactures growth for low-density metros

WUP defines a city as the connected blob of cells above ~1,500 people/km². As suburbs thicken,
more cells cross the threshold and join the blob — so the number rises because the *definition*
is eating territory. **261 of the 1,866 WUP centres ≥300k in 2025 grew ≥3x since 1995; 682 grew
≥2x.** Some is real (Doha, Abuja, Las Vegas); much is not — Luxor 45.9x, Mesa 15.2x, Columbus
8.7x (83,665 → 731,335), Atlanta 8.2x (56,211 → 458,225). Memphis TN: WUP 92,515 (1995) →
356,215 (2025), against a city that barely grew.

The tell is **Detroit**, whose density fell and whose blob therefore *shrank* — not something
that happens to a real measure of population.

`SWITCH_STEPS` measures the discontinuity *at* the handover and says nothing about the fake slope
*after* it, so there is currently no metric for this. First move is to add one (WUP 1995→2025
growth vs the historical series' own late trend) so the size is visible. No fix inside this
pipeline without another source.

Caveat on the graft fix in §3.5: Antioch now reaches 2025 *through* this artifact, 74k (1975) →
530k (2025). Antakya did not grow sevenfold.

The **seam** has the same flavour and is separate from the slope: Delhi steps 7,210,000 (2000,
populstat city proper) → 18,500,000 (2001, WUP agglomeration) in one year. 35 of the 166 grafted
cities peaking ≥3M step ≥2x. This is structural rather than a bug — both numbers are honest for
different things — but Delhi is where people look, so expect it to be reported as a defect.
Lagos is the inverse and worth knowing: populstat ends at a stale 1,430,000 and WUP's 7.91M is
the accurate figure, so there the modern source is the one telling the truth.

### 6.3 Definition oscillation remainder (validate F)

~25 real entries once seam flips are excluded, and 5–6 of those should be left alone. They are
**alternating runs** — the source interleaves two definitions for decades, so there is no single
interloper to delete. Beijing 1861–1948, Wuhan, Chongqing, Hangzhou, Johannesburg.
`python tools/analyze_osc.py --detail` prints the worklist.

Verdict: **hand fix**. Every general rule fails on evidence already in the dataset:

- *"delete the short up-plateau exceeding both flanks"* erases Japan's pre-war peak — the same
  mechanical signature (an exactly-equal pair, 1935 = 1940) is the 1935 census held to 1940 for
  Tokyo 5,875,700, Osaka, Kyoto, Kobe + 7 more. Tokyo's flanks agree to 1.17x and its amplitude
  is 1.81x, so neither guard saves it.
- **Wuhan inverts the rule**: there the *high* rows are correct (the tri-city Hankou + Wuchang +
  Hanyang) and the low ones are Hankou alone. No Hankou entry exists, so an entry named Wuhan
  must carry the tri-city figure.
- Majority vote is not truth: Newcastle's *wrong* rows outnumber its right ones 4:3, and
  Frederiksberg's minority ~105k rows were the accurate ones.

Why despike is structurally blind: populstat holds its benchmark verbatim to the next round year,
so the wrong-unit value arrives as an exactly-**equal pair** and neither member is out of line
with both neighbours.

**APPLIED** (2026-08-15). 22 entries, each verified against the raw rows first, which caught two
errors in the proposal. Check F fell **27 → 10**, rank impact 0%.

- `DROP_YEARS`, conurbation pairs: Birmingham, Liverpool, Newcastle, Dublin, Edinburgh, Glasgow,
  Katowice, Calcutta, Johannesburg, Antwerpen, Frankfurt, Bordeaux, Marseille, Riga.
- `DROP_YEARS`, China: Beijing, Wuhan, Hangzhou, Changsha, Chongqing, Chengdu, Harbin, Wenzhou.
- `CLIP_BEFORE`: Qingdao 1911, Gentofte 1921 (Gentofte carries Copenhagen 1780–1890, genuine from
  34,500 in 1921 — Frederiksberg again). `DROP_KEYS`: Mont-Royal (carries Montreal; Montréal
  exists 0.6 km away with a fuller series).
- All ranges, never bare years, so the source's straight-line fill goes with each anchor.

Two proposals were **wrong and were corrected**:

- **Hangzhou** was proposed as 1876–1924. That range swallows the genuine 1911 (594,000) and 1918
  (684,100) censuses and leaves a 50-year hole. Narrowed to 1876–1910, which kills the carried
  400,000 and its fill while keeping both.
- **St Petersburg** was proposed as a bare `{1920}`, but the raw holds 2,318,600 flat from 1915 to
  ~1924, so one year does nothing — and the 722,000 census cited as the payoff is not in the
  source at all. That is a carry-forward, not a flip. **Held back**, with Guangzhou and Shanghai
  (compilers disagreeing rather than two definitions).

**Known consequence, wants a look in the product:** Beijing now reads ~750,000 in 1900 and leaves
the 1900 top five. The source has no real anchor between 1875 (900,000) and 1911 (693,000) — the
whole span was the carried 1850 figure — so 750k is what interpolation gives once that is removed.
Chandler puts Beijing nearer 1.1M in 1900, but that figure is not in this source; the choice is
between it and a value we know is thirty years stale.

**Leave alone:** the war troughs (Köln/Nürnberg/Wrocław 1946, Kobe 1945, Riga 1917) — all real.

### 6.4 Graft principal — why it is a hand list

No general rule is available, and the evidence says so rather than the effort running out.

- A **size gate** ("name only wins within Nx of the biggest claimant") does not separate: at 3x
  it hands Jerusalem — our own `RENAME` of Al-Quds, 2,800 years deep — to the modern
  "Yerushalayim" entry, which is this defect inverted.
- **"and starts earlier"** fixes Jerusalem and still fires on ten pairs of genuinely separate
  cities: Rawalpindi takes Islamabad's centre, Haifa takes Hadera's 40km away, Menton takes
  Monaco's, Savannakhét takes Mukdahan's across the Mekong *and* an international border. ~65%
  precision.
- **`is_agglomeration_of`** would be the principled signal and is populated for only 4 of the 45
  candidates — it covers suburbs of famous cities, not the Parma/Bethany/Cannock tier.

Residual: Venice still resolves to Mestre's centre rather than the 67,753 "Venice" centre, which
`TIGHT_MIN_FRAC` rejects at 0.16. Neither is really right — Venice's true agglomeration is both.

### 6.5 Orphans and duplicates the strict join exposed

All pre-existing, newly visible. Duplicate entries dedup missed because coords differ by >0.1°:
Karâchi (4.9M, ends 1981) beside Karachi (21.4M, 2025), Miensk/Minsk, To`skent/Tashkent,
Ni`'znij Novgorod, a second Minneapolis. Plus mis-geocoded Jilin (3km from Changchun's centroid)
and "Jîzah, Al-" (Giza, nothing within 60km).

Genuinely lost by the strict join: cities whose WUP centre carries a different name 15–40km away.
Conakry's centre is "Coyah (Conacry)" 25km off, so it now ends 1996. These resolve themselves
once WUP-only cities are added.

Residual duplicate: a second "Zagreb (town)" 771k entry draws alongside "Zagreb (municip.)",
because "(town)" is a city-*proper* marker, not a metro one. Same shape as "La Habana Viejo" /
"La Habana del Este", which are districts. Wants the `subdistricts.txt` treatment, not a marker.

### 6.6 V-troughs remaining

41 left, a different shape from §3.6 — continental European cities whose mid-century values are
agglomeration and whose tail is city proper, but without the flat-plateau-then-census signature
the trim keys on: Frankfurt 1.45M → 0.65M → 0.90M, Lyon 1.17 → 0.45 → 1.20, Mannheim,
Düsseldorf, Washington, Manchester.

### 6.7 `is_agglomeration_of` is in the source and we ignore it

2,584 entries carry it (the parent city's name); 2,297 have their parent on the map too. That is
Stadestér saying, per entry, "this is a piece of that" — Yokohama inside Tokyo, Iztapalapa inside
Mexico City, Bogor and Tangerang inside Jakarta. Not a drop list (Yokohama is a real city people
expect to see) but it is the definition flag the pipeline has been missing, and the principled
version of the hand-built drop lists. Related: `particulars` (4,552 entries of free text) and
`certainty` (Chandler's own 1–3 rating, 190 entries).

Note its coverage is patchy — see §6.4, where it resolved 4 of 45 cases.

### 6.8 Modern era thins out

Grafted principals reach 2025 but WUP has ~6.4k more centres not in the historical set: 6,402
unclaimed, ~22 ≥1M, ~88 ≥500k, ~241 ≥300k. Gate on `Pop_plausibility` (in the source csv, we
currently discard it; 946 of the unclaimed are flagged Low) — that flag separates a real city
from a contiguity blob in the Bihar/Fars/Nile rural corridors (Dighwara 2.5M, Mehsi 2.0M are not
cities). Also decide what a WUP-only city's pre-1975 series is: with none it pops into existence
full-size at 1975, the same defect the New World ramp exists to fix.

### 6.9 Terminal unit switch — the residue after local arbitration

§3.6 now arbitrates the WUP-less entries against their own record, which took terminal-class
jumps from 34 to 19. Two shapes are left, and neither should be fixed by loosening that rule.

**Whole-series mis-joins.** Funza is still the single largest jump on the map (×90.8): 19,500 in
1776 rising smoothly to 4,530,000 in 2001, then one row at 49,900. Every figure in it is Bogotá's
except the last, so the plateau is *continuous* with the record and the local arbiter correctly
refuses it — there is no terminal unit switch here, there is a bad join. This is §6.5's family
and it needs `MERGE_INTO` / `DROP_KEYS`, or `is_agglomeration_of` (§6.7), which would settle it
directly. Deleting the 49,900 instead would hold a 4.5M phantom Bogotá-under-another-name to
2025, which is why the rule is one-directional.

**Anchors older than the lookback.** Ciledug (×13.9) has 40 rows before its plateau, but they are
one straight-line fill running 58 years from 20,000 (1930) to 1,223,797 (1988). `base` is the
minimum over the last 40 of those years, which is a fill point at ~393,000, so the plateau reads
as continuous and the trim declines. Widening `TRIM_LOCAL_LOOKBACK` fixes Ciledug and weakens the
`TRIM_LOCAL_TAIL_FLOOR` guard everywhere else, because a longer window reaches further down a
growing city's record and makes a junk final row look plausible — that is exactly the Sefton
failure. The right fix is to find the fill's start rather than to look back a fixed distance.

**The China 1989→1990 pairs** (Boyang, Changshou, Nahe, Juancheng, Qixia, Jiangling, Yichuan) are
not this defect at all: a four-row administrative record where *both* levels are the county. They
belong to §6.3 and want a source that reports 市辖区, per §7.

### 6.10 One curve copied across several New World entries

Caracol, Tiahuanaco, Tikal and Tula de Allende all open `200BC:800 / AD0:43,100 / AD100:100,000`
— the same three figures to the digit, for four sites on two continents — then diverge. Chandler–
Modelski appears to carry one New World curve across several entries rather than four independent
estimates, and because every copy contributes the same jump it inflates the pre-AD1 counts in
§5.1's output fourfold.

Detected by run-of-three matching rather than whole-series equality (they diverge after the
shared opening) and ranked by size rather than by how many cities share the block: Buringh's
medieval European towns share round modelled values by the hundred and that means nothing, but
nobody independently estimates four cities at 43,100. Only two other blocks ≥25k are shared by
3+ cities, both Chandler's round Greek/Phoenician figures (Agrigento/Argos/Taranto at 40,000 in
430 BC), which are plausible as stated.

### 6.11 What the fallback drop costs

`drop_fallback_stacks()` (§3.1) removes 375 mislocated entries, and check D falls 51 points /
352 entries → 3 / 9. It is a placement fix, not a data fix: the towns are gone from the map, not
corrected. Every one of them carries its standard transliteration in `other_names` (`Gat`'cina`
→ Gatchina, `Mon`'cegorsk` → Monchegorsk), so a coordinate source that goes below WUP's 50k floor
would restore the lot — GeoNames, already noted in §7 as "fine as a coordinate spine", is the
obvious candidate and the reason to keep this listed as a defect rather than a solved problem.

The largest losses are Teesside (393,800, on the UK centroid), Xuguit Qi / Daxian (425,600, on
China's), and 13 Kuwaiti entries topping out at 114,800. Nothing above 100k survives on a
centroid, and nothing lost is a city the map would be judged on.

`Atomgrad-Russia` is the one member of that Siberian pile settled permanently, by `DROP_KEYS`
rather than by placement: no coordinate, no `other_names`, and a single row (97,500 in 1991).
"Atom city" was a nickname several Soviet nuclear towns shared, so unlike its neighbours it is
not waiting on a gazetteer — it cannot be identified at all.

---

### 6.12 Open findings from the 2026-08-21 scan sweep

Found and measured, not yet fixed. Ordered by visibility. "ty" = real years the city holds an
on-screen top-12 slot over a 406-year sample grid.

**~~Haifa is drawn 35 km from Haifa.~~ FIXED.** `Hefa-Israel`'s coordinate was `32.4814,
34.9948`, next to Hadera. So the 485k bubble where Haifa actually is was labelled **Qiryat
Motzkin** (a suburb of ~40k that won the centre and stepped 14.6× at its seam), and the only
"Hefa" was a 276k dot down the coast, ungrafted and frozen 2002→2025 — no entry named Haifa
within 12 km of Haifa. Note §6.4 lists "Haifa takes Hadera's centre 40 km away" as a *false
positive* of a rejected rule; the coordinate that made that sentence true was itself the bug.
Fixed by `MANUAL` in `tools/make_coordfix.py` (→ WUP's own centre, which is named `Haifa`) plus
`GRAFT_PRINCIPAL_WINS`. Hefa now runs to 485,000 at 2025 and Qiryat Motzkin reverts to its own
40,400. Found by a precise rule worth keeping: *an `is_agglomeration_of` child that steps ≥3× at
its own seam **and** ends larger than the parent it names* — **6 hits in the dataset, 5 real, 0
junk**. **Still open from that rule: Durg/Bhilai**, where the smaller twin won the centre and
runs to 1.39M while Bhilai — the larger city — is frozen at 685,000 from 1991.

**~~`chandler_modelski_key` names the ancient city and we only read it in comments.~~ SWEPT
2026-08-21, 9 entries applied.** The field is on 1,497 entries and is the systematic version of
`RENAME`. The scan must compare against the **drawn** name, i.e. post-`RENAME` — comparing raw
source names conflates already-fixed entries with new ones and roughly triples the apparent
yield. Done correctly: 65 candidates (CM name materially different by difflib ≤0.72, neither a
substring, not in the entry's `other_names`, **and** pre-1500 data ≥5,000).

**Most candidates are not defects.** The large majority are **exonyms**, where the map's
local-name style is already right: Halab/Aleppo, Dimashq/Damascus, Napoli/Naples, Wien/Vienna,
Moskva/Moscow, Köln/Cologne, Kyjiv/Kiev, Makkah/Mecca, Venezia/Venice, Firenze/Florence and ~30
more. Renaming those is a policy change, not a repair. What remains is this table's actual
subject — a modern town standing in for the ancient city underneath it.

Applied: **Susa (Shush)** — held a top-12 slot for 1,200 sampled years labelled as a town of
63,000; **Merv (Mary)**; **Cyrene (Shahat)**; **Petra (Wadi Musa)**; **Ostia (Fiumicino)**;
**Zafar (Yarim)**; **Anshan (Aliabad)** — flagged lower-confidence, since chandlerV2 has no
`Anshan-Iran` row so the identification is the source's key and an 8.6 km coordinate without the
value corroboration every other entry has; plus the two-part **Kathmandu** repair below.

**Deliberately not applied**, with reasons, so they are not re-walked:
- **Argos = Mycenae.** Chandler's Mycenae row is a *single* value, −1360 = 30,000. The rest of
  the entry (−430 = 40,000, and the modern tail) is genuinely Argos. One label cannot be both,
  and there is no separate Mycenae entry to move the Bronze Age point to — the Copenhagen trap.
  Wants a split, which the pipeline has no mechanism for.
- **Krivodol = Ohrid.** The series *is* Ohrid's (1000 = 40,000, Samuel's Bulgarian capital, and
  this is check A's standing `40k frozen 1000→1300`) but chandlerV2's own Ohrid coordinate is in
  **Bulgaria**, ~300 km from Ohrid, which is how the geocoder landed on Krivodol. Renaming
  without a coordinate move would put "Ohrid" 300 km from Ohrid, and Ohrid at ~42,000 today may
  be under WUP's 50k floor, so there may be no centre to move it to.
- **Mussayab = Akkad, Balad = Akshak.** Both ancient sites are **unlocated** — asserting them on
  a modern town is a stronger claim than the source can support.
- **Bûr Sa'îd = Tinnis, Trujillo = Chan Chan, Jember = Majapahit.** The entry is dominated by a
  large modern city (Port Said 850k, Trujillo 949k, Jember 530k), so a rename mislabels the
  modern half. These are `CLIP_BEFORE` cases, not renames.
- **Guiyang = "Guizhou"** is a province name, i.e. a bad link.

`chandler_modelski_coords` gives the ancient coordinate too, but 5–10 km offsets are mostly
benign centroid differences (Luoyang 5.8, Xi'an 8.5, Delhi 13.5), so the coordinate test is only
safe alongside the name test.

**Still open: it settles §6.9's largest jump for free — `Funza-Colombia`'s CM key is
`Bogota-Columbia`.**

#### Kathmandu — the same shape as Haifa, on a national capital

Found by the sweep and needing all three tables at once. WUP has a centre named `Kathmandu`
(3,231,516). It was won by **Pâtan** — Lalitpur, the twin city 4.6 km south, whose own record
peaks at 161,600 — so the map drew Kathmandu as Patan at 3.23M. The reason is that Kathmandu's
own entry, `Kâthmândau-Nepal` (1911–2001, peak 696,900), was **geocoded 90 km north-west** into
the Gandaki hills and so never reached its centre; it was drawn frozen at 697,000. Meanwhile the
city's *pre-1911* history sat 0.15 km from that centre under the name **"Sihara"**, whose CM key
is literally `Kathmandu-Nepal` and whose series is Chandler's Kathmandu row (630 = 22,000, exact).

Promoting Sihara directly does not work and should not: at a peak of 40,000 against a 3.2M
centre it is below `GRAFT_MIN_FRAC`, which is the guard doing its job. The repair is
`MANUAL` (coordinate, both the entry and its agglomeration variant) → `RENAME` (`Kâthmândau`
fails `names_agree` against `Kathmandu` — `norm()` gives "kathmandau", neither a substring of
the other) → `MERGE_INTO` (`Sihara-Nepal` → `Kâthmândau-Nepal`, the two never overlap). Kathmandu
now runs 630–2025, peaks at 3.23M, and Patan reverts to its own 162,000.

**~~Alexandria halves at AD 1 and hands over the #1 slot.~~ FIXED** — see §3.2b. It was the
year-0 row, which is never a datum; dropping it spreads the same decline over the 101 years the
source is actually claiming.

**APPLIED 2026-08-21.** The batch below is done: `Hollywood/Ontario/Mesquite/Aurora/Westminster/
Springfield-United States` → `DROP_KEYS` (same series, wrong state — Hollywood's Florida figures
were drawn inside Los Angeles); `Mesquita-Brazil` → `DROP_KEYS` (Nova Iguaçu's series entire,
held forward at 1.29M); `Belford Roxo` and `Queimados` → `CLIP_BEFORE 2000`; `Vorst` →
`CLIP_BEFORE 1901` + `DROP_YEARS {1999, 2000}`; `Chorzów` → `DROP_YEARS 1906–1920`;
`San Bernardino-United States` → `MERGE_INTO` (county held flat 1974–1999 on a city people know);
`Bhilai` → `MERGE_INTO Durg` + `RENAME "Durg-Bhilai"` (2.08M drawn for 1.4M of people); and the
`Waukesha-Wisconsin` row commented out of `tools/us_modern.csv`. Check C fell 17 → 11 and E 3 → 2
before the E fix below. **Still open from that sweep: Hartford** — its `-United States` copy is at
Hartford, Wisconsin but carries Hartford CT's *city proper* (164,440), which is the more accurate
of the two, since `Hartford-Connecticut` holds the 905,091 metro. Dropping it keeps the worse
number, keeping it keeps the wrong place; it wants a coordinate fix.

**~~Two structural blind spots in validate.~~ BOTH FIXED — see §5.0b.** Kept here for the
measurements:
*Check E is blind to 62% of the map*: it reads `p[-2]` vs `p[-1]`, and §3.11's hold-forward makes
those equal by construction for **13,809 of 22,144 cities**. Re-pointing it at the last non-flat
point takes it 3 → 6 hits and surfaces **Codru** (`2000:653,000 → 2001:11,500`) and **Harburg**
(`1999:113,000 → 2000:5,000` — and its series is Harburg-near-Hamburg while its coordinate is
Harburg in Bavaria, 600 km away).
*A "never regains" check would be new*: a fall ≥2.5× within 15y from ≥50k that never returns to
0.9× of the pre-drop level, excluding the seam → 86 hits, **78 not covered by B/E/F**, 14 with
peak ≥200k. Only 5 are pre-1900 and 3 of those are real history, so post-1900 precision is
essentially total. Worst: **Belford Roxo and Queimados**, two 1.2M dots 15 km apart west of Rio
through the 1990s carrying near-identical values; **San Bernardino CA** holding the county figure
525,000 for 25 years; **Chorzów** `1914–1920: 500,000` against 66k/75k flanks — the §6.3
equal-pair defeating F on both legs; **Vorst (Forest, Brussels)** carrying Brussels' entire
pre-1900 series (173,000 at 1900 → 10,600 at 1901), a new instance of the Frederiksberg/Gazi
family; Drapetsona carrying Piraeus; the London/UK boroughs §3.6's `TRIM_LOCAL_WHOLE_MAX`
deliberately spares.

**Duplicate series drawn twice, 15 new instances of §6.5.** Eight `X-<State>` vs `X-United
States` pairs with identical series and different coordinates, so dedup kept both and one dot is
in the wrong place entirely: **Hollywood's 139,000 is drawn inside Los Angeles** 3,922 km from
the Florida entry, Ontario NY 3,847 km from Ontario CA, Mesquite, Aurora, Hartford, Westminster,
Springfield. Plus 16 co-located pairs sharing ≥3 identical points (Honolulu/Honolulu CDP at
0.0 km, Augusta/Augusta-Richmond County, Athens/Athens-Clarke County).

**One bad row in `tools/us_modern.csv`:** `Waukesha-Wisconsin` has the **City** figure for 2010
and the **Town** for 2020/2024, drawing `2010:70,700 → 2020:8,470`. §3.3c validates 2010 against
populstat and never checks 2020 came from the same record. `y2010/y2020 ≥ 2` gives **exactly 1
hit in 1,594 rows**. Must be one-directional — the inverse gives 3 hits, all real growth
(Kirkland WA's 2011 annexations, Conroe TX, Williston ND).

**Tested and low-yield, recorded so it is not re-walked.** Round-number-vs-census-shaped
provenance: 3 hits in the whole dataset — populstat's modern rows are already census-shaped.
`is_agglomeration_of` × parent value on its own: 111 children exceed their parent but most are
real cities larger than a loosely-assigned parent (Nice>Cannes, Dortmund>Essen), and only 2 hold
a top-12 slot; the identical-values and ratio≈1 variants give 4 and 0 hits, because Funza, Gazi
and Frederiksberg do not carry the parent's numbers verbatim and **Gazi and Frederiksberg have
no `is_agglomeration_of` at all**. Only the seam-step version (above) is worth building on.
`found.yr` standalone is ~30% precise (dirty values like `16401681`, BC years stored positive)
but is good **corroboration** — it independently confirms Bûr Sa'îd, and adds Mérida MX (1542,
data from 900), Trujillo PE (1535 / Chan Chan), Faizâbâd (1730 / Ayodhya).
A source-grid seam at **AD 622** (Chandler's Hijra benchmark) is real — 51 cities drawn at 622
against 36 at 621 — but it is ±0.27M and moves no top-12 city. Not worth a fix; AD 1 is.

**~~Wants a decision: Lahore.~~ RESOLVED — no action.** The apparent 7–9× disagreement is
Chandler contradicting *himself*, not us. His Lahore row reads 1590: 300,000 · 1600: **34,000** ·
1622: 500,000 · 1627: 500,000 · 1631: **84,000** · 1650: 360,000 · 1700: **54,000** · 1707:
54,000. Nobody thinks Lahore went 300,000 → 34,000 → 500,000 in thirty-two years: that is a
definition oscillation inside a single source row, the same family as §6.3's Beijing and Wuhan,
and the low figures are presumably the walled city against the metropolis. The map is already
taking the coherent half.

**~~Rome: pick a source or accept the seam.~~ RESOLVED — `DROP_YEARS`, and it is the most
editorial entry in that table.** Rome fell 999,497 (300) → 150,000 (361) — 6.7× in 61 years —
and then held 150,000 flat for 239. Neither half is a measurement. The cliff is a source switch
(300 is populstat, 361 is Chandler, and §4.0b now draws that step white); the plateau is
Chandler's 361 benchmark carried forward while Chandler's own row says 100,000 at 500 and 50,000
at 600. Since we already prefer populstat on *both* sides — 999,497 over Chandler's 450,000 at
100, and 150,000 over his 50,000 at 600 — the 361 row is the one place the entry switches and is
exactly what manufactures the cliff. Dropping it and its fill gives **729k at 350, 531k at 400,
282k at 500, 206k at 550**: a late-antique decline rather than a cliff and a plateau, matching
the mainstream reading. The cost is that the 600 → 622 step (150,000 → 50,000) becomes the
visible seam instead — which is the honest one, since it lands in the right era for Rome's real
collapse and the growth mode now labels it as a handover.

### 6.13 Open findings from the 2026-08-22 pass

Ten cases were looked at. Seven were pipeline defects and are fixed — §3.4a, §3.6b, and the
`DROP_YEARS` / `CLIP_BEFORE` / `CENSUS` / `CF_END` / `DISAPPEARED` entries for Benin City,
Kaohsiung, Abomey, Allada and Córdoba. Three (Nubia before AD 800, Elmina/Ouidah, Aksum's
floruit) are absences of *data* and cannot be fixed with the machinery that exists; GAPS.md
Part 4 has each one. What follows is what the pass turned up and left.

**A `CF_KEEP` entry hides itself from the check that would catch it.** `strip_carry_forward` and
check G both `continue` on a `CF_KEEP` key before doing any work, so once an exemption is written
neither the strip nor the compiler test ever looks at that run again. Baghdad's stood for a day
and was reported by check A the whole time, marked `*` — "deliberate, not a defect" — which is
exactly what suppressed it. **The cheap mitigation** is for check A's `*` line to print
Chandler's value at each exempted year beside the run's, so a `CF_KEEP` entry that contradicts
Chandler is visible in the report that is supposed to be reassuring. Not implemented.

**Check C cannot see a graft onto the wrong place when the wrong place grows.** It scores the
historical peak against the post-2000 *maximum*, so Soweto → Lenasia (10.4× down at the handover)
lands at peak/2.8 and passes the peak/5 gate, because Lenasia trebles after 2000. The quantity
that catches it is the ratio **at the seam**, restricted to name-mismatched `TIGHT` matches —
where a large drop cannot be the definitional seam §3.7 exists for, since a `TIGHT` match is by
construction not a city-versus-its-own-dense-core comparison. Worth measuring across the 6,656
grafted cities before it becomes either a check or a `TIGHT_MIN_FRAC` replacement; both were left
alone here because `TIGHT_MIN_FRAC`'s note records 27 catches and 0 false positives at 0.2 and
Soweto sits at 0.36, well clear.

**Check G's 5km coordinate join is a measurable blind spot.** It joins an entry to its Chandler
row by position, because — the docstring says — "the shipped `stadester_cities.json` has no
`chandler_modelski_key` to join on". **That is no longer true**: §6.12's sweep found the field on
**1,497 entries**, and where it is present it is an exact join needing no radius at all. Where it
is absent the coordinate is the only option, and 5km is too tight for the entries whose Chandler
geocode is itself sloppy — Córdoba's is 20km off the city (§4.9 of GAPS.md), which is why a
500-year carry-forward at the top of the AD 1000 frame was never reported.

Re-running check G's own logic at wider radii: **5km → 169 runs, 10km → 185 (+16, 5 with a ≥2×
cliff), 20km → 189, 30km → 193 (+24, 9 steep)**. Real catches in the new band include Xi'an
(`805..1000` at 600,000, `/13.3×`, 8.5km), Basra (`1123..1500`, `/6.0×`), Luoyang and Kano.
But the radius **cannot simply be widened**: 30km also matches `Lille-Belgium` to Chandler's
*Antwerp*, `Xianyang-China` to *Xi'an* and `Sololá-Guatemala` to *Q'umarkaj*, which are different
cities. The design that follows from that is three-tier — `chandler_modelski_key` where present
(exact, no radius), then coordinates within 5km unchanged, then 5–15km **gated on name agreement**
using the existing `names_agree()`. Not implemented; the tier-3 hits want reviewing one by one
before they become report lines.

**`CENSUS`-recovered figures read as interpolation in the provenance string.** `source_codes()`
classifies a control point by asking `provenance.py` about its *year* in the source entry, and
`CENSUS` replaces values at years the source already had — so every recovered figure inherits the
attribution of the number it replaced, or falls through to `i`. Teotihuacán's restored 500:
125,000 is `i`; Lhasa's 1700/1750/1840 are `i`; Baghdad's restored 1100 is `p` when the figure is
Chandler's. These are among the most carefully sourced points in the dataset and the `s` string
calls them fill.
**No visual effect today** — `SOURCE_CHANGE_ON = false` in index.html and `sourceSpans()` is `s`'s
only consumer — but it is wrong the moment that flag is flipped, and it would draw three spurious
compiler handovers across Baghdad's 933–1400, which is a span with a single source. The fix is to
carry the intended code on the `CENSUS` value (recovered-Chandler → `c`; a hand-typed census is
neither `c` nor `p` and probably wants its own letter, which means a viewer change too).

**Córdoba is drawn at 79,000 in AD 1000, and it is now the top of that frame's shortfall.** With
Baghdad corrected the AD 1000 leaderboard reads Istanbul 330k / Kaifeng 321k / Kyôto 155k / Merv
144k / Baghdád 125k, and Córdoba — which Chandler puts first in the world that year — is 79,100.
Not a simple recovery: the entry is a **Buringh** one (`t=buringh`) whose 700/800/900/1000 values
(18,000 · 56,500 · 78,200 · 79,100) are a coherent modern model series, and Chandler's row is not
coherent at all — `800: 160,000 · 900: 20,000 · 1000: 450,000 · 1100: 60,000`, a 22× spike between
two benchmarks. What *is* a plain defect is the hole after it: the Buringh grid runs 800–1800 at
100-year steps and Córdoba's entry jumps 1000 → 1550, so the map draws a straight line through
both the 1009–1031 fitna and the 1236 Reconquista. Check G cannot see it either — Chandler's
coordinate for Córdoba is 20km off the city, outside the 5km join.

## 7. Source survey

**No drop-in replacement for populstat exists** — tens of thousands of cities, global, NSO
agglomeration definitions, deep series, bulk CSV is not a product anyone ships.

| source | verdict |
|---|---|
| **UNSD Demographic Yearbook T8** ([table 240](http://data.un.org/Data.aspx?d=POP&f=tableCode%3A240)) | The only global source stating the definition **per row** (City proper / Urban agglomeration). 5,271 cities, 1970–2025, free CSV. **Tested against the trough list and mostly does not help**: 35/65 found by name, only 14 have an agglomeration row, only 13 carry both labels. Fails on the biggest cluster — Germany reports only "City proper", so Frankfurt/Mannheim/Düsseldorf/Nürnberg have 11 rows each and zero agglomeration figures. Keep as an anchor for countries that *do* report agglomerations. |
| citypopulation.de | Closest definitional match, CC BY 3.0, per-entry City/Aggl/Town labels — but no bulk download, no coordinates, 3–4 census points per city. Fixes definition, adds no history. |
| Buringh, European urban population 700–2000 | DANS, CC0, 2,262 cities with lat/lon, extends ours to 2000 at 50-yr steps. Balanced panel, 47% imputed — filter on `natureofestimate`. |
| **GHS-FUA / eFUA R2019A** | **ADOPTED, §3.5c.** 9,031 functional urban areas worldwide, EC reuse licence, 3 MB gpkg (`data/efua.gpkg`). The only global source on a *metropolitan* delineation rather than a density one, which is what the seam actually needs. Ships one epoch (`FUA_p_2015`), so the series is built from its member WUP centres. |
| **US Census CBSA** | **ADOPTED, §3.5c.** Annual MSA 2000–2024, 373 metros, public domain, three csv files. Strictly better than eFUA inside the US and the only fix for the fragmentation. |
| **Africapolis** | **BUILT, DEFAULT OFF — `--africapolis`. See §3.5d.** OECD/SWAC 2025 update (v9), CC BY 4.0, `data/Africapolis_agglomeration_2025.xlsx`, browser download from africapolis.org/en/data (the data browser builds the CSV client-side, so curl cannot reach it). 12,904 agglomerations, 8,763 usable, nine observed epochs 1950–2020. `prep_africapolis.py` matches 1,899 of them to WUP centres. |
| **A China-specific source — not yet looked for, and the clearest remaining gap.** | `trim_admin_tail` (§3.5b) stops populstat's county and municipality figures being drawn as cities, but it leaves the modern half on WUP's DEGURBA centre, and for China that is conservative: Chongqing now reads 7.07M in 2025, against an urban built-up area usually put at two or three times that. Right shape but low. 82 entries are also refused by the trim because their series is administrative throughout (Wuwei starts at 804,000), and only a source that reports Chinese *urban district* population — 市辖区, the standard NBS unit — can fix those at all. China Statistical Yearbook / NBS city-level tables are the obvious place to look. Untried. |
| Geopolis (Moriconi-Ebrard) | ~26,000 agglomerations 1800–present on exactly our preferred definition. **No public download** of the global database. Effectively unobtainable without contacting them. |
| Dead ends | Demographia (PDF-only, licence forbids alteration, and it is urban area not metro); GHS-UCDB (same DEGURBA definition — relabels the seam, does not fix it); OECD FUA (37 countries, and eFUA is the global version of the same delineation); OWID (100 cities repackaging WUP2025); Reba/Reitsma/Seto (built *from* Chandler+Modelski, already ours); Wikidata (1,865 cities with ≥3 dated values); GeoNames (undated snapshot, fine as a coordinate spine); World Gazetteer (dead); Oxford Economics Global Cities (paid). |
| UN WUP **2018** | Not tried, and the one obvious remaining option: 1,860 agglomerations ≥300k, 1950–2035, on national agglomeration/metro definitions — i.e. the *pre*-DEGURBA revision, which is the "mostly metro" source the historical side blends with. Small roster, but it covers exactly the cities where a seam is visible. Usable as a fourth layer above eFUA where it has a city. |

---

## 8. Rejected approaches

Short list, so none of these get re-attempted:

- **Per-city active-span trim** (hand-declaring when each city was alive) — rejected in favour of
  the general, data-driven fade. **Partially reversed:** `DISAPPEARED` (§3.10b) is exactly this
  for 13 cities, because the general rule provably cannot make the call. It is a supplement to
  the data-driven fade, not a replacement — the other 248 fades are still automatic.
- **A real-year second arm on the fade** (`FADE_GAP_REAL`) — fades Aleppo, Varanasi, Samarkand,
  Trabzon, Changsha, Messina and Old Kingdom Memphis. Gap length cannot distinguish *derelict*
  from *under-recorded* (§3.10a).
- **Deleting the fade entirely** — the medieval leaderboard collapses (§3.10).
- **Widening `OSC_SPAN` for antiquity** — deletes real history (§3.9).
- **Widening `DROP_MARKERS`** to catch more metro suffixes — changes which entries get spliced
  and deletes cities that have no plain entry (§3.1).
- **A slope test for the trim's backward walk** — populstat's fill is piecewise-linear between
  several anchors of the *same* rejected definition (§3.6).
- **A size gate for the graft principal** (§6.4).
- **Majority vote for definition oscillation** (§6.3).
- **Blending the seam** (§3.7).
