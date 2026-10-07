# languagedots: design record

A world dot map of first languages, one dot per 1,000 people, from censuses. Sister to
`religiondots`, sharing its geography and dot machinery but not its code tree. Started
2026-10-04 with India, Nepal and Pakistan. The coverage sweep that motivated it is in
`coverage/COVERAGE.md`.

Sections are numbered and the numbers are stable ids. Mark each DECIDED, PROPOSED or OPEN.

## 1. What a dot is (DECIDED)

One dot is 1,000 people whose census answer to the language question was that language. One
person, one dot, one language: the first-language answer where a census asks several (mother
tongue over home language over "languages spoken"). Which question a country asked is said in
its `how` line, because mother tongue, home language and ex-USSR "native language" are not the
same quantity.

## 2. Relationship to religiondots (DECIDED, Anita 2026-10-04)

A separate project, not a setting inside religiondots. Shared, read-only:

- **Placement layers**, `religiondots/data/geo/<cc>/`. For a country whose language table is on
  the same units as its religion table, the join religiondots checked is reused unchanged. India,
  Nepal and Pakistan all are.
- **Sea clip, Kontur density cap, geography checks** (`water.py`, `kontur_cap.py`,
  `sources/geo_checks.py`), loaded by path through `rdlink.py`, never by sys.path (religiondots'
  countries.py and taxonomy modules would shadow ours).
- **Nothing is written into religiondots.** The sea clip's cache is read from religiondots when
  current, else written under `languagedots/data/geo/_waterclip`.

Copied and cut down: `scatter.py`, `tiles.py`; `mvt.py` copied unchanged. The viewer is new:
religiondots' is 9,600 lines with the religion palette and roll-up woven through it, and a
smaller fresh page in the same style was cheaper than stripping it. The two maps can later link
to each other at the same view (a corner switch), which gives the "setting" feel without the
coupling.

## 3. Categories (DECIDED)

### 3.1 Draw what the source names

Every label a census prints as a mother tongue gets a node of its own, even a small one, even
where linguists would call it a dialect. The map does not merge Bhojpuri into Hindi because the
census of India's language table does; it reads the mother-tongue rows (`xxx001`-`xxx998`), not
the language rows (`xxx000`). It does merge spelling variants of one answer (Gujrao into Gujarati,
Khari Boli into Hindi) where the variant is one community's name for the same speech; each such
merge is commented in the mapping module.

### 3.2 Unnamed remainders sit on the narrowest node that contains them

"Others under BENGALI", Pakistan's OTHERS, Nepal's Others: people the census filed under a group
without naming the language. They are drawn on the narrowest tree node that contains everything
the census put in that group, never guessed into a member. Drawn desaturated (`color_own`), and
the legend calls them "<group>, language not named", so a remainder does not read as one more
language. The big one: India's "Others under HINDI", 16.7M, 14.9M of them in Bihar, almost
certainly Bajjika and Angika speakers, drawn on Indo-Aryan because the census does not say so.

### 3.3 A label whose meaning depends on place may be split by place

India's "Pahari" is Pahari-Pothwari (a Lahnda variety) in Jammu and Kashmir and Western Pahari
in Himachal. The census row is per area, so the split is the census's own geography, not an
estimate. `in2011.resolve(code, state)` takes the state for exactly this. Use sparingly and say
why in the mapping.

### 3.5 Indigenous-only censuses (DECIDED, Anita 2026-10-04: "for now that's fine; ideally we find other sources to corroborate")

Where a census asks only about indigenous languages (Mexico, Colombia, Argentina, Chile, Brazil,
Venezuela, Nicaragua, Costa Rica), the indigenous languages are drawn as measured and everyone
else on the country's main language node with `tier="derived"`, and `how` says so. The remainder
is the census's own complement, not a guess about who they are, but it does assume they speak the
main language. Flipping it means dropping the derived rows in each such `countries/<cc>.py`.
Where a second source speaks to the remainder (a survey asking home language, an older census
that asked everyone, a language-use table), the agent notes in `sources/<cc>.md` whether it
corroborates; finding one is wanted, not required.

### 3.6 Multi-answer tables: split each person's dot (DECIDED, Anita 2026-10-04: "lets split")

A table where a person may give several languages (Morocco's local languages, 117% in total;
Paraguay 2022; Ecuador 2022; New Zealand) is drawn by sharing each person across the languages
they named. Without microdata the exact per-person split is unknowable, so within each unit the
mentions are scaled to the unit's population: `count_l = mentions_l * population / sum(mentions)`
(or, for a table of shares, `share_l / sum(shares) * population`). Every row is `tier="derived"`;
`how` says "languages used, several allowed; each person shared across the languages they
named". Where microdata or a cross-table gives the combinations, the exact split (1/k of a
person to each of their k languages) replaces the scaling. A unit's people who named no
language stay in `gap`.

### 3.4 The tree

`taxonomy/tree.txt`: one line per node, the dotted id is the hierarchy. Countries added by agents
put their new nodes in `taxonomy/tree.d/<cc>.txt` fragments (merged by `taxonomy/build.py`,
identical repeats allowed), so parallel agents never edit one shared file. Roots are families plus
`signlanguage` and `other`. The middle levels are the conventional ones (Masica's zones for
Indo-Aryan, the usual Tibeto-Burman subgroups), checked against Glottolog
(`data/raw/glottolog/`, CC BY 4.0), because they are what a reader expects and what the colours
band by. Glottolog's deeper trees are a reference, not the display. No glottocodes are stored
yet; add them only verified, never from memory.

## 4. Dots (DECIDED, inherited)

religiondots §4 unchanged: fractions carried along a Hilbert curve through the units, a dot
dropped where the running total crosses 1,000; inside a unit, split across placement polygons by
their population with the same carry. A language that reaches no dot in a country is drawn as
ONE dot of its own true weight (300 speakers = 0.3 of a dot's area) at its largest
concentration. Not a ring symbol: Anita 2026-10-04, since mark area is proportional to people
(§4.1), a tiny dot already says "few". scatter.py still writes these as rings_<cc>.geojson with
a `count`; tiles.py folds them into the dot layers.

Merge for low zooms (`tiles.py`): nearby dots of one language become one mark of area
proportional to `k`. **CELL_BITS is 6 here (8px cells), not religiondots' 5**: at 16px the merged
marks read as a lattice over dense India and the colours could not be judged. Tiles go to z12.

### 4.1 A dot has a fixed size on the GROUND (DECIDED, Anita 2026-10-04)

Every 1,000 people get the same ink area on the ground (`DOT_KM2`, 1.5 km² at slider 1x), a
merged mark of k dots k times that, with NO cap; the radius in pixels doubles per zoom level like
the map. Replaces religiondots' screen-sized dots (√k radius, 1.26x per level, capped at half a
merge cell), which Anita caught doing two things wrong here:

- **the cap over-inked minorities.** In dense cells Hindi's marks hit the cap and stopped growing
  while small languages' marks kept their full size.
- **coverage snapped up at whole zoom levels**, because capped marks split into uncapped ones.

With ink fixed per person on the ground a mark splitting four ways keeps its total ink exactly,
so how full a place looks depends on how many people live there and not on the zoom. Her ask
was "the screen mostly full at all times; rescale only at the thresholds"; this goes one further
and needs no rescale at the thresholds at all. Past the last zoom at which marks still split, the
pixel radius freezes and the map turns into an ordinary dot map.

**The cap is per CELL, not per mark.** Uncapped, a city's ink spreads far past it (Delhi at
3 km² a dot is a disc ~140 km across). So each mark carries `t`, the people of every language in
its cell, and the cell's total disc is capped at CAP_FRAC (0.65) of the cell's width; every
language gets its share p/t of that disc's area. A full cell shrinks all its languages alike, so
shares stay honest and nothing spills; the cap is a fraction of the cell's ground size (via the
feature's tile zoom `z`), so it grows with the map and a split still keeps ink.

Measured over Bihar (coverage of the frame, either side of each threshold):
old sizing 26%→50% at z5, 28%→58% at z6, 30%→60% at z7; ground-fixed sizing (before the cell
cap) dips instead, 16%→12%, 29%→24%, 51%→46%: a speckle of gaps between the smaller marks. A
per-level sawtooth boost cannot remove the dip: MapLibre evaluates a zoom-and-data expression at
whole zooms only and blends, so the sawtooth came out as a uniform enlargement (tried).

### 4.2 Aggregation: how deep the marks come from (DECIDED, Anita 2026-10-04)

`tiles.py` writes the same dots merged four ways in every tile: `dotsm1` on a 32x32 grid,
`dots` on 64x64, `dots1` on 128x128, `dots2` on 256x256 (16, 8, 4 and 2 px cells; the name's
suffix is the offset from CELL_BITS). The viewer's Aggregation control (called Detail until
2026-10-04) picks the layer, right being coarser: low (2 px), medium (4 px, the default), high
(8 px), highest (16 px, added 2026-10-04 at Anita's ask). Each step is 4x fewer marks, each
covering 4x the ground. Because size depends only on people (§4.1), aggregation changes grain
and never fullness. (Tried first and dropped: declaring the source's tiles 256 or 128 px so
MapLibre fetches deeper tiles. MapLibre rejects any vector tileSize but 512.) Four layers roughly
quadruple the archive; trim before deploying if size matters. The coarsest layer is ~17% of all
marks (each of its zooms holds what `dots` holds one zoom lower).

**Selection on top (2026-10-04).** With a language or group selected, dots mode adds 1000 to
the `circle-sort-key` of marks under it (the colouring's own test), so they draw above the
rest, bigger on top within each set; within a tile only. Pies re-sort by the selection's people
in each cell, fewest first, and a cell with more than 8 languages keeps the selection's wedges
rather than folding them into the leftover one. Cost, measured headless on 2026-10-04: the sort
key adds nothing measurable, because the selection's data-driven colour already reloads every
loaded tile and the two share one reload; that reload is ~1.3 s (medium) and ~2.3 s (low) over
northern India at z6, ~0.35 s over the US east coast at z7, with panning smooth afterwards. A
separate filtered overlay layer would not avoid it (`setFilter` reloads too); drawing the dots
as a GPU custom layer like the pies would (a selection would only rewrite the palette). The pie
re-sort is 0.15-0.33 s over northern India and under 0.05 s over the US, with no reload: in pies
mode the hidden dots layer's colour is no longer updated, and `setFilter(null)` is only called
when the filter changes (MapLibre 4.7 treated it as a change every time and reloaded the source).

### 4.3 Pies (PROPOSED, Anita asked to try them 2026-10-04)

"Draw as: pies" draws one pie per merge cell on a canvas over the map, from the same tile
features the dots use: the cell's disc (same size and per-cell cap as §4.1), cut into its
languages in tree order so one family's wedges sit together. No border and the dots' opacity,
per her ask to keep it light. The cell a mark belongs to is recovered by flooring its position on
that zoom's cell grid (a mark is the people-weighted mean of its dots, so it lies inside its cell);
no tile change was needed. The dot layer stays in the style at opacity 0 while pies show, because
`querySourceFeatures` returns nothing for a layer with visibility none. Aggregation and size apply.
Hover shows the pie's top six languages.

**Drawn on the GPU since 2026-10-04** (Anita: "a bit laggy, slow to pan and zoom"). The first
version was a 2D canvas that projected every centre and filled every wedge each frame (~45k pies
at z5). Now a MapLibre custom layer draws each pie as one instanced quad; a fragment shader picks
the wedge from up to 8 cumulative fractions and a palette texture. Panning only changes the
matrix: measured headless, 1.3-1.6 ms a frame for pies against 1.3-1.4 for dots. The pie list
is rebuilt from the tiles once the map stops (~140 ms at z5, a third of it MapLibre's own
query; the rest is decoding each feature, which a worker could hide if the hitch shows). A cell
with more than 8 languages keeps its 7 largest and draws the rest as one wedge in the colour of
the largest of them. Needs WebGL2.

### 4.4 Cap slider (DECIDED, Anita 2026-10-04)

The per-cell cap of §4.1 is a third slider with Anita's stops: 0.5x, 1x (default), 1.5x, 2x,
3x, 5x and none, as multipliers on the cap disc's area (CAP_FRAC 0.65 of the cell width at 1x).
Higher lets dense places spread into their neighbours; "none" removes the cap, so a city's ink
spreads as far as its people's area takes it.

## 4.5 Legend fold (DECIDED, Anita 2026-10-04)

Under each group, two or more members under 50,000 people collapse into one "n small groups"
row, as religiondots §6.10 does. Simplified: a fixed people cut rather than religiondots' share
ladder fitted to the panel's height, and folded languages keep their own colours on the map.
Anything on the path to the selection never folds.

## 5. Colour (PROPOSED, 2026-10-04, for Anita's review)

`taxonomy/build.py`. One colour per node, the same in every country, hand-tunable.

- Each family owns a region of the wheel: Indo-Aryan the warm half, Dravidian greens and teals,
  Sino-Tibetan blues and violets, Austroasiatic magentas, Iranian sand and tan, Dardic cyan.
- Inside a family, the big languages are hand-picked (`HAND`) for contrast with their neighbours
  ON THE GROUND, not banded by subgroup, because on a language map the interesting edges are
  inside one family (Hindi | Bhojpuri | Maithili | Bengali). Small languages are generated near
  their group's colour (`GROUP`, `STEPS`).
- The viewer's "colour by branch" and "colour by family" modes recolour to the ancestor's colour,
  so the family-level picture is one click away rather than baked in.

## 6. Countries

| cc | source | question | units | notes |
|---|---|---|---|---|
| in | Census 2011 C-16 | mother tongue | 5,988 sub-districts | 270 mother tongues; Madhya Pradesh prints one district total on a sub-district line (warned, harmless) |
| np | NPHC 2021 Table 3 | mother tongue | 753 local levels | 125 mother tongues; institutional 239,098 not drawn (district only) |
| pk | Census 2023 Table 11 via PakPC2023 | mother tongue | 136 districts | 14 languages + OTHERS; tehsil (591) available, hexes are by district; AJK and GB not in Table 11 |

Each normaliser checks its own sums; Pakistan's district totals are checked against the same
census's religion table as read by religiondots, to the person.

## 6a. Parallel agents (DECIDED, Anita 2026-10-04)

religiondots' setup, slimmed: `AGENT_BRIEF.md` (the standing brief), `/ld` and `/ld-super`
(`maps/.claude/commands/`), `tools/claim.py` over `queue.csv` (built from the coverage sweep by
`tools/make_queue.py`: tiers A and B `free`, C D E `ruling`), `tools/ask.py` (`ask/OPEN.md`),
`tools/build_tail.py` (one lock around the language tree and the tile build),
`tools/check_country.py`, `sources/_grid.py` (a Kontur placement layer for new units), and
`runlog.md`. One file per country (`countries/<cc>.py`, found by the loader, no shared list) and
one tree fragment per country, because religiondots' shared ORDER list and tree were where
parallel edits collided. New Kontur cap blocks go in `languagedots/kontur_cap.csv`, merged with
religiondots' registry at load (`rdlink.py`).

## 7. Not done yet

- Tehsil level for Pakistan (needs a tehsil key on the hexes).
- Per-country `note_public` text.
- The religiondots link switch; regions mode; mobile pass on the viewer.
- Deploy (religiondots' route: data to R2, page to the website repo).
- Every other country: start from `coverage/coverage.csv`, tier A first.
