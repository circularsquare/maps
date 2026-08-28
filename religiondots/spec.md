# religiondots — how it works, and why

Interactive world map of religious affiliation as dots, at the finest branch/sect granularity
each region's data supports, with a religion genealogy panel that doubles as the legend.

**This file is Claude-managed.** `todo.txt` is Anita's and is not mine to edit. `sources.md` is
the per-source inventory and is also mine.

**Status: nothing is built.** This is a design document, not a record of a pipeline. Everything
below is either (a) a decision taken with the reasoning attached, (b) a proposal that a prototype
has to settle, or (c) an open question. Each section says which. When code exists this file
converts, section by section, into a record of what it does and what was tried and failed —
the shape `cityhistory/spec.md` has.

---

## 1. What the map has to do

Four requirements, from the brief, restated so they can be checked:

| | requirement | test |
|---|---|---|
| R1 | **Composition is readable at a glance.** In any region, how many groups are there and which. | Point at Lebanon at world zoom and read three colours, not noise. |
| R2 | **Granularity down to branches and sects**, not seven world religions. | Beta Israel, Old Believers, Alevis, Mar Thoma, Tenrikyō each have their own colour, wherever a source supports it. |
| R3 | **Very small groups are visible and not misrepresented.** A 20-person monastic community should be findable; it must not read as a town. | A Carthusian charterhouse can be found on the map; nothing on screen implies it has more people than it has. |
| R4 | **Double- and undercounting are bounded and declared**, not silently averaged away. | Every drawn dot traces to one source figure on one stated basis; every modelled region is marked as modelled. |

R3 and R4 are the two that make this hard, and they are the two that most religion maps get
wrong. §5 and §4 are the answers.

R1 and R2 pull against each other — 200 distinguishable colours do not exist. §6 is the answer
(hue = family, shade = branch, and the genealogy tree is the key).

## 2. The taxonomy is the spine — DECIDED

One file, `taxonomy/religions.json`, is the single source of truth for:

1. **the tree** — every group's parent, so counts nest;
2. **stable ids** — `christianity.catholic.latin.jesuit`, dotted path, never renumbered;
3. **colour** — derived from position in the tree, not stored per group (§6);
4. **the genealogy graph** — the side project, and the legend, and the selection model.

Everything else in the repo — every source adapter, the reconciler, the viewer, the genealogy
panel — refers to groups only by these ids. A source that reports a category we have no node for
does not get a new node invented at ingest time; it goes to `unmapped.csv` and waits for a
decision. Silent node creation is how a taxonomy turns into 900 near-duplicate leaves.

### 2.1 Two relations, not one

The tree and the genealogy are **different graphs over the same nodes**, and conflating them is
the first mistake available:

- **`parent`** — the *containment* relation, a strict tree. Used for counting and for colour.
  "Every Jesuit is a Latin Catholic" is a statement about people alive now.
- **`from`** — the *descent* relation, a DAG with dates. Used only by the genealogy panel.
  "The Old Believers separated from the Russian Orthodox Church in 1666" is a statement about
  history, and it is many-to-many: Sikhism draws on both Hindu and Islamic currents, the
  Reformed churches have several parents, Mandaeism's parentage is disputed.

The tree must stay a tree because §3's arithmetic depends on it. The genealogy must be a DAG
because history is one. Edges carry `{from, to, year, kind}` where kind ∈ {schism, reform,
revival, syncretism, revival-of-extinct, disputed}, and `disputed` is a real value that gets
drawn differently — a dashed edge — rather than a claim we quietly pick a side on.

### 2.2 Where the tree comes from

Seeded by hand from a small number of reference works, not scraped. Wikidata's `subclass of`
(P279) over religions is available and is *not* usable as the tree: it mixes containment with
influence, has cycles in practice, and its depth is wildly uneven. It is worth harvesting as a
**candidate list** for nodes we are missing and for the genealogy edges (`P144 based on`,
Q126287984 `religious schism`), reviewed one at a time. That is a `tools/` scan, not a build
stage.

Depth is uneven by design and that is correct — confirmed 2026-08-27. Christianity in the United
States can run five or six levels deep because ASARB enumerates 372 bodies by county; Chinese folk
religion is one node because nothing enumerates it. The tree records what is *countable*, not what
exists, and it will look lopsided because the world's statistical agencies are.

### 2.3 Source categories are not all the same kind of thing

Found on opening the first source, and it will recur. The US Religion Census's 374 codes mix at
least four kinds of category:

| kind | example | maps to |
|---|---|---|
| a denomination | *Greek Orthodox Archdiocese of America* | a leaf, cleanly |
| a whole tradition | *Mahayana Buddhist*, *Theravada Buddhist* | an internal node, and it is the only depth available there |
| a building type standing in for a tradition | *Hindu Temples* | an internal node, with a note that the unit is temples |
| **a practice, which is not an affiliation at all** | *Hindu Yoga and Meditation* — 437k, 396 counties | nothing. Held out. |

The last row is the one to be careful about. It is not a religious body, its people are mostly
counted elsewhere or are not adherents of anything, and folding it into `hinduism` would both
double count against *Hindu Temples* and assert something false about 437,000 people. It goes to
`unmapped.csv` under §2's no-silent-node rule, and the decision is Anita's to make explicitly.

**And the mapping cannot be automated on names.** "Orthodox" appears in this one file across
Eastern Orthodoxy, *Orthodox Judaism*, *Orthodox Presbyterian Church*, *Orthodox Anglican Church*,
*Orthodox Mennonite Church* and *Orthodox Old Roman Catholic Communion* — six unrelated families.
The mapping is by group code, by hand, once per source.

### 2.4 The first tree — built 2026-08-27 from one source

`taxonomy/` now holds the working tree. Three hand files and a validator, which is the shape the
rest of the repo uses:

| file | what it is |
|---|---|
| `branches.py` | **68 internal nodes**, source-independent, each with a label and the reasoning where it needs one |
| `usrc2020.py` | the 372 ASARB codes → leaf ids, plus `REVIEW` (24 arguable calls, each with its reason) and `UNMAPPED` (1) |
| `build_tree.py` | five checks, then emits `religions.json` and fills `path` in `usrc_groups.csv` |
| `religions.json` | generated — **428 nodes, 68 branches, 360 leaves, depth 4** |

**The arithmetic reconciles exactly**, which is the check worth having: adherents rolled up the
tree total **160,786,973** against ASARB's national **161,224,088**, and the difference is
**437,115 — precisely the one category held off the tree** (*Hindu Yoga and Meditation*). Nothing
is lost or double counted in the mapping.

Depth 1: Christianity 152.0M · Islam 4.45M · Judaism 2.07M · Buddhism 1.04M · Hinduism 831k ·
Unitarian Universalist 202k · Bahá'í 179k, then eleven families with no adherent figure at all.

**The three checks that earn their place** in `build_tree.py`, because a taxonomy fails silently:

1. **a leaf whose parent branch does not exist is an error**, which is §2's no-silent-node rule
   made mechanical — the only way to add a branch is to add it to `branches.py` deliberately;
2. **duplicate group codes are read out of the file text, not the dict** — a Python dict literal
   silently keeps the last of a repeated key, so a mapping could be quietly overwritten and the
   totals would still look fine;
3. **every code in the source data must be mapped or explicitly unmapped**, so a new source
   release with new bodies fails loudly instead of dropping them.

**Where the depth actually is.** 50 Mennonite leaves, 28 trinitarian Pentecostal, 22 Presbyterian,
19 Lutheran, 15 Schwarzenau Brethren, 13 canonical Orthodox jurisdictions. The Anabaptist branch
is **87 bodies totalling 846,198 people** — an average of under 10,000 each. That branch alone is
the R2 and R3 case: it is the finest religious granularity available anywhere in the world, and
almost all of it is below or near the dot floor.

**And the non-Christian side is nearly all rings.** Eleven of the seventeen top-level families —
Sikhism, Jainism, Zoroastrianism, Shinto, Daoism, New Thought, Spiritualism, Unification, Hebrew
Israelite, secular/ethical — are represented by **exactly one body each with no adherent count**.
So on the best religion dataset in the world, most of the world's religions are a single
unquantified row. That is the shape of the problem this map is trying to show.

**The tree grows where a source reaches, and nowhere else.** ASARB has one Shinto row, so
`shinto` is a leaf today; Japan's 宗教統計調査 enumerates shrine associations by sect and will
push depth under it. That is the same rule as §2.2, seen from the other end — a family is shallow
because *this* source is shallow there, not because the family is simple. Expect the tree to
deepen unevenly, one source at a time, and expect the US to stay the deepest branch of
Christianity for a long time.

**Cross-source denomination matching is deferred, deliberately.** Anita's call, 2026-08-27: the
meticulous work of deciding that a body in the US file and a body in the Canadian file are the
same body is a later task. The consequence to hold to in the meantime is that **the first source
into a branch defines its shape**, and later sources map into it and leave their genuinely hard
cases in `REVIEW` rather than forcing a merge. `sources` on each node is a dict keyed by source id
precisely so that a node can accumulate several without either being lost.

## 3. Counting rules — DECIDED

These are the R4 answers. All four exist because of a specific failure mode.

### 3.1 Every figure carries a basis, and bases are never mixed

`basis` ∈:

| basis | what it means | who reports it |
|---|---|---|
| `self_id` | a person said this about themselves | censuses, general population surveys |
| `roll` | an institution counted its members | church statistics, Japan's 宗教統計調査, Annuario Pontificio |
| `estimate` | a compiler's judgement | Pew, WRD/WCD, ARDA |
| `attendance` | people present at services | some denominational reporting |

A region's composition is built on **exactly one basis**. Figures on other bases may be used to
*split* a category (§3.4), never to add to it.

**`basis` is a property of the row, not of the source** — found 2026-08-27 in the first dataset
opened. The US Religion Census is a `roll` dataset throughout except that group code 267 is
literally named **"Muslim Estimate"**, 4.45M adherents, and codes 890/891/892 ("Mahayana
Buddhist", "Theravada Buddhist", "Vajarayana Buddhist") and 895 ("Hindu Temples") are compiler
estimates for traditions that do not keep membership rolls in the first place. So the compilers
did the honest thing and labelled them, and a per-source basis field would have thrown that away.
Every ingest maps basis per row.

The reason is not fussiness. Japan's Agency for Cultural Affairs collects adherent counts from
religious corporations and the national total comes to roughly 180 million people against a
population of 125 million, because Shintō shrine parishes count residents of the parish and
Buddhist temples count households, and the same person is in both. Those numbers are useful and
they are not a partition of the population. Adding a `roll` figure to a `self_id` figure produces
a number that is not about anything.

### 3.2 The tree is a partition, so nesting cannot double count

For every unit, over every level of the tree, the children of a node sum to that node.

**Residuals are computed everywhere, at every level, and drawn** — this is a build step in
`reconcile.py`, not something done where it happens to be convenient. Wherever a parent total is
known and its children are enumerated, `residual = parent − Σ children` becomes a real node,
`…other/unspecified`, with its own colour and its own dots. A large "Protestant, denomination not
reported" slice in a country whose census only asked the top level is a true fact about the data,
and dropping it would make the detailed slices look far more complete than they are — the map
would show a country's Baptists and silently omit the 90% of its Protestants nobody enumerated.

This applies at every level and in both directions: Christian minus the named branches, Protestant
minus the named denominations, Buddhist minus the named schools, and the national total minus every
religion, which is the unaffiliated-plus-not-stated residual.

**A negative residual is a finding, never a fudge.** If a node's children exceed their parent, one
of three things has happened, and all three are real defects worth chasing rather than clamping to
zero:

1. **mixed bases** (§3.1) — a `roll` denominational figure has been set against a `self_id` parent
   total, and rolls run high. This is the common case and Japan is the extreme;
2. **an overlap** the tree does not model — the case §3.3 opens a syncretic node for;
3. **a mis-mapped source category**, sitting under the wrong parent.

So the residual is not only an output, it is **the main automatic detector of §3.1 and §3.2
violations**, and it is cheap because it comes out of arithmetic the pipeline is doing anyway.
Negative residuals go to a findings list with the unit, the node, the size of the overshoot and
the contributing sources. Expect the list to be long; cityhistory's experience is that the queue
never empties and the useful stopping rule is visibility — work the ones big enough to see.

**A monastic order is a slice of its parent, not an extra.** A Jesuit is one person, counted once,
in `…catholic.latin.jesuit` and therefore *not* in `…catholic.latin.other`. This is the one place
the containment tree feels wrong — membership of an order is a vocation, not an affiliation, and a
Jesuit is obviously also a Catholic — and it is still right, because the alternative is a layer
whose members are also counted elsewhere, which is R4's failure by construction. The tree answers
"where is this person counted", and each person is counted once.

Confirmed 2026-08-27, and so is the rendering: **orders are drawn on top.** Being a slice in the
arithmetic does not force being a peer on screen. An order's mark sits above its parent's dots
rather than displacing one of them — which is also the natural z-order for rings generally (§4.3),
since a ring buried under a dot layer is a ring nobody finds.

There is a small tension worth naming: drawn on top, an order looks additive even though it was
subtracted. At order scale it is invisible — a 20-monk community against a county of Catholics is
far below one dot — so the two rules do not actually collide anywhere on screen. If some large body
ever sits in an order-like slot the question comes back.

### 3.3 Syncretism gets a node, not a split

Where dual practice is the norm — Japan, China, Vietnam, Korea, much of West Africa and the
Andes — forcing a partition manufactures a precision nobody has. The rule: **a combination is a
node.** `japan.shinbutsu`, `china.folk` (which includes Buddhist and Daoist practice by
construction, and says so in its description), `vodun.catholic-syncretic`. The partition survives,
and the node name is where the overlap is declared rather than hidden.

The test for opening one: a source reports the combination, or reports categories that sum past
100%. Not "we suspect overlap".

### 3.4 Structure from the detailed source, totals from the recent one

The common shape: a recent source has the right total and coarse categories; an older source has
the fine split. Brazil is the case that shows it — the 2022 census gives religion by municipality
and the evangelical share by municipality, but IBGE has **not** published the denominational
breakdown (quality problems in the collected detail, still under evaluation), while the 2010
census did publish it. India is the same shape a decade wider.

So: take the 2022 municipal evangelical *total*, split it by the 2010 municipal denominational
*shares*, and record `structure_year: 2010, total_year: 2022` on every resulting figure. The
viewer shows both years. This is an interpolation and is labelled as one; what it must never do
is silently present 2010 shares as 2022 data.

**Where there is no recent total to rescale to, the old figure is used as it stands** — decided
2026-08-27, against the brief's 2015-or-later preference, because the alternative is worse. India's
last religion census is 2011 and the 2021 census has not happened; Russia's best subnational source
is the Sreda Arena atlas of 2012 and has no successor. Leaving them out means a blank India, which
is 18% of humanity, and a blank Russia. They go in at their own year, the year is on every figure
and visible in the unit panel, and the confidence tier (§7) is not reduced for age alone — an
old census is a measurement, unlike a modelled estimate, and the two should not be rendered as
though they were the same kind of claim.

### 3.5 Undercounting is marked, not filled

Countries that do not ask: China, Nigeria, France, the United States (federally — the US is the
best-served country on the map anyway, see `sources.md`), most of the Gulf. For these the
composition is a survey- or compiler-derived estimate and the dots are drawn **desaturated**
(§7). cityhistory dims years with no measurement within ±5; this is the same instrument on a
different axis, and for the same reason: the absence of data is itself something the map should
show rather than paper over.

### 3.6 A roll counts the institution's location, not the member's — FOUND 2026-08-27

The first real check on the first real dataset, and it is a defect class nobody would have
predicted from the methodology page. The US Religion Census attributes adherents to the county of
the **congregation**, and people do not always worship in the county they live in. So:

**30 of 3,144 counties report more adherents than residents.** King County, Texas: population 265,
adherents 1,199 — **452%**. Stonewall County TX 167%, Harmon County OK 164%, Harding County NM
156%, Fredericksburg city VA 133%. The pattern is rural counties with one substantial church
drawing from a wide area, plus Virginia's independent cities, which are tiny polygons surrounded
by the county whose residents fill their churches.

This is not a mixed-basis error (§3.1) and not a mis-mapped category (§3.2) — it is a third thing,
and it means §3.2's negative-residual list has a fourth entry:

4. **the roll is attributed to the institution's location rather than the member's**, so a unit's
   figure is a catchment total, not a resident total.

**Handling — decided 2026-08-27: clamp to zero for display, keep computing it, flag only the big
ones.** A negative residual cannot be drawn in any case, so display clamps at zero always. What the
decision settles is when it is worth *reporting*, and the measurement says: almost never.

| | |
|---|---|
| counties with a negative residual | **30 of 3,144** |
| total overshoot nationally | **42,892 people = 0.0065% of US population** |
| largest single county | Fredericksburg city VA, 9,189 people (133%) |
| largest by ratio | King County TX, 934 people (452%) |
| under 1,000 people | 21 of the 30 |

At that size it is noise. The rule: **surface a negative residual as a finding when it exceeds both
5% of the unit's population and 1,000 people** — which leaves about nine US counties, a list
someone could actually work — and otherwise log it silently and move on. Redistributing over a
commuting shed would invent a model this project has no evidence for, and capping *inputs* at
population would hide a real property of the source; clamping the output does neither.

The 0.0065% figure is also the yardstick for the next source. A source whose overshoot runs at a
few hundredths of a percent is behaving like ASARB; one running at whole percentage points has a
different problem and the threshold should not quietly absorb it. Worth re-checking whenever a
finer geography than county is used, since the error grows as units shrink.

The general form, which will reach every `roll` source: **a roll is a fact about buildings, and a
dot map is a claim about residents.** For most units the two coincide closely enough. Where they
do not, the dots are placed by population grid inside the unit anyway (§8), so the map is already
saying "somewhere in this unit" rather than "at this church" — the failure is confined to the
unit's total, which is exactly where the residual can see it.

## 4. The size problem — §4.1 and §4.3 DECIDED, §4.2 open

Christianity is ~2.4 billion. A Carthusian charterhouse is ~20 monks. Eight orders of magnitude,
and R3 says both must be on the same map without the small one lying about its size.

### 4.1 What is decided: dot count stays linear in people

No log scale, no sqrt, no per-group rescaling. Within any single view, one dot is one fixed
number of people for **every** group, so two groups' dot counts are exactly their population
ratio. This is the property the whole map is for and nothing below is allowed to break it.

### 4.2 Zooming out merges dots, it does not drop them — DECIDED 2026-08-27

**Two different things, and the first draft of this section confused them.** Say them separately:

- **dot value** — how many people one dot stands for. A data quantity.
- **dot size** — the mark's radius in pixels. A rendering quantity.

**Dot size** grows sublinearly with zoom, roughly like a square root of the scale factor, so dots
stay visible when zoomed out and do not swell into blobs when zoomed in. Same idiom as elsewhere in
this repo; in MapLibre it is an exponential `circle-radius` interpolation with a base under 2.

**Dot value** cannot stay fixed across all zooms — 8.1 billion people at any legible dot value is
far more marks than a world view can draw. It changes with zoom, and **the way it changes is by
merging, not by dropping.** At each zoom there is a cell size; within a cell, all of a group's
atomic dots collapse into **one mark whose area is proportional to how many merged**. Zoom in, the
cells subdivide, marks split into smaller marks, and at the finest zoom every mark is a single
atomic dot of N people.

So a cell containing 40,000 Catholics and 3,000 Alevis draws two marks, and the Catholic one has
13× the area. Count the colours in a cell and you have R1; compare their areas and you have the
composition. This replaces the packed-blob glyph of §5, which is dropped.

**Why merging beats the subsample this section used to propose.** Subsampling by rank drops dots
at random when you zoom out, so a small group *stochastically vanishes* — present at one zoom,
gone at the next, back again on a pan. Merging keeps every person represented at every zoom; a
small group's mark just gets small. Nothing disappears for a reason the reader cannot see. It also
keeps §4.1 exactly: area is strictly proportional to count, no log, no per-group rescaling.

**What merging does not fix** is a group whose merged mark is under a pixel. That is still §4.3's
job, and the boundary is now clean: marks handle everything down to sub-pixel, rings take over
below it.

**Storage.** Atoms at 1 dot = 1,000 people is 8.1M features, about what ancestrydots reaches
across 50 states, and it goes to R2 the same way. 1 dot = 100 would be 81M and is not happening.
Aggregated zooms are much cheaper than the atomic one, so the finest level dominates the total —
and the merge is a natural fit for the tile pyramid, since each zoom level stores its own
aggregate rather than the viewer filtering a single flat set. (ancestrydots uses
`--drop-rate 1.0 --drop-densest-as-needed`, i.e. no per-zoom culling at all. That works for 3.3M
dots and will not work here.)

**Open:** whether the aggregate marks want a cap on radius in the densest cells, and if so what
happens visually at the cap. Straight area-proportionality across Tokyo and rural Mongolia in one
view is a big dynamic range. cityhistory answered the same question with sublinear bubbles, but
that trade is not available here — there, bubble area is the only encoding of a city's size, while
here the mark stands for a countable number of atomic dots, and inflating it would break the
"count the dots" property that §4.1 exists to protect.

### 4.3 Presence marks: a second grammar that carries no magnitude — DECIDED 2026-08-27

Every group present in a unit gets **one** mark — a small hollow ring in its colour — placed at
its centre of mass, or at the actual site where the thing is site-bound (a monastery, a temple, a
single surviving congregation). One mark, regardless of size, by construction.

Because the mark is size-independent it **cannot** misstate magnitude; the reader learns two
symbols, "filled dot = N people" and "ring = present here". A charterhouse of 20 monks is a ring.
So is a group of 40 million — but that one is sitting on 40,000 of its own dots, so rings are
suppressed above a size threshold to keep the grammar clean.

This is the honest form of the thing people usually do with a log scale. A log-scaled dot says
"this is small but not that small" and the reader cannot recover the number. A ring says "present"
and says nothing else, which is exactly the claim we can support.

**How many bodies actually need rings** — measured 2026-08-27 on the US Religion Census, which is
the best-resourced source on the map. 372 bodies, of which:

| | bodies |
|---|---|
| report adherents, **under 1,000 nationally** — certainly below a 1-per-1,000 dot floor | **50** |
| report adherents, over 10M | 4 |
| **report congregations but no adherent count — size unknown** | **155** |

The smallest body with a count is the **Reformed Congregations of North America: 26 adherents, 1
congregation, 1 county**, and the Anabaptist tail behind it is dozens of Old Order Mennonite,
Amish and Hutterite groups in the hundreds. That tail is the R3 test case and it is real.

**But "no adherent count" is not the same as "small", and assuming it was is a mistake worth
recording.** The 155 congregations-only bodies hold **27,005 congregations — 7.6% of every
congregation in the country** — and at the reporting bodies' average of 489 adherents per
congregation that is on the order of **13 million people**. Four of them have over 1,000
congregations each:

| body | congregations | counties |
|---|---|---|
| United Pentecostal Church International | 4,549 | 1,692 |
| Church of God of Prophecy | 1,614 | 790 |
| Evangelical Free Church of America | 1,602 | 729 |
| Baptist Missionary Association of America | 1,144 | 326 |

UPCI is a top-15 denomination by congregation count with no adherent figure at all. Drawing it as
one ring per county would be as wrong as drawing the Carthusians as dots.

**So §4.4's congregation-to-adherent conversion is required, not an optional extra**, and it needs
a defensible per-family ratio rather than one national average — the 489 figure mixes Catholic
parishes of thousands with Old Order meetings of forty. Sizing it by comparable bodies within the
same branch is the obvious approach and the ratio's spread within a branch is the thing to check
before trusting it.

The split that matters is therefore three-way, not two-way: **counted** → dots; **uncounted but
demonstrably substantial** → converted dots, desaturated (§7); **uncounted and small**, the 24
congregations-only bodies with under 10 congregations, plus the 50 tiny counted ones → rings.

**Rejected, with reasons:**

| approach | why not |
|---|---|
| log-scaled dot size | destroys R1's at-a-glance quantity reading for *everything*, to fix the tail. The whole map pays for the smallest 0.001%. |
| minimum dot size / "at least one dot per group" | silently inflates: a 20-monk order and a 900-person village group both draw one dot at 1:1000, i.e. both read as 1,000 people. This is the dishonest version of §4.3 and the difference is only that the ring *looks* different from a dot. |
| per-group dot values | breaks §4.1. Two dots on screen would mean different numbers of people depending on colour, which is unreadable and unfixable by a legend. |
| separate "small groups" layer at 1:10 | same as minimum dot size, one step more elaborate. Rejected unless the prototype shows rings are not findable. |

**Open:** whether rings are on by default. Proposal: off at z<6, on above, plus a "small groups"
toggle that forces them at any zoom, plus search — selecting Carthusians in the genealogy panel
should fly to and highlight their rings regardless of the toggle.

### 4.4 Sources that answer the question backwards — DECIDED to use, they feed rings

Nearly every source in §1–§3 of `sources.md` answers *given this place, which religions*. A second
kind answers *given this religion, where is it*: lists of monasteries, congregations, dioceses,
temples and their coordinates. They are worth pulling in because they are strongest exactly where
the first kind is weakest — small groups, which no census has a row for, and countries that do not
ask at all.

They need no special case in the grammar. **A location-by-religion source produces rings.** It
knows a group is present at a point and says nothing about how many people are in it, which is
precisely what a ring means (§4.3). The two source shapes and the two symbols line up one-to-one,
which is a good sign that both distinctions are real.

Where a body's congregation count can be converted to adherents there is a route to dots — the US
Religion Census does this for the 155 of its 372 bodies that reported congregations only — but the
result is `estimate` basis, `derived` tier, and desaturated (§7). Rings first; dots only where the
conversion is defensible and declared.

**The trap, and the constraint that contains it.** These sources are dense where mapping effort
has been spent, not where religion is: OSM has far better coverage of German churches than of
Indonesian mosques, and Wikidata's coverage follows Wikipedia's. So they may set **presence and
never density**. 400 mapped churches in one county and 4 in the next is a fact about OSM. Hence:
**one ring per group per unit, never a ring per building.** That single rule is what keeps the
collection bias out of the picture — it throws away the count, which is the biased part, and keeps
the presence, which mostly is not.

## 5. Reading a region at a glance — the merge does it; no glyphs

R1's problem: 200 colours scattered at random over a country is salt-and-pepper noise. You can see
a place is mixed; you cannot see *how* mixed or of what.

**Everything stays dots.** §4.2's merge is the whole answer — zoomed out, each cell shows one mark
per group present, sized by count, so the number of colours in a cell *is* the number of groups
and their areas *are* the shares. No second visual language.

**Rejected 2026-08-27: the packed-blob cell.** The earlier proposal sorted a cell's dots into a
tiny waffle chart with contiguous colour wedges. It answers the same question and costs a second
rendering mode, a hard switch at a zoom threshold, and a claim it cannot support — clumping dots by
colour inside a cell reads as neighbourhood-scale religious segregation, which we do not know and
which in many cities is false. Sized marks say the same thing without asserting anything about
where inside the cell anybody lives.

**Still worth having, both cheap:**

- a diversity toggle colouring units by effective number of groups (`exp(H)`, Shannon), answering
  "how many" numerically where the marks answer it visually;
- a hover/click panel listing the unit's composition, with count, source and year per line — which
  is also where R4 becomes checkable by a reader rather than a promise in a spec.

## 6. Two colourings, not one — DECIDED 2026-08-27

The complaint about ancestrydots: there was no way to select Western European and see the
differences *within* it. That is not a missing feature, it is a consequence of having one palette.
With everything drawn at once a family has to read as a family, so its members take shades of one
hue and become mutually indistinguishable — right for the overview, useless for "what are the
Christianities".

So colour is a function of **(node, what is selected)**, and there are two modes.

**Overview palette** — nothing selected. Hue by top-level family, lightness and saturation by depth
and sibling index. Stable, hand-tunable, memorable: orange is always Islam. This is the palette
people learn, and the one that answers R1.

**Focus palette** — a subtree is selected. Its children spread across the **full hue wheel**, deeper
descendants take shades within their child's hue, and everything outside the selection drops to
near-black. Select Christianity and Catholic / Orthodox / Protestant / Oriental / Church of the
East / Restorationist are six well-separated hues rather than six blues.

The implementation is one function rather than two tables: `colour(node, scope)`, where the
overview is just `scope = root` and focus mode is the same algorithm re-rooted on the selection.
Five consequences worth writing down:

- **The wheel is divided among exactly the categories currently drawn**, not among all leaves.
  There is a display-depth control as well as a scope. Select Christianity at branch depth and get
  six hues; select it at full depth and get eighty, which will *not* all be distinguishable — that
  is the honest limit of the medium and it is the reason the depth control exists rather than
  something the palette should paper over.
- **Multi-selection** splits the wheel between the selected subtrees in proportion to how many
  drawn categories each holds. Christianity + Islam gets roughly half each. This falls out of the
  same function and needs no separate rule.
- **The tree recolours with the map, at the same instant.** Two keys that disagree is worse than
  either alone.
- **No transition on the swap.** House preference, and a hard swap is clearer than a hue rotation
  across several million dots in any case.
- **Hand overrides are expected, and bounded to a few scopes.** ancestrydots'
  `ancestry_colors.csv` is hand-maintained and that is not changing here
  ([[feedback_colors_csv_manual]]). But a table keyed by `(scope, node)` is unbounded if any node
  can be a scope, so overrides exist for the root scope and for the dozen or so scopes people
  actually select — christianity, islam, buddhism, hinduism, judaism, and so on. Everything else
  takes the generated palette.

**The accepted cost:** a colour is not stable across modes, so what you learned in the overview is
not what you see in focus. The mitigation is that the legend *is* the tree and recolours at the
same moment, so the key on screen is never stale. The alternative — keeping focus colours near the
family's own hue so they stay recognisable — is exactly the constraint being removed, so it is not
available.

**This is cheap, which is why it is affordable at all.** Dot features carry only a node id; the
viewer builds a MapLibre `match` expression for `circle-color`. Changing palette is a paint
update. No second dot set, no re-tiling, nothing precomputed per scope.

**Still true from before:** sibling groups that are large and adjacent (Sunni/Shia,
Catholic/Protestant) need separation that survives a 2px dot, while distant tiny leaves can share a
shade because they never appear in the same view. And **the genealogy tree is the colour key** —
there is no other legend that holds 200 entries legibly, and showing descent and colour in one
object is what makes either learnable.

## 7. Confidence is drawn — PROPOSED

Three tiers, from the source inventory, per unit per group:

| tier | example | rendering |
|---|---|---|
| measured | a census or register question, asked in this unit, any year (§3.4) | full saturation |
| derived | §3.4 structure-from-older-source; a national figure distributed by a proxy | desaturated |
| modelled | no subnational data; country estimate spread by population | desaturated and, proposed, a visible texture or stipple |

The unit panel names the source and year for every line, so "why is Sichuan pale" has an answer
one click away. §3.5 is the reason this is not optional: the map's honest claim in China is
different in kind from its claim in Utah, and a map that renders them identically is making the
weaker claim silently.

## 8. Pipeline — PROPOSED shape

```
taxonomy/religions.json      the tree + genealogy + colour rules        (hand)
sources/<id>.py              one fetch+normalise per source, in one of two shapes:
   counts   (§1-3)           → data/normalized/<id>.csv
   columns: geo_id, geo_level, node_id, count, basis, year, source_id, note
   sites    (§4.4)           → data/sites/<id>.csv
   columns: lon, lat, node_id, kind, name, year, source_id, note
data/geo/                    geoBoundaries ADM1/ADM2 + population grid
reconcile.py                 per unit: pick basis, resolve to a partition summing to population,
                             apply §3.4 splits, tag confidence  → data/units.json
                             sites collapse to one ring per (node, unit)  → data/rings.json
scatter.py                   units × population grid → atomic dots (lon, lat, node)
                             then one merged set per zoom (lon, lat, node, n)  §4.2
                                                              → data/dots.geojson
                             rings carry the site coordinate where one is known, the unit's
                             centre of mass where it is not
tippecanoe                   → data/dots.pmtiles → R2                  (WSL, as ancestrydots)
index.html                   MapLibre + PMTiles viewer + genealogy panel
tools/                       one-off scans; never build stages
```

`reconcile.py` is where R4 lives and it is the file to be most careful with. It should refuse
rather than guess: a unit whose figures do not sum to population within tolerance, or that mixes
bases, goes to a findings list, not into a fudge.

**Geography.** geoBoundaries (CC BY) rather than GADM, whose licence is non-commercial and
awkward. ADM1 everywhere, ADM2 where a source supports it — the level varies by country and that
is fine, because dots are placed by population weight, not by unit area.

### 8.1 Boundaries must be the vintage the data was collected on — FOUND 2026-08-27

Not the newest available. Taking the newest is the obvious default and it silently deletes places.

**Connecticut abolished its counties** for statistical purposes in 2022, replacing them with nine
Councils of Governments planning regions with new FIPS codes (09110–09190). ASARB 2020 reports
eight old counties (09001–09015). Joined against the 2024 cartographic boundaries, **every one of
them fails to match**: the whole state — 3,605,944 people and 1,707,793 adherents — has no polygon
and drops out with no error anywhere. Against the 2020 boundaries, all 3,143 ASARB counties match
exactly.

The failure mode is what makes this worth a rule rather than a fix. A join that silently drops
Connecticut looks identical to a working join everywhere else; nothing is malformed, no count is
wrong, a state is just missing. **So the join is checked in both directions and both sides are
reported**, always — unmatched data rows *and* unmatched polygons.

The check also confirms what should be absent: 91 polygons have no ASARB row and all 91 are
Puerto Rico, American Samoa, Guam, the Northern Marianas and the US Virgin Islands, which ASARB
does not cover. An expected absence and an accidental one look the same until you name the
expected ones.

This will recur everywhere and worse: municipal mergers in Japan, Brazil and Indonesia run
continuously, and a census's own geography is the only safe join target for that census.

### 8.2 Placement: the population weight is per-country, and can be the census's own

Kontur/GHS-POP is the **global fallback**, not the first move. Where a country publishes
population on a finer unit than its religion data, that finer unit *is* the population grid, and
it is measured rather than modelled.

For the US: religion is by county, and **2020 census tracts** carry the weight inside each one.
ancestrydots' uniform-random-in-polygon works there because census tracts are small and roughly
population-equal by construction; counties are neither — San Bernardino is 20,000 square miles of
mostly desert, and scattering its adherents uniformly would put dots across the Mojave.

Tract populations come from the **2020 decennial PL 94-171** file, not ACS, so that the weights
sum to the same 2020 census count ASARB measured its own county population against.

**Placement.** Dots are placed by **population grid**, not uniformly in the polygon.
ancestrydots' `random_points_in_polygon` is fine for US census tracts, which are small and roughly
population-equal by construction; at ADM1 scale globally it would fill the Sahara, the Amazon and
Siberia with people. Kontur Population (400m H3, HDX, and already vector) or GHS-POP (100m raster)
— Kontur is likely the better fit since it is hexagons out of the box and §5 wants cells anyway.

**Placement is not a claim about which people are where.** Within a unit, a group's dots are
scattered in proportion to total population. Where a source gives a finer unit we get finer truth;
where it does not, the dots say "this many people of this group live somewhere in this unit". The
about panel has to say so, because a dot map invites exactly the opposite reading.

## 9. Viewer

MapLibre GL JS + PMTiles, the ancestrydots stack, which also means the R2 hosting route and the
`npx serve` dev server (Python's `http.server` does not do range requests).

**Style is copied from ancestrydots, not from nycriders** — Anita's call, 2026-08-27, and this
supersedes the general house-style note in `feedback_map_ui_style`. Its tokens: `body` `#111`,
panels `rgba(20,20,20,0.758)`, hairlines `#2a2a2a`/`#333`, scrollbar thumb `#555`, Nunito, the
`i` button bottom-left for prose. Dark, like ancestrydots and unlike citybrowser.

**And its dot sizing is already the answer to §4.2's dot-size half:**

```js
'circle-radius': ['interpolate', ['exponential', 1.26], ['zoom'], …]
```

1.26 per zoom against a scale that doubles — radius grows about as the square root of the scale
factor, which is exactly the sublinear behaviour §4.2 asks for. It is tuned and it transfers; do
not re-derive it.

Panels: genealogy/legend (§10), unit composition on click (§5), about behind an `i` button.

## 10. The tree panel, and the genealogy drawn on it

**§6 splits this into two things that were one thing.** The panel is the legend, the selection
control and the palette scope all at once, so **it is in the first build** — the map does not work
without it. Only the *genealogy* half, the descent edges and the time axis, is later.

In the first build:

- The containment tree, collapsible per family (`<details>`, as ancestrydots), each node showing
  its current colour.
- **Selection is shared with the map, both directions.** Click a node → the focus palette applies,
  its dots and rings recolour and everything else drops to near-black. Click a dot or a line in a
  unit's composition panel → the tree scrolls to and highlights that node.
- Selecting a node selects its subtree, which is how you ask "where is Orthodoxy" rather than
  "where is the Romanian Orthodox Church".
- A **display-depth control** per selection, since it decides how many categories the wheel is
  divided among (§6). Probably ± buttons rather than a slider — the useful range is about four
  values.

Later, on the same panel:

- Vertical time axis, `from` edges as lines, dashed where `kind: disputed` (§2.1).
- The descent edges are the reason §2.1 exists now rather than later: retrofitting them onto ids
  that were not designed to carry them is the expensive version.

## 11. Open questions

Design questions for Anita are in `todo.txt`. The ones that are mine to resolve with a prototype:

1. Rings are decided; their *drawing* is not. Does a hollow ring read as "present, unquantified",
   or do people read it as a small dot? Ring weight, size and whether it sits above or below the
   dot layer all bear on this, and it is worth getting right — the honesty of §4.3 depends on the
   two symbols not being confusable.
2. The merge's dynamic range (§4.2): does area-proportional sizing hold across Tokyo and rural
   Mongolia in one view, or do the densest cells need a radius cap? And what does the cell size
   need to be, per zoom, for a mixed region to read as mixed?
3. How many drawn categories can the focus palette (§6) actually separate — 8, 15, 30? The answer
   sets the sensible default display depth for each scope, and it is the same number that decides
   how deep the tree is worth building (`todo.txt`).
4. Storage: 8.1M atomic dots plus the merged levels and the rings, through tippecanoe — what does
   it actually weigh, and do the merged low zooms hold tile sizes on their own without any
   dropping?
5. Whether the overview palette should be reachable while a selection is active — i.e. an escape
   hatch for "show me Christianity but keep it looking like Christianity". Cheap to add, and
   possibly clutter.

## 12. Things deliberately not being done

- **No world-history time slider.** cityhistory is that map. Religion over time at this
  granularity is a different and much worse-sourced problem, and mixing them would sink both.
- **No adherent-count aggregation across bases** (§3.1), however tempting the coverage would be.
- **No node invented at ingest.** Unmapped source categories go to a file and wait.
- **No log scale** (§4.1).
