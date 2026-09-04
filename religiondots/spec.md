# religiondots — how it works, and why

Interactive world map of religious affiliation as dots, at the finest branch/sect granularity
each region's data supports, with a religion genealogy panel that doubles as the legend.

**This file is Claude-managed.** `todo.txt` is Anita's and is not mine to edit. `sources.md` is
the per-source inventory and is also mine.

**Status as of 2026-09-03: twelve countries are drawn and tiled** — the US, Canada, Czechia,
Brazil, Australia, Ireland, Mexico, New Zealand, the United Kingdom, Poland, Romania and
Estonia. So this is now part design document and part record. Sections still marked PROPOSED
or open are the design half; the rest describes something that exists. `sources.md` §9a–§9e is
the running log of what each ingest taught, and `COMMANDS.txt` is how to rebuild any of it.

**If you are an agent about to add a country, read §12 first.** It is the accumulated list of
traps — the ones that cost an hour each and would have cost five minutes to read — and it is
explicitly meant to be added to. Put anything you learn there before you finish.

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
- **`from`** — the *descent* relation, a DAG with dates. Used by the genealogy panel — and, since
  2026-09-03, in a coarse linear form (`LINEAGE` in `branches.py`) for the *order* colours are
  allocated in, which is §6.5: sorting a parent's children by size put the two largest bodies
  adjacent on the wheel, and sorting them by descent does not.
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

### 2.5 The Catholic Church sits at a different depth in the US — DONE 2026-09-03

Four sources name the same body and three of them agree:

| source | category | maps to |
|---|---|---|
| `usrc2020` | `081` Catholic Church | `christianity.catholic.latin.`**`catholic-church`** |
| `ca2021` | Roman Catholic | `christianity.catholic.latin` |
| `cz2021` | Církev římskokatolická (and SSPX) | `christianity.catholic.latin` |
| `br2010` | Católica Apostólica Romana | `christianity.catholic.latin` |

`catholic-church` is the **only** leaf under `…catholic.latin` anywhere, and it holds 61,858 US
dots — over a third of the country. So in the US the branch and its single child are the same
people at two depths, and everywhere else they are one node.

Two things follow, and the first is the one a reader actually hits. A depth cut that shows "Latin
Catholic" puts the dots one level below the row, so before §6.3 they were drawn in a shade with no
swatch, and the tooltip said "Catholic Church" for a body that does not appear in the visible
legend. §6.3 fixed the *colour*, and the tooltip/label mismatch is still there. Second, the level
costs a step of depth in the US that buys no distinction, so US depth 2 is a shallower cut of
Christianity than Czech depth 2 for reasons of mapping rather than data.

**The fix is one line** — `"081": "christianity.catholic.latin"` in `usrc2020.py`, which
`build_tree.py` already supports (a mapping may point at a branch; the branch takes the `sources`
entry, as Islam and Sikhism already do) — and the `catholic-church` node then disappears, since
leaves are generated from the mappings. Nothing is lost: the US has no other Latin-rite body to
distinguish it from, and Czechia already folds SSPX into the same branch.

**Done 2026-09-03**, together with a second case of exactly the same shape that a scan turned up.

**`tools/scan_identities.py` is that scan**, and it is worth keeping because this class of defect
is invisible in the taxonomy and obvious on the map: it lists every branch with a single child,
every branch where one side of a split holds under 2% of the total, and what each side's dots are
per country. Run it when a source lands. What it found:

| branch | the child that was the same thing | why |
|---|---|---|
| `christianity.catholic.latin` | `…latin.catholic-church`, 61,858 dots | the case above |
| `hinduism` | `hinduism.temples`, 831 dots | **"Hindu Temples" is ASARB's row for Hindus, counted by temple** (§2.3) — a building type standing in for a tradition, not a sect, and every other source files Hindus on the branch |

Both are now mapped at the branch. `hinduism.vedanta` is the only other child there, it has no dot
and no ring in any built country, so Hinduism is one row in the world view rather than three.

**What the scan says is NOT an identity, which is the more useful half.** `spiritualism` /
Kardecist Spiritism (96% of the branch), `christianity.pietist` / Evangelical Covenant (94%),
`…baptist.landmark` / American Baptist Association (98%) all look identical in the tallies and are
not: each has real siblings that are merely below the dot floor. Presence pruning (§6.2) already
draws them as one row, and collapsing the taxonomy would throw away a distinction the next source
will report. The test is whether a sibling *could* exist, not whether one is currently drawn.

**The cost was the rebuild, and it was paid the cheap way.** `build_tree.py`, then the two node ids
rewritten in `dots_us.geojson` rather than a full `scatter.py --country us` — exact, because
`usrc2020.py` maps no other code to either branch, so the per-unit group sums and therefore the
largest-remainder allocation (§4.1a) are unchanged. `tiles.py` still has to run, since the ids are
baked into the archive, and until it does those dots carry a node that no longer exists and draw
grey — the one thing §6.6 says must never happen.

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

**The US/Canada border is the first place this becomes visible, and the numbers are in — 2026-08-27.**
The United States is `roll`: ASARB counts congregational membership, and **48.6%** of Americans
appear on one. Canada is `self_id`: the census asks the person, and **53.3%** call themselves
Christian, with a further 34.6% reporting no religion. Those two percentages are not comparable
and their difference is not a fact about religion in North America — one counts institutions'
records, the other counts self-description, and self-description is always the larger number.
Canada is also a **25% long-form sample**, not a full count, so it carries sampling error the US
roll does not.

Drawn naively, the 49th parallel will show a step change that is entirely an artifact of how the
two countries were measured. §3.1 forbids summing across that boundary; what it cannot do is stop
a reader comparing the two sides by eye, so the unit panel must name the basis in words, and the
step is a thing to point at in the about text rather than smooth away.

**Two different questions can both be called "religion", and the famous number is often the wrong
one — Northern Ireland, 2026-08-27.** NISRA asks "what religion do you belong to?" (MS-B19) and,
separately, "what religion were you brought up in?" (MS-B23/24). Q14 was **put only to people who
answered "None" or did not answer the first question**, so the brought-up-in table reassigns
**181,000 people — 9.5% of Northern Ireland — to a religion they had just said they do not belong
to.**

The two tables give materially different pictures: **42.3 / 37.4** on current belonging, **45.7 /
43.5** on upbringing. The second pair is the one in general circulation, including in most
reporting of the 2021 census.

Neither is wrong; they answer different questions, and the upbringing figure is the meaningful one
for some purposes. But they cannot be mixed, they cannot be compared with any other country's
self-identification, and the one a naive ingest would pick up is the one that is not about present
affiliation. Held as a **separate `source_id`** (`uk_ni_census_2021_brought_up_in`) rather than as
a variant of the same source, so nothing can group the two together by accident.

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

**Two ways the residual gets defeated, both found 2026-08-27 in New Zealand.**

- **The agency may have filled it in already.** 15.6% of New Zealand's 2023 religion answers are
  not 2023 answers — 9.2% carried forward from the person's 2018 or 2013 form, 6.5% imputed. The
  visible consequence is that **"Residual Categories" is zero in all 2,395 SA2s**: the not-stated
  residual has been silently absorbed. A zero residual therefore does not mean full coverage, it
  can mean the gap was filled where we cannot see it, and the detector reports nothing precisely
  where there is most to report. Every adapter should record whether the source imputes, because
  a pre-filled residual has to be treated as `derived` (§7) rather than measured.
- **In-band sentinels turn suppression into arithmetic.** Stats NZ writes **`-999` for
  "Confidential" in the same integer column as the counts** — 1,326 cells across 112 small SA2s.
  Summed as delivered, Islam comes to **−36,753** and No Religion lands 4% low but entirely
  plausible, which is the dangerous half: one result is obviously broken and the other quietly
  wrong. Every adapter must state its source's sentinel values, and the reconciliation check
  against published national figures is what catches the plausible ones.

So the residual is not only an output, it is **the main automatic detector of §3.1 and §3.2
violations**, and it is cheap because it comes out of arithmetic the pipeline is doing anyway —
but it is only as good as the two conditions above, and both must be checked per source.
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

**Measured for the first time, in New Zealand — 2026-08-27.** The NZ census accepts up to four
religious affiliations per person, so responses genuinely exceed people, and by how much depends on
how finely you cut:

| | inflation |
|---|---|
| level 1 (11 categories) | **+0.18%** (9,192 responses) |
| level 3 (163 categories) | **+0.70%** (32,886) |
| Christian, level 3 | +1.24% |
| Māori religions, level 3 | +1.64% |
| Islam, level 3 | +1.28% |
| **No Religion, level 3** | **+0.01%** |

No Religion is the control and it is what makes the rest trustworthy: nobody holds "no religion"
*and* something else, so its inflation should be ~0, and it is. The others are real multiple
affiliation rather than a processing artifact.

Two things follow. The inflation is **small enough to draw without correction** at these
granularities — under 1% is well inside the disclosure-control noise of §3.8. And it **grows as
categories get finer**, which is the direction this project keeps pushing, so it is worth
re-measuring rather than assuming 0.7% is a ceiling. A country where dual practice is the norm
rather than the exception (Japan, China, Vietnam) will not look like this at all, and that is what
the syncretic node above is for.

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

### 3.5a The United States is re-based on self-identification — DECIDED 2026-09-03

**The finding that forces it.** ASARB's 161.2M adherents are 48.6% of the country, so **171
million Americans — 51.6% — are currently drawn as nothing at all.** Not as unaffiliated, not as
unknown: absent. And because the residual of a roll means "on no roll", the map cannot draw the
American non-religious at all while Canada draws 34.6% of itself that way. The 49th parallel is
not a step in the data, it is a step between two questions.

**The decision, Anita's, 2026-09-03: the survey supplies the totals and ASARB supplies the
structure.** This is not a new mechanism. It is §3.4's "structure from the detailed source,
totals from the recent one" with `basis` where Brazil has `year`; it is the split §3.1 explicitly
permits and not the addition it forbids; the leftover is §3.2's `…other/unspecified` residual,
which §6.6 already draws; and §3.5 above has been asking for it all along, listing the United
States among the countries that do not ask and saying their composition should be survey-derived
and desaturated. The current build is the exception, taken because ASARB was too good subnationally
to pass up.

`tools/scan_selfid_gap.py` is the feasibility check, run before deciding. Against Pew's 2023-24
Religious Landscape Study, applied to the whole population:

| | self-ID | ASARB roll | residual |
|---|---|---|---|
| christianity | 205.5M | 152.0M | **+53.5M** |
| unaffiliated | 96.1M | — | **+96.1M**, entirely new |
| judaism | 5.6M | 2.1M | +3.6M |
| buddhism | 3.6M | 1.0M | +2.6M |
| hinduism | 3.0M | 0.8M | +2.2M |
| catholic | 63.0M | 61.9M | +1.1M |
| islam | 4.0M | 4.5M | **−0.5M** |
| latter-day saint | 6.6M | 6.7M | **−0.1M** |

**RESIDUAL AT L1 ONLY.** The survey's totals are taken at the root and nowhere else. Nine of
nineteen state-level subtractions tested came out negative, and every one was a body that keeps a
**baptismal register rather than a membership list** — a Catholic diocese reports the baptised
living in a parish's territory, an LDS ward reports everyone baptised who has not formally
resigned, and both hold people who would tell a surveyor they are something else now. Requiring
Catholic to clear its own self-ID line fails in Rhode Island, Massachusetts, New York, Louisiana
and New Mexico; requiring it only of Christianity fails in Utah alone. So the denominations sit
inside the root as structure and are never asked to clear a survey line of their own.

**The residual is spread over the people who are not on any roll.** Per state and per root,
`residual = self_id − Σ rolls`, and it is distributed across the state's counties in proportion
to `county population − county adherents` rather than to population or to the rolls. That pool is
real per-county data, it is the same pool the unaffiliated come out of, and it puts the
unspecified Christians of New Hampshire where New Hampshire's unchurched actually are instead of
smoothing them over the state. It also makes each county's drawn total come to its population
exactly, which is what every census country already gets for free.

**CAP AND RECORD where a roll still exceeds the survey.** The roll is a measurement of real
congregations and is drawn as it stands; the residual is floored at zero; and the overflow comes
out of that state's **unaffiliated** residual, because the likeliest reading of a name on a roll
that the surveys cannot find is a person who now says "nothing in particular". Utah is the case:
a 417k Christian overflow against a 1.11M unaffiliated residual, so Utah draws about 695k nones
instead. Every overflow is recorded per state and root at build time.

**The declaration stays quiet — Anita, 2026-09-03.** One sentence in the country note, the numbers
in the build log and in `counts.json` for anyone who looks, and nothing on the map itself. No
overlay, no badge, no second legend. The desaturation of §7 is already carrying "this is
modelled"; a reader who wants the size of the disagreement can find it, and a reader who does not
is not made to step over it.

**Three things this leaves open, recorded so they are not rediscovered.**

- **The child assumption is load-bearing and must be said out loud.** Adults are 78% of the
  population, so applying adult shares to everyone scales the survey by 1.28×. Catholic then
  clears its roll by 1.1M, under 2%. Applied to adults only it is 49.1M against a 61.9M roll,
  negative by 12.8M. What this map says about American Catholics rests on an assumption about
  children.
- **Islam is the calibration case.** ASARB's figure is a body literally named *Muslim Estimate*,
  so its −0.5M is not a roll against a survey but two estimates of one population disagreeing by
  12%. It is the only place the two instruments can be compared with the roll question removed.
- **This does not solve §4.4.** Pew publishes Sikh, Daoist, Bahá'í and Zoroastrian as a single
  "other world religions" line at <0.3%, and Unitarian, pantheist and Wiccan as "other religious
  identifications" at 1.9%; at n=36,000 nothing smaller can be broken out. The eleven
  congregations-only religions still need their own per-body sources. The two pieces of work are
  complementary, not alternatives.

**BUILT 2026-09-04, `us_rebase.py`.** The arithmetic came out where the feasibility scan said it
would, and the map now draws **326,813,748 of 331,449,281 Americans — 98.6%** against 48.4%
before. The residual is 166.2M people, so **a little over half of the American map is now a
derived row** and §7's desaturation (built the same day, for this) is what says so on screen.
Where it lands:

| | | |
|---|---|---|
| unaffiliated | 60.6M | 36.4% of the residual, and none of it drawable before |
| christianity | 54.2M | "Christian, no roll names them" |
| secular | 35.3M | atheist + agnostic + humanist |
| judaism, buddhism, hinduism | 8.4M | |
| unchurched, paganism, esoteric, other.us | 6.3M | |
| unitarianuniversalist, indigenous, spiritualism, newthought | 0.8M | |

Four things the build settled that the decision had not.

- **The residual is measured against the roll AS DRAWN, which is the county sheet.** ASARB's
  state sheet totals 160,786,973 mapped adherents and its county sheet 160,572,400: **214,573
  people are reported for a state and attributable to no county in it.** Subtracting the state
  figure computes a residual against a roll the map does not draw, and those 214,573 disappear
  from the country's total — which is how the first build came out at 326,599,176 instead of
  326,813,748. The rule generalises past this source: **a residual must be taken against what is
  actually drawn, not against the tidiest published version of it.**

- **The residual is `derived`, not `modelled`.** §7's `derived` is "a national figure
  distributed by a proxy" and this is a state figure distributed by a proxy; `modelled` is for a
  country estimate where there is no subnational data at all. It is the weak end of derived and
  worth saying so: the coarse total is a survey of 36,908 cut 51 ways, at a state margin of error
  of 3 to 8 points, converted by the child assumption — where Ireland's equivalent coarse total
  is a census count.
- **The residual must NOT go through §8.4's weights.** Every beta there was fitted to predict
  ASARB's own within-metro variation, so it says where the people on a roll live. Running
  "the people on no roll" through it would place them on top of the congregations they are
  defined by not belonging to. They take tract population and nothing else — `weights(...,
  plain=True)`, which is also the general rule: a model fitted on measured rows may not place
  derived ones.
- **Thirty counties get no residual at all**, and correctly. §3.6's counties that report more
  adherents than residents have a pool of `population − adherents` that is negative, it clips
  to zero, and 3,113 of 3,143 counties receive the spread. A county already over-full of
  other people's parishioners has no room for the unchurched, which is the same clamp §3.6
  applies for display.

**The overflow, all of it, as §3.5a asked.** 94 (state, root) pairs, 1,630,571 people — 0.49% of
the population — charged to those states' unaffiliated, and every state's unaffiliated residual
was large enough to absorb its share, so nothing went unplaced. It is not 94 findings: **Islam is
1,182,008 of it across 30 states**, and ASARB's figure there is a body literally named *Muslim
Estimate*, so that is two estimates disagreeing rather than a roll beating a survey. Utah's
Christianity is 295,066 more. The remaining 92 pairs are 153,497 people between them, and 57 of
those pairs exist because the survey returned a true zero for a small religion in a small state.

**The mapping is a cut, not a category match — done 2026-09-03, `taxonomy/us_pew2024.py`.**

"Totals at the root and nowhere else" turns the mapping into an unusual shape, and the shape is
worth naming because it is the first of its kind here. Every other mapping file answers "what is
this category?" for every category the source publishes. This one answers "where does Pew's tree
have to be cut so that each piece lands on exactly one of our roots?" — and it maps **28 of 149
categories**, leaving `southern-baptist-convention` and `global-methodist-church` deliberately
untouched. Most of the cut is Pew's own top level. It descends in two places only, both because a
single Pew node spans several of our roots: `other-christian`, since Spiritualism and New Thought
are roots of ours and Pew files them under Christianity; and `something-else`, whose descendants
run from Unitarian Universalism to Wicca to Native American religions.

**A cut entry is a set of roots, not one root.** `other-world-religions` is one line covering
Sikhs, Daoists, Bahá'ís and Zoroastrians, so its residual is the line minus the ASARB rolls of
*every* root in it — which subtracts the 178,727 Bahá'ís ASARB counts instead of drawing them
twice. Reading an irreducible lump as a single opaque bucket is how double counting gets in.

Two consequences to have on the record. **Non-response is excluded**, as Czechia's 30.05% and
Ireland's 6.7% are, so a county's drawn total comes to **98.60%** of its population rather than
to it exactly as claimed above; the share runs 0.09% in Massachusetts to 4.88% in Alaska, a
54-fold spread, so it is not a uniform haircut. And **atheist and agnostic go to `secular`,
"nothing in particular" to `unaffiliated`** — the line branches.py already draws for Canada's
identical answers, now with a second source needing it.

**Cap-and-record, measured over all 51 states rather than eight — 2026-09-03.** The feasibility
scan hand-checked the eight states where a roll was most likely to win. Run over every state and
every root the two instruments can both see, with the child conversion applied:

| | |
|---|---|
| (state, root) pairs where the roll exceeds the survey | **62 of 255** |
| total overflow | **1,694,041 people, 0.51% of the population** |
| of which Islam | 1,182,008 — **70%**, across 30 states |
| of which Utah, Christianity | 395,685 |
| Judaism + Buddhism + Hinduism, 31 pairs | 116,348 |
| negatives caused by a **true zero** in the survey | 31 of the 62 |

Two readings, and they point the same way. **Christianity is negative in Utah and nowhere else**,
across all 51 states — the finding above, confirmed at full coverage rather than inferred from a
sample of the likely cases. And the overflow is not really 62 findings: 70% of it is Islam, which
is the calibration case already named — ASARB's *Muslim Estimate* is not a roll — and half the
remaining pairs are a survey of 36,908 people returning a true zero for a small religion in a
small state, which is a fact about the survey rather than a disagreement about people. What is
left is small enough that "record it and charge it to the unaffiliated" remains the right rule.

### 3.6 A roll counts the institution's location, not the member's — FOUND 2026-08-27

The first real check on the first real dataset, and it is a defect class nobody would have
predicted from the methodology page. The US Religion Census attributes adherents to the county of
the **congregation**, and people do not always worship in the county they live in. So:

**30 of 3,143 counties report more adherents than residents.** King County, Texas: population 265,
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
| counties with a negative residual | **30 of 3,143** |
| total overshoot nationally | **42,892 people = 0.0065% of US population** |
| largest single county | Fredericksburg city VA, 9,189 people (133%) |
| largest by ratio | King County TX, 934 people (452%) |
| under 1,000 people | 21 of the 30 |

(The denominator read 3,144 until 2026-09-04. ASARB's summary sheets end with a blank row and a
`Totals` row whose key is the **string** `Totals`, so `notna()` keeps it — the same slip doubles
the US population if the figure taken is a sum. §8.1 has the real count from the boundary join:
3,143, all matching.)

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

### 3.8 Disclosure control biases the rare categories downward — FOUND 2026-08-27

Statistical agencies perturb published cells to protect individuals. The ABS perturbs **every cell
of every table independently**; StatCan random-rounds; others suppress below a threshold. The
national totals barely move — Australia's SA2 sums come to 25,422,677 against a published
25,422,788, off by 111 people, 0.0004%.

**The error is not distributed evenly, and it lands on us.** Measured on the Australian tables, the
bias runs systematically downward for small categories: Australian Aboriginal Traditional Religions
**−6.29%**, Brethren **−1.71%**, against −0.0004% for the total. Perturbation is roughly constant in
absolute terms, so as a group gets smaller the relative distortion grows without bound — and small
groups are this project's subject. R2 and R3 ask for precisely the categories the disclosure
control damages most.

**Rounding is the second mechanism, and it manufactures residuals.** StatCan random-rounds every
count to a multiple of 5 — verified exactly: **0 of 321,757 Canadian counts is not a multiple of
5**. So a parent and the sum of its children disagree by ±5 to ±25 as a matter of course, two
StatCan products disagree by ±5 on the same national figure, and **none of it is an error**.

That interacts directly with §3.2's residual detector and with §3.6's threshold. A residual inside
the rounding envelope is noise and must not be chased; §3.6's absolute floor of 1,000 people
already covers Canada's ±25, which is why the threshold has an absolute term and not only a
percentage one. Each source needs its rounding rule recorded so the envelope is known rather than
guessed.

Nothing can be done to recover the true figure, so the response is to record it: a group's
published count carries an uncertainty that is a function of its size, not of the source's quality.
It is also a reason not to over-read a small difference between two rare groups in one place, and a
reason the presence ring (§4.3) is on firmer ground than a small dot count would be — "present"
survives perturbation in a way that "4,123 people" does not.

### 3.9 Category detail and spatial detail trade off inside one source — FOUND 2026-08-27

Not the §3.4 case, which is two sources of different vintages. This is **one source, one year,
where the agency publishes fine categories OR fine geography and refuses to publish both**, because
the cross-tabulation would identify people.

Australia is the clean example. The ASCRG 2021 classification has **150 religious groups**, and the
ABS publishes all 150 — nationally. At SA2 the same census gives **34**, and everything small is
folded into a single `Other Religious Groups` column of 107,127 people that contains Bahá'í,
Taoism, Shinto, Paganism, Wicca, Jainism, Zoroastrianism, Mandaean, Yezidi, Druze, Caodaism,
Spiritualism and Rastafari together.

**Mexico is the same shape and confirms it is structural, not an ABS quirk.** INEGI publishes
**24 denominations × 32 entidades**, or **4 aggregate groups × 2,469 municipios**, and there is no
table joining the two. Worse, its classification *codes* 46 denominations and publishes 24 —
Mennonites, Lutherans, Buddhists and Hindus exist in the database and in no released table at all.
Two of three countries checked so far have this; assume it until shown otherwise.

So for Australia the map can show *where* 34 categories are, or *how many* of 150 there are, and
the thing R2 actually wants — where the Yezidis are — is withheld by design. Options, in order of
honesty: take the coarse-geography detail and place it by §3.4's structure-from-elsewhere rule,
clearly marked as derived; or draw the `Other` bucket as itself, which is truthful and useless; or
find a custom tabulation. This is the shape to expect wherever a census has good categories, and it
is worth checking per country rather than assuming the finest geography carries the finest
categories — it usually does not.

### 3.9a A register does not make the trade at all — FOUND 2026-09-04 with Germany

§3.9 reads like a law about sources. It is a law about **survey** sources, and Germany is the
case that shows the difference, because it sits at both extremes at once.

**Zensus 2022 does not ask about religion.** There is no question on the form. The published
figures are read off the *Melderegister*, which records membership of a public-law religious
society because it determines **church-tax liability**. So the basis is `roll` (§3.1) —
institutions' records — and it is the first source here that is neither a question nor a
church's own count, but a **state register kept for a fiscal purpose**.

What falls out is not a trade-off but both endpoints:

| | |
|---|---|
| geography | **10,786 Gemeinden**, and the same figures on a **100m grid, 3,088,036 cells** |
| categories | **three** |
| suppression | 178 true-zero cells of 32,358; nothing withheld |

A register covers everybody exactly and knows almost nothing, so there is no
cross-tabulation risk to manage and nothing to withhold — and equally nothing to reveal. The
finest geography on this map and the coarsest categories on it have **one cause**.

**The rule this gives, and it is the general one: ask what INSTRUMENT produced a category
list before treating the list as a classification.** Germany's three categories are the set
of corporations that levy church tax. That is a fact about German public law, not about
German religion, and no amount of looking for a better table will deepen it — `sources/de.md`
§2 records why Zensus 2011, which *did* ask, is worse rather than better.

**The half of the country this cannot see.** "Sonstige, keine, ohne Angabe" is 51.8%, and
destatis is explicit that the register's entries for *other* public-law bodies cannot
"zuverlässig statistisch abbilden" their membership. So one category holds people in another
body, people in no body, and people with no entry — Germany's roughly four million Muslims,
its Orthodox Christians, its Jewish communities, its Freikirchen and its Alt-Katholiken among
them, unrecoverable.

That is §3.5 ("undercounting is marked, not filled") in its sharpest form so far, and §14.3's
rule forbids the obvious rescue: estimating those groups from national totals would invent
both magnitude and location, at a resolution the source does not publish. It gets a node that
says what it is instead — `unrecorded`, §6.3a — and the about panel carries the rest.

### 3.10 Reuniting fine categories with fine geography — MEASURED 2026-08-27

§3.9 leaves every source split in two: fine categories at coarse geography, coarse categories at
fine geography. The obvious repair is to combine them. Because our categories nest inside branches,
the estimate is a within-branch proportional allocation —

```
est[fine unit, body] = coarse_cat[fine unit, branch] × fine_cat[coarse unit, body]
                                                     / fine_cat[coarse unit, branch]
```

— which is what iterative proportional fitting reduces to when the fine categories nest. It assumes
**a branch's internal composition is the same in every fine unit inside a coarse one**, which is
false precisely for clustered minorities.

**The US can price that assumption**, because ASARB publishes fine categories *and* fine geography.
`tools/test_allocation.py` coarsens it to (county × 33 branches) + (state × 216 bodies), runs the
allocation, and compares against the known county × body truth. 160.6M adherents, 64,568 cells:

| | median misallocated |
|---|---|
| bodies over 1M | **12.0%** |
| 100k – 1M | 28.3% |
| 10k – 100k | 34.8% |
| **under 10k** | **41.7%** |
| **all adherents (total variation distance)** | **5.84%** |

**So it works for what you can already see and fails for what the project is about.** 94% of people
land in the right body; a body under 10,000 has about 42% of its members put in the wrong county.

Three things sharpen that further, all in the wrong direction:

1. **This is a lower bound.** The test allocates state → county. The real cases are worse jumps:
   Australia is nation → SA2, Canada province → CSD, New Zealand nation → SA2.
2. **The headline is flattered by branches with one populated child.** Bahá'í, Episcopal, Mahayana
   Buddhist and others score 0.0%, not because the method is good but because there is nothing to
   allocate. The genuine multi-child cases are all worse than 5.84% suggests.
3. **We cannot reliably predict which bodies it will fail on.** Concentration correlates with error
   in the right direction — counties-per-state against misallocation gives r = −0.30 — but that
   explains under 10% of the variance, so "flag the clustered ones" is not available as a fix.

**Decisions.**

- **Do it, and mark it.** An allocated figure is `derived` in §7's terms and draws desaturated, in
  the same visual class as a modelled country. It carries `structure_year` / `structure_geo` so the
  unit panel can say where the split came from.
- **Never let an allocated count reach a ring.** §4.3's ring means "present here", and allocation
  cannot establish presence — it spreads a coarse total over units that may have none of that body
  at all. A ring must come from a real count or a location source (§4.4), never from this.
- **Prefer a proxy where one exists.** Flat proportional allocation is the fallback, not the
  method. Several censuses publish ancestry, birthplace or language at the fine geography — the ABS
  gives country of birth at SA2 — and conditioning the split on a correlated variable beats
  spreading uniformly. Yezidis follow Iraqi birthplace; Jains follow Indian ancestry. Unmeasured so
  far, and the obvious next experiment, since the same US ground truth can price it.

### 3.10a Built for Australia — and what it does and does not buy

`allocate.py`, run on the ABS data: **30 categories at SA2 become 147**, 363,384 rows, people
conserved exactly, every category's national sum landing within perturbation of its published
figure (Paganism +7, Yezidi +2, Greek Orthodox −110).

Three things the build settled:

**The mapping must be validated arithmetically, never trusted from codes.** Australia looks like a
clean prefix hierarchy and is not: the SA2 column `603 Other Religious Groups` carries its own
prefix children *and* every other narrow group in broad group 6. A pure prefix join drops 30
categories and 92,331 people in silence. `allocate.py` therefore sums each fine column's children
against that column's own total and **refuses to allocate a column that does not reconcile** — which
caught `000 Religious affiliation not stated` (−94.7%, the two sources define it differently) and
`601 Australian Aboriginal Traditional Religions` (+6.7%, §3.8 perturbation on a small column).
Fourteen columns turned out to have a single child, so they are exact and are tagged `measured`
rather than `derived`.

**The failure mode is visible, not statistical.** Every category inside a bucket receives the *same*
distribution, so Yezidi and Paganism come out with an identical SA2 ranking differing by a constant
4.52. The map would assert that Yezidis and Pagans live in the same places in the same proportions.
That is the 42% figure of §3.10 made concrete, and it is more useful stated this way.

**So allocation rescues the middle, not the tail — which was the point of doing it.** Greek Orthodox
(390,853), Serbian, Russian and Antiochian Orthodox were hidden inside one `Eastern Orthodox` column
and are now separable at SA2, and that is a real and defensible gain. But Australia's largest
bucketed minority peaks at **72.9 allocated Yezidis in one SA2** — far under a 1,000-person dot, and
barred from a ring because allocation cannot establish presence. The groups R3 cares about stay
invisible after allocation, and only §4.4's location sources reach them.

A consequence for R1: because a bucket's members all inherit the bucket's footprint, **every SA2 with
a non-zero `Other` now nominally contains 29 religions**. Counting drawn categories would therefore
overstate diversity. Allocated categories below the dot floor must not be counted as present.

### 3.10b Canada, and the two checks that earned their place

| | fine geography | before | after | rows |
|---|---|---|---|---|
| Australia | SA2 (2,472) | 29 | **148** | 365,856 |
| Canada | CSD (5,161) | 23 | **147** | 758,667 |

Both conserve people exactly. Canada is the larger prize: Old Order Mennonites, nine Eastern
Orthodox jurisdictions, Doukhobors and Mar Thoma are now placed at census-subdivision level.

**Sources encode their hierarchy differently and there is no use pretending otherwise.** Australia
nests by code prefix (ASCRG `2233` under `223`); Canada names each row's parent. `allocate.py`
takes `--hierarchy prefix|parent`. Two failures worth keeping:

- **Only the coarse tree's *leaves* may be allocated.** Canada's province table contains the whole
  tree — every aggregate as well as every leaf — so climbing each category to its CSD column summed
  `Catholic` *and* `Eastern Catholic` *and* `Roman Catholic` into one column and produced children
  at **2.008× the column**. A structurally wrong mapping shows up as a clean multiple, which is what
  makes the reconciliation check worth running.
- **The reconciliation is a check on the mapping, not on the totals**, and getting that backwards
  cost six Canadian columns. Shares are normalised *within* a column, so a mismatch between the
  children's sum and the column total cannot affect the answer — only relative composition can. A
  2% tolerance therefore rejected `Anabaptist` because StatCan's province and CSD products disagree
  by 2.5–4.4% on the same category, which is a fact about StatCan, not a mapping error — and
  dropping it would have discarded every Old Order Mennonite group, precisely the granularity this
  project exists for. The band is now 10%, wide enough for product disagreement and far too tight
  for a 2× structural error.

**A gap in the normalized format, which is mine.** The source contract fixed the CSV *columns* but
said nothing about recording the source's own classification hierarchy. Australia and Canada
encoded it anyway (`ascrg=`/`parent=`) and can be allocated today; **New Zealand, Ireland and Mexico
did not**, so their hierarchies exist only as prose in the `sources/*.md` files and each needs a
small hand-written mapping table before it can be allocated — Mexico's is 24 categories onto 4, and
is self-checking, since a wrong assignment fails the column reconciliation. The normalized format
should require a machine-readable parent or code for every row, and any future source adapter
should be asked for it explicitly.

### 3.10c Allocation must sometimes run WITHIN a coarse unit — FOUND 2026-09-03 with India

`allocate.py` pooled every coarse unit into one national composition. For Australia that is
literally right (`--coarse nation`) and for Canada it is close enough. **India makes it a
catastrophe**, and the reason is one line of data: Sanamahi is 100% Manipur, Niam Khasi 100%
Meghalaya, Donyi-Polo 98% Arunachal Pradesh, Sarna 83% Jharkhand. A pooled national share would
put Manipuri and Arunachali religions into every sub-district in India in proportion to its
`Other` count — §3.10a's "Yezidi and Paganism come out with an identical ranking" failure, scaled
to a billion people.

**`--within N`** allocates inside each coarse unit: a fine unit takes the composition of the
coarse unit whose `geo_id` is the first N characters of its own. Each religion then reproduces its
published state distribution exactly, because that is now what it is being asked to do.

**The side effect is worth more than the fix, and it changes what §3.10 is *for*.** The
single-child test — a column with one child needs no allocation and stays `measured` — becomes per
*(coarse unit, column)*. A state whose only named `Other` religion is Sanamahi has nothing to
allocate, so its sub-districts get an **exact** split. **245 of India's (state, column) pairs are
exact that way.** Splitting the coarse geography does not merely produce better estimates; it
converts estimates into measurements wherever a coarse unit has only one answer. §3.10's measured
42%-misallocation cost applies only to what is left.

India's derived share ends at **0.66%** — the 7.94M in `Other religions and persuasions`, against
six religions measured on all 5,988 sub-districts. The best measured/derived ratio of any allocated
country by a wide margin, and almost all of it comes from `--within`.

**The rule: any source whose coarse table has many units, whose categories are regionally
clustered, should allocate within them.** Which is most sources; pooling was a convenience that
happened to suit the first two countries that needed allocating.

### 3.10d Arithmetic consistency is not evidence of meaning — FOUND 2026-09-03

The most transferable thing India taught, and it is a limit on every check in §3.10.

India's C-01 **Annexure** is titled *Details of sects/religions clubbed under specific religious
communities*, and it is arithmetically flawless: for every state and every one of the six
religions, `Religion:X` = an unspecified remainder + the named sects, to within a few hundred
people nationally. It is a true partition. **Every structural test `allocate.py` applies passes
it** — the children reconcile against the column, the hierarchy is unambiguous, the totals are
exact.

It names **573 Shia Muslims** among 172.2 million, and **8,399 Catholics** among 27.8 million
Christians.

What it actually counts is people who wrote a *sect* where the form asked for a *religion* — a
measure of insistence, not of membership. Every figure in it undercounts its community by one to
three orders of magnitude. Allocating it would have put numbers on the map wrong by 100×, in a
direction no confidence marking in §7 can express, and the rows would have carried `derived`
honestly while being nonsense.

**So the reconciliation in §3.10b checks that a mapping is structurally right and cannot check
that a category means what its label says.** Nothing inside the data could have caught this. What
caught it was reading five numbers and knowing roughly how many Catholics India has.

**Every allocation source needs one sanity check from outside the data, and it should be a number
a person already knows.** The trap is specifically that the *large* entries look fine: Lingayat is
2,663,229 and 99% Karnataka, which would have drawn beautifully and is wrong by a factor of four
against a community usually put near 10 million. The small absurd ones are what give it away, so
**read the whole list, not the top of it.**

### 3.11 Reducing "other", and the floor under it

The complement of §3.10: rather than splitting a bucket we cannot split, shrink it honestly.

- **An external national estimate can name a category the census refuses to.** Mexico files
  Orthodox Christians inside *otras religiones*; a published national estimate of Orthodox Christians
  in Mexico, allocated across the *otras* bucket, converts an anonymous residual into a named group.
  This is §3.4's structure-from-elsewhere rule applied to categories instead of geography, and it
  inherits §3.10's error bars — it is `derived`, and it is still better than a bucket labelled
  "other".
- **What remains stays named by its source, never merged.** `mexico.otras-religiones` and
  Australia's `Other Religious Groups` contain different things — one holds Orthodox Christians, the
  other holds Bahá'í, Jain, Yezidi and Wiccans. A single global `other` node would assert they are
  the same, which is false and unnecessary. So residual buckets are **per source**, named for it,
  and a country's "other" is a fact about that country's statistical agency rather than about its
  people.
- **The floor is real and will stay high.** Some of it is irreducible — England and Wales publish no
  Christian denominations at all, and no external estimate reconstructs 60% of a country's
  population at output-area level. The goal is to shrink the bucket where evidence allows and label
  it honestly where it does not, not to drive it to zero.

### 3.7 A census counts households, and monasteries are not households — FOUND 2026-08-27

Found in the Philippine census and it is not a Philippine quirk: census religion tables are
generally tabulated on the **household population**, which excludes the *institutional* population
— people living in barracks, prisons, hospitals, dormitories, **monasteries and seminaries**.

The Philippines: household population 108,667,043 against a total of 109,035,343. The gap is
**368,300 people, 0.338%** — small, and composed of exactly the residents this map most wants to
see. R3 asks for monastic communities to be findable, and **the census-shaped half of our sources
structurally cannot see them**, no matter how fine the geography or how granular the categories.

This is not a defect to correct; it is a statement about what these sources measure, and it
sharpens why §4.4's location-by-religion sources are load-bearing rather than a nice extra. The
institutional population is precisely the gap that an Annuario Pontificio or a monastery register
fills, and it is the reason those sources cannot simply be dropped in favour of "better censuses".

Every source adapter should record whether its universe is household or total population, and
`sources.md` should state it per country, because a country reporting *total* population is
measuring a different thing from one reporting household population and the difference lands
entirely in this project's subject matter.

## 4. The size problem — §4.1 and §4.3 DECIDED, §4.2 open

Christianity is ~2.4 billion. A Carthusian charterhouse is ~20 monks. Eight orders of magnitude,
and R3 says both must be on the same map without the small one lying about its size.

### 4.1 What is decided: dot count stays linear in people

No log scale, no sqrt, no per-group rescaling. Within any single view, one dot is one fixed
number of people for **every** group, so two groups' dot counts are exactly their population
ratio. This is the property the whole map is for and nothing below is allowed to break it.

### 4.1a Fractions carry along a spatial order — DECIDED 2026-09-03, twice wrong before

The first build floored every (unit, node) pair on its own — `count // dot_value`, remainder
discarded — and with 147 categories over 5,161 units that is a great many remainders. They are not
rounding noise. **Canada lost 5.5M people, 15% of the country**, to pairs that each rounded to
zero, and a denomination with 400 adherents in each of a hundred subdivisions is 40,000 people
that drew nothing anywhere.

**The first fix rolled sub-floor fragments up the taxonomy** — 400 Old Order Mennonites became 400
Anabaptists — which preserved the mass by spending the category to do it. It also put dots on the
map under branch ids: `christianity` was the fourth-largest "category" in the US at 12,153 dots,
none of which is anybody's denomination. Removed.

**The second fix spread each node's national total by LARGEST REMAINDER** — floor everywhere, then
hand the dots still owed to the units with the biggest leftovers. It preserved the mass exactly
and **destroyed the geography**, which took two weeks of looking at it to notice because every
national total stayed right. The error was purely spatial.

The mechanism: the leftover *is* the local count whenever no unit can reach a whole dot on its
own. In England and Wales the median Output Area holds 306 people against a 1,000-person dot, so
**2 of 188,880 units earned a dot from the floor** and all 27,520 remaining Christian dots went by
rank. Ranking on absolute count hands every dot to the places where a group is already densest:

| | drawn, as a share of the people actually there |
|---|---|
| Christians in OAs that are 10–20% Christian | **10%** |
| Christians in OAs that are 20–30% Christian | 12% |
| Christians in OAs that are 50–70% Christian | 148% |
| Christians in OAs that are 70%+ Christian | **282%** |
| Muslims in OAs that are 5–10% Muslim | **1%** |
| Muslims in OAs that are 50–70% Muslim | **332%** |

It is a contrast amplifier: it deletes a group from everywhere it is a minority and multiplies it
everywhere it is a majority. Whitechapel is 22% Christian and drew **no Christian dot at all**,
because an Output Area needed 193 Christians to win one and it had about 49. The map said 99%
Muslim about a borough that is 40% Muslim. Found 2026-09-03 by looking at London and disbelieving
it.

**The rule, decided by Anita and general to the project: never hand dots to the top n. Accumulate
along a geographic order and drop each dot wherever the accumulator happens to be when it passes
`dot_value`.** Walk the units in spatial sequence, add up the people, and every time the running
total crosses another dot, put a dot in the unit you are standing in. A unit holding a third of a
dot's worth of people gets a dot about a third of the time, and the dot lands *among the people
who contributed it*.

**Why this and not largest remainder.** The previous decision rejected a sequential carry as "a
spatial bias that means nothing and changes if the input is sorted differently", and it was right
about that — a carry in FIPS or DGUID order walks state by state for no reason. The answer is not
to abandon the carry but to **fix the order**: `scatter.py` walks a Hilbert curve through the
units, so consecutive units are neighbours on the ground. That is the property that makes the
carry land in the right place, and it is what makes the objection go away rather than being
traded against. Deterministic, no rng, and reproducible without being alphabetical.

Verified against ONS ground truth after the change — Tower Hamlets, clipped to the real borough
boundary and counted from the drawn dots:

| | drawn | ONS |
|---|---|---|
| Muslim | 43% | 40% |
| No religion | 28% | 27% |
| Christian | 24% | 22% |

Every band in the table above lands within a few points of 100% instead of between 1% and 332%.

**What is still lost is under one dot per node, nationally.** A node whose entire national total
is under `dot_value` draws nothing at all, and that is intended rather than regrettable: below one
dot the map says nothing rather than inventing a thousand people. §4.1 is the constraint — a dot
is a fixed number of people or it is not a dot, and there is no honest way to draw 600 people at
1:1,000.

**This changes what a below-floor ring claims** (§4.3). It used to mean "these people are counted
here and represented nowhere on the map". It now means "no dot landed *here*", while the people it
stands for are on the map in a neighbouring unit of the same node — and under a spatial carry that
neighbour is genuinely adjacent, which it was not under largest remainder. That is a much weaker
statement, and it is half of why rings are no longer drawn by default.

### 4.1b People per dot is a setting, not a constant — DECIDED 2026-09-04, Anita's call

India made the archive 1.87M dots, of which it is 1.21M on its own, and the reasonable question
followed: can the reader ask for fewer? The answer is yes, at a cost that has to be stated
rather than buried, because **coarsening the dot value is not a rendering option — it changes
what the map asserts exists.**

**Two editions ship: 1:1,000 (default) and 1:10,000.** `scatter.py --dot-value` writes
`dots_<cc>_10k.geojson`, `tiles.py --coarse` packs both into one archive as `dots10k` /
`atomic10k` / `rings10k`, and the viewer swaps layer visibility exactly as §4.2b's consolidation
toggle already does. India goes 1,207,981 → 120,790 dots.

**Measured cost of carrying both**, on the 14-country build: the archive goes **121.1 MB →
149.5 MB**, +23% for a tenth-scale copy of everything. Not the +10% a feature count would
suggest, because MVT's per-tile overhead does not shrink with the features in it and the coarse
edition still touches nearly every tile — z10 holds 177,259 coarse marks against 1,540,546 fine
ones, but they are spread over the same 33,321 tiles. `--coarse` is therefore opt-in at build
time, and the viewer reads `dot_values` out of `counts.json` rather than assuming, so an archive
built without it greys the control out instead of offering a setting that does nothing.

**Why a second scatter and not a subsample of the first.** Showing one dot in ten would have
been far cheaper and is wrong twice. The counts would hold only in expectation, which breaks
§4.1's "count the dots"; and a group with three dots nationally would appear or vanish on the
random seed, where a real 1:10,000 run drops it *deterministically* and §4.3 can then give it a
ring. A coarse dot is a different measurement, not a filtered fine one.

**What it costs, and this is the part that must not be hidden in a tooltip.** At 1:10,000 the
floor under a group rises tenfold, so small groups leave the map. Where a group's count is
`measured` it becomes a presence ring and is still there to be seen — the US gains 109 rings,
Czechia 31, Estonia 12. **Where its count is `derived`, it simply goes, with no mark at all**,
because §3.10 forbids an allocated count from asserting presence and a ring is exactly that
assertion. India is the worked example: at 1:10,000 it holds 15 nodes instead of 17, and the two
that vanish — Bahá'í (4,572) and Judaism (4,429), including the Bnei Menashe of Manipur — leave
nothing behind, because both come from the allocated Appendix.

Two correct rules meeting to delete 9,001 people from the picture is not a bug and is not
fixable without breaking one of them. It is, however, the whole argument for **1:1,000 staying
the default**: the coarse edition is the performance escape hatch, not the map.

**Drawn 2.5× larger, and constant ink was the wrong target.** A 1:10,000 dot stands for ten
times the people, so it has to be drawn bigger — at the same size the coarse map would just
look like a country that had lost 90% of its population. The obvious gain is `√10 ≈ 3.16`,
which holds total ink constant, and it was built that way and is wrong: **ink does not add.**
Ten 1:1,000 dots overlap, so they cover appreciably less than ten times one dot's area, and a
single dot of exactly ten times the area therefore over-inks — the same reason §4.2a caps a
merged mark instead of letting it stay strictly area-proportional. `COARSE_GAIN = 2.5`, Anita's
call on the New York view at 1:10,000, where 3.16 read as blobby.

**§4.2a's refusal to go sublinear does not bind here, and the distinction is the point.** There,
radius ∝ √k encodes a per-mark magnitude the reader is meant to read back, so bending the curve
would misstate `k`. This is one uniform constant over every dot in the edition, with the dot
value stated in the legend — it encodes nothing per-mark, so it can be tuned by eye without any
figure on the map becoming untrue. A number chosen for legibility and a number chosen to carry
magnitude are different kinds of number, and only the second is bound by §4.1.

**1:100 was asked about and is not built.** It is the nicer map everywhere small, and it is
12.1M dots for India alone — the wrong direction for the problem that prompted this.

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

### 4.2a Built 2026-08-27 — `tiles.py`, and why not tippecanoe

The merge is done by **`tiles.py`, which writes PMTiles directly**, and tippecanoe is
deliberately not in the pipeline even though ancestrydots uses it and it is the obvious tool.

**Tippecanoe's low-zoom job is to drop features** until a tile fits a byte budget
(`--drop-densest-as-needed`). Which dots survive is close to arbitrary, so a small group blinks in
and out as you zoom or pan — the stochastic-disappearance failure §4.2 exists to avoid, arriving
through the back door of the packaging step. Merging is a different operation and no tiler does
it, because it needs to know that two dots are *the same religion* and may be combined.

So each zoom gets its own aggregate: the tile is divided into 32×32 merge cells (`CELL_BITS = 5`,
about 16px on a 512px tile), all dots of one religion in one cell become one mark carrying `k`,
and the mark sits at the **mean position of its members** rather than the cell centre, so marks
follow the real point cloud instead of snapping to a lattice.

Measured on the US at 1:1,000 — 141,501 dots in, nothing dropped at any zoom:

| zoom | marks | largest merge |
|---|---|---|
| 0 | 861 | 11,175 dots |
| 2 | 3,499 | 7,767 |
| 4 | 12,907 | 3,406 |
| 6 | 40,327 | 799 |

A useful side effect: **this removes the WSL dependency entirely**, which matters because WSL's
C: mount is broken on this machine and tippecanoe was not reachable at all.

**The radius cap, which was §4.2's open question, is now answered.** Merging turns many
overlapping small dots into one solid circle, and a circle whose area is the *sum* of overlapping
dots is larger than their union — so strict area-proportionality over-inks the densest cells.
Radius is `min(r(z)·√k, 8px)`, the cap being half a merge cell so a mark cannot spill over its
neighbours. **Above the cap a mark has stopped reporting magnitude**, which is the same kind of
statement a ring makes and should be read the same way: the cell is full. It is a real loss of
information in exactly the densest cells and it is bounded, visible, and better than the
alternative of letting one circle swallow a state.

cityhistory answered the same dynamic-range question with sublinear bubbles; that trade is not
available here, because there bubble area is the only encoding of a city's size, while here the
mark stands for a countable number of atomic dots and inflating it would break the "count the
dots" property §4.1 exists to protect. A hard cap loses information honestly; a sublinear curve
misstates it everywhere.

**Which end of the range to sacrifice is the actual decision, and it is the bottom that must be
protected.** At z4 the merge spans k = 1 … 3,406, a radius ratio of 58, against the roughly 9×
that fits between "visible" and "not overlapping the neighbours". Something has to give. The
first build set the base radius low, so a `k = 1` mark drew at 0.42px and **the countryside
emptied out** — which is not a rendering artifact but a false claim about where people are. The
base is now set so a lone dot stays visible and the cap bites from about k ≈ 80. Losing
resolution among the largest metros costs nothing anyone can read; losing rural America costs the
map its subject.

### 4.2b Consolidation is toggleable — added 2026-08-27

The merge happens at build time, so switching it off is not a rendering option — the unmerged dots
have to be *in the archive*. `tiles.py` emits them as an `atomic` layer beside `dots`, and the
viewer swaps layer visibility (**overlapping dots: merged / separate** in the legend).

The atomic layer is every dot at every zoom, where the merged pyramid collapses them, so it is the
expensive half of the archive — and much less expensive than expected:

| | archive |
|---|---|
| merged only | 11.3 MB |
| merged + atomic | **17.6 MB** |

+6.3MB for 1.56M feature-instances, against an estimate of ~35MB. MVT deduplicates the repeated
category strings almost completely, so the marginal cost of a point is close to its coordinates
alone. `--no-atomic` drops it for when that stops being true.

**Both views are honest, and they answer different questions.** Merged: area is proportional to
people, so you can compare quantities across a view. Separate: one mark per 1,000 people, so you
read texture and mixing. Separate is also the more familiar dot-map idiom and is what ancestrydots
does.

**Separate is the default as of 2026-09-02** (Anita's call). Merged used to be, on the grounds that
at 1:1,000 four out of five body-county pairs were rings and merging kept the rest legible when
zoomed out. Both halves of that have gone: rings are no longer drawn unasked (§4.3), and §4.1a's
carry means far fewer pairs are sub-floor in the first place. What is left is that the plain
scatter is the more honest-looking object — a merged mark is a circle whose area the reader has to
decode, where a field of equal dots is read by counting, which is the property §4.1 exists to
protect. Merged stays one click away and is still the better view at continental zoom.

**One MapLibre trap, which cost a build.** `['zoom']` must be the *outermost* expression of a
paint property. Wrapping the zoom curve in `['min', …]` to apply the cap fails validation, and
MapLibre's response is to **drop the entire layer** with the error only on the console — the map
still renders, with rings and no dots, looking like a data problem rather than a syntax one. The
cap and the √k factor go inside each interpolation stop's output instead.

### 4.2c Draw order is randomised, or the biggest group loses — FOUND 2026-09-03

**Symptom:** São Paulo read Spiritualist zoomed out and Catholic zoomed in. Brazil has 169M
Christians and 3.9M Spiritualists, 43 to 1.

**Cause:** vector-tile features paint in file order, so where dots overlap the last one written to
the tile is the only one you see. `scatter.py` emits dots grouped by node in sorted order within
each placement polygon, which means the alphabetically-last religion present in an area paints
over every other one — `spiritualism.*` after `christianity.*`, every time. Zoomed in the dots do
not overlap and the truth shows through; zoomed out they do, and one arbitrary group takes every
contested pixel. Measured in central São Paulo before the fix: **the last 5% of features emitted
was 100% a single node**, against a true composition of 58% Catholic.

**Fix:** `tiles.py` shuffles each tile's feature list, with a fixed seed, immediately before
encoding. The visible dot at a pixel then becomes a uniform draw from the dots covering it, so the
zoomed-out picture is a representative sample of the zoomed-in one — which is the property a dot
map is for, and it was silently false at every zoom until now.

Verified on the built archive at z6, z8 and z10: the composition of the last 5% of each São Paulo
tile now matches the composition of the whole tile to within sampling noise.

**It costs 16% of the archive** — 28.5 MB to 33.1 MB for four countries. MVT delta-encodes
consecutive points, so ordering points spatially is what makes them cheap, and shuffling maximises
every delta. That is the price of the map being true, and it is worth paying; if it ever needs
clawing back, per-node random *draw keys* would preserve locality at the cost of a property per
feature and a runtime sort.

**CORRECTION 2026-09-04: 16% is the whole-archive average and it badly understates the cost
where it matters.** Re-encoding one z3 tile over India — 1,164,088 atomic features — under
different orderings, gzipped:

| | MB | vs as built |
|---|---|---|
| as built (shuffled) | 5.04 | — |
| sorted by position | 2.58 | **−49%** |
| extent 4096 → 1024 | 4.23 | −16% |
| drop `c` and `t` | 4.99 | −1% |

**Sorting halves a dense low-zoom tile.** The average is diluted by high-zoom tiles, where dots
do not overlap, deltas are large anyway and the shuffle changes little — but the low zooms are
exactly where the bytes and the overplotting both are. The shuffle is still right and still
worth paying for; the figure to plan against is ~50%, not 16%.

Two consequences. `c` and `t` are already near-free because MVT interns repeated values, so
there is nothing to win by dropping them. And the honest MVT-only saving available without
breaking anything is about 16% (extent 1024; 2 units/px is well past what a dot needs, while
extent 512 quantises to whole pixels and lattices the scatter when you zoom past a tile's
native level).

**§4.2d makes the whole cost disappear** rather than reducing it: in a flat binary buffer there
are no deltas for a shuffle to spoil, so the ordering is free.

**The general form, which will outlive this instance:** any time a renderer resolves overlap by
"last one wins", the drawing encodes whatever order the data happened to arrive in. If that order
correlates with a category — and sorted-by-id always does — the map is making a claim about
category that comes from the sort, not the world. Every future layer that overplots needs this
same shuffle.

### 4.2d The unmerged dots leave the tile pyramid — BUILT 2026-09-04

**The one sentence:** a tile pyramid is the right structure for data that THINS as you zoom out,
§4.2 forbids this data from thinning, so tiling the unmerged dots bought an elevenfold
duplication and MapLibre's per-feature cost and nothing else. They are now one flat binary
buffer per country per edition (`buffers.py`), drawn by a MapLibre custom layer in a single
instanced call.

**What it cost before.** Biggest tile at each zoom, feature counts:

| zoom | `dots` (merged) | `atomic` (separate) |
|---|---|---|
| 0 | 3,278 | **2,129,793** |
| 3 | 1,188 | **1,164,088** |
| 6 | 2,726 | 212,941 |
| 10 | 2,819 | 15,618 |

The merged pyramid is flat at ~2–3k marks per tile at every zoom, which is the pyramid working.
`atomic` is 650× that at z0 and was ~85% of a 152 MB archive; the z0 tile alone was 11 MB. And
`merged = false` is the default (§4.2b), so the expensive layer is the one readers land on.

**The diagnosis is not bytes.** MVT is already about 5 bytes per feature on the wire, so a
cleverer encoding was never going to be the answer. The 152 MB is the pyramid storing every dot
eleven times, and the *runtime* cost is per-feature JS:

- a circle is **4 vertices**, and the same dot is resident in every loaded tile at every loaded
  zoom;
- each tile carries a JS feature index for `queryRenderedFeatures`;
- `circle-color` is re-evaluated per feature on every recolour — the >120 ms that `pumpPaint`
  exists to ration;
- **`setFilter` is worse and was the hidden one.** Filters are applied when the worker populates
  a bucket, so changing one reparses every loaded tile. Every country switch, every scope
  change, every `unaffiliated` toggle paid seconds for it.

**What replaces them.** The buffer stores a node INDEX; a 512-entry palette texture maps index →
colour and visibility. So `circle-color` becomes the texture's RGB and `setFilter` becomes its
alpha, and the vertex shader collapses a hidden dot to a degenerate quad — which is §6's
"removed, not dimmed" enforced for free rather than by taking features out of a layer. Country
selection is not a filter at all any more: it is which buffers get drawn.

Measured in the viewer, 2.13M dots over sixteen countries: **recolour 0.28 ms, hover 0.25 ms**,
against a filter change that used to reparse every tile. `pumpPaint` still rations the merged
tile layer and no longer rations the scatter.

**Format** — struct-of-arrays, 10 bytes a dot: `x uint32`, `y uint32`, `ni uint16` (node index
in the low 14 bits, §7 tier in the top 2). India is 1,207,981 dots in 12.08 MB, **6.6 bytes a
dot gzipped**, against ~5 bytes × 11 zooms in MVT. All sixteen countries: 21.3 MB fine, 2.1 MB
coarse.

**uint32 fixed point, not float32, and the matrix matters more than the positions.** float32
mercator has an ulp of ~1.2 m — a quarter pixel at z14, four at z18. But the larger error is
that MapLibre's matrix must be downcast for `uniformMatrix4fv`, and that alone is several pixels
at z18 no matter how positions are stored. Both are fixed together: subtract a local origin in
exact integer arithmetic BEFORE anything is scaled, folding the origin into the matrix in
float64 on the CPU. Replaying both formulations through `Math.fround` against `map.project`:

| zoom | with the local origin | the obvious way |
|---|---|---|
| 14 | **0.0000 px** | 0.73 |
| 18 | **0.0000 px** | 9.74 |
| 22 | **0.0000 px** | 132.74 |

The obvious way degrades smoothly enough to look fine in testing and be wrong at street zoom.
**The origin must snap to a whole uint32 unit** (~1 cm), not to a coarser grid — snapping to
1/65536 of mercator instead left 1.24 px at z20, because the residual the trick exists to cancel
comes back scaled.

**Instanced quads, never `GL_POINTS`.** `ALIASED_POINT_SIZE_RANGE` maxes at 1024 on desktop
ANGLE and **63–64 on Mali and Adreno**, and this map's worst case is 8 px × 1.3 `DOT_GAIN` × 3
slider × 2.5 `COARSE_GAIN` × 2 DPR = 156 device px. Worse, GLES culls a point whose *centre*
leaves the clip volume, so large dots pop out at the screen edges. Desktop would have passed
this and phones would not.

**§4.2c is kept, and here it is free.** Dots are Hilbert-sorted, cut into buckets, shuffled
*within* each bucket, and the buckets are written in random order. Local uniformity is all
§4.2c actually asks for — dots only overlap locally — while global sorting is what buys
compression and viewport culling, which a globally shuffled list forbids. Each bucket carries a
bbox and the viewer draws one run per visible span.

**Culling pad must scale with zoom.** A pad expressed as a fraction of the world is ~800 km at
every zoom, so most of a country falsely intersects a street-level view: 433,837 dots submitted
for one Mumbai junction, against 12,288 once the pad is a few pixels' worth.

**Picking is now correct, and it was not before.** §4.2c makes the visible dot at a pixel the
LAST one drawn that covers it. `queryRenderedFeatures`'s `features[0]` is the FIRST in tile
order, so in any dense area the old tooltip could name a religion other than the one under the
cursor. Draw order here is buffer order, so the answer is the highest index among the dots whose
disc covers the cursor — exact, from a uniform grid, comparing in uint32 rather than projecting
each candidate (which is 0.03 ms against 4.7 ms).

**Editions do not mix.** The coarse edition loads first and complete — 2.1 MB paints every
country at 1:10,000 immediately, and §4.1b makes that an authored edition rather than an
approximation — then the fine edition replaces it **wholesale**, never country by country. The
US at 1:10,000 beside India at 1:1,000 would put two dot values in one view, which §4.1 forbids.

**What is NOT done.** Context loss re-upload is written (the typed arrays stay resident) but has
not been exercised. Nothing has been tested on a real mobile GPU. And `render(gl, matrix)` is
the v4 signature; a MapLibre v5 or globe-projection upgrade would need the shader ported to
`getProjectionData()`.

**Rejected on the way: a capped merge.** Extend §4.2a's merge with a cap on how many dots one
mark may absorb — `m = ceil(k / CAP(z))` marks of weight `w = k/m`, drawn at `r√w`, with
`CAP(z)` falling to 1 at high zoom so it becomes the plain scatter exactly where the scatter is
readable. It works, and it has the property the merge always had: thinning is driven by each
religion's OWN local density, so rare groups are never thinned. Measured on a z3 India tile at
cap 16, Hinduism keeps 6.3% of its dots and Bahá'í keeps 100%. Rejected because it is still
merging — marks grow above unit size in dense cells at low zoom — and Anita's preference is for
the plain scatter to stay a plain scatter (2026-09-04). Worth remembering that it exists: it is
a continuum between `dots` and `atomic` with one parameter, and it would have left the
elevenfold duplication and the per-feature wall exactly where they were.

### 4.3 Presence marks: a second grammar that carries no magnitude — DECIDED 2026-08-27

**Amended 2026-09-03: one ring per (country, religion), and only where the religion draws no
dot at all.** The section below stands as the reasoning for having a presence symbol; what
changed is how many of them there are and what each one claims. The old rule was one ring per
(area, religion), which came to 152,396 rings against 204,046 dots across three countries.

**Most of those rings were redundant, and §4.1a is why.** Once a node's national total is spread
across its units by largest remainder, the only people no dot represents are the final
`total % dot_value` — under one dot, **once, for the whole country**. A per-area ring was
therefore saying "this group is also here" about people who were already drawn a unit or two
away. Anita, 2026-09-03: *"since we're doing carry, from case 2 we should only be getting one
ring per country-religion."* That is exactly right and it falls straight out of the carry.

**So the rule is now:**

> A religion gets one ring in a country when it reaches no dot anywhere in that country. The
> ring is placed at its largest concentration and says only "here", never how many.

By construction such a religion has under `dot_value` adherents nationally, so the ring is a
true and bounded statement. A religion that draws even one dot gets no ring, because the dots
already say it is present.

**Uncounted rows now draw nothing at all** — also Anita's call, *"it's not a real number, so
let's not show anything."* A body that reports congregations and never reports membership is not
a small group, it is an **unmeasured** one: the US has 155 of them holding 27,433 congregations,
on the order of 13 million people, and the United Pentecostal Church International alone has
4,549 congregations with no adherent figure. Drawing "this is tiny" for them is the opposite of
true, which is the trap §4.3 was written to avoid and which the old code walked into anyway.
**§4.4's congregation-to-adherent conversion is the right answer and it is not built.** Until it
is, those bodies are absent rather than misdrawn, and that absence is the argument for building
§4.4 next.

**A bug this found, worth recording — 2026-09-03.** Canada has **644 census subdivisions that
publish no religion data whatsoever**; it arrives as a blank in all 147 categories. Those blanks
were taking the uncounted path and becoming 5,152 rings, each asserting that a religion was
present in a place whose source had said nothing at all. An absence of data drawn as a presence
of people, which is §3.5 exactly backwards. The tell was that all 14,877 US uncounted rings carry
congregations and all 5,152 Canadian ones carry zero: **a blank earns a symbol only when the
source counted something else that establishes presence.** The new rule drops the whole class,
but the general lesson outlives it — every future source needs its blanks classified as
"present, unquantified" or "not reported", and they are not the same thing.

**What the old rule cost, kept here because it is the evidence for the new one.** Rings per dot
ran 0.3 in the US, 1.1 in Canada and **8.1 in Czechia**, where Catholic alone had 5,804 rings
against 984 dots. That spread is a property of the instrument, not of the countries: a membership
roll reports nothing where nobody is on a roll, while a census reports a small non-zero for
nearly every category nearly everywhere and 1:1,000 puts almost all of it under the floor. Any
per-area presence symbol will degenerate the same way on any full-census source.

**The legend follows the map, not the data.** A node with rings and no dots leaves the tree while
rings are off and comes back with them. Czechia forced the rule: 21 of its 48 rows were ring-only,
so with rings hidden nearly half its legend was hollow swatches for marks that were not on screen.

**Rings remain off by default** (Anita, 2026-09-02), from when they were a blanket. That reason
has largely gone with the blanket, and the standing cost is that a religion too small to draw is
now absent from both the map and the legend unless the toggle is on.

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

**Built 2026-08-27, and the ring is the dominant symbol — by a lot.** Of roughly 80,680
(body, county) pairs in the US data:

| dot value | dots | rings | share of pairs that are rings |
|---|---|---|---|
| 1 per 100 | 1,576,707 | 32,617 | **40%** |
| 1 per 1,000 | 141,501 | 64,739 | **80%** |

At the dot value a global build can afford, **four out of five body-county pairs cannot be drawn
as a dot at all**. That is not a tail, it is the map. Two consequences:

1. **Ring gating is load-bearing, not polish.** 64,739 rings drawn at once at z4 out-ink the
   141,501 dots they are meant to sit beside, which inverts the entire point of having two
   symbols. **Decided 2026-08-27: rings are hidden until a group is selected**, and then only
   that group's are drawn. A zoom threshold was tried first and was still too noisy.

   Two implementation traps, both hit:

   - **A ring must not sit at the unit's centroid.** Placing every ring for a county at one
     representative point stacks dozens of them on a single coordinate — visually one ring, and a
     hover answers with whichever is first in the file. Rings are now placed in a random tract of
     the county, the same rule as dots, so they neither coincide nor claim a location the data
     does not have.
   - **Hiding must be a `filter`, not an opacity of 0.** A circle at zero opacity is still
     hit-tested, so invisible rings went on answering hovers meant for visible ones: with Sikhism
     selected, mousing over a Sikh ring reported "Wesleyan Church". Anything hidden for a reason
     the reader can see has to leave the query too.
2. **The finer the dot value, the more honest the map** — the 1:100 build has four times the
   dots and half the rings. This is the strongest argument yet for getting tiling working, since
   the dot value is bounded by what can be shipped, not by anything about religion.

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

**Amended by §6.2 (2026-09-02): everything below is scoped to one country. The taxonomy is shared, the legend and the palette are not.**

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

### 6.1 What building it changed — 2026-08-27

**Hue alone is not enough at overview width.** The overview draws about 40 categories, which is
9° of hue apart, and two 2px dots 9° apart are the same colour. Fixed by alternating *lightness
and saturation* between adjacent entries (odd entries light and desaturated, even entries dark
and saturated) so neighbours differ on three axes rather than one. Hue order still follows the
tree, so a family stays contiguous on the wheel.

**The overview cannot draw at depth 1.** Christianity is 95% of the US data, so a family-level
palette renders the country in one colour. Depth 2 is the working default and the depth control
is how you get back to depth 1 deliberately. Worth re-checking when a country lands whose data
is not 95% one family — the right default may be per-scope rather than global.

**In focus mode the panel must expand to the scope**, or the legend for what is on screen sits
hidden behind a collapsed triangle. The first build got this wrong and the fix is `isOpen()`:
open every ancestor of the scope, plus the scope's own subtree down to the drawn cut, and leave
everything outside collapsed.

**What it looks like when it works:** selecting Baptist isolates the Bible Belt and splits it into
Southern Baptist (16,571 dots) against the four National Baptist conventions (3,643), which is a
real and legible geography — the second is urban and Deep South where the first is everywhere.
And the overview alone reads Utah as Latter Day Saints, the upper Midwest as Lutheran, the
Northeast and Southwest as Catholic, without anyone being told to look.

**Still true from before:** sibling groups that are large and adjacent (Sunni/Shia,
Catholic/Protestant) need separation that survives a 2px dot, while distant tiny leaves can share a
shade because they never appear in the same view. And **the genealogy tree is the colour key** —
there is no other legend that holds 200 entries legibly, and showing descent and colour in one
object is what makes either learnable.

### 6.2 One legend per country — DECIDED 2026-09-02

§6 assumed one taxonomy and two colourings of it. That is right *within* a country and wrong
*between* them, and the reason is §2.3 and §3.9 rather than anything about colour: what differs
between sources is not mainly how big the categories are, it is **which categories exist at all**.
England and Wales publishes no Christian denomination and fifty minority write-ins. The US
publishes 372 bodies and cannot produce "no religion" at any granularity. Mexico files Orthodox
Christians under *otras religiones*, so the same religion sits in a different *place* in the tree.
A single palette stretched over the union of all of that gives every country a sparse handful of
the wheel, and spends its resolution on distinctions that country never drew.

So the viewer draws **one country at a time, with a legend and a palette built only from what that
country's source reports** — and the picker for it is the first control on the page, top left,
above the map's own title.

**Presence prunes the tree, and a ring counts as presence.** The panel holds a node only if that
country has a dot or a ring under it, plus its ancestors. Everything else is not greyed out, it is
gone: a branch a source never asked about is not a zero, and drawing it as one invites the reader
to conclude the group is absent rather than unmeasured (§3.5 is the same argument at the level of
a number). This is also what makes the depth control useful again — the wheel now divides among
categories that exist here, so depth 2 in Canada and depth 2 in the US are different cuts of
different trees rather than the same cut of one mostly-empty one.

**A hue therefore means different things in two countries.** — **REVERSED by §6.8 (2026-09-03).
The presence pruning below stands; the per-country palette does not.** Kept here because the
argument is the one §6.8 had to answer. Original text follows.

This is the §6 trade taken one step
further, and it is accepted for the same reason: the panel *is* the key, it recolours at the same
instant, and no view ever shows two palettes at once. What is bought is worth more than what is
lost — inside one country the wheel is spread over that country's real distinctions, which is the
question the map is for.

**The all-countries view survives as the entry point.** It draws every built country on the shared
upper taxonomy and carries the comparability warning (§3.1) in the about panel, listing each
country's basis. It is the honest version of "the whole map", and it is explicitly the *less*
informative one: it can only draw what the sources share.

**Two ways to select a country, because they answer different questions.** The picker, for "show
me Ireland"; and the camera itself, for "I have zoomed into Canada, stop showing me a legend built
for two countries". The camera reads the *source* rather than the rendered layers, so it can still
see a country whose dots the current filter is hiding; otherwise selecting Canada would make the
US invisible to the detector and panning south could never switch back.

There was a third — clicking a dot named its country — and it is **removed, 2026-09-03**. It made
every attempt to look at a dot a change of view, and on a touch screen it was worse than that: a
tap is the only way to raise the hover card, so wanting to know what a dot is and wanting to
rebuild the whole legend were the same gesture. The card already names the country. A single
click on a dot now does nothing at all.

**The camera one is a MODE, not a behaviour — amended 2026-09-03.** It was originally just
something that happened while you panned, and Anita's objection was that this is confusing: the
legend changes and nothing tells you why or how to stop it. So the picker's first entry is **Auto**,
it is the default, and it is the only state in which the camera may change anything:

```
Auto                follows the map      <- default; button reads "Auto (United States)" once resolved
All countries              395m
Brazil                     190m
Canada                      36m
...
```

Two properties make it legible. The button reads **"Auto (United States)"**, which says both that
the map is choosing and what it chose, so a change of legend is never unattributed. And picking
anything else — a country, *or* All countries — leaves Auto for good: nothing moves the selection
again until Auto is picked back.

**Auto reverts to all countries when you zoom out**, which is the half that was missing: without
it, zooming out from Canada to the hemisphere left a Canadian legend over a continental view.

**The threshold is a fill, not a zoom — amended 2026-09-03.** It was two fixed zooms, 3.8 and 4.2,
and those were read off the United States: 4.2 frames it. In Europe the same number frames a third
of the continent, so Auto was handing the legend to whichever built country happened to dominate a
view that was not about any one country. The zoom at which a country becomes the subject is one
number per country, so the rule now measures **fill** — how much of the screen the country's own
view box spans, along whichever axis it presses hardest against. 1 is a country touching two
opposite edges; it is the same quantity `fitBounds` works to, so a country flown to lands near it.

**Three bands, and the middle one is the point.** Under a fill of 0.6 the country on screen has
shrunk into a wider view that is no longer about it, so Auto goes back to all countries. At 0.85 a
country takes the legend if it also holds ≥ 85% of the dots in the middle half of the screen.
**Between 0.6 and 0.85 nothing changes at all** — without that dead band a single wheel notch
across one threshold would swap the legend back and forth. Reluctance is deliberate throughout:
changing the legend out from under someone is worse than being slow to change it.

On a 1920×950 canvas that puts the United States within a tenth of a zoom level of where it was
(engages 4.11, reverts below 3.61) and moves Europe to where it should have been all along:
Poland 5.84/5.34, Romania 6.33/5.83, Estonia 6.94/6.44, Czechia 7.24/6.74. A hard floor of zoom 3
sits under all of it, so nothing wins the legend at hemisphere range however large it is. Because
the measure is relative to the viewport, the thresholds also follow the window instead of assuming
one size, which the fixed zooms did not — the old 4.2 was a filled screen at 1600×900 and a
two-thirds-filled one at 2560×1300.

**And a second test, because fill says nothing about WHERE — amended 2026-09-03.** Anita, on the
fill rule: "I can focus on the UK and then drag away and have no UK on the screen at all but it
still doesn't reset." Fill is a question about scale, and dragging east does not change it by a
hair. Nor can the dots object, and this is the part that made it permanent: a country nobody has
built has no dots to object with, so over Germany the tally comes back empty, hits its
`total < 150` bail, and returns before anything is reconsidered. The legend read "United Kingdom"
over Munich and there was no camera move that could clear it.

So **overlap** is the other half: the country's box against the middle of the screen, as a share
of the most the two could overlap at that zoom. 1 is dead on it — equally the whole country in
the middle of the screen, the middle of the screen deep inside the country, and a country too
tall for the screen filling the middle of it, all of which are the same answer to "am I over it".
Engages at 0.5, reverts under 0.25, and the revert is checked **before** the tally, which is what
fixes the stuck case. A country has to pass both tests and the 85% dot share to take the legend.

**Both halves of it are load-bearing, and so is the middle-of-the-screen part.** Measured over
the whole viewport, overlap cannot separate Britain from Germany at a zoom wide enough to hold
the United Kingdom — at that width the viewport is 37° across and most of Britain is genuinely
still on screen from Munich: 0.53 against 0.84, which no threshold divides. Over the middle half
— the same `AUTO_BOX` the tally uses, and for the same reason — it is 0.08 against 1.00.

**Double-click a dot to select its religion — added 2026-09-03.** The gesture the dot click
vacated goes to the question a dot actually raises. It selects the node the dot carries, which is
exactly what the hover card just said: a Latin Catholic dot selects Latin Catholics, and a
"Christianity (unspecified)" dot — which carries `christianity` itself, not a child of it —
selects Christianity whole. Over empty map the double-click still zooms; the handler only takes
the gesture when it hits a dot, and MapLibre's `clickZoom` yields to a prevented `dblclick`
because `mapEvent` sits ahead of it in the handler chain.

**"Exactly what the card just said" is a constraint, not a description, and it was broken on the
first cut.** The card reads `e.features[0]` from a layer-delegated `mousemove`, which queries the
single pixel under the pointer; the double-click ran its own 8-pixel box query and took the first
feature in it. Over crowded dots those are different features, so the card could say Baptist while
the selection came out Latin Catholic — a double-click that names something you did not click on.
They are now registered as a pair in one loop and read the same expression, which is the only form
of this that stays true: any second way of deciding what is under the pointer will drift from the
first.

**Escape backs out of one thing at a time, innermost first**: the swatch picker, then the country
menu, then the about panel, then the religion selection. Never two on one key — clearing a
religion the reader could not see they still had, because a panel was covering it, is the failure
this ordering exists to avoid.

**Selecting a country moves the camera only if the camera is not already showing it** — over 80%
of the country's view box on screen, and a viewport under 1.8× its area. Both halves are needed:
the continental view contains all of the United States and is still not a view *of* the United
States.

**Three things this forced downstream.**

- Every tile feature carries `c`, its country, and **the merge groups by it**. That reverses the
  original reason for putting several countries in one archive — "so the merge works across a
  border rather than stopping at it" — and it has to, because a mark merged across the 49th
  parallel carries a count belonging to neither side of it. The archive stays shared because the
  tile pyramid is shared, not because the marks are.
- `counts.json` is per country: name, source, basis, public note, bbox, and the dot and ring
  tallies. The viewer cannot count anything itself (§9), so *which nodes a country has* is a
  build-time fact like every other total.
- A country needs a **`view` box that is not its data box**. Fitting the US to its dots spans
  Hawaii to Maine and shows the reader an ocean. `countries.py` carries an optional override and
  everything else defaults to the data extent.

**"No religion" was hidden by default and is not any more — DECIDED 2026-09-03, Anita's call.**
It was hidden for two reasons and the paragraph that used to stand here said so: comparability
across countries, which is airtight, and "it is a single category that can be a third of the map
and crowds out the rest of the legend", which is a design preference. The first reason argues for
the *warning*, not for the hiding — it is in the about panel, which lists every country's basis,
and hiding the category does not make the other categories more comparable. The second is now the
reader's to make: §6.11 lets any group be hidden from its swatch, and this one keeps its own
control in the legend besides. **Hiding a group by default is a claim that it is not part of the
picture, and it is.** The mechanism stays — `HIDDEN_DEFAULT` in `index.html` — and is empty.

One thing to watch, and it is §6.3's doing rather than this decision's: `unaffiliated` is authored
*muted* on purpose (contrast 2.6, below the 3.0 the rest clear) precisely because it is 81% of the
Czech map. Shown by default, that mutedness is now working harder than it was — it is the only
thing stopping the largest category on several maps from drowning the ones the map is for, and it
is also the reason those dots are hard to see at all. If it ever needs to read more clearly, the
lever is its lightness and not its saturation.

### 6.3 The family palette is authored, and the depth cut is now flat — DECIDED 2026-09-03

Two things §6 got wrong in the build, both found by looking at the US at depth 2.

**1. The top level was generated, and it should never have been.** §6 says the overview palette is
"hue by top-level family … stable, hand-tunable, memorable: orange is always Islam", and then the
build divided the wheel among the roots *that country reports*. So Christianity was hue 0 in the US
and hue 33 in Czechia, and Islam's colour depended on how many families its neighbours in the list
happened to be. A family colour that moves between countries is not a colour anyone can learn,
which was the entire argument for having an overview palette separate from the focus one.

`ROOT_HSL` in `index.html` now names all thirty roots. Nine are named outright — Christianity
yellow, Islam green, Judaism blue, Hinduism orange, Buddhism red, Sikhism red-orange, Shinto pink,
Bahá'í and Zoroastrianism blue-green. The remaining twenty-one take an indigo→magenta wedge.

**The wedge is the honest limit of the idea and it is worth writing down why.** Twenty-one groups
over 90° is 4° apart, and two 2px dots 4° apart are the same colour. They are therefore also cycled
through three lightness/saturation tiers — *period three, not two*: with a two-step cycle every
OTHER neighbour comes out identical in S and L and 8° apart in hue, which measures as ΔE 3 and is
no separation at all. That mistake was made and caught by measurement, not by eye.

Order inside the wedge is chosen against the data rather than by theme: the groups that co-occur
**at size** in one country are the ones spaced apart. Brazil carries five of them at once
(Spiritism 3.9m, Afro-diasporic 588k, Japanese new 155k, esoteric 74k, indigenous 63k), so those
five set the spacing and the rest fill in between.

**What is measured, and what is accepted.** `tools/check_palette.py` reads `ROOT_HSL` back out of
`index.html` — not a copy of it, so it cannot drift — and checks every pair of roots that co-occurs
in a built country, in CIE Lab, against `counts.json`. In the US and Czechia every pair over 20 dots
is at least ΔE 25 apart. What remains:

| pair | ΔE | where | why accepted |
|---|---|---|---|
| Japanese new / indigenous | 13 | br, world | 155k and 180k against a 190m country — specks, never adjacent areas of colour |
| indigenous / pagan | 13 | ca, world | 180k and 82k |
| esoteric / pagan | 15 | world | 94k and 82k |
| the not-a-religion family | 20–23 | everywhere | superseded by §6.3a, which re-authored all five as one ramp: `secular`/`other` and `unaffiliated`/`unchurched` are the tightest at ΔE 19.9, and the reasoning for accepting that is there rather than here |
| Hinduism / Sikhism | 26 | ca | inherent: four of the nine named families are warm and sit inside 60°. Pushed apart in lightness as far as "still vivid" allows |

**The warm end was re-authored on 2026-09-03 and this row is now 31**, not because the crowding
eased but because Sikhism stopped being chosen by eye. Hinduism moved to #fa7420 at Anita's
request, which took the arc between Buddhism and Hinduism from 40° to 27° and made the row above
unfixable by nudging; §6.9 has the search that replaced it, and the four hues that moved together.

### 6.3a Not-a-religion is one grey ramp — DECIDED 2026-09-04

Five roots are answers *about not belonging* rather than traditions, and they are now drawn as one
neutral ramp separated by lightness alone. The membership is the whole list, and anything added
must be an answer of the same kind:

| | | what it is |
|---|---|---|
| `unaffiliated` | No religion | the largest single answer in Canada, the US and Czechia |
| `secular` | Secular and ethical | a stated non-theistic **position** — Ethical Culture, and the Atheist / Agnostic / Humanist answers Canada and Pew both collect. Not the same as `unaffiliated`, which is the absence of one |
| `unchurched` | Believing, no church | reports religious belief **and** explicitly no institution. Czechia's 960,201; Pew's "spiritual but not religious" |
| `other.<source>` | Other, by source | the per-source residual containers of §3.11. The greyest of the five, being the one that says nothing |
| `parody` | Parody and protest answers | Jedi, Pastafarian — a protest, not a belief |

#### 6.3a-i A sixth member, and it is a different kind of thing — ADDED 2026-09-04 with Germany

`unrecorded`, "Religion not recorded". Germany's *Sonstige, keine, ohne Angabe* — 42.8M
people, **51.8% of the country**, and the largest node any single country contributes after the
US (§3.9a).

The five above are all **answers**: somebody was asked and said something, including saying no.
This one is the residual of a source that **never asked** — an administrative register, read for
church tax, which holds no religious body for these people. Every existing home asserts something
false: `unaffiliated` is a person reporting no religion; `other.<source>` is a religion the source
named and could not be placed, and here the source named nothing; `unchurched` is a positive report
of belief without institution.

**The test that admits it, and it is the test for anything added next: this category's composition
is a property of the INSTRUMENT rather than of the people in it.** Germany's bucket holds Muslims,
Orthodox Christians, Jews, Freikirchen and the wholly irreligious together, and which of them are
in it is decided by German tax law. Any register-basis source of the same shape belongs here;
Austria is the obvious next one.

**The label had to change, and the legend could not hold the reason — Anita, 2026-09-04.** It was
"No religious body recorded", which is accurate and still lets a reader conclude that 43 million
Germans are irreligious. The obvious fix — "…(includes Islam, Orthodox, etc.)" — cannot be done in
the label, because `.row .lb` is `nowrap` with `text-overflow: ellipsis`: a label long enough to
carry a caveat is one that gets **cut off mid-caveat**, which is worse than saying nothing.

So the two halves were split. The visible label is **"Religion not recorded"** — shorter than what
it replaced, and grammatically unable to be read as a statement about belief, because the subject
is the record. The caveat goes in the row's **tooltip**, which every legend row already had and
which until now repeated the label it was truncating.

That needed a new field, and it is the general one this project was missing: `branches.py` has
carried a `note` since the first tree and **nothing in the viewer has ever shown it** — it is for
whoever maintains the taxonomy. `PUBLIC_NOTE` is the other kind, written for a reader, and
`build_tree.py` carries it into `religions.json` as `public_note`. **Only add a node when the label
alone would mislead**: "Lutheran" means Lutheran and needs nothing. The test is whether a reader
who reads the label and stops comes away believing something false.

It remains a global label rather than a German one, and the tooltip names the groups generically
for the same reason — the node is not Germany's, it is any register country's, and Austria's bucket
will hold a different mix. A country-specific version belongs in `note_public`, which the about
panel already renders and which already leads with the four million Muslims.

**It takes `other`'s hue at the ramp's darkest lightness — `hsl(228, 10, 37)`, `#555968`** — and
both halves are the rules below applied rather than bent. By the ordering principle it reports less
than any of the five, not even "no religion". By the prominence rule it must be quiet, and harder
than `unaffiliated` must, because a pale mass of 42.8M dots would drown the Catholic/Protestant
signal that is the only real information Germany carries.

Measured with `check_palette.py` before it was written down: **dE 21.1 to `unaffiliated`**, its
nearest neighbour here, which is *wider* than this family's existing tightest pair (`secular` /
`other`, 19.9) — so a sixth member does not squeeze the ramp. dE 51.1 to the nearest religion.
Contrast 2.70, below the 2.9 floor, and in `DIM_ON_PURPOSE` for exactly `unaffiliated`'s reason.
`check_overview.py` now reads that same list rather than keeping its own.

**It is not green, and the search wanted green.** `hsl(120,30,43)` scores dE 41.8 inside the
family, twice as well. Rejected on meaning rather than measurement: Islam is at hue 138, and
colouring the one bucket that hides Germany's Muslims a muted green is the worst available accident
(§14.2). **A palette search optimises separation and cannot see what a colour would say.**

**Anita, 2026-09-04: these should look like what they are.** `secular` was authored at saturation
72 and `parody` at 82 — more vivid than most religions, for categories that are the absence of one.

**Lightness is prominence, and the map is dark — Anita, 2026-09-04.** A light dot on a near-black
background is the loudest mark available, so the ramp runs the opposite way to the obvious one:
the thing we least want shouting is `unaffiliated`, which is the largest node on the map, so it is
the **darkest**. What falls out is a reading and not only an aesthetic — the ramp ends up ordering
the five by how much religion is being reported:

| | L | |
|---|---|---|
| `unaffiliated` | 41 | no religion at all |
| `parody` | 46 | a joke, not a belief |
| `secular` | 50 | a position, and not a religious one |
| `unchurched` | 58 | a belief, held outside any institution |
| `other` | 64 | an actual religion, which the source did not name |

`unaffiliated` is **deliberately below the contrast floor** — 2.59 against the 2.9 the checker
wants — and `check_palette.py` now exempts it by name in `DIM_ON_PURPOSE`, because being quiet is
the whole point of the colour and reporting it as too faint invites undoing the decision. `secular`
is blue and `unchurched` purple by request, and both hues do real work: with two members down at
the dark end, hue is the only axis left to separate them by.

**The cost is real and is accepted.** ΔE 25 is not reachable inside a family this narrow; the ramp
gets **19.9** at its tightest. That is accepted rather than tolerated, because the ΔE 25 rule
exists so two **different religions** are never confused, and inside this family a mix-up is
between neighbours on one spectrum — reading "secular" as "no religion" is a small error where
reading either as Islam is not. Every member clears **ΔE 27.7** from the nearest religion, which is
the threshold that actually matters here.

**The children land inside the ramp too, and that is not fixable by moving the parent.**
`other` is a container; what actually draws is `other.us`, `other.cz` and the rest, and the
viewer lightens a child off its parent. `other.us` renders `#acaeb9` — ΔE 9 to 10 from three of
the other four. The band is 44–78 and the family plus its children do not fit in it with room to
spare, so there is nowhere to move `other` to that does not collide with something else. It is
accepted on the same argument and one more: `other.us` is 1,124 dots against `unaffiliated`'s
60,554, so the confusion is 54-to-1 in favour of reading the small one as the large one, and both
mean "not classified" in any case. `tools/check_palette.py` compares roots and cannot see this
class of collision at all — worth knowing before trusting a clean run.

**`parody` is the one warm grey**, and that is what separates it: at lightness 46, with four cool
neutrals around it, cool-vs-warm is the axis left — and it happens to suit the one member of the
five that is a joke rather than a position.

**Re-run the checker after every re-tile.** `counts.json` is rewritten by each `tiles.py` run and
holds only the countries that ran, so which pairs are close is a property of what is in the archive
today. Australia, Ireland and Mexico landing on 2026-09-03 moved every number in that table, and
brought three roots — Mandaeism, Yazidism, Cao Đài — that had no authored colour at all. The
checker named them; that is what its first section is for.

Adding those three took the wedge from 16 entries to 19 and its step from 4° to 3.3°, which is why
the small-group collisions above are what they are. **The wedge is the part of this that does not
scale**, and it will keep not scaling as countries land.

Two colours at the blue-violet end of the wedge — Daoism and the Unification Church — sit at
contrast 2.3 and 2.8 against the map background, under the 3.0 the rest clear, because blue carries
little luminance at a given lightness. Both are ring-only in every built country (5 dots and 0), so
lifting them at the cost of breaking the tier rhythm against their neighbours has not been worth
it. It becomes worth it the day a source reports either at size.

The remedy if any of this starts to matter is not more tuning — 21 groups do not fit in 90° — it is
to promote a group out of the wedge into a named hue. **Spiritism is the candidate**: 3.9m people in
Brazil, larger there than Judaism, Buddhism and Hinduism combined are in the US, and it is sitting
in the overflow bucket because the brief that set the nine named families was written against
US/Canada data.

Muted on purpose: "No religion", "Believing, no church" and the source catch-all. In Czechia they
are 81% of the dots and in Brazil and Canada the largest thing after Christianity, so a vivid
colour there drowns the groups the map is for. The catch-all is neutral grey rather than a hue,
because it is not a family — it is the source declining to name one.

**A new root with no line in `ROOT_HSL` falls back to a generated wheel position, which will look
fine and mean nothing.** Add a line when a country brings a new family in. This has already
happened twice and the checker caught both: **Alevism and Ravidassia** arrived with the UK and had
no authored colour until 2026-09-03, so Alevism took a fallback that measured dE 15.7 from Paganism
and 20.9 from Jainism in the one country where all three are drawn. Both are now placed in free hue
rather than squeezed into the wedge — Ravidassia in the 16→35 gap between Sikhism and Hinduism,
Alevism in the 168→190 gap between Bahá'í and Zoroastrianism. Each sits beside a tradition whose
boundary with it is the contested thing, which is either a useful statement or an unwanted one;
`branches.py` records that both groups reject the filing, and if the adjacency reads as a claim
they belong in the wedge with Druze and the wedge needs re-spacing. Alevism was tried at 152, next
to Islam, and moved: it measured dE 17.7 from Islam in the UK, where Islam is 3,999 dots and
Alevism 25, and a speck that reads as a shade of the largest thing beside it is worse off than
uncoloured.

**2. Almost none of the map's colours were in the legend.** The palette shaded *within* each drawn
node so a branch's children stayed separable. That sounds harmless. Measured against the tallies it
was not: at depth 2 in the US, **93% of the dots on screen were a shade of something whose own
legend row sat a level further down behind a collapsed triangle** (98% at depth 1; 90% in Canada;
52% even inside a Christianity selection). The map was mostly colours that appeared nowhere in the
key. The visible case was the Catholic Church — 61,858 dots, over a third of the country, in a
washed-out red that the "Latin Catholic" swatch above it did not match.

This contradicted §6's own premise. The genealogy tree *is* the colour key; a shade with no row in
the tree is a colour with no key. So the rule is now flat: **a node below the drawn cut takes its
drawn ancestor's colour exactly.** Every colour on the map has a swatch in the panel, at the depth
you are looking at, and `+` is how you tell a branch's children apart. Expanding a triangle without
raising the depth now shows several rows sharing one swatch, which is the true statement — they are
one colour on the map at this depth.

**3. Selecting a single sect no longer repaints it.** Dividing a hue wheel among a set of one puts
that one on hue 0, so selecting Islam in the US — which ASARB reports as a single category with
nothing below it — turned it from green to red for no reason a reader could see. When the drawn set
has one member the palette does not change at all: the node keeps the colour it had in the view the
selection was made from, and selecting it only dims everything else.

This inherits from whatever the unscoped palette says at the current depth, so it tracks the middle
level automatically. **Consequence while the middle level is still generated:** Islam selected from
the depth-1 view is green, and from depth 2 or 3 it is whatever wheel position the generated
palette gave it. That resolves itself when §6.4 does.

### 6.4 The middle level — ANSWERED by §6.9, 2026-09-03

Depth 1 is authored (§6.3) and focus mode is the §6 wheel. The level between them — every branch of
every family drawn at once, which is the working default — is still the generated wheel over the
drawn categories, so Christianity's branches run red → orange → yellow → green and Islam is
wherever its index falls.

The tension is stated in §6 and is not resolved by either answer alone. Hue-by-family keeps Islam
green but gives Christianity's twenty US branches twenty shades of yellow, which is the ancestrydots
complaint that started all of this. The full wheel separates the branches and throws away every
family colour the reader just learned at depth 1.

One further constraint the build has since produced: the authored hues are **thirty-three
families**, and the middle level is **~30 branches of one of them** — so the two cannot be the same
table, and a hybrid that reserves the nine named hues leaves the generated wheel running straight
through green, teal and magenta.

*(The second constraint that stood here — a node above the cut drawing grey — is closed by §6.6.)*

**The answer is the first horn, taken deliberately: hue by family, and the branches inside it do
not separate.** §6.9 builds it, and the thing that made it affordable is that the second horn is
still available one click away — the full wheel is what a *selection* draws. The paragraph above
frames this as a choice between two palettes, and the resolution is that it was never a choice
between two palettes; it is a choice about which one is the default.

### 6.5 Colour follows descent, not size — DECIDED 2026-09-03

The wheel was divided in **size order**, because that is how the panel sorted a parent's children.
That has one specific and bad consequence: the two largest bodies are always adjacent on the wheel,
and they are exactly the two a reader most needs to tell apart. In the US, Catholic (62m) and
Baptist (24m) came out 18° apart, red and orange.

So `branches.py` gains **`LINEAGE`** — the second of §2.1's two relations, in the smallest form
that is useful. Not the full `from` DAG with dates and edge kinds; §10 still owes that. This is one
thing: for a parent with many children, the order they descend in, cut into named groups. Six for
Christianity — ancient communions, Reformation, separatist and believers' churches, Pietist and
Wesleyan revival, restorationist and adventist, and the bodies that are on no single line — plus
Judaism's three and Buddhism's one.

`buildTree` sorts by that where it exists and by size everywhere else, and because `drawnSet` walks
`KIDS` in order, **the same sort decides the panel order and the hue order**. That coupling is the
point: the panel reads down in the order the colours run. Catholic and Baptist are now 144° apart,
and not by luck — a body is usually large *because* it is its own tradition, so descent order tends
to separate the big ones on its own. Measured, the closest pair among the six largest under US
Christianity went from ΔE 18 to ΔE 27.

**The generated wheel's tiers went from two to three at the same time**, and for the reason §6.3
already found in the authored wedge: with a two-step lightness cycle, every entry and the one two
places along are identical in S and L and separated by hue alone. That had Latter Day Saints and
Pentecostal — 6.5m and 6.0m — at ΔE 17.

**The panel captions the groups**, which is what makes the order legible rather than merely
non-arbitrary; without a caption a reader sees a list that is not sorted by size and is not told
what it is sorted by, which is worse than sorting by size. A caption is **not a node**: it cannot
be selected, has no count, and is not a level of the tree — see `christianity.protestant` in
`branches.py` for why there is no Protestant super-node, and this is not one either. A parent whose
present children fall in one group gets no caption; Buddhism's three vehicles do not need to be
told they are all vehicles.

**Captions appear only under the node you have selected** — amended 2026-09-03, from looking at
it. In the all-religions view they sat inside Christianity's children while Islam and Judaism,
Christianity's *siblings*, sat below them at the same indent, so "ANCIENT COMMUNIONS" read as a
bigger division than Christianity itself. A caption is legible only when the thing it divides is
the thing you are looking at.

**`build_tree.py` fails if a child of a lineage-carrying parent has no group.** Without that check,
adding a branch drops it silently to the end of the panel and the end of the wheel — still drawn,
still the right size, just quietly not where it descends. That is the §2.4 class of failure this
file exists to make loud.

**What a linear order cannot carry.** Baptists come out of English Separatism with Dutch Mennonite
contact; Methodism comes out of Anglicanism by way of the Moravians. Each is placed on its main
line with the other edge written down as a comment. That is the honest limit of a list, and it is
the argument for §10's DAG rather than a replacement for it.

### 6.6 A branch that carries dots is a category — DECIDED 2026-09-03

A branch whose children are drawn sat *above* the cut, so it got no hue and fell through to grey.
That is fine when the branch is an empty container and wrong when the source counted people on it,
which happens whenever a source names a branch and no body below it. It is not an edge case:

| | dots on a branch | share of that map |
|---|---|---|
| United States | 0 | ASARB files every figure on a leaf |
| Czechia | 306 | 4% |
| Canada | 6,156 | **18%** — StatCan's "Christian, n.o.s." |
| all countries | 180,739 | **46%** |

The all-countries figure is that large because of §2.5: Canada, Czechia and Brazil file Roman
Catholics on `…catholic.latin` while the US files them on a leaf below it, so in the shared view
every non-US Catholic was grey. Anita's other example was the same shape — Independent Catholic
carrying 563k of its own against a 2k child that had a real colour and looked like it owned the
branch.

So **a branch with dots of its own is a drawn category** and takes a hue beside its children.
`drawnSet` pushes it ahead of them, which is what makes the palette come out right: the parent
paints its whole subtree, each drawn child paints over its own part, and what is left holding the
parent's colour is exactly the parent.

The panel then has to say which dots that colour marks, and it is not the branch's total —
"Christianity 19m" beside a swatch that marks 6.2m of it would be a worse lie than the grey it
replaced. So a split branch keeps a grey container swatch and its own dots get **a row of their
own, labelled `unspecified`**, carrying the colour and the branch's own count. That row is §3.2's
`…, other or unspecified` residual, read out of the tallies the viewer already has rather than
materialised by the `reconcile.py` that does not exist. It is not a node and cannot be selected —
there is nothing in the tiles to select, those dots carry the parent's id.

The invariant this buys, and it is checkable: **no dot on the map is grey.** Grey is the container
colour, and a dot in it is a religion with no key.

### 6.8 One palette for every country — DECIDED 2026-09-03, and it reverses §6.2

§6.2 divided the wheel among the categories **present in the country on screen**, and accepted that
"a hue means different things in two countries" because the panel is the key and recolours with the
map. Anita's call, and the argument that settles it: **a palette nobody can learn cannot be
hand-corrected either.** §6 always planned hand overrides for the scopes people actually select;
an override that only holds in one country is not an override, it is a per-country table with no
end. Stability is what makes the hand-tuning §6.3 started possible at all.

So **every node has one colour, computed once from the whole taxonomy.** Which divisions a country
*shows* is still per country — §6.2's presence pruning stands, and it is the half of §6.2 that was
about honesty rather than about colour. Only the colours stop moving.

Allocation, per level, in an order that has nothing to do with any country — descent where
`branches.py` writes a `LINEAGE`, file order otherwise. Each parent's children go at
`phase + k/n` around the wheel, `phase` being a golden-angle offset per parent. Two simpler
orderings were built and measured first, and both fail:

| ordering | parents whose six largest children include a pair under ΔE 25 |
|---|---|
| contiguous — each family in one arc, siblings adjacent | 31 of 37 |
| interleaved — all first children, then all second children | 11 of 37 |
| **spread with a per-parent phase** | **9 of 37** |

Contiguous fails because siblings are exactly what a reader compares, and 126 level-2 nodes is
2.9° apart — it took Catholic's children to ΔE 12 and Lutheran's to ΔE 9. Interleaving fixes that
until it runs out of parents: past the index where only the largest family still has children, its
remaining ones are adjacent again, which put twenty of Christianity's branches in one run of
greens. The phase matters as much as the spread — without it every family starts at hue 0, and the
fifteen-odd families with a single child pile into one arc.

**What it costs, and it is not small.** §6's focus palette is gone: selecting Baptist no longer
spreads the whole wheel over its nine children, it shows the nine colours they already had. And
Christianity has 29 of the 55 level-1 categories, so its branches can be at best **12.4° apart**
against the 18° a per-country wheel gave them in the US. That is arithmetic, not tuning — five of
its branches sit in a run of greens and no allocation fixes it while they share one wheel. The
remedy is the one the whole change is for: `PIN` in `index.html`, keyed by node id, hand-set and
now worth writing because it stays true.

### 6.7 Out of scope is hidden, not dimmed — DECIDED 2026-09-03

§6 said everything outside the selection "drops to near-black", and at 0.18 opacity it did. But
there is one dot layer, so the dimmed dots draw in feature order, which means about half of them
draw *on top of* the selection. Select Buddhism over New York and the 549 Mahayana dots sit under a
grey wash of Catholics. Opacity cannot fix that at any value; only taking them out of the layer
can, so the scope is now part of the layer filter. What is lost is the silhouette of the country,
and the basemap already carries that.

### 6.9 Two palettes again, split by scope and not by country — DECIDED 2026-09-03, and it amends §6.8

§6.8 froze every node's colour so that panning from Canada to the United States could not repaint
anything. That was right and it stands. But it froze the colour across **scopes** at the same time,
and that half was an overreach: the all-religions view then spent the whole wheel on Christianity's
27 branches, because Christianity holds 29 of the 55 level-1 categories and the allocator has no
way to know that the other 26 belong to families a reader is trying to *find*. Five Christian
branches came out green, beside Islam.

**The two properties are separable and only one of them was ever the problem.** A colour that moves
when you pan is unlearnable and cannot be hand-corrected, which is §6.8's argument entire. A colour
that changes when you deliberately select a family is §6's original trade, taken knowingly, with
the panel repainting in the same instant so the key on screen is never stale. So:

| view | palette | what it is for |
|---|---|---|
| nothing selected | **overview** — every family inside a band around its authored hue | which family is this dot |
| a subtree selected | **focus** — §6.8's whole-taxonomy wheel, unchanged | which branch is this dot |

Both are computed once from the whole taxonomy. Neither moves with the country. §6.2's presence
pruning still decides which rows a country *shows*.

**The band is authored, in `ROOT_BAND`, for the same reason `ROOT_HSL` is** — a width derived from
how many branches a family has would move the day a source landed. Only three families have an
entry. That is not an oversight: **24 of the 30 families draw exactly one row at depth 2**, so they
take their own hue and nothing else, and the hue budget is far less contended than the taxonomy
makes it look.

**Two rules the build produced, both from a measured failure:**

- **A band must not contain its own root's hue.** A split family draws a row for its own dots at
  the root colour (§6.6's `unspecified`, and that row is 30% of Judaism and 60% of Buddhism
  worldwide), so a branch landing on the root hue cannot be told from the single largest thing
  beside it. Judaism's band runs below 214 and Buddhism's below 356 for that reason.
- **Nothing in the indigo→magenta wedge gets a band at all.** The wedge is already at 3.3° per
  family (§6.3), so a band even ±3° wide reaches its neighbours: giving Afro-diasporic one put
  Umbanda at dE 9.7 from Jainism, a different family. Wedge families fall through to a ±2° default
  and separate their children by lightness alone, which is the right answer where each has two or
  three rows and none is ever a large area of colour.

**Christianity's band stops at 98, short of green, and that costs it 12° it could have had.** At
110 the last two lineage groups came out green, so nine Christian branches — Non-denominational at
21m and Protestant-unspecified at 19m among them — sat in the legend as green dots above Islam's
green dot. Every one of those pairs measured over dE 25 and the reading was still wrong: what a
family palette sells is that **green means Islam**, and nine greens above it cancel that whether or
not any single pair is separable.

**And since 2026-09-03 it is two arcs, 30→44 and 56→98, with its own hue 50 in the gap.** A band
had to sit entirely on one side of its root, by the rule above; that put its floor at 52 and left
it 46°. Moving Hinduism to #fa7420 (below) opened the 23→50 arc, and taking it means straddling
the family's own hue — so a band is now a list of arcs and positions are allocated over them laid
end to end. 56° instead of 46°. Which is the truer picture anyway: a family's own shade belongs
among its branches rather than off one end of them.

**That expansion is for the look and not for legibility**, and the table above is why — widening
does almost nothing, and 46°→56° is well inside the range where it does nothing. What it buys is
that the run now *starts* in orange beside Hinduism instead of in yellow, which is the thing a
reader actually sees. It does not buy Catholic against Orthodox and nothing will.

**Moving one family moved four.** Hinduism 35 → 23 landed on top of Ravidassia at 25 and left
Sikhism 7° away, and the whole warm end had to be re-authored around it, because Buddhism 356 and
Hinduism 23 leave Sikhism **27° of arc** with both neighbours drawn at size wherever it appears —
Canada, the UK, New Zealand, Australia. Hue cannot separate three families at that spacing, so
Sikhism is the one colour in the table that is **searched rather than chosen**: the most saturated
value in the arc clearing dE 28 from both. It comes out `hsl(7, 100%, 70%)`, a coral — lighter
than a "red-orange" would normally be drawn, and that lightness *is* the separation, against
Buddhism at L 45 and Hinduism at L 55. dE 31 from each. Ravidassia gave up its place and now sits
**beside** Sikhism at dE 13, which is the honest statement: it declared itself separate in 2010,
most of its people are still counted Sikh, and there is no room in the arc to claim otherwise.

The closest pairs the new warm end leaves, all clearing: Catholic/Hinduism 34, Hinduism/Sikhism 31,
Buddhism/Sikhism 31, and Islam against Christianity's last branch at **26.8**, which is the
tightest thing in the palette and the reason the band stops where it does.

**Inside the band, the six LINEAGE groups (§6.5) take an equal share each and stay contiguous**, so
the ancient communions are one run of yellows and the Pietist–Wesleyan line another, and the pairs
a reader compares — Catholic against Baptist against Pentecostal — are in different groups and
therefore a whole block apart. Equal shares per *group* and not per member: weighting by size would
be better on the map and would move every colour on the next retile.

**THE HONEST LIMIT, AND IT IS THE POINT OF THE SECTION.** 22 Christian branches in 46° cannot be
told apart, and no allocation fixes it. Measured, against `tools/check_overview.py`:

| lever tried | worst pair among the branches over 500 dots |
|---|---|
| three tiers, as §6.3 uses | dE 2.9 |
| six tiers | dE 6.2 |
| widening the band as far as Islam allows | dE 7.6 |
| authored size weights, up to 8× on Catholic | dE 6.9 — no change |

Weighting does nothing because the constraint is not how the arc is divided, it is how long the arc
is. So the overview does not claim branch legibility and should not be tuned as though it might
acquire it: **it claims that a dot's family is readable at a glance, and that one click gets the
branches back.** That is §6's own sentence — "with everything drawn at once a family has to read as
a family, so its members take shades of one hue and become mutually indistinguishable — right for
the overview, useless for 'what are the Christianities'" — arrived at a second time, by measurement
rather than by argument.

**`tools/check_overview.py` is the checker, and it asks two questions rather than one.** Pairs
*across* families under dE 25 are failures, because the first thing a dot has to say is which
family it is. Pairs *inside* one family are held to dE 12 and reported separately, because the
overview means those to be close and a single bar would bury the two real failures under 300
expected ones. Colours come out of `index.html` itself through `tools/palette_dump.js`, which
slices the allocator's own declarations out of the inline script and runs them in node — so, like
`check_palette.py` reading `ROOT_HSL`, the checker cannot drift from the palette it measures.

What survives across families is the §6.3 list and nothing new: the wedge's small pairs, and
unchurched against the source catch-all at dE 23.

### 6.10 A row too small to be worth a line folds into one — DECIDED 2026-09-03

The legend has a fixed budget and the tree does not. At depth 2 the all-religions view drew 28 rows
under Christianity and five of them — Church of the East, Hussite, Moravian, Plymouth Brethren,
Swedenborgian — were **115 dots between them against Christianity's 501,322**. Each cost a line, a
swatch and a slice of the band, and not one is findable on the map at that size.

So a child holding less than **1e-4 of its parent's total** folds into a single row, provided at
least two of them do. Two properties earn their place:

- **The denominator is the parent, not the map.** Inside Judaism a row is measured against Jews.
  That is what makes the rule scale-free, and it is why Reconstructionist Judaism (39 dots, but
  1.3% of Judaism) stays and Hussite (23 dots, 4e-5 of Christianity) does not.
- **Two is the minimum because folding one row into a row that says "1 small group" saves nothing.**
  The rule has to buy legend space to be worth the indirection.

**Nothing is hidden by it.** The bucket expands, its members keep their counts and stay selectable,
and selecting one takes it out of the bucket for as long as it is selected — you cannot select a
thing and have it disappear. What the fold actually removes is the *claim to a colour*: the folded
groups take their family's own shade, which is the true statement about a speck at this scale.

Like §6.6's `unspecified`, the bucket row is **not a node** — no count of its own in the tree, no
id, cannot be selected — and it is computed in the viewer from the tallies rather than materialised
in the taxonomy. It sits last under its parent, because it is the residual of the list above it and
belongs to no lineage group.

**It fires where the tail is long and nowhere else**, which is the check that it is measuring the
right thing: four rows fold in the all-countries view and none in the United States or Czechia,
whose Christian branches are all above the line. Anita's threshold; 2e-4 would also take Maori
Christian churches and the Quakers, and the constant is one line.

### 6.11 The reader gets the hand overrides too — DECIDED 2026-09-03

§6 said "hand overrides are expected, and bounded to a few scopes", and meant Anita editing a
table in the source. `PIN` and `PIN_OVERVIEW` are that. This adds the same two powers to the
legend itself, because the argument for them does not depend on who is holding it: **click any
swatch to set that group's colour or to hide it.**

**Hiding stops being a special case.** `unaffiliated` started hidden for the reason §6.2 gives,
and it was implemented as a constant plus a boolean — so it was the only thing in the taxonomy
that could ever be hidden, and "no religion: hidden" was a feature rather than an instance of one.
It is now simply the entry the hidden set starts with. The control and its tooltip stay, because
the reason for hiding *that* one is a measurement problem and not a preference, but Baptists can
now be hidden by the same mechanism and cleared by the same reset.

**A hand-set colour wins over both palettes, in every scope and every country** — which is what
makes it worth setting. It is applied at the **drawn category**, though, not to the node's dots
wherever they fall, so §6.3's flat rule survives: pin Amish purple while Anabaptist's children are
drawn and Amish is purple; take the cut back out and Anabaptist paints its whole subtree, Amish
included, as it does for every other colour. The alternative is a colour on the map with no swatch
in the panel, which is the defect §6.3 measured at 93% of dots and fixed.

**The picker is ancestrydots', ported unchanged in style** — Anita's call, and the right one: that
one is already known to be pleasant, and there was no reason for this map to invent a second idiom
for the same job. Three gradient sliders whose saturation and lightness tracks repaint from the
current hue, a preview swatch with the hex, and a sixteen-colour preset grid. Not the browser's
native colour dialog, which is an OS window that covers the map you are picking against — and the
whole point of picking here rather than in the source is watching the dots change while you drag.
The presets are deliberately **not** the family hues: offering Islam's green for a Christian branch
would invite the exact collision the authored tables exist to prevent.

**What is set is kept in `localStorage`, and that is a per-browser promise, not a per-map one.**
So `copyOverrides()` puts the current overrides on the clipboard **in `PIN`'s own `[h, s, l]`
form** — a block that has to be translated before it can be pasted is not a paste-ready block.
That closes the loop §6.8 opened: a stable palette is worth hand-correcting, and a correction that
lives in one browser is not a correction. The route for a colour worth keeping is picker →
`copyOverrides()` → `PIN` and `PIN_OVERVIEW`, where every viewer gets it.

**It is a console call and not a button — Anita's edit, 2026-09-03**, and the reason is worth
keeping: the panel is the legend, its audience is readers, and getting a colour back into the
source is a maintainer's errand. A control that only one person will ever press does not belong
in the key to the map.

**And one reset for both**, sitting with the other legend controls in the same blue-link idiom,
greyed out when there is nothing to reset. Nothing is hidden by default any more — §6.2 has why.

**What made it usable, found by using it.** Three things, and the third is the one worth writing
down:

- The `hide` and `reset` links describe the node's *current* state, so they are recomputed on
  every change and not only when the popover opens. They were written once at open time, so
  setting a colour and then reaching for `reset` found it still greyed out — the only way back to
  the default was to close the popover and reopen it.
- The dot stays 9px because that is what reads as a legend key, but the hit area is a square the
  full height of the row, with negative margins so the rows do not indent further than they did.
  A 9px circle is a legend mark; it is not a button.
- **A live colour preview costs more per update than anyone guesses, and the fix is to gate on
  completion rather than on time.** Dragging the hue slider left the map still changing colour
  ten seconds after the pointer came up, working through states already dragged past. Three
  things were wrong and all three had to go:

  1. `applyPaint` re-set the layer **filter** on every call, and a filter change makes MapLibre
     re-evaluate it against every feature in every loaded tile — six hundred thousand dots across
     thirteen countries at low zoom. A colour drag does not change the filter. `applyColors` sets
     only the colour expression.
  2. It painted `dots`, `atomic` and `rings` every time, and **exactly one of the first two is
     ever visible** (§4.2b) while rings are usually off. A hidden layer costs the same
     re-evaluation as a visible one. A live drag now touches only the layer on screen, and the
     other two are brought up to date when the drag ends.
  3. **Both a per-frame and a per-120ms throttle still backed up**, and this is the part worth
     remembering: `circle-color` is a data-driven `match` over the node id, so each change makes
     MapLibre re-evaluate the property per feature and re-upload vertex attributes for every
     loaded tile — well over 120ms here. Any *timer* is guessing at a cost it cannot see, and the
     surplus piles up inside MapLibre where no timer of ours can reach it. So the gate is
     **completion**: issue a repaint, wait for the map's `idle`, then issue the next — and only
     ever the current value. Queue depth is one, releasing the slider costs one more repaint, and
     a slow phone simply draws fewer intermediate states, which is the right way to be slow. A
     timeout guards it, because `idle` is not contractually guaranteed and a pipeline that can
     wedge is worse than one that occasionally double-paints.

  Measured on a 600-event, 1.4-second drag: **3 repaints during, 2 after release, settled in
  537 ms**, on the value the slider ended at. Before: ten seconds of catching up.

## 7. Confidence is carried, not drawn — REVERSED 2026-09-04

**The desaturation below was built on 2026-09-04 and removed the same day. Anita: every colour on
the map has to be a colour in the legend.** It failed on contact with §3.5a, which is what it was
built for: 51% of the American dots are the survey residual, so Christianity's own `unspecified`
row drew as a **dull tan while the legend beside it showed bright yellow**, and a large category
appeared on the map in a colour that was nowhere in the key.

The rule that replaces it is general, and larger than fading: **the legend is the whole colour
vocabulary, and a reader matching a dot to a row must always find it.** Anything applied to a
colour *after* the palette is authored breaks that by construction, however principled the
modifier is. That rules out desaturation, opacity and lightening alike, and it is why the fix was
not a gentler fade.

**It runs both ways, and the second half was found the same day.** §6.3's flat rule already said
every colour on the map has a swatch in the panel; the converse — every swatch in the panel is a
colour on the map — was not true. Headings showed a **grey dot**: a §6.6 split branch, whose own
colour belongs to its `unspecified` row, and a branch that is not a drawn category at all
(`other` above `other.us`, `indigenous` above `indigenous.northamerican`). There are no grey
Buddhism dots anywhere on the map. Those rows now show a **blank the width of a dot**, which keeps
the labels aligned and leaves the count where it was, and they are no longer click targets for the
colour editor, because a heading has no colour to set or hide. The test is the fallback colour
rather than a list of node kinds, so a third case gets the blank for free.

**What survives.** `tier` still travels from the adapter in `countries.py` through `scatter.py` to
a `t` on every dot and into the tiles — it is a true fact about the row, `measured` is the default
and is written nowhere, so it costs no tile bytes. Nothing renders it. **If confidence is drawn
again it has to be in something that is not colour**: size, a second mark, an overlay toggle that
the reader turns on knowing what it means, or the unit panel below. The three tiers and the
response-rate gate keep their definitions and stay the right ones.

**And §3.5a is left exposed by this, which should be said plainly.** Its decision that the
declaration "stays quiet" rested on the desaturation carrying "this is modelled" on screen. With
the fade gone, 166.2M residual dots — half the American map — are drawn identically to ASARB's
counted adherents, and the only thing saying so is the country note. That is a smaller claim than
§3.5a assumed it was making, and it is now the strongest argument for the non-colour treatment
above.

Three tiers, from the source inventory, per unit per group:

| tier | example | rendering |
|---|---|---|
| measured | a census or register question, asked in this unit, any year (§3.4), **and answered** | full saturation |
| derived | §3.4 structure-from-older-source; a national figure distributed by a proxy | desaturated |
| modelled | no subnational data; country estimate spread by population | desaturated and, proposed, a visible texture or stipple |

**"Measured" needs a response rate, not just a question — found 2026-08-27.** StatCan dropped its
quality suppression for 2021, so **241 Canadian census subdivisions publish religion counts built
on ≥50% long-form non-response**. Those numbers look exactly like every other number in the file.
Australia's religion question is voluntary with 6.9% non-response nationally, and England, Wales
and Scotland are voluntary too. So the measured tier is gated on the unit's own response rate,
which every adapter records per row (`tnr_lf=` in the Canadian rows); a unit over some
non-response threshold drops to `derived` however good its source is.

The unit panel names the source and year for every line, so "why is Sichuan pale" has an answer
one click away. §3.5 is the reason this is not optional: the map's honest claim in China is
different in kind from its claim in Utah, and a map that renders them identically is making the
weaker claim silently.

**What was built and what was kept, 2026-09-04.** `tier` travels from the adapter in
`countries.py` through `scatter.py` to a `t` on the dot and into the tiles; `measured` is the
default and is written nowhere, so the common case costs no tile bytes. **That pipeline stays.**
It also reached the viewer's colour expression for a few hours, and that part is gone — see the
reversal at the top of this section. Still unbuilt: the response-rate gate below (no adapter
records `tnr_lf` yet), and the stipple, which cannot be seen inside a 1.5px circle anyway.

**The tier belongs to the people, not to the (unit, node) pair — FOUND 2026-09-04.** The first
build took the weakest tier on each pair, reasoning that a pair which is part census and part
spread-out total is not a measurement. That is true of the pair and false of its people, and the
result was backwards: **Ireland's 4.30M measured people and 508k derived ones drew 764 measured
dots against 4,030 derived**, because most Catholic pairs carry one large measured row beside one
tiny allocated one, and taking the weakest let the tiny row relabel the lot. The fix is not a
better rule but a smaller key — `tier` joins `(unit, node)` in the grouping, a mixed pair becomes
two rows, and the dots divide in proportion by construction. It now draws 4,295 against 499,
which is the input to within the ten dots the national carry loses. The general form: **a
qualifier on a row must not be aggregated over the rows it qualifies; it must key them.**

**Kept from the removed fade, because it will be true of any replacement: the palette's own
saturation varies 7×, so no single modifier applies evenly.** Roots are authored from saturation
83 (Christianity) down to 12 (No religion), and a saturation multiplier cannot make an
already-grey colour greyer — `unaffiliated` moved #92a2aa → #8c9397, which nobody can see, and
that is exactly where §3.5a's residual lands. Any future confidence treatment has to be uniform
in the thing it changes, which is the second reason not to reach for colour: **colour is the one
channel the palette has already spent.**

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

### 8.1 Boundaries must be the vintage the data was *published on* — FOUND 2026-08-27

Not the newest available, and — refined 2026-08-27 — **not the year it was collected either.**

Taking the newest is the obvious default and it silently deletes places. But "use the collection
year" is also wrong: **Stats NZ recodes its 2013 and 2018 census addresses forward onto the 2023
SA2 boundaries**, so New Zealand's older columns want the *newer* geography, the exact mirror of
Connecticut. The rule that covers both is the vintage the table is published on, which the
publisher states and which no amount of reasoning from the data's date will recover.

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

**The cost, measured on Australia 2026-08-27:** using the 2016 boundaries against 2021 data
matches **87.7% of codes** and silently drops **303 SA2s — 3,866,694 people, 15.2% of the
country**. An 87.7% match rate is exactly the kind of number that looks like success in a log.

**And there is a third direction, which two-way matching does not catch: a code can match and
still have no geometry.** Australia has 18 special-purpose SA2s — migratory, offshore, no usable
address — whose codes join perfectly and whose polygons are empty; they hold 52,920 people who
would be scattered nowhere at all. Worse, their `AREASQKM21` is **NaN rather than 0**, so the
obvious guard (`area == 0`) matches nothing and keeps them. So the check is three-way: unmatched
data, unmatched polygons, **and matched-but-empty geometry**.

Canada is the counter-example that shows the good design: StatCan's DGUID embeds its own vintage
(`2021A0005…`), so joining 2021 data to a wrong-vintage file yields **zero** matches rather than a
plausible 88%. A loud total failure is a far better property than a quiet partial one, and it is
worth preferring a vintage-stamped key wherever a source offers one.

**A fourth way to get the vintage wrong: the file format renames the column for you.** Ireland's
Small Area shapefile carries both the 2022 and the 2016 keys, and **the DBF format truncates field
names to ten characters**, so `SA_GUID_2022` arrives as `SA_GUID__1` while `SA_GUID_20` — the name
that looks like the 2022 key — is in fact **the 2016 one**. Joining on the obvious-looking column
silently gives you the previous census's geography, on which 1,448 of 18,919 codes have changed.
The rule that survives this: **confirm the key by joining, not by reading its name.** A correct key
matches 100% and the wrong one does not, which is the only reliable signal available.

And Mexico's own geography clears the geoBoundaries problem noted above — INEGI's Marco
Geoestadístico 2020 has all 2,469 municipios including Coatetelco, Xoxocotla and Hueyapan, and
joins 0/0 both ways.

The check also confirms what should be absent: 91 polygons have no ASARB row and all 91 are
Puerto Rico, American Samoa, Guam, the Northern Marianas and the US Virgin Islands, which ASARB
does not cover. An expected absence and an accidental one look the same until you name the
expected ones.

This will recur everywhere and worse: municipal mergers in Japan, Brazil and Indonesia run
continuously, and a census's own geography is the only safe join target for that census.

**And the global boundary source has the same disease — including the one §8 recommends.**
geoBoundaries' Mexico ADM2 is **2012 vintage, 2,457 units, against the 2020 census's 2,469**; three
Morelos municipios (`17034`–`17036`) simply do not exist in it, so the Connecticut failure is
waiting there in exactly the same silent form. This does not disqualify geoBoundaries — it is still
the right licence and the right coverage — but it does mean **its vintage is a per-country fact to
check, not a property of the dataset**, and that where a country publishes its own census
geography, that is what the join should use. Kept as an explicit pre-flight: for every country,
compare unit counts and report both directions before scattering a single dot.

### 8.2 Placement needs no population data — DECIDED 2026-08-27

The project does not depend on a population layer, and the reasoning is worth keeping because the
obvious version of this decision is wrong in both directions.

**Two jobs get confused.** Population is used for (a) the residual — how many people belong to
nothing — and (b) placement, deciding where inside a unit the dots go. They are unrelated. (a)
needs one number per unit; (b) needs relative weight *within* the unit.

**(a) is free wherever the religion source is a census**, because a census reports population too.
ASARB ships a `2020 Population` column per county in its own summaries workbook. So the US residual
costs nothing and needs no API. It is computed and kept out of the first render by choice, not by
necessity.

**(b) is free in the United States too, and this is the useful trick.** Census tracts are *designed*
to hold about 4,000 people. So allocating a county's dots **equally across its tracts** is already
a population weighting — the geometry carries the weight, and no population figure is read at all.

Measured, to check the design is actually adhered to rather than merely intended:

| | |
|---|---|
| tracts, 2020, nationally | 85,187 across 3,143 counties |
| people per tract (county means) | median **3,424**, IQR **2,818 – 4,043** |
| log-log correlation, county population vs tract count | **r = 0.98** |

**What this is not.** The measurement above is *between* counties; the error that matters is
*within* one, and it cannot be measured without the tract populations we are declining to fetch.
The honest bound is the design range itself — tracts run roughly 1,200–8,000, so two tracts in the
same county can differ by about 3×, and a dense tract is under-dotted against a sparse one by up
to that. Against the alternative it is nothing: uniform-random over a *county* polygon is wrong by
two orders of magnitude in the western US, where it would scatter San Bernardino's adherents across
20,000 square miles of Mojave.

**The consequence, stated plainly:** without a population layer the map shows **counts, not
shares**. A county that is 48% adherent and one that is 90% adherent look the same. That is a real
limitation, it is reversible, and for a first build it is the right trade.

`fetch_tract_pop.py` is written and kept for the day exact weights are wanted; it needs a free
`CENSUS_API_KEY`, since the API stopped serving unkeyed requests. Nothing depends on it.

**Where the placement layer ships its own population, use it rather than approximating.** New
Zealand's SA1 file carries a 2023 population per polygon (median 150, IQR 120–183), so NZ can be
weighted exactly and the equal-share assumption is not needed at all. The approximation is a
fallback for the common case where no population travels with the geometry, not a preference.

**It generalises, and Australia is a better case than the US — measured 2026-08-27.** ABS Statistical
Area 1s hold a median of **406 people, IQR 359–447**, against the US tract median of 3,424 with an
IQR nearly twice as wide in relative terms; the correlation between SA2 population and SA1 count is
**r = 0.923**, and every real SA2 has at least one usable SA1. So equal-dots-per-SA1 is a sound
population weighting, on a unit eight times finer than a US tract. Canada's dissemination areas are
the same idea. The trick is not a US accident — it is what happens wherever a statistical agency
designs its smallest unit to a population target, which is nearly everywhere.

**Globally**, the same two observations should mostly hold: a country whose religion data comes
from a census has that census's population alongside it, and countries publish *some* finer unit
than their religion tabulation. Kontur/GHS-POP stays the fallback for where neither is true, and
it moved from "second stage of the pipeline" to "fallback" on the strength of this.

**Placement.** Dots are placed by **population grid**, not uniformly in the polygon.
ancestrydots' `random_points_in_polygon` is fine for US census tracts, which are small and roughly
population-equal by construction; at ADM1 scale globally it would fill the Sahara, the Amazon and
Siberia with people. Kontur Population (400m H3, HDX, and already vector) or GHS-POP (100m raster)
— Kontur is likely the better fit since it is hexagons out of the box and §5 wants cells anyway.

**Placement is not a claim about which people are where.** Within a unit, a group's dots are
scattered in proportion to total population. Where a source gives a finer unit we get finer truth;
where it does not, the dots say "this many people of this group live somewhere in this unit". The
about panel has to say so, because a dot map invites exactly the opposite reading.

### 8.2b Germany is the first country that needs no trick at all — BUILT 2026-09-04

§8.2 and §8.4 are both answers to the same missing thing: **nobody publishes where a religion
sits inside a unit**, so the map either assumes an equal share over units engineered to a
population target (§8.2), or fits a model to guess it (§8.4). Germany publishes it.

destatis puts **the same three categories on the 1km INSPIRE grid** it puts in the Gemeinde
table. So the weight for `christianity.catholic` inside Munich is Munich's own per-cell
Catholic count, and the placement stops being an approximation:

| | |
|---|---|
| placement layer | **209,154** 1km cells, replacing 10,786 Gemeinde polygons |
| Berlin | **799 cells**, against one polygon holding 3,596,999 people |
| rows placed on measured weights | **17,215 of 17,215** — no fallback used |

It had to be done, because §8.2's trick fails completely here: Gemeinden are historical units,
not units built to a population target, and they run from 9 people to 3.6 million. 78 of them
hold 31.6% of the country, and inside those the old placement said only "somewhere in this
city" — Neukölln and Zehlendorf came out identical, and dots landed in the Grunewald and the
Müggelsee.

**Nothing here is fitted, and that is the point.** §8.4's US model has parameters, a residual
model, and a §7 confidence mark; this has none, because it is a count. §14.4's rule — never
estimate a magnitude a source does not publish, refine placement only — is satisfied in the
strongest possible way: the refinement is itself published.

**Why 1km and not the 100m file**, which also exists at 3,088,036 cells: Germany draws 82,710
dots, so 100m would be 37 cells per dot. **The dot value binds before the grid does.** A finer
placement layer than the dots it carries is bytes, not information.

The general shape, for the next country that has one: *grid cell → containing admin unit →
clip to it → per-node weight column*, counts still from the admin table. What does not
generalise is the good part — most grids carry population only, which is a better proxy but
still a proxy. Germany is unusual in publishing the **same variable** on the grid as in the
table (`sources/de_grid.md`).

### 8.2a India is the first country the trick does not work on — FOUND 2026-09-03

§8.2's whole argument is that **a statistical agency designs its fine unit to a population
target**, so an equal share per polygon is already a population weighting and no population figure
need be read. US tracts ~4,000 people, Australian SA1s ~406, Irish Small Areas ~100 households,
Canadian dissemination areas. Six countries in, it looked like a property of censuses.

It is a property of *statistical* geographies, and India has none. Below the sub-district India has
**645,828 villages and 4,135 towns**, and those are administrative and natural settlements ranging
from ten people to two million. An equal share per village would weight a hamlet like a small city,
and India has a very great many hamlets. There is no engineered layer anywhere between the
sub-district and the settlement.

So India places on its count layer, as Poland, Romania and Brazil do — and pays for it:

| | India | Brazil | Poland | US |
|---|---|---|---|---|
| median unit population | **~204,000** | ~38,000 | ~7,500 | ~4,000 |
| median unit area | **551 km²** | 1,527 km² | 126 km² | — |

**The population figure is the coarsest count unit on the map; the area figure is finer than a
Brazilian município's.** So the damage is zoom-dependent in a way worth stating: at national and
state zoom India's grain is comparable to a country already drawn, and it is at city zoom that
India looks blockier than anywhere else.

**The fix is specific and the machinery for it now exists.** SHRUG publishes 645,828 village
POINTS with `t_pop2011`, summing to 828,886,066 — India's entire rural population — and the towns
file carries the urban half. Weighting placement by that would put rural dots on actual settlements
instead of spreading them over a polygon.

`scatter.py`'s `place_weight` hook (the US's §8.4 weighter) is exactly the right shape for it, so
what is left is not a capability but two pieces of wiring: a placement layer of villages and towns
keyed to their sub-district, and a weighter returning each settlement's 2011 population. **India
does not use `place_weight` today only because its placement layer is its count layer — one polygon
per unit — so there is nothing inside a unit to weight.** That is the single biggest available
improvement to how India looks, and India is the obvious second customer for the hook after the US.

### 8.3 Placing dots by church location — TRIED AND REJECTED 2026-09-03

**The idea.** US religion data is county-level and cannot be finer (§8.2, and `sources.md` §3:
PL 94-521 bars the Census Bureau from asking, so ASARB is the only national enumeration and it is
compiled by county). A county is ~105,000 people — fifteen times coarser than a Canadian CSD and
sixty times a Czech obec — so every tract in Cook County gets an identical religious mix and
Chicago renders as a uniform blend with no neighbourhood structure at all.

The proposal: keep ASARB's county total as the magnitude, but use **church locations** to decide
where inside the county the dots go. This is §3.4's structure-from-elsewhere applied to geography
instead of categories, and §3.6 argues for it — ASARB already attributes adherents to the
*congregation's* county, so the quantity being counted is congregation-located, and spreading it
evenly across a county is a fiction the source does not contain.

**Two of the three things needed turned out to be fine.**

1. **OSM coverage is good enough, for Catholics.** Measured against ASARB's own congregation
   counts across eight counties chosen to break it, urban through the Navajo Nation:

   | | min | max | median | spread |
   |---|---|---|---|---|
   | all congregations | 41% | 104% | 64% | 2.5x |
   | of those, carrying a `denomination` tag | 8% | 66% | 43% | **8.0x** |
   | **Catholic only** (7 of 8 counties) | 77% | 104% | **100%** | **1.3x** |

   So §4.5's mapping-effort worry is **confirmed for the general case and wrong for Catholics**.
   In McKenzie County ND only 8% of mapped churches say what they are; weighting by that would
   draw tagging habits as religious geography. But Catholic churches are large, named, landmark
   buildings, and OSM has essentially all of them. Note that between-county variance barely
   matters anyway — the county total stays ASARB's, and the weights only decide distribution
   *within* a county.

2. **The obvious model produces impossible numbers, and a kernel fixes it cleanly.** Giving each
   parish an equal share of the county total and assigning tracts to their nearest parish
   (Voronoi ≈ the territorial boundaries canon law implies) put **13% of Cook County's Catholic
   dots in tracts implied over 100% Catholic** — a one-tract catchment receives 4,287 people into
   ~3,400 residents. Replacing Voronoi with a Gaussian kernel and sweeping the radius gives a
   clean window at σ = 2–3 km: nothing over 100%, interquartile share still 12–50% against a
   county mean of 28.9%. And σ = 2,642 m, the mean parish spacing, is a **parameter-free** choice
   from the data that lands in the middle of that window. Tidy.

**The third thing killed it: parish density does not measure where Catholics live.**

Checked against Chicago's actual, well-established religious geography — the only validation
available, since any dataset good enough to confirm the model would be good enough to use
*instead* of it:

| neighbourhood | reality | parishes ≤3 km | model |
|---|---|---|---|
| Mount Greenwood | Irish-Catholic heartland | 8 | **29%** |
| Garfield Ridge | Polish/Irish Catholic | 8 | 32% |
| Englewood | Black Protestant since the 1960s | **7** | **39%** |
| Washington Park | Black Protestant | 8 | 33% |
| Little Village | Mexican Catholic | 19 | 78% |

Parish density is **the same** in Chicago's most Catholic neighbourhood and its least, and the
model therefore rates Englewood *more* Catholic than Mount Greenwood. Strip out Little Village
and the HIGH group averages 37% against LOW's 34% — no discrimination at all.

**The mechanism, which is why no amount of tuning saves it.** South Side parishes were built
1890–1930 for Irish, German and Polish immigrants. The Great Migration turned those
neighbourhoods over; the buildings stayed. So parish density there is a **fossil of 1920s
settlement**, not a measure of 2020 Catholics. Meanwhile Mount Greenwood's eight parishes are
large and full. The model's load-bearing assumption — that parishes hold roughly equal numbers —
fails in a *spatially structured* way: inner-city parishes are remnants, outer ones are full.
Little Village scores correctly only by accident, being an old dense parish network that happened
to stay Catholic (Bohemian → Mexican).

Every older US city has the same fossil pattern, so this is general, not a Chicago quirk. And it
fails hardest in precisely the neighbourhoods most worth resolving.

**What would actually be needed:** parish-level registered households or mass attendance, which
some dioceses hold and none publish uniformly. That is a real per-diocese data hunt, not a
modelling trick.

**The lesson worth keeping.** Every internal check passed. Coverage was excellent, the σ sweep was
clean, the parameter chose itself, the median share matched the county truth. It looked finished.
Only comparison against known ground truth revealed that the signal was a century out of date —
and had it shipped, "Englewood is Catholic" would have read as a discovery rather than an
artefact. **A model built from a proxy needs an external check against something known, or it
should not ship; internal consistency cannot detect a decorrelated input.**

### 8.4 Placing dots by demographic composition — BUILT 2026-09-03, and it passes §8.3's test

The second attempt at the same problem, and the one that ships. `us_weights.py`.

**There is no sub-county data to get, and that was checked first.** The US Religion Census
publishes at county and **does not collect congregation addresses** — its own "Data Collected"
page says so — so there is no finer ASARB file to find, and PL 94-521 means no census question
exists to fall back on. Two things do exist and neither solves this:

| | |
|---|---|
| **Per-congregation membership** | Real and public — UMC (`umdata.org`), ELCA, PCUSA, Episcopal parochial reports, UCC all publish membership per church with an address. This is exactly what §8.3 said was missing, and it is not a fossil: a dying parish reports 150 members. But it covers mainline Protestants, ~12% of adherents, and *the wrong 12%* for the three counties that need it most, where the mass is Catholic, Black Protestant, Hispanic, Jewish and Muslim. Catholic per-parish figures are diocesan and unpublished. |
| **Jewish community studies** | The one genuinely measured sub-county source in the country. UJA-Federation's **2023 Jewish Community Study of New York** gives Jewish population by sub-county ZIP cluster, by denomination, for the eight New York counties. JUF Chicago 2020 gives about ten metro regions, much coarser. Brandeis' AJPP models on ZIP clusters internally but publishes only county and up, and forbids scraping. **Not yet used, and it is the top of the list** — see the Judaism failure below. |

So the magnitude stays ASARB's and only the placement is modelled, as in §8.3. The proxy this
time is **demographic composition** — ancestry, birthplace and race at tract level, which
ancestrydots already holds for every state, so no new download and no API key.

**The check is the point.** §8.3's lesson was that a proxy model needs external ground truth,
and there is some: **ASARB's own county numbers inside a metro**. Fit on counties, hold out
WHOLE METROS, and see whether the model predicts variation it never saw. 448 counties in 48
metros, scored on population-weighted correlation of the within-metro deviation — correlation
rather than R², because the allocation is raked to the ASARB county total, so the level is
fixed for free and only the relative pattern has to be right.

| | held-out r | | held-out r |
|---|---:|---|---:|
| Church of God in Christ | **0.61** | Jehovah's Witnesses | 0.21 |
| National Baptist Convention USA | 0.57 | Hindu Temples | 0.20 |
| Seventh-day Adventist | 0.57 | Lutheran — Missouri Synod | 0.12 |
| National Missionary Baptist | 0.56 | United Church of Christ | 0.09 |
| Reform Judaism | 0.46 | Assemblies of God | 0.07 |
| **Catholic Church** | **0.45** | **Orthodox Judaism** | **0.05** |
| Episcopal, AME, American Baptist | 0.42–0.45 | | |

Calibration slopes run 0.7–0.96, so the predicted spread is about the right size rather than a
muted smear. **26 fitted nodes clear r ≥ 0.25**; everything else stays population-uniform. That
gate is the whole design — it is a per-node confidence claim, which is what §7 wants anyway,
rather than one switch for the country.

**Against §8.3's own validation set**, same five Chicago neighbourhoods, same question:

| | reality | parish model (§8.3) | this |
|---|---|---:|---:|
| Mount Greenwood | Irish-Catholic heartland | 29% | **78%** |
| Garfield Ridge | Polish/Irish Catholic | 32% | 64% |
| Little Village | Mexican Catholic | 78% | 54% |
| Englewood | Black Protestant since the 1960s | 39% | **14%** |
| Washington Park | Black Protestant | 33% | 17% |
| | **HIGH vs LOW mean** | **37% vs 34%** | **66% vs 15%** |

Cook County is 53% Catholic and the old placement drew that everywhere. Englewood also goes
from 10% to 34% Black Protestant. Glendale draws Armenian Apostolic at 15× the LA County rate;
Richmond Hill draws Hindu at 2.7×.

#### Two things it got wrong before it was right, both worth keeping

**1. The fitter destroyed the signal and nearly got the method thrown out.** The first three
specifications returned NEGATIVE R² on every family, which reads as a clean kill — an
alternating per-metro scale step was absorbing exactly the variation it was meant to explain.
What caught it was correlating each demographic segment against each body **directly, with no
model at all**: National Baptist against `black_resid` is r = 0.60 raw. The signal had been
there the whole time. **When a model says there is no signal, check for the signal without
the model** — a null from a fitted model is a statement about the fitter first.

**2. A node can clear the gate on a coefficient that means nothing.** Armenian Apostolic scored
r = 0.35 and passed — with **no `armenian` coefficient at all**, its largest term `afro_carib`,
and `east_asian` positive. It drew Armenian dots in San Gabriel and none in Glendale. The cause
is that one ridge penalty against columns whose scale differs by three orders of magnitude
shrinks the small ones far harder, and `armenian` is 0.2% of the population. Swept both ways:

| | nodes | adherents kept | mean r | median slope |
|---|---:|---:|---:|---:|
| raw shares, λ = 0.15 | 26 | 66.6M | 0.42 | 0.92 |
| standardised, λ = 0.05 | 12 | 40.4M | 0.36 | 0.37 |

Standardising recovers `armenian` perfectly — rank 15 to rank 1, +0.55 — and wrecks everything
else, because slope 0.37 means the predictions run three times as far as the truth. **Neither
penalty serves both cases, and the honest reading is that a body defined by a 0.2% ethnicity is
not learnable from between-county variation at all.** Shrinking a small segment harder is not a
defect; it is the correct response to there being less information in it.

So there are **two tracks**, and the second exists because of that finding:

- **fitted** — 26 nodes that clear r ≥ 0.25 on held-out metros. Evidence.
- **authored** — 31 nodes whose ethnicity is *constitutive rather than correlated*: the
  Armenian Apostolic Church is Armenian by canon, Mar Thoma is Kerala, the Ethiopian Orthodox
  Tewahedo Church is Ethiopian. These are asserted, in the same spirit as §2.4's 372 hand-mapped
  placements, and carry `basis: authored` so the claim is never read as a measurement. The bar
  is narrow: the body's own name or canon must name the ethnicity. Bodies that merely *skew*
  ethnic — Southern Baptist, the Church of God in Christ — stay with the fit, which is what
  evidence is for.

Together, 89% of adherents. The rest are population-uniform.

**Placement also stopped approximating.** §8.2 spread a county's dots equally across its tracts
because tracts are designed to a population target; the ACS tract total is a real population and
is now used directly for every node, weighted or not. §8.2 already said to prefer a shipped
population where the placement layer carries one — it does, and the 1,200–8,000 spread inside
that approximation is gone.

**Connecticut is §8.1 for the third time, and in the new direction: two vintages are needed at
once.** The placement layer must be 2020 tracts, because their county prefix is what joins to
ASARB — but ACS 2020–2024 publishes Connecticut on the **2022 planning regions**, 09110–09190
against 09001–09015, so 879 of 884 tracts fail the GEOID join and the state silently reverts to
uniform. The tracts did not move, only their numbering, so a representative-point join from the
2020 polygons onto the 2024 ones recovers all 879. Every state that adopts a new
county-equivalent scheme will need this.

#### What it is not, and the about panel has to say so

1. **It is an estimate.** Nothing below county level here was counted.
2. **It is partly the race map.** "Englewood is Black Protestant" is the input, not a finding.
   The part that is not circular is ancestry and birthplace — Guyanese Hindus in Richmond Hill,
   Armenians in Glendale — which no race map contains.
3. **The check is between counties and the use is within one.** Coefficients get extrapolated
   well past their fitted range: metro counties run 5–25% Black, Cook County tracts run 0–100%.
   The direction is favourable — more demographic contrast, not less — but it is extrapolation
   and nothing exists to validate it there.
4. **`islam`, `hinduism` and the three Buddhist nodes are ASARB compiler estimates** (group
   codes 267, 890–892, 895; §3.1). A number modelled from population can be reproduced by a
   model of population, so `islam`'s r = 0.64 — the highest here — is not independent evidence.
   The flag travels into the model file. They still take weights: uniform is not the safer
   answer, only a different wrong one.

#### The open failure: Judaism, and it is a data limit

**Orthodox Judaism scores 0.05.** Brooklyn is +6.9pp against its metro and the model says
+0.1pp. The ACS has no Jewish marker of any kind: `israeli` is Israelis, who are not most
American Jews and are barely any Haredim, and the Haredi neighbourhoods report European
ancestries that the segment table reads as `euro_catholic`.

The damage is not confined to Judaism, which is the part worth understanding. Orthodox Judaism
correctly gets no weights — but **Catholic does**, so it takes the space instead: Borough Park
draws 58% Catholic, 1.3× the Brooklyn rate, in the most Jewish neighbourhood in America. Skokie
and Beverly-Fairfax do the same thing. **A missing predictor is not neutral; the bodies that do
have one absorb what it should have held.**

Two fixes, in order:

1. **ACS B16001, language spoken at home, at tract level.** It carries **Yiddish** and Hebrew —
   Yiddish is very nearly a Haredi marker — and also Gujarati, Punjabi, Urdu, Bengali,
   Malayalam, Armenian, Amharic, Arabic, Persian and Somali, which would sharpen half the
   authored table into fitted evidence. Needs a free `CENSUS_API_KEY`; the API now refuses
   unkeyed requests, decennial and ACS alike.
2. **UJA-Federation 2023 for the eight New York counties**, which is measured rather than
   modelled and would make Judaism in New York the best-placed body on the US map instead of
   the worst.

#### Language fixed the Judaism hole — 2026-09-04

The failure above was real and its shape is the general lesson: **a missing predictor is not
neutral.** Orthodox Judaism correctly got no weights, and Catholic, which did, took the space
instead — Borough Park drew 58% Catholic, 1.3× the Brooklyn rate, in the most Jewish
neighbourhood in America. The bodies that *can* be placed absorb what the missing one should
have held, so an honest gap in one node becomes a confident error in another.

**The block was self-imposed.** `api.census.gov` refuses unkeyed requests now, and that was
taken as the end of the road for the language table. It is not: the ACS **summary file** on
`www2.census.gov` is the same data as flat `.dat` files, one per table, no key. Worth
remembering before concluding a census table is unreachable.

B16001 carries **Yiddish** (as "Yiddish, Pennsylvania Dutch or other West Germanic" — the
Census does not split them) and Hebrew, plus Gujarati, Punjabi, Urdu, Bengali, Malayalam,
Armenian, Persian, Arabic and Amharic. Its finest geography is the **PUMA**, ~100,000 people,
not the tract — but that is 28 units inside Brooklyn against ASARB's one, and PUMAs are drawn
on neighbourhood lines.

**The ground-truth check passes on the nose.** The top Yiddish PUMAs in the United States, in
order: Monsey, Borough Park, Kiryas Joel, Williamsburg, then Holmes County OH and Lancaster PA
— which are Amish, correctly, because the category is a lump and both halves of it are wanted.
County Orthodox-Judaism share against Yiddish share is r = 0.52 across 991 counties.

| | before | after | | after |
|---|---:|---:|---|---:|
| Borough Park | 1.0× | **4.4×** | Kew Gardens Hills (Modern Orthodox) | 4.3× |
| Williamsburg | — | 2.7× | Pico-Robertson, LA | 5.8× |
| Midwood | — | 2.2× | Beverly-Fairfax, LA | 8.0× |
| Bed-Stuy (no Jews) | — | 1.1× | West Rogers Park, Chicago | 2.5× |

It is **authored, not fitted**, and always will be: r = 0.07 under metro holdout because 79% of
Orthodox Judaism is in one metro, exactly the Armenian Apostolic case. Yiddish and Hebrew go in
at equal weight and balance themselves — Brooklyn is 4.4% Yiddish against 0.8% Hebrew and lands
on Borough Park, while Queens and Los Angeles are Hebrew-dominant and land on Kew Gardens Hills
and Pico-Robertson. No tuning was needed, which is the sign the segments are the right ones.

**A PUMA is not a neighbourhood, and one case proves it.** PUMA 3604303 is "Bedford-Stuyvesant
& Crown Heights North" and is 6.2% Yiddish, because Crown Heights is the world centre of
Chabad. Spread evenly across the PUMA that made Bed-Stuy — which has essentially no Jews — 1.3×
the borough rate. The two halves of that PUMA are unalike in every other respect too: one is
Black and one is not. So a PUMA's speakers are now split across its tracts **by the ancestry
that carries the language** — Yiddish by white population, Malayalam by South Asian, Armenian by
Armenian — which is a disaggregation rather than a new assumption, since Yiddish-speaking
Haredim are recorded as white by the race question. Bed-Stuy fell to 1.1×, and it sharpened
everything else at the same time: Beverly-Fairfax 2.8× → 8.0×, Kew Gardens Hills 3.1× → 4.3×,
Artesia 1.8× → 3.9×.

**Two AUTHORED extras came free**, because the same table separates things ancestry cannot:
Malayalam for the four Kerala churches (Mar Thoma, the two Malankara bodies, Knanaya), Amharic
for Ethiopian and Eritrean Orthodox, Punjabi for Sikhism, Gujarati for Jainism and the Hindu
share of the Indian diaspora — `south_asian` alone contains Bangladeshis, who are Muslim.

**Still open.** Hindu placement in the South Asian corridors is the weakest of the authored
ties — Jackson Heights 0.9× and Devon Avenue 0.8× where both should be well above 1 — because
`hinduism` competes with a Guyanese term tuned for Richmond Hill. And the §3.5a residual
(`unaffiliated`, `secular`, unspecified `christianity`) is 46% of US dots and is placed by
population alone, which is a flat wash over every neighbourhood contrast on the map. PRRI's
county file is **not** usable as ground truth for it — those estimates are themselves a
Bayesian small-area model built from ACS demographics, so scoring a demographic model against
them proves nothing, the same trap as ASARB's "Muslim Estimate". Pooled CES microdata carries
county FIPS and a religion question and is genuinely independent.


### 8.4a The residual gets its own model, and its own ground truth — BUILT 2026-09-04

§3.5a's residual is **166.2M people against the roll's 160.6M** — more than half the US map —
and it was placed by population alone: a flat wash of `unaffiliated`, `secular` and unspecified
`christianity` laid identically over every neighbourhood, diluting every contrast §8.4 draws.
Three nodes are 150M of the 166M.

**It cannot use §8.4's model, and countries.py already says why:** applying a model of where
ASARB's adherents live to the people on nobody's roll would place the residual exactly where
the measured people already are, which is the one place they are not. So it needs its own
coefficients from its own target.

**ASARB cannot be that target** — the residual is by definition what no roll holds. **PRRI's
county file cannot either, and this is the trap worth naming:** those county estimates are
themselves a Bayesian small-area model built from ACS demographics, so scoring a demographic
model against them would prove nothing. It is ASARB's "Muslim Estimate" again — a number
modelled from population reproduced by a model of population (§3.1, §8.4's fourth caveat).

**The Cooperative Election Study is genuinely independent**: raw survey microdata, 680,895
respondents, a county FIPS and a religion question on every row, downloadable without
credentials. 407,874 of them from 2016 on, across 3,045 counties — Los Angeles has 16,070,
Cook 11,360, Kings 4,053. Same protocol as §8.4 otherwise: demeaned within metro, whole metros
held out. Two differences, both because the target is a survey rather than a census — counties
are weighted by CES sample size rather than population, since that is the precision of the
target, and only counties with ≥150 respondents are scored, because a share off thirty people
is mostly noise and would understate any model.

| | held-out r | slope | | held-out r |
|---|---:|---:|---|---:|
| hinduism | **0.62** | 1.14 | judaism | 0.45 |
| christianity (unspecified) | 0.51 | 0.88 | unaffiliated | 0.44 |
| secular | 0.46 | 0.73 | other.us | 0.32 |
| islam | 0.46 | 0.83 | *buddhism* | *below the bar* |

Seven of eight clear R_MIN, and they are as strong as the roll models. **Education and age had
to be added** to make it work — ancestry says who people descend from, not whether they go to
church, and degree share alone correlates +0.56 with a county's atheist-and-agnostic share.
Both are tract-level in the same keyless summary file (B15003, B01002).

**What it draws, and the shape is the non-obvious part.** Los Angeles County tracts by share of
adults holding a degree:

| degree share | secular | unaffiliated | unspec. Christian | Catholic |
|---|---:|---:|---:|---:|
| 7% (bottom decile) | 6.1% | 22.1% | 13.4% | 35.5% |
| 74% (top decile) | **20.8%** | **13.2%** | 6.8% | 29.9% |

**The two irreligion categories move in opposite directions.** Atheist and agnostic is 3.4×
higher in graduate neighbourhoods; "nothing in particular" is 1.7× higher in the least educated
ones. A single "irreligion" axis would have drawn both the same way and been wrong about one of
them. Across neighbourhoods: Lincoln Park 2.1× secular, Park Slope 1.7×, Silver Lake 1.6×,
against Little Village 0.37×, East LA 0.40× and Washington Heights 0.62×.

#### Two bugs, and both were caught by the same discipline

**1. ACS jam values.** `-666666666` means "no estimate", not a number. Left in B01002, median age
had a minimum of −52,769 and a standard deviation of 2,560; that one poisoned column dominated
the raw-scale ridge trace and shrank **every** coefficient to about zero. The fit reported no
signal at all — `secular` scored −0.01 — while the raw correlations behind it were 0.4 to 0.6.
Same lesson as §8.4's first three specifications, reached from the other side: **a null from a
fitted model is a claim about the fitter first.** Check the raw relationship before believing it.

**2. The design matrix drifted from the fit, silently.** The Weighter built its own column list
by hand, `ba_share` and `age` were not in it, `beta.get(name, 0.0)` returned zero for both, and
the residual model's two strongest predictors were dropped on the floor. Everything looked
right: the fit validated, the weights inspected directly looked sensible, the run counted 345
rows on the residual model. **Only a check against the DRAWN OUTPUT caught it** — dots came out
flat across education deciles, 0.98× top to bottom, while the model was predicting 3.4×. There
is now one `design()` function used by both fits and the Weighter, and a `KeyError` if a beta
names a column the design does not have. A comment warning that two code paths must agree is
not a mechanism; this is why.

**The spot-check that looked like a failure was confounded**, and it is worth knowing before
reading one. Measuring a node as a share of the dots in a radius is distorted by whatever else
is drawn there: Borough Park is 49% Orthodox Jewish, which mechanically depresses every other
node's share. The first neighbourhood table read East LA as *more* atheist than Silver Lake and
was simply not measuring what it appeared to. Aggregating by decile, where composition and
sampling noise both cancel, is the check that means something.

## 9. Viewer

MapLibre GL JS + PMTiles, the ancestrydots stack, which also means the R2 hosting route and the
`npx serve` dev server (Python's `http.server` does not do range requests).

**Style is copied from ancestrydots, not from nycriders** — Anita's call, 2026-08-27, and this
supersedes the general house-style note in `feedback_map_ui_style`. Its tokens: `body` `#111`,
panels `rgba(20,20,20,0.758)`, hairlines `#2a2a2a`/`#333`, scrollbar thumb `#555`, Nunito, the
`i` button bottom-left for prose. Dark, like ancestrydots and unlike citybrowser.

**One thing tiling takes away: the viewer can no longer count anything.** With GeoJSON it totalled
the country by walking features; with tiles it only ever holds the current viewport, so the panel's
per-religion totals are precomputed into `data/processed/counts.json` by `tiles.py`. Anything else
the UI wants to state about the whole dataset has to be computed at build time for the same reason.

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

### 10.0 The families are in a FIXED order — DECIDED 2026-09-03

The top level used to sort by prevalence, like every level below it. That is right inside the
tree and wrong at the top, and the reason is that **the panel is the legend**: a legend whose
rows move between countries is one you re-read at every country instead of learning once, and
between-country comparison is the thing this map exists for. `unaffiliated` alone was second in
Poland, first in Czechia and Estonia, fifth in Romania, and absent from the US.

The order, by node id, is in `ROOT_ORDER` in `index.html`:

```
unaffiliated  christianity  islam  judaism  hinduism  sikhism  buddhism
secular  unchurched  other      … then everything else, in branches.py order
```

— the answers a census actually collects, largest first; then the other world traditions; then
the two rows that are positions rather than affiliations (`Secular and ethical`, `Believing, no
church`); then the residual container. Families a country does not report are dropped by
`PRESENT` as before, so nobody sees an empty row.

What this gives up is "the biggest thing is always at the top". Worth it: the sizes are on the
rows for anyone who wants the ranking, whereas "where is Islam in this country" cannot be
answered by searching a list that moves.

**It changes no colours.** Root hues are authored in `ROOT_HSL` (§6.3) and `buildPalette` walks
`NODES` in file order rather than panel order, so display order and hue order are independent at
the top level — which is exactly why this could be changed without re-tiling. Below the top they
are still the same sort, deliberately (§6.8).

### 10.1 What the panel says about itself — DECIDED 2026-09-03

Four things, all of them found by looking at the panel rather than reasoned to, and all of them
about the same failure: the panel showed *what* without showing *how much of what*.

**Selecting a node shows one level below it, and no more.** `depth` counts levels below the scope
and carried the unscoped default of 2 into a selection, so clicking Christianity drew its
grandchildren — every Anglican and every Catholic body at once, eighty rows against a question
("what are the Christianities") that has twenty-seven. One level answers it; `+` is still there
for the next one. Two stays right when nothing is selected, because there the first level is
thirty families of which one is 84% of the map (§6.1), so the unscoped view has to start a level
down to say anything at all.

**A white line at the top of the panel names the cut: `Viewing: Adventist subgroups (L3), United
States`.** Group, level, country. The levels are numbered from the families — Christianity is L1,
Lutheran L2, the Missouri Synod L3 — and the number is read off the drawn set rather than computed
from `depth`, because the two differ wherever the tree runs out early. Select Islam, which no
source in the archive divides, and `depth` says one level below while the drawn set is Islam
itself; the line reads `Viewing: Islam (L1)` because that is what is on the map. It is the only
white text in the panel, and it exists because until it did there was no way to tell Adventist's
own row from a view *of* Adventist's subgroups.

**A rule marks the selection and nothing else.** Rules used to sit on every lineage caption, which
drew a line across the middle of Christianity between Jehovah's Witnesses and "no single line" —
a division of a family that no reader has a use for, in the same weight as a division between
families. The division worth drawing is the one the map is actually filtered by, so two rules now
bracket the selected subtree, above its row and below its last descendant. The captions stay; they
were never the problem.

**Outside the selection the names go quiet.** §6.7 removes those dots from the map entirely, so
their rows are a list of things that are not drawn, and they were competing at full contrast with
the rows that are. Dimmed, not removed: they are still the way back out.

*(Also 2026-09-03, and not a panel decision: every dot radius is 30% larger than its stop table
says, as a single gain on the base rather than a new slider default, so the control still reads
1.0× at rest and the zoom curves stay readable as themselves.)*

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
6. **How bright should `unaffiliated` be?** `check_palette.py` flags it at contrast **2.6**
   against the basemap (with `daoism` 2.3 and `unification` 2.7), and it is now the majority
   answer in two drawn countries — 68.3% of Czechia, 58.4% of Estonia — at
   `HSL(234, 26%, 41%)`.

   **It is legible when drawn**, which is worth stating because it was briefly written up
   here as a defect on the strength of a screenshot where the no-religion layer was simply
   switched OFF. Czechia looked empty for that reason and not for this one. The correction
   matters more than the original point: the tool's contrast number is a note to watch, not
   evidence of a failure, and the way to test it is with the layer shown.

   What is genuinely true and worth keeping in view: Poland and Czechia have the **same dot
   density** — 0.096 vs 0.093 dots/km², three percent apart — and read as completely
   different maps. That difference is entirely composition, which is R1 working. Poland's
   89.9% Catholic sits at `HSL(50, 92%, 64%)`, the brightest hue on the wheel, against
   Czechia's darkest, so the contrast between the two countries is somewhat flattered by the
   palette even though both are readable on their own.

   Open, therefore, as a question of degree rather than a bug: whether a lightness nearer 50
   would serve the irreligious countries better without unbalancing the rest. ROOT_HSL is
   hand-authored (§6.3) and stays that way.

## 12. Adding a country — the playbook

**This section is for whoever adds the next country, and it is meant to be added to.** If you
find a trap that is not here, put it here before you finish, even if it seems obvious in
hindsight — most of the entries below cost an hour each and would have cost five minutes to
read. Keep it to things that *generalise*; a fact about one country belongs in its
`sources/<cc>.md`. As the list of easy countries shrinks the rate of new tricks will drop,
which is fine — a short section that stays true is better than a long one that rots.

Seventeen countries in, this is what the work actually looks like.

**Read §14 before starting a country whose state does not publish religion, or whose religious
minorities are persecuted.** The question of whether a country should be drawn at all, and at
what resolution, is prior to every technical step below — and §14 asks you to raise it with
Anita rather than settle it yourself.

### The order that avoids wasted effort

1. **Find the data and check it goes deep enough**, before anything else. The killer
   question is not "does this country ask about religion" but "does it publish the answer
   at a fine geography" — see §3.9. Look at a sample of the actual table.
   **And ask what instrument produced the category list**, because it may not be a
   classification at all: Germany's three categories are the set of corporations that levy
   church tax, a fact about public law rather than about religion, and no better table
   exists to find (§3.9a). Where the list comes from a register rather than a question,
   its ceiling is fixed and hunting for depth is wasted effort.
2. **Check the boundaries exist and join**, second. A source with no joinable geography is
   not a source. Do this before writing a normaliser, not after.
3. Then normaliser → taxonomy → `countries.py` → scatter → tile → docs.

### Finding the data

- **Try a PxWeb API before anything else.** `https://<host>/api/v1/<lang>/<db>/` returns a
  JSON tree you can walk. It is the Nordic/Baltic standard and Estonia took three minutes
  against three days of hunting for Slovakia. `POST` with
  `{"query":[...],"response":{"format":"json-stat2"}}` and `"filter":"all","values":["*"]`
  to get everything.
- **json-stat2 is a flat cube, not a table.** One `value` array in row-major order over the
  dimensions in `id`, sizes in `size`. Read it as rows and you will silently transpose the
  data. Compute strides.
- **PxWeb is not a Nordic/Baltic thing.** North Macedonia's `makstat.stat.gov.mk` runs it and
  was listed as "unchecked" for a day while being one request away. Try `/pxweb/api/v1/en/`
  and `/api/v1/en/` on any office before concluding anything.
- **Do not search a PxWeb tree with a depth-limited keyword walk.** North Macedonia's
  religion table is five levels down and a bounded walk returned ZERO hits on a database that
  has it. Worse, a *national-only* religion table sits in a sibling folder, so a shallow
  search finds the wrong one and suggests the country publishes religion with no geography.
  **The same variable routinely appears at several geographies in different folders** —
  enumerate the census branch in full and compare before choosing.
- **Fetch the table in more than one language when the join and the taxonomy want different
  ones.** North Macedonia needs BOTH: the English edition for the category labels the
  taxonomy keys on, the Macedonian for the Cyrillic municipality names GISCO carries. Neither
  edition alone builds the country, and the English one alone silently makes the join
  impossible.
- If there is no API, the census results are usually a handful of XLSX behind a "results"
  page. Fetch the page and regex out `href="...xlsx"` **with the link text**, because
  offices routinely name the files `Tabel-2.04.xlsx` and put the title elsewhere (Romania).
- **A statistical office's shiny data portal is often a shell.** `podaci.dzs.hr`,
  `data.gov.sk` and `data.stat.gov.rs` all return a JavaScript app, not data. The real
  files are usually on the *old* site (`dzs.gov.hr`) linked from a "Popis 2021" page.
- **Bot protection is a stop sign, not a puzzle.** census2021.bg returns 403 to scripted
  clients. Do not iterate on headers — hand the URL to Anita, who will fetch it in a
  browser. Same for anything Cloudflare-interstitial (the Philippines came from the Wayback
  `id_` endpoint instead). **But re-test a 403 before believing it**: KSH's was gone by the
  time anyone tried again, and a stale "blocked" note reads as a dead end for months.
- **"A JavaScript app with no data endpoint" is a claim about the searcher, not the site.**
  Two cheap tests before writing one off. **(1) Compare the 404s.** KSH's `/api/anything`
  returns 88 bytes of `{"timestamp":…,"status":404,"path":…}` while every other unknown path
  returns the same 2,180-byte HTML shell — a Spring Boot error body IS a live API namespace,
  and content-type plus length distinguishes a router from a catch-all. **(2) Grep the
  bundle for `/api`.** KSH's four routes were plain string literals in `app.js`, which one
  session had already concluded contained nothing. `podaci.dzs.hr`, `data.gov.sk` and
  `data.stat.gov.rs` were all written off as shells and none has had this done to it.
- **An SDMX backend gives you the CODELISTS, which may be the thing you actually need.**
  `/api/structure/<flow>/<version>` returned KSH's category labels in two languages and its
  full geography hierarchy — parent chain included, so a settlement's county came off the
  source rather than out of a boundary file. Recognise the shape: `dataflows`, `structure`,
  `version` in any combination means SDMX.

### Downloading

- **HTTP 200 is not a download** (`sources.md` §5a), and it keeps finding new disguises:
  a truncated file (Czechia), an SPA shell, and — best of all — Maa-amet's
  `linnaosa_shp.zip`, which returns **200 with a 282-byte PNG of an error message**.
  Always assert size *and* type (`zipfile.is_zipfile`, sheet names present) after fetching.
- **A TLS failure can be the server's fault.** `stat.gov.pl` omits its intermediate
  certificate, so curl, requests and certifi all fail identically with "unable to get local
  issuer certificate". That is not a bad URL and not a proxy. Turning verification off for
  one named host *and validating the bytes structurally instead* is the honest fix; say so
  in the script and in COMMANDS.txt.
- **And it can be YOUR fault, with a near-identical error and the opposite fix.**
  `urllib.request` cannot reach `ksh.hu` on this machine — "self signed certificate in
  certificate chain" — while `curl` and `requests` verify it fine. That is a local trust
  store, not a server omission. **The distinguishing test is one line: try a second client.**
  All clients failing means the server; one failing means you. Reaching for the `stat.gov.pl`
  fix on the second case disables verification to route around a problem that is not there.

### Parsing the table

- **Look for in-band sentinels in numeric columns.** New Zealand's `-999` "Confidential",
  Romania's `*` (suppressed) and `-` (true zero). Read cells one at a time, classify them,
  and **raise on anything unrecognised** so a new sentinel cannot appear silently. Never
  `errors="coerce"` a count column: it turns suppression into NaN and the people vanish.
- **Do not assume every sheet in one workbook has the same shape.** Poland's TABL.2/6/7 are
  flat and TABL.5 carries the full 7-level classification; summing it the same way counts
  the Latin rite four times. Where the office publishes a depth column, use it.
- **A table of CODES is not a table of categories, and the plausible reading is wrong often
  enough to be dangerous.** Hungary's exports carry `RE_C`, `RE_CA`, `RE_CO`, `RE_OU` and no
  labels. `RE_CA` is **Calvinist**, not Catholic — Catholic is `RE_C`. `RE_CO` is "Other
  Christian", not Coptic. `RE_OU` is **Ukrainian** Orthodox, a jurisdiction absent from
  KSH's own prose list of the five Orthodox churches in Hungary, so domain knowledge would
  have rejected the correct answer too. **Pin every code against a published national total
  before writing a row**, and re-derive the pinning in `check()` so a reordered codelist
  fails the run instead of silently relabelling the map. Where labels exist at all, read
  them from the source at run time rather than transcribing them.
- **Arithmetic pins STRUCTURE even when no labels exist.** Hungary's three category
  groupings were forced to the person by summation — 11,042 + 7,983 + 3,307 + 7,645 = 29,977
  exactly — before any label was in hand. A hierarchy file deduced that way is stronger than
  Ireland's or Mexico's hand-written ones, and the deduction is worth doing first: it tells
  you what the labels have to mean, which is a check on them when they arrive.
- **Universe rows are not categories.** Every source has some nest of
  total ⊃ answered ⊃ affiliated ⊃ the religions, and drawing an intermediate one doubles
  everything below it. Put them in `EXCLUDED` with a sentence on what they are.
- **Indentation is often the only structure.** Leading dots (Estonia), or *which column* the
  text lands in (Poland). Parse by position, not by matching label text — the labels carry
  trailing "w tym:", embedded newlines, and typos.
- **The office's own typos are part of the data.** Statistics Estonia writes
  `Taara Beliver` in one table and `Taara Believer` in another. Map both in the taxonomy;
  do NOT repair it in the normaliser, because the normalised CSV is supposed to reproduce
  the source verbatim (§2.4).
- **Watch for a percentage twin beside every count column.** Croatia's sheet 2 is
  `Katolici` in column 7 and `Katolici, %` in column 8, for all twelve categories. Taking
  the wrong one of each pair gives a map where every unit holds about 100 people and
  nothing else complains. List the count columns explicitly rather than striding.
- **Headers can be two languages in one cell.** `Katolici Catholics`,
  `Ostali kršćani1) Other Christians1)` — footnote markers included. Pick one language as
  the mapping key and keep it *verbatim*, footnote and all, so the taxonomy key matches
  what the normaliser writes rather than a tidied version of it.
- **Set `sys.stdout.reconfigure(encoding="utf-8")`** at the top of every source script. The
  Windows console is cp1252 and will kill a run on `ł`, `ș` or `õ` — at the *print*, which
  makes it look like a data error.

- **A nested GEOGRAPHY can hide a second universe, and it is harder to see than a nested
  category.** India's C-01 puts state, district, sub-district and town in one column set,
  distinguished only by which code is non-zero — and **town rows are urban-only subsets of the
  sub-district above them**, so summing the file as delivered counts urban India twice. They
  happen to carry only `Urban` and never `Total`, which makes the obvious filter work by luck.
  Assert the property; do not rely on it.
- **Two tables of one census can spell one category two ways.** C-01 writes `Other religions and
  persuasions`; its own Appendix writes `Other Religions and Persuasions` as the parent row inside
  every state block. Matching the parent by label silently failed to recognise it and added each
  state's bucket total as though it were a named religion — 15.7M against a 7.9M bucket. **Match a
  parent on its code wherever the source gives codes.**
- **Excel type inference differs between two files of the same release.** India's state files store
  codes as text (`"00"`), the Appendix stores the same codes as numbers, so `str(cell)` yields
  `"00"` and `"0"` for one code. India's own row became a 36th state and the whole tail doubled.
  Put every code through one zero-padding helper at the point of reading, not at the point of use.
- **It also differs WITHIN one column of one sheet, and then it eats people rather than codes.**
  Germany's Sonderauswertung stores some counts as numbers and some as text in the same column.
  An `isinstance(v, (int, float))` filter — the natural way to skip a sentinel — silently dropped
  **2,228,001 people**, and every national total still looked plausible because the shortfall
  landed in the largest category. Classify every cell through one function that RAISES on
  anything it does not recognise; never filter numeric cells by type.
- **A percentage column is not always count ÷ population, and the difference can be deliberate.**
  Germany's disclosure method perturbs the count and then *adjusts the published share* where the
  perturbed count would give an implausible percentage — Ammeldingen an der Our is 18 people with
  20 Catholics, published as 100.0%. 75 cells disagree by over 0.6pp and every one is a Gemeinde
  of 9–122 people. **Assert the residual in the units the method works in.** Converting the
  disagreement back into PEOPLE bounds it at 3.46; a tolerance in percentage points either passes
  everything or fails the villages, and neither would catch a percentage column read as a count —
  which is the thing the check exists for, and which would be wrong by hundreds of thousands in
  every large city.
- **A source with a publication floor needs its remainder emitted as a category** — *and then that
  category needs mapping.* India's Appendix names a religion only at 100+ adherents nationally,
  leaving 1.9% of the bucket unnamed; without an explicit row for it `allocate.py` normalises
  shares over the named categories and inflates every one by ~2%, silently and in the direction
  that flatters the map. Emitting the row then created the *other* silent failure: it resolved to
  nothing in the taxonomy and `countries.py` dropped 149,668 people without a word, while every
  reconciliation upstream of the taxonomy still passed. `tools/check_mapping.py` caught it, which
  is precisely what the Taxonomy section below says it is for. **Adding a category is a taxonomy
  change even when it comes out of the normaliser.**

### Joining to boundaries — where the real traps are

- **A matching unit count is not a join.** Poland: GISCO's 13-digit `LAU_ID` and GUS's
  7-digit TERYT share no substring, both sides have exactly 2,477 units, and joining as
  delivered matches **zero**. The count check passed while the join failed completely.
  Always print the join **both ways** and fail on either side.
- **Verify a derived key with something independent.** For Poland it was names: 2,476 of
  2,477 agreed, and the one that did not was a real 2021 rename. A wrong offset rule cannot
  produce that.
- **ID formats are per country.** Romania's `LAU_ID` simply *is* the SIRUTA code; Poland's
  needs slicing; Estonia's PxWeb code is a concatenation of EHAK codes. Assert the format
  (length, digits) before slicing, so a reissue fails loudly.
- **Vintage, always** (§8.1). geoBoundaries POL ADM3 is 2017. Four Estonian EHAK codes were
  retired between the 2021 census and the 2024 boundary file. Prefer a boundary set from the
  census year; when you cannot, prove the join instead of arguing about it.
- **Resolve leftovers by elimination, never by guessing.** Romania had 8 unmatched names of
  3,181 (`Râşca`/`Rişca`, `Sfântu`/`Sfântul Gheorghe`); each was the only one left in its
  county, so it is a deduction. Refuse when two or more remain on either side.
- **Derive alias maps, do not hard-code them.** A frozen list of four renames goes stale in
  silence at the next release; a rule that re-derives them fails loudly instead.
- **Watch for the capital in one polygon.** Tallinn is 33% of Estonia, Prague 12.4%,
  Bucharest 9.8%, Warsaw 4.7%. If the office publishes religion for city districts, use
  them and let them REPLACE the parent (Czechia, Estonia). If it does not, leave one polygon
  and say so — subdividing invents structure the source does not have (§3.10).
- **When the CENSUS is finer than the boundary file, that is a different problem and it is
  usually solvable.** Zagreb and Budapest are the same case — religion published per city
  district, GISCO LAU stopping at the city — and Croatia lost it while Hungary won it,
  purely because someone looked in a second place. Budapest's 23 kerület are in
  **geoBoundaries ADM2**, whose Hungarian level is járás and therefore includes them. Check
  ADM2/ADM3 there before accepting one polygon for a capital; and **check the licence per
  level**, because geoBoundaries HUN is CC0 at ADM1 and ODbL at ADM2.
- **Clip a borrowed sub-layer to the parent it subdivides.** Districts from a different
  vintage agreed with GISCO's Budapest on total area to two decimal places and still
  overhung the city edge by tens of metres, which would have put Budapest's dots in
  Budaörs. Intersecting with the parent makes the union exactly the parent; the cost is a
  thin unfilled ring, which a dot map does not care about and a wrong municipality is.
- **A residual geography unit that is EMPTY is a proof, and worth asserting rather than
  filtering.** KSH publishes `Budapest kerületre nem bontható adatai` — figures not
  divisible by district — and it carries no religion rows at all. That absence is what
  guarantees the 23 districts account for the whole city. Dropping it silently would have
  discarded the evidence along with the row; if it ever fills, the assertion fails and the
  map is short by exactly that many people.
- **Eurostat GISCO LAU 2021 is the boundary answer for 34 European countries** in one 98MB
  zip, and its companion LAU–NUTS correspondence workbook carries
  `NUTS3 | LAU CODE | LAU NAME NATIONAL` — which is how Romania, whose census has no codes
  at all, was joinable.
- **Those 34 are NOT the EU27 — candidate countries are in it.** `MK`, `RS` and `AL` all
  have full LAU coverage, so North Macedonia needed no boundary download at all, and Serbia
  and Albania are already solved if their counts are ever found. The correspondence WORKBOOK
  is EU27 and excludes them, which is the asymmetry to remember: polygons yes, code bridge no.
- **A source's population and a boundary file's population measure different things, and
  asserting them equal fails on honest data.** North Macedonia's census counts *residents*;
  GISCO's `POP_2021` does not; the country has lost a fifth of its people to emigration and
  the two disagree by a median 11.7%, worst in exactly the western emigration municipalities.
  **Assert the RELATIONSHIP instead**: a correct join keeps every unit's ratio inside a
  factor of two around a tight median, a scrambled one pairs villages with cities and
  scatters it over orders of magnitude. That is what the check can actually detect. Written
  as an equality it either fails on every real difference of definition or gets loosened
  until it detects nothing.
- **GISCO's `POP_2021` is 0 for seven of Skopje's ten municipalities**, so the MK column sums
  to 1,746,833 against a census 1,836,713. §8.2 means nothing here uses it, but it is a live
  trap for anyone reaching for GISCO population as a weight. Print such holes rather than
  filtering them.
- **Diacritics that look identical are not.** `ş` U+015F (cedilla) vs `ș` U+0219
  (comma-below), and `ţ`/`ț`. INS writes one, Eurostat writes the other, for the same names.
  Fold to ASCII on both sides or a third of Romania silently fails to match — and it looks
  exactly like a vintage problem.
- **"The 2022 boundary file" can be two different files.** BKG publishes a **01.01 and a
  31.12 edition of every year**, and destatis never states which Gebietsstand it published
  on. Against the German census: 01.01.2022 leaves 2 unmatched, 01.01.2023 leaves 10,
  31.12.2022 leaves **none**. Try them all and let the leftovers pick; do not reason about
  which *ought* to be right.
- **A longer key can be the safer one, which is the exact reverse of Poland.** Poland's LAU
  id had to be sliced DOWN to six digits; Germany's 12-digit ARS must not be shortened to
  the 8-digit AGS. The difference is what the extra digits carry: Poland's were a unit TYPE
  the boundary file omits, Germany's are the *Verbandsschlüssel*, which changes when a
  Gemeinde moves between Ämter. Joining Germany on the AGS makes the two leftovers disappear
  and looks like a fix — while orphaning three populated polygons whose people are counted
  elsewhere, placing ~3,000 people in the wrong villages **with every count still
  reconciling**. There is no rule about key length. There is only printing the join both ways
  and asking what the leftovers *are*.
- **"N polygons unmatched" and "N polygons unmatched that are all uninhabited" are different
  findings, and only one is fine.** Germany's 204 leftovers all carry
  `BEZ == 'Gemeindefreies Gebiet'` — forest, lake and military areas with no residents and so
  no religion row. Assert the property; a populated polygon landing in that pile is a silent
  hole in the map.

- **Look for the sub-level before accepting that a unit has no geography.** India publishes units
  called `Area not under any Sub-district` — 17.4M people, including the whole Kolkata
  metropolitan fringe — for which no polygon exists at that level, and whose district's polygons
  tile it completely, so there is no leftover shape to give them. The census also publishes their
  **town** rows, which sum to the unit's population **exactly, 100.0%**, and every one of those
  towns has a polygon. A census that publishes a residual usually publishes its parts somewhere,
  and the union of the parts is a fact rather than an estimate.
- **A form-gated boundary file may be mirrored somewhere ungated.** SHRUG's own download needs a
  form; the identical parquets are plain GitHub release assets in `yashveeeeeeer/india-geodata`.
  Check for a mirror before treating a form as a wall — and **check the licence on the mirror**,
  because SHRUG's is CC-BY-NC-SA, the first non-commercial source in the project.

### Taxonomy

- **Map to branches, not leaves** (`cz2021.py` is the model). §2.4 defers cross-source
  matching and `source_category` travels on every row, so deepening later costs nothing.
- **Only map to paths declared in `branches.py`.** Nothing validates a country mapping at
  build time except `tools/check_mapping.py <cc>` — run it. An unmapped category is not an
  error anywhere downstream; `countries.py` just drops the rows, so people disappear quietly.
- **Adding a branch under christianity/judaism/buddhism fails the build until it has a
  LINEAGE group.** That is deliberate.
- **EXCLUDED and REVIEW are the deliverable**, as much as MAP is. Every arguable call gets a
  sentence on why, so it can be overturned by someone who knows better.
- Expect the source to disagree with the tree about *where* things go, not just what they
  are called: INEGI files Orthodox Christians under "other religions", GUS files Unitarians
  under Christianity. Follow `branches.py` and record the disagreement.
- **The same write-in string can mean opposite things in two countries, and only the place
  decides.** `animismus` in Czechia is a Western neo-animist self-description and goes to
  `paganism`; `Animist` in Sikkim is an outsider's word for a tribal religion and goes to
  `indigenous`. Likewise `Pagan` in Meghalaya is the colonial-era label for the traditional Khasi
  religion, not the neo-pagan revival. Never map a category on its string alone — look at which
  units it is in first.
- **A parent published BESIDE two of its own children needs the remainder emitted — and the
  remainder must exist at every level the allocation touches.** KSH gives `Katolikus`
  (2,886,619) and, labelled as subsets, Roman Catholic and Greek Catholic, but never their
  77,629-person difference. Drawing the parent too double-counts 2.8M; drawing only the
  children drops 77,629. This is the publication-floor rule again (India's 100-adherent
  threshold) in a shape that does not look like it: nothing is below a threshold, the
  remainder is simply never printed. **The second half is the one that bites**: emitting it
  at the fine level alone silently deletes it at the allocation step, because `allocate.py`
  carries a fine column forward only when some coarse category lands on it — and every
  reconciliation upstream still passes, exactly as with India's unmapped remainder.
- **A new top-level family costs more than it looks.** `ROOT_HSL` (§6.3) is hand-authored for
  thirty roots and its indigo→magenta wedge is already at the 4°-apart limit, so a 31st root makes
  every other small family harder to tell apart. India's Nirankaris and Dera Sacha Sauda are real
  distinct movements and still went to `other.in`, because a group that draws one dot should not
  cost the whole palette a degree.

### Reconciliation discipline

- **Assert what should be exact; report what cannot be.** Totals per level against the
  published national figure: exact. Categories summing to the total: exact only if the
  source neither suppresses nor rounds.
- **Where the source rounds, compute the band from the rounding** rather than picking a
  tolerance that passes. Estonia rounds to base 10, so a sum of n units is within ±5n — and
  assert *that every figure is a multiple of 10*, so the band is never applied on a false
  premise. Same reasoning for Canada's base-5.
- **The check that catches the bug is rarely the one you expect.** Romania's county-header
  misparse (600,861 people double-counted) was caught only because two different counties'
  `Păuleşti` happened to collide into one key. Had they not, the run would have passed.
  Prefer checks that would fail *loudly* on a structural error, not just a lucky one.

### Finishing

- `tiles.py --countries` **REPLACES** the archive and `counts.json` with it. Always pass
  every country that has a `dots_<cc>.geojson` — a short list silently drops the rest of the
  map, and this has already happened once.
- Update `sources.md` (the row, the drawn count, a §9x entry with what generalises) and
  `COMMANDS.txt` (fetch, geo, scatter, the tiles line). `data/` is gitignored, so the
  `.md` files are the only record that survives.
- Run `tools/check_palette.py` after a re-tile: separation is a property of the palette
  *against a country's tallies*, so it changes whenever a country lands.

## 13. Things deliberately not being done

- **No world-history time slider.** cityhistory is that map. Religion over time at this
  granularity is a different and much worse-sourced problem, and mixing them would sink both.
- **No adherent-count aggregation across bases** (§3.1), however tempting the coverage would be.
- **No node invented at ingest.** Unmapped source categories go to a file and wait.
- **No log scale** (§4.1).

## 14. What this map could do harm with — ASSESSED 2026-09-04

Not a legal opinion. It is the standing assessment, so that nobody has to work it out from
scratch, and so the line is drawn before a country is half-built rather than after.

**IF ANY OF THIS COMES UP FOR REAL, RAISE IT WITH ANITA RATHER THAN DECIDING ALONE.** That is
an explicit invitation, not a fallback: a judgement call about who gets drawn and how finely is
hers to make, and flagging one costs a message. It does not need to be a crisis first — "this
country's situation looks like §14" is enough to start the conversation. The same goes for a
source whose terms are unclear, or a group whose safety the resolution might affect.

### 14.1 Where the project stands now

Everything ingested so far is aggregate, published, and lawfully obtained — ASARB is free to
download, US Census and ACS products are government works, CES is CC0. The finest unit anywhere
is a census tract of about 4,000 people and much of §8.4 is really PUMA-resolution, about
100,000. No individual is identifiable in anything here, so no data-protection regime is
engaged. `sources/us_pew.py` scrapes, and is the one input whose terms are worth a second look
rather than an assumption.

PL 94-521 bars the **Census Bureau** from asking about religion. It does not restrict anyone
else from estimating it, and §8.4 is not in tension with it.

### 14.2 The three real risks, ranked

1. **§8.4 is substantially a race map wearing religion's labels.** Ethnicity is the strongest
   input, so in many places the pattern drawn IS the ethnic pattern relabelled. Modelling that
   correlation is ordinary demography — Pew, PRRI and Brandeis all do it — and the danger is
   presentational rather than methodological. If a reader takes the inference for an
   observation, the map quietly teaches that every Black neighbourhood is Black Protestant and
   every Mexican one Catholic, which is false about individuals and increasingly false about
   groups. The about-panel text saying so is load-bearing, not boilerplate.

2. **Getting a community wrong is itself a harm, and the failure is asymmetric.** Drawing
   Borough Park as 58% Catholic did not merely mislead, it erased the most visible Jewish
   neighbourhood in America (§8.4's language section). In the other direction, overstating a
   minority somewhere feeds a genre that already exists: "Muslim population maps of Europe" are
   a staple of the far right. That is not a reason not to draw the map. It is the reason the
   honesty of the labelling matters more here than on an ordinary data map.

3. **Targeting.** A neighbourhood-resolution map of where Haredi Jews or Muslims live is in
   principle useful to someone with bad intent. The marginal risk is genuinely low where the
   map REFLECTS what is already public — Borough Park's character is visible from the street
   and in every guidebook. It rises where a map would REVEAL: a small, dispersed or deliberately
   unadvertised minority. Some luck helps here, in that the model is least confident about
   exactly those groups, but luck is not a policy.

### 14.3 Other countries, and the distinction that actually matters

**What §8.4 did is much weaker than "estimating religion where it is not recorded", and the
difference is the whole argument.** The US has a real count at county level; §8.4 changed only
WHERE INSIDE a county the dots sit, and every county total is still exactly ASARB's. Nothing
was invented, only placed.

France has no count at all. Estimating religion there would mean inventing the magnitude as
well as the location, most plausibly from surnames, origin or nationality — far less accurate,
and much closer to the thing France's ban on ethnic statistics exists to prevent. That is a
categorically larger claim and it would deserve the criticism it attracted. **The rule that
falls out of this: never model at a finer resolution, or a stronger claim, than the source
publishes its magnitude at.** A country with no religion data is a country this map does not
draw, which is what it already does.

On the law, briefly: France's prohibition and GDPR Article 9 both bite on **processing personal
data**. An estimate about an area, built from published aggregate tables, is probably outside
them; building the same model from individual-level microdata — the French equivalent of what
CES supplied for §8.4a — is squarely inside. Germany is the opposite case and publishes religion
itself, for church tax.

**The genuinely dangerous list is ethical rather than legal**, and it is short: China, Myanmar,
Iran, Pakistan, and increasingly India, which is already drawn. A fine-grained map of where a
persecuted minority lives is a different object from a map of American denominations, whatever
any statute says. For a group facing persecution, do not go below what that state itself
publishes.

### 14.4 The rules that follow

- Never estimate a magnitude a source does not publish; refine placement only.
- Never estimate religion from ethnicity in a country with no religion count. Say the data does
  not exist — the map already handles that by not drawing the country.
- For a persecuted group, no resolution finer than the state's own publication.
- Keep the measured / fitted / authored / uniform distinction visible to the reader (§7, §8.4).
  It is the difference between "counted", "inferred" and "we do not know", and it is the main
  thing standing between this map and the genre in §14.2.
- Jurisdictional detail changes and none of the above is legal advice. If this ever becomes
  commercial or draws institutional attention, that is a lawyer's question — and §14's opening
  line applies well before then.
