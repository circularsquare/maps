# religiondots — how it works, and why

Interactive world map of religious affiliation as dots, at the finest branch/sect granularity
each region's data supports, with a religion genealogy panel that doubles as the legend.

## Which file is which

| file | what it holds | mine? |
|---|---|---|
| `spec.md` | this file — every design decision and the reasoning behind it | yes |
| `sources.md` | per-source inventory, and §9a–§9x, the running log of what each ingest taught | yes |
| `COMMANDS.txt` | how to rebuild anything, and the add-a-country checklist | yes |
| `todo.txt` | **Anita's.** Not mine to edit ([[feedback_todo_files_are_hers]]) | no |

**Do not put a country count in this file.** The one that used to stand here said "nineteen"
long after it was wrong, and every figure quoted in a section below is true only of the day it
was measured. `COUNTRIES` in `countries.py` is the registry, `counts.json` is what the current
archive actually holds, and `sources.md` §9 is the per-country record. Sections still marked
PROPOSED or OPEN are the design half of this file; everything else describes something that
exists.

## Read this first — where to go for what

**You almost certainly do not need to read this file end to end.** Find your job below.

| doing this | read |
|---|---|
| **adding a country** | **§12 first** (the playbook — the traps that cost an hour each), then §3.1 basis, §3.9 depth trade, §8.1 boundaries, §14 if the state is repressive |
| deciding whether a country may be drawn at all | §14, and **raise it with Anita** — §14's opening line is an invitation, not a fallback |
| changing colours or the palette | §6.3 → §6.9 → §6.13 → §6.14. Run `tools/check_palette.py` and `tools/check_overview.py` |
| working on the legend / tree panel | §6.6, §6.10, §10 |
| working on the viewer or its performance | §4.2d (the scatter is a custom WebGL layer, not tiles), §6.2's Auto rules, §9 |
| touching how dots are counted or placed | §4.1, §4.1a, §4.1b, §3.2, then the §8.2 family |
| a country draws nothing / draws the wrong legend | `COMMANDS.txt`'s checklist — three build steps fail silently |
| understanding what a number on this map means | §3.1 (basis), §7 (confidence tier) |

**The section numbers are stable ids, not an ordering.** They are never renumbered — sources.md,
COMMANDS.txt and several docstrings reference them — so they are not chronological, and a later
section routinely reverses an earlier one. Sections are now filed in numeric order regardless of
when they were written. **The status word in a heading is the truth**: DECIDED, BUILT, OPEN,
REVERSED, SUPERSEDED, REJECTED, PARKED. Where a section has been overturned, its heading says so
and the surviving part is stated first.

## Contents

**§1 What the map has to do** — R1–R4, the four requirements everything is checked against.

**§2 The taxonomy is the spine** — DECIDED
- 2.1 Two relations: `parent` (containment, a tree) and `from` (descent, a DAG)
- 2.2 Where the tree comes from — hand-seeded, not Wikidata
- 2.3 Source categories are not all the same kind of thing
- 2.4 The first tree — 428 nodes, and the three checks that earn their place
- 2.5 The Catholic Church sits at a different depth in the US — DONE
- 2.6 A school is never assigned from outside the source — RE-CONFIRMED, and the registry that
  says 99.9% is not the only registry

**§3 Counting rules** — the R4 answers
- 3.1 Every figure carries a `basis`, and bases are never mixed
- 3.2 The tree is a partition; residuals are computed everywhere and drawn
- 3.3 Syncretism gets a node, not a split
- 3.4 Structure from the detailed source, totals from the recent one — BUILT (Brazil)
- 3.5 Undercounting is marked, not filled — and say which way the hole leans
- 3.5a The United States is re-based on self-identification — BUILT
- 3.5b England, and a source whose COVERAGE varies along the axis you are mapping — BUILT
- 3.6 A roll counts the institution's location, not the member's
- 3.7 A census counts households, and monasteries are not households
- 3.8 Disclosure control biases rare categories downward — perturbation, rounding, suppression
- 3.9 Category detail and spatial detail trade off inside one source
- 3.9a A register does not make the trade at all — Germany
- 3.9b There is NO minimum unit count — DECIDED, and it withdraws a floor two scouting notes had invented
- 3.10 Reuniting fine categories with fine geography — allocation, and what it costs
- 3.10a Built for Australia · 3.10b Canada and two checks · 3.10c `--within` (India) ·
  3.10d Arithmetic consistency is not evidence of meaning
- 3.11 Reducing "other", and the floor under it

**§4 The size problem** — eight orders of magnitude on one map
- 4.1 Dot count stays linear in people
- 4.1a Fractions carry along a spatial order — the Hilbert carry
- 4.1b People per dot is a setting: two editions, 1:1,000 and 1:10,000
- 4.2 Zooming out merges dots, it does not drop them
- 4.2a `tiles.py`, and why not tippecanoe · 4.2b Consolidation is toggleable ·
  4.2c Draw order is randomised · **4.2d The unmerged dots leave the tile pyramid** — BUILT
- 4.3 Presence rings: a second grammar that carries no magnitude
- 4.4 Sources that answer the question backwards — they feed rings

**§5 Reading a region at a glance** — the merge does it; no glyphs

**§6 Two colourings, not one** — the largest section, and the most amended
- 6.1 What building it changed
- 6.2 One legend per country — presence pruning stands; the per-country *palette* is reversed
  by §6.8. **Also holds the Auto country picker and its six tests.**
- 6.3 The family palette is authored, and the depth cut is flat
- 6.3a Not-a-religion is one grey ramp · 6.3a-i `unrecorded` (Germany) · 6.3a-ii `unknown` (Vietnam)
- 6.4 The middle level — ANSWERED by §6.9
- 6.5 Colour follows descent, not size — `LINEAGE`
- 6.6 A branch that carries dots is a category — no dot on the map is grey
- 6.7 Out of scope is hidden, not dimmed
- 6.8 One palette for every country — REVERSES §6.2
- 6.9 Two palettes again, split by scope and not by country — AMENDS §6.8
- 6.10 A row too small to be worth a line folds into one · 6.10a the denominator is the VIEW,
  not the parent — REVERSED
- 6.11 The reader gets the hand overrides too — the swatch picker
- 6.12 An empty map means two opposite things, so say which — the coverage wash
- 6.13 Christianity's branches are authored, not allocated
- 6.14 The overview draws a lineage group as ONE colour · 6.14a the bronze ·
  6.14b the gold is reserved · 6.14c the band ends at 80
- 6.15 A branch whose `unspecified` row is its own child — `MERGE_OWN`

**§7 Confidence is carried, not drawn** — REVERSED: never in colour
- 7a The non-colour treatment — `inferred dots` is a MODE — BUILT
- 7a-i The toggle ROLLS UP to the nearest MEASURED ancestor, it does not hide — BUILT
- 7a-i-1 …and the target is the SOURCE'S OWN COLUMN, not an ancestor — BUILT
- 7a-ii The hit test and both hover cards had not followed the roll-up — FIXED
- 7a-iii A country that measured NOTHING draws nothing — China and Switzerland — BUILT
- 7b The US residual is `modelled`, not `derived`
- 7c The header says what kind of map this is, and how much of it we made up — BUILT
- 7d `fill` names the table each derivation came from; `gap` moves in; the about panel — BUILT

**§8 Pipeline**
- 8.1 Boundaries must be the vintage the data was *published* on
- 8.2 Placement needs no population data — the fine-unit trick
- 8.2a India is the first country the trick does not work on
- 8.2b Germany needs no trick at all — placement is *measured*
- 8.2c Administrative units own water · 8.2c-i some people live on the water
- 8.2d Brazil is the second customer for `place_weight`
- 8.2e A population grid has a resolution floor — FOUND, and Kontur stops paying below ~1 km²/unit
- 8.3 Placing dots by church location — TRIED AND REJECTED
- 8.4 Placing dots by demographic composition — BUILT · 8.4a the residual gets its own model

**§9 Viewer** — MapLibre, the dark ancestrydots style, and what tiling took away ·
9a Auto's minimum-dots floor made every country under ~150k invisible to it — FIXED ·
9b Auto will not enter a country that has none of what is selected

**§10 The tree panel** — 10.0 fixed family order · 10.0a the grey family is contiguous ·
10.1 what the panel says about itself · 10.2 the settings are segmented pairs ·
10.3 a share bar per row, and a column of checkboxes ·
**10.4 one bar for the whole scope, and a hatched segment for what nobody counted** ·
10.4a half of the undrawn share is computed, and subtracting from a modern population is not

**§11 Open questions** — mine to resolve with a prototype; Anita's are in `todo.txt`

**§12 Adding a country — the playbook.** **No country is closed for good** · finding the
data · downloading · parsing · joining to boundaries · taxonomy · reconciliation · finishing.
**Meant to be added to.**

**§13 Things deliberately not being done**

**§14 What this map could do harm with** — 14.1 where it stands · 14.2 the three risks ·
14.3 reflect vs reveal · 14.4 the rules · 14.5 religio-ethnic derivation · 14.6 China ·
14.7 the Han as a grey residual (NOT BUILT) · 14.8 the gap (closed by 14.9) ·
14.9 the mixed-group ban is a preference, not a ban · **14.10 the map may run the model itself** ·
**14.11 France is drawn, and what that does to 14.3** · 14.12 a fractional-share model is not
uniformly trustworthy (Kazakhstan) · **14.13 CFPS refused — the Han residual is unblocked, and CGSS
is checked and rejected** · **14.14 China is 100% drawn, and a threshold over an interested source
is not a rule** · **14.15 what is left for China, ranked — and CFPS was the wrong target all
along, on §3.1 grounds**

---

## 1. What the map has to do

Four requirements, from the brief, restated so they can be checked:

| | requirement | test |
|---|---|---|
| R1 | **Composition is readable at a glance.** In any region, how many groups are there and which. | Point at Lebanon at world zoom and read three colours, not noise. |
| R2 | **Granularity down to branches and sects**, not seven world religions. | Beta Israel, Old Believers, Alevis, Mar Thoma, Tenrikyō each have their own colour, wherever a source supports it. |
| R3 | **Very small groups are visible and not misrepresented.** A 20-person monastic community should be findable; it must not read as a town. | A Carthusian charterhouse can be found on the map; nothing on screen implies it has more people than it has. |
| R4 | **Double- and undercounting are bounded and declared**, not silently averaged away. | Every drawn dot traces to one source figure on one stated basis; every modelled region is marked as modelled. |

R3 and R4 are the two that make this hard, and the two most religion maps get wrong. §4 and §5
are the answers.

R1 and R2 pull against each other — 200 distinguishable colours do not exist. §6 is the answer
(hue = family, shade = branch, and the genealogy tree is the key).

## 2. The taxonomy is the spine — DECIDED

One file, `taxonomy/religions.json`, is the single source of truth for:

1. **the tree** — every group's parent, so counts nest;
2. **stable ids** — `christianity.catholic.latin.jesuit`, dotted path, never renumbered;
3. **colour** — derived from position in the tree, not stored per group (§6);
4. **the genealogy graph** — the side project, and the legend, and the selection model.

Everything else in the repo refers to groups only by these ids. A source that reports a category
we have no node for does not get a new node invented at ingest time; it goes to `unmapped.csv`
and waits for a decision. Silent node creation is how a taxonomy turns into 900 near-duplicate
leaves.

### 2.1 Two relations, not one

The tree and the genealogy are **different graphs over the same nodes**, and conflating them is
the first mistake available:

- **`parent`** — the *containment* relation, a strict tree. Used for counting and for colour.
  "Every Jesuit is a Latin Catholic" is a statement about people alive now.
- **`from`** — the *descent* relation, a DAG with dates. Used by the genealogy panel, and since
  2026-09-03 in a coarse linear form (`LINEAGE` in `branches.py`) for the *order* colours are
  allocated in (§6.5). It is many-to-many: Sikhism draws on both Hindu and Islamic currents, the
  Reformed churches have several parents, Mandaeism's parentage is disputed.

The tree must stay a tree because §3's arithmetic depends on it. The genealogy must be a DAG
because history is one. Edges carry `{from, to, year, kind}` where kind ∈ {schism, reform,
revival, syncretism, revival-of-extinct, disputed}, and `disputed` is a real value that gets
drawn differently — a dashed edge — rather than a claim we quietly pick a side on.

### 2.2 Where the tree comes from

Seeded by hand from a small number of reference works, not scraped. Wikidata's `subclass of`
(P279) over religions is **not** usable as the tree: it mixes containment with influence, has
cycles in practice, and its depth is wildly uneven. It is worth harvesting as a **candidate
list** for missing nodes and for genealogy edges (`P144 based on`, Q126287984 `religious
schism`), reviewed one at a time. That is a `tools/` scan, not a build stage.

**Depth is uneven by design and that is correct.** Christianity in the United States can run
five or six levels deep because ASARB enumerates 372 bodies by county; Chinese folk religion is
one node because nothing enumerates it. The tree records what is *countable*, not what exists,
and it will look lopsided because the world's statistical agencies are.

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
double count against *Hindu Temples* and assert something false about 437,000 people.

**And the mapping cannot be automated on names.** "Orthodox" appears in this one file across
Eastern Orthodoxy, *Orthodox Judaism*, *Orthodox Presbyterian Church*, *Orthodox Anglican
Church*, *Orthodox Mennonite Church* and *Orthodox Old Roman Catholic Communion* — six unrelated
families. The mapping is by group code, by hand, once per source.

### 2.4 The first tree — built 2026-08-27 from one source

`taxonomy/` holds the working tree. Three hand files and a validator, which is the shape the rest
of the repo uses:

| file | what it is |
|---|---|
| `branches.py` | **68 internal nodes**, source-independent, each with a label and the reasoning where it needs one |
| `usrc2020.py` | the 372 ASARB codes → leaf ids, plus `REVIEW` (24 arguable calls, each with its reason) and `UNMAPPED` (1) |
| `build_tree.py` | five checks, then emits `religions.json` and fills `path` in `usrc_groups.csv` |
| `religions.json` | generated — **428 nodes, 68 branches, 360 leaves, depth 4** |

**The arithmetic reconciles exactly**, which is the check worth having: adherents rolled up the
tree total **160,786,973** against ASARB's national **161,224,088**, and the difference is
**437,115 — precisely the one category held off the tree** (*Hindu Yoga and Meditation*).

**The three checks that earn their place** in `build_tree.py`, because a taxonomy fails silently:

1. **a leaf whose parent branch does not exist is an error** — the only way to add a branch is to
   add it to `branches.py` deliberately;
2. **duplicate group codes are read out of the file text, not the dict** — a Python dict literal
   silently keeps the last of a repeated key, so a mapping could be quietly overwritten and the
   totals would still look fine;
3. **every code in the source data must be mapped or explicitly unmapped**, so a new source
   release with new bodies fails loudly instead of dropping them.

**Where the depth actually is.** 50 Mennonite leaves, 28 trinitarian Pentecostal, 22
Presbyterian, 19 Lutheran, 15 Schwarzenau Brethren, 13 canonical Orthodox jurisdictions. The
Anabaptist branch is **87 bodies totalling 846,198 people** — an average of under 10,000 each.
That branch alone is the R2 and R3 case: the finest religious granularity available anywhere in
the world, and almost all of it below or near the dot floor.

**And the non-Christian side is nearly all rings.** Eleven of the seventeen top-level families —
Sikhism, Jainism, Zoroastrianism, Shinto, Daoism, New Thought, Spiritualism, Unification, Hebrew
Israelite, secular/ethical — are **exactly one body each with no adherent count**. On the best
religion dataset in the world, most of the world's religions are a single unquantified row. That
is the shape of the problem this map is trying to show.

**The tree grows where a source reaches, and nowhere else.** ASARB has one Shinto row, so
`shinto` is a leaf today; Japan's 宗教統計調査 enumerates shrine associations by sect and will
push depth under it. A family is shallow because *this* source is shallow there, not because the
family is simple.

**Cross-source denomination matching is deferred, deliberately** (Anita, 2026-08-27). Deciding
that a body in the US file and a body in the Canadian file are the same body is a later task. The
consequence to hold to meanwhile is that **the first source into a branch defines its shape**, and
later sources map into it and leave their genuinely hard cases in `REVIEW` rather than forcing a
merge. `sources` on each node is a dict keyed by source id precisely so a node can accumulate
several without either being lost. (This finally paid off in 2026-09-05 — see §12's taxonomy
section.)

### 2.5 The Catholic Church sits at a different depth in the US — DONE 2026-09-03

Four sources name the same body and three of them agree: `usrc2020` files code `081` on
`christianity.catholic.latin.`**`catholic-church`**, while `ca2021`, `cz2021` and `br2010` all
file it on the branch `christianity.catholic.latin`. That leaf was the **only** child under
`…catholic.latin` anywhere and held 61,858 US dots, over a third of the country — so in the US the
branch and its single child were the same people at two depths.

**The fix was one line** — `"081": "christianity.catholic.latin"` — since `build_tree.py` already
supports a mapping pointing at a branch (the branch takes the `sources` entry, as Islam and
Sikhism do), and leaves are generated from the mappings, so the node disappeared. Nothing was
lost: the US has no other Latin-rite body, and Czechia already folds SSPX into the same branch.

**`tools/scan_identities.py` is the scan that found the second case**, and it is worth keeping
because this class of defect is invisible in the taxonomy and obvious on the map. It lists every
branch with a single child, every branch where one side of a split holds under 2% of the total,
and each side's dots per country. Run it when a source lands.

| branch | the child that was the same thing | why |
|---|---|---|
| `christianity.catholic.latin` | `…latin.catholic-church`, 61,858 dots | the case above |
| `hinduism` | `hinduism.temples`, 831 dots | **"Hindu Temples" is ASARB's row for Hindus, counted by temple** (§2.3) — a building type standing in for a tradition, and every other source files Hindus on the branch |

Both are now mapped at the branch. `hinduism.vedanta` is the only other child there and has no dot and
no ring in any built country, so Hinduism is one row in the world view rather than three.

**What the scan says is NOT an identity is the more useful half.** `spiritualism` / Kardecist
Spiritism (96% of the branch), `christianity.pietist` / Evangelical Covenant (94%),
`…baptist.landmark` / American Baptist Association (98%) all look identical in the tallies and
are not: each has real siblings merely below the dot floor. **The test is whether a sibling
*could* exist, not whether one is currently drawn.** Presence pruning (§6.2) already draws them
as one row.

**The rebuild was paid the cheap way**: `build_tree.py`, then the two node ids rewritten in
`dots_us.geojson` rather than a full `scatter.py --country us` — exact, because `usrc2020.py`
maps no other code to either branch, so the per-unit group sums and therefore the
largest-remainder allocation (§4.1a) are unchanged. `tiles.py` still has to run, since the ids
are baked into the archive; until it does, those dots carry a node that no longer exists and draw
grey, the one thing §6.6 says must never happen.

### 2.6 A school is never assigned from outside the source — RE-CONFIRMED 2026-09-07, with evidence this time

**The bare-`buddhism` call has now been made seven times** — `lk2024` (Sri Lanka, 15.2M),
`mm2014` (Myanmar, 45.2M), `kh2019` (Cambodia, 15.1M), `th2010` (Thailand, 61.7M), `bd2011`
(Bangladesh, 890k), `in2011` (India, 8.4M) and `kr2015` (South Korea, 7.6M → not Mahayana) —
and `kr2015.py` closes its note with *"if that trade is ever reversed, Korea is the country to
reverse it with."* Anita asked the obvious question on 2026-09-07: **is any country so nearly
one school that the map could just assign it?**

**The state of the map is what makes the question fair.** 176,354 of the 185,000 Buddhist dots
sit on the undivided parent — **95%** — while `buddhism.theravada` holds 1,550 and
`buddhism.mahayana` 549. Worse, the fullest school node is filled by the WEAKEST evidence on
the map: China's Dai, derived from ethnicity under §14.5. So the map currently gets *more*
specific as the evidence gets *worse*, and `buddhism.theravada`'s own node note names
"Sri Lanka, Myanmar, Thailand, Laos, Cambodia" before saying Sri Lanka is not drawn there.

#### The evidence that settles it, and it is Thailand's

The argument for assigning was that these are not general-knowledge countries — **the state
itself registers the sangha by school**, which is a §14.10-shaped documented coefficient. That
is true and it is about the wrong thing. Thailand's National Office of Buddhism, 30 December
2023, wats with resident monks:

| | temples |
|---|---|
| Maha Nikaya (Theravada) | 38,934 |
| Dhammayuttika (Theravada) | 4,588 |
| Chinese Nikaya (Mahayana) | **16** |
| Annam Nikaya (Mahayana) | **25** |
| | **43,563** |

41 of 43,563 is **0.094% Mahayana** — the 99.9% the argument wanted. **But the Ministry of the
Interior separately registers 17 Chinese temples, 19 Vietnamese temples and 682 Chinese
shrines**, because a Chinese shrine is not a wat and is licensed under a different law. Count
all Buddhist places of worship and Mahayana infrastructure is **17× larger** than the sangha
registry shows.

**So the 99.9% was an artefact of which registry you read, and §3.6 already says why: a roll
counts the institution's location, not the member's.** Neither registry counts a person, and
**no source anywhere counts Thai Buddhists by school** — not the census, not the surveys.

**The doubt is larger than the node it would fill.** If even 5% of Thailand's 61.7M Buddhists
are meaningfully Mahayana that is 3.1M people, **twice the entire current
`buddhism.theravada`**. And the distinction may not be well formed for the population it turns
on: a Thai Chinese family that visits a wat, a Chinese shrine and keeps Qingming has no school,
and forcing one invents a boundary the way §14.7 refuses to invent the folk/irreligious one.

#### Decided

**Anita, 2026-09-07: leave it as is.** No country's Buddhists are re-filed, and **China's Dai
stay on `buddhism.theravada`** — the option of demoting them to the parent for uniformity was
put and declined, because the Dai case is §14.5's strongest row and the vehicle is part of what
the ethnonym carries (`cn2000.py`'s `Tibetan` note makes the same point).

**What the next session should not redo:** the four-country ranking, if this ever reopens, is
**Sri Lanka cleanest, then Cambodia, then Myanmar, and Thailand weakest** — the reverse of the
order they come to mind in, because the risk scales with the size of the Chinese-descended
population and Thailand's is by far the largest. Korea remains the Mahayana case `kr2015.py`
names.

**The general form, and it is worth more than Buddhism:** *an institutional registry
under-counts a tradition by exactly as much of it as the registry does not register* — and that
share is invisible from inside the registry. Ask which OTHER register might hold the same
tradition under a different law before quoting a 99% from one of them.

## 3. Counting rules — DECIDED

These are the R4 answers. Each exists because of a specific failure mode.

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

**The reason is not fussiness.** Japan's Agency for Cultural Affairs collects adherent counts
from religious corporations and the national total comes to roughly 180 million against a
population of 125 million, because Shintō shrine parishes count residents of the parish and
Buddhist temples count households, and the same person is in both. Those numbers are useful and
they are not a partition of the population. Adding a `roll` figure to a `self_id` figure produces
a number that is not about anything.

**The US/Canada border is where this first becomes visible.** The United States is `roll`: ASARB
counts congregational membership, and **48.6%** of Americans appear on one. Canada is `self_id`:
the census asks, and **53.3%** call themselves Christian with a further 34.6% reporting no
religion. Those percentages are not comparable and their difference is not a fact about religion
in North America. Canada is also a **25% long-form sample**, so it carries sampling error the US
roll does not. §3.1 forbids summing across that boundary; what it cannot do is stop a reader
comparing the two sides by eye, so the unit panel names the basis in words. (§3.5a has since
re-based the US on self-identification, for this reason among others.)

**Two different questions can both be called "religion", and the famous number is often the wrong
one — Northern Ireland.** NISRA asks "what religion do you belong to?" (MS-B19) and, separately,
"what religion were you brought up in?" (MS-B23/24). Q14 was **put only to people who answered
"None" or did not answer the first**, so the brought-up-in table reassigns **181,000 people —
9.5% of Northern Ireland — to a religion they had just said they do not belong to.** The two give
materially different pictures: **42.3 / 37.4** on current belonging, **45.7 / 43.5** on
upbringing, and the second pair is the one in general circulation. Neither is wrong; they answer
different questions and cannot be mixed. Held as a **separate `source_id`**
(`uk_ni_census_2021_brought_up_in`) rather than a variant of the same source, so nothing can group
them by accident.

**`basis` is a property of the row, not of the source.** The US Religion Census is `roll`
throughout except that group code 267 is literally named **"Muslim Estimate"** (4.45M), and codes
890/891/892 (Mahayana, Theravada, Vajrayana Buddhist) and 895 (Hindu Temples) are compiler
estimates for traditions that keep no membership rolls. The compilers did the honest thing and
labelled them; a per-source basis field would have thrown that away. Every ingest maps basis per
row.

### 3.1a Two sources can share a basis and still be incomparable, because their ANSWER SETS differ — FOUND 2026-09-07 at the Kazakh/Russian border

**§3.1 is about the basis, and it is not the only way two countries can fail to be comparable.**
Anita, looking at the finished Kazakhstan (§14.12, `sources.md` §9aq): *"the kazakh cities near the
russian border seem to have way way less secular people than just across in russia. do we think thats
realistic?"*

**No. Most of that cliff is the questionnaire, not belief**, and the mechanism is worth stating because
nothing in §3.1 catches it — both sides are self-identification, so the basis check passes.

| across ~200 km of steppe | | |
|---|---|---|
| **Omsk oblast** (Russia, Arena 2012) | 13.0% *does not believe in God* | **39.1% *believes, professes no religion*** |
| **North Kazakhstan** (census 2021) | 3.6% *non-believer* | **no such box exists on the form** |

**The single largest cause is a category that exists on one form and not the other.** Arena offers
*"I believe in God (in a higher power), but do not profess a particular religion"* and **24.9% of
Russia takes it** — the second-largest answer in the country. **Kazakhstan's census offers no
equivalent at all.** Its list is the religions, `non-believer`, and `refused to state`. So a Russian in
Petropavl who believes vaguely and attends nothing has three options and takes the first: **85.3% of
Kazakhstan's Russians are recorded Orthodox**, against 43.1% of Russia's whole population on the ROC
row. Russia's biggest non-institutional category has not disappeared across the border — **it is inside
Kazakhstan's Orthodox count.**

**A category absent from a form does not produce a zero. It produces a redistribution, and the map
draws the redistribution.**

Three smaller contributions, none of which is the main one:

* **An excluded non-answer.** Kazakhstan's `refused to state` is 10.3–11.1% in those northern oblasts
  and is not drawn at all (§3.5). Some real share of it is irreligion and nothing published says how
  much.
* **A census against an anonymous survey.** A state enumerator at the door, in a country that requires
  religious groups to register and prosecutes unregistered worship, is not Arena's questionnaire.
  Saying *I do not believe* to each is a different act.
* **A real difference, which does exist and is small next to the rest.** Kazakhstan's post-1991 revival
  attached Islam and Orthodoxy to ancestry more firmly than Russia's did; Arena catches Russia's as
  nominal, which is what its 24.9% *is*. Some of the gap is true. A factor of fourteen is not.

**And the modelled country's own share of it, which is measured rather than guessed.** Kazakhstan is
drawn by a model whose held-out test (§14.12) says it **under-draws non-believers in urban Kazakhstan
by 13.0%** — and the question above is about *cities*. Correcting for that takes North Kazakhstan from
3.6% to about 4.1%. It is a real contribution, it is in the direction Anita noticed, and it is nowhere
near enough to explain the cliff.

#### What follows

**The rule: before reading anything across a border, compare the two forms' ANSWER LISTS, not their
bases.** Two `self_id` sources are comparable only where they offered the same choices. The places this
will bite hardest are the borders where a census meets a survey — Kazakhstan/Russia, Georgia/Russia,
Spain/France — and the tell is always the same: one side has a category the other has never heard of.

**What the map does about it is say so, and that is all it can do.** §3.1 already concedes the point
for the US/Canada border — *"what it cannot do is stop a reader comparing the two sides by eye"* — and
the answer there was to name the basis in the panel. Here the answer is a sentence in Kazakhstan's
`note_public` naming the missing category, because "this is a census and that is a survey" would not
have told a reader the thing that actually matters.

**What it must NOT do is harmonise them.** Folding Russia's `unchurched` into Orthodoxy to match
Kazakhstan, or inventing a Kazakh `unchurched` from the refusal cell, would be inventing an answer
nobody gave — §14.4's rule 1 in the shape it takes when the temptation is tidiness rather than
coverage. The categories are what each state asked. The map draws them and explains the seam.

### 3.2 The tree is a partition, so nesting cannot double count

For every unit, over every level of the tree, the children of a node sum to that node.

**Residuals are computed everywhere, at every level, and drawn.** Wherever a parent total is
known and its children are enumerated, `residual = parent − Σ children` becomes a real node,
`…other/unspecified`, with its own colour and its own dots. A large "Protestant, denomination not
reported" slice in a country whose census only asked the top level is a true fact about the data;
dropping it would make the detailed slices look far more complete than they are — the map would
show a country's Baptists and silently omit the 90% of its Protestants nobody enumerated. This
applies in both directions and at every level, down to national total minus every religion, which
is the unaffiliated-plus-not-stated residual.

**A negative residual is a finding, never a fudge.** If a node's children exceed their parent,
one of four things has happened, all real defects worth chasing rather than clamping to zero:

1. **mixed bases** (§3.1) — a `roll` denominational figure set against a `self_id` parent total.
   The common case; Japan is the extreme;
2. **an overlap** the tree does not model — what §3.3 opens a syncretic node for;
3. **a mis-mapped source category**, sitting under the wrong parent;
4. **the roll is attributed to the institution's location rather than the member's** (§3.6), so a
   unit's figure is a catchment total rather than a resident total.

**Two ways the residual gets defeated, both found in New Zealand.**

- **The agency may have filled it in already.** 15.6% of New Zealand's 2023 religion answers are
  not 2023 answers — 9.2% carried forward from 2018 or 2013, 6.5% imputed. The visible
  consequence is that **"Residual Categories" is zero in all 2,395 SA2s**: the not-stated residual
  has been silently absorbed. **A zero residual does not mean full coverage**, and the detector
  reports nothing precisely where there is most to report. Every adapter should record whether the
  source imputes, because a pre-filled residual must be treated as `derived` (§7) rather than
  measured.
- **In-band sentinels turn suppression into arithmetic.** Stats NZ writes **`-999` for
  "Confidential" in the same integer column as the counts** — 1,326 cells across 112 small SA2s.
  Summed as delivered, Islam comes to **−36,753** and No Religion lands 4% low but entirely
  plausible, which is the dangerous half: one result is obviously broken and the other quietly
  wrong. Every adapter must state its source's sentinel values.

So the residual is not only an output, it is **the main automatic detector of §3.1 and §3.2
violations**, and it is cheap because the arithmetic is happening anyway — but it is only as good
as the two conditions above, and both must be checked per source. Negative residuals go to a
findings list with the unit, node, overshoot and contributing sources. Expect the list to be long;
cityhistory's experience is that the queue never empties and the useful stopping rule is
visibility — work the ones big enough to see.

**A monastic order is a slice of its parent, not an extra.** A Jesuit is one person, counted once,
in `…catholic.latin.jesuit` and therefore *not* in `…catholic.latin.other`. This is the one place
the containment tree feels wrong — membership of an order is a vocation, and a Jesuit is obviously
also a Catholic — and it is still right, because the alternative is a layer whose members are also
counted elsewhere, which is R4's failure by construction. The tree answers "where is this person
counted", and each person is counted once.

**Orders are drawn on top.** Being a slice in the arithmetic does not force being a peer on
screen; an order's mark sits above its parent's dots rather than displacing one — which is also
the natural z-order for rings (§4.3). Drawn on top, an order looks additive even though it was
subtracted; at order scale that is invisible (a 20-monk community against a county of Catholics is
far below one dot), so the two rules do not collide anywhere on screen. If some large body ever
sits in an order-like slot the question comes back.

### 3.3 Syncretism gets a node, not a split

Where dual practice is the norm — Japan, China, Vietnam, Korea, much of West Africa and the
Andes — forcing a partition manufactures a precision nobody has. The rule: **a combination is a
node.** `japan.shinbutsu`, `china.folk` (which includes Buddhist and Daoist practice by
construction, and says so in its description), `vodun.catholic-syncretic`. The partition survives,
and the node name is where the overlap is declared rather than hidden.

**The test for opening one:** a source reports the combination, or reports categories summing past
100%. Not "we suspect overlap".

**Measured for the first time in New Zealand**, whose census accepts up to four affiliations per
person, so responses genuinely exceed people — and by how much depends on how finely you cut:

| | inflation |
|---|---|
| level 1 (11 categories) | **+0.18%** (9,192 responses) |
| level 3 (163 categories) | **+0.70%** (32,886) |
| Christian / Māori religions / Islam, level 3 | +1.24% / +1.64% / +1.28% |
| **No Religion, level 3** | **+0.01%** |

No Religion is the control and it is what makes the rest trustworthy: nobody holds "no religion"
*and* something else, so its inflation should be ~0, and it is. The others are real multiple
affiliation rather than a processing artifact. The inflation is **small enough to draw without
correction** at these granularities — under 1% is well inside §3.8's disclosure-control noise —
and it **grows as categories get finer**, which is the direction this project keeps pushing, so it
is worth re-measuring rather than assuming 0.7% is a ceiling. A country where dual practice is the
norm will not look like this at all, and that is what the syncretic node is for.

### 3.4 Structure from the detailed source, totals from the recent one — BUILT 2026-09-04 for Brazil

The common shape: a recent source has the right total and coarse categories; an older source has
the fine split. Brazil is the case — the 2022 census gives religion and the evangelical share by
município, but IBGE has **not** published the denominational breakdown, while the 2010 census did.
India is the same shape a decade wider.

So: take the 2022 municipal evangelical *total*, split it by the 2010 municipal denominational
*shares*, and record `structure_year: 2010, total_year: 2022` on every resulting figure. The
viewer shows both years. This is an interpolation and is labelled as one; what it must never do is
silently present 2010 shares as 2022 data.

**Four things `br_rescale.py` settled that the paragraph above did not anticipate.**

- **The split is per município, not national.** Each município's 2022 group total is divided by
  *that município's own* 2010 mix, so Assembleia de Deus keeps its regional shape instead of being
  smeared to a national average. Where a município has no 2010 people in a group the fallback runs
  state, then nation — needed for **Umbanda e Candomblé in 3,972 of 5,570 municípios**, a direct
  consequence of its 3.1× rise: most places had none in 2010.
- **The category map is by hand and one entry inverts.** `Outras religiosidades` exists in both
  censuses and means opposite things — 11,307 people in 2010, **7,079,124 in 2022, holding
  Judaism, Islam, Buddhism, the Witnesses, the Latter-day Saints, Hinduism, Orthodoxy and the
  esoteric traditions**, each of which is its own 2010 row. A join on category name produces
  confident nonsense at 626× scale. The map is written out and asserted to partition every 2010
  root exactly once.
- **The tier is not uniform, and "this country is derived" would have been wrong.** Three 2022
  categories map to a single 2010 leaf — Católica Apostólica Romana, Espírita, Tradições indígenas
  — so their counts pass through untouched and stay `measured`: **58.7% of the drawn people**.
  Only the remaining 41.3% is a 2022 magnitude on a 2010 shape. **Tier is a property of the row,
  exactly as `basis` is.**
- **Rescaling changes the universe, and that has to be carried.** The 2022 question was asked of
  people **aged 10 or over**, so Brazil's drawn population falls 190.8M → 176.3M and its shares
  become shares of the 10+. It is not scaled back up, for §14.4's reason.

**And it moves the boundary vintage** (§8.1): the drawn geography is now 2022's 5,570 municípios,
so `br_geo.py --year 2022` replaces the 2010 mesh. Keeping the 2010 mesh would have been the
mirror image of the trap that file was written to avoid.

**Where there is no recent total to rescale to, the old figure is used as it stands** — against
the brief's 2015-or-later preference, because the alternative is worse. India's last religion
census is 2011; Russia's best subnational source is the Sreda Arena atlas of 2012. Leaving them
out means a blank India, which is 18% of humanity, and a blank Russia. They go in at their own
year, the year is on every figure and visible in the unit panel, and **the confidence tier (§7) is
not reduced for age alone** — an old census is a measurement, unlike a modelled estimate, and the
two should not be rendered as though they were the same kind of claim.

### 3.5 Undercounting is marked, not filled

Countries that do not ask: China, Nigeria, France, the United States federally, most of the Gulf.
For these the composition is a survey- or compiler-derived estimate, and it is marked (§7, and
§7a for how). cityhistory dims years with no measurement within ±5; this is the same instrument on
a different axis, and for the same reason: the absence of data is itself something the map should
show rather than paper over.

**AND SAY WHICH WAY THE HOLE LEANS — added 2026-09-05 with Serbia.** Marking an undercount is half
the job; the other half is that dropping a non-response is almost never neutral, and its direction
is *measurable* from the data already in hand. Serbia excludes 355,484 people RZS records as
`Непознато`, and across its 168 municipalities that residual correlates **+0.60 with the
declared-atheist share** — 17.3% in Savski venac against 0.97% in Preševo. So excluding it removes
proportionally more people from the least religious places, and every share drawn for Serbia is
slightly more religious than Serbia is.

The check is one line — correlate each excluded residual against the categories that *are* drawn,
per unit — and it is cheap enough to run on every country that excludes anything. **Nothing
corrects for it**; correcting would be inventing a magnitude (§14.4). What changes is that
`note_public` says which way it leans instead of only how big it is. Czechia's 30%, Hungary's 40%,
Australia's 7% and North Macedonia's 7.2% have never had this asked of them, and Hungary's is
large enough that the answer matters.

### 3.5a The United States is re-based on self-identification — BUILT 2026-09-04

**The finding that forces it.** ASARB's 161.2M adherents are 48.6% of the country, so **171
million Americans — 51.6% — were drawn as nothing at all.** Not as unaffiliated, not as unknown:
absent. And because the residual of a roll means "on no roll", the map could not draw the American
non-religious at all while Canada drew 34.6% of itself that way. The 49th parallel was not a step
in the data, it was a step between two questions.

**The decision (Anita, 2026-09-03): the survey supplies the totals and ASARB supplies the
structure.** This is not a new mechanism. It is §3.4's rule with `basis` where Brazil has `year`;
it is the split §3.1 permits and not the addition it forbids; the leftover is §3.2's residual,
which §6.6 already draws; and §3.5 has been asking for it all along.

`tools/scan_selfid_gap.py` was the feasibility check. Against Pew's 2023-24 Religious Landscape
Study, applied to the whole population:

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
living in a parish's territory, an LDS ward everyone baptised who has not formally resigned, and
both hold people who would tell a surveyor they are something else now. Requiring Catholic to
clear its own self-ID line fails in five states; requiring it only of Christianity fails in Utah
alone. So denominations sit inside the root as structure and are never asked to clear a survey
line of their own.

**The residual is spread over the people who are not on any roll.** Per state and per root,
`residual = self_id − Σ rolls`, distributed across the state's counties in proportion to
`county population − county adherents` rather than to population or to the rolls. That pool is
real per-county data, it is the same pool the unaffiliated come out of, and it puts the
unspecified Christians of New Hampshire where New Hampshire's unchurched actually are instead of
smoothing them over the state.

**Cap and record where a roll still exceeds the survey.** The roll is a measurement of real
congregations and is drawn as it stands; the residual is floored at zero; the overflow comes out
of that state's **unaffiliated** residual, because the likeliest reading of a name on a roll the
surveys cannot find is a person who now says "nothing in particular". Measured over all 51 states
and every root both instruments can see, with the child conversion applied:

| | |
|---|---|
| (state, root) pairs where the roll exceeds the survey | **62 of 255** |
| total overflow | **1,694,041 people, 0.51% of the population** |
| of which Islam | 1,182,008 — **70%**, across 30 states |
| of which Utah, Christianity | 395,685 |
| Judaism + Buddhism + Hinduism, 31 pairs | 116,348 |
| negatives caused by a **true zero** in the survey | 31 of the 62 |

Two readings, pointing the same way. **Christianity is negative in Utah and nowhere else**, across
all 51 states. And the overflow is not really 62 findings: 70% is Islam, where ASARB's figure is a
body literally named *Muslim Estimate* and so two estimates disagreeing rather than a roll beating
a survey, and half the remaining pairs are a survey of 36,908 people returning a true zero for a
small religion in a small state. What is left is small enough that "record it and charge it to the
unaffiliated" remains the right rule. (As built: 94 pairs, 1,630,571 people, every state's
unaffiliated residual large enough to absorb its share.)

**The declaration stays quiet — Anita, 2026-09-03.** One sentence in the country note, the numbers
in the build log and in `counts.json`, and nothing on the map itself. No overlay, no badge, no
second legend. (§7 later removed the desaturation this leaned on; §7a is the replacement, and §7's
closing note says plainly that this decision was left exposed in between.)

**Three things this leaves open, recorded so they are not rediscovered.**

- **The child assumption is load-bearing and must be said out loud.** Adults are 78% of the
  population, so applying adult shares to everyone scales the survey by 1.28×. Catholic then
  clears its roll by 1.1M, under 2%. Applied to adults only it is 49.1M against a 61.9M roll,
  negative by 12.8M. **What this map says about American Catholics rests on an assumption about
  children.**
- **Islam is the calibration case.** ASARB's figure is *Muslim Estimate*, so its −0.5M is two
  estimates of one population disagreeing by 12% — the only place the two instruments can be
  compared with the roll question removed.
- **This does not solve §4.4.** Pew publishes Sikh, Daoist, Bahá'í and Zoroastrian as a single
  "other world religions" line at <0.3%, and Unitarian, pantheist and Wiccan as "other religious
  identifications" at 1.9%; at n=36,000 nothing smaller can be broken out. The eleven
  congregations-only religions still need their own per-body sources.

**BUILT 2026-09-04, `us_rebase.py`.** The map now draws **326,813,748 of 331,449,281 Americans —
98.6%** against 48.4% before. The residual is 166.2M people, so **a little over half of the
American map is a derived row.** Where it lands:

| | | |
|---|---|---|
| unaffiliated | 60.6M | 36.4% of the residual, and none of it drawable before |
| christianity | 54.2M | "Christian, no roll names them" |
| secular | 35.3M | atheist + agnostic + humanist |
| judaism, buddhism, hinduism | 8.4M | |
| unchurched, paganism, esoteric, other.us | 6.3M | |
| unitarianuniversalist, indigenous, spiritualism, newthought | 0.8M | |

Four things the build settled that the decision had not.

- **The residual is measured against the roll AS DRAWN, which is the county sheet.** ASARB's state
  sheet totals 160,786,973 mapped adherents and its county sheet 160,572,400: **214,573 people are
  reported for a state and attributable to no county in it.** Subtracting the state figure computes
  a residual against a roll the map does not draw, and those people disappear from the country's
  total. **A residual must be taken against what is actually drawn, not against the tidiest
  published version of it.**
- **The residual must NOT go through §8.4's weights.** Every beta there was fitted to predict
  ASARB's own within-metro variation, so it says where the people *on* a roll live. Running "the
  people on no roll" through it would place them on top of the congregations they are defined by
  not belonging to. They take tract population and nothing else — `weights(..., plain=True)`,
  which is the general rule: **a model fitted on measured rows may not place derived ones.**
  (§8.4a then gave the residual its own model against its own ground truth.)
- **Thirty counties get no residual at all**, and correctly. §3.6's counties that report more
  adherents than residents have a `population − adherents` pool that is negative, it clips to
  zero, and 3,113 of 3,143 counties receive the spread.
- **Its tier is `modelled`, not `derived`** — §7b, decided 2026-09-05.

**The mapping is a cut, not a category match — `taxonomy/us_pew2024.py`.** "Totals at the root and
nowhere else" makes this the first mapping of its kind here. Every other mapping file answers
"what is this category?" for every category the source publishes; this one answers "where does
Pew's tree have to be cut so each piece lands on exactly one of our roots?" — and it maps **28 of
149 categories**, leaving `southern-baptist-convention` and `global-methodist-church` deliberately
untouched. Most of the cut is Pew's own top level; it descends in two places only, both because a
single Pew node spans several of our roots: `other-christian` (Spiritualism and New Thought are
roots of ours and Pew files them under Christianity) and `something-else` (Unitarian Universalism
to Wicca to Native American religions).

**A cut entry is a set of roots, not one root.** `other-world-religions` is one line covering
Sikhs, Daoists, Bahá'ís and Zoroastrians, so its residual is the line minus the ASARB rolls of
*every* root in it — which subtracts the 178,727 Bahá'ís ASARB counts instead of drawing them
twice. **Reading an irreducible lump as a single opaque bucket is how double counting gets in.**

Two consequences. **Non-response is excluded**, as Czechia's 30.05% and Ireland's 6.7% are, so a
county's drawn total comes to **98.60%** of its population; the share runs 0.09% in Massachusetts
to 4.88% in Alaska, a 54-fold spread, so it is not a uniform haircut. And **atheist and agnostic
go to `secular`, "nothing in particular" to `unaffiliated`** — the line `branches.py` already drew
for Canada's identical answers.

### 3.5b England, and a source whose COVERAGE varies along the axis you are mapping — BUILT 2026-09-07

§3.5a's mechanism, run one level down: not a survey's roots over a roll's structure, but a
**census's Christian total over a church register's geography**. England's 26.2M Christians were
one flat node — the largest unresolved block on the map, §3.11's floor made literal — because
ONS asks one tick box and Wales, Scotland, Northern Ireland and Ireland all split theirs.
They now draw as `anglican` 64.6%, `catholic` 18.2%, `methodist` 4.6%, `baptist` 2.4%,
`reformed` 2.2%, and **8.0% deliberately unplaced**. Three sources, each doing only what it can:
the census keeps every magnitude, a current church register says where each denomination is, the
English Church Census 2005 says how large its congregations are, and the British Election Study
says how many people belong to each. `sources.md` has the build; this is the transferable part.

**THE FAILURE MODE, AND IT IS NOT ABOUT ENGLAND.** Two bugs, both of which produced a precise,
plausible, wrong map with no error anywhere: every total reconciled, every category resolved,
every check the project already had passed.

1. The English Church Census took bulk data from *"ten Church of England and eight Roman Catholic
   Dioceses"*, so its coverage is near-total in some counties and a postal response rate in
   others. Its Anglican response runs **25.9% in Norfolk against 92.5% in Greater Manchester**.
   Placing denominations by its attendance totals, corrected by the national 55% its own user
   guide publishes, drew **a Merseyside that was 72% Anglican and 16% Catholic**.
2. Its settlement code is missing for 37% of churches and the missingness is denominational —
   68% of Anglicans carry one against 54% of Catholics and 6% of Orthodox. Reading a cell's mix
   off the coded subset made every English conurbation twelve points more Anglican.

Both are the same shape: **a source whose coverage varies along the very axis the map is
about.** A source can be complete enough, recent enough, fine enough, on the right basis, and
still be unusable for geography because *what it missed, it missed unevenly.* §3.8 is the
disclosure-control version of this and §3.6 the location version; this is the response-rate one.

**THE TEST THAT CAUGHT IT, WHICH IS THE THING TO REUSE.** Neither bug is visible from inside
the data. What found them was **holding the source against an independent register of the same
thing**: the Church of England publishes a complete list of its own churches, so the church
census's Anglican count per county divided by the real number *is* its response rate, county by
county. It reproduced the published national 55% — which is what says the method works — and
then showed the 3.5-fold spread underneath it.

So: **any `roll` or `attendance` source being used to place people deserves a denominator from
somewhere else.** Ask what a complete count of the same institutions would look like and go and
find one. Where no register exists the source may still be fine, but that is an assumption and
should be written down as one.

**THE REPAIRS, both of which are "use the part that survives".** A mean congregation size
survives an uneven response where a total does not, because losing half a county's chapels
changes how many you saw and not how big they were. And a settlement profile taken *within* a
denomination survives a coding gap that varies *between* denominations, because the coding rate
cancels. In both cases the fix was not to correct the biased quantity but to stop using it.

**WHERE NO REGISTER REACHES, A CENSUS VARIABLE CAN — SOMETIMES.** Pentecostal, New church and
Orthodox had national anchors and no usable geography: a 2005 church census and a map of
buildings both miss congregations that rent halls and parishes founded after 2004.
OpenStreetMap has **355 Pentecostal and 129 Orthodox churches in all of England**; using them
put 6.0% of Norfolk's Christians on Orthodoxy.

**Orthodoxy was rescued by a proxy and the other two were not, and the difference is the
general point.** Country of birth places Orthodoxy well because the mapping from origin to
religion is tight and, crucially, *correctable where it is not*: Romania is 85% Orthodox and
Albania 7%, so weighting each origin by its own census turns a variable about migration into
one about religion, and it moves Albania from 5.9% of the shape to 0.6%. Ethnic group does not
place Pentecostalism, because Black African in England is heavily Anglican and Catholic as
well, and no weighting fixes that — the proxy would say where the Black-majority congregations
are, which is a different question with the same answer in some places and not others. Anita's
call, 2026-09-08: Orthodoxy yes, Pentecostal no.

**So the test for a proxy is not correlation, it is whether the residual is nameable.** An
origin's non-Orthodox share is a number somebody has published. An ethnicity's
non-Pentecostal share is a question nobody has asked. The first can be weighted away and the
second cannot, and a proxy you cannot weight is a proxy you are asserting rather than using.

What is still unplaced stays on the census's own `Christian` category — §3.2's residual, and
§14.4's refusal to invent a magnitude, applied to two legs the map would visibly like to have.

### 3.6 A roll counts the institution's location, not the member's — FOUND 2026-08-27

The US Religion Census attributes adherents to the county of the **congregation**, and people do
not always worship in the county they live in. So **30 of 3,143 counties report more adherents
than residents.** King County, Texas: population 265, adherents 1,199 — **452%**. Stonewall County
TX 167%, Harmon County OK 164%, Harding County NM 156%, Fredericksburg city VA 133%. The pattern
is rural counties with one substantial church drawing from a wide area, plus Virginia's independent
cities, which are tiny polygons surrounded by the county whose residents fill their churches.

**Handling: clamp to zero for display, keep computing it, flag only the big ones.** A negative
residual cannot be drawn in any case. What the decision settles is when it is worth *reporting*,
and the measurement says almost never:

| | |
|---|---|
| counties with a negative residual | **30 of 3,143** |
| total overshoot nationally | **42,892 people = 0.0065% of US population** |
| largest single county | Fredericksburg city VA, 9,189 people (133%) |
| largest by ratio | King County TX, 934 people (452%) |
| under 1,000 people | 21 of the 30 |

The rule: **surface a negative residual as a finding when it exceeds both 5% of the unit's
population and 1,000 people** — about nine US counties, a list someone could actually work — and
otherwise log it silently. Redistributing over a commuting shed would invent a model this project
has no evidence for, and capping *inputs* at population would hide a real property of the source;
clamping the output does neither.

The 0.0065% figure is the yardstick for the next source. A source whose overshoot runs at a few
hundredths of a percent is behaving like ASARB; one running at whole percentage points has a
different problem and the threshold should not quietly absorb it. Worth re-checking whenever a
finer geography than county is used, since the error grows as units shrink.

*(The denominator read 3,144 until 2026-09-04. ASARB's summary sheets end with a blank row and a
`Totals` row whose key is the **string** `Totals`, so `notna()` keeps it — the same slip doubles
the US population if the figure taken is a sum.)*

The general form, which will reach every `roll` source: **a roll is a fact about buildings, and a
dot map is a claim about residents.** Where they diverge, the dots are placed by population grid
inside the unit anyway (§8), so the map is already saying "somewhere in this unit" — the failure is
confined to the unit's total, which is exactly where the residual can see it.

### 3.7 A census counts households, and monasteries are not households — FOUND 2026-08-27

Found in the Philippine census and not a Philippine quirk: census religion tables are generally
tabulated on the **household population**, which excludes the *institutional* population — people
in barracks, prisons, hospitals, dormitories, **monasteries and seminaries**.

The Philippines: household population 108,667,043 against a total of 109,035,343. The gap is
**368,300 people, 0.338%** — small, and composed of exactly the residents this map most wants to
see. R3 asks for monastic communities to be findable, and **the census-shaped half of our sources
structurally cannot see them**, however fine the geography or granular the categories.

This is not a defect to correct; it is a statement about what these sources measure, and it
sharpens why §4.4's location-by-religion sources are load-bearing rather than a nice extra. Every
source adapter should record whether its universe is household or total population, and
`sources.md` should state it per country.

### 3.8 Disclosure control biases the rare categories downward — FOUND 2026-08-27

Statistical agencies perturb, round or suppress published cells to protect individuals. Three
mechanisms, and they get worse in that order.

**1. Perturbation.** The ABS perturbs **every cell of every table independently**. National totals
barely move — Australia's SA2 sums come to 25,422,677 against a published 25,422,788, off by 111
people, 0.0004%. **The error is not distributed evenly, and it lands on us**: Australian
Aboriginal Traditional Religions **−6.29%**, Brethren **−1.71%**, against −0.0004% for the total.
Perturbation is roughly constant in absolute terms, so as a group gets smaller the relative
distortion grows without bound — and small groups are this project's subject.

**2. Rounding, which manufactures residuals.** StatCan random-rounds every count to a multiple of
5 — verified exactly: **0 of 321,757 Canadian counts is not a multiple of 5**. So a parent and the
sum of its children disagree by ±5 to ±25 as a matter of course, two StatCan products disagree by
±5 on the same national figure, and **none of it is an error**. A residual inside the rounding
envelope is noise and must not be chased; §3.6's absolute floor of 1,000 people already covers
Canada's ±25, which is why that threshold has an absolute term and not only a percentage one. Each
source needs its rounding rule recorded so the envelope is known rather than guessed.

**3. Suppression, and it is the worst of them — measured 2026-09-05 with Lithuania.** Perturbation
and rounding move a number; suppression removes it, precisely where the category is rare:

| withheld at municipality | | | |
|---|---|---|---|
| Karaims **60.4%** | Greek Catholics **29.3%** | Adventists **28.2%** | New Apostolics 25.7% |
| Baptists 13.2% | Jews 11.7% | Sunni Muslims 7.3% | Pentecostals 6.9% |
| Roman Catholic, Orthodox, no religion, not stated | **0%** | | |

**Whose total is 1,683 people, 0.06% of Lithuania.** That single number is the reason to report
suppression PER CATEGORY and never as a headline: 0.06% reads as nothing, and the truth is that
the map loses most of one religion and a quarter of three others. `lt.py`'s `check()` prints a row
per category with the count and the number of units it is missing from, which is the shape every
source with suppression should use.

Two consequences. A source's *national* exactness says nothing about what survives to the drawn
tier — check the two against each other per category, which also catches a misparse. And this is
the strongest argument the presence ring (§4.3) has: where a religion is suppressed everywhere but
one place, a dot count is a fiction and "present here" is the only true statement available.
Lithuania's Karaims have lived in Trakai since 1397, Trakai's cell is withheld, and the 101 the map
can draw are all in Vilnius.

**And the in-band trap that comes with it: a null can mean two opposite things.** Lithuania's
`OBS_STATUS` separates *konfidencialūs duomenys* (withheld) from *tokio reiškinio … nebuvo* (a true
zero), and 414 of 1,020 cells are null across both meanings. Reading them all as zero deletes 1,683
people; reading them all as withheld invents structure. **Resolve such a flag on its TEXT, not on
its index** in the attribute array — an index is a fact about one download — and fail loudly if a
third value appears. Romania's `*` and Hungary's `Q` were the one-meaning version of this.

Nothing can recover the true figure, so the response is to record it: **a group's published count
carries an uncertainty that is a function of its size, not of the source's quality.** It is also a
reason not to over-read a small difference between two rare groups in one place.

### 3.9 Category detail and spatial detail trade off inside one source — FOUND 2026-08-27

Not §3.4's case, which is two sources of different vintages. This is **one source, one year, where
the agency publishes fine categories OR fine geography and refuses to publish both**, because the
cross-tabulation would identify people.

Australia is the clean example. The ASCRG 2021 classification has **150 religious groups** and the
ABS publishes all 150 — nationally. At SA2 the same census gives **34**, and everything small is
folded into a single `Other Religious Groups` column of 107,127 people containing Bahá'í, Taoism,
Shinto, Paganism, Wicca, Jainism, Zoroastrianism, Mandaean, Yezidi, Druze, Caodaism, Spiritualism
and Rastafari together.

**Mexico is the same shape and confirms it is structural, not an ABS quirk.** INEGI publishes **24
denominations × 32 entidades**, or **4 aggregate groups × 2,469 municipios**, and there is no table
joining the two. Worse, its classification *codes* 46 denominations and publishes 24 — Mennonites,
Lutherans, Buddhists and Hindus exist in the database and in no released table at all.

So for Australia the map can show *where* 34 categories are, or *how many* of 150 there are, and
the thing R2 wants — where the Yezidis are — is withheld by design. Options, in order of honesty:
take the coarse-geography detail and place it by §3.4's rule, clearly marked as derived; draw the
`Other` bucket as itself, which is truthful and useless; or find a custom tabulation. **Check this
per country rather than assuming the finest geography carries the finest categories — it usually
does not.**

### 3.9a A register does not make the trade at all — FOUND 2026-09-04 with Germany

§3.9 reads like a law about sources. It is a law about **survey** sources, and Germany sits at both
extremes at once.

**Zensus 2022 does not ask about religion.** There is no question on the form. The published
figures are read off the *Melderegister*, which records membership of a public-law religious
society because it determines **church-tax liability**. So the basis is `roll` (§3.1) — and it is
the first source here that is neither a question nor a church's own count, but a **state register
kept for a fiscal purpose**.

| | |
|---|---|
| geography | **10,786 Gemeinden**, and the same figures on a **100m grid, 3,088,036 cells** |
| categories | **three** |
| suppression | 178 true-zero cells of 32,358; nothing withheld |

A register covers everybody exactly and knows almost nothing, so there is no cross-tabulation risk
to manage and nothing to withhold — and equally nothing to reveal. **The finest geography on this
map and the coarsest categories on it have one cause.**

**The general rule: ask what INSTRUMENT produced a category list before treating the list as a
classification.** Germany's three categories are the set of corporations that levy church tax. That
is a fact about German public law, not about German religion, and no amount of looking for a better
table will deepen it — `sources/de.md` §2 records why Zensus 2011, which *did* ask, is worse rather
than better.

**The half of the country this cannot see.** "Sonstige, keine, ohne Angabe" is 51.8%, and destatis
is explicit that the register's entries for *other* public-law bodies cannot "zuverlässig
statistisch abbilden" their membership. So one category holds people in another body, people in no
body, and people with no entry — Germany's roughly four million Muslims, its Orthodox Christians,
its Jewish communities, its Freikirchen and its Alt-Katholiken among them, unrecoverable. That is
§3.5 in its sharpest form, and §14.3's rule forbids the obvious rescue. It gets a node that says
what it is instead — `unrecorded`, §6.3a-i — and the about panel carries the rest.

### 3.9b There is NO minimum unit count — DECIDED 2026-09-06, and it withdraws a rule nobody wrote down

**Anita, 2026-09-06:** *"i think especially for smaller countries, we dont need that many
regions for it to be a cool plot."*

`sources.md` §11d rejected Albania because *"12 units is below North Macedonia's 80, the
current floor"*, and §11j then weighed Jamaica's 14 parishes against that precedent and left
the country unbuilt as *"a judgement call"*. **Both are withdrawn. There is no floor, and
there never was one in this file** — it was invented in a scouting note, cited by the next
scouting note as though it were settled, and never appeared in spec.md at all.

**It was already dead when it was last cited, which is the part worth noticing.** By the time
§11j invoked "Albania's 12-unit rejection", this project had drawn:

| country | units | people per unit |
|---|---|---|
| **Guyana** (§9r) | **10 regions** | 74,700 |
| **Georgia** (§9x) | **11 regions** | 340,000 |
| Greece (§9z) | 14 NUTS-2 | 750,000 |
| Kenya (§9o) | 47 counties | 1,000,000 |

So the rule had been overturned twice by actual builds and survived as text anyway. **A
heuristic that lives only in prose does not get checked against the map**, and this one
outlived its own counter-examples by a day. When a scouting note states a threshold, it is a
note about one country, not a rule — and if it is meant as a rule it belongs here, where a
later section can be seen to reverse it.

#### What replaces it: nothing, and that is deliberate

There is no replacement threshold, on units or on people per unit. §11k already recorded a
softer version of Anita's position — *"judge a geography by people per unit against the rest
of the map, not by unit count"* — and even that is stricter than what she wants. **A country's
geography is not a gate.** What a coarse country costs is stated plainly and is bounded:

- **The map already says so, in the legend, on every country.** `grain` prints *"religion data
  granularity: municipalities, 25,000 people on average"* or *"regions, 340,000 people on
  average"*, and a reader can see which they are looking at. Georgia's says regions and nobody
  is misled.
- **§6.12's empty-map machinery and §7's confidence rendering are the real defences**, and
  neither cares about unit count.
- **A coarse unit is not a wrong unit.** Dots inside it are placed on Kontur population where a
  grid exists (§8.2), so the visible pattern is finer than the counting tier and the *counts*
  are still exactly the source's.

#### The argument the floor was reaching for, kept

One true thing sat underneath the bad rule and should not be lost with it: **a coarse
geography wastes a deep category list, and a shallow list wastes a fine geography** — which is
§3.9's trade, stated from the other end. That is a reason to *prefer* one source over another
when both exist for the same country, and to say in `sources/<cc>.md` what the country cannot
show. **It is not a reason to decline to draw a country.** Kenya (§9o) was drawn on 47 units
because its categories are the best in Africa and that was already the right call; the mistake
was treating it as an *exception* to a floor rather than as the ordinary way to decide.

#### What this unblocks immediately

**Albania** (12 qarku, ten categories, and the only source anywhere here that counts
**Bektashi** apart from Sunni — in the country where the order is headquartered) and
**Jamaica** (14 parishes, 19 categories, and the only source that names **Rastafari** and
**Revivalist**). Both were rejected on this rule alone.

**Jamaica is ordinary work today** — the file is USCB's, already verified end-to-end in
`sources.md` §11j, and `sources/bd.py` is the template. **Albania is unblocked as a
JUDGEMENT and not as a download**: `instat.gov.al`, `databaza.instat.gov.al` and every
census path under them were unreachable from this machine on 2026-09-06, where §11d had
found the site wide open a day earlier. That is §12's retry rule, not a wall — but the
country is blocked on a host, and this section only removes the reason it was not wanted.
**Saint Vincent** (§11j) was never blocked by the floor — its 221 enumeration districts are the
finest per-capita geography in the project — and is small rather than coarse.

### 3.9c A country being religiously UNIFORM is not a reason to skip it — DECIDED 2026-09-07, Anita's call

§3.9b withdrew a floor on *units*. This withdraws the same kind of unwritten floor on
*variety*, and it came up on Cambodia — 97.1% Buddhist, four categories, 25 provinces, which
is about as unpromising as a country looks from the outside.

> *"in general, we shouldn't let 'this country is boring individually' make us unwilling to add
> countries."* — Anita, 2026-09-07

**Three reasons, in increasing order of how much they generalise.**

1. **The interesting part is usually the remainder, and it is not small where it lives.**
   Cambodia's 3% is Tbong Khmum at 11.8% Muslim and Ratanak Kiri at 23.2% *other religion*.
   A national share says nothing about whether a country has a sharp internal geography, and
   it is the internal geography this map draws.
2. **A hole in a region says something false.** §6.12 already establishes that an empty area
   is ambiguous between "nobody asked" and "nobody is there". A drawn Vietnam, Malaysia,
   Indonesia, Philippines and Sri Lanka around an undrawn Cambodia does not read as *not yet
   done*; it reads as an absence of people. **Adding a uniform country improves its
   NEIGHBOURS' legibility**, which is a benefit that never shows up in an assessment of the
   country on its own.
3. **Uniformity is a finding.** That a country is 97% one religion is a fact about the world
   and one of the more striking ones this map can show. **A map that only draws plural
   countries has selected for its own conclusion** — it would make the world look more
   religiously mixed than it is, which is a bias in exactly the direction a viewer would not
   detect.

**What this does NOT withdraw.** The reasons to decline a country are unchanged and are all
about the source rather than the subject: no religion question, no subnational publication, a
wall, or §14. "Not worth it" is not on that list. **The cost of an easy country is a few
hours** — Cambodia was one afternoon end to end — and §11b's predictor still applies, so the
question to ask is how many pages the table occupies, not how varied the answer is.

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
Three things sharpen that, all in the wrong direction:

1. **This is a lower bound.** The test allocates state → county. The real cases are worse jumps:
   Australia nation → SA2, Canada province → CSD, New Zealand nation → SA2.
2. **The headline is flattered by branches with one populated child.** Bahá'í, Episcopal, Mahayana
   Buddhist and others score 0.0%, not because the method is good but because there is nothing to
   allocate. The genuine multi-child cases are all worse than 5.84% suggests.
3. **We cannot reliably predict which bodies it will fail on.** Counties-per-state against
   misallocation gives r = −0.30, which explains under 10% of the variance, so "flag the clustered
   ones" is not available as a fix.

**Decisions.**

- **Do it, and mark it.** An allocated figure is `derived` in §7's terms. It carries
  `structure_year` / `structure_geo` so the unit panel can say where the split came from.
- **Never let an allocated count reach a ring.** §4.3's ring means "present here", and allocation
  cannot establish presence — it spreads a coarse total over units that may have none of that body
  at all. A ring must come from a real count or a location source (§4.4).
- **Prefer a proxy where one exists.** Flat proportional allocation is the fallback, not the
  method. Several censuses publish ancestry, birthplace or language at the fine geography, and
  conditioning the split on a correlated variable beats spreading uniformly. Yezidis follow Iraqi
  birthplace; Jains follow Indian ancestry. Unmeasured so far, and the obvious next experiment,
  since the same US ground truth can price it.

### 3.10a Built for Australia — and what it does and does not buy

`allocate.py`, run on the ABS data: **30 categories at SA2 become 147**, 363,384 rows, people
conserved exactly, every category's national sum landing within perturbation of its published
figure (Paganism +7, Yezidi +2, Greek Orthodox −110).

**The mapping must be validated arithmetically, never trusted from codes.** Australia looks like a
clean prefix hierarchy and is not: the SA2 column `603 Other Religious Groups` carries its own
prefix children *and* every other narrow group in broad group 6. A pure prefix join drops 30
categories and 92,331 people in silence. `allocate.py` therefore sums each fine column's children
against that column's own total and **refuses to allocate a column that does not reconcile** —
which caught `000 Religious affiliation not stated` (−94.7%, the two sources define it differently)
and `601 Australian Aboriginal Traditional Religions` (+6.7%, §3.8 perturbation). Fourteen columns
turned out to have a single child, so they are exact and are tagged `measured`.

**The failure mode is visible, not statistical.** Every category inside a bucket receives the
*same* distribution, so Yezidi and Paganism come out with an identical SA2 ranking differing by a
constant 4.52. The map would assert that Yezidis and Pagans live in the same places in the same
proportions. That is §3.10's 42% made concrete, and it is more useful stated this way.

**So allocation rescues the middle, not the tail — which was the point.** Greek Orthodox (390,853),
Serbian, Russian and Antiochian Orthodox were hidden inside one `Eastern Orthodox` column and are
now separable at SA2, a real and defensible gain. But Australia's largest bucketed minority peaks
at **72.9 allocated Yezidis in one SA2** — far under a 1,000-person dot, and barred from a ring.
The groups R3 cares about stay invisible after allocation; only §4.4's location sources reach them.

A consequence for R1: because a bucket's members all inherit the bucket's footprint, **every SA2
with a non-zero `Other` now nominally contains 29 religions**. Counting drawn categories would
overstate diversity. **Allocated categories below the dot floor must not be counted as present.**

### 3.10b Canada, and the two checks that earned their place

| | fine geography | before | after | rows |
|---|---|---|---|---|
| Australia | SA2 (2,472) | 29 | **148** | 365,856 |
| Canada | CSD (5,161) | 23 | **147** | 758,667 |

Both conserve people exactly. Canada is the larger prize: Old Order Mennonites, nine Eastern
Orthodox jurisdictions, Doukhobors and Mar Thoma placed at census-subdivision level.

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
encoded it anyway (`ascrg=`/`parent=`); **New Zealand, Ireland and Mexico did not**, so their
hierarchies exist only as prose in `sources/*.md` and each needs a small hand-written mapping table
before it can be allocated. **The normalized format should require a machine-readable parent or
code for every row, and any future source adapter should be asked for it explicitly.**

### 3.10c Allocation must sometimes run WITHIN a coarse unit — FOUND 2026-09-03 with India

`allocate.py` pooled every coarse unit into one national composition. For Australia that is
literally right (`--coarse nation`) and for Canada close enough. **India makes it a catastrophe**,
for one line of data: Sanamahi is 100% Manipur, Niam Khasi 100% Meghalaya, Donyi-Polo 98%
Arunachal Pradesh, Sarna 83% Jharkhand. A pooled national share would put Manipuri and Arunachali
religions into every sub-district in India in proportion to its `Other` count — §3.10a's failure
scaled to a billion people.

**`--within N`** allocates inside each coarse unit: a fine unit takes the composition of the coarse
unit whose `geo_id` is the first N characters of its own. Each religion then reproduces its
published state distribution exactly, because that is now what it is being asked to do.

**The side effect is worth more than the fix, and it changes what §3.10 is *for*.** The
single-child test — a column with one child needs no allocation and stays `measured` — becomes per
*(coarse unit, column)*. A state whose only named `Other` religion is Sanamahi has nothing to
allocate, so its sub-districts get an **exact** split. **245 of India's (state, column) pairs are
exact that way.** Splitting the coarse geography does not merely produce better estimates; it
converts estimates into measurements wherever a coarse unit has only one answer.

India's derived share ends at **0.66%** — the 7.94M in `Other religions and persuasions`, against
six religions measured on all 5,988 sub-districts. The best measured/derived ratio of any allocated
country by a wide margin, and almost all of it comes from `--within`.

**The rule: any source whose coarse table has many units, whose categories are regionally
clustered, should allocate within them.** Which is most sources; pooling was a convenience that
happened to suit the first two countries that needed allocating.

### 3.10d Arithmetic consistency is not evidence of meaning — FOUND 2026-09-03

The most transferable thing India taught, and a limit on every check in §3.10.

India's C-01 **Annexure** is titled *Details of sects/religions clubbed under specific religious
communities*, and it is arithmetically flawless: for every state and every one of the six
religions, `Religion:X` = an unspecified remainder + the named sects, to within a few hundred
people nationally. **Every structural test `allocate.py` applies passes it.**

It names **573 Shia Muslims** among 172.2 million, and **8,399 Catholics** among 27.8 million
Christians.

What it actually counts is people who wrote a *sect* where the form asked for a *religion* — a
measure of insistence, not of membership. Every figure undercounts its community by one to three
orders of magnitude. Allocating it would have put numbers on the map wrong by 100×, in a direction
no confidence marking in §7 can express, and the rows would have carried `derived` honestly while
being nonsense.

**So the reconciliation in §3.10b checks that a mapping is structurally right and cannot check that
a category means what its label says.** Nothing inside the data could have caught this. What caught
it was reading five numbers and knowing roughly how many Catholics India has.

**Every allocation source needs one sanity check from outside the data, and it should be a number a
person already knows.** The trap is specifically that the *large* entries look fine: Lingayat is
2,663,229 and 99% Karnataka, which would have drawn beautifully and is wrong by a factor of four
against a community usually put near 10 million. The small absurd ones are what give it away, so
**read the whole list, not the top of it.**

### 3.11 Reducing "other", and the floor under it

The complement of §3.10: rather than splitting a bucket we cannot split, shrink it honestly.

- **An external national estimate can name a category the census refuses to.** Mexico files
  Orthodox Christians inside *otras religiones*; a published national estimate of Orthodox
  Christians in Mexico, allocated across that bucket, converts an anonymous residual into a named
  group. This is §3.4's rule applied to categories instead of geography, and it inherits §3.10's
  error bars — it is `derived`, and it is still better than a bucket labelled "other".
- **What remains stays named by its source, never merged.** `mexico.otras-religiones` and
  Australia's `Other Religious Groups` contain different things — one holds Orthodox Christians,
  the other Bahá'í, Jain, Yezidi and Wiccans. A single global `other` node would assert they are
  the same. So residual buckets are **per source**, named for it, and a country's "other" is a fact
  about that country's statistical agency rather than about its people.
- **The floor is real and will stay high.** Some of it is irreducible — England and Wales publish
  no Christian denominations at all, and no external estimate reconstructs 60% of a country's
  population at output-area level. The goal is to shrink the bucket where evidence allows and label
  it honestly where it does not, not to drive it to zero.

## 4. The size problem — §4.1 and §4.3 DECIDED, §4.2 built

Christianity is ~2.4 billion. A Carthusian charterhouse is ~20 monks. Eight orders of magnitude,
and R3 says both must be on the same map without the small one lying about its size.

### 4.1 Dot count stays linear in people — DECIDED

No log scale, no sqrt, no per-group rescaling. Within any single view, one dot is one fixed number
of people for **every** group, so two groups' dot counts are exactly their population ratio. This
is the property the whole map is for and nothing below is allowed to break it.

### 4.1a Fractions carry along a spatial order — DECIDED 2026-09-03, twice wrong before

The first build floored every (unit, node) pair on its own — `count // dot_value`, remainder
discarded — and with 147 categories over 5,161 units that is a great many remainders. They are not
rounding noise. **Canada lost 5.5M people, 15% of the country**, and a denomination with 400
adherents in each of a hundred subdivisions is 40,000 people that drew nothing anywhere.

**The first fix rolled sub-floor fragments up the taxonomy** — 400 Old Order Mennonites became 400
Anabaptists — preserving the mass by spending the category to do it, and putting dots on the map
under branch ids: `christianity` was the fourth-largest "category" in the US at 12,153 dots, none
of which is anybody's denomination. Removed.

**The second fix spread each node's national total by LARGEST REMAINDER** — floor everywhere, then
hand the dots still owed to the units with the biggest leftovers. It preserved the mass exactly and
**destroyed the geography**, which took two weeks to notice because every national total stayed
right. The error was purely spatial.

The mechanism: the leftover *is* the local count whenever no unit can reach a whole dot on its own.
In England and Wales the median Output Area holds 306 people against a 1,000-person dot, so **2 of
188,880 units earned a dot from the floor** and all 27,520 remaining Christian dots went by rank.
Ranking on absolute count hands every dot to the places where a group is already densest:

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
Muslim about a borough that is 40% Muslim.

**The rule, decided by Anita and general to the project: never hand dots to the top n. Accumulate
along a geographic order and drop each dot wherever the accumulator happens to be when it passes
`dot_value`** ([[feedback_dot_allocation_spatial_carry]]). Walk the units in spatial sequence, add
up the people, and every time the running total crosses another dot, put a dot in the unit you are
standing in. A unit holding a third of a dot's worth of people gets a dot about a third of the
time, and the dot lands *among the people who contributed it*.

**Why this and not largest remainder.** The previous decision rejected a sequential carry as "a
spatial bias that means nothing and changes if the input is sorted differently", and it was right
about that — a carry in FIPS or DGUID order walks state by state for no reason. The answer is not
to abandon the carry but to **fix the order**: `scatter.py` walks a Hilbert curve through the
units, so consecutive units are neighbours on the ground. Deterministic, no rng, reproducible
without being alphabetical.

Verified against ONS ground truth after the change — Tower Hamlets, clipped to the real borough
boundary and counted from the drawn dots: Muslim 43% drawn against 40% ONS, No religion 28/27,
Christian 24/22. Every band in the table above lands within a few points of 100% instead of between
1% and 332%.

**What is still lost is under one dot per node, nationally.** A node whose entire national total is
under `dot_value` draws nothing, and that is intended: below one dot the map says nothing rather
than inventing a thousand people.

**This changes what a below-floor ring claims** (§4.3). It used to mean "these people are counted
here and represented nowhere on the map". It now means "no dot landed *here*", while the people it
stands for are on the map in a genuinely adjacent unit of the same node. That is a much weaker
statement, and it is half of why rings are no longer drawn by default.

### 4.1b People per dot is a setting, not a constant — DECIDED 2026-09-04, Anita's call

India made the archive 1.87M dots, of which 1.21M is India, and the reasonable question followed:
can the reader ask for fewer? Yes, at a cost that has to be stated rather than buried, because
**coarsening the dot value is not a rendering option — it changes what the map asserts exists.**

**Two editions ship: 1:1,000 (default) and 1:10,000.** `scatter.py --dot-value` writes
`dots_<cc>_10k.geojson`, `tiles.py --coarse` packs both into one archive as `dots10k` / `atomic10k`
/ `rings10k`, and the viewer swaps layer visibility as §4.2b's consolidation toggle does. India
goes 1,207,981 → 120,790 dots.

**Measured cost of carrying both** on the 14-country build: **121.1 MB → 149.5 MB**, +23% for a
tenth-scale copy of everything. Not the +10% a feature count would suggest, because MVT's per-tile
overhead does not shrink with the features in it and the coarse edition still touches nearly every
tile — z10 holds 177,259 coarse marks against 1,540,546 fine ones, spread over the same 33,321 tiles.
`--coarse` is therefore opt-in at build time, and the viewer reads `dot_values` out of
`counts.json` rather than assuming, so an archive built without it greys the control out instead of
offering a setting that does nothing.

**Why a second scatter and not a subsample of the first.** Showing one dot in ten is far cheaper
and wrong twice. The counts would hold only in expectation, which breaks §4.1's "count the dots";
and a group with three dots nationally would appear or vanish on the random seed, where a real
1:10,000 run drops it *deterministically* and §4.3 can then give it a ring. **A coarse dot is a
different measurement, not a filtered fine one.**

**What it costs, and this must not be hidden in a tooltip.** At 1:10,000 the floor under a group
rises tenfold, so small groups leave the map. Where a group's count is `measured` it becomes a
presence ring and is still there — the US gains 109 rings, Czechia 31, Estonia 12. **Where its
count is `derived`, it simply goes, with no mark at all**, because §3.10 forbids an allocated count
from asserting presence and a ring is exactly that assertion. India at 1:10,000 holds 15 nodes
instead of 17, and the two that vanish — Bahá'í (4,572) and Judaism (4,429), including the Bnei
Menashe of Manipur — leave nothing behind, because both come from the allocated Appendix. Two
correct rules meeting to delete 9,001 people is not a bug and is not fixable without breaking one
of them. It is the whole argument for **1:1,000 staying the default**: the coarse edition is the
performance escape hatch, not the map.

**Drawn 2.5× larger, and constant ink was the wrong target.** A 1:10,000 dot stands for ten times
the people, so it has to be drawn bigger. The obvious gain is `√10 ≈ 3.16`, which holds total ink
constant, and it is wrong: **ink does not add.** Ten 1:1,000 dots overlap, so they cover
appreciably less than ten times one dot's area, and a single dot of exactly ten times the area
over-inks — the same reason §4.2a caps a merged mark. `COARSE_GAIN = 2.5`, Anita's call on the New
York view, where 3.16 read as blobby.

**§4.2a's refusal to go sublinear does not bind here, and the distinction is the point.** There,
radius ∝ √k encodes a per-mark magnitude the reader is meant to read back, so bending the curve
would misstate `k`. This is one uniform constant over every dot in the edition, with the dot value
stated in the legend — it encodes nothing per-mark, so it can be tuned by eye without any figure on
the map becoming untrue. **A number chosen for legibility and a number chosen to carry magnitude
are different kinds of number, and only the second is bound by §4.1.**

**1:100 was asked about and is not built.** It is the nicer map everywhere small, and it is 12.1M
dots for India alone — the wrong direction for the problem that prompted this.

**THE EDITION THE LEGEND CLAIMS IS THE ONE THAT LOADS FIRST — REVERSED 2026-09-06, Anita's call.**
The coarse edition used to be fetched first, and the argument was good as far as it went: 3.1 MB
paints every country at 1:10,000 straight away, and because this section makes that an *authored*
edition with its own tallies and rings, the first thing on screen was a whole honest map rather
than a partial one. What it left out is the legend, which reads **"1 dot = 1,000 people" from the
first frame**. So the opening seconds put the coarse map under the fine map's caption and then
multiplied every dot on screen tenfold. Anita: *"we currently show 10k person dots then quickly
adjust to show the smaller 1000 dots."* A reader cannot tell that apart from the map having been
wrong a moment ago, and **an edition is only honest next to its own dot value**.

The cost is real: 30.8 MB against 3.1 is a longer wait for the first dot, so `#loading` now stays
up until the scatter is ready instead of going when the style is. An empty map that says it is
loading is honest; a map that misstates its own dot value is not. The coarse edition still loads,
behind the fine one, and `SCATTER.edition()` falls back **both** ways, so reaching for the 1:10,000
control in the first second gives a coarser map rather than an empty one.

### 4.2 Zooming out merges dots, it does not drop them — DECIDED 2026-08-27

**Two different things, and the first draft of this section confused them:**

- **dot value** — how many people one dot stands for. A data quantity.
- **dot size** — the mark's radius in pixels. A rendering quantity.

**Dot size** grows sublinearly with zoom, roughly like a square root of the scale factor, so dots
stay visible zoomed out and do not swell into blobs zoomed in. In MapLibre an exponential
`circle-radius` interpolation with a base under 2 (§9 has the tuned constant).

**Dot value** cannot stay fixed across all zooms — 8.1 billion people at any legible dot value is
far more marks than a world view can draw. It changes with zoom, and **the way it changes is by
merging, not by dropping.** At each zoom there is a cell size; within a cell, all of a group's
atomic dots collapse into **one mark whose area is proportional to how many merged**. Zoom in, the
cells subdivide, marks split, and at the finest zoom every mark is a single atomic dot.

So a cell containing 40,000 Catholics and 3,000 Alevis draws two marks, and the Catholic one has
13× the area. Count the colours in a cell and you have R1; compare their areas and you have the
composition.

**Why merging beats subsampling.** Subsampling by rank drops dots at random when you zoom out, so a
small group *stochastically vanishes* — present at one zoom, gone at the next, back on a pan.
Merging keeps every person represented at every zoom; a small group's mark just gets small. Nothing
disappears for a reason the reader cannot see. It also keeps §4.1 exactly.

**What merging does not fix** is a group whose merged mark is under a pixel. That is §4.3's job,
and the boundary is clean: marks handle everything down to sub-pixel, rings take over below it.

### 4.2a Built 2026-08-27 — `tiles.py`, and why not tippecanoe

The merge is done by **`tiles.py`, which writes PMTiles directly**, and tippecanoe is deliberately
not in the pipeline even though ancestrydots uses it and it is the obvious tool.

**Tippecanoe's low-zoom job is to drop features** until a tile fits a byte budget
(`--drop-densest-as-needed`). Which dots survive is close to arbitrary, so a small group blinks in
and out as you zoom or pan — §4.2's stochastic-disappearance failure arriving through the back door
of the packaging step. Merging is a different operation and no tiler does it, because it needs to
know that two dots are *the same religion* and may be combined. A useful side effect: this removes
the WSL dependency entirely, which matters because WSL's C: mount is broken on this machine.

Each zoom gets its own aggregate: the tile is divided into 32×32 merge cells (`CELL_BITS = 5`,
about 16px on a 512px tile), all dots of one religion in one cell become one mark carrying `k`, and
the mark sits at the **mean position of its members** rather than the cell centre, so marks follow
the real point cloud instead of snapping to a lattice. Measured on the US at 1:1,000 — 141,501 dots
in, nothing dropped at any zoom: z0 861 marks (largest merge 11,175), z2 3,499 (7,767), z4 12,907
(3,406), z6 40,327 (799).

**The radius cap.** Merging turns many overlapping small dots into one solid circle, and a circle
whose area is the *sum* of overlapping dots is larger than their union — so strict
area-proportionality over-inks the densest cells. Radius is `min(r(z)·√k, 8px)`, the cap being half
a merge cell so a mark cannot spill over its neighbours. **Above the cap a mark has stopped
reporting magnitude**, which is the same kind of statement a ring makes: the cell is full. A real
loss of information in exactly the densest cells, bounded and visible, and better than letting one
circle swallow a state.

cityhistory answered the same dynamic-range question with sublinear bubbles; that trade is not
available here, because there bubble area is the only encoding of a city's size, while here the
mark stands for a countable number of atomic dots and inflating it would break §4.1. **A hard cap
loses information honestly; a sublinear curve misstates it everywhere.**

**Which end of the range to sacrifice is the actual decision, and it is the bottom that must be
protected.** At z4 the merge spans k = 1 … 3,406, a radius ratio of 58, against the roughly 9× that
fits between "visible" and "not overlapping the neighbours". The first build set the base radius
low, so a `k = 1` mark drew at 0.42px and **the countryside emptied out** — not a rendering artifact
but a false claim about where people are. The base is now set so a lone dot stays visible and the
cap bites from about k ≈ 80. Losing resolution among the largest metros costs nothing anyone can
read; losing rural America costs the map its subject.

### 4.2b Consolidation is toggleable — added 2026-08-27

The merge happens at build time, so switching it off is not a rendering option — the unmerged dots
have to be *in the archive*. `tiles.py` emitted them as an `atomic` layer beside `dots`, and the
viewer swaps layer visibility (**overlapping dots: merged / separate** in the legend). *(As of
§4.2d the unmerged edition is no longer a tile layer at all; the toggle is unchanged.)*

**Both views are honest, and they answer different questions.** Merged: area is proportional to
people, so you can compare quantities across a view. Separate: one mark per 1,000 people, so you
read texture and mixing, which is also the more familiar dot-map idiom.

**Separate is the default as of 2026-09-02** (Anita's call). Merged used to be, on the grounds that
at 1:1,000 four out of five body-county pairs were rings and merging kept the rest legible zoomed
out. Both halves of that have gone: rings are no longer drawn unasked (§4.3), and §4.1a's carry
means far fewer pairs are sub-floor. What is left is that the plain scatter is the more
honest-looking object — a merged mark is a circle whose area the reader has to decode, where a
field of equal dots is read by counting, which is the property §4.1 exists to protect. Merged stays
one click away and is still the better view at continental zoom.

**One MapLibre trap, which cost a build.** `['zoom']` must be the *outermost* expression of a paint
property. Wrapping the zoom curve in `['min', …]` to apply the cap fails validation, and MapLibre's
response is to **drop the entire layer** with the error only on the console — the map still renders,
with rings and no dots, looking like a data problem rather than a syntax one. The cap and the √k
factor go inside each interpolation stop's output instead.

### 4.2c Draw order is randomised, or the biggest group loses — FOUND 2026-09-03

**Symptom:** São Paulo read Spiritualist zoomed out and Catholic zoomed in. Brazil has 169M
Christians and 3.9M Spiritualists, 43 to 1.

**Cause:** vector-tile features paint in file order, so where dots overlap the last one written to
the tile is the only one you see. `scatter.py` emits dots grouped by node in sorted order within
each placement polygon, so the alphabetically-last religion present paints over every other one —
`spiritualism.*` after `christianity.*`, every time. Measured in central São Paulo before the fix:
**the last 5% of features emitted was 100% a single node**, against a true composition of 58%
Catholic.

**Fix:** shuffle each tile's feature list, with a fixed seed, immediately before encoding. The
visible dot at a pixel then becomes a uniform draw from the dots covering it, so the zoomed-out
picture is a representative sample of the zoomed-in one — which is the property a dot map is for,
and it was silently false at every zoom until now. Verified at z6, z8 and z10.

**The cost is ~50% on a dense low-zoom tile, not the 16% the whole-archive average suggests.**
MVT delta-encodes consecutive points, so ordering points spatially is what makes them cheap, and
shuffling maximises every delta. Re-encoding one z3 tile over India (1,164,088 atomic features),
gzipped: as built 5.04 MB, sorted by position 2.58 MB (**−49%**), extent 4096→1024 4.23 MB (−16%),
dropping `c` and `t` 4.99 MB (−1%). The average is diluted by high-zoom tiles where dots do not
overlap and deltas are large anyway — but the low zooms are exactly where the bytes and the
overplotting both are. `c` and `t` are near-free because MVT interns repeated values.

**§4.2d makes the whole cost disappear** rather than reducing it: in a flat binary buffer there are
no deltas for a shuffle to spoil, so the ordering is free.

**The general form, which will outlive this instance:** any time a renderer resolves overlap by
"last one wins", the drawing encodes whatever order the data happened to arrive in. If that order
correlates with a category — and sorted-by-id always does — the map is making a claim about
category that comes from the sort, not the world. Every future layer that overplots needs this same
shuffle.

### 4.2d The unmerged dots leave the tile pyramid — BUILT 2026-09-04

**The one sentence:** a tile pyramid is the right structure for data that THINS as you zoom out,
§4.2 forbids this data from thinning, so tiling the unmerged dots bought an elevenfold duplication
and MapLibre's per-feature cost and nothing else. They are now one flat binary buffer per country
per edition (`buffers.py`), drawn by a MapLibre custom layer in a single instanced call.

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

**The diagnosis is not bytes.** MVT is already about 5 bytes per feature on the wire. The 152 MB is
the pyramid storing every dot eleven times, and the *runtime* cost is per-feature JS: a circle is
**4 vertices** and the same dot is resident in every loaded tile at every loaded zoom; each tile
carries a JS feature index for `queryRenderedFeatures`; `circle-color` is re-evaluated per feature
on every recolour (the >120 ms `pumpPaint` exists to ration); and **`setFilter` is worse and was
the hidden one** — filters are applied when the worker populates a bucket, so changing one reparses
every loaded tile. Every country switch, scope change and `unaffiliated` toggle paid seconds for it.

**What replaces them.** The buffer stores a node INDEX; a 512-entry palette texture maps index →
colour and visibility. So `circle-color` becomes the texture's RGB and `setFilter` becomes its
alpha, and the vertex shader collapses a hidden dot to a degenerate quad — §6's "removed, not
dimmed" enforced for free rather than by taking features out of a layer. Country selection is not a
filter at all any more: it is which buffers get drawn. Measured in the viewer, 2.13M dots over
sixteen countries: **recolour 0.28 ms, hover 0.25 ms.**

**Format** — struct-of-arrays, 10 bytes a dot: `x uint32`, `y uint32`, `ni uint16` (node index in
the low 14 bits, §7 tier in the top 2). India is 1,207,981 dots in 12.08 MB, **6.6 bytes a dot
gzipped**. All sixteen countries: 21.3 MB fine, 2.1 MB coarse.

**uint32 fixed point, not float32, and the matrix matters more than the positions.** float32
mercator has an ulp of ~1.2 m — a quarter pixel at z14, four at z18. But the larger error is that
MapLibre's matrix must be downcast for `uniformMatrix4fv`, and that alone is several pixels at z18
however positions are stored. Both are fixed together: subtract a local origin in exact integer
arithmetic BEFORE anything is scaled, folding the origin into the matrix in float64 on the CPU.
Replaying both formulations through `Math.fround` against `map.project`, error with the local origin
is **0.0000 px at z14, z18 and z22**, against 0.73 / 9.74 / 132.74 the obvious way — which degrades
smoothly enough to look fine in testing and be wrong at street zoom. **The origin must snap to a
whole uint32 unit** (~1 cm); snapping to 1/65536 of mercator left 1.24 px at z20, because the
residual the trick exists to cancel comes back scaled.

**Instanced quads, never `GL_POINTS`.** `ALIASED_POINT_SIZE_RANGE` maxes at 1024 on desktop ANGLE
and **63–64 on Mali and Adreno**, and this map's worst case is 8 px × 1.3 `DOT_GAIN` × 3 slider ×
2.5 `COARSE_GAIN` × 2 DPR = 156 device px. Worse, GLES culls a point whose *centre* leaves the clip
volume, so large dots pop out at the screen edges. Desktop would have passed this and phones would
not.

**§4.2c is kept, and here it is free.** Dots are Hilbert-sorted, cut into buckets, shuffled *within*
each bucket, and the buckets written in random order. Local uniformity is all §4.2c asks for — dots
only overlap locally — while global sorting is what buys compression and viewport culling, which a
globally shuffled list forbids. Each bucket carries a bbox and the viewer draws one run per visible
span.

**Culling pad must scale with zoom.** A pad expressed as a fraction of the world is ~800 km at every
zoom, so most of a country falsely intersects a street-level view: 433,837 dots submitted for one
Mumbai junction, against 12,288 once the pad is a few pixels' worth.

**THIS LAYER HAS TO DRAW ITS OWN WORLD COPIES, AND UNTIL 2026-09-08 IT DID NOT.** MapLibre calls a
custom layer's `render()` **once per frame**, not once per visible world copy the way it draws its
own layers — so every dot lived in exactly one copy of the world and **the antimeridian cut the map
in half**. Anita, the day Fiji landed: *"if i'm on one side it doesnt show dots on the other side."*
Fiji is 176°E–179°W, so at any zoom showing the whole country half of it was missing; Kiribati and
Tonga would have been worse. Two things were wrong and both are §9bd's lesson that geometry fails
without complaining:

- **`viewBounds()` returned an inverted range.** `getBounds()` on a straddling view can give a west
  *greater* than its east, so the bucket test `bk[2] < x0 || bk[0] > x1` rejected **every** bucket
  and Fiji went blank on both sides at once. The range is now unwrapped by adding a world to the
  east end when it comes back inverted.
- **Only one copy was ever drawn.** The dots' mercator x is absolute and the origin is folded into
  the matrix on the CPU, so a copy is just the same geometry with ±1 world added to that origin —
  no shader change, no second buffer. Which copies to draw falls straight out of which whole worlds
  the unwrapped range spans.

**The perf guard is the interesting half.** Extra copies are drawn **only while the viewport is
narrower than the world**. Zoomed out past that, every dot is already on screen in the central copy,
the side copies are duplicate scenery, and three passes would be a straight multiple of the most
expensive frame this map draws — z0 submits all 5.1M dots. So the wide case keeps exactly its old
cost, and the fix is nearly free where it is needed, because a view narrow enough to straddle 180
has only a handful of buckets in it. Measured after: **z1 world view 5,170,886 instances — the
reference total exactly, each dot once**; Peru 23,193, its dot count exactly; Fiji straddling 180
draws both copies from either side.

**Picking is now correct, and it was not before.** §4.2c makes the visible dot at a pixel the LAST
one drawn that covers it. `queryRenderedFeatures`'s `features[0]` is the FIRST in tile order, so in
any dense area the old tooltip could name a religion other than the one under the cursor. Draw
order here is buffer order, so the answer is the highest index among the dots whose disc covers the
cursor — exact, from a uniform grid, comparing in uint32 rather than projecting each candidate
(0.03 ms against 4.7 ms).

**Editions do not mix.** The coarse edition and the fine edition replace each other **wholesale**,
never country by country. The US at 1:10,000 beside India at 1:1,000 would put two dot values in
one view, which §4.1 forbids. (Which loads first is §4.1b, reversed 2026-09-06.)

**What is NOT done.** Context-loss re-upload is written (the typed arrays stay resident) but has not
been exercised. Nothing has been tested on a real mobile GPU. And `render(gl, matrix)` is the v4
signature; a MapLibre v5 or globe-projection upgrade would need the shader ported to
`getProjectionData()`.

**Rejected on the way: a capped merge.** Extend §4.2a's merge with a cap on how many dots one mark
may absorb — `m = ceil(k / CAP(z))` marks of weight `w = k/m`, drawn at `r√w`, with `CAP(z)` falling
to 1 at high zoom so it becomes the plain scatter exactly where the scatter is readable. It works,
and it has the property the merge always had: thinning is driven by each religion's OWN local
density, so rare groups are never thinned (measured on a z3 India tile at cap 16, Hinduism keeps
6.3% of its dots and Bahá'í keeps 100%). Rejected because it is still merging — marks grow above
unit size in dense cells at low zoom — and Anita's preference is for the plain scatter to stay a
plain scatter. Worth remembering it exists: it is a continuum between `dots` and `atomic` with one
parameter, and it would have left the elevenfold duplication exactly where it was.

### 4.3 Presence marks: a second grammar that carries no magnitude — DECIDED 2026-08-27

**The current rule, as amended 2026-09-03:**

> A religion gets **one ring per country**, and only where it reaches no dot anywhere in that
> country. The ring is placed at its largest concentration and says only "here", never how many.
> **Rings are off by default.**

By construction such a religion has under `dot_value` adherents nationally, so the ring is a true
and bounded statement. A religion that draws even one dot gets no ring, because the dots already
say it is present.

**Why the ring exists at all.** Because the mark is size-independent it **cannot** misstate
magnitude; the reader learns two symbols, "filled dot = N people" and "ring = present here". A
charterhouse of 20 monks is a ring. This is the honest form of the thing people usually do with a
log scale: a log-scaled dot says "this is small but not that small" and the reader cannot recover
the number, where a ring says "present" and says nothing else, which is exactly the claim we can
support. Rings sit above the dot layer — a ring buried under dots is a ring nobody finds.

**What the old per-area rule cost, kept because it is the evidence for the new one.** One ring per
(area, religion) came to 152,396 rings against 204,046 dots across three countries. Rings per dot
ran 0.3 in the US, 1.1 in Canada and **8.1 in Czechia**, where Catholic alone had 5,804 rings
against 984 dots. That spread is a property of the instrument, not of the countries: a membership
roll reports nothing where nobody is on a roll, while a census reports a small non-zero for nearly
every category nearly everywhere and 1:1,000 puts almost all of it under the floor. **Any per-area
presence symbol will degenerate the same way on any full-census source.** Most of those rings were
also redundant once §4.1a's carry landed — Anita, 2026-09-03: *"since we're doing carry, from case
2 we should only be getting one ring per country-religion."*

**And the dot value is what decides how much of the map a ring has to carry**, which is the standing
argument for §4.1b keeping 1:1,000 as the default. Of roughly 80,680 (body, county) pairs in the US
data, at 1 per 100 the split is 1,576,707 dots to 32,617 rings — 40% of pairs are rings — and at 1 per
1,000 it is 141,501 to 64,739, **80%**. At the dot value a global build can afford, four out of five
body-county pairs could not be drawn as a dot at all. **The finer the dot value, the more honest the
map**, and the dot value is bounded by what can be shipped rather than by anything about religion.

**Uncounted rows now draw nothing at all** — also Anita's call, *"it's not a real number, so let's
not show anything."* A body that reports congregations and never reports membership is not a small
group, it is an **unmeasured** one: the US has 155 of them holding 27,005 congregations — 7.6% of
every congregation in the country — and at the reporting bodies' average of 489 adherents per
congregation that is on the order of **13 million people**. Four have over 1,000 congregations each:

| body | congregations | counties |
|---|---|---|
| United Pentecostal Church International | 4,549 | 1,692 |
| Church of God of Prophecy | 1,614 | 790 |
| Evangelical Free Church of America | 1,602 | 729 |
| Baptist Missionary Association of America | 1,144 | 326 |

UPCI is a top-15 denomination by congregation count with no adherent figure at all. Drawing it as
one ring per county would be as wrong as drawing the Carthusians as dots. **§4.4's
congregation-to-adherent conversion is the right answer and it is not built**; until it is, those
bodies are absent rather than misdrawn, and that absence is the argument for building §4.4 next. It
needs a defensible per-family ratio rather than one national average — 489 mixes Catholic parishes
of thousands with Old Order meetings of forty.

**A bug this found, worth recording.** Canada has **644 census subdivisions that publish no religion
data whatsoever**, arriving as a blank in all 147 categories. Those blanks were taking the uncounted
path and becoming 5,152 rings, each asserting a religion was present in a place whose source had
said nothing at all — §3.5 exactly backwards. The tell was that all 14,877 US uncounted rings carry
congregations and all 5,152 Canadian ones carry zero: **a blank earns a symbol only when the source
counted something else that establishes presence.** Every future source needs its blanks classified
as "present, unquantified" or "not reported", and they are not the same thing.

**The legend follows the map, not the data.** A node with rings and no dots leaves the tree while
rings are off and comes back with them. Czechia forced the rule: 21 of its 48 rows were ring-only,
so with rings hidden nearly half its legend was hollow swatches for marks that were not on screen.

**Two implementation traps from when rings were dense, both still true.**

- **A ring must not sit at the unit's centroid.** Placing every ring for a county at one
  representative point stacks dozens on a single coordinate — visually one ring, and a hover
  answers with whichever is first in the file. Rings are placed in a random tract, the same rule as
  dots, so they neither coincide nor claim a location the data does not have.
- **Hiding must be a `filter`, not an opacity of 0.** A circle at zero opacity is still hit-tested,
  so invisible rings went on answering hovers meant for visible ones: with Sikhism selected, mousing
  over a Sikh ring reported "Wesleyan Church". **Anything hidden for a reason the reader can see has
  to leave the query too.** (§7a had to obey the same rule.)

**Rejected, with reasons:**

| approach | why not |
|---|---|
| log-scaled dot size | destroys R1's at-a-glance quantity reading for *everything*, to fix the tail. The whole map pays for the smallest 0.001%. |
| minimum dot size / "at least one dot per group" | silently inflates: a 20-monk order and a 900-person village group both draw one dot at 1:1000, i.e. both read as 1,000 people. This is the dishonest version of §4.3 and the difference is only that the ring *looks* different from a dot. |
| per-group dot values | breaks §4.1. Two dots on screen would mean different numbers of people depending on colour, which is unreadable and unfixable by a legend. |
| separate "small groups" layer at 1:10 | same as minimum dot size, one step more elaborate. |

**Open:** whether rings should return by default at high zoom, plus a "small groups" toggle forcing
them at any zoom, plus search — selecting Carthusians in the panel should fly to and highlight
their rings regardless of the toggle.

### 4.4 Sources that answer the question backwards — DECIDED to use, they feed rings

Nearly every source in §1–§3 of `sources.md` answers *given this place, which religions*. A second
kind answers *given this religion, where is it*: lists of monasteries, congregations, dioceses,
temples and their coordinates. They are worth pulling in because they are strongest exactly where
the first kind is weakest — small groups, which no census has a row for, and countries that do not
ask at all. §3.7 is the sharpest version of that: the institutional population is precisely the gap
an Annuario Pontificio or a monastery register fills.

They need no special case in the grammar. **A location-by-religion source produces rings.** It knows
a group is present at a point and says nothing about how many people are in it, which is precisely
what a ring means. The two source shapes and the two symbols line up one-to-one.

Where a body's congregation count can be converted to adherents there is a route to dots, but the
result is `estimate` basis, `derived` tier. Rings first; dots only where the conversion is
defensible and declared.

**The trap, and the constraint that contains it.** These sources are dense where mapping effort has
been spent, not where religion is: OSM has far better coverage of German churches than of Indonesian
mosques, and Wikidata's coverage follows Wikipedia's. So they may set **presence and never
density**. 400 mapped churches in one county and 4 in the next is a fact about OSM. Hence: **one
ring per group per unit, never a ring per building.** That single rule keeps the collection bias out
of the picture — it throws away the count, which is the biased part, and keeps the presence, which
mostly is not.

## 5. Reading a region at a glance — the merge does it; no glyphs

R1's problem: 200 colours scattered at random over a country is salt-and-pepper noise. You can see a
place is mixed; you cannot see *how* mixed or of what.

**Everything stays dots.** §4.2's merge is the whole answer — zoomed out, each cell shows one mark
per group present, sized by count, so the number of colours in a cell *is* the number of groups and
their areas *are* the shares. No second visual language.

**Rejected 2026-08-27: the packed-blob cell.** The earlier proposal sorted a cell's dots into a tiny
waffle chart with contiguous colour wedges. It answers the same question and costs a second
rendering mode, a hard switch at a zoom threshold, and a claim it cannot support — clumping dots by
colour inside a cell reads as neighbourhood-scale religious segregation, which we do not know and
which in many cities is false. Sized marks say the same thing without asserting anything about where
inside the cell anybody lives.

**Still worth having, both cheap:** a diversity toggle colouring units by effective number of groups
(`exp(H)`, Shannon); and a hover/click panel listing the unit's composition, with count, source and
year per line — which is also where R4 becomes checkable by a reader rather than a promise in a spec.

## 6. Two colourings, not one — DECIDED 2026-08-27

The complaint about ancestrydots: there was no way to select Western European and see the
differences *within* it. That is not a missing feature, it is a consequence of having one palette.
With everything drawn at once a family has to read as a family, so its members take shades of one
hue and become mutually indistinguishable — right for the overview, useless for "what are the
Christianities".

So colour is a function of **(node, what is selected)**, and there are two modes:

- **Overview palette** — nothing selected. Hue by top-level family, lightness and saturation by
  depth and sibling index. Stable, hand-tunable, memorable: orange is always Islam. This is the
  palette people learn, and the one that answers R1.
- **Focus palette** — a subtree is selected. Its children spread across the **full hue wheel**,
  deeper descendants take shades within their child's hue, and everything outside the selection
  leaves the map. Select Christianity and Catholic / Orthodox / Protestant / Oriental / Church of
  the East / Restorationist are six well-separated hues rather than six blues.

One function rather than two tables: `colour(node, scope)`, where the overview is `scope = root`
and focus mode is the same algorithm re-rooted on the selection. Consequences worth writing down:

- **The wheel is divided among exactly the categories currently drawn**, not among all leaves.
  There is a display-depth control as well as a scope. Select Christianity at branch depth and get
  six hues; at full depth eighty, which will *not* all be distinguishable — that is the honest limit
  of the medium and the reason the depth control exists rather than something the palette should
  paper over.
- **Multi-selection** splits the wheel between the selected subtrees in proportion to how many drawn
  categories each holds. Falls out of the same function.
- **The tree recolours with the map, at the same instant.** Two keys that disagree is worse than
  either alone.
- **No transition on the swap** ([[feedback_no_transitions]]), and a hard swap is clearer than a hue
  rotation across several million dots in any case.
- **Hand overrides are expected.** ancestrydots' `ancestry_colors.csv` is hand-maintained and that
  is not changing here ([[feedback_colors_csv_manual]]).

**The accepted cost:** a colour is not stable across modes, so what you learned in the overview is
not what you see in focus. The mitigation is that the legend *is* the tree and recolours at the same
moment, so the key on screen is never stale.

**This is cheap, which is why it is affordable at all.** Dot features carry only a node index; the
palette is a texture (§4.2d). Changing palette is a texture upload. No second dot set, no re-tiling,
nothing precomputed per scope.

*(Everything above is scoped **per country** for which rows are shown — §6.2 — but not for colour:
§6.8 froze every node's colour across countries, and §6.9 restored the two-palette split by scope.
Read §6.8 and §6.9 before changing anything here.)*

### 6.1 What building it changed — 2026-08-27

**Hue alone is not enough at overview width.** The overview draws about 40 categories, which is 9°
of hue apart, and two 2px dots 9° apart are the same colour. Fixed by alternating *lightness and
saturation* between adjacent entries so neighbours differ on three axes rather than one. Hue order
still follows the tree, so a family stays contiguous on the wheel.

**The overview cannot draw at depth 1.** Christianity is 95% of the US data, so a family-level
palette renders the country in one colour. Depth 2 is the working default and the depth control is
how you get back to depth 1 deliberately.

**In focus mode the panel must expand to the scope**, or the legend for what is on screen sits
hidden behind a collapsed triangle. The fix is `isOpen()`: open every ancestor of the scope, plus
the scope's own subtree down to the drawn cut, and leave everything outside collapsed.

**What it looks like when it works:** selecting Baptist isolates the Bible Belt and splits it into
Southern Baptist (16,571 dots) against the four National Baptist conventions (3,643) — a real and
legible geography, the second urban and Deep South where the first is everywhere. And the overview
alone reads Utah as Latter Day Saints, the upper Midwest as Lutheran, the Northeast and Southwest as
Catholic, without anyone being told to look.

**Still true from before:** sibling groups that are large and adjacent (Sunni/Shia,
Catholic/Protestant) need separation that survives a 2px dot, while distant tiny leaves can share a
shade because they never appear in the same view. And **the genealogy tree is the colour key** —
there is no other legend that holds 200 entries legibly.

### 6.2 One legend per country — DECIDED 2026-09-02

> **The presence pruning below stands. The per-country *palette* it also proposed is REVERSED by
> §6.8** — every node now has one colour everywhere, because a palette nobody can learn cannot be
> hand-corrected either. Which rows a country *shows* is still per country.

§6 assumed one taxonomy and two colourings of it. That is right *within* a country and wrong
*between* them, and the reason is §2.3 and §3.9 rather than anything about colour: what differs
between sources is not mainly how big the categories are, it is **which categories exist at all**.
England and Wales publishes no Christian denomination and fifty minority write-ins. The US publishes
372 bodies and cannot produce "no religion" at any granularity. Mexico files Orthodox Christians
under *otras religiones*, so the same religion sits in a different *place* in the tree.

So the viewer draws **one country at a time, with a legend built only from what that country's
source reports** — and the picker for it is the first control on the page, top left, above the map's
own title.

**Presence prunes the tree, and a ring counts as presence.** The panel holds a node only if that
country has a dot or a ring under it, plus its ancestors. Everything else is not greyed out, it is
gone: **a branch a source never asked about is not a zero**, and drawing it as one invites the
reader to conclude the group is absent rather than unmeasured. This is also what makes the depth
control useful again — depth 2 in Canada and depth 2 in the US are different cuts of different trees
rather than the same cut of one mostly-empty one.

**The all-countries view survives as the entry point.** It draws every built country on the shared
upper taxonomy and carries the comparability warning (§3.1) in the about panel, listing each
country's basis. It is the honest version of "the whole map", and it is explicitly the *less*
informative one: it can only draw what the sources share.

**Three things this forced downstream.**

- Every tile feature carries `c`, its country, and **the merge groups by it**. That reverses the
  original reason for putting several countries in one archive — "so the merge works across a border
  rather than stopping at it" — and it has to, because a mark merged across the 49th parallel
  carries a count belonging to neither side of it.
- `counts.json` is per country: name, source, basis, public note, bbox, and the dot and ring
  tallies. The viewer cannot count anything itself (§9), so *which nodes a country has* is a
  build-time fact like every other total.
- A country needs a **`view` box that is not its data box**. Fitting the US to its dots spans Hawaii
  to Maine and shows the reader an ocean. `countries.py` carries an optional override.

#### Selecting a country: the picker, and Auto

**Two ways, because they answer different questions.** The picker, for "show me Ireland"; and the
camera itself, for "I have zoomed into Canada, stop showing me a legend built for two countries".

**The camera one is a MODE, not a behaviour.** It was originally just something that happened while
you panned, and Anita's objection was that this is confusing: the legend changes and nothing tells
you why or how to stop it. So the picker's first entry is **Auto**, it is the default, and it is the
only state in which the camera may change anything. The button reads **"Auto (United States)"**,
which says both that the map is choosing and what it chose, so a change of legend is never
unattributed; and picking anything else — a country *or* All countries — leaves Auto for good.

**Auto's tests, in the order they were added, each from a real failure.** A country must pass the
IN thresholds to *take* the legend and fall below the OUT thresholds to *lose* it; the gap between
them is a dead band, because a single wheel notch swapping the legend back and forth is worse than
being slow to change.

| # | test | the failure that forced it |
|---|---|---|
| 1 | **fill** (`viewFill`) — how much of the screen the country's view box spans, along whichever axis it presses hardest. IN 0.85, OUT 0.6, hard floor of zoom 3. | Two fixed zooms (3.8/4.2) were read off the United States. In Europe 4.2 frames a third of the continent, so Auto handed the legend to whichever built country dominated a view that was not about any one country. Fill is the quantity `fitBounds` works to, and it follows the window instead of assuming one size. |
| 2 | **overlap** (`viewOverlap`) — the country's box against the **middle half** of the screen (`AUTO_BOX`), as a share of the most the two could overlap at that zoom. IN 0.5, OUT 0.25, and the revert is checked **before** the tally. | Anita: *"I can focus on the UK and then drag away and have no UK on the screen at all but it still doesn't reset."* Fill is a question about scale and dragging east does not change it. Nor could the dots object: a country nobody has built has no dots, so over Germany the tally came back empty, hit its `total < 150` bail and returned — the legend read "United Kingdom" over Munich with no camera move that could clear it. Measured over the whole viewport, overlap cannot separate Britain from Germany at UK-framing zoom (0.53 vs 0.84); over the middle half it is 0.08 vs 1.00. |
| 3 | **the shapes decide *where*, and the dots get no vote** — Auto asks §6.12's coverage-wash polygons which country the camera centre is inside; where they answer, that answer wins. A **miss is not an answer** and falls through to the dot tally. | Anita: *"auto mode near india selects india even if we mostly are hovering over china."* §14.6 draws 2.3% of China — 30,672 dots, nearly all in the west — against India's 1,207,981, in a view box overlapping China's from 73°E to 97.8°E. Frame Yunnan and *every* existing test passed for India. No threshold repairs that: **a country's dot count says how much of it was asked about religion, not where it is**, and a bounding box is the wrong shape for "where am I". The tally still frames Indonesia and the Philippines, both of whose view boxes centre on sea. |
| 4 | **hold** — a country that *already has* the legend keeps it while it still holds `AUTO_HOLD` (0.35) of a 5×5 grid over `AUTO_BOX`. | Anita, on Bangladesh: *"maybe add more hysteresis… so that if we zoom in on bangladesh and zoom out we dont snap to india as fast."* Test 1's band is a band on how big a country looks; the country's *name* had none. **Scroll-zooming out moves the centre away from the cursor**, so a wheel notch over Dhaka walked the centre into West Bengal. At z7 the box is ~110 km, so drifting over the border leaves Bangladesh at 0.48 and it keeps the legend; panned onto Kolkata it holds 0.32 and India takes over. The margin scales with zoom for free. |
| 5 | **`contested()`** — a country may be *selected* only while no other built country holds `AUTO_RIVAL` (0.4) of the middle box. | Anita: *"lean a bit more towards 'all countries'… especially if we have like multiple countries all covering large portions of the screen."* Half India and half Bangladesh is a view of neither. Catches Dhaka at z6.5 (India 0.55 / Bangladesh 0.45), Dublin at z6 (Britain 0.52 / Ireland 0.48), Prague at z6 (Germany 0.43 / Czechia 0.39). |
| 6 | **`outweighed()`** — the selected country lets go once another built country holds `AUTO_OUTWEIGH` (0.25) more of the middle than it does. | Anita: *"china is barely on screen, but it's still autoing onto china."* China at the top edge of an India-to-Indonesia view held **none** of the middle box against India's 0.80, and nothing could dislodge it, because the zoom-out revert works off China's bounding box — 59° wide and 34° tall, so tests 1 and 2 both sail past their OUT thresholds with the country off screen. **Third time a bounding box has been the wrong shape for this question.** |

**Three properties of the design that are load-bearing.**

- **The asymmetry is the whole design.** `contested` gates *selecting*; `outweighed` and the hold
  govern *keeping*; and the margins differ so the two do not fight. Drifting off Bangladesh at z7 is
  both a contested view AND one where Bangladesh still holds the middle, so an already-chosen
  Bangladesh stays while a cold arrival at the same camera gets all countries. **It is deliberately
  harder to change the legend than to set it, and easier to fall back to all countries than to name
  the wrong country.** The first attempt at test 4 replaced the centre probe with an outright vote
  and read worse in three places: over Munich the samples spread across Austria, Switzerland and
  Czechia so Germany never reached a majority, and zooming out over Dhaka snapped to India at z5.5.
  **A vote is a bad way to say where you are and a good way to say where you still are.**
- **Shares are of the samples that hit a BUILT country, not of the whole box**, in both directions.
  Counting empty ocean against a country would drop Indonesia and the Philippines, both framed from
  the sea and both holding 1.00 of their hits. Counting *unbuilt neighbours* against one would have
  handed Bavaria to all-countries, since the box over Munich is half Austria and Switzerland.
- **The binding constraint on the thresholds was the small European countries, not the cases being
  fixed.** A bar that reads well over Bengal is easy to set so tight that Czechia or Croatia can
  never be named. At the zoom each is naturally framed at, the leader holds Prague 0.82, Budapest
  1.00, Vilnius 0.96, Skopje 0.83, Zagreb 0.76, Tallinn 1.00, Dublin 0.97 — all far clear of 0.4.

**A maintenance finding that is not about Auto at all.** The Hanoi report — *"im definitely hovering
over vietnam but it's still autoing onto china"* — was not a threshold problem. Vietnam was drawn
and correct and **missing from `country_shapes.geojson`**, because `country_shapes.py` is the one
build step nothing else depends on. **A country absent from the shapes is invisible to Auto while
looking perfectly built everywhere else.** That is the third silent failure in the add-a-country
sequence after `--countries` and `--coarse`, and it is why `COMMANDS.txt` now opens with a
checklist.

**One thing Auto does *not* fix, and should not:** Natural Earth draws Arunachal Pradesh as India,
so Auto says India at 95°E 29°N. That is the boundary file's answer, and the same one the wash has
been drawing since §6.12.

**Selecting a country moves the camera only if the camera is not already showing it** — over 80% of
the country's view box on screen, and a viewport under 1.8× its area. Both halves are needed: the
continental view contains all of the United States and is still not a view *of* it.

#### Three interaction decisions that came out of the same work

**A single click on a dot does nothing.** Clicking a dot used to name its country, and it is removed
(2026-09-03). It made every attempt to look at a dot a change of view, and on a touch screen it was
worse: a tap is the only way to raise the hover card, so wanting to know what a dot is and wanting
to rebuild the whole legend were the same gesture. The card already names the country.

**Double-click a dot to select its religion.** The gesture the click vacated goes to the question a
dot actually raises. It selects the node the dot carries: a Latin Catholic dot selects Latin
Catholics, and a "Christianity (unspecified)" dot — which carries `christianity` itself — selects
Christianity whole. Over empty map the double-click still zooms; MapLibre's `clickZoom` yields to a
prevented `dblclick` because `mapEvent` sits ahead of it in the handler chain.

> **"Exactly what the card just said" is a constraint, not a description, and it was broken on the
> first cut.** The card reads `e.features[0]` from a layer-delegated `mousemove`, which queries the
> single pixel under the pointer; the double-click ran its own 8-pixel box query and took the first
> feature in it. Over crowded dots those are different features, so the card could say Baptist while
> the selection came out Latin Catholic. They are now registered as a pair in one loop and read the
> same expression, which is the only form of this that stays true: **any second way of deciding what
> is under the pointer will drift from the first.**

**Escape backs out of one thing at a time, innermost first**: the swatch picker, then the country
menu, then the about panel, then the religion selection. Never two on one key — clearing a religion
the reader could not see they still had, because a panel was covering it, is the failure this
ordering exists to avoid.

**"No religion" was hidden by default and is not any more — 2026-09-03, Anita's call.** It was
hidden for two reasons: comparability across countries, which is airtight, and "it is a single
category that can be a third of the map and crowds out the rest of the legend", which is a design
preference. The first argues for the *warning*, not for the hiding — hiding the category does not
make the other categories more comparable. The second is now the reader's to make (§6.11). **Hiding
a group by default is a claim that it is not part of the picture, and it is.** The mechanism stays —
`HIDDEN_DEFAULT` in `index.html` — and is empty. One thing to watch, and it is §6.3a's doing rather
than this decision's: `unaffiliated` is authored *muted* on purpose precisely because it is 81% of
the Czech map, and shown by default that mutedness is working harder than it was. If it ever needs
to read more clearly, **the lever is its lightness and not its saturation.**

### 6.3 The family palette is authored, and the depth cut is now flat — DECIDED 2026-09-03

Two things §6 got wrong in the build, both found by looking at the US at depth 2.

#### The top level was generated, and it should never have been

§6 says the overview palette is "hue by top-level family … stable, hand-tunable, memorable: orange
is always Islam", and then the build divided the wheel among the roots *that country reports*. So
Christianity was hue 0 in the US and hue 33 in Czechia, and Islam's colour depended on how many
families its neighbours happened to be. **A family colour that moves between countries is not a
colour anyone can learn**, which was the entire argument for having an overview palette
([[feedback_stable_palettes]]).

`ROOT_HSL` in `index.html` now names all thirty roots. Nine are named outright — Christianity
yellow, Islam green, Judaism blue, Hinduism orange, Buddhism red, Sikhism red-orange, Shinto pink,
Bahá'í and Zoroastrianism blue-green. The remaining twenty-one take an indigo→magenta wedge.

**The wedge is the honest limit of the idea.** Twenty-one groups over 90° is 4° apart, and two 2px
dots 4° apart are the same colour. They are therefore also cycled through three lightness/saturation
tiers — **period three, not two**: with a two-step cycle every OTHER neighbour comes out identical
in S and L and 8° apart in hue, which measures as ΔE 3 and is no separation at all. **That mistake
was made and caught by measurement, not by eye**, and §6.5 made it again in the generated wheel.

Order inside the wedge is chosen against the data rather than by theme: the groups that co-occur
**at size** in one country are the ones spaced apart. Brazil carries five at once (Spiritism 3.9m,
Afro-diasporic 588k, Japanese new 155k, esoteric 74k, indigenous 63k), so those five set the
spacing and the rest fill in between.

**What is measured, and what is accepted.** `tools/check_palette.py` reads `ROOT_HSL` back out of
`index.html` — not a copy of it, so it cannot drift — and checks every pair of roots that co-occurs
in a built country, in CIE Lab, against `counts.json`. In the US and Czechia every pair over 20 dots
is at least ΔE 25 apart. What remains:

| pair | ΔE | where | why accepted |
|---|---|---|---|
| Japanese new / indigenous | 13 | br, world | 155k and 180k against a 190m country — specks, never adjacent areas of colour |
| indigenous / pagan | 13 | ca, world | 180k and 82k |
| esoteric / pagan | 15 | world | 94k and 82k |
| the not-a-religion family | 20–23 | everywhere | superseded by §6.3a, which re-authored them as one ramp and has the reasoning |
| Hinduism / Sikhism | 31 | ca | inherent: four of the nine named families are warm and sit inside 60°. Re-authored 2026-09-03 — see §6.9's search |

**Re-run the checker after every re-tile.** `counts.json` is rewritten by each `tiles.py` run and
holds only the countries that ran, so which pairs are close is a property of what is in the archive
today. Australia, Ireland and Mexico landing on 2026-09-03 moved every number in that table, and
brought three roots — Mandaeism, Yazidism, Cao Đài — that had no authored colour at all. **A new
root with no line in `ROOT_HSL` falls back to a generated wheel position, which will look fine and
mean nothing.** The checker names them; that is what its first section is for.

Adding those three took the wedge from 16 entries to 19 and its step from 4° to 3.3°. **The wedge is
the part of this that does not scale**, and it will keep not scaling as countries land. The remedy
if it starts to matter is not more tuning — 21 groups do not fit in 90° — it is to **promote a group
out of the wedge into a named hue. Spiritism is the candidate**: 3.9m people in Brazil, larger there
than Judaism, Buddhism and Hinduism combined are in the US, and it is in the overflow bucket only
because the brief that set the nine named families was written against US/Canada data.

Two colours at the blue-violet end of the wedge — Daoism and the Unification Church — sit at
contrast 2.3 and 2.8 against the map background, under the 3.0 the rest clear, because blue carries
little luminance at a given lightness. Both are ring-only in every built country, so lifting them at
the cost of breaking the tier rhythm has not been worth it. It becomes worth it the day a source
reports either at size.

**Two roots that arrived with no authored colour, and where they went.** Alevism and Ravidassia came
with the UK; Alevism took a fallback measuring dE 15.7 from Paganism and 20.9 from Jainism in the
one country where all three are drawn. Both are now placed in free hue rather than squeezed into the
wedge — Ravidassia in the 16→35 gap between Sikhism and Hinduism, Alevism in the 168→190 gap between
Bahá'í and Zoroastrianism. Each sits beside a tradition whose boundary with it is the contested
thing, which is either a useful statement or an unwanted one; `branches.py` records that both groups
reject the filing. Alevism was tried at 152, next to Islam, and moved: it measured dE 17.7 from
Islam in the UK, where Islam is 3,999 dots and Alevism 25, and **a speck that reads as a shade of
the largest thing beside it is worse off than uncoloured.**

#### Almost none of the map's colours were in the legend

The palette shaded *within* each drawn node so a branch's children stayed separable. That sounds
harmless. Measured against the tallies it was not: at depth 2 in the US, **93% of the dots on screen
were a shade of something whose own legend row sat a level further down behind a collapsed
triangle** (98% at depth 1; 90% in Canada; 52% even inside a Christianity selection). The visible
case was the Catholic Church — 61,858 dots, over a third of the country, in a washed-out red that
the "Latin Catholic" swatch above it did not match.

This contradicted §6's own premise: the tree *is* the colour key, and a shade with no row in the
tree is a colour with no key. **So the rule is now flat: a node below the drawn cut takes its drawn
ancestor's colour exactly.** Every colour on the map has a swatch in the panel, at the depth you are
looking at, and `+` is how you tell a branch's children apart. Expanding a triangle without raising
the depth shows several rows sharing one swatch, which is the true statement — they are one colour
on the map at this depth. (§7 later generalised this into the rule that killed the confidence fade:
[[feedback_legend_is_the_palette]].)

#### Selecting a single sect no longer repaints it

Dividing a hue wheel among a set of one puts that one on hue 0, so selecting Islam in the US — which
ASARB reports as a single category with nothing below it — turned it from green to red for no reason
a reader could see. **When the drawn set has one member the palette does not change at all**: the
node keeps the colour it had in the view the selection was made from, and selecting it only removes
everything else.

### 6.3a Not-a-religion is one grey ramp — DECIDED 2026-09-04

Seven roots are answers *about not belonging* rather than traditions, and they are drawn as one
neutral ramp separated mostly by lightness. The membership is the whole list, and anything added
must be an answer of the same kind:

| | | what it is |
|---|---|---|
| `unaffiliated` | No religion | the largest single answer in Canada, the US and Czechia |
| `secular` | Secular and ethical | a stated non-theistic **position** — Ethical Culture, and the Atheist / Agnostic / Humanist answers Canada and Pew both collect. Not the same as `unaffiliated`, which is the absence of one |
| `unchurched` | Believing, no church | reports religious belief **and** explicitly no institution. Czechia's 960,201; Pew's "spiritual but not religious" |
| `other.<source>` | Other | the per-source residual containers of §3.11. The greyest of the five, being the one that says nothing |
| `parody` | Parody and protest answers | Jedi, Pastafarian — a protest, not a belief |
| `unrecorded` | Religion not recorded | §6.3a-i — the source **never asked** |
| `unknown` | Religion unknown | §6.3a-ii — the source asked and the answer set was too narrow |

**Lightness is prominence, and the map is dark — Anita, 2026-09-04.** A light dot on a near-black
background is the loudest mark available, so the ramp runs the opposite way to the obvious one: the
thing we least want shouting is `unaffiliated`, the largest node on the map, so it is the
**darkest**. What falls out is a reading and not only an aesthetic — the ramp ends up ordering the
five original members by how much religion is being reported:

| | L | |
|---|---|---|
| `unaffiliated` | 41 | no religion at all |
| `parody` | 46 | a joke, not a belief |
| `secular` | 50 | a position, and not a religious one |
| `unchurched` | 58 | a belief, held outside any institution |
| `other` | 64 | an actual religion, which the source did not name |

`unaffiliated` is **deliberately below the contrast floor** — 2.59 against the 2.9 the checker wants
— and `check_palette.py` exempts it by name in `DIM_ON_PURPOSE`, because being quiet is the whole
point of the colour and reporting it as too faint invites undoing the decision. `secular` is blue
and `unchurched` purple by request, and both hues do real work: with two members at the dark end,
hue is the only axis left to separate them by. **`parody` is the one warm grey**, and at lightness
46 with four cool neutrals around it, cool-vs-warm is the axis left — which happens to suit the one
member that is a joke rather than a position. (It is now one of two — see §6.3a-ii.)

**Anita, 2026-09-04: these should look like what they are.** `secular` was authored at saturation 72
and `parody` at 82 — more vivid than most religions, for categories that are the absence of one.

**The cost is real and is accepted.** ΔE 25 is not reachable inside a family this narrow; the ramp
gets **19.9** at its tightest. That is accepted rather than tolerated, because **the ΔE 25 rule
exists so two *different religions* are never confused**, and inside this family a mix-up is between
neighbours on one spectrum — reading "secular" as "no religion" is a small error where reading
either as Islam is not. Every member clears **ΔE 27.7** from the nearest religion, which is the
threshold that actually matters here.

**The children land inside the ramp too, and that is not fixable by moving the parent.** `other` is
a container; what draws is `other.us`, `other.cz` and the rest, and the viewer lightens a child off
its parent. `other.us` renders `#acaeb9` — ΔE 9 to 10 from three of the other four. The band is
44–78 and the family plus its children do not fit in it with room to spare. Accepted on the same
argument and one more: `other.us` is 1,124 dots against `unaffiliated`'s 60,554, so the confusion is
54-to-1 in favour of reading the small one as the large one, and both mean "not classified" in any
case. **`tools/check_palette.py` compares roots and cannot see this class of collision at all** —
worth knowing before trusting a clean run.

#### 6.3a-i A sixth member, and it is a different kind of thing — ADDED 2026-09-04 with Germany

`unrecorded`, "Religion not recorded". Germany's *Sonstige, keine, ohne Angabe* — 42.8M people,
**51.8% of the country**, and the largest node any single country contributes after the US (§3.9a).

The five above are all **answers**: somebody was asked and said something, including saying no. This
one is the residual of a source that **never asked** — an administrative register, read for church
tax, which holds no religious body for these people. Every existing home asserts something false:
`unaffiliated` is a person reporting no religion; `other.<source>` is a religion the source named
and could not place, and here the source named nothing; `unchurched` is a positive report of belief
without institution.

**The test that admits it, and it is the test for anything added next: this category's composition
is a property of the INSTRUMENT rather than of the people in it.** Germany's bucket holds Muslims,
Orthodox Christians, Jews, Freikirchen and the wholly irreligious together, and which of them are in
it is decided by German tax law. Any register-basis source of the same shape belongs here; Austria
is the obvious next one.

**The label had to change, and the legend could not hold the reason — Anita, 2026-09-04.** It was
"No religious body recorded", which is accurate and still lets a reader conclude that 43 million
Germans are irreligious. The obvious fix — "…(includes Islam, Orthodox, etc.)" — cannot be done in
the label, because `.row .lb` is `nowrap` with `text-overflow: ellipsis`: a label long enough to
carry a caveat gets **cut off mid-caveat**, which is worse than saying nothing.

So the two halves were split. The visible label is **"Religion not recorded"** — shorter than what it
replaced, and grammatically unable to be read as a statement about belief, because the subject is
the record. The caveat goes in the row's **tooltip**.

> That needed a new field, and it is the general one this project was missing: `branches.py` has
> carried a `note` since the first tree and **nothing in the viewer has ever shown it** — it is for
> whoever maintains the taxonomy. **`PUBLIC_NOTE` is the other kind, written for a reader**, and
> `build_tree.py` carries it into `religions.json` as `public_note`. **Only add one when the label
> alone would mislead**: "Lutheran" means Lutheran and needs nothing. The test is whether a reader
> who reads the label and stops comes away believing something false.

It remains a global label rather than a German one, and the tooltip names the groups generically for
the same reason — the node is any register country's, and Austria's bucket will hold a different
mix. A country-specific version belongs in `note_public`, which the about panel renders.

**It takes `other`'s hue at the ramp's darkest lightness — `hsl(228, 10, 37)`, `#555968`** — and both
halves are the rules applied rather than bent. By the ordering principle it reports less than any of
the five. By the prominence rule it must be quiet, and harder than `unaffiliated` must, because a
pale mass of 42.8M dots would drown the Catholic/Protestant signal that is the only real information
Germany carries. Measured: **dE 21.1 to `unaffiliated`**, which is *wider* than the family's existing
tightest pair, so a sixth member does not squeeze the ramp; dE 51.1 to the nearest religion; contrast
2.70, in `DIM_ON_PURPOSE` for `unaffiliated`'s reason.

**It is not green, and the search wanted green.** `hsl(120,30,43)` scores dE 41.8 inside the family,
twice as well. Rejected on meaning rather than measurement: Islam is at hue 138, and colouring the
one bucket that hides Germany's Muslims a muted green is the worst available accident (§14.2). **A
palette search optimises separation and cannot see what a colour would say.**

#### 6.3a-ii A seventh member, `unknown` — ADDED 2026-09-06 with Vietnam, and §14.7 asked for it

`unknown`, **"Religion unknown"**: people the source counted and whose religion it did not
establish. Vietnam's census asks which **state-registered** religious organisation a person belongs
to, and **70,195,530 people — 81.8% of the country — answered none of them.**

**It is not any of the other six, and the near miss is `unrecorded`.** §6.3a-i's whole argument for
splitting `unrecorded` from `unaffiliated` is that *being asked* and *not being asked* are different
things: Germany's bucket is a register that never asked anybody. **Vietnam's respondents were asked;
the answer set was too narrow for their answer to mean what it says.** Folding the two together
would blur exactly the distinction §6.3a-i created.

**The admission test, and it is narrower than "we are not sure":** the source **counted** these
people, and what they practise is not determinable from it *at any geography*. Not a small residual,
which is `other.<source>`; not a suppressed cell; not a report of anything.

**§14.7 specified this node for China and Vietnam got here first.** China's 900M–1B will land on the
same node when it is built — at which point this becomes much the largest thing on the map.

**`#68665a` = `hsl(51, 7, 38)`, Anita's hex, and it took three goes.** S 7 is barely more chroma than
`other`'s and half `unrecorded`'s: it is **a grey with a warm cast rather than a colour**, which is
what §14.7's "slightly yellow" was after and what the first two attempts overshot.

**It is the third quietest of the seven, not the first, and that is deliberate.** Contrast 1.55,
against `unrecorded` 1.21 and `unaffiliated` 1.23, with `parody` next at 2.06. It was authored at
1.23 by §6.3a's rule, rendered, looked at, and moved up — because at 70,195 dots against Vietnam's
15,646 this node **is** the country's settlement pattern, and the argument for drawing the residual
at all (§14.7) is that the people are visible as people. A backdrop nobody can see does not do that
job. It still sits below everything that reports a religion, and it is no longer in `DIM_ON_PURPOSE`,
because at 3.3 on the root palette it is not dim and **a list of deliberate exceptions should not
contain non-exceptions.**

> **AND THE LIGHTNESS NUMBER IS THE TRAP, WHICH IS THE PART THAT NEARLY SHIPPED WRONG.** The first
> draft was `hsl(48, 16, 35)` — L 35, comfortably below `unrecorded`'s 38 and therefore "obviously
> the darkest". **It measured contrast 1.47 against `unrecorded`'s 1.21**, because yellow is a
> luminous hue and the ramp's ordering is by *measured luminance*, which the third HSL column is
> not. The largest node on the map would have been the third brightest grey **by accident**. **An
> HSL lightness is not comparable across hues; author the ramp against `check_palette.py`'s contrast
> column, never against the L you typed.**

Its nearest neighbour is `parody`, the family's other warm member, at **ΔE 10.3** — the same distance
as `unaffiliated`/`unrecorded` — and the two never co-occur: `parody` is 52 dots across Czechia, the
UK, Australia and New Zealand. Anita, 2026-09-06: *collisions are fine where the pair does not
co-occur.*

**It is NOT in the viewer's `NO_RELIGION_IDS`**, for `unrecorded`'s reason and more strongly: hiding
it behind a control labelled "no religion" asserts in one click the thing the node exists to avoid
asserting.

**What it bought, which is the argument for having it at all.** Vietnam without it draws 15,646 dots
on an empty country, and §6.12's machinery can *label* that blank but cannot fix it. With it the
country draws its own settlement pattern in grey — the two deltas, the coastal strip, the empty
uplands — and every religious concentration sits legibly on top: An Giang's Hòa Hảo, Tây Ninh's
Caodaists, Nam Định's Catholics. **The people are visible as people rather than as absence**, and
nothing on screen claims to know what they believe.

#### 6.3a-iii An eighth member, `unenumerated` — ADDED 2026-09-07 with Myanmar, Anita's call

`unenumerated`, **"Not enumerated"**: people the source did not COUNT, drawn at the source's own
estimate of how many of them there are.

**Every one of the other seven is a residual of some answer.** `unaffiliated` said none, `secular`
stated a position, `parody` refused the question, `unchurched` reported belief without institution,
`other.<source>` gave a religion the tree cannot place, `unrecorded` was never asked because the
instrument is a register, `unknown` was asked from too narrow a list. **This one was never reached
at all**, and the number is an estimate rather than a count — which makes it the weakest figure on
the map and a different kind of thing from all seven.

**Myanmar is the first, at 1,206,353 people, and Rakhine State is 1,090,000 of them — 34% of that
state.** The 2014 census report says why in its own words: *"In Rakhine, an estimated 1.09 million
people were not enumerated in the Census because they were not allowed to self-identify using a name
not recognized by the Government."* Kayin (69,753) and Kachin (46,600) are areas that were not under
government control.

**Why not `unknown`, which is the near neighbour.** §6.3a-ii admits a category there on a
deliberately narrow test — *the source COUNTED these people, and what they practise is not
determinable from it*. Vietnam's 70 million were counted. Myanmar's 1.2 million were not. **Counted
but unclear, and not counted at all, is exactly the kind of distinction §6.3a-i created this family
to keep**, and folding them together would lose the only thing this node actually knows.

**Why not `islam`, which the SOURCE ITSELF suggests — and this is the decision worth reading.** The
report states its own assumption, *"it is assumed that the non-enumerated population in Rakhine is
mainly affiliated with the Islamic faith"*, and applies it to publish a second set of national
figures (Islam 2.3% → 4.3%). It applies it **at Union level only**, and this map draws states. Taking
it to state level would be estimating a magnitude at a finer resolution than the source publishes
it, which is §14 rule 1 — the one rule that has never moved. So the map draws the people and not the
inference, and the state's own assumption is quoted in `note_public` so the reader is told what DOP
itself concluded.

**The alternative was not drawing them, and §14.2 is why that is worse.** Enumerated Rakhine is
2,098,807 people of whom 28,731 are Muslim, so a map built from the enumerated columns alone renders
Rakhine **96.2% Buddhist** — the census's own exclusion reproduced as a finding. That is §14.2's
second risk (*getting a community wrong is itself a harm, and the failure is asymmetric*) in its
purest available form, and it is the reason the country is worth drawing at all.

**Every row on the node is `modelled` (§7), which makes the toggle the honest test.** The tiers are
about whether anybody was counted; here nobody was, and 1,090,000 is a round number in the source
because it is an estimate. `unenumerated` is a root with nothing measured above it, so the rolled-up
view removes it outright — showing the census exactly as the state published it, hole and all.

**The colour is the family's only TRUE NEUTRAL — `hsl(0, 0, 51)`, `#828282` — and the constraint that
set it was §6.3a-i's, reappearing sharper.** Germany's node rejected a green that scored twice as
well inside the family, because Islam sits at hue 138 and colouring the bucket that hides Germany's
Muslims green is the worst available accident. **Myanmar is the same trap with the state's own
assumption behind it**: a green-cast grey here would quietly make the assertion this node exists to
refuse to make. Saturation 0 is also the honest reading — every other member has a hue because
something is known about the answer, and here nothing is.

Measured: **ΔE 73.5 to `islam`**, 32.1 to the nearest actual religion (`druze`), clearing the 27.7
this family holds itself to; contrast 4.88, fourth loudest of the eight. Inside the family its
nearest are `secular` 10.7, `parody` 12.7 and `unknown` 13.4 — **tighter than the family's previous
worst of 19.9, and accepted on §6.3a-ii's own test**: no country draws `unenumerated` together with
any of the three, and Myanmar's nearest co-occurring pairs are `unaffiliated` at 19.4 and `other.mm`
at 19.7. Lightness is set for visibility rather than by the ramp's "biggest is darkest" rule, on
`unknown`'s precedent — in Rakhine this is a third of the state, and a backdrop nobody can see
cannot do the one job it has.

**And it is NOT in `NO_RELIGION_IDS`**, for `unrecorded`'s and `unknown`'s reason and more strongly:
these people were not asked and did not decline, so hiding them behind a control labelled "no
religion" would assert in one click the thing the node exists to refuse to assert.

### 6.4 The middle level — ANSWERED by §6.9, 2026-09-03

Depth 1 is authored (§6.3) and focus mode is the §6 wheel. The level between them — every branch of
every family drawn at once, which is the working default — was the generated wheel over the drawn
categories, so Christianity's branches ran red → orange → yellow → green and Islam was wherever its
index fell.

The tension: hue-by-family keeps Islam green but gives Christianity's twenty US branches twenty
shades of yellow, which is the ancestrydots complaint that started all of this. The full wheel
separates the branches and throws away every family colour the reader just learned at depth 1. A
hybrid is not available either: the authored hues are **thirty-three families**, the middle level is
**~30 branches of one of them**, so the two cannot be one table, and reserving the nine named hues
leaves the generated wheel running straight through green, teal and magenta.

**The answer is the first horn, taken deliberately: hue by family, and the branches inside it do not
separate.** §6.9 builds it, and what made it affordable is that the second horn is one click away —
the full wheel is what a *selection* draws. It was never a choice between two palettes; it is a
choice about which one is the default.

### 6.5 Colour follows descent, not size — DECIDED 2026-09-03

The wheel was divided in **size order**, because that is how the panel sorted a parent's children.
That has one specific and bad consequence: the two largest bodies are always adjacent on the wheel,
and they are exactly the two a reader most needs to tell apart. In the US, Catholic (62m) and Baptist
(24m) came out 18° apart, red and orange.

So `branches.py` gains **`LINEAGE`** — the second of §2.1's two relations, in the smallest form that
is useful. Not the full `from` DAG with dates and edge kinds; §10 still owes that. This is one thing:
for a parent with many children, the order they descend in, cut into named groups. Six for
Christianity — ancient communions, Reformation, separatist and believers' churches, Pietist and
Wesleyan revival, restorationist and adventist, and the bodies on no single line — plus Judaism's
three and Buddhism's one.

`buildTree` sorts by that where it exists and by size everywhere else, and because `drawnSet` walks
`KIDS` in order, **the same sort decides the panel order and the hue order.** That coupling is the
point: the panel reads down in the order the colours run. Catholic and Baptist are now 144° apart,
and not by luck — a body is usually large *because* it is its own tradition, so descent order tends
to separate the big ones on its own. Measured, the closest pair among the six largest under US
Christianity went from ΔE 18 to ΔE 27. *(§6.13 later broke the panel/hue coupling for Christianity,
knowingly.)*

**The generated wheel's tiers went from two to three at the same time**, for §6.3's reason: with a
two-step lightness cycle, every entry and the one two places along are identical in S and L and
separated by hue alone. That had Latter Day Saints and Pentecostal — 6.5m and 6.0m — at ΔE 17.

**The panel captions the groups**, which is what makes the order legible rather than merely
non-arbitrary; without a caption a reader sees a list that is not sorted by size and is not told what
it is sorted by, which is worse than sorting by size. **A caption is not a node**: it cannot be
selected, has no count, and is not a level of the tree. A parent whose present children fall in one
group gets no caption; Buddhism's three vehicles do not need to be told they are all vehicles. And
**captions appear only under the node you have selected** — in the all-religions view they sat inside
Christianity's children while Islam and Judaism, Christianity's *siblings*, sat below them at the
same indent, so "ANCIENT COMMUNIONS" read as a bigger division than Christianity itself.

**`build_tree.py` fails if a child of a lineage-carrying parent has no group.** Without that check,
adding a branch drops it silently to the end of the panel and the end of the wheel — still drawn,
still the right size, just quietly not where it descends.

**What a linear order cannot carry.** Baptists come out of English Separatism with Dutch Mennonite
contact; Methodism comes out of Anglicanism by way of the Moravians. Each is placed on its main line
with the other edge written down as a comment. That is the honest limit of a list, and the argument
for §10's DAG rather than a replacement for it.

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
every non-US Catholic was grey.

So **a branch with dots of its own is a drawn category** and takes a hue beside its children.
`drawnSet` pushes it ahead of them, which is what makes the palette come out right: the parent paints
its whole subtree, each drawn child paints over its own part, and what is left holding the parent's
colour is exactly the parent.

The panel then has to say which dots that colour marks, and it is not the branch's total —
"Christianity 19m" beside a swatch that marks 6.2m of it would be a worse lie than the grey it
replaced. So a split branch keeps a grey container swatch and its own dots get **a row of their own,
labelled `unspecified`**, carrying the colour and the branch's own count. That row is §3.2's residual,
read out of the tallies the viewer already has. It is not a node and cannot be selected — there is
nothing in the tiles to select, those dots carry the parent's id. *(§6.15 stops drawing it where the
branch's own dots and one child's are the same people.)*

**The invariant this buys, and it is checkable: no dot on the map is grey.** Grey is the container
colour, and a dot in it is a religion with no key.

### 6.7 Out of scope is hidden, not dimmed — DECIDED 2026-09-03

§6 said everything outside the selection "drops to near-black", and at 0.18 opacity it did. But there
is one dot layer, so the dimmed dots draw in feature order, which means about half of them draw *on
top of* the selection. Select Buddhism over New York and the 549 Mahayana dots sit under a grey wash
of Catholics. Opacity cannot fix that at any value; only taking them out of the layer can, so the
scope is part of the layer filter. What is lost is the silhouette of the country, and the basemap
already carries that. (§4.2d makes this free: a hidden node's palette entry has zero alpha and the
vertex shader collapses the quad.)

### 6.8 One palette for every country — DECIDED 2026-09-03, and it reverses §6.2

§6.2 divided the wheel among the categories **present in the country on screen**, and accepted that
"a hue means different things in two countries" because the panel is the key and recolours with the
map. Anita's call, and the argument that settles it: **a palette nobody can learn cannot be
hand-corrected either.** §6 always planned hand overrides for the scopes people actually select; an
override that only holds in one country is not an override, it is a per-country table with no end.
**Stability is what makes hand-tuning possible at all** ([[feedback_stable_palettes]]).

So **every node has one colour, computed once from the whole taxonomy.** Which divisions a country
*shows* is still per country — §6.2's presence pruning stands, and it is the half of §6.2 that was
about honesty rather than about colour.

Allocation, per level, in an order that has nothing to do with any country — descent where
`branches.py` writes a `LINEAGE`, file order otherwise. Each parent's children go at `phase + k/n`
around the wheel, `phase` being a golden-angle offset per parent. Two simpler orderings were built
and measured first, and both fail:

| ordering | parents whose six largest children include a pair under ΔE 25 |
|---|---|
| contiguous — each family in one arc, siblings adjacent | 31 of 37 |
| interleaved — all first children, then all second children | 11 of 37 |
| **spread with a per-parent phase** | **9 of 37** |

Contiguous fails because siblings are exactly what a reader compares, and 126 level-2 nodes is 2.9°
apart — it took Catholic's children to ΔE 12 and Lutheran's to ΔE 9. Interleaving fixes that until it
runs out of parents: past the index where only the largest family still has children, its remaining
ones are adjacent again, which put twenty of Christianity's branches in one run of greens. The phase
matters as much as the spread — without it every family starts at hue 0, and the fifteen-odd families
with a single child pile into one arc.

**What it costs, and it is not small.** §6's focus palette is gone: selecting Baptist no longer
spreads the whole wheel over its nine children, it shows the nine colours they already had. And
Christianity has 29 of the 55 level-1 categories, so its branches can be at best **12.4° apart**
against the 18° a per-country wheel gave them in the US. That is arithmetic, not tuning. The remedy
is the one the whole change is for: `PIN` in `index.html`, keyed by node id, hand-set and now worth
writing because it stays true. *(§6.9 restores the focus palette; §6.13 makes Christianity's
branches a table.)*

### 6.9 Two palettes again, split by scope and not by country — DECIDED 2026-09-03, and it amends §6.8

§6.8 froze every node's colour so that panning from Canada to the United States could not repaint
anything. That was right and it stands. But it froze the colour across **scopes** at the same time,
and that half was an overreach: the all-religions view then spent the whole wheel on Christianity's
27 branches, because Christianity holds 29 of the 55 level-1 categories and the allocator has no way
to know that the other 26 belong to families a reader is trying to *find*. Five Christian branches
came out green, beside Islam.

**The two properties are separable and only one was ever the problem.** A colour that moves when you
pan is unlearnable and cannot be hand-corrected, which is §6.8's argument entire. A colour that
changes when you deliberately select a family is §6's original trade, taken knowingly, with the panel
repainting in the same instant. So:

| view | palette | what it is for |
|---|---|---|
| nothing selected | **overview** — every family inside a band around its authored hue | which family is this dot |
| a subtree selected | **focus** — §6.8's whole-taxonomy wheel, unchanged | which branch is this dot |

Both are computed once from the whole taxonomy. Neither moves with the country.

**The band is authored, in `ROOT_BAND`, for the same reason `ROOT_HSL` is** — a width derived from
how many branches a family has would move the day a source landed. Only three families have an entry.
That is not an oversight: **24 of the 30 families draw exactly one row at depth 2**, so they take
their own hue and nothing else, and the hue budget is far less contended than the taxonomy makes it
look.

**Two rules the build produced, both from a measured failure:**

- **A band must not contain its own root's hue.** A split family draws a row for its own dots at the
  root colour (§6.6's `unspecified`, and that row is 30% of Judaism and 60% of Buddhism worldwide),
  so a branch landing on the root hue cannot be told from the single largest thing beside it.
  Judaism's band runs below 214 and Buddhism's below 356 for that reason. *(§6.14a is the same rule
  met from the outside, by a node that had left the band.)*
- **Nothing in the indigo→magenta wedge gets a band at all.** The wedge is already at 3.3° per family
  (§6.3), so a band even ±3° wide reaches its neighbours: giving Afro-diasporic one put Umbanda at dE
  9.7 from Jainism, a different family. Wedge families fall through to a ±2° default and separate
  their children by lightness alone.

**Christianity's band stops short of green, and that is deliberate.** At 110 the last two lineage
groups came out green, so nine Christian branches — Non-denominational at 21m and
Protestant-unspecified at 19m among them — sat in the legend as green dots above Islam's green dot.
Every one of those pairs measured over dE 25 and the reading was still wrong: **what a family palette
sells is that green means Islam, and nine greens above it cancel that whether or not any single pair
is separable.** (It ended at 98, and §6.14c later pulled it to 80.)

**And it is two arcs, 30→44 and 56→98, with its own hue 50 in the gap.** A band had to sit entirely
on one side of its root, by the rule above; that put its floor at 52 and left it 46°. Moving Hinduism
to `#fa7420` opened the 23→50 arc, and taking it means straddling the family's own hue — so a band is
a list of arcs and positions are allocated over them laid end to end. 56° instead of 46°, which is
the truer picture anyway: a family's own shade belongs among its branches rather than off one end of
them. **That expansion is for the look and not for legibility** — widening does almost nothing (see
the table below). What it buys is that the run now *starts* in orange beside Hinduism instead of in
yellow. It does not buy Catholic against Orthodox and nothing will.

**Moving one family moved four.** Hinduism 35 → 23 landed on top of Ravidassia at 25 and left Sikhism
7° away, and the whole warm end had to be re-authored, because Buddhism 356 and Hinduism 23 leave
Sikhism **27° of arc** with both neighbours drawn at size wherever it appears — Canada, the UK, New
Zealand, Australia. Hue cannot separate three families at that spacing, so **Sikhism is the one
colour in the table that is searched rather than chosen**: the most saturated value in the arc
clearing dE 28 from both. `hsl(7, 100%, 70%)`, a coral — lighter than a "red-orange" would normally
be drawn, and **that lightness *is* the separation**, against Buddhism at L 45 and Hinduism at L 55.
dE 31 from each. Ravidassia gave up its place and now sits **beside** Sikhism at dE 13, which is the
honest statement: it declared itself separate in 2010, most of its people are still counted Sikh, and
there is no room in the arc to claim otherwise. Closest pairs the new warm end leaves, all clearing:
Catholic/Hinduism 34, Hinduism/Sikhism 31, Buddhism/Sikhism 31, Islam against Christianity's last
branch **26.8** — the tightest thing in the palette and the reason the band stops where it does.

**Inside the band, the six LINEAGE groups (§6.5) take an equal share each and stay contiguous**, so
the ancient communions are one run of yellows and the Pietist–Wesleyan line another, and the pairs a
reader compares — Catholic against Baptist against Pentecostal — are in different groups and
therefore a whole block apart. Equal shares per *group* and not per member: weighting by size would
be better on the map and would move every colour on the next retile.

**THE HONEST LIMIT, AND IT IS THE POINT OF THE SECTION.** 22 Christian branches in 46° cannot be told
apart, and no allocation fixes it:

| lever tried | worst pair among the branches over 500 dots |
|---|---|
| three tiers, as §6.3 uses | dE 2.9 |
| six tiers | dE 6.2 |
| widening the band as far as Islam allows | dE 7.6 |
| authored size weights, up to 8× on Catholic | dE 6.9 — no change |

Weighting does nothing because the constraint is not how the arc is divided, it is how long the arc
is. So **the overview does not claim branch legibility and should not be tuned as though it might
acquire it: it claims that a dot's family is readable at a glance, and that one click gets the
branches back.** That is §6's own sentence arrived at a second time, by measurement rather than by
argument. (§6.14 then took the conclusion one step further and stopped drawing the branches in
distinct colours at all.)

**`tools/check_overview.py` is the checker, and it asks two questions rather than one.** Pairs
*across* families under dE 25 are failures, because the first thing a dot has to say is which family
it is. Pairs *inside* one family are held to dE 12 and reported separately, because the overview
means those to be close and a single bar would bury the two real failures under 300 expected ones.
Colours come out of `index.html` itself through `tools/palette_dump.js`, which slices the allocator's
own declarations out of the inline script and runs them in node — so, like `check_palette.py` reading
`ROOT_HSL`, **the checker cannot drift from the palette it measures.**

### 6.10 A row too small to be worth a line folds into one — DECIDED 2026-09-03, denominator REVERSED 2026-09-07 (§6.10a)

The legend has a fixed budget and the tree does not. At depth 2 the all-religions view drew 28 rows
under Christianity and five of them — Church of the East, Hussite, Moravian, Plymouth Brethren,
Swedenborgian — were **115 dots between them against Christianity's 501,322**.

So a child holding less than **1e-4 of its parent's total** folds into a single row, provided at
least two of them do. Two properties earn their place:

- **The denominator is the parent, not the map.** Inside Judaism a row is measured against Jews. That
  is what makes the rule scale-free, and it is why Reconstructionist Judaism (39 dots, but 1.3% of
  Judaism) stays and Hussite (23 dots, 4e-5 of Christianity) does not.
- **Two is the minimum**, because folding one row into a row that says "1 small group" saves nothing.

**Nothing is hidden by it.** The bucket expands, its members keep their counts and stay selectable,
and selecting one takes it out of the bucket for as long as it is selected — you cannot select a
thing and have it disappear. What the fold removes is the *claim to a colour*: the folded groups take
their family's own shade, which is the true statement about a speck at this scale.

Like §6.6's `unspecified`, the bucket row is **not a node** and is computed in the viewer from the
tallies. It sits last under its parent, being the residual of the list above it.

**It fires where the tail is long and nowhere else**, which is the check that it is measuring the
right thing: four rows fold in the all-countries view and none in the United States or Czechia. 2e-4
would also take Maori Christian churches and the Quakers; the constant is one line.

### 6.10a The denominator is the VIEW, not the parent — REVERSED 2026-09-07

§6.10's first bullet is now wrong, and it names its own counter-example: *"Reconstructionist
Judaism (39 dots, but 1.3% of Judaism) stays"*. Anita looked at that row and said it should not.

> *"i think maybe it works based on percentage of parent like percentage of judaism. thats not
> right. i want to do as percentage of everything we're looking at (so if we're looking at american
> christianity, as percentage of american christian count. if we're looking at world all religions,
> as percentage of world all religions count)."*

**The parent rule is coherent and answers the wrong question, and two rows show it exactly.**

| row | share of its parent | people | dots (world view) | old rule | new rule |
|---|---|---|---|---|---|
| Reconstructionist Judaism | **0.292%** of Judaism | 39,000 | 39 | keeps a row | folds |
| Shia Islam | **0.049%** of Islam | 430,000 | 430 | keeps a row | keeps a row |

Shia's share of its parent is **six times smaller** and it is **eleven times more people**. A share
of a parent is only comparable between parents that contain comparable things, and these do not:
Islam's total is 98% `unspecified` (864m of 879m), because almost no source that reports Muslims
divides them, while Judaism's total is nearly all named denominations. So the same fraction means
"one of the few named Muslim bodies" under one parent and "a rounding error" under another.

**Measured against the view, both come right**: 39 dots in 5,062,789 folds and 430 does not.

**It is the same denominator the share bars use (§10.3)** — the scope's total, or every root's when
nothing is selected — and that matters more than the arithmetic does. **A reader watching a row
vanish into the bucket can see why in the bar beside it: the fold is the point where the bar has
nothing left to say.** Two mechanisms that both answer "how big is this here" cannot be allowed to
answer it differently on the same row.

**5e-5 is measured, not picked.** It is the largest cut that folds nothing anyone would look for:

- every one of the 37 rows the parent rule folded **still folds** — Moravian, Plymouth Brethren,
  Pietist and Hussite among them, which are §6.10's own motivating cases;
- Shia, Mahayana, Ahmadiyya and Anabaptist all keep their rows;
- the all-countries legend loses **194 rows into 34 buckets**, against 37 into 9.

Raising the *parent* cut to reach the same two Judaism rows would have needed 3e-3, and at that
value **Shia Islam, Mahayana Buddhism, Ahmadiyya and Eastern Catholic all fold** — measured, not
feared. That is the whole argument against the old denominator in one line.

**WHAT IT COSTS, and it is real.** A country's total is a fiftieth of the world's, so the same
fraction is a far smaller number of dots, and the per-country tails the parent rule folded stop
folding: **the United Kingdom goes from 7 rows folded to none, Poland from 5.** That is the honest
consequence of the denominator asked for — a group that is a real part of Poland is not a rounding
error because Poland is small — and it points the same way as §14's general rule about not letting a
country's size decide how finely it is drawn. **If the country legends turn out to want it back, the
fix is one `||` putting the parent test alongside this one, not a different threshold.**

**`FOLD_MIN` was never the blocker and has always been 2.** It is worth writing down because it was
the first suspect: *"is 2 not enough? lets make the threshold 2 so that 2 is enough."* It was
already 2, and `small.length >= FOLD_MIN` already admitted a pair. The size cut was the whole of it.

### 6.11 The reader gets the hand overrides too — DECIDED 2026-09-03

§6 said "hand overrides are expected, and bounded to a few scopes", and meant Anita editing a table
in the source. `PIN` and `PIN_OVERVIEW` are that. This adds the same two powers to the legend itself,
because the argument for them does not depend on who is holding it: **click any swatch to set that
group's colour or to hide it.**

**Hiding stops being a special case.** `unaffiliated` started hidden for §6.2's reason, implemented
as a constant plus a boolean — so it was the only thing in the taxonomy that could ever be hidden,
and "no religion: hidden" was a feature rather than an instance of one. It is now simply the entry
the hidden set starts with (and that set is now empty). Baptists can be hidden by the same mechanism
and cleared by the same reset.

**A hand-set colour wins over both palettes, in every scope and every country** — which is what makes
it worth setting. It is applied at the **drawn category**, though, not to the node's dots wherever
they fall, so §6.3's flat rule survives: pin Amish purple while Anabaptist's children are drawn and
Amish is purple; take the cut back out and Anabaptist paints its whole subtree, Amish included. The
alternative is a colour on the map with no swatch in the panel, which is the defect §6.3 measured at
93% of dots and fixed.

**The picker is ancestrydots', ported unchanged in style** — Anita's call, and the right one: that
one is already known to be pleasant, and there was no reason to invent a second idiom for the same
job. Three gradient sliders whose saturation and lightness tracks repaint from the current hue, a
preview swatch with the hex, and a sixteen-colour preset grid. **Not the browser's native colour
dialog**, which is an OS window that covers the map you are picking against — and the whole point of
picking here rather than in the source is watching the dots change while you drag. The presets are
deliberately **not** the family hues: offering Islam's green for a Christian branch would invite the
exact collision the authored tables exist to prevent.

**What is set is kept in `localStorage`, and that is a per-browser promise, not a per-map one.** So
`copyOverrides()` puts the current overrides on the clipboard **in `PIN`'s own `[h, s, l]` form** — a
block that has to be translated before it can be pasted is not a paste-ready block. That closes the
loop §6.8 opened: a stable palette is worth hand-correcting, and a correction that lives in one
browser is not a correction. The route is picker → `copyOverrides()` → `PIN` and `PIN_OVERVIEW`.

**It is a console call and not a button — Anita's edit**, and the reason is worth keeping: the panel
is the legend, its audience is readers, and getting a colour back into the source is a maintainer's
errand. **A control that only one person will ever press does not belong in the key to the map.** One
reset serves both, in the same blue-link idiom as the other legend controls, greyed out when there is
nothing to reset.

**What made it usable, found by using it.** Three things, and the third is the one worth writing
down:

- The `hide` and `reset` links describe the node's *current* state, so they are recomputed on every
  change and not only when the popover opens. Written once at open time, setting a colour and then
  reaching for `reset` found it still greyed out.
- The dot stays 9px because that is what reads as a legend key, but the hit area is a square the full
  height of the row, with negative margins. **A 9px circle is a legend mark; it is not a button.**
- **A live colour preview costs more per update than anyone guesses, and the fix is to gate on
  completion rather than on time.** Dragging the hue slider left the map still changing colour ten
  seconds after the pointer came up. Three things were wrong and all three had to go:

  1. `applyPaint` re-set the layer **filter** on every call, and a filter change makes MapLibre
     re-evaluate it against every feature in every loaded tile. A colour drag does not change the
     filter. `applyColors` sets only the colour expression.
  2. It painted `dots`, `atomic` and `rings` every time, and **exactly one of the first two is ever
     visible** (§4.2b) while rings are usually off. A hidden layer costs the same re-evaluation as a
     visible one. A live drag touches only the layer on screen; the others are brought up to date
     when the drag ends.
  3. **Both a per-frame and a per-120ms throttle still backed up.** `circle-color` is a data-driven
     `match` over the node id, so each change makes MapLibre re-evaluate the property per feature and
     re-upload vertex attributes for every loaded tile — well over 120ms. **Any *timer* is guessing
     at a cost it cannot see, and the surplus piles up inside MapLibre where no timer of ours can
     reach it** ([[reference_maplibre_paint_perf]]). So the gate is **completion**: issue a repaint,
     wait for the map's `idle`, then issue the next — and only ever the current value. Queue depth
     is one, and a slow phone simply draws fewer intermediate states, which is the right way to be
     slow. A timeout guards it, because `idle` is not contractually guaranteed and a pipeline that
     can wedge is worse than one that occasionally double-paints.

  Measured on a 600-event, 1.4-second drag: **3 repaints during, 2 after release, settled in 537 ms**,
  on the value the slider ended at. Before: ten seconds of catching up. *(§4.2d has since taken the
  scatter out of MapLibre's paint path entirely; the merged tile layer still goes through this.)*

### 6.12 An empty map means two opposite things, so say which — BUILT 2026-09-05

Select Sikhism and Poland draws nothing. That is either *GUS asked and almost nobody said Sikh* or
*GUS never offered the category and Poland's Sikhs are inside a residual*, and those are opposite
facts that look identical. §3.5 already insists undercounting is marked rather than filled; this is
the same rule one level up, about **the question rather than the answer**.

So countries whose source **can see** the selected node are lit, and three states become readable
where there were two:

| | |
|---|---|
| lit, with dots | the source asked, and these people are there |
| lit, no dots | the source asked, and essentially nobody said it |
| unlit | the source never offered the category — **not** evidence of absence |

**A COUNTRY WHOSE CLASSIFICATION CHANGES INSIDE ITS OWN BORDERS CANNOT BE ONE SHAPE — found by Anita
in the layer built to prevent exactly this error.** The United Kingdom is three censuses: England and
Wales publish no Christian denomination at all for 27.5 million people, Scotland names the Church of
Scotland and the Roman Catholics, Northern Ireland names twenty-two bodies including four kinds of
Presbyterian. Selecting Latin Catholic lit the whole UK while dots appeared only around Glasgow and
Belfast — and the empty half read as *England asked and nobody said it* when the truth is that
**England was never asked**.

So the wash draws **regions, not countries**: usually one per country and equal to it, three for the
UK, keyed off `source_id` the same way `_uk_counts` reads them (`coverage.py` `UK_REGIONS`), with
geometry from Natural Earth's `admin_0_map_units`. What falls out is worth having beyond the Catholic
case: **Sikhism lights England-and-Wales and Scotland but not Northern Ireland**, whose category list
is Christian-focused, and no country-level shape could have said so.

The caption still counts **countries**, because "13 of 21 countries" is the sentence a reader wants
and "16 of 24 regions" is an implementation detail; a partly-covered country counts once. With
exactly one it names it instead — *Australia records this*.

**Coverage is the mapping's targets, not the data's contents** (`coverage.py`). A category on the form
that scores zero nationally still counts as asked; reading coverage off the counts would collapse the
two cases this exists to separate. A country covers a node if it targets that node **or anything
below it** — a census naming five Baptist bodies can answer "Baptist", one offering only "Christian"
cannot, and **mapping to an *ancestor* is not coverage**. The check that keeps it honest is that
every node which draws a dot must be in its country's coverage; `coverage.py verify()` fails
otherwise, because the alternative is a country going dark for a religion it demonstrably contains.

**Three things the implementation had to concede to the basemap.**

- **LIT, NOT GREYED OUT — and that is arithmetic, not taste.** The instinct is to dim the countries
  that are out of scope. OpenFreeMap dark paints land `rgb(12,12,12)` and water `rgb(27,27,29)`: the
  land is already darker than the sea and all but black, so there is no headroom to dim into. **On a
  dark map the same distinction has to run the other way.**
- **The lift must clear the WATER, not just the land**, and it is **neutral grey**: a tinted wash
  competes with whatever it sits under — at `#8fa0c0` it was a blue veil beneath Judaism's blue dots
  — so it separates by lightness alone and cannot collide with a §6.3 hue.
- **It settled at 0.1, walked down from 0.22 through 0.15 against the real map.** At each earlier
  step it still read as a thing in its own right rather than as backing for the dots. At 0.1 it is
  `rgb(27,27,27)` — level with the sea in lightness and told apart from it only by being neutral
  where the sea is faintly blue. Quiet is the correct target: it is the ground the dots stand on.

**The geometry started deliberately crude and that was wrong.** Natural Earth 110m is 59 KB for
twenty countries, and the argument for it — a continental-scale question needs no precision — does
not survive contact with an archipelago: 110m draws the Philippines as **seven polygons and 110
vertices**, so at z6 the wash was straight lines cutting across the Visayas with dots on both sides.
**A reader cannot tell a deliberately coarse boundary from a broken one**, and a layer whose job is
to say *this country is in scope* fails the moment it stops looking like the country. 10m gives the
same country 97 parts and 7,406 vertices for 3.3 MB across twenty — one cached, gzipped fetch,
smaller than the coarse dot buffers.

**10m is still visibly coarse, and the fade is the answer rather than a finer file.** The simplify
step costs only 26% of the vertices, so removing it buys 26% more detail for 35% more bytes and still
leaves the Philippines at 97 parts. So the wash runs full strength to **z6.5 and is gone by z8.5** —
out before its own resolution can be examined. That fade earns its keep twice: the question it
answers stops being interesting once you are inside one country, and the coastline stops being
convincing at about the same zoom. The exact fix, if ever wanted, is dissolving the placement layers
— **1,915 parts for the Philippines against 97** — at ~20 MB and minutes of build.

**THE CAPTION NAMES THE COUNTRIES — added 2026-09-06, Anita's ask.** "17 of 31 countries record this"
says the wash is partial without saying which part, and at the zoom the caption is read at, a lit
country is a slightly lighter patch among thirty others. The tooltip lists them.

**The list is the whole tooltip**, and that took a second pass the same day. It shipped with two more
blocks: a dimmed *"No such category"* list, and a paragraph explaining what lit means. Both are gone —
the first is the negation of the list above it, and the second is the about panel's job. Anita:
*"lets reduce the tooltip text… just keep the list of countries."* **Three blocks of text hanging off
a one-line caption is a reason not to hover it a second time.**

**It is a custom tooltip, not `title`, and that is the load-bearing half.** The browser's own waits
about a second, cannot be styled, and **does not exist at all on a touch screen** — so on a phone the
caption's explanation was simply unreachable. `data-tip` names a key into a registry that holds the
markup, rather than carrying HTML in an attribute, because escaping it shows the reader the tags and
not escaping it is an injection waiting for the first country with an ampersand in its name.

**And the wash's polygons now have a second reader**: §6.2's Auto uses them to answer which country
the camera is over. The file is fetched once and parsed once.

### 6.13 Christianity's branches are authored, not allocated — DECIDED 2026-09-05

Anita, looking at the focus view: the Reformation churches were yellow beside Catholic, and the seven
small nineteenth-century restorationist bodies held the whole blue range on their own. The allocator
was not wrong, it was **indifferent to which pairs a reader actually compares** — no rule inside it
knows that Lutheran-against-Catholic in Germany is the comparison that country exists on this map to
support.

So Christianity's 29 branches are a table (`PIN`, 32 entries) rather than an allocation, and the
groups are placed by how much separation each needs:

| hue     | group                        | why there                                              |
|---------|------------------------------|--------------------------------------------------------|
| 0–43    | Ancient communions           | red → gold. Oriental Orthodox `#e27373` and Catholic `#ba8a12` are Anita's exact colours. |
| 43–72   | *empty*                      | the buffer. Yellow is Catholic's.                      |
| 72–105  | Separatist and believers'    | yellow-green.                                          |
| 116–160 | Pietist and Wesleyan         | green → teal.                                          |
| 172–221 | Restorationist and adventist | teal-blue → blue. Squeezed 70° → 49.                   |
| 232–265 | Reformation                  | blue → blue-violet, and the point of the rearrangement.|
| 292–340 | No single line               | purple → magenta, minus `protestant`.                  |

**It took two passes, and the second is the instructive one.** The first put the Reformation at
248–278, which is violet rather than blue-violet, and left Anglican 14° from Non-denominational — two
of the largest Protestant rows reading as the same purple. Anita: *"move anabaptist through lutheran
all toward yellow in hue. i think reformation is much too purple and close to non denominational
now."* Pulling the four groups below it down 12–16° is what buys the Reformation a range that is
unmistakably **blue**. The buffer lost 10° of its 39 and did not need them: Anabaptist at 72 measures
ΔE 42 against Catholic.

**This breaks §6.5's coupling between panel order and hue order, knowingly.** The panel still reads
in descent order and the colours no longer run down it — Reformation is second in the list and sixth
on the wheel. §6.5's argument was that a legend sorted by *something* beats one sorted by size, and
that survives; what does not is the claim that one order can serve both jobs.

**`christianity.protestant` leaves its own lineage group** and sits beside Lutheran rather than at
the end of the residual block — Anita: *"mostly a thing in germany, so it reflects lutherans"*.
`evangelisch` is 19m people and overwhelmingly the Landeskirchen. It is lighter than Lutheran because
it is still a residual, but **not a pale tint** of it: the first attempt at `[250, 36, 84]` came out
a near-grey lilac and read as absence rather than as a colour.

**Contrast is the binding constraint and it is strongly hue-dependent** — at hue 240–270 nothing under
L 55 clears the 2.9 floor, while at hue 60 anything over L 21 does. So the period-three tier cycle is
kept for *adjacency*, and the tier is overridden wherever a large node landed on a dark tier in a
dark part of the wheel. `check_overview.py --focus christianity` reports **zero** close pairs and
zero dim categories at depths 1 and 2.

### 6.14 The overview draws a lineage group as ONE colour — DECIDED 2026-09-05

Anita: *"i think there are just too many christian colors ... each of the other groupings should be
mapped onto a single color (they can stay as different rows, they should just have same color)."*

The arithmetic says how right that is. At depth 2 Christianity drew **24 rows against Islam's one**,
so five sixths of the legend was one family — and the band exists precisely to say that a Christian
dot is Christian *first* (§6.9). Twenty-four shades inside 56° never separated anyway. So the map
stops pretending: six colours, one per lineage group, and a group is a thing a reader can learn.

**The rows stay.** Four Reformation rows with the same swatch have not lost information — names,
counts and tree are all still there, and pressing `+` still separates them, because flattening paints
the *member* and leaves its subtree dividing the group's slice as before. What it drops is the claim
that Lutheran-vs-Anglican is visible on a world map at 2px.

**Three tables, three different powers**, applied in this order:

- `OVERVIEW_FLAT` — one authored colour for a whole lineage group. Keyed by parent id *and* group
  label, so Judaism's "No single line" cannot collide with Christianity's.
- `OVERVIEW_ARC` — one node **and its whole subtree** moved to an absolute arc outside the family
  band. Beats `OVERVIEW_FLAT`; it is how Catholic escapes. Subtree inclusion is the whole point — a
  plain pin would leave Eastern Orthodox red at depth 2 and yellow again at depth 3.
- `PIN_OVERVIEW` — one node, one colour, subtree not included. The blunt last resort.

**Catholic escapes its group** — 434m dots, more than half the Christians on the map, and Anita named
it separately. It keeps its gold. What it leaves behind is the two Orthodox communions and the Church
of the East: the three ancient non-Latin communions, a coherent thing to be one colour.

**Reddish orange for them, borrowed from Hinduism and Sikhism.** Eastern Orthodox against Catholic
measured **ΔE 8.2 over 17,506 and 434,399 dots**, the worst big pair on the map, and nothing inside
the band could fix it: a lineage group is contiguous by construction, so the four shared 9° of yellow
of which Catholic was 434m. 16→26 is the arc between Sikhism and Hinduism and the only room the warm
end has. Anita's call: *"they can come close to overlapping with hindu, they dont show up in the same
places so i think it sok."* Measured, `#fc9964` clears 31 from Sikhism, 29 from Hinduism, 46 from
Buddhism and 29 from Catholic. **Sikhism moved to hue 0** (`#ff6666`) in the same pass, costing
31.0 → 27.4 against Buddhism and buying 31.0 → 36.5 against Hinduism.

**The band is one arc again, and a flattened block takes a smaller share of it.** 30→44 existed for
the ancient communions and they have all left it, so `christianity` is `[56, 98]`. And a group that
draws one colour needs a slice only for its members' *children*, so it takes `FLAT_SHARE` (0.2) of a
normal block. What it gives up goes to "No single line", the one Christian group still dividing —
seven rows including Protestant-unspecified at 42m and Non-denominational at 22m, which used to share
9° and now have 21°. Families with nothing flattened are untouched.

**`other` stops being twenty rows.** Anita: *"i'd rather have Other, by source stay collapsed... i
dont think theres much value in seeing these rows."* §3.11 gives every source its own residual
container, which is right for counting and pointless to look at — nineteen greys, one per country,
differing by a tier. `OVERVIEW_LEAF` makes the family itself the drawn category in the overview at
any depth, so it is one grey row with one swatch. Selecting it still opens the whole list. And the
panel now starts any node closed when it is *itself* the drawn category and none of its children are.

**Two rows then left the green end, both on §14.2's argument.** Anita: *"lets maybe have L2 filipino
independent churches be not green, as the philippines does have muslims"*, and *"other christian
looks extremely green. would rather color it same as yellow unspecified tbh"*.

`christianity.other` takes the family's own yellow, and it is the row least entitled to a colour of
its own: a source's "other Christian" residual and `christianity`'s own "Christian, n.o.s." are the
same answer given to two differently worded forms. Three rows now share that swatch — those two and
§6.10's folded bucket — and the swatch means "Christian, no body named", which all three are.

`christianity.filipinoindependent` is the harder one, because it is 3.0m people almost entirely in the
one country where **green has to mean Islam**: 6,981 dots of Muslims in Mindanao and Sulu. Anita asked
whether it could share a Protestant group's colour. It cannot, and the tree already says why — Iglesia
ni Cristo is nontrinitarian, *"which is why it is not filed among the Protestant families it is
otherwise sometimes grouped with"* (`branches.py`). The nearest kin is Restorationist and adventist,
and that was tried and rejected: in the Philippines that group is 2,113 dots of Adventists,
Witnesses, Mormons and Stone-Campbell churches, all American imports, and painting the country's third
largest religious body as one of them is a worse claim than the green was. So it takes the gap between
Catholic's arc and the pale Protestants.

#### 6.14a The bronze was measured right and looked wrong — CHANGED 2026-09-06

It shipped as 46→54 at `[62, 38]`, `#9d8925`, and the pair test was fine: ΔE 28 to Catholic, 58 to
Islam, nothing under the floor. Anita: *"a bit out of place how brown it is."*

**The failure is one no pair test can see, and it is about the register rather than the distance.**
Every other Christian category in the Philippines is bright — Baptist at contrast 7.45, the pale
Protestants 7.43, Pentecostal 6.91, Non-denominational 6.14, Catholic 4.36. The bronze was **2.59**,
the only dim colour in the family, so the country's third largest religious body read as a smudge
among six clean yellows. **ΔE says whether two colours can be told apart and says nothing about
whether one of them belongs, and a palette checked only pairwise will keep producing this.**

**And the reason it was dark is real, which is why the fix is a hue move rather than a brightening.**
Hue 50 cannot be light: it is **Christianity's own root hue**, so the family's `unspecified` row draws
there at `[92, 64]`, and a light gold at 50 measures **ΔE 1.3** from it — §6.9's *"a band must not
contain its own root's hue"*, met from the outside by a node that had left the band. At 50, lightness
was the only free axis and downwards the only direction.

Moving to **hue 46** buys the light register, because the whole 38→56 span is empty apart from that
root hue. **`#d3bb69`**, contrast 4.74: ΔE 28.0 to Catholic, 28.0 to Christianity's own `unspecified`,
27.5 to the pale Protestants and 58.5 to Islam — every one wider than the bronze's worst.

#### 6.14b The gold is RESERVED, and three nodes share it — DECIDED 2026-09-06

Anita, immediately after 6.14a: *"have some yellow color be reserved for things similar to filipino
independent churches … is there anything its similar to we can just group it with?"*

**There is, and `branches.py` had already said so twice without acting on it.** Both
`christianity.filipinoindependent` and `christianity.africaninstituted` carry the sentence *"the same
idea as `christianity.maori`"*. All three were nevertheless drawn in three unrelated colours: a
searched apricot for the African churches, the new gold for the Philippine ones, and a **lime green**
for the Māori ones — the last being §14.2's Philippine problem in a second country, and ΔE 7.3 from
United and uniting churches, small enough on both sides that the checker never reported it. **The tree
asserted a grouping the map contradicted.**

They are now one LINEAGE group, `christianity / Locally founded churches`, drawn as one swatch by
§6.14's rule.

**What the group is.** Churches founded by local converts in a mission field, outside missionary
control, belonging to none of the imported families afterwards. It is a standard bloc in the study of
world Christianity — Barrett's *World Christian Encyclopedia* calls it "Independents" — and Kenya's
own census category, *African Instituted Churches*, is the same construction.

**What it is not, because it is easy to over-read.** The group is defined by a church's **relation to
a mission**, a fact about colonial history rather than about doctrine, and the three do not share a
theology: the African and Māori churches are broadly trinitarian mission-church breakaways that added
prophecy and healing, while the Philippine node is explicitly the *nontrinitarian restorationist*
remainder. On doctrine, `maori` and `africaninstituted` are the close pair. **A shared colour asserts
the historical grouping, which is exactly what "Ancient communions" and "Reformation" assert; it does
not assert a creed.**

**And one colour costs nothing here.** The three never co-occur — Kenya, the Philippines, New Zealand,
Australia. Where §6.14's flat groups normally trade separation for legibility, this one trades
nothing: the shared `#d3bb69` measures ΔE 27.5 at worst across all four countries, against the
searched apricot's 25.9 and the Māori green's 7.3. **Both of the other two improve.** It also retires
a `PIN_OVERVIEW`.

#### 6.14c Christianity's band ends at 80, not 98 — CHANGED 2026-09-06

Anita: *"maybe lets pull them out of green yeah theyre a bit close."*

**Why the green end was occupied at all, which is the part worth keeping.** `No single line` is the
last group in Christianity's lineage order and the only one that is not flattened, so it takes weight
1.0 against each flat group's `FLAT_SHARE` of 0.2 — **45% of the whole band, at the top**. Its six
members then divided 56→98's upper half. And **three of the six are pinned** (`protestant`,
`evangelical`, `other`), with `other` in `OVERVIEW_LEAF` as well, so half the group's arc was spent on
nodes that never draw a band colour at all — pushing the three that do into chartreuse.
`nondenominational` is 21m people in the US, the second largest Christian node there, and it sat at
hue 84.

**80 is a floor found by bisection, not a chosen number.** At 78 and below, `nondenominational` lands
within 4° of the Pietist and Wesleyan revival swatch and measures **ΔE 11.9 at 78, 10.5 at 74** —
against Pentecostal's 49,481 US dots, which is big on both sides and a genuine failure. At 80 it sits
at hue 72 and the checker flags nothing involving any of the three, in any country.

**What it bought:** `nondenominational` 84 → 72, `united` 80 → 70, `messianic` with them, and the
**Christian family now stops at hue 72 with a 66° gap to Islam at 138**, against 54° before. Close
pairs over the size threshold go from **88 to 60** across all countries, and every one that remains
was there before, in the indigo→magenta wedge.

**What it did not buy, stated so the next person does not re-derive it.** Hue 72 at `[83, 56]` is
`#c7ec32`, still a chartreuse, and the band cannot go lower. If that is still too green the remaining
move is a **pin**, and it needs a decision rather than a search — the honest warm slots left are the
light apricot the African churches vacated in §6.14b, or the Separatist and believers' churches
swatch `evangelical` already uses. The second is the interesting one and it is a claim, not a colour
choice: it says most American non-denominational congregations are baptistic believers' churches,
which is largely true and **is not something a palette should assert without being asked.**

### 6.15 A branch whose `unspecified` row is its own child — DECIDED 2026-09-05

Selecting Catholic drew **Latin Catholic at 407m and, above it, an `unspecified` row at 24m in a
different colour**. Anita: *"they seem to be the same."* They are. The 24m are the sources that answer
"Catholic" and stop — Germany's *Römisch-katholische Kirche*, StatCan's *Catholic*, Croatia's
*Katolici*, Hungary's *Catholic, rite not stated* — against the four that name the Latin church
outright. **That is a difference between census forms, not between people**, and §6.6's rule was
drawing it as a permanent second colour for the largest thing on the map.

So `MERGE_OWN` names the child a branch's own dots belong to. The parent's dots take that child's
colour, the `unspecified` row is not drawn, and the child's row counts both — 432m in one gold. The
parent stays a heading with a blank mark, exactly as §6.6 leaves any split branch.

**What it costs.** Hungary's "rite not stated" and Croatia's "Katolici" genuinely include Greek
Catholics, who have their own row two lines down; those people are now drawn as Latin. Hungary reports
179k of them separately and it is the unstated remainder that moves, which nobody can size. Taken
deliberately: an invisible error of at most a few per cent inside one branch, against a visible and
permanent second colour for 434m people.

**It is not a data fix, and the data fix is the honest version.** Mapping those source categories onto
`christianity.catholic.latin` at ingest says the same thing where it belongs — and needs a retile of
eight countries. Doing it in the viewer keeps the claim readable and reversible in one line, which is
the right place for a claim this size to start.

## 7. Confidence is carried, not drawn — REVERSED 2026-09-04

**Confidence must never be expressed in colour.** A desaturation was built on 2026-09-04 and removed
the same day. Anita: **every colour on the map has to be a colour in the legend.** It failed on
contact with §3.5a, which is what it was built for: 51% of the American dots are the survey residual,
so Christianity's own `unspecified` row drew as a **dull tan while the legend beside it showed bright
yellow**, and a large category appeared on the map in a colour that was nowhere in the key.

The rule that replaces it is general, and larger than fading: **the legend is the whole colour
vocabulary, and a reader matching a dot to a row must always find it**
([[feedback_legend_is_the_palette]]). Anything applied to a colour *after* the palette is authored
breaks that by construction, however principled the modifier is. That rules out desaturation, opacity
and lightening alike, and it is why the fix was not a gentler fade.

**It runs both ways, and the second half was found the same day.** §6.3's flat rule already said every
colour on the map has a swatch in the panel; the converse — every swatch in the panel is a colour on
the map — was not true. Headings showed a **grey dot**: a §6.6 split branch, whose own colour belongs
to its `unspecified` row, and a branch that is not a drawn category at all (`other` above `other.us`,
`indigenous` above `indigenous.northamerican`). There are no grey Buddhism dots anywhere on the map.
Those rows now show a **blank the width of a dot**, which keeps the labels aligned and leaves the
count where it was, and they are no longer click targets for the colour editor, because a heading has
no colour to set or hide. The test is the fallback colour rather than a list of node kinds, so a third
case gets the blank for free.

**Kept from the removed fade, because it will be true of any replacement: the palette's own saturation
varies 7×, so no single modifier applies evenly.** Roots are authored from saturation 83
(Christianity) down to 12 (No religion), and a saturation multiplier cannot make an already-grey
colour greyer — `unaffiliated` moved #92a2aa → #8c9397, which nobody can see, and that is exactly
where §3.5a's residual lands. Any future confidence treatment has to be uniform in the thing it
changes, which is the second reason not to reach for colour: **colour is the one channel the palette
has already spent.**

**What survives.** `tier` travels from the adapter in `countries.py` through `scatter.py` to a `t` on
every dot and into the tiles and buffers — it is a true fact about the row, `measured` is the default
and is written nowhere, so it costs no tile bytes. §7a is what draws it.

Three tiers, from the source inventory, per unit per group:

| tier | what it means | example |
|---|---|---|
| `measured` | a census or register question, asked in this unit, any year (§3.4), **and answered** | ASARB's county rolls; India's six religions at sub-district |
| `derived` | somebody counted this and the number was carried to a finer place — §3.4's structure-from-older-source, or a coarse total distributed by a proxy | Ireland's coarse census total; §3.10's allocation |
| `modelled` | nobody was counted at any level; a country or survey estimate spread by population | Russia's Arena rows; the US self-ID residual (§7b) |

**"Measured" needs a response rate, not just a question — found 2026-08-27.** StatCan dropped its
quality suppression for 2021, so **241 Canadian census subdivisions publish religion counts built on
≥50% long-form non-response**, and those numbers look exactly like every other number in the file.
Australia's religion question is voluntary with 6.9% non-response nationally, and England, Wales and
Scotland are voluntary too. So the measured tier should be gated on the unit's own response rate,
recorded per row (`tnr_lf=` in the Canadian rows). **Still unbuilt: no adapter records it yet.**

**The tier belongs to the people, not to the (unit, node) pair — FOUND 2026-09-04.** The first build
took the weakest tier on each pair, reasoning that a pair which is part census and part spread-out
total is not a measurement. That is true of the pair and false of its people, and the result was
backwards: **Ireland's 4.30M measured people and 508k derived ones drew 764 measured dots against
4,030 derived**, because most Catholic pairs carry one large measured row beside one tiny allocated
one, and taking the weakest let the tiny row relabel the lot. The fix is not a better rule but a
smaller key — `tier` joins `(unit, node)` in the grouping, a mixed pair becomes two rows, and the dots
divide in proportion by construction. It now draws 4,295 against 499. **The general form: a qualifier
on a row must not be aggregated over the rows it qualifies; it must key them.**

### 7a. The non-colour treatment, second attempt — BUILT 2026-09-05

§7 ended with "if confidence is drawn again it has to be in something that is not colour". This is
that thing, and §14.5 makes it a precondition rather than a nicety: **no ethnicity-derived country
ships until the map can say which of itself is counted.**

**It is a MODE, not a modifier, and that is the whole reason it survives where the desaturation did
not.** The palette is untouched. While `inferred dots: hidden` is on, the map is answering a different
question — *which of this did somebody count?* — and the answer is the measured map standing on its
own, with nothing added to the legend to look up. The rule the fade broke is not engaged, because no
colour is created.

**Hiding, not greying — Anita's call.** A dark grey mark was the first idea and the ramp already owns
dark grey: §6.3a-i puts `unrecorded` at `hsl(228,10,37)` and **Germany contributes 42.8M dots in it**,
so a grey "derived" mark would have collided with the one country a reader is already most likely to
misread. At 1.5px, gone is unambiguous in a way no colour is.

**It cost one comparison, because `buffers.py` had already done the hard part.** The tier is packed
into the **top two bits of `ni`**, which is why the vertex shader's palette lookup always read
`mod(a_ni, 16384.0)`. So `a_ni >= 16384.0` *is* "not measured", and a `step()` against one uniform
collapses the quad exactly as a hidden node's zero alpha does. **No new attribute, no buffer change,
no rebuild, no tile change.** The merged tile layers get it from `layerFilter` as `['!', ['has',
't']]` — `tiles.py::tier_codes` writes `t` only when it is 1 or 2, so its absence is the test. Rings
carry no `t` and pass unaffected, which is right: a ring only exists where some row of its node was
measured. **Picking had to follow the same rule, and §4.3 is why it was not forgotten** — an invisible
mark that still answers a hover is the "Sikh ring says Wesleyan Church" bug in a new costume.

**The passive half matters more than the toggle.** §14.4 asks for the distinction to be kept
*visible*, and a control nobody finds keeps nothing visible. So the legend always carries the share of
the drawn dots that are not measured, counted off the buffers' own tier bits and cached per country.
**It is shown at 0% too** — the point is that a reader learns the distinction exists, which a control
that appears only when it would do something cannot teach.

**And the first time this map has said out loud how much of itself is counted** — the shares as they
stood on 2026-09-05, which are also a check on every adapter at once. Countries added since are not in
it; the viewer computes the live figure off the buffers' own tier bits:

| | derived or modelled | reads as |
|---|---|---|
| **ru** | **100.0% modelled** | every Arena row is `modelled` |
| **ca** | 71.3% derived | §3.10b's allocation, and the heaviest in the file |
| **us** | **50.9% modelled** | §3.5a predicted 51%. Half the American map, and it was `derived` until 2026-09-05 — see §7b |
| **nz** | 44.4% | 2018 structure inside 2023 totals |
| **br** | 41.2% | §3.4's rescale, 2010 categories inside 2022 municipal totals |
| **au** | 39.5% | §3.10a |
| **uk** | 36.0% | |
| **mx** | 22.0% | |
| **ie** | 10.5% | |
| **hu** | 3.1% | |
| **in** | **0.7%** | only `Other religions and persuasions` is derived; the six big religions are measured at sub-district |
| **cl cz de ee gh hr ke lk lt mk ph pl ro rs** | **0.0%** | measured at the geography their source publishes, with nothing carried |
| | **19% of the map overall** | |

**The three tiers stay ONE control and TWO numbers.** A three-state toggle would be more precision
than the question anyone asks, which is "what is counted", and both answers to that are "not this".
But the *readout* names them separately — `inferred dots: shown — 7% derived, 12% modelled` — because
they are different claims, and collapsing them into one 19% is what hid the fact that the bucket's two
largest occupants were Russia and the American residual. A zero term is dropped, and a country with
neither reads `all counted`.

### 7a-i. The toggle ROLLS UP, it does not hide — DECIDED and BUILT 2026-09-07 with Israel

**`inferred dots: hidden` used to remove every non-measured dot, and that is the wrong claim.** Anita,
looking at Israel: *"if we say hidden we should actually show the blue unspecified dots, rather than
hiding them."*

The problem is that §7's tier is **per row**, and a row can be measured at one level and inferred at
another. Israel's `judaism.haredi` rows come from a Jewish count CBS actually made, split by an
observance distribution CBS also published; what is inferred is only **which branch**, never that the
people exist or that they are Jews. Removing them said "these people are not counted", which is false
about 5.2 million of them — 71% of the country went off the map, and the control that did it was
labelled as a disclosure aid.

**The fix is a fallback, not a filter.** An inferred dot is redrawn at **the nearest ancestor its own
country actually measured**, and removed only if there is no such ancestor.

**"Nearest ancestor the country measured" and not "the parent", and the difference is the whole
correctness of it.** A naive parent walk would keep China on the map: its rows are derived from
ethnicity, nothing above them was ever counted, and rolling `islam` up to a root would assert a
measurement nobody made. The set of measured nodes is a fact about the *country*, not about the tree,
so it is counted off that country's own buffers — which the viewer already walks for §7's readout.
Verified both ways on screen: Israel keeps its dots and they turn Judaism-blue; **China empties
completely, which is what `100% derived` should look like.**

**The control is renamed `shown` / `rolled up`.** Leaving it saying `hidden` while five million dots
stayed on screen would have been the label lying about the thing it controls.

**Both renderers implement the same rule from the same helper** (`rollUpMap`), because `separate` and
`merged` disagreeing about what the toggle means would be worse than either behaviour alone. In the
WebGL path it costs no branch: the palette texture is twice as wide, the second half holds each node's
roll-up colour, and the shader shifts the lookup instead of collapsing the quad.

**The control says `shown` / `not shown`** — Anita, 2026-09-07, and the third label tried. `hidden` said
the people were gone and stopped being true; `rolled up` was accurate and was jargon. What is not shown
is the finer division, and that is what the words should say.

#### 7a-i-1. THE ROLL-UP TARGET IS THE SOURCE'S COLUMN, NOT AN ANCESTOR — DECIDED and BUILT 2026-09-07

`tools/check_rollup.py` reported **185.9 million orphaned against 27.1 million that roll up**, and the
first reading of that list was wrong about why. It said the big bucket was **placement**-derived —
"`allocate.py` only decided which small area each sits in" — and that is not what `allocate.py` does.
Its formula is `est[fine unit, leaf] = fine[fine unit, home(leaf)] × coarse share`: **the unit is
fixed**. The UK's 22.1 million `No religion` answers were counted at their own Output Area, and only
the split into Agnostic, Atheist and Humanist came from the MSOA. Nobody was ever moved.

**The real defect is that the roll-up walked the religion tree, while the thing a source measures is
its own COLUMN — and a column's node need not be an ancestor of what was split out of it.**

| | the cell the source published | the tree | before |
|---|---|---|---|
| **UK** | `No religion`, at the Output Area | `unaffiliated` is a ROOT — no ancestor exists | 22.1M vanished |
| **Hungary** | `Other Christian denomination`, at the settlement | `christianity.other` is a SIBLING of Baptist | 184k vanished |
| **Ireland** | `Other religion`, at the Small Area | `christianity.anglican` is nowhere under `other.ie` | 508k vanished |
| **Israel** | a Jewish count CBS made | `judaism` IS above `judaism.haredi` | rolled ✓ |

Only the last of those is an ancestor relation, which is why the tree walk worked for the country that
prompted the control and for almost nothing else.

**The arithmetic is exact, which is the argument for doing it this way rather than by judgement.**
`allocate.py` normalises its shares within a column, so a column's leaves sum back to the column's own
measured total. Rolling every leaf to the column's node reconstructs a number the source published.

**The rule that keeps the honest emptiness honest: the column must have been measured AT THE SAME
UNIT.** True by construction for `allocate.py` and for `br_rescale.py`'s 2022 município totals. False
for Switzerland, whose measured number is a CANTON total spread over communes; false for Israel's
lumped localities, split from district composition; false for China, which counted nobody. Those three
still empty, and that is the map saying so.

**It is 44 columns, not every adapter.** The scale was the surprise: across the eight `allocate.py`
countries and Brazil there are 44 allocated columns in total, so the semantic content of the fix is a
`COLUMNS = {…}` dict in each mapping module naming what its own fine columns mean. `countries.py`
attaches a `roll` per row from it, `rollup.py` turns that into a per-country `node -> target` table,
and `tiles.py` carries the table in `counts.json` beside `covers` — **no tile attribute, no buffer
attribute, and `--refresh-meta` ships a mapping fix without re-encoding 36,000 tiles.**

| | before | after |
|---|---|---|
| rolls up | 27.1M | **173.8M** |
| still gone | 185.9M | **39.3M** |

Measured 2026-09-07 on the 61 countries then built, and China was being rebuilt in another session as
this landed — its own line moves, the other sixty do not, because a country's table is its own. **The
figure that does not move is 8.63M**: Switzerland 7,438,908 + Israel 1,106,848 + New Zealand 65,151 +
Kosovo 16,369, everything still orphaned outside China. Check against that one rather than the total.

**And what is left is now ALL of it honest**, which is the state this section was trying to reach:
China 30.7M counted nobody's religion anywhere; Switzerland 7.4M has a CANTON total spread over
communes; Israel's lumped localities 1.1M come from district composition; Kosovo 16k is an estimate.
**Every one of those is "the source did not measure this at the unit it is drawn on", and none of them
is "the tree could not name what the source measured".**

The last cell of the second kind is New Zealand's `Māori Religions, Beliefs and Philosophies`, 65,151
people, and it stays deliberately — Anita, 2026-09-07. This tree has already split that cell in
writing: 59,656 are Rātana, Ringatū and Pai Mārire under `christianity.maori`, and `indigenous.maori`
defines ITSELF as "the remainder, 5,496". So `indigenous.maori` contradicts its own note and denies
59,656 people their Christianity; `christianity.maori` is 91.6% right and tells 5,496 followers of the
pre-Christian religion they are in a church; `other.nz` files Māori religion under "Other" on a map of
New Zealand. **The accurate-by-mass answer was available and was not taken**, because the roll-up's
promise is "this is the category your source counted you in" and Stats NZ counted these people as Māori
religions — a category this tree has no node for. That is a fact about the tree, and an absence states
it better than a near-miss papers over it.

Two calls that went the other way, both Anita's on the same day and both recorded in the module rather
than here. Mexico's `sin religión o sin adscripción religiosa` (13.3M) rolls to `unaffiliated`, folding
away the 3.10M `creyentes` that `unchurched` exists to hold — against 10.2M relabelled the other way,
which is three times worse, and against all 13.3M leaving the map, which is the error §7a-i fixes.
New Zealand's `Spiritualism and New Age Religions` (21,180) rolls to `other.nz`, which therefore now
means Stats NZ's residual groups generally rather than one named cell.

**THE LEGEND HAD TO MOVE WITH THE DOTS, and the old comment on the toggle said the opposite** — "this
changes no node's presence in the legend, only how many dots stand behind rows that are there either
way". True of a control that only removed things; false the moment it started moving them. Brazil's
47.4M evangelicals land on `christianity.evangelical`, which draws nothing at all in Brazil in the
default state, so the map would have painted a colour with no row to read it off. `rebuild()` now
takes the rolled tallies, which also gives Anita what she asked for: **a node that only ever holds
rolled dots is absent from the legend until you roll them.** Ireland's legend goes from 19 rows to 3 —
Catholic, No religion, Other religion — which is the CSO's Small Area table exactly.

**Two bugs found on the way, both invisible while nothing read the thing they broke.**

`allocate.py` had been writing `parent_column=` as a pandas Series repr since the `--within` change
(commit `b69fdd1`): the merge leaves two columns called `home` once `code` is renamed. It cost `in` and
`hu` their audit trail and nothing caught it, because until this section nothing read that field.
Fixed; those two files want regenerating.

And `SCATTER.setPalette` walked the tree **itself** rather than calling `rollUpMap`, despite §7a-i
saying in as many words that both renderers must implement the rule "from the same helper, because
`separate` and `merged` disagreeing about what the toggle means would be worse than either behaviour
alone". It went unnoticed while the two rules agreed. This section makes them disagree about 160
million people, and `separate` is the DEFAULT renderer — so the copy was the one a reader would have
seen. **A duplicated rule is a rule that will diverge; the comment forbidding it was not enough.**

### 7a-ii. The hit test and the two hover cards had not followed — FIXED 2026-09-07

The third and fourth copies of the same rule, found the way §7a-i's was: by using the map.

**`SCATTER.pick` was still on §7's original rule.** One line — `if (onlyMeasured && nid >= 0x4000)
continue` — from when the control REMOVED every inferred dot. §7a-i turned removal into a roll-up in
the vertex shader and left the hit test alone, so from §7a-i until now **every rolled dot was drawn and
none of them would answer a hover**: you could see 30,672 dots over China and 4,897 over Israel and ask
nothing about any of them. It now transcribes the shader — tier, then the palette slot the roll-up
sends that tier to, then the alpha there — and the transcription is the point: the shader is the
authority and this is a copy of it, not a second opinion. Verified on the running page: a Chinese dot
carrying `buddhism.vajrayana` and an Israeli one carrying `judaism.masorti` now hit, and answer
`buddhism` and `judaism`.

**And the cards named the node the dot carries, not the node it was drawn as.** A Haredi dot painted in
Judaism's colour whose card said "Haredi" asserts the very distinction the control just took off the
map, and its swatch — from `colorOf`, keyed on the name — did not match the mark it was pointing at.
Both hover paths and both double-clicks now read the rolled id. §4.2c already required the card and the
double-click to agree with each other; they also have to agree with the shader.

**The lesson §7a-i drew is now three for three.** A duplicated rule is a rule that will diverge, and
the rule had been copied into four places: `colorExpr`, `layerFilter`, `setPalette`, and `pick`. Three
of the four had drifted. What made them findable was that each one is a *different symptom* of the same
divergence — a colour, a missing dot, a dead hover — so none of them looked like the same bug.

### 7a-iii. A country that measured NOTHING draws nothing — DECIDED and BUILT 2026-09-07

Found while fixing §7a-ii. **Anita's call, and it is none of the three options this section first
weighed** — those are kept at the bottom, because the reason they were all wrong is the useful part.

**First, the thing that is NOT happening, because the numbers below read as if it were.** China has no
category of its own. Its Muslims are the node `islam`, the same node and the same green as Indonesia's,
and with `inferred dots: shown` — the default — the two are indistinguishable. What differs is the
**tier, which is stored per DOT and not per node**: of the 852,004 dots on `islam`, 805,567 are
measured and 23,895 are derived, and all 23,068 of China's are in the second group because nobody
counted Muslims in China — the number is inferred from ethnicity. The roll-up only ever touches the
derived ones, so Indonesia's 207,176 and Pakistan's 200,362 never move. **The bug is that the roll
target is looked up by node id alone, with no reference to the country the dot is in**, so one answer
serves all nine countries that have derived `islam` dots.

`rollTable()` with no country selected merges sixty-three tables and lets **the country with the most
dots on that node win**. §7a-i-1 accepted that as "the majority of the marks under a true statement and
the rest under a coarser one" — and that reasoning holds only while the winning target is *coarser*.
Where the winner's target is its own residual cell it is not coarser, it is **elsewhere**:

- `islam` is claimed by the UK, which counted Muslims in a cell called `Other religion` at Output Area
  level, so the world map takes **`islam → other.uk`**. China's 23,068 derived Muslim dots then draw
  in the colour of a British census residual. The UK wins on 3,999 dots, over New Zealand's
  `islam → islam`, because only six countries publish a roll entry for `islam` at all.
- The same shape elsewhere: `christianity.witnesses → other.br` reaching Mexico's dots (1,530),
  `christianity.other → other.br` reaching Canada's (582).

The deeper half is that **a country that measured nothing can borrow a target from one that did**. With
`inferred dots: not shown` and no country selected, China draws 30,672 dots — 23,068 Muslim, 7,604
Buddhist — where the honest answer, and the answer the map gives the moment you select China, is zero.
1,226,842 of China's 1,259,338 dots do correctly go.

**THE DECISION: don't roll them, drop them.** A country with no measured dot anywhere in it draws
nothing at all under `inferred dots: not shown`. It does not get a rolled colour, because there is no
level it was counted at — that is the whole of what the tier says — and the answer the map already
gives the moment you select that country is zero. This makes the world view agree with the country
view, which §7a-i requires of the two RENDERERS and had never been asked of the two VIEWS.

**Why this is cheap where the per-country roll-up was not.** The expensive thing was making the
*target* per country: a target is per node, so it needs a table per country, which is a texture in the
WebGL path and a sixty-three-way `match` over three hundred nodes in the tile path. But "did this
country measure anything at all" is **one bit per country**. So the gate is a country test — a filter
on the scatter's country loop, which was already per country, and one `['in', ['get','c'], …]` over a
list of eight codes in `layerFilter`. `SCATTER.rollMutes()` is the one helper all three call sites
share, which is §7a-ii's lesson applied on the way in rather than three sections later.

**Eight countries qualify, and for six of them it changes nothing:**

| | dots | measured | derived | modelled | drawn before | after |
|---|---|---|---|---|---|---|
| China | 1,259,338 | 0 | 1,257,514 | 1,824 | 30,672 | **0** |
| Russia | 136,900 | 0 | 0 | 136,900 | 0 | 0 |
| France | 67,235 | 0 | 0 | 67,235 | 0 | 0 |
| Italy | 58,911 | 0 | 0 | 58,911 | 0 | 0 |
| Spain | 48,964 | 0 | 0 | 48,964 | 0 | 0 |
| Kazakhstan | 19,182 | 0 | 0 | 19,182 | 0 | 0 |
| Greece | 10,447 | 0 | 0 | 10,447 | 0 | 0 |
| Switzerland | 7,431 | 0 | 7,431 | 0 | 7,417 | **0** |

Russia, France, Italy, Spain, Kazakhstan and Greece are **100% modelled**, and §7b already removed
every modelled dot outright — they were drawing nothing under this control before and after. The mute
only bites where a country is entirely *derived*, which is China and Switzerland. 38,089 dots leave;
3,253,203 are untouched.

**Switzerland is the one to look at again.** Its dots are all derived for a different reason from
China's: the 2000 census asked everybody, at commune level, and the archive rescales that composition
onto 2024 canton totals — so nobody was counted *as drawn*, but everybody was counted. China's
`basis` is "ethnicity, derived — nobody was asked about religion"; Switzerland's is
"self-identification; current magnitudes, 2000 composition". This section treats them the same because
the tier does, and **selecting Switzerland already blanked it** — its `measuredNodes` is empty and it
publishes no roll table, so every node was an orphan. The change makes the world view match. Whether
Switzerland's composition should count as measured — a roll table of its own, or tier 0 on the 2000
shape — is a question about `ch`'s adapter and not about this control.

---

The three options weighed before Anita's call, kept because each was worse in an instructive way:

1. **Rank the merge, don't just count it.** Prefer a target that is the node itself or an ancestor of
   it over a country-scoped residual; fall back to most-dots among equals. Fixes `islam → other.uk`
   (it becomes `islam → islam`). Does **not** fix China: its Muslim dots still draw, in Islam's green
   rather than a British grey. **It treats the symptom — a wrong colour — and not the claim, which is
   that anybody in Kashgar was counted.**
2. **Say so instead of fixing it.** Leave the rule and put the approximation in the control's tooltip
   and §7's about-panel line. **Documenting a map that says something false is not a fix**, and it
   would have had to explain why the world view and the country view disagree.
3. **Make the roll-up per-country.** A 2D palette (width 2N, one row per country, a `u_row` uniform in
   the render loop) makes `separate` exact for about 260 KB, but the merged path cannot follow without
   the paint expression §4.2d exists to avoid — so it buys an exact `separate` and a §7a-i violation.
   **The right rule turned out not to need it**: the question worth asking was one bit per country,
   not a target per node.

**Still open, and much smaller.** The sideways targets survive for countries that DID measure
something: `christianity.witnesses → other.br` reaching Mexico's 1,530 dots, `christianity.other →
other.br` reaching Canada's 582, and `islam → other.uk` still standing for New Zealand's 75, Ireland's
83 and Brazil's 47 — a few thousand dots against China's 23,068, and option 1 above is the fix if it is
worth one.

### 7b. The US residual is `modelled`, not `derived` — DECIDED 2026-09-05

Anita's call. **166,291 dots — half of every non-measured dot on the map and 6.7% of the whole thing —
move from `derived` to `modelled`.** The full argument on both sides is in `us_rebase.py`'s
`CONFIDENCE` docstring; the short form:

§7's tiers are not about which administrative level the coarse total sits at, they are about **whether
anybody was counted**. `derived` reads as "somebody counted this and the number was carried to a finer
place" — Ireland's coarse total is a census count, India's six religions are counted at sub-district,
and only the placement is inferred. Nothing in the American residual was counted at any level: a
survey of 36,908 people cut 51 ways, state margin of error 3 to 8 points, converted by the child
assumption, then spread over 3,143 counties by a proxy. `us_rebase.py` had already called it *"the
weak end of derived"*, and **a tier with an end that weak is two tiers.**

**What made this decidable was §7a existing.** While nothing on screen distinguished the two, the
classification was a comment in a docstring and the argument stayed abstract. With a control and a
split readout the answer is visible, and "is half the American map counted?" has an answer a reader
can see rather than one they must be told.

Requires a rebuild — `scatter_all.py --countries us`, then `buffers.py` and `tiles.py` for all
countries, because the manifest and the archive are whole-project files. About four minutes.

### 7c. The header says what kind of map this is, and how much of it we made up — BUILT 2026-09-07

Anita, looking at the France panel: *"i think they maybe contain some not super useful information
(for a random curious human exploring) and are omitting some things that should really be like the
main emphasis (these dots are kinda pulled out of our ass / modelled suspiciously)."*

**§7a's readout was in the right project and the wrong place.** It sits under the `inferred dots`
control at the bottom of the legend, which is correct for somebody who has already found that control
and useless for somebody meeting a country for the first time. What they met instead was a title, a
citation and a granularity line — **three descriptions of an instrument, none of which said whether
the instrument had ever touched the places on screen.** France read as a well-sourced map of French
religion. It is a 12,678-person survey at the *région*, and nothing on the left of the screen said so.

**Three labelled rows under the title, replacing the granularity line.**

| row | from | says |
|---|---|---|
| `data type` | `how`, new in `countries.py` | what kind of instrument, in the same words for all 68 |
| `data modelled` | live, off the buffers' tier bits | how much of the drawn dots NOBODY counted |
| `granularity` | `grain`, unchanged but for its label | the size of the count units |

**`data modelled` counts what nobody counted, and the first draft counted the other way.** It shipped
as `counted: 78%, we filled in the rest from coarser or older counts` — Anita: *"78% ... -> 22% filled
in from earlier counts"*. Naming the measured share made the reader subtract to reach the number the
row exists to give them. It now reads `22% filled in from broader counts`, `51%`, `0%`, `100%`. The
word `modelled` is in the label, so a country whose only inferred tier is `modelled` gives the bare
number and France reads **100%**; `derived` is a different claim from the label's and always names
itself. The same `tierWords()` builds the legend's terse caption, so the two can never drift apart.

**"broader counts", and it is the third wording.** `derived` is reached three ways — a coarser unit,
an earlier census, a wider category — and both shorter phrasings are false of real countries: "earlier"
of Canada and the UK, whose allocation is same-year and coarser-geography, and "coarser" of Brazil,
whose 2010 denominations sit inside 2022 totals at the same município. Broader is the one word true of
all three, and the tooltip spells them out.

**It is the one row allowed to leave the grey.** Past 50% modelled the value goes amber. Everywhere
else on this map a caveat is grey and waits to be looked for; "none of this was counted" is not a
caveat about the picture, it *is* the picture. The threshold is the claim and not a taste: the United
States clears it by a whisker at 51%, and that is the country the warning is most for.

**`how` exists because `source` cannot do this job.** `source` is the citation and has to stay one.
But *"Sčítání 2021 (Czech Statistical Office)"* and *"Sreda «Arena» Atlas of Religions 2012"* look
alike on the page and are a full census and a 56,900-person survey. Sixty-eight agencies in five
languages cannot be ranked by a reader at a glance. `how` says `census, 2021, voluntary question`
against `survey, 56,900 people, 2012; no census asks`, in one vocabulary, and those *can* be.

### The wording rules, and they are the reason the block was rewritten a day after it shipped

Anita: *"lets try to word this in a way thats not as much ai-smell."* The first draft was correct and
sounded machine-written, which on a map whose whole subject is whether to trust it is not a small
defect. Four rules, and they bind `how`, `grain` and everything `tierWords()` builds:

* **No em dashes.** They were doing the work of a comma, a semicolon and a bracket at once, and they
  are most of what the smell is. `countries.py` asserts on them.
* **No bold.** With `grain` bolding its whole string and `how` bolding none of its own, emphasis
  landed on whichever row happened to carry markup. Nothing in the block is bold now, so both fields
  are plain text and the viewer escapes them.
* **No article, no "a … question".** `a census question, 2021` became `census, 2021`. Sixty-eight rows
  in one grammar are read as a column, and the leading article is noise repeated sixty-eight times.
* **No "we".** The first draft wrote `we filled in the rest` to make sure the arithmetic was owned by
  the project rather than the source. The label does that already: a statistical office does not
  describe its own figures as modelled.

**And the block is drawn for a country only.** It carried the `data modelled` row alone in the
all-countries view for a day, which was one share averaged over sixty-eight different instruments,
under a title naming none of them. The subtitle there says it in words instead, and this is Anita's
sentence: *"data from official censuses where available, in many cases modelled, see details"* — where
the old line said the countries were "measured differently", which is true and is the smaller half of
it. Differently is a comparability problem; most of a reader's trouble with this map is that a lot of
it was never measured at all.

**Two things were dropped, both for the same reason: a line nobody reads costs the lines around it.**

* **`basis` left the subtitle.** It is spec §3.1's quantity and it matters, but sixty of the
  sixty-eight countries begin it with "self-identification" and the rest of the string is a qualifier,
  so it read as boilerplate, got skipped, and took the citation on the line above it with it. It is
  unchanged in the about panel, where a reader is already reading prose.
* **`grain` lost its `religion data granularity:` prefix.** The row's label says what the number is;
  the string said it again. It only ever carried the prefix because it stood alone with nothing else
  to identify it.

**Kazakhstan is the one country the trim changed rather than shortened.** Its granularity line said
*"modelled — the national religion table applied to 17 regions' ethnic composition"*, which is a
confidence statement wearing a granularity label, because that line was then the only one under the
title able to carry it. With `data modelled` above it saying 100%, `granularity` is free to answer its
own question: **17 regions, 1.1m people on average**.

**No rebuild.** `how` and `grain` are display fields (`tiles.py --refresh-meta`, about a second) and
`data modelled` is counted off buffers that already carry the tier in `ni`'s top two bits (§7a).

### 7d. What the filling-in was done FROM, and the about panel it points at — BUILT 2026-09-07

Three follow-ups to §7c, all of them Anita's, all the same afternoon.

#### `fill`: name the table, because there is one

*"'broader counts' feels weak and maybe in cases where we just used an earlier census we could say
that it was just another census? sounds more legitimate which it is."* It is. §7c had settled on one
phrase general enough to cover every `derived` country, and the cost of that generality was the whole
disclosure: **fifteen countries have derived rows and every one of them draws on a real published
table.** Brazil's is the 2010 census, Switzerland's the 2000 one, Canada's its own provincial columns,
Israel's the household-lifestyle table for the same statistical area. Describing all fifteen as
"broader counts" reads as evasion and understates the work.

So `fill` is a per-country phrase, and the row reads `41% filled in from the 2010 census`. The values
came from the `allocate.py` invocations in `COMMANDS.txt`, which name each country's coarse level, and
from the four adapters that never went through `allocate.py`. A country with derived rows and no
`fill` falls back to the old phrase rather than breaking.

#### `gap` moves into the block

§6.12's coverage note was put in the legend's scope bar on 2026-09-06 because that is beside the map,
*"where a reader is when the question occurs to them"*. §7c gave the project somewhere better: a block
where every line is the same kind of statement about the same source. **Who a census left out belongs
next to how much of it was counted, not next to the religion picker.** Fifteen countries have one, it
is labelled `not drawn`, and the strings were restyled to read as a thing that is missing rather than
as a standalone sentence.

Suriname's was never a gap. *"the 2012 and 2024 censuses publish religion nationally only, so this is
2004"* says why the DATA IS OLD, which is `how`'s job; under a `not drawn` label it would have been
the one row on the map naming nothing missing. Folded into `how` and deleted.

#### The about panel had been printing raw markdown since it was written

*"i havent looked at the see details link in a while."* Reading it as a reader rather than as its
author turned up the largest single defect in this map's presentation:

**531 bold runs, 80 italics and 43 code spans across the 69 country notes were on screen as literal
asterisks and backticks.** France alone showed 106 asterisks. The notes are authored in `countries.py`
in a markdown-ish voice, because that is what a docstring-shaped constant looks like, and nothing
between there and the panel ever converted the markers. `md()` now does, and `check_md.py` is the
check: apply its three regexes to every note and assert no `*` or backtick survives. It found three
notes that needed a tempered bold pattern — a bold run may contain an italic one — and none after.

**And the bolded topic sentences became paragraph breaks instead.** With the markers finally rendering,
the notes read as a listicle: a 900-word block of running text with a bold lead every eighty words.
That structure is real and worth keeping; saying it twice, in bold, is the register Anita means by
*"so people dont get ick"*. A bold run that is a whole sentence AND starts one now opens a paragraph
and loses its bold; everything else keeps it, which is the other half of why the notes use the marker
— `Catholicism is **38.0%** and falling` is emphasis on a figure and survives. **285 paragraphs, 279
of the 531 bold runs kept.** Sixteen of the shorter notes contain no topic sentence and stay as one
paragraph, which is what they always were.

**The all-countries half was still making §7c's superseded claim.** It opened *"The countries are not
measured the same way, and the border shows it"* and then listed all 69 `basis` strings — the exact
line §7c had just taken out of the subtitle for being boilerplate, printed 69 times. Meanwhile the
subtitle it is linked from now promises *"in many cases modelled"*. It leads with that instead, in a
figure counted live off the buffers rather than written down, names the eight countries with no
religion census, and replaces the basis list with **every country sorted by how much of it was worked
out rather than counted**. That list is the answer to what the link promises, and the old one was
sixty-nine variations on "self-identification".

**Still open: the em dashes inside the notes.** §7c's no-em-dash rule binds `how`, `fill`, `grain` and
`gap`, which are short authored fields with a checker behind them. The notes are 69 long essays and a
mechanical substitution there would produce comma splices, so they are untouched and the dashes are
still in the prose.

## 8. Pipeline

```
taxonomy/religions.json      the tree + genealogy + colour rules        (hand)
sources/<id>.py              one fetch+normalise per source, in one of two shapes:
   counts   (§1-3)           → data/normalized/<id>.csv
   columns: geo_id, geo_level, node_id, count, basis, year, source_id, note
   sites    (§4.4)           → data/sites/<id>.csv
   columns: lon, lat, node_id, kind, name, year, source_id, note
sources/<id>_geo.py          units + placement layer                    → data/geo/<id>/
countries.py                 per country: counts=, place=, place_weight=, note_public=, gap=
scatter.py                   units × placement weights → atomic dots, one per edition  §4.1a
buffers.py                   → data/buffers/<cc>.bin   the unmerged scatter             §4.2d
tiles.py                     → the merged pyramid + rings + counts.json, as PMTiles     §4.2a
country_shapes.py            → country_shapes.geojson  the coverage wash and Auto       §6.12
index.html                   MapLibre + PMTiles + the custom scatter layer + the panel
tools/                       one-off scans and checkers; never build stages
```

`COMMANDS.txt` has the runnable version and the order, including the three steps that fail silently.

**`reconcile.py` was proposed and does not exist.** Its job — resolve each unit to a partition, apply
§3.4's splits, tag confidence — is done inside the per-country adapters and `countries.py` instead.
The design intent still holds wherever that work happens: **it should refuse rather than guess.** A
unit whose figures do not sum to population within tolerance, or that mixes bases, goes to a findings
list, not into a fudge. §6.6's `unspecified` row is read out of the viewer's tallies for the same
reason — the residual is real arithmetic, it just has no file of its own.

**Geography.** geoBoundaries (CC BY) rather than GADM, whose licence is non-commercial and awkward.
ADM1 everywhere, ADM2 where a source supports it — the level varies by country and that is fine,
because dots are placed by population weight, not by unit area. §8.1 and §12's boundary section are
where the real work is.

### 8.1 Boundaries must be the vintage the data was *published* on — FOUND 2026-08-27

Not the newest available, and **not the year it was collected either.**

Taking the newest is the obvious default and it silently deletes places. But "use the collection year"
is also wrong: **Stats NZ recodes its 2013 and 2018 census addresses forward onto the 2023 SA2
boundaries**, so New Zealand's older columns want the *newer* geography — the exact mirror of
Connecticut. The rule that covers both is the vintage the table is published on, which the publisher
states and which no amount of reasoning from the data's date will recover.

**Connecticut abolished its counties** for statistical purposes in 2022, replacing them with nine
Councils of Governments planning regions with new FIPS codes (09110–09190). ASARB 2020 reports eight
old counties (09001–09015). Joined against the 2024 cartographic boundaries, **every one of them fails
to match**: the whole state — 3,605,944 people and 1,707,793 adherents — has no polygon and drops out
with no error anywhere. Against the 2020 boundaries, all 3,143 ASARB counties match exactly.

**The failure mode is what makes this worth a rule rather than a fix.** A join that silently drops
Connecticut looks identical to a working join everywhere else; nothing is malformed, no count is
wrong, a state is just missing. **So the join is checked in both directions and both sides are
reported**, always — unmatched data rows *and* unmatched polygons.

**The cost, measured on Australia:** using the 2016 boundaries against 2021 data matches **87.7% of
codes** and silently drops **303 SA2s — 3,866,694 people, 15.2% of the country**. An 87.7% match rate
is exactly the kind of number that looks like success in a log.

**And there is a third direction, which two-way matching does not catch: a code can match and still
have no geometry.** Australia has 18 special-purpose SA2s — migratory, offshore, no usable address —
whose codes join perfectly and whose polygons are empty; they hold 52,920 people who would be
scattered nowhere at all. Worse, their `AREASQKM21` is **NaN rather than 0**, so the obvious guard
(`area == 0`) matches nothing. **So the check is three-way: unmatched data, unmatched polygons, and
matched-but-empty geometry.**

**Canada is the counter-example that shows the good design:** StatCan's DGUID embeds its own vintage
(`2021A0005…`), so joining 2021 data to a wrong-vintage file yields **zero** matches rather than a
plausible 88%. **A loud total failure is a far better property than a quiet partial one**, and it is
worth preferring a vintage-stamped key wherever a source offers one.

**A fourth way to get the vintage wrong: the file format renames the column for you.** Ireland's Small
Area shapefile carries both the 2022 and 2016 keys, and **the DBF format truncates field names to ten
characters**, so `SA_GUID_2022` arrives as `SA_GUID__1` while `SA_GUID_20` — the name that looks like
the 2022 key — is in fact **the 2016 one**. Joining on the obvious-looking column silently gives you
the previous census's geography, on which 1,448 of 18,919 codes have changed. **Confirm the key by
joining, not by reading its name.** A correct key matches 100% and the wrong one does not, which is
the only reliable signal available.

**The check also confirms what should be absent.** 91 US polygons have no ASARB row and all 91 are
Puerto Rico, American Samoa, Guam, the Northern Marianas and the US Virgin Islands, which ASARB does
not cover. **An expected absence and an accidental one look the same until you name the expected
ones.**

**And the global boundary source has the same disease — including the one §8 recommends.**
geoBoundaries' Mexico ADM2 is **2012 vintage, 2,457 units, against the 2020 census's 2,469**; three
Morelos municipios (`17034`–`17036`) simply do not exist in it, so the Connecticut failure is waiting
there in the same silent form. **Its vintage is a per-country fact to check, not a property of the
dataset**, and where a country publishes its own census geography, that is what the join should use.
Kept as an explicit pre-flight: for every country, compare unit counts and report both directions
before scattering a single dot. (INEGI's own Marco Geoestadístico 2020 has all 2,469 and joins 0/0.)

**AND CHINA IS THE SAME DISEASE AT A SCALE THAT DISQUALIFIES THE FILE — FOUND 2026-09-05.** Mexico's
ADM2 is merely a stale vintage. `CHN ADM2` is not a vintage at all: 2,391 units against the 2000
census's 2,859, **duplicated polygons** (`Huinongxian` twice in Ningxia, `Banmaxian` and `Geermushi`
twice in Qinghai), **units under the wrong province** (Gansu's `Maquxian` under Qinghai), **counties
abolished in the 1980s** (Xizang gets 78 polygons for 73 counties), and **romanisation corrupted in a
patterned way** (`Erminxian` for Emin, `Duinongdeqingxian` for Duilongdeqing). It matched **59.9%** of
census counties against DataV GeoAtlas's 94.1%. Its `boundaryYearRepresented` says 2017, and
`Tongxian` in it was renamed in 1997.

**So the transferable rule is stronger than "check the vintage": treat `boundaryYearRepresented` as a
claim rather than a fact, and check the unit count against the census before anything else.** A file
can be internally inconsistent — several decades of administrative geography mixed together — in a way
no single vintage would explain, and no amount of joining will reveal it if the join is only ever
measured in one direction.

This will recur everywhere and worse: municipal mergers in Japan, Brazil and Indonesia run
continuously, and **a census's own geography is the only safe join target for that census.**

### 8.2 Placement needs no population data — DECIDED 2026-08-27

The project does not depend on a population layer, and the reasoning is worth keeping because the
obvious version of this decision is wrong in both directions.

**Two jobs get confused.** Population is used for (a) the residual — how many people belong to nothing
— and (b) placement, deciding where inside a unit the dots go. They are unrelated. (a) needs one
number per unit; (b) needs relative weight *within* the unit.

**(a) is free wherever the religion source is a census**, because a census reports population too.
ASARB ships a `2020 Population` column per county in its own summaries workbook.

**(b) is free in the United States too, and this is the useful trick.** Census tracts are *designed* to
hold about 4,000 people. So allocating a county's dots **equally across its tracts** is already a
population weighting — the geometry carries the weight, and no population figure is read at all.
Measured, to check the design is adhered to rather than merely intended: 85,187 tracts across 3,143
counties; people per tract (county means) median **3,424**, IQR **2,818 – 4,043**; log-log correlation
of county population against tract count **r = 0.98**.

**What this is not.** That measurement is *between* counties; the error that matters is *within* one,
and it cannot be measured without the tract populations we are declining to fetch. The honest bound is
the design range itself — tracts run roughly 1,200–8,000, so two tracts in the same county can differ
by about 3×. Against the alternative it is nothing: uniform-random over a *county* polygon is wrong by
two orders of magnitude in the western US, where it would scatter San Bernardino's adherents across
20,000 square miles of Mojave.

**It generalises, and Australia is a better case than the US.** ABS Statistical Area 1s hold a median
of **406 people, IQR 359–447**, against the US tract median of 3,424 with an IQR nearly twice as wide
in relative terms; the correlation between SA2 population and SA1 count is **r = 0.923**, and every
real SA2 has at least one usable SA1. Canada's dissemination areas are the same idea. **The trick is
not a US accident — it is what happens wherever a statistical agency designs its smallest unit to a
population target**, which is nearly everywhere. (§8.2a is where it stops being true.)

**Where the placement layer ships its own population, use it rather than approximating.** New
Zealand's SA1 file carries a 2023 population per polygon (median 150, IQR 120–183). The approximation
is a fallback for the common case where no population travels with the geometry, not a preference —
and §8.4 dropped it for the US as soon as the ACS tract totals were in hand for another reason.

**Placement is by population weight, never uniformly in the polygon.** ancestrydots'
`random_points_in_polygon` is fine for US census tracts; at ADM1 scale globally it would fill the
Sahara, the Amazon and Siberia with people. Kontur Population (400m/800m H3, HDX, already vector) is
the fallback where no engineered fine unit exists; §12's boundary section has the mechanics and the
resolution trap.

**The consequence, stated plainly:** without a population layer the map shows **counts, not shares**.
A county that is 48% adherent and one that is 90% adherent look the same. That is a real limitation,
it is reversible, and for a first build it was the right trade. (`fetch_tract_pop.py` is written and
kept for the day exact weights are wanted; it needs a free `CENSUS_API_KEY`.)

**Placement is not a claim about which people are where.** Within a unit, a group's dots are scattered
in proportion to total population. Where a source gives a finer unit we get finer truth; where it does
not, the dots say "this many people of this group live somewhere in this unit". The about panel has to
say so, because a dot map invites exactly the opposite reading.

### 8.2a India is the first country the trick does not work on — FOUND 2026-09-03

**Answered by §8.2f on 2026-09-07.** What follows is right about India's geography and wrong
about what follows from it, and is kept because the diagnosis is why the fix has the shape it
does. From "So India places on its count layer" down, it no longer describes the map.

§8.2's whole argument is that **a statistical agency designs its fine unit to a population target**.
Six countries in, it looked like a property of censuses. It is a property of *statistical*
geographies, and India has none. Below the sub-district India has **645,828 villages and 4,135 towns**
— administrative and natural settlements ranging from ten people to two million. An equal share per
village would weight a hamlet like a small city, and India has a very great many hamlets. There is no
engineered layer anywhere between the sub-district and the settlement.

So India places on its count layer, and pays for it:

| | India | Brazil (before §8.2d) | Poland | US |
|---|---|---|---|---|
| median unit population | **~204,000** | ~38,000 | ~7,500 | ~4,000 |
| median unit area | **551 km²** | 1,527 km² | 126 km² | — |

**The population figure is the coarsest count unit on the map; the area figure is finer than a
Brazilian município's.** So the damage is zoom-dependent: at national and state zoom India's grain is
comparable to a country already drawn, and it is at city zoom that India looks blockier than anywhere
else.

**The fix is specific and the machinery for it now exists.** SHRUG publishes 645,828 village POINTS
with `t_pop2011`, summing to 828,886,066 — India's entire rural population — and the towns file
carries the urban half. `scatter.py`'s `place_weight` hook is exactly the right shape, so what is left
is not a capability but two pieces of wiring: a placement layer of villages and towns keyed to their
sub-district, and a weighter returning each settlement's 2011 population. **India does not use
`place_weight` today only because its placement layer is its count layer** — one polygon per unit, so
there is nothing inside a unit to weight. That is the single biggest available improvement to how
India looks.

**Brazil got there first (§8.2d)**, and one lesson from it applies directly: prefer a placement layer
whose key NESTS in the count layer's over one that has to be joined geometrically — which is what
SHRUG's village points already do via their sub-district.

### 8.2b Germany is the first country that needs no trick at all — BUILT 2026-09-04

§8.2 and §8.4 are both answers to the same missing thing: **nobody publishes where a religion sits
inside a unit**, so the map either assumes an equal share over units engineered to a population target,
or fits a model to guess it. Germany publishes it.

destatis puts **the same three categories on the 1km INSPIRE grid** it puts in the Gemeinde table. So
the weight for `christianity.catholic` inside Munich is Munich's own per-cell Catholic count, and the
placement stops being an approximation:

| | |
|---|---|
| placement layer | **209,154** 1km cells, replacing 10,786 Gemeinde polygons |
| Berlin | **799 cells**, against one polygon holding 3,596,999 people |
| rows placed on measured weights | **17,215 of 17,215** — no fallback used |

It had to be done, because §8.2's trick fails completely here: Gemeinden are historical units, not
units built to a population target, and they run from 9 people to 3.6 million. 78 of them hold 31.6%
of the country, and inside those the old placement said only "somewhere in this city" — Neukölln and
Zehlendorf came out identical, and dots landed in the Grunewald and the Müggelsee.

**Nothing here is fitted, and that is the point.** §8.4's US model has parameters, a residual model and
a §7 confidence mark; this has none, because it is a count. §14.4's rule — never estimate a magnitude a
source does not publish, refine placement only — is satisfied in the strongest possible way: **the
refinement is itself published.**

**Why 1km and not the 100m file**, which also exists at 3,088,036 cells: Germany draws 82,710 dots, so
100m would be 37 cells per dot. **The dot value binds before the grid does. A finer placement layer
than the dots it carries is bytes, not information.**

The general shape, for the next country that has one: *grid cell → containing admin unit → clip to it
→ per-node weight column*, counts still from the admin table. What does not generalise is the good
part — most grids carry population only, which is a better proxy but still a proxy. Germany is unusual
in publishing the **same variable** on the grid as in the table (`sources/de_grid.md`).

### 8.2c Administrative units own water, and dots were landing in it — FIXED 2026-09-05

A shoreline census tract reaches into the middle of the river, because that is where the boundary
legally is. Sampling uniformly inside it puts dots on the water, and those are the most conspicuous
dots on the map: **370 of the 12,399 US dots in the New York bbox, 3.0%, were in open water**, with
San Francisco Bay at 1.4% and Puget Sound at 0.4%. Nothing was miscounted — every dot was in its
correct unit — so **no reconciliation could ever have caught it.** It is a placement error, and this
section is the one that licenses placement to use information the count does not.

**Subtract the water from the polygon; do not reject the dot.** Rejection sampling pays a
point-in-polygon test per dot forever, and point-in-polygon is already the hot path of a scatter
([[reference_dotmap_hotspots]]). Clipping the placement layer is paid once per build, is cached, and
leaves sampling *faster* than before because every candidate point now lands. `water.py` does it with
OSM's `water-polygons-split-4326`, already in the repo for other maps. After: New York 9 dots, 0.1%.
Dot counts unchanged, by construction.

Three things worth keeping:

- **OSM's coastline layer is the right one and not merely the available one.** `natural=coastline` is
  carried up an estuary to the tidal limit, so the Hudson, the East River and the Thames are all in it
  — exactly the set of cases that puts a dot somewhere a reader can see is wrong.
- **Clip the ocean polygon to the unit's bounding box before differencing it.** An Atlantic polygon is
  millions of vertices and a tract is a few dozen; `clip_by_rect` first halves the whole job (US: 211s
  → 107s) for bit-identical output.
- **A unit that is entirely water keeps its original shape.** Two US tracts and seventeen Philippine
  barangays are all water and have people in them. §4.1 says the dot count may not move, so those are
  left alone and reported rather than clipped to nothing.

**Inland water is still not handled globally.** Lakes and non-tidal rivers are a separate OSM layer.
Mostly the agency has already done it — Lake Lanao is a hole in the Philippine barangays and the Great
Lakes are absent from the US tract file — but where it has not, the lake still takes dots. Ghana's Lake
Volta is the case that needed a local fix; §12's boundary section has it and the rule about when the
code should move into `water.py`.

#### 8.2c-i SOME PEOPLE LIVE ON THE WATER — raised by Anita 2026-09-05

The rule "no dot on water" assumes nobody lives there, and in the Sulu Archipelago that is false. Ten
populated Philippine barangays lost more than 90% of their area to the clip, and they split into two
kinds the polygons cannot tell apart:

| barangay | people | lost | and it is |
|---|---|---|---|
| Port Holland Zone III (Samal Village), **Basilan** | 4,904 | 98.0% | a Sama stilt village |
| Tungbangkaw, **Tawi-Tawi** | 3,274 | 90.6% | over-water settlement |
| Tonggasang · Sisangat · South Silumpak · Sibaud, **Sulu** | 4,992 | 91–99.6% | over-water settlements |
| Nasingin · Batasan · Ubay Island, **Bohol** | 3,209 | 92–97.7% | real coral islets |

**For Bohol the clip is right** — it moves dots off the surrounding sea onto the islet, which is where
the houses are. **For the Sulu Archipelago it is wrong**: those villages stand on stilts over water,
OSM has no land under them, so the clip crams ~14,000 Sama-Bajau onto whatever shore sliver survives.
There is no signal in a boundary file that separates the two cases.

It is 17,474 people, 0.016% of the country, 17 dots at 1:1,000 — small enough that the current
behaviour is defensible and **not** small enough to leave unnamed, because the people it displaces are
a sea-dwelling minority in the province this map already resolves worst. §3.7 is the same shape of
problem: **the instrument cannot see a way of living that does not match its assumptions.**

**DECIDED 2026-09-05 (Anita): `KEEP_WHOLE_ABOVE = 0.95`.** A unit that loses more than 95% of its area
to the sea is left unclipped, on the argument that a 2% sliver is not a plausible home for 4,904
people and the unclipped polygon at least puts them over the village. It rescues six of the ten —
and, as the price, un-clips Nasingin and Ubay Island in Bohol, which were correctly clipped before.

**The reasoning behind the number matters more than the number: no single threshold is right
everywhere.** The honest value is near 1.0 in the United States and Europe, where an all-but-water unit
really is a measurement artefact, and much lower across maritime Southeast Asia, where it is a village.
0.95 is a compromise chosen in full knowledge that it is one, and worth revisiting per-region the day a
second country turns out to have Sulu's problem.

**It is applied when the clipped layer is READ, not when it is computed.** The cache holds the raw
clip, so moving this threshold re-clips nothing and costs seconds rather than the twenty minutes a full
re-clip takes. **A number nobody can derive should be cheap to change.**

### 8.2d Brazil is the second customer for `place_weight` — BUILT 2026-09-05

§8.2a predicted the fix for Brazil would be wiring rather than capability. It was.
`sources/br_setores.py` builds a placement layer of **466,996 census setores** weighted by setor
population, and `countries.py` gains `_BrSetorWeighter`. Nothing in `scatter.py` changed.

**The case for it in one number: São Paulo is one polygon holding 11.5M people.** Its ~11,500 dots were
spread uniformly over the Serra da Cantareira, the Billings and Guarapiranga reservoirs and Avenida
Paulista alike. Rio, Manaus, Brasília and Belém the same. It is the most conspicuous thing wrong with a
country at city zoom and completely invisible at national zoom, which is why it survived four
countries' worth of review.

**Setores beat the 1 km grid for one reason worth generalising: `CD_SETOR[:7]` IS `CD_MUN`.** The fine
unit's code *contains* the coarse unit's, so the assignment is a string slice — no spatial join, no
cells straddling a boundary, no clip, no slivers. **When choosing a placement layer, prefer one whose
key nests in the count layer's over one that has to be joined geometrically**, even at four times the
download.

**And the US argument does NOT transfer, which was the near-miss.** §8.2 says an equal share per tract
is already a population weighting because tracts are built to ~4,000 people. The tempting move was to
apply that to setores and skip the population file. Setores are built to ~300 households in cities and
*fewer* in the country, with rural ones covering enormous areas, so equal shares would have pulled
Brazil's dots systematically into the countryside — a smaller version of the error being fixed. **A
design target is only a weighting if it is the same target everywhere in the country.** The real
`v0001` population is joined instead; it totals 203,080,756, which is the check that the join is
complete.

Result: 176,291 dots before and after — §4.1's invariant, **a placement change may not move a count** —
now across 150,088 polygons instead of 5,565, with **zero** fallbacks to equal shares.

**One data trap, recorded because it is a shape rather than a Brazilian quirk.** IBGE delivers 914
setores as *several rows each*, one per disjoint part — river islands, mostly. Population is keyed on
the setor CODE, so joining it onto the parts as delivered gives a five-part setor five times its
population and five times its pull. **Whenever a weight is joined by key onto a geometry layer, check
the key is unique in that layer first.** A plain duplicate-key assertion caught it; the parts are
dissolved and the layer then has exactly the 468,099 distinct codes the population file has.

### 8.2e A population grid has a resolution floor — FOUND 2026-09-07 with Saint Vincent

Every country drawn since Kenya has had its dots placed on **Kontur's 400 m H3 grid**, and it
had become the reflex: a `place_weight` hook, a per-country extract, a ratio check. Saint
Vincent is the first country where it was **built, measured and thrown away**, and the reason
is not about Saint Vincent.

**A Kontur r8 hex is about 0.16 km². Saint Vincent's counting tier is the enumeration
district, whose median area is 0.66 km².** So the grid is roughly four cells across a typical
unit, and 43 of the 221 units are smaller than a single hex. What that produces, measured:

| | |
|---|---|
| Kontur hexes over the whole country | **509** (417 after assignment) |
| populated units getting **no** hex at all | **78 of 219 — 36%** |
| per-unit Kontur / census ratio | **p10 0.00, median 0.45, p90 2.68** |

**A weighting that is absent for a third of the units and scatters over an order of magnitude
for the rest is not a weighting, it is noise.** Using it would apply two different placement
rules essentially at random across one island — Kontur where a hex happened to land,
equal-shares where it did not — with no reason to believe the first is better.

**The rule: a population grid must be finer than the counting tier to be worth anything, and
Kontur r8 stops paying at roughly 1 km² per unit.** Below that, §8.2's uniform-within-unit is
not a fallback but the better answer — a unit a few hundred metres across does not need its
interior modelled, because at any zoom this map reaches, uniform inside it is indistinguishable
from correct.

**Why the floor had never been hit before.** Every earlier customer was far above it — Kenya's
47 counties, Ethiopia's 738 woredas, Bosnia's municipalities at a median 304 km², Kosovo's at
287. The grid was always orders of magnitude finer than the thing it was refining. §8.2a records the
opposite failure (India, where the units were too coarse for uniform to be honest); this is the
same trade seen from the other end, and the two together bracket where a grid belongs.

**The practical instruction, before writing a `place_weight`:** divide the median unit area by
0.16 km². If the answer is single digits, or if a large share of units would come back with no
hex, do not build the grid — and say in `sources/<cc>_geo.md` that it was measured rather than
skipped, because the next reader will otherwise assume it was an oversight.

**And name what uniform costs.** Saint Vincent's unit areas are skewed by a factor of 6,000
(0.007 km² to 44 km²), and the largest are the uninhabited Soufrière massif, so uniform scatter
puts dots on a volcano. The bound is what makes it acceptable: at 1:1,000 a unit of a thousand
people draws one dot, so this is one or two dots up a mountain rather than a misread
distribution. State the bound; do not leave it to be discovered.

### 8.2f India after all, and §8.2a's mistake was a hidden premise — BUILT 2026-09-07

§8.2a is the section that said the trick fails in India, and it was right: there is no layer
between the sub-district and the settlement built to a population target, and India's 645,828
villages run from ten people to two million. What it then concluded — that India therefore
places on its count layer — does not follow, and the gap between the two is a premise §8.2
never states out loud.

**§8.2's method is not "use a fine layer". It is "give each polygon an equal share".** The
equal share is the part that needs units engineered to a target, and it is the part India
breaks. The fine layer itself was never the problem. **Weight each settlement by its own
population and the objection disappears entirely** — a hamlet is weighted like a hamlet
because the weight is its population, and the fact that settlements vary by five orders of
magnitude stops being a defect and becomes the signal.

That the machinery was already there made this cheap: `place_weight` (§8.4, §8.2d) took the
layer unchanged and scatter.py was not touched at all. **All of the work was in the joins,
and none of it in the placement.**

| | before | after |
|---|---|---|
| placement polygons | 5,988 sub-districts | **544,615** villages and towns, + 3,103 unit outlines |
| per unit | 1 | **91** |
| weight on a real settlement | — | **94.3%** (64.3% village, 30.0% town) |
| weight on the sub-district fallback | 100% | **5.7%** |

**Where the population comes from, and the trap that would have been invisible.** SHRUG
publishes village points with `t_pop2011` summing to 828,886,066, India's whole rural
population; C-01 supplies the urban half from its own 8,067 town rows. But **3,892 six-digit
codes name both a village and a town**, so joining population on the code alone hands 3,892
town polygons a village's population — and nothing downstream could ever see it, because
every dot count stays exactly right. What is unique is (unit, code). This is §8.1's lesson in
its sharpest form: *a key that matches is not a key that is right.*

**The `(Pt)` trap from the other direction.** The points file's own unit codes split units
into parts, so unit+code matches only 82.4% of the polygons while the village code alone
reaches 99.92%. So the population joins on the code and the UNIT comes from SHRUG's polygon
file — the same file the count layer is built from, which is what §8.2d means by preferring a
key that nests over one that has to be reconciled.

**EVERY UNIT'S WEIGHTS SUM TO ITS CENSUS TOTAL, AND THAT IS THE DESIGN AND NOT AN OUTCOME.**
Each unit carries one extra placement polygon — its own outline — holding whatever its
settlements do not account for. 3,103 of the 5,988 use it for some part of their population.
The alternative considered and rejected was spreading a unit's shortfall over whichever of
its settlements lacked a population row, which asserts the missing people live in those
particular villages; they generally do not, since a shortfall is mostly people whose
settlement has no polygon at all. **The outline claims only "somewhere in this sub-district",
which is exactly what §8.2a's map claimed about all of them.** So the floor is the old
behaviour, per unit and in proportion, and no part of India is drawn worse than it was.

**Assam is where that floor earns its keep.** Its 26,599 villages carry 353 non-zero
populations between them and sum to 449,486 people against a rural Assam of about 26.8
million. The names match exactly — Mankachar, Kuchnimara, Jhawdanga Pt.III — so it is a hole
in the file, not a failed join, and no key work will fill it. Assam's towns place properly
from C-01 and its villages fall back. **A zero therefore cannot be read as "nobody lives
here" without asking what else the unit knows**, and the rule is per unit: villages are taken
at their word where the unit has any village figures at all, and treated as unknown where it
has none. India has genuinely uninhabited revenue villages and they must stay empty.

**One failure mode worth naming, because it was silent.** 464 settlements had no population
row and carried NaN rather than zero. The weighter sums a unit's weight column, one NaN makes
the sum NaN, `NaN > 0` is false — and the unit falls back to *an equal share per settlement*,
which is precisely the error §8.2a exists to describe, in the one country the section is
about. It was visible only as a line reading "3 on equal shares" in a Kerala test run. **A
fallback that triggers on a condition you did not intend is worse than no fallback**, so
in_place.py now refuses to write a layer containing a NaN weight.

### 8.3 Placing dots by church location — TRIED AND REJECTED 2026-09-03

**The idea.** US religion data is county-level and cannot be finer (PL 94-521 bars the Census Bureau
from asking, so ASARB is the only national enumeration and it is compiled by county). A county is
~105,000 people — fifteen times coarser than a Canadian CSD and sixty times a Czech obec — so every
tract in Cook County gets an identical religious mix and Chicago renders as a uniform blend with no
neighbourhood structure at all. The proposal: keep ASARB's county total as the magnitude, but use
**church locations** to decide where inside the county the dots go.

**Two of the three things needed turned out to be fine.**

1. **OSM coverage is good enough, for Catholics.** Measured against ASARB's own congregation counts
   across eight counties chosen to break it, urban through the Navajo Nation:

   | | min | max | median | spread |
   |---|---|---|---|---|
   | all congregations | 41% | 104% | 64% | 2.5x |
   | of those, carrying a `denomination` tag | 8% | 66% | 43% | **8.0x** |
   | **Catholic only** (7 of 8 counties) | 77% | 104% | **100%** | **1.3x** |

   So §4.4's mapping-effort worry is **confirmed for the general case and wrong for Catholics**. In
   McKenzie County ND only 8% of mapped churches say what they are; weighting by that would draw
   tagging habits as religious geography. But Catholic churches are large, named, landmark buildings,
   and OSM has essentially all of them.

2. **The obvious model produces impossible numbers, and a kernel fixes it cleanly.** Giving each parish
   an equal share of the county total and assigning tracts to their nearest parish (Voronoi ≈ the
   territorial boundaries canon law implies) put **13% of Cook County's Catholic dots in tracts implied
   over 100% Catholic** — a one-tract catchment receiving 4,287 people into ~3,400 residents. Replacing
   Voronoi with a Gaussian kernel and sweeping the radius gives a clean
   window at σ = 2–3 km, and σ = 2,642 m — the mean parish spacing — is a **parameter-free** choice from
   the data that lands in the middle of it. Tidy.

**The third thing killed it: parish density does not measure where Catholics live.**

Checked against Chicago's actual, well-established religious geography — the only validation available,
since any dataset good enough to confirm the model would be good enough to use *instead* of it:

| neighbourhood | reality | parishes ≤3 km | model |
|---|---|---|---|
| Mount Greenwood | Irish-Catholic heartland | 8 | **29%** |
| Garfield Ridge | Polish/Irish Catholic | 8 | 32% |
| Englewood | Black Protestant since the 1960s | **7** | **39%** |
| Washington Park | Black Protestant | 8 | 33% |
| Little Village | Mexican Catholic | 19 | 78% |

Parish density is **the same** in Chicago's most Catholic neighbourhood and its least, and the model
therefore rates Englewood *more* Catholic than Mount Greenwood. Strip out Little Village and the HIGH
group averages 37% against LOW's 34% — no discrimination at all.

**The mechanism, which is why no amount of tuning saves it.** South Side parishes were built 1890–1930
for Irish, German and Polish immigrants. The Great Migration turned those neighbourhoods over; the
buildings stayed. So parish density there is a **fossil of 1920s settlement**, not a measure of 2020
Catholics. The model's load-bearing assumption — that parishes hold roughly equal numbers — fails in a
*spatially structured* way: inner-city parishes are remnants, outer ones are full. Little Village
scores correctly only by accident. Every older US city has the same fossil pattern, and it fails
hardest in precisely the neighbourhoods most worth resolving.

**What would actually be needed:** parish-level registered households or mass attendance, which some
dioceses hold and none publish uniformly. A real per-diocese data hunt, not a modelling trick.

**The lesson worth keeping.** Every internal check passed. Coverage was excellent, the σ sweep was
clean, the parameter chose itself, the median share matched the county truth. It looked finished. Only
comparison against known ground truth revealed that the signal was a century out of date — and had it
shipped, "Englewood is Catholic" would have read as a discovery rather than an artefact. **A model
built from a proxy needs an external check against something known, or it should not ship; internal
consistency cannot detect a decorrelated input.**

### 8.4 Placing dots by demographic composition — BUILT 2026-09-03, and it passes §8.3's test

The second attempt at the same problem, and the one that ships. `us_weights.py`.

**There is no sub-county data to get, and that was checked first.** The US Religion Census publishes at
county and **does not collect congregation addresses** — its own "Data Collected" page says so. Two
things do exist and neither solves this:

| | |
|---|---|
| **Per-congregation membership** | Real and public — UMC (`umdata.org`), ELCA, PCUSA, Episcopal parochial reports, UCC all publish membership per church with an address, and it is not a fossil: a dying parish reports 150 members. But it covers mainline Protestants, ~12% of adherents, and *the wrong 12%* for the three counties that need it most, where the mass is Catholic, Black Protestant, Hispanic, Jewish and Muslim. Catholic per-parish figures are diocesan and unpublished. |
| **Jewish community studies** | The one genuinely measured sub-county source in the country. UJA-Federation's **2023 Jewish Community Study of New York** gives Jewish population by sub-county ZIP cluster, by denomination, for the eight New York counties. JUF Chicago 2020 gives about ten metro regions, much coarser. Brandeis' AJPP models on ZIP clusters internally but publishes only county and up, and forbids scraping. **Not yet used.** |

So the magnitude stays ASARB's and only the placement is modelled. The proxy this time is
**demographic composition** — ancestry, birthplace and race at tract level, which ancestrydots already
holds for every state, so no new download and no API key.

**The check is the point.** §8.3's lesson was that a proxy model needs external ground truth, and there
is some: **ASARB's own county numbers inside a metro**. Fit on counties, hold out WHOLE METROS, and see
whether the model predicts variation it never saw. 448 counties in 48 metros, scored on
population-weighted correlation of the within-metro deviation — correlation rather than R², because the
allocation is raked to the ASARB county total, so the level is fixed for free and only the relative
pattern has to be right.

| | held-out r | | held-out r |
|---|---:|---|---:|
| Church of God in Christ | **0.61** | Jehovah's Witnesses | 0.21 |
| National Baptist Convention USA | 0.57 | Hindu Temples | 0.20 |
| Seventh-day Adventist | 0.57 | Lutheran — Missouri Synod | 0.12 |
| National Missionary Baptist | 0.56 | United Church of Christ | 0.09 |
| Reform Judaism | 0.46 | Assemblies of God | 0.07 |
| **Catholic Church** | **0.45** | **Orthodox Judaism** | **0.05** |
| Episcopal, AME, American Baptist | 0.42–0.45 | | |

Calibration slopes run 0.7–0.96, so the predicted spread is about the right size rather than a muted
smear. **26 fitted nodes clear r ≥ 0.25**; everything else stays population-uniform. **That gate is the
whole design — it is a per-node confidence claim, which is what §7 wants anyway, rather than one switch
for the country.**

**Against §8.3's own validation set**, same five Chicago neighbourhoods:

| | reality | parish model (§8.3) | this |
|---|---|---:|---:|
| Mount Greenwood | Irish-Catholic heartland | 29% | **78%** |
| Garfield Ridge | Polish/Irish Catholic | 32% | 64% |
| Little Village | Mexican Catholic | 78% | 54% |
| Englewood | Black Protestant since the 1960s | 39% | **14%** |
| Washington Park | Black Protestant | 33% | 17% |
| | **HIGH vs LOW mean** | **37% vs 34%** | **66% vs 15%** |

Cook County is 53% Catholic and the old placement drew that everywhere. Englewood also goes from 10%
to 34% Black Protestant. Glendale draws Armenian Apostolic at 15× the LA County rate; Richmond Hill
draws Hindu at 2.7×.

#### Two things it got wrong before it was right, both worth keeping

**1. The fitter destroyed the signal and nearly got the method thrown out.** The first three
specifications returned NEGATIVE R² on every family, which reads as a clean kill — an alternating
per-metro scale step was absorbing exactly the variation it was meant to explain. What caught it was
correlating each demographic segment against each body **directly, with no model at all**: National
Baptist against `black_resid` is r = 0.60 raw. The signal had been there the whole time. **When a
model says there is no signal, check for the signal without the model — a null from a fitted model is
a statement about the fitter first.**

**2. A node can clear the gate on a coefficient that means nothing.** Armenian Apostolic scored r =
0.35 and passed — with **no `armenian` coefficient at all**, its largest term `afro_carib`, and
`east_asian` positive. It drew Armenian dots in San Gabriel and none in Glendale. The cause is that one
ridge penalty against columns whose scale differs by three orders of magnitude shrinks the small ones
far harder, and `armenian` is 0.2% of the population. Swept both ways:

| | nodes | adherents kept | mean r | median slope |
|---|---:|---:|---:|---:|
| raw shares, λ = 0.15 | 26 | 66.6M | 0.42 | 0.92 |
| standardised, λ = 0.05 | 12 | 40.4M | 0.36 | 0.37 |

Standardising recovers `armenian` perfectly — rank 15 to rank 1, +0.55 — and wrecks everything else,
because slope 0.37 means the predictions run three times as far as the truth. **Neither penalty serves
both cases, and the honest reading is that a body defined by a 0.2% ethnicity is not learnable from
between-county variation at all.** Shrinking a small segment harder is not a defect; it is the correct
response to there being less information in it.

So there are **two tracks**, and the second exists because of that finding:

- **fitted** — 26 nodes that clear r ≥ 0.25 on held-out metros. Evidence.
- **authored** — 31 nodes whose ethnicity is *constitutive rather than correlated*: the Armenian
  Apostolic Church is Armenian by canon, Mar Thoma is Kerala, the Ethiopian Orthodox Tewahedo Church is
  Ethiopian. Asserted, in the same spirit as §2.4's 372 hand-mapped placements, and carrying
  `basis: authored` so the claim is never read as a measurement. **The bar is narrow: the body's own
  name or canon must name the ethnicity.** Bodies that merely *skew* ethnic — Southern Baptist, the
  Church of God in Christ — stay with the fit, which is what evidence is for.

Together, 89% of adherents. The rest are population-uniform.

**Placement also stopped approximating.** §8.2 spread a county's dots equally across its tracts because
tracts are designed to a population target; the ACS tract total is a real population and is now used
directly for every node, weighted or not.

**Connecticut is §8.1 for the third time, and in a new direction: two vintages are needed at once.**
The placement layer must be 2020 tracts, because their county prefix is what joins to ASARB — but ACS
2020–2024 publishes Connecticut on the **2022 planning regions**, so 879 of 884 tracts fail the GEOID
join and the state silently reverts to uniform. The tracts did not move, only their numbering, so a
representative-point join from the 2020 polygons onto the 2024 ones recovers all 879. Every state that
adopts a new county-equivalent scheme will need this.

#### What it is not, and the about panel has to say so

1. **It is an estimate.** Nothing below county level here was counted.
2. **It is partly the race map.** "Englewood is Black Protestant" is the input, not a finding. The part
   that is not circular is ancestry and birthplace — Guyanese Hindus in Richmond Hill, Armenians in
   Glendale — which no race map contains.
3. **The check is between counties and the use is within one.** Coefficients get extrapolated well past
   their fitted range: metro counties run 5–25% Black, Cook County tracts run 0–100%. The direction is
   favourable — more demographic contrast, not less — but it is extrapolation and nothing exists to
   validate it there.
4. **`islam`, `hinduism` and the three Buddhist nodes are ASARB compiler estimates** (group codes 267,
   890–892, 895; §3.1). **A number modelled from population can be reproduced by a model of
   population**, so `islam`'s r = 0.64 — the highest here — is not independent evidence. The flag
   travels into the model file. They still take weights: uniform is not the safer answer, only a
   different wrong one.

#### Judaism was the open failure, and language fixed it — 2026-09-04

**Orthodox Judaism scored 0.05.** Brooklyn is +6.9pp against its metro and the model said +0.1pp. The
ACS has no Jewish marker of any kind: `israeli` is Israelis, who are not most American Jews and are
barely any Haredim, and the Haredi neighbourhoods report European ancestries that the segment table
reads as `euro_catholic`.

**The damage was not confined to Judaism, which is the general lesson.** Orthodox Judaism correctly got
no weights — but **Catholic did**, so it took the space instead: Borough Park drew 58% Catholic, 1.3×
the Brooklyn rate, in the most Jewish neighbourhood in America. Skokie and Beverly-Fairfax did the same
thing. **A missing predictor is not neutral; the bodies that do have one absorb what it should have
held**, so an honest gap in one node becomes a confident error in another.

**The block was self-imposed.** `api.census.gov` refuses unkeyed requests now, and that was taken as
the end of the road for the language table. It is not: the ACS **summary file** on `www2.census.gov` is
the same data as flat `.dat` files, one per table, no key. **Worth remembering before concluding a
census table is unreachable.**

B16001 carries **Yiddish** (as "Yiddish, Pennsylvania Dutch or other West Germanic" — the Census does
not split them) and Hebrew, plus Gujarati, Punjabi, Urdu, Bengali, Malayalam, Armenian, Persian, Arabic
and Amharic. Its finest geography is the **PUMA**, ~100,000 people, not the tract — but that is 28
units inside Brooklyn against ASARB's one, and PUMAs are drawn on neighbourhood lines.

**The ground-truth check passes on the nose.** The top Yiddish PUMAs in the United States, in order:
Monsey, Borough Park, Kiryas Joel, Williamsburg, then Holmes County OH and Lancaster PA — which are
Amish, correctly, because the category is a lump and both halves of it are wanted. County
Orthodox-Judaism share against Yiddish share is r = 0.52 across 991 counties.

| | before | after | | after |
|---|---:|---:|---|---:|
| Borough Park | 1.0× | **4.4×** | Kew Gardens Hills (Modern Orthodox) | 4.3× |
| Williamsburg | — | 2.7× | Pico-Robertson, LA | 5.8× |
| Midwood | — | 2.2× | Beverly-Fairfax, LA | 8.0× |
| Bed-Stuy (no Jews) | — | 1.1× | West Rogers Park, Chicago | 2.5× |

It is **authored, not fitted**, and always will be: r = 0.07 under metro holdout because 79% of
Orthodox Judaism is in one metro — exactly the Armenian Apostolic case. Yiddish and Hebrew go in at
equal weight and balance themselves: Brooklyn is 4.4% Yiddish against 0.8% Hebrew and lands on Borough
Park, while Queens and Los Angeles are Hebrew-dominant and land on Kew Gardens Hills and
Pico-Robertson. **No tuning was needed, which is the sign the segments are the right ones.**

**A PUMA is not a neighbourhood, and one case proves it.** PUMA 3604303 is "Bedford-Stuyvesant & Crown
Heights North" and is 6.2% Yiddish, because Crown Heights is the world centre of Chabad. Spread evenly
across the PUMA that made Bed-Stuy — which has essentially no Jews — 1.3× the borough rate. So a PUMA's
speakers are now split across its tracts **by the ancestry that carries the language** — Yiddish by
white population, Malayalam by South Asian, Armenian by Armenian — which is a disaggregation rather
than a new assumption, since Yiddish-speaking Haredim are recorded as white by the race question.
Bed-Stuy fell to 1.1×, and it sharpened everything else at the same time: Beverly-Fairfax 2.8× → 8.0×,
Kew Gardens Hills 3.1× → 4.3×, Artesia 1.8× → 3.9×.

**Two AUTHORED extras came free**, because the same table separates things ancestry cannot: Malayalam
for the four Kerala churches (Mar Thoma, the two Malankara bodies, Knanaya), Amharic for Ethiopian and
Eritrean Orthodox, Punjabi for Sikhism, Gujarati for Jainism and the Hindu share of the Indian diaspora
— `south_asian` alone contains Bangladeshis, who are Muslim.

**Still open.** Hindu placement in the South Asian corridors is the weakest of the authored ties —
Jackson Heights 0.9× and Devon Avenue 0.8× where both should be well above 1 — because `hinduism`
competes with a Guyanese term tuned for Richmond Hill.

### 8.4a The residual gets its own model, and its own ground truth — BUILT 2026-09-04

§3.5a's residual is **166.2M people against the roll's 160.6M** — more than half the US map — and it
was placed by population alone: a flat wash of `unaffiliated`, `secular` and unspecified `christianity`
laid identically over every neighbourhood, diluting every contrast §8.4 draws. Three nodes are 150M of
the 166M.

**It cannot use §8.4's model**: applying a model of where ASARB's adherents live to the people on
nobody's roll would place the residual exactly where the measured people already are, which is the one
place they are not. So it needs its own coefficients from its own target.

**ASARB cannot be that target** — the residual is by definition what no roll holds. **PRRI's county
file cannot either, and this is the trap worth naming:** those county estimates are themselves a
Bayesian small-area model built from ACS demographics, so scoring a demographic model against them
would prove nothing. It is ASARB's "Muslim Estimate" again — **a number modelled from population
reproduced by a model of population.**

**The Cooperative Election Study is genuinely independent**: raw survey microdata, 680,895 respondents,
a county FIPS and a religion question on every row, downloadable without credentials. 407,874 of them
from 2016 on, across 3,045 counties — Los Angeles has 16,070, Cook 11,360, Kings 4,053. Same protocol
as §8.4 otherwise: demeaned within metro, whole metros held out. Two differences, both because the
target is a survey rather than a census — counties are weighted by CES sample size rather than
population, since that is the precision of the target, and only counties with ≥150 respondents are
scored, because a share off thirty people is mostly noise and would understate any model.

| | held-out r | slope | | held-out r |
|---|---:|---:|---|---:|
| hinduism | **0.62** | 1.14 | judaism | 0.45 |
| christianity (unspecified) | 0.51 | 0.88 | unaffiliated | 0.44 |
| secular | 0.46 | 0.73 | other.us | 0.32 |
| islam | 0.46 | 0.83 | *buddhism* | *below the bar* |

Seven of eight clear R_MIN, and they are as strong as the roll models. **Education and age had to be
added** to make it work — ancestry says who people descend from, not whether they go to church, and
degree share alone correlates +0.56 with a county's atheist-and-agnostic share. Both are tract-level in
the same keyless summary file (B15003, B01002).

**What it draws, and the shape is the non-obvious part.** Los Angeles County tracts by share of adults
holding a degree:

| degree share | secular | unaffiliated | unspec. Christian | Catholic |
|---|---:|---:|---:|---:|
| 7% (bottom decile) | 6.1% | 22.1% | 13.4% | 35.5% |
| 74% (top decile) | **20.8%** | **13.2%** | 6.8% | 29.9% |

**The two irreligion categories move in opposite directions.** Atheist and agnostic is 3.4× higher in
graduate neighbourhoods; "nothing in particular" is 1.7× higher in the least educated ones. **A single
"irreligion" axis would have drawn both the same way and been wrong about one of them.** Across
neighbourhoods: Lincoln Park 2.1× secular, Park Slope 1.7×, Silver Lake 1.6×, against Little Village
0.37×, East LA 0.40× and Washington Heights 0.62×.

#### Two bugs, and both were caught by the same discipline

**1. ACS jam values.** `-666666666` means "no estimate", not a number. Left in B01002, median age had a
minimum of −52,769 and a standard deviation of 2,560; that one poisoned column dominated the raw-scale
ridge trace and shrank **every** coefficient to about zero. The fit reported no signal at all —
`secular` scored −0.01 — while the raw correlations behind it were 0.4 to 0.6. Same lesson as §8.4's
first three specifications, reached from the other side: **a null from a fitted model is a claim about
the fitter first.**

**2. The design matrix drifted from the fit, silently.** The Weighter built its own column list by
hand, `ba_share` and `age` were not in it, `beta.get(name, 0.0)` returned zero for both, and the
residual model's two strongest predictors were dropped on the floor. Everything looked right: the fit
validated, the weights inspected directly looked sensible, the run counted 345 rows. **Only a check
against the DRAWN OUTPUT caught it** — dots came out flat across education deciles, 0.98× top to
bottom, while the model was predicting 3.4×. There is now one `design()` function used by both fits and
the Weighter, and a `KeyError` if a beta names a column the design does not have. **A comment warning
that two code paths must agree is not a mechanism; this is why.**

**The spot-check that looked like a failure was confounded**, and it is worth knowing before reading
one. Measuring a node as a share of the dots in a radius is distorted by whatever else is drawn there:
Borough Park is 49% Orthodox Jewish, which mechanically depresses every other node's share. The first
neighbourhood table read East LA as *more* atheist than Silver Lake and was simply not measuring what
it appeared to. **Aggregating by decile, where composition and sampling noise both cancel, is the check
that means something.**

## 9. Viewer

MapLibre GL JS + PMTiles, the ancestrydots stack, which also means the R2 hosting route and the
`npx serve` dev server (Python's `http.server` does not do range requests). The unmerged scatter is a
custom WebGL layer over a flat binary buffer, not a tile source — §4.2d.

**Style is copied from ancestrydots, not from nycriders** — Anita's call, 2026-08-27, and this
supersedes the general house-style note in [[feedback_map_ui_style]]. Its tokens: `body` `#111`,
panels `rgba(20,20,20,0.758)`, hairlines `#2a2a2a`/`#333`, scrollbar thumb `#555`, Nunito, the `i`
button bottom-left for prose. Dark, like ancestrydots and unlike citybrowser.

**One thing tiling takes away: the viewer can no longer count anything.** With GeoJSON it totalled the
country by walking features; with tiles it only ever holds the current viewport, so the panel's
per-religion totals are precomputed into `data/processed/counts.json` by `tiles.py`. **Anything else
the UI wants to state about the whole dataset has to be computed at build time for the same reason.**

**Dot sizing is already the answer to §4.2's dot-size half:**

```js
'circle-radius': ['interpolate', ['exponential', 1.26], ['zoom'], …]
```

1.26 per zoom against a scale that doubles — radius grows about as the square root of the scale factor,
which is exactly the sublinear behaviour §4.2 asks for. **It is tuned and it transfers; do not
re-derive it.** (Every radius then carries a single 1.3 `DOT_GAIN` on the base, so the control still
reads 1.0× at rest and the zoom curves stay readable as themselves.)

Panels: genealogy/legend (§10), unit composition on click (§5), about behind an `i` button.

**IT OPENS ON THE WORLD — Anita, 2026-09-06.** The United States framing was carried over from
ancestrydots, where the map *is* the United States. Here it made a map of thirty-one countries open on
one of them, with Auto resolving to it before the reader had seen there was anything else to look at.
Three things make that fit work, and the first two are not obvious:

- **`bounds`, not a centre and a zoom**, so the frame follows the window rather than assuming a desktop
  one.
- **The box is the data, not the globe** — -127°E is the west coast of the United States and 180°E is
  New Zealand, the outermost view boxes any built country has. Cutting the empty Pacific out is worth
  about a third of a zoom level of dot size. It is a constant rather than a walk over `META`, because
  `META` does not exist until `counts.json` lands and **a camera that jumps once the data arrives is
  worse than a number that needs widening the day a country outside it is built.**
- **Asymmetric padding, and only the right side is real.** The fly-to padding table is wrong for this
  one view: it reserves 250 px on the left for a panel a hundred pixels tall, so the world lands
  smaller for nothing — and at this zoom a world narrower than the window means MapLibre draws a second
  New Zealand at the opposite edge. The legend on the right is the only occluder running the height of
  the map.

`minZoom` drops from 2 to 1 for the same reason: 512·2² px of world against a 390 px phone is 68° of
longitude, which is a view of the Mediterranean. Nothing below zoom 3 can take the legend anyway
(§6.2). **A window narrow enough still gets the fit clamped at `minZoom`** — a phone opens on
Europe-to-Indonesia rather than on the world — and that is left as it is rather than hand-centred.

**The hover card is dark, and it carries the legend's own colour key — 2026-09-06.** MapLibre ships the
popup white, with `10px 10px 15px` of padding (the extra 5 px clears a close button this one does not
have), and it was the single piece of light-mode chrome on the map. Same tokens as every other panel,
padding even on all four sides. **The key matters more than the colour scheme does**: zoomed out, the
mark under the pointer is three pixels of a crowded field, and the name alone does not settle which of
them you are reading — the legend answers *what colour is this religion*, and the card has to answer it
the other way round, in the same shape, off the same `colorOf` table. Both hover paths — the delegated
one for merged marks and rings, and SCATTER's own — build the card through one function, because **a
card that changed shape with the layer under the pointer would be reporting the renderer rather than
the mark.**

### 9a A guard on a TRANSIENT condition became a guard on a PERMANENT one — FIXED 2026-09-07 with Saint Vincent

Auto picks the country the camera is over. It samples the middle of the viewport, tallies which
built country the dots belong to, and then tests the winner's bounding box for fill and overlap.
Before any of that it bails:

```js
if (total < AUTO_MIN) return;   // nothing loaded here yet; `idle` will bring us back
```

`AUTO_MIN` is 150 and the comment says exactly what it is for: **the tiles have not arrived
yet.** That is a transient condition, and returning is right for it.

**But it is expressed as an absolute dot count, and a small country never has one.** Saint
Vincent draws **98 dots in the entire country** at 1:1,000. Liechtenstein draws **37**. Neither
can reach 150 from any camera position, ever — so the tally always bailed, `viewFill` and
`viewOverlap` were **never evaluated**, and Auto fell through to *"all countries"* over them
permanently. The country legend, the grain line, the `gap` note and the country note were all
unreachable in the default mode, on countries whose dots, tiles, buffers and `counts.json`
entries were all perfect.

**Every country was too big to notice for forty-four builds.** The 45th was not.

**The fix separates the two questions.** When the tally is short, ask the geometry instead: if
the camera unambiguously frames exactly ONE built country, that answers the question the tally
was being asked and the dot count is irrelevant.

```js
if (total < AUTO_MIN) {
  const framed = BUILT.filter(cc => META[cc] && viewFill(cc) >= AUTO_FILL_IN
                                    && viewOverlap(cc) >= AUTO_OVER_IN);
  if (framed.length === 1 && framed[0] !== country && !contested(sh, framed[0]))
    setCountry(framed[0], { fly: false });
  return;
}
```

Both thresholds are the strict IN ones and `contested` still applies, so this cannot fire over
a border, over open water, or on a view wide enough to hold a rival. Where nothing is framed it
returns exactly as before, so the loading case is untouched.

**THE GENERAL LESSON, AND IT IS WORTH MORE THAN THE BUG.** A guard written for a transient
condition — *not loaded yet*, *not ready*, *too early* — and expressed as an **absolute count**
silently becomes a guard on a permanent property: *too small to matter*. Nothing in the code
says so and no test fails; the feature simply never fires for one class of input. **When a
threshold stands in for "not yet", check what it says about the smallest legitimate case**, and
prefer a test of the thing you actually mean.

### 9b Auto will not enter a country that has none of what is selected — DECIDED 2026-09-07

> *"if we have a religion selected like daoism and zoom in on a country that doesn't have any
> daoism, in auto mode, it just autos to the country and then clears the selected religion cuz
> theres no daoism to show. would rather have it just not auto to countries that dont have any of
> the religion."*

Exactly what happened, and by design at every step: `setCountry` ends in `rebuild`, `rebuild` drops
a scope that is not `PRESENT`, and so **a reader loses a selection they never touched, by panning.**
Daoism is in 4 of the 68 built countries, so 64 of them were a trapdoor.

**The fix is a veto, not a repair**, and the alternative is worth stating because it is the obvious
one: enter the country and keep the dead scope. That gives a panel naming a religion with no rows
over a map drawing no dots — a country view that answers nothing. Refusing the country instead
leaves Auto at **all countries**, where the selection is still live and still drawn everywhere it
exists, which is the state the reader was already in.

**It is not a lock.** The country menu still switches to it, because going somewhere deliberately to
find out what it *does* have is a reasonable thing to want, and the veto is about what the camera
does on its own. §9's rule that Auto is a mode and everything under it is a place survives intact.

**`scopeDrawsIn` mirrors `rebuild` line for line** — the rolled tally under §7a's roll-up and the raw
one otherwise, rings only while rings are drawn, the same "nothing tallied at all" escape. Anything
else and Auto would refuse a country whose legend would have shown rows, or enter one it would not.

**The escape is the important half, and §9a is the standing warning.** A country whose `counts.json`
entry has not arrived must read as *no evidence*, never as *evidence of absence* — otherwise this is
§9a again exactly: a guard written for "not loaded yet" quietly becoming a guard on a permanent
property of one class of country.

**The menu carries the other half.** A veto is silent: nothing happens, and a reader watching nothing
happen cannot learn why. So with a religion selected the figure beside every country in the picker
is **that religion's**, and the countries with none say so with an em dash and a tooltip. Pick Daoism
and the menu is the answer to "where is there any".

## 10. The tree panel, and the genealogy drawn on it

**§6 splits this into two things that were one thing.** The panel is the legend, the selection control
and the palette scope all at once, so **it is in the first build** — the map does not work without it.
Only the *genealogy* half, the descent edges and the time axis, is later.

In the build today:

- The containment tree, collapsible per family (`<details>`, as ancestrydots), each node showing its
  current colour.
- **Selection is shared with the map, both directions.** Select a node → the focus palette applies and
  everything else leaves the map (§6.7). Click a line in a unit's composition panel → the tree scrolls
  to and highlights that node.
- Selecting a node selects its subtree, which is how you ask "where is Orthodoxy" rather than "where is
  the Romanian Orthodox Church".
- A **display-depth control** per selection, since it decides how many categories the wheel is divided
  among (§6). ± buttons rather than a slider — the useful range is about four values.

Later, on the same panel: a vertical time axis, `from` edges as lines, dashed where `kind: disputed`
(§2.1). **The descent edges are the reason §2.1 exists now rather than later** — retrofitting them onto
ids that were not designed to carry them is the expensive version.

### 10.0 The families are in a FIXED order — DECIDED 2026-09-03

The top level used to sort by prevalence, like every level below it. That is right inside the tree and
wrong at the top, and the reason is that **the panel is the legend**: a legend whose rows move between
countries is one you re-read at every country instead of learning once, and between-country comparison
is the thing this map exists for. `unaffiliated` alone was second in Poland, first in Czechia and
Estonia, fifth in Romania, and absent from the US.

What this gives up is "the biggest thing is always at the top". Worth it: the sizes are on the rows for
anyone who wants the ranking, whereas "where is Islam in this country" cannot be answered by searching
a list that moves.

**It changes no colours.** Root hues are authored in `ROOT_HSL` (§6.3) and `buildPalette` walks `NODES`
in file order rather than panel order, so display order and hue order are independent at the top level
— which is exactly why this could be changed without re-tiling. Below the top they are still the same
sort, deliberately (§6.5, §6.8).

### 10.0a The grey family is contiguous, and `unaffiliated` leaves the top — DECIDED 2026-09-05

Anita's call, and it amends §10.0's order without touching its argument. `ROOT_ORDER` in `index.html`
is now:

```
christianity  islam  judaism  hinduism  sikhism  buddhism
unaffiliated  secular  parody  unrecorded  unchurched  other   … then branches.py order
```

**The defect it fixes is that §6.3a's grey family was drawn in two places.** "No religion" led the
panel and the other five trailed it, with every world religion in between — so the six rows that are
one kind of answer never appeared as one kind of answer, and §7a's "no religion" control acted on rows
at both ends of a scrolling list. §10.0's reason for putting `unaffiliated` first was that it is one of
the three answers a census actually collects, which is true and is not worth splitting the family for.
**`parody` joins the list**; it was never in `ROOT_ORDER` and fell through to "everything else", which
put a grey row after the small world traditions — the same defect in miniature.

**The run is ordered person → instrument**, which is the honest axis and reads down as a scale:
`unaffiliated` (asked, said none), `secular` (asked, stated a position), `parody` (asked, refused the
question), `unrecorded` (never asked at all), `unchurched` (asked, reported belief without
institution), `other` (asked, answered, and the tree cannot place it). §6.3a-i's point that
`unrecorded` is a property of the instrument rather than of the people is what the position in the run
now says on its own. (`unknown`, §6.3a-ii, joins the run on the same logic.)

Families a country does not report are dropped by `PRESENT`, so nobody sees an empty row. It changes no
colours, for §10.0's reason — verified after the change.

### 10.1 What the panel says about itself — DECIDED 2026-09-03

Four things, all found by looking at the panel rather than reasoned to, and all about the same failure:
the panel showed *what* without showing *how much of what*.

**Selecting a node shows one level below it, and no more.** `depth` counts levels below the scope and
carried the unscoped default of 2 into a selection, so clicking Christianity drew its grandchildren —
every Anglican and every Catholic body at once, eighty rows against a question ("what are the
Christianities") that has twenty-seven. One level answers it; `+` is still there for the next. Two
stays right when nothing is selected, because there the first level is thirty families of which one is
84% of the map (§6.1).

**A white line at the top of the panel names the cut: `Viewing: Adventist subgroups (L3), United
States`.** Group, level, country. The levels are numbered from the families — Christianity is L1,
Lutheran L2, the Missouri Synod L3 — and **the number is read off the drawn set rather than computed
from `depth`**, because the two differ wherever the tree runs out early. Select Islam, which no source
in the archive divides, and `depth` says one level below while the drawn set is Islam itself; the line
reads `Viewing: Islam (L1)` because that is what is on the map. It is the only white text in the panel,
and it exists because until it did there was no way to tell Adventist's own row from a view *of*
Adventist's subgroups.

**A rule marks the selection and nothing else.** Rules used to sit on every lineage caption, which drew
a line across the middle of Christianity between Jehovah's Witnesses and "no single line" — a division
of a family that no reader has a use for, in the same weight as a division between families. **The
division worth drawing is the one the map is actually filtered by**, so two rules now bracket the
selected subtree, above its row and below its last descendant. The captions stay; they were never the
problem.

**Outside the selection the names go quiet.** §6.7 removes those dots from the map entirely, so their
rows are a list of things that are not drawn, and they were competing at full contrast with the rows
that are. Dimmed, not removed: they are still the way back out.

### 10.2 The legend's settings are SEGMENTED PAIRS, not blue words — DECIDED 2026-09-07

The block under the tree carries five settings — people per dot, dot style, presence rings, show
non-religions, inferred dots — and a reset. Every one of them was a line of grey label plus a blue word:
`presence rings: hidden`. The line states the setting correctly and that is all it does. **Nothing in
it says the word is a switch.** A reader who has not already guessed that the panel is interactive
reads `presence rings: hidden` as the map telling them a fact about itself, which it also is, and
there is no second reading that makes the word look like a control. The blue was the whole affordance
and blue is the colour this map uses for links to prose (`basis-link`, `scope-clear`), so it did not
even mean *switch* consistently.

**Each setting is now a pair of segments with the live one filled.** `[ hidden ][ shown ]`, one lit.
Three things change at once:

- **Both states are on screen.** The old control named the current state and left the reader to work
  out what the other one would be — which for `inferred dots: shown` is not guessable. The pair says
  what the choice is *between* before anyone commits to it, which is most of why the tooltips were
  carrying so much weight.
- **The target is a chip, not a six-character word.** `shown` as a link was about thirty pixels of
  clickable text. The chips are 22px tall with the padding, and the whole row carries the tooltip.
  They are also **set at the legend's own 12px** (Anita, same day). They shipped a size down, on the
  argument that a setting is subordinate to the thing it sets; two sizes below the rows above them
  read as fine print instead, which is what moving off links was meant to stop. The widest row,
  `show non-religions [hidden][shown]`, measures 220 of the panel's 266px of content width, so the
  size is not bought on credit — 46px spare and nothing truncates. Re-measure after any change to
  `#panel`'s width, which moved from 268 to 286 the same day.
- **A fill, not a colour.** A blue word among grey words is exactly what nobody noticed, so the live
  segment is a filled blue-grey (`#7d9cb8`) with near-black type — the one loud thing in a panel that
  is otherwise all low-contrast greys, and deliberately so. Reset is one chip in the same frame,
  unfilled, because it is an action and not a state.

**Disabled is `.off` on the frame, and it means two different things that look alike.** Reset greys
when there is nothing to put back. The dot-value control greys when the archive was built without
`--coarse` — and there the second segment is *removed* rather than greyed, because a greyed `10,000`
advertises an edition this build does not contain, which is a worse lie than saying nothing. What is
left is one dim chip stating the value.

**`1 dot = 1,000 people` is gone as a sentence** and is `people per dot  [1,000][10,000]` as a row. It
was the only one of the six that read as prose and it could not join the grid without saying the same
thing twice. The about panel still opens with the sentence, which is where it was doing the explaining
anyway. The derived/modelled readout moved off the end of the `inferred dots` line onto its own line
under it, and lost the em dash it needed there.

### 10.3 A share bar per row, and a column of checkboxes — DECIDED 2026-09-07

Two asks, and they turned out to share a denominator (§6.10a) and nothing else.

#### The bar

> *"show rough like 'what percent of the total dots in scope is this religion for' in the legend as
> a partially filled in bar the same color as the dot... 80% full bar for 10% of total, 60% bar for
> 1%, 40% bar for 0.1%, 20% bar for 0.01%."*

The counts were already on every row and they are the *exact* answer. What a column of numbers is
bad at is the **comparison**: religion sizes run over five orders of magnitude inside one country,
and `1.2b` above `9.7m` above `342k` gives a reader three figures and no picture.

**The scale is logarithmic and has to be.** Linear, everything under a few per cent is an empty bar
— which is most of the legend, and exactly the rows nobody can size by eye from the number alone.
Anita's anchors are a fifth of the bar per decade: `fill = 1 + 0.2·log10(share)`.

**The bottom compresses rather than terminating.** Clamped at zero the scale ran out at 0.001%, and
with no track behind the bar a zero-length fill is not a short mark, it is **no mark** —
indistinguishable from a row with no figure at all, which in the all-countries view is a lot of
rows. So below the 0.01% knee each further decade adds **half** of what the last one did: a
geometric series converging on 6% and never reaching it (0.001% → 13%, 0.0001% → 9.5%, → 6%).
Continuous at the knee, still monotonic, and **above the knee the four anchors are untouched**.

**A full bar means "all of it", not "the biggest one here"**, which is why the selected node's own
row draws full — it is 100% of itself.

**It ignores what is hidden.** Re-scaling to the rows left standing would make the bars mean
something different after every click, and would put a full bar on the last row a reader had not yet
unticked. The bar answers "how much of this map is this", which no checkbox changes.

**A rule, not a gauge.** It shipped 3px on a filled track and that was a widget; on sixty rows it was
the loudest thing in a legend whose job is to stay quieter than the map. At 1px with no track it is
an underline in the number's own colour, and the eye takes the column as a shape rather than reading
sixty little meters. What the track was doing — showing the length the fill is a fraction *of* — the
**column** does instead: rows sit at a fixed 36px, so the longest rule in view is the reference. That
works because these are always read as a group and would not work for a bar on its own.

**Under the number and filling leftward.** Above it, the rule sat between two rows and the eye had to
decide which one owned it. Under it, it is an underline, and an underline has never belonged to
anything but the text above it. The counts are right-aligned, so a bar growing rightward from a fixed
left edge grew *away* from the number it measures; anchored to the same right edge the two share a
margin. **No gap between the two**: the number's own line box already leaves descender space under
the digits, and a gap on top of that read as a detached rule rather than as an underline.

#### The checkboxes

> *"ability to select religions for viewing, like in ancestrydots. people might wanna show just like
> baptist and catholic or something."*

**`USER_HIDDEN` had to stop being a set of subtree roots**, and "just the Baptists" is why. Under the
old set, hide everything and re-check one body and there is no way to say what is left:
`christianity`'s **own** dots come back with it, because one id governed a node and its subtree
together. Those are the sources that answer "Christian" and name no church below — **46% of the marks
across the archive and 18% of Canada.** The reader asked for Baptists and got Baptists plus every
unspecified Christian on the map.

So the set is now **exact**: one entry per node whose own dots are off, and hiding a branch adds
every id under it. The cost is a bigger set (~570 entries after "hide all" against one) and it is
paid nowhere that matters — `isHidden` becomes a hash lookup, the WebGL palette is a per-node loop
either way, and only the merged layers' *filter* wants the old shape, which `hiddenCover` gives back
minimally at the point of use, as the largest wholly-hidden subtrees plus the residue. `localStorage`
keeps those two halves under the old key, so a store written before this loads and still means what
it said.

**The `unspecified` row gets its own box**, and it is the one place a node moves without its subtree.
That row is the reason the model changed, so it is the row that has to be able to act alone.

**The state is computed over the PRESENT subtree; the action runs over the whole taxonomy.** Hiding
Christianity in Poland hides the Baptists Poland does not report and the United States does — or
switching country would bring them back. But the *tick* is computed over the rows in front of the
reader, because a global count would draw Christianity half-ticked in Poland with every Polish row
under it ticked, which reads as a broken panel rather than as a fact about a country not on screen.

**A column on the right, not a staircase.** Beside the swatch they would follow the indent, so four
levels down they sit forty pixels in from where they started and a reader ticking three of them off
is chasing a diagonal. **The resting state recedes**: every one of 381 boxes is ticked at rest, so a
ticked box is half-opacity dark steel and an unticked one is full strength — a reader scans this
column for what they turned **off**. It is deliberately not §10.2's `#7d9cb8`; there are four lit
segments and 381 of these.

**THE HOVER CUE BELONGS TO THE HIT AREA, NOT TO THE ROW AROUND IT**, and this shipped wrong:

> *"if i hover over the row (like over the religion title, for instance) it makes the checkbox look
> brighter so it feels like i'm gonna click it. but it actually just focuses the religion."*

The brightening was on `.row:hover`, on the reasoning that the row the pointer is on is the row worth
lighting. But **a row and the box inside it do two different things** — select the religion, toggle
its dots — and lighting the box on the row's hover promised the second while the click delivered the
first. The general rule, which applies to the swatch beside it and to anything else nested in a
clickable row: **a control may only light up for a pointer that is actually over it.** Hover feedback
is a promise about what a click will do, and a control inside another control cannot borrow its
parent's.

**36px for the count column is measured.** The widest string `fmtPop` can produce is four digits and
an `m`, which is 35.8px in Nunito 11. Every pixel over comes off the label: at 38px with a 15px box
nine labels truncated against five before the row grew a bar and a box at all; at 36 with 13 it is
five again, which is the pre-§10.3 legend exactly. *(Those counts are with the scrollbar hidden. The
tree scrolls in practice, and its 6px takes it to ten either way — the comparison holds, the absolute
number does not.)*

**The hit area is a square the height of the row**, which is §6.11's argument for the swatch applied
to the box beside it: *"a 9px circle is a legend mark; it is not a button"*, and an 11px checkbox is
not one either. `align-self: stretch` gives it the row's full height and a negative right margin
claws the width back out of the row's own padding — so a 19px square target costs the labels exactly
what a 13px one did, because the pixels come from padding that was doing nothing. Clicking anywhere
in the row's right margin toggles the row.

**One link that flips**, and which way is read off the tree rather than remembered: everything shown,
the only useful offer is `hide all`; anything else, `show all`.

**It shipped stepping aside for `clear` when something was selected, and that was taken back the same
day.** The argument for hiding it was that two blue words a few pixels apart do unrelated things and
`clear` is the one wanted at that moment. The argument against is that **both are wanted**, and a
control that vanishes when a selection is made is one a reader has to discover twice. It also could
not be applied consistently: `show all` had to keep showing regardless, being the only way back from
a map with nothing on it, so the bar offered one link or two depending on a state nobody was
tracking. Both show, always.

**The panel keeps its scroll position across renders.** The tree is thrown away and rebuilt on every
change, and a checkbox is a control a reader clicks several of in a row — each one used to send them
back to the top of a scrolling list to find the next.

### 10.4 One bar for the whole scope, and a hatched segment for what nobody counted — DECIDED 2026-09-08

> *"a bar that's filled proportionally with colored segments for each of the religions being shown
> in the legend currently … in this case i'd like to see ~45% of the bar be gray for no religion, a
> big chunk be catholic."*

It sits between the `Viewing:` line and the first religion row, on a line of its own, because it is
that sentence drawn instead of written: *all religions, France* is what the caption says and this is
what the answer looks like. France comes out 47% grey, 37% amber, 8% green, which is the country in
one glance and is not recoverable from §10.3's column of bars at all — those are read *down*, one
row against another, and answer "how big is this one". **Nothing on the panel answered what the
place adds up to.** Summing sixty rows by eye is not a way to find out, and hunting the map for a
colour that is 3% of it is worse.

**The segments are the DRAWN categories, not the legend rows.** The rows nest, so adding them up
counts every French Catholic three times over — Christianity, Catholic, Latin Catholic. `DRAWN` is
the set that partitions the map: exactly the nodes carrying a colour of their own, so its members
sum to the scope once each. A segment's value is its whole subtree **minus the subtrees of the drawn
categories under it**, which is the same arithmetic `paletteFor` does in paint — the parent covers
its subtree, each drawn child paints over its own part, and what is left holding the parent's colour
is its own dots plus whatever folded into it (§6.10). `MERGE_OWN` hands its leftover to the child
that draws it (§6.15). **The bar is a picture of the picture**, and it has to be exactly that or the
two disagree about a country in a way nobody would ever catch.

**Legend order, not size order**, off the same `KIDS` walk `renderTree` reads down. It keeps the
families contiguous (§10.0), so a run of ambers in the bar is the run of Christian rows beside it and
the eye can carry one to the other. Size order would sort that run apart for a ranking the numbers
already give.

**It ignores what is hidden**, which is §10.3's rule unchanged. A bar that re-scaled to the ticked
rows would read "100% Catholic" of a France that is 37% Catholic.

**No minimum segment width, and that is the honest choice.** A country's smallest drawn group is four
decimal places below its largest — Chinese religions are 4k of France's 68m, a sixtieth of a pixel —
and a one-pixel floor would hand thirty specks a third of the bar between them. Flex children lay out
at 1/64px and antialias, so a sub-pixel segment tints its pixel and disappears at the right rate.
**`flex-grow` on a zero basis, not a percentage width**: percentages that sum to `1.0000` in floating
point leave a hairline of track showing at the right end, and on a bar whose whole claim is that it
is full, that is the one artefact that matters.

**A count beside it, in the rows' own format.** The bar is a set of proportions and says
nothing about size, so without a figure the same shape can be Iceland or Indonesia. It went above
the bar first, in §10.3's number-above-rule shape, and came straight back down — *"maybe the number
should be to the right of the bar rather than on top of it so we dont use a whole row of space"*.
Over the bar it cost a full line of a panel that is already the tallest thing on screen; beside it
costs six pixels of height and 36 of width, which is `.row .ct`'s measured column reused. **Fixed
width and not shrink-to-fit**, so the bar's right edge does not move between a country reading `29m`
and one reading `1335m`; these get compared across countries. **It is the DRAWN total, not the
bar's** — the sum of the coloured segments, matching the rows below it — and the hatched share sits
outside it, which the tooltip says. A headline number that quietly included people the map does not
contain would be the one figure here nobody could check.

**Five pixels drawn inside a seven-pixel target**, which is why there are two elements. It shipped at 7
and was the heaviest thing in a panel whose job is to stay quieter than the map; a 5px rule is the
right weight beside 11px type. But it is also a miserable thing to point at, and unlike every other
control here the pointer has to *stay* on it while it reads along, so `#mixbar` keeps the original 7px
and holds the pointer while `#mixtrack` inside it is the 5px that is drawn. §10.3's rule about the
checkbox, again: **the target is not the mark.**

#### The hatched segment: what the source never counted

> *"for many coutnries we have like ~10% missing data cuz of not counting children or missing data
> for certain regions … we can show the misssing data as a segment of the full bar thats like gray
> diagonal striped."*

§7c's `gap` row has said **who** is missing since 2026-09-07 and could never say **how many**, because
nothing in the pipeline knows: the dots are what the source published, so drawn-plus-undrawn is a fact
about the *source's universe* and not about the archive. Peru's census asked nobody under twelve and
no row anywhere counts them. So `gap_share` in `countries.py` is **authored, stated once beside the
`gap` sentence that explains it**, and it is the only number in that file the pipeline cannot check.

**Only where a published figure says so.** It is absent where `gap` names something nobody has
quantified (Georgia's Abkhazia and South Ossetia, Turkey's Alevis, Jamaica's four faiths left out of
the parish tables), where the gap is a *pooling* rather than a hole (the Bahamas' small religions are
drawn, in one cell), and where `gap` opens with "none". **No number is a fine answer and a guessed one
is not**, because unlike every other authored string here this one is *drawn* — it resizes a segment
the reader reads off the screen, and [[feedback_dont_draw_unsourced_breakdowns]] applies to a width as
much as to a chart. Twenty-four of the thirty-eight countries with a `gap` have one at first fill.

**Diagonal grey on grey**, and it is the one fill on this bar that is visibly *not a religion*: no
colour of the map's own can stand for people the map does not contain, and §6's rule that every
colour drawn appears in the legend would be broken by any that could. It reads as hatching at any
width.

**One country, and only at the top level.** In the all-countries view there is no country whose
universe it could be a share of — `renderProvenance` declines the whole §7c block there for the same
reason. And under a selection the hole is not part of what is selected: France's uncounted are not
Christians, so a stripe inside a Christianity bar would be claiming they were. Anita's own condition,
*"when we dont have specific religions selected"*.

#### 10.4a Half of it is computed, and the half that is not was worth ruling out — `tools/gap_share.py`

The first fill of `gap_share` was twenty-four hand-typed figures, read off the `gap` sentences that
already stated them. The question that replaced them was Anita's: *"how feasible is it to quantify it
by just subtracting from the total country's population in some year?"* **It splits in two, and only
one half needs a number from outside the archive.**

**Kind 1, a column the census printed and this project declined to draw** — a non-response, a "not
stated", an explicit refusal. Those people are *inside* the census total, so there is nothing to
subtract from: the share is `1 - drawn / the unit's own population total` and **both numbers are in
`data/normalized/<cc>.csv` already.** No outside figure, no vintage, no definition to reconcile.
`tools/gap_share.py` computes it, and it reaches the residual from both ends — `universe - drawn`
where an excluded row is big enough to contain the drawn population, and the sum of the excluded rows
that are smaller, which are the refusals. **Where the two agree, that agreement is the evidence.**
Austria comes out 2.00% against a hand-typed 2.00%, Eswatini 2.19% against 2.20%, Armenia 1.68%
against a figure another session had typed the same day. It went from 24 countries to 55.

**What it caught.** Montenegro was authored at 4.7%, MONSTAT's stated disclosure-control share, and
the data says **6.61%** — the withheld cells plus a refusal cell nobody had added to them. That is
the one direction `--check` fails on: an authored figure *smaller* than a residual the data can prove
means the bar is understating a hole on screen. The other direction is the healthy state and is left
alone — Barbados authored at 18% against a computed 1.23% is a kind-2 hole somebody researched
sitting on top of a kind-1 one.

**Kind 2, people who were never in any table** — Peru's under-twelves, the Galápagos, Abkhazia. Only
here does a country total come in, and **subtracting from a modern population is the one thing not to
do.** Austria's religion figures are 2001: subtract its 7.87M drawn from Austria's population now and
you get 14% "not drawn" where the truth is 2.0%, and the other twelve points are twenty-three years
of growth. Seven times the real number. Definitions bite as hard — de facto against de jure, resident
against present — and Singapore's gap *is* the definitional difference, so differencing two national
totals there measures the thing in dispute rather than resolving it. **Anita's call, 2026-09-08:
leave kind 2 alone.** *"maybe we shouldnt do kind 2 cuz there are a lot of edge cases."* It stays
hand-written, from the source's own age or population table, and absent where nobody has quantified
it.

**And never off the dot totals.** They are fine for a big country — Côte d'Ivoire −0.02%, Peru
−0.01%, China −0.00% against the counts — but a group under one dot leaves the map for a ring (§4.3),
so a small country loses whole percentage points: Antigua −8.7%, the Cook Islands −24.9%, and
Montserrat and Niue draw *zero* dots at 1:1,000. A dot-based subtraction would call those two 100%
undrawn.

**The dots are the one thing that catches a wrong `geo_level`**, though, and that is what they are for
here. Every other figure on a row is computed off the same level, so a wrong level is
self-consistent and looks perfectly reasonable: Ireland's 345,165 not-stated over 755,455 counted is a
believable 31%, and the country is five million people. **The rounding is one-sided** — nothing
invents people — so dots *over* counts means the counts are part of a country, and Ireland's are 535%
out. Switzerland, Croatia and New Zealand were all caught the same way and are all refused.

**Three classes of excluded row, and only size tells them apart.** A universe (at least as big as
everything drawn), a residual (the refusals), and a **nested subtotal** — Poland's `należący do
wyznania w tym:` at 92% of the drawn, Serbia's `Christian - All` at 94%, Myanmar's `Total` at 98% —
which route B would otherwise sum as a hole. The threshold cannot be tightened past **Hungary**,
whose 3.85 million no-answers are 67% of what it draws and whose `Catholic` subtotal is 50%: no size
separates those, so Hungary is a country a person has to look at, and its 40.1% is hand-entered from
the row the tool printed. A fourth class needs listing outright, because size and wording both fail
on it: a row taken off the tree because something downstream **redraws it in a different shape**.
India's `Other religions and persuasions` is 7.9 million people who are all on the map as their 83
Appendix children (§3.10), and counting it as undrawn puts India at 0.89% against a real 0.24%.
`REPLACED` in the tool names those, and the tell that one is missing is the dot column going
positive — the map drawing more people than the level counts is only possible when such a row is
being read as a hole.

**What it costs, and it is a real cost.** §7c's `not drawn` row said *"Fifteen countries have one and
that is the healthy state: a row every country carries is a row nobody reads."* Sixty-five of a
hundred and four carry one now, because most censuses print a non-response cell. That note is no
longer true and `countries.py` says so; whether the left panel's row wants a threshold of its own is
open, and the bar does not — a hatched segment is exactly as loud as the hole is wide.

#### The tooltip, and the two denominators

The hover card is the **dot card's shape** — a 9px key, the name, a grey second line — deliberately,
because that card already answers "what is this colour" the other way round and a reader should not
have to translate between two card designs six pixels apart.

**Each segment names its own denominator, and they are not the same one.** A religion reads *"25m
people · 37% of all religions in France"*, a share of the **scope**, which is the number the row above
it prints; two figures for one religion, that close together, would be worse than any tidiness gained
by making them agree. The hatched segment reads *"21% of Peru"*, a share of the **country**, because
that is the only frame in which "not drawn" is a quantity at all. So on a country with a gap the
religion percentages sum to 100 across a bar that is visibly not full. That is the honest reading of
it, and the hatch's own label says what the rest is.

**The hatched segment's card prints `gap` and nothing else**, and it took a correction to get
there. It shipped as `21% of Peru · under-twelves, 21.1% of the country, who were not asked`, which
is one fact told twice at two roundings with a middle dot between them, and Anita read it the way it
was written: *"we seem to be saying that 21% is not drawn and then another 21% is not drawn for a
separate reason."* Her fix was *"lets say like 21% not drawn cuz of case x or case y. or break it
down. whatevers available"* — one statement, and a breakdown where there is one.

**So the sentence is the single statement and `countries.py` asserts it carries the figure**: a
`gap_share` whose `gap` states no percentage within a tenth of a point of it fails the import. Fifty
three of the fifty five already did, because that is how the field has always been written; Ecuador
and Laos gave a headcount and got a percentage added, and Armenia had one written the same day by
another session and got the same treatment. **A composed figure could not do the other half of the
job**, which is why this direction rather than stripping the prose: Montenegro's hole is 4.7%
withheld by MONSTAT's disclosure control plus 1.9% who declined to declare, and only the sentence
can say that. Where a country has a quantified part and an unquantified one, the semicolon does the
work — Angola reads `the 2.3% who did not answer or did not know; and children under 2, who were not
asked the religion question`, and the missing second figure is the point.

**The bar is one trigger with thirty answers**, so `showTip` grew an optional `atX` and the tooltip
follows the pointer along it. It is the only trigger on the page that needs it; everything else means
one thing wherever you point at it and keeps aligning to its own left edge.

## 11. Open questions

Design questions for Anita are in `todo.txt`. The ones that are mine to resolve with a prototype:

1. **Does a hollow ring read as "present, unquantified", or as a small dot?** Rings are decided (§4.3);
   their *drawing* is not. Ring weight, size and z-order all bear on this, and the honesty of §4.3
   depends on the two symbols not being confusable.
2. **What cell size does the merge need, per zoom, for a mixed region to read as mixed?** The radius cap
   half of this is answered (§4.2a); the cell size is not.
3. **How many drawn categories can the focus palette actually separate — 8, 15, 30?** The answer sets
   the sensible default display depth for each scope, and it is the same number that decides how deep
   the tree is worth building. §6.9 has the measured answer for the *overview* (Christianity's branches:
   none of them) and not for focus.
4. **Whether the overview palette should be reachable while a selection is active** — an escape hatch
   for "show me Christianity but keep it looking like Christianity". Cheap to add, and possibly clutter.
5. **How bright should `unaffiliated` be?** `check_palette.py` flags it at contrast **2.6** against the
   basemap (with `daoism` 2.3 and `unification` 2.7), and it is the majority answer in two drawn
   countries — 68.3% of Czechia, 58.4% of Estonia — at `HSL(234, 26%, 41%)`.

   **It is legible when drawn**, which is worth stating because it was briefly written up here as a
   defect on the strength of a screenshot where the no-religion layer was simply switched OFF. **The
   correction matters more than the original point: the tool's contrast number is a note to watch, not
   evidence of a failure, and the way to test it is with the layer shown.**

   What is genuinely true: Poland and Czechia have the **same dot density** — 0.096 vs 0.093 dots/km²,
   three percent apart — and read as completely different maps. That difference is entirely
   composition, which is R1 working. Poland's 89.9% Catholic sits at `HSL(50, 92%, 64%)`, the brightest
   hue on the wheel, against Czechia's darkest, so the contrast between the two countries is somewhat
   flattered by the palette even though both are readable on their own. Open as a question of degree:
   whether a lightness nearer 50 would serve the irreligious countries better without unbalancing the
   rest.

## 12. Adding a country — the playbook

**This section is for whoever adds the next country, and it is meant to be added to.** If you find a
trap that is not here, put it here before you finish, even if it seems obvious in hindsight — most of
the entries below cost an hour each and would have cost five minutes to read. Keep it to things that
*generalise*; a fact about one country belongs in its `sources/<cc>.md`. As the list of easy countries
shrinks the rate of new tricks will drop, which is fine — **a short section that stays true is better
than a long one that rots.**

`COMMANDS.txt` has the runnable checklist. This is the reasoning behind it.

**YOU ARE PROBABLY NOT THE ONLY SESSION ADDING A COUNTRY RIGHT NOW, AND THAT IS BY DESIGN.** Anita
runs two or three agents at a time on this directory — her words, 2026-09-07: *"im generally
running 2-3 agents at a time to add countries."* So treat all of this as the normal working state
rather than as something to investigate or stop for:

- **`countries.py`, `taxonomy/branches.py`, `spec.md`, `sources.md` and `COMMANDS.txt` will change
  under you**, and your tooling will say so on an edit that applied perfectly well. Make surgical
  edits against unique anchors — they fail loudly if the anchor moved, which is what you want —
  and re-read before any edit that depends on surrounding context. **Never rewrite a shared file
  wholesale**; that is the only move here that actually destroys someone's work.
- **A country you did not add can appear in `COUNTRIES` mid-session.** That is why the build list
  is derived and never pasted (see below, and `sources.md`'s note on the generator).
- **§9-series letters in `sources.md` are claimed first-come.** Check the existing headings
  immediately before you write one; §9ag and §9ah were taken between drafting and appending on
  2026-09-07, and there are already two §9ac's.
- **The taxonomy is shared, so another session's node rename can invalidate YOUR already-scattered
  dots.** `buffers.py`'s `WARNING: n node(s) not in religions.json` is the only signal, and it
  means "some country needs re-scattering", not "re-run `build_tree.py`".

**SO CLAIM THE COUNTRY BEFORE YOU START, AND CHECK BEFORE YOU WRITE.** `queue.md` is the
candidate list and `tools/claim.py` is a one-file-per-country advisory lock:

    python tools/claim.py                       # what is claimed, what is free
    python tools/claim.py take <cc> --id <sid> --note "what you are doing"
    python tools/claim.py done <cc> --id <sid>

`<sid>` is your session id — the last component of the scratchpad path in your system prompt.
Claims are one file each under `data/claims/` created with `O_CREAT|O_EXCL`, so a race has
exactly one winner; a shared list everyone edits would be `index.html` again. It is advisory
and cannot stop anyone. **The thing that actually works is the habit: before `Write` to any
`sources/<cc>*`, `taxonomy/<cc>*` or `data/` path, look at whether it already exists.**
Added 2026-09-08, after a session spent an hour building Peru that another had already
finished and overwrote its `sources/pe.py` with a `Write` to a path it had not checked
(recovered from Claude Code's `~/.claude/file-history/`, which snapshots a file immediately
before it is overwritten — worth knowing).

None of this needs raising with Anita. Getting a fright and stopping costs more than the collision
would have.

**Read §14 before starting a country whose state does not publish religion, or whose religious
minorities are persecuted.** Whether a country should be drawn at all, and at what resolution, is prior
to every technical step below — and §14 asks you to raise it with Anita rather than settle it yourself
([[feedback_flag_ethics_for_discussion]]).

### No country is closed for good — a negative is a record of what was tried, not a verdict

**Anita, 2026-09-07:** *"on many occasions on a first pass we have ruled a country out as not
possible. but then it turns out it was possible when we looked a little harder for more data. i
think its totally fine to give up / just do a coarse first search, and we should definitely mark
what we tried. but one dead end is not the end, and i will probably eventually go back and try
some of the ones we said were dead. nothing is truly dead."*

This is a standing instruction about how to write a negative, and it has two halves.

**Giving up early is allowed and expected.** A coarse first pass that stops is the right call —
there are more countries than time, and a shallow sweep that closes ten countries badly is worth
more than one closed perfectly. **Do not treat "I could not finish this" as a failure to hide or
apologise for.** Stop, write down where you got to, move on.

**But write the negative so it can be reopened, because it will be.** The record is the
deliverable, not the verdict. Every one of these turned out to be wrong later:

- **Nicaragua** — `sources.md` §11t filtered the oracle to the region, saw the Central American
  mainland listed absent, and never probed an office. §11x found INIDE running an open census
  microdata server the whole time, at 153 municipalities, with a category nothing else on this
  map has.
- **Peru** — declined by §11t on *"the prize is four categories"*, read off the oracle. The
  census microdata has **eight** (§11y). The four was what INEI forwarded to UNSD.
- **Nepal** — §11r's backlog row said district and four categories. It is local level (753) and
  ten (§9ao).
- **Zambia** — §11p read one release and recorded "5 categories, do not build". A different
  census has eight (§11w).
- **Mozambique**, **Kosovo**, **Montenegro**, **Bosnia**, **Albania** — all recorded unreachable
  or not found before a later session reached them, several on one guessed URL.

So the shape of a good negative is: **what was asked, of what, on what date, and what came
back** — not "this country is not possible". Specifically:

- **Say which release you read**, not which country you read it about. *"The 2022 report has five
  categories"* is true and reopenable; *"the office publishes shallow categories"* is a claim
  about the office that will be believed and is often false for a different census.
- **Distinguish the four negatives, because they age differently.** *Nobody asked the question*
  (Panama, five censuses, from the microdata dictionary — the most durable). *Asked and never
  published* (Türkiye since 1965 — a §14 object, not a sourcing problem). *Published and not
  found* (the reopenable one — always say what was searched). *Reachable but walled* (the most
  perishable: **re-test a 403 before believing it**, KSH's was gone by the next attempt).
- **Record the route you tried, so the next pass does something different.** A negative that
  does not say what was searched costs the next session the whole search again.
- **Never let a stale note read as a verdict.** Put the date in the sentence. `sources.md` is
  append-only and later sections reverse earlier ones — that is the design, and reversing an
  earlier finding is normal work, not a correction anybody needs to feel bad about.

**And when reopening: prefer a cheaper question over a harder version of the same one.** The
technique that reopened Nicaragua and Peru was not persistence against a wall, it was asking a
different server — see `sources.md` §11x's rule, *ask whether the office runs a REDATAM instance
before reading its PDFs*, and [[reference_redatam_servers]]. §11r's *"reachability is no longer
the binding constraint"* remains true; what changes a negative is usually a new route, not more
force against the old one.

### The shapes of failure that cost the most

Almost everything below is one of five things. If you are debugging and nothing here matches, ask
which of these it is:

1. **A silent drop.** A join, a filter or a mapping quietly removes rows and every remaining total
   still reconciles. Connecticut, the `None` category, the unmapped remainder, `--countries` short
   lists. **Print both sides of every join and assert the count.**
2. **A confident wrong pairing.** A key matches almost everything and pairs some units with the wrong
   polygon. Sri Lanka, Vietnam's two code spaces, Ghana's `TMA`, Serbia's two Palilulas. **A shared
   code is trustworthy only as far up the hierarchy as you have independently verified it.**
   Nicaragua is the sharpest case (§9ay): COD's pcode is `NI` + INIDE's own code and **145 of 153
   match**, but ten municipalities were renumbered between the census and the boundary vintage and
   **five of them collide instead of going missing** — INIDE's `9105` is Waspám and COD's `NI9105`
   is Mulukukú, so a code join moves a 43.6% Moravian border municipality inland and every total
   still reconciles, because a permutation of units preserves every sum.
   **AND THE COUNTER TO IT IS FREE WHENEVER A DRAWN CATEGORY HAS A KNOWN GEOGRAPHY.** Names and
   codes both come from the same two offices and can be wrong together; *the Moravians being on the
   Caribbean coast is a fact about Nicaragua that neither office authored.* So after joining, assert
   the category lands where it belongs — `ni_geo.py` checks the six most Moravian municipios against
   COD's own centroid longitudes, which uses neither of the join keys. Malawi's Anglicans on Likoma,
   Zimbabwe's Vapostori in Mashonaland and Belize's Mennonites in the north can all afford the same
   assertion, and it is one line.
   **BUT THE WITNESS MUST NOT ENCODE THE FACT THE MAP IS BEING BUILT TO DISCOVER, AND PERU IS WHY
   THIS SENTENCE EXISTS** (§9bc). The Moravian check was rewritten for Peru's Adventists as *"the
   most Adventist districts are the Puno altiplano"*, on the well-documented history of the 1898
   Platería mission — **and it fired on a join that was correct.** Three of the ten came out 800 km
   north in the Alto Mayo, which is Peru's *other* Adventist region. The tell, available in advance:
   "the Moravians are one coastline" is a claim that there is **one** cluster, and Nicaragua's data
   said so; "the Adventists are the altiplano" is a claim about **where** the cluster is, which is
   exactly what the map was built to show. When you cannot make the weaker claim, use the form that
   names nowhere at all: **religion shares are spatially smooth**, so correlate each unit's share
   with its k nearest neighbours' and calibrate the threshold against random re-pairings of the same
   shares on every run. Peru gets r=0.75 against a best of 0.11 over 200 shuffles, it is a dozen
   lines, and it cannot be wrong about the country because it asserts nothing about it.
   **AND "JOIN ON NAME" IS NOT THE RULE — "MEASURE BOTH AND SAY WHICH ONE IS CARRYING IT" IS.**
   Nicaragua joins on name because its codes were renumbered; Peru joins on **code** because COD's
   `adm3_pcode` is the census's own ubigeo, 1,870 of 1,872 pairs agree on the name outright, and a
   name join would be the risky one there — Peru has many districts sharing a name across provinces
   and disambiguating them would need the code. Two countries wired four days apart sit on opposite
   answers. Whichever key you use, the other one becomes the check, and the file says so out loud.
3. **A duplicated or missing level.** An extra tier hides inside the drawn one, or a parent's child
   list is short. Serbia's `Grad`, India's towns, Indonesia's regencies. **Only the parent/child sum
   sees either, and it has to be computed on every column.**
4. **A response that is not what it claims.** HTTP 200 with a PNG, a JS alert, a stray byte, the wrong
   file format, a zero-feature read. **Assert size, type and count, never the absence of an
   exception.**
   **AND FOR GEOMETRY, COUNT IS NOT ENOUGH — ASSERT THE MAGNITUDE. Fiji is why** (§9bd). It
   straddles the 180th meridian, and three separate steps produced a file that opened, had exactly
   the right feature count and exactly the right names while being geometrically absurd:
   reprojecting the provinces to EPSG:4326 tore Cakaudrove, Lau and Macuata into **360-degree**
   polygons; **nine of Kontur's own hexes ship torn** across the EPSG:3857 plane, so their
   centroids computed to longitude ~0 and landed in the Atlantic and the Sahara at Fiji's
   latitude; and "fix it by projecting into a Pacific CRS" **does not work, because pyproj does
   not wrap longitude** — it moves the tear somewhere less recognisable. A centroid-in-polygon
   join against any of those silently drops or mispairs units and every total still reconciles.
   **The check that catches all three is two lines: compare the layer's bounding box against how
   big the country actually is.** Fiji is about 5° wide; the first attempt came out 28,670 km
   across. Add it wherever a country is near 180°, near a pole, or spans a UTM zone — and note
   that the fix is arithmetic in degrees (shift negative longitudes +360 and join there), not a
   cleverer projection.
5. **A category that does not mean its label.** India's Annexure, Kenya's "Evangelical", Germany's
   three church-tax boxes. **Nothing inside the data catches this. Read the whole list and check one
   number you already know.**

### The order that avoids wasted effort

1. **Find the data and check it goes deep enough**, before anything else. The killer question is not
   "does this country ask about religion" but **"does it publish the answer at a fine geography"**
   (§3.9). Look at a sample of the actual table. **And ask what instrument produced the category
   list**, because it may not be a classification at all: Germany's three categories are the set of
   corporations that levy church tax, a fact about public law (§3.9a), and where the list comes from a
   register rather than a question its ceiling is fixed and hunting for depth is wasted effort.
   **"Deep enough" is about finding the BEST tier a country offers, never a bar it has to clear —
   §3.9b, and there is no minimum unit count.** Take the finest geography published and the fullest
   category list published, say in `sources/<cc>.md` what the country therefore cannot show, and
   draw it. Guyana is drawn on 10 regions and Georgia on 11.
2. **Ask whether the religion table carries a geographic CODE.** It decides whether the join is a
   lookup or a day's work — see the boundary section. It is the second question, not a later one.
3. **Check the boundaries exist and join**, third. A source with no joinable geography is not a source.
   Do this before writing a normaliser, not after.
4. Then normaliser → taxonomy → `countries.py` → scatter → buffers/tiles → docs.

### Finding the data

**CHECK WHETHER A NEWER CENSUS HAS LANDED SINCE THE QUEUE ROW WAS WRITTEN, BEFORE PARSING THE ONE IT
NAMES.** `queue.md` priced Moldova at 2,804,801 people and 12 categories, which is the **2014** census;
the **2024** one had published final results and was better on every axis but one. Geography: 901 UATs
against 35 raions, because 2014 published religion at raion level only. Coverage: 2014 enumerated
2,804,801 against the office's own estimate of 2,998,235, so about one person in fifteen was never
reached. Non-response: 0.75% against 6.88%. **What a newer census usually loses is category names** —
2024 dropped Moldova's `Iudaism` and Lutheran columns into a residual, so no Jewish dot is drawn in
Bessarabia. Weigh that against the other three rather than assuming either direction, and record the
older figures in `sources/<cc>.md` if they name something the new one does not (§9bv).

**Try a machine-readable endpoint before anything else.**

- **PxWeb.** `https://<host>/api/v1/<lang>/<db>/` returns a JSON tree you can walk. It is the
  Nordic/Baltic standard and Estonia took three minutes against three days of hunting for Slovakia.
  `POST` with `{"query":[...],"response":{"format":"json-stat2"}}` and `"filter":"all","values":["*"]`
  to get everything. **It is not a Nordic thing** — North Macedonia's `makstat.stat.gov.mk` runs it and
  sat listed as "unchecked" for a day while being one request away. Try `/pxweb/api/v1/en/` *and*
  `/api/v1/en/` on any office before concluding anything.
- **json-stat2 is a flat cube, not a table.** One `value` array in row-major order over the dimensions
  in `id`, sizes in `size`. Read it as rows and you will silently transpose the data. Compute strides.
- **Do not search a PxWeb tree with a depth-limited keyword walk.** North Macedonia's religion table is
  five levels down and a bounded walk returned ZERO hits on a database that has it. Worse, a
  *national-only* religion table sits in a sibling folder, so a shallow search finds the wrong one and
  suggests the country publishes religion with no geography. **The same variable routinely appears at
  several geographies in different folders** — enumerate the census branch in full and compare.
- **A PxWeb ASP.NET HTML tree cannot be walked by following links.** Navigation is `__doPostBack` and
  every folder page re-renders the same sibling list, so a link-follower returns nothing and the
  database looks empty. The node ids are in the postback arguments if the API really is gone.
- **A PxWeb 403 can be a CELL LIMIT, not an auth failure.** Ghana's religion table has six dimensions;
  asking for all of them is 663,750 cells and PxWeb answers 403. Dimensions with `elimination: true`
  return their totals when simply left out of the query, which is what was wanted anyway.
- **REDATAM. Ask whether the office runs one BEFORE reading its PDFs** — added 2026-09-07 with
  Nicaragua (`sources.md` §11x, §9ay), and it is the largest single resolution win in this file.
  Many Latin American offices run CELADE's Redatam webserver over their own **census microdata**,
  unauthenticated, and it will run an arbitrary tabulation program. Nicaragua **prints** religion
  at 17 departments and **serves** it at 153 municipios and 2,579 comarcas, from the same website,
  for the same census. Two things to take from a live instance:
  **(a) the variable picker is a free dictionary.** `.../RpWebStats.exe/Frequency?BASE=<base>&ITEM=FREQPOB`
  ships the person-variable list and the geography levels inline in the HTML, so *"does this census
  ask religion, at what grain, in which vintage"* is two GETs — that is how Panama's five censuses
  and Ecuador's six were closed. Note the `<option>` tags are often **unclosed**, so a regex
  expecting `</option>` finds nothing and the page reads as empty.
  **(b) the tabulation itself**, by POSTing `CMDSET=RUNDEF Job / SELECTION ALL / TABLE T / AS
  AREALIST / OF <GEOLEVEL>, <ENTITY>.<VAR>` to `.../RpWebStats.exe/CmdSet?`; the reply names a
  minted temp file to fetch. **A bad program answers `Tabla vacía` with HTTP 200** — §5a again, and
  the shape that writes an empty csv. Two traps worth knowing: a shared host (`prod.redatam.org`)
  does **not** namespace `BASE=` by country, so asking under the wrong CGI directory returns another
  country's census with a 200 and no warning; and a deployment can be a CELADE **demo stub**
  (`redatam.one.gob.do` serves only `NMIROLD`, "Nueva Miranda"). See [[reference_redatam_servers]].
- **SDMX gives you the CODELISTS, which may be the thing you actually need.**
  `/api/structure/<flow>/<version>` returned KSH's category labels in two languages and its full
  geography hierarchy — parent chain included, so a settlement's county came off the source rather than
  out of a boundary file. Recognise the shape: `dataflows`, `structure`, `version` in any combination
  means SDMX. **Where a PxWeb tree has to be walked, an SDMX catalogue can be grepped** — Lithuania's
  is one file listing 9,521 dataflows with bilingual names, so `religi|tikyb` enumerates everything the
  office holds in a single pass.
- **AND ONE PART OF AN API ANSWERING JSON SAYS NOTHING ABOUT THE NEXT.** Lithuania's catalogue is
  XML-only — `/rest_json/dataflow/` 404s while `/rest_xml/dataflow/` returns 7.4 MB — while the *data*
  is served as either. Try the other representation of the same endpoint before concluding it does not
  exist.

**A PROBE'S STATUS IS A CLAIM ABOUT A HOSTNAME, AND OFFICES GET RENAMED — FOUND 2026-09-07 WITH
NEPAL.** `sources.md` §12 already says a lead's status is a claim about a *date*; this is the same
mistake in space rather than time. Nepal's `censusnepal.cbs.gov.np` answers **200 with 18 bytes of
`<p>...working</p>`**, and `cbs.gov.np` answers 200 with nothing at all. Two separate sweeps recorded
Nepal as reachable-but-thin on that evidence and moved on. **The Central Bureau of Statistics had
become the National Statistics Office**: `nsonepal.gov.np` is alive, its census results are on
`censusresults.nsonepal.gov.np`, and what was there is 753 local levels and ten religions — the finest
counting geography of any large Asian country on this map.

Nothing redirected. **A placeholder is worse than a 404** for exactly this reason: a probe records it
as *up*, so the country reads as "checked, and thin" rather than "not found yet". When an office's host
answers with a holding page, or with an empty 200, **check whether the agency still exists under that
name** before writing the country down — and follow the *organisation's* current site to its data host
rather than trusting a hostname from an old note.

**A wall is a fact about a host and a path, not about a country.** Four shapes of the same mistake,
and every one of them nearly wrote off a country that was reachable:

- **Test the specific host, then its siblings.** Lithuania's `osp.stat.gov.lt` returns Cloudflare's
  challenge — that is the human web UI; the statistics are on **`osp-rs.stat.gov.lt`**, a plain SDMX
  endpoint with no protection at all. Taiwan is the same with no tidy explanation:
  `religion.moi.gov.tw` and `statdb.dgbas.gov.tw` time out on connect over both IPv4 and IPv6 while
  `www.moi.gov.tw`, `ws.moi.gov.tw`, `segis.moi.gov.tw` and `data.gov.tw` all answer normally. There is
  no rule to it — not UI vs API, not old vs new.
- **The API prefix is not always under the UI prefix, and the wrong one can return 500 rather than
  404.** Ghana's StatsBank interface is at `/pxweb/en/…`, so `/pxweb/api/v1/en/` is the obvious API
  root; it returns **HTTP 500 on a PX-Web ASP.NET error page**, which reads as "the API exists and is
  broken". It is at **`/api/v1/en/`**, with no `/pxweb`. **A 500 rendered in an application's own error
  template is evidence about the ROUTE, not the feature** — it means the request reached the app and
  the app did not recognise it, which is what a 404 says, dressed to read the opposite way. Equally, a
  404 on a wrong path under the right host reads exactly like absence (Lithuania's probe missed by one
  path segment: `/rest_xml/` instead of `/rest_xml/dataflow/`).
- **Re-test a 403 before believing it.** KSH's was gone by the time anyone tried again, and a stale
  "blocked" note reads as a dead end for months.
- **But bot protection is a stop sign, not a puzzle.** census2021.bg returns 403 to scripted clients.
  Do not iterate on headers — hand the URL to Anita, who will fetch it in a browser
  ([[feedback_long_scripts]] is the same shape of hand-off). Same for anything Cloudflare-interstitial;
  the Philippines came from the Wayback `id_` endpoint instead. **And a reactive wall SPREADS while
  being probed** — see the KOSIS entry below — so pushing harder costs access rather than gaining it.
  That is the practical reason this is a rule.

**A statistical office often runs two sites, and they are found by different searches.**

- **A shiny data portal is often a shell.** `podaci.dzs.hr`, `data.gov.sk` and `data.stat.gov.rs` all
  return a JavaScript app, not data. The real files are usually on the *old* site (`dzs.gov.hr`) linked
  from a "Popis 2021" page.
- **A dissemination database is not a census results portal.** `data.stat.gov.rs` was written off as an
  SPA shell, correctly, and it is the wrong site: Serbia's census results live on
  **`popis2022.stat.gov.rs`**, a separate host listing every published table as a direct `.xlsx` under
  `/media/<id>/`. Religion by municipality is 83 KB of it and needed no API. **Before concluding a
  country is blocked, look for `popis`/`census`/`recensement` + the year as a HOSTNAME**, not only as a
  path on the office's main domain.
- **"A JavaScript app with no data endpoint" is a claim about the searcher, not the site**
  ([[reference_spa_hidden_apis]]). Two cheap tests. **(1) Compare the 404s.** KSH's `/api/anything`
  returns 88 bytes of `{"timestamp":…,"status":404,"path":…}` while every other unknown path returns
  the same 2,180-byte HTML shell — a Spring Boot error body IS a live API namespace, and content-type
  plus length distinguishes a router from a catch-all. **(2) Grep the bundle for `/api`.** KSH's four
  routes were plain string literals in `app.js`, which one session had already concluded contained
  nothing. `podaci.dzs.hr`, `data.gov.sk` and `data.stat.gov.rs` have never had this done to them.
  **(3) SEARCH THE WEB FOR THE HOST'S OWN API PATHS**, added 2026-09-07 with Israel, where both tests
  above FAILED: `census.cbs.gov.il/api/<anything>` returns the same 2,804-byte shell as every other
  unknown path, so the 404 comparison says there is no router, and the site is Astro — the bundle is a
  page component with no endpoint literals in it. The working route was found because a search engine
  had indexed one `…/api/get-pdf?…` URL. The site never links to `/api/` from anywhere, so nothing on
  it could have led there. **One web query for `<host> api` costs nothing and is now the third test.**
- **A SITE BUILT ON htmx OR ALPINE KEEPS ITS WHOLE API SURFACE IN THE DOM, AND THAT IS FASTER THAN
  GREPPING A BUNDLE.** Israel's per-area census data is addressed by opaque 7-hex-character IDs with no
  relation to any CBS code, sparse enough to rule out enumeration — so the endpoint alone was useless
  without the ID mapping. The mapping is the site's own autocomplete: the search input carries
  `hx-get="/en/partials/search/area"`, and that endpoint returns `data-id`/`data-search`/`data-type`
  for **every geographic unit in the country**. **The diagnostic generalises past htmx**: load the page
  in headless Chrome and dump every element carrying an `hx-*` (or `x-on:`, `@click`, `wire:`)
  attribute. It took one CDP call after two days of the API looking absent, because those frameworks
  put the URL in an HTML attribute rather than in JavaScript, where every search had been looking.
- **A WordPress statistical office has a search API and it beats its search box.**
  `GET /wp-json/wp/v2/search?search=<term>&per_page=50` returns title + URL for every post. Vietnam's
  two census volumes came back as the top two hits of one request after the site's own search and two
  web searches had missed the older one. **Search the publication's NAME, not the variable** —
  `search=kết quả toàn bộ` found everything and `search=tôn giáo` found nothing. Recognise the shape
  from `/wp-content/` in any asset URL.
- If there is no API at all, the census results are usually a handful of XLSX behind a "results" page.
  Fetch the page and regex out `href="...xlsx"` **with the link text**, because offices routinely name
  the files `Tabel-2.04.xlsx` and put the title elsewhere (Romania).

**Two things to check about the publication itself before writing it off.**

- **Fetch the table in more than one language when the join and the taxonomy want different ones.**
  North Macedonia needs BOTH: the English edition for the category labels the taxonomy keys on, the
  Macedonian for the Cyrillic municipality names GISCO carries. Neither edition alone builds the
  country, and the English one alone silently makes the join impossible.
- **CHECK THE PREVIOUS CENSUS BEFORE CONCLUDING A COUNTRY PUBLISHES RELIGION WITH NO GEOGRAPHY.**
  §3.9's trade between category depth and spatial depth has always been a choice made *inside* one
  publication. It can also be made **across censuses**: Vietnam's 2019 *Kết quả toàn bộ* gives religion
  **one page, national**, while the **2009** volume of the same name, same office, same series, gives
  religion **32 pages by province**. Nothing in the newer volume says the older one was finer, and
  every secondary compilation cites the newer one. The check is one file. **And it usually converts the
  country into §3.4** — old geography, new totals — rather than into a refusal.

**Two API-shaped traps that produce plausible wrong data rather than an error.**

- **An id in a response and an id in the URL that fetched it can be different spaces that overlap.**
  BPS's SP2010 endpoint returns Aceh with `id_wilayah: "1675"`. Asking that endpoint for `wid=1675`
  returns **Kabupaten Merangin, in Jambi**: HTTP 200, well-formed, a real unit, and the wrong one — and
  `1674`, `1675` and `1676` all return it, so the space is not injective either. Nothing raises and no
  total is wrong, because every number returned is genuine; only the *unit* is not the one asked for,
  which is the one thing no reconciliation downstream can see. **Never feed an id from a payload back
  into a URL unless the source says they are the same space.** Enumerate the URL parameter
  positionally and read the unit's identity back out of the response.
- **A parameter that looks like geography may be format, and file size cannot tell you which.** The
  same endpoint's trailing path segment reads exactly like an admin level: `…/0/2` returns 133 KB and
  `…/0/3` returns 5.4 MB, so `3` is obviously the finer geography. It is not — `2` is a **PDF export**
  of the national table and `3` is the **same national table as JSON**. A finer geography and a more
  verbose serialisation both predict a bigger file. **Look at what a response *is* — magic bytes,
  content-type, the first line — before drawing any conclusion about what it *contains*.**
- **A dead statistics office is not a dead census, and the volumes are as likely to be filed under the
  ministry that paid for them — found with Eswatini 2026-09-08 (§9bq).** §11w closed Eswatini as *"no
  reachable host"* on evidence that is all still true: `eswatinistats.org.sz` resolves and times out on
  both ports with any User-Agent, `swazistats.org.sz` does not resolve, and the Wayback CDX has 61
  captures of the first hostname and **not one PDF**. There is no route through the office and there
  never was one. The 2017 census volumes are Joomla articles on `www.gov.sz` under
  **`/images/FinanceDocuments/`**, the finance ministry's upload folder, and the Census Atlas is under
  `/images/planningministry/`. No amount of probing the statistics office would have reached either.
  **So when an office is dead, sweep the WHOLE government domain rather than the office's hostname**:
  `web.archive.org/cdx/search/cdx?url=<gov domain>&matchType=domain&limit=60000&fl=original` is one
  GET, grep it locally for `census`, and fetch the last good capture of whatever article id turns up
  (the `id_` suffix gets the raw body). Ten minutes end to end, and it recovered the highest-ranked
  undrawn country in Africa. Note the article listing the files 404s today while **every file it links
  still serves from the live host**, so a dead index is not a dead directory.
- **Two CDX failure modes that both read as a negative result.** It **refuses port 80** —
  `http://web.archive.org/cdx/...` is connection-refused, which reads as the archive being down rather
  than as a scheme problem — and it answers a `filter=` regex it does not like with **HTTP 500 and an
  empty body**, which reads as zero matches. Ask for the unfiltered list over https and grep locally;
  it is one request either way.
- **A bot wall can answer HTTP 200, and it has already cost this project a country.**
  `statssa.gov.za` and `cs2016.statssa.gov.za` sit behind Incapsula/Imperva: **curl gets a 200
  carrying a 212-byte `_Incapsula_Resource` stub** instead of content, and `www.statssa.gov.za`
  **fails cert verification (exit 60)** before answering at all. Both read as a dead or empty host.
  §11b concluded StatsSA *"is the blocker"* and detoured to a DataFirst account — while the census
  religion table it wanted was openly downloadable the whole time (§11ag). WebFetch walks straight
  through and saves the real PDF. Same animal as `[[reference_dead_stats_office]]`'s 418.
  **A 200 with a sub-kilobyte body, or a TLS failure, is a wall and not an answer: refetch with a
  browser-shaped client before concluding anything about what an office publishes.**
- **A survey variable can be revised under the same name.** ESS carries `rlgdnanl`, `rlgdnase`,
  `rlgdnaua`, `rlgdnapl`/`rlgdnbpl`, `rlgdnask`/`rlgdnbsk` — `a`/`b` revisions with **different
  category lists**, added when a country's denomination card changed. **Pooling on the bare name
  silently drops the later rounds**: no error, just a smaller n and an older card. Match on the
  prefix and assert the round count you expected. §11ad's flattened-value-label trap in another
  costume, and the general form is that **a pooled multi-wave file will not tell you when the
  instrument changed underneath it; only an assertion will.**
- **AN OFFICE'S LANGUAGE VERSIONS ARE NOT TRANSLATIONS OF ONE SITE. THEY ARE SEPARATE TREES THAT
  CAN DIFFER IN WHAT EXISTS AT ALL**, and Armenia cost §11o the whole country on it (§9bx).
  `armstat.am/en/?nid=945` through `?nid=957` are the eleven 2022 census marz volumes, and every
  one of them is a bare `<h1>` over the sentence *"Information is not available in English"*. The
  same eleven nids under `/am/` each carry nine section archives, religion included. **The English
  tree there is a strict subset, and the part missing from it is the geography** — which is the
  worst thing to lose, because the national volume IS in English and looks like the whole
  publication. Always re-check a negative in the national language before recording it, and say in
  the record which tree it was taken from.
- **A results page can be an image map whose links all go to the same file.** Armstat's
  `?nid=944` renders a GIF of Armenia with eleven `<area>` polygons over it and **all eleven point
  at the national volume**, so clicking any province returns the same download and the page reads
  as *"there is only a national volume"*. The per-province pages existed the whole time, one nid
  each, reachable only from the left navigation. **Enumerate a site's nav tree, not its landing
  pages**: the same shape as §9bu's per-district booklets and §11ag's bot wall, and all three were
  closures overturned on 2026-09-08 without the office having published anything new.

### Estimating the work, and downloading

**ASK HOW MANY PAGES THE TABLE OCCUPIES, AND NOTHING ELSE.** The rule written one morning was "the
useful countries are the ones with a dissemination platform rather than a report series", and it was
wrong by that afternoon. Ghana had a PxWeb API and took an afternoon; Kenya had a **498-page PDF** and
took an afternoon too, because the whole religion table is **one page**. Guyana is the extreme case:
the Bureau of Statistics runs no dissemination platform *at all* — just a list of PDFs on a WordPress
page — and it still took an afternoon, because the religion cross-tabulation is one page of a 66-page
compendium. A one-page county table parses in an hour whatever it is wrapped in; a per-district table
spread over eleven regional volumes does not, whatever portal fronts it. **The size of the table is the
predictor, and it is visible from the list of tables before anything is downloaded.**

- **When a report is dense with one geography and your variable is coarser, read what the other tables
  do.** Kenya's Volume IV has forty-odd tables and *every one except religion* is published "by County
  and Sub-County". That converts "I could not find a sub-county religion table" into "KNBS did not
  publish one", which is a different fact and a much more useful one — it ends the search instead of
  leaving it open, and it says the ceiling is editorial rather than technical. (The questionnaire annex
  in the same volume showed the census captured six geographic levels, so the data exists and is
  withheld.)
- **A "code lists" or "classification" page is evidence about SOME census, not necessarily the current
  one.** StatsSA maintains a religion code-list page whose depth (hundreds of bodies) made South Africa
  look like the best source in Africa. Those are the 2001 lists; Census 2022's actual variable has
  **twelve** categories with `Christianity` as one undivided cell holding 85% of the country. **Check
  the variable in the current sample, not the classification the office happens to still host.**

**HTTP 200 is not a download, and it keeps finding new disguises.** Always assert size *and* type
(`zipfile.is_zipfile`, sheet names present, magic bytes) after fetching:

- a truncated file (Czechia); an SPA shell; and Maa-amet's `linnaosa_shp.zip`, which returns **200 with
  a 282-byte PNG of an error message**.
- **A 200 can carry a JavaScript dialog.** KOSIS's data endpoints return **status 200, content-type
  `text/html`, and a 335-byte body whose entire content is
  `alert("비정상적인 서비스 이용으로 접근이 차단되었습니다")`** — access blocked, abnormal use.
  `raise_for_status()` passes; a size check passes if the threshold is low. Two things generalise. **A
  bot wall can be per-endpoint on a host that is otherwise wide open** — KOSIS's metadata endpoint
  served 291 KB of table structure, unauthenticated, throughout, which is where the table id, its 327
  geography items and all 12 category labels came from; only the endpoints returning *data* refuse. So
  "the site is blocked" and "the data is blocked" are different findings. **And the wall SPREAD while
  being probed**: the download endpoints refused first and the grid endpoint refused afterwards.
- **A WALL YOU BUILT YOURSELF LOOKS EXACTLY LIKE ONE THEY BUILT FOR YOU** — 2026-09-07, Israel, and
  it is the other side of KOSIS. There is no bulk file for the 2022 census: religion is published per
  area through a dashboard, one request per unit, so drawing 3,236 units means ~4,600 calls against a
  small national statistical office. Run at 0.15 s intervals with a second process pointed at the same
  host, it took about thirty minutes to get **every connection reset, the static home page included** —
  an IP-level block, not per-endpoint, and still up an hour later. It cleared overnight. Three rules,
  and the third is the one that actually cost a run:
  1. **Count the requests before starting.** The number was knowable from the unit list and nobody
     looked at it. A four-figure request count is a design decision, not an implementation detail.
  2. **Never point a second process at a host a harvest is already walking.** One extra connection is
     what turned a working run into a block.
  3. **A cache written only at the end is a cache that never gets written.** The first attempt held
     everything in memory and flushed every 200 units; killed at thirty minutes, it had written
     nothing. Flush small, flush through a temp file and `os.replace` ([[reference_wb_truncates]]), and
     make resumption the ordinary path rather than the recovery path — on a run this long it *will* be
     interrupted, so a resumable fetch is a correctness property and not a convenience.
- **AND THE RETRY LAYER IS WHERE A LONG FETCH ACTUALLY DIES, NOT THE PARSER.** Both failures on
  Israel's second attempt were in the plumbing and neither was the source's fault:
  - **`SystemExit` is not an `Exception`.** The shell-page guard raised `SystemExit`, which
    `except Exception` around the per-unit call cannot catch, so **one intermittent bad response killed
    a two-hour run at unit 1,699.** A guard that fires on transient junk must raise something the
    caller's handler catches, and it belongs *inside* the fetch helper where the retry can act on it.
  - **A one-way backoff ratchet saturates and stays there.** Delay was multiplied by 1.5 on every error
    and never reduced. Against a host emitting a transient shell page on ~1.5% of requests it pinned
    itself at the 3 s cap within minutes, which turns a two-hour run into a six-hour one. **Decay back
    towards the base on sustained success** — 1.5x up per error against 0.98x down per success is far
    slower to recover than to back off, which is the asymmetry you want.
- **A content-type is a claim about the body, not about its first byte.** `mozdata.ine.gov.mz`'s NADA
  API answers `application/json` and emits a stray `n` before the JSON on *every* endpoint, so
  `r.json()` dies with "Expecting value: line 1 column 1". That reads as "this endpoint is broken" and
  it is not — the payload behind it is complete. **Find the first `{"` and parse from there**, and
  treat a decode failure at offset 0 as a framing problem to look at rather than a verdict.
- **An extension is a claim about nothing at all: check the magic bytes.** Every INSTAT table is served
  as `application/vnd.ms-excel` with a `.xls` extension and is actually **SpreadsheetML 2003 XML**,
  starting `<?xml`. Pandas refuses it with *"Excel file format cannot be determined, you must specify
  an engine manually"*, which reads as a missing dependency and sends you installing `xlrd` — no engine
  will ever open it, and `xml.etree` opens it in four lines. `PK\x03\x04` is a real xlsx,
  `\xd0\xcf\x11\xe0` a real legacy xls, `<?xm` is neither. **When parsing SpreadsheetML, honour
  `ss:Index`** — it omits empty cells rather than emitting blanks, so a positional read of a row with a
  gap shifts every later column left.
- **AND THE TRUNCATION CAN BE ONE CELL OF A PERFECT FILE, AND IT PARSES AS A NUMBER.** Guyana's PDF is
  intact, every other column is right, and the rightmost column of Table 2.19 simply overflows its cell
  in the text layer, so `38,962` arrives as **`38,96`** and `746,955` as `746,9`. **A clipped number is
  a valid number.** Nothing raises and the error is one or two digits in a figure nobody has
  independently in mind. The fix generalises past "do not read that column": recompute the derived
  figure from the parts, then **assert that the source's clipped string is a PREFIX of your sum**. That
  turns the defect into a check — it fails on a misparse, and it keeps passing if the office ever
  re-renders the file. *Where a source's own derived figure is unusable, do not merely drop it; assert
  the relationship it still has to the figure you computed.*

**Two file-reading traps that return success and no data.**

- **A read that succeeds is not a read that returned data.** COD ships Chile as a 92 MB geodatabase.
  On GDAL 3.8.5's OpenFileGDB driver it **opens, lists all six layers with the right geometry types,
  reports `crs=EPSG:4326`, and returns ZERO features from every one of them** — including `admin0`,
  which is one polygon. No exception, no warning. Nothing downstream would have noticed until the join
  came back empty, and the symptom would have pointed at the join. **Assert the feature count after
  every `read_file`, not the absence of an exception.**
- **And the zero-feature read is the ENGINE, not the format.** COD's `idn_admin_boundaries.gdb` returns
  **522 features with `engine="fiona"` and ZERO with `engine="pyogrio"`** — same file, same driver,
  same machine, one call apart. pyogrio reports `EPSG:4326`, a `geometry`-only column list and no
  error, which is precisely Chile's symptom; and **pyogrio is geopandas' default whenever it is
  installed**, so the failing path is the one you get without asking. That reframes Chile: the
  geodatabase was probably readable all along. This is the TLS test below, one layer down — **all
  readers failing means the file, one reader failing means the reader** — and
  `read_file(..., engine="fiona")` costs nothing to try.

**A TLS failure can be the server's fault, or yours, with a near-identical error and opposite fixes.**
`stat.gov.pl` omits its intermediate certificate, so curl, requests and certifi all fail identically
with "unable to get local issuer certificate" — turning verification off for that one named host *and
validating the bytes structurally instead* is the honest fix, said out loud in the script and in
COMMANDS.txt. But `urllib.request` cannot reach `ksh.hu` on this machine — "self signed certificate in
certificate chain" — while `curl` and `requests` verify it fine, and that is a local trust store.
**The distinguishing test is one line: try a second client. All clients failing means the server; one
failing means you.** Reaching for the `stat.gov.pl` fix on the second case disables verification to
route around a problem that is not there.

### Parsing the table

**Sentinels, missing values and the things that look like numbers.**

- **Look for in-band sentinels in numeric columns.** New Zealand's `-999` "Confidential", Romania's `*`
  (suppressed) and `-` (true zero). Read cells one at a time, classify them, and **raise on anything
  unrecognised** so a new sentinel cannot appear silently. **Never `errors="coerce"` a count column**:
  it turns suppression into NaN and the people vanish. Where a flag distinguishes two meanings, resolve
  it on its TEXT, not its index (§3.8).
- **A CATEGORY CAN BE NAMED `None`, AND PANDAS WILL DELETE IT.** PSA's cell for no religion is the
  literal string `None`, and default `read_csv` parsing turns it into `NaN`. It then fails to resolve
  in the taxonomy mapping, and every reader that drops unresolved rows — which is all of them,
  correctly — removes 43,931 people **with no error, no warning and no count anywhere**. The category
  most likely to be hit is precisely the one a religion map cares about being honest regarding.
  **Read normalised files with `keep_default_na=False, na_values=[""]`**, and treat `None`, `NA`,
  `N/A`, `NaN`, `null` and `-` as category names a source is entitled to use. Guyana hit it a day
  later, and **where it was still invisible is the part worth having**: `gy.py`'s reconciliation was
  exact and `check_mapping.py` reported 746,955 people on 13 nodes, because both read the normalised
  CSV correctly — the loss happened in `countries.py`'s per-country `_counts()`, the one reader without
  the flags. **The rule has to be applied at every `read_csv` of a normalised file, not at the ones
  that have a checker attached**, because the checkers are precisely the readers that will keep telling
  you it is fine. **THIRD SIGHTING, 2026-09-07: ZIMSTAT's no-religion cell is also the literal string
  `None`, at 1,255,578 people and 8.3% of Zimbabwe** — so this is not a Philippine quirk, it is what a
  census office writes when its category is "none" and the answer must not be blank. Three of the
  three occurrences have been the *no religion* row. **Stop relying on the flags being remembered and
  ASSERT THE CATEGORY IS STILL THERE after reading** (`if "None" not in set(df["source_category"]):
  raise`), which is one line and survives an edit that drops the flags.
- **Excel type inference differs between two files of one release, and within one column of one
  sheet.** India's state files store codes as text (`"00"`) and the Appendix stores the same codes as
  numbers, so `str(cell)` yields `"00"` and `"0"`; India's own row became a 36th state and the whole
  tail doubled. **Put every code through one zero-padding helper at the point of reading.** Germany's
  Sonderauswertung stores some counts as numbers and some as text in the same column, and an
  `isinstance(v, (int, float))` filter — the natural way to skip a sentinel — silently dropped
  **2,228,001 people**, with every national total still plausible because the shortfall landed in the
  largest category. **Classify every cell through one function that RAISES on anything it does not
  recognise; never filter numeric cells by type.**
- **Watch for a percentage twin beside every count column.** Croatia's sheet 2 is `Katolici` in column
  7 and `Katolici, %` in column 8, for all twelve categories. Taking the wrong one of each pair gives a
  map where every unit holds about 100 people and nothing else complains. **List the count columns
  explicitly rather than striding.**
- **A percentage column is not always count ÷ population, and the difference can be deliberate.**
  Germany's disclosure method perturbs the count and then *adjusts the published share* where the
  perturbed count would give an implausible percentage — Ammeldingen an der Our is 18 people with 20
  Catholics, published as 100.0%. 75 cells disagree by over 0.6pp and every one is a Gemeinde of 9–122
  people. **Assert the residual in the units the method works in.** Converting the disagreement back
  into PEOPLE bounds it at 3.46; a tolerance in percentage points either passes everything or fails the
  villages, and neither would catch a percentage column read as a count — which is the thing the check
  exists for, and which would be wrong by hundreds of thousands in every large city.

**Structure, levels and universes.**

- **Universe rows are not categories.** Every source has some nest of
  total ⊃ answered ⊃ affiliated ⊃ the religions, and drawing an intermediate one doubles everything
  below it. Put them in `EXCLUDED` with a sentence on what they are.
- **A SOURCE THAT PUBLISHES ONLY PERCENTAGES IS NOT THEREBY A §3.4 CASE — LOOK FOR THE DENOMINATOR IN
  THE SAME PUBLICATION FIRST.** Benin was recorded for a day as needing its commune totals joined from
  a second document, which prices it as a cross-vintage rescale with everything that implies for §7a's
  tier. **Tableau 2 of the same booklet prints the population of every commune in Tableau 8.** So there
  is no join, each booklet is self-contained, and `count = published share × published total` is
  `measured` rather than `derived`: nothing is carried from a coarser level and nothing is fitted, and
  what the percentage costs is *precision*, which is computable — one decimal on a share is ±0.05% of
  the unit, ±34 people in a 68,000-person commune. **A report written for a prefect almost always
  prints the population first; it is the table before the one you came for.** Say the precision bound
  out loud, and only reach for §3.4 when the denominator genuinely is in another document or another
  year.
- **Do not assume every sheet in one workbook has the same shape.** Poland's TABL.2/6/7 are flat and
  TABL.5 carries the full 7-level classification; summing it the same way counts the Latin rite four
  times. Where the office publishes a depth column, use it.
- **A nested GEOGRAPHY can hide a second universe, and it is harder to see than a nested category.**
  India's C-01 puts state, district, sub-district and town in one column set, distinguished only by
  which code is non-zero — and **town rows are urban-only subsets of the sub-district above them**, so
  summing the file as delivered counts urban India twice. They happen to carry only `Urban` and never
  `Total`, which makes the obvious filter work by luck. **Assert the property; do not rely on it.**
- **AND "FIND THE LEVELS" IS NOT FINISHED WHEN YOU HAVE FOUND THE BOTTOM ONE.** India's and Hungary's
  extra levels sit *above* the drawn tier, so keeping the finest one is the fix. Serbia's sixth level
  sits *inside* it: `Grad Niš`, `Grad Požarevac`, `Grad Užice` and `Grad Vranje` are municipality-level
  rows that are parents of their own city municipalities, **462,527 people counted twice at exactly the
  level you would draw**. There is no structural marker of any kind — after Niš's five city
  municipalities the next row is an ordinary municipality of the same oblast, same column, same indent
  — and every national, regional and oblast total reconciles either way, because the duplication never
  leaves the tier. The `Grad ` prefix is a name, not a field. **The test is arithmetic and it doubles
  as a parse check**: a parent's children are the consecutive rows whose totals sum to it *exactly*,
  and the code refuses to continue if they do not. Once the level count is settled, **assert the count
  of rows at EVERY level, not just the drawn one.**
- **A source can publish a unit as EMPTY, and that is not the same as omitting it.** `Регион Косовo и
  Метохија` is in RZS's sheet with `...` in every cell because the 2022 census did not enumerate it. It
  carries no sex breakdown either, so the natural row filter — keep the rows marked as totals — drops
  it without a word. **Keep such a row and assert on it**: *exactly* one all-empty unit, at the level
  you expect. The source is telling you it did not measure somewhere, which is worth more than silence.
- **A PARENT'S CHILD LISTING CAN BE INCOMPLETE WHILE EVERY ROW IN IT IS CORRECT.** Ten of BPS's 33
  province responses omit between one and five of their own regencies — 16 units and 2,674,311 people —
  with **no gap in the sequence, no marker, no error, and no change to any figure that is present**.
  Sumatera Utara returns 31 consecutive-looking rows and is simply missing Pematangsiantar and
  Padangsidimpuan. This is Serbia's lesson from the opposite direction: there an extra level hid INSIDE
  the drawn tier and inflated it, here units are missing FROM the drawn tier and deflate it, and in
  both cases every total that does not cross the boundary reconciles perfectly. **Only the parent/child
  sum sees either. Compute it for every parent, always, even when the child list looks obviously
  complete — especially then, because "obviously complete" is what a contiguous run of correct rows
  looks like.**
- **AND WHERE THE SOURCE PRINTS LEVELS SIDE BY SIDE RATHER THAN NESTED, RECONSTRUCT THE COMPOSITION AND
  LET THE ARITHMETIC VERIFY IT.** The rule above assumes the child list is *in* the file. Vietnam's
  census prints the nation, then all six socio-economic regions, then all 63 provinces — three flat
  blocks, in code order, with no marker of which province belongs to which region — so the parent/child
  check has no input and the natural move is to skip it. **Write the composition out from the published
  standard instead.** It is a transcription and therefore a risk, and that is the point: the check
  requires every region to equal the sum of its provinces in *every* category, so one province in the
  wrong region breaks 28 equations at once. **A hand-written mapping that reconciles to the person is
  evidence; one that does not is a loud failure rather than a regional map that is quietly wrong.**
- **Recovering a missing child: the residual is exact when the missing children are CONTIGUOUS.** Two
  passes learned this. First, Indonesia's omitted units are reachable one level down, so the obvious
  fix is to rebuild each from its own children — and for two of them that listing is *also* incomplete:
  one sums to 193,661 against a true 234,021, the other to 281,162 against 290,142. **Both UNDERSTATE,
  which is worse than missing**, because an understated unit still draws, at a plausible size, and
  nothing looks wrong. So **where a parent is missing children, prefer the parent's own residual** —
  its row minus the children it did list — over summing grandchildren, which is the fallback and must
  then be checked against that residual. Second, the condition is **not** that exactly one child is
  missing: it is the SHAPE. Kalimantan Utara's territory, 524,656 people, was five missing regencies
  and was written off as an unrecoverable 0.22%, leaving **a visible empty hole on the map**, found by
  looking at the render rather than at any number. The residual was exact all along — it sums to the
  published total to the person, is non-negative in every category, and matches the successor
  province's documented population — **because those five were carved wholly out of one province, so
  the leftover is a single contiguous block with a real geography and can be drawn as one coarse
  unit.** A residual scattered over unrelated places has no shape and genuinely cannot be drawn.
  **THE GENERAL FORM, FOR ANY COUNTRY THAT HAS REDISTRICTED SINCE ITS CENSUS:** a source re-based onto
  a later geography loses precisely the units that MOVED, and those are almost always contiguous,
  because that is what a boundary change is. Their old parent's row still contains them. So before
  recording such a loss, ask whether the orphaned territory is one block. (Note also that the first
  pass had *already measured* the gap, on the total, and discarded it; re-deriving it per category cost
  nothing and closed it.)
- **RECONCILING ON THE TOTAL IS NOT RECONCILING, and it is the cheapest of these rules to get wrong.**
  Deciding which Indonesian tier to draw meant testing whether each regency's kecamatan sum to it, and
  the obvious test is the one on the population total. **Nduga (9429) in Papua publishes eight kecamatan
  that carry a `Total` row and NO religion categories at all.** The Totals sum to the regency exactly,
  so a Total-only test calls the unit complete and promotes it to the fine tier — where its 79,053
  Kristen become 79,053 people with no religion, a unit drawn on the map with nothing in it. One
  regency in 492, invisible in every national figure, and the map would simply have had a hole in
  Papua. **Test the parent/child identity on EVERY column, not on the one that looks like the sum**; a
  total can reconcile because both sides are complete or because both sides are equally empty, and only
  the categories tell you which.

**Labels, codes and encodings.**

- **A table of CODES is not a table of categories, and the plausible reading is wrong often enough to
  be dangerous.** Hungary's exports carry `RE_C`, `RE_CA`, `RE_CO`, `RE_OU` and no labels. `RE_CA` is
  **Calvinist**, not Catholic — Catholic is `RE_C`. `RE_CO` is "Other Christian", not Coptic. `RE_OU`
  is **Ukrainian** Orthodox, a jurisdiction absent from KSH's own prose list of the five Orthodox
  churches in Hungary, **so domain knowledge would have rejected the correct answer too.** Pin every
  code against a published national total before writing a row, and re-derive the pinning in `check()`
  so a reordered codelist fails the run instead of silently relabelling the map. Where labels exist at
  all, read them from the source at run time rather than transcribing them.
- **Arithmetic pins STRUCTURE even when no labels exist.** Hungary's three category groupings were
  forced to the person by summation — 11,042 + 7,983 + 3,307 + 7,645 = 29,977 exactly — before any
  label was in hand. A hierarchy deduced that way is stronger than a hand-written one, and **the
  deduction is worth doing first: it tells you what the labels have to mean, which is a check on them
  when they arrive.**
- **Two tables of one census can spell one category two ways.** India's C-01 writes `Other religions
  and persuasions`; its own Appendix writes `Other Religions and Persuasions` as the parent row inside
  every state block. Matching the parent by label silently failed and added each state's bucket total
  as though it were a named religion — 15.7M against a 7.9M bucket. **Match a parent on its code
  wherever the source gives codes.**
- **Indentation is often the only structure.** Leading dots (Estonia), or *which column* the text lands
  in (Poland). Parse by position, not by matching label text — the labels carry trailing "w tym:",
  embedded newlines, and typos.
- **The office's own typos are part of the data.** Statistics Estonia writes `Taara Beliver` in one
  table and `Taara Believer` in another. Map both in the taxonomy; do NOT repair it in the normaliser,
  because the normalised CSV is supposed to reproduce the source verbatim.
- **Headers can be two languages in one cell.** `Katolici Catholics`, `Ostali kršćani1) Other
  Christians1)` — footnote markers included. Pick one language as the mapping key and keep it
  *verbatim*, footnote and all, so the taxonomy key matches what the normaliser writes.
- **UNESCAPE ANYTHING SCRAPED OUT OF AN HTML ATTRIBUTE, AND THE LANGUAGES THAT NEED IT ARE NOT THE
  ONES YOU EXPECT.** Israel's locality names are joined on strings pulled from `data-search="…"`, and
  Hebrew abbreviations carry the **gershayim**, which is a literal `"` — `בני עי"ש`, `כפר ביל"ו`,
  `גבעת ח"ן`. Inside an attribute that must be `&quot;`, so a raw regex capture yields
  `גבעת ח&quot;ן` and the join against the census file's own spelling fails. **35 real localities
  resolved to nothing and every check still passed**, because 1.1% of a country is inside every
  tolerance anyone would set; they were visible only because the run prints what it could not
  resolve. Two rules: `html.unescape` every scraped attribute, and **make a fetch report the units it
  missed BY NAME**, because a count alone would have read as ordinary attrition.
- **TWO CATEGORY LABELS CAN DIFFER BY ONE WORD AND MEAN UNRELATED THINGS.** CBS's dashboard returns
  `Others` — the population register's "not classified by religion", a real category with a real node
  — and `Other religions`, which is **everything except the dominant group, lumped**, and is not a
  category at all. Nazareth comes back `Muslims 73.1% / Other religions 26.9%` where that 26.9% is
  essentially all Christian. Mapping the two alike, in either direction, erases Israel's Christians.
  **Nothing in the data distinguishes them**: both sum to 100% with their siblings, and a lumped unit
  is indistinguishable from a genuinely homogeneous one. It was caught by checking six places whose
  composition is known independently — a Christian village, a Druze village, a mixed town — which is
  this section's "check one number you already know", done six times because one would not have been
  enough to see the pattern.
- **Set `sys.stdout.reconfigure(encoding="utf-8")` at the top of every source script — and every
  TOOL.** The Windows console is cp1252 and will kill a run on `ł`, `ș` or `õ` at the *print*, which
  makes it look like a data error. `tools/check_overview.py` did not have it and died on
  `Tứ Ân Hiếu Nghĩa` after the measurement had already succeeded.
- **A PDF'S TEXT LAYER NEED NOT BE IN THE SAME UNICODE NORMAL FORM AS YOUR SOURCE FILE, AND ONLY SOME
  WORDS WILL SHOW IT.** GSO's 2019 volume returns `Giá o hội Cơ đố c Phục lâm Việt Nam` with `á` and
  `ố` **decomposed** — base letter plus combining acute — while every other Vietnamese label on the
  same page comes back precomposed. The two affected syllables are exactly the two the typesetter split
  across glyph runs, so the defect follows the *typesetting* and not the language. Compared as bytes
  the string is unequal to a visually identical literal in the taxonomy file; the category resolves to
  nothing, `countries.py` drops the rows, and **the two strings are identical in a terminal, in a diff
  and in code review.** `tools/check_mapping.py` caught it and nothing else would have. **Normalise to
  NFC on write AND on read**, and do not treat this as a Vietnamese problem.

**Reading numbers out of a PDF.**

- **TWO VOLUMES OF ONE SERIES CAN USE DIFFERENT THOUSANDS SEPARATORS, AND A SPACE-SEPARATED ONE CANNOT
  BE PARSED FROM TEXT AT ALL.** The 2009 *Kết quả toàn bộ* writes `85.846.997` and the 2019 volume of
  the same series writes `96 208 984`. The ambiguity is real rather than theoretical:
  `Tôn giáo Baha'i 2 153 1 089 1 064 841 419 422 1 312 670 642` has a second valid reading in which
  841, 419 and 422 are three values, and an anchored nine-number regex reaches it by backtracking,
  silently, with every value a genuine integer. **Read it by geometry instead** — right-aligned columns
  on a fixed grid, digit groups ~2.6pt apart inside one number and >11pt between columns — and
  **calibrate the grid off a row whose values you already know** rather than hard-coding pixel
  positions. Two details that cost time: the columns are right-aligned, so a two-digit cell starts ~5pt
  further right than a three-digit one and a left-edge grid misses it; and the label/number x-cutoff
  has to clear the rightmost label word on the whole page, not the typical one. Then **assert the
  arithmetic the table already contains** — total = male + female, total = urban + rural — which is 34
  equations over 17 rows and breaks on any misplaced group.
- **A WRAPPED LABEL CAN HAVE FRAGMENTS ON BOTH SIDES OF ITS FIGURES.** `gy.py`'s rule is that a
  record's label is the row's own text plus the label-only rows that TRAIL it, which is right for
  Guyana. Vietnam's Latter-day Saints row prints half its name above the numbers and half below, so the
  trailing rule gives the *previous* category the opening half and the *next* category the closing
  half: **one wrapped row corrupts three labels**, all into plausible-looking strings. Assign each
  fragment to the NEAREST record by y — the two halves are 5pt from their own figures and 18pt from
  their neighbours'.
- **A TABLE'S OWN CAPTION NUMBER IS A DATA-SHAPED TOKEN SITTING INSIDE THE COLUMN BAND.** Where a
  column parse takes "every digit token between the title and the first data row" as the population
  row, `Tableau 2 :` contributes a bare `2` — at x=115 in INStaD's booklets, which is 190pt from most
  departments' first column and **14pt from two of them**. So **Cotonou reads 2,679,012 people instead
  of 679,012** and the Plateau 2,622,372 instead of 622,372, in two booklets of twelve, with the other
  ten correct, both wrong figures plausible and both parsing as integers. Only the national sum sees
  it. **Cut the caption's own line off explicitly rather than assuming the header band is text.**
- **AND A LABEL DOES NOT SHARE A BASELINE WITH ITS FIGURES**, so binning rows on the top edge splits
  some of them. The label is usually a point smaller and sits a little lower; in Benin's Atacora
  booklet that put seven of ten religion labels in a different bin from their own values, and the
  table came back looking as though the office had **changed its category list**. Cluster rows on the
  vertical CENTRE with a tolerance, swept in order, never on a fixed bin.
- **A SOURCE CAN NUMBER ITS COLUMNS IN A DIFFERENT ORDER FROM THE ONE IT PRINTS THEM IN, AND TWO
  VOLUMES OF ONE SERIES CAN DISAGREE ABOUT WHICH.** Austria's Volkszählung 2001 heads Tabelle 4
  `1 2 3 5 4 6 7 8 9 10 11` in all eight Bundesländer volumes — Orthodox is printed **fourth** and
  numbered **5**, Evangelisch printed fifth and numbered 4 — while the **Wien** volume numbers the
  same eleven columns in print order. A parser keying on the printed number therefore swaps
  **Orthodoxy and Protestantism in eight volumes of nine**, and nothing downstream sees it: both are
  plausible sizes, the categories still sum to the row total, the Gemeinden still sum to their Bezirk,
  the Bezirke still sum to the nation, and even UNSD's independently forwarded national figures still
  reconcile — because the swap is *consistent inside each volume*. The country would simply have
  looked oddly Orthodox. **This is a different species from every other trap in this list**: Malawi's
  rotated table and Benin's caption band are extraction failures that rendering the page reveals, and
  this one survives rendering, because the page is right and the *source* is internally inconsistent
  across its own volumes. It is visible only by comparing two volumes, or by ignoring the numbers.
  So: **identify a column by its header LABEL, assert the resulting order against a written-out list,
  and treat a printed column number as decoration.** Two cheap confirmations were available here and
  usually are — a regional government's re-publication of the same table as a spreadsheet (Vorarlberg
  prints the same anomaly, so it is the source's), and the volume's own prose, which quoted three
  figures that pin the label order.

**And one thing the office may have done to the data before you see it.**

- **AN OFFICE CAN HAVE PRORATED ITS NON-RESPONSE AWAY BEFORE PUBLISHING, AND ONLY A FOOTNOTE SAYS SO.**
  Every rule in §3.5 assumes non-response is a column you can choose not to draw. Guyana's Table 2.19
  has no such column because the Bureau took its 363 not-stated, 16,331 no-contact and 7,443
  institutional people — 3.23% of the country — and **distributed them across the thirteen religion
  categories in proportion**, saying so in four lines under the table and nowhere else. So "100% of the
  census is drawn" and "some of what is drawn is the office's estimate" are both true, and no figure at
  any geography separates them. **Ask whether the office has already redistributed its non-response,
  and read the footnote to find out.** The trap is that the symptom — a category set that partitions
  the population exactly, with no non-response cell — is *also* what an unusually clean source looks
  like, so it is invisible from the numbers alone. **Record it and do not undo it**: reversing a
  proration means inventing the distribution it replaced (§14.4).
- **A source with a publication floor needs its remainder emitted as a category** — *and then that
  category needs mapping.* India's Appendix names a religion only at 100+ adherents nationally, leaving
  1.9% of the bucket unnamed; without an explicit row for it `allocate.py` normalises shares over the
  named categories and inflates every one by ~2%, silently and in the direction that flatters the map.
  Emitting the row then created the *other* silent failure: it resolved to nothing in the taxonomy and
  `countries.py` dropped 149,668 people without a word, while every reconciliation upstream of the
  taxonomy still passed. **Adding a category is a taxonomy change even when it comes out of the
  normaliser.**
- **A FLAT EXPORT THAT NESTS BY POSITION FAILS SILENTLY ON ONE MISSING SPACE, AND THE VICTIM IS THE ROW
  ABOVE.** SingStat's subzone population CSV marks a parent as a header row reading `<name> - Total`
  with its children following unindented and unmarked, so the parent is carried positionally.
  **`Changi- Total` is printed with no space before the hyphen**, alone among 55. An
  `endswith(" - Total")` match misses it, Changi is never opened, and its three subzones are
  attributed to the PREVIOUS header, Central Water Catchment — an uninhabited reservoir catchment
  that then holds 3,700 people while its own total row says nil. **Nothing else complains**: every
  other parent still reconciles against its own children and the grand total is untouched, because the
  rows were merely moved between parents. This is [[reference_name_join_wrong_neighbour]] in
  positional form, and it is invisible to arithmetic for the same reason. **The defence is to assert
  the NUMBER OF PARENTS against an independent list — here URA's 55 planning areas — and never
  against the file's own totals.** Match the separator loosely (`^(.*?)\s*-\s*Total$`) and let the
  count assertion be the thing that catches you.

### Joining to boundaries

**Get the right FILE first — the tabulation geography, not the administrative one.**

- **WHEN THE COUNTING GEOGRAPHY IS NOT THE ADMINISTRATIVE GEOGRAPHY, LOOK FOR A BOUNDARY FILE CUT FOR
  THE TABLES.** PSA tabulates religion on *province excluding any highly urbanised city inside it*:
  Cebu means Cebu minus Cebu City minus Lapu-Lapu minus Mandaue, and the 33 HUCs are separate rows.
  Every general boundary source carries the plain provinces instead, so **an ADM2 join double counts
  all 33 HUCs while every unit count and every name looks right.** COD had the wrong tier and a partial
  ADM4; geoBoundaries had no ADM4 at all. What worked was the **U.S. Census Bureau's per-country
  geodatabases**, which are built to link to another country's census tables and therefore carry that
  country's *tabulation* tier ([[reference_uscb_country_gdb]]). **The tell is independent cities, HUCs,
  census-only units, or any "excluding…" in a row label.**
- **BUT LOOK ON THE OFFICE'S OWN SITE FIRST.** Before COD, geoBoundaries or a USCB geodatabase, look
  for a *census atlas*, *geo-files* or *GIS* link on the statistical office's own platform. When one
  exists it is the census vintage by construction and cut to the tabulation geography — the two things
  the Philippines cost a session to get. GSS's was a single link on a page already open, and it held
  BOTH published tiers (261 districts and the 272 with metros split into sub-metros), against
  geoBoundaries' 260 units on a 2019 vintage with no sub-metros.
- **"THE OFFICE'S OWN SITE" INCLUDES A GIS SERVER, WHICH IS NOT LINKED FROM THE DOWNLOAD PAGES AND IS
  NOT IN THE NATIONAL OPEN-DATA PORTAL.** Moldova (§9bv) looked like a forced name join: geoBoundaries
  has `MDA` at ADM0/ADM1 only, HDX's COD-AB the same, Kontur's extract 287 units against the 901
  wanted, and an OSM join was written and got to 898 of 901 before **`gis.statistica.md`** turned up
  with 212 hosted FeatureServers of 2024 census indicators pre-joined to geometry at three tiers.
  The national portal is not where to look — `dataset.gov.md` returns **zero** results for
  `geospatial`, `shapefile`, `hotare` and `cadastru`, and `geoportal.md` is 410 Gone. **Two hosts,
  two different names:** `gis.` or `geo.` prefixed on the office's own domain, browsable at
  `/server/rest/services?f=json`; and the office's **ArcGIS Online organisation** at
  `services-eu1.arcgis.com/<orgid>/`, where BNS publishes its LAU and NUTS layers under CC-BY. Try
  both before accepting a name join, and see [[reference_gis_server_census.md]], which is the same
  finding from the other direction.
- **A POPULATION COLUMN ON THE POLYGON TURNS A CODE JOIN INTO A PROVED ONE.** A code join can still be
  a join to the wrong *vintage* of the same units, and nothing about matching codes detects that.
  Moldova's commune layer carries `p_distrib`, the office's own 2024 census population per polygon,
  and it equalled the total computed from the religion table **to the person on all 896 joined
  units**. Ask what population field a boundary service offers and assert against it; it is free and
  it is a stronger statement than any name join can make.
- **A COUNTRY CAN HAVE TWO OFFICIAL CODE SYSTEMS THAT DO NOT CORRESPOND.** Moldova's CUATM carries a
  7-digit *cod statistic* and a 4-digit *cod unic*. Below the raion neither is derivable from the
  other (Drepcăuţi is `1422000` and `1426`) because the unique code numbers sub-village localities
  that the statistical code does not, and they coincide only for towns, which both systems number
  first. The census publishes one and OpenStreetMap tags the other. **A code that looks like a
  truncation of the other code may not be one; check a village and not only a town.**
- **TWO FILES LABELLED "ADM2" ARE NOT TWO FILES AT THE SAME LEVEL.** HDX's `ken_admpop_2019.xlsx` has
  345 ADM2 rows; COD's `ken_admin2.shp` has 290 ADM2 polygons. Kenya's administrative **sub-counties**
  and its **constituencies** are different tiers, both routinely called ADM2, and neither file says
  which it means. Joined within county by folded name it is 182 rows of 345 and **40.5% of the
  population unplaced**. The tell was not a name mismatch — it was the COUNT, visible before any join
  was attempted. **Compare the row counts of two files at the "same" level before writing the join.**
- **A BOUNDARY FILE WITH ONE FEATURE TOO MANY MAY BE TELLING YOU HOW TO FIX IT.** geoBoundaries VNM
  ADM1 has 64 features and 63 distinct `shapeISO` values: the extra is **Côn Đảo**, an offshore
  *district* of Bà Rịa–Vũng Tàu carried separately and correctly given the parent's code. Dissolving on
  `shapeISO` reassembles the province. A feature-count assertion alone reads 64-against-63 as an
  off-by-one and sends you to `drop_duplicates()`, which discards the islands or the mainland depending
  on row order. **Count the KEYS, not the features, and treat a duplicated key as a grouping
  instruction until proved otherwise.**
- **AND CHECK WHETHER THE COUNTRY HAS REDISTRICTED SINCE THE COMMIT YOU PINNED, NOT ONLY SINCE THE
  CENSUS.** §8.1's rule is about the data's vintage; this is about the file's. Vietnam merged 63
  provinces into **34** on 1 July 2025, so a boundary release from after that has no Hà Nam, no Bạc
  Liêu and no Ninh Thuận, and its An Giang is An Giang plus Kiên Giang — not a subtly wrong file but a
  different country, joining at maybe half strength with no obvious symptom. **Pin the release, assert
  the feature count, and put the reason in the error message.**
- **Vintage, always** (§8.1). geoBoundaries POL ADM3 is 2017. Four Estonian EHAK codes were retired
  between the 2021 census and the 2024 boundary file. Prefer a boundary set from the census year; when
  you cannot, prove the join instead of arguing about it.
- **A form-gated boundary file may be mirrored somewhere ungated.** SHRUG's own download needs a form;
  the identical parquets are plain GitHub release assets in `yashveeeeeeer/india-geodata`
  ([[reference_india_census_geo]]). **Check for a mirror before treating a form as a wall — and check
  the licence on the mirror**, because SHRUG's is CC-BY-NC-SA, the first non-commercial source here.
- **Eurostat GISCO LAU 2021 is the boundary answer for 34 European countries** in one 98MB zip, and its
  companion LAU–NUTS correspondence workbook carries `NUTS3 | LAU CODE | LAU NAME NATIONAL` — which is
  how Romania, whose census has no codes at all, was joinable. **Those 34 are NOT the EU27** — `MK`,
  `RS` and `AL` all have full LAU coverage, so North Macedonia needed no boundary download at all. The
  correspondence WORKBOOK is EU27 and excludes them: **polygons yes, code bridge no.**

**Then join on a CODE if one exists, and prove it.**

- **ASK EARLY WHETHER THE RELIGION TABLE CARRIES A GEOGRAPHIC CODE — it decides whether the join is a
  lookup or a day's work.** Every trap in Romania's, Ghana's and Serbia's boundary work — folded
  transliteration, parenthesised aliases, acronym collisions, two municipalities called Palilula —
  descends from one fact: those sources publish names only. Lithuania publishes codes, GISCO's `LAU_ID`
  **is** that code, and the whole join is `zfill(2)` and a merge. **It is the second question to ask a
  candidate source, right after "does it go deep enough", because the answer changes the estimate by a
  day and occasionally decides whether a country is worth doing at all.**
- **A matching unit count is not a join.** Poland: GISCO's 13-digit `LAU_ID` and GUS's 7-digit TERYT
  share no substring, both sides have exactly 2,477 units, and joining as delivered matches **zero**.
  **Always print the join both ways and fail on either side.**
- **TWO NUMERIC-LOOKING CODE SPACES CAN OVERLAP COMPLETELY AND AGREE NOWHERE.** Vietnam's census keys
  provinces by GSO's administrative code; geoBoundaries keys them by `shapeISO`, which is ISO 3166-2:VN.
  Both are two digits, both run over the same 63 units, and **`04` Cao Bằng is the only province where
  they mean the same place** — GSO's `02` is Hà Giang and ISO's `VN-02` is Lào Cai. A numeric join
  produces 63 wrong assignments, and the unit count, the national total and every category total still
  reconcile, because nothing downstream can see which polygon a correct number was drawn on. **Bridge
  the two spaces explicitly, re-derive the bridge by name on every run, and check it against a quantity
  neither side determines** — here, Kontur population per province against census population.
- **TWO FILES THAT SHARE A CODE *SHAPE* DO NOT SHARE A CODE, and a 96% match is not evidence that the
  96% is right.** This is the inverse of Poland's trap and far more dangerous, because it *looks like
  it worked*. COD's `adm4_pcode` is exactly `LK` + the census's district/DS/GN triple; joining on it
  matched 13,472 of 14,003, and the misses looked exactly like the vintage gap a 2022 file and a 2024
  census would predict. But **13 DS divisions had been renumbered between the two vintages**, so the
  join did not miss those units, it paired each of them with a real polygon somewhere else in the same
  district — **762,824 people, 3.5% of the country, in the wrong valley, with no symptom at all.**
  Poland's failure announced itself by matching zero; this one by matching *almost everything*. **The
  general rule: a shared code is only trustworthy as far up the hierarchy as you have independently
  verified it.** The fix is to stop using the code as a global key — align the COARSER level by NAME
  first, then match the finer code only *within* an aligned pair, which makes the code local, which is
  all it was ever reliable as. **Names survive a renumbering and codes do not.**
- **A partial match rate has two explanations and they need telling apart.** "Vintage gap" and "the key
  is wrong for part of the file" produce the same number. Distinguish them by checking the matched side
  for something the key does not determine — for Sri Lanka, comparing DS *names* across the code join,
  since a correct pairing cannot put Walapane's polygon under Nildandahinna's name.
- **A code can stop being comparable altogether when one side splits a unit.** COD numbers Kalmunai's
  58 GN divisions 005–300, while the census splits the DS division in two and **restarts each half at
  005**. Matching on the code gives all 29 low numbers to whichever half is seen first and orphans the
  other. **Where two units on one side share one unit on the other, pool them and match on names only.**
- **WHEN A UNIT CANNOT BE PLACED, ITS FALLBACK IS THE UNMATCHED REMAINDER OF ITS PARENT, NOT THE WHOLE
  PARENT.** Everything that did match belongs to some other child, so what is left is where the unplaced
  people must be — a tighter and strictly more honest area, for free. Kalmunai is the case that proves
  it matters rather than being tidy: COD holds the town as one DS division and names only the 29 GN
  polygons of its Tamil half, leaving the Muslim half's 29 blank, so no name can reach them. The
  remainder puts those **52,798 people in the correct half of the town**; the whole-parent fallback
  would have smeared them across both halves and **erased the sharpest religious boundary in the
  country.**
- **ID formats are per country.** Romania's `LAU_ID` simply *is* the SIRUTA code; Poland's needs
  slicing; Estonia's PxWeb code is a concatenation of EHAK codes. **Assert the format (length, digits)
  before slicing**, so a reissue fails loudly.
- **A longer key can be the safer one, which is the exact reverse of Poland.** Poland's LAU id had to be
  sliced DOWN to six digits; Germany's 12-digit ARS must not be shortened to the 8-digit AGS. The
  difference is what the extra digits carry: Poland's were a unit TYPE the boundary file omits,
  Germany's are the *Verbandsschlüssel*, which changes when a Gemeinde moves between Ämter. Joining
  Germany on the AGS makes the two leftovers disappear and looks like a fix — while orphaning three
  populated polygons whose people are counted elsewhere, placing ~3,000 people in the wrong villages
  **with every count still reconciling**. **There is no rule about key length. There is only printing
  the join both ways and asking what the leftovers *are*.**
- **Verify a derived key with something independent.** For Poland it was names: 2,476 of 2,477 agreed,
  and the one that did not was a real 2021 rename. A wrong offset rule cannot produce that.
- **TWO ORDERINGS THAT AGREE 92% OF THE TIME ARE NOT THE SAME ORDERING.** Malawi got a free
  independent key out of the fact that the census's print order reproduced COD's `adm2_pcode` order,
  and Benin looks identical — both alphabetical within a department — and **fails on six of 77**. All
  six are adjacent transpositions with dull explanations: COD sorts under its OWN spelling, so `Kobli`
  follows `Kérou` where the census's `Cobly` precedes it; a hyphen sorts before a letter for one side
  and is ignored by the other, swapping `Za-Kpota` and `Zagnanado`; and one department is simply not
  alphabetical. **Six rows in 77 is exactly the size of discrepancy a stale vintage produces**, so an
  assertion kept and then loosened until it passed would have stopped detecting anything at all. Two
  responses, both needed: **mint the positional id so it cannot be mistaken for the official code**
  (`BJ12-07`, not `BJ1207`), and **assert the weaker true thing** — that no unit's rank moves by more
  than one place, which a transposition of neighbours survives and a shifted block does not.
- **"The 2022 boundary file" can be two different files.** BKG publishes a **01.01 and a 31.12 edition
  of every year**, and destatis never states which Gebietsstand it published on. Against the German
  census: 01.01.2022 leaves 2 unmatched, 01.01.2023 leaves 10, 31.12.2022 leaves **none**. **Try them
  all and let the leftovers pick; do not reason about which *ought* to be right.**
- **WHERE THE CENSUS NUMBERS ITS UNITS AND NAMES NONE OF THEM, TRY ISO 3166-2.** Guyana has the
  opposite of the usual problem: Table 2.19's columns are `Region 1` … `Region 10` and the compendium
  never prints `Barima-Waini` anywhere, so there is no name on the census side to match on.
  **geoBoundaries carries `shapeISO`, and ISO 3166-2:GY is exactly the ten regions in region-number
  order**, so the join is a transcription of a published standard, checkable by asserting the code set
  matches and is unique. `shapeISO` is present on many geoBoundaries layers and nothing here had looked
  for it. It earned its keep immediately: the file misspells Region 1 as `Barina-Waini`, so a name join
  would have failed on exactly one region of ten and read as a vintage gap.

**Joining on NAMES, where there is no code.**

- **AN ACRONYM COLLIDES THE WAY A CODE DOES, AND LOOKS FRIENDLIER WHILE DOING IT.** GSS names sub-metros
  by their parent's abbreviation, and **`TMA` is Tema Metropolitan Area in Greater Accra and Tamale
  Metropolitan Area in Northern**, 600 km apart. Matching `TMA-` on a single global acronym hands all
  four sub-metros to one of them and orphans the other, and **every national and regional total still
  reconciles**, because the rows are all present and all in the same country. Sri Lanka's rule in a
  human-readable costume: resolve the prefix inside the parent block. Beside it, the near-miss that
  shapes the rule: `Nkwanta North (Kpassa)` is a parenthesised *alias*, not a metro, so the test has to
  be "a parenthesised acronym that some row in the same region uses as a prefix" — both halves
  load-bearing.
- **A PLAIN PLACE NAME COLLIDES TOO, AND IT LOOKS EVEN FRIENDLIER.** Belgrade has a Palilula and so does
  Niš, 200 km apart, and RZS publishes no codes of any kind. An acronym at least *invites* the question;
  a real name does not, and the failure is Ghana's exactly. **Compute which names repeat rather than
  listing the ones you found**: `rs_geo.py` collects the bare names appearing more than once on *either*
  side of the join and qualifies only those with their parent, so a collision introduced by a future
  census is caught by the code that already exists. **A hand-written exception table is a note that goes
  stale; a derived one is a check.**
- **Diacritics that look identical are not.** `ş` U+015F (cedilla) vs `ș` U+0219 (comma-below), and
  `ţ`/`ț`. INS writes one, Eurostat writes the other, for the same names. Fold to ASCII on both sides or
  a third of Romania silently fails to match — **and it looks exactly like a vintage problem.**
- **Where a name join is the fix, fold transliteration — but only ever within a parent.** DCS and COD
  romanise Sinhala and Tamil differently on 81 of 340 DS names (Mathugama/Matugama, Dickwella/Dikwella,
  Vadamaradchi/Vadamaradchchi). A fold aggressive enough to catch those — collapsing aspirate digraphs,
  `w`/`v`, `ee`/`i`, doubled letters, and `h` entirely — is far too aggressive to be a national key and
  must be applied inside one district, with **every match required to be 1:1** so a collision is
  reported rather than resolved.
- **Resolve leftovers by elimination, never by guessing.** Romania had 8 unmatched names of 3,181
  (`Râşca`/`Rişca`, `Sfântu`/`Sfântul Gheorghe`); each was the only one left in its county, so it is a
  deduction. **Refuse when two or more remain on either side.**
- **Derive alias maps, do not hard-code them.** A frozen list of four renames goes stale in silence at
  the next release; a rule that re-derives them fails loudly instead.
- **When names disagree across a good join, find out WHICH side is wrong before calling it spelling.**
  Chile's CUT join is 345 of 346 both ways with no spares, and the name cross-check still turned up
  three disagreements. Two were spelling. The third was `CL01401`, named **Tocopilla** in COD against
  **Pozo Almonte** in the census — different towns 400 km apart, which reads exactly like Sri Lanka's
  wrong-unit pairing. It is not: `CL01401`'s province is Tamarugal, its region Tarapacá and its area
  13,738 km², which is Pozo Almonte to within 0.2%, while the real Tocopilla is `CL02301` and is present
  and correct. **COD's name field is wrong and its geometry is right.** Two rules: resolve such a
  disagreement against the PARENT UNITS AND THE AREA, which the code join does not determine; and
  **take names from the statistical source rather than the boundary file**, since the census office is
  authoritative for its own place names. Keep the resolved list in the script so a *fourth* disagreement
  fails the build instead of joining the known-harmless pile. **And the corollary for the case Chile
  does not cover** (Guyana's): when the statistical source publishes *no* names, the order becomes
  **prefer a standard to a file, and a file to a guess.**

**When a unit has no polygon, or the wrong parent.**

- **Look for the sub-level before accepting that a unit has no geography.** India publishes units called
  `Area not under any Sub-district` — 17.4M people, including the whole Kolkata metropolitan fringe —
  for which no polygon exists at that level, and whose district's polygons tile it completely, so there
  is no leftover shape. The census also publishes their **town** rows, which sum to the unit's
  population **exactly, 100.0%**, and every one of those towns has a polygon. **A census that publishes
  a residual usually publishes its parts somewhere, and the union of the parts is a fact rather than an
  estimate.**
- **A unit with no polygon anywhere may still be reconstructable from its parts.** The BARMM Interim
  Province — 63 barangays moved by the 2019 plebiscite — is younger than every published Philippine
  boundary layer. The USCB had folded those barangays back into Cotabato *and tagged each one* with the
  cluster it came from, so the unit was rebuilt from the tags. **The two halves of the discrepancy were
  the same fact**: Cotabato was the single unit failing the total check, and it failed by exactly the
  missing province's population. **When one unit is missing and one unit's total is too large, check
  whether they are the same people before treating them as two problems.**
- **AND WHEN ONE UNIT IS MISSING, THE FAILURE SURFACES SOMEWHERE ELSE.** geoBoundaries omits an entire
  Korean county (Yeonggwang-gun, 53,984 people): no polygon of that name, 228 polygons against 229
  census units. Rebuilt from the eleven ADM3 eup and myeon that fall inside no ADM2 polygon, 481 km²
  against a published 475, with the selection checked three ways (the eleven must be present, they must
  dissolve to one shape, the area must match) so a later release that fills the hole fails loudly. **The
  two things to carry are about diagnosis, not repair.** The missing unit was NOT the one that looked
  obvious — Sejong was the expectation, being both a province and its own single sigungu, and it turned
  out present. And **one missing unit cascaded into a second, wrong-looking failure**: with the county
  absent its province was a polygon short, the count-constrained assignment took a neighbouring city's
  district to fill the gap, and the visible symptom was that CITY failing, 500 km away. **When two units
  fail in different places, look for one defect before assuming two.**
- **A BOUNDARY FILE'S OWN PARENT LAYER CAN BE UNUSABLE FOR ASSIGNING ITS CHILDREN.** The natural way to
  give each ADM2 unit a parent is point-in-polygon against ADM1, and in Korea it fails twice over.
  **First, the ADM1 polygons OVERLAP each other**: six metropolitan cities are enclaves carved out of
  the province around them, and geoBoundaries draws the surrounding province *without cutting the city
  out* — so a point in Gwangju's Dong-gu is inside both `Gwangju` and `South Jeolla`, `sjoin` returns
  two rows, and dropping duplicates hands whole metropolitan cities to the wrong province. 14 units
  landed in a neighbour, and the symptom was not a spatial error but a *name* failure downstream, in a
  different province. **Second, greatest-overlap does not rescue it**, because parent and child layers
  are different vintages: 85 of 228 districts sit less than 90% inside their best province and an island
  district came out 77% inside the wrong one. **A geometric assignment is only as good as the geometry,
  and two layers from one publisher are not thereby aligned.** The way out generalises: **derive the
  parent from the NAMES where the names are unique, and use geometry only for the remainder.** A fold
  unique on both sides matches globally with no parent needed and thereby *tells* you its polygon's
  parent (199 of 229 here); the colliding remainder takes the nearest anchored neighbour's parent,
  **constrained by the per-parent counts the statistical source already gives**, so a parent that is
  full cannot take another's.
- **WHERE THERE IS NO POPULATION COLUMN, THE PARENT UNIT IS THE FREE INDEPENDENT CHECK.** Ghana's
  boundary file carries only names and a `Region` attribute, and on the census side a unit's region comes
  *only from row order*. Those two are genuinely independent, so their agreement on all 272 confirms the
  join AND the positional parse in one move. It is exactly the check that would have caught Romania's
  county-header bug, it is available wherever a boundary file carries a parent column, and it costs
  nothing. It also found the only real defect in the file — one row spelling Greater Accra
  `Greate Accra`.
- **The strongest join check is a quantity the join does not determine — and a boundary file may hand
  you one.** The Philippine USCB geodatabase carries the census's own religion table as a layer, so
  `RLG_HPOP` could be compared against the independently-read `ph.csv`: exact agreement, to the person,
  on 115 of 116 units. That is worth more than any amount of name agreement, because a wrong pairing
  cannot produce it. **Ask what else is inside a boundary download before treating it as only geometry.**
- **A source's population and a boundary file's population measure different things, and asserting them
  equal fails on honest data.** North Macedonia's census counts *residents*; GISCO's `POP_2021` does
  not; the country has lost a fifth of its people to emigration and the two disagree by a median 11.7%,
  worst in exactly the western emigration municipalities. **Assert the RELATIONSHIP instead**: a correct
  join keeps every unit's ratio inside a factor of two around a tight median, a scrambled one pairs
  villages with cities and scatters it over orders of magnitude. Written as an equality it either fails
  on every real difference of definition or gets loosened until it detects nothing. (Also:
  **GISCO's `POP_2021` is 0 for seven of Skopje's ten municipalities** — a live trap for anyone reaching
  for it as a weight. Print such holes rather than filtering them.)
- **A CODE HIERARCHY TELLS YOU THE CURRENT PARENT AND SAYS NOTHING ABOUT THE HISTORICAL ONE.** BPS
  sub-district codes are `regency(4) + kecamatan(3)`, and a new regency is carved out of whole
  kecamatan, so dissolving a modern file's ADM3 by the first four digits looks like a free way to
  rebuild the census vintage's ADM2. It is not: **a regency created after the census gets entirely NEW
  kecamatan codes under its own prefix** — Mahakam Hulu's five are `6411010`..`6411050`, not Kutai
  Barat's `6402xxx` — and the boundary file's ADM3 codes agree with its own ADM2 for all 7,069. So the
  dissolve rebuilds only the part of the old unit that stayed put. **Stable-looking leaf codes are no
  evidence either way; the only thing that carries history is a source that states the parentage.**

**Capitals and sub-city geography.**

- **Watch for the capital in one polygon.** Tallinn is 33% of Estonia, Prague 12.4%, Bucharest 9.8%,
  Warsaw 4.7%. If the office publishes religion for city districts, use them and let them REPLACE the
  parent (Czechia, Estonia). If it does not, leave one polygon and say so — subdividing invents
  structure the source does not have (§3.10).
- **When the CENSUS is finer than the boundary file, that is a different problem and it is usually
  solvable.** Zagreb and Budapest are the same case — religion published per city district, GISCO LAU
  stopping at the city — and Croatia lost it while Hungary won it, purely because someone looked in a
  second place. Budapest's 23 kerület are in **geoBoundaries ADM2**, whose Hungarian level is járás and
  therefore includes them. **Check ADM2/ADM3 there before accepting one polygon for a capital; and check
  the licence per level**, because geoBoundaries HUN is CC0 at ADM1 and ODbL at ADM2.
- **AND THERE IS A THIRD ANSWER: THE SUB-LAYER EXISTS AND IS NOT GOOD ENOUGH.** Hungary won and
  Croatia lost, so the rule read as "look harder". Benin is the case in between and it looks exactly
  like Hungary's win until something independent is measured. INStaD publishes religion for Cotonou's
  **13 arrondissements** — 6.8% of the country — COD-AB ships no ADM3 for Benin at all, and
  geoBoundaries' 546 arrondissements (OpenStreetMap via a uMap, ODbL) contain all thirteen, correctly
  numbered. Their union is **81.65 km² against COD's Cotonou at 80.58**, agreeing to 1.3%: on area and
  on names it is a clean result. **The IoU is 0.729**, so 13 km² sticks out and 12 km² is uncovered,
  and census population per arrondissement against Kontur's comes back at a **0.64×–1.98× band with
  r = 0.81** where the standard is a factor of two around a tight median. Clipping to the parent — the
  next bullet's fix — makes it worse, leaving one arrondissement with 16% of itself.
  **So the capital stayed one polygon**, and the parsed thirteen rows were written to the normalised
  CSV undrawn so a future layer is a lookup. **A finer tier is a gain only if something the join does
  not determine says the pairing is right; area agreement and name agreement are not that thing.**
- **Clip a borrowed sub-layer to the parent it subdivides.** Districts from a different vintage agreed
  with GISCO's Budapest on total area to two decimal places and still overhung the city edge by tens of
  metres, which would have put Budapest's dots in Budaörs. Intersecting with the parent makes the union
  exactly the parent; the cost is a thin unfilled ring, which a dot map does not care about and a wrong
  municipality is.
- **A residual geography unit that is EMPTY is a proof, and worth asserting rather than filtering.** KSH
  publishes `Budapest kerületre nem bontható adatai` — figures not divisible by district — and it carries
  no religion rows at all. **That absence is what guarantees the 23 districts account for the whole
  city.** If it ever fills, the assertion fails and the map is short by exactly that many people.
- **"N polygons unmatched" and "N polygons unmatched that are all uninhabited" are different findings,
  and only one is fine.** Germany's 204 leftovers all carry `BEZ == 'Gemeindefreies Gebiet'` — forest,
  lake and military areas with no residents and so no religion row. **Assert the property**; a populated
  polygon landing in that pile is a silent hole in the map.

### Choosing a placement layer

§8.2 is the design; this is what goes wrong in practice.

- **A FINE COUNTING GEOGRAPHY DOES NOT REMOVE THE NEED FOR A PLACEMENT WEIGHT — IT MOVES WHERE THE
  ARTEFACT SHOWS.** §8.2's placement problem was learned on Kenya, where 47 huge counties washed empty
  desert in one colour, so the reflex is that a country drawn at a fine tier does not need a grid.
  Indonesia is drawn at sub-district for 403 of its 492 regencies and needed one anyway. The 89 whole
  regencies are Kenya's case again — but the visible failure is **the dense urban unit**: Cengkareng is
  513,920 people in one kecamatan, and an even wash across its polygon makes a city read as flat-shaded
  tiles with administrative edges instead of a built-up area with a shape. **Empty units make a wash
  where nobody is; crowded units make a rectangle where everybody is, and the second is the one a reader
  notices first.**
- **WHERE THE COUNTING GEOGRAPHY IS COARSE AND UNEVENLY INHABITED, A POPULATION GRID IS THE ANSWER AND
  KONTUR IS ALREADY CHOSEN.** Kenya is 47 counties for 47.2M people and Turkana alone is 68,680 km² of
  mostly desert; uniform placement washes the empty north in evenly spaced dots, and because Wajir,
  Mandera and Garissa are 97–99% Muslim that wash is ONE COLOUR and becomes the loudest thing on the
  map. 16.5 MB gzipped, 231,360 hexes, each carrying its own population, no administrative alignment
  needed. Three mechanics worth copying: join on hex **centroids** so no hex is split between units;
  **drop and report** hexes whose centroid is outside every unit (0.55% here); and **assert every unit
  gets hexes**, because a unit with none silently empties. And **the grid is a MODEL** — assert its
  national total against the census as a *ratio band*, never an equality (Kontur 55.0M vs census 47.6M,
  ratio 1.156, four years of growth plus modelling); it is a within-unit weight, so only the shape
  matters.

- **A UNIT MISSING FROM THE `place` LAYER IS NOT DRAWN ON ITS POLYGON INSTEAD; ITS PEOPLE MOVE TO A
  DIFFERENT UNIT.** Kenya's rule above says *assert every unit gets hexes, because a unit with none
  silently empties* — and emptying is the optimistic reading. `countries.py` points `place` at ONE
  layer, so a unit absent from that layer has no geometry at all, and `scatter.py` carries its people
  into other units of the same node: they are drawn, in the wrong village, with every total still
  reconciling. Cyprus hit it on **Akrotiri**, 931 people inside the Western Sovereign Base Area, where
  Kontur models nobody — a military exclusion, not a modelling failure, so no ratio band would ever
  have caught it. **The fix is for the grid builder to append the unit's own polygon at its census
  population**, which invents neither geometry nor people, and then to assert the placement layer's
  unit count equals the counting layer's. Any country with military land, a special zone or an island
  the model skips wants that assertion; the symptom otherwise is a village that is simply not there.
- **PICK THE GRID'S RESOLUTION AGAINST THE SMALLEST UNIT, NOT THE COUNTRY.** Russia is placed on
  Kontur's **global r6** file (~36 km² hexes) and that is right for Russia; the same file for Serbia
  gives **1,991 hexes for the whole country and four municipalities with no hex CENTRE at all** —
  Vračar, Stari grad, Medijana and Sremski Karlovci, among the densest places in it. **A population
  layer that fails hardest where people are densest is the wrong layer, and the symptom is silent.**
  The test is the assert above. Keep the assert *and* a fallback: a unit smaller than one cell carries
  its own polygon as a single cell, which is `de_grid.py`'s answer for 34 German Gemeinden. **Kontur
  publishes per-country extracts at r8 and they are small** — Serbia's is 4.2 MB against the global
  r8's 2.4 GB, at `…/kontur_datasets/kontur_population_<ISO2>_<date>.gpkg.gz`, and the URL takes any
  code. So the global file is for countries too big to be worth an extract.
- **AND A UNIT ABSENT FROM THE PLACEMENT LAYER DRAWS NOTHING, SILENTLY.** Kenya could assert that all 47
  counties received hexes and stop; at 5,211 units that assertion becomes a run-stopper for four Papuan
  sub-districts where the population grid models nobody at all. **Give such a unit its own polygon as a
  single fallback cell** — which degrades to uniform placement for it and to nothing worse — and print
  the list. Failing the run would be wrong, and dropping them would empty four real places with no error
  anywhere.
- **MEASURE COVER, NOT PRESENCE.** Bangladesh passes the every-unit-has-hexes assertion and is still
  placed badly in four units, because **the counting unit can be smaller than the placement cell**: a
  Kontur r8 hex is ~0.80 km² and central Dhaka's thanas are 0.8–3 km², so the centroid join gives Adabor
  (2.29 km², 203,989 people) exactly one hex covering about a third of it. Every unit has a hex, every
  total is exact, nothing fails, and the dots crowd into a third of the thana anyway. **The presence
  check is not a cover check**, and on any country whose units approach the placement grain the number
  to print is `hexes × cell_area / unit_area`. Whether to correct it is a separate question — in
  Bangladesh it was left alone, because the fallback is an equal share over the same 2 km².
- **AND NAME THE UNITS THE GRID MODELS WORST, RATHER THAN PRINTING A MIN AND A MAX.** The ratio band
  verifies the join; it does not tell the reader anything. Per unit it does: Kontur misses about four
  fifths of Serbia's Albanian-majority Preševo valley (Bujanovac 0.18×, Preševo 0.19×), so **the
  country's two most Muslim southern municipalities sit on its weakest placement surface**. That moves
  dots inside a unit and never a count, but it is exactly the sort of thing a reader would want flagged.
  **A modelled population surface is least accurate where the model's inputs — night lights, building
  footprints — are thinnest, which is not at random.**
- **THE RATIO BAND IS NOT A CHECK UNTIL YOU HAVE SHOWN IT DISCRIMINATES, AND TWO LINES OF SHUFFLING
  SHOW IT.** Every country here that joins by name asserts a Kontur/census band per unit, on the
  reasoning that a scrambled join scatters over orders of magnitude. **Nobody had ever tested whether
  that is true of the country in front of them, and for Benin it is false**: the communes are mostly
  50,000–250,000 people and look alike, so shuffling the census populations across the polygons leaves
  all but ~11 of 77 **inside the same factor-of-four band**. A band a scrambled join mostly passes is
  decoration, and widening it to admit an honest outlier — Benin has two, both real — quietly makes it
  decoration even where it started out useful.
  **The fix is to measure the null rather than argue about it.** Compute the statistic under a few
  hundred random pairings and compare. `bj_grid.py` uses the log-log correlation of census against
  modelled population: **r = 0.9052 as built against a best of 0.3776 over 500 shuffles**, and asserts
  that the real join beats every one of them. **Keep both** — the band still catches the gross failures
  a correlation is blind to, an empty unit or a wrong CRS — **but know which one is doing the work.**
- **AND WHICH ONE IS DOING THE WORK FLIPS WITH THE COUNTRY'S SHAPE — ZIMBABWE IS BENIN EXACTLY
  REVERSED.** Ten provinces, wildly uneven (Harare is 2.4M people in 872 km², Matabeleland North is
  828k in 75,025), and a grid whose vintage is one year off the census rather than ten. There the
  **band** is 0.83×–1.11× and discriminates hard — a shuffle fails a median 4 of 10 and only 0.6% of
  shuffles pass — while the **correlation** is nearly useless, because ten log-populations of similar
  size correlate by luck: r = 0.979 for the real join against a best of **0.976** over 2,000 shuffles.
  **A band is strong where the units are uneven and weak where they are alike; a correlation is the
  reverse, and it needs enough units to have a null at all.** So the rule is not "prefer the
  correlation", it is: **measure both nulls, assert on whichever the country's own shape makes
  discriminating, and say in the file which one it was.** Two lines either way.
- **AND CHECK AN ENCLAVE CITY TOGETHER WITH ITS RING, BECAUSE A BLURRED SURFACE CLOSES AND A BAD JOIN
  DOES NOT.** Six Lithuanian cities are their own municipality sitting inside the rural municipality
  named after them. Kontur reads Šiauliai city at 0.48× and Šiauliai rajono at 1.92×, which looks
  alarming per unit and is simply a footprint-based model spreading Soviet apartment-district population
  outward across an internal boundary. **Summed as a pair, all six close between 0.89× and 1.05×** — and
  a wrong join would not. Derive the pairs from the names rather than listing them, and run this wherever
  a country has cities carved out of rural districts, which in Europe is most of the post-Soviet ones.
- **CHECK WHETHER A BIG INLAND LAKE IS INSIDE THE UNITS, AND MEASURE IT RATHER THAN ASSUMING.**
  `water.py` subtracts the sea and states plainly that inland water is a known gap (§8.2c), on the
  grounds that agencies usually cut lakes out themselves. GSS does not: its districts run straight
  across **Lake Volta**, 6,045 km² and the largest reservoir on earth by surface area. **397 of 30,750
  dots, 1.29% of Ghana, were in open water**, against the 3.0% of the New York bbox that made `water.py`
  exist at all. One line of measurement finds it; nothing else will, because a dot in a lake is in the
  right unit and every total is correct. HydroLAKES is already in `../data/`. Done locally in
  `gh_geo.py` rather than in `water.py`: the gap is global, the demonstrated need is one country, and
  **the second country to need it is when the code should move.**
- **…AND A POPULATION GRID REMOVES THAT PROBLEM AS A SIDE EFFECT, BECAUSE WATER HAS NO POPULATION.**
  Kenya has Lake Turkana and a share of Lake Victoria and needed no clip at all: it is placed on Kontur
  hexes, which exist only where people do. 90 of 47,128 dots land inside a HydroLAKES polygon and most
  are the inhabited islands of Lake Victoria. **The rule: a country placed on administrative polygons
  may need the clip; a country placed on a population grid will not.**
- **A COMPLETE POLYGON COVER IS NOT GOOD NEWS ABOUT WATER.** Ethiopia's ADM3 layer has 756 polygons for
  738 counted woredas, and the 18 extras are lakes and parks cut OUT of the units, so §8.2c's problem
  never arose there. Bangladesh has 544 for 544 — a clean identity join, and precisely the warning sign,
  because no spare polygons means the agency has NOT removed the water and the rivers are inside the
  units. In the largest delta on earth that is the difference between dots on land and dots
  mid-Jamuna. **Spare polygons in an administrative file usually mean the lakes are already gone; their
  absence means they are not.** The reading to avoid is that 544-and-544 means everything is fine — it
  means the *join* is fine.
- **A FINER GEOGRAPHY CAN BE WORSE, AND §3.9's TRADE IS NOT THE ONLY TRADE.** The usual question is
  categories against geography. Indonesia's sub-district tier poses a different one — **geography
  against completeness**: 6,357 units against 492, and 18% of the parents have an incomplete child
  listing, so eighty-eight places would draw understated, one by 68%, while every national and regional
  figure still looked reasonable. **Do not take the finer tier just because it exists.** Measure it
  against the coarser one you already trust, per parent, and where it falls short prefer Ghana's answer:
  **draw the fine unit where it reconciles and the coarse one where it does not**, so the drawn tier is
  two `geo_level`s and every drawn row is still `measured`. Reaching for allocation to paper over the
  shortfall would be inventing a magnitude the source does not publish, which §14.4 forbids outright.
- **A POPULATION GRID CAN BE FINE ENOUGH AND STILL BE THE WRONG INSTRUMENT, BECAUSE RESOLUTION IS NOT
  THE ONLY TEST. THE OTHER ONE IS WHETHER IT SHARES A UNIVERSE WITH THE COUNTS.**
  [[reference_kontur_resolution_floor]] asks whether the grid is finer than the counting tier, and
  Singapore passes that easily: Kontur at 400 m over 31 planning areas. It is refused anyway (§9bp).
  Kontur counts everybody physically present; Singapore's census counts *residents*, and **1,641,590
  people in Singapore are non-residents the religion table does not cover**, many of them in worker
  dormitories. The resident population of Tuas is **70** and of Sungei Kadut **750**, so a
  presence-weighted surface puts resident dots in both. The grid was never too coarse; it was
  measuring different people. **So before reaching for the grid, ask what its denominator is and
  whether the source's own office publishes population at a finer tier on the SAME universe.** Where
  it does, that beats a modelled surface on universe, vintage and provenance at once, and Singapore is
  weighted on its census's own 332 subzone resident populations instead (r = 0.99857 against the
  religion table's own unit totals, and a correlation between two cuts of one census is *allowed* to
  be tight, unlike every Kontur check on this map). The countries most exposed to this are the ones
  with large counted-out populations: Gulf states, Singapore, Brunei, and anywhere the census word is
  `resident` or `citizen` rather than `population`.

- **AND A POPULATION GRID CAN SIMPLY BE WRONG, on its own terms, about a whole region — found with
  Eswatini 2026-09-08 (§9bq).** Singapore's grid was accurate and counted the wrong people. Kontur's
  Eswatini extract counts the right people and puts them in the wrong place: normalised by its own
  national ratio it reads **0.38x on Hhohho and 2.24x on Lubombo**, dropping 43% of the country into a
  region the census counts at 19%. The cause is what Kontur is built from. **Building footprints
  inherit where the mapping happened rather than where the people are**, and Eswatini's northern
  Lowveld sugar estates were mapped house by house in OSM while the Highveld *imiti*, the dispersed
  homesteads most Swazis live in, are barely mapped at all. In a small country one mapping campaign is
  enough to tip a whole region, so **this risk goes up as the country gets smaller, not down.**

  Three things follow, and they are cheap.

  **The per-unit band is not a formality and it must be believed when it fires.** It is the only thing
  in the pipeline that would ever have caught this; every count reconciles, every join is a bijection,
  and the map would simply have been wrong. A band failure is evidence about the grid at least as
  often as about the join.

  **Clear the boundaries before blaming them, and in that order.** Point-in-polygon for half a dozen
  towns whose region you can look up, the polygons' areas against the office's published areas, and a
  second independent spatial join. All three take ten minutes and all three passed here, which is what
  turned "our join is broken" into "the grid is wrong" rather than leaving it ambiguous.

  **Two independently built grids agreeing is what makes a modelled weight trustworthy; one grid
  agreeing with the census is not.** Eswatini is placed on WorldPop's constrained 100 m `maxar_v1`
  raster (machine-extracted footprints, not volunteered mapping) at 0.95-1.04, with the *unconstrained
  2017* raster — a different model of the census's own year — carried as a **control rather than a
  second opinion**, agreeing to within 0.037 on every region and asserted on every run. Any country
  whose Kontur band looks ugly should get the same two-raster test before anything else is suspected.
  Prefer the **constrained** release for placement even at the cost of a worse year: unconstrained
  spreads people across open country, which is the failure §8.2 exists to avoid. (WorldPop ships
  constrained rasters as **BigTIFF**, `II+\0`, and unconstrained ones as classic TIFF, `II*\0`, so a
  §5a magic check accepting only `II*\0` rejects the file the country is built on.)

### Taxonomy

- **Map to branches, not leaves** (`cz2021.py` is the model). §2.4 defers cross-source matching and
  `source_category` travels on every row, so deepening later costs nothing.
- **Only map to paths declared in `branches.py`.** Nothing validates a country mapping at build time
  except `tools/check_mapping.py <cc>` — run it. **An unmapped category is not an error anywhere
  downstream**; `countries.py` just drops the rows, so people disappear quietly.
- **Adding a branch under christianity/judaism/buddhism fails the build until it has a LINEAGE group**
  (§6.5). That is deliberate.
- **EXCLUDED and REVIEW are the deliverable**, as much as MAP is. Every arguable call gets a sentence on
  why, so it can be overturned by someone who knows better.
- **§2.4's DEFERRED MATCHING FINALLY PAID, AND IT IS WORTH KNOWING WHAT THAT LOOKS LIKE.** On 2026-09-05
  Ghana's `Other Christian` swallowed the Musama Disco Christo Church and the Twelve Apostles because no
  cell existed for them, and `gh2021.py` wrote down that a node was wanted and where its people would
  sit meanwhile. The next day Kenya arrived counting **African Instituted Churches** as 3,292,573 people,
  the node was added, and Ghana's rows needed no change at all. **The practical instruction: when a
  category has no home, say in the REVIEW note what node you would want and what is inside the bucket in
  the meantime.** That note is what makes the later fix a lookup instead of an investigation.
- **Expect the source to disagree with the tree about *where* things go**, not just what they are
  called: INEGI files Orthodox Christians under "other religions", GUS files Unitarians under
  Christianity. Follow `branches.py` and record the disagreement.
- **The same write-in string can mean opposite things in two countries, and only the place decides.**
  `animismus` in Czechia is a Western neo-animist self-description and goes to `paganism`; `Animist` in
  Sikkim is an outsider's word for a tribal religion and goes to `indigenous`. Likewise `Pagan` in
  Meghalaya is the colonial-era label for the traditional Khasi religion, not the neo-pagan revival.
  **Never map a category on its string alone — look at which units it is in first.** *Israel supplies
  the sharpest case of this yet: `Masorti` there means traditional-but-not-strictly-observant, and
  everywhere else in the Jewish world "Masorti" is the name of the Conservative movement. The two are
  `judaism.masorti` and `judaism.conservative` and they are different objects.*
- **A SOURCE THAT PUBLISHES TWO VARIABLES PER UNIT HAS NOT PUBLISHED THEIR CROSS-TABULATION**, however
  much a per-area dashboard looks like it has. CBS gives every Israeli unit a religion breakdown AND a
  household-observance breakdown — Secular 53.2%, Traditional 24.4%, Religious 12.2%, Ultra-religious
  6.4% — and it is tempting to read the second as a split of the first, because in Bene Beraq (97.4%
  Jewish, 83.8% ultra-religious) it effectively is. **It is not**: the question is asked of the whole
  population, and Umm al-Fahm is 99.8% Muslim and returns 47.2% Traditional. Multiplying the two per
  unit is a model, and in a mixed unit it attributes one group's answers to another. What Israel does
  instead is the narrow honest version — **apply the second variable only where one group is at least
  95% of the unit, and leave everyone else on the parent** — which covers most Israeli Jews without
  ever claiming a cross-tabulation nobody computed. Mark those rows `modelled` (§7) either way.
- **AN AXIS THAT IS NOT A LINE OF DESCENT NEEDS ITS OWN LINEAGE GROUP, NOT A SLOT BESIDE THE
  MOVEMENTS.** Judaism's other children — Orthodox, Conservative, Reform, Reconstructionist — are
  movements a person joins. Haredi/Dati/Masorti/Hiloni answer "how observant is this household", which
  is a different question, and an Israeli Hiloni Jew has not left Orthodoxy for a liberal movement.
  They therefore sit in their own `By observance` group rather than being interleaved, and **a country
  uses one axis or the other and never both**. The general form: before adding children to a family,
  ask whether the source is cutting it the same way the existing children cut it.
- **A CATEGORY THAT IS A PEER CAN LOOK EXACTLY LIKE A CHILD.** KNBS prints `Evangelical Churches` beside
  `Protestant`, not under it: in Kenya `Protestant` means the mainline mission inheritance (Anglican,
  Presbyterian, Methodist) and `Evangelical` the faith-mission stream (Africa Inland Church, Baptists,
  Pentecostal Assemblies of God). An Anglican there is a Protestant and a Baptist there is an
  Evangelical, **which is the reverse of the usual English usage and would silently mis-file 9.6M
  people.** Check the arithmetic — peers sum with their siblings to the total, children sum to their
  parent — and then read what the office means by the word rather than what you do.
- **A parent published BESIDE two of its own children needs the remainder emitted — and the remainder
  must exist at every level the allocation touches.** KSH gives `Katolikus` (2,886,619) and, labelled as
  subsets, Roman Catholic and Greek Catholic, but never their 77,629-person difference. Drawing the
  parent too double-counts 2.8M; drawing only the children drops 77,629. **The second half is the one
  that bites**: emitting it at the fine level alone silently deletes it at the allocation step, because
  `allocate.py` carries a fine column forward only when some coarse category lands on it — and every
  reconciliation upstream still passes. **AND THE TEST FOR WHICH CASE YOU ARE IN IS ARITHMETIC, NOT
  STRUCTURAL:** GSS publishes `Christian` beside its four Christian categories and they sum to it
  **exactly**, so there is no remainder and the parent is simply a duplicate to drop. The two look
  identical in a table of contents and differ only in whether the published children add up. **Check it
  per row, not nationally** — an identity that holds at the top and fails in one district is what a
  misparse looks like.
- **WHERE A SOURCE NAMES MANY BODIES OVER MANY UNITS, GEOGRAPHY IS EVIDENCE ABOUT LINEAGE — WITH TWO
  CONDITIONS.** Two churches planted by the same mission are still in the same provinces sixty years
  later, so provincial co-location says something a body's name cannot fake. 129 categories × 117 units
  was enough to confirm six calls made from names and overturn one (`Evangelical Christian Outreach
  Foundation`, 115,626 people, filed charismatic and actually a 1954 tribal faith mission). The
  conditions:
  1. **Normalise within the stream, not the population.** Correlating population shares measures "is
     this province Protestant" and nothing else — on that basis Bible Baptist correlates 0.68 with the
     Adventists, which is a fact about Mindanao. Use each body's share of its province's *named
     non-Catholic Christian* population.
  2. **Only act where the neighbours are classified independently of your own mapping.** Run over the
     whole undocumented tail the check disagreed with 22 of 56 calls, and almost every disagreement was
     circular: the nearest neighbours of a small unknown ministry are *other small unknown ministries
     the same file placed by name*, so the vote just counts your guesses back to you.

  **And it has a hard floor: it cannot separate categories that differ by doctrine and agree by
  geography.** Charismatic and non-denominational megachurches are both Metro Manila, so the check put
  the documented-non-denominational Christ's Commission Fellowship among the charismatics. **Use it to
  audit a mapping, never to build one.**
- **A new top-level family costs more than it looks.** `ROOT_HSL` (§6.3) is hand-authored for thirty
  roots and its indigo→magenta wedge is already at the 3.3°-apart limit, so a 31st root makes every
  other small family harder to tell apart. India's Nirankaris and Dera Sacha Sauda are real distinct
  movements and still went to `other.in`, because **a group that draws one dot should not cost the whole
  palette a degree.** Ghana's 999,319 Traditionalists went to a REGIONAL CHILD, `indigenous.african`, on
  the pattern of `indigenous.indian` and `indigenous.philippine` — **the cheap move that gets a family
  drawn without spending a root.**

### Reconciliation discipline

- **Assert what should be exact; report what cannot be.** Totals per level against the published
  national figure: exact. Categories summing to the total: exact only if the source neither suppresses
  nor rounds.
- **Where the source rounds, compute the band from the rounding** rather than picking a tolerance that
  passes. Estonia rounds to base 10, so a sum of n units is within ±5n — **and assert that every figure
  is a multiple of 10**, so the band is never applied on a false premise. Same reasoning for Canada's
  base-5.
- **AN IDENTITY COMPUTED WITHIN ONE TABLE IS A PARSE CHECK ONLY IF THE PARSE CANNOT HAVE PERMUTED THE
  TABLE.** Zimbabwe's Table 2.14 satisfies both of the obvious identities — the eleven categories sum
  to each province's total, and the ten provinces sum to the national row — and **both would still
  hold if every column had been read in the wrong order**, as long as it was the same wrong order
  throughout. The only check that crosses tables is `Male + Female == Total`, on the two sex
  breakdowns published beside the drawn one, and it is therefore the only one that could catch a
  column landing in the wrong place. Malawi found the same thing from the other direction (§9bb).
  **Ask of every reconciliation: what would a consistent permutation do to it?** If the answer is
  "nothing", it is checking the census and not the read, and a second table is needed.
- **The check that catches the bug is rarely the one you expect.** Romania's county-header misparse
  (600,861 people double-counted) was caught only because two different counties' `Păuleşti` happened to
  collide into one key. Had they not, the run would have passed. **Prefer checks that would fail
  *loudly* on a structural error, not just a lucky one.**
- **AN ALLOCATION REPRODUCES EXACTLY THE CONCENTRATIONS THAT LIVE IN THE DIMENSION IT ALLOCATES ON,
  AND IS BLIND TO EVERY CONCENTRATION INSIDE ONE OF ITS OWN CELLS.** Where a country publishes a
  variable nationally and a splitter subnationally, the arithmetic
  `SUM_g P(religion | g) x N(g, unit)` reconciles perfectly to the national totals by construction —
  which means **reconciliation says nothing at all about whether the geography is right**, and the
  output looks equally confident whether it is or not. What decides it is whether the thing being
  drawn varies *within* a cell of the splitter. Cyprus is the worked example and the one case where
  it could be measured: 2021 religion allocated on four citizenship groups, checked against 2001's
  Table 29, the only religion-by-district table any Cypriot census published. Orthodoxy came back
  right across all five districts (it is the modal answer of every group); the British Anglicans of
  Pafos came back at half strength (Britain is non-EU, so the splitter half-sees them, but non-EU also
  holds Syrians, Filipinos and Sri Lankans and the average flattens the peak); and the **Armenians and
  Maronites came back completely flat, 1.0x against a measured 1.88x and 2.07x in Lefkosia**, because
  both sit inside `Cypriots` and a group with one national profile has one profile everywhere.
  **So the error is bounded by how concentrated a group is inside a splitter cell, and that is usually
  something you can look up BEFORE deciding to draw** — no historic table required. Where one does
  exist, an older measured cut is worth parsing purely as a check even when it is far too stale to
  draw: an upper bound on the error is not a measurement of it, but it is the difference between a
  named cost and an unexamined one. `sources/cy_2001.py` is the shape of that check.
- **WHERE A SOURCE PUBLISHES THE SAME CENSUS AT TWO GRAINS, ASK WHAT THE FINER GRAIN DOES WITH A
  CATEGORY IT HAS TOO FEW PEOPLE FOR *BEFORE* WRITING THE RECONCILIATION.** The obvious check is
  that each category's sum over the fine units equals the national figure, and it is the wrong
  check wherever the fine tables suppress by **folding into their own residual** rather than by
  masking a cell. Armenia is the worked example (§9bx): each marz volume prints only the columns
  that marz has people in, so Syunik's table names five religions and Yerevan's fourteen, and an
  answer with no column in a marz is inside that marz's `Other`. **483 of the country's 515 Muslims
  are printed across four marzes and the other 32 are in the residuals of the seven without a
  Muslim column.** A per-category equality fails on a correct read; the right assertions are that
  **each unit closes on its own published total**, that **the units sum to the national total**,
  and that **each category's shortfall is non-negative and reappears in the residual**. That last
  one is the load-bearing part, because it is what would still fail if the read were actually
  wrong.
  - **And run the comparison in both directions, because the useful residue is the impossible
    half.** Folding can only make a category *smaller* at the fine grain, so any category that
    comes out *larger* is not folding. Armenia has three (`Refused to answer` +6, `Evangelical`
    +1, `Jehovah's witness` +1): two publications of one census disagreeing by 177 people,
    0.0060%. That is worth a bounded assertion rather than a silent tolerance, because the bound
    is what distinguishes an editorial difference from a real divergence later.
- **A TWO-ROW HEADER CAN PUT A COLUMN IN THE UPPER ROW ONLY, AND LOSING IT STILL BALANCES.**
  Armstat's marz tables span the religions under a *Religious belief* title in the lower header
  row and print `No religion` and `Refused to answer` outside that span, **one row higher**, at
  the far right. Read the lower row alone and every unit loses its irreligious and its
  non-answers, 66,854 people nationally — and nothing looks wrong, because what remains still
  equals the table's own printed `has a religious belief` sub-total. **A sub-total that closes is
  not evidence the row was read whole; only the unit's population is.**

### Finishing

`COMMANDS.txt`'s checklist is the authority; these are the reasons behind the steps that bite.

- **`tiles.py --countries` REPLACES the archive and `counts.json`.** Always pass every country that has
  a `dots_<cc>.geojson` — a short list silently drops the rest of the map, and this has already happened
  once.
- **`buffers.py --countries` REPLACES `manifest.json` the same way — and this one is worse, because it
  breaks only the DEFAULT view.** Since §4.2d, "overlapping dots: separate" — the default — draws from
  `data/buffers/<cc>.bin` and not from the pmtiles at all. A country missing from the buffers has
  correct tiles, a correct `counts.json` entry, a correct legend with correct totals, and **draws
  nothing**. Ghana shipped that way for half an hour. **Re-tiling and re-buffering are ONE step with two
  commands; never do the first without the second**, and never drop `--coarse` from either.
- **`country_shapes.py` is the third silent one**, and the only build step nothing else depends on — so
  nothing complains. A country missing from it is invisible to §6.2's Auto, which falls through to the
  dot tally and hands the legend to whichever neighbour has dots in frame.
- **`tools/check_mapping.py` defaults to the `geo_level` with the most units, which is wrong for a
  country whose drawn tier is more than one level.** Ghana's 272 units are 255 `district` plus 17
  `submetro`; the default reported 255 and 1.7M too few people, and nothing about the output looked
  wrong. `--level` takes a comma-separated list, and a split tier declares itself in `DEFAULT_LEVELS`.
- **Run `coverage.py` and `tools/check_palette.py` after a re-tile.** Coverage must hold for every drawn
  node (§6.12); palette separation is a property of the palette *against a country's tallies*, so it
  changes whenever a country lands (§6.3).
- **Update `sources.md`** (the row, the drawn count, a §9x entry with what generalises) **and
  `COMMANDS.txt`** (fetch, geo, scatter, the tiles line). `data/` is gitignored, so the `.md` files are
  the only record that survives.
- **COMPUTE A NEW CATEGORY'S GEOGRAPHY BEFORE WRITING PROSE ABOUT IT — FOUND 2026-09-07 WITH NEPAL.**
  Every note this project writes about a new node is a factual claim, and the reconciliation checks
  cannot see one of them. Nepal's `Bon` was written up first as trans-Himalayan — Mustang, Dolpa,
  Humla, the districts with Yungdrung Bon monasteries — which is where Bon is in general and is not
  where this census puts it. It is in **Gandaki's Gurung middle hills**: Gorkha 5.7%, Dharche 32.7%,
  against 0.09% in Bagmati. The map would have shipped a note telling readers to look at one end of
  the country while the dots clustered at the other, and **every arithmetic check would still have
  passed**, because the arithmetic was never wrong.

  This is §3.10d — *arithmetic consistency is not evidence of meaning* — aimed at the documentation
  rather than at the data, and it is the more dangerous direction, because prose is where a reader's
  understanding actually comes from. **The habit that fixes it costs about a minute**: before writing
  a word about a category, print its top ten units at each tier and its bottom one. Do it for every
  category the country adds, not only the new nodes; it is also how Cambodia's Cham belt, Nepal's
  Kirat edge and Zimbabwe's Vapostori got described correctly rather than plausibly.
- **A BLANK OR DEAD VIEWER MAY NOT BE YOUR COUNTRY AT ALL — SYNTAX-CHECK `index.html` FIRST, IT COSTS
  ONE COMMAND.** Extract the inline `<script>` blocks and run `node --check` on them; it names the
  line, where the browser console only says `SyntaxError: Unexpected identifier` with no location
  you can use. Found 2026-09-07 while verifying Myanmar: a *comment* inside the GLSL vertex shader
  had been given backticks around `` `derived` `` and `` `modelled` ``, and **the shader is a JS
  template literal, so a backtick closes it** and everything after is parsed as code. The whole
  viewer was dead — every country, not the new one — and the line immediately above the offending
  comment already said *"NO BACKTICKS IN HERE — this comment had two and broke the page"*, so it was
  the second time. **With several sessions editing `index.html` at once, the page being broken when
  you go to look at it is a normal state and not evidence about your own work.**
- **Look at the country, and look at it in the DEFAULT mode.** The panel and the legend can be entirely
  correct while nothing draws, and only a screenshot says so. Headless needs
  `--enable-unsafe-swiftshader` and must NOT have `--disable-gpu`, or the WebGL scatter layer silently
  paints nothing and reproduces the same symptom for a different reason — which is how the buffers bug
  got misdiagnosed once before it was found ([[reference_headless_map_screenshots]]). Forcing
  `merged = true; applyPaint()` over CDP switches to the plain MapLibre circle layers and is a useful
  A/B, but **a country that draws only when merged is a country that is broken.**
- **WHEN A TABLE IS A DISTRIBUTION, FIND IT BY ITS ARITHMETIC AND NOT BY ITS CAPTION.** Mongolia
  (§9bt) is published as twenty-two provincial volumes typeset by twenty-two provincial offices,
  and they agree on nothing: table numbers, caption wording, declension, column order, whether
  the aimag's own name is prefixed, whether two tables are merged into one, whether the table is
  transposed, whether both census years are printed, and whether a religion with no adherents
  gets a zero or no row at all. One volume misspells its own row label and one spells `ХҮН` as
  `ХУН`. Every caption regex written for that country was wrong within three files. What works
  instead is to scan pages and accept the one whose numbers satisfy an identity only the wanted
  table can satisfy — two shares summing to 100.0, five shares summing to 100.0, age bands
  summing to their own printed total. This is [[reference_pdf_table_geometry]]'s "anchor the
  header" taken one step further, and it is strictly safer for the reason that matters: **a wrong
  page fails the identity, whereas a wrong caption match returns numbers.** It also needs no
  advance knowledge of any of the twenty-two differences.
- **A NATIONAL REPORT WITH A RELIGION CHAPTER AND NO GEOGRAPHY IS NOT EVIDENCE THAT THE OFFICE
  PUBLISHES NONE.** Mongolia's 2020 and 2010 national reports both carry a chapter called
  CITIZENSHIP, ETHNICITY AND RELIGION and both give religion by sex, age and ethnicity only. The
  sub-national tables exist, in twenty-two separate per-province volumes on a static host nothing
  links to. Where a statistics office devolves publication to its provinces, "the national report
  stops at the nation" says nothing about the country, and a session that stops there looks
  thorough while being wrong.
- **A RETIRED CMS DOWNLOAD HANDLER IS A FILENAME CATALOGUE FOR THE STATIC HOST THAT REPLACED IT.**
  `1212.mn/BookLibraryDownload.ashx?url=<filename>` now 404s on every path, but its links are
  archived in bulk, and that `url=` parameter is exactly the filename on the index-less
  `downloads.1212.mn` that replaced it. A Wayback CDX sweep of the OLD dynamic route therefore
  enumerates the NEW static one. Three Mongolian aimag volumes whose names share nothing with the
  other nineteen were found this way after 273 guesses at the pattern all 404'd. Related to
  [[reference_cms_download_id_sweep]] and [[reference_dead_stats_office]].
- **A PDF CAN HAVE A TEXT LAYER FOR ITS PROSE AND PICTURES FOR ITS TABLES, and that combination
  reads as a working file.** A whole-file scan is obvious the moment anything is extracted.
  Darkhan-Uul's Mongolian census volume is the nastier case: captions and paragraphs are real
  text, so a parser locates the table and reports its page number, and only the numbers are
  absent. The one-line tell, worth running on any volume that "finds the table but reads no
  rows", is `len(page.get_images())` against the count of parsed data rows — nine images and two
  rows means stop.
- **DO NOT LOCATE A YEAR COLUMN BY SCANNING A PAGE FOR YEAR TOKENS**, and assert a plausible
  RANGE rather than only a sum. Both are Mongolian scars and both produced numbers instead of
  errors. One volume's caption ends `..., 2010 ОН, 2020 ОН`, wraps, and so begins a line with
  `2020` further left than the `2010` above it, which convinced an x-position comparison that the
  columns were reversed and made it read 2010 as 2020 — undetectable downstream, because the
  shares barely moved between the censuses. Separately, an appendix table printed its own
  continuation block lower on the same page, so every unit appeared twice and a plain dict
  assignment took the second, reading four age columns as a total and three child bands; the
  row-total identity passed either way because both blocks are internally consistent, and only
  the fact that the result went NEGATIVE gave it away.

**A SWEEP'S NEGATIVE IS A VERDICT ON ONE PUBLICATION SERIES, NOT ON A COUNTRY — Botswana,
2026-09-08, §9bu.** §11p closed Botswana with *"religion crossed with language and not with
geography"*, which is a true and careful statement about the **2022** census: it asks the
question, cross-tabulates it four ways, and publishes no subnational table in any of its five
volumes. The **2011** census has the same property at national level and the opposite property
one tier down, because Statistics Botswana issued a per-district *Selected Indicators* booklet
and **all eighteen print religion by named village**. Nothing in a report-set sweep finds that,
because the booklets are not part of the census report set.

- **The tell is in the negative itself.** *"Crossed with language and not with geography"* means
  the variable was asked, coded and tabulated. An office that cross-tabulates religion four
  ways has it in the microdata, and the only open question is which of its publications carries
  the place. Compare *"the census does not ask"*, which is a fact about the country. **Only the
  second kind of negative closes anything.**
- **So the probe is: does this office publish a per-district or per-province SERIES about
  anything at all?** Malawi's religion table was in the main report (§9bb), Benin's was in a
  per-department booklet (§9ai), Laos's was on a data platform that outlived its own atlas
  (§9bk), Botswana's is a per-district booklet. **Three of those four are not the census
  report**, and a sweep that enumerates report sets will keep missing them.
- **And an older census is a different publication programme, not just older numbers.** Offices
  change what they print far more than they change what they ask.

**CHECK THE SHORTFALL PER CATEGORY, NOT JUST OVERALL.** When part of a country cannot be drawn,
the natural summary is one number — Botswana draws 93.7% of its national total because two
district booklets were never published. But **Badimo draws to only 88.2% of its own national
figure**, so the missing districts are more traditional than the country, and the map understates
exactly the category the country is most worth drawing for. It costs one loop against the
oracle's national row and it changed what `note_public` had to say. A country that draws 94% of
its people does not draw 94% of everything.

**READ THE NUMBERS BEFORE BELIEVING THE CAPTION.** [[reference_pdf_table_geometry]] says render
the page before blaming the parser; this is its companion for a table that parses fine and is
labelled wrongly. In one booklet series both halves of a count/percentage pair were captioned
`(%)` while the first held the counts, and a religion table was captioned *"Number of people by
marital status"*. **Anchor on a column HEADER, and tell counts from percentages on the values.**
The same series put the row-total column first in ten booklets, last in six and nowhere in one:
detect that arithmetically (the total column is the one equal to the sum of the others, on every
row), because assuming a width silently shifts every category by one and then reconciles against
nothing. **A per-district series is as many typesetters as it has districts.**


**A BOT WALL AND A TLS FAILURE LOOK THE SAME FROM A SCRIPT AND WANT OPPOSITE FIXES** — found
2026-09-08 on South Africa, and it had already cost a country once. `sources.md` §11b closed
South Africa partly on *"behind a DataFirst account"*, which was wrong: the census table was
open all along and a scripted client had simply failed to fetch it. §11ag then recorded that
`statssa.gov.za` is behind Imperva and that plain `curl` returns a 212-byte
`_Incapsula_Resource` stub. Both halves are half right, and the distinction is in the exit
code:

- **`curl` exit 60, "SSL certificate problem", nothing downloaded.** That is the TLS chain,
  not a wall. The host presents a self-signed intermediate and curl gives up *before it sends
  the request*, so the server never saw you. Relax the certificate check and the identical
  URL returns the real file at full size. This is what `cs2016.statssa.gov.za` does, and its
  PDFs are not protected in any way.
- **HTTP 200 with a kilobyte of HTML.** That is the wall. No client-side flag helps, and a
  browser User-Agent does not either.

The two are told apart in one command and the wrong diagnosis is expensive in both
directions: reading a chain failure as a wall abandons an open file, and reading a wall as a
chain failure sends you round a retry loop. **An office can also be walled on its HTML and
open on its files at the same time**, which is exactly South Africa: its `?page_id=` listings
are unreadable for curl *and* WebFetch, while every PDF underneath them fetches cleanly. So
the method there is to find the file URL some other way and never try to read a listing.
Companion to [[reference_dead_stats_office]], which is about the same confusion one layer up.

**MATCHING CATEGORY LABELS ARE NOT EVIDENCE OF A SHARED ANSWER SET, and there is a cheap test**
— found 2026-09-08 on South Africa, and it is §3.1a with a way to *detect* it rather than only
a warning. Stats SA publishes religion twice, in Census 2022 and in Community Survey 2016, over
category lists that match word for word. That makes a §3.4 rescale look safe. It is not:

| | CS 2016 | Census 2022 |
|---|---|---|
| Islam | 1.62% | 1.60% |
| Hinduism | 1.02% | 1.06% |
| **No religious affiliation** | **10.9%** | **2.9%** |
| **Traditional African religion** | **4.5%** | **7.8%** |

**Sort the categories by how unambiguous the answer is, and look at which ones moved.** The
two nobody is unsure about are stable to a hundredth of a point; the ones whose boundary
depends on how the question is put move by factors, and in opposite directions. Six years
cannot do that, so it is the instrument. Where a shared basis is real, the *fuzzy* categories
move and the sharp ones move with them; where it is not, the sharp ones hold still and the
fuzzy ones swing. A label-level diff shows none of this and will report the two lists as
identical.

**AN OFFICE THAT TABULATES EVERY VARIABLE BUT ONE AT A FINE GEOGRAPHY HAS MADE A DECISION** —
found 2026-09-08 on South Africa. In the Census 2022 provincial profiles, population, density,
age, population group, marital status, birthplace, education, dwelling, tenure and water are
each tabulated *"by district and local municipality"*, and religion alone is province-only.
Stats SA's own keyless dissemination API serves 24 topics down to Main Place and religion is
on none of them. When the pattern looks like that, **stop searching the published reports and
go and price the microdata**, because the omission is deliberate and no further report will
have it. The corollary is the cheerful one: a variable that is *missing at every tier* is
usually just unpublished, while a variable that is coarse *while its neighbours are fine* is
being withheld, and those two want completely different next moves.

**A CITATION IS A FIGURE, AND NOTHING DOWNSTREAM CHECKS ONE** — found 2026-09-08 on South
Africa. The nine CS 2016 provincial profiles carry report numbers that do not run in province
order: Western Cape is 03-01-07 and Mpumalanga is 03-01-13, which is the number a code-order
guess hands to Western Cape. Three of nine were guessed wrong on the first pass and every
check in the pipeline still passed, because a report number lives in the CSV's `note` column
and nothing reconciles against it. **If a source's own identifier is going into the record,
read it out of the file and assert it**; every profile carries it in the running header, so
it cost four lines. This generalises past report numbers to any provenance string a build
types rather than reads.

**AND A HEADLINE MULTIPLE IS A FIGURE TOO.** South Africa's write-up claimed its African
Instituted Church count *"more than doubles"* what that node held; a reviewer agent summed the
other six countries out of `countries.py`'s own `counts()` and the true figure is 1.21x. The
claim had been written from an impression of the node being small, and it survived into four
files before anyone derived it. **Anything of the form "X times", "the largest" or "the
sharpest" is a computation, and it should be run.** In the same pass, *"the sharpest
denominational gradient in the country"* turned out to be the second sharpest, and its
neighbouring entry called the actual first one *"the second"*.

### A CLOSURE RECORDS THE TIER IT TESTED, NOT THE COUNTRY — FOUND 2026-09-08 with Finland

§11k closed Finland, Norway, Denmark, Iceland and Sweden in one move: *"the register tier is a
mirage and it fails the same way four times"*, with the instruction *"do not re-scout the
Nordics without a specific new release to point at"*. **Every word of that is true about the
register and none of it is true about the country.** Finland is in all seven usable ESS rounds
with `region` at NUTS 3, which is a finer geography than four drawn countries have, and it was
sitting there the whole time.

**The mechanism is worth naming because it is not carelessness.** §11ai went looking for
survey routes into the Nordics and listed eight countries — Norway, Sweden, Denmark, the
Netherlands, Belgium, Latvia, Ukraine, Luxembourg. Finland is missing from that list, and the
reason is that §11k had already closed it. **A country recorded as closed stops appearing in
the candidate lists that later sweeps are built from**, so the closure protects itself: the one
pass that would have caught it was the pass that had already crossed it off. That is why the
fifth Finland-shaped reversal in one day was still available to find.

So, two things to write into any negative:

1. **Name the tier.** *"Statistics Finland publishes no religion below the country"* is a
   finding. *"Finland is out"* is not, and the difference is invisible six sections later when
   somebody greps for a country name and reads the verdict rather than the evidence.
2. **A closure is a lead for the OTHER tiers**, not a lead for nothing. A state that keeps a
   register detailed enough to close the census question is a state whose survey programme is
   usually well funded and well sampled, which is the opposite of the inference the closure
   invites.

**And close on what the instrument measures, not on whether you could reach it.** The
strongest version of Finland's closure is not *"the table is national only"* — it is that a
register counts formal membership of a registered community, a records status you leave by
filing a form, and that is a different quantity from affiliation. Finland is where the size of
that difference is finally visible: **62.24% of Finns are on the Lutheran church's register and
45.00% say they belong to it.** §3.9a already had the principle from Germany; Finland is the
measurement.

**The half of that comparison worth carrying is the half that goes the other way.** The
register puts Finland at 0.48% Muslim and 1.03% Orthodox; this map gets 1.67% and 1.86%.
A register only sees members of a *registered congregation*, so it undercounts precisely the
groups with no reason to join one — which means **"the register is exact" is true about its own
quantity and false about the country**, and a build that reaches for a register as the better
source should ask which groups it is structurally blind to before preferring it.

## 13. Things deliberately not being done

- **No world-history time slider.** cityhistory is that map. Religion over time at this granularity is a
  different and much worse-sourced problem, and mixing them would sink both.
- **No adherent-count aggregation across bases** (§3.1), however tempting the coverage would be.
- **No node invented at ingest** (§2). Unmapped source categories go to a file and wait.
- **No log scale** (§4.1), and no confidence expressed in colour (§7).

## 14. What this map could do harm with — ASSESSED 2026-09-04

Not a legal opinion. It is the standing assessment, so that nobody has to work it out from scratch, and
so the line is drawn before a country is half-built rather than after.

**IF ANY OF THIS COMES UP FOR REAL, RAISE IT WITH ANITA RATHER THAN DECIDING ALONE.** That is an
explicit invitation, not a fallback: a judgement call about who gets drawn and how finely is hers to
make, and flagging one costs a message ([[feedback_flag_ethics_for_discussion]]). It does not need to be
a crisis first — "this country's situation looks like §14" is enough to start the conversation. The same
goes for a source whose terms are unclear, or a group whose safety the resolution might affect.

**The current rules, after §14.5 and §14.9 amended §14.4** — the full reasoning is below, and read it
before applying any of them to a new case:

1. **Never estimate a magnitude a source does not publish; refine placement only.** This is the one that
   has never moved, and it is the real limit.
2. **For a persecuted group, no resolution finer than the state's own publication** — a limit, not a
   target.
3. **Ethnicity may derive religion where the ethnic category is itself religio-ethnic**, at no finer
   geography than the ethnicity is published at (§14.5, which withdrew §14.4's blanket ban). **The test
   is per-group and must be re-applied, never inherited from §14.5's illustrative list** (§14.6).
4. **A religiously mixed group is a preference against, not a prohibition** (§14.9). Fractional shares
   are permitted; prefer a better source where one exists; mark it in §7 either way. **And the map may
   run the model itself rather than only consuming a published one** (§14.10) — the magnitude must
   still be the host state's, the coefficients documented, and the output checked against something
   independent.
5. **Keep the measured / derived / modelled distinction visible to the reader** (§7a). It is the
   difference between "counted", "inferred" and "we do not know", and the main thing standing between
   this map and the genre in §14.2.
6. Jurisdictional detail changes and none of this is legal advice. If this ever becomes commercial or
   draws institutional attention, that is a lawyer's question — and §14's opening line applies well
   before then.

### 14.1 Where the project stands now

Everything ingested is aggregate, published, and lawfully obtained — ASARB is free to download, US
Census and ACS products are government works, CES is CC0. The finest unit anywhere is a census tract of
about 4,000 people and much of §8.4 is really PUMA-resolution, about 100,000. No individual is
identifiable in anything here, so no data-protection regime is engaged. `sources/us_pew.py` scrapes, and
is the one input whose terms are worth a second look rather than an assumption.

PL 94-521 bars the **Census Bureau** from asking about religion. It does not restrict anyone else from
estimating it, and §8.4 is not in tension with it.

### 14.2 The three real risks, ranked

1. **§8.4 is substantially a race map wearing religion's labels.** Ethnicity is the strongest input, so
   in many places the pattern drawn IS the ethnic pattern relabelled. Modelling that correlation is
   ordinary demography — Pew, PRRI and Brandeis all do it — and the danger is presentational rather than
   methodological. If a reader takes the inference for an observation, the map quietly teaches that
   every Black neighbourhood is Black Protestant and every Mexican one Catholic, which is false about
   individuals and increasingly false about groups. **The about-panel text saying so is load-bearing,
   not boilerplate.**

2. **Getting a community wrong is itself a harm, and the failure is asymmetric.** Drawing Borough Park
   as 58% Catholic did not merely mislead, it erased the most visible Jewish neighbourhood in America
   (§8.4). In the other direction, overstating a minority somewhere feeds a genre that already exists:
   "Muslim population maps of Europe" are a staple of the far right. That is not a reason not to draw
   the map. It is the reason the honesty of the labelling matters more here than on an ordinary data
   map. (It is also why several palette decisions — §6.3a-i's rejected green, §6.14's Philippine and
   Māori rows — are made on meaning rather than on measurement.)

3. **Targeting.** A neighbourhood-resolution map of where Haredi Jews or Muslims live is in principle
   useful to someone with bad intent. **The marginal risk is genuinely low where the map REFLECTS what
   is already public** — Borough Park's character is visible from the street and in every guidebook. **It
   rises where a map would REVEAL**: a small, dispersed or deliberately unadvertised minority. Some luck
   helps here, in that the model is least confident about exactly those groups, but luck is not a
   policy.

### 14.3 Other countries, and the distinction that actually matters

**What §8.4 did is much weaker than "estimating religion where it is not recorded", and the difference
is the whole argument.** The US has a real count at county level; §8.4 changed only WHERE INSIDE a
county the dots sit, and every county total is still exactly ASARB's. Nothing was invented, only placed.

France has no count at all. Estimating religion there would mean inventing the magnitude as well as the
location, most plausibly from surnames, origin or nationality — far less accurate, and much closer to
the thing France's ban on ethnic statistics exists to prevent. **The rule that falls out: never model at
a finer resolution, or a stronger claim, than the source publishes its magnitude at.** A country with no
religion data is a country this map does not draw.

On the law, briefly: France's prohibition and GDPR Article 9 both bite on **processing personal data**.
An estimate about an area, built from published aggregate tables, is probably outside them; building the
same model from individual-level microdata — the French equivalent of what CES supplied for §8.4a — is
squarely inside. Germany is the opposite case and publishes religion itself, for church tax.

**The genuinely dangerous list is ethical rather than legal**, and it is short: China, Myanmar, Iran,
Pakistan, and increasingly India, which is already drawn. A fine-grained map of where a persecuted
minority lives is a different object from a map of American denominations, whatever any statute says.

### 14.4 The rules that follow

Superseded in part — the summary at the top of §14 is the current list. The original four bullets were:
never estimate a magnitude a source does not publish; **never estimate religion from ethnicity in a
country with no religion count** (withdrawn by §14.5); for a persecuted group, no resolution finer than
the state's own publication; and keep the measured / fitted / authored / uniform distinction visible.

### 14.5 Ethnicity may derive religion where the category is religio-ethnic — DECIDED 2026-09-05

**§14.4's second bullet is withdrawn.** It was blunter than the reasoning in §14.3 it was supposed to
follow from. What replaced it, at the time:

> **Ethnicity may derive religion where the ethnic category is itself religio-ethnic, at no finer
> geography than the ethnicity is published at, and never where the group is religiously mixed.**

Anita's call, prompted by §11f in `sources.md` — scolbert08's map is finer than this one in exactly
three countries and got there by this route in all three. *(The final clause was itself downgraded by
§14.9 from a ban to a preference.)*

**The load-bearing argument is about the adversary's information, not about the map's obscurity.** For
Uyghurs and Hui the source of danger is the Chinese state; the only data this map would use is the
Chinese state's own published county tabulations; and **no compilation of a government's published
tables about its own territory can tell that government something it does not have.** That is not a new
principle — §14.2's third risk already draws the line at **reflect vs reveal**. Chinese census ethnicity
is public, national and tabulated by the state. It is the reflect case, stated in §14.2 before anyone
went looking for it.

**The argument deliberately NOT relied on is the audience one** — *"only map nerds will see it"*. It was
raised and it is probably true today. It is not written into the rule, for two reasons: it is a claim
about the present state of a thing this project is actively trying to change (a public viewer and
printed posters both exist), and §14.2 already records that "Muslim population maps of Europe" are an
established far-right genre, which is the audience that finds such a map without being invited. **A rule
that depends on nobody looking stops being true at the moment it matters.** The reflect-vs-reveal
argument needs no such assumption.

**What §14.4 got wrong: two operations, not one.**

| | example | what the derivation does |
|---|---|---|
| **religio-ethnic** | Hui, Uyghur, Kazakh, Dongxiang, Salar, Kyrgyz, Tajik, Uzbek, Bonan, Tatar → Muslim. Tibetan, Yugur, Monba, Pumi → Tibetan Buddhist. Dai → Theravada. Russia's Tatar, Bashkir, Chechen, Avar, Dargin, Kumyk, Lezgin → Muslim; Buryat, Kalmyk, Tuvan → Buddhist | almost nothing. The group boundary and the religion boundary are the same historical object; the coefficient is near 1 and is **documented rather than fitted**. A census that asked would return roughly the same numbers. ~35M people in China. |
| **religiously mixed** | Han → folk religion or irreligious. Yoruba. Nigeria's middle belt. | all the work. The output is a model whose error bars are wider than the differences it draws, and §14.2's first risk applies in full. |

**The test is not "is ethnicity correlated with religion here"** — it is almost everywhere — **but
whether the ethnic category was constituted religiously.** Where it was, naming the religion adds no
information the ethnonym did not already carry, which is precisely why it is safe and also why it is
honest.

**§14.4's other bullets survive, and this satisfies them.**

- *"Never estimate a magnitude a source does not publish."* The magnitude published is the ethnic count,
  and the claim is that for these categories the religious count is the same object to within a few
  percent. That is still an estimate and should be called one — but it is an asserted identity with a
  documented history, not a coefficient fitted on a proxy.
- *"For a persecuted group, no resolution finer than the state's own publication."* Satisfied by
  construction: the resolution IS the state's own publication, because the state's table is the only
  input. **This is the rule that made Egypt a refusal** — there the state collected and withheld, so any
  resolution at all was finer than it published. China publishes.
- *"Keep the measured / derived / modelled distinction visible."* **This was the real blocker.** §7 was
  reversed on 2026-09-04 and the desaturation went with it, so a derived dot rendered identically to an
  ASARB-counted one. **No derived country shipped until §7a existed**, because adding a country whose
  every dot is derived, in that state, would have put the map's largest unmarked claim on screen
  immediately after the marking was removed.

### 14.6 China is built, and it trims §14.5's own list — BUILT 2026-09-05

`sources/cn.md`, `sources/cn_geo.md`, `taxonomy/cn2000.py`. **30.67M people of 1.33 billion, 2.3% of the
country, `derived` on every row.** §14.5's precondition was satisfied first: §7a's non-colour confidence
mode existed before this shipped, and **China is the first country on the map that reads 100%
not-measured**, so `inferred dots: hidden` empties it completely. That is the honest test of the country
and it is worth performing rather than describing.

**Two of §14.5's own religio-ethnic examples do not survive contact with their size.** Its table listed
*"Tibetan, Mongol, Tu, Yugur, Monba, Pumi → Tibetan Buddhist"*. **Mongol and Tu are not drawn** —
Anita's call:

> Mongols were 5.81M in 2000, **more than Tibetans (5.42M)**, so following §14.5's list would have made
> Inner Mongolia rather than Tibet the largest block of Vajrayana dots in China. The test §14.5 sets is
> that the coefficient be *"near 1 and DOCUMENTED rather than fitted"*. For Tibetans, Uyghurs and Hui
> that documentation is everywhere. For Mongols there is nothing comparable to point at, decades after
> the Gelug monastic system they would have been counted through was dismantled, and the surveys that
> exist put a large share at no religion. Tu (241k) goes with them for consistency.

**The general lesson is that §14.5's test is per-group and has to be re-applied, not inherited from the
list.** The list was written to illustrate the rule, and it was read as though it were the rule. **A
group can be religio-ethnic in its history and religiously mixed in its present, and the second is what
the map draws**; the size of the group decides how much that costs. `Pumi` is the same problem left in
on purpose at 34 dots, flagged in `cn2000.py`'s REVIEW rather than quietly kept.

**The Han stay out**, per §14.5's open question and Anita's confirmation. What that costs is worth
naming: China renders as a populated west and a nearly empty east, and §6.12's machinery is doing more
work here than anywhere else on the map. The country note says outright that the blank is an absent
question rather than an absent belief, and that Chinese folk religion, Buddhism, Daoism and tens of
millions of Christians are all inside it. *(§14.7 decides to fill it.)*

**And the resolution ceiling was taken as a ceiling.** The 2000 census tables reach **township** —
roughly 40,000 units — and county is what is drawn, because §14.5's *"no finer than the state
publishes"* **is a limit and not a target**. A township map of Uyghur settlement is a different object
from a county one. Anita: *"i dont think most people will be looking super closely at this area of china
tbh"*, and county was chosen anyway.

### 14.7 The Han get drawn after all, as a grey residual — DECIDED 2026-09-06, NOT BUILT

§14.6 left the Han out and said the cost was that China renders as a populated west and an empty east.
Anita's decision is to fill it — not by resolving the folk-religion / irreligious boundary, which is
still the thing nobody can draw, but by refusing to:

> **Draw Han Buddhism from CFPS at province level, and put everything else Han into ONE node that says
> outright that we do not know what it is.**

*"i think 97% grey seems fine, and honest."*

**Why this is not the banned operation.** §14.5's second row forbids deriving religion for a religiously
mixed group, and the Han are the canonical case. This does not derive anything for them: Buddhism comes
from a survey that ASKED (basis `self_id`, tier `modelled`, the Russia shape), and the residual asserts
nothing at all. **The undrawn boundary stays undrawn; what changes is that the people are visible as
people rather than as absence**, which is what §6.12 has wanted since it was written.

**The node it needs already exists.** §14.7 specified a new member of §6.3a's grey family — *not*
`unrecorded`, which means "the register only ever saw church-tax bodies" and is Germany-shaped — and
Anita asked for it **slightly yellow**. It was built for Vietnam instead (§6.3a-ii): `unknown`,
"Religion unknown", whose admission test covers China's case unchanged. **So when the Han residual is
built it needs no new node, no new colour and no new decision**; it needs the CFPS Buddhist share
subtracted and the remainder mapped to `unknown`. Vietnam is now the worked example of what this section
predicted: the country reads as populated rather than empty, and nothing on screen claims to know what
the grey believes.

**What it will cost, so nobody is surprised by it.** The residual is roughly 900M–1B people — about a
million dots at 1:1,000, more than India, making China the largest country on the map and roughly 97% of
it one colour. That was put to Anita before the decision and is the decision.

**And it swallows Christianity**, deliberately. The 40–70M Chinese Christians sit inside the residual
rather than getting a layer of their own, because the magnitude is disputed by a factor of two and every
instrument misses house churches. Pulling Christianity out later as its own province-level layer
disturbs nothing else, and is the obvious next move if the numbers ever firm up.

**Blocked on CFPS, which may never arrive** (`sources.md` §6a): the account was applied for on
2026-09-05 and should be assumed refused unless Anita says otherwise. **Do not build the residual
first** — it is defined as what is left after Buddhism, so building it before the Buddhist share exists
means building it twice.

**And the working arrangement is a constraint on the code, not a preference.** CFPS forbids sending its
data or derived datasets to AI tools ([[reference_cfps_terms]]). So the aggregation script and the
adapter are written blind, Anita runs them, and the reconciliation checks are assertions in the script
that she reads. **This will be the only country here with no check the author saw.**

### 14.8 §14.5 had a gap between its permission and its ban — FOUND 2026-09-06, closed by §14.9

Asked whether more of China's minorities could be drawn, and the answer turned out to be about the rule
rather than about sources. **§14.5's permission and its ban were not complements.** The ban was *"never
where the group is religiously mixed"*; the permission was *"where the ethnic category is itself
religio-ethnic"*, tested by *"whether the ethnic category was **constituted** religiously"*.

**The southwestern mission peoples fall between the two.** Lisu (702,839 in 2010, and 73% of Fugong
county), Jingpo, Derung, Nu, parts of Va and of the northwest-Guizhou Miao are widely reported as
overwhelmingly Christian since the Fraser and Pollard missions of the 1900s–1920s. They are therefore
**not** religiously mixed, so the ban did not reach them — but Lisu identity predates the missions by
centuries, so the category was not *constituted* religiously and the permission did not reach them
either.

**The point that matters, because it will come back: "it is a proxy" is not the objection.** Every
derivation in §14.5 is ethnicity standing in for religion, Hui included. What the section actually cares
about is two things its test bundles together — *"the coefficient is near 1 and is documented rather
than fitted"*, and *"naming the religion adds no information the ethnonym did not already carry"*.
**"Constituted religiously" is a historical shorthand that guarantees both at once**, and for the Hui it
is definitional. For the Lisu the shorthand and the substance come apart: disclosure is satisfied — the
input is the state's own published table and Nujiang's Christianity is not a secret — and accuracy might
be too, **but only by measurement, never by definition**, and the only source publishing
per-people-group Christian shares for China is Joshua Project, an evangelical missions organisation.
That is the interested party.

**If the rule is ever widened, the clean form is a SEPARATE clause with its own evidentiary bar** — "or
where an independently measured share exceeds some threshold, from a source with no stake in the answer"
— rather than stretching "constituted religiously", which would blur the one clean test the section has.

**Revisiting is cheap**: Lisu, Jingpo, Derung, Nu and Va are already in `cn.csv` at county level, joined
and rescaled. Permitting them is a few lines in `taxonomy/cn2000.py` and a rebuild — no new fetching and
no new join. §14.9 resolves this by implication; see its last bullet.

### 14.9 The mixed-group ban is downgraded to a preference — DECIDED 2026-09-06, and it closes §14.8

**§14.5's ban is withdrawn as a ban.** What replaces it, in Anita's words, is a ranking rather than a
prohibition:

> **Fractional shares over a religiously mixed category are permitted. Prefer a better source where one
> exists; where none does, a modelled share is an acceptable input — it is modelled, and §7 is what says
> so to the reader.**

Prompted by §11l in `sources.md`. The occasion was UCIDE's Spanish study, which publishes Muslim
population for all 52 provinces by applying per-nationality shares to the *padrón* — and does so over
nationalities it names as mixed: Nigeria 50%, Guinea-Bissau 43%, Ivory Coast 39%, Cameroon 21%, Togo
14%. Under the old wording those five rows disqualified the table.

**Why the old ban was too strong.** §14.5's own reasoning was never about mixedness; it was about
*reflect vs reveal* and about a coefficient that is **documented rather than fitted**. A mixed group
fails neither. Nigeria at 50% is not a secret and not a guess — it is a published, sourced national
share, and applying it to a padrón count is arithmetic on two public numbers. **What mixedness actually
costs is precision, not legitimacy**: a 50% coefficient carries twice the error of a 100% one and none
of the extra disclosure. Precision is what §7's confidence rendering exists to communicate, and the map
already draws far weaker things — Vietnam's residual, §14.7's Han grey — and says so.

**The preference clause is the part that does work.** "Fine" is not "equivalent". Where a country has
both a survey and a nationality model for the same group, the survey wins; the nationality model is what
you reach for when the alternative is not drawing the group at all. In Spain both exist and **the model
is the better one**: CIS puts all non-Catholic religions at 3.2% and the confessional counts put them
near 8%. So the ordering is not "measured beats modelled" mechanically — it is **"whichever is less
wrong, argued in `sources/<cc>.md`, and marked in §7 either way."**

**What this does to the rest of §14.**

- **§14.4's first bullet still stands and is the real limit.** *Never estimate a magnitude a source does
  not publish.* Consuming UCIDE's province table is not covered by this section at all — UCIDE published
  the magnitude; the map is a reader of it. **This section is about the case where the map itself
  applies the shares.**
- **The distinction §14.9 does not settle, and it is worth settling before it is used:** whether the
  project may run its own fractional-share model, or only consume a third party's published one. Every
  case now in hand is the second kind. The first — taking ISTAT's foreigners by comune and multiplying
  by origin-country composition ourselves — is a larger claim, because nobody but us has stood behind
  the coefficients. **Raised, not decided.**
- **§14.8 is unparked and resolved by implication, and this is the consequence most worth noticing.**
  §14.8 parked Lisu, Jingpo, Derung, Nu and Va because the only per-group Christian shares come from
  Joshua Project — *"the interested party"*. This section accepts UCIDE, which is the interested party
  for Spanish Muslims by exactly the same test. **Consistency says the southwestern mission peoples are
  now permitted too**, at county, from a documented share, on the modelled tier. That is five China
  nodes and a rebuild, and it follows from this decision rather than being part of it — so it is flagged
  here for Anita and left unbuilt.

### 14.10 The project may run its own fractional-share model — DECIDED 2026-09-07, and it closes §14.9's open question

**§14.9 raised this and deliberately left it open:** *"whether the project may run its own
fractional-share model, or only consume a third party's published one. Every case now in hand is the
second kind. The first … is a larger claim, because nobody but us has stood behind the coefficients.
Raised, not decided."*

**Anita's call, 2026-09-07:**

> *"i think fractional-share models are actually seeming increasingly reliable and like that we need to
> rely on them in order to fill out more of the map. i think it's okay."*

**Permitted. The map may apply the shares itself.**

**The first thing to notice is that it had already happened, twice.** §14.9 described the permitted case
as consuming UCIDE's published province table — and by the time it was written, Spain's *other* half and
the whole of Greece were already Eurostat citizenship counts multiplied by Pew compositions, which is
the project's own arithmetic and nobody else's. The distinction was violated by the build that prompted
it. So the real choice was between withdrawing Greece and writing the permission down, and it was never
going to be the first.

**Why the publisher was the wrong thing to key on.** §14.9's own test is that a coefficient be
*documented rather than fitted*. Applying Pew's published national composition for Morocco to Eurostat's
published count of Moroccan citizens in Seine-Saint-Denis is arithmetic on two public numbers; UCIDE
doing the identical arithmetic does not make the coefficient better documented, it only moves who typed
it. **The publisher is not the evidence.** And the selection effect runs the wrong way: the third
parties who publish these tables are UCIDE for Spanish Muslims, CESNUR for Italy, Joshua Project for
China's people groups — every one of them a body with a stake in the number. *"Only consume a published
one"* would have preferred the interested party to the neutral source, which is the opposite of what
§14.9 wanted.

**What still binds, and it is rule 1, which has never moved.** *Never estimate a magnitude a source does
not publish.* The magnitude in every one of these builds is the host state's own count of its own
residents, at its own geography. **The model distributes a published count across categories; it never
creates people.** A fractional-share model that invented the denominator would be a different object and
is still forbidden.

**So the whole test is now four conditions, and they are the ones §14.9 already stated:**

1. **The magnitude is published** by the host state, at the geography drawn or coarser (rule 1, §14.4).
2. **The coefficients are documented and attributable**, not fitted to make the answer come out. §9z's
   Albanian case is the worked example of the discipline: the adjustment that felt more careful was the
   one that would have been the error.
3. **It is `modelled` in §7**, and `note_public` says in plain words what the model cannot see —
   conversion, lapse, the second generation, and the fact that it stops at the passport.
4. **Prefer a better source where one exists.** §14.9's preference clause survives whole. This section
   changes who may hold the pen, not the ordering.

**And one thing both builds already do that is worth writing down as the fifth:** *say what the output
was checked against.* Greece's model was believable because it landed at 5.08% Muslim against Pew's
independent 5.12% for the country, from inputs that did not include Pew's Greece row. **An independent
check on the output is what separates a documented coefficient from a merely plausible one**, and a
model that has none is not forbidden — it is required to say so.

**What this does not decide.** It says nothing about whether a given country should be drawn. §14.3's
France paragraph, §14.5's per-group test and §14.2's three risks all still apply on their own terms, and
a country that passes this section can still fail those. **§14.8's southwestern mission peoples are
unaffected** — they were unparked by §14.9 already and remain unbuilt.

### 14.11 France is drawn, and §14.3's paragraph about it no longer describes the route — DECIDED 2026-09-07

**§14.3 uses France, by name, as the worked example of the move this project does not make:**

> *"France has no count at all. Estimating religion there would mean inventing the magnitude as well as
> the location, most plausibly from surnames, origin or nationality."*

`sources.md` §11l reached the same verdict a day before Greece was built — *"France is not a gap in the
sweep; it is the boundary the sweep is drawn against"* — and it is worth being precise about what has
changed, because it is not this project's appetite.

**The premise is now false in one specific clause. France has a count.** The European Social Survey asks
French residents which religion or denomination they belong to, has done in seven rounds since 2010, and
publishes it by region. Pooled, that is 12,678 citizen respondents over 21 régions — the same
self-identification basis as Russia's Arena, Georgia's Caucasus Barometer or Greece's own citizen half.
**92.7% of the people France draws come from a survey that asked them.** No surname model, no origin
inference, and no microdata: the ESS API cross-tabulates server-side, so nothing individual-level ever
moves.

**§11l did not consider it, and the reason is chronology rather than judgement.** §11l was written
2026-09-06 and its France section reviews TeO2, the diocesan returns and IFOP. The ESS API was found
later the same day, by §9z, looking for something else. **§14.3's refusal was written against a France
reached by microdata or surnames, and this is neither** — on §14.3's own legal reasoning it is further
from the line than the route it rejected, because that paragraph's concern is *"building the same model
from individual-level microdata"* and this build never touches any.

**The foreign half is the §14.10 case and is 7.3% of the country.** 4.9 million foreign nationals from
Eurostat's 2021 census, crossed with Pew's origin compositions. That is exactly what §14.10 permits and
carries §14.10's conditions: the magnitude is the French state's own census count, the coefficients are
Pew's published national shares, the whole country is `modelled` in §7, and the output is checked — it
comes out **7.96% Muslim against Pew's independent 9.10%** for France.

**What §14.3 keeps, and it is most of it.** The rule that paragraph exists to state — *"never model at a
finer resolution, or a stronger claim, than the source publishes its magnitude at"* — is untouched and
is what caps France at 21 régions. The foreign half is available at 101 départements and **is not drawn
there**, on the reasoning `countries.py` already gives for Greece: mixing the two would put the sharper
geography on the half with the weaker claim to it. **§14.3's sentence about France should be read as
what it was — a correct description of every route known when it was written — and not as a standing
exclusion of the country.**

**The cost is stated rather than solved, and it is the geography.** 2.6 million people per unit is the
coarsest counting geography on this map, ahead of Russia's federal subjects at 1.82M. Île-de-France is
12.3M drawn as one composition, so **the map cannot show Seine-Saint-Denis**, which is the thing a
reader would most want from it. §3.9b says a country's geography is not a gate and the `grain` line says
which one it is; this is the country where that promise is doing the most work.

**Amended the same day: the five overseas régions are drawn too, from Pew, and they are the cleanest
case of §14.3's rule on the map.** France shipped at 21 units and 96.43%, leaving Corsica and the DOM
undrawn rather than borrowing a metropolitan composition for Martinique. Pew publishes French Guiana,
Guadeloupe, Martinique, Mayotte and Réunion as separate countries — and **each of them is exactly one
NUTS 2 unit**, so a Pew country row *is* a unit row and nothing is downscaled at all. §14.3's *never
model at a finer resolution than the source publishes its magnitude at* is satisfied **by identity
rather than by argument**, which is worth naming because every other modelled figure here satisfies it
by argument. Basis `estimate`, §3.1's own word for a Pew figure; the magnitudes cross-check at
0.98–1.01× the census for the four units the census carries; Mayotte has no census row at all and is on
the map only because a second source counted it. **France is 26 units and 99.30%**, and what is left
undrawn is Corsica, which has no ISO code of its own and is therefore invisible to every country-level
compiler in `sources.md` §1 — a fact about how compilers are organised, not about the place.

### 14.12 A fractional-share model is not uniformly trustworthy, and the split is predictable — FOUND 2026-09-07 with Kazakhstan

**§14.10 permitted the project to run its own fractional-share model and set five conditions. Its
fifth — *say what the output was checked against* — has been the soft one**, because most modelled
countries have nothing to check against and the section explicitly allows that as long as they say so.
Kazakhstan is the first build where a real check existed, and what it found is not "the model is fine".

**The check.** Kazakhstan's census publishes religion × nationality nationally and nowhere else, so the
country is drawn as ethnicity-by-region × religion-by-ethnicity (§9aq). The same volume also publishes
that coefficient table **separately for urban and rural Kazakhstan** — which is the model's own
assumption, *`share(religion | ethnicity)` does not vary by place*, written down as something
falsifiable. Building from the national coefficients and predicting the urban/rural split puts **1.64%
of the country on the wrong side of the town/country line**, and the error is wildly uneven:

```
    Ислам           +1.7% urban / -2.3% rural      69.3% of the country
    Православие     +0.9%        / -2.4%           17.0%
    Отказались      -7.8%        / +15.0%          a refusal, not drawn
    Неверующие     -13.0%        / +41.8%          drawn, and now flagged as the weakest cell
```

**THE RULE: an ethnicity model predicts ANCESTRY-SHAPED cells well and ATTITUDE-SHAPED cells badly, and
you can tell which is which before you build.** Islam and Orthodoxy in Kazakhstan are close to functions
of descent — Kazakhs 89.2% Muslim, Russians 85.5% Christian — and come back within three percent.
Non-belief and refusal are not functions of descent at all: they are urban behaviours *inside every
ethnic group at once* (Kazakhs are 1.4% non-believing in town and 0.6% in the country; Koreans 17.3%
and 10.7%), and no model whose only input is ancestry can see that.

This sharpens §14.2's first risk rather than restating it. That paragraph says §8.4 is "substantially a
race map wearing religion's labels" and treats the danger as presentational. **This adds that the
mislabelling is not uniform across the output: the cells a reader is most likely to find surprising —
irreligion, refusal, small movements — are exactly the cells the model is worst at**, because those are
the ones that are about a person rather than about their descent. A country note that says "modelled"
once, at the top, spreads the warning evenly over an error that is not evenly spread.

**So the rule has a second half, about disclosure.** Where a modelled country has a check, `note_public`
should name the WEAKEST DRAWN CELL specifically, not just declare the country modelled. Kazakhstan's
says *"wherever this map draws non-belief, treat the location as the weakest thing on the page"*, which
is a sentence a reader can act on; "this country is modelled" is not.

#### And the trap that comes with a good check: do not spend it

**Kazakhstan's urban/rural coefficients would model better than its national ones.** Using them was the
obvious move and is refused, because they are the only independent evidence the country has and **a
check you have consumed is not a check**. A model fitted to every published cut of its own coefficients
is unfalsifiable by construction, and §14.10's fifth condition would then be satisfiable only by the
empty statement that nothing was held out.

This is the same discipline §9z's Albanian case records — *the adjustment that felt more careful was the
one that would have been the error* — arriving from the other direction. There, restraint meant not
adjusting. Here it means **not using data you have**, which is harder to hold to and easier to undo
later by accident. It is written here because the next person to touch `sources/kz.py` will see an
unused, better coefficient set sitting in the same file and a one-line change that improves every
number.

**When a second cut IS available and is not needed as the check** — a third dimension, or a country
where the model can be validated some other way — using it is ordinary improvement and this section does
not forbid it. The rule is about the last check, not about all of them.

#### What to look for in the next country

The mechanism generalises and its conditions are exact. A country can be modelled this way if:

1. **a variable X is published by unit** — ethnicity, nationality, language, caste;
2. **religion × X is published nationally**, ideally by the same instrument;
3. **and the coefficient table is cut by some second dimension** — urban/rural, sex, age — so
   condition 5 can be met with evidence rather than with an apology.

**Condition 2 is rarer than it looks, and condition 3 rarer still.** A state that publishes religion ×
ethnicity nationally has usually published religion by region as well, and then there is no model to
run. Kazakhstan is unusual precisely because it did the first and not the second. **A model without
condition 3 is still permitted — §14.10 says so — but it must say so in `note_public`, where a reader
will see it, and not only in a source file.**


### 14.13 CFPS refused, and the Han residual is unblocked rather than lost — DECIDED 2026-09-07

**The application was refused on 2026-09-07** (`sources.md` §6b) — boilerplate about incomplete
account information, which for an unaffiliated applicant means no. §14.7 planned the Han layer as
*CFPS Buddhism, then everything else grey*, and said outright: **do not build the residual first,
because it is defined as what is left after Buddhism.** That ordering constraint was the only thing
blocking it.

**With no Buddhism coming there is nothing to subtract, so the constraint is void and the residual is
buildable today.** Everyone the ethnic derivation does not reach — the 1.14 billion Han and the 42
nationalities `taxonomy/cn2000.py` excludes — goes to `unknown`, at the county geography already
parsed and joined. No new source, no new fetch, no new join, and no new node: §6.3a-ii built
`unknown` for Vietnam and §14.7 records that its admission test *"covers China's case unchanged"*.
The residual asserts nothing, so it needs no coefficient and cannot be wrong about anybody.

#### What the refusal actually cost, which is less than it looks

**CFPS was never going to hand over a number. It was going to hand over a choice.** Pew's *Measuring
Religion in China* (2023) puts Chinese Buddhism at **4% of adults by self-identification (CGSS 2018)
and 33% by belief in Buddha or a bodhisattva (CFPS 2018)** — an eightfold spread on the same
population, from the same pair of instruments, and it is the *same* artefact §14.7 refused to
resolve for folk religion, wearing Buddhism's name. §3.1 says pick a basis, the basis would have had
to be `self_id` to sit beside anything else on this map, and self-id Buddhism is ~4%. **So the layer
CFPS was blocking was 4% of the grey, not the grey.** China would have rendered 97% one colour with
it and 100% one colour without it.

#### CGSS is the open substitute, and it was checked rather than assumed

**CGSS 2021 is mirrored on figshare as an 8 MB Stata file under CC BY with no account** — the second
instance of §9r's foreign-mirror finding, and the reason §6b now makes mirror-hunting a rule. It
carries exactly the two variables the layer needs: `provinces` (省份名称) and `A5`
(您的宗教信仰是什么？), n = 8,148. It has no AI-use policy, so unlike CFPS **a reconciliation check
would have been possible** — the thing §6a said this country would have to do without.

Its national shares are the self-id picture, and they are stable across instruments:

| | share of adults |
|---|---|
| no religion | **92.50%** |
| Buddhism | 3.50% |
| Islam | 1.72% |
| Protestantism | 1.47% |
| folk (Mazu, Guandi…) | 0.26% |
| Daoism | 0.22% |
| Catholicism | 0.20% |

**And it is still not enough to draw, which is the finding.** It reaches **19 of 31 provinces** —
missing Guangdong, Sichuan, Yunnan, Guizhou, Xinjiang, Tibet, Qinghai, Shanghai and four more — and
the per-province Buddhist cell runs from **1 respondent (Anhui, of 414) to 55 (Zhejiang, of 502)**.
Anhui at 0.24% Buddhist against Zhejiang at 11.0% is not a gradient, it is sampling noise with a
religion legend on it, and §14.10's five conditions cannot be met by a coefficient whose confidence
interval spans an order of magnitude. **Not drawn.**

**What would change the answer is pooling waves, and it is worth someone's afternoon.** CGSS ran at
~12,000 respondents over at least 28 provinces in most waves from 2010 to 2018; pooled, that is
~80,000 and a per-province Buddhist cell in the dozens rather than in single digits. Only the 2021
wave has been found openly mirrored so far. A second lead is unchecked: a PLOS ONE supplementary
file (`pone.0318221.s001.sav`, 22 MB, CC BY on figshare) that is CGSS-derived and may or may not be
multi-wave.

#### The two things that would add real colour to China, and neither needs a new source

1. **The residual, above.** It is what fixes the map — China currently reads as a populated west and
   an empty east, and §6.12's machinery is carrying more weight here than anywhere else.
2. **§14.9's southwestern mission peoples, still unparked and still unbuilt.** Lisu, Jingpo, Derung,
   Nu and Va are already in `cn.csv` at county level, joined and rescaled; `taxonomy/cn2000.py` still
   excludes them with a line citing §14.5's *ban*, which §14.9 withdrew. That is the one change that
   puts a genuinely new colour on China — Christianity in Nujiang and Dehong — for a few lines of
   taxonomy and a rebuild.

### 14.14 China is 100% drawn, and a threshold over an interested source is not a rule — BUILT 2026-09-07

Both halves of §14.13's closing list, built the same day on Anita's instruction. `taxonomy/cn2000.py`
is rewritten and `countries.py::_cn_counts` fans one source row out to several nodes.

**China went from 2.2% of its own population to 100% of it**, and the arithmetic is:

| | people | share | node | tier |
|---|---|---|---|---|
| §14.5's religio-ethnic derivation, 15 nationalities | 30,673,455 | 2.44% | `islam`, `buddhism.vajrayana`, `buddhism.theravada` | `derived` |
| §14.9's mission peoples, 6 nationalities | 875,255 | 0.07% | `christianity.protestant` | **`modelled`** |
| everyone else, 40 categories | 1,227,791,820 | **97.50%** | **`unknown`** | `derived` |

`unknown` is now **much the largest node on the map**, bigger than every other node put together,
and `christianity` reaches China for the first time.

#### The residual needed no argument, which is the point of it

Everyone the derivation does not reach — the 1.14 billion Han, the 42 nationalities `cn2000.py`
declines to name a religion for, the unidentified and the naturalised — goes to `unknown` at the
county geography already parsed and joined. No new source, no new fetch, no new node. §6.3a-ii built
`unknown` for Vietnam and §14.7 predicted its admission test would cover China unchanged; it did.

**Two calls inside it are worth recording.**

- **`unknown` is `derived`, not `measured`, and the reason is §3.4 rather than religion.** The row
  makes no religious claim, so the religion question does not settle its tier — but the county figure
  is a 2000 count carried onto a 2010 provincial total, which is §7's definition of `derived` and is
  why Brazil is 41.2% derived. So **§14.6's honest test survives**: `inferred dots: hidden` still
  empties China. What changed is only that it now empties a populated map instead of an empty one.
- **`gap` had to be rewritten rather than deleted.** It said *"Han majority not shown"*, which is no
  longer true. §6.12's line still earns its place, because the failure mode it guards against has
  moved rather than gone: a reader who mistook the blank for irreligion will now mistake the grey for
  it. It reads *"no religion question — 97% of these dots say only that somebody was counted"*.

#### The mission peoples, and the finding that cost the first design

§14.9's last bullet left Lisu, Jingpo, Derung, Nu and Va "flagged here for Anita and left unbuilt".
Built now, **with the Lahu added and the list re-derived rather than inherited** — §14.6's lesson,
arriving from the opposite direction. The Lahu are 486,000 people whom the same Baptist mission
reached as the Wa, Joshua Project puts their main group at 55%, and §14.8's list simply did not
mention them.

**The coefficients are Joshua Project's, per nationality rather than per people-group.** JP's PGIC
file is free, keyless and complete (`joshuaproject.net/resources/datasets/1`); it publishes a
Christian adherent share for each of 546 people-groups in China. The census names a *nationality* and
the state packs several people-groups into each one, so every figure below is JP's shares
population-weighted over the groups the state classifies under that nationality:

| nationality | census 2010 | share | drawn | JP groups |
|---|---|---|---|---|
| Lisu | 702,839 | **79.7%** | 560,283 | Lisu 80%, Lemo 0% |
| Lahu | 485,966 | 42.0% | 203,873 | Lahu 55%, Lahu Shi 10% |
| Va | 429,709 | 15.3% | 65,865 | Wa Parauk 19%, Wa Vo 0.2% |
| Jingpo | 147,828 | 23.9% | 35,331 | Kachin Jingpo 54%, **Zaiwa 0.25%**, Maru 79%, Lashi 35% |
| Nu | 37,523 | 23.2% | 8,687 | Nu 17%, Ayi/Anong 59%, Zauzou 5% |
| Derung | 6,930 | 28.1% | 1,945 | Drung 25%, Rawang 60% |

**Jingpo is the row that shows why the distinction matters.** Its headline group is 54% Christian and
its nationality is 24%, because the Zaiwa are the larger half and JP puts them at 0.25%. Taking the
people-group figure would have drawn 80,000 instead of 35,000.

#### ***A THRESHOLD OVER AN INTERESTED SOURCE'S OWN NUMBERS IS NOT A RULE***

This is the general finding and it is worth more than the country. The first gate written was
mechanical — *draw any nationality whose weighted share clears 10%* — on the good reasoning that a
stated threshold beats a hand-picked list, and it is exactly how the Lahu were found. Applied to all
546 of JP's China groups, the top of what it returns is:

| | population | JP Christian share |
|---|---|---|
| Han Chinese, Wu | 80,857,000 | **13.4%** |
| Han Chinese, Min Nan | 22,446,000 | 10.0% |
| Han Chinese, Min Dong | 10,143,000 | 10.0% |
| Han Chinese, Min Bei | 8,198,000 | 10.0% |

**121 million Han Christians in the southeast**, against CGSS 2021's 1.47% Protestant and 0.20%
Catholic nationally (§14.13). JP's `PercentAdherents` is not one quantity measured consistently: for
a small evangelised minority it is close to a church counting its own members, and for the Han it is
a national estimate spread across dialect groups. **An interested source's bias is not uniform across
its own rows, so a threshold over it inherits the bias instead of controlling for it.** §14.12 found
that a model can be trustworthy in one cut of a country and not another; this is the same shape one
level up, in the *source* rather than in the model.

**So the gate is three conditions and only the third comes from Joshua Project.**

1. **Selection is made on evidence outside the missionary literature** — Chinese state and academic
   reporting on Nujiang, Dehong and Lancang, the "first Christian county" description of Fugong,
   Yunnan's own >1M Protestants. This is §14.8's proposed clean form (*"an independently measured
   share … from a source with no stake in the answer"*) used to **choose** the groups rather than to
   number them.
2. **The JP groups under the nationality are co-located**, so spreading one share over the
   nationality's counties moves nobody — §14.3's resolution rule, checkable from JP's own centroids.
3. **The coefficient is JP's weighted share**, marked `modelled`, with the direction of its error in
   `note_public`.

#### What condition 2 costs, and it is the biggest single omission in China

**The A-Hmao and Gha-Mu, ~590,000 people whom JP puts at 80% Christian** — the Pollard mission's
harvest in northwestern Guizhou, as Christian as the Lisu — **are not drawn.** They are 6% of a
9.4-million `Miao` nationality spread over five provinces, and their centroid is **427 km** from that
of the 2.1-million Northern Hmu at 0.3%. The census column says `Miao` and cannot tell them apart, so
a nationality-wide share would put half a million Christians into Hunan and eastern Guizhou, which is
§8.1's failure in a new hat. The same argument excludes the Yi, whose Christian Lipo, Naluo and Laka
are ~2% of 8.7 million. **A source that placed the A-Hmao by county is the single highest-value
missing input for China**, and `note_public` tells the reader they are in the grey.

#### The check, and the one open judgement

**The check §14.10's fifth condition asks for**: the six peoples come to ~876,000 Christians,
essentially all in Yunnan, against a province repeatedly reported to hold **over a million
Protestants**. Consistent, and it leaves room for the A-Hmao and the Han that this map cannot place.
**And the check that does not fully agree, which matters more**: for the Lisu, JP says 80% where the
figure usually cited as China's official one — 300,000 Christian Lisu in Yunnan — works out to 43%,
and the churches' own claim is essentially 100%. The truth is inside a factor of two and **this map
takes the missionary source's number, which is the high end**. Recorded rather than corrected, per
§14.12: the adjustment that feels more careful is usually the error, and one source applied
consistently beats a per-group blend. `note_public` says outright that the layer probably runs high.

**The open judgement is the KOREANS, and it is left to Anita rather than taken.** JP puts the 1.83M
Korean nationality at 30% Christian; they are co-located; that would be ~549,000 people and the
second-largest Christian block in China, in a northeast that currently has nothing but grey. The old
reason for excluding them was §14.5's *constituted religiously* test, **which §14.9 withdrew** — so
that objection is dead and what remains is condition 1, where the outside attestation is thinner than
Nujiang's and rests largely on analogy with South Korea, and the population is the most urban and
migratory of the candidates. `cn2000.py`'s REVIEW says so under `Korean`.

#### One thing raised rather than settled: whether the grey should survive the toggle

`tools/check_rollup.py` pushes back on this build, and it is right to. China now reports **99.9% of
its people orphaned** — derived, with no measured ancestor — so `inferred dots: hidden` removes
1.2 billion people who *were* counted, about whom the map makes no religious claim at all. The
tool's own note says a root node in that list usually means "the religion was counted and only the
geography was inferred, so hiding these people is wrong rather than cautious." China is the country
that note was written without.

**The call made here is `derived`, on §3.4's ground rather than on any ground about religion**: the
county figure is a 2000 count carried onto a 2010 provincial total, which is what §7's table calls
`derived` and is why Brazil is 41.2% of it. That keeps §14.6's honest test intact and keeps China
consistent with every other §3.4 country.

**The argument the other way is not weak and is worth putting to Anita.** §7a says the mode answers
*"which of this did somebody count?"* — and in China somebody counted 1.24 billion people and where
they live, which is the entire content of the `unknown` rows. Under that reading the grey is the
measured layer and the toggle should strip the colour and leave the population standing. It would
also make Vietnam and China behave alike: the same node, meaning the same thing, currently survives
the toggle in one country and not the other, and the only reason is §3.4's carry.

**Cost of changing it later: one word in `_cn_counts`, then scatter, tiles and buffers for `cn`.**
About ten minutes. Nothing else in the pipeline depends on the choice.

### 14.15 What is left for China, ranked — WRITTEN DOWN 2026-09-07 so it is not re-derived

§14.13 and §14.14 record what was built and why. This is the part that was only in the
session: **what to do next, in what order, and the one reframing that changes which source is
worth chasing.**

#### THE REFRAMING, AND IT IS THE MOST USEFUL THING HERE: CFPS WAS THE WRONG TARGET ALL ALONG

§14.7 planned the Han layer around CFPS and §14.13 treated its refusal as a loss to be worked
around. **It was not a loss. CFPS asks the wrong question**, and the evidence was in the same
Pew sentence that gave §14.13 its 4%-versus-33%:

| survey | what it asks | Buddhism, 2018 | §3.1 basis |
|---|---|---|---|
| **CGSS** | *which religion do you belong to* | **4%** | **`self_id`** |
| CFPS | *do you believe in Buddha or a bodhisattva* | 33% | belief — no basis on this map |

**§3.1 says bases are never mixed, and `self_id` is the basis every other country here is drawn
on** — Vietnam's census, Korea's, Thailand's, Russia's Arena, Greece's and France's ESS. A
CFPS-derived Chinese layer could not have sat beside any of them without breaking the one rule
§3.1 exists to enforce; it would have made China the only country on the map answering a
different question, in the country where that difference is largest. **So the account refusal
cost nothing except the time spent designing around it**, and the thing to notice is that
nobody checked the question wording before applying for the data. *Check what a survey ASKS
against §3.1 before treating access to it as the blocker.*

**CGSS is the basis-compatible source and always was.** That is the argument for chasing it
specifically, over and above its being the one that turned out to be openly mirrored.

#### Two survey leads that were never checked at all

Both are bigger than the CGSS wave in hand and neither was searched for an open mirror:

- **CLDS** — China Labor-force Dynamics Survey, Sun Yat-sen University. **~21,000 adults over
  29 provincial units** in 2014, against CGSS 2021's 8,148 over 19. Pew used it and notes the
  CLDS restricted their access to the 2016 and 2018 waves, which says the earlier ones are less
  restricted. **On size and coverage this is a better single target than pooling CGSS**, and it
  is untouched.
- **CSLS 2007** — Chinese Spiritual Life Survey, Purdue/Horizon, 7,021 respondents over 56
  sites. Old and site-based rather than province-representative, so it is the weakest of the
  three; listed here only so the next session does not rediscover it as new.

Neither has been checked against [[feedback_gated_data_last_resort]]'s mirror list, which is
the whole of the work: Dataverse, figshare, Zenodo, GitHub, a university library guide.

#### The A-Hmao input is a COUNTY LIST, not a survey — and §14.14 did not say so

§14.14 calls a source that places the A-Hmao by county *"the single highest-value missing input
for China"* and stops there, which is not actionable. Being precise about what is missing:

**The magnitude is already in hand.** `cn.csv` has Miao by county for all 2,691 counties, and
Joshua Project has the share (A-Hmao 448,000 at 80%, Gha-Mu 142,000 at 80%). What is missing is
only **which counties the A-Hmao are the Miao of** — a partition of one nationality's county
column, not a religion figure at all.

That is an ethnographic fact rather than a statistical one, and it is well documented in a
literature this project has not touched: **Chinese local gazetteers (地方志), the provincial
民族志 volumes, and linguistic atlases of the Miao languages**, which distinguish 大花苗
(A-Hmao) from the Hmu, Ghao-Xong and the rest. The concentration is small and named — Weining,
Hezhang, Nayong and Zhijin in Guizhou; Wuding, Luquan and Yongshan in Yunnan — so a source
naming those counties would be enough. **~590,000 people at 80%, and it would put a second
Christian block on the map as large as the Lisu.** The same route would settle the Yi (Lipo,
Naluo, Laka).

#### So, ranked

1. **The Koreans, and it is free.** `cn2000.py`'s REVIEW has the argument; Joshua Project puts
   them at 30%, they are co-located, and it is ~549,000 people in a northeast that currently
   draws nothing but grey. **One line of taxonomy and a rebuild**, and the only reason it is not
   done is that it is Anita's call (§14 asks for exactly this to be raised).
2. **CLDS**, then pooled CGSS. Basis-compatible, province-level, and the mirror hunt is a
   couple of hours. This is the only route to a Han layer that could sit beside the rest of the
   map.
3. **The A-Hmao county list**, above. Different kind of hunt — a library, not a data portal.
4. **The `unknown` tier question** §14.14 raised and left open: whether a billion grey dots
   should survive `inferred dots: hidden`. One word in `_cn_counts`, then scatter/tiles/buffers
   for `cn`.

**And what is NOT worth doing**, so it is not attempted again: any route through CFPS (wrong
basis, and the terms forbid the tooling); the 2020 census ethnic table (a JPEG scan, §14.6); a
township-level rebuild (§14.5's ceiling is a limit, not a target); and the registered-venue
registry as a magnitude source — §2.6 was written the same day about exactly that mistake in
Thailand, and China's registry omits house churches, which is most of what is there.

### 14.16 China is drawn from self-identification, and the affirmative half of that question is a measurement while the negative half is not — BUILT 2026-09-08

§14.15 ranked the work and this is items 1 and 2 of it, plus the Pumi flag closed. Anita's
calls throughout. `sources/cn_cgss.py` and `sources/cn_cgss.md` carry the detail; this records
what generalises.

**China goes from 2.5% coloured to 8.4%**, and for the first time the colour is in the east:

| | people | node | tier | basis |
|---|---|---|---|---|
| §14.5 religio-ethnic, 14 nationalities | 30.63M | `islam`, `buddhism.vajrayana`, `buddhism.theravada` | `derived` | ethnicity |
| §14.9 mission peoples, 6 nationalities | 0.88M | `christianity.protestant` | `modelled` | ethnicity |
| **§14.16 CGSS, 29 provinces** | **54.60M** | **`buddhism.mahayana`** | **`modelled`** | **`self_id`** |
| **§14.16 CGSS, 29 provinces** | **20.19M** | **`christianity.protestant`** | **`modelled`** | **`self_id`** |
| everyone else | 1,153.05M | `unknown` | `derived` | — |

**The Buddhist layer alone is larger than everything China had before it.** The total is
preserved to the person: the carve eats the `unknown` residual and invents nobody.

#### THE DESIGN FINDING, AND IT IS NOT ABOUT CHINA

Anita, framing it before any data was fetched: *"what **is** religion in china. since its
vague, maybe we want to find self identification (even if its really low count and only
reflects like strong believers / priests or whatever) and if we can find that, display those
small numbers."*

That is the right instinct and it sharpens into a rule §3.1 did not have:

> **The two halves of a self-identification question are not equally trustworthy, and where
> they come apart, draw the affirmative half and leave the rest UNKNOWN rather than
> irreligious.**

In China *"yes, I am a Buddhist"* is stable across instruments and waves; *"none"* is not a
finding about belief but an artefact of the wording — 4% Buddhist by self-id against 33% by
belief, same population, same year (§14.13). §14.7's *"refusing to draw the boundary is the
point"* was right about the boundary and wrong to conclude that nothing could be drawn: **the
affirmative side is a measurement even when the negative side is an artefact.** Carving the
first out of the grey claims exactly as little as the grey did about everyone else.

**This is not a China special case.** It is the shape of every dual-practice country §6.3a
names — Japan, Vietnam, Korea, much of West Africa — and it is the reason `unknown` is worth
having as a node at all. Where a country's "no religion" box is doing work the question did not
earn, the affirmative rows are still drawable.

#### A SURVEY'S PROVINCIAL CUT CAN BE A LOTTERY, AND THE CENSUS MARGIN IS THE TEST

The most reusable operational finding. CGSS's provincial subsamples come from a handful of
PSUs, so for a minority concentrated *within* a province the sample lands on it or misses it:

| | census, Muslim nationalities | CGSS pooled self-id Islam | |
|---|---|---|---|
| Qinghai | 16.9% | 1.1% | **0.07×** |
| Gansu | 7.4% | 0.6% | 0.1× |
| Xinjiang | 58.3% | 92.0% | 1.6× |
| Ningxia | 34.5% | 93.1% | **2.7×** |

**Fourteenfold out one way and nearly threefold the other is not a bias a weight can correct.**
So: **before drawing a survey's sub-national cut, check it against a census margin for a
variable both carry.** Here that variable is ethnicity, and it says plainly that CGSS may be
used for the evenly-spread religions and not for the concentrated one. Islam therefore stays on
§14.5's county derivation, which is better at exactly the thing the survey is worst at.

**And nationally the same comparison is the first external check §14.5 has ever had.** CGSS
self-id Islam runs 1.87%–2.56% against the derivation's 1.83%. The survey finds at least as
many Muslims as the derivation predicts, so **§14.5's coefficient of ~1.0 is vindicated rather
than merely asserted** — at national level, and only there.

**It also closes a question Anita raised** — whether Tibetans should be drawn at less than 1.0,
with the remainder on `unknown`. CGSS cannot answer it: Qinghai's expected 27.3% Buddhist
against an observed 11.8% looks like support for ~0.4, but the same sample found 7% of
Qinghai's Muslims, so it is a Han and urban sample that missed the Tibetans for the same reason
it missed the Hui. **The test fails its own control**, and Tibet is not sampled at all. Written
down so it is not re-run. Changing the coefficient would need a documented source, and §14.5's
own test is *documented rather than fitted* — §9z's lesson is that the adjustment which feels
more careful is usually the error.

#### POOLING BUYS PRECISION AND SPENDS CURRENCY, AND THE DENOMINATOR PICKS THE WAVE

Reported religiosity falls monotonically across the three waves — any religion 14.47% (2012) →
10.61% (2017) → 7.50% (2021), in every category at once, Islam included. Partly a real decline
in willingness to report, partly the multi-select→single-choice change at 2021; the 2012→2017
fall happens with the instrument held constant, so it is not only the instrument.

**So a pooled share is an average over a moving target and the choice of level is real.** The
rule taken: **the level follows the DENOMINATOR's vintage, not the newest wave.** cn.csv is 2000
structure on 2010 totals, so the people being coloured are the 2010 census's people; pooled and
n-weighted CGSS centres on about 2015, where 2021 alone is eleven years downstream of its own
denominator. Pooled is the closer fit. Cost, stated in `note_public`: this layer is about half
again larger than 2021 alone would draw.

#### WHAT SURVIVED §14.10 AND WHAT DID NOT

Two categories of six. **Buddhism passes cleanly** — χ² p = 4e-184, Zhejiang 15.7%
(CI 14.0–17.5) against Anhui 0.9% (0.4–1.4), and the pattern is the southeastern coastal belt
the literature describes. **Protestantism is drawn on Anita's call with its weakness
disclosed**: its spatial variation is highly significant (χ² p = 1.3e-84) and Henan comes out
top unaided, but its 2012↔2021 rank stability is **+0.17** against Buddhism's +0.63.

**A rank correlation is a stability test, not a reality test, and +0.17 on 19 provinces is a
failure to demonstrate signal rather than a demonstration of noise** — its CI includes zero and
reaches past +0.55. Some of the instability is likely real: if enforcement varied by province
while reporting halved, the ordering *should* move. §14.12's disclosure rule applies and
`note_public` names Protestantism as the layer to trust least, in those words.

**folk, Daoism and Catholicism are not drawn.** Daoism is 80 respondents in 32,495 and
Catholicism 65, with every province under ten. **folk is the interesting refusal**: 681
respondents, but its national share collapses 3.43% → 2.11% → 0.27% across the waves and
Guangdong reads 22.5% against zero in Beijing and Hunan. That is §14.7's artefact category
proving itself, and it is the strongest evidence yet that `chinesefolk` should stay empty.

#### THE PUMI ARE UNDRAWN, AND A LIST IS STILL NOT A RULE

`cn2000.py`'s REVIEW had flagged them since 2026-09-07 as *"THE ONE DRAWN GROUP THAT PROBABLY
SHOULD NOT BE"*: 33,599 people whose religion is Hangui held alongside Gelug Buddhism, which is
§14.5's *religiously mixed* row and not its *religio-ethnic* one. Drawn only because §12's list
named them. Anita's call to remove them, 2026-09-08. **This is §14.6's lesson arriving from the
opposite direction** — there, applying the stated test to all 56 nationalities *added* the Lahu
that the illustrative list had missed; here it *removes* one the list had wrongly included.
`_VAJRAYANA` is now three nationalities. 34 dots either way; the point is the consistency.

#### THE HUNT, AND WHAT IS STILL OUT THERE

§6b's mirror rule went four for four again. **CGSS 2012 and 2017 are on Harvard Dataverse under
CC0** (`doi:10.7910/DVN/R1UDF2`, `doi:10.7910/DVN/SZUSBS`), sitting loose in the root
collection so a collection crawl misses them; `api/search?q=title:CGSS` finds exactly those two.
With the figshare 2021 wave that is 32,495 respondents over 29 provinces — 99.2% of China's
population, against §14.13's rejected 8,148 over 19.

Still open, in order:

1. **CLDS is found and behind a login.** Science Data Bank `doi:10.57760/sciencedb.02333`,
   77.5 MB, advertising **CC BY 4.0 and `conditionsOfAccess: PUBLIC`**, covering the 2011
   pilot and the 2012/2014/2016/2018 waves — but the listing API answers `70001 无访问权限`.
   A free ScienceDB account is email-only, which is not the Korean-ID wall
   ([[reference_korea_open_data]]); this is a cheap ask, not a dead end. Note CLDS's own use
   agreement forbids redistribution, so §6b's "cite the origin, not the badge" applies hard.
2. **CGSS 2006 at PKU is CC0 and `restricted: false`, and their file server 500s.**
   `doi:10.18170/DVN/21HKLB`. Metadata API works; access API does not. Worth one retry.
3. **CNSDA and `cgss.ruc.edu.cn`** hold 2010, 2011, 2013, 2015 and 2018 behind an ordinary
   free email registration. The only route to those five waves, and pooling them would let the
   Protestant layer be re-tested rather than disclosed.
4. **The A-Hmao county list** — unchanged from §14.15, and now the largest single omission
   again, since Han Buddhism is drawn and they are not.

**Checked and dead, so it is not reopened:** the PLOS ONE supplementary §14.13 flagged as
possibly multi-wave (`pone.0318221.s001.sav`) is a single wave and **carries no religion
variable at all** — its only "religio" string is the ISCO occupation code *religious
professionals*. Zenodo has neither survey. ICPSR holds only the EASS cross-national sets.

### 14.17 73.5 million Chinese people were being read and then thrown away, and the check that should have caught it was measuring the wrong denominator — FIXED 2026-09-08

Anita, looking at the finished §14.16 map: *"it seems like kontur (if thats what we're using)
kinda sucks in china. it says here no people in zhongshan."*

**It was not Kontur.** Kontur has 3,361,377 people in Zhongshan. `cn.csv` had none, and
Zhongshan was one of **168 counties — 67,805,092 people, 5.47% of China — that `sources/cn.py`
read out of the census volumes and then discarded** because the romanised name matched no
adcode. With Hainan's separate source gap that is 71.0M, against an observed shortfall of 73.5M
(the rest is drift in the §3.4 rescale factors).

**It was never a boundary problem, which is the first thing to say because it is where an hour
went.** The instinct was to hunt for 2000-vintage county boundaries. The geometry has 2,848
polygons against the volumes' 2,859 counties and every code that resolved found one; the
missing people were in `data/raw/cn/` the whole time. *Check whether the join failed before
concluding the geography is the wrong vintage.*

#### THE FINDING WORTH KEEPING: A CHECK'S DENOMINATOR CAN GO STALE

`cn.py` has always printed this, and it has always looked healthy:

```
drawn population stranded: 69,728 of 26,950,708 = 0.26%
```

**That number was true and it was about the wrong people.** It counts §14.5's religio-ethnic
population, because when it was written those were the only people drawn — China was 2.2% of
itself and the minorities were the whole map. §14.13 put all 1.24 billion on `unknown`, and
from that moment the meaningful figure was the *total* stranded: **5.47%, twenty times larger.**

**The check did not become wrong. It became irrelevant, and nothing said so.** Nothing could:
it was still measuring exactly what it claimed to measure. This is the third time in two days
that a correct-looking artefact went stale because the country underneath it changed — §14.16's
`covers` list omitted `buddhism.mahayana` while 54.6M Buddhists were drawn, and §14.14's `gap`
line still said *"Han majority not shown"* after the Han were shown.

> **When a country's DRAWN POPULATION changes, re-read every check that has a denominator.
> A ratio whose numerator you fixed can keep reporting on a population that is no longer the
> one at risk.**

`cn.py` now prints the total alongside the drawn figure, and flags outright that stranded people
are dropped rather than redistributed.

#### THE 168, AND THEY FALL INTO FIVE CLEAN CLASSES

Both the census file and DataV's index run in GB/T 2260 code order within a province, so an
unresolved county lies in a known interval between its resolved neighbours and the successor is
the modern unit in that interval. That is the same argument `cn.py`'s `ordered` tier already
used, applied to the residue. **`free` was not used as a filter** — a 2000 unit is more often
absorbed than renamed, so the target is frequently a code another census county already claims,
which the file has always allowed.

| class | n | example |
|---|---|---|
| **市辖区 / 郊区 dissolved into modern districts** | ~90 | 无锡市郊区 → 滨湖区; 苏州沧浪+平江+金阊 → 姑苏区 |
| **a character's place-name reading** | 29 | 番禺 written FANYU not PANYU; 鄱阳 BOYANG not POYANG; 乐清 LEQING not YUEQING |
| **a county sharing its prefecture's name** | ~30 | 遵义县 → 播州区; 毕节市 → 七星关区 |
| **the suffix dropped** | 14 | KAI = 开县, DA = 达县, HENG = 横县, HU = 户县 |
| **the source is simply wrong** | 3 | **中山市 is romanised ZHONGZHAN**, and two Hubei names arrive byte-corrupted |

**Zhongshan is the single biggest miss and it is a typo.** 2,363,322 people absent from a map
of the Pearl River Delta because the volume wrote ZHONGZHAN. The two corrupted Hubei names —
硚口区 and 猇亭区, both rare characters — are keyed in `OVERRIDES` by their mojibake literal,
since no romanisation rule can ever reach them and the adcode interval identifies both without
the name.

**市中区 is the sharpest case of the generic name**: Sichuan has three, in 遂宁, 内江 and 乐山.
`OVERRIDES` takes a list there and the nth occurrence in file order takes the nth code, which is
the mechanism §12 built for 伊宁市/伊宁县 and which turns out to generalise.

#### THE VERIFICATION, AND IT IS UNUSUALLY STRONG

Not "it looks better" — the census is its own held-out test, because the per-province 2010
totals were never used to place anybody:

```
county -> adcode: 2859/2859 resolved       (was 2691/2859)
unresolved: 0                              (was 168)
TOTAL population stranded: 0 = 0.00%       (was 67,805,092 = 5.47%)
national total: 1,332,810,852              against the census's 1,332,810,869
```

**Seventeen people out of 1.33 billion**, and **every one of the 31 provinces now reconciles to
100.0%** — where before Hainan was at 82.4%, Zhejiang 86.6%, Jiangxi 88.3%, Guangxi 88.4%. China
draws 1,332,805 dots, up from 1,259,338.

#### THE RESIDUAL, WHICH IS A PLACEMENT ERROR AND NO LONGER A COUNTING ONE

80 modern polygons still carry no rows, holding 27.7M people by Kontur (down from 194 and
95.2M). **Nobody is missing** — the province totals prove it — but where a 2000 county was
*split* into two modern districts, the whole of its population is drawn inside whichever
successor the override names. 潮南区's people are drawn in 潮阳区, 相城区's in 吴中区. The
displacement is between adjacent districts of one city and never crosses a prefecture.

Fixing it properly means mapping one census county onto *several* polygons and splitting by
Kontur population, which is a change to the shape of `OVERRIDES` rather than more entries. Left
undone deliberately; the counting error was the serious one.

**And Yichun is the one group placed only approximately even at prefecture grain.** 伊春 was 15
districts in 2000 and became 4 districts + 4 counties in 2019, so thirteen census units map onto
seven modern ones; the pairings follow the reorganisation but a 2000 district that was split
goes whole to the successor holding its seat. ~500,000 people, all inside Yichun. 大兴安岭's
松岭, 新林 and 呼中 are 林业局 areas with no GB/T 2260 code at all in DataV and go to the
adjacent county that administers them — 131,000 people, and the weakest placement in the block.

#### WHAT IT DOES TO §14.16's NUMBERS

The recovered 73.5M are disproportionately urban — **142 of the 168 were 市辖区** — so the CGSS
layer, which carves the `unknown` residual at province grain, gains most of them:

| | before | after |
|---|---|---|
| `buddhism.mahayana` | 54.60M | **58.39M** |
| `christianity.protestant` | 21.07M | **22.15M** |
| `islam` | 23.07M | 23.14M |
| `unknown` | 1,153.05M | 1,221.56M |
| coloured share | 8.44% | 8.35% |

**And it removes a bias §14.16 could not have seen.** A layer whose coefficients are provincial
but whose denominator was missing 5.5% of the province — concentrated in its cities — was
applying urban-inclusive shares to a rural-skewed population. That is fixed as a side effect,
and it is the reason this was worth doing before anything else on the China list.

### 14.18 Arunachal goes to India, the Koreans are drawn, and the `unknown` toggle question is closed — DECIDED 2026-09-08

Three of §14.15's open items and one new defect, all Anita's calls in one sitting.

#### THE MAP WAS DRAWING 197,000 CHINESE PEOPLE INSIDE INDIA

`todo.txt` had carried *"china shows buddhists in arunachal, double counting?"* since before
§14.16, and §14.16 made it visible by putting colour in eastern China. It is real, and the
mechanism is worth stating because it is not the obvious one.

**DataV's county boundaries follow the PRC's territorial claim.** Tibet's Cona, Lhünzê, Mêdog,
Zayü and nine neighbours extend south of the McMahon line over Arunachal Pradesh, which India
administers and which India's own census draws. `cn_geo.py` assigns each Kontur hex to the
county containing it, so **1,516,251 Arunachal residents' hexes were assigned to Chinese
counties** — and `scatter.py` then places a county's dots in proportion to Kontur population.

**So the dots were not invented people. They were real people, counted by China on land China
administers, dragged south onto land it does not.** Anita: *"the people come from land in china
right? and not in arunachal."* Exactly that. Of the 333,234 the census puts in those counties,
**about 197,000 — 59% — were drawn inside India.** Cona is the pure case: 17,530 census people,
and Kontur put *none* of its population on the Chinese side and a million on the Indian side, so
every one of its dots landed in Arunachal.

**The rule taken is DE FACTO ADMINISTRATION** — Anita: *"i think we give arunachal to india, i
thin thats the usual stance and its de facto right."* Implemented from data rather than a hand
list: `sources/cn_geo.py::clip_to_de_facto` subtracts every polygon in
`ne_10m_admin_0_disputed_areas` whose `NOTE_BRK` says China claims it and somebody else
administers it. **Natural Earth records exactly this distinction** — "Admin. by India; Claimed by
China" — which is why that layer is the right one and why `country_shapes.py` already used it for
Abkhazia.

That removes Arunachal (87,238 km²), Demchok, the Samdu, Tirpani and Bara Hotii valleys and the
two Bhutanese salients — **and keeps Aksai Chin and the Shaksam Valley, which China administers
and India and Pakistan claim.** The clip runs on the counties *before* the grid is built from
them, so the hexes are clipped by construction and a future rebuild cannot forget.

| | before | after |
|---|---|---|
| Arunachal residents assigned to Chinese counties | 1,516,251 | 265,446 |
| China dots inside Natural Earth's India (1:10k) | 28 | **4** |
| India dots inside Natural Earth's China | 0 | 0 |

The residual 4 dots are boundary-precision noise between two outlines that disagree by a few
hundred metres — the same order as the mx/us and de/fr borders below.

#### AND THE GENERAL CHECK, WHICH SAYS THIS IS THE ONLY CASE

Before fixing it, every country's dots were binned to a 0.05° grid and cells claimed by two
countries counted. **The map has essentially no double-drawn ground**: cn/in was one shared
cell; everything else is 1–6 cells of ordinary border noise (mx/us 16, ca/us 4, de/fr 3, ch/it
3). So this was not a class of bug, it was one country's boundary source following a claim.

**It did surface one piece of litter**: a stray `dots_in_32_10k.geojson` overlapping India's
national file across 866 cells in Kerala. It is not in `tools/built_countries.py`, so it never
reached the tiles — a leftover from a per-state build, and safe to delete.

#### THE KOREANS ARE DRAWN, AND THE DOUBT IS DRAWN WITH THEM

§14.14 flagged them, §14.15 ranked them first, and §14.16 postponed them. Now built: **1.83M
Korean nationals at Joshua Project's 30%, ~549,000 people on `christianity.protestant`,
`modelled`** — the second-largest Christian block in China and the first colour in a northeast
that had nothing but grey.

**What is being traded is stated rather than buried.** Of the seven MISSION rows this is the only
one whose coefficient has no local corroboration: for the Lisu, Lahu, Jingpo, Wa, Nu and Derung
the Christian pattern is attested in Chinese state and academic reporting on Nujiang, Dehong and
Lancang, and for the Korean-Chinese the outside attestation is largely analogical. **And JP's 30%
sits within two points of South Korea's own self-identified Christian share** (2015 census: 19.7%
Protestant + 7.9% Catholic = 27.6%). That may be a real convergence — the Yanbian church is
genuinely well documented — or it may be a South Korean figure carried across the border, and
nothing available here distinguishes them. **If it is the analogy, then the split is wrong as
well as the level**, because a third of South Korea's Christians are Catholic and this row is
Protestant-only. `cn2000.py`'s REVIEW keeps the whole pre-decision argument for that reason.

The visible effect is narrower than "the northeast": Koreans are 3.8% of Jilin, 0.9% of
Heilongjiang and 0.6% of Liaoning, but **47–58% of Yanji, Longjing, Tumen and Helong**. So it is
one prefecture lighting up and a faint dusting over three provinces.

#### THE `unknown` TIER QUESTION IS CLOSED, AND THE ANSWER IS THE ONE ALREADY BUILT

§14.14 raised it and §14.15 ranked it fourth: should `inferred dots: hidden` keep China's grey
standing, on the ground that those people *were* counted and the row makes no religious claim?

**No.** Anita: *"if they didnt vanish we would just show all of china as brown which is kinda
silly."* A control whose purpose is to show what was measured about RELIGION should not leave a
billion dots standing that say nothing about religion — it would replace an honest emptiness with
an uninformative mass. §14.6 called China emptying under that toggle the honest test of the
country and it stays. No change to `_cn_counts`; the §3.4 tier reasoning it already rests on is
sufficient and Vietnam's differing behaviour is a consequence of Vietnam's geography being
measured at its own grain, not an inconsistency to repair.

#### CLOSED WITHOUT ACTION

**`tools/check_palette` tests only authored ROOT colours, and that is deliberate.** §14.16 noted
that Mahayana and Theravada are dE 13.2 apart, under the floor, and proposed extending the tool
to child shades. Anita: *"i dont think we actually want this to test child colors."* Right — the
child shades are generated from the root hue *in order to* read as one family, so a within-branch
distance under the floor is the design working. Not a gap.

#### STILL OPEN AFTER THIS

- **The 80 split-county polygons** (§14.17), 27.7M people. Anita found one unaided —
  *"nansha is empty of people on our map"* — and Nansha is exactly the shape: created 2005 out
  of 番禺市, which the 2000 census knows only as `FANYU`, so all 1.63M of old Panyu draws inside
  modern 番禺区 and Nansha's 652,857 residents' ground draws nothing. **Nobody is missing** — the
  national total is 17 people off the census — but 2.0% of China lives on ground that draws
  nothing. The fix is letting one census county map onto several polygons, split by Kontur.
- **CLDS**, behind a free ScienceDB registration (§14.16).
- **CGSS 2010/2011/2013/2015/2018** behind CNSDA registration — the only route to re-testing the
  Protestant layer rather than disclosing its weakness.
- **The A-Hmao county list** (§14.15).

### 14.19 The split counties are filled, and chasing them turned up 5.9 million people drawn in the wrong place entirely — FIXED 2026-09-08

Anita asked for §14.17's 80 orphan polygons. Doing them surfaced a worse class of error that
nothing in the pipeline could see, so this section is mostly about that.

#### THE ERROR NOTHING COULD SEE: A REAL ADCODE IN THE WRONG PREFECTURE

`sources/cn.py` joins the census volumes to DataV by romanised NAME, and China has many
counties that romanise identically. When the resolver picks the wrong one **it still returns
a real adcode in the right province**, so every check downstream passes — the county total is
right, the province reconciles, the national figure is exact to 17 people, `check_mapping` is
happy. The only thing wrong is that half a million people are drawn 400 km from home.

**Two shapes of it, ~5.9M people:**

| | people | |
|---|---|---|
| **same romanisation, several real counties** | ~2.48M | Hebei's three Wei counties — 魏县 (Handan), 威县 (Xingtai), 蔚县 (Zhangjiakou) — are all `WEIXIAN`, and all three were drawn as one. Two real counties drew nothing. |
| **wrong prefecture outright** | ~3.42M | 郧县's 584,315 residents drawn in 云梦县; 南宁新城区's 426,346 in 忻城县, another prefecture; 安庆市郊区's 264,670 in 铜陵's 郊区. |

**And two of them were OVERRIDES entries whose COMMENT named the right county while the
digits named another** — `ZHONGDIAN` pointed at 533422 (德钦县) under a comment reading
*"中甸县 -> 香格里拉市"*, and `WEILI` at 652927 (乌什县) under *"尉犁县"*. Nothing about
reading the file would ever have caught those; only the arithmetic could.

#### THE CHECK, AND IT IS ONE SENTENCE OF REASONING

`tools/check_cn_prefecture.py`, new. **Both the census file and DataV's index run in GB/T
2260 code order within a province, so a county's NEIGHBOURS IN THE FILE are its neighbours in
code space.** A resolution landing in a prefecture that neither neighbour is in is almost
always the wrong same-named county.

It flags 35. **Twenty-five are legitimate** — a county really did change prefecture (简阳 to
Chengdu, 无为 to Wuhu, 寿县 to Huainan, 公主岭 to Changchun, 枞阳 to Tongling, 海原 to
Zhongwei), or is provincially administered (济源, 儋州, 石河子, 嘉峪关), or is a
prefecture-level city with no counties (东莞, 中山). Those are in an allowlist **with the
reason written next to each**, so that anything not on the list is a finding rather than
noise. Ten were real; all ten are fixed and the check now reports zero unexplained.

**The general lesson, and it is the same one §14.17 recorded from the other side.** §14.17's
stranded-population check went stale because its denominator stopped being the population at
risk. This is the complementary failure: **a join that fails LOUDLY is safe, and a join that
fails into a plausible neighbour is not.** Every totals-based check in this project would
pass on a map that has swapped two counties. Where a join is by name and names repeat, the
check has to be about POSITION, not about totals.

#### THE ORIGINAL TASK: 80 ORPHAN POLYGONS, AND WHAT EACH ONE ACTUALLY WAS

Anita found the symptom unaided: *"nansha is empty of people on our map... seems like our
population coverage is actually in general still very spotty."* **The coverage is not spotty
— nobody is missing** (the national total is 17 people off the census) — but 2.0% of China
lived on ground that drew nothing.

The 80 turned out to be four different problems:

| | n | what |
|---|---|---|
| name collisions and wrong-prefecture errors | 11 | fixed in `OVERRIDES`, above — these were never splits |
| **genuine post-2000 splits** | **55** | fixed by `ABSORB`, below |
| Hainan's incomplete census volume | 11 | **left blank on purpose** |
| not administered by the PRC, or created after 2000 over empty islands | 3 | 金门县, 西沙区, 南沙区 |

**`ABSORB` in `sources/cn_geo.py` is the fix and it changes no count anywhere.** Where a 2000
county was split, `OVERRIDES` names one successor and the whole population draws inside it,
leaving the sibling's ground blank. The hexes are already clipped to the sibling's polygon,
so **relabelling the hex to the parent's adcode** makes `scatter.py` spread that county's dots
over its original 2000 territory, weighted by Kontur — which is what the county's population
always meant. Nansha's 652,857 residents' ground now draws Panyu's dots, because in 2000 that
is precisely what it was.

1,547 hexes moved, 19.5M people's ground. **Units with a polygon and no rows: 80 → 14**, and
the dot count is identical before and after, which is the proof that this was placement and
not counting.

**Hainan's eleven are left blank deliberately**, and the distinction matters: 澄迈, 临高,
定安, 屯昌, 东方, 乐东, 陵水, 昌江, 白沙, 琼中 and 保亭 are ordinary counties that have
existed throughout. They draw nothing because **the Hainan volume of the 2000 census is
incomplete** — `cn.py` has always reported it, *"the shortfall IS Hainan, exactly"*, 3,159,377
people. Absorbing them into 五指山 or 儋州 would invent a geography the source never had, and
§3.5 says an undercount is marked rather than filled. **Hainan is now the one place on the
Chinese map where blank ground means "not in the source" rather than "nobody lives here"**,
and that is worth a `note_public` sentence if it is ever drawn on.

#### WHAT IS LEFT, AND IT IS SMALL

- **The parent of a split is chosen by longest shared boundary inside the prefecture,
  hand-corrected where the administrative history is known.** The residual risk is naming a
  sibling rather than the true parent, which moves people between two adjacent districts of
  one city. Bounded, and far smaller than the blank it replaces.
- **Hainan's 3.16M** still needs a complete 2000 volume, or another source.
- The 14 remaining blanks, above, all deliberate.

### 14.20 CLDS arrived, and it replicates the layer China's weakest colour rests on — FOUND 2026-09-08, and DECIDED the same day: it is EVIDENCE, not dots

Anita registered a ScienceDB account and retrieved `CLDS2016.rar`, which is the lead §14.16
ranked first and could not walk itself. `sources/cn_clds.md` is the full record and
`sources/cn_clds.py` reproduces every number here. **Nothing is wired into `countries.py`;
what to do with it is below and it is Anita's call.**

21,086 respondents, 29 provinces, 402 communities, one wave (2016). The religion variable is
`I7_1 宗教信仰`, single choice, and it is `self_id` — so unlike CFPS this one can sit beside
the rest of the map, which is §14.15's reframing paying off exactly as predicted.

#### THE FINDING WORTH THE MOST: A SURVEY CAN BE CHECKED AGAINST A BUILDING

CLDS's community questionnaire asks the interviewer whether the community has a church, a
temple, a mosque, a Daoist temple or an ancestral hall. §14.15 ruled out the registered-venue
registry as a *magnitude* source and that stands — §2.6, and China's registry omits house
churches. **But a venue observed in the same community whose residents answered the question
is a coherence check, and it is the first one this country has ever had that is not another
survey:**

| interviewer found | communities | matching self-report | |
|---|---|---|---|
| a church | 37 of 398 | **6.46%** Protestant vs 1.53% | **4.2x** |
| a temple | 106 | 11.97% Buddhist vs 4.31% | 2.8x |
| a mosque | 16 | 51.73% Muslim vs 0.88% | 59x |
| an ancestral hall | 69 | 12.96% Buddhist vs 4.97% | 2.6x |

**The church row is the one that matters**, because §14.16 drew Protestantism flagged and
`cn_cgss.py` calls its evidence *"a failure to demonstrate signal"*. People who call themselves
Protestant live, four times over, where the churches are. That is not a survey agreeing with a
survey.

**The general rule, and it generalises past China:** when a self-report layer is thin, look for
a variable in the SAME instrument that was recorded by the interviewer rather than answered by
the respondent. It is not a magnitude and §2.6 still forbids using it as one, but it is
independent of every bias that makes the self-report thin.

#### AND THE SECOND: RANK STABILITY ACROSS WAVES IS NOT RANK STABILITY ACROSS SURVEYS

§14.10's fifth condition asks whether a geography is stable, and §14.16 could only test CGSS
against itself, getting **+0.17** for Protestantism and drawing it anyway with a disclosure.
CLDS answers the question the condition was actually asking:

| | CGSS 2012 vs 2021 | **CLDS 2016 vs CGSS pooled** |
|---|---|---|
| `buddhism.mahayana` | +0.63 | **+0.596** |
| `christianity.protestant` | **+0.17** | **+0.595** |

CLDS independently puts **Henan first at 10.9%**, the largest Protestant cell in either survey
(97 respondents). **A wave-to-wave wobble measures the temporal instability of REPORTING; it is
not evidence that there is no geography.** Two surveys sharing no fieldwork, no house and no
questionnaire agreeing at +0.60 is. §14.16's disclosure in `note_public` is now stronger than
the evidence requires, and that is worth fixing whichever way the drawing decision goes.

#### THE COMMUNITY COUNT IS THE THING TO CONDITION ON, PROVED FROM BOTH SIDES

§14.16 showed CGSS's provincial cut of Islam was a lottery. CLDS fails the same census-margin
test in the *opposite* direction, which promotes a suspicion about one survey into a rule about
survey design:

| | census | CGSS | CLDS | CLDS communities |
|---|---|---|---|---|
| Xinjiang | 58.3% | 92.0% | **61.3%** | 17 |
| Ningxia | 34.5% | 90.4% | **1.1%** | **4** |
| Qinghai | 16.9% | 1.1% | 20.5% | **4** |

**Ningxia's four communities returned four Muslims between them** in a province a third Hui —
and those same four communities produce its **18.4% Buddhist** figure, third on the drawn list
and pure accident. Where the count is high the margin is reproduced almost exactly: Xinjiang
1.05x, Beijing 1.02x, Gansu 1.15x. **Neither survey is better; the design is.** `cn_clds.py`
names the six thin provinces on every run.

#### WHAT DOES NOT EXPLAIN THE LEVEL GAP, SO IT IS NOT RE-PROPOSED

CLDS puts Zhejiang at 36.1% Buddhist against CGSS's 14.8%, and Fujian 32.1% against 11.5%.
Three explanations were tested and all three fail:

- **CLDS offers no folk-religion option, so folk believers pick 佛教.** No: the gap correlates
  with CGSS's folk share at **+0.10**, adding folk to CGSS *lowers* agreement (+0.662 ->
  +0.492), and Guangdong has the highest folk share in the country with **no gap at all** while
  Zhejiang has almost the lowest with the largest gap.
- **Different universes.** No: restricting CLDS to CGSS's 18+ moves Buddhism 6.50% -> 6.51%,
  and CLDS's age gradient is flat (12.1 / 13.0 / 12.0 / 11.7% across four bands), which also
  quietly contradicts the assumption that the old are more religious.
- **A few lucky communities.** No: Zhejiang's 36% is spread over all seventeen of them, at 74,
  65, 62, 59, 52, 49, 41, 38, 35, 23, 22, 16, 16, 9, 8, 3 and 3 per cent.

**So the shapes agree and the levels do not, for reasons nobody here can name.** That is §3.4's
split arriving as a choice rather than a vintage, and it is why this cannot simply be averaged
in.

#### THE COST, WHICH IS NOT TECHNICAL

The archive's own use agreement, clause 2(3): the data **may not be used for any commercial or
political purpose**, and 2(2) forbids releasing raw data to a third party. ScienceDB advertises
CC BY 4.0; §6b's rule is that the badge is the depositor's claim and the terms are the
origin's, and here the two flatly disagree. **CGSS is CC0 and has no such clause.** So drawing
CLDS trades a commercially clean country for a better-evidenced one, and
[[reference_poster_commercial_licences]] gains a second religiondots blocker. A citation is
also mandated verbatim and is in `sources/cn_clds.md`.

#### THE DECISION: EVIDENCE ONLY, NO DOTS — ANITA, 2026-09-08

*"yeah we dont have to place dots."* And, asked as a question rather than a claim: **are the
percentages mostly the same as what we already have? THE ORDERING IS. THE LEVELS ARE NOT, AND
THE FIRST DRAFT OF THIS SECTION SAID OTHERWISE — corrected 2026-09-08.**

| | drawn (CGSS pooled) | CLDS 2016 | |
|---|---|---|---|
| Buddhism, population-weighted | 4.44% | **5.93%** | CLDS **+34%** |
| Protestantism | 1.63% | **2.55%** | CLDS **+56%** |
| provinces within ±50% of the drawn share | | 13 / 29 and 6 / 29 | |
| median absolute deviation per province | | **42%** and **67%** | |
| Spearman on the ordering | | **+0.596 / +0.595** | |

**In dots that is 58.4M Buddhists against 78.1M, and 21.3M Protestants against 33.3M.** So the
two surveys agree on which provinces are Buddhist and disagree by a third to a half on how
many people that is. "A source that agrees adds confidence, not information" is the wrong
summary and is not why the decision goes this way.

**The decision holds on two other grounds.** First, nobody can say which level is right: the
gap survives every explanation tested (folk absorption, universe, clustering, vintage — CLDS
2016 sits ~32% above what CGSS's own 2012→2017 slope predicts for 2016), so swapping would
trade one unexplained level for another. Second, **CLDS cannot add geography at all**: it is a
21,086-person survey whose county codes are randomised by the depositor and whose prefecture
cells are too thin to use, with 11 of 157 cities reaching ten Protestant respondents. Province
is its ceiling, which is exactly where CGSS already sits. So there is no redraw it enables,
and drawing it would cost clause 2(3).

**What it does tell us, and this should not be buried: the drawn levels may be low by a third.**
That is a live input to the levels decision below, not a reason to redraw today.

**This is the same shape as §14.13's finding about CFPS and it is worth naming as a rule: a
survey is worth chasing for what it can CHECK, not only for what it can DRAW.** CLDS was ranked
first in §14.16 as the route to a better Han layer. It is not that. It is the first external
check the Han layer has ever had, which is more useful and was not what anyone was looking for.

What the decision leaves standing:

- **`note_public` is NOT to mention any of this. Anita, 2026-09-08: *"lets ntot note it
  publically. its whtaever."*** A draft rewrite of the Protestant disclosure was offered,
  citing the second survey and the church check, and declined. The disclosure therefore still
  says the provincial ordering is unsteady, which remains true of CGSS on its own and is now
  known to be an incomplete account. **A deliberate understatement of the project's own
  confidence, not an oversight to fix later.**

  The commercial-licence argument originally recorded here for that choice **is withdrawn**.
  Anita, same day: *"im not gonna make this map commercial, so i dont care."* Clause 2(3) is
  moot for religiondots, and it was never the reason not to draw CLDS anyway — see the two
  real reasons above, which are that it adds no geography and that its levels are unexplained.
  **The licence was the third reason and it was given too much weight when this section was
  first written.** With it gone, *pooling* CLDS into the CGSS layer is blocked only by the
  level gap and the missing folk category, which is a smaller objection than it looked.
- **`cn_cgss.py`'s docstring** calls the Protestant evidence *"a failure to demonstrate
  signal"*. That was accurate when only CGSS existed. Leave the sentence, add the cross-survey
  result beside it.
- **Islam stays on the ethnic derivation.** Nothing here changes that and CLDS's Ningxia is a
  second demonstration of why.

#### STILL OPEN, AND INDEPENDENT OF ALL OF THE ABOVE

1. **Shape from pooled waves, level from the most recent** — the §3.4 pattern China already
   uses for ethnicity, which `cn_cgss.md` raised and left as *"a decision, not a default"*.
   Worth 58.4M Buddhist dots against ~32.7M, and unmade with or without CLDS. **CLDS is mild
   evidence for the pooled end**: its 2016 level sits above what CGSS's own decline predicts
   for 2016, so the most recent wave is the low reading of the three rather than the true one.
2. **The remaining CLDS waves.** The ScienceDB deposit advertises 2012, 2014 and 2018; the
   archive retrieved is 2016 only. A second wave would give the cross-survey check a time
   dimension, and the account now exists.
3. **CNSDA HOLDS THIRTEEN CGSS WAVES AND THE CATALOGUE IS OPEN — swept 2026-09-08.**
   `www.cnsda.org` is up (`cnsda.ruc.edu.cn` does not resolve; use the .org). The catalogue is
   **826 datasets over 166 pages, readable with no account**, and the pagination parameter is
   `Projects_page`. It lists **CGSS 2003, 2005, 2006, 2008, 2010, 2011, 2012, 2013, 2015, 2017,
   2018, 2021 and 2023**, plus a merged 2003+2013 file and CLDS at `id=75023529`. §14.16 said
   five waves were behind this wall; it is thirteen, and **2023 is newer than anything drawn
   anywhere on this map for China**.

   Each dataset page carries `index.php?r=projects/download&id=<...>` links, five of them for
   CGSS 2023. **[[reference_cms_download_id_sweep]]'s open-download trick does not work here**
   — every one 302s to `site/login`, checked. The gate is real and the account is the only way
   through. Registration is `index.php?r=users/create`, labelled 免费注册, and its first step is
   an agreement checkbox behind a JS `fn_next()`, so the field list cannot be read from outside.
   Nothing on the visible page asks for 手机 or 身份证.

   **The thing to read at signup is the 数据使用协议, and the question is whether it bans
   commercial use the way CLDS's clause 2(3) does.** The dataset pages themselves show only an
   attribution requirement, which would make CGSS-via-CNSDA strictly better than CLDS on terms
   as well as on coverage. Unconfirmed, and it decides whether China stays sellable.

   Sweep gotcha, since it cost a run: the catalogue's hrefs are `index.php?...` on page 1 and
   `/index.php?...` on every paginated page, so a regex anchored to the bare form silently
   returns page 1 over and over and reports five datasets instead of 826.

   **What is NOT there: the China Religion Survey (CRS) 2015.** It appears on the homepage only
   as a *report* announcement (`site/article&id=126`); there is no CRS dataset in the
   catalogue. Do not go looking for it here.
3. **The A-Hmao county list** (§14.15), unchanged and still the largest known omission.

### 14.21 The evidence for China's weakest layer was judged on a single pair of waves, and five waves say it was the worst pair — BUILT 2026-09-08

Anita registered at CNSDA to reach the CGSS waves §14.20 ranked. **The free account does not
clear the gate: every download needs a separate reviewed data application**, which for an
unaffiliated applicant abroad is the shape that already refused this project once
([[reference_cfps_terms]]). She said it looked hopeless. It was not.

#### §6b'S MIRROR RULE WENT FIVE FOR FIVE

**CGSS 2010, 2011 and 2013 are sitting unrestricted in a replication package** —
`doi:10.7910/DVN/R1S5RP`, *"Meritocracy as Authoritarian Co-Optation"* — which also carries a
second copy of 2012. `sources/cn_cgss_fetch.py` pulls them.

**A replication package is normally a variable SUBSET and that is why nobody had looked.**
These are not: 963, 592 and 650 variables, and all three keep `s41` (province of interview)
*and* the religion block. Checking that was the whole of the work, and the DDI endpoint
(`/api/access/datafile/<id>/metadata/ddi`) answers it **without downloading the file**, which
is the cheap move worth remembering: Dataverse exposes variable names and labels for any
ingested table.

*Two-thirds of a walled archive's most wanted holdings were in other people's replication
packages.* Search the file level, not the dataset level, and search for the FILENAME a
researcher would have used (`cgss2013`), not the project title.

#### THE FINDING: A PAIRWISE STATISTIC ON ONE PAIR IS A SAMPLE OF SIZE ONE

§14.16 drew Protestantism flagged, and `cn_cgss.py` called its evidence *"a failure to
demonstrate signal"*, on a 2012↔2021 rank correlation of **+0.17**. With three waves there was
exactly **one** pair to compute. Five waves give ten:

    2010 vs 2017  +0.857     2012 vs 2013  +0.702     2013 vs 2021  +0.409
    2010 vs 2012  +0.742     2012 vs 2017  +0.682     2017 vs 2021  +0.320
    2012 vs 2013  +0.702     2013 vs 2017  +0.572     2010 vs 2021  +0.214
    2010 vs 2013  +0.545                              2012 vs 2021  +0.166

**Median +0.559, and the pair the old conclusion rested on is the worst of the ten.** Every
weak pair involves 2021, the wave with 19 provinces and the smallest cells, so what read as an
unstable geography is mostly one unstable WAVE. CLDS 2016 independently ranks the provinces at
+0.595 against this pool (§14.20), which is the same answer from outside.

**The general rule, and it is not about China.** Where a §14.10 condition is computed from a
statistic over PAIRS, count the pairs before believing the number. Three waves feel like
enough data and give one degree of freedom. This project came within one decision of
disclosing a layer as unreliable on the strength of a single unlucky comparison.

#### WHAT IS DRAWN NOW

Anita's call: pool 2010 and 2013, hold 2011 back as an independent check because it is small
(5,620) and carries no weight column.

| | 3 waves | 5 waves |
|---|---|---|
| respondents | 32,495 | **55,637** |
| provinces | 29 | **30** |
| Buddhist respondents | 1,592 | 2,793 |
| Protestant respondents | 585 | **1,016** |
| provinces with <10 Protestants | 13 / 29 | **9 / 30** |
| `buddhism.mahayana` | 58.4M | **62.4M** |
| `christianity.protestant` | 22.1M | **24.7M** |

The dots barely move, by design: national levels rise 6% and 10%. **The gain is evidential, not
cartographic**, and that is the honest way to describe it.

#### TWO THINGS THAT HAD TO BE HANDLED, AND ONE IS A TRAP

**2013 is pooled UNWEIGHTED**, because the open copy has no weight column. Measured rather than
assumed: on the waves that DO have weights, weighting moves a province's drawn share by a
median 0.15 points and preserves the provincial ordering at +0.98 (2012) and +0.93 (2017). A
disclosable cost, recorded in each row's `note`.

**Xizang is read and then DROPPED, and this is the trap.** CGSS 2010 is the only wave that
samples Tibet, 79 respondents, **53 of whom answer 佛教 — 58.5%, which would have made Tibet
the top `buddhism.mahayana` province in China.** CGSS's answer set has no 藏传佛教 row (CLDS's
does), so those 53 are Tibetan Buddhists about to be filed as Mahayana, in the one province
§14.5 already draws as Vajrayana from ethnicity. `cn_cgss.py`'s own docstring had justified
reading CGSS 佛教 as "Han Mahayana practice" **on the premise that Xizang is not sampled at
all** — a premise that was true of the old wave set and silently false of the new one.
`DROP_PROVINCES` enforces it instead of assuming it.

*Adding data can falsify a premise that an existing argument rests on, and nothing will warn
you.* The check that caught it was reading the top-five line of the module's own output.

#### STILL WALLED, AND NOW PRECISELY

**CGSS 2015, 2018 and 2023.** Searched hard on Dataverse and figshare, file level and dataset
level: 2015 exists only as a 0.5 MB subset, 2018 and 2023 not at all. **2023 is the one worth
an application**, being newer than anything drawn for China anywhere on this map, and the only
thing that could say whether the fall through 2021 continued or bottomed out.

### 14.22 China gets a Chinese religion — folk practice is drawn, and the reason it had been refused was one wave's questionnaire — BUILT 2026-09-08

Anita, looking at a finished China: *"i feel like its a bit sad that we dont have any
'confucian' or 'daoist' or anything other than protestant/islam/buddhist dots."* She is right,
and it was fixable from data already on disk. **44.4 million `chinesefolk` dots**, the second
largest colour in the country, above Protestantism.

#### THE CATEGORY HAD BEEN JUDGED ON A NUMBER THAT ONE WAVE'S INSTRUMENT PRODUCED

§14.16 tested 民间信仰 against §14.10 and refused it. On five waves it clears the same bar
Protestantism clears, and on most axes it clears it better:

| | respondents | provinces cell <10 | median rank stability |
|---|---|---|---|
| folk | **1,173** | 16 / 30 | **+0.565** |
| Protestantism (drawn since §14.16) | 1,016 | 9 / 30 | +0.559 |

**The thing that had made it look unusable was the 2021 wave.** Its national share runs
2.90 → 3.43 → 1.91 → 2.11 → **0.27** per cent while Buddhism moves 4.66 → 3.76 and everything
else drifts. 2021 is the **single-choice** wave.

##### THE FIRST EXPLANATION WAS WRONG AND TESTING IT IS THE LESSON — CORRECTED SAME DAY

This section first argued that single choice makes a respondent who tends a Mazu shrine *and*
calls themselves Buddhist pick one, and that folk religion is the answer that loses **to
Buddhism**. 2021 was excluded for folk on that basis.

**That argument makes a prediction and the prediction fails.** If folk answers were being
absorbed by Buddhism, Buddhism would RISE in 2021. It falls, 4.66 → 3.76, and `none` gains 3.1
points. **The folk respondents went to NO RELIGION.** What single choice actually does is make
people who tend a shrine say they have no religion at all, because they do not consider it a
宗教 — so the multi-select waves measure a **permissive** threshold and 2021 a **strict** one,
and excluding 2021 was silently choosing the permissive reading.

**And it bought almost nothing**: 3.05% pooled over five waves against 3.31% over four, 37.2M
dots against 40.4M. A per-category vintage inconsistency, and the appearance of dropping the
inconvenient wave, for 8% more dots. Anita's call to reverse it, 2026-09-08, on being asked
whether the layer was honest. **`EXCLUDE_WAVES` stays in `cn_cgss.py` as machinery and is
empty**, with the reversal written next to it.

***A wave that disagrees is evidence about the QUESTION, and dropping it is a claim about
which asking is right.*** That claim needs a mechanism that survives being tested, not one
that merely sounds plausible — and the test is usually one line, because a mechanism that
moves a category has to move some other category too.

#### WHAT IT LOOKS LIKE, AND WHY IT IS THE RIGHT COLOUR FOR THIS COUNTRY

**Guangdong 18.5%, Fujian 16.7%**, Hainan 8.7%, Guangxi 5.3%, against under 1% across most of
the north. That is the Mazu and Guandi coast, which is exactly what the answer option names
(`民间信仰（拜妈祖、关公等）`) and exactly what the literature would predict unaided. The
gradient was not fitted; it fell out, and it is the same gradient with or without 2021.

**China now has a colour that is not an imported religion**, which was the complaint. 40.9M
dots, the second largest layer in the country; grey falls 91.0% → **88.1%**.

The `chinesefolk` node already existed and was drawn in seven countries — Indonesia's 117 dots,
plus the Canadian, Spanish, French, Italian and Mauritian diasporas. **China was the hole in
the middle of its own diaspora.**

#### WHAT STAYS OUT, AND THE TWO REASONS ARE DIFFERENT

- **Daoism: 143 respondents, 25 of 30 provinces under ten.** Fails on cell size alone. Worth
  recording so it is not reopened as an oversight: **the familiar "hundreds of millions of
  Daoists" figures are BELIEF measures.** On `self_id`, which is the basis this whole map is
  drawn on, Daoism is about 0.3% of China, so the small number IS the finding. §3.1 forbids
  reaching for the belief figure to make the layer bigger.
- **Catholicism: 129 respondents, 28 of 30 under ten.** Same failure, no nuance.
- **Confucianism cannot be drawn at any sample size, because it is not an ANSWER.** Neither
  CGSS's list nor CLDS's offers 儒教. Korea draws 75 Confucian dots and Thailand 16 because
  those censuses ask; China's survey does not. *Check the answer set before treating a missing
  category as a data problem.*

#### THE DISCLOSURE, WHICH IS THE COST OF DRAWING IT

Anita asked the right question about this layer before it had been asked internally: *"do you
think this folk religion map is honest? idk."* The answer is that the geography is and the
magnitude is the softest on the Chinese map, so `note_public` leads with the magnitude:

- **16.6x instrument sensitivity**, 4.40% to 0.27% across waves, against Buddhism's 1.5x,
  Protestantism's 1.8x and Islam's 3.1x. **No other drawn category on this map is within a
  factor of five of that**, and it is the reason the note calls these the least certain dots
  on it and the number a floor rather than a count.
- **Sixteen of thirty provinces hold fewer than ten folk respondents.** They contribute 1.7M
  of the total, about 4%, so this is smaller than it sounds; Jilin's entire folk population
  rests on one respondent and is ten dots.
- **75 respondents named both folk religion and Buddhism**, 8.5% of the folk cell, and the
  multi-select flags are independent so those people are drawn on both nodes. 3.6 points of
  Fujian. Bounded, disclosed, not corrected.

#### AND THE MEASUREMENT THAT MAKES THE WHOLE LAYER LEGIBLE — SLSC 2007

Anita fetched the **Spiritual Life Study of Chinese Residents** from ARDA the same day
(`sources/cn_slsc.md`, free, no application). It is not drawable — 56 sampling points is
§14.16's lottery design — and it is worth more than most drawable things, because it is **the
only survey here that asks the naming question and the practice question of the same 7,021
people**:

| | | |
|---|---|---|
| *Do you have any religious belief?* | yes | **15.8%** |
| *Have you worshipped God or gods/spirits in the past year?* | "I never worship" | **37.6%** |

***About four times as many Chinese people practise as will name it.*** §14.14 and §14.16 both
asserted that gap from Pew's summary; this measures it, in one survey, on one sample, and
`note_public` now carries it in those terms. It is the honest frame for every grey dot in China
as well as for the folk ones.

It also answers the Confucianism question from the respondents' own side: asked *do you think
Confucianism is a religion*, **58% said no** and 24% said hard to say. So the reason it is not
drawn is not merely that the answer sets omit it.

### 14.23 Hainan had 3.34 million people in the wrong county, and the guard that was supposed to prevent it only guards against a different failure — FIXED 2026-09-08

Anita, on a finished China: *"osme things i think we should take care of are — hainan
placement"*. She was right, and it was much worse than a blank hole. **38.5% of the province
was being drawn in the wrong county.**

#### THE HOLE WAS KNOWN SINCE THE FIRST BUILD AND THE WRONG THING WAS CONCLUDED FROM IT

`sources/cn.py` has reported from day one that the 31 provincial volumes fall exactly
3,159,377 short of the published 2000 total, and that *"the shortfall IS Hainan, exactly"* —
whose file carries 14 of its 24 county-level units as a name, a tab, and nothing else. That
check was correct and is still there. What was concluded from it was that the gap cost the map
nothing, because Sanya holds the Utsul Muslims and Sanya is present. **Both halves of that
sentence are true and the conclusion does not follow**, because it is a statement about colour
and the failure was about placement.

| | drawn before | actually held, 2010 |
|---|---|---|
| the ten Hainan units in the volume | 8,671,485 | 5,334,323 |
| the eleven that are missing | 0 | 3,336,751 |

Danzhou was drawn at **3,268,523 against a real 932,362** — which would have made it one of
the larger county-level units in China. Wuzhishan at 472,425 against 104,122. Sanya was given
**595,912 Li where the 2000 census counted 183,865**. Every Hainan county's Han was inflated ×1.841 and its Li
×3.241, and Baisha, Changjiang, Ledong, Lingshui, Baoting, Qiongzhong, Dongfang, Chengmai,
Lingao, Dingan and Tunchang — the whole centre and west of the island — drew nothing.

#### THE GENERALISABLE PART: A GUARD AGAINST SPREADING ONLY GUARDS THE CASE IT WAS WRITTEN FOR

`cn.py`'s rescale carries a comment that has been right about itself and silent about this:

> Denominator is the sum over ALL county rows in the file, not just the resolved ones. Using
> the resolved subset would silently redistribute an unresolved county's people into its
> neighbours, which is spec §8.1's Connecticut failure wearing a different hat.

That is exactly correct **for a county whose name failed to resolve** — it is in the file, so
the denominator sees it, so its people are dropped rather than spread. §14.17 leaned on this
and was entitled to. But Hainan's eleven are *not in the file at all*, so the denominator never
sees them, and the province's whole 2010 nationality vector went to the ten survivors. **The
guard was load-bearing against one failure mode and mute about its twin, and the comment
asserting it read as though it covered both.**

The lesson is §14.17's again in a new shape and it is worth stating as a rule: **a
reconciliation constraint is only as good as the completeness of the thing in its
denominator.** When a margin is enforced — a provincial total — against a structure that is
missing rows, the enforcement does not fail loudly. It silently pushes the missing rows' mass
into whatever remains. §14.17 was this error with the rows present but unjoined; this is the
rows absent. Both were invisible because the totals reconciled perfectly, which is the point:
**arithmetic consistency is not evidence of meaning** (§3.10d), and reconciling to a margin is
arithmetic consistency.

#### THE GAP IS IN THE VOLUME, WHICH WAS CHECKED RATHER THAN ASSUMED

The Hainan dataset on the Harvard Dataverse holds **111 tables**, not the one A0106 we use. All
111 were pulled and all 111 carry the same 24 unit rows with only 9 or 10 carrying data. There
is no other table to reach for and no other digitisation of this volume in the open. **Pulling
a source's whole dataset before concluding it is short is cheap and worth doing** — it is what
established that `J46A0201` through `J46L0814` had nothing to add.

#### THE FIX: ONE PROVINCE RECONCILES TO COUNTY TOTALS, AND IT IS FORCED

You cannot divide a provincial nationality total across counties when 42% of the counties are
absent from the structure source. So Hainan — and only Hainan — is reconciled to its own
published 2010 county totals:

- **the ten present units keep their 2000 nationality shares**, which is the only thing the
  volume tells us about them, and are scaled by their own `county_2010 / county_2000`. Factors
  run ×1.033 to ×1.421 against the ×1.841 and ×3.241 they carried. Haikou's four census rows
  share 海口市's single 2010 figure in proportion to their 2000 populations, because 琼山市
  merged into the city in 2002.
- **the eleven missing counties are written at their published 2010 total** on one category,
  `Unpublished`, which claims no nationality because none was published. `taxonomy/cn2000.py`
  sends it to `unknown` alongside Han and Li, with the argument recorded in `NOT_ASSERTED`.

Hainan now draws **8,671,074 of a published 8,671,518**; the 444 missing are 西南中沙群岛,
which `sources/cn_geo.py` leaves blank as islands the 2000 census had no county for. The
national figure lands 428 people from the census, of which 411 are those islanders.

**And coloured dots move as well as grey ones, which was not obvious in advance.** §14.16's
CGSS layer carves a province's Buddhist, folk and Protestant shares out of each unit's
`unknown` residual, so it could only ever colour units that had rows: **229,000 Mahayana
Buddhists and 290,000 folk-religion adherents were being drawn on Hainan's coast and now
appear in its interior**, where the people they represent live. A placement error in a grey
layer propagates into every layer computed on top of it.

#### TWO PUBLISHED TABLES AND A GROWTH RATE AGREE, WHICH IS WHY THE CONSTANTS ARE TRUSTED

The eleven counties' 2000 totals come from the NBS *第五次人口普查公报——海南* in 万人 to two
decimals, so each is exact to ±50. **They sum to 3,159,600 against the 3,159,377 shortfall
`cn.py` has reported since the first build — 223 apart, inside ±50 × 11.** Independently, the
2010 communiqué publishes a per-county average annual growth rate, and it reproduces the pair:
Dongfang at 1.31%/yr takes 358,000 to 408,100 against a published 408,309. A single figure
recovered from a rounded table would have been a guess; three sources agreeing is the check.

`cn.py` also re-reads the volume each run to confirm the eleven are *still* blank, so if
Harvard ever completes the digitisation it reports that instead of writing published totals
over real data.

#### WHAT IS GIVEN UP, AND WHY IT IS A LABEL RATHER THAN A DOT

Hainan's per-nationality provincial reconciliation. It is now **reported as a residual rather
than enforced**, which is the honest form of a constraint that cannot be met: 2.49M Han, 777k
Li and 45k Miao sit inside `Unpublished`.

**Li, Han, Miao and Zhuang all resolve to `unknown`, so not one dot changes colour and not one
person is lost.** What is lost is the word `Li` on rows that were never drawn as Li. All of
Hainan's religio-ethnic population is about 13,600 people and every one of them is in a county
the volume covers; **the Hui reconcile to within 200**, because the county growth factors
happen to bracket the provincial one. The entire claiming residual is **Kazakh 1,535 and Dai
779**, which is §6's migration case — 14 Kazakhs in Hainan in 2000 against 1,553 in 2010 is a
×110 factor, and the old code was applying it. Two dots, now dropped rather than invented
(§3.5).

#### THIS IS NOT §6'S REJECTED IPF AND IS NOT OFFERED FOR THE OTHER THIRTY PROVINCES

`sources/cn.md` §6 records that constraining to modern county totals was considered and
rejected, because urban growth in Xinjiang and Tibet was disproportionately Han and the method
would inflate the Uyghur and Tibetan share of exactly the cities where that figure is most
contested. **That argument is about a contested minority share in a growing city, and the only
group this moves at any size is the Li, who claim nothing.** The other thirty provinces
reconcile provincially and should go on doing so; this is a forced local exception, recorded so
it is not read as a new default.

What it does inherit is §6's standing caveat in a sharper form: Sanya's composition is frozen
at 2000 while its size is 2010, and Sanya's growth was overwhelmingly Han in-migration, so its
261,297 Li are too many. That is an `unknown`-on-`unknown` error, and it is a great deal
smaller than the 595,912 it replaces.

### 14.24 Hong Kong is drawn, its coefficients come from this map's own countries, and the figure everybody quotes is refused with evidence — BUILT 2026-09-08

Anita: *"hong kong religion coverage"*, and then, once the options were laid out, *"self
identification at territory grain seems like a good way to go... i think supplementing it with
ethnicity / verifying it with ethnicity would be nice."* Both halves of that are built.

**Hong Kong, Macau and Taiwan were all excluded from the mainland by `cn_geo.py`'s
`NOT_MAINLAND` and none of the three had ever been considered** — `hk` appears nowhere in
spec.md, sources.md or queue.md before today. So 7.4 million people sat blank against a fully
drawn China. `sources/hk.md` is the record; this is the reasoning.

#### IT IS CHINA'S PROBLEM AGAIN AND IT GETS CHINA'S ANSWER

The 2021 Population Census publishes the 46 topics it covered and religion is not among them,
in that round or any before it, and Hong Kong is absent from UNSD table 28. So it is built in
the same two layers as the mainland and in the same order: §14.5's ethnic derivation at the 18
District Council districts, and a `self_id` survey for the territory carved out of the
`unknown` residual, which is `_cn_counts`'s arithmetic unchanged.

| layer | source | grain | tier |
|---|---|---|---|
| Indonesian and Pakistani → `islam`, Filipino → `christianity.catholic.latin` | 2021 census ethnicity | 18 districts | `modelled` |
| Buddhism, Protestantism, Catholicism, Daoism, Hinduism, Sikhism | Hong Kong Political Culture Survey 2021 | territory | `modelled` |
| everyone else | the census counted them | 18 districts | `derived` |

#### THE GENERALISABLE FINDING: A COEFFICIENT CAN COME FROM THE MAP'S OWN COUNTRIES

§14.5 wants a derivation coefficient "documented rather than fitted", and §14.12 is the
standing warning about laying a national share over a selected subpopulation. Hong Kong's
three coefficients are **Indonesia's, Pakistan's and the Philippines' own census figures as
this project already draws them**:

    id  islam                        87.51%
    pk  islam                        96.47%
    ph  christianity.catholic.latin  78.88%

recomputable at any time with `countries.COUNTRIES[cc]["counts"]()` grouped on node. What that
buys is real but narrow: the coefficient **inherits every correction ever made to the source
country** and cannot drift away from the rest of the map, where a number transcribed from a
webpage does both.

**AND THE FIRST DRAFT OF THIS SECTION OVERCLAIMED IT — Anita, 2026-09-08:** *"i think doing
diaspora coefficients from maps own drawn countries is probably pretty bad for large origin
countries cuz the people migrating are probably skewed in some way."* She is right, and the
overclaim was saying this "should be the default for any diaspora derivation". **Sourcing the
number well does nothing about the thing that actually breaks it, which is §14.12: migration
selects, and it selects on region, class and ethnicity, which are exactly the axes religion
varies on.** A national share is the right number for the origin country and the wrong number
for the stream that left it, and the bigger and more religiously varied the origin, the worse
it gets — Nigeria's or India's national share would be close to meaningless applied to a
particular diaspora.

So the rule is a permission with conditions, not a default. **A national coefficient over a
migrant stream is admissible when at least one of these holds, and it should say which:**

1. **The origin share is near 1**, so no plausible selection moves it much. Pakistan at 96.47%
   is this case: a Pakistani migrant stream is Muslim almost whatever it selects on.
2. **The direction of the selection is known and stated.** Hong Kong's Indonesians are ~93%
   domestic workers from Central and East Java, which is *more* Muslim than Indonesia as a
   whole, so 87.51% is a floor; its Filipinos come from Luzon and the Visayas rather than
   Muslim Mindanao, so 78.88% Catholic is also a floor. Both are named in `hk2021.py`'s REVIEW
   as floors rather than adjusted upward, per §14.12's own lesson.
3. **There is an independent check on the result.** Here there is: the census's ethnic counts
   and the survey agree to within 20% on the Muslim total.

**And when none of them holds, the derivation should be refused rather than sourced better** —
which is exactly what happened to Hong Kong's Indians and Nepalese two sections below. They
were refused for the selection problem, and no amount of good provenance for India's or
Nepal's national share would have fixed them. That is the honest summary of this technique:
it improves the *provenance* of a coefficient and does nothing for its *applicability*.

China's Joshua Project coefficients could not use it at all, because no country on the map
publishes a Lisu share; a migrant-origin derivation usually can, and should still pass the test
above before it does.

#### THE REFUSAL, WHICH IS THE MOST USEFUL THING HERE

**`gov.hk`'s *Hong Kong: The Facts — Religion* is where every published figure about religion
in Hong Kong comes from, and it is not drawn.** Over 1 million Buddhists, over 1 million
Taoists, 1,040,000 Protestants, 390,000 Catholics, 300,000 Muslims, 100,000 Hindus, 15,000
Sikhs. Every one is the religious body's own estimate of itself; the sheet says so outright for
Islam (*"according to the Incorporated Trustees of the Islamic Community Fund"*) and Sikhism.

That alone would make it a basis this map has nowhere else and §3.1 would forbid mixing it. But
it also **fails the only external checks there are, in both directions**:

| | gov.hk | what checks it |
|---|---|---|
| Protestants | **1,040,000** (Jan 2026) | the same office said **480,000** in July 2022; over those years the churches' own 2024 Hong Kong Church Survey counted **255,091 congregants and 197,935 at weekly worship, down 26% in five years** |
| Muslims | 300,000 | census ethnicity ~148,000; the survey ~176,000 |
| Hindus | 100,000 | census ethnicity ~72,000 as an *upper* bound; the survey ~44,000 |

**The government's Protestant figure more than doubled over exactly the period in which the
only measurement of it fell by a quarter.** And the rule that decides the other two rows is
worth stating on its own: **when two sources with no lineage in common agree with each other
and a third disagrees with both, the third is the one to leave out.** A census of 7.4 million
and a survey of 3,740 have nothing to do with one another; their agreeing to within 20% is
evidence, and it is the evidence Anita asked for when she said verifying it with ethnicity
would be nice.

#### THE SECOND FINDING: A MIGRANT DERIVATION'S GEOGRAPHY MAY BE AN EMPLOYMENT GEOGRAPHY

The expectation going in was an enclave map. It is not one. The Muslim share runs from **3.00%
in Wan Chai to 1.51% in Kwun Tong** and the Catholic share from 11.33% to 4.95%, with both
**highest in the wealthiest districts on Hong Kong Island**. The reason is that Hong Kong's
Indonesian and Filipino residents are ~93% live-in domestic workers, so what the census's
ethnicity column locates is not where a community settled but **where the households that
employ them are**.

The enclave geography does exist, and it belongs to the two nationalities this map refuses to
derive from: Yau Tsim Mong holds 42.3% of Hong Kong's Nepalese and 18.8% of its Indians against
4.9% of its Pakistanis. **So the layer with the flat geography is the one that could be drawn
and the layer with the sharp geography is the one that could not** — worth knowing before
reading any diaspora-derived layer as a settlement map.

#### WHY INDIAN AND NEPALESE ARE COUNTED AND LEFT GREY

They are the obvious next candidates, 42,569 and 29,701 people, and both are §14.12 in clean
form. Hong Kong's Indian community is disproportionately Sindhi Hindu and Punjabi Sikh rather
than a cross-section of India; its Nepalese are the families of Gurkha soldiers, recruited from
the Gurung, Magar, Rai and Limbu, who are far more Buddhist and Kirat than Nepal's 81% Hindu
average. **The selection runs along exactly the axis the coefficient would have to be stable
on.** Thai is the close call and is argued in `taxonomy/hk2021.py`'s REVIEW; it is 12 dots.

Hinduism and Sikhism therefore come from the survey, where they rest on 22 and 2 respondents,
drawn at territory grain with no geographic claim. That is Guatemala's rule (§9bi): nobody is
deleted, only the claim to know where they are.

#### ISLAM IS TAKEN FROM THE CENSUS AND NOT FROM THE SURVEY, WHICH IS CHINA'S RULE

89 Muslim respondents in a 72-cluster design carry no geography; the census counts 142,065
Indonesians and 24,385 Pakistanis exactly and by district. §14.16 refuses CGSS's Islam for the
mainland on the same reasoning. The two agree on the magnitude to within 16%, which is what
makes either believable.

#### HONG KONG MEASURES §14.22'S GAP IN ONE INSTRUMENT, WHICH THE MAINLAND CANNOT

65.83% of Hong Kong reports no religious affiliation and it is emphatically not drawn as
irreligion, because **the same table records that 2,097 of those respondents — 56.07% of the
whole sample — practise folk religion anyway.** China has to borrow that finding from Pew and
from the 2007 Spiritual Life Study, across instruments and across years; here one survey asked
both questions of the same people in the same interview. It is the cleanest statement of the
naming-versus-doing gap anywhere on this map.

**Nothing in Hong Kong is drawn on `chinesefolk` all the same**, and that is §3.1 rather than
timidity: what the 56% measures is practice, and the naming layer beside it cannot be mixed
with it. The 1988 and 1995 surveys the same table prints *did* offer folk religion as an
affiliation and found 23.0% and 15.3%; the 2021 instrument dropped the answer box. Carrying
15.3% forward thirty years is available and is almost certainly the wrong thing to do.

#### THE CHECKS, AND ONE THRESHOLD THAT WAS WRONG AND WAS FIXED RATHER THAN LOOSENED

Table 8.1 of the census's thematic report is parsed out of a PDF whose rows print ten numbers
*before* the district's name, so a slipped column would be silent. Two independent things have
to hold and neither is a tolerance: its Filipino and Indonesian columns must reproduce
`DC_21C.CSV`'s exact counts (worst 0.05 pp over 36 comparisons), and its South Asian subtotal
must equal the weighted mean of its own four South Asian columns (worst 0.07 pp over 18
districts).

**The Kontur correlation check was written at r ≥ 0.95 and failed at 0.87, and the fix was to
correct what it was measuring rather than to move the number.** Hong Kong is the hardest place
on earth for a footprint-derived population grid: it reads a 40-storey housing estate much as
it reads a village, so it undercounts the vertical districts (Wong Tai Sin 0.60×, Sham Shui Po
0.75×) and overcounts the spread-out ones (North 1.61×). **The disagreement has a shape, and a
scrambled join has no shape** — it pairs a large district with a small one and the correlation
collapses toward zero. The check also cannot affect the output at all, because dots per
district come from the census and Kontur only decides which street inside a district they land
on. It now asserts against a scramble and reports the band, which is §9i's principle and
`cn_geo.py`'s.

#### A SIDE EFFECT WORTH KNOWING: HONG KONG IS WHERE DAOISM GETS DRAWN AT ALL

**287 of the 302 Daoism dots on this map are now Hong Kong's**, the other fifteen being
diaspora counts in Australia, Canada, the UK and New Zealand, whose censuses carry a Daoist
box. Mainland China draws none: CGSS asks, and 80 respondents in 32,495 said Daoism, which
§14.16 judged too thin to place. So selecting Daoism now shows Hong Kong and a scatter of
migrant communities, which is an honest picture of **where the question has been asked** rather
than of where Daoists are. Hong Kong's own figure rests on 151 respondents.

#### THE BIGGEST WEAKNESS, AND IT IS SPATIAL: HONG KONG IS ALMOST ONE UNIT

Anita, 2026-09-08, on the finished country: *"hong kong being 1 spatial component is pretty
bad, so we should mark that theres definitely room for improvement here if future agents want
to pick it up."* **She is right and this is the thing to fix next.** Hong Kong is registered at
18 districts, but almost nothing varies across them:

| layer | share of the country | varies by district? |
|---|---:|---|
| the survey: Buddhism, Protestantism, Catholicism, Daoism, Hinduism, Sikhism | 31.8% of every district | **no — identical shares everywhere** |
| the ethnic derivation: Islam, part of Catholicism | 4.1% | yes, but only 1.51%–3.00% Muslim across all 18 |
| `unknown` | 65.4% | only as the two above move |

So a reader zooming around Hong Kong sees the same mixture everywhere, and the 18 districts do
almost no work. **§3.9b removed the granularity floor, so a coarse country is drawn rather than
skipped — but "coarse" is a fact to state, not a resting place**, and Hong Kong is by some
distance the least spatially informative country of its size on this map.

**Four routes, roughly in order of what they would buy:**

1. **A survey with district-level religion.** This is the one that matters, because it would
   give the 31.8% a geography instead of a constant. The Hong Kong Panel Study of Social
   Dynamics and the Asian Barometer both carry religion and finer location; both are behind an
   application, which §11b's rule says to attempt only after the open routes are exhausted.
2. **The 2024 Hong Kong Church Survey's district tables** — 1,318 congregations and their
   attendance by district, which would give Protestantism a real geography on a
   `congregations` basis. The report is a paid publication; only the summary figures used above
   are public. Worth an email to the Hong Kong Church Renewal Movement.
3. **The Catholic Diocese of Hong Kong's parish statistics**, same shape for Catholicism.
4. **452 District Council constituency areas**, with boundaries and population on data.gov.hk.
   They are not used because the ethnicity detail the derived layer needs is published only at
   the 18 districts, so adopting them would mix grains inside one layer. They would help only
   in combination with (1).

Until one of those lands, **read Hong Kong as a national pie chart with a population-weighted
scatter**, which is what it is. The `grain` field says so and `note_public` says so.

#### WHAT IS LEFT ELSEWHERE

- **Macau and Taiwan**, still blank and still excluded by `NOT_MAINLAND`. **Taiwan is much the
  larger prize**: a religion-carrying social survey with county geography and a Ministry of the
  Interior register of religious bodies, and a census that does not ask either.
