# 004 — am: Armenia puts 2.79M on christianity.oriental. Does the Armenian church get its own node?

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-am`. Anita's call; nothing is waiting on it.*

## What I did

**Filed Armenia's `Հայ առաքելական` on `christianity.oriental`, the Oriental Orthodox parent,
and shipped the country.** That is 2,793,041 people, 95.2% of Armenia, and it follows what
every other file here already does: `au2021`, `ee2021`, `cy2021`, `ge2014`, `pl2021` and
`ro2021` all put Coptic, Syriac, Ethiopian and Armenian bodies on the same parent node.
`ge2014.py`'s REVIEW already recorded, before Armenia existed here, that a
`christianity.oriental.armenian` child was becoming arguable on Georgia's 109,041, and gave
this same reason for not taking it.

## What it costs to reverse

**One line in each of six mapping files and a re-scatter of six drawn countries.** Nothing
that exists is thrown away and no data is refetched. Armenia is on the map either way; this
decides only whether the largest religious body in the Caucasus is drawn as itself or as a
branch it shares with the Copts.

## Why it is yours rather than mine

AGENT_BRIEF.md §3: **something that changes an already-drawn country's numbers**, and **a
node that adds a legend row**. Both apply. Australia, Estonia, Cyprus, Georgia, Poland and
Romania would each move some of their people onto a node that did not exist when they were
drawn, and `christianity.oriental` would stop being a leaf on the legend.

Your `todo.txt` also carries *"maybe get rid of some of the jewish categories"*, so you are
already watching legend bloat, and this proposes going the other way.

## The detail

### What has changed, and it is only the size

The node's counts before Armenia, from the files that feed it:

| country | Armenian Apostolic |
|---|---:|
| Georgia 2014 | 109,041 |
| Cyprus 2021 | 2,025 |
| Australia, Estonia, Poland, Romania | smaller |
| **Armenia 2022** | **2,793,041** |

So the node is now about 95% Armenian and roughly 96% of it is inside Armenia. The argument
for the parent was always that no source separates the Oriental Orthodox churches finely
enough to justify children. That is still true of every source; what is no longer true is
that the parent reads as a branch. **At 95.2% of one country it reads as "Armenia", and the
Copts and Ethiopians on it become a rounding error inside a colour that is mostly Armenian.**

### The three ways this could go

1. **Leave it.** One colour for Oriental Orthodoxy everywhere, which is what is shipped. The
   cost is the one above: the branch is now effectively a country.
2. **Add `christianity.oriental.armenian` and re-point all six.** Consistent, and it lets
   Georgia's Samtskhe-Javakheti and Armenia read as the same church, which they are. Adds one
   legend row. Every one of the six sources names the Armenian church explicitly, so no source
   is being second-guessed.
3. **Add the whole set** (`.armenian`, `.coptic`, `.syriac`, `.ethiopian`) so the branch has
   real children rather than one named child and a residual. Four legend rows. This is the
   version I would not do without you, because it is where the bloat is.

I would take **(2)** if it were mine, and it is not, because (2) is exactly the shape §3
reserves for you. **(1) is what is on the map right now and is defensible**; nothing here is
broken.

### What is not part of this question

Armenia's other calls are all in `taxonomy/am2022.py`'s REVIEW and none of them touch another
country: Molokans on `christianity.other`, Nestorian on `christianity.churchofeast`, Catholic
at the parent rather than `.eastern`, and the 1,672 Yezidis who answered `Pagan` left where
the census put them. Those are mine and are recorded rather than asked.

---

## Added by the review, 2026-09-08, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-am-rev`

**Not re-arguing the call. One fact above changes the shape of the question and the ask does
not have it: `christianity.oriental` already has ten children, they are already drawn, and one
of them is already the Armenian Apostolic Church.**

`religions.json` carries `christianity.oriental.armenian-etchmiadzin`,
`.armenian-cilicia`, `.coptic`, `.ethiopian`, `.eritrean`, `.syriac`, `.malankara-orthodox`,
`.malankara-syriac`, `.marthoma` and `.knanaya`. Recomputed from `data/processed/counts.json`,
every one of them has dots on it today:

| node | dots | from |
|---|---:|---|
| `.ethiopian` | 35,419 | **et 35,301**, us 105, it/es/gr/fi/fr, bm 2 |
| **`christianity.oriental`** (bare parent) | **3,351** | **am 3,072**, ge 119, ca 70, au 66, bg, cy, cz, pl, ro, es, fi, fr, gr, it |
| `.coptic` | 211 | us 196, it, es, fi, fr, gr |
| **`.armenian-etchmiadzin`** | **156** | **us 104, fr 27, es 15, gr 6, fi 2** |
| `.eritrean` | 55 | us 45, fr, it, gr, fi |
| **`.armenian-cilicia`** | **38** | us 38 |
| `.malankara-orthodox` / `.syriac` / `.malankara-syriac` / `.marthoma` / `.knanaya` | 52 | us |

Three things follow that the ask as filed says the opposite of.

**The legend row already exists.** *"`christianity.oriental` would stop being a leaf on the
legend"* is not what would happen, because it stopped being a leaf before Armenia arrived.
`usrc2020.py` files ten Oriental Orthodox bodies at children and `et2007.py` files Ethiopia's
32,092,182 at `.ethiopian`. Option (2) adds no legend row that is not there; option (3)'s
*"add the whole set (`.armenian`, `.coptic`, `.syriac`, `.ethiopian`)"* is three-quarters
already built.

**The map already files a home country's own national church below the parent, and it is the
biggest body on the branch.** `.ethiopian` holds ten times what the bare parent holds.
`et2007.py`'s REVIEW calls that line *"the most consequential in the file"*. So the precedent
paragraph, *"moving one country below a node the others sit on asserts a distinction those
sources do not make"*, is describing five small European censuses and not the branch. The
branch is already mixed, and Armenia at the parent is what makes it look uniform.

**And the Armenian Apostolic Church is currently drawn two ways on one map.** Armenians in
Armenia, Georgia, Canada, Australia, Bulgaria, Cyprus, Czechia, Poland and Romania are on the
bare parent. Armenians in the United States, France, Spain, Greece and Finland are on
`.armenian-etchmiadzin`, because `origin_religion.py:154` sends `"AM"` there at 0.94 and
`usrc2020.py` maps ASARB codes 049 and 050 there. That is the same church on two nodes, and it
is true of the shipped map right now rather than of any option here.

**The caveat that cuts the other way, and it is why this is still yours.** Those two Armenian
nodes are labelled for American jurisdictions: *"Armenian Church of North America
(Catholicosate of Etchmiadzin)"* and *"Armenian Apostolic Church of America (Catholicosate of
Cilicia)"*, with `sources: {usrc2020: ...}` and nothing else. They are ASARB's US dioceses, not
the worldwide church, so `origin_religion.py` putting Armenians in Lyon on the North American
diocese is already a stretch, and Armenia's 2.79 million would be a bigger one. The clean
version of option (2) is therefore probably **a general `christianity.oriental.armenian` with
the two US jurisdictions moved under it or merged into it**, which is a bigger edit than the
ask prices at six mapping files, and is a real reason to prefer (1) if you want this small.

Nothing changed. This is one table and a correction to the precedent paragraph, so that
whichever way you go, it is not on the belief that the branch has no children.

---

## Ruled 2026-09-08 by Anita

**Record whatever the source actually says.** Her view is that this class of inconsistency is par
for the course and not a defect: some censuses name the Armenian church, some can only say
Oriental Orthodox because their own categories cannot distinguish, and some say Coptic. Each
source should be recorded as it answers.

So, concretely: a source that names the Armenian Apostolic Church files at an Armenian node; a
source whose category genuinely cannot distinguish stays at the bare `christianity.oriental`
parent, and that is a correct outcome rather than a gap to harmonise away. This is option 2 in
substance, but arrived at from the opposite direction — not to make the map consistent, but to
stop the map asserting a distinction the Armenian census does make.

The review section above stands as the implementation note: the clean version needs a general
`christianity.oriental.armenian` node, with the two ASARB US jurisdictions merged under it or
moved below it, because those two are labelled for American dioceses rather than the worldwide
church.

---

## Implemented 2026-09-08, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-armnode`

**Done. Three nodes, thirteen mapping lines, fifteen countries re-scattered, one build tail.**
The durable record is spec **§2.7** and `sources.md` **§9ca**; this is only what changed.

**The nodes.** `christianity.oriental.armenian` (*Armenian Apostolic Church*), and under it
`.armenian.etchmiadzin` (*Catholicosate of Etchmiadzin*) and `.armenian.cilicia`
(*Catholicosate of Cilicia*). The last two are the old `armenian-etchmiadzin` and
`armenian-cilicia`, **moved and relabelled, not merged.** Merging was ruled out on the ruling's
own test: ASARB counts the two catholicosates separately at codes 049 and 050, so folding them
would lose a division a source actually makes. `build_tree.py` would also have refused it, with
*leaf claimed by several codes*. Relabelling came with the move because those were the tree's
only Armenian nodes and their ASARB names say *of North America* and *of America*, which is
exactly what made `origin_religion.py:154` a stretch for a French Armenian.

**The list in the ask was six and the real list was nine, plus two more places.** Verified from
the files rather than taken: `am2022`, `ge2014`, `cy2021`, `au2021`, `ee2021`, `pl2021`,
`ro2021` were all correct, and two the ask did not have were **`bg2021`** (`Арменско
апостолическо`, 5,002, derived through `bg_split.py`) and **`cz2021`** (`Církev Svatého Řehoře
Osvětitele`, 3 people, the Armenian church registered under its patron saint after the Ministry
of Culture refused the name *Arménská apoštolská pravoslavná církev* in 2006). Plus
`usrc2020.py` and `origin_religion.py`. **Poland turned out to have two Armenian cells, not
one**, and the second names a catholicosate, so it is the only source outside ASARB that
reaches `.armenian.etchmiadzin`.

**Left at the parent, on purpose.** `ca2021`'s `Oriental Orthodox`; `au2021`'s `COLUMNS` entry
for ABS group 221 and its `Oriental Orthodox, nec/nfd`; `pl2021`'s `różne inne chrześcijańskie
kościoły wschodnie`; `es_origin`'s `Resto de África` residual. None can distinguish.
`origin_religion`'s other `ORIENT` rows (GE, TR, AZ, LY, SD, DJ, IR, IQ, SY, LB and the Gulf)
also stayed: several are mostly Armenian or mostly Coptic, but each is a mixed residual in a
model rather than a category anyone published. `at2001` and `uk2021` were checked and are not
this question at all: both have Armenians folded inside an undivided *Orthodox* cell and both
already say so.

**Also left, and worth naming so it is a decision rather than an oversight:** `au2021`'s
`Coptic Orthodox Church`, `Syrian Orthodox Church` and `Ethiopian Orthodox Church`, `nz2023`'s
`Coptic Orthodox` and `pl2021`'s `Kościół koptyjski` all name a church and all sit at the
parent while `.coptic`, `.syriac` and `.ethiopian` exist. That is the same defect this fixed
for the Armenians and it is out of this brief's scope; it is four lines and a re-scatter of
three countries for whoever wants it.

**Does a reader see a different legend? At the default view, no.** `depth` starts at 2 and
`drawnSet` stops there, so `christianity.oriental` is the drawn category, `paletteFor` paints
its whole subtree in its colour, and `counts[]` is a subtree total, so the row's label, colour
and number are all unchanged. `isOpen` also keeps a drawn node with no drawn children closed,
so the new rows sit behind the same twisty the other ten children were already behind. The
change is visible at depth 3 or with Oriental Orthodox selected: one new row, *Armenian
Apostolic Church*, and the two American diocese labels replaced by the catholicosates. The
overview band for `christianity.oriental` is re-sliced among nine children instead of ten, so
those deeper colours shift; nothing at depth 1 or 2 moves, and no root or `PIN`/`OVERVIEW_FLAT`
entry was touched.

**The trap, for whoever does the Coptic version.** Five countries (`es`, `fi`, `fr`, `gr`,
`it`) carry the node id **baked into `data/normalized/<cc>_foreign.csv`** — `origin_religion.py`
resolves the node at fetch time, not at scatter time. Editing that file alone leaves the CSVs
pointing at a node that no longer exists, and the only symptom is `coverage.py` naming the dead
id. Re-run `python sources/<cc>.py` for each (no `--fetch`, it works off cached raw files)
before re-scattering.

---

## Finished 2026-09-08, session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-orientalrest`

**The five cells the section above named as out of scope are done, and nothing else on the
branch was left at the parent by mistake.** Five mapping lines, three countries re-scattered,
three nodes promoted from ASARB leaf to branch. The record is `sources.md` §9ca's last
subsection and spec §2.7; this is the two-line version.

`au2021`'s `Coptic Orthodox Church`, `Syrian Orthodox Church` and `Ethiopian Orthodox Church`,
`nz2023`'s `Coptic Orthodox` and `pl2021`'s `Kościół koptyjski` now file at `.coptic`,
`.syriac` and `.ethiopian`. **Those three nodes had to become branches in `branches.py`
first** — they were drawn already but only ASARB could reach them, because `build_tree.py`
refuses any other mapping module that targets a non-branch id. Their labels are unchanged, so
no legend row is renamed: unlike the two Armenian nodes, none of the three was named for an
American diocese.

**A sweep of every mapping module and `origin_religion.py` found nothing else.** Staying at the
parent, on top of the four already recorded: `au2021`'s `Oriental Orthodox, nec` and `nfd`,
and all of `origin_religion.py`'s `ORIENT` rows. `origin_religion.py` needed no edit at all,
so the five `_foreign.csv` files with baked-in node ids were never at risk; their ids were
checked and all are live.

**What a reader sees is still nothing at the default depth.** At depth 3 with Australia in
view there is one difference the earlier pass could not have seen: the bare `Oriental
Orthodox` row disappears from Australia's legend, because Australia now has no dots on it.
