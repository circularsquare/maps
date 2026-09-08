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
