# 005 — rw: ADEPR is 21.3% of Rwanda and is one named church; it is drawn as generic Pentecostal

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-rw`. Anita's call; nothing is waiting on it.*

## What I did

Mapped Rwanda's `ADEPR` cell to `christianity.pentecostal`, the existing parent, with no node
of its own, and wrote the reasoning into `taxonomy/rw2022.py`'s REVIEW entry. Rwanda is built
and drawn on that basis.

## What it costs to reverse

A node in `taxonomy/branches.py`, a one-line change in `taxonomy/rw2022.py`, then
`build_tree.py`, `scatter.py --country rw` twice and `build_tail.py`. About 25 minutes, no
re-fetch, no re-parse.

## Why it is yours rather than mine

AGENT_BRIEF §3: *a single-country node that adds a legend row nobody else uses*. Your
`todo.txt` already carries "maybe get rid of some of the jewish categories", so legend bloat
is something you are watching, and this would be one more row that only Rwanda ever fills.

The reason I am asking rather than only recording it is the size. Tonga's four Methodist
churches were flagged under this bar and are tens of thousands of people; this is **2,820,813**,
a fifth of a country, and the second-largest religious answer in it.

## The detail

**What the cell is.** `ADEPR` is the Association des Églises de Pentecôte du Rwanda, one
denomination, named as such by NISR in all thirty RPHC-5 district profiles. It grew out of the
Swedish Free Mission that reached Rwanda from the Belgian Congo in 1940 and took its present
name in 1983. 2,820,813 people, **21.29%** of Rwanda, second only to Catholicism at 39.91%.

**Why a node is arguable.** It has a geography of its own that a generic label does not
explain: it peaks at **33.90% in Rusizi**, which is where its first congregation was, and at
30.70% in neighbouring Nyamasheke, thinning eastward to **11.12%** in Nyamagabe. That is a
mission map, and it is the kind of thing this project draws elsewhere by giving the body a
node (Zimbabwe's Vapostori went to `christianity.africaninstituted`, which already existed;
Kiribati's and Tonga's churches got new leaves).

**Why I did not mint one.**

* The precedent runs the other way. `zw2022.py` maps `Pentecost` to `christianity.pentecostal`
  with "no branch given and none inferred", and that cell is 2,582,565 people. Every country
  that names a Pentecostal body the tree has no leaf for does the same.
* The obvious parent for a leaf would be `christianity.pentecostal.trinitarian`, whose fourteen
  existing children are **all United States denominations** from the ASARB source. Hanging a
  Rwandan church there would be the first non-US member of a branch that is otherwise a US
  artefact, and would make that branch mean something different.
* Nothing is lost on screen at the default depth. `christianity.pentecostal` carries the
  colour and the legend row either way, and the church's name is in `source_category`,
  `sources/rw.md`, the country note and the mapping.

**What tips it the other way, if you want to flip it.** The UNSD Demographic Yearbook prints
this exact figure as generic `Pentecostal`, and the whole reason Rwanda is worth drawing from
the district profiles rather than from the oracle is that NISR names the church. Mapping it to
a generic node reproduces the Yearbook's loss inside the tree, in the one country where the
office was more specific than the UN.

**If you do want it**, the shape I would build is `christianity.pentecostal.adepr` as a direct
child of `christianity.pentecostal` rather than under `.trinitarian`, which keeps the US
branch what it is. Your ruling on `ask/004-am` (sources.md §9ca) already established that a
branch is allowed to have children at different depths.

---

## Ruled 2026-09-09 by Anita

**Leave it on the generic `christianity.pentecostal` node. No new node.**

Her reasoning, in her words: if it is mostly just Pentecostal then it should stay Pentecostal —
ADEPR reads as the Pentecostal church *of Rwanda* rather than a distinct body, and the fact that
the UNSD Yearbook prints the identical figure as generic `Pentecostal` is further support for
filing it that way.

Note for whoever meets this again: the yearbook agreeing is evidence about how the body is
classified, but it is not independent evidence about what the body *is* — the yearbook is a
return from the same office. Both point the same way here, so nothing turns on it. The mapping
is unchanged and Rwanda is not re-scattered.
