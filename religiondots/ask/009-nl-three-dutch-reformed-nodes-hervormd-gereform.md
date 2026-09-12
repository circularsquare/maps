# 009 — nl: Three Dutch Reformed nodes: hervormd, gereformeerd, PKN

*Filed 2026-09-09 by session `f95259a4-nl`. Anita's call; nothing is waiting on it.*

## What I did

I added three leaves under `christianity.reformed.continental` and shipped the Netherlands on
them: `.hervormd` (7.30% of the country), `.gereformeerd` (3.58%) and `.pkn` (5.85%), because
CBS publishes `Nederlands hervormd`, `Gereformeerd` and `PKN` as three separate boxes for
every gemeente and their three maps are not the same map.

## What it costs to reverse

Three lines in `taxonomy/nl2014.py` pointing all three at `christianity.reformed.continental`,
three tuples out of `branches.py`, then `build_tree.py`, `scatter.py --country nl` twice and
`build_tail.py`. About twenty minutes, no re-fetch, and nothing else on the map is touched.

## Why it is yours rather than mine

AGENT_BRIEF §3, fourth bullet: **a single-country node that adds a legend row nobody else
uses**. These are three of them at once, in one subtree, and `todo.txt` already carries
"maybe get rid of some of the jewish categories", so legend bloat is live. Tonga's four
Methodist churches were flagged for exactly this shape and that was right.

## The detail

**What the source actually offers.** `Religie en kerkbezoek naar gemeente 2010-2014` (CBS
maatwerk 2015/20) prints nine denominations per gemeente. Four of them are Christian:
Katholiek, Hervormd, Gereformeerd, PKN. National shares over the 394 drawn gemeenten:

| answer | share | strongest three gemeenten |
|---|---:|---|
| Katholiek | 25.86% | Simpelveld 88.2, Gulpen-Wittem 85.8, Nederweert 83.3 |
| Hervormd | 7.30% | Staphorst 47.5, Putten 41.7, Twenterand 39.6 |
| PKN | 5.85% | Dongeradeel 32.4, Ferwerderadiel 32.4, Grootegast 27.6 |
| Gereformeerd | 3.58% | Urk 52.2, Bunschoten 51.5, Reimerswaal 28.6 |

The three Reformed answers together are 16.73%. **No two of them peak in the same place.**
Hervormd is the Veluwe and north-west Overijssel, gereformeerd is Urk, Bunschoten and
Reimerswaal, PKN is the Frisian and Groningen north. Collapsing them onto the parent gives one
Reformed colour, 16.73%, over the country that invented the distinction, and the Bible Belt
stops being legible as anything but "more Protestant than Limburg".

**What the alternatives were.**

1. All three on `christianity.reformed.continental`. Zero new rows. Loses the country's
   structure, as above. South Africa's NGK and Australia's `Reformed` already sit there, so
   the Netherlands would also be sharing a colour with them.
2. `.pkn` only, the other two on the parent. One new row. `.pkn` is easy to defend on its own
   because it is a real church with a real membership roll, but hervormd and gereformeerd then
   merge into each other, which is the pair whose separation actually carries the map.
3. What I did. Three rows.

**What is already in that subtree.** `christianity.reformed.continental` has thirteen leaves
and they all came in with the US Religion Census: `.rca`, `.crcna`, `.urcna`, `.prca`, `.nrc`
(Netherlands Reformed Congregations), `.frcna` (Free Reformed Churches of North America),
`.canrc`, `.heritage`, `.federation`, `.rcus`, `.crec`, `.reformed-bible`, `.rcna`. Most of
those are the American emigrant branches of exactly these three Dutch answers. So the tree
already draws this division at leaf depth; it just draws it in Michigan and not in Zeeland.

**The honest weakness.** Only `.pkn` names a body. `Hervormd` and `Gereformeerd` are what a
Dutch respondent calls their tradition, and after the 2004 merger both words span the PKN and
the churches that stayed out of it (the Hersteld Hervormde Kerk on one side, the Gereformeerde
Gemeenten, the CGK and the vrijgemaakten on the other). The node descriptions say this
outright. If the rule is that a node must be a body rather than a self-description, then only
`.pkn` survives and options 2 or 1 is the answer.

**One thing that would be lost either way, so it is not an argument for three rows.** The
bevindelijk gereformeerde churches proper are inside the `Gereformeerd` cell and are not
separable from it: Urk's 52.2% is what the answer looks like where the Gereformeerde Gemeenten
are the local church, but nothing in the table tells them apart from a vrijgemaakte in Kampen.
