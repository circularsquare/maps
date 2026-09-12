# 010 — ng: Should the map publish a state-level Christian/Muslim balance for Nigeria

*Filed 2026-09-09 by session `f95259a4-ng`. Anita's call; nothing is waiting on it.*

## What I did

I drew Nigeria: 37 states, 216.8 million people, five categories, every row `modelled` and
desaturated, from six pooled Afrobarometer rounds on the 2022 COD-PS state populations. The
`note_public` opens by saying that Nigeria stopped counting religion on purpose and why, and
it prints this map's national balance beside the two published figures that disagree with it.

## What it costs to reverse

Delete the `"ng"` entry from `countries.py` and re-run `tools/build_tail.py`, about twenty
minutes. Nothing else in the tree depends on it. A softer reversal, drawing Nigeria at the six
geopolitical zones instead of the 37 states, is a one-function change in `sources/ng.py` plus
a zone boundary dissolve, maybe an hour, and it would lose Kaduna and the middle belt, which
is the whole interesting part.

## Why it is yours rather than mine

AGENT_BRIEF §3's first bar: **whether a country may be drawn at all, and at what resolution.**
Nigeria is the case that bar was written for. Its state does not count religion *because* the
count allocates federal revenue and federal-character appointments, so publishing a state-level
Christian and Muslim picture is doing the thing Nigeria has three times declined to do. It is
also, from `queue.md`, "the most consequential unmapped religious boundary on earth", so this
map would be one of very few places a reader sees a state-level version of it.

I decided to draw it, and the reasons are in the detail below. But I would not have decided it
alone if the brief did not say to, and §14's opening line is an invitation.

## The detail

**Nigeria has published nothing since 1963.** The 1973 census asked and was annulled with
nothing released; 1991 and 2006 did not ask; the NPC has said the postponed 2023 census will
not either. Nigeria is absent from the UNSD oracle. `sources/ng.md` §2 lists every route.

**What is drawn, against what else is published:**

| | Christian | Muslim |
|---|---|---|
| **this map, as drawn** | **51.4%** | **47.9%** |
| Afrobarometer R4-R9, its own pooled weighting | 56.0% | 43.3% |
| NDHS 2018 FR359 Table 3.1, women 15-49 | 46.0% | 53.5% |
| Pew Research Center, 2020 | 43.4% | 56.1% |

**The map draws Nigeria Christian-majority and two of the three other figures say the
opposite.** That is the politically live direction, and it is the specific thing I want you to
see. The map's figure is not the survey's own 56.0%: each state is drawn at its own measured
mix and its own COD-PS population, and recomposing that way moves it 4.6 points toward Islam
for two documented reasons (round 6 sampled no Adamawa, Borno or Yobe during the insurgency,
and pooling fourteen years averages over a period in which the northern states grew fastest).
It is not fitted to the NDHS or to Pew, and `sources/ng.md` §3 gives the reasons: §3.1 does not
let an `estimate` set a `self_id` magnitude, and the NDHS is 15-to-49 only, so fitting to it
needs the scale-up §3.4 refused for Brazil.

**What is NOT at issue.** The state geography is the strongest signal on this map: split-half
rank correlations of +0.960 and +0.952 against a +0.327 bar, and it reproduces an outside legal
record it was never shown (all twelve states that adopted the Sharia penal code between 1999
and 2001 come out Muslim-majority). Nobody is being placed by inference. Shia is refused under
§14.4 rule 2 and not drawn at all.

**Why I think drawing it is right.** The tier is the coarsest on this map, 5.9 million people
per unit; the north/south pattern is in every atlas and reveals nothing; §14.3's genuinely
dangerous list is China, Myanmar, Iran, Pakistan and India, and Nigeria is not on it; and §12
says modelled is the floor rather than the plan. Your own `sources.md` §3 row for Nigeria
already says "modelled, and the about panel should say why", which is what the note does.

**Three ways you could rule.** Draw it as it stands. Draw it with a shorter or blunter note.
Or draw it only at the six geopolitical zones, which is how Nigerians themselves group the
states and which would put the national balance on screen while refusing the state-by-state
picture. I did not take the third because it loses Kaduna, and Kaduna at 66.6% Muslim beside
Plateau at 89.9% Christian is the fact worth having.
