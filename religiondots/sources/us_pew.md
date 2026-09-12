# United States — Pew Religious Landscape Study 2023-24, by state

Ingested 2026-09-03. `sources/us_pew.py` rebuilds `data/normalized/us_pew.csv` from
`data/raw/us_pew/`. All of `data/` is gitignored, so this file is the record.

**One sentence: this is the self-identification half of spec §3.5a — the survey supplies the
root totals for the United States, ASARB keeps supplying the structure inside them, and the
difference is §3.2's `…unspecified` residual.**

---

## 1. Re-fetch recipe

```
python sources/us_pew.py
```

51 pages (50 states + DC) at `https://www.pewresearch.org/religious-landscape-study/state/<slug>/`,
cached under `data/raw/us_pew/<slug>.html` at about 1MB each, 50MB in total. The script skips
any page already cached above 100KB, so a re-run re-parses without re-downloading; delete the
cache directory to force a refresh.

Suggested citation: Pew Research Center, *2023-24 Religious Landscape Study*, February 26 2025.
Fielded by NORC, July 17 2023 – March 4 2024, n = 36,908 US adults.

## 2. Why it scrapes, which is not a preference

Pew's **public-use file carries no geography at all**. State identifiers live only in the
restricted-use file, which needs an institutional agreement. openICPSR and ARDA (`RELLAND24`)
both host the public file and neither can answer "how many Catholics in Rhode Island". Pew's own
51 state pages are the only public route to a state-level number, so the pages are the source.

## 3. What the pages carry, which is far more than they show

They are WordPress Interactivity API components. The whole page state is server-rendered into a
single `<script type="application/json">` block, and `state["prc-rls/context-provider"]` in it
holds far more than the dozen rows the page displays:

| field | what it is |
|---|---|
| `religiousTree` | the **state's** chart tree, nested, 149 categories — down to `southern-baptist-convention` and `global-methodist-church`, 51 to 122 of them present per state |
| `religiousTreeUS` | the national tree, for the comparison tab. A separate key, which is the point of §4 |
| `data.value` | the **weighted population estimate** — 1,417,495.68 Catholic adults in Ohio, not the "16%" printed |
| `data.sample_size` | respondents behind the cell. Some are 1 |
| `percent.total` | the state's weighted **adult** population, the denominator for everything |
| `moes` | the state's margin of error and effective sample size, per study year. Ohio ±3.4 points, Rhode Island ±7.5 |

The exact values matter more than convenience. Most states print `<1%` for Jewish, Muslim,
Buddhist and Hindu, and a whole-percent figure cannot carry a residual for a group that size —
"<1%" of California spans a million people.

**The nesting is carried through to the CSV**, in `parent`, `depth` and `group`, because the
tree is the trap: `something-else` is the parent of `unitarians-and-other-liberal-faiths`, and
anything that sums a flat list of category names counts those people twice.

## 4. Two traps, both hit; the first is now structurally impossible

**A state page carries two complete trees**: the state's, and a United States one for the
comparison tab. The first version of this ingest matched category nodes with a regex over the
markup, so it had to separate the two by the denominator each node carried — and a category with
no respondents in a state is simply **absent from the state tree**, so a name-based search falls
through into the national tree. Utah has no `muslim` node, and that version gave Utah 3,026,029
Muslims, every Muslim adult in America, without a murmur.

Read as JSON the two trees are `religiousTree` and `religiousTreeUS` — different keys, no
proximity heuristic, nothing to fall through. The validation that caught the bug is kept anyway,
and is now stronger: every state's figures must **sum to the national tree's** for all 148
categories, not merely land near a published percentage.

**A summarising model reading the rendered page reconstructs rather than reads.** The first
attempt at this collected the printed percentages that way and produced Georgia subgroups
summing to 67 against a reported "approximately 66%". Numbers that go into the map come out of
the source or they do not go in.

`us_pew.py` ends by checking six categories against Pew's own published national percentages,
which is the end-to-end test that would have caught both:

| | scraped, summed over states | Pew publishes |
|---|---|---|
| Muslim | 3,026,029 — 1.18% | 1.2% |
| Hindu | 2,401,369 — 0.93% | 0.9% |
| Jewish | 4,324,617 — 1.68% | 1.7% |
| Buddhist | 2,769,109 — 1.08% | 1.1% |
| Catholic | 48,640,373 — 18.89% | 19% |
| Evangelical Protestant | 59,432,590 — 23.08% | 23% |

## 4a. An absent cell is a zero, not a suppression — CORRECTED 2026-09-03

This file said the opposite until the tree could be read properly, and the correction matters
because the old reading invented ~50,000 unaccounted-for people. Three identities hold, all
three checked on every run:

| | |
|---|---|
| present children vs their parent | equal **exactly**, at every level of every state's tree |
| the 18 top-level nodes vs the published adult denominator | equal **exactly**, all 51 states |
| the 51 states vs `religiousTreeUS`, category by category | equal **exactly**, all 148 categories |

None of those could hold if withheld cells were being carried anywhere. So Muslim appearing in
38 states of 51 means Pew's weighted estimate is **zero** in the other 13 — an n=36,908 survey
cut 51 ways runs out of respondents — and nothing is lost downstream. The 1.18% against Pew's
published 1.2% is rounding of the published figure, not a missing remainder.

That does not make the zeros harmless, it moves the problem: a true zero for Islam in a state
where ASARB counts 25,000 Muslims is a real disagreement, and §3.5a's cap-and-record rule is
what absorbs it. `tools/check_pew_mapping.py` counts them — 62 of 255 (state, root) pairs, 31
from exactly this cause.

## 5. Known gaps, to be handled downstream rather than papered over

- **These are adults.** ASARB counts everyone including children. §3.5a's conversion applies
  adult shares to the whole population, which assumes children hold their household's religion,
  and §3.5a records that the Catholic reconciliation rests entirely on that assumption.
- **Thin cells are still thin.** `sample_size` is carried through precisely so a downstream
  step can refuse a number built on eleven respondents. Nothing refuses one yet.
- **Pew's categories are traditions, not denominations, at the top.** Evangelical / mainline /
  historically Black Protestant do not map onto ASARB's 372 bodies, which is exactly why §3.5a
  takes the residual at L1 only.
- **The eleven religions of §4.4 are not in here.** Sikh, Daoist, Bahá'í and Zoroastrian are one
  `other-world-religions` line, 776,032 adults in 33 states, and no depth of the tree separates
  them. At n=36,908 nothing smaller can be broken out, so this source does not answer §4.4 and
  was never going to. The tree does reach `unitarian`, `pagan-or-wiccan`, `spiritualist` and
  `native-american-religions` individually, which the published summary tables do not — that is
  the one place reading the JSON buys categories rather than precision.

## 6. What comes out, and what reads it

`data/normalized/us_pew.csv`, 4,637 rows, one per (state, category) with a value:

| column | |
|---|---|
| `state` | Pew's slug, `district-of-columbia` and all |
| `adult_total` | the state's weighted adult population — the denominator, repeated on every row |
| `state_moe`, `state_ess` | the state's margin of error and effective sample size for 2024 |
| `group` | `christians` / `others` / `unaffiliated` / `no_answer` — **Pew's** grouping, not ours |
| `parent`, `depth` | the tree. `parent` is `""` at the top level |
| `name`, `label` | Pew's slug and its printed label |
| `adults`, `sample_size` | the weighted estimate and the respondents behind it |

`taxonomy/us_pew2024.py` is the only thing that interprets it. It does not map the 149
categories — §3.5a takes totals at the root and nowhere else — it cuts across the tree at the
28 nodes that reach a religiondots root, and `tools/check_pew_mapping.py` proves the cut is a
partition of all 51 states.
