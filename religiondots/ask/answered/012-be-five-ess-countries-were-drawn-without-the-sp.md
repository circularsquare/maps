# 012 — be: Five ESS countries were drawn without the split-half Belgium now runs

*Filed 2026-09-11 by session `f95259a4-be`. Anita's call; nothing is waiting on it.*

## What I did

Belgium runs §14.16's split-half over its seven ESS rounds and draws only the four
categories that clear it where they were measured; the other five go at the national rate
inside each province's residual. Greece, Finland, France, Germany and Italy were all drawn
from the same instrument without that test, so every category in those five is drawn where
it was measured, including some that are as thin as the ones Belgium just declined to place.
I did not touch them.

## What it costs to reverse

Belgium: one line, `carries = cats`, and a re-run plus retile, about half an hour.
The other five: each needs the test ported into its own module (the code is 60 lines in
`sources/be.py`, `_rank` / `_rho` / `_median_rho` / `_stability`) and then a rebuild of that
country, so roughly an hour each, plus new `note_public` wording anywhere a category moves.
Nothing is blocked either way.

## Why it is yours rather than mine

AGENT_BRIEF §3, "something that changes an already-drawn country's numbers". Applying it to
Belgium alone was mine and is done. Applying it backwards to five drawn countries would move
their dots, and declining to apply it leaves the map using two different standards for the
same survey, which is a rule everyone shares rather than a per-country call.

## The detail

Belgium's table, 10,877 pooled citizen respondents over 11 provinces, 35 three-against-four
round splits against a 2,000-draw permutation null:

| category | respondents | p | verdict |
|---|---:|---:|---|
| No religion | 6,475 | 0.0620 | not distinguishable |
| Roman Catholic | 3,572 | 0.0150 | own geography |
| Islam | 503 | 0.0005 | own geography |
| Protestant | 82 | 0.1554 | not distinguishable |
| Other Christian denomination | 75 | 0.7241 | not distinguishable |
| Eastern Orthodox | 45 | 0.0005 | own geography |
| Other Non-Christian religions | 43 | 0.6702 | not distinguishable |
| Eastern religions | 34 | 0.0270 | own geography |
| Jewish | 15 | 0.1404 | not distinguishable |

What it changed for Belgium: the Jewish population is now drawn flat at 0.16% of every
province instead of wherever 15 respondents happened to land, and `note_public` says the
community is really in Antwerp and Brussels and that this map cannot show it. Protestant,
other Christian and other non-Christian moved the same way. Nothing was deleted.

**The five others are exposed to the same thing and two of them more than Belgium.**
Finland's card is fourteen codes over 12,741 respondents and 19 maakunnat, so its per-cell
counts are thinner than Belgium's on a finer geography; `sources/fi.md` already records that
`Mormon` and `Jewish` drop out of the later rounds' category lists entirely. Greece pools
7,885 respondents over 13 regions with a card that shrinks from eight denominations to four.
Italy is NUTS 1 in its later rounds. France and Germany are the largest samples and the
least exposed.

**The case for leaving them alone**, which is why I am not calling it: the test is not free
of judgement either. It needs a multiplicity decision (Belgium's is plain alpha = 0.05 over
nine categories, and a Holm correction would have flattened Catholicism), and with seven
rounds as the resampling unit it has little power, so it will fail real categories as well
as noise. Rerunning five drawn countries to make them stricter would move dots on the
strength of a test that was itself a judgement call.

**The cheap middle option**, if you want one: run the test on the five as a report only, print
the tables into their `sources/<cc>.md` files, and change nothing until a table says
something alarming. That costs an afternoon and no dots move.


---

## Sweden is a sixth ESS country and ran it too — added 2026-09-11 by session `f95259a4-se`

Not a second ask. Sweden (§9cz) was built the same day and uses `sources/be.py`'s
`_stability` unchanged, so the count in the title is now **five drawn without the test,
two with it**. Three things from Sweden that bear on the ruling:

- **The test is not a formality on this instrument.** Sweden draws 3 of 10 categories on
  their own län shares and 7 at the national rate. `No religion`, 68% of the country, is one
  of the seven at p = 0.0925 — the same result Belgium got, for the same reason, and it
  matters as little for the same reason (it is 95% of the residual, so it still varies).
- **It is not only a caution: it changed the LEVEL Sweden is drawn at.** The first build used
  one chronological halving against `spearman_null`'s bar and concluded that pooling ESS
  rounds 9 and 11 destroyed the ordering at NUTS 2. On the median-over-splits statistic it
  does not. So the two tests do not merely differ in strictness, they can send a country to a
  different geography. That is worth knowing before deciding whether to re-run the five.
- **Sweden adds one requirement that the five would also need**: a spatial chi-square at 0.05
  beside the permutation p. Two Swedish categories on 21 and 23 respondents CLEARED the
  permutation test with chi-squares of 0.32 and 0.40, because a Spearman over a column that is
  zero in most units is decided by how the ties break and permuting that same column
  reproduces the tie structure. Belgium's `Eastern religions` at n=34 and p=0.027 is flagged
  in §9cy as the one provisional verdict there; it is the same shape and its chi-square is
  worth printing before any of this is generalised.

Nothing here changes what is being asked or what it costs to reverse.

---

## Ruled 2026-09-14 by Anita

**The report-only option.** Run the split-half test, with Sweden's spatial chi-square beside it,
on Greece, Finland, France, Germany and Italy. Print each result table into that country's
`sources/<cc>.md`. Move no dots and rebuild nothing. Anything alarming comes back as a separate
decision.
