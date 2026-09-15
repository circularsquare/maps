# 013 — gr: No ESS category in Greece clears the split-half, and three note_public claims rest on single rounds

*Filed 2026-09-14 by session `f95259a4-essrep`. Anita's call; nothing is waiting on it.*

## What I did

Ran the split-half on Greece as a report, per your ruling on ask 012, and changed nothing.
Greece still draws every ESS category at its own regional share and `note_public` still names
regions, although none of Greece's eight ESS categories clears the test and three of the note's
regional claims do not recur from one survey round to the next.

## What it costs to reverse

Two separate things, either without the other. Rewording `note_public`'s regional sentences is a
`countries.py` edit and `tiles.py --refresh-meta`, minutes. Applying the test means porting
`_stability` into `sources/gr.py`, then a rebuild and a retile, about an hour, and it makes every
region's citizen composition the national one, so the regional sentences go either way.

## Why it is yours rather than mine

AGENT_BRIEF §3, "something that changes an already-drawn country's numbers", and your ruling on
012 that anything alarming comes back as a separate decision. Greece meets both halves of what
the ruling called alarming: a large category fails (Orthodox, 91.5% of citizens), and everything
ESS contributes to Greece's regional picture rests on categories that fail.

## The detail

`python tools/ess_split_half.py gr`; the full write-up is `sources/gr.md` §9. 7,935 Greek
citizens, ESS rounds 5, 10 and 11, 13 regions, three splits of one round against two, be.py's
statistic, draw count and seed, and Sweden's chi-square beside it.

| category | n | median rho | null 95th | p | chi² p |
|---|---:|---:|---:|---:|---:|
| Eastern Orthodox | 7,239 | -0.038 | +0.374 | 0.58 | 1.9e-29 |
| No religion | 573 | -0.066 | +0.396 | 0.62 | 2.1e-26 |
| Other Christian denomination | 54 | +0.197 | +0.435 | 0.24 | 2.2e-07 |
| Roman Catholic | 33 | +0.538 | +0.568 | 0.0625 | 9.5e-113 |
| Other Non-Christian religions | 25 | +0.303 | +0.444 | 0.13 | 1.1e-20 |
| Islam | 7 | no test: nobody in rounds 10 or 11 | | | |
| Eastern religions | 3 | -0.123 | +0.736 | 1.00 | 0.20 |
| Protestant | 1 | no test | | | |

The regions do differ inside each round (every chi-square but one is tiny), and they do not differ
the same way from round to round. The recode from round 5's NUTS 2006 codes to the later NUTS
2016 ones was checked label by label and is right, so this is the sample and not a join.

The three regional claims in `note_public`, against the rounds (unweighted %, rounds 5 / 10 / 11):

- *"concentrated in Attiki, Peloponnisos and Thessalia"*: Attiki 11.9 / 9.2 / 10.7, which recurs.
  Peloponnisos 19.5 / 11.0 / 7.2, a falling trend. Thessalia 1.1 / 32.1 / 0.5, so **61 of its 64
  no-religion respondents are from round 10.**
- *"Dytiki Makedonia is the most Orthodox"*: 6th, 4th and 1st of 13 in the three rounds.
- *"Notio Aigaio comes out 10% Catholic"*: 24.6 / 6.5 / 0.0 of 69 / 93 / 55 respondents; 17 of its
  23 Catholic respondents are from 2010.

**The Cyclades are why this is a trade and not a cleanup.** Syros, Tinos and Naxos have real
Latin-rite communities, so applying the test would flatten a true geography along with the noise
(Catholic p = 0.0625, just over the bar, 8 of 13 regions with no Catholic respondent). That is the
test doing what §9cy says a failure means, a failure to demonstrate and not a demonstration of
noise, but it is a visible loss.

The options:

1. Leave Greece as it is. The note keeps a Thessalia claim that comes from one round.
2. Keep the drawing and reword `note_public` to name only what recurs: Attiki's no-religion share,
   and the Cyclades with a line that it rests mostly on 2010. Minutes, no rebuild. **This is the one
   I would take**: it fixes the sentence that is wrong without making Greece stricter than three
   rounds can support. It leaves Greece on a different standard from Belgium and Sweden, which is
   the state your 012 ruling already accepted.
3. Apply the test as Belgium does. Every region gets the national citizen composition, and the
   Thracian minority and the foreign half become Greece's only regional variation. Consistent with
   Belgium and Sweden, and it loses the Cyclades.

**The other four are not in this ask because none met the bar.** Their tables are in their own
`sources/<cc>.md`. Finland keeps 5 of 15 categories, 96.96% of citizens, and on this statistic
Lutheran passes, reversing the one-halving failure `fi.md` §8 recorded. France keeps 6 of 9,
98.60%. Germany keeps 6 of the 8 categories it splits out of the register's residual, Islam at
+0.857. Italy's seven NUTS 1 minority categories all fail, 2.55% of citizens, but five
ripartizioni leave the test little room (Catholic and unaffiliated only just pass there), and most
of Italy's minorities are in the census-based foreign half.


---

## Ruled 2026-09-14 by Anita: deferred

**Deferred.** Nothing changes in Greece for now; the dots and `note_public` stay as built. It goes
on the list to address later, recorded in `queue.md` at the head of the survey refinement list.
The three options above stand for whoever picks it up. Not for an unprompted session.
