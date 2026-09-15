# 015 — bo: LAPOP countries draw a pooled 2010-2023 level; Norway's rule would take the recent one

*Filed 2026-09-14 by session `f95259a4-borev`. Anita's call; nothing is waiting on it.*

## What I did

Left Bolivia at the pooled 2010-2023 level, as built, and added one clause to its `note_public`
saying the map is more Catholic than any round since 2016 found (Colombia's note already says the
same of Colombia). Guatemala, El Salvador, Ecuador, Panama, Costa Rica and Colombia are built the
same way and were not touched.

## What it costs to reverse

Per country, a level step in `sources/<cc>.py` (department pattern from the pool, national level
from the recent rounds, spec §12's Norway entry), a re-run and the build tail. About an hour for
Bolivia; seven LAPOP countries in all.

## Why it is yours rather than mine

AGENT_BRIEF.md §3: it changes already-drawn countries' numbers, and it is a rule every LAPOP build
shares. Norway's ESS build took the recent level today; every LAPOP build kept the pool.

## The detail

- **Bolivia, Catholic by round** (post-stratified): 80.3% in 2010, 76.7%, 70.5%, 65.7% in 2016/17,
  65.5% in 2018/19, 64.8% in 2023. The map draws 70.6%. The three rounds since 2016 together give
  65.3% Catholic, 22.2% Protestant plus evangelical (drawn 19.3%), 7.1% believer without a church
  (6.2%) and 1.4% agnostic or atheist (1.0%).
- **The builder's reason for keeping the pool** is that the Protestant and evangelical boxes swap
  between rounds. That affects the split between those two boxes, not the Catholic level, which has
  been flat for three rounds; scaling the two boxes together sidesteps it.
- **It is not a sample-design artefact.** The weighted urban share is 67-71% in every round, and
  rural respondents fell as far as urban ones (84.3% Catholic in 2010, 63.0% in 2023).
- **Size.** 5.3 points against the recent rounds. Spec §12's Norway entry reaches for §3.4 above
  Sweden's 3.5 points. Colombia's note gives 75.8% in 2010 to 67.0% in 2023, and Ecuador's queue row
  a twelve-point fall; Guatemala, El Salvador, Panama and Costa Rica were not measured here.
- **The error has one direction**: each of these countries is drawn more Catholic than its recent
  rounds found.
- **Against flipping**: a level from three rounds of about 1,650 Bolivian respondents is noisier
  than the pool. The department pattern would still come from all six rounds, so only the national
  scaling changes.


---

## Ruled 2026-09-14 by Anita: keep the level as displayed

*"i dont care too much about the national levels tbh. i lean to keep using whatever we actually
display. consistency is nice."*

No rescale. The seven LAPOP countries keep the pooled all-rounds level they are drawn at, and new
LAPOP countries (Venezuela next) follow the same construction for consistency. The reviewer's
figures stay recorded in `sources/bo.md` §12 for whoever revisits this.
