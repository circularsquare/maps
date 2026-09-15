# 022 — no: ESS countries rescale to their late rounds while LAPOP keeps the pooled level

Summary: Norway and Latvia were rescaled to their late ESS rounds (spec §12); ask 015 kept LAPOP at its pooled level for consistency. Should the next ESS country still rescale? Nothing is blocked; recommended: keep spec §12 for ESS.

*Filed 2026-09-14 by session `afaeb3fc-asks`. Anita's call; nothing is waiting on it.*

## What I did

Nothing new. Spec §12's rule stays in force for ESS: when a survey pool spans a fast change, keep
the pool's regional pattern and scale each category to the late rounds' national level (§3.4).
Norway and Latvia were built that way. Ukraine was not, because its late round cannot reach occupied
oblasts (ask 016, item 3). Ask 015 kept the LAPOP countries at their pooled level and does not
cover ESS (`playbooks/ess.md`, Rulings).

## What it costs to reverse

For a future ESS country, nothing: nothing is blocked on it. Also un-rescaling Norway and Latvia
means `RESCALE_TO_LATE = False` in `sources/no.py`, the matching edit around `sources/lv.py::_rescale`,
new `note_public` wording, a re-run and the build tail for each.

## Why it is yours rather than mine

AGENT_BRIEF.md §3: a rule every ESS build shares, and carrying ask 015's reasoning over to ESS
would change Norway's and Latvia's drawn numbers.

## The detail

- **The rule as written.** Spec §12, "A SURVEY POOL THAT SPANS A FAST CHANGE" (Norway): pattern
  from the pool, level from the recent rounds, weighted by census citizens per unit. It adds
  *"Sweden's drift was under 3.5 points and was left alone; check the size before reaching for
  this."* The Ukraine entry adds that the late rounds must reach every unit. **3.5 points is
  Sweden's size, not a written bar.** The only coded bar is `sources/ua.py::DRIFT_BAR = 0.035`.
  `no.py` and `lv.py` rescale on a flag with no size test, and `playbooks/ess.md` paraphrases it as
  "over 3.5 points".
- **What it moved.** Norway: Church of Norway 44.14% of citizens in rounds 5-9, 32.24% in rounds
  10-11. Latvia: No religion 51.79% in round 4, 62.87% in the late rounds (`sources/lv.md` §6).
  Denmark was not rescaled (flat 2010-2019). For comparison, Bolivia's LAPOP drift in ask 015 was
  5.3 points.
- **Your words on ask 015:** *"i dont care too much about the national levels tbh. i lean to keep
  using whatever we actually display. consistency is nice."* Consistency within one survey says the
  next ESS country rescales like Norway and Latvia. Consistency across the map says it keeps its
  pool like LAPOP, and Norway and Latvia become the two exceptions.

Options:

1. **Keep spec §12 for ESS**: a new ESS country rescales when its drift is large and the late
   rounds reach every unit. No drawn country changes. (Recommended: it is what the map displays today.)
2. **New ESS countries keep the pooled level**; Norway and Latvia stay rescaled as built.
3. **New ESS countries keep the pooled level, and Norway and Latvia are rebuilt without the
   rescale.**

If you pick 1, it may be worth writing a size bar into spec §12, since today the only coded bar is Ukraine's.


---

## Ruled 2026-09-14 by Anita: keep spec §12's rescale for ESS

*"yeah rescaling is good."*

Option 1. A new ESS country rescales to its late rounds when the drift is large and the late rounds
reach every unit; Norway and Latvia stay as built. LAPOP keeps its pooled level under ask 015. She
did not rule on writing a size bar into spec §12.
