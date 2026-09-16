# 033 — ly: Libya drawn from Libyans only, near the Gulf ruling on non-citizens

Summary: Libya is drawn for Libyans only (survey and BSC estimate cover citizens); non-Libyans were 359,540 in 2006, likely more now. The Gulf ruling left such countries off. Keep Libya as built, or take it off?

*Filed 2026-09-15 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Left Libya as its builder drew it: 6,872,674 Libyans at 22 districts, on the Bureau of Statistics
and Census's 2020 estimate, from Arab Barometer waves V to VII, with non-Libyans named in `gap`.

## What it costs to reverse

Taking Libya off: remove `ly` from `ORDER` in `countries.py` and the next build tail drops it.
Adding non-Libyans later needs a source for them, which nothing on disk gives.

## Why it is yours rather than mine

Whether a country is drawn at all is yours (AGENT_BRIEF §3), and this sits between two of your
rulings in `ask/RULINGS.md`. The Gulf ruling says a citizens-only figure is not what the map is for
where non-citizens are a large unmeasured share. The Maghreb ruling reopened Libya but left "Libya
on IOM DTM" undecided.

## The detail

From the Libya build (`sources/ly.md`; `runlog.md` 2026-09-15):

- **Who is drawn.** The Arab Barometer samples citizens (wave VII technical report), and BSC's
  regional population figures are Libyans only. So both halves of the build cover Libyans.
- **Who is not.** The 2006 census counted 359,540 non-Libyans, named in `gap` with no `gap_share`.
  No table gives foreigners by district. Migrant numbers since 2011 are not a state count; IOM's
  displacement tracking is the route your Maghreb ruling left undecided.
- **What the map shows for Libyans:** 0.102% non-Muslim (Christian 0.084%, no religion 0.018%) at
  one share in every district. Most of Libya's Christians are probably among the non-Libyans the
  map leaves out, so the drawn Christian share is likely well under the country's.
- **Precedent on the other side:** Jordan is drawn from a mostly-citizen sample although about three
  in ten of its population denominator are not citizens, with the drawn Christian share called a
  ceiling (`playbooks/arabbarometer.md`).

Options:

1. **Keep Libya as built**, Libyans only, with the non-Libyans named in `gap`.
2. **Take Libya off** until a source for non-Libyans is ruled on, as the Gulf countries were.


---

## Ruled 2026-09-15 by Anita: keep Libya, and estimate who is not drawn

*"libya fine. ideally in general for non-libyans / non-nationals if we cant draw them we estimate the
amount of those people and put them in the "not drawn" section in the bar so we have a rough idea of
how many foreigners there are not being drawn."*

Option 1 for Libya. And a general rule: wherever non-nationals cannot be drawn, estimate how many
there are and put them in the not-drawn part of the bar (`gap` and `gap_share`), so a reader sees
roughly how many foreigners are missing. For Libya that means an estimate of today's non-Libyans
rather than the bare 2006 count with no `gap_share`.
