# 048 — gq: Equatorial Guinea drawn from the DHS report's national table; the microdata needs a DHS registration

Summary: Built at one national mix from DHS 2011's printed table. Its microdata, behind a DHS registration, would place religion across four domains (Malabo, Bata, rest of Bioko, rest of mainland). Register, or leave as is?

*Filed 2026-10-03 by session `fafd1067-gq`. Anita's call; nothing is waiting on it.*

## What I did

Neither of your two downloads has a religion table, so Equatorial Guinea is drawn at 7 provinces
with one national mix: the 2011 DHS report's printed religion table (women and men 15-49), with
its one Christian box split 88:5 Catholic:Protestant at the government's 2015 estimate. Every
province looks the same.

## What it costs to reverse

Leaving it costs nothing. Registering means a DHS Program project application in your name, about
a day's wait, then a rewrite of `sources/gq.py` to read `v130` by domain and a rescatter, about an
hour.

## Why it is yours rather than mine

AGENT_BRIEF.md §3: it needs an account and your identity. The same registration would also open
Nigeria (`ng`), Papua New Guinea (`pg`), DR Congo's 2023-24 survey and Haiti, so it is worth deciding
once rather than per country (`queue.md` §D).

## The detail

- What it would add here is small. The survey's design has four domains (Malabo urban, the rest of
  Bioko, Bata urban, the rest of the mainland; report Appendix A), not seven provinces, and about
  5,000 households. The visible change would be where Muslims sit: 5.4% of men and 1.9% of women
  were Muslim, the State Department says most are West African migrants, and the census found
  foreign residents concentrated in Litoral, Bioko Norte and Wele-Nzas. Today they are drawn at
  3.81% everywhere.
- My recommendation: leave Equatorial Guinea as is; register only if you want DHS for Nigeria or
  Papua New Guinea anyway, and pick this up then.
- Record: `sources/gq.md` §3 and §6.
