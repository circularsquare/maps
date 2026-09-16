# 041 — sd: Sudan: drawn during the war at one national share (section 14)

Summary: Sudan drawn from two surveys at one national non-Muslim share, pre-war positions, refugees only in the gap. Section 14: the war, Darfur, Nuba Christians. Keep as built, take it off, or allow refugees by state later?

*Filed 2026-09-15 by session `cb8b206e-sd`. Anita's call; nothing is waiting on it.*

## What I did

Drew Sudan at its 18 states from the pooled Afrobarometer (2013-2022) and Arab Barometer (2018-2022),
at one national non-Muslim share, 0.69%, in every state, on the pre-war 2022 population projection.
Nothing is placed below the nation, and foreigners and refugees are in the not drawn part of the bar
rather than on the map.

## What it costs to reverse

Taking Sudan off: remove `sd` from `ORDER` in `countries.py`; the next build tail drops it. Adding
refugees by state later: a new module on UNHCR's pre-war dashboard, about one session.

## Why it is yours rather than mine

AGENT_BRIEF §3, first bar (§14): whether a country may be drawn at all while a war with ethnic and
religious lines goes on, and whether placing a group (refugee camps, or Kordofan's Christians) could
affect anyone's safety.

## The detail

- **The situation.** War since April 2023; the UN Population Fund's 2025 note counts more than 12
  million people who have fled their homes. Killing along ethnic lines in Darfur; fighting in South
  Kordofan and Blue Nile, where the Nuba Mountains hold many of Sudan's Christians. Apostasy carried
  the death penalty until 2020.
- **What the map shows.** Dots at pre-war positions; every state 0.504% Christian and 0.182% no
  religion. It points at no community. The 2021-22 interviews put Christians in Kordofan above the
  rest (13 against 5.85 expected, P 0.0037), the 2013-19 ones do not, so it is not drawn; a later
  round that repeats it would pass the test, and drawing Kordofan's Christians at region grain would
  then be the question.
- **Refugees.** UNHCR counted about 1.14 million refugees and asylum seekers before the war, 796,831
  of them South Sudanese (Pew: South Sudan 61% Christian), in camps in White Nile, East Darfur,
  Kassala and Gedaref that UNHCR publishes. Drawn by state they would be the largest non-Muslim
  population on Sudan's map. Not built.
- **Precedent.** Chad (017), Burkina Faso and Mali (018) drawn at their published tiers despite
  attacks; Zanzibar at one Christian share for safety; Ukraine's occupied oblasts from pre-war
  rounds (016).
- `sources/sd.md` §2, §4 and §7.

Options:

1. **Keep Sudan as built.**
2. **Take Sudan off** until the war ends.
3. **Keep it, and allow a refugee layer by state** later (and Kordofan's Christians at region grain,
   if a later round supports it).
