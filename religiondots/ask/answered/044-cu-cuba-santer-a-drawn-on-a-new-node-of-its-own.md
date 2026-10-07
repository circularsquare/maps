# 044 — cu: Cuba: Santería drawn on a new node of its own, afrodiasporic.santeria (16.9%)

Summary: Built Cuba with Santería (NORC 2016, 16.9%) on a new Cuba-only node beside Vodou and Orisha. Keep the node, or fold it into the Afro-diasporic parent?

*Filed 2026-10-03 by session `fafd1067-cu`. Anita's call; nothing is waiting on it.*

## What I did

Cuba is drawn from NORC's 2016 survey (835 adults, national shares in every province). Its second
largest answer, "Santeria or Order of Osha" at 16.9%, is on a new node `afrodiasporic.santeria`,
"Santería (Regla de Ocha)", beside Vodou (Haiti), Orisha (Trinidad) and Revival Zion (Jamaica).

## What it costs to reverse

A one-word edit in `taxonomy/cu2016.py` (to `afrodiasporic`), then `taxonomy/build_tree.py` and the
build tail; no rescatter.

## Why it is yours rather than mine

AGENT_BRIEF §3: a single-country node that adds a legend row nobody else uses.

## The detail

- About 1.65 million dots' worth of people, the size of a whole family elsewhere; every other
  Afro-diasporic tradition a source names already has its own node, each drawn in one country.
- The fallback, the parent `afrodiasporic` ("Afro-diasporic religions"), already carries Brazil's
  and Mexico's unnamed Afro-diasporic answers, so folding saves the one row and loses the name.
- Record: `sources/cu.md`, `taxonomy/cu2016.py` REVIEW, `taxonomy/branches.py` (the node's note).
