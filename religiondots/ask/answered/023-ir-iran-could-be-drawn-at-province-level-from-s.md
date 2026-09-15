# 023 — ir: Iran could be drawn at province level from SCI's 2011 shares; held on section 14

Summary: Iran: SCI published 2011 religion shares for all 31 provinces. May the map draw them, knowing Bahais have no census answer and would sit inside other? In force: held, not built.

*Filed 2026-09-14 by session `afaeb3fc` (supervising the workflow plan), from the scout
`afaeb3fc-scout-ir`. Anita's call; nothing is waiting on it.*

## What I did

Nothing is built. The scout recorded Iran as buildable at province level from SCI's own 2011 table
and held it for §14; the Iran sentence in `queue.md`'s §11ag paragraph says so.

## What it costs to reverse

Nothing to undo. A yes means a build (a `sources/ir.py` from the table, 31 provinces), after a counts
table settles the odd Christian column below.

## Why it is yours rather than mine

`AGENT_BRIEF.md` §3's §14 bar: whether a country may be drawn at all, and how finely, when a group's
safety is affected. Baháʼís are persecuted and have no census answer; converts from Islam are at
legal risk.

## The detail

- **Source.** Elham Fathi (SCI), *Amar* no. 21 (Azar-Dey 1395), pp. 23-26, Table 3: religion shares
  for all 31 provinces from the 1390 (2011) census, to two decimals. The country row agrees with
  UNSD: Muslim 99.38, the two Christian columns 0.16 together, Jewish 0.01, Zoroastrian 0.03, other
  or not stated 0.42. `sources.md` §11n had said SCI published no geography at all; that was wrong.
- **What a province map would show.** Zoroastrians in Yazd (0.32%), Jews in Fars (0.06%), Christians
  in Tehran (0.26%), and Bushehr's 2.95% "other or not stated" (30,473 people) as a standout.
- **Baháʼís.** No census since 1375 (1996) offers the answer. At best they sit in "other or not
  stated", which is merged with non-answers, so the map would not name them; but a province standout
  in that column could still be read as locating them.
- **Two data problems before any build.** The "Assyrian or Chaldean" column is 0.04-0.18 in every
  province, a flat pattern no Assyrian community has, so it looks mislabelled. Shares are rounded to
  0.01%, so Jewish reads 0.00 in 27 provinces. 2016 counts by province exist (amarfact.com's figures
  for two provinces sum exactly to their census totals), but SCI's own release was not found; the
  2017 national yearbook in the Wayback Machine is the lead.
- **Options.** (a) Draw the 31 provinces once a counts table settles the Christian columns.
  (b) Draw them, but put "other or not stated" in `gap` rather than on a coloured node, so no
  province standout suggests where Baháʼís live. (c) Draw Iran nationally only, or not at all.
- **Record.** `sources.md` `## scout-2026-09-14-iran`.


---

## Ruled 2026-09-14 by Anita: draw Iran at province

*"yes, bahais sitting in other is the usual case. it is to be expected. we can try finding a baha'i
specific source later maybe but its not super high prio. i think we have many small religion cases
like this."*

Option (a): draw the 31 provinces, with "other or not stated" on its node like any other country's
other, not moved to `gap`. Baháʼís inside it are the normal case for a small religion with no answer
of its own. A Baháʼí-specific source is worth a look later, low priority. The two data problems above
(the flat Assyrian or Chaldean column, and shares rounded to 0.01%) are the builder's to settle.
Iran is `free` in `queue.csv`.
