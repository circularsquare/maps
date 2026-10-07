# 052 — kp: North Korea drawn at one national mix from the World Religion Database; Christians included (spec 14)

Summary: North Korea built flat on Pew 2020, which is the World Religion Database: 72.9% none, 25.2% other (Cheondogyo and shamanism, one residual), 1.5% Buddhist, 0.4% Christian (90 dots). Draw it, drop the Christian dots, or leave North Korea off?

*Filed 2026-10-03 by session `fafd1067-kp`. Anita's call; nothing is waiting on it.*

## What I did

Drew North Korea at one national mix in all 11 provinces: Pew Research Center 2020, which for North
Korea is the World Religion Database's estimate (Pew calls it the only source there is), on the 2008
census's people by province. 72.9% no religion, 25.2% other religions, 1.5% Buddhist, 0.4% Christian
(89,671 people, about 90 dots at 1:1,000). Nobody is placed: every province has the same mix, and the
note says the map does not show where any community lives.

## What it costs to reverse

Dropping the Christian dots into the gap: one line in `sources/kp.py` (`Christians` to `TAIL`), a note
edit, a rescatter and the build tail, about 20 minutes. Taking North Korea off the map: drop `kp` from
`ORDER` in `countries.py` and retile.

## Why it is yours rather than mine

Spec §14: whether a country may be drawn at all, and whether drawing a group affects its safety. North
Korea punishes religious practice, Christianity hardest (executions reported in North Hamgyong 2011,
South Hwanghae 2015, South Pyongan 2018; State Department IRF report 2022). The scout's row
(`queue.md`, 2026-10-03) said the builder files the ask.

## The detail

- **Why I think drawing it is safe.** A national mix places nobody: the 90 Christian dots land where
  the population grid puts people, in proportion, not where any church or believer is, and the note
  says exactly that. The figure is a compiler's national total that has been public for years.
- **Why you might still not want it.** A reader zooming into a town sees a Christian dot there, and
  the map cannot stop that reading as a location. Syria (ask 050) is the same shape, and there the
  groups at risk were also drawn flat.
- **The figure is not a measurement.** No census or survey of residents has ever asked. The World
  Religion Database's 100,000 Christians sits between the state's 12,800 (to the UN, 2002) and Open
  Doors' 400,000. Its 25% "other" is 12.9% Cheondogyo and 12.3% shamanism by ascription; I kept those
  on one residual node rather than drawing about 3 million Cheondoists against the state's own 15,000.
  That is a mapping call, recorded in `taxonomy/kp2020.py`, and not part of this ask unless you want
  the split.
- **Options.** (a) Leave it as built. (b) Draw it without the Christian dots (they go into the "not
  drawn" bar with a sentence). (c) Leave North Korea off. I would take (a).
- Record: `sources/kp.md`; `sources.md` §kp-2026-10-03.
