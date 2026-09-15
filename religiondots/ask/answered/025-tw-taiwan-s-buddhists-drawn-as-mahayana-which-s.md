# 025 — tw: Taiwan's Buddhists drawn as Mahayana, which spec 2.6 left open

Summary: Taiwan files TSCS Buddhists on buddhism.mahayana, the map's largest Mahayana group; only Hong Kong does the same. Five of seven rounds name sects; 1994 and 2015 do not. Keep, or plain buddhism like Korea? Recommended: plain.

*Filed 2026-09-15 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Left Taiwan as the builder drew it: TSCS Buddhists on `buddhism.mahayana` (`taxonomy/tw2018.py`).
The counts are the same whichever node they sit on; only the legend row changes.

## What it costs to reverse

A node edit in `taxonomy/tw2018.py`, Taiwan's scatter and the build tail, about 40 minutes.

## Why it is yours rather than mine

Spec §2.6 is your ruling of 2026-09-07: leave Buddhists as the sources file them, and assign no
country's Buddhists a school. What it did not decide is a source that names schools itself (Japan's
survey asked, and ask 014 gave Japan school nodes). Taiwan's survey sits between the two.

## The detail

From the review of `tw` (`sources/tw.md` §8):

- **What the source says.** Five of the seven TSCS rounds used (1994-2018) record a Buddhist sect;
  the 1994 and 2015 answer cards offer only "Buddhism".
- **What the map does elsewhere.** Singapore, Malaysia, Vietnam and Korea keep plain `buddhism`.
  Hong Kong is the only other country on `buddhism.mahayana`.
- **Size.** Taiwan's 3.3 million would be the largest Mahayana group on the map, from a survey.

Options:

1. **Plain `buddhism`**, like Korea, Singapore, Malaysia and Vietnam, because two of the rounds name no
   school. The reviewer leans this way, and so do I.
2. **Keep `buddhism.mahayana`**, on the grounds that most rounds name a sect and Taiwan's Buddhism is
   Mahayana in practice.


---

## Ruled 2026-09-15 by Anita: keep Mahayana

*"i mean if the survey says people said mahayana, its fine to say mahayana, right? i lean to say
mahayana unless its like significantly off from what we think the national estimate should be. we
just leave it generic for other countries cuz their sources dont specify, i think"*

Option 2. Taiwan stays on `buddhism.mahayana`. A source that records the school may be drawn at the
school unless it is well off the national estimate; countries whose sources name no school stay on
plain `buddhism`.
