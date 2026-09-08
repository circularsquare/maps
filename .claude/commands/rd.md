---
description: Add one country to religiondots, unsupervised
---

You are a religiondots agent. Everything lives in `C:\Users\anita\projects\maps\religiondots` —
work there and use absolute paths; do not assume the shell starts in it. Everything you need is
written down:

1. Read that directory's `CLAUDE.md`, then its `AGENT_BRIEF.md` in full. The brief is the
   standing instruction for an unsupervised session and it overrides your instincts about
   asking permission.
2. Your session id is the last path component of the scratchpad directory in your system
   prompt. Claim with it.
3. `$ARGUMENTS` — if that names a country code, take that one. If it is empty, run
   `python C:/Users/anita/projects/maps/religiondots/tools/claim.py` and pick, preferring a
   parked country over a fresh one.

The three things people get wrong here, so hold them explicitly:

- **Decide the arguable calls yourself and write down why.** Vintages, joins, category
  mappings, whether to abandon the country. `AGENT_BRIEF.md` §2 is the list. Arguable category
  calls go in the mapping's `REVIEW` dict, uncapped.
- **The ask inbox is for §14 and almost nothing else**, it is capped at about one per country,
  and an ask is never a blocker: state the decision you already took and what reversing it
  costs, then carry on.
- **Park at a checkpoint rather than running out of context.** `AGENT_BRIEF.md` §4 says where
  the checkpoints are and why checkpoint B is the cheap one.

Do not commit or push. Run your own builds, backgrounded if slow.
