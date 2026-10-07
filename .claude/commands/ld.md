---
description: Add one country to languagedots, unsupervised
---

You are a languagedots agent. Everything lives in `C:\Users\anita\projects\maps\languagedots` —
work there and use absolute paths; do not assume the shell starts in it.

1. Read that directory's `CLAUDE.md`, then its `AGENT_BRIEF.md` in full. The brief is the standing
   instruction for an unsupervised session and it overrides your instincts about asking permission.
2. Your session id is the id your prompt gives you; if it gives none, the last path component of
   the scratchpad directory in your system prompt. Claim with it.
3. `$ARGUMENTS` — if that names a country code, take that one. If it is empty, run
   `python C:/Users/anita/projects/maps/languagedots/tools/claim.py` and pick, preferring a parked
   country over a fresh one.

The three things people get wrong here, so hold them explicitly:

- **A named census label never sits on a group node** (AGENT_BRIEF.md §3). Only a true remainder does.
- **Tiers C, D and E are not yours to build** (§2): they wait on a ruling. Indigenous-only and
  multi-answer tables ARE buildable, by the rules in §2.
- **Park at a checkpoint rather than running out of context** (§5).

Do not commit or push. Never write into `religiondots/`.
