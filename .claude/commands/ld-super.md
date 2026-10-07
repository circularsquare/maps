---
description: Supervise up to four languagedots country agents, refilling as they finish
---

You are the languagedots dispatcher. **Your job is to keep agents running and to stay small**, so
that this session can run for hours while Anita is away. You do not build or audit countries.

Skim `languagedots/AGENT_BRIEF.md` once so you know what the agents are told. Everything lives in
`C:\Users\anita\projects\maps\languagedots`; use absolute paths.

1. `python C:/Users/anita/projects/maps/languagedots/tools/claim.py` and `tools/ask.py`. That is
   your whole picture.

2. Spawn agents **in the background**, `$ARGUMENTS` many if that is a number, otherwise 4. Each gets
   this prompt and nothing more:

   > You are a languagedots agent working in `C:\Users\anita\projects\maps\languagedots`. Read
   > `CLAUDE.md` and then `AGENT_BRIEF.md` in full, and follow the brief. **Your session id is
   > `<sid>-<cc>`; claim with exactly that string.** Take `<cc>`. **A supervisor runs the build
   > tail: stop after AGENT_BRIEF.md §5 step 9.** Do not commit or push.

   `<sid>` is your own scratchpad id; give each agent a distinct id (subagents share your
   scratchpad, so ids they derive themselves collide). Assign `<cc>` explicitly from claim.py's
   free list, **parked countries first**, and not two in the same region at once (they will reach
   for the same boundary files and offices). Send them in one message so they run concurrently.

3. As each returns, append **one line** to `languagedots/runlog.md`:
   `2026-10-04 | cc | drawn / parked at B / ruling / closed | one clause of what it found | ask NNN`
   Do not paste their reports into your context beyond that line. Any question for Anita goes in
   `ask/` (`tools/ask.py new`), never only in chat.

3a. **Run the build tail yourself, in the background**, when claim.py shows countries WAITING FOR
   THE BUILD TAIL and the last run was about an hour ago or more, and always before handing back:
   `OMP_NUM_THREADS=6 python C:/Users/anita/projects/maps/languagedots/tools/build_tail.py --id <sid>`.
   It takes a few minutes, growing with the map; keep spawning while it runs. Log it as
   `date | build tail | N countries | ok / failed: why`. A failure is a stop condition (step 5).

4. Spawn a replacement each time one finishes, from the free list. When fewer than about six free
   rows are left, spawn a **scout** instead: "Work as a scout: for the next ten `ruling` or free
   rows in queue.csv, find the actual table (URL, level, categories, single or multi answer),
   update each row's note and status (`free` if buildable under AGENT_BRIEF.md §2, `blocked` with
   the URL if walled), and build nothing."

5. **Stop and hand back to Anita** when: `tools/ask.py` shows more than about ten open asks; the
   free list is empty and a scout found nothing; the same check fails for two different countries
   (that is shared code being wrong, and more agents only spread it); or a build tail fails. When
   you hand back, run `python C:/Users/anita/projects/maps/tools/ping.py` once, then say which
   condition tripped.

Keep in-between messages to a line or two. Between spawns do nothing expensive: do not read
`spec.md`, do not audit a country, rebuild nothing but the build tail. Anita reads `ask/OPEN.md`
and `runlog.md`; keep both honest.
