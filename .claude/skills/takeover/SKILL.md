---
name: takeover
description: Start orchestrating a project (trainworld, isochrone, ...) from its live handoff file, while the previous orchestrator may still be finishing its in-flight work
argument-hint: <project>
---

You are taking over as orchestrator of `$ARGUMENTS`. The previous orchestrator may still be
running: Anita starts a new one without waiting for the old one to wind down. You tell the old one,
it lands whatever it has in flight at its own pace and tells you when it is done, and meanwhile you
hold off on anything that would collide with it. These steps are guidelines; if one is clearly
wrong for the situation, do the sensible thing and say why in your first message.

1. **Read** `$ARGUMENTS/CLAUDE.md`, then `$ARGUMENTS/notes/handoff.md`. If there is no
   `notes/handoff.md` yet, use the newest `notes/handoff-*.md` and create `notes/handoff.md` from it
   in step 3, moving the dated one to `notes/archive/`. Then `git status`, `git log --oneline -8`,
   and grep `TASKS.md` for `taking`. Read SPEC.md and other docs on demand, not up front.
2. **Check what the file claims** before trusting it: each long run's liveness check, and that the
   commit it names for 8800 or master matches `git log`. Note anything stale.
3. **Claim the file.** Your session id is in the session-start context ("Session id: ..."); your
   name for messages is what `ListAgents` reports as "This session is <name>". Edit the top of
   `notes/handoff.md`:

       Orchestrator: session <your id>, name <your name>, since <YYYY-MM-DD HH:MM>
       Previous: session <old id>, name <old name>, handed over <YYYY-MM-DD HH:MM>, finishing in-flight work
       Updated: <YYYY-MM-DD HH:MM>

   Once `Previous` names it, the old session cannot start new agents (a hook refuses them unless
   the description starts with `in-flight:`).
4. **Tell the old session.** `SendMessage` to the name on the old Orchestrator line, something like:
   "I've taken over $ARGUMENTS as <your name>. No rush: wind down per /handoff (no new work, land
   what's in flight), then message me when you're done." If the old line has no name (an
   orchestrator from before names were recorded), check `ListAgents` for it; if you can't tell
   which session it is, ask Anita in your first message to tell the old tab.
5. **While `Previous:` is there, hold off on shared work.** The old session is landing its
   work, so leave alone: landing worktrees or merges, commits, rebuilding or restarting what 8800
   serves, publishing, writes to shared build outputs (`data/packs` and the like), and long runs
   that would compete for the cores its runs hold. Fine meanwhile: reading, planning, filing tasks,
   and briefing and launching agents on tasks that are not in flight; queue their landings.
   Lines under "In flight" that name the old session are its to land: don't relaunch those tasks
   or edit their files.
6. **The wait ends** when the old session messages you, or its `Previous:` line is gone. If
   neither happens within a couple of hours, look at it in `ListAgents` and the file: if it is idle
   or gone with lines still in flight, check that work directly (the task's notes, the working
   tree, `git log`), adopt it, delete the `Previous:` line and carry on.
7. **Tell Anita in a few lines** what you picked up, anything stale from step 2, and what you will
   start on. Then work from "Next".

While orchestrating, keep `notes/handoff.md` current as you go (see the `handoff` skill), so that
the next takeover never has to wait for you.
