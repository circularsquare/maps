---
name: handoff
description: Bring an orchestrator's live handoff file (<project>/notes/handoff.md) up to date, or wind down after another session has taken over
argument-hint: "[project]"
---

`<project>/notes/handoff.md` is the orchestrator's live state: what a fresh orchestrator needs to
carry on if this session ended right now. Keep it current as you go, not only at the end, so Anita
can start a new orchestrator at any moment without waiting for this one. These are guidelines; if a
part of this is clearly unhelpful for your project or situation, adapt it and say so.

## When to touch it

A small edit to the relevant line whenever something changes that a successor would need:
an agent launched or landed, a long run started or finished, a commit, a publish, a new standing
ask from Anita, a change of plan. One or two lines each time; rewrite a section only when it has
drifted. Also run this skill when the stop hook says context is past ~500k, and when Anita asks.

## Shape

Keep it under about 60 lines. Pointers, not detail: task ids, notes/ paths, commands. Detail
belongs in `notes/T-###.md`, TASKS.md, DECISIONS.md.

    # Orchestrator handoff, <project>

    Orchestrator: session <id>, name <name for messages>, since <YYYY-MM-DD HH:MM>
    Updated: <YYYY-MM-DD HH:MM>

    Read <project>/CLAUDE.md and TASKS.md first.

    ## State
    What is true now: master's commit and what landed lately, what 8800 runs, what is published,
    servers deployed. Anything uncommitted and whose it is.

    ## In flight
    - T-### <one line>: agent <name or id>, session <id short>, worktree or files, started HH:MM.
      How to tell it is done if this session is gone (its notes file, its output path).

    ## Long runs
    - <what>: PID, log path, liveness check, restart command, what to do when it ends.

    ## Next
    1. In order, with task ids and rough durations.

    ## Anita's standing asks
    Ones that still hold today, in her words where short.

## After another session takes over

If the new orchestrator messages you, the `Previous:` line names this session, or the agent hook
refused a new agent, you are winding down. There is no rush, and the new orchestrator holds off on
commits, landings, publishing and 8800 until you are done, so land things properly:

- Start no new work. Land what is in flight: review, smoke, commit as the project's CLAUDE.md says.
  An agent you genuinely need to finish in-flight work gets a description starting `in-flight:`.
- As each item lands, move its line from "In flight" into "State" (re-read the file first: the new
  orchestrator edits it too).
- When nothing of yours is left in flight, delete the `Previous:` line, update `Updated:`,
  `SendMessage` the new orchestrator (the name on its Orchestrator line) that you are done and what
  you landed, and end with a short message to Anita saying the same.
- If an agent will run for hours more, don't wait for it: make sure its "In flight" line says how to
  check its result without you, delete your `Previous:` line, tell the new orchestrator it is
  theirs to adopt, and stop.
