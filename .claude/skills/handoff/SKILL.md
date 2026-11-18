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

In a project with a state command (trainworld: `venv\Scripts\python.exe trainworld/tools/handoff.py status`),
edit the file when the plan or her words change, not when an agent launches or lands: the command already shows
that. In a project without one, a small edit to the relevant line whenever something changes that a successor would
need: an agent launched or landed, a long run started or finished, a commit, a publish, a new standing ask from
Anita, a change of plan. One or two lines each time; rewrite a section only when it has drifted. Also run this skill when the stop hook says context is past ~500k (then start the successor: docs/orchestrator.md "Your own cycling"), and when Anita asks. From about 450k
start no new agents: list what you would start under "Next" and the new orchestrator starts them.

## Shape

**With a state command (trainworld):** under about 40 lines, holding only what no command can print: the plan, her
words and judgement. Live state (trees, agents, uncommitted paths, builds, asks, runs, core claims) comes from the
command; don't copy it in. Open asks are never written here: `trainworld/tools/asks.py add|answer|list` is their
only editor, and an ask she answers is `asks.py answer <id> --note "..."` in the same turn.

    # Orchestrator handoff, <project>

    Orchestrator: session <id>, name <name for messages>, since <YYYY-MM-DD HH:MM>
    Updated: <YYYY-MM-DD HH:MM>

    ## Next
    1. In order, with task ids and rough durations; what to start, what to hold and why.

    ## Her words
    Standing asks that still hold, in her words where short; calls she has made this week.

    ## Only a person knows
    Why something is held, which uncommitted file is left on purpose and whose, what a tree is for when its
    name does not say, a run's restart command if the command cannot find it.

**Without one:** keep it under about 60 lines. Pointers, not detail: task ids, notes/ paths, commands. Detail
belongs in `notes/T-###.md`, TASKS.md, DECISIONS.md. Procedures and lessons that outlast this run
go in `<project>/docs/orchestrator.md`; a standing ask that binds subagents goes in the project's
CLAUDE.md, since they never read this file. Finished work leaves "In flight" once it is committed.

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

- **With an agent report inbox (trainworld: `tools/reports.py`):** start no new work and finish only the landing
  you are in. Agents still running are the new orchestrator's: their reports reach it from the inbox, so never pass
  one on. Each of your agents still wakes you when it finishes; end that turn with one word and no tool calls, and
  keep the tab open until they are done, since they run inside it. Then delete the `Previous:` line, update
  `Updated:`, message the new orchestrator once (what you landed, which agents are still running), send Anita the one
  line below, and stop.
- **Without one:** start no new work. Agents' reports that still reach you go to the new orchestrator by
  `SendMessage`, nothing else. Land what is in flight: review, smoke, commit as the project's CLAUDE.md says.
  An agent you genuinely need to finish in-flight work gets a description starting `in-flight:`.
- As each item lands, move its line from "In flight" into "State" (re-read the file first: the new
  orchestrator edits it too). With a state command, nothing to move: landed work shows there; delete its line
  from Next if it was there.
- When nothing of yours is left in flight, delete the `Previous:` line, update `Updated:`,
  `SendMessage` the new orchestrator (the name on its Orchestrator line) that you are done and what
  you landed. Your one message to Anita is a single line: "talk to <new name> now". After it you write nothing
  more to her and never ask her anything (she was getting the same news from two tabs, 2026-11-17).
- If an agent will run for hours more, don't wait for it: make sure its "In flight" line says how to
  check its result without you, delete your `Previous:` line, tell the new orchestrator it is
  theirs to adopt, send Anita the one line, and stop.
