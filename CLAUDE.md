# maps — notes for agents

## Where rules live

A project's rules live once, in its own CLAUDE.md or docs. When a rule changes, change it there and delete the
copies (handoff, memory); a memory file links to the project file rather than restating it. Don't write down what
is better looked up in code or a ledger (core caps, ports, file lists).

## AWS

A friend's account, shared free, for long data builds only. **Read `trainworld/docs/aws.md` before any `aws`
command**; create and destroy only through `trainworld/tools/aws/`. A hook (`trainworld/tools/aws/guard.py`) refuses
raw AWS calls that change anything.

## Serving maps locally

Anita views every map through one server, `python serve.py` at this root (http://localhost:8800/). It is the way to
serve a map, for her and for your own screenshots; don't start another server for a page she will look at.

    python serve.py list
    python serve.py add <name> <folder with index.html> "a few words"
    python serve.py remove <name>

- Register a map when you start real work on one that is not listed; remove one only when she says. Use the
  commands, not a hand edit (they lock the file). No restart needed.
- Pages use relative URLs (`data/x.pmtiles`), since each is served under `/<name>/`.
- Older per-map servers (noritetsu on 8767) still run for existing tooling; add no new ones.

## Sharing CPU cores

`cores.py` at the root is the shared core ledger for every session. The cap is whatever `python cores.py list` says;
it is set with `python cores.py cap <n>`, never written into a doc.

- **A long job** (more than 2 cores) claims first: `python cores.py claim <n> <name> --pid <pid> "what it is"`. The
  last line printed is what you may use; size workers to it. Trainworld's Playwright checks claim for themselves.
- **Release when done**: `python cores.py release <name>`. A claim whose pid died is dropped on the next call.

## Orchestrator sessions

Projects run by an orchestrator (trainworld, isochrone) keep its live state in `<project>/notes/handoff.md`, so a new
orchestrator can start at any moment: `/takeover <project>` starts one, `/handoff` keeps the file current and covers
winding down (`.claude/skills/`). Hooks (`.claude/hooks/orchestrator.py`) give each session its id, note when context
passes 500k, and stop a taken-over session starting new agents (an `in-flight:` description gets through).

## Compact instructions

If a session compacts, keep: task ids, agent names and ids still running, PIDs, log paths and restart commands, what
is uncommitted and whose, and Anita's latest asks in her words. An orchestrator re-reads its `notes/handoff.md`
after compacting.
