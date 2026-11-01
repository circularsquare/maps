# maps — notes for agents

## AWS

AWS here is a friend's account, shared for free, used only for long data builds. **Read
`trainworld/docs/aws.md` before any `aws` command**; create and destroy things only through
`trainworld/tools/aws/`. A PreToolUse hook (`trainworld/tools/aws/guard.py`, in `.claude/settings.json`)
refuses raw AWS calls that create, change or destroy anything.

## Serving maps locally

Anita views the maps through one server, `python serve.py` at the root of this tree
(http://localhost:8800/, a front page with a link to each map). Which maps it serves is the
register `served.json`, and keeping it current is the agents' job:

    python serve.py list
    python serve.py add <name> <folder with index.html> "a few words"
    python serve.py remove <name>

- **This is THE way to serve a map, for her and for you.** Do not start a server of your own
  (`npx serve`, `python -m http.server`, a new per-map serve.py) for a page she will look at;
  register it here and give her the `localhost:8800/<name>/` link. Point your own headless
  screenshots at 8800 too. It speaks HTTP Range, so .pmtiles work, and it opens files per
  request rather than holding them, so a rebuild can replace an archive while it is being
  served (`npx serve` holds files open on Windows and the rebuild's rename fails).
- **Register a map when you start real work on one that is not listed**, so she can open it
  at `localhost:8800/<name>/`. Remove one only when she says it is no longer being worked on.
- Use these commands, not a hand edit; they lock the file, since sessions run at once.
- The running server re-reads the register on every request: no restart after add or remove.
- A map's own older server (`noritetsu/serve.py` on 8767 and the like) still works, and some
  existing screenshot tooling points at it; leave those alone, but do not add new ones.
- A map's page must use relative URLs (`data/x.pmtiles`, never `/data/x.pmtiles`), because
  it is served under `/<name>/`.

## Sharing CPU cores

Sessions from every map project run long jobs at once on a 16-core machine; `cores.py` at the root
is the shared ledger (`cores.json`, cap 10 in all while Anita is using the machine, raised by
`python cores.py cap <n>` when she is away).

- **Any long job** (a build, a batch or a sweep using more than 2 cores) claims first:
  `python cores.py claim <n> <name> --pid <pid> "what it is"`. The last line printed is the number
  of cores you may use; size `--jobs` or workers to it. It never refuses, but under a full cap it
  grants 1.
- **Release when done**: `python cores.py release <name>`. A claim whose `--pid` has died is
  dropped on the next call, so a crashed job cleans up after itself.
- `python cores.py list` shows each claim, the total, and how many cores are free under the cap.

## Orchestrator sessions

Projects run by an orchestrator session (trainworld, isochrone) keep its live state in
`<project>/notes/handoff.md`, current as the work goes, so Anita can start a new orchestrator at any
moment without waiting for the old one to wind down. Like the rest of this section, these are
guidelines: if one is clearly unhelpful in your situation, adapt it and say why.

- `/takeover <project>` starts a new orchestrator from the file; `/handoff` brings it up to date,
  and covers winding down once another session has taken over (`.claude/skills/`).
- Hooks (`.claude/hooks/orchestrator.py`): each session starts knowing its own id and where each
  handoff stands; past 500k tokens of context, a note says it is a good time to cycle; a session
  that has been taken over can't start new agents (an `in-flight:` description gets through).
- **Compact instructions.** If a session compacts anyway, keep: task ids, agent names and ids still
  running, PIDs, log paths and restart commands, what is uncommitted and whose, and Anita's latest
  asks in her words. An orchestrator re-reads its `notes/handoff.md` after compacting.
