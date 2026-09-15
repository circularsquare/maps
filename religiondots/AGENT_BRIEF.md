# religiondots — standing brief for an agent

**You were spun up to move this map forward by one country, and nobody is watching this session
turn by turn.** That is the point. Anita's instruction, 2026-09-08: *"trust yourself to make a
reasonable decision, and mark it for review"* — with the rider that things marked for review
should stay **rare**, because a pile of them is just the decisions handed back.

Read `CLAUDE.md` (short) and this. `COMMANDS.txt` has the runnable checklist and `spec.md` has the
reasoning; read the section that applies, not the file.

**Do not commit or push.** Anita reviews and commits herself, everywhere in `maps/`.

---

## The one-paragraph version

Claim a country. Build it, or scout it and write down what you found. Decide the arguable calls
yourself and record the reasoning where the next session will trip over it. Ask Anita only for
what §3 lists, which for most countries is nothing. **Stop while you still have context left to
write the record** — a country parked cleanly at checkpoint B is worth more than one abandoned at
90% with the findings still in your head.

---

## 1. Orient, then pick — ten minutes, not an hour

```
python tools/claim.py                 # what is claimed, what is parked, what is free
python tools/ask.py                   # what is already waiting on Anita, so you don't repeat it
python tools/oracle.py --list         # UNSD's own list, with counts
python tools/where.py <cc>            # once you have a candidate: its files, record sections,
                                      #   queue rows, asks and rulings, with line numbers
```

**`WebSearch` and `WebFetch` are the source-hunting tools.** A Gemini API route was built and
withdrawn on 2026-09-08: grounded search is not in the free tier and Anita's call was that paying
for it was not a sure improvement over WebSearch. Do not rebuild it.

The rule that outlives the tool: **a search result is a lead, not evidence.** Open the URL, read
the table, cite the release. No figure from a search summary reaches `sources.md`, a mapping or a
`note_public` — and *"the office does not publish this"* from a search is worth nothing at all,
because that is §12's standing instruction that nothing is truly dead.

**Take a parked country first if there is one.** Its handoff note in `handoff/<cc>.md` says which
`COMMANDS.txt` step it stopped after and what is already on disk. Resuming is cheaper than
starting, and a parked country is the one thing here that rots.

Otherwise take the top free row in `queue.md` unless you have a reason not to. Reasons that
count: it needs an account or a login (skip, it is blocked, see `[[reference_ipums_account]]`);
another session is working its neighbourhood; the row's own note says it is a browser job.
"It looks hard" is not a reason — the queue is roughly ordered by what the country would add.

```
python tools/claim.py take <cc> --id <sid> --note "what you are doing"
```

**`<sid>` is whatever id you were handed.** If your prompt named one, use exactly that string and
do not derive your own — subagents share their parent's scratchpad directory, so three agents
deriving it independently all get the *same* id, and `claim.py` can then neither say who holds
what nor stop one of them releasing another's claim. Only fall back to the last path component of
your scratchpad directory if you were given nothing.

**Then read the playbook for your route, and `playbooks/geography.md` always.** A census table is
`playbooks/census_table.md`; a survey is `ess`, `lapop`, `afrobarometer`, `arabbarometer`, `cab`,
`lits`, `wvs` or `dhs_mics`. Each is the short, current form of spec §12 for that route: how to load
it, its traps and the check that catches each, the shared code, and the rulings. Spec §12 keeps the
reasoning; go there when a playbook names a section.

**And read the lines in `ask/RULINGS.md` for your country and route** (`where.py` lists them).
Each ruling says what Anita decided and what she did not. A case that is only similar is not
settled by it; Tanzania's builder stretched the Nigeria ruling that way.

**Scratch files go in `<scratchpad>/<sid>/`, never loose in the shared scratchpad.** Subagents
share one scratchpad, and on 2026-09-14 two agents silently overwrote each other's helper scripts
(`shot.js`, `pdftext.py`) there. Make your own folder first and keep everything under it.

**Questions for Anita go in `ask/`, never only in a report or chat.** `ask/OPEN.md` is the one
file she keeps open, and it only shows ask files (see `tools/ask.py`). Give every ask a
`--summary` of 40 words or fewer.

**Your final report to whoever spawned you has a fixed shape, under about 300 words:**
1. one outcome line: drawn / parked at A-B-C / closed / scouted, `cc`, grain, people;
2. at most three findings another session needs;
3. calls you made that someone might reverse, one line each;
4. asks filed, by number;
5. files touched.

Everything else lives in `sources/<cc>.md`. Anita, 2026-09-14: long reports are more than
she can keep up with.

**If the free queue is thin or every row left is walled, switch to SCOUT mode**: take a region
nothing has swept, probe five or six offices, and write what came back into `queue.md` and a
`sources.md` section headed `## scout-<YYYY-MM-DD>-<region>.`. A good scout leaves the next five
build agents something to do.
`sources.md`'s existing §11 sweeps are the model. Claim the countries you probe so two scouts do
not sweep the same band.

**Or re-check an old negative.** `python tools/negatives.py --shape` lists closed records shaped
like the ones that turned out wrong on 2026-09-14: closed because one release or listing lacked the
table, never on the questionnaire, and silent about what was left unchecked. Its top rows are
scout work too.

---

## 2. Decide it yourself, and write it down

Almost everything here is reversible, and the record is the deliverable. **These are yours:**

- Which release, which vintage, which census year. Which population base — COD-PS or the office's
  own, and Ecuador (§9bn) chose the office's over COD's 3.4% error.
- How to join names to boundaries, and which of two boundary files is right. Name joins are the
  single biggest silent-failure source here (`[[reference_name_join_wrong_neighbour]]`); assert
  the join, don't eyeball it.
- Every source category → node mapping. Arguable ones go in the mapping module's `REVIEW` dict
  **with the reason**, which is the existing convention in 95 files and is *not* capped — that is
  the cheap tier and you should use it freely.
- Whether a category carries its own geography, when the source is a survey. There is a stated
  test (split-half rank correlation, §9bi/§9bl) — apply it, and if you override it say so in code
  the way `sources/gt.py`'s `OVERRIDE` does.
- Whether a country is worth finishing, and when to stop. **Giving up early is allowed and
  expected** (§12). Write the negative so it can be reopened: what was asked, of what, on what
  date, and what came back.
- Every word of `note_public`, `how`, `fill`, `grain`, `gap`. Read the field docstring at the top
  of `countries.py` first; the voice rules there are hers and are asserted at import.
- `gap_share` beside `gap`. **Run `python tools/gap_share.py` instead of working it out** — it
  computes the share exactly wherever the hole is a column the census printed, and `--write`
  fills it in. Hand-write it only for people who were never in any table (an age cutoff, a
  region nobody enumerated), and leave it out entirely where nobody has quantified them: it is
  the width of a segment on the legend's composition bar (spec §10.4), so a guess is drawn.
- Running things. In this directory, **run anything you estimate under ~3 hours yourself** rather
  than handing it to Anita ([[feedback_religiondots_run_freely]]). Background the slow ones,
  `tiles.py` included, and keep working while they go.

**A new node in the shared taxonomy is yours too**, if it is a real body a census counted —
`christianity.melanesianindependent.cfc` and `indigenous.solomon` were both added without asking.
Run `taxonomy/build_tree.py` after, or the viewer greys out whole countries.

---

## 3. What actually goes to Anita — and the bar is high

Use `python tools/ask.py new <cc> --title "..."`, then fill in the file it prints. **An ask is
never a block.** The template forces you to state the decision you already took and what
reversing it costs, so the work ships either way and she is choosing whether to flip it, not
being asked to unstick you.

**Aim for zero to one per country.** Ten open asks is worse than one wrong call, because each one
is a decision she has to load context for.

It goes to Anita if it is:

- **§14** — whether a country may be drawn at all, at what resolution, or whether a group's safety
  is affected by publishing where they live. §14's opening line is an invitation and not a
  fallback ([[feedback_flag_ethics_for_discussion]]). This is the one case where "this looks like
  §14" is enough on its own, and where stopping to ask is right.
- **A source whose terms are unclear or which bans what we are doing** — Nişanyan's ToS, CFPS's
  refusal. Do not push through one hoping.
- **Something that changes an already-drawn country's numbers**, or a rule everyone shares: moving
  a threshold, changing what `derived` means, reordering the legend.
- **A new top-level root, or a single-country node that adds a legend row nobody else uses.** Her
  `todo.txt` already carries "maybe get rid of some of the jewish categories" — she is watching
  legend bloat. Tonga's four Methodist churches were flagged for exactly this and that was right.
- **Anything needing an account, money, or her identity.**

It does **not** go to Anita if it is a mapping call, a join, a vintage, a threshold you applied as
written, a wording question, or a country you decided to abandon. Those go in the record.

### If you want a second opinion, get one — from an agent, not from Anita

When a call is genuinely close and you would like someone to look at it, spawn a short-lived
reviewer rather than filing an ask. Its brief is `.claude/commands/rd-review.md`; give it the
question and let it read.

**Point it at the files, not at your reasoning.** *"Read `taxonomy/sb2019.py` and `sources/sb.md`
and tell me where the South Sea Evangelical Church should sit"* gets you something. *"I put the
SSEC on `christianity.evangelical` because of Kenya, does that sound right"* gets you agreement,
because you handed it the answer with the question. The whole value of a second agent is that it
did not start where you started, and a summary throws that away.

Take its answer as an opinion. You still own the call, and *"I asked and disagreed, here is why"*
is a perfectly good line in `sources/<cc>.md`.

---

## 4. The context rule — park at a checkpoint, not at the wall

Anita's numbers, 2026-09-08: *"its fine to run up to like 75% context but like 50% is where we
should start to consider stopping."* So there are two lines, and they do different jobs.

**At about 50% — start looking for the exit.** You are not stopping yet. You are declining to
take on *new* scope: no second data source, no fresh geography hunt, no rescue of a join that is
already failing. Aim at the next checkpoint below and get there.

**At about 75% — stop where you are.** Park, whatever state you are in. Everything below 75% is
yours to spend; what is not yours to spend is the last quarter, because that is the budget for
writing the record, and a country whose findings died with the session is the only genuinely
wasted run available here.

The checkpoints, because they are where a handoff is cheap:

| | what is true | if you cross 50% here |
|---|---|---|
| **A** | You know the table exists, and have its URL, its tier and its category list. Nothing downloaded. | **Do not start the fetch.** Write the scouting record into `queue.md` and `sources/<cc>.md`, park, stop. This is a good outcome, not a failed run. |
| **B** | `sources/<cc>.py --fetch` produces `data/normalized/<cc>.csv` and it reconciles. | **Park here by preference.** The expensive, un-resumable part is on disk; the next agent writes the mapping against a CSV that exists. This is the designed handoff line. |
| **C** | Mapping written, `countries.py` entry in, `check_mapping.py` passes. | **Push on through to the end**, even past 75% — steps 5–9 are mechanical and cheap, and under a supervisor steps 10–12 are not yours (§6). A country registered without dots leaves the tree in the half-state `claim.py` reports as *registered but NOT built*, which is worse than either finishing or never having started. |

To park:

```
python tools/claim.py park <cc> --id <sid>
```

It writes `handoff/<cc>.md` and drops your claim. **Fill the handoff in before you stop** — the
last `COMMANDS.txt` step you completed, what is on disk, what you were about to do, and the one
thing that will bite the next person. A handoff you did not write is a country nobody resumes.

Parking is not failure and does not need apologising for. Running out of context mid-mapping with
nothing written down is the only bad outcome available here.

**One thing worth saying plainly.** If the country you took turns out to be miserable — a portal
that lies, a join that will not close, a source whose terms you cannot read — park it and say so
in the handoff, in those words. That is more useful to the next session than a cheerful note, and
Anita would rather read it than have you grind. Same if the brief itself is wrong about
something; `spec.md` §12 is meant to be added to.

---

## 5. Finishing

`python tools/claim.py done <cc> --id <sid>` prints the tail. It is:

- `sources/<cc>.md` written, and a section appended to `sources.md` headed
  `## <cc>-<YYYY-MM-DD>. <title>`, with `b` after the date if your country already has one that
  day. Cite it as `sources.md §<cc>-<YYYY-MM-DD>`. No new §9 letters: they were claimed
  first-come and collided (spec §12).
- The `countries.py` entry with `note_public` and `gap=`, `python tools/gap_share.py <cc>` run
  and its figure written if it found one, and `python tools/check_md.py` clean.
- `python tools/built_countries.py --check` naming nothing.
- The row moved to *Drawn* in `queue.md`, or taken out.
- `handoff/<cc>.md` deleted if you resumed one.
- Anything that generalises added to spec §12. A trap that cost you an hour costs the next
  session five minutes to read.

Then write your final message in the fixed shape from §1. A supervisor reads it, not Anita.

---

## 6. Two or three of you are running at once, by design

`countries.py`, `spec.md`, `sources.md`, `taxonomy/branches.py` and `COMMANDS.txt` **will change
under you** and your tooling will say so on an edit that applied fine. Make surgical edits against
unique anchors. **Never rewrite a shared file wholesale** — that is the only move here that
destroys someone's work. Before any `Write` to `sources/<cc>*`, `taxonomy/<cc>*` or `data/`, look
at whether the path already exists; that habit, not the claim, is what stopped the Peru accident
happening twice. Full detail in spec §12.

Getting a fright and stopping costs more than the collision would have.

**The one collision the country claim does not cover is the build tail.** Steps 10-12 are not
per-country: `country_shapes.py`, `tiles.py --countries` and `buffers.py --countries` each rewrite
a file covering the whole map, and step 11 runs `--no-atomic`, writing the archive in place. Two
agents finishing twenty minutes apart both run all three over the same files, and the symptom is
not an error — it is a few wrong tiles on a map that looks built. So run them through the lock:

```
python tools/build_tail.py --id <sid>        # waits if held, then runs 10, 11 and 12 in order
python tools/build_tail.py                   # just says who holds it
```

**Waiting is the correct outcome, not a delay to route around.** The build covers every country
from whatever dots are on disk, so the run you are waiting on very likely already includes yours.

**Under a supervisor, the build tail is not yours at all.** If `rd-super` spawned you, your prompt
says so: stop after `COMMANDS.txt` step 9 and `claim.py done`. `claim.py` then lists your country
as waiting for the build tail, and the supervisor runs it for every waiting country at once, about
hourly. Working on your own, run it yourself as above.

---

## 7. If you are the supervisor

Follow `.claude/commands/rd-super.md`. It holds the supervisor's whole job: a distinct id for each
agent, when a reviewer or a scout takes a builder's slot, and when to stop and hand back to Anita.
It lives in one place so two copies cannot drift apart.

Anita reads `ask/OPEN.md` and `runlog.md`. That is the whole interface.
