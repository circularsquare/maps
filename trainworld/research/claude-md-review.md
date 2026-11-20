# Review of the CLAUDE.md files (agent mdreview, 2026-11-20 UTC)

Read-only review. Nothing was edited.

## Verdict

The files are in good shape and mostly match what Anthropic recommends: concrete rules, real commands, pointers to
docs, and hooks for the rules that can't bend. I would not rewrite anything.

Size is not the main problem. A trainworld agent carries about 12k tokens of instructions and the orchestrator about
21k, which is 2-4% of a 500k session. The real costs are elsewhere:

1. A few rules say different things in different files. Anthropic's docs say that when two instructions contradict,
   "Claude may pick one arbitrarily".
2. The same procedure is written in three or four places, against the maps file's own rule ("A project's rules live
   once").
3. A handful of very dense passages (the changelog bullet, the tasks bullet, the asks bullet) pack five or more rules
   into one paragraph. Those are the ones most likely to be half followed.

About two hours of agent time would cover changes 1-6 below. The rest can wait until a section is next touched.

## Sizes

| File | Lines | Rough tokens | Who loads it |
|---|---|---|---|
| `C:\Users\anita\.claude\CLAUDE.md` | 288 | 3,900 | every session and every subagent, in every project |
| `C:\Users\anita\projects\maps\CLAUDE.md` | 49 | 700 | every maps session and subagent |
| memory index `MEMORY.md` | 127 | 3,800 | every maps session (and this subagent got it too) |
| `trainworld\CLAUDE.md` | 189 | 4,100 | every trainworld agent, when it first works in trainworld |
| `trainworld\docs\orchestrator.md` | 233 | 5,900 | the orchestrator only |
| takeover skill | 72 | 1,400 | the orchestrator, at takeover |
| handoff skill | 96 | 1,450 | the orchestrator, when run |
| `trainworld\notes\handoff.md` (today) | 74 | 1,600 | the orchestrator |
| `trainworld\SPEC.md`, for scale | 414 | 11,100 | any agent that follows "Read this, then SPEC.md" |

Anthropic's target is "under 200 lines per CLAUDE.md file. Longer files consume more context and reduce adherence."
The global file is over (288), the trainworld file is just under, and orchestrator.md is under in lines but has
single lines of up to 951 characters, so the line count flatters it.

The memory index loads only its first 200 lines or 25 KB. It is at 127 lines and 15.4 KB. Past the cap, entries
stop loading without any warning.

One thing I saw that differs from the docs: they say a subagent does not get the main session's memory index, but
this subagent did get it. I rank nothing on that.

## Changes I'd make, ranked by benefit

### 1. Settle what the "For you" block holds in an orchestrator session (conflict)

- Global file, Final messages: "It is a standalone account of the whole task, never a delta", "the final message
  restates all of it", and the block collects "**every** ask you have for her".
- `trainworld\docs\orchestrator.md`, TODO.md and asks: "Final messages are short and their For you block carries
  only asks new in that message, not the standing ones (Anita 2026-11-17: 'stop repeating things unless i ask')".

For an orchestrator that runs for a day, "the whole task" and "every ask" pull one way and the 11-17 rule pulls the
other. Which one an orchestrator follows is luck.

What I'd do: reword the global file, one sentence. Something like: "In an orchestrator session the task is the round
since her last message, and the block holds only asks new in this message; standing asks live on the board." The
orchestrator.md line then stays as it is.

Related, smaller: the memory entry "Next-up outline" asks for a few bullets of what comes next at the end of round
reports. The global file doesn't mention it and orchestrator.md says final messages are short. If she still wants
the outline, it belongs in orchestrator.md as one line; if not, delete the memory entry.

### 2. One number for the handoff file's length (conflict)

- Handoff skill: "under about 40 lines" for trainworld.
- `trainworld\tools\tasks.py check`: flags it only over 80 lines ("keep it under 80").
- The file today: 74 lines, about half of it a dated diary of her words going back two days.

What I'd do: pick one number and put it in both places. I'd say 60, and add to the skill that "Her words" keeps
standing asks and this session's calls, with older answered items dropped once the task or DECISIONS line holds
them. The 40 is being ignored, and an ignored number teaches agents that the numbers in these files are soft.

### 3. Write the cycling and wind-down procedure once (duplicate, four places)

The same steps (start no new agents from about 450k, hand over at 500k, finish the landing you are in, one line to
her, message the new orchestrator once, end agent wake-ups with one word) appear in:

- `orchestrator.md`, "Your own cycling" (9 lines)
- the handoff skill, "When to touch it" and "After another session takes over"
- the takeover skill, steps 4 and 5
- `maps\CLAUDE.md`, "Orchestrator sessions", plus the memory entry "Orchestrating subagents"

They agree today, but four copies will drift, and the maps file's own rule is "change it there and delete the
copies".

What I'd do: keep the full steps in the handoff skill (it is the one read at the moment of winding down). Cut
"Your own cycling" in orchestrator.md to three lines: the 450k and 500k numbers, the `next_orchestrator.py` command,
and "then wind down per /handoff". Trim the memory entry to its two points that live nowhere else (briefs and
thanks) or delete it, since orchestrator.md already has both.

Same kind, smaller:
- The advisor rule is written twice with nearly the same list (`trainworld\CLAUDE.md` "Advisor" and orchestrator.md
  "Advisor trial"). Keep the list in CLAUDE.md. orchestrator.md keeps only what is the orchestrator's: the launch
  flag, about 3 calls a session, the log.
- "Show her every in-game string, old -> new; she answers 'text acknowledged'" is in both files. Keep the CLAUDE.md
  one.
- "Keep the work moving" is a standing rule but lives at the bottom of the takeover skill, which is read once at the
  start. Move it to orchestrator.md "Supervising", which already points at it.

### 4. Break up the three densest passages (reword, no rule changes)

Anthropic: "group related instructions under markdown headers and bullets. Organized sections are easier for Claude
to follow than dense paragraphs."

- `trainworld\CLAUDE.md`, the Changelog bullet: 12 lines, five dated quotes, and a command whose argument has the
  tense rule written inside its own quotation marks. I'd split it into: when an item is due (2 lines), the command
  in a code block, the area list, what gets no item (one short list). The dated quotes go (see change 7).
- `trainworld\CLAUDE.md`, the Tasks bullet: 14 lines covering format, priorities, tags, commands and rewriting. The
  format half is already enforced by `tasks.py check`. I'd keep one example line, the field order, and the commands,
  and say "check lists what is malformed". Humanlayer's version of this advice: "Never send an LLM to do a linter's
  job."
- `orchestrator.md`, the asks bullet: one 951-character line holding four rules (how to add, how to answer, what the
  block carries, how to catch her up). Four bullets.

Also the "Playing the game in a check" bullet repeats `play.mjs`'s own how-to ("how-to at its top"). Two lines would
do: what it is for and where the how-to is.

### 5. Line up the commit and push rules (soft conflict, three files)

- Global file: "In shonei and shonei-server (private repos) commit freely as you go" and orchestrators' "subagents
  never commit".
- `trainworld\CLAUDE.md`: "no commits (a brief that touches shonei-server may say to commit there ...)".
- Global file: the trainworld orchestrator "may push maps master a couple of times a session". orchestrator.md:
  "push maps master regularly".

A trainworld agent working in shonei-server can read this either way. The global paragraph also carries three
projects' exceptions and two dated quotes into every session of every project.

What I'd do: in the global file keep the default, the list of projects that have their own rule, and "pushing needs
her say-so unless that project's file says otherwise". Move the trainworld details (push frequency, shonei-server
after a deploy) to orchestrator.md, where most of it already is. Add "(trainworld subagents: only when the brief
says)" to the shonei-server line.

### 6. Prune the memory index (stale and duplicate entries)

- Possibly stale: "Publish needs a text ack" (11-12) says every new changelog line needs her explicit ok before a
  publish. `trainworld\CLAUDE.md` (11-19) says "Changelog items are not asked about: she reviews them separately on
  the board". They can both hold (landing is not publishing), but an orchestrator reading both may ask her about
  changelog text again. Hers to settle: is the board review the ack now? Then one line in orchestrator.md
  "Publishing" and the memory entry goes.
- About ten entries restate a rule a CLAUDE.md already holds, which the maps file says not to do ("a memory file
  links to the project file rather than restating it"): One local server, CPU cap is the ledger's, Showing her
  files, Never edit todo.txt, Permission refusals, Orchestrator models, Orchestrating subagents, Map label voice and
  Panel prose voice (both covered by "User-facing text"), No identity in requests.
- The hub entries (China census, Korean rail, MapLibre, religion sources, data access) are the right pattern: one
  line in the index, 60 files behind them. More hubs would keep the index well under its 200-line cap. The
  twenty-odd reference lines for religiondots, languagedots and helper1m data sources could become two or three
  hubs.

### 7. Dated "(Anita, 2026-11-xx)" quotes: sort them, don't strip them

There are 17 in `trainworld\CLAUDE.md` and 31 in orchestrator.md. They cost few tokens (about 300 in all). The cost
is that they lengthen sentences that are already long. Anthropic's own trimming tool keeps "pitfalls, rationale, and
conventions", so the reason should stay. The date and name mostly should not. Four kinds:

- The quote sets how strong the rule is or where its edge is. Keep the reason, drop the date. Examples: "reports ran
  30-60 lines", "25 minutes without an answer", "its a pet peeve of mine", "just don't auto address player requests
  where we have to make an actual new design decision".
- A trial or a "for now" rule. Keep the date, since it says when to look again: "Advisor trial (2026-11-16)",
  "Resources, once a day for now (2026-11-11)".
- A bare attribution with no reason: "(Anita, 2026-11-17)" after the wait wording, "(Anita, 2026-11-11)" after
  reused text. Move the date to a DECISIONS.md line, which is grepped when someone wants to know.
- An incident story: "five copies on 11-17", "11-12 the main tree lost .bin and 15 packages", "(d4164171)". Keep the
  rule and the symptom to recognise ("Cannot find native binding"); the story goes to notes.

Do this when a section is next edited, not as its own pass. The three dated quotes in the global file are fine.

### 8. Move publishing into a skill (optional)

`orchestrator.md` "Publishing and servers" is about 45 lines: the changelog steps, playtest list, changelog review,
new zones image, open-source sync, PR replies, R2 purge. A publish happens every few days; the orchestrator reads
all of it at every takeover.

Anthropic: "A long release procedure sitting in a skill doesn't weigh on your unrelated bug fix the way it would in
CLAUDE.md." The larger gain here is timing: a skill is read at the moment of publishing, instead of 300k tokens
earlier. orchestrator.md keeps the heading rule ("each only on her word, never overnight") and "run /tw-publish".
The 11-19 playtest list rule exists because a step was skipped, which is the kind of slip a just-read checklist
prevents.

### 9. "Read this, then SPEC.md" against "Read sections, not whole files" (wording)

`trainworld\CLAUDE.md` opens with "Read this, then `SPEC.md`", and Doc rules says "Read sections, not whole files."
SPEC.md is about 11k tokens, as much as every instruction file an agent loads put together. I did not measure
whether agents read all of it. If a CSS agent does, that is the largest avoidable read in the whole setup.

What I'd do: "Read this, then the SPEC.md sections your task touches (sections 1 and 2 always)". The one study of
this (ETH Zurich, ICLR 2026) found that context files gave no improvement in task success and raised cost by over
20%, and that repository overviews "are not helpful". It tested single Python bug fixes, not a setup like this one,
so take it as a nudge only.

### 10. Small cuts

- `orchestrator.md` Tools: "`mark_taking.py` is superseded by `tools/tasks.py start`". A note about a tool nobody
  should use. Cut it (and the file, if nothing calls it).
- Global file, Subagent model tier, and orchestrator.md "Model" say the same thing at two lengths. Fine to leave;
  if either changes, change both.
- `maps\CLAUDE.md` "Compact instructions": add that an orchestrator re-reads `docs/orchestrator.md` too. The docs
  say only the project-root CLAUDE.md is put back after a compaction; a file that was read in conversation is not.

## What I checked and found fine

- 68 files and tools named in these files: all exist. No rule points at something that is gone.
- All 125 memory files the index links exist.
- `trainworld\docs\ui-panels.md` "Hover tags" is about how tags work, not wording, so the global file's hover rules
  are not duplicated there.

## Good, keep

- **Hooks for the hard rules**: the AWS guard, the tree guard, the agent guard after a takeover, the report hook,
  the 500k stop hook. This is what Anthropic recommends: "If something has to happen every time, make it a hook."
  A handful of hooks is a sensible number; they warn against dozens.
- **"Where rules live"** in the maps file. It is the right rule. Changes 3, 5 and 6 are just applying it.
- **Live state from a command** (`handoff.py status`, `asks.py`, `tasks.py`, `cores.py list`) and not from prose
  that goes stale. "never written into a doc" for the core cap is the model for the rest.
- **The "User-facing text" section**. It is a third of the global file and I'd still leave it whole. The rules are
  concrete, each has an example, and it holds a preference that no code or tool can show an agent. Moving it to a
  rule file that loads only for some file types would risk it being absent when an agent drafts text in chat.
- **"Final messages"** and the For you block. Specific, and an agent can check its own message against it.
- **Reports capped at 15 lines with detail in notes/**. With many agents reporting into one context, this matters
  more than any trimming of the files.
- **Reasons next to rules.** "it pops dialogs on her screen", "each report stays in the orchestrator's context".
  A reason tells an agent where the rule stops applying, and Anthropic's trimming tool keeps rationale on purpose.
- **Skills that say "these are guidelines; adapt and say why"**, and the standing invitation to say when something
  is wrong.
- **Little shouting.** Capitals appear three times in the global file (commits, killing processes, AskUserQuestion),
  each where it is meant. Anthropic notes that piling on
  emphasis "mostly makes Claude over-apply the rule".
- **The Traps section** of orchestrator.md. Every item is something the code can't tell an agent. Dense, but earned.

## A cheap way to repeat this

Claude Code has a built-in check: `/doctor prompt-audit`. It looks for "references to files or commands that don't
exist, and files that contradict each other" across CLAUDE.md files, skills and rules, and proposes edits without
applying them. Worth running once a month or after a busy week of rule changes.

## Sources

Read in full (primary):
- Anthropic, "How Claude remembers your project": https://code.claude.com/docs/en/memory
  (the 200-line target, contradictions, memory index cap, what survives compaction, prompt audit)
- Anthropic, "Best practices for Claude Code": https://code.claude.com/docs/en/best-practices
  (keep CLAUDE.md short, hooks for rules without exceptions, skills for procedures, emphasis)
- Anthropic, "Subagents", What loads at startup: https://code.claude.com/docs/en/sub-agents
  (subagents load every CLAUDE.md level; Explore and Plan agents skip them)

Read as summaries:
- Gloaguen and others, "Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?", ICLR
  2026: https://arxiv.org/abs/2602.11988 (abstract only)
- Humanlayer, "Writing a good CLAUDE.md": https://www.humanlayer.dev/blog/writing-a-good-claude-md
  (their own file is under 60 lines; "Prefer pointers to copies"; one team's view)
- "The Demotion Ladder: a year of governing Claude Code":
  https://hackernoon.com/the-demotion-ladder-a-year-of-governing-claude-code
  (one person's account: rules that fail as prose move to hooks and checks; 5,000 lines of rules cut to 1,000;
  self-reported numbers)
- "How I structure CLAUDE.md after 1000 sessions": https://dev.to/joozio/how-i-structure-claudemd-after-1000-sessions-2gkk
  ("a long list with exceptions creates conflicts"; project details kept out of the global file)

Not done: I did not read any public repository's own CLAUDE.md or AGENTS.md. The one I tried (openai/codex) was not
at the address I guessed, and I stopped there to save the shared search cap. Used 5 searches and 8 page reads.
