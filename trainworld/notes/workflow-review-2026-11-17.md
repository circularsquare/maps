# Workflow review, 2026-11-17

By workflowreview (Fable), for Anita. Covers about 11:40 am to 2:10 pm your time: the maps-58 and maps-45
orchestrator sessions, their agents, the commits, the diary and the board. Times are your local time.
Nothing in the project was changed.

## Short version

The work itself is going well: 76 commits since 8 am, most things you asked for were in master within the hour,
and checkers caught real bugs. What cost you attention was mostly around the edges: not knowing whether the
orchestrator was alive, two or three tabs talking to you at once, and small questions that should not have
needed you. The five changes below would take about 2 hours of agent time in all.

## Changes, ranked by what they save you

### 1. The orchestrator never goes quiet on you

What happened
- 12:22 to 12:47 pm: maps-58 landed five agents' work, then ran the sim tests and the full game test while
  waiting on each. You asked at 12:41 how there can be more agents than cores, and at 12:47 "why have you been
  thinking for 25 minutes straight and not responding?"
- maps-58 wrote the lesson in the diary, but not in the rules. The next orchestrator did the same thing in
  smaller form an hour later: a landing that rebuilt WebAssembly (1:31 to 1:35) and a game test (1:37 to 1:41).
  Your "oh sorry if i stopped something it was by accident" (1:32) and "hows 3565 going btw" (1:40) both waited
  until 1:41.
- The board could have answered both, but it was stale. At 2:09 pm it still named maps-58 as orchestrator and
  showed 8800 on the build from 12:47. It can show how long each agent has been running, but the start time
  is never written, so that part stays blank.

Change
- One line in docs/orchestrator.md: anything that takes over a minute (a landing with a WebAssembly rebuild,
  the game test, an 8800 rebuild) starts in the background, and the orchestrator answers you first.
- The tools keep the board current instead of the orchestrator: starting an agent writes its start time, the
  8800 rebuild writes its own build id, and the takeover writes the orchestrator's name.

Cost: the rule is free. The board part is about 45 minutes for a Sonnet agent.

### 2. One tab talks to you after a takeover

What happened
- Five orchestrators today, so four takeovers (6:13, 9:40, 11:41 am and 1:15 pm). The last three sessions ran
  3.5 h, 2 h and 1.5 h.
- Old tabs kept talking. You wrote "answered some of the service period questions in other orchestrator, sorry
  lemme know if i missed anything" (11:50 am) and "oh sorry was responding to wrong agent" (12:10 pm). maps-60
  was still passing your words along at 1:38 pm, two orchestrators later.
- After handing over at 1:15 pm, maps-58 wrote 15 more messages in its own tab. Most repeated what maps-45 was
  telling you in the same minute, and the rest said there was nothing new.
- maps-58 handed over with seven agents running. Each of their reports reached the old tab and was copied by
  hand to the new one, so each was paid for twice. Three of the seven were started in its last 15 minutes.

Change (both are edits to the handoff and takeover skills, not new process)
- After handing over, the old tab sends you one line ("talk to maps-45 now") and then writes nothing more to
  you. It only passes agents' reports to the new orchestrator. It never asks you anything.
- An orchestrator close to its limit (about 350k) starts no new agents. It lists them in the handoff and the
  new one starts them. Today that would have cut seven passed-on reports to four.

Cost: two short edits. An agent started by the new orchestrator begins a few minutes later than it would have.

### 3. Changelog duplicates get fixed without asking you

What happened
- c116 "ponytails now sit behind the head" fixes a hairstyle that is itself new in this update (c031). Players
  never saw the old ponytail.
- c121 "real world stations are drawn as soft purple dots" was added 22 minutes after c117, which still says
  "as small rings". Both are unpublished, and c117 is now wrong.
- c106 and c111 contradicted each other, and c104 needed hiding. You were asked about both ("ok can hide
  changelog thing", "sure @ changelog. not sure if it was already done"), though your standing ask is no
  questions about changelog lines.
- Agents quote changelog ids from their own copy, and the ids change when the work lands. Three agents
  reported "c113" today and two "c119". The ids in reports mean nothing.

Change
- trainworld/CLAUDE.md, changelog rule, one sentence: if the thing you changed is new or already listed in the
  unpublished changelog, reword that item instead of adding one.
- You can already reword an item on the board. Agents can't: the changelog command gets a way to do it, and
  the landing command prints the final ids.
- The orchestrator merges or hides a duplicate itself when it lands the work and tells you in one line, the
  same way it may fold duplicate tasks.

Cost: about 30 minutes for a Sonnet agent, plus the sentence.

### 4. A redraw starts with three versions on one sheet

What happened
- The ponytail took 40 minutes from your message (1:21 pm) to 8800 (2:01 pm), with three agent rounds. The
  first redraw moved the tail from 12 px past the face to 7 px, which barely showed. The orchestrator sent it
  back and asked for three versions. That second round took 3 minutes, and you picked from its sheet in two.
- The real station dots took 9 minutes from your message to 8800, in one pass, because you gave numbers
  ("60% of current, filled in, purple").

Change
- In the briefing rules: when you ask for a look to change and give no numbers, the first brief asks for three
  clearly different versions beside the current one, at map size, on one sheet. The orchestrator lands its
  pick and sends you the sheet. You swap by letter if you prefer another, as you did with b.

Cost: free.

### 5. The landing command stops failing on its last step

What happened
- The one-command landing built at noon failed its last step 9 times out of 19 today. Briefs tell agents to
  archive their task when done, and the command then tries to archive it again and reports a failure. Nothing
  breaks, but the orchestrator stops and checks each time.

Change
- The command treats a task that is already archived as done.

Cost: about 15 minutes for a Sonnet agent.

## Smaller things

- **Agents waiting on a build.** The tram terminal agent started its WebAssembly build detached at 2:01 pm
  and tried to wait for it. A plain wait was refused by the harness, and the watch it used instead does not
  hold the turn open. Its turn ended, the harness demanded its report, and it reported unfinished work three
  times in a minute. It was resumed twice before it could finish. The Waiting line in trainworld/CLAUDE.md should say: run a
  build or test you need the result of inside one call with a long time limit, and never end your turn while
  it is running. The earlier fix did hold: since "stop background work before reporting" went into briefs at
  1:03 pm, I found no report that arrived twice. Before that, two reports came four or more times each.
- **A landing that breaks a check for everyone else.** The panel subtitles landed at 12:10 pm with a changed
  placeholder in English only. The translation check then failed on master until 12:44. Four agents each
  reported "28 of 29" and explained it, and a fifth was started to fix it. The landing command could run the
  quick checks (the translation one takes seconds) on the merged code and stop if one fails. About 30 minutes
  to add.
- **Questions that cost a second round.** "Allow crossovers on tram track?" got "what does crossover mean in
  this context?". The feedback inbox dot was listed among things to test on 8800, and you replied "the light
  blue dot should not be for players". A question should say in a few words what the player would see, and a
  change to one of your own tools should be named as yours. No new rule needed. The orchestrator can just
  do this.
- **Handoff file churn.** maps-45 edited the handoff file 30 times in 50 minutes, mostly its list of agents.
  The board file already holds that list and is kept by the task command. The handoff could point to it.
  About an hour, and it only saves orchestrator tokens, so it is the lowest priority here.
- **Report length.** The 15-line rule holds for small tasks (5 to 10 lines). Big ones still ran 17 to 53 lines
  with long lines. I would leave it. Reports, passed-on ones included, were roughly a tenth of what filled
  maps-58's context. Its own reading, briefs and bookkeeping were more. A takeover's first ten minutes alone
  added 50k to 100k tokens.

## What the orchestrator noticed, checked

- The landing command fails its last step when the agent archived its own task: confirmed, 9 of 19 (item 5).
- An agent kept ending its turn during its build: confirmed, but the agent was trying to wait. The rule it
  followed is the problem (first bullet under Smaller things).
- The ponytail came back too timid: confirmed (item 4). Landing c before your pick cost almost nothing.
- A changelog item repeated an earlier one: confirmed, and there were three such cases today (item 3).
- Seven agents reporting through the old session: confirmed (item 2).
- Decisions reaching you as questions: mostly not a problem. The tram terminal, tram crossovers and the fare
  base were yours by the rules, each came with a recommendation, and you overrode one. The ones that were not
  yours were the changelog questions (item 3) and "file a task at P3 for short single-track lines?", which
  the orchestrator could have decided.

## Working well, keep

- You answer in a minute or two, and the orchestrator files and starts things within a minute of your message.
  Your scratch pad at 2:03 pm was sorted and filed by 2:04.
- Checkers on risky changes. This morning one failed the first weekend counts build and found a real bug in
  headway lines. Seven more ran this afternoon.
- Asking you about what players can build, with a recommendation each time.
- Problems turned into fixes the same day: the landing command, short reports, stopping agents after their
  report, the shared playtest harness (asked for at noon, in master by 12:47).
- One review image with its full path, and "text ack" for strings.
- One 8800 rebuild per batch, and master pushed after each.

## For you

Questions
- Do items 1 to 5? I would do all five. About 2 hours of agent time, none of it touching the game.
- The orchestrator hands over at 400k tokens, which now comes after about 1.5 hours. Raise it to 500k? That
  is roughly one takeover fewer per afternoon, at a higher cost per message late in a session. I would try it
  once item 2 is in.
