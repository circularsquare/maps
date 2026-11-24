# Where some rules came from

The rules themselves live in `trainworld/CLAUDE.md`, `docs/orchestrator.md` and the maps skills. This file keeps the
dates and the incidents behind them, moved out of those files on 2026-11-20 and 2026-11-23 (T-4115) so the rules
read shorter. Grep it;
don't read it whole. Add a line when a rule is added for a reason worth remembering.

## trainworld/CLAUDE.md

- Estimates, no deadlines unless she gives one: Anita, 2026-11-17.
- A player request alone is no reason to change what can be built (350 km/h high speed): Anita, 2026-11-12.
- Speed and waiting wording ("loads", "shows up", "calculating trips"): Anita, 2026-11-17.
- Changelog items are not asked about at landing; she reviews them on the board: Anita, 2026-11-19. A publish still
  waits for her OK on new player text (Anita, 2026-11-20; v0.3.1 went out on 2026-11-12 with lines she had not seen).
- "we need to be less willing to put stuff on the changelog": Anita, 2026-11-18.
- Changelog tense (past for the change, present for how things are now): Anita, 2026-11-17.
- No changelog item for a change too fiddly to describe (undo after a load's cap lowering): Anita, 2026-11-17.
- No changelog item for a tooltip, alert wording or diagnostic text: Anita, 2026-11-18.
- Text copied from her own message is not reviewed: Anita, 2026-11-11. Text reused from the game: 2026-11-18.
- "Manually", never "by hand" ("by hand is a bit weird"): Anita, 2026-11-18.
- "turn around", never "turn round" ("its a pet peeve of mine"): Anita, 2026-11-12.
- Task fields very short, written to her: Anita, 2026-11-16. Rewrite a task to the part that's left: 2026-11-18.
- Nothing may open a window on her screen (`windowsHide`, `CREATE_NO_WINDOW`): background children were flashing
  terminals, 2026-11-18.
- Lean testing ("a bit too test heavy"): Anita, 2026-11-17.
- Local demand factors allowed again where published figures show a gap: Anita, 2026-11-11.
- Estimates checked against similar past tasks: T-3137 C1 was guessed at days and took about 45 minutes.
- A time saving as a percent of the whole wait, since seconds alone don't say much: Anita, 2026-11-23.
- "Segment", never "run" ("i dont like the terminology runs in general"): Anita, 2026-11-21.
- Advisor only for big design decisions ("more sparing ... only for like big design decisions"): Anita, 2026-11-23.
- Git worktrees only through agent-tree.mjs: 2026-11-23; a tree's junctions mean a forced remove or a deleted folder
  deletes the shared node_modules and data.
- Core claims wait at the cap instead of going over: T-4108, 2026-11-23.
- The agent cargo profile: about 70 s warm against 180 s for `cargo test --release` (T-3664).
- Speedups that shift riders are measured on a big real save, and she decides: Anita, 2026-11-23. Hand-built saves are
  too small to show a saving against the noise ("not feasible for me to build it"). Moved here from orchestrator.md
  on 2026-11-23, since the agent measuring needs it.
- No `cmd /c start ... /b` through Git Bash: /b becomes B:/ and the quotes are stripped, so Windows error dialogs
  popped up on her screen.
- Never edit docs with PowerShell Get-Content/Set-Content: moved from orchestrator.md on 2026-11-23, since every agent
  edits docs.

## docs/orchestrator.md

- Edit the handoff with the turn's other bookkeeping, not once per message: Anita, 2026-11-09.
- No Fable as a model unless she asks: Anita, 2026-11-09 and 2026-11-17. Advisor trial: 2026-11-16.
- Reports capped at 15 lines: Anita, 2026-11-17; reports had been running 30-60 lines.
- Stop background work before the final report: on 2026-11-17 an agent left with background work sent its report
  five times.
- Three versions on one sheet when she asks for a look to change without numbers: Anita, 2026-11-17, T-3586.
- Cycling at 500k: Anita, 2026-11-16, number set 2026-11-17. Overnight, keep going past it: 2026-11-05.
- Never go quiet on her: 2026-11-17, she waited 25 minutes for an answer during a landing.
- Duplicates may be folded without asking ("its fine to fold in duplicates if you think its a duplicate, since
  we're not ognna be losing that much info"): Anita, 2026-11-16.
- Triage of player reports: Anita, 2026-11-12.
- Ask cards short, with a title: Anita, 2026-11-17 (cards were too long).
- For you carries only new asks ("stop repeating things unless i ask"): Anita, 2026-11-17; made the rule for every
  session in her global file on 2026-11-20.
- Player request groups that are clearly done or duplicates are settled without her ("just don't auto address plyer
  requests where we have to make an actual new design decision"): Anita, 2026-11-16.
- Checker before landing risky changes: Anita, 2026-11-16, T-3451. Concise checkers: 2026-11-17.
- Commit board-related serve.py changes ("yes please commit board-related serve changes from now on"): Anita,
  2026-11-16. Push regularly ("feel free to be pushing this regulrly"): 2026-11-16; one wording for every
  committing session (commit freely, a couple of pushes a session) on 2026-11-20.
- snap8800 and serve.py restarts allowed without a prompt ("serve.py restarts are totally fine. they only affect
  me"): Anita, 2026-11-16.
- She edits the changelog before a publish ("ill amost always want to edit it unless we only have like 2 things in
  the update"): Anita, 2026-11-18. Highlights: 2026-11-17, T-3591. Sort before her final check: 2026-11-19.
- Playtest list before every publish ("maybe we were a bit too hasty to deploy last update ... before every update
  deploy, please give me like a full list of things to test"): Anita, 2026-11-19.
- New zones image: Anita, 2026-11-18, T-3614. Open-source sync under the publish yes: 2026-11-12. PR reply style:
  2026-11-13.
- shonei-server push covered by the deploy yes: Anita, 2026-11-18.
- Feedback inbox read-only, and reading a player's saved games only when a report can't be settled otherwise:
  Anita, 2026-11-12.
- After a publish, look for what to start next ("sometimes we start publishing and then just kinda dont take on
  much work at all and mostly sit idle, when in reality theres lots to do"): Anita, 2026-11-20.
- Handoff file under 60 lines: Anita, 2026-11-20 (the skill had said 40 and the check 80).
- Deleting a tree folder by hand: on 2026-11-12 it deleted through the `node_modules` junction and the main tree
  lost `.bin` and 15 packages ("Cannot find native binding"); `npm install` from trainworld/app put them back.
- Staging part of a JSON file: a line-level stage put two keys after the closing brace and snapbuild failed
  (commit d4164171).
- One line to her when winding down: on 2026-11-17 she was getting the same news from two tabs.
- Keep the work moving ("there are times when we have lots of stuff to potentially work on but we're only doing a
  bit at a time, or stopping"): Anita, 2026-11-03.
- Advisor calls logged in notes/handoff-log.md; more sparing from 2026-11-23 ("only for like big design decisions").
- A TODO.md short-term item goes into the brief verbatim: a paraphrase lost her formula tooltip (T-1839).
- Agent reports inbox (`notes/agent-reports/`, SubagentStop hook): T-3668.
- Standby session for the next orchestrator: her idea, 2026-11-21. next_orchestrator.py in Code Mode is her usual
  procedure since 2026-11-23; she wants the orchestrator in Windows Terminal, not the VS Code panel.
- Never go quiet, second time: on 2026-11-23 she waited 25 minutes again while the orchestrator dug into a full disk
  itself; since then the orchestrator doesn't investigate. A reply-nudge hook followed (T-4090).
- Agents 2-8: 2026-11-23, down from 4-12 on 2026-11-21. Briefly "usually 3-5, 8 a ceiling" ("why do we have 8 agents
  running? we dont need to do that right?"), then run by load the same day. 11-24, 12 running: "why didn't you just ask for
  clarification if we were going to exit the 2-8 range?" Now asked first.
- Smoke only on master after a batch and before a publish: Anita, 2026-11-24, after agents' tree smokes took 35+ min
  each on a full machine.
- Cores 0-1 hers: Anita, 2026-11-21. claude.exe itself stays on all cores: pinned, her typing lagged 4 s.
- Overnight core cap at the machine's max ("we should have been running with max core cap cuz it was overnight"):
  Anita, 2026-11-21.
- One worktree per code agent: T-792. Bookkeeping by command: T-3451. finish in one command: T-3554.
- Disk alert in the watch (30 GB, then 10 GB): T-4088, 2026-11-23, after the disk filled.
- Batch checking of risky changes: Anita, 2026-11-20, T-3902. 8800 rebuilt without waiting for the checker:
  2026-11-21.
- Check the staged diff of shared docs before committing: an agent's half-written paragraph rode along once.
- Triage: Sendai's Senzan Line is the example of a report few players would meet; rebuilding the player's own case
  found the express boarding rule in T-4019, where averages had looked fine.
- Don't ask her when to publish ("it just depends on what we get done, which neither of us really knows"): Anita,
  2026-11-23.
- Nobody drafts a reply to a player unless she asks: Anita, 2026-11-23.
- Deleted games restore: T-3789. Resources check once a day since 2026-11-11 (T-3243).
- Task sweeps leave a dated notes file: 2026-11-23, when she asked when the last sweep was and no record said.
